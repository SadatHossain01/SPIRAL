"""
SAE evaluation metrics.

Evaluation now operates on real BiRNA-BERT vocabulary logits and respects the
saved activation standardization statistics:
  1. raw hidden states are standardized before entering the SAE
  2. SAE reconstructions are unstandardized before being reinserted
  3. special tokens are excluded from both hidden-state and logits metrics
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.stats import entropy
from torch.utils.data import DataLoader

from .model import BiRNABERTEmbedder
from .sae import SparseAutoEncoder



def _standardize(
    activations: torch.Tensor,
    mean: torch.Tensor | None,
    std: torch.Tensor | None,
) -> torch.Tensor:
    if mean is None or std is None:
        return activations
    activations = activations.float()
    mean = mean.to(device=activations.device, dtype=torch.float32)
    std = std.to(device=activations.device, dtype=torch.float32).clamp(min=1e-6)
    return (activations - mean) / std



def _unstandardize(
    activations: torch.Tensor,
    mean: torch.Tensor | None,
    std: torch.Tensor | None,
) -> torch.Tensor:
    if mean is None or std is None:
        return activations
    activations = activations.float()
    mean = mean.to(device=activations.device, dtype=torch.float32)
    std = std.to(device=activations.device, dtype=torch.float32).clamp(min=1e-6)
    return activations * std + mean



def _explained_variance(original: torch.Tensor, reconstruction: torch.Tensor) -> float:
    original = original.float()
    reconstruction = reconstruction.float()
    residual = original - reconstruction
    original_var = original.var(dim=0, unbiased=False).mean().item()
    residual_var = residual.var(dim=0, unbiased=False).mean().item()
    if original_var <= 1e-12:
        return 0.0
    return 1.0 - (residual_var / original_var)



def _mean_cosine_similarity(original: torch.Tensor, reconstruction: torch.Tensor) -> float:
    original = original.float()
    reconstruction = reconstruction.float()
    cosine = torch.nn.functional.cosine_similarity(original, reconstruction, dim=-1)
    return cosine.mean().item()


def _embedder_eval_autocast(embedder: BiRNABERTEmbedder, device: str):
    quant = getattr(embedder, "runtime_quantization", embedder.cfg.quantization).lower()
    use_amp = device.startswith("cuda") and torch.cuda.is_available() and quant in {"fp16", "bf16"}
    amp_dtype = torch.bfloat16 if quant == "bf16" else torch.float16
    return torch.amp.autocast(
        device_type=device.split(":")[0],
        dtype=amp_dtype,
        enabled=use_amp,
    )



def _create_sae_hook(
    sae: SparseAutoEncoder,
    non_special_mask: torch.Tensor,
    act_mean: torch.Tensor | None,
    act_std: torch.Tensor | None,
):
    """
    Create a forward hook that replaces non-special-token hidden states with
    SAE reconstructions while leaving special tokens unchanged.
    """
    mean_dev = None if act_mean is None else act_mean.to(device=sae.W_dec.device, dtype=torch.float32)
    std_dev = None if act_std is None else act_std.to(device=sae.W_dec.device, dtype=torch.float32)

    def hook_fn(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        mask = non_special_mask.to(hidden.device)
        if mask.sum().item() == 0:
            return output

        reconstructed = hidden.clone()
        if hidden.dim() == 3:
            selected = hidden[:, mask, :].reshape(-1, hidden.shape[-1])
            sae_input = _standardize(selected.float(), mean_dev, std_dev)
            _loss, x_recon_std, _acts, _l2, _l1 = sae(sae_input)
            x_recon = _unstandardize(x_recon_std.float(), mean_dev, std_dev)
            reconstructed[:, mask, :] = x_recon.reshape(1, -1, hidden.shape[-1]).to(hidden.dtype)
        else:
            selected = hidden[mask]
            sae_input = _standardize(selected.float(), mean_dev, std_dev)
            _loss, x_recon_std, _acts, _l2, _l1 = sae(sae_input)
            x_recon = _unstandardize(x_recon_std.float(), mean_dev, std_dev)
            reconstructed[mask] = x_recon.to(hidden.dtype)

        if isinstance(output, tuple):
            return (reconstructed,) + output[1:]
        return reconstructed

    return hook_fn


@torch.no_grad()
def evaluate_sae(
    sae: SparseAutoEncoder,
    embedder: BiRNABERTEmbedder,
    sequences: list[str],
    device: str = "cuda",
    act_mean: torch.Tensor | None = None,
    act_std: torch.Tensor | None = None,
) -> dict[str, float]:
    """
    Evaluate SAE quality on a held-out sequence subset.

    Metrics:
      - l0 / l1 / l2: sparsity and reconstruction losses in standardized space
      - raw_mse: reconstruction MSE in original hidden-state space
      - explained_variance / cosine_similarity: hidden-state fidelity
      - seq_rec_accuracy / kl_divergence / cross_entropy_increase:
        downstream logits fidelity after SAE intervention
    """
    ce_loss_fn = torch.nn.CrossEntropyLoss()

    l0_total = 0.0
    l1_total = 0.0
    l2_total = 0.0
    raw_mse_total = 0.0
    explained_variance_total = 0.0
    cosine_total = 0.0
    seq_acc_total = 0.0
    kl_total = 0.0
    ce_delta_total = 0.0

    hook_layer_name = embedder._hook_layer_name
    target_module = None
    for name, module in embedder.model.named_modules():
        if name == hook_layer_name:
            target_module = module
            break
    if target_module is None:
        raise ValueError(f"Could not find module '{hook_layer_name}' for evaluation hook.")

    mean_dev = None if act_mean is None else act_mean.to(device=device, dtype=torch.float32)
    std_dev = None if act_std is None else act_std.to(device=device, dtype=torch.float32)

    evaluated_sequences = 0
    for sequence in sequences:
        inputs = embedder.tokenize(sequence)
        valid_mask_cpu = inputs["attention_mask"][0].bool() & ~inputs["special_tokens_mask"][0].bool()
        if valid_mask_cpu.sum().item() == 0:
            continue

        inputs_dev = {
            key: value.to(device)
            for key, value in inputs.items()
            if key != "special_tokens_mask"
        }
        with _embedder_eval_autocast(embedder, device):
            base_outputs = embedder.model(**inputs_dev)
        base_logits = base_outputs.logits.detach()
        if base_logits.dim() == 3:
            base_logits = base_logits[0]
        valid_mask = valid_mask_cpu.to(base_logits.device)
        base_logits_valid = base_logits[valid_mask]
        base_probs = torch.softmax(base_logits_valid.float(), dim=-1).cpu().numpy()

        embs = embedder._intermediate[hook_layer_name].detach()
        if embs.dim() == 3:
            embs = embs[0]
        embs = embs[valid_mask]

        sae_input = _standardize(embs.float(), mean_dev, std_dev)
        _loss, x_recon_std, acts, l2_loss, l1_loss = sae(sae_input)
        x_recon = _unstandardize(x_recon_std.float(), mean_dev, std_dev)

        l2_total += l2_loss.item()
        l1_total += l1_loss.item()
        l0_total += (acts > 0).float().sum(dim=-1).mean().item()
        raw_mse_total += (x_recon - embs.float()).pow(2).mean().item()
        explained_variance_total += _explained_variance(embs, x_recon)
        cosine_total += _mean_cosine_similarity(embs, x_recon)

        sae_hook = _create_sae_hook(sae, valid_mask_cpu, act_mean, act_std)
        handle = target_module.register_forward_hook(sae_hook)
        with _embedder_eval_autocast(embedder, device):
            sae_outputs = embedder.model(**inputs_dev)
        handle.remove()

        sae_logits = sae_outputs.logits.detach()
        if sae_logits.dim() == 3:
            sae_logits = sae_logits[0]
        sae_logits_valid = sae_logits[valid_mask]
        sae_probs = torch.softmax(sae_logits_valid.float(), dim=-1).cpu().numpy()

        seq_acc = (
            base_logits_valid.argmax(dim=-1) == sae_logits_valid.argmax(dim=-1)
        ).float().mean().item()
        seq_acc_total += seq_acc

        kl = entropy(base_probs, sae_probs, axis=1)
        kl_total += float(np.mean(kl[np.isfinite(kl)]))

        input_ids_valid = inputs["input_ids"][0][valid_mask_cpu].to(device)
        ce_base = ce_loss_fn(base_logits_valid.float(), input_ids_valid).item()
        ce_sae = ce_loss_fn(sae_logits_valid.float(), input_ids_valid).item()
        ce_delta_total += (ce_sae - ce_base)

        evaluated_sequences += 1

    n = max(evaluated_sequences, 1)
    return {
        "l0": l0_total / n,
        "l1": l1_total / n,
        "l2": l2_total / n,
        "raw_mse": raw_mse_total / n,
        "explained_variance": explained_variance_total / n,
        "cosine_similarity": cosine_total / n,
        "seq_rec_accuracy": seq_acc_total / n,
        "kl_divergence": kl_total / n,
        "cross_entropy_increase": ce_delta_total / n,
    }


@torch.no_grad()
def count_dead_neurons(
    sae: SparseAutoEncoder,
    dataloader: DataLoader,
    num_batches: int = 50,
    threshold: float = 1e-5,
    device: str = "cuda",
) -> int:
    freqs = sae.compute_neuron_frequencies(dataloader, num_batches=num_batches, device=device)
    return int((freqs < threshold).sum().item())
