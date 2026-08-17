"""
Sparse Autoencoder (SAE) module.

Follows the SAE-RNA architecture:
    Encoder:  f = ReLU(W_e · x + b)
    Decoder:  x̂ = W_d · f + c

where W_e ∈ R^{d×k}, W_d ∈ R^{k×d}, b ∈ R^k, c ∈ R^d.

Loss = ‖x − x̂‖₂² + λ · ‖f‖₁

Key design choices from the paper:
  - Encoder: Kaiming initialization
  - Decoder: Xavier initialization
  - Untied weights (W_e and W_d are independent)
  - No unit-norm constraint on decoder (by default)
  - No dead neuron resampling (by default)

Reference:
  SAE-RNA: A Sparse Autoencoder Model for Interpreting RNA
  Language Model Representations
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from huggingface_hub import PyTorchModelHubMixin
from torch import nn

from .config import SAEConfig


DTYPES = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


class SparseAutoEncoder(nn.Module, PyTorchModelHubMixin):
    """
    Sparse Autoencoder following the SAE-RNA architecture.

    Default mode (SAE-RNA style):
        encode:  f = ReLU(x @ W_enc + b_enc)
        decode:  x̂ = f @ W_dec + b_dec

    Legacy mode (Anthropic/Nanda style, enabled via config):
        encode:  f = ReLU((x − b_dec) @ W_enc + b_enc)
        decode:  x̂ = f @ W_dec + b_dec

    Loss = MSE(x̂, x) + λ · ‖f‖₁
    """

    def __init__(
        self,
        act_size: int,
        sae_cfg: SAEConfig,
        save_dir: str = "./checkpoints",
    ):
        super().__init__()

        self.act_size = act_size
        self.d_hidden = act_size * sae_cfg.expansion_factor
        self.l1_coeff = sae_cfg.l1_coeff
        self.save_dir = save_dir
        self.subtract_b_dec = sae_cfg.subtract_b_dec_in_encoder
        self.use_decoder_unit_norm = sae_cfg.decoder_unit_norm

        dtype = DTYPES[sae_cfg.enc_dtype]
        torch.manual_seed(sae_cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(sae_cfg.seed)

        # ── Weight initialization ────────────────────────────────────────
        # Encoder weights: Kaiming initialization (SAE-RNA default)
        if sae_cfg.enc_init == "kaiming":
            W_enc_init = nn.init.kaiming_uniform_(
                torch.empty(act_size, self.d_hidden, dtype=dtype)
            )
        elif sae_cfg.enc_init == "xavier":
            W_enc_init = nn.init.xavier_uniform_(
                torch.empty(act_size, self.d_hidden, dtype=dtype)
            )
        else:
            raise ValueError(f"Unsupported sae.enc_init: {sae_cfg.enc_init!r}")

        # Decoder weights: Xavier initialization (SAE-RNA default)
        if sae_cfg.dec_init == "xavier":
            W_dec_init = nn.init.xavier_uniform_(
                torch.empty(self.d_hidden, act_size, dtype=dtype)
            )
        elif sae_cfg.dec_init == "kaiming":
            W_dec_init = nn.init.kaiming_uniform_(
                torch.empty(self.d_hidden, act_size, dtype=dtype)
            )
        else:
            raise ValueError(f"Unsupported sae.dec_init: {sae_cfg.dec_init!r}")

        self.W_enc = nn.Parameter(W_enc_init)
        self.W_dec = nn.Parameter(W_dec_init)
        self.b_enc = nn.Parameter(torch.zeros(self.d_hidden, dtype=dtype))
        self.b_dec = nn.Parameter(torch.zeros(act_size, dtype=dtype))

        # Optionally normalise decoder columns to unit norm at init
        if self.use_decoder_unit_norm:
            with torch.no_grad():
                self.W_dec.data = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)

    # ── Forward / encode / decode ────────────────────────────────────────

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns (total_loss, x_reconstruct, hidden_acts, l2_loss, l1_loss).
        """
        x = x.to(dtype=self.W_enc.dtype)
        if self.subtract_b_dec:
            # Legacy/Anthropic-style: center input by decoder bias
            x_input = x - self.b_dec
        else:
            # SAE-RNA style: direct encoding
            x_input = x

        acts = torch.relu(x_input @ self.W_enc + self.b_enc)
        x_reconstruct = acts @ self.W_dec + self.b_dec

        # Reconstruction loss (per-sample MSE, then mean over batch)
        l2_loss = (x_reconstruct.float() - x.float()).pow(2).sum(-1).mean(0)

        # Sparsity loss (L1 penalty on activations)
        l1_loss = self.l1_coeff * acts.float().abs().sum(-1).mean(0)

        total_loss = l2_loss + l1_loss
        return total_loss, x_reconstruct, acts, l2_loss, l1_loss

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(dtype=self.W_enc.dtype)
        if self.subtract_b_dec:
            x = x - self.b_dec
        return torch.relu(x @ self.W_enc + self.b_enc)

    def decode(self, acts: torch.Tensor) -> torch.Tensor:
        acts = acts.to(dtype=self.W_dec.dtype)
        return acts @ self.W_dec + self.b_dec

    # ── Decoder unit-norm constraint (optional) ──────────────────────────

    @torch.no_grad()
    def make_decoder_weights_and_grad_unit_norm(self) -> None:
        """
        Project the decoder weights (and their gradients) onto the unit sphere.
        Must be called *after* ``loss.backward()`` and *before* ``optimizer.step()``.

        Only used when ``decoder_unit_norm=True`` in config. SAE-RNA does NOT
        use this constraint.
        """
        if not self.use_decoder_unit_norm:
            return  # No-op for SAE-RNA style

        W_dec_normed = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        W_dec_grad_proj = (
            (self.W_dec.grad * W_dec_normed).sum(-1, keepdim=True) * W_dec_normed
        )
        self.W_dec.grad -= W_dec_grad_proj
        self.W_dec.data = W_dec_normed

    # ── Dead neuron handling (optional) ──────────────────────────────────

    @torch.no_grad()
    def compute_neuron_frequencies(
        self,
        dataloader: torch.utils.data.DataLoader,
        num_batches: int = 25,
        device: str = "cuda",
    ) -> torch.Tensor:
        """
        Compute the activation frequency of each hidden neuron over
        ``num_batches`` of data.
        """
        freq = torch.zeros(self.d_hidden, dtype=torch.float32, device=device)
        total = 0
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
            batch = batch.to(device)
            acts = self.encode(batch)
            freq += (acts > 0).float().sum(0)
            total += acts.shape[0]
        freq /= max(total, 1)
        dead_frac = (freq == 0).float().mean().item()
        print(f"  Dead neuron fraction: {dead_frac:.4f}")
        return freq

    @torch.no_grad()
    def reinit_dead_neurons(self, dead_mask: torch.Tensor) -> int:
        """Re-initialise neurons whose mask entry is True."""
        n_dead = dead_mask.sum().item()
        if n_dead == 0:
            return 0
        new_W_enc = nn.init.kaiming_uniform_(torch.zeros_like(self.W_enc))
        new_W_dec = nn.init.xavier_uniform_(torch.zeros_like(self.W_dec))
        self.W_enc.data[:, dead_mask] = new_W_enc[:, dead_mask]
        self.W_dec.data[dead_mask, :] = new_W_dec[dead_mask, :]
        self.b_enc.data[dead_mask] = 0.0
        return int(n_dead)

    @torch.no_grad()
    def resample_dead_neurons_anthropic(
        self,
        dead_mask: torch.Tensor,
        dataloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        num_batches: int = 50,
        device: str = "cuda",
        encoder_scale: float = 0.2,
    ) -> int:
        """
        Resample dead neurons using high-error inputs, following the spirit of
        Anthropic's resampling strategy.

        Dead decoder directions are reset toward sampled residual directions from
        inputs with high reconstruction loss; encoder columns are aligned with the
        same directions at a reduced norm.
        """
        dead_indices = dead_mask.nonzero(as_tuple=False).flatten()
        if dead_indices.numel() == 0:
            return 0

        residual_pool: list[torch.Tensor] = []
        input_pool: list[torch.Tensor] = []
        loss_pool: list[torch.Tensor] = []

        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= num_batches:
                break
            batch = batch.to(device=device, dtype=torch.float32)
            _loss, x_recon, _acts, _l2, _l1 = self(batch)
            residual = (batch - x_recon).float()
            losses = residual.pow(2).sum(dim=-1)
            residual_pool.append(residual.detach().cpu())
            input_pool.append(batch.detach().cpu())
            loss_pool.append(losses.detach().cpu())

        if not loss_pool:
            return 0

        residuals = torch.cat(residual_pool, dim=0)
        inputs = torch.cat(input_pool, dim=0)
        losses = torch.cat(loss_pool, dim=0).clamp(min=1e-12)

        replacement_count = dead_indices.numel()
        sample_indices = torch.multinomial(
            losses / losses.sum(),
            replacement_count,
            replacement=replacement_count > losses.numel(),
        )

        directions = residuals[sample_indices]
        fallback = inputs[sample_indices]
        zero_norm = directions.norm(dim=-1) < 1e-12
        if zero_norm.any():
            directions[zero_norm] = fallback[zero_norm]

        directions = directions.to(device=self.W_dec.device, dtype=self.W_dec.dtype)
        directions = directions / directions.norm(dim=-1, keepdim=True).clamp(min=1e-12)

        alive_mask = ~dead_mask
        if alive_mask.any():
            alive_norm = self.W_enc[:, alive_mask].norm(dim=0).mean().to(self.W_enc.dtype)
        else:
            alive_norm = torch.tensor(1.0, device=self.W_enc.device, dtype=self.W_enc.dtype)
        encoder_norm = alive_norm * encoder_scale

        self.W_dec.data[dead_indices, :] = directions
        self.W_enc.data[:, dead_indices] = directions.T * encoder_norm
        self.b_enc.data[dead_indices] = 0.0

        self._reset_optimizer_state_for_resampled_neurons(optimizer, dead_indices)
        return int(replacement_count)

    @staticmethod
    def _zero_optimizer_slice(
        state: dict,
        key: str,
        dead_indices: torch.Tensor,
        axis: str,
    ) -> None:
        tensor = state.get(key)
        if tensor is None or not torch.is_tensor(tensor):
            return
        if axis == "col":
            tensor[:, dead_indices] = 0
        elif axis == "row":
            tensor[dead_indices, :] = 0
        elif axis == "vec":
            tensor[dead_indices] = 0

    def _reset_optimizer_state_for_resampled_neurons(
        self,
        optimizer: torch.optim.Optimizer,
        dead_indices: torch.Tensor,
    ) -> None:
        for param, axis in (
            (self.W_enc, "col"),
            (self.W_dec, "row"),
            (self.b_enc, "vec"),
        ):
            state = optimizer.state.get(param)
            if not state:
                continue
            for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                self._zero_optimizer_slice(state, key, dead_indices, axis)

    # ── Persistence ──────────────────────────────────────────────────────

    def save_checkpoint(self, name: str, save_dir: Optional[str] = None) -> Path:
        save_dir = Path(save_dir or self.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        pt_path = save_dir / f"{name}.pt"
        torch.save(self.state_dict(), pt_path)

        # Also save a lightweight JSON with key hyperparameters
        meta = {
            "act_size": self.act_size,
            "d_hidden": self.d_hidden,
            "l1_coeff": self.l1_coeff,
            "subtract_b_dec": self.subtract_b_dec,
            "decoder_unit_norm": self.use_decoder_unit_norm,
        }
        with open(save_dir / f"{name}_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        print(f"  Checkpoint saved → {pt_path}")
        return pt_path

    @classmethod
    def load_checkpoint(cls, path: str | Path, sae_cfg: SAEConfig) -> "SparseAutoEncoder":
        """Load weights from a ``.pt`` checkpoint."""
        pt_path = Path(path)
        meta_path = pt_path.with_name(f"{pt_path.stem}_meta.json")
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            act_size = meta["act_size"]
        else:
            raise FileNotFoundError(
                f"Could not find metadata file {meta_path}; cannot recreate SAE."
            )
        sae = cls(act_size=act_size, sae_cfg=sae_cfg)
        sae.load_state_dict(torch.load(pt_path, map_location="cpu", weights_only=True))
        return sae
