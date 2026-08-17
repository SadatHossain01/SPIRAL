"""
BiRNA-BERT embedding extractor.

Handles model / tokenizer loading, forward-hook based embedding extraction,
and batched activation generation from RNA sequences.

Supports quantization via bitsandbytes (int8 / int4) and native
half-precision (fp16 / bf16) to reduce GPU memory during extraction.
"""

from __future__ import annotations

import gc

import torch
import transformers
from transformers import AutoModelForMaskedLM, AutoTokenizer, BitsAndBytesConfig
from typing import Optional

from .config import ModelConfig


def _gpu_mem_mb() -> str:
    """Return a short string with current GPU memory usage."""
    if not torch.cuda.is_available():
        return "(no CUDA)"
    alloc = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    return f"GPU mem: {alloc:.0f} MB allocated / {reserved:.0f} MB reserved"


def _load_birna_bert(model_name: str, load_kwargs: dict):
    """
    Load BiRNA-BERT, working around a compatibility issue with transformers ≥ 5.x.

    Transformers 5.x always initialises model parameters on the ``meta`` device
    during ``from_pretrained`` (for memory efficiency).  However, BiRNA-BERT's
    custom ``BertEncoder.__init__`` calls ``rebuild_alibi_tensor()`` which
    performs real tensor arithmetic — and that crashes on meta tensors.

    The fix: temporarily monkey-patch ``rebuild_alibi_tensor`` to be a no-op
    during model construction, then call it on CPU once the real weights have
    been loaded.
    """
    import importlib, sys

    # ── 1. Import the remote-code module that defines BertEncoder ────────
    #    (it's already cached after the first load)
    mod_path = "transformers_modules.buetnlpbio.birna_hyphen_bert"
    bert_mod = None
    for key, mod in sys.modules.items():
        if "bert_layers" in key and mod_path in key:
            bert_mod = mod
            break

    if bert_mod is None:
        # First load: let transformers discover and cache the module.
        # We'll catch the error and retry with the patch.
        try:
            return AutoModelForMaskedLM.from_pretrained(model_name, **load_kwargs)
        except RuntimeError as e:
            if "meta" not in str(e):
                raise
            # Now the module should be cached in sys.modules
            for key, mod in sys.modules.items():
                if "bert_layers" in key and mod_path in key:
                    bert_mod = mod
                    break
            if bert_mod is None:
                raise RuntimeError(
                    "Could not locate BiRNA-BERT bert_layers module after initial load attempt."
                ) from e

    # ── 2. Monkey-patch rebuild_alibi_tensor to be a no-op during init ───
    BertEncoder = bert_mod.BertEncoder
    _original_rebuild = BertEncoder.rebuild_alibi_tensor

    def _noop_rebuild(self, size, device=None):
        """Skip ALiBi tensor build on meta device — will be rebuilt later."""
        self._current_alibi_size = int(size)

    BertEncoder.rebuild_alibi_tensor = _noop_rebuild

    try:
        model = AutoModelForMaskedLM.from_pretrained(model_name, **load_kwargs)
    finally:
        # Always restore the original method
        BertEncoder.rebuild_alibi_tensor = _original_rebuild

    # ── 3. Rebuild ALiBi tensors now that weights are on CPU ─────────────
    for name, module in model.named_modules():
        if isinstance(module, BertEncoder):
            module.rebuild_alibi_tensor(
                size=module._current_alibi_size, device="cpu"
            )

    return model


class BiRNABERTEmbedder:
    """
    Wraps BiRNA-BERT to extract intermediate-layer embeddings via
    forward hooks — no modification to the model weights.
    """

    def __init__(self, cfg: ModelConfig, device: str = "cuda"):
        self.cfg = cfg
        self.device = device
        self._is_loaded = False
        self.runtime_quantization = cfg.quantization.lower()

        # --- Load tokenizer ---------------------------------------------------
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer)

        # --- Load model (with optional quantization) --------------------------
        bert_config = transformers.BertConfig.from_pretrained(cfg.name)
        self._hidden_dim = int(getattr(bert_config, "hidden_size", 768))
        # Extend ALiBi context if needed
        if cfg.max_seq_length > getattr(bert_config, "alibi_starting_size", 1024):
            bert_config.alibi_starting_size = cfg.max_seq_length

        load_kwargs: dict = {
            "config": bert_config,
            "trust_remote_code": True,
        }

        quant = self.runtime_quantization
        if not self.device.startswith("cuda") and quant in ("fp16", "bf16"):
            print(
                f"  Requested {quant.upper()} model loading on CPU; "
                "falling back to full precision"
            )
            quant = "none"
            self.runtime_quantization = quant
        if quant in ("fp16", "bf16"):
            # Native half-precision — simple and effective
            target_dtype = torch.float16 if quant == "fp16" else torch.bfloat16
            load_kwargs["torch_dtype"] = target_dtype
            print(f"  Loading BiRNA-BERT in {quant.upper()} precision")
        elif quant == "int8":
            # 8-bit quantization via bitsandbytes
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
            load_kwargs["quantization_config"] = bnb_config
            print("  Loading BiRNA-BERT in INT8 (bitsandbytes)")
        elif quant == "int4":
            # 4-bit quantization via bitsandbytes (NF4)
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )
            load_kwargs["quantization_config"] = bnb_config
            print("  Loading BiRNA-BERT in INT4 / NF4 (bitsandbytes)")
        elif quant != "none":
            raise ValueError(
                f"Unknown quantization '{quant}'. "
                f"Choose from: none, fp16, bf16, int8, int4"
            )

        self.model = _load_birna_bert(cfg.name, load_kwargs)

        # For non-quantized models, move to device explicitly
        if quant in ("none", "fp16", "bf16"):
            self.model.to(self.device)
        # For int8/int4, bitsandbytes handles device placement automatically

        self.model.eval()
        self._is_loaded = True
        print(f"  BiRNA-BERT loaded.  {_gpu_mem_mb()}")

        # --- Prepare hook -----------------------------------------------------
        self._intermediate: dict[str, torch.Tensor] = {}
        self._hook_handle: Optional[torch.utils.hooks.RemovableHook] = None
        self._hook_layer_name = f"bert.encoder.layer.{cfg.layer_index}"
        self._register_hook()

    # ── Hook management ──────────────────────────────────────────────────

    def _register_hook(self) -> None:
        """Attach a forward hook to the requested transformer layer."""
        target_module = None
        for name, module in self.model.named_modules():
            if name == self._hook_layer_name:
                target_module = module
                break

        if target_module is None:
            raise ValueError(
                f"Layer '{self._hook_layer_name}' not found. "
                f"Available: {[n for n, _ in self.model.named_modules() if 'layer.' in n]}"
            )

        def _hook_fn(module, input, output):
            # BiRNA-BERT layers return a tuple; first element is hidden states
            if isinstance(output, tuple):
                self._intermediate[self._hook_layer_name] = output[0]
            else:
                self._intermediate[self._hook_layer_name] = output

        self._hook_handle = target_module.register_forward_hook(_hook_fn)

    def remove_hook(self) -> None:
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    def unload(self) -> None:
        """
        Remove the model from GPU and free memory.
        Call this after activation extraction to reclaim VRAM for SAE training.
        """
        if not self._is_loaded:
            return
        self.remove_hook()
        del self.model
        self.model = None  # type: ignore[assignment]
        self._is_loaded = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"  BiRNA-BERT unloaded from GPU.  {_gpu_mem_mb()}")

    # ── Tokenisation helpers ─────────────────────────────────────────────

    def tokenize(self, sequence: str) -> dict:
        """
        Tokenize a raw RNA string.

        If ``use_bpe`` is False, the sequence is space-separated (nucleotide-level).
        """
        if not self.cfg.use_bpe:
            sequence = " ".join(sequence)
        return self.tokenizer(
            sequence,
            return_tensors="pt",
            return_special_tokens_mask=True,
            truncation=True,
            max_length=self.cfg.max_seq_length,
            padding=False,
        )

    def tokenize_batch(self, sequences: list[str]) -> dict:
        """Tokenize a list of RNA sequences with padding."""
        if not self.cfg.use_bpe:
            sequences = [" ".join(s) for s in sequences]
        return self.tokenizer(
            sequences,
            return_tensors="pt",
            return_special_tokens_mask=True,
            truncation=True,
            max_length=self.cfg.max_seq_length,
            padding=True,
        )

    def count_tokens(self, sequence: str) -> int:
        """Return the number of non-special tokens for a single sequence."""
        return self.count_tokens_batch([sequence])[0]

    def count_tokens_batch(self, sequences: list[str]) -> list[int]:
        """Return exact non-special token counts for a batch of sequences."""
        inputs = self.tokenize_batch(sequences)
        keep_mask = inputs["attention_mask"].bool() & ~inputs["special_tokens_mask"].bool()
        return keep_mask.sum(dim=1).tolist()

    def _prepare_model_inputs(self, inputs: dict) -> dict:
        """Move tokenizer outputs to device, excluding auxiliary masks the model does not accept."""
        return {
            key: value.to(self.device)
            for key, value in inputs.items()
            if key != "special_tokens_mask"
        }

    # ── Embedding extraction ─────────────────────────────────────────────

    @torch.no_grad()
    def extract_embeddings(self, sequence: str) -> torch.Tensor:
        """
        Run a single sequence through BiRNA-BERT and return the
        intermediate activations from the configured layer.

        Returns
        -------
        torch.Tensor  — shape ``(seq_len, hidden_dim)``
        """
        inputs = self.tokenize(sequence)
        model_inputs = self._prepare_model_inputs(inputs)
        _ = self.model(**model_inputs)
        embs = self._intermediate[self._hook_layer_name].detach()
        special_mask = inputs["special_tokens_mask"].bool()[0].to(embs.device)
        # Handle both 3-D (B, T, D) and 2-D (T, D) output formats
        # (transformers ≥ 5.x may use unpadded / flash-attention, yielding 2-D)
        if embs.dim() == 3:
            return embs[0][~special_mask]  # remove batch dim and special tokens
        return embs[~special_mask]  # already (T, D)

    @torch.no_grad()
    def extract_embeddings_batch(self, sequences: list[str]) -> list[torch.Tensor]:
        """
        Extract embeddings for a batch.  Returns a *list* of tensors (one per
        sequence, without padding tokens) so downstream code doesn't need to
        worry about padding masks.
        """
        inputs = self.tokenize_batch(sequences)
        attention_mask = inputs["attention_mask"]
        special_tokens_mask = inputs["special_tokens_mask"]
        model_inputs = self._prepare_model_inputs(inputs)
        _ = self.model(**model_inputs)

        hidden = self._intermediate[self._hook_layer_name].detach()
        results: list[torch.Tensor] = []

        if hidden.dim() == 3:
            # Standard (B, T, D) format
            for i in range(hidden.size(0)):
                mask = attention_mask[i].to(hidden.device).bool()
                mask &= ~special_tokens_mask[i].to(hidden.device).bool()
                results.append(hidden[i][mask])
        else:
            # Flattened (total_tokens, D) format (unpadded / flash-attention)
            total_lengths = attention_mask.sum(dim=1).tolist()
            offset = 0
            for i, total_length in enumerate(total_lengths):
                total_length = int(total_length)
                seq_hidden = hidden[offset : offset + total_length]
                keep_mask = attention_mask[i, :total_length].to(hidden.device).bool()
                keep_mask &= ~special_tokens_mask[i, :total_length].to(hidden.device).bool()
                results.append(seq_hidden[keep_mask])
                offset += total_length

        return results

    @property
    def hidden_dim(self) -> int:
        """Dimensionality of the embeddings (= act_size for the SAE)."""
        return self._hidden_dim

    # ── Full-model inference (for evaluation) ────────────────────────────

    @torch.no_grad()
    def full_forward(self, sequence: str) -> torch.Tensor:
        """
        Full forward pass returning the model's output logits.
        Used during SAE evaluation to compare original vs. SAE-modified outputs.
        """
        inputs = self.tokenize(sequence)
        model_inputs = self._prepare_model_inputs(inputs)
        outputs = self.model(**model_inputs)
        logits = outputs.logits.detach()
        if logits.dim() == 3:
            return logits[0]  # remove batch dim
        return logits  # already (T, D) in unpadded mode
