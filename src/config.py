"""
Structured configuration for the BiRNA-BERT SAE training pipeline.

Uses Python dataclasses for type-safe, IDE-friendly configuration that can
be loaded from / merged with YAML files and CLI overrides.

Follows the SAE-RNA training methodology (adapted for BiRNA-BERT):
  Reference: SAE-RNA: A Sparse Autoencoder Model for Interpreting RNA
             Language Model Representations
"""

from __future__ import annotations

import os
import yaml
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


# ── Sub-configs ──────────────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    """Settings for the BiRNA-BERT backbone."""
    name: str = "buetnlpbio/birna-bert"
    tokenizer: str = "buetnlpbio/birna-tokenizer"
    layer_index: int = 3
    layer_indices: Optional[list[int]] = None
    max_seq_length: int = 512
    use_bpe: bool = True
    quantization: str = "none"       # "none" | "fp16" | "bf16" | "int8" | "int4"
    unload_after_extraction: bool = True  # Free BiRNA-BERT from GPU after extracting activations


@dataclass
class SAEConfig:
    """Settings for the Sparse Autoencoder architecture."""
    expansion_factor: int = 8
    l1_coeff: float = 3e-3           # SAE-RNA default: 3e-3
    enc_dtype: str = "fp32"          # "fp32" | "fp16" | "bf16"
    seed: int = 42
    # Initialization
    enc_init: str = "kaiming"        # "kaiming" (SAE-RNA default)
    dec_init: str = "xavier"         # "xavier" (SAE-RNA default)
    # Architecture variant
    subtract_b_dec_in_encoder: bool = False   # False = SAE-RNA style (direct encoding)
    decoder_unit_norm: bool = False           # False = SAE-RNA style (no unit-norm constraint)


@dataclass
class DataConfig:
    """Data source and preprocessing settings."""
    fasta_path: Optional[str] = None
    sequence_list: Optional[str] = None
    hf_dataset: Optional[str] = None
    hf_dataset_split: str = "train"
    hf_sequence_column: str = "sequence"
    num_sequences: Optional[int] = None
    test_split_ratio: float = 0.001
    # Activation standardization (SAE-RNA: standardize per-dimension)
    standardize_activations: bool = True
    stats_num_sequences: int = 1_000_000
    planning_batch_size: int = 2048
    planning_prefetch_batches: int = 4
    sequence_plan_dir: str = ".cache/sequence_plan"
    activation_strategy: str = "auto"      # "auto" | "full" | "rolling"
    activation_cache_budget_gb: float = 350.0
    dataloader_num_workers: Optional[int] = None  # null = auto-select from available CPUs
    dataloader_reserved_cores: int = min(max((os.cpu_count() or 1) // 8, 1), 2)
    dataloader_prefetch_factor: int = 2
    # Disk-backed activation sharding (enables unbounded dataset sizes)
    shard_dir: str = ".cache/act_shards"     # Base directory for activation shard files
    shard_size: int = 500_000                 # Max tokens per shard file
    extraction_chunk_size: int = 1000         # Sequences processed per extraction chunk (controls peak RAM)
    max_disk_shards: int = 16                 # Rolling window: max shards on disk at once (controls disk usage)


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    batch_size: int = 1024                   # SAE-RNA default: 1024 token activations
    learning_rate: float = 1e-3              # SAE-RNA default: 1e-3
    weight_decay: float = 1e-4               # SAE-RNA default: 1e-4 (AdamW)
    beta1: float = 0.9
    beta2: float = 0.999
    num_epochs: float = 10                   # Supports fractional values
    device: str = "cuda"
    extraction_batch_size: int = 16
    log_every_n_steps: int = 50
    # LR schedule (SAE-RNA uses cosine annealing)
    lr_scheduler: str = "cosine"             # "cosine" | "none"
    lr_warmup_steps: int = 0                 # Linear warmup steps before scheduler kicks in
    # Gradient clipping (SAE-RNA: clip norm at 1.0)
    grad_clip_norm: Optional[float] = 1.0
    # Dead neuron handling
    dead_neuron_check_steps: Optional[int] = 50_000
    dead_neuron_threshold: float = 1e-5
    dead_neuron_sample_batches: int = 50
    checkpoint_every_n_steps: Optional[int] = 125_000
    checkpoint_keep_last: Optional[int] = 3  # Keep the most recent N step_* checkpoints (null = keep all)
    mixed_precision: bool = True             # Use AMP (fp16/bf16) for SAE training
    resume_from: Optional[str] = None        # Path to a run directory to resume training from


@dataclass
class EvaluationConfig:
    """Evaluation settings."""
    enabled: bool = True
    mid_training: bool = True
    num_test_sequences: int = 256
    eval_every_n_steps: Optional[int] = None
    best_metric: str = "kl_divergence"
    best_mode: str = "min"


@dataclass
class LoggingConfig:
    """Logging and checkpointing settings."""
    wandb_enabled: bool = False               # Opt in via config or --set logging.wandb_enabled=true
    wandb_project: str = "sae-rna"
    wandb_entity: Optional[str] = None        # Set to your own W&B entity when enabling
    wandb_api_key: Optional[str] = None       # Set via config or WANDB_API_KEY env var
    save_dir: str = "./checkpoints"
    # HuggingFace Hub upload settings
    hf_upload_enabled: bool = False           # Upload model to HF Hub periodically
    hf_repo_id: Optional[str] = None          # e.g. "username/birna-sae-model"
    hf_upload_every_n_epochs: Optional[int] = None  # Upload every N epochs (null = end only)
    hf_token: Optional[str] = None            # HF token (or set HF_TOKEN env var)


# ── Top-level config ────────────────────────────────────────────────────────

@dataclass
class PipelineConfig:
    """Complete pipeline configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    sae: SAEConfig = field(default_factory=SAEConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # ── I/O helpers ──────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Serialise to a plain dict (for W&B / JSON / YAML)."""
        return asdict(self)

    def save_yaml(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        """Load from a YAML file and return a typed config object."""
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls._from_dict(raw)

    @classmethod
    def _from_dict(cls, d: dict) -> "PipelineConfig":
        return cls(
            model=ModelConfig(**d.get("model", {})),
            sae=SAEConfig(**d.get("sae", {})),
            data=DataConfig(**d.get("data", {})),
            training=TrainingConfig(**d.get("training", {})),
            evaluation=EvaluationConfig(**d.get("evaluation", {})),
            logging=LoggingConfig(**d.get("logging", {})),
        )

    def merge_overrides(self, overrides: dict) -> None:
        """
        Apply flat dot-notation overrides, e.g.
        ``{"training.learning_rate": 1e-4, "sae.l1_coeff": 5e-3}``.
        """
        for key, value in overrides.items():
            parts = key.split(".")
            if len(parts) != 2:
                raise ValueError(f"Override key must be 'section.field', got: {key}")
            section, field_name = parts
            sub = getattr(self, section, None)
            if sub is None:
                raise ValueError(f"Unknown config section: {section}")
            if not hasattr(sub, field_name):
                raise ValueError(f"Unknown field '{field_name}' in section '{section}'")
            # Cast to the target type
            current_value = getattr(sub, field_name)
            target_type = type(current_value)
            if value is None or target_type is type(None):
                # Optional field or explicitly setting to None — keep as-is
                setattr(sub, field_name, value)
            elif target_type in (list, dict):
                # Container types — assign directly (don't try to cast)
                setattr(sub, field_name, value)
            else:
                setattr(sub, field_name, target_type(value))

    def ensure_save_dir(self) -> Path:
        """Create the checkpoint directory if it doesn't exist."""
        p = Path(self.logging.save_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p
