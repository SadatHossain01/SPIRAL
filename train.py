#!/usr/bin/env python3
"""
CLI entry point for training a Sparse Autoencoder on BiRNA-BERT embeddings.

Follows the SAE-RNA training methodology adapted for BiRNA-BERT.

Usage
-----
  # Quick debug run (tiny dataset, few steps — validates the whole pipeline)
  python train.py --debug

  # Train with defaults (edit config.yaml first!)
  python train.py --config config.yaml

  # Override specific parameters
  python train.py --config config.yaml \\
      --set training.learning_rate=1e-4 \\
      --set sae.l1_coeff=5e-3 \\
      --set data.fasta_path=./my_rna.fasta

  # Resume from a previous run (auto-finds latest checkpoint subdir)
  python train.py --resume ./checkpoints/20260212_143000_run_name

  # Resume from a specific checkpoint subdirectory
  python train.py --resume ./checkpoints/20260212_143000_run_name/step_50000

  # Resume and extend training to more epochs
  python train.py --resume ./checkpoints/20260212_143000_run_name \\
      --set training.num_epochs=20

  # Debug on CPU
  python train.py --debug --set training.device=cpu
"""

from __future__ import annotations
from src.trainer import SAETrainer
from src.config import PipelineConfig
from pathlib import Path
import sys
import csv
import copy
import argparse

import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# ── Debug preset ─────────────────────────────────────────────────────────────
# These overrides create a minimal run that exercises every part of the
# pipeline (data → embedding extraction → standardization → SAE training
# → eval → save) in about 1-2 minutes, making it easy to catch bugs before
# a full run.

DEBUG_OVERRIDES = {
    # Data — will be pointed at a generated sample file (see _create_debug_data)
    "model.layer_indices": [],
    "data.test_split_ratio": 0.2,
    "data.shard_dir": ".cache/debug_shards",
    "data.shard_size": 5000,
    "data.extraction_chunk_size": 25,
    # Short training
    "training.num_epochs": 2,
    "training.log_every_n_steps": 5,
    "training.extraction_batch_size": 8,
    "training.lr_scheduler": "cosine",
    "training.grad_clip_norm": 1.0,
    "training.checkpoint_every_n_steps": 25,
    # Enable evaluation to test that path
    "evaluation.enabled": True,
    "evaluation.num_test_sequences": 5,
    "evaluation.eval_every_n_steps": 10,
    # Keep debug self-contained and offline by default
    "logging.wandb_enabled": False,
    "logging.save_dir": "./checkpoints/debug",
}


def _create_debug_data(path: str = ".cache/debug_sequences.fasta", n: int = 50) -> str:
    """Generate a small FASTA file with random RNA sequences for debug runs."""
    import random as _rng
    os.makedirs(os.path.dirname(path), exist_ok=True)
    _rng.seed(42)
    bases = "ACGU"
    with open(path, "w") as f:
        for i in range(n):
            length = _rng.randint(30, 100)
            seq = "".join(_rng.choice(bases) for _ in range(length))
            f.write(f">debug_seq_{i:04d}\n{seq}\n")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a Sparse Autoencoder on BiRNA-BERT embeddings.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config.yaml",
        help="Path to the YAML configuration file (default: config.yaml).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help=(
            "Run a tiny debug training session to validate the full pipeline. "
            "Uses 50 sequences, 2 epochs, small SAE, with evaluation enabled. "
            "Overrides most config values — useful for catching bugs fast."
        ),
    )
    parser.add_argument(
        "--resume", "-r",
        type=str,
        default=None,
        metavar="RUN_DIR",
        help=(
            "Resume training from a previous run directory or checkpoint subdirectory. "
            "If pointed at a run directory, will automatically find the latest checkpoint. "
            "If pointed at a subdirectory (e.g. step_10000/), will resume from that exact point. "
            "Example: --resume ./checkpoints/20260212_143000_run_name "
            "or --resume ./checkpoints/20260212_143000_run_name/step_50000"
        ),
    )
    parser.add_argument(
        "--set", "-s",
        action="append",
        default=[],
        dest="overrides",
        metavar="KEY=VALUE",
        help=(
            "Override a config value using dot notation. "
            "Can be specified multiple times.  Example: --set training.learning_rate=1e-4"
        ),
    )
    return parser.parse_args()


def _resolve_layer_indices(cfg: PipelineConfig) -> list[int]:
    raw_layers = cfg.model.layer_indices or [cfg.model.layer_index]
    if not raw_layers:
        raise ValueError(
            "No layer indices configured. Set model.layer_index or model.layer_indices.")

    layer_indices: list[int] = []
    seen: set[int] = set()
    for value in raw_layers:
        layer_index = int(value)
        if layer_index < 0:
            raise ValueError(
                f"Layer indices must be non-negative, got {layer_index}")
        if layer_index in seen:
            continue
        seen.add(layer_index)
        layer_indices.append(layer_index)
    return layer_indices


def _write_summary_csv(base_save_dir: str, rows: list[dict]) -> Path:
    summary_path = Path(base_save_dir) / "training_runs_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with open(summary_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return summary_path


def main() -> None:
    args = parse_args()

    # ── Load config ──────────────────────────────────────────────────────
    print(f"\nLoading config from: {args.config}")
    cfg = PipelineConfig.from_yaml(args.config)

    # ── Debug mode ───────────────────────────────────────────────────────
    if args.debug:
        print("\n" + "=" * 70)
        print("  🐛  DEBUG MODE — minimal run to validate the full pipeline")
        print("=" * 70)

        # Generate a small synthetic FASTA file
        debug_fasta = _create_debug_data()
        DEBUG_OVERRIDES["data.fasta_path"] = debug_fasta
        print(f"  Generated 50 random RNA sequences → {debug_fasta}")

        cfg.merge_overrides(DEBUG_OVERRIDES)
        print("  Applied debug overrides (50 sequences, 2 epochs, tiny SAE)")

    # ── Load .env file if present (for W&B API key, etc.) ────────────────
    env_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env_file):
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ.setdefault(key.strip(), value.strip())

    # ── Apply CLI overrides (these take highest priority) ────────────────
    if args.overrides:
        overrides = {}
        for o in args.overrides:
            if "=" not in o:
                print(
                    f"ERROR: Override must be KEY=VALUE, got: {o}", file=sys.stderr)
                sys.exit(1)
            key, value = o.split("=", 1)

            # Try to interpret the value as a Python literal
            try:
                import ast
                value = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                pass  # keep as string

            overrides[key] = value

        cfg.merge_overrides(overrides)
        print(f"  Applied {len(overrides)} CLI override(s)")

    # ── Resume mode ──────────────────────────────────────────────────────
    if args.resume:
        cfg.training.resume_from = args.resume
        print(f"\n  ♻  Resume requested from: {args.resume}")

    layer_indices = _resolve_layer_indices(cfg)
    if args.resume and len(layer_indices) > 1:
        print(
            "ERROR: --resume is ambiguous when model.layer_indices contains multiple layers. "
            "Resume a single layer run at a time.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"\nConfigured SAE layers: {layer_indices}")

    # ── Run ──────────────────────────────────────────────────────────────
    base_save_dir = cfg.logging.save_dir
    run_summaries: list[dict] = []
    losses: list[dict] = []

    for position, layer_index in enumerate(layer_indices, start=1):
        layer_cfg = copy.deepcopy(cfg)
        layer_cfg.model.layer_index = layer_index
        layer_cfg.model.layer_indices = None

        print("\n" + "=" * 70)
        print(
            f"  Layer run {position}/{len(layer_indices)} — layer_index={layer_index}")
        print("=" * 70)

        trainer = SAETrainer(layer_cfg)
        trainer.setup()
        losses = trainer.train()

        summary_row = {
            "layer_index": layer_index,
            "run_dir": "" if trainer.run_dir is None else str(trainer.run_dir),
            "best_metric_name": layer_cfg.evaluation.best_metric,
            "best_metric_value": trainer.best_checkpoint_metric,
            "best_checkpoint_path": trainer.best_checkpoint_path or "",
            "planned_train_sequences": trainer.total_train_sequences,
            "planned_train_tokens": trainer.total_train_tokens,
            "effective_train_tokens_per_epoch": trainer.effective_train_tokens_per_epoch,
            "dropped_train_tokens_per_epoch": trainer.dropped_train_tokens_per_epoch,
            "target_train_tokens": trainer.total_training_token_budget,
            "activation_strategy": trainer.activation_strategy,
            "estimated_activation_cache_gb": trainer.estimated_activation_cache_gb,
        }
        for metric_name, metric_value in trainer.final_eval.items():
            summary_row[f"final_{metric_name}"] = metric_value
        run_summaries.append(summary_row)

    summary_path = _write_summary_csv(base_save_dir, run_summaries)
    print(f"\nRun summary written to: {summary_path}")

    if args.debug:
        print("\n" + "=" * 70)
        print("  ✅  Debug run complete — pipeline is working correctly!")
        n_steps = len(losses)
        if losses:
            last = losses[-1]
            print(f"  Final step {n_steps}: loss={last.get('train/loss', 'N/A'):.4f}, "
                  f"l0={last.get('train/l0_sparsity', 'N/A'):.1f}")
        print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
