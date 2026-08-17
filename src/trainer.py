"""
SAE Trainer — orchestrates sequence planning, normalization, activation caching,
and SAE training.

Key design updates in this version:
  - sequence planning is exact-token and disk-backed, not RAM-backed
  - normalization stats are computed once per layer on a fixed train subset
  - activations are standardized on write, not with per-window drift
  - checkpoints can be compared via a CSV of evaluation metrics
  - automatic resume is robust to differing checkpoint digit lengths
"""

from __future__ import annotations

import csv
import gc
import json
import math
import pickle
import random
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import tqdm

from .config import PipelineConfig
from .dataset import (
    build_sharded_dataloader,
    cleanup_shards,
    compute_activation_statistics_from_sequences,
    extract_activations_to_shards,
    iter_sequences_from_chunk,
    iter_train_sequences_from_plan,
    load_eval_sequences,
    plan_or_load_sequence_chunks,
    resolve_activation_strategy,
)
from .evaluate import count_dead_neurons, evaluate_sae
from .model import BiRNABERTEmbedder
from .sae import SparseAutoEncoder


class SAETrainer:
    """End-to-end trainer for a Sparse Autoencoder on BiRNA-BERT activations."""

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.device = cfg.training.device
        self.layer_index = cfg.model.layer_index
        self.layer_tag = f"layer_{self.layer_index:02d}"

        self.embedder: Optional[BiRNABERTEmbedder] = None
        self.sae: Optional[SparseAutoEncoder] = None

        self.sequence_plan: Optional[dict] = None
        self.eval_sequences: list[str] = []
        self.total_train_sequences: int = 0
        self.total_train_tokens: int = 0
        self.effective_train_tokens_per_epoch: int = 0
        self.dropped_train_tokens_per_epoch: int = 0
        self.total_training_token_budget: int = 0
        self.num_full_epochs: int = 0
        self.total_planned_steps: int = 0
        self.stats_num_sequences: int = 0
        self.stats_total_tokens: int = 0
        self.act_size: int = 0

        self.activation_strategy: str = "rolling"
        self.estimated_activation_cache_gb: float = 0.0
        self.layer_cache_root: Optional[Path] = None
        self.full_cache_dir: Optional[Path] = None
        self.rolling_cache_dir: Optional[Path] = None
        self.stats_cache_path: Optional[Path] = None

        self.act_mean: Optional[torch.Tensor] = None
        self.act_std: Optional[torch.Tensor] = None

        self.run_dir: Optional[Path] = None
        self.run_name: Optional[str] = None
        self.final_eval: dict[str, float] = {}
        self.best_checkpoint_path: Optional[str] = None
        self.best_checkpoint_metric: Optional[float] = None
        self.metrics_csv_path: Optional[Path] = None

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _set_global_seeds(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @staticmethod
    def _checkpoint_dir_sort_key(path: Path) -> tuple[int, int, str]:
        if path.name == "final":
            return (2, 2**31 - 1, path.name)
        if path.name.startswith("step_"):
            suffix = path.name.split("_", 1)[1]
            try:
                return (1, int(suffix), path.name)
            except ValueError:
                return (1, -1, path.name)
        if path.name.startswith("epoch_"):
            suffix = path.name.split("_", 1)[1]
            try:
                return (0, int(suffix), path.name)
            except ValueError:
                return (0, -1, path.name)
        return (-1, -1, path.name)

    @classmethod
    def _resolve_resume_dir(cls, resume_path: str | Path) -> tuple[Path, Path]:
        p = Path(resume_path)
        state_file = p / "training_state.pt"

        if state_file.exists():
            if p.name.startswith("step_") or p.name.startswith("epoch_") or p.name == "final":
                return p.parent, p
            return p, p

        candidates: list[Path] = []
        for child in p.iterdir():
            if child.is_dir() and (child / "training_state.pt").exists():
                candidates.append(child)
        if candidates:
            latest = max(candidates, key=cls._checkpoint_dir_sort_key)
            return p, latest

        raise FileNotFoundError(
            f"Cannot resume: no training_state.pt found in {p} or its subdirectories. "
            "Make sure to point to a run directory or a checkpoint subdirectory."
        )

    @staticmethod
    def _normalize_resume_cursor(resume_state: Optional[dict]) -> dict[str, int | str]:
        if not resume_state:
            return {
                "next_epoch": 0,
                "next_chunk_idx": 0,
                "batches_completed_in_epoch": 0,
                "batches_completed_in_chunk": 0,
                "activation_strategy": "full",
            }

        cursor = resume_state.get("resume_cursor")
        if cursor is None:
            return {
                "next_epoch": int(resume_state.get("epoch", -1)) + 1,
                "next_chunk_idx": 0,
                "batches_completed_in_epoch": 0,
                "batches_completed_in_chunk": 0,
                "activation_strategy": "legacy",
            }

        return {
            "next_epoch": int(cursor.get("next_epoch", resume_state.get("epoch", -1) + 1)),
            "next_chunk_idx": int(cursor.get("next_chunk_idx", 0)),
            "batches_completed_in_epoch": int(cursor.get("batches_completed_in_epoch", 0)),
            "batches_completed_in_chunk": int(cursor.get("batches_completed_in_chunk", 0)),
            "activation_strategy": str(cursor.get("activation_strategy", "unknown")),
        }

    def _epoch_boundary_cursor(self, next_epoch: int) -> dict[str, int | str]:
        return {
            "next_epoch": int(next_epoch),
            "next_chunk_idx": 0,
            "batches_completed_in_epoch": 0,
            "batches_completed_in_chunk": 0,
            "activation_strategy": self.activation_strategy,
        }

    def _build_resume_cursor(
        self,
        epoch: int,
        current_batch_in_epoch: int,
        current_batch_in_chunk: int,
        total_batches_in_chunk: int,
        chunk_idx: Optional[int] = None,
        total_chunks: Optional[int] = None,
    ) -> dict[str, int | str]:
        cursor: dict[str, int | str] = {
            "next_epoch": int(epoch),
            "next_chunk_idx": int(chunk_idx or 0),
            "batches_completed_in_epoch": int(current_batch_in_epoch),
            "batches_completed_in_chunk": int(current_batch_in_chunk),
            "activation_strategy": self.activation_strategy,
        }

        loader_complete = current_batch_in_chunk >= total_batches_in_chunk
        if self.activation_strategy == "full":
            if loader_complete:
                return self._epoch_boundary_cursor(epoch + 1)
            return cursor

        if chunk_idx is None or total_chunks is None:
            raise ValueError("Rolling resume cursor requires chunk metadata")

        if loader_complete:
            if chunk_idx + 1 < total_chunks:
                cursor["next_chunk_idx"] = int(chunk_idx + 1)
                cursor["batches_completed_in_chunk"] = 0
            else:
                return self._epoch_boundary_cursor(epoch + 1)

        return cursor

    def _ensure_embedder_loaded(self) -> None:
        if self.embedder is not None and self.embedder._is_loaded:
            return
        self.embedder = BiRNABERTEmbedder(self.cfg.model, device=self.device)
        self.act_size = self.embedder.hidden_dim

    def _maybe_unload_embedder(self) -> None:
        if self.cfg.model.unload_after_extraction and self.embedder is not None and self.embedder._is_loaded:
            self.embedder.unload()

    def _cache_dirs(self) -> tuple[Path, Path, Path]:
        assert self.sequence_plan is not None
        base = Path(self.cfg.data.shard_dir) / self.sequence_plan["signature"] / self.layer_tag
        full_dir = base / "full_cache"
        rolling_dir = base / "rolling_tmp"
        return base, full_dir, rolling_dir

    def _stats_cache_file(self) -> Path:
        assert self.sequence_plan is not None
        return Path(self.cfg.data.sequence_plan_dir) / (
            f"{self.sequence_plan['signature']}_{self.layer_tag}_act_stats.pt"
        )

    def _compute_effective_train_tokens_per_epoch(self) -> tuple[int, int]:
        if self.sequence_plan is None:
            return 0, 0

        batch_size = self.cfg.training.batch_size
        if self.activation_strategy == "full":
            effective_tokens = (self.total_train_tokens // batch_size) * batch_size
        else:
            effective_tokens = sum(
                (int(chunk["total_tokens"]) // batch_size) * batch_size
                for chunk in self.sequence_plan["chunks"]
            )

        dropped_tokens = max(self.total_train_tokens - effective_tokens, 0)
        return effective_tokens, dropped_tokens

    def _build_frequency_loader(
        self,
        active_loader: Optional[torch.utils.data.DataLoader],
    ) -> Optional[torch.utils.data.DataLoader]:
        dataset = None if active_loader is None else getattr(active_loader, "dataset", None)
        shard_dir = None if dataset is None else getattr(dataset, "shard_dir", None)
        if shard_dir is None:
            return None

        shard_dir = Path(shard_dir)
        if not shard_dir.exists():
            return None

        freq_loader, _, _ = build_sharded_dataloader(
            shard_dir,
            batch_size=self.cfg.training.batch_size,
            shuffle=False,
            num_workers=self.cfg.data.dataloader_num_workers,
            reserved_cores=self.cfg.data.dataloader_reserved_cores,
            prefetch_factor=self.cfg.data.dataloader_prefetch_factor,
        )
        return freq_loader

    def _prune_old_step_checkpoints(self) -> None:
        keep_last = self.cfg.training.checkpoint_keep_last
        if keep_last is None or keep_last < 1 or self.run_dir is None:
            return

        checkpoint_dirs = sorted(
            (
                path
                for path in self.run_dir.iterdir()
                if path.is_dir()
                and path.name.startswith("step_")
                and (path / "training_state.pt").exists()
            ),
            key=self._checkpoint_dir_sort_key,
        )
        if len(checkpoint_dirs) <= keep_last:
            return

        protected = {path.resolve() for path in checkpoint_dirs[-keep_last:]}
        if self.best_checkpoint_path:
            protected.add(Path(self.best_checkpoint_path).resolve())

        for checkpoint_dir in checkpoint_dirs:
            if checkpoint_dir.resolve() in protected:
                continue
            shutil.rmtree(checkpoint_dir)

    def _write_checkpoint_metrics_row(self, row: dict) -> None:
        if self.metrics_csv_path is None:
            return
        file_exists = self.metrics_csv_path.exists()
        with open(self.metrics_csv_path, "a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    def _metric_should_improve(self, value: float) -> bool:
        mode = self.cfg.evaluation.best_mode.lower()
        if self.best_checkpoint_metric is None:
            return True
        if mode == "min":
            return value < self.best_checkpoint_metric
        if mode == "max":
            return value > self.best_checkpoint_metric
        raise ValueError("evaluation.best_mode must be 'min' or 'max'")

    def _maybe_update_best_checkpoint(self, metrics_row: dict, checkpoint_dir: Optional[Path]) -> None:
        if checkpoint_dir is None:
            return
        metric_name = self.cfg.evaluation.best_metric
        metric_value = metrics_row.get(metric_name)
        if metric_value is None:
            return
        metric_value = float(metric_value)
        if self._metric_should_improve(metric_value):
            self.best_checkpoint_metric = metric_value
            self.best_checkpoint_path = str(checkpoint_dir)
            checkpoint_relpath = None
            if self.run_dir is not None:
                try:
                    checkpoint_relpath = str(checkpoint_dir.relative_to(self.run_dir))
                except ValueError:
                    checkpoint_relpath = None
            best_payload = {
                "metric": metric_name,
                "mode": self.cfg.evaluation.best_mode,
                "value": metric_value,
                "checkpoint_dir": str(checkpoint_dir),
                "checkpoint_relpath": checkpoint_relpath,
                "layer_index": self.layer_index,
            }
            if self.run_dir is not None:
                (self.run_dir / "best_checkpoint.json").write_text(json.dumps(best_payload, indent=2))

    def _annotate_shard_manifest(self, shard_dir: Path, extra: dict) -> None:
        manifest_path = shard_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest.update(extra)
        manifest_path.write_text(json.dumps(manifest, indent=2))

    def _load_or_compute_stats(self) -> None:
        assert self.sequence_plan is not None
        assert self.embedder is not None
        stats_path = self._stats_cache_file()
        self.stats_cache_path = stats_path
        if not self.cfg.data.standardize_activations:
            self.act_mean = None
            self.act_std = None
            self.stats_total_tokens = 0
            self.stats_num_sequences = 0
            print("  Activation standardization disabled — raw activations will be used")
            return

        if stats_path.exists():
            payload = torch.load(stats_path, map_location="cpu", weights_only=True)
            self.act_mean = payload["mean"]
            self.act_std = payload["std"]
            self.stats_total_tokens = int(payload.get("total_tokens", self.sequence_plan["stats_total_tokens"]))
            self.stats_num_sequences = int(payload.get("num_sequences", self.sequence_plan["stats_num_sequences"]))
            print(f"  Reusing cached activation stats → {stats_path}")
            return

        print("\n[3/5] Computing fixed activation statistics …")
        sequence_iter = iter_train_sequences_from_plan(
            self.sequence_plan,
            limit=self.sequence_plan["stats_num_sequences"],
        )
        mean, std, total_tokens, num_sequences = compute_activation_statistics_from_sequences(
            self.embedder,
            sequence_iter,
            num_sequences=self.sequence_plan["stats_num_sequences"],
            extraction_batch_size=self.cfg.training.extraction_batch_size,
            extraction_chunk_size=self.cfg.data.extraction_chunk_size,
            device=self.device,
        )
        self.act_mean = mean
        self.act_std = std
        self.stats_total_tokens = total_tokens
        self.stats_num_sequences = num_sequences
        torch.save(
            {
                "mean": mean,
                "std": std,
                "total_tokens": total_tokens,
                "num_sequences": num_sequences,
                "layer_index": self.layer_index,
                "plan_signature": self.sequence_plan["signature"],
            },
            stats_path,
        )
        print(f"  Activation stats saved → {stats_path}")

    def _prepare_full_activation_cache(self) -> Path:
        assert self.embedder is not None
        assert self.full_cache_dir is not None
        assert self.sequence_plan is not None

        manifest_path = self.full_cache_dir / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if (
                manifest.get("plan_signature") == self.sequence_plan["signature"]
                and int(manifest.get("layer_index", -1)) == self.layer_index
                and bool(manifest.get("standardized", False)) == bool(self.cfg.data.standardize_activations)
                and int(manifest.get("total_tokens", -1)) == int(self.total_train_tokens)
            ):
                print(f"  Reusing full activation cache → {self.full_cache_dir}")
                return self.full_cache_dir

        print("\n  Building full activation cache once for all epochs …")
        manifest = extract_activations_to_shards(
            self.embedder,
            iter_train_sequences_from_plan(self.sequence_plan),
            shard_dir=self.full_cache_dir,
            extraction_batch_size=self.cfg.training.extraction_batch_size,
            extraction_chunk_size=self.cfg.data.extraction_chunk_size,
            shard_size=self.cfg.data.shard_size,
            device=self.device,
            total_sequences=self.total_train_sequences,
            progress_desc=f"{self.layer_tag} full-cache extract",
            normalize_mean=self.act_mean,
            normalize_std=self.act_std,
        )
        self._annotate_shard_manifest(
            self.full_cache_dir,
            {
                "plan_signature": self.sequence_plan["signature"],
                "layer_index": self.layer_index,
                "stats_path": str(self.stats_cache_path),
                "total_train_sequences": self.total_train_sequences,
            },
        )
        self._maybe_unload_embedder()
        return self.full_cache_dir

    def _prepare_rolling_window_cache(self, chunk: dict) -> Path:
        assert self.embedder is not None
        assert self.rolling_cache_dir is not None

        chunk_idx = int(chunk["chunk_idx"])
        print(
            f"\n  ▸ {self.layer_tag} rolling window {chunk_idx + 1}/{len(self.sequence_plan['chunks'])} "
            f"({int(chunk['num_sequences']):,} seqs / {int(chunk['total_tokens']):,} tokens)"
        )
        extract_activations_to_shards(
            self.embedder,
            iter_sequences_from_chunk(chunk["path"]),
            shard_dir=self.rolling_cache_dir,
            extraction_batch_size=self.cfg.training.extraction_batch_size,
            extraction_chunk_size=self.cfg.data.extraction_chunk_size,
            shard_size=self.cfg.data.shard_size,
            device=self.device,
            total_sequences=int(chunk["num_sequences"]),
            progress_desc=f"{self.layer_tag} rolling extract",
            normalize_mean=self.act_mean,
            normalize_std=self.act_std,
        )
        self._annotate_shard_manifest(
            self.rolling_cache_dir,
            {
                "plan_signature": self.sequence_plan["signature"],
                "layer_index": self.layer_index,
                "stats_path": str(self.stats_cache_path),
                "chunk_idx": chunk_idx,
                "chunk_tokens": int(chunk["total_tokens"]),
            },
        )
        return self.rolling_cache_dir

    def _evaluate_checkpoint(
        self,
        sae: SparseAutoEncoder,
        global_step: int,
        epoch: int,
        checkpoint_dir: Optional[Path],
        lr: float,
        train_snapshot: dict,
        dead_loader: Optional[torch.utils.data.DataLoader],
    ) -> dict[str, float]:
        if not self.cfg.evaluation.enabled or not self.eval_sequences:
            return {}

        self._ensure_embedder_loaded()
        eval_sequences = self.eval_sequences[: self.cfg.evaluation.num_test_sequences]
        eval_metrics = evaluate_sae(
            sae,
            self.embedder,
            eval_sequences,
            device=self.device,
            act_mean=self.act_mean,
            act_std=self.act_std,
        )

        dead_neurons = None
        if dead_loader is not None:
            dead_neurons = count_dead_neurons(
                sae,
                dead_loader,
                num_batches=self.cfg.training.dead_neuron_sample_batches,
                threshold=self.cfg.training.dead_neuron_threshold,
                device=self.device,
            )
            eval_metrics["dead_neurons"] = float(dead_neurons)

        metrics_row = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "layer_index": self.layer_index,
            "epoch": epoch + 1,
            "global_step": global_step,
            "checkpoint_dir": "" if checkpoint_dir is None else str(checkpoint_dir),
            "learning_rate": lr,
            "tokens_seen": train_snapshot.get("train/tokens_seen"),
            "tokens_seen_pct": train_snapshot.get("train/tokens_seen_pct"),
            "tokens_seen_epoch": train_snapshot.get("train/tokens_seen_epoch"),
            "tokens_seen_epoch_pct": train_snapshot.get("train/tokens_seen_epoch_pct"),
            "planned_train_tokens": self.total_train_tokens,
            "effective_train_tokens_per_epoch": self.effective_train_tokens_per_epoch,
            "dropped_train_tokens_per_epoch": self.dropped_train_tokens_per_epoch,
            "target_train_tokens": self.total_training_token_budget,
            "train_loss": train_snapshot.get("train/loss"),
            "train_l2_loss": train_snapshot.get("train/l2_loss"),
            "train_l1_loss": train_snapshot.get("train/l1_loss"),
            "train_l0_sparsity": train_snapshot.get("train/l0_sparsity"),
        }
        metrics_row.update(eval_metrics)
        self._write_checkpoint_metrics_row(metrics_row)
        self._maybe_update_best_checkpoint(metrics_row, checkpoint_dir)

        if checkpoint_dir is not None:
            with open(checkpoint_dir / "eval_metrics.json", "w") as handle:
                json.dump(eval_metrics, handle, indent=2)
        return {f"eval/{key}": value for key, value in eval_metrics.items()}

    # ── Setup ────────────────────────────────────────────────────────────

    def setup(self) -> None:
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            print("  ⚠  CUDA requested but not available — falling back to CPU")
            self.device = "cpu"
            self.cfg.training.device = "cpu"

        self._set_global_seeds(self.cfg.sae.seed)

        print("=" * 70)
        print(f"  BiRNA-BERT × Sparse Autoencoder — Setup ({self.layer_tag})")
        print("=" * 70)

        print("\n[1/5] Planning exact train/eval sequence chunks …")
        self.sequence_plan = plan_or_load_sequence_chunks(
            self.cfg.data,
            self.cfg.model,
            self.cfg.evaluation,
            seed=self.cfg.sae.seed,
        )
        self.eval_sequences = load_eval_sequences(self.sequence_plan)
        self.total_train_sequences = int(self.sequence_plan["total_train_sequences"])
        self.total_train_tokens = int(self.sequence_plan["total_train_tokens"])
        self.stats_num_sequences = int(self.sequence_plan["stats_num_sequences"])
        self.stats_total_tokens = int(self.sequence_plan["stats_total_tokens"])

        print("\n[2/5] Loading BiRNA-BERT …")
        self._ensure_embedder_loaded()
        print(f"  Activation dimension (hidden_dim): {self.act_size}")

        self.activation_strategy, self.estimated_activation_cache_gb = resolve_activation_strategy(
            self.sequence_plan,
            self.act_size,
            self.cfg.data,
        )
        self.effective_train_tokens_per_epoch, self.dropped_train_tokens_per_epoch = (
            self._compute_effective_train_tokens_per_epoch()
        )
        self.num_full_epochs = math.ceil(self.cfg.training.num_epochs)
        steps_per_epoch = self.effective_train_tokens_per_epoch // self.cfg.training.batch_size
        self.total_planned_steps = int(steps_per_epoch * self.cfg.training.num_epochs)
        self.total_training_token_budget = int(
            self.effective_train_tokens_per_epoch * self.cfg.training.num_epochs
        )
        if self.effective_train_tokens_per_epoch <= 0:
            raise RuntimeError(
                "No full training batches can be formed with the current token plan and training.batch_size. "
                "Reduce training.batch_size or increase the amount of training data."
            )
        self.layer_cache_root, self.full_cache_dir, self.rolling_cache_dir = self._cache_dirs()
        self.layer_cache_root.mkdir(parents=True, exist_ok=True)
        print(
            f"  Activation strategy: {self.activation_strategy} | "
            f"estimated full-cache size: {self.estimated_activation_cache_gb:.2f} GB"
        )

        self._load_or_compute_stats()

        print("\n[4/5] Building Sparse Autoencoder …")
        self.sae = SparseAutoEncoder(
            act_size=self.act_size,
            sae_cfg=self.cfg.sae,
            save_dir=self.cfg.logging.save_dir,
        ).to(self.device)
        dict_size = self.act_size * self.cfg.sae.expansion_factor
        print(f"  SAE: {self.act_size} → {dict_size} → {self.act_size}")
        print(f"  Train sequences: {self.total_train_sequences:,} | Train tokens: {self.total_train_tokens:,}")
        print(
            f"  Effective train tokens/epoch: {self.effective_train_tokens_per_epoch:,} | "
            f"Dropped tail tokens/epoch: {self.dropped_train_tokens_per_epoch:,}"
        )
        print(f"  Eval sequences kept: {len(self.eval_sequences):,}")
        if self.cfg.data.standardize_activations:
            print(f"  Stats subset: {self.stats_num_sequences:,} seqs / {self.stats_total_tokens:,} tokens")
        else:
            print("  Stats subset: disabled")

        if self.cfg.training.resume_from:
            _, ckpt_dir = self._resolve_resume_dir(self.cfg.training.resume_from)
            state_path = ckpt_dir / "training_state.pt"
            print("\n[5/5] Restoring SAE weights from resume checkpoint …")
            state = torch.load(state_path, map_location=self.device, weights_only=True)
            self.sae.load_state_dict(state["model_state_dict"])
            print(f"  ✓ SAE weights restored from {ckpt_dir}")
        else:
            print("\n[5/5] Fresh training run (no checkpoint restore) …")

        self.cfg.ensure_save_dir()
        print("\n  Setup complete ✓")

    # ── Training ─────────────────────────────────────────────────────────

    def train(self) -> list[dict]:
        assert self.sae is not None, "Call .setup() first."
        assert self.sequence_plan is not None
        assert self.full_cache_dir is not None
        assert self.rolling_cache_dir is not None

        cfg = self.cfg
        sae = self.sae
        device = self.device

        resume_state = None
        ckpt_dir: Optional[Path] = None
        run_dir: Path
        if cfg.training.resume_from:
            run_dir, ckpt_dir = self._resolve_resume_dir(cfg.training.resume_from)
            state_path = ckpt_dir / "training_state.pt"
            resume_state = torch.load(state_path, map_location="cpu", weights_only=True)
            print(f"  ♻  Resuming into existing run directory: {run_dir}")
            print(f"     Checkpoint loaded from: {ckpt_dir}")

        run_name: str = ""
        wandb_run = None
        if cfg.logging.wandb_enabled:
            import os
            import wandb

            api_key = cfg.logging.wandb_api_key or os.environ.get("WANDB_API_KEY")
            if api_key:
                wandb.login(key=api_key)

            wandb_init_kwargs = {
                "entity": cfg.logging.wandb_entity,
                "project": cfg.logging.wandb_project,
                "config": cfg.to_dict(),
                "tags": ["birna-bert", "sae", self.layer_tag, self.activation_strategy],
                "notes": (
                    f"SAE training on BiRNA-BERT layer {cfg.model.layer_index}, "
                    f"strategy={self.activation_strategy}, "
                    f"stats_subset={self.stats_num_sequences}"
                ),
            }
            if resume_state is not None and resume_state.get("run_name"):
                wandb_init_kwargs["name"] = resume_state["run_name"]

            wandb_run = wandb.init(
                **wandb_init_kwargs,
            )
            run_name = (resume_state or {}).get("run_name") or wandb_run.name
            print(f"  W&B run: {wandb_run.url}")
        else:
            run_name = (resume_state or {}).get("run_name") or f"run_{int(time.time())}"

        self.run_name = run_name

        if not cfg.training.resume_from:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = Path(cfg.logging.save_dir) / f"{timestamp}_{self.layer_tag}_{run_name}"
            run_dir.mkdir(parents=True, exist_ok=True)
        self.run_dir = run_dir
        self.metrics_csv_path = run_dir / "checkpoint_metrics.csv"
        cfg.logging.save_dir = str(run_dir)
        print(f"  Run output directory: {run_dir}")

        if self.cfg.data.standardize_activations and self.stats_cache_path is not None and self.stats_cache_path.exists():
            shutil.copy2(self.stats_cache_path, run_dir / self.stats_cache_path.name)
        with open(run_dir / "sequence_plan_manifest.json", "w") as handle:
            json.dump(self.sequence_plan, handle, indent=2)

        optimizer = torch.optim.AdamW(
            sae.parameters(),
            lr=cfg.training.learning_rate,
            betas=(cfg.training.beta1, cfg.training.beta2),
            weight_decay=cfg.training.weight_decay,
        )
        print(
            f"  Optimizer: AdamW (lr={cfg.training.learning_rate}, "
            f"weight_decay={cfg.training.weight_decay})"
        )

        steps_per_epoch = self.effective_train_tokens_per_epoch // cfg.training.batch_size
        total_steps = self.total_planned_steps
        scheduler = None
        if cfg.training.lr_scheduler == "cosine":
            if cfg.training.lr_warmup_steps > 0:
                def lr_lambda(step: int) -> float:
                    if step < cfg.training.lr_warmup_steps:
                        return step / max(cfg.training.lr_warmup_steps, 1)
                    progress = (step - cfg.training.lr_warmup_steps) / max(
                        total_steps - cfg.training.lr_warmup_steps, 1
                    )
                    return 0.5 * (1.0 + np.cos(np.pi * progress))

                scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            else:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=total_steps,
                    eta_min=0,
                )
            print(f"  LR scheduler: Cosine Annealing (T_max={total_steps:,})")
        else:
            print("  LR scheduler: None (constant LR)")

        use_amp = cfg.training.mixed_precision and device != "cpu"
        scaler = torch.amp.GradScaler(enabled=use_amp)
        amp_dtype = torch.float16 if use_amp else None
        if use_amp:
            print("  Mixed precision enabled")

        losses_history: list[dict] = []
        global_step = 0
        start_epoch = 0
        resume_cursor = self._epoch_boundary_cursor(0)

        if resume_state is not None:
            print("\n  ♻  Restoring optimizer / scheduler / scaler state …")
            sae.load_state_dict(resume_state["model_state_dict"])
            optimizer.load_state_dict(resume_state["optimizer_state_dict"])
            if scheduler is not None and "scheduler_state_dict" in resume_state:
                scheduler.load_state_dict(resume_state["scheduler_state_dict"])
            if "scaler_state_dict" in resume_state:
                scaler.load_state_dict(resume_state["scaler_state_dict"])
            resume_cursor = self._normalize_resume_cursor(resume_state)
            start_epoch = int(resume_cursor["next_epoch"])
            global_step = resume_state["global_step"]
            losses_history = resume_state.get("losses_history", [])
            if start_epoch >= self.num_full_epochs:
                print(
                    "  ✓ Resume checkpoint already reached the configured training horizon; "
                    "no additional optimizer steps remain."
                )
            elif self.activation_strategy == "rolling":
                print(
                    "  ✓ Resumed at "
                    f"epoch {start_epoch + 1}, chunk {int(resume_cursor['next_chunk_idx']) + 1}, "
                    f"batch {int(resume_cursor['batches_completed_in_chunk'])}, "
                    f"global step {global_step}"
                )
            else:
                print(
                    "  ✓ Resumed at "
                    f"epoch {start_epoch + 1}, batch {int(resume_cursor['batches_completed_in_epoch'])}, "
                    f"global step {global_step}"
                )

        hf_token = None
        if cfg.logging.hf_upload_enabled:
            import os as _os
            hf_token = cfg.logging.hf_token or _os.environ.get("HF_TOKEN")
            if not hf_token:
                print("  ⚠  HF upload enabled but no token found; upload will be skipped.")
            elif not cfg.logging.hf_repo_id:
                print("  ⚠  HF upload enabled but no hf_repo_id specified; upload will be skipped.")
                hf_token = None
            else:
                print(f"  HF Hub upload enabled → {cfg.logging.hf_repo_id}")

        if self.activation_strategy == "full":
            self._prepare_full_activation_cache()

        eval_interval = (
            cfg.evaluation.eval_every_n_steps
            or cfg.training.checkpoint_every_n_steps
            or cfg.training.log_every_n_steps
        )

        print("\n" + "=" * 70)
        epoch_label = (
            f"{cfg.training.num_epochs}" if cfg.training.num_epochs == int(cfg.training.num_epochs)
            else f"{cfg.training.num_epochs:.2f}"
        )
        print(
            f"  Training ({self.layer_tag}) — {epoch_label} epoch(s) | "
            f"{steps_per_epoch:,} steps/full-epoch | "
            f"{total_steps:,} total steps | strategy={self.activation_strategy}"
        )
        print(
            f"  Planned tokens/epoch: {self.total_train_tokens:,} | "
            f"Effective tokens/epoch: {self.effective_train_tokens_per_epoch:,} | "
            f"Target tokens: {self.total_training_token_budget:,}"
        )
        print(
            f"  Train tokens: {self.total_train_tokens:,} | Eval sequences: {len(self.eval_sequences):,} | "
            f"Checkpoint every {cfg.training.checkpoint_every_n_steps:,} steps"
            if cfg.training.checkpoint_every_n_steps is not None
            else "Checkpoints: final only"
        )
        print("=" * 70 + "\n")

        for epoch in range(start_epoch, self.num_full_epochs):
            is_last_epoch = (epoch == self.num_full_epochs - 1)
            print(f"\n{'─' * 60}")
            if is_last_epoch and cfg.training.num_epochs != int(cfg.training.num_epochs):
                frac = cfg.training.num_epochs - math.floor(cfg.training.num_epochs)
                print(f"  Epoch {epoch + 1}/{self.num_full_epochs} (partial — {frac:.0%} of data)")
            else:
                print(f"  Epoch {epoch + 1}/{self.num_full_epochs}")
            print(f"{'─' * 60}")

            epoch_loss_sum = 0.0
            epoch_l2_sum = 0.0
            epoch_l1_sum = 0.0
            epoch_l0_sum = 0.0
            epoch_steps = 0
            avg_loss = None
            avg_l2 = None
            avg_l1 = None
            avg_l0 = None
            resume_this_epoch = resume_state is not None and epoch == start_epoch

            if self.activation_strategy == "full":
                start_batch = int(resume_cursor["batches_completed_in_epoch"]) if resume_this_epoch else 0
                train_loader, train_ds, train_sampler = build_sharded_dataloader(
                    self.full_cache_dir,
                    batch_size=cfg.training.batch_size,
                    shuffle=True,
                    seed=cfg.sae.seed + epoch,
                    start_batch=start_batch,
                    num_workers=cfg.data.dataloader_num_workers,
                    reserved_cores=cfg.data.dataloader_reserved_cores,
                    prefetch_factor=cfg.data.dataloader_prefetch_factor,
                )
                if train_sampler is not None:
                    train_sampler.set_epoch(epoch)

                if len(train_loader) == 0:
                    print("  No remaining full-cache batches in this epoch — skipping ahead.")
                else:
                    pbar = tqdm.tqdm(
                        train_loader,
                        desc=f"  {self.layer_tag} E{epoch + 1}",
                        total=len(train_loader),
                    )
                    epoch_steps = self._train_one_loader(
                        pbar,
                        sae,
                        optimizer,
                        scheduler,
                        scaler,
                        losses_history,
                        global_step,
                        epoch,
                        wandb_run,
                        eval_interval,
                        run_dir,
                        train_loader,
                        epoch_loss_sum,
                        epoch_l2_sum,
                        epoch_l1_sum,
                        epoch_l0_sum,
                        epoch_batch_offset=0,
                        loader_batch_offset=start_batch,
                        loader_total_batches=start_batch + len(train_loader),
                        total_step_limit=total_steps,
                    )
                    global_step += epoch_steps
                    epoch_loss_sum = self._last_epoch_loss_sum
                    epoch_l2_sum = self._last_epoch_l2_sum
                    epoch_l1_sum = self._last_epoch_l1_sum
                    epoch_l0_sum = self._last_epoch_l0_sum
                del train_loader, train_ds, train_sampler
            else:
                total_chunks = len(self.sequence_plan["chunks"])
                epoch_batches_completed = (
                    int(resume_cursor["batches_completed_in_epoch"]) if resume_this_epoch else 0
                )
                resume_chunk_idx = int(resume_cursor["next_chunk_idx"]) if resume_this_epoch else 0
                resume_chunk_batch = (
                    int(resume_cursor["batches_completed_in_chunk"]) if resume_this_epoch else 0
                )
                for chunk in self.sequence_plan["chunks"]:
                    chunk_idx = int(chunk["chunk_idx"])
                    if resume_this_epoch and chunk_idx < resume_chunk_idx:
                        continue

                    self._ensure_embedder_loaded()
                    rolling_dir = self._prepare_rolling_window_cache(chunk)
                    start_batch = (
                        resume_chunk_batch if resume_this_epoch and chunk_idx == resume_chunk_idx else 0
                    )
                    batches_before_chunk = (
                        epoch_batches_completed - start_batch
                        if resume_this_epoch and chunk_idx == resume_chunk_idx
                        else epoch_batches_completed
                    )
                    train_loader, train_ds, train_sampler = build_sharded_dataloader(
                        rolling_dir,
                        batch_size=cfg.training.batch_size,
                        shuffle=True,
                        seed=cfg.sae.seed + epoch,
                        start_batch=start_batch,
                        num_workers=cfg.data.dataloader_num_workers,
                        reserved_cores=cfg.data.dataloader_reserved_cores,
                        prefetch_factor=cfg.data.dataloader_prefetch_factor,
                    )
                    if train_sampler is not None:
                        train_sampler.set_epoch(epoch)

                    if len(train_loader) == 0:
                        print("  No remaining full batches in this rolling window — skipping ahead.")
                        resume_this_epoch = False
                    else:
                        pbar = tqdm.tqdm(
                            train_loader,
                            desc=f"  {self.layer_tag} E{epoch + 1} W{chunk_idx + 1}",
                            total=len(train_loader),
                        )
                        window_steps = self._train_one_loader(
                            pbar,
                            sae,
                            optimizer,
                            scheduler,
                            scaler,
                            losses_history,
                            global_step,
                            epoch,
                            wandb_run,
                            eval_interval,
                            run_dir,
                            train_loader,
                            epoch_loss_sum,
                            epoch_l2_sum,
                            epoch_l1_sum,
                            epoch_l0_sum,
                            epoch_batch_offset=batches_before_chunk,
                            loader_batch_offset=start_batch,
                            loader_total_batches=start_batch + len(train_loader),
                            chunk_idx=chunk_idx,
                            total_chunks=total_chunks,
                            total_step_limit=total_steps,
                        )
                        global_step += window_steps
                        epoch_steps += window_steps
                        epoch_batches_completed = batches_before_chunk + start_batch + window_steps
                        epoch_loss_sum = self._last_epoch_loss_sum
                        epoch_l2_sum = self._last_epoch_l2_sum
                        epoch_l1_sum = self._last_epoch_l1_sum
                        epoch_l0_sum = self._last_epoch_l0_sum
                        resume_this_epoch = False
                    del train_loader, train_ds, train_sampler
                    cleanup_shards(rolling_dir)
                    gc.collect()

                    # Break chunk loop early for fractional-epoch termination
                    if global_step >= total_steps:
                        break
                self._maybe_unload_embedder()

            if epoch_steps > 0:
                avg_loss = epoch_loss_sum / epoch_steps
                avg_l2 = epoch_l2_sum / epoch_steps
                avg_l1 = epoch_l1_sum / epoch_steps
                avg_l0 = epoch_l0_sum / epoch_steps
                print(
                    f"\n  Epoch {epoch + 1} summary: loss={avg_loss:.4f}, "
                    f"l2={avg_l2:.4f}, l1={avg_l1:.4f}, l0={avg_l0:.1f}"
                )
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "epoch/loss": avg_loss,
                            "epoch/l2_loss": avg_l2,
                            "epoch/l1_loss": avg_l1,
                            "epoch/l0_sparsity": avg_l0,
                            "epoch/tokens_seen": epoch_steps * cfg.training.batch_size,
                            "epoch": epoch + 1,
                        }
                    )

            epoch_subdir = run_dir / f"epoch_{epoch + 1}"
            epoch_subdir.mkdir(parents=True, exist_ok=True)
            sae.save_checkpoint(run_name, save_dir=str(epoch_subdir))
            self._save_training_state(
                epoch_subdir,
                sae,
                optimizer,
                scheduler,
                scaler,
                epoch,
                global_step,
                losses_history,
                run_name,
                self._epoch_boundary_cursor(epoch + 1),
            )
            if self.act_mean is not None and self.act_std is not None:
                torch.save({"mean": self.act_mean, "std": self.act_std}, epoch_subdir / f"{run_name}_act_stats.pt")
            with open(epoch_subdir / f"{run_name}_losses.pkl", "wb") as handle:
                pickle.dump(losses_history, handle, protocol=pickle.HIGHEST_PROTOCOL)
            cfg.save_yaml(epoch_subdir / f"{run_name}_config.yaml")

            if cfg.evaluation.enabled:
                dead_loader = None
                if self.activation_strategy == "full":
                    dead_loader, _, _ = build_sharded_dataloader(
                        self.full_cache_dir,
                        batch_size=cfg.training.batch_size,
                        shuffle=False,
                        num_workers=cfg.data.dataloader_num_workers,
                        reserved_cores=cfg.data.dataloader_reserved_cores,
                        prefetch_factor=cfg.data.dataloader_prefetch_factor,
                    )
                eval_payload = self._evaluate_checkpoint(
                    sae,
                    global_step=global_step,
                    epoch=epoch,
                    checkpoint_dir=epoch_subdir,
                    lr=optimizer.param_groups[0]["lr"],
                    train_snapshot={
                        "train/loss": avg_loss if epoch_steps > 0 else None,
                        "train/l2_loss": avg_l2 if epoch_steps > 0 else None,
                        "train/l1_loss": avg_l1 if epoch_steps > 0 else None,
                        "train/l0_sparsity": avg_l0 if epoch_steps > 0 else None,
                    },
                    dead_loader=dead_loader,
                )
                if wandb_run is not None and eval_payload:
                    wandb_run.log(eval_payload | {"epoch": epoch + 1, "step": global_step})
                del dead_loader

            if (
                hf_token
                and cfg.logging.hf_upload_every_n_epochs
                and (epoch + 1) % cfg.logging.hf_upload_every_n_epochs == 0
            ):
                self._upload_to_hf(
                    sae,
                    run_name,
                    epoch + 1,
                    hf_token,
                    cfg.logging.hf_repo_id,
                    str(epoch_subdir),
                )

        if cfg.evaluation.enabled and self.eval_sequences:
            print("\nRunning final evaluation …")
            self._ensure_embedder_loaded()
            self.final_eval = evaluate_sae(
                sae,
                self.embedder,
                self.eval_sequences[: cfg.evaluation.num_test_sequences],
                device=device,
                act_mean=self.act_mean,
                act_std=self.act_std,
            )
            if self.activation_strategy == "full":
                freq_loader, _, _ = build_sharded_dataloader(
                    self.full_cache_dir,
                    batch_size=cfg.training.batch_size,
                    shuffle=False,
                    num_workers=cfg.data.dataloader_num_workers,
                    reserved_cores=cfg.data.dataloader_reserved_cores,
                    prefetch_factor=cfg.data.dataloader_prefetch_factor,
                )
                n_dead = count_dead_neurons(
                    sae,
                    freq_loader,
                    num_batches=cfg.training.dead_neuron_sample_batches,
                    threshold=cfg.training.dead_neuron_threshold,
                    device=device,
                )
                self.final_eval["dead_neurons"] = float(n_dead)
                del freq_loader
            print(f"  Final eval: {self.final_eval}")
            if wandb_run is not None:
                wandb_run.log({f"final/{key}": value for key, value in self.final_eval.items()})

        self._maybe_unload_embedder()

        final_subdir = run_dir / "final"
        final_subdir.mkdir(parents=True, exist_ok=True)
        sae.save_checkpoint(run_name, save_dir=str(final_subdir))
        if self.act_mean is not None and self.act_std is not None:
            torch.save({"mean": self.act_mean, "std": self.act_std}, final_subdir / f"{run_name}_act_stats.pt")
        loss_path = final_subdir / f"{run_name}_losses.pkl"
        with open(loss_path, "wb") as handle:
            payload = (losses_history, self.final_eval) if self.final_eval else losses_history
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        cfg.save_yaml(final_subdir / f"{run_name}_config.yaml")
        cfg.save_yaml(run_dir / f"{run_name}_config.yaml")
        self._save_training_state(
            final_subdir,
            sae,
            optimizer,
            scheduler,
            scaler,
            self.num_full_epochs - 1,
            global_step,
            losses_history,
            run_name,
            self._epoch_boundary_cursor(self.num_full_epochs),
        )

        if hf_token and cfg.logging.hf_repo_id:
            self._upload_to_hf(
                sae,
                run_name,
                self.num_full_epochs,
                hf_token,
                cfg.logging.hf_repo_id,
                str(final_subdir),
                final=True,
            )

        if wandb_run is not None:
            wandb_run.finish()

        print("\n  Training complete ✓")
        return losses_history

    def _train_one_loader(
        self,
        pbar,
        sae: SparseAutoEncoder,
        optimizer: torch.optim.Optimizer,
        scheduler,
        scaler: torch.amp.GradScaler,
        losses_history: list[dict],
        global_step_start: int,
        epoch: int,
        wandb_run,
        eval_interval: int,
        run_dir: Path,
        dead_loader_source,
        epoch_loss_sum: float,
        epoch_l2_sum: float,
        epoch_l1_sum: float,
        epoch_l0_sum: float,
        epoch_batch_offset: int,
        loader_batch_offset: int,
        loader_total_batches: int,
        chunk_idx: Optional[int] = None,
        total_chunks: Optional[int] = None,
        total_step_limit: Optional[int] = None,
    ) -> int:
        cfg = self.cfg
        device = self.device
        use_amp = cfg.training.mixed_precision and device != "cpu"
        amp_dtype = torch.float16 if use_amp else None
        run_name = self.run_name or "checkpoint"

        local_steps = 0
        for batch in pbar:
            # ── Fractional-epoch early stop ────────────────────────────────
            current_global = global_step_start + local_steps
            if total_step_limit is not None and current_global >= total_step_limit:
                break

            batch = batch.to(device=device, dtype=torch.float32)
            with torch.amp.autocast(device_type=device.split(":")[0], dtype=amp_dtype, enabled=use_amp):
                loss, _x_recon, acts, l2_loss, l1_loss = sae(batch)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            if cfg.training.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(sae.parameters(), cfg.training.grad_clip_norm)

            if sae.use_decoder_unit_norm:
                sae.make_decoder_weights_and_grad_unit_norm()

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()

            l0 = (acts > 0).float().sum(dim=-1).mean().item()
            current_lr = optimizer.param_groups[0]["lr"]
            current_step = global_step_start + local_steps + 1
            current_batch_in_chunk = loader_batch_offset + local_steps + 1
            current_batch_in_epoch = epoch_batch_offset + current_batch_in_chunk
            tokens_seen = current_step * cfg.training.batch_size
            tokens_seen_epoch = current_batch_in_epoch * cfg.training.batch_size

            epoch_loss_sum += loss.item()
            epoch_l2_sum += l2_loss.item()
            epoch_l1_sum += l1_loss.item()
            epoch_l0_sum += l0

            loss_dict = {
                "train/loss": loss.item(),
                "train/l2_loss": l2_loss.item(),
                "train/l1_loss": l1_loss.item(),
                "train/l0_sparsity": l0,
                "train/lr": current_lr,
                "train/tokens_seen": tokens_seen,
                "train/tokens_seen_pct": tokens_seen / max(self.total_training_token_budget, 1),
                "train/tokens_seen_epoch": tokens_seen_epoch,
                "train/tokens_seen_epoch_pct": tokens_seen_epoch / max(self.effective_train_tokens_per_epoch, 1),
                "epoch": epoch + 1,
                "step": current_step,
                "layer_index": self.layer_index,
            }

            if current_step % cfg.training.log_every_n_steps == 0:
                losses_history.append(loss_dict)
                if wandb_run is not None:
                    wandb_run.log(loss_dict)
                pbar.set_postfix(
                    loss=f"{loss_dict['train/loss']:.4f}",
                    l2=f"{loss_dict['train/l2_loss']:.4f}",
                    l0=f"{l0:.1f}",
                    lr=f"{current_lr:.2e}",
                    tok=f"{tokens_seen / 1_000_000:.2f}M",
                )

            checkpoint_due = (
                cfg.training.checkpoint_every_n_steps is not None
                and current_step % cfg.training.checkpoint_every_n_steps == 0
            )
            eval_due = (
                cfg.evaluation.enabled
                and cfg.evaluation.mid_training
                and eval_interval is not None
                and current_step % eval_interval == 0
            )

            if checkpoint_due:
                ckpt_subdir = run_dir / f"step_{current_step}"
                ckpt_subdir.mkdir(parents=True, exist_ok=True)
                sae.save_checkpoint(run_name, save_dir=str(ckpt_subdir))
                resume_cursor = self._build_resume_cursor(
                    epoch=epoch,
                    current_batch_in_epoch=current_batch_in_epoch,
                    current_batch_in_chunk=current_batch_in_chunk,
                    total_batches_in_chunk=loader_total_batches,
                    chunk_idx=chunk_idx,
                    total_chunks=total_chunks,
                )
                self._save_training_state(
                    ckpt_subdir,
                    sae,
                    optimizer,
                    scheduler,
                    scaler,
                    epoch,
                    current_step,
                    losses_history,
                    run_name,
                    resume_cursor,
                )
                if self.act_mean is not None and self.act_std is not None:
                    torch.save({"mean": self.act_mean, "std": self.act_std}, ckpt_subdir / f"{run_name}_act_stats.pt")
                with open(ckpt_subdir / f"{run_name}_losses.pkl", "wb") as handle:
                    pickle.dump(losses_history, handle, protocol=pickle.HIGHEST_PROTOCOL)
                cfg.save_yaml(ckpt_subdir / f"{run_name}_config.yaml")

            if eval_due:
                dead_loader = dead_loader_source if self.activation_strategy == "full" else None
                eval_payload = self._evaluate_checkpoint(
                    sae,
                    global_step=current_step,
                    epoch=epoch,
                    checkpoint_dir=(run_dir / f"step_{current_step}") if checkpoint_due else None,
                    lr=current_lr,
                    train_snapshot=loss_dict,
                    dead_loader=dead_loader,
                )
                if wandb_run is not None and eval_payload:
                    wandb_run.log(eval_payload | {"epoch": epoch + 1, "step": current_step})

            if (
                cfg.training.dead_neuron_check_steps is not None
                and cfg.training.dead_neuron_check_steps > 0
                and current_step % cfg.training.dead_neuron_check_steps == 0
            ):
                freq_loader = self._build_frequency_loader(dead_loader_source)
                if freq_loader is None or len(freq_loader) == 0:
                    print(
                        f"  Step {current_step}: skipping dead-neuron resampling because the active shard cache is unavailable"
                    )
                else:
                    freqs = sae.compute_neuron_frequencies(
                        freq_loader,
                        num_batches=cfg.training.dead_neuron_sample_batches,
                        device=device,
                    )
                    dead_mask = freqs < cfg.training.dead_neuron_threshold
                    n_dead = int(dead_mask.sum().item())
                    n_reset = sae.resample_dead_neurons_anthropic(
                        dead_mask,
                        freq_loader,
                        optimizer,
                        num_batches=cfg.training.dead_neuron_sample_batches,
                        device=device,
                    )
                    print(f"  Step {current_step}: dead={n_dead}, resampled={n_reset}")
                    if wandb_run is not None:
                        wandb_run.log(
                            {
                                "neurons/dead_count": n_dead,
                                "neurons/dead_pct": 100.0 * n_dead / sae.d_hidden,
                                "neurons/reset_count": n_reset,
                                "step": current_step,
                                "epoch": epoch + 1,
                            }
                        )
                del freq_loader

            if checkpoint_due:
                self._prune_old_step_checkpoints()

            del loss, acts, l2_loss, l1_loss, batch
            local_steps += 1

        self._last_epoch_loss_sum = epoch_loss_sum
        self._last_epoch_l2_sum = epoch_l2_sum
        self._last_epoch_l1_sum = epoch_l1_sum
        self._last_epoch_l0_sum = epoch_l0_sum
        return local_steps

    # ── Training state persistence ───────────────────────────────────────

    @staticmethod
    def _save_training_state(
        run_dir: Path,
        sae: SparseAutoEncoder,
        optimizer: torch.optim.Optimizer,
        scheduler,
        scaler: torch.amp.GradScaler,
        epoch: int,
        global_step: int,
        losses_history: list[dict],
        run_name: str,
        resume_cursor: Optional[dict[str, int | str]] = None,
    ) -> None:
        state = {
            "model_state_dict": sae.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "losses_history": losses_history,
            "run_name": run_name,
        }
        if resume_cursor is not None:
            state["resume_cursor"] = resume_cursor
        if scheduler is not None:
            state["scheduler_state_dict"] = scheduler.state_dict()
        torch.save(state, run_dir / "training_state.pt")

    # ── HuggingFace Hub Upload ───────────────────────────────────────────

    @staticmethod
    def _upload_to_hf(
        sae: SparseAutoEncoder,
        run_name: str,
        epoch: int,
        hf_token: str,
        repo_id: str,
        save_dir: str,
        final: bool = False,
    ) -> None:
        try:
            from huggingface_hub import HfApi

            api = HfApi(token=hf_token)
            api.create_repo(repo_id=repo_id, exist_ok=True, private=True)

            save_path = Path(save_dir)
            label = "final" if final else f"epoch_{epoch}"
            commit_msg = (
                f"Final model after {epoch} epochs (run: {run_name})"
                if final
                else f"Checkpoint at epoch {epoch} (run: {run_name})"
            )

            print(f"\n  📤 Uploading to HF Hub ({repo_id}) [{label}] …")
            api.upload_folder(
                folder_path=str(save_path),
                repo_id=repo_id,
                path_in_repo=f"runs/{run_name}",
                commit_message=commit_msg,
            )
            print(f"  ✓ Uploaded to https://huggingface.co/{repo_id}")
        except Exception as exc:
            print(f"  ⚠  HF Hub upload failed: {exc}")
