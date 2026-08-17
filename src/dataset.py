"""
Dataset and caching utilities for BiRNA-BERT SAE training.

The current pipeline is built around three ideas:
1. Stream raw RNA sequences from FASTA / text / HuggingFace datasets.
2. Plan exact token-count-aware train chunks on disk so the full train set is
   never materialised in RAM.
3. Cache activations either as a full layer-wide shard set (fastest when disk
   allows) or as rolling windows (bounded-disk fallback).

A small, fixed training subset is used to compute global activation mean/std
statistics once per layer. Those statistics are then reused for every window,
checkpoint, and inference path.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import random
import shutil
import threading
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path
from queue import Queue
from typing import Iterable, Iterator, Optional

import torch
import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from .config import DataConfig, EvaluationConfig, ModelConfig
from .model import BiRNABERTEmbedder


# ═══════════════════════════════════════════════════════════════════════════
#  1.  Lazy Sequence Iterators
# ═══════════════════════════════════════════════════════════════════════════


def iterate_rna_sequences(cfg: DataConfig) -> Iterator[str]:
    """
    Yield RNA sequences lazily from the configured data source.
    """
    if cfg.fasta_path:
        yield from _iter_fasta(cfg.fasta_path)
    elif cfg.sequence_list:
        yield from _iter_text(cfg.sequence_list)
    elif cfg.hf_dataset:
        yield from _iter_hf(
            cfg.hf_dataset,
            split=cfg.hf_dataset_split,
            column=cfg.hf_sequence_column,
            num_sequences=cfg.num_sequences,
        )
    else:
        raise ValueError(
            "No data source specified. Set one of: "
            "data.fasta_path, data.sequence_list, or data.hf_dataset"
        )



def _iter_fasta(path: str) -> Iterator[str]:
    current: list[str] = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current:
                    yield "".join(current).upper().replace("T", "U")
                    current = []
            else:
                current.append(line)
    if current:
        yield "".join(current).upper().replace("T", "U")



def _iter_text(path: str) -> Iterator[str]:
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield line.upper().replace("T", "U")



def _iter_hf(
    dataset_id: str,
    split: str,
    column: str,
    num_sequences: int | None = None,
) -> Iterator[str]:
    from datasets import load_dataset

    print(f"  Streaming sequences from {dataset_id} (split={split}) …")
    dataset = load_dataset(dataset_id, split=split, streaming=True)
    count = 0
    for row in dataset:
        seq = row[column].upper().replace("T", "U")
        yield seq
        count += 1
        if num_sequences is not None and count >= num_sequences:
            break


# ── Legacy eager loader (kept for backward compat / small datasets) ──────


def load_rna_sequences(cfg: DataConfig) -> list[str]:
    sequences = list(iterate_rna_sequences(cfg))
    if cfg.num_sequences is not None and cfg.num_sequences < len(sequences):
        random.shuffle(sequences)
        sequences = sequences[: cfg.num_sequences]
    print(f"  Loaded {len(sequences)} RNA sequences.")
    return sequences


# ═══════════════════════════════════════════════════════════════════════════
#  2.  Legacy Train / Test Split Helpers
# ═══════════════════════════════════════════════════════════════════════════


def split_sequences_streaming(
    seq_iter: Iterator[str],
    test_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    """
    Legacy helper retained for notebook/backward compatibility.
    """
    rng = random.Random(seed)
    train: list[str] = []
    test: list[str] = []

    for seq in seq_iter:
        if rng.random() < test_ratio:
            test.append(seq)
        else:
            train.append(seq)

    if not test and train:
        test.append(train.pop())

    print(f"  Train: {len(train)} sequences | Test: {len(test)} sequences")
    return train, test



def split_sequences(
    sequences: list[str], test_ratio: float = 0.1, seed: int = 42
) -> tuple[list[str], list[str]]:
    rng = random.Random(seed)
    indices = list(range(len(sequences)))
    rng.shuffle(indices)
    n_test = max(1, int(len(sequences) * test_ratio))
    test_idx = set(indices[:n_test])
    train = [sequences[i] for i in range(len(sequences)) if i not in test_idx]
    test = [sequences[i] for i in test_idx]
    print(f"  Train: {len(train)} sequences | Test: {len(test)} sequences")
    return train, test


# ═══════════════════════════════════════════════════════════════════════════
#  3.  Exact Token Count Planning
# ═══════════════════════════════════════════════════════════════════════════


def _batched(iterator: Iterable[str], batch_size: int) -> Iterator[list[str]]:
    iterator = iter(iterator)
    while True:
        batch = list(islice(iterator, batch_size))
        if not batch:
            return
        yield batch



def _prefetch_items(iterator: Iterable[list[str]], max_prefetch: int) -> Iterator[list[str]]:
    if max_prefetch <= 1:
        yield from iterator
        return

    queue: Queue[tuple[str, object]] = Queue(maxsize=max_prefetch)

    def producer() -> None:
        try:
            for item in iterator:
                queue.put(("item", item))
        except Exception as exc:
            queue.put(("error", exc))
        finally:
            queue.put(("done", None))

    thread = threading.Thread(target=producer, name="sequence-plan-prefetch", daemon=True)
    thread.start()

    while True:
        kind, payload = queue.get()
        if kind == "item":
            yield payload  # type: ignore[misc]
            continue
        if kind == "error":
            raise payload  # type: ignore[misc]
        break

    thread.join()



def _normalise_sequence_for_tokenizer(sequence: str, use_bpe: bool) -> str:
    if use_bpe:
        return sequence
    return " ".join(sequence)



def _count_tokens_batch(
    tokenizer,
    sequences: list[str],
    use_bpe: bool,
    max_seq_length: int,
) -> list[int]:
    inputs = tokenizer(
        [_normalise_sequence_for_tokenizer(seq, use_bpe) for seq in sequences],
        return_tensors="pt",
        return_attention_mask=True,
        return_special_tokens_mask=True,
        truncation=True,
        max_length=max_seq_length,
        padding=True,
    )
    keep_mask = inputs["attention_mask"].bool() & ~inputs["special_tokens_mask"].bool()
    return keep_mask.sum(dim=1).tolist()



def _plan_signature(
    data_cfg: DataConfig,
    model_cfg: ModelConfig,
    eval_cfg: EvaluationConfig,
    seed: int,
) -> str:
    payload = {
        "fasta_path": data_cfg.fasta_path,
        "sequence_list": data_cfg.sequence_list,
        "hf_dataset": data_cfg.hf_dataset,
        "hf_dataset_split": data_cfg.hf_dataset_split,
        "hf_sequence_column": data_cfg.hf_sequence_column,
        "num_sequences": data_cfg.num_sequences,
        "test_split_ratio": data_cfg.test_split_ratio,
        "stats_num_sequences": data_cfg.stats_num_sequences,
        "planning_batch_size": data_cfg.planning_batch_size,
        "tokenizer": model_cfg.tokenizer,
        "use_bpe": model_cfg.use_bpe,
        "max_seq_length": model_cfg.max_seq_length,
        "eval_enabled": eval_cfg.enabled,
        "eval_subset_size": eval_cfg.num_test_sequences,
        "seed": seed,
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8"))
    return digest.hexdigest()[:16]



def _sequence_chunk_path(train_dir: Path, chunk_idx: int) -> Path:
    return train_dir / f"train_chunk_{chunk_idx:05d}.txt"



def load_sequence_plan(manifest_path: str | Path) -> dict:
    return json.loads(Path(manifest_path).read_text())



def plan_or_load_sequence_chunks(
    data_cfg: DataConfig,
    model_cfg: ModelConfig,
    eval_cfg: EvaluationConfig,
    seed: int = 42,
) -> dict:
    """
    Stream the dataset once, compute exact non-special token counts, and spool
    train sequences into token-budgeted chunk files on disk.

    The resulting manifest is layer-agnostic and can be reused across multiple
    SAE runs for different layer indices.
    """
    plan_root = Path(data_cfg.sequence_plan_dir)
    plan_root.mkdir(parents=True, exist_ok=True)

    signature = _plan_signature(data_cfg, model_cfg, eval_cfg, seed)
    train_dir = plan_root / f"{signature}_train"
    eval_path = plan_root / f"{signature}_eval.txt"
    manifest_path = plan_root / f"{signature}_manifest.json"
    chunk_token_budget = max(1, int(data_cfg.max_disk_shards * data_cfg.shard_size))

    if manifest_path.exists():
        manifest = load_sequence_plan(manifest_path)
        chunk_paths = [Path(chunk["path"]) for chunk in manifest.get("chunks", [])]
        manifest_budget = int(manifest.get("chunk_token_budget", -1))
        if (
            manifest_budget == chunk_token_budget
            and chunk_paths
            and all(path.exists() for path in chunk_paths)
            and Path(manifest["eval_sequences_path"]).exists()
        ):
            print(f"  Reusing cached sequence plan → {manifest_path}")
            return manifest
        print("  Rebuilding cached sequence-plan layout to match current shard budget …")

    if train_dir.exists():
        shutil.rmtree(train_dir)
    if eval_path.exists():
        eval_path.unlink()

    train_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_cfg.tokenizer)
    rng = random.Random(seed)

    total_train_sequences = 0
    total_train_tokens = 0
    stats_num_sequences = 0
    stats_total_tokens = 0
    eval_tokens = 0
    eval_sequences: list[str] = []

    chunks: list[dict] = []
    chunk_idx = 0
    chunk_path = _sequence_chunk_path(train_dir, chunk_idx)
    chunk_handle = chunk_path.open("w")
    chunk_sequences = 0
    chunk_tokens = 0

    def _finalize_chunk() -> None:
        nonlocal chunk_idx, chunk_path, chunk_handle, chunk_sequences, chunk_tokens
        chunk_handle.close()
        if chunk_sequences == 0:
            chunk_path.unlink(missing_ok=True)
            return
        chunks.append(
            {
                "chunk_idx": chunk_idx,
                "path": str(chunk_path),
                "num_sequences": chunk_sequences,
                "total_tokens": chunk_tokens,
            }
        )
        chunk_idx += 1
        chunk_path = _sequence_chunk_path(train_dir, chunk_idx)
        chunk_handle = chunk_path.open("w")
        chunk_sequences = 0
        chunk_tokens = 0

    sequence_batches = _batched(iterate_rna_sequences(data_cfg), data_cfg.planning_batch_size)
    sequence_batches = _prefetch_items(sequence_batches, data_cfg.planning_prefetch_batches)
    progress = tqdm.tqdm(desc="Planning exact token counts", unit="seq")

    for seq_batch in sequence_batches:
        token_counts = _count_tokens_batch(
            tokenizer,
            seq_batch,
            use_bpe=model_cfg.use_bpe,
            max_seq_length=model_cfg.max_seq_length,
        )
        for sequence, token_count in zip(seq_batch, token_counts):
            progress.update(1)
            if token_count <= 0:
                continue

            keep_for_eval = (
                eval_cfg.enabled
                and data_cfg.test_split_ratio > 0
                and len(eval_sequences) < eval_cfg.num_test_sequences
                and rng.random() < data_cfg.test_split_ratio
            )
            if keep_for_eval:
                eval_sequences.append(sequence)
                eval_tokens += token_count
                continue

            if chunk_sequences > 0 and chunk_tokens + token_count > chunk_token_budget:
                _finalize_chunk()

            chunk_handle.write(sequence)
            chunk_handle.write("\n")
            chunk_sequences += 1
            chunk_tokens += token_count
            total_train_sequences += 1
            total_train_tokens += token_count

            if stats_num_sequences < data_cfg.stats_num_sequences:
                stats_num_sequences += 1
                stats_total_tokens += token_count

    progress.close()
    chunk_handle.close()

    if chunk_sequences > 0:
        chunks.append(
            {
                "chunk_idx": chunk_idx,
                "path": str(chunk_path),
                "num_sequences": chunk_sequences,
                "total_tokens": chunk_tokens,
            }
        )
    else:
        chunk_path.unlink(missing_ok=True)

    if total_train_sequences == 0:
        raise RuntimeError("Sequence planning produced no training sequences.")

    with open(eval_path, "w") as handle:
        for sequence in eval_sequences:
            handle.write(sequence)
            handle.write("\n")

    manifest = {
        "signature": signature,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "plan_root": str(plan_root),
        "train_dir": str(train_dir),
        "eval_sequences_path": str(eval_path),
        "chunks": chunks,
        "chunk_token_budget": chunk_token_budget,
        "total_train_sequences": total_train_sequences,
        "total_train_tokens": total_train_tokens,
        "stats_num_sequences": stats_num_sequences,
        "stats_total_tokens": stats_total_tokens,
        "num_eval_sequences": len(eval_sequences),
        "eval_total_tokens": eval_tokens,
        "model_tokenizer": model_cfg.tokenizer,
        "max_seq_length": model_cfg.max_seq_length,
        "use_bpe": model_cfg.use_bpe,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(
        "  Sequence plan ready: "
        f"{total_train_sequences:,} train seqs / {total_train_tokens:,} train tokens / "
        f"{len(eval_sequences):,} eval seqs"
    )
    print(f"  Sequence chunks: {len(chunks)} | Target tokens per chunk: {chunk_token_budget:,}")
    print(f"  Stats subset: first {stats_num_sequences:,} train seqs / {stats_total_tokens:,} tokens")
    return manifest



def iter_sequences_from_chunk(path: str | Path) -> Iterator[str]:
    with open(path) as handle:
        for line in handle:
            sequence = line.strip()
            if sequence:
                yield sequence



def iter_train_sequences_from_plan(manifest: dict, limit: Optional[int] = None) -> Iterator[str]:
    emitted = 0
    for chunk in manifest.get("chunks", []):
        for sequence in iter_sequences_from_chunk(chunk["path"]):
            yield sequence
            emitted += 1
            if limit is not None and emitted >= limit:
                return



def load_eval_sequences(manifest: dict) -> list[str]:
    path = Path(manifest["eval_sequences_path"])
    if not path.exists():
        return []
    with open(path) as handle:
        return [line.strip() for line in handle if line.strip()]



def estimate_activation_storage_bytes(
    total_tokens: int,
    hidden_dim: int,
    storage_dtype: torch.dtype = torch.float16,
) -> int:
    bytes_per_value = torch.tensor([], dtype=storage_dtype).element_size()
    return int(total_tokens) * int(hidden_dim) * bytes_per_value



def resolve_activation_strategy(
    manifest: dict,
    hidden_dim: int,
    data_cfg: DataConfig,
    storage_dtype: torch.dtype = torch.float16,
) -> tuple[str, float]:
    strategy = data_cfg.activation_strategy.lower()
    if strategy not in {"auto", "full", "rolling"}:
        raise ValueError("data.activation_strategy must be one of: auto, full, rolling")

    total_tokens = int(manifest["total_train_tokens"])
    est_bytes = estimate_activation_storage_bytes(total_tokens, hidden_dim, storage_dtype)
    est_gb = est_bytes / (1000**3)

    if strategy == "auto":
        if est_gb <= data_cfg.activation_cache_budget_gb:
            return "full", est_gb
        return "rolling", est_gb
    return strategy, est_gb


# ═══════════════════════════════════════════════════════════════════════════
#  4.  Activation Statistics
# ═══════════════════════════════════════════════════════════════════════════


def compute_activation_statistics_from_sequences(
    embedder: BiRNABERTEmbedder,
    sequences: Iterable[str],
    num_sequences: Optional[int] = None,
    extraction_batch_size: int = 16,
    extraction_chunk_size: int = 1000,
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """
    Compute per-dimension mean/std on a fixed subset of sequences.

    Returns
    -------
    mean, std, total_tokens, num_sequences_processed
    """
    seq_iter = iter(sequences)
    running_sum = torch.zeros(embedder.hidden_dim, dtype=torch.float64)
    running_sq = torch.zeros(embedder.hidden_dim, dtype=torch.float64)
    total_tokens = 0
    total_sequences = 0

    progress_total = num_sequences if num_sequences is not None else None
    progress = tqdm.tqdm(total=progress_total, desc="Stats subset forward", unit="seq")

    while True:
        remaining = None if num_sequences is None else num_sequences - total_sequences
        if remaining is not None and remaining <= 0:
            break
        take = extraction_chunk_size if remaining is None else min(extraction_chunk_size, remaining)
        chunk = list(islice(seq_iter, take))
        if not chunk:
            break

        total_sequences += len(chunk)
        progress.update(len(chunk))
        chunk = sorted(chunk, key=len)

        for batch_start in range(0, len(chunk), extraction_batch_size):
            batch = chunk[batch_start : batch_start + extraction_batch_size]
            embs = embedder.extract_embeddings_batch(batch)
            for emb in embs:
                if emb.numel() == 0:
                    continue
                emb64 = emb.to(dtype=torch.float64, device="cpu")
                running_sum += emb64.sum(dim=0)
                running_sq += (emb64 * emb64).sum(dim=0)
                total_tokens += emb64.shape[0]

    progress.close()

    if total_tokens <= 0:
        raise RuntimeError("Could not compute activation statistics: no non-special tokens were extracted.")

    mean = running_sum / total_tokens
    variance = (running_sq / total_tokens) - mean.pow(2)
    variance = variance.clamp(min=1e-12)
    std = variance.sqrt().clamp(min=1e-6)

    print(
        f"  Stats subset complete: {total_sequences:,} seqs / {total_tokens:,} tokens | "
        f"mean_of_means={mean.mean():.6f}, mean_of_stds={std.mean():.6f}"
    )
    return mean.float(), std.float(), total_tokens, total_sequences


# ═══════════════════════════════════════════════════════════════════════════
#  5.  Activation Extraction → Disk Shards
# ═══════════════════════════════════════════════════════════════════════════


def _shard_path(shard_dir: Path, idx: int) -> Path:
    return shard_dir / f"shard_{idx:05d}.pt"



def _manifest_path(shard_dir: Path) -> Path:
    return shard_dir / "manifest.json"



def extract_activations_to_shards(
    embedder: BiRNABERTEmbedder,
    sequences: Iterable[str],
    shard_dir: str | Path,
    extraction_batch_size: int = 16,
    extraction_chunk_size: int = 1000,
    shard_size: int = 500_000,
    device: str = "cuda",
    storage_dtype: torch.dtype = torch.float16,
    total_sequences: Optional[int] = None,
    progress_desc: str = "Extracting → shards",
    normalize_mean: Optional[torch.Tensor] = None,
    normalize_std: Optional[torch.Tensor] = None,
) -> dict:
    """
    Extract activations and write them to numbered shard files.

    If normalize_mean/std are provided, activations are standardized on write,
    which avoids an extra read/write pass over the shards.
    """
    shard_dir = Path(shard_dir)
    if shard_dir.exists():
        shutil.rmtree(shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)

    hidden_dim = embedder.hidden_dim
    shard_idx = 0
    total_tokens = 0
    processed_sequences = 0
    shard_token_counts: list[int] = []

    mean_cpu = None if normalize_mean is None else normalize_mean.to(dtype=torch.float32, device="cpu")
    std_cpu = None if normalize_std is None else normalize_std.to(dtype=torch.float32, device="cpu")

    buffer: list[torch.Tensor] = []
    buffer_tokens = 0
    sequence_iter = iter(sequences)
    use_amp = device.startswith("cuda") and torch.cuda.is_available()

    def _flush_buffer() -> None:
        nonlocal shard_idx, buffer, buffer_tokens
        if not buffer:
            return
        shard_tensor = torch.cat(buffer, dim=0)
        shard_token_counts.append(int(shard_tensor.shape[0]))
        torch.save(shard_tensor, _shard_path(shard_dir, shard_idx))
        shard_idx += 1
        buffer = []
        buffer_tokens = 0
        del shard_tensor

    progress = tqdm.tqdm(total=total_sequences, desc=progress_desc, unit="seq")

    while True:
        chunk = list(islice(sequence_iter, extraction_chunk_size))
        if not chunk:
            break
        processed_sequences += len(chunk)
        progress.update(len(chunk))
        chunk = sorted(chunk, key=len)

        for batch_start in range(0, len(chunk), extraction_batch_size):
            batch = chunk[batch_start : batch_start + extraction_batch_size]
            if use_amp:
                with torch.amp.autocast("cuda"):
                    embs = embedder.extract_embeddings_batch(batch)
            else:
                embs = embedder.extract_embeddings_batch(batch)

            for emb in embs:
                if emb.numel() == 0:
                    continue
                tensor = emb.to(dtype=torch.float32, device="cpu")
                if mean_cpu is not None and std_cpu is not None:
                    tensor = (tensor - mean_cpu) / std_cpu
                tensor = tensor.to(dtype=storage_dtype)
                buffer.append(tensor)
                buffer_tokens += tensor.shape[0]
                total_tokens += tensor.shape[0]

            if buffer_tokens >= shard_size:
                _flush_buffer()

    progress.close()
    _flush_buffer()

    manifest = {
        "num_shards": shard_idx,
        "total_tokens": total_tokens,
        "processed_sequences": processed_sequences,
        "hidden_dim": hidden_dim,
        "storage_dtype": str(storage_dtype),
        "shard_size": shard_size,
        "shard_token_counts": shard_token_counts,
        "standardized": mean_cpu is not None and std_cpu is not None,
    }
    with open(_manifest_path(shard_dir), "w") as handle:
        json.dump(manifest, handle, indent=2)

    size_gb = total_tokens * hidden_dim * torch.tensor([], dtype=storage_dtype).element_size() / (1000**3)
    print(
        f"  Wrote {shard_idx} shards ({total_tokens:,} tokens) → {shard_dir} "
        f"(~{size_gb:.2f} GB on disk)"
    )
    return manifest


# ── Legacy shard statistics helpers (kept for compatibility) ─────────────


def compute_shard_statistics(shard_dir: str | Path) -> tuple[torch.Tensor, torch.Tensor, int]:
    shard_dir = Path(shard_dir)
    manifest = json.loads(_manifest_path(shard_dir).read_text())
    num_shards = manifest["num_shards"]
    hidden_dim = manifest["hidden_dim"]

    total_count = 0
    running_sum = torch.zeros(hidden_dim, dtype=torch.float64)
    for i in tqdm.trange(num_shards, desc="Stats pass 1/2 (mean)"):
        shard = torch.load(_shard_path(shard_dir, i), map_location="cpu", weights_only=True).to(torch.float64)
        running_sum += shard.sum(dim=0)
        total_count += shard.shape[0]
        del shard
        gc.collect()

    mean = running_sum / max(total_count, 1)
    running_sq = torch.zeros(hidden_dim, dtype=torch.float64)
    for i in tqdm.trange(num_shards, desc="Stats pass 2/2 (std)"):
        shard = torch.load(_shard_path(shard_dir, i), map_location="cpu", weights_only=True).to(torch.float64)
        diff = shard - mean.unsqueeze(0)
        running_sq += (diff * diff).sum(dim=0)
        del shard, diff
        gc.collect()

    variance = (running_sq / max(total_count, 1)).clamp(min=1e-12)
    std = variance.sqrt().clamp(min=1e-6)
    print(f"  Stats over {total_count:,} tokens: mean_of_means={mean.mean():.6f}, mean_of_stds={std.mean():.6f}")
    return mean.float(), std.float(), total_count



def standardize_shards_inplace(
    shard_dir: str | Path,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> None:
    shard_dir = Path(shard_dir)
    manifest = json.loads(_manifest_path(shard_dir).read_text())
    num_shards = manifest["num_shards"]

    mean_cpu = mean.to(dtype=torch.float32, device="cpu")
    std_cpu = std.to(dtype=torch.float32, device="cpu")

    for i in tqdm.trange(num_shards, desc="Standardizing shards"):
        path = _shard_path(shard_dir, i)
        shard = torch.load(path, map_location="cpu", weights_only=True)
        dtype = shard.dtype
        shard = ((shard.float() - mean_cpu) / std_cpu).to(dtype)
        torch.save(shard, path)
        del shard
        gc.collect()

    print(f"  Standardized {num_shards} shards in-place.")


# ═══════════════════════════════════════════════════════════════════════════
#  6.  ShardedActivationDataset
# ═══════════════════════════════════════════════════════════════════════════


class ShardedActivationDataset(Dataset):
    """
    Dataset backed by numbered activation shards on disk.

    At most one shard is cached in memory at a time.
    """

    def __init__(self, shard_dir: str | Path):
        self.shard_dir = Path(shard_dir)
        manifest = json.loads(_manifest_path(self.shard_dir).read_text())

        self.num_shards: int = manifest["num_shards"]
        self.total_tokens: int = manifest["total_tokens"]
        self.hidden_dim: int = manifest["hidden_dim"]

        self._shard_ranges: list[tuple[int, int]] = []
        offset = 0
        shard_lengths = manifest.get("shard_token_counts")
        if shard_lengths is None:
            shard_lengths = []
            for i in range(self.num_shards):
                shard = torch.load(_shard_path(self.shard_dir, i), map_location="cpu", weights_only=True)
                shard_lengths.append(int(shard.shape[0]))
                del shard

        for length in shard_lengths:
            self._shard_ranges.append((offset, int(length)))
            offset += int(length)
        self.total_tokens = offset

        self._cached_shard_idx: int = -1
        self._cached_shard: Optional[torch.Tensor] = None

    def __len__(self) -> int:
        return self.total_tokens

    def __getitem__(self, idx: int) -> torch.Tensor:
        shard_idx = self._find_shard(idx)
        start, _length = self._shard_ranges[shard_idx]
        local_idx = idx - start

        if self._cached_shard_idx != shard_idx:
            self._cached_shard = torch.load(
                _shard_path(self.shard_dir, shard_idx),
                map_location="cpu",
                weights_only=True,
            )
            self._cached_shard_idx = shard_idx

        return self._cached_shard[local_idx]  # type: ignore[index]

    def _find_shard(self, idx: int) -> int:
        lo, hi = 0, self.num_shards - 1
        while lo < hi:
            mid = (lo + hi) // 2
            start, length = self._shard_ranges[mid]
            if idx < start + length:
                hi = mid
            else:
                lo = mid + 1
        return lo


class ShuffledShardSampler(torch.utils.data.Sampler):
    """
    Shuffle shard order, then shuffle indices within each shard.
    """

    def __init__(
        self,
        dataset: ShardedActivationDataset,
        seed: int = 42,
        start_offset: int = 0,
    ):
        self.dataset = dataset
        self.seed = seed
        self.epoch = 0
        self.start_offset = max(int(start_offset), 0)

    def set_epoch(self, epoch: int, start_offset: Optional[int] = None) -> None:
        self.epoch = epoch
        if start_offset is not None:
            self.start_offset = max(int(start_offset), 0)

    def set_start_offset(self, start_offset: int) -> None:
        self.start_offset = max(int(start_offset), 0)

    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self.seed + self.epoch)
        shard_order = list(range(self.dataset.num_shards))
        rng.shuffle(shard_order)
        skipped = 0

        for shard_idx in shard_order:
            start, length = self.dataset._shard_ranges[shard_idx]
            indices = list(range(start, start + length))
            rng.shuffle(indices)
            if skipped >= self.start_offset:
                yield from indices
                continue

            if skipped + len(indices) <= self.start_offset:
                skipped += len(indices)
                continue

            start_at = max(self.start_offset - skipped, 0)
            skipped += len(indices)
            yield from indices[start_at:]

    def __len__(self) -> int:
        return max(len(self.dataset) - self.start_offset, 0)



def build_sharded_dataloader(
    shard_dir: str | Path,
    batch_size: int,
    shuffle: bool = True,
    seed: int = 42,
    start_batch: int = 0,
    num_workers: Optional[int] = None,
    reserved_cores: int = 2,
    prefetch_factor: int = 2,
) -> tuple[DataLoader, ShardedActivationDataset, Optional[ShuffledShardSampler]]:
    dataset = ShardedActivationDataset(shard_dir)
    sampler = (
        ShuffledShardSampler(dataset, seed=seed, start_offset=max(int(start_batch), 0) * batch_size)
        if shuffle
        else None
    )
    num_batches = len(dataset) // batch_size
    if num_workers is None:
        available_cpus = max((os.cpu_count() or 1) - max(int(reserved_cores), 0), 0)
        num_workers = min(max(available_cpus, 0), dataset.num_shards, num_batches)
    num_workers = max(int(num_workers), 0)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )
    return loader, dataset, sampler


# ═══════════════════════════════════════════════════════════════════════════
#  7.  Legacy In-Memory Helpers
# ═══════════════════════════════════════════════════════════════════════════


class ActivationDataset(Dataset):
    """Simple in-memory dataset wrapping a 2-D activation tensor."""

    def __init__(self, activations: torch.Tensor):
        self.activations = activations

    def __len__(self) -> int:
        return self.activations.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.activations[idx]



def build_dataloader(
    activations: torch.Tensor,
    batch_size: int,
    shuffle: bool = True,
    num_workers: Optional[int] = None,
    reserved_cores: int = 2,
    prefetch_factor: int = 2,
) -> DataLoader:
    dataset = ActivationDataset(activations)
    num_batches = len(dataset) // batch_size
    if num_workers is None:
        available_cpus = max((os.cpu_count() or 1) - max(int(reserved_cores), 0), 0)
        num_workers = min(max(available_cpus, 0), num_batches)
    num_workers = max(int(num_workers), 0)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )



def standardize_activations(
    activations: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = activations.mean(dim=0)
    std = activations.std(dim=0).clamp(min=1e-6)
    standardized = (activations - mean) / std
    print(f"  Standardized activations: mean={mean.mean():.6f}, std={std.mean():.6f}")
    return standardized, mean, std


# ═══════════════════════════════════════════════════════════════════════════
#  8.  Cache Cleanup
# ═══════════════════════════════════════════════════════════════════════════


def cleanup_shards(shard_dir: str | Path) -> None:
    shard_dir = Path(shard_dir)
    if shard_dir.exists():
        shutil.rmtree(shard_dir)
        shard_dir.mkdir(parents=True, exist_ok=True)