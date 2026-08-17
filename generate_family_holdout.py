"""Generate the cached RNACentral family holdout with CD-HIT-EST-2D.

The process is resumable. It writes the training FASTA once, processes candidate
batches until the requested number of type-filtered sequences is available, and
then writes the deterministic parquet/audit pair consumed by the notebooks.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from itertools import islice
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from datasets import load_dataset
from tqdm.auto import tqdm


POLICY = "cd_hit_est_2d_80pct_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="multimolecule/rnacentral")
    parser.add_argument("--split", default="train")
    parser.add_argument("--training-size", type=int, default=1_000_000)
    parser.add_argument("--num-sequences", type=int, default=25_000)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--min-length", type=int, default=10)
    parser.add_argument("--min-type-count", type=int, default=50)
    parser.add_argument("--candidate-batch-size", type=int, default=50_000)
    parser.add_argument("--identity", type=float, default=0.80)
    parser.add_argument("--word-size", type=int, default=5)
    parser.add_argument(
        "--cd-hit-est-2d", dest="cd_hit_est_2d", default="cd-hit-est-2d"
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path(".cache/analysis/shared_holdouts"),
    )
    parser.add_argument(
        "--work-root",
        type=Path,
        default=Path(".cache/analysis/holdout_work/cd_hit_est_2d_80pct_v1"),
    )
    return parser.parse_args()


def normalize_sequence(sequence: str, max_seq_length: int) -> str:
    return sequence.upper().replace("T", "U")[: max(1, max_seq_length - 2)]


def write_fasta_record(handle, sequence_id: str, sequence: str) -> None:
    handle.write(f">{sequence_id}\n")
    for start in range(0, len(sequence), 80):
        handle.write(f"{sequence[start:start + 80]}\n")


def read_fasta_ids(path: Path) -> set[str]:
    with path.open() as handle:
        return {
            line[1:].strip().split()[0]
            for line in handle
            if line.startswith(">")
        }


def select_type_filtered_records(
    records: list[dict[str, Any]],
    *,
    target: int,
    min_type_count: int,
) -> list[dict[str, Any]] | None:
    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        return None
    counts = frame["rna_type"].value_counts()
    eligible_types = counts[counts >= min_type_count].index.tolist()
    eligible = frame[frame["rna_type"].isin(eligible_types)].copy()
    if len(eligible) < target:
        return None

    reserved_indices: list[int] = []
    for rna_type in eligible_types:
        reserved_indices.extend(
            eligible.index[eligible["rna_type"] == rna_type][:min_type_count].tolist()
        )
    if len(reserved_indices) > target:
        return None
    reserved = set(reserved_indices)
    selected_indices = reserved_indices + [
        index for index in eligible.index if index not in reserved
    ][: target - len(reserved_indices)]
    selected = frame.loc[selected_indices].copy()
    selected["seq_length"] = selected["sequence"].str.len().astype("int32")
    return selected.to_dict(orient="records")


def build_training_fasta(args: argparse.Namespace, training_fasta: Path) -> int:
    metadata_path = training_fasta.with_suffix(".metadata.json")
    expected = {
        "dataset": args.dataset,
        "split": args.split,
        "training_size": args.training_size,
        "max_seq_length": args.max_seq_length,
        "min_length": args.min_length,
    }
    if training_fasta.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if all(metadata.get(key) == value for key, value in expected.items()):
            print(f"Reusing training FASTA: {training_fasta}", flush=True)
            return int(metadata["sequences_written"])

    dataset = load_dataset(args.dataset, split=args.split, streaming=True)
    written = 0
    temporary_path = training_fasta.with_suffix(".tmp")
    with temporary_path.open("w") as handle:
        for row in tqdm(
            islice(dataset, args.training_size),
            total=args.training_size,
            desc="Writing training FASTA",
        ):
            sequence = normalize_sequence(str(row["sequence"]), args.max_seq_length)
            if len(sequence) < args.min_length:
                continue
            write_fasta_record(handle, f"training_{written:09d}", sequence)
            written += 1
    temporary_path.replace(training_fasta)
    metadata_path.write_text(
        json.dumps({**expected, "sequences_written": written}, indent=2, sort_keys=True)
    )
    return written


def run_cd_hit_est_2d(
    args: argparse.Namespace,
    training_fasta: Path,
    candidate_fasta: Path,
    filtered_fasta: Path,
) -> list[str]:
    executable = (
        shutil.which(args.cd_hit_est_2d)
        if "/" not in args.cd_hit_est_2d
        else args.cd_hit_est_2d
    )
    if not executable:
        raise FileNotFoundError(
            f"cd-hit-est-2d executable not found: {args.cd_hit_est_2d}"
        )
    command = [
        executable,
        "-i", str(training_fasta),
        "-i2", str(candidate_fasta),
        "-o", str(filtered_fasta),
        "-c", str(args.identity),
        "-n", str(args.word_size),
        "-G", "1",
        "-M", "0",
        "-T", "0",
        "-d", "0",
    ]
    started_at = time.monotonic()
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    next_heartbeat = started_at + 30.0
    while process.poll() is None:
        remaining = max(0.0, next_heartbeat - time.monotonic())
        try:
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            elapsed = time.monotonic() - started_at
            print(
                f"CD-HIT-EST-2D active: {filtered_fasta.name}; "
                f"elapsed {elapsed / 60:.1f} minutes; batch completion pending.",
                flush=True,
            )
            next_heartbeat += 30.0
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        diagnostic = "\n".join(
            part for part in (stdout, stderr) if part
        )
        raise RuntimeError(f"cd-hit-est-2d failed:\n{diagnostic[-8000:]}")
    if not filtered_fasta.exists():
        raise RuntimeError("cd-hit-est-2d did not produce its output FASTA")
    return command


def main() -> None:
    args = parse_args()
    if args.identity != 0.80:
        raise ValueError("This cache policy requires exactly 80% identity.")
    if args.word_size != 5:
        raise ValueError("This cache policy requires CD-HIT-EST-2D word size 5.")
    args.cache_root.mkdir(parents=True, exist_ok=True)
    args.work_root.mkdir(parents=True, exist_ok=True)

    stem = (
        f"rnacentral_holdout_{args.num_sequences}_{args.max_seq_length}_"
        f"{args.min_type_count}_{POLICY}"
    )
    output_path = args.cache_root / f"{stem}.parquet"
    audit_path = args.cache_root / (
        f"rnacentral_holdout_audit_{args.num_sequences}_{args.max_seq_length}_"
        f"{args.min_type_count}_{POLICY}.json"
    )
    if output_path.exists() and audit_path.exists():
        existing = pd.read_parquet(output_path)
        if len(existing) == args.num_sequences:
            print(f"Holdout already complete: {output_path}", flush=True)
            return

    training_fasta = args.work_root / "training.fasta"
    training_written = build_training_fasta(args, training_fasta)

    progress_path = args.work_root / "progress.json"
    retained_path = args.work_root / "retained.parquet"
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
    candidate_rows_consumed = int(progress.get("candidate_rows_consumed", 0))
    checkpoint_rows_before_deduplication = 0
    if retained_path.exists():
        retained_frame = pd.read_parquet(retained_path)
        checkpoint_rows_before_deduplication = len(retained_frame)
        retained_records = retained_frame.drop_duplicates(subset=["sequence"]).to_dict(orient="records")
    else:
        retained_records = []
    unique_checkpoint_records = len(retained_records)
    commands: list[list[str]] = list(progress.get("cd_hit_est_2d_commands", []))

    candidate_start = args.training_size + candidate_rows_consumed
    candidate_stream: Iterable[dict[str, Any]] | None = None
    batch_index = int(progress.get("batches_completed", 0))

    while True:
        selected = select_type_filtered_records(
            retained_records,
            target=args.num_sequences,
            min_type_count=args.min_type_count,
        )
        if selected is not None:
            break

        if candidate_stream is None:
            candidate_dataset = load_dataset(args.dataset, split=args.split, streaming=True)
            candidate_stream = islice(candidate_dataset, candidate_start, None)

        batch_rows = list(islice(candidate_stream, args.candidate_batch_size))
        if not batch_rows:
            raise RuntimeError("RNACentral candidate stream ended before the target was reached.")

        candidate_fasta = args.work_root / f"candidate_batch_{batch_index:05d}.fasta"
        filtered_fasta = args.work_root / f"filtered_batch_{batch_index:05d}.fasta"
        batch_records: dict[str, dict[str, Any]] = {}
        with candidate_fasta.open("w") as handle:
            for index, row in enumerate(batch_rows):
                sequence = normalize_sequence(str(row["sequence"]), args.max_seq_length)
                if len(sequence) < args.min_length:
                    continue
                sequence_id = f"candidate_{candidate_rows_consumed + index:012d}"
                batch_records[sequence_id] = {
                    "upi": row.get("upi", sequence_id),
                    "sequence": sequence,
                    "rna_type": row.get("type") or row.get("rna_type") or "unknown",
                    "seq_length": len(sequence),
                }
                write_fasta_record(handle, sequence_id, sequence)

        command = run_cd_hit_est_2d(args, training_fasta, candidate_fasta, filtered_fasta)
        commands.append(command)
        retained_ids = read_fasta_ids(filtered_fasta)
        retained_records.extend(
            record for sequence_id, record in batch_records.items() if sequence_id in retained_ids
        )
        candidate_rows_consumed += len(batch_rows)
        batch_index += 1

        pd.DataFrame.from_records(retained_records).to_parquet(retained_path, index=False)
        progress = {
            "policy": POLICY,
            "candidate_rows_consumed": candidate_rows_consumed,
            "raw_sequences_retained": len(retained_records),
            "batches_completed": batch_index,
            "cd_hit_est_2d_commands": commands,
        }
        progress_path.write_text(json.dumps(progress, indent=2, sort_keys=True))
        print(
            f"Batch {batch_index}: scanned {candidate_rows_consumed:,} candidates; "
            f"CD-HIT-EST-2D retained {len(retained_records):,} raw sequences.",
            flush=True,
        )

    retained_frame = pd.DataFrame.from_records(retained_records)
    retained_unique = retained_frame.drop_duplicates(subset=["sequence"])
    eligible_type_counts = retained_unique["rna_type"].value_counts()
    eligible_type_counts = eligible_type_counts[eligible_type_counts >= args.min_type_count]
    output = pd.DataFrame.from_records(selected)
    output.to_parquet(output_path, index=False)
    type_counts = output["rna_type"].value_counts().to_dict()
    audit = {
        "holdout_policy": POLICY,
        "similarity_filter": "cd-hit-est-2d",
        "similarity_threshold": args.identity,
        "global_identity": True,
        "word_size": args.word_size,
        "training_dataset_id": args.dataset,
        "training_split": args.split,
        "training_source_rows": args.training_size,
        "training_sequences_written": training_written,
        "candidate_rows_scanned": candidate_rows_consumed,
        "raw_sequences_retained_by_cd_hit_est_2d": len(retained_records),
        "generation_statistics": {
            "training_source_rows_checked": args.training_size,
            "candidate_rows_checked": candidate_rows_consumed,
            "total_source_rows_checked": args.training_size + candidate_rows_consumed,
            "candidate_batches_completed": batch_index,
            "checkpoint_rows_before_deduplication": checkpoint_rows_before_deduplication,
            "duplicate_checkpoint_rows_removed": (
                checkpoint_rows_before_deduplication - unique_checkpoint_records
            ),
            "unique_cd_hit_retained_before_selection": len(retained_unique),
            "eligible_rna_types": len(eligible_type_counts),
            "eligible_unique_sequences": int(eligible_type_counts.sum()),
            "final_holdout_sequences": len(output),
        },
        "returned_sequences": len(output),
        "min_type_count": args.min_type_count,
        "rna_type_counts": type_counts,
        "cd_hit_est_2d_commands": commands,
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True))
    print(f"Wrote {len(output):,} sequences to {output_path}", flush=True)
    print(f"Wrote audit to {audit_path}", flush=True)


if __name__ == "__main__":
    main()
