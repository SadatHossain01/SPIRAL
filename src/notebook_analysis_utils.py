from __future__ import annotations

import gc
import json
import shutil
import subprocess
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from datasets import load_dataset
from matplotlib import colors as mcolors
from scipy import stats as sp_stats
from tqdm.auto import tqdm

from .config import PipelineConfig
from .model import BiRNABERTEmbedder
from .sae import SparseAutoEncoder


STRUCTURE_LABELS = {
    "S": "Stem",
    "H": "Hairpin Loop",
    "I": "Internal Loop",
    "M": "Multi-loop",
    "B": "Bulge",
    "E": "External Loop",
    "X": "Ambiguous",
    "K": "Pseudoknot",
}

STRUCTURE_PALETTE = {
    "S": "#b20d30",
    "H": "#245c96",
    "I": "#0f7c6e",
    "M": "#d99b00",
    "B": "#db6b29",
    "E": "#72b7b2",
    "X": "#8d8d8d",
    "K": "#5b2a86",
}

CONTRAST_COLORS = [
    "#0057D9",
    "#C00000",
    "#008F4F",
    "#D06A00",
    "#7A00CC",
    "#00A6D6",
    "#8C510A",
    "#D81B60",
    "#004D40",
    "#6A3D9A",
    "#AD1457",
    "#2E7D32",
    "#1565C0",
    "#EF6C00",
    "#C62828",
    "#00838F",
    "#9E9D24",
    "#5D4037",
    "#283593",
    "#7B1FA2",
]

RNA_ALPHABET = tuple("ACGU")


@dataclass(frozen=True)
class LayerBundle:
    raw_path: Path
    checkpoint_dir: Path
    run_dir: Path
    weights: Path
    config: Path
    meta: Path | None
    act_stats: Path | None
    metrics_csv: Path
    best_checkpoint_json: Path
    analysis_slug: str
    layer_index: int | None
    best_metric_name: str | None
    best_metric_mode: str | None
    best_metric_value: float | None


def resolve_device(device_preference: str = "auto") -> str:
    """Resolve a requested runtime device across CUDA, Apple MPS, and CPU.

    ``auto`` is intentionally the notebook-friendly default: it uses CUDA when
    available, then Apple's Metal backend, and finally CPU.  An unavailable
    accelerator request also falls back to the next usable device so the
    notebooks remain runnable on machines without the requested backend.
    """
    preference = str(device_preference).strip().lower()
    mps_available = bool(
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
        and torch.backends.mps.is_built()
    )

    if preference in {"auto", "default"}:
        if torch.cuda.is_available():
            return "cuda"
        if mps_available:
            return "mps"
        return "cpu"

    if preference.startswith("cuda"):
        if torch.cuda.is_available():
            return preference
        return "mps" if mps_available else "cpu"

    if preference == "mps":
        return "mps" if mps_available else "cpu"

    if preference == "cpu":
        return "cpu"

    raise ValueError(
        f"Unknown device preference {device_preference!r}; use auto, cuda, mps, or cpu."
    )


def _checkpoint_dir_sort_key(path: Path) -> tuple[int, int, str]:
    if path.name == "final":
        return (2, 2**31 - 1, path.name)
    if path.name.startswith("step_"):
        suffix = path.name.split("_", 1)[1]
        return (1, int(suffix) if suffix.isdigit() else -1, path.name)
    if path.name.startswith("epoch_"):
        suffix = path.name.split("_", 1)[1]
        return (0, int(suffix) if suffix.isdigit() else -1, path.name)
    return (-1, -1, path.name)


def _slugify(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "-" for ch in value).strip("-")


def sha1_digest(payload: Mapping[str, Any], length: int = 10) -> str:
    import hashlib

    digest = hashlib.sha1(json.dumps(
        dict(payload), sort_keys=True).encode("utf-8"))
    return digest.hexdigest()[:length]


def _discover_checkpoint_artifacts(directory: Path) -> dict[str, Path | None]:
    weights = sorted(
        path
        for path in directory.glob("*.pt")
        if path.name != "training_state.pt" and not path.name.endswith("_act_stats.pt")
    )
    configs = sorted(directory.glob("*_config.yaml"))
    metas = sorted(directory.glob("*_meta.json"))
    act_stats = sorted(directory.glob("*_act_stats.pt"))

    if not weights:
        raise FileNotFoundError(
            f"No SAE checkpoint weights found in {directory}")
    if not configs:
        raise FileNotFoundError(f"No config YAML found in {directory}")

    return {
        "weights": weights[0],
        "config": configs[0],
        "meta": metas[0] if metas else None,
        "act_stats": act_stats[0] if act_stats else None,
    }


def _is_usable_checkpoint_dir(directory: Path) -> bool:
    if not directory.is_dir():
        return False
    try:
        _discover_checkpoint_artifacts(directory)
    except FileNotFoundError:
        return False
    return True


def _resolve_best_checkpoint_dir(run_dir: Path, best_payload: Mapping[str, Any]) -> Path | None:
    candidates: list[Path] = []
    checkpoint_relpath = best_payload.get("checkpoint_relpath")
    checkpoint_dir_value = best_payload.get("checkpoint_dir")

    if checkpoint_relpath:
        candidates.append(run_dir / str(checkpoint_relpath))

    if checkpoint_dir_value:
        raw_candidate = Path(str(checkpoint_dir_value)).expanduser()
        candidates.append(
            raw_candidate if raw_candidate.is_absolute() else run_dir / raw_candidate)
        if raw_candidate.name:
            candidates.append(run_dir / raw_candidate.name)

    seen: set[str] = set()
    for candidate in candidates:
        candidate_key = str(candidate)
        if candidate_key in seen:
            continue
        seen.add(candidate_key)
        if _is_usable_checkpoint_dir(candidate):
            return candidate

    fallback_candidates = [
        child
        for child in run_dir.iterdir()
        if _is_usable_checkpoint_dir(child)
    ]
    if fallback_candidates:
        return max(fallback_candidates, key=_checkpoint_dir_sort_key)
    return None


def resolve_checkpoint_bundle(checkpoint_path: str | Path) -> LayerBundle:
    raw_path = Path(checkpoint_path).expanduser().resolve()
    if not raw_path.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {raw_path}")

    if raw_path.is_file():
        checkpoint_dir = raw_path.parent
    elif _is_usable_checkpoint_dir(raw_path):
        checkpoint_dir = raw_path
    elif (raw_path / "best_checkpoint.json").exists():
        best_payload = json.loads(
            (raw_path / "best_checkpoint.json").read_text())
        checkpoint_dir = _resolve_best_checkpoint_dir(raw_path, best_payload)
        if checkpoint_dir is None:
            raise FileNotFoundError(
                f"best_checkpoint.json exists in {raw_path}, but no checkpoint directory could be resolved"
            )
    else:
        candidates = [
            child
            for child in raw_path.iterdir()
            if _is_usable_checkpoint_dir(child)
        ]
        if not candidates:
            raise FileNotFoundError(
                f"No checkpoint directories with SAE weights and config found under {raw_path}"
            )
        checkpoint_dir = max(candidates, key=_checkpoint_dir_sort_key)

    run_dir = (
        checkpoint_dir.parent
        if checkpoint_dir.name == "final" or checkpoint_dir.name.startswith(("step_", "epoch_"))
        else checkpoint_dir
    )
    artifacts = _discover_checkpoint_artifacts(checkpoint_dir)
    selection_name = checkpoint_dir.name if checkpoint_dir != run_dir else "run-root"
    cfg = PipelineConfig.from_yaml(artifacts["config"])

    best_metric_name = None
    best_metric_mode = None
    best_metric_value = None
    best_json = run_dir / "best_checkpoint.json"
    if best_json.exists():
        best_payload = json.loads(best_json.read_text())
        best_metric_name = best_payload.get("metric")
        best_metric_mode = best_payload.get("mode")
        if best_payload.get("value") is not None:
            best_metric_value = float(best_payload["value"])

    return LayerBundle(
        raw_path=raw_path,
        checkpoint_dir=checkpoint_dir,
        run_dir=run_dir,
        weights=artifacts["weights"],
        config=artifacts["config"],
        meta=artifacts["meta"],
        act_stats=artifacts["act_stats"],
        metrics_csv=run_dir / "checkpoint_metrics.csv",
        best_checkpoint_json=best_json,
        analysis_slug=_slugify(f"{run_dir.name}__{selection_name}"),
        layer_index=int(
            cfg.model.layer_index) if cfg.model.layer_index is not None else None,
        best_metric_name=best_metric_name,
        best_metric_mode=best_metric_mode,
        best_metric_value=best_metric_value,
    )


def auto_discover_latest_layer_paths(checkpoint_root: str | Path) -> dict[int, Path]:
    root = Path(checkpoint_root).expanduser().resolve()
    discovered: dict[int, tuple[float, Path]] = {}
    for candidate in root.iterdir():
        if not candidate.is_dir():
            continue
        try:
            bundle = resolve_checkpoint_bundle(candidate)
        except Exception:
            continue
        if bundle.layer_index is None:
            continue
        stamp = bundle.run_dir.stat().st_mtime
        cached = discovered.get(bundle.layer_index)
        if cached is None or stamp > cached[0]:
            discovered[bundle.layer_index] = (stamp, bundle.raw_path)
    return {layer: path for layer, (_, path) in sorted(discovered.items())}


def load_layer_catalog(layer_paths: Mapping[int, str | Path]) -> tuple[dict[int, LayerBundle], pd.DataFrame]:
    bundles = {int(layer): resolve_checkpoint_bundle(path)
               for layer, path in layer_paths.items()}
    rows: list[dict[str, Any]] = []
    for layer_index, bundle in sorted(bundles.items()):
        cfg = PipelineConfig.from_yaml(bundle.config)
        row: dict[str, Any] = {
            "layer_index": layer_index,
            "layer_label": f"Layer {layer_index}",
            "run_dir": str(bundle.run_dir),
            "checkpoint_dir": str(bundle.checkpoint_dir),
            "best_metric": bundle.best_metric_name,
            "best_metric_value": bundle.best_metric_value,
            "checkpoint_name": bundle.checkpoint_dir.name,
            "l1_coeff": cfg.sae.l1_coeff,
            "expansion_factor": cfg.sae.expansion_factor,
            "quantization": cfg.model.quantization,
        }
        metrics_csv = bundle.metrics_csv
        if metrics_csv.exists():
            metrics_df = pd.read_csv(metrics_csv)
            if not metrics_df.empty:
                if bundle.best_metric_name and bundle.best_metric_name in metrics_df.columns:
                    metric_values = pd.to_numeric(
                        metrics_df[bundle.best_metric_name], errors="coerce")
                    valid_values = metric_values.dropna()
                    if not valid_values.empty:
                        best_idx = valid_values.idxmin()
                        if bundle.best_metric_mode == "max":
                            best_idx = valid_values.idxmax()
                        best_row = metrics_df.loc[best_idx]
                    else:
                        best_row = metrics_df.iloc[-1]
                else:
                    best_row = metrics_df.iloc[-1]
                for metric_name in (
                    "l0",
                    "l1",
                    "l2",
                    "raw_mse",
                    "explained_variance",
                    "cosine_similarity",
                    "seq_rec_accuracy",
                    "kl_divergence",
                    "cross_entropy_increase",
                ):
                    if metric_name in best_row:
                        row[metric_name] = float(best_row[metric_name])
        rows.append(row)
    catalog = pd.DataFrame(rows).sort_values(
        "layer_index").reset_index(drop=True)
    return bundles, catalog


def set_publication_style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update(
        {
            "figure.dpi": 180,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.edgecolor": "#1f2937",
            "axes.linewidth": 1.2,
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
            "axes.titlesize": 17,
            "axes.labelsize": 14,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "font.family": "sans-serif",
            "font.sans-serif": ["Aptos", "Liberation Sans", "Noto Sans", "DejaVu Sans"],
            "font.weight": "bold",
            "legend.fontsize": 12,
            "legend.title_fontsize": 12,
            "figure.titlesize": 18,
            "grid.alpha": 0.22,
            "grid.color": "#9ca3af",
            "lines.linewidth": 2.3,
            "patch.edgecolor": "white",
            "patch.linewidth": 0.8,
            "mathtext.fontset": "stixsans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def beautify_axes(ax: plt.Axes, *, rotate_x: float | None = None) -> plt.Axes:
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="both", labelsize=12)
    if rotate_x is not None:
        ax.tick_params(axis="x", rotation=rotate_x)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
    return ax


def build_type_palette(labels: Sequence[str]) -> dict[str, str]:
    ordered = list(dict.fromkeys(labels))
    if not ordered:
        return {}
    if len(ordered) <= len(CONTRAST_COLORS):
        palette = CONTRAST_COLORS[: len(ordered)]
    else:
        extra_count = len(ordered) - len(CONTRAST_COLORS)
        fallback = sns.husl_palette(extra_count, s=0.95, l=0.42)
        palette = CONTRAST_COLORS + \
            [mcolors.to_hex(color) for color in fallback]
    return {label: color for label, color in zip(ordered, palette)}


def load_layer_model(
    bundle: LayerBundle,
    *,
    device: str = "auto",
    quantization: str | None = None,
) -> tuple[PipelineConfig, BiRNABERTEmbedder, SparseAutoEncoder, torch.Tensor | None, torch.Tensor | None]:
    resolved_device = resolve_device(device)
    cfg = PipelineConfig.from_yaml(bundle.config)
    if quantization is not None:
        cfg.model.quantization = quantization
    embedder = BiRNABERTEmbedder(cfg.model, device=resolved_device)
    sae = SparseAutoEncoder.load_checkpoint(
        bundle.weights, cfg.sae).to(resolved_device).eval()
    act_mean = None
    act_std = None
    if bundle.act_stats is not None and bundle.act_stats.exists():
        stats = torch.load(
            bundle.act_stats, map_location="cpu", weights_only=True)
        act_mean = stats["mean"].to(device=resolved_device, dtype=torch.float32)
        act_std = stats["std"].to(device=resolved_device, dtype=torch.float32).clamp(min=1e-6)
    return cfg, embedder, sae, act_mean, act_std


def safe_unload(embedder: BiRNABERTEmbedder | None = None, sae: SparseAutoEncoder | None = None) -> None:
    if embedder is not None:
        try:
            embedder.unload()
        except Exception:
            pass
    if sae is not None:
        try:
            del sae
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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


def align_tokens_to_nucleotides(
    sequence: str,
    tokenizer,
    use_bpe: bool,
    max_length: int,
) -> tuple[list[list[int]], int]:
    seq_input = sequence if use_bpe else " ".join(sequence)
    encoding = tokenizer(
        seq_input,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=False,
        return_offsets_mapping=True,
        return_special_tokens_mask=True,
    )
    n_tokens = encoding["input_ids"].shape[1]
    token_ids = encoding["input_ids"][0].tolist()
    special_tokens_mask = encoding["special_tokens_mask"][0].tolist()
    if "offset_mapping" in encoding and encoding["offset_mapping"] is not None:
        offsets = encoding["offset_mapping"][0].tolist()
        token_to_nuc: list[list[int]] = []
        for token_id, is_special, (start, end) in zip(token_ids, special_tokens_mask, offsets):
            if is_special or (start == 0 and end == 0):
                continue
            if not use_bpe:
                nuc_indices = [
                    pos // 2 for pos in range(start, end) if pos % 2 == 0]
                token_to_nuc.append(nuc_indices)
            else:
                token_to_nuc.append(list(range(start, end)))
        return token_to_nuc, n_tokens

    token_to_nuc = []
    nuc_cursor = 0
    for token_id, is_special in zip(token_ids, special_tokens_mask):
        if is_special:
            continue
        decoded = tokenizer.decode(
            [token_id], skip_special_tokens=False).strip()
        if decoded in (
            tokenizer.cls_token,
            tokenizer.sep_token,
            tokenizer.pad_token,
            tokenizer.unk_token,
            tokenizer.mask_token,
            "",
        ):
            continue
        else:
            cleaned = decoded.replace(" ", "").replace("▁", "")
            token_to_nuc.append(
                list(range(nuc_cursor, nuc_cursor + len(cleaned))))
            nuc_cursor += len(cleaned)
    return token_to_nuc, n_tokens


@torch.no_grad()
def extract_hidden_and_acts(
    sequence: str,
    embedder: BiRNABERTEmbedder,
    sae: SparseAutoEncoder,
    *,
    act_mean: torch.Tensor | None = None,
    act_std: torch.Tensor | None = None,
    device: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    resolved_device = device or embedder.device
    hidden = embedder.extract_embeddings(sequence).to(resolved_device).float()
    sae_input = _standardize(hidden, act_mean, act_std)
    acts = sae.encode(sae_input)
    return hidden, acts


def stream_bprna_dataframe(
    dataset_id: str,
    *,
    split: str = "train",
    num_sequences: int = 25_000,
) -> pd.DataFrame:
    ds = load_dataset(dataset_id, split=split, streaming=True)
    records: list[dict[str, Any]] = []
    for row in tqdm(ds, total=num_sequences, desc="Streaming bpRNA-90"):
        seq = row["sequence"].upper().replace("T", "U")
        struct_annot = row.get("structural_annotation")
        if struct_annot is None or len(struct_annot) != len(seq):
            continue
        records.append(
            {
                "id": row.get("id", f"seq_{len(records)}"),
                "sequence": seq,
                "structural_annotation": struct_annot,
                "seq_length": len(seq),
            }
        )
        if len(records) >= num_sequences:
            break
    return pd.DataFrame(records)


def _normalize_rna_sequence(sequence: str) -> str:
    return sequence.upper().replace("T", "U")


def _truncate_sequence_for_model(sequence: str, max_seq_length: int) -> str:
    return sequence[: max(1, int(max_seq_length) - 2)]


def _resolve_rnacentral_training_source(
    layer_bundles: Mapping[int, LayerBundle],
) -> dict[str, Any]:
    if not layer_bundles:
        raise ValueError("At least one layer bundle is required to resolve the training source.")

    reference: dict[str, Any] | None = None
    for layer_index, bundle in sorted(layer_bundles.items()):
        cfg = PipelineConfig.from_yaml(bundle.config)
        current = {
            "dataset_id": cfg.data.hf_dataset,
            "split": cfg.data.hf_dataset_split,
            "num_sequences": cfg.data.num_sequences,
            "max_seq_length": cfg.model.max_seq_length,
            "seed": cfg.sae.seed,
        }
        if reference is None:
            reference = current
            continue
        if current != reference:
            raise ValueError(
                "All layer bundles must share the same RNACentral training source to build a disjoint holdout. "
                f"Layer {layer_index} disagrees with the reference training configuration."
            )

    assert reference is not None
    if reference["dataset_id"] is None:
        raise ValueError("The selected layer bundles were not trained from a Hugging Face dataset.")
    if reference["num_sequences"] is None:
        raise ValueError(
            "The selected layer bundles do not record a finite training source window, so a disjoint holdout "
            "cannot be inferred automatically."
        )

    return {
        "dataset_id": str(reference["dataset_id"]),
        "split": str(reference["split"]),
        "source_window_size": int(reference["num_sequences"]),
        "max_seq_length": int(reference["max_seq_length"]),
        "seed": int(reference["seed"]),
    }


def resolve_rnacentral_training_source(
    layer_bundles: Mapping[int, LayerBundle],
) -> dict[str, Any]:
    """Return the shared RNACentral training-source metadata for a bundle set."""

    return _resolve_rnacentral_training_source(layer_bundles)


def _json_safe_payload(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe_payload(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_payload(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def rnacentral_holdout_cache_paths(
    cache_dir: str | Path,
    *,
    num_sequences: int,
    max_seq_length: int,
    min_type_count: int,
    cache_key: str | None = None,
) -> dict[str, Path]:
    root = Path(cache_dir)
    root.mkdir(parents=True, exist_ok=True)

    stem = f"rnacentral_holdout_{int(num_sequences)}_{int(max_seq_length)}_{int(min_type_count)}"
    audit_stem = (
        f"rnacentral_holdout_audit_{int(num_sequences)}_{int(max_seq_length)}_{int(min_type_count)}"
    )
    if cache_key:
        stem = f"{stem}_{cache_key}"
        audit_stem = f"{audit_stem}_{cache_key}"

    return {
        "parquet": root / f"{stem}.parquet",
        "pickle": root / f"{stem}.pkl",
        "audit": root / f"{audit_stem}.json",
        "metadata": root / f"{stem}.metadata.json",
    }


def save_cached_rnacentral_holdout(
    df_data: pd.DataFrame,
    audit: Mapping[str, Any] | None,
    cache_dir: str | Path,
    *,
    num_sequences: int,
    max_seq_length: int,
    min_type_count: int,
    cache_key: str | None = None,
    dataset_id: str | None = None,
    split: str | None = None,
    source_metadata: Mapping[str, Any] | None = None,
    write_legacy_pickle: bool = True,
) -> dict[str, Path]:
    cache_paths = rnacentral_holdout_cache_paths(
        cache_dir,
        num_sequences=num_sequences,
        max_seq_length=max_seq_length,
        min_type_count=min_type_count,
        cache_key=cache_key,
    )

    df_to_save = df_data.copy()
    if "seq_length" not in df_to_save.columns and "sequence" in df_to_save.columns:
        df_to_save["seq_length"] = df_to_save["sequence"].astype(str).str.len().astype(np.int32)

    df_to_save.to_parquet(cache_paths["parquet"], index=False)
    if write_legacy_pickle:
        df_to_save.to_pickle(cache_paths["pickle"])

    serialized_audit = _json_safe_payload(dict(audit or {}))
    cache_paths["audit"].write_text(json.dumps(serialized_audit, indent=2, sort_keys=True))

    metadata = {
        "format": "parquet",
        "cache_key": cache_key,
        "dataset_id": dataset_id,
        "split": split,
        "num_rows": int(len(df_to_save)),
        "columns": [
            {"name": str(column), "dtype": str(dtype)}
            for column, dtype in df_to_save.dtypes.items()
        ],
        "audit_file": cache_paths["audit"].name,
        "legacy_pickle_file": cache_paths["pickle"].name if write_legacy_pickle else None,
        "source_metadata": _json_safe_payload(dict(source_metadata or {})),
        "hf_upload_ready": True,
    }
    cache_paths["metadata"].write_text(json.dumps(metadata, indent=2, sort_keys=True))
    return cache_paths


def load_cached_rnacentral_holdout(
    cache_dir: str | Path,
    *,
    num_sequences: int,
    max_seq_length: int,
    min_type_count: int,
    cache_key: str | None = None,
    migrate_legacy_pickle: bool = False,
    write_legacy_pickle: bool = True,
    dataset_id: str | None = None,
    split: str | None = None,
    source_metadata: Mapping[str, Any] | None = None,
) -> tuple[pd.DataFrame | None, dict[str, Any], dict[str, Path]]:
    cache_paths = rnacentral_holdout_cache_paths(
        cache_dir,
        num_sequences=num_sequences,
        max_seq_length=max_seq_length,
        min_type_count=min_type_count,
        cache_key=cache_key,
    )
    audit = json.loads(cache_paths["audit"].read_text()) if cache_paths["audit"].exists() else {}

    if cache_paths["parquet"].exists():
        return pd.read_parquet(cache_paths["parquet"]), audit, cache_paths

    if not cache_paths["pickle"].exists():
        return None, audit, cache_paths

    df_data = pd.read_pickle(cache_paths["pickle"])
    if migrate_legacy_pickle:
        cache_paths = save_cached_rnacentral_holdout(
            df_data,
            audit,
            cache_dir,
            num_sequences=num_sequences,
            max_seq_length=max_seq_length,
            min_type_count=min_type_count,
            cache_key=cache_key,
            dataset_id=dataset_id,
            split=split,
            source_metadata=source_metadata,
            write_legacy_pickle=write_legacy_pickle,
        )
    return df_data, audit, cache_paths


def load_rnacentral_release_dataframe(
    dataset_id: str,
    *,
    split: str = "train",
    max_seq_length: int | None = None,
) -> pd.DataFrame:
    ds = load_dataset(dataset_id, split=split)
    df_data = ds.to_pandas() if hasattr(ds, "to_pandas") else pd.DataFrame(ds)
    if df_data.empty:
        return pd.DataFrame(columns=["upi", "sequence", "rna_type", "seq_length"])

    rename_map: dict[str, str] = {}
    if "type" in df_data.columns and "rna_type" not in df_data.columns:
        rename_map["type"] = "rna_type"
    df_data = df_data.rename(columns=rename_map).copy()

    if "sequence" not in df_data.columns:
        raise ValueError(f"Dataset {dataset_id} split {split} does not contain a sequence column.")

    df_data["sequence"] = df_data["sequence"].astype(str).map(_normalize_rna_sequence)
    if max_seq_length is not None:
        truncated_length = max(1, int(max_seq_length) - 2)
        df_data["sequence"] = df_data["sequence"].str.slice(0, truncated_length)

    if "upi" not in df_data.columns:
        df_data["upi"] = [f"seq_{index}" for index in range(len(df_data))]
    if "rna_type" not in df_data.columns:
        df_data["rna_type"] = "unknown"

    df_data["seq_length"] = df_data["sequence"].str.len().astype(np.int32)
    keep_columns = [column for column in ["upi", "sequence", "rna_type", "seq_length"] if column in df_data.columns]
    return df_data.loc[:, keep_columns].reset_index(drop=True)


def _write_fasta_record(handle, sequence_id: str, sequence: str, *, line_width: int = 80) -> None:
    handle.write(f">{sequence_id}\n")
    for start in range(0, len(sequence), line_width):
        handle.write(f"{sequence[start:start + line_width]}\n")


def _read_fasta_ids(path: Path) -> set[str]:
    sequence_ids: set[str] = set()
    with path.open() as handle:
        for line in handle:
            if line.startswith(">"):
                sequence_ids.add(line[1:].strip().split()[0])
    return sequence_ids


def stream_cd_hit_2d_rnacentral_holdout_dataframe(
    dataset_id: str,
    *,
    layer_bundles: Mapping[int, LayerBundle],
    split: str = "train",
    num_sequences: int = 25_000,
    max_seq_length: int = 512,
    min_length: int = 10,
    candidate_multiplier: int = 8,
    similarity_threshold: float = 0.40,
    cd_hit_executable: str | None = None,
    word_size: int = 2,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Build a holdout by comparing candidate sequences against the training source
    with cd-hit-2d. CD-HIT-2D is the only sequence-similarity filter here.

    ``-i`` is the training database and ``-i2`` is the candidate database;
    CD-HIT-2D's output contains candidates that do not meet the threshold against
    any training sequence.
    """

    from itertools import islice

    if not 0.0 < float(similarity_threshold) <= 1.0:
        raise ValueError("similarity_threshold must be in the interval (0, 1].")
    if int(word_size) < 1:
        raise ValueError("word_size must be positive.")

    executable = cd_hit_executable or shutil.which("cd-hit-2d")
    if executable is None:
        raise FileNotFoundError(
            "cd-hit-2d was not found on PATH. Install CD-HIT or set SPIRAL_CD_HIT_2D."
        )

    training_source = _resolve_rnacentral_training_source(layer_bundles)
    effective_max_seq_length = min(int(max_seq_length), int(training_source["max_seq_length"]))
    candidate_window = max(int(num_sequences), int(num_sequences * max(1, candidate_multiplier)))

    training_ds = load_dataset(
        training_source["dataset_id"],
        split=training_source["split"],
        streaming=True,
    )
    training_stream = islice(training_ds, training_source["source_window_size"])
    same_stream_as_training = (
        dataset_id == training_source["dataset_id"] and split == training_source["split"]
    )
    candidate_start = training_source["source_window_size"] if same_stream_as_training else 0
    candidate_stop = candidate_start + candidate_window

    candidate_records: dict[str, dict[str, Any]] = {}
    training_sequences_written = 0
    short_training_rejections = 0
    short_candidate_rejections = 0
    candidate_rows_scanned = 0

    with tempfile.TemporaryDirectory(prefix="spiral_cd_hit_2d_") as temporary_dir:
        temporary_root = Path(temporary_dir)
        training_fasta = temporary_root / "training.fasta"
        candidate_fasta = temporary_root / "candidate.fasta"
        filtered_fasta = temporary_root / "candidate_cd_hit_2d.fasta"

        with training_fasta.open("w") as handle:
            for row in tqdm(
                training_stream,
                total=training_source["source_window_size"],
                desc="Writing CD-HIT-2D training database",
            ):
                sequence = _truncate_sequence_for_model(
                    _normalize_rna_sequence(str(row["sequence"])),
                    effective_max_seq_length,
                )
                if len(sequence) < min_length:
                    short_training_rejections += 1
                    continue
                _write_fasta_record(handle, f"training_{training_sequences_written:09d}", sequence)
                training_sequences_written += 1

        sample_ds = load_dataset(dataset_id, split=split, streaming=True)
        sample_stream = islice(sample_ds, candidate_start, candidate_stop)
        with candidate_fasta.open("w") as handle:
            for index, row in enumerate(
                tqdm(sample_stream, total=candidate_window, desc="Writing CD-HIT-2D candidate database")
            ):
                candidate_rows_scanned += 1
                sequence = _truncate_sequence_for_model(
                    _normalize_rna_sequence(str(row["sequence"])),
                    effective_max_seq_length,
                )
                if len(sequence) < min_length:
                    short_candidate_rejections += 1
                    continue
                sequence_id = f"candidate_{index:09d}"
                candidate_records[sequence_id] = {
                    "upi": row.get("upi", f"holdout_{candidate_start + index}"),
                    "sequence": sequence,
                    "rna_type": row.get("type") or row.get("rna_type") or "unknown",
                    "seq_length": len(sequence),
                }
                _write_fasta_record(handle, sequence_id, sequence)

        command = [
            str(executable),
            "-i", str(training_fasta),
            "-i2", str(candidate_fasta),
            "-o", str(filtered_fasta),
            "-c", str(float(similarity_threshold)),
            "-n", str(int(word_size)),
            "-G", "1",
            "-M", "0",
            "-T", "0",
            "-d", "0",
        ]
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode != 0:
            diagnostic = "\n".join(
                part for part in (completed.stdout, completed.stderr) if part
            )
            raise RuntimeError(
                "cd-hit-2d failed with exit code "
                f"{completed.returncode}: {diagnostic[-4000:]}"
            )
        if not filtered_fasta.exists():
            raise RuntimeError("cd-hit-2d completed without producing its output FASTA.")

        retained_ids = _read_fasta_ids(filtered_fasta)

    records = [
        candidate_records[sequence_id]
        for sequence_id in candidate_records
        if sequence_id in retained_ids
    ][: int(num_sequences)]
    audit = {
        "holdout_policy": "cd_hit_2d_40pct_v1",
        "similarity_filter": "cd-hit-2d",
        "similarity_threshold": float(similarity_threshold),
        "cd_hit_executable": str(executable),
        "cd_hit_word_size": int(word_size),
        "cd_hit_global_identity": True,
        "training_dataset_id": training_source["dataset_id"],
        "training_split": training_source["split"],
        "sample_dataset_id": dataset_id,
        "sample_split": split,
        "same_stream_as_training": same_stream_as_training,
        "training_source_rows_skipped": int(candidate_start),
        "requested_sequences": int(num_sequences),
        "returned_sequences": int(len(records)),
        "candidate_rows_scanned": int(candidate_rows_scanned),
        "candidate_window": int(candidate_window),
        "training_sequences_written": int(training_sequences_written),
        "candidate_sequences_written": int(len(candidate_records)),
        "candidate_sequences_retained_by_cd_hit_2d": int(len(retained_ids)),
        "effective_max_seq_length": int(effective_max_seq_length),
        "min_length": int(min_length),
        "candidate_multiplier": int(candidate_multiplier),
        "short_training_rejections": int(short_training_rejections),
        "short_candidate_rejections": int(short_candidate_rejections),
        "cd_hit_command": command,
    }
    return pd.DataFrame.from_records(records), audit


def stream_rnacentral_dataframe(
    dataset_id: str,
    *,
    split: str = "train",
    num_sequences: int = 25_000,
    min_length: int = 10,
    oversample_factor: int = 3,
) -> pd.DataFrame:
    from itertools import islice

    ds = load_dataset(dataset_id, split=split, streaming=True)
    stream = islice(ds, num_sequences * oversample_factor)
    records: list[dict[str, Any]] = []
    for index, row in enumerate(
        tqdm(stream, total=num_sequences *
             oversample_factor, desc="Streaming RNAcentral")
    ):
        seq = _normalize_rna_sequence(str(row["sequence"]))
        if len(seq) < min_length:
            continue
        records.append(
            {
                "upi": row.get("upi", f"seq_{index}"),
                "sequence": seq,
                "rna_type": row.get("type") or "unknown",
                "seq_length": len(seq),
            }
        )
        if len(records) >= num_sequences:
            break
    return pd.DataFrame.from_records(records)


@torch.no_grad()
def collect_structure_alignment(
    df_data: pd.DataFrame,
    embedder: BiRNABERTEmbedder,
    sae: SparseAutoEncoder,
    *,
    act_mean: torch.Tensor | None,
    act_std: torch.Tensor | None,
    max_seq_length: int,
    reservoir_size: int,
    activation_thresholds: Sequence[float],
    structure_label_map: Mapping[str, str] | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    structure_label_map = structure_label_map or STRUCTURE_LABELS
    structure_chars = sorted(structure_label_map.keys())
    struct_to_idx = {label: index for index,
                     label in enumerate(structure_chars)}
    n_structs = len(structure_chars)
    d_hidden = sae.d_hidden
    reservoir_cap = int(reservoir_size)
    rng = np.random.default_rng(seed)

    feature_struct_counts = np.zeros((d_hidden, n_structs), dtype=np.int64)
    feature_seq_sets = [set() for _ in range(d_hidden)]
    global_struct_counts = np.zeros(n_structs, dtype=np.int64)
    threshold_feature_struct_counts = [
        np.zeros((d_hidden, n_structs), dtype=np.int64) for _ in activation_thresholds
    ]

    reservoir_sae = np.empty((reservoir_cap, d_hidden), dtype=np.float16)
    reservoir_hidden = np.empty(
        (reservoir_cap, embedder.hidden_dim), dtype=np.float16)
    reservoir_labels = np.empty(reservoir_cap, dtype=np.int8)

    tokenizer = embedder.tokenizer
    use_bpe = embedder.cfg.use_bpe
    reservoir_fill = 0
    total_nucleotides = 0
    n_processed = 0
    n_skipped = 0

    for row_index in tqdm(range(len(df_data)), desc="Extracting structure alignment"):
        row = df_data.iloc[row_index]
        sequence = row["sequence"][: max_seq_length - 2]
        annotation = row["structural_annotation"][: len(sequence)]
        try:
            token_to_nuc, _ = align_tokens_to_nucleotides(
                sequence,
                tokenizer,
                use_bpe,
                max_seq_length,
            )
            hidden, acts = extract_hidden_and_acts(
                sequence,
                embedder,
                sae,
                act_mean=act_mean,
                act_std=act_std,
            )
            if acts.shape[0] != len(token_to_nuc):
                n_skipped += 1
                continue

            acts_np = acts.cpu().numpy()
            hidden_np = hidden.cpu().numpy()

            for token_idx, nuc_indices in enumerate(token_to_nuc):
                if not nuc_indices:
                    continue
                act_vec = acts_np[token_idx]
                hidden_vec = hidden_np[token_idx]
                active_features = np.where(act_vec > 0)[0]
                threshold_masks = [
                    act_vec > threshold for threshold in activation_thresholds]

                for nuc_idx in nuc_indices:
                    if nuc_idx >= len(annotation):
                        continue
                    label = annotation[nuc_idx]
                    if label not in struct_to_idx:
                        continue
                    struct_idx = struct_to_idx[label]
                    global_struct_counts[struct_idx] += 1

                    if active_features.size:
                        feature_struct_counts[active_features, struct_idx] += 1

                    for threshold_idx, threshold_mask in enumerate(threshold_masks):
                        threshold_feature_struct_counts[threshold_idx][threshold_mask,
                                                                       struct_idx] += 1

                    total_nucleotides += 1
                    if reservoir_fill < reservoir_cap:
                        reservoir_sae[reservoir_fill] = act_vec
                        reservoir_hidden[reservoir_fill] = hidden_vec
                        reservoir_labels[reservoir_fill] = struct_idx
                        reservoir_fill += 1
                    else:
                        replace_index = rng.integers(0, total_nucleotides)
                        if replace_index < reservoir_cap:
                            reservoir_sae[replace_index] = act_vec
                            reservoir_hidden[replace_index] = hidden_vec
                            reservoir_labels[replace_index] = struct_idx

                for feature_idx in active_features:
                    feature_seq_sets[feature_idx].add(row_index)

            del hidden, acts
            n_processed += 1
        except Exception:
            n_skipped += 1

        if (row_index + 1) % 250 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    feature_seq_counts = np.array(
        [len(seq_set) for seq_set in feature_seq_sets], dtype=np.int64)
    return {
        "structure_chars": structure_chars,
        "feature_struct_counts": feature_struct_counts,
        "global_struct_counts": global_struct_counts,
        "feature_seq_counts": feature_seq_counts,
        "threshold_feature_struct_counts": threshold_feature_struct_counts,
        "threshold_values": np.array(activation_thresholds, dtype=np.float32),
        "reservoir_sae": reservoir_sae[:reservoir_fill],
        "reservoir_hidden": reservoir_hidden[:reservoir_fill],
        "reservoir_labels": reservoir_labels[:reservoir_fill],
        "n_processed": n_processed,
        "n_skipped": n_skipped,
        "total_nucleotides": total_nucleotides,
    }


def compute_structure_feature_statistics(
    feature_struct_counts: np.ndarray,
    global_struct_counts: np.ndarray,
    feature_seq_counts: np.ndarray,
    *,
    structure_chars: Sequence[str],
    structure_name_map: Mapping[str, str] | None = None,
    min_sequence_count: int = 5,
    significance_alpha: float = 0.01,
) -> pd.DataFrame:
    structure_name_map = structure_name_map or STRUCTURE_LABELS
    qualifying_mask = feature_seq_counts >= min_sequence_count
    qualifying_features = np.where(qualifying_mask)[0]
    bg_total = global_struct_counts.sum()
    bg_probs = global_struct_counts / \
        bg_total if bg_total > 0 else np.ones(
            len(structure_chars)) / len(structure_chars)
    results: list[dict[str, Any]] = []

    for feature_idx in tqdm(qualifying_features, desc="Structure statistics"):
        counts = feature_struct_counts[feature_idx]
        total = counts.sum()
        if total == 0:
            continue
        probs = counts / total
        preferred_idx = int(probs.argmax())
        preferred_structure = structure_chars[preferred_idx]
        selectivity = float(probs[preferred_idx])
        enrichment = (
            float(probs[preferred_idx] / bg_probs[preferred_idx])
            if bg_probs[preferred_idx] > 0
            else 0.0
        )
        expected = bg_probs * total
        valid = expected > 0
        if valid.sum() < 2:
            continue
        chi2, p_value = sp_stats.chisquare(
            counts[valid], f_exp=expected[valid])
        results.append(
            {
                "feature_idx": int(feature_idx),
                "n_sequences": int(feature_seq_counts[feature_idx]),
                "n_activations": int(total),
                "selectivity": selectivity,
                "preferred_structure": preferred_structure,
                "preferred_name": structure_name_map.get(preferred_structure, preferred_structure),
                "enrichment_ratio": enrichment,
                "chi2": float(chi2),
                "p_value": float(p_value),
                **{f"frac_{label}": float(probs[idx]) for idx, label in enumerate(structure_chars)},
            }
        )

    df_results = pd.DataFrame(results)
    if df_results.empty:
        return pd.DataFrame(
            columns=[
                "feature_idx",
                "n_sequences",
                "n_activations",
                "selectivity",
                "preferred_structure",
                "preferred_name",
                "enrichment_ratio",
                "chi2",
                "p_value",
                "p_bonferroni",
                "significant",
                "significant_bonf",
            ]
        )

    n_tests = len(df_results)
    df_results["p_bonferroni"] = np.minimum(
        df_results["p_value"] * n_tests, 1.0)
    df_results["significant"] = df_results["p_value"] < significance_alpha
    df_results["significant_bonf"] = df_results["p_bonferroni"] < significance_alpha
    return df_results.sort_values(
        ["selectivity", "enrichment_ratio", "n_activations"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


@torch.no_grad()
def extract_family_profiles(
    df_data: pd.DataFrame,
    embedder: BiRNABERTEmbedder,
    sae: SparseAutoEncoder,
    *,
    act_mean: torch.Tensor | None,
    act_std: torch.Tensor | None,
    active_threshold: float,
    max_seq_length: int,
) -> dict[str, Any]:
    n_sequences = len(df_data)
    d_hidden = sae.d_hidden
    mean_act_matrix = np.zeros((n_sequences, d_hidden), dtype=np.float16)
    frac_act_matrix = np.zeros((n_sequences, d_hidden), dtype=np.float16)
    max_act_matrix = np.zeros((n_sequences, d_hidden), dtype=np.float16)
    bert_mean_matrix = np.zeros(
        (n_sequences, embedder.hidden_dim), dtype=np.float16)
    valid_mask = np.ones(n_sequences, dtype=bool)
    sequence_types: list[str | None] = []
    n_processed = 0
    n_skipped = 0

    for row_index in tqdm(range(n_sequences), desc="Extracting family profiles"):
        row = df_data.iloc[row_index]
        sequence = row["sequence"][: max_seq_length - 2]
        try:
            hidden, acts = extract_hidden_and_acts(
                sequence,
                embedder,
                sae,
                act_mean=act_mean,
                act_std=act_std,
            )
            mean_act_matrix[row_index] = acts.mean(dim=0).cpu().numpy()
            frac_act_matrix[row_index] = (
                acts > active_threshold).float().mean(dim=0).cpu().numpy()
            max_act_matrix[row_index] = acts.max(dim=0).values.cpu().numpy()
            bert_mean_matrix[row_index] = hidden.mean(dim=0).cpu().numpy()
            sequence_types.append(row["rna_type"])
            del hidden, acts
            n_processed += 1
        except Exception:
            valid_mask[row_index] = False
            sequence_types.append(None)
            n_skipped += 1

        if (row_index + 1) % 250 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    valid_indices = np.where(valid_mask)[0]
    return {
        "mean_act_matrix": mean_act_matrix[valid_mask].astype(np.float32),
        "frac_act_matrix": frac_act_matrix[valid_mask].astype(np.float32),
        "max_act_matrix": max_act_matrix[valid_mask].astype(np.float32),
        "bert_mean_matrix": bert_mean_matrix[valid_mask].astype(np.float32),
        "valid_indices": valid_indices,
        "seq_types": np.array([seq_type for seq_type in sequence_types if seq_type is not None]),
        "n_processed": n_processed,
        "n_skipped": n_skipped,
    }


def compute_family_feature_statistics(
    mean_act_matrix: np.ndarray,
    seq_types: Sequence[str],
    *,
    active_threshold: float,
    min_feature_seqs: int,
    significance_alpha: float = 0.01,
    nuisance: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Per-feature RNA-type association statistics.

    When `nuisance` is supplied (we pass sequence length), the Kruskal-Wallis
    test is run on the feature's rank profile after the linear rank-dependence on
    the nuisance variable has been removed. Because the sequence profile is a
    mean over tokens, a feature with bounded activation is diluted in proportion
    to sequence length, and RNA types differ systematically in length; without
    this adjustment a feature that merely tracks "short sequence" scores a large
    effect size. Descriptive quantities (type means, fold enrichment,
    selectivity, preferred type) are always computed on the raw activations,
    since the residuals are not interpretable as activation magnitudes. The
    uncontrolled effect size is retained as `eta_squared_uncontrolled` for
    comparison.
    """
    types_arr = np.array(seq_types)
    unique_types = sorted(set(seq_types))
    type_indices = {rna_type: np.where(types_arr == rna_type)[
        0] for rna_type in unique_types}
    n_types = len(unique_types)
    results: list[dict[str, Any]] = []

    control_nuisance = nuisance is not None
    if control_nuisance:
        nuisance_ranks = sp_stats.rankdata(np.asarray(nuisance, dtype=float))
        nuisance_centred = nuisance_ranks - nuisance_ranks.mean()
        nuisance_denom = float((nuisance_centred ** 2).sum())

    def _kruskal_eta(values: np.ndarray) -> tuple[float, float, float]:
        groups_local = [values[type_indices[rna_type]]
                        for rna_type in unique_types]
        h_local, p_local = sp_stats.kruskal(*groups_local)
        n_samples_local = len(values)
        eta_local = (
            max(0.0, (float(h_local) - n_types + 1) / (n_samples_local - n_types))
            if n_samples_local > n_types
            else 0.0
        )
        return float(h_local), float(p_local), float(eta_local)

    for feature_idx in tqdm(range(mean_act_matrix.shape[1]), desc="Family statistics"):
        feat_vals = mean_act_matrix[:, feature_idx]
        n_active = int(np.sum(feat_vals > active_threshold))
        if n_active < min_feature_seqs:
            continue

        n_samples = len(feat_vals)
        group_means = {rna_type: float(
            feat_vals[type_indices[rna_type]].mean()) for rna_type in unique_types}
        global_mean = float(feat_vals.mean())

        try:
            h_raw, p_raw, eta_raw = _kruskal_eta(feat_vals)
            if control_nuisance:
                feat_ranks = sp_stats.rankdata(feat_vals)
                feat_centred = feat_ranks - feat_ranks.mean()
                beta = float(feat_centred @ nuisance_centred) / nuisance_denom
                residuals = feat_centred - beta * nuisance_centred
                h_stat, p_value, eta_sq = _kruskal_eta(residuals)
            else:
                h_stat, p_value, eta_sq = h_raw, p_raw, eta_raw
        except ValueError:
            continue

        preferred_type = max(group_means, key=group_means.get)
        preferred_mean = group_means[preferred_type]
        fold_enrichment = preferred_mean / global_mean if global_mean > 0 else 0.0
        total_mean = sum(group_means.values())
        selectivity = preferred_mean / total_mean if total_mean > 0 else 0.0

        active_per_type = np.array(
            [(feat_vals[type_indices[rna_type]] > active_threshold).sum()
             for rna_type in unique_types]
        )
        total_per_type = np.array([len(type_indices[rna_type])
                                  for rna_type in unique_types])
        overall_rate = float(
            (feat_vals > active_threshold).sum() / max(n_samples, 1))
        expected_active = total_per_type * overall_rate
        valid_bins = expected_active > 0
        if valid_bins.sum() >= 2:
            chi2_stat, chi2_p = sp_stats.chisquare(
                active_per_type[valid_bins],
                f_exp=expected_active[valid_bins],
            )
        else:
            chi2_stat, chi2_p = 0.0, 1.0

        results.append(
            {
                "feature_idx": int(feature_idx),
                "n_active_seqs": n_active,
                "global_mean_act": global_mean,
                "preferred_type": preferred_type,
                "preferred_mean": preferred_mean,
                "fold_enrichment": float(fold_enrichment),
                "selectivity": float(selectivity),
                "eta_squared": float(eta_sq),
                "eta_squared_uncontrolled": float(eta_raw),
                "h_statistic": float(h_stat),
                "p_value": float(p_value),
                "chi2_stat": float(chi2_stat),
                "chi2_p": float(chi2_p),
                **{f"mean_{rna_type}": float(group_means[rna_type]) for rna_type in unique_types},
            }
        )

    df_results = pd.DataFrame(results)
    if df_results.empty:
        return pd.DataFrame(
            columns=[
                "feature_idx",
                "n_active_seqs",
                "global_mean_act",
                "preferred_type",
                "preferred_mean",
                "fold_enrichment",
                "selectivity",
                "eta_squared",
                "eta_squared_uncontrolled",
                "h_statistic",
                "p_value",
                "chi2_stat",
                "chi2_p",
                "p_bonferroni",
                "significant",
                "significant_bonf",
                "chi2_p_bonf",
            ]
        )

    n_tests = len(df_results)
    df_results["p_bonferroni"] = np.minimum(
        df_results["p_value"] * n_tests, 1.0)
    df_results["significant"] = df_results["p_value"] < significance_alpha
    df_results["significant_bonf"] = df_results["p_bonferroni"] < significance_alpha
    df_results["chi2_p_bonf"] = np.minimum(df_results["chi2_p"] * n_tests, 1.0)
    return df_results.sort_values(
        ["eta_squared", "fold_enrichment", "selectivity"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def compute_group_centroids(
    profile_matrix: np.ndarray,
    labels: Sequence[str],
) -> dict[str, np.ndarray]:
    labels_arr = np.array(labels)
    centroids: dict[str, np.ndarray] = {}
    for label in sorted(set(labels)):
        mask = labels_arr == label
        centroids[label] = profile_matrix[mask].mean(axis=0)
    return centroids


def cosine_similarity_to_centroids(
    profile: np.ndarray,
    centroids: Mapping[str, np.ndarray],
) -> pd.Series:
    profile = np.asarray(profile, dtype=np.float64)
    profile_norm = np.linalg.norm(profile) + 1e-12
    similarities = {}
    for label, centroid in centroids.items():
        centroid = np.asarray(centroid, dtype=np.float64)
        similarities[label] = float(
            np.dot(profile, centroid) / (profile_norm * (np.linalg.norm(centroid) + 1e-12)))
    return pd.Series(similarities).sort_values(ascending=False)


def decode_token_ids_to_rna(tokenizer, token_ids: Sequence[int]) -> str:
    decoded = tokenizer.decode(list(token_ids), skip_special_tokens=True)
    return decoded.replace(" ", "").replace("▁", "").upper().replace("T", "U")


def _replace_hidden_with_reconstruction(
    output: torch.Tensor | tuple,
    non_special_mask: torch.Tensor,
    reconstruction: torch.Tensor,
):
    hidden = output[0] if isinstance(output, tuple) else output
    mask = non_special_mask.to(hidden.device)
    updated = hidden.clone()
    if hidden.dim() == 3:
        updated[:, mask, :] = reconstruction.reshape(
            1, -1, hidden.shape[-1]).to(hidden.dtype)
    else:
        updated[mask] = reconstruction.to(hidden.dtype)
    if isinstance(output, tuple):
        return (updated,) + output[1:]
    return updated


@torch.no_grad()
def run_sae_feature_steering(
    sequence: str,
    embedder: BiRNABERTEmbedder,
    sae: SparseAutoEncoder,
    *,
    feature_deltas: Mapping[int, float],
    act_mean: torch.Tensor | None = None,
    act_std: torch.Tensor | None = None,
    clamp_nonnegative: bool = True,
) -> dict[str, Any]:
    device = embedder.device
    inputs = embedder.tokenize(sequence)
    valid_mask_cpu = inputs["attention_mask"][0].bool(
    ) & ~inputs["special_tokens_mask"][0].bool()
    if valid_mask_cpu.sum().item() == 0:
        raise ValueError("Sequence produced no non-special tokens.")

    model_inputs = {key: value.to(
        device) for key, value in inputs.items() if key != "special_tokens_mask"}
    base_outputs = embedder.model(**model_inputs)
    base_logits = base_outputs.logits.detach()
    if base_logits.dim() == 3:
        base_logits = base_logits[0]

    hook_layer_name = embedder._hook_layer_name
    hidden = embedder._intermediate[hook_layer_name].detach()
    if hidden.dim() == 3:
        hidden = hidden[0]
    hidden_valid = hidden[valid_mask_cpu.to(hidden.device)]

    sae_input = _standardize(hidden_valid.float(), act_mean, act_std)
    acts_before = sae.encode(sae_input)
    acts_after = acts_before.clone()
    for feature_idx, delta in feature_deltas.items():
        acts_after[:, int(feature_idx)] = acts_after[:,
                                                     int(feature_idx)] + float(delta)
    if clamp_nonnegative:
        acts_after = acts_after.clamp_min(0.0)
    reconstruction = _unstandardize(sae.decode(
        acts_after).float(), act_mean, act_std)

    target_module = None
    for module_name, module in embedder.model.named_modules():
        if module_name == hook_layer_name:
            target_module = module
            break
    if target_module is None:
        raise ValueError(
            f"Could not find module '{hook_layer_name}' for steering.")

    def hook_fn(module, input, output):
        return _replace_hidden_with_reconstruction(output, valid_mask_cpu, reconstruction)

    handle = target_module.register_forward_hook(hook_fn)
    steered_outputs = embedder.model(**model_inputs)
    handle.remove()

    steered_logits = steered_outputs.logits.detach()
    if steered_logits.dim() == 3:
        steered_logits = steered_logits[0]

    valid_mask = valid_mask_cpu.to(base_logits.device)
    base_logits_valid = base_logits[valid_mask]
    steered_logits_valid = steered_logits[valid_mask]
    base_token_ids = base_logits_valid.argmax(dim=-1).cpu().tolist()
    steered_token_ids = steered_logits_valid.argmax(dim=-1).cpu().tolist()
    token_to_nuc, _ = align_tokens_to_nucleotides(
        sequence,
        embedder.tokenizer,
        embedder.cfg.use_bpe,
        embedder.cfg.max_seq_length,
    )
    token_to_nuc = [indices for indices in token_to_nuc if indices]

    return {
        "input_sequence": sequence,
        "base_decoded_sequence": decode_token_ids_to_rna(embedder.tokenizer, base_token_ids),
        "steered_decoded_sequence": decode_token_ids_to_rna(embedder.tokenizer, steered_token_ids),
        "base_token_ids": base_token_ids,
        "steered_token_ids": steered_token_ids,
        "base_tokens": embedder.tokenizer.convert_ids_to_tokens(base_token_ids),
        "steered_tokens": embedder.tokenizer.convert_ids_to_tokens(steered_token_ids),
        "base_logits": base_logits_valid.float().cpu().numpy(),
        "steered_logits": steered_logits_valid.float().cpu().numpy(),
        "acts_before": acts_before.float().cpu().numpy(),
        "acts_after": acts_after.float().cpu().numpy(),
        "token_to_nuc": token_to_nuc,
        "feature_deltas": {int(key): float(value) for key, value in feature_deltas.items()},
    }


@torch.no_grad()
def profile_single_sequence(
    sequence: str,
    embedder: BiRNABERTEmbedder,
    sae: SparseAutoEncoder,
    *,
    act_mean: torch.Tensor | None = None,
    act_std: torch.Tensor | None = None,
    active_threshold: float = 0.05,
) -> dict[str, np.ndarray]:
    hidden, acts = extract_hidden_and_acts(
        sequence,
        embedder,
        sae,
        act_mean=act_mean,
        act_std=act_std,
    )
    return {
        "mean_act": acts.mean(dim=0).float().cpu().numpy(),
        "frac_active": (acts > active_threshold).float().mean(dim=0).cpu().numpy(),
        "max_act": acts.max(dim=0).values.float().cpu().numpy(),
        "bert_mean": hidden.mean(dim=0).float().cpu().numpy(),
    }


def global_align_sequences(
    source: str,
    target: str,
    *,
    match_score: int = 2,
    mismatch_penalty: int = -1,
    gap_penalty: int = -2,
) -> tuple[str, str]:
    n_rows = len(source) + 1
    n_cols = len(target) + 1
    scores = np.zeros((n_rows, n_cols), dtype=np.int32)
    pointers = np.zeros((n_rows, n_cols), dtype=np.int8)

    for row in range(1, n_rows):
        scores[row, 0] = row * gap_penalty
        pointers[row, 0] = 1
    for col in range(1, n_cols):
        scores[0, col] = col * gap_penalty
        pointers[0, col] = 2

    for row in range(1, n_rows):
        for col in range(1, n_cols):
            diag = scores[row - 1, col - 1] + \
                (match_score if source[row - 1] ==
                 target[col - 1] else mismatch_penalty)
            up = scores[row - 1, col] + gap_penalty
            left = scores[row, col - 1] + gap_penalty
            best = max(diag, up, left)
            scores[row, col] = best
            if best == diag:
                pointers[row, col] = 0
            elif best == up:
                pointers[row, col] = 1
            else:
                pointers[row, col] = 2

    aligned_source: list[str] = []
    aligned_target: list[str] = []
    row = len(source)
    col = len(target)
    while row > 0 or col > 0:
        pointer = pointers[row, col]
        if row > 0 and col > 0 and pointer == 0:
            aligned_source.append(source[row - 1])
            aligned_target.append(target[col - 1])
            row -= 1
            col -= 1
        elif row > 0 and (col == 0 or pointer == 1):
            aligned_source.append(source[row - 1])
            aligned_target.append("-")
            row -= 1
        else:
            aligned_source.append("-")
            aligned_target.append(target[col - 1])
            col -= 1
    return "".join(reversed(aligned_source)), "".join(reversed(aligned_target))


def build_sequence_alignment_frame(source: str, target: str) -> pd.DataFrame:
    aligned_source, aligned_target = global_align_sequences(source, target)
    rows: list[dict[str, Any]] = []
    for index, (src_base, tgt_base) in enumerate(zip(aligned_source, aligned_target), start=1):
        if src_base == tgt_base:
            change_type = "match"
        elif src_base == "-":
            change_type = "insert"
        elif tgt_base == "-":
            change_type = "delete"
        else:
            change_type = "replace"
        rows.append(
            {
                "alignment_position": index,
                "source_base": src_base,
                "target_base": tgt_base,
                "change_type": change_type,
            }
        )
    return pd.DataFrame(rows)


def summarize_structure_distribution(df_data: pd.DataFrame) -> pd.DataFrame:
    all_labels = "".join(df_data["structural_annotation"].tolist())
    counts = Counter(all_labels)
    total = sum(counts.values())
    rows = []
    for label, count in sorted(counts.items(), key=lambda item: item[1], reverse=True):
        rows.append(
            {
                "structure": label,
                "structure_name": STRUCTURE_LABELS.get(label, label),
                "count": count,
                "fraction": count / max(total, 1),
            }
        )
    return pd.DataFrame(rows)


def summarize_type_distribution(df_data: pd.DataFrame) -> pd.DataFrame:
    counts = df_data["rna_type"].value_counts()
    rows = [
        {
            "rna_type": rna_type,
            "count": int(count),
            "fraction": float(count / max(len(df_data), 1)),
        }
        for rna_type, count in counts.items()
    ]
    return pd.DataFrame(rows)
