# SPIRAL

[![Preprint](https://img.shields.io/badge/bioRxiv-10.64898%2F2026.08.11.744228-b31b1b)](https://doi.org/10.64898/2026.08.11.744228)
[![Zenodo](https://img.shields.io/badge/Zenodo-10.5281%2Fzenodo.21891845-1682D4)](https://doi.org/10.5281/zenodo.21891845)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB)](https://www.python.org/)

**SPIRAL** (Sparse Autoencoders for Interpretable RNA Analysis) is an
interpretability analysis of the hidden representations learned by
[BiRNA-BERT](https://huggingface.co/buetnlpbio/birna-bert). It trains
independent sparse autoencoders (SAEs) at transformer layers 0, 5, and 11, then
measures their reconstruction fidelity and their alignment with RNA
secondary-structure annotations (bpRNA-90) and RNA-type labels (RNAcentral).

This repository is the research code for the manuscript **"Sparse Autoencoders
Reveal Structural and Family-level Features in BiRNA-BERT."**

| Resource | Link |
| --- | --- |
| Preprint | [10.64898/2026.08.11.744228](https://doi.org/10.64898/2026.08.11.744228) (bioRxiv) |
| Reproducibility archive (code + data + SAE checkpoints) | [10.5281/zenodo.21891845](https://doi.org/10.5281/zenodo.21891845) |
| Base language model | [`buetnlpbio/birna-bert`](https://huggingface.co/buetnlpbio/birna-bert) |
| Datasets | [`multimolecule/rnacentral`](https://huggingface.co/datasets/multimolecule/rnacentral), [`multimolecule/bprna-90`](https://huggingface.co/datasets/multimolecule/bprna-90) |

## Repository layout

```text
config.yaml                   Paper training configuration
train.py                      SAE training entry point
generate_family_holdout.py    Builds the similarity-filtered RNAcentral holdout
requirements.txt              Recorded Python dependencies
src/                          Training, evaluation, and analysis library
  sae.py                        Sparse autoencoder model
  trainer.py                    Training loop, checkpointing, evaluation
  dataset.py                    RNAcentral / bpRNA streaming and tokenisation
  model.py                      BiRNA-BERT wrapper and hidden-state extraction
  evaluate.py                   Reconstruction, MLM-fidelity, and sparsity metrics
  config.py                     Typed configuration schema
  notebook_analysis_utils.py    Shared analysis and plotting helpers
notebooks/
  biological_alignment_analysis.ipynb   Secondary-structure analysis
  family_alignment_analysis.ipynb       RNA-type, kNN, and PCA analysis
data/                         Evaluation inputs (see data/README.md)
models/                       Trained SAE checkpoints (see models/README.md)
```

## Installation

The paper experiments used Python 3.12. Install a PyTorch build matching your
CPU or CUDA setup first, then the recorded dependencies:

```bash
git clone https://github.com/SadatHossain01/SPIRAL.git
cd SPIRAL
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Building the RNAcentral holdout from scratch additionally requires
`cd-hit-est-2d` from [CD-HIT](https://github.com/weizhongli/cdhit).

## Quick check

This runs a short end-to-end pipeline test on CPU — a few training steps on a
small sample — and needs no downloaded checkpoints:

```bash
python train.py --debug --set training.device=cpu
```

## Getting the checkpoints and evaluation data

The trained SAEs and the fixed evaluation inputs are too large for Git and live
in the Zenodo archive. Download
[SPIRAL-v1.0.zip](https://doi.org/10.5281/zenodo.21891845), then copy its
`models/` and `data/` contents into this clone so the paths look like:

```text
models/layer_00/  models/layer_05/  models/layer_11/
data/rnacentral_holdout/  data/bprna/
```

The notebooks resolve these locations automatically: they use `./models` and
`./data` when run from the repository root, and the archive's `../models` and
`../data` when the code is run from inside the unpacked archive. No path editing
is required in either case.

## Reproducing the paper

Run the notebooks from the repository root so that `import src` resolves:

```bash
jupyter lab
```

| Notebook | Produces |
| --- | --- |
| `biological_alignment_analysis.ipynb` | Reconstruction, MLM-fidelity, and sparsity metrics; background structure distribution; structure selectivity and enrichment distributions; top structure-selective features; structure composition profiles |
| `family_alignment_analysis.ipynb` | RNA-type and sequence-length distributions; length-controlled effect sizes; top enriched and selective features; activation heatmaps; kNN probe and confusion matrices; PCA projections |

Figure files are written with the numeric prefixes used in the manuscript
(`01_…`, `02_…`, …), so each output maps directly onto a figure or supplementary
figure in the paper. Training from scratch is only needed to regenerate the
checkpoints:

```bash
python train.py --config config.yaml
```

Full training requires substantial GPU time and disk space; the released
checkpoints reproduce every reported analysis without retraining.

## Optional experiment tracking

Weights & Biases logging is off by default. To enable it, set
`logging.wandb_enabled: true` in `config.yaml`, point `logging.wandb_entity` at
your own entity, and provide credentials via the `WANDB_API_KEY` environment
variable.

## Citation

If you use this code, please cite the paper and the archive:

```bibtex
@article{spiral2026,
  title   = {Sparse Autoencoders Reveal Structural and Family-level Features in BiRNA-BERT},
  author  = {Hossain, Mohammad Sadat and Sojib, MD. Roqunuzzaman and Tahmid, Md Toki and Rahman, M Saifur},
  journal = {bioRxiv},
  year    = {2026},
  doi     = {10.64898/2026.08.11.744228}
}

@dataset{spiral2026archive,
  title     = {SPIRAL Reproducibility Archive},
  author    = {Hossain, Mohammad Sadat and Sojib, MD. Roqunuzzaman and Tahmid, Md Toki and Rahman, M Saifur},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.21891845}
}
```

Machine-readable metadata is in [`CITATION.cff`](CITATION.cff).

## License

The source code in this repository is released under the [MIT License](LICENSE).
BiRNA-BERT, RNAcentral, bpRNA-90, and any derived materials remain subject to
their own licenses and terms of use.
