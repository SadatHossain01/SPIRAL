# Model release

This trained sparse autoencoder weights are distributed in the Zenodo archive,
<https://doi.org/10.5281/zenodo.21891845>.

The archive contains one directory per analysed BiRNA-BERT layer — layers 0, 5,
and 11 — each providing:

- inference weights for the sparse autoencoder;
- the fixed activation mean and standard deviation used to standardise hidden
  states at that layer;
- SAE architecture metadata;
- the exact training configuration; and
- the final evaluation metrics.
