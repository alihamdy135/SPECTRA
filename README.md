# SPECTRA

Project materials for **SPECTRA: Domain-Adaptive LoRA Merging with Test-Time Coefficient Optimization**.

## Contents

- `spectra-2 (1).ipynb` — the research notebook, including its existing saved outputs.
- `main.tex` — the manuscript variant already present in this project directory. A separate `elsarticle` variant in the Downloads folder was updated separately; the two manuscript versions have not been merged.

## Data input

The notebook looks for the SPECTRA Day 1–2 artifact on Kaggle:

- Dataset: <https://www.kaggle.com/datasets/ali320230101/spectra-day1-2-artifacts-v1>
- Notebook input path: `/kaggle/input/datasets/ali320230101/spectra-day1-2-artifacts-v1/`
- The notebook's saved loader output records 5,000 examples in each of the `math`, `coding`, and `hybrid` splits, with 4,935/15/50 train/adaptation/test examples per split.

The Kaggle page identifies version 2 but currently has no dataset description and lists the license as **Unknown**. The artifact itself is not included in this repository. Confirm its underlying data provenance and license before redistribution or a public release.

## Reproduction note

If the Kaggle artifact is not mounted, the notebook falls back to generated toy examples (200 per domain). That fallback is not the 5,000-example dataset described in the manuscript and should not be used to reproduce the reported paper results. The saved notebook outputs were inspected but the notebook was not rerun as part of this setup.
