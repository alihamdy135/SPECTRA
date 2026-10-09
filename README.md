# SPECTRA

## Source of truth

`SPECTRA.ipynb` is the original notebook and remains unchanged. The Python files divide its implementation into modules; they are not a replacement or an edited version of the notebook. `run_spectra.py` is the full-run entry point and keeps training, Experiment 1, Experiments 2–6, result state, and final reporting in one process and in notebook order.

## Layout

| Path | Responsibility |
| --- | --- |
| `config.py`, `configs/experiment.yaml` | Notebook defaults and runtime configuration |
| `data/loader.py`, `data/splits.py` | Dataset discovery, fallback data, and deterministic splits |
| `models/base.py` | Model discovery, loading, generation, and scoring |
| `models/lora.py` | LoRA training and adapter reuse |
| `models/merging.py` | Adapter deltas, weighted merging, and TIES |
| `models/gates.py` | Domain router |
| `models/spectra.py` | SPECTRA adaptation and CMA-ES |
| `metrics.py`, `results.py` | Notebook metrics, result rows, reports, and archive |
| `train.py`, `evaluate.py`, `ablate.py` | Individually callable stages |
| `run_spectra.py` | Full notebook-order run using shared model, splits, and results |

## Run

Run from the repository root with the packages in `requirements.txt` installed and the expected Kaggle data/model inputs mounted:

```bash
python run_spectra.py --config configs/experiment.yaml
```

The full driver uses the notebook's default adapter-reuse behavior. The stage-specific `train.py`, `evaluate.py`, and `ablate.py` commands are available for isolated work; they do not by themselves reproduce the full notebook's shared in-memory state or final sequence.

The defaults use `/kaggle/input/spectra-day1-2-artifacts-v1` and `/kaggle/working`. Dataset lookup follows the notebook's ordered roots and creates its notebook-compatible synthetic `DatasetDict` only when none load. Model remote-code execution is disabled by default; set `trust_remote_code: true` only after auditing and trusting the code bundled with the local model. Verify that the mounted model and dataset inputs are the intended ones before a full run.

## Outputs

The full run writes the notebook's `spectra_results.csv`, `spectra_results.zip`, coefficient artifacts, `exp4_coefficients.csv`, `exp5_timing.csv`, and LoRA adapter directories under the configured work directory. The result table is initialized for the current run and accumulates rows in memory in notebook order.

## Verification

Run the unit tests with:

```bash
python -m unittest discover -s tests -v
```

Tests use fakes for model, tokenizer, optimizer, and dataset interfaces; they do not constitute a real training/evaluation run. A full run requires the model weights, dataset or expected fallback environment, and a compatible PyTorch/CUDA setup. No full experiment is run by the unit-test command.

For bin-only checkpoints, module loading uses PyTorch's `weights_only=True` path; unsupported legacy pickle contents are rejected rather than loaded unsafely. This is a deliberate safety constraint and the one known compatibility limitation relative to unrestricted notebook loading.
