import shutil
from pathlib import Path

import pandas as pd


METHOD_ORDER = ["no_adaptation", "equal_merge", "ties_merge", "single_lora", "scalar_fusion", "spectra"]
METRICS = ["em", "f1", "rouge_l", "composite"]


class ResultStore:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results_path = self.output_dir / "spectra_results.csv"
        self.rows = []
        self.scores = {}
        self.columns = []

    def append(self, row, score_key=None, method=None, scores=None):
        complete = dict(row)
        self.rows.append(complete)
        for field in complete:
            if field not in self.columns:
                self.columns.append(field)
        if score_key is not None and method is not None and scores is not None:
            self.scores.setdefault(score_key, {})[method] = list(scores)
        self.flush()

    def save_scores(self, score_key, method, scores):
        self.scores.setdefault(score_key, {})[method] = list(scores)

    def flush(self):
        if self.rows:
            pd.DataFrame(self.rows).to_csv(self.results_path, index=False)


def print_final_results(result_store):
    results_path = result_store.results_path
    if not results_path.is_file():
        return
    result_frame = pd.read_csv(results_path)
    exp1 = result_frame[result_frame["condition"] == "exp1_main"].copy()
    exp1["method_norm"] = exp1["method"].apply(
        lambda method: "single_lora" if method.startswith("single_lora_") else method
    )

    print(f"\n{'Method':<28}", end="")
    for metric in METRICS:
        print(f"  {metric:>10}", end="")
    print(f"  {'95% CI composite':>22}")
    print("─" * 95)

    for method in METHOD_ORDER:
        subset = exp1[exp1["method_norm"] == method]
        if subset.empty:
            continue
        print(f"  {method:<26}", end="")
        for metric in METRICS:
            value = subset[metric].mean() if metric in subset.columns else 0
            print(f"  {value:>10.4f}", end="")
        if "comp_ci_lo" in subset.columns:
            lower = subset["comp_ci_lo"].mean()
            upper = subset["comp_ci_hi"].mean()
            print(f"  [{lower:.4f},{upper:.4f}]", end="")
        marker = " ◄ SPECTRA" if method == "spectra" else ""
        print(marker)

    print(f"\n{'=' * 50}")
    print("SPECTRA FINAL RANKING:")
    spectra_wins = 0
    for metric in METRICS:
        if metric not in exp1.columns:
            continue
        aggregate = (
            exp1.groupby("method_norm")[metric]
            .mean()
            .reindex(METHOD_ORDER)
            .dropna()
            .sort_values(ascending=False)
        )
        winner = aggregate.index[0] if not aggregate.empty else "?"
        if winner == "spectra":
            spectra_wins += 1
        spectra_rank = list(aggregate.index).index("spectra") + 1 if "spectra" in aggregate.index else "?"
        print(
            f"  {metric:12s}: #1={winner:<20} SPECTRA=#{spectra_rank}"
            f"{'  ✓ WINS' if winner == 'spectra' else ''}"
        )
    print(f"\n  SPECTRA wins {spectra_wins}/{len(METRICS)} metrics")
    print("\n  Key advantage: SPECTRA uses domain routing + TTA")
    print("  → No extra labeled data needed vs oracle single_lora")
    print("  → Automatic domain identification from unlabeled prompts")


def archive_results(output_dir):
    root = Path(output_dir)
    archive = shutil.make_archive(str(root / "spectra_results"), "zip", str(root))
    return Path(archive)
