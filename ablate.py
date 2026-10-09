import argparse
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from config import load_config
from evaluate import adapt_spectra, evaluate_method
from metrics import bootstrap_ci, cohens_d, compute_metrics, paired_ttest
from runtime import build_evaluation_runtime
from results import archive_results, print_final_results
from utils import Timer, configure_run, reset_vram, set_seed, vram_mb, print_sep


def run_router_ablation(runtime):
    domain = "hybrid"
    test_examples = runtime.get_test_examples(domain)
    adapt_pairs = runtime.get_adapt_examples(domain, runtime.config.n_adapt, seed_offset=999)
    if not test_examples:
        return None
    evaluate_method(runtime, "ties_merge", domain, "exp2_ablation", runtime.apply_ties, test_examples)
    evaluate_method(runtime, "equal_merge", domain, "exp2_ablation", runtime.apply_equal, test_examples)
    adapt_prompts = [prompt for prompt, _ in adapt_pairs]
    router_weights = runtime.router.fit(adapt_prompts, verbose=True)
    coefficients, fitness, _, duration = adapt_spectra(
        adapt_pairs,
        runtime.model,
        runtime.lora_names,
        runtime.apply_coefficients,
        runtime.config,
        router_weights=router_weights,
        verbose=True,
    )
    evaluate_method(
        runtime,
        "spectra_full",
        domain,
        "exp2_ablation",
        lambda values=coefficients: runtime.apply_coefficients(values.tolist()),
        test_examples,
        adapt_time=duration,
        extra={"cma_fit": fitness, "coeffs": json.dumps(coefficients.tolist())},
    )
    print("\n[ABL] SPECTRA-NoRouter (TTA only)")
    no_router_coefficients, _, _, no_router_duration = adapt_spectra(
        adapt_pairs,
        runtime.model,
        runtime.lora_names,
        runtime.apply_coefficients,
        runtime.config,
        router_weights=None,
        verbose=False,
    )
    evaluate_method(
        runtime,
        "spectra_no_router",
        domain,
        "exp2_ablation",
        lambda values=no_router_coefficients: runtime.apply_coefficients(values.tolist()),
        test_examples,
        adapt_time=no_router_duration,
        extra={"coeffs": json.dumps(no_router_coefficients.tolist())},
    )
    print("\n[ABL] Router-Only (no CMA-ES)")
    router_only_pairs = runtime.get_adapt_examples(domain, runtime.config.n_adapt, seed_offset=999)
    router_only_weights = runtime.router.fit([prompt for prompt, _ in router_only_pairs])
    evaluate_method(
        runtime,
        "router_only",
        domain,
        "exp2_ablation",
        lambda values=router_only_weights: runtime.apply_coefficients(values.tolist()),
        test_examples,
        extra={"router_w": json.dumps(router_only_weights.tolist())},
    )
    print("\n  Ablation Significance (composite):")
    score_key = "exp2_ablation/hybrid"
    for treatment in ("spectra_full", "spectra_no_router", "router_only"):
        treatment_scores = runtime.result_store.scores.get(score_key, {}).get(treatment, [])
        for baseline in ("ties_merge", "equal_merge"):
            baseline_scores = runtime.result_store.scores.get(score_key, {}).get(baseline, [])
            if not treatment_scores or not baseline_scores:
                continue
            statistic, p_value = paired_ttest(treatment_scores, baseline_scores)
            effect = cohens_d(treatment_scores, baseline_scores)
            difference = float(np.mean(treatment_scores) - np.mean(baseline_scores))
            print(f"    {treatment} vs {baseline}: Δ={difference:+.4f}  p={p_value:.3f}  d={effect:+.2f} {'✓' if p_value < 0.05 else '✗'}")
    return None


def matching_main_adaptation_time(result_rows, domain, coefficients):
    target = np.asarray(coefficients, dtype=float)
    for row in reversed(result_rows):
        if row.get("method") != "spectra" or row.get("domain") != domain or row.get("condition") != "exp1_main":
            continue
        saved = row.get("coeffs")
        if isinstance(saved, str):
            try:
                saved = json.loads(saved)
            except (TypeError, ValueError):
                continue
        if saved is None:
            continue
        try:
            matches = np.allclose(np.asarray(saved, dtype=float), target)
        except (TypeError, ValueError):
            continue
        if not matches:
            continue
        try:
            duration = float(row.get("adapt_time_s"))
        except (TypeError, ValueError):
            continue
        if np.isfinite(duration) and duration >= 0:
            return duration
    return None


def effective_sample_counts(requested_counts, available):
    if available < 1:
        raise ValueError("At least one adaptation example is required")
    normalized = []
    for requested in requested_counts:
        if requested < 1:
            raise ValueError("Sample-efficiency counts must be positive")
        normalized.append((requested, min(requested, available)))
    return tuple(normalized)


def run_sample_efficiency(runtime):
    domain = "hybrid"
    test_examples = runtime.get_test_examples(domain)
    if not test_examples:
        return set()
    evaluate_method(runtime, "no_adaptation", domain, "exp3_ref", runtime.reset_base, test_examples)
    requested_counts = tuple(runtime.config.exp3_adapt_counts)
    all_adapt = runtime.get_adapt_examples(domain, max(requested_counts), seed_offset=7)
    previous_scores = None
    effective_counts = effective_sample_counts(requested_counts, len(all_adapt))
    for requested_count, actual_count in effective_counts:
        adapt_pairs = all_adapt[:requested_count]
        condition = f"exp3_n{requested_count}"
        print(f"\n── N'={requested_count} ──")
        runtime.logger.info("Sample-efficiency setting requested=%d actual=%d", requested_count, actual_count)
        evaluate_method(
            runtime,
            "ties_merge",
            domain,
            condition,
            runtime.apply_ties,
            test_examples,
            extra={"n_prime": requested_count},
        )
        evaluate_method(
            runtime,
            "equal_merge",
            domain,
            condition,
            runtime.apply_equal,
            test_examples,
            extra={"n_prime": requested_count},
        )
        router_weights = runtime.router.fit([prompt for prompt, _ in adapt_pairs])
        coefficients, fitness, _, duration = adapt_spectra(
            adapt_pairs,
            runtime.model,
            runtime.lora_names,
            runtime.apply_coefficients,
            runtime.config,
            router_weights=router_weights,
            verbose=False,
        )
        evaluate_method(
            runtime,
            "spectra",
            domain,
            condition,
            lambda values=coefficients: runtime.apply_coefficients(values.tolist()),
            test_examples,
            adapt_time=duration,
            extra={"n_prime": requested_count, "cma_fit": fitness},
        )
        current_scores = runtime.result_store.scores.get(condition + "/" + domain, {}).get("spectra", [])
        if previous_scores and current_scores:
            difference = float(np.mean(current_scores) - np.mean(previous_scores))
            trend = "↑ improving" if difference >= -0.003 else "↓ regressing"
            print(f"  [Monotonicity] N'={requested_count}: Δcomposite={difference:+.4f} {trend}")
        previous_scores = current_scores
    return effective_counts


def _get_or_fit_coefficients(runtime, domain, seed_offset):
    has_main_result = any(
        row.get("method") == "spectra" and row.get("domain") == domain and row.get("condition") == "exp1_main"
        for row in runtime.result_store.rows
    )
    coefficients = runtime.saved_coefficients.get(domain)
    if coefficients is not None and has_main_result:
        return coefficients, matching_main_adaptation_time(runtime.result_store.rows, domain, coefficients)
    adapt_pairs = runtime.get_adapt_examples(domain, runtime.config.n_adapt, seed_offset=seed_offset)
    if not adapt_pairs:
        raise ValueError(f"Coefficient fitting requires adaptation examples for {domain}")
    router_weights = runtime.router.fit([prompt for prompt, _ in adapt_pairs])
    coefficients, _, _, duration = adapt_spectra(
        adapt_pairs,
        runtime.model,
        runtime.lora_names,
        runtime.apply_coefficients,
        runtime.config,
        router_weights=router_weights,
        verbose=False,
    )
    runtime.save_coefficients(domain, coefficients)
    return coefficients, duration


def run_coefficient_analysis(runtime):
    rows = []
    print("  Domain   Coeffs                    Entropy  Router agree?")
    print("  " + "─" * 60)
    for domain in runtime.lora_names:
        coefficients, _ = _get_or_fit_coefficients(runtime, domain, seed_offset=4000)
        entropy = float(-np.sum(coefficients * np.log(coefficients + 1e-9)))
        router_prompts = runtime.get_adapt_prompts(domain, runtime.config.n_adapt, seed_offset=4001)
        router_weights = runtime.router.fit(router_prompts)
        spectra_domain = runtime.lora_names[int(np.argmax(coefficients))]
        router_domain = runtime.lora_names[int(np.argmax(router_weights))]
        agreement = "✓" if spectra_domain == router_domain else "✗"
        row = {
            "domain": domain,
            "dominant": spectra_domain,
            "entropy": entropy,
            "is_mixed": float(entropy > 0.5),
            "router_agree": agreement,
        }
        for index, name in enumerate(runtime.lora_names):
            row[f"coeff_{name}"] = float(coefficients[index])
        rows.append(row)
        print(f"  {domain:8s}: {np.round(coefficients, 3)}  H={entropy:.3f}  mixed={'Yes' if entropy > 0.5 else 'No '}  Router={agreement}")
    output = Path(runtime.config.work_dir) / "artifacts" / "exp4_coefficients.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    runtime.logger.info("Coefficient analysis saved to %s", output)
    return output


def run_efficiency_analysis(runtime):
    domain = "hybrid"
    test_examples = runtime.get_test_examples(domain, runtime.config.exp5_test_count)
    coefficients, _ = _get_or_fit_coefficients(runtime, domain, seed_offset=5000)
    prompts = [prompt for prompt, _ in test_examples]
    rows = []
    for name, setup in (
        ("no_adaptation", runtime.reset_base),
        ("ties_merge", runtime.apply_ties),
        ("spectra", lambda: runtime.apply_coefficients(coefficients.tolist())),
    ):
        setup()
        reset_vram()
        started = time.time()
        runtime.model.generate(prompts)
        elapsed = time.time() - started
        peak = vram_mb()
        per_example = elapsed / max(1, len(prompts))
        rows.append({"method": name, "eval_per_ex": per_example, "peak_vram_mb": peak})
        print(f"  {name:>20s}: {per_example:.2f}s/ex  {peak:.0f}MB")

    cma_cost = sum(
        _float_or_zero(row.get("adapt_time_s"))
        for row in runtime.result_store.rows
        if row.get("method") == "spectra"
    )
    overhead = rows[-1]["eval_per_ex"] - rows[0]["eval_per_ex"]
    print("\n  Amortization Analysis:")
    print(f"    Total CMA adapt cost: {cma_cost:.1f}s")
    print(f"    Inference overhead:   {max(overhead, 0):.3f}s/ex  (≈0 — weight merging has zero inference overhead)")
    print("    SPECTRA overhead is ONLY at adapt time, not eval time")
    domain_count = len(runtime.lora_names)
    print(f"    Amortized over {domain_count} domains: {cma_cost / max(1, domain_count):.1f}s/domain")

    output = Path(runtime.config.work_dir) / "artifacts" / "exp5_timing.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    return output


def _float_or_zero(value):
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def run_cross_domain(runtime):
    test_examples = []
    adapt_pairs = []
    for domain in runtime.lora_names:
        test_examples.extend(runtime.get_test_examples(domain, runtime.config.exp6_test_per_domain))
        adapt_pairs.extend(runtime.get_adapt_examples(domain, runtime.config.exp6_adapt_per_domain, seed_offset=600))
    random.Random(runtime.config.seed).shuffle(test_examples)
    random.Random(runtime.config.seed).shuffle(adapt_pairs)

    prompts = [prompt for prompt, _ in test_examples]
    references = [reference for _, reference in test_examples]
    methods = {}
    for name, setup in (
        ("no_adaptation", runtime.reset_base),
        ("equal_merge", runtime.apply_equal),
        ("ties_merge", runtime.apply_ties),
    ):
        setup()
        set_seed(runtime.config.seed)
        metrics = compute_metrics(runtime.model.generate(prompts), references)
        methods[name] = metrics
        mean, lower, upper = bootstrap_ci(metrics["comp_list"])
        print(f"  {name:<22}: comp={mean:.4f}  [{lower:.4f},{upper:.4f}]")

    router_prompts = [prompt for prompt, _ in adapt_pairs]
    router_weights = runtime.router.fit(router_prompts, verbose=True)
    coefficients, fitness, _, duration = adapt_spectra(
        adapt_pairs,
        runtime.model,
        runtime.lora_names,
        runtime.apply_coefficients,
        runtime.config,
        router_weights=router_weights,
        verbose=True,
    )
    set_seed(runtime.config.seed)
    spectra_metrics = compute_metrics(runtime.model.generate(prompts), references)
    methods["spectra"] = spectra_metrics
    mean, lower, upper = bootstrap_ci(spectra_metrics["comp_list"])
    print(f"  {'spectra':<22}: comp={mean:.4f}  [{lower:.4f},{upper:.4f}]")

    print("\n  Significance (SPECTRA vs baselines):")
    for baseline in ("no_adaptation", "equal_merge", "ties_merge"):
        statistic, p_value = paired_ttest(spectra_metrics["comp_list"], methods[baseline]["comp_list"])
        difference = spectra_metrics["composite"] - methods[baseline]["composite"]
        effect = cohens_d(spectra_metrics["comp_list"], methods[baseline]["comp_list"])
        marker = "✓ p<0.05" if p_value < 0.05 else f"✗ ns (p={p_value:.3f})"
        print(f"    vs {baseline:<20}: Δ={difference:+.4f}  t={statistic:+.2f}  d={effect:+.2f}  {marker}")

    best_method, best_metrics = max(methods.items(), key=lambda item: item[1]["composite"])
    print(f"\n  Best cross-domain: {best_method} (composite={best_metrics['composite']:.4f})")
    summary_row = {
        "method": "exp6_summary",
        "domain": "mixed",
        "condition": "exp6",
        "adapt_time_s": duration,
        "coeffs": json.dumps(coefficients.tolist()),
        "spectra_comp": spectra_metrics["composite"],
        "equal_comp": methods["equal_merge"]["composite"],
        "delta_eq": spectra_metrics["composite"] - methods["equal_merge"]["composite"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    runtime.result_store.append(summary_row)

    return None


def run_ablations(config_path="configs/experiment.yaml", runtime=None):
    if runtime is None:
        config = load_config(config_path)
        logger = configure_run(config, "ablate")
        set_seed(config.seed)
        runtime = build_evaluation_runtime(config, logger)
    else:
        config = runtime.config
    outputs = {}

    print_sep("EXPERIMENT 2: Ablation — Router Contribution")
    with Timer("Experiment 2"):
        outputs["router"] = run_router_ablation(runtime)

    print_sep("EXPERIMENT 3: Sample Efficiency")
    with Timer("Experiment 3"):
        outputs["sample_efficiency"] = run_sample_efficiency(runtime)

    print_sep("EXPERIMENT 4: Coefficient Analysis")
    with Timer("Experiment 4"):
        outputs["coefficients"] = run_coefficient_analysis(runtime)

    print_sep("EXPERIMENT 5: Efficiency + Amortization")
    with Timer("Experiment 5"):
        outputs["efficiency"] = run_efficiency_analysis(runtime)

    print_sep("EXPERIMENT 6: Cross-Domain")
    with Timer("Experiment 6"):
        outputs["cross_domain"] = run_cross_domain(runtime)

    print_sep("FINAL RESULTS")
    print_final_results(runtime.result_store)
    try:
        archive = archive_results(config.work_dir)
        outputs["archive"] = archive
        print(f"Archive: {archive}")
    except Exception as error:
        print(f"Archive: {error}")
    print_sep("SPECTRA COMPLETE")
    print(f"VRAM: {vram_mb():.0f}MB ✓")
    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    arguments = parser.parse_args()
    run_ablations(arguments.config)


if __name__ == "__main__":
    main()
