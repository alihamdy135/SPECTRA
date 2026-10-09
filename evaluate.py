import argparse
import json
import random
import time

import numpy as np
import pandas as pd

from config import load_config
from metrics import bootstrap_ci, cohens_d, compute_metrics, paired_ttest
from models.spectra import softmax, spectra_tta_fitness
from runtime import build_evaluation_runtime
from results import print_final_results
from utils import Timer, configure_run, reset_vram, set_seed, vram_mb, print_sep


METHOD_ORDER = ("no_adaptation", "equal_merge", "ties_merge", "single_lora", "scalar_fusion", "spectra")
METRICS = ("em", "f1", "rouge_l", "composite")


def simplex_grid(adapter_count: int, point_count: int) -> list[list[float]]:
    if adapter_count < 1 or point_count < 2:
        raise ValueError("Simplex grid requires at least one adapter and two points")
    if adapter_count == 1:
        return [[1.0]]
    units = point_count - 1

    def compositions(remaining, dimensions, prefix=()):
        if dimensions == 1:
            yield prefix + (remaining,)
            return
        for value in range(remaining + 1):
            yield from compositions(remaining - value, dimensions - 1, prefix + (value,))

    return [[value / units for value in point] for point in compositions(units, adapter_count)]


def evaluate_method(runtime, name, domain, condition, setup, examples, adapt_time=None, extra=None, seeds=None):
    seeds = tuple(runtime.config.evaluation_seeds if seeds is None else seeds)
    if not seeds:
        raise ValueError("Evaluation requires at least one random seed")
    all_em = []
    all_f1 = []
    all_rouge = []
    flat_composite = []
    reset_vram()
    prompts = [prompt for prompt, _ in examples]
    references = [reference for _, reference in examples]
    setup()
    for seed in seeds:
        set_seed(seed)
        indices = list(range(len(examples)))
        random.shuffle(indices)
        ordered_prompts = [prompts[index] for index in indices]
        ordered_references = [references[index] for index in indices]
        started = time.time()
        try:
            predictions = runtime.model.generate(ordered_prompts)
        except Exception as error:
            print(f"  [WARN] {error}")
            predictions = [""] * len(ordered_prompts)
        metrics = compute_metrics(predictions, ordered_references, do_bootstrap=False)
        all_em.append(metrics["em"])
        all_f1.append(metrics["f1"])
        all_rouge.append(metrics["rouge_l"])
        flat_composite.extend(metrics["comp_list"])
        print(
            f"  [{name:>26s}|{domain:>7s}|s={seed}]  "
            f"EM={metrics['em']:.4f}  F1={metrics['f1']:.4f}  "
            f"RL={metrics['rouge_l']:.4f}  Comp={metrics['composite']:.4f}  "
            f"t={time.time()-started:.1f}s"
        )
    composite_mean, lower, upper = bootstrap_ci(flat_composite)
    score_key = f"{condition}/{domain}"
    row = {
        "method": name,
        "domain": domain,
        "condition": condition,
        "n_test": len(examples),
        "n_seeds": len(seeds),
        "em": float(np.mean(all_em)),
        "em_std": float(np.std(all_em)),
        "f1": float(np.mean(all_f1)),
        "rouge_l": float(np.mean(all_rouge)),
        "composite": composite_mean,
        "comp_ci_lo": lower,
        "comp_ci_hi": upper,
        "adapt_time_s": adapt_time,
        "peak_vram_mb": vram_mb(),
        "timestamp": pd.Timestamp.utcnow().isoformat(),
    }
    if extra:
        row.update(extra)
    runtime.result_store.append(row, score_key, name, flat_composite)
    runtime.logger.info(
        "%s/%s EM=%.4f±%.4f F1=%.4f ROUGE-L=%.4f composite=%.4f CI=[%.4f, %.4f]",
        name,
        domain,
        row["em"],
        row["em_std"],
        row["f1"],
        row["rouge_l"],
        row["composite"],
        lower,
        upper,
    )
    print(
        f"  >>> {name}/{domain}:  EM={row['em']:.4f}±{row['em_std']:.4f}  "
        f"F1={row['f1']:.4f}  RL={row['rouge_l']:.4f}  "
        f"Composite={composite_mean:.4f} [{lower:.4f},{upper:.4f}]"
    )
    return row


def adapt_spectra(adapt_pairs, model, lora_names, apply_coefficients, config, router_weights=None, verbose=True):
    try:
        import cma
    except ImportError as error:
        raise RuntimeError("Install the cma package to run SPECTRA adaptation") from error
    dimensions = len(lora_names)
    if dimensions == 0:
        raise ValueError("At least one adapter is required")
    fitness = lambda params: spectra_tta_fitness(
        np.asarray(params),
        adapt_pairs,
        model,
        apply_coefficients,
        router_weights,
        config.router_prior_weight,
    )
    candidates = []
    if router_weights is not None:
        raw_router = np.log(np.clip(np.asarray(router_weights, dtype=float), 1e-6, 1.0))
        raw_router -= raw_router.mean()
        fit_router = fitness(raw_router)
        candidates.append((fit_router, raw_router.copy(), "router"))
        if verbose:
            print(f"    init router:       fit={fit_router:.4f}  c={np.round(softmax(raw_router), 3)}")
    for index, domain in enumerate(lora_names):
        raw = np.zeros(dimensions)
        raw[index] = 2.0
        fit = fitness(raw)
        candidates.append((fit, raw.copy(), f"single_{domain}"))
        if verbose:
            print(f"    init single_{domain}: fit={fit:.4f}  c={np.round(softmax(raw), 3)}")
    equal_raw = np.zeros(dimensions)
    fit_equal = fitness(equal_raw)
    candidates.append((fit_equal, equal_raw.copy(), "equal"))
    if verbose:
        print(f"    init equal:        fit={fit_equal:.4f}  c={np.round(softmax(equal_raw), 3)}")
    rng = np.random.RandomState(config.seed)
    for _ in range(5):
        raw = rng.randn(dimensions) * 0.5
        candidates.append((fitness(raw), raw.copy(), "random"))
    best_fit, best_raw, best_name = min(candidates, key=lambda candidate: candidate[0])
    if verbose:
        print(f"  Best init: {best_name}  fit={best_fit:.4f}  c={np.round(softmax(best_raw), 3)}")
    strategy = cma.CMAEvolutionStrategy(
        best_raw,
        config.cma_sigma0,
        {
            "popsize": config.cma_pop,
            "maxiter": config.cma_iters,
            "seed": config.seed,
            "verbose": -9,
            "tolx": 1e-6,
            "tolfun": 1e-6,
        },
    )
    history = []
    started = time.time()
    generation = 0
    while not strategy.stop():
        solutions = strategy.ask()
        losses = [fitness(solution) for solution in solutions]
        strategy.tell(solutions, losses)
        history.append(float(np.min(losses)))
        generation += 1
        if verbose and generation % 10 == 0:
            print(
                f"    gen {generation:3d}/{config.cma_iters}  fit={history[-1]:.4f}  "
                f"c={np.round(softmax(strategy.result.xbest), 3)}"
            )
    duration = time.time() - started
    best_coefficients = softmax(strategy.result.xbest)
    apply_coefficients(best_coefficients.tolist())
    if verbose:
        print(
            f"  Done: fit={strategy.result.fbest:.4f}  gens={generation}  t={duration:.1f}s"
        )
        print(f"  Final coeffs: {np.round(best_coefficients, 3)}")
    return best_coefficients, float(strategy.result.fbest), history, duration


def grid_search_scalar_fusion(runtime, adapt_pairs):
    point_count = runtime.config.grid_points
    points = simplex_grid(len(runtime.lora_names), point_count)
    random.Random(runtime.config.seed).shuffle(points)
    best_coefficients = None
    best_loss = float("inf")
    for coefficients in points[: runtime.config.grid_evaluations]:
        runtime.apply_coefficients(coefficients)
        loss = runtime.model.nll_answer(adapt_pairs[:8])
        if loss < best_loss:
            best_loss = loss
            best_coefficients = coefficients[:]
    if best_coefficients is None:
        raise RuntimeError("Scalar fusion grid search produced no candidate")
    return best_coefficients, best_loss


def _run_main_experiments(runtime):
    for domain in runtime.lora_names:
        test_examples = runtime.get_test_examples(domain)
        adapt_pairs = runtime.get_adapt_examples(domain, runtime.config.n_adapt)
        print(f"\n{'─' * 60}\n Domain: {domain.upper()}\n{'─' * 60}")
        runtime.logger.info("Main comparison domain=%s test=%d adapt=%d", domain, len(test_examples), len(adapt_pairs))
        if not test_examples:
            continue
        print(f"N_test={len(test_examples)}  N_adapt={len(adapt_pairs)}")
        evaluate_method(runtime, "no_adaptation", domain, "exp1_main", runtime.reset_base, test_examples)
        evaluate_method(
            runtime,
            f"single_lora_{domain}",
            domain,
            "exp1_main",
            lambda current=domain: runtime.apply_single(current),
            test_examples,
        )
        evaluate_method(runtime, "equal_merge", domain, "exp1_main", runtime.apply_equal, test_examples)
        evaluate_method(runtime, "ties_merge", domain, "exp1_main", runtime.apply_ties, test_examples)
        print("  Grid search SF …")
        scalar_started = time.time()
        scalar_coefficients, scalar_loss = grid_search_scalar_fusion(runtime, adapt_pairs)
        scalar_duration = time.time() - scalar_started
        print(f"    SF: {np.round(scalar_coefficients, 3)}  NLL={scalar_loss:.4f}  t={scalar_duration:.1f}s")
        runtime.logger.info(
            "Scalar fusion domain=%s coefficients=%s NLL=%.6f seconds=%.2f",
            domain,
            np.round(scalar_coefficients, 3).tolist(),
            scalar_loss,
            scalar_duration,
        )
        evaluate_method(
            runtime,
            "scalar_fusion",
            domain,
            "exp1_main",
            lambda values=scalar_coefficients: runtime.apply_coefficients(values),
            test_examples,
            extra={"sf_weights": json.dumps(scalar_coefficients)},
        )
        print("  SPECTRA (Router + TTA) …")
        adapt_prompts = [prompt for prompt, _ in adapt_pairs]
        print("  [Router] Measuring domain NLL …")
        router_weights = runtime.router.fit(adapt_prompts, verbose=True)
        with Timer("  SPECTRA"):
            coefficients, fitness, history, adaptation_time = adapt_spectra(
                adapt_pairs,
                runtime.model,
                runtime.lora_names,
                runtime.apply_coefficients,
                runtime.config,
                router_weights=router_weights,
                verbose=True,
            )
        runtime.save_coefficients(domain, coefficients)
        evaluate_method(
            runtime,
            "spectra",
            domain,
            "exp1_main",
            lambda values=coefficients: runtime.apply_coefficients(values.tolist()),
            test_examples,
            adapt_time=adaptation_time,
            extra={
                "cma_fit": fitness,
                "coeffs": json.dumps(coefficients.tolist()),
                "history": json.dumps(history),
                "router_w": json.dumps(router_weights.tolist()),
            },
        )
    return None


def run_main_comparison(runtime):
    print_sep("Evaluation Harness")
    print_sep("EXPERIMENT 1: Main Comparison")
    with Timer("Experiment 1"):
        _run_main_experiments(runtime)
    print_sep("EXPERIMENT 1 RANKINGS + SIGNIFICANCE")
    return summarize_main_comparison(runtime)


def summarize_main_comparison(runtime):
    exp1 = pd.DataFrame(
        [row for row in runtime.result_store.rows if row.get("condition") == "exp1_main"]
    )
    exp1["method_norm"] = exp1["method"].apply(
        lambda method: "single_lora" if method.startswith("single_lora_") else method
    )

    winners = {}
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
        winners[metric] = winner if not aggregate.empty else None
        if winner == "spectra":
            spectra_wins += 1
        print(f"\n  {metric.upper()} ranking:")
        for rank, (method, value) in enumerate(aggregate.items()):
            mark = " ← SPECTRA #1 ✓" if rank == 0 and method == "spectra" else (" ← #1" if rank == 0 else "")
            print(f"    {rank + 1}. {method:<28} {value:.4f}{mark}")

    print("\n  Significance (SPECTRA vs baselines, composite):")
    for domain in runtime.lora_names:
        score_key = f"exp1_main/{domain}"
        spectra_scores = runtime.result_store.scores.get(score_key, {}).get("spectra", [])
        if not spectra_scores:
            continue
        print(f"\n    Domain={domain}  (n={len(spectra_scores)} per-example scores)")
        for baseline in ("no_adaptation", "equal_merge", "ties_merge", "scalar_fusion", "single_lora"):
            baseline_scores = runtime.result_store.scores.get(score_key, {}).get(baseline, [])
            if not baseline_scores and baseline == "single_lora":
                baseline_scores = runtime.result_store.scores.get(score_key, {}).get(f"single_lora_{domain}", [])
            if not baseline_scores:
                continue
            statistic, p_value = paired_ttest(spectra_scores, baseline_scores)
            effect = cohens_d(spectra_scores, baseline_scores)
            difference = np.mean(spectra_scores) - np.mean(baseline_scores)
            marker = "✓ p<0.05" if p_value < 0.05 else "✗ ns"
            print(f"      vs {baseline:<22}: Δ={difference:+.4f}  t={statistic:+.2f}  p={p_value:.3f}  d={effect:+.2f}  {marker}")
    print(f"\n{'=' * 50}")
    print(f"SPECTRA wins {spectra_wins}/{len(METRICS)} metrics")
    return winners


def run(config_path="configs/experiment.yaml"):
    config = load_config(config_path)
    logger = configure_run(config, "evaluate")
    set_seed(config.seed)
    runtime = build_evaluation_runtime(config, logger)
    return run_main_comparison(runtime)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    arguments = parser.parse_args()
    run(arguments.config)


if __name__ == "__main__":
    main()
