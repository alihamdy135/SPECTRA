import time
from typing import Callable, Sequence

import numpy as np


def softmax(values: Sequence[float] | np.ndarray) -> np.ndarray:
    logits = np.asarray(values, dtype=float)
    shifted = logits - np.max(logits)
    weights = np.exp(shifted)
    return weights / weights.sum()


def spectra_tta_fitness(
    raw_params: np.ndarray,
    adapt_pairs,
    model,
    apply_coefficients: Callable[[Sequence[float]], None],
    router_weights: np.ndarray | None,
    prior_weight: float,
) -> float:
    coefficients = softmax(raw_params)
    apply_coefficients(coefficients.tolist())
    answer_nll = float(model.nll_answer(adapt_pairs))
    penalty = 0.0
    if router_weights is not None:
        router = np.clip(np.asarray(router_weights, dtype=float), 1e-6, 1.0)
        current = np.clip(coefficients, 1e-6, 1.0)
        penalty = prior_weight * float(np.sum(router * np.log(router / current)))
    return answer_nll + penalty


def spectra_adapt(
    adapt_pairs,
    model,
    lora_names: Sequence[str],
    apply_coefficients: Callable[[Sequence[float]], None],
    config,
    router_weights: np.ndarray | None = None,
    verbose: bool = True,
    started_at: float | None = None,
):
    started = started_at
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
    equal_fit = fitness(equal_raw)
    candidates.append((equal_fit, equal_raw.copy(), "equal"))
    if verbose:
        print(f"    init equal:        fit={equal_fit:.4f}  c={np.round(softmax(equal_raw), 3)}")
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
    if started is None:
        started = time.perf_counter()
    history = []
    generation = 0
    while not strategy.stop():
        solutions = strategy.ask()
        losses = [fitness(solution) for solution in solutions]
        strategy.tell(solutions, losses)
        history.append(float(np.min(losses)))
        generation += 1
        if verbose and generation % 10 == 0:
            best_coefficients = softmax(strategy.result.xbest)
            print(f"    gen {generation:3d}/{config.cma_iters}  fit={history[-1]:.4f}  c={np.round(best_coefficients, 3)}")
    best_raw = strategy.result.xbest
    best_fit = strategy.result.fbest
    coefficients = softmax(best_raw)
    apply_coefficients(coefficients.tolist())
    duration = time.perf_counter() - started
    if verbose:
        print(f"  Done: fit={best_fit:.4f}  gens={generation}  t={duration:.1f}s")
        print(f"  Final coeffs: {np.round(coefficients, 3)}")
    return coefficients, float(best_fit), history, duration
