import re
import string
from collections import Counter
from typing import Sequence

import numpy as np
from scipy import stats


def bootstrap_ci(scores: Sequence[float], n_boot: int = 1000, alpha: float = 0.05, seed: int = 42) -> tuple[float, float, float]:
    if not scores:
        return 0.0, 0.0, 0.0
    rng = np.random.RandomState(seed)
    values = np.asarray(scores)
    means = [np.mean(rng.choice(values, len(values), replace=True)) for _ in range(n_boot)]
    return (
        float(np.mean(values)),
        float(np.percentile(means, 100 * alpha / 2)),
        float(np.percentile(means, 100 * (1 - alpha / 2))),
    )


def paired_ttest(a: Sequence[float], b: Sequence[float]) -> tuple[float, float]:
    if len(a) < 2 or len(b) < 2:
        return 0.0, 1.0
    count = min(len(a), len(b))
    statistic, p_value = stats.ttest_rel(a[:count], b[:count])
    return float(statistic), float(p_value)


def cohens_d(a: Sequence[float], b: Sequence[float]) -> float:
    count = min(len(a), len(b))
    first = np.asarray(a[:count])
    second = np.asarray(b[:count])
    pooled = np.sqrt((np.std(first) ** 2 + np.std(second) ** 2) / 2 + 1e-9)
    return float((np.mean(first) - np.mean(second)) / pooled)


def normalize_answer(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\\boxed\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\$+", "", s)
    s = "".join(character if character not in string.punctuation else " " for character in s)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def extract_final_answer(text: str) -> str:
    matches = re.findall(r"\\boxed\{([^}]+)\}", text)
    if matches:
        return matches[-1].strip()
    matches = re.findall(
        r"(?:the answer is|answer:|=|result is|therefore)[:\s]+([^\n.]+)",
        text,
        re.IGNORECASE,
    )
    if matches:
        return matches[-1].strip()
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    return lines[-1] if lines else text.strip()


def em_score(pred: str, gold: str) -> float:
    return float(normalize_answer(extract_final_answer(pred)) == normalize_answer(extract_final_answer(gold)))


def token_f1(pred: str, gold: str) -> float:
    prediction_tokens = normalize_answer(extract_final_answer(pred)).split()
    reference_tokens = normalize_answer(extract_final_answer(gold)).split()
    if not prediction_tokens or not reference_tokens:
        return 0.0
    common = Counter(prediction_tokens) & Counter(reference_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def rouge_l(prediction: str, reference: str) -> float:
    try:
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        return float(scorer.score(reference, prediction)["rougeL"].fmeasure)
    except:
        prediction_tokens = prediction.lower().split()
        reference_tokens = reference.lower().split()
        if not prediction_tokens or not reference_tokens:
            return 0.0
        rows, columns = len(reference_tokens), len(prediction_tokens)
        table = [[0] * (columns + 1) for _ in range(rows + 1)]
        for row in range(1, rows + 1):
            for column in range(1, columns + 1):
                table[row][column] = (
                    table[row - 1][column - 1] + 1
                    if reference_tokens[row - 1] == prediction_tokens[column - 1]
                    else max(table[row - 1][column], table[row][column - 1])
                )
        common = table[rows][columns]
        precision = common / columns if columns > 0 else 0
        recall = common / rows if rows > 0 else 0
        return 2 * precision * recall / (precision + recall + 1e-8)


def compute_metrics(
    preds: Sequence[str],
    golds: Sequence[str],
    do_bootstrap: bool = True,
) -> dict[str, float | list[float]]:
    if not preds:
        return {}
    exact_match = [em_score(pred, gold) for pred, gold in zip(preds, golds)]
    f1_scores = [token_f1(pred, gold) for pred, gold in zip(preds, golds)]
    rouge_scores = [rouge_l(pred, gold) for pred, gold in zip(preds, golds)]
    composite_scores = [
        0.4 * exact + 0.35 * f1 + 0.25 * rouge
        for exact, f1, rouge in zip(exact_match, f1_scores, rouge_scores)
    ]
    result: dict[str, float | list[float]] = {
        "em": float(np.mean(exact_match)),
        "f1": float(np.mean(f1_scores)),
        "rouge_l": float(np.mean(rouge_scores)),
        "composite": float(np.mean(composite_scores)),
        "em_list": exact_match,
        "f1_list": f1_scores,
        "rl_list": rouge_scores,
        "comp_list": composite_scores,
    }
    if do_bootstrap:
        for name, values in (("em", exact_match), ("f1", f1_scores), ("rouge_l", rouge_scores), ("composite", composite_scores)):
            _, lower, upper = bootstrap_ci(values)
            result[f"{name}_ci_lo"] = lower
            result[f"{name}_ci_hi"] = upper
    return result
