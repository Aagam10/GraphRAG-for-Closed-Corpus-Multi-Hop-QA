from __future__ import annotations
"""
Evaluation metrics for multi-hop QA: Exact Match, F1, retrieval precision/recall.
"""

import re
import string
import json
from collections import Counter
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import config


# ── Text normalization (standard SQuAD-style) ────────────────────────

def _normalize(text: str) -> str:
    """Lowercase, strip articles, punctuation, and extra whitespace."""
    text = text.lower()
    # remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # collapse whitespace
    text = " ".join(text.split())
    return text


def _get_tokens(text: str) -> list[str]:
    return _normalize(text).split()


# ── Core metrics ─────────────────────────────────────────────────────

def exact_match(predicted: str, gold: str) -> float:
    return float(_normalize(predicted) == _normalize(gold))


def f1_score(predicted: str, gold: str) -> float:
    pred_tokens = _get_tokens(predicted)
    gold_tokens = _get_tokens(gold)

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def retrieval_precision_recall(
    retrieved_titles: list[str],
    gold_titles: list[str],
) -> dict:
    """Compute retrieval precision and recall based on supporting-fact document titles."""
    retrieved_set = set(t.lower().strip() for t in retrieved_titles)
    gold_set = set(t.lower().strip() for t in gold_titles)

    if not gold_set:
        return {"precision": 1.0, "recall": 1.0}

    true_pos = len(retrieved_set & gold_set)
    precision = true_pos / len(retrieved_set) if retrieved_set else 0.0
    recall = true_pos / len(gold_set) if gold_set else 0.0
    return {"precision": precision, "recall": recall}


# ── Batch evaluation ─────────────────────────────────────────────────

def run_evaluation(
    dataset: list[dict],
    pipeline_fn,
    label: str = "system",
) -> dict:
    """
    Run a QA pipeline on a dataset and compute aggregate metrics.

    Args:
        dataset: list of {"question", "answer", "supporting_facts_titles" (optional)}
        pipeline_fn: callable(question: str) -> {"answer": str, "retrieved_titles": list[str]}
        label: name of the system being evaluated
    Returns:
        dict with averaged EM, F1, precision, recall
    """
    ems, f1s, precisions, recalls = [], [], [], []
    per_question = []

    for rec in tqdm(dataset, desc=f"Evaluating {label}"):
        result = pipeline_fn(rec["question"])
        pred = result["answer"]
        gold = rec["answer"]

        em = exact_match(pred, gold)
        f1 = f1_score(pred, gold)
        ems.append(em)
        f1s.append(f1)

        gold_titles = rec.get("supporting_facts_titles", [])
        if gold_titles:
            pr = retrieval_precision_recall(result.get("retrieved_titles", []), gold_titles)
            precisions.append(pr["precision"])
            recalls.append(pr["recall"])

        per_question.append({
            "question": rec["question"],
            "gold": gold,
            "predicted": pred,
            "em": em,
            "f1": f1,
        })

    results = {
        "system": label,
        "EM": sum(ems) / len(ems) if ems else 0,
        "F1": sum(f1s) / len(f1s) if f1s else 0,
        "precision": sum(precisions) / len(precisions) if precisions else 0,
        "recall": sum(recalls) / len(recalls) if recalls else 0,
        "n": len(dataset),
    }

    # Save per-question results
    out_path = config.RESULTS_DIR / f"{label}_per_question.json"
    out_path.write_text(json.dumps(per_question, indent=2, ensure_ascii=False), encoding="utf-8")

    return results


def compare_systems(results_list: list[dict]) -> pd.DataFrame:
    """Create a comparison table from multiple system results."""
    df = pd.DataFrame(results_list)
    df = df.set_index("system")
    display_cols = ["EM", "F1", "precision", "recall", "n"]
    df = df[[c for c in display_cols if c in df.columns]]

    # Save to CSV
    out_path = config.RESULTS_DIR / "comparison.csv"
    df.to_csv(out_path)
    print(f"\nComparison saved to {out_path}")
    print(df.to_string())
    return df


if __name__ == "__main__":
    # Sanity checks
    assert exact_match("Albert Einstein", "albert einstein") == 1.0
    assert exact_match("Einstein", "albert einstein") == 0.0
    assert f1_score("Albert Einstein was a physicist", "Einstein was a great physicist") > 0.5
    print("All metric tests passed.")
