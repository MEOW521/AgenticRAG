import re
from typing import Iterable


def normalize_answer(text: str) -> str:
    if text is None:
        return ""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def exact_match(prediction: str, gold: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(gold))


def token_f1(prediction: str, gold: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    pred_counts: dict[str, int] = {}
    gold_counts: dict[str, int] = {}
    for t in pred_tokens:
        pred_counts[t] = pred_counts.get(t, 0) + 1
    for t in gold_tokens:
        gold_counts[t] = gold_counts.get(t, 0) + 1

    common = 0
    for t, c in pred_counts.items():
        if t in gold_counts:
            common += min(c, gold_counts[t])
    if common == 0:
        return 0.0

    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def gold_exact_match(prediction: str, gold_answers: list[str]) -> float:
    if not gold_answers:
        return 0.0
    return max(exact_match(prediction, g) for g in gold_answers)


def gold_token_f1(prediction: str, gold_answers: list[str]) -> float:
    if not gold_answers:
        return 0.0
    return max(token_f1(prediction, g) for g in gold_answers)


def hit_at_k(pred_chunk_ids: Iterable[str], gold_chunk_ids: Iterable[str]) -> float:
    pred_set = {x for x in pred_chunk_ids if x}
    gold_set = {x for x in gold_chunk_ids if x}
    if not gold_set:
        return 0.0
    return float(bool(pred_set & gold_set))


def safe_avg(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def summarize_by_subset(rows: list[dict]) -> dict:
    """rows: [{'subset': ..., 'em': float, 'f1': float, 'hit_at_k': float}, ...]"""
    from collections import defaultdict

    buckets: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        key = str(r.get("subset") or "unknown")
        buckets[key].append(r)

    out = {}
    for key, lst in sorted(buckets.items()):
        out[key] = {
            "n": len(lst),
            "em": safe_avg([float(x["em"]) for x in lst]),
            "f1": safe_avg([float(x["f1"]) for x in lst]),
            "hit_at_k": safe_avg([float(x["hit_at_k"]) for x in lst]),
        }
    return out
