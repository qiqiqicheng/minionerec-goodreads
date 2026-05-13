from __future__ import annotations

import math


def hit_at_k(predictions: list[str], target: str, k: int) -> float:
    return float(target in predictions[:k])


def ndcg_at_k(predictions: list[str], target: str, k: int) -> float:
    for rank, item_id in enumerate(predictions[:k], start=1):
        if item_id == target:
            return 1.0 / math.log2(rank + 1)
    return 0.0


def mrr_at_k(predictions: list[str], target: str, k: int) -> float:
    for rank, item_id in enumerate(predictions[:k], start=1):
        if item_id == target:
            return 1.0 / rank
    return 0.0


def ranking_metrics(predictions: list[str], target: str, ks: tuple[int, ...]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for k in ks:
        metrics[f"HR@{k}"] = hit_at_k(predictions, target, k)
        metrics[f"NDCG@{k}"] = ndcg_at_k(predictions, target, k)
    metrics[f"MRR@{max(ks)}"] = mrr_at_k(predictions, target, max(ks))
    return metrics
