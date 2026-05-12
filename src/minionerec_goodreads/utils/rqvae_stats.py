from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np


def _as_code_matrix(codes: np.ndarray) -> np.ndarray:
    array = np.asarray(codes, dtype=np.int64)
    if array.ndim != 2:
        raise ValueError(f"Expected codes with shape [N, C], got {array.shape}")
    if array.shape[0] == 0:
        raise ValueError("Expected at least one code path")
    return array


def compute_level_stats(
    codes: np.ndarray,
    codebook_sizes: list[int],
    code_offset: int = 0,
    include_histogram: bool = False,
) -> list[dict[str, Any]]:
    code_matrix = _as_code_matrix(codes)
    if code_matrix.shape[1] != len(codebook_sizes):
        raise ValueError(f"Expected {len(codebook_sizes)} code levels, got {code_matrix.shape[1]}")

    level_stats = []
    num_items = code_matrix.shape[0]
    for level_idx, size in enumerate(codebook_sizes):
        shifted_codes = code_matrix[:, level_idx] - code_offset
        min_code = int(shifted_codes.min())
        max_code = int(shifted_codes.max())
        if min_code < 0 or max_code >= size:
            raise ValueError(
                f"Code level {level_idx} has values outside [{code_offset}, {code_offset + size - 1}]: "
                f"min={min_code + code_offset}, max={max_code + code_offset}"
            )

        counts = np.bincount(shifted_codes, minlength=size)[:size]
        used_count = int(np.count_nonzero(counts))
        probs = counts[counts > 0].astype(np.float64) / num_items
        entropy = float(-(probs * np.log(probs)).sum())
        perplexity = float(np.exp(entropy))
        stat: dict[str, Any] = {
            "level": level_idx,
            "codebook_size": int(size),
            "used_codes": used_count,
            "used_ratio": used_count / size,
            "dead_codes": int(size - used_count),
            "max_bucket_count": int(counts.max()),
            "max_bucket_share": float(counts.max() / num_items),
            "entropy": entropy,
            "normalized_entropy": float(entropy / np.log(size)) if size > 1 else 1.0,
            "perplexity": perplexity,
        }
        if include_histogram:
            stat["histogram"] = [
                {"code": int(code_idx + code_offset), "count": int(count)}
                for code_idx, count in enumerate(counts.tolist())
            ]
        level_stats.append(stat)
    return level_stats


def compute_path_stats(codes: np.ndarray, top_k: int = 10) -> dict[str, Any]:
    code_matrix = _as_code_matrix(codes)
    paths = [tuple(int(value) for value in row) for row in code_matrix.tolist()]
    counter = Counter(paths)
    num_items = len(paths)
    num_unique = len(counter)
    duplicated_paths = [(path, count) for path, count in counter.most_common() if count > 1]
    return {
        "num_items": num_items,
        "num_unique_code_paths": num_unique,
        "unique_code_path_ratio": num_unique / num_items,
        "num_collisions": num_items - num_unique,
        "collision_rate": (num_items - num_unique) / num_items,
        "num_duplicated_code_paths": len(duplicated_paths),
        "top_duplicated_paths": [
            {"code_path": list(path), "count": int(count)} for path, count in duplicated_paths[:top_k]
        ],
    }


def compare_code_assignments(left_codes: np.ndarray, right_codes: np.ndarray) -> dict[str, Any]:
    left = _as_code_matrix(left_codes)
    right = _as_code_matrix(right_codes)
    if left.shape != right.shape:
        raise ValueError(f"Assignment shapes differ: {left.shape} != {right.shape}")

    per_level = []
    for level_idx in range(left.shape[1]):
        per_level.append({
            "level": level_idx,
            "agreement_rate": float(np.mean(left[:, level_idx] == right[:, level_idx])),
        })
    return {
        "exact_path_agreement_rate": float(np.mean(np.all(left == right, axis=1))),
        "per_level_agreement": per_level,
    }


def compute_sid_health_report(
    codes: np.ndarray,
    codebook_sizes: list[int],
    code_offset: int = 0,
    top_k: int = 10,
    include_histograms: bool = False,
) -> dict[str, Any]:
    code_matrix = _as_code_matrix(codes)
    level_stats = compute_level_stats(code_matrix, codebook_sizes, code_offset, include_histograms)
    path_stats = compute_path_stats(code_matrix, top_k)

    warnings = []
    if path_stats["num_collisions"] > 0:
        warnings.append(
            "Raw code path collisions detected before deduplication; inspect duplicated paths before SFT."
        )
    for stat in level_stats:
        if stat["used_ratio"] < 0.5:
            warnings.append(
                f"Low codebook utilization at level {stat['level']}: {stat['used_ratio']:.2%} of codes are used."
            )
        if stat["max_bucket_share"] > 0.1:
            warnings.append(
                f"Dominant bucket at level {stat['level']}: one code contains {stat['max_bucket_share']:.2%} of items."
            )

    return {
        "status": "warning" if warnings else "healthy",
        "summary": "SID health report for raw residual quantizer code paths before deduplication.",
        "num_items": int(code_matrix.shape[0]),
        "num_code_levels": int(code_matrix.shape[1]),
        "levels": level_stats,
        "paths": path_stats,
        "warnings": warnings,
    }
