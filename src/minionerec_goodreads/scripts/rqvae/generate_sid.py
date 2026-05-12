from __future__ import annotations

import ast
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import hydra
import lightning as L
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, TensorDataset

from minionerec_goodreads.utils import RankedLogger, extras, task_wrapper
from minionerec_goodreads.utils.rqvae_stats import compare_code_assignments, compute_sid_health_report

log = RankedLogger(__name__, rank_zero_only=True)
OmegaConf.register_new_resolver("eval", ast.literal_eval)
torch.multiprocessing.set_sharing_strategy("file_system")
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


def load_item_ids(item_path: Path) -> list[int]:
    with item_path.open("r", encoding="utf-8") as file:
        item_payload = json.load(file)
    item_ids = sorted(int(item_id) for item_id in item_payload)
    return item_ids


def code_to_tokens(code: np.ndarray) -> list[str]:
    tokens = []
    for level_idx, value in enumerate(code.tolist()):
        prefix = chr(ord("a") + level_idx)
        tokens.append(f"<{prefix}_{value}>")
    return tokens


def deduplicate_codes(codes: np.ndarray) -> list[list[int]]:
    dedup_codes = []
    code_counts: dict[tuple[int, ...], int] = defaultdict(int)

    for code in codes.tolist():
        base_code = tuple(code)
        code_counts[base_code] += 1
        dup_rank = code_counts[base_code]
        if dup_rank == 1:
            dedup_codes.append(list(base_code))
        else:
            dedup_codes.append([*base_code, dup_rank])

    return dedup_codes


def default_report_path(output_path: Path, suffix: str) -> Path:
    stem = output_path.stem
    if stem.endswith(".index"):
        stem = stem[: -len(".index")]
    return output_path.with_name(f"{stem}.{suffix}")


def write_health_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# SID Codebook Health Report",
        "",
        f"Status: **{report['status'].upper()}**",
        "",
        "This report summarizes raw residual quantizer code paths before duplicate-rank fallback tokens are appended.",
        "",
        "## Path collisions",
        "",
        f"- Items: {report['paths']['num_items']}",
        f"- Unique raw code paths: {report['paths']['num_unique_code_paths']}",
        f"- Collision rate: {report['paths']['collision_rate']:.4%}",
        f"- Collided items: {report['paths']['num_collisions']}",
        "",
        "## Per-level utilization",
        "",
        "| Level | Used / Size | Dead codes | Usage | Perplexity | Max bucket share |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for level in report["levels"]:
        lines.append(
            f"| {level['level']} | {level['used_codes']} / {level['codebook_size']} | "
            f"{level['dead_codes']} | {level['used_ratio']:.2%} | {level['perplexity']:.2f} | "
            f"{level['max_bucket_share']:.2%} |"
        )

    if report.get("assignment_comparison"):
        comparison = report["assignment_comparison"]
        lines.extend([
            "",
            "## SK assignment vs argmin assignment",
            "",
            f"- Exact path agreement: {comparison['exact_path_agreement_rate']:.2%}",
        ])
        for level in comparison["per_level_agreement"]:
            lines.append(f"- Level {level['level']} agreement: {level['agreement_rate']:.2%}")

    duplicated = report["paths"].get("top_duplicated_paths", [])
    if duplicated:
        lines.extend(["", "## Top duplicated raw paths", ""])
        for item in duplicated:
            path_text = "-".join(str(value) for value in item["code_path"])
            lines.append(f"- `{path_text}`: {item['count']} items")

    if report["warnings"]:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in report["warnings"])

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def infer_codes(model: Any, dataloader: DataLoader, device: torch.device, use_sk: bool) -> np.ndarray:
    all_codes = []
    with torch.no_grad():
        for (batch,) in dataloader:
            batch = batch.to(device)  # [B, D]
            batch_codes = model.rqvae.get_indices(batch, use_sk=use_sk)  # [B, C]
            all_codes.append(batch_codes.cpu().numpy())
    return np.concatenate(all_codes, axis=0).astype(np.int64)


@task_wrapper
def generate_sid(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    item_path = Path(cfg.item_path)
    embedding_path = Path(cfg.embedding_path)
    output_path = Path(cfg.output_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    item_ids = load_item_ids(item_path)
    embeddings = np.load(embedding_path).astype(np.float32)  # [N, D]
    if len(item_ids) != embeddings.shape[0]:
        raise ValueError(f"Mismatched items and embeddings: {len(item_ids)} != {embeddings.shape[0]}")

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model = hydra.utils.instantiate(cfg.model)
    if embeddings.shape[1] != model.rqvae.in_dim:
        raise ValueError(f"Embedding dim mismatch: {embeddings.shape[1]} != model.rqvae.in_dim {model.rqvae.in_dim}")
    checkpoint = torch.load(cfg.ckpt_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    dataset = TensorDataset(torch.from_numpy(embeddings))
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=False,
    )

    compare_sk_assignments = bool(cfg.get("compare_sk_assignments", True))
    argmin_codes = infer_codes(model, dataloader, device, use_sk=False) if (not cfg.use_sk_infer or compare_sk_assignments) else None
    sk_codes = infer_codes(model, dataloader, device, use_sk=True) if (cfg.use_sk_infer or compare_sk_assignments) else None
    selected_codes = sk_codes if cfg.use_sk_infer else argmin_codes
    if selected_codes is None:
        raise RuntimeError("No SID assignment was generated")

    raw_codes = selected_codes + 1  # [N, C]
    dedup_codes = deduplicate_codes(raw_codes)

    payload = {}
    for item_id, code in zip(item_ids, dedup_codes):
        tokens = code_to_tokens(np.asarray(code, dtype=np.int64))
        payload[str(item_id)] = tokens

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)

    stats_output_path = Path(cfg.stats_output_path) if cfg.get("stats_output_path") else default_report_path(output_path, "sid_stats.json")
    health_output_path = Path(cfg.health_output_path) if cfg.get("health_output_path") else default_report_path(output_path, "sid_health.md")
    health_report = compute_sid_health_report(
        raw_codes,
        model.rqvae.rq.codebook_size_list,
        code_offset=1,
        top_k=10,
        include_histograms=True,
    )
    health_report["assignment_mode"] = "sk" if cfg.use_sk_infer else "argmin"
    if compare_sk_assignments and sk_codes is not None and argmin_codes is not None:
        health_report["assignment_comparison"] = compare_code_assignments(sk_codes + 1, argmin_codes + 1)

    stats_output_path.parent.mkdir(parents=True, exist_ok=True)
    with stats_output_path.open("w", encoding="utf-8") as file:
        json.dump(health_report, file, ensure_ascii=False, indent=2)
    write_health_markdown(health_report, health_output_path)

    metrics = {
        "num_items": len(item_ids),
        "num_code_levels": int(raw_codes.shape[1]),
        "num_unique_code_paths": int(health_report["paths"]["num_unique_code_paths"]),
        "num_collisions": int(health_report["paths"]["num_collisions"]),
        "collision_rate": float(health_report["paths"]["collision_rate"]),
    }
    object_dict = {
        "cfg": cfg,
        "item_path": item_path,
        "embedding_path": embedding_path,
        "output_path": output_path,
        "stats_output_path": stats_output_path,
        "health_output_path": health_output_path,
    }
    log.info(f"Saved SIDs to {output_path}")
    log.info(f"Saved SID stats to {stats_output_path}")
    log.info(f"Saved SID health report to {health_output_path}")
    return metrics, object_dict


@hydra.main(version_base="1.3", config_path="../../configs", config_name="generate_sid")
def main(cfg: DictConfig) -> None:
    extras(cfg)
    generate_sid(cfg)


if __name__ == "__main__":
    main()
