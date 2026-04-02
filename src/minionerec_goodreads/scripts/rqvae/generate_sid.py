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
    checkpoint = torch.load(cfg.ckpt_path, map_location="cpu")
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
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

    all_codes = []
    with torch.no_grad():
        for (batch,) in dataloader:
            batch = batch.to(device)  # [B, D]
            batch_codes = model.rqvae.get_indices(batch, use_sk=cfg.use_sk_infer)  # [B, C]
            all_codes.append(batch_codes.cpu().numpy())

    raw_codes = np.concatenate(all_codes, axis=0).astype(np.int64) + 1  # [N, C]
    dedup_codes = deduplicate_codes(raw_codes)

    payload = {}
    for item_id, code in zip(item_ids, dedup_codes):
        tokens = code_to_tokens(np.asarray(code, dtype=np.int64))
        payload[str(item_id)] = tokens

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)

    num_unique = len({tuple(code) for code in raw_codes.tolist()})
    metrics = {
        "num_items": len(item_ids),
        "num_code_levels": int(raw_codes.shape[1]),
        "num_unique_code_paths": num_unique,
        "num_collisions": len(item_ids) - num_unique,
    }
    object_dict = {
        "cfg": cfg,
        "item_path": item_path,
        "embedding_path": embedding_path,
        "output_path": output_path,
    }
    log.info(f"Saved SIDs to {output_path}")
    return metrics, object_dict


@hydra.main(version_base="1.3", config_path="../../configs", config_name="generate_sid")
def main(cfg: DictConfig) -> None:
    extras(cfg)
    generate_sid(cfg)


if __name__ == "__main__":
    main()
