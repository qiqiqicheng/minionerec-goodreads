from __future__ import annotations

import json
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import gather_object
from omegaconf import DictConfig
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from minionerec_goodreads.utils import RankedLogger, extras, task_wrapper

log = RankedLogger(__name__, rank_zero_only=True)


def load_item_texts(item_path: Path) -> list[tuple[int, str]]:
    with item_path.open("r", encoding="utf-8") as file:
        item_payload = json.load(file)

    item_texts = []
    for item_id, payload in item_payload.items():
        text = payload["item_text"].strip()
        if text == "":
            raise ValueError(f"Empty item_text for item_id={item_id}")
        item_texts.append((int(item_id), text))
    item_texts.sort(key=lambda item: item[0])
    return item_texts


def mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand_as(last_hidden).float()  # [B, L, D]
    hidden_sum = torch.sum(last_hidden * mask, dim=1)  # [B, D]
    token_count = torch.clamp(mask.sum(dim=1), min=1e-9)  # [B, D]
    return hidden_sum / token_count


def encode_items(
    item_texts: list[tuple[int, str]],
    tokenizer: Any,
    model: Any,
    accelerator: Accelerator,
    batch_size: int,
    max_length: int,
) -> list[tuple[int, np.ndarray]]:
    num_items = len(item_texts)
    chunk_size = int(np.ceil(num_items / accelerator.num_processes))
    start = accelerator.process_index * chunk_size
    end = min(start + chunk_size, num_items)
    local_items = item_texts[start:end]

    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    if tokenizer.pad_token is None:
        raise ValueError("Tokenizer has no usable pad token")

    outputs = []
    model.eval()

    steps = range(0, len(local_items), batch_size)
    if accelerator.is_local_main_process:
        steps = tqdm(steps, desc="Encoding Items")

    with torch.no_grad():
        for offset in steps:
            batch_items = local_items[offset : offset + batch_size]
            batch_ids = [item_id for item_id, _ in batch_items]
            batch_texts = [text for _, text in batch_items]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(accelerator.device)
            hidden = model(**encoded).last_hidden_state  # [B, L, D]
            pooled = mean_pool(hidden, encoded.attention_mask).cpu().numpy()  # [B, D]

            # pooled_np = pooled.detach().cpu().numpy()

            for item_id, embedding in zip(batch_ids, pooled):
                outputs.append((item_id, embedding))
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

    accelerator.wait_for_everyone()
    gathered = gather_object(outputs)
    gathered.sort(key=lambda item: item[0])
    return gathered


@task_wrapper
def text2emb(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    accelerator = Accelerator()
    item_path = Path(cfg.item_path)
    output_path = Path(cfg.output_path)

    if torch.cuda.is_available() or torch.backends.mps.is_available():
        target_dtype = torch.bfloat16
    else:
        target_dtype = torch.float32
    
    # target_device = torch.device(cfg.device) if torch.cuda.is_available() else accelerator.device    
    target_device = accelerator.device
    
    log.info(f"Using target dtype: {target_dtype} and device: {target_device}")
    item_texts = load_item_texts(item_path)
    log.info(f"Loaded {len(item_texts)} items from {item_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(cfg.plm_checkpoint, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        cfg.plm_checkpoint,
        trust_remote_code=True,
        torch_dtype=target_dtype,
        low_cpu_mem_usage=True,
    ).to(accelerator.device)
    model.eval()

    gathered = encode_items(item_texts, tokenizer, model, accelerator, cfg.batch_size, cfg.max_length)

    metrics: dict[str, Any] = {"num_items": len(item_texts)}
    if accelerator.is_main_process:
        matrix = np.stack([embedding for _, embedding in gathered], axis=0)  # [N, D]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, matrix)
        metrics["emb_dim"] = int(matrix.shape[1])
        log.info(f"Saved embeddings to {output_path} with shape {matrix.shape}")

    object_dict = {
        "cfg": cfg,
        "item_path": item_path,
        "output_path": output_path,
        "accelerator": accelerator,
    }
    return metrics, object_dict


@hydra.main(version_base="1.3", config_path="../../configs", config_name="text2emb")
def main(cfg: DictConfig) -> None:
    extras(cfg)
    text2emb(cfg)


if __name__ == "__main__":
    main()
