from __future__ import annotations

import argparse
import ast
import csv
import json
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Any
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM

from minionerec_goodreads.metrics.rec import ranking_metrics
from minionerec_goodreads.models.sft import SFTModule
from minionerec_goodreads.utils.sft import build_item_sid_map, build_tokenizer
from minionerec_goodreads.utils.sft_generation import (
    constrained_sid_beam_search,
    format_seq_sid_prompt,
    format_title_history_prompt,
)
from minionerec_goodreads.utils.sid_trie import SIDTrie


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SFT generative recommendation with constrained SID decoding.")
    parser.add_argument("--model-path", required=True, help="Base HF model path or exported SFT model directory.")
    parser.add_argument("--checkpoint-path", default=None, type=Path, help="Optional Lightning SFT .ckpt path.")
    parser.add_argument(
        "--pretrained-model-name-or-path",
        default=None,
        help="Tokenizer source. Defaults to --model-path.",
    )
    parser.add_argument("--split-path", default="data/processed/rqvae/test.csv", type=Path)
    parser.add_argument("--sid-index-path", default="data/processed/rqvae/goodreads.index.json", type=Path)
    parser.add_argument("--max-samples", default=128, type=int)
    parser.add_argument("--num-beams", default=10, type=int)
    parser.add_argument("--ks", default="3,5,10", help="Comma separated K values.")
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--train-mode", default="qlora", choices=["full_finetune", "new_token_only", "qlora"])
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--lora-r", default=16, type=int)
    parser.add_argument("--lora-alpha", default=32, type=int)
    parser.add_argument("--lora-dropout", default=0.05, type=float)
    parser.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj",
    )
    parser.add_argument("--task", default="seq_sid_to_sid", choices=["seq_sid_to_sid", "title_history_to_sid"])
    parser.add_argument("--output-path", default=None, type=Path)
    return parser.parse_args()


def resolve_dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[name]


def load_rows(path: Path, max_samples: int | None) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if max_samples is not None and max_samples >= 0:
        rows = rows[:max_samples]
    if not rows:
        raise ValueError(f"No rows loaded from {path}")
    return rows


def mean_metrics(metric_rows: list[dict[str, float]]) -> dict[str, float]:
    totals: dict[str, float] = defaultdict(float)
    for row in metric_rows:
        for key, value in row.items():
            totals[key] += value
    return {key: value / len(metric_rows) for key, value in sorted(totals.items())}


def _load_exported_manifest(model_path: str) -> dict[str, Any] | None:
    manifest_path = Path(model_path) / "manifest.json"
    if not manifest_path.exists():
        return None
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _load_exported_model(args: argparse.Namespace, manifest: dict[str, Any]) -> tuple[torch.nn.Module, Any, dict[str, list[str]]]:
    export_dir = Path(args.model_path)
    tokenizer, sid_index, _ = build_tokenizer(
        pretrained_model_name_or_path=str(export_dir / "tokenizer"),
        sid_index_path=export_dir / "goodreads.index.json",
    )
    export_type = manifest["export_type"]
    if export_type == "adapter":
        base_model_path = args.pretrained_model_name_or_path or manifest["base_model_path"]
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=resolve_dtype(args.torch_dtype),
            trust_remote_code=True,
        )
        model.resize_token_embeddings(len(tokenizer))
        model = PeftModel.from_pretrained(model, export_dir / "adapter")
        return model, tokenizer, sid_index
    model = AutoModelForCausalLM.from_pretrained(
        export_dir / "model",
        torch_dtype=resolve_dtype(args.torch_dtype),
        trust_remote_code=True,
    )
    model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer, sid_index


def load_eval_model(args: argparse.Namespace) -> tuple[torch.nn.Module, Any, dict[str, list[str]]]:
    exported_manifest = _load_exported_manifest(args.model_path)
    if exported_manifest is not None and args.checkpoint_path is None:
        return _load_exported_model(args, exported_manifest)

    tokenizer_path = args.pretrained_model_name_or_path or args.model_path
    if args.checkpoint_path is None:
        tokenizer, sid_index, _ = build_tokenizer(
            pretrained_model_name_or_path=tokenizer_path,
            sid_index_path=args.sid_index_path,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=resolve_dtype(args.torch_dtype),
            trust_remote_code=True,
        )
        model.resize_token_embeddings(len(tokenizer))
        return model, tokenizer, sid_index

    module = SFTModule(
        pretrained_model_name_or_path=tokenizer_path,
        sid_index_path=str(args.sid_index_path),
        train_mode=args.train_mode,
        warmup_ratio=0.0,
        gradient_checkpointing=False,
        torch_dtype=args.torch_dtype,
        optimizer=partial(torch.optim.AdamW, lr=0.0),
        scheduler=None,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=[name for name in args.lora_target_modules.split(",") if name],
        load_in_4bit=args.load_in_4bit,
    )
    checkpoint = torch.load(args.checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    load_result = module.load_state_dict(state_dict, strict=False)
    if load_result.missing_keys or load_result.unexpected_keys:
        print(
            "Loaded checkpoint with key mismatch: "
            f"missing={len(load_result.missing_keys)}, unexpected={len(load_result.unexpected_keys)}"
        )
    return module.model, module.tokenizer, module.sid_index


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    model, tokenizer, sid_index = load_eval_model(args)
    model.to(args.device)
    model.eval()
    item_sid_map = build_item_sid_map(sid_index)
    trie = SIDTrie(tokenizer=tokenizer, sid_index=sid_index)

    rows = load_rows(args.split_path, max_samples=args.max_samples)
    ks = tuple(int(k) for k in args.ks.split(",") if k)
    max_k = max(ks)
    max_sid_length = max(len(tokens) for tokens in sid_index.values())
    metric_rows = []
    invalid_total = 0
    duplicate_total = 0
    examples = []

    for row in rows:
        history_item_ids = [str(item_id) for item_id in ast.literal_eval(row["history_item_ids"])]
        target_item_id = str(row["item_id"])
        if args.task == "seq_sid_to_sid":
            history_sids = [item_sid_map[item_id] for item_id in history_item_ids]
            prompt = format_seq_sid_prompt(history_sids)
        else:
            history_titles = [str(title) for title in ast.literal_eval(row["history_titles"])]
            prompt = format_title_history_prompt(history_titles)
        generated, stats = constrained_sid_beam_search(
            model=model,
            tokenizer=tokenizer,
            trie=trie,
            prompt=prompt,
            num_beams=max(args.num_beams, max_k),
            max_sid_length=max_sid_length,
        )
        predictions = [item.item_id for item in generated[:max_k]]
        metric_rows.append(ranking_metrics(predictions=predictions, target=target_item_id, ks=ks))
        invalid_total += stats.invalid_count
        duplicate_total += stats.duplicate_count
        if len(examples) < 5:
            examples.append(
                {
                    "target_item_id": target_item_id,
                    "target_sid": item_sid_map[target_item_id],
                    "predictions": [
                        {"item_id": item.item_id, "sid": item.sid, "score": item.score} for item in generated[:max_k]
                    ],
                }
            )

    metrics = mean_metrics(metric_rows)
    metrics["invalid_rate"] = invalid_total / len(rows)
    metrics["duplicate_rate"] = duplicate_total / len(rows)
    return {"metrics": metrics, "num_samples": len(rows), "examples": examples}


def main() -> None:
    args = parse_args()
    result = evaluate(args)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output_path is None:
        print(text)
    else:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        args.output_path.write_text(text, encoding="utf-8")
        print(f"Wrote SFT recommendation metrics to {args.output_path}")


if __name__ == "__main__":
    main()
