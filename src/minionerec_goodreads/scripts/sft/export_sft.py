from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from omegaconf import OmegaConf
from safetensors.torch import save_file

if TYPE_CHECKING:
    from minionerec_goodreads.models.sft import SFTModule

DEFAULT_BASE_MODEL_PATH = "/mnt/disk2/chengqi/models/llm/Qwen/Qwen2.5-3B-Instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export an SFT Lightning checkpoint for reproducible inference/eval.")
    parser.add_argument("--checkpoint-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--pretrained-model-name-or-path", default=None)
    parser.add_argument("--sid-index-path", required=True, type=Path)
    parser.add_argument("--item-path", required=True, type=Path)
    parser.add_argument("--resolved-config-path", default=None, type=Path)
    parser.add_argument("--train-mode", default=None, choices=["full_finetune", "new_token_only", "qlora"])
    parser.add_argument("--torch-dtype", default=None, choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--lora-backend", default=None, choices=["peft", "unsloth"])
    parser.add_argument("--max-seq-length", default=None, type=int)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--lora-r", default=None, type=int)
    parser.add_argument("--lora-alpha", default=None, type=int)
    parser.add_argument("--lora-dropout", default=None, type=float)
    parser.add_argument(
        "--lora-target-modules",
        default=None,
    )
    parser.add_argument("--export-type", default="auto", choices=["auto", "adapter", "full_model"])
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def infer_resolved_config_path(checkpoint_path: Path, resolved_config_path: Path | None) -> Path | None:
    if resolved_config_path is not None:
        return resolved_config_path
    for parent in checkpoint_path.parents:
        candidate = parent / ".hydra" / "config.yaml"
        if candidate.exists():
            return candidate
    return None


def copy_artifact(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def load_sft_module(args: argparse.Namespace) -> SFTModule:
    checkpoint = torch.load(args.checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_hparams = checkpoint.get("hyper_parameters", {})
    train_mode = args.train_mode or checkpoint_hparams.get("train_mode", "qlora")
    torch_dtype = args.torch_dtype or checkpoint_hparams.get("torch_dtype", "bfloat16")
    base_model_path = args.pretrained_model_name_or_path or checkpoint_hparams.get("pretrained_model_name_or_path") or DEFAULT_BASE_MODEL_PATH
    lora_backend = args.lora_backend or checkpoint_hparams.get("lora_backend", "peft")
    max_seq_length = args.max_seq_length if args.max_seq_length is not None else checkpoint_hparams.get("max_seq_length")
    lora_r = args.lora_r if args.lora_r is not None else checkpoint_hparams.get("lora_r", 16)
    lora_alpha = args.lora_alpha if args.lora_alpha is not None else checkpoint_hparams.get("lora_alpha", 32)
    lora_dropout = args.lora_dropout if args.lora_dropout is not None else checkpoint_hparams.get("lora_dropout", 0.05)
    if args.lora_target_modules is None:
        lora_target_modules = checkpoint_hparams.get("lora_target_modules")
        if lora_target_modules is None:
            lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
    else:
        lora_target_modules = [name for name in args.lora_target_modules.split(",") if name]
    if lora_target_modules is not None and not isinstance(lora_target_modules, list):
        lora_target_modules = list(lora_target_modules)
    gradient_checkpointing = args.gradient_checkpointing or checkpoint_hparams.get("gradient_checkpointing", False)
    load_in_4bit = args.load_in_4bit or checkpoint_hparams.get("load_in_4bit", False)
    if lora_backend == "unsloth":
        import unsloth  # noqa: F401

    from minionerec_goodreads.models.sft import SFTModule

    module = SFTModule(
        pretrained_model_name_or_path=base_model_path,
        sid_index_path=str(args.sid_index_path),
        train_mode=train_mode,
        warmup_ratio=0.0,
        gradient_checkpointing=gradient_checkpointing,
        torch_dtype=torch_dtype,
        optimizer=partial(torch.optim.AdamW, lr=0.0),
        scheduler=None,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_target_modules=lora_target_modules,
        lora_backend=lora_backend,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
    )
    state_dict = checkpoint.get("state_dict", checkpoint)
    load_result = module.load_state_dict(state_dict, strict=False)
    if load_result.missing_keys or load_result.unexpected_keys:
        print(
            "Loaded checkpoint with key mismatch: "
            f"missing={len(load_result.missing_keys)}, unexpected={len(load_result.unexpected_keys)}"
        )
    module.eval()
    return module


def save_new_token_embeddings(module: SFTModule, adapter_dir: Path) -> None:
    start = module.original_vocab_size
    end = len(module.tokenizer)
    if end <= start:
        raise ValueError("SID adapter export requires added tokenizer rows")

    input_embedding = module.model.get_input_embeddings()
    if input_embedding is None:
        raise ValueError("Model must expose input embeddings")

    tensors = {"input_embeddings": input_embedding.weight.detach()[start:end].cpu()}
    output_embedding = module.model.get_output_embeddings()
    if output_embedding is not None:
        tensors["output_embeddings"] = output_embedding.weight.detach()[start:end].cpu()
    save_file(tensors, adapter_dir / "new_embeddings.safetensors", metadata={"format": "pt"})


def save_model_artifact(module: SFTModule, output_dir: Path, export_type: str) -> str:
    if export_type == "auto":
        export_type = "adapter" if module.hparams.train_mode == "qlora" else "full_model"
    if export_type == "adapter":
        if module.hparams.train_mode != "qlora":
            raise ValueError("adapter export is only valid for LoRA train_mode=qlora")
        adapter_dir = output_dir / "adapter"
        module.model.save_pretrained(adapter_dir, save_embedding_layers=False)
        save_new_token_embeddings(module, adapter_dir)
        return "adapter"
    module.model.save_pretrained(output_dir / "model")
    return "full_model"


def build_manifest(args: argparse.Namespace, module: SFTModule, saved_model_type: str) -> dict[str, Any]:
    from minionerec_goodreads.utils.sft import check_sid_tokenizer_atomicity

    sid_lengths = [len(tokens) for tokens in module.sid_index.values()]
    tokenizer_check = check_sid_tokenizer_atomicity(module.tokenizer, module.sid_index, sample_size=None)
    manifest = {
        "checkpoint_path": str(args.checkpoint_path),
        "base_model_path": module.hparams.pretrained_model_name_or_path,
        "train_mode": module.hparams.train_mode,
        "lora_backend": module.hparams.lora_backend,
        "max_seq_length": module.hparams.max_seq_length,
        "export_type": saved_model_type,
        "torch_dtype": module.hparams.torch_dtype,
        "load_in_4bit": module.hparams.load_in_4bit,
        "sid_index_path": str(args.sid_index_path),
        "sid_index_sha256": file_sha256(args.sid_index_path),
        "item_path": str(args.item_path),
        "item_sha256": file_sha256(args.item_path),
        "sid_item_count": len(module.sid_index),
        "sid_token_count": tokenizer_check.sid_token_count,
        "sid_min_length": min(sid_lengths),
        "sid_max_length": max(sid_lengths),
        "original_vocab_size": module.original_vocab_size,
        "augmented_vocab_size": len(module.tokenizer),
        "adapter_embedding_format": "new_token_rows" if saved_model_type == "adapter" else None,
        "sid_token_row_count": len(module.tokenizer) - module.original_vocab_size,
    }
    resolved_config_path = infer_resolved_config_path(args.checkpoint_path, args.resolved_config_path)
    if resolved_config_path is not None:
        manifest["resolved_config_path"] = str(resolved_config_path)
        manifest["resolved_config_sha256"] = file_sha256(resolved_config_path)
    return manifest


def export_sft(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    module = load_sft_module(args)
    saved_model_type = save_model_artifact(module, args.output_dir, args.export_type)
    module.tokenizer.save_pretrained(args.output_dir / "tokenizer")
    copy_artifact(args.sid_index_path, args.output_dir / "goodreads.index.json")
    copy_artifact(args.item_path, args.output_dir / "goodreads.item.json")
    resolved_config_path = infer_resolved_config_path(args.checkpoint_path, args.resolved_config_path)
    if resolved_config_path is not None:
        copy_artifact(resolved_config_path, args.output_dir / "resolved_config.yaml")
    manifest = build_manifest(args, module, saved_model_type)
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    OmegaConf.save(config=OmegaConf.create(manifest), f=args.output_dir / "manifest.yaml")
    print(f"Exported SFT checkpoint to {args.output_dir} as {saved_model_type}")


def main() -> None:
    export_sft(parse_args())


if __name__ == "__main__":
    main()
