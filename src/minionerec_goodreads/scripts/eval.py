from __future__ import annotations

import argparse
import gc
import json
import os
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from tqdm import tqdm

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

from minionerec_goodreads.dataset.sft_dataset import load_split_csv, parse_list_column
from minionerec_goodreads.metrics.rec import ranking_metrics
from minionerec_goodreads.utils.sft import build_item_sid_map, load_sid_index
from minionerec_goodreads.utils.sft_generation import (
    GeneratedItem,
    GenerationStats,
    constrained_sid_beam_search,
    format_seq_sid_prompt,
    format_title_history_prompt,
)
from minionerec_goodreads.utils.sid_trie import SIDTrie
from minionerec_goodreads.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)

METHOD_RANDOM = "random_sid"
METHOD_POPULARITY = "popularity_topk"
METHOD_BASE = "base_qwen_sid_constrained"
METHOD_SFT = "sft_model"
GENERATION_METHODS = {METHOD_BASE, METHOD_SFT}


@dataclass(frozen=True)
class ExportBundle:
    export_dir: Path
    manifest: dict[str, Any]
    tokenizer: PreTrainedTokenizerBase
    sid_index: dict[str, list[str]]
    item_sid_map: dict[str, str]
    trie: SIDTrie


@dataclass(frozen=True)
class EvalExample:
    row_index: int
    user_id: str
    history_item_ids: list[str]
    target_item_id: str
    target_sid: str
    prompt: str
    history_len: int
    history_slice: str


@dataclass(frozen=True)
class PredictionResult:
    items: list[GeneratedItem]
    stats: GenerationStats


@dataclass(frozen=True)
class MethodEvalResult:
    report: dict[str, Any]
    example_predictions: dict[int, list[GeneratedItem]]


class HFConstrainedGenerator:
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: PreTrainedTokenizerBase,
        trie: SIDTrie,
        num_beams: int,
        max_sid_length: int,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.trie = trie
        self.num_beams = num_beams
        self.max_sid_length = max_sid_length

    def generate(self, prompt: str, max_k: int) -> PredictionResult:
        items, stats = constrained_sid_beam_search(
            model=self.model,
            tokenizer=self.tokenizer,
            trie=self.trie,
            prompt=prompt,
            num_beams=max(self.num_beams, max_k),
            max_sid_length=self.max_sid_length,
        )
        return PredictionResult(items=items[:max_k], stats=stats)


class VLLMConstrainedGenerator:  # noqa: C901
    def __init__(
        self,
        model_path: str,
        tokenizer_path: Path,
        tokenizer: PreTrainedTokenizerBase,
        trie: SIDTrie,
        num_beams: int,
        max_sid_length: int,
        dtype: str,
        seed: int,
        adapter_path: Path | None = None,
    ):
        from vllm import LLM, SamplingParams

        self.SamplingParams = SamplingParams
        self.tokenizer = tokenizer
        self.trie = trie
        self.num_beams = num_beams
        self.max_sid_length = max_sid_length
        self.lora_request = None
        kwargs: dict[str, Any] = {
            "model": model_path,
            "tokenizer": str(tokenizer_path),
            "trust_remote_code": True,
            "tensor_parallel_size": 1,
            "dtype": vllm_dtype(dtype),
            "seed": seed,
        }
        if adapter_path is not None:
            from vllm.lora.request import LoRARequest

            kwargs["enable_lora"] = True
            self.lora_request = LoRARequest("eval_adapter", 1, lora_path=str(adapter_path))
        self.llm = LLM(**kwargs)

    def generate(self, prompt: str, max_k: int) -> PredictionResult:
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if self.tokenizer.bos_token_id is not None:
            prompt_ids = [self.tokenizer.bos_token_id, *prompt_ids]

        beams: list[tuple[tuple[int, ...], float]] = [((), 0.0)]
        completed: list[tuple[tuple[int, ...], float]] = []
        invalid_count = 0
        beam_size = max(self.num_beams, max_k)

        for _ in range(self.max_sid_length):
            beam_candidates: list[tuple[tuple[int, ...], float]] = []
            for prefix, score in beams:
                next_token_ids = self.trie.next_token_ids(prefix)
                if not next_token_ids:
                    invalid_count += 1
                    continue
                next_scores = self._score_next_tokens(prompt_ids, prefix, next_token_ids, beam_size)
                for token_id, token_score in next_scores:
                    next_prefix = (*prefix, token_id)
                    next_score = score + token_score
                    item_id = self.trie.item_id(next_prefix)
                    if item_id is not None:
                        completed.append((next_prefix, next_score))
                    if self.trie.next_token_ids(next_prefix):
                        beam_candidates.append((next_prefix, next_score))
            beams = sorted(beam_candidates, key=lambda row: row[1], reverse=True)[:beam_size]
            if len(completed) >= beam_size and not beams:
                break

        seen_item_ids: set[str] = set()
        generated: list[GeneratedItem] = []
        duplicate_count = 0
        for token_ids, score in sorted(completed, key=lambda row: row[1], reverse=True):
            item_id = self.trie.item_id(token_ids)
            if item_id is None:
                invalid_count += 1
                continue
            if item_id in seen_item_ids:
                duplicate_count += 1
                continue
            seen_item_ids.add(item_id)
            sid = self.tokenizer.decode(token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            generated.append(GeneratedItem(item_id=item_id, sid=sid, score=score))
            if len(generated) == beam_size:
                break

        return PredictionResult(
            items=generated[:max_k],
            stats=GenerationStats(invalid_count=invalid_count, duplicate_count=duplicate_count),
        )

    def _score_next_tokens(
        self,
        prompt_ids: list[int],
        prefix: tuple[int, ...],
        next_token_ids: list[int],
        beam_size: int,
    ) -> list[tuple[int, float]]:
        sampling_params = self.SamplingParams(
            temperature=0.0,
            max_tokens=1,
            logprobs=min(len(next_token_ids), beam_size),
            allowed_token_ids=next_token_ids,
            detokenize=False,
            skip_special_tokens=False,
        )
        prompt = {"prompt_token_ids": [*prompt_ids, *prefix]}
        outputs = self.llm.generate([prompt], sampling_params, use_tqdm=False, lora_request=self.lora_request)
        completion = outputs[0].outputs[0]
        scores: dict[int, float] = {}
        if completion.logprobs:
            for token_id, logprob in completion.logprobs[0].items():
                int_token_id = int(token_id)
                if int_token_id in next_token_ids:
                    value = getattr(logprob, "logprob", logprob)
                    scores[int_token_id] = float(value)
        if not scores and completion.token_ids:
            token_id = int(completion.token_ids[0])
            if token_id in next_token_ids:
                scores[token_id] = float(completion.cumulative_logprob or 0.0)
        return sorted(scores.items(), key=lambda row: row[1], reverse=True)[:beam_size]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate export_sft recommendation artifacts.")
    parser.add_argument("--export-dir", required=True, type=Path, help="Directory produced by export_sft.py.")
    parser.add_argument("--split-path", default=Path("data/processed/rqvae/test.csv"), type=Path)
    parser.add_argument("--train-path", default=Path("data/processed/rqvae/train.csv"), type=Path)
    parser.add_argument("--num-samples", default="null", help="Integer sample count, or null/None for the full test split.")
    parser.add_argument("--sample-seed", default=728, type=int)
    parser.add_argument("--ks", default="3,5,10", help="Comma separated K values.")
    parser.add_argument("--num-beams", default=10, type=int)
    parser.add_argument("--task", default="seq_sid_to_sid", choices=["seq_sid_to_sid", "title_history_to_sid"])
    parser.add_argument("--backend", default="vllm", choices=["vllm", "hf"])
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--torch-dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--output-path", default=None, type=Path)
    parser.add_argument("--history-boundaries", default="20,50", help="Comma separated history length boundaries.")
    parser.add_argument("--include-examples", default=5, type=int)
    return parser.parse_args()


def default_device() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        return "cuda"
    if torch.cuda.device_count() > 1:
        return "cuda:1"
    return "cpu"


def validate_vllm_visible_devices() -> None:
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible_devices:
        raise RuntimeError("vLLM backend requires CUDA_VISIBLE_DEVICES to exclude cuda:0, for example CUDA_VISIBLE_DEVICES=1")
    if "0" in {device.strip() for device in visible_devices.split(",")}:
        raise RuntimeError(f"vLLM backend refuses CUDA_VISIBLE_DEVICES={visible_devices!r} because cuda:0 is excluded for this project")


def validate_hf_device(device: str) -> None:
    if not device.startswith("cuda"):
        return
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices:
        first_visible = visible_devices.split(",")[0].strip()
        if first_visible == "0" and device in {"cuda", "cuda:0"}:
            raise RuntimeError(f"HF backend refuses device={device!r} with CUDA_VISIBLE_DEVICES={visible_devices!r} because it maps to cuda:0")
        return
    if device in {"cuda", "cuda:0"}:
        raise RuntimeError(f"HF backend refuses device={device!r} because cuda:0 is excluded for this project")


def parse_num_samples(value: str | int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        if value <= 0:
            raise ValueError(f"num_samples must be positive or null, got {value}")
        return value
    normalized = value.strip().lower()
    if normalized in {"", "null", "none"}:
        return None
    parsed = int(normalized)
    if parsed <= 0:
        raise ValueError(f"num_samples must be positive or null, got {value}")
    return parsed


def parse_int_list(value: str) -> list[int]:
    parsed = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not parsed:
        raise ValueError(f"Expected at least one integer in {value!r}")
    return parsed


def resolve_dtype_name(value: str, manifest: dict[str, Any]) -> str:
    if value != "auto":
        return value
    return str(manifest.get("torch_dtype", "bfloat16"))


def resolve_torch_dtype(value: str) -> torch.dtype:
    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[value]


def vllm_dtype(value: str) -> str:
    return {"float32": "float32", "float16": "float16", "bfloat16": "bfloat16"}[value]


def load_export_bundle(export_dir: Path) -> ExportBundle:
    manifest_path = export_dir / "manifest.json"
    required_paths = [
        manifest_path,
        export_dir / "tokenizer",
        export_dir / "goodreads.index.json",
        export_dir / "goodreads.item.json",
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Export directory is missing required export_sft artifacts: {missing}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    export_type = manifest.get("export_type")
    if export_type == "adapter":
        model_artifact = export_dir / "adapter"
    elif export_type == "full_model":
        model_artifact = export_dir / "model"
    else:
        raise ValueError(f"Unsupported export_type={export_type!r}; expected adapter or full_model")
    if not model_artifact.exists():
        raise FileNotFoundError(f"Missing exported model artifact: {model_artifact}")

    tokenizer = AutoTokenizer.from_pretrained(export_dir / "tokenizer", trust_remote_code=True)
    if tokenizer.eos_token_id is None:
        raise ValueError("Exported tokenizer must define eos_token_id")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    sid_index = load_sid_index(export_dir / "goodreads.index.json")
    item_sid_map = build_item_sid_map(sid_index)
    trie = SIDTrie(tokenizer=tokenizer, sid_index=sid_index)
    return ExportBundle(
        export_dir=export_dir,
        manifest=manifest,
        tokenizer=tokenizer,
        sid_index=sid_index,
        item_sid_map=item_sid_map,
        trie=trie,
    )


def select_rows(rows: list[dict[str, str]], num_samples: int | None, seed: int) -> list[dict[str, str]]:
    if num_samples is None or num_samples >= len(rows):
        return rows
    return random.Random(seed).sample(rows, num_samples)


def history_slice_name(history_len: int, boundaries: list[int]) -> str:
    sorted_boundaries = sorted(boundaries)
    previous = 0
    for boundary in sorted_boundaries:
        if history_len <= boundary:
            return f"history_len_{previous}_{boundary}" if previous == 0 else f"history_len_{previous + 1}_{boundary}"
        previous = boundary
    return f"history_len_{previous + 1}_plus"


def build_eval_examples(
    rows: list[dict[str, str]],
    item_sid_map: dict[str, str],
    task: str,
    history_boundaries: list[int],
) -> list[EvalExample]:
    examples = []
    for row_index, row in enumerate(rows):
        history_item_ids = [str(item_id) for item_id in parse_list_column(row["history_item_ids"])]
        target_item_id = str(row["item_id"])
        if task == "seq_sid_to_sid":
            history_sids = [item_sid_map[item_id] for item_id in history_item_ids]
            prompt = format_seq_sid_prompt(history_sids)
        else:
            history_titles = [str(title) for title in parse_list_column(row["history_titles"])]
            prompt = format_title_history_prompt(history_titles)
        history_len = len(history_item_ids)
        examples.append(
            EvalExample(
                row_index=row_index,
                user_id=str(row["user_id"]),
                history_item_ids=history_item_ids,
                target_item_id=target_item_id,
                target_sid=item_sid_map[target_item_id],
                prompt=prompt,
                history_len=history_len,
                history_slice=history_slice_name(history_len, history_boundaries),
            )
        )
    if not examples:
        raise ValueError("No eval examples were built")
    return examples


def build_popularity_ranking(train_path: Path, catalog_item_ids: list[str]) -> list[tuple[str, int]]:
    counts = Counter(str(row["item_id"]) for row in load_split_csv(train_path))
    return [(item_id, counts.get(item_id, 0)) for item_id in sorted(catalog_item_ids, key=lambda item_id: (-counts.get(item_id, 0), item_id))]


def baseline_random_predictor(
    item_ids: list[str],
    item_sid_map: dict[str, str],
    rng: random.Random,
    max_k: int,
) -> PredictionResult:
    sampled = rng.sample(item_ids, min(max_k, len(item_ids)))
    return PredictionResult(
        items=[GeneratedItem(item_id=item_id, sid=item_sid_map[item_id], score=0.0) for item_id in sampled],
        stats=GenerationStats(invalid_count=0, duplicate_count=0),
    )


def baseline_popularity_predictor(
    ranked_items: list[tuple[str, int]],
    item_sid_map: dict[str, str],
    max_k: int,
) -> PredictionResult:
    return PredictionResult(
        items=[
            GeneratedItem(item_id=item_id, sid=item_sid_map[item_id], score=float(count))
            for item_id, count in ranked_items[:max_k]
        ],
        stats=GenerationStats(invalid_count=0, duplicate_count=0),
    )


def mean_metrics(metric_rows: list[dict[str, float]]) -> dict[str, float]:
    if not metric_rows:
        return {}
    totals: dict[str, float] = defaultdict(float)
    for row in metric_rows:
        for key, value in row.items():
            totals[key] += value
    return {key: value / len(metric_rows) for key, value in sorted(totals.items())}


def evaluate_method(
    examples: list[EvalExample],
    method_name: str,
    predictor: Any,
    ks: tuple[int, ...],
    max_k: int,
    include_examples: int,
    backend_used: str | None = None,
    fallback_reason: str | None = None,
) -> MethodEvalResult:
    metric_rows = []
    metrics_by_slice: dict[str, list[dict[str, float]]] = defaultdict(list)
    example_predictions: dict[int, list[GeneratedItem]] = {}
    invalid_total = 0
    duplicate_total = 0
    empty_total = 0
    prediction_total = 0

    for index, example in tqdm(enumerate(examples), total=len(examples), desc=f"Evaluating {method_name}"):
        result = predictor(example)
        items = result.items[:max_k]
        predictions = [item.item_id for item in items]
        metrics = ranking_metrics(predictions=predictions, target=example.target_item_id, ks=ks)
        metric_rows.append(metrics)
        metrics_by_slice[example.history_slice].append(metrics)
        invalid_total += result.stats.invalid_count
        duplicate_total += result.stats.duplicate_count
        empty_total += int(not items)
        prediction_total += len(items)
        if index < include_examples:
            example_predictions[index] = items

    row_count = len(examples)
    report: dict[str, Any] = {
        "overall": mean_metrics(metric_rows),
        "by_history_slice": {key: mean_metrics(value) for key, value in sorted(metrics_by_slice.items())},
        "generation": {
            "invalid_rate": invalid_total / row_count,
            "duplicate_rate": duplicate_total / row_count,
            "empty_prediction_rate": empty_total / row_count,
            "avg_prediction_count": prediction_total / row_count,
        },
    }
    if method_name in GENERATION_METHODS:
        report["backend_used"] = backend_used
        if fallback_reason is not None:
            report["backend_fallback_reason"] = fallback_reason
    return MethodEvalResult(report=report, example_predictions=example_predictions)


def load_hf_model(bundle: ExportBundle, method_name: str, dtype_name: str, device: str) -> torch.nn.Module:
    model_kwargs = {"torch_dtype": resolve_torch_dtype(dtype_name), "trust_remote_code": True}
    if method_name == METHOD_BASE:
        model = AutoModelForCausalLM.from_pretrained(bundle.manifest["base_model_path"], **model_kwargs)
        model.resize_token_embeddings(len(bundle.tokenizer))
    elif bundle.manifest["export_type"] == "adapter":
        model = AutoModelForCausalLM.from_pretrained(bundle.manifest["base_model_path"], **model_kwargs)
        model.resize_token_embeddings(len(bundle.tokenizer))
        model = PeftModel.from_pretrained(model, bundle.export_dir / "adapter")
    else:
        model = AutoModelForCausalLM.from_pretrained(bundle.export_dir / "model", **model_kwargs)
        model.resize_token_embeddings(len(bundle.tokenizer))
    model.to(device)
    model.eval()
    return model


def build_hf_generator(bundle: ExportBundle, method_name: str, dtype_name: str, device: str, num_beams: int) -> HFConstrainedGenerator:
    validate_hf_device(device)
    model = load_hf_model(bundle=bundle, method_name=method_name, dtype_name=dtype_name, device=device)
    return HFConstrainedGenerator(
        model=model,
        tokenizer=bundle.tokenizer,
        trie=bundle.trie,
        num_beams=num_beams,
        max_sid_length=max(len(tokens) for tokens in bundle.sid_index.values()),
    )


def build_vllm_generator(bundle: ExportBundle, method_name: str, dtype_name: str, num_beams: int, seed: int) -> VLLMConstrainedGenerator:
    if method_name == METHOD_BASE:
        raise RuntimeError("vLLM cannot resize the base model embedding table for exported SID tokens")
    adapter_path = bundle.export_dir / "adapter" if bundle.manifest["export_type"] == "adapter" else None
    model_path = str(bundle.manifest["base_model_path"] if adapter_path is not None else bundle.export_dir / "model")
    return VLLMConstrainedGenerator(
        model_path=model_path,
        tokenizer_path=bundle.export_dir / "tokenizer",
        tokenizer=bundle.tokenizer,
        trie=bundle.trie,
        num_beams=num_beams,
        max_sid_length=max(len(tokens) for tokens in bundle.sid_index.values()),
        dtype=dtype_name,
        seed=seed,
        adapter_path=adapter_path,
    )


def evaluate_generative_method(
    examples: list[EvalExample],
    bundle: ExportBundle,
    method_name: str,
    args: argparse.Namespace,
    dtype_name: str,
    ks: tuple[int, ...],
    max_k: int,
) -> MethodEvalResult:
    backend_used = args.backend
    fallback_reason = None
    generator: Any = None
    try:
        if args.backend == "vllm":
            validate_vllm_visible_devices()
            generator = build_vllm_generator(
                bundle=bundle,
                method_name=method_name,
                dtype_name=dtype_name,
                num_beams=args.num_beams,
                seed=args.sample_seed,
            )
        else:
            generator = build_hf_generator(bundle, method_name, dtype_name, args.device, args.num_beams)
        return evaluate_method(
            examples=examples,
            method_name=method_name,
            predictor=lambda example: generator.generate(example.prompt, max_k),
            ks=ks,
            max_k=max_k,
            include_examples=args.include_examples,
            backend_used=backend_used,
        )
    except Exception as error:
        if args.backend != "vllm":
            raise
        fallback_reason = f"{type(error).__name__}: {error}"
        backend_used = "hf"
        release_generator(generator)
        generator = build_hf_generator(bundle, method_name, dtype_name, args.device, args.num_beams)
        return evaluate_method(
            examples=examples,
            method_name=method_name,
            predictor=lambda example: generator.generate(example.prompt, max_k),
            ks=ks,
            max_k=max_k,
            include_examples=args.include_examples,
            backend_used=backend_used,
            fallback_reason=fallback_reason,
        )
    finally:
        release_generator(generator)


def release_generator(generator: Any) -> None:
    del generator
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def compact_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "checkpoint_path",
        "base_model_path",
        "train_mode",
        "export_type",
        "torch_dtype",
        "sid_item_count",
        "sid_token_count",
        "sid_min_length",
        "sid_max_length",
        "original_vocab_size",
        "augmented_vocab_size",
    ]
    return {key: manifest[key] for key in keys if key in manifest}


def predictions_to_json(items: list[GeneratedItem]) -> list[dict[str, Any]]:
    return [{"item_id": item.item_id, "sid": item.sid, "score": item.score} for item in items]


def build_example_reports(
    examples: list[EvalExample],
    method_results: dict[str, MethodEvalResult],
    include_examples: int,
) -> list[dict[str, Any]]:
    reports = []
    for index, example in enumerate(examples[:include_examples]):
        reports.append(
            {
                "user_id": example.user_id,
                "history_len": example.history_len,
                "history_slice": example.history_slice,
                "target_item_id": example.target_item_id,
                "target_sid": example.target_sid,
                "predictions": {
                    method_name: predictions_to_json(result.example_predictions.get(index, []))
                    for method_name, result in method_results.items()
                },
            }
        )
    return reports


def default_output_path(export_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("data/reports") / f"eval_{export_dir.name}_{timestamp}.json"


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    bundle = load_export_bundle(args.export_dir)
    dtype_name = resolve_dtype_name(args.torch_dtype, bundle.manifest)
    num_samples = parse_num_samples(args.num_samples)
    ks = tuple(parse_int_list(args.ks))
    max_k = max(ks)
    history_boundaries = parse_int_list(args.history_boundaries)
    rows = select_rows(load_split_csv(args.split_path), num_samples=num_samples, seed=args.sample_seed)
    examples = build_eval_examples(rows, bundle.item_sid_map, args.task, history_boundaries)
    catalog_item_ids = sorted(bundle.sid_index)
    popularity_ranking = build_popularity_ranking(args.train_path, catalog_item_ids)
    rng = random.Random(args.sample_seed)

    method_results: dict[str, MethodEvalResult] = {}
    method_results[METHOD_RANDOM] = evaluate_method(
        examples=examples,
        method_name=METHOD_RANDOM,
        predictor=lambda _example: baseline_random_predictor(catalog_item_ids, bundle.item_sid_map, rng, max_k),
        ks=ks,
        max_k=max_k,
        include_examples=args.include_examples,
    )
    method_results[METHOD_POPULARITY] = evaluate_method(
        examples=examples,
        method_name=METHOD_POPULARITY,
        predictor=lambda _example: baseline_popularity_predictor(popularity_ranking, bundle.item_sid_map, max_k),
        ks=ks,
        max_k=max_k,
        include_examples=args.include_examples,
    )
    method_results[METHOD_BASE] = evaluate_generative_method(
        examples=examples,
        bundle=bundle,
        method_name=METHOD_BASE,
        args=args,
        dtype_name=dtype_name,
        ks=ks,
        max_k=max_k,
    )
    method_results[METHOD_SFT] = evaluate_generative_method(
        examples=examples,
        bundle=bundle,
        method_name=METHOD_SFT,
        args=args,
        dtype_name=dtype_name,
        ks=ks,
        max_k=max_k,
    )

    return {
        "run": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "export_dir": str(args.export_dir),
            "split_path": str(args.split_path),
            "train_path": str(args.train_path),
            "num_samples_requested": num_samples,
            "num_samples_evaluated": len(examples),
            "sample_seed": args.sample_seed,
            "ks": list(ks),
            "num_beams": args.num_beams,
            "task": args.task,
            "backend_requested": args.backend,
            "history_boundaries": history_boundaries,
            "torch_dtype": dtype_name,
            "device": args.device,
        },
        "manifest": compact_manifest(bundle.manifest),
        "methods": {method_name: result.report for method_name, result in method_results.items()},
        "examples": build_example_reports(examples, method_results, args.include_examples),
    }


def print_metric_table(result: dict[str, Any]) -> None:
    ks = result["run"]["ks"]
    metric_keys = [f"HR@{k}" for k in ks] + [f"NDCG@{k}" for k in ks] + [f"MRR@{max(ks)}"]
    print(f"Evaluated {result['run']['num_samples_evaluated']} samples")
    print("method\t" + "\t".join(metric_keys))
    for method_name, report in result["methods"].items():
        overall = report["overall"]
        values = [f"{overall.get(key, 0.0):.6f}" for key in metric_keys]
        print(method_name + "\t" + "\t".join(values))


def main() -> None:
    args = parse_args()
    result = evaluate(args)
    output_path = args.output_path or default_output_path(args.export_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print_metric_table(result)
    print(f"Wrote eval report to {output_path}")


if __name__ == "__main__":
    main()
