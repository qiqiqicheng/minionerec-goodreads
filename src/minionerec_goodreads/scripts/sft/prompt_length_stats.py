from __future__ import annotations

import ast
import math
from collections import defaultdict
from dataclasses import dataclass
from statistics import fmean

import hydra
from omegaconf import DictConfig, OmegaConf
from transformers import PreTrainedTokenizerBase

from minionerec_goodreads.dataset.sft_dataset import SFTSample, TokenizedSFTDataset
from minionerec_goodreads.utils.sft_generation import format_sft_prompt

if not OmegaConf.has_resolver("eval"):
    OmegaConf.register_new_resolver("eval", ast.literal_eval)


@dataclass(frozen=True)
class SampleLengths:
    task: str
    prompt: int
    response: int
    total: int


def quantile(sorted_values: list[int], q: float) -> float:
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    index = (len(sorted_values) - 1) * q
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return float(sorted_values[lower])
    lower_value = sorted_values[lower] * (upper - index)
    upper_value = sorted_values[upper] * (index - lower)
    return float(lower_value + upper_value)


def fmt(value: int | float | str) -> str:
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        if value.is_integer():
            return f"{int(value):,}"
        return f"{value:,.1f}"
    return value


def table(headers: list[str], rows: list[list[int | float | str]]) -> str:
    header = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(fmt(cell) for cell in row) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def length_of(sample: SFTSample, tokenizer: PreTrainedTokenizerBase) -> SampleLengths:
    prompt = format_sft_prompt(sample.instruction, sample.user_input)
    prompt_len = len(tokenizer.encode(prompt, add_special_tokens=False))
    response_len = len(tokenizer.encode(sample.response, add_special_tokens=False))
    if tokenizer.bos_token_id is not None:
        prompt_len += 1
    if tokenizer.eos_token_id is not None:
        response_len += 1
    return SampleLengths(
        task=sample.task,
        prompt=prompt_len,
        response=response_len,
        total=prompt_len + response_len,
    )


def collect_lengths(dataset: TokenizedSFTDataset, tokenizer: PreTrainedTokenizerBase) -> list[SampleLengths]:
    return [length_of(sample, tokenizer) for sample in dataset.samples]


def summarize(values: list[int], max_length: int) -> list[int | float | str]:
    sorted_values = sorted(values)
    over = sum(value > max_length for value in sorted_values)
    return [
        len(sorted_values),
        sorted_values[0],
        quantile(sorted_values, 0.50),
        quantile(sorted_values, 0.90),
        quantile(sorted_values, 0.95),
        quantile(sorted_values, 0.99),
        sorted_values[-1],
        fmean(sorted_values),
        f"{over:,} ({over / len(sorted_values) * 100:.2f}%)",
    ]


def summary_rows(
    split_name: str,
    lengths: list[SampleLengths],
    field_name: str,
    max_length: int,
) -> list[list[int | float | str]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for row in lengths:
        grouped["all"].append(getattr(row, field_name))
        grouped[row.task].append(getattr(row, field_name))
    task_names = ["all", *sorted(task for task in grouped if task != "all")]
    return [[split_name, task, *summarize(grouped[task], max_length)] for task in task_names]


def require_dataset(dataset: TokenizedSFTDataset | None, split_name: str) -> TokenizedSFTDataset:
    if dataset is None:
        raise RuntimeError(f"Missing {split_name} dataset after datamodule.setup()")
    return dataset


@hydra.main(version_base="1.3", config_path="../../configs", config_name="train_sft.yaml")
def main(cfg: DictConfig) -> None:
    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.prepare_data()
    datamodule.setup(stage=None)
    if datamodule.tokenizer is None:
        raise RuntimeError("Missing tokenizer after datamodule.setup()")

    max_length = int(cfg.data.max_length)
    split_datasets = {
        "train": require_dataset(datamodule.data_train, "train"),
        "valid": require_dataset(datamodule.data_valid, "valid"),
        "test": require_dataset(datamodule.data_test, "test"),
    }
    split_lengths = {
        split_name: collect_lengths(dataset, datamodule.tokenizer)
        for split_name, dataset in split_datasets.items()
    }

    print(f"Tokenizer: {cfg.data.pretrained_model_name_or_path}")
    print(f"Current data.max_length: {max_length:,}")
    print(
        "Sample caps: "
        f"train={cfg.data.max_train_samples}, valid={cfg.data.max_valid_samples}, test={cfg.data.max_test_samples}"
    )
    print()

    headers = ["split", "task", "count", "min", "p50", "p90", "p95", "p99", "max", "mean", f">{max_length}"]
    prompt_rows = [
        row
        for split_name, lengths in split_lengths.items()
        for row in summary_rows(split_name, lengths, "prompt", max_length)
    ]
    total_rows = [
        row
        for split_name, lengths in split_lengths.items()
        for row in summary_rows(split_name, lengths, "total", max_length)
    ]

    print("Prompt token lengths (includes BOS if tokenizer has one)")
    print(table(headers, prompt_rows))
    print()
    print("Total token lengths (prompt + response, includes EOS if tokenizer has one)")
    print(table(headers, total_rows))
    print()
    print("Override data.max_*_samples=null to scan the full split when the current config uses caps.")


if __name__ == "__main__":
    main()
