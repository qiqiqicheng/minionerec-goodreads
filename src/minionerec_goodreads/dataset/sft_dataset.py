"""
build_seq_samples:
    - seq_sid_to_sid: given history (sid), predict next item (sid)
build_item_task_samples:
    - sid_to_title: given item (sid), predict title
    - title_to_sid: given item (title), predict sid
build_fusion_samples:
    - title_history_to_sid: given history (titles), predict next item (sid)
"""

from __future__ import annotations

import ast
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerBase

from minionerec_goodreads.utils.sft import SFT_TASK_TO_ID, build_item_sid_map, build_tokenizer

EXPECTED_SPLIT_COLUMNS = [
    "user_id",
    "history_book_ids",
    "book_id",
    "history_item_ids",
    "item_id",
    "history_titles",
    "item_title",
    "history_ratings",
    "rating",
    "history_timestamps",
    "timestamp",
]


def load_split_csv(path: str | Path) -> list[dict[str, str]]:
    csv_path = Path(path)
    with csv_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        actual_columns = reader.fieldnames or []
        if actual_columns != EXPECTED_SPLIT_COLUMNS:
            raise ValueError(
                f"Unexpected split columns in {csv_path}: expected {EXPECTED_SPLIT_COLUMNS}, got {actual_columns}"
            )
        rows = list(reader)
    if not rows:
        raise ValueError(f"Empty split file: {csv_path}")
    return rows


def load_item_payload(path: str | Path) -> dict[str, dict[str, Any]]:
    json_path = Path(path)
    with json_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not payload:
        raise ValueError(f"Empty item payload: {json_path}")
    return payload


def parse_list_column(text: str) -> list[Any]:
    value = ast.literal_eval(text)
    if not isinstance(value, list):
        raise TypeError(f"Expected list column, got: {type(value)}")
    return value

def format_quoted_titles(titles: list[str]) -> str:
    if not titles:
        raise ValueError("History titles must not be empty")
    return ", ".join(f'"{title}"' for title in titles)


@dataclass
class SFTSample:
    task: str
    instruction: str
    user_input: str
    response: str


def _build_sample(task: str, instruction: str, user_input: str, response: str) -> SFTSample:
    if not response:
        raise ValueError(f"Empty response for task={task}")
    return SFTSample(task=task, instruction=instruction, user_input=user_input, response=response)


def build_seq_samples(
    rows: list[dict[str, str]],
    item_sid_map: dict[str, str]
) -> list[SFTSample]:
    """
    seq_sid_to_sid: given history (sid), predict next item (sid)

    Returns:
        list[SFTSample]: _description_
    """
    samples: list[SFTSample] = []
    instruction = "Predict the semantic ID of the next book from the chronological interaction history."
    for row in rows:
        history_item_ids = [str(item_id) for item_id in parse_list_column(row["history_item_ids"])]
        target_item_id = str(row["item_id"])
        history_sids = [item_sid_map[item_id] for item_id in history_item_ids]  # [H]
        target_sid = item_sid_map[target_item_id]
        history_text = ", ".join(history_sids)
        user_input = (
            f"The user has interacted with books {history_text} in chronological order. "
            "Predict the semantic ID of the next book."
        )
        samples.append(_build_sample("seq_sid_to_sid", instruction, user_input, f"{target_sid}\n"))
    if not samples:
        raise ValueError("No sequential SFT samples were built")
    return samples


def build_item_task_samples(
    rows: list[dict[str, str]],
    item_payload: dict[str, dict[str, Any]],
    item_sid_map: dict[str, str],
) -> list[SFTSample]:
    """
    sid_to_title: given item (sid), predict title
    title_to_sid: given item (title), predict sid

    Returns:
        list[SFTSample]: _description_
    """
    item_ids = sorted({str(row["item_id"]) for row in rows})
    samples: list[SFTSample] = []
    instruction = "Answer the question about the mapping between a book title and its semantic ID."
    for item_id in item_ids:
        title = str(item_payload[item_id]["title"]).strip()
        if not title:
            raise ValueError(f"Empty title for item_id={item_id}")
        sid = item_sid_map[item_id]
        samples.append(
            _build_sample(
                "sid_to_title",
                instruction,
                f'What is the title of the book with semantic ID "{sid}"?',
                f"{title}\n",
            )
        )
        samples.append(
            _build_sample(
                "title_to_sid",
                instruction,
                f'What is the semantic ID of the book titled "{title}"?',
                f"{sid}\n",
            )
        )
    if not samples:
        raise ValueError("No item SFT samples were built")
    return samples


def build_fusion_samples(
    rows: list[dict[str, str]],
    item_sid_map: dict[str, str]
) -> list[SFTSample]:
    """
    title_history_to_sid: given history (titles), predict next item (sid)

    Args:
        rows (list[dict[str, str]]): _description_
        item_sid_map (dict[str, str]): _description_

    Raises:
        ValueError: _description_

    Returns:
        list[SFTSample]: _description_
    """
    samples: list[SFTSample] = []
    instruction = "Predict the semantic ID of the next book from the chronological title history."
    for row in rows:
        history_titles = [str(title) for title in parse_list_column(row["history_titles"])]
        target_item_id = str(row["item_id"])
        user_input = (
            f"The user has interacted with the following book titles in chronological order: "
            f"{format_quoted_titles(history_titles)}. Predict the semantic ID of the next book."
        )
        target_sid = item_sid_map[target_item_id]
        samples.append(_build_sample("title_history_to_sid", instruction, user_input, f"{target_sid}\n"))
    if not samples:
        raise ValueError("No fusion SFT samples were built")
    return samples


def _repeat_samples(samples: list[SFTSample], repeat: int) -> list[SFTSample]:
    if repeat <= 0:
        raise ValueError(f"repeat must be positive, got {repeat}")
    return samples * repeat


def _limit_samples(samples: list[SFTSample], max_samples: int | None, seed: int) -> list[SFTSample]:
    if max_samples is None or max_samples >= len(samples):
        return samples
    shuffled = samples.copy()
    random.Random(seed).shuffle(shuffled)  # noqa: S311
    return shuffled[:max_samples]


class TokenizedSFTDataset(Dataset):
    def __init__(self, samples: list[SFTSample], tokenizer: PreTrainedTokenizerBase, max_length: int):
        super().__init__()
        if not samples:
            raise ValueError("samples must not be empty")
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        """
        Args:
            index (int): _description_

        Returns:
            dict[str, list[int]]:
                - input_ids
                - attention_mask
                - labels (with prompt tokens masked as -100)
        """
        sample = self.samples[index]
        prompt_text = (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{sample.instruction}\n\n"
            f"### User Input:\n{sample.user_input}\n\n"
            "### Response:\n"
        )
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)  # [P]
        response_ids = self.tokenizer.encode(sample.response, add_special_tokens=False)  # [R]

        if self.tokenizer.bos_token_id is not None:
            prompt_ids = [self.tokenizer.bos_token_id, *prompt_ids]
        if self.tokenizer.eos_token_id is not None:
            response_ids = [*response_ids, self.tokenizer.eos_token_id]

        input_ids = [*prompt_ids, *response_ids]  # [T]
        attention_mask = [1] * len(input_ids)  # [T]
        labels = [-100] * len(prompt_ids) + response_ids  # [T]

        return {
            "input_ids": input_ids[-self.max_length :],
            "attention_mask": attention_mask[-self.max_length :],  # for attention score compute
            "labels": labels[-self.max_length :],  # for loss compute
            "task_id": SFT_TASK_TO_ID[sample.task],
        }


class SFTBatchCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        max_length = max(len(row["input_ids"]) for row in batch)
        input_ids = []
        attention_mask = []
        labels = []
        task_ids = []

        for row in batch:
            pad_len = max_length - len(row["input_ids"])
            input_ids.append(row["input_ids"] + [self.pad_token_id] * pad_len)  # [T]
            attention_mask.append(row["attention_mask"] + [0] * pad_len)  # [T]
            labels.append(row["labels"] + [-100] * pad_len)  # [T]
            task_ids.append(row["task_id"])

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),  # [B, T]
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),  # [B, T]
            "labels": torch.tensor(labels, dtype=torch.long),  # [B, T]
            "task_id": torch.tensor(task_ids, dtype=torch.long),  # [B]
        }


class SFTDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_path: str,
        valid_path: str,
        test_path: str,
        item_path: str,
        sid_index_path: str,
        pretrained_model_name_or_path: str,
        max_length: int,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        persistent_workers: bool = False,
        include_seq_task: bool = True,
        include_item_task: bool = True,
        include_fusion_task: bool = True,
        seq_repeat: int = 1,
        item_repeat: int = 1,
        fusion_repeat: int = 1,
        max_train_samples: int | None = None,
        max_valid_samples: int | None = None,
        max_test_samples: int | None = None,
        seed: int = 728,
    ):
        super().__init__()
        self.train_path = Path(train_path)
        self.valid_path = Path(valid_path)
        self.test_path = Path(test_path)
        self.item_path = Path(item_path)
        self.sid_index_path = Path(sid_index_path)
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers and num_workers > 0
        self.include_seq_task = include_seq_task
        self.include_item_task = include_item_task
        self.include_fusion_task = include_fusion_task
        self.seq_repeat = seq_repeat
        self.item_repeat = item_repeat
        self.fusion_repeat = fusion_repeat
        self.max_train_samples = max_train_samples
        self.max_valid_samples = max_valid_samples
        self.max_test_samples = max_test_samples
        self.seed = seed

        self.tokenizer: PreTrainedTokenizerBase | None = None
        self.collator: SFTBatchCollator | None = None
        self.data_train: TokenizedSFTDataset | None = None
        self.data_valid: TokenizedSFTDataset | None = None
        self.data_test: TokenizedSFTDataset | None = None

    def prepare_data(self) -> None:
        for path in [self.train_path, self.valid_path, self.test_path, self.item_path, self.sid_index_path]:
            if not path.exists():
                raise FileNotFoundError(path)

    def setup(self, stage: str | None = None) -> None:
        _ = stage
        if self.data_train is not None:
            return

        train_rows = load_split_csv(self.train_path)
        valid_rows = load_split_csv(self.valid_path)
        test_rows = load_split_csv(self.test_path)
        item_payload = load_item_payload(self.item_path)
        tokenizer, sid_index, _ = build_tokenizer(
            pretrained_model_name_or_path=self.pretrained_model_name_or_path,
            sid_index_path=self.sid_index_path,
        )
        item_sid_map = build_item_sid_map(sid_index)

        self.tokenizer = tokenizer
        self.collator = SFTBatchCollator(pad_token_id=tokenizer.pad_token_id)

        train_samples = self._build_split_samples(train_rows, item_payload, item_sid_map, split_seed=self.seed)
        valid_samples = self._build_split_samples(valid_rows, item_payload, item_sid_map, split_seed=self.seed + 1)
        test_samples = self._build_split_samples(test_rows, item_payload, item_sid_map, split_seed=self.seed + 2)

        train_samples = _limit_samples(train_samples, self.max_train_samples, self.seed)
        valid_samples = _limit_samples(valid_samples, self.max_valid_samples, self.seed + 1)
        test_samples = _limit_samples(test_samples, self.max_test_samples, self.seed + 2)

        self.data_train = TokenizedSFTDataset(train_samples, tokenizer=tokenizer, max_length=self.max_length)
        self.data_valid = TokenizedSFTDataset(valid_samples, tokenizer=tokenizer, max_length=self.max_length)
        self.data_test = TokenizedSFTDataset(test_samples, tokenizer=tokenizer, max_length=self.max_length)

    def _build_split_samples(
        self,
        rows: list[dict[str, str]],
        item_payload: dict[str, dict[str, Any]],
        item_sid_map: dict[str, str],
        split_seed: int,
    ) -> list[SFTSample]:
        samples: list[SFTSample] = []
        if self.include_seq_task:
            samples.extend(_repeat_samples(build_seq_samples(rows, item_sid_map), self.seq_repeat))
        if self.include_item_task:
            samples.extend(_repeat_samples(build_item_task_samples(rows, item_payload, item_sid_map), self.item_repeat))
        if self.include_fusion_task:
            samples.extend(_repeat_samples(build_fusion_samples(rows, item_sid_map), self.fusion_repeat))
        if not samples:
            raise ValueError("At least one SFT task must be enabled")
        random.Random(split_seed).shuffle(samples)  # noqa: S311
        return samples

    def train_dataloader(self) -> DataLoader:
        if self.data_train is None or self.collator is None:
            raise RuntimeError("Call setup() before requesting train_dataloader()")
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            collate_fn=self.collator,
        )

    def val_dataloader(self) -> DataLoader:
        if self.data_valid is None or self.collator is None:
            raise RuntimeError("Call setup() before requesting val_dataloader()")
        return DataLoader(
            dataset=self.data_valid,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            collate_fn=self.collator,
        )

    def test_dataloader(self) -> DataLoader:
        if self.data_test is None or self.collator is None:
            raise RuntimeError("Call setup() before requesting test_dataloader()")
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            collate_fn=self.collator,
        )
