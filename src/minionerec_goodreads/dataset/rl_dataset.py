from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightning as L
from torch.utils.data import DataLoader, Dataset

from minionerec_goodreads.dataset.sft_dataset import load_split_csv, parse_list_column
from minionerec_goodreads.utils.sft import build_item_sid_map, load_sid_index
from minionerec_goodreads.utils.sft_generation import format_seq_sid_prompt


@dataclass(frozen=True)
class RLExample:
    prompt: str
    target_item_id: str
    target_sid: str
    history_item_ids: list[str]


class RLDataset(Dataset):
    def __init__(self, examples: list[RLExample]):
        super().__init__()
        if not examples:
            raise ValueError("examples must not be empty")
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> RLExample:
        return self.examples[index]


class RLBatchCollator:
    def __call__(self, batch: list[RLExample]) -> dict[str, Any]:
        return {
            "prompts": [example.prompt for example in batch],
            "target_item_ids": [example.target_item_id for example in batch],
            "target_sids": [example.target_sid for example in batch],
            "history_item_ids": [example.history_item_ids for example in batch],
        }


def build_rl_examples(rows: list[dict[str, str]], item_sid_map: dict[str, str]) -> list[RLExample]:
    examples: list[RLExample] = []
    for row in rows:
        history_item_ids = [str(item_id) for item_id in parse_list_column(row["history_item_ids"])]
        target_item_id = str(row["item_id"])
        history_sids = [item_sid_map[item_id] for item_id in history_item_ids]
        target_sid = item_sid_map[target_item_id]
        examples.append(
            RLExample(
                prompt=format_seq_sid_prompt(history_sids),
                target_item_id=target_item_id,
                target_sid=target_sid,
                history_item_ids=history_item_ids,
            )
        )
    if not examples:
        raise ValueError("No RL examples were built")
    return examples


def limit_examples(examples: list[RLExample], max_samples: int | None, seed: int) -> list[RLExample]:
    if max_samples is None or max_samples >= len(examples):
        return examples
    shuffled = examples.copy()
    random.Random(seed).shuffle(shuffled)  # noqa: S311
    return shuffled[:max_samples]


class RLDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_path: str,
        valid_path: str,
        test_path: str,
        sid_index_path: str,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        persistent_workers: bool = False,
        max_train_samples: int | None = None,
        max_valid_samples: int | None = None,
        max_test_samples: int | None = None,
        seed: int = 728,
        pretrained_model_name_or_path: str | None = None,
        sft_export_dir: str | None = None,
    ):
        super().__init__()
        self.train_path = Path(train_path)
        self.valid_path = Path(valid_path)
        self.test_path = Path(test_path)
        self.sid_index_path = Path(sid_index_path)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers and num_workers > 0
        self.max_train_samples = max_train_samples
        self.max_valid_samples = max_valid_samples
        self.max_test_samples = max_test_samples
        self.seed = seed
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.sft_export_dir = sft_export_dir

        self.collator = RLBatchCollator()
        self.data_train: RLDataset | None = None
        self.data_valid: RLDataset | None = None
        self.data_test: RLDataset | None = None

    def prepare_data(self) -> None:
        for path in [self.train_path, self.valid_path, self.test_path, self.sid_index_path]:
            if not path.exists():
                raise FileNotFoundError(path)

    def setup(self, stage: str | None = None) -> None:
        _ = stage
        if self.data_train is not None:
            return

        sid_index = load_sid_index(self.sid_index_path)
        item_sid_map = build_item_sid_map(sid_index)

        train_examples = build_rl_examples(load_split_csv(self.train_path), item_sid_map)
        valid_examples = build_rl_examples(load_split_csv(self.valid_path), item_sid_map)
        test_examples = build_rl_examples(load_split_csv(self.test_path), item_sid_map)

        self.data_train = RLDataset(limit_examples(train_examples, self.max_train_samples, self.seed))
        self.data_valid = RLDataset(limit_examples(valid_examples, self.max_valid_samples, self.seed + 1))
        self.data_test = RLDataset(limit_examples(test_examples, self.max_test_samples, self.seed + 2))

    def train_dataloader(self) -> DataLoader:
        if self.data_train is None:
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
        if self.data_valid is None:
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
        if self.data_test is None:
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
