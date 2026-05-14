from __future__ import annotations

import ast
import random
from collections import defaultdict

import lightning as L

from minionerec_goodreads.metrics.rec import ranking_metrics
from minionerec_goodreads.utils.sft import build_item_sid_map
from minionerec_goodreads.utils.sft_generation import (
    constrained_sid_beam_search,
    format_seq_sid_prompt,
    format_title_history_prompt,
)
from minionerec_goodreads.utils.sid_trie import SIDTrie


class SFTRecommendationValidationCallback(L.Callback):
    def __init__(
        self,
        num_samples: int | None = 256,
        num_beams: int = 10,
        ks: list[int] | tuple[int, ...] = (3, 5, 10),
        task: str = "seq_sid_to_sid",
        seed: int = 728,
    ):
        super().__init__()
        if num_samples is not None and num_samples <= 0:
            raise ValueError(f"num_samples must be positive or None, got {num_samples}")
        if num_beams <= 0:
            raise ValueError(f"num_beams must be positive, got {num_beams}")
        if task not in {"seq_sid_to_sid", "title_history_to_sid"}:
            raise ValueError(f"Unsupported recommendation validation task: {task}")
        self.num_samples = num_samples
        self.num_beams = num_beams
        self.ks = tuple(int(k) for k in ks)
        self.task = task
        self.seed = seed
        if not self.ks or any(k <= 0 for k in self.ks):
            raise ValueError(f"ks must contain positive integers, got {ks}")
        self._trie: SIDTrie | None = None
        self._item_sid_map: dict[str, str] | None = None
        self._max_sid_length: int | None = None

    def _enabled(self, trainer: L.Trainer) -> bool:
        return self.num_samples is not None and not trainer.sanity_checking

    def _valid_rows(self, trainer: L.Trainer) -> list[dict[str, str]]:
        datamodule = trainer.datamodule
        if datamodule is None:
            raise RuntimeError("SFT recommendation validation requires a datamodule")
        rows = getattr(datamodule, "valid_rows", None)
        if rows is None:
            raise RuntimeError("Call SFTDataModule.setup() before recommendation validation")
        return rows

    def _sample_rows(self, rows: list[dict[str, str]], epoch: int) -> list[dict[str, str]]:
        sample_size = min(self.num_samples, len(rows))
        return random.Random(self.seed + epoch).sample(rows, sample_size)  # noqa: S311

    def _prepare_sid_helpers(self, pl_module: L.LightningModule) -> tuple[dict[str, str], SIDTrie, int]:
        sid_index = getattr(pl_module, "sid_index", None)
        tokenizer = getattr(pl_module, "tokenizer", None)
        if sid_index is None or tokenizer is None:
            raise RuntimeError("SFT recommendation validation requires tokenizer and sid_index on the module")
        if self._item_sid_map is None:
            self._item_sid_map = build_item_sid_map(sid_index)
        if self._trie is None:
            self._trie = SIDTrie(tokenizer=tokenizer, sid_index=sid_index)
        if self._max_sid_length is None:
            self._max_sid_length = max(len(tokens) for tokens in sid_index.values())
        return self._item_sid_map, self._trie, self._max_sid_length

    def _prompt_and_target(self, row: dict[str, str], item_sid_map: dict[str, str]) -> tuple[str, str]:
        target_item_id = str(row["item_id"])
        if self.task == "seq_sid_to_sid":
            history_item_ids = [str(item_id) for item_id in ast.literal_eval(row["history_item_ids"])]
            history_sids = [item_sid_map[item_id] for item_id in history_item_ids]
            return format_seq_sid_prompt(history_sids), target_item_id
        history_titles = [str(title) for title in ast.literal_eval(row["history_titles"])]
        return format_title_history_prompt(history_titles), target_item_id

    def _mean_metrics(self, metric_rows: list[dict[str, float]]) -> dict[str, float]:
        totals: dict[str, float] = defaultdict(float)
        for row in metric_rows:
            for key, value in row.items():
                totals[key] += value
        return {key: value / len(metric_rows) for key, value in sorted(totals.items())}

    def _evaluate_rows(self, pl_module: L.LightningModule, rows: list[dict[str, str]]) -> dict[str, float]:
        model = getattr(pl_module, "model", None)
        if model is None:
            raise RuntimeError("SFT recommendation validation requires model on the module")
        item_sid_map, trie, max_sid_length = self._prepare_sid_helpers(pl_module)
        max_k = max(self.ks)
        metric_rows = []
        invalid_total = 0
        duplicate_total = 0
        was_training = model.training
        model.eval()
        try:
            for row in rows:
                prompt, target_item_id = self._prompt_and_target(row, item_sid_map)
                generated, stats = constrained_sid_beam_search(
                    model=model,
                    tokenizer=pl_module.tokenizer,
                    trie=trie,
                    prompt=prompt,
                    num_beams=max(self.num_beams, max_k),
                    max_sid_length=max_sid_length,
                )
                predictions = [item.item_id for item in generated[:max_k]]
                metric_rows.append(ranking_metrics(predictions=predictions, target=target_item_id, ks=self.ks))
                invalid_total += stats.invalid_count
                duplicate_total += stats.duplicate_count
        finally:
            if was_training:
                model.train()
        metrics = self._mean_metrics(metric_rows)
        metrics["invalid_rate"] = invalid_total / len(rows)
        metrics["duplicate_rate"] = duplicate_total / len(rows)
        metrics["num_samples"] = float(len(rows))
        return metrics

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if not self._enabled(trainer):
            return
        rows = self._sample_rows(self._valid_rows(trainer), trainer.current_epoch)
        metrics = self._evaluate_rows(pl_module, rows)
        pl_module.log_dict(
            {f"val/rec_{key}": value for key, value in metrics.items()},
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
