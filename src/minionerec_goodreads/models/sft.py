from __future__ import annotations

import logging
from dataclasses import dataclass
from importlib.util import find_spec
from typing import Any, Callable

import lightning as L
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

from minionerec_goodreads.utils.sft import SFT_TASK_NAMES, build_tokenizer

log = logging.getLogger(__name__)


def _resolve_dtype(name: str) -> torch.dtype:
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if name not in dtype_map:
        raise ValueError(f"Unsupported dtype name: {name}")
    return dtype_map[name]


def _import_bitsandbytes() -> None:
    if find_spec("bitsandbytes") is None:
        raise ImportError("QLoRA requires bitsandbytes to be installed")


def _import_peft() -> tuple[Any, Any, Any, Any]:
    try:
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    except ImportError as error:
        raise ImportError("LoRA training requires peft to be installed") from error
    return LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training


@dataclass
class TrainableRange:
    start: int
    end: int


class SFTModule(L.LightningModule):
    def __init__(
        self,
        pretrained_model_name_or_path: str,
        sid_index_path: str,
        train_mode: str,
        warmup_ratio: float,
        gradient_checkpointing: bool,
        torch_dtype: str,
        optimizer: Callable,
        scheduler: Callable | None,
        attn_implementation: str | None = None,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: list[str] | None = None,
        load_in_4bit: bool = False,
        bnb_4bit_compute_dtype: str = "bfloat16",
        bnb_4bit_quant_type: str = "nf4",
        bnb_4bit_use_double_quant: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["optimizer", "scheduler"])
        self._optimizer = optimizer
        self._scheduler = scheduler

        self.tokenizer, sid_index, self.original_vocab_size = build_tokenizer(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            sid_index_path=sid_index_path,
        )
        self.sid_index = sid_index
        self.trainable_range: TrainableRange | None = None

        if train_mode == "qlora":
            self.model = self._build_lora_model()
        else:
            self.model = self._build_dense_model()

        self._configure_trainable_parameters()
        self._log_trainable_parameters()

    def _build_dense_model(self) -> torch.nn.Module:
        model_kwargs: dict[str, Any] = {
            "torch_dtype": _resolve_dtype(self.hparams.torch_dtype),
            "trust_remote_code": True,
        }
        if self.hparams.attn_implementation is not None:
            model_kwargs["attn_implementation"] = self.hparams.attn_implementation
        model = AutoModelForCausalLM.from_pretrained(self.hparams.pretrained_model_name_or_path, **model_kwargs)
        model.resize_token_embeddings(len(self.tokenizer))
        if self.hparams.gradient_checkpointing:
            model.gradient_checkpointing_enable()
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            model.config.use_cache = False
        return model

    def _build_lora_model(self) -> torch.nn.Module:
        LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training = _import_peft()

        model_kwargs: dict[str, Any] = {
            "torch_dtype": _resolve_dtype(self.hparams.torch_dtype),
            "trust_remote_code": True,
        }
        if self.hparams.attn_implementation is not None:
            model_kwargs["attn_implementation"] = self.hparams.attn_implementation
        if self.hparams.load_in_4bit:
            if not torch.cuda.is_available():
                raise RuntimeError("4-bit LoRA loading requires CUDA. Set load_in_4bit=False or use full_finetune/new_token_only.")
            _import_bitsandbytes()
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=_resolve_dtype(self.hparams.bnb_4bit_compute_dtype),
                bnb_4bit_quant_type=self.hparams.bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=self.hparams.bnb_4bit_use_double_quant,
            )

        model = AutoModelForCausalLM.from_pretrained(self.hparams.pretrained_model_name_or_path, **model_kwargs)
        model.resize_token_embeddings(len(self.tokenizer))
        if self.hparams.load_in_4bit:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=self.hparams.gradient_checkpointing)
        elif self.hparams.gradient_checkpointing:
            model.gradient_checkpointing_enable()
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            model.config.use_cache = False
        lora_config = LoraConfig(
            r=self.hparams.lora_r,
            lora_alpha=self.hparams.lora_alpha,
            lora_dropout=self.hparams.lora_dropout,
            target_modules=self.hparams.lora_target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        model.config.use_cache = False
        return model

    def _configure_trainable_parameters(self) -> None:
        if self.hparams.train_mode == "full_finetune":
            self._enable_full_finetune()
            return
        if self.hparams.train_mode == "qlora":
            self._validate_qlora_parameters()
            return
        if self.hparams.train_mode == "new_token_only":
            self._enable_new_token_only()
            return
        raise ValueError(f"Unsupported train_mode: {self.hparams.train_mode}")

    def _enable_full_finetune(self) -> None:
        for parameter in self.model.parameters():
            parameter.requires_grad = True

    def _validate_qlora_parameters(self) -> None:
        self._enable_new_token_rows()
        trainable_count = sum(parameter.requires_grad for parameter in self.model.parameters())
        if trainable_count == 0:
            raise ValueError("LoRA mode did not expose any trainable parameters")

    def _enable_new_token_only(self) -> None:
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        self._enable_new_token_rows()
        trainable_count = sum(parameter.requires_grad for parameter in self.model.parameters())
        if trainable_count == 0:
            raise ValueError("new_token_only mode left the model with zero trainable parameters")

    def _enable_new_token_rows(self) -> TrainableRange:
        if len(self.tokenizer) <= self.original_vocab_size:
            raise ValueError("SID training requires tokenizer resize with new SID tokens")

        embedding = self.model.get_input_embeddings()
        if embedding is None:
            raise ValueError("Model must expose input embeddings")
        embedding.weight.requires_grad = True
        trainable_range = TrainableRange(start=self.original_vocab_size, end=embedding.weight.shape[0])
        self._register_row_mask(embedding.weight, trainable_range)
        self._enable_output_embedding_if_needed(embedding.weight, trainable_range)
        self.trainable_range = trainable_range
        return trainable_range

    def _enable_output_embedding_if_needed(
        self,
        input_embedding_weight: torch.nn.Parameter,
        trainable_range: TrainableRange,
    ) -> None:
        output_embedding = self.model.get_output_embeddings()
        if output_embedding is None or output_embedding.weight is input_embedding_weight:
            return
        output_embedding.weight.requires_grad = True
        self._register_row_mask(output_embedding.weight, trainable_range)

    def _register_row_mask(self, weight: torch.nn.Parameter, trainable_range: TrainableRange) -> None:
        def mask_gradient(grad: torch.Tensor) -> torch.Tensor:
            grad[: trainable_range.start].zero_()  # [V_old, H]
            return grad

        weight.register_hook(mask_gradient)
        weight._sid_row_masked = True

    def _log_trainable_parameters(self) -> None:
        total_params = sum(parameter.numel() for parameter in self.model.parameters())
        trainable_params = sum(parameter.numel() for parameter in self.model.parameters() if parameter.requires_grad)
        ratio = 100 * trainable_params / total_params
        mode = "lora_4bit" if self.hparams.train_mode == "qlora" and self.hparams.load_in_4bit else self.hparams.train_mode
        if self.hparams.train_mode == "qlora" and not self.hparams.load_in_4bit:
            mode = "lora"
        log.info("SFT train mode=%s trainable_params=%s total_params=%s ratio=%.4f%%", mode, trainable_params, total_params, ratio)
        if self.trainable_range is not None:
            log.info("Trainable SID token rows=[%s, %s)", self.trainable_range.start, self.trainable_range.end)

    def forward(self, batch: dict[str, torch.Tensor]) -> Any:
        model_batch = {key: value for key, value in batch.items() if key in {"input_ids", "attention_mask", "labels"}}
        return self.model(**model_batch)

    def _task_losses(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        task_ids: torch.Tensor,
    ) -> dict[str, tuple[torch.Tensor, int]]:
        shifted_logits = logits[..., :-1, :].contiguous()  # [B, T - 1, V]
        shifted_labels = labels[..., 1:].contiguous()  # [B, T - 1]
        token_losses = F.cross_entropy(
            shifted_logits.view(-1, shifted_logits.shape[-1]),
            shifted_labels.view(-1),
            ignore_index=-100,
            reduction="none",
        ).view_as(shifted_labels)
        valid_mask = shifted_labels.ne(-100)
        token_counts = valid_mask.sum(dim=1).clamp_min(1)  # [B]
        sample_losses = (token_losses * valid_mask).sum(dim=1) / token_counts  # [B]
        losses = {}
        for task_id, task_name in enumerate(SFT_TASK_NAMES):
            sample_mask = task_ids.eq(task_id)  # [B]
            if sample_mask.any():
                losses[task_name] = (sample_losses[sample_mask].mean(), int(sample_mask.sum().item()))
        return losses

    def _shared_step(self, batch: dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        outputs = self(batch)
        loss = outputs.loss
        if not torch.isfinite(loss):
            raise ValueError(f"Non-finite {stage} loss: {loss}")
        batch_size = batch["input_ids"].shape[0]
        log_step = stage == "train"
        if log_step:
            self.log(f"{stage}/loss_step", loss, prog_bar=True, on_step=True, on_epoch=False, batch_size=batch_size)
        self.log(f"{stage}/loss", loss, prog_bar=(stage != "test"), on_step=False, on_epoch=True, batch_size=batch_size)
        if "task_id" in batch:
            with torch.no_grad():
                task_losses = self._task_losses(outputs.logits, batch["labels"], batch["task_id"])
            recommendation_loss_sum = None
            recommendation_batch_size = 0
            for task_name, (task_loss, task_batch_size) in task_losses.items():
                if log_step:
                    self.log(
                        f"{stage}/loss_{task_name}_step",
                        task_loss,
                        prog_bar=False,
                        on_step=True,
                        on_epoch=False,
                        batch_size=task_batch_size,
                    )
                self.log(
                    f"{stage}/loss_{task_name}",
                    task_loss,
                    prog_bar=False,
                    on_step=False,
                    on_epoch=True,
                    batch_size=task_batch_size,
                )
                if task_name in {"seq_sid_to_sid", "title_history_to_sid"}:
                    weighted_loss = task_loss * task_batch_size
                    recommendation_loss_sum = weighted_loss if recommendation_loss_sum is None else recommendation_loss_sum + weighted_loss
                    recommendation_batch_size += task_batch_size
            if recommendation_loss_sum is not None:
                recommendation_loss = recommendation_loss_sum / recommendation_batch_size
                if log_step:
                    self.log(
                        f"{stage}/loss_recommendation_step",
                        recommendation_loss,
                        prog_bar=True,
                        on_step=True,
                        on_epoch=False,
                        batch_size=recommendation_batch_size,
                    )
                self.log(
                    f"{stage}/loss_recommendation",
                    recommendation_loss,
                    prog_bar=(stage != "test"),
                    on_step=False,
                    on_epoch=True,
                    batch_size=recommendation_batch_size,
                )
        return loss

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        _ = batch_idx
        return self._shared_step(batch, stage="train")

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        _ = batch_idx
        return self._shared_step(batch, stage="val")

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        _ = batch_idx
        return self._shared_step(batch, stage="test")

    def configure_optimizers(self) -> dict[str, Any]:
        named_parameters = [(name, parameter) for name, parameter in self.named_parameters() if parameter.requires_grad]
        if not named_parameters:
            raise ValueError("No trainable parameters found for optimizer construction")

        decay_parameters = []
        no_decay_parameters = []
        for name, parameter in named_parameters:
            if getattr(parameter, "_sid_row_masked", False) or parameter.ndim == 1 or name.endswith(".bias") or "norm" in name.lower():
                no_decay_parameters.append(parameter)
            else:
                decay_parameters.append(parameter)

        parameter_groups = [
            {"params": decay_parameters},
            {"params": no_decay_parameters, "weight_decay": 0.0},
        ]
        optimizer = self._optimizer(parameter_groups)

        if self._scheduler is None:
            return {"optimizer": optimizer}

        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(total_steps * self.hparams.warmup_ratio)
        scheduler = self._scheduler(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
