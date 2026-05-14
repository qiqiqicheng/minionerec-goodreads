from __future__ import annotations

import json
import logging
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import lightning as L
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

from minionerec_goodreads.models.sft import _resolve_dtype
from minionerec_goodreads.utils.sft import build_tokenizer, load_sid_index
from minionerec_goodreads.utils.sid_trie import SIDTrie

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RLCandidate:
    """
    item_id, sid, token_ids, score, rank
    """
    item_id: str
    sid: str
    token_ids: tuple[int, ...]
    score: float
    rank: int


@dataclass(frozen=True)
class RLRolloutStats:
    """
    invalid_count, duplicate_count, completed_count
    """
    invalid_count: int
    duplicate_count: int
    completed_count: int


@dataclass(frozen=True)
class ResponseLogProbs:
    """
    Args:
        token_logprobs: [C, T-1] - log probabilities of the generated tokens
        response_mask: [C, T-1]
        sequence_logprobs: [C,] - sum of log probabilities for each sequence
    """
    token_logprobs: torch.Tensor
    response_mask: torch.Tensor
    sequence_logprobs: torch.Tensor


@dataclass(frozen=True)
class GRPOLoss:
    """
    loss, policy_loss, kl
    """
    loss: torch.Tensor
    policy_loss: torch.Tensor
    kl: torch.Tensor


@dataclass(frozen=True)
class RolloutBatch:
    """
    Args:
        prompts [C] * str, 
        token_ids [C] * tuple[int, ...], 
        advantages [C] * float, 
        rewards [C] * float, 
        batch_size B, 
        candidate_count C, 
        target_in_beam, short_group_rate, invalid_rate, duplicate_rate
    """
    prompts: list[str]
    token_ids: list[tuple[int, ...]]
    advantages: torch.Tensor
    rewards: torch.Tensor
    batch_size: int
    candidate_count: int
    target_in_beam: float
    short_group_rate: float
    invalid_rate: float
    duplicate_rate: float


def _import_peft_model() -> Any:
    try:
        from peft import PeftModel
    except ImportError as error:
        raise ImportError("RL LoRA training requires peft to be installed") from error
    return PeftModel


def compute_rule_rewards(candidates: list[RLCandidate], target_item_id: str) -> list[float]:
    """
    0 / 1 list: 0 for neq, 1 for eq
    """
    return [1.0 if candidate.item_id == target_item_id else 0.0 for candidate in candidates]


def compute_rank_rewards(candidates: list[RLCandidate], target_item_id: str) -> list[float]:
    group_size = len(candidates)
    if group_size == 0:
        return []
    rewards = []
    for candidate in candidates:
        if candidate.item_id == target_item_id:
            rewards.append(0.0)
        else:
            rank = min(max(candidate.rank, 0), group_size - 1)
            rewards.append(-float(group_size - rank) / group_size)
    return rewards


def compute_rewards(
    candidates: list[RLCandidate],
    target_item_id: str,
    rank_reward_lambda: float,
) -> torch.Tensor:
    """
    [Nb,] reward = rule_reward + rank_reward_lambda * rank_reward
    """
    rule_rewards = compute_rule_rewards(candidates, target_item_id)
    rank_rewards = compute_rank_rewards(candidates, target_item_id)
    rewards = [rule + rank_reward_lambda * rank for rule, rank in zip(rule_rewards, rank_rewards)]
    return torch.tensor(rewards, dtype=torch.float32)


def normalize_group_advantages(rewards: torch.Tensor) -> torch.Tensor:
    if rewards.numel() <= 1:
        return torch.zeros_like(rewards)
    std = rewards.std(unbiased=False)
    if std <= 1e-6:
        return torch.zeros_like(rewards)
    return (rewards - rewards.mean()) / (std + 1e-6)


def build_response_batch(
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[str],
    candidate_token_ids: list[tuple[int, ...]],
    max_length: int,
    device: torch.device | str,
) -> dict[str, torch.Tensor]:
    if len(prompts) != len(candidate_token_ids):
        raise ValueError(f"prompts and candidate_token_ids length mismatch: {len(prompts)} vs {len(candidate_token_ids)}")

    rows = []
    for prompt, response_ids_tuple in zip(prompts, candidate_token_ids):
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        if tokenizer.bos_token_id is not None:
            prompt_ids = [tokenizer.bos_token_id, *prompt_ids]
        response_ids = list(response_ids_tuple)
        if len(response_ids) >= max_length:
            raise ValueError(f"Response length {len(response_ids)} must be smaller than max_length={max_length}")
        overflow = len(prompt_ids) + len(response_ids) - max_length
        if overflow > 0:
            if overflow >= len(prompt_ids):
                raise ValueError(f"Prompt is too short to truncate for max_length={max_length}")
            prompt_ids = prompt_ids[overflow:]
        input_ids = [*prompt_ids, *response_ids]
        rows.append(
            {
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
                "response_mask": [0] * len(prompt_ids) + [1] * len(response_ids),
            }
        )

    max_row_length = max(len(row["input_ids"]) for row in rows)
    input_ids = []
    attention_mask = []
    response_mask = []
    for row in rows:
        pad_len = max_row_length - len(row["input_ids"])
        input_ids.append(row["input_ids"] + [tokenizer.pad_token_id] * pad_len)
        attention_mask.append(row["attention_mask"] + [0] * pad_len)
        response_mask.append(row["response_mask"] + [0] * pad_len)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long, device=device),
        "response_mask": torch.tensor(response_mask, dtype=torch.float32, device=device),
    }


def gather_response_logprobs(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    response_mask: torch.Tensor,
) -> ResponseLogProbs:
    """
    Args:
        logits: [C, T, V] - logits for each token
        input_ids: [C, T] - input token IDs
        response_mask: [C, T] - mask for response tokens
    """
    shifted_logits = logits[:, :-1, :].float().contiguous()  # [C, T-1, V]
    shifted_labels = input_ids[:, 1:].contiguous()  # [C, T-1]
    shifted_mask = response_mask[:, 1:].contiguous()  # [C, T-1]
    log_probs = F.log_softmax(shifted_logits, dim=-1)  # [C, T-1, V]
    token_logprobs = log_probs.gather(dim=-1, index=shifted_labels.unsqueeze(-1)).squeeze(-1)  # [C, T-1] select the logprobs of the actual generated tokens
    token_logprobs = token_logprobs * shifted_mask  # [C, T-1]
    return ResponseLogProbs(
        token_logprobs=token_logprobs,  # [C, T-1]
        response_mask=shifted_mask,  # [C, T-1]
        sequence_logprobs=token_logprobs.sum(dim=1),  # [C,] sum logprobs of generated tokens for each sequence
    )


def compute_grpo_loss(
    current_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    reference_logprobs: torch.Tensor,
    response_mask: torch.Tensor,
    advantages: torch.Tensor,
    clip_epsilon: float,
    kl_beta: float,
) -> GRPOLoss:
    """
    Args:
        current_logprobs: [C, T-1]
        old_logprobs: [C, T-1]
        reference_logprobs: [C, T-1]
        response_mask: [C, T-1]
        advantages: [C,]
    """
    token_count = response_mask.sum().clamp_min(1.0)
    token_advantages = advantages[:, None]  # [C, 1]
    ratio = torch.exp(current_logprobs - old_logprobs)  # [C, T-1]
    unclipped = ratio * token_advantages  # [C, T-1]
    clipped = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon) * token_advantages  # [C, T-1]
    policy_loss = -(torch.minimum(unclipped, clipped) * response_mask).sum() / token_count  # scalar
    ref_actor_delta = reference_logprobs - current_logprobs  # [C, T-1]
    kl = ((torch.exp(ref_actor_delta) - ref_actor_delta - 1.0) * response_mask).sum() / token_count  # scalar
    loss = policy_loss + kl_beta * kl  # scalar
    return GRPOLoss(loss=loss, policy_loss=policy_loss, kl=kl)


def constrained_sid_beam_rollout(  # noqa: C901
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    trie: SIDTrie,
    prompt: str,
    num_generations: int,
    max_sid_length: int,
) -> tuple[list[RLCandidate], RLRolloutStats]:
    """ 
    RLCandidate List, RLRolloutStats
    """
    if num_generations <= 0:
        raise ValueError(f"num_generations must be positive, got {num_generations}")
    device = next(model.parameters()).device
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    prompt_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    if tokenizer.bos_token_id is not None:
        bos = torch.tensor([[tokenizer.bos_token_id]], dtype=torch.long, device=device)
        prompt_ids = torch.cat([bos, prompt_ids], dim=1)
        attention_mask = torch.cat([torch.ones_like(bos), attention_mask], dim=1)

    beams: list[tuple[tuple[int, ...], float]] = [((), 0.0)]
    completed: list[tuple[tuple[int, ...], float]] = []
    invalid_count = 0

    for _ in range(max_sid_length):
        beam_candidates: list[tuple[tuple[int, ...], float]] = []
        for prefix, score in beams:
            next_token_ids = trie.next_token_ids(prefix)
            if not next_token_ids:
                invalid_count += 1
                continue
            if prefix:
                prefix_tensor = torch.tensor([prefix], dtype=torch.long, device=device)
                input_ids = torch.cat([prompt_ids, prefix_tensor], dim=1)  # prompt + prefix
                full_attention_mask = torch.cat(
                    [attention_mask, torch.ones((1, len(prefix)), dtype=torch.long, device=device)],
                    dim=1,
                )
            else:
                input_ids = prompt_ids
                full_attention_mask = attention_mask
            logits = model(input_ids=input_ids, attention_mask=full_attention_mask).logits[0, -1]  # [V,]
            log_probs = torch.log_softmax(logits[next_token_ids], dim=-1)  # [Nb,]
            k = min(num_generations, len(next_token_ids))
            top_scores, top_indices = torch.topk(log_probs, k=k)
            for top_score, top_index in zip(top_scores.tolist(), top_indices.tolist()):
                token_id = next_token_ids[top_index]
                next_prefix = (*prefix, token_id)
                next_score = score + top_score  # using log_softmax scores, so we add
                item_id = trie.item_id(next_prefix)
                if item_id is not None:
                    completed.append((next_prefix, next_score))
                if trie.next_token_ids(next_prefix):
                    beam_candidates.append((next_prefix, next_score))
        beams = sorted(beam_candidates, key=lambda row: row[1], reverse=True)[:num_generations]  # beams_size always <= num_generations
        if len(completed) >= num_generations and not beams:
            break

    seen_item_ids: set[str] = set()
    candidates: list[RLCandidate] = []
    duplicate_count = 0
    for token_ids, score in sorted(completed, key=lambda row: row[1], reverse=True):
        item_id = trie.item_id(token_ids)
        if item_id is None:
            invalid_count += 1
            continue
        if item_id in seen_item_ids:  # deduplication based oon item id
            duplicate_count += 1
            continue
        seen_item_ids.add(item_id)
        candidates.append(
            RLCandidate(
                item_id=item_id,
                sid=tokenizer.decode(token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False),
                token_ids=token_ids,
                score=score,
                rank=len(candidates),
            )
        )
        if len(candidates) == num_generations:
            break

    stats = RLRolloutStats(
        invalid_count=invalid_count,
        duplicate_count=duplicate_count,
        completed_count=len(completed),
    )
    return candidates, stats


class RLModule(L.LightningModule):
    def __init__(
        self,
        pretrained_model_name_or_path: str,
        sft_export_dir: str,
        sid_index_path: str,
        max_length: int,
        num_generations: int,
        rank_reward_lambda: float,
        kl_beta: float,
        clip_epsilon: float,
        warmup_ratio: float,
        gradient_checkpointing: bool,
        torch_dtype: str,
        optimizer: Callable,
        scheduler: Callable | None,
        max_sid_length: int | None = None,
        attn_implementation: str | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["optimizer", "scheduler"])
        self._optimizer = optimizer
        self._scheduler = scheduler

        self.sft_export_dir = Path(sft_export_dir)
        self.tokenizer, self.sid_index, self.original_vocab_size = self._load_tokenizer_and_sid_index()
        self.trie = SIDTrie(self.tokenizer, self.sid_index)
        self.max_sid_length = max_sid_length or max(len(tokens) for tokens in self.sid_index.values())

        self.model = self._build_adapter_model()  # adapter: actor* and reference
        self._validate_trainable_parameters()
        self._log_trainable_parameters()

    def _load_tokenizer_and_sid_index(self) -> tuple[PreTrainedTokenizerBase, dict[str, list[str]], int]:
        """ 
        tokenizer, sid_index, original_vocab_size
        """
        export_tokenizer_path = self.sft_export_dir / "tokenizer"
        export_sid_index_path = self.sft_export_dir / "goodreads.index.json"
        manifest_path = self.sft_export_dir / "manifest.json"

        if export_tokenizer_path.exists() and export_sid_index_path.exists():
            tokenizer = AutoTokenizer.from_pretrained(export_tokenizer_path, trust_remote_code=True)
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "right"
            sid_index = load_sid_index(export_sid_index_path)
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                original_vocab_size = int(manifest["original_vocab_size"])
            else:
                base_tokenizer = AutoTokenizer.from_pretrained(
                    self.hparams.pretrained_model_name_or_path,
                    trust_remote_code=True,
                )
                original_vocab_size = len(base_tokenizer)
            return tokenizer, sid_index, original_vocab_size

        return build_tokenizer(
            pretrained_model_name_or_path=self.hparams.pretrained_model_name_or_path,
            sid_index_path=self.hparams.sid_index_path,
        )

    def _build_adapter_model(self) -> torch.nn.Module:
        adapter_dir = self.sft_export_dir / "adapter"
        if not adapter_dir.exists():
            raise FileNotFoundError(f"Missing SFT adapter export: {adapter_dir}")
        PeftModel = _import_peft_model()
        model_kwargs: dict[str, Any] = {
            "torch_dtype": _resolve_dtype(self.hparams.torch_dtype),
            "trust_remote_code": True,
        }
        if self.hparams.attn_implementation is not None:
            model_kwargs["attn_implementation"] = self.hparams.attn_implementation
        base_model = AutoModelForCausalLM.from_pretrained(self.hparams.pretrained_model_name_or_path, **model_kwargs)
        base_model.resize_token_embeddings(len(self.tokenizer))
        if self.hparams.gradient_checkpointing:
            base_model.gradient_checkpointing_enable()
            if hasattr(base_model, "enable_input_require_grads"):
                base_model.enable_input_require_grads()
            base_model.config.use_cache = False

        model = PeftModel.from_pretrained(base_model, adapter_dir, adapter_name="actor", is_trainable=True)
        model.load_adapter(adapter_dir, adapter_name="reference", is_trainable=False)
        model.set_adapter("actor")
        model.config.use_cache = False
        self._freeze_to_actor_lora_parameters(model)
        return model

    def _freeze_to_actor_lora_parameters(self, model: torch.nn.Module) -> None:
        actor_markers = (".lora_A.actor.", ".lora_B.actor.", ".lora_embedding_A.actor", ".lora_embedding_B.actor")
        for name, parameter in model.named_parameters():
            parameter.requires_grad = any(marker in name for marker in actor_markers)

    def _validate_trainable_parameters(self) -> None:
        trainable_names = [name for name, parameter in self.named_parameters() if parameter.requires_grad]
        if not trainable_names:
            raise ValueError("RL module has no trainable parameters")
        reference_trainable = [name for name in trainable_names if "reference" in name]
        if reference_trainable:
            raise ValueError(f"Reference adapter parameters must be frozen, got {reference_trainable[:5]}")

    def _log_trainable_parameters(self) -> None:
        total_params = sum(parameter.numel() for parameter in self.model.parameters())
        trainable_params = sum(parameter.numel() for parameter in self.model.parameters() if parameter.requires_grad)
        ratio = 100 * trainable_params / total_params
        log.info("RL train mode=lora trainable_params=%s total_params=%s ratio=%.4f%%", trainable_params, total_params, ratio)

    def _set_actor(self) -> None:
        self.model.set_adapter("actor")
        self._freeze_to_actor_lora_parameters(self.model)

    def _set_reference(self) -> None:
        self.model.set_adapter("reference")
        self._freeze_to_actor_lora_parameters(self.model)

    def _rollout_one(self, prompt: str) -> tuple[list[RLCandidate], RLRolloutStats]:
        was_training = self.model.training
        self.model.eval()
        self._set_actor()
        with torch.inference_mode():
            candidates, stats = constrained_sid_beam_rollout(
                model=self.model,
                tokenizer=self.tokenizer,
                trie=self.trie,
                prompt=prompt,
                num_generations=self.hparams.num_generations,
                max_sid_length=self.max_sid_length,
            )
        if was_training:
            self.model.train()
        return candidates, stats

    def _build_rollout_batch(self, batch: dict[str, Any]) -> RolloutBatch:
        flat_prompts: list[str] = []
        flat_token_ids: list[tuple[int, ...]] = []
        advantages_by_group = []
        rewards_by_group = []
        target_hits = 0
        short_groups = 0
        invalid_count = 0
        duplicate_count = 0
        completed_count = 0

        prompts = batch["prompts"]  # list
        target_item_ids = batch["target_item_ids"]  # list
        for prompt, target_item_id in zip(prompts, target_item_ids):
            candidates, stats = self._rollout_one(prompt)
            if not candidates:
                raise ValueError("Constrained beam rollout produced no valid candidates")
            rewards = compute_rewards(candidates, target_item_id, self.hparams.rank_reward_lambda)  # [Nb,]
            advantages = normalize_group_advantages(rewards)  # [Nb,]
            flat_prompts.extend([prompt] * len(candidates))  # [Nb] * str
            flat_token_ids.extend([candidate.token_ids for candidate in candidates])  # [Nb] * tuple[int, ...]
            advantages_by_group.append(advantages)
            rewards_by_group.append(rewards)
            target_hits += int(any(candidate.item_id == target_item_id for candidate in candidates))
            short_groups += int(len(candidates) < self.hparams.num_generations)
            invalid_count += stats.invalid_count
            duplicate_count += stats.duplicate_count
            completed_count += stats.completed_count

        batch_size = len(prompts)
        candidate_count = len(flat_token_ids)
        if candidate_count == 0:
            raise ValueError("RL rollout batch has no candidates")
        normalizer = max(completed_count, candidate_count)
        return RolloutBatch(  # C: C <= B*Nb
            prompts=flat_prompts,  # [C] * str
            token_ids=flat_token_ids,  # [C] * tuple[int, ...], candidate token ids
            advantages=torch.cat(advantages_by_group),  # [C,]
            rewards=torch.cat(rewards_by_group),  # [C,]
            batch_size=batch_size,  # B
            candidate_count=candidate_count,  # C
            target_in_beam=target_hits / batch_size,
            short_group_rate=short_groups / batch_size,
            invalid_rate=invalid_count / max(normalizer, 1),
            duplicate_rate=duplicate_count / max(normalizer, 1),
        )

    def _compute_adapter_logprobs(
        self,
        prompts: list[str],
        token_ids: list[tuple[int, ...]],
        adapter_name: str,
        grad: bool,
    ) -> ResponseLogProbs:
        was_training = self.model.training
        self.model.eval()
        if adapter_name == "actor":
            self._set_actor()
        elif adapter_name == "reference":
            self._set_reference()
        else:
            raise ValueError(f"Unknown adapter_name={adapter_name}")
        encoded = build_response_batch(
            tokenizer=self.tokenizer,
            prompts=prompts,
            candidate_token_ids=token_ids,
            max_length=self.hparams.max_length,
            device=self.device,
        )
        context = nullcontext() if grad else torch.no_grad()
        with context:
            outputs = self.model(input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"])
            logprobs: ResponseLogProbs = gather_response_logprobs(outputs.logits, encoded["input_ids"], encoded["response_mask"])
        if was_training:
            self.model.train()
        return logprobs

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        _ = batch_idx
        rollout: RolloutBatch = self._build_rollout_batch(batch)
        advantages = rollout.advantages.to(self.device)  # [C,]
        with torch.no_grad():
            old_logprobs: ResponseLogProbs = self._compute_adapter_logprobs(
                rollout.prompts, 
                rollout.token_ids, 
                adapter_name="actor", 
                grad=False
            )
            reference_logprobs: ResponseLogProbs = self._compute_adapter_logprobs(
                rollout.prompts,
                rollout.token_ids,
                adapter_name="reference",
                grad=False,
            )
        current_logprobs: ResponseLogProbs = self._compute_adapter_logprobs(
            rollout.prompts, 
            rollout.token_ids, 
            adapter_name="actor", 
            grad=True
        )
        loss = compute_grpo_loss(
            current_logprobs=current_logprobs.token_logprobs,
            old_logprobs=old_logprobs.token_logprobs.detach(),
            reference_logprobs=reference_logprobs.token_logprobs.detach(),
            response_mask=current_logprobs.response_mask,
            advantages=advantages,
            clip_epsilon=self.hparams.clip_epsilon,
            kl_beta=self.hparams.kl_beta,
        )
        if not torch.isfinite(loss.loss):
            raise ValueError(f"Non-finite train loss: {loss.loss}")
        self._set_actor()
        self.log("train/loss", loss.loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=rollout.batch_size)
        self.log("train/policy_loss", loss.policy_loss, prog_bar=False, on_step=True, on_epoch=True, batch_size=rollout.batch_size)
        self.log("train/kl", loss.kl, prog_bar=True, on_step=True, on_epoch=True, batch_size=rollout.batch_size)
        self.log("train/reward_mean", rollout.rewards.mean(), prog_bar=True, on_step=True, on_epoch=True, batch_size=rollout.candidate_count)
        self.log("train/rule_hit_at_g", rollout.target_in_beam, prog_bar=True, on_step=True, on_epoch=True, batch_size=rollout.batch_size)
        self.log("train/target_in_beam", rollout.target_in_beam, prog_bar=False, on_step=True, on_epoch=True, batch_size=rollout.batch_size)
        self.log("train/short_group_rate", rollout.short_group_rate, prog_bar=False, on_step=True, on_epoch=True, batch_size=rollout.batch_size)
        self.log("train/invalid_rate", rollout.invalid_rate, prog_bar=False, on_step=True, on_epoch=True, batch_size=rollout.batch_size)
        self.log("train/duplicate_rate", rollout.duplicate_rate, prog_bar=False, on_step=True, on_epoch=True, batch_size=rollout.batch_size)
        return loss.loss

    def _eval_step(self, batch: dict[str, Any], stage: str) -> None:
        rollout = self._build_rollout_batch(batch)
        self.log(f"{stage}/reward_mean", rollout.rewards.mean(), prog_bar=False, on_step=False, on_epoch=True, batch_size=rollout.candidate_count)
        self.log(f"{stage}/rule_hit_at_g", rollout.target_in_beam, prog_bar=True, on_step=False, on_epoch=True, batch_size=rollout.batch_size)
        self.log(f"{stage}/target_in_beam", rollout.target_in_beam, prog_bar=False, on_step=False, on_epoch=True, batch_size=rollout.batch_size)
        self.log(f"{stage}/short_group_rate", rollout.short_group_rate, prog_bar=False, on_step=False, on_epoch=True, batch_size=rollout.batch_size)
        self.log(f"{stage}/invalid_rate", rollout.invalid_rate, prog_bar=False, on_step=False, on_epoch=True, batch_size=rollout.batch_size)
        self.log(f"{stage}/duplicate_rate", rollout.duplicate_rate, prog_bar=False, on_step=False, on_epoch=True, batch_size=rollout.batch_size)

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        _ = batch_idx
        self._eval_step(batch, stage="val")

    def test_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        _ = batch_idx
        self._eval_step(batch, stage="test")

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

        optimizer = self._optimizer(
            [
                {"params": decay_parameters},
                {"params": no_decay_parameters, "weight_decay": 0.0},
            ]
        )
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
