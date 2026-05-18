from __future__ import annotations

import json
import logging
import math
import shutil
from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import lightning as L
import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, PreTrainedTokenizerBase

from minionerec_goodreads.dataset.sft_dataset import load_split_csv
from minionerec_goodreads.metrics.rec import hit_at_k, mrr_at_k, ndcg_at_k
from minionerec_goodreads.models.sft import _import_bitsandbytes, _resolve_dtype
from minionerec_goodreads.utils.sft import build_tokenizer, load_sid_index, load_tokenizer
from minionerec_goodreads.utils.sft_generation import forward_last_token_logits
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
    loss, policy_loss, kl, kl_seq_mean, clip_fraction, ref_actor_logprob_delta_mean
    """

    loss: torch.Tensor
    policy_loss: torch.Tensor
    kl: torch.Tensor
    kl_seq_mean: torch.Tensor
    clip_fraction: torch.Tensor
    ref_actor_logprob_delta_mean: torch.Tensor


@dataclass(frozen=True)
class RewardBreakdown:
    """
    total/exact/rank/partial/long_tail: [Nb]
    """

    total: torch.Tensor
    exact: torch.Tensor
    rank: torch.Tensor
    partial: torch.Tensor
    long_tail: torch.Tensor


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
        rec_metrics
    """

    prompts: list[str]
    token_ids: list[tuple[int, ...]]
    advantages: torch.Tensor
    rewards: torch.Tensor
    exact_rewards: torch.Tensor
    rank_rewards: torch.Tensor
    partial_rewards: torch.Tensor
    long_tail_rewards: torch.Tensor
    batch_size: int
    candidate_count: int
    target_in_beam: float
    short_group_rate: float
    invalid_rate: float
    duplicate_rate: float
    rec_metrics: dict[str, float]


def _import_peft_rl() -> tuple[Any, Any]:
    try:
        from peft import PeftModel, prepare_model_for_kbit_training
    except ImportError as error:
        raise ImportError("RL LoRA training requires peft to be installed") from error
    return PeftModel, prepare_model_for_kbit_training


def _import_vllm_rl() -> tuple[Any, Any, Any]:
    try:
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest
    except ImportError as error:
        raise ImportError("vLLM rollout requires vllm to be installed") from error
    return LLM, SamplingParams, LoRARequest


def _is_saved_embedding_key(key: str) -> bool:
    return key.endswith("embed_tokens.weight") or key.endswith("lm_head.weight")


def _adapter_tensor_path(adapter_dir: Path) -> Path:
    path = adapter_dir / "adapter_model.safetensors"
    if not path.exists():
        raise FileNotFoundError(f"Missing adapter safetensors: {path}")
    return path


def _file_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _new_embedding_rows_from_adapter(
    adapter_dir: Path,
    original_vocab_size: int,
    augmented_vocab_size: int,
) -> dict[str, torch.Tensor]:
    new_embeddings_path = adapter_dir / "new_embeddings.safetensors"
    if new_embeddings_path.exists():
        rows = load_file(new_embeddings_path, device="cpu")
        if "input_embeddings" not in rows:
            raise ValueError(f"{new_embeddings_path} must contain input_embeddings")
        return rows

    tensor_path = _adapter_tensor_path(adapter_dir)
    rows: dict[str, torch.Tensor] = {}
    with safe_open(tensor_path, framework="pt", device="cpu") as file:
        for key in file.keys():
            tensor_name = None
            if key.endswith("embed_tokens.weight"):
                tensor_name = "input_embeddings"
            elif key.endswith("lm_head.weight"):
                tensor_name = "output_embeddings"
            if tensor_name is None:
                continue
            weight = file.get_tensor(key)
            if weight.shape[0] == augmented_vocab_size:
                rows[tensor_name] = weight[original_vocab_size:augmented_vocab_size].contiguous()
            elif weight.shape[0] == augmented_vocab_size - original_vocab_size:
                rows[tensor_name] = weight.contiguous()
            else:
                raise ValueError(
                    f"Unexpected embedding shape for {key}: {tuple(weight.shape)}, "
                    f"expected vocab {augmented_vocab_size} or new rows {augmented_vocab_size - original_vocab_size}"
                )
    if "input_embeddings" not in rows:
        raise ValueError(f"Adapter {adapter_dir} does not contain new SID token embeddings")
    return rows


def prepare_lora_only_adapter_dir(
    adapter_dir: Path,
    original_vocab_size: int,
    augmented_vocab_size: int,
) -> Path:
    tensor_path = _adapter_tensor_path(adapter_dir)
    with safe_open(tensor_path, framework="pt", device="cpu") as file:
        keys = list(file.keys())
        metadata = file.metadata() or {"format": "pt"}
    if not any(_is_saved_embedding_key(key) for key in keys):
        if not (adapter_dir / "new_embeddings.safetensors").exists():
            raise ValueError(f"LoRA-only adapter {adapter_dir} is missing new_embeddings.safetensors")
        return adapter_dir

    target_dir = adapter_dir.parent / "adapter_lora_only"
    target_tensor_path = target_dir / "adapter_model.safetensors"
    target_embeddings_path = target_dir / "new_embeddings.safetensors"
    if target_tensor_path.exists() and target_embeddings_path.exists():
        return target_dir

    target_dir.mkdir(parents=True, exist_ok=True)
    for filename in ["adapter_config.json", "README.md"]:
        src = adapter_dir / filename
        if src.exists():
            shutil.copy2(src, target_dir / filename)

    rows = _new_embedding_rows_from_adapter(adapter_dir, original_vocab_size, augmented_vocab_size)
    save_file(rows, target_embeddings_path, metadata={"format": "pt"})

    tensors: dict[str, torch.Tensor] = {}
    with safe_open(tensor_path, framework="pt", device="cpu") as file:
        for key in keys:
            if not _is_saved_embedding_key(key):
                tensors[key] = file.get_tensor(key)
    if not tensors:
        raise ValueError(f"Adapter {adapter_dir} has no LoRA tensors after removing embeddings")
    save_file(tensors, target_tensor_path, metadata=metadata)
    return target_dir


def prepare_augmented_vllm_base_dir(
    base_model_path: str,
    tokenizer_dir: Path,
    adapter_dir: Path,
    output_dir: Path,
    original_vocab_size: int,
    augmented_vocab_size: int,
    torch_dtype: str,
) -> Path:
    source_hash = _file_sha256(adapter_dir / "new_embeddings.safetensors")
    manifest_path = output_dir / "minionerec_vllm_manifest.json"
    config_path = output_dir / "config.json"
    model_index_path = output_dir / "model.safetensors.index.json"
    cache_valid = False
    if config_path.exists() and model_index_path.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        cache_valid = (
            manifest.get("base_model_path") == base_model_path
            and manifest.get("original_vocab_size") == original_vocab_size
            and manifest.get("augmented_vocab_size") == augmented_vocab_size
            and manifest.get("torch_dtype") == torch_dtype
            and manifest.get("new_embeddings_sha256") == source_hash
        )
    if cache_valid:
        return output_dir

    if output_dir.exists():
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=_resolve_dtype(torch_dtype),
        trust_remote_code=True,
        device_map="cpu",
    )
    tokenizer = load_tokenizer(tokenizer_dir, trust_remote_code=True)
    model.resize_token_embeddings(augmented_vocab_size)
    rows = _new_embedding_rows_from_adapter(adapter_dir, original_vocab_size, augmented_vocab_size)
    input_embedding = model.get_input_embeddings()
    if input_embedding is None:
        raise ValueError("Base model must expose input embeddings")
    with torch.no_grad():
        input_rows = rows["input_embeddings"].to(dtype=input_embedding.weight.dtype)
        input_embedding.weight[original_vocab_size:augmented_vocab_size].copy_(input_rows)
        output_embedding = model.get_output_embeddings()
        if output_embedding is not None and output_embedding.weight is not input_embedding.weight:
            output_rows = rows.get("output_embeddings", rows["input_embeddings"]).to(dtype=output_embedding.weight.dtype)
            output_embedding.weight[original_vocab_size:augmented_vocab_size].copy_(output_rows)
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    manifest_path.write_text(
        json.dumps(
            {
                "base_model_path": base_model_path,
                "original_vocab_size": original_vocab_size,
                "augmented_vocab_size": augmented_vocab_size,
                "torch_dtype": torch_dtype,
                "new_embeddings_sha256": source_hash,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    del model
    return output_dir


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


def common_prefix_ratio(candidate_tokens: tuple[int, ...], target_tokens: tuple[int, ...]) -> float:
    matched = 0
    for candidate_token, target_token in zip(candidate_tokens, target_tokens):
        if candidate_token != target_token:
            break
        matched += 1
    return matched / len(target_tokens)


def compute_partial_match_rewards(
    candidates: list[RLCandidate],
    target_item_id: str,
    target_token_ids: tuple[int, ...] | None,
) -> list[float]:
    if target_token_ids is None:
        return [0.0 for _ in candidates]
    rewards = []
    for candidate in candidates:
        if candidate.item_id == target_item_id:
            rewards.append(0.0)
        else:
            rewards.append(common_prefix_ratio(candidate.token_ids, target_token_ids))
    return rewards


def compute_long_tail_rewards(
    candidates: list[RLCandidate],
    target_item_id: str,
    item_popularity: dict[str, float] | None,
) -> list[float]:
    if item_popularity is None:
        return [0.0 for _ in candidates]
    rewards = []
    for candidate in candidates:
        pop_norm = item_popularity.get(candidate.item_id, 0.0)
        tail = 1.0 - pop_norm
        rewards.append(tail if candidate.item_id == target_item_id else -pop_norm)
    return rewards


def compute_reward_breakdown(
    candidates: list[RLCandidate],
    target_item_id: str,
    rank_reward_lambda: float,
    use_partial_match_reward: bool = False,
    partial_match_lambda: float = 0.0,
    target_token_ids: tuple[int, ...] | None = None,
    use_long_tail_reward: bool = False,
    long_tail_reward_lambda: float = 0.0,
    item_popularity: dict[str, float] | None = None,
) -> RewardBreakdown:
    """
    [Nb,] reward components and total reward
    """
    exact = torch.tensor(compute_rule_rewards(candidates, target_item_id), dtype=torch.float32)
    rank = torch.tensor(compute_rank_rewards(candidates, target_item_id), dtype=torch.float32)
    partial = torch.tensor(
        compute_partial_match_rewards(candidates, target_item_id, target_token_ids)
        if use_partial_match_reward
        else [0.0 for _ in candidates],
        dtype=torch.float32,
    )
    long_tail = torch.tensor(
        compute_long_tail_rewards(candidates, target_item_id, item_popularity)
        if use_long_tail_reward
        else [0.0 for _ in candidates],
        dtype=torch.float32,
    )
    total = exact + rank_reward_lambda * rank + partial_match_lambda * partial + long_tail_reward_lambda * long_tail
    return RewardBreakdown(total=total, exact=exact, rank=rank, partial=partial, long_tail=long_tail)


def compute_rewards(
    candidates: list[RLCandidate],
    target_item_id: str,
    rank_reward_lambda: float,
    use_partial_match_reward: bool = False,
    partial_match_lambda: float = 0.0,
    target_token_ids: tuple[int, ...] | None = None,
    use_long_tail_reward: bool = False,
    long_tail_reward_lambda: float = 0.0,
    item_popularity: dict[str, float] | None = None,
) -> torch.Tensor:
    return compute_reward_breakdown(
        candidates=candidates,
        target_item_id=target_item_id,
        rank_reward_lambda=rank_reward_lambda,
        use_partial_match_reward=use_partial_match_reward,
        partial_match_lambda=partial_match_lambda,
        target_token_ids=target_token_ids,
        use_long_tail_reward=use_long_tail_reward,
        long_tail_reward_lambda=long_tail_reward_lambda,
        item_popularity=item_popularity,
    ).total


def compute_ranking_monitor_metrics(
    candidates: list[RLCandidate],
    target_item_id: str,
    ks: tuple[int, ...],
) -> dict[str, float]:
    predictions = [candidate.item_id for candidate in sorted(candidates, key=lambda item: item.rank)]
    metrics = {}
    for k in ks:
        metrics[f"HR@{k}"] = hit_at_k(predictions, target_item_id, k)
        metrics[f"NDCG@{k}"] = ndcg_at_k(predictions, target_item_id, k)
        metrics[f"MRR@{k}"] = mrr_at_k(predictions, target_item_id, k)
    return metrics


def mean_metric_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = rows[0].keys()
    return {key: sum(row[key] for row in rows) / len(rows) for key in keys}


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
        raise ValueError(
            f"prompts and candidate_token_ids length mismatch: {len(prompts)} vs {len(candidate_token_ids)}"
        )

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
        rows.append({
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "response_mask": [0] * len(prompt_ids) + [1] * len(response_ids),
        })

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
    token_logprobs = log_probs.gather(dim=-1, index=shifted_labels.unsqueeze(-1)).squeeze(
        -1
    )  # [C, T-1] select the logprobs of the actual generated tokens
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
    token_count_normalizer: torch.Tensor | None = None,
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
    loss_token_count = token_count if token_count_normalizer is None else token_count_normalizer.clamp_min(1.0)
    token_advantages = advantages[:, None]  # [C, 1]
    ratio = torch.exp(current_logprobs - old_logprobs)  # [C, T-1]
    unclipped = ratio * token_advantages  # [C, T-1]
    clipped = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon) * token_advantages  # [C, T-1]
    policy_loss = -(torch.minimum(unclipped, clipped) * response_mask).sum() / loss_token_count  # scalar
    ref_actor_delta = reference_logprobs - current_logprobs  # [C, T-1]
    token_kl = (torch.exp(ref_actor_delta) - ref_actor_delta - 1.0) * response_mask  # [C, T-1]
    kl = token_kl.sum() / loss_token_count  # scalar
    sequence_kl = token_kl.sum(dim=1)  # [C,]
    clip_mask = ((ratio < 1.0 - clip_epsilon) | (ratio > 1.0 + clip_epsilon)).float() * response_mask
    clip_fraction = clip_mask.sum() / token_count
    ref_actor_logprob_delta_mean = (ref_actor_delta * response_mask).sum() / token_count
    loss = policy_loss + kl_beta * kl  # scalar
    return GRPOLoss(
        loss=loss,
        policy_loss=policy_loss,
        kl=kl,
        kl_seq_mean=sequence_kl.mean(),
        clip_fraction=clip_fraction,
        ref_actor_logprob_delta_mean=ref_actor_logprob_delta_mean,
    )


def constrained_sid_beam_rollout(  # noqa: C901
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    trie: SIDTrie,
    prompt: str,
    num_generations: int,
    max_sid_length: int,
    max_length: int | None = None,
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
    if max_length is not None:
        overflow = prompt_ids.shape[1] + max_sid_length - max_length
        if overflow > 0:
            if overflow >= prompt_ids.shape[1]:
                raise ValueError(f"Prompt is too short to truncate for max_length={max_length}")
            prompt_ids = prompt_ids[:, overflow:]
            attention_mask = attention_mask[:, overflow:]

    beams: list[tuple[tuple[int, ...], float]] = [((), 0.0)]
    completed: list[tuple[tuple[int, ...], float]] = []
    invalid_count = 0
    next_token_cache: dict[tuple[int, ...], list[int]] = {}
    item_cache: dict[tuple[int, ...], str | None] = {}

    def next_token_ids(prefix: tuple[int, ...]) -> list[int]:
        if prefix not in next_token_cache:
            next_token_cache[prefix] = trie.next_token_ids(prefix)
        return next_token_cache[prefix]

    def item_id(prefix: tuple[int, ...]) -> str | None:
        if prefix not in item_cache:
            item_cache[prefix] = trie.item_id(prefix)
        return item_cache[prefix]

    for _ in range(max_sid_length):
        active_rows = [(prefix, score, next_token_ids(prefix)) for prefix, score in beams]
        valid_rows = [(prefix, score, allowed_ids) for prefix, score, allowed_ids in active_rows if allowed_ids]
        invalid_count += len(active_rows) - len(valid_rows)
        if not valid_rows:
            break

        prefix_length = len(valid_rows[0][0])
        batch_size = len(valid_rows)
        batch_prompt_ids = prompt_ids.expand(batch_size, -1)
        batch_attention_mask = attention_mask.expand(batch_size, -1)
        if prefix_length:
            prefix_tensor = torch.tensor([prefix for prefix, _, _ in valid_rows], dtype=torch.long, device=device)
            input_ids = torch.cat([batch_prompt_ids, prefix_tensor], dim=1)
            prefix_mask = torch.ones((batch_size, prefix_length), dtype=torch.long, device=device)
            full_attention_mask = torch.cat([batch_attention_mask, prefix_mask], dim=1)
        else:
            input_ids = batch_prompt_ids
            full_attention_mask = batch_attention_mask

        logits = forward_last_token_logits(model, input_ids=input_ids, attention_mask=full_attention_mask)  # [Nb, V]
        beam_candidates: list[tuple[tuple[int, ...], float]] = []
        for row_idx, (prefix, score, allowed_ids) in enumerate(valid_rows):
            allowed_tensor = torch.tensor(allowed_ids, dtype=torch.long, device=device)
            log_probs = torch.log_softmax(logits[row_idx, allowed_tensor], dim=-1)  # [A]
            k = min(num_generations, len(allowed_ids))
            top_scores, top_indices = torch.topk(log_probs, k=k)
            for top_score, top_index in zip(top_scores.tolist(), top_indices.tolist()):
                token_id = allowed_ids[top_index]
                next_prefix = (*prefix, token_id)
                next_score = score + top_score
                if item_id(next_prefix) is not None:
                    completed.append((next_prefix, next_score))
                if next_token_ids(next_prefix):
                    beam_candidates.append((next_prefix, next_score))
        beams = sorted(beam_candidates, key=lambda row: row[1], reverse=True)[
            :num_generations
        ]  # beams_size always <= num_generations
        if len(completed) >= num_generations and not beams:
            break

    seen_item_ids: set[str] = set()
    candidates: list[RLCandidate] = []
    duplicate_count = 0
    for token_ids, score in sorted(completed, key=lambda row: row[1], reverse=True):
        completed_item_id = item_id(token_ids)
        if completed_item_id is None:
            invalid_count += 1
            continue
        if completed_item_id in seen_item_ids:  # deduplication based oon item id
            duplicate_count += 1
            continue
        seen_item_ids.add(completed_item_id)
        candidates.append(
            RLCandidate(
                item_id=completed_item_id,
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


def _vllm_dtype(value: str) -> str:
    return {"float32": "float32", "float16": "float16", "bfloat16": "bfloat16"}[value]


class VLLMSIDRolloutEngine:  # noqa: C901
    def __init__(
        self,
        model_path: str,
        tokenizer_path: Path,
        tokenizer: PreTrainedTokenizerBase,
        trie: SIDTrie,
        adapter_path: Path,
        num_generations: int,
        max_sid_length: int,
        max_length: int,
        dtype: str,
        seed: int,
        max_lora_rank: int,
        gpu_memory_utilization: float,
        enable_prefix_caching: bool,
    ):
        LLM, SamplingParams, LoRARequest = _import_vllm_rl()
        self.SamplingParams = SamplingParams
        self.LoRARequest = LoRARequest
        self.tokenizer = tokenizer
        self.trie = trie
        self.num_generations = num_generations
        self.max_sid_length = max_sid_length
        self.max_length = max_length
        self.lora_int_id = 1
        self.lora_request = self.LoRARequest("rl_actor", self.lora_int_id, lora_path=str(adapter_path))
        self.llm = LLM(
            model=model_path,
            tokenizer=str(tokenizer_path),
            trust_remote_code=True,
            tensor_parallel_size=1,
            dtype=_vllm_dtype(dtype),
            seed=seed,
            enable_lora=True,
            max_lora_rank=max_lora_rank,
            max_model_len=max_length,
            gpu_memory_utilization=gpu_memory_utilization,
            enable_prefix_caching=enable_prefix_caching,
        )

    def set_adapter_path(self, adapter_path: Path) -> None:
        self.lora_int_id += 1
        self.lora_request = self.LoRARequest(f"rl_actor_{self.lora_int_id}", self.lora_int_id, lora_path=str(adapter_path))

    def rollout(self, prompt: str) -> tuple[list[RLCandidate], RLRolloutStats]:
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if self.tokenizer.bos_token_id is not None:
            prompt_ids = [self.tokenizer.bos_token_id, *prompt_ids]
        overflow = len(prompt_ids) + self.max_sid_length - self.max_length
        if overflow > 0:
            if overflow >= len(prompt_ids):
                raise ValueError(f"Prompt is too short to truncate for max_length={self.max_length}")
            prompt_ids = prompt_ids[overflow:]

        beams: list[tuple[tuple[int, ...], float]] = [((), 0.0)]
        completed: list[tuple[tuple[int, ...], float]] = []
        invalid_count = 0
        next_token_cache: dict[tuple[int, ...], list[int]] = {}
        item_cache: dict[tuple[int, ...], str | None] = {}

        def next_token_ids(prefix: tuple[int, ...]) -> list[int]:
            if prefix not in next_token_cache:
                next_token_cache[prefix] = self.trie.next_token_ids(prefix)
            return next_token_cache[prefix]

        def item_id(prefix: tuple[int, ...]) -> str | None:
            if prefix not in item_cache:
                item_cache[prefix] = self.trie.item_id(prefix)
            return item_cache[prefix]

        for _ in range(self.max_sid_length):
            active_rows = [(prefix, score, next_token_ids(prefix)) for prefix, score in beams]
            valid_rows = [(prefix, score, allowed_ids) for prefix, score, allowed_ids in active_rows if allowed_ids]
            invalid_count += len(active_rows) - len(valid_rows)
            if not valid_rows:
                break

            prompts = [{"prompt_token_ids": [*prompt_ids, *prefix]} for prefix, _, _ in valid_rows]
            sampling_params = [
                self.SamplingParams(
                    temperature=0.0,
                    max_tokens=1,
                    logprobs=min(self.num_generations, len(allowed_ids)),
                    allowed_token_ids=allowed_ids,
                    detokenize=False,
                    skip_special_tokens=False,
                )
                for _, _, allowed_ids in valid_rows
            ]
            outputs = self.llm.generate(
                prompts,
                sampling_params,
                use_tqdm=False,
                lora_request=self.lora_request,
            )

            beam_candidates: list[tuple[tuple[int, ...], float]] = []
            for output, (prefix, score, allowed_ids) in zip(outputs, valid_rows):
                next_scores = self._extract_next_scores(output, allowed_ids)
                for token_id, token_score in next_scores:
                    next_prefix = (*prefix, token_id)
                    next_score = score + token_score
                    if item_id(next_prefix) is not None:
                        completed.append((next_prefix, next_score))
                    if next_token_ids(next_prefix):
                        beam_candidates.append((next_prefix, next_score))

            beams = sorted(beam_candidates, key=lambda row: row[1], reverse=True)[: self.num_generations]
            if len(completed) >= self.num_generations and not beams:
                break

        candidates, duplicate_count = self._deduplicate_completed(completed, item_id)
        return candidates, RLRolloutStats(
            invalid_count=invalid_count,
            duplicate_count=duplicate_count,
            completed_count=len(completed),
        )

    def _extract_next_scores(self, output: Any, allowed_ids: list[int]) -> list[tuple[int, float]]:
        completion = output.outputs[0]
        allowed = set(allowed_ids)
        scores: dict[int, float] = {}
        if completion.logprobs:
            for token_id, logprob in completion.logprobs[0].items():
                int_token_id = int(token_id)
                if int_token_id in allowed:
                    value = getattr(logprob, "logprob", logprob)
                    scores[int_token_id] = float(value)
        if not scores and completion.token_ids:
            token_id = int(completion.token_ids[0])
            if token_id in allowed:
                scores[token_id] = float(completion.cumulative_logprob or 0.0)
        return sorted(scores.items(), key=lambda row: row[1], reverse=True)[: self.num_generations]

    def _deduplicate_completed(
        self,
        completed: list[tuple[tuple[int, ...], float]],
        item_id: Callable[[tuple[int, ...]], str | None],
    ) -> tuple[list[RLCandidate], int]:
        seen_item_ids: set[str] = set()
        candidates: list[RLCandidate] = []
        duplicate_count = 0
        for token_ids, score in sorted(completed, key=lambda row: row[1], reverse=True):
            completed_item_id = item_id(token_ids)
            if completed_item_id is None:
                continue
            if completed_item_id in seen_item_ids:
                duplicate_count += 1
                continue
            seen_item_ids.add(completed_item_id)
            candidates.append(
                RLCandidate(
                    item_id=completed_item_id,
                    sid=self.tokenizer.decode(token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False),
                    token_ids=token_ids,
                    score=score,
                    rank=len(candidates),
                )
            )
            if len(candidates) == self.num_generations:
                break
        return candidates, duplicate_count


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
        load_in_4bit: bool,
        load_in_8bit: bool,
        bnb_4bit_compute_dtype: str,
        bnb_4bit_quant_type: str,
        bnb_4bit_use_double_quant: bool,
        optimizer: Callable,
        scheduler: Callable | None,
        max_sid_length: int | None = None,
        attn_implementation: str | None = None,
        logprob_micro_batch_size: int | None = None,
        optimizer_accumulate_batches: int = 1,
        rollout_backend: str = "hf",
        vllm_gpu_memory_utilization: float = 0.30,
        vllm_enable_prefix_caching: bool = True,
        vllm_max_lora_rank: int = 16,
        use_partial_match_reward: bool = True,
        partial_match_lambda: float = 0.2,
        use_long_tail_reward: bool = False,
        long_tail_reward_lambda: float = 0.05,
        popularity_train_path: str | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["optimizer", "scheduler"])
        self._optimizer = optimizer
        self._scheduler = scheduler
        self.automatic_optimization = False

        self.sft_export_dir = Path(sft_export_dir)
        self.tokenizer, self.sid_index, self.original_vocab_size = self._load_tokenizer_and_sid_index()
        self.trie = SIDTrie(self.tokenizer, self.sid_index)
        self.max_sid_length = max_sid_length or max(len(tokens) for tokens in self.sid_index.values())
        self.item_popularity = self._load_item_popularity()
        self.adapter_dir = prepare_lora_only_adapter_dir(
            self.sft_export_dir / "adapter",
            self.original_vocab_size,
            len(self.tokenizer),
        )
        self.vllm_adapter_step = 0
        self.vllm_runtime_adapter_dir = self.sft_export_dir / "adapter_vllm_runtime"
        self.vllm_augmented_base_dir = self.sft_export_dir / "vllm_augmented_base"
        self.rollout_engine: VLLMSIDRolloutEngine | None = None

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
            tokenizer = load_tokenizer(export_tokenizer_path, trust_remote_code=True)
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "right"
            sid_index = load_sid_index(export_sid_index_path)
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                original_vocab_size = int(manifest["original_vocab_size"])
            else:
                base_tokenizer = load_tokenizer(
                    self.hparams.pretrained_model_name_or_path,
                    trust_remote_code=True,
                )
                original_vocab_size = len(base_tokenizer)
            return tokenizer, sid_index, original_vocab_size

        return build_tokenizer(
            pretrained_model_name_or_path=self.hparams.pretrained_model_name_or_path,
            sid_index_path=self.hparams.sid_index_path,
        )

    def _load_item_popularity(self) -> dict[str, float] | None:
        if not self.hparams.use_long_tail_reward:
            return None
        if self.hparams.popularity_train_path is None:
            raise ValueError("popularity_train_path is required when use_long_tail_reward=True")

        cache_path = Path("data/processed/rl/item_popularity.json")
        if cache_path.exists():
            log.info("Loading item popularity from cache: %s", cache_path)
            return json.loads(cache_path.read_text(encoding="utf-8"))

        log.info("Computing item popularity from training data...")
        counts = Counter(str(row["item_id"]) for row in load_split_csv(Path(self.hparams.popularity_train_path)))
        max_log_count = max(math.log1p(count) for count in counts.values())
        popularity = {item_id: math.log1p(count) / max_log_count for item_id, count in counts.items()}

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(popularity, indent=2), encoding="utf-8")

        return popularity

    def _target_token_ids(self, target_sid: str) -> tuple[int, ...]:
        return tuple(self.tokenizer.encode(target_sid, add_special_tokens=False))

    def _build_quantization_config(self) -> BitsAndBytesConfig | None:
        if self.hparams.load_in_4bit and self.hparams.load_in_8bit:
            raise ValueError("Only one of load_in_4bit and load_in_8bit can be enabled")
        if not self.hparams.load_in_4bit and not self.hparams.load_in_8bit:
            return None
        if not torch.cuda.is_available():
            raise RuntimeError("Quantized RL LoRA loading requires CUDA. Disable load_in_4bit/load_in_8bit for CPU runs.")
        _import_bitsandbytes()
        if self.hparams.load_in_4bit:
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=_resolve_dtype(self.hparams.bnb_4bit_compute_dtype),
                bnb_4bit_quant_type=self.hparams.bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=self.hparams.bnb_4bit_use_double_quant,
            )
        return BitsAndBytesConfig(load_in_8bit=True)

    def _build_adapter_model(self) -> torch.nn.Module:
        adapter_dir = self.adapter_dir
        if not adapter_dir.exists():
            raise FileNotFoundError(f"Missing SFT adapter export: {adapter_dir}")
        PeftModel, prepare_model_for_kbit_training = _import_peft_rl()
        model_kwargs: dict[str, Any] = {
            "torch_dtype": _resolve_dtype(self.hparams.torch_dtype),
            "trust_remote_code": True,
        }
        quantization_config = self._build_quantization_config()
        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config
        if self.hparams.attn_implementation is not None:
            model_kwargs["attn_implementation"] = self.hparams.attn_implementation
        base_model = AutoModelForCausalLM.from_pretrained(self.hparams.pretrained_model_name_or_path, **model_kwargs)
        base_model.resize_token_embeddings(len(self.tokenizer))
        self._load_new_token_embeddings_into_base(base_model, adapter_dir)
        if quantization_config is not None:
            base_model = prepare_model_for_kbit_training(
                base_model,
                use_gradient_checkpointing=self.hparams.gradient_checkpointing,
            )
        elif self.hparams.gradient_checkpointing:
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

    def _load_new_token_embeddings_into_base(self, base_model: torch.nn.Module, adapter_dir: Path) -> None:
        rows = _new_embedding_rows_from_adapter(adapter_dir, self.original_vocab_size, len(self.tokenizer))
        input_embedding = base_model.get_input_embeddings()
        if input_embedding is None:
            raise ValueError("Base model must expose input embeddings")
        input_rows = rows["input_embeddings"].to(device=input_embedding.weight.device, dtype=input_embedding.weight.dtype)
        with torch.no_grad():
            input_embedding.weight[self.original_vocab_size : len(self.tokenizer)].copy_(input_rows)

            output_embedding = base_model.get_output_embeddings()
            if output_embedding is None or output_embedding.weight is input_embedding.weight:
                return
            output_rows = rows.get("output_embeddings", rows["input_embeddings"])
            output_rows = output_rows.to(device=output_embedding.weight.device, dtype=output_embedding.weight.dtype)
            output_embedding.weight[self.original_vocab_size : len(self.tokenizer)].copy_(output_rows)

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
        mode = "lora"
        if self.hparams.load_in_4bit:
            mode = "lora_4bit"
        elif self.hparams.load_in_8bit:
            mode = "lora_8bit"
        log.info(
            "RL train mode=%s trainable_params=%s total_params=%s ratio=%.4f%%",
            mode,
            trainable_params,
            total_params,
            ratio,
        )

    def _set_actor(self) -> None:
        self.model.set_adapter("actor")
        self._freeze_to_actor_lora_parameters(self.model)

    def _set_reference(self) -> None:
        self.model.set_adapter("reference")
        self._freeze_to_actor_lora_parameters(self.model)

    def _save_vllm_actor_adapter(self) -> Path:
        self.vllm_adapter_step += 1
        adapter_dir = self.vllm_runtime_adapter_dir / f"step_{self.vllm_adapter_step:08d}"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        peft_save_dir = adapter_dir / "_peft"
        self.model.save_pretrained(peft_save_dir, selected_adapters=["actor"], save_embedding_layers=False)
        actor_save_dir = peft_save_dir / "actor"
        source_dir = actor_save_dir if actor_save_dir.exists() else peft_save_dir
        for path in source_dir.iterdir():
            if path.is_file():
                shutil.move(str(path), adapter_dir / path.name)
        shutil.rmtree(peft_save_dir)
        return adapter_dir

    def _ensure_vllm_rollout_engine(self) -> VLLMSIDRolloutEngine:
        if self.rollout_engine is not None:
            return self.rollout_engine
        tokenizer_path = self.sft_export_dir / "tokenizer"
        if not tokenizer_path.exists():
            raise FileNotFoundError(f"Missing tokenizer export for vLLM rollout: {tokenizer_path}")
        augmented_base_dir = prepare_augmented_vllm_base_dir(
            base_model_path=self.hparams.pretrained_model_name_or_path,
            tokenizer_dir=tokenizer_path,
            adapter_dir=self.adapter_dir,
            output_dir=self.vllm_augmented_base_dir,
            original_vocab_size=self.original_vocab_size,
            augmented_vocab_size=len(self.tokenizer),
            torch_dtype=self.hparams.torch_dtype,
        )
        actor_adapter_dir = self._save_vllm_actor_adapter()
        self.rollout_engine = VLLMSIDRolloutEngine(
            model_path=str(augmented_base_dir),
            tokenizer_path=augmented_base_dir,
            tokenizer=self.tokenizer,
            trie=self.trie,
            adapter_path=actor_adapter_dir,
            num_generations=self.hparams.num_generations,
            max_sid_length=self.max_sid_length,
            max_length=self.hparams.max_length,
            dtype=self.hparams.torch_dtype,
            seed=int(self.trainer.global_rank if self.trainer is not None else 0),
            max_lora_rank=self.hparams.vllm_max_lora_rank,
            gpu_memory_utilization=self.hparams.vllm_gpu_memory_utilization,
            enable_prefix_caching=self.hparams.vllm_enable_prefix_caching,
        )
        return self.rollout_engine

    def _refresh_vllm_actor_adapter(self) -> None:
        if self.rollout_engine is None:
            return
        self.rollout_engine.set_adapter_path(self._save_vllm_actor_adapter())

    def _rollout_one(self, prompt: str) -> tuple[list[RLCandidate], RLRolloutStats]:
        if self.hparams.rollout_backend == "vllm":
            was_training = self.model.training
            self.model.eval()
            self._set_actor()
            with torch.inference_mode():
                candidates, stats = self._ensure_vllm_rollout_engine().rollout(prompt)
            if was_training:
                self.model.train()
            return candidates, stats
        if self.hparams.rollout_backend != "hf":
            raise ValueError(f"Unsupported rollout_backend={self.hparams.rollout_backend!r}")
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
                max_length=self.hparams.max_length,
            )
        if was_training:
            self.model.train()
        return candidates, stats

    def _build_rollout_batch(self, batch: dict[str, Any]) -> RolloutBatch:
        flat_prompts: list[str] = []
        flat_token_ids: list[tuple[int, ...]] = []
        advantages_by_group = []
        rewards_by_group = []
        exact_rewards_by_group = []
        rank_rewards_by_group = []
        partial_rewards_by_group = []
        long_tail_rewards_by_group = []
        ranking_metric_rows = []
        target_hits = 0
        short_groups = 0
        invalid_count = 0
        duplicate_count = 0
        completed_count = 0

        prompts = batch["prompts"]  # list
        target_item_ids = batch["target_item_ids"]  # list
        target_sids = batch["target_sids"]  # list
        ranking_ks = tuple(sorted({1, 3, 5, 10, int(self.hparams.num_generations)}))
        for prompt, target_item_id, target_sid in zip(prompts, target_item_ids, target_sids):
            candidates, stats = self._rollout_one(prompt)
            if not candidates:
                raise ValueError("Constrained beam rollout produced no valid candidates")
            reward_breakdown = compute_reward_breakdown(
                candidates=candidates,
                target_item_id=target_item_id,
                rank_reward_lambda=self.hparams.rank_reward_lambda,
                use_partial_match_reward=self.hparams.use_partial_match_reward,
                partial_match_lambda=self.hparams.partial_match_lambda,
                target_token_ids=self._target_token_ids(target_sid),
                use_long_tail_reward=self.hparams.use_long_tail_reward,
                long_tail_reward_lambda=self.hparams.long_tail_reward_lambda,
                item_popularity=self.item_popularity,
            )
            advantages = normalize_group_advantages(reward_breakdown.total)  # [Nb,]
            flat_prompts.extend([prompt] * len(candidates))  # [Nb] * str
            flat_token_ids.extend([candidate.token_ids for candidate in candidates])  # [Nb] * tuple[int, ...]
            advantages_by_group.append(advantages)
            rewards_by_group.append(reward_breakdown.total)
            exact_rewards_by_group.append(reward_breakdown.exact)
            rank_rewards_by_group.append(reward_breakdown.rank)
            partial_rewards_by_group.append(reward_breakdown.partial)
            long_tail_rewards_by_group.append(reward_breakdown.long_tail)
            ranking_metric_rows.append(
                compute_ranking_monitor_metrics(candidates, target_item_id=target_item_id, ks=ranking_ks)
            )
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
            exact_rewards=torch.cat(exact_rewards_by_group),  # [C,]
            rank_rewards=torch.cat(rank_rewards_by_group),  # [C,]
            partial_rewards=torch.cat(partial_rewards_by_group),  # [C,]
            long_tail_rewards=torch.cat(long_tail_rewards_by_group),  # [C,]
            batch_size=batch_size,  # B
            candidate_count=candidate_count,  # C
            target_in_beam=target_hits / batch_size,
            short_group_rate=short_groups / batch_size,
            invalid_rate=invalid_count / max(normalizer, 1),
            duplicate_rate=duplicate_count / max(normalizer, 1),
            rec_metrics=mean_metric_rows(ranking_metric_rows),
        )

    def _compute_adapter_logprobs(
        self,
        prompts: list[str],
        token_ids: list[tuple[int, ...]],
        adapter_name: str,
        grad: bool,
        logits_to_keep: int | None = None,
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
        if logits_to_keep is None:
            max_response_len = max(len(ids) for ids in token_ids)
            logits_to_keep = max_response_len + 1
        context = nullcontext() if grad else torch.inference_mode()
        with context:
            try:
                outputs = self.model(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                    use_cache=False,
                    logits_to_keep=logits_to_keep,
                )
            except TypeError:
                outputs = self.model(input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"], use_cache=False)
            logits = outputs.logits
            input_ids = encoded["input_ids"]
            response_mask = encoded["response_mask"]
            if logits.shape[1] != logits_to_keep:
                logits = logits[:, -logits_to_keep:, :]
            input_ids = input_ids[:, -logits.shape[1] :]
            response_mask = response_mask[:, -logits.shape[1] :]
            logprobs: ResponseLogProbs = gather_response_logprobs(
                logits, input_ids, response_mask
            )
        if was_training:
            self.model.train()
        return logprobs

    def _logprob_chunks(
        self,
        prompts: list[str],
        token_ids: list[tuple[int, ...]],
    ) -> list[tuple[int, int]]:
        chunk_size = self.hparams.logprob_micro_batch_size or len(token_ids)
        if chunk_size <= 0:
            raise ValueError(f"logprob_micro_batch_size must be positive, got {chunk_size}")
        return [(start, min(start + chunk_size, len(token_ids))) for start in range(0, len(token_ids), chunk_size)]

    def _compute_adapter_logprobs_chunks(
        self,
        prompts: list[str],
        token_ids: list[tuple[int, ...]],
        adapter_name: str,
        grad: bool,
    ) -> list[ResponseLogProbs]:
        chunks = self._logprob_chunks(prompts, token_ids)
        return [
            self._compute_adapter_logprobs(
                prompts[start:end],
                token_ids[start:end],
                adapter_name=adapter_name,
                grad=grad,
                logits_to_keep=max(len(ids) for ids in token_ids[start:end]) + 1,
            )
            for start, end in chunks
        ]

    def _log_rollout_metrics(self, rollout: RolloutBatch, stage: str, on_step: bool, prog_bar: bool) -> None:
        self.log(
            f"{stage}/reward_mean",
            rollout.rewards.mean(),
            prog_bar=prog_bar,
            on_step=on_step,
            on_epoch=True,
            batch_size=rollout.candidate_count,
        )
        self.log(
            f"{stage}/reward_std",
            rollout.rewards.std(unbiased=False),
            prog_bar=False,
            on_step=on_step,
            on_epoch=True,
            batch_size=rollout.candidate_count,
        )
        self.log(
            f"{stage}/reward_nonzero_rate",
            (rollout.rewards != 0).float().mean(),
            prog_bar=False,
            on_step=on_step,
            on_epoch=True,
            batch_size=rollout.candidate_count,
        )
        self.log(
            f"{stage}/exact_reward_nonzero_rate",
            (rollout.exact_rewards != 0).float().mean(),
            prog_bar=False,
            on_step=on_step,
            on_epoch=True,
            batch_size=rollout.candidate_count,
        )
        self.log(
            f"{stage}/rank_reward_mean",
            rollout.rank_rewards.mean(),
            prog_bar=False,
            on_step=on_step,
            on_epoch=True,
            batch_size=rollout.candidate_count,
        )
        self.log(
            f"{stage}/partial_reward_mean",
            rollout.partial_rewards.mean(),
            prog_bar=False,
            on_step=on_step,
            on_epoch=True,
            batch_size=rollout.candidate_count,
        )
        self.log(
            f"{stage}/partial_reward_nonzero_rate",
            (rollout.partial_rewards != 0).float().mean(),
            prog_bar=False,
            on_step=on_step,
            on_epoch=True,
            batch_size=rollout.candidate_count,
        )
        self.log(
            f"{stage}/long_tail_reward_mean",
            rollout.long_tail_rewards.mean(),
            prog_bar=False,
            on_step=on_step,
            on_epoch=True,
            batch_size=rollout.candidate_count,
        )
        self.log(
            f"{stage}/advantage_std",
            rollout.advantages.std(unbiased=False),
            prog_bar=False,
            on_step=on_step,
            on_epoch=True,
            batch_size=rollout.candidate_count,
        )
        self.log(
            f"{stage}/rule_hit_at_g",
            rollout.target_in_beam,
            prog_bar=prog_bar,
            on_step=on_step,
            on_epoch=True,
            batch_size=rollout.batch_size,
        )
        self.log(
            f"{stage}/target_in_beam",
            rollout.target_in_beam,
            prog_bar=False,
            on_step=on_step,
            on_epoch=True,
            batch_size=rollout.batch_size,
        )
        self.log(
            f"{stage}/short_group_rate",
            rollout.short_group_rate,
            prog_bar=False,
            on_step=on_step,
            on_epoch=True,
            batch_size=rollout.batch_size,
        )
        self.log(
            f"{stage}/invalid_rate",
            rollout.invalid_rate,
            prog_bar=False,
            on_step=on_step,
            on_epoch=True,
            batch_size=rollout.batch_size,
        )
        self.log(
            f"{stage}/duplicate_rate",
            rollout.duplicate_rate,
            prog_bar=False,
            on_step=on_step,
            on_epoch=True,
            batch_size=rollout.batch_size,
        )
        self.log_dict(
            {f"{stage}/rec_{key}": value for key, value in rollout.rec_metrics.items()},
            prog_bar=False,
            on_step=on_step,
            on_epoch=True,
            batch_size=rollout.batch_size,
        )

    def _accumulate_grad_batches(self) -> int:
        accumulate = int(self.hparams.optimizer_accumulate_batches)
        if not isinstance(accumulate, int):
            raise ValueError(f"RL manual optimization expects integer optimizer_accumulate_batches, got {accumulate!r}")
        if accumulate <= 0:
            raise ValueError(f"optimizer_accumulate_batches must be positive, got {accumulate}")
        return accumulate

    def _should_step_optimizer(self, batch_idx: int, accumulate: int) -> bool:
        num_training_batches = self.trainer.num_training_batches
        is_last_batch = isinstance(num_training_batches, int) and batch_idx + 1 >= num_training_batches
        return (batch_idx + 1) % accumulate == 0 or is_last_batch

    def _step_scheduler(self) -> None:
        scheduler = self.lr_schedulers()
        if scheduler is None:
            return
        if isinstance(scheduler, list):
            for item in scheduler:
                item.step()
            return
        scheduler.step()

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        optimizer = self.optimizers()
        accumulate = self._accumulate_grad_batches()
        if batch_idx % accumulate == 0:
            optimizer.zero_grad(set_to_none=True)

        rollout: RolloutBatch = self._build_rollout_batch(batch)
        advantages = rollout.advantages.to(self.device)  # [C,]
        chunks = self._logprob_chunks(rollout.prompts, rollout.token_ids)
        with torch.inference_mode():
            old_logprob_chunks = self._compute_adapter_logprobs_chunks(
                rollout.prompts, rollout.token_ids, adapter_name="actor", grad=False
            )
            reference_logprob_chunks = self._compute_adapter_logprobs_chunks(
                rollout.prompts,
                rollout.token_ids,
                adapter_name="reference",
                grad=False,
            )

        total_token_count = sum(chunk.response_mask.sum() for chunk in old_logprob_chunks).to(self.device).clamp_min(1.0)
        total_loss = torch.zeros((), device=self.device)
        total_policy_loss = torch.zeros((), device=self.device)
        total_kl = torch.zeros((), device=self.device)
        kl_seq_sum = torch.zeros((), device=self.device)
        clip_token_sum = torch.zeros((), device=self.device)
        ref_delta_token_sum = torch.zeros((), device=self.device)

        for (start, end), old_logprobs, reference_logprobs in zip(chunks, old_logprob_chunks, reference_logprob_chunks):
            current_logprobs = self._compute_adapter_logprobs(
                rollout.prompts[start:end],
                rollout.token_ids[start:end],
                adapter_name="actor",
                grad=True,
                logits_to_keep=max(len(ids) for ids in rollout.token_ids[start:end]) + 1,
            )
            chunk_loss = compute_grpo_loss(
                current_logprobs=current_logprobs.token_logprobs,
                old_logprobs=old_logprobs.token_logprobs.detach(),
                reference_logprobs=reference_logprobs.token_logprobs.detach(),
                response_mask=current_logprobs.response_mask,
                advantages=advantages[start:end],
                clip_epsilon=self.hparams.clip_epsilon,
                kl_beta=self.hparams.kl_beta,
                token_count_normalizer=total_token_count,
            )
            if not torch.isfinite(chunk_loss.loss):
                raise ValueError(f"Non-finite train loss: {chunk_loss.loss}")
            self.manual_backward(chunk_loss.loss / accumulate)

            chunk_token_count = current_logprobs.response_mask.sum().detach()
            total_loss = total_loss + chunk_loss.loss.detach()
            total_policy_loss = total_policy_loss + chunk_loss.policy_loss.detach()
            total_kl = total_kl + chunk_loss.kl.detach()
            kl_seq_sum = kl_seq_sum + chunk_loss.kl_seq_mean.detach() * (end - start)
            clip_token_sum = clip_token_sum + chunk_loss.clip_fraction.detach() * chunk_token_count
            ref_delta_token_sum = ref_delta_token_sum + chunk_loss.ref_actor_logprob_delta_mean.detach() * chunk_token_count

        if self._should_step_optimizer(batch_idx, accumulate):
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            self._step_scheduler()
            self._refresh_vllm_actor_adapter()

        loss = GRPOLoss(
            loss=total_loss,
            policy_loss=total_policy_loss,
            kl=total_kl,
            kl_seq_mean=kl_seq_sum / rollout.candidate_count,
            clip_fraction=clip_token_sum / total_token_count,
            ref_actor_logprob_delta_mean=ref_delta_token_sum / total_token_count,
        )
        self._set_actor()
        self.log("train/loss", loss.loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=rollout.batch_size)
        self.log(
            "train/policy_loss",
            loss.policy_loss,
            prog_bar=False,
            on_step=True,
            on_epoch=True,
            batch_size=rollout.batch_size,
        )
        self.log("train/kl", loss.kl, prog_bar=True, on_step=True, on_epoch=True, batch_size=rollout.batch_size)
        self.log(
            "train/kl_token_mean", loss.kl, prog_bar=False, on_step=True, on_epoch=True, batch_size=rollout.batch_size
        )
        self.log(
            "train/kl_seq_mean",
            loss.kl_seq_mean,
            prog_bar=False,
            on_step=True,
            on_epoch=True,
            batch_size=rollout.batch_size,
        )
        self.log(
            "train/clip_fraction",
            loss.clip_fraction,
            prog_bar=False,
            on_step=True,
            on_epoch=True,
            batch_size=rollout.batch_size,
        )
        self.log(
            "train/ref_actor_logprob_delta_mean",
            loss.ref_actor_logprob_delta_mean,
            prog_bar=False,
            on_step=True,
            on_epoch=True,
            batch_size=rollout.batch_size,
        )
        self._log_rollout_metrics(rollout, stage="train", on_step=True, prog_bar=True)
        return loss.loss

    def _eval_step(self, batch: dict[str, Any], stage: str) -> None:
        rollout = self._build_rollout_batch(batch)
        self._log_rollout_metrics(rollout, stage=stage, on_step=False, prog_bar=stage == "val")

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
            if (
                getattr(parameter, "_sid_row_masked", False)
                or parameter.ndim == 1
                or name.endswith(".bias")
                or "norm" in name.lower()
            ):
                no_decay_parameters.append(parameter)
            else:
                decay_parameters.append(parameter)

        optimizer = self._optimizer([
            {"params": decay_parameters},
            {"params": no_decay_parameters, "weight_decay": 0.0},
        ])
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
