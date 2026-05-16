from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import PreTrainedTokenizerBase

from minionerec_goodreads.utils.sid_trie import SIDTrie


@dataclass(frozen=True)
class GeneratedItem:
    item_id: str
    sid: str
    score: float


@dataclass(frozen=True)
class GenerationStats:
    invalid_count: int
    duplicate_count: int


def format_sft_prompt(instruction: str, user_input: str) -> str:
    return (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{instruction}\n\n"
        f"### User Input:\n{user_input}\n\n"
        "### Response:\n"
    )


def format_seq_sid_prompt(history_sids: list[str]) -> str:
    if not history_sids:
        raise ValueError("history_sids must not be empty")
    history_text = ", ".join(history_sids)
    instruction = "Predict the semantic ID of the next book from the chronological interaction history."
    user_input = (
        f"The user has interacted with books {history_text} in chronological order. "
        "Predict the semantic ID of the next book."
    )
    return format_sft_prompt(instruction=instruction, user_input=user_input)


def format_title_history_prompt(history_titles: list[str]) -> str:
    if not history_titles:
        raise ValueError("history_titles must not be empty")
    title_text = ", ".join(f'"{title}"' for title in history_titles)
    instruction = "Predict the semantic ID of the next book from the chronological title history."
    user_input = (
        "The user has interacted with the following book titles in chronological order: "
        f"{title_text}. Predict the semantic ID of the next book."
    )
    return format_sft_prompt(instruction=instruction, user_input=user_input)


def constrained_sid_beam_search(  # noqa: C901
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    trie: SIDTrie,
    prompt: str,
    num_beams: int,
    max_sid_length: int,
) -> tuple[list[GeneratedItem], GenerationStats]:
    if num_beams <= 0:
        raise ValueError(f"num_beams must be positive, got {num_beams}")
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
        candidates: list[tuple[tuple[int, ...], float]] = []
        for prefix, score in beams:
            next_token_ids = trie.next_token_ids(prefix)
            if not next_token_ids:
                invalid_count += 1
                continue
            input_ids = prompt_ids if not prefix else torch.cat(
                [prompt_ids, torch.tensor([prefix], dtype=torch.long, device=device)],
                dim=1,
            )
            full_attention_mask = attention_mask if not prefix else torch.cat(
                [attention_mask, torch.ones((1, len(prefix)), dtype=torch.long, device=device)],
                dim=1,
            )
            with torch.inference_mode():
                next_token_logits = model(input_ids=input_ids, attention_mask=full_attention_mask, use_cache=False).logits[
                    0, -1, next_token_ids
                ]
            log_probs = torch.log_softmax(next_token_logits, dim=-1)
            k = min(num_beams, len(next_token_ids))
            top_scores, top_indices = torch.topk(log_probs, k=k)
            for top_score, top_index in zip(top_scores.tolist(), top_indices.tolist()):
                token_id = next_token_ids[top_index]
                next_prefix = (*prefix, token_id)
                next_score = score + top_score
                item_id = trie.item_id(next_prefix)
                if item_id is not None:
                    completed.append((next_prefix, next_score))
                if trie.next_token_ids(next_prefix):
                    candidates.append((next_prefix, next_score))
        beams = sorted(candidates, key=lambda row: row[1], reverse=True)[:num_beams]
        if len(completed) >= num_beams and not beams:
            break

    seen_item_ids: set[str] = set()
    generated: list[GeneratedItem] = []
    duplicate_count = 0
    for token_ids, score in sorted(completed, key=lambda row: row[1], reverse=True):
        item_id = trie.item_id(token_ids)
        if item_id is None:
            invalid_count += 1
            continue
        if item_id in seen_item_ids:
            duplicate_count += 1
            continue
        seen_item_ids.add(item_id)
        sid = tokenizer.decode(token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        generated.append(GeneratedItem(item_id=item_id, sid=sid, score=score))
        if len(generated) == num_beams:
            break

    return generated, GenerationStats(invalid_count=invalid_count, duplicate_count=duplicate_count)
