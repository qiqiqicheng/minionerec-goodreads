from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from transformers import AutoTokenizer, PreTrainedTokenizerBase

SFT_TASK_NAMES = ("seq_sid_to_sid", "sid_to_title", "title_to_sid", "title_history_to_sid")
SFT_TASK_TO_ID = {task: task_id for task_id, task in enumerate(SFT_TASK_NAMES)}


@dataclass(frozen=True)
class SIDTokenizerCheckResult:
    checked_items: int
    sid_token_count: int
    min_sid_length: int
    max_sid_length: int


def load_sid_index(path: str | Path) -> dict[str, list[str]]:
    """ 
    payload: {item_id: [sid_token1, sid_token2, ...], ...}
    """
    json_path = Path(path)
    with json_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not payload:
        raise ValueError(f"Empty SID index: {json_path}")
    return payload


def collect_sid_tokens(sid_index: dict[str, list[str]]) -> list[str]:
    sid_tokens = sorted({token for tokens in sid_index.values() for token in tokens})
    if not sid_tokens:
        raise ValueError("No SID tokens found in index")
    return sid_tokens


def build_item_sid_map(sid_index: dict[str, list[str]]) -> dict[str, str]:
    return {item_id: "".join(tokens) for item_id, tokens in sid_index.items()}


def add_sid_tokens(tokenizer: PreTrainedTokenizerBase, sid_index: dict[str, list[str]]) -> list[str]:
    sid_tokens = collect_sid_tokens(sid_index)
    tokenizer.add_tokens(sid_tokens)
    return sid_tokens


def _tokenizer_compat_kwargs(
    pretrained_model_name_or_path: str | Path,
    kwargs: dict[str, object],
) -> dict[str, object]:
    config_path = Path(pretrained_model_name_or_path) / "tokenizer_config.json"
    if not config_path.exists():
        return kwargs

    config = json.loads(config_path.read_text(encoding="utf-8"))
    extra_special_tokens = config.get("extra_special_tokens")
    if not isinstance(extra_special_tokens, list):
        return kwargs

    fixed_kwargs = dict(kwargs)
    fixed_kwargs.setdefault("extra_special_tokens", {})
    fixed_kwargs.setdefault("additional_special_tokens", extra_special_tokens)
    return fixed_kwargs


def load_tokenizer(pretrained_model_name_or_path: str | Path, **kwargs: object) -> PreTrainedTokenizerBase:
    tokenizer_kwargs = _tokenizer_compat_kwargs(pretrained_model_name_or_path, kwargs)
    return AutoTokenizer.from_pretrained(pretrained_model_name_or_path, **tokenizer_kwargs)


def build_tokenizer(
    pretrained_model_name_or_path: str,
    sid_index_path: str | Path,
    trust_remote_code: bool = True,
) -> tuple[PreTrainedTokenizerBase, dict[str, list[str]], int]:
    sid_index = load_sid_index(sid_index_path)
    tokenizer = load_tokenizer(pretrained_model_name_or_path, trust_remote_code=trust_remote_code)
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must define eos_token_id")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    original_vocab_size = len(tokenizer)
    add_sid_tokens(tokenizer, sid_index)
    return tokenizer, sid_index, original_vocab_size


def _select_sid_check_items(
    sid_index: dict[str, list[str]],
    sample_size: int | None,
    seed: int,
) -> list[tuple[str, list[str]]]:
    items = list(sid_index.items())
    min_len = min(len(tokens) for _, tokens in items)
    max_len = max(len(tokens) for _, tokens in items)
    first_min_item_id, first_min_tokens = next((item_id, tokens) for item_id, tokens in items if len(tokens) == min_len)
    selected = {first_min_item_id: first_min_tokens}
    if max_len != min_len:
        item_id, tokens = next((item_id, tokens) for item_id, tokens in items if len(tokens) == max_len)
        selected[item_id] = tokens

    if sample_size is None:
        selected.update(items)
    elif sample_size > len(selected):
        remaining = [(item_id, tokens) for item_id, tokens in items if item_id not in selected]
        selected.update(random.Random(seed).sample(remaining, min(sample_size - len(selected), len(remaining))))  # noqa: S311

    return list(selected.items())


def check_sid_tokenizer_atomicity(
    tokenizer: PreTrainedTokenizerBase,
    sid_index: dict[str, list[str]],
    sample_size: int | None = 512,
    seed: int = 728,
) -> SIDTokenizerCheckResult:
    sid_tokens = collect_sid_tokens(sid_index)
    token_ids = tokenizer.convert_tokens_to_ids(sid_tokens)
    if any(token_id is None for token_id in token_ids):
        raise ValueError("Some SID tokens are missing from the tokenizer vocabulary")
    if tokenizer.unk_token_id is not None and any(token_id == tokenizer.unk_token_id for token_id in token_ids):
        raise ValueError("Some SID tokens resolve to unk_token_id")

    selected = _select_sid_check_items(sid_index, sample_size=sample_size, seed=seed)
    for item_id, tokens in selected:
        sid = "".join(tokens)
        expected_ids = tokenizer.convert_tokens_to_ids(tokens)
        actual_ids = tokenizer.encode(sid, add_special_tokens=False)
        if actual_ids != expected_ids:
            raise ValueError(
                f"SID tokenizer atomicity failed for item_id={item_id}: expected {expected_ids}, got {actual_ids}"
            )
        decoded = tokenizer.decode(actual_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        if decoded != sid:
            raise ValueError(f"SID tokenizer decode failed for item_id={item_id}: expected {sid}, got {decoded}")

    lengths = [len(tokens) for tokens in sid_index.values()]
    return SIDTokenizerCheckResult(
        checked_items=len(selected),
        sid_token_count=len(sid_tokens),
        min_sid_length=min(lengths),
        max_sid_length=max(lengths),
    )
