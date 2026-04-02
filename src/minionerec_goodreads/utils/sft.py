from __future__ import annotations

import json
from pathlib import Path

from transformers import AutoTokenizer, PreTrainedTokenizerBase


def load_sid_index(path: str | Path) -> dict[str, list[str]]:
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


def build_tokenizer(
    pretrained_model_name_or_path: str,
    sid_index_path: str | Path,
    trust_remote_code: bool = True,
) -> tuple[PreTrainedTokenizerBase, dict[str, list[str]], int]:
    sid_index = load_sid_index(sid_index_path)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, trust_remote_code=trust_remote_code)
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must define eos_token_id")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    original_vocab_size = len(tokenizer)
    add_sid_tokens(tokenizer, sid_index)
    return tokenizer, sid_index, original_vocab_size
