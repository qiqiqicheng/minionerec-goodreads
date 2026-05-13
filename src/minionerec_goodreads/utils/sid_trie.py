from __future__ import annotations

from dataclasses import dataclass, field

from transformers import PreTrainedTokenizerBase


@dataclass
class SIDTrieNode:
    children: dict[int, SIDTrieNode] = field(default_factory=dict)
    item_id: str | None = None


class SIDTrie:
    def __init__(self, tokenizer: PreTrainedTokenizerBase, sid_index: dict[str, list[str]]):
        self.root = SIDTrieNode()
        self.sid_token_ids: set[int] = set()
        self.item_id_by_token_ids: dict[tuple[int, ...], str] = {}
        self.token_ids_by_item_id: dict[str, tuple[int, ...]] = {}
        self._build(tokenizer, sid_index)

    def _build(self, tokenizer: PreTrainedTokenizerBase, sid_index: dict[str, list[str]]) -> None:
        for item_id, tokens in sid_index.items():
            token_ids = tuple(tokenizer.convert_tokens_to_ids(tokens))
            if any(token_id is None for token_id in token_ids):
                raise ValueError(f"SID token missing from tokenizer for item_id={item_id}: {tokens}")
            node = self.root
            for token_id in token_ids:
                self.sid_token_ids.add(token_id)
                node = node.children.setdefault(token_id, SIDTrieNode())
            if node.item_id is not None:
                raise ValueError(f"Duplicate SID token path for item_id={item_id} and item_id={node.item_id}")
            node.item_id = str(item_id)
            self.item_id_by_token_ids[token_ids] = str(item_id)
            self.token_ids_by_item_id[str(item_id)] = token_ids

    def next_token_ids(self, prefix: tuple[int, ...]) -> list[int]:
        node = self._node(prefix)
        return list(node.children.keys()) if node is not None else []

    def item_id(self, token_ids: tuple[int, ...]) -> str | None:
        node = self._node(token_ids)
        return None if node is None else node.item_id

    def is_valid(self, token_ids: tuple[int, ...]) -> bool:
        return self.item_id(token_ids) is not None

    def _node(self, prefix: tuple[int, ...]) -> SIDTrieNode | None:
        node = self.root
        for token_id in prefix:
            node = node.children.get(token_id)
            if node is None:
                return None
        return node
