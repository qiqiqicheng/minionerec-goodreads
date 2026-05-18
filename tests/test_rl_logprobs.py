import json

import torch
from safetensors.torch import load_file, save_file

from minionerec_goodreads.models.rl import (
    _SIDConstrainedLogitsProcessor,
    _new_embedding_rows_from_adapter,
    constrained_sid_beam_rollout,
    compute_grpo_loss,
    gather_response_logprobs,
    prepare_lora_only_adapter_dir,
)


def _logits_for_next_tokens(input_ids: torch.Tensor, vocab_size: int) -> torch.Tensor:
    batch_size, seq_len = input_ids.shape
    logits = torch.full((batch_size, seq_len, vocab_size), -20.0)
    for row in range(batch_size):
        for pos in range(seq_len - 1):
            logits[row, pos, input_ids[row, pos + 1]] = 20.0
    return logits


def test_response_logprobs_include_first_sid_token_predicted_by_prompt() -> None:
    input_ids = torch.tensor([[10, 11, 12, 13]])
    response_mask = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
    logits = _logits_for_next_tokens(input_ids, vocab_size=20)

    logprobs = gather_response_logprobs(logits, input_ids, response_mask)

    assert logprobs.response_mask.tolist() == [[0.0, 1.0, 1.0]]
    assert torch.allclose(logprobs.sequence_logprobs, torch.zeros(1), atol=1e-6)


def test_response_logprobs_ignore_prompt_and_padding_tokens() -> None:
    input_ids = torch.tensor([[1, 2, 3, 4, 0]])
    response_mask = torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0]])
    logits = torch.zeros((1, 5, 8))
    logits[0, 0, 2] = 100.0
    logits[0, 1, 3] = 100.0
    logits[0, 2, 4] = 100.0
    logits[0, 3, 7] = 100.0

    logprobs = gather_response_logprobs(logits, input_ids, response_mask)

    assert logprobs.response_mask.tolist() == [[0.0, 0.0, 1.0, 0.0]]
    assert torch.allclose(logprobs.sequence_logprobs, torch.zeros(1), atol=1e-6)


def test_response_logprobs_do_not_change_when_prompt_logits_change() -> None:
    input_ids = torch.tensor([[1, 2, 3, 4]])
    response_mask = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
    logits = _logits_for_next_tokens(input_ids, vocab_size=8)
    changed_logits = logits.clone()
    changed_logits[0, 0, :] = 0.0

    original = gather_response_logprobs(logits, input_ids, response_mask)
    changed = gather_response_logprobs(changed_logits, input_ids, response_mask)

    assert torch.allclose(original.sequence_logprobs, changed.sequence_logprobs, atol=1e-6)


def test_grpo_loss_is_finite_and_non_negative_kl() -> None:
    current = torch.tensor([[-0.2, -0.4], [-0.3, -0.5]])
    old = current.detach().clone()
    reference = torch.tensor([[-0.25, -0.45], [-0.35, -0.55]])
    mask = torch.ones_like(current)
    advantages = torch.tensor([1.0, -1.0])

    loss = compute_grpo_loss(
        current_logprobs=current,
        old_logprobs=old,
        reference_logprobs=reference,
        response_mask=mask,
        advantages=advantages,
        clip_epsilon=0.2,
        kl_beta=0.02,
    )

    assert torch.isfinite(loss.loss)
    assert torch.isfinite(loss.policy_loss)
    assert loss.kl >= 0.0


def test_tail_response_logprobs_match_full_sequence() -> None:
    input_ids = torch.tensor([[1, 2, 3, 4, 0]])
    response_mask = torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0]])
    logits = torch.zeros((1, 5, 8))
    logits[0, 0, 2] = 100.0
    logits[0, 1, 3] = 100.0
    logits[0, 2, 4] = 100.0
    logits[0, 3, 7] = 100.0

    full = gather_response_logprobs(logits, input_ids, response_mask)
    tail = gather_response_logprobs(logits[:, -2:, :], input_ids[:, -2:], response_mask[:, -2:])

    assert torch.allclose(full.sequence_logprobs, tail.sequence_logprobs, atol=1e-6)


def test_prepare_lora_only_adapter_dir_extracts_new_embedding_rows(tmp_path) -> None:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA"}), encoding="utf-8")
    tensors = {
        "base_model.model.model.embed_tokens.weight": torch.arange(20, dtype=torch.float32).view(5, 4),
        "base_model.model.lm_head.weight": torch.arange(100, 120, dtype=torch.float32).view(5, 4),
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.ones((2, 4)),
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": torch.ones((4, 2)),
    }
    save_file(tensors, adapter_dir / "adapter_model.safetensors", metadata={"format": "pt"})

    slim_dir = prepare_lora_only_adapter_dir(adapter_dir, original_vocab_size=3, augmented_vocab_size=5)
    slim_tensors = load_file(slim_dir / "adapter_model.safetensors")
    rows = _new_embedding_rows_from_adapter(slim_dir, original_vocab_size=3, augmented_vocab_size=5)

    assert "base_model.model.model.embed_tokens.weight" not in slim_tensors
    assert "base_model.model.lm_head.weight" not in slim_tensors
    assert torch.equal(rows["input_embeddings"], tensors["base_model.model.model.embed_tokens.weight"][3:5])
    assert torch.equal(rows["output_embeddings"], tensors["base_model.model.lm_head.weight"][3:5])


class _TinyTrie:
    def next_token_ids(self, prefix: tuple[int, ...]) -> list[int]:
        return {
            (): [1, 2],
            (1,): [3],
        }.get(prefix, [])

    def item_id(self, token_ids: tuple[int, ...]) -> str | None:
        return {
            (1,): "prefix-item",
            (1, 3): "child-item",
            (2,): "leaf-item",
        }.get(token_ids)


def test_sid_constrained_logits_processor_normalizes_allowed_children_and_eos() -> None:
    processor = _SIDConstrainedLogitsProcessor(trie=_TinyTrie(), prompt_length=2, eos_token_id=9)
    root_scores = torch.log_softmax(torch.arange(10, dtype=torch.float32).view(1, 10), dim=-1)
    root_processed = processor(torch.tensor([[7, 8]]), root_scores)

    branch_scores = torch.log_softmax(torch.arange(20, dtype=torch.float32).view(2, 10), dim=-1)
    branch_processed = processor(torch.tensor([[7, 8, 1], [7, 8, 2]]), branch_scores)

    assert torch.isneginf(root_processed[0, 0])
    assert torch.allclose(torch.logsumexp(root_processed[0, [1, 2]], dim=0), torch.tensor(0.0), atol=1e-6)
    assert branch_processed[0, 3] == 0.0
    assert branch_processed[0, 9] == 0.0
    assert branch_processed[1, 9] == 0.0
    assert torch.isneginf(branch_processed[1, 1])


class _Encoding(dict):
    def to(self, device: torch.device) -> "_Encoding":
        return _Encoding({key: value.to(device) for key, value in self.items()})


class _TinyTokenizer:
    bos_token_id = 0
    eos_token_id = 9
    pad_token_id = 9

    def __call__(self, prompt: str, return_tensors: str, add_special_tokens: bool) -> _Encoding:
        _ = prompt, return_tensors, add_special_tokens
        return _Encoding({
            "input_ids": torch.tensor([[7, 8]], dtype=torch.long),
            "attention_mask": torch.ones((1, 2), dtype=torch.long),
        })

    def decode(
        self,
        token_ids: tuple[int, ...],
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str:
        _ = skip_special_tokens, clean_up_tokenization_spaces
        return " ".join(str(token_id) for token_id in token_ids)


class _GenerateOutput:
    def __init__(self, sequences: torch.Tensor, sequences_scores: torch.Tensor):
        self.sequences = sequences
        self.sequences_scores = sequences_scores


class _TinyGenerateModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(()))
        self.generate_kwargs = None

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        return _GenerateOutput(
            sequences=torch.tensor([[0, 7, 8, 1, 3]], dtype=torch.long),
            sequences_scores=torch.tensor([-0.25], dtype=torch.float32),
        )


def test_hf_rollout_uses_generate_with_kv_cache() -> None:
    model = _TinyGenerateModel()
    candidates, stats = constrained_sid_beam_rollout(
        model=model,
        tokenizer=_TinyTokenizer(),
        trie=_TinyTrie(),
        prompt="prompt",
        num_generations=1,
        max_sid_length=2,
        max_length=8,
    )

    assert model.generate_kwargs["use_cache"] is True
    assert model.generate_kwargs["max_new_tokens"] == 2
    assert candidates[0].item_id == "child-item"
    assert candidates[0].token_ids == (1, 3)
    assert candidates[0].score == -0.25
    assert stats.invalid_count == 0
