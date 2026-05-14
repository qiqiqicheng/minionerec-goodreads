import torch

from minionerec_goodreads.models.rl import compute_grpo_loss, gather_response_logprobs


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
