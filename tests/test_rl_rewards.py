import torch

from minionerec_goodreads.models.rl import (
    RLCandidate,
    compute_rank_rewards,
    compute_rewards,
    compute_rule_rewards,
    normalize_group_advantages,
)


def _candidate(item_id: str, rank: int) -> RLCandidate:
    return RLCandidate(item_id=item_id, sid=f"<sid_{item_id}>", token_ids=(rank + 1,), score=-float(rank), rank=rank)


def test_rule_reward_marks_only_target_candidate() -> None:
    candidates = [_candidate("1", 0), _candidate("2", 1), _candidate("3", 2)]

    assert compute_rule_rewards(candidates, target_item_id="2") == [0.0, 1.0, 0.0]


def test_rank_reward_penalizes_high_rank_false_positive_more() -> None:
    candidates = [_candidate("1", 0), _candidate("2", 1), _candidate("3", 2)]

    rewards = compute_rank_rewards(candidates, target_item_id="3")

    assert rewards == [-1.0, -2.0 / 3.0, 0.0]


def test_combined_reward_handles_target_absent_group() -> None:
    candidates = [_candidate("1", 0), _candidate("2", 1), _candidate("3", 2)]

    rewards = compute_rewards(candidates, target_item_id="9", rank_reward_lambda=0.1)

    assert torch.allclose(rewards, torch.tensor([-0.1, -0.06666667, -0.03333334]), atol=1e-6)


def test_group_advantages_are_zero_mean_when_rewards_vary() -> None:
    rewards = torch.tensor([1.0, 0.0, -1.0])

    advantages = normalize_group_advantages(rewards)

    assert torch.allclose(advantages.mean(), torch.tensor(0.0), atol=1e-6)
    assert advantages[0] > advantages[1] > advantages[2]


def test_group_advantages_return_zero_for_all_negative_identical_rewards() -> None:
    rewards = torch.tensor([-0.1, -0.1, -0.1])

    advantages = normalize_group_advantages(rewards)

    assert torch.equal(advantages, torch.zeros_like(rewards))


def test_group_advantages_return_zero_for_single_candidate() -> None:
    rewards = torch.tensor([1.0])

    advantages = normalize_group_advantages(rewards)

    assert torch.equal(advantages, torch.zeros_like(rewards))
