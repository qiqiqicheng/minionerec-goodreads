import torch

from minionerec_goodreads.models.rl import (
    RLCandidate,
    compute_long_tail_rewards,
    compute_partial_match_rewards,
    compute_rank_rewards,
    compute_reward_breakdown,
    compute_rewards,
    compute_rule_rewards,
    normalize_group_advantages,
)


def _candidate(item_id: str, rank: int, token_ids: tuple[int, ...] | None = None) -> RLCandidate:
    token_ids = token_ids or (rank + 1,)
    return RLCandidate(item_id=item_id, sid=f"<sid_{item_id}>", token_ids=token_ids, score=-float(rank), rank=rank)


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


def test_partial_match_reward_increases_with_longer_prefix() -> None:
    candidates = [
        _candidate("1", 0, token_ids=(1, 2, 9)),
        _candidate("2", 1, token_ids=(1, 2, 3)),
        _candidate("3", 2, token_ids=(1, 8, 9)),
    ]

    rewards = compute_partial_match_rewards(candidates, target_item_id="9", target_token_ids=(1, 2, 3, 4))

    assert rewards == [0.5, 0.75, 0.25]


def test_partial_match_reward_does_not_double_count_exact_hit() -> None:
    candidates = [_candidate("9", 0, token_ids=(1, 2, 3, 4))]

    rewards = compute_partial_match_rewards(candidates, target_item_id="9", target_token_ids=(1, 2, 3, 4))

    assert rewards == [0.0]


def test_disabled_partial_reward_preserves_old_combined_reward() -> None:
    candidates = [_candidate("1", 0, token_ids=(1, 2)), _candidate("2", 1, token_ids=(1, 3))]

    rewards = compute_rewards(
        candidates,
        target_item_id="9",
        rank_reward_lambda=0.1,
        use_partial_match_reward=False,
        partial_match_lambda=0.2,
        target_token_ids=(1, 2),
    )

    assert torch.allclose(rewards, torch.tensor([-0.1, -0.05]), atol=1e-6)


def test_long_tail_reward_penalizes_popular_wrong_candidate_more() -> None:
    candidates = [_candidate("popular", 0), _candidate("tail", 1)]

    rewards = compute_long_tail_rewards(
        candidates,
        target_item_id="target",
        item_popularity={"popular": 1.0, "tail": 0.2},
    )

    assert rewards == [-1.0, -0.2]


def test_disabled_long_tail_reward_removes_component() -> None:
    candidates = [_candidate("popular", 0)]

    breakdown = compute_reward_breakdown(
        candidates,
        target_item_id="target",
        rank_reward_lambda=0.1,
        use_long_tail_reward=False,
        long_tail_reward_lambda=0.05,
        item_popularity={"popular": 1.0},
    )

    assert torch.equal(breakdown.long_tail, torch.zeros(1))


def test_reward_breakdown_total_matches_weighted_components() -> None:
    candidates = [_candidate("1", 0, token_ids=(1, 2, 9)), _candidate("target", 1, token_ids=(1, 2, 3))]

    breakdown = compute_reward_breakdown(
        candidates,
        target_item_id="target",
        rank_reward_lambda=0.1,
        use_partial_match_reward=True,
        partial_match_lambda=0.2,
        target_token_ids=(1, 2, 3),
        use_long_tail_reward=True,
        long_tail_reward_lambda=0.05,
        item_popularity={"1": 0.8, "target": 0.1},
    )
    expected = breakdown.exact + 0.1 * breakdown.rank + 0.2 * breakdown.partial + 0.05 * breakdown.long_tail

    assert torch.allclose(breakdown.total, expected, atol=1e-6)


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
