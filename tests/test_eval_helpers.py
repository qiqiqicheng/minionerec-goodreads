from minionerec_goodreads.scripts.eval import history_slice_name, parse_num_samples


def test_parse_num_samples_accepts_null_values() -> None:
    assert parse_num_samples("null") is None
    assert parse_num_samples("None") is None
    assert parse_num_samples("") is None


def test_parse_num_samples_accepts_positive_integer() -> None:
    assert parse_num_samples("8") == 8


def test_history_slice_name_uses_20_and_50_boundaries() -> None:
    boundaries = [20, 50]

    assert history_slice_name(20, boundaries) == "history_len_0_20"
    assert history_slice_name(21, boundaries) == "history_len_21_50"
    assert history_slice_name(50, boundaries) == "history_len_21_50"
    assert history_slice_name(51, boundaries) == "history_len_51_plus"
