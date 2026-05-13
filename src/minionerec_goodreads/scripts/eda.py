from __future__ import annotations

import ast
import csv
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any

import hydra
import numpy as np
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from minionerec_goodreads.utils import RankedLogger, extras, task_wrapper

log = RankedLogger(__name__, rank_zero_only=True)

TIME_FORMAT = "%a %b %d %H:%M:%S %z %Y"
TIMESTAMP_KEYS = ("read_at", "date_updated", "date_added", "started_at")
NOISY_SHELVES = {
    "to read",
    "currently reading",
    "read",
    "owned",
    "kindle",
    "ebook",
    "ebooks",
    "audiobook",
    "audiobooks",
    "library",
    "borrowed",
    "netgalley",
    "default",
    "favorites",
    "books i own",
    "owned books",
}
RQVAE_SPLIT_NAMES = ("train", "valid", "test")
RQVAE_SPLIT_COLUMNS = [
    "user_id",
    "history_book_ids",
    "book_id",
    "history_item_ids",
    "item_id",
    "history_titles",
    "item_title",
    "history_ratings",
    "rating",
    "history_timestamps",
    "timestamp",
]


def as_path(value: str | Path) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else Path(to_absolute_path(str(path)))


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                yield json.loads(line)


def normalize_text(text: str) -> str:
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_shelf(name: Any) -> str:
    return normalize_text(str(name).replace("-", " ").replace("_", " ")).lower()


def word_count(text: str) -> int:
    text = normalize_text(text)
    return 0 if text == "" else len(text.split())


def parse_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def parse_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def parse_datetime(value: str) -> datetime:
    return datetime.strptime(value, TIME_FORMAT)


def parse_timestamp(record: dict[str, Any]) -> tuple[str, int]:
    for key in TIMESTAMP_KEYS:
        value = record.get(key, "")
        if value:
            return key, int(parse_datetime(value).timestamp())
    raise ValueError(f"No valid timestamp found for record: {record}")


def month_from_timestamp(timestamp: int) -> str:
    dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
    return f"{dt.year:04d}-{dt.month:02d}"


def date_from_timestamp(timestamp: int) -> str:
    dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
    return f"{dt.year:04d}-{dt.month:02d}-{dt.day:02d}"


def parse_date_boundary(value: str, end_of_day: bool) -> int:
    dt = datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    if end_of_day:
        dt = dt.replace(hour=23, minute=59, second=59)
    return int(dt.timestamp())


def in_window(timestamp: int, start_timestamp: int, end_timestamp: int) -> bool:
    return start_timestamp <= timestamp <= end_timestamp


def pct(part: int | float, total: int | float) -> float:
    return 0.0 if total == 0 else part / total * 100


def quantile(sorted_values: list[int | float], q: float) -> float:
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    index = (len(sorted_values) - 1) * q
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return float(sorted_values[lower])
    lower_value = sorted_values[lower] * (upper - index)
    upper_value = sorted_values[upper] * (index - lower)
    return float(lower_value + upper_value)


def numeric_summary(values: list[int | float]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    sorted_values = sorted(values)
    return {
        "count": len(sorted_values),
        "min": sorted_values[0],
        "p25": quantile(sorted_values, 0.25),
        "p50": quantile(sorted_values, 0.50),
        "p75": quantile(sorted_values, 0.75),
        "p90": quantile(sorted_values, 0.90),
        "p95": quantile(sorted_values, 0.95),
        "p99": quantile(sorted_values, 0.99),
        "max": sorted_values[-1],
        "mean": fmean(sorted_values),
    }


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        if value.is_integer():
            return f"{int(value):,}"
        return f"{value:,.2f}"
    return str(value)


def fmt_pct(part: int | float, total: int | float) -> str:
    return f"{pct(part, total):.2f}%"


def md_escape(value: Any) -> str:
    return fmt(value).replace("|", "\\|")


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    if not rows:
        return "_No rows._"
    header = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(md_escape(cell) for cell in row) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def counter_rows(counter: Counter, top_n: int, total: int | None = None) -> list[list[Any]]:
    rows = []
    denominator = total if total is not None else sum(counter.values())
    for key, count in counter.most_common(top_n):
        rows.append([key, count, fmt_pct(count, denominator)])
    return rows


def summary_rows(summary: dict[str, Any]) -> list[list[Any]]:
    keys = ["count", "min", "p25", "p50", "p75", "p90", "p95", "p99", "max", "mean"]
    return [[key, summary.get(key)] for key in keys if key in summary]


def time_range(timestamps: list[int]) -> str:
    if not timestamps:
        return "-"
    return f"{date_from_timestamp(min(timestamps))} to {date_from_timestamp(max(timestamps))}"


def count_below(counter: Counter, threshold: int) -> int:
    return sum(1 for count in counter.values() if count < threshold)


def analyze_books(books_path: Path, min_description_words: int) -> tuple[dict[str, Any], dict[str, Any]]:
    book_ids = set()
    description_words_by_book: dict[str, int] = {}
    language_counter = Counter()
    format_counter = Counter()
    publisher_counter = Counter()
    ebook_counter = Counter()
    year_counter = Counter()
    shelf_book_counter = Counter()
    shelf_total_counter = Counter()
    topical_shelf_counter = Counter()
    author_counter = Counter()
    series_counter = Counter()
    title_counter = Counter()
    work_counter = Counter()
    missing_counter = Counter()
    title_words: list[int] = []
    description_words: list[int] = []
    shelves_per_book: list[int] = []
    authors_per_book: list[int] = []
    series_per_book: list[int] = []
    similar_books_per_book: list[int] = []
    average_ratings: list[float] = []
    ratings_counts: list[int] = []
    text_reviews_counts: list[int] = []
    num_pages: list[int] = []
    total = 0

    for book in iter_jsonl(books_path):
        total += 1
        book_id = str(book["book_id"])
        book_ids.add(book_id)
        title = normalize_text(book.get("title", ""))
        description = normalize_text(book.get("description", ""))
        description_word_count = word_count(description)
        description_words_by_book[book_id] = description_word_count
        title_words.append(word_count(title))
        description_words.append(description_word_count)

        update_book_missing_counts(book, missing_counter, min_description_words)
        update_book_numeric_fields(book, average_ratings, ratings_counts, text_reviews_counts, num_pages)
        update_book_categorical_fields(book, language_counter, format_counter, publisher_counter, ebook_counter, year_counter)
        update_book_relation_fields(book, author_counter, series_counter, authors_per_book, series_per_book)
        update_book_shelf_fields(book, shelf_book_counter, shelf_total_counter, topical_shelf_counter, shelves_per_book)
        similar_books_per_book.append(len(book.get("similar_books", [])))

        if title:
            title_counter[title.lower()] += 1
        work_id = normalize_text(str(book.get("work_id", "")))
        if work_id:
            work_counter[work_id] += 1

    metrics = {
        "path": str(books_path),
        "size_mb": books_path.stat().st_size / 1024 / 1024,
        "total_books": total,
        "unique_book_ids": len(book_ids),
        "duplicate_book_id_rows": total - len(book_ids),
        "duplicate_title_groups": sum(1 for count in title_counter.values() if count > 1),
        "duplicate_title_extra_rows": sum(count - 1 for count in title_counter.values() if count > 1),
        "duplicate_work_groups": sum(1 for count in work_counter.values() if count > 1),
        "duplicate_work_extra_rows": sum(count - 1 for count in work_counter.values() if count > 1),
        "missing": dict(missing_counter),
        "title_words": numeric_summary(title_words),
        "description_words": numeric_summary(description_words),
        "shelves_per_book": numeric_summary(shelves_per_book),
        "authors_per_book": numeric_summary(authors_per_book),
        "series_per_book": numeric_summary(series_per_book),
        "similar_books_per_book": numeric_summary(similar_books_per_book),
        "average_rating": numeric_summary(average_ratings),
        "ratings_count": numeric_summary(ratings_counts),
        "text_reviews_count": numeric_summary(text_reviews_counts),
        "num_pages": numeric_summary(num_pages),
        "languages": dict(language_counter.most_common(30)),
        "formats": dict(format_counter.most_common(30)),
        "publishers": dict(publisher_counter.most_common(30)),
        "is_ebook": dict(ebook_counter),
        "publication_years": dict(sorted(year_counter.items())),
        "top_shelves_by_books": dict(shelf_book_counter.most_common(30)),
        "top_shelves_by_counts": dict(shelf_total_counter.most_common(30)),
        "top_topical_shelves": dict(topical_shelf_counter.most_common(30)),
        "top_authors": dict(author_counter.most_common(30)),
        "top_series": dict(series_counter.most_common(30)),
    }
    context = {
        "book_ids": book_ids,
        "description_words_by_book": description_words_by_book,
    }
    return metrics, context


def update_book_missing_counts(book: dict[str, Any], missing_counter: Counter, min_description_words: int) -> None:
    scalar_fields = [
        "title",
        "title_without_series",
        "description",
        "language_code",
        "publication_year",
        "average_rating",
        "ratings_count",
        "text_reviews_count",
        "num_pages",
        "publisher",
        "isbn",
        "isbn13",
        "asin",
    ]
    for field in scalar_fields:
        if book.get(field, "") in (None, ""):
            missing_counter[field] += 1
    if not book.get("authors", []):
        missing_counter["authors"] += 1
    if not book.get("series", []):
        missing_counter["series"] += 1
    if not book.get("popular_shelves", []):
        missing_counter["popular_shelves"] += 1
    description_word_count = word_count(str(book.get("description", "")))
    if description_word_count < min_description_words:
        missing_counter[f"description_lt_{min_description_words}_words"] += 1


def update_book_numeric_fields(
    book: dict[str, Any],
    average_ratings: list[float],
    ratings_counts: list[int],
    text_reviews_counts: list[int],
    num_pages: list[int],
) -> None:
    average_rating = parse_float(book.get("average_rating"))
    ratings_count = parse_int(book.get("ratings_count"))
    text_reviews_count = parse_int(book.get("text_reviews_count"))
    page_count = parse_int(book.get("num_pages"))
    if average_rating is not None:
        average_ratings.append(average_rating)
    if ratings_count is not None:
        ratings_counts.append(ratings_count)
    if text_reviews_count is not None:
        text_reviews_counts.append(text_reviews_count)
    if page_count is not None:
        num_pages.append(page_count)


def update_book_categorical_fields(
    book: dict[str, Any],
    language_counter: Counter,
    format_counter: Counter,
    publisher_counter: Counter,
    ebook_counter: Counter,
    year_counter: Counter,
) -> None:
    language_counter[normalize_text(str(book.get("language_code") or "<missing>"))] += 1
    format_counter[normalize_text(str(book.get("format") or "<missing>"))] += 1
    publisher_counter[normalize_text(str(book.get("publisher") or "<missing>"))] += 1
    ebook_counter[str(book.get("is_ebook"))] += 1
    year = parse_int(book.get("publication_year"))
    if year is not None:
        year_counter[year] += 1


def update_book_relation_fields(
    book: dict[str, Any],
    author_counter: Counter,
    series_counter: Counter,
    authors_per_book: list[int],
    series_per_book: list[int],
) -> None:
    authors = [str(author.get("author_id")) for author in book.get("authors", []) if author.get("author_id")]
    series = [str(series_id) for series_id in book.get("series", []) if str(series_id).strip()]
    authors_per_book.append(len(authors))
    series_per_book.append(len(series))
    author_counter.update(authors)
    series_counter.update(series)


def update_book_shelf_fields(
    book: dict[str, Any],
    shelf_book_counter: Counter,
    shelf_total_counter: Counter,
    topical_shelf_counter: Counter,
    shelves_per_book: list[int],
) -> None:
    shelf_names = set()
    for shelf in book.get("popular_shelves", []):
        name = normalize_shelf(shelf.get("name", ""))
        if name == "":
            continue
        shelf_names.add(name)
        shelf_total_counter[name] += parse_int(shelf.get("count")) or 0
        if name not in NOISY_SHELVES:
            topical_shelf_counter[name] += 1
    shelves_per_book.append(len(shelf_names))
    shelf_book_counter.update(shelf_names)


def analyze_interactions(
    interactions_path: Path,
    book_ids: set[str],
    history_max_len: int,
    target_start: str,
    target_end: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    start_timestamp = parse_date_boundary(target_start, end_of_day=False)
    end_timestamp = parse_date_boundary(target_end, end_of_day=True)
    all_user_counter = Counter()
    all_book_counter = Counter()
    valid_user_counter = Counter()
    valid_book_counter = Counter()
    target_window_user_counter = Counter()
    target_window_book_counter = Counter()
    rating_counter = Counter()
    valid_rating_counter = Counter()
    target_window_rating_counter = Counter()
    read_rating_counter = Counter()
    timestamp_source_counter = Counter()
    month_counter = Counter()
    target_window_month_counter = Counter()
    pair_counter = Counter()
    valid_events_by_user: dict[str, list[tuple[int, str, int]]] = defaultdict(list)
    valid_timestamps: list[int] = []
    target_window_timestamps: list[int] = []
    missing_book_valid_rows = 0
    missing_book_all_rows = 0
    total = 0
    valid_total = 0
    target_window_total = 0

    for row in iter_jsonl(interactions_path):
        total += 1
        user_id = str(row["user_id"])
        book_id = str(row["book_id"])
        rating = int(row.get("rating", 0))
        is_read = bool(row.get("is_read"))
        all_user_counter[user_id] += 1
        all_book_counter[book_id] += 1
        rating_counter[rating] += 1
        read_rating_counter[(is_read, "rating>0" if rating > 0 else "rating=0")] += 1
        if book_id not in book_ids:
            missing_book_all_rows += 1
        if not is_read or rating <= 0:
            continue

        source, timestamp = parse_timestamp(row)
        valid_total += 1
        valid_user_counter[user_id] += 1
        valid_book_counter[book_id] += 1
        valid_rating_counter[rating] += 1
        timestamp_source_counter[source] += 1
        month_counter[month_from_timestamp(timestamp)] += 1
        pair_counter[(user_id, book_id)] += 1
        valid_timestamps.append(timestamp)
        valid_events_by_user[user_id].append((timestamp, book_id, rating))
        if book_id not in book_ids:
            missing_book_valid_rows += 1
        if in_window(timestamp, start_timestamp, end_timestamp):
            target_window_total += 1
            target_window_user_counter[user_id] += 1
            target_window_book_counter[book_id] += 1
            target_window_rating_counter[rating] += 1
            target_window_month_counter[month_from_timestamp(timestamp)] += 1
            target_window_timestamps.append(timestamp)

    sequence_metrics, split_context = build_sequence_metrics(valid_events_by_user, history_max_len, start_timestamp, end_timestamp)
    metrics = {
        "path": str(interactions_path),
        "size_mb": interactions_path.stat().st_size / 1024 / 1024,
        "total_rows": total,
        "unique_users_all": len(all_user_counter),
        "unique_books_all": len(all_book_counter),
        "missing_book_all_rows": missing_book_all_rows,
        "rating_distribution_all": dict(sorted(rating_counter.items())),
        "read_rating_distribution": {f"{key[0]}:{key[1]}": value for key, value in read_rating_counter.items()},
        "valid_rows": valid_total,
        "valid_rows_pct": pct(valid_total, total),
        "valid_unique_users": len(valid_user_counter),
        "valid_unique_books": len(valid_book_counter),
        "target_window_start": target_start,
        "target_window_end": target_end,
        "target_window_rows": target_window_total,
        "target_window_rows_pct_valid": pct(target_window_total, valid_total),
        "target_window_unique_users": len(target_window_user_counter),
        "target_window_unique_books": len(target_window_book_counter),
        "target_window_time_range": time_range(target_window_timestamps),
        "target_window_rating_distribution": dict(sorted(target_window_rating_counter.items())),
        "target_window_months": dict(sorted(target_window_month_counter.items())),
        "missing_book_valid_rows": missing_book_valid_rows,
        "valid_rating_distribution": dict(sorted(valid_rating_counter.items())),
        "timestamp_sources": dict(timestamp_source_counter),
        "valid_time_range": time_range(valid_timestamps),
        "valid_months": dict(sorted(month_counter.items())),
        "all_user_degree": numeric_summary(list(all_user_counter.values())),
        "all_book_degree": numeric_summary(list(all_book_counter.values())),
        "valid_user_degree": numeric_summary(list(valid_user_counter.values())),
        "valid_book_degree": numeric_summary(list(valid_book_counter.values())),
        "target_window_user_degree": numeric_summary(list(target_window_user_counter.values())),
        "target_window_book_degree": numeric_summary(list(target_window_book_counter.values())),
        "valid_users_lt_2": count_below(valid_user_counter, 2),
        "valid_users_lt_5": count_below(valid_user_counter, 5),
        "valid_users_lt_30": count_below(valid_user_counter, 30),
        "valid_books_lt_2": count_below(valid_book_counter, 2),
        "valid_books_lt_5": count_below(valid_book_counter, 5),
        "valid_books_lt_30": count_below(valid_book_counter, 30),
        "duplicate_user_book_pairs": sum(1 for count in pair_counter.values() if count > 1),
        "duplicate_user_book_extra_rows": sum(count - 1 for count in pair_counter.values() if count > 1),
        "sequence": sequence_metrics,
    }
    context = {
        "valid_book_ids": set(valid_book_counter),
        "valid_user_ids": set(valid_user_counter),
        "target_window_book_ids": set(target_window_book_counter),
        "target_window_user_ids": set(target_window_user_counter),
        "split": split_context,
    }
    return metrics, context


def build_sequence_metrics(
    valid_events_by_user: dict[str, list[tuple[int, str, int]]],
    history_max_len: int,
    target_start_timestamp: int,
    target_end_timestamp: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    row_events: list[tuple[int, str, str, int, int]] = []
    history_lengths: list[int] = []
    target_window_history_lengths: list[int] = []
    user_spans_days: list[float] = []
    single_event_users = 0
    repeated_item_users = 0
    repeated_item_extra_events = 0
    adjacent_same_timestamp_pairs = 0
    capped_rows = 0
    target_window_rows = 0

    for user_id, events in valid_events_by_user.items():
        events.sort(key=lambda item: item[0])
        books = [book_id for _, book_id, _ in events]
        unique_books = set(books)
        if len(events) == 1:
            single_event_users += 1
        if len(unique_books) < len(books):
            repeated_item_users += 1
            repeated_item_extra_events += len(books) - len(unique_books)
        if len(events) > 1:
            user_spans_days.append((events[-1][0] - events[0][0]) / 86400)
        adjacent_same_timestamp_pairs += sum(1 for left, right in zip(events, events[1:]) if left[0] == right[0])

        for index in range(1, len(events)):
            target_timestamp, target_book_id, target_rating = events[index]
            history_len = min(index, history_max_len)
            capped_rows += int(index > history_max_len)
            history_lengths.append(history_len)
            if in_window(target_timestamp, target_start_timestamp, target_end_timestamp):
                target_window_rows += 1
                target_window_history_lengths.append(history_len)
            row_events.append((target_timestamp, user_id, target_book_id, target_rating, history_len))

    row_events.sort(key=lambda item: item[0])
    split_metrics = summarize_splits(row_events, target_start_timestamp, target_end_timestamp)
    metrics = {
        "generated_rows": len(row_events),
        "target_window_generated_rows": target_window_rows,
        "target_window_generated_rows_pct": pct(target_window_rows, len(row_events)),
        "history_max_len": history_max_len,
        "history_length": numeric_summary(history_lengths),
        "target_window_history_length": numeric_summary(target_window_history_lengths),
        "history_capped_rows": capped_rows,
        "history_capped_rows_pct": pct(capped_rows, len(row_events)),
        "single_event_users": single_event_users,
        "user_span_days": numeric_summary(user_spans_days),
        "repeated_item_users": repeated_item_users,
        "repeated_item_extra_events": repeated_item_extra_events,
        "adjacent_same_timestamp_pairs": adjacent_same_timestamp_pairs,
        "splits": split_metrics,
    }
    context = {"row_events": row_events}
    return metrics, context


def summarize_splits(row_events: list[tuple[int, str, str, int, int]], target_start_timestamp: int, target_end_timestamp: int) -> dict[str, Any]:
    train_end = int(len(row_events) * 0.8)
    valid_end = int(len(row_events) * 0.9)
    split_rows = {
        "train": row_events[:train_end],
        "valid": row_events[train_end:valid_end],
        "test": row_events[valid_end:],
    }
    train_users = {row[1] for row in split_rows["train"]}
    train_books = {row[2] for row in split_rows["train"]}
    metrics = {}
    for name, rows in split_rows.items():
        timestamps = [row[0] for row in rows]
        users = {row[1] for row in rows}
        books = {row[2] for row in rows}
        ratings = Counter(row[3] for row in rows)
        history_lengths = [row[4] for row in rows]
        target_window_rows = [row for row in rows if in_window(row[0], target_start_timestamp, target_end_timestamp)]
        metrics[name] = {
            "rows": len(rows),
            "time_range": time_range(timestamps),
            "target_window_rows": len(target_window_rows),
            "target_window_rows_pct": pct(len(target_window_rows), len(rows)),
            "unique_target_users": len(users),
            "unique_target_books": len(books),
            "new_target_users_vs_train": len(users - train_users) if name != "train" else 0,
            "new_target_books_vs_train": len(books - train_books) if name != "train" else 0,
            "rating_distribution": dict(sorted(ratings.items())),
            "history_length": numeric_summary(history_lengths),
        }
    return metrics


def analyze_reviews(
    reviews_path: Path,
    book_ids: set[str],
    valid_book_ids: set[str],
    valid_user_ids: set[str],
    description_words_by_book: dict[str, int],
    min_review_rating: int,
    min_review_words: int,
    min_description_words: int,
) -> dict[str, Any]:
    rating_counter = Counter()
    review_user_counter = Counter()
    review_book_counter = Counter()
    word_counts: list[int] = []
    votes: list[int] = []
    comments: list[int] = []
    candidate_books = set()
    candidate_reviews_by_book = Counter()
    non_empty_reviews = 0
    total = 0
    candidate_rows = 0

    for review in iter_jsonl(reviews_path):
        total += 1
        book_id = str(review["book_id"])
        user_id = str(review["user_id"])
        rating = int(review.get("rating", 0))
        text = normalize_text(review.get("review_text", ""))
        words = word_count(text)
        n_votes = int(review.get("n_votes", 0))
        n_comments = int(review.get("n_comments", 0))
        rating_counter[rating] += 1
        review_user_counter[user_id] += 1
        review_book_counter[book_id] += 1
        word_counts.append(words)
        votes.append(n_votes)
        comments.append(n_comments)
        non_empty_reviews += int(words > 0)
        if rating >= min_review_rating and words >= min_review_words:
            candidate_rows += 1
            candidate_books.add(book_id)
            candidate_reviews_by_book[book_id] += 1

    valid_short_description_books = {
        book_id for book_id in valid_book_ids if description_words_by_book.get(book_id, 0) < min_description_words
    }
    valid_empty_description_books = {book_id for book_id in valid_book_ids if description_words_by_book.get(book_id, 0) == 0}
    review_source_books = valid_short_description_books & candidate_books
    none_source_books = valid_empty_description_books - candidate_books
    description_source_books = valid_book_ids - review_source_books - none_source_books

    return {
        "path": str(reviews_path),
        "size_mb": reviews_path.stat().st_size / 1024 / 1024,
        "total_rows": total,
        "non_empty_reviews": non_empty_reviews,
        "non_empty_reviews_pct": pct(non_empty_reviews, total),
        "unique_review_users": len(review_user_counter),
        "unique_review_books": len(review_book_counter),
        "review_books_in_book_metadata": len(set(review_book_counter) & book_ids),
        "review_books_in_valid_items": len(set(review_book_counter) & valid_book_ids),
        "review_users_in_valid_interactions": len(set(review_user_counter) & valid_user_ids),
        "rating_distribution": dict(sorted(rating_counter.items())),
        "word_count": numeric_summary(word_counts),
        "n_votes": numeric_summary(votes),
        "n_comments": numeric_summary(comments),
        "fallback_candidate_reviews": candidate_rows,
        "fallback_candidate_books": len(candidate_books),
        "fallback_candidate_reviews_per_book": numeric_summary(list(candidate_reviews_by_book.values())),
        "valid_short_description_books": len(valid_short_description_books),
        "valid_empty_description_books": len(valid_empty_description_books),
        "simulated_description_source": len(description_source_books),
        "simulated_review_source": len(review_source_books),
        "simulated_none_source": len(none_source_books),
    }


def build_report(
    cfg: DictConfig,
    books: dict[str, Any],
    interactions: dict[str, Any],
    reviews: dict[str, Any],
) -> str:
    split_rows = interactions["sequence"]["splits"]
    lines = [
        "# GoodReads Mystery-Thriller-Crime Raw EDA",
        "",
        f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Configuration",
        "",
        md_table(
            ["key", "value"],
            [
                ["books_file", cfg.books_file],
                ["interactions_file", cfg.interactions_file],
                ["reviews_file", cfg.reviews_file],
                ["min_review_rating", cfg.min_review_rating],
                ["min_review_words", cfg.min_review_words],
                ["min_description_words", cfg.min_description_words],
                ["history_max_len", cfg.history_max_len],
                ["target_start", cfg.target_start],
                ["target_end", cfg.target_end],
            ],
        ),
        "",
        "## Executive Summary",
        "",
        "- The raw subset has "
        f"{fmt(books['total_books'])} books, {fmt(interactions['total_rows'])} interaction rows, "
        f"and {fmt(reviews['total_rows'])} full review rows.",
        "- The current sequential signal keeps "
        f"{fmt(interactions['valid_rows'])} read-and-rated interactions "
        f"({interactions['valid_rows_pct']:.2f}% of raw interactions), covering "
        f"{fmt(interactions['valid_unique_users'])} users and {fmt(interactions['valid_unique_books'])} books.",
        "- The project target window "
        f"{interactions['target_window_start']} to {interactions['target_window_end']} contains "
        f"{fmt(interactions['target_window_rows'])} read-and-rated interactions "
        f"({interactions['target_window_rows_pct_valid']:.2f}% of valid interactions).",
        "- The MiniOneRec-style row builder produces "
        f"{fmt(interactions['sequence']['generated_rows'])} next-item rows, with "
        f"{fmt(interactions['sequence']['target_window_generated_rows'])} targets in the project window, then uses a global "
        "chronological 80/10/10 split by target timestamp.",
        "- Item text is mostly book description, but the review fallback is important for short or empty descriptions: "
        f"{fmt(reviews['simulated_review_source'])} valid items use review text and "
        f"{fmt(reviews['simulated_none_source'])} valid items still have no usable text source.",
        "",
        "## Raw Files",
        "",
        md_table(
            ["file", "rows", "size_mb"],
            [
                [Path(books["path"]).name, books["total_books"], books["size_mb"]],
                [Path(interactions["path"]).name, interactions["total_rows"], interactions["size_mb"]],
                [Path(reviews["path"]).name, reviews["total_rows"], reviews["size_mb"]],
            ],
        ),
        "",
        "## Book Metadata",
        "",
        md_table(
            ["metric", "value"],
            [
                ["unique_book_ids", books["unique_book_ids"]],
                ["duplicate_book_id_rows", books["duplicate_book_id_rows"]],
                ["duplicate_title_groups", books["duplicate_title_groups"]],
                ["duplicate_title_extra_rows", books["duplicate_title_extra_rows"]],
                ["duplicate_work_groups", books["duplicate_work_groups"]],
                ["duplicate_work_extra_rows", books["duplicate_work_extra_rows"]],
            ],
        ),
        "",
        "### Missing / Sparse Fields",
        "",
        md_table(
            ["field", "missing_or_sparse", "pct_books"],
            [[key, value, fmt_pct(value, books["total_books"])] for key, value in sorted(books["missing"].items())],
        ),
        "",
        "### Text And Metadata Distributions",
        "",
        md_table(
            ["distribution", "count", "min", "p50", "p90", "p95", "p99", "max", "mean"],
            [
                summary_compact_row("title_words", books["title_words"]),
                summary_compact_row("description_words", books["description_words"]),
                summary_compact_row("shelves_per_book", books["shelves_per_book"]),
                summary_compact_row("authors_per_book", books["authors_per_book"]),
                summary_compact_row("series_per_book", books["series_per_book"]),
                summary_compact_row("similar_books_per_book", books["similar_books_per_book"]),
                summary_compact_row("average_rating", books["average_rating"]),
                summary_compact_row("ratings_count", books["ratings_count"]),
                summary_compact_row("text_reviews_count", books["text_reviews_count"]),
                summary_compact_row("num_pages", books["num_pages"]),
            ],
        ),
        "",
        "### Top Languages / Formats / Shelves",
        "",
        "Languages:",
        "",
        md_table(["language", "books", "pct_books"], counter_rows(Counter(books["languages"]), 15, books["total_books"])),
        "",
        "Formats:",
        "",
        md_table(["format", "books", "pct_books"], counter_rows(Counter(books["formats"]), 15, books["total_books"])),
        "",
        "Top shelves by book coverage:",
        "",
        md_table(["shelf", "books", "pct_books"], counter_rows(Counter(books["top_shelves_by_books"]), 20, books["total_books"])),
        "",
        "Top topical shelves after removing logistics shelves:",
        "",
        md_table(["shelf", "books", "pct_books"], counter_rows(Counter(books["top_topical_shelves"]), 20, books["total_books"])),
        "",
        "## Interactions",
        "",
        md_table(
            ["metric", "value"],
            [
                ["raw_rows", interactions["total_rows"]],
                ["raw_unique_users", interactions["unique_users_all"]],
                ["raw_unique_books", interactions["unique_books_all"]],
                ["read_and_rated_rows", interactions["valid_rows"]],
                ["read_and_rated_pct", f"{interactions['valid_rows_pct']:.2f}%"],
                ["valid_unique_users", interactions["valid_unique_users"]],
                ["valid_unique_books", interactions["valid_unique_books"]],
                ["target_window", f"{interactions['target_window_start']} to {interactions['target_window_end']}"],
                ["target_window_rows", interactions["target_window_rows"]],
                ["target_window_rows_pct_valid", f"{interactions['target_window_rows_pct_valid']:.2f}%"],
                ["target_window_unique_users", interactions["target_window_unique_users"]],
                ["target_window_unique_books", interactions["target_window_unique_books"]],
                ["target_window_time_range", interactions["target_window_time_range"]],
                ["missing_book_valid_rows", interactions["missing_book_valid_rows"]],
                ["valid_time_range", interactions["valid_time_range"]],
            ],
        ),
        "",
        "Raw read/rating cross-tab:",
        "",
        md_table(
            ["bucket", "rows", "pct_rows"],
            counter_rows(Counter(interactions["read_rating_distribution"]), 10, interactions["total_rows"]),
        ),
        "",
        "Target-window rating distribution:",
        "",
        md_table(
            ["rating", "rows", "pct_target_window"],
            counter_rows(
                Counter(interactions["target_window_rating_distribution"]),
                10,
                interactions["target_window_rows"],
            ),
        ),
        "",
        "Rating distribution for all read-and-rated rows:",
        "",
        md_table(
            ["rating", "rows", "pct_valid"],
            counter_rows(Counter(interactions["valid_rating_distribution"]), 10, interactions["valid_rows"]),
        ),
        "",
        "### User / Item Degree",
        "",
        md_table(
            ["distribution", "count", "min", "p50", "p90", "p95", "p99", "max", "mean"],
            [
                summary_compact_row("raw_user_degree", interactions["all_user_degree"]),
                summary_compact_row("raw_book_degree", interactions["all_book_degree"]),
                summary_compact_row("valid_user_degree", interactions["valid_user_degree"]),
                summary_compact_row("valid_book_degree", interactions["valid_book_degree"]),
                summary_compact_row("target_window_user_degree", interactions["target_window_user_degree"]),
                summary_compact_row("target_window_book_degree", interactions["target_window_book_degree"]),
            ],
        ),
        "",
        md_table(
            ["metric", "value"],
            [
                ["valid_users_lt_2", interactions["valid_users_lt_2"]],
                ["valid_users_lt_5", interactions["valid_users_lt_5"]],
                ["valid_users_lt_30", interactions["valid_users_lt_30"]],
                ["valid_books_lt_2", interactions["valid_books_lt_2"]],
                ["valid_books_lt_5", interactions["valid_books_lt_5"]],
                ["valid_books_lt_30", interactions["valid_books_lt_30"]],
                ["duplicate_user_book_pairs", interactions["duplicate_user_book_pairs"]],
                ["duplicate_user_book_extra_rows", interactions["duplicate_user_book_extra_rows"]],
            ],
        ),
        "",
        "### Target-Window Interaction Months",
        "",
        md_table(
            ["month", "valid_rows", "pct_target_window"],
            [
                [month, count, fmt_pct(count, interactions["target_window_rows"])]
                for month, count in interactions["target_window_months"].items()
            ],
        ),
        "",
        "### Top Valid Interaction Months",
        "",
        md_table(
            ["month", "valid_rows", "pct_valid"],
            counter_rows(Counter(interactions["valid_months"]), 30, interactions["valid_rows"]),
        ),
        "",
        "### Sequence Rows And Split",
        "",
        md_table(
            ["metric", "value"],
            [
                ["generated_rows", interactions["sequence"]["generated_rows"]],
                ["target_window_generated_rows", interactions["sequence"]["target_window_generated_rows"]],
                [
                    "target_window_generated_rows_pct",
                    f"{interactions['sequence']['target_window_generated_rows_pct']:.2f}%",
                ],
                ["history_max_len", interactions["sequence"]["history_max_len"]],
                ["history_capped_rows", interactions["sequence"]["history_capped_rows"]],
                ["history_capped_rows_pct", f"{interactions['sequence']['history_capped_rows_pct']:.2f}%"],
                ["single_event_users", interactions["sequence"]["single_event_users"]],
                ["repeated_item_users", interactions["sequence"]["repeated_item_users"]],
                ["repeated_item_extra_events", interactions["sequence"]["repeated_item_extra_events"]],
                ["adjacent_same_timestamp_pairs", interactions["sequence"]["adjacent_same_timestamp_pairs"]],
            ],
        ),
        "",
        md_table(
            [
                "split",
                "rows",
                "time_range",
                "target_window_rows",
                "target_window_pct",
                "target_users",
                "target_books",
                "new_users_vs_train",
                "new_books_vs_train",
                "history_p50",
                "history_p95",
            ],
            [split_compact_row(name, values) for name, values in split_rows.items()],
        ),
        "",
        "## Reviews And Text Fallback",
        "",
        md_table(
            ["metric", "value"],
            [
                ["review_rows", reviews["total_rows"]],
                ["non_empty_reviews", reviews["non_empty_reviews"]],
                ["non_empty_reviews_pct", f"{reviews['non_empty_reviews_pct']:.2f}%"],
                ["unique_review_users", reviews["unique_review_users"]],
                ["unique_review_books", reviews["unique_review_books"]],
                ["review_books_in_book_metadata", reviews["review_books_in_book_metadata"]],
                ["review_books_in_valid_items", reviews["review_books_in_valid_items"]],
                ["review_users_in_valid_interactions", reviews["review_users_in_valid_interactions"]],
                ["fallback_candidate_reviews", reviews["fallback_candidate_reviews"]],
                ["fallback_candidate_books", reviews["fallback_candidate_books"]],
                ["valid_short_description_books", reviews["valid_short_description_books"]],
                ["valid_empty_description_books", reviews["valid_empty_description_books"]],
                ["simulated_description_source", reviews["simulated_description_source"]],
                ["simulated_review_source", reviews["simulated_review_source"]],
                ["simulated_none_source", reviews["simulated_none_source"]],
            ],
        ),
        "",
        md_table(
            ["distribution", "count", "min", "p50", "p90", "p95", "p99", "max", "mean"],
            [
                summary_compact_row("review_word_count", reviews["word_count"]),
                summary_compact_row("review_votes", reviews["n_votes"]),
                summary_compact_row("review_comments", reviews["n_comments"]),
                summary_compact_row("fallback_reviews_per_book", reviews["fallback_candidate_reviews_per_book"]),
            ],
        ),
        "",
        "Review rating distribution:",
        "",
        md_table(["rating", "reviews", "pct_reviews"], counter_rows(Counter(reviews["rating_distribution"]), 10, reviews["total_rows"])),
        "",
        "## Interview Notes",
        "",
        "- Filtering matters: only read-and-rated interactions become supervision; most raw rows are not direct positive signals.",
        "- Raw k-core=30 does not survive the read-and-rated filter, so the training matrix is much sparser than the original subset.",
        "- The current split is globally chronological by target event, not leave-one-out per user. This is good for temporal realism, but evaluation history can include interactions from earlier validation/test timestamps.",
        "- Some raw timestamps are outside the project window, so always state whether an analysis is full-history or 2016-09 to 2017-12 only.",
        "- The item encoder is transductive: it embeds every valid item before train/valid/test split, including items that are only targets outside the training window.",
        "- Description sparsity is material, so review fallback is not a cosmetic step; it decides whether RQVAE sees meaningful text for cold or sparse items.",
        "- Shelf names are noisy because logistics shelves dominate, so any shelf feature should filter shelves such as to-read/currently-reading/kindle before treating them as semantics.",
        "- Repeated user-book pairs exist; if the target task is next unique item rather than next event, deduplication policy should be stated explicitly.",
        "",
    ]
    return "\n".join(lines)


def summary_compact_row(name: str, summary: dict[str, Any]) -> list[Any]:
    return [
        name,
        summary.get("count"),
        summary.get("min"),
        summary.get("p50"),
        summary.get("p90"),
        summary.get("p95"),
        summary.get("p99"),
        summary.get("max"),
        summary.get("mean"),
    ]


def split_compact_row(name: str, values: dict[str, Any]) -> list[Any]:
    history = values["history_length"]
    return [
        name,
        values["rows"],
        values["time_range"],
        values["target_window_rows"],
        f"{values['target_window_rows_pct']:.2f}%",
        values["unique_target_users"],
        values["unique_target_books"],
        values["new_target_users_vs_train"],
        values["new_target_books_vs_train"],
        history.get("p50"),
        history.get("p95"),
    ]


def parse_list_column(text: str) -> list[Any]:
    value = ast.literal_eval(text)
    if not isinstance(value, list):
        raise TypeError(f"Expected list column, got: {type(value)}")
    return value


def load_json_artifact(path: Path, required: bool) -> dict[str, Any]:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Missing required artifact: {path}")
        return {}
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected object artifact at {path}, got: {type(payload)}")
    return payload


def load_embedding_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    embedding = np.load(path, mmap_mode="r")
    return {
        "path": str(path),
        "exists": True,
        "size_mb": path.stat().st_size / 1024 / 1024,
        "shape": [int(dim) for dim in embedding.shape],
        "dtype": str(embedding.dtype),
    }


def analyze_rqvae_artifacts(rqvae_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    stats_path = rqvae_dir / "stats.json"
    item_path = rqvae_dir / "goodreads.item.json"
    item2id_path = rqvae_dir / "goodreads.item2id.json"
    index_path = rqvae_dir / "goodreads.index.json"
    sid_stats_path = rqvae_dir / "goodreads.sid_stats.json"
    embedding_path = rqvae_dir / "goodreads.emb-bge.npy"

    preprocess_stats = load_json_artifact(stats_path, required=False)
    item_payload = load_json_artifact(item_path, required=True)
    item2id = load_json_artifact(item2id_path, required=True)
    sid_payload = load_json_artifact(index_path, required=False)
    sid_stats = load_json_artifact(sid_stats_path, required=False)
    embedding = load_embedding_metadata(embedding_path)

    item_payload_ids = set(item_payload)
    item2id_item_ids = {str(item_id) for item_id in item2id.values()}
    item2id_book_ids = set(item2id)
    payload_book_ids = {str(item.get("book_id")) for item in item_payload.values() if item.get("book_id") is not None}
    sid_ids = set(sid_payload)
    text_source_counter = Counter(str(item.get("text_source") or "<missing>") for item in item_payload.values())
    item_text_words = [word_count(str(item.get("item_text", ""))) for item in item_payload.values()]
    title_words = [word_count(str(item.get("title", ""))) for item in item_payload.values()]
    description_words = [word_count(str(item.get("description", ""))) for item in item_payload.values()]
    average_ratings = [rating for item in item_payload.values() if (rating := parse_float(item.get("average_rating"))) is not None]
    ratings_counts = [count for item in item_payload.values() if (count := parse_int(item.get("ratings_count"))) is not None]
    sid_strings = Counter(" ".join(str(token) for token in tokens) for tokens in sid_payload.values())
    sid_token_lengths = [len(tokens) for tokens in sid_payload.values()]

    sid_health = summarize_sid_health(sid_stats)
    metrics = {
        "directory": str(rqvae_dir),
        "paths": {
            "stats": str(stats_path),
            "item": str(item_path),
            "item2id": str(item2id_path),
            "sid_index": str(index_path),
            "sid_stats": str(sid_stats_path),
            "embedding": str(embedding_path),
        },
        "preprocess_stats": preprocess_stats,
        "item_payload_items": len(item_payload_ids),
        "item2id_books": len(item2id_book_ids),
        "item2id_unique_item_ids": len(item2id_item_ids),
        "item2id_duplicate_item_ids": len(item2id) - len(item2id_item_ids),
        "item2id_items_missing_payload": len(item2id_item_ids - item_payload_ids),
        "payload_items_missing_item2id": len(item_payload_ids - item2id_item_ids),
        "payload_book_ids_missing_item2id": len(payload_book_ids - item2id_book_ids),
        "item2id_book_ids_missing_payload": len(item2id_book_ids - payload_book_ids),
        "text_sources": dict(text_source_counter),
        "item_text_words": numeric_summary(item_text_words),
        "title_words": numeric_summary(title_words),
        "description_words": numeric_summary(description_words),
        "average_rating": numeric_summary(average_ratings),
        "ratings_count": numeric_summary(ratings_counts),
        "embedding": embedding,
        "sid_items": len(sid_ids),
        "sid_items_missing_payload": len(item_payload_ids - sid_ids),
        "sid_extra_items_not_in_payload": len(sid_ids - item_payload_ids),
        "sid_token_length": numeric_summary(sid_token_lengths),
        "unique_sid_paths": len(sid_strings),
        "duplicate_sid_path_groups": sum(1 for count in sid_strings.values() if count > 1),
        "duplicate_sid_path_extra_items": sum(count - 1 for count in sid_strings.values() if count > 1),
        "sid_health": sid_health,
    }
    context = {
        "item_payload": item_payload,
        "item2id": item2id,
        "sid_payload": sid_payload,
        "item_payload_ids": item_payload_ids,
        "item2id_item_ids": item2id_item_ids,
        "sid_ids": sid_ids,
    }
    return metrics, context


def summarize_sid_health(sid_stats: dict[str, Any]) -> dict[str, Any]:
    if not sid_stats:
        return {}
    return {
        "status": sid_stats.get("status"),
        "num_items": sid_stats.get("num_items"),
        "num_code_levels": sid_stats.get("num_code_levels"),
        "assignment_mode": sid_stats.get("assignment_mode"),
        "num_unique_code_paths": sid_stats.get("paths", {}).get("num_unique_code_paths"),
        "num_collisions": sid_stats.get("paths", {}).get("num_collisions"),
        "collision_rate": sid_stats.get("paths", {}).get("collision_rate"),
        "assignment_comparison": sid_stats.get("assignment_comparison", {}),
        "levels": [
            {
                "level": level.get("level"),
                "codebook_size": level.get("codebook_size"),
                "used_codes": level.get("used_codes"),
                "used_ratio": level.get("used_ratio"),
                "dead_codes": level.get("dead_codes"),
                "perplexity": level.get("perplexity"),
                "max_bucket_share": level.get("max_bucket_share"),
            }
            for level in sid_stats.get("levels", [])
        ],
    }


def new_rqvae_split_state() -> dict[str, Any]:
    return {
        "target_user_counter": Counter(),
        "target_book_counter": Counter(),
        "target_item_counter": Counter(),
        "history_book_counter": Counter(),
        "history_item_counter": Counter(),
        "rating_counter": Counter(),
        "history_rating_counter": Counter(),
        "target_month_counter": Counter(),
        "target_text_source_counter": Counter(),
        "history_provenance_counter": Counter(),
        "user_item_pair_counter": Counter(),
        "user_book_pair_counter": Counter(),
        "target_timestamps": [],
        "history_lengths": [],
        "target_lag_days": [],
        "history_span_days": [],
        "title_words": [],
        "integrity_counter": Counter(),
        "previous_timestamp": None,
        "last_user_timestamp": {},
        "rows": 0,
    }


def parse_rqvae_split_row(row: dict[str, str]) -> dict[str, Any]:
    history_book_ids = [str(value) for value in parse_list_column(row["history_book_ids"])]
    history_item_ids = [str(value) for value in parse_list_column(row["history_item_ids"])]
    return {
        "user_id": str(row["user_id"]),
        "book_id": str(row["book_id"]),
        "item_id": str(row["item_id"]),
        "rating": int(row["rating"]),
        "timestamp": int(row["timestamp"]),
        "history_book_ids": history_book_ids,
        "history_item_ids": history_item_ids,
        "history_ratings": [int(value) for value in parse_list_column(row["history_ratings"])],
        "history_timestamps": [int(value) for value in parse_list_column(row["history_timestamps"])],
        "item_title": row["item_title"],
        "history_len": len(history_item_ids),
    }


def update_rqvae_split_counts(state: dict[str, Any], parsed: dict[str, Any]) -> None:
    state["rows"] += 1
    state["target_user_counter"][parsed["user_id"]] += 1
    state["target_book_counter"][parsed["book_id"]] += 1
    state["target_item_counter"][parsed["item_id"]] += 1
    state["history_book_counter"].update(parsed["history_book_ids"])
    state["history_item_counter"].update(parsed["history_item_ids"])
    state["rating_counter"][parsed["rating"]] += 1
    state["history_rating_counter"].update(parsed["history_ratings"])
    state["target_month_counter"][month_from_timestamp(parsed["timestamp"])] += 1
    state["user_item_pair_counter"][(parsed["user_id"], parsed["item_id"])] += 1
    state["user_book_pair_counter"][(parsed["user_id"], parsed["book_id"])] += 1
    state["target_timestamps"].append(parsed["timestamp"])
    state["history_lengths"].append(parsed["history_len"])
    state["title_words"].append(word_count(parsed["item_title"]))


def update_rqvae_time_checks(
    state: dict[str, Any],
    parsed: dict[str, Any],
    history_max_len: int,
    target_start_timestamp: int,
    target_end_timestamp: int,
) -> None:
    integrity = state["integrity_counter"]
    timestamp = parsed["timestamp"]
    history_timestamps = parsed["history_timestamps"]
    if in_window(timestamp, target_start_timestamp, target_end_timestamp):
        integrity["target_window_rows"] += 1
    integrity["negative_timestamp_rows"] += int(timestamp < 0)
    integrity["empty_history_rows"] += int(parsed["history_len"] == 0)
    integrity["history_len_gt_max_rows"] += int(parsed["history_len"] > history_max_len)
    integrity["history_len_eq_max_rows"] += int(parsed["history_len"] == history_max_len)
    lengths = {len(parsed["history_book_ids"]), parsed["history_len"], len(parsed["history_ratings"]), len(history_timestamps)}
    integrity["history_length_mismatch_rows"] += int(len(lengths) != 1)
    integrity["nonmonotonic_history_rows"] += int(
        any(right < left for left, right in zip(history_timestamps, history_timestamps[1:]))
    )
    update_rqvae_lag_checks(state, parsed)
    update_rqvae_order_checks(state, parsed)


def update_rqvae_lag_checks(state: dict[str, Any], parsed: dict[str, Any]) -> None:
    history_timestamps = parsed["history_timestamps"]
    if not history_timestamps:
        return
    timestamp = parsed["timestamp"]
    integrity = state["integrity_counter"]
    integrity["target_before_history_rows"] += int(timestamp < max(history_timestamps))
    integrity["target_equal_last_history_rows"] += int(timestamp == history_timestamps[-1])
    state["target_lag_days"].append((timestamp - history_timestamps[-1]) / 86400)
    state["history_span_days"].append((history_timestamps[-1] - history_timestamps[0]) / 86400)


def update_rqvae_order_checks(state: dict[str, Any], parsed: dict[str, Any]) -> None:
    timestamp = parsed["timestamp"]
    user_id = parsed["user_id"]
    integrity = state["integrity_counter"]
    previous_timestamp = state["previous_timestamp"]
    last_user_timestamp = state["last_user_timestamp"]
    if previous_timestamp is not None and timestamp < previous_timestamp:
        integrity["row_order_violations"] += 1
    state["previous_timestamp"] = timestamp
    if user_id in last_user_timestamp and timestamp < last_user_timestamp[user_id]:
        integrity["user_order_violations"] += 1
    last_user_timestamp[user_id] = timestamp


def update_rqvae_artifact_checks(
    state: dict[str, Any],
    parsed: dict[str, Any],
    split_name: str,
    artifact_context: dict[str, Any],
    train_max_timestamp: int | None,
    valid_max_timestamp: int | None,
) -> None:
    integrity = state["integrity_counter"]
    item_id = parsed["item_id"]
    history_item_ids = parsed["history_item_ids"]
    integrity["target_item_in_history_rows"] += int(item_id in history_item_ids)
    integrity["immediate_item_repeat_rows"] += int(bool(history_item_ids) and item_id == history_item_ids[-1])
    update_rqvae_target_artifact_checks(state, parsed, artifact_context)
    update_rqvae_history_artifact_checks(
        state,
        parsed,
        split_name,
        artifact_context,
        train_max_timestamp,
        valid_max_timestamp,
    )


def update_rqvae_target_artifact_checks(
    state: dict[str, Any], parsed: dict[str, Any], artifact_context: dict[str, Any]
) -> None:
    integrity = state["integrity_counter"]
    item_payload = artifact_context["item_payload"]
    item2id = artifact_context["item2id"]
    sid_ids = artifact_context["sid_ids"]
    book_id = parsed["book_id"]
    item_id = parsed["item_id"]
    if item_id not in item_payload:
        integrity["target_item_missing_payload_rows"] += 1
    else:
        payload = item_payload[item_id]
        state["target_text_source_counter"][str(payload.get("text_source") or "<missing>")] += 1
        integrity["target_book_item_mismatch_rows"] += int(str(payload.get("book_id")) != book_id)
    if item2id.get(book_id) is None:
        integrity["target_book_missing_item2id_rows"] += 1
    elif str(item2id[book_id]) != item_id:
        integrity["target_book_item2id_mismatch_rows"] += 1
    if sid_ids and item_id not in sid_ids:
        integrity["target_sid_missing_rows"] += 1


def update_rqvae_history_artifact_checks(
    state: dict[str, Any],
    parsed: dict[str, Any],
    split_name: str,
    artifact_context: dict[str, Any],
    train_max_timestamp: int | None,
    valid_max_timestamp: int | None,
) -> None:
    item_payload = artifact_context["item_payload"]
    item2id = artifact_context["item2id"]
    sid_ids = artifact_context["sid_ids"]
    integrity = state["integrity_counter"]
    for book_id, item_id, timestamp in zip(
        parsed["history_book_ids"], parsed["history_item_ids"], parsed["history_timestamps"]
    ):
        bucket = history_split_bucket(split_name, timestamp, train_max_timestamp, valid_max_timestamp)
        state["history_provenance_counter"][bucket] += 1
        integrity["history_item_missing_payload_events"] += int(item_id not in item_payload)
        if item2id.get(book_id) is None:
            integrity["history_book_missing_item2id_events"] += 1
        elif str(item2id[book_id]) != item_id:
            integrity["history_book_item2id_mismatch_events"] += 1
        if sid_ids and item_id not in sid_ids:
            integrity["history_sid_missing_events"] += 1


def finalize_rqvae_split(
    split_path: Path,
    state: dict[str, Any],
    item_payload: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    target_items = set(state["target_item_counter"])
    unique_target_text_sources = Counter(
        str(item_payload[item_id].get("text_source") or "<missing>") if item_id in item_payload else "<missing>"
        for item_id in target_items
    )
    total_history_events = sum(state["history_lengths"])
    metrics = {
        "path": str(split_path),
        "size_mb": split_path.stat().st_size / 1024 / 1024,
        "rows": state["rows"],
        "time_range": time_range(state["target_timestamps"]),
        "min_timestamp": min(state["target_timestamps"]) if state["target_timestamps"] else None,
        "max_timestamp": max(state["target_timestamps"]) if state["target_timestamps"] else None,
        "unique_target_users": len(state["target_user_counter"]),
        "unique_target_books": len(state["target_book_counter"]),
        "unique_target_items": len(state["target_item_counter"]),
        "unique_history_books": len(state["history_book_counter"]),
        "unique_history_items": len(state["history_item_counter"]),
        "history_events": total_history_events,
        "target_window_rows": state["integrity_counter"]["target_window_rows"],
        "target_window_rows_pct": pct(state["integrity_counter"]["target_window_rows"], state["rows"]),
        "negative_timestamp_rows": state["integrity_counter"]["negative_timestamp_rows"],
        "history_length": numeric_summary(state["history_lengths"]),
        "history_len_eq_max_rows": state["integrity_counter"]["history_len_eq_max_rows"],
        "history_len_eq_max_rows_pct": pct(state["integrity_counter"]["history_len_eq_max_rows"], state["rows"]),
        "target_lag_days": numeric_summary(state["target_lag_days"]),
        "history_span_days": numeric_summary(state["history_span_days"]),
        "target_title_words": numeric_summary(state["title_words"]),
        "rating_distribution": dict(sorted(state["rating_counter"].items())),
        "history_rating_distribution": dict(sorted(state["history_rating_counter"].items())),
        "target_months": dict(sorted(state["target_month_counter"].items())),
        "target_text_sources_by_rows": dict(state["target_text_source_counter"]),
        "target_text_sources_by_unique_items": dict(unique_target_text_sources),
        "history_provenance": dict(state["history_provenance_counter"]),
        "integrity": dict(state["integrity_counter"]),
        "duplicate_user_item_pairs": sum(1 for count in state["user_item_pair_counter"].values() if count > 1),
        "duplicate_user_item_extra_rows": sum(count - 1 for count in state["user_item_pair_counter"].values() if count > 1),
        "duplicate_user_book_pairs": sum(1 for count in state["user_book_pair_counter"].values() if count > 1),
        "duplicate_user_book_extra_rows": sum(count - 1 for count in state["user_book_pair_counter"].values() if count > 1),
        "target_item_in_history_rows": state["integrity_counter"]["target_item_in_history_rows"],
        "target_item_in_history_rows_pct": pct(state["integrity_counter"]["target_item_in_history_rows"], state["rows"]),
        "immediate_item_repeat_rows": state["integrity_counter"]["immediate_item_repeat_rows"],
        "immediate_item_repeat_rows_pct": pct(state["integrity_counter"]["immediate_item_repeat_rows"], state["rows"]),
    }
    context = {
        "target_users": set(state["target_user_counter"]),
        "target_books": set(state["target_book_counter"]),
        "target_items": set(state["target_item_counter"]),
        "history_books": set(state["history_book_counter"]),
        "history_items": set(state["history_item_counter"]),
        "target_item_counter": state["target_item_counter"],
        "target_user_counter": state["target_user_counter"],
        "target_book_counter": state["target_book_counter"],
    }
    return metrics, context


def analyze_rqvae_split(
    split_path: Path,
    split_name: str,
    artifact_context: dict[str, Any],
    history_max_len: int,
    target_start_timestamp: int,
    target_end_timestamp: int,
    train_max_timestamp: int | None,
    valid_max_timestamp: int | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    state = new_rqvae_split_state()
    with split_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        actual_columns = reader.fieldnames or []
        if actual_columns != RQVAE_SPLIT_COLUMNS:
            raise ValueError(f"Unexpected split columns in {split_path}: expected {RQVAE_SPLIT_COLUMNS}, got {actual_columns}")
        for row in reader:
            parsed = parse_rqvae_split_row(row)
            update_rqvae_split_counts(state, parsed)
            update_rqvae_time_checks(state, parsed, history_max_len, target_start_timestamp, target_end_timestamp)
            update_rqvae_artifact_checks(
                state,
                parsed,
                split_name,
                artifact_context,
                train_max_timestamp,
                valid_max_timestamp,
            )
    return finalize_rqvae_split(split_path, state, artifact_context["item_payload"])


def history_split_bucket(
    split_name: str,
    timestamp: int,
    train_max_timestamp: int | None,
    valid_max_timestamp: int | None,
) -> str:
    if split_name == "train" or train_max_timestamp is None or timestamp <= train_max_timestamp:
        return "train"
    if split_name == "valid" or valid_max_timestamp is None or timestamp <= valid_max_timestamp:
        return "valid"
    return "test"


def analyze_rqvae_outputs(rqvae_dir: Path, cfg: DictConfig) -> dict[str, Any]:
    artifact_metrics, artifact_context = analyze_rqvae_artifacts(rqvae_dir)
    target_start_timestamp = parse_date_boundary(cfg.target_start, end_of_day=False)
    target_end_timestamp = parse_date_boundary(cfg.target_end, end_of_day=True)
    history_max_len = int(cfg.history_max_len)
    split_metrics: dict[str, dict[str, Any]] = {}
    split_contexts: dict[str, dict[str, Any]] = {}

    train_metrics, train_context = analyze_rqvae_split(
        rqvae_dir / "train.csv",
        "train",
        artifact_context,
        history_max_len,
        target_start_timestamp,
        target_end_timestamp,
        train_max_timestamp=None,
        valid_max_timestamp=None,
    )
    train_max_timestamp = train_metrics["max_timestamp"]
    valid_metrics, valid_context = analyze_rqvae_split(
        rqvae_dir / "valid.csv",
        "valid",
        artifact_context,
        history_max_len,
        target_start_timestamp,
        target_end_timestamp,
        train_max_timestamp=train_max_timestamp,
        valid_max_timestamp=None,
    )
    valid_max_timestamp = valid_metrics["max_timestamp"]
    test_metrics, test_context = analyze_rqvae_split(
        rqvae_dir / "test.csv",
        "test",
        artifact_context,
        history_max_len,
        target_start_timestamp,
        target_end_timestamp,
        train_max_timestamp=train_max_timestamp,
        valid_max_timestamp=valid_max_timestamp,
    )

    split_metrics.update({"train": train_metrics, "valid": valid_metrics, "test": test_metrics})
    split_contexts.update({"train": train_context, "valid": valid_context, "test": test_context})
    cross_split = build_rqvae_cross_split_metrics(split_metrics, split_contexts, artifact_metrics, artifact_context)
    return {
        "artifacts": artifact_metrics,
        "splits": split_metrics,
        "cross_split": cross_split,
    }


def build_rqvae_cross_split_metrics(
    split_metrics: dict[str, dict[str, Any]],
    split_contexts: dict[str, dict[str, Any]],
    artifact_metrics: dict[str, Any],
    artifact_context: dict[str, Any],
) -> dict[str, Any]:
    train_context = split_contexts["train"]
    item_payload = artifact_context["item_payload"]
    all_target_users: set[str] = set()
    all_target_books: set[str] = set()
    all_target_items: set[str] = set()
    all_history_items: set[str] = set()
    all_target_item_counter = Counter()
    all_target_user_counter = Counter()

    for name, context in split_contexts.items():
        new_users = context["target_users"] - train_context["target_users"] if name != "train" else set()
        new_items = context["target_items"] - train_context["target_items"] if name != "train" else set()
        new_books = context["target_books"] - train_context["target_books"] if name != "train" else set()
        split_metrics[name]["new_target_users_vs_train"] = len(new_users)
        split_metrics[name]["new_target_items_vs_train"] = len(new_items)
        split_metrics[name]["new_target_books_vs_train"] = len(new_books)
        all_target_users.update(context["target_users"])
        all_target_books.update(context["target_books"])
        all_target_items.update(context["target_items"])
        all_history_items.update(context["history_items"])
        all_target_item_counter.update(context["target_item_counter"])
        all_target_user_counter.update(context["target_user_counter"])

    train_max = split_metrics["train"]["max_timestamp"]
    valid_min = split_metrics["valid"]["min_timestamp"]
    valid_max = split_metrics["valid"]["max_timestamp"]
    test_min = split_metrics["test"]["min_timestamp"]
    preprocess_rows = artifact_metrics["preprocess_stats"].get("num_rows")
    total_rows = sum(split["rows"] for split in split_metrics.values())
    return {
        "total_rows": total_rows,
        "preprocess_num_rows": preprocess_rows,
        "row_count_matches_preprocess_stats": preprocess_rows == total_rows if preprocess_rows is not None else None,
        "target_users_all": len(all_target_users),
        "target_books_all": len(all_target_books),
        "target_items_all": len(all_target_items),
        "history_items_all": len(all_history_items),
        "target_items_seen_in_history": len(all_target_items & all_history_items),
        "target_items_never_seen_in_history": len(all_target_items - all_history_items),
        "train_max_le_valid_min": train_max <= valid_min if train_max is not None and valid_min is not None else None,
        "valid_max_le_test_min": valid_max <= test_min if valid_max is not None and test_min is not None else None,
        "train_valid_gap_days": (valid_min - train_max) / 86400 if train_max is not None and valid_min is not None else None,
        "valid_test_gap_days": (test_min - valid_max) / 86400 if valid_max is not None and test_min is not None else None,
        "item_payload_items_used_as_targets": len(all_target_items & artifact_context["item_payload_ids"]),
        "target_items_missing_payload": len(all_target_items - artifact_context["item_payload_ids"]),
        "target_items_missing_sid": len(all_target_items - artifact_context["sid_ids"]) if artifact_context["sid_ids"] else None,
        "history_items_missing_payload": len(all_history_items - artifact_context["item_payload_ids"]),
        "history_items_missing_sid": len(all_history_items - artifact_context["sid_ids"]) if artifact_context["sid_ids"] else None,
        "top_target_items": top_target_item_rows(all_target_item_counter, item_payload, total_rows, top_n=20),
        "top_target_users": counter_rows(all_target_user_counter, 20, total_rows),
    }


def top_target_item_rows(
    counter: Counter,
    item_payload: dict[str, dict[str, Any]],
    total: int,
    top_n: int,
) -> list[list[Any]]:
    rows = []
    for item_id, count in counter.most_common(top_n):
        payload = item_payload.get(item_id, {})
        rows.append([
            item_id,
            payload.get("book_id", "<missing>"),
            payload.get("title", "<missing>"),
            count,
            fmt_pct(count, total),
        ])
    return rows


def rqvae_split_overview_row(name: str, values: dict[str, Any], total_rows: int) -> list[Any]:
    history = values["history_length"]
    return [
        name,
        values["rows"],
        fmt_pct(values["rows"], total_rows),
        values["time_range"],
        values["target_window_rows"],
        f"{values['target_window_rows_pct']:.2f}%",
        values["unique_target_users"],
        values["unique_target_items"],
        values["new_target_users_vs_train"],
        values["new_target_items_vs_train"],
        history.get("p50"),
        history.get("p95"),
        f"{values['history_len_eq_max_rows_pct']:.2f}%",
    ]


def rqvae_integrity_row(name: str, values: dict[str, Any]) -> list[Any]:
    integrity = Counter(values["integrity"])
    return [
        name,
        integrity["history_length_mismatch_rows"],
        integrity["history_len_gt_max_rows"],
        integrity["nonmonotonic_history_rows"],
        integrity["target_before_history_rows"],
        integrity["row_order_violations"],
        integrity["target_item_missing_payload_rows"],
        integrity["target_book_item2id_mismatch_rows"],
        integrity["target_sid_missing_rows"],
        integrity["history_sid_missing_events"],
    ]


def rqvae_history_provenance_row(name: str, values: dict[str, Any]) -> list[Any]:
    provenance = Counter(values["history_provenance"])
    total = values["history_events"]
    return [
        name,
        total,
        provenance["train"],
        fmt_pct(provenance["train"], total),
        provenance["valid"],
        fmt_pct(provenance["valid"], total),
        provenance["test"],
        fmt_pct(provenance["test"], total),
    ]


def rqvae_rating_row(name: str, values: dict[str, Any]) -> list[Any]:
    ratings = Counter(values["rating_distribution"])
    return [name, ratings[1], ratings[2], ratings[3], ratings[4], ratings[5]]


def rqvae_repeat_row(name: str, values: dict[str, Any]) -> list[Any]:
    return [
        name,
        values["target_item_in_history_rows"],
        f"{values['target_item_in_history_rows_pct']:.2f}%",
        values["immediate_item_repeat_rows"],
        f"{values['immediate_item_repeat_rows_pct']:.2f}%",
        values["duplicate_user_item_pairs"],
        values["duplicate_user_item_extra_rows"],
    ]


def rqvae_text_source_rows(rqvae: dict[str, Any]) -> list[list[Any]]:
    artifacts = rqvae["artifacts"]
    rows = [["item_payload_unique_items", source, count, fmt_pct(count, artifacts["item_payload_items"])] for source, count in artifacts["text_sources"].items()]
    for split_name, split in rqvae["splits"].items():
        for source, count in split["target_text_sources_by_rows"].items():
            rows.append([f"{split_name}_target_rows", source, count, fmt_pct(count, split["rows"])])
        for source, count in split["target_text_sources_by_unique_items"].items():
            rows.append([
                f"{split_name}_target_unique_items",
                source,
                count,
                fmt_pct(count, split["unique_target_items"]),
            ])
    return rows


def sid_health_level_rows(sid_health: dict[str, Any]) -> list[list[Any]]:
    return [
        [
            level["level"],
            level["codebook_size"],
            level["used_codes"],
            f"{level['used_ratio'] * 100:.2f}%" if level.get("used_ratio") is not None else None,
            level["dead_codes"],
            level["perplexity"],
            f"{level['max_bucket_share'] * 100:.2f}%" if level.get("max_bucket_share") is not None else None,
        ]
        for level in sid_health.get("levels", [])
    ]


def build_rqvae_report(cfg: DictConfig, rqvae: dict[str, Any]) -> str:
    artifacts = rqvae["artifacts"]
    splits = rqvae["splits"]
    cross = rqvae["cross_split"]
    total_rows = cross["total_rows"]
    embedding = artifacts["embedding"]
    sid_health = artifacts["sid_health"]
    embedding_shape = embedding.get("shape") if embedding.get("exists") else None
    lines = [
        "# GoodReads RQVAE Split EDA",
        "",
        f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Configuration",
        "",
        md_table(
            ["key", "value"],
            [
                ["rqvae_dir", cfg.rqvae_dir],
                ["history_max_len", cfg.history_max_len],
                ["target_start", cfg.target_start],
                ["target_end", cfg.target_end],
                ["rqvae_output_markdown", cfg.rqvae_output_markdown],
                ["rqvae_output_json", cfg.rqvae_output_json],
            ],
        ),
        "",
        "## Executive Summary",
        "",
        "- The processed sequence files contain "
        f"{fmt(total_rows)} next-item rows: {fmt(splits['train']['rows'])} train, "
        f"{fmt(splits['valid']['rows'])} valid, and {fmt(splits['test']['rows'])} test.",
        "- Split row counts "
        f"{'match' if cross['row_count_matches_preprocess_stats'] else 'do not match'} stats.json num_rows "
        f"({fmt(cross['preprocess_num_rows'])}); this is the first reproducibility check for preprocessing.",
        "- The split is globally chronological by target timestamp: "
        f"train<=valid is {cross['train_max_le_valid_min']}, valid<=test is {cross['valid_max_le_test_min']}. "
        f"Valid/test histories can still include earlier events from the same evaluation split.",
        "- Target item coverage uses "
        f"{fmt(cross['target_items_all'])} unique items from an item payload of {fmt(artifacts['item_payload_items'])} items; "
        f"valid/test introduce {fmt(splits['valid']['new_target_items_vs_train'])}/"
        f"{fmt(splits['test']['new_target_items_vs_train'])} target items not seen as train targets.",
        "- SID/embedding artifact check reports "
        f"item payload={fmt(artifacts['item_payload_items'])}, SID index={fmt(artifacts['sid_items'])}, "
        f"embedding_shape={embedding_shape or '-'}.",
        "",
        "## Processed Artifacts",
        "",
        md_table(
            ["artifact", "value"],
            [
                ["preprocess_num_rows", artifacts["preprocess_stats"].get("num_rows")],
                ["observed_split_rows", total_rows],
                ["row_count_matches_preprocess_stats", cross["row_count_matches_preprocess_stats"]],
                ["preprocess_num_users", artifacts["preprocess_stats"].get("num_users")],
                ["preprocess_num_items", artifacts["preprocess_stats"].get("num_items")],
                ["item_payload_items", artifacts["item_payload_items"]],
                ["item2id_books", artifacts["item2id_books"]],
                ["item2id_unique_item_ids", artifacts["item2id_unique_item_ids"]],
                ["item2id_items_missing_payload", artifacts["item2id_items_missing_payload"]],
                ["payload_items_missing_item2id", artifacts["payload_items_missing_item2id"]],
                ["sid_items", artifacts["sid_items"]],
                ["sid_items_missing_payload", artifacts["sid_items_missing_payload"]],
                ["sid_extra_items_not_in_payload", artifacts["sid_extra_items_not_in_payload"]],
                ["embedding_exists", embedding.get("exists")],
                ["embedding_shape", embedding_shape],
                ["embedding_dtype", embedding.get("dtype")],
            ],
        ),
        "",
        "### File Sizes",
        "",
        md_table(
            ["file", "rows_or_items", "size_mb", "time_range"],
            [
                [Path(splits[name]["path"]).name, splits[name]["rows"], splits[name]["size_mb"], splits[name]["time_range"]]
                for name in RQVAE_SPLIT_NAMES
            ]
            + [
                ["goodreads.item.json", artifacts["item_payload_items"], Path(artifacts["paths"]["item"]).stat().st_size / 1024 / 1024, "-"],
                ["goodreads.item2id.json", artifacts["item2id_books"], Path(artifacts["paths"]["item2id"]).stat().st_size / 1024 / 1024, "-"],
                ["goodreads.index.json", artifacts["sid_items"], Path(artifacts["paths"]["sid_index"]).stat().st_size / 1024 / 1024 if Path(artifacts["paths"]["sid_index"]).exists() else None, "-"],
                ["goodreads.emb-bge.npy", embedding_shape, embedding.get("size_mb"), "-"],
            ],
        ),
        "",
        "### Item Text And Metadata",
        "",
        md_table(
            ["distribution", "count", "min", "p50", "p90", "p95", "p99", "max", "mean"],
            [
                summary_compact_row("item_text_words", artifacts["item_text_words"]),
                summary_compact_row("title_words", artifacts["title_words"]),
                summary_compact_row("description_words", artifacts["description_words"]),
                summary_compact_row("average_rating", artifacts["average_rating"]),
                summary_compact_row("ratings_count", artifacts["ratings_count"]),
                summary_compact_row("sid_token_length", artifacts["sid_token_length"]),
            ],
        ),
        "",
        "Text source coverage:",
        "",
        md_table(["scope", "source", "count", "pct_scope"], rqvae_text_source_rows(rqvae)),
        "",
        "## Split Overview",
        "",
        md_table(
            [
                "split",
                "rows",
                "pct_rows",
                "time_range",
                "target_window_rows",
                "target_window_pct",
                "target_users",
                "target_items",
                "new_users_vs_train",
                "new_items_vs_train",
                "history_p50",
                "history_p95",
                "history_eq_max_pct",
            ],
            [rqvae_split_overview_row(name, splits[name], total_rows) for name in RQVAE_SPLIT_NAMES],
        ),
        "",
        "### Rating Distribution",
        "",
        md_table(["split", "rating_1", "rating_2", "rating_3", "rating_4", "rating_5"], [rqvae_rating_row(name, splits[name]) for name in RQVAE_SPLIT_NAMES]),
        "",
        "### History Provenance",
        "",
        md_table(
            ["split", "history_events", "train_events", "train_pct", "valid_events", "valid_pct", "test_events", "test_pct"],
            [rqvae_history_provenance_row(name, splits[name]) for name in RQVAE_SPLIT_NAMES],
        ),
        "",
        "### Integrity Checks",
        "",
        md_table(
            [
                "split",
                "length_mismatch_rows",
                "history_len_gt_max",
                "nonmonotonic_history",
                "target_before_history",
                "row_order_violations",
                "target_missing_payload",
                "target_item2id_mismatch",
                "target_missing_sid",
                "history_missing_sid_events",
            ],
            [rqvae_integrity_row(name, splits[name]) for name in RQVAE_SPLIT_NAMES],
        ),
        "",
        "### Repeats And Duplicates",
        "",
        md_table(
            [
                "split",
                "target_in_history_rows",
                "target_in_history_pct",
                "immediate_repeat_rows",
                "immediate_repeat_pct",
                "duplicate_user_item_pairs",
                "duplicate_user_item_extra_rows",
            ],
            [rqvae_repeat_row(name, splits[name]) for name in RQVAE_SPLIT_NAMES],
        ),
        "",
        "## SID Health",
        "",
        md_table(
            ["metric", "value"],
            [
                ["status", sid_health.get("status")],
                ["assignment_mode", sid_health.get("assignment_mode")],
                ["sid_health_num_items", sid_health.get("num_items")],
                ["num_code_levels", sid_health.get("num_code_levels")],
                ["unique_raw_code_paths", sid_health.get("num_unique_code_paths")],
                ["num_collisions", sid_health.get("num_collisions")],
                ["collision_rate", sid_health.get("collision_rate")],
                ["dedup_unique_sid_paths", artifacts["unique_sid_paths"]],
                ["dedup_duplicate_sid_path_groups", artifacts["duplicate_sid_path_groups"]],
                ["dedup_duplicate_sid_path_extra_items", artifacts["duplicate_sid_path_extra_items"]],
            ],
        ),
        "",
        md_table(
            ["level", "codebook_size", "used_codes", "used_pct", "dead_codes", "perplexity", "max_bucket_share"],
            sid_health_level_rows(sid_health),
        ),
        "",
        "## Popularity Skew",
        "",
        "Top target items across all splits:",
        "",
        md_table(["item_id", "book_id", "title", "target_rows", "pct_rows"], cross["top_target_items"]),
        "",
        "Top target users across all splits:",
        "",
        md_table(["user_id", "target_rows", "pct_rows"], cross["top_target_users"]),
        "",
        "## Interview Notes",
        "",
        "- First verify artifact consistency before discussing model quality: split rows should match stats.json, item2id should match item payload, and SID/embedding counts should match the current item payload.",
        "- If the SID index has more rows than the current item payload, treat it as a stale-artifact signal and regenerate SIDs before trusting SFT data.",
        "- This dataset is a next-event task, not next-unique-item: rows where the target already appears in history quantify repeat-consumption pressure.",
        "- The split is global chronological by target event. Valid/test histories containing earlier valid/test events imply roll-forward evaluation rather than a strict train-only user history snapshot.",
        "- New valid/test target items versus train targets are not necessarily item-encoder cold-start items because item text, embeddings, and SIDs are built transductively over the full processed item set.",
        "- History length at the configured maximum should be reported because SFT context cost and truncation bias both depend on it.",
        "- Review-sourced item text matters for SID quality; if those items cluster in validation/test, text fallback quality becomes an evaluation confounder.",
        "",
    ]
    return "\n".join(lines)


def jsonable(value: Any) -> Any:
    if isinstance(value, Counter):
        return dict(value)
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [jsonable(item) for item in value]
    return value


@task_wrapper
def run_eda(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    data_dir = as_path(cfg.data_dir)
    output_markdown = as_path(cfg.output_markdown)
    output_json = as_path(cfg.output_json)
    rqvae_dir = as_path(cfg.rqvae_dir)
    rqvae_output_markdown = as_path(cfg.rqvae_output_markdown)
    rqvae_output_json = as_path(cfg.rqvae_output_json)
    books_path = data_dir / cfg.books_file
    interactions_path = data_dir / cfg.interactions_file
    reviews_path = data_dir / cfg.reviews_file

    books, book_context = analyze_books(books_path, cfg.min_description_words)
    interactions, interaction_context = analyze_interactions(
        interactions_path,
        book_context["book_ids"],
        cfg.history_max_len,
        cfg.target_start,
        cfg.target_end,
    )
    reviews = analyze_reviews(
        reviews_path,
        book_context["book_ids"],
        interaction_context["valid_book_ids"],
        interaction_context["valid_user_ids"],
        book_context["description_words_by_book"],
        cfg.min_review_rating,
        cfg.min_review_words,
        cfg.min_description_words,
    )
    rqvae = analyze_rqvae_outputs(rqvae_dir, cfg)

    metrics = {
        "config": OmegaConf.to_container(cfg, resolve=True),
        "books": books,
        "interactions": interactions,
        "reviews": reviews,
    }
    rqvae_metrics = {
        "config": OmegaConf.to_container(cfg, resolve=True),
        "rqvae": rqvae,
    }
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    rqvae_output_markdown.parent.mkdir(parents=True, exist_ok=True)
    rqvae_output_json.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.write_text(build_report(cfg, books, interactions, reviews), encoding="utf-8")
    output_json.write_text(json.dumps(jsonable(metrics), ensure_ascii=False, indent=2), encoding="utf-8")
    rqvae_output_markdown.write_text(build_rqvae_report(cfg, rqvae), encoding="utf-8")
    rqvae_output_json.write_text(json.dumps(jsonable(rqvae_metrics), ensure_ascii=False, indent=2), encoding="utf-8")

    log.info(f"Saved EDA markdown to {output_markdown}")
    log.info(f"Saved EDA metrics to {output_json}")
    log.info(f"Saved RQVAE EDA markdown to {rqvae_output_markdown}")
    log.info(f"Saved RQVAE EDA metrics to {rqvae_output_json}")
    return metrics, {
        "cfg": cfg,
        "books_path": books_path,
        "interactions_path": interactions_path,
        "reviews_path": reviews_path,
        "rqvae_dir": rqvae_dir,
        "output_markdown": output_markdown,
        "output_json": output_json,
        "rqvae_output_markdown": rqvae_output_markdown,
        "rqvae_output_json": rqvae_output_json,
    }


@hydra.main(version_base="1.3", config_path="../configs", config_name="eda")
def main(cfg: DictConfig) -> None:
    extras(cfg)
    run_eda(cfg)


if __name__ == "__main__":
    main()
