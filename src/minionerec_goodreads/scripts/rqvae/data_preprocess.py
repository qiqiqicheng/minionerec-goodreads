from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig

from minionerec_goodreads.utils import RankedLogger, extras, task_wrapper

log = RankedLogger(__name__, rank_zero_only=True)
TIME_FORMAT = "%a %b %d %H:%M:%S %z %Y"


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


def parse_time(record: dict[str, Any]) -> int:
    for key in ["read_at", "date_updated", "date_added", "started_at"]:
        value = record.get(key, "")
        if value:
            return int(datetime.strptime(value, TIME_FORMAT).timestamp())
    raise ValueError(f"No valid timestamp found for record: {record}")


def count_words(text: str) -> int:
    cleaned = normalize_text(text)
    return 0 if cleaned == "" else len(cleaned.split())


def select_review(left: dict[str, Any] | None, right: dict[str, Any]) -> dict[str, Any]:
    if left is None:
        return right
    left_key = (left["rating"], left["n_votes"], left["word_count"])
    right_key = (right["rating"], right["n_votes"], right["word_count"])
    return right if right_key > left_key else left


def build_review_fallbacks(review_path: Path, min_rating: int, min_review_words: int) -> dict[str, dict[str, Any]]:
    best_reviews: dict[str, dict[str, Any]] = {}
    for review in iter_jsonl(review_path):
        review_text = normalize_text(review.get("review_text", ""))
        word_count = count_words(review_text)
        rating = int(review.get("rating", 0))
        if rating < min_rating or word_count < min_review_words:
            continue
        candidate = {
            "text": review_text,
            "rating": rating,
            "n_votes": int(review.get("n_votes", 0)),
            "word_count": word_count,
        }
        book_id = review["book_id"]
        best_reviews[book_id] = select_review(best_reviews.get(book_id), candidate)
    return best_reviews


def format_shelves(popular_shelves: list[dict[str, Any]], topk: int) -> list[str]:
    cleaned = []
    for shelf in popular_shelves:
        name = normalize_text(str(shelf.get("name", "")).replace("-", " ").replace("_", " "))
        if name:
            cleaned.append((int(shelf.get("count", 0)), name))
    cleaned.sort(key=lambda item: (-item[0], item[1]))
    return [name for _, name in cleaned[:topk]]


def build_item_text(
    book: dict[str, Any],
    fallback_review: dict[str, Any] | None,
    min_description_words: int,
    shelf_topk: int,
) -> tuple[str, str]:
    title = normalize_text(book.get("title", ""))
    title_without_series = normalize_text(book.get("title_without_series", ""))
    description = normalize_text(book.get("description", ""))
    shelves = format_shelves(book.get("popular_shelves", []), topk=shelf_topk)
    author_ids = [str(author["author_id"]) for author in book.get("authors", []) if author.get("author_id")]
    series_ids = [str(series_id) for series_id in book.get("series", []) if str(series_id).strip()]

    if title == "":
        raise ValueError(f"Missing title for book_id={book['book_id']}")

    body = description
    body_source = "description"
    if count_words(description) < min_description_words:
        if fallback_review is None:
            if description == "":
                log.warning(f"No usable text for book_id={book['book_id']}")
                body = "No description or review available."
                body_source = "none"
        else:
            body = fallback_review["text"]
            body_source = "review"

    parts = [f"Title: {title}"]
    if title_without_series and title_without_series != title:
        parts.append(f"Title without series: {title_without_series}")
    if author_ids:
        parts.append(f"Authors: {' '.join(author_ids)}")
    if series_ids:
        parts.append(f"Series: {' '.join(series_ids)}")
    if shelves:
        parts.append(f"Shelves: {', '.join(shelves)}")
    parts.append(body)
    return "\n".join(parts), body_source


def collect_interactions(interaction_path: Path) -> list[dict[str, Any]]:
    interactions = []
    for row in iter_jsonl(interaction_path):
        if not row.get("is_read"):
            continue
        if int(row.get("rating", 0)) <= 0:
            continue
        interactions.append({
            "user_id": row["user_id"],
            "book_id": row["book_id"],
            "rating": int(row["rating"]),
            "timestamp": parse_time(row),
        })
    if not interactions:
        raise ValueError("No valid interactions found")
    return interactions


def build_item_artifacts(
    books_path: Path,
    kept_book_ids: set[str],
    review_fallbacks: dict[str, dict[str, Any]],
    min_description_words: int,
    shelf_topk: int,
) -> tuple[dict[str, int], dict[str, dict[str, Any]], dict[str, int]]:
    item2id: dict[str, int] = {}
    item_payload: dict[str, dict[str, Any]] = {}
    stats = {"num_items": 0, "description_source": 0, "review_source": 0, "none_source": 0}

    item_id = 0
    for book in iter_jsonl(books_path):
        book_id = book["book_id"]
        if book_id not in kept_book_ids:
            continue
        item_text, body_source = build_item_text(book, review_fallbacks.get(book_id), min_description_words, shelf_topk)
        item2id[book_id] = item_id
        item_payload[str(item_id)] = {
            "book_id": book_id,
            "title": normalize_text(book.get("title", "")),
            "title_without_series": normalize_text(book.get("title_without_series", "")),
            "description": normalize_text(book.get("description", "")),
            "item_text": item_text,
            "text_source": body_source,
            "average_rating": book.get("average_rating", ""),
            "ratings_count": book.get("ratings_count", ""),
            "text_reviews_count": book.get("text_reviews_count", ""),
        }
        stats[f"{body_source}_source"] += 1
        item_id += 1

    if len(item2id) != len(kept_book_ids):
        missing = kept_book_ids.difference(item2id)
        raise ValueError(f"Missing books after metadata build: {len(missing)}")

    stats["num_items"] = len(item2id)
    return item2id, item_payload, stats


def build_rows(
    interactions: list[dict[str, Any]],
    item2id: dict[str, int],
    item_payload: dict[str, dict[str, Any]],
    history_max_len: int,
) -> list[list[object]]:
    user_histories = defaultdict(list)
    for row in interactions:
        user_histories[row["user_id"]].append(row)

    rows = []
    for user_id, records in user_histories.items():
        records.sort(key=lambda row: row["timestamp"])
        for index in range(1, len(records)):
            history = records[max(0, index - history_max_len) : index]
            target = records[index]
            history_book_ids = [row["book_id"] for row in history]
            history_item_ids = [item2id[row["book_id"]] for row in history]
            history_titles = [item_payload[str(item2id[row["book_id"]])]["title"] for row in history]
            history_ratings = [row["rating"] for row in history]
            history_timestamps = [row["timestamp"] for row in history]
            target_item_id = item2id[target["book_id"]]
            target_title = item_payload[str(target_item_id)]["title"]
            rows.append([
                user_id,
                history_book_ids,
                target["book_id"],
                history_item_ids,
                target_item_id,
                history_titles,
                target_title,
                history_ratings,
                target["rating"],
                history_timestamps,
                target["timestamp"],
            ])
    rows.sort(key=lambda row: row[-1])
    return rows


def dump_csv(rows: list[list[object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
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
        ])
        writer.writerows(rows)


@task_wrapper
def preprocess(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    books_path = Path(cfg.data_dir) / cfg.books_file
    interactions_path = Path(cfg.data_dir) / cfg.interactions_file
    reviews_path = Path(cfg.data_dir) / cfg.reviews_file
    output_dir = Path(cfg.output_dir)

    review_fallbacks = build_review_fallbacks(
        books_path.parent / cfg.reviews_file, cfg.min_review_rating, cfg.min_review_words
    )
    interactions = collect_interactions(interactions_path)
    kept_book_ids = {row["book_id"] for row in interactions}
    item2id, item_payload, item_stats = build_item_artifacts(
        books_path=books_path,
        kept_book_ids=kept_book_ids,
        review_fallbacks=review_fallbacks,
        min_description_words=cfg.min_description_words,
        shelf_topk=cfg.shelf_topk,
    )
    rows = build_rows(interactions, item2id, item_payload, cfg.history_max_len)

    train_end = int(len(rows) * 0.8)
    valid_end = int(len(rows) * 0.9)

    output_dir.mkdir(parents=True, exist_ok=True)
    dump_csv(rows[:train_end], output_dir / "train.csv")
    dump_csv(rows[train_end:valid_end], output_dir / "valid.csv")
    dump_csv(rows[valid_end:], output_dir / "test.csv")

    with (output_dir / "goodreads.item.json").open("w", encoding="utf-8") as file:
        json.dump(item_payload, file, ensure_ascii=False, indent=2)
    with (output_dir / "goodreads.item2id.json").open("w", encoding="utf-8") as file:
        json.dump(item2id, file, ensure_ascii=False, indent=2)

    metrics = {
        "num_interactions": len(interactions),
        "num_rows": len(rows),
        "num_users": len({row["user_id"] for row in interactions}),
        **item_stats,
    }
    with (output_dir / "stats.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)

    object_dict = {
        "cfg": cfg,
        "books_path": books_path,
        "interactions_path": interactions_path,
        "reviews_path": reviews_path,
        "output_dir": output_dir,
    }
    log.info(f"Saved preprocess artifacts to {output_dir}")
    return metrics, object_dict


@hydra.main(version_base="1.3", config_path="../../configs", config_name="data_preprocess")
def main(cfg: DictConfig) -> None:
    extras(cfg)
    preprocess(cfg)


if __name__ == "__main__":
    main()
