from __future__ import annotations

import argparse
from pathlib import Path

from minionerec_goodreads.utils.sft import build_tokenizer, check_sid_tokenizer_atomicity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check that SID tokens stay atomic in the SFT tokenizer.")
    parser.add_argument(
        "--pretrained-model-name-or-path",
        default="/mnt/disk2/chengqi/models/llm/Qwen/Qwen2.5-3B-Instruct",
        help="Base tokenizer path or Hugging Face model id.",
    )
    parser.add_argument(
        "--sid-index-path",
        default="data/processed/rqvae/goodreads.index.json",
        type=Path,
        help="Path to goodreads.index.json.",
    )
    parser.add_argument(
        "--sample-size",
        default=512,
        type=int,
        help="Number of item SIDs to check. Use -1 to check all items.",
    )
    parser.add_argument("--seed", default=728, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample_size = None if args.sample_size < 0 else args.sample_size
    tokenizer, sid_index, original_vocab_size = build_tokenizer(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        sid_index_path=args.sid_index_path,
    )
    result = check_sid_tokenizer_atomicity(
        tokenizer=tokenizer,
        sid_index=sid_index,
        sample_size=sample_size,
        seed=args.seed,
    )
    print(
        "SID tokenizer atomicity check passed: "
        f"checked_items={result.checked_items}, "
        f"sid_token_count={result.sid_token_count}, "
        f"sid_length=[{result.min_sid_length}, {result.max_sid_length}], "
        f"original_vocab_size={original_vocab_size}, "
        f"augmented_vocab_size={len(tokenizer)}"
    )


if __name__ == "__main__":
    main()
