# GoodReads RQVAE Split EDA

Generated at: 2026-05-13 05:06:48

## Configuration

| key | value |
| --- | --- |
| rqvae_dir | ./data//processed/rqvae |
| history_max_len | 100 |
| target_start | 2016-09-01 |
| target_end | 2017-12-31 |
| rqvae_output_markdown | ./notes/rqvae_eda.md |
| rqvae_output_json | ./notes/rqvae_eda_metrics.json |

## Executive Summary

- The processed sequence files contain 295,730 next-item rows: 236,584 train, 29,573 valid, and 29,573 test.
- Split row counts do not match stats.json num_rows (-); this is the first reproducibility check for preprocessing.
- The split is globally chronological by target timestamp: train<=valid is True, valid<=test is True. Valid/test histories can still include earlier events from the same evaluation split.
- Target item coverage uses 6,125 unique items from an item payload of 6,125 items; valid/test introduce 36/47 target items not seen as train targets.
- SID/embedding artifact check reports item payload=6,125, SID index=6,125, embedding_shape=[6125, 2560].

## Processed Artifacts

| artifact | value |
| --- | --- |
| preprocess_num_rows | - |
| observed_split_rows | 295,730 |
| row_count_matches_preprocess_stats | - |
| preprocess_num_users | - |
| preprocess_num_items | - |
| item_payload_items | 6,125 |
| item2id_books | 6,125 |
| item2id_unique_item_ids | 6,125 |
| item2id_items_missing_payload | 0 |
| payload_items_missing_item2id | 0 |
| sid_items | 6,125 |
| sid_items_missing_payload | 0 |
| sid_extra_items_not_in_payload | 0 |
| embedding_exists | True |
| embedding_shape | [6125, 2560] |
| embedding_dtype | float32 |

### File Sizes

| file | rows_or_items | size_mb | time_range |
| --- | --- | --- | --- |
| train.csv | 236,584 | 306.57 | 1964-02-07 to 2017-05-16 |
| valid.csv | 29,573 | 55.67 | 2017-05-16 to 2017-06-18 |
| test.csv | 29,573 | 59.36 | 2017-06-18 to 2017-12-30 |
| goodreads.item.json | 6,125 | 13.59 | - |
| goodreads.item2id.json | 6,125 | 0.11 | - |
| goodreads.index.json | 6,125 | 0.43 | - |
| goodreads.emb-bge.npy | [6125, 2560] | 59.81 | - |

### Item Text And Metadata

| distribution | count | min | p50 | p90 | p95 | p99 | max | mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| item_text_words | 6,125 | 34 | 172 | 270 | 308 | 424.52 | 1,056 | 179.08 |
| title_words | 6,125 | 1 | 6 | 9 | 10 | 12 | 26 | 5.52 |
| description_words | 6,125 | 0 | 150 | 248 | 285.80 | 386.28 | 716 | 154.67 |
| average_rating | 6,125 | 2.68 | 3.94 | 4.22 | 4.32 | 4.49 | 4.66 | 3.93 |
| ratings_count | 6,125 | 49 | 3,864 | 23,162.40 | 39,280.60 | 144,879.76 | 2,046,499 | 12,521.67 |
| sid_token_length | 6,125 | 4 | 4 | 4 | 5 | 5 | 5 | 4.06 |

Text source coverage:

| scope | source | count | pct_scope |
| --- | --- | --- | --- |
| item_payload_unique_items | review | 193 | 3.15% |
| item_payload_unique_items | description | 5,932 | 96.85% |
| train_target_rows | description | 232,694 | 98.36% |
| train_target_rows | review | 3,890 | 1.64% |
| train_target_unique_items | description | 5,885 | 96.82% |
| train_target_unique_items | review | 193 | 3.18% |
| valid_target_rows | description | 29,061 | 98.27% |
| valid_target_rows | review | 512 | 1.73% |
| valid_target_unique_items | description | 5,242 | 97.24% |
| valid_target_unique_items | review | 149 | 2.76% |
| test_target_rows | description | 29,099 | 98.40% |
| test_target_rows | review | 474 | 1.60% |
| test_target_unique_items | description | 4,987 | 97.23% |
| test_target_unique_items | review | 142 | 2.77% |

## Split Overview

| split | rows | pct_rows | time_range | target_window_rows | target_window_pct | target_users | target_items | new_users_vs_train | new_items_vs_train | history_p50 | history_p95 | history_eq_max_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 236,584 | 80.00% | 1964-02-07 to 2017-05-16 | 228,251 | 96.48% | 11,176 | 6,078 | 0 | 0 | 12 | 65 | 2.27% |
| valid | 29,573 | 10.00% | 2017-05-16 to 2017-06-18 | 29,573 | 100.00% | 7,538 | 5,391 | 182 | 36 | 20 | 84 | 3.51% |
| test | 29,573 | 10.00% | 2017-06-18 to 2017-12-30 | 29,573 | 100.00% | 7,432 | 5,129 | 167 | 47 | 22 | 92 | 4.12% |

### Rating Distribution

| split | rating_1 | rating_2 | rating_3 | rating_4 | rating_5 |
| --- | --- | --- | --- | --- | --- |
| train | 2,955 | 11,544 | 56,452 | 96,936 | 68,697 |
| valid | 378 | 1,589 | 7,176 | 12,046 | 8,384 |
| test | 386 | 1,698 | 7,196 | 12,373 | 7,920 |

### History Provenance

| split | history_events | train_events | train_pct | valid_events | valid_pct | test_events | test_pct |
| --- | --- | --- | --- | --- | --- | --- | --- |
| train | 4,431,647 | 4,431,647 | 100.00% | 0 | 0.00% | 0 | 0.00% |
| valid | 829,195 | 621,859 | 75.00% | 207,336 | 25.00% | 0 | 0.00% |
| test | 898,102 | 607,953 | 67.69% | 108,966 | 12.13% | 181,183 | 20.17% |

### Integrity Checks

| split | length_mismatch_rows | history_len_gt_max | nonmonotonic_history | target_before_history | row_order_violations | target_missing_payload | target_item2id_mismatch | target_missing_sid | history_missing_sid_events |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| valid | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| test | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

### Repeats And Duplicates

| split | target_in_history_rows | target_in_history_pct | immediate_repeat_rows | immediate_repeat_pct | duplicate_user_item_pairs | duplicate_user_item_extra_rows |
| --- | --- | --- | --- | --- | --- | --- |
| train | 0 | 0.00% | 0 | 0.00% | 0 | 0 |
| valid | 0 | 0.00% | 0 | 0.00% | 0 | 0 |
| test | 0 | 0.00% | 0 | 0.00% | 0 | 0 |

## SID Health

| metric | value |
| --- | --- |
| status | warning |
| assignment_mode | argmin |
| sid_health_num_items | 6,125 |
| num_code_levels | 4 |
| unique_raw_code_paths | 5,773 |
| num_collisions | 352 |
| collision_rate | 0.06 |
| dedup_unique_sid_paths | 6,125 |
| dedup_duplicate_sid_path_groups | 0 |
| dedup_duplicate_sid_path_extra_items | 0 |

| level | codebook_size | used_codes | used_pct | dead_codes | perplexity | max_bucket_share |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 128 | 128 | 100.00% | 0 | 61.70 | 8.07% |
| 1 | 128 | 126 | 98.44% | 2 | 86.02 | 4.95% |
| 2 | 256 | 230 | 89.84% | 26 | 125.29 | 3.22% |
| 3 | 256 | 241 | 94.14% | 15 | 124.21 | 4.49% |

## Popularity Skew

Top target items across all splits:

| item_id | book_id | title | target_rows | pct_rows |
| --- | --- | --- | --- | --- |
| 795 | 28187230 | The Woman in Cabin 10 | 1,598 | 0.54% |
| 52 | 28815474 | The Couple Next Door | 1,354 | 0.46% |
| 2071 | 33151805 | Into the Water | 1,354 | 0.46% |
| 5163 | 22557272 | The Girl on the Train | 1,342 | 0.45% |
| 844 | 29437949 | Behind Closed Doors | 1,342 | 0.45% |
| 577 | 28016509 | The Girl Before | 1,066 | 0.36% |
| 1455 | 28965131 | Behind Her Eyes | 1,005 | 0.34% |
| 4732 | 19288043 | Gone Girl | 962 | 0.33% |
| 3316 | 27833670 | Dark Matter | 946 | 0.32% |
| 4624 | 19486412 | Big Little Lies | 889 | 0.30% |
| 5772 | 968 | The Da Vinci Code (Robert Langdon, #2) | 881 | 0.30% |
| 5771 | 960 | Angels & Demons (Robert Langdon, #1) | 875 | 0.30% |
| 3390 | 2429135 | The Girl with the Dragon Tattoo (Millennium, #1) | 834 | 0.28% |
| 94 | 23212667 | All the Missing Girls | 695 | 0.24% |
| 5852 | 23346377 | In a Dark, Dark Wood | 689 | 0.23% |
| 2605 | 29430013 | The Trespasser (Dublin Murder Squad #6) | 686 | 0.23% |
| 1336 | 27824826 | The Dry (Aaron Falk, #1) | 682 | 0.23% |
| 3651 | 26245850 | Before the Fall | 668 | 0.23% |
| 5499 | 18045891 | Sharp Objects | 643 | 0.22% |
| 103 | 5060378 | The Girl Who Played with Fire (Millennium, #2) | 622 | 0.21% |

Top target users across all splits:

| user_id | target_rows | pct_rows |
| --- | --- | --- |
| bb1163b8cc4aee152f1e99af99a87490 | 521 | 0.18% |
| 677db40dcc394b96708514b2556b6e0f | 448 | 0.15% |
| 0460bec9c281aaae528a02be6e9c5f8b | 406 | 0.14% |
| 6f21e5d4f2ff2a07e878dea8317b91c7 | 378 | 0.13% |
| 9d3033c818113bd2b02bdfd607070bf3 | 331 | 0.11% |
| 3a0b6865d75a3a32ae620484b61d48c4 | 316 | 0.11% |
| e1bee9904679f4847ecec0700140c6b2 | 274 | 0.09% |
| a0002c45abd2a4c73b414082bc1d2619 | 269 | 0.09% |
| dbc9dd3bc9e9a8a36e76220ef813d726 | 256 | 0.09% |
| 06d17b0ac0f8be4d8a820936181d6995 | 250 | 0.08% |
| 850a14bd35fbeb218851ffdeb8a2ee8f | 250 | 0.08% |
| b762192fb6533d14cfb48a878f868c50 | 247 | 0.08% |
| 0120b61139bff5571cb0d8520d988d40 | 242 | 0.08% |
| 1a637cbff7ee5e09796ac36ae3b1080c | 236 | 0.08% |
| cc94815ec6a178ad4e34b9fdfdfecd83 | 233 | 0.08% |
| 2d20885f39418edac3a0a18f9e1c3309 | 232 | 0.08% |
| 1e4971853c5e9a451bd8366d5ff118c9 | 227 | 0.08% |
| e9ba5a04aeaf29e521b4bc9f288caffb | 227 | 0.08% |
| 7bc03cf5e9e7044a76890983bb857343 | 222 | 0.08% |
| 93d40b1e28189ca4ff7bb5ad229a1000 | 205 | 0.07% |

## Interview Notes

- First verify artifact consistency before discussing model quality: split rows should match stats.json, item2id should match item payload, and SID/embedding counts should match the current item payload.
- If the SID index has more rows than the current item payload, treat it as a stale-artifact signal and regenerate SIDs before trusting SFT data.
- This dataset is a next-event task, not next-unique-item: rows where the target already appears in history quantify repeat-consumption pressure.
- The split is global chronological by target event. Valid/test histories containing earlier valid/test events imply roll-forward evaluation rather than a strict train-only user history snapshot.
- New valid/test target items versus train targets are not necessarily item-encoder cold-start items because item text, embeddings, and SIDs are built transductively over the full processed item set.
- History length at the configured maximum should be reported because SFT context cost and truncation bias both depend on it.
- Review-sourced item text matters for SID quality; if those items cluster in validation/test, text fallback quality becomes an evaluation confounder.
