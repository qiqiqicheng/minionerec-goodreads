# GoodReads Mystery-Thriller-Crime Raw EDA

Generated at: 2026-05-12 03:26:17

## Configuration

| key | value |
| --- | --- |
| books_file | goodreads_books_mystery_thriller_crime.json |
| interactions_file | goodreads_interactions_mystery_thriller_crime.json |
| reviews_file | goodreads_reviews_mystery_thriller_crime.json |
| min_review_rating | 3 |
| min_review_words | 50 |
| min_description_words | 20 |
| history_max_len | 100 |
| target_start | 2016-09-01 |
| target_end | 2017-12-31 |

## Executive Summary

- The raw subset has 16,219 books, 2,327,295 interaction rows, and 118,067 full review rows.
- The current sequential signal keeps 552,092 read-and-rated interactions (23.72% of raw interactions), covering 25,971 users and 15,766 books.
- The project target window 2016-09-01 to 2017-12-31 contains 537,958 read-and-rated interactions (97.44% of valid interactions).
- The MiniOneRec-style row builder produces 526,121 next-item rows, with 514,491 targets in the project window, then uses a global chronological 80/10/10 split by target timestamp.
- Item text is mostly book description, but the review fallback is important for short or empty descriptions: 596 valid items use review text and 292 valid items still have no usable text source.

## Raw Files

| file | rows | size_mb |
| --- | --- | --- |
| goodreads_books_mystery_thriller_crime.json | 16,219 | 80.14 |
| goodreads_interactions_mystery_thriller_crime.json | 2,327,295 | 697.87 |
| goodreads_reviews_mystery_thriller_crime.json | 118,067 | 121.87 |

## Book Metadata

| metric | value |
| --- | --- |
| unique_book_ids | 16,219 |
| duplicate_book_id_rows | 0 |
| duplicate_title_groups | 1,238 |
| duplicate_title_extra_rows | 1,748 |
| duplicate_work_groups | 1,467 |
| duplicate_work_extra_rows | 2,052 |

### Missing / Sparse Fields

| field | missing_or_sparse | pct_books |
| --- | --- | --- |
| asin | 12,880 | 79.41% |
| description | 890 | 5.49% |
| description_lt_20_words | 924 | 5.70% |
| isbn | 4,205 | 25.93% |
| isbn13 | 3,783 | 23.32% |
| language_code | 3,578 | 22.06% |
| num_pages | 3,484 | 21.48% |
| publication_year | 3,893 | 24.00% |
| publisher | 4,344 | 26.78% |
| series | 5,732 | 35.34% |

### Text And Metadata Distributions

| distribution | count | min | p50 | p90 | p95 | p99 | max | mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| title_words | 16,219 | 1 | 5 | 9 | 10 | 13 | 26 | 5.43 |
| description_words | 16,219 | 0 | 150 | 248 | 289 | 409.82 | 2,607 | 153.48 |
| shelves_per_book | 16,219 | 4 | 100 | 100 | 100 | 100 | 100 | 92.67 |
| authors_per_book | 16,219 | 1 | 1 | 2 | 2 | 3 | 51 | 1.17 |
| series_per_book | 16,219 | 0 | 1 | 1 | 1 | 2 | 4 | 0.68 |
| similar_books_per_book | 16,219 | 0 | 12 | 18 | 18 | 18 | 18 | 10.05 |
| average_rating | 16,219 | 0 | 3.92 | 4.25 | 4.36 | 4.56 | 5 | 3.91 |
| ratings_count | 16,219 | 0 | 1,069 | 9,890.80 | 18,690.80 | 66,243.48 | 2,046,499 | 5,475.22 |
| text_reviews_count | 16,219 | 1 | 110 | 651.20 | 1,111.20 | 3,234.48 | 78,438 | 324.34 |
| num_pages | 12,735 | 0 | 334 | 460 | 512 | 683.32 | 2,435 | 338.37 |

### Top Languages / Formats / Shelves

Languages:

| language | books | pct_books |
| --- | --- | --- |
| eng | 10,266 | 63.30% |
| <missing> | 3,578 | 22.06% |
| en-US | 1,706 | 10.52% |
| en-GB | 437 | 2.69% |
| en-CA | 164 | 1.01% |
| fre | 19 | 0.12% |
| swe | 15 | 0.09% |
| spa | 9 | 0.06% |
| ger | 8 | 0.05% |
| dan | 5 | 0.03% |
| nor | 4 | 0.02% |
| en | 4 | 0.02% |
| isl | 1 | 0.01% |
| cze | 1 | 0.01% |
| ara | 1 | 0.01% |

Formats:

| format | books | pct_books |
| --- | --- | --- |
| Paperback | 5,819 | 35.88% |
| <missing> | 4,166 | 25.69% |
| Hardcover | 3,812 | 23.50% |
| Kindle Edition | 984 | 6.07% |
| Mass Market Paperback | 645 | 3.98% |
| ebook | 539 | 3.32% |
| Audio CD | 135 | 0.83% |
| Audiobook | 26 | 0.16% |
| Audio | 23 | 0.14% |
| Audible Audio | 17 | 0.10% |
| Audio Cassette | 13 | 0.08% |
| Trade Paperback | 11 | 0.07% |
| Unknown Binding | 10 | 0.06% |
| Library Binding | 5 | 0.03% |
| MP3 CD | 3 | 0.02% |

Top shelves by book coverage:

| shelf | books | pct_books |
| --- | --- | --- |
| to read | 16,153 | 99.59% |
| currently reading | 15,846 | 97.70% |
| mystery | 15,333 | 94.54% |
| fiction | 14,814 | 91.34% |
| kindle | 13,820 | 85.21% |
| ebook | 12,504 | 77.09% |
| owned | 12,298 | 75.82% |
| mystery thriller | 11,839 | 72.99% |
| thriller | 11,741 | 72.39% |
| crime | 11,605 | 71.55% |
| books i own | 11,388 | 70.21% |
| suspense | 11,115 | 68.53% |
| library | 11,114 | 68.52% |
| favorites | 11,043 | 68.09% |
| ebooks | 10,322 | 63.64% |
| mysteries | 10,283 | 63.40% |
| series | 9,963 | 61.43% |
| mystery suspense | 9,909 | 61.10% |
| owned books | 9,253 | 57.05% |
| to buy | 9,163 | 56.50% |

Top topical shelves after removing logistics shelves:

| shelf | books | pct_books |
| --- | --- | --- |
| mystery | 15,342 | 94.59% |
| fiction | 14,828 | 91.42% |
| mystery thriller | 11,846 | 73.04% |
| thriller | 11,749 | 72.44% |
| crime | 11,605 | 71.55% |
| suspense | 11,117 | 68.54% |
| mysteries | 10,283 | 63.40% |
| series | 9,972 | 61.48% |
| mystery suspense | 9,939 | 61.28% |
| to buy | 9,163 | 56.50% |
| wish list | 8,923 | 55.02% |
| adult | 8,668 | 53.44% |
| audio | 8,327 | 51.34% |
| contemporary | 8,117 | 50.05% |
| read in 2017 | 7,984 | 49.23% |
| e book | 7,250 | 44.70% |
| thrillers | 7,168 | 44.20% |
| adult fiction | 7,001 | 43.17% |
| e books | 6,704 | 41.33% |
| read 2017 | 6,676 | 41.16% |

## Interactions

| metric | value |
| --- | --- |
| raw_rows | 2,327,295 |
| raw_unique_users | 28,592 |
| raw_unique_books | 16,219 |
| read_and_rated_rows | 552,092 |
| read_and_rated_pct | 23.72% |
| valid_unique_users | 25,971 |
| valid_unique_books | 15,766 |
| target_window | 2016-09-01 to 2017-12-31 |
| target_window_rows | 537,958 |
| target_window_rows_pct_valid | 97.44% |
| target_window_unique_users | 25,917 |
| target_window_unique_books | 15,757 |
| target_window_time_range | 2016-09-01 to 2017-12-29 |
| missing_book_valid_rows | 0 |
| valid_time_range | 0028-12-30 to 3103-03-15 |

Raw read/rating cross-tab:

| bucket | rows | pct_rows |
| --- | --- | --- |
| False:rating=0 | 1,701,200 | 73.10% |
| True:rating>0 | 552,092 | 23.72% |
| True:rating=0 | 74,003 | 3.18% |

Target-window rating distribution:

| rating | rows | pct_target_window |
| --- | --- | --- |
| 4 | 220,370 | 40.96% |
| 5 | 149,707 | 27.83% |
| 3 | 130,584 | 24.27% |
| 2 | 29,473 | 5.48% |
| 1 | 7,824 | 1.45% |

Rating distribution for all read-and-rated rows:

| rating | rows | pct_valid |
| --- | --- | --- |
| 4 | 225,965 | 40.93% |
| 5 | 153,479 | 27.80% |
| 3 | 134,464 | 24.36% |
| 2 | 30,166 | 5.46% |
| 1 | 8,018 | 1.45% |

### User / Item Degree

| distribution | count | min | p50 | p90 | p95 | p99 | max | mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| raw_user_degree | 28,592 | 30 | 50 | 141 | 215 | 556.36 | 3,929 | 81.40 |
| raw_book_degree | 16,219 | 30 | 69 | 278 | 438 | 1,335.48 | 8,938 | 143.49 |
| valid_user_degree | 25,971 | 1 | 14 | 46 | 64 | 123 | 835 | 21.26 |
| valid_book_degree | 15,766 | 1 | 15 | 71 | 121 | 326.35 | 3,179 | 35.02 |
| target_window_user_degree | 25,917 | 1 | 13 | 46 | 62 | 119 | 793 | 20.76 |
| target_window_book_degree | 15,757 | 1 | 15 | 69 | 118.20 | 320.44 | 3,150 | 34.14 |

| metric | value |
| --- | --- |
| valid_users_lt_2 | 1,725 |
| valid_users_lt_5 | 5,482 |
| valid_users_lt_30 | 19,835 |
| valid_books_lt_2 | 624 |
| valid_books_lt_5 | 2,245 |
| valid_books_lt_30 | 11,582 |
| duplicate_user_book_pairs | 0 |
| duplicate_user_book_extra_rows | 0 |

### Target-Window Interaction Months

| month | valid_rows | pct_target_window |
| --- | --- | --- |
| 2016-09 | 27,229 | 5.06% |
| 2016-10 | 33,025 | 6.14% |
| 2016-11 | 32,348 | 6.01% |
| 2016-12 | 34,859 | 6.48% |
| 2017-01 | 48,461 | 9.01% |
| 2017-02 | 37,439 | 6.96% |
| 2017-03 | 41,663 | 7.74% |
| 2017-04 | 40,193 | 7.47% |
| 2017-05 | 40,966 | 7.62% |
| 2017-06 | 41,644 | 7.74% |
| 2017-07 | 44,918 | 8.35% |
| 2017-08 | 42,629 | 7.92% |
| 2017-09 | 40,752 | 7.58% |
| 2017-10 | 31,488 | 5.85% |
| 2017-11 | 330 | 0.06% |
| 2017-12 | 14 | 0.00% |

### Top Valid Interaction Months

| month | valid_rows | pct_valid |
| --- | --- | --- |
| 2017-01 | 48,461 | 8.78% |
| 2017-07 | 44,918 | 8.14% |
| 2017-08 | 42,629 | 7.72% |
| 2017-03 | 41,663 | 7.55% |
| 2017-06 | 41,644 | 7.54% |
| 2017-05 | 40,966 | 7.42% |
| 2017-09 | 40,752 | 7.38% |
| 2017-04 | 40,193 | 7.28% |
| 2017-02 | 37,439 | 6.78% |
| 2016-12 | 34,859 | 6.31% |
| 2016-10 | 33,025 | 5.98% |
| 2016-11 | 32,348 | 5.86% |
| 2017-10 | 31,488 | 5.70% |
| 2016-09 | 27,229 | 4.93% |
| 2016-01 | 3,226 | 0.58% |
| 2016-08 | 819 | 0.15% |
| 2015-01 | 608 | 0.11% |
| 2016-07 | 438 | 0.08% |
| 2014-01 | 432 | 0.08% |
| 2016-06 | 349 | 0.06% |
| 2017-11 | 330 | 0.06% |
| 2016-05 | 297 | 0.05% |
| 2013-01 | 280 | 0.05% |
| 2012-01 | 274 | 0.05% |
| 2016-03 | 271 | 0.05% |
| 2016-04 | 270 | 0.05% |
| 2011-01 | 253 | 0.05% |
| 2016-02 | 234 | 0.04% |
| 2010-01 | 221 | 0.04% |
| 2005-01 | 170 | 0.03% |

### Sequence Rows And Split

| metric | value |
| --- | --- |
| generated_rows | 526,121 |
| target_window_generated_rows | 514,491 |
| target_window_generated_rows_pct | 97.79% |
| history_max_len | 100 |
| history_capped_rows | 24,058 |
| history_capped_rows_pct | 4.57% |
| single_event_users | 1,725 |
| repeated_item_users | 0 |
| repeated_item_extra_events | 0 |
| adjacent_same_timestamp_pairs | 13,801 |

| split | rows | time_range | target_window_rows | target_window_pct | target_users | target_books | new_users_vs_train | new_books_vs_train | history_p50 | history_p95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 420,896 | 1965-01-01 to 2017-08-07 | 409,289 | 97.24% | 23,081 | 15,108 | 0 | 0 | 14 | 87 |
| valid | 52,612 | 2017-08-07 to 2017-09-14 | 52,612 | 100.00% | 14,356 | 10,435 | 645 | 357 | 25 | 100 |
| test | 52,613 | 2017-09-14 to 3103-03-15 | 52,590 | 99.96% | 14,566 | 10,530 | 810 | 503 | 28 | 100 |

## Reviews And Text Fallback

| metric | value |
| --- | --- |
| review_rows | 118,067 |
| non_empty_reviews | 118,036 |
| non_empty_reviews_pct | 99.97% |
| unique_review_users | 13,275 |
| unique_review_books | 14,024 |
| review_books_in_book_metadata | 14,024 |
| review_books_in_valid_items | 13,989 |
| review_users_in_valid_interactions | 13,220 |
| fallback_candidate_reviews | 64,360 |
| fallback_candidate_books | 11,462 |
| valid_short_description_books | 910 |
| valid_empty_description_books | 876 |
| simulated_description_source | 14,878 |
| simulated_review_source | 596 |
| simulated_none_source | 292 |

| distribution | count | min | p50 | p90 | p95 | p99 | max | mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| review_word_count | 118,067 | 0 | 73 | 337 | 452 | 736 | 3,437 | 132.55 |
| review_votes | 118,067 | 0 | 0 | 5 | 12 | 52 | 432 | 2.80 |
| review_comments | 118,067 | 0 | 0 | 0 | 2 | 12 | 117 | 0.51 |
| fallback_reviews_per_book | 11,462 | 1 | 2 | 12 | 19 | 52 | 328 | 5.62 |

Review rating distribution:

| rating | reviews | pct_reviews |
| --- | --- | --- |
| 4 | 43,749 | 37.05% |
| 5 | 34,617 | 29.32% |
| 3 | 25,299 | 21.43% |
| 2 | 8,351 | 7.07% |
| 0 | 3,234 | 2.74% |
| 1 | 2,817 | 2.39% |

## Interview Notes

- Filtering matters: only read-and-rated interactions become supervision; most raw rows are not direct positive signals.
- Raw k-core=30 does not survive the read-and-rated filter, so the training matrix is much sparser than the original subset.
- The current split is globally chronological by target event, not leave-one-out per user. This is good for temporal realism, but evaluation history can include interactions from earlier validation/test timestamps.
- Some raw timestamps are outside the project window, so always state whether an analysis is full-history or 2016-09 to 2017-12 only.
- The item encoder is transductive: it embeds every valid item before train/valid/test split, including items that are only targets outside the training window.
- Description sparsity is material, so review fallback is not a cosmetic step; it decides whether RQVAE sees meaningful text for cold or sparse items.
- Shelf names are noisy because logistics shelves dominate, so any shelf feature should filter shelves such as to-read/currently-reading/kindle before treating them as semantics.
- Repeated user-book pairs exist; if the target task is next unique item rather than next event, deduplication policy should be stated explicitly.
