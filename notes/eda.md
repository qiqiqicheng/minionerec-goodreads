# GoodReads Mystery-Thriller-Crime Raw EDA

Generated at: 2026-05-13 05:06:48

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

- The raw subset has 6,225 books, 340,019 interaction rows, and 52,885 full review rows.
- The current sequential signal keeps 309,454 read-and-rated interactions (91.01% of raw interactions), covering 11,526 users and 6,225 books.
- The project target window 2016-09-01 to 2017-12-31 contains 299,497 read-and-rated interactions (96.78% of valid interactions).
- The MiniOneRec-style row builder produces 297,928 next-item rows, with 289,495 targets in the project window, then uses a global chronological 80/10/10 split by target timestamp.
- Item text is mostly book description, but the review fallback is important for short or empty descriptions: 193 valid items use review text and 91 valid items still have no usable text source.

## Raw Files

| file | rows | size_mb |
| --- | --- | --- |
| goodreads_books_mystery_thriller_crime.json | 6,225 | 31.74 |
| goodreads_interactions_mystery_thriller_crime.json | 340,019 | 117.11 |
| goodreads_reviews_mystery_thriller_crime.json | 52,885 | 52.81 |

## Book Metadata

| metric | value |
| --- | --- |
| unique_book_ids | 6,225 |
| duplicate_book_id_rows | 0 |
| duplicate_title_groups | 336 |
| duplicate_title_extra_rows | 415 |
| duplicate_work_groups | 405 |
| duplicate_work_extra_rows | 501 |

### Missing / Sparse Fields

| field | missing_or_sparse | pct_books |
| --- | --- | --- |
| asin | 4,934 | 79.26% |
| description | 280 | 4.50% |
| description_lt_20_words | 293 | 4.71% |
| isbn | 1,497 | 24.05% |
| isbn13 | 1,421 | 22.83% |
| language_code | 842 | 13.53% |
| num_pages | 1,142 | 18.35% |
| publication_year | 1,643 | 26.39% |
| publisher | 1,763 | 28.32% |
| series | 1,691 | 27.16% |

### Text And Metadata Distributions

| distribution | count | min | p50 | p90 | p95 | p99 | max | mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| title_words | 6,225 | 1 | 6 | 9 | 10 | 12 | 26 | 5.53 |
| description_words | 6,225 | 0 | 149 | 247 | 285 | 383.76 | 716 | 152.20 |
| shelves_per_book | 6,225 | 12 | 100 | 100 | 100 | 100 | 100 | 98.18 |
| authors_per_book | 6,225 | 1 | 1 | 2 | 2 | 2 | 25 | 1.13 |
| series_per_book | 6,225 | 0 | 1 | 1 | 2 | 2 | 4 | 0.79 |
| similar_books_per_book | 6,225 | 0 | 11 | 18 | 18 | 18 | 18 | 10.34 |
| average_rating | 6,225 | 2.68 | 3.94 | 4.22 | 4.32 | 4.49 | 4.66 | 3.93 |
| ratings_count | 6,225 | 49 | 3,844 | 22,892.80 | 38,982.00 | 142,936.44 | 2,046,499 | 12,392.01 |
| text_reviews_count | 6,225 | 5 | 266 | 1,307.60 | 2,136.60 | 6,040.84 | 78,438 | 663.69 |
| num_pages | 5,083 | 0 | 347 | 480 | 535 | 710.62 | 1,796 | 351.77 |

### Top Languages / Formats / Shelves

Languages:

| language | books | pct_books |
| --- | --- | --- |
| eng | 4,273 | 68.64% |
| <missing> | 842 | 13.53% |
| en-US | 794 | 12.76% |
| en-GB | 172 | 2.76% |
| en-CA | 95 | 1.53% |
| ben | 18 | 0.29% |
| swe | 10 | 0.16% |
| fre | 7 | 0.11% |
| dan | 5 | 0.08% |
| spa | 3 | 0.05% |
| nor | 2 | 0.03% |
| ger | 1 | 0.02% |
| ara | 1 | 0.02% |
| en | 1 | 0.02% |
| nl | 1 | 0.02% |

Formats:

| format | books | pct_books |
| --- | --- | --- |
| <missing> | 1,775 | 28.51% |
| Paperback | 1,770 | 28.43% |
| Hardcover | 1,715 | 27.55% |
| Kindle Edition | 346 | 5.56% |
| Mass Market Paperback | 335 | 5.38% |
| ebook | 179 | 2.88% |
| Audio CD | 54 | 0.87% |
| Audiobook | 15 | 0.24% |
| Audio Cassette | 9 | 0.14% |
| Audio | 7 | 0.11% |
| Audible Audio | 6 | 0.10% |
| MP3 CD | 3 | 0.05% |
| Unknown Binding | 3 | 0.05% |
| Library Binding | 2 | 0.03% |
| Trade Paperback | 2 | 0.03% |

Top shelves by book coverage:

| shelf | books | pct_books |
| --- | --- | --- |
| to read | 6,211 | 99.78% |
| currently reading | 6,168 | 99.08% |
| mystery | 6,141 | 98.65% |
| fiction | 6,036 | 96.96% |
| kindle | 5,844 | 93.88% |
| ebook | 5,533 | 88.88% |
| mystery thriller | 5,441 | 87.41% |
| owned | 5,400 | 86.75% |
| library | 5,256 | 84.43% |
| crime | 5,203 | 83.58% |
| favorites | 5,158 | 82.86% |
| books i own | 5,140 | 82.57% |
| suspense | 4,988 | 80.13% |
| thriller | 4,908 | 78.84% |
| mystery suspense | 4,882 | 78.43% |
| mysteries | 4,864 | 78.14% |
| ebooks | 4,826 | 77.53% |
| audiobook | 4,788 | 76.92% |
| audio | 4,756 | 76.40% |
| audiobooks | 4,559 | 73.24% |

Top topical shelves after removing logistics shelves:

| shelf | books | pct_books |
| --- | --- | --- |
| mystery | 6,142 | 98.67% |
| fiction | 6,040 | 97.03% |
| mystery thriller | 5,443 | 87.44% |
| crime | 5,203 | 83.58% |
| suspense | 4,988 | 80.13% |
| thriller | 4,909 | 78.86% |
| mystery suspense | 4,893 | 78.60% |
| mysteries | 4,864 | 78.14% |
| audio | 4,757 | 76.42% |
| series | 4,508 | 72.42% |
| adult | 4,448 | 71.45% |
| to buy | 4,313 | 69.29% |
| wish list | 3,879 | 62.31% |
| read in 2017 | 3,831 | 61.54% |
| adult fiction | 3,805 | 61.12% |
| contemporary | 3,749 | 60.22% |
| thrillers | 3,619 | 58.14% |
| audible | 3,613 | 58.04% |
| audio books | 3,594 | 57.73% |
| e book | 3,566 | 57.29% |

## Interactions

| metric | value |
| --- | --- |
| raw_rows | 340,019 |
| raw_unique_users | 11,731 |
| raw_unique_books | 6,225 |
| read_and_rated_rows | 309,454 |
| read_and_rated_pct | 91.01% |
| valid_unique_users | 11,526 |
| valid_unique_books | 6,225 |
| target_window | 2016-09-01 to 2017-12-31 |
| target_window_rows | 299,497 |
| target_window_rows_pct_valid | 96.78% |
| target_window_unique_users | 11,515 |
| target_window_unique_books | 6,225 |
| target_window_time_range | 2016-09-01 to 2017-12-30 |
| missing_book_valid_rows | 0 |
| valid_time_range | 0028-12-30 to 2017-12-30 |

Raw read/rating cross-tab:

| bucket | rows | pct_rows |
| --- | --- | --- |
| True:rating>0 | 309,454 | 91.01% |
| True:rating=0 | 30,565 | 8.99% |

Target-window rating distribution:

| rating | rows | pct_target_window |
| --- | --- | --- |
| 4 | 123,091 | 41.10% |
| 5 | 86,447 | 28.86% |
| 3 | 71,237 | 23.79% |
| 2 | 14,976 | 5.00% |
| 1 | 3,746 | 1.25% |

Rating distribution for all read-and-rated rows:

| rating | rows | pct_valid |
| --- | --- | --- |
| 4 | 126,919 | 41.01% |
| 5 | 89,211 | 28.83% |
| 3 | 73,967 | 23.90% |
| 2 | 15,469 | 5.00% |
| 1 | 3,888 | 1.26% |

### User / Item Degree

| distribution | count | min | p50 | p90 | p95 | p99 | max | mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| raw_user_degree | 11,731 | 15 | 22 | 49 | 63 | 113 | 532 | 28.98 |
| raw_book_degree | 6,225 | 15 | 29 | 109 | 168 | 410 | 1,856 | 54.62 |
| valid_user_degree | 11,526 | 1 | 21 | 46.50 | 61 | 109 | 532 | 26.85 |
| valid_book_degree | 6,225 | 6 | 27 | 99 | 154 | 387.04 | 1,721 | 49.71 |
| target_window_user_degree | 11,515 | 1 | 20 | 45 | 60 | 107 | 501 | 26.01 |
| target_window_book_degree | 6,225 | 6 | 26 | 96 | 150 | 370.80 | 1,698 | 48.11 |

| metric | value |
| --- | --- |
| valid_users_lt_2 | 84 |
| valid_users_lt_5 | 230 |
| valid_users_lt_30 | 8,555 |
| valid_books_lt_2 | 0 |
| valid_books_lt_5 | 0 |
| valid_books_lt_30 | 3,457 |
| duplicate_user_book_pairs | 0 |
| duplicate_user_book_extra_rows | 0 |

### Target-Window Interaction Months

| month | valid_rows | pct_target_window |
| --- | --- | --- |
| 2016-09 | 21,655 | 7.23% |
| 2016-10 | 25,227 | 8.42% |
| 2016-11 | 25,213 | 8.42% |
| 2016-12 | 26,792 | 8.95% |
| 2017-01 | 36,714 | 12.26% |
| 2017-02 | 28,276 | 9.44% |
| 2017-03 | 30,849 | 10.30% |
| 2017-04 | 29,636 | 9.90% |
| 2017-05 | 29,432 | 9.83% |
| 2017-06 | 27,680 | 9.24% |
| 2017-07 | 7,743 | 2.59% |
| 2017-08 | 4,568 | 1.53% |
| 2017-09 | 3,227 | 1.08% |
| 2017-10 | 2,419 | 0.81% |
| 2017-11 | 55 | 0.02% |
| 2017-12 | 11 | 0.00% |

### Top Valid Interaction Months

| month | valid_rows | pct_valid |
| --- | --- | --- |
| 2017-01 | 36,714 | 11.86% |
| 2017-03 | 30,849 | 9.97% |
| 2017-04 | 29,636 | 9.58% |
| 2017-05 | 29,432 | 9.51% |
| 2017-02 | 28,276 | 9.14% |
| 2017-06 | 27,680 | 8.94% |
| 2016-12 | 26,792 | 8.66% |
| 2016-10 | 25,227 | 8.15% |
| 2016-11 | 25,213 | 8.15% |
| 2016-09 | 21,655 | 7.00% |
| 2017-07 | 7,743 | 2.50% |
| 2017-08 | 4,568 | 1.48% |
| 2017-09 | 3,227 | 1.04% |
| 2016-01 | 2,497 | 0.81% |
| 2017-10 | 2,419 | 0.78% |
| 2016-08 | 586 | 0.19% |
| 2015-01 | 401 | 0.13% |
| 2016-07 | 320 | 0.10% |
| 2016-06 | 292 | 0.09% |
| 2014-01 | 266 | 0.09% |
| 2016-05 | 231 | 0.07% |
| 2013-01 | 197 | 0.06% |
| 2016-03 | 194 | 0.06% |
| 2016-04 | 194 | 0.06% |
| 2012-01 | 188 | 0.06% |
| 2011-01 | 170 | 0.05% |
| 2016-02 | 160 | 0.05% |
| 2010-01 | 143 | 0.05% |
| 2009-01 | 104 | 0.03% |
| 2008-01 | 93 | 0.03% |

### Sequence Rows And Split

| metric | value |
| --- | --- |
| generated_rows | 297,928 |
| target_window_generated_rows | 289,495 |
| target_window_generated_rows_pct | 97.17% |
| history_max_len | 100 |
| history_capped_rows | 7,701 |
| history_capped_rows_pct | 2.58% |
| single_event_users | 84 |
| repeated_item_users | 0 |
| repeated_item_extra_events | 0 |
| adjacent_same_timestamp_pairs | 8,790 |

| split | rows | time_range | target_window_rows | target_window_pct | target_users | target_books | new_users_vs_train | new_books_vs_train | history_p50 | history_p95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 238,342 | 1964-02-07 to 2017-05-16 | 229,909 | 96.46% | 11,176 | 6,178 | 0 | 0 | 12 | 66 |
| valid | 29,793 | 2017-05-16 to 2017-06-18 | 29,793 | 100.00% | 7,541 | 5,477 | 183 | 36 | 20 | 85 |
| test | 29,793 | 2017-06-18 to 2017-12-30 | 29,793 | 100.00% | 7,444 | 5,201 | 168 | 47 | 22 | 93 |

## Reviews And Text Fallback

| metric | value |
| --- | --- |
| review_rows | 52,885 |
| non_empty_reviews | 52,877 |
| non_empty_reviews_pct | 99.98% |
| unique_review_users | 5,363 |
| unique_review_books | 5,708 |
| review_books_in_book_metadata | 5,708 |
| review_books_in_valid_items | 5,708 |
| review_users_in_valid_interactions | 5,354 |
| fallback_candidate_reviews | 28,313 |
| fallback_candidate_books | 4,748 |
| valid_short_description_books | 293 |
| valid_empty_description_books | 280 |
| simulated_description_source | 5,941 |
| simulated_review_source | 193 |
| simulated_none_source | 91 |

| distribution | count | min | p50 | p90 | p95 | p99 | max | mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| review_word_count | 52,885 | 0 | 69 | 324 | 437 | 709 | 2,274 | 126.25 |
| review_votes | 52,885 | 0 | 0 | 5 | 13 | 54 | 432 | 2.87 |
| review_comments | 52,885 | 0 | 0 | 0 | 2 | 12 | 117 | 0.52 |
| fallback_reviews_per_book | 4,748 | 1 | 3 | 14 | 20 | 52.53 | 216 | 5.96 |

Review rating distribution:

| rating | reviews | pct_reviews |
| --- | --- | --- |
| 4 | 19,947 | 37.72% |
| 5 | 15,715 | 29.72% |
| 3 | 11,320 | 21.40% |
| 2 | 3,645 | 6.89% |
| 1 | 1,221 | 2.31% |
| 0 | 1,037 | 1.96% |

## Interview Notes

- Filtering matters: only read-and-rated interactions become supervision; most raw rows are not direct positive signals.
- Raw k-core=30 does not survive the read-and-rated filter, so the training matrix is much sparser than the original subset.
- The current split is globally chronological by target event, not leave-one-out per user. This is good for temporal realism, but evaluation history can include interactions from earlier validation/test timestamps.
- Some raw timestamps are outside the project window, so always state whether an analysis is full-history or 2016-09 to 2017-12 only.
- The item encoder is transductive: it embeds every valid item before train/valid/test split, including items that are only targets outside the training window.
- Description sparsity is material, so review fallback is not a cosmetic step; it decides whether RQVAE sees meaningful text for cold or sparse items.
- Shelf names are noisy because logistics shelves dominate, so any shelf feature should filter shelves such as to-read/currently-reading/kindle before treating them as semantics.
- Repeated user-book pairs exist; if the target task is next unique item rather than next event, deduplication policy should be stated explicitly.
