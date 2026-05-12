# MiniOneRec Reproduction Notes for GoodReads

Purpose: token-efficient project-specific paper note for future AI agents. Read this before `notes/paper.md` unless exact paper wording is needed.

## 1. Paper In One Page

MiniOneRec is an open-source generative recommendation framework. The main idea is to replace huge item embedding tables with compact Semantic IDs (SIDs), then train an autoregressive LLM to generate the SID of the next item from a user's chronological history.

Pipeline:
1. Build item text from metadata.
2. Encode item text with a frozen text encoder.
3. Train RQ-VAE on item embeddings to tokenize each item into multi-level discrete SIDs.
4. Add SID tokens to an LLM tokenizer.
5. SFT the LLM on next-item recommendation plus SID-text alignment tasks.
6. Optionally apply recommendation-oriented RL, usually GRPO with constrained beam search and rule/rank rewards.

Why this matters:
- Traditional recommenders put many parameters in item/user embedding tables and often plateau when scaled.
- Generative recommenders put the catalog in a small SID vocabulary and allocate capacity to Transformer reasoning/sequential modeling.
- The paper's main claims: generative rec has better scaling, full-process SID-language alignment matters, and RL with constrained valid-item decoding improves ranking/diversity.

## 2. Formal Task

For each user `u`, sort interactions by time:

`H_u = [i_1, i_2, ..., i_T]`

Each item `i` is mapped to a SID sequence:

`SID(i) = [c_0^i, c_1^i, c_2^i]`

The LLM learns:

`p(SID(i_{t+1}) | SID(i_1), ..., SID(i_t))`

In this repo, we also use title-history fusion tasks:

`p(SID(i_{t+1}) | title(i_1), ..., title(i_t))`

Inference should generate valid SID tokens only, then map generated SIDs back to catalog items.

## 3. Item Tokenization / RQ-VAE

Paper default:
- Text source: title + description.
- Text encoder: Qwen3-Embedding-4B.
- RQ-VAE: 3 residual codebooks, each size 256; each item gets 3 code indices.
- Code capacity: `256^3 = 16,777,216`, enough for much larger catalogs.
- Loss: reconstruction MSE plus RQ commitment/codebook loss.
- Codebook init: k-means on early training batch to avoid collapse.

Math:
- residual starts `r_0 = x`, where `x` is item text embedding.
- level `l`: choose nearest codebook vector `e_{c_l}^{(l)}`.
- update `r_{l+1} = r_l - e_{c_l}^{(l)}`.
- quantized vector `z_q = sum_l e_{c_l}^{(l)}`.

Current repo:
- `src/minionerec_goodreads/scripts/rqvae/data_preprocess.py`: builds item payload and sequence rows.
- `src/minionerec_goodreads/scripts/rqvae/text2emb.py`: Qwen3-Embedding-4B -> `goodreads.emb-bge.npy`.
- `src/minionerec_goodreads/models/rqvae.py`: MLP encoder/decoder + residual vector quantizer.
- `src/minionerec_goodreads/scripts/rqvae/generate_sid.py`: checkpoint -> `goodreads.index.json`.
- `src/minionerec_goodreads/configs/model/rqvae.yaml`: `codebook_size_list: [256, 256, 256]`, `emb_dim: 64`, `loss_beta: 0.25`, k-means init, Sinkhorn during train.
- Important mismatch to watch: config currently says `in_dim: 768`, but existing Qwen3 embeddings have shape `(15766, 2560)`. Future runs must align RQ-VAE `in_dim` with embedding dim or reduce embeddings first.
- SID token format in repo: `<a_12><b_34><c_56>`. Raw code indices are shifted by `+1`. If collisions occur, `generate_sid.py` appends a duplicate rank as a 4th token level, but this is a collision workaround, not the paper default.

## 4. LLM Alignment / SFT Tasks

Paper alignment tasks:
- Recommendation: SID history -> next SID.
- Asymmetric item prediction: title history -> next SID; SID history -> next title.
- SID-text semantic alignment: SID -> title; title -> SID.
- Description reconstruction: SID <-> description, SFT only.
- User preference summarization: SID sequence -> natural-language user profile, SFT only; paper uses external LLM pseudo labels.

Current repo implements a smaller subset in `src/minionerec_goodreads/dataset/sft_dataset.py`:
- `seq_sid_to_sid`: SID history -> next SID.
- `sid_to_title`: SID -> title.
- `title_to_sid`: title -> SID.
- `title_history_to_sid`: title history -> next SID.

Current repo does not yet implement:
- SID history -> next title.
- SID <-> description reconstruction.
- preference summarization.
- RL/GRPO.
- constrained decoding / top-K HR/NDCG evaluation.

Current SFT model:
- `src/minionerec_goodreads/models/sft.py` adds SID tokens to tokenizer.
- `train_mode` options: `new_token_only`, `qlora`, `full_finetune`.
- Default `sft_new_token_only` freezes base model and trains only new SID token embeddings, useful as cheap alignment but weaker than full SFT/QLoRA.
- QLoRA config exists and is likely the practical next step for a 24GB 3090/4090.

## 5. RL / GRPO From Paper

Paper uses GRPO after SFT:
1. For each prompt, generate `G` candidates.
2. Score each candidate.
3. Normalize rewards within group to produce advantages.
4. Optimize clipped policy objective with KL to reference model.

Recommendation-specific issues:
- Output space is closed: valid item SIDs only.
- Random sampling duplicates candidates because catalog/SID action space is small.
- Binary correct/incorrect reward is too sparse for ranking.

Paper solution:
- Constrained decoding: only legal SID/title tokens can be generated at each step.
- Beam search, width 16, default sampler; more efficient/diverse than dynamic sampling.
- Reward = rule reward + rank reward.
- Rule reward: `1` if generated item equals target else `0`.
- Rank reward: penalize hard negatives more strongly when the model ranks them high.
- Collaborative reward from SASRec was tried but hurt due to reward hacking/misalignment.

For this repo, RL is not implemented yet. If adding it, first implement valid SID trie decoding and HR/NDCG evaluation; otherwise RL reward cannot be trusted.

## 6. Evaluation Metrics

Paper reports HR@K and NDCG@K, usually K in `{3,5,10}`.

Definitions:
- HR@K: target item appears in top-K generated candidates.
- NDCG@K: target item appears at rank `r <= K`, score `1/log2(r+1)`, else 0.

Important for this repo:
- Current `src/minionerec_goodreads/scripts/eval.py` only uses Lightning test loss. It is not recommendation ranking evaluation.
- Proper MiniOneRec reproduction needs generation-time evaluation: prompt -> constrained decode top-K SIDs -> map to item_id -> compute HR/NDCG.
- Need to decide whether candidate generation is over all catalog items or only valid split items. Paper style is catalog-level valid items.

## 7. Current GoodReads Dataset Facts

Raw files:
- books: `16,219` rows.
- interactions: `2,327,295` rows.
- reviews: `118,067` rows.
- category: GoodReads mystery/thriller/crime.
- project target window: 2016-09-01 to 2017-12-31.

After current preprocessing:
- keep `is_read=True` and `rating>0` only.
- valid interactions: `552,092` rows, `25,971` users, `15,766` books.
- target project window contains `537,958` valid interactions, 97.44% of valid rows.
- row builder creates `526,121` next-item samples.
- split is global chronological 80/10/10 by target timestamp:
  - train `420,896`
  - valid `52,612`
  - test `52,613`
- processed items: `15,766`.
- item text source:
  - description: `14,878`
  - review fallback: `596`
  - none placeholder: `292`
- Qwen3 embedding artifact: `data/processed/rqvae/goodreads.emb-bge.npy`, shape `(15766, 2560)`.
- embedding sanity from routine: no NaN/Inf; norm mean about 94.34; unique embeddings `15,279 / 15,766`.

Important caveats:
- Raw k-core=30 does not survive filtering to read+rated interactions. Valid user/item degree can be below 30.
- Raw timestamps include very old/future anomalies; current preprocessing does not filter by target window.
- Split is global chronological, not per-user leave-one-out. Valid/test have target users/books unseen as targets in train.
- RQVAE item embedding is transductive: all valid items are embedded before split, including items that appear only in validation/test targets.
- There are no duplicate valid user-book pairs in current EDA.

## 8. Current Data Construction Details

`data_preprocess.py` does:
1. Build review fallback per `book_id` from full review file.
   - candidate if `rating >= 3` and `review_words >= 50`.
   - choose best by `(rating, n_votes, word_count)`.
2. Collect interactions.
   - keep rows with `is_read` and `rating > 0`.
   - timestamp priority: `read_at`, `date_updated`, `date_added`, `started_at`.
3. Keep only books with valid interactions.
4. Build `item_text`:
   - title.
   - title_without_series if different.
   - author IDs.
   - series IDs.
   - top shelves.
   - description, or selected review fallback, or placeholder.
5. Sort each user history by timestamp.
6. For each index from 1 onward, use previous items as history, cap `history_max_len=100`, target next item.
7. Sort all rows globally by target timestamp and split 8/1/1.

Processed files:
- `train.csv`, `valid.csv`, `test.csv`
- `goodreads.item.json`
- `goodreads.item2id.json`
- `stats.json`
- later: `goodreads.emb-bge.npy`, `goodreads.index.json`

Current caveat: existing CSV headers use older names such as `history_item_id` and `history_item_title`, while current `data_preprocess.py` writes `history_item_ids` and `history_titles`. If SFT loading fails, first reconcile the processed CSV schema with `sft_dataset.py` before changing model code.

## 9. Reproduction Scope In This Repo

Reasonable staged reproduction:

Stage A: data + EDA
- `make data` should run preprocessing, but current `uv.lock` may be broken.
- `make eda` writes `notes/eda.md` and `notes/eda_metrics.json`.
- EDA is already strong enough for interview discussion.

Stage B: SID tokenizer
- Run text embedding with Qwen3-Embedding-4B.
- Train RQ-VAE on item embeddings.
- Generate SID index.
- Check collisions and codebook usage.
- For interview, emphasize how SID code hierarchy compresses item identity while preserving text semantics.

Stage C: SFT
- Use Qwen2.5/Qwen family causal LM.
- Add SID tokens.
- Train on `seq_sid_to_sid`, `sid_to_title`, `title_to_sid`, `title_history_to_sid`.
- On 24GB GPU, prefer new-token-only for smoke tests and QLoRA for real runs.
- Track token length truncation because title histories can be long; `max_length=1024` may truncate long histories.

Stage D: ranking evaluation
- Implement constrained generation over SID token trie.
- Compute HR@3/5/10 and NDCG@3/5/10.
- Compare seq SID prompt vs title-history prompt.

Stage E: RL, optional
- Add GRPO only after constrained decoding and ranking eval are correct.
- Start with rule reward and beam search; add rank reward later.

## 10. Interview Talking Points

Core story:
- This is a public-data reproduction of MiniOneRec on GoodReads mystery/thriller/crime, not Amazon Office/Industrial.
- We use the same generative-rec paradigm: item text -> embedding -> RQ-VAE SID -> LLM generates next SID.
- The project adapts the paper by using GoodReads metadata/reviews and a smaller compute budget.

Why SID:
- SID compresses item IDs into short discrete token sequences, enabling LLM-style generation and scaling.
- Compared with title generation, SID output is shorter, closed-vocabulary, easier to constrain, and cheaper at inference.
- Compared with arbitrary IDs, RQ-VAE SIDs preserve semantic proximity from text embeddings.

Why alignment:
- SID tokens are newly added symbols; pretrained LLM has no prior meaning for them.
- Title<->SID and title-history->SID tasks connect language knowledge to SID space.
- Paper ablation says no alignment is worst; full-process alignment is best.

Why review fallback matters here:
- GoodReads descriptions are missing/short for a nontrivial tail.
- 596 valid items use review fallback; 292 still have no usable text. This directly affects SID quality.

Evaluation protocol honesty:
- Current split is global chronological, not leave-one-out.
- There are unseen target users/books in valid/test relative to train targets.
- Current `eval.py` is loss-only, not HR/NDCG; ranking eval is still needed for paper-level reproduction.

Compute honesty:
- Paper uses H100-scale SFT/RL; this repo has a 24GB 3090/4090 constraint.
- Practical recipe: small Qwen, QLoRA, shorter batch, gradient checkpointing, maybe skip RL until SFT/eval are solid.

## 11. Differences From Original Paper

Original paper:
- Amazon Review Office/Industrial.
- 5-core-ish trimming and history max 10 in appendix.
- Qwen3-Embedding-4B for item text embeddings.
- RQ-VAE with 3x256 codes.
- Qwen2.5-Instruct backbones 0.5B to 7B.
- SFT + GRPO.
- full-process SID alignment.
- constrained beam search and hybrid reward.

This repo currently:
- GoodReads mystery/thriller/crime, raw subset claims k-core=30 before filtering.
- history max 100.
- global chronological 8/1/1 split.
- item text includes title, title_without_series, author IDs, series IDs, shelves, description/review fallback.
- text embedding artifact uses Qwen3-Embedding-4B and dim 2560.
- RQ-VAE and SID generation scripts exist.
- SFT subset exists; RL and ranking eval not yet implemented.
