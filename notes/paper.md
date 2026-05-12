# MiniOneRec: An Open-Source Framework for Scaling Generative Recommendation

## ABSTRACT

The recent success of large language models (LLMs) has renewed interest in whether recommender systems can achieve similar scaling benefits. Conventional recommenders, dominated by massive embedding tables, tend to plateau as embedding dimensions grow. In contrast, the emerging generative paradigm replaces embeddings with compact Semantic ID (SID) sequences produced by autoregressive Transformers. Yet most industrial deployments remain proprietary, leaving two fundamental questions open: (1) Do the expected scaling laws hold on public benchmarks? (2) What is the minimal post-training recipe that enables competitive performance?

We present MiniOneRec, to the best of our knowledge, the first fully open-source generative recommendation framework, which provides an end-to-end workflow spanning SID construction, supervised fine-tuning, and recommendation-oriented reinforcement learning. We generate SIDs via a Residual Quantized VAE and post-train Qwen backbones ranging from 0.5B to 7B parameters on the Amazon Review dataset. Our experiments reveal a consistent downward trend in both training and evaluation losses with increasing model size, validating the parameter efficiency of the generative approach. To further enhance performance, we propose a lightweight yet effective post-training pipeline that (1) enforces full-process SID alignment and (2) applies reinforcement learning with constrained decoding and hybrid rewards. Together, these techniques yield significant improvements in both ranking accuracy and candidate diversity.

<div style="text-align: center;"><img src="imgs/img_in_chart_box_216_1047_581_1326.jpg" alt="Image" width="29%" /></div>

<div style="text-align: center;"><img src="imgs/img_in_chart_box_636_1054_1002_1325.jpg" alt="Image" width="29%" /></div>

<div style="text-align: center;">Figure 1: Left: Scaling curves from 0.5B to 7B parameters. Right: Effect of world knowledge on model performance: MiniOneRec-W/O ALIGN uses pretrained LLM weights but omits SID-text alignment, while MiniOneRec-Scratch is trained from random initialization and omits alignment.</div>


---

## MINIONEREC

## CONTENTS

1 Introduction 3  
2 Background and Related Work 4  
2.1 Generative Recommendation 4  
2.2 LLM and RL 4  
3 Modeling 5  
3.1 Task formulation 5  
3.2 Item Tokenization 5  
3.3 Alignment with LLMs 6  
3.4 Reinforced Preference Optimization 6  
3.4.1 Sampling Strategy 7  
3.4.2 Reward Design 7  
4 Training 8  
5 Evaluation 8  
5.1 Evaluation Setup 9  
5.2 Scaling 9  
5.3 Performance Comparison 9  
5.4 Transferability 10  
5.5 Ablation Study 10  
5.5.1 Aligning Strategy 10  
5.5.2 Sampling Strategy 11  
5.5.3 Reward Design 11  
5.6 Pre-trained LLM Impact 12  
6 Conclusion 12  
A Alignment Prompts 18  
A.1 Overview 18  
A.2 Recommendation Tasks 18  
A.3 Alignment Tasks 18  
B Datasets 19

---

## 1 INTRODUCTION

The scaling behavior of recommendation models has recently attracted significant attention (Deng et al., 2025; Wang et al., 2025; Zhai et al., 2024; Zhang et al., 2024; Jiang et al., 2025; Zhu et al., 2025; Chen et al., 2024a; Dai et al., 2025), driven largely by the success of large language models (LLMs) in demonstrating predictable performance gains with increased model size (Brown et al., 2020; Achiam et al., 2023; DeepSeek-AI et al., 2024; Yang et al., 2024; Jiang et al., 2023; Dubey et al., 2024; Touvron et al., 2023; DeepSeek-AI et al., 2025). However, traditional recommendation models have struggled to exhibit similar scaling laws (Kang and McAuley, 2018; Hidasi et al., 2016; Tang and Wang, 2018; Wu et al., 2021; Fang et al., 2020). These systems typically allocate the majority of parameters to large embedding tables for storing user and item representations, while using only inner-product or shallow scoring networks for final predictions. Such an embedding-heavy design leads to performance plateaus: even as embedding dimensions or table sizes increase, improvements diminish quickly beyond moderate scales (Zhou et al., 2018; Covington et al., 2016; Wang et al., 2021; Guo et al., 2017).

Generative recommendation offers a fundamental paradigm shift. By compressing items into sequences of discrete Semantic IDs (SIDs) through quantization techniques (Luo et al., 2024; Zeghidour et al., 2022), it uses compact vocabularies and redirects the bulk of parameters toward deep autoregressive Transformers that generate SID sequences (Rajput et al., 2023; Zheng et al., 2024; Qu et al., 2024; Deng et al., 2025; Dai et al., 2025). This design enables scaling behaviors more characteristic of language models, where increased depth and capacity translate to consistent performance gains (Deng et al., 2025).

Recent industrial deployments have demonstrated the promise of this generative paradigm. OneRec (Deng et al., 2025; Zhou et al., 2025a) achieves significant improvements on Kuaishou's platform with 400 million daily active users, while OneRec-V2 (Zhou et al., 2025b) further advances through lazy decoder architecture and preference alignment. OnePiece (Dai et al., 2025) integrates LLM-style context engineering and reasoning into both retrieval and ranking models of industrial cascaded pipelines. However, these results rely on massive proprietary datasets and remain closed-source, leaving critical questions unanswered for the research community:

• Do the scaling advantages of generative recommendation transfer to public datasets?

• What is the minimal post-training recipe needed to achieve strong performance?

In this work, we present MiniOneRec, the first fully open-source framework for generative recommendation, which provides an end-to-end workflow encompassing SID generation, model supervised fine-tuning (SFT), and recommendation-driven reinforcement learning (RL). The release includes complete source code, reproducible training pipelines, and publicly available model checkpoints. We implement SID construction using residual quantized variational autoencoder (RQ-VAE) (Zeghidour et al., 2022), in line with prior work such as TIGER (Rajput et al., 2023; Deng et al., 2025). We post-train Qwen-based (backbone) generative models at multiple scales, ranging from 0.5B to 7B parameters, on public benchmarks from the Amazon Review Dataset. Following OneRec's successful two-stage post-training pipeline, we adopt SFT on user-item interaction sequences, followed by group relative policy gradient (GRPO) (DeepSeek-AI et al., 2025; Shao et al., 2024) for alignment. The main contributions of our work are summarized as follows:

• Validating the Scaling Laws of Generative Recommendation: We present the first systematic investigation of how generative recommendation models scale on public data, to the best of our knowledge. Specifically, we train four MiniOneRec variants, spanning 0.5 B to 7 B parameters, on the Amazon Review corpus (Hou et al., 2024) with identical data splits and hyper-parameter settings. Both the final training loss and the held-out evaluation loss consistently decrease as model size grows (see Figures 1 Left and 3), highlighting the superior parameter efficiency of the generative paradigm.

• Optimizing Post-training Strategies for Generative Recommendation: We design a lightweight yet comprehensive post-training pipeline that tightly couples full-process SID alignment with reinforced preference optimization. First, we verify the influence of world knowledge on the model's generative recommendation performance (see Figure 1 Right) and augment the vocabulary with dedicated SID tokens and enforce auxiliary alignment objectives throughout the two-stage optimization process. Second, during the RL phase,

---

we shape the generation process itself: invalid tokens are masked to guarantee that every step produces a legal item, beam search is employed for efficient exploration of diverse candidates, and the reward signal combines rule-based accuracy with a ranking-aware penalty that pushes the model away from hard negatives. As a result, MiniOneRec offers a concise yet effective recipe for bringing RL with value-based ranking into generative recommendation, simultaneously improving candidate diversity and ranking fidelity.

The remainder of this report is organized as follows: Section 2 reviews background on generative recommendation and the application of LLMs with RL in recommender systems. Section 3 presents our proposed MiniOneRec, including task formulation, item tokenization, and post-training pipeline. Section 4 describes the training details while Section 5 provides comprehensive experimental results. Section 6 concludes with discussions of future directions.

## 2 BACKGROUND AND RELATED WORK

### 2.1 GENERATIVE RECOMMENDATION

In the last few years, formulating recommendation as a sequence generation problem has become a vibrant research direction (Rajput et al., 2023; Hou et al., 2025). Under this view, a recommender is trained to predict the tokens of the next item in an end-to-end manner, typically with a Transformer backbone (Vaswani et al., 2017). Such a design removes the rigid multi-stage "matching-ranking" pipeline in traditional retrieval-based recommender systems (Kang and McAuley, 2018; Hidasi et al., 2016) and therefore pushes the performance upper bound.

Early explorations illustrate two key components: (1) turning an item into a discrete code (SIDs) and (2) letting the model produce the code. TIGER (Rajput et al., 2023) adopts RQ-VAE (Zeghidour et al., 2022) to map textual embeddings of titles and descriptions into SIDs. HSTU (Zhai et al., 2024) introduces a streaming architecture that is friendly to high-cardinality and non-stationary logs. LC-Rec (Zheng et al., 2024) aligns an LLM with the SIDs through multi-task learning so that the model can understand these symbols during generation.

Subsequent work focuses on designing better codes. RecForest (Feng et al., 2022) clusters items via hierarchical k-means and uses the cluster indices as tokens. EAGER (Wang et al., 2024) and TokenRec (Qu et al., 2024) fuse collaborative and semantic evidence directly into the tokenizer.

At industrial scale, generative recommenders have started to replace heavy cascade systems. MTGR (Wang et al., 2025) keeps the original DLRM features, adds user-level compression, and accelerates both training and inference. OneRec (Zhou et al., 2025b) reduces serving cost through a lazy decoder-only layout and stabilizes optimization with an improved RL algorithm. OnePiece (Dai et al., 2025) discovers that performing inference in the latent space can further improve generative recommendation performance.

### 2.2 LLM AND RL

RL aims at maximizing cumulative reward through repeated interaction (Kaelbling et al., 1996). When fine-tuning LLMs, RL with Human Feedback (RLHF) has become a standard recipe; PPO (Schulman et al., 2017) is the most common optimizer but is memory-intensive for billion-scale parameters. To reduce cost, Direct Preference Optimization (DPO) (Rafailov et al., 2023) removes the separate value network and maximizes the log-likelihood gap between preferred and dispreferred outputs. S-DPO (Chen et al., 2024b) adapts this idea to recommendation by treating softmax-based negative sampling as implicit pairwise preference. Nevertheless, preference-based objectives are off-policy and may converge prematurely.

Light-weight on-line methods have therefore been proposed. GRPO (Shao et al., 2024) normalizes rewards within a small batch of roll-outs and replaces the learned reward model by rule-based signals, achieving strong gains in maths and code generation.

---

<div style="text-align: center;"><img src="imgs/img_in_image_box_278_135_938_465.jpg" alt="Image" width="53%" /></div>

<div style="text-align: center;">Figure 2: MiniOneRec framework. RQ-VAE builds the item SID codebook. We then perform SFT to warm up the LLM and obtain an initial alignment. In RL, beam search with constrained decoding, thereby the model sequentially produces a ranked list of distinct, valid SIDs. GRPO updates the policy, and SID alignment is enforced end-to-end. This alignment objective is preserved throughout both the SFT and RL stages, fostering deeper semantic understanding.</div>

## 3 MODELING

In this section, we present the modeling strategies of MiniOneRec, as illustrated in Figure 2. MiniOneRec first converts the textual information into SIDs with the RQ-VAE tokenizer (Section 3.2). For better incorporating the huge world knowledge within LLMs (Liao et al., 2023), MiniOneRec further introduces the alignment with LLMs as one expansion of the original OneRec architecture (Section 3.3). During the training stage, MiniOneRec is optimized with next token prediction first, and then followed by reinforced preference optimization (Section 3.4).

### 3.1 TASK FORMULATION

We first formulate recommendation as a sequence–generation problem. For every user  $u$, the items he or she has interacted with are sorted chronologically to form a sequence  $H_u = [i_1, i_2, \ldots, i_T]$. Each item  $i_t$ in the sequence is encoded by a three-level structural ID,  $\{c_0^i, c_1^i, c_2^i\}$. Such structural IDs are typically called SIDs, which preserve hierarchical semantics through quantization techniques with semantic embeddings (Rajput et al., 2023).

A generative policy  $\pi_{\theta}$, implemented as an autoregressive model with parameters  $\theta$, reads the entire history  $H_{u}$ and is trained to predict the next item  $i^{+}$that best matches the taste of user u among all candidates in the catalog. During inference, the model recursively produces item tokens; we keep the k most promising beams by the standard beam-search algorithm and return them as the recommendation list. Model performance is reported with the evaluation measures commonly adopted in generative recommendation.

### 3.2 ITEM TOKENIZATION

In SID-style generative recommenders, the first task is to convert each item into a sequence of discrete tokens. Following the practices of TIGER (Rajput et al., 2023), we employ RQ-VAE (Zeghidour et al., 2022) for this purpose. Concretely, the pipeline is:

1. For every item i, we concatenate its title and textual description to form a single sentence;

2. This sentence is passed through a frozen text encoder (Sheng et al., 2025), producing a $d$-dimensional semantic vector $\mathbf{x} \in \mathbb{R}^d$;

3. Apply RQ-VAE to x. At each level  $l$ ( $0 \leq l < L$) we have a separate codebook  $C_l = \{e_k^{(l)}\}_{k=1}^K$, where  $K$ is the codebook size. We set  $L=3$ and  $K=256$, so each item is

---

represented by three bytes; this choice provides  $2^{24}$ possible codes, which is sufficient for catalogs containing hundreds of millions of products while keeping the vocabulary small. The residual is initialized as  $r_0 = x$ and updated by

 
$$
c_{l}=\arg\min_{k}\left\|\mathbf{r}_{l}-\mathbf{e}_{k}^{(l)}\right\|_{2},\qquad\mathbf{r}_{l+1}=\mathbf{r}_{l}-\mathbf{e}_{c_{l}}^{(l)}.
$$
 

4. Collect the indices  $(c_0, \ldots, c_{L-1})$ as the discrete token sequence for item  $i$; these indices constitute the item tokens consumed by the subsequent generative recommender.

The quantized latent is reconstructed by

 
$$
\mathbf{z}_{\mathrm{q}}=\sum_{l=0}^{L-1}\mathbf{e}_{c_{l}}^{(l)},\qquad\hat{\mathbf{x}}=D(\mathbf{z}_{\mathrm{q}}),
$$
 

where  $D(\cdot)$ is a decoder. The codebooks, the encoder and the decoder are trained jointly, while the text encoder remains frozen. The loss is the sum of a reconstruction term and an RQ regularizer:

 
$$
\mathcal{L}(\mathbf{x})=\underbrace{\left\|\mathbf{x}-\hat{\mathbf{x}}\right\|_{2}^{2}}_{\mathcal{L}_{\mathrm{R E C O}}}+\underbrace{\sum_{l=0}^{L-1}\big(\left\|\mathrm{s g}[\mathbf{r}_{l}]-\mathbf{e}_{c_{l}}^{(l)}\right\|_{2}^{2}+\beta\left\|\mathbf{r}_{l}-\mathrm{s g}[\mathbf{e}_{c_{l}}^{(l)}]\right\|_{2}^{2}\big)}_{\mathcal{L}_{\mathrm{R O}}},
$$
 

with  $sg[\cdot]$ the stop-gradient operator and  $\beta$ a hyper-parameter controlling the commitment term.

To prevent codebook collapse, we follow the warm-start trick in prior work (Zeghidour et al., 2022; Zheng et al., 2024) and initialize each codebook with k-means centroids computed on the first training batch.

### 3.3 ALIGNMENT WITH LLMs

LLMs possess extensive understanding of the world and human behaviors (Sheng et al., 2025), which can serve as a supplement to the collaborative signals in recommender systems (Ren et al., 2024). Existing work has shown that linking an LLM's world knowledge to SID representations noticeably strengthens generative recommendation (Liao et al., 2023; Zheng et al., 2024). Therefore, rather than training on SIDs alone as in prior work (Rajput et al., 2023; Deng et al., 2025; Wang et al., 2025), we introduce several alignment objectives that tie the language space to SID signals. Two major groups of tasks are employed:

• Recommendation Tasks: The LLM receives a time-ordered history together with a clear instruction and is asked to predict the SID of the next item the user might engage with.

• Alignment Tasks: A collection of bridging tasks enforces a two-way mapping between natural language and SID space, grounding the discrete codes in text while injecting linguistic knowledge into their embeddings.

Tasks from both groups are optimized jointly throughout the SFT stage and the subsequent RL stage. During RL, we adopt constrained decoding so that the model can only produce tokens from a predefined list containing every item's SID and its canonical title. This constraint guarantees valid outputs and enables straightforward, rule-based reward computation. Detailed examples of the prompts are provided in the Appendix A.

### 3.4 REINFORCED PREFERENCE OPTIMIZATION

After SFT, we further polish the policy with GRPO (Shao et al., 2024; DeepSeek-AI et al., 2024). GRPO differs from classic RLHF in that it draws multiple candidates per prompt and normalizes rewards within the group, which reduces gradient variance.

Steps: (1) For every prompt  $x \sim D$, the frozen policy  $\pi_{\theta_{\text{old}}}$ is rolled out G times, yielding  $\mathcal{Y}(x) = \{y^{(1)}, \ldots, y^{(G)}\}$. (2) Each candidate  $y^{(i)}$ is assigned a scalar score  $S_i$. (3) Advantages are standardized inside the group:

 
$$
\hat{A}_{i}=\frac{S_{i}-\mu_{1:G}}{\sigma_{1:G}},
$$
 

---

where  $\mu_{1:G}$ and  $\sigma_{1:G}$ denote the mean and standard deviation of the G rewards.

The surrogate objective becomes

 
$$
J_{\mathrm{G R P O}}(\theta)=\mathbb{E}_{x,y^{(i)}}\bigg[\frac{1}{G}\sum_{i=1}^{G}\frac{1}{\left|y^{(i)}\right|}\sum_{t=1}^{\left|y^{(i)}\right|}\Big(\min\big(w_{i,t}\hat{A}_{i,t},\mathrm{clip}(w_{i,t},1-\epsilon,1+\epsilon)\hat{A}_{i,t}\big)-\beta\mathrm{K L}\big[\pi_{\theta}\|\pi_{\mathrm{r e f}}\big)\Big)\bigg],
$$
 

with token-level importance ratio  $w_{i,t} = \frac{\pi_{\theta}(y_{t}^{(i)} | x, y_{<t}^{(i)})}{\pi_{\theta_{\text{old}}}(y_{t}^{(i)} | x, y_{<t}^{(i)})}$. The parameter  $\epsilon$ controls clipping, while  $\beta$ keeps the updated policy near a reference model through a KL term.

Applying RL with verifiable rewards (RLVR) to recommendation brings two obstacles:

• Unique generation space. The action space is a closed set of item SIDs, orders of magnitude smaller than natural-language vocabularies. Re-sampling tends to produce duplicates, wasting computation. We therefore mix dynamic sampling (Yu et al., 2025) with constrained beam search to enlarge coverage while keeping outputs valid.

• Sparse ranking supervision. A hard binary reward (1 for the correct item, 0 otherwise) offers little guidance on ranking quality. We introduce an auxiliary ranking-shaped reward to penalize harder negatives with lower scores. Additionally, dense signals such as semantic similarity and collaborative scores are explored to furnish richer supervision.

#### 3.4.1 SAMPLING STRATEGY

A practical obstacle when porting RLVR to recommendation is the poor sampling diversity brought by the limited action space: querying the policy multiple times with the same prompt often returns identical items, so the model observes few distinct negatives. We measure diversity via

 
$$
\mathrm{D i v}\big(\{e_{k}\}_{k=1}^{G}\big)=\frac{\big|\mathrm{U n i q u e}\big(\{e_{k}\}_{1}^{G}\big)\big|}{G},
$$
 

where the numerator counts unique items among the G generations. Higher values indicate richer supervision.

Two complementary remedies are investigated:

• Dynamic Sampling (Yu et al., 2025). We first over-sample, then pick a subset that (i) must include the ground-truth item and (ii) maximizes internal diversity. Although helpful, this demands extra forward passes and still deteriorates as training progresses.

• Beam Search. We ultimately switch to beam search without length normalization (Bao et al., 2024; Tan et al., 2025). By construction, all beams differ, so the method guarantees zero duplication within each group and yields better diversity–efficiency trade-offs.

Based on our findings, MiniOneRec ultimately employs constrained beam search as its default sampler, ensuring that every generated item is valid while still providing a diverse set of candidate trajectories.

#### 3.4.2 REWARD DESIGN

Recommendation models are usually judged with ranking measures such as NDCG. In contrast, the standard GRPO setup supplies a binary reward, giving a value of 1 to the true item and 0 to every other candidate. This strategy treats all negatives as equally harmful. Earlier studies (Wu et al., 2021; Chen et al., 2024b) have shown that focusing on hard negatives produces stronger rankers. Motivated by these results, we introduce a rank-aware reward that assigns different penalties based on how prominently a negative item appears in the model's own ranking (Tan et al., 2025).

Given a negative candidate  $e_k$ whose generation probability ranks  $\rho_k$ (with  $\rho = 1$ the most probable), we set

 
$$
\tilde{R}_{\mathrm{r a n k}}(e_{k},e_{t})=\begin{cases}0,&e_{k}=e_{t},\\-\frac{1}{\log(\rho_{k}+1)},&otherwise,\end{cases}\quad R_{\mathrm{r a n k}}(e_{k},e_{t})=-\frac{\tilde{R}_{\mathrm{r a n k}}(e_{k},e_{t})}{\sum_{j=1}^{G}\tilde{R}_{\mathrm{r a n k}}(e_{j},e_{t})}.
$$
 

---

Thus, negatives that the model was very confident about (low  $\rho_{k}$) receive stronger penalties. The final reward combines rule-based and ranking components:

 
$$
R(e_{k},e_{t})=R_{r u l e}(e_{k},e_{t})+R_{r a n k}(e_{k},e_{t}),
$$
 

where the rule-based term is

 
$$
R_{\mathrm{r u l e}}(e_{k},e_{t})=\begin{cases}1,&e_{k}=e_{t},\\ 0,&\mathrm{o t h e r w i s e}.\end{cases}
$$
 

The mixed reward discussed combines binary correctness with a soft ranking term, helping the LLM tell hard negatives apart. Yet recommendation data contain additional, under-utilized cues. To see whether such information can enhance or even replace the rule-based component in RLVR, we experiment with the following collaborative reward. Specifically, for every item suggested by the policy, we obtain its logit from a pre-trained collaborative-filtering model and pass that score back as reward, thereby injecting knowledge extracted from historical user-item interactions. MiniOneRec ultimately adopts the combined ranking-and-rule reward as its default choice.

## 4 TRAINING

MiniOneRec first converts item text into discrete codes. Titles and descriptions are embedded by the Qwen3-Embedding-4B encoder, after which RQ-VAE performs residual quantization. This tokenizer is trained on a single GPU with a batch size of 20480, a learning rate of  $1 \times 10^{-3}$, and 10,000 training epochs. The resulting SID vocabulary is plugged into a Qwen2.5-Instruct backbone. SFT takes place on eight NVIDIA H100, each holding 128 samples. Training lasts up to ten epochs with early stopping (patience one epoch). The initial learning rate is  $3 \times 10^{-4}$ and follows a cosine decay schedule. Starting from the SFT checkpoint, we apply GRPO for two additional epochs while keeping the KL weight  $\beta$ unchanged. During roll-out, beam search with width 16 is adopted, so every input generates sixteen distinct candidate sequences.

In the performance comparison with existing recommenders, conventional recommenders are trained with binary cross-entropy and the Adam optimizer; learning rates are chosen from {1e-2, 1e-3, 1e-4}, and the weight-decay term is scanned over {1e-2, 1e-3, 1e-4, 1e-5, 1e-6}. The batch size is 1024. For TIGER, we rely on a T5 encoder–decoder and reuse Qwen3-Embedding-4B for item embeddings. All LLM-powered systems, including ours, share the Qwen2.5-Instruct backbone and use AdamW. Batches contain 128 examples for SFT and preference alignment, and 512 samples for RL. The learning rate is  $3 \times 10^{-4}$ during SFT, while S-DPO and RL use  $1 \times 10^{-5}$. S-DPO runs for one epoch with  $\beta = 0.1$ and samples three negative items;  $D^{3}$ tries interpolation factors  $\alpha$ of 0.8, 0.9, and 1.0.

## 5 EVALUATION

This section details our empirical study. We begin with a scaling investigation on two real-world benchmarks (Hou et al., 2024), examining how MiniOneRec's loss curves change as model size grows. We then benchmark MiniOneRec against three groups of baselines: traditional sequential recommenders, SID-based generators, and recent LLM-driven systems. To evaluate the model's capability for cross-domain recommendation generalization, we frame SID next-item prediction as a task of recommendation rule discovery and evaluate the model on domains that never appeared during training. Comprehensive ablation tests follow, which help us locate the parts of the framework that contribute most to the final score. Finally, we look into how the broad knowledge stored in pre-trained LLM weights affects generative recommendation. In summary, we study the following questions within this section:

• How does the loss of MiniOneRec scale with different model size?

• How does MiniOneRec perform in comparison to other baseline methods?

• How does MiniOneRec perform under completely unseen domains?

• How do the designed components contribute to MiniOneRec’s recommendation efficiency?

• How do the pre-trained weights of the LLM influence the performance of generative recommendation?

---
**Table 1: Performance of MiniOneRec Compared to Traditional Methods, Generative Methods, and LLM-based Methods**

| Dataset | Category | Methods | HR@3 | NDCG@3 | HR@5 | NDCG@5 | HR@10 | NDCG@10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Industrial** | Traditional | GRU4Rec | 0.0638 | 0.0542 | 0.0774 | 0.0598 | 0.0999 | 0.0669 |
|  |  | Caser | 0.0618 | 0.0514 | 0.0717 | 0.0555 | 0.0942 | 0.0628 |
|  |  | SASRec | 0.0790 | 0.0700 | 0.0909 | 0.0748 | 0.1088 | 0.0806 |
|  | Generative | HSTU | 0.0927 | 0.0885 | 0.1037 | 0.0918 | 0.1163 | 0.0958 |
|  |  | TIGER | 0.0852 | 0.0742 | 0.1010 | 0.0807 | 0.1321 | 0.0908 |
|  |  | LCRec | 0.0915 | 0.0805 | 0.1057 | 0.0862 | 0.1332 | 0.0952 |
|  | LLM-based | BIGRec | 0.0931 | 0.0841 | 0.1092 | 0.0907 | 0.1370 | 0.0997 |
|  |  | D³ | 0.1024 | 0.0991 | 0.1213 | 0.0989 | 0.1500 | 0.1082 |
|  |  | S-DPO | 0.1032 | 0.0906 | 0.1238 | 0.0991 | 0.1524 | 0.1082 |
|  | Ours | MiniOneRec | 0.1143 | 0.1011 | 0.1321 | 0.1084 | 0.1586 | 0.1167 |
| **Office** | Traditional | GRU4Rec | 0.0629 | 0.0528 | 0.0789 | 0.0595 | 0.1019 | 0.0669 |
|  |  | Caser | 0.0748 | 0.0615 | 0.0865 | 0.0664 | 0.1093 | 0.0737 |
|  |  | SASRec | 0.0861 | 0.0769 | 0.0949 | 0.0805 | 0.1120 | 0.0858 |
|  | Generative | HSTU | 0.1134 | 0.1031 | 0.1252 | 0.1079 | 0.1400 | 0.1126 |
|  |  | TIGER | 0.0986 | 0.0852 | 0.1163 | 0.0960 | 0.1408 | 0.1002 |
|  |  | LCRec | 0.0921 | 0.0807 | 0.1048 | 0.0859 | 0.1237 | 0.0920 |
|  | LLM-based | BIGRec | 0.1069 | 0.0961 | 0.1204 | 0.1017 | 0.1434 | 0.1091 |
|  |  | D³ | 0.1204 | 0.1055 | 0.1406 | 0.1139 | 0.1634 | 0.1213 |
|  |  | S-DPO | 0.1169 | 0.1033 | 0.1356 | 0.1110 | 0.1587 | 0.1255 |
|  | Ours | MiniOneRec | 0.1217 | 0.1088 | 0.1420 | 0.1172 | 0.1634 | 0.1242 |

### 5.1 EVALUATION SETUP

Experiments are carried out on two real-world slices of the Amazon Review dataset (Hou et al., 2024), namely Office and Industrial. To measure top-K recommendation accuracy, we follow standard practice and compute Hit Rate (HR@K) as well as Normalized Discounted Cumulative Gain (NDCG@K).

### 5.2 SCALING

We demonstrate the scaling capabilities of MiniOneRec, where the convergence loss decrease consistently as the model size increases. As Figure 1 shows, under the generative recommendation paradigm, there is a clear correlation between model size and loss. As the LLM scale increases, the convergence loss continues to decrease. Furthermore, Figure 3 tracks the evaluation loss on the SFT training set, recorded every 0.5 epoch. It is evident that models with larger parameter counts maintain lower evaluation losses throughout nearly the entire training process and converge more rapidly. Such a superior scaling effect demonstrates the potential of generative recommenders as the next-generation recommendation models.

### 5.3 PERFORMANCE COMPARISON

Our baselines contain three categories: (1) Traditional recommendation models, including GRU4Rec (Hidasi et al., 2016), Caser (Tang and Wang, 2018), SASRec (Kang and McAuley, 2018); (2)

---

Generative recommendation models: HSTU (Zhai et al., 2024), TIGER (Rajput et al., 2023), LC-Rec (Zheng et al., 2024); (3) LLM-based recommendation models, including BigRec (Bao et al., 2023), D $^{3}$ (Bao et al., 2024), S-DPO (Chen et al., 2024b).

We evaluate MINIONEREC on the Industrial and Office benchmarks and summarize the outcomes in Table 1. Two key insights stand out:

• Utility of LLM World Knowledge. Recommenders powered by LLMs, such as BIGRec and  $D^{3}$, clearly surpass traditional systems like GRU4Rec and Caser, showing that the broad knowledge embedded in LLMs translates into better recommendation accuracy.

• Effectiveness of MiniOneRec. MiniOneRec goes a step further: by aligning the full generation process with the task objective and using reinforced preference optimization during RL, it consistently outperforms previous generative solutions across most reported metrics. Moreover, by operating in the compact SID space rather than on verbose textual titles, MiniOneRec requires substantially fewer context tokens and yields faster inference, translating into lower latency and smaller memory footprints at serving time.

### 5.4 TRANSFERABILITY

We evaluate the out-of-distribution (OOD) robustness of MiniOneRec through an experiment we call SID pattern discovery. The model is trained exclusively on the Industrial domain and then deployed, without any further tuning, to the never-seen Office domain. Prior work (Jin et al., 2025; Yue et al., 2025; Yoshihara et al., 2025; Cheng et al., 2025) suggests that SFT may overfit to the source domain and hurt transfer, so we add an RL-only variant, MiniOneRec-w/ RL-OOD, that skips SFT entirely to emphasize generalization.

The comparison involves four systems: (1) GRU4Rec, trained and tested inside the Office domain; (2) Qwen-Text, which represents user histories as plain sentences and predicts the next item by its title with no additional fine-tuning; (3) Qwen-SID, which encodes the same history with SID tokens and produces the next SID with no additional fine-tuning; (4) MiniOneRec-w/ RL-OOD, trained via GRPO on Industrial only and evaluated on Office, to evaluate its OOD (out-of-distribution) performance.

Table 2 reports the outcomes. Qwen-Text performs poorly, while Qwen-SID does noticeably better, indicating that a structured SID vocabulary

<div style="text-align: center;"><img src="imgs/img_in_chart_box_627_713_1005_940.jpg" alt="Image" width="30%" /></div>

is easier for an LLM to exploit. Although MiniOneRec-w/ RL-OOD falls short of the full MiniOneRec on in-domain scores, its reinforcement-only training grants excellent transfer, achieving competitive accuracy on the unseen catalog. Despite the substantial domain shift and possible semantic drift among SIDs, MiniOneRec successfully uncovers reusable interaction patterns, underscoring the framework's promise for cross-domain recommendation.

### 5.5 ABLATION STUDY

To validate the effectiveness of each component in the MiniOneRec framework, we compare it with the following alternative approaches.

#### 5.5.1 ALIGNING STRATEGY

To isolate the impact of each component, we compare the full MiniOneRec with three pared-down variants. (1) MINIONEREC-W/O ALIGN: removes any language-SID alignment and treats recommendation purely as a SID-to-SID task. (2) MINIONEREC-W/ SFTALIGN: keeps the alignment objective during SFT stage only, while RL uses SID data alone. (3) MINIONEREC-W/ RLALIGN: SFT relies solely on SID supervision, and the alignment tasks are introduced later in the RL stage.

---

**Table 2: Performance of MiniOneRec and its variants on completely unseen recommendation domains**

| Dataset | Method | HR@3 | NDCG@3 | HR@5 | NDCG@5 | HR@10 | NDCG@10 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Office | GRU4Rec | 0.0629 | 0.0528 | 0.0789 | 0.0595 | 0.1019 | 0.0669 |
|  | Qwen-Text | 0.0031 | 0.0021 | 0.0044 | 0.0026 | 0.0057 | 0.0030 |
|  | Qwen-SID | 0.0300 | 0.0214 | 0.0456 | 0.0282 | 0.0733 | 0.0373 |
|  | MiniOneRec-w/ RL-OOD | 0.0553 | 0.0433 | 0.0691 | 0.0489 | 0.0892 | 0.0553 |

<div style="text-align: center;"><img src="imgs/img_in_chart_box_261_289_487_456.jpg" alt="Image" width="18%" /></div>

<div style="text-align: center;">(a) Aligning Strategy.</div>

<div style="text-align: center;"><img src="imgs/img_in_chart_box_499_290_724_456.jpg" alt="Image" width="18%" /></div>

<div style="text-align: center;">(b) Sampling Strategy.</div>

<div style="text-align: center;"><img src="imgs/img_in_chart_box_726_290_951_455.jpg" alt="Image" width="18%" /></div>

<div style="text-align: center;">(c) Reward Design.</div>

<div style="text-align: center;">Figure 4: Study on the effectiveness of MiniOneRec's individual components. Figure 4a examines model performance under different alignment strategies; Figure 4b investigates various sampling strategies; Figure 4c evaluates the impact of alternative reward designs.</div>

Figure 4a summarizes the findings. The complete MiniOneRec, which maintains alignment throughout the whole pipeline, delivers the highest scores on every metric. The MINIONEREC-W/O ALIGN performs worst, indicating that grounding SID generation in world knowledge is essential.

#### 5.5.2 SAMPLING STRATEGY

We study how different roll-out methods affect MiniOneRec by switching only the trajectory generator while keeping everything else fixed: (1) MINIONEREC–COMMON relies on a plain Top-k decoder to produce exactly the required number of paths. (2) MINIONEREC–DYNAMIC follows our two-step sampler: it first draws one and a half times the budget, then retains as many unique items as possible for RL. The full model adopts beam search with width 16.

As plotted in Figure 4b, the complete MiniOneRec delivers the highest accuracy while using roughly two-thirds of the samples needed by the dynamic variant, demonstrating that beam search is the most cost-efficient choice among the tested strategies.

#### 5.5.3 REWARD DESIGN

Three variants are compared: (1) MINIONEREC–W/ ACC that relies solely on a binary correctness signal; (2) MINIONEREC–W/ COLLABORATIVE that replaces the ranking term with logits taken from a frozen SASRec model so as to supply collaborative cues;

As shown in the Figure 4c, the full MiniOneRec achieves the best overall performance. To be noticed, injecting the collaborative reward information into the RL process instead led to significant degradation. We hypothesize that, this stems from reward hacking: as recommendation accuracy declines, the reward continues to increase, revealing a misalignment between this collaborative reward signal and the true objective.

**Table 3: Performance of MiniOneRec initialized with pre-trained weights versus random initialization across two domains**

| Datasets | Methods | HR@3 | NDCG@3 | HR@5 | NDCG@5 | HR@10 | NDCG@10 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Industrial | MiniOneRec-scratch | 0.0757 | 0.0672 | 0.0891 | 0.0726 | 0.1134 | 0.0804 |
| Industrial | MiniOneRec | 0.1125 | 0.0988 | 0.1259 | 0.1046 | 0.1546 | 0.1139 |
| Office | MiniOneRec-scratch | 0.0959 | 0.0855 | 0.1057 | 0.0896 | 0.1196 | 0.0941 |
| Office | MiniOneRec | 0.1217 | 0.1088 | 0.1420 | 0.1172 | 0.1634 | 0.1242 |

---

### 5.6 PRE-TRAINED LLM IMPACT

To isolate the impact of pre-trained LLMs on generative recommendation, we instantiate two variants of MiniOneRec: one initialized from a general-purpose pre-trained LLM and the other trained from scratch with random weights. As shown in Table 3, Experiments on two benchmark datasets reveal a consistent pattern: the model that starts from pre-trained weights significantly outperforms its randomly initialized counterpart. We conjecture that (i) the general reasoning ability acquired during large-scale language pre-training allows the model to cast the next-SID prediction task as a problem of pattern discovery, as discussed in Section 5.4, and (ii) the factual knowledge already encoded in the LLM offers a head start in understanding the real-world semantics behind each SID, part of which can be transferred to the recommendation domain.

## 6 CONCLUSION

This report introduces MiniOneRec, the first fully open-source generative recommendation framework, which to the best of our knowledge provides an end-to-end workflow spanning SID construction, SFT, and recommendation-oriented RL. By systematically validating scaling laws on public benchmarks, we demonstrate that larger generative recommenders achieve lower training and evaluation losses than their smaller counterparts, confirming the parameter-efficiency advantage of the SID-based paradigm over traditional embedding-centric models. Building on this insight, we introduce post-training techniques: (i) full-process SID alignment, which embeds SID tokens into the model vocabulary and imposes auxiliary alignment tasks across both SFT and RL stages, and (ii) reinforced preference optimization, which combines constrained decoding, beam-based sampling, and hybrid reward design. Extensive experiments on Amazon Review show that MiniOneRec consistently surpasses strong sequential, generative, and LLM-based baselines, while maintaining a lean post-training footprint.

Looking forward, we will keep maintaining and extending the MiniOneRec codebase. A public roadmap will guide future developments, and we warmly welcome community contributions. Planned updates include new datasets, more advanced tokenisation schemes, larger backbone models, and enhanced training pipelines, ensuring that MiniOneRec remains a solid reference platform for research and practice in large-scale generative recommendation.

---

## REFERENCES
...

### A ALIGNMENT PROMPTS

### A.1 OVERVIEW

In this section we describe how we align the world knowledge of a LLM with the SID space so that the model can be directly used for generative recommendation.

We append a three-layer codebook, each layer containing 256 unique SIDs, to the original tokenizer vocabulary. The inserted codes are treated as indivisible tokens, enabling the LLM to read or write SID sequences without any sub-word splitting. Then we introduce a set of auxiliary Alignment Tasks to bridge the semantic gap between the textual vocabulary and the newly added SIDs. A specialized Recommendation Tasks further teach the model to predict the SID of the next item given an user history.

### A.2 RECOMMENDATION TASKS

1. Generative Retrieval. The main task of generative recommendation. The LLM receives a chronologically ordered SID sequence that represents the user's recent interactions, together with an explicit instruction such as "Recommend the next item.", and is asked to predict the SID of the next item the user might engage with. A sample prompt is shown in the Figure 5.

<div style="text-align: center;"><img src="imgs/img_in_image_box_294_561_924_692.jpg" alt="Image" width="51%" /></div>

<div style="text-align: center;">Figure 5: Generative Retrieval Prompt.</div>

2. Asymmetric Item Prediction. (a) Given a textual user history, predict the SID of the next item; (b) given a SID-only history, generate the textual title of the next item. Sample prompts are shown in the Figure 6 and Figure 7.

<div style="text-align: center;"><img src="imgs/img_in_image_box_295_827_922_949.jpg" alt="Image" width="51%" /></div>

<div style="text-align: center;">Figure 6: Asymmetric Item Prediction Prompt1.</div>

<div style="text-align: center;"><img src="imgs/img_in_image_box_295_1009_921_1138.jpg" alt="Image" width="51%" /></div>

<div style="text-align: center;">Figure 7: Asymmetric Item Prediction Prompt2.</div>

### A.3 ALIGNMENT TASKS

1. SID-Text Semantic Alignment. (a) Predict an item's textual title from its SID; (b) predict the SID from the item's textual title. Sample prompts are shown in the Figure 11 and Figure 9.

<div style="text-align: center;"><img src="imgs/img_in_image_box_432_1300_784_1394.jpg" alt="Image" width="28%" /></div>

<div style="text-align: center;">Figure 8: SID–Text Semantic Alignment Prompt1.</div>


---

<div style="text-align: center;"><img src="imgs/img_in_image_box_363_116_860_205.jpg" alt="Image" width="40%" /></div>

<div style="text-align: center;">Figure 9: SID–Text Semantic Alignment Prompt2.</div>

2. Item Description Reconstruction. To ground SIDs in richer semantics, we ask the model to generate the item description from a single SID and, conversely, infer the SID from the description. We perform this task only during the SFT stage because the description space is large and diverse. Sample prompts is shown in the Figure 10.

<div style="text-align: center;"><img src="imgs/img_in_image_box_297_362_920_511.jpg" alt="Image" width="50%" /></div>

<div style="text-align: center;">Figure 10: Item Description Reconstruction Prompt.</div>

3. User Preference Summarization. Given a sequence of SIDs, the model produces a short natural–language profile that summarizes the user's interests. Because the raw dataset lacks explicit preference labels, we employ DEEPSEEK (DeepSeek-AI et al., 2024) to extract summaries from the item's meta data and users' textual reviews and use them as pseudo labels. This task, too, is restricted to the SFT stage due to the open-ended output space.

<div style="text-align: center;"><img src="imgs/img_in_image_box_298_695_922_835.jpg" alt="Image" width="50%" /></div>

<div style="text-align: center;">Figure 11: User Preference Summarization Prompt.</div>

### B DATASETS

We evaluate our model on two Amazon Review subsets: Amazon Review subsets (Industrial_and_Scientific and Office_Products). To keep computational costs affordable, we follow the trimming strategy used in (Bao et al., 2024). The main steps are: (1) remove users and items with fewer than five interactions; (2) for Toys_and_Games, keep events from October 2016 to November 2018; (3) for Industrial_and_Scientific, which is smaller, keep all events between October 1996 and November 2018; (4) truncate every user history to at most ten items; (5) finally, split each dataset chronologically into training, validation and test sets with an 8:1:1 ratio. Key statistics of the resulting training splits are reported in Table 4.

<div style="text-align: center;">Table 4: Statistics of datasets.</div>

<table border=1 style='margin: auto; word-wrap: break-word;'><tr><td style='text-align: center; word-wrap: break-word;'>Datasets</td><td style='text-align: center; word-wrap: break-word;'>Inductrial</td><td style='text-align: center; word-wrap: break-word;'>Office</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>Items</td><td style='text-align: center; word-wrap: break-word;'>3,685</td><td style='text-align: center; word-wrap: break-word;'>3,459</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>Train</td><td style='text-align: center; word-wrap: break-word;'>3,6259</td><td style='text-align: center; word-wrap: break-word;'>3,8924</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>Valid</td><td style='text-align: center; word-wrap: break-word;'>4,532</td><td style='text-align: center; word-wrap: break-word;'>4,866</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>Test</td><td style='text-align: center; word-wrap: break-word;'>4,533</td><td style='text-align: center; word-wrap: break-word;'>4,866</td></tr></table>