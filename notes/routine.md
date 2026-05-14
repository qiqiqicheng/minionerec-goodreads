## 0514
在 autodl 上完成了第一次的 sft 训练，仅训练了 4 epoch，从 evaluate 结果看还是很差，本阶段的结果见 `logs/train_sft/runs/2026-05-13_21-53-27` 和 

export:
```bash
uv run src/minionerec_goodreads/scripts/sft/export_sft.py \
    --checkpoint-path logs/train_sft/runs/2026-05-13_21-53-27/checkpoints/epoch_003.ckpt \
    --output-dir data/ckpts/sft \
    --sid-index-path data/processed/rqvae/goodreads.index.json \ 
    --item-path data/processed/rqvae/goodreads.item.json
```

evaluate:
```bash
CUDA_VISIBLE_DEVICES=3 uv run src/minionerec_goodreads/scripts/sft/eval_sft.py 
    --model-path data/ckpts/sft \
    --split-path data/processed/rqvae/test.csv \
    --max-samples 256
```

感觉还是 epoch 数太少了

暂时先转向 RLHF。。。
下次项目一定不能用这种框架了，写不了一点。。。

## 0513
完成了 rqvae 的完整训练过程（获得 item_text，生成 item embedding，训练 rqvae，挑选合适的 ckpt，最后 generate_sid），已经将最好的 ckpt 保存在了 ckpts/rqvae 下（参考 https://wandb.ai/qiqiqicheng1301-nonplace/minionerec-goodreads/runs/ktlcy9ir?nw=nwuserqiqiqicheng1301）
最优先考虑 collision_rate 尽量小的 ckpt，其实训练到后期码本坍塌的现象几乎没有

获得了 data/processed/rqvae/goodreads.index.json 作为 rqvae 阶段的最终成果

根据原始论文中的数据集选择了更小的数据集， 6000 左右的 item 数量，但是总共有 295,730 条 next_token row，交互数量达到 309454，是原论文 3,6259 的十倍

优化了 train_rqvae 的 infra 内容，每次直接跳过 val 选择直接执行 generate_sid 来获得验证的数据，从中选择最佳的 ckpt，效果不错

优化了 eda 的内容，但是目前没有仔细研读

准备 sft。。。

## 0512

补了 raw EDA，脚本在 `src/minionerec_goodreads/scripts/eda.py`，输出在 `notes/eda.md` 和 `notes/eda_metrics.json`。这次才意识到中期补 EDA 也不算晚，面试里反而能讲清楚一些之前只是“跑通了”的东西：原始 interaction 有 2327295 行，但真正进入序列监督的 read+rating>0 只有 552092 行，目标时间窗 2016-09 到 2017-12 覆盖 537958 行；raw 说是 k-core=30，但过滤掉未读/未评分以后 user/item 侧都不再是 30-core，这个如果面试官追问数据稀疏性，一定要主动说清楚。还有一个坑是 timestamp 全历史里有特别早和未来时间，不能只报 full range，应该明确区分 full-history 和项目窗口。

回看了 `src/minionerec_goodreads/scripts/rqvae/data_preprocess.py`，它做的事情挺直接：流式读 reviews，挑每本书 rating>=3、review_words>=50、votes/评分/长度更高的 review 做 description fallback；流式读 interactions，只保留 is_read 且 rating>0 的行为，并用 read_at/date_updated/date_added/started_at 兜底转 timestamp；再只为出现过有效交互的 book 建 item2id 和 item_text，item_text 由 title、title_without_series、author_id、series、top shelves 和正文拼起来；最后按用户时间排序，用前缀历史预测下一个 item，history_max_len=100，所有样本再按 target timestamp 做全局 8/1/1 split。

整体是适合当前 MiniOneRec/RQVAE 路线的，尤其是 item text fallback 这步很必要：有效 item 里 596 个最后靠 review 文本，292 个还是没法拿到可用文本；不做的话 SID 学到的语义会被空 description 拖累。现在的口径也和后续 SFT 能接上，CSV 里既有 item_id/history_item_id 给 SID 序列，又有 title/history title 给 fusion task。

但这里也有几个我得记住的碎碎念：第一，split 是全局 chronological，不是 per-user leave-one-out，所以验证/测试里会出现 train 没见过的 target user/book，报告里 valid 新 target user=645、test 新 target user=810；如果面试官问 evaluation protocol，不能装作是经典序列推荐 leave-one-out。第二，RQVAE embedding 是对全部有效 item 先做的，这是 transductive item vocabulary，严格冷启动不在当前实验范围内。第三，popular_shelves 很吵，to-read/currently-reading/kindle 这种占比太高，当前只是拼进文本给 PLM/RQVAE 用还可以，如果以后显式建 shelf feature 要先过滤。第四，当前预处理没有对异常时间戳做窗口过滤，所以 train 里会混入一些 1965 之类的早期 target；好处是最大化利用全历史，坏处是和“2016-09 到 2017-12”的项目叙述有一点 tension，后续如果要更严谨可以加 target_start/target_end 控制，并比较是否影响指标。

## 0510
重启本项目，跑通 make embedding，使用 Qwen3-Embedding-4B，bf16，batch size 24，max length 2048，GTX 3090，显存占用 22G，约 20min，1.68it/s
Saved embeddings to data/processed/rqvae/goodreads.emb-bge.npy with shape (15766, 2560)

检查向量结果：
shape: (15766, 2560)
has nan: False
has inf: False
norm min/max/mean: 78.47169 114.91355 94.339226
unique embeddings: 15279 / 15766

已同步到本地

