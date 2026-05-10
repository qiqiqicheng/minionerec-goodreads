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

