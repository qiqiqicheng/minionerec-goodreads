这是一个基于 MiniOneRec 生成式推荐范式的推荐系统项目，使用 GoodReads 数据集中的 mystery thriller crime 类别的从 2016 年 9 月到 2017 年 12 月的满足 k-core=30 的数据
我希望复现这篇文章的完整工作，目标是针对推荐算法 / LLM 岗位的面试拷打，同时夯实一些和 SID，rqvae 等推荐系统前沿的内容

原始论文见 notes/paper.md

如果需要编写代码，请注意：
1. 使用 uv 管理环境，使用 Makefile 来管理命令
2. 使用 lightning + hydra 库搭建
3. 不要写防御式编程，有问题提前 raise
4. 不要过度包装，尽量简洁易读
5. 不要冗长注释，使用英文数据，且写清楚 tensor ，如 `a = get_embedding()  # [B, T, D]`
6. 注意节约计算资源，我只能使用 24576MiB 的 NVIDIA GeForce RTX 3090 / 4090 显卡
7. apply_patch 在这个环境里不存在，请改用文件编辑工具

如果需要处理报错，请注意：
1. 从原始数据出发
2. 检查中间计算结果，即 data/processed 中的内容

如果需要看数据，请注意不要全量读入，内容都很大，防止污染上下文

notes/routine.md 中是我进行项目过程中的随手记录
