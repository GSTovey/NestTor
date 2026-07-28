# NestTor 处理后样本

[English](README.md)

本目录提供 650-flow 处理后兼容性样本，用于检查公开数据格式。它不是固定开发集或留出确认集，也不包含纯代理负类。

目录文件如下：

- `manifest.csv`：匿名样本标识、标签、流量分组、代理类型和划分名称；
- `past_features.csv`：99 维流行为与阶段动态数值特征；
- `sequences.npz`：按 `sample_id` 对齐的包序列和突发序列张量；
- `dataset_metadata.json`：该公开样本的来源与用途边界。

`manifest.csv`、`past_features.csv` 和 `sequences.npz` 通过 `sample_id` 逐行对齐。模型输入排除地址、端口、域名、文件名、绝对时间戳、载荷内容和其他高基数标识。

该样本只用于格式检查和代码接入。复现论文结果需要论文定义的固定清单和完整处理后数据。
