# NestTor processed samples

此目录只存放已经完成预处理的 flow samples，不存放原始 PCAP、抓包目录或模型 checkpoint。

标准样本由以下文件组成：

```text
manifest.csv
past_features.csv
sequences.npz
```

这些文件通过 `sample_id` 对齐。模型输入不得包含地址、端口、域名、文件名、时间戳或其他高基数身份字段。
