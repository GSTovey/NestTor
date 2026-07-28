# NESTTOR: Detecting Tor-over-Proxy Traffic with Factorized Decisions

[English](README.md)

NestTor 用于判断一条加密代理 flow 内部是否承载 Tor。模型不使用地址、端口、域名、文件名、绝对时间戳、载荷内容或高基数标识，覆盖 Shadowsocks、Trojan、VMess 和 VLESS 四种外层代理。

![NESTTOR 主框架图](main_framework.png)

本仓库仅供科研使用。

## 方法

Tor-over-Proxy 同时要求两个条件成立：存在外层代理，并且该代理内部存在 Tor。NestTor 分别建模这两个条件：

- 代理门控器 \(Q\) 判断是否存在外层代理证据；
- 内层 Tor 检测器 \(T\) 判断代理内部是否存在 Tor 证据；
- 最终结果由两个阈值判断的硬合取给出。

每条双向 TCP flow 使用包方向、有效载荷长度、相对包间隔、突发行为以及早期、中期和晚期的阶段变化进行表示。主阶段动态表示由 62 维流行为特征和 37 维阶段动态特征组成。代理门控器和内层检测器是独立训练的 HGB 模型，不共享参数、编码器或联合损失。

协议残留表示只作为内层检测器的替代输入，不与主表示融合。对于未见代理，标签访问校准只允许使用与评价分组互斥的纯代理参考集调整内层阈值，不访问目标 ToP 标签，不重新训练模型，也不建立流量对。

## 仓库内容

```text
data/
  manifest.csv
  past_features.csv
  sequences.npz
  dataset_metadata.json
code/
  common.py
  build_pilot_dataset.py
  run_pilot.py
  protocol_residual.py
  run_protocol_residual_audit.py
  run_residual_hgb_probe.py
  compare_role_separated_systems.py
main_framework.png
```

公开代码包含当前核心数据构建、表示提取、协议残留审计、职责分离 HGB 流程和系统比较入口。历史实验、修订记录、原始 PCAP 和模型 checkpoint 不在本仓库中。

## 安装

建议使用 Python 3.10 或更高版本。

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

从 PCAP 构建样本时还需要安装 `tshark`。

## 运行核心流程

完整复现需要论文定义的固定 5,000-flow 开发数据。数据目录必须包含 `manifest.csv`、`past_features.csv`、`sequences.npz` 和 `dataset_metadata.json`。

```bash
python3 code/run_protocol_residual_audit.py \
  --data-dir /path/to/development-data \
  --output-dir /path/to/residual-audit

python3 code/run_residual_hgb_probe.py \
  --data-dir /path/to/development-data \
  --output-dir /path/to/nesttor-output
```

两个实验入口都会拒绝覆盖已经存在的输出目录。公开 `data/` 目录提供的是 650-flow 处理后兼容性样本，只用于检查数据格式。它不是开发集或确认集，也不包含纯代理负类，不能用于复现论文主结果。

只检查代码是否可解析时，可运行：

```bash
python3 -m py_compile code/*.py
```

## 数据

当前论文中的 NestTor Benchmark 包含 349,891 条唯一双向 TCP flow、77,826 个 PCAP 和 1,663 个可核验域名，覆盖四种代理协议。实验清单在域名和抓包层面保持分组互斥。

本仓库不分发原始 PCAP。科研需要获取数据时，可联系 `tovey_zhou@foxmail.com`。

公开样本格式见 [data/README_zh.md](data/README_zh.md)。

## 结果与边界

当前论文报告的留出确认结果为 99.45% macro-F1。在相同数据权限和划分下的原生端到端比较中，NestTor 相对次优复现方法高 11.99 个百分点。

这些结果不代表模型在所有条件下都具有稳定鲁棒性。论文分别评价了未见代理、跨采集变化和包级变换，并报告了分布偏移下的敏感性。因此，结果表述必须限定在对应清单、标签访问规则和评价条件内。

## 引用

如果本仓库对你的研究有帮助，请引用 NestTor 论文。正式书目信息将在论文公开后补充。

## 许可证

许可证信息将在公开版本定稿后补充。
