# NestTor

**NestTor: Detecting Tor within Encrypted Proxy Tunnels by Separating Proxy and Tor Evidence**

本仓库提供 NestTor 的核心代码、处理后的 flow samples 以及论文中使用的主要实验入口。NestTor 面向客户端到代理链路上的 Tor over Proxy Detection：在不依赖地址、端口、域名、文件名、时间戳和高基数标识的条件下，判断加密代理隧道内部是否运行 Tor。

⚠️ 本仓库仅供科研用途。 ⚠️

## 核心思想

代理隧道会隐藏内层连接，但外层代理行为和内层 Tor 行为并不完全消失。NestTor 将两类证据分开建模：

- **代理门控器（outer proxy gate）**：判断流量是否具有外层代理隧道行为；
- **内层 Tor 检测器（inner Tor detector）**：在代理条件下判断是否存在 Tor 证据；
- **合取决策（conjunctive decision）**：只有两类证据同时成立时，才输出 Tor over Proxy。

整体流程如下：

```text
处理后的 flow sample
        ↓
流行为与阶段动态表示
        ↓
外层代理门控器
        ↓
内层 Tor 检测器（协议残留表示为可选增强）
        ↓
NestTor 合取决策
```

## 代码结构

当前 NestTor 主线代码位于 `experiments/top_binary_pilot/`：

```text
NestTor/
├── data/                                  # 仅放处理后的 flow samples
├── experiments/top_binary_pilot/
│   ├── common.py                          # 共享解析、表示和评价函数
│   ├── build_pilot_dataset.py             # 从输入流记录建立开发清单
│   ├── protocol_residual.py               # 提取协议残留表示
│   ├── run_residual_hgb_probe.py          # NestTor 核心模型开发入口
│   ├── compare_role_separated_systems.py  # 统一组合外层与内层检测器
└── ...                                     # 其他复现实验脚本
```

公开版本只保留当前主线所需的核心脚本和处理后样本；论文修订材料与历史实验不放入本仓库。

## 输入与处理流程

公开代码使用已经完成预处理的 flow samples，不要求用户下载原始 PCAP。常用输入包括：

- `manifest.csv`：样本标签、载体、采集分组和防泄漏元数据；
- `past_features.csv`：流行为和阶段动态数值表示；
- `sequences.npz`：按 `sample_id` 对齐的包序列和突发序列张量。

原始数据处理流程在论文中定义为：

```text
原始 PCAP
  → 双向 TCP flow 提取
  → 删除身份字段和高基数标识
  → 生成流行为、阶段动态和协议残留表示
  → 训练独立代理门控器与内层 Tor 检测器
  → 使用合取规则输出 NestTor 结果
```

模型输入不包含地址、端口、域名、文件名、时间戳或其他高基数身份字段。

## 运行说明

安装 Python 依赖后，将处理后的样本放入 `data/`，并按照脚本参数指定数据目录。核心代码的静态检查可以使用：

```bash
python3 -m py_compile \
  experiments/top_binary_pilot/common.py \
  experiments/top_binary_pilot/build_pilot_dataset.py \
  experiments/top_binary_pilot/protocol_residual.py \
  experiments/top_binary_pilot/run_residual_hgb_probe.py \
  experiments/top_binary_pilot/compare_role_separated_systems.py
```

完整实验还需要论文协议中指定的训练、校准和评价清单。VLESS 等未见代理类型只允许使用与评价分组互斥的纯代理背景进行阈值校准，不使用目标 ToP 标签重新训练或选模。

## 数据可用性说明

本仓库的 `data/` 目录仅提供已经完成预处理的 flow samples，输入格式与 NestTor 的特征提取和检测流程一致，具体格式详见论文描述。

本工作汇总使用的数据包括：

- 已完成提取的 **268,968 条 flow samples**；
- 共计 **28,748 个 PCAP 文件**。

上述统计合并了本项目所有采集轮次，不区分新旧数据。仓库不公开原始 PCAP 文件；由于原始流量规模和分发限制，科研需要获取原始数据时，请联系：

📧 `tovey_zhou@foxmail.com`

## 论文结果

在锁定的 617-flow 未见域名确认清单上，NestTor 取得 **99.00% macro-F1**。相对 FS-Net 的差异在统计上相当，相对 YaTC 和 ET-BERT 分别提高 **2.47** 和 **13.11** 个百分点。该结果属于同一采集环境下的未见域名确认，不宣称全条件绝对 SOTA。

## 引用

如果本仓库对你的研究有帮助，请引用 NestTor 论文。正式书目信息将在论文公开后更新。

## 许可证

许可证信息将在公开版本定稿后补充。
