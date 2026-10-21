# NESTTOR: Detecting Tor-over-Proxy Traffic with Factorized Decisions

[English](README.md) | [论文](https://ieeexplore.ieee.org/document/11735245) | [DOI](https://doi.org/10.1109/TNSM.2026.3743261) | [项目](https://github.com/GSTovey/NestTor)

**IEEE Transactions on Network and Service Management（TNSM），2026**

NESTTOR 研究如何从客户端与加密代理之间观测到的单条双向 TCP 流中识别 Tor-over-Proxy。我们关注代理封装后仍可观测的 Tor 行为，以及如何结合外层代理证据与代理内部的 Tor 证据。

## 主要贡献

- **代理变换后的可观测行为。** 形式化客户端到代理的观测路径，分析代理分帧、缓冲与 TCP 分段对 Tor 流量的变换。通过协议审计，识别方向交互、阶段演化及突发与响应耦合中的可观测证据。
- **因子化检测。** NESTTOR 独立训练代理门控器与代理条件下的 Tor 检测器，通过 AND 规则组合两者的判断。阶段特征描述连接的相对演化，无需重构 Tor 信元。
- **控制信息泄漏的纵向评估。** 构建覆盖 Shadowsocks、Trojan、VMess 和 VLESS 的基准，区分 Tor-over-Proxy、Pure Proxy、Direct Tor 与 Ordinary 流量。评估排除标识特征，在域名和抓包层面划分互斥分区，考察检测有效性、设计贡献、包级变换与分布偏移。

## 研究问题

- **RQ1：可观测行为。** 加密代理封装后，哪些 Tor 行为仍能跨代理被一致地观测？
- **RQ2：检测有效性。** 在控制信息泄漏的条件下，Tor-over-Proxy 检测有多有效？
- **RQ3：设计贡献。** 哪些设计决定了 NESTTOR 的检测性能、鲁棒性和效率？
- **RQ4：泛化能力。** NESTTOR 能否跨代理类型、采集时期和背景分布泛化？

## 方法总览

[![NESTTOR 方法总览](main_framework.png)](figures/nesttor_overview.pdf)

点击主图可查看论文中的矢量 PDF。

## 引用

C. Zhou, C. Huang, Q. Zhang, Q. Xie, and Z. Li, “NESTTOR: Detecting Tor-over-Proxy Traffic with Factorized Decisions,” *IEEE Transactions on Network and Service Management*, 2026, doi: [10.1109/TNSM.2026.3743261](https://doi.org/10.1109/TNSM.2026.3743261).

```bibtex
@article{zhou2026nesttor,
  author  = {Zhou, Chenxin and Huang, Cheng and Zhang, Qiang and Xie, Qiliang and Li, Zeqing},
  title   = {{NESTTOR}: Detecting {Tor-over-Proxy} Traffic with Factorized Decisions},
  journal = {IEEE Transactions on Network and Service Management},
  year    = {2026},
  doi     = {10.1109/TNSM.2026.3743261},
  url     = {https://ieeexplore.ieee.org/document/11735245}
}
```
