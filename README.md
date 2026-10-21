# NESTTOR: Detecting Tor-over-Proxy Traffic with Factorized Decisions

[中文](README_zh.md) | [Paper](https://ieeexplore.ieee.org/document/11735245) | [DOI](https://doi.org/10.1109/TNSM.2026.3743261) | [Project](https://github.com/GSTovey/NestTor)

**IEEE Transactions on Network and Service Management (TNSM), 2026**

NESTTOR studies Tor-over-Proxy detection from a single bidirectional TCP flow observed between the client and an encrypted proxy. We investigate which Tor behaviors remain observable after proxy encapsulation and how to combine evidence of an outer proxy with evidence of Tor inside it.

## Main Contributions

- **Observable behavior after proxy transformation.** We formalize the client-to-proxy observation path and analyze how proxy framing, buffering, and TCP segmentation transform Tor traffic. A protocol audit identifies directional interaction, stage evolution, and coupling between bursts and responses as observable sources of evidence.
- **Factorized detection.** NESTTOR trains a proxy gate and a proxy-conditioned Tor detector independently, then combines their decisions with an AND rule. Stage-based flow features describe relative connection evolution without reconstructing Tor cells.
- **Longitudinal evaluation with leakage controls.** We construct a benchmark spanning Shadowsocks, Trojan, VMess, and VLESS, with Tor-over-Proxy, Pure Proxy, Direct Tor, and Ordinary traffic. The evaluation excludes identifier features, separates domains and captures across partitions, and examines detection effectiveness, design contributions, packet transformations, and distribution shifts.

## Research Questions

- **RQ1: Observable behavior.** What Tor behaviors remain consistently observable across proxies after encrypted proxy encapsulation?
- **RQ2: Detection effectiveness.** How effective is ToP detection under controls against information leakage?
- **RQ3: Design contributions.** Which design choices drive NESTTOR's performance, robustness, and efficiency?
- **RQ4: Generalization.** Can NESTTOR generalize across proxy types, capture periods, and background distributions?

## Framework

[![NESTTOR framework](main_framework.png)](figures/nesttor_overview.pdf)

Click the figure to open the vector PDF from the paper.

## Contact

For research inquiries or dataset access, contact [tovey_zhou@foxmail.com](mailto:tovey_zhou@foxmail.com).

## Citation

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
