# NestTor processed sample

[中文](README_zh.md)

This directory contains a 650-flow processed compatibility sample for inspecting the public data format. It is not the fixed development dataset or the held-out confirmatory dataset, and it contains no Pure Proxy negatives.

The files are:

- `manifest.csv`: anonymized sample identifiers, labels, traffic groups, carriers, and split names;
- `past_features.csv`: 99 numerical flow-behavior and stage-dynamics features;
- `sequences.npz`: packet and burst tensors aligned by `sample_id`;
- `dataset_metadata.json`: provenance and scope of this public sample.

`manifest.csv`, `past_features.csv`, and `sequences.npz` are row-aligned by `sample_id`. The model inputs exclude addresses, ports, domains, filenames, absolute timestamps, payload content, and other high-cardinality identifiers.

The sample is provided for schema inspection and code integration only. Reproducing the reported experiments requires the fixed manifests and full processed datasets defined by the paper.
