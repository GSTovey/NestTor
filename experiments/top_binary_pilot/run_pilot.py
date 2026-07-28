from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier

from common import (
    carrier_stable_ranking,
    false_positive_rates,
    group_scores,
    metric_row,
    select_threshold,
)


MODEL_SEEDS = (0, 1, 2)


def load_dataset(data_dir: Path):
    manifest = pd.read_csv(data_dir / "manifest.csv", keep_default_na=False)
    features = pd.read_csv(data_dir / "past_features.csv")
    metadata = json.loads((data_dir / "dataset_metadata.json").read_text(encoding="utf-8"))
    sequences = np.load(data_dir / "sequences.npz", allow_pickle=False)
    if manifest["sample_id"].tolist() != features["sample_id"].tolist():
        raise ValueError("manifest and PAST features are not row-aligned")
    if manifest["sample_id"].tolist() != sequences["sample_ids"].tolist():
        raise ValueError("manifest and sequence tensors are not row-aligned")
    if len(manifest) != 5000:
        raise ValueError(f"Expected exactly 5,000 flows, found {len(manifest)}")
    feature_frame = features.drop(columns=["sample_id"]).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return manifest, feature_frame, metadata, sequences


def leakage_audit(manifest: pd.DataFrame) -> dict[str, object]:
    report: dict[str, object] = {
        "row_count": len(manifest),
        "sample_id_unique": bool(manifest["sample_id"].is_unique),
        "flow_identity_unique": bool(
            ~manifest.duplicated(["capture_id", "flow_id"]).any()
        ),
    }
    capture_splits = manifest.groupby("capture_id")["split"].nunique()
    report["capture_split_overlap_count"] = int((capture_splits > 1).sum())
    domains = manifest.loc[manifest["domain_group"] != ""]
    domain_splits = domains.groupby("domain_group")["split"].nunique()
    report["domain_split_overlap_count"] = int((domain_splits > 1).sum())
    report["quota_table"] = (
        manifest.groupby(["split", "background_group", "carrier"])
        .size().rename("flow_count").reset_index().to_dict(orient="records")
    )
    report["passed"] = bool(
        report["sample_id_unique"]
        and report["flow_identity_unique"]
        and report["capture_split_overlap_count"] == 0
        and report["domain_split_overlap_count"] == 0
    )
    return report


def fit_shortcut_audit(manifest: pd.DataFrame, X: pd.DataFrame, columns: list[str]):
    audit = pd.DataFrame(index=X.index)
    for column in columns:
        if column == "capture_valid_flow_count":
            audit[column] = manifest[column].to_numpy(dtype=float)
        else:
            audit[column] = X[column].to_numpy(dtype=float)
    train = manifest["split"] == "train"
    validation = manifest["split"] == "validation"
    models = []
    for leaf_nodes in (7, 15):
        model = HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.05, max_leaf_nodes=leaf_nodes,
            l2_regularization=1.0, random_state=42,
        ).fit(audit.loc[train], manifest.loc[train, "binary_label"])
        probability = model.predict_proba(audit.loc[validation])[:, 1]
        threshold, worst, _ = select_threshold(manifest.loc[validation].reset_index(drop=True), probability)
        models.append((worst, model, threshold))
    _, model, threshold = max(models, key=lambda item: item[0])
    probabilities = model.predict_proba(audit)[:, 1]
    return probabilities, threshold, audit.columns.tolist()


def single_feature_audit(manifest: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
    train = manifest["split"] == "train"
    validation = manifest["split"] == "validation"
    test = manifest["split"] == "test"
    rows = []
    for feature in X.columns:
        model = DecisionTreeClassifier(max_depth=1, min_samples_leaf=10, random_state=42)
        model.fit(X.loc[train, [feature]], manifest.loc[train, "binary_label"])
        validation_probability = model.predict_proba(X.loc[validation, [feature]])[:, 1]
        threshold, validation_worst, _ = select_threshold(
            manifest.loc[validation].reset_index(drop=True), validation_probability
        )
        test_probability = model.predict_proba(X.loc[test, [feature]])[:, 1]
        test_f1 = metric_row(
            manifest.loc[test, "binary_label"].to_numpy(dtype=int), test_probability, threshold
        )["macro_f1"]
        rows.append(
            {
                "feature": feature,
                "validation_worst_group_macro_f1": validation_worst,
                "test_macro_f1": test_f1,
                "threshold": threshold,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["test_macro_f1", "validation_worst_group_macro_f1"], ascending=False
    )


def sklearn_candidates(model_family: str, seed: int):
    if model_family == "RF":
        for max_features in ("sqrt", 0.5):
            for min_samples_leaf in (1, 3):
                yield {
                    "max_features": max_features,
                    "min_samples_leaf": min_samples_leaf,
                }, RandomForestClassifier(
                    n_estimators=300, max_features=max_features,
                    min_samples_leaf=min_samples_leaf, class_weight="balanced",
                    n_jobs=-1, random_state=seed,
                )
    else:
        for learning_rate in (0.05, 0.10):
            for max_leaf_nodes in (15, 31):
                yield {
                    "learning_rate": learning_rate,
                    "max_leaf_nodes": max_leaf_nodes,
                }, HistGradientBoostingClassifier(
                    max_iter=300, learning_rate=learning_rate,
                    max_leaf_nodes=max_leaf_nodes, l2_regularization=1.0,
                    random_state=seed,
                )


def tune_sklearn_model(
    name: str,
    family: str,
    feature_sets: list[tuple[str, list[str]]],
    manifest: pd.DataFrame,
    X: pd.DataFrame,
) -> tuple[dict[str, object], np.ndarray, float]:
    train = manifest["split"] == "train"
    validation = manifest["split"] == "validation"
    best = None
    for feature_set_name, columns in feature_sets:
        for parameters, model in sklearn_candidates(family, 42):
            model.fit(X.loc[train, columns], manifest.loc[train, "binary_label"])
            probability = model.predict_proba(X.loc[validation, columns])[:, 1]
            threshold, worst, scores = select_threshold(
                manifest.loc[validation].reset_index(drop=True), probability
            )
            candidate = {
                "validation_worst_group_macro_f1": worst,
                "threshold": threshold,
                "group_scores": scores,
                "feature_set": feature_set_name,
                "features": columns,
                "parameters": parameters,
            }
            if best is None or candidate["validation_worst_group_macro_f1"] > best["validation_worst_group_macro_f1"]:
                best = candidate
    assert best is not None

    seed_probabilities = []
    for seed in MODEL_SEEDS:
        chosen = None
        for parameters, model in sklearn_candidates(family, seed):
            if parameters == best["parameters"]:
                chosen = model
                break
        assert chosen is not None
        columns = best["features"]
        chosen.fit(X.loc[train, columns], manifest.loc[train, "binary_label"])
        seed_probabilities.append(chosen.predict_proba(X[columns])[:, 1])
    probabilities = np.mean(seed_probabilities, axis=0)
    validation_probability = probabilities[validation]
    threshold, worst, scores = select_threshold(
        manifest.loc[validation].reset_index(drop=True), validation_probability
    )
    best.update(
        {
            "name": name,
            "threshold": threshold,
            "validation_worst_group_macro_f1": worst,
            "group_scores": scores,
            "model_seeds": list(MODEL_SEEDS),
        }
    )
    return best, probabilities, threshold


def normalize_sequence(array: np.ndarray, mask: np.ndarray, train_indices: np.ndarray):
    output = array.astype(np.float32).copy()
    valid = mask[train_indices].astype(bool)
    values = output[train_indices][valid]
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    std[std < 1e-6] = 1.0
    output = (output - mean) / std
    output[~mask.astype(bool)] = 0.0
    return output, mean.tolist(), std.tolist()


def train_tcn(
    name: str,
    array: np.ndarray,
    mask: np.ndarray,
    manifest: pd.DataFrame,
    output_dir: Path,
    epochs: int,
):
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    class TemporalEncoder(nn.Module):
        def __init__(self, input_dim: int):
            super().__init__()
            layers = []
            channels = input_dim
            for dilation in (1, 2, 4):
                layers.extend(
                    [
                        nn.Conv1d(channels, 64, kernel_size=3, padding=dilation, dilation=dilation),
                        nn.BatchNorm1d(64),
                        nn.ReLU(),
                        nn.Dropout(0.10),
                    ]
                )
                channels = 64
            self.encoder = nn.Sequential(*layers)
            self.head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.20), nn.Linear(64, 1))

        def forward(self, x, valid_mask):
            encoded = self.encoder(x.transpose(1, 2))
            mask_float = valid_mask.unsqueeze(1).float()
            mean_pool = (encoded * mask_float).sum(dim=2) / mask_float.sum(dim=2).clamp_min(1.0)
            max_pool = encoded.masked_fill(~valid_mask.unsqueeze(1), -1e9).max(dim=2).values
            max_pool = torch.where(torch.isfinite(max_pool), max_pool, torch.zeros_like(max_pool))
            return self.head(torch.cat([mean_pool, max_pool], dim=1)).squeeze(1)

    train_indices = np.where(manifest["split"].to_numpy() == "train")[0]
    validation_indices = np.where(manifest["split"].to_numpy() == "validation")[0]
    normalized, mean, std = normalize_sequence(array, mask, train_indices)
    labels = manifest["binary_label"].to_numpy(dtype=np.float32)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    seed_probabilities = []
    seed_details = []

    for seed in MODEL_SEEDS:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        model = TemporalEncoder(normalized.shape[2]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss()
        generator = torch.Generator().manual_seed(seed)
        dataset = TensorDataset(
            torch.from_numpy(normalized[train_indices]),
            torch.from_numpy(mask[train_indices].astype(bool)),
            torch.from_numpy(labels[train_indices]),
        )
        loader = DataLoader(dataset, batch_size=64, shuffle=True, generator=generator)
        best_state = None
        best_score = -1.0
        best_threshold = 0.5
        stale = 0
        history = []
        for epoch in range(epochs):
            model.train()
            losses = []
            for batch_x, batch_mask, batch_y in loader:
                batch_x, batch_mask, batch_y = batch_x.to(device), batch_mask.to(device), batch_y.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(batch_x, batch_mask), batch_y)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            model.eval()
            with torch.no_grad():
                validation_logits = model(
                    torch.from_numpy(normalized[validation_indices]).to(device),
                    torch.from_numpy(mask[validation_indices].astype(bool)).to(device),
                )
                validation_probability = torch.sigmoid(validation_logits).cpu().numpy()
            threshold, worst, _ = select_threshold(
                manifest.iloc[validation_indices].reset_index(drop=True), validation_probability
            )
            history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "validation_worst": worst})
            if worst > best_score + 1e-5:
                best_score = worst
                best_threshold = threshold
                best_state = copy.deepcopy(model.state_dict())
                stale = 0
            else:
                stale += 1
                if stale >= 5:
                    break
        assert best_state is not None
        model.load_state_dict(best_state)
        model.eval()
        probabilities = []
        with torch.no_grad():
            for start in range(0, len(normalized), 256):
                logits = model(
                    torch.from_numpy(normalized[start:start + 256]).to(device),
                    torch.from_numpy(mask[start:start + 256].astype(bool)).to(device),
                )
                probabilities.append(torch.sigmoid(logits).cpu().numpy())
        seed_probabilities.append(np.concatenate(probabilities))
        seed_details.append(
            {
                "seed": seed,
                "best_validation_worst_group_macro_f1": best_score,
                "best_threshold": best_threshold,
                "epochs_ran": len(history),
                "history": history,
            }
        )

    averaged = np.mean(seed_probabilities, axis=0)
    threshold, worst, scores = select_threshold(
        manifest.iloc[validation_indices].reset_index(drop=True), averaged[validation_indices]
    )
    detail = {
        "name": name,
        "threshold": threshold,
        "validation_worst_group_macro_f1": worst,
        "group_scores": scores,
        "normalization_mean": mean,
        "normalization_std": std,
        "device": str(device),
        "model_seeds": list(MODEL_SEEDS),
        "seed_details": seed_details,
    }
    (output_dir / f"{name}_training.json").write_text(
        json.dumps(detail, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return detail, averaged, threshold


def result_rows(name: str, manifest: pd.DataFrame, probabilities: np.ndarray, threshold: float):
    rows = []
    for split in ("validation", "test", "vless_test"):
        mask = manifest["split"].to_numpy() == split
        labels = manifest.loc[mask, "binary_label"].to_numpy(dtype=int)
        row = {"model": name, "split": split, "threshold": threshold}
        row.update(metric_row(labels, probabilities[mask], threshold))
        fprs = false_positive_rates(manifest.loc[mask].reset_index(drop=True), probabilities[mask], threshold)
        row.update({f"fpr_{group}": value for group, value in fprs.items()})
        groups = group_scores(manifest.loc[mask].reset_index(drop=True), probabilities[mask], threshold)
        row["worst_group_macro_f1"] = min(groups.values()) if groups else float("nan")
        rows.append(row)

    vless = manifest["split"].to_numpy() == "vless_test"
    top_indices = np.where(vless & (manifest["binary_label"].to_numpy() == 1))[0]
    pure_indices = np.where(vless & (manifest["background_group"].to_numpy() == "pure_proxy"))[0]
    top_order = sorted(top_indices, key=lambda index: manifest.iloc[index]["sample_id"])
    hard_indices = np.asarray(top_order[:300] + pure_indices.tolist(), dtype=int)
    hard_row = {"model": name, "split": "vless_hard", "threshold": threshold}
    hard_row.update(
        metric_row(
            manifest.iloc[hard_indices]["binary_label"].to_numpy(dtype=int),
            probabilities[hard_indices], threshold,
        )
    )
    hard_row["fpr_pure_proxy"] = float(
        np.mean(probabilities[pure_indices] >= threshold)
    )
    rows.append(hard_row)
    return rows


def bootstrap_difference(
    manifest: pd.DataFrame,
    route_a: np.ndarray,
    threshold_a: float,
    route_b: np.ndarray,
    threshold_b: float,
    iterations: int = 1000,
):
    subset = manifest["split"] == "vless_test"
    meta = manifest.loc[subset].reset_index(drop=True)
    probabilities_a = route_a[subset.to_numpy()]
    probabilities_b = route_b[subset.to_numpy()]
    clusters = meta["domain_group"].where(meta["domain_group"] != "", meta["capture_id"])
    unique_clusters = np.asarray(sorted(clusters.unique()))
    by_cluster = {cluster: np.where(clusters.to_numpy() == cluster)[0] for cluster in unique_clusters}
    rng = np.random.default_rng(42)
    differences = []
    labels = meta["binary_label"].to_numpy(dtype=int)
    for _ in range(iterations):
        sampled = rng.choice(unique_clusters, size=len(unique_clusters), replace=True)
        indices = np.concatenate([by_cluster[cluster] for cluster in sampled])
        if len(np.unique(labels[indices])) < 2:
            continue
        f1_a = f1_score(labels[indices], probabilities_a[indices] >= threshold_a, average="macro", zero_division=0)
        f1_b = f1_score(labels[indices], probabilities_b[indices] >= threshold_b, average="macro", zero_division=0)
        differences.append(float(f1_b - f1_a))
    array = np.asarray(differences)
    return {
        "iterations": len(array),
        "mean_difference": float(np.mean(array)),
        "ci_low": float(np.quantile(array, 0.025)),
        "ci_high": float(np.quantile(array, 0.975)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--continue-after-shortcut", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest, X, metadata, sequences = load_dataset(args.data_dir)
    leakage = leakage_audit(manifest)
    (args.output_dir / "leakage_audit.json").write_text(
        json.dumps(leakage, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if not leakage["passed"]:
        raise SystemExit("Leakage audit failed; model training was not started.")

    flow_shortcut_features = ["packet_count", "payload_packet_count", "duration_ms"]
    shortcut_probability, shortcut_threshold, shortcut_features = fit_shortcut_audit(
        manifest, X, flow_shortcut_features
    )
    capture_probability, capture_threshold, capture_features = fit_shortcut_audit(
        manifest, X, ["capture_valid_flow_count"]
    )
    shortcut_rows = result_rows("FlowShortcut-HGB", manifest, shortcut_probability, shortcut_threshold)
    capture_rows = result_rows("CaptureMetadata-HGB", manifest, capture_probability, capture_threshold)
    pd.DataFrame(shortcut_rows + capture_rows).to_csv(args.output_dir / "shortcut_audit.csv", index=False)
    single_feature_audit(manifest, X).to_csv(args.output_dir / "single_feature_audit.csv", index=False)
    closed_shortcut = next(row for row in shortcut_rows if row["split"] == "test")
    closed_capture = next(row for row in capture_rows if row["split"] == "test")
    audit_summary = {
        "flow_shortcut_features": shortcut_features,
        "flow_shortcut_threshold": shortcut_threshold,
        "flow_shortcut_closed_test_macro_f1": closed_shortcut["macro_f1"],
        "capture_metadata_features": capture_features,
        "capture_metadata_threshold": capture_threshold,
        "capture_metadata_closed_test_macro_f1": closed_capture["macro_f1"],
        "gate_threshold": 0.80,
        "gate_triggered": bool(closed_shortcut["macro_f1"] > 0.80),
        "gate_scope": "Only fields observable from the prediction-unit flow; capture_valid_flow_count is reported separately and is never a model input.",
    }
    (args.output_dir / "audit_summary.json").write_text(
        json.dumps(audit_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if audit_summary["gate_triggered"] and not args.continue_after_shortcut:
        raise SystemExit("Shortcut macro-F1 exceeded 0.80; method comparison paused by protocol.")

    train = manifest["split"] == "train"
    ranking = carrier_stable_ranking(X.loc[train].reset_index(drop=True), manifest.loc[train].reset_index(drop=True))
    ranking.to_csv(args.output_dir / "carrier_stable_feature_ranking.csv", index=False)
    base_features = [column for column in metadata["base_feature_names"] if column in X.columns]
    extended_feature_sets = []
    ranked = ranking["feature"].tolist()
    for size in (16, 32, 64, len(ranked)):
        selected = ranked[: min(size, len(ranked))]
        name = "all" if size >= len(ranked) else f"top{size}"
        if not extended_feature_sets or selected != extended_feature_sets[-1][1]:
            extended_feature_sets.append((name, selected))

    model_details = {}
    probabilities_by_model = {
        "FlowShortcut-HGB": shortcut_probability,
        "CaptureMetadata-HGB": capture_probability,
    }
    thresholds = {
        "FlowShortcut-HGB": shortcut_threshold,
        "CaptureMetadata-HGB": capture_threshold,
    }
    for name, family, feature_sets in (
        ("PAST-PT-RF", "RF", [("base", base_features)]),
        ("PAST-PT-HGB", "HGB", [("base", base_features)]),
        ("CS-PAST-RF", "RF", extended_feature_sets),
        ("CS-PAST-HGB", "HGB", extended_feature_sets),
    ):
        detail, probability, threshold = tune_sklearn_model(name, family, feature_sets, manifest, X)
        model_details[name] = detail
        probabilities_by_model[name] = probability
        thresholds[name] = threshold

    packet_detail, packet_probability, packet_threshold = train_tcn(
        "Packet-TCN", sequences["packet_x"], sequences["packet_mask"], manifest, args.output_dir, args.epochs
    )
    burst_detail, burst_probability, burst_threshold = train_tcn(
        "Burst-TCN", sequences["burst_x"], sequences["burst_mask"], manifest, args.output_dir, args.epochs
    )
    model_details["Packet-TCN"] = packet_detail
    model_details["Burst-TCN"] = burst_detail
    probabilities_by_model["Packet-TCN"] = packet_probability
    probabilities_by_model["Burst-TCN"] = burst_probability
    thresholds["Packet-TCN"] = packet_threshold
    thresholds["Burst-TCN"] = burst_threshold

    result_records = []
    for name, probability in probabilities_by_model.items():
        result_records.extend(result_rows(name, manifest, probability, thresholds[name]))
    pd.DataFrame(result_records).to_csv(args.output_dir / "results.csv", index=False)

    predictions = manifest[[
        "sample_id", "split", "binary_label", "background_group", "carrier",
        "capture_id", "domain_group",
    ]].copy()
    for name, probability in probabilities_by_model.items():
        predictions[f"probability_{name}"] = probability
        predictions[f"prediction_{name}"] = (probability >= thresholds[name]).astype(int)
    predictions.to_csv(args.output_dir / "predictions.csv", index=False)
    (args.output_dir / "model_selection.json").write_text(
        json.dumps(model_details, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    route_a_names = ("PAST-PT-RF", "PAST-PT-HGB", "CS-PAST-RF", "CS-PAST-HGB")
    route_b_names = ("Packet-TCN", "Burst-TCN")
    route_a_name = max(route_a_names, key=lambda name: model_details[name]["validation_worst_group_macro_f1"])
    route_b_name = max(route_b_names, key=lambda name: model_details[name]["validation_worst_group_macro_f1"])
    bootstrap = bootstrap_difference(
        manifest,
        probabilities_by_model[route_a_name], thresholds[route_a_name],
        probabilities_by_model[route_b_name], thresholds[route_b_name],
    )
    vless_rows = {
        (row["model"], row["split"]): row
        for row in result_records if row["split"] in {"vless_test", "vless_hard"}
    }
    improvement = (
        vless_rows[(route_b_name, "vless_test")]["macro_f1"]
        - vless_rows[(route_a_name, "vless_test")]["macro_f1"]
    )
    fpr_degradation = (
        vless_rows[(route_b_name, "vless_hard")]["fpr_at_95_tpr"]
        - vless_rows[(route_a_name, "vless_hard")]["fpr_at_95_tpr"]
    )
    decision = {
        "route_a_model": route_a_name,
        "route_b_model": route_b_name,
        "vless_macro_f1_improvement": improvement,
        "bootstrap": bootstrap,
        "vless_hard_fpr_at_95_tpr_degradation": fpr_degradation,
        "route_b_selected": bool(
            improvement >= 0.02
            and bootstrap["ci_low"] > 0.0
            and fpr_degradation <= 0.01
        ),
        "rule": "B requires >=2.00 pp VLESS macro-F1, bootstrap CI low > 0, and <=1.00 pp hard-negative FPR@95TPR degradation.",
    }
    (args.output_dir / "route_decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"audit": audit_summary, "decision": decision}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
