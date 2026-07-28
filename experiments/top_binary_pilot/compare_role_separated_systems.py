\
\
\
\
\
\
\
\
\
\
\
\


from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)


CURRENT_METHODS = {
    "NestTor-PAST+Residual": {
        "model": "PAST+Residual-HGB",
        "status": "v8_frozen_selected",
    },
    "NestTor-PAST-Extended": {
        "model": "PAST-Extended-HGB",
        "status": "retrospective_exploratory_candidate_not_confirmatory",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compose_prediction(outer: np.ndarray, inner: np.ndarray) -> np.ndarray:

    return np.logical_and(outer.astype(bool), inner.astype(bool)).astype(int)


def fpr_at_target_tpr(labels: np.ndarray, scores: np.ndarray, target: float = 0.95) -> float:
    fpr, tpr, _ = roc_curve(labels, scores)
    eligible = fpr[tpr >= target]
    return float(np.min(eligible)) if len(eligible) else float("nan")


def metric_row(frame: pd.DataFrame, evaluation: str, method: str, status: str) -> dict:
    labels = frame["binary_label"].to_numpy(dtype=int)
    predictions = frame["prediction"].to_numpy(dtype=int)
    scores = frame["composite_score"].to_numpy(dtype=float)
    positives = labels == 1
    negatives = ~positives
    row = {
        "evaluation": evaluation,
        "method": method,
        "method_status": status,
        "flow_count": int(len(frame)),
        "positive_count": int(positives.sum()),
        "negative_count": int(negatives.sum()),
        "error_count": int(np.sum(labels != predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "auroc_composite_score": float(roc_auc_score(labels, scores)),
        "auprc_composite_score": float(average_precision_score(labels, scores)),
        "tpr": float(predictions[positives].mean()),
        "overall_fpr": float(predictions[negatives].mean()),
        "fpr_at_95tpr_composite_score": fpr_at_target_tpr(labels, scores),
    }
    for background in ("pure_proxy", "normal", "direct_tor"):
        mask = frame["background_group"].to_numpy() == background
        row[f"fpr_{background}"] = float(predictions[mask].mean()) if mask.any() else float("nan")
    for carrier in ("ss", "trojan", "vmess", "vless"):
        mask = positives & (frame["carrier"].to_numpy() == carrier)
        row[f"tpr_{carrier}"] = float(predictions[mask].mean()) if mask.any() else float("nan")
    return row


def cluster_bootstrap_absolute(
    frame: pd.DataFrame, repetitions: int, seed: int
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    clusters = frame["cluster_id"].drop_duplicates().to_numpy()
    cluster_values = frame["cluster_id"].to_numpy()
    groups = {key: np.where(cluster_values == key)[0] for key in clusters}
    labels = frame["binary_label"].to_numpy(dtype=int)
    predictions = frame["prediction"].to_numpy(dtype=int)
    values = np.empty(repetitions, dtype=float)
    for index in range(repetitions):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        positions = np.concatenate([groups[key] for key in sampled])
        values[index] = f1_score(
            labels[positions], predictions[positions], average="macro", zero_division=0
        )
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def cluster_bootstrap_difference(
    left: pd.DataFrame, right: pd.DataFrame, repetitions: int, seed: int
) -> dict[str, float]:
    aligned = left[["sample_id", "binary_label", "cluster_id", "prediction"]].merge(
        right[["sample_id", "prediction"]],
        on="sample_id",
        validate="one_to_one",
        suffixes=("_left", "_right"),
    )
    rng = np.random.default_rng(seed)
    clusters = aligned["cluster_id"].drop_duplicates().to_numpy()
    cluster_values = aligned["cluster_id"].to_numpy()
    groups = {key: np.where(cluster_values == key)[0] for key in clusters}
    labels = aligned["binary_label"].to_numpy(dtype=int)
    left_prediction = aligned["prediction_left"].to_numpy(dtype=int)
    right_prediction = aligned["prediction_right"].to_numpy(dtype=int)
    differences = np.empty(repetitions, dtype=float)
    for index in range(repetitions):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        positions = np.concatenate([groups[key] for key in sampled])
        differences[index] = f1_score(
            labels[positions], left_prediction[positions], average="macro", zero_division=0
        ) - f1_score(
            labels[positions], right_prediction[positions], average="macro", zero_division=0
        )
    point_left = f1_score(labels, left_prediction, average="macro", zero_division=0)
    point_right = f1_score(labels, right_prediction, average="macro", zero_division=0)
    return {
        "flow_count": int(len(aligned)),
        "cluster_count": int(len(clusters)),
        "left_macro_f1": float(point_left),
        "right_macro_f1": float(point_right),
        "difference_pp": float(100.0 * (point_left - point_right)),
        "difference_ci_low_pp": float(100.0 * np.quantile(differences, 0.025)),
        "difference_ci_high_pp": float(100.0 * np.quantile(differences, 0.975)),
        "probability_left_not_better": float(np.mean(differences <= 0.0)),
    }


def current_inner(
    predictions: pd.DataFrame,
    model_dir: Path,
    method: str,
    evaluation: str,
) -> pd.DataFrame:
    model_name = CURRENT_METHODS[method]["model"]
    subset = predictions.loc[
        (predictions["model"] == model_name)
        & (predictions["fold"] == ("closed" if evaluation == "closed" else "vless"))
    ].copy()
    if evaluation == "closed":
        metadata = json.loads((model_dir / f"model_closed_{model_name}.json").read_text())
        threshold = float(metadata["threshold"])
        subset["inner_prediction"] = (subset["probability"] >= threshold).astype(int)
        permission = "source_validation_threshold"
    else:
        subset["inner_prediction"] = subset[
            "negative_calibrated_prediction_alpha01"
        ].astype(int)
        threshold = float("nan")
        permission = "pure_vless_negative_only_alpha01"
    return subset[["sample_id", "probability", "inner_prediction"]].assign(
        inner_threshold=threshold,
        threshold_permission=permission,
    )


def baseline_inner(source_path: Path, calibrated_path: Path, evaluation: str) -> pd.DataFrame:
    if evaluation == "closed":
        subset = pd.read_csv(source_path, keep_default_na=False)
        subset = subset.loc[subset["split"] == "test"].copy()
        subset["inner_prediction"] = subset["prediction_source_threshold"].astype(int)
        permission = "source_validation_threshold"
    else:
        subset = pd.read_csv(calibrated_path, keep_default_na=False)
        subset = subset.loc[subset["evaluation"] == "vless"].copy()
        subset["inner_prediction"] = subset["prediction"].astype(int)
        permission = "pure_vless_negative_only_alpha01"
    return subset[["sample_id", "probability", "inner_prediction"]].assign(
        inner_threshold=float("nan"),
        threshold_permission=permission,
    )


def compose_frame(
    outer: pd.DataFrame,
    inner: pd.DataFrame,
    evaluation: str,
    outer_threshold: float,
) -> pd.DataFrame:
    proxy_ids = set(inner["sample_id"])
    split = "test" if evaluation == "closed" else "vless_test"
    base = outer.loc[outer["split"] == split].copy()
    if evaluation == "vless":
        base = base.loc[
            base["sample_id"].isin(proxy_ids)
            | base["background_group"].isin(["normal", "direct_tor"])
        ].copy()
    proxy_mask = base["background_group"].isin(["positive", "pure_proxy"])
    expected_proxy = base.loc[proxy_mask, "sample_id"]
    if len(expected_proxy) != len(inner) or set(expected_proxy) != proxy_ids:
        raise ValueError(
            f"Inner coverage mismatch for {evaluation}: expected {len(expected_proxy)}, "
            f"received {len(inner)}"
        )
    base = base.merge(inner, on="sample_id", how="left", validate="one_to_one")


    proxy_mask = base["background_group"].isin(["positive", "pure_proxy"])
    base.loc[~proxy_mask, "probability"] = 1.0
    base.loc[~proxy_mask, "inner_prediction"] = 1
    base.loc[~proxy_mask, "threshold_permission"] = "outer_gate_only_nonproxy_background"
    base["outer_prediction"] = (
        base["outer_proxy_probability"].astype(float) >= outer_threshold
    ).astype(int)
    base["prediction"] = compose_prediction(
        base["outer_prediction"].to_numpy(), base["inner_prediction"].to_numpy()
    )
    base["composite_score"] = (
        base["outer_proxy_probability"].astype(float) * base["probability"].astype(float)
    )
    base["cluster_id"] = np.where(
        base["domain_group"].astype(str) != "",
        "domain:" + base["domain_group"].astype(str),
        "capture:" + base["capture_id"].astype(str),
    )
    return base


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outer-predictions", type=Path, required=True)
    parser.add_argument("--outer-model", type=Path, required=True)
    parser.add_argument("--current-predictions", type=Path, required=True)
    parser.add_argument("--current-model-dir", type=Path, required=True)
    for baseline in ("etbert", "fsnet", "yatc"):
        parser.add_argument(f"--{baseline}-source", type=Path, required=True)
        parser.add_argument(f"--{baseline}-calibrated", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    input_paths = [
        args.outer_predictions,
        args.outer_model,
        args.current_predictions,
        args.current_model_dir / "model_closed_PAST+Residual-HGB.json",
        args.current_model_dir / "model_closed_PAST-Extended-HGB.json",
        args.etbert_source,
        args.etbert_calibrated,
        args.fsnet_source,
        args.fsnet_calibrated,
        args.yatc_source,
        args.yatc_calibrated,
    ]
    outer = pd.read_csv(args.outer_predictions, keep_default_na=False)
    if len(outer) != 5000 or outer["sample_id"].duplicated().any():
        raise ValueError("Outer prediction audit requires exactly 5,000 unique flow rows")
    outer_model = json.loads(args.outer_model.read_text())
    outer_threshold = float(outer_model["threshold"])
    current_predictions = pd.read_csv(args.current_predictions, keep_default_na=False)

    baseline_paths = {
        "ET-BERT-Payload": (args.etbert_source, args.etbert_calibrated),
        "FS-Net-SII-free": (args.fsnet_source, args.fsnet_calibrated),
        "YaTC-Payload-MFR": (args.yatc_source, args.yatc_calibrated),
    }
    method_status = {name: spec["status"] for name, spec in CURRENT_METHODS.items()}
    method_status.update({name: "same_manifest_baseline" for name in baseline_paths})

    frames: dict[tuple[str, str], pd.DataFrame] = {}
    for evaluation in ("closed", "vless"):
        for method in CURRENT_METHODS:
            inner = current_inner(
                current_predictions, args.current_model_dir, method, evaluation
            )
            frames[(evaluation, method)] = compose_frame(
                outer, inner, evaluation, outer_threshold
            )
        for method, (source, calibrated) in baseline_paths.items():
            inner = baseline_inner(source, calibrated, evaluation)
            frames[(evaluation, method)] = compose_frame(
                outer, inner, evaluation, outer_threshold
            )

    expected = {
        "closed": {"flow_count": 800, "positive": 400, "pure_proxy": 200,
                   "normal": 100, "direct_tor": 100},
        "vless": {"flow_count": 563, "positive": 229, "pure_proxy": 134,
                  "normal": 100, "direct_tor": 100},
    }
    result_rows = []
    bootstrap_rows = []
    prediction_rows = []
    for (evaluation, method), frame in frames.items():
        counts = frame["background_group"].value_counts().to_dict()
        if len(frame) != expected[evaluation]["flow_count"]:
            raise ValueError(f"Unexpected {evaluation} flow count: {len(frame)}")
        for group in ("positive", "pure_proxy", "normal", "direct_tor"):
            if counts.get(group, 0) != expected[evaluation][group]:
                raise ValueError(f"Unexpected {evaluation}/{group} count: {counts.get(group, 0)}")
        metrics = metric_row(frame, evaluation, method, method_status[method])
        ci_low, ci_high = cluster_bootstrap_absolute(frame, args.repetitions, args.seed)
        metrics.update({
            "bootstrap_repetitions": args.repetitions,
            "bootstrap_cluster_count": int(frame["cluster_id"].nunique()),
            "macro_f1_ci_low": ci_low,
            "macro_f1_ci_high": ci_high,
        })
        result_rows.append(metrics)
        output = frame.copy()
        output.insert(0, "evaluation", evaluation)
        output.insert(1, "method", method)
        prediction_rows.append(output[[
            "evaluation", "method", "sample_id", "binary_label", "background_group",
            "carrier", "capture_id", "domain_group", "cluster_id",
            "outer_proxy_probability", "outer_prediction", "probability",
            "inner_prediction", "threshold_permission", "composite_score", "prediction",
        ]])

    comparisons = [
        ("NestTor-PAST-Extended", "NestTor-PAST+Residual"),
        ("NestTor-PAST-Extended", "ET-BERT-Payload"),
        ("NestTor-PAST-Extended", "FS-Net-SII-free"),
        ("NestTor-PAST-Extended", "YaTC-Payload-MFR"),
        ("NestTor-PAST+Residual", "ET-BERT-Payload"),
        ("NestTor-PAST+Residual", "FS-Net-SII-free"),
        ("NestTor-PAST+Residual", "YaTC-Payload-MFR"),
    ]
    for evaluation in ("closed", "vless"):
        for left_name, right_name in comparisons:
            bootstrap_rows.append({
                "evaluation": evaluation,
                "left_method": left_name,
                "right_method": right_name,
                "left_status": method_status[left_name],
                "right_status": method_status[right_name],
                "bootstrap_repetitions": args.repetitions,
                "seed": args.seed,
                **cluster_bootstrap_difference(
                    frames[(evaluation, left_name)],
                    frames[(evaluation, right_name)],
                    args.repetitions,
                    args.seed,
                ),
            })

    results = pd.DataFrame(result_rows).sort_values(["evaluation", "macro_f1"], ascending=[True, False])
    bootstraps = pd.DataFrame(bootstrap_rows)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    results.to_csv(args.output_dir / "role_separated_system_results.csv", index=False)
    bootstraps.to_csv(args.output_dir / "role_separated_pairwise_bootstrap.csv", index=False)
    predictions.to_csv(args.output_dir / "role_separated_system_predictions.csv", index=False)
    audit = {
        "protocol": "shared_outer_gate_permission_aligned_inner_thresholds",
        "outer_threshold": outer_threshold,
        "seed": args.seed,
        "bootstrap_repetitions": args.repetitions,
        "input_sha256": {str(path): sha256(path) for path in input_paths},
        "expected_counts": expected,
        "method_status": method_status,
        "selection_warning": (
            "PAST-Extended is a retrospective exploratory candidate whose 99.5-level point "
            "estimate was already observed; it requires a new frozen confirmatory block."
        ),
        "target_top_labels_used_for_vless_calibration": False,
        "uses_flow_pairs": False,
    }
    (args.output_dir / "role_separated_system_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n"
    )
    print(results.to_string(index=False))
    print("\nPairwise cluster-bootstrap differences (percentage points):")
    print(bootstraps.to_string(index=False))


if __name__ == "__main__":
    main()
