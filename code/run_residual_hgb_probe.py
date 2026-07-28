from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score

from protocol_residual import eta_squared
from common import stable_hash
from run_pilot import load_dataset
from run_protocol_residual_audit import binary_auc, residual_frame


SEED = 42
DEVELOPMENT_CARRIERS = ("ss", "trojan", "vmess")
FOLDS = ("closed", "ss", "trojan", "vmess", "vless")
CALIBRATION_ALPHAS = (0.01, 0.03, 0.05)


def proxy_mask(manifest: pd.DataFrame) -> np.ndarray:
    return manifest["background_group"].isin(["positive", "pure_proxy"]).to_numpy()


def balanced_group_weights(groups: np.ndarray) -> np.ndarray:
    groups = np.asarray(groups)
    weights = np.zeros(len(groups), dtype=float)
    unique, counts = np.unique(groups, return_counts=True)
    for group, count in zip(unique, counts):
        weights[groups == group] = len(groups) / (len(unique) * count)
    return weights


def inner_indices(manifest: pd.DataFrame, fold: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    split = manifest["split"].to_numpy()
    carriers = manifest["carrier"].to_numpy()
    proxy = proxy_mask(manifest)
    allowed = tuple(carrier for carrier in DEVELOPMENT_CARRIERS if carrier != fold)
    train = np.where(proxy & (split == "train") & np.isin(carriers, allowed))[0]
    validation = np.where(proxy & (split == "validation") & np.isin(carriers, allowed))[0]
    if fold == "closed":
        test = np.where(proxy & (split == "test") & np.isin(carriers, DEVELOPMENT_CARRIERS))[0]
    elif fold == "vless":
        test = np.where(proxy & (split == "vless_test") & (carriers == "vless"))[0]
    else:
        test = np.where(proxy & (split == "test") & (carriers == fold))[0]
    return train, validation, test, allowed


def train_only_residual_ranking(
    manifest: pd.DataFrame,
    residual: pd.DataFrame,
    allowed_carriers: tuple[str, ...],
) -> pd.DataFrame:
    split = manifest["split"].to_numpy()
    carriers = manifest["carrier"].to_numpy()
    background = manifest["background_group"].to_numpy()
    labels = manifest["binary_label"].to_numpy(dtype=int)
    eligible = proxy_mask(manifest) & (split == "train") & np.isin(carriers, allowed_carriers)
    direct_normal = (split == "train") & np.isin(background, ["direct_tor", "normal"])
    reference_labels = (background[direct_normal] == "direct_tor").astype(int)
    rows = []
    for feature in residual.columns:
        values = residual[feature].to_numpy(dtype=float)
        aucs = []
        for carrier in allowed_carriers:
            mask = eligible & (carriers == carrier)
            aucs.append(binary_auc(labels[mask], values[mask]))
        orientation = 1 if np.mean(aucs) >= 0.5 else -1
        oriented = [auc if orientation > 0 else 1.0 - auc for auc in aucs]
        eta_values = []
        for kind in ("positive", "pure_proxy"):
            mask = eligible & (background == kind)
            eta_values.append(eta_squared(values[mask], carriers[mask]))
        leakage = max(eta_values)
        reference_auc = binary_auc(reference_labels, values[direct_normal])
        reference_oriented = reference_auc if orientation > 0 else 1.0 - reference_auc
        rows.append(
            {
                "feature": feature,
                "orientation": orientation,
                "train_worst_oriented_auc": min(oriented),
                "train_mean_oriented_auc": float(np.mean(oriented)),
                "carrier_leakage_eta2": leakage,
                "train_direct_tor_alignment_auc": reference_oriented,
                "stable_pass": bool(min(oriented) >= 0.58 and leakage <= 0.20),
                "tor_aligned_pass": bool(
                    min(oriented) >= 0.58 and leakage <= 0.20 and reference_oriented >= 0.55
                ),
                "ranking_score": min(oriented) - 0.25 * leakage,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["stable_pass", "ranking_score", "train_mean_oriented_auc"], ascending=False
    )


def feature_sets(
    past: pd.DataFrame,
    residual: pd.DataFrame,
    base_names: list[str],
    ranking: pd.DataFrame,
) -> dict[str, tuple[pd.DataFrame, list[str]]]:
    stable = ranking.loc[ranking["stable_pass"], "feature"].head(64).tolist()
    if len(stable) < 8:
        stable = ranking.head(32)["feature"].tolist()
    tor_aligned = ranking.loc[ranking["tor_aligned_pass"], "feature"].head(64).tolist()
    if len(tor_aligned) < 8:
        tor_aligned = stable
    structural = [
        feature for feature in residual.columns
        if "empty" not in feature and "reaches" not in feature and "boundary" not in feature
    ]
    past_base = [feature for feature in base_names if feature in past.columns]
    combined = pd.concat(
        [past[past_base].add_prefix("past__"), residual[stable].add_prefix("residual__")], axis=1
    )
    return {
        "PAST-Base-HGB": (past, past_base),
        "PAST-Extended-HGB": (past, past.columns.tolist()),
        "Residual-All-HGB": (residual, residual.columns.tolist()),
        "Residual-Structural-HGB": (residual, structural),
        "Residual-Stable-HGB": (residual, stable),
        "Residual-TorAligned-HGB": (residual, tor_aligned),
        "PAST+Residual-HGB": (combined, combined.columns.tolist()),
    }


def inner_threshold(
    labels: np.ndarray, probabilities: np.ndarray, carriers: np.ndarray
) -> tuple[float, float, float]:
    best = None
    for threshold in np.linspace(0.05, 0.95, 181):
        scores = [
            f1_score(
                labels[carriers == carrier],
                probabilities[carriers == carrier] >= threshold,
                average="macro",
                zero_division=0,
            )
            for carrier in sorted(set(carriers))
        ]
        candidate = (min(scores), float(np.mean(scores)), -abs(float(threshold) - 0.5), float(threshold))
        if best is None or candidate > best:
            best = candidate
    assert best is not None
    return best[3], best[0], best[1]


def hgb_candidates():
    for learning_rate in (0.03, 0.07):
        for leaves in (7, 15, 31):
            for l2 in (1.0, 3.0):
                parameters = {
                    "learning_rate": learning_rate,
                    "max_leaf_nodes": leaves,
                    "l2_regularization": l2,
                }
                model = HistGradientBoostingClassifier(
                    max_iter=300,
                    learning_rate=learning_rate,
                    max_leaf_nodes=leaves,
                    l2_regularization=l2,
                    early_stopping=False,
                    random_state=SEED,
                )
                yield parameters, model


def fit_inner_model(
    manifest: pd.DataFrame,
    frame: pd.DataFrame,
    columns: list[str],
    train: np.ndarray,
    validation: np.ndarray,
) -> tuple[HistGradientBoostingClassifier, np.ndarray, float, dict[str, object]]:
    labels = manifest["binary_label"].to_numpy(dtype=int)
    carriers = manifest["carrier"].to_numpy()
    train_groups = np.asarray([f"{carrier}:{label}" for carrier, label in zip(carriers[train], labels[train])])
    weights = balanced_group_weights(train_groups)
    best = None
    for parameters, model in hgb_candidates():
        model.fit(frame.iloc[train][columns], labels[train], sample_weight=weights)
        probability = model.predict_proba(frame.iloc[validation][columns])[:, 1]
        threshold, worst, mean = inner_threshold(labels[validation], probability, carriers[validation])
        candidate = (worst, mean, -abs(threshold - 0.5), parameters, model, threshold)
        if best is None or candidate[:3] > best[:3]:
            best = candidate
    assert best is not None
    _, _, _, parameters, model, threshold = best
    probabilities = model.predict_proba(frame[columns])[:, 1]
    threshold, worst, mean = inner_threshold(
        labels[validation], probabilities[validation], carriers[validation]
    )
    details = {
        "features": columns,
        "feature_count": len(columns),
        "parameters": parameters,
        "threshold": threshold,
        "validation_worst_carrier_macro_f1": worst,
        "validation_mean_carrier_macro_f1": mean,
    }
    return model, probabilities, threshold, details


def binary_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    predictions = probabilities >= threshold
    return {
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "auroc": float(roc_auc_score(labels, probabilities)),
        "auprc": float(average_precision_score(labels, probabilities)),
        "tpr": float(np.mean(predictions[labels == 1])) if np.any(labels == 1) else float("nan"),
        "fpr": float(np.mean(predictions[labels == 0])) if np.any(labels == 0) else float("nan"),
    }


def metrics_from_predictions(
    labels: np.ndarray, predictions: np.ndarray, probabilities: np.ndarray
) -> dict[str, float]:
    labels = np.asarray(labels, dtype=int)
    predictions = np.asarray(predictions, dtype=bool)
    probabilities = np.asarray(probabilities, dtype=float)
    return {
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "auroc": float(roc_auc_score(labels, probabilities)),
        "auprc": float(average_precision_score(labels, probabilities)),
        "tpr": float(np.mean(predictions[labels == 1])) if np.any(labels == 1) else float("nan"),
        "fpr": float(np.mean(predictions[labels == 0])) if np.any(labels == 0) else float("nan"),
    }


def negative_only_threshold(scores: np.ndarray, alpha: float) -> float:

    scores = np.asarray(scores, dtype=float)
    if not len(scores):
        raise ValueError("Negative-only calibration requires at least one pure-proxy score")
    try:
        quantile = np.quantile(scores, 1.0 - alpha, method="higher")
    except TypeError:
        quantile = np.quantile(scores, 1.0 - alpha, interpolation="higher")
    return float(np.nextafter(float(quantile), np.inf))


def negative_calibration_layout(
    manifest: pd.DataFrame, fold: str
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, object]]:
    split = manifest["split"].to_numpy()
    carrier = manifest["carrier"].to_numpy()
    background = manifest["background_group"].to_numpy()
    proxy = proxy_mask(manifest)
    pure = background == "pure_proxy"
    if fold == "closed":
        calibration = {
            name: np.where((split == "validation") & pure & (carrier == name))[0]
            for name in DEVELOPMENT_CARRIERS
        }
        evaluation = np.where((split == "test") & proxy & np.isin(carrier, DEVELOPMENT_CARRIERS))[0]
        details = {
            "setting": "seen-carrier negative calibration",
            "calibration_splits": ["validation"],
            "target_top_labels_used": False,
        }
        return calibration, evaluation, details
    if fold in DEVELOPMENT_CARRIERS:
        calibration = {
            fold: np.where(np.isin(split, ["train", "validation"]) & pure & (carrier == fold))[0]
        }
        evaluation = np.where((split == "test") & proxy & (carrier == fold))[0]
        details = {
            "setting": "unseen-carrier pure-proxy-only calibration",
            "calibration_splits": ["train", "validation"],
            "target_top_labels_used": False,
        }
        return calibration, evaluation, details
    if fold == "vless":
        keys = manifest["domain_group"].where(
            manifest["domain_group"] != "", manifest["capture_id"]
        ).astype(str)
        calibration_domain = keys.map(lambda key: stable_hash(f"vless-negative-calibration:{key}") % 2 == 0).to_numpy()
        calibration = {
            "vless": np.where((split == "vless_test") & pure & (carrier == "vless") & calibration_domain)[0]
        }
        evaluation = np.where(
            (split == "vless_test") & proxy & (carrier == "vless") & ~calibration_domain
        )[0]
        details = {
            "setting": "VLESS pure-proxy-only domain-disjoint calibration",
            "calibration_splits": ["vless_test/calibration-domain-half"],
            "evaluation_splits": ["vless_test/evaluation-domain-half"],
            "target_top_labels_used": False,
        }
        return calibration, evaluation, details
    raise ValueError(f"Unknown fold: {fold}")


def apply_negative_calibration(
    manifest: pd.DataFrame,
    probabilities: np.ndarray,
    fold: str,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float], dict[str, object]]:
    calibration, evaluation, details = negative_calibration_layout(manifest, fold)
    carriers = manifest["carrier"].to_numpy()
    predictions = np.zeros(len(evaluation), dtype=bool)
    thresholds = {}
    for name, calibration_indices in calibration.items():
        threshold = negative_only_threshold(probabilities[calibration_indices], alpha)
        thresholds[name] = threshold
        mask = carriers[evaluation] == name
        predictions[mask] = probabilities[evaluation[mask]] > threshold
    details = {
        **details,
        "alpha": alpha,
        "calibration_counts": {name: len(indices) for name, indices in calibration.items()},
        "evaluation_count": len(evaluation),
    }
    return evaluation, predictions, thresholds, details


def outer_threshold(labels: np.ndarray, probabilities: np.ndarray, groups: np.ndarray) -> tuple[float, float]:
    best = None
    for threshold in np.linspace(0.05, 0.95, 181):
        predictions = probabilities >= threshold
        recalls = []
        for group in sorted(set(groups)):
            mask = groups == group
            expected = int(round(float(np.mean(labels[mask]))))
            recalls.append(float(np.mean(predictions[mask] == expected)))
        candidate = (min(recalls), float(np.mean(recalls)), -abs(float(threshold) - 0.5), float(threshold))
        if best is None or candidate > best:
            best = candidate
    assert best is not None
    return best[3], best[0]


def outer_groups(manifest: pd.DataFrame) -> np.ndarray:
    groups = []
    for background, carrier in zip(manifest["background_group"], manifest["carrier"]):
        if background in {"positive", "pure_proxy"}:
            groups.append(f"proxy:{carrier}")
        else:
            groups.append(background)
    return np.asarray(groups)


def fit_outer_gate(
    manifest: pd.DataFrame, past: pd.DataFrame
) -> tuple[np.ndarray, float, dict[str, object]]:
    split = manifest["split"].to_numpy()
    carriers = manifest["carrier"].to_numpy()
    background = manifest["background_group"].to_numpy()
    labels = np.isin(background, ["positive", "pure_proxy"]).astype(int)
    groups = outer_groups(manifest)
    train = np.where((split == "train") & (carriers != "vless"))[0]
    validation = np.where((split == "validation") & (carriers != "vless"))[0]
    weights = balanced_group_weights(groups[train])
    best = None
    for parameters, model in hgb_candidates():
        model.fit(past.iloc[train], labels[train], sample_weight=weights)
        probability = model.predict_proba(past.iloc[validation])[:, 1]
        threshold, worst = outer_threshold(labels[validation], probability, groups[validation])
        macro = f1_score(labels[validation], probability >= threshold, average="macro", zero_division=0)
        candidate = (worst, macro, -abs(threshold - 0.5), parameters, model, threshold)
        if best is None or candidate[:3] > best[:3]:
            best = candidate
    assert best is not None
    _, _, _, parameters, model, _ = best
    probabilities = model.predict_proba(past)[:, 1]
    threshold, worst = outer_threshold(labels[validation], probabilities[validation], groups[validation])
    return probabilities, threshold, {
        "features": past.columns.tolist(),
        "feature_count": past.shape[1],
        "parameters": parameters,
        "threshold": threshold,
        "validation_worst_group_recall": worst,
    }


def cluster_bootstrap_difference(
    manifest: pd.DataFrame,
    indices: np.ndarray,
    left_probability: np.ndarray,
    left_threshold: float,
    right_probability: np.ndarray,
    right_threshold: float,
    iterations: int,
) -> dict[str, float | int]:
    meta = manifest.iloc[indices].reset_index(drop=True)
    labels = meta["binary_label"].to_numpy(dtype=int)
    clusters = meta["domain_group"].where(meta["domain_group"] != "", meta["capture_id"]).to_numpy()
    unique_clusters = np.asarray(sorted(set(clusters)))
    by_cluster = {cluster: np.where(clusters == cluster)[0] for cluster in unique_clusters}
    rng = np.random.default_rng(SEED)
    differences = []
    left = left_probability[indices]
    right = right_probability[indices]
    for _ in range(iterations):
        chosen = rng.choice(unique_clusters, size=len(unique_clusters), replace=True)
        sampled = np.concatenate([by_cluster[cluster] for cluster in chosen])
        if len(np.unique(labels[sampled])) != 2:
            continue
        left_f1 = f1_score(
            labels[sampled], left[sampled] >= left_threshold, average="macro", zero_division=0
        )
        right_f1 = f1_score(
            labels[sampled], right[sampled] >= right_threshold, average="macro", zero_division=0
        )
        differences.append(float(right_f1 - left_f1))
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
    parser.add_argument("--bootstrap-iterations", type=int, default=500)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"Refusing to overwrite existing output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    manifest, past, metadata, sequences = load_dataset(args.data_dir)
    residual = residual_frame(sequences)
    results = []
    calibrated_results = []
    calibrated_predictions: dict[tuple[str, str, float], tuple[np.ndarray, np.ndarray, dict[str, float]]] = {}
    long_predictions = []
    predictions: dict[tuple[str, str], tuple[np.ndarray, float]] = {}
    selections = {}

    for fold in FOLDS:
        train, validation, test, allowed = inner_indices(manifest, fold)
        ranking = train_only_residual_ranking(manifest, residual, allowed)
        ranking.to_csv(args.output_dir / f"residual_ranking_{fold}.csv", index=False)
        selections[fold] = {
            "allowed_training_carriers": list(allowed),
            "train_count": len(train),
            "validation_count": len(validation),
            "test_count": len(test),
            "stable_pass_count": int(ranking["stable_pass"].sum()),
            "tor_aligned_pass_count": int(ranking["tor_aligned_pass"].sum()),
        }
        for name, (frame, columns) in feature_sets(
            past, residual, metadata["base_feature_names"], ranking
        ).items():
            _, probability, threshold, detail = fit_inner_model(
                manifest, frame, columns, train, validation
            )
            predictions[(fold, name)] = (probability, threshold)
            row = {
                "fold": fold,
                "model": name,
                "training_carriers": "+".join(allowed),
                "test_carrier": "ss+trojan+vmess" if fold == "closed" else fold,
                "threshold": threshold,
                "feature_count": len(columns),
                "validation_worst_carrier_macro_f1": detail["validation_worst_carrier_macro_f1"],
            }
            row.update(
                binary_metrics(
                    manifest.iloc[test]["binary_label"].to_numpy(dtype=int),
                    probability[test],
                    threshold,
                )
            )
            results.append(row)
            for alpha in CALIBRATION_ALPHAS:
                calibration_test, calibration_prediction, calibration_thresholds, calibration_details = (
                    apply_negative_calibration(manifest, probability, fold, alpha)
                )
                calibrated_predictions[(fold, name, alpha)] = (
                    calibration_test,
                    calibration_prediction,
                    calibration_thresholds,
                )
                calibrated_row = {
                    "fold": fold,
                    "model": name,
                    "alpha": alpha,
                    "training_carriers": "+".join(allowed),
                    "test_carrier": "ss+trojan+vmess" if fold == "closed" else fold,
                    "calibration_counts": json.dumps(
                        calibration_details["calibration_counts"], sort_keys=True
                    ),
                    "evaluation_count": len(calibration_test),
                    "thresholds": json.dumps(calibration_thresholds, sort_keys=True),
                    "target_top_labels_used_for_calibration": False,
                }
                calibrated_row.update(
                    metrics_from_predictions(
                        manifest.iloc[calibration_test]["binary_label"].to_numpy(dtype=int),
                        calibration_prediction,
                        probability[calibration_test],
                    )
                )
                calibrated_results.append(calibrated_row)
                if alpha == 0.01:
                    for position, index in enumerate(calibration_test):
                        long_predictions.append(
                            {
                                "sample_id": manifest.iloc[index]["sample_id"],
                                "fold": fold,
                                "model": name,
                                "carrier": manifest.iloc[index]["carrier"],
                                "binary_label": int(manifest.iloc[index]["binary_label"]),
                                "probability": float(probability[index]),
                                "negative_calibrated_prediction_alpha01": int(calibration_prediction[position]),
                            }
                        )
            (args.output_dir / f"model_{fold}_{name}.json").write_text(
                json.dumps(detail, ensure_ascii=False, indent=2), encoding="utf-8"
            )

    result_frame = pd.DataFrame(results)
    result_frame.to_csv(args.output_dir / "inner_probe_results.csv", index=False)
    calibrated_frame = pd.DataFrame(calibrated_results)
    calibrated_frame.to_csv(args.output_dir / "negative_calibrated_results.csv", index=False)
    pd.DataFrame(long_predictions).to_csv(
        args.output_dir / "negative_calibrated_predictions_alpha01.csv", index=False
    )
    development = calibrated_frame[
        calibrated_frame["fold"].isin(DEVELOPMENT_CARRIERS) & (calibrated_frame["alpha"] == 0.01)
    ]
    method_ranking = (
        development.groupby("model")["macro_f1"]
        .agg(["min", "mean"])
        .reset_index()
        .sort_values(["min", "mean"], ascending=False)
    )
    method_ranking.to_csv(args.output_dir / "development_method_ranking.csv", index=False)
    selected_method = str(method_ranking.iloc[0]["model"])
    past_methods = method_ranking[method_ranking["model"].str.startswith("PAST-")]
    best_past_method = str(past_methods.iloc[0]["model"])

    bootstrap_rows = []
    for fold in DEVELOPMENT_CARRIERS + ("vless",):
        _, _, test, _ = inner_indices(manifest, fold)
        left_probability, left_threshold = predictions[(fold, best_past_method)]
        right_probability, right_threshold = predictions[(fold, selected_method)]
        row = {
            "fold": fold,
            "baseline": best_past_method,
            "candidate": selected_method,
        }
        row.update(
            cluster_bootstrap_difference(
                manifest,
                test,
                left_probability,
                left_threshold,
                right_probability,
                right_threshold,
                args.bootstrap_iterations,
            )
        )
        bootstrap_rows.append(row)
    pd.DataFrame(bootstrap_rows).to_csv(args.output_dir / "inner_probe_bootstrap.csv", index=False)

    outer_probability, outer_threshold_value, outer_detail = fit_outer_gate(manifest, past)
    (args.output_dir / "outer_gate_model.json").write_text(
        json.dumps(outer_detail, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    outer_labels = manifest["background_group"].isin(["positive", "pure_proxy"]).to_numpy(dtype=int)
    outer_rows = []
    for split_name, mask in (
        ("closed", manifest["split"].to_numpy() == "test"),
        ("vless", manifest["split"].to_numpy() == "vless_test"),
    ):
        row = {"split": split_name, "threshold": outer_threshold_value}
        row.update(binary_metrics(outer_labels[mask], outer_probability[mask], outer_threshold_value))
        for background in ("direct_tor", "normal"):
            group = mask & (manifest["background_group"].to_numpy() == background)
            row[f"fpr_{background}"] = float(np.mean(outer_probability[group] >= outer_threshold_value))
        proxy_group = mask & np.isin(manifest["background_group"].to_numpy(), ["positive", "pure_proxy"])
        row["proxy_recall"] = float(np.mean(outer_probability[proxy_group] >= outer_threshold_value))
        outer_rows.append(row)
    pd.DataFrame(outer_rows).to_csv(args.output_dir / "outer_gate_results.csv", index=False)

    inner_probability, inner_threshold_value = predictions[("closed", selected_method)]
    final_rows = []
    final_labels = manifest["binary_label"].to_numpy(dtype=int)
    final_prediction_output = np.zeros(len(manifest), dtype=int)
    for split_name, fold in (("closed", "closed"), ("vless", "vless")):
        fold_probability, fold_source_threshold = predictions[(fold, selected_method)]
        calibrated_indices, calibrated_inner_prediction, calibrated_thresholds = calibrated_predictions[
            (fold, selected_method, 0.01)
        ]
        inner_decision = fold_probability >= fold_source_threshold
        inner_decision[calibrated_indices] = calibrated_inner_prediction
        final_prediction = (outer_probability >= outer_threshold_value) & inner_decision
        if split_name == "closed":
            mask = manifest["split"].to_numpy() == "test"
        else:
            proxy_evaluation = np.zeros(len(manifest), dtype=bool)
            proxy_evaluation[calibrated_indices] = True
            shared_background = (
                (manifest["split"].to_numpy() == "vless_test")
                & manifest["background_group"].isin(["normal", "direct_tor"]).to_numpy()
            )
            mask = proxy_evaluation | shared_background
        final_prediction_output[mask] = final_prediction[mask].astype(int)
        labels = final_labels[mask]
        predictions_split = final_prediction[mask]
        row = {
            "split": split_name,
            "inner_model": selected_method,
            "outer_threshold": outer_threshold_value,
            "inner_source_threshold": fold_source_threshold,
            "negative_calibration_alpha": 0.01,
            "negative_calibration_thresholds": json.dumps(calibrated_thresholds, sort_keys=True),
            "evaluation_count": int(np.sum(mask)),
            "macro_f1": float(f1_score(labels, predictions_split, average="macro", zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(labels, predictions_split)),
        }
        for background in ("pure_proxy", "normal", "direct_tor"):
            group = mask & (manifest["background_group"].to_numpy() == background)
            row[f"fpr_{background}"] = float(np.mean(final_prediction[group]))
        final_rows.append(row)
    pd.DataFrame(final_rows).to_csv(args.output_dir / "role_separated_results.csv", index=False)

    selected_rows = result_frame[result_frame["model"] == selected_method].set_index("fold")
    selected_calibrated_rows = calibrated_frame[
        (calibrated_frame["model"] == selected_method) & (calibrated_frame["alpha"] == 0.01)
    ].set_index("fold")
    decision = {
        "selection_rule": "Select by worst SS/Trojan/VMess holdout macro-F1 after alpha=0.01 pure-proxy-only calibration; VLESS is reported only after selection.",
        "selected_inner_method": selected_method,
        "best_past_baseline": best_past_method,
        "zero_shot_development_holdout_worst_macro_f1": float(
            selected_rows.loc[list(DEVELOPMENT_CARRIERS), "macro_f1"].min()
        ),
        "calibrated_development_holdout_worst_macro_f1": float(
            selected_calibrated_rows.loc[list(DEVELOPMENT_CARRIERS), "macro_f1"].min()
        ),
        "zero_shot_closed_proxy_conditioned_macro_f1": float(selected_rows.loc["closed", "macro_f1"]),
        "zero_shot_vless_proxy_conditioned_macro_f1": float(selected_rows.loc["vless", "macro_f1"]),
        "calibrated_closed_proxy_conditioned_macro_f1": float(
            selected_calibrated_rows.loc["closed", "macro_f1"]
        ),
        "calibrated_vless_proxy_conditioned_macro_f1": float(
            selected_calibrated_rows.loc["vless", "macro_f1"]
        ),
        "target_99_5_reached_on_calibrated_closed": bool(
            selected_calibrated_rows.loc["closed", "macro_f1"] >= 0.995
        ),
        "target_99_5_reached_on_calibrated_vless_diagnostic": bool(
            selected_calibrated_rows.loc["vless", "macro_f1"] >= 0.995
        ),
        "outer_gate_threshold": outer_threshold_value,
        "inner_source_threshold": inner_threshold_value,
        "negative_calibration_alpha": 0.01,
        "calibration_uses_target_top_labels": False,
        "vless_is_confirmatory": False,
        "uses_flow_pairs": False,
        "next_action": "Retain only if cross-carrier and role-separated error structure improve; this diagnostic cannot establish SOTA.",
    }
    (args.output_dir / "probe_decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    prediction_output = manifest[[
        "sample_id", "split", "binary_label", "background_group", "carrier", "capture_id", "domain_group"
    ]].copy()
    prediction_output["outer_proxy_probability"] = outer_probability
    prediction_output["inner_tor_probability"] = inner_probability
    prediction_output["role_separated_prediction_on_evaluation_rows"] = final_prediction_output
    prediction_output.to_csv(args.output_dir / "role_separated_predictions.csv", index=False)
    (args.output_dir / "probe_metadata.json").write_text(
        json.dumps(
            {
                "seed": SEED,
                "row_count": len(manifest),
                "packet_budget": int(sequences["packet_x"].shape[1]),
                "sii_fields_used": [],
                "selection_by_vless": False,
                "selections": selections,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
