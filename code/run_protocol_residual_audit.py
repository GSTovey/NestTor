from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from protocol_residual import eta_squared, feature_family, flow_residual_features
from run_pilot import load_dataset


DEVELOPMENT_CARRIERS = ("ss", "trojan", "vmess")
ALL_CARRIERS = DEVELOPMENT_CARRIERS + ("vless",)
SEED = 42


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if len(np.unique(labels)) != 2 or np.std(scores) < 1e-12:
        return 0.5
    return float(roc_auc_score(labels, scores))


def proxy_carrier_mask(manifest: pd.DataFrame, carrier: str) -> np.ndarray:
    background = manifest["background_group"].to_numpy()
    return np.isin(background, ["positive", "pure_proxy"]) & (manifest["carrier"].to_numpy() == carrier)


def residual_frame(sequences) -> pd.DataFrame:
    rows = [
        flow_residual_features(packet_x, packet_mask)
        for packet_x, packet_mask in zip(sequences["packet_x"], sequences["packet_mask"])
    ]
    frame = pd.DataFrame(rows).sort_index(axis=1)
    if frame.isna().any().any() or not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError("Residual feature frame contains missing or non-finite values")
    return frame


def audit_features(manifest: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    labels = manifest["binary_label"].to_numpy(dtype=int)
    backgrounds = manifest["background_group"].to_numpy()
    carriers = manifest["carrier"].to_numpy()
    dev_proxy = np.isin(backgrounds, ["positive", "pure_proxy"]) & np.isin(carriers, DEVELOPMENT_CARRIERS)
    direct_normal = np.isin(backgrounds, ["direct_tor", "normal"])
    tor_reference_labels = (backgrounds[direct_normal] == "direct_tor").astype(int)
    rows = []
    for feature in features.columns:
        values = features[feature].to_numpy(dtype=float)
        carrier_auc = {}
        for carrier in ALL_CARRIERS:
            mask = proxy_carrier_mask(manifest, carrier)
            carrier_auc[carrier] = binary_auc(labels[mask], values[mask])
        dev_mean = float(np.mean([carrier_auc[carrier] for carrier in DEVELOPMENT_CARRIERS]))
        orientation = 1 if dev_mean >= 0.5 else -1
        oriented = {
            carrier: (carrier_auc[carrier] if orientation > 0 else 1.0 - carrier_auc[carrier])
            for carrier in ALL_CARRIERS
        }
        direct_auc = binary_auc(tor_reference_labels, values[direct_normal])
        direct_oriented_auc = direct_auc if orientation > 0 else 1.0 - direct_auc
        eta_by_background = {}
        for background in ("positive", "pure_proxy"):
            mask = dev_proxy & (backgrounds == background)
            eta_by_background[background] = eta_squared(values[mask], carriers[mask])
        carrier_leakage = max(eta_by_background.values())
        dev_worst = min(oriented[carrier] for carrier in DEVELOPMENT_CARRIERS)
        all_worst = min(oriented.values())
        direction_agreement_dev = all(oriented[carrier] >= 0.5 for carrier in DEVELOPMENT_CARRIERS)
        direction_agreement_all = all(value >= 0.5 for value in oriented.values())
        candidate_pass = bool(
            dev_worst >= 0.58
            and oriented["vless"] >= 0.58
            and direction_agreement_all
            and carrier_leakage <= 0.20
        )
        row = {
            "feature": feature,
            "family": feature_family(feature),
            "orientation": orientation,
            **{f"auc_{carrier}": carrier_auc[carrier] for carrier in ALL_CARRIERS},
            **{f"oriented_auc_{carrier}": oriented[carrier] for carrier in ALL_CARRIERS},
            "dev_worst_oriented_auc": dev_worst,
            "all_carrier_worst_oriented_auc": all_worst,
            "direction_agreement_dev": direction_agreement_dev,
            "direction_agreement_all": direction_agreement_all,
            "direct_tor_vs_normal_auc": direct_auc,
            "direct_tor_vs_normal_oriented_auc": direct_oriented_auc,
            "tor_family_alignment": bool(direct_oriented_auc >= 0.55),
            "carrier_eta2_top": eta_by_background["positive"],
            "carrier_eta2_pure_proxy": eta_by_background["pure_proxy"],
            "carrier_leakage_eta2": carrier_leakage,
            "candidate_pass": candidate_pass,
            "audit_score": all_worst - 0.25 * carrier_leakage,
        }
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["candidate_pass", "audit_score", "dev_worst_oriented_auc"], ascending=False
    )


def family_summary(audit: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for family, subset in audit.groupby("family", sort=True):
        best = subset.sort_values("audit_score", ascending=False).iloc[0]
        rows.append(
            {
                "family": family,
                "feature_count": len(subset),
                "candidate_pass_count": int(subset["candidate_pass"].sum()),
                "best_feature": best["feature"],
                "best_audit_score": best["audit_score"],
                "best_dev_worst_oriented_auc": best["dev_worst_oriented_auc"],
                "best_vless_oriented_auc": best["oriented_auc_vless"],
                "best_all_carrier_worst_oriented_auc": best["all_carrier_worst_oriented_auc"],
                "best_direct_tor_vs_normal_oriented_auc": best["direct_tor_vs_normal_oriented_auc"],
                "best_carrier_leakage_eta2": best["carrier_leakage_eta2"],
                "family_pass": bool(subset["candidate_pass"].any()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["family_pass", "best_audit_score"], ascending=False
    )


def permutation_null(
    manifest: pd.DataFrame, features: pd.DataFrame, observed: float, iterations: int
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    backgrounds = manifest["background_group"].to_numpy()
    carriers = manifest["carrier"].to_numpy()
    labels = manifest["binary_label"].to_numpy(dtype=int)
    values = features.to_numpy(dtype=float)
    masks = {carrier: proxy_carrier_mask(manifest, carrier) for carrier in DEVELOPMENT_CARRIERS}
    rng = np.random.default_rng(SEED)
    null_maxima = []
    for iteration in range(iterations):
        auc_matrix = np.zeros((len(DEVELOPMENT_CARRIERS), values.shape[1]), dtype=float)
        for carrier_index, carrier in enumerate(DEVELOPMENT_CARRIERS):
            mask = masks[carrier]
            carrier_labels = labels[mask].copy()
            rng.shuffle(carrier_labels)
            carrier_values = values[mask]
            for feature_index in range(values.shape[1]):
                auc_matrix[carrier_index, feature_index] = binary_auc(
                    carrier_labels, carrier_values[:, feature_index]
                )
        orientations = np.where(np.mean(auc_matrix, axis=0) >= 0.5, 1.0, -1.0)
        oriented = np.where(orientations[None, :] > 0.0, auc_matrix, 1.0 - auc_matrix)
        null_maxima.append(float(np.max(np.min(oriented, axis=0))))
    null = pd.DataFrame({"iteration": np.arange(iterations), "null_max_dev_worst_auc": null_maxima})
    array = null["null_max_dev_worst_auc"].to_numpy()
    summary = {
        "iterations": iterations,
        "observed_max_dev_worst_auc": observed,
        "null_mean": float(np.mean(array)),
        "null_q95": float(np.quantile(array, 0.95)),
        "familywise_empirical_p": float((1 + np.sum(array >= observed)) / (iterations + 1)),
    }
    return null, summary


def cluster_bootstrap(
    manifest: pd.DataFrame,
    values: np.ndarray,
    carrier: str,
    orientation: int,
    iterations: int,
    seed_offset: int,
) -> tuple[float, float, float, int]:
    mask = proxy_carrier_mask(manifest, carrier)
    meta = manifest.loc[mask].reset_index(drop=True)
    labels = meta["binary_label"].to_numpy(dtype=int)
    scores = np.asarray(values[mask], dtype=float)
    clusters = meta["domain_group"].where(meta["domain_group"] != "", meta["capture_id"]).to_numpy()
    unique_clusters = np.asarray(sorted(set(clusters)))
    by_cluster = {cluster: np.where(clusters == cluster)[0] for cluster in unique_clusters}
    rng = np.random.default_rng(SEED + seed_offset)
    samples = []
    for _ in range(iterations):
        chosen = rng.choice(unique_clusters, size=len(unique_clusters), replace=True)
        indices = np.concatenate([by_cluster[cluster] for cluster in chosen])
        if len(np.unique(labels[indices])) != 2:
            continue
        auc = binary_auc(labels[indices], scores[indices])
        samples.append(auc if orientation > 0 else 1.0 - auc)
    array = np.asarray(samples, dtype=float)
    point = binary_auc(labels, scores)
    point = point if orientation > 0 else 1.0 - point
    return point, float(np.quantile(array, 0.025)), float(np.quantile(array, 0.975)), len(array)


def top_feature_bootstrap(
    manifest: pd.DataFrame, features: pd.DataFrame, audit: pd.DataFrame, iterations: int
) -> pd.DataFrame:
    chosen = []
    for _, subset in audit.groupby("family", sort=True):
        chosen.extend(subset.sort_values("audit_score", ascending=False).head(2)["feature"].tolist())
    rows = []
    audit_by_feature = audit.set_index("feature")
    for feature_index, feature in enumerate(dict.fromkeys(chosen)):
        orientation = int(audit_by_feature.loc[feature, "orientation"])
        for carrier_index, carrier in enumerate(ALL_CARRIERS):
            point, low, high, completed = cluster_bootstrap(
                manifest,
                features[feature].to_numpy(dtype=float),
                carrier,
                orientation,
                iterations,
                feature_index * len(ALL_CARRIERS) + carrier_index,
            )
            rows.append(
                {
                    "feature": feature,
                    "family": feature_family(feature),
                    "carrier": carrier,
                    "oriented_auc": point,
                    "ci_low": low,
                    "ci_high": high,
                    "iterations": completed,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=500)
    parser.add_argument("--permutation-iterations", type=int, default=200)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"Refusing to overwrite existing output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    manifest, _, metadata, sequences = load_dataset(args.data_dir)
    features = residual_frame(sequences)
    audit = audit_features(manifest, features)
    families = family_summary(audit)
    observed = float(audit["dev_worst_oriented_auc"].max())
    null, permutation = permutation_null(manifest, features, observed, args.permutation_iterations)
    bootstrap = top_feature_bootstrap(manifest, features, audit, args.bootstrap_iterations)

    feature_output = pd.concat([manifest[["sample_id"]].reset_index(drop=True), features], axis=1)
    feature_output.to_csv(args.output_dir / "residual_features.csv", index=False)
    audit.to_csv(args.output_dir / "feature_audit.csv", index=False)
    families.to_csv(args.output_dir / "family_summary.csv", index=False)
    bootstrap.to_csv(args.output_dir / "top_feature_bootstrap.csv", index=False)
    null.to_csv(args.output_dir / "permutation_null.csv", index=False)

    passed_features = audit.loc[audit["candidate_pass"], "feature"].tolist()
    passed_families = families.loc[families["family_pass"], "family"].tolist()
    decision = {
        "selection_scope": "Exploratory 5,000-flow diagnostic; VLESS has already been consumed by v3/v4 and is not a final test.",
        "main_task": "Proxy-conditioned inner-Tor detection: ToP versus independent pure-proxy flows.",
        "uses_flow_pairs": False,
        "feature_gate": {
            "development_worst_oriented_auc_min": 0.58,
            "vless_oriented_auc_min": 0.58,
            "direction_agreement_all_carriers": True,
            "carrier_leakage_eta2_max": 0.20,
            "direct_tor_alignment_is_supporting_not_mandatory": True,
        },
        "permutation_gate": "familywise empirical p < 0.05 on the maximum development-carrier worst AUC",
        "passed_features": passed_features,
        "passed_families": passed_families,
        "permutation_summary": permutation,
        "proceed_to_hgb_probe": bool(
            passed_features and permutation["familywise_empirical_p"] < 0.05
        ),
    }
    (args.output_dir / "audit_decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    audit_metadata = {
        "seed": SEED,
        "row_count": len(manifest),
        "feature_count": features.shape[1],
        "feature_families": sorted(audit["family"].unique().tolist()),
        "packet_budget": int(sequences["packet_x"].shape[1]),
        "early_packet_budget": 25,
        "bootstrap_iterations": args.bootstrap_iterations,
        "permutation_iterations": args.permutation_iterations,
        "manifest_sha256": sha256(args.data_dir / "manifest.csv"),
        "sequences_sha256": sha256(args.data_dir / "sequences.npz"),
        "source_manifest_sha256_from_builder": metadata["manifest_sha256"],
        "sii_fields_used": [],
        "input_channels": ["direction", "log1p_tcp_payload_length", "log1p_clipped_iat", "valid_mask"],
    }
    (args.output_dir / "audit_metadata.json").write_text(
        json.dumps(audit_metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"metadata": audit_metadata, "decision": decision}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
