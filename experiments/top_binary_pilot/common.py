from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)


SEED = 42
MAX_FLOWS_PER_CAPTURE = 5
MAX_PACKETS = 128
MAX_BURSTS = 64


QUOTAS: dict[tuple[str, str, str], int] = {
    ("train", "top", "ss"): 400,
    ("train", "top", "trojan"): 400,
    ("train", "top", "vmess"): 400,
    ("train", "pure_proxy", "ss"): 200,
    ("train", "pure_proxy", "trojan"): 200,
    ("train", "pure_proxy", "vmess"): 200,
    ("train", "normal", "none"): 300,
    ("train", "direct_tor", "direct"): 300,
    ("validation", "top", "ss"): 134,
    ("validation", "top", "trojan"): 133,
    ("validation", "top", "vmess"): 133,
    ("validation", "pure_proxy", "ss"): 68,
    ("validation", "pure_proxy", "trojan"): 66,
    ("validation", "pure_proxy", "vmess"): 66,
    ("validation", "normal", "none"): 100,
    ("validation", "direct_tor", "direct"): 100,
    ("test", "top", "ss"): 133,
    ("test", "top", "trojan"): 134,
    ("test", "top", "vmess"): 133,
    ("test", "pure_proxy", "ss"): 66,
    ("test", "pure_proxy", "trojan"): 68,
    ("test", "pure_proxy", "vmess"): 66,
    ("test", "normal", "none"): 100,
    ("test", "direct_tor", "direct"): 100,
    ("vless_test", "top", "vless"): 500,
    ("vless_test", "pure_proxy", "vless"): 300,
    ("vless_test", "normal", "none"): 100,
    ("vless_test", "direct_tor", "direct"): 100,
}


def stable_hash(value: str, seed: int = SEED) -> int:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def assigned_split(group_key: str) -> str:

    bucket = stable_hash(group_key) % 100
    if bucket < 50:
        return "train"
    if bucket < 60:
        return "validation"
    if bucket < 70:
        return "test"
    return "vless_test"


def background_group(kind: str) -> str:
    return {
        "pure_proxy": "pure_proxy",
        "normal": "normal",
        "direct_tor": "direct_tor",
        "top": "positive",
    }[kind]


def binary_label(kind: str) -> int:
    return int(kind == "top")


@dataclass
class Packet:
    timestamp: float
    src_ip: str
    src_port: int
    dst_ip: str
    dst_port: int
    payload_len: int
    frame_len: int
    flags: int

    @property
    def syn(self) -> bool:
        return bool(self.flags & 0x02)

    @property
    def ack(self) -> bool:
        return bool(self.flags & 0x10)


def endpoint(ip: str, port: int) -> tuple[str, int]:
    return ip, int(port)


def canonical_flow_key(packet: Packet) -> tuple[tuple[str, int], tuple[str, int]]:
    left = endpoint(packet.src_ip, packet.src_port)
    right = endpoint(packet.dst_ip, packet.dst_port)
    return tuple(sorted((left, right)))


def flow_key_text(key: tuple[tuple[str, int], tuple[str, int]]) -> str:
    return f"{key[0][0]}:{key[0][1]}-{key[1][0]}:{key[1][1]}-tcp"


def _safe_stats(values: Iterable[float], prefix: str) -> dict[str, float]:
    array = np.asarray(list(values), dtype=float)
    if not len(array):
        return {
            f"{prefix}_min": 0.0,
            f"{prefix}_max": 0.0,
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_q25": 0.0,
            f"{prefix}_q50": 0.0,
            f"{prefix}_q75": 0.0,
        }
    return {
        f"{prefix}_min": float(np.min(array)),
        f"{prefix}_max": float(np.max(array)),
        f"{prefix}_mean": float(np.mean(array)),
        f"{prefix}_std": float(np.std(array)),
        f"{prefix}_q25": float(np.quantile(array, 0.25)),
        f"{prefix}_q50": float(np.quantile(array, 0.50)),
        f"{prefix}_q75": float(np.quantile(array, 0.75)),
    }


def _initiator_and_data(packets: list[Packet]) -> tuple[tuple[str, int], list[Packet]]:
    initiator = endpoint(packets[0].src_ip, packets[0].src_port)
    syn_index = None
    synack_index = None
    ack_index = None
    for index, packet in enumerate(packets):
        if packet.syn and not packet.ack:
            initiator = endpoint(packet.src_ip, packet.src_port)
            syn_index = index
            break
    if syn_index is not None:
        for index in range(syn_index + 1, len(packets)):
            packet = packets[index]
            if packet.syn and packet.ack:
                synack_index = index
                break
    if synack_index is not None:
        for index in range(synack_index + 1, len(packets)):
            packet = packets[index]
            if packet.ack and not packet.syn and endpoint(packet.src_ip, packet.src_port) == initiator:
                ack_index = index
                break
    start = (ack_index + 1) if ack_index is not None else 0
    data_packets = [packet for packet in packets[start:] if packet.payload_len > 0]
    return initiator, data_packets


def valid_flow(packets: list[Packet]) -> bool:
    if len(packets) < 4:
        return False
    endpoints = {endpoint(p.src_ip, p.src_port) for p in packets}
    if len(endpoints) != 2:
        return False
    _, data_packets = _initiator_and_data(packets)
    if len(data_packets) < 2:
        return False
    directions = {
        endpoint(packet.src_ip, packet.src_port)
        for packet in packets
    }
    return len(directions) == 2


def _direction(packet: Packet, initiator: tuple[str, int]) -> float:
    return 1.0 if endpoint(packet.src_ip, packet.src_port) == initiator else -1.0


def _burst_records(data_packets: list[Packet], initiator: tuple[str, int]) -> list[dict[str, float]]:
    bursts: list[list[Packet]] = []
    for packet in data_packets:
        direction = _direction(packet, initiator)
        if not bursts or _direction(bursts[-1][-1], initiator) != direction:
            bursts.append([packet])
        else:
            bursts[-1].append(packet)
    records: list[dict[str, float]] = []
    for index, burst in enumerate(bursts):
        lengths = np.asarray([packet.payload_len for packet in burst], dtype=float)
        duration_ms = max((burst[-1].timestamp - burst[0].timestamp) * 1000.0, 0.0)
        next_gap_ms = 0.0
        if index + 1 < len(bursts):
            next_gap_ms = max((bursts[index + 1][0].timestamp - burst[-1].timestamp) * 1000.0, 0.0)
        records.append(
            {
                "direction": _direction(burst[0], initiator),
                "packet_count": float(len(burst)),
                "payload_bytes": float(np.sum(lengths)),
                "payload_mean": float(np.mean(lengths)),
                "duration_ms": duration_ms,
                "next_gap_ms": next_gap_ms,
            }
        )
    return records


def tensorize_flow(packets: list[Packet]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    initiator, data_packets = _initiator_and_data(packets)
    packet_x = np.zeros((MAX_PACKETS, 3), dtype=np.float32)
    packet_mask = np.zeros(MAX_PACKETS, dtype=np.uint8)
    previous = data_packets[0].timestamp if data_packets else 0.0
    for index, packet in enumerate(data_packets[:MAX_PACKETS]):
        delta_ms = max((packet.timestamp - previous) * 1000.0, 0.0) if index else 0.0
        previous = packet.timestamp
        packet_x[index] = (
            _direction(packet, initiator),
            math.log1p(packet.payload_len),
            math.log1p(min(delta_ms, 10_000.0)),
        )
        packet_mask[index] = 1

    bursts = _burst_records(data_packets, initiator)
    burst_x = np.zeros((MAX_BURSTS, 6), dtype=np.float32)
    burst_mask = np.zeros(MAX_BURSTS, dtype=np.uint8)
    for index, burst in enumerate(bursts[:MAX_BURSTS]):
        burst_x[index] = (
            burst["direction"],
            math.log1p(burst["packet_count"]),
            math.log1p(burst["payload_bytes"]),
            math.log1p(burst["payload_mean"]),
            math.log1p(min(burst["duration_ms"], 10_000.0)),
            math.log1p(min(burst["next_gap_ms"], 10_000.0)),
        )
        burst_mask[index] = 1
    return packet_x, packet_mask, burst_x, burst_mask


def _segment_features(
    packets: list[Packet], initiator: tuple[str, int], prefix: str
) -> dict[str, float]:
    if not packets:
        return {
            f"{prefix}_packet_count": 0.0,
            f"{prefix}_payload_bytes": 0.0,
            f"{prefix}_fwd_ratio": 0.0,
            f"{prefix}_switch_rate": 0.0,
            f"{prefix}_iat_mean_ms": 0.0,
            f"{prefix}_iat_std_ms": 0.0,
            f"{prefix}_payload_mean": 0.0,
            f"{prefix}_payload_std": 0.0,
            f"{prefix}_payload_q50": 0.0,
            f"{prefix}_burst_count": 0.0,
        }
    directions = np.asarray([_direction(packet, initiator) for packet in packets])
    lengths = np.asarray([packet.payload_len for packet in packets], dtype=float)
    iats = np.diff([packet.timestamp for packet in packets]) * 1000.0
    switches = np.sum(directions[1:] != directions[:-1]) if len(directions) > 1 else 0
    return {
        f"{prefix}_packet_count": float(len(packets)),
        f"{prefix}_payload_bytes": float(np.sum(lengths)),
        f"{prefix}_fwd_ratio": float(np.mean(directions > 0)),
        f"{prefix}_switch_rate": float(switches / max(len(directions) - 1, 1)),
        f"{prefix}_iat_mean_ms": float(np.mean(iats)) if len(iats) else 0.0,
        f"{prefix}_iat_std_ms": float(np.std(iats)) if len(iats) else 0.0,
        f"{prefix}_payload_mean": float(np.mean(lengths)),
        f"{prefix}_payload_std": float(np.std(lengths)),
        f"{prefix}_payload_q50": float(np.quantile(lengths, 0.5)),
        f"{prefix}_burst_count": float(len(_burst_records(packets, initiator))),
    }


def past_features(packets: list[Packet]) -> tuple[dict[str, float], list[str]]:

    initiator, data_packets = _initiator_and_data(packets)
    directions = np.asarray([_direction(packet, initiator) for packet in data_packets])
    payload_lengths = np.asarray([packet.payload_len for packet in data_packets], dtype=float)
    frame_lengths = np.asarray([packet.frame_len for packet in packets], dtype=float)
    packet_iats_ms = np.diff([packet.timestamp for packet in packets]) * 1000.0
    data_iats_ms = np.diff([packet.timestamp for packet in data_packets]) * 1000.0
    duration_ms = max((packets[-1].timestamp - packets[0].timestamp) * 1000.0, 0.0)
    switches = np.sum(directions[1:] != directions[:-1]) if len(directions) > 1 else 0
    fwd = directions > 0
    bwd = directions < 0
    bursts = _burst_records(data_packets, initiator)

    features: dict[str, float] = {
        "packet_count": float(len(packets)),
        "payload_packet_count": float(len(data_packets)),
        "total_payload_bytes": float(np.sum(payload_lengths)),
        "duration_ms": duration_ms,
        "flow_bytes_per_ms": float(np.sum(payload_lengths) / max(duration_ms, 1e-6)),
        "flow_packets_per_ms": float(len(data_packets) / max(duration_ms, 1e-6)),
        "fwd_packet_count": float(np.sum(fwd)),
        "bwd_packet_count": float(np.sum(bwd)),
        "fwd_payload_bytes": float(np.sum(payload_lengths[fwd])),
        "bwd_payload_bytes": float(np.sum(payload_lengths[bwd])),
        "down_up_ratio": float(np.sum(bwd) / max(np.sum(fwd), 1)),
        "direction_switch_rate": float(switches / max(len(directions) - 1, 1)),
        "burst_count": float(len(bursts)),
    }
    features.update(_safe_stats(payload_lengths, "payload_len"))
    features.update(_safe_stats(frame_lengths, "frame_len"))
    features.update(_safe_stats(packet_iats_ms, "packet_iat_ms"))
    features.update(_safe_stats(data_iats_ms, "data_iat_ms"))
    features.update(_safe_stats((b["packet_count"] for b in bursts), "burst_packets"))
    features.update(_safe_stats((b["payload_bytes"] for b in bursts), "burst_bytes"))
    features.update(_safe_stats((b["duration_ms"] for b in bursts), "burst_duration_ms"))
    base_features = list(features)

    segments = np.array_split(np.arange(len(data_packets)), 3)
    names = ("early", "middle", "late")
    for name, indices in zip(names, segments):
        segment = [data_packets[int(index)] for index in indices]
        features.update(_segment_features(segment, initiator, name))
    for metric in (
        "payload_bytes", "fwd_ratio", "switch_rate", "iat_mean_ms",
        "payload_mean", "payload_std", "burst_count",
    ):
        features[f"delta_late_early_{metric}"] = (
            features[f"late_{metric}"] - features[f"early_{metric}"]
        )
    return features, base_features


def jsd_score(values: pd.Series, labels: pd.Series, bins: int = 32) -> float:
    numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if numeric.nunique() < 2 or labels.nunique() < 2:
        return 0.0
    low, high = float(numeric.min()), float(numeric.max())
    if low == high:
        return 0.0
    edges = np.linspace(low, high, bins + 1)
    distributions = []
    for label in sorted(labels.unique()):
        histogram, _ = np.histogram(numeric[labels == label], bins=edges)
        probability = histogram.astype(float) + 1e-9
        distributions.append(probability / probability.sum())
    if len(distributions) != 2:
        return 0.0
    return float(jensenshannon(distributions[0], distributions[1]))


def carrier_stable_ranking(X: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:

    rows = []
    for feature in X.columns:
        utilities = []
        for carrier in ("ss", "trojan", "vmess"):
            mask = (
                ((meta["background_group"] == "positive") & (meta["carrier"] == carrier))
                | ((meta["background_group"] == "pure_proxy") & (meta["carrier"] == carrier))
            )
            target = (meta.loc[mask, "background_group"] == "positive").astype(int)
            utilities.append(jsd_score(X.loc[mask, feature], target))
        for group in ("normal", "direct_tor"):
            mask = (meta["background_group"] == "positive") | (meta["background_group"] == group)
            target = (meta.loc[mask, "background_group"] == "positive").astype(int)
            utilities.append(jsd_score(X.loc[mask, feature], target))

        carrier_values = []
        positive = meta["background_group"] == "positive"
        for left, right in (("ss", "trojan"), ("ss", "vmess"), ("trojan", "vmess")):
            mask = positive & meta["carrier"].isin([left, right])
            target = (meta.loc[mask, "carrier"] == right).astype(int)
            carrier_values.append(jsd_score(X.loc[mask, feature], target))
        mean_utility = float(np.mean(utilities))
        utility_std = float(np.std(utilities))
        carrier_leakage = float(np.mean(carrier_values))
        score = mean_utility - 0.5 * utility_std - 0.25 * carrier_leakage
        rows.append(
            {
                "feature": feature,
                "stable_score": score,
                "mean_utility": mean_utility,
                "utility_std": utility_std,
                "carrier_leakage": carrier_leakage,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["stable_score", "mean_utility", "feature"], ascending=[False, False, True]
    ).reset_index(drop=True)


def metric_row(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    prediction = (probabilities >= threshold).astype(int)
    row = {
        "macro_f1": float(f1_score(y_true, prediction, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "auprc": float(average_precision_score(y_true, probabilities)),
        "auroc": float(roc_auc_score(y_true, probabilities)) if len(np.unique(y_true)) == 2 else float("nan"),
    }
    if len(np.unique(y_true)) == 2:
        fpr, tpr, _ = roc_curve(y_true, probabilities)
        qualifying = np.where(tpr >= 0.95)[0]
        row["fpr_at_95_tpr"] = float(np.min(fpr[qualifying])) if len(qualifying) else 1.0
    else:
        row["fpr_at_95_tpr"] = float("nan")
    return row


def group_scores(meta: pd.DataFrame, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    scores: dict[str, float] = {}
    labels = meta["binary_label"].to_numpy(dtype=int)
    positive = labels == 1
    for group in ("pure_proxy", "normal", "direct_tor"):
        mask = positive | (meta["background_group"].to_numpy() == group)
        if len(np.unique(labels[mask])) == 2:
            scores[f"background_{group}"] = metric_row(labels[mask], probabilities[mask], threshold)["macro_f1"]
    for carrier in ("ss", "trojan", "vmess"):
        mask = ((positive) & (meta["carrier"].to_numpy() == carrier)) | (~positive)
        if len(np.unique(labels[mask])) == 2:
            scores[f"carrier_{carrier}"] = metric_row(labels[mask], probabilities[mask], threshold)["macro_f1"]
    return scores


def select_threshold(meta: pd.DataFrame, probabilities: np.ndarray) -> tuple[float, float, dict[str, float]]:
    best: tuple[float, float, dict[str, float]] | None = None
    labels = meta["binary_label"].to_numpy(dtype=int)
    for threshold in np.linspace(0.10, 0.90, 81):
        scores = group_scores(meta, probabilities, float(threshold))
        worst = min(scores.values()) if scores else metric_row(labels, probabilities, float(threshold))["macro_f1"]
        candidate = (float(threshold), float(worst), scores)
        if best is None or (candidate[1], -abs(candidate[0] - 0.5)) > (best[1], -abs(best[0] - 0.5)):
            best = candidate
    assert best is not None
    return best


def false_positive_rates(meta: pd.DataFrame, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    prediction = probabilities >= threshold
    rows = {}
    for group in ("pure_proxy", "normal", "direct_tor"):
        mask = meta["background_group"].to_numpy() == group
        rows[group] = float(np.mean(prediction[mask])) if np.any(mask) else float("nan")
    return rows
