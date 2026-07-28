from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np


EARLY_PACKETS = 25
CELL_SCALES = (498, 509, 512, 514)
COARSE_LENGTH_EDGES = np.asarray([256, 512, 768, 1024, 1280, 1500], dtype=float)
SIGNED_TRIGRAM_BUCKETS = 24


def _safe_entropy(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    total = float(np.sum(values))
    if total <= 0.0:
        return 0.0
    probabilities = values[values > 0.0] / total
    if len(probabilities) <= 1:
        return 0.0
    return float(-np.sum(probabilities * np.log(probabilities)) / math.log(len(values)))


def _safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if len(left) < 3 or len(right) < 3 or np.std(left) < 1e-9 or np.std(right) < 1e-9:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _phase_concentration(values: np.ndarray, period: float) -> float:
    values = np.asarray(values, dtype=float)
    if not len(values):
        return 0.0
    phase = 2.0 * np.pi * values / period
    return float(np.abs(np.mean(np.exp(1j * phase))))


def _nearest_period_residual(values: np.ndarray, period: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    remainder = np.mod(values, period)
    return np.minimum(remainder, period - remainder)


def _burst_arrays(
    directions: np.ndarray, lengths: np.ndarray, iats_ms: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not len(lengths):
        empty = np.asarray([], dtype=float)
        return empty, empty, empty, empty
    burst_directions: list[float] = []
    burst_bytes: list[float] = []
    burst_packets: list[float] = []
    burst_gaps: list[float] = []
    start = 0
    for index in range(1, len(lengths) + 1):
        if index == len(lengths) or directions[index] != directions[start]:
            burst_directions.append(float(directions[start]))
            burst_bytes.append(float(np.sum(lengths[start:index])))
            burst_packets.append(float(index - start))
            burst_gaps.append(float(iats_ms[index]) if index < len(lengths) else 0.0)
            start = index
    return (
        np.asarray(burst_directions),
        np.asarray(burst_bytes),
        np.asarray(burst_packets),
        np.asarray(burst_gaps),
    )


def _early_features(directions: np.ndarray, lengths: np.ndarray, iats_ms: np.ndarray) -> dict[str, float]:
    directions = directions[:EARLY_PACKETS]
    lengths = lengths[:EARLY_PACKETS]
    iats_ms = iats_ms[:EARLY_PACKETS]
    features: dict[str, float] = {}
    count = len(lengths)
    if not count:
        return {"early__empty": 1.0}

    switches = np.diff(directions) != 0 if count > 1 else np.asarray([], dtype=bool)
    negative_positions = np.where(directions < 0)[0]
    burst_directions, burst_bytes, burst_packets, _ = _burst_arrays(directions, lengths, iats_ms)
    features.update(
        {
            "early__empty": 0.0,
            "early__fwd_fraction": float(np.mean(directions > 0)),
            "early__switch_rate": float(np.mean(switches)) if len(switches) else 0.0,
            "early__roundtrip_density": float(max(len(burst_directions) - 1, 0) / max(count - 1, 1)),
            "early__first_response_position": (
                float(negative_positions[0] / max(count - 1, 1)) if len(negative_positions) else 1.0
            ),
            "early__length_cv": float(np.std(lengths) / max(np.mean(lengths), 1.0)),
            "early__iat_cv": float(np.std(iats_ms) / max(np.mean(iats_ms), 1.0)),
            "early__burst_count_norm": float(len(burst_bytes) / count),
            "early__burst_bytes_cv": float(np.std(burst_bytes) / max(np.mean(burst_bytes), 1.0)),
            "early__burst_packets_cv": float(np.std(burst_packets) / max(np.mean(burst_packets), 1.0)),
        }
    )

    length_bins = np.digitize(lengths, COARSE_LENGTH_EDGES, right=False)
    length_hist = np.bincount(length_bins, minlength=len(COARSE_LENGTH_EDGES) + 1).astype(float)
    length_hist /= max(float(np.sum(length_hist)), 1.0)
    for index, value in enumerate(length_hist):
        features[f"early__coarse_length_bin_{index}"] = float(value)
    features["early__coarse_length_entropy"] = _safe_entropy(length_hist)

    direction_bits = (directions < 0).astype(int)
    direction_trigrams = np.zeros(8, dtype=float)
    signed_trigrams = np.zeros(SIGNED_TRIGRAM_BUCKETS, dtype=float)
    signed_tokens = direction_bits * (len(COARSE_LENGTH_EDGES) + 1) + length_bins
    for index in range(max(count - 2, 0)):
        direction_code = int(direction_bits[index] * 4 + direction_bits[index + 1] * 2 + direction_bits[index + 2])
        direction_trigrams[direction_code] += 1.0
        a, b, c = (int(value) for value in signed_tokens[index:index + 3])
        bucket = (a * 131 + b * 31 + c * 7) % SIGNED_TRIGRAM_BUCKETS
        signed_trigrams[bucket] += 1.0
    direction_trigrams /= max(float(np.sum(direction_trigrams)), 1.0)
    signed_trigrams /= max(float(np.sum(signed_trigrams)), 1.0)
    for index, value in enumerate(direction_trigrams):
        features[f"early__direction_trigram_{index:02d}"] = float(value)
    for index, value in enumerate(signed_trigrams):
        features[f"early__signed_coarse_trigram_{index:02d}"] = float(value)
    features["early__direction_trigram_entropy"] = _safe_entropy(direction_trigrams)
    features["early__signed_coarse_trigram_entropy"] = _safe_entropy(signed_trigrams)
    return features


def _cell_features(directions: np.ndarray, lengths: np.ndarray, iats_ms: np.ndarray) -> dict[str, float]:
    directions = directions[EARLY_PACKETS:]
    lengths = lengths[EARLY_PACKETS:]
    iats_ms = iats_ms[EARLY_PACKETS:]
    _, burst_bytes, _, _ = _burst_arrays(directions, lengths, iats_ms)
    features: dict[str, float] = {"cell__steady_empty": float(len(lengths) == 0)}
    packet_concentrations = []
    cumulative_concentrations = []
    for scale in CELL_SCALES:
        cumulative = np.cumsum(lengths)
        packet_residual = _nearest_period_residual(lengths, scale)
        burst_residual = _nearest_period_residual(burst_bytes, scale)
        packet_concentration = _phase_concentration(lengths, scale)
        cumulative_concentration = _phase_concentration(cumulative, scale)
        packet_concentrations.append(packet_concentration)
        cumulative_concentrations.append(cumulative_concentration)
        prefix = f"cell__q{scale}"
        features.update(
            {
                f"{prefix}_packet_phase_concentration": packet_concentration,
                f"{prefix}_cumulative_phase_concentration": cumulative_concentration,
                f"{prefix}_burst_phase_concentration": _phase_concentration(burst_bytes, scale),
                f"{prefix}_packet_residual_mean": (
                    float(np.mean(packet_residual) / scale) if len(packet_residual) else 0.0
                ),
                f"{prefix}_burst_residual_mean": (
                    float(np.mean(burst_residual) / scale) if len(burst_residual) else 0.0
                ),
                f"{prefix}_packet_soft_closeness": (
                    float(np.mean(np.exp(-np.square(packet_residual / 32.0)))) if len(packet_residual) else 0.0
                ),
                f"{prefix}_burst_soft_closeness": (
                    float(np.mean(np.exp(-np.square(burst_residual / 64.0)))) if len(burst_residual) else 0.0
                ),
            }
        )
    features["cell__packet_scale_contrast"] = float(np.ptp(packet_concentrations))
    features["cell__cumulative_scale_contrast"] = float(np.ptp(cumulative_concentrations))
    return features


def _temporal_histograms(
    directions: np.ndarray, lengths: np.ndarray, iats_ms: np.ndarray, bins: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not len(lengths):
        zeros = np.zeros(bins, dtype=float)
        return zeros, zeros.copy(), zeros.copy(), zeros.copy()
    times = np.cumsum(np.maximum(iats_ms, 0.0))
    if times[-1] <= 1e-9:
        times = np.arange(len(lengths), dtype=float)
    denominator = max(float(times[-1]), 1.0)
    positions = np.minimum((times / denominator * bins).astype(int), bins - 1)
    fwd_count = np.zeros(bins, dtype=float)
    bwd_count = np.zeros(bins, dtype=float)
    fwd_bytes = np.zeros(bins, dtype=float)
    bwd_bytes = np.zeros(bins, dtype=float)
    for position, direction, length in zip(positions, directions, lengths):
        if direction > 0:
            fwd_count[position] += 1.0
            fwd_bytes[position] += float(length)
        else:
            bwd_count[position] += 1.0
            bwd_bytes[position] += float(length)
    return fwd_count, bwd_count, fwd_bytes, bwd_bytes


def _shape_summary(prefix: str, fwd: np.ndarray, bwd: np.ndarray) -> dict[str, float]:
    total = fwd + bwd
    signed = fwd - bwd
    positions = (np.arange(len(total), dtype=float) + 0.5) / len(total)
    total_sum = float(np.sum(total))
    normalized = total / total_sum if total_sum > 0 else np.zeros_like(total)
    center = float(np.sum(positions * normalized))
    spread = float(np.sqrt(np.sum(np.square(positions - center) * normalized)))
    cosine_denominator = float(np.linalg.norm(fwd) * np.linalg.norm(bwd))
    return {
        f"{prefix}_active_fraction": float(np.mean(total > 0)),
        f"{prefix}_entropy": _safe_entropy(total),
        f"{prefix}_center": center,
        f"{prefix}_spread": spread,
        f"{prefix}_burstiness": float(np.std(total) / max(np.mean(total), 1e-9)),
        f"{prefix}_total_variation": float(np.sum(np.abs(np.diff(total))) / max(total_sum, 1.0)),
        f"{prefix}_signed_variation": float(np.sum(np.abs(np.diff(signed))) / max(total_sum, 1.0)),
        f"{prefix}_direction_overlap": float(np.sum(np.minimum(fwd, bwd)) / max(total_sum, 1.0)),
        f"{prefix}_direction_cosine": (
            float(np.dot(fwd, bwd) / cosine_denominator) if cosine_denominator > 0 else 0.0
        ),
    }


def _multires_features(directions: np.ndarray, lengths: np.ndarray, iats_ms: np.ndarray) -> dict[str, float]:
    features: dict[str, float] = {}
    for bins in (4, 8, 16):
        fwd_count, bwd_count, fwd_bytes, bwd_bytes = _temporal_histograms(
            directions, lengths, iats_ms, bins
        )
        features.update(_shape_summary(f"multires__b{bins}_count", fwd_count, bwd_count))
        features.update(_shape_summary(f"multires__b{bins}_bytes", fwd_bytes, bwd_bytes))
    return features


def _flow_control_features(
    directions: np.ndarray, lengths: np.ndarray, iats_ms: np.ndarray
) -> dict[str, float]:
    _, burst_bytes, burst_packets, burst_gaps = _burst_arrays(directions, lengths, iats_ms)
    cell_equivalents = burst_bytes / 498.0
    cumulative_cells = np.cumsum(cell_equivalents)
    total_cells = float(cumulative_cells[-1]) if len(cumulative_cells) else 0.0

    def boundary_score(boundary: float) -> float:
        if not len(cumulative_cells):
            return 0.0
        distance = float(np.min(np.abs(cumulative_cells - boundary)))
        return float(math.exp(-((distance / 10.0) ** 2)))

    return {
        "flow_control__burst_cell_cv": (
            float(np.std(cell_equivalents) / max(np.mean(cell_equivalents), 1e-9))
            if len(cell_equivalents) else 0.0
        ),
        "flow_control__burst_packet_cv": (
            float(np.std(burst_packets) / max(np.mean(burst_packets), 1e-9))
            if len(burst_packets) else 0.0
        ),
        "flow_control__cell_autocorr_lag1": _safe_corr(cell_equivalents[:-1], cell_equivalents[1:]),
        "flow_control__cells_to_next_gap_corr": _safe_corr(cell_equivalents[:-1], burst_gaps[:-1]),
        "flow_control__large_burst_fraction_ge25": (
            float(np.mean(cell_equivalents >= 25.0)) if len(cell_equivalents) else 0.0
        ),
        "flow_control__large_burst_fraction_ge50": (
            float(np.mean(cell_equivalents >= 50.0)) if len(cell_equivalents) else 0.0
        ),
        "flow_control__boundary50_soft": boundary_score(50.0),
        "flow_control__boundary100_soft": boundary_score(100.0),
        "flow_control__reaches50": float(total_cells >= 50.0),
        "flow_control__reaches100": float(total_cells >= 100.0),
    }


def flow_residual_features(packet_x: np.ndarray, packet_mask: np.ndarray) -> dict[str, float]:

    valid = np.asarray(packet_mask, dtype=bool)
    packet_x = np.asarray(packet_x, dtype=float)[valid]
    if packet_x.ndim != 2 or (len(packet_x) and packet_x.shape[1] != 3):
        raise ValueError("packet_x must contain direction, log-length, and log-IAT columns")
    directions = np.where(packet_x[:, 0] >= 0.0, 1.0, -1.0)
    lengths = np.maximum(np.rint(np.expm1(packet_x[:, 1])), 0.0)
    iats_ms = np.maximum(np.expm1(packet_x[:, 2]), 0.0)
    features: dict[str, float] = {}
    features.update(_early_features(directions, lengths, iats_ms))
    features.update(_cell_features(directions, lengths, iats_ms))
    features.update(_multires_features(directions, lengths, iats_ms))
    features.update(_flow_control_features(directions, lengths, iats_ms))
    values = np.asarray(list(features.values()), dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Protocol residual extraction produced non-finite values")
    return features


def feature_family(feature: str) -> str:
    family = feature.split("__", 1)[0]
    if family not in {"early", "cell", "multires", "flow_control"}:
        raise ValueError(f"Unknown residual family for {feature}")
    return family


def eta_squared(values: Iterable[float], groups: Iterable[str]) -> float:
    values_array = np.asarray(list(values), dtype=float)
    groups_array = np.asarray(list(groups))
    if not len(values_array) or np.var(values_array) < 1e-12:
        return 0.0
    overall = float(np.mean(values_array))
    between = 0.0
    for group in np.unique(groups_array):
        subset = values_array[groups_array == group]
        between += len(subset) * float((np.mean(subset) - overall) ** 2)
    total = float(np.sum(np.square(values_array - overall)))
    return float(between / total) if total > 0.0 else 0.0
