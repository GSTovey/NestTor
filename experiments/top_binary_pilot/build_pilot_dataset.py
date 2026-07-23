#!/usr/bin/env python3
"""Incrementally build the fixed 5,000-flow ToP pilot dataset from pcaps."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from common import (
    MAX_FLOWS_PER_CAPTURE,
    QUOTAS,
    SEED,
    Packet,
    assigned_split,
    background_group,
    binary_label,
    canonical_flow_key,
    flow_key_text,
    past_features,
    stable_hash,
    tensorize_flow,
    valid_flow,
)


@dataclass(frozen=True)
class Candidate:
    pcap_path: Path
    kind: str
    carrier: str
    source_dataset: str
    capture_id: str
    domain_group: str
    split_group_key: str


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_int(value: str, default: int = 0) -> int:
    text = (value or "").strip()
    if not text:
        return default
    try:
        return int(text, 0)
    except ValueError:
        try:
            return int(float(text))
        except ValueError:
            return default


def parse_pcap(path: Path, tshark: str, timeout: int) -> dict[str, list[Packet]]:
    fields = (
        "frame.time_epoch", "ip.src", "tcp.srcport", "ip.dst", "tcp.dstport",
        "tcp.len", "frame.len", "tcp.flags",
    )
    command = [
        tshark, "-n", "-r", str(path), "-Y", "tcp && ip",
        "-T", "fields", "-E", "separator=\t", "-E", "quote=n", "-E", "occurrence=f",
    ]
    for field in fields:
        command.extend(["-e", field])
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"tshark failed for {path}: {completed.stderr.strip()[:500]}")

    flows: dict[object, list[Packet]] = defaultdict(list)
    reader = csv.reader(completed.stdout.splitlines(), delimiter="\t")
    for row in reader:
        if len(row) < len(fields):
            row += [""] * (len(fields) - len(row))
        timestamp, src_ip, src_port, dst_ip, dst_port, payload_len, frame_len, flags = row[:8]
        if not timestamp or not src_ip or not dst_ip or not src_port or not dst_port:
            continue
        try:
            packet = Packet(
                timestamp=float(timestamp),
                src_ip=src_ip,
                src_port=int(src_port),
                dst_ip=dst_ip,
                dst_port=int(dst_port),
                payload_len=max(parse_int(payload_len), 0),
                frame_len=max(parse_int(frame_len), 0),
                flags=parse_int(flags),
            )
        except ValueError:
            continue
        flows[canonical_flow_key(packet)].append(packet)
    return {
        flow_key_text(key): sorted(packets, key=lambda packet: packet.timestamp)
        for key, packets in flows.items()
    }


def new_candidates(root: Path) -> list[Candidate]:
    rows: list[Candidate] = []
    for carrier in ("ss", "trojan", "vmess", "vless"):
        specifications = (
            ("top", "pcap", "stage3_clash.pcap"),
            ("pure_proxy", "pcap-clash_only", "stage2_clash.pcap"),
        )
        for kind, subdirectory, filename in specifications:
            for path in sorted((root / carrier / subdirectory).glob(f"*/{filename}")):
                domain = path.parent.name
                rows.append(
                    Candidate(
                        pcap_path=path,
                        kind=kind,
                        carrier=carrier,
                        source_dataset="continual_flowcrawl",
                        capture_id=f"{carrier}:{subdirectory}:{domain}",
                        domain_group=domain,
                        split_group_key=f"domain:{domain}",
                    )
                )
    return rows


def background_candidates(root: Path) -> list[Candidate]:
    """Use independent stages from the same crawl corpus to avoid source shortcuts."""
    rows: list[Candidate] = []
    canonical_root = root / "ss" / "pcap"
    for kind, filename, carrier in (
        ("normal", "stage1_raw.pcap", "none"),
        ("direct_tor", "stage2_tor.pcap", "direct"),
    ):
        for path in sorted(canonical_root.glob(f"*/{filename}")):
            domain = path.parent.name
            capture_id = f"continual:{kind}:{domain}"
            rows.append(
                Candidate(
                    pcap_path=path,
                    kind=kind,
                    carrier=carrier,
                    source_dataset="continual_flowcrawl",
                    capture_id=capture_id,
                    domain_group=domain,
                    split_group_key=f"domain:{domain}",
                )
            )
    return rows


def candidate_key(candidate: Candidate) -> tuple[str, str, str]:
    return assigned_split(candidate.split_group_key), candidate.kind, candidate.carrier


def inventory(candidates: list[Candidate]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for candidate in candidates:
        split, kind, carrier = candidate_key(candidate)
        counts[f"{split}/{kind}/{carrier}"] += 1
    return dict(sorted(counts.items()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tshark", default="tshark")
    parser.add_argument("--pcap-timeout", type=int, default=120)
    parser.add_argument("--inventory-only", action="store_true")
    args = parser.parse_args()

    candidates = new_candidates(args.new_root) + background_candidates(args.new_root)
    candidate_inventory = inventory(candidates)
    print(json.dumps({"candidate_captures": candidate_inventory}, ensure_ascii=False, indent=2))
    if args.inventory_only:
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    grouped: dict[tuple[str, str, str], list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        key = candidate_key(candidate)
        if key in QUOTAS:
            grouped[key].append(candidate)
    for rows in grouped.values():
        rows.sort(key=lambda candidate: (stable_hash(str(candidate.pcap_path)), str(candidate.pcap_path)))

    manifest_rows: list[dict[str, object]] = []
    feature_rows: list[dict[str, object]] = []
    packet_tensors: list[np.ndarray] = []
    packet_masks: list[np.ndarray] = []
    burst_tensors: list[np.ndarray] = []
    burst_masks: list[np.ndarray] = []
    base_feature_names: list[str] | None = None
    failures: list[dict[str, str]] = []
    actual: dict[str, int] = {}
    pcap_hash_cache: dict[Path, str] = {}

    for quota_key, target in QUOTAS.items():
        split, kind, carrier = quota_key
        selected = 0
        quota_name = f"{split}/{kind}/{carrier}"
        print(f"[quota] {quota_name}: target={target}, candidates={len(grouped.get(quota_key, []))}", flush=True)
        for candidate in grouped.get(quota_key, []):
            if selected >= target:
                break
            try:
                parsed = parse_pcap(candidate.pcap_path, args.tshark, args.pcap_timeout)
            except (RuntimeError, subprocess.TimeoutExpired) as error:
                failures.append({"pcap_path": str(candidate.pcap_path), "error": str(error)})
                print(f"[warning] {error}", flush=True)
                continue
            valid = [(flow_id, packets) for flow_id, packets in parsed.items() if valid_flow(packets)]
            valid.sort(key=lambda item: (stable_hash(f"{candidate.capture_id}:{item[0]}"), item[0]))
            capture_valid_count = len(valid)
            remaining = target - selected
            for flow_id, packets in valid[: min(MAX_FLOWS_PER_CAPTURE, remaining)]:
                sample_id = hashlib.sha256(
                    f"{candidate.capture_id}:{flow_id}".encode("utf-8")
                ).hexdigest()[:24]
                features, base_names = past_features(packets)
                if base_feature_names is None:
                    base_feature_names = base_names
                packet_x, packet_mask, burst_x, burst_mask = tensorize_flow(packets)
                if candidate.pcap_path not in pcap_hash_cache:
                    pcap_hash_cache[candidate.pcap_path] = file_sha256(candidate.pcap_path)
                manifest_rows.append(
                    {
                        "sample_id": sample_id,
                        "pcap_path": str(candidate.pcap_path.resolve()),
                        "binary_label": binary_label(kind),
                        "background_group": background_group(kind),
                        "carrier": carrier,
                        "source_dataset": candidate.source_dataset,
                        "capture_id": candidate.capture_id,
                        "domain_group": candidate.domain_group,
                        "flow_id": flow_id,
                        "split": split,
                        "capture_valid_flow_count": capture_valid_count,
                        "pcap_sha256": pcap_hash_cache[candidate.pcap_path],
                    }
                )
                feature_rows.append({"sample_id": sample_id, **features})
                packet_tensors.append(packet_x)
                packet_masks.append(packet_mask)
                burst_tensors.append(burst_x)
                burst_masks.append(burst_mask)
                selected += 1
            if selected and selected % 50 == 0:
                print(f"[progress] {quota_name}: {selected}/{target}", flush=True)
        actual[quota_name] = selected
        if selected != target:
            print(f"[deficit] {quota_name}: selected={selected}, target={target}", flush=True)

    manifest = pd.DataFrame(manifest_rows)
    features = pd.DataFrame(feature_rows)
    manifest.to_csv(args.output_dir / "manifest.csv", index=False)
    features.to_csv(args.output_dir / "past_features.csv", index=False)
    pd.DataFrame(failures, columns=["pcap_path", "error"]).to_csv(
        args.output_dir / "extraction_failures.csv", index=False
    )
    np.savez_compressed(
        args.output_dir / "sequences.npz",
        sample_ids=np.asarray(manifest["sample_id"].tolist()),
        packet_x=np.stack(packet_tensors) if packet_tensors else np.empty((0, 128, 3), dtype=np.float32),
        packet_mask=np.stack(packet_masks) if packet_masks else np.empty((0, 128), dtype=np.uint8),
        burst_x=np.stack(burst_tensors) if burst_tensors else np.empty((0, 64, 6), dtype=np.float32),
        burst_mask=np.stack(burst_masks) if burst_masks else np.empty((0, 64), dtype=np.uint8),
    )
    metadata = {
        "seed": SEED,
        "max_flows_per_capture": MAX_FLOWS_PER_CAPTURE,
        "requested_total": int(sum(QUOTAS.values())),
        "actual_total": len(manifest),
        "requested_quotas": {"/".join(key): value for key, value in QUOTAS.items()},
        "actual_quotas": actual,
        "candidate_inventory": candidate_inventory,
        "base_feature_names": base_feature_names or [],
        "manifest_sha256": file_sha256(args.output_dir / "manifest.csv"),
        "past_features_sha256": file_sha256(args.output_dir / "past_features.csv"),
        "failure_count": len(failures),
    }
    (args.output_dir / "dataset_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)
    if len(manifest) != sum(QUOTAS.values()) or any(
        actual.get("/".join(key), 0) != target for key, target in QUOTAS.items()
    ):
        raise SystemExit("The fixed 5,000-flow quotas were not satisfied; no group was backfilled.")


if __name__ == "__main__":
    main()
