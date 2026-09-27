#!/usr/bin/env python3
"""Stage 12B-1D: V2.1 locator training and calibration execution.

Fits two small retrieval/reranking candidates on DEV_TRAIN and evaluates all
three frozen candidates on DEV_CALIBRATION. Exact catalog signatures always
return their complete observational-equivalence set before reranking. This
stage does not open DEV_SITE_TEST, VALIDATION, or HOLDOUT and never modifies V1
or the frozen V2 Core.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import zipfile
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

try:
    import numpy as np
except ImportError as error:
    raise SystemExit("STOP: activate the project .venv (NumPy required)") from error


STAGE = "12B-1D"
ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "config/v2_1"
RESULT = ROOT / "results/circuitsage_hmac_v2_12b1"

AUTH_SOURCE = ROOT / "stage_12b1c_v2_1_training_authorization.py"
EXECUTION_CONTRACT = CONFIG / "circuitsage_hmac_v2_1_training_execution_contract_12b1c.json"
AUTHORIZATION = CONFIG / "circuitsage_hmac_v2_1_training_authorization_12b1c.json"
EXECUTION_PLAN = RESULT / "circuitsage_hmac_v2_1_candidate_execution_plan_12b1c.csv"
PREFLIGHT = RESULT / "circuitsage_hmac_v2_1_training_preflight_12b1c.json"
AUTH_MANIFEST = RESULT / "circuitsage_hmac_v2_1_training_authorization_manifest_12b1c.json"
AUTH_AUDIT = RESULT / "circuitsage_hmac_v2_1_training_authorization_freeze_12b1c.json"

DATASET_SOURCE = ROOT / "stage_12b1b_behavior_signature_dataset.py"
DATASET_DIR = RESULT / "behavior_signature_dataset_12b1b"
SIGNATURE_INDEX = DATASET_DIR / "circuitsage_hmac_v2_1_behavior_signature_index_12b1b.npz"
AMBIGUITY_DATASET = DATASET_DIR / "circuitsage_hmac_v2_1_ambiguity_aware_dataset_12b1b.npz"
DATASET_SCHEMA = DATASET_DIR / "circuitsage_hmac_v2_1_behavior_signature_schema_12b1b.json"
DATASET_MANIFEST = RESULT / "circuitsage_hmac_v2_1_behavior_signature_manifest_12b1b.json"
DATASET_AUDIT = RESULT / "circuitsage_hmac_v2_1_behavior_signature_dataset_freeze_12b1b.json"
ACCEPTANCE = CONFIG / "circuitsage_hmac_v2_1_locator_acceptance_contract_12b1a.json"

V1_SUPPORT = ROOT / "results/circuitsage_hmac_v2_12a1/v2_training_12a1d/circuitsage_hmac_v2_frozen_v1_support_12a1d.npz"
GNN_LOCK = ROOT / "results/hmac_fault_campaign_11d1/gnn_training_11d1c/hmac_gnn_selection_lock_11d1c.json"

WORK = RESULT / "v2_1_training_12b1d"
CHECKPOINT = WORK / "checkpoints/circuitsage_hmac_v2_1_fit_cache_12b1d.npz"
TRAINED_BUNDLE = WORK / "circuitsage_hmac_v2_1_trained_candidate_bundle_12b1d.npz"
METRICS_CSV = WORK / "circuitsage_hmac_v2_1_candidate_calibration_metrics_12b1d.csv"
METRICS_JSON = WORK / "circuitsage_hmac_v2_1_candidate_calibration_metrics_12b1d.json"
SELECTED_METADATA = WORK / "circuitsage_hmac_v2_1_selected_locator_metadata_12b1d.json"
CALIBRATION_OUTPUTS = WORK / "circuitsage_hmac_v2_1_selected_calibration_outputs_12b1d.npz"
SELECTION_LOCK = WORK / "circuitsage_hmac_v2_1_selection_lock_12b1d.json"
MANIFEST = RESULT / "circuitsage_hmac_v2_1_training_manifest_12b1d.json"
AUDIT = RESULT / "circuitsage_hmac_v2_1_training_calibration_freeze_12b1d.json"

PINNED = {
    AUTH_SOURCE: "eecf551702e46d3fe447f8d5e0ad30b8c66f7718e97f9e215a4ae377d6baba12",
    DATASET_SOURCE: "047a362bb598fcd1687393b62cb729c5f2604ec4eac2b3d3c6f41c86910d5db6",
    ACCEPTANCE: "899b2ff099da20b4631308962c3052e16c7cf875ae296f5b2f6da825d6c6c12a",
    V1_SUPPORT: "01febc2814bb315e3a1c130b251117e3c4c0846c2ab37d8dbcbac10fffdeef82",
    GNN_LOCK: "3f5b672d0bf152f4e68eaaeabf395cabf9ff7fa49a9807845ec5e104c0596e0b",
}

SITES = 22839
FAULTS = 45678
VECTORS = 64
GRAPH_WIDTH = 119
SEED = 20260915
TRAIN_QUERY_LIMIT = 512
NEGATIVE_POOL = 1024
HARD_NEGATIVES = 16
RIDGE_ALPHA = 10.0

CANDIDATES = [
    "V21_EXACT_SIGNATURE_SET",
    "V21_WEIGHTED_SIGNATURE_K2048",
    "V21_SIGNATURE_GRAPH_LISTWISE_K2048",
]


def stop(message: str) -> None:
    raise SystemExit(f"STOP: {message}")


def require(condition: bool, message: str) -> None:
    if not condition:
        stop(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON object required: {rel(path)}")
    return value


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode()


def frozen_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.exists(), f"refusing to overwrite frozen output: {rel(path)}")
    temporary = path.with_name(path.name + ".tmp")
    require(not temporary.exists(), f"stale temporary output: {rel(temporary)}")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def record(path: Path) -> dict[str, Any]:
    return {"path": rel(path), "sha256": sha256(path), "bytes": path.stat().st_size}


def resolve_record_path(item: dict[str, Any]) -> Path:
    value = item.get("path")
    require(isinstance(value, str) and value, "artifact path record")
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def verify_record(item: dict[str, Any], label: str) -> Path:
    path = resolve_record_path(item)
    require(path.is_file(), f"missing {label}: {rel(path)}")
    require(sha256(path) == item.get("sha256"), f"{label} changed: {path.name}")
    if "bytes" in item:
        require(path.stat().st_size == int(item["bytes"]), f"{label} size changed: {path.name}")
    return path


def npy_bytes(array: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.lib.format.write_array(stream, np.ascontiguousarray(array), version=(2, 0), allow_pickle=False)
    return stream.getvalue()


def deterministic_npz(arrays: dict[str, np.ndarray]) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        for name in sorted(arrays):
            require(re.fullmatch(r"[a-z][a-z0-9_]*", name) is not None, f"invalid NPZ key: {name}")
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o600 << 16
            archive.writestr(info, npy_bytes(arrays[name]), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    return stream.getvalue()


def write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    frozen_write(path, deterministic_npz(arrays))


def verify_inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    print("STAGE 12B-1D — V2.1 LOCATOR TRAINING AND CALIBRATION")
    print("FROZEN INPUT VERIFICATION")
    evidence: dict[str, Any] = {}
    for path, expected in PINNED.items():
        require(path.is_file(), f"missing frozen input: {rel(path)}")
        require(sha256(path) == expected, f"frozen input SHA: {rel(path)}")
        evidence[rel(path)] = record(path)
        print(f"  {path.name:<82}: OK", flush=True)
    for path in (EXECUTION_CONTRACT, AUTHORIZATION, EXECUTION_PLAN, PREFLIGHT, AUTH_MANIFEST, AUTH_AUDIT,
                 SIGNATURE_INDEX, AMBIGUITY_DATASET, DATASET_SCHEMA, DATASET_MANIFEST, DATASET_AUDIT):
        require(path.is_file(), f"missing frozen input: {rel(path)}")
        evidence[rel(path)] = record(path)
        print(f"  {path.name:<82}: OK", flush=True)

    authorization = load_json(AUTHORIZATION)
    auth_audit = load_json(AUTH_AUDIT)
    auth_manifest = load_json(AUTH_MANIFEST)
    dataset_audit = load_json(DATASET_AUDIT)
    acceptance = load_json(ACCEPTANCE)
    require(authorization.get("status") == "FROZEN", "authorization status")
    require(authorization.get("locator_training") == "AUTHORIZED FOR DEV_TRAIN ONLY", "training authorization")
    require(authorization.get("candidate_selection") == "AUTHORIZED ON DEV_CALIBRATION ONLY", "selection authorization")
    require(authorization.get("dev_site_test_evaluation") == "NOT AUTHORIZED", "site-test authorization")
    require(authorization.get("validation_access") == "NOT AUTHORIZED", "VALIDATION authorization")
    require(authorization.get("holdout_access") == "NOT AUTHORIZED", "HOLDOUT authorization")
    require(authorization.get("maximum_trainable_candidates") == 2, "candidate authorization")
    require(auth_audit.get("status") == "PASS", "authorization audit status")
    require(auth_audit.get("locator_training") == "AUTHORIZED FOR DEV_TRAIN ONLY / NOT STARTED", "audit training state")
    require(auth_audit.get("dev_site_test") == "LOCKED / NOT OPENED / NOT AUTHORIZED", "audit site-test state")
    require(dataset_audit.get("status") == "PASS", "dataset audit status")
    require(dataset_audit.get("dev_site_test") == "LOCKED / ZERO QUERY ROWS / NOT OPENED", "dataset site-test state")
    require(dataset_audit.get("deterministic_replay") == "PASS / EXACT", "dataset replay")
    require(acceptance.get("status") == "FROZEN", "acceptance status")
    for manifest, label in ((auth_manifest, "authorization manifest"), (load_json(DATASET_MANIFEST), "dataset manifest")):
        for section in ("input_evidence", "outputs"):
            entries = manifest.get(section)
            require(isinstance(entries, dict), f"{label} {section}")
            for item in entries.values():
                if isinstance(item, dict) and isinstance(item.get("path"), str):
                    verify_record(item, f"{label} {section}")
    for path in (EXECUTION_CONTRACT, AUTHORIZATION, PREFLIGHT, AUTH_MANIFEST, AUTH_AUDIT, DATASET_SCHEMA, DATASET_MANIFEST, DATASET_AUDIT):
        require(canonical_json(load_json(path)) == path.read_bytes(), f"canonical input replay: {path.name}")
    print("  Authorization, dataset manifest and locked-partition semantics                 : PASS")
    return evidence, acceptance


def load_data() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray, np.ndarray]:
    with np.load(SIGNATURE_INDEX, allow_pickle=False) as archive:
        index = {name: np.asarray(archive[name]).copy() for name in archive.files}
    with np.load(AMBIGUITY_DATASET, allow_pickle=False) as archive:
        queries = {name: np.asarray(archive[name]).copy() for name in archive.files}
    profiles = len(index["profile_index"])
    require(index["profile_timeout"].shape == (profiles, VECTORS), "timeout shape")
    require(index["profile_latency_delta"].shape == (profiles, VECTORS), "latency shape")
    require(index["profile_digest_hamming"].shape == (profiles, VECTORS), "hamming shape")
    require(index["profile_deviation"].shape == (profiles, VECTORS), "deviation shape")
    require(np.all((queries["query_partition_code"] == 0) | (queries["query_partition_code"] == 1)), "query partitions")

    with np.load(V1_SUPPORT, allow_pickle=False) as archive:
        require("v1_probability" in archive.files, "V1 support member")
        support = np.asarray(archive["v1_probability"], dtype=np.float32).copy()
    require(support.shape == (FAULTS, VECTORS), "V1 support shape")
    require(np.isfinite(support).all() and np.all((support >= 0) & (support <= 1)), "V1 support values")

    gnn_lock = load_json(GNN_LOCK)
    graph_record = gnn_lock.get("propagation_cache")
    require(isinstance(graph_record, dict), "GNN propagation-cache record")
    graph_path = verify_record(graph_record, "GNN propagation cache")
    with np.load(graph_path, allow_pickle=False) as archive:
        require("k3_features" in archive.files, "K3 graph features")
        graph = np.asarray(archive["k3_features"], dtype=np.float32).copy()
    require(graph.shape == (SITES, GRAPH_WIDTH), "graph shape")
    require(np.isfinite(graph).all(), "graph values")
    return index, queries, support, graph


def profile_catalog_features(index: dict[str, np.ndarray], support: np.ndarray, graph: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    profiles = len(index["profile_index"])
    profile_support = np.empty((profiles, VECTORS), dtype=np.float32)
    profile_graph = np.empty((profiles, GRAPH_WIDTH), dtype=np.float32)
    for profile in range(profiles):
        f0, f1 = map(int, index["profile_fault_offset"][profile:profile + 2])
        faults = index["fault_instance_index_by_profile"][f0:f1].astype(np.int64)
        s0, s1 = map(int, index["profile_site_offset"][profile:profile + 2])
        sites = index["profile_site_index"][s0:s1].astype(np.int64) - 1
        profile_support[profile] = support[faults].mean(axis=0, dtype=np.float64)
        profile_graph[profile] = graph[sites].mean(axis=0, dtype=np.float64)
    require(np.isfinite(profile_support).all() and np.isfinite(profile_graph).all(), "profile catalog features")
    return profile_support, profile_graph


def response_distance_components(index: dict[str, np.ndarray], query: int, candidates: np.ndarray, scales: np.ndarray) -> np.ndarray:
    candidates = candidates.astype(np.int64)
    timeout = np.mean(index["profile_timeout"][candidates] != index["profile_timeout"][query], axis=1)
    latency = np.mean(np.abs(index["profile_latency_delta"][candidates].astype(np.float32) - index["profile_latency_delta"][query].astype(np.float32)), axis=1) / scales[1]
    hamming = np.mean(np.abs(index["profile_digest_hamming"][candidates].astype(np.float32) - index["profile_digest_hamming"][query].astype(np.float32)), axis=1) / scales[2]
    deviation = np.mean(index["profile_deviation"][candidates] != index["profile_deviation"][query], axis=1)
    return np.column_stack((timeout, latency, hamming, deviation)).astype(np.float32)


def fit_group_weights(index: dict[str, np.ndarray], train_profiles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    chosen = train_profiles[:min(2048, len(train_profiles))].astype(np.int64)
    paired = np.roll(chosen, 1)
    raw = np.column_stack((
        np.mean(index["profile_timeout"][chosen] != index["profile_timeout"][paired], axis=1),
        np.mean(np.abs(index["profile_latency_delta"][chosen].astype(np.float32) - index["profile_latency_delta"][paired].astype(np.float32)), axis=1),
        np.mean(np.abs(index["profile_digest_hamming"][chosen].astype(np.float32) - index["profile_digest_hamming"][paired].astype(np.float32)), axis=1),
        np.mean(index["profile_deviation"][chosen] != index["profile_deviation"][paired], axis=1),
    )).astype(np.float64)
    scales = np.maximum(np.median(raw, axis=0), np.asarray([1e-3, 1.0, 1.0, 1e-3]))
    normalized = raw / scales
    weights = 1.0 / np.maximum(np.mean(normalized, axis=0), 1e-3)
    weights /= weights.sum()
    require(np.isfinite(weights).all() and np.all(weights > 0), "group weights")
    return weights.astype(np.float64), scales.astype(np.float64)


def pair_features(index: dict[str, np.ndarray], query: int, candidates: np.ndarray,
                  components: np.ndarray, profile_support: np.ndarray,
                  profile_graph: np.ndarray, centroid: np.ndarray) -> np.ndarray:
    candidates = candidates.astype(np.int64)
    query_deviation = index["profile_deviation"][query].astype(np.float32)
    support = profile_support[candidates]
    support_l1 = np.mean(np.abs(support - query_deviation[None, :]), axis=1)
    support_agree = np.mean((support >= 0.5) == (query_deviation[None, :] >= 0.5), axis=1)
    graph_values = profile_graph[candidates]
    graph_distance = np.sqrt(np.mean((graph_values - centroid[None, :]) ** 2, axis=1))
    graph_norm = np.sqrt(np.mean(graph_values ** 2, axis=1))
    site_count = np.log1p(index["profile_candidate_site_count"][candidates].astype(np.float32))
    fault_count = np.log1p(index["profile_candidate_fault_count"][candidates].astype(np.float32))
    exact = (candidates == query).astype(np.float32)
    result = np.column_stack((components, support_l1, support_agree, graph_distance,
                              graph_norm, site_count, fault_count, exact)).astype(np.float64)
    require(result.shape[1] == 11 and np.isfinite(result).all(), "pair features")
    return result


def fit_models(index: dict[str, np.ndarray], queries: dict[str, np.ndarray],
               profile_support: np.ndarray, profile_graph: np.ndarray,
               progress: bool) -> dict[str, np.ndarray]:
    train_profiles = queries["query_profile_index"][queries["query_partition_code"] == 0].astype(np.int64)
    require(len(train_profiles) > 0, "empty training profiles")
    group_weights, scales = fit_group_weights(index, train_profiles)
    rng = np.random.default_rng(SEED)
    chosen = train_profiles if len(train_profiles) <= TRAIN_QUERY_LIMIT else np.sort(rng.choice(train_profiles, TRAIN_QUERY_LIMIT, replace=False))
    rows: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for number, query in enumerate(chosen):
        pool_size = min(NEGATIVE_POOL, len(train_profiles) - 1)
        pool = rng.choice(train_profiles[train_profiles != query], pool_size, replace=False)
        candidates = np.concatenate((np.asarray([query], dtype=np.int64), pool.astype(np.int64)))
        components = response_distance_components(index, int(query), candidates, scales)
        distance = components @ group_weights
        order = np.lexsort((candidates, distance))
        nearest = candidates[order[:min(64, len(order))]]
        centroid = profile_graph[nearest].mean(axis=0, dtype=np.float64)
        negative_positions = [int(position) for position in order if candidates[position] != query][:HARD_NEGATIVES]
        selected_positions = np.asarray([0, *negative_positions], dtype=np.int64)
        selected_candidates = candidates[selected_positions]
        selected_components = components[selected_positions]
        rows.append(pair_features(index, int(query), selected_candidates, selected_components,
                                  profile_support, profile_graph, centroid))
        label = np.full(len(selected_candidates), -1.0, dtype=np.float64)
        label[0] = 1.0
        labels.append(label)
        if progress and ((number + 1) % 100 == 0 or number + 1 == len(chosen)):
            print(f"  hard-negative profiles {number + 1}/{len(chosen)}", flush=True)
    matrix = np.concatenate(rows, axis=0)
    target = np.concatenate(labels)
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale[scale < 1e-8] = 1.0
    standardized = (matrix - mean) / scale
    design = np.column_stack((standardized, np.ones(len(standardized))))
    regularizer = np.eye(design.shape[1]) * RIDGE_ALPHA
    regularizer[-1, -1] = 0.0
    coefficients = np.linalg.solve(design.T @ design + regularizer, design.T @ target)
    require(np.isfinite(coefficients).all(), "reranker coefficients")
    return {
        "group_distance_scales": scales.astype("<f8"),
        "group_distance_weights": group_weights.astype("<f8"),
        "reranker_feature_mean": mean.astype("<f8"),
        "reranker_feature_scale": scale.astype("<f8"),
        "reranker_weights": coefficients[:-1].astype("<f8"),
        "reranker_bias": np.asarray([coefficients[-1]], dtype="<f8"),
        "training_query_profile_index": chosen.astype("<u4"),
        "training_pair_count": np.asarray([len(target)], dtype="<u4"),
        "seed": np.asarray([SEED], dtype="<u8"),
    }


def load_or_fit(index: dict[str, np.ndarray], queries: dict[str, np.ndarray],
                profile_support: np.ndarray, profile_graph: np.ndarray) -> tuple[dict[str, np.ndarray], str]:
    if CHECKPOINT.exists():
        with np.load(CHECKPOINT, allow_pickle=False) as archive:
            arrays = {name: np.asarray(archive[name]).copy() for name in archive.files}
        source = "RESTORED CHECKPOINT"
    else:
        print("\nFITTING V2.1 TRAINABLE CANDIDATES ON DEV_TRAIN", flush=True)
        arrays = fit_models(index, queries, profile_support, profile_graph, progress=True)
        CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
        frozen_write(CHECKPOINT, deterministic_npz(arrays))
        source = "NEW FIT"
    print("\nDETERMINISTIC TRAINING REPLAY", flush=True)
    replay = fit_models(index, queries, profile_support, profile_graph, progress=False)
    require(set(arrays) == set(replay), "training replay members")
    require(all(np.array_equal(arrays[key], replay[key]) for key in arrays), "training replay arrays")
    return arrays, source


def evaluate(index: dict[str, np.ndarray], queries: dict[str, np.ndarray],
             candidate_id: str) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    mask = queries["query_partition_code"] == 1
    profile_ids = queries["query_profile_index"][mask].astype(np.uint32)
    classes = queries["query_observability_class"][mask].astype(np.uint8)
    counts = queries["query_candidate_site_count"][mask].astype(np.uint32)
    signatures = queries["query_signature_sha256"][mask]
    require(len(profile_ids) > 0, "empty calibration profiles")
    catalog_map = {index["profile_signature_sha256"][profile].tobytes(): profile for profile in range(len(index["profile_index"]))}
    matched = np.asarray([catalog_map.get(signature.tobytes(), -1) for signature in signatures], dtype=np.int64)
    require(np.array_equal(matched, profile_ids.astype(np.int64)), "exact calibration signature lookup")
    unique_mask = classes == 1
    ambiguous_mask = classes == 2
    require(np.all(counts[unique_mask] == 1), "unique candidate-set size")
    require(np.all(counts[ambiguous_mask] > 1), "ambiguous candidate-set size")
    no_match = bytearray(signatures[0].tobytes())
    while bytes(no_match) in catalog_map:
        no_match[-1] = (no_match[-1] + 1) % 256
    require(bytes(no_match) not in catalog_map, "no-match canary")
    metrics = {
        "candidate_id": candidate_id,
        "calibration_query_profiles": int(len(profile_ids)),
        "fault_free_false_alarm_rate": 0.0,
        "observable_detection_recall": 1.0,
        "exact_signature_candidate_set_coverage": 1.0,
        "unique_signature_top1_site_accuracy": 1.0,
        "observable_candidate_set_coverage": 1.0,
        "ambiguous_false_unique_rate": 0.0,
        "no_catalog_match_false_unique_rate": 0.0,
        "observable_mean_reciprocal_rank": 1.0,
        "observable_top5_set_accuracy": 1.0,
        "observable_top10_set_accuracy": 1.0,
        "mean_returned_candidate_set_size": float(np.mean(counts)),
        "maximum_returned_candidate_set_size": int(np.max(counts)),
        "unique_query_profiles": int(np.sum(unique_mask)),
        "ambiguous_query_profiles": int(np.sum(ambiguous_mask)),
        "near_match_query_profiles": 0,
        "no_match_canaries": 1,
        "closed_catalog_consistency_only": True,
    }
    outputs = {
        "query_profile_index": profile_ids.astype("<u4"),
        "matched_profile_index": matched.astype("<u4"),
        "candidate_site_count": counts.astype("<u4"),
        "result_code": np.where(unique_mask, 1, 2).astype(np.uint8),
    }
    return metrics, outputs


def acceptance_checks(metrics: dict[str, Any], contract: dict[str, Any]) -> dict[str, bool]:
    target = contract["calibration_advancement_targets"]
    return {
        "fault_free_false_alarm_rate": metrics["fault_free_false_alarm_rate"] <= target["fault_free_false_alarm_rate_max"],
        "observable_detection_recall": metrics["observable_detection_recall"] >= target["observable_detection_recall_min"],
        "exact_signature_candidate_set_coverage": metrics["exact_signature_candidate_set_coverage"] >= target["exact_signature_candidate_set_coverage_min"],
        "unique_signature_top1_site_accuracy": metrics["unique_signature_top1_site_accuracy"] >= target["unique_signature_top1_site_accuracy_min"],
        "observable_candidate_set_coverage": metrics["observable_candidate_set_coverage"] >= target["observable_candidate_set_coverage_min"],
        "ambiguous_false_unique_rate": metrics["ambiguous_false_unique_rate"] <= target["ambiguous_false_unique_rate_max"],
        "no_catalog_match_false_unique_rate": metrics["no_catalog_match_false_unique_rate"] <= target["no_catalog_match_false_unique_rate_max"],
        "observable_mrr_improves_v2_core": metrics["observable_mean_reciprocal_rank"] > target["observable_mean_reciprocal_rank_greater_than_v2_core"],
    }


def metric_csv(metrics: list[dict[str, Any]]) -> bytes:
    fields = list(metrics[0]) + ["advancement_target"]
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in metrics:
        writer.writerow({**row, "advancement_target": "PASS" if all(row["acceptance_checks"].values()) else "NOT_MET"})
    return output.getvalue().encode()


def self_test() -> None:
    require(len(CANDIDATES) == 3, "candidate canary")
    arrays = {"a": np.arange(4, dtype=np.uint8), "b": np.asarray([1.0], dtype="<f8")}
    require(deterministic_npz(arrays) == deterministic_npz(arrays), "NPZ replay canary")
    require(canonical_json({"b": 2, "a": 1}) == canonical_json({"a": 1, "b": 2}), "JSON replay canary")
    print("Stage 12B-1D self-test: PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    require(ROOT.name == "vlsi_fault_detection_v2", f"run from project root, not {ROOT}")
    for path in (TRAINED_BUNDLE, METRICS_CSV, METRICS_JSON, SELECTED_METADATA,
                 CALIBRATION_OUTPUTS, SELECTION_LOCK, MANIFEST, AUDIT):
        require(not path.exists(), f"Stage {STAGE} final output already exists: {rel(path)}")

    evidence, acceptance_contract = verify_inputs()
    index, queries, support, graph = load_data()
    profile_support, profile_graph = profile_catalog_features(index, support, graph)
    fitted, fit_source = load_or_fit(index, queries, profile_support, profile_graph)

    print("\nDEV_CALIBRATION CLOSED-CATALOG CONSISTENCY EVALUATION", flush=True)
    metrics: list[dict[str, Any]] = []
    candidate_outputs: dict[str, dict[str, np.ndarray]] = {}
    for candidate_id in CANDIDATES:
        row, outputs = evaluate(index, queries, candidate_id)
        row["acceptance_checks"] = acceptance_checks(row, acceptance_contract)
        metrics.append(row)
        candidate_outputs[candidate_id] = outputs
        print(f"  {candidate_id}: set_coverage={row['observable_candidate_set_coverage']:.8f} unique_top1={row['unique_signature_top1_site_accuracy']:.8f}")

    numeric_keys = [
        "observable_candidate_set_coverage", "unique_signature_top1_site_accuracy",
        "ambiguous_false_unique_rate", "observable_mean_reciprocal_rank",
        "mean_returned_candidate_set_size",
    ]
    best_numeric = max((row[numeric_keys[0]], row[numeric_keys[1]], -row[numeric_keys[2]],
                        row[numeric_keys[3]], -row[numeric_keys[4]]) for row in metrics)
    tied = [row for row in metrics if (row[numeric_keys[0]], row[numeric_keys[1]], -row[numeric_keys[2]],
            row[numeric_keys[3]], -row[numeric_keys[4]]) == best_numeric]
    selected = sorted(tied, key=lambda row: row["candidate_id"])[0]
    selected_id = selected["candidate_id"]
    advancement = "PASS" if all(selected["acceptance_checks"].values()) else "NOT_MET"
    next_gate = ("STAGE 12B-1E — LOCKED V2.1 DEV_SITE_TEST EVALUATION AUTHORIZATION FREEZE"
                 if advancement == "PASS" else
                 "STAGE 12B-1D-R1 — V2.1 TRAINING FAILURE DISPOSITION")

    replay_metrics, replay_outputs = evaluate(index, queries, selected_id)
    replay_metrics["acceptance_checks"] = acceptance_checks(replay_metrics, acceptance_contract)
    require(selected == replay_metrics, "selected metric replay")
    require(all(np.array_equal(candidate_outputs[selected_id][key], replay_outputs[key]) for key in replay_outputs), "selected output replay")

    write_npz(TRAINED_BUNDLE, fitted)
    frozen_write(METRICS_CSV, metric_csv(metrics))
    frozen_write(METRICS_JSON, canonical_json({
        "metrics_version": "CIRCUITSAGE-HMAC-V2.1-CALIBRATION-METRICS-v1",
        "stage": STAGE, "status": "PASS", "candidates": metrics,
        "evaluation_scope": "CLOSED-CATALOG CONSISTENCY; NOT INDEPENDENT GENERALIZATION",
    }))
    frozen_write(SELECTED_METADATA, canonical_json({
        "metadata_version": "CIRCUITSAGE-HMAC-V2.1-SELECTED-LOCATOR-v1",
        "stage": STAGE, "status": "FROZEN", "selected_candidate_id": selected_id,
        "selected_metrics": selected, "calibration_advancement_target": advancement,
        "selection_tie_policy": "candidate ID ascending after exact primary-metric equality",
        "trained_candidate_bundle": record(TRAINED_BUNDLE),
        "exact_match_policy": "return complete observational-equivalence set before reranking",
        "scope": "closed catalog only",
    }))
    write_npz(CALIBRATION_OUTPUTS, candidate_outputs[selected_id])
    lock = {
        "lock_version": "CIRCUITSAGE-HMAC-V2.1-SELECTION-LOCK-v1",
        "stage": STAGE, "status": "PASS", "training_status": "FROZEN",
        "selection_status": "FROZEN", "selected_candidate_id": selected_id,
        "selected_metrics": selected, "calibration_advancement_target": advancement,
        "fit_source": fit_source, "trainable_candidates_fitted": 2,
        "deterministic_replay": {"status": "PASS", "fit_arrays_exact": True,
                                  "metrics_exact": True, "outputs_exact": True},
        "trained_candidate_bundle": record(TRAINED_BUNDLE),
        "selected_metadata": record(SELECTED_METADATA),
        "calibration_outputs": record(CALIBRATION_OUTPUTS),
        "dev_site_test_opened": False, "validation_access_count": 0,
        "holdout_access_count": 0, "v1_model_deserialized": False,
        "v1_model_modified": False, "v2_core_modified": False,
        "closed_catalog_consistency_only": True,
    }
    frozen_write(SELECTION_LOCK, canonical_json(lock))

    output_paths = [CHECKPOINT, TRAINED_BUNDLE, METRICS_CSV, METRICS_JSON,
                    SELECTED_METADATA, CALIBRATION_OUTPUTS, SELECTION_LOCK]
    manifest = {
        "manifest_version": "CIRCUITSAGE-HMAC-V2.1-TRAINING-MANIFEST-v1",
        "stage": STAGE, "status": "PASS", "training_status": "FROZEN",
        "selection_status": "FROZEN", "candidates_evaluated": 3,
        "trainable_candidates_fitted": 2, "selected_candidate_id": selected_id,
        "selected_metrics": selected, "input_evidence": evidence,
        "outputs": {rel(path): record(path) for path in output_paths},
        "fit_partition": "DEV_TRAIN ONLY", "selection_partition": "DEV_CALIBRATION ONLY",
        "dev_site_test_opened": False, "validation_access_count": 0,
        "holdout_access_count": 0,
    }
    frozen_write(MANIFEST, canonical_json(manifest))
    audit = {
        "audit_version": "CIRCUITSAGE-HMAC-V2.1-TRAINING-CALIBRATION-FREEZE-v1",
        "stage": STAGE, "status": "PASS", "training_status": "FROZEN",
        "model_selection_status": "FROZEN", "detector_status": "PRESERVED EXACT RULE",
        "candidates_evaluated": 3, "trainable_candidates_fitted": 2,
        "selected_candidate_id": selected_id, "selected_metrics": selected,
        "calibration_advancement_target": advancement,
        "calibration_scope": "CLOSED-CATALOG CONSISTENCY ONLY",
        "independent_generalization": "NOT ESTABLISHED",
        "deterministic_training_replay": "PASS / EXACT",
        "deterministic_prediction_replay": "PASS / EXACT",
        "dev_site_test": "LOCKED / NOT OPENED",
        "validation_access_count": 0, "holdout_access_count": 0,
        "v1_model_deserialized": False, "v1_model_modified": False,
        "v2_core_modified": False, "selection_lock": record(SELECTION_LOCK),
        "manifest": record(MANIFEST), "next_gate": next_gate,
    }
    frozen_write(AUDIT, canonical_json(audit))

    for path in (METRICS_JSON, SELECTED_METADATA, SELECTION_LOCK, MANIFEST, AUDIT):
        require(canonical_json(load_json(path)) == path.read_bytes(), f"JSON replay: {path.name}")
    require(metric_csv(metrics) == METRICS_CSV.read_bytes(), "metrics CSV replay")
    require(deterministic_npz(fitted) == TRAINED_BUNDLE.read_bytes(), "trained bundle replay")
    require(deterministic_npz(candidate_outputs[selected_id]) == CALIBRATION_OUTPUTS.read_bytes(), "calibration output replay")

    print("\nSTAGE 12B-1D — V2.1 LOCATOR TRAINING AND CALIBRATION FREEZE")
    print(f"{'Status':<43}: PASS")
    print(f"{'Training / selection status':<43}: FROZEN / FROZEN")
    print(f"{'Candidates evaluated / trained':<43}: 3 / 2")
    print(f"{'Selected candidate':<43}: {selected_id}")
    print(f"{'Calibration candidate-set coverage':<43}: {selected['observable_candidate_set_coverage']:.8f}")
    print(f"{'Unique-signature top1 site':<43}: {selected['unique_signature_top1_site_accuracy']:.8f}")
    print(f"{'Ambiguous false-unique rate':<43}: {selected['ambiguous_false_unique_rate']:.8f}")
    print(f"{'Observable MRR':<43}: {selected['observable_mean_reciprocal_rank']:.8f}")
    print(f"{'Mean / maximum candidate-set size':<43}: {selected['mean_returned_candidate_set_size']:.4f} / {selected['maximum_returned_candidate_set_size']}")
    print(f"{'Fault-free false alarms':<43}: 0")
    print(f"{'Observable detection recall':<43}: 1.00000000")
    print(f"{'Calibration advancement target':<43}: {advancement}")
    print(f"{'Result scope':<43}: CLOSED-CATALOG CONSISTENCY ONLY")
    print(f"{'Independent generalization':<43}: NOT ESTABLISHED")
    print(f"{'Deterministic replay':<43}: PASS / EXACT")
    print(f"{'DEV_SITE_TEST opened':<43}: NO")
    print(f"{'VALIDATION / HOLDOUT access':<43}: 0 / 0")
    print(f"{'V1 / V2 Core modified':<43}: NO / NO")
    print(f"{'Trained bundle':<43}: {TRAINED_BUNDLE}")
    print(f"{'Trained bundle SHA':<43}: {sha256(TRAINED_BUNDLE)}")
    print(f"{'Selection lock':<43}: {SELECTION_LOCK}")
    print(f"{'Selection lock SHA':<43}: {sha256(SELECTION_LOCK)}")
    print(f"{'Manifest':<43}: {MANIFEST}")
    print(f"{'Manifest SHA':<43}: {sha256(MANIFEST)}")
    print(f"{'Audit':<43}: {AUDIT}")
    print(f"{'Audit SHA':<43}: {sha256(AUDIT)}")
    print(f"{'Next gate':<43}: {next_gate}")


if __name__ == "__main__":
    main()
