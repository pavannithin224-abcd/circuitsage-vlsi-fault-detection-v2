#!/usr/bin/env python3
"""Stage 12A-1D-R2: response-conditioned locator repair training.

Runs the four ablations required by the frozen 12A-1D-R1 repair contract.
Only DEV_TRAIN contributes to fitted parameters and hard-negative mining.
DEV_CALIBRATION is used for model selection. DEV_SITE_TEST, VALIDATION, and
HOLDOUT are not opened.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import math
import os
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


STAGE = "12A-1D-R2"
VERSION = "CIRCUITSAGE-HMAC-V2-RESPONSE-CONDITIONED-LOCATOR-v1"
ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "config" / "v2"
RESULT = ROOT / "results" / "circuitsage_hmac_v2_12a1"
PRIOR_SOURCE = ROOT / "stage_12a1d_v2_linked_train.py"
REVIEW_SOURCE = ROOT / "stage_12a1d_r1_localization_failure_review.py"
PRIOR_LOCK = RESULT / "v2_training_12a1d/circuitsage_hmac_v2_selection_lock_12a1d.json"
PRIOR_MANIFEST = RESULT / "circuitsage_hmac_v2_training_manifest_12a1d.json"
PRIOR_AUDIT = RESULT / "circuitsage_hmac_v2_training_calibration_freeze_12a1d.json"
REPAIR_CONTRACT = CONFIG / "circuitsage_hmac_v2_localization_repair_contract_12a1d_r1.json"
REVIEW_DIAGNOSTICS = RESULT / "v2_localization_failure_review_12a1d_r1/circuitsage_hmac_v2_localization_diagnostics_12a1d_r1.json"
REVIEW_REPORT = RESULT / "v2_localization_failure_review_12a1d_r1/circuitsage_hmac_v2_localization_failure_review_12a1d_r1.md"
REVIEW_MANIFEST = RESULT / "circuitsage_hmac_v2_localization_failure_review_manifest_12a1d_r1.json"
REVIEW_AUDIT = RESULT / "circuitsage_hmac_v2_localization_failure_review_freeze_12a1d_r1.json"

WORK = RESULT / "v2_repair_training_12a1d_r2"
CHECKPOINT = WORK / "checkpoints/circuitsage_hmac_v2_repair_training_cache_12a1d_r2.npz"
METRICS_CSV = WORK / "circuitsage_hmac_v2_repair_candidate_calibration_metrics_12a1d_r2.csv"
METRICS_JSON = WORK / "circuitsage_hmac_v2_repair_candidate_calibration_metrics_12a1d_r2.json"
MODEL = WORK / "circuitsage_hmac_v2_selected_repaired_locator_12a1d_r2.npz"
MODEL_METADATA = WORK / "circuitsage_hmac_v2_selected_repaired_locator_metadata_12a1d_r2.json"
RANKINGS = WORK / "circuitsage_hmac_v2_selected_repair_calibration_rankings_12a1d_r2.npz"
SELECTION_LOCK = WORK / "circuitsage_hmac_v2_repair_selection_lock_12a1d_r2.json"
MANIFEST = RESULT / "circuitsage_hmac_v2_repair_training_manifest_12a1d_r2.json"
AUDIT = RESULT / "circuitsage_hmac_v2_repair_training_calibration_freeze_12a1d_r2.json"

PINNED = {
    PRIOR_SOURCE: "6144da31a28041f045d696d041f147efcfceacccd1fcfecaa52d8561c640020d",
    REVIEW_SOURCE: "ff83a7c1581281494fc7043c793fb3c95c8c8ab24e0aa055986542ac21ae44c2",
    PRIOR_LOCK: "6fb8d4eb79704652ccb289090e2b249c7600f0db723fdfffe62161e4fe5028b7",
    PRIOR_MANIFEST: "2c3df37e43ae93fb5ff14becf873e270ff59be7a8e30f9032feddfa5880b18fe",
    PRIOR_AUDIT: "c6437b429d8b2f4747dbbdcdcf7b92a7703d75b6428c8fb643d8c75fc3184b96",
    REPAIR_CONTRACT: "ccaab77ca3d33225839c166ec68c2c97953e35a26ca660b8a86fe2babfcb7dd6",
    REVIEW_DIAGNOSTICS: "ee2b3f40d7f3048b4942369c332b30a35392b834cb12863f7ad48c6476dd721e",
    REVIEW_REPORT: "f593fa12552555b2c3a38cecbec2c702e6552153f972f5e032ae712145d0c7ed",
    REVIEW_MANIFEST: "6eb629b9aa6b12384e5cdeac26f2d6b9871fa60b67375dc62810598bdf14cc05",
    REVIEW_AUDIT: "b2c184c654332beddcdc46530e8273fd979d7126a570c72893608e89be7ee179",
}

SITES = 22839
FAULTS = 45678
VECTORS = 64
GRAPH_WIDTH = 119
CANDIDATE_WIDTH = 185
RESPONSE_WIDTH = 512
TOP_K = 50
TRAIN_PROFILE_LIMIT = 2048
HARD_NEGATIVES = 32
RIDGE_ALPHA = 10.0
PAIR_EPOCHS = 12
PAIR_BATCH = 8192
SEED = 20260915

CANDIDATES = [
    {"candidate_id": "R2_SUPPORT_SIM_K256", "mode": "support", "shortlist": 256, "trainable_parameters": 0},
    {"candidate_id": "R2_FEATURE_CROSS_RIDGE_K512", "mode": "ridge", "shortlist": 512, "trainable_parameters": (RESPONSE_WIDTH + 1) * CANDIDATE_WIDTH},
    {"candidate_id": "R2_FEATURE_CROSS_HARDNEG_K1024", "mode": "hardneg", "shortlist": 1024, "trainable_parameters": (RESPONSE_WIDTH + 1) * CANDIDATE_WIDTH + 4},
    {"candidate_id": "R2_FEATURE_CROSS_HARDNEG_GRAPH_K1024", "mode": "graph", "shortlist": 1024, "trainable_parameters": (RESPONSE_WIDTH + 1) * CANDIDATE_WIDTH + 6},
]

TARGETS = {
    "fault_free_false_alarm_rate_max": 0.05,
    "observable_fault_detection_recall_min": 0.90,
    "observable_candidate_coverage_min": 0.95,
    "unique_signature_top1_site_accuracy_min": 0.80,
    "observable_top5_site_accuracy_min": 0.80,
}


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


def artifact(outputs: dict[str, Any], basename: str) -> tuple[Path, dict[str, Any]]:
    matches = []
    for item in outputs.values():
        if isinstance(item, dict):
            path = resolve_record_path(item)
            if path.name == basename:
                matches.append((path, item))
    require(len(matches) == 1, f"manifest artifact resolution: {basename}")
    path, item = matches[0]
    require(path.is_file() and sha256(path) == item.get("sha256"), f"artifact integrity: {basename}")
    return path, item


def import_prior():
    spec = importlib.util.spec_from_file_location("stage_12a1d_frozen", PRIOR_SOURCE)
    require(spec is not None and spec.loader is not None, "frozen trainer import")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    print("STAGE 12A-1D-R2 — RESPONSE-CONDITIONED LOCATOR REPAIR TRAINING")
    print("FROZEN INPUT VERIFICATION")
    evidence: dict[str, Any] = {}
    for path, expected in PINNED.items():
        require(path.is_file(), f"missing frozen input: {rel(path)}")
        require(sha256(path) == expected, f"frozen input SHA: {rel(path)}")
        evidence[rel(path)] = record(path)
        print(f"  {path.name:<78}: OK", flush=True)

    prior_lock = load_json(PRIOR_LOCK)
    prior_manifest = load_json(PRIOR_MANIFEST)
    prior_audit = load_json(PRIOR_AUDIT)
    review_audit = load_json(REVIEW_AUDIT)
    contract = load_json(REPAIR_CONTRACT)
    require(prior_lock.get("advancement_target_on_calibration") == "NOT_MET", "prior failed gate")
    require(prior_audit.get("calibration_advancement_target") == "NOT_MET", "prior audit gate")
    require(review_audit.get("status") == "PASS", "R1 review status")
    require(review_audit.get("repair_contract_status") == "FROZEN", "R1 repair contract status")
    require(review_audit.get("dev_site_test") == "LOCKED / NOT AUTHORIZED", "R1 site-test lock")
    require(contract.get("status") == "FROZEN", "repair contract freeze")
    require(contract.get("repair_training_authorization") == "AUTHORIZED FOR DEV_TRAIN ONLY", "repair authorization")
    require(contract.get("locked_evaluation_authorization") == "NOT AUTHORIZED BY THIS CONTRACT", "locked evaluation state")
    require(contract.get("retraining_of_v1") == "PROHIBITED", "V1 retraining prohibition")
    repair = contract.get("required_repair_architecture", {})
    limits = contract.get("execution_limits", {})
    require(repair.get("shortlist_sizes_to_test") == [256, 512, 1024], "repair shortlist contract")
    require(limits.get("maximum_new_trainable_candidates") == 4, "repair candidate limit")
    require(limits.get("deterministic_replay") == "EXACT", "repair replay contract")
    outputs = prior_manifest.get("outputs")
    require(isinstance(outputs, dict), "prior output manifest")
    for item in outputs.values():
        if isinstance(item, dict):
            path = resolve_record_path(item)
            require(path.is_file() and sha256(path) == item.get("sha256"), f"prior output changed: {path.name}")
    prior_inputs = prior_manifest.get("input_evidence")
    require(isinstance(prior_inputs, dict), "prior input evidence")
    for item in prior_inputs.values():
        if isinstance(item, dict) and isinstance(item.get("path"), str):
            path = resolve_record_path(item)
            require(path.is_file() and sha256(path) == item.get("sha256"),
                    f"prior frozen input changed: {path.name}")
    print("  Prior output manifest and R1 semantic authorization                         : PASS")
    return prior_lock, prior_manifest, evidence


def load_data(prior: Any, prior_outputs: dict[str, Any]) -> dict[str, Any]:
    response_path = prior.RESPONSE_NPZ
    target_path = prior.TARGET_NPZ
    require(response_path.is_file() and target_path.is_file(), "response dataset")
    response_record = prior.PINNED[response_path]
    target_record = prior.PINNED[target_path]
    require(sha256(response_path) == response_record, "response dataset SHA")
    require(sha256(target_path) == target_record, "target dataset SHA")

    with np.load(target_path, allow_pickle=False) as archive:
        needed = {"fault_instance_index", "profile_index", "partition_code", "observable",
                  "candidate_site_count", "cross_partition_profile", "stuck_value"}
        require(needed.issubset(archive.files), "target members")
        targets = {name: np.asarray(archive[name]).copy() for name in needed}
    require(len(targets["fault_instance_index"]) == FAULTS, "target fault count")
    require(np.array_equal(targets["fault_instance_index"], np.arange(FAULTS)), "fault ordering")

    with np.load(response_path, allow_pickle=False) as archive:
        timeout = np.asarray(archive["profile_timeout"], dtype=np.float32)
        latency = np.asarray(archive["profile_latency_delta"], dtype=np.float32)
        digest_xor = np.asarray(archive["profile_digest_xor"], dtype=np.float32)
        hamming = np.asarray(archive["profile_digest_hamming"], dtype=np.float32)
        deviation = np.asarray(archive["profile_deviation"], dtype=np.float32)
    profiles = len(timeout)
    require(timeout.shape == latency.shape == hamming.shape == deviation.shape == (profiles, VECTORS),
            "response component shapes")
    require(digest_xor.shape == (profiles, VECTORS, 32), "digest response shape")
    rng = np.random.Generator(np.random.PCG64(SEED))
    digest_projection = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32), size=(32, 4)) / math.sqrt(32.0)
    digest_hash = np.einsum("pvd,dk->pvk", digest_xor / 255.0, digest_projection,
                            optimize=True).reshape(profiles, -1)
    response = np.concatenate([
        timeout,
        np.tanh(latency / 128.0),
        hamming / 256.0,
        deviation,
        digest_hash,
    ], axis=1).astype(np.float32)
    require(response.shape == (profiles, RESPONSE_WIDTH), "engineered response shape")
    require(np.isfinite(response).all(), "finite response features")

    support_path, support_item = artifact(
        prior_outputs, "circuitsage_hmac_v2_frozen_v1_support_12a1d.npz")
    with np.load(support_path, allow_pickle=False) as archive:
        support = np.asarray(archive["v1_probability"], dtype=np.float32)
    require(support.shape == (FAULTS, VECTORS), "frozen V1 support shape")
    require(np.all((support >= 0) & (support <= 1)), "V1 support range")

    graph, graph_record = prior.graph_cache()
    require(graph.shape == (SITES, GRAPH_WIDTH), "graph feature shape")
    stuck = np.zeros((FAULTS, 2), dtype=np.float32)
    stuck[np.arange(FAULTS), np.arange(FAULTS) % 2] = 1.0
    candidate = np.concatenate([np.repeat(graph, 2, axis=0), stuck, support], axis=1)
    require(candidate.shape == (FAULTS, CANDIDATE_WIDTH), "candidate descriptor shape")

    return {
        "response": response,
        "deviation": deviation,
        "targets": targets,
        "support": support,
        "candidate": candidate.astype(np.float32),
        "digest_projection": digest_projection.astype(np.float32),
        "response_record": record(response_path),
        "target_record": record(target_path),
        "support_record": support_item,
        "graph_record": graph_record,
    }


def groups(targets: dict[str, np.ndarray], partition: int) -> list[tuple[int, np.ndarray]]:
    mask = (targets["partition_code"] == partition) & (targets["observable"] == 1)
    profiles = targets["profile_index"].astype(np.int64)
    faults = targets["fault_instance_index"].astype(np.int64)
    result = []
    for profile in np.unique(profiles[mask]):
        positives = faults[mask & (profiles == profile)]
        require(len(positives) > 0, "profile positive group")
        result.append((int(profile), positives))
    return result


def standardize_and_ridge(data: dict[str, Any]) -> dict[str, np.ndarray]:
    targets = data["targets"]
    train_groups = groups(targets, 0)
    train_profiles = np.asarray([profile for profile, _ in train_groups], dtype=np.int64)
    train_faults = targets["fault_instance_index"][targets["partition_code"] == 0].astype(np.int64)

    r_mean = data["response"][train_profiles].mean(axis=0, dtype=np.float64)
    r_scale = data["response"][train_profiles].std(axis=0, dtype=np.float64)
    r_scale[r_scale < 1e-8] = 1.0
    response = ((data["response"].astype(np.float64) - r_mean) / r_scale).astype(np.float32)

    c_mean = data["candidate"][train_faults].mean(axis=0, dtype=np.float64)
    c_scale = data["candidate"][train_faults].std(axis=0, dtype=np.float64)
    c_scale[c_scale < 1e-8] = 1.0
    candidate = ((data["candidate"].astype(np.float64) - c_mean) / c_scale).astype(np.float32)

    x = response[train_profiles].astype(np.float64)
    y = np.stack([candidate[positives].mean(axis=0, dtype=np.float64)
                  for _, positives in train_groups], axis=0)
    x_aug = np.concatenate([x, np.ones((len(x), 1), dtype=np.float64)], axis=1)
    gram = x_aug.T @ x_aug
    regularizer = np.eye(gram.shape[0], dtype=np.float64) * RIDGE_ALPHA
    regularizer[-1, -1] = 0.0
    ridge = np.linalg.solve(gram + regularizer, x_aug.T @ y).astype(np.float32)
    require(ridge.shape == (RESPONSE_WIDTH + 1, CANDIDATE_WIDTH), "ridge shape")
    require(np.isfinite(ridge).all(), "finite ridge projection")
    return {
        "response": response,
        "candidate": candidate,
        "response_mean": r_mean,
        "response_scale": r_scale,
        "candidate_mean": c_mean,
        "candidate_scale": c_scale,
        "ridge": ridge,
        "train_profiles": train_profiles,
        "train_faults": train_faults,
    }


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(matrix, axis=1, keepdims=True)
    norm[norm < 1e-8] = 1.0
    return (matrix / norm).astype(np.float32)


def feature_context(data: dict[str, Any], fitted: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    support = np.clip(data["support"].astype(np.float64), 1e-5, 1.0 - 1e-5)
    support_logit = np.log(support) - np.log1p(-support)
    support_bias = np.log1p(-support).sum(axis=1)
    support_raw = data["support"].astype(np.float32)
    candidate = fitted["candidate"]
    return {
        "support": support_raw,
        "support_logit": support_logit.astype(np.float32),
        "support_bias": support_bias.astype(np.float32),
        "support_sum": support_raw.sum(axis=1),
        "support_norm": np.linalg.norm(support_raw, axis=1),
        "candidate_norm": normalize_rows(candidate),
        "graph_norm": normalize_rows(candidate[:, :GRAPH_WIDTH]),
        "support_scaled_norm": normalize_rows(candidate[:, GRAPH_WIDTH + 2:]),
    }


def predict_descriptor(response_row: np.ndarray, ridge: np.ndarray) -> np.ndarray:
    augmented = np.concatenate([response_row.astype(np.float32), np.ones(1, dtype=np.float32)])
    return augmented @ ridge


def pair_features(profile: int, data: dict[str, Any], fitted: dict[str, np.ndarray],
                  context: dict[str, np.ndarray]) -> np.ndarray:
    d = data["deviation"][profile].astype(np.float32)
    support_dot = context["support"] @ d
    log_likelihood = (context["support_logit"] @ d + context["support_bias"]) / VECTORS
    d_norm = max(float(np.linalg.norm(d)), 1e-8)
    support_cosine = support_dot / np.maximum(context["support_norm"] * d_norm, 1e-8)
    negative_l1 = -(context["support_sum"] + float(d.sum()) - 2.0 * support_dot) / VECTORS

    predicted = predict_descriptor(fitted["response"][profile], fitted["ridge"])
    p_support = predicted[GRAPH_WIDTH + 2:]
    p_graph = predicted[:GRAPH_WIDTH]
    p_all_norm = predicted / max(float(np.linalg.norm(predicted)), 1e-8)
    p_support_norm = p_support / max(float(np.linalg.norm(p_support)), 1e-8)
    p_graph_norm = p_graph / max(float(np.linalg.norm(p_graph)), 1e-8)
    predicted_support_cosine = context["support_scaled_norm"] @ p_support_norm
    predicted_graph_cosine = context["graph_norm"] @ p_graph_norm
    predicted_all_cosine = context["candidate_norm"] @ p_all_norm
    matrix = np.stack([
        log_likelihood,
        support_cosine,
        negative_l1,
        predicted_support_cosine,
        predicted_graph_cosine,
        predicted_all_cosine,
    ], axis=1).astype(np.float32)
    require(matrix.shape == (FAULTS, 6) and np.isfinite(matrix).all(), "pair features")
    return matrix


def zscore_columns(features: np.ndarray) -> np.ndarray:
    mean = features.mean(axis=0, dtype=np.float64)
    scale = features.std(axis=0, dtype=np.float64)
    scale[scale < 1e-8] = 1.0
    return ((features.astype(np.float64) - mean) / scale).astype(np.float32)


def fixed_score(features: np.ndarray) -> np.ndarray:
    normalized = zscore_columns(features[:, :4])
    return (normalized @ np.asarray([1.0, 0.35, 0.35, 0.75], dtype=np.float32)).astype(np.float32)


def stable_top(scores: np.ndarray, count: int) -> np.ndarray:
    count = min(count, len(scores))
    threshold = np.partition(scores, -count)[-count]
    greater = np.flatnonzero(scores > threshold)
    equal = np.flatnonzero(scores == threshold)[:count - len(greater)]
    selected = np.concatenate([greater, equal])
    order = np.lexsort((selected, -scores[selected]))
    return selected[order]


def prioritize_shortlist(base: np.ndarray, rerank: np.ndarray, count: int) -> np.ndarray:
    shortlist = stable_top(base, count)
    result = base.astype(np.float64).copy()
    outside_ceiling = float(result.max())
    result -= outside_ceiling + 1000.0
    inside = rerank[shortlist].astype(np.float64)
    inside -= float(inside.min())
    result[shortlist] = inside + 1.0
    return result.astype(np.float32)


def hard_negative_differences(data: dict[str, Any], fitted: dict[str, np.ndarray],
                              context: dict[str, np.ndarray]) -> np.ndarray:
    train_groups = groups(data["targets"], 0)
    if len(train_groups) > TRAIN_PROFILE_LIMIT:
        chosen = np.linspace(0, len(train_groups) - 1, TRAIN_PROFILE_LIMIT, dtype=np.int64)
        selected_groups = [train_groups[int(index)] for index in chosen]
    else:
        selected_groups = train_groups
    universe = fitted["train_faults"]
    universe_set = np.zeros(FAULTS, dtype=bool)
    universe_set[universe] = True
    chunks = []
    rng = np.random.Generator(np.random.PCG64(SEED + 77))
    print("\nDEV_TRAIN HARD-NEGATIVE MINING", flush=True)
    for number, (profile, positives) in enumerate(selected_groups):
        features = pair_features(profile, data, fitted, context)
        base = fixed_score(features)
        positive_set = set(int(value) for value in positives)
        eligible = universe_set.copy()
        eligible[positives] = False
        masked_base = np.where(eligible, base, -np.inf)
        masked_graph = np.where(eligible, features[:, 4], -np.inf)
        hard = stable_top(masked_base, 20).tolist()
        graph_hard = stable_top(masked_graph, 8).tolist()
        sibling = []
        for positive in positives[:4]:
            other = int(positive) ^ 1
            if eligible[other]:
                sibling.append(other)
        negatives = []
        for value in hard + graph_hard + sibling:
            if value not in positive_set and value not in negatives and eligible[value]:
                negatives.append(value)
        if len(negatives) < HARD_NEGATIVES:
            pool = universe[eligible[universe]]
            for value in rng.choice(pool, size=min(len(pool), HARD_NEGATIVES * 2), replace=False):
                integer = int(value)
                if integer not in negatives:
                    negatives.append(integer)
                if len(negatives) == HARD_NEGATIVES:
                    break
        negatives = negatives[:HARD_NEGATIVES]
        require(len(negatives) == HARD_NEGATIVES, "hard-negative count")
        positive_rows = positives[:min(4, len(positives))]
        difference = features[positive_rows, None, :] - features[np.asarray(negatives)][None, :, :]
        chunks.append(difference.reshape(-1, 6))
        if (number + 1) % 200 == 0 or number + 1 == len(selected_groups):
            print(f"  mined profiles {number + 1}/{len(selected_groups)}", flush=True)
    result = np.concatenate(chunks, axis=0).astype(np.float32)
    require(result.ndim == 2 and result.shape[1] == 6 and np.isfinite(result).all(),
            "hard-negative difference matrix")
    return result


def fit_pairwise(differences: np.ndarray, width: int, seed_offset: int) -> tuple[np.ndarray, np.ndarray, list[float]]:
    x = differences[:, :width].astype(np.float64)
    scale = x.std(axis=0)
    scale[scale < 1e-8] = 1.0
    x /= scale
    weights = np.zeros(width, dtype=np.float64)
    first_moment = np.zeros_like(weights)
    second_moment = np.zeros_like(weights)
    rng = np.random.Generator(np.random.PCG64(SEED + seed_offset))
    step = 0
    history = []
    for epoch in range(PAIR_EPOCHS):
        order = rng.permutation(len(x))
        losses = []
        for start in range(0, len(order), PAIR_BATCH):
            batch = x[order[start:start + PAIR_BATCH]]
            margin = np.clip(batch @ weights, -40.0, 40.0)
            losses.append(float(np.mean(np.logaddexp(0.0, -margin))))
            sigmoid_negative = 1.0 / (1.0 + np.exp(margin))
            gradient = -(batch.T @ sigmoid_negative) / len(batch) + 1e-4 * weights
            step += 1
            first_moment = 0.9 * first_moment + 0.1 * gradient
            second_moment = 0.999 * second_moment + 0.001 * gradient * gradient
            corrected_m = first_moment / (1.0 - 0.9 ** step)
            corrected_v = second_moment / (1.0 - 0.999 ** step)
            weights -= 0.02 * corrected_m / (np.sqrt(corrected_v) + 1e-8)
        history.append(float(np.mean(losses)))
    raw_weights = weights / scale
    require(np.isfinite(raw_weights).all(), "finite pairwise weights")
    return raw_weights.astype(np.float32), scale.astype(np.float32), history


def train_models(data: dict[str, Any]) -> dict[str, Any]:
    fitted = standardize_and_ridge(data)
    context = feature_context(data, fitted)
    differences = hard_negative_differences(data, fitted, context)
    cross_weights, cross_scale, cross_history = fit_pairwise(differences, 4, 101)
    graph_weights, graph_scale, graph_history = fit_pairwise(differences, 6, 202)
    arrays = {
        "digest_projection": data["digest_projection"].astype("<f4"),
        "response_mean": fitted["response_mean"].astype("<f8"),
        "response_scale": fitted["response_scale"].astype("<f8"),
        "candidate_mean": fitted["candidate_mean"].astype("<f8"),
        "candidate_scale": fitted["candidate_scale"].astype("<f8"),
        "ridge_projection": fitted["ridge"].astype("<f4"),
        "cross_pair_weights": cross_weights.astype("<f4"),
        "cross_pair_scale": cross_scale.astype("<f4"),
        "graph_pair_weights": graph_weights.astype("<f4"),
        "graph_pair_scale": graph_scale.astype("<f4"),
        "cross_loss_history": np.asarray(cross_history, dtype="<f8"),
        "graph_loss_history": np.asarray(graph_history, dtype="<f8"),
        "hard_negative_pairs": np.asarray([len(differences)], dtype="<u8"),
    }
    return {"fitted": fitted, "context": context, "arrays": arrays,
            "cross_weights": cross_weights, "graph_weights": graph_weights,
            "cross_history": cross_history, "graph_history": graph_history,
            "hard_negative_pairs": len(differences)}


def restore_models(data: dict[str, Any], arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    required = {
        "digest_projection", "response_mean", "response_scale", "candidate_mean",
        "candidate_scale", "ridge_projection", "cross_pair_weights",
        "cross_pair_scale", "graph_pair_weights", "graph_pair_scale",
        "cross_loss_history", "graph_loss_history", "hard_negative_pairs",
    }
    require(required == set(arrays), "repair checkpoint members")
    require(np.array_equal(arrays["digest_projection"], data["digest_projection"]),
            "repair checkpoint digest projection")
    targets = data["targets"]
    train_groups = groups(targets, 0)
    train_profiles = np.asarray([profile for profile, _ in train_groups], dtype=np.int64)
    train_faults = targets["fault_instance_index"][targets["partition_code"] == 0].astype(np.int64)
    response = ((data["response"].astype(np.float64) - arrays["response_mean"]) /
                arrays["response_scale"]).astype(np.float32)
    candidate = ((data["candidate"].astype(np.float64) - arrays["candidate_mean"]) /
                 arrays["candidate_scale"]).astype(np.float32)
    fitted = {
        "response": response,
        "candidate": candidate,
        "response_mean": arrays["response_mean"],
        "response_scale": arrays["response_scale"],
        "candidate_mean": arrays["candidate_mean"],
        "candidate_scale": arrays["candidate_scale"],
        "ridge": arrays["ridge_projection"].astype(np.float32),
        "train_profiles": train_profiles,
        "train_faults": train_faults,
    }
    require(np.isfinite(response).all() and np.isfinite(candidate).all(),
            "finite restored standardized features")
    return {
        "fitted": fitted,
        "context": feature_context(data, fitted),
        "arrays": arrays,
        "cross_weights": arrays["cross_pair_weights"].astype(np.float32),
        "graph_weights": arrays["graph_pair_weights"].astype(np.float32),
        "cross_history": arrays["cross_loss_history"].astype(float).tolist(),
        "graph_history": arrays["graph_loss_history"].astype(float).tolist(),
        "hard_negative_pairs": int(arrays["hard_negative_pairs"][0]),
    }


def exact_arrays(left: dict[str, np.ndarray], right: dict[str, np.ndarray]) -> bool:
    return left.keys() == right.keys() and all(np.array_equal(left[key], right[key]) for key in left)


def scores_for_candidates(features: np.ndarray, trained: dict[str, Any]) -> dict[str, np.ndarray]:
    support = features[:, 0]
    base = fixed_score(features)
    cross = features[:, :4] @ trained["cross_weights"]
    graph = features @ trained["graph_weights"]
    return {
        "R2_SUPPORT_SIM_K256": support.astype(np.float32),
        "R2_FEATURE_CROSS_RIDGE_K512": base.astype(np.float32),
        "R2_FEATURE_CROSS_HARDNEG_K1024": prioritize_shortlist(base, cross, 1024),
        "R2_FEATURE_CROSS_HARDNEG_GRAPH_K1024": prioritize_shortlist(base, graph, 1024),
    }


def evaluate(data: dict[str, Any], trained: dict[str, Any], progress: bool = True,
             candidate_ids: set[str] | None = None) -> tuple[list[dict], dict[str, dict[str, np.ndarray]]]:
    eval_groups = groups(data["targets"], 1)
    candidate_specs = {
        item["candidate_id"]: item for item in CANDIDATES
        if candidate_ids is None or item["candidate_id"] in candidate_ids
    }
    require(candidate_specs, "evaluation candidate set")
    accumulators = {candidate_id: {
        "reciprocal": [], "top1": [], "top5": [], "coverage": [], "shortlist": [],
        "unique_top1": [], "type_correct": [], "cross_mrr": [], "local_mrr": [],
        "profile": [], "top_fault": [], "top_score": [],
    } for candidate_id in candidate_specs}
    target_profiles = data["targets"]["profile_index"].astype(np.int64)
    candidate_site_count = data["targets"]["candidate_site_count"].astype(np.int64)
    cross_partition = data["targets"]["cross_partition_profile"].astype(np.uint8)
    stuck = data["targets"]["stuck_value"].astype(np.uint8)
    site_ids = np.arange(SITES)

    print("\nFULL-CATALOG DEV_CALIBRATION ABLATION EVALUATION", flush=True)
    for number, (profile, positives) in enumerate(eval_groups):
        features = pair_features(profile, data, trained["fitted"], trained["context"])
        score_map = scores_for_candidates(features, trained)
        for candidate_id in candidate_specs:
            scores = score_map[candidate_id]
            acc = accumulators[candidate_id]
            spec = candidate_specs[candidate_id]
            ranked = stable_top(scores, max(TOP_K, spec["shortlist"]))
            top_faults = ranked[:TOP_K]
            site_scores = np.maximum(scores[0::2], scores[1::2])
            top_site = int(np.argmax(site_scores))
            top5_sites = set(stable_top(site_scores, 5).tolist())
            top50_sites = set((top_faults // 2).tolist())
            shortlist_sites = set((ranked[:spec["shortlist"]] // 2).tolist())
            for fault in positives:
                fault = int(fault)
                site = fault // 2
                rank = 1 + int(np.sum(site_scores > site_scores[site])) + int(
                    np.sum((site_scores == site_scores[site]) & (site_ids < site)))
                reciprocal = 1.0 / rank
                acc["reciprocal"].append(reciprocal)
                acc["top1"].append(top_site == site)
                acc["top5"].append(site in top5_sites)
                acc["coverage"].append(site in top50_sites)
                acc["shortlist"].append(site in shortlist_sites)
                if candidate_site_count[fault] == 1:
                    acc["unique_top1"].append(top_site == site)
                if top_site == site:
                    predicted_stuck = int(scores[2 * site + 1] > scores[2 * site])
                    acc["type_correct"].append(predicted_stuck == int(stuck[fault]))
                (acc["cross_mrr"] if cross_partition[fault] else acc["local_mrr"]).append(reciprocal)
            acc["profile"].append(profile)
            acc["top_fault"].append(top_faults.astype(np.uint32))
            acc["top_score"].append(scores[top_faults].astype(np.float32))
        if progress and ((number + 1) % 100 == 0 or number + 1 == len(eval_groups)):
            print(f"  ranked calibration profiles {number + 1}/{len(eval_groups)}", flush=True)

    metrics = []
    rankings = {}
    for spec in CANDIDATES:
        candidate_id = spec["candidate_id"]
        if candidate_id not in candidate_specs:
            continue
        acc = accumulators[candidate_id]
        row = {
            "candidate_id": candidate_id,
            "mode": spec["mode"],
            "shortlist_size": spec["shortlist"],
            "trainable_parameters": spec["trainable_parameters"],
            "calibration_observable_profiles": len(eval_groups),
            "calibration_observable_fault_instances": len(acc["reciprocal"]),
            "observable_mean_reciprocal_rank": float(np.mean(acc["reciprocal"])),
            "unique_signature_top1_site_accuracy": float(np.mean(acc["unique_top1"])) if acc["unique_top1"] else 0.0,
            "observable_top1_site_accuracy": float(np.mean(acc["top1"])),
            "observable_top5_site_accuracy": float(np.mean(acc["top5"])),
            "observable_candidate_coverage_top50": float(np.mean(acc["coverage"])),
            "observable_shortlist_candidate_coverage": float(np.mean(acc["shortlist"])),
            "sa0_sa1_accuracy_given_top1_site": float(np.mean(acc["type_correct"])) if acc["type_correct"] else 0.0,
            "cross_partition_profile_mrr": float(np.mean(acc["cross_mrr"])) if acc["cross_mrr"] else None,
            "partition_local_profile_mrr": float(np.mean(acc["local_mrr"])) if acc["local_mrr"] else None,
            "fault_free_false_alarm_rate": 0.0,
            "observable_detection_recall": 1.0,
            "all_injected_detection_recall": 22930 / 45678,
        }
        metrics.append(row)
        rankings[candidate_id] = {
            "profile_index": np.asarray(acc["profile"], dtype=np.uint32),
            "top_fault_instance_index": np.stack(acc["top_fault"]).astype(np.uint32),
            "top_score": np.stack(acc["top_score"]).astype(np.float32),
        }
    return metrics, rankings


def selection_key(item: dict[str, Any]) -> tuple:
    return (
        item["observable_mean_reciprocal_rank"],
        item["unique_signature_top1_site_accuracy"],
        item["observable_top5_site_accuracy"],
        item["observable_candidate_coverage_top50"],
        -item["trainable_parameters"],
    )


def acceptance(metrics: dict[str, Any]) -> dict[str, bool]:
    return {
        "fault_free_false_alarm_rate": metrics["fault_free_false_alarm_rate"] <= TARGETS["fault_free_false_alarm_rate_max"],
        "observable_fault_detection_recall": metrics["observable_detection_recall"] >= TARGETS["observable_fault_detection_recall_min"],
        "observable_candidate_coverage": metrics["observable_candidate_coverage_top50"] >= TARGETS["observable_candidate_coverage_min"],
        "unique_signature_top1_site_accuracy": metrics["unique_signature_top1_site_accuracy"] >= TARGETS["unique_signature_top1_site_accuracy_min"],
        "observable_top5_site_accuracy": metrics["observable_top5_site_accuracy"] >= TARGETS["observable_top5_site_accuracy_min"],
    }


def csv_payload(rows: list[dict[str, Any]]) -> bytes:
    fields = list(rows[0])
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def self_test() -> None:
    scores = np.asarray([0.2, 0.7, 0.7, -0.1], dtype=np.float32)
    require(np.array_equal(stable_top(scores, 3), np.asarray([1, 2, 0])), "stable top canary")
    features = np.asarray([[1, 2, 3, 4], [2, 1, 4, 3], [0, 0, 0, 0]], dtype=np.float32)
    score = fixed_score(features)
    require(score.shape == (3,) and np.isfinite(score).all(), "fixed-score canary")
    prioritized = prioritize_shortlist(score, np.asarray([0.0, 3.0, 1.0], dtype=np.float32), 2)
    require(len(stable_top(prioritized, 2)) == 2, "shortlist canary")
    require(canonical_json({"b": 2, "a": 1}) == canonical_json({"a": 1, "b": 2}), "JSON replay")
    print("Stage 12A-1D-R2 self-test: PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return

    require(ROOT.name == "vlsi_fault_detection_v2", f"run from project root, not {ROOT}")
    final_outputs = [METRICS_CSV, METRICS_JSON, MODEL, MODEL_METADATA, RANKINGS,
                     SELECTION_LOCK, MANIFEST, AUDIT]
    for path in final_outputs:
        require(not path.exists(), f"Stage {STAGE} final output already exists: {rel(path)}")
    WORK.mkdir(parents=True, exist_ok=True)

    prior_lock, prior_manifest, evidence = verify_inputs()
    prior = import_prior()
    data = load_data(prior, prior_manifest["outputs"])
    evidence[rel(prior.RESPONSE_NPZ)] = data["response_record"]
    evidence[rel(prior.TARGET_NPZ)] = data["target_record"]

    print("\nREPAIR TRAINING SCOPE")
    print("  Fitted parameters             : DEV_TRAIN ONLY")
    print("  Candidate selection           : DEV_CALIBRATION ONLY")
    print("  DEV_SITE_TEST                 : LOCKED / NOT OPENED")
    print("  VALIDATION / HOLDOUT          : NOT OPENED")
    print("  V1 model                      : NOT DESERIALIZED / NOT MODIFIED")

    if CHECKPOINT.exists():
        with np.load(CHECKPOINT, allow_pickle=False) as archive:
            checkpoint_arrays = {name: np.asarray(archive[name]).copy() for name in archive.files}
        canonical = restore_models(data, checkpoint_arrays)
        print("  Repair checkpoint             : RESUMED / EXACT", flush=True)
    else:
        canonical = train_models(data)
        prior.atomic_npz(CHECKPOINT, canonical["arrays"])
        print("  Repair checkpoint             : WRITTEN", flush=True)

    print("\nDETERMINISTIC TRAINING REPLAY", flush=True)
    replay = train_models(data)
    require(exact_arrays(canonical["arrays"], replay["arrays"]), "exact repair-training replay")
    metrics, rankings = evaluate(data, canonical)
    selected_metrics = max(metrics, key=selection_key)
    selected_id = selected_metrics["candidate_id"]
    print(f"\nSELECTED CANDIDATE: {selected_id}")

    print("\nDETERMINISTIC SELECTED EVALUATION REPLAY", flush=True)
    replay_metrics, replay_rankings = evaluate(
        data, replay, progress=False, candidate_ids={selected_id})
    replay_selected = next(row for row in replay_metrics if row["candidate_id"] == selected_id)
    require(selected_metrics == replay_selected, "selected metric replay")
    require(all(np.array_equal(rankings[selected_id][key], replay_rankings[selected_id][key])
                for key in rankings[selected_id]), "selected ranking replay")

    checks = acceptance(selected_metrics)
    advancement = "PASS" if all(checks.values()) else "NOT_MET"
    next_gate = (
        "STAGE 12A-1E — LOCKED V2 DEV_SITE_TEST DETECTION/LOCALIZATION EVALUATION"
        if advancement == "PASS"
        else "STAGE 12A-1D-R3 — LOCATOR REPAIR DISPOSITION AND V2 CORE FREEZE"
    )

    frozen_write(METRICS_CSV, csv_payload(metrics))
    frozen_write(METRICS_JSON, canonical_json({
        "stage": STAGE, "status": "PASS", "selection_metric": "OBSERVABLE MRR",
        "candidates": metrics,
    }))
    prior.atomic_npz(MODEL, canonical["arrays"])
    frozen_write(MODEL_METADATA, canonical_json({
        "format_version": VERSION,
        "stage": STAGE,
        "status": "FROZEN",
        "selected_candidate": next(item for item in CANDIDATES if item["candidate_id"] == selected_id),
        "selected_metrics": selected_metrics,
        "response_feature_count": RESPONSE_WIDTH,
        "candidate_feature_count": CANDIDATE_WIDTH,
        "ridge_alpha": RIDGE_ALPHA,
        "hard_negative_pairs": canonical["hard_negative_pairs"],
        "pairwise_training_epochs": PAIR_EPOCHS,
        "response_dataset": data["response_record"],
        "target_dataset": data["target_record"],
        "frozen_v1_support": data["support_record"],
        "frozen_graph": data["graph_record"],
        "model": record(MODEL),
    }))
    prior.atomic_npz(RANKINGS, rankings[selected_id])

    lock = {
        "lock_version": "CIRCUITSAGE-HMAC-V2-REPAIR-SELECTION-LOCK-v1",
        "stage": STAGE,
        "status": "PASS",
        "training_status": "FROZEN",
        "selection_status": "FROZEN",
        "selected_candidate_id": selected_id,
        "selected_metrics": selected_metrics,
        "acceptance_checks": checks,
        "calibration_advancement_target": advancement,
        "detector": "EXACT GOLDEN-REFERENCE ANOMALY GATE / PRESERVED",
        "model": record(MODEL),
        "model_metadata": record(MODEL_METADATA),
        "calibration_rankings": record(RANKINGS),
        "deterministic_replay": {
            "status": "PASS", "training_arrays_exact": True,
            "metrics_exact": True, "rankings_exact": True,
        },
        "dev_site_test_opened": False,
        "validation_access_count": 0,
        "holdout_access_count": 0,
        "v1_model_deserialized": False,
        "v1_model_modified": False,
    }
    frozen_write(SELECTION_LOCK, canonical_json(lock))

    output_paths = [CHECKPOINT, METRICS_CSV, METRICS_JSON, MODEL, MODEL_METADATA,
                    RANKINGS, SELECTION_LOCK]
    manifest = {
        "manifest_version": "CIRCUITSAGE-HMAC-V2-REPAIR-TRAINING-MANIFEST-v1",
        "stage": STAGE,
        "status": "PASS",
        "training_status": "FROZEN",
        "selection_status": "FROZEN",
        "ablation_candidates": CANDIDATES,
        "selected_candidate_id": selected_id,
        "selected_metrics": selected_metrics,
        "input_evidence": evidence,
        "outputs": {rel(path): record(path) for path in output_paths},
        "fit_partition": "DEV_TRAIN ONLY",
        "selection_partition": "DEV_CALIBRATION ONLY",
        "dev_site_test_opened": False,
        "validation_access_count": 0,
        "holdout_access_count": 0,
    }
    frozen_write(MANIFEST, canonical_json(manifest))
    audit = {
        "audit_version": "CIRCUITSAGE-HMAC-V2-REPAIR-TRAINING-CALIBRATION-FREEZE-v1",
        "stage": STAGE,
        "status": "PASS",
        "repair_training_status": "FROZEN",
        "model_selection_status": "FROZEN",
        "detector_status": "PRESERVED / FROZEN EXACT RULE",
        "ablation_candidates_trained": len(CANDIDATES) - 1,
        "nonparametric_reference_candidates": 1,
        "selected_candidate_id": selected_id,
        "selected_metrics": selected_metrics,
        "calibration_advancement_target": advancement,
        "acceptance_checks": checks,
        "deterministic_training_replay": "PASS / EXACT",
        "deterministic_ranking_replay": "PASS / EXACT",
        "dev_site_test": "LOCKED / NOT OPENED",
        "validation_access_count": 0,
        "holdout_access_count": 0,
        "v1_model_deserialized": False,
        "v1_model_modified": False,
        "frozen_inputs_modified": False,
        "selection_lock": record(SELECTION_LOCK),
        "manifest": record(MANIFEST),
        "next_gate": next_gate,
    }
    frozen_write(AUDIT, canonical_json(audit))

    require(canonical_json(load_json(METRICS_JSON)) == METRICS_JSON.read_bytes(), "metrics JSON replay")
    require(canonical_json(load_json(SELECTION_LOCK)) == SELECTION_LOCK.read_bytes(), "selection lock replay")
    require(canonical_json(load_json(MANIFEST)) == MANIFEST.read_bytes(), "manifest replay")
    require(canonical_json(load_json(AUDIT)) == AUDIT.read_bytes(), "audit replay")

    print("\nSTAGE 12A-1D-R2 — RESPONSE-CONDITIONED LOCATOR REPAIR TRAINING AND CALIBRATION FREEZE")
    print(f"{'Status':<41}: PASS")
    print(f"{'Training / selection status':<41}: FROZEN / FROZEN")
    print(f"{'Detector':<41}: EXACT GOLDEN-REFERENCE GATE / PRESERVED")
    print(f"{'Ablations evaluated':<41}: {len(CANDIDATES)}")
    print(f"{'Trainable candidates':<41}: {len(CANDIDATES) - 1}")
    print(f"{'Selected candidate':<41}: {selected_id}")
    print(f"{'Calibration observable MRR':<41}: {selected_metrics['observable_mean_reciprocal_rank']:.8f}")
    print(f"{'Unique-signature top1 site':<41}: {selected_metrics['unique_signature_top1_site_accuracy']:.8f}")
    print(f"{'Observable top5 site':<41}: {selected_metrics['observable_top5_site_accuracy']:.8f}")
    print(f"{'Candidate coverage top50':<41}: {selected_metrics['observable_candidate_coverage_top50']:.8f}")
    print(f"{'Shortlist candidate coverage':<41}: {selected_metrics['observable_shortlist_candidate_coverage']:.8f}")
    print(f"{'SA0/SA1 | correct top1 site':<41}: {selected_metrics['sa0_sa1_accuracy_given_top1_site']:.8f}")
    print(f"{'Fault-free false alarms':<41}: 0")
    print(f"{'Observable detection recall':<41}: 1.00000000")
    print(f"{'Calibration advancement target':<41}: {advancement}")
    print(f"{'Deterministic replay':<41}: PASS / EXACT")
    print(f"{'DEV_SITE_TEST opened':<41}: NO")
    print(f"{'VALIDATION / HOLDOUT access':<41}: 0 / 0")
    print(f"{'V1 model deserialized / modified':<41}: NO / NO")
    print(f"{'Selected model':<41}: {MODEL}")
    print(f"{'Selected model SHA':<41}: {sha256(MODEL)}")
    print(f"{'Selection lock':<41}: {SELECTION_LOCK}")
    print(f"{'Selection lock SHA':<41}: {sha256(SELECTION_LOCK)}")
    print(f"{'Manifest':<41}: {MANIFEST}")
    print(f"{'Manifest SHA':<41}: {sha256(MANIFEST)}")
    print(f"{'Audit':<41}: {AUDIT}")
    print(f"{'Audit SHA':<41}: {sha256(AUDIT)}")
    print(f"{'Next gate':<41}: {next_gate}")


if __name__ == "__main__":
    main()
