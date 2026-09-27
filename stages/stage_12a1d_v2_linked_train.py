#!/usr/bin/env python3
"""Stage 12A-1D: train/calibrate the linked CircuitSage-HMAC V2 ranker.

The detector is the frozen exact golden-response deviation gate.  The learned
component ranks all 45,678 frozen SA0/SA1 candidates from a blinded response.
Only DEV_TRAIN drives gradients and DEV_CALIBRATION selects the candidate.
DEV_SITE_TEST, VALIDATION, and HOLDOUT are not opened.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Any

try:
    import joblib
    import numpy as np
except ImportError as error:
    raise SystemExit("STOP: activate the frozen project .venv (NumPy and joblib required)") from error


STAGE = "12A-1D"
VERSION = "CIRCUITSAGE-HMAC-V2-LINKED-TRAINER-v1"
ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "config" / "v2"
RESULT = ROOT / "results" / "circuitsage_hmac_v2_12a1"
DATASET = RESULT / "blinded_train_response_dataset_12a1b"
WORK = RESULT / "v2_training_12a1d"
CHECKPOINTS = WORK / "checkpoints"

CONTRACT_SOURCE = ROOT / "stage_12a1c_v2_detector_locator_contract.py"
ARCHITECTURE = CONFIG / "circuitsage_hmac_v2_detector_locator_architecture_12a1c.json"
CONTRACT = CONFIG / "circuitsage_hmac_v2_training_contract_12a1c.json"
GRID = CONFIG / "circuitsage_hmac_v2_candidate_grid_12a1c.csv"
ENVIRONMENT = RESULT / "circuitsage_hmac_v2_environment_12a1c.json"
CONTRACT_AUDIT = RESULT / "circuitsage_hmac_v2_architecture_training_contract_freeze_12a1c.json"

RESPONSE_NPZ = DATASET / "circuitsage_hmac_v2_train_response_features_12a1b.npz"
TARGET_NPZ = DATASET / "circuitsage_hmac_v2_train_response_targets_12a1b.npz"
RESPONSE_SCHEMA = DATASET / "circuitsage_hmac_v2_train_response_schema_12a1b.json"
DATASET_MANIFEST = RESULT / "circuitsage_hmac_v2_train_response_manifest_12a1b.json"
DATASET_AUDIT = RESULT / "circuitsage_hmac_v2_train_response_dataset_freeze_12a1b.json"

FEATURE_ROOT = ROOT / "results/hmac_fault_campaign_11c5/feature_matrix_11c5e"
TRAIN_MATRIX = FEATURE_ROOT / "hmac_dev_train_feature_matrix_11c5e.npz"
CAL_MATRIX = FEATURE_ROOT / "hmac_dev_calibration_feature_matrix_11c5e.npz"
FEATURE_SCHEMA = FEATURE_ROOT / "hmac_leakage_safe_feature_matrix_schema_11c5e.json"
FEATURE_MANIFEST = ROOT / "results/hmac_fault_campaign_11c5/hmac_leakage_safe_feature_matrix_manifest_11c5e.json"
GNN_LOCK = ROOT / "results/hmac_fault_campaign_11d1/gnn_training_11d1c/hmac_gnn_selection_lock_11d1c.json"
V1_MODEL = ROOT / "results/hmac_fault_campaign_11d2/hybrid_training_11d2b/hmac_selected_hybrid_model_11d2b.joblib"
V1_LOCK = ROOT / "results/hmac_fault_campaign_11d2/hybrid_training_11d2b/hmac_hybrid_selection_lock_11d2b.json"

SUPPORT_CACHE = WORK / "circuitsage_hmac_v2_frozen_v1_support_12a1d.npz"
SCALERS = WORK / "circuitsage_hmac_v2_ranker_scalers_12a1d.npz"
METRICS_CSV = WORK / "circuitsage_hmac_v2_candidate_calibration_metrics_12a1d.csv"
METRICS_JSON = WORK / "circuitsage_hmac_v2_candidate_calibration_metrics_12a1d.json"
HISTORY_CSV = WORK / "circuitsage_hmac_v2_training_history_12a1d.csv"
SELECTED_MODEL = WORK / "circuitsage_hmac_v2_selected_ranker_12a1d.npz"
SELECTED_METADATA = WORK / "circuitsage_hmac_v2_selected_ranker_metadata_12a1d.json"
CAL_RANKINGS = WORK / "circuitsage_hmac_v2_selected_calibration_rankings_12a1d.npz"
SELECTION_LOCK = WORK / "circuitsage_hmac_v2_selection_lock_12a1d.json"
MANIFEST = RESULT / "circuitsage_hmac_v2_training_manifest_12a1d.json"
AUDIT = RESULT / "circuitsage_hmac_v2_training_calibration_freeze_12a1d.json"

PINNED = {
    CONTRACT_SOURCE: "6a5a9e429b7ee1522ad5d21a6623a25ee58cf060fe3c561f0266c2be918a741a",
    RESPONSE_NPZ: "09f5dded67a6f82d14f02b4de23765b66ff954eb6a9779817c8b7286dc6ac2b3",
    TARGET_NPZ: "268e407a9d13adab6a81b4650b4c16761dffb99d6d3d7a32ab2c28ff9a57a572",
    RESPONSE_SCHEMA: "5743d02ad5a8067aef331e955b6e91acd599b066956b0e572c7d25b2c1f98032",
    DATASET_MANIFEST: "dd667b287285e0fb6cca6930de8415bc97b2a8b4213840205aa7b4b842dbdbca",
    DATASET_AUDIT: "e4b5e6f3aa60c2f49668ed5105549898729af5985e5e8db7d8eabc1e86aaf222",
    TRAIN_MATRIX: "b4feebe3080ac540edf38dcde35212b790dfb0f93fa57f1f5fc71411bde3b6b0",
    CAL_MATRIX: "ee18d68e4b1c742e8b8b163ab3f97f92f954e3da6af88f460b5b886ea73dd05b",
    FEATURE_SCHEMA: "0bf1edffb8078b003a1116b276615d5544979d007712ab49870b1b93784e486c",
    FEATURE_MANIFEST: "413cbad01681b4f209de6fce44d3777bccfe424c50c3c95582c627e44965e186",
    GNN_LOCK: "3f5b672d0bf152f4e68eaaeabf395cabf9ff7fa49a9807845ec5e104c0596e0b",
    V1_MODEL: "12fea5eabf4a6c605325b6ce4c1217f6a37cc59750065db8b85c3da14713f6b8",
    V1_LOCK: "005cecbd94456d3d1aa6805b2dc898e895cc178564673b6b0aa269fcd64260c6",
}

SITES = 22839
FAULTS = 45678
VECTORS = 64
RESPONSE_WIDTH = 2304
GRAPH_WIDTH = 119
CANDIDATE_WIDTH = 185
TOP_K = 50
V1_BATCH_CANDIDATES = 256
RANK_BATCH = 8192


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
    return str(path.relative_to(ROOT))


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON object required: {rel(path)}")
    return value


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode()


def atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.exists(), f"refusing to overwrite output: {rel(path)}")
    temporary = path.with_name(path.name + ".tmp")
    require(not temporary.exists(), f"stale temporary file: {rel(temporary)}")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def array_npy(array: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.lib.format.write_array(stream, np.ascontiguousarray(array), allow_pickle=False)
    return stream.getvalue()


def npz_bytes(arrays: dict[str, np.ndarray]) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name in sorted(arrays):
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, array_npy(arrays[name]))
    return stream.getvalue()


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    data = npz_bytes(arrays)
    require(data == npz_bytes(arrays), f"NPZ replay: {path.name}")
    atomic_bytes(path, data)


def record(path: Path) -> dict[str, Any]:
    return {"path": rel(path), "sha256": sha256(path), "bytes": path.stat().st_size}


def resolve_record_path(item: dict[str, Any]) -> Path:
    value = item.get("path")
    require(isinstance(value, str) and value, "artifact path record")
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def verify_contract_bundle() -> tuple[dict, dict, list[dict], dict]:
    require(CONTRACT_AUDIT.is_file(), "Stage 12A-1C has not been run")
    audit = load_json(CONTRACT_AUDIT)
    require(audit.get("status") == "PASS", "Stage 12A-1C status")
    require(audit.get("architecture_status") == "FROZEN", "12A-1C architecture status")
    require(audit.get("training_contract_status") == "FROZEN", "12A-1C training status")
    require(audit.get("dev_site_test_opened") is False, "12A-1C site-test state")
    frozen = audit.get("frozen_outputs")
    require(isinstance(frozen, dict), "12A-1C frozen outputs")
    for path in (ARCHITECTURE, CONTRACT, GRID, ENVIRONMENT):
        item = frozen.get(rel(path))
        require(isinstance(item, dict), f"12A-1C output record: {path.name}")
        require(path.is_file() and sha256(path) == item.get("sha256"), f"12A-1C output changed: {path.name}")

    architecture = load_json(ARCHITECTURE)
    contract = load_json(CONTRACT)
    environment = load_json(ENVIRONMENT)
    require(architecture.get("status") == "FROZEN", "architecture freeze")
    require(contract.get("status") == "FROZEN", "training-contract freeze")
    require(contract.get("source_partition") == "TRAIN RESPONSES ONLY", "training source")
    require(contract.get("backend") == "CUSTOM DETERMINISTIC NUMPY BACKPROPAGATION / ADAMW", "training backend")
    require(environment.get("packages", {}).get("numpy") == np.__version__, "NumPy environment changed")
    require(environment.get("packages", {}).get("joblib") == joblib.__version__, "joblib environment changed")

    rows: list[dict] = []
    with GRID.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        for raw in reader:
            row = dict(raw)
            for key in ("trainable_parameters", "epochs", "negative_candidates_per_query", "query_batch_size", "random_seed"):
                row[key] = int(row[key])
            for key in ("learning_rate", "weight_decay"):
                row[key] = float(row[key])
            for key in ("response_hidden", "candidate_hidden", "fusion_hidden"):
                row[key] = [int(value) for value in row[key].split(";")]
            rows.append(row)
    require(len(rows) == 2, "candidate-grid count")
    return architecture, contract, rows, audit


def verify_inputs() -> dict[str, dict[str, Any]]:
    print("FROZEN INPUT VERIFICATION")
    evidence = {}
    for path, expected in PINNED.items():
        require(path.is_file(), f"missing frozen input: {rel(path)}")
        require(sha256(path) == expected, f"frozen input SHA: {rel(path)}")
        evidence[rel(path)] = record(path)
        print(f"  {path.name:<78}: OK", flush=True)
    return evidence


def load_response_data() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    with np.load(RESPONSE_NPZ, allow_pickle=False) as archive:
        required = {"profile_timeout", "profile_latency_delta", "profile_digest_xor", "profile_digest_hamming", "profile_deviation"}
        require(required.issubset(archive.files), "response NPZ members")
        response = np.concatenate([
            np.asarray(archive["profile_timeout"], dtype=np.float32),
            np.asarray(archive["profile_latency_delta"], dtype=np.float32),
            np.asarray(archive["profile_digest_xor"], dtype=np.float32).reshape(len(archive["profile_timeout"]), -1),
            np.asarray(archive["profile_digest_hamming"], dtype=np.float32),
            np.asarray(archive["profile_deviation"], dtype=np.float32),
        ], axis=1)
    require(response.ndim == 2 and response.shape[1] == RESPONSE_WIDTH, "response matrix shape")
    require(np.isfinite(response).all(), "finite response features")
    with np.load(TARGET_NPZ, allow_pickle=False) as archive:
        targets = {name: np.asarray(archive[name]).copy() for name in archive.files}
    require(len(targets["fault_instance_index"]) == FAULTS, "target fault count")
    require(np.array_equal(targets["fault_instance_index"], np.arange(FAULTS)), "fault ordering")
    require(int(np.sum(targets["observable"])) == 22930, "observable count")
    return response, targets


def graph_cache() -> tuple[np.ndarray, dict[str, Any]]:
    lock = load_json(GNN_LOCK)
    item = lock.get("propagation_cache")
    require(isinstance(item, dict), "GNN propagation-cache record")
    path = resolve_record_path(item)
    require(path.is_file() and sha256(path) == item.get("sha256"), "GNN propagation cache integrity")
    with np.load(path, allow_pickle=False) as archive:
        require("k3_features" in archive.files, "K3 graph cache")
        graph = np.asarray(archive["k3_features"], dtype=np.float32).copy()
    require(graph.shape == (SITES, GRAPH_WIDTH), "K3 graph feature shape")
    require(np.isfinite(graph).all(), "finite K3 graph features")
    return graph, record(path)


def site_feature_catalog() -> tuple[np.ndarray, dict[str, Any]]:
    manifest = load_json(FEATURE_MANIFEST)
    item = manifest.get("site_feature_catalog")
    require(isinstance(item, dict), "site-feature catalog record")
    path = resolve_record_path(item)
    require(path.is_file() and sha256(path) == item.get("sha256"), "site-feature catalog integrity")
    schema = load_json(FEATURE_SCHEMA)
    vocabulary = schema.get("categorical_vocabulary", {})
    drivers = list(vocabulary.get("driver_cell_type", []))
    categories = list(vocabulary.get("site_category", []))
    require(len(drivers) == 9 and len(categories) == 2, "site categorical vocabulary")
    matrix = np.zeros((SITES, 14), dtype=np.float32)
    seen = np.zeros(SITES, dtype=np.uint8)
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            index = int(row["site_index"]) - 1
            require(0 <= index < SITES and not seen[index], "site catalog index")
            matrix[index, 0] = float(row["cell_fanout_z"])
            matrix[index, 1] = float(row["is_primary_output_stem"])
            matrix[index, 2] = float(row["is_sequential_stem"])
            matrix[index, 3 + int(row["driver_cell_type_code"])] = 1.0
            matrix[index, 3 + len(drivers) + int(row["site_category_code"])] = 1.0
            seen[index] = 1
    require(int(seen.sum()) == SITES and np.isfinite(matrix).all(), "site-feature catalog completeness")
    return matrix, record(path)


def vector_features() -> np.ndarray:
    with np.load(TRAIN_MATRIX, allow_pickle=False) as archive:
        vectors = np.asarray(archive["vector_features"], dtype=np.float32).copy()
        require(archive["partition"].tolist() == ["DEV_TRAIN"], "train matrix identity")
    require(vectors.shape == (VECTORS, 512), "vector feature shape")
    return vectors


def build_v1_support(site: np.ndarray, vectors: np.ndarray, graph: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    if SUPPORT_CACHE.exists():
        with np.load(SUPPORT_CACHE, allow_pickle=False) as archive:
            support = np.asarray(archive["v1_probability"], dtype=np.float32).copy()
            model_sha = archive["v1_model_sha256"].tolist()[0]
        require(support.shape == (FAULTS, VECTORS), "support cache shape")
        require(model_sha == PINNED[V1_MODEL], "support cache model SHA")
        print("  Frozen V1 support cache      : RESUMED / VERIFIED", flush=True)
        return support, record(SUPPORT_CACHE)

    print("\nFROZEN V1 CANDIDATE-SUPPORT INFERENCE", flush=True)
    model = joblib.load(V1_MODEL)
    require(hasattr(model, "predict_proba") and int(model.n_features_in_) == 646, "frozen V1 model interface")
    positive = int(np.flatnonzero(np.asarray(model.classes_) == 1)[0])
    support = np.empty((FAULTS, VECTORS), dtype=np.float32)
    for start in range(0, FAULTS, V1_BATCH_CANDIDATES):
        stop_index = min(start + V1_BATCH_CANDIDATES, FAULTS)
        candidates = np.arange(start, stop_index, dtype=np.int64)
        sites = candidates // 2
        stuck = (candidates % 2).astype(np.float32)
        count = len(candidates)
        matrix = np.empty((count * VECTORS, 646), dtype=np.float32)
        matrix[:, 0] = np.repeat(stuck, VECTORS)
        matrix[:, 1:15] = np.repeat(site[sites], VECTORS, axis=0)
        matrix[:, 15:527] = np.tile(vectors, (count, 1))
        matrix[:, 527:] = np.repeat(graph[sites], VECTORS, axis=0)
        probability = model.predict_proba(matrix)[:, positive].reshape(count, VECTORS)
        support[start:stop_index] = probability.astype(np.float32)
        if stop_index == FAULTS or (stop_index // V1_BATCH_CANDIDATES) % 20 == 0:
            print(f"  candidate support {stop_index}/{FAULTS}", flush=True)
    require(np.isfinite(support).all() and np.all((support >= 0) & (support <= 1)), "V1 support probabilities")
    atomic_npz(SUPPORT_CACHE, {
        "fault_instance_index": np.arange(FAULTS, dtype="<u4"),
        "v1_model_sha256": np.asarray([PINNED[V1_MODEL]]),
        "v1_probability": support.astype("<f4"),
    })
    return support, record(SUPPORT_CACHE)


def candidate_matrix(graph: np.ndarray, support: np.ndarray) -> np.ndarray:
    matrix = np.empty((FAULTS, CANDIDATE_WIDTH), dtype=np.float32)
    matrix[:, :GRAPH_WIDTH] = np.repeat(graph, 2, axis=0)
    matrix[:, GRAPH_WIDTH:GRAPH_WIDTH + 2] = 0.0
    matrix[np.arange(FAULTS), GRAPH_WIDTH + (np.arange(FAULTS) % 2)] = 1.0
    matrix[:, GRAPH_WIDTH + 2:] = support
    require(np.isfinite(matrix).all(), "finite candidate matrix")
    return matrix


def fit_scalers(response: np.ndarray, targets: dict[str, np.ndarray], candidates: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    train_fault = targets["partition_code"] == 0
    train_observable = train_fault & (targets["observable"] == 1)
    train_profiles = np.unique(targets["profile_index"][train_observable].astype(np.int64))
    train_candidates = targets["fault_instance_index"][train_fault].astype(np.int64)
    r_mean = response[train_profiles].mean(axis=0, dtype=np.float64)
    r_scale = response[train_profiles].std(axis=0, dtype=np.float64)
    r_scale[r_scale == 0] = 1.0
    c_mean = candidates[train_candidates].mean(axis=0, dtype=np.float64)
    c_scale = candidates[train_candidates].std(axis=0, dtype=np.float64)
    c_scale[c_scale == 0] = 1.0
    scaled_r = ((response.astype(np.float64) - r_mean) / r_scale).astype(np.float32)
    scaled_c = ((candidates.astype(np.float64) - c_mean) / c_scale).astype(np.float32)
    require(np.isfinite(scaled_r).all() and np.isfinite(scaled_c).all(), "scaled matrices")
    if not SCALERS.exists():
        atomic_npz(SCALERS, {
            "response_mean": r_mean.astype("<f8"), "response_scale": r_scale.astype("<f8"),
            "candidate_mean": c_mean.astype("<f8"), "candidate_scale": c_scale.astype("<f8"),
            "train_profile_index": train_profiles.astype("<u4"),
            "train_fault_instance_index": train_candidates.astype("<u4"),
        })
    else:
        with np.load(SCALERS, allow_pickle=False) as archive:
            require(np.array_equal(archive["response_mean"], r_mean), "resumed response scaler")
            require(np.array_equal(archive["candidate_mean"], c_mean), "resumed candidate scaler")
    return scaled_r, scaled_c, record(SCALERS)


def groups(targets: dict[str, np.ndarray], partition_code: int) -> list[tuple[int, np.ndarray]]:
    mask = (targets["partition_code"] == partition_code) & (targets["observable"] == 1)
    profiles = targets["profile_index"].astype(np.int64)
    faults = targets["fault_instance_index"].astype(np.int64)
    result = []
    for profile in np.unique(profiles[mask]):
        positive = faults[mask & (profiles == profile)]
        require(len(positive) > 0, "positive group")
        result.append((int(profile), positive))
    return result


def init_layers(widths: list[int], rng: np.random.Generator) -> list[dict[str, np.ndarray]]:
    layers = []
    for left, right in zip(widths, widths[1:]):
        weight = (rng.standard_normal((left, right)) * math.sqrt(2.0 / left)).astype(np.float32)
        layers.append({"w": weight, "b": np.zeros(right, dtype=np.float32)})
    return layers


def new_model(candidate: dict) -> dict[str, Any]:
    rng = np.random.Generator(np.random.PCG64(candidate["random_seed"]))
    return {
        "response": init_layers([RESPONSE_WIDTH, *candidate["response_hidden"]], rng),
        "candidate": init_layers([CANDIDATE_WIDTH, *candidate["candidate_hidden"]], rng),
        "fusion": init_layers([candidate["response_hidden"][-1] + candidate["candidate_hidden"][-1], *candidate["fusion_hidden"], 1], rng),
    }


def forward(layers: list[dict[str, np.ndarray]], x: np.ndarray, final_linear: bool) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray, bool]]]:
    cache = []
    value = x
    for index, layer in enumerate(layers):
        z = value @ layer["w"] + layer["b"]
        relu = not (final_linear and index == len(layers) - 1)
        cache.append((value, z, relu))
        value = np.maximum(z, 0.0) if relu else z
    return value, cache


def backward(layers: list[dict[str, np.ndarray]], cache: list, gradient: np.ndarray) -> tuple[np.ndarray, list[dict[str, np.ndarray]]]:
    grads = [None] * len(layers)
    current = gradient
    for index in range(len(layers) - 1, -1, -1):
        x, z, relu = cache[index]
        if relu:
            current = current * (z > 0)
        grads[index] = {"w": x.T @ current, "b": current.sum(axis=0)}
        current = current @ layers[index]["w"].T
    return current, grads


def zero_optimizer(model: dict[str, Any]) -> dict[str, Any]:
    state = {"step": 0, "m": {}, "v": {}}
    for section in ("response", "candidate", "fusion"):
        for index, layer in enumerate(model[section]):
            for name in ("w", "b"):
                key = f"{section}_{index}_{name}"
                state["m"][key] = np.zeros_like(layer[name])
                state["v"][key] = np.zeros_like(layer[name])
    return state


def adamw(model: dict, optimizer: dict, grads: dict, learning_rate: float, weight_decay: float) -> None:
    optimizer["step"] += 1
    step = optimizer["step"]
    beta1, beta2, epsilon = 0.9, 0.999, 1e-8
    for section in ("response", "candidate", "fusion"):
        for index, layer in enumerate(model[section]):
            for name in ("w", "b"):
                key = f"{section}_{index}_{name}"
                gradient = grads[section][index][name].astype(np.float32)
                norm = float(np.linalg.norm(gradient))
                if norm > 5.0:
                    gradient *= np.float32(5.0 / norm)
                m, v = optimizer["m"][key], optimizer["v"][key]
                m *= beta1; m += (1.0 - beta1) * gradient
                v *= beta2; v += (1.0 - beta2) * gradient * gradient
                update = (m / (1.0 - beta1 ** step)) / (np.sqrt(v / (1.0 - beta2 ** step)) + epsilon)
                if name == "w":
                    update = update + weight_decay * layer[name]
                layer[name] -= np.float32(learning_rate) * update.astype(np.float32)


def model_arrays(model: dict, optimizer: dict | None = None) -> dict[str, np.ndarray]:
    arrays = {}
    for section in ("response", "candidate", "fusion"):
        for index, layer in enumerate(model[section]):
            arrays[f"model_{section}_{index}_w"] = layer["w"].astype("<f4")
            arrays[f"model_{section}_{index}_b"] = layer["b"].astype("<f4")
    if optimizer is not None:
        arrays["optimizer_step"] = np.asarray([optimizer["step"]], dtype="<u8")
        for moment in ("m", "v"):
            for key, value in optimizer[moment].items():
                arrays[f"optimizer_{moment}_{key}"] = value.astype("<f4")
    return arrays


def parameter_count(model: dict) -> int:
    return sum(
        int(value.size)
        for section in ("response", "candidate", "fusion")
        for layer in model[section]
        for value in (layer["w"], layer["b"])
    )


def restore_model(path: Path, candidate: dict) -> tuple[dict, dict, int]:
    model = new_model(candidate)
    optimizer = zero_optimizer(model)
    metadata = load_json(path.with_suffix(".json"))
    require(metadata.get("candidate") == candidate, "checkpoint candidate contract")
    require(metadata.get("checkpoint_sha256") == sha256(path), "checkpoint integrity")
    with np.load(path, allow_pickle=False) as archive:
        for section in ("response", "candidate", "fusion"):
            for index, layer in enumerate(model[section]):
                for name in ("w", "b"):
                    layer[name][...] = archive[f"model_{section}_{index}_{name}"]
        optimizer["step"] = int(archive["optimizer_step"][0])
        for moment in ("m", "v"):
            for key in optimizer[moment]:
                optimizer[moment][key][...] = archive[f"optimizer_{moment}_{key}"]
        epoch = int(archive["completed_epochs"][0])
    return model, optimizer, epoch


def sample_negatives(rng: np.random.Generator, universe: np.ndarray, positives: np.ndarray, count: int) -> np.ndarray:
    chosen = []
    positive_set = set(int(value) for value in positives)
    while len(chosen) < count:
        draw = rng.choice(universe, size=min(count * 2, len(universe)), replace=False)
        for value in draw:
            integer = int(value)
            if integer not in positive_set and integer not in chosen:
                chosen.append(integer)
                if len(chosen) == count:
                    break
    return np.asarray(chosen, dtype=np.int64)


def train_batch(model: dict, optimizer: dict, response: np.ndarray, candidate: np.ndarray,
                batch_groups: list[tuple[int, np.ndarray]], universe: np.ndarray,
                rng: np.random.Generator, negative_count: int, learning_rate: float,
                weight_decay: float) -> float:
    query_index = np.asarray([item[0] for item in batch_groups], dtype=np.int64)
    pair_candidates = []
    owner = []
    positive_counts = []
    for group_index, (_, positives) in enumerate(batch_groups):
        negatives = sample_negatives(rng, universe, positives, negative_count)
        pair_candidates.extend(positives.tolist()); pair_candidates.extend(negatives.tolist())
        owner.extend([group_index] * (len(positives) + len(negatives)))
        positive_counts.append(len(positives))
    pair_candidates = np.asarray(pair_candidates, dtype=np.int64)
    owner = np.asarray(owner, dtype=np.int64)

    q, q_cache = forward(model["response"], response[query_index], False)
    c, c_cache = forward(model["candidate"], candidate[pair_candidates], False)
    fused = np.concatenate([q[owner], c], axis=1)
    score, f_cache = forward(model["fusion"], fused, True)
    score = score[:, 0]
    dscore = np.zeros_like(score)
    loss = 0.0
    cursor = 0
    group_count = len(batch_groups)
    for positive_count in positive_counts:
        length = positive_count + negative_count
        values = score[cursor:cursor + length].astype(np.float64)
        maximum = float(values.max())
        exp_all = np.exp(values - maximum)
        exp_positive = exp_all[:positive_count]
        loss += math.log(float(exp_all.sum())) + maximum - (math.log(float(exp_positive.sum())) + maximum)
        gradient = exp_all / exp_all.sum()
        gradient[:positive_count] -= exp_positive / exp_positive.sum()
        dscore[cursor:cursor + length] = (gradient / group_count).astype(np.float32)
        cursor += length

    dfused, f_grads = backward(model["fusion"], f_cache, dscore[:, None])
    q_width = q.shape[1]
    dq = np.zeros_like(q)
    np.add.at(dq, owner, dfused[:, :q_width])
    _, q_grads = backward(model["response"], q_cache, dq)
    _, c_grads = backward(model["candidate"], c_cache, dfused[:, q_width:])
    adamw(model, optimizer, {"response": q_grads, "candidate": c_grads, "fusion": f_grads}, learning_rate, weight_decay)
    return loss / group_count


def checkpoint_path(candidate_id: str, replay: bool) -> Path:
    suffix = "replay" if replay else "canonical"
    return CHECKPOINTS / f"{candidate_id}_{suffix}.npz"


def train_candidate(candidate_spec: dict, response: np.ndarray, candidate: np.ndarray,
                    train_groups: list, train_universe: np.ndarray, replay: bool = False) -> tuple[dict, list[dict]]:
    path = checkpoint_path(candidate_spec["candidate_id"], replay)
    history = []
    if path.exists():
        model, optimizer, completed = restore_model(path, candidate_spec)
        history_path = path.with_suffix(".json")
        metadata = load_json(history_path)
        history = list(metadata.get("history", []))
        require(len(history) == completed, "checkpoint history")
        print(f"  RESUME after epoch {completed}", flush=True)
    else:
        model, optimizer, completed = new_model(candidate_spec), None, 0
        optimizer = zero_optimizer(model)

    for epoch in range(completed, candidate_spec["epochs"]):
        epoch_rng = np.random.Generator(np.random.PCG64(candidate_spec["random_seed"] + epoch * 1000003))
        order = epoch_rng.permutation(len(train_groups))
        losses = []
        batch_size = candidate_spec["query_batch_size"]
        for start in range(0, len(order), batch_size):
            batch = [train_groups[int(index)] for index in order[start:start + batch_size]]
            losses.append(train_batch(
                model, optimizer, response, candidate, batch, train_universe, epoch_rng,
                candidate_spec["negative_candidates_per_query"], candidate_spec["learning_rate"],
                candidate_spec["weight_decay"],
            ))
        row = {
            "candidate_id": candidate_spec["candidate_id"], "replay": replay,
            "epoch": epoch + 1, "mean_ranking_loss": float(np.mean(losses)),
            "optimizer_steps": optimizer["step"],
        }
        history.append(row)
        arrays = model_arrays(model, optimizer)
        arrays["completed_epochs"] = np.asarray([epoch + 1], dtype="<u2")
        data = npz_bytes(arrays)
        temporary = path.with_name(path.name + ".tmp")
        temporary.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_bytes(data); os.replace(temporary, path)
        metadata_path = path.with_suffix(".json")
        temporary_json = metadata_path.with_name(metadata_path.name + ".tmp")
        temporary_json.write_bytes(json_bytes({"status": "IN_PROGRESS", "candidate": candidate_spec, "history": history, "checkpoint_sha256": sha256(path)}))
        os.replace(temporary_json, metadata_path)
        print(f"  epoch {epoch + 1}/{candidate_spec['epochs']} loss={row['mean_ranking_loss']:.8f}", flush=True)
    return model, history


def candidate_embedding(model: dict, candidate: np.ndarray) -> np.ndarray:
    result = np.empty((len(candidate), model["candidate"][-1]["b"].shape[0]), dtype=np.float32)
    for start in range(0, len(candidate), RANK_BATCH):
        stop_index = min(start + RANK_BATCH, len(candidate))
        result[start:stop_index], _ = forward(model["candidate"], candidate[start:stop_index], False)
    return result


def score_profile(model: dict, response_row: np.ndarray, c_embed: np.ndarray) -> np.ndarray:
    q, _ = forward(model["response"], response_row[None, :], False)
    scores = np.empty(len(c_embed), dtype=np.float32)
    for start in range(0, len(c_embed), RANK_BATCH):
        stop_index = min(start + RANK_BATCH, len(c_embed))
        fused = np.concatenate([np.repeat(q, stop_index - start, axis=0), c_embed[start:stop_index]], axis=1)
        value, _ = forward(model["fusion"], fused, True)
        scores[start:stop_index] = value[:, 0]
    return scores


def stable_top(scores: np.ndarray, count: int) -> np.ndarray:
    threshold = np.partition(scores, -count)[-count]
    greater = np.flatnonzero(scores > threshold)
    equal = np.flatnonzero(scores == threshold)[:count - len(greater)]
    candidates = np.concatenate([greater, equal])
    require(len(candidates) == count, "stable top-k size")
    order = np.lexsort((candidates, -scores[candidates]))
    return candidates[order]


def evaluate(candidate_spec: dict, model: dict, response: np.ndarray, candidate: np.ndarray,
             targets: dict[str, np.ndarray], partition_code: int = 1) -> tuple[dict, dict[str, np.ndarray]]:
    eval_groups = groups(targets, partition_code)
    c_embed = candidate_embedding(model, candidate)
    target_profiles = targets["profile_index"].astype(np.int64)
    target_faults = targets["fault_instance_index"].astype(np.int64)
    candidate_site_count = targets["candidate_site_count"].astype(np.int64)
    cross = targets["cross_partition_profile"].astype(np.uint8)
    stuck = targets["stuck_value"].astype(np.uint8)

    top_faults = np.empty((len(eval_groups), TOP_K), dtype=np.uint32)
    top_scores = np.empty((len(eval_groups), TOP_K), dtype=np.float32)
    profile_ids = np.empty(len(eval_groups), dtype=np.uint32)
    reciprocal, top1, top5, coverage, type_correct = [], [], [], [], []
    unique_top1, cross_mrr, local_mrr = [], [], []
    for group_number, (profile, partition_faults) in enumerate(eval_groups):
        scores = score_profile(model, response[profile], c_embed)
        top = stable_top(scores, TOP_K)
        top_faults[group_number] = top.astype(np.uint32)
        top_scores[group_number] = scores[top]
        profile_ids[group_number] = profile
        site_scores = np.maximum(scores[0::2], scores[1::2])
        top_site = int(np.argmax(site_scores))
        top5_sites = set(stable_top(site_scores, 5).tolist())
        top50_sites = set((top // 2).tolist())
        for fault in partition_faults:
            fault = int(fault); site = fault // 2
            rank = 1 + int(np.sum(site_scores > site_scores[site])) + int(np.sum((site_scores == site_scores[site]) & (np.arange(SITES) < site)))
            reciprocal.append(1.0 / rank)
            top1.append(top_site == site)
            top5.append(site in top5_sites)
            coverage.append(site in top50_sites)
            if top_site == site:
                predicted_stuck = int(scores[site * 2 + 1] > scores[site * 2])
                type_correct.append(int(predicted_stuck == int(stuck[fault])))
            if candidate_site_count[fault] == 1:
                unique_top1.append(top_site == site)
            (cross_mrr if cross[fault] else local_mrr).append(1.0 / rank)
        if (group_number + 1) % 100 == 0 or group_number + 1 == len(eval_groups):
            print(f"    ranked profiles {group_number + 1}/{len(eval_groups)}", flush=True)

    metrics = {
        "candidate_id": candidate_spec["candidate_id"],
        "calibration_observable_profiles": len(eval_groups),
        "calibration_observable_fault_instances": len(reciprocal),
        "observable_mean_reciprocal_rank": float(np.mean(reciprocal)),
        "unique_signature_top1_site_accuracy": float(np.mean(unique_top1)) if unique_top1 else 0.0,
        "observable_top1_site_accuracy": float(np.mean(top1)),
        "observable_top5_site_accuracy": float(np.mean(top5)),
        "observable_candidate_coverage_top50": float(np.mean(coverage)),
        "sa0_sa1_accuracy_given_top1_site": float(np.mean(type_correct)) if type_correct else 0.0,
        "cross_partition_profile_mrr": float(np.mean(cross_mrr)) if cross_mrr else None,
        "partition_local_profile_mrr": float(np.mean(local_mrr)) if local_mrr else None,
        "fault_free_false_alarm_rate": 0.0,
        "observable_detection_recall": 1.0,
        "all_injected_detection_recall": 22930 / 45678,
        "trainable_parameters": candidate_spec["trainable_parameters"],
    }
    rankings = {"profile_index": profile_ids, "top_fault_instance_index": top_faults, "top_score": top_scores}
    return metrics, rankings


def selection_key(item: dict) -> tuple:
    return (
        item["observable_mean_reciprocal_rank"],
        item["unique_signature_top1_site_accuracy"],
        item["observable_top5_site_accuracy"],
        -item["trainable_parameters"],
    )


def csv_payload(fields: list[str], rows: list[dict]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
    writer.writeheader(); writer.writerows(rows)
    return stream.getvalue().encode()


def main() -> None:
    require(ROOT.name == "vlsi_fault_detection_v2", f"run from project root, not {ROOT}")
    require(not AUDIT.exists(), "Stage 12A-1D is already frozen")
    for path in (METRICS_CSV, METRICS_JSON, HISTORY_CSV, SELECTED_MODEL, SELECTED_METADATA, CAL_RANKINGS, SELECTION_LOCK, MANIFEST):
        require(not path.exists(), f"partial final output exists: {rel(path)}")
    WORK.mkdir(parents=True, exist_ok=True)

    input_evidence = verify_inputs()
    architecture, contract, candidates, contract_audit = verify_contract_bundle()
    input_evidence[rel(CONTRACT_AUDIT)] = record(CONTRACT_AUDIT)
    for path in (ARCHITECTURE, CONTRACT, GRID, ENVIRONMENT):
        input_evidence[rel(path)] = record(path)
    response, targets = load_response_data()
    graph, graph_record = graph_cache()
    site, site_record = site_feature_catalog()
    vectors = vector_features()
    support, support_record = build_v1_support(site, vectors, graph)
    candidate = candidate_matrix(graph, support)
    response, candidate, scaler_record = fit_scalers(response, targets, candidate)
    train_groups = groups(targets, 0)
    train_universe = targets["fault_instance_index"][targets["partition_code"] == 0].astype(np.int64)

    print("\nTRAINING SCOPE")
    print(f"  DEV_TRAIN observable profiles : {len(train_groups)}")
    print(f"  Candidate universe             : {FAULTS}")
    print("  DEV_SITE_TEST                  : LOCKED / NOT OPENED")
    print("  VALIDATION / HOLDOUT           : NOT OPENED")
    results, histories, models, rankings = [], [], {}, {}
    for candidate_spec in candidates:
        print(f"\nCANDIDATE {candidate_spec['candidate_id']}", flush=True)
        model, history = train_candidate(candidate_spec, response, candidate, train_groups, train_universe)
        require(parameter_count(model) == candidate_spec["trainable_parameters"], "trained parameter count")
        print("  FULL-CATALOG DEV_CALIBRATION RANKING", flush=True)
        metrics, candidate_rankings = evaluate(candidate_spec, model, response, candidate, targets)
        print(f"  MRR={metrics['observable_mean_reciprocal_rank']:.8f} top1_unique={metrics['unique_signature_top1_site_accuracy']:.8f} top5={metrics['observable_top5_site_accuracy']:.8f}")
        results.append(metrics); histories.extend(history); models[candidate_spec["candidate_id"]] = model
        rankings[candidate_spec["candidate_id"]] = candidate_rankings

    selected_metrics = max(results, key=selection_key)
    selected_id = selected_metrics["candidate_id"]
    selected_spec = next(item for item in candidates if item["candidate_id"] == selected_id)
    selected_model = models[selected_id]
    print(f"\nSELECTED {selected_id}")
    print("\nDETERMINISTIC SELECTED-CANDIDATE REPLAY", flush=True)
    replay_model, replay_history = train_candidate(selected_spec, response, candidate, train_groups, train_universe, replay=True)
    require(parameter_count(replay_model) == selected_spec["trainable_parameters"], "replay parameter count")
    replay_metrics, replay_rankings = evaluate(selected_spec, replay_model, response, candidate, targets)
    weights_exact = all(np.array_equal(model_arrays(selected_model)[key], model_arrays(replay_model)[key]) for key in model_arrays(selected_model))
    scores_exact = all(np.array_equal(rankings[selected_id][key], replay_rankings[key]) for key in rankings[selected_id])
    metrics_exact = selected_metrics == replay_metrics
    require(weights_exact and scores_exact and metrics_exact, "deterministic selected-candidate replay")
    histories.extend(replay_history)

    metric_fields = [
        "candidate_id", "trainable_parameters", "calibration_observable_profiles",
        "calibration_observable_fault_instances", "observable_mean_reciprocal_rank",
        "unique_signature_top1_site_accuracy", "observable_top1_site_accuracy",
        "observable_top5_site_accuracy", "observable_candidate_coverage_top50",
        "sa0_sa1_accuracy_given_top1_site", "cross_partition_profile_mrr",
        "partition_local_profile_mrr", "fault_free_false_alarm_rate",
        "observable_detection_recall", "all_injected_detection_recall",
    ]
    history_fields = ["candidate_id", "replay", "epoch", "mean_ranking_loss", "optimizer_steps"]
    atomic_bytes(METRICS_CSV, csv_payload(metric_fields, results))
    atomic_bytes(METRICS_JSON, json_bytes({"stage": STAGE, "status": "PASS", "selection_metric": "OBSERVABLE MRR", "candidates": results}))
    atomic_bytes(HISTORY_CSV, csv_payload(history_fields, histories))
    atomic_npz(SELECTED_MODEL, model_arrays(selected_model))
    atomic_bytes(SELECTED_METADATA, json_bytes({
        "format_version": "CIRCUITSAGE-HMAC-V2-RANKER-NPZ-v1", "stage": STAGE,
        "status": "FROZEN", "selected_candidate": selected_spec,
        "response_scaler": scaler_record, "candidate_scaler": scaler_record,
        "v1_support": support_record, "graph_features": graph_record,
        "model": record(SELECTED_MODEL),
    }))
    atomic_npz(CAL_RANKINGS, rankings[selected_id])

    acceptance = contract["acceptance"]
    acceptance_checks = {
        "fault_free_false_alarm_rate": selected_metrics["fault_free_false_alarm_rate"] <= acceptance["fault_free_false_alarm_rate_max"],
        "observable_fault_detection_recall": selected_metrics["observable_detection_recall"] >= acceptance["observable_fault_detection_recall_min"],
        "observable_candidate_coverage": selected_metrics["observable_candidate_coverage_top50"] >= acceptance["observable_candidate_coverage_min"],
        "unique_signature_top1_site_accuracy": selected_metrics["unique_signature_top1_site_accuracy"] >= acceptance["unique_signature_top1_site_accuracy_min"],
        "observable_top5_site_accuracy": selected_metrics["observable_top5_site_accuracy"] >= acceptance["observable_top5_site_accuracy_min"],
    }
    advancement = "PASS" if all(acceptance_checks.values()) else "NOT_MET"
    lock = {
        "lock_version": "CIRCUITSAGE-HMAC-V2-SELECTION-LOCK-v1", "stage": STAGE,
        "status": "PASS", "training_status": "FROZEN", "selection_status": "FROZEN",
        "selected_candidate_id": selected_id, "selected_candidate": selected_spec,
        "selected_metrics": selected_metrics, "acceptance_checks": acceptance_checks,
        "advancement_target_on_calibration": advancement,
        "detector": "EXACT GOLDEN-REFERENCE ANOMALY GATE",
        "model": record(SELECTED_MODEL), "model_metadata": record(SELECTED_METADATA),
        "calibration_rankings": record(CAL_RANKINGS),
        "deterministic_replay": {"status": "PASS", "weights_exact": True, "scores_exact": True, "rankings_exact": True, "metrics_exact": True},
        "dev_site_test_opened": False, "validation_access_count": 0, "holdout_access_count": 0,
        "v1_model_modified": False,
    }
    atomic_bytes(SELECTION_LOCK, json_bytes(lock))

    output_paths = [SUPPORT_CACHE, SCALERS, METRICS_CSV, METRICS_JSON, HISTORY_CSV, SELECTED_MODEL, SELECTED_METADATA, CAL_RANKINGS, SELECTION_LOCK]
    outputs = {rel(path): record(path) for path in output_paths}
    manifest = {
        "manifest_version": "CIRCUITSAGE-HMAC-V2-TRAINING-MANIFEST-v1", "stage": STAGE,
        "status": "PASS", "training_status": "FROZEN", "model_selection_status": "FROZEN",
        "detector_training": "NONE", "ranker_candidates_trained": len(candidates),
        "selected_candidate_id": selected_id, "selected_metrics": selected_metrics,
        "input_evidence": input_evidence, "derived_static_inputs": {"graph": graph_record, "site_features": site_record},
        "outputs": outputs, "dev_site_test_opened": False,
        "validation_access_count": 0, "holdout_access_count": 0,
    }
    atomic_bytes(MANIFEST, json_bytes(manifest))
    audit = {
        "audit_version": "CIRCUITSAGE-HMAC-V2-TRAINING-CALIBRATION-FREEZE-v1", "stage": STAGE,
        "status": "PASS", "training_status": "FROZEN", "model_selection_status": "FROZEN",
        "detector_status": "FROZEN EXACT RULE", "ranker_status": "FROZEN",
        "selected_candidate_id": selected_id, "selected_metrics": selected_metrics,
        "calibration_advancement_target": advancement, "acceptance_checks": acceptance_checks,
        "full_catalog_candidate_count": FAULTS, "v1_support_inference_performed": True,
        "v1_model_modified": False, "deterministic_training_replay": "PASS",
        "weights_exact": True, "scores_exact": True, "rankings_exact": True,
        "dev_site_test_state": "LOCKED", "dev_site_test_opened": False,
        "validation_access_count": 0, "holdout_access_count": 0,
        "frozen_inputs_modified": False, "selection_lock": record(SELECTION_LOCK),
        "manifest": record(MANIFEST),
        "next_gate": "STAGE 12A-1E — LOCKED V2 DEV_SITE_TEST DETECTION/LOCALIZATION EVALUATION",
    }
    atomic_bytes(AUDIT, json_bytes(audit))
    shutil.rmtree(CHECKPOINTS, ignore_errors=True)

    print("\nSTAGE 12A-1D — V2 LINKED DETECTOR/LOCATOR TRAINING AND CALIBRATION FREEZE")
    print(f"{'Status':<38}: PASS")
    print(f"{'Training / selection status':<38}: FROZEN / FROZEN")
    print(f"{'Detector':<38}: EXACT GOLDEN-REFERENCE GATE")
    print(f"{'Ranker candidates trained':<38}: {len(candidates)}")
    print(f"{'Selected candidate':<38}: {selected_id}")
    print(f"{'Calibration observable MRR':<38}: {selected_metrics['observable_mean_reciprocal_rank']:.8f}")
    print(f"{'Unique-signature top1 site':<38}: {selected_metrics['unique_signature_top1_site_accuracy']:.8f}")
    print(f"{'Observable top5 site':<38}: {selected_metrics['observable_top5_site_accuracy']:.8f}")
    print(f"{'Candidate coverage top50':<38}: {selected_metrics['observable_candidate_coverage_top50']:.8f}")
    print(f"{'SA0/SA1 | correct top1 site':<38}: {selected_metrics['sa0_sa1_accuracy_given_top1_site']:.8f}")
    print(f"{'Fault-free false alarms':<38}: 0")
    print(f"{'Observable detection recall':<38}: 1.00000000")
    print(f"{'Calibration advancement target':<38}: {advancement}")
    print(f"{'Deterministic replay':<38}: PASS")
    print(f"{'DEV_SITE_TEST opened':<38}: NO")
    print(f"{'VALIDATION / HOLDOUT access':<38}: 0 / 0")
    print(f"{'V1 model modified':<38}: NO")
    print(f"{'Selected model':<38}: {SELECTED_MODEL}")
    print(f"{'Selected model SHA':<38}: {sha256(SELECTED_MODEL)}")
    print(f"{'Selection lock':<38}: {SELECTION_LOCK}")
    print(f"{'Selection lock SHA':<38}: {sha256(SELECTION_LOCK)}")
    print(f"{'Manifest':<38}: {MANIFEST}")
    print(f"{'Manifest SHA':<38}: {sha256(MANIFEST)}")
    print(f"{'Audit':<38}: {AUDIT}")
    print(f"{'Audit SHA':<38}: {sha256(AUDIT)}")
    print(f"{'Next gate':<38}: STAGE 12A-1E — LOCKED V2 DEV_SITE_TEST DETECTION/LOCALIZATION EVALUATION")


if __name__ == "__main__":
    main()
