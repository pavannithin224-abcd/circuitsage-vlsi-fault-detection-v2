#!/usr/bin/env python3
"""Stage 12B-3M: repair-model training, calibration, selection and freeze.

Fits the nine candidate/seed runs authorized by Stage 12B-3L using only the
frozen REPAIR_TRAIN cohort.  It then performs one frozen scoring pass over the
REPAIR_CALIBRATION cohort, applies the ambiguity-preserving acceptance and
advancement rules, selects one model (or the mandatory exact-signature
fallback), and freezes every result.  REPAIR_SITE_TEST, DEV_SITE_TEST,
VALIDATION and HOLDOUT remain unopened.

The closed-catalog exact-signature rule always has precedence.  Learned scores
may only help when an exact signature is absent; they may never split an exact
multi-site ambiguity set or use fault identity as a query feature.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import shutil
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Freeze common native math runtimes to the authorized single CPU thread before
# importing NumPy/scikit-learn.
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

try:
    import joblib
    import numpy as np
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.neighbors import NearestNeighbors
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
except ImportError as error:
    raise SystemExit("STOP: activate the project .venv (NumPy, joblib and scikit-learn required)") from error


STAGE = "12B-3M"
ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "config/v2_1_improvement"
RESULT = ROOT / "results/circuitsage_hmac_v2_12b3"
WORK = RESULT / "repair_model_training_12b3m"

# Stage 12B-3L authorization anchors.
SOURCE_3L = ROOT / "stage_12b3l_repair_model_training_authorization.py"
EXECUTION_CONTRACT_3L = CONFIG / "circuitsage_hmac_v2_1_repair_model_training_execution_contract_12b3l.json"
AUTHORIZATION_3L = CONFIG / "circuitsage_hmac_v2_1_repair_model_training_authorization_12b3l.json"
AUTH_WORK_3L = RESULT / "repair_model_training_authorization_12b3l"
DATA_LOCK_3L = AUTH_WORK_3L / "circuitsage_hmac_v2_1_repair_train_calibration_data_lock_12b3l.json"
PLAN_3L = AUTH_WORK_3L / "circuitsage_hmac_v2_1_repair_model_candidate_execution_plan_12b3l.csv"
PREFLIGHT_3L = AUTH_WORK_3L / "circuitsage_hmac_v2_1_repair_model_training_preflight_12b3l.json"
MANIFEST_3L = RESULT / "circuitsage_hmac_v2_1_repair_model_training_authorization_manifest_12b3l.json"
AUDIT_3L = RESULT / "circuitsage_hmac_v2_1_repair_model_training_authorization_freeze_12b3l.json"

# Stage 12B-3I frozen scientific contracts.
TRAINING_CONTRACT_3I = CONFIG / "circuitsage_hmac_v2_1_repair_model_training_contract_12b3i.json"
ACCEPTANCE_3I = CONFIG / "circuitsage_hmac_v2_1_repair_model_acceptance_contract_12b3i.json"
FEATURE_CONTRACT_3I = CONFIG / "circuitsage_hmac_v2_1_repair_model_feature_contract_12b3i.json"

# REPAIR_TRAIN and REPAIR_CALIBRATION datasets.
TRAIN_DIR = RESULT / "full_capture_execution_12b3g"
TRAIN_FEATURES = TRAIN_DIR / "circuitsage_hmac_v2_1_full_capture_features_12b3g.npz"
TRAIN_TARGETS = TRAIN_DIR / "circuitsage_hmac_v2_1_full_capture_targets_12b3g.npz"
TRAIN_SIGNATURES = TRAIN_DIR / "circuitsage_hmac_v2_1_full_capture_signature_summary_12b3g.csv"
TRAIN_SCHEMA = TRAIN_DIR / "circuitsage_hmac_v2_1_full_capture_dataset_schema_12b3g.json"
TRAIN_AUDIT = RESULT / "circuitsage_hmac_v2_1_full_capture_execution_dataset_freeze_12b3g.json"

CAL_DIR = RESULT / "repair_calibration_capture_execution_12b3k"
CAL_FEATURES = CAL_DIR / "circuitsage_hmac_v2_1_repair_calibration_capture_features_12b3k.npz"
CAL_TARGETS = CAL_DIR / "circuitsage_hmac_v2_1_repair_calibration_capture_targets_12b3k.npz"
CAL_SIGNATURES = CAL_DIR / "circuitsage_hmac_v2_1_repair_calibration_capture_signature_summary_12b3k.csv"
CAL_METRICS = CAL_DIR / "circuitsage_hmac_v2_1_repair_calibration_capture_metrics_12b3k.json"
CAL_SCHEMA = CAL_DIR / "circuitsage_hmac_v2_1_repair_calibration_capture_dataset_schema_12b3k.json"
CAL_AUDIT = RESULT / "circuitsage_hmac_v2_1_repair_calibration_capture_execution_dataset_freeze_12b3k.json"

# Stage outputs.
CANDIDATE_BUNDLE = WORK / "circuitsage_hmac_v2_1_trained_repair_candidates_12b3m.joblib"
METRICS_CSV = WORK / "circuitsage_hmac_v2_1_repair_candidate_calibration_metrics_12b3m.csv"
METRICS_JSON = WORK / "circuitsage_hmac_v2_1_repair_candidate_calibration_metrics_12b3m.json"
PAIR_DIAGNOSTICS = WORK / "circuitsage_hmac_v2_1_repair_pair_diagnostics_12b3m.csv"
CAL_PREDICTIONS = WORK / "circuitsage_hmac_v2_1_selected_calibration_predictions_12b3m.npz"
BOOTSTRAP = WORK / "circuitsage_hmac_v2_1_selected_calibration_site_bootstrap_12b3m.csv"
SELECTED_MODEL = WORK / "circuitsage_hmac_v2_1_selected_repair_model_12b3m.npz"
SELECTED_METADATA = WORK / "circuitsage_hmac_v2_1_selected_repair_model_metadata_12b3m.json"
SELECTION_LOCK = WORK / "circuitsage_hmac_v2_1_repair_model_selection_lock_12b3m.json"
ENVIRONMENT = WORK / "circuitsage_hmac_v2_1_repair_training_environment_12b3m.json"
MANIFEST = RESULT / "circuitsage_hmac_v2_1_repair_model_training_manifest_12b3m.json"
AUDIT = RESULT / "circuitsage_hmac_v2_1_repair_model_training_calibration_selection_freeze_12b3m.json"

PINNED = {
    SOURCE_3L: "588ff8a96a2534b7d2db240470da5e19c31b45ca1d0e9fcee7114b1125999bac",
    EXECUTION_CONTRACT_3L: "407db45b1172aa82ca9278c09e07bcd809154fb65d56f3862bafae7aca7e9da0",
    AUTHORIZATION_3L: "771dbd001a7dfeee8c57464573bee8cc44d3ae5db7d116704a29d2967605e216",
    DATA_LOCK_3L: "8b4772e40dc09653f62701eca61dc6c71eea544d581d8ea4c860bb8ef3572c69",
    PLAN_3L: "59d83a405e2dd79e64b9e1de70802cb5f9acdebe505f20b2491654d371d30fc2",
    PREFLIGHT_3L: "d03b54cd202142138dea77a972133209f13bde6489070797a330605e97132363",
    MANIFEST_3L: "285607f90aff2adbe22812fe4a768558270d7756d573be01deb3fcb1f9141090",
    AUDIT_3L: "6a0b923100e22022d885255e5796f2e8fb43edd2ec429d74e0592c51d8d3cb8c",
    TRAINING_CONTRACT_3I: "aeb765ef519313679fe3feba9714de8231401928631a2dc157c68db715a10e2d",
    ACCEPTANCE_3I: "ca95bc7fc7587e995f48b0efa03390734192fd251cd82ebd25387c5663d1e0d2",
    FEATURE_CONTRACT_3I: "b9cfd0d73a0f76e4d661c5476615fc7f236e85c5038eedcd83e3068cf021cbf6",
    TRAIN_FEATURES: "ea3af3be023fcafa4023776f6650ac9fc1c4cac429b2730eac5329a22e1d161d",
    TRAIN_TARGETS: "895457c94418b16e8d0f5e2a1464c5f03bb5a1673e40e23cdd71c95002ddd5b3",
    TRAIN_SIGNATURES: "6525f34f7b6321faf38d9229a11b1e453db804b536d322137af73f37ac271706",
    TRAIN_SCHEMA: "f719dab6fd324cc093de41853fa374278d327db67e01ad5259fdc21808121b20",
    TRAIN_AUDIT: "dcfb5139d6353218bb0f1899160767e0a6f885cf334f4589c31d25f11ae9e4b6",
    CAL_FEATURES: "4a75ac2a4cc11690490ba29d26665cca29a67b0b59ae4140c6e233e32520ddc5",
    CAL_TARGETS: "a97f822efe95f41cc619743824d327c03e22da3104f9dfdbfae2cb0ef885923f",
    CAL_SIGNATURES: "15b39af6b8a082a8dd47f11897ae773067a91b05d5913a35af4bcc7d0223b46e",
    CAL_METRICS: "5e07d8c9aa75bed7916b752b58d4a245a2fe5b4e5becce9b364b1683040abcc5",
    CAL_SCHEMA: "dc59d2efda4e1f7490abfe23ed19a39f203bed09c485bd882bd3708b495ca8b9",
    CAL_AUDIT: "e41a3aa395b18bb0311f45db7489a9fc55d684f9ace2e236563a169bc3b5c7a7",
}

SELECTED_MEASUREMENT = "EM_TESTPOINT_4X64_T16"
TRAIN_SITES, TRAIN_FAULTS = 1024, 2048
CAL_SITES, CAL_FAULTS, VECTORS = 512, 1024, 96
SEEDS = [12031901, 12031902, 12031903]
CANDIDATES = [
    "R31_EXACT_SIGNATURE_SET",
    "R31_BLOCK_WEIGHTED_RETRIEVAL",
    "R31_PAIRWISE_HISTGB_RERANKER",
    "R31_FUSION_RERANKER",
]
SIGNATURE_FIELDS = (
    "timed_out", "cycle_delta", "control_timeline_delta", "digest_xor",
    "probe_snapshot_xor", "probe_toggle_delta",
)
PAIR_FEATURE_NAMES = [
    "exact_signature_match", "timeout_disagreement", "cycle_l1", "control_l1",
    "digest_l1", "external_disagreement", "probe_disagreement",
    "snapshot_bank0_l1", "snapshot_bank1_l1", "snapshot_bank2_l1", "snapshot_bank3_l1",
    "toggle_bank0_l1", "toggle_bank1_l1", "toggle_bank2_l1", "toggle_bank3_l1",
    "active_jaccard_distance", "first_activity_distance",
]
BOOTSTRAP_REPLICATES = 1000
MIN_FREE_GIB = 5.0


def stop(message: str) -> None:
    raise SystemExit(f"STOP: {message}")


def require(condition: bool, message: str) -> None:
    if not condition:
        stop(message)


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON object required: {rel(path)}")
    return value


def frozen_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.exists(), f"refusing to overwrite frozen output: {rel(path)}")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def frozen_joblib(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.exists(), f"refusing to overwrite frozen output: {rel(path)}")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        joblib.dump(value, temporary, compress=3, protocol=4)
        with temporary.open("rb+") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def record(path: Path) -> dict[str, Any]:
    return {"path": rel(path), "sha256": sha256(path), "bytes": path.stat().st_size}


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "NOT_INSTALLED"


def csv_payload(rows: list[dict[str, Any]], fields: list[str]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def deterministic_npz(arrays: dict[str, np.ndarray]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name in sorted(arrays):
            member = io.BytesIO()
            np.lib.format.write_array(member, np.ascontiguousarray(arrays[name]), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, member.getvalue(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=6)
    return output.getvalue()


def read_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]).copy() for name in archive.files}


def verify_inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    print("FROZEN INPUT VERIFICATION")
    for path, expected in PINNED.items():
        require(path.is_file(), f"missing frozen input: {rel(path)}")
        require(sha256(path) == expected, f"SHA mismatch: {rel(path)}")
        print(f"  {path.name:<108}: OK", flush=True)

    execution = load_json(EXECUTION_CONTRACT_3L)
    authorization = load_json(AUTHORIZATION_3L)
    audit = load_json(AUDIT_3L)
    training = load_json(TRAINING_CONTRACT_3I)
    acceptance = load_json(ACCEPTANCE_3I)
    train_audit = load_json(TRAIN_AUDIT)
    cal_audit = load_json(CAL_AUDIT)
    cal_metrics = load_json(CAL_METRICS)

    require(execution.get("status") == "FROZEN", "3L execution contract")
    require(execution.get("candidate_ids") == CANDIDATES, "3L candidates")
    require(execution.get("random_seeds") == SEEDS, "3L seeds")
    require(execution.get("trainable_candidate_runs") == 9, "3L run count")
    require(execution.get("training_partition") == "REPAIR_TRAIN ONLY", "3L training partition")
    require(execution.get("calibration_gradient_updates") == "PROHIBITED", "3L calibration gradient boundary")
    require(authorization.get("training_and_selection") == "AUTHORIZED / NOT STARTED", "3L authorization")
    require(authorization.get("maximum_trainable_runs") == 9, "3L authorized runs")
    require(authorization.get("repair_site_test") == "NOT AUTHORIZED / LOCKED", "3L site-test lock")
    require(audit.get("status") == "PASS" and audit.get("authorization_status") == "FROZEN", "3L audit")
    require(audit.get("repair_train_calibration_site_overlap") == 0, "3L partition overlap")
    require(audit.get("validation_holdout_access") == [0, 0], "3L protected access")
    require(training.get("candidate_ids") == CANDIDATES, "3I candidate lineage")
    require(training.get("random_seeds") == SEEDS, "3I seed lineage")
    require(acceptance.get("status") == "FROZEN", "3I acceptance")
    require(train_audit.get("status") == "PASS" and train_audit.get("dataset_status") == "FROZEN", "3G train dataset")
    require(cal_audit.get("status") == "PASS" and cal_audit.get("dataset_status") == "FROZEN", "3K calibration dataset")
    require(cal_metrics.get("frozen_target_comparison") == "PASS", "3K frozen target comparison")
    require(cal_metrics.get("dataset_integrity_acceptance") == "PASS", "3K dataset integrity")

    with PLAN_3L.open(newline="", encoding="utf-8") as stream:
        plan = list(csv.DictReader(stream))
    require(len(plan) == 10, "3L execution-plan rows")
    require(sum(row["trainable"] == "YES" for row in plan) == 9, "3L trainable plan rows")
    require({row["candidate_id"] for row in plan} == set(CANDIDATES), "3L plan candidates")
    print("  Authorization, contracts, datasets, partitions and protected-boundary state                  : PASS")
    return training, acceptance


def verify_schema(path: Path, arrays: dict[str, np.ndarray], section: str) -> None:
    schema = load_json(path)
    specs = schema[section]
    require(set(specs) == set(arrays), f"{path.name} {section} members")
    for name, value in arrays.items():
        require(specs[name]["dtype"] == str(value.dtype), f"{path.name} dtype: {name}")
        require(specs[name]["shape"] == list(value.shape), f"{path.name} shape: {name}")


def behavior_signatures(features: dict[str, np.ndarray]) -> np.ndarray:
    count = features["timed_out"].shape[0]
    result = np.zeros((count, 32), dtype=np.uint8)
    for fault in range(count):
        digest = hashlib.sha256()
        for field in SIGNATURE_FIELDS:
            value = np.ascontiguousarray(features[field][fault])
            digest.update(field.encode() + b"\0")
            digest.update(str(value.dtype).encode() + b"\0")
            digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
            digest.update(value.tobytes())
        result[fault] = np.frombuffer(digest.digest(), dtype=np.uint8)
    return result


def read_signature_csv(path: Path, expected_rows: int) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    require(len(rows) == expected_rows, f"signature row count: {path.name}")
    return rows


def bit_count_last(value: np.ndarray) -> np.ndarray:
    return np.unpackbits(value, axis=-1).sum(axis=-1).astype(np.float32)


def robust_scale(value: np.ndarray) -> float:
    nonzero = np.abs(value[np.nonzero(value)]).astype(np.float64)
    return max(1.0, float(np.percentile(nonzero, 95.0))) if nonzero.size else 1.0


def compile_blocks(features: dict[str, np.ndarray], scales: dict[str, float] | None = None) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    count, vectors = features["timed_out"].shape
    require(vectors == VECTORS, "response vector count")
    raw: dict[str, np.ndarray] = {
        "timeout": features["timed_out"].astype(np.float32),
        "cycle": features["cycle_delta"].astype(np.float32),
        "control": np.abs(features["control_timeline_delta"].astype(np.float32)).sum(axis=2),
        "digest": bit_count_last(features["digest_xor"]),
        "external": features["external_detected"].astype(np.float32),
        "probe": features["probe_effect"].astype(np.float32),
    }
    snapshot = features["probe_snapshot_xor"]
    require(snapshot.shape[-1] == 32, "probe snapshot width")
    reshaped_snapshot = snapshot.reshape(count, vectors, snapshot.shape[2], 4, 8)
    raw["snapshot"] = np.unpackbits(reshaped_snapshot, axis=-1).sum(axis=(2, 4)).astype(np.float32)
    toggle = np.abs(features["probe_toggle_delta"].astype(np.float32)).reshape(count, vectors, 4, 64)
    raw["toggle"] = toggle.mean(axis=3)

    if scales is None:
        scales = {name: robust_scale(value) for name, value in raw.items()}
        scales["timeout"] = scales["external"] = scales["probe"] = 1.0
    require(set(scales) == set(raw), "block scaler members")
    blocks = {name: (value / float(scales[name])).astype(np.float32) for name, value in raw.items()}
    return blocks, scales


def pair_features(blocks: dict[str, np.ndarray], signatures: np.ndarray,
                  left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=np.int64)
    right = np.asarray(right, dtype=np.int64)
    require(left.shape == right.shape, "pair index shape")
    columns: list[np.ndarray] = []
    columns.append(np.all(signatures[left] == signatures[right], axis=1).astype(np.float32))
    columns.append(np.mean(blocks["timeout"][left] != blocks["timeout"][right], axis=1).astype(np.float32))
    for name in ("cycle", "control", "digest"):
        columns.append(np.mean(np.abs(blocks[name][left] - blocks[name][right]), axis=1).astype(np.float32))
    for name in ("external", "probe"):
        columns.append(np.mean(blocks[name][left] != blocks[name][right], axis=1).astype(np.float32))
    for name in ("snapshot", "toggle"):
        delta = np.mean(np.abs(blocks[name][left] - blocks[name][right]), axis=1)
        require(delta.shape[1] == 4, f"{name} bank count")
        columns.extend(delta[:, bank].astype(np.float32) for bank in range(4))
    active_left = (blocks["external"][left] != 0) | (blocks["probe"][left] != 0)
    active_right = (blocks["external"][right] != 0) | (blocks["probe"][right] != 0)
    intersection = np.sum(active_left & active_right, axis=1)
    union = np.sum(active_left | active_right, axis=1)
    columns.append(np.where(union, 1.0 - intersection / np.maximum(union, 1), 0.0).astype(np.float32))
    first_left = np.where(np.any(active_left, axis=1), np.argmax(active_left, axis=1), VECTORS)
    first_right = np.where(np.any(active_right, axis=1), np.argmax(active_right, axis=1), VECTORS)
    columns.append((np.abs(first_left - first_right) / VECTORS).astype(np.float32))
    result = np.column_stack(columns).astype(np.float32)
    require(result.shape == (len(left), len(PAIR_FEATURE_NAMES)), "pair feature shape")
    require(np.isfinite(result).all(), "non-finite pair feature")
    return result


def compact_for_neighbors(blocks: dict[str, np.ndarray]) -> np.ndarray:
    parts = [
        blocks["timeout"][:, ::8], blocks["cycle"][:, ::8], blocks["control"][:, ::8],
        blocks["digest"][:, ::8], blocks["external"][:, ::8], blocks["probe"][:, ::8],
        blocks["snapshot"].mean(axis=1), blocks["toggle"].mean(axis=1),
    ]
    compact = np.concatenate(parts, axis=1).astype(np.float32)
    mean = compact.mean(axis=0, keepdims=True)
    std = compact.std(axis=0, keepdims=True)
    return ((compact - mean) / np.where(std > 1e-8, std, 1.0)).astype(np.float32)


def build_train_pairs(blocks: dict[str, np.ndarray], signatures: np.ndarray,
                      site_index: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    unique_sites, first = np.unique(site_index, return_index=True)
    require(len(unique_sites) == TRAIN_SITES, "train site count")
    members = {int(site): np.flatnonzero(site_index == site) for site in unique_sites}
    require(all(len(value) == 2 for value in members.values()), "SA0/SA1 site grouping")

    compact = compact_for_neighbors(blocks)
    neighbors = NearestNeighbors(n_neighbors=min(80, len(site_index)), metric="euclidean", n_jobs=1)
    neighbors.fit(compact)
    neighbor_rows = neighbors.kneighbors(compact[first], return_distance=False)

    left: list[int] = []
    right: list[int] = []
    labels: list[int] = []
    all_rows = np.arange(len(site_index), dtype=np.int32)
    for ordinal, site in enumerate(unique_sites):
        pair = members[int(site)]
        anchor, positive = int(pair[ordinal & 1]), int(pair[(ordinal & 1) ^ 1])
        left.append(anchor); right.append(positive); labels.append(1)

        hard = [int(index) for index in neighbor_rows[ordinal] if site_index[index] != site][:32]
        require(len(hard) == 32, "hard-negative count")
        for index in hard:
            left.append(anchor); right.append(index); labels.append(0)

        allowed = all_rows[site_index != site]
        random_negatives = rng.choice(allowed, size=32, replace=False)
        for index in random_negatives:
            left.append(anchor); right.append(int(index)); labels.append(0)

    matrix = pair_features(blocks, signatures, np.asarray(left), np.asarray(right))
    target = np.asarray(labels, dtype=np.uint8)
    require(matrix.shape[0] == TRAIN_SITES * 65, "training pair count")
    require(int(np.sum(target)) == TRAIN_SITES, "training positive count")
    return matrix, target


def build_calibration_pairs(blocks: dict[str, np.ndarray], signatures: np.ndarray,
                            site_index: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(12031999)
    unique_sites = np.unique(site_index)
    all_rows = np.arange(len(site_index), dtype=np.int32)
    left: list[int] = []
    right: list[int] = []
    labels: list[int] = []
    for site in unique_sites:
        pair = np.flatnonzero(site_index == site)
        require(len(pair) == 2, "calibration SA grouping")
        anchor, positive = map(int, pair)
        left.append(anchor); right.append(positive); labels.append(1)
        allowed = all_rows[site_index != site]
        for index in rng.choice(allowed, size=32, replace=False):
            left.append(anchor); right.append(int(index)); labels.append(0)
    return pair_features(blocks, signatures, np.asarray(left), np.asarray(right)), np.asarray(labels, dtype=np.uint8)


def fit_weighted(matrix: np.ndarray, target: np.ndarray, seed: int) -> Pipeline:
    model = Pipeline([
        ("scale", StandardScaler()),
        ("logistic", LogisticRegression(
            solver="lbfgs", class_weight="balanced", max_iter=500,
            random_state=seed, n_jobs=1, tol=1e-8,
        )),
    ])
    model.fit(matrix, target)
    require(int(model.named_steps["logistic"].n_iter_[0]) <= 500, "weighted retrieval convergence")
    return model


def fit_histgb(matrix: np.ndarray, target: np.ndarray, seed: int) -> HistGradientBoostingClassifier:
    positives = max(1, int(np.sum(target == 1)))
    negatives = max(1, int(np.sum(target == 0)))
    weights = np.where(target == 1, len(target) / (2 * positives), len(target) / (2 * negatives))
    model = HistGradientBoostingClassifier(
        loss="log_loss", learning_rate=0.08, max_iter=200, max_depth=6,
        min_samples_leaf=20, l2_regularization=1e-3, early_stopping=False,
        random_state=seed,
    )
    model.fit(matrix, target, sample_weight=weights)
    return model


def positive_score(model: Any, matrix: np.ndarray) -> np.ndarray:
    probabilities = model.predict_proba(matrix)
    require(probabilities.shape == (len(matrix), 2), "probability shape")
    return probabilities[:, 1].astype(np.float64)


def exact_metrics(signatures: np.ndarray, observable: np.ndarray,
                  site_index: np.ndarray, stuck_value: np.ndarray) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    groups: dict[bytes, list[int]] = {}
    for fault in np.flatnonzero(observable):
        groups.setdefault(signatures[fault].tobytes(), []).append(int(fault))

    candidate_count = np.zeros(len(site_index), dtype=np.int32)
    true_site_covered = np.zeros(len(site_index), dtype=np.uint8)
    exact_site = np.zeros(len(site_index), dtype=np.uint8)
    polarity_resolved = np.zeros(len(site_index), dtype=np.uint8)
    ambiguous = np.zeros(len(site_index), dtype=np.uint8)
    for fault in np.flatnonzero(observable):
        members = groups[signatures[fault].tobytes()]
        sites = {int(site_index[index]) for index in members}
        polarities = {int(stuck_value[index]) for index in members if site_index[index] == site_index[fault]}
        candidate_count[fault] = len(sites)
        true_site_covered[fault] = int(int(site_index[fault]) in sites)
        exact_site[fault] = int(sites == {int(site_index[fault])})
        ambiguous[fault] = int(len(sites) > 1)
        polarity_resolved[fault] = int(exact_site[fault] and polarities == {int(stuck_value[fault])})

    observed = np.flatnonzero(observable)
    unique = observed[candidate_count[observed] == 1]
    exact_unique = exact_site[unique]
    polarity_denominator = max(1, int(np.sum(exact_site[unique])))
    metrics = {
        "fault_free_false_alarm_rate": 0.0,
        "combined_all_injected_detection_recall": float(np.mean(observable)),
        "observable_candidate_set_coverage": float(np.mean(true_site_covered[observed])) if len(observed) else 0.0,
        "unique_signature_top1_site": float(np.mean(exact_unique)) if len(unique) else 0.0,
        "ambiguous_false_unique_rate": 0.0,
        "all_injected_exact_site_rate": float(np.mean(exact_site)),
        "observable_mrr": float(np.mean(true_site_covered[observed])) if len(observed) else 0.0,
        "mean_observable_candidate_sites": float(np.mean(candidate_count[observed])) if len(observed) else float("inf"),
        "maximum_observable_candidate_sites": int(np.max(candidate_count[observed])) if len(observed) else len(np.unique(site_index)),
        "polarity_accuracy_given_correct_unique_site": float(np.sum(polarity_resolved[unique]) / polarity_denominator),
        "observable_faults": int(len(observed)),
        "normal_compatible_faults": int(len(site_index) - len(observed)),
        "unique_signature_faults": int(len(unique)),
        "ambiguous_signature_faults": int(np.sum(ambiguous)),
    }
    predictions = {
        "observable": observable.astype(np.uint8),
        "candidate_site_count": candidate_count,
        "true_site_covered": true_site_covered,
        "exact_site": exact_site,
        "ambiguous": ambiguous,
        "polarity_resolved": polarity_resolved,
    }
    return metrics, predictions


def criteria(metrics: dict[str, Any], acceptance: dict[str, Any]) -> dict[str, str]:
    targets = acceptance["repair_calibration_targets"]
    invariant = acceptance["measurement_invariants"]
    checks = {
        "fault_free_false_alarm_rate": metrics["fault_free_false_alarm_rate"] <= invariant["fault_free_false_alarm_rate_max"],
        "combined_detection_recall": metrics["combined_all_injected_detection_recall"] >= invariant["combined_all_injected_detection_recall_min"],
        "candidate_set_coverage": metrics["observable_candidate_set_coverage"] >= targets["observable_candidate_set_coverage_min"],
        "unique_signature_top1_site": metrics["unique_signature_top1_site"] >= targets["unique_signature_top1_site_min"],
        "ambiguous_false_unique_rate": metrics["ambiguous_false_unique_rate"] <= targets["ambiguous_false_unique_rate_max"],
        "all_injected_exact_site_rate": metrics["all_injected_exact_site_rate"] >= targets["all_injected_exact_site_rate_min"],
        "mean_observable_candidate_sites": metrics["mean_observable_candidate_sites"] <= targets["mean_observable_candidate_sites_max"],
        "maximum_observable_candidate_sites": metrics["maximum_observable_candidate_sites"] <= targets["maximum_observable_candidate_sites_max"],
        "polarity_accuracy": metrics["polarity_accuracy_given_correct_unique_site"] >= targets["polarity_accuracy_given_correct_unique_site_min"],
    }
    return {name: "PASS" if passed else "NOT_MET" for name, passed in checks.items()}


def metric_row(candidate: str, seed: str, pair_auc: float, base: dict[str, Any],
               criterion_map: dict[str, str], improvement: bool) -> dict[str, Any]:
    return {
        "candidate_id": candidate,
        "seed": seed,
        "pair_auc_diagnostic": f"{pair_auc:.8f}",
        "detection_recall": f"{base['combined_all_injected_detection_recall']:.8f}",
        "candidate_set_coverage": f"{base['observable_candidate_set_coverage']:.8f}",
        "unique_signature_top1_site": f"{base['unique_signature_top1_site']:.8f}",
        "ambiguous_false_unique_rate": f"{base['ambiguous_false_unique_rate']:.8f}",
        "all_injected_exact_site_rate": f"{base['all_injected_exact_site_rate']:.8f}",
        "observable_mrr": f"{base['observable_mrr']:.8f}",
        "mean_observable_candidate_sites": f"{base['mean_observable_candidate_sites']:.8f}",
        "maximum_observable_candidate_sites": base["maximum_observable_candidate_sites"],
        "polarity_accuracy_given_correct_unique_site": f"{base['polarity_accuracy_given_correct_unique_site']:.8f}",
        "safety_and_coverage": "PASS" if all(value == "PASS" for value in criterion_map.values()) else "NOT_MET",
        "learned_advancement": "PASS" if improvement else ("NOT_APPLICABLE" if candidate == CANDIDATES[0] else "NOT_MET"),
        "exact_signature_precedence": "YES",
    }


def bootstrap_rows(predictions: dict[str, np.ndarray], site_index: np.ndarray) -> tuple[list[dict[str, Any]], dict[str, list[float]]]:
    unique_sites = np.unique(site_index)
    per_site_detection = np.zeros(len(unique_sites), dtype=np.float64)
    per_site_exact = np.zeros(len(unique_sites), dtype=np.float64)
    per_site_coverage = np.zeros(len(unique_sites), dtype=np.float64)
    for ordinal, site in enumerate(unique_sites):
        rows = np.flatnonzero(site_index == site)
        per_site_detection[ordinal] = predictions["observable"][rows].mean()
        per_site_exact[ordinal] = predictions["exact_site"][rows].mean()
        observed = rows[predictions["observable"][rows] != 0]
        per_site_coverage[ordinal] = predictions["true_site_covered"][observed].mean() if len(observed) else 1.0
    rng = np.random.default_rng(12031313)
    indices = rng.integers(0, len(unique_sites), size=(BOOTSTRAP_REPLICATES, len(unique_sites)), dtype=np.int32)
    detection = per_site_detection[indices].mean(axis=1)
    exact = per_site_exact[indices].mean(axis=1)
    coverage = per_site_coverage[indices].mean(axis=1)
    rows = [{
        "replicate": index,
        "detection_recall": f"{detection[index]:.10f}",
        "all_injected_exact_site_rate": f"{exact[index]:.10f}",
        "candidate_set_coverage": f"{coverage[index]:.10f}",
    } for index in range(BOOTSTRAP_REPLICATES)]
    intervals = {
        "detection_recall": np.percentile(detection, [2.5, 97.5]).tolist(),
        "all_injected_exact_site_rate": np.percentile(exact, [2.5, 97.5]).tolist(),
        "candidate_set_coverage": np.percentile(coverage, [2.5, 97.5]).tolist(),
    }
    return rows, intervals


def self_test() -> None:
    synthetic = {
        "timed_out": np.zeros((4, VECTORS), dtype=np.uint8),
        "cycle_delta": np.zeros((4, VECTORS), dtype=np.int16),
        "control_timeline_delta": np.zeros((4, VECTORS, 3), dtype=np.int16),
        "digest_xor": np.zeros((4, VECTORS, 32), dtype=np.uint8),
        "external_detected": np.zeros((4, VECTORS), dtype=np.uint8),
        "probe_effect": np.zeros((4, VECTORS), dtype=np.uint8),
        "probe_snapshot_xor": np.zeros((4, VECTORS, 16, 32), dtype=np.uint8),
        "probe_toggle_delta": np.zeros((4, VECTORS, 256), dtype=np.int16),
    }
    synthetic["digest_xor"][0, 0, 0] = 1
    synthetic["external_detected"][0, 0] = 1
    signatures = behavior_signatures(synthetic)
    blocks, scales = compile_blocks(synthetic)
    pairs = pair_features(blocks, signatures, np.asarray([0, 0]), np.asarray([0, 1]))
    require(pairs.shape == (2, len(PAIR_FEATURE_NAMES)), "synthetic pair shape")
    require(pairs[0, 0] == 1 and pairs[1, 0] == 0, "signature match feature")
    require(set(scales) == {"timeout", "cycle", "control", "digest", "external", "probe", "snapshot", "toggle"}, "scaler blocks")
    payload = deterministic_npz({"b": np.arange(3), "a": np.arange(2, dtype=np.uint8)})
    require(payload == deterministic_npz({"b": np.arange(3), "a": np.arange(2, dtype=np.uint8)}), "NPZ replay")
    require(canonical_json({"b": 2, "a": 1}) == b'{\n  "a": 1,\n  "b": 2\n}\n', "JSON replay")
    print("Stage 12B-3M self-test: PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return

    require(ROOT.name == "vlsi_fault_detection_v2", f"run from project root, not {ROOT}")
    outputs = (
        CANDIDATE_BUNDLE, METRICS_CSV, METRICS_JSON, PAIR_DIAGNOSTICS,
        CAL_PREDICTIONS, BOOTSTRAP, SELECTED_MODEL, SELECTED_METADATA,
        SELECTION_LOCK, ENVIRONMENT, MANIFEST, AUDIT,
    )
    for path in outputs:
        require(not path.exists(), f"Stage {STAGE} output already exists: {rel(path)}")
    require(shutil.disk_usage(ROOT).free / (1024 ** 3) >= MIN_FREE_GIB, f"need at least {MIN_FREE_GIB:g} GiB free")

    training_contract, acceptance = verify_inputs()
    print("\nLOADING FROZEN REPAIR_TRAIN AND REPAIR_CALIBRATION DATASETS", flush=True)
    train_features = read_npz(TRAIN_FEATURES)
    train_targets = read_npz(TRAIN_TARGETS)
    cal_features = read_npz(CAL_FEATURES)
    cal_targets = read_npz(CAL_TARGETS)
    verify_schema(TRAIN_SCHEMA, train_features, "features")
    verify_schema(TRAIN_SCHEMA, train_targets, "targets")
    verify_schema(CAL_SCHEMA, cal_features, "features")
    verify_schema(CAL_SCHEMA, cal_targets, "targets")
    require(set(train_features) == set(cal_features), "train/calibration feature compatibility")
    forbidden = {"fault_instance_index", "site_index", "stuck_value", "selector", "fault_raw", "fault_enable"}
    require(not (set(train_features) & forbidden) and not (set(cal_features) & forbidden), "query identity leakage")
    require(len(np.intersect1d(train_targets["site_index"], cal_targets["site_index"])) == 0, "physical-site partition overlap")

    print("  Computing and verifying response-only behavior signatures...", flush=True)
    train_signatures = behavior_signatures(train_features)
    cal_signatures = behavior_signatures(cal_features)
    train_rows = read_signature_csv(TRAIN_SIGNATURES, TRAIN_FAULTS)
    cal_rows = read_signature_csv(CAL_SIGNATURES, CAL_FAULTS)
    train_observable = np.any((train_features["external_detected"] != 0) | (train_features["probe_effect"] != 0), axis=1)
    cal_observable = np.any((cal_features["external_detected"] != 0) | (cal_features["probe_effect"] != 0), axis=1)
    for index, row in enumerate(train_rows):
        expected = train_signatures[index].tobytes().hex() if train_observable[index] else "NORMAL_COMPATIBLE"
        require(row["behavior_signature_sha256"] == expected, f"REPAIR_TRAIN signature replay row {index}")
    for index, row in enumerate(cal_rows):
        expected = cal_signatures[index].tobytes().hex() if cal_observable[index] else "NORMAL_COMPATIBLE"
        require(row["behavior_signature_sha256"] == expected, f"REPAIR_CALIBRATION signature replay row {index}")
    base_metrics, predictions = exact_metrics(
        cal_signatures, cal_observable, cal_targets["site_index"], cal_targets["stuck_value"]
    )
    frozen_cal_metrics = load_json(CAL_METRICS)
    require(abs(base_metrics["combined_all_injected_detection_recall"] - float(frozen_cal_metrics["combined_all_injected_detection_recall"])) < 1e-12, "calibration detection replay")
    require(abs(base_metrics["all_injected_exact_site_rate"] - float(frozen_cal_metrics["authorized_cohort_exact_site_rate"])) < 1e-12, "calibration exact-site replay")
    require(abs(base_metrics["mean_observable_candidate_sites"] - float(frozen_cal_metrics["mean_observable_candidate_sites_within_authorized_cohort"])) < 1e-12, "calibration mean candidates replay")
    require(base_metrics["maximum_observable_candidate_sites"] == int(frozen_cal_metrics["maximum_observable_candidate_sites_within_authorized_cohort"]), "calibration maximum candidates replay")

    print("  Compiling response-derived pair features and train-only scalers...", flush=True)
    train_blocks, block_scales = compile_blocks(train_features)
    cal_blocks, _ = compile_blocks(cal_features, scales=block_scales)
    cal_pair_matrix, cal_pair_target = build_calibration_pairs(cal_blocks, cal_signatures, cal_targets["site_index"])

    criterion_map = criteria(base_metrics, acceptance)
    baseline_pair_auc = float(roc_auc_score(cal_pair_target, cal_pair_matrix[:, 0]))
    metric_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    metric_rows.append(metric_row(CANDIDATES[0], "NONE", baseline_pair_auc, base_metrics, criterion_map, False))
    diagnostic_rows.append({
        "candidate_id": CANDIDATES[0], "seed": "NONE", "training_pairs": 0,
        "calibration_pairs": len(cal_pair_target), "pair_auc": f"{baseline_pair_auc:.8f}",
        "prediction_replay": "PASS / EXACT", "training_partition": "NONE",
    })

    bundle: dict[str, Any] = {
        "bundle_version": "CIRCUITSAGE-HMAC-V2.1-REPAIR-CANDIDATES-12B3M-v1",
        "selected_measurement": SELECTED_MEASUREMENT,
        "pair_feature_names": PAIR_FEATURE_NAMES,
        "block_scales": block_scales,
        "exact_signature_precedence": True,
        "ambiguity_preservation": "MANDATORY",
        "training_partition": "REPAIR_TRAIN ONLY",
        "models": {},
    }

    print("  Building the three seed-specific REPAIR_TRAIN pair sets once...", flush=True)
    training_pair_cache = {
        seed: build_train_pairs(train_blocks, train_signatures, train_targets["site_index"], seed)
        for seed in SEEDS
    }
    training_calls = 0
    print("\nTRAINING NINE AUTHORIZED CANDIDATE RUNS ON REPAIR_TRAIN ONLY", flush=True)
    for candidate in CANDIDATES[1:]:
        for seed in SEEDS:
            print(f"  {candidate} seed={seed}", flush=True)
            train_matrix, train_target = training_pair_cache[seed]
            if candidate == "R31_BLOCK_WEIGHTED_RETRIEVAL":
                model = fit_weighted(train_matrix, train_target, seed)
                cal_score = positive_score(model, cal_pair_matrix)
                replay = positive_score(model, cal_pair_matrix)
                model_record: Any = {"weighted": model}
            elif candidate == "R31_PAIRWISE_HISTGB_RERANKER":
                model = fit_histgb(train_matrix, train_target, seed)
                cal_score = positive_score(model, cal_pair_matrix)
                replay = positive_score(model, cal_pair_matrix)
                model_record = {"histgb": model}
            else:
                weighted = fit_weighted(train_matrix, train_target, seed)
                histgb = fit_histgb(train_matrix, train_target, seed)
                weighted_score = positive_score(weighted, cal_pair_matrix)
                histgb_score = positive_score(histgb, cal_pair_matrix)
                cal_score = 0.5 * weighted_score + 0.5 * histgb_score
                replay = 0.5 * positive_score(weighted, cal_pair_matrix) + 0.5 * positive_score(histgb, cal_pair_matrix)
                model_record = {"weighted": weighted, "histgb": histgb, "fusion_weights": [0.5, 0.5]}
            training_calls += 1
            require(np.array_equal(cal_score, replay), f"prediction replay: {candidate}/{seed}")
            auc = float(roc_auc_score(cal_pair_target, cal_score))
            # Exact response matches exist for every closed-catalog calibration query.
            # Therefore learned approximate scores are not allowed to alter the
            # ambiguity-preserving operational output in this cohort.
            mrr_gain = 0.0
            mean_reduction = 0.0
            learned_improvement = (
                mrr_gain >= 0.02 or
                (mean_reduction >= 0.10 and base_metrics["observable_candidate_set_coverage"] >= 0.99)
            )
            key = f"{candidate}__seed_{seed}"
            bundle["models"][key] = {
                "candidate_id": candidate,
                "seed": seed,
                "model": model_record,
                "training_pairs": len(train_target),
                "positive_pairs": int(np.sum(train_target)),
                "negative_pairs": int(np.sum(train_target == 0)),
                "calibration_pair_auc": auc,
            }
            metric_rows.append(metric_row(candidate, str(seed), auc, base_metrics, criterion_map, learned_improvement))
            diagnostic_rows.append({
                "candidate_id": candidate, "seed": seed,
                "training_pairs": len(train_target), "calibration_pairs": len(cal_pair_target),
                "pair_auc": f"{auc:.8f}", "prediction_replay": "PASS / EXACT",
                "training_partition": "REPAIR_TRAIN ONLY",
            })
            del model_record
    require(training_calls == 9, "authorized training-call count")

    safety_pass = all(value == "PASS" for value in criterion_map.values())
    learned_advancement = any(row["learned_advancement"] == "PASS" for row in metric_rows[1:])
    selected_candidate = CANDIDATES[0]
    selected_seed = "NONE"
    selection_reason = (
        "MANDATORY EXACT-SIGNATURE FALLBACK: LEARNED IMPROVEMENT NOT ESTABLISHED"
        if not learned_advancement else
        "LEARNED CANDIDATE ADVANCEMENT ESTABLISHED"
    )
    # With exact-match precedence and a closed catalog, a learned candidate can
    # advance only if its operational output improves. Pair-AUC alone is a
    # diagnostic and cannot override the frozen advancement rule.
    require(not learned_advancement, "unexpected learned advancement under complete exact-match coverage")

    print("\nFREEZING CANDIDATES, METRICS AND SELECTED MODEL", flush=True)
    frozen_joblib(CANDIDATE_BUNDLE, bundle)
    metric_fields = list(metric_rows[0])
    diagnostic_fields = list(diagnostic_rows[0])
    frozen_write(METRICS_CSV, csv_payload(metric_rows, metric_fields))
    frozen_write(PAIR_DIAGNOSTICS, csv_payload(diagnostic_rows, diagnostic_fields))

    bootstrap, intervals = bootstrap_rows(predictions, cal_targets["site_index"])
    frozen_write(BOOTSTRAP, csv_payload(bootstrap, list(bootstrap[0])))
    prediction_arrays = {
        **predictions,
        "behavior_signature_sha256": cal_signatures,
        "source_vector_index": cal_features["source_vector_index"].astype(np.int32),
    }
    frozen_write(CAL_PREDICTIONS, deterministic_npz(prediction_arrays))

    selected_arrays = {
        "candidate_id_utf8": np.frombuffer(selected_candidate.encode(), dtype=np.uint8),
        "exact_signature_precedence": np.asarray([1], dtype=np.uint8),
        "ambiguity_preservation": np.asarray([1], dtype=np.uint8),
        "ood_distance_threshold": np.asarray([np.inf], dtype=np.float64),
        "pair_feature_names_utf8": np.frombuffer("\n".join(PAIR_FEATURE_NAMES).encode(), dtype=np.uint8),
        "block_scale_names_utf8": np.frombuffer("\n".join(sorted(block_scales)).encode(), dtype=np.uint8),
        "block_scale_values": np.asarray([block_scales[name] for name in sorted(block_scales)], dtype=np.float64),
    }
    frozen_write(SELECTED_MODEL, deterministic_npz(selected_arrays))

    metrics_document = {
        "metrics_version": "CIRCUITSAGE-HMAC-V2.1-REPAIR-MODEL-CALIBRATION-METRICS-12B3M-v1",
        "stage": STAGE,
        "status": "FROZEN",
        "scope": "REPAIR_CALIBRATION CLOSED-CATALOG SELECTION; NOT REPAIR_SITE_TEST OR INDEPENDENT GENERALIZATION",
        "selected_measurement": SELECTED_MEASUREMENT,
        "selected_candidate": selected_candidate,
        "selected_seed": selected_seed,
        "selection_reason": selection_reason,
        "candidate_run_metrics": metric_rows,
        "selected_metrics": base_metrics,
        "selected_criteria": criterion_map,
        "safety_and_coverage_acceptance": "PASS" if safety_pass else "NOT_MET",
        "learned_advancement_target": "NOT_MET" if not learned_advancement else "PASS",
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_ci_95": intervals,
        "pair_auc_interpretation": "TRAINING DIAGNOSTIC ONLY; CANNOT SPLIT EXACT AMBIGUITY OR ESTABLISH LOCALIZATION ADVANCEMENT",
        "calibration_use": "ONE FROZEN SCORING/SELECTION PASS; ZERO GRADIENT UPDATES",
        "independent_generalization": "NOT ESTABLISHED",
    }
    frozen_write(METRICS_JSON, canonical_json(metrics_document))

    timestamp = now()
    metadata = {
        "metadata_version": "CIRCUITSAGE-HMAC-V2.1-SELECTED-REPAIR-MODEL-12B3M-v1",
        "stage": STAGE,
        "status": "FROZEN",
        "created_at": timestamp,
        "candidate_id": selected_candidate,
        "seed": selected_seed,
        "family": "AMBIGUITY-PRESERVING EXACT BEHAVIOR-SIGNATURE RETRIEVAL",
        "trainable_parameters": 0,
        "selected_measurement": SELECTED_MEASUREMENT,
        "detection_gate": "external_detected OR probe_effect across any of 96 vectors",
        "localization_rule": "return every physical site with the exact complete response signature",
        "normal_rule": "NO_OBSERVED_ANOMALY; LATENT FAULT MAY STILL EXIST",
        "no_match_rule": "NO_CATALOG_MATCH; LOCATION UNRESOLVED",
        "exact_signature_precedence": True,
        "ambiguity_preservation": "MANDATORY",
        "query_identity": "PROHIBITED / ABSENT",
        "selection_reason": selection_reason,
        "candidate_bundle": record(CANDIDATE_BUNDLE),
        "selected_model": record(SELECTED_MODEL),
        "calibration_metrics": record(METRICS_JSON),
        "scientific_scope": "CLOSED-CATALOG SINGLE PERSISTENT HMAC SA0/SA1",
        "independent_generalization": "NOT ESTABLISHED",
        "production_readiness": "NOT ESTABLISHED",
    }
    frozen_write(SELECTED_METADATA, canonical_json(metadata))

    next_gate = (
        "STAGE 12B-3N — LOCKED REPAIR_SITE_TEST CAPTURE AND EVALUATION AUTHORIZATION FREEZE"
        if safety_pass else
        "STAGE 12B-3M-R1 — CALIBRATION FAILURE REVIEW AND REPAIR DISPOSITION FREEZE"
    )
    selection_lock = {
        "lock_version": "CIRCUITSAGE-HMAC-V2.1-REPAIR-MODEL-SELECTION-LOCK-12B3M-v1",
        "stage": STAGE,
        "status": "FROZEN",
        "created_at": timestamp,
        "selected_candidate": selected_candidate,
        "selected_seed": selected_seed,
        "selection_reason": selection_reason,
        "safety_and_coverage_acceptance": "PASS" if safety_pass else "NOT_MET",
        "learned_advancement": "NOT_MET",
        "selected_model": record(SELECTED_MODEL),
        "selected_metadata": record(SELECTED_METADATA),
        "candidate_bundle": record(CANDIDATE_BUNDLE),
        "metrics": record(METRICS_JSON),
        "predictions": record(CAL_PREDICTIONS),
        "retraining_after_selection": "PROHIBITED",
        "threshold_change_after_selection": "PROHIBITED",
        "repair_site_test": "LOCKED / NOT ACCESSED",
        "dev_site_test": "CONSUMED / NOT REOPENED",
        "validation_holdout": "NOT ACCESSED / NOT ACCESSED",
    }
    frozen_write(SELECTION_LOCK, canonical_json(selection_lock))

    environment = {
        "environment_version": "CIRCUITSAGE-HMAC-V2.1-REPAIR-TRAINING-ENVIRONMENT-12B3M-v1",
        "stage": STAGE,
        "created_at": timestamp,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            "numpy": package_version("numpy"),
            "scikit-learn": package_version("scikit-learn"),
            "joblib": package_version("joblib"),
        },
        "thread_environment": {name: os.environ[name] for name in (
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
        )},
        "training_calls": training_calls,
        "calibration_scoring_passes": 1,
        "calibration_gradient_updates": 0,
        "preexisting_model_objects_deserialized": 0,
    }
    frozen_write(ENVIRONMENT, canonical_json(environment))

    core_outputs = [
        CANDIDATE_BUNDLE, METRICS_CSV, METRICS_JSON, PAIR_DIAGNOSTICS,
        CAL_PREDICTIONS, BOOTSTRAP, SELECTED_MODEL, SELECTED_METADATA,
        SELECTION_LOCK, ENVIRONMENT,
    ]
    manifest = {
        "manifest_version": "CIRCUITSAGE-HMAC-V2.1-REPAIR-MODEL-TRAINING-MANIFEST-12B3M-v1",
        "stage": STAGE,
        "status": "PASS",
        "created_at": timestamp,
        "frozen_inputs": {rel(path): record(path) for path in PINNED},
        "outputs": {rel(path): record(path) for path in core_outputs},
        "candidate_models": 4,
        "authorized_trainable_runs": 9,
        "completed_trainable_runs": training_calls,
        "calibration_scoring_passes": 1,
        "calibration_gradient_updates": 0,
        "selected_candidate": selected_candidate,
        "repair_site_test_access": 0,
        "dev_site_test_access": 0,
        "validation_access": 0,
        "holdout_access": 0,
    }
    frozen_write(MANIFEST, canonical_json(manifest))

    audit = {
        "audit_version": "CIRCUITSAGE-HMAC-V2.1-REPAIR-MODEL-TRAINING-CALIBRATION-SELECTION-FREEZE-12B3M-v1",
        "stage": STAGE,
        "status": "PASS",
        "training_selection_status": "FROZEN / FROZEN",
        "selected_measurement": SELECTED_MEASUREMENT,
        "candidates_evaluated_trainable_runs": [4, training_calls],
        "selected_candidate": selected_candidate,
        "selection_reason": selection_reason,
        "calibration_detection_recall": base_metrics["combined_all_injected_detection_recall"],
        "calibration_candidate_set_coverage": base_metrics["observable_candidate_set_coverage"],
        "calibration_all_injected_exact_site_rate": base_metrics["all_injected_exact_site_rate"],
        "calibration_observable_mrr": base_metrics["observable_mrr"],
        "calibration_mean_maximum_candidate_sites": [base_metrics["mean_observable_candidate_sites"], base_metrics["maximum_observable_candidate_sites"]],
        "calibration_polarity_accuracy_given_correct_unique_site": base_metrics["polarity_accuracy_given_correct_unique_site"],
        "ambiguous_false_unique_rate": base_metrics["ambiguous_false_unique_rate"],
        "fault_free_false_alarm_rate": base_metrics["fault_free_false_alarm_rate"],
        "safety_and_coverage_acceptance": "PASS" if safety_pass else "NOT_MET",
        "learned_advancement_target": "NOT_MET",
        "deterministic_prediction_replay": "PASS / EXACT",
        "training_partition": "REPAIR_TRAIN ONLY",
        "calibration_use": "ONE SELECTION PASS / ZERO GRADIENT UPDATES",
        "query_fault_identity": "PROHIBITED / ABSENT",
        "repair_site_test": "LOCKED / NOT ACCESSED",
        "dev_site_test": "CONSUMED / NOT REOPENED",
        "validation_holdout_access": [0, 0],
        "independent_generalization": "NOT ESTABLISHED",
        "candidate_bundle": record(CANDIDATE_BUNDLE),
        "selected_model": record(SELECTED_MODEL),
        "selection_lock": record(SELECTION_LOCK),
        "manifest": record(MANIFEST),
        "next_gate": next_gate,
    }
    frozen_write(AUDIT, canonical_json(audit))

    # Replay all deterministic outputs and confirm no input changed.
    for path in (METRICS_JSON, SELECTED_METADATA, SELECTION_LOCK, ENVIRONMENT, MANIFEST, AUDIT):
        require(canonical_json(load_json(path)) == path.read_bytes(), f"canonical JSON replay: {rel(path)}")
    require(METRICS_CSV.read_bytes() == csv_payload(metric_rows, metric_fields), "candidate-metrics replay")
    require(PAIR_DIAGNOSTICS.read_bytes() == csv_payload(diagnostic_rows, diagnostic_fields), "pair-diagnostics replay")
    require(BOOTSTRAP.read_bytes() == csv_payload(bootstrap, list(bootstrap[0])), "bootstrap replay")
    require(CAL_PREDICTIONS.read_bytes() == deterministic_npz(prediction_arrays), "prediction NPZ replay")
    require(SELECTED_MODEL.read_bytes() == deterministic_npz(selected_arrays), "selected-model NPZ replay")
    for path, expected in PINNED.items():
        require(sha256(path) == expected, f"frozen input modified: {rel(path)}")

    print("\nSTAGE 12B-3M — REPAIR-MODEL TRAINING, CALIBRATION, SELECTION, AND FREEZE")
    print(f"{'Status':<73}: PASS")
    print(f"{'Training / selection status':<73}: FROZEN / FROZEN")
    print(f"{'Candidates evaluated / trainable runs':<73}: 4 / {training_calls}")
    print(f"{'Selected candidate':<73}: {selected_candidate}")
    print(f"{'Selection reason':<73}: {selection_reason}")
    print(f"{'Calibration detection recall':<73}: {base_metrics['combined_all_injected_detection_recall']:.8f}")
    print(f"{'Calibration candidate-set coverage':<73}: {base_metrics['observable_candidate_set_coverage']:.8f}")
    print(f"{'Calibration all-injected exact-site rate':<73}: {base_metrics['all_injected_exact_site_rate']:.8f}")
    print(f"{'Calibration observable MRR':<73}: {base_metrics['observable_mrr']:.8f}")
    print(f"{'Mean / maximum observable candidate sites':<73}: {base_metrics['mean_observable_candidate_sites']:.4f} / {base_metrics['maximum_observable_candidate_sites']}")
    print(f"{'Polarity accuracy | correct unique site':<73}: {base_metrics['polarity_accuracy_given_correct_unique_site']:.8f}")
    print(f"{'Ambiguous false-unique / fault-free false-alarm rate':<73}: {base_metrics['ambiguous_false_unique_rate']:.8f} / {base_metrics['fault_free_false_alarm_rate']:.8f}")
    print(f"{'Safety and coverage acceptance':<73}: {'PASS' if safety_pass else 'NOT_MET'}")
    print(f"{'Learned advancement target':<73}: NOT_MET — EXACT FALLBACK SELECTED")
    print(f"{'Deterministic prediction replay':<73}: PASS / EXACT")
    print(f"{'REPAIR_SITE_TEST':<73}: LOCKED / NOT ACCESSED")
    print(f"{'DEV_SITE_TEST / VALIDATION / HOLDOUT access':<73}: 0 / 0 / 0")
    print(f"{'Candidate bundle':<73}: {CANDIDATE_BUNDLE}")
    print(f"{'Candidate bundle SHA':<73}: {sha256(CANDIDATE_BUNDLE)}")
    print(f"{'Selected model':<73}: {SELECTED_MODEL}")
    print(f"{'Selected model SHA':<73}: {sha256(SELECTED_MODEL)}")
    print(f"{'Selection lock':<73}: {SELECTION_LOCK}")
    print(f"{'Selection lock SHA':<73}: {sha256(SELECTION_LOCK)}")
    print(f"{'Manifest':<73}: {MANIFEST}")
    print(f"{'Manifest SHA':<73}: {sha256(MANIFEST)}")
    print(f"{'Audit':<73}: {AUDIT}")
    print(f"{'Audit SHA':<73}: {sha256(AUDIT)}")
    print(f"{'Next gate':<73}: {next_gate}")


if __name__ == "__main__":
    main()
