#!/usr/bin/env python3
"""Stage 12B-3L: repair-model training and calibration authorization freeze.

Verifies the frozen 12B-3I model contracts, the 12B-3G REPAIR_TRAIN
measurement dataset, and the 12B-3K REPAIR_CALIBRATION measurement dataset.
It freezes a deterministic candidate-run plan and authorizes the next stage to
fit only on REPAIR_TRAIN and to use REPAIR_CALIBRATION once for selection and
confidence calibration.  This stage performs no fitting, inference, threshold
selection, or model deserialization and does not open REPAIR_SITE_TEST,
DEV_SITE_TEST, VALIDATION, or HOLDOUT.
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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import numpy as np
except ImportError as error:
    raise SystemExit("STOP: activate the project .venv (NumPy required)") from error


STAGE = "12B-3L"
ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "config/v2_1_improvement"
RESULT = ROOT / "results/circuitsage_hmac_v2_12b3"
WORK = RESULT / "repair_model_training_authorization_12b3l"

# Frozen Stage 12B-3I contracts.
SOURCE_3I = ROOT / "stage_12b3i_repair_model_contract.py"
ARCHITECTURE_3I = CONFIG / "circuitsage_hmac_v2_1_repair_model_architecture_12b3i.json"
FEATURE_CONTRACT_3I = CONFIG / "circuitsage_hmac_v2_1_repair_model_feature_contract_12b3i.json"
PARTITION_CONTRACT_3I = CONFIG / "circuitsage_hmac_v2_1_repair_model_data_partition_contract_12b3i.json"
TRAINING_CONTRACT_3I = CONFIG / "circuitsage_hmac_v2_1_repair_model_training_contract_12b3i.json"
ACCEPTANCE_3I = CONFIG / "circuitsage_hmac_v2_1_repair_model_acceptance_contract_12b3i.json"
GRID_3I = CONFIG / "circuitsage_hmac_v2_1_repair_model_candidate_grid_12b3i.csv"
ENVIRONMENT_3I = RESULT / "repair_model_contract_12b3i/circuitsage_hmac_v2_1_repair_model_environment_12b3i.json"
REPORT_3I = RESULT / "repair_model_contract_12b3i/circuitsage_hmac_v2_1_repair_model_contract_report_12b3i.md"
MANIFEST_3I = RESULT / "circuitsage_hmac_v2_1_repair_model_contract_manifest_12b3i.json"
AUDIT_3I = RESULT / "circuitsage_hmac_v2_1_repair_model_contract_freeze_12b3i.json"

# Frozen REPAIR_TRAIN measurement dataset from Stage 12B-3G.
TRAIN_DIR = RESULT / "full_capture_execution_12b3g"
TRAIN_FEATURES = TRAIN_DIR / "circuitsage_hmac_v2_1_full_capture_features_12b3g.npz"
TRAIN_TARGETS = TRAIN_DIR / "circuitsage_hmac_v2_1_full_capture_targets_12b3g.npz"
TRAIN_SIGNATURES = TRAIN_DIR / "circuitsage_hmac_v2_1_full_capture_signature_summary_12b3g.csv"
TRAIN_METRICS = TRAIN_DIR / "circuitsage_hmac_v2_1_full_capture_metrics_12b3g.json"
TRAIN_BOOTSTRAP = TRAIN_DIR / "circuitsage_hmac_v2_1_full_capture_site_bootstrap_12b3g.csv"
TRAIN_SCHEMA = TRAIN_DIR / "circuitsage_hmac_v2_1_full_capture_dataset_schema_12b3g.json"
TRAIN_MANIFEST = RESULT / "circuitsage_hmac_v2_1_full_capture_dataset_manifest_12b3g.json"
TRAIN_AUDIT = RESULT / "circuitsage_hmac_v2_1_full_capture_execution_dataset_freeze_12b3g.json"

# Frozen REPAIR_CALIBRATION measurement dataset from Stage 12B-3K.
SOURCE_3K = ROOT / "stage_12b3k_repair_calibration_capture.py"
CAL_DIR = RESULT / "repair_calibration_capture_execution_12b3k"
CAL_FEATURES = CAL_DIR / "circuitsage_hmac_v2_1_repair_calibration_capture_features_12b3k.npz"
CAL_TARGETS = CAL_DIR / "circuitsage_hmac_v2_1_repair_calibration_capture_targets_12b3k.npz"
CAL_SIGNATURES = CAL_DIR / "circuitsage_hmac_v2_1_repair_calibration_capture_signature_summary_12b3k.csv"
CAL_METRICS = CAL_DIR / "circuitsage_hmac_v2_1_repair_calibration_capture_metrics_12b3k.json"
CAL_BOOTSTRAP = CAL_DIR / "circuitsage_hmac_v2_1_repair_calibration_capture_site_bootstrap_12b3k.csv"
CAL_SCHEMA = CAL_DIR / "circuitsage_hmac_v2_1_repair_calibration_capture_dataset_schema_12b3k.json"
CAL_MANIFEST = RESULT / "circuitsage_hmac_v2_1_repair_calibration_capture_dataset_manifest_12b3k.json"
CAL_AUDIT = RESULT / "circuitsage_hmac_v2_1_repair_calibration_capture_execution_dataset_freeze_12b3k.json"

# Stage 12B-3L outputs.
EXECUTION_CONTRACT = CONFIG / "circuitsage_hmac_v2_1_repair_model_training_execution_contract_12b3l.json"
AUTHORIZATION = CONFIG / "circuitsage_hmac_v2_1_repair_model_training_authorization_12b3l.json"
DATA_LOCK = WORK / "circuitsage_hmac_v2_1_repair_train_calibration_data_lock_12b3l.json"
EXECUTION_PLAN = WORK / "circuitsage_hmac_v2_1_repair_model_candidate_execution_plan_12b3l.csv"
PREFLIGHT = WORK / "circuitsage_hmac_v2_1_repair_model_training_preflight_12b3l.json"
MANIFEST = RESULT / "circuitsage_hmac_v2_1_repair_model_training_authorization_manifest_12b3l.json"
AUDIT = RESULT / "circuitsage_hmac_v2_1_repair_model_training_authorization_freeze_12b3l.json"

PINNED = {
    SOURCE_3I: "91791f1963313bb25f69810b391bf597a96882acdaea76f1386c4d7d5f25762b",
    ARCHITECTURE_3I: "61d593a3f490d39bcb2d5dfd731e37ec0ab76af9978a44bc0f9463f6042ab407",
    FEATURE_CONTRACT_3I: "b9cfd0d73a0f76e4d661c5476615fc7f236e85c5038eedcd83e3068cf021cbf6",
    PARTITION_CONTRACT_3I: "44420dd5fb37cdca9a8207920d69fd5e3bfee6847a4f63db34b91bae54ece562",
    TRAINING_CONTRACT_3I: "aeb765ef519313679fe3feba9714de8231401928631a2dc157c68db715a10e2d",
    ACCEPTANCE_3I: "ca95bc7fc7587e995f48b0efa03390734192fd251cd82ebd25387c5663d1e0d2",
    GRID_3I: "d9026b2f9c487adc3abc552637a54791960b20f0fbc8c023d55a0f49f2f56fdc",
    ENVIRONMENT_3I: "c6c79520bc320544616f533646935040445db2021232a37347c00770e8bf7455",
    REPORT_3I: "ec44396e32d2fcf1c88017a1feca78d49370c1cf52c531caa7f41741745b768b",
    MANIFEST_3I: "b3d62d856dc420a0e5cc51db53fbb09c2fcc2d985b8f4b7bb6ef124d17cd5fc2",
    AUDIT_3I: "ab2c5096b9287137371ec35db46bf53c0c9ed512985820a4b2705c71929aa108",
    TRAIN_FEATURES: "ea3af3be023fcafa4023776f6650ac9fc1c4cac429b2730eac5329a22e1d161d",
    TRAIN_TARGETS: "895457c94418b16e8d0f5e2a1464c5f03bb5a1673e40e23cdd71c95002ddd5b3",
    TRAIN_SIGNATURES: "6525f34f7b6321faf38d9229a11b1e453db804b536d322137af73f37ac271706",
    TRAIN_METRICS: "f41502e7e76c9bb381b926f81db538cfff47bf4d11c5988bdd419925de96b1de",
    TRAIN_BOOTSTRAP: "780181271af787dfd09a5f45181b4cd7e8e937063a42ce4c28e58b46af992f79",
    TRAIN_SCHEMA: "f719dab6fd324cc093de41853fa374278d327db67e01ad5259fdc21808121b20",
    TRAIN_MANIFEST: "9af923d5596491fb730a1574a6d3c299ca970a2e2e6a64bc0ee1f8a43b24110a",
    TRAIN_AUDIT: "dcfb5139d6353218bb0f1899160767e0a6f885cf334f4589c31d25f11ae9e4b6",
    SOURCE_3K: "8d600461ca0399d9f351266fb5af1d13bb6cf67388a3bef0b741049e100e5538",
    CAL_FEATURES: "4a75ac2a4cc11690490ba29d26665cca29a67b0b59ae4140c6e233e32520ddc5",
    CAL_TARGETS: "a97f822efe95f41cc619743824d327c03e22da3104f9dfdbfae2cb0ef885923f",
    CAL_SIGNATURES: "15b39af6b8a082a8dd47f11897ae773067a91b05d5913a35af4bcc7d0223b46e",
    CAL_METRICS: "5e07d8c9aa75bed7916b752b58d4a245a2fe5b4e5becce9b364b1683040abcc5",
    CAL_BOOTSTRAP: "f2335da8a8322f144c93fe30d3e2d481304190a8c4c430a51b64015a30b65b6c",
    CAL_SCHEMA: "dc59d2efda4e1f7490abfe23ed19a39f203bed09c485bd882bd3708b495ca8b9",
    CAL_MANIFEST: "99b1bd0240fd76e2a4ecbfc13d262b60f3ec0f5f4db003d67dfc8d653248a8fb",
    CAL_AUDIT: "e41a3aa395b18bb0311f45db7489a9fc55d684f9ace2e236563a169bc3b5c7a7",
}

SELECTED_MEASUREMENT = "EM_TESTPOINT_4X64_T16"
TRAIN_SITES, TRAIN_FAULTS, CAL_SITES, CAL_FAULTS, VECTORS = 1024, 2048, 512, 1024, 96
SEEDS = [12031901, 12031902, 12031903]
MIN_FREE_GIB = 5.0
CANDIDATES = [
    "R31_EXACT_SIGNATURE_SET",
    "R31_BLOCK_WEIGHTED_RETRIEVAL",
    "R31_PAIRWISE_HISTGB_RERANKER",
    "R31_FUSION_RERANKER",
]
TRAINABLE = set(CANDIDATES[1:])
FORBIDDEN_FEATURES = {
    "fault_instance_index", "full_rank", "calibration_capture_rank", "site_index",
    "site_id", "selector", "stuck_value", "fault_enable", "fault_raw", "batch_id",
}


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


def record(path: Path) -> dict[str, Any]:
    return {"path": rel(path), "sha256": sha256(path), "bytes": path.stat().st_size}


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "NOT_INSTALLED"


def plan_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = [{
        "execution_order": 1,
        "candidate_id": CANDIDATES[0],
        "trainable": "NO",
        "seed": "NONE",
        "fit_partition": "NONE",
        "selection_partition": "REPAIR_CALIBRATION ONCE",
        "gradient_updates_from_calibration": "PROHIBITED",
        "maximum_threads": 1,
    }]
    order = 2
    for candidate in CANDIDATES[1:]:
        for seed in SEEDS:
            rows.append({
                "execution_order": order,
                "candidate_id": candidate,
                "trainable": "YES",
                "seed": seed,
                "fit_partition": "REPAIR_TRAIN ONLY",
                "selection_partition": "REPAIR_CALIBRATION ONCE",
                "gradient_updates_from_calibration": "PROHIBITED",
                "maximum_threads": 1,
            })
            order += 1
    return rows


def csv_payload(rows: list[dict[str, Any]]) -> bytes:
    fields = [
        "execution_order", "candidate_id", "trainable", "seed", "fit_partition",
        "selection_partition", "gradient_updates_from_calibration", "maximum_threads",
    ]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def verify_pinned_inputs() -> None:
    print("FROZEN INPUT VERIFICATION")
    for path, expected in PINNED.items():
        require(path.is_file(), f"missing frozen input: {rel(path)}")
        require(sha256(path) == expected, f"SHA mismatch: {rel(path)}")
        print(f"  {path.name:<108}: OK", flush=True)


def verify_contracts() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    architecture = load_json(ARCHITECTURE_3I)
    partition = load_json(PARTITION_CONTRACT_3I)
    training = load_json(TRAINING_CONTRACT_3I)
    acceptance = load_json(ACCEPTANCE_3I)
    audit = load_json(AUDIT_3I)
    manifest = load_json(MANIFEST_3I)

    require(architecture.get("status") == "FROZEN", "3I architecture status")
    require(architecture.get("measurement_frontend") == SELECTED_MEASUREMENT, "3I measurement")
    require(architecture.get("query_identity_inputs") == "PROHIBITED", "3I identity boundary")
    require(architecture.get("ambiguity_preservation") == "MANDATORY", "3I ambiguity rule")
    require(partition.get("status") == "FROZEN", "3I partition status")
    require(partition.get("partitions", {}).get("REPAIR_TRAIN", {}).get("state") == "CAPTURED AND FROZEN", "3I train partition")
    require(partition.get("partitions", {}).get("REPAIR_SITE_TEST", {}).get("state") == "LOCKED / NOT OPENED", "3I site-test lock")
    require(training.get("status") == "FROZEN", "3I training contract")
    require(training.get("training_authorization") == "NOT GRANTED BY THIS CONTRACT", "3I prior authorization")
    require(training.get("candidate_ids") == CANDIDATES, "3I candidate order")
    require(training.get("random_seeds") == SEEDS, "3I seeds")
    require(int(training.get("cpu_budget", {}).get("threads", 0)) == 1, "3I thread budget")
    require(int(training.get("cpu_budget", {}).get("maximum_candidate_runs", 0)) == 9, "3I run budget")
    require(acceptance.get("status") == "FROZEN", "3I acceptance status")
    require(audit.get("status") == "PASS", "3I audit")
    require(audit.get("training") == "NOT AUTHORIZED / NOT STARTED", "3I training state")
    require(audit.get("repair_site_test") == "LOCKED / NOT OPENED", "3I site-test state")
    require(audit.get("dev_site_test_validation_holdout_access") == [0, 0, 0], "3I protected access")
    require(manifest.get("training_calls") == 0 and manifest.get("inference_calls") == 0, "3I execution counts")

    with GRID_3I.open(newline="", encoding="utf-8") as stream:
        grid = list(csv.DictReader(stream))
    require([row.get("candidate_id") for row in grid] == CANDIDATES, "3I candidate grid")
    require(sum(row.get("trainable") == "YES" for row in grid) == 3, "3I trainable candidates")
    print("  12B-3I architecture, partition, training, acceptance and candidate contracts                 : PASS")
    return training, acceptance, partition


def verify_dataset_semantics() -> tuple[dict[str, Any], dict[str, Any]]:
    train_schema = load_json(TRAIN_SCHEMA)
    train_metrics = load_json(TRAIN_METRICS)
    train_manifest = load_json(TRAIN_MANIFEST)
    train_audit = load_json(TRAIN_AUDIT)
    cal_schema = load_json(CAL_SCHEMA)
    cal_metrics = load_json(CAL_METRICS)
    cal_manifest = load_json(CAL_MANIFEST)
    cal_audit = load_json(CAL_AUDIT)

    require(train_schema.get("status") == "FROZEN", "train schema")
    require(train_schema.get("source_partition") == "REPAIR_TRAIN ONLY", "train source partition")
    require(train_schema.get("model_training_or_inference") is False, "train capture execution boundary")
    require(train_schema.get("identity_exclusion") == "site/fault/stuck/selector absent from feature NPZ", "train identity exclusion")
    require(train_metrics.get("full_measurement_acceptance") == "PASS", "train measurement acceptance")
    require(float(train_metrics.get("combined_all_injected_detection_recall", 0.0)) >= 0.70, "train detection target")
    require(int(train_metrics.get("fault_free_false_alarms", -1)) == 0, "train false alarms")
    require(train_manifest.get("status") == "PASS", "train manifest")
    require(train_manifest.get("sites") == TRAIN_SITES and train_manifest.get("fault_instances") == TRAIN_FAULTS, "train cohort sizes")
    require(train_manifest.get("training_calls") == 0 and train_manifest.get("inference_calls") == 0, "train capture calls")
    require(train_manifest.get("repair_site_test_access") == 0, "train site-test access")
    require(train_manifest.get("validation_access") == 0 and train_manifest.get("holdout_access") == 0, "train protected access")
    require(train_audit.get("status") == "PASS" and train_audit.get("dataset_status") == "FROZEN", "train audit")

    require(cal_schema.get("status") == "FROZEN", "calibration schema")
    require(cal_schema.get("source_partition") == "REPAIR_CALIBRATION AUTHORIZED COHORT ONLY", "calibration source partition")
    require(cal_schema.get("model_training_or_inference") is False, "calibration capture execution boundary")
    require(cal_schema.get("identity_exclusion") == "site/fault/stuck/selector absent from feature NPZ", "calibration identity exclusion")
    require(cal_metrics.get("frozen_target_comparison") == "PASS", "calibration target comparison")
    require(cal_metrics.get("dataset_integrity_acceptance") == "PASS", "calibration integrity")
    require(cal_metrics.get("interpretation", "").startswith("AUTHORIZED-COHORT CONSISTENCY"), "calibration interpretation")
    require(cal_manifest.get("status") == "PASS", "calibration manifest")
    require(cal_manifest.get("sites") == CAL_SITES and cal_manifest.get("fault_instances") == CAL_FAULTS, "calibration cohort sizes")
    require(cal_manifest.get("training_calls") == 0 and cal_manifest.get("inference_calls") == 0, "calibration capture calls")
    require(cal_manifest.get("repair_site_test_access") == 0, "calibration site-test access")
    require(cal_manifest.get("validation_access") == 0 and cal_manifest.get("holdout_access") == 0, "calibration protected access")
    require(cal_audit.get("status") == "PASS" and cal_audit.get("dataset_status") == "FROZEN", "calibration audit")
    require(cal_audit.get("source_partition") == "REPAIR_CALIBRATION", "calibration audit partition")
    require(cal_audit.get("repair_site_test") == "LOCKED / NOT ACCESSED", "calibration site-test state")
    print("  12B-3G REPAIR_TRAIN and 12B-3K REPAIR_CALIBRATION semantic freezes                         : PASS")
    return train_schema, cal_schema


def read_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]).copy() for name in archive.files}


def verify_arrays(train_schema: dict[str, Any], cal_schema: dict[str, Any]) -> dict[str, Any]:
    print("\nSTRUCTURAL DATA PREFLIGHT")
    train_features = read_npz(TRAIN_FEATURES)
    train_targets = read_npz(TRAIN_TARGETS)
    cal_features = read_npz(CAL_FEATURES)
    cal_targets = read_npz(CAL_TARGETS)

    require(not (set(train_features) & FORBIDDEN_FEATURES), "identity field in REPAIR_TRAIN features")
    require(not (set(cal_features) & FORBIDDEN_FEATURES), "identity field in REPAIR_CALIBRATION features")
    require(set(train_features) == set(train_schema.get("features", {})), "train feature members")
    require(set(cal_features) == set(cal_schema.get("features", {})), "calibration feature members")
    require(set(train_features) == set(cal_features), "train/calibration feature compatibility")
    require(set(train_targets) == {"fault_instance_index", "full_rank", "site_index", "stuck_value"}, "train target members")
    require(set(cal_targets) == {"fault_instance_index", "calibration_capture_rank", "site_index", "stuck_value"}, "calibration target members")

    for name, value in train_features.items():
        spec = train_schema["features"][name]
        require(str(value.dtype) == spec.get("dtype") and list(value.shape) == spec.get("shape"), f"train feature schema: {name}")
    for name, value in cal_features.items():
        spec = cal_schema["features"][name]
        require(str(value.dtype) == spec.get("dtype") and list(value.shape) == spec.get("shape"), f"calibration feature schema: {name}")
    for name, value in train_targets.items():
        spec = train_schema["targets"][name]
        require(str(value.dtype) == spec.get("dtype") and list(value.shape) == spec.get("shape"), f"train target schema: {name}")
    for name, value in cal_targets.items():
        spec = cal_schema["targets"][name]
        require(str(value.dtype) == spec.get("dtype") and list(value.shape) == spec.get("shape"), f"calibration target schema: {name}")

    require(len(train_targets["site_index"]) == TRAIN_FAULTS, "train target rows")
    require(len(cal_targets["site_index"]) == CAL_FAULTS, "calibration target rows")
    require(len(np.unique(train_targets["site_index"])) == TRAIN_SITES, "train site count")
    require(len(np.unique(cal_targets["site_index"])) == CAL_SITES, "calibration site count")
    overlap = np.intersect1d(train_targets["site_index"], cal_targets["site_index"])
    require(len(overlap) == 0, "REPAIR_TRAIN/REPAIR_CALIBRATION site leakage")
    require(np.array_equal(train_targets["stuck_value"], np.tile(np.asarray([0, 1], dtype=np.uint8), TRAIN_SITES)), "train SA order")
    require(np.array_equal(cal_targets["stuck_value"], np.tile(np.asarray([0, 1], dtype=np.uint8), CAL_SITES)), "calibration SA order")
    require(train_features["source_vector_index"].shape == (VECTORS,), "train vector map")
    require(cal_features["source_vector_index"].shape == (VECTORS,), "calibration vector map")
    require(np.array_equal(train_features["source_vector_index"], cal_features["source_vector_index"]), "vector-source alignment")
    require(np.array_equal(train_features["vector_rank"], cal_features["vector_rank"]), "vector-rank alignment")
    require(np.isfinite(train_features["cycle_delta"]).all(), "non-finite training values")
    require(np.isfinite(cal_features["cycle_delta"]).all(), "non-finite calibration values")

    del train_features, train_targets, cal_features, cal_targets
    print(f"  REPAIR_TRAIN sites / faults             : {TRAIN_SITES} / {TRAIN_FAULTS}")
    print(f"  REPAIR_CALIBRATION sites / faults       : {CAL_SITES} / {CAL_FAULTS}")
    print(f"  Shared response vectors                 : {VECTORS}")
    print("  Physical-site overlap                   : 0")
    print("  Query identity fields                   : ABSENT")
    return {
        "train_sites": TRAIN_SITES,
        "train_faults": TRAIN_FAULTS,
        "calibration_sites": CAL_SITES,
        "calibration_faults": CAL_FAULTS,
        "shared_vectors": VECTORS,
        "site_overlap": 0,
        "identity_fields_in_features": 0,
    }


def self_test() -> None:
    rows = plan_rows()
    require(len(rows) == 10, "plan row count")
    require(sum(row["trainable"] == "YES" for row in rows) == 9, "trainable run count")
    require({row["candidate_id"] for row in rows} == set(CANDIDATES), "candidate coverage")
    require(all(row["maximum_threads"] == 1 for row in rows), "thread limit")
    require(csv_payload(rows) == csv_payload(rows), "plan replay")
    require(canonical_json({"b": 2, "a": 1}) == b'{\n  "a": 1,\n  "b": 2\n}\n', "canonical JSON")
    print("Stage 12B-3L self-test: PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return

    require(ROOT.name == "vlsi_fault_detection_v2", f"run from project root, not {ROOT}")
    outputs = (EXECUTION_CONTRACT, AUTHORIZATION, DATA_LOCK, EXECUTION_PLAN, PREFLIGHT, MANIFEST, AUDIT)
    for path in outputs:
        require(not path.exists(), f"Stage {STAGE} output already exists: {rel(path)}")

    verify_pinned_inputs()
    training_contract, acceptance, partition_contract = verify_contracts()
    train_schema, cal_schema = verify_dataset_semantics()
    array_checks = verify_arrays(train_schema, cal_schema)

    available_gib = shutil.disk_usage(ROOT).free / (1024 ** 3)
    require(available_gib >= MIN_FREE_GIB, f"need at least {MIN_FREE_GIB:g} GiB free")
    sklearn_version = package_version("scikit-learn")
    require(sklearn_version != "NOT_INSTALLED", "scikit-learn environment")
    timestamp = now()
    rows = plan_rows()

    data_lock = {
        "lock_version": "CIRCUITSAGE-HMAC-V2.1-REPAIR-TRAIN-CALIBRATION-DATA-LOCK-12B3L-v1",
        "stage": STAGE,
        "status": "FROZEN",
        "created_at": timestamp,
        "selected_measurement": SELECTED_MEASUREMENT,
        "repair_train": {
            "use": "FIT TRAINABLE PARAMETERS, TRAIN-ONLY SCALERS, AND INTERNAL GROUP-FOLD EARLY STOPPING",
            "sites": TRAIN_SITES,
            "fault_instances": TRAIN_FAULTS,
            "features": record(TRAIN_FEATURES),
            "targets": record(TRAIN_TARGETS),
            "schema": record(TRAIN_SCHEMA),
            "metrics": record(TRAIN_METRICS),
            "manifest": record(TRAIN_MANIFEST),
            "audit": record(TRAIN_AUDIT),
        },
        "repair_calibration": {
            "use": "ONE-TIME CANDIDATE SELECTION, CONFIDENCE/OOD CALIBRATION, AND STOPPING DECISION",
            "gradient_updates": "PROHIBITED",
            "sites": CAL_SITES,
            "fault_instances": CAL_FAULTS,
            "features": record(CAL_FEATURES),
            "targets": record(CAL_TARGETS),
            "schema": record(CAL_SCHEMA),
            "metrics": record(CAL_METRICS),
            "manifest": record(CAL_MANIFEST),
            "audit": record(CAL_AUDIT),
        },
        "physical_site_overlap": 0,
        "query_identity_fields": "PROHIBITED / ABSENT",
        "repair_site_test": "LOCKED / NOT AUTHORIZED",
        "dev_site_test": "CONSUMED / NOT AUTHORIZED",
        "validation_holdout": "NOT AUTHORIZED / NOT AUTHORIZED",
    }

    execution_contract = {
        "execution_contract_version": "CIRCUITSAGE-HMAC-V2.1-REPAIR-MODEL-TRAINING-EXECUTION-12B3L-v1",
        "stage": STAGE,
        "status": "FROZEN",
        "created_at": timestamp,
        "candidate_ids": CANDIDATES,
        "deterministic_baseline_runs": 1,
        "trainable_candidate_runs": 9,
        "random_seeds": SEEDS,
        "maximum_cpu_threads": 1,
        "training_partition": "REPAIR_TRAIN ONLY",
        "calibration_partition": "REPAIR_CALIBRATION — ONE SELECTION PASS",
        "calibration_gradient_updates": "PROHIBITED",
        "fit_operations_on_repair_site_test": 0,
        "selection_order": training_contract["selection_order"],
        "early_stopping": training_contract["early_stopping"],
        "negative_sampling": training_contract["negative_sampling"],
        "ambiguity_preservation": "MANDATORY",
        "exact_signature_precedence": True,
        "query_fault_identity": "PROHIBITED",
        "acceptance_contract": record(ACCEPTANCE_3I),
        "selection_output_requirements": [
            "serialized candidates and exact configuration",
            "per-candidate REPAIR_CALIBRATION metrics",
            "1000-replicate physical-site bootstrap",
            "selected candidate and threshold lock",
            "deterministic replay evidence",
            "explicit fallback to R31_EXACT_SIGNATURE_SET when learned improvement is not established",
        ],
        "post_selection_retraining": "PROHIBITED",
        "post_selection_threshold_changes": "PROHIBITED",
        "repair_site_test": "LOCKED UNTIL A LATER SEPARATE AUTHORIZATION",
        "dev_site_test_validation_holdout": "PROHIBITED / PROHIBITED / PROHIBITED",
    }

    authorization = {
        "authorization_version": "CIRCUITSAGE-HMAC-V2.1-REPAIR-MODEL-TRAINING-AUTHORIZATION-12B3L-v1",
        "stage": STAGE,
        "status": "FROZEN",
        "created_at": timestamp,
        "training_and_selection": "AUTHORIZED / NOT STARTED",
        "authorized_next_stage": "12B-3M",
        "authorized_actions": [
            "execute the frozen deterministic exact-signature baseline",
            "fit at most nine frozen trainable candidate/seed runs on REPAIR_TRAIN only",
            "fit scalers and mine hard negatives on REPAIR_TRAIN only",
            "score frozen candidates on REPAIR_CALIBRATION once",
            "select confidence/OOD thresholds using REPAIR_CALIBRATION once",
            "freeze exactly one selected candidate or the mandatory exact-signature fallback",
        ],
        "prohibited_actions": [
            "use REPAIR_CALIBRATION for gradient updates or scaler fitting",
            "open or infer on REPAIR_SITE_TEST",
            "reopen DEV_SITE_TEST",
            "access VALIDATION or HOLDOUT",
            "use fault identity, selector, site, stuck value, or raw diagnostic controls as query features",
            "change the measurement, candidate grid, seeds, acceptance criteria, or partitions",
            "retrain or change thresholds after selection",
        ],
        "maximum_trainable_runs": 9,
        "maximum_threads": 1,
        "repair_train": "AUTHORIZED FOR FITTING",
        "repair_calibration": "AUTHORIZED ONCE FOR SELECTION/CALIBRATION; NO GRADIENTS",
        "repair_site_test": "NOT AUTHORIZED / LOCKED",
        "dev_site_test": "NOT AUTHORIZED / CONSUMED",
        "validation_holdout": "0 / 0",
        "independent_generalization_claim": "NOT ESTABLISHED",
    }

    preflight = {
        "preflight_version": "CIRCUITSAGE-HMAC-V2.1-REPAIR-MODEL-TRAINING-PREFLIGHT-12B3L-v1",
        "stage": STAGE,
        "status": "PASS",
        "created_at": timestamp,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scikit_learn": sklearn_version,
        "available_disk_gib": round(available_gib, 2),
        "minimum_disk_gib": MIN_FREE_GIB,
        "candidate_models": len(CANDIDATES),
        "deterministic_baseline_runs": 1,
        "trainable_runs": 9,
        "array_checks": array_checks,
        "training_calls": 0,
        "inference_calls": 0,
        "model_objects_deserialized": 0,
        "scaler_fits": 0,
        "threshold_selections": 0,
        "repair_site_test_access": 0,
        "dev_site_test_access": 0,
        "validation_access": 0,
        "holdout_access": 0,
    }

    frozen_write(DATA_LOCK, canonical_json(data_lock))
    frozen_write(EXECUTION_CONTRACT, canonical_json(execution_contract))
    frozen_write(AUTHORIZATION, canonical_json(authorization))
    frozen_write(EXECUTION_PLAN, csv_payload(rows))
    frozen_write(PREFLIGHT, canonical_json(preflight))

    output_core = [DATA_LOCK, EXECUTION_CONTRACT, AUTHORIZATION, EXECUTION_PLAN, PREFLIGHT]
    manifest = {
        "manifest_version": "CIRCUITSAGE-HMAC-V2.1-REPAIR-MODEL-TRAINING-AUTHORIZATION-MANIFEST-12B3L-v1",
        "stage": STAGE,
        "status": "PASS",
        "created_at": timestamp,
        "frozen_inputs": {rel(path): record(path) for path in PINNED},
        "outputs": {rel(path): record(path) for path in output_core},
        "candidate_models": len(CANDIDATES),
        "trainable_runs_authorized": 9,
        "training_calls": 0,
        "inference_calls": 0,
        "model_objects_deserialized": 0,
        "repair_train_rows_opened_by_authorization_stage": 0,
        "repair_calibration_rows_opened_by_authorization_stage": 0,
        "repair_site_test_access": 0,
        "dev_site_test_access": 0,
        "validation_access": 0,
        "holdout_access": 0,
    }
    frozen_write(MANIFEST, canonical_json(manifest))

    audit = {
        "audit_version": "CIRCUITSAGE-HMAC-V2.1-REPAIR-MODEL-TRAINING-AUTHORIZATION-FREEZE-12B3L-v1",
        "stage": STAGE,
        "status": "PASS",
        "authorization_status": "FROZEN",
        "selected_measurement": SELECTED_MEASUREMENT,
        "repair_train_dataset": "VERIFIED / FROZEN",
        "repair_calibration_dataset": "VERIFIED / FROZEN",
        "repair_train_calibration_site_overlap": 0,
        "candidate_models_trainable_runs": [len(CANDIDATES), 9],
        "training_selection": "AUTHORIZED / NOT STARTED",
        "calibration_gradient_updates": "PROHIBITED",
        "query_fault_identity": "PROHIBITED / ABSENT",
        "ambiguity_preservation": "MANDATORY",
        "training_inference_model_deserialization": [0, 0, 0],
        "repair_site_test": "LOCKED / NOT AUTHORIZED",
        "dev_site_test": "CONSUMED / NOT REOPENED",
        "validation_holdout_access": [0, 0],
        "v1_v2_core_v2_1_modified": [False, False, False],
        "data_lock": record(DATA_LOCK),
        "execution_contract": record(EXECUTION_CONTRACT),
        "authorization": record(AUTHORIZATION),
        "execution_plan": record(EXECUTION_PLAN),
        "preflight": record(PREFLIGHT),
        "manifest": record(MANIFEST),
        "next_gate": "STAGE 12B-3M — REPAIR-MODEL TRAINING, CALIBRATION, SELECTION, AND FREEZE",
    }
    frozen_write(AUDIT, canonical_json(audit))

    for path in (DATA_LOCK, EXECUTION_CONTRACT, AUTHORIZATION, PREFLIGHT, MANIFEST, AUDIT):
        require(canonical_json(load_json(path)) == path.read_bytes(), f"canonical output replay: {rel(path)}")
    require(EXECUTION_PLAN.read_bytes() == csv_payload(rows), "execution-plan replay")
    for path, expected in PINNED.items():
        require(sha256(path) == expected, f"frozen input modified: {rel(path)}")

    print("\nSTAGE 12B-3L — REPAIR-MODEL TRAINING AND CALIBRATION AUTHORIZATION FREEZE")
    print(f"{'Status':<72}: PASS")
    print(f"{'Authorization status':<72}: FROZEN")
    print(f"{'Selected measurement':<72}: {SELECTED_MEASUREMENT}")
    print(f"{'REPAIR_TRAIN sites / faults':<72}: {TRAIN_SITES} / {TRAIN_FAULTS}")
    print(f"{'REPAIR_CALIBRATION sites / faults':<72}: {CAL_SITES} / {CAL_FAULTS}")
    print(f"{'Train/calibration physical-site overlap':<72}: 0")
    print(f"{'Candidate models / authorized trainable runs':<72}: {len(CANDIDATES)} / 9")
    print(f"{'Training / selection':<72}: AUTHORIZED / NOT STARTED")
    print(f"{'Calibration gradient updates':<72}: PROHIBITED")
    print(f"{'Query fault identity':<72}: PROHIBITED / ABSENT")
    print(f"{'Training / inference / model deserialization':<72}: 0 / 0 / 0")
    print(f"{'REPAIR_SITE_TEST':<72}: LOCKED / NOT AUTHORIZED")
    print(f"{'DEV_SITE_TEST / VALIDATION / HOLDOUT access':<72}: 0 / 0 / 0")
    print(f"{'Execution contract':<72}: {EXECUTION_CONTRACT}")
    print(f"{'Execution contract SHA':<72}: {sha256(EXECUTION_CONTRACT)}")
    print(f"{'Authorization':<72}: {AUTHORIZATION}")
    print(f"{'Authorization SHA':<72}: {sha256(AUTHORIZATION)}")
    print(f"{'Data lock':<72}: {DATA_LOCK}")
    print(f"{'Data lock SHA':<72}: {sha256(DATA_LOCK)}")
    print(f"{'Manifest':<72}: {MANIFEST}")
    print(f"{'Manifest SHA':<72}: {sha256(MANIFEST)}")
    print(f"{'Audit':<72}: {AUDIT}")
    print(f"{'Audit SHA':<72}: {sha256(AUDIT)}")
    print(f"{'Next gate':<72}: STAGE 12B-3M — REPAIR-MODEL TRAINING, CALIBRATION, SELECTION, AND FREEZE")


if __name__ == "__main__":
    main()
