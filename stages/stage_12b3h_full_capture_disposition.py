#!/usr/bin/env python3
"""Stage 12B-3H: full-capture disposition and training-readiness freeze.

Verifies the frozen Stage 12B-3G-R1 execution evidence and dataset, records the
measurement result honestly, locks the EM_TESTPOINT_4X64_T16 winner, and
authorizes creation of a repair-model architecture/data-partition contract.

This stage does not deserialize or train a model, perform inference, reopen any
evaluation partition, or authorize training by itself.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import platform
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import numpy as np
except ImportError as error:
    raise SystemExit("STOP: activate the project .venv (NumPy required)") from error


STAGE = "12B-3H"
ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "config/v2_1_improvement"
RESULT = ROOT / "results/circuitsage_hmac_v2_12b3"
SOURCE_RESULT = RESULT / "full_capture_execution_12b3g"
WORK = RESULT / "full_capture_disposition_12b3h"

SOURCE_3G = ROOT / "stage_12b3g_full_capture_execution.py"
MASTER_LOG = RESULT / "stage_12b3g_r1_resume_20260917_171406.log"
RESOURCE_LOG = RESULT / "stage_12b3g_r1_resume_resources_20260917_171406.log"
CHECKPOINT = SOURCE_RESULT / "circuitsage_hmac_v2_1_full_capture_checkpoint_12b3g.json"
FEATURES = SOURCE_RESULT / "circuitsage_hmac_v2_1_full_capture_features_12b3g.npz"
TARGETS = SOURCE_RESULT / "circuitsage_hmac_v2_1_full_capture_targets_12b3g.npz"
SIGNATURES = SOURCE_RESULT / "circuitsage_hmac_v2_1_full_capture_signature_summary_12b3g.csv"
METRICS = SOURCE_RESULT / "circuitsage_hmac_v2_1_full_capture_metrics_12b3g.json"
BOOTSTRAP = SOURCE_RESULT / "circuitsage_hmac_v2_1_full_capture_site_bootstrap_12b3g.csv"
SCHEMA = SOURCE_RESULT / "circuitsage_hmac_v2_1_full_capture_dataset_schema_12b3g.json"
MANIFEST_3G = RESULT / "circuitsage_hmac_v2_1_full_capture_dataset_manifest_12b3g.json"
AUDIT_3G = RESULT / "circuitsage_hmac_v2_1_full_capture_execution_dataset_freeze_12b3g.json"

POLICY = CONFIG / "circuitsage_hmac_v2_1_full_capture_disposition_policy_12b3h.json"
READINESS = CONFIG / "circuitsage_hmac_v2_1_repair_model_training_readiness_contract_12b3h.json"
WINNER_LOCK = WORK / "circuitsage_hmac_v2_1_full_capture_winner_dataset_lock_12b3h.json"
REGISTRY_CSV = WORK / "circuitsage_hmac_v2_1_full_capture_capability_registry_12b3h.csv"
REGISTRY_JSON = WORK / "circuitsage_hmac_v2_1_full_capture_capability_registry_12b3h.json"
REPORT = WORK / "circuitsage_hmac_v2_1_full_capture_disposition_report_12b3h.md"
MANIFEST = RESULT / "circuitsage_hmac_v2_1_full_capture_disposition_manifest_12b3h.json"
AUDIT = RESULT / "circuitsage_hmac_v2_1_full_capture_disposition_training_readiness_freeze_12b3h.json"

PINNED = {
    SOURCE_3G: "54e965fa81a6d152cd31f123d16639b404316af87637c0598fa2ea91a2852b4d",
    MASTER_LOG: "cc85e25395114f24fd4ed16fccf4481c7d7705cb3096b36408e0ff4c6417d172",
    RESOURCE_LOG: "bea2f7622b5c0870071fb8485cc9802364c6c09d50d24d08d828cab0689ea16d",
    CHECKPOINT: "fccb28d533db1cae51e0ca53da98e46d57afa27c4cc6cb5d91b788f1a798f27d",
    FEATURES: "ea3af3be023fcafa4023776f6650ac9fc1c4cac429b2730eac5329a22e1d161d",
    TARGETS: "895457c94418b16e8d0f5e2a1464c5f03bb5a1673e40e23cdd71c95002ddd5b3",
    SIGNATURES: "6525f34f7b6321faf38d9229a11b1e453db804b536d322137af73f37ac271706",
    METRICS: "f41502e7e76c9bb381b926f81db538cfff47bf4d11c5988bdd419925de96b1de",
    BOOTSTRAP: "780181271af787dfd09a5f45181b4cd7e8e937063a42ce4c28e58b46af992f79",
    SCHEMA: "f719dab6fd324cc093de41853fa374278d327db67e01ad5259fdc21808121b20",
    MANIFEST_3G: "9af923d5596491fb730a1574a6d3c299ca970a2e2e6a64bc0ee1f8a43b24110a",
    AUDIT_3G: "dcfb5139d6353218bb0f1899160767e0a6f885cf334f4589c31d25f11ae9e4b6",
}

SELECTED = "EM_TESTPOINT_4X64_T16"
SITES = 1024
FAULTS = 2048
VECTORS = 96
BATCHES = 45
BASELINE_RECORDS = 4320
ENABLED_RECORDS = 196608
TOTAL_RECORDS = 200928
DETECTION_TARGET = 0.70
EXACT_SITE_TARGET = 0.35
MEAN_CANDIDATES_MAX = 50.0
MAX_CANDIDATES_MAX = 500


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
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode()


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def frozen_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        stop(f"refusing to overwrite frozen output: {rel(path)}")
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


def csv_bytes(rows: list[dict[str, Any]], fields: list[str]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def npz_headers(path: Path) -> dict[str, dict[str, Any]]:
    headers: dict[str, dict[str, Any]] = {}
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        require(names and len(names) == len(set(names)), f"invalid NPZ members: {rel(path)}")
        for member in names:
            require(member.endswith(".npy") and "/" not in member, f"unexpected NPZ member: {member}")
            with archive.open(member) as stream:
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
                elif version in ((2, 0), (3, 0)):
                    shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
                else:
                    stop(f"unsupported NPY version {version}: {member}")
            require(not fortran, f"Fortran-order array prohibited: {member}")
            headers[member[:-4]] = {"shape": list(shape), "dtype": str(dtype)}
    return headers


def close(left: float, right: float, tolerance: float = 5e-9) -> bool:
    return abs(left - right) <= tolerance


def verify_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    print("FROZEN INPUT VERIFICATION")
    for path, expected in PINNED.items():
        require(path.is_file(), f"missing frozen input: {rel(path)}")
        require(sha256(path) == expected, f"SHA mismatch: {rel(path)}")
        print(f"  {path.name:<96}: OK")

    checkpoint = load_json(CHECKPOINT)
    metrics = load_json(METRICS)
    schema = load_json(SCHEMA)
    manifest = load_json(MANIFEST_3G)
    audit = load_json(AUDIT_3G)

    require(checkpoint.get("status") == "PASS", "checkpoint status")
    require(len(checkpoint.get("completed_batches", [])) == BATCHES, "checkpoint batch count")
    require(sorted(checkpoint["completed_batches"]) == list(range(BATCHES)), "checkpoint batch coverage")
    require(checkpoint.get("runner_version") == "CIRCUITSAGE-HMAC-V2.1-FULL-CAPTURE-RUNNER-12B3G-R1", "R1 runner checkpoint")
    require(metrics.get("status") == "FROZEN" and audit.get("status") == "PASS", "3G status")
    require(audit.get("full_measurement_acceptance") == "PASS", "3G acceptance")
    require(metrics.get("selected_candidate") == SELECTED, "selected candidate")
    require([metrics.get("sites"), metrics.get("fault_instances"), metrics.get("vectors")] == [SITES, FAULTS, VECTORS], "cohort dimensions")
    require(manifest.get("baseline_records") == BASELINE_RECORDS, "baseline records")
    require(manifest.get("enabled_records") == ENABLED_RECORDS, "enabled records")
    require(manifest.get("total_records") == TOTAL_RECORDS, "total records")
    require([manifest.get("missing_samples"), manifest.get("duplicate_samples"), manifest.get("unknown_records"), manifest.get("baseline_failures")] == [0, 0, 0, 0], "dataset integrity counters")
    require([manifest.get("training_calls"), manifest.get("inference_calls")] == [0, 0], "training/inference counters")
    require([manifest.get("repair_site_test_access"), manifest.get("original_dev_site_test_access"), manifest.get("validation_access"), manifest.get("holdout_access")] == [0, 0, 0, 0], "protected partition access")

    detection = float(metrics["combined_all_injected_detection_recall"])
    exact = float(metrics["all_injected_exact_site_rate"])
    mean_candidates = float(metrics["mean_observable_candidate_sites"])
    max_candidates = int(metrics["maximum_observable_candidate_sites"])
    false_alarms = int(metrics["fault_free_false_alarms"])
    require(detection >= DETECTION_TARGET, "detection target")
    require(exact >= EXACT_SITE_TARGET, "exact-site target")
    require(mean_candidates <= MEAN_CANDIDATES_MAX, "mean candidate target")
    require(max_candidates <= MAX_CANDIDATES_MAX, "maximum candidate target")
    require(false_alarms == 0, "false alarms")
    require(int(metrics["external_observable_faults"]) == 1027, "external-observable count")
    require(int(metrics["probe_rescued_faults"]) == 527, "rescued count")
    require(int(metrics["combined_observable_faults"]) == 1554, "combined-observable count")

    feature_headers = npz_headers(FEATURES)
    target_headers = npz_headers(TARGETS)
    require(feature_headers == schema.get("features"), "feature NPZ/schema header agreement")
    require(target_headers == schema.get("targets"), "target NPZ/schema header agreement")
    require(schema.get("identity_exclusion") == "site/fault/stuck/selector absent from feature NPZ", "identity exclusion")
    require(schema.get("source_partition") == "REPAIR_TRAIN ONLY", "source partition")

    rows = []
    with SIGNATURES.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    require(len(rows) == FAULTS, "signature row count")
    observable = [row for row in rows if int(row["observable"]) == 1]
    require(len(observable) == int(metrics["combined_observable_faults"]), "signature observable count")
    exact_count = sum(int(row["exact_site"]) for row in rows)
    require(close(exact_count / FAULTS, exact), "signature exact-site replay")
    candidate_counts = [int(row["candidate_site_count"]) for row in observable]
    require(close(sum(candidate_counts) / len(candidate_counts), mean_candidates), "signature mean-candidate replay")
    require(max(candidate_counts) == max_candidates, "signature maximum-candidate replay")

    log_text = MASTER_LOG.read_text(encoding="utf-8", errors="replace")
    require("Stage 12B-3G-R1 return code: 0" in log_text, "R1 return-code evidence")
    require("Full measurement acceptance" in log_text and "PASS" in log_text, "R1 PASS evidence")
    print("  Metrics, NPZ headers, signatures, checkpoint and access counters                         : PASS")
    return metrics, manifest, audit, schema


def self_test() -> None:
    require(TOTAL_RECORDS == BASELINE_RECORDS + ENABLED_RECORDS, "record arithmetic")
    require(close(1554 / 2048, 0.7587890625), "detection arithmetic")
    require(close(1390 / 2048, 0.6787109375), "exact-site arithmetic")
    sample_json = {"z": 1, "a": [2, 3]}
    require(canonical_json(sample_json) == canonical_json(json.loads(canonical_json(sample_json))), "canonical JSON")
    temporary = ROOT / f".stage_12b3h_selftest_{os.getpid()}.npz"
    try:
        np.savez(temporary, a=np.arange(6, dtype=np.uint8).reshape(2, 3))
        require(npz_headers(temporary) == {"a": {"shape": [2, 3], "dtype": "uint8"}}, "NPZ header reader")
    finally:
        temporary.unlink(missing_ok=True)
    print("Stage 12B-3H self-test: PASS")


def main() -> None:
    require(ROOT.name == "vlsi_fault_detection_v2", f"run from project root, not {ROOT}")
    for output in (POLICY, READINESS, WINNER_LOCK, REGISTRY_CSV, REGISTRY_JSON, REPORT, MANIFEST, AUDIT):
        require(not output.exists(), f"Stage {STAGE} output already exists: {rel(output)}")

    metrics, source_manifest, source_audit, schema = verify_inputs()
    timestamp = now()
    detection = float(metrics["combined_all_injected_detection_recall"])
    exact = float(metrics["all_injected_exact_site_rate"])
    mean_candidates = float(metrics["mean_observable_candidate_sites"])
    max_candidates = int(metrics["maximum_observable_candidate_sites"])
    invisible = FAULTS - int(metrics["combined_observable_faults"])

    policy = {
        "policy_version": "CIRCUITSAGE-HMAC-V2.1-FULL-CAPTURE-DISPOSITION-12B3H-v1",
        "stage": STAGE,
        "status": "FROZEN",
        "created_at": timestamp,
        "source_stage": "12B-3G-R1",
        "source_result": "PASS / FROZEN",
        "disposition": "FULL MEASUREMENT TARGET MET",
        "selected_measurement": SELECTED,
        "scientific_scope": "CLOSED-CATALOG REPAIR_TRAIN MEASUREMENT",
        "not_claimed": ["MODEL ACCURACY", "INDEPENDENT GENERALIZATION", "PRODUCTION OR SILICON READINESS"],
        "acceptance": {
            "combined_detection_recall": {"value": detection, "minimum": DETECTION_TARGET, "status": "PASS"},
            "all_injected_exact_site_rate": {"value": exact, "minimum": EXACT_SITE_TARGET, "status": "PASS"},
            "mean_observable_candidate_sites": {"value": mean_candidates, "maximum": MEAN_CANDIDATES_MAX, "status": "PASS"},
            "maximum_observable_candidate_sites": {"value": max_candidates, "maximum": MAX_CANDIDATES_MAX, "status": "PASS"},
            "fault_free_false_alarms": {"value": 0, "maximum": 0, "status": "PASS"},
        },
        "remaining_normal_compatible_faults": invisible,
        "winner_and_dataset": "LOCKED FOR REPAIR-MODEL CONTRACT DESIGN",
        "training": "NOT AUTHORIZED BY THIS POLICY",
    }

    readiness = {
        "contract_version": "CIRCUITSAGE-HMAC-V2.1-REPAIR-MODEL-TRAINING-READINESS-12B3H-v1",
        "stage": STAGE,
        "status": "FROZEN",
        "readiness": "READY FOR ARCHITECTURE, PARTITION, AND TRAINING-CONTRACT DESIGN",
        "actual_model_training": "NOT YET AUTHORIZED",
        "allowed_next_stage": "12B-3I",
        "allowed_actions": [
            "DEFINE REPAIR-MODEL ARCHITECTURE AND CANDIDATE GRID",
            "DEFINE GROUP-DISJOINT REPAIR_TRAIN AND REPAIR_CALIBRATION USAGE",
            "DEFINE AMBIGUITY-AWARE DETECTION AND LOCALIZATION OBJECTIVES",
            "DEFINE TRAINING, SELECTION, CALIBRATION, AND STOPPING RULES",
        ],
        "prohibited_actions": [
            "TRAIN OR FIT A MODEL BEFORE SEPARATE AUTHORIZATION",
            "OPEN REPAIR_SITE_TEST",
            "REOPEN ORIGINAL DEV_SITE_TEST",
            "ACCESS VALIDATION OR HOLDOUT",
            "PUT SITE, SELECTOR, STUCK VALUE, OR FAULT IDENTITY IN QUERY FEATURES",
            "MODIFY V1, V2 CORE, V2.1, FROZEN RTL, OR CANONICAL NETLISTS",
        ],
        "frozen_input_dataset": record(FEATURES),
        "separate_scoring_targets": record(TARGETS),
        "dataset_schema": record(SCHEMA),
        "selected_measurement": SELECTED,
        "required_output_semantics": [
            "NO_OBSERVED_ANOMALY WITH EXPLICIT UNDETECTED-FAULT CAVEAT",
            "UNIQUE SITE ONLY WHEN THE CANDIDATE SET IS EXACTLY ONE",
            "AMBIGUOUS CANDIDATE SET OTHERWISE",
            "NO_CATALOG_MATCH FOR OUT-OF-CATALOG BEHAVIOR",
        ],
        "partition_access_at_freeze": {
            "REPAIR_TRAIN": "MEASUREMENT DATASET FROZEN",
            "REPAIR_CALIBRATION": "LOCKED",
            "REPAIR_SITE_TEST": "LOCKED",
            "DEV_SITE_TEST": "CONSUMED / NOT REOPENED",
            "VALIDATION": "NOT ACCESSED",
            "HOLDOUT": "NOT ACCESSED",
        },
    }

    frozen_write(POLICY, canonical_json(policy))
    frozen_write(READINESS, canonical_json(readiness))
    winner_lock = {
        "lock_version": "CIRCUITSAGE-HMAC-V2.1-FULL-CAPTURE-WINNER-DATASET-LOCK-12B3H-v1",
        "stage": STAGE,
        "status": "FROZEN",
        "selected_measurement": SELECTED,
        "features": record(FEATURES),
        "targets": record(TARGETS),
        "schema": record(SCHEMA),
        "metrics": record(METRICS),
        "signature_summary": record(SIGNATURES),
        "source_manifest": record(MANIFEST_3G),
        "source_audit": record(AUDIT_3G),
        "feature_identity_fields": 0,
        "frozen_dimensions": {"sites": SITES, "fault_instances": FAULTS, "vectors": VECTORS, "probe_bits": 256, "snapshots": 16},
    }
    frozen_write(WINNER_LOCK, canonical_json(winner_lock))

    registry_rows = [
        {"capability": "external_output_detection", "value": "0.50146484", "status": "BASELINE", "scope": "REPAIR_TRAIN closed catalog"},
        {"capability": "enhanced_combined_detection", "value": f"{detection:.8f}", "status": "PASS", "scope": "REPAIR_TRAIN closed catalog"},
        {"capability": "probe_rescued_faults", "value": "527", "status": "PASS", "scope": "of 2048 injected faults"},
        {"capability": "all_injected_exact_site", "value": f"{exact:.8f}", "status": "PASS", "scope": "exact behavior-signature sets"},
        {"capability": "mean_observable_candidate_sites", "value": f"{mean_candidates:.4f}", "status": "PASS", "scope": "observable faults only"},
        {"capability": "maximum_observable_candidate_sites", "value": str(max_candidates), "status": "PASS", "scope": "observable faults only"},
        {"capability": "fault_free_false_alarms", "value": "0", "status": "PASS", "scope": "96 vectors x 45 batches"},
        {"capability": "remaining_normal_compatible_faults", "value": str(invisible), "status": "LIMITATION", "scope": "current measurement schedule"},
        {"capability": "independent_generalization", "value": "NOT_ESTABLISHED", "status": "LIMITATION", "scope": "unseen circuits/signatures"},
        {"capability": "repair_model_accuracy", "value": "NOT_MEASURED", "status": "NOT_STARTED", "scope": "no model trained"},
    ]
    fields = ["capability", "value", "status", "scope"]
    frozen_write(REGISTRY_CSV, csv_bytes(registry_rows, fields))
    frozen_write(REGISTRY_JSON, canonical_json({"stage": STAGE, "status": "FROZEN", "rows": registry_rows}))

    report = f"""# CircuitSage-HMAC V2.1 Full-Capture Disposition — Stage 12B-3H

## Decision

The enhanced measurement target is **met**. `EM_TESTPOINT_4X64_T16` is retained
as the frozen measurement winner for repair-model contract design.

## Frozen result

- Cohort: {SITES:,} REPAIR_TRAIN sites, {FAULTS:,} injected faults, {VECTORS} vectors
- Combined detection recall: {detection:.8%}
- All-injected exact-site rate: {exact:.8%}
- Probe-rescued faults: {int(metrics['probe_rescued_faults']):,}
- Mean / maximum observable candidate sites: {mean_candidates:.4f} / {max_candidates}
- Fault-free false alarms: 0
- Remaining normal-compatible faults: {invisible:,}

## Interpretation

These values measure deterministic consistency inside the known HMAC SA0/SA1
catalog using the frozen REPAIR_TRAIN cohort. They are not trained-model
accuracy and do not establish performance on unseen circuits, unseen fault
types, physical silicon, or out-of-catalog behavior.

## Disposition

Stage 12B-3I may define the repair-model architecture, group-disjoint data
usage, candidate grid, objectives and stopping rules. Training remains blocked
until a separate authorization verifies that contract. REPAIR_SITE_TEST,
DEV_SITE_TEST, VALIDATION and HOLDOUT remain closed.
""".encode()
    frozen_write(REPORT, report)

    outputs = [POLICY, READINESS, WINNER_LOCK, REGISTRY_CSV, REGISTRY_JSON, REPORT]
    manifest = {
        "manifest_version": "CIRCUITSAGE-HMAC-V2.1-FULL-CAPTURE-DISPOSITION-MANIFEST-12B3H-v1",
        "stage": STAGE,
        "status": "PASS",
        "created_at": timestamp,
        "frozen_inputs": {rel(path): record(path) for path in PINNED},
        "outputs": {rel(path): record(path) for path in outputs},
        "source_dataset_acceptance": "PASS",
        "repair_model_training_readiness": "READY FOR CONTRACT DESIGN",
        "training_calls": 0,
        "inference_calls": 0,
        "model_objects_deserialized": 0,
        "repair_calibration_access": 0,
        "repair_site_test_access": 0,
        "original_dev_site_test_access": 0,
        "validation_access": 0,
        "holdout_access": 0,
    }
    frozen_write(MANIFEST, canonical_json(manifest))
    audit = {
        "audit_version": "CIRCUITSAGE-HMAC-V2.1-FULL-CAPTURE-DISPOSITION-TRAINING-READINESS-FREEZE-12B3H-v1",
        "stage": STAGE,
        "status": "PASS",
        "disposition_status": "FROZEN",
        "full_measurement_target": "MET",
        "selected_measurement": SELECTED,
        "combined_detection_recall": detection,
        "all_injected_exact_site_rate": exact,
        "mean_maximum_observable_candidate_sites": [mean_candidates, max_candidates],
        "fault_free_false_alarms": 0,
        "remaining_normal_compatible_faults": invisible,
        "winner_dataset_lock": record(WINNER_LOCK),
        "repair_model_training_readiness": "READY FOR ARCHITECTURE AND CONTRACT DESIGN",
        "repair_model_training": "NOT AUTHORIZED / NOT STARTED",
        "repair_calibration_repair_site_test": "LOCKED / LOCKED",
        "dev_site_test_validation_holdout_access": [0, 0, 0],
        "v1_v2_core_v2_1_modified": [False, False, False],
        "independent_generalization": "NOT ESTABLISHED",
        "production_readiness": "NOT ESTABLISHED",
        "policy": record(POLICY),
        "readiness_contract": record(READINESS),
        "report": record(REPORT),
        "manifest": record(MANIFEST),
        "next_gate": "STAGE 12B-3I — REPAIR-MODEL ARCHITECTURE, DATA-PARTITION, AND TRAINING-CONTRACT FREEZE",
    }
    frozen_write(AUDIT, canonical_json(audit))

    for path in (POLICY, READINESS, WINNER_LOCK, REGISTRY_JSON, MANIFEST, AUDIT):
        require(canonical_json(load_json(path)) == path.read_bytes(), f"canonical output replay: {rel(path)}")
    require(REGISTRY_CSV.read_bytes() == csv_bytes(registry_rows, fields), "registry CSV replay")
    require(REPORT.read_bytes() == report, "report replay")
    for path, expected in PINNED.items():
        require(sha256(path) == expected, f"frozen input modified: {rel(path)}")

    print("\nSTAGE 12B-3H — FULL-CAPTURE RESULT DISPOSITION AND REPAIR-MODEL TRAINING READINESS FREEZE")
    print(f"{'Status':<67}: PASS")
    print(f"{'Disposition status':<67}: FROZEN")
    print(f"{'Full measurement target':<67}: MET")
    print(f"{'Selected measurement':<67}: {SELECTED}")
    print(f"{'Combined detection recall':<67}: {detection:.8f}")
    print(f"{'All-injected exact-site rate':<67}: {exact:.8f}")
    print(f"{'Mean / maximum observable candidate sites':<67}: {mean_candidates:.4f} / {max_candidates}")
    print(f"{'Remaining normal-compatible faults':<67}: {invisible}")
    print(f"{'Fault-free false alarms':<67}: 0")
    print(f"{'Repair-model contract design':<67}: AUTHORIZED")
    print(f"{'Repair-model training':<67}: NOT AUTHORIZED / NOT STARTED")
    print(f"{'REPAIR_CALIBRATION / REPAIR_SITE_TEST':<67}: LOCKED / LOCKED")
    print(f"{'DEV_SITE_TEST / VALIDATION / HOLDOUT access':<67}: 0 / 0 / 0")
    print(f"{'Independent generalization':<67}: NOT ESTABLISHED")
    print(f"{'Policy':<67}: {POLICY}")
    print(f"{'Policy SHA':<67}: {sha256(POLICY)}")
    print(f"{'Readiness contract':<67}: {READINESS}")
    print(f"{'Readiness contract SHA':<67}: {sha256(READINESS)}")
    print(f"{'Winner/dataset lock':<67}: {WINNER_LOCK}")
    print(f"{'Winner/dataset lock SHA':<67}: {sha256(WINNER_LOCK)}")
    print(f"{'Report':<67}: {REPORT}")
    print(f"{'Report SHA':<67}: {sha256(REPORT)}")
    print(f"{'Manifest':<67}: {MANIFEST}")
    print(f"{'Manifest SHA':<67}: {sha256(MANIFEST)}")
    print(f"{'Audit':<67}: {AUDIT}")
    print(f"{'Audit SHA':<67}: {sha256(AUDIT)}")
    print(f"{'Next gate':<67}: STAGE 12B-3I — REPAIR-MODEL ARCHITECTURE, DATA-PARTITION, AND TRAINING-CONTRACT FREEZE")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    arguments = parser.parse_args()
    if arguments.self_test:
        self_test()
    else:
        main()
