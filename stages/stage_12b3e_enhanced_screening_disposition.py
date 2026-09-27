#!/usr/bin/env python3
"""Stage 12B-3E: enhanced-screening disposition and full-capture readiness.

Verifies the frozen Stage 12B-3D bounded-screen dataset, applies the already
frozen Stage 12B-3A screening rule, locks at most one advancing measurement
candidate, and records whether a later single-winner REPAIR_TRAIN full capture
is ready for a separate authorization stage.

This stage performs no simulation, capture, training, calibration or model
inference.  It does not open REPAIR_CALIBRATION, REPAIR_SITE_TEST, the consumed
DEV_SITE_TEST partition, VALIDATION or HOLDOUT.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
from typing import Any


STAGE = "12B-3E"
ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "config/v2_1_improvement"
RESULT = ROOT / "results/circuitsage_hmac_v2_12b3"
WORK = RESULT / "enhanced_screening_disposition_12b3e"

SOURCE_3D = ROOT / "stage_12b3d_enhanced_screening_execution.py"
DATA_3D = RESULT / "enhanced_screening_execution_12b3d"
CHECKPOINT_3D = DATA_3D / "circuitsage_hmac_v2_1_enhanced_screening_checkpoint_12b3d.json"
TOPOLOGY_FEATURES_3D = DATA_3D / "circuitsage_hmac_v2_1_topology_screening_features_12b3d.npz"
STATE_FEATURES_3D = DATA_3D / "circuitsage_hmac_v2_1_state_screening_features_12b3d.npz"
TESTPOINT_FEATURES_3D = DATA_3D / "circuitsage_hmac_v2_1_testpoint_screening_features_12b3d.npz"
TARGETS_3D = DATA_3D / "circuitsage_hmac_v2_1_enhanced_screening_targets_12b3d.npz"
METRICS_CSV_3D = DATA_3D / "circuitsage_hmac_v2_1_enhanced_screening_candidate_metrics_12b3d.csv"
METRICS_JSON_3D = DATA_3D / "circuitsage_hmac_v2_1_enhanced_screening_candidate_metrics_12b3d.json"
BOOTSTRAP_3D = DATA_3D / "circuitsage_hmac_v2_1_enhanced_screening_site_bootstrap_12b3d.csv"
SCHEMA_3D = DATA_3D / "circuitsage_hmac_v2_1_enhanced_screening_dataset_schema_12b3d.json"
MANIFEST_3D = RESULT / "circuitsage_hmac_v2_1_enhanced_screening_dataset_manifest_12b3d.json"
AUDIT_3D = RESULT / "circuitsage_hmac_v2_1_enhanced_screening_execution_dataset_freeze_12b3d.json"

DISPOSITION = CONFIG / "circuitsage_hmac_v2_1_enhanced_screening_disposition_policy_12b3e.json"
READINESS = CONFIG / "circuitsage_hmac_v2_1_full_repair_capture_readiness_contract_12b3e.json"
WINNER_LOCK = WORK / "circuitsage_hmac_v2_1_enhanced_measurement_winner_lock_12b3e.json"
REGISTRY_CSV = WORK / "circuitsage_hmac_v2_1_enhanced_screening_candidate_disposition_12b3e.csv"
REPORT = WORK / "circuitsage_hmac_v2_1_enhanced_screening_disposition_report_12b3e.md"
MANIFEST = RESULT / "circuitsage_hmac_v2_1_enhanced_screening_disposition_manifest_12b3e.json"
AUDIT = RESULT / "circuitsage_hmac_v2_1_enhanced_screening_disposition_full_capture_readiness_freeze_12b3e.json"

PINNED = {
    SOURCE_3D: "3392345b66e470e1975ef32baa3a7946df8f636f2f20748b91c9d12fbfcc65ff",
    CHECKPOINT_3D: "b2a98422b2f020b400cf315cd19803dd8d98f6331bf6c6fb208f4f06edb807df",
    TOPOLOGY_FEATURES_3D: "bd3321993598469f18bd883702c19c308dff8fb0f6ab1dcc4f31ef2bc73d1f3b",
    STATE_FEATURES_3D: "c1d36a0a6d95477b74a3abb2a4a3e018805cf9549d40b151e743027bd3a89d85",
    TESTPOINT_FEATURES_3D: "08c0b0e8645bee9dcc063f51ff1744207a18c47e5a717839e4bb463b34107f15",
    TARGETS_3D: "f6f7d7cd4829448c5dfde133811de84a7489a24f27e874d41379fc789e1f833a",
    METRICS_CSV_3D: "9205f9815ed7e4c300eb2ccaa07b22ba3103307adad3ecf2e51fbef1a9e590e7",
    METRICS_JSON_3D: "3e4b20ba8d51452c7f895f8c595ab383495cd5234d5455afd37bbbfc316405b3",
    BOOTSTRAP_3D: "d82dee4c3c8566ce46d4682263dae58b24be7fe5a9d111794add44527b55e0e6",
    SCHEMA_3D: "fbf9e10b354efb4c577a765fdb68e4ee8baed2cdce185a3e0163c506fad0ac94",
    MANIFEST_3D: "5ac0b7fa29349d5e53bc8bfdd16a25f66237077280399b6590aa2be7f94dcfc4",
    AUDIT_3D: "892e305c41099b6ba986631632227de66ead1c6dc00fe6b1f930890c85c2ea4b",
}

CANDIDATE_ORDER = [
    "EM_TOPOLOGY_4X64_T24",
    "EM_STATE_CHECKPOINT_2X64_T32",
    "EM_TESTPOINT_4X64_T16",
]
SELECTED = "EM_TESTPOINT_4X64_T16"
SCREEN_SITES = 256
SCREEN_FAULTS = 512
SCREEN_VECTORS = 48
FULL_SITES = 1024
FULL_FAULTS = 2048
FULL_VECTORS = 96
FULL_ENABLED = FULL_FAULTS * FULL_VECTORS
FROZEN_TARGET = 0.70
FALSE_ALARM_TARGET = 0.0
BOOTSTRAP_REPLICATES = 1000


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


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode()


def load_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing JSON: {rel(path)}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON object required: {rel(path)}")
    return value


def load_csv(path: Path) -> list[dict[str, str]]:
    require(path.is_file(), f"missing CSV: {rel(path)}")
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def csv_payload(rows: list[dict[str, Any]]) -> bytes:
    require(bool(rows), "CSV rows")
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode()


def frozen_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.exists(), f"refusing to overwrite frozen output: {rel(path)}")
    temporary = path.with_name(path.name + ".tmp")
    require(not temporary.exists(), f"stale temporary output: {rel(temporary)}")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def record(path: Path) -> dict[str, Any]:
    return {"path": rel(path), "sha256": sha256(path), "bytes": path.stat().st_size}


def resolve_record(item: dict[str, Any]) -> Path:
    value = item.get("path")
    require(isinstance(value, str) and value, "artifact record path")
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def verify_record(item: Any, expected: Path, label: str) -> None:
    require(isinstance(item, dict), f"{label} record")
    path = resolve_record(item)
    require(path.resolve() == expected.resolve(), f"{label} path")
    require(path.is_file(), f"missing {label}: {rel(path)}")
    require(item.get("sha256") == sha256(path), f"{label} SHA")
    require(int(item.get("bytes", -1)) == path.stat().st_size, f"{label} size")


def percentile(values: list[float], q: float) -> float:
    require(bool(values), "percentile values")
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def verify_inputs() -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, list[float]]]:
    print("STAGE 12B-3E — ENHANCED-SCREENING DISPOSITION")
    print("FROZEN INPUT VERIFICATION")
    for path, expected in PINNED.items():
        require(path.is_file(), f"missing frozen input: {rel(path)}")
        require(sha256(path) == expected, f"frozen input SHA: {rel(path)}")
        print(f"  {path.name:<96}: OK", flush=True)

    checkpoint = load_json(CHECKPOINT_3D)
    require(checkpoint.get("status") == "PASS", "12B-3D checkpoint status")
    completed = checkpoint.get("completed_batches")
    require(isinstance(completed, list) and len(completed) == 45, "checkpoint completed batches")

    audit = load_json(AUDIT_3D)
    require(audit.get("status") == "PASS", "12B-3D status")
    require(audit.get("screening_execution") == "COMPLETED / FROZEN", "screen execution state")
    require(audit.get("dataset_status") == "FROZEN", "screen dataset state")
    require(audit.get("batches_verified") == "45/45", "screen batch integrity")
    require(audit.get("candidate_batch_pairs") == "135/135", "candidate-batch integrity")
    require(audit.get("screen_sites_faults_vectors") == [SCREEN_SITES, SCREEN_FAULTS, SCREEN_VECTORS], "screen dimensions")
    require(audit.get("physical_enabled_simulations") == SCREEN_FAULTS * SCREEN_VECTORS, "physical simulation count")
    require(audit.get("logical_candidate_evaluations") == SCREEN_FAULTS * SCREEN_VECTORS * 3, "logical evaluation count")
    require(audit.get("passing_candidates") == [SELECTED], "single advancing candidate")
    require(audit.get("advancement_selection") == "NOT PERFORMED", "selection entry state")
    require(audit.get("missing_duplicate_unknown_baseline_failures") == [0, 0, 0, 0], "dataset semantic integrity")
    require(audit.get("deterministic_dataset_replay") == "PASS / EXACT", "dataset replay")
    require(audit.get("identity_fields_in_feature_npz") == 0, "identity leakage")
    require(audit.get("fault_selector_value_raw_as_features") == "PROHIBITED / ABSENT", "forbidden features")
    require(audit.get("model_training_inference") == "0 / 0", "model-operation boundary")
    require(audit.get("repair_calibration") == "LOCKED / NOT ACCESSED", "repair-calibration boundary")
    require(audit.get("repair_site_test") == "LOCKED / NOT ACCESSED", "repair-test boundary")
    require(audit.get("original_dev_site_test") == "CONSUMED / NOT REOPENED", "original site-test boundary")
    require(audit.get("validation_access") == 0 and audit.get("holdout_access") == 0, "protected partition access")
    require(audit.get("frozen_rtl_modified") is False and audit.get("canonical_netlists_modified") is False, "frozen design integrity")
    verify_record(audit.get("metrics"), METRICS_JSON_3D, "12B-3D metrics")
    verify_record(audit.get("manifest"), MANIFEST_3D, "12B-3D manifest")

    manifest = load_json(MANIFEST_3D)
    require(manifest.get("status") == "PASS", "12B-3D manifest status")
    require(manifest.get("canonical_batches") == 45, "manifest batch count")
    require(manifest.get("screen_sites") == SCREEN_SITES and manifest.get("fault_instances") == SCREEN_FAULTS, "manifest cohort")
    require(manifest.get("selected_vectors") == SCREEN_VECTORS, "manifest vector count")
    require(manifest.get("missing_samples") == 0 and manifest.get("duplicate_samples") == 0, "manifest sample integrity")
    require(manifest.get("training_calls") == 0 and manifest.get("inference_calls") == 0, "manifest model calls")
    feature_records = manifest.get("candidate_feature_datasets")
    require(isinstance(feature_records, dict), "candidate feature records")
    expected_features = {
        "EM_TOPOLOGY_4X64_T24": TOPOLOGY_FEATURES_3D,
        "EM_STATE_CHECKPOINT_2X64_T32": STATE_FEATURES_3D,
        "EM_TESTPOINT_4X64_T16": TESTPOINT_FEATURES_3D,
    }
    for candidate, path in expected_features.items():
        verify_record(feature_records.get(candidate), path, f"{candidate} features")
    outputs = manifest.get("outputs")
    require(isinstance(outputs, dict), "12B-3D output records")
    for path in (TARGETS_3D, METRICS_CSV_3D, METRICS_JSON_3D, BOOTSTRAP_3D, SCHEMA_3D):
        verify_record(outputs.get(rel(path)), path, path.name)

    metrics = load_json(METRICS_JSON_3D)
    require(metrics.get("status") == "FROZEN", "metrics status")
    require(metrics.get("scope") == "REPAIR_TRAIN BOUNDED SCREEN ONLY", "metrics scope")
    require(metrics.get("screen_sites") == SCREEN_SITES and metrics.get("fault_instances") == SCREEN_FAULTS, "metrics cohort")
    require(metrics.get("vectors") == SCREEN_VECTORS, "metrics vectors")
    require(abs(float(metrics.get("frozen_detection_target")) - FROZEN_TARGET) < 1e-15, "frozen target")
    require(abs(float(metrics.get("fault_free_false_alarm_target")) - FALSE_ALARM_TARGET) < 1e-15, "false-alarm target")
    require(metrics.get("site_bootstrap_replicates") == BOOTSTRAP_REPLICATES, "bootstrap count")
    require(metrics.get("passing_candidates") == [SELECTED], "metrics advancing candidates")
    require(metrics.get("maximum_advancing_candidates") == 1, "maximum advancing candidates")

    raw_rows = metrics.get("candidate_metrics")
    require(isinstance(raw_rows, list) and len(raw_rows) == 3, "candidate metrics rows")
    rows: dict[str, dict[str, Any]] = {}
    for row in raw_rows:
        require(isinstance(row, dict), "candidate metric object")
        name = row.get("candidate_id")
        require(name in CANDIDATE_ORDER and name not in rows, "candidate metric identity")
        rows[name] = row
    require(list(rows) == CANDIDATE_ORDER, "candidate metric order")
    csv_rows = load_csv(METRICS_CSV_3D)
    require([row["candidate_id"] for row in csv_rows] == CANDIDATE_ORDER, "metrics CSV candidate order")
    for csv_row in csv_rows:
        row = rows[csv_row["candidate_id"]]
        require(abs(float(csv_row["combined_detection_recall"]) - float(row["combined_detection_recall"])) < 1e-12, "CSV/JSON recall agreement")
        require(csv_row["acceptance"] == row["acceptance"], "CSV/JSON acceptance agreement")

    selected = rows[SELECTED]
    require(abs(float(selected["combined_detection_recall"]) - 0.75976562) < 1e-8, "selected recall")
    require(int(selected["probe_rescued_faults"]) == 142, "selected rescued faults")
    require(int(selected["fault_free_false_alarms"]) == 0, "selected false alarms")
    require(selected["acceptance"] == "PASS", "selected screening acceptance")
    for name in CANDIDATE_ORDER[:2]:
        require(abs(float(rows[name]["combined_detection_recall"]) - 0.5) < 1e-12, f"{name} recall")
        require(rows[name]["acceptance"] == "NOT_MET", f"{name} disposition")

    bootstrap_rows = load_csv(BOOTSTRAP_3D)
    require(len(bootstrap_rows) == BOOTSTRAP_REPLICATES * 3, "bootstrap row count")
    bootstrap: dict[str, list[float]] = {name: [] for name in CANDIDATE_ORDER}
    for row in bootstrap_rows:
        name = row.get("candidate_id")
        require(name in bootstrap, "bootstrap candidate")
        bootstrap[name].append(float(row["combined_detection_recall"]))
    for name in CANDIDATE_ORDER:
        require(len(bootstrap[name]) == BOOTSTRAP_REPLICATES, f"{name} bootstrap count")
        lower = percentile(bootstrap[name], 0.025)
        upper = percentile(bootstrap[name], 0.975)
        require(abs(lower - float(rows[name]["site_bootstrap_ci_low"])) < 5e-8, f"{name} bootstrap lower CI")
        require(abs(upper - float(rows[name]["site_bootstrap_ci_high"])) < 5e-8, f"{name} bootstrap upper CI")
    return audit, rows, bootstrap


def self_test() -> None:
    require(FULL_ENABLED == 196608, "full-capture transaction canary")
    require(SCREEN_FAULTS * SCREEN_VECTORS == 24576, "screen transaction canary")
    require(0.75976562 >= FROZEN_TARGET, "selected-candidate canary")
    require(0.5 < FROZEN_TARGET, "rejected-candidate canary")
    values = [0.0, 0.25, 0.5, 0.75, 1.0]
    require(abs(percentile(values, 0.5) - 0.5) < 1e-15, "percentile canary")
    print("Stage 12B-3E self-test: PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return

    require(ROOT.name == "vlsi_fault_detection_v2", f"run from project root, not {ROOT}")
    outputs = (DISPOSITION, READINESS, WINNER_LOCK, REGISTRY_CSV, REPORT, MANIFEST, AUDIT)
    for path in outputs:
        require(not path.exists(), f"Stage {STAGE} output already exists: {rel(path)}")

    audit_3d, rows, bootstrap = verify_inputs()
    selected = rows[SELECTED]
    selected_recall = float(selected["combined_detection_recall"])
    selected_rescued = int(selected["probe_rescued_faults"])
    selected_ci = [
        float(selected["site_bootstrap_ci_low"]),
        float(selected["site_bootstrap_ci_high"]),
    ]
    margin = selected_recall - FROZEN_TARGET
    runner_up_recall = max(float(rows[name]["combined_detection_recall"]) for name in CANDIDATE_ORDER if name != SELECTED)
    winner_gap = selected_recall - runner_up_recall
    screening_gate = (
        selected_recall >= FROZEN_TARGET
        and int(selected["fault_free_false_alarms"]) == 0
        and len(bootstrap[SELECTED]) >= BOOTSTRAP_REPLICATES
    )
    require(screening_gate, "frozen screening gate")

    disposition = {
        "policy_version": "CIRCUITSAGE-HMAC-V2.1-ENHANCED-SCREENING-DISPOSITION-12B3E-v1",
        "stage": STAGE,
        "status": "FROZEN",
        "screening_dataset": "VALID / FROZEN",
        "screening_scope": "REPAIR_TRAIN STRATIFIED BOUNDED SCREEN ONLY",
        "candidate_count": 3,
        "passing_candidates": [SELECTED],
        "selected_candidate": SELECTED,
        "selection_rule": "UNIQUE CANDIDATE MEETING COMBINED DETECTION >=0.70 WITH ZERO FALSE ALARMS",
        "selected_combined_detection_recall": selected_recall,
        "selected_probe_rescued_faults": selected_rescued,
        "selected_site_bootstrap_ci_95": selected_ci,
        "screening_target": FROZEN_TARGET,
        "margin_over_target": margin,
        "runner_up_recall": runner_up_recall,
        "winner_recall_gap": winner_gap,
        "screening_acceptance": "PASS",
        "statistical_interpretation": "BOOTSTRAP CI REPORTED; FROZEN GATE USES POINT RECALL, ZERO FALSE ALARMS AND 1000 SITE BOOTSTRAPS",
        "claim_boundary": "SCREENING FEASIBILITY ONLY; NOT FULL REPAIR_TRAIN PERFORMANCE, MODEL ACCURACY OR INDEPENDENT GENERALIZATION",
        "rejected_candidates": [name for name in CANDIDATE_ORDER if name != SELECTED],
        "threshold_changes": 0,
        "candidate_reselection": "PROHIBITED AFTER THIS FREEZE",
    }
    readiness = {
        "contract_version": "CIRCUITSAGE-HMAC-V2.1-FULL-REPAIR-CAPTURE-READINESS-12B3E-v1",
        "stage": STAGE,
        "status": "FROZEN",
        "readiness": "READY FOR SEPARATE AUTHORIZATION",
        "selected_candidate": SELECTED,
        "selected_candidate_source": record(TESTPOINT_FEATURES_3D),
        "full_capture_partition": "REPAIR_TRAIN ONLY",
        "full_capture_sites": FULL_SITES,
        "full_capture_fault_instances": FULL_FAULTS,
        "full_capture_vectors": FULL_VECTORS,
        "maximum_enabled_transactions": FULL_ENABLED,
        "execution": "SEQUENTIAL",
        "parallel_batches": 1,
        "build_jobs": 1,
        "checkpoint_resume": "REQUIRED AFTER EACH CANONICAL BATCH",
        "minimum_free_disk_gib": 10,
        "measurement_schedule": "EM_TESTPOINT_4X64_T16 — 256 GLOBAL TEST-POINT BITS / 16 FROZEN SNAPSHOTS",
        "fault_identity_in_query": "PROHIBITED",
        "fault_selector_value_raw_as_features": "PROHIBITED",
        "full_capture_acceptance_gate": {
            "combined_all_injected_detection_recall_min": 0.70,
            "all_injected_exact_site_rate_min": 0.35,
            "mean_observable_candidate_sites_max": 50.0,
            "maximum_observable_candidate_sites_max": 500,
            "fault_free_false_alarm_rate_max": 0.0,
        },
        "full_capture_execution": "NOT AUTHORIZED BY THIS STAGE",
        "model_training": "NOT AUTHORIZED",
        "training_authorization_rule": "FULL REPAIR_TRAIN MEASUREMENT GATE MUST PASS, THEN A SEPARATE TRAINING CONTRACT MUST BE FROZEN",
        "repair_calibration": "LOCKED / NOT AUTHORIZED",
        "repair_site_test": "LOCKED / NOT AUTHORIZED",
        "original_dev_site_test": "CONSUMED / REOPENING PROHIBITED",
        "validation": "PROHIBITED",
        "holdout": "PROHIBITED",
        "v1_v2_core_v2_1_modified": [False, False, False],
    }
    winner_lock = {
        "lock_version": "CIRCUITSAGE-HMAC-V2.1-ENHANCED-MEASUREMENT-WINNER-LOCK-12B3E-v1",
        "stage": STAGE,
        "status": "FROZEN",
        "selected_candidate": SELECTED,
        "candidate_family": "SIMULATION_ONLY_TEST_POINT",
        "probe_bits": int(selected["probe_bits"]),
        "snapshots": int(selected["snapshots"]),
        "screening_metrics": {
            "combined_observable_faults": int(selected["combined_observable_faults"]),
            "combined_detection_recall": selected_recall,
            "probe_rescued_faults": selected_rescued,
            "site_bootstrap_ci_95": selected_ci,
            "fault_free_false_alarms": 0,
            "screening_acceptance": "PASS",
        },
        "source_feature_dataset": record(TESTPOINT_FEATURES_3D),
        "source_metrics": record(METRICS_JSON_3D),
        "selection_replay": "PASS / UNIQUE",
        "reselection_or_threshold_change": "PROHIBITED",
    }
    registry_rows: list[dict[str, Any]] = []
    for name in CANDIDATE_ORDER:
        row = rows[name]
        registry_rows.append({
            "candidate_id": name,
            "combined_detection_recall": row["combined_detection_recall"],
            "probe_rescued_faults": row["probe_rescued_faults"],
            "site_bootstrap_ci_low": row["site_bootstrap_ci_low"],
            "site_bootstrap_ci_high": row["site_bootstrap_ci_high"],
            "fault_free_false_alarms": row["fault_free_false_alarms"],
            "screening_acceptance": row["acceptance"],
            "disposition": "SELECTED_AND_LOCKED" if name == SELECTED else "REJECTED_PRESERVED",
        })
    registry_payload = csv_payload(registry_rows)
    report = f"""# CircuitSage-HMAC V2.1 Enhanced-Screening Disposition

## Result

The frozen Stage 12B-3D bounded screen is valid and complete. Of the three
precommitted candidates, only `{SELECTED}` met the frozen screening rule.
It reached combined all-injected detection recall {selected_recall:.8f},
rescued {selected_rescued} faults beyond the matched external observation, and
produced zero fault-free false alarms. Its 95% site-bootstrap interval is
[{selected_ci[0]:.8f}, {selected_ci[1]:.8f}] using {BOOTSTRAP_REPLICATES}
frozen replicates.

The two topology/state candidates each reached only 0.50000000 recall and are
rejected but preserved as negative screening results. The selected candidate's
point-recall margin over the unchanged 0.70000000 target is {margin:.8f}; its
gap over the best rejected candidate is {winner_gap:.8f}.

## Scientific boundary

This was a stratified 256-site, 512-fault REPAIR_TRAIN screen. It establishes
that the test-point measurement is worth a full capture; it does not establish
its full-cohort recall, localization performance, model accuracy, unseen-fault
generalization, unseen-circuit generalization, or production readiness.

## Disposition

- `{SELECTED}` is selected and frozen as the sole advancing measurement.
- A later 1,024-site / 2,048-fault / 96-vector REPAIR_TRAIN capture is ready
  for a separate authorization stage ({FULL_ENABLED} enabled transactions).
- This stage does not authorize that execution and does not authorize model
  training.
- The full capture must still meet recall >=0.70, all-injected exact-site rate
  >=0.35, mean observable candidate sites <=50, maximum <=500, and zero
  fault-free false alarms.
- REPAIR_CALIBRATION, REPAIR_SITE_TEST, the consumed DEV_SITE_TEST,
  VALIDATION and HOLDOUT remain locked or prohibited.
"""

    frozen_write(DISPOSITION, canonical_json(disposition))
    frozen_write(READINESS, canonical_json(readiness))
    frozen_write(WINNER_LOCK, canonical_json(winner_lock))
    frozen_write(REGISTRY_CSV, registry_payload)
    frozen_write(REPORT, report.encode())

    primary = (DISPOSITION, READINESS, WINNER_LOCK, REGISTRY_CSV, REPORT)
    manifest = {
        "manifest_version": "CIRCUITSAGE-HMAC-V2.1-ENHANCED-SCREENING-DISPOSITION-MANIFEST-12B3E-v1",
        "stage": STAGE,
        "status": "PASS",
        "stage_12b3d_audit": record(AUDIT_3D),
        "stage_12b3d_manifest": record(MANIFEST_3D),
        "outputs": {rel(path): record(path) for path in primary},
        "selected_candidate": SELECTED,
        "screening_acceptance": "PASS",
        "full_capture_readiness": "READY FOR SEPARATE AUTHORIZATION",
        "simulation_calls": 0,
        "capture_calls": 0,
        "model_objects_deserialized": 0,
        "training_calls": 0,
        "inference_calls": 0,
        "repair_calibration_access": 0,
        "repair_site_test_access": 0,
        "original_dev_site_test_access": 0,
        "validation_access": 0,
        "holdout_access": 0,
        "frozen_rtl_modified": False,
        "canonical_netlists_modified": False,
    }
    frozen_write(MANIFEST, canonical_json(manifest))
    audit = {
        "audit_version": "CIRCUITSAGE-HMAC-V2.1-ENHANCED-SCREENING-DISPOSITION-FULL-CAPTURE-READINESS-FREEZE-12B3E-v1",
        "stage": STAGE,
        "status": "PASS",
        "disposition_status": "FROZEN",
        "screening_dataset": "VALID / FROZEN",
        "screening_candidates_passing": "1/3",
        "selected_candidate": SELECTED,
        "selected_combined_detection_recall": selected_recall,
        "selected_probe_rescued_faults": selected_rescued,
        "selected_site_bootstrap_ci_95": selected_ci,
        "fault_free_false_alarms": 0,
        "frozen_detection_target": FROZEN_TARGET,
        "screening_acceptance": "PASS",
        "full_capture_readiness": "READY FOR SEPARATE AUTHORIZATION",
        "full_capture_sites_faults_vectors": [FULL_SITES, FULL_FAULTS, FULL_VECTORS],
        "full_capture_enabled_transactions": FULL_ENABLED,
        "full_capture_execution": "NOT AUTHORIZED",
        "model_training": "NOT AUTHORIZED",
        "claim_scope": "REPAIR_TRAIN BOUNDED SCREENING FEASIBILITY ONLY",
        "repair_calibration": "LOCKED / NOT ACCESSED",
        "repair_site_test": "LOCKED / NOT ACCESSED",
        "original_dev_site_test": "CONSUMED / NOT REOPENED",
        "validation_access": 0,
        "holdout_access": 0,
        "v1_modified": False,
        "v2_core_modified": False,
        "v2_1_modified": False,
        "disposition": record(DISPOSITION),
        "readiness_contract": record(READINESS),
        "winner_lock": record(WINNER_LOCK),
        "report": record(REPORT),
        "manifest": record(MANIFEST),
        "next_gate": "STAGE 12B-3F — SINGLE-WINNER FULL REPAIR_TRAIN CAPTURE AUTHORIZATION FREEZE",
    }
    frozen_write(AUDIT, canonical_json(audit))

    require(registry_payload == REGISTRY_CSV.read_bytes(), "candidate registry replay")
    for path in (DISPOSITION, READINESS, WINNER_LOCK, MANIFEST, AUDIT):
        require(canonical_json(load_json(path)) == path.read_bytes(), f"canonical JSON replay: {path.name}")
    for path, expected in PINNED.items():
        require(sha256(path) == expected, f"frozen input changed: {rel(path)}")

    print("\nSTAGE 12B-3E — ENHANCED MEASUREMENT SCREENING DISPOSITION AND FULL-CAPTURE READINESS FREEZE")
    print(f"{'Status':<67}: PASS")
    print(f"{'Disposition status':<67}: FROZEN")
    print(f"{'Screening dataset':<67}: VALID / FROZEN")
    print(f"{'Passing candidates':<67}: 1/3")
    print(f"{'Selected candidate':<67}: {SELECTED}")
    print(f"{'Selected combined detection recall':<67}: {selected_recall:.8f}")
    print(f"{'Selected rescued faults':<67}: {selected_rescued}")
    print(f"{'Selected 95% site-bootstrap CI':<67}: [{selected_ci[0]:.8f}, {selected_ci[1]:.8f}]")
    print(f"{'Frozen target / margin':<67}: {FROZEN_TARGET:.8f} / +{margin:.8f}")
    print(f"{'Fault-free false alarms':<67}: 0")
    print(f"{'Full-capture readiness':<67}: READY FOR SEPARATE AUTHORIZATION")
    print(f"{'Full-capture sites / faults / vectors':<67}: {FULL_SITES} / {FULL_FAULTS} / {FULL_VECTORS}")
    print(f"{'Full-capture enabled transactions':<67}: {FULL_ENABLED}")
    print(f"{'Full-capture execution / model training':<67}: NOT AUTHORIZED / NOT AUTHORIZED")
    print(f"{'REPAIR_CALIBRATION / REPAIR_SITE_TEST':<67}: LOCKED / LOCKED")
    print(f"{'DEV_SITE_TEST / VALIDATION / HOLDOUT access':<67}: 0 / 0 / 0")
    print(f"{'V1 / V2 Core / V2.1 modified':<67}: NO / NO / NO")
    print(f"{'Disposition':<67}: {DISPOSITION}")
    print(f"{'Disposition SHA':<67}: {sha256(DISPOSITION)}")
    print(f"{'Readiness contract':<67}: {READINESS}")
    print(f"{'Readiness contract SHA':<67}: {sha256(READINESS)}")
    print(f"{'Winner lock':<67}: {WINNER_LOCK}")
    print(f"{'Winner lock SHA':<67}: {sha256(WINNER_LOCK)}")
    print(f"{'Report':<67}: {REPORT}")
    print(f"{'Report SHA':<67}: {sha256(REPORT)}")
    print(f"{'Manifest':<67}: {MANIFEST}")
    print(f"{'Manifest SHA':<67}: {sha256(MANIFEST)}")
    print(f"{'Audit':<67}: {AUDIT}")
    print(f"{'Audit SHA':<67}: {sha256(AUDIT)}")
    print(f"{'Next gate':<67}: STAGE 12B-3F — SINGLE-WINNER FULL REPAIR_TRAIN CAPTURE AUTHORIZATION FREEZE")


if __name__ == "__main__":
    main()
