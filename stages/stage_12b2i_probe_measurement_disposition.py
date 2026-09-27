#!/usr/bin/env python3
"""Stage 12B-2I: probe-measurement feasibility disposition freeze.

Freezes the Stage 12B-2H scientific result, preserves the useful localization
gain from adaptive vectors, blocks training on an observation system whose
frozen detection ceiling is below target, and authorizes only creation of a
new measurement-redesign contract.  No simulation, capture, model operation,
or protected-partition access is performed.
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


STAGE = "12B-2I"
ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "config/v2_1_improvement"
RESULT = ROOT / "results/circuitsage_hmac_v2_12b2"
WORK = RESULT / "probe_measurement_disposition_12b2i"

SOURCE_2H = ROOT / "stage_12b2h_bounded_probe_capture.py"
AUDIT_2H = RESULT / "circuitsage_hmac_v2_1_probe_capture_dataset_integrity_freeze_12b2h.json"
MANIFEST_2H = RESULT / "circuitsage_hmac_v2_1_probe_capture_dataset_manifest_12b2h.json"
FEATURES_2H = RESULT / "probe_capture_12b2h/circuitsage_hmac_v2_1_probe_response_features_12b2h.npz"
TARGETS_2H = RESULT / "probe_capture_12b2h/circuitsage_hmac_v2_1_probe_response_targets_12b2h.npz"
METRICS_2H = RESULT / "probe_capture_12b2h/circuitsage_hmac_v2_1_probe_observability_metrics_12b2h.json"
SCHEMA_2H = RESULT / "probe_capture_12b2h/circuitsage_hmac_v2_1_probe_response_schema_12b2h.json"

AUDIT_2D = RESULT / "circuitsage_hmac_v2_1_adaptive_selection_campaign_contract_freeze_12b2d.json"
AUDIT_2E = RESULT / "circuitsage_hmac_v2_1_observability_ceiling_alternative_measurement_freeze_12b2e.json"
AUDIT_2F = RESULT / "circuitsage_hmac_v2_1_probe_discovery_instrumentation_feasibility_freeze_12b2f.json"

DISPOSITION = CONFIG / "circuitsage_hmac_v2_1_probe_measurement_disposition_policy_12b2i.json"
REDESIGN = CONFIG / "circuitsage_hmac_v2_1_enhanced_measurement_redesign_contract_12b2i.json"
REGISTRY_CSV = WORK / "circuitsage_hmac_v2_1_improvement_capability_registry_12b2i.csv"
REGISTRY_JSON = WORK / "circuitsage_hmac_v2_1_improvement_capability_registry_12b2i.json"
REPORT = WORK / "circuitsage_hmac_v2_1_probe_measurement_disposition_report_12b2i.md"
MANIFEST = RESULT / "circuitsage_hmac_v2_1_probe_measurement_disposition_manifest_12b2i.json"
AUDIT = RESULT / "circuitsage_hmac_v2_1_probe_measurement_disposition_freeze_12b2i.json"

PINNED = {
    SOURCE_2H: "b4132cba10b245c5d3e25d4651ea6adbc820c7a59b80db17b86d52e4d7efd919",
    AUDIT_2H: "cc68202cc1ef92d32319c43ca1d8afd2c9bb324ac24fb9125e630d50a7403d91",
    AUDIT_2D: "83c39f566ca45d5dbc58a7b5a54adfae6b83a0fb2ee2115cd2ae562bbb2348f1",
    AUDIT_2E: "5165e617f75717233b5b6619861071e6c6b0a848fce35fc69309ac06b573fef1",
    AUDIT_2F: "f1103b9e2f07e1e2b6bffe2f65fff7c92c357084e511c5358751313eafb71188",
}

FAULTS = 2048
EXTERNAL_OBSERVABLE = 1027
COMBINED_OBSERVABLE = 1030
FROZEN_TARGET = 0.70


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


def verify_record(item: dict[str, Any], expected: Path, label: str) -> None:
    require(isinstance(item, dict), f"{label} record")
    path = resolve_record(item)
    require(path.resolve() == expected.resolve(), f"{label} path")
    require(path.is_file(), f"missing {label}: {rel(path)}")
    require(item.get("sha256") == sha256(path), f"{label} SHA")
    require(int(item.get("bytes", -1)) == path.stat().st_size, f"{label} size")


def verify_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    print("STAGE 12B-2I — PROBE-MEASUREMENT FEASIBILITY DISPOSITION")
    print("FROZEN INPUT VERIFICATION")
    for path, expected in PINNED.items():
        require(path.is_file(), f"missing frozen input: {rel(path)}")
        require(sha256(path) == expected, f"frozen input SHA: {rel(path)}")
        print(f"  {path.name:<91}: OK", flush=True)

    audit = load_json(AUDIT_2H)
    require(audit.get("status") == "PASS", "12B-2H status")
    require(audit.get("capture_execution") == "COMPLETED / FROZEN", "capture state")
    require(audit.get("dataset_status") == "FROZEN", "probe dataset state")
    require(audit.get("batches_verified") == "45/45", "batch integrity")
    require(audit.get("fault_instances") == FAULTS, "fault-instance count")
    require(audit.get("external_observable_faults") == EXTERNAL_OBSERVABLE, "external-observable count")
    require(audit.get("combined_observable_faults") == COMBINED_OBSERVABLE, "combined-observable count")
    require(audit.get("probe_rescued_external_invisible_faults") == 3, "probe-rescue count")
    require(abs(float(audit.get("combined_measurement_detection_ceiling")) - COMBINED_OBSERVABLE / FAULTS) < 1e-15, "combined ceiling")
    require(float(audit.get("frozen_detection_target")) == FROZEN_TARGET, "frozen target")
    require(audit.get("target_feasibility") == "NOT_MET", "target-feasibility disposition")
    require(audit.get("missing_duplicate_samples") == [0, 0], "sample integrity")
    require(audit.get("unknown_records") == 0 and audit.get("baseline_failures") == 0, "semantic integrity")
    require(audit.get("identity_fields_in_feature_npz") == 0, "identity leakage")
    require(audit.get("model_training_inference") == "0 / 0", "model-operation boundary")
    require(audit.get("repair_site_test") == "LOCKED / NOT ACCESSED", "repair-test boundary")
    require(audit.get("validation_access") == 0 and audit.get("holdout_access") == 0, "protected partition access")
    verify_record(audit["features"], FEATURES_2H, "12B-2H features")
    verify_record(audit["targets"], TARGETS_2H, "12B-2H targets")
    verify_record(audit["metrics"], METRICS_2H, "12B-2H metrics")
    verify_record(audit["manifest"], MANIFEST_2H, "12B-2H manifest")

    manifest = load_json(MANIFEST_2H)
    require(manifest.get("status") == "PASS", "12B-2H manifest status")
    outputs = manifest.get("outputs")
    require(isinstance(outputs, dict), "12B-2H outputs")
    for path in (FEATURES_2H, TARGETS_2H, METRICS_2H, SCHEMA_2H):
        verify_record(outputs[rel(path)], path, f"12B-2H {path.name}")
    metrics = load_json(METRICS_2H)
    require(metrics.get("status") == "FROZEN", "measurement metrics status")
    require(metrics.get("external_observable_faults") == EXTERNAL_OBSERVABLE, "metrics external count")
    require(metrics.get("combined_observable_faults") == COMBINED_OBSERVABLE, "metrics combined count")
    require(metrics.get("target_feasible_on_pilot_with_probes") is False, "metrics target feasibility")

    selection = load_json(AUDIT_2D)
    require(selection.get("status") == "PASS", "12B-2D status")
    require(selection.get("selection_status") == "FROZEN", "adaptive-vector selection state")
    require(abs(float(selection.get("selected_exact_site_rate")) - 0.41748046875) < 1e-12, "localization exact-site preservation")
    mean_max = selection.get("selected_mean_maximum_candidates")
    require(isinstance(mean_max, list) and len(mean_max) == 2, "localization candidate metrics")
    require(abs(float(mean_max[0]) - 6.0925) < 1e-4 and int(mean_max[1]) == 69, "localization candidate preservation")
    require(selection.get("full_repair_campaign") == "BLOCKED — PILOT ACCEPTANCE NOT MET", "repair campaign boundary")
    require(selection.get("model_training") == "NOT AUTHORIZED", "training boundary")

    ceiling_review = load_json(AUDIT_2E)
    discovery = load_json(AUDIT_2F)
    require(ceiling_review.get("status") == "PASS" and discovery.get("status") == "PASS", "measurement lineage status")
    require(ceiling_review.get("fault_selector_value_raw_as_features") == "PROHIBITED", "12B-2E leakage boundary")
    require(discovery.get("fault_selector_value_raw_as_features") == "PROHIBITED", "12B-2F leakage boundary")
    return audit, metrics, selection


def csv_bytes(rows: list[dict[str, Any]]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode()


def self_test() -> None:
    ceiling = COMBINED_OBSERVABLE / FAULTS
    require(abs(ceiling - 0.5029296875) < 1e-15, "combined-ceiling canary")
    require(COMBINED_OBSERVABLE - EXTERNAL_OBSERVABLE == 3, "probe-rescue canary")
    require(ceiling < FROZEN_TARGET, "target-disposition canary")
    print("Stage 12B-2I self-test: PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    require(ROOT.name == "vlsi_fault_detection_v2", f"run from project root, not {ROOT}")
    outputs = (DISPOSITION, REDESIGN, REGISTRY_CSV, REGISTRY_JSON, REPORT, MANIFEST, AUDIT)
    for path in outputs:
        require(not path.exists(), f"Stage {STAGE} output already exists: {rel(path)}")

    audit_2h, metrics_2h, selection = verify_inputs()
    external_ceiling = EXTERNAL_OBSERVABLE / FAULTS
    combined_ceiling = COMBINED_OBSERVABLE / FAULTS
    absolute_gain = combined_ceiling - external_ceiling
    relative_fault_gain = (COMBINED_OBSERVABLE - EXTERNAL_OBSERVABLE) / EXTERNAL_OBSERVABLE
    gap = FROZEN_TARGET - combined_ceiling

    disposition = {
        "policy_version": "CIRCUITSAGE-HMAC-V2.1-PROBE-MEASUREMENT-DISPOSITION-12B2I-v1",
        "stage": STAGE, "status": "FROZEN",
        "experiment": "FIXED 64-BIT TOPOLOGY-ONLY PROBE BANK WITH 96 ADAPTIVE VECTORS",
        "scientific_disposition": "MEASUREMENT TARGET NOT MET",
        "dataset_disposition": "VALID / FROZEN / RETAINED AS NEGATIVE RESULT",
        "external_observable_faults": EXTERNAL_OBSERVABLE,
        "combined_observable_faults": COMBINED_OBSERVABLE,
        "probe_rescued_faults": COMBINED_OBSERVABLE - EXTERNAL_OBSERVABLE,
        "external_detection_ceiling": external_ceiling,
        "combined_detection_ceiling": combined_ceiling,
        "absolute_detection_gain": absolute_gain,
        "relative_observable_fault_gain": relative_fault_gain,
        "frozen_detection_target": FROZEN_TARGET,
        "gap_to_target": gap,
        "current_measurement_target": "NOT_MET",
        "repair_model_training": "BLOCKED — OBSERVATION CEILING BELOW FROZEN TARGET",
        "full_repair_campaign": "BLOCKED",
        "adaptive_vector_localization_result": {
            "disposition": "PRESERVED FOR OBSERVABLE CLOSED-PILOT FAULTS",
            "exact_site_rate": float(selection["selected_exact_site_rate"]),
            "mean_candidate_sites": float(selection["selected_mean_maximum_candidates"][0]),
            "maximum_candidate_sites": int(selection["selected_mean_maximum_candidates"][1]),
        },
        "claim_boundary": "REPAIR_TRAIN PILOT MEASUREMENT FEASIBILITY ONLY; NOT MODEL ACCURACY OR GENERALIZATION",
        "v1_v2_core_v2_1_modified": [False, False, False],
        "repair_calibration": "LOCKED / NOT ACCESSED",
        "repair_site_test": "LOCKED / NOT ACCESSED",
        "original_dev_site_test": "CONSUMED / NOT REOPENED",
        "validation": "PROHIBITED", "holdout": "PROHIBITED",
    }
    redesign = {
        "contract_version": "CIRCUITSAGE-HMAC-V2.1-ENHANCED-MEASUREMENT-REDESIGN-12B2I-v1",
        "stage": STAGE, "status": "FROZEN",
        "authorization": "ARCHITECTURE AND BUDGET CONTRACT CREATION ONLY",
        "simulation_capture_training": "NOT AUTHORIZED / NOT AUTHORIZED / NOT AUTHORIZED",
        "problem_statement": "A single global 64-bit high-fanout probe bank rescued only 3 of 1021 externally invisible pilot faults",
        "required_redesign_families": [
            "MULTI-BANK TIME-MULTIPLEXED TOPOLOGY PROBES",
            "SEQUENTIAL-STATE CHECKPOINT AND TRANSITION SIGNATURES",
            "CONTROLLED SIMULATION-ONLY TEST-POINT OBSERVATION",
        ],
        "minimum_candidate_architectures": 3,
        "maximum_total_observation_bits_per_transaction": 256,
        "maximum_probe_banks": 4,
        "maximum_bits_per_bank": 64,
        "maximum_snapshots_per_transaction": 32,
        "selection_source": "GOLDEN GRAPH TOPOLOGY AND REPAIR_TRAIN RESPONSES ONLY",
        "selection_rule": "ONE GLOBAL FROZEN SCHEDULE PER CANDIDATE; NEVER CONDITIONED ON QUERY FAULT IDENTITY",
        "fault_identity_in_query": "PROHIBITED",
        "forbidden_features": [
            "fault_selector_i", "fault_enable_i", "fault_value_i", "fault_raw_o",
            "site_id", "fault_instance_id", "stuck_value",
        ],
        "required_pre_capture_analysis": [
            "OBSERVABILITY CONE COVERAGE",
            "SEQUENTIAL STATE DIVERSITY",
            "ESTIMATED SIMULATION COST",
            "IDENTITY-LEAKAGE REVIEW",
        ],
        "frozen_acceptance_targets": {
            "combined_all_injected_detection_recall_min": FROZEN_TARGET,
            "all_injected_exact_site_rate_min": 0.35,
            "mean_observable_candidate_sites_max": 50.0,
            "maximum_observable_candidate_sites_max": 500,
            "fault_free_false_alarm_rate_max": 0.0,
        },
        "partition_scope": "REPAIR_TRAIN FOR DESIGN; NEW REPAIR_CALIBRATION/REPAIR_SITE_TEST AUTHORIZATION REQUIRED LATER",
        "protected_boundaries": {
            "original_dev_site_test": "CONSUMED / REOPENING PROHIBITED",
            "validation": "PROHIBITED", "holdout": "PROHIBITED",
        },
        "frozen_rtl_and_canonical_netlists": "READ-ONLY; ONLY ISOLATED DERIVED OBSERVER COPIES MAY BE PROPOSED",
        "stop_rule": "DO NOT TRAIN IF THE MEASURED REPAIR_TRAIN OBSERVABILITY CEILING REMAINS BELOW 0.70",
    }
    registry_rows = [
        {"capability": "adaptive_vector_localization", "status": "PRESERVED", "value": "exact_site=0.41748047; mean_candidates=6.0925; max_candidates=69", "scope": "REPAIR_TRAIN PILOT / OBSERVABLE SUBSET"},
        {"capability": "external_io_detection", "status": "LIMITED", "value": f"{EXTERNAL_OBSERVABLE}/{FAULTS} ({external_ceiling:.8f})", "scope": "REPAIR_TRAIN PILOT"},
        {"capability": "fixed_64bit_probe_detection", "status": "NOT_MET", "value": f"{COMBINED_OBSERVABLE}/{FAULTS} ({combined_ceiling:.8f}); rescued=3", "scope": "REPAIR_TRAIN PILOT"},
        {"capability": "repair_model_training", "status": "BLOCKED", "value": "observation ceiling below 0.70", "scope": "ALL REPAIR PARTITIONS"},
        {"capability": "enhanced_measurement_redesign", "status": "CONTRACT_CREATION_AUTHORIZED", "value": "up to 4x64 probes and 32 snapshots; no capture yet", "scope": "ARCHITECTURE ONLY"},
        {"capability": "independent_generalization", "status": "NOT_ESTABLISHED", "value": "no unseen circuit or unseen fault family", "scope": "PROJECT"},
    ]
    registry_payload = csv_bytes(registry_rows)
    registry_json = {
        "registry_version": "CIRCUITSAGE-HMAC-V2.1-IMPROVEMENT-CAPABILITY-REGISTRY-12B2I-v1",
        "stage": STAGE, "status": "FROZEN", "capabilities": registry_rows,
    }
    report = f"""# CircuitSage-HMAC V2.1 Probe-Measurement Disposition

## Result

Stage 12B-2H produced a valid and complete probe-response dataset, but the
measurement objective was not met. External behavior exposed
{EXTERNAL_OBSERVABLE} of {FAULTS} pilot fault instances. Adding the fixed
64-bit probe bank exposed {COMBINED_OBSERVABLE} faults, an increase of only
{COMBINED_OBSERVABLE - EXTERNAL_OBSERVABLE} faults ({absolute_gain:.8f}
absolute recall).

The combined ceiling is {combined_ceiling:.8f}, below the frozen target of
{FROZEN_TARGET:.8f} by {gap:.8f}. This is a measurement limitation, not a
dataset-integrity or simulator failure.

## Disposition

- The Stage 12B-2H dataset is valid, frozen, and retained as a negative result.
- Training a repair detector or locator from this measurement system is blocked.
- The adaptive-vector localization gain remains preserved for observable pilot faults:
  exact-site rate 0.41748047, mean candidate set 6.0925, maximum 69.
- REPAIR_CALIBRATION, REPAIR_SITE_TEST, original DEV_SITE_TEST, VALIDATION and
  HOLDOUT remain unopened or locked under their existing contracts.

## Authorized follow-up

Only an enhanced-measurement architecture and budget contract may be created.
It must compare at least three identity-independent designs using multi-bank
topology probes, sequential-state checkpoints, or controlled simulation-only
test points. Capture and training still require separate authorization.

The 0.70 detection target is unchanged. If the redesigned measurement still
cannot reach that ceiling on REPAIR_TRAIN, model training must remain blocked.
"""

    frozen_write(DISPOSITION, canonical_json(disposition))
    frozen_write(REDESIGN, canonical_json(redesign))
    frozen_write(REGISTRY_CSV, registry_payload)
    frozen_write(REGISTRY_JSON, canonical_json(registry_json))
    frozen_write(REPORT, report.encode())
    primary = (DISPOSITION, REDESIGN, REGISTRY_CSV, REGISTRY_JSON, REPORT)
    manifest = {
        "manifest_version": "CIRCUITSAGE-HMAC-V2.1-PROBE-MEASUREMENT-DISPOSITION-MANIFEST-v1",
        "stage": STAGE, "status": "PASS", "stage_12b2h_audit": record(AUDIT_2H),
        "outputs": {rel(path): record(path) for path in primary},
        "simulation_calls": 0, "capture_calls": 0,
        "model_objects_deserialized": 0, "training_calls": 0, "inference_calls": 0,
        "repair_calibration_access": 0, "repair_site_test_access": 0,
        "original_dev_site_test_access": 0, "validation_access": 0, "holdout_access": 0,
        "frozen_rtl_modified": False, "canonical_netlists_modified": False,
    }
    frozen_write(MANIFEST, canonical_json(manifest))
    audit = {
        "audit_version": "CIRCUITSAGE-HMAC-V2.1-PROBE-MEASUREMENT-DISPOSITION-FREEZE-v1",
        "stage": STAGE, "status": "PASS", "disposition_status": "FROZEN",
        "probe_capture_dataset": "VALID / FROZEN",
        "measurement_target": "NOT_MET",
        "external_observable_faults": f"{EXTERNAL_OBSERVABLE}/{FAULTS}",
        "combined_observable_faults": f"{COMBINED_OBSERVABLE}/{FAULTS}",
        "probe_rescued_faults": COMBINED_OBSERVABLE - EXTERNAL_OBSERVABLE,
        "external_detection_ceiling": external_ceiling,
        "combined_detection_ceiling": combined_ceiling,
        "absolute_detection_gain": absolute_gain,
        "frozen_detection_target": FROZEN_TARGET, "gap_to_target": gap,
        "adaptive_vector_localization": "PRESERVED — EXACT 0.41748047 / MEAN 6.0925 / MAX 69",
        "repair_model_training": "BLOCKED / NOT AUTHORIZED",
        "full_repair_campaign": "BLOCKED",
        "enhanced_measurement_contract_creation": "AUTHORIZED",
        "enhanced_measurement_capture_training": "NOT AUTHORIZED / NOT AUTHORIZED",
        "repair_calibration": "LOCKED / NOT ACCESSED",
        "repair_site_test": "LOCKED / NOT ACCESSED",
        "original_dev_site_test": "CONSUMED / NOT REOPENED",
        "validation_access": 0, "holdout_access": 0,
        "v1_modified": False, "v2_core_modified": False, "v2_1_modified": False,
        "disposition": record(DISPOSITION), "redesign_contract": record(REDESIGN),
        "report": record(REPORT), "manifest": record(MANIFEST),
        "next_gate": "STAGE 12B-3A — ENHANCED MEASUREMENT ARCHITECTURE AND BUDGET CONTRACT FREEZE",
    }
    frozen_write(AUDIT, canonical_json(audit))

    require(registry_payload == REGISTRY_CSV.read_bytes(), "registry CSV replay")
    for path in (DISPOSITION, REDESIGN, REGISTRY_JSON, MANIFEST, AUDIT):
        require(canonical_json(load_json(path)) == path.read_bytes(), f"canonical JSON replay: {path.name}")
    for path, expected in PINNED.items():
        require(sha256(path) == expected, f"frozen input changed: {rel(path)}")

    print("\nSTAGE 12B-2I — PROBE-MEASUREMENT FEASIBILITY DISPOSITION FREEZE")
    print(f"{'Status':<61}: PASS")
    print(f"{'Disposition status':<61}: FROZEN")
    print(f"{'Probe dataset':<61}: VALID / FROZEN")
    print(f"{'External observable faults':<61}: {EXTERNAL_OBSERVABLE}/{FAULTS}")
    print(f"{'Combined observable faults':<61}: {COMBINED_OBSERVABLE}/{FAULTS}")
    print(f"{'Probe-rescued faults':<61}: {COMBINED_OBSERVABLE - EXTERNAL_OBSERVABLE}")
    print(f"{'External / combined ceiling':<61}: {external_ceiling:.8f} / {combined_ceiling:.8f}")
    print(f"{'Frozen target / gap':<61}: {FROZEN_TARGET:.8f} / {gap:.8f}")
    print(f"{'Measurement target':<61}: NOT_MET")
    print(f"{'Adaptive-vector localization':<61}: PRESERVED — EXACT 0.41748047 / MEAN 6.0925 / MAX 69")
    print(f"{'Repair model training / full campaign':<61}: BLOCKED / BLOCKED")
    print(f"{'Enhanced measurement contract creation':<61}: AUTHORIZED")
    print(f"{'Enhanced capture / training':<61}: NOT AUTHORIZED / NOT AUTHORIZED")
    print(f"{'REPAIR_CALIBRATION / REPAIR_SITE_TEST':<61}: LOCKED / LOCKED")
    print(f"{'Original DEV_SITE_TEST / VALIDATION / HOLDOUT':<61}: CONSUMED / 0 / 0")
    print(f"{'V1 / V2 Core / V2.1 modified':<61}: NO / NO / NO")
    print(f"{'Disposition':<61}: {DISPOSITION}")
    print(f"{'Disposition SHA':<61}: {sha256(DISPOSITION)}")
    print(f"{'Redesign contract':<61}: {REDESIGN}")
    print(f"{'Redesign contract SHA':<61}: {sha256(REDESIGN)}")
    print(f"{'Report':<61}: {REPORT}")
    print(f"{'Report SHA':<61}: {sha256(REPORT)}")
    print(f"{'Manifest':<61}: {MANIFEST}")
    print(f"{'Manifest SHA':<61}: {sha256(MANIFEST)}")
    print(f"{'Audit':<61}: {AUDIT}")
    print(f"{'Audit SHA':<61}: {sha256(AUDIT)}")
    print(f"{'Next gate':<61}: STAGE 12B-3A — ENHANCED MEASUREMENT ARCHITECTURE AND BUDGET CONTRACT FREEZE")


if __name__ == "__main__":
    main()
