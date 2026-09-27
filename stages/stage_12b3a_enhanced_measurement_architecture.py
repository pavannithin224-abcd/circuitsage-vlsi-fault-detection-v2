#!/usr/bin/env python3
"""Stage 12B-3A: enhanced measurement architecture and budget freeze.

Consumes the frozen Stage 12B-2I disposition and defines three bounded,
identity-independent measurement candidates, a staged screening plan, resource
budgets, and unchanged scientific acceptance criteria.  This stage performs no
probe discovery, simulation, capture, model operation, or protected-partition
access.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Any


STAGE = "12B-3A"
ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "config/v2_1_improvement"
RESULT_2 = ROOT / "results/circuitsage_hmac_v2_12b2"
RESULT = ROOT / "results/circuitsage_hmac_v2_12b3"
WORK = RESULT / "enhanced_measurement_architecture_12b3a"

SOURCE_2I = ROOT / "stage_12b2i_probe_measurement_disposition.py"
DISPOSITION_2I = CONFIG / "circuitsage_hmac_v2_1_probe_measurement_disposition_policy_12b2i.json"
REDESIGN_2I = CONFIG / "circuitsage_hmac_v2_1_enhanced_measurement_redesign_contract_12b2i.json"
REGISTRY_CSV_2I = RESULT_2 / "probe_measurement_disposition_12b2i/circuitsage_hmac_v2_1_improvement_capability_registry_12b2i.csv"
REGISTRY_JSON_2I = RESULT_2 / "probe_measurement_disposition_12b2i/circuitsage_hmac_v2_1_improvement_capability_registry_12b2i.json"
REPORT_2I = RESULT_2 / "probe_measurement_disposition_12b2i/circuitsage_hmac_v2_1_probe_measurement_disposition_report_12b2i.md"
MANIFEST_2I = RESULT_2 / "circuitsage_hmac_v2_1_probe_measurement_disposition_manifest_12b2i.json"
AUDIT_2I = RESULT_2 / "circuitsage_hmac_v2_1_probe_measurement_disposition_freeze_12b2i.json"
AUDIT_2H = RESULT_2 / "circuitsage_hmac_v2_1_probe_capture_dataset_integrity_freeze_12b2h.json"

ARCHITECTURE = CONFIG / "circuitsage_hmac_v2_1_enhanced_measurement_architecture_12b3a.json"
BUDGET = CONFIG / "circuitsage_hmac_v2_1_enhanced_measurement_budget_policy_12b3a.json"
ACCEPTANCE = CONFIG / "circuitsage_hmac_v2_1_enhanced_measurement_acceptance_contract_12b3a.json"
CANDIDATES = WORK / "circuitsage_hmac_v2_1_enhanced_measurement_candidate_grid_12b3a.csv"
SCHEDULE = WORK / "circuitsage_hmac_v2_1_enhanced_measurement_screening_schedule_12b3a.csv"
REPORT = WORK / "circuitsage_hmac_v2_1_enhanced_measurement_architecture_report_12b3a.md"
ENVIRONMENT = WORK / "circuitsage_hmac_v2_1_enhanced_measurement_environment_12b3a.json"
MANIFEST = RESULT / "circuitsage_hmac_v2_1_enhanced_measurement_architecture_manifest_12b3a.json"
AUDIT = RESULT / "circuitsage_hmac_v2_1_enhanced_measurement_architecture_budget_freeze_12b3a.json"

PINNED = {
    SOURCE_2I: "1686395024ef9d1f412430da74a167ed153720d6c5892ec66f01049f5619d68d",
    DISPOSITION_2I: "89b7a73279cf39b74572931090563b2009406c2ef8f45c91047357086664b17e",
    REDESIGN_2I: "89c478eaa01e884cebae087a4dde2eec9093643f67c265b0fb7956f9695f4e4f",
    REGISTRY_CSV_2I: "35148d4ad5f7ee9856e5901b19bc86be8d9d493f9d0e0581e681f7172ab5eda8",
    REGISTRY_JSON_2I: "2da0379ee24b0a128e70d7385c09bab756f91b4ed8af85479a074aa25ce44793",
    REPORT_2I: "948a443aa3ebc1b619d0732abda9e2d79ccbc899da4a0ffd4ba93058182e4d6f",
    MANIFEST_2I: "5cb064b8e248445e73741f0049380e7363024cabd539aa5f4d992f4261a3e476",
    AUDIT_2I: "355e71a08bf6f5e914ee8bb9be55155610c47ec89a3f1fa93be47e745eb1df57",
    AUDIT_2H: "cc68202cc1ef92d32319c43ca1d8afd2c9bb324ac24fb9125e630d50a7403d91",
}

FROZEN_TARGET = 0.70
PILOT_FAULTS = 2048
PILOT_SITES = 1024
SELECTED_VECTORS = 96
SCREEN_SITES = 256
SCREEN_FAULTS = 512
SCREEN_VECTORS = 48
SCREEN_TRANSACTIONS_PER_CANDIDATE = SCREEN_FAULTS * SCREEN_VECTORS
MAX_BANKS = 4
MAX_BITS_PER_BANK = 64
MAX_CONCURRENT_BITS = 256
MAX_SNAPSHOTS = 32
MAX_SCREEN_CANDIDATES = 3


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


def csv_payload(rows: list[dict[str, Any]]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode()


def verify_inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    print("STAGE 12B-3A — ENHANCED MEASUREMENT ARCHITECTURE AND BUDGET CONTRACT")
    print("FROZEN INPUT VERIFICATION")
    for path, expected in PINNED.items():
        require(path.is_file(), f"missing frozen input: {rel(path)}")
        require(sha256(path) == expected, f"frozen input SHA: {rel(path)}")
        print(f"  {path.name:<91}: OK", flush=True)
    audit = load_json(AUDIT_2I)
    disposition = load_json(DISPOSITION_2I)
    redesign = load_json(REDESIGN_2I)
    require(audit.get("status") == "PASS" and audit.get("disposition_status") == "FROZEN", "12B-2I disposition")
    require(audit.get("measurement_target") == "NOT_MET", "measurement result")
    require(audit.get("combined_observable_faults") == "1030/2048", "combined observability")
    require(abs(float(audit.get("combined_detection_ceiling")) - 0.5029296875) < 1e-15, "combined ceiling")
    require(float(audit.get("frozen_detection_target")) == FROZEN_TARGET, "frozen target")
    require(audit.get("repair_model_training") == "BLOCKED / NOT AUTHORIZED", "training boundary")
    require(audit.get("enhanced_measurement_contract_creation") == "AUTHORIZED", "architecture authorization")
    require(audit.get("enhanced_measurement_capture_training") == "NOT AUTHORIZED / NOT AUTHORIZED", "capture boundary")
    require(audit.get("repair_site_test") == "LOCKED / NOT ACCESSED", "repair-test boundary")
    require(audit.get("validation_access") == 0 and audit.get("holdout_access") == 0, "protected partitions")
    require(disposition.get("status") == "FROZEN", "disposition policy status")
    require(disposition.get("repair_model_training") == "BLOCKED — OBSERVATION CEILING BELOW FROZEN TARGET", "disposition training rule")
    require(redesign.get("status") == "FROZEN", "redesign contract status")
    require(redesign.get("authorization") == "ARCHITECTURE AND BUDGET CONTRACT CREATION ONLY", "redesign authorization")
    require(redesign.get("simulation_capture_training") == "NOT AUTHORIZED / NOT AUTHORIZED / NOT AUTHORIZED", "redesign execution boundary")
    require(redesign.get("minimum_candidate_architectures") == MAX_SCREEN_CANDIDATES, "candidate minimum")
    require(redesign.get("maximum_probe_banks") == MAX_BANKS, "probe-bank budget")
    require(redesign.get("maximum_bits_per_bank") == MAX_BITS_PER_BANK, "bits-per-bank budget")
    require(redesign.get("maximum_total_observation_bits_per_transaction") == MAX_CONCURRENT_BITS, "concurrent-bit budget")
    require(redesign.get("maximum_snapshots_per_transaction") == MAX_SNAPSHOTS, "snapshot budget")
    require(redesign.get("fault_identity_in_query") == "PROHIBITED", "identity boundary")
    return audit, redesign


def candidate_rows() -> list[dict[str, Any]]:
    rows = [
        {
            "candidate_id": "EM_TOPOLOGY_4X64_T24",
            "family": "MULTI_BANK_TOPOLOGY",
            "probe_banks": 4,
            "bits_per_bank": 64,
            "concurrent_observation_bits": 256,
            "snapshots": 24,
            "sampled_bits_per_transaction": 6144,
            "selection_inputs": "GOLDEN_GRAPH_TOPOLOGY",
            "probe_classes": "64_SEQ_STATE|64_CONTROL|64_HIGH_FANOUT|64_BOUNDARY",
            "state_transition_counts": "YES",
            "derived_netlist_only": "YES",
            "fault_conditioned": "NO",
            "screen_priority": 1,
        },
        {
            "candidate_id": "EM_STATE_CHECKPOINT_2X64_T32",
            "family": "SEQUENTIAL_STATE_CHECKPOINT",
            "probe_banks": 2,
            "bits_per_bank": 64,
            "concurrent_observation_bits": 128,
            "snapshots": 32,
            "sampled_bits_per_transaction": 4096,
            "selection_inputs": "GOLDEN_GRAPH_STATE_AND_CONTROL_TOPOLOGY",
            "probe_classes": "96_SEQ_STATE|32_CONTROL",
            "state_transition_counts": "YES",
            "derived_netlist_only": "YES",
            "fault_conditioned": "NO",
            "screen_priority": 2,
        },
        {
            "candidate_id": "EM_TESTPOINT_4X64_T16",
            "family": "SIMULATION_ONLY_TEST_POINT",
            "probe_banks": 4,
            "bits_per_bank": 64,
            "concurrent_observation_bits": 256,
            "snapshots": 16,
            "sampled_bits_per_transaction": 4096,
            "selection_inputs": "GOLDEN_GRAPH_PLUS_AGGREGATED_REPAIR_TRAIN_OBSERVABILITY",
            "probe_classes": "256_GLOBAL_COVERAGE_TEST_POINTS",
            "state_transition_counts": "YES",
            "derived_netlist_only": "YES",
            "fault_conditioned": "NO",
            "screen_priority": 3,
        },
    ]
    require(len(rows) == MAX_SCREEN_CANDIDATES, "candidate count")
    for row in rows:
        require(int(row["probe_banks"]) <= MAX_BANKS, "candidate bank budget")
        require(int(row["bits_per_bank"]) <= MAX_BITS_PER_BANK, "candidate bit budget")
        require(int(row["concurrent_observation_bits"]) <= MAX_CONCURRENT_BITS, "candidate concurrent budget")
        require(int(row["snapshots"]) <= MAX_SNAPSHOTS, "candidate snapshot budget")
        require(row["fault_conditioned"] == "NO", "candidate identity independence")
    return rows


def schedule_rows() -> list[dict[str, Any]]:
    return [
        {
            "phase": 1,
            "name": "STRUCTURAL_DISCOVERY_AND_LINT",
            "candidate_count": 3,
            "sites": 0,
            "fault_instances": 0,
            "vectors": 0,
            "enabled_transactions": 0,
            "authorization_after_12b3a": "NOT_YET_AUTHORIZED",
            "advancement_rule": "ALL STRUCTURAL, CONSISTENCY AND LEAKAGE CHECKS PASS",
        },
        {
            "phase": 2,
            "name": "BOUNDED_REPAIR_TRAIN_SCREEN",
            "candidate_count": 3,
            "sites": SCREEN_SITES,
            "fault_instances": SCREEN_FAULTS,
            "vectors": SCREEN_VECTORS,
            "enabled_transactions": SCREEN_TRANSACTIONS_PER_CANDIDATE * 3,
            "authorization_after_12b3a": "NOT_AUTHORIZED",
            "advancement_rule": "AT LEAST ONE CANDIDATE REACHES 0.70 COMBINED DETECTION WITH ZERO FALSE ALARMS",
        },
        {
            "phase": 3,
            "name": "SINGLE_WINNER_FULL_REPAIR_TRAIN_CAPTURE",
            "candidate_count": 1,
            "sites": PILOT_SITES,
            "fault_instances": PILOT_FAULTS,
            "vectors": SELECTED_VECTORS,
            "enabled_transactions": PILOT_FAULTS * SELECTED_VECTORS,
            "authorization_after_12b3a": "NOT_AUTHORIZED",
            "advancement_rule": "SCREEN WINNER ONLY; FULL CEILING >=0.70 BEFORE ANY MODEL TRAINING",
        },
    ]


def self_test() -> None:
    rows = candidate_rows()
    require({row["family"] for row in rows} == {
        "MULTI_BANK_TOPOLOGY", "SEQUENTIAL_STATE_CHECKPOINT", "SIMULATION_ONLY_TEST_POINT"
    }, "candidate-family coverage")
    require(SCREEN_TRANSACTIONS_PER_CANDIDATE == 24576, "screen transaction canary")
    require(sum(int(row["enabled_transactions"]) for row in schedule_rows()) == 270336, "staged transaction canary")
    print("Stage 12B-3A self-test: PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    require(ROOT.name == "vlsi_fault_detection_v2", f"run from project root, not {ROOT}")
    outputs = (ARCHITECTURE, BUDGET, ACCEPTANCE, CANDIDATES, SCHEDULE, REPORT, ENVIRONMENT, MANIFEST, AUDIT)
    for path in outputs:
        require(not path.exists(), f"Stage {STAGE} output already exists: {rel(path)}")

    audit_2i, redesign_2i = verify_inputs()
    candidates = candidate_rows()
    schedule = schedule_rows()
    candidate_payload = csv_payload(candidates)
    schedule_payload = csv_payload(schedule)
    architecture = {
        "architecture_version": "CIRCUITSAGE-HMAC-V2.1-ENHANCED-MEASUREMENT-12B3A-v1",
        "stage": STAGE, "status": "FROZEN",
        "scientific_scope": "CLOSED-CATALOG HMAC SINGLE PERSISTENT SA0/SA1 / REPAIR_TRAIN DESIGN ONLY",
        "problem": {
            "previous_external_observable_faults": 1027,
            "previous_combined_observable_faults": 1030,
            "previous_fault_instances": PILOT_FAULTS,
            "previous_combined_ceiling": 0.5029296875,
            "frozen_target": FROZEN_TARGET,
            "gap": 0.1970703125,
        },
        "candidate_count": len(candidates),
        "candidate_ids": [row["candidate_id"] for row in candidates],
        "candidate_grid_sha256": hashlib.sha256(candidate_payload).hexdigest(),
        "selection": "NO WINNER SELECTED; STRUCTURAL DISCOVERY AND BOUNDED SCREENING REQUIRED",
        "measurement_output": [
            "FIXED-CYCLE PROBE SNAPSHOTS XOR MATCHED GOLDEN",
            "PER-PROBE TRANSITION COUNTS AND GOLDEN DELTAS",
            "CONTROL TIMELINE DELTAS",
            "EXTERNAL DIGEST AND TIMEOUT DIFFERENCES",
        ],
        "global_schedule_rule": "EACH CANDIDATE USES ONE FROZEN GLOBAL PROBE/SNAPSHOT SCHEDULE FOR EVERY QUERY",
        "fault_identity_in_query": "PROHIBITED",
        "fault_conditioned_probe_selection": False,
        "offline_design_inputs": "GOLDEN GRAPH AND AGGREGATED REPAIR_TRAIN RESPONSES ONLY",
        "source_modification": "PROHIBITED; ISOLATED DERIVED OBSERVER NETLISTS ONLY",
        "training": "NOT AUTHORIZED",
    }
    budget = {
        "budget_version": "CIRCUITSAGE-HMAC-V2.1-ENHANCED-MEASUREMENT-BUDGET-12B3A-v1",
        "stage": STAGE, "status": "FROZEN",
        "maximum_candidates": MAX_SCREEN_CANDIDATES,
        "maximum_probe_banks_per_candidate": MAX_BANKS,
        "maximum_bits_per_bank": MAX_BITS_PER_BANK,
        "maximum_concurrent_observation_bits": MAX_CONCURRENT_BITS,
        "maximum_snapshots_per_transaction": MAX_SNAPSHOTS,
        "maximum_sampled_bits_per_transaction": MAX_CONCURRENT_BITS * MAX_SNAPSHOTS,
        "screening": {
            "sites": SCREEN_SITES, "fault_instances": SCREEN_FAULTS,
            "vectors": SCREEN_VECTORS,
            "transactions_per_candidate": SCREEN_TRANSACTIONS_PER_CANDIDATE,
            "maximum_total_enabled_transactions": SCREEN_TRANSACTIONS_PER_CANDIDATE * MAX_SCREEN_CANDIDATES,
        },
        "full_capture": {
            "maximum_advancing_candidates": 1,
            "sites": PILOT_SITES, "fault_instances": PILOT_FAULTS,
            "vectors": SELECTED_VECTORS,
            "enabled_transactions": PILOT_FAULTS * SELECTED_VECTORS,
        },
        "execution": "SEQUENTIAL", "parallel_batches": 1, "build_jobs": 1,
        "checkpoint_resume": "REQUIRED AFTER EACH CANONICAL BATCH",
        "minimum_free_disk_gib": 10,
        "maximum_intermediate_storage_gib": 20,
        "estimated_structural_discovery_cpu_hours": "0.5-3",
        "estimated_three_candidate_screen_cpu_hours": "2-12",
        "estimated_single_winner_full_capture_cpu_hours": "2-8",
        "cleanup_policy": "PRESERVE FROZEN OUTPUTS; BUILD DIRECTORIES MAY BE ARCHIVED AFTER HASH VERIFICATION",
    }
    acceptance = {
        "acceptance_version": "CIRCUITSAGE-HMAC-V2.1-ENHANCED-MEASUREMENT-ACCEPTANCE-12B3A-v1",
        "stage": STAGE, "status": "FROZEN",
        "structural_gate": {
            "all_probe_bits_present_in_45_batches": True,
            "yosys_structural_check": "PASS REQUIRED",
            "verilator_lint": "PASS REQUIRED",
            "fault_or_selector_named_probe_count": 0,
            "fault_identity_conditioning": False,
            "canonical_source_modifications": 0,
        },
        "screening_gate": {
            "combined_all_injected_detection_recall_min": FROZEN_TARGET,
            "fault_free_false_alarm_rate_max": 0.0,
            "unknown_or_missing_samples": 0,
            "minimum_site_bootstrap_replicates": 1000,
            "advancing_candidates_max": 1,
        },
        "full_repair_train_gate": {
            "combined_all_injected_detection_recall_min": FROZEN_TARGET,
            "all_injected_exact_site_rate_min": 0.35,
            "mean_observable_candidate_sites_max": 50.0,
            "maximum_observable_candidate_sites_max": 500,
            "fault_free_false_alarm_rate_max": 0.0,
        },
        "training_authorization_rule": "BLOCKED UNTIL FULL REPAIR_TRAIN MEASUREMENT GATE PASSES AND A SEPARATE TRAINING CONTRACT IS FROZEN",
        "failure_disposition": "STOP, FREEZE NEGATIVE RESULT, AND DO NOT LOWER TARGETS",
        "threshold_changes": "PROHIBITED AFTER SCREENING BEGINS",
    }
    environment = {
        "environment_version": "CIRCUITSAGE-HMAC-V2.1-ENHANCED-MEASUREMENT-ENVIRONMENT-12B3A-v1",
        "stage": STAGE, "status": "RECORDED",
        "python": sys.version.split()[0], "platform": platform.platform(),
        "yosys_available": shutil.which("yosys") is not None,
        "verilator_available": shutil.which("verilator") is not None,
        "architecture_calls": 1, "simulation_calls": 0, "capture_calls": 0,
        "training_calls": 0, "inference_calls": 0,
    }
    report = f"""# CircuitSage-HMAC V2.1 Enhanced Measurement Architecture

## Why this stage exists

The previous fixed 64-bit probe bank raised pilot observability from 1027 to
1030 of {PILOT_FAULTS} faults. Its {1030 / PILOT_FAULTS:.8f} ceiling remained
well below the frozen {FROZEN_TARGET:.8f} target. Training was therefore
correctly blocked.

## Frozen candidates

1. **EM_TOPOLOGY_4X64_T24** — four topology-stratified banks covering state,
   control, high-fanout logic and boundary logic.
2. **EM_STATE_CHECKPOINT_2X64_T32** — dense sequential-state and control
   checkpoints with the longest temporal schedule.
3. **EM_TESTPOINT_4X64_T16** — simulation-only global test points selected from
   topology plus aggregated REPAIR_TRAIN observability evidence.

Every candidate uses a single global schedule. The query never includes a
fault selector, fault value, site identity or fault-instance identity.

## Cost control

All candidates must first pass structural discovery and lint. A later,
separately authorized screen is limited to {SCREEN_SITES} sites,
{SCREEN_FAULTS} faults and {SCREEN_VECTORS} vectors. At most one candidate may
advance to a full {PILOT_FAULTS}-fault REPAIR_TRAIN capture.

No candidate may advance unless it reaches 0.70 combined detection with zero
fault-free false alarms. The target will not be lowered to force progression.

## Boundaries

This stage authorizes no discovery, simulation, capture or training. The
REPAIR_CALIBRATION and REPAIR_SITE_TEST partitions remain locked. The consumed
original DEV_SITE_TEST cannot be reopened, and VALIDATION/HOLDOUT remain
prohibited.
"""

    frozen_write(ARCHITECTURE, canonical_json(architecture))
    frozen_write(BUDGET, canonical_json(budget))
    frozen_write(ACCEPTANCE, canonical_json(acceptance))
    frozen_write(CANDIDATES, candidate_payload)
    frozen_write(SCHEDULE, schedule_payload)
    frozen_write(REPORT, report.encode())
    frozen_write(ENVIRONMENT, canonical_json(environment))
    primary = (ARCHITECTURE, BUDGET, ACCEPTANCE, CANDIDATES, SCHEDULE, REPORT, ENVIRONMENT)
    manifest = {
        "manifest_version": "CIRCUITSAGE-HMAC-V2.1-ENHANCED-MEASUREMENT-ARCHITECTURE-MANIFEST-v1",
        "stage": STAGE, "status": "PASS", "stage_12b2i_audit": record(AUDIT_2I),
        "outputs": {rel(path): record(path) for path in primary},
        "candidate_architectures": len(candidates),
        "probe_discovery_calls": 0, "simulation_calls": 0, "capture_calls": 0,
        "model_objects_deserialized": 0, "training_calls": 0, "inference_calls": 0,
        "repair_calibration_access": 0, "repair_site_test_access": 0,
        "original_dev_site_test_access": 0, "validation_access": 0, "holdout_access": 0,
        "frozen_rtl_modified": False, "canonical_netlists_modified": False,
    }
    frozen_write(MANIFEST, canonical_json(manifest))
    audit = {
        "audit_version": "CIRCUITSAGE-HMAC-V2.1-ENHANCED-MEASUREMENT-ARCHITECTURE-BUDGET-FREEZE-v1",
        "stage": STAGE, "status": "PASS",
        "architecture_status": "FROZEN", "budget_status": "FROZEN",
        "acceptance_status": "FROZEN",
        "candidate_architectures": len(candidates),
        "candidate_families": [row["family"] for row in candidates],
        "maximum_banks_bits_snapshots": [MAX_BANKS, MAX_BITS_PER_BANK, MAX_SNAPSHOTS],
        "maximum_concurrent_observation_bits": MAX_CONCURRENT_BITS,
        "bounded_screen_sites_faults_vectors": [SCREEN_SITES, SCREEN_FAULTS, SCREEN_VECTORS],
        "screen_enabled_transactions_max": SCREEN_TRANSACTIONS_PER_CANDIDATE * MAX_SCREEN_CANDIDATES,
        "frozen_detection_target": FROZEN_TARGET,
        "fault_identity_in_query": "PROHIBITED",
        "probe_discovery": "NOT YET AUTHORIZED",
        "simulation_capture_training": "0 / 0 / 0 — NOT AUTHORIZED",
        "repair_calibration": "LOCKED / NOT ACCESSED",
        "repair_site_test": "LOCKED / NOT ACCESSED",
        "original_dev_site_test": "CONSUMED / NOT REOPENED",
        "validation_access": 0, "holdout_access": 0,
        "v1_modified": False, "v2_core_modified": False, "v2_1_modified": False,
        "architecture": record(ARCHITECTURE), "budget": record(BUDGET),
        "acceptance": record(ACCEPTANCE), "candidate_grid": record(CANDIDATES),
        "report": record(REPORT), "manifest": record(MANIFEST),
        "next_gate": "STAGE 12B-3B — ENHANCED PROBE-BANK DISCOVERY AND STRUCTURAL FEASIBILITY FREEZE",
    }
    frozen_write(AUDIT, canonical_json(audit))

    require(candidate_payload == CANDIDATES.read_bytes(), "candidate-grid replay")
    require(schedule_payload == SCHEDULE.read_bytes(), "screening-schedule replay")
    for path in (ARCHITECTURE, BUDGET, ACCEPTANCE, ENVIRONMENT, MANIFEST, AUDIT):
        require(canonical_json(load_json(path)) == path.read_bytes(), f"canonical JSON replay: {path.name}")
    for path, expected in PINNED.items():
        require(sha256(path) == expected, f"frozen input changed: {rel(path)}")

    print("\nSTAGE 12B-3A — ENHANCED MEASUREMENT ARCHITECTURE AND BUDGET CONTRACT FREEZE")
    print(f"{'Status':<62}: PASS")
    print(f"{'Architecture / budget / acceptance':<62}: FROZEN / FROZEN / FROZEN")
    print(f"{'Candidate architectures':<62}: {len(candidates)}")
    print(f"{'Candidate families':<62}: MULTI-BANK / STATE CHECKPOINT / TEST-POINT")
    print(f"{'Maximum banks / bits per bank / snapshots':<62}: {MAX_BANKS} / {MAX_BITS_PER_BANK} / {MAX_SNAPSHOTS}")
    print(f"{'Maximum concurrent observation bits':<62}: {MAX_CONCURRENT_BITS}")
    print(f"{'Bounded screen sites / faults / vectors':<62}: {SCREEN_SITES} / {SCREEN_FAULTS} / {SCREEN_VECTORS}")
    print(f"{'Maximum screen enabled transactions':<62}: {SCREEN_TRANSACTIONS_PER_CANDIDATE * MAX_SCREEN_CANDIDATES}")
    print(f"{'Frozen combined-detection target':<62}: {FROZEN_TARGET:.8f}")
    print(f"{'Fault identity in query':<62}: PROHIBITED")
    print(f"{'Probe discovery':<62}: NOT YET AUTHORIZED")
    print(f"{'Simulation / capture / training':<62}: 0 / 0 / 0 — NOT AUTHORIZED")
    print(f"{'REPAIR_CALIBRATION / REPAIR_SITE_TEST':<62}: LOCKED / LOCKED")
    print(f"{'Original DEV_SITE_TEST / VALIDATION / HOLDOUT':<62}: CONSUMED / 0 / 0")
    print(f"{'V1 / V2 Core / V2.1 modified':<62}: NO / NO / NO")
    print(f"{'Architecture':<62}: {ARCHITECTURE}")
    print(f"{'Architecture SHA':<62}: {sha256(ARCHITECTURE)}")
    print(f"{'Budget':<62}: {BUDGET}")
    print(f"{'Budget SHA':<62}: {sha256(BUDGET)}")
    print(f"{'Acceptance':<62}: {ACCEPTANCE}")
    print(f"{'Acceptance SHA':<62}: {sha256(ACCEPTANCE)}")
    print(f"{'Candidate grid':<62}: {CANDIDATES}")
    print(f"{'Candidate grid SHA':<62}: {sha256(CANDIDATES)}")
    print(f"{'Manifest':<62}: {MANIFEST}")
    print(f"{'Manifest SHA':<62}: {sha256(MANIFEST)}")
    print(f"{'Audit':<62}: {AUDIT}")
    print(f"{'Audit SHA':<62}: {sha256(AUDIT)}")
    print(f"{'Next gate':<62}: STAGE 12B-3B — ENHANCED PROBE-BANK DISCOVERY AND STRUCTURAL FEASIBILITY FREEZE")


if __name__ == "__main__":
    main()
