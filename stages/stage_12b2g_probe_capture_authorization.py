#!/usr/bin/env python3
"""Stage 12B-2G: bounded probe-capture execution authorization freeze.

Verifies the frozen vector selection and global probe bank, creates an exact
REPAIR_TRAIN-only execution plan, and authorizes capture of fixed-cycle probe
snapshots for feasibility analysis. No simulation, model operation, protected
partition access, RTL edit, or canonical-netlist edit is performed.
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

try:
    import numpy as np
except ImportError as error:
    raise SystemExit("STOP: activate the project .venv (NumPy required)") from error


STAGE = "12B-2G"
ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "config/v2_1_improvement"
RESULT = ROOT / "results/circuitsage_hmac_v2_12b2"
WORK = RESULT / "probe_capture_authorization_12b2g"

SOURCE_2F = ROOT / "stage_12b2f_probe_discovery_feasibility.py"
PROBE_DIR = RESULT / "probe_discovery_12b2f"
PROBE_BANK_CSV = PROBE_DIR / "circuitsage_hmac_v2_1_global_probe_bank_12b2f.csv"
PROBE_BANK_JSON = PROBE_DIR / "circuitsage_hmac_v2_1_global_probe_bank_12b2f.json"
CONSISTENCY = PROBE_DIR / "circuitsage_hmac_v2_1_cross_batch_probe_consistency_12b2f.json"
CANARY_DIR = PROBE_DIR / "observer_canary"
CANARY_JSON = CANARY_DIR / "opentitan_hmac_sha256_msg32_faultbatch000_probe_canary.json"
CANARY_VERILOG = CANARY_DIR / "opentitan_hmac_sha256_msg32_faultbatch000_probe_canary.v"
CANARY_TB = CANARY_DIR / "tb_v21_probe_canary.sv"
YOSYS_LOG = CANARY_DIR / "yosys_observer_canary.log"
VERILATOR_LOG = CANARY_DIR / "verilator_observer_lint.log"
CAPTURE_CONTRACT_2F = CONFIG / "circuitsage_hmac_v2_1_probe_capture_contract_12b2f.json"
MANIFEST_2F = RESULT / "circuitsage_hmac_v2_1_probe_discovery_manifest_12b2f.json"
AUDIT_2F = RESULT / "circuitsage_hmac_v2_1_probe_discovery_instrumentation_feasibility_freeze_12b2f.json"

SELECTED_DIR = RESULT / "adaptive_vector_selection_12b2d"
SELECTED_VECTORS = SELECTED_DIR / "circuitsage_hmac_v2_1_selected_adaptive_vectors_12b2d.npz"
SELECTION_LOCK = SELECTED_DIR / "circuitsage_hmac_v2_1_adaptive_vector_selection_lock_12b2d.json"
PILOT_SITES = RESULT / "adaptive_vector_pool_12b2b/circuitsage_hmac_v2_1_pilot_screening_sites_12b2b.csv"
ACCEPTANCE_2E = CONFIG / "circuitsage_hmac_v2_1_instrumentation_feasibility_acceptance_12b2e.json"

EXECUTION_PLAN = WORK / "circuitsage_hmac_v2_1_probe_capture_execution_plan_12b2g.csv"
PREFLIGHT = WORK / "circuitsage_hmac_v2_1_probe_capture_preflight_12b2g.json"
EXECUTION_CONTRACT = CONFIG / "circuitsage_hmac_v2_1_probe_capture_execution_contract_12b2g.json"
AUTHORIZATION = CONFIG / "circuitsage_hmac_v2_1_probe_capture_execution_authorization_12b2g.json"
MANIFEST = RESULT / "circuitsage_hmac_v2_1_probe_capture_authorization_manifest_12b2g.json"
AUDIT = RESULT / "circuitsage_hmac_v2_1_probe_capture_authorization_freeze_12b2g.json"

PINNED = {
    SOURCE_2F: "b0a3dbd5a8c906c749d8f23c9af4923f72c08df8331dcda6eaaccd8adc9582a8",
    PROBE_BANK_CSV: "e6abe61682aaaccbaea8bbd6d90f1036e1bac8835a5999590bc545c396d5ab51",
    PROBE_BANK_JSON: "23cba70a6fa199bb59c62b84ad2546603903abc68cd9d5e7eb7e2e15e5f95632",
    CONSISTENCY: "9d9b349ed7f615867a24e911676d626dc6c612774457a6d2b8b5422ea14d794c",
    CANARY_JSON: "c8266b157bc890b77badda02d717bf9ba848c501480a6db6a1edff32607382de",
    CANARY_VERILOG: "36c1bb4c427c568a2fb81ef6459c5e9792ad1b2b52ee5b36cd87a379b21e512a",
    CANARY_TB: "d47d7f8cbc5e58af34ab260921845ff66818f57bdb2150c4d5f8e04da2570d2c",
    YOSYS_LOG: "0b6a5d2d1fb1466250812095459a78ca776d1e3f17a18f77f08b315bfb39cc52",
    VERILATOR_LOG: "89b8c503f4b86d4f009b8cd6d59732868ffbb47ca571b03dc405ebce492ba44a",
    CAPTURE_CONTRACT_2F: "7b1977406c2208b0287653ae5395330945963dc2388bc649586e9501fbed8919",
    MANIFEST_2F: "801b647764cd0fac7c0995ff1fbe36aa5bc9dd428b9c8b7b4ca0e126c460db29",
    AUDIT_2F: "f1103b9e2f07e1e2b6bffe2f65fff7c92c357084e511c5358751313eafb71188",
    SELECTED_VECTORS: "be9df0a3a70e61b54fc793439328bbcc307b9373fd881b0e97d500919ec27bff",
    SELECTION_LOCK: "963fa242559e6681b8c3f58d3de103f1b638bc4476425c7a80776c6b435597b1",
    PILOT_SITES: "87eceaf80a800cf60b8a7e3cb20e0ec3f934158dc35ae3d10f73e8cf9d25313c",
    ACCEPTANCE_2E: "824e3a65ad38a63c3b5a59219ad938976ee3c73b72b2ef4f6095416e1edc0483",
}

PROBES = 64
VECTORS = 96
PILOT_SITES_COUNT = 1024
FAULTS = 2048
SNAPSHOTS = 16
BATCHES = 45
ENABLED_TRANSACTIONS = FAULTS * VECTORS
SAMPLE_CYCLES = [0, 1, 2, 4, 8, 16, 32, 64, 96, 128, 160, 192, 224, 256, 320]
FINAL_EVENT = "DONE_OR_TIMEOUT"
MIN_FREE_GIB = 5


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


def verify_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    print("STAGE 12B-2G — BOUNDED PROBE-CAPTURE EXECUTION AUTHORIZATION")
    print("FROZEN INPUT VERIFICATION")
    for path, expected in PINNED.items():
        require(path.is_file(), f"missing frozen input: {rel(path)}")
        require(sha256(path) == expected, f"frozen input SHA: {rel(path)}")
        print(f"  {path.name:<91}: OK", flush=True)
    audit = load_json(AUDIT_2F)
    capture = load_json(CAPTURE_CONTRACT_2F)
    selection = load_json(SELECTION_LOCK)
    acceptance = load_json(ACCEPTANCE_2E)
    require(audit.get("status") == "PASS", "12B-2F audit status")
    require(audit.get("probe_discovery_status") == "FROZEN / COMPLETE", "probe discovery state")
    require(audit.get("cross_batch_consistency") == "PASS / 45/45", "cross-batch probe consistency")
    require(audit.get("observer_canary") == "YOSYS PASS / VERILATOR LINT PASS / NOT EXECUTED", "observer canary")
    require(audit.get("probe_capture") == "NOT AUTHORIZED", "prior capture boundary")
    require(audit.get("fault_selector_value_raw_as_features") == "PROHIBITED", "leakage boundary")
    require(audit.get("repair_site_test") == "LOCKED / NOT AUTHORIZED", "repair test boundary")
    require(audit.get("validation_access") == 0 and audit.get("holdout_access") == 0, "protected partition access")
    require(capture.get("status") == "FROZEN" and capture.get("capture_authorization") == "NOT YET AUTHORIZED", "capture contract state")
    require(capture.get("probe_bank_width") == PROBES, "probe width")
    require(capture.get("allowed_future_scope") == "REPAIR_TRAIN PILOT SITES ONLY", "capture scope")
    require(capture.get("next_stage_requires_separate_authorization") is True, "separate authorization rule")
    require(selection.get("status") == "FROZEN", "vector-selection lock")
    require(selection.get("selected_vectors") == VECTORS, "selected-vector count")
    require(selection.get("selection_partition") == "REPAIR_TRAIN PILOT ONLY", "vector-selection partition")
    require(acceptance.get("status") == "FROZEN", "feasibility acceptance status")
    require(shutil.which("yosys") is not None and shutil.which("verilator") is not None, "toolchain unavailable")
    return audit, capture, acceptance


def load_sites() -> list[dict[str, str]]:
    with PILOT_SITES.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    require(len(rows) == PILOT_SITES_COUNT, "pilot-site count")
    require([int(row["pilot_rank"]) for row in rows] == list(range(PILOT_SITES_COUNT)), "pilot-site ordering")
    require(all(row["fault_polarities"] == "SA0|SA1" for row in rows), "pilot fault polarity")
    return rows


def verify_vectors_and_probes() -> tuple[list[int], list[dict[str, str]]]:
    with np.load(SELECTED_VECTORS, allow_pickle=False) as archive:
        required = {"key_u8", "message_u8", "selection_rank", "selection_round", "vector_index"}
        require(set(archive.files) == required, "selected-vector NPZ members")
        indices = np.asarray(archive["vector_index"], dtype=np.int32)
        require(indices.shape == (VECTORS,), "selected-vector shape")
        require(len(set(indices.tolist())) == VECTORS, "selected-vector uniqueness")
        require(np.asarray(archive["key_u8"]).shape == (VECTORS, 32), "selected key shape")
        require(np.asarray(archive["message_u8"]).shape == (VECTORS, 32), "selected message shape")
    with PROBE_BANK_CSV.open(newline="", encoding="utf-8") as stream:
        probes = list(csv.DictReader(stream))
    require(len(probes) == PROBES, "probe-bank count")
    require([int(row["probe_bit"]) for row in probes] == list(range(PROBES)), "probe-bank ordering")
    require(len({row["site_id"] for row in probes}) == PROBES, "probe-site uniqueness")
    require(len({int(row["bit_id"]) for row in probes}) == PROBES, "probe-bit uniqueness")
    return indices.tolist(), probes


def execution_plan_bytes(sites: list[dict[str, str]]) -> tuple[bytes, list[dict[str, Any]]]:
    counts = {batch_id: 0 for batch_id in range(BATCHES)}
    for row in sites:
        site_index = int(row["site_index"])
        require(1 <= site_index <= 22839, "pilot site-index range")
        batch_id = (site_index - 1) // 512
        require(0 <= batch_id < BATCHES, "derived batch ID")
        counts[batch_id] += 1
    require(sum(counts.values()) == PILOT_SITES_COUNT, "execution-plan site total")
    rows: list[dict[str, Any]] = []
    for batch_id, site_count in counts.items():
        if not site_count:
            continue
        faults = site_count * 2
        baselines = VECTORS
        enabled = faults * VECTORS
        rows.append({
            "execution_order": len(rows), "batch_id": batch_id,
            "pilot_sites": site_count, "fault_instances": faults,
            "selected_vectors": VECTORS, "baseline_transactions": baselines,
            "enabled_transactions": enabled, "total_transactions": baselines + enabled,
            "probe_bits": PROBES, "snapshots_per_transaction": SNAPSHOTS,
            "enabled_probe_values": enabled * PROBES * SNAPSHOTS,
        })
    require(len(rows) == BATCHES, "all canonical batches must be represented")
    require(sum(row["fault_instances"] for row in rows) == FAULTS, "execution-plan fault total")
    require(sum(row["enabled_transactions"] for row in rows) == ENABLED_TRANSACTIONS, "execution-plan transaction total")
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader(); writer.writerows(rows)
    return output.getvalue().encode(), rows


def self_test() -> None:
    require(len(SAMPLE_CYCLES) + 1 == SNAPSHOTS, "snapshot schedule canary")
    require(SAMPLE_CYCLES == sorted(set(SAMPLE_CYCLES)), "sample-cycle ordering")
    require(ENABLED_TRANSACTIONS == 196608, "transaction-count canary")
    print("Stage 12B-2G self-test: PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    require(ROOT.name == "vlsi_fault_detection_v2", f"run from project root, not {ROOT}")
    outputs = (EXECUTION_PLAN, PREFLIGHT, EXECUTION_CONTRACT, AUTHORIZATION, MANIFEST, AUDIT)
    for path in outputs:
        require(not path.exists(), f"Stage {STAGE} output already exists: {rel(path)}")

    audit_2f, capture_2f, acceptance = verify_inputs()
    sites = load_sites()
    vector_indices, probes = verify_vectors_and_probes()
    plan_payload, plan_rows = execution_plan_bytes(sites)
    free_bytes = shutil.disk_usage(ROOT).free
    require(free_bytes >= MIN_FREE_GIB * 1024**3, f"less than {MIN_FREE_GIB} GiB free disk")
    raw_snapshot_bytes = ENABLED_TRANSACTIONS * SNAPSHOTS * 8
    raw_toggle_bytes = ENABLED_TRANSACTIONS * PROBES * 2
    estimated_binary_bytes = raw_snapshot_bytes + raw_toggle_bytes
    preflight = {
        "preflight_version": "CIRCUITSAGE-HMAC-V2.1-PROBE-CAPTURE-PREFLIGHT-v1",
        "stage": STAGE, "status": "PASS",
        "canonical_batches": len(plan_rows), "pilot_sites": len(sites),
        "fault_instances": FAULTS, "selected_vectors": len(vector_indices),
        "probe_bits": len(probes), "snapshots_per_transaction": SNAPSHOTS,
        "enabled_transactions": ENABLED_TRANSACTIONS,
        "available_disk_bytes": free_bytes,
        "minimum_free_disk_bytes": MIN_FREE_GIB * 1024**3,
        "estimated_uncompressed_snapshot_bytes": raw_snapshot_bytes,
        "estimated_uncompressed_toggle_bytes": raw_toggle_bytes,
        "estimated_core_binary_payload_bytes": estimated_binary_bytes,
        "toolchain": {"python": sys.version.split()[0], "numpy": np.__version__, "platform": platform.platform()},
        "frozen_input_verification": "PASS",
        "capture_or_simulation_performed": False,
    }
    execution_contract = {
        "contract_version": "CIRCUITSAGE-HMAC-V2.1-BOUNDED-PROBE-CAPTURE-12B2G-v1",
        "stage": STAGE, "status": "FROZEN",
        "scope": "REPAIR_TRAIN PILOT ONLY",
        "canonical_batches": BATCHES, "pilot_sites": PILOT_SITES_COUNT,
        "fault_instances": FAULTS, "selected_vectors": VECTORS,
        "probe_bits": PROBES, "enabled_transactions": ENABLED_TRANSACTIONS,
        "sample_cycles_after_start": SAMPLE_CYCLES,
        "final_sample_event": FINAL_EVENT,
        "snapshots_per_transaction": SNAPSHOTS,
        "control_timeline": "BUSY/DONE TRANSITION CYCLES AND COMPLETION LATENCY",
        "probe_snapshot": "64-BIT EXACT VALUE; DATASET STORES XOR AGAINST MATCHED GOLDEN",
        "toggle_measurement": "PER-PROBE TRANSITION COUNT, UINT16 SATURATING",
        "execution": "SEQUENTIAL", "parallel_batches": 1, "build_jobs": 1,
        "checkpoint_resume": "REQUIRED AFTER EACH CANONICAL BATCH",
        "dataset_format": "DETERMINISTIC PER-BATCH NPZ PLUS CONSOLIDATED NPZ",
        "fault_free_baseline": "ONE GOLDEN TRACE PER SELECTED VECTOR PER CANONICAL BATCH",
        "identity_independence": "SAME PROBES, SAMPLE EVENTS AND VECTORS FOR EVERY TRANSACTION",
        "forbidden_dataset_features": [
            "fault_selector_i", "fault_enable_i", "fault_value_i", "fault_raw_o",
            "site_id as model input", "fault_instance_id as model input",
        ],
        "fault_identity_retention": "SEPARATE TARGET/METADATA ONLY FOR SCORING; NEVER INPUT FEATURE",
        "failure_policy": "STOP ON BASELINE FAILURE, UNKNOWN VALUE, MISSING SAMPLE, DUPLICATE SAMPLE OR IDENTITY-LEAKAGE VIOLATION",
    }
    authorization = {
        "authorization_version": "CIRCUITSAGE-HMAC-V2.1-PROBE-CAPTURE-AUTHORIZATION-12B2G-v1",
        "stage": STAGE, "status": "FROZEN",
        "bounded_probe_capture": "AUTHORIZED / NOT STARTED",
        "authorized_plan": record_placeholder(EXECUTION_PLAN, plan_payload),
        "selected_vectors": record(SELECTED_VECTORS),
        "global_probe_bank": record(PROBE_BANK_JSON),
        "pilot_sites": record(PILOT_SITES),
        "full_repair_campaign": "NOT AUTHORIZED",
        "adaptive_probe_or_vector_reselection": "PROHIBITED",
        "model_training": "NOT AUTHORIZED",
        "repair_calibration": "LOCKED / NOT AUTHORIZED",
        "repair_site_test": "LOCKED / NOT AUTHORIZED",
        "original_dev_site_test": "CONSUMED / REOPENING PROHIBITED",
        "validation": "PROHIBITED", "holdout": "PROHIBITED",
        "rtl_or_canonical_netlist_modification": "PROHIBITED",
        "derived_observer_netlists": "AUTHORIZED IN ISOLATED BUILD DIRECTORY",
    }
    frozen_write(EXECUTION_PLAN, plan_payload)
    frozen_write(PREFLIGHT, canonical_json(preflight))
    frozen_write(EXECUTION_CONTRACT, canonical_json(execution_contract))
    frozen_write(AUTHORIZATION, canonical_json(authorization))
    primary = (EXECUTION_PLAN, PREFLIGHT, EXECUTION_CONTRACT, AUTHORIZATION)
    manifest = {
        "manifest_version": "CIRCUITSAGE-HMAC-V2.1-PROBE-CAPTURE-AUTHORIZATION-MANIFEST-v1",
        "stage": STAGE, "status": "PASS", "stage_12b2f_audit": record(AUDIT_2F),
        "outputs": {rel(path): record(path) for path in primary},
        "canonical_batches": BATCHES, "pilot_sites": PILOT_SITES_COUNT,
        "fault_instances": FAULTS, "selected_vectors": VECTORS,
        "probe_bits": PROBES, "snapshots_per_transaction": SNAPSHOTS,
        "enabled_transactions": ENABLED_TRANSACTIONS,
        "simulation_calls": 0, "probe_values_captured": 0,
        "model_objects_deserialized": 0, "training_calls": 0, "inference_calls": 0,
        "repair_calibration_access": 0, "repair_site_test_access": 0,
        "original_dev_site_test_access": 0, "validation_access": 0, "holdout_access": 0,
        "frozen_rtl_modified": False, "canonical_netlists_modified": False,
    }
    frozen_write(MANIFEST, canonical_json(manifest))
    audit = {
        "audit_version": "CIRCUITSAGE-HMAC-V2.1-PROBE-CAPTURE-AUTHORIZATION-FREEZE-v1",
        "stage": STAGE, "status": "PASS",
        "authorization_status": "FROZEN",
        "bounded_probe_capture": "AUTHORIZED / NOT STARTED",
        "execution_plan": f"{BATCHES} BATCHES / {PILOT_SITES_COUNT} SITES / {FAULTS} FAULTS",
        "selected_vectors_probe_bits_snapshots": [VECTORS, PROBES, SNAPSHOTS],
        "enabled_transactions": ENABLED_TRANSACTIONS,
        "identity_independent_schedule": "FROZEN / PASS",
        "fault_selector_value_raw_as_features": "PROHIBITED",
        "full_repair_campaign": "NOT AUTHORIZED",
        "model_training": "NOT AUTHORIZED",
        "repair_calibration": "LOCKED / NOT AUTHORIZED",
        "repair_site_test": "LOCKED / NOT AUTHORIZED",
        "original_dev_site_test": "CONSUMED / NOT REOPENED",
        "validation_access": 0, "holdout_access": 0,
        "frozen_rtl_modified": False, "canonical_netlists_modified": False,
        "manifest": record(MANIFEST),
        "next_gate": "STAGE 12B-2H — BOUNDED PROBE-CAPTURE EXECUTION AND DATASET INTEGRITY FREEZE",
    }
    frozen_write(AUDIT, canonical_json(audit))

    require(plan_payload == EXECUTION_PLAN.read_bytes(), "execution-plan replay")
    for path in (PREFLIGHT, EXECUTION_CONTRACT, AUTHORIZATION, MANIFEST, AUDIT):
        require(canonical_json(load_json(path)) == path.read_bytes(), f"canonical JSON replay: {path.name}")
    for path, expected in PINNED.items():
        require(sha256(path) == expected, f"frozen input changed during stage: {rel(path)}")

    print("\nSTAGE 12B-2G — BOUNDED PROBE-CAPTURE EXECUTION AUTHORIZATION FREEZE")
    print(f"{'Status':<57}: PASS")
    print(f"{'Authorization status':<57}: FROZEN")
    print(f"{'Bounded probe capture':<57}: AUTHORIZED / NOT STARTED")
    print(f"{'Canonical batches':<57}: {BATCHES}")
    print(f"{'Pilot sites / fault instances':<57}: {PILOT_SITES_COUNT} / {FAULTS}")
    print(f"{'Selected vectors / probe bits / snapshots':<57}: {VECTORS} / {PROBES} / {SNAPSHOTS}")
    print(f"{'Enabled transactions':<57}: {ENABLED_TRANSACTIONS}")
    print(f"{'Execution / build jobs':<57}: SEQUENTIAL / 1")
    print(f"{'Checkpoint/resume':<57}: REQUIRED")
    print(f"{'Available / minimum disk':<57}: {free_bytes / 1024**3:.2f} / {MIN_FREE_GIB} GiB")
    print(f"{'Simulation / probe capture performed':<57}: 0 / 0")
    print(f"{'Full repair campaign / model training':<57}: NOT AUTHORIZED / NOT AUTHORIZED")
    print(f"{'Fault selector/value/raw diagnostic features':<57}: PROHIBITED")
    print(f"{'REPAIR_CALIBRATION / REPAIR_SITE_TEST':<57}: LOCKED / LOCKED")
    print(f"{'Original DEV_SITE_TEST / VALIDATION / HOLDOUT':<57}: CONSUMED / 0 / 0")
    print(f"{'Frozen RTL / canonical netlists modified':<57}: NO / NO")
    print(f"{'Execution plan':<57}: {EXECUTION_PLAN}")
    print(f"{'Execution plan SHA':<57}: {sha256(EXECUTION_PLAN)}")
    print(f"{'Execution contract':<57}: {EXECUTION_CONTRACT}")
    print(f"{'Execution contract SHA':<57}: {sha256(EXECUTION_CONTRACT)}")
    print(f"{'Authorization':<57}: {AUTHORIZATION}")
    print(f"{'Authorization SHA':<57}: {sha256(AUTHORIZATION)}")
    print(f"{'Manifest':<57}: {MANIFEST}")
    print(f"{'Manifest SHA':<57}: {sha256(MANIFEST)}")
    print(f"{'Audit':<57}: {AUDIT}")
    print(f"{'Audit SHA':<57}: {sha256(AUDIT)}")
    print(f"{'Next gate':<57}: STAGE 12B-2H — BOUNDED PROBE-CAPTURE EXECUTION AND DATASET INTEGRITY FREEZE")


def record_placeholder(path: Path, payload: bytes) -> dict[str, Any]:
    return {"path": rel(path), "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}


if __name__ == "__main__":
    main()
