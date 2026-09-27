#!/usr/bin/env python3
"""Stage 12B-3F: single-winner full REPAIR_TRAIN capture authorization.

Verifies the frozen Stage 12B-3E disposition, binds the selected
EM_TESTPOINT_4X64_T16 measurement to the complete 1,024-site REPAIR_TRAIN
pilot cohort and 96 frozen adaptive vectors, constructs the deterministic
45-batch plan, and authorizes only the bounded full capture.

No simulation, response capture, model loading, training, calibration or
inference is performed.  REPAIR_CALIBRATION, REPAIR_SITE_TEST, the consumed
DEV_SITE_TEST partition, VALIDATION and HOLDOUT remain unopened.
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


STAGE = "12B-3F"
ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "config/v2_1_improvement"
RESULT = ROOT / "results/circuitsage_hmac_v2_12b3"
RESULT_2 = ROOT / "results/circuitsage_hmac_v2_12b2"
WORK = RESULT / "full_capture_authorization_12b3f"

SOURCE_3E = ROOT / "stage_12b3e_enhanced_screening_disposition.py"
DISPOSITION_3E = CONFIG / "circuitsage_hmac_v2_1_enhanced_screening_disposition_policy_12b3e.json"
READINESS_3E = CONFIG / "circuitsage_hmac_v2_1_full_repair_capture_readiness_contract_12b3e.json"
WORK_3E = RESULT / "enhanced_screening_disposition_12b3e"
WINNER_LOCK_3E = WORK_3E / "circuitsage_hmac_v2_1_enhanced_measurement_winner_lock_12b3e.json"
REGISTRY_3E = WORK_3E / "circuitsage_hmac_v2_1_enhanced_screening_candidate_disposition_12b3e.csv"
REPORT_3E = WORK_3E / "circuitsage_hmac_v2_1_enhanced_screening_disposition_report_12b3e.md"
MANIFEST_3E = RESULT / "circuitsage_hmac_v2_1_enhanced_screening_disposition_manifest_12b3e.json"
AUDIT_3E = RESULT / "circuitsage_hmac_v2_1_enhanced_screening_disposition_full_capture_readiness_freeze_12b3e.json"

SOURCE_2D = ROOT / "stage_12b2d_adaptive_vector_selection_contract.py"
PILOT_SITES_2B = RESULT_2 / "adaptive_vector_pool_12b2b/circuitsage_hmac_v2_1_pilot_screening_sites_12b2b.csv"
SELECTED_VECTORS_2D = RESULT_2 / "adaptive_vector_selection_12b2d/circuitsage_hmac_v2_1_selected_adaptive_vectors_12b2d.npz"
VECTOR_LOCK_2D = RESULT_2 / "adaptive_vector_selection_12b2d/circuitsage_hmac_v2_1_adaptive_vector_selection_lock_12b2d.json"

DISCOVERY_3B = RESULT / "enhanced_probe_discovery_12b3b"
PROBE_CSV_3B = DISCOVERY_3B / "circuitsage_hmac_v2_1_enhanced_probe_banks_12b3b.csv"
PROBE_JSON_3B = DISCOVERY_3B / "circuitsage_hmac_v2_1_enhanced_probe_banks_12b3b.json"
CONSISTENCY_3B = DISCOVERY_3B / "circuitsage_hmac_v2_1_enhanced_cross_batch_consistency_12b3b.json"

FULL_SITES = WORK / "circuitsage_hmac_v2_1_full_capture_sites_12b3f.csv"
EXECUTION_PLAN = WORK / "circuitsage_hmac_v2_1_full_capture_execution_plan_12b3f.csv"
SCHEMA = WORK / "circuitsage_hmac_v2_1_full_capture_schema_12b3f.json"
PREFLIGHT = WORK / "circuitsage_hmac_v2_1_full_capture_preflight_12b3f.json"
EXECUTION_CONTRACT = CONFIG / "circuitsage_hmac_v2_1_full_capture_execution_contract_12b3f.json"
AUTHORIZATION = CONFIG / "circuitsage_hmac_v2_1_full_capture_authorization_12b3f.json"
MANIFEST = RESULT / "circuitsage_hmac_v2_1_full_capture_authorization_manifest_12b3f.json"
AUDIT = RESULT / "circuitsage_hmac_v2_1_full_capture_authorization_freeze_12b3f.json"

PINNED = {
    SOURCE_3E: "e79b697786d8b004e76f2fa50787dede932f108962c30cbe3030fb8f22307df9",
    DISPOSITION_3E: "2d943c938394ea1bdbbc9bade050b5e4d132c2dcc27adb702df7f491a4508ec3",
    READINESS_3E: "239fb7544df7fe05da631a0223a5259f1d9bea647db6e8625ccd777a043d43cb",
    WINNER_LOCK_3E: "f735fc23c28d7a009841274c85b0cc59ddc82c3b4dad6778ae1896406eb3f0ef",
    REGISTRY_3E: "695b26f48341bd0621e0dae69a2e893096f9e79e8b625167b92b1d4279867c18",
    REPORT_3E: "e576f5d2d1966d8995079975670ed8516da7d0e56962a74735817d640b8ad4aa",
    MANIFEST_3E: "536625d02780418f3a572fba02ed8ec94457e10ba36c715c6986a8ac103e34f7",
    AUDIT_3E: "cc113c7bd00fba6e4d5d946e213332b4cc9292f9525ee6dfae214d5e103b6695",
    SOURCE_2D: "ee125219771c1495608b0fef1bcf188fb63c3c492b141c0a927a0d7c7742ebce",
    PILOT_SITES_2B: "87eceaf80a800cf60b8a7e3cb20e0ec3f934158dc35ae3d10f73e8cf9d25313c",
    SELECTED_VECTORS_2D: "be9df0a3a70e61b54fc793439328bbcc307b9373fd881b0e97d500919ec27bff",
    VECTOR_LOCK_2D: "963fa242559e6681b8c3f58d3de103f1b638bc4476425c7a80776c6b435597b1",
    PROBE_CSV_3B: "d7f9b984d48db4e9a8c242e1ee4e859ef5f321d6d5546b901bde569d5009eae2",
    PROBE_JSON_3B: "f7ee21e7b1f166c88486607689a0ec1c951de484175a7a54757ed185d58db537",
    CONSISTENCY_3B: "6c94117035b979135657d2b384b11db40a7c63029f34df365f8f34360d4bbb20",
}

SELECTED = "EM_TESTPOINT_4X64_T16"
BATCHES = 45
SITES = 1024
FAULTS = 2048
VECTORS = 96
PROBE_BITS = 256
SNAPSHOTS = 16
BASELINE_RECORDS = BATCHES * VECTORS
ENABLED_RECORDS = FAULTS * VECTORS
TOTAL_RECORDS = BASELINE_RECORDS + ENABLED_RECORDS
MIN_FREE_GIB = 10
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


def placeholder(path: Path, data: bytes) -> dict[str, Any]:
    return {"path": rel(path), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def resolve_record(item: dict[str, Any]) -> Path:
    value = item.get("path")
    require(isinstance(value, str) and value, "artifact record path")
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def verify_record(item: Any, expected: Path | None, label: str) -> Path:
    require(isinstance(item, dict), f"{label} record")
    path = resolve_record(item)
    if expected is not None:
        require(path.resolve() == expected.resolve(), f"{label} path")
    require(path.is_file(), f"missing {label}: {rel(path)}")
    require(item.get("sha256") == sha256(path), f"{label} SHA")
    require(int(item.get("bytes", -1)) == path.stat().st_size, f"{label} size")
    return path


def find_sites(value: Any) -> list[dict[str, Any]] | None:
    if isinstance(value, dict):
        for child in value.values():
            found = find_sites(child)
            if found is not None:
                return found
    if isinstance(value, list):
        if value and isinstance(value[0], dict) and {"fault_site_id", "selector_code"} <= set(value[0]):
            return value
        for child in value:
            found = find_sites(child)
            if found is not None:
                return found
    return None


def verify_frozen_inputs() -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    print("STAGE 12B-3F — SINGLE-WINNER FULL-CAPTURE AUTHORIZATION")
    print("FROZEN INPUT VERIFICATION")
    for path, expected in PINNED.items():
        require(path.is_file(), f"missing frozen input: {rel(path)}")
        require(sha256(path) == expected, f"frozen input SHA: {rel(path)}")
        print(f"  {path.name:<98}: OK", flush=True)

    disposition = load_json(DISPOSITION_3E)
    readiness = load_json(READINESS_3E)
    winner = load_json(WINNER_LOCK_3E)
    manifest = load_json(MANIFEST_3E)
    audit = load_json(AUDIT_3E)
    require(disposition.get("status") == "FROZEN", "3E disposition status")
    require(disposition.get("selected_candidate") == SELECTED, "3E selected candidate")
    require(disposition.get("screening_acceptance") == "PASS", "3E screening acceptance")
    require(disposition.get("passing_candidates") == [SELECTED], "3E unique winner")
    require(float(disposition.get("selected_combined_detection_recall")) >= FROZEN_TARGET, "3E selected recall")
    require(disposition.get("candidate_reselection") == "PROHIBITED AFTER THIS FREEZE", "winner-lock policy")
    require(readiness.get("status") == "FROZEN", "3E readiness status")
    require(readiness.get("readiness") == "READY FOR SEPARATE AUTHORIZATION", "3E readiness")
    require(readiness.get("selected_candidate") == SELECTED, "3E readiness candidate")
    require(readiness.get("full_capture_sites") == SITES, "full site count")
    require(readiness.get("full_capture_fault_instances") == FAULTS, "full fault count")
    require(readiness.get("full_capture_vectors") == VECTORS, "full vector count")
    require(readiness.get("maximum_enabled_transactions") == ENABLED_RECORDS, "full transaction budget")
    require(readiness.get("full_capture_execution") == "NOT AUTHORIZED BY THIS STAGE", "3E execution boundary")
    require(readiness.get("model_training") == "NOT AUTHORIZED", "3E training boundary")
    require(winner.get("status") == "FROZEN" and winner.get("selected_candidate") == SELECTED, "winner lock")
    require(winner.get("probe_bits") == PROBE_BITS and winner.get("snapshots") == SNAPSHOTS, "winner dimensions")
    require(winner.get("selection_replay") == "PASS / UNIQUE", "winner replay")
    require(manifest.get("status") == "PASS" and manifest.get("selected_candidate") == SELECTED, "3E manifest")
    outputs = manifest.get("outputs")
    require(isinstance(outputs, dict), "3E outputs")
    for path in (DISPOSITION_3E, READINESS_3E, WINNER_LOCK_3E, REGISTRY_3E, REPORT_3E):
        verify_record(outputs.get(rel(path)), path, f"3E {path.name}")
    require(audit.get("status") == "PASS" and audit.get("disposition_status") == "FROZEN", "3E audit")
    require(audit.get("screening_acceptance") == "PASS", "3E audit acceptance")
    require(audit.get("full_capture_readiness") == "READY FOR SEPARATE AUTHORIZATION", "3E audit readiness")
    require(audit.get("full_capture_execution") == "NOT AUTHORIZED", "3E audit execution boundary")
    require(audit.get("model_training") == "NOT AUTHORIZED", "3E audit training boundary")
    require(audit.get("repair_calibration") == "LOCKED / NOT ACCESSED", "repair-calibration boundary")
    require(audit.get("repair_site_test") == "LOCKED / NOT ACCESSED", "repair-test boundary")
    require(audit.get("validation_access") == 0 and audit.get("holdout_access") == 0, "protected partition boundary")

    vector_lock = load_json(VECTOR_LOCK_2D)
    require(vector_lock.get("status") == "FROZEN", "vector lock status")
    require(vector_lock.get("selected_vectors") == VECTORS, "selected-vector count")
    require(vector_lock.get("selected_vectors_sha256") == sha256(SELECTED_VECTORS_2D), "selected-vector lock SHA")
    require(vector_lock.get("selection_partition") == "REPAIR_TRAIN PILOT ONLY", "vector selection partition")
    require(vector_lock.get("reselection") == "PROHIBITED AFTER FREEZE", "vector reselection boundary")
    require(vector_lock.get("repair_calibration_used") is False, "vector calibration boundary")
    require(vector_lock.get("repair_site_test_used") is False, "vector test boundary")

    with np.load(SELECTED_VECTORS_2D, allow_pickle=False) as data:
        require(set(data.files) == {"key_u8", "message_u8", "selection_rank", "selection_round", "vector_index"}, "selected-vector fields")
        require(data["key_u8"].shape == (VECTORS, 32), "selected key shape")
        require(data["message_u8"].shape == (VECTORS, 32), "selected message shape")
        require(data["selection_rank"].tolist() == list(range(VECTORS)), "selection ranks")
        require(data["selection_round"].tolist() == [index // 24 + 1 for index in range(VECTORS)], "selection rounds")
        indices = data["vector_index"].astype(np.int32, copy=True).tolist()
    require(len(set(indices)) == VECTORS and all(0 <= value < 512 for value in indices), "selected vector uniqueness")

    probe_bank = load_json(PROBE_JSON_3B)
    consistency = load_json(CONSISTENCY_3B)
    require(probe_bank.get("status") == "FROZEN", "probe-bank status")
    require(probe_bank.get("candidate_count") == 3, "probe candidate count")
    require(probe_bank.get("query_fault_identity_used") is False, "probe identity independence")
    require(probe_bank.get("fault_selector_value_raw_used") is False, "probe forbidden inputs")
    candidate = probe_bank.get("candidates", {}).get(SELECTED)
    require(isinstance(candidate, dict), "selected probe candidate")
    require(candidate.get("family") == "SIMULATION_ONLY_TEST_POINT", "selected probe family")
    require(candidate.get("bits") == PROBE_BITS and candidate.get("banks") == 4, "selected probe bank dimensions")
    require(candidate.get("snapshots") == SNAPSHOTS, "selected snapshot count")
    require(len(candidate.get("bit_ids", [])) == PROBE_BITS, "selected bit IDs")
    require(len(set(candidate.get("bit_ids", []))) == PROBE_BITS, "selected bit uniqueness")
    probe_rows = load_csv(PROBE_CSV_3B)
    selected_rows = [row for row in probe_rows if row["candidate_id"] == SELECTED]
    require(len(selected_rows) == PROBE_BITS, "selected probe CSV rows")
    require([int(row["probe_bit"]) for row in selected_rows] == list(range(PROBE_BITS)), "selected probe order")
    require([int(row["bit_id"]) for row in selected_rows] == candidate["bit_ids"], "selected JSON/CSV bit agreement")
    require(consistency.get("status") == "PASS" and consistency.get("canonical_batches") == BATCHES, "probe consistency")
    require(consistency.get("all_batches_all_candidates_exact") is True, "all-batch probe consistency")
    batches = consistency.get("batches")
    require(isinstance(batches, list) and len(batches) == BATCHES, "probe consistency batch registry")
    return readiness, probe_bank, batches


def build_full_site_plan(batch_registry: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[int, list[dict[str, Any]]]]:
    pilot_rows = load_csv(PILOT_SITES_2B)
    require(len(pilot_rows) == SITES, "pilot-site row count")
    require([int(row["pilot_rank"]) for row in pilot_rows] == list(range(SITES)), "pilot-site rank order")
    require(all(row["fault_polarities"] == "SA0|SA1" for row in pilot_rows), "pilot fault polarities")
    wanted = {row["site_id"]: row for row in pilot_rows}
    require(len(wanted) == SITES, "pilot-site uniqueness")

    found: set[str] = set()
    by_batch: dict[int, list[dict[str, Any]]] = {}
    for batch_id, entry in enumerate(batch_registry):
        require(entry.get("batch_id") == batch_id, f"Batch {batch_id:03d} registry order")
        require(entry.get("all_candidates_exact") is True, f"Batch {batch_id:03d} candidate consistency")
        counts = entry.get("candidate_probe_bits_present")
        require(isinstance(counts, dict) and counts.get(SELECTED) == PROBE_BITS, f"Batch {batch_id:03d} selected probes")
        mapping_path = verify_record(entry.get("mapping"), None, f"Batch {batch_id:03d} mapping")
        verify_record(entry.get("canonical_json"), None, f"Batch {batch_id:03d} canonical JSON")
        sites = find_sites(load_json(mapping_path))
        expected = 311 if batch_id == 44 else 512
        require(sites is not None and len(sites) == expected, f"Batch {batch_id:03d} mapping site count")
        require([int(site["selector_code"]) for site in sites] == list(range(expected)), f"Batch {batch_id:03d} selector order")
        selected: list[dict[str, Any]] = []
        for site in sites:
            site_id = str(site["fault_site_id"])
            if site_id in wanted:
                require(site_id not in found, f"site mapped twice: {site_id}")
                found.add(site_id)
                source = wanted[site_id]
                selected.append({
                    "full_rank": int(source["pilot_rank"]),
                    "pilot_rank": int(source["pilot_rank"]),
                    "batch_id": batch_id,
                    "site_id": site_id,
                    "site_index": int(source["site_index"]),
                    "selector": int(site["selector_code"]),
                    "fault_polarities": "SA0|SA1",
                })
        by_batch[batch_id] = sorted(selected, key=lambda row: row["full_rank"])
    require(found == set(wanted), f"unmapped full-capture sites: {len(set(wanted) - found)}")
    require(sum(len(rows) for rows in by_batch.values()) == SITES, "mapped full-site total")
    require(all(len(by_batch[index]) > 0 for index in range(BATCHES)), "all canonical batches represented")
    ordered = sorted((row for rows in by_batch.values() for row in rows), key=lambda row: row["full_rank"])
    require([row["full_rank"] for row in ordered] == list(range(SITES)), "full-site rank continuity")
    return ordered, by_batch


def self_test() -> None:
    require(BASELINE_RECORDS == 4320, "baseline-record canary")
    require(ENABLED_RECORDS == 196608, "enabled-record canary")
    require(TOTAL_RECORDS == 200928, "total-record canary")
    require(PROBE_BITS == 4 * 64 and SNAPSHOTS == 16, "winner-dimension canary")
    print("Stage 12B-3F self-test: PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return

    require(ROOT.name == "vlsi_fault_detection_v2", f"run from project root, not {ROOT}")
    outputs = (FULL_SITES, EXECUTION_PLAN, SCHEMA, PREFLIGHT, EXECUTION_CONTRACT, AUTHORIZATION, MANIFEST, AUDIT)
    for path in outputs:
        require(not path.exists(), f"Stage {STAGE} output already exists: {rel(path)}")

    readiness, probe_bank, batch_registry = verify_frozen_inputs()
    sites, by_batch = build_full_site_plan(batch_registry)
    site_payload = csv_payload(sites)
    plan_rows: list[dict[str, Any]] = []
    for batch_id in range(BATCHES):
        site_count = len(by_batch[batch_id])
        fault_count = site_count * 2
        enabled = fault_count * VECTORS
        plan_rows.append({
            "execution_order": batch_id,
            "batch_id": batch_id,
            "module": f"opentitan_hmac_sha256_msg32_faultbatch{batch_id:03d}",
            "site_count": site_count,
            "fault_instances": fault_count,
            "vectors": VECTORS,
            "baseline_transactions": VECTORS,
            "enabled_transactions": enabled,
            "total_records": VECTORS + enabled,
            "probe_bits": PROBE_BITS,
            "snapshots": SNAPSHOTS,
        })
    require(sum(int(row["site_count"]) for row in plan_rows) == SITES, "plan site total")
    require(sum(int(row["fault_instances"]) for row in plan_rows) == FAULTS, "plan fault total")
    require(sum(int(row["baseline_transactions"]) for row in plan_rows) == BASELINE_RECORDS, "plan baseline total")
    require(sum(int(row["enabled_transactions"]) for row in plan_rows) == ENABLED_RECORDS, "plan enabled total")
    require(sum(int(row["total_records"]) for row in plan_rows) == TOTAL_RECORDS, "plan record total")
    plan_payload = csv_payload(plan_rows)

    free_bytes = shutil.disk_usage(ROOT).free
    require(free_bytes >= MIN_FREE_GIB * 1024**3, f"minimum free disk: {MIN_FREE_GIB} GiB")
    yosys = shutil.which("yosys")
    verilator = shutil.which("verilator")
    require(yosys is not None and verilator is not None, "Yosys/Verilator unavailable; export OSS-CAD-Suite PATH")

    selected_probe = probe_bank["candidates"][SELECTED]
    schema = {
        "schema_version": "CIRCUITSAGE-HMAC-V2.1-FULL-CAPTURE-PLAN-12B3F-v1",
        "stage": STAGE,
        "status": "FROZEN",
        "selected_candidate": SELECTED,
        "partition": "REPAIR_TRAIN ONLY",
        "site_plan_columns": list(sites[0]),
        "execution_plan_columns": list(plan_rows[0]),
        "sample_primary_key": ["site_id", "stuck_value", "selection_rank"],
        "site_count": SITES,
        "fault_instances": FAULTS,
        "vector_count": VECTORS,
        "selected_vector_source": record(SELECTED_VECTORS_2D),
        "selected_probe_source": record(PROBE_JSON_3B),
        "probe_bits": PROBE_BITS,
        "probe_banks": 4,
        "snapshots": SNAPSHOTS,
        "probe_bit_ids_sha256": hashlib.sha256(canonical_json(selected_probe["bit_ids"])).hexdigest(),
        "query_inputs": ["key", "message", "external response", "fixed global test-point observations"],
        "forbidden_feature_fields": [
            "fault_selector_i", "fault_enable_i", "fault_value_i", "fault_raw_o",
            "site_id", "site_index", "fault_instance_id", "stuck_value",
        ],
        "identity_metadata_policy": "SEPARATE SCORING METADATA ONLY; NEVER DETECTOR OR MODEL INPUT",
        "expected_future_capture_outputs": [
            "EXTERNAL DIGEST/TIMEOUT/CYCLE DELTAS",
            "16 FIXED TEST-POINT SNAPSHOT XOR VALUES",
            "PER-PROBE TOGGLE COUNTS AND GOLDEN DELTAS",
            "CONTROL TIMELINE DELTAS",
        ],
    }
    preflight = {
        "preflight_version": "CIRCUITSAGE-HMAC-V2.1-FULL-CAPTURE-PREFLIGHT-12B3F-v1",
        "stage": STAGE,
        "status": "PASS",
        "selected_candidate": SELECTED,
        "canonical_batches": BATCHES,
        "all_batches_represented": True,
        "sites": SITES,
        "fault_instances": FAULTS,
        "vectors": VECTORS,
        "baseline_records": BASELINE_RECORDS,
        "enabled_transactions": ENABLED_RECORDS,
        "total_records": TOTAL_RECORDS,
        "probe_bits": PROBE_BITS,
        "snapshots": SNAPSHOTS,
        "available_disk_bytes": free_bytes,
        "minimum_free_disk_bytes": MIN_FREE_GIB * 1024**3,
        "toolchain": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
            "yosys": yosys,
            "verilator": verilator,
        },
        "frozen_input_verification": "PASS",
        "simulation_or_capture_performed": False,
        "model_objects_deserialized": 0,
    }
    execution_contract = {
        "contract_version": "CIRCUITSAGE-HMAC-V2.1-SINGLE-WINNER-FULL-CAPTURE-12B3F-v1",
        "stage": STAGE,
        "status": "FROZEN",
        "scope": "REPAIR_TRAIN FULL 1,024-SITE PILOT ONLY",
        "selected_candidate": SELECTED,
        "selected_measurement": "256 GLOBAL TEST-POINT BITS / 4 BANKS / 16 FROZEN SNAPSHOTS",
        "selected_probe_bank": record(PROBE_JSON_3B),
        "selected_vectors": record(SELECTED_VECTORS_2D),
        "authorized_site_plan": placeholder(FULL_SITES, site_payload),
        "authorized_execution_plan": placeholder(EXECUTION_PLAN, plan_payload),
        "canonical_batches": BATCHES,
        "sites": SITES,
        "fault_instances": FAULTS,
        "vectors": VECTORS,
        "baseline_records": BASELINE_RECORDS,
        "enabled_transactions": ENABLED_RECORDS,
        "maximum_total_records": TOTAL_RECORDS,
        "execution": "SEQUENTIAL",
        "parallel_batches": 1,
        "build_jobs": 1,
        "checkpoint_resume": "REQUIRED AFTER EACH CANONICAL BATCH",
        "failure_policy": "STOP ON HASH MISMATCH, BASELINE FAILURE, UNKNOWN VALUE, MISSING/DUPLICATE SAMPLE OR IDENTITY LEAKAGE",
        "fault_identity_in_query": "PROHIBITED",
        "fault_selector_value_raw_as_features": "PROHIBITED",
        "full_capture_acceptance_gate": readiness["full_capture_acceptance_gate"],
        "model_training": "NOT AUTHORIZED",
        "candidate_reselection_or_threshold_change": "PROHIBITED",
    }
    authorization = {
        "authorization_version": "CIRCUITSAGE-HMAC-V2.1-FULL-CAPTURE-AUTHORIZATION-12B3F-v1",
        "stage": STAGE,
        "status": "FROZEN",
        "single_winner_full_repair_train_capture": "AUTHORIZED / NOT STARTED",
        "selected_candidate": SELECTED,
        "winner_lock": record(WINNER_LOCK_3E),
        "authorized_sites": placeholder(FULL_SITES, site_payload),
        "authorized_vectors": record(SELECTED_VECTORS_2D),
        "authorized_execution_plan": placeholder(EXECUTION_PLAN, plan_payload),
        "authorized_probe_bank": record(PROBE_JSON_3B),
        "maximum_enabled_transactions": ENABLED_RECORDS,
        "execution": "SEQUENTIAL / ONE BUILD JOB",
        "checkpoint_resume": "REQUIRED",
        "model_training": "NOT AUTHORIZED",
        "repair_calibration": "LOCKED / NOT AUTHORIZED",
        "repair_site_test": "LOCKED / NOT AUTHORIZED",
        "original_dev_site_test": "CONSUMED / REOPENING PROHIBITED",
        "validation": "PROHIBITED",
        "holdout": "PROHIBITED",
        "frozen_rtl_or_canonical_netlist_modification": "PROHIBITED",
        "derived_observer_netlists": "AUTHORIZED IN ISOLATED BUILD DIRECTORIES ONLY",
    }

    frozen_write(FULL_SITES, site_payload)
    frozen_write(EXECUTION_PLAN, plan_payload)
    frozen_write(SCHEMA, canonical_json(schema))
    frozen_write(PREFLIGHT, canonical_json(preflight))
    frozen_write(EXECUTION_CONTRACT, canonical_json(execution_contract))
    frozen_write(AUTHORIZATION, canonical_json(authorization))
    primary = (FULL_SITES, EXECUTION_PLAN, SCHEMA, PREFLIGHT, EXECUTION_CONTRACT, AUTHORIZATION)
    manifest = {
        "manifest_version": "CIRCUITSAGE-HMAC-V2.1-FULL-CAPTURE-AUTHORIZATION-MANIFEST-12B3F-v1",
        "stage": STAGE,
        "status": "PASS",
        "stage_12b3e_audit": record(AUDIT_3E),
        "outputs": {rel(path): record(path) for path in primary},
        "selected_candidate": SELECTED,
        "canonical_batches": BATCHES,
        "sites": SITES,
        "fault_instances": FAULTS,
        "vectors": VECTORS,
        "baseline_records": BASELINE_RECORDS,
        "enabled_transactions": ENABLED_RECORDS,
        "total_records": TOTAL_RECORDS,
        "simulation_calls": 0,
        "response_values_captured": 0,
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
        "audit_version": "CIRCUITSAGE-HMAC-V2.1-FULL-CAPTURE-AUTHORIZATION-FREEZE-12B3F-v1",
        "stage": STAGE,
        "status": "PASS",
        "authorization_status": "FROZEN",
        "selected_candidate": SELECTED,
        "screening_acceptance": "PASS / VERIFIED",
        "full_capture_sites_faults_vectors": [SITES, FAULTS, VECTORS],
        "canonical_batches": f"{BATCHES}/{BATCHES}",
        "all_batches_represented": True,
        "baseline_records": BASELINE_RECORDS,
        "enabled_transactions": ENABLED_RECORDS,
        "total_records": TOTAL_RECORDS,
        "probe_bits_banks_snapshots": [PROBE_BITS, 4, SNAPSHOTS],
        "full_capture_execution": "AUTHORIZED / NOT STARTED",
        "model_training": "NOT AUTHORIZED",
        "fault_identity_in_query": "PROHIBITED",
        "repair_calibration": "LOCKED / NOT ACCESSED",
        "repair_site_test": "LOCKED / NOT ACCESSED",
        "original_dev_site_test": "CONSUMED / NOT REOPENED",
        "validation_access": 0,
        "holdout_access": 0,
        "v1_modified": False,
        "v2_core_modified": False,
        "v2_1_modified": False,
        "execution_contract": record(EXECUTION_CONTRACT),
        "authorization": record(AUTHORIZATION),
        "manifest": record(MANIFEST),
        "next_gate": "STAGE 12B-3G — SINGLE-WINNER FULL REPAIR_TRAIN CAPTURE EXECUTION AND DATASET FREEZE",
    }
    frozen_write(AUDIT, canonical_json(audit))

    require(site_payload == FULL_SITES.read_bytes(), "full-site plan replay")
    require(plan_payload == EXECUTION_PLAN.read_bytes(), "execution-plan replay")
    for path in (SCHEMA, PREFLIGHT, EXECUTION_CONTRACT, AUTHORIZATION, MANIFEST, AUDIT):
        require(canonical_json(load_json(path)) == path.read_bytes(), f"canonical JSON replay: {path.name}")
    for path, expected in PINNED.items():
        require(sha256(path) == expected, f"frozen input changed: {rel(path)}")

    print("\nSTAGE 12B-3F — SINGLE-WINNER FULL REPAIR_TRAIN CAPTURE AUTHORIZATION FREEZE")
    print(f"{'Status':<69}: PASS")
    print(f"{'Authorization status':<69}: FROZEN")
    print(f"{'Selected candidate':<69}: {SELECTED}")
    print(f"{'Screening acceptance':<69}: PASS / VERIFIED")
    print(f"{'Full-capture sites / faults / vectors':<69}: {SITES} / {FAULTS} / {VECTORS}")
    print(f"{'Canonical batches represented':<69}: {BATCHES}/{BATCHES}")
    print(f"{'Baseline / enabled / total records':<69}: {BASELINE_RECORDS} / {ENABLED_RECORDS} / {TOTAL_RECORDS}")
    print(f"{'Probe bits / banks / snapshots':<69}: {PROBE_BITS} / 4 / {SNAPSHOTS}")
    print(f"{'Execution / build jobs / checkpoint':<69}: SEQUENTIAL / 1 / REQUIRED")
    print(f"{'Available / minimum disk':<69}: {free_bytes / 1024**3:.2f} / {MIN_FREE_GIB} GiB")
    print(f"{'Full-capture execution':<69}: AUTHORIZED / NOT STARTED")
    print(f"{'Model training':<69}: NOT AUTHORIZED")
    print(f"{'REPAIR_CALIBRATION / REPAIR_SITE_TEST':<69}: LOCKED / LOCKED")
    print(f"{'DEV_SITE_TEST / VALIDATION / HOLDOUT access':<69}: 0 / 0 / 0")
    print(f"{'V1 / V2 Core / V2.1 modified':<69}: NO / NO / NO")
    print(f"{'Execution contract':<69}: {EXECUTION_CONTRACT}")
    print(f"{'Execution contract SHA':<69}: {sha256(EXECUTION_CONTRACT)}")
    print(f"{'Authorization':<69}: {AUTHORIZATION}")
    print(f"{'Authorization SHA':<69}: {sha256(AUTHORIZATION)}")
    print(f"{'Manifest':<69}: {MANIFEST}")
    print(f"{'Manifest SHA':<69}: {sha256(MANIFEST)}")
    print(f"{'Audit':<69}: {AUDIT}")
    print(f"{'Audit SHA':<69}: {sha256(AUDIT)}")
    print(f"{'Next gate':<69}: STAGE 12B-3G — SINGLE-WINNER FULL REPAIR_TRAIN CAPTURE EXECUTION AND DATASET FREEZE")


if __name__ == "__main__":
    main()
