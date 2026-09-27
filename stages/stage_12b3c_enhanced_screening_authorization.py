#!/usr/bin/env python3
"""Stage 12B-3C: enhanced-measurement bounded-screen authorization freeze.

Verifies the frozen Stage 12B-3B candidates, deterministically commits a
REPAIR_TRAIN-only 256-site/512-fault cohort and 48-vector subset, freezes the
three-candidate execution plan, and authorizes only that bounded screen.  It
does not simulate, capture responses, deserialize a model, train, infer, or
open REPAIR_CALIBRATION, REPAIR_SITE_TEST, DEV_SITE_TEST, VALIDATION or HOLDOUT.
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
import zipfile
from pathlib import Path
from typing import Any

try:
    import numpy as np
except ImportError as error:
    raise SystemExit("STOP: activate the project .venv (NumPy required)") from error


STAGE = "12B-3C"
ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "config/v2_1_improvement"
RESULT_2 = ROOT / "results/circuitsage_hmac_v2_12b2"
RESULT = ROOT / "results/circuitsage_hmac_v2_12b3"
WORK = RESULT / "enhanced_screening_authorization_12b3c"

SOURCE_3B = ROOT / "stage_12b3b_enhanced_probe_discovery.py"
DISCOVERY_DIR = RESULT / "enhanced_probe_discovery_12b3b"
PROBE_BANK_CSV = DISCOVERY_DIR / "circuitsage_hmac_v2_1_enhanced_probe_banks_12b3b.csv"
PROBE_BANK_JSON = DISCOVERY_DIR / "circuitsage_hmac_v2_1_enhanced_probe_banks_12b3b.json"
CONSISTENCY_3B = DISCOVERY_DIR / "circuitsage_hmac_v2_1_enhanced_cross_batch_consistency_12b3b.json"
STRUCTURAL_METRICS_3B = DISCOVERY_DIR / "circuitsage_hmac_v2_1_enhanced_structural_metrics_12b3b.csv"
SCREENING_CONTRACT_3B = CONFIG / "circuitsage_hmac_v2_1_enhanced_probe_screening_contract_12b3b.json"
MANIFEST_3B = RESULT / "circuitsage_hmac_v2_1_enhanced_probe_discovery_manifest_12b3b.json"
AUDIT_3B = RESULT / "circuitsage_hmac_v2_1_enhanced_probe_discovery_structural_freeze_12b3b.json"

PILOT_SITES = RESULT_2 / "adaptive_vector_pool_12b2b/circuitsage_hmac_v2_1_pilot_screening_sites_12b2b.csv"
SELECTED_VECTORS = RESULT_2 / "adaptive_vector_selection_12b2d/circuitsage_hmac_v2_1_selected_adaptive_vectors_12b2d.npz"
SELECTION_LOCK = RESULT_2 / "adaptive_vector_selection_12b2d/circuitsage_hmac_v2_1_adaptive_vector_selection_lock_12b2d.json"
AUDIT_2H = RESULT_2 / "circuitsage_hmac_v2_1_probe_capture_dataset_integrity_freeze_12b2h.json"

SCREEN_SITES_CSV = WORK / "circuitsage_hmac_v2_1_enhanced_screening_sites_12b3c.csv"
SCREEN_VECTORS_NPZ = WORK / "circuitsage_hmac_v2_1_enhanced_screening_vectors_12b3c.npz"
SCREEN_SCHEMA = WORK / "circuitsage_hmac_v2_1_enhanced_screening_schema_12b3c.json"
EXECUTION_PLAN = WORK / "circuitsage_hmac_v2_1_enhanced_screening_execution_plan_12b3c.csv"
PREFLIGHT = WORK / "circuitsage_hmac_v2_1_enhanced_screening_preflight_12b3c.json"
EXECUTION_CONTRACT = CONFIG / "circuitsage_hmac_v2_1_enhanced_screening_execution_contract_12b3c.json"
AUTHORIZATION = CONFIG / "circuitsage_hmac_v2_1_enhanced_screening_authorization_12b3c.json"
MANIFEST = RESULT / "circuitsage_hmac_v2_1_enhanced_screening_authorization_manifest_12b3c.json"
AUDIT = RESULT / "circuitsage_hmac_v2_1_enhanced_screening_authorization_freeze_12b3c.json"

PINNED = {
    SOURCE_3B: "4c25f7cee68a1a158ec4e777f2abf8dced8e263e80e97e65ee0a140c02cb1460",
    PROBE_BANK_CSV: "d7f9b984d48db4e9a8c242e1ee4e859ef5f321d6d5546b901bde569d5009eae2",
    PROBE_BANK_JSON: "f7ee21e7b1f166c88486607689a0ec1c951de484175a7a54757ed185d58db537",
    CONSISTENCY_3B: "6c94117035b979135657d2b384b11db40a7c63029f34df365f8f34360d4bbb20",
    STRUCTURAL_METRICS_3B: "6ad9e9ee07752b7f3cd32d001602d53fdac35d8c137fcea220d5761fad04ac12",
    SCREENING_CONTRACT_3B: "c62c2a319be63d6c3d2557a0b69fd2b8508af9cabadf5e5cf21fad81de930e38",
    MANIFEST_3B: "7d587576cfdc308080f0cec9a53b9ad427525da5e8ae359260ba24b4d54ad5d9",
    AUDIT_3B: "30f335ac2ad532183efac322d76d39392a4e6127043df94aab1febf891db3a67",
    PILOT_SITES: "87eceaf80a800cf60b8a7e3cb20e0ec3f934158dc35ae3d10f73e8cf9d25313c",
    SELECTED_VECTORS: "be9df0a3a70e61b54fc793439328bbcc307b9373fd881b0e97d500919ec27bff",
    SELECTION_LOCK: "963fa242559e6681b8c3f58d3de103f1b638bc4476425c7a80776c6b435597b1",
    AUDIT_2H: "cc68202cc1ef92d32319c43ca1d8afd2c9bb324ac24fb9125e630d50a7403d91",
}

CANDIDATES = {
    "EM_TOPOLOGY_4X64_T24": {"probe_bits": 256, "probe_banks": 4, "snapshots": 24},
    "EM_STATE_CHECKPOINT_2X64_T32": {"probe_bits": 128, "probe_banks": 2, "snapshots": 32},
    "EM_TESTPOINT_4X64_T16": {"probe_bits": 256, "probe_banks": 4, "snapshots": 16},
}
PILOT_SITE_COUNT = 1024
PILOT_FAULTS = 2048
SCREEN_SITES = 256
SCREEN_FAULTS = 512
SCREEN_VECTORS = 48
BATCHES = 45
ENABLED_PER_CANDIDATE = SCREEN_FAULTS * SCREEN_VECTORS
TOTAL_ENABLED = ENABLED_PER_CANDIDATE * len(CANDIDATES)
FROZEN_TARGET = 0.70
MIN_FREE_GIB = 10
SITE_DOMAIN = "CIRCUITSAGE-HMAC-V2.1-ENHANCED-SCREEN-SITES-12B3C-v1"


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


def placeholder(path: Path, data: bytes) -> dict[str, Any]:
    return {"path": rel(path), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def resolve_record(item: dict[str, Any]) -> Path:
    value = item.get("path")
    require(isinstance(value, str) and value, "artifact record path")
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def verify_record(item: dict[str, Any], expected: Path | None, label: str) -> Path:
    path = resolve_record(item)
    if expected is not None:
        require(path.resolve() == expected.resolve(), f"{label} path")
    require(path.is_file(), f"missing {label}: {rel(path)}")
    require(item.get("sha256") == sha256(path), f"{label} SHA")
    require(int(item.get("bytes", -1)) == path.stat().st_size, f"{label} size")
    return path


def csv_payload(rows: list[dict[str, Any]]) -> bytes:
    require(bool(rows), "CSV rows")
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode()


def npy_bytes(array: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.lib.format.write_array(stream, array, allow_pickle=False)
    return stream.getvalue()


def deterministic_npz(arrays: dict[str, np.ndarray]) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in sorted(arrays):
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            info.create_system = 3
            archive.writestr(info, npy_bytes(arrays[name]), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    return stream.getvalue()


def verify_inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    print("STAGE 12B-3C — ENHANCED MEASUREMENT BOUNDED-SCREENING AUTHORIZATION")
    print("FROZEN INPUT VERIFICATION")
    for path, expected in PINNED.items():
        require(path.is_file(), f"missing frozen input: {rel(path)}")
        require(sha256(path) == expected, f"frozen input SHA: {rel(path)}")
        print(f"  {path.name:<91}: OK", flush=True)
    audit = load_json(AUDIT_3B)
    contract = load_json(SCREENING_CONTRACT_3B)
    manifest = load_json(MANIFEST_3B)
    require(audit.get("status") == "PASS", "12B-3B status")
    require(audit.get("probe_discovery_status") == "FROZEN / COMPLETE", "probe discovery state")
    require(audit.get("candidate_structural_feasibility") == "PASS / 3 OF 3", "candidate feasibility")
    require(audit.get("cross_batch_consistency") == "PASS / 45/45", "cross-batch consistency")
    require(audit.get("yosys_structural_checks") == "PASS / 3 OF 3", "Yosys checks")
    require(audit.get("verilator_lints") == "PASS / 3 OF 3", "Verilator checks")
    require(audit.get("canary_execution_response_capture") == "0 / 0", "canary non-execution")
    require(audit.get("bounded_screening") == "NOT YET AUTHORIZED", "screening entry state")
    require(audit.get("model_training") == "NOT AUTHORIZED", "training boundary")
    require(audit.get("repair_site_test") == "LOCKED / NOT ACCESSED", "repair-site-test boundary")
    require(audit.get("validation_access") == 0 and audit.get("holdout_access") == 0, "protected access")
    require(contract.get("status") == "FROZEN", "3B screening contract")
    require(contract.get("candidate_count") == 3, "candidate count")
    require(contract.get("screening_authorization") == "NOT YET AUTHORIZED", "screen authorization state")
    require(contract.get("allowed_future_partition") == "REPAIR_TRAIN BOUNDED SCREEN ONLY", "screen partition")
    require(contract.get("screening_sites") == SCREEN_SITES and contract.get("screening_fault_instances") == SCREEN_FAULTS, "screen cohort")
    require(contract.get("screening_vectors") == SCREEN_VECTORS and contract.get("maximum_enabled_transactions") == TOTAL_ENABLED, "screen transaction budget")
    require(float(contract.get("frozen_detection_target")) == FROZEN_TARGET, "frozen target")
    require(contract.get("fault_identity_in_query") == "PROHIBITED", "identity boundary")
    require(contract.get("model_training") == "NOT AUTHORIZED", "contract training boundary")
    require(manifest.get("status") == "PASS", "3B manifest")
    outputs = manifest.get("outputs")
    require(isinstance(outputs, dict) and len(outputs) >= 20, "3B output records")
    for name, item in outputs.items():
        require(isinstance(item, dict), f"3B output record: {name}")
        verify_record(item, None, f"3B output {name}")
    with STRUCTURAL_METRICS_3B.open(newline="", encoding="utf-8") as stream:
        metrics = list(csv.DictReader(stream))
    require(len(metrics) == 3, "structural metric rows")
    require([row["candidate_id"] for row in metrics] == list(CANDIDATES), "candidate order")
    require(all(row["yosys_structural_check"] == "PASS" and row["verilator_lint"] == "PASS" for row in metrics), "candidate structural checks")
    return audit, contract


def load_prior_visibility() -> tuple[dict[int, tuple[bool, bool]], dict[str, Any]]:
    audit = load_json(AUDIT_2H)
    require(audit.get("status") == "PASS" and audit.get("dataset_status") == "FROZEN", "12B-2H dataset state")
    feature_record = audit.get("features")
    target_record = audit.get("targets")
    require(isinstance(feature_record, dict) and isinstance(target_record, dict), "12B-2H data records")
    features_path = verify_record(feature_record, None, "12B-2H features")
    targets_path = verify_record(target_record, None, "12B-2H targets")
    with np.load(features_path, allow_pickle=False) as features:
        external = features["external_detected"].astype(np.uint8, copy=True)
        probe = features["probe_effect"].astype(np.uint8, copy=True)
    with np.load(targets_path, allow_pickle=False) as targets:
        site_index = targets["site_index"].astype(np.int32, copy=True)
        fault_index = targets["fault_instance_index"].astype(np.int32, copy=True)
    require(external.shape == (PILOT_FAULTS, 96) and probe.shape == (PILOT_FAULTS, 96), "prior response shapes")
    require(site_index.shape == (PILOT_FAULTS,) and fault_index.tolist() == list(range(PILOT_FAULTS)), "prior target order")
    visible = np.any((external != 0) | (probe != 0), axis=1)
    lookup: dict[int, tuple[bool, bool]] = {}
    for offset in range(0, PILOT_FAULTS, 2):
        require(site_index[offset] == site_index[offset + 1], "SA-pair site grouping")
        lookup[int(site_index[offset])] = (bool(visible[offset]), bool(visible[offset + 1]))
    require(len(lookup) == PILOT_SITE_COUNT, "prior site lookup")
    return lookup, {"features": feature_record, "targets": target_record}


def hash_rank(site_id: str) -> str:
    return hashlib.sha256(f"{SITE_DOMAIN}|{site_id}".encode()).hexdigest()


def select_screen_sites(visibility: dict[int, tuple[bool, bool]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    with PILOT_SITES.open(newline="", encoding="utf-8") as stream:
        pilot = list(csv.DictReader(stream))
    require(len(pilot) == PILOT_SITE_COUNT, "pilot site count")
    require([int(row["pilot_rank"]) for row in pilot] == list(range(PILOT_SITE_COUNT)), "pilot order")
    enriched: list[dict[str, Any]] = []
    for row in pilot:
        site_index = int(row["site_index"])
        require(site_index in visibility, "site visibility lookup")
        sa0, sa1 = visibility[site_index]
        prior_class = "BOTH_OBSERVABLE" if sa0 and sa1 else "BOTH_UNOBSERVABLE" if not sa0 and not sa1 else "MIXED"
        batch_id = (site_index - 1) // 512
        require(0 <= batch_id < BATCHES, "batch ID")
        enriched.append({**row, "batch_id": batch_id, "prior_sa0_observable": int(sa0),
                         "prior_sa1_observable": int(sa1), "prior_visibility_class": prior_class,
                         "selection_hash": hash_rank(row["site_id"])})
    ranked = sorted(enriched, key=lambda row: (row["selection_hash"], int(row["site_index"])))
    selected = ranked[:SCREEN_SITES]
    selected_ids = {row["site_id"] for row in selected}
    missing_batches = sorted(set(range(BATCHES)) - {row["batch_id"] for row in selected})
    for missing in missing_batches:
        addition = min((row for row in ranked if row["batch_id"] == missing and row["site_id"] not in selected_ids), key=lambda row: row["selection_hash"])
        counts: dict[int, int] = {}
        for row in selected:
            counts[row["batch_id"]] = counts.get(row["batch_id"], 0) + 1
        removable = max((row for row in selected if counts[row["batch_id"]] > 1), key=lambda row: row["selection_hash"])
        selected.remove(removable)
        selected_ids.remove(removable["site_id"])
        selected.append(addition)
        selected_ids.add(addition["site_id"])
    selected.sort(key=lambda row: (row["batch_id"], row["selection_hash"], int(row["site_index"])))
    require(len(selected) == SCREEN_SITES and len(selected_ids) == SCREEN_SITES, "screen-site selection")
    require({row["batch_id"] for row in selected} == set(range(BATCHES)), "all-batch screen coverage")
    output: list[dict[str, Any]] = []
    class_counts = {"BOTH_OBSERVABLE": 0, "MIXED": 0, "BOTH_UNOBSERVABLE": 0}
    for rank, row in enumerate(selected):
        class_counts[row["prior_visibility_class"]] += 1
        output.append({
            "screen_rank": rank, "batch_id": row["batch_id"], "site_id": row["site_id"],
            "site_index": int(row["site_index"]), "pilot_rank": int(row["pilot_rank"]),
            "repair_partition_rank": int(row["repair_partition_rank"]), "fault_polarities": "SA0|SA1",
            "prior_sa0_observable": row["prior_sa0_observable"],
            "prior_sa1_observable": row["prior_sa1_observable"],
            "prior_visibility_class": row["prior_visibility_class"],
            "selection_hash_sha256": row["selection_hash"],
        })
    return output, class_counts


def select_screen_vectors() -> tuple[bytes, list[int]]:
    lock = load_json(SELECTION_LOCK)
    require(lock.get("status") == "FROZEN" and lock.get("selected_vectors") == 96, "vector-selection lock")
    require(lock.get("selection_partition") == "REPAIR_TRAIN PILOT ONLY", "vector partition")
    with np.load(SELECTED_VECTORS, allow_pickle=False) as archive:
        required = {"key_u8", "message_u8", "selection_rank", "selection_round", "vector_index"}
        require(set(archive.files) == required, "selected-vector members")
        arrays = {name: np.asarray(archive[name]).copy() for name in required}
    order = np.argsort(arrays["selection_rank"], kind="stable")
    require(arrays["selection_rank"][order].tolist() == list(range(96)), "selection-rank order")
    chosen = order[:SCREEN_VECTORS]
    output = {name: arrays[name][chosen] for name in required}
    require(output["key_u8"].shape == (SCREEN_VECTORS, 32), "screen key shape")
    require(output["message_u8"].shape == (SCREEN_VECTORS, 32), "screen message shape")
    indices = output["vector_index"].astype(np.int32).tolist()
    require(len(set(indices)) == SCREEN_VECTORS, "screen vector uniqueness")
    return deterministic_npz(output), indices


def load_probe_counts() -> dict[str, int]:
    with PROBE_BANK_CSV.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    counts = {candidate: 0 for candidate in CANDIDATES}
    orders = {candidate: [] for candidate in CANDIDATES}
    for row in rows:
        candidate = row["candidate_id"]
        require(candidate in CANDIDATES, "unknown candidate in probe bank")
        counts[candidate] += 1
        orders[candidate].append(int(row["probe_bit"]))
    for candidate, spec in CANDIDATES.items():
        require(counts[candidate] == spec["probe_bits"], f"{candidate} probe count")
        require(orders[candidate] == list(range(spec["probe_bits"])), f"{candidate} probe order")
    return counts


def execution_plan(sites: list[dict[str, Any]]) -> tuple[bytes, list[dict[str, Any]]]:
    batch_counts = {batch: 0 for batch in range(BATCHES)}
    for row in sites:
        batch_counts[int(row["batch_id"])] += 1
    require(all(count > 0 for count in batch_counts.values()), "all-batch execution plan")
    rows: list[dict[str, Any]] = []
    for candidate, spec in CANDIDATES.items():
        for batch_id in range(BATCHES):
            site_count = batch_counts[batch_id]
            faults = site_count * 2
            enabled = faults * SCREEN_VECTORS
            baselines = SCREEN_VECTORS
            rows.append({
                "execution_order": len(rows), "candidate_id": candidate, "batch_id": batch_id,
                "screen_sites": site_count, "fault_instances": faults,
                "screen_vectors": SCREEN_VECTORS, "baseline_transactions": baselines,
                "enabled_transactions": enabled, "total_transactions": baselines + enabled,
                "probe_banks": spec["probe_banks"], "probe_bits": spec["probe_bits"],
                "snapshots_per_transaction": spec["snapshots"],
                "enabled_sampled_probe_bits": enabled * spec["probe_bits"] * spec["snapshots"],
            })
    require(len(rows) == len(CANDIDATES) * BATCHES, "execution-plan rows")
    for candidate in CANDIDATES:
        candidate_rows = [row for row in rows if row["candidate_id"] == candidate]
        require(sum(row["screen_sites"] for row in candidate_rows) == SCREEN_SITES, f"{candidate} site total")
        require(sum(row["fault_instances"] for row in candidate_rows) == SCREEN_FAULTS, f"{candidate} fault total")
        require(sum(row["enabled_transactions"] for row in candidate_rows) == ENABLED_PER_CANDIDATE, f"{candidate} enabled total")
    require(sum(row["enabled_transactions"] for row in rows) == TOTAL_ENABLED, "total enabled transactions")
    return csv_payload(rows), rows


def self_test() -> None:
    require(ENABLED_PER_CANDIDATE == 24576 and TOTAL_ENABLED == 73728, "transaction canary")
    require(sum(spec["probe_bits"] for spec in CANDIDATES.values()) == 640, "probe-width canary")
    require(len({hash_rank(f"SITE-{index}") for index in range(100)}) == 100, "site-hash canary")
    arrays = {"x": np.arange(8, dtype=np.int32)}
    require(deterministic_npz(arrays) == deterministic_npz(arrays), "NPZ replay canary")
    print("Stage 12B-3C self-test: PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    require(ROOT.name == "vlsi_fault_detection_v2", f"run from project root, not {ROOT}")
    outputs = (SCREEN_SITES_CSV, SCREEN_VECTORS_NPZ, SCREEN_SCHEMA, EXECUTION_PLAN,
               PREFLIGHT, EXECUTION_CONTRACT, AUTHORIZATION, MANIFEST, AUDIT)
    for path in outputs:
        require(not path.exists(), f"Stage {STAGE} output already exists: {rel(path)}")

    verify_inputs()
    visibility, prior_records = load_prior_visibility()
    sites, class_counts = select_screen_sites(visibility)
    site_bytes = csv_payload(sites)
    vector_bytes, vector_indices = select_screen_vectors()
    probe_counts = load_probe_counts()
    plan_bytes, plan_rows = execution_plan(sites)
    free_bytes = shutil.disk_usage(ROOT).free
    require(free_bytes >= MIN_FREE_GIB * 1024**3, f"less than {MIN_FREE_GIB} GiB free disk")
    estimated_observer_bytes = 0
    for spec in CANDIDATES.values():
        estimated_observer_bytes += ENABLED_PER_CANDIDATE * spec["probe_bits"] * spec["snapshots"] // 8

    schema = {
        "schema_version": "CIRCUITSAGE-HMAC-V2.1-ENHANCED-SCREENING-SCHEMA-12B3C-v1",
        "stage": STAGE, "status": "FROZEN",
        "site_primary_key": ["site_id"], "fault_primary_key": ["site_id", "stuck_value"],
        "sample_primary_key": ["candidate_id", "site_id", "stuck_value", "vector_rank"],
        "site_count": SCREEN_SITES, "fault_instances": SCREEN_FAULTS, "vector_count": SCREEN_VECTORS,
        "candidate_ids": list(CANDIDATES), "candidate_probe_counts": probe_counts,
        "query_inputs": ["key", "message", "external response", "fixed global probe observations"],
        "forbidden_feature_fields": ["fault_selector_i", "fault_enable_i", "fault_value_i", "fault_raw_o",
                                     "site_id", "site_index", "fault_instance_id", "stuck_value"],
        "identity_metadata_policy": "SEPARATE SCORING METADATA ONLY; NEVER MODEL OR DETECTOR INPUT",
        "selection_lineage": "REPAIR_TRAIN ONLY",
    }
    preflight = {
        "preflight_version": "CIRCUITSAGE-HMAC-V2.1-ENHANCED-SCREENING-PREFLIGHT-12B3C-v1",
        "stage": STAGE, "status": "PASS", "candidate_count": len(CANDIDATES),
        "canonical_batches": BATCHES, "screen_sites": SCREEN_SITES,
        "fault_instances": SCREEN_FAULTS, "screen_vectors": SCREEN_VECTORS,
        "enabled_transactions_per_candidate": ENABLED_PER_CANDIDATE,
        "maximum_total_enabled_transactions": TOTAL_ENABLED,
        "site_visibility_strata": class_counts, "all_batches_represented": True,
        "candidate_probe_bits": probe_counts,
        "estimated_packed_observer_payload_bytes": estimated_observer_bytes,
        "available_disk_bytes": free_bytes, "minimum_free_disk_bytes": MIN_FREE_GIB * 1024**3,
        "toolchain": {"python": sys.version.split()[0], "numpy": np.__version__, "platform": platform.platform(),
                      "yosys": shutil.which("yosys"), "verilator": shutil.which("verilator")},
        "frozen_input_verification": "PASS", "simulation_or_capture_performed": False,
    }
    execution_contract = {
        "contract_version": "CIRCUITSAGE-HMAC-V2.1-ENHANCED-BOUNDED-SCREEN-12B3C-v1",
        "stage": STAGE, "status": "FROZEN", "scope": "REPAIR_TRAIN BOUNDED SCREEN ONLY",
        "candidate_ids": list(CANDIDATES), "canonical_batches": BATCHES,
        "screen_sites": SCREEN_SITES, "fault_instances": SCREEN_FAULTS,
        "screen_vectors": SCREEN_VECTORS, "enabled_transactions_per_candidate": ENABLED_PER_CANDIDATE,
        "maximum_total_enabled_transactions": TOTAL_ENABLED,
        "execution": "SEQUENTIAL", "parallel_batches": 1, "build_jobs": 1,
        "checkpoint_resume": "REQUIRED AFTER EACH CANDIDATE/BATCH PAIR",
        "candidate_schedule": CANDIDATES,
        "fault_free_baseline": "ONE MATCHED GOLDEN TRACE PER VECTOR, CANDIDATE AND CANONICAL BATCH",
        "measurement": ["EXTERNAL DIGEST/TIMEOUT/CYCLE DELTA", "FIXED-CYCLE PROBE XOR",
                        "PER-PROBE TOGGLE DELTA", "CONTROL TIMELINE DELTA"],
        "identity_independence": "SAME FROZEN PROBES, SNAPSHOT SCHEDULE AND VECTORS FOR EVERY FAULT QUERY",
        "fault_identity_in_query": "PROHIBITED",
        "failure_policy": "STOP ON BASELINE FAILURE, UNKNOWN, MISSING/DUPLICATE SAMPLE, HASH MISMATCH OR IDENTITY LEAKAGE",
        "candidate_advancement": "AT MOST ONE; REQUIRES COMBINED DETECTION >=0.70 AND ZERO FALSE ALARMS",
        "model_training": "NOT AUTHORIZED",
    }
    authorization = {
        "authorization_version": "CIRCUITSAGE-HMAC-V2.1-ENHANCED-SCREENING-AUTHORIZATION-12B3C-v1",
        "stage": STAGE, "status": "FROZEN",
        "bounded_three_candidate_screening": "AUTHORIZED / NOT STARTED",
        "authorized_execution_plan": placeholder(EXECUTION_PLAN, plan_bytes),
        "authorized_sites": placeholder(SCREEN_SITES_CSV, site_bytes),
        "authorized_vectors": placeholder(SCREEN_VECTORS_NPZ, vector_bytes),
        "enhanced_probe_banks": record(PROBE_BANK_JSON),
        "full_repair_train_capture": "NOT AUTHORIZED", "model_training": "NOT AUTHORIZED",
        "candidate_reselection_or_threshold_change": "PROHIBITED",
        "repair_calibration": "LOCKED / NOT AUTHORIZED", "repair_site_test": "LOCKED / NOT AUTHORIZED",
        "original_dev_site_test": "CONSUMED / REOPENING PROHIBITED",
        "validation": "PROHIBITED", "holdout": "PROHIBITED",
        "rtl_or_canonical_netlist_modification": "PROHIBITED",
        "derived_observer_netlists": "AUTHORIZED IN ISOLATED BUILD DIRECTORIES ONLY",
    }

    frozen_write(SCREEN_SITES_CSV, site_bytes)
    frozen_write(SCREEN_VECTORS_NPZ, vector_bytes)
    frozen_write(SCREEN_SCHEMA, canonical_json(schema))
    frozen_write(EXECUTION_PLAN, plan_bytes)
    frozen_write(PREFLIGHT, canonical_json(preflight))
    frozen_write(EXECUTION_CONTRACT, canonical_json(execution_contract))
    frozen_write(AUTHORIZATION, canonical_json(authorization))
    primary = (SCREEN_SITES_CSV, SCREEN_VECTORS_NPZ, SCREEN_SCHEMA, EXECUTION_PLAN,
               PREFLIGHT, EXECUTION_CONTRACT, AUTHORIZATION)
    manifest = {
        "manifest_version": "CIRCUITSAGE-HMAC-V2.1-ENHANCED-SCREENING-AUTHORIZATION-MANIFEST-12B3C-v1",
        "stage": STAGE, "status": "PASS", "stage_12b3b_audit": record(AUDIT_3B),
        "prior_response_records": prior_records, "outputs": {rel(path): record(path) for path in primary},
        "candidate_count": len(CANDIDATES), "canonical_batches": BATCHES,
        "screen_sites": SCREEN_SITES, "fault_instances": SCREEN_FAULTS, "screen_vectors": SCREEN_VECTORS,
        "enabled_transactions_per_candidate": ENABLED_PER_CANDIDATE,
        "maximum_total_enabled_transactions": TOTAL_ENABLED,
        "simulation_calls": 0, "response_values_captured": 0,
        "model_objects_deserialized": 0, "training_calls": 0, "inference_calls": 0,
        "repair_calibration_access": 0, "repair_site_test_access": 0,
        "original_dev_site_test_access": 0, "validation_access": 0, "holdout_access": 0,
        "frozen_rtl_modified": False, "canonical_netlists_modified": False,
    }
    frozen_write(MANIFEST, canonical_json(manifest))
    audit = {
        "audit_version": "CIRCUITSAGE-HMAC-V2.1-ENHANCED-SCREENING-AUTHORIZATION-FREEZE-12B3C-v1",
        "stage": STAGE, "status": "PASS", "authorization_status": "FROZEN",
        "bounded_three_candidate_screening": "AUTHORIZED / NOT STARTED",
        "candidate_count": 3, "canonical_batches": BATCHES,
        "screen_sites_faults_vectors": [SCREEN_SITES, SCREEN_FAULTS, SCREEN_VECTORS],
        "enabled_transactions_per_candidate": ENABLED_PER_CANDIDATE,
        "maximum_total_enabled_transactions": TOTAL_ENABLED,
        "site_visibility_strata": class_counts, "all_batches_represented": True,
        "selected_vector_indices": vector_indices,
        "fault_identity_in_query": "PROHIBITED", "identity_independent_schedule": "FROZEN / PASS",
        "frozen_detection_target": FROZEN_TARGET, "false_alarm_target": 0.0,
        "full_repair_train_capture": "NOT AUTHORIZED", "model_training": "NOT AUTHORIZED",
        "repair_calibration": "LOCKED / NOT ACCESSED", "repair_site_test": "LOCKED / NOT ACCESSED",
        "original_dev_site_test": "CONSUMED / NOT REOPENED", "validation_access": 0, "holdout_access": 0,
        "frozen_rtl_modified": False, "canonical_netlists_modified": False,
        "manifest": record(MANIFEST),
        "next_gate": "STAGE 12B-3D — ENHANCED MEASUREMENT BOUNDED-SCREENING EXECUTION AND DATASET FREEZE",
    }
    frozen_write(AUDIT, canonical_json(audit))

    require(site_bytes == SCREEN_SITES_CSV.read_bytes(), "site-plan replay")
    require(vector_bytes == SCREEN_VECTORS_NPZ.read_bytes(), "vector replay")
    require(plan_bytes == EXECUTION_PLAN.read_bytes(), "execution-plan replay")
    for path in (SCREEN_SCHEMA, PREFLIGHT, EXECUTION_CONTRACT, AUTHORIZATION, MANIFEST, AUDIT):
        require(canonical_json(load_json(path)) == path.read_bytes(), f"canonical JSON replay: {path.name}")
    for path, expected in PINNED.items():
        require(sha256(path) == expected, f"frozen input changed: {rel(path)}")

    print("\nSTAGE 12B-3C — ENHANCED MEASUREMENT BOUNDED-SCREENING AUTHORIZATION FREEZE")
    print(f"{'Status':<65}: PASS")
    print(f"{'Authorization status':<65}: FROZEN")
    print(f"{'Bounded three-candidate screening':<65}: AUTHORIZED / NOT STARTED")
    print(f"{'Candidates / canonical batches':<65}: 3 / {BATCHES}")
    print(f"{'Screen sites / faults / vectors':<65}: {SCREEN_SITES} / {SCREEN_FAULTS} / {SCREEN_VECTORS}")
    print(f"{'Enabled transactions per candidate / maximum total':<65}: {ENABLED_PER_CANDIDATE} / {TOTAL_ENABLED}")
    print(f"{'Prior visibility strata':<65}: {class_counts}")
    print(f"{'All canonical batches represented':<65}: YES")
    print(f"{'Fault identity in query':<65}: PROHIBITED")
    print(f"{'Frozen detection / false-alarm targets':<65}: {FROZEN_TARGET:.8f} / 0.00000000")
    print(f"{'Execution / build jobs / checkpoint':<65}: SEQUENTIAL / 1 / REQUIRED")
    print(f"{'Available / minimum disk':<65}: {free_bytes / 1024**3:.2f} / {MIN_FREE_GIB} GiB")
    print(f"{'Simulation / response capture / model training':<65}: 0 / 0 / 0")
    print(f"{'Full repair capture / model training':<65}: NOT AUTHORIZED / NOT AUTHORIZED")
    print(f"{'REPAIR_CALIBRATION / REPAIR_SITE_TEST':<65}: LOCKED / LOCKED")
    print(f"{'Original DEV_SITE_TEST / VALIDATION / HOLDOUT':<65}: CONSUMED / 0 / 0")
    print(f"{'Screen sites':<65}: {SCREEN_SITES_CSV}")
    print(f"{'Screen sites SHA':<65}: {sha256(SCREEN_SITES_CSV)}")
    print(f"{'Screen vectors':<65}: {SCREEN_VECTORS_NPZ}")
    print(f"{'Screen vectors SHA':<65}: {sha256(SCREEN_VECTORS_NPZ)}")
    print(f"{'Execution plan':<65}: {EXECUTION_PLAN}")
    print(f"{'Execution plan SHA':<65}: {sha256(EXECUTION_PLAN)}")
    print(f"{'Execution contract':<65}: {EXECUTION_CONTRACT}")
    print(f"{'Execution contract SHA':<65}: {sha256(EXECUTION_CONTRACT)}")
    print(f"{'Authorization':<65}: {AUTHORIZATION}")
    print(f"{'Authorization SHA':<65}: {sha256(AUTHORIZATION)}")
    print(f"{'Manifest':<65}: {MANIFEST}")
    print(f"{'Manifest SHA':<65}: {sha256(MANIFEST)}")
    print(f"{'Audit':<65}: {AUDIT}")
    print(f"{'Audit SHA':<65}: {sha256(AUDIT)}")
    print(f"{'Next gate':<65}: STAGE 12B-3D — ENHANCED MEASUREMENT BOUNDED-SCREENING EXECUTION AND DATASET FREEZE")


if __name__ == "__main__":
    main()
