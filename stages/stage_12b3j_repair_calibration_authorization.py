#!/usr/bin/env python3
"""Stage 12B-3J: enhanced REPAIR_CALIBRATION capture authorization.

Verifies the Stage 12B-3I repair-model contracts, binds the winning
EM_TESTPOINT_4X64_T16 measurement and frozen 96-vector set to a deterministic,
batch-stratified cohort drawn only from the pre-existing REPAIR_CALIBRATION
partition, and freezes a sequential checkpointed capture plan.

No simulation, response capture, model loading, training, inference, scaler
fit, threshold selection, or REPAIR_SITE_TEST/DEV_SITE_TEST/VALIDATION/HOLDOUT
access is performed by this stage.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    import numpy as np
except ImportError as error:
    raise SystemExit("STOP: activate the project .venv (NumPy required)") from error

try:
    import stage_12b3d_enhanced_screening_execution as common
except ImportError as error:
    raise SystemExit("STOP: Stage 12B-3D runner is required beside this script") from error


STAGE = "12B-3J"
ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "config/v2_1_improvement"
RESULT = ROOT / "results/circuitsage_hmac_v2_12b3"
WORK = RESULT / "repair_calibration_authorization_12b3j"

SOURCE_3I = ROOT / "stage_12b3i_repair_model_contract.py"
LOG_3I = RESULT / "stage_12b3i_20260917_184613.log"
RESOURCE_3I = RESULT / "stage_12b3i_resources_20260917_184613.log"
ARCHITECTURE_3I = CONFIG / "circuitsage_hmac_v2_1_repair_model_architecture_12b3i.json"
FEATURE_CONTRACT_3I = CONFIG / "circuitsage_hmac_v2_1_repair_model_feature_contract_12b3i.json"
PARTITION_CONTRACT_3I = CONFIG / "circuitsage_hmac_v2_1_repair_model_data_partition_contract_12b3i.json"
TRAINING_CONTRACT_3I = CONFIG / "circuitsage_hmac_v2_1_repair_model_training_contract_12b3i.json"
ACCEPTANCE_CONTRACT_3I = CONFIG / "circuitsage_hmac_v2_1_repair_model_acceptance_contract_12b3i.json"
CANDIDATE_GRID_3I = CONFIG / "circuitsage_hmac_v2_1_repair_model_candidate_grid_12b3i.csv"
WORK_3I = RESULT / "repair_model_contract_12b3i"
ENVIRONMENT_3I = WORK_3I / "circuitsage_hmac_v2_1_repair_model_environment_12b3i.json"
REPORT_3I = WORK_3I / "circuitsage_hmac_v2_1_repair_model_contract_report_12b3i.md"
MANIFEST_3I = RESULT / "circuitsage_hmac_v2_1_repair_model_contract_manifest_12b3i.json"
AUDIT_3I = RESULT / "circuitsage_hmac_v2_1_repair_model_contract_freeze_12b3i.json"

SOURCE_COMMON = ROOT / "stage_12b3d_enhanced_screening_execution.py"
RESULT_2 = ROOT / "results/circuitsage_hmac_v2_12b2"
RESULT_11C5 = ROOT / "results/hmac_fault_campaign_11c5"
ORIGINAL_SPLIT = RESULT_11C5 / "hmac_fault_site_group_split_11c5d.csv"
ORIGINAL_SPLIT_MANIFEST = RESULT_11C5 / "hmac_fault_site_group_split_manifest_11c5d.json"
REPAIR_SPLIT = RESULT_2 / "circuitsage_hmac_v2_1_repair_site_split_12b2a.csv"
MANIFEST_2A = RESULT_2 / "circuitsage_hmac_v2_1_improvement_contract_manifest_12b2a.json"
AUDIT_2A = RESULT_2 / "circuitsage_hmac_v2_1_improvement_contract_freeze_12b2a.json"
MANIFEST_2B = RESULT_2 / "circuitsage_hmac_v2_1_pilot_authorization_manifest_12b2b.json"
AUDIT_2B = RESULT_2 / "circuitsage_hmac_v2_1_pilot_authorization_freeze_12b2b.json"
SELECTED_VECTORS = RESULT_2 / "adaptive_vector_selection_12b2d/circuitsage_hmac_v2_1_selected_adaptive_vectors_12b2d.npz"
DISCOVERY = RESULT / "enhanced_probe_discovery_12b3b"
PROBE_CSV = DISCOVERY / "circuitsage_hmac_v2_1_enhanced_probe_banks_12b3b.csv"
PROBE_JSON = DISCOVERY / "circuitsage_hmac_v2_1_enhanced_probe_banks_12b3b.json"
CONSISTENCY = DISCOVERY / "circuitsage_hmac_v2_1_enhanced_cross_batch_consistency_12b3b.json"

SITES_CSV = WORK / "circuitsage_hmac_v2_1_repair_calibration_capture_sites_12b3j.csv"
PLAN_CSV = WORK / "circuitsage_hmac_v2_1_repair_calibration_execution_plan_12b3j.csv"
SCHEMA = WORK / "circuitsage_hmac_v2_1_repair_calibration_capture_schema_12b3j.json"
PREFLIGHT = WORK / "circuitsage_hmac_v2_1_repair_calibration_capture_preflight_12b3j.json"
EXECUTION_CONTRACT = CONFIG / "circuitsage_hmac_v2_1_repair_calibration_capture_execution_contract_12b3j.json"
AUTHORIZATION = CONFIG / "circuitsage_hmac_v2_1_repair_calibration_capture_authorization_12b3j.json"
MANIFEST = RESULT / "circuitsage_hmac_v2_1_repair_calibration_authorization_manifest_12b3j.json"
AUDIT = RESULT / "circuitsage_hmac_v2_1_repair_calibration_authorization_freeze_12b3j.json"

PINNED = {
    SOURCE_3I: "91791f1963313bb25f69810b391bf597a96882acdaea76f1386c4d7d5f25762b",
    LOG_3I: "a69545761413b68b552b1382bb4a9f8645ae92b6487d63af0c354d03442f2930",
    RESOURCE_3I: "5e199a25d2a43210a6dce96e7c4fa4257fffd03bfa17b3be3c6734f57d2e11a8",
    ARCHITECTURE_3I: "61d593a3f490d39bcb2d5dfd731e37ec0ab76af9978a44bc0f9463f6042ab407",
    FEATURE_CONTRACT_3I: "b9cfd0d73a0f76e4d661c5476615fc7f236e85c5038eedcd83e3068cf021cbf6",
    PARTITION_CONTRACT_3I: "44420dd5fb37cdca9a8207920d69fd5e3bfee6847a4f63db34b91bae54ece562",
    TRAINING_CONTRACT_3I: "aeb765ef519313679fe3feba9714de8231401928631a2dc157c68db715a10e2d",
    ACCEPTANCE_CONTRACT_3I: "ca95bc7fc7587e995f48b0efa03390734192fd251cd82ebd25387c5663d1e0d2",
    CANDIDATE_GRID_3I: "d9026b2f9c487adc3abc552637a54791960b20f0fbc8c023d55a0f49f2f56fdc",
    ENVIRONMENT_3I: "c6c79520bc320544616f533646935040445db2021232a37347c00770e8bf7455",
    REPORT_3I: "ec44396e32d2fcf1c88017a1feca78d49370c1cf52c531caa7f41741745b768b",
    MANIFEST_3I: "b3d62d856dc420a0e5cc51db53fbb09c2fcc2d985b8f4b7bb6ef124d17cd5fc2",
    AUDIT_3I: "ab2c5096b9287137371ec35db46bf53c0c9ed512985820a4b2705c71929aa108",
    SOURCE_COMMON: "3392345b66e470e1975ef32baa3a7946df8f636f2f20748b91c9d12fbfcc65ff",
    ORIGINAL_SPLIT: "602fa310547f68d1e9b5ceb6f148d6b125c69df0589929f9edfde9d264922c61",
    ORIGINAL_SPLIT_MANIFEST: "1c96b7c7c7f5301e70037aaa9a0d18effd28cae6bf15e9f85cc85f0c413473fb",
    REPAIR_SPLIT: "db18f0560de900ae2a5ea4e358471468252c0533d50b40a81194e032aa05df3b",
    MANIFEST_2A: "6cec27daefce3ecedc061c2b9eb85cd4f0f92d4ffa59d3fc10095913753ae3a1",
    AUDIT_2A: "8cd3485144447d82fb615a5f2fed47ae5c193f6b27a10bc10cc7383056877299",
    MANIFEST_2B: "c3fdc621bec4ffe51e98f4c63a190f33f7f0db3cdc41dfa49c6ee55947169bb0",
    AUDIT_2B: "95f49d71efcabc0d56673dd389ee50c9e474cee1697b5e2cb039fc2da362bc80",
    SELECTED_VECTORS: "be9df0a3a70e61b54fc793439328bbcc307b9373fd881b0e97d500919ec27bff",
    PROBE_CSV: "d7f9b984d48db4e9a8c242e1ee4e859ef5f321d6d5546b901bde569d5009eae2",
    PROBE_JSON: "f7ee21e7b1f166c88486607689a0ec1c951de484175a7a54757ed185d58db537",
    CONSISTENCY: "6c94117035b979135657d2b384b11db40a7c63029f34df365f8f34360d4bbb20",
}

SELECTED = "EM_TESTPOINT_4X64_T16"
PARTITION = "REPAIR_CALIBRATION"
BATCHES = 45
VECTORS = 96
PROBE_BITS = 256
PROBE_BANKS = 4
SNAPSHOTS = 16
MAX_CALIBRATION_SITES = 512
MIN_FREE_GIB = 10
REPAIR_CLASSIFICATION = "SPLIT-ANCHOR ROUTING ONLY; NO CONTRACT, DATA, OR SELECTION CHANGE"
SPLIT_CHAIN = (
    ORIGINAL_SPLIT, ORIGINAL_SPLIT_MANIFEST, REPAIR_SPLIT,
    MANIFEST_2A, AUDIT_2A, MANIFEST_2B, AUDIT_2B,
)


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


def csv_payload(rows: list[dict[str, Any]], fields: list[str]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def normalized(value: str) -> str:
    return "_".join(part for part in "".join(character if character.isalnum() else " " for character in value.upper()).split())


def nested_records(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        if isinstance(value.get("sha256"), str) and any(isinstance(value.get(key), str) for key in ("path", "name", "file")):
            yield value
        for child in value.values():
            yield from nested_records(child)
    elif isinstance(value, list):
        for child in value:
            yield from nested_records(child)


def record_matches(value: Any, path: Path) -> bool:
    actual = sha256(path)
    for item in nested_records(value):
        label = str(item.get("path") or item.get("name") or item.get("file") or "")
        if Path(label).name == path.name and item.get("sha256") == actual:
            return True
    serialized = json.dumps(value, sort_keys=True)
    return path.name in serialized and actual in serialized


def verify_split_chain() -> None:
    """Verify the frozen original-to-repair partition lineage explicitly."""
    original_manifest = load_json(ORIGINAL_SPLIT_MANIFEST)
    manifest_2a = load_json(MANIFEST_2A)
    audit_2a = load_json(AUDIT_2A)
    manifest_2b = load_json(MANIFEST_2B)
    audit_2b = load_json(AUDIT_2B)

    require(original_manifest.get("status") == "PASS", "original split manifest status")
    # The legacy 11C-5D manifest records the split assignment by filename and
    # structural commitments rather than the later generic {path, sha256,
    # bytes} record form.  Both legacy files are byte-pinned above.
    require(ORIGINAL_SPLIT.name in json.dumps(original_manifest, sort_keys=True),
            "original split manifest does not name original split CSV")
    require(manifest_2a.get("stage") == "12B-2A" and manifest_2a.get("status") == "PASS",
            "Stage 12B-2A manifest status")
    require(record_matches(manifest_2a, ORIGINAL_SPLIT_MANIFEST),
            "Stage 12B-2A manifest does not bind original split manifest")
    require(record_matches(manifest_2a, REPAIR_SPLIT),
            "Stage 12B-2A manifest does not bind repair split CSV")
    require(audit_2a.get("stage") == "12B-2A" and audit_2a.get("status") == "PASS",
            "Stage 12B-2A audit status")
    require(record_matches(audit_2a, MANIFEST_2A),
            "Stage 12B-2A audit does not bind its manifest")
    require(manifest_2b.get("stage") == "12B-2B" and manifest_2b.get("status") == "PASS",
            "Stage 12B-2B manifest status")
    require(record_matches(manifest_2b, MANIFEST_2A) and record_matches(manifest_2b, AUDIT_2A),
            "Stage 12B-2B manifest does not bind Stage 12B-2A")
    require(audit_2b.get("stage") == "12B-2B" and audit_2b.get("status") == "PASS",
            "Stage 12B-2B audit status")
    require(record_matches(audit_2b, MANIFEST_2B),
            "Stage 12B-2B audit does not bind its manifest")


def read_split_rows(split_csv: Path) -> tuple[list[dict[str, str]], str, str, str | None]:
    with split_csv.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames is not None, "split CSV header")
        fields = list(reader.fieldnames)
        rows = list(reader)
    partition_field = next((name for name in ("repair_partition", "partition", "split", "subset", "role") if name in fields), None)
    site_field = next((name for name in ("site_id", "fault_site_id", "physical_site_id") if name in fields), None)
    site_index_field = next((name for name in ("site_index", "physical_site_index", "global_site_index") if name in fields), None)
    require(partition_field is not None, f"partition column not found in {fields}")
    require(site_field is not None, f"site ID column not found in {fields}")
    selected = [
        row for row in rows
        if normalized(row[partition_field]) in {PARTITION, "CALIBRATION"}
        or normalized(row[partition_field]).endswith(f"_{PARTITION}")
    ]
    require(selected, f"no {PARTITION} rows in split")
    require(len({row[site_field] for row in selected}) == len(selected), "duplicate calibration sites")
    return selected, site_field, partition_field, site_index_field


def catalog_bindings() -> tuple[dict[str, dict[str, int]], dict[int, dict[str, Any]]]:
    consistency = load_json(CONSISTENCY)
    entries = consistency.get("batches")
    require(consistency.get("status") == "PASS" and consistency.get("all_batches_all_candidates_exact") is True, "probe consistency")
    require(isinstance(entries, list) and len(entries) == BATCHES, "batch consistency entries")
    batch_entries = {int(item["batch_id"]): item for item in entries}
    require(set(batch_entries) == set(range(BATCHES)), "batch consistency coverage")
    lookup: dict[str, dict[str, int]] = {}
    for batch_id in range(BATCHES):
        item = batch_entries[batch_id]
        mapping = common.resolve_record(item["mapping"])
        common.verify_record(item["mapping"], mapping, f"Batch {batch_id:03d} mapping")
        sites = common.find_sites(load_json(mapping))
        require(isinstance(sites, list) and sites, f"Batch {batch_id:03d} mapping sites")
        for site in sites:
            site_id = str(site["fault_site_id"])
            require(site_id not in lookup, f"duplicate catalog site {site_id}")
            lookup[site_id] = {"batch_id": batch_id, "selector": int(site["selector_code"])}
    return lookup, batch_entries


def select_sites(source_rows: list[dict[str, str]], site_field: str,
                 site_index_field: str | None, catalog: dict[str, dict[str, int]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for source_rank, row in enumerate(source_rows):
        site_id = row[site_field]
        require(site_id in catalog, f"calibration site absent from canonical catalog: {site_id}")
        binding = catalog[site_id]
        site_index = int(row[site_index_field]) if site_index_field is not None and row[site_index_field] != "" else -1
        selection_hash = hashlib.sha256(f"CIRCUITSAGE-12B3J|{site_id}".encode()).hexdigest()
        candidates.append({
            "source_partition_rank": source_rank,
            "site_id": site_id,
            "site_index": site_index,
            "batch_id": binding["batch_id"],
            "selector": binding["selector"],
            "selection_hash": selection_hash,
        })
    by_batch: dict[int, list[dict[str, Any]]] = {batch_id: [] for batch_id in range(BATCHES)}
    for row in candidates:
        by_batch[row["batch_id"]].append(row)
    require(all(by_batch.values()), "REPAIR_CALIBRATION does not cover all canonical batches")
    for rows in by_batch.values():
        rows.sort(key=lambda row: (row["selection_hash"], row["site_id"]))

    target = min(len(candidates), MAX_CALIBRATION_SITES)
    require(target >= BATCHES, "calibration capture budget cannot represent all batches")
    selected = [by_batch[batch_id].pop(0) for batch_id in range(BATCHES)]
    remaining = sorted((row for rows in by_batch.values() for row in rows), key=lambda row: (row["selection_hash"], row["site_id"]))
    selected.extend(remaining[:target - BATCHES])
    selected.sort(key=lambda row: (row["batch_id"], row["selector"], row["site_id"]))
    for rank, row in enumerate(selected):
        row["calibration_capture_rank"] = rank
        row["fault_instances"] = 2
        row["vector_count"] = VECTORS
        row["enabled_transactions"] = 2 * VECTORS
    require(len(selected) == target and len({row["site_id"] for row in selected}) == target, "selected calibration coverage")
    return selected


def verify_inputs() -> tuple[list[dict[str, Any]], int, dict[int, dict[str, Any]]]:
    print("FROZEN INPUT VERIFICATION")
    for path, expected in PINNED.items():
        require(path.is_file(), f"missing frozen input: {rel(path)}")
        require(sha256(path) == expected, f"SHA mismatch: {rel(path)}")
        print(f"  {path.name:<104}: OK")

    audit = load_json(AUDIT_3I)
    architecture = load_json(ARCHITECTURE_3I)
    partition = load_json(PARTITION_CONTRACT_3I)
    training = load_json(TRAINING_CONTRACT_3I)
    require(audit.get("status") == "PASS", "3I audit")
    require(audit.get("selected_measurement") == SELECTED, "3I measurement")
    require(audit.get("training") == "NOT AUTHORIZED / NOT STARTED", "3I training boundary")
    require(audit.get("repair_calibration") == "LOCKED / NOT OPENED", "3I calibration boundary")
    require(audit.get("repair_site_test") == "LOCKED / NOT OPENED", "3I site-test boundary")
    require(audit.get("dev_site_test_validation_holdout_access") == [0, 0, 0], "3I protected access")
    require(architecture.get("measurement_frontend") == SELECTED, "architecture measurement")
    require(architecture.get("query_identity_inputs") == "PROHIBITED", "identity boundary")
    require(partition.get("partitions", {}).get(PARTITION, {}).get("state") == "LOCKED / NOT OPENED", "partition state")
    require(partition.get("partitions", {}).get("REPAIR_SITE_TEST", {}).get("state") == "LOCKED / NOT OPENED", "site-test state")
    require(training.get("training_authorization") == "NOT GRANTED BY THIS CONTRACT", "training authorization boundary")

    with np.load(SELECTED_VECTORS, allow_pickle=False) as archive:
        require(set(archive.files) == {"key_u8", "message_u8", "selection_rank", "selection_round", "vector_index"}, "vector NPZ members")
        require(archive["key_u8"].shape == (VECTORS, 32), "key-vector shape")
        require(archive["message_u8"].shape == (VECTORS, 32), "message-vector shape")
        require(np.asarray(archive["selection_rank"]).tolist() == list(range(VECTORS)), "vector ranks")

    probe = load_json(PROBE_JSON).get("candidates", {}).get(SELECTED)
    require(isinstance(probe, dict), "selected probe candidate")
    require([probe.get("bits"), probe.get("snapshots")] == [PROBE_BITS, SNAPSHOTS], "probe dimensions")
    if probe.get("banks") is not None:
        require(probe.get("banks") == PROBE_BANKS, "probe-bank count")
    with PROBE_CSV.open(newline="", encoding="utf-8") as stream:
        probe_rows = [row for row in csv.DictReader(stream) if row["candidate_id"] == SELECTED]
    require(len(probe_rows) == PROBE_BITS, "probe row count")

    verify_split_chain()
    source_rows, site_field, _partition_field, site_index_field = read_split_rows(REPAIR_SPLIT)
    catalog, batch_entries = catalog_bindings()
    selected = select_sites(source_rows, site_field, site_index_field, catalog)
    require(shutil.which("yosys") is not None and shutil.which("verilator") is not None, "toolchain unavailable")
    free_gib = shutil.disk_usage(ROOT).free // 1024**3
    require(free_gib >= MIN_FREE_GIB, f"less than {MIN_FREE_GIB} GiB free disk")
    print("  3I contracts, split chain, vectors, probe schedule, mappings and toolchain                         : PASS")
    return selected, len(source_rows), batch_entries


def self_test() -> None:
    require(MAX_CALIBRATION_SITES >= BATCHES, "site budget")
    require(PROBE_BITS == PROBE_BANKS * 64, "probe banks")
    require(normalized("repair-calibration") == PARTITION, "partition normalization")
    require(2 * VECTORS == 192, "per-site transaction budget")
    print("Stage 12B-3J self-test: PASS")


def main() -> None:
    require(ROOT.name == "vlsi_fault_detection_v2", f"run from project root, not {ROOT}")
    outputs = (SITES_CSV, PLAN_CSV, SCHEMA, PREFLIGHT, EXECUTION_CONTRACT, AUTHORIZATION, MANIFEST, AUDIT)
    for output in outputs:
        require(not output.exists(), f"Stage {STAGE} output already exists: {rel(output)}")

    selected, partition_sites, batch_entries = verify_inputs()
    timestamp = now()
    selected_sites = len(selected)
    faults = selected_sites * 2
    baseline_records = BATCHES * VECTORS
    enabled_transactions = faults * VECTORS
    total_records = baseline_records + enabled_transactions
    free_gib = shutil.disk_usage(ROOT).free / 1024**3

    site_fields = [
        "calibration_capture_rank", "source_partition_rank", "site_id", "site_index",
        "batch_id", "selector", "fault_instances", "vector_count",
        "enabled_transactions", "selection_hash",
    ]
    frozen_write(SITES_CSV, csv_payload(selected, site_fields))

    plan_rows: list[dict[str, Any]] = []
    for batch_id in range(BATCHES):
        count = sum(row["batch_id"] == batch_id for row in selected)
        require(count > 0, f"empty selected Batch {batch_id:03d}")
        enabled = count * 2 * VECTORS
        plan_rows.append({
            "batch_id": batch_id,
            "site_count": count,
            "fault_instances": count * 2,
            "baseline_records": VECTORS,
            "enabled_transactions": enabled,
            "total_records": VECTORS + enabled,
            "execution_order": batch_id,
        })
    plan_fields = ["batch_id", "site_count", "fault_instances", "baseline_records", "enabled_transactions", "total_records", "execution_order"]
    frozen_write(PLAN_CSV, csv_payload(plan_rows, plan_fields))

    schema = {
        "schema_version": "CIRCUITSAGE-HMAC-V2.1-REPAIR-CALIBRATION-CAPTURE-12B3J-v1",
        "stage": STAGE,
        "status": "FROZEN",
        "partition": PARTITION,
        "source_partition_sites": partition_sites,
        "selected_capture_sites": selected_sites,
        "fault_instances": faults,
        "vectors": VECTORS,
        "measurement": {"candidate": SELECTED, "probe_bits": PROBE_BITS, "banks": PROBE_BANKS, "snapshots": SNAPSHOTS},
        "raw_capture_fields": [
            "batch_id", "run_type", "site_id_control_only", "selector_control_only",
            "stuck_value_scoring_only", "vector_rank", "source_vector_index", "cycles",
            "timed_out", "busy_first_cycle", "busy_last_cycle", "done_cycle", "unknown",
            "expected_digest", "actual_digest", "probe_snapshot", "probe_toggle_count",
        ],
        "future_query_features": [
            "timed_out", "cycle_delta", "control_timeline_delta", "digest_xor",
            "probe_effect", "probe_snapshot_xor", "probe_toggle_delta",
        ],
        "separate_targets": ["site_index", "stuck_value"],
        "forbidden_query_features": ["site_id", "site_index", "selector", "stuck_value", "fault identity", "batch identity"],
        "missing_duplicate_unknown_baseline_failures_allowed": [0, 0, 0, 0],
    }
    frozen_write(SCHEMA, canonical_json(schema))

    preflight = {
        "preflight_version": "CIRCUITSAGE-HMAC-V2.1-REPAIR-CALIBRATION-PREFLIGHT-12B3J-v1",
        "stage": STAGE,
        "status": "PASS",
        "created_at": timestamp,
        "repair_classification": REPAIR_CLASSIFICATION,
        "original_split_csv": record(ORIGINAL_SPLIT),
        "original_split_manifest": record(ORIGINAL_SPLIT_MANIFEST),
        "repair_split_csv": record(REPAIR_SPLIT),
        "stage_12b2a_manifest": record(MANIFEST_2A),
        "stage_12b2a_audit": record(AUDIT_2A),
        "stage_12b2b_manifest": record(MANIFEST_2B),
        "stage_12b2b_audit": record(AUDIT_2B),
        "source_partition_sites": partition_sites,
        "selected_sites": selected_sites,
        "selection_method": "ALL SITES WHEN <=512; OTHERWISE SHA256-DETERMINISTIC BATCH-STRATIFIED 512-SITE COHORT",
        "canonical_batches_represented": BATCHES,
        "duplicate_sites": 0,
        "unknown_catalog_sites": 0,
        "vector_set": record(SELECTED_VECTORS),
        "probe_bank": record(PROBE_JSON),
        "toolchain": {"yosys": shutil.which("yosys"), "verilator": shutil.which("verilator")},
        "available_disk_gib": free_gib,
        "minimum_disk_gib": MIN_FREE_GIB,
        "simulation_calls": 0,
        "training_calls": 0,
        "inference_calls": 0,
    }
    frozen_write(PREFLIGHT, canonical_json(preflight))

    execution_contract = {
        "contract_version": "CIRCUITSAGE-HMAC-V2.1-REPAIR-CALIBRATION-CAPTURE-EXECUTION-12B3J-v1",
        "stage": STAGE,
        "status": "FROZEN",
        "scope": f"BOUNDED {PARTITION} ENHANCED-MEASUREMENT CAPTURE",
        "selected_measurement": SELECTED,
        "selected_sites": selected_sites,
        "fault_instances": faults,
        "vectors": VECTORS,
        "canonical_batches": BATCHES,
        "baseline_records": baseline_records,
        "enabled_transactions": enabled_transactions,
        "total_records": total_records,
        "execution": "SEQUENTIAL",
        "build_jobs": 1,
        "checkpoint_resume": "REQUIRED AFTER EACH CANONICAL BATCH",
        "deterministic_replay": "REQUIRED",
        "maximum_timeout_cycles": 2000,
        "fault_identity_in_query": "PROHIBITED",
        "raw_control_identity": "PERMITTED ONLY INSIDE CAPTURE RUNNER; REMOVED FROM FEATURE NPZ",
        "model_training": "NOT AUTHORIZED",
        "model_inference": "NOT AUTHORIZED",
        "threshold_selection": "NOT AUTHORIZED",
        "repair_site_test": "LOCKED / NOT AUTHORIZED",
        "dev_site_test": "CONSUMED / NOT AUTHORIZED",
        "validation_holdout": "PROHIBITED / PROHIBITED",
    }
    frozen_write(EXECUTION_CONTRACT, canonical_json(execution_contract))

    authorization = {
        "authorization_version": "CIRCUITSAGE-HMAC-V2.1-REPAIR-CALIBRATION-CAPTURE-AUTHORIZATION-12B3J-v1",
        "stage": STAGE,
        "status": "FROZEN",
        "bounded_repair_calibration_capture": "AUTHORIZED / NOT STARTED",
        "selected_measurement": SELECTED,
        "selected_sites_faults_vectors": [selected_sites, faults, VECTORS],
        "maximum_enabled_transactions": enabled_transactions,
        "maximum_total_records": total_records,
        "site_registry": record(SITES_CSV),
        "execution_plan": record(PLAN_CSV),
        "schema": record(SCHEMA),
        "execution_contract": record(EXECUTION_CONTRACT),
        "model_training": "NOT AUTHORIZED",
        "repair_site_test": "NOT AUTHORIZED",
        "validation_holdout": "NOT AUTHORIZED / NOT AUTHORIZED",
    }
    frozen_write(AUTHORIZATION, canonical_json(authorization))

    generated = [SITES_CSV, PLAN_CSV, SCHEMA, PREFLIGHT, EXECUTION_CONTRACT, AUTHORIZATION]
    manifest = {
        "manifest_version": "CIRCUITSAGE-HMAC-V2.1-REPAIR-CALIBRATION-AUTHORIZATION-MANIFEST-12B3J-v1",
        "stage": STAGE,
        "status": "PASS",
        "created_at": timestamp,
        "pinned_inputs": {rel(path): record(path) for path in PINNED},
        "repair_classification": REPAIR_CLASSIFICATION,
        "verified_split_chain": {rel(path): record(path) for path in SPLIT_CHAIN},
        "outputs": {rel(path): record(path) for path in generated},
        "source_partition_sites": partition_sites,
        "selected_capture_sites": selected_sites,
        "fault_instances": faults,
        "vectors": VECTORS,
        "baseline_records": baseline_records,
        "enabled_transactions": enabled_transactions,
        "total_records": total_records,
        "simulation_calls": 0,
        "training_calls": 0,
        "inference_calls": 0,
        "repair_site_test_access": 0,
        "dev_site_test_access": 0,
        "validation_access": 0,
        "holdout_access": 0,
    }
    frozen_write(MANIFEST, canonical_json(manifest))

    audit = {
        "audit_version": "CIRCUITSAGE-HMAC-V2.1-REPAIR-CALIBRATION-AUTHORIZATION-FREEZE-12B3J-v1",
        "stage": STAGE,
        "status": "PASS",
        "authorization_status": "FROZEN",
        "repair_classification": REPAIR_CLASSIFICATION,
        "selected_measurement": SELECTED,
        "source_partition": PARTITION,
        "source_partition_sites": partition_sites,
        "selected_sites_faults_vectors": [selected_sites, faults, VECTORS],
        "canonical_batches": f"{BATCHES}/{BATCHES}",
        "baseline_enabled_total_records": [baseline_records, enabled_transactions, total_records],
        "probe_bits_banks_snapshots": [PROBE_BITS, PROBE_BANKS, SNAPSHOTS],
        "capture_execution": "AUTHORIZED / NOT STARTED",
        "model_training_inference": "NOT AUTHORIZED / NOT AUTHORIZED",
        "repair_site_test": "LOCKED / NOT ACCESSED",
        "dev_site_test": "CONSUMED / NOT REOPENED",
        "validation_holdout_access": [0, 0],
        "v1_v2_core_v2_1_modified": [False, False, False],
        "authorization": record(AUTHORIZATION),
        "manifest": record(MANIFEST),
        "next_gate": "STAGE 12B-3K — ENHANCED REPAIR_CALIBRATION CAPTURE EXECUTION AND DATASET FREEZE",
    }
    frozen_write(AUDIT, canonical_json(audit))

    for path in (SCHEMA, PREFLIGHT, EXECUTION_CONTRACT, AUTHORIZATION, MANIFEST, AUDIT):
        require(canonical_json(load_json(path)) == path.read_bytes(), f"canonical output replay: {rel(path)}")
    require(SITES_CSV.read_bytes() == csv_payload(selected, site_fields), "site registry replay")
    require(PLAN_CSV.read_bytes() == csv_payload(plan_rows, plan_fields), "execution-plan replay")
    for path, expected in PINNED.items():
        require(sha256(path) == expected, f"pinned input modified: {rel(path)}")
    for path in SPLIT_CHAIN:
        require(record(path) == manifest["verified_split_chain"][rel(path)], f"split input modified: {rel(path)}")

    print("\nSTAGE 12B-3J — ENHANCED REPAIR_CALIBRATION CAPTURE AUTHORIZATION FREEZE")
    print(f"{'Status':<70}: PASS")
    print(f"{'Repair classification':<70}: {REPAIR_CLASSIFICATION}")
    print(f"{'Authorization status':<70}: FROZEN")
    print(f"{'Selected measurement':<70}: {SELECTED}")
    print(f"{'Source partition / sites':<70}: {PARTITION} / {partition_sites}")
    print(f"{'Selected sites / faults / vectors':<70}: {selected_sites} / {faults} / {VECTORS}")
    print(f"{'Canonical batches represented':<70}: {BATCHES}/{BATCHES}")
    print(f"{'Baseline / enabled / total records':<70}: {baseline_records} / {enabled_transactions} / {total_records}")
    print(f"{'Probe bits / banks / snapshots':<70}: {PROBE_BITS} / {PROBE_BANKS} / {SNAPSHOTS}")
    print(f"{'Execution / build jobs / checkpoint':<70}: SEQUENTIAL / 1 / REQUIRED")
    print(f"{'Available / minimum disk':<70}: {free_gib:.2f} / {MIN_FREE_GIB} GiB")
    print(f"{'Capture execution':<70}: AUTHORIZED / NOT STARTED")
    print(f"{'Model training / inference':<70}: NOT AUTHORIZED / NOT AUTHORIZED")
    print(f"{'REPAIR_SITE_TEST / DEV_SITE_TEST':<70}: LOCKED / CONSUMED")
    print(f"{'VALIDATION / HOLDOUT access':<70}: 0 / 0")
    print(f"{'Site registry':<70}: {SITES_CSV}")
    print(f"{'Site registry SHA':<70}: {sha256(SITES_CSV)}")
    print(f"{'Execution plan':<70}: {PLAN_CSV}")
    print(f"{'Execution plan SHA':<70}: {sha256(PLAN_CSV)}")
    print(f"{'Execution contract':<70}: {EXECUTION_CONTRACT}")
    print(f"{'Execution contract SHA':<70}: {sha256(EXECUTION_CONTRACT)}")
    print(f"{'Authorization':<70}: {AUTHORIZATION}")
    print(f"{'Authorization SHA':<70}: {sha256(AUTHORIZATION)}")
    print(f"{'Manifest':<70}: {MANIFEST}")
    print(f"{'Manifest SHA':<70}: {sha256(MANIFEST)}")
    print(f"{'Audit':<70}: {AUDIT}")
    print(f"{'Audit SHA':<70}: {sha256(AUDIT)}")
    print(f"{'Next gate':<70}: STAGE 12B-3K — ENHANCED REPAIR_CALIBRATION CAPTURE EXECUTION AND DATASET FREEZE")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    arguments = parser.parse_args()
    if arguments.self_test:
        self_test()
    else:
        main()
