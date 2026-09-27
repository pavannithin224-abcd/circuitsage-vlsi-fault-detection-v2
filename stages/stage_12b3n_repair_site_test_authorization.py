#!/usr/bin/env python3
"""Stage 12B-3N: locked REPAIR_SITE_TEST capture/evaluation authorization.

Verifies the frozen Stage 12B-3M selection, resolves every site in the
pre-existing REPAIR_SITE_TEST partition, binds the selected
EM_TESTPOINT_4X64_T16 measurement and 96-vector set, and freezes a sequential
checkpointed capture plus one-pass evaluation plan.

This authorization stage performs no simulation, response capture, model
deserialization, training, inference, threshold change, candidate reselection,
or access to DEV_SITE_TEST, VALIDATION, or HOLDOUT.  The execution stage must
commit predictions before opening scoring targets and must preserve exact
multi-site ambiguity sets.
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


STAGE = "12B-3N"
ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "config/v2_1_improvement"
RESULT = ROOT / "results/circuitsage_hmac_v2_12b3"
RESULT_2 = ROOT / "results/circuitsage_hmac_v2_12b2"
RESULT_11C5 = ROOT / "results/hmac_fault_campaign_11c5"
WORK = RESULT / "repair_site_test_authorization_12b3n"

# Frozen Stage 12B-3M selection.
SOURCE_3M = ROOT / "stage_12b3m_repair_model_train_select.py"
MODEL_WORK_3M = RESULT / "repair_model_training_12b3m"
CANDIDATE_BUNDLE_3M = MODEL_WORK_3M / "circuitsage_hmac_v2_1_trained_repair_candidates_12b3m.joblib"
METRICS_CSV_3M = MODEL_WORK_3M / "circuitsage_hmac_v2_1_repair_candidate_calibration_metrics_12b3m.csv"
METRICS_JSON_3M = MODEL_WORK_3M / "circuitsage_hmac_v2_1_repair_candidate_calibration_metrics_12b3m.json"
PAIR_DIAGNOSTICS_3M = MODEL_WORK_3M / "circuitsage_hmac_v2_1_repair_pair_diagnostics_12b3m.csv"
CAL_PREDICTIONS_3M = MODEL_WORK_3M / "circuitsage_hmac_v2_1_selected_calibration_predictions_12b3m.npz"
BOOTSTRAP_3M = MODEL_WORK_3M / "circuitsage_hmac_v2_1_selected_calibration_site_bootstrap_12b3m.csv"
SELECTED_MODEL_3M = MODEL_WORK_3M / "circuitsage_hmac_v2_1_selected_repair_model_12b3m.npz"
SELECTED_METADATA_3M = MODEL_WORK_3M / "circuitsage_hmac_v2_1_selected_repair_model_metadata_12b3m.json"
SELECTION_LOCK_3M = MODEL_WORK_3M / "circuitsage_hmac_v2_1_repair_model_selection_lock_12b3m.json"
MANIFEST_3M = RESULT / "circuitsage_hmac_v2_1_repair_model_training_manifest_12b3m.json"
AUDIT_3M = RESULT / "circuitsage_hmac_v2_1_repair_model_training_calibration_selection_freeze_12b3m.json"

# Frozen scientific acceptance and partition contracts.
PARTITION_CONTRACT_3I = CONFIG / "circuitsage_hmac_v2_1_repair_model_data_partition_contract_12b3i.json"
ACCEPTANCE_3I = CONFIG / "circuitsage_hmac_v2_1_repair_model_acceptance_contract_12b3i.json"

# Frozen split, vectors, probes and canonical catalog mappings.
SOURCE_COMMON = ROOT / "stage_12b3d_enhanced_screening_execution.py"
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

# Stage 12B-3N outputs.
SITES_CSV = WORK / "circuitsage_hmac_v2_1_repair_site_test_sites_12b3n.csv"
PLAN_CSV = WORK / "circuitsage_hmac_v2_1_repair_site_test_execution_plan_12b3n.csv"
SCHEMA = WORK / "circuitsage_hmac_v2_1_repair_site_test_capture_evaluation_schema_12b3n.json"
PREFLIGHT = WORK / "circuitsage_hmac_v2_1_repair_site_test_preflight_12b3n.json"
TRUTH_COMMITMENT = WORK / "circuitsage_hmac_v2_1_repair_site_test_truth_commitment_12b3n.json"
EXECUTION_CONTRACT = CONFIG / "circuitsage_hmac_v2_1_repair_site_test_capture_evaluation_contract_12b3n.json"
AUTHORIZATION = CONFIG / "circuitsage_hmac_v2_1_repair_site_test_capture_evaluation_authorization_12b3n.json"
MANIFEST = RESULT / "circuitsage_hmac_v2_1_repair_site_test_authorization_manifest_12b3n.json"
AUDIT = RESULT / "circuitsage_hmac_v2_1_repair_site_test_authorization_freeze_12b3n.json"

PINNED = {
    SOURCE_3M: "e25a27bbe9ac9c282601896d78cc694b0df50c447d696bf575d6d4256e7787a1",
    CANDIDATE_BUNDLE_3M: "39b7f8e38672f6a34da42117ff1850b4ef52fa6c43dd2b679074c992c6d2147d",
    METRICS_CSV_3M: "ff055a7a3b30ad99368bfbc6a8b17194cb8d628fd93eda82181eb1996628793d",
    METRICS_JSON_3M: "1f6b98157e26e3b90e57e51aaa712bcb4427c5200bfa9f3f910d2c3f7b521951",
    PAIR_DIAGNOSTICS_3M: "6c2a86c07e7f8ea6b7a622d941585b25367f051eed06aaa8dbb57fe283e56159",
    CAL_PREDICTIONS_3M: "1536b5fbb7720b27085695f378f0b0f2bf83dd0720bd15f469094bd0d4398e98",
    BOOTSTRAP_3M: "2762fefaf27fc8ec01af5bf6faef3f51813b3a83100a4f3e549e6d9fdea06a44",
    SELECTED_MODEL_3M: "2f1d35c3f2b71f975859a99238d07e1e0d8620f02d79f3808ca1e5c2ccf6f39a",
    SELECTED_METADATA_3M: "8e80d3e3cfc6bfb0c76698150a24097a254f908f566e333d2dcc70d67afcc247",
    SELECTION_LOCK_3M: "7b9cb7d6c4dd9e8123fdfc3b327bcd6213b827d6338cdc7bd0326f739704bf53",
    MANIFEST_3M: "b6d2f8820dc6fba71ae1ad48a42c817d8dec83f29cea906b9122af71ccbbe95e",
    AUDIT_3M: "6774985c88180e85e6146b0ccee9170b28880fa5093b533892a76cc75c958581",
    PARTITION_CONTRACT_3I: "44420dd5fb37cdca9a8207920d69fd5e3bfee6847a4f63db34b91bae54ece562",
    ACCEPTANCE_3I: "ca95bc7fc7587e995f48b0efa03390734192fd251cd82ebd25387c5663d1e0d2",
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

SELECTED_MEASUREMENT = "EM_TESTPOINT_4X64_T16"
SELECTED_MODEL_ID = "R31_EXACT_SIGNATURE_SET"
PARTITION = "REPAIR_SITE_TEST"
BATCHES = 45
SITES = 2398
FAULTS = 4796
VECTORS = 96
PROBE_BITS = 256
PROBE_BANKS = 4
SNAPSHOTS = 16
BASELINE_RECORDS = BATCHES * VECTORS
ENABLED_RECORDS = FAULTS * VECTORS
TOTAL_RECORDS = BASELINE_RECORDS + ENABLED_RECORDS
MIN_FREE_GIB = 10
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
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
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
    original_manifest = load_json(ORIGINAL_SPLIT_MANIFEST)
    manifest_2a = load_json(MANIFEST_2A)
    audit_2a = load_json(AUDIT_2A)
    manifest_2b = load_json(MANIFEST_2B)
    audit_2b = load_json(AUDIT_2B)
    require(original_manifest.get("status") == "PASS", "original split manifest")
    require(ORIGINAL_SPLIT.name in json.dumps(original_manifest, sort_keys=True), "original split binding")
    require(manifest_2a.get("stage") == "12B-2A" and manifest_2a.get("status") == "PASS", "2A manifest")
    require(record_matches(manifest_2a, ORIGINAL_SPLIT_MANIFEST), "2A original-split lineage")
    require(record_matches(manifest_2a, REPAIR_SPLIT), "2A repair-split lineage")
    require(audit_2a.get("stage") == "12B-2A" and audit_2a.get("status") == "PASS", "2A audit")
    require(record_matches(audit_2a, MANIFEST_2A), "2A manifest binding")
    require(manifest_2b.get("stage") == "12B-2B" and manifest_2b.get("status") == "PASS", "2B manifest")
    require(record_matches(manifest_2b, MANIFEST_2A) and record_matches(manifest_2b, AUDIT_2A), "2B to 2A lineage")
    require(audit_2b.get("stage") == "12B-2B" and audit_2b.get("status") == "PASS", "2B audit")
    require(record_matches(audit_2b, MANIFEST_2B), "2B manifest binding")


def read_site_test_rows() -> tuple[list[dict[str, str]], str, str, str | None]:
    with REPAIR_SPLIT.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames is not None, "repair split header")
        fields = list(reader.fieldnames)
        rows = list(reader)
    partition_field = next((name for name in ("repair_partition", "partition", "split", "subset", "role") if name in fields), None)
    site_field = next((name for name in ("site_id", "fault_site_id", "physical_site_id") if name in fields), None)
    site_index_field = next((name for name in ("site_index", "physical_site_index", "global_site_index") if name in fields), None)
    require(partition_field is not None and site_field is not None, "repair split fields")
    selected = [row for row in rows if normalized(row[partition_field]) in {PARTITION, "SITE_TEST"} or normalized(row[partition_field]).endswith(f"_{PARTITION}")]
    require(len(selected) == SITES, f"expected {SITES} {PARTITION} sites; found {len(selected)}")
    require(len({row[site_field] for row in selected}) == SITES, "duplicate site-test site")
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


def bind_sites(source_rows: list[dict[str, str]], site_field: str,
               site_index_field: str | None, catalog: dict[str, dict[str, int]]) -> list[dict[str, Any]]:
    bound: list[dict[str, Any]] = []
    for source_rank, source in enumerate(source_rows):
        site_id = source[site_field]
        require(site_id in catalog, f"site-test site absent from canonical catalog: {site_id}")
        binding = catalog[site_id]
        site_index = int(source[site_index_field]) if site_index_field and source[site_index_field] != "" else -1
        commitment = hashlib.sha256(f"CIRCUITSAGE-12B3N|{site_id}|{site_index}|{binding['batch_id']}|{binding['selector']}".encode()).hexdigest()
        bound.append({
            "source_partition_rank": source_rank,
            "site_id": site_id,
            "site_index": site_index,
            "batch_id": binding["batch_id"],
            "selector": binding["selector"],
            "fault_instances": 2,
            "vector_count": VECTORS,
            "enabled_transactions": 2 * VECTORS,
            "site_commitment_sha256": commitment,
        })
    require(len(bound) == SITES, "bound site-test size")
    require({row["batch_id"] for row in bound} == set(range(BATCHES)), "site-test canonical batch coverage")
    bound.sort(key=lambda row: (row["batch_id"], row["selector"], row["site_id"]))
    for rank, row in enumerate(bound):
        row["site_test_rank"] = rank
    return bound


def verify_inputs() -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]], dict[str, Any]]:
    print("FROZEN INPUT VERIFICATION")
    for path, expected in PINNED.items():
        require(path.is_file(), f"missing frozen input: {rel(path)}")
        require(sha256(path) == expected, f"SHA mismatch: {rel(path)}")
        print(f"  {path.name:<110}: OK", flush=True)

    metrics = load_json(METRICS_JSON_3M)
    metadata = load_json(SELECTED_METADATA_3M)
    lock = load_json(SELECTION_LOCK_3M)
    manifest = load_json(MANIFEST_3M)
    audit = load_json(AUDIT_3M)
    partition = load_json(PARTITION_CONTRACT_3I)
    acceptance = load_json(ACCEPTANCE_3I)
    require(metrics.get("status") == "FROZEN", "3M metrics freeze")
    require(metrics.get("selected_candidate") == SELECTED_MODEL_ID, "3M selected candidate")
    require(metrics.get("safety_and_coverage_acceptance") == "PASS", "3M safety acceptance")
    require(metadata.get("status") == "FROZEN" and metadata.get("candidate_id") == SELECTED_MODEL_ID, "3M model metadata")
    require(metadata.get("selected_measurement") == SELECTED_MEASUREMENT, "3M selected measurement")
    require(metadata.get("query_identity") == "PROHIBITED / ABSENT", "3M identity boundary")
    require(lock.get("status") == "FROZEN" and lock.get("selected_candidate") == SELECTED_MODEL_ID, "3M selection lock")
    require(lock.get("retraining_after_selection") == "PROHIBITED", "post-selection retraining lock")
    require(lock.get("threshold_change_after_selection") == "PROHIBITED", "post-selection threshold lock")
    require(lock.get("repair_site_test") == "LOCKED / NOT ACCESSED", "3M site-test state")
    require(manifest.get("status") == "PASS" and manifest.get("completed_trainable_runs") == 9, "3M manifest")
    require(manifest.get("repair_site_test_access") == 0, "3M site-test access")
    require(manifest.get("validation_access") == 0 and manifest.get("holdout_access") == 0, "3M protected access")
    require(audit.get("status") == "PASS" and audit.get("training_selection_status") == "FROZEN / FROZEN", "3M audit")
    require(audit.get("selected_candidate") == SELECTED_MODEL_ID, "3M audit selection")
    require(audit.get("safety_and_coverage_acceptance") == "PASS", "3M audit acceptance")
    require(audit.get("repair_site_test") == "LOCKED / NOT ACCESSED", "3M audit site-test")
    require(audit.get("validation_holdout_access") == [0, 0], "3M audit protected access")
    require(partition.get("partitions", {}).get(PARTITION, {}).get("state") == "LOCKED / NOT OPENED", "partition contract site-test state")
    require(acceptance.get("status") == "FROZEN", "acceptance contract")

    verify_split_chain()
    source_rows, site_field, _partition_field, site_index_field = read_site_test_rows()
    catalog, batch_entries = catalog_bindings()
    bound = bind_sites(source_rows, site_field, site_index_field, catalog)

    with np.load(SELECTED_VECTORS, allow_pickle=False) as archive:
        require(set(archive.files) == {"key_u8", "message_u8", "selection_rank", "selection_round", "vector_index"}, "vector NPZ members")
        require(archive["key_u8"].shape == (VECTORS, 32), "key-vector shape")
        require(archive["message_u8"].shape == (VECTORS, 32), "message-vector shape")
        require(np.asarray(archive["selection_rank"]).tolist() == list(range(VECTORS)), "vector ranks")
    probe = load_json(PROBE_JSON).get("candidates", {}).get(SELECTED_MEASUREMENT)
    require(isinstance(probe, dict), "selected probe candidate")
    require([probe.get("bits"), probe.get("snapshots")] == [PROBE_BITS, SNAPSHOTS], "probe dimensions")
    with PROBE_CSV.open(newline="", encoding="utf-8") as stream:
        probe_rows = [row for row in csv.DictReader(stream) if row["candidate_id"] == SELECTED_MEASUREMENT]
    require(len(probe_rows) == PROBE_BITS, "probe row count")
    require(shutil.which("yosys") is not None and shutil.which("verilator") is not None, "toolchain unavailable")
    require(shutil.disk_usage(ROOT).free / 1024**3 >= MIN_FREE_GIB, f"less than {MIN_FREE_GIB} GiB free disk")
    print("  Selection lock, split chain, full site-test cohort, vectors, probes, mappings and toolchain       : PASS")
    return bound, batch_entries, acceptance


def self_test() -> None:
    require(SITES * 2 == FAULTS, "site/fault arithmetic")
    require(BASELINE_RECORDS == 4320, "baseline record count")
    require(ENABLED_RECORDS == 460416 and TOTAL_RECORDS == 464736, "enabled/total record counts")
    require(PROBE_BITS == PROBE_BANKS * 64, "probe banks")
    require(normalized("repair-site-test") == PARTITION, "partition normalization")
    print("Stage 12B-3N self-test: PASS")


def main() -> None:
    require(ROOT.name == "vlsi_fault_detection_v2", f"run from project root, not {ROOT}")
    outputs = (SITES_CSV, PLAN_CSV, SCHEMA, PREFLIGHT, TRUTH_COMMITMENT,
               EXECUTION_CONTRACT, AUTHORIZATION, MANIFEST, AUDIT)
    for output in outputs:
        require(not output.exists(), f"Stage {STAGE} output already exists: {rel(output)}")

    sites, batch_entries, acceptance = verify_inputs()
    timestamp = now()
    free_gib = shutil.disk_usage(ROOT).free / 1024**3
    site_fields = [
        "site_test_rank", "source_partition_rank", "site_id", "site_index",
        "batch_id", "selector", "fault_instances", "vector_count",
        "enabled_transactions", "site_commitment_sha256",
    ]
    site_payload = csv_payload(sites, site_fields)
    frozen_write(SITES_CSV, site_payload)

    plan_rows: list[dict[str, Any]] = []
    for batch_id in range(BATCHES):
        site_count = sum(row["batch_id"] == batch_id for row in sites)
        require(site_count > 0, f"empty site-test Batch {batch_id:03d}")
        enabled = site_count * 2 * VECTORS
        plan_rows.append({
            "batch_id": batch_id,
            "site_count": site_count,
            "fault_instances": site_count * 2,
            "baseline_records": VECTORS,
            "enabled_transactions": enabled,
            "total_records": VECTORS + enabled,
            "execution_order": batch_id,
        })
    require(sum(row["site_count"] for row in plan_rows) == SITES, "execution-plan sites")
    require(sum(row["enabled_transactions"] for row in plan_rows) == ENABLED_RECORDS, "execution-plan transactions")
    plan_fields = ["batch_id", "site_count", "fault_instances", "baseline_records", "enabled_transactions", "total_records", "execution_order"]
    frozen_write(PLAN_CSV, csv_payload(plan_rows, plan_fields))

    truth_payload = "\n".join(
        f"{row['site_test_rank']}|{row['site_id']}|{row['site_index']}|{row['batch_id']}|{row['selector']}|SA0|SA1"
        for row in sites
    ).encode() + b"\n"
    truth_commitment = {
        "commitment_version": "CIRCUITSAGE-HMAC-V2.1-REPAIR-SITE-TEST-TRUTH-COMMITMENT-12B3N-v1",
        "stage": STAGE,
        "status": "FROZEN",
        "created_at": timestamp,
        "partition": PARTITION,
        "sites": SITES,
        "fault_instances": FAULTS,
        "commitment_sha256": hashlib.sha256(truth_payload).hexdigest(),
        "commitment_scope": "ORDERED SITE/Fault CONTROL TRUTH; NOT A QUERY FEATURE",
        "opening_rule": "PREDICTION COMMITMENT MUST BE FROZEN BEFORE SCORING",
    }
    frozen_write(TRUTH_COMMITMENT, canonical_json(truth_commitment))

    targets = acceptance["repair_calibration_targets"]
    invariants = acceptance["measurement_invariants"]
    schema = {
        "schema_version": "CIRCUITSAGE-HMAC-V2.1-REPAIR-SITE-TEST-CAPTURE-EVALUATION-12B3N-v1",
        "stage": STAGE,
        "status": "FROZEN",
        "partition": PARTITION,
        "sites": SITES,
        "fault_instances": FAULTS,
        "vectors": VECTORS,
        "measurement": {"candidate": SELECTED_MEASUREMENT, "probe_bits": PROBE_BITS, "banks": PROBE_BANKS, "snapshots": SNAPSHOTS},
        "selected_model": SELECTED_MODEL_ID,
        "capture_feature_fields": [
            "timed_out", "cycle_delta", "control_timeline_delta", "digest_xor",
            "external_detected", "probe_effect", "probe_snapshot_xor", "probe_toggle_delta",
        ],
        "separate_scoring_targets": ["site_index", "stuck_value"],
        "forbidden_query_fields": ["fault_instance_index", "site_id", "site_index", "selector", "stuck_value", "batch identity", "raw fault control"],
        "prediction_outputs": [
            "NO_OBSERVED_ANOMALY", "UNIQUE_SITE_IN_CATALOG",
            "AMBIGUOUS_SITES_IN_CATALOG", "NO_CATALOG_MATCH",
        ],
        "ambiguity_rule": "RETURN EVERY SITE WITH THE EXACT COMPLETE SIGNATURE; NEVER FALSELY COLLAPSE A TIE",
    }
    frozen_write(SCHEMA, canonical_json(schema))

    preflight = {
        "preflight_version": "CIRCUITSAGE-HMAC-V2.1-REPAIR-SITE-TEST-PREFLIGHT-12B3N-v1",
        "stage": STAGE,
        "status": "PASS",
        "created_at": timestamp,
        "source_partition_sites": SITES,
        "selected_sites": SITES,
        "selection_method": "ALL FROZEN REPAIR_SITE_TEST SITES; NO SUBSAMPLING",
        "canonical_batches_represented": BATCHES,
        "duplicate_sites": 0,
        "unknown_catalog_sites": 0,
        "selected_model": record(SELECTED_MODEL_3M),
        "selection_lock": record(SELECTION_LOCK_3M),
        "vector_set": record(SELECTED_VECTORS),
        "probe_bank": record(PROBE_JSON),
        "repair_split": record(REPAIR_SPLIT),
        "available_disk_gib": round(free_gib, 2),
        "minimum_disk_gib": MIN_FREE_GIB,
        "toolchain": {"yosys": shutil.which("yosys"), "verilator": shutil.which("verilator")},
        "simulation_calls": 0,
        "training_calls": 0,
        "inference_calls": 0,
        "model_objects_deserialized": 0,
    }
    frozen_write(PREFLIGHT, canonical_json(preflight))

    execution_contract = {
        "contract_version": "CIRCUITSAGE-HMAC-V2.1-LOCKED-REPAIR-SITE-TEST-EXECUTION-12B3N-v1",
        "stage": STAGE,
        "status": "FROZEN",
        "scope": "ONE LOCKED CLOSED-CATALOG REPAIR_SITE_TEST CAPTURE AND EVALUATION",
        "selected_measurement": SELECTED_MEASUREMENT,
        "selected_model": SELECTED_MODEL_ID,
        "sites_faults_vectors": [SITES, FAULTS, VECTORS],
        "canonical_batches": BATCHES,
        "baseline_enabled_total_records": [BASELINE_RECORDS, ENABLED_RECORDS, TOTAL_RECORDS],
        "execution": "SEQUENTIAL",
        "build_jobs": 1,
        "checkpoint_resume": "REQUIRED AFTER EACH CANONICAL BATCH",
        "maximum_timeout_cycles": 2000,
        "capture_replay": "REQUIRED / EXACT",
        "prediction_commitment_before_truth": "REQUIRED",
        "scoring_replay": "REQUIRED / EXACT",
        "query_fault_identity": "PROHIBITED",
        "raw_control_identity": "PERMITTED ONLY INSIDE CAPTURE WORKER; REMOVED BEFORE DIAGNOSIS",
        "exact_signature_precedence": True,
        "ambiguity_preservation": "MANDATORY",
        "frozen_acceptance_targets": {
            "fault_free_false_alarm_rate_max": invariants["fault_free_false_alarm_rate_max"],
            "combined_detection_recall_min": invariants["combined_all_injected_detection_recall_min"],
            **targets,
        },
        "bootstrap": "1000 PHYSICAL-SITE REPLICATES / 95% INTERVALS",
        "training_scaler_fit_threshold_change_candidate_reselection": "PROHIBITED / PROHIBITED / PROHIBITED / PROHIBITED",
        "repair_site_test_use": "ONE CAPTURE AND ONE FINAL LOCKED EVALUATION",
        "dev_site_test_validation_holdout": "PROHIBITED / PROHIBITED / PROHIBITED",
        "scientific_scope": "CLOSED-CATALOG SITE-HELD-OUT CONSISTENCY; NOT INDEPENDENT-CIRCUIT GENERALIZATION",
    }
    frozen_write(EXECUTION_CONTRACT, canonical_json(execution_contract))

    authorization = {
        "authorization_version": "CIRCUITSAGE-HMAC-V2.1-REPAIR-SITE-TEST-CAPTURE-EVALUATION-AUTHORIZATION-12B3N-v1",
        "stage": STAGE,
        "status": "FROZEN",
        "locked_capture_and_evaluation": "AUTHORIZED / NOT STARTED",
        "selected_measurement": SELECTED_MEASUREMENT,
        "selected_model": SELECTED_MODEL_ID,
        "sites_faults_vectors": [SITES, FAULTS, VECTORS],
        "maximum_enabled_transactions": ENABLED_RECORDS,
        "maximum_total_records": TOTAL_RECORDS,
        "site_registry": record(SITES_CSV),
        "execution_plan": record(PLAN_CSV),
        "schema": record(SCHEMA),
        "truth_commitment": record(TRUTH_COMMITMENT),
        "execution_contract": record(EXECUTION_CONTRACT),
        "training": "NOT AUTHORIZED",
        "threshold_change": "NOT AUTHORIZED",
        "candidate_reselection": "NOT AUTHORIZED",
        "validation_holdout": "NOT AUTHORIZED / NOT AUTHORIZED",
    }
    frozen_write(AUTHORIZATION, canonical_json(authorization))

    generated = [SITES_CSV, PLAN_CSV, SCHEMA, PREFLIGHT, TRUTH_COMMITMENT, EXECUTION_CONTRACT, AUTHORIZATION]
    manifest = {
        "manifest_version": "CIRCUITSAGE-HMAC-V2.1-REPAIR-SITE-TEST-AUTHORIZATION-MANIFEST-12B3N-v1",
        "stage": STAGE,
        "status": "PASS",
        "created_at": timestamp,
        "pinned_inputs": {rel(path): record(path) for path in PINNED},
        "verified_split_chain": {rel(path): record(path) for path in SPLIT_CHAIN},
        "outputs": {rel(path): record(path) for path in generated},
        "sites": SITES,
        "fault_instances": FAULTS,
        "vectors": VECTORS,
        "baseline_records": BASELINE_RECORDS,
        "enabled_transactions": ENABLED_RECORDS,
        "total_records": TOTAL_RECORDS,
        "simulation_calls": 0,
        "training_calls": 0,
        "inference_calls": 0,
        "model_objects_deserialized": 0,
        "repair_site_test_rows_accessed_by_authorization": 0,
        "dev_site_test_access": 0,
        "validation_access": 0,
        "holdout_access": 0,
    }
    frozen_write(MANIFEST, canonical_json(manifest))

    audit = {
        "audit_version": "CIRCUITSAGE-HMAC-V2.1-REPAIR-SITE-TEST-AUTHORIZATION-FREEZE-12B3N-v1",
        "stage": STAGE,
        "status": "PASS",
        "authorization_status": "FROZEN",
        "selected_measurement": SELECTED_MEASUREMENT,
        "selected_model": SELECTED_MODEL_ID,
        "selection_lock": "VERIFIED / FROZEN",
        "source_partition": PARTITION,
        "sites_faults_vectors": [SITES, FAULTS, VECTORS],
        "canonical_batches": f"{BATCHES}/{BATCHES}",
        "baseline_enabled_total_records": [BASELINE_RECORDS, ENABLED_RECORDS, TOTAL_RECORDS],
        "probe_bits_banks_snapshots": [PROBE_BITS, PROBE_BANKS, SNAPSHOTS],
        "capture_evaluation": "AUTHORIZED / NOT STARTED",
        "prediction_before_truth_commitment": "REQUIRED",
        "training_threshold_change_candidate_reselection": "0 / 0 / 0",
        "model_objects_deserialized": 0,
        "repair_site_test_state": "AUTHORIZED / NOT OPENED",
        "dev_site_test": "CONSUMED / NOT REOPENED",
        "validation_holdout_access": [0, 0],
        "v1_v2_core_v2_1_modified": [False, False, False],
        "authorization": record(AUTHORIZATION),
        "manifest": record(MANIFEST),
        "next_gate": "STAGE 12B-3O — LOCKED REPAIR_SITE_TEST CAPTURE, EVALUATION, AND DATASET FREEZE",
    }
    frozen_write(AUDIT, canonical_json(audit))

    for path in (SCHEMA, PREFLIGHT, TRUTH_COMMITMENT, EXECUTION_CONTRACT, AUTHORIZATION, MANIFEST, AUDIT):
        require(canonical_json(load_json(path)) == path.read_bytes(), f"canonical output replay: {rel(path)}")
    require(SITES_CSV.read_bytes() == site_payload, "site registry replay")
    require(PLAN_CSV.read_bytes() == csv_payload(plan_rows, plan_fields), "execution-plan replay")
    for path, expected in PINNED.items():
        require(sha256(path) == expected, f"pinned input modified: {rel(path)}")
    for path in SPLIT_CHAIN:
        require(record(path) == manifest["verified_split_chain"][rel(path)], f"split input modified: {rel(path)}")

    print("\nSTAGE 12B-3N — LOCKED REPAIR_SITE_TEST CAPTURE AND EVALUATION AUTHORIZATION FREEZE")
    print(f"{'Status':<76}: PASS")
    print(f"{'Authorization status':<76}: FROZEN")
    print(f"{'Selected measurement / model':<76}: {SELECTED_MEASUREMENT} / {SELECTED_MODEL_ID}")
    print(f"{'Source partition':<76}: {PARTITION}")
    print(f"{'Sites / faults / vectors':<76}: {SITES} / {FAULTS} / {VECTORS}")
    print(f"{'Canonical batches represented':<76}: {BATCHES}/{BATCHES}")
    print(f"{'Baseline / enabled / total records':<76}: {BASELINE_RECORDS} / {ENABLED_RECORDS} / {TOTAL_RECORDS}")
    print(f"{'Probe bits / banks / snapshots':<76}: {PROBE_BITS} / {PROBE_BANKS} / {SNAPSHOTS}")
    print(f"{'Execution / build jobs / checkpoint':<76}: SEQUENTIAL / 1 / REQUIRED")
    print(f"{'Available / minimum disk':<76}: {free_gib:.2f} / {MIN_FREE_GIB} GiB")
    print(f"{'Capture and one locked evaluation':<76}: AUTHORIZED / NOT STARTED")
    print(f"{'Training / threshold changes / candidate reselection':<76}: NOT AUTHORIZED / NOT AUTHORIZED / NOT AUTHORIZED")
    print(f"{'Prediction commitment before scoring truth':<76}: REQUIRED")
    print(f"{'Model objects deserialized / inference calls':<76}: 0 / 0")
    print(f"{'DEV_SITE_TEST / VALIDATION / HOLDOUT access':<76}: 0 / 0 / 0")
    print(f"{'Site registry':<76}: {SITES_CSV}")
    print(f"{'Site registry SHA':<76}: {sha256(SITES_CSV)}")
    print(f"{'Execution plan':<76}: {PLAN_CSV}")
    print(f"{'Execution plan SHA':<76}: {sha256(PLAN_CSV)}")
    print(f"{'Truth commitment':<76}: {TRUTH_COMMITMENT}")
    print(f"{'Truth commitment SHA':<76}: {sha256(TRUTH_COMMITMENT)}")
    print(f"{'Execution contract':<76}: {EXECUTION_CONTRACT}")
    print(f"{'Execution contract SHA':<76}: {sha256(EXECUTION_CONTRACT)}")
    print(f"{'Authorization':<76}: {AUTHORIZATION}")
    print(f"{'Authorization SHA':<76}: {sha256(AUTHORIZATION)}")
    print(f"{'Manifest':<76}: {MANIFEST}")
    print(f"{'Manifest SHA':<76}: {sha256(MANIFEST)}")
    print(f"{'Audit':<76}: {AUDIT}")
    print(f"{'Audit SHA':<76}: {sha256(AUDIT)}")
    print(f"{'Next gate':<76}: STAGE 12B-3O — LOCKED REPAIR_SITE_TEST CAPTURE, EVALUATION, AND DATASET FREEZE")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    arguments = parser.parse_args()
    if arguments.self_test:
        self_test()
    else:
        main()
