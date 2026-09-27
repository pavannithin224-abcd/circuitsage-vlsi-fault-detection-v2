#!/usr/bin/env python3
"""Stage 12C-1K: full TRAIN/CALIBRATION campaign contract and authorization.

Verifies the frozen bounded-pilot disposition, full vector corpus, synthesized
netlists and original campaign contracts.  It then freezes a deterministic,
resource-bounded, resumable execution plan for the complete TRAIN/CALIBRATION
SA0/SA1 campaign.  No simulation, fault injection, response generation,
dataset construction, model loading, training or inference occurs here.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import io
import json
import math
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STAGE = "12C-1K"
ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "config/v2_2"
RESULT = ROOT / "results/circuitsage_hmac_v2_12c1"
WORK = RESULT / "full_campaign_authorization_12c1k"
LOCK_FILE = WORK / ".stage_12c1k.lock"

SOURCE_1J = ROOT / "stage_12c1j_pilot_disposition_full_campaign_readiness.py"
POLICY_1J = CONFIG / "circuitsage_hmac_v2_2_bounded_pilot_disposition_policy_12c1j.json"
READINESS_1J = CONFIG / "circuitsage_hmac_v2_2_full_campaign_readiness_contract_12c1j.json"
MANIFEST_1J = RESULT / "circuitsage_hmac_v2_2_pilot_disposition_manifest_12c1j.json"
AUDIT_1J = RESULT / "circuitsage_hmac_v2_2_pilot_disposition_full_campaign_readiness_freeze_12c1j.json"

VECTOR_CONTRACT_1F = CONFIG / "circuitsage_hmac_v2_2_transaction_vector_contract_12c1f.json"
CAMPAIGN_CONTRACT_1F = CONFIG / "circuitsage_hmac_v2_2_fault_campaign_contract_12c1f.json"
RESPONSE_SCHEMA_1F = CONFIG / "circuitsage_hmac_v2_2_response_schema_contract_12c1f.json"
ACCESS_POLICY_1F = CONFIG / "circuitsage_hmac_v2_2_campaign_partition_access_policy_12c1f.json"
CAMPAIGN_BUDGET_1F = RESULT / "transaction_fault_campaign_contract_12c1f/circuitsage_hmac_v2_2_campaign_budget_registry_12c1f.csv"

WORK_1G = RESULT / "portable_adapter_vector_generation_12c1g"
VECTORS_1G = WORK_1G / "circuitsage_hmac_v2_2_deterministic_transaction_vectors_12c1g.npz"
MANIFEST_1G = RESULT / "circuitsage_hmac_v2_2_portable_adapter_vector_manifest_12c1g.json"
AUDIT_1G = RESULT / "circuitsage_hmac_v2_2_portable_adapter_vector_generation_freeze_12c1g.json"
MANIFEST_1E = RESULT / "circuitsage_hmac_v2_2_elaboration_synthesis_manifest_12c1e.json"
AUDIT_1E = RESULT / "circuitsage_hmac_v2_2_elaboration_wrapper_generic_synthesis_freeze_12c1e.json"
AUDIT_1I = RESULT / "circuitsage_hmac_v2_2_adapter_functional_validation_pilot_execution_freeze_12c1i.json"

EXECUTION_CONTRACT = CONFIG / "circuitsage_hmac_v2_2_full_campaign_execution_contract_12c1k.json"
AUTHORIZATION = CONFIG / "circuitsage_hmac_v2_2_full_campaign_execution_authorization_12c1k.json"
DATASET_SCHEMA = CONFIG / "circuitsage_hmac_v2_2_full_campaign_dataset_schema_12c1k.json"
FAMILY_BUDGET_CSV = WORK / "circuitsage_hmac_v2_2_full_campaign_family_budget_12c1k.csv"
FAMILY_BUDGET_JSON = WORK / "circuitsage_hmac_v2_2_full_campaign_family_budget_12c1k.json"
EXECUTION_PLAN = WORK / "circuitsage_hmac_v2_2_full_campaign_execution_plan_12c1k.csv"
PREFLIGHT = WORK / "circuitsage_hmac_v2_2_full_campaign_preflight_12c1k.json"
REPORT = WORK / "circuitsage_hmac_v2_2_full_campaign_authorization_report_12c1k.md"
MANIFEST = RESULT / "circuitsage_hmac_v2_2_full_campaign_authorization_manifest_12c1k.json"
AUDIT = RESULT / "circuitsage_hmac_v2_2_full_campaign_execution_authorization_freeze_12c1k.json"

PINNED = {
    SOURCE_1J: "d201f742f1fe3629d062f1b6c472b810f8d9133fe7f0da4caff3cb1b019a1424",
    POLICY_1J: "8a62653792496a926e2348e1f49b772c8a2a6fbac04df643dc82da1c3338a951",
    READINESS_1J: "d7c01c90f03438dd417289a4ca5094463bfc8221f22535ba46df021012e9ea18",
    MANIFEST_1J: "3c75270f372268c670d68a439c211365a0489f92375c7def7d5b3e66223d1e20",
    AUDIT_1J: "fce2becb5bb5e4cb36f223ef64be31d2cf7d0e49cd9d112b3b6a97fc1260e36e",
    VECTOR_CONTRACT_1F: "530f6d5b7264990e02513f06bb2192fb88c1bae22f4d2b0496dba691be8c1826",
    CAMPAIGN_CONTRACT_1F: "5e3590b376e4b61a18ae802bcc73a4a17fd5978a94b5aa045aa947d793280d85",
    RESPONSE_SCHEMA_1F: "e433599a185823808fd02e6f9b792f148caf75d2f98c77f2061620f046945f2e",
    ACCESS_POLICY_1F: "107bb7b9c09e8f77c792883b2a6215e3fafed909089152e2bc4d8ffb2a095dc5",
    CAMPAIGN_BUDGET_1F: "46a5fe2abb62051c7aa2daaf2c8503acd7fceacb29b907b5a6a5d8a56663d7c1",
    VECTORS_1G: "0d8fce6c73c5d484fc9880ef97c0663b6dbfc9cbda73fcbd413f1970f418fa7b",
    MANIFEST_1G: "a987b1c13767143a2ffdf0561a93ce352615e5bbe4c81cc77a4a2801e62f108e",
    AUDIT_1G: "4f079aa98c97751bdd2215104df9839b1523b4a4a65df1bc3a6fedbd6e7fd82d",
    MANIFEST_1E: "4d3d606e174ae06589e0e537486a15e042521fe9a02d49f9b4e1c596198476b6",
    AUDIT_1E: "3ac4f454173166ee6848e2617c7072742665e7166345ce5ffdb837f574d59220",
    AUDIT_1I: "cc96e5d7d11f9f75658365875c6521789416399b7bf09471f0fab334f09b2fa7",
}

FAMILIES = (
    "opentitan_hmac_sha256",
    "picorv32_cpu",
    "secworks_aes",
    "secworks_sha256",
)
PARTITIONS = {
    "opentitan_hmac_sha256": "GENERALIZATION_TRAIN",
    "picorv32_cpu": "GENERALIZATION_TRAIN",
    "secworks_aes": "GENERALIZATION_TRAIN",
    "secworks_sha256": "GENERALIZATION_CALIBRATION",
}
BATCH_SITES = 64
TOTAL_GENERIC_SITES = 63749
TOTAL_MAXIMUM_FAULTS = 127498
TOTAL_VECTORS = 240
MIN_FREE_GIB = 40
FUTURE_BRAND = "Faultiva 1.0 — Hybrid VLSI Fault Intelligence"


def stop(message: str) -> None:
    raise SystemExit(f"STOP: {message}")


def require(condition: bool, message: str) -> None:
    if not condition:
        stop(message)


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def record(path: Path) -> dict[str, Any]:
    return {"path": rel(path), "sha256": sha256(path), "bytes": path.stat().st_size}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def csv_bytes(rows: list[dict[str, Any]], fields: list[str]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def frozen_write(path: Path, payload: bytes) -> None:
    require(not path.exists(), f"refusing to overwrite frozen output: {rel(path)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def safe_record(item: dict[str, Any], label: str) -> Path:
    require(isinstance(item, dict), f"{label} record")
    raw = item.get("path")
    require(isinstance(raw, str) and raw, f"{label} path")
    path = (ROOT / raw).resolve()
    require(path.is_relative_to(ROOT), f"{label} path escapes project root")
    require(path.is_file(), f"missing {label}: {raw}")
    require(sha256(path) == item.get("sha256"), f"{label} SHA")
    require(path.stat().st_size == int(item.get("bytes", -1)), f"{label} size")
    return path


def verify_manifest_outputs(manifest: dict[str, Any], label: str, minimum: int) -> None:
    outputs = manifest.get("outputs")
    require(isinstance(outputs, dict) and len(outputs) >= minimum, f"{label} outputs")
    for name, item in sorted(outputs.items()):
        safe_record(item, f"{label} output {name}")


def verify_inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    print("FROZEN INPUT VERIFICATION", flush=True)
    for path, expected in PINNED.items():
        require(path.is_file(), f"missing frozen input: {rel(path)}")
        require(sha256(path) == expected, f"frozen input SHA: {path.name}")
        print(f"  {path.name:<110}: OK", flush=True)

    policy = load_json(POLICY_1J)
    readiness = load_json(READINESS_1J)
    manifest_1j = load_json(MANIFEST_1J)
    audit_1j = load_json(AUDIT_1J)
    vectors = load_json(VECTOR_CONTRACT_1F)
    campaign = load_json(CAMPAIGN_CONTRACT_1F)
    response = load_json(RESPONSE_SCHEMA_1F)
    access = load_json(ACCESS_POLICY_1F)
    manifest_1g = load_json(MANIFEST_1G)
    audit_1g = load_json(AUDIT_1G)
    manifest_1e = load_json(MANIFEST_1E)
    audit_1e = load_json(AUDIT_1E)
    audit_1i = load_json(AUDIT_1I)

    for path, value in ((POLICY_1J, policy), (READINESS_1J, readiness),
                        (MANIFEST_1J, manifest_1j), (AUDIT_1J, audit_1j)):
        require(path.read_bytes() == canonical_json(value), f"canonical immediate predecessor: {path.name}")

    require(manifest_1j.get("status") == "PASS" and audit_1j.get("status") == "PASS", "12C-1J freeze")
    require(manifest_1j.get("stage_source", {}).get("sha256") == PINNED[SOURCE_1J], "12C-1J stage-source anchor")
    require(audit_1j.get("manifest_record", {}).get("sha256") == PINNED[MANIFEST_1J], "12C-1J manifest anchor")
    require(audit_1j.get("pilot_advancement", "").startswith("CONDITIONAL PASS"), "pilot advancement")
    require(audit_1j.get("full_campaign_readiness") == "READY FOR SEPARATE EXECUTION-CONTRACT AND AUTHORIZATION FREEZE", "full-campaign readiness")
    require(audit_1j.get("full_campaign_execution_model_training") == "NOT AUTHORIZED / NOT AUTHORIZED", "predecessor execution boundary")
    require(audit_1j.get("independent_test_validation_holdout_access") == [0, 0, 0], "12C-1J protected access")
    verify_manifest_outputs(manifest_1j, "12C-1J", 6)

    require(vectors.get("status") == "FROZEN" and campaign.get("status") == "FROZEN", "12C-1F campaign contracts")
    require(campaign.get("fault_model") == "SINGLE PERSISTENT CELL-OUTPUT SA0/SA1", "fault model")
    require(campaign.get("execution") == "NOT AUTHORIZED", "original campaign boundary")
    require(response.get("status") == "FROZEN", "response schema")
    require("fault_id" in response.get("prohibited_model_input_fields", []), "identity firewall")
    require(access.get("INDEPENDENT_CIRCUIT_TEST", "").startswith("LOCKED"), "independent TEST lock")
    require(access.get("GENERALIZATION_HOLDOUT", "").startswith("SEALED"), "HOLDOUT seal")
    require(access.get("validation_access") == 0, "VALIDATION access")

    require(manifest_1g.get("status") == "PASS" and audit_1g.get("status") == "PASS", "12C-1G freeze")
    require(audit_1g.get("vector_generation") == "COMPLETED / FROZEN / 240 TOTAL / 60 PILOT", "full vectors")
    require(audit_1g.get("fault_identity_in_vectors") == "ABSENT / PROHIBITED", "vector identity firewall")
    require(audit_1g.get("vectors_record", {}).get("sha256") == PINNED[VECTORS_1G], "vector audit anchor")
    verify_manifest_outputs(manifest_1g, "12C-1G", 10)

    require(manifest_1e.get("status") == "PASS" and audit_1e.get("status") == "PASS", "12C-1E synthesis freeze")
    require(audit_1e.get("deterministic_replay") == "PASS / BYTE-EXACT / 4 OF 4", "netlist replay")
    verify_manifest_outputs(manifest_1e, "12C-1E", 4)
    require(audit_1i.get("adapter_functional_validation") == "PASS / 4 OF 4 / BYTE-EXACT REPLAY", "adapter validation")
    require(audit_1i.get("dataset_integrity") == "PASS", "pilot integrity")
    require(audit_1i.get("independent_test_validation_holdout_access") == [0, 0, 0], "12C-1I protected access")

    budget_rows = read_csv(CAMPAIGN_BUDGET_1F)
    require([row["family_id"] for row in budget_rows] == list(FAMILIES), "campaign budget family order")
    require([item.get("family_id") for item in campaign.get("campaigns", [])] == list(FAMILIES), "campaign family order")
    require([item.get("family_id") for item in vectors.get("budgets", [])] == list(FAMILIES), "vector family order")
    require(sum(int(row["generic_cells"]) for row in budget_rows) == TOTAL_GENERIC_SITES, "total generic sites")
    require(sum(int(row["maximum_catalog_faults"]) for row in budget_rows) == TOTAL_MAXIMUM_FAULTS, "maximum fault budget")
    require(sum(int(item["total_vector_budget"]) for item in vectors["budgets"]) == TOTAL_VECTORS, "total vector budget")

    print("  Pilot disposition, full vectors, adapters, netlists, campaign bounds and partition locks          : PASS", flush=True)
    return vectors, campaign


def build_plan(vectors: dict[str, Any], campaign: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    vector_map = {item["family_id"]: item for item in vectors["budgets"]}
    campaign_map = {item["family_id"]: item for item in campaign["campaigns"]}
    family_rows: list[dict[str, Any]] = []
    plan_rows: list[dict[str, Any]] = []
    global_batch = 0
    global_site_offset = 0
    global_fault_offset = 0
    for family_id in FAMILIES:
        item = campaign_map[family_id]
        vector_count = int(vector_map[family_id]["total_vector_budget"])
        maximum_sites = int(item["generic_cells"])
        maximum_faults = int(item["maximum_catalog_faults"])
        require(maximum_faults == maximum_sites * 2, f"SA0/SA1 budget: {family_id}")
        batches = math.ceil(maximum_sites / BATCH_SITES)
        enabled = maximum_faults * vector_count
        baseline = batches * vector_count
        family_rows.append({
            "family_id": family_id, "partition": PARTITIONS[family_id],
            "maximum_sites": maximum_sites, "maximum_faults": maximum_faults,
            "full_vectors": vector_count, "site_batches": batches,
            "baseline_records": baseline, "maximum_enabled_transactions": enabled,
            "maximum_total_records": baseline + enabled,
            "campaign_status": "AUTHORIZED / NOT STARTED",
        })
        for local_batch in range(batches):
            site_start = local_batch * BATCH_SITES
            site_count = min(BATCH_SITES, maximum_sites - site_start)
            fault_count = site_count * 2
            plan_rows.append({
                "global_batch_id": global_batch, "family_id": family_id,
                "partition": PARTITIONS[family_id], "family_batch_id": local_batch,
                "site_rank_start": site_start, "site_count_maximum": site_count,
                "global_site_offset": global_site_offset + site_start,
                "fault_index_start": global_fault_offset + site_start * 2,
                "fault_count_maximum": fault_count, "vector_count": vector_count,
                "baseline_records": vector_count,
                "maximum_enabled_transactions": fault_count * vector_count,
                "checkpoint_after_batch": "REQUIRED", "status": "AUTHORIZED / NOT STARTED",
            })
            global_batch += 1
        global_site_offset += maximum_sites
        global_fault_offset += maximum_faults
    require(global_site_offset == TOTAL_GENERIC_SITES, "plan site total")
    require(global_fault_offset == TOTAL_MAXIMUM_FAULTS, "plan fault total")
    return family_rows, plan_rows


def execute() -> None:
    require(ROOT.name == "vlsi_fault_detection_v2", f"place script in project root, not {ROOT}")
    outputs = (EXECUTION_CONTRACT, AUTHORIZATION, DATASET_SCHEMA, FAMILY_BUDGET_CSV,
               FAMILY_BUDGET_JSON, EXECUTION_PLAN, PREFLIGHT, REPORT, MANIFEST, AUDIT)
    require(not any(path.exists() for path in outputs), "frozen Stage 12C-1K output already exists; use --status")
    vectors, campaign = verify_inputs()
    family_rows, plan_rows = build_plan(vectors, campaign)
    available_gib = shutil.disk_usage(ROOT).free / 1024**3
    require(available_gib >= MIN_FREE_GIB, f"insufficient disk: {available_gib:.2f} GiB; {MIN_FREE_GIB} GiB required")

    total_batches = len(plan_rows)
    total_baselines = sum(int(row["baseline_records"]) for row in family_rows)
    total_enabled = sum(int(row["maximum_enabled_transactions"]) for row in family_rows)
    total_records = total_baselines + total_enabled
    created = now()

    family_fields = ["family_id", "partition", "maximum_sites", "maximum_faults", "full_vectors",
                     "site_batches", "baseline_records", "maximum_enabled_transactions",
                     "maximum_total_records", "campaign_status"]
    plan_fields = ["global_batch_id", "family_id", "partition", "family_batch_id",
                   "site_rank_start", "site_count_maximum", "global_site_offset",
                   "fault_index_start", "fault_count_maximum", "vector_count",
                   "baseline_records", "maximum_enabled_transactions",
                   "checkpoint_after_batch", "status"]

    execution_contract = {
        "contract_version": "CIRCUITSAGE-HMAC-V2.2-FULL-CAMPAIGN-EXECUTION-12C1K-v1",
        "stage": STAGE, "status": "FROZEN", "created_at": created,
        "scope": "GENERALIZATION_TRAIN 3 FAMILIES + GENERALIZATION_CALIBRATION 1 FAMILY",
        "fault_model": "SINGLE PERSISTENT CELL-OUTPUT SA0/SA1",
        "families": list(FAMILIES), "family_skipping": "PROHIBITED",
        "maximum_sites": TOTAL_GENERIC_SITES, "maximum_faults": TOTAL_MAXIMUM_FAULTS,
        "full_vectors": TOTAL_VECTORS, "site_batch_size": BATCH_SITES,
        "planned_batches": total_batches, "baseline_records": total_baselines,
        "maximum_enabled_transactions": total_enabled, "maximum_total_records": total_records,
        "execution": "SEQUENTIAL / SIMULATION JOBS 1 / BUILD JOBS 1",
        "checkpoint": "REQUIRED AFTER EVERY COMPLETE BATCH",
        "resume": "VERIFY BATCH HASH, SCHEMA AND RECORD COUNT; NEVER APPEND TO PARTIAL BATCH",
        "batch_outputs": "DETERMINISTIC COMPRESSED NPZ + COMPACT SUMMARY JSON; RAW TEXT OPTIONAL DEBUG ONLY",
        "fault_free_policy": "ONE BASELINE PER FAMILY/VECTOR IN EVERY BATCH; BYTE-EXACT CROSS-BATCH CONSISTENCY",
        "fault_catalog_policy": "DERIVE ELIGIBLE SITES DETERMINISTICALLY; ACTUAL COUNT MAY NOT EXCEED FROZEN MAXIMUM",
        "detection_definition": campaign["detection_definition"],
        "network_requirement": "NONE / OFFLINE EXECUTION",
        "upstream_mutation": "PROHIBITED",
    }
    authorization = {
        "authorization_version": "CIRCUITSAGE-HMAC-V2.2-FULL-CAMPAIGN-EXECUTION-AUTHORIZATION-12C1K-v1",
        "stage": STAGE, "status": "FROZEN", "created_at": created,
        "authorization_basis": record(AUDIT_1J),
        "full_train_calibration_campaign": "AUTHORIZED / NOT STARTED",
        "response_generation": "AUTHORIZED ONLY FOR FROZEN FULL VECTOR SCHEDULE",
        "fault_injection": "AUTHORIZED ONLY FOR DETERMINISTIC ELIGIBLE TRAIN/CALIBRATION CATALOG",
        "dataset_consolidation": "AUTHORIZED AFTER ALL BATCHES PASS INTEGRITY REPLAY",
        "model_training_selection_inference": "NOT AUTHORIZED / NOT AUTHORIZED / NOT AUTHORIZED",
        "independent_test": "LOCKED / 2 FAMILIES", "validation": "UNOPENED",
        "holdout": "SEALED / 1 FAMILY",
        "protected_access": {"independent_test": 0, "validation": 0, "holdout": 0},
        "threshold_or_vector_changes": "PROHIBITED",
        "execution_plan": rel(EXECUTION_PLAN),
    }
    dataset_schema = {
        "schema_version": "CIRCUITSAGE-HMAC-V2.2-FULL-CAMPAIGN-DATASET-12C1K-v1",
        "stage": STAGE, "status": "FROZEN", "created_at": created,
        "sample_axis": "FAULT INSTANCE", "vector_axis": "FAMILY-SPECIFIC FULL FROZEN VECTOR SCHEDULE",
        "model_facing_features": [
            "family_index", "family_graph_reference", "vector_mask", "response_validity_mask",
            "response_xor", "completion_cycle_delta", "timeout", "protocol_error", "observable",
            "baseline_response", "baseline_cycles", "transaction_id",
        ],
        "supervision_only_targets": [
            "opaque_fault_id", "local_site_index", "stuck_value", "candidate_site_count",
            "exact_site", "behavior_signature_sha256",
        ],
        "prohibited_features": [
            "fault_id", "site_id", "cell_name", "net_name", "stuck_value", "target_site",
            "target_polarity", "raw_injection_selector", "partition_truth",
        ],
        "identity_firewall": "FAULT IDENTITY ABSENT FROM MODEL-FACING FEATURES",
        "ambiguity": "PRESERVE COMPLETE BEHAVIORALLY CONSISTENT SITE SET",
        "partition_rule": "TRAIN AND CALIBRATION STORED SEPARATELY; CALIBRATION NEVER USED FOR GRADIENTS",
        "protected_partitions": {"independent_test": "LOCKED", "validation": "UNOPENED", "holdout": "SEALED"},
    }
    family_json = {
        "registry_version": "CIRCUITSAGE-HMAC-V2.2-FULL-CAMPAIGN-FAMILY-BUDGET-12C1K-v1",
        "stage": STAGE, "status": "FROZEN", "families": family_rows,
        "maximum_totals": {"sites": TOTAL_GENERIC_SITES, "faults": TOTAL_MAXIMUM_FAULTS,
                           "vectors": TOTAL_VECTORS, "batches": total_batches,
                           "enabled_transactions": total_enabled, "records": total_records},
    }
    preflight = {
        "preflight_version": "CIRCUITSAGE-HMAC-V2.2-FULL-CAMPAIGN-PREFLIGHT-12C1K-v1",
        "stage": STAGE, "status": "PASS", "created_at": created,
        "available_disk_gib": round(available_gib, 2), "minimum_disk_gib": MIN_FREE_GIB,
        "families": 4, "full_vectors": TOTAL_VECTORS, "planned_batches": total_batches,
        "maximum_sites": TOTAL_GENERIC_SITES, "maximum_faults": TOTAL_MAXIMUM_FAULTS,
        "maximum_enabled_transactions": total_enabled,
        "execution_mode": "SEQUENTIAL / CHECKPOINTED / RESUMABLE",
        "internet_required": False, "simulation_calls": 0, "fault_injection_calls": 0,
        "training_calls": 0, "inference_calls": 0,
    }
    report = f"""# CircuitSage-HMAC V2.2 full-campaign authorization — Stage 12C-1K

The frozen bounded pilot received a conditional advancement disposition.  This
gate authorizes a separate full TRAIN/CALIBRATION campaign using all
{TOTAL_VECTORS} frozen vectors and no protected circuit families.

The conservative execution ceiling is {TOTAL_GENERIC_SITES} synthesized-cell
sites, {TOTAL_MAXIMUM_FAULTS} SA0/SA1 fault instances, {total_batches} batches,
and {total_enabled} enabled fault/vector transactions.  The execution stage
must derive the exact eligible site catalog deterministically; it may reduce
this ceiling for contracted exclusions but may never exceed it.

Execution is sequential, offline, checkpointed after every complete batch, and
must store deterministic compressed artifacts.  Model training, selection and
inference remain unauthorized.  Independent TEST is locked, VALIDATION is
unopened and HOLDOUT is sealed.  Independent generalization remains not
established.  The future hybrid brand remains **{FUTURE_BRAND}**.
""".encode()

    frozen_write(EXECUTION_CONTRACT, canonical_json(execution_contract))
    frozen_write(AUTHORIZATION, canonical_json(authorization))
    frozen_write(DATASET_SCHEMA, canonical_json(dataset_schema))
    frozen_write(FAMILY_BUDGET_CSV, csv_bytes(family_rows, family_fields))
    frozen_write(FAMILY_BUDGET_JSON, canonical_json(family_json))
    frozen_write(EXECUTION_PLAN, csv_bytes(plan_rows, plan_fields))
    frozen_write(PREFLIGHT, canonical_json(preflight))
    frozen_write(REPORT, report)

    stage_outputs = (EXECUTION_CONTRACT, AUTHORIZATION, DATASET_SCHEMA, FAMILY_BUDGET_CSV,
                     FAMILY_BUDGET_JSON, EXECUTION_PLAN, PREFLIGHT, REPORT)
    manifest = {
        "manifest_version": "CIRCUITSAGE-HMAC-V2.2-FULL-CAMPAIGN-AUTHORIZATION-MANIFEST-12C1K-v1",
        "stage": STAGE, "status": "PASS", "created_at": created,
        "stage_source": record(Path(__file__).resolve()),
        "frozen_inputs": {rel(path): record(path) for path in PINNED},
        "outputs": {rel(path): record(path) for path in stage_outputs},
        "families": list(FAMILIES), "maximum_sites": TOTAL_GENERIC_SITES,
        "maximum_faults": TOTAL_MAXIMUM_FAULTS, "full_vectors": TOTAL_VECTORS,
        "planned_batches": total_batches, "maximum_enabled_transactions": total_enabled,
        "simulation_calls": 0, "fault_injection_calls": 0, "dataset_records_created": 0,
        "model_deserializations": 0, "training_calls": 0, "inference_calls": 0,
        "independent_test_access": 0, "validation_access": 0, "holdout_access": 0,
    }
    frozen_write(MANIFEST, canonical_json(manifest))
    audit = {
        "audit_version": "CIRCUITSAGE-HMAC-V2.2-FULL-CAMPAIGN-EXECUTION-AUTHORIZATION-FREEZE-12C1K-v1",
        "stage": STAGE, "status": "PASS", "contract_authorization_status": "FROZEN / FROZEN",
        "pilot_disposition": "CONDITIONAL PASS / VERIFIED",
        "families_train_calibration": "4 / 3 / 1", "maximum_sites_faults_vectors": [TOTAL_GENERIC_SITES, TOTAL_MAXIMUM_FAULTS, TOTAL_VECTORS],
        "planned_batches": total_batches, "baseline_enabled_total_records": [total_baselines, total_enabled, total_records],
        "execution_build_jobs_checkpoint": "SEQUENTIAL / 1 / REQUIRED PER BATCH",
        "available_minimum_disk_gib": [round(available_gib, 2), MIN_FREE_GIB],
        "full_campaign_execution": "AUTHORIZED / NOT STARTED",
        "model_training_selection_inference": "NOT AUTHORIZED / NOT AUTHORIZED / NOT AUTHORIZED",
        "independent_test_validation_holdout_access": [0, 0, 0],
        "independent_generalization": "NOT ESTABLISHED",
        "frozen_rtl_netlists_vectors_modified": [False, False, False],
        "execution_contract_record": record(EXECUTION_CONTRACT), "authorization_record": record(AUTHORIZATION),
        "dataset_schema_record": record(DATASET_SCHEMA), "execution_plan_record": record(EXECUTION_PLAN),
        "preflight_record": record(PREFLIGHT), "manifest_record": record(MANIFEST),
        "future_combined_model_brand": FUTURE_BRAND,
        "next_gate": "STAGE 12C-1L — FULL TRAIN/CALIBRATION FAULT-CAMPAIGN EXECUTION AND DATASET FREEZE",
    }
    frozen_write(AUDIT, canonical_json(audit))

    for path in (EXECUTION_CONTRACT, AUTHORIZATION, DATASET_SCHEMA, FAMILY_BUDGET_JSON,
                 PREFLIGHT, MANIFEST, AUDIT):
        require(path.read_bytes() == canonical_json(load_json(path)), f"canonical output replay: {path.name}")
    require(FAMILY_BUDGET_CSV.read_bytes() == csv_bytes(family_rows, family_fields), "family-budget CSV replay")
    require(EXECUTION_PLAN.read_bytes() == csv_bytes(plan_rows, plan_fields), "execution-plan CSV replay")
    for path, expected in PINNED.items():
        require(sha256(path) == expected, f"frozen input modified: {rel(path)}")

    print("\nSTAGE 12C-1K — FULL TRAIN/CALIBRATION FAULT-CAMPAIGN EXECUTION-CONTRACT AND AUTHORIZATION FREEZE")
    print(f"{'Status':<90}: PASS")
    print(f"{'Contract / authorization status':<90}: FROZEN / FROZEN")
    print(f"{'Pilot disposition':<90}: CONDITIONAL PASS / VERIFIED")
    print(f"{'TRAIN / CALIBRATION families':<90}: 3 / 1")
    print(f"{'Maximum sites / faults / full vectors':<90}: {TOTAL_GENERIC_SITES} / {TOTAL_MAXIMUM_FAULTS} / {TOTAL_VECTORS}")
    print(f"{'Planned batches':<90}: {total_batches}")
    print(f"{'Baseline / enabled / maximum total records':<90}: {total_baselines} / {total_enabled} / {total_records}")
    print(f"{'Execution / build jobs / checkpoint':<90}: SEQUENTIAL / 1 / REQUIRED PER BATCH")
    print(f"{'Available / minimum disk':<90}: {available_gib:.2f} / {MIN_FREE_GIB} GiB")
    print(f"{'Full campaign execution':<90}: AUTHORIZED / NOT STARTED")
    print(f"{'Model training / selection / inference':<90}: NOT AUTHORIZED / NOT AUTHORIZED / NOT AUTHORIZED")
    print(f"{'Independent TEST / VALIDATION / HOLDOUT access':<90}: 0 / 0 / 0")
    print(f"{'Independent generalization':<90}: NOT ESTABLISHED")
    print(f"{'Execution contract':<90}: {EXECUTION_CONTRACT}")
    print(f"{'Execution contract SHA':<90}: {sha256(EXECUTION_CONTRACT)}")
    print(f"{'Authorization':<90}: {AUTHORIZATION}")
    print(f"{'Authorization SHA':<90}: {sha256(AUTHORIZATION)}")
    print(f"{'Execution plan':<90}: {EXECUTION_PLAN}")
    print(f"{'Execution plan SHA':<90}: {sha256(EXECUTION_PLAN)}")
    print(f"{'Manifest':<90}: {MANIFEST}")
    print(f"{'Manifest SHA':<90}: {sha256(MANIFEST)}")
    print(f"{'Audit':<90}: {AUDIT}")
    print(f"{'Audit SHA':<90}: {sha256(AUDIT)}")
    print(f"{'Next gate':<90}: STAGE 12C-1L — FULL TRAIN/CALIBRATION FAULT-CAMPAIGN EXECUTION AND DATASET FREEZE")


def status() -> None:
    print("STAGE 12C-1K — FULL-CAMPAIGN-AUTHORIZATION STATUS")
    if not MANIFEST.is_file() or not AUDIT.is_file():
        print("Status                    : NOT FROZEN")
        print(f"Expected audit            : {AUDIT}")
        return
    manifest = load_json(MANIFEST)
    audit = load_json(AUDIT)
    require(manifest.get("status") == "PASS" and audit.get("status") == "PASS", "frozen status")
    require(audit.get("manifest_record", {}).get("sha256") == sha256(MANIFEST), "manifest audit anchor")
    verify_manifest_outputs(manifest, "12C-1K", 8)
    print("Status                    : PASS / FROZEN")
    print(f"Execution                 : {audit['full_campaign_execution']}")
    print(f"Maximum sites/faults      : {audit['maximum_sites_faults_vectors'][:2]}")
    print(f"Planned batches           : {audit['planned_batches']}")
    print(f"Audit                     : {AUDIT}")
    print(f"Audit SHA                 : {sha256(AUDIT)}")


def self_test() -> None:
    sample_sites = [18392, 9665, 17000, 18692]
    require(sum(sample_sites) == TOTAL_GENERIC_SITES, "site-total arithmetic")
    require(sum(value * 2 for value in sample_sites) == TOTAL_MAXIMUM_FAULTS, "fault-total arithmetic")
    require(math.ceil(65 / BATCH_SITES) == 2 and math.ceil(64 / BATCH_SITES) == 1, "batch arithmetic")
    require(TOTAL_VECTORS == 64 + 48 + 64 + 64, "vector arithmetic")
    print("Stage 12C-1K self-test: PASS")


def locked_execute() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    with LOCK_FILE.open("a+", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            stop("Stage 12C-1K execution lock is held by another process")
        execute()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--status", action="store_true")
    mode.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.status:
        status()
    elif args.self_test:
        self_test()
    else:
        locked_execute()


if __name__ == "__main__":
    main()
