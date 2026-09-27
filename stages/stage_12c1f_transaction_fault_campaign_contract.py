#!/usr/bin/env python3
"""Stage 12C-1F: portable transaction/vector and fault-campaign contract freeze.

Verifies the frozen Stage 12C-1E multi-circuit synthesis results and freezes
the behavioral adapter, deterministic vector, response-schema, partition and
SA0/SA1 campaign contracts for TRAIN/CALIBRATION families.  This stage does
not generate adapters or vectors and performs no simulation, fault injection,
dataset construction, training, inference, or protected-partition access.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STAGE = "12C-1F"
ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "config/v2_2"
RESULT = ROOT / "results/circuitsage_hmac_v2_12c1"
PREV_WORK = RESULT / "train_calibration_synthesis_12c1e"
WORK = RESULT / "transaction_fault_campaign_contract_12c1f"

SOURCE_1E = ROOT / "stage_12c1e_train_calibration_synthesis.py"
METRICS_1E = PREV_WORK / "circuitsage_hmac_v2_2_synthesis_metrics_12c1e.json"
MANIFEST_1E = RESULT / "circuitsage_hmac_v2_2_elaboration_synthesis_manifest_12c1e.json"
AUDIT_1E = RESULT / "circuitsage_hmac_v2_2_elaboration_wrapper_generic_synthesis_freeze_12c1e.json"

PINNED = {
    SOURCE_1E: "286f5a12fc5f9a1749d2297eb8c15b49da1435d315cfbdbb3fdb33fd3e4d0392",
    METRICS_1E: "64069b19884dd1226225891b06734dda8c960b373ee5b5467f2dea380417d09c",
    MANIFEST_1E: "4d3d606e174ae06589e0e537486a15e042521fe9a02d49f9b4e1c596198476b6",
    AUDIT_1E: "3ac4f454173166ee6848e2617c7072742665e7166345ce5ffdb837f574d59220",
}

TRANSACTION_CONTRACT = CONFIG / "circuitsage_hmac_v2_2_portable_transaction_contract_12c1f.json"
VECTOR_CONTRACT = CONFIG / "circuitsage_hmac_v2_2_transaction_vector_contract_12c1f.json"
CAMPAIGN_CONTRACT = CONFIG / "circuitsage_hmac_v2_2_fault_campaign_contract_12c1f.json"
RESPONSE_SCHEMA = CONFIG / "circuitsage_hmac_v2_2_response_schema_contract_12c1f.json"
ACCESS_POLICY = CONFIG / "circuitsage_hmac_v2_2_campaign_partition_access_policy_12c1f.json"
FAMILY_CSV = WORK / "circuitsage_hmac_v2_2_family_transaction_registry_12c1f.csv"
FAMILY_JSON = WORK / "circuitsage_hmac_v2_2_family_transaction_registry_12c1f.json"
VECTOR_BUDGET = WORK / "circuitsage_hmac_v2_2_vector_budget_registry_12c1f.csv"
CAMPAIGN_BUDGET = WORK / "circuitsage_hmac_v2_2_campaign_budget_registry_12c1f.csv"
REPORT = WORK / "circuitsage_hmac_v2_2_transaction_campaign_contract_report_12c1f.md"
MANIFEST = RESULT / "circuitsage_hmac_v2_2_transaction_campaign_contract_manifest_12c1f.json"
AUDIT = RESULT / "circuitsage_hmac_v2_2_transaction_vector_fault_campaign_contract_freeze_12c1f.json"

AUTHORIZED = (
    "opentitan_hmac_sha256",
    "picorv32_cpu",
    "secworks_aes",
    "secworks_sha256",
)
FUTURE_BRAND = "Faultiva 1.0 — Hybrid VLSI Fault Intelligence"

FAMILY_SPEC = {
    "opentitan_hmac_sha256": {
        "partition": "GENERALIZATION_TRAIN",
        "operation": "HMAC_SHA256_FIXED_AND_VARIABLE_MESSAGE",
        "request_fields": "key_bytes,message_bytes,message_length,sha_mode",
        "response_fields": "digest_256,done,timeout,error",
        "reset": "ASSERT 5 CLOCKS; DEASSERT; WAIT 2 CLOCKS",
        "timeout_cycles": 4096,
        "vector_budget": 64,
        "pilot_vectors": 16,
        "stimulus_classes": "ZERO,ONES,COUNTING,ONEHOT,RANDOM,BOUNDARY_LENGTH",
    },
    "picorv32_cpu": {
        "partition": "GENERALIZATION_TRAIN",
        "operation": "BOUNDED_RISCV_PROGRAM_EXECUTION",
        "request_fields": "program_image,initial_pc,cycle_budget,interrupt_schedule",
        "response_fields": "architectural_signature,memory_signature,trap,retired_count,timeout,error",
        "reset": "ASSERT 8 CLOCKS; DEASSERT; BEGIN MEMORY HANDSHAKE",
        "timeout_cycles": 8192,
        "vector_budget": 48,
        "pilot_vectors": 12,
        "stimulus_classes": "ALU,BRANCH,LOAD_STORE,CSR,INTERRUPT,MIXED",
    },
    "secworks_aes": {
        "partition": "GENERALIZATION_TRAIN",
        "operation": "AES_BLOCK_ENCRYPT_DECRYPT",
        "request_fields": "key_128_or_256,key_length,block_128,direction",
        "response_fields": "result_block_128,ready,valid,timeout,error",
        "reset": "ASSERT 5 CLOCKS; DEASSERT; WAIT READY",
        "timeout_cycles": 2048,
        "vector_budget": 64,
        "pilot_vectors": 16,
        "stimulus_classes": "NIST_KAT,ZERO,ONES,COUNTING,AVALANCHE,RANDOM",
    },
    "secworks_sha256": {
        "partition": "GENERALIZATION_CALIBRATION",
        "operation": "SHA256_BLOCK_SEQUENCE",
        "request_fields": "message_blocks,block_count,init_next_schedule",
        "response_fields": "digest_256,ready,digest_valid,timeout,error",
        "reset": "ASSERT 5 CLOCKS; DEASSERT; WAIT READY",
        "timeout_cycles": 4096,
        "vector_budget": 64,
        "pilot_vectors": 16,
        "stimulus_classes": "NIST_KAT,ZERO,ONES,COUNTING,AVALANCHE,RANDOM",
    },
}


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


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def record(path: Path) -> dict[str, Any]:
    return {"path": rel(path), "sha256": sha256(path), "bytes": path.stat().st_size}


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


def csv_bytes(rows: list[dict[str, Any]], fields: list[str]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def verify_manifest_record(item: Any, path: Path) -> None:
    require(isinstance(item, dict), f"missing 12C-1E manifest record: {rel(path)}")
    require(item.get("path") == rel(path), f"12C-1E manifest path: {rel(path)}")
    require(item.get("sha256") == sha256(path), f"12C-1E manifest SHA: {rel(path)}")
    require(int(item.get("bytes", -1)) == path.stat().st_size, f"12C-1E manifest size: {rel(path)}")


def verify_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]:
    print("STAGE 12C-1F — TRAIN/CALIBRATION PORTABLE TRANSACTION-VECTOR AND FAULT-CAMPAIGN CONTRACT FREEZE")
    print("FROZEN INPUT VERIFICATION")
    for path, expected in PINNED.items():
        require(path.is_file(), f"missing frozen input: {rel(path)}")
        require(sha256(path) == expected, f"frozen input SHA: {rel(path)}")
        print(f"  {path.name:<112}: OK")

    manifest = load_json(MANIFEST_1E)
    audit = load_json(AUDIT_1E)
    metrics = load_json(METRICS_1E)
    require(manifest.get("status") == "PASS", "12C-1E manifest status")
    require(audit.get("status") == "PASS", "12C-1E audit status")
    require(audit.get("execution_status") == "COMPLETED / FROZEN", "12C-1E lifecycle")
    require(audit.get("generic_synthesis") == "PASS / 4 OF 4", "12C-1E synthesis")
    require(audit.get("structural_checks") == "PASS / 4 OF 4", "12C-1E structural checks")
    require(audit.get("deterministic_replay") == "PASS / BYTE-EXACT / 4 OF 4", "12C-1E replay")
    require(audit.get("independent_test") == "LOCKED / 2 FAMILIES", "TEST lock")
    require(audit.get("holdout") == "SEALED / 1 FAMILY", "HOLDOUT seal")
    require(audit.get("independent_test_validation_holdout_access") == [0, 0, 0], "protected access")
    require(tuple(audit.get("authorized_families", [])) == AUTHORIZED, "authorized family order")
    require(metrics.get("status") == "PASS" and len(metrics.get("families", [])) == 4, "metrics status")
    require(metrics.get("total_cells") == 63749 and metrics.get("total_wire_bits") == 201330, "aggregate topology")

    outputs = manifest.get("outputs", {})
    require(isinstance(outputs, dict) and outputs, "12C-1E output records")
    for item in outputs.values():
        path = ROOT / item["path"]
        require(path.is_file(), f"missing 12C-1E output: {item['path']}")
        verify_manifest_record(item, path)

    family_results: dict[str, dict[str, Any]] = {}
    for family_id in AUTHORIZED:
        path = PREV_WORK / "families" / family_id / "family_result_12c1e.json"
        verify_manifest_record(outputs.get(rel(path)), path)
        item = load_json(path)
        require(item.get("status") == "PASS" and item.get("family_id") == family_id, f"family result: {family_id}")
        require(item.get("replay_method") == "FULL SOURCE RE-SYNTHESIS", f"family replay method: {family_id}")
        require(item.get("deterministic_netlist_replay") == "PASS / BYTE-EXACT", f"family replay: {family_id}")
        require(item.get("upstream_rtl_modified") is False, f"upstream RTL: {family_id}")
        family_results[family_id] = item
    print("  Four family netlists, full re-synthesis replay, partition locks and source integrity          : PASS")
    return manifest, metrics, family_results


def main() -> None:
    require(ROOT.name == "vlsi_fault_detection_v2", f"run from project root, not {ROOT}")
    outputs = (TRANSACTION_CONTRACT, VECTOR_CONTRACT, CAMPAIGN_CONTRACT, RESPONSE_SCHEMA,
               ACCESS_POLICY, FAMILY_CSV, FAMILY_JSON, VECTOR_BUDGET, CAMPAIGN_BUDGET,
               REPORT, MANIFEST, AUDIT)
    for path in outputs:
        require(not path.exists(), f"Stage {STAGE} output already exists: {rel(path)}")

    previous_manifest, metrics, family_results = verify_inputs()
    timestamp = now()
    metric_map = {item["family_id"]: item for item in metrics["families"]}

    family_rows: list[dict[str, Any]] = []
    vector_rows: list[dict[str, Any]] = []
    campaign_rows: list[dict[str, Any]] = []
    for family_id in AUTHORIZED:
        spec = FAMILY_SPEC[family_id]
        result = family_results[family_id]
        metric = metric_map[family_id]
        cells = int(metric["cells"])
        pilot_faults = min(cells * 2, 4096)
        family_rows.append({
            "family_id": family_id,
            "partition": spec["partition"],
            "top_module": result["top_module"],
            "operation": spec["operation"],
            "request_fields": spec["request_fields"],
            "response_fields": spec["response_fields"],
            "reset_protocol": spec["reset"],
            "timeout_cycles": spec["timeout_cycles"],
            "fault_identity_input": "PROHIBITED",
            "adapter_status": "CONTRACTED / NOT GENERATED",
        })
        vector_rows.append({
            "family_id": family_id,
            "partition": spec["partition"],
            "total_vector_budget": spec["vector_budget"],
            "pilot_vector_budget": spec["pilot_vectors"],
            "stimulus_classes": spec["stimulus_classes"],
            "seed_derivation": "SHA256(FAMILY_ID || VECTOR_INDEX || 12C1F)",
            "truth_usage": "EXPECTED RESPONSE GENERATION ONLY; NEVER QUERY INPUT",
        })
        campaign_rows.append({
            "family_id": family_id,
            "partition": spec["partition"],
            "generic_cells": cells,
            "fault_model": "SINGLE PERSISTENT SA0/SA1",
            "maximum_catalog_faults": cells * 2,
            "pilot_fault_budget": pilot_faults,
            "pilot_vector_budget": spec["pilot_vectors"],
            "maximum_pilot_transactions": pilot_faults * int(spec["pilot_vectors"]),
            "campaign_status": "CONTRACTED / NOT AUTHORIZED",
        })

    family_fields = ["family_id", "partition", "top_module", "operation", "request_fields",
                     "response_fields", "reset_protocol", "timeout_cycles", "fault_identity_input", "adapter_status"]
    vector_fields = ["family_id", "partition", "total_vector_budget", "pilot_vector_budget",
                     "stimulus_classes", "seed_derivation", "truth_usage"]
    campaign_fields = ["family_id", "partition", "generic_cells", "fault_model",
                       "maximum_catalog_faults", "pilot_fault_budget", "pilot_vector_budget",
                       "maximum_pilot_transactions", "campaign_status"]

    transaction_contract = {
        "contract_version": "CIRCUITSAGE-HMAC-V2.2-PORTABLE-TRANSACTION-CONTRACT-12C1F-v1",
        "stage": STAGE,
        "status": "FROZEN",
        "scope": "TRAIN AND CALIBRATION CIRCUIT FAMILIES ONLY",
        "logical_request": {
            "schema_version": "uint16",
            "family_id": "registered categorical identifier",
            "transaction_id": "uint64 unique within campaign",
            "operation_code": "registered family operation",
            "payload": "packed bytes interpreted only by registered adapter",
            "payload_validity_mask": "one bit per payload bit",
            "cycle_budget": "uint32 bounded by family timeout",
        },
        "logical_response": {
            "transaction_id": "echo only; no fault metadata",
            "response_payload": "packed architectural/external response",
            "response_validity_mask": "one bit per response bit",
            "completion_cycle": "uint32",
            "done": "boolean",
            "timeout": "boolean",
            "protocol_error": "boolean",
        },
        "adapter_invariants": [
            "one deterministic adapter per family",
            "upstream RTL and synthesized netlist remain byte-identical",
            "reset and handshake behavior fixed before response generation",
            "fault selector, site, polarity, truth label and partition label absent from model-facing input",
            "normal and faulty executions use identical transaction schedules",
            "timeouts and invalid bits are explicit and never silently imputed",
        ],
        "families": {row["family_id"]: row for row in family_rows},
        "adapter_generation": "AUTHORIZED NEXT GATE / NOT PERFORMED",
        "functional_validation": "NOT PERFORMED",
    }

    vector_contract = {
        "contract_version": "CIRCUITSAGE-HMAC-V2.2-TRANSACTION-VECTOR-CONTRACT-12C1F-v1",
        "stage": STAGE,
        "status": "FROZEN",
        "determinism": "ALL PSEUDORANDOM FIELDS DERIVED BY SHA-256 COUNTER MODE",
        "seed_domain": "CIRCUITSAGE-HMAC-V2.2/12C-1F/VECTOR",
        "duplicate_policy": "BYTE-IDENTICAL REQUESTS PROHIBITED WITHIN A FAMILY",
        "coverage_requirements": [
            "reset recovery and first legal transaction",
            "all-zero, all-one, counting and one-bit avalanche payloads where applicable",
            "family-standard known-answer tests where available",
            "minimum and maximum supported transaction lengths",
            "legal back-to-back request schedules",
            "timeout canaries separated from accepted functional vectors",
        ],
        "budgets": vector_rows,
        "vector_generation": "AUTHORIZED NEXT GATE / NOT PERFORMED",
        "test_holdout_vectors": "NOT DESIGNED / NOT ACCESSED",
    }

    campaign_contract = {
        "contract_version": "CIRCUITSAGE-HMAC-V2.2-FAULT-CAMPAIGN-CONTRACT-12C1F-v1",
        "stage": STAGE,
        "status": "FROZEN",
        "fault_model": "SINGLE PERSISTENT CELL-OUTPUT SA0/SA1",
        "fault_catalog_rules": [
            "enumerate eligible synthesized cell-output bits in deterministic hierarchical order",
            "exclude clocks, resets, constants and unsupported multi-driver nets with recorded reasons",
            "assign stable opaque fault IDs after eligibility freeze",
            "never expose fault ID, physical site, stuck value or truth-derived feature to model query",
            "run one fault per simulation and restore the fault-free design between runs",
            "capture a fault-free baseline for every vector and family",
        ],
        "detection_definition": "ANY VALID RESPONSE-BIT DIFFERENCE OR CONTRACTED TIMEOUT/PROTOCOL DIFFERENCE",
        "localization_target": "AMBIGUITY-AWARE SET OF STRUCTURAL SITES CONSISTENT WITH OBSERVED RESPONSE",
        "polarity_target": "SA0 OR SA1; SCORED ONLY WHEN IDENTIFIABLE",
        "campaigns": campaign_rows,
        "pilot_first": True,
        "full_campaign_unlock": "SEPARATE ACCEPTANCE AND RESOURCE AUTHORIZATION REQUIRED",
        "execution": "NOT AUTHORIZED",
    }

    response_schema = {
        "schema_version": "CIRCUITSAGE-HMAC-V2.2-PORTABLE-RESPONSE-SCHEMA-12C1F-v1",
        "stage": STAGE,
        "status": "FROZEN",
        "record_key": ["family_id", "transaction_id", "opaque_run_id"],
        "model_input_fields": [
            "family_graph_features",
            "request_payload_hash_or_registered_embedding",
            "observed_response_bits",
            "response_validity_mask",
            "golden_difference_bits",
            "completion_cycle_delta",
            "timeout_and_protocol_flags",
        ],
        "prohibited_model_input_fields": [
            "fault_id", "site_id", "cell_name", "net_name", "stuck_value",
            "target_site", "target_polarity", "partition_truth", "raw_injection_selector",
        ],
        "supervision_only_fields": [
            "fault_present", "eligible_site_set", "fault_polarity", "observability_class",
        ],
        "outputs": {
            "fault_probability": "float32 [0,1]",
            "ood_probability": "float32 [0,1]",
            "candidate_site_ids": "ordered opaque identifiers",
            "candidate_scores": "float32 aligned with candidates",
            "predicted_polarity": "SA0, SA1, AMBIGUOUS or NOT_APPLICABLE",
            "ambiguity_size": "uint32",
            "status": "FAULT_FREE, DETECTED, OOD, TIMEOUT or INDETERMINATE",
        },
        "normalization_training": "TRAIN ONLY",
        "threshold_selection": "CALIBRATION ONLY / NO GRADIENT UPDATES",
    }

    access_policy = {
        "policy_version": "CIRCUITSAGE-HMAC-V2.2-CAMPAIGN-PARTITION-ACCESS-12C1F-v1",
        "stage": STAGE,
        "status": "FROZEN",
        "GENERALIZATION_TRAIN": "ADAPTER/VECTOR GENERATION AUTHORIZED NEXT GATE; CAMPAIGN NOT YET AUTHORIZED",
        "GENERALIZATION_CALIBRATION": "ADAPTER/VECTOR GENERATION AUTHORIZED NEXT GATE; NO MODEL GRADIENTS",
        "INDEPENDENT_CIRCUIT_TEST": "LOCKED; NO ADAPTER, VECTOR, SIMULATION OR SEMANTIC ACCESS",
        "GENERALIZATION_HOLDOUT": "SEALED; NO SEMANTIC ACCESS",
        "validation_access": 0,
        "reassignment": "PROHIBITED",
    }

    family_json = {
        "registry_version": "CIRCUITSAGE-HMAC-V2.2-FAMILY-TRANSACTION-REGISTRY-12C1F-v1",
        "stage": STAGE,
        "status": "FROZEN",
        "families": family_rows,
    }
    report = f"""# CircuitSage-HMAC V2.2 Transaction and Campaign Contract — Stage 12C-1F

All four frozen TRAIN/CALIBRATION netlists passed predecessor integrity replay.
This stage defines a shared request/response envelope while preserving each
circuit's legal protocol: HMAC messages, bounded RISC-V programs, AES blocks,
and SHA-256 block sequences.

Fault campaigns are restricted to single persistent SA0/SA1 faults and must
begin with bounded pilots. Fault identity, site and polarity are prohibited
from model-facing inputs. Localization is ambiguity-aware. The two independent
TEST families remain locked and the HOLDOUT family remains sealed.

No adapters, vectors, simulations, faults, datasets or models were created.
The future combined release remains **{FUTURE_BRAND}**.
""".encode()

    frozen_write(TRANSACTION_CONTRACT, canonical_json(transaction_contract))
    frozen_write(VECTOR_CONTRACT, canonical_json(vector_contract))
    frozen_write(CAMPAIGN_CONTRACT, canonical_json(campaign_contract))
    frozen_write(RESPONSE_SCHEMA, canonical_json(response_schema))
    frozen_write(ACCESS_POLICY, canonical_json(access_policy))
    frozen_write(FAMILY_CSV, csv_bytes(family_rows, family_fields))
    frozen_write(FAMILY_JSON, canonical_json(family_json))
    frozen_write(VECTOR_BUDGET, csv_bytes(vector_rows, vector_fields))
    frozen_write(CAMPAIGN_BUDGET, csv_bytes(campaign_rows, campaign_fields))
    frozen_write(REPORT, report)

    stage_outputs = [TRANSACTION_CONTRACT, VECTOR_CONTRACT, CAMPAIGN_CONTRACT,
                     RESPONSE_SCHEMA, ACCESS_POLICY, FAMILY_CSV, FAMILY_JSON,
                     VECTOR_BUDGET, CAMPAIGN_BUDGET, REPORT]
    manifest = {
        "manifest_version": "CIRCUITSAGE-HMAC-V2.2-TRANSACTION-CAMPAIGN-CONTRACT-MANIFEST-12C1F-v1",
        "stage": STAGE,
        "status": "PASS",
        "created_at": timestamp,
        "frozen_inputs": {rel(path): record(path) for path in PINNED},
        "predecessor_output_count_verified": len(previous_manifest["outputs"]),
        "outputs": {rel(path): record(path) for path in stage_outputs},
        "families_contracted": list(AUTHORIZED),
        "adapter_generation_calls": 0,
        "vector_records_created": 0,
        "simulation_calls": 0,
        "fault_injection_calls": 0,
        "dataset_records_created": 0,
        "models_deserialized": 0,
        "training_calls": 0,
        "inference_calls": 0,
        "independent_test_access": 0,
        "validation_access": 0,
        "holdout_access": 0,
    }
    frozen_write(MANIFEST, canonical_json(manifest))

    audit = {
        "audit_version": "CIRCUITSAGE-HMAC-V2.2-TRANSACTION-VECTOR-FAULT-CAMPAIGN-CONTRACT-FREEZE-12C1F-v1",
        "stage": STAGE,
        "status": "PASS",
        "contracts": "FROZEN",
        "authorized_train_calibration_families": "4 / 4",
        "portable_transaction_schema": "FROZEN",
        "family_protocols": "FROZEN / 4",
        "vector_budgets": "FROZEN / 240 TOTAL / 60 PILOT",
        "fault_model": "SINGLE PERSISTENT SA0/SA1",
        "fault_identity_input": "PROHIBITED",
        "ambiguity_aware_localization": True,
        "adapter_vector_generation": "AUTHORIZED NEXT GATE / NOT STARTED",
        "campaign_execution": "NOT AUTHORIZED",
        "independent_test": "LOCKED / 2 FAMILIES",
        "holdout": "SEALED / 1 FAMILY",
        "generation_simulation_fault_dataset": [0, 0, 0, 0],
        "training_inference": [0, 0],
        "independent_test_validation_holdout_access": [0, 0, 0],
        "future_combined_model_brand": FUTURE_BRAND,
        "transaction_contract_record": record(TRANSACTION_CONTRACT),
        "vector_contract_record": record(VECTOR_CONTRACT),
        "campaign_contract_record": record(CAMPAIGN_CONTRACT),
        "response_schema_record": record(RESPONSE_SCHEMA),
        "access_policy_record": record(ACCESS_POLICY),
        "family_registry_record": record(FAMILY_JSON),
        "report_record": record(REPORT),
        "manifest_record": record(MANIFEST),
        "next_gate": "STAGE 12C-1G — TRAIN/CALIBRATION PORTABLE ADAPTER AND DETERMINISTIC VECTOR GENERATION FREEZE",
    }
    frozen_write(AUDIT, canonical_json(audit))

    for path in (TRANSACTION_CONTRACT, VECTOR_CONTRACT, CAMPAIGN_CONTRACT,
                 RESPONSE_SCHEMA, ACCESS_POLICY, FAMILY_JSON, MANIFEST, AUDIT):
        require(canonical_json(load_json(path)) == path.read_bytes(), f"canonical output replay: {path.name}")
    require(FAMILY_CSV.read_bytes() == csv_bytes(family_rows, family_fields), "family CSV replay")
    require(VECTOR_BUDGET.read_bytes() == csv_bytes(vector_rows, vector_fields), "vector CSV replay")
    require(CAMPAIGN_BUDGET.read_bytes() == csv_bytes(campaign_rows, campaign_fields), "campaign CSV replay")
    for path, expected in PINNED.items():
        require(sha256(path) == expected, f"frozen input modified: {rel(path)}")

    total_pilot_faults = sum(int(row["pilot_fault_budget"]) for row in campaign_rows)
    max_pilot_transactions = sum(int(row["maximum_pilot_transactions"]) for row in campaign_rows)
    print("\nSTAGE 12C-1F — TRAIN/CALIBRATION PORTABLE TRANSACTION-VECTOR AND FAULT-CAMPAIGN CONTRACT FREEZE")
    print(f"{'Status':<78}: PASS")
    print(f"{'Contracts / family protocols':<78}: FROZEN / 4")
    print(f"{'Authorized TRAIN / CALIBRATION families':<78}: 3 / 1")
    print(f"{'Total / pilot vectors':<78}: 240 / 60")
    print(f"{'Pilot fault budget / maximum pilot transactions':<78}: {total_pilot_faults} / {max_pilot_transactions}")
    print(f"{'Fault model':<78}: SINGLE PERSISTENT SA0/SA1")
    print(f"{'Fault identity in model input':<78}: PROHIBITED")
    print(f"{'Ambiguity-aware localization / OOD output':<78}: REQUIRED / REQUIRED")
    print(f"{'Adapter/vector generation':<78}: AUTHORIZED NEXT GATE / NOT STARTED")
    print(f"{'Simulation / fault injection / dataset records':<78}: 0 / 0 / 0")
    print(f"{'Training / inference':<78}: 0 / 0")
    print(f"{'Independent TEST / HOLDOUT':<78}: LOCKED 2 / SEALED 1")
    print(f"{'Independent TEST / VALIDATION / HOLDOUT access':<78}: 0 / 0 / 0")
    print(f"{'Transaction contract':<78}: {TRANSACTION_CONTRACT}")
    print(f"{'Transaction contract SHA':<78}: {sha256(TRANSACTION_CONTRACT)}")
    print(f"{'Campaign contract':<78}: {CAMPAIGN_CONTRACT}")
    print(f"{'Campaign contract SHA':<78}: {sha256(CAMPAIGN_CONTRACT)}")
    print(f"{'Manifest':<78}: {MANIFEST}")
    print(f"{'Manifest SHA':<78}: {sha256(MANIFEST)}")
    print(f"{'Audit':<78}: {AUDIT}")
    print(f"{'Audit SHA':<78}: {sha256(AUDIT)}")
    print(f"{'Next gate':<78}: STAGE 12C-1G — TRAIN/CALIBRATION PORTABLE ADAPTER AND DETERMINISTIC VECTOR GENERATION FREEZE")


def self_test() -> None:
    require(tuple(sorted(FAMILY_SPEC)) == AUTHORIZED, "family identity set")
    require(sum(int(item["vector_budget"]) for item in FAMILY_SPEC.values()) == 240, "vector budget")
    require(sum(int(item["pilot_vectors"]) for item in FAMILY_SPEC.values()) == 60, "pilot vector budget")
    require(canonical_json({"z": 1, "a": 2}) == b'{"a":2,"z":1}\n', "canonical JSON")
    print("Stage 12C-1F self-test: PASS")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        main()
