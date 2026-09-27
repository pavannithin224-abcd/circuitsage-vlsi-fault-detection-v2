#!/usr/bin/env python3
"""Stage 12C-1A: V2.2 generalization architecture and contract freeze.

Verifies the completed Stage 12B-3P V2.1 improvement freeze and creates only
the V2.2 scientific scope, graph/behavior architecture, circuit-family data
partition, candidate-grid, interface, and acceptance contracts.  No RTL is
downloaded, no dataset is constructed, no model is loaded, trained, selected,
or evaluated, and no protected partition is opened.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STAGE = "12C-1A"
ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "config/v2_2"
RESULT = ROOT / "results/circuitsage_hmac_v2_12c1"

SOURCE_3P = ROOT / "stage_12b3p_repair_site_test_disposition.py"
RESULT_3P = ROOT / "results/circuitsage_hmac_v2_12b3"
WORK_3P = RESULT_3P / "repair_site_test_disposition_12b3p"
LOG_3P = RESULT_3P / "stage_12b3p_20260918_003800.log"
RESOURCE_3P = RESULT_3P / "stage_12b3p_resources_20260918_003800.log"
POLICY_3P = ROOT / "config/v2_1_improvement/circuitsage_hmac_v2_1_improvement_final_disposition_policy_12b3p.json"
FINAL_LOCK_3P = WORK_3P / "circuitsage_hmac_v2_1_improvement_final_lock_12b3p.json"
COMPARISON_3P = WORK_3P / "circuitsage_hmac_v2_1_improvement_comparison_12b3p.json"
REGISTRY_CSV_3P = WORK_3P / "circuitsage_hmac_v2_1_improvement_capability_registry_12b3p.csv"
REGISTRY_JSON_3P = WORK_3P / "circuitsage_hmac_v2_1_improvement_capability_registry_12b3p.json"
REPORT_3P = WORK_3P / "circuitsage_hmac_v2_1_improvement_final_report_12b3p.md"
MANIFEST_3P = RESULT_3P / "circuitsage_hmac_v2_1_improvement_disposition_manifest_12b3p.json"
AUDIT_3P = RESULT_3P / "circuitsage_hmac_v2_1_improvement_final_freeze_12b3p.json"

READINESS_12B1G = CONFIG / "circuitsage_hmac_v2_2_generalization_readiness_policy_12b1g.json"
GRAPH_SCHEMA = ROOT / "results/hmac_fault_campaign_11d1/graph_dataset_11d1a/hmac_golden_netlist_graph_schema_11d1a.json"
V1_REMOTE_LOCK = ROOT / "config/release/hmac_github_remote_integrity_lock_11e1k.json"

SCOPE = CONFIG / "circuitsage_hmac_v2_2_generalization_scope_contract_12c1a.json"
ARCHITECTURE = CONFIG / "circuitsage_hmac_v2_2_generalization_architecture_12c1a.json"
PARTITION = CONFIG / "circuitsage_hmac_v2_2_circuit_family_partition_contract_12c1a.json"
INTERFACE = CONFIG / "circuitsage_hmac_v2_2_inference_interface_contract_12c1a.json"
TRAINING = CONFIG / "circuitsage_hmac_v2_2_training_selection_contract_12c1a.json"
ACCEPTANCE = CONFIG / "circuitsage_hmac_v2_2_acceptance_contract_12c1a.json"
CANDIDATE_GRID = CONFIG / "circuitsage_hmac_v2_2_candidate_grid_12c1a.csv"
ENVIRONMENT = RESULT / "circuitsage_hmac_v2_2_contract_environment_12c1a.json"
REPORT = RESULT / "circuitsage_hmac_v2_2_generalization_contract_report_12c1a.md"
MANIFEST = RESULT / "circuitsage_hmac_v2_2_generalization_contract_manifest_12c1a.json"
AUDIT = RESULT / "circuitsage_hmac_v2_2_generalization_contract_freeze_12c1a.json"

PINNED = {
    SOURCE_3P: "7468b813117ff2633fffd97f8dd3a6a58d209b432d8ca87052c333cb15bda904",
    LOG_3P: "8c4e2df9b78b55918e881d98bed13265c0d14c34111327b01f427cb89ace8a8b",
    RESOURCE_3P: "baf9da66f123a5bf60abc30e6e9e0bbb601b5cfcd04e6ac0fe19b300db76ce7b",
    POLICY_3P: "92ddd5c5c8d92ce619e6f9ef52e872d33e512b920f3dffbe59dc2c3241355e2d",
    FINAL_LOCK_3P: "1c4f00097d1e89747341656e930af9c3350aa47f09f23f71c2c454973722a799",
    COMPARISON_3P: "0f2d5bda6ad24ba1cd6f19edeca06eee6b298a5baad0097ab971ae4d47d31e54",
    REGISTRY_CSV_3P: "3912468bc206dabb4bf0a322325cf3f413ab7909298af9d76d2523f31f325475",
    REGISTRY_JSON_3P: "7d7404da07ca6a42c5b257eaa20856807dba0ddb6d05594e302f035a4826c97f",
    REPORT_3P: "163fad39db473a681771a0de42fbdcb9667b641754479e2b97fc86b2b95d6ff7",
    MANIFEST_3P: "f0ae308277731d93835c80a6210a7633d58991312f87b6d7f4f6539b3df90720",
    AUDIT_3P: "a7ae20cf0df0b2b1d0bd2858f9d2375d450c4617472fbdcf759cc39ab8289e0c",
    READINESS_12B1G: "08e2320e70c4d30d90b045521824f7175a5dd3b8595d6244f5711d8d9e9ffc85",
    GRAPH_SCHEMA: "d24c3dd285891e284d20d1adcd93f793bedfd3042f20159a8a486359bc25e393",
    V1_REMOTE_LOCK: "3d93cd8bb9c3bba47dd665ba6a45ad4e193613c1f6d386e32b3265471d94dd5a",
}

FUTURE_BRAND = "Faultiva 1.0 — Hybrid VLSI Fault Intelligence"
PARTITIONS = {
    "GENERALIZATION_TRAIN": 3,
    "GENERALIZATION_CALIBRATION": 1,
    "INDEPENDENT_CIRCUIT_TEST": 2,
    "GENERALIZATION_HOLDOUT": 1,
}
MINIMUM_CIRCUIT_FAMILIES = sum(PARTITIONS.values())


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
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
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


def csv_bytes(rows: list[dict[str, Any]], fields: list[str]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def verify_record(item: Any, path: Path, label: str) -> None:
    require(isinstance(item, dict), f"missing record: {label}")
    require(item.get("path") == rel(path), f"record path: {label}")
    require(item.get("sha256") == sha256(path), f"record SHA: {label}")
    require(int(item.get("bytes", -1)) == path.stat().st_size, f"record size: {label}")


def verify_inputs() -> dict[str, Any]:
    print("STAGE 12C-1A — V2.2 GENERALIZATION ARCHITECTURE, DATA-PARTITION, AND ACCEPTANCE-CONTRACT FREEZE")
    print("FROZEN INPUT VERIFICATION")
    for path, expected in PINNED.items():
        require(path.is_file(), f"missing frozen input: {rel(path)}")
        require(sha256(path) == expected, f"frozen input SHA: {rel(path)}")
        print(f"  {path.name:<105}: OK")

    policy = load_json(POLICY_3P)
    lock = load_json(FINAL_LOCK_3P)
    comparison = load_json(COMPARISON_3P)
    manifest = load_json(MANIFEST_3P)
    audit = load_json(AUDIT_3P)
    readiness = load_json(READINESS_12B1G)
    graph_schema = load_json(GRAPH_SCHEMA)
    v1_lock = load_json(V1_REMOTE_LOCK)

    require(policy.get("status") == "FROZEN", "3P policy status")
    require(policy.get("lifecycle") == "COMPLETED AND FROZEN CLOSED-CATALOG V2.1 IMPROVEMENT EXPERIMENT", "3P lifecycle")
    require(policy.get("locked_evaluation_acceptance") == "NOT_MET / PRESERVED", "3P acceptance preservation")
    require(policy.get("repair_site_test") == "CONSUMED / FROZEN / DO NOT REOPEN", "REPAIR_SITE_TEST lock")
    require(policy.get("dev_site_test") == "CONSUMED / DO NOT REOPEN", "DEV_SITE_TEST lock")
    require(policy.get("validation") == "NOT ACCESSED" and policy.get("holdout") == "NOT ACCESSED", "protected access")
    require(policy.get("allowed_next_action") == "CREATE STAGE 12C-1A GENERALIZATION CONTRACT ONLY", "12C-1A authorization")
    require(policy.get("v2_2_dataset_or_training") == "NOT AUTHORIZED BY THIS POLICY", "dataset/training boundary")
    require(policy.get("future_combined_model_brand") == FUTURE_BRAND, "future brand reservation")

    require(lock.get("status") == "FROZEN", "3P final lock")
    require(lock.get("locked_evaluation_acceptance") == "NOT_MET", "final-lock acceptance")
    require(lock.get("retraining_threshold_change_candidate_reselection") == "PROHIBITED / PROHIBITED / PROHIBITED", "adaptation lock")
    require(comparison.get("status") == "FROZEN", "comparison freeze")
    require(comparison.get("interpretation") == "SUBSTANTIAL IMPROVEMENT OBSERVED; PRE-REGISTERED FINAL ACCEPTANCE STILL NOT MET", "comparison interpretation")

    require(manifest.get("status") == "PASS", "3P manifest status")
    require(manifest.get("locked_evaluation_acceptance") == "NOT_MET / PRESERVED", "manifest acceptance")
    require([manifest.get("training_calls"), manifest.get("inference_calls"), manifest.get("model_objects_deserialized")] == [0, 0, 0], "3P model activity")
    require([manifest.get("repair_site_test_access"), manifest.get("dev_site_test_access"), manifest.get("validation_access"), manifest.get("holdout_access")] == [0, 0, 0, 0], "3P protected access")
    outputs = manifest.get("outputs")
    require(isinstance(outputs, dict), "3P output registry")
    for path in (POLICY_3P, FINAL_LOCK_3P, COMPARISON_3P, REGISTRY_CSV_3P, REGISTRY_JSON_3P, REPORT_3P):
        verify_record(outputs.get(rel(path)), path, path.name)

    require(audit.get("status") == "PASS" and audit.get("disposition_status") == "FROZEN", "3P audit")
    require(audit.get("locked_evaluation_acceptance") == "NOT_MET / PRESERVED", "audit acceptance")
    require(audit.get("repair_site_test") == "CONSUMED / FROZEN / DO NOT REOPEN", "audit test lock")
    require(audit.get("dev_site_test_validation_holdout_access") == [0, 0, 0], "audit protected access")
    require(audit.get("next_gate_authorization") == "CONTRACT CREATION ONLY; DATASET AND TRAINING NOT AUTHORIZED", "audit next authorization")
    require(audit.get("future_combined_model_brand") == FUTURE_BRAND, "audit brand reservation")

    require(readiness.get("status") == "FROZEN", "12B-1G readiness")
    require(readiness.get("v2_2_contract_creation") == "AUTHORIZED", "V2.2 contract authorization")
    require(readiness.get("v2_2_dataset_construction") == "NOT YET AUTHORIZED", "V2.2 dataset boundary")
    require(readiness.get("v2_2_model_training") == "NOT YET AUTHORIZED", "V2.2 training boundary")
    require(v1_lock.get("status") == "PASS", "V1 remote lock status")
    require(isinstance(graph_schema, dict) and graph_schema, "graph schema")

    for path in (POLICY_3P, FINAL_LOCK_3P, COMPARISON_3P, REGISTRY_JSON_3P, MANIFEST_3P, AUDIT_3P, READINESS_12B1G, V1_REMOTE_LOCK):
        require(canonical_json(load_json(path)) == path.read_bytes(), f"canonical JSON replay: {path.name}")
    print("  V2.1 final freeze, consumed tests, V1 lock and graph-schema lineage                    : PASS")
    return audit


def self_test() -> None:
    require(MINIMUM_CIRCUIT_FAMILIES == 7, "minimum family arithmetic")
    require(PARTITIONS["INDEPENDENT_CIRCUIT_TEST"] >= 2, "independent-test circuit count")
    require(PARTITIONS["GENERALIZATION_HOLDOUT"] >= 1, "holdout circuit count")
    sample = {"z": 1, "a": [2, 3]}
    require(canonical_json(sample) == canonical_json(json.loads(canonical_json(sample))), "canonical JSON")
    print("Stage 12C-1A self-test: PASS")


def main() -> None:
    require(ROOT.name == "vlsi_fault_detection_v2", f"run from project root, not {ROOT}")
    outputs = (SCOPE, ARCHITECTURE, PARTITION, INTERFACE, TRAINING, ACCEPTANCE,
               CANDIDATE_GRID, ENVIRONMENT, REPORT, MANIFEST, AUDIT)
    for path in outputs:
        require(not path.exists(), f"Stage {STAGE} output already exists: {rel(path)}")

    predecessor_audit = verify_inputs()
    timestamp = now()

    scope = {
        "contract_version": "CIRCUITSAGE-HMAC-V2.2-GENERALIZATION-SCOPE-12C1A-v1",
        "stage": STAGE,
        "status": "FROZEN",
        "project": "CircuitSage-HMAC V2.2 independent-generalization research branch",
        "objective": "DETECT AND LOCALIZE SINGLE PERSISTENT SA0/SA1 FAULTS ON CIRCUIT FAMILIES ABSENT FROM TRAINING",
        "primary_generalization_axes": [
            "unseen circuit families",
            "unseen physical sites",
            "unseen response signatures",
            "unseen synthesis instances grouped with their parent circuit",
        ],
        "fault_scope": "SINGLE PERSISTENT SA0/SA1 ONLY",
        "out_of_scope": [
            "multiple simultaneous faults",
            "delay, transient, bridging, analog, power, thermal, and physical-defect faults",
            "production or physical-silicon readiness",
            "self-learning or online model updates",
        ],
        "query_fault_identity": "PROHIBITED",
        "prohibited_query_fields": ["fault_id", "site_id", "selector", "stuck_value", "injected_node_index", "partition_truth"],
        "allowed_query_evidence": [
            "golden-versus-observed external response deltas",
            "timing/status/timeout masks defined before capture",
            "fault-identity-independent internal probe deltas when licensed and pre-registered",
            "normalized netlist graph features available at deployment",
        ],
        "required_output_semantics": ["FAULT_FREE", "FAULT_DETECTED", "AMBIGUOUS_CANDIDATE_SET", "UNKNOWN_OR_OUT_OF_DISTRIBUTION"],
        "v1_role": "OPTIONAL FROZEN VERIFICATION SUPPORT AFTER V2.2 DETECTION/LOCALIZATION; NEVER A SOURCE OF TEST LABELS",
        "v2_1_role": "IMMUTABLE CLOSED-CATALOG COMPARATOR",
        "future_combined_release_brand": FUTURE_BRAND,
        "brand_boundary": "RESERVED; V2.2 IS NOT YET FAULTIVA",
        "dataset_construction": "NOT AUTHORIZED BY THIS STAGE",
        "training_or_inference": "NOT AUTHORIZED BY THIS STAGE",
    }

    architecture = {
        "architecture_version": "CIRCUITSAGE-HMAC-V2.2-GENERALIZATION-ARCHITECTURE-12C1A-v1",
        "stage": STAGE,
        "status": "FROZEN",
        "model_family": "CIRCUIT-NORMALIZED GRAPH ENCODER + RESPONSE ENCODER + CROSS-MODAL RETRIEVAL/RERANKING",
        "components": {
            "graph_encoder": {
                "input": "normalized directed netlist graph with portable semantic node/edge features",
                "families": ["GraphSAGE", "GATv2"],
                "absolute_node_identity": "PROHIBITED",
                "maximum_trainable_parameters": 1500000,
            },
            "response_encoder": {
                "input": "external/probe delta tensors and validity masks only",
                "architecture": "masked temporal MLP or 1D convolution with circuit-independent dimensions",
                "fault_identity": "ABSENT",
            },
            "fusion": "behavior-to-node metric retrieval followed by graph-constrained reranking",
            "detector_head": "calibrated fault probability with fixed fault-free threshold",
            "locator_head": "ranked node/candidate-set probabilities with ambiguity preservation",
            "polarity_head": "SA0/SA1 probability only after a candidate site is proposed",
            "open_set_head": "unknown-signature/OOD score and mandatory abstention",
            "verification_adapter": "optional frozen V1 support when its feature contract is valid; otherwise unavailable",
        },
        "normalization": [
            "graph statistics normalized per circuit using TRAIN-frozen rules",
            "no circuit name, path, source label, site index, or partition token",
            "response lengths represented by masks rather than circuit identity",
        ],
        "ambiguity_rule": "UNIQUE OUTPUT ONLY WHEN CALIBRATED POSTERIOR AND MARGIN GATES PASS; OTHERWISE RETURN A SET OR ABSTAIN",
        "serialization": "NPZ/JSON or safe tensor format preferred; executable pickle prohibited for public release",
    }

    partition = {
        "partition_contract_version": "CIRCUITSAGE-HMAC-V2.2-CIRCUIT-FAMILY-PARTITION-12C1A-v1",
        "stage": STAGE,
        "status": "FROZEN",
        "split_unit": "CIRCUIT FAMILY — NOT SITE, FAULT, VECTOR, NETLIST, OR SYNTHESIS SEED",
        "minimum_independent_circuit_families": MINIMUM_CIRCUIT_FAMILIES,
        "minimum_family_counts": PARTITIONS,
        "family_definition": "same RTL ancestry, fork, parameterization, derivative, generated variant, or functionally equivalent implementation remains one family",
        "synthesis_variants": "ALL VARIANTS OF ONE FAMILY MUST REMAIN IN ONE PARTITION",
        "split_order": "LICENSE/PROVENANCE ELIGIBILITY -> FAMILY GROUPING -> HASH-BASED PARTITION -> SYNTHESIS -> FAULT INJECTION",
        "split_hash_domain": "CIRCUITSAGE-HMAC-V2.2-CIRCUIT-FAMILY-SPLIT-12C1A-v1",
        "partition_access": {
            "GENERALIZATION_TRAIN": "dataset construction requires separate authorization",
            "GENERALIZATION_CALIBRATION": "locked until capture authorization; no gradient updates",
            "INDEPENDENT_CIRCUIT_TEST": "sealed until one committed final evaluation",
            "GENERALIZATION_HOLDOUT": "blocked; excluded from V2.2 development and final test",
        },
        "minimum_per_family_reporting": ["fault-free cases", "injected faults", "observable ceiling", "detection", "localization", "abstention", "false alarms"],
        "leakage_prohibitions": [
            "no source clone or near-duplicate across partitions",
            "no fitted scaler, threshold, vocabulary, probe selector, or feature selector using test/holdout",
            "no test or holdout vector optimization",
            "no test-label inspection before prediction commitment",
        ],
        "license_and_provenance": "EVERY RTL/NETLIST FAMILY MUST HAVE SOURCE URL/COMMIT, SPDX LICENSE, REDISTRIBUTION DECISION, AND RESTRICTED-MATERIAL REVIEW",
    }

    interface = {
        "interface_version": "FAULTIVA-RESEARCH-INFERENCE-INTERFACE-12C1A-v1",
        "stage": STAGE,
        "status": "FROZEN",
        "input": {
            "circuit_graph": "portable node features, directed edges, graph metadata and schema version",
            "golden_response": "expected external/probe response tensor",
            "observed_response": "captured external/probe response tensor",
            "validity_mask": "valid vectors, cycles, probes and timeout/status fields",
            "test_vector_commitment": "hash binding the ordered vector schedule",
        },
        "output": {
            "status": ["FAULT_FREE", "FAULT_DETECTED", "AMBIGUOUS_CANDIDATE_SET", "UNKNOWN_OR_OUT_OF_DISTRIBUTION"],
            "fault_probability": "float [0,1]",
            "candidate_nodes": "ordered list of portable node references with scores",
            "candidate_set_complete": "boolean under the frozen catalog and threshold contract",
            "polarity_probability": "SA0/SA1 distribution when supported",
            "ood_score": "float with frozen rejection threshold",
            "confidence": "calibrated score plus explicit calibration-scope label",
            "evidence": "vector/probe groups contributing to the decision without exposing fault identity",
        },
        "online_learning": "DISABLED",
        "model_or_threshold_update_during_inference": "PROHIBITED",
        "unsupported_input_behavior": "REJECT WITH SCHEMA OR OOD STATUS; DO NOT SILENTLY COERCE",
    }

    training = {
        "training_contract_version": "CIRCUITSAGE-HMAC-V2.2-TRAINING-SELECTION-12C1A-v1",
        "stage": STAGE,
        "status": "FROZEN",
        "training": "NOT YET AUTHORIZED",
        "selection_partition": "GENERALIZATION_CALIBRATION ONLY AFTER SEPARATE CAPTURE AUTHORIZATION",
        "gradient_partitions": ["GENERALIZATION_TRAIN"],
        "gradient_prohibited_partitions": ["GENERALIZATION_CALIBRATION", "INDEPENDENT_CIRCUIT_TEST", "GENERALIZATION_HOLDOUT"],
        "objectives": [
            "fault-free versus faulty detection",
            "contrastive behavior-to-faulty-node retrieval",
            "pairwise graph reranking",
            "candidate-set coverage",
            "polarity classification conditioned on supported localization",
            "OOD rejection and calibration",
        ],
        "hard_negatives": "full-catalog or deterministic topology-stratified negatives within TRAIN circuits only",
        "early_stopping": "circuit-macro calibration metric with pre-registered patience",
        "selection_order": ["safety", "OOD/false alarms", "circuit-macro detection", "candidate coverage", "MRR/exact site", "simplicity"],
        "determinism": "fixed seeds, deterministic data order, single declared environment and exact replay required",
        "after_selection": "weights, scalers, thresholds, vector schedule and probe policy frozen before independent test",
    }

    acceptance = {
        "acceptance_contract_version": "CIRCUITSAGE-HMAC-V2.2-INDEPENDENT-GENERALIZATION-ACCEPTANCE-12C1A-v1",
        "stage": STAGE,
        "status": "FROZEN",
        "scientific_claim": "INDEPENDENT-CIRCUIT GENERALIZATION IS ESTABLISHED ONLY IF EVERY MANDATORY GATE PASSES ON BOTH LOCKED TEST CIRCUITS",
        "mandatory_test_population": "at least two circuit families absent from training, calibration and architecture tuning",
        "macro_gates": {
            "fault_free_false_alarm_rate_max": 0.01,
            "observable_detection_recall_min": 0.90,
            "all_injected_detection_recall_min": 0.60,
            "observable_candidate_set_coverage_min": 0.90,
            "unique_signature_top1_site_min": 0.70,
            "all_injected_exact_site_rate_min": 0.25,
            "observable_mrr_min": 0.30,
            "ambiguous_false_unique_rate_max": 0.01,
            "polarity_accuracy_given_correct_unique_site_min": 0.80,
            "ood_auroc_min": 0.75,
        },
        "per_circuit_floor": {
            "fault_free_false_alarm_rate_max": 0.02,
            "observable_detection_recall_min": 0.80,
            "observable_candidate_set_coverage_min": 0.80,
            "all_injected_exact_site_rate_min": 0.15,
        },
        "uncertainty": "hierarchical bootstrap by circuit then physical site; 1000 replicates; 95% intervals",
        "required_comparators": [
            "fault-free exact golden-reference detector",
            "non-learning exact-signature retrieval",
            "topology-only graph baseline",
            "frozen V2.1 where interface-compatible",
        ],
        "prediction_commitment": "REQUIRED BEFORE OPENING INDEPENDENT_TEST TRUTH",
        "one_shot_rule": "ONE LOCKED TEST EVALUATION; NO RETRAINING, THRESHOLD CHANGE, OR RESELECTION AFTERWARD",
        "failure_disposition": "FREEZE NOT_ESTABLISHED; DO NOT REOPEN TEST; DESIGN A NEW TRAIN/CALIBRATION-ONLY STUDY",
        "holdout": "REMAINS BLOCKED EVEN IF V2.2 PASSES",
        "production_readiness": "NOT ESTABLISHED BY THIS CONTRACT",
    }

    candidate_rows = [
        {
            "candidate_id": "V22_EXACT_SIGNATURE_GRAPH_BASELINE",
            "trainable": "NO",
            "graph_encoder": "NONE",
            "response_encoder": "EXACT HASH",
            "fusion": "GRAPH-CONSTRAINED EXACT RETRIEVAL",
            "parameter_cap": "0",
            "role": "mandatory non-learning comparator",
        },
        {
            "candidate_id": "V22_GRAPHSAGE_METRIC_SMALL",
            "trainable": "YES",
            "graph_encoder": "GraphSAGE 3-layer",
            "response_encoder": "masked MLP",
            "fusion": "contrastive metric retrieval + pairwise reranker",
            "parameter_cap": "750000",
            "role": "primary compact candidate",
        },
        {
            "candidate_id": "V22_GATV2_CROSS_FUSION",
            "trainable": "YES",
            "graph_encoder": "GATv2 3-layer",
            "response_encoder": "masked temporal convolution",
            "fusion": "cross-attention + graph reranker",
            "parameter_cap": "1500000",
            "role": "higher-capacity candidate",
        },
        {
            "candidate_id": "V22_GRAPHSAGE_OOD_ENSEMBLE",
            "trainable": "YES",
            "graph_encoder": "GraphSAGE shared encoder",
            "response_encoder": "masked MLP ensemble",
            "fusion": "metric retrieval + calibrated OOD ensemble",
            "parameter_cap": "1500000",
            "role": "uncertainty-focused candidate",
        },
    ]
    candidate_fields = ["candidate_id", "trainable", "graph_encoder", "response_encoder", "fusion", "parameter_cap", "role"]
    candidate_payload = csv_bytes(candidate_rows, candidate_fields)

    environment = {
        "environment_version": "CIRCUITSAGE-HMAC-V2.2-CONTRACT-ENVIRONMENT-12C1A-v1",
        "stage": STAGE,
        "status": "FROZEN",
        "created_at": timestamp,
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "contracts_only": True,
        "rtl_or_netlists_read": 0,
        "datasets_read": 0,
        "models_deserialized": 0,
        "training_calls": 0,
        "inference_calls": 0,
        "dev_site_test_access": 0,
        "repair_site_test_access": 0,
        "validation_access": 0,
        "holdout_access": 0,
    }

    frozen_write(SCOPE, canonical_json(scope))
    frozen_write(ARCHITECTURE, canonical_json(architecture))
    frozen_write(PARTITION, canonical_json(partition))
    frozen_write(INTERFACE, canonical_json(interface))
    frozen_write(TRAINING, canonical_json(training))
    frozen_write(ACCEPTANCE, canonical_json(acceptance))
    frozen_write(CANDIDATE_GRID, candidate_payload)
    frozen_write(ENVIRONMENT, canonical_json(environment))

    report = f"""# CircuitSage-HMAC V2.2 Generalization Contract — Stage 12C-1A

## Purpose

V2.2 is a separate research branch intended to test whether a portable
graph-and-behavior model can detect and localize persistent SA0/SA1 faults on
circuit families that are absent from training. It does not reopen or repair
the consumed V2.1 test sets.

## Frozen design

- Minimum independent circuit families: {MINIMUM_CIRCUIT_FAMILIES}
- Family split: {PARTITIONS['GENERALIZATION_TRAIN']} TRAIN, {PARTITIONS['GENERALIZATION_CALIBRATION']} CALIBRATION, {PARTITIONS['INDEPENDENT_CIRCUIT_TEST']} locked TEST, {PARTITIONS['GENERALIZATION_HOLDOUT']} blocked HOLDOUT
- Model: circuit-normalized graph encoder + response encoder + metric retrieval/reranking
- Query fault identity: prohibited
- Ambiguous and unknown/OOD outputs: mandatory
- Online learning: disabled
- V1: optional frozen verification support only
- V2.1: immutable closed-catalog comparator

## Claim boundary

Passing calibration is not generalization. Independent-circuit generalization
may be claimed only after predictions are committed and every mandatory macro
and per-circuit gate passes on at least two locked circuit families. HOLDOUT
remains blocked even after a V2.2 pass.

## Current authorization

This stage authorizes contracts only. It does not authorize downloading RTL,
constructing datasets, synthesizing circuits, injecting faults, selecting
probes, training models, or evaluating protected partitions.

## Naming

**{FUTURE_BRAND}** remains reserved for the eventual completed V1+V2 hybrid
release. The V2.2 research component must not use that release identity yet.
""".encode()
    frozen_write(REPORT, report)

    contract_outputs = [SCOPE, ARCHITECTURE, PARTITION, INTERFACE, TRAINING, ACCEPTANCE, CANDIDATE_GRID, ENVIRONMENT, REPORT]
    manifest = {
        "manifest_version": "CIRCUITSAGE-HMAC-V2.2-GENERALIZATION-CONTRACT-MANIFEST-12C1A-v1",
        "stage": STAGE,
        "status": "PASS",
        "created_at": timestamp,
        "frozen_inputs": {rel(path): record(path) for path in PINNED},
        "outputs": {rel(path): record(path) for path in contract_outputs},
        "minimum_circuit_families": MINIMUM_CIRCUIT_FAMILIES,
        "dataset_construction_authorized": False,
        "training_authorized": False,
        "evaluation_authorized": False,
        "rtl_or_netlists_read": 0,
        "dataset_payloads_read": 0,
        "models_deserialized": 0,
        "training_calls": 0,
        "inference_calls": 0,
        "repair_site_test_access": 0,
        "dev_site_test_access": 0,
        "validation_access": 0,
        "holdout_access": 0,
    }
    frozen_write(MANIFEST, canonical_json(manifest))

    audit = {
        "audit_version": "CIRCUITSAGE-HMAC-V2.2-GENERALIZATION-CONTRACT-FREEZE-12C1A-v1",
        "stage": STAGE,
        "status": "PASS",
        "architecture_status": "FROZEN",
        "partition_contract_status": "FROZEN",
        "training_contract_status": "FROZEN",
        "acceptance_contract_status": "FROZEN",
        "scientific_scope": "INDEPENDENT-CIRCUIT SINGLE PERSISTENT SA0/SA1 GENERALIZATION RESEARCH",
        "model_family": architecture["model_family"],
        "candidate_models": len(candidate_rows),
        "trainable_candidates": sum(row["trainable"] == "YES" for row in candidate_rows),
        "minimum_circuit_families": MINIMUM_CIRCUIT_FAMILIES,
        "minimum_family_partition_counts": PARTITIONS,
        "query_fault_identity": "PROHIBITED",
        "unknown_ood_abstention": "MANDATORY",
        "v2_1_locked_acceptance": predecessor_audit["locked_evaluation_acceptance"],
        "v2_1_repair_site_test": "CONSUMED / FROZEN / NOT REOPENED",
        "v1_v2_core_v2_1_modified": [False, False, False],
        "dataset_construction": "NOT AUTHORIZED",
        "training_selection_evaluation": "NOT AUTHORIZED / NOT AUTHORIZED / NOT AUTHORIZED",
        "validation_holdout_access": [0, 0],
        "independent_generalization": "CONTRACTED / NOT YET EVALUATED",
        "future_combined_model_brand": FUTURE_BRAND,
        "scope_contract": record(SCOPE),
        "architecture": record(ARCHITECTURE),
        "partition_contract": record(PARTITION),
        "interface_contract": record(INTERFACE),
        "training_contract": record(TRAINING),
        "acceptance_contract": record(ACCEPTANCE),
        "candidate_grid": record(CANDIDATE_GRID),
        "report": record(REPORT),
        "manifest": record(MANIFEST),
        "next_gate": "STAGE 12C-1B — MULTI-CIRCUIT RTL CORPUS, LICENSE-PROVENANCE, AND FAMILY-SPLIT AUTHORIZATION FREEZE",
    }
    frozen_write(AUDIT, canonical_json(audit))

    for path in (SCOPE, ARCHITECTURE, PARTITION, INTERFACE, TRAINING, ACCEPTANCE, ENVIRONMENT, MANIFEST, AUDIT):
        require(canonical_json(load_json(path)) == path.read_bytes(), f"canonical output replay: {path.name}")
    require(CANDIDATE_GRID.read_bytes() == candidate_payload, "candidate-grid replay")
    require(REPORT.read_bytes() == report, "report replay")
    for path, expected in PINNED.items():
        require(sha256(path) == expected, f"frozen input modified: {rel(path)}")

    print("\nSTAGE 12C-1A — V2.2 GENERALIZATION ARCHITECTURE, DATA-PARTITION, AND ACCEPTANCE-CONTRACT FREEZE")
    print(f"{'Status':<73}: PASS")
    print(f"{'Architecture / partition / acceptance':<73}: FROZEN / FROZEN / FROZEN")
    print(f"{'Scientific scope':<73}: INDEPENDENT-CIRCUIT SINGLE PERSISTENT SA0/SA1 RESEARCH")
    print(f"{'Model family':<73}: GRAPH + RESPONSE ENCODER + RETRIEVAL/RERANKING")
    print(f"{'Candidates / trainable':<73}: {len(candidate_rows)} / {sum(row['trainable'] == 'YES' for row in candidate_rows)}")
    print(f"{'Minimum circuit families':<73}: {MINIMUM_CIRCUIT_FAMILIES}")
    print(f"{'TRAIN / CALIBRATION / TEST / HOLDOUT families':<73}: 3 / 1 / 2 / 1")
    print(f"{'Query fault identity':<73}: PROHIBITED")
    print(f"{'OOD rejection / ambiguity preservation':<73}: MANDATORY / MANDATORY")
    print(f"{'V2.1 locked result':<73}: NOT_MET / PRESERVED / NOT REOPENED")
    print(f"{'Dataset construction / model training':<73}: NOT AUTHORIZED / NOT AUTHORIZED")
    print(f"{'Protected dataset access':<73}: 0")
    print(f"{'Future combined-model brand':<73}: {FUTURE_BRAND}")
    print(f"{'Scope contract':<73}: {SCOPE}")
    print(f"{'Scope SHA':<73}: {sha256(SCOPE)}")
    print(f"{'Architecture':<73}: {ARCHITECTURE}")
    print(f"{'Architecture SHA':<73}: {sha256(ARCHITECTURE)}")
    print(f"{'Partition contract':<73}: {PARTITION}")
    print(f"{'Partition SHA':<73}: {sha256(PARTITION)}")
    print(f"{'Acceptance contract':<73}: {ACCEPTANCE}")
    print(f"{'Acceptance SHA':<73}: {sha256(ACCEPTANCE)}")
    print(f"{'Candidate grid':<73}: {CANDIDATE_GRID}")
    print(f"{'Candidate grid SHA':<73}: {sha256(CANDIDATE_GRID)}")
    print(f"{'Manifest':<73}: {MANIFEST}")
    print(f"{'Manifest SHA':<73}: {sha256(MANIFEST)}")
    print(f"{'Audit':<73}: {AUDIT}")
    print(f"{'Audit SHA':<73}: {sha256(AUDIT)}")
    print(f"{'Next gate':<73}: STAGE 12C-1B — MULTI-CIRCUIT RTL CORPUS, LICENSE-PROVENANCE, AND FAMILY-SPLIT AUTHORIZATION FREEZE")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        main()
