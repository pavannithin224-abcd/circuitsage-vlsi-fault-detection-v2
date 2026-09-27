#!/usr/bin/env python3
"""Stage 12B-2A: V2.1 observability/localization improvement contract freeze.

Creates a new repair branch without modifying frozen V1, V2 Core, or V2.1.
It freezes the scope, a fresh site split drawn only from original DEV_TRAIN,
deterministic development-vector policy, architecture direction, and acceptance
criteria. No response payload is generated and no model is trained or run.
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
from pathlib import Path
from typing import Any

try:
    import numpy as np
except ImportError as error:
    raise SystemExit("STOP: activate the project .venv (NumPy required)") from error


STAGE = "12B-2A"
ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "config/v2_1_improvement"
RESULT = ROOT / "results/circuitsage_hmac_v2_12b2"

DISPOSITION_SOURCE = ROOT / "stage_12b1g_v2_1_disposition_generalization_readiness.py"
V21_POLICY = ROOT / "config/v2_1/circuitsage_hmac_v2_1_final_disposition_policy_12b1g.json"
V22_READINESS = ROOT / "config/v2_2/circuitsage_hmac_v2_2_generalization_readiness_policy_12b1g.json"
V21_REGISTRY_CSV = ROOT / "results/circuitsage_hmac_v2_12b1/circuitsage_hmac_v2_1_capability_registry_12b1g.csv"
V21_REGISTRY_JSON = ROOT / "results/circuitsage_hmac_v2_12b1/circuitsage_hmac_v2_1_capability_registry_12b1g.json"
V21_REPORT = ROOT / "results/circuitsage_hmac_v2_12b1/circuitsage_hmac_v2_1_final_report_12b1g.md"
V21_MANIFEST = ROOT / "results/circuitsage_hmac_v2_12b1/circuitsage_hmac_v2_1_disposition_manifest_12b1g.json"
V21_AUDIT = ROOT / "results/circuitsage_hmac_v2_12b1/circuitsage_hmac_v2_1_disposition_generalization_readiness_freeze_12b1g.json"

ORIGINAL_SPLIT = ROOT / "results/hmac_fault_campaign_11c5/hmac_fault_site_group_split_11c5d.csv"
ORIGINAL_SPLIT_MANIFEST = ROOT / "results/hmac_fault_campaign_11c5/hmac_fault_site_group_split_manifest_11c5d.json"
GRAPH = ROOT / "results/hmac_fault_campaign_11d1/graph_dataset_11d1a/hmac_golden_netlist_graph_11d1a.npz"
GRAPH_SCHEMA = ROOT / "results/hmac_fault_campaign_11d1/graph_dataset_11d1a/hmac_golden_netlist_graph_schema_11d1a.json"
GRAPH_AUDIT = ROOT / "results/hmac_fault_campaign_11d1/hmac_golden_netlist_graph_topology_integrity_freeze_11d1a.json"
V1_REMOTE_LOCK = ROOT / "config/release/hmac_github_remote_integrity_lock_11e1k.json"

SCOPE = CONFIG / "circuitsage_hmac_v2_1_improvement_scope_contract_12b2a.json"
PARTITION_CONTRACT = CONFIG / "circuitsage_hmac_v2_1_repair_partition_contract_12b2a.json"
VECTOR_POLICY = CONFIG / "circuitsage_hmac_v2_1_adaptive_vector_policy_12b2a.json"
ARCHITECTURE = CONFIG / "circuitsage_hmac_v2_1_repair_architecture_contract_12b2a.json"
ACCEPTANCE = CONFIG / "circuitsage_hmac_v2_1_repair_acceptance_contract_12b2a.json"
CANDIDATE_GRID = CONFIG / "circuitsage_hmac_v2_1_repair_candidate_grid_12b2a.csv"
REPAIR_SPLIT = RESULT / "circuitsage_hmac_v2_1_repair_site_split_12b2a.csv"
ENVIRONMENT = RESULT / "circuitsage_hmac_v2_1_improvement_environment_12b2a.json"
MANIFEST = RESULT / "circuitsage_hmac_v2_1_improvement_contract_manifest_12b2a.json"
AUDIT = RESULT / "circuitsage_hmac_v2_1_improvement_contract_freeze_12b2a.json"

PINNED = {
    DISPOSITION_SOURCE: "a9dbf56cc8e7ed1fe3495297dec924e1e562eefdc7c7dd313f482d0ed5c431c2",
    V21_POLICY: "a041be430e0b9238c8d0f20aebfaa60f542ebb33a9ccc6fdc39039aeb32d1c11",
    V22_READINESS: "08e2320e70c4d30d90b045521824f7175a5dd3b8595d6244f5711d8d9e9ffc85",
    V21_REGISTRY_CSV: "5c8fc6992b1cfcbdb90f427f2b28807c1a4830cfecc65cfaa7854a693fa55025",
    V21_REGISTRY_JSON: "51749922926dab7531dceb9c4a164cfe0bd727ec8a07671b11763947251ba6cd",
    V21_REPORT: "8ebe509066e3055a37cdcaeecf78582e27874c248cec61e5fe975935a48cf404",
    V21_MANIFEST: "92dec9ac010fa836b6c3922d36640722aeb15c0c3f2608dfe46e0784074b266c",
    V21_AUDIT: "d238465921cddf866cc8f72a9552948e95e5c06566aacb48855bf307fb367b22",
    ORIGINAL_SPLIT: "602fa310547f68d1e9b5ceb6f148d6b125c69df0589929f9edfde9d264922c61",
    ORIGINAL_SPLIT_MANIFEST: "1c96b7c7c7f5301e70037aaa9a0d18effd28cae6bf15e9f85cc85f0c413473fb",
    GRAPH: "e3c2dd2214b544231186c29d8d9cb5aa6621b4d4b4bc9002150ac4f6c208c052",
    GRAPH_SCHEMA: "d24c3dd285891e284d20d1adcd93f793bedfd3042f20159a8a486359bc25e393",
    GRAPH_AUDIT: "4b9aef6468b467350667338373c28dc5797e3f59ce989af71a18369f086dfeb9",
    V1_REMOTE_LOCK: "3d93cd8bb9c3bba47dd665ba6a45ad4e193613c1f6d386e32b3265471d94dd5a",
}

ORIGINAL_TRAIN_SITES = 15987
REPAIR_TRAIN_SITES = 11191
REPAIR_CALIBRATION_SITES = 2398
REPAIR_SITE_TEST_SITES = 2398
SPLIT_DOMAIN = b"CIRCUITSAGE-HMAC-V2.1-REPAIR-SITE-SPLIT-12B2A-v1\0"
VECTOR_DOMAIN = "CIRCUITSAGE-HMAC-V2.1-ADAPTIVE-DEVELOPMENT-VECTORS-12B2A-v1"
VECTOR_SEED = 20260916
CANDIDATE_VECTOR_POOL = 512
PILOT_SITES = 1024
MAX_SELECTED_VECTORS = 96
SELECTION_ROUNDS = 4
VECTORS_PER_ROUND = 24


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


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON object required: {rel(path)}")
    return value


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode()


def frozen_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.exists(), f"refusing to overwrite frozen output: {rel(path)}")
    temporary = path.with_name(path.name + ".tmp")
    require(not temporary.exists(), f"stale temporary output: {rel(temporary)}")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def record(path: Path) -> dict[str, Any]:
    return {"path": rel(path), "sha256": sha256(path), "bytes": path.stat().st_size}


def require_canonical(path: Path) -> None:
    require(canonical_json(load_json(path)) == path.read_bytes(), f"canonical replay: {path.name}")


def verify_inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    print("STAGE 12B-2A — V2.1 OBSERVABILITY AND LOCALIZATION IMPROVEMENT CONTRACT")
    print("FROZEN INPUT VERIFICATION")
    evidence: dict[str, Any] = {}
    for path, expected in PINNED.items():
        require(path.is_file(), f"missing frozen input: {rel(path)}")
        require(sha256(path) == expected, f"frozen input SHA: {rel(path)}")
        evidence[rel(path)] = record(path)
        print(f"  {path.name:<91}: OK", flush=True)
    # Frozen inputs come from several earlier stages whose JSON writers used
    # different, but valid, formatting conventions.  Their pinned SHA-256
    # values above are the byte-level integrity contract; do not reject an
    # otherwise verified legacy artifact merely because its whitespace or key
    # order differs from this stage's canonical_json() formatter.  Outputs
    # created by this stage are still canonical-replay checked below.
    disposition = load_json(V21_POLICY)
    readiness = load_json(V22_READINESS)
    audit = load_json(V21_AUDIT)
    graph_audit = load_json(GRAPH_AUDIT)
    v1_lock = load_json(V1_REMOTE_LOCK)
    require(disposition.get("status") == "FROZEN", "V2.1 disposition status")
    require(disposition.get("lifecycle") == "COMPLETED AND FROZEN CLOSED-CATALOG RESEARCH COMPONENT", "V2.1 lifecycle")
    require(disposition.get("dev_site_test") == "CONSUMED AND FROZEN", "V2.1 test state")
    require(readiness.get("v2_2_contract_creation") == "AUTHORIZED", "generalization contract authorization")
    require(readiness.get("v2_2_dataset_construction") == "NOT YET AUTHORIZED", "dataset authorization boundary")
    require(readiness.get("v2_2_model_training") == "NOT YET AUTHORIZED", "training authorization boundary")
    require(audit.get("status") == "PASS", "V2.1 audit status")
    require(audit.get("dev_site_test") == "CONSUMED AND FROZEN / REOPENING PROHIBITED", "consumed test protection")
    require(audit.get("validation_access_count") == 0 and audit.get("holdout_access_count") == 0, "locked partition access")
    require(graph_audit.get("status") == "PASS", "graph freeze status")
    require(v1_lock.get("status") == "PASS", "V1 remote lock status")
    print("  Frozen V1/V2 lineage, consumed test protection and graph integrity                         : PASS")
    return evidence, disposition


def load_original_train_sites() -> list[dict[str, Any]]:
    with ORIGINAL_SPLIT.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"site_id", "site_index", "partition", "partition_rank", "split_hash_sha256"}
        require(set(reader.fieldnames or []) == required, "original split header")
        rows = [row for row in reader if row["partition"] == "DEV_TRAIN"]
    require(len(rows) == ORIGINAL_TRAIN_SITES, "original DEV_TRAIN site count")
    require(len({row["site_id"] for row in rows}) == len(rows), "duplicate original train site")
    return rows


def repair_split(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], bytes]:
    ranked = []
    for row in rows:
        site_id = row["site_id"]
        digest = hashlib.sha256(SPLIT_DOMAIN + site_id.encode()).hexdigest()
        ranked.append((digest, int(row["site_index"]), site_id))
    ranked.sort()
    output_rows = []
    counts = {"REPAIR_TRAIN": 0, "REPAIR_CALIBRATION": 0, "REPAIR_SITE_TEST": 0}
    for global_rank, (digest, site_index, site_id) in enumerate(ranked):
        if global_rank < REPAIR_TRAIN_SITES:
            partition = "REPAIR_TRAIN"
        elif global_rank < REPAIR_TRAIN_SITES + REPAIR_CALIBRATION_SITES:
            partition = "REPAIR_CALIBRATION"
        else:
            partition = "REPAIR_SITE_TEST"
        partition_rank = counts[partition]
        counts[partition] += 1
        output_rows.append({
            "site_id": site_id,
            "site_index": site_index,
            "repair_partition": partition,
            "repair_partition_rank": partition_rank,
            "repair_split_hash_sha256": digest,
            "source_partition": "DEV_TRAIN",
        })
    require(counts == {
        "REPAIR_TRAIN": REPAIR_TRAIN_SITES,
        "REPAIR_CALIBRATION": REPAIR_CALIBRATION_SITES,
        "REPAIR_SITE_TEST": REPAIR_SITE_TEST_SITES,
    }, "repair split counts")
    ordered = sorted(output_rows, key=lambda row: row["site_index"])
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(ordered[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(ordered)
    return ordered, stream.getvalue().encode()


def candidate_grid_csv() -> bytes:
    rows = [
        {
            "candidate_id": "REPAIR_GRAPH_RULE_RERANK",
            "trainable": "NO",
            "response_encoder": "EXACT_PLUS_WEIGHTED_DISTANCE",
            "candidate_generator": "GRAPH_CONE_AND_OBSERVABILITY_FILTER",
            "reranker": "DETERMINISTIC_RULE",
            "maximum_candidates": 500,
        },
        {
            "candidate_id": "REPAIR_DUAL_ENCODER_METRIC_SMALL",
            "trainable": "YES",
            "response_encoder": "MLP_256_128_64",
            "candidate_generator": "GRAPH_CONE_AND_TOPK_RETRIEVAL",
            "reranker": "COSINE_METRIC_PLUS_V1_SUPPORT",
            "maximum_candidates": 500,
        },
        {
            "candidate_id": "REPAIR_GRAPH_CROSS_RERANK_SMALL",
            "trainable": "YES",
            "response_encoder": "MLP_256_128_64",
            "candidate_generator": "GRAPH_CONE_AND_TOPK_RETRIEVAL",
            "reranker": "PAIRWISE_MLP_128_64_1",
            "maximum_candidates": 500,
        },
    ]
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode()


def self_test() -> None:
    sample = [
        {"site_id": f"HMAC-STEM-{index:06d}", "site_index": str(index),
         "partition": "DEV_TRAIN", "partition_rank": str(index - 1), "split_hash_sha256": "0" * 64}
        for index in range(1, ORIGINAL_TRAIN_SITES + 1)
    ]
    rows, payload = repair_split(sample)
    require(len(rows) == ORIGINAL_TRAIN_SITES and payload == repair_split(sample)[1], "split replay canary")
    require(candidate_grid_csv() == candidate_grid_csv(), "candidate-grid replay canary")
    print("Stage 12B-2A self-test: PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    require(ROOT.name == "vlsi_fault_detection_v2", f"run from project root, not {ROOT}")
    outputs = (SCOPE, PARTITION_CONTRACT, VECTOR_POLICY, ARCHITECTURE, ACCEPTANCE,
               CANDIDATE_GRID, REPAIR_SPLIT, ENVIRONMENT, MANIFEST, AUDIT)
    for path in outputs:
        require(not path.exists(), f"Stage {STAGE} output already exists: {rel(path)}")

    evidence, prior = verify_inputs()
    source_rows = load_original_train_sites()
    split_rows, split_payload = repair_split(source_rows)
    grid_payload = candidate_grid_csv()
    split_commitment = hashlib.sha256(split_payload).hexdigest()

    scope = {
        "contract_version": "CIRCUITSAGE-HMAC-V2.1-IMPROVEMENT-SCOPE-v1",
        "stage": STAGE, "status": "FROZEN",
        "branch": "V2.1 OBSERVABILITY AND LOCALIZATION REPAIR",
        "purpose": "improve all-injected detection and exact-site localization without altering frozen V2.1",
        "fault_scope": "OpenTitan HMAC; single persistent SA0/SA1",
        "timebox": "ONE WEEK TARGET; OUTCOME MAY BE NEGATIVE",
        "frozen_comparators": ["CircuitSage-HMAC V1", "V2 Core", "V2.1"],
        "current_metrics": {
            "all_injected_detection_recall": prior["overall_detection_recall"],
            "all_injected_exact_site_rate": prior["overall_exact_site_rate"],
            "mean_observable_candidate_sites": 141.6268,
            "maximum_observable_candidate_sites": 1668,
        },
        "unknown_fault_identity_in_query": "PROHIBITED",
        "consumed_dev_site_test": "REOPENING PROHIBITED",
        "validation_access": "PROHIBITED",
        "holdout_access": "PROHIBITED",
        "allowed_source_sites": "ORIGINAL DEV_TRAIN ONLY",
        "new_development_vectors": "DETERMINISTICALLY GENERATED; NO HELD-OUT PAYLOAD SOURCE",
    }
    partition_contract = {
        "contract_version": "CIRCUITSAGE-HMAC-V2.1-REPAIR-PARTITION-v1",
        "stage": STAGE, "status": "FROZEN",
        "source_partition": "ORIGINAL DEV_TRAIN PHYSICAL SITES ONLY",
        "source_sites": ORIGINAL_TRAIN_SITES,
        "repair_train_sites": REPAIR_TRAIN_SITES,
        "repair_calibration_sites": REPAIR_CALIBRATION_SITES,
        "repair_site_test_sites": REPAIR_SITE_TEST_SITES,
        "fault_instances_per_site": 2,
        "split_method": "SHA256 domain-separated ordering by immutable site_id",
        "split_domain_hex": SPLIT_DOMAIN.hex(),
        "split_commitment_sha256": split_commitment,
        "repair_site_test_policy": "LOCK UNTIL A SEPARATE AUTHORIZATION AFTER MODEL SELECTION",
        "original_dev_site_test": "EXCLUDED AND PERMANENTLY CONSUMED",
        "validation_and_holdout": "EXCLUDED",
    }
    vector_policy = {
        "policy_version": "CIRCUITSAGE-HMAC-V2.1-ADAPTIVE-VECTOR-POLICY-v1",
        "stage": STAGE, "status": "FROZEN",
        "candidate_vector_domain": VECTOR_DOMAIN,
        "candidate_vector_seed": VECTOR_SEED,
        "candidate_vectors": CANDIDATE_VECTOR_POOL,
        "key_bytes": 32, "message_bytes": 32,
        "derivation": "SHA256(domain || seed_le64 || vector_index_le32 || field_label || counter_le32)",
        "held_out_vector_payloads_used": 0,
        "pilot_screening_sites": PILOT_SITES,
        "pilot_site_source": "REPAIR_TRAIN ONLY",
        "pilot_screening_fault_instances": PILOT_SITES * 2,
        "selection_objective_order": [
            "newly observable repair-train fault instances",
            "reduction in observational-equivalence set size",
            "balanced SA0/SA1 activation",
            "deterministic vector index",
        ],
        "selection_rounds": SELECTION_ROUNDS,
        "vectors_per_round": VECTORS_PER_ROUND,
        "maximum_selected_vectors": MAX_SELECTED_VECTORS,
        "adaptive_feedback_partition": "REPAIR_TRAIN ONLY",
        "repair_calibration_feedback": "PROHIBITED FOR VECTOR SELECTION",
        "repair_site_test_feedback": "PROHIBITED",
    }
    architecture = {
        "contract_version": "CIRCUITSAGE-HMAC-V2.1-REPAIR-ARCHITECTURE-v1",
        "stage": STAGE, "status": "FROZEN",
        "pipeline": [
            "golden-reference behavior comparator",
            "multi-vector response encoder",
            "graph cone-of-influence candidate filter",
            "behavior-signature retrieval",
            "ambiguity-aware graph reranker",
            "open-set and no-evidence output",
        ],
        "graph_source": record(GRAPH),
        "graph_features": 119,
        "candidate_models": 3,
        "trainable_candidates": 2,
        "maximum_trainable_parameters_per_candidate": 250000,
        "training_negatives": "FULL-CATALOG HARD NEGATIVES FROM REPAIR_TRAIN ONLY",
        "target_type": "VALID CANDIDATE-SITE SET; NEVER FORCED SINGLE SITE",
        "v1_role": "FROZEN READ-ONLY SUPPORT FEATURE / COMPARATOR",
        "model_training": "NOT YET AUTHORIZED",
    }
    acceptance = {
        "contract_version": "CIRCUITSAGE-HMAC-V2.1-REPAIR-ACCEPTANCE-v1",
        "stage": STAGE, "status": "FROZEN",
        "primary_selection_partition": "REPAIR_CALIBRATION",
        "locked_evaluation_partition": "REPAIR_SITE_TEST",
        "absolute_targets": {
            "all_injected_detection_recall_min": 0.70,
            "all_injected_exact_site_rate_min": 0.35,
            "observable_candidate_set_coverage_min": 0.95,
            "mean_observable_candidate_sites_max": 50.0,
            "maximum_observable_candidate_sites_max": 500,
            "fault_free_false_alarm_rate_max": 0.01,
            "ambiguous_false_unique_rate_max": 0.01,
        },
        "relative_targets": {
            "all_injected_detection_recall_delta_vs_v2_1_min": 0.10,
            "all_injected_exact_site_rate_delta_vs_v2_1_min": 0.10,
            "mean_candidate_set_relative_reduction_min": 0.50,
            "paired_site_bootstrap_delta_ci_lower_bound": 0.0,
        },
        "deterministic_replay": "EXACT",
        "target_freeze_policy": "NO RELAXATION AFTER ANY REPAIR_CALIBRATION OR REPAIR_SITE_TEST RESULT",
        "claim_limit": "HMAC SA0/SA1 repair experiment; independent-circuit generalization remains unestablished",
    }
    environment = {
        "environment_version": "CIRCUITSAGE-HMAC-V2.1-IMPROVEMENT-ENVIRONMENT-v1",
        "stage": STAGE, "status": "FROZEN",
        "python": sys.version.split()[0], "numpy": np.__version__,
        "platform": platform.platform(), "cpu_count": os.cpu_count(),
        "internet_required": False, "gpu_required": False,
        "candidate_execution": "SEQUENTIAL", "parallel_candidates": 1,
        "estimated_total_cpu_time": "8-30 HOURS ACROSS THE REPAIR BRANCH",
    }

    frozen_write(SCOPE, canonical_json(scope))
    frozen_write(PARTITION_CONTRACT, canonical_json(partition_contract))
    frozen_write(VECTOR_POLICY, canonical_json(vector_policy))
    frozen_write(ARCHITECTURE, canonical_json(architecture))
    frozen_write(ACCEPTANCE, canonical_json(acceptance))
    frozen_write(CANDIDATE_GRID, grid_payload)
    frozen_write(REPAIR_SPLIT, split_payload)
    frozen_write(ENVIRONMENT, canonical_json(environment))
    primary_outputs = (SCOPE, PARTITION_CONTRACT, VECTOR_POLICY, ARCHITECTURE,
                       ACCEPTANCE, CANDIDATE_GRID, REPAIR_SPLIT, ENVIRONMENT)
    manifest = {
        "manifest_version": "CIRCUITSAGE-HMAC-V2.1-IMPROVEMENT-CONTRACT-MANIFEST-v1",
        "stage": STAGE, "status": "PASS", "input_evidence": evidence,
        "outputs": {rel(path): record(path) for path in primary_outputs},
        "repair_split_rows": len(split_rows), "model_objects_deserialized": 0,
        "model_training_calls": 0, "model_inference_calls": 0,
        "response_payloads_generated": 0, "dev_site_test_reopened": False,
        "validation_access_count": 0, "holdout_access_count": 0,
    }
    frozen_write(MANIFEST, canonical_json(manifest))
    audit = {
        "audit_version": "CIRCUITSAGE-HMAC-V2.1-IMPROVEMENT-CONTRACT-FREEZE-v1",
        "stage": STAGE, "status": "PASS", "scope_status": "FROZEN",
        "partition_contract_status": "FROZEN", "vector_policy_status": "FROZEN",
        "architecture_status": "FROZEN", "acceptance_status": "FROZEN",
        "repair_branch": "NEW / ISOLATED FROM FROZEN V2.1",
        "source_sites": "ORIGINAL DEV_TRAIN ONLY",
        "repair_train_calibration_test_sites": [REPAIR_TRAIN_SITES, REPAIR_CALIBRATION_SITES, REPAIR_SITE_TEST_SITES],
        "candidate_vector_pool": CANDIDATE_VECTOR_POOL,
        "maximum_selected_vectors": MAX_SELECTED_VECTORS,
        "adaptive_pilot_generation": "AUTHORIZED / NOT STARTED",
        "full_campaign_generation": "NOT YET AUTHORIZED",
        "model_training": "NOT YET AUTHORIZED",
        "repair_site_test": "LOCKED / NOT AUTHORIZED",
        "original_dev_site_test": "CONSUMED / NOT REOPENED",
        "validation_access_count": 0, "holdout_access_count": 0,
        "v1_modified": False, "v2_core_modified": False, "v2_1_modified": False,
        "model_objects_deserialized": 0, "model_training_calls": 0,
        "model_inference_calls": 0, "response_payloads_generated": 0,
        "manifest": record(MANIFEST),
        "next_gate": "STAGE 12B-2B — ADAPTIVE TEST-VECTOR CANDIDATE POOL AND PILOT-SCREENING AUTHORIZATION FREEZE",
    }
    frozen_write(AUDIT, canonical_json(audit))

    for path in (SCOPE, PARTITION_CONTRACT, VECTOR_POLICY, ARCHITECTURE,
                 ACCEPTANCE, ENVIRONMENT, MANIFEST, AUDIT):
        require(canonical_json(load_json(path)) == path.read_bytes(), f"JSON replay: {path.name}")
    require(grid_payload == CANDIDATE_GRID.read_bytes(), "candidate-grid replay")
    require(split_payload == REPAIR_SPLIT.read_bytes(), "repair-split replay")

    print("\nSTAGE 12B-2A — V2.1 OBSERVABILITY AND LOCALIZATION IMPROVEMENT CONTRACT FREEZE")
    print(f"{'Status':<48}: PASS")
    print(f"{'Scope / partition / acceptance':<48}: FROZEN / FROZEN / FROZEN")
    print(f"{'Repair branch':<48}: NEW / ISOLATED FROM FROZEN V2.1")
    print(f"{'Source physical sites':<48}: ORIGINAL DEV_TRAIN ONLY — {ORIGINAL_TRAIN_SITES}")
    print(f"{'Repair TRAIN / CALIBRATION / SITE_TEST':<48}: {REPAIR_TRAIN_SITES} / {REPAIR_CALIBRATION_SITES} / {REPAIR_SITE_TEST_SITES}")
    print(f"{'Candidate / maximum selected vectors':<48}: {CANDIDATE_VECTOR_POOL} / {MAX_SELECTED_VECTORS}")
    print(f"{'Adaptive selection rounds':<48}: {SELECTION_ROUNDS} x {VECTORS_PER_ROUND}")
    print(f"{'Candidate models / trainable':<48}: 3 / 2")
    print(f"{'Detection target':<48}: >= 0.70 ALL INJECTED")
    print(f"{'Exact-site target':<48}: >= 0.35 ALL INJECTED")
    print(f"{'Mean / maximum candidate targets':<48}: <= 50 / <= 500")
    print(f"{'Pilot vector generation':<48}: AUTHORIZED / NOT STARTED")
    print(f"{'Full campaign / model training':<48}: NOT YET AUTHORIZED / NOT YET AUTHORIZED")
    print(f"{'Original DEV_SITE_TEST':<48}: CONSUMED / NOT REOPENED")
    print(f"{'VALIDATION / HOLDOUT access':<48}: 0 / 0")
    print(f"{'V1 / V2 Core / V2.1 modified':<48}: NO / NO / NO")
    print(f"{'Scope contract':<48}: {SCOPE}")
    print(f"{'Scope SHA':<48}: {sha256(SCOPE)}")
    print(f"{'Partition contract':<48}: {PARTITION_CONTRACT}")
    print(f"{'Partition contract SHA':<48}: {sha256(PARTITION_CONTRACT)}")
    print(f"{'Vector policy':<48}: {VECTOR_POLICY}")
    print(f"{'Vector policy SHA':<48}: {sha256(VECTOR_POLICY)}")
    print(f"{'Acceptance contract':<48}: {ACCEPTANCE}")
    print(f"{'Acceptance SHA':<48}: {sha256(ACCEPTANCE)}")
    print(f"{'Repair split':<48}: {REPAIR_SPLIT}")
    print(f"{'Repair split SHA':<48}: {sha256(REPAIR_SPLIT)}")
    print(f"{'Manifest':<48}: {MANIFEST}")
    print(f"{'Manifest SHA':<48}: {sha256(MANIFEST)}")
    print(f"{'Audit':<48}: {AUDIT}")
    print(f"{'Audit SHA':<48}: {sha256(AUDIT)}")
    print(f"{'Next gate':<48}: STAGE 12B-2B — ADAPTIVE TEST-VECTOR CANDIDATE POOL AND PILOT-SCREENING AUTHORIZATION FREEZE")


if __name__ == "__main__":
    main()
