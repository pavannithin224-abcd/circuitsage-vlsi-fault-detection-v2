#!/usr/bin/env python3
"""Stage 12B-1C: V2.1 locator training authorization and contract freeze.

Validates the frozen behavior-signature index and ambiguity-aware dataset,
freezes the candidate execution plan, and authorizes the next stage to fit on
DEV_TRAIN only.  This stage performs no model training or inference and does
not open DEV_SITE_TEST, VALIDATION, or HOLDOUT.
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


STAGE = "12B-1C"
ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "config/v2_1"
RESULT = ROOT / "results/circuitsage_hmac_v2_12b1"

CONTRACT_SOURCE = ROOT / "stage_12b1a_v2_1_locator_redesign_contract.py"
ARCHITECTURE = CONFIG / "circuitsage_hmac_v2_1_locator_architecture_12b1a.json"
BASE_TRAINING_CONTRACT = CONFIG / "circuitsage_hmac_v2_1_locator_training_contract_12b1a.json"
ACCEPTANCE = CONFIG / "circuitsage_hmac_v2_1_locator_acceptance_contract_12b1a.json"
BASE_GRID = CONFIG / "circuitsage_hmac_v2_1_locator_candidate_grid_12b1a.csv"
CONTRACT_MANIFEST = RESULT / "circuitsage_hmac_v2_1_contract_manifest_12b1a.json"
CONTRACT_AUDIT = RESULT / "circuitsage_hmac_v2_1_locator_redesign_contract_freeze_12b1a.json"

DATASET_SOURCE = ROOT / "stage_12b1b_behavior_signature_dataset.py"
DATASET_DIR = RESULT / "behavior_signature_dataset_12b1b"
SIGNATURE_INDEX = DATASET_DIR / "circuitsage_hmac_v2_1_behavior_signature_index_12b1b.npz"
AMBIGUITY_DATASET = DATASET_DIR / "circuitsage_hmac_v2_1_ambiguity_aware_dataset_12b1b.npz"
DATASET_SCHEMA = DATASET_DIR / "circuitsage_hmac_v2_1_behavior_signature_schema_12b1b.json"
DATASET_STATISTICS = DATASET_DIR / "circuitsage_hmac_v2_1_behavior_signature_statistics_12b1b.json"
DATASET_MANIFEST = RESULT / "circuitsage_hmac_v2_1_behavior_signature_manifest_12b1b.json"
DATASET_AUDIT = RESULT / "circuitsage_hmac_v2_1_behavior_signature_dataset_freeze_12b1b.json"

EXECUTION_CONTRACT = CONFIG / "circuitsage_hmac_v2_1_training_execution_contract_12b1c.json"
AUTHORIZATION = CONFIG / "circuitsage_hmac_v2_1_training_authorization_12b1c.json"
EXECUTION_PLAN = RESULT / "circuitsage_hmac_v2_1_candidate_execution_plan_12b1c.csv"
PREFLIGHT = RESULT / "circuitsage_hmac_v2_1_training_preflight_12b1c.json"
MANIFEST = RESULT / "circuitsage_hmac_v2_1_training_authorization_manifest_12b1c.json"
AUDIT = RESULT / "circuitsage_hmac_v2_1_training_authorization_freeze_12b1c.json"

PINNED = {
    CONTRACT_SOURCE: "8a5a211c5adc073bf3d67560d50bd4457f181a0c19ca7e2f136c9f905bb6f979",
    ARCHITECTURE: "370fa36bfa952a1fe54d1b7b7f4e27950692387849d94aa73b733de4c45b02fe",
    BASE_TRAINING_CONTRACT: "36d6da41f359dee47d146866601655ed46b39d941ef79dcbd48cb689d3260e67",
    ACCEPTANCE: "899b2ff099da20b4631308962c3052e16c7cf875ae296f5b2f6da825d6c6c12a",
    BASE_GRID: "c159aa527d6720093fa3578d0d4e0042d1b0c2138535ac01e567efdddd600653",
    CONTRACT_MANIFEST: "eae8cf0d02155c8fbf3e736307ffca79f13d2268442c10a49a945477d7e62577",
    CONTRACT_AUDIT: "95ccfb93fcc11c5b87b21d0a7fd869a0562c42175ec6aa60310ca20ee272fa01",
    DATASET_SOURCE: "047a362bb598fcd1687393b62cb729c5f2604ec4eac2b3d3c6f41c86910d5db6",
}

SITES = 22839
FAULTS = 45678
VECTORS = 64
MIN_FREE_GIB = 2.0
SEED = 20260915

PLAN = [
    {
        "execution_order": 1,
        "candidate_id": "V21_EXACT_SIGNATURE_SET",
        "fit_partition": "NONE",
        "selection_partition": "DEV_CALIBRATION",
        "training_method": "NONPARAMETRIC EXACT HASH LOOKUP",
        "epochs": 0,
        "shortlist": FAULTS,
        "maximum_trainable_parameters": 0,
    },
    {
        "execution_order": 2,
        "candidate_id": "V21_WEIGHTED_SIGNATURE_K2048",
        "fit_partition": "DEV_TRAIN ONLY",
        "selection_partition": "DEV_CALIBRATION",
        "training_method": "DETERMINISTIC GROUP-WEIGHT SEARCH WITH LISTWISE SET LOSS",
        "epochs": 12,
        "shortlist": 2048,
        "maximum_trainable_parameters": 32,
    },
    {
        "execution_order": 3,
        "candidate_id": "V21_SIGNATURE_GRAPH_LISTWISE_K2048",
        "fit_partition": "DEV_TRAIN ONLY",
        "selection_partition": "DEV_CALIBRATION",
        "training_method": "DETERMINISTIC LISTWISE LINEAR RERANKER WITH HARD NEGATIVES",
        "epochs": 20,
        "shortlist": 2048,
        "maximum_trainable_parameters": 4096,
    },
]


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


def resolve_record_path(item: dict[str, Any]) -> Path:
    value = item.get("path")
    require(isinstance(value, str) and value, "artifact path record")
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def verify_record(item: dict[str, Any], label: str) -> Path:
    path = resolve_record_path(item)
    require(path.is_file(), f"missing {label}: {rel(path)}")
    require(sha256(path) == item.get("sha256"), f"{label} changed: {path.name}")
    if "bytes" in item:
        require(path.stat().st_size == int(item["bytes"]), f"{label} size changed: {path.name}")
    return path


def find_record(entries: dict[str, Any], basename: str) -> dict[str, Any]:
    matches = []
    for item in entries.values():
        if isinstance(item, dict) and isinstance(item.get("path"), str):
            if Path(item["path"]).name == basename:
                matches.append(item)
    require(len(matches) == 1, f"artifact resolution: {basename}")
    return matches[0]


def plan_csv() -> bytes:
    fields = [
        "execution_order", "candidate_id", "fit_partition", "selection_partition",
        "training_method", "epochs", "shortlist", "maximum_trainable_parameters",
    ]
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(PLAN)
    return output.getvalue().encode()


def verify_lineage() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    print("STAGE 12B-1C — V2.1 LOCATOR TRAINING AUTHORIZATION")
    print("FROZEN INPUT VERIFICATION")
    evidence: dict[str, Any] = {}
    for path, expected in PINNED.items():
        require(path.is_file(), f"missing frozen input: {rel(path)}")
        require(sha256(path) == expected, f"frozen input SHA: {rel(path)}")
        evidence[rel(path)] = record(path)
        print(f"  {path.name:<82}: OK", flush=True)

    for path in (SIGNATURE_INDEX, AMBIGUITY_DATASET, DATASET_SCHEMA, DATASET_STATISTICS, DATASET_MANIFEST, DATASET_AUDIT):
        require(path.is_file(), f"missing Stage 12B-1B artifact: {rel(path)}")
        evidence[rel(path)] = record(path)
        print(f"  {path.name:<82}: OK", flush=True)

    dataset_manifest = load_json(DATASET_MANIFEST)
    dataset_audit = load_json(DATASET_AUDIT)
    dataset_schema = load_json(DATASET_SCHEMA)
    dataset_statistics = load_json(DATASET_STATISTICS)
    base_training = load_json(BASE_TRAINING_CONTRACT)
    acceptance = load_json(ACCEPTANCE)

    require(dataset_manifest.get("status") == "PASS", "dataset manifest status")
    require(dataset_manifest.get("dataset_status") == "FROZEN", "dataset freeze")
    require(dataset_manifest.get("dev_site_test_query_rows") == 0, "dataset site-test query rows")
    require(dataset_manifest.get("validation_access_count") == 0, "dataset VALIDATION access")
    require(dataset_manifest.get("holdout_access_count") == 0, "dataset HOLDOUT access")
    require(dataset_audit.get("status") == "PASS", "dataset audit status")
    require(dataset_audit.get("signature_index_status") == "FROZEN", "signature index freeze")
    require(dataset_audit.get("ambiguity_aware_dataset_status") == "FROZEN", "ambiguity dataset freeze")
    require(dataset_audit.get("deterministic_replay") == "PASS / EXACT", "dataset replay")
    require(dataset_audit.get("signature_hash_collisions") == 0, "signature collisions")
    require(dataset_audit.get("dev_site_test") == "LOCKED / ZERO QUERY ROWS / NOT OPENED", "dataset site-test state")
    require(dataset_audit.get("validation_access_count") == 0, "dataset audit VALIDATION access")
    require(dataset_audit.get("holdout_access_count") == 0, "dataset audit HOLDOUT access")
    require(dataset_audit.get("training_performed") is False, "dataset training state")
    require(dataset_schema.get("status") == "FROZEN", "dataset schema status")
    require(dataset_schema.get("identity_boundary", {}).get("query identity features") == 0, "query identity boundary")
    require(dataset_statistics.get("signature_hash_collisions") == 0, "statistics signature collisions")
    require(dataset_statistics.get("dev_site_test_query_profiles") == 0, "statistics site-test queries")
    require(int(dataset_statistics.get("train_query_profiles", 0)) > 0, "empty train queries")
    require(int(dataset_statistics.get("calibration_query_profiles", 0)) > 0, "empty calibration queries")
    require(base_training.get("status") == "FROZEN", "base training contract")
    require(acceptance.get("status") == "FROZEN", "acceptance contract")

    verify_record(dataset_audit.get("signature_index", {}), "audit signature index")
    verify_record(dataset_audit.get("ambiguity_dataset", {}), "audit ambiguity dataset")
    verify_record(dataset_audit.get("schema", {}), "audit schema")
    verify_record(dataset_audit.get("statistics", {}), "audit statistics")
    verify_record(dataset_audit.get("manifest", {}), "audit manifest")
    for item in dataset_manifest.get("outputs", {}).values():
        if isinstance(item, dict):
            verify_record(item, "dataset manifest output")
    contract_lineage = find_record(dataset_manifest.get("input_evidence", {}), CONTRACT_AUDIT.name)
    require(contract_lineage.get("sha256") == PINNED[CONTRACT_AUDIT], "12B-1A lineage")
    for path in (DATASET_SCHEMA, DATASET_STATISTICS, DATASET_MANIFEST, DATASET_AUDIT):
        require(canonical_json(load_json(path)) == path.read_bytes(), f"canonical replay: {path.name}")
    print("  Stage 12B-1B semantic freeze, artifact chain and 12B-1A lineage               : PASS")
    return evidence, dataset_statistics, acceptance


def validate_arrays(statistics: dict[str, Any]) -> dict[str, Any]:
    print("\nSTRUCTURAL DATASET PREFLIGHT", flush=True)
    with np.load(SIGNATURE_INDEX, allow_pickle=False) as archive:
        required = {
            "fault_instance_index_by_profile", "profile_candidate_fault_count",
            "profile_candidate_site_count", "profile_deviation", "profile_digest_hamming",
            "profile_fault_offset", "profile_index", "profile_latency_delta",
            "profile_observability_class", "profile_observable", "profile_pure_partition_code",
            "profile_signature_sha256", "profile_signature_sort_order", "profile_site_index",
            "profile_site_offset", "profile_timeout", "vector_commitment_sha256", "vector_id",
        }
        require(required == set(archive.files), "signature-index members")
        index = {name: np.asarray(archive[name]).copy() for name in archive.files}
    profiles = len(index["profile_index"])
    require(profiles == int(statistics["distinct_response_profiles"]), "profile count")
    require(np.array_equal(index["profile_index"], np.arange(profiles)), "profile ordering")
    require(index["profile_signature_sha256"].shape == (profiles, 32), "signature shape")
    require(index["profile_signature_sort_order"].shape == (profiles,), "signature order shape")
    require(np.array_equal(np.sort(index["profile_signature_sort_order"]), np.arange(profiles)), "signature-order permutation")
    keys = index["profile_signature_sha256"].view(np.dtype((np.void, 32))).reshape(-1)
    order = index["profile_signature_sort_order"].astype(np.int64)
    require(len(np.unique(keys)) == profiles, "signature uniqueness")
    require(all(bytes(keys[order[i]]) < bytes(keys[order[i + 1]]) for i in range(profiles - 1)), "signature sort order")
    require(index["profile_fault_offset"].shape == (profiles + 1,), "fault-offset shape")
    require(int(index["profile_fault_offset"][0]) == 0 and int(index["profile_fault_offset"][-1]) == FAULTS, "fault-offset closure")
    require(np.all(np.diff(index["profile_fault_offset"].astype(np.int64)) > 0), "empty profile")
    require(np.array_equal(np.sort(index["fault_instance_index_by_profile"]), np.arange(FAULTS)), "fault catalog permutation")
    require(index["profile_site_offset"].shape == (profiles + 1,), "site-offset shape")
    require(int(index["profile_site_offset"][0]) == 0, "site-offset start")
    require(int(index["profile_site_offset"][-1]) == len(index["profile_site_index"]), "site-offset closure")
    require(np.all((index["profile_site_index"] >= 1) & (index["profile_site_index"] <= SITES)), "site range")
    require(index["profile_timeout"].shape == (profiles, VECTORS), "timeout shape")
    require(index["profile_latency_delta"].shape == (profiles, VECTORS), "latency shape")
    require(index["profile_digest_hamming"].shape == (profiles, VECTORS), "hamming shape")
    require(index["profile_deviation"].shape == (profiles, VECTORS), "deviation shape")

    with np.load(AMBIGUITY_DATASET, allow_pickle=False) as archive:
        required = {
            "query_candidate_fault_index", "query_candidate_fault_offset",
            "query_candidate_site_index", "query_candidate_site_offset",
            "query_candidate_site_count", "query_observability_class",
            "query_partition_code", "query_profile_index", "query_signature_sha256",
        }
        require(required == set(archive.files), "ambiguity-dataset members")
        queries = {name: np.asarray(archive[name]).copy() for name in archive.files}
    query_count = len(queries["query_profile_index"])
    require(query_count == int(statistics["query_profiles_total"]), "query count")
    require(np.all((queries["query_partition_code"] == 0) | (queries["query_partition_code"] == 1)), "query partition boundary")
    require(not np.any(queries["query_partition_code"] == 2), "DEV_SITE_TEST query present")
    require(len(np.unique(queries["query_profile_index"])) == query_count, "duplicate query profile")
    require(np.array_equal(queries["query_signature_sha256"], index["profile_signature_sha256"][queries["query_profile_index"]]), "query signature binding")
    require(len(queries["query_candidate_fault_offset"]) == query_count + 1, "query fault offsets")
    require(len(queries["query_candidate_site_offset"]) == query_count + 1, "query site offsets")
    require(int(queries["query_candidate_fault_offset"][0]) == 0, "query fault offset start")
    require(int(queries["query_candidate_fault_offset"][-1]) == len(queries["query_candidate_fault_index"]), "query fault offset closure")
    require(int(queries["query_candidate_site_offset"][0]) == 0, "query site offset start")
    require(int(queries["query_candidate_site_offset"][-1]) == len(queries["query_candidate_site_index"]), "query site offset closure")
    require(np.array_equal(np.diff(queries["query_candidate_site_offset"].astype(np.int64)), queries["query_candidate_site_count"]), "query site counts")
    require(np.all((queries["query_candidate_site_index"] >= 1) & (queries["query_candidate_site_index"] <= SITES)), "query site range")
    require(np.all((queries["query_candidate_fault_index"] >= 0) & (queries["query_candidate_fault_index"] < FAULTS)), "query fault range")

    for row, profile_u32 in enumerate(queries["query_profile_index"]):
        profile = int(profile_u32)
        qs0, qs1 = map(int, queries["query_candidate_site_offset"][row:row + 2])
        ps0, ps1 = map(int, index["profile_site_offset"][profile:profile + 2])
        require(np.array_equal(queries["query_candidate_site_index"][qs0:qs1], index["profile_site_index"][ps0:ps1]), "query-to-catalog site set")
        qf0, qf1 = map(int, queries["query_candidate_fault_offset"][row:row + 2])
        pf0, pf1 = map(int, index["profile_fault_offset"][profile:profile + 2])
        require(np.array_equal(queries["query_candidate_fault_index"][qf0:qf1], index["fault_instance_index_by_profile"][pf0:pf1]), "query-to-catalog fault set")
    print(f"  Signature profiles              : {profiles:,}")
    print(f"  DEV_TRAIN query profiles         : {int(np.sum(queries['query_partition_code'] == 0)):,}")
    print(f"  DEV_CALIBRATION query profiles   : {int(np.sum(queries['query_partition_code'] == 1)):,}")
    print("  DEV_SITE_TEST query profiles     : 0")
    print("  Exact signature/set bindings     : PASS")
    return {
        "distinct_profiles": profiles,
        "query_profiles": query_count,
        "train_query_profiles": int(np.sum(queries["query_partition_code"] == 0)),
        "calibration_query_profiles": int(np.sum(queries["query_partition_code"] == 1)),
        "unique_query_profiles": int(np.sum(queries["query_observability_class"] == 1)),
        "ambiguous_query_profiles": int(np.sum(queries["query_observability_class"] == 2)),
        "signature_collisions": 0,
        "dev_site_test_query_profiles": 0,
    }


def self_test() -> None:
    require(len(PLAN) == 3, "candidate count canary")
    require(sum(int(row["epochs"]) > 0 for row in PLAN) == 2, "trainable count canary")
    require(PLAN[0]["maximum_trainable_parameters"] == 0, "reference canary")
    require(plan_csv() == plan_csv(), "execution-plan replay canary")
    require(canonical_json({"b": 2, "a": 1}) == canonical_json({"a": 1, "b": 2}), "JSON replay canary")
    print("Stage 12B-1C self-test: PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return

    require(ROOT.name == "vlsi_fault_detection_v2", f"run from project root, not {ROOT}")
    for path in (EXECUTION_CONTRACT, AUTHORIZATION, EXECUTION_PLAN, PREFLIGHT, MANIFEST, AUDIT):
        require(not path.exists(), f"Stage {STAGE} output already exists: {rel(path)}")

    evidence, statistics, acceptance = verify_lineage()
    structural = validate_arrays(statistics)
    free_gib = shutil.disk_usage(ROOT).free / (1024 ** 3)
    require(free_gib >= MIN_FREE_GIB, f"insufficient free disk: {free_gib:.2f} GiB")

    execution_contract = {
        "contract_version": "CIRCUITSAGE-HMAC-V2.1-TRAINING-EXECUTION-v1",
        "stage": STAGE,
        "status": "FROZEN",
        "model_family": "AMBIGUITY-AWARE BEHAVIOR-SIGNATURE RETRIEVAL AND GRAPH RERANKING",
        "candidate_execution_plan": PLAN,
        "data_use": {
            "DEV_TRAIN": "only partition allowed for fitted parameters, hard negatives and optimization",
            "DEV_CALIBRATION": "metrics and final candidate selection only; no gradients or weight updates",
            "DEV_SITE_TEST": "LOCKED / NOT AUTHORIZED / ZERO QUERY ROWS IN TRAINING DATASET",
            "VALIDATION": "PROHIBITED",
            "HOLDOUT": "PROHIBITED",
        },
        "retrieval_features": {
            "timeout_mismatch": "per-vector binary mismatch",
            "latency_distance": "clipped normalized per-vector absolute difference",
            "digest_distance": "per-vector bit count of candidate/query digest-XOR difference",
            "deviation_mask_distance": "per-vector binary mismatch",
            "exact_signature_match": "SHA-256 equality under frozen vector commitment",
        },
        "reranking_features": {
            "response_distance_components": 4,
            "exact_match_indicator": 1,
            "candidate_set_size_features": 2,
            "frozen_v1_support_agreement_features": 4,
            "frozen_graph_similarity_features": 4,
            "fault_type_sibling_features": 2,
            "maximum_total_features": 32,
        },
        "losses": {
            "retrieval": "multi-positive listwise set loss",
            "ambiguity": "all sites in the exact observational-equivalence set are positives",
            "hard_negative_margin": 0.10,
            "fault_type": "conditional binary loss only after site success",
        },
        "hard_negative_policy": {
            "source_partition": "DEV_TRAIN ONLY",
            "per_query_maximum": 128,
            "sources": [
                "nearest incorrect response signatures",
                "highest-scoring incorrect shortlist candidates",
                "graph-neighbor candidates",
                "SA0/SA1 sibling",
            ],
        },
        "selection": {
            "partition": "DEV_CALIBRATION ONLY",
            "order": acceptance["primary_selection_order"],
            "acceptance": acceptance["calibration_advancement_targets"],
            "ties": "candidate ID ascending after exact metric equality",
        },
        "reproducibility": {
            "seed": SEED,
            "candidate_execution": "SEQUENTIAL",
            "parallel_candidates": 1,
            "thread_limit": 1,
            "checkpoint_after_candidate": True,
            "checkpoint_resume": "REQUIRED",
            "training_replay": "EXACT",
            "metric_replay": "EXACT",
        },
        "resource_limits": {
            "estimated_cpu_time": "1-6 HOURS",
            "minimum_free_disk_gib": MIN_FREE_GIB,
            "gpu_required": False,
            "internet_required": False,
        },
        "failure_routes": {
            "calibration_target_not_met": "freeze negative result and keep DEV_SITE_TEST locked",
            "calibration_target_met": "freeze model and request separate locked DEV_SITE_TEST authorization",
            "replay_failure": "STOP; no model selection",
        },
    }

    authorization = {
        "authorization_version": "CIRCUITSAGE-HMAC-V2.1-TRAINING-AUTHORIZATION-v1",
        "stage": STAGE,
        "status": "FROZEN",
        "locator_training": "AUTHORIZED FOR DEV_TRAIN ONLY",
        "nonparametric_reference_evaluation": "AUTHORIZED ON DEV_CALIBRATION",
        "candidate_selection": "AUTHORIZED ON DEV_CALIBRATION ONLY",
        "dev_site_test_evaluation": "NOT AUTHORIZED",
        "validation_access": "NOT AUTHORIZED",
        "holdout_access": "NOT AUTHORIZED",
        "v1_model": "FROZEN READ-ONLY SUPPORT; RETRAINING AND THRESHOLD CHANGES PROHIBITED",
        "v2_core": "FROZEN READ-ONLY PREDECESSOR; MODIFICATION PROHIBITED",
        "unknown_fault_identity_in_query": "PROHIBITED",
        "maximum_trainable_candidates": 2,
        "authorization_consumed": False,
        "next_authorized_stage": "12B-1D",
    }

    preflight = {
        "preflight_version": "CIRCUITSAGE-HMAC-V2.1-TRAINING-PREFLIGHT-v1",
        "stage": STAGE,
        "status": "PASS",
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "available_disk_gib": round(free_gib, 6),
        "minimum_disk_gib": MIN_FREE_GIB,
        "dataset_structure": structural,
        "candidate_count": len(PLAN),
        "trainable_candidates": 2,
        "sequential_execution": True,
        "checkpoint_resume_required": True,
        "deterministic_replay_required": True,
        "models_deserialized": 0,
        "training_calls": 0,
        "inference_calls": 0,
        "dev_site_test_opened": False,
        "validation_access_count": 0,
        "holdout_access_count": 0,
    }

    frozen_write(EXECUTION_CONTRACT, canonical_json(execution_contract))
    frozen_write(AUTHORIZATION, canonical_json(authorization))
    frozen_write(EXECUTION_PLAN, plan_csv())
    frozen_write(PREFLIGHT, canonical_json(preflight))
    manifest = {
        "manifest_version": "CIRCUITSAGE-HMAC-V2.1-TRAINING-AUTHORIZATION-MANIFEST-v1",
        "stage": STAGE,
        "status": "PASS",
        "input_evidence": evidence,
        "outputs": {
            rel(path): record(path)
            for path in (EXECUTION_CONTRACT, AUTHORIZATION, EXECUTION_PLAN, PREFLIGHT)
        },
        "models_deserialized": 0,
        "training_calls": 0,
        "inference_calls": 0,
        "dev_site_test_opened": False,
        "validation_access_count": 0,
        "holdout_access_count": 0,
    }
    frozen_write(MANIFEST, canonical_json(manifest))
    audit = {
        "audit_version": "CIRCUITSAGE-HMAC-V2.1-TRAINING-AUTHORIZATION-FREEZE-v1",
        "stage": STAGE,
        "status": "PASS",
        "execution_contract_status": "FROZEN",
        "authorization_status": "FROZEN",
        "dataset_preflight": "PASS",
        "candidate_execution": "SEQUENTIAL",
        "candidates": len(PLAN),
        "trainable_candidates": 2,
        "fit_partition": "DEV_TRAIN ONLY",
        "selection_partition": "DEV_CALIBRATION ONLY",
        "locator_training": "AUTHORIZED FOR DEV_TRAIN ONLY / NOT STARTED",
        "dev_site_test": "LOCKED / NOT OPENED / NOT AUTHORIZED",
        "validation_access_count": 0,
        "holdout_access_count": 0,
        "v1_modified": False,
        "v2_core_modified": False,
        "training_calls": 0,
        "inference_calls": 0,
        "execution_contract": record(EXECUTION_CONTRACT),
        "authorization": record(AUTHORIZATION),
        "execution_plan": record(EXECUTION_PLAN),
        "preflight": record(PREFLIGHT),
        "manifest": record(MANIFEST),
        "next_gate": "STAGE 12B-1D — V2.1 LOCATOR TRAINING AND CALIBRATION EXECUTION",
    }
    frozen_write(AUDIT, canonical_json(audit))

    for path in (EXECUTION_CONTRACT, AUTHORIZATION, PREFLIGHT, MANIFEST, AUDIT):
        require(canonical_json(load_json(path)) == path.read_bytes(), f"deterministic JSON replay: {path.name}")
    require(plan_csv() == EXECUTION_PLAN.read_bytes(), "deterministic execution-plan replay")

    print("\nSTAGE 12B-1C — V2.1 LOCATOR TRAINING AUTHORIZATION AND EXECUTION-CONTRACT FREEZE")
    print(f"{'Status':<43}: PASS")
    print(f"{'Execution contract status':<43}: FROZEN")
    print(f"{'Authorization status':<43}: FROZEN")
    print(f"{'Dataset structural preflight':<43}: PASS")
    print(f"{'Candidate models / trainable':<43}: {len(PLAN)} / 2")
    print(f"{'TRAIN / CALIBRATION query profiles':<43}: {structural['train_query_profiles']} / {structural['calibration_query_profiles']}")
    print(f"{'Fit partition':<43}: DEV_TRAIN ONLY")
    print(f"{'Selection partition':<43}: DEV_CALIBRATION ONLY")
    print(f"{'Execution':<43}: SEQUENTIAL / 1 CANDIDATE AT A TIME")
    print(f"{'Checkpoint/resume':<43}: REQUIRED")
    print(f"{'Estimated CPU training time':<43}: 1-6 HOURS")
    print(f"{'Available / minimum disk':<43}: {free_gib:.2f} / {MIN_FREE_GIB:.2f} GiB")
    print(f"{'Locator training':<43}: AUTHORIZED / NOT STARTED")
    print(f"{'DEV_SITE_TEST':<43}: LOCKED / NOT OPENED / NOT AUTHORIZED")
    print(f"{'VALIDATION / HOLDOUT access':<43}: 0 / 0")
    print(f"{'V1 / V2 Core modified':<43}: NO / NO")
    print(f"{'Training / inference calls':<43}: 0 / 0")
    print(f"{'Execution contract':<43}: {EXECUTION_CONTRACT}")
    print(f"{'Execution contract SHA':<43}: {sha256(EXECUTION_CONTRACT)}")
    print(f"{'Authorization':<43}: {AUTHORIZATION}")
    print(f"{'Authorization SHA':<43}: {sha256(AUTHORIZATION)}")
    print(f"{'Execution plan':<43}: {EXECUTION_PLAN}")
    print(f"{'Execution plan SHA':<43}: {sha256(EXECUTION_PLAN)}")
    print(f"{'Manifest':<43}: {MANIFEST}")
    print(f"{'Manifest SHA':<43}: {sha256(MANIFEST)}")
    print(f"{'Audit':<43}: {AUDIT}")
    print(f"{'Audit SHA':<43}: {sha256(AUDIT)}")
    print(f"{'Next gate':<43}: STAGE 12B-1D — V2.1 LOCATOR TRAINING AND CALIBRATION EXECUTION")


if __name__ == "__main__":
    main()
