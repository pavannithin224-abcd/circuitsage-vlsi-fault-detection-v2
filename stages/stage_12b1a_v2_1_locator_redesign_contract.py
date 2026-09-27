#!/usr/bin/env python3
"""Stage 12B-1A: CircuitSage-HMAC V2.1 locator redesign contract freeze.

Verifies the frozen V2 Core disposition and creates an ambiguity-aware,
closed-catalog behavior-signature retrieval architecture and leakage-safe
training contract.  This stage performs no training, model deserialization,
inference, or access to DEV_SITE_TEST, VALIDATION, or HOLDOUT.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any


STAGE = "12B-1A"
ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "config" / "v2_1"
V2_RESULT = ROOT / "results" / "circuitsage_hmac_v2_12a1"
RESULT = ROOT / "results" / "circuitsage_hmac_v2_12b1"

R3_SOURCE = ROOT / "stage_12a1d_r3_locator_disposition_v2_core_freeze.py"
R3_POLICY = ROOT / "config/v2/circuitsage_hmac_v2_core_disposition_policy_12a1d_r3.json"
R3_CORE_LOCK = ROOT / "config/v2/circuitsage_hmac_v2_core_lock_12a1d_r3.json"
R3_COMPARISON = V2_RESULT / "v2_core_disposition_12a1d_r3/circuitsage_hmac_v2_locator_comparison_12a1d_r3.csv"
R3_REPORT = V2_RESULT / "v2_core_disposition_12a1d_r3/circuitsage_hmac_v2_core_disposition_report_12a1d_r3.md"
R3_MANIFEST = V2_RESULT / "circuitsage_hmac_v2_core_disposition_manifest_12a1d_r3.json"
R3_AUDIT = V2_RESULT / "circuitsage_hmac_v2_core_freeze_12a1d_r3.json"

R2_MANIFEST = V2_RESULT / "circuitsage_hmac_v2_repair_training_manifest_12a1d_r2.json"
R2_AUDIT = V2_RESULT / "circuitsage_hmac_v2_repair_training_calibration_freeze_12a1d_r2.json"

ARCHITECTURE = CONFIG / "circuitsage_hmac_v2_1_locator_architecture_12b1a.json"
TRAINING_CONTRACT = CONFIG / "circuitsage_hmac_v2_1_locator_training_contract_12b1a.json"
ACCEPTANCE = CONFIG / "circuitsage_hmac_v2_1_locator_acceptance_contract_12b1a.json"
CANDIDATE_GRID = CONFIG / "circuitsage_hmac_v2_1_locator_candidate_grid_12b1a.csv"
ENVIRONMENT = RESULT / "circuitsage_hmac_v2_1_environment_12b1a.json"
MANIFEST = RESULT / "circuitsage_hmac_v2_1_contract_manifest_12b1a.json"
AUDIT = RESULT / "circuitsage_hmac_v2_1_locator_redesign_contract_freeze_12b1a.json"

PINNED = {
    R3_SOURCE: "3b12c06c6860c065973bbef6cd89b5bd900163b1415362af309a6f50897c8b0f",
    R2_MANIFEST: "06518b9d7bfd73c6dad54501bec86f7e120156b5e34ea9073d3453224e61b7b4",
    R2_AUDIT: "bff78ea6919d5e4305e0b9d48178b342cb8fe7992c34f5f294a8b6c0f9375f3f",
}

SITES = 22839
FAULTS = 45678
TRAIN_VECTORS = 64
GRAPH_FEATURES = 119
V1_COMBINED_FEATURES = 646
BEST_V2_MRR = 0.00629651

CANDIDATES = [
    {
        "candidate_id": "V21_EXACT_SIGNATURE_SET",
        "candidate_type": "NONPARAMETRIC_REFERENCE",
        "retrieval": "EXACT CANONICAL RESPONSE-SIGNATURE HASH",
        "reranker": "NONE",
        "trainable_parameters": 0,
        "maximum_shortlist": FAULTS,
    },
    {
        "candidate_id": "V21_WEIGHTED_SIGNATURE_K2048",
        "candidate_type": "TRAINABLE RETRIEVAL",
        "retrieval": "GROUP-WEIGHTED RESPONSE DISTANCE",
        "reranker": "DETERMINISTIC DISTANCE ORDER",
        "trainable_parameters_max": 32,
        "maximum_shortlist": 2048,
    },
    {
        "candidate_id": "V21_SIGNATURE_GRAPH_LISTWISE_K2048",
        "candidate_type": "TRAINABLE RETRIEVAL AND RERANKING",
        "retrieval": "EXACT-THEN-WEIGHTED RESPONSE DISTANCE",
        "reranker": "AMBIGUITY-AWARE LISTWISE LINEAR RERANKER WITH FROZEN GRAPH AND V1 SUPPORT",
        "trainable_parameters_max": 4096,
        "maximum_shortlist": 2048,
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
    require(sha256(path) == item.get("sha256"), f"{label} SHA changed: {path.name}")
    if "bytes" in item:
        require(path.stat().st_size == int(item["bytes"]), f"{label} size changed: {path.name}")
    return path


def verify_manifest(manifest: dict[str, Any], label: str) -> None:
    for section in ("input_evidence", "outputs"):
        entries = manifest.get(section)
        require(isinstance(entries, dict), f"{label} {section}")
        for item in entries.values():
            if isinstance(item, dict) and isinstance(item.get("path"), str):
                verify_record(item, f"{label} {section}")


def find_record(entries: dict[str, Any], basename: str) -> dict[str, Any]:
    matches = []
    for item in entries.values():
        if isinstance(item, dict) and isinstance(item.get("path"), str):
            if Path(item["path"]).name == basename:
                matches.append(item)
    require(len(matches) == 1, f"lineage artifact resolution: {basename}")
    return matches[0]


def candidate_csv() -> bytes:
    fields = [
        "candidate_id", "candidate_type", "retrieval", "reranker",
        "trainable_parameters", "trainable_parameters_max", "maximum_shortlist",
    ]
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for candidate in CANDIDATES:
        writer.writerow(candidate)
    return output.getvalue().encode()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def verify_r3() -> dict[str, Any]:
    print("STAGE 12B-1A — V2.1 LOCATOR REDESIGN ARCHITECTURE AND TRAINING CONTRACT")
    print("FROZEN INPUT VERIFICATION")
    evidence: dict[str, Any] = {}
    for path, expected in PINNED.items():
        require(path.is_file(), f"missing frozen input: {rel(path)}")
        require(sha256(path) == expected, f"frozen input SHA: {rel(path)}")
        evidence[rel(path)] = record(path)
        print(f"  {path.name:<80}: OK", flush=True)

    for path in (R3_POLICY, R3_CORE_LOCK, R3_COMPARISON, R3_REPORT, R3_MANIFEST, R3_AUDIT):
        require(path.is_file(), f"missing R3 frozen artifact: {rel(path)}")
        evidence[rel(path)] = record(path)
        print(f"  {path.name:<80}: OK", flush=True)

    r3_policy = load_json(R3_POLICY)
    r3_lock = load_json(R3_CORE_LOCK)
    r3_manifest = load_json(R3_MANIFEST)
    r3_audit = load_json(R3_AUDIT)
    r2_audit = load_json(R2_AUDIT)

    require(r3_policy.get("status") == "FROZEN", "R3 disposition policy")
    require(r3_policy.get("automatic_continuation") == "NONE", "R3 automatic continuation")
    require(r3_lock.get("status") == "PASS", "R3 core lock status")
    require(r3_lock.get("v2_core_result") == "PARTIAL SUCCESS", "R3 V2 outcome")
    require(r3_lock.get("calibration_advancement_target") == "NOT_MET", "R3 advancement state")
    require(r3_lock.get("locked_dev_site_test_evaluation_authorized") is False, "R3 site-test authorization")
    require(r3_lock.get("dev_site_test_opened") is False, "R3 site-test access")
    require(r3_lock.get("validation_access_count") == 0, "R3 VALIDATION access")
    require(r3_lock.get("holdout_access_count") == 0, "R3 HOLDOUT access")
    require(r3_lock.get("v1_model_modified") is False, "R3 V1 state")
    require(r3_audit.get("status") == "PASS", "R3 audit status")
    require(r3_audit.get("disposition_status") == "FROZEN", "R3 disposition freeze")
    require(r3_audit.get("calibration_advancement_target") == "NOT_MET", "R3 audit gate")
    require(r3_audit.get("dev_site_test") == "LOCKED / NOT OPENED / NOT AUTHORIZED", "R3 audit site test")
    require(r3_audit.get("validation_access_count") == 0, "R3 audit VALIDATION access")
    require(r3_audit.get("holdout_access_count") == 0, "R3 audit HOLDOUT access")
    require(r2_audit.get("calibration_advancement_target") == "NOT_MET", "R2 lineage state")

    verify_record(r3_audit.get("manifest", {}), "R3 audit manifest")
    verify_record(r3_audit.get("core_lock", {}), "R3 audit core lock")
    verify_manifest(r3_manifest, "R3 manifest")
    r2_lineage = find_record(r3_manifest["input_evidence"], R2_AUDIT.name)
    require(r2_lineage.get("sha256") == PINNED[R2_AUDIT], "R3-to-R2 lineage")
    require(canonical_json(r3_policy) == R3_POLICY.read_bytes(), "R3 policy canonical replay")
    require(canonical_json(r3_lock) == R3_CORE_LOCK.read_bytes(), "R3 lock canonical replay")
    require(canonical_json(r3_manifest) == R3_MANIFEST.read_bytes(), "R3 manifest canonical replay")
    require(canonical_json(r3_audit) == R3_AUDIT.read_bytes(), "R3 audit canonical replay")
    print("  R3 semantic disposition, recursive manifest and R2 lineage                  : PASS")
    return evidence


def self_test() -> None:
    require(len(CANDIDATES) == 3, "candidate count canary")
    require(CANDIDATES[0]["trainable_parameters"] == 0, "reference candidate canary")
    require(CANDIDATES[-1]["maximum_shortlist"] < FAULTS, "shortlist canary")
    require(canonical_json({"b": 2, "a": 1}) == canonical_json({"a": 1, "b": 2}), "JSON replay canary")
    require(candidate_csv() == candidate_csv(), "CSV replay canary")
    print("Stage 12B-1A self-test: PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return

    require(ROOT.name == "vlsi_fault_detection_v2", f"run from project root, not {ROOT}")
    outputs = (ARCHITECTURE, TRAINING_CONTRACT, ACCEPTANCE, CANDIDATE_GRID, ENVIRONMENT, MANIFEST, AUDIT)
    for path in outputs:
        require(not path.exists(), f"Stage {STAGE} output already exists: {rel(path)}")

    evidence = verify_r3()

    architecture = {
        "architecture_version": "CIRCUITSAGE-HMAC-V2.1-AMBIGUITY-AWARE-LOCATOR-v1",
        "stage": STAGE,
        "status": "FROZEN",
        "model_family": "CLOSED-CATALOG BEHAVIOR-SIGNATURE RETRIEVAL WITH GRAPH-AWARE RERANKING",
        "scientific_scope": {
            "circuit": "FROZEN OPENTITAN HMAC-SHA256 GENERIC NETLIST",
            "faults": "SINGLE PERSISTENT SA0/SA1",
            "catalog_fault_instances": FAULTS,
            "physical_sites": SITES,
            "query_vectors": TRAIN_VECTORS,
            "closed_catalog_only": True,
            "independent_chip_generalization_claimed": False,
        },
        "query_input": {
            "allowed": [
                "ordered test-vector commitment",
                "observed output digest or timeout per vector",
                "golden-reference response for the same vectors",
            ],
            "derived": [
                "digest XOR and Hamming summaries",
                "timeout and latency deltas",
                "64-vector anomaly/deviation mask",
                "canonical response-signature hash",
            ],
            "forbidden": [
                "fault instance identity",
                "physical-site identity",
                "SA0/SA1 ground truth",
                "detection or localization target label",
            ],
        },
        "pipeline": [
            {
                "order": 1,
                "component": "FROZEN EXACT GOLDEN-REFERENCE DETECTOR",
                "action": "return NO_OBSERVED_ANOMALY when every response matches; otherwise continue",
            },
            {
                "order": 2,
                "component": "CANONICAL SIGNATURE INDEX",
                "action": "return every exact catalog match as a candidate set; never collapse an ambiguous signature",
            },
            {
                "order": 3,
                "component": "WEIGHTED NEAR-SIGNATURE RETRIEVAL",
                "action": "when no exact match exists, retrieve a deterministic shortlist using learned group weights",
            },
            {
                "order": 4,
                "component": "GRAPH-AWARE LISTWISE RERANKER",
                "action": "rerank only the shortlist using response distance, 119 frozen graph features and frozen V1 support",
            },
            {
                "order": 5,
                "component": "AMBIGUITY-AWARE OUTPUT",
                "action": "emit UNIQUE_SITE, AMBIGUOUS_CANDIDATE_SET, NO_CATALOG_MATCH, or NO_OBSERVED_ANOMALY",
            },
        ],
        "static_inputs": {
            "graph_features": GRAPH_FEATURES,
            "v1_combined_features": V1_COMBINED_FEATURES,
            "v1_role": "FROZEN READ-ONLY SUPPORT / OPTIONAL TIE BREAKER",
            "new_graph_propagation": 0,
            "v1_weight_changes": 0,
        },
        "candidate_grid": CANDIDATES,
    }

    training_contract = {
        "contract_version": "CIRCUITSAGE-HMAC-V2.1-LOCATOR-TRAINING-v1",
        "stage": STAGE,
        "status": "FROZEN",
        "purpose": "test whether direct behavior-signature retrieval and ambiguity-aware scoring improve closed-catalog localization",
        "data_partitions": {
            "DEV_TRAIN": "signature-index construction, trainable distance weights, hard negatives and listwise fitting",
            "DEV_CALIBRATION": "candidate selection and frozen acceptance measurement only",
            "DEV_SITE_TEST": "LOCKED UNTIL EVERY V2.1 CALIBRATION ADVANCEMENT CRITERION PASSES",
            "VALIDATION": "PROHIBITED",
            "HOLDOUT": "PROHIBITED",
        },
        "fit_rules": {
            "physical_site_grouping": "SA0 AND SA1 OF A SITE MUST REMAIN TOGETHER",
            "query_identity_feature_count": 0,
            "target_feature_count": 0,
            "candidate_identity_use": "INDEXING AND LOSS TARGET ONLY; NEVER QUERY INPUT",
            "catalog_construction_partition": "DEV_TRAIN RESPONSES ONLY",
            "ambiguous_signature_loss": "MULTI-POSITIVE SET LOSS; ALL OBSERVATIONALLY EQUIVALENT SITES ARE CORRECT",
            "unique_signature_loss": "SITE-LEVEL LISTWISE RETRIEVAL",
            "fault_type_loss": "CONDITIONAL SA0/SA1 LOSS ONLY AFTER CORRECT SITE RETRIEVAL",
            "hard_negatives": [
                "nearest incorrect response signatures",
                "graph-neighbor sites",
                "SA0/SA1 sibling",
                "highest-scoring incorrect shortlist members",
            ],
        },
        "execution": {
            "candidate_count": len(CANDIDATES),
            "trainable_candidates": 2,
            "candidate_execution": "SEQUENTIAL",
            "parallel_candidates": 1,
            "checkpoint_resume": "REQUIRED",
            "deterministic_replay": "EXACT",
            "estimated_cpu_time": "1-6 HOURS",
            "gpu_required": False,
            "internet_required": False,
        },
        "authorizations": {
            "next_stage_signature_dataset_construction": "AUTHORIZED FROM DEV_TRAIN ONLY",
            "model_training": "NOT YET AUTHORIZED",
            "dev_site_test_evaluation": "NOT AUTHORIZED",
            "validation": "NOT AUTHORIZED",
            "holdout": "NOT AUTHORIZED",
        },
        "frozen_predecessors": {
            "V1": "READ ONLY / MUST NOT CHANGE",
            "V2_CORE_12A": "COMPLETED / READ ONLY / MUST NOT CHANGE",
        },
    }

    acceptance = {
        "contract_version": "CIRCUITSAGE-HMAC-V2.1-LOCATOR-ACCEPTANCE-v1",
        "stage": STAGE,
        "status": "FROZEN",
        "rationale": "exact top-k accuracy is not a valid sole target when multiple catalog sites have identical observable signatures",
        "primary_selection_order": [
            "observable candidate-set coverage",
            "unique-signature top-1 site accuracy",
            "ambiguous false-unique rate",
            "observable mean reciprocal rank",
            "mean returned candidate-set size",
        ],
        "calibration_advancement_targets": {
            "fault_free_false_alarm_rate_max": 0.05,
            "observable_detection_recall_min": 0.90,
            "exact_signature_candidate_set_coverage_min": 0.95,
            "unique_signature_top1_site_accuracy_min": 0.95,
            "observable_candidate_set_coverage_min": 0.95,
            "ambiguous_false_unique_rate_max": 0.01,
            "no_catalog_match_false_unique_rate_max": 0.01,
            "observable_mean_reciprocal_rank_greater_than_v2_core": BEST_V2_MRR,
            "deterministic_replay": "EXACT",
        },
        "reported_not_primary_gates": [
            "top-5 site accuracy",
            "top-10 site accuracy",
            "top-50 candidate coverage",
            "conditional SA0/SA1 accuracy",
            "candidate-set size distribution",
            "exact-match, near-match and no-match breakdowns",
        ],
        "claim_boundaries": {
            "successful_result": "closed-catalog HMAC SA0/SA1 behavior-based localization only",
            "independent_generalization": "NOT ESTABLISHED BY THIS EXPERIMENT",
            "production_or_silicon_readiness": "NOT ESTABLISHED",
            "normal_response": "CANNOT PROVE ABSENCE OF AN INVISIBLE OR UNACTIVATED FAULT",
        },
        "if_not_met": "freeze negative result; do not open DEV_SITE_TEST",
        "if_met": "authorize a separate locked DEV_SITE_TEST evaluation contract",
    }

    environment = {
        "environment_version": "CIRCUITSAGE-HMAC-V2.1-ENVIRONMENT-v1",
        "stage": STAGE,
        "status": "FROZEN",
        "python": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": {
            name: package_version(name)
            for name in ("numpy", "scipy", "scikit-learn", "joblib")
        },
        "environment_variables": {
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
        },
        "internet_required": False,
        "gpu_required": False,
    }

    frozen_write(ARCHITECTURE, canonical_json(architecture))
    frozen_write(TRAINING_CONTRACT, canonical_json(training_contract))
    frozen_write(ACCEPTANCE, canonical_json(acceptance))
    frozen_write(CANDIDATE_GRID, candidate_csv())
    frozen_write(ENVIRONMENT, canonical_json(environment))

    manifest = {
        "manifest_version": "CIRCUITSAGE-HMAC-V2.1-CONTRACT-MANIFEST-v1",
        "stage": STAGE,
        "status": "PASS",
        "input_evidence": evidence,
        "outputs": {
            rel(path): record(path)
            for path in (ARCHITECTURE, TRAINING_CONTRACT, ACCEPTANCE, CANDIDATE_GRID, ENVIRONMENT)
        },
        "model_training_calls": 0,
        "model_inference_calls": 0,
        "model_objects_deserialized": 0,
        "dev_site_test_opened": False,
        "validation_access_count": 0,
        "holdout_access_count": 0,
    }
    frozen_write(MANIFEST, canonical_json(manifest))
    audit = {
        "audit_version": "CIRCUITSAGE-HMAC-V2.1-LOCATOR-REDESIGN-CONTRACT-FREEZE-v1",
        "stage": STAGE,
        "status": "PASS",
        "architecture_status": "FROZEN",
        "training_contract_status": "FROZEN",
        "acceptance_contract_status": "FROZEN",
        "environment_status": "FROZEN",
        "candidate_grid_status": "FROZEN",
        "model_family": architecture["model_family"],
        "ambiguity_aware": True,
        "closed_catalog_scope": True,
        "candidates": len(CANDIDATES),
        "trainable_candidates": 2,
        "v1_status": "FROZEN / READ ONLY / UNCHANGED",
        "v2_core_status": "FROZEN / READ ONLY / UNCHANGED",
        "model_training": "NOT STARTED / NOT AUTHORIZED BY THIS STAGE",
        "dev_site_test": "LOCKED / NOT OPENED / NOT AUTHORIZED",
        "validation_access_count": 0,
        "holdout_access_count": 0,
        "independent_generalization": "NOT CLAIMED",
        "manifest": record(MANIFEST),
        "architecture": record(ARCHITECTURE),
        "training_contract": record(TRAINING_CONTRACT),
        "acceptance_contract": record(ACCEPTANCE),
        "candidate_grid": record(CANDIDATE_GRID),
        "environment": record(ENVIRONMENT),
        "next_gate": "STAGE 12B-1B — BEHAVIOR-SIGNATURE INDEX AND AMBIGUITY-AWARE DATASET FREEZE",
    }
    frozen_write(AUDIT, canonical_json(audit))

    for path in (ARCHITECTURE, TRAINING_CONTRACT, ACCEPTANCE, ENVIRONMENT, MANIFEST, AUDIT):
        require(canonical_json(load_json(path)) == path.read_bytes(), f"deterministic JSON replay: {path.name}")
    require(candidate_csv() == CANDIDATE_GRID.read_bytes(), "deterministic candidate-grid replay")

    print("\nSTAGE 12B-1A — V2.1 LOCATOR REDESIGN ARCHITECTURE AND TRAINING-CONTRACT FREEZE")
    print(f"{'Status':<43}: PASS")
    print(f"{'Architecture status':<43}: FROZEN")
    print(f"{'Training contract status':<43}: FROZEN")
    print(f"{'Acceptance contract status':<43}: FROZEN")
    print(f"{'Model family':<43}: BEHAVIOR-SIGNATURE RETRIEVAL + GRAPH RERANKING")
    print(f"{'Scientific scope':<43}: CLOSED-CATALOG HMAC SA0/SA1")
    print(f"{'Ambiguity-aware localization':<43}: YES")
    print(f"{'Candidate models':<43}: {len(CANDIDATES)}")
    print(f"{'Trainable candidates':<43}: 2")
    print(f"{'Fault catalog / physical sites':<43}: {FAULTS} / {SITES}")
    print(f"{'TRAIN response vectors':<43}: {TRAIN_VECTORS}")
    print(f"{'Graph features':<43}: {GRAPH_FEATURES}")
    print(f"{'Unknown identity in query input':<43}: PROHIBITED")
    print(f"{'V1 / V2 Core modified':<43}: NO / NO")
    print(f"{'Model training':<43}: NOT YET AUTHORIZED")
    print(f"{'DEV_SITE_TEST':<43}: LOCKED / NOT OPENED")
    print(f"{'VALIDATION / HOLDOUT access':<43}: 0 / 0")
    print(f"{'Estimated CPU training time':<43}: 1-6 HOURS")
    print(f"{'Architecture':<43}: {ARCHITECTURE}")
    print(f"{'Architecture SHA':<43}: {sha256(ARCHITECTURE)}")
    print(f"{'Training contract':<43}: {TRAINING_CONTRACT}")
    print(f"{'Training contract SHA':<43}: {sha256(TRAINING_CONTRACT)}")
    print(f"{'Acceptance contract':<43}: {ACCEPTANCE}")
    print(f"{'Acceptance contract SHA':<43}: {sha256(ACCEPTANCE)}")
    print(f"{'Candidate grid':<43}: {CANDIDATE_GRID}")
    print(f"{'Candidate grid SHA':<43}: {sha256(CANDIDATE_GRID)}")
    print(f"{'Environment':<43}: {ENVIRONMENT}")
    print(f"{'Environment SHA':<43}: {sha256(ENVIRONMENT)}")
    print(f"{'Audit':<43}: {AUDIT}")
    print(f"{'Audit SHA':<43}: {sha256(AUDIT)}")
    print(f"{'Next gate':<43}: STAGE 12B-1B — BEHAVIOR-SIGNATURE INDEX AND AMBIGUITY-AWARE DATASET FREEZE")


if __name__ == "__main__":
    main()
