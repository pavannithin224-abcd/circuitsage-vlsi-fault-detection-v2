#!/usr/bin/env python3
"""Stage 12A-1C: CircuitSage-HMAC V2 detector/locator architecture freeze.

This stage verifies the frozen V2 core and response dataset, then writes the
architecture, training, candidate-grid, environment, and audit contracts for
the linked behavior detector and graph locator.  It does not train or load a
model and it does not open DEV_SITE_TEST, VALIDATION, or HOLDOUT data.
"""

from __future__ import annotations

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


STAGE = "12A-1C"
VERSION = "CIRCUITSAGE-HMAC-V2-DETECTOR-LOCATOR-CONTRACT-v1"
ROOT = Path(__file__).resolve().parent
CONFIG_ROOT = ROOT / "config" / "v2"
RESULT_ROOT = ROOT / "results" / "circuitsage_hmac_v2_12a1"
DATASET_ROOT = RESULT_ROOT / "blinded_train_response_dataset_12a1b"

ARCHITECTURE_PATH = CONFIG_ROOT / "circuitsage_hmac_v2_detector_locator_architecture_12a1c.json"
TRAINING_CONTRACT_PATH = CONFIG_ROOT / "circuitsage_hmac_v2_training_contract_12a1c.json"
CANDIDATE_GRID_PATH = CONFIG_ROOT / "circuitsage_hmac_v2_candidate_grid_12a1c.csv"
ENVIRONMENT_PATH = RESULT_ROOT / "circuitsage_hmac_v2_environment_12a1c.json"
AUDIT_PATH = RESULT_ROOT / "circuitsage_hmac_v2_architecture_training_contract_freeze_12a1c.json"

INPUTS: dict[Path, str] = {
    ROOT / "stage_12a1a_v2_core_contract.py":
        "1fca14eb7384794baa5b8a0bc68aa2124f3a92fbd1f8d80f63984e96e39f9b9e",
    CONFIG_ROOT / "circuitsage_hmac_v2_core_scope_contract_12a1a.json":
        "38118c0ba90c7d2af6ef8a01fcf7bd1e86ef13929fa7766e640db1ee08d482e6",
    CONFIG_ROOT / "circuitsage_hmac_v2_linked_pipeline_contract_12a1a.json":
        "312fde4cb5ecc7819418e4affe88d1b07c7dbe5cbdeab5af92f14403ffd7427d",
    CONFIG_ROOT / "circuitsage_hmac_v2_acceptance_contract_12a1a.json":
        "3943601adcdd603e78b07699317066b3a187d22d1f7f9b71106aab4feed95c93",
    RESULT_ROOT / "circuitsage_hmac_v2_core_contract_freeze_12a1a.json":
        "b0d1a0c2bebbe0a4579dad5fca0efec01b8d569926cb4cdab1efc9e9a086fa75",
    ROOT / "stage_12a1b_blinded_train_response_dataset.py":
        "d0dd96765a10f9821c284b21f6d757771710b46064016c0a1506fbd97d237032",
    DATASET_ROOT / "circuitsage_hmac_v2_train_response_features_12a1b.npz":
        "09f5dded67a6f82d14f02b4de23765b66ff954eb6a9779817c8b7286dc6ac2b3",
    DATASET_ROOT / "circuitsage_hmac_v2_train_response_targets_12a1b.npz":
        "268e407a9d13adab6a81b4650b4c16761dffb99d6d3d7a32ab2c28ff9a57a572",
    DATASET_ROOT / "circuitsage_hmac_v2_train_response_schema_12a1b.json":
        "5743d02ad5a8067aef331e955b6e91acd599b066956b0e572c7d25b2c1f98032",
    RESULT_ROOT / "circuitsage_hmac_v2_train_response_manifest_12a1b.json":
        "dd667b287285e0fb6cca6930de8415bc97b2a8b4213840205aa7b4b842dbdbca",
    RESULT_ROOT / "circuitsage_hmac_v2_train_response_dataset_freeze_12a1b.json":
        "e4b5e6f3aa60c2f49668ed5105549898729af5985e5e8db7d8eabc1e86aaf222",
    ROOT / "results/hmac_fault_campaign_11d1/graph_dataset_11d1a/hmac_golden_netlist_graph_11d1a.npz":
        "e3c2dd2214b544231186c29d8d9cb5aa6621b4d4b4bc9002150ac4f6c208c052",
    ROOT / "results/hmac_fault_campaign_11d1/graph_dataset_11d1a/hmac_golden_netlist_graph_schema_11d1a.json":
        "d24c3dd285891e284d20d1adcd93f793bedfd3042f20159a8a486359bc25e393",
    ROOT / "results/hmac_fault_campaign_11d2/hybrid_training_11d2b/hmac_selected_hybrid_model_11d2b.joblib":
        "12fea5eabf4a6c605325b6ce4c1217f6a37cc59750065db8b85c3da14713f6b8",
    ROOT / "results/hmac_fault_campaign_11d2/hybrid_training_11d2b/hmac_hybrid_selection_lock_11d2b.json":
        "005cecbd94456d3d1aa6805b2dc898e895cc178564673b6b0aa269fcd64260c6",
    ROOT / "config/diagnostic_model/hmac_final_diagnostic_model_lock_11d2d.json":
        "7985b534c62d93717a179d6d2b20f247a531ec0452003e35f0f65cf640d0b30d",
    ROOT / "results/hmac_fault_campaign_11c5/feature_matrix_11c5e/hmac_dev_train_feature_matrix_11c5e.npz":
        "b4feebe3080ac540edf38dcde35212b790dfb0f93fa57f1f5fc71411bde3b6b0",
    ROOT / "results/hmac_fault_campaign_11c5/feature_matrix_11c5e/hmac_dev_calibration_feature_matrix_11c5e.npz":
        "ee18d68e4b1c742e8b8b163ab3f97f92f954e3da6af88f460b5b886ea73dd05b",
}

RESPONSE_FEATURES = 2304
GRAPH_FEATURES = 119
FAULT_TYPE_FEATURES = 2
V1_SUPPORT_FEATURES = 64
SITES = 22839
FAULT_INSTANCES = 45678
TRAIN_SITES = 15987
CALIBRATION_SITES = 3426
SITE_TEST_SITES = 3426
OBSERVABLE_FAULTS = 22930
NORMAL_COMPATIBLE_FAULTS = 22748
UNIQUE_SITE_FAULTS = 9171
AMBIGUOUS_SITE_FAULTS = 13759


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"STOP: {message}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def write_frozen(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(path.is_file(), f"output is not a regular file: {relative(path)}")
        require(path.read_bytes() == payload, f"refusing to overwrite different frozen output: {relative(path)}")
        return
    temporary = path.with_name(path.name + ".tmp")
    require(not temporary.exists(), f"stale temporary output exists: {relative(temporary)}")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON root is not an object: {relative(path)}")
    return value


def verify_inputs() -> dict[str, dict[str, Any]]:
    evidence: dict[str, dict[str, Any]] = {}
    print("FROZEN INPUT VERIFICATION")
    for path, expected in INPUTS.items():
        require(path.is_file(), f"missing frozen input: {relative(path)}")
        actual = sha256_file(path)
        require(actual == expected, f"SHA mismatch: {relative(path)}")
        evidence[relative(path)] = {"sha256": actual, "bytes": path.stat().st_size}
        print(f"  {path.name:<76}: OK")
    return evidence


def verify_semantics() -> dict[str, Any]:
    core = load_json(RESULT_ROOT / "circuitsage_hmac_v2_core_contract_freeze_12a1a.json")
    dataset = load_json(RESULT_ROOT / "circuitsage_hmac_v2_train_response_dataset_freeze_12a1b.json")
    schema = load_json(DATASET_ROOT / "circuitsage_hmac_v2_train_response_schema_12a1b.json")
    manifest = load_json(RESULT_ROOT / "circuitsage_hmac_v2_train_response_manifest_12a1b.json")
    acceptance = load_json(CONFIG_ROOT / "circuitsage_hmac_v2_acceptance_contract_12a1a.json")

    require(core.get("status") == "PASS", "Stage 12A-1A status")
    require(core.get("unknown_fault_identity_in_input") == "PROHIBITED", "V2 blinding status")
    require(int(core.get("validation_access_count", -1)) == 0, "Stage 12A-1A validation access")
    require(int(core.get("holdout_access_count", -1)) == 0, "Stage 12A-1A HOLDOUT access")
    require(dataset.get("status") == "PASS", "Stage 12A-1B status")
    require(dataset.get("dataset_status") == "FROZEN", "Stage 12A-1B dataset status")
    require(dataset.get("schema_status") == "FROZEN", "Stage 12A-1B schema status")
    require(dataset.get("unknown_fault_identity_in_features") == "NO", "response blinding")
    require(dataset.get("fault_and_site_targets_separate") == "YES", "target separation")
    require(int(dataset.get("validation_access_count", -1)) == 0, "Stage 12A-1B validation access")
    require(int(dataset.get("holdout_access_count", -1)) == 0, "Stage 12A-1B HOLDOUT access")
    require(schema.get("status") == "FROZEN", "response schema status")
    require(int(schema.get("flattened_response_feature_count", -1)) == RESPONSE_FEATURES, "response dimension")
    require(manifest.get("source_partition") == "TRAIN ONLY", "response dataset source partition")
    require(int(manifest.get("physical_sites", -1)) == SITES, "physical-site count")
    require(int(manifest.get("fault_instances", -1)) == FAULT_INSTANCES, "fault-instance count")

    targets = acceptance.get("minimum_advancement_targets")
    require(isinstance(targets, dict), "V2 acceptance targets")
    expected = {
        "fault_free_false_alarm_rate_max": 0.05,
        "observable_fault_detection_recall_min": 0.90,
        "observable_candidate_coverage_min": 0.95,
        "unique_signature_top1_site_accuracy_min": 0.80,
        "observable_top5_site_accuracy_min": 0.80,
        "deterministic_replay": "EXACT",
        "unknown_identity_feature_count": 0,
        "post_simulation_target_feature_count": 0,
    }
    for key, wanted in expected.items():
        require(targets.get(key) == wanted, f"acceptance target changed: {key}")
    return {"minimum_advancement_targets": targets}


def dense_parameters(widths: list[int]) -> int:
    return sum((left * right) + right for left, right in zip(widths, widths[1:]))


def candidate_parameters(response_hidden: list[int], candidate_hidden: list[int], fusion_hidden: list[int]) -> int:
    response = dense_parameters([RESPONSE_FEATURES, *response_hidden])
    candidate = dense_parameters([
        GRAPH_FEATURES + FAULT_TYPE_FEATURES + V1_SUPPORT_FEATURES,
        *candidate_hidden,
    ])
    fusion = dense_parameters([response_hidden[-1] + candidate_hidden[-1], *fusion_hidden, 1])
    return response + candidate + fusion


def candidate_grid() -> list[dict[str, Any]]:
    rows = [
        {
            "candidate_id": "V2_LINKED_RANKER_SMALL_A1E5",
            "response_hidden": "128;64",
            "candidate_hidden": "64;32",
            "fusion_hidden": "64;16",
            "epochs": 8,
            "learning_rate": "0.001",
            "weight_decay": "0.00001",
            "negative_candidates_per_query": 128,
            "query_batch_size": 64,
            "random_seed": 20260914,
        },
        {
            "candidate_id": "V2_LINKED_RANKER_WIDE_A1E5",
            "response_hidden": "192;64",
            "candidate_hidden": "96;32",
            "fusion_hidden": "96;24",
            "epochs": 6,
            "learning_rate": "0.0005",
            "weight_decay": "0.00001",
            "negative_candidates_per_query": 128,
            "query_batch_size": 48,
            "random_seed": 20260914,
        },
    ]
    for row in rows:
        response = [int(value) for value in row["response_hidden"].split(";")]
        candidate = [int(value) for value in row["candidate_hidden"].split(";")]
        fusion = [int(value) for value in row["fusion_hidden"].split(";")]
        row["trainable_parameters"] = candidate_parameters(response, candidate, fusion)
    require(rows[0]["trainable_parameters"] == 324545, "small candidate parameter canary")
    require(rows[1]["trainable_parameters"] == 487537, "wide candidate parameter canary")
    return rows


def csv_bytes(rows: list[dict[str, Any]]) -> bytes:
    fields = [
        "candidate_id", "response_hidden", "candidate_hidden", "fusion_hidden",
        "trainable_parameters", "epochs", "learning_rate", "weight_decay",
        "negative_candidates_per_query", "query_batch_size", "random_seed",
    ]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "NOT_INSTALLED"


def main() -> None:
    print("STAGE 12A-1C — V2 DETECTOR/LOCATOR ARCHITECTURE AND TRAINING CONTRACT")
    input_evidence = verify_inputs()
    semantic_evidence = verify_semantics()
    candidates = candidate_grid()

    v1_support = {
        "role": "READ-ONLY CANDIDATE DETECTABILITY SUPPORT",
        "source_model": relative(ROOT / "results/hmac_fault_campaign_11d2/hybrid_training_11d2b/hmac_selected_hybrid_model_11d2b.joblib"),
        "model_type": "sklearn.neural_network.MLPClassifier",
        "model_input_features": 646,
        "support_vector_features": V1_SUPPORT_FEATURES,
        "construction": "one frozen V1 detectability probability per TRAIN response vector for each candidate fault instance",
        "allowed_inputs": ["527 frozen pre-simulation sample features", "119 frozen K3 graph features"],
        "forbidden_inputs": ["observed query identity", "ground-truth site", "ground-truth SA0/SA1", "detection label"],
        "weights_trainable": False,
        "threshold_trainable": False,
        "weight_merging": "PROHIBITED",
    }

    architecture = {
        "architecture_version": VERSION,
        "stage": STAGE,
        "status": "FROZEN",
        "model_name": "CircuitSage-HMAC V2 Linked Behavior Detector and Graph Locator",
        "task": "BLINDED OBSERVATION-ONLY FAULT DETECTION AND CANDIDATE RANKING",
        "fault_scope": "SINGLE PERSISTENT SA0/SA1 ON FROZEN HMAC CATALOG",
        "detection": {
            "type": "EXACT GOLDEN-REFERENCE ANOMALY GATE",
            "trainable_parameters": 0,
            "input": "64 ordered observed HMAC responses plus frozen normal responses",
            "rule": "anomaly iff any response deviation indicator is one",
            "observable_fault_semantics": "detects only faults that change at least one captured response",
            "normal_compatible_semantics": "NO_OBSERVED_ANOMALY; latent fault cannot be excluded",
            "output": ["OBSERVABLE_ANOMALY", "NO_OBSERVED_ANOMALY"],
        },
        "localization": {
            "type": "SHARED-ENCODER DEEP PAIRWISE GRAPH RANKER",
            "query_response_features": RESPONSE_FEATURES,
            "candidate_graph_features": GRAPH_FEATURES,
            "candidate_fault_type_features": FAULT_TYPE_FEATURES,
            "candidate_v1_support_features": V1_SUPPORT_FEATURES,
            "response_encoder": "2304 -> {candidate response_hidden final 64}",
            "candidate_encoder": "185 -> {candidate candidate_hidden final 32}",
            "fusion_input_features": 96,
            "score_output": "one unbounded compatibility logit per candidate fault instance",
            "candidate_universe": FAULT_INSTANCES,
            "candidate_batching_required": True,
            "top_k_outputs": [1, 5, 10, 50],
            "fault_type_output": "SA0/SA1 of ranked fault instance",
            "ambiguous_response_policy": "return ranked candidate set and uncertainty; never force exact site",
            "no_catalog_match_policy": "return UNRESOLVED/OUT-OF-CATALOG; never invent a site",
        },
        "v1_link": v1_support,
        "candidate_architectures": candidates,
        "maximum_trainable_parameters": max(row["trainable_parameters"] for row in candidates),
        "unknown_fault_identity_in_query": "PROHIBITED",
        "catalog_membership_during_locked_query_inference": "PROHIBITED",
        "target_present_in_model_input": False,
        "post_simulation_target_features": 0,
    }

    training = {
        "contract_version": "CIRCUITSAGE-HMAC-V2-TRAINING-CONTRACT-v1",
        "stage": STAGE,
        "status": "FROZEN",
        "backend": "CUSTOM DETERMINISTIC NUMPY BACKPROPAGATION / ADAMW",
        "candidate_execution": "SEQUENTIAL",
        "parallel_candidates": 1,
        "source_vectors": 64,
        "source_partition": "TRAIN RESPONSES ONLY",
        "site_partitions": {
            "DEV_TRAIN": {"sites": TRAIN_SITES, "use": "gradient updates"},
            "DEV_CALIBRATION": {"sites": CALIBRATION_SITES, "use": "candidate selection and uncertainty calibration only"},
            "DEV_SITE_TEST": {"sites": SITE_TEST_SITES, "use": "LOCKED; not opened by training"},
        },
        "fault_instances": FAULT_INSTANCES,
        "observable_fault_instances": OBSERVABLE_FAULTS,
        "normal_compatible_fault_instances": NORMAL_COMPATIBLE_FAULTS,
        "unique_site_fault_instances": UNIQUE_SITE_FAULTS,
        "ambiguous_site_fault_instances": AMBIGUOUS_SITE_FAULTS,
        "detector_training": "NONE; exact frozen rule",
        "locator_training": {
            "eligible_queries": "observable DEV_TRAIN fault instances only",
            "loss": "MULTI-POSITIVE SAMPLED SOFTMAX RANKING LOSS",
            "positive_definition": "same observable response profile within the active partition",
            "negative_sampling": "deterministic, within DEV_TRAIN; exclude all known same-profile positives in that partition",
            "candidate_static_features": "frozen graph K3 features + SA0/SA1 one-hot + frozen V1 support vector",
            "candidate_static_cache": "may be generated from frozen pre-simulation inputs; labels must not enter cache",
            "optimizer": "ADAMW",
            "activation": "RELU",
            "gradient_clip_l2": 5.0,
            "early_stopping": "DISABLED; fixed candidate epochs",
            "checkpoint_resume": "ENABLED; exact epoch boundary",
        },
        "selection": {
            "partition": "DEV_CALIBRATION",
            "primary_metric": "OBSERVABLE MEAN RECIPROCAL RANK",
            "tie_breakers": ["UNIQUE-SIGNATURE TOP1 SITE ACCURACY", "OBSERVABLE TOP5 SITE ACCURACY", "LOWER PARAMETER COUNT"],
            "profile_overlap_policy": "report cross-partition and profile-novel subsets separately; unique-site top1 is mandatory",
            "threshold_selection": "NONE FOR EXACT DETECTOR; ranker uncertainty calibration only",
        },
        "locked_evaluation_after_selection": {
            "DEV_SITE_TEST": "one evaluation only after model and calibration lock",
            "VALIDATION": "historical V1 evidence only; V2 access prohibited",
            "HOLDOUT": "blocked until a separately frozen authorization",
        },
        "required_metrics": [
            "fault-free false-alarm rate", "observable detection recall",
            "observable candidate coverage", "unique-signature top1 site accuracy",
            "observable top5 site accuracy", "mean reciprocal rank",
            "median candidate-set size", "SA0/SA1 accuracy conditional on observable anomaly",
        ],
        "required_breakdowns": [
            "unique vs ambiguous response", "cross-partition profile vs partition-local profile",
            "SA0 vs SA1", "site category", "driver cell type",
        ],
        "acceptance": semantic_evidence["minimum_advancement_targets"],
        "deterministic_replay": {
            "weights": "EXACT",
            "scores": "EXACT",
            "ranking": "EXACT",
            "metrics": "EXACT",
        },
        "estimated_cpu_training_time": "60-180 MINUTES",
        "internet_required": False,
        "gpu_required": False,
        "training_authorization": "AUTHORIZED FOR DEV_TRAIN ONLY AFTER THIS FREEZE PASSES",
        "prohibitions": [
            "opening DEV_SITE_TEST during fitting or selection",
            "opening VALIDATION or HOLDOUT",
            "modifying or fine-tuning V1",
            "using fault/site/type identity from the query as input",
            "using profile_index or catalog membership as a query feature",
            "treating normal-compatible faults as observable detections",
        ],
    }

    environment = {
        "environment_version": "CIRCUITSAGE-HMAC-V2-ENVIRONMENT-v1",
        "stage": STAGE,
        "status": "PASS",
        "environment_status": "FROZEN",
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "packages": {
            "numpy": package_version("numpy"),
            "scikit-learn": package_version("scikit-learn"),
            "scipy": package_version("scipy"),
            "joblib": package_version("joblib"),
        },
        "training_backend": training["backend"],
        "thread_policy": "single candidate; BLAS thread count inherited but recorded at execution",
        "internet_required": False,
        "gpu_required": False,
    }

    architecture_bytes = canonical_json(architecture)
    training_bytes = canonical_json(training)
    grid_bytes = csv_bytes(candidates)
    environment_bytes = canonical_json(environment)
    require(architecture_bytes == canonical_json(architecture), "architecture deterministic replay")
    require(training_bytes == canonical_json(training), "training contract deterministic replay")
    require(grid_bytes == csv_bytes(candidates), "candidate grid deterministic replay")
    require(environment_bytes == canonical_json(environment), "environment deterministic replay")

    write_frozen(ARCHITECTURE_PATH, architecture_bytes)
    write_frozen(TRAINING_CONTRACT_PATH, training_bytes)
    write_frozen(CANDIDATE_GRID_PATH, grid_bytes)
    write_frozen(ENVIRONMENT_PATH, environment_bytes)

    outputs = {
        relative(path): {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        for path in (ARCHITECTURE_PATH, TRAINING_CONTRACT_PATH, CANDIDATE_GRID_PATH, ENVIRONMENT_PATH)
    }
    audit = {
        "audit_version": "CIRCUITSAGE-HMAC-V2-ARCHITECTURE-TRAINING-FREEZE-v1",
        "stage": STAGE,
        "status": "PASS",
        "architecture_status": "FROZEN",
        "training_contract_status": "FROZEN",
        "environment_status": "FROZEN",
        "candidate_grid_status": "FROZEN",
        "detector_type": architecture["detection"]["type"],
        "locator_type": architecture["localization"]["type"],
        "trainable_candidates": len(candidates),
        "parameter_range": [
            min(row["trainable_parameters"] for row in candidates),
            max(row["trainable_parameters"] for row in candidates),
        ],
        "full_catalog_ranking_required": True,
        "v1_model_verified": True,
        "v1_model_loaded": False,
        "v1_model_modified": False,
        "model_training_performed": False,
        "model_inference_performed": False,
        "dev_site_test_opened": False,
        "validation_access_count": 0,
        "holdout_access_count": 0,
        "unknown_fault_identity_feature_count": 0,
        "post_simulation_target_feature_count": 0,
        "frozen_inputs_modified": False,
        "input_evidence": input_evidence,
        "frozen_outputs": outputs,
        "next_gate": "STAGE 12A-1D — V2 LINKED DETECTOR/LOCATOR TRAINING AND CALIBRATION EXECUTION",
    }
    audit_bytes = canonical_json(audit)
    require(audit_bytes == canonical_json(audit), "audit deterministic replay")
    write_frozen(AUDIT_PATH, audit_bytes)

    print("\nSTAGE 12A-1C — V2 DETECTOR/LOCATOR ARCHITECTURE AND TRAINING-CONTRACT FREEZE")
    print(f"{'Status':<36}: PASS")
    print(f"{'Architecture status':<36}: FROZEN")
    print(f"{'Training contract status':<36}: FROZEN")
    print(f"{'Detector':<36}: EXACT GOLDEN-REFERENCE ANOMALY GATE")
    print(f"{'Detector trainable parameters':<36}: 0")
    print(f"{'Locator':<36}: DEEP SHARED-ENCODER GRAPH RANKER")
    print(f"{'Response / graph / V1 support':<36}: 2304 / 119 / 64")
    print(f"{'Fault-type candidate features':<36}: 2")
    print(f"{'Candidate universe':<36}: {FAULT_INSTANCES}")
    print(f"{'Trainable candidates':<36}: {len(candidates)}")
    print(f"{'Parameter range':<36}: {candidates[0]['trainable_parameters']} - {candidates[1]['trainable_parameters']}")
    print(f"{'Training backend':<36}: DETERMINISTIC NUMPY / ADAMW")
    print(f"{'Selection partition':<36}: DEV_CALIBRATION")
    print(f"{'Primary localization metric':<36}: MRR")
    print(f"{'Ambiguity handling':<36}: RANKED SET + UNCERTAINTY")
    print(f"{'V1 hybrid':<36}: VERIFIED / FROZEN / READ-ONLY SUPPORT")
    print(f"{'Unknown identity in query':<36}: PROHIBITED")
    print(f"{'DEV_SITE_TEST opened':<36}: NO")
    print(f"{'VALIDATION / HOLDOUT access':<36}: 0 / 0")
    print(f"{'V2 training':<36}: AUTHORIZED FOR DEV_TRAIN")
    print(f"{'Estimated CPU time':<36}: 60-180 MINUTES")
    print(f"{'Architecture':<36}: {ARCHITECTURE_PATH}")
    print(f"{'Architecture SHA':<36}: {sha256_file(ARCHITECTURE_PATH)}")
    print(f"{'Training contract':<36}: {TRAINING_CONTRACT_PATH}")
    print(f"{'Training contract SHA':<36}: {sha256_file(TRAINING_CONTRACT_PATH)}")
    print(f"{'Candidate grid':<36}: {CANDIDATE_GRID_PATH}")
    print(f"{'Candidate grid SHA':<36}: {sha256_file(CANDIDATE_GRID_PATH)}")
    print(f"{'Environment':<36}: {ENVIRONMENT_PATH}")
    print(f"{'Environment SHA':<36}: {sha256_file(ENVIRONMENT_PATH)}")
    print(f"{'Audit':<36}: {AUDIT_PATH}")
    print(f"{'Audit SHA':<36}: {sha256_file(AUDIT_PATH)}")
    print(f"{'Next gate':<36}: STAGE 12A-1D — V2 LINKED DETECTOR/LOCATOR TRAINING AND CALIBRATION EXECUTION")


if __name__ == "__main__":
    main()
