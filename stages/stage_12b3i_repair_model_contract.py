#!/usr/bin/env python3
"""Stage 12B-3I: repair-model architecture, partition and training contract.

Freezes a CPU-feasible, ambiguity-aware behavior-retrieval/reranking design
around the accepted EM_TESTPOINT_4X64_T16 measurement. This stage verifies the
12B-3H disposition and defines future data use, candidates, objectives and
acceptance rules. It does not capture new responses, deserialize/train a model,
fit a scaler, select thresholds, or open protected evaluation partitions.
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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STAGE = "12B-3I"
ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "config/v2_1_improvement"
RESULT = ROOT / "results/circuitsage_hmac_v2_12b3"
WORK = RESULT / "repair_model_contract_12b3i"

SOURCE_3H = ROOT / "stage_12b3h_full_capture_disposition.py"
LOG_3H = RESULT / "stage_12b3h_20260917_183516.log"
RESOURCE_3H = RESULT / "stage_12b3h_resources_20260917_183516.log"
POLICY_3H = CONFIG / "circuitsage_hmac_v2_1_full_capture_disposition_policy_12b3h.json"
READINESS_3H = CONFIG / "circuitsage_hmac_v2_1_repair_model_training_readiness_contract_12b3h.json"
WORK_3H = RESULT / "full_capture_disposition_12b3h"
WINNER_LOCK_3H = WORK_3H / "circuitsage_hmac_v2_1_full_capture_winner_dataset_lock_12b3h.json"
REGISTRY_CSV_3H = WORK_3H / "circuitsage_hmac_v2_1_full_capture_capability_registry_12b3h.csv"
REGISTRY_JSON_3H = WORK_3H / "circuitsage_hmac_v2_1_full_capture_capability_registry_12b3h.json"
REPORT_3H = WORK_3H / "circuitsage_hmac_v2_1_full_capture_disposition_report_12b3h.md"
MANIFEST_3H = RESULT / "circuitsage_hmac_v2_1_full_capture_disposition_manifest_12b3h.json"
AUDIT_3H = RESULT / "circuitsage_hmac_v2_1_full_capture_disposition_training_readiness_freeze_12b3h.json"

CAPTURE = RESULT / "full_capture_execution_12b3g"
FEATURES = CAPTURE / "circuitsage_hmac_v2_1_full_capture_features_12b3g.npz"
TARGETS = CAPTURE / "circuitsage_hmac_v2_1_full_capture_targets_12b3g.npz"
SCHEMA_3G = CAPTURE / "circuitsage_hmac_v2_1_full_capture_dataset_schema_12b3g.json"
METRICS_3G = CAPTURE / "circuitsage_hmac_v2_1_full_capture_metrics_12b3g.json"

ARCHITECTURE = CONFIG / "circuitsage_hmac_v2_1_repair_model_architecture_12b3i.json"
FEATURE_CONTRACT = CONFIG / "circuitsage_hmac_v2_1_repair_model_feature_contract_12b3i.json"
PARTITION_CONTRACT = CONFIG / "circuitsage_hmac_v2_1_repair_model_data_partition_contract_12b3i.json"
TRAINING_CONTRACT = CONFIG / "circuitsage_hmac_v2_1_repair_model_training_contract_12b3i.json"
ACCEPTANCE_CONTRACT = CONFIG / "circuitsage_hmac_v2_1_repair_model_acceptance_contract_12b3i.json"
CANDIDATE_GRID = CONFIG / "circuitsage_hmac_v2_1_repair_model_candidate_grid_12b3i.csv"
ENVIRONMENT = WORK / "circuitsage_hmac_v2_1_repair_model_environment_12b3i.json"
REPORT = WORK / "circuitsage_hmac_v2_1_repair_model_contract_report_12b3i.md"
MANIFEST = RESULT / "circuitsage_hmac_v2_1_repair_model_contract_manifest_12b3i.json"
AUDIT = RESULT / "circuitsage_hmac_v2_1_repair_model_contract_freeze_12b3i.json"

PINNED = {
    SOURCE_3H: "d8bbc069f93f8168013da40702af4b0c1c1c6061e849bfa8021304982db1ffca",
    LOG_3H: "dd7b5b1a40fcdcf33795c0dfce39ccec0a60cc8525d70988993148c35379d707",
    RESOURCE_3H: "6d37d262b35c561ea048cb34756936560489791f78b430eb8e89ed85faa5af05",
    POLICY_3H: "fe3a60d818426ef124329426623915ab8f81ea16b93961ebf183dedf3443c4d0",
    READINESS_3H: "d31afdbd6e1895fe3d68652af0999561052c750c94e05ac3efeaece9ff19133f",
    WINNER_LOCK_3H: "0138b4e4860063ae6f3a341ac19896cc996923e185355e72e9840aa7eb306c2a",
    REGISTRY_CSV_3H: "89a1319b78913037ac8ec72b2cf1665189a440582ab1340e9163f74ea552a20e",
    REGISTRY_JSON_3H: "a4fc514597e4d273ca4192d5a554f381373510bd0e18c522610dc024d7d7af34",
    REPORT_3H: "752b5e91e8891da3ab317c680f7a5ffb75fc1be381e7b693b84b82a19ad4b2f2",
    MANIFEST_3H: "b13651acdda00d58580200dee30e3674495dcb8a0f26a3a4071f65102ec29230",
    AUDIT_3H: "21ac83a5189b3b020c646af8edcb425f17dd168e3e1ee369d74a426abce0c70f",
    FEATURES: "ea3af3be023fcafa4023776f6650ac9fc1c4cac429b2730eac5329a22e1d161d",
    TARGETS: "895457c94418b16e8d0f5e2a1464c5f03bb5a1673e40e23cdd71c95002ddd5b3",
    SCHEMA_3G: "f719dab6fd324cc093de41853fa374278d327db67e01ad5259fdc21808121b20",
    METRICS_3G: "f41502e7e76c9bb381b926f81db538cfff47bf4d11c5988bdd419925de96b1de",
}

SELECTED_MEASUREMENT = "EM_TESTPOINT_4X64_T16"
SITES = 1024
FAULTS = 2048
VECTORS = 96
DETECTION_RECALL = 0.7587890625
EXACT_SITE_RATE = 0.6787109375
MEAN_CANDIDATES = 1.9595
MAX_CANDIDATES = 36


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


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "NOT_INSTALLED"


def verify_inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    print("FROZEN INPUT VERIFICATION")
    for path, expected in PINNED.items():
        require(path.is_file(), f"missing frozen input: {rel(path)}")
        require(sha256(path) == expected, f"SHA mismatch: {rel(path)}")
        print(f"  {path.name:<100}: OK")

    disposition = load_json(POLICY_3H)
    readiness = load_json(READINESS_3H)
    winner = load_json(WINNER_LOCK_3H)
    audit = load_json(AUDIT_3H)
    metrics = load_json(METRICS_3G)
    schema = load_json(SCHEMA_3G)

    require(disposition.get("disposition") == "FULL MEASUREMENT TARGET MET", "3H disposition")
    require(readiness.get("readiness") == "READY FOR ARCHITECTURE, PARTITION, AND TRAINING-CONTRACT DESIGN", "3H readiness")
    require(readiness.get("actual_model_training") == "NOT YET AUTHORIZED", "training remains blocked")
    require(audit.get("status") == "PASS" and audit.get("full_measurement_target") == "MET", "3H audit")
    require(audit.get("selected_measurement") == SELECTED_MEASUREMENT, "measurement lock")
    require(audit.get("repair_model_training") == "NOT AUTHORIZED / NOT STARTED", "training state")
    require(audit.get("dev_site_test_validation_holdout_access") == [0, 0, 0], "protected access")
    require(winner.get("features", {}).get("sha256") == PINNED[FEATURES], "winner feature binding")
    require(winner.get("targets", {}).get("sha256") == PINNED[TARGETS], "winner target binding")
    require(winner.get("feature_identity_fields") == 0, "identity exclusion lock")

    require(metrics.get("full_measurement_acceptance") == "PASS", "measurement acceptance")
    require(abs(float(metrics["combined_all_injected_detection_recall"]) - DETECTION_RECALL) < 1e-12, "detection replay")
    require(abs(float(metrics["all_injected_exact_site_rate"]) - EXACT_SITE_RATE) < 1e-12, "exact-site replay")
    require(round(float(metrics["mean_observable_candidate_sites"]), 4) == MEAN_CANDIDATES, "mean-candidate replay")
    require(int(metrics["maximum_observable_candidate_sites"]) == MAX_CANDIDATES, "max-candidate replay")
    require(int(metrics["fault_free_false_alarms"]) == 0, "false alarms")

    expected_feature_names = {
        "baseline_control_timeline", "baseline_cycles", "baseline_probe_snapshots",
        "baseline_probe_toggle_count", "control_timeline_delta", "cycle_delta",
        "digest_xor", "external_detected", "probe_effect", "probe_snapshot_xor",
        "probe_toggle_count", "probe_toggle_delta", "source_vector_index",
        "timed_out", "vector_rank",
    }
    require(set(schema.get("features", {})) == expected_feature_names, "feature schema fields")
    require(schema.get("identity_exclusion") == "site/fault/stuck/selector absent from feature NPZ", "query identity exclusion")
    require(schema.get("source_partition") == "REPAIR_TRAIN ONLY", "source partition")
    require(schema.get("model_training_or_inference") is False, "source training state")
    print("  3H disposition, winner lock, measurement metrics and feature schema                         : PASS")
    return metrics, schema


def self_test() -> None:
    require(SITES * 2 == FAULTS, "fault/site arithmetic")
    require(1554 / FAULTS == DETECTION_RECALL, "detection arithmetic")
    require(1390 / FAULTS == EXACT_SITE_RATE, "exact-site arithmetic")
    require(len({SELECTED_MEASUREMENT, "R31_EXACT_SIGNATURE_SET"}) == 2, "identifier separation")
    payload = canonical_json({"b": 2, "a": 1})
    require(payload == b'{\n  "a": 1,\n  "b": 2\n}\n', "canonical JSON")
    print("Stage 12B-3I self-test: PASS")


def main() -> None:
    require(ROOT.name == "vlsi_fault_detection_v2", f"run from project root, not {ROOT}")
    outputs = (
        ARCHITECTURE, FEATURE_CONTRACT, PARTITION_CONTRACT, TRAINING_CONTRACT,
        ACCEPTANCE_CONTRACT, CANDIDATE_GRID, ENVIRONMENT, REPORT, MANIFEST, AUDIT,
    )
    for output in outputs:
        require(not output.exists(), f"Stage {STAGE} output already exists: {rel(output)}")

    metrics, source_schema = verify_inputs()
    timestamp = now()

    architecture = {
        "architecture_version": "CIRCUITSAGE-HMAC-V2.1-REPAIR-MODEL-ARCHITECTURE-12B3I-v1",
        "stage": STAGE,
        "status": "FROZEN",
        "created_at": timestamp,
        "scientific_scope": "CLOSED-CATALOG SINGLE PERSISTENT HMAC SA0/SA1",
        "measurement_frontend": SELECTED_MEASUREMENT,
        "pipeline": [
            {
                "order": 1,
                "component": "DETERMINISTIC_ANOMALY_GATE",
                "trainable": False,
                "rule": "external_detected OR probe_effect across any frozen vector",
                "output": "NO_OBSERVED_ANOMALY or OBSERVED_ANOMALY",
            },
            {
                "order": 2,
                "component": "EXACT_SIGNATURE_CANDIDATE_SET",
                "trainable": False,
                "rule": "return every catalog site with the exact complete measurement signature",
                "output": "exact ambiguity-preserving candidate set",
            },
            {
                "order": 3,
                "component": "BLOCK_WEIGHTED_RETRIEVAL",
                "trainable": True,
                "rule": "learn nonnegative modality weights; retrieve top-64 only when no exact signature exists",
                "output": "bounded approximate shortlist",
            },
            {
                "order": 4,
                "component": "PAIRWISE_BEHAVIOR_RERANKER",
                "trainable": True,
                "rule": "score query-candidate distance features inside the frozen shortlist",
                "output": "ranked candidates plus calibrated confidence",
            },
            {
                "order": 5,
                "component": "AMBIGUITY_AND_OOD_GUARD",
                "trainable": False,
                "rule": "never declare unique when multiple sites have an identical complete signature; reject beyond calibrated distance",
                "output": "UNIQUE_SITE, AMBIGUOUS_SITES, NO_CATALOG_MATCH, or NO_OBSERVED_ANOMALY",
            },
        ],
        "candidate_shortlist_limit": 64,
        "exact_signature_precedence": True,
        "ambiguity_preservation": "MANDATORY",
        "query_identity_inputs": "PROHIBITED",
        "candidate_identity": "SCORING INDEX ONLY; NOT A QUERY FEATURE",
        "polarity_output": "SA0/SA1 prediction reported only with candidate site set and calibrated confidence",
        "v1_v2_core_v2_1_modification": "NONE",
        "independent_generalization_claim": "PROHIBITED",
    }

    feature_contract = {
        "feature_contract_version": "CIRCUITSAGE-HMAC-V2.1-REPAIR-MODEL-FEATURES-12B3I-v1",
        "stage": STAGE,
        "status": "FROZEN",
        "source_features": record(FEATURES),
        "separate_targets": record(TARGETS),
        "source_schema": record(SCHEMA_3G),
        "query_feature_blocks": [
            {"name": "external_behavior", "fields": ["timed_out", "cycle_delta", "control_timeline_delta", "digest_xor"]},
            {"name": "probe_temporal_behavior", "fields": ["probe_effect", "probe_snapshot_xor"]},
            {"name": "probe_transition_behavior", "fields": ["probe_toggle_delta"]},
        ],
        "deterministic_pooling": {
            "external_per_vector": ["timeout", "absolute_cycle_delta", "control_event_distance", "digest_hamming_weight"],
            "probe_per_vector_per_bank": ["snapshot_hamming_occupancy", "toggle_l1_distance"],
            "banks": 4,
            "vectors": VECTORS,
            "global_summary": ["active_vector_count", "first_active_vector", "maximum_activity", "mean_activity", "activity_quantiles"],
            "maximum_compiled_query_dimensions": 2048,
        },
        "pairwise_candidate_features": [
            "exact_signature_match", "external_hamming_distance", "timeout_disagreement",
            "cycle_l1_distance", "control_timeline_l1_distance",
            "probe_snapshot_hamming_distance_by_bank", "probe_toggle_l1_distance_by_bank",
            "active_vector_overlap", "first_activity_distance", "shortlist_rank",
        ],
        "pairwise_feature_dimension_max": 32,
        "fitting_rules": {
            "scaler_fit": "REPAIR_TRAIN ONLY",
            "missing_value_imputation": "NONE EXPECTED; ANY MISSING VALUE IS FATAL",
            "feature_selection": "FROZEN BEFORE REPAIR_CALIBRATION SCORING",
            "new_graph_propagation": "NOT USED IN THIS TIMEBOXED REPAIR",
        },
        "forbidden_query_fields": [
            "fault_instance_index", "full_rank", "site_index", "site_id", "selector",
            "stuck_value", "fault_enable", "fault_raw", "batch_id as identity",
        ],
        "target_fields": ["site_index", "stuck_value"],
        "target_storage": "SEPARATE NPZ; SCORING/TRAINING LABELS ONLY",
    }

    partition_contract = {
        "partition_contract_version": "CIRCUITSAGE-HMAC-V2.1-REPAIR-MODEL-PARTITIONS-12B3I-v1",
        "stage": STAGE,
        "status": "FROZEN",
        "grouping_unit": "PHYSICAL SITE; SA0 AND SA1 MUST REMAIN TOGETHER",
        "partitions": {
            "REPAIR_TRAIN": {
                "state": "CAPTURED AND FROZEN",
                "authorized_use": ["fit candidate models", "fit train-only scalers", "hard-negative mining", "training diagnostics"],
                "sites": SITES,
                "fault_instances": FAULTS,
            },
            "REPAIR_CALIBRATION": {
                "state": "LOCKED / NOT OPENED",
                "authorized_future_use": ["candidate selection", "confidence calibration", "OOD threshold selection", "stopping decision"],
                "gradient_updates": "PROHIBITED",
                "next_required_action": "SEPARATE ENHANCED-CAPTURE AUTHORIZATION",
            },
            "REPAIR_SITE_TEST": {
                "state": "LOCKED / NOT OPENED",
                "authorized_future_use": "ONE FINAL LOCKED EVALUATION AFTER MODEL SELECTION",
                "training_or_threshold_selection": "PROHIBITED",
            },
            "DEV_SITE_TEST": {"state": "CONSUMED", "access": "PROHIBITED"},
            "VALIDATION": {"state": "NOT ACCESSED", "access": "PROHIBITED"},
            "HOLDOUT": {"state": "NOT ACCESSED", "access": "PROHIBITED"},
        },
        "leakage_controls": [
            "physical site groups never cross partitions",
            "SA0 and SA1 for one site never cross partitions",
            "scalers and trainable parameters fit only on REPAIR_TRAIN",
            "candidate choice and thresholds use REPAIR_CALIBRATION once",
            "REPAIR_SITE_TEST is opened once after selection lock",
            "VALIDATION and HOLDOUT remain prohibited",
        ],
        "partition_membership_changes": "PROHIBITED",
        "partition_rows_opened_by_this_stage": 0,
    }

    candidate_rows = [
        {
            "candidate_id": "R31_EXACT_SIGNATURE_SET",
            "trainable": "NO",
            "family": "exact retrieval baseline",
            "shortlist": "all exact matches",
            "max_trainable_parameters": "0",
            "purpose": "mandatory ambiguity-aware baseline",
        },
        {
            "candidate_id": "R31_BLOCK_WEIGHTED_RETRIEVAL",
            "trainable": "YES",
            "family": "nonnegative block-weighted distance",
            "shortlist": "64",
            "max_trainable_parameters": "32",
            "purpose": "approximate retrieval when exact match is absent",
        },
        {
            "candidate_id": "R31_PAIRWISE_HISTGB_RERANKER",
            "trainable": "YES",
            "family": "histogram gradient boosting pairwise scorer",
            "shortlist": "64",
            "max_trainable_parameters": "implementation bounded; depth<=6, iterations<=200",
            "purpose": "nonlinear reranking of behavior-distance features",
        },
        {
            "candidate_id": "R31_FUSION_RERANKER",
            "trainable": "YES",
            "family": "weighted retrieval plus calibrated pairwise reranker",
            "shortlist": "64",
            "max_trainable_parameters": "candidate components only",
            "purpose": "final trainable candidate with exact-match precedence",
        },
    ]
    candidate_fields = ["candidate_id", "trainable", "family", "shortlist", "max_trainable_parameters", "purpose"]

    training_contract = {
        "training_contract_version": "CIRCUITSAGE-HMAC-V2.1-REPAIR-MODEL-TRAINING-12B3I-v1",
        "stage": STAGE,
        "status": "FROZEN",
        "training_authorization": "NOT GRANTED BY THIS CONTRACT",
        "candidate_ids": [row["candidate_id"] for row in candidate_rows],
        "training_partition": "REPAIR_TRAIN ONLY",
        "selection_partition": "REPAIR_CALIBRATION ONLY AFTER SEPARATE CAPTURE AUTHORIZATION",
        "final_evaluation_partition": "REPAIR_SITE_TEST ONLY AFTER SELECTION LOCK",
        "objective": {
            "retrieval": "multi-positive pairwise logistic ranking with all indistinguishable true sites treated as positives",
            "polarity": "binary SA0/SA1 auxiliary loss, reported separately",
            "ood": "distance calibration only; no synthetic claim of generalization",
        },
        "negative_sampling": {
            "random_negatives_per_positive": 32,
            "hard_negatives_per_positive": 32,
            "hard_negative_source": "nearest behavior signatures inside REPAIR_TRAIN",
            "same_site_opposite_polarity": "included where behavior differs",
        },
        "class_and_group_weighting": "site-balanced; SA0/SA1 balanced within site",
        "random_seeds": [12031901, 12031902, 12031903],
        "cpu_budget": {"threads": 1, "estimated_training_hours": "0.5-4", "maximum_candidate_runs": 9},
        "selection_order": [
            "reject false-unique or leakage violations",
            "maximize observable candidate-set coverage",
            "maximize observable MRR",
            "minimize mean candidate-set size",
            "prefer simpler nontrainable model within tolerance",
        ],
        "early_stopping": "REPAIR_TRAIN internal group folds only; REPAIR_CALIBRATION cannot drive gradient updates",
        "retraining_after_selection": "PROHIBITED",
        "threshold_change_after_selection": "PROHIBITED",
        "deterministic_replay": "REQUIRED",
        "model_serialization": "NPZ/JSON preferred; joblib allowed only with exact environment lock",
    }

    acceptance_contract = {
        "acceptance_contract_version": "CIRCUITSAGE-HMAC-V2.1-REPAIR-MODEL-ACCEPTANCE-12B3I-v1",
        "stage": STAGE,
        "status": "FROZEN",
        "measurement_invariants": {
            "fault_free_false_alarm_rate_max": 0.0,
            "combined_all_injected_detection_recall_min": 0.70,
            "detection_logic": "DETERMINISTIC GATE; NOT RETUNED BY MODEL",
        },
        "repair_calibration_targets": {
            "observable_candidate_set_coverage_min": 0.99,
            "unique_signature_top1_site_min": 0.95,
            "ambiguous_false_unique_rate_max": 0.0,
            "all_injected_exact_site_rate_min": 0.60,
            "mean_observable_candidate_sites_max": 5.0,
            "maximum_observable_candidate_sites_max": 64,
            "polarity_accuracy_given_correct_unique_site_min": 0.90,
        },
        "advancement_rule": {
            "required": "all safety and coverage criteria pass",
            "trainable_candidate": "must improve observable MRR by >=0.02 or reduce mean candidate size by >=10% without reducing coverage",
            "simplicity_fallback": "select R31_EXACT_SIGNATURE_SET when learned improvement is not established",
        },
        "uncertainty": "1000 site-group bootstrap replicates with 95% intervals",
        "independent_generalization": "NOT ESTABLISHED BY PASSING THIS CONTRACT",
        "production_readiness": "NOT ESTABLISHED",
    }

    environment = {
        "environment_version": "CIRCUITSAGE-HMAC-V2.1-REPAIR-MODEL-ENVIRONMENT-12B3I-v1",
        "stage": STAGE,
        "created_at": timestamp,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            "numpy": package_version("numpy"),
            "scipy": package_version("scipy"),
            "scikit-learn": package_version("scikit-learn"),
            "joblib": package_version("joblib"),
        },
        "training_calls": 0,
        "inference_calls": 0,
        "model_objects_deserialized": 0,
    }
    require(environment["packages"]["numpy"] != "NOT_INSTALLED", "NumPy environment")
    require(environment["packages"]["scikit-learn"] != "NOT_INSTALLED", "scikit-learn environment")

    frozen_write(ARCHITECTURE, canonical_json(architecture))
    frozen_write(FEATURE_CONTRACT, canonical_json(feature_contract))
    frozen_write(PARTITION_CONTRACT, canonical_json(partition_contract))
    frozen_write(TRAINING_CONTRACT, canonical_json(training_contract))
    frozen_write(ACCEPTANCE_CONTRACT, canonical_json(acceptance_contract))
    frozen_write(CANDIDATE_GRID, csv_payload(candidate_rows, candidate_fields))
    frozen_write(ENVIRONMENT, canonical_json(environment))

    report = f"""# CircuitSage-HMAC V2.1 Repair-Model Contract — Stage 12B-3I

## Purpose

This contract defines a small CPU-feasible localization repair around the
frozen `{SELECTED_MEASUREMENT}` behavior measurements. Detection remains a
deterministic anomaly gate. The trainable portion is limited to approximate
retrieval and reranking when an exact catalog signature is unavailable.

## Scientific guardrails

- Fault identity, site, selector and stuck value are forbidden query inputs.
- Exact matches take precedence over learned scores.
- Physically indistinguishable sites remain an explicit candidate set.
- A learned model cannot declare one site unique when the complete signatures
  of multiple sites are identical.
- REPAIR_TRAIN is the only fitting partition.
- REPAIR_CALIBRATION is reserved for selection and confidence calibration.
- REPAIR_SITE_TEST remains sealed for one final evaluation.
- Original DEV_SITE_TEST, VALIDATION and HOLDOUT remain prohibited.

## Current frozen measurement

- Detection recall: {DETECTION_RECALL:.8f}
- All-injected exact-site rate: {EXACT_SITE_RATE:.8f}
- Mean / maximum observable candidates: {MEAN_CANDIDATES:.4f} / {MAX_CANDIDATES}
- False alarms: 0

These are measurement-consistency results, not trained-model accuracy.

## Next action

Stage 12B-3J may create a bounded authorization for capturing the same enhanced
measurement on the already-frozen REPAIR_CALIBRATION partition. It may not
train a model or open REPAIR_SITE_TEST.
""".encode()
    frozen_write(REPORT, report)

    contract_outputs = [
        ARCHITECTURE, FEATURE_CONTRACT, PARTITION_CONTRACT, TRAINING_CONTRACT,
        ACCEPTANCE_CONTRACT, CANDIDATE_GRID, ENVIRONMENT, REPORT,
    ]
    manifest = {
        "manifest_version": "CIRCUITSAGE-HMAC-V2.1-REPAIR-MODEL-CONTRACT-MANIFEST-12B3I-v1",
        "stage": STAGE,
        "status": "PASS",
        "created_at": timestamp,
        "frozen_inputs": {rel(path): record(path) for path in PINNED},
        "outputs": {rel(path): record(path) for path in contract_outputs},
        "candidate_models": len(candidate_rows),
        "trainable_candidates": sum(row["trainable"] == "YES" for row in candidate_rows),
        "training_calls": 0,
        "inference_calls": 0,
        "scaler_fits": 0,
        "repair_calibration_access": 0,
        "repair_site_test_access": 0,
        "dev_site_test_access": 0,
        "validation_access": 0,
        "holdout_access": 0,
    }
    frozen_write(MANIFEST, canonical_json(manifest))
    audit = {
        "audit_version": "CIRCUITSAGE-HMAC-V2.1-REPAIR-MODEL-CONTRACT-FREEZE-12B3I-v1",
        "stage": STAGE,
        "status": "PASS",
        "architecture_status": "FROZEN",
        "feature_contract_status": "FROZEN",
        "data_partition_contract_status": "FROZEN",
        "training_contract_status": "FROZEN",
        "acceptance_contract_status": "FROZEN",
        "model_family": "EXACT SIGNATURE RETRIEVAL + BLOCK-WEIGHTED RETRIEVAL + PAIRWISE RERANKING",
        "selected_measurement": SELECTED_MEASUREMENT,
        "candidate_models": len(candidate_rows),
        "trainable_candidates": sum(row["trainable"] == "YES" for row in candidate_rows),
        "query_fault_identity": "PROHIBITED",
        "ambiguity_preservation": "MANDATORY",
        "training": "NOT AUTHORIZED / NOT STARTED",
        "repair_calibration": "LOCKED / NOT OPENED",
        "repair_site_test": "LOCKED / NOT OPENED",
        "dev_site_test_validation_holdout_access": [0, 0, 0],
        "v1_v2_core_v2_1_modified": [False, False, False],
        "architecture": record(ARCHITECTURE),
        "feature_contract": record(FEATURE_CONTRACT),
        "partition_contract": record(PARTITION_CONTRACT),
        "training_contract": record(TRAINING_CONTRACT),
        "acceptance_contract": record(ACCEPTANCE_CONTRACT),
        "candidate_grid": record(CANDIDATE_GRID),
        "environment": record(ENVIRONMENT),
        "report": record(REPORT),
        "manifest": record(MANIFEST),
        "next_gate": "STAGE 12B-3J — ENHANCED REPAIR_CALIBRATION CAPTURE AUTHORIZATION FREEZE",
    }
    frozen_write(AUDIT, canonical_json(audit))

    for path in (ARCHITECTURE, FEATURE_CONTRACT, PARTITION_CONTRACT, TRAINING_CONTRACT,
                 ACCEPTANCE_CONTRACT, ENVIRONMENT, MANIFEST, AUDIT):
        require(canonical_json(load_json(path)) == path.read_bytes(), f"canonical output replay: {rel(path)}")
    require(CANDIDATE_GRID.read_bytes() == csv_payload(candidate_rows, candidate_fields), "candidate-grid replay")
    require(REPORT.read_bytes() == report, "report replay")
    for path, expected in PINNED.items():
        require(sha256(path) == expected, f"frozen input modified: {rel(path)}")

    print("\nSTAGE 12B-3I — REPAIR-MODEL ARCHITECTURE, DATA-PARTITION, AND TRAINING-CONTRACT FREEZE")
    print(f"{'Status':<67}: PASS")
    print(f"{'Architecture / feature contract':<67}: FROZEN / FROZEN")
    print(f"{'Partition / training / acceptance contracts':<67}: FROZEN / FROZEN / FROZEN")
    print(f"{'Model family':<67}: EXACT RETRIEVAL + WEIGHTED RETRIEVAL + PAIRWISE RERANKING")
    print(f"{'Candidate models / trainable':<67}: {len(candidate_rows)} / {sum(row['trainable'] == 'YES' for row in candidate_rows)}")
    print(f"{'Selected measurement':<67}: {SELECTED_MEASUREMENT}")
    print(f"{'Query fault identity':<67}: PROHIBITED")
    print(f"{'Ambiguity preservation':<67}: MANDATORY")
    print(f"{'Training':<67}: NOT AUTHORIZED / NOT STARTED")
    print(f"{'REPAIR_CALIBRATION / REPAIR_SITE_TEST':<67}: LOCKED / LOCKED")
    print(f"{'DEV_SITE_TEST / VALIDATION / HOLDOUT access':<67}: 0 / 0 / 0")
    print(f"{'Architecture':<67}: {ARCHITECTURE}")
    print(f"{'Architecture SHA':<67}: {sha256(ARCHITECTURE)}")
    print(f"{'Partition contract':<67}: {PARTITION_CONTRACT}")
    print(f"{'Partition contract SHA':<67}: {sha256(PARTITION_CONTRACT)}")
    print(f"{'Training contract':<67}: {TRAINING_CONTRACT}")
    print(f"{'Training contract SHA':<67}: {sha256(TRAINING_CONTRACT)}")
    print(f"{'Acceptance contract':<67}: {ACCEPTANCE_CONTRACT}")
    print(f"{'Acceptance contract SHA':<67}: {sha256(ACCEPTANCE_CONTRACT)}")
    print(f"{'Candidate grid':<67}: {CANDIDATE_GRID}")
    print(f"{'Candidate grid SHA':<67}: {sha256(CANDIDATE_GRID)}")
    print(f"{'Manifest':<67}: {MANIFEST}")
    print(f"{'Manifest SHA':<67}: {sha256(MANIFEST)}")
    print(f"{'Audit':<67}: {AUDIT}")
    print(f"{'Audit SHA':<67}: {sha256(AUDIT)}")
    print(f"{'Next gate':<67}: STAGE 12B-3J — ENHANCED REPAIR_CALIBRATION CAPTURE AUTHORIZATION FREEZE")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    arguments = parser.parse_args()
    if arguments.self_test:
        self_test()
    else:
        main()
