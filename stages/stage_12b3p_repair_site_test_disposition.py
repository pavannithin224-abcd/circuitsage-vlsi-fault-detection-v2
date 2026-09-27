#!/usr/bin/env python3
"""Stage 12B-3P: REPAIR_SITE_TEST disposition and V2.1 improvement freeze.

This disposition-only stage verifies the immutable Stage 12B-3O capture and
locked evaluation, records every met and unmet criterion without changing any
model, threshold, candidate, dataset, or prediction, and freezes the V2.1
improvement experiment as a completed closed-catalog research result.

It does not train, deserialize, or run a model.  It does not reopen
REPAIR_SITE_TEST or DEV_SITE_TEST and does not access VALIDATION or HOLDOUT.
Only creation of the separate Stage 12C-1A generalization contract is
authorized; dataset construction and training remain unauthorized.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STAGE = "12B-3P"
ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "config/v2_1_improvement"
RESULT = ROOT / "results/circuitsage_hmac_v2_12b3"
SOURCE_RESULT = RESULT / "repair_site_test_capture_execution_12b3o"
WORK = RESULT / "repair_site_test_disposition_12b3p"

SOURCE_3O = ROOT / "stage_12b3o_repair_site_test_capture_evaluate.py"
MASTER_LOG = RESULT / "stage_12b3o_20260917_224701.log"
RESOURCE_LOG = RESULT / "stage_12b3o_resources_20260917_224701.log"
CHECKPOINT = SOURCE_RESULT / "circuitsage_hmac_v2_1_repair_site_test_capture_checkpoint_12b3o.json"
FEATURES = SOURCE_RESULT / "circuitsage_hmac_v2_1_repair_site_test_features_12b3o.npz"
TARGETS = SOURCE_RESULT / "circuitsage_hmac_v2_1_repair_site_test_targets_12b3o.npz"
PREDICTIONS = SOURCE_RESULT / "circuitsage_hmac_v2_1_repair_site_test_predictions_12b3o.npz"
CANDIDATES = SOURCE_RESULT / "circuitsage_hmac_v2_1_repair_site_test_candidate_sets_12b3o.csv"
PREDICTION_COMMITMENT = SOURCE_RESULT / "circuitsage_hmac_v2_1_repair_site_test_prediction_commitment_12b3o.json"
SIGNATURES = SOURCE_RESULT / "circuitsage_hmac_v2_1_repair_site_test_signature_summary_12b3o.csv"
METRICS = SOURCE_RESULT / "circuitsage_hmac_v2_1_repair_site_test_metrics_12b3o.json"
BOOTSTRAP = SOURCE_RESULT / "circuitsage_hmac_v2_1_repair_site_test_site_bootstrap_12b3o.csv"
SCHEMA = SOURCE_RESULT / "circuitsage_hmac_v2_1_repair_site_test_dataset_schema_12b3o.json"
MANIFEST_3O = RESULT / "circuitsage_hmac_v2_1_repair_site_test_evaluation_manifest_12b3o.json"
AUDIT_3O = RESULT / "circuitsage_hmac_v2_1_repair_site_test_capture_evaluation_freeze_12b3o.json"

ACCEPTANCE = CONFIG / "circuitsage_hmac_v2_1_repair_model_acceptance_contract_12b3i.json"
MODEL_DIR = RESULT / "repair_model_training_12b3m"
SELECTED_MODEL = MODEL_DIR / "circuitsage_hmac_v2_1_selected_repair_model_12b3m.npz"
SELECTION_LOCK = MODEL_DIR / "circuitsage_hmac_v2_1_repair_model_selection_lock_12b3m.json"

BASELINE_METRICS = (
    ROOT / "results/circuitsage_hmac_v2_12b1/v2_1_evaluation_12b1f"
    / "circuitsage_hmac_v2_1_site_test_metrics_12b1f.json"
)

POLICY = CONFIG / "circuitsage_hmac_v2_1_improvement_final_disposition_policy_12b3p.json"
FINAL_LOCK = WORK / "circuitsage_hmac_v2_1_improvement_final_lock_12b3p.json"
COMPARISON = WORK / "circuitsage_hmac_v2_1_improvement_comparison_12b3p.json"
REGISTRY_CSV = WORK / "circuitsage_hmac_v2_1_improvement_capability_registry_12b3p.csv"
REGISTRY_JSON = WORK / "circuitsage_hmac_v2_1_improvement_capability_registry_12b3p.json"
REPORT = WORK / "circuitsage_hmac_v2_1_improvement_final_report_12b3p.md"
MANIFEST = RESULT / "circuitsage_hmac_v2_1_improvement_disposition_manifest_12b3p.json"
AUDIT = RESULT / "circuitsage_hmac_v2_1_improvement_final_freeze_12b3p.json"

PINNED = {
    SOURCE_3O: "0c49b53eb1ed849f573ca227c3ec82f5bf14d6e8a3ad322f742f93e0ce737843",
    MASTER_LOG: "9587d4a71ad355e98a75d2269d5d7988519b25c773f26c34e3a6c0d4ad3d60a5",
    RESOURCE_LOG: "8413de5d4968753a9d45bd63f4e6c1804ee65e409400f34c70ff5b53865cbb94",
    CHECKPOINT: "b664986873a4a9e48757dc1da3fb4952a509dcacc5ce3b254f121945a865892a",
    FEATURES: "ac5a9a8982c3ab0de8b9b534954e2087a8cb18d2235bd7b25d60d3e73ab98b93",
    TARGETS: "dfaeb6e0d5a371275afa785367a9b6c6128eaffdf600051f3f100475d4ace9d4",
    PREDICTIONS: "06bfc5e2ff3a8c0e2ec178a7f53c5789cb1d855a9b6fffcde74637b394d3f54b",
    CANDIDATES: "b2d1b428d3175e76788ccbcc5819c5742cc11f0e0cde1b8cfa58fc29c204e908",
    PREDICTION_COMMITMENT: "d62385b1d89db1911daabdff6d725788e2c00f68c5f4975d419a2b953f0c5871",
    SIGNATURES: "bb7507068f8dc5feadf54ea416e422e25ebcacf944d906c1fc74fec5aa0bbf48",
    METRICS: "c67debe4bdba2f55a68a34c7df1912ae341bb091ddf7b3ae277e08f4fb06cf37",
    BOOTSTRAP: "d31fb6d4b158a398b031f60b794c40a86413fd66f2fdcb89957bf285cf2cf84a",
    SCHEMA: "90172fe8e0b61387dd6763653bf39246f0da8b0961a9103add5963fde61b1f73",
    MANIFEST_3O: "e4bd3f0e7a5c85946b9b2ff61e851c6b26bc0e1c077582908ca0c1f94248d482",
    AUDIT_3O: "49ab57043bfdc4166c99e1276705c84a5ef0f80910e054843a5b3a9196c1ee7c",
    ACCEPTANCE: "ca95bc7fc7587e995f48b0efa03390734192fd251cd82ebd25387c5663d1e0d2",
    SELECTED_MODEL: "2f1d35c3f2b71f975859a99238d07e1e0d8620f02d79f3808ca1e5c2ccf6f39a",
    SELECTION_LOCK: "7b9cb7d6c4dd9e8123fdfc3b327bcd6213b827d6338cdc7bd0326f739704bf53",
    BASELINE_METRICS: "0c7de49c735a0068c5471344e58748f652f4fe5042f9370e96f88aa7bd2f5c10",
}

SELECTED_MEASUREMENT = "EM_TESTPOINT_4X64_T16"
SELECTED_MODEL_ID = "R31_EXACT_SIGNATURE_SET"
SITES = 2398
FAULTS = 4796
VECTORS = 96
BATCHES = 45
BASELINE_RECORDS = 4320
ENABLED_RECORDS = 460416
TOTAL_RECORDS = 464736
FUTURE_BRAND = "Faultiva 1.0 — Hybrid VLSI Fault Intelligence"


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


def close(left: float, right: float, tolerance: float = 5e-9) -> bool:
    return abs(left - right) <= tolerance


def verify_manifest_record(item: Any, expected_path: Path, label: str) -> None:
    require(isinstance(item, dict), f"missing manifest record: {label}")
    require(item.get("path") == rel(expected_path), f"manifest path: {label}")
    require(item.get("sha256") == sha256(expected_path), f"manifest SHA: {label}")
    require(int(item.get("bytes", -1)) == expected_path.stat().st_size, f"manifest size: {label}")


def verify_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    print("STAGE 12B-3P — REPAIR_SITE_TEST RESULT DISPOSITION AND V2.1 IMPROVEMENT FREEZE")
    print("FROZEN INPUT VERIFICATION")
    for path, expected in PINNED.items():
        require(path.is_file(), f"missing frozen input: {rel(path)}")
        require(sha256(path) == expected, f"frozen input SHA: {rel(path)}")
        print(f"  {path.name:<104}: OK")

    metrics = load_json(METRICS)
    manifest = load_json(MANIFEST_3O)
    audit = load_json(AUDIT_3O)
    acceptance = load_json(ACCEPTANCE)
    selection = load_json(SELECTION_LOCK)
    commitment = load_json(PREDICTION_COMMITMENT)
    baseline_bundle = load_json(BASELINE_METRICS)
    baseline = baseline_bundle.get("metrics")
    require(isinstance(baseline, dict), "baseline metric object")

    require(metrics.get("stage") == "12B-3O" and metrics.get("status") == "FROZEN", "3O metrics status")
    require(metrics.get("selected_measurement") == SELECTED_MEASUREMENT, "selected measurement")
    require(metrics.get("selected_model") == SELECTED_MODEL_ID, "selected model")
    require([metrics.get("sites"), metrics.get("fault_instances"), metrics.get("vectors")] == [SITES, FAULTS, VECTORS], "metric dimensions")
    require(metrics.get("locked_evaluation_acceptance") == "NOT_MET", "locked acceptance must remain NOT_MET")
    require(metrics.get("prediction_commitment_before_truth") == "PASS / VERIFIED", "prediction commitment")
    require([metrics.get("training_calls"), metrics.get("threshold_changes"), metrics.get("candidate_reselections")] == [0, 0, 0], "no post-test adaptation")
    require(metrics.get("independent_circuit_generalization") == "NOT ESTABLISHED", "generalization boundary")

    require(manifest.get("status") == "PASS", "3O manifest status")
    require([manifest.get("canonical_batches"), manifest.get("sites"), manifest.get("fault_instances"), manifest.get("vectors")] == [BATCHES, SITES, FAULTS, VECTORS], "manifest dimensions")
    require([manifest.get("baseline_records"), manifest.get("enabled_records"), manifest.get("total_records")] == [BASELINE_RECORDS, ENABLED_RECORDS, TOTAL_RECORDS], "record counts")
    require(manifest.get("missing_duplicate_unknown_baseline_failures") == [0, 0, 0, 0], "dataset integrity counters")
    require(manifest.get("prediction_commitment_before_truth") is True, "manifest prediction commitment")
    require([manifest.get("training_calls"), manifest.get("threshold_changes"), manifest.get("candidate_reselections")] == [0, 0, 0], "manifest adaptation counters")
    require([manifest.get("dev_site_test_access"), manifest.get("validation_access"), manifest.get("holdout_access")] == [0, 0, 0], "protected access counters")
    outputs = manifest.get("outputs")
    require(isinstance(outputs, dict), "3O output registry")
    for path in (FEATURES, TARGETS, SIGNATURES, PREDICTIONS, CANDIDATES, PREDICTION_COMMITMENT, METRICS, BOOTSTRAP, SCHEMA):
        verify_manifest_record(outputs.get(rel(path)), path, path.name)

    require(audit.get("status") == "PASS", "3O audit status")
    require(audit.get("capture_evaluation") == "COMPLETED / FROZEN", "capture/evaluation state")
    require(audit.get("dataset_status") == "FROZEN", "dataset freeze")
    require(audit.get("selected_measurement_model") == [SELECTED_MEASUREMENT, SELECTED_MODEL_ID], "audit selection")
    require(audit.get("locked_evaluation_acceptance") == "NOT_MET", "audit locked acceptance")
    require(audit.get("repair_site_test") == "CONSUMED / FROZEN / DO NOT REOPEN", "test consumption lock")
    require(audit.get("dev_site_test") == "CONSUMED / NOT REOPENED", "development test lock")
    require(audit.get("validation_holdout_access") == [0, 0], "audit protected access")
    require(audit.get("independent_generalization") == "NOT ESTABLISHED", "audit generalization boundary")

    require(acceptance.get("status") == "FROZEN", "acceptance contract status")
    require(selection.get("selected_candidate") == SELECTED_MODEL_ID, "selection lock")
    require(selection.get("retraining_after_selection") == "PROHIBITED", "retraining lock")
    require(selection.get("threshold_change_after_selection") == "PROHIBITED", "threshold lock")
    require(commitment.get("status") == "FROZEN BEFORE SCORING", "commitment status")
    verify_manifest_record(commitment.get("predictions"), PREDICTIONS, "committed predictions")
    verify_manifest_record(commitment.get("candidate_sets"), CANDIDATES, "committed candidate sets")

    targets = metrics.get("acceptance_targets")
    criteria = metrics.get("criteria")
    require(isinstance(targets, dict) and isinstance(criteria, dict), "acceptance targets/criteria")
    expected_criteria = {
        "fault_free_false_alarm_rate": "PASS",
        "combined_detection_recall": "NOT_MET",
        "candidate_set_coverage": "PASS",
        "unique_signature_top1_site": "PASS",
        "ambiguous_false_unique_rate": "PASS",
        "all_injected_exact_site_rate": "NOT_MET",
        "mean_observable_candidate_sites": "PASS",
        "maximum_observable_candidate_sites": "NOT_MET",
        "polarity_accuracy": "PASS",
    }
    require(criteria == expected_criteria, "locked criterion disposition")
    require(close(float(targets["combined_detection_recall_min"]), 0.70), "detection target")
    require(close(float(targets["all_injected_exact_site_rate_min"]), 0.60), "exact-site target")
    require(close(float(targets["mean_observable_candidate_sites_max"]), 5.0), "mean-candidate target")
    require(int(targets["maximum_observable_candidate_sites_max"]) == 64, "maximum-candidate target")

    require(baseline_bundle.get("status") == "PASS", "baseline status")
    require(baseline.get("selected_candidate_id") == "V21_EXACT_SIGNATURE_SET", "baseline candidate")
    require(baseline.get("closed_catalog_consistency_only") is True, "baseline scope")

    for path in (METRICS, MANIFEST_3O, AUDIT_3O, PREDICTION_COMMITMENT, ACCEPTANCE, SELECTION_LOCK, BASELINE_METRICS):
        require(canonical_json(load_json(path)) == path.read_bytes(), f"canonical JSON replay: {path.name}")

    log_text = MASTER_LOG.read_text(encoding="utf-8", errors="replace")
    require("Status" in log_text and "PASS" in log_text, "3O PASS log evidence")
    require("Locked evaluation acceptance" in log_text and "NOT_MET" in log_text, "3O NOT_MET log evidence")
    print("  Locked metrics, commitments, criteria, access counters and predecessor evidence       : PASS")
    return metrics, manifest, audit, baseline


def self_test() -> None:
    require(TOTAL_RECORDS == BASELINE_RECORDS + ENABLED_RECORDS, "record arithmetic")
    require(close(3317 / 4796, 0.6916180150125104), "detection arithmetic")
    require(close(0.70 - 3317 / 4796, 0.0083819849874896), "detection gap arithmetic")
    require(close(0.548790658882402, 2632 / 4796), "exact-site arithmetic")
    sample = {"z": 1, "a": [2, 3]}
    require(canonical_json(sample) == canonical_json(json.loads(canonical_json(sample))), "canonical JSON")
    print("Stage 12B-3P self-test: PASS")


def main() -> None:
    require(ROOT.name == "vlsi_fault_detection_v2", f"run from project root, not {ROOT}")
    outputs = (POLICY, FINAL_LOCK, COMPARISON, REGISTRY_CSV, REGISTRY_JSON, REPORT, MANIFEST, AUDIT)
    for path in outputs:
        require(not path.exists(), f"Stage {STAGE} output already exists: {rel(path)}")

    metrics, source_manifest, source_audit, baseline = verify_inputs()
    timestamp = now()

    detection = float(metrics["combined_all_injected_detection_recall"])
    exact = float(metrics["all_injected_exact_site_rate"])
    coverage = float(metrics["observable_candidate_set_coverage"])
    unique_top1 = float(metrics["unique_signature_top1_site"])
    mean_candidates = float(metrics["mean_observable_candidate_sites"])
    max_candidates = int(metrics["maximum_observable_candidate_sites"])
    polarity = float(metrics["polarity_accuracy_given_correct_unique_site"])
    normal_compatible = int(metrics["normal_compatible_faults"])
    rescued = int(metrics["probe_rescued_faults"])
    criteria = dict(metrics["criteria"])
    unmet = sorted(name for name, state in criteria.items() if state != "PASS")
    require(unmet == [
        "all_injected_exact_site_rate",
        "combined_detection_recall",
        "maximum_observable_candidate_sites",
    ], "unmet criterion set")

    baseline_detection = float(baseline["all_injected_detection_recall"])
    baseline_exact = float(baseline["all_injected_exact_site_rate"])
    baseline_mean = float(baseline["mean_candidate_set_size_observable"])
    baseline_max = int(baseline["maximum_candidate_set_size_observable"])
    comparison = {
        "comparison_version": "CIRCUITSAGE-HMAC-V2.1-IMPROVEMENT-COMPARISON-12B3P-v1",
        "stage": STAGE,
        "status": "FROZEN",
        "comparison_scope": "CLOSED-CATALOG PHYSICAL-SITE-HELD-OUT RESEARCH RESULTS; COHORTS DIFFER",
        "original_v2_1_reference": {
            "stage": "12B-1F",
            "all_injected_detection_recall": baseline_detection,
            "all_injected_exact_site_rate": baseline_exact,
            "mean_observable_candidate_sites": baseline_mean,
            "maximum_observable_candidate_sites": baseline_max,
        },
        "improved_v2_1_locked_result": {
            "stage": "12B-3O",
            "combined_all_injected_detection_recall": detection,
            "all_injected_exact_site_rate": exact,
            "mean_observable_candidate_sites": mean_candidates,
            "maximum_observable_candidate_sites": max_candidates,
        },
        "absolute_changes": {
            "detection_recall": detection - baseline_detection,
            "exact_site_rate": exact - baseline_exact,
            "mean_candidate_sites": mean_candidates - baseline_mean,
            "maximum_candidate_sites": max_candidates - baseline_max,
        },
        "interpretation": "SUBSTANTIAL IMPROVEMENT OBSERVED; PRE-REGISTERED FINAL ACCEPTANCE STILL NOT MET",
        "not_a_claim": "NOT A PAIRED SAME-COHORT SUPERIORITY TEST AND NOT INDEPENDENT GENERALIZATION",
    }
    frozen_write(COMPARISON, canonical_json(comparison))

    policy = {
        "policy_version": "CIRCUITSAGE-HMAC-V2.1-IMPROVEMENT-FINAL-DISPOSITION-12B3P-v1",
        "stage": STAGE,
        "status": "FROZEN",
        "created_at": timestamp,
        "lifecycle": "COMPLETED AND FROZEN CLOSED-CATALOG V2.1 IMPROVEMENT EXPERIMENT",
        "source_stage": "12B-3O",
        "source_status": "PASS / FROZEN",
        "locked_evaluation_acceptance": "NOT_MET / PRESERVED",
        "selected_measurement": SELECTED_MEASUREMENT,
        "selected_model": SELECTED_MODEL_ID,
        "disposition": "RETAIN AS RESEARCH BASELINE; DO NOT CLAIM FINAL ACCEPTANCE",
        "met_criteria": sorted(name for name, state in criteria.items() if state == "PASS"),
        "unmet_criteria": unmet,
        "no_post_test_adaptation": True,
        "repair_site_test": "CONSUMED / FROZEN / DO NOT REOPEN",
        "dev_site_test": "CONSUMED / DO NOT REOPEN",
        "validation": "NOT ACCESSED",
        "holdout": "NOT ACCESSED",
        "independent_generalization": "NOT ESTABLISHED",
        "production_or_silicon_readiness": "NOT ESTABLISHED",
        "allowed_next_action": "CREATE STAGE 12C-1A GENERALIZATION CONTRACT ONLY",
        "v2_2_dataset_or_training": "NOT AUTHORIZED BY THIS POLICY",
        "future_combined_model_brand": FUTURE_BRAND,
        "brand_status": "RESERVED FOR THE COMPLETED V1+V2 HYBRID RELEASE; NOT APPLIED TO THIS COMPONENT",
    }
    frozen_write(POLICY, canonical_json(policy))

    final_lock = {
        "lock_version": "CIRCUITSAGE-HMAC-V2.1-IMPROVEMENT-FINAL-LOCK-12B3P-v1",
        "stage": STAGE,
        "status": "FROZEN",
        "selected_measurement": SELECTED_MEASUREMENT,
        "selected_model": SELECTED_MODEL_ID,
        "model": record(SELECTED_MODEL),
        "features": record(FEATURES),
        "targets": record(TARGETS),
        "predictions": record(PREDICTIONS),
        "candidate_sets": record(CANDIDATES),
        "prediction_commitment": record(PREDICTION_COMMITMENT),
        "metrics": record(METRICS),
        "schema": record(SCHEMA),
        "source_manifest": record(MANIFEST_3O),
        "source_audit": record(AUDIT_3O),
        "locked_evaluation_acceptance": "NOT_MET",
        "retraining_threshold_change_candidate_reselection": "PROHIBITED / PROHIBITED / PROHIBITED",
    }
    frozen_write(FINAL_LOCK, canonical_json(final_lock))

    rows = [
        {"capability": "combined_all_injected_detection", "value": f"{detection:.8f}", "target": ">=0.70000000", "status": "NOT_MET", "scope": "locked REPAIR_SITE_TEST"},
        {"capability": "observable_candidate_set_coverage", "value": f"{coverage:.8f}", "target": ">=0.99000000", "status": "PASS", "scope": "observable faults"},
        {"capability": "unique_signature_top1_site", "value": f"{unique_top1:.8f}", "target": ">=0.95000000", "status": "PASS", "scope": "unique signatures"},
        {"capability": "all_injected_exact_site", "value": f"{exact:.8f}", "target": ">=0.60000000", "status": "NOT_MET", "scope": "all injected faults"},
        {"capability": "mean_observable_candidate_sites", "value": f"{mean_candidates:.4f}", "target": "<=5.0000", "status": "PASS", "scope": "observable faults"},
        {"capability": "maximum_observable_candidate_sites", "value": str(max_candidates), "target": "<=64", "status": "NOT_MET", "scope": "observable faults"},
        {"capability": "polarity_given_correct_unique_site", "value": f"{polarity:.8f}", "target": ">=0.90000000", "status": "PASS", "scope": "correct unique sites"},
        {"capability": "fault_free_false_alarms", "value": "0", "target": "0", "status": "PASS", "scope": "frozen vectors and batches"},
        {"capability": "normal_compatible_faults", "value": str(normal_compatible), "target": "informational", "status": "LIMITATION", "scope": "current measurement"},
        {"capability": "independent_generalization", "value": "NOT_ESTABLISHED", "target": "separate V2.2 study", "status": "LIMITATION", "scope": "unseen circuits/signatures"},
    ]
    fields = ["capability", "value", "target", "status", "scope"]
    registry_payload = csv_bytes(rows, fields)
    frozen_write(REGISTRY_CSV, registry_payload)
    frozen_write(REGISTRY_JSON, canonical_json({"stage": STAGE, "status": "FROZEN", "rows": rows}))

    detection_gap = 0.70 - detection
    exact_gap = 0.60 - exact
    max_excess = max_candidates - 64
    report = f"""# CircuitSage-HMAC V2.1 Improvement Final Disposition — Stage 12B-3P

## Decision

The V2.1 improvement experiment is **completed and frozen**, but its locked
final acceptance is **NOT_MET**. The Stage 12B-3O result is retained as a
useful closed-catalog research baseline. The consumed REPAIR_SITE_TEST must not
be reopened for tuning, retraining, threshold changes, or candidate reselection.

## Frozen final result

- Measurement / model: `{SELECTED_MEASUREMENT}` / `{SELECTED_MODEL_ID}`
- Cohort: {SITES:,} sites, {FAULTS:,} injected SA0/SA1 faults, {VECTORS} vectors
- Combined detection recall: {detection:.8%} (target 70%; gap {detection_gap:.8%})
- All-injected exact-site rate: {exact:.8%} (target 60%; gap {exact_gap:.8%})
- Observable candidate-set coverage: {coverage:.8%}
- Unique-signature top-1 site accuracy: {unique_top1:.8%}
- Mean / maximum observable candidate sites: {mean_candidates:.4f} / {max_candidates}
- Maximum-candidate excess over target: {max_excess}
- Probe-rescued faults: {rescued:,}
- Fault-free false alarms: 0

## What improved

Relative to the frozen Stage 12B-1F reference, detection recall increased from
{baseline_detection:.8%} to {detection:.8%}, exact-site rate increased from
{baseline_exact:.8%} to {exact:.8%}, and mean observable candidates decreased
from {baseline_mean:.4f} to {mean_candidates:.4f}. The cohorts differ, so this
is a bounded engineering comparison rather than a paired superiority test.

## Unmet criteria

1. Combined detection recall: {detection:.8f} < 0.70000000
2. All-injected exact-site rate: {exact:.8f} < 0.60000000
3. Maximum observable candidate sites: {max_candidates} > 64

## Scientific boundary

The result measures closed-catalog, physical-site-held-out consistency for the
frozen HMAC SA0/SA1 experiment. It does not establish independent-circuit
generalization, unseen fault-family performance, physical-silicon readiness,
or production readiness.

## Next gate

Stage 12C-1A may define a separate V2.2 generalization architecture,
data-partition plan and acceptance contract. This disposition does not
authorize V2.2 dataset construction or training. The name
**{FUTURE_BRAND}** remains reserved for the eventual completed V1+V2 hybrid
release and is not assigned to this V2.1 component.
""".encode()
    frozen_write(REPORT, report)

    output_paths = [POLICY, FINAL_LOCK, COMPARISON, REGISTRY_CSV, REGISTRY_JSON, REPORT]
    manifest = {
        "manifest_version": "CIRCUITSAGE-HMAC-V2.1-IMPROVEMENT-DISPOSITION-MANIFEST-12B3P-v1",
        "stage": STAGE,
        "status": "PASS",
        "created_at": timestamp,
        "frozen_inputs": {rel(path): record(path) for path in PINNED},
        "outputs": {rel(path): record(path) for path in output_paths},
        "locked_evaluation_acceptance": "NOT_MET / PRESERVED",
        "training_calls": 0,
        "inference_calls": 0,
        "model_objects_deserialized": 0,
        "threshold_changes": 0,
        "candidate_reselections": 0,
        "repair_site_test_access": 0,
        "dev_site_test_access": 0,
        "validation_access": 0,
        "holdout_access": 0,
    }
    frozen_write(MANIFEST, canonical_json(manifest))

    audit = {
        "audit_version": "CIRCUITSAGE-HMAC-V2.1-IMPROVEMENT-FINAL-FREEZE-12B3P-v1",
        "stage": STAGE,
        "status": "PASS",
        "disposition_status": "FROZEN",
        "v2_1_improvement_lifecycle": "COMPLETED AND FROZEN CLOSED-CATALOG RESEARCH EXPERIMENT",
        "selected_measurement_model": [SELECTED_MEASUREMENT, SELECTED_MODEL_ID],
        "locked_evaluation_acceptance": "NOT_MET / PRESERVED",
        "combined_detection_recall": detection,
        "detection_target_gap": detection_gap,
        "all_injected_exact_site_rate": exact,
        "exact_site_target_gap": exact_gap,
        "observable_candidate_set_coverage": coverage,
        "unique_signature_top1_site": unique_top1,
        "mean_maximum_observable_candidate_sites": [mean_candidates, max_candidates],
        "fault_free_false_alarms": 0,
        "unmet_criteria": unmet,
        "prediction_commitment_before_truth": "PASS / VERIFIED",
        "repair_site_test": "CONSUMED / FROZEN / DO NOT REOPEN",
        "training_threshold_change_candidate_reselection": [0, 0, 0],
        "dev_site_test_validation_holdout_access": [0, 0, 0],
        "independent_generalization": "NOT ESTABLISHED",
        "production_or_silicon_readiness": "NOT ESTABLISHED",
        "v1_v2_core_v2_1_modified": [False, False, False],
        "future_combined_model_brand": FUTURE_BRAND,
        "policy": record(POLICY),
        "final_lock": record(FINAL_LOCK),
        "comparison": record(COMPARISON),
        "report": record(REPORT),
        "manifest": record(MANIFEST),
        "next_gate": "STAGE 12C-1A — V2.2 GENERALIZATION ARCHITECTURE, DATA-PARTITION, AND ACCEPTANCE-CONTRACT FREEZE",
        "next_gate_authorization": "CONTRACT CREATION ONLY; DATASET AND TRAINING NOT AUTHORIZED",
    }
    frozen_write(AUDIT, canonical_json(audit))

    for path in (POLICY, FINAL_LOCK, COMPARISON, REGISTRY_JSON, MANIFEST, AUDIT):
        require(canonical_json(load_json(path)) == path.read_bytes(), f"canonical output replay: {path.name}")
    require(REGISTRY_CSV.read_bytes() == registry_payload, "registry CSV replay")
    require(REPORT.read_bytes() == report, "report replay")
    for path, expected in PINNED.items():
        require(sha256(path) == expected, f"frozen input modified: {rel(path)}")

    print("\nSTAGE 12B-3P — REPAIR_SITE_TEST RESULT DISPOSITION AND V2.1 IMPROVEMENT FREEZE")
    print(f"{'Status':<72}: PASS")
    print(f"{'Disposition status':<72}: FROZEN")
    print(f"{'V2.1 improvement lifecycle':<72}: COMPLETED AND FROZEN CLOSED-CATALOG RESEARCH EXPERIMENT")
    print(f"{'Selected measurement / model':<72}: {SELECTED_MEASUREMENT} / {SELECTED_MODEL_ID}")
    print(f"{'Locked evaluation acceptance':<72}: NOT_MET / PRESERVED")
    print(f"{'Combined detection recall / target / gap':<72}: {detection:.8f} / 0.70000000 / {detection_gap:.8f}")
    print(f"{'All-injected exact-site / target / gap':<72}: {exact:.8f} / 0.60000000 / {exact_gap:.8f}")
    print(f"{'Observable candidate coverage / unique top1':<72}: {coverage:.8f} / {unique_top1:.8f}")
    print(f"{'Mean / maximum observable candidate sites':<72}: {mean_candidates:.4f} / {max_candidates}")
    print(f"{'Unmet criteria':<72}: {', '.join(unmet)}")
    print(f"{'Fault-free false alarms':<72}: 0")
    print(f"{'Training / threshold changes / candidate reselection':<72}: 0 / 0 / 0")
    print(f"{'REPAIR_SITE_TEST':<72}: CONSUMED / FROZEN / DO NOT REOPEN")
    print(f"{'DEV_SITE_TEST / VALIDATION / HOLDOUT access':<72}: 0 / 0 / 0")
    print(f"{'Independent generalization':<72}: NOT ESTABLISHED")
    print(f"{'Future combined-model brand':<72}: {FUTURE_BRAND}")
    print(f"{'Policy':<72}: {POLICY}")
    print(f"{'Policy SHA':<72}: {sha256(POLICY)}")
    print(f"{'Final lock':<72}: {FINAL_LOCK}")
    print(f"{'Final lock SHA':<72}: {sha256(FINAL_LOCK)}")
    print(f"{'Report':<72}: {REPORT}")
    print(f"{'Report SHA':<72}: {sha256(REPORT)}")
    print(f"{'Manifest':<72}: {MANIFEST}")
    print(f"{'Manifest SHA':<72}: {sha256(MANIFEST)}")
    print(f"{'Audit':<72}: {AUDIT}")
    print(f"{'Audit SHA':<72}: {sha256(AUDIT)}")
    print(f"{'Next gate':<72}: STAGE 12C-1A — V2.2 GENERALIZATION ARCHITECTURE, DATA-PARTITION, AND ACCEPTANCE-CONTRACT FREEZE")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        main()
