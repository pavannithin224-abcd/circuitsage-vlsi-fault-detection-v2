#!/usr/bin/env python3
"""Stage 12A-1D-R3: locator repair disposition and V2 Core freeze.

This is a read-only disposition gate.  It verifies the frozen 12A-1D,
12A-1D-R1 and 12A-1D-R2 evidence, compares the original and repaired
calibration locators, preserves the exact-reference detector, and freezes the
honest V2 Core result.  It does not deserialize a model, train, infer, or open
DEV_SITE_TEST, VALIDATION, or HOLDOUT.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
from typing import Any


STAGE = "12A-1D-R3"
ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "config" / "v2"
RESULT = ROOT / "results" / "circuitsage_hmac_v2_12a1"

ORIGINAL_SOURCE = ROOT / "stage_12a1d_v2_linked_train.py"
ORIGINAL_LOCK = RESULT / "v2_training_12a1d/circuitsage_hmac_v2_selection_lock_12a1d.json"
ORIGINAL_MANIFEST = RESULT / "circuitsage_hmac_v2_training_manifest_12a1d.json"
ORIGINAL_AUDIT = RESULT / "circuitsage_hmac_v2_training_calibration_freeze_12a1d.json"

R1_SOURCE = ROOT / "stage_12a1d_r1_localization_failure_review.py"
R1_CONTRACT = CONFIG / "circuitsage_hmac_v2_localization_repair_contract_12a1d_r1.json"
R1_MANIFEST = RESULT / "circuitsage_hmac_v2_localization_failure_review_manifest_12a1d_r1.json"
R1_AUDIT = RESULT / "circuitsage_hmac_v2_localization_failure_review_freeze_12a1d_r1.json"

R2_SOURCE = ROOT / "stage_12a1d_r2_response_conditioned_locator_train.py"
R2_WORK = RESULT / "v2_repair_training_12a1d_r2"
R2_METRICS = R2_WORK / "circuitsage_hmac_v2_repair_candidate_calibration_metrics_12a1d_r2.csv"
R2_MODEL = R2_WORK / "circuitsage_hmac_v2_selected_repaired_locator_12a1d_r2.npz"
R2_METADATA = R2_WORK / "circuitsage_hmac_v2_selected_repaired_locator_metadata_12a1d_r2.json"
R2_RANKINGS = R2_WORK / "circuitsage_hmac_v2_selected_repair_calibration_rankings_12a1d_r2.npz"
R2_LOCK = R2_WORK / "circuitsage_hmac_v2_repair_selection_lock_12a1d_r2.json"
R2_MANIFEST = RESULT / "circuitsage_hmac_v2_repair_training_manifest_12a1d_r2.json"
R2_AUDIT = RESULT / "circuitsage_hmac_v2_repair_training_calibration_freeze_12a1d_r2.json"

WORK = RESULT / "v2_core_disposition_12a1d_r3"
COMPARISON = WORK / "circuitsage_hmac_v2_locator_comparison_12a1d_r3.csv"
REPORT = WORK / "circuitsage_hmac_v2_core_disposition_report_12a1d_r3.md"
POLICY = CONFIG / "circuitsage_hmac_v2_core_disposition_policy_12a1d_r3.json"
CORE_LOCK = CONFIG / "circuitsage_hmac_v2_core_lock_12a1d_r3.json"
MANIFEST = RESULT / "circuitsage_hmac_v2_core_disposition_manifest_12a1d_r3.json"
AUDIT = RESULT / "circuitsage_hmac_v2_core_freeze_12a1d_r3.json"

PINNED = {
    ORIGINAL_SOURCE: "6144da31a28041f045d696d041f147efcfceacccd1fcfecaa52d8561c640020d",
    ORIGINAL_LOCK: "6fb8d4eb79704652ccb289090e2b249c7600f0db723fdfffe62161e4fe5028b7",
    ORIGINAL_MANIFEST: "2c3df37e43ae93fb5ff14becf873e270ff59be7a8e30f9032feddfa5880b18fe",
    ORIGINAL_AUDIT: "c6437b429d8b2f4747dbbdcdcf7b92a7703d75b6428c8fb643d8c75fc3184b96",
    R1_SOURCE: "ff83a7c1581281494fc7043c793fb3c95c8c8ab24e0aa055986542ac21ae44c2",
    R1_CONTRACT: "ccaab77ca3d33225839c166ec68c2c97953e35a26ca660b8a86fe2babfcb7dd6",
    R1_MANIFEST: "6eb629b9aa6b12384e5cdeac26f2d6b9871fa60b67375dc62810598bdf14cc05",
    R1_AUDIT: "b2c184c654332beddcdc46530e8273fd979d7126a570c72893608e89be7ee179",
    R2_SOURCE: "06f3fef9b9bcb6b48995f799cb23c621b9d15c9e16402cd7923dff55bfee40a3",
    R2_METRICS: "5efc533646de5dc8e551bed1a1d4762bdda5b3a64a47f581535e8d74cfc1cecc",
    R2_MODEL: "6c4ddca604054c204e20b3465393d1cf3c414fd5431f37ed7bf722aedc465513",
    R2_METADATA: "1c1385ed92d15fbba02d35680019d9d5bd36370961aa53ea73bc6c4c3e02c377",
    R2_RANKINGS: "757f508d6f695fcf573113c4feaac7096aabb435ee918c3feb2e70a96e612354",
    R2_LOCK: "8a9b4394f13df2fd03058a5475bfa00a049585384ca68e9cc4853819aa9d26c4",
    R2_MANIFEST: "06518b9d7bfd73c6dad54501bec86f7e120156b5e34ea9073d3453224e61b7b4",
    R2_AUDIT: "bff78ea6919d5e4305e0b9d48178b342cb8fe7992c34f5f294a8b6c0f9375f3f",
}

TARGETS = {
    "fault_free_false_alarm_rate_max": 0.05,
    "observable_detection_recall_min": 0.90,
    "observable_candidate_coverage_top50_min": 0.95,
    "unique_signature_top1_site_accuracy_min": 0.80,
    "observable_top5_site_accuracy_min": 0.80,
}

METRIC_KEYS = (
    "observable_mean_reciprocal_rank",
    "unique_signature_top1_site_accuracy",
    "observable_top5_site_accuracy",
    "observable_candidate_coverage_top50",
    "sa0_sa1_accuracy_given_top1_site",
    "fault_free_false_alarm_rate",
    "observable_detection_recall",
)


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


def verify_record(item: dict[str, Any], context: str) -> None:
    path = resolve_record_path(item)
    require(path.is_file(), f"missing {context}: {rel(path)}")
    require(sha256(path) == item.get("sha256"), f"{context} changed: {path.name}")
    if "bytes" in item:
        require(path.stat().st_size == int(item["bytes"]), f"{context} size changed: {path.name}")


def verify_manifest_records(manifest: dict[str, Any], label: str) -> None:
    for section in ("input_evidence", "outputs"):
        items = manifest.get(section)
        require(isinstance(items, dict), f"{label} {section}")
        for item in items.values():
            if isinstance(item, dict) and isinstance(item.get("path"), str):
                verify_record(item, f"{label} {section}")


def finite_metrics(payload: Any, label: str) -> dict[str, float]:
    require(isinstance(payload, dict), f"{label} selected metrics")
    result: dict[str, float] = {}
    for key in METRIC_KEYS:
        require(key in payload, f"{label} metric missing: {key}")
        value = float(payload[key])
        require(math.isfinite(value), f"{label} metric not finite: {key}")
        require(0.0 <= value <= 1.0, f"{label} metric range: {key}")
        result[key] = value
    if "observable_shortlist_candidate_coverage" in payload:
        value = float(payload["observable_shortlist_candidate_coverage"])
        require(math.isfinite(value) and 0.0 <= value <= 1.0, "R2 shortlist metric range")
        result["observable_shortlist_candidate_coverage"] = value
    return result


def acceptance(metrics: dict[str, float]) -> dict[str, bool]:
    return {
        "fault_free_false_alarm_rate": metrics["fault_free_false_alarm_rate"] <= TARGETS["fault_free_false_alarm_rate_max"],
        "observable_detection_recall": metrics["observable_detection_recall"] >= TARGETS["observable_detection_recall_min"],
        "observable_candidate_coverage_top50": metrics["observable_candidate_coverage_top50"] >= TARGETS["observable_candidate_coverage_top50_min"],
        "unique_signature_top1_site_accuracy": metrics["unique_signature_top1_site_accuracy"] >= TARGETS["unique_signature_top1_site_accuracy_min"],
        "observable_top5_site_accuracy": metrics["observable_top5_site_accuracy"] >= TARGETS["observable_top5_site_accuracy_min"],
    }


def locator_order(metrics: dict[str, float]) -> tuple[float, float, float, float]:
    return (
        metrics["observable_mean_reciprocal_rank"],
        metrics["observable_candidate_coverage_top50"],
        metrics["observable_top5_site_accuracy"],
        metrics["unique_signature_top1_site_accuracy"],
    )


def csv_bytes(rows: list[dict[str, Any]]) -> bytes:
    fields = [
        "experiment", "candidate_id", "disposition",
        "observable_mean_reciprocal_rank", "unique_signature_top1_site_accuracy",
        "observable_top5_site_accuracy", "observable_candidate_coverage_top50",
        "observable_shortlist_candidate_coverage",
        "sa0_sa1_accuracy_given_top1_site", "fault_free_false_alarm_rate",
        "observable_detection_recall", "calibration_advancement_target",
    ]
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode()


def verify_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    print("STAGE 12A-1D-R3 — LOCATOR REPAIR DISPOSITION AND V2 CORE FREEZE")
    print("FROZEN INPUT VERIFICATION")
    evidence: dict[str, Any] = {}
    for path, expected in PINNED.items():
        require(path.is_file(), f"missing frozen input: {rel(path)}")
        require(sha256(path) == expected, f"frozen input SHA: {rel(path)}")
        evidence[rel(path)] = record(path)
        print(f"  {path.name:<82}: OK", flush=True)

    original_lock = load_json(ORIGINAL_LOCK)
    original_manifest = load_json(ORIGINAL_MANIFEST)
    original_audit = load_json(ORIGINAL_AUDIT)
    r1_contract = load_json(R1_CONTRACT)
    r1_audit = load_json(R1_AUDIT)
    r2_lock = load_json(R2_LOCK)
    r2_manifest = load_json(R2_MANIFEST)
    r2_audit = load_json(R2_AUDIT)

    require(original_lock.get("status") == "PASS", "original selection status")
    require(original_lock.get("training_status") == "FROZEN", "original training freeze")
    require(original_lock.get("selection_status") == "FROZEN", "original selection freeze")
    require(original_lock.get("advancement_target_on_calibration") == "NOT_MET", "original advancement state")
    require(original_lock.get("dev_site_test_opened") is False, "original DEV_SITE_TEST state")
    require(original_audit.get("calibration_advancement_target") == "NOT_MET", "original audit gate")
    require(original_audit.get("dev_site_test_opened") is False, "original audit DEV_SITE_TEST state")

    require(r1_contract.get("status") == "FROZEN", "R1 contract status")
    require(r1_contract.get("locked_evaluation_authorization") == "NOT AUTHORIZED BY THIS CONTRACT", "R1 evaluation prohibition")
    require(r1_contract.get("retraining_of_v1") == "PROHIBITED", "R1 V1 prohibition")
    require(r1_audit.get("status") == "PASS", "R1 audit status")

    require(r2_lock.get("status") == "PASS", "R2 selection status")
    require(r2_lock.get("training_status") == "FROZEN", "R2 training freeze")
    require(r2_lock.get("selection_status") == "FROZEN", "R2 selection freeze")
    require(r2_lock.get("calibration_advancement_target") == "NOT_MET", "R2 advancement state")
    require(r2_lock.get("dev_site_test_opened") is False, "R2 DEV_SITE_TEST state")
    require(r2_lock.get("validation_access_count") == 0, "R2 VALIDATION access")
    require(r2_lock.get("holdout_access_count") == 0, "R2 HOLDOUT access")
    require(r2_lock.get("v1_model_deserialized") is False, "R2 V1 deserialization state")
    require(r2_lock.get("v1_model_modified") is False, "R2 V1 modification state")
    replay = r2_lock.get("deterministic_replay")
    require(isinstance(replay, dict) and replay.get("status") == "PASS", "R2 deterministic replay")
    require(r2_audit.get("status") == "PASS", "R2 audit status")
    require(r2_audit.get("calibration_advancement_target") == "NOT_MET", "R2 audit gate")
    require(r2_audit.get("dev_site_test") == "LOCKED / NOT OPENED", "R2 audit DEV_SITE_TEST state")
    require(r2_audit.get("validation_access_count") == 0, "R2 audit VALIDATION access")
    require(r2_audit.get("holdout_access_count") == 0, "R2 audit HOLDOUT access")

    verify_manifest_records(original_manifest, "original manifest")
    verify_manifest_records(r2_manifest, "R2 manifest")
    print("  Prior semantic locks, recursive manifests and partition prohibitions             : PASS")
    return original_lock, r2_lock, evidence


def self_test() -> None:
    sample = {
        "observable_mean_reciprocal_rank": 0.1,
        "unique_signature_top1_site_accuracy": 0.81,
        "observable_top5_site_accuracy": 0.82,
        "observable_candidate_coverage_top50": 0.96,
        "sa0_sa1_accuracy_given_top1_site": 0.5,
        "fault_free_false_alarm_rate": 0.0,
        "observable_detection_recall": 1.0,
    }
    require(all(acceptance(sample).values()), "acceptance canary")
    lower = dict(sample)
    lower["observable_mean_reciprocal_rank"] = 0.01
    require(locator_order(sample) > locator_order(lower), "locator ordering canary")
    require(canonical_json({"b": 2, "a": 1}) == canonical_json({"a": 1, "b": 2}), "JSON replay canary")
    rows = [{
        "experiment": "x", "candidate_id": "y", "disposition": "z",
        **sample, "observable_shortlist_candidate_coverage": "NOT_APPLICABLE",
        "calibration_advancement_target": "PASS",
    }]
    require(csv_bytes(rows) == csv_bytes(rows), "CSV replay canary")
    print("Stage 12A-1D-R3 self-test: PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return

    require(ROOT.name == "vlsi_fault_detection_v2", f"run from project root, not {ROOT}")
    for path in (COMPARISON, REPORT, POLICY, CORE_LOCK, MANIFEST, AUDIT):
        require(not path.exists(), f"Stage {STAGE} output already exists: {rel(path)}")

    original_lock, r2_lock, evidence = verify_inputs()
    original = finite_metrics(original_lock.get("selected_metrics"), "original")
    repaired = finite_metrics(r2_lock.get("selected_metrics"), "R2")
    original_checks = acceptance(original)
    repaired_checks = acceptance(repaired)
    require(not all(original_checks.values()), "original unexpectedly passes advancement")
    require(not all(repaired_checks.values()), "R2 unexpectedly passes advancement")
    require(original_checks["fault_free_false_alarm_rate"] and original_checks["observable_detection_recall"], "original detector criteria")
    require(repaired_checks["fault_free_false_alarm_rate"] and repaired_checks["observable_detection_recall"], "R2 detector criteria")

    best_is_original = locator_order(original) >= locator_order(repaired)
    require(best_is_original, "R2 result no longer matches frozen negative disposition")
    original_id = str(original_lock.get("selected_candidate_id"))
    repaired_id = str(r2_lock.get("selected_candidate_id"))

    deltas = {key: repaired[key] - original[key] for key in METRIC_KEYS}
    rows = [
        {
            "experiment": "12A-1D ORIGINAL",
            "candidate_id": original_id,
            "disposition": "BEST OBSERVED CALIBRATION LOCATOR; RESEARCH REFERENCE ONLY",
            **original,
            "observable_shortlist_candidate_coverage": "NOT_APPLICABLE",
            "calibration_advancement_target": "NOT_MET",
        },
        {
            "experiment": "12A-1D-R2 REPAIR",
            "candidate_id": repaired_id,
            "disposition": "REJECTED FOR ADVANCEMENT; FROZEN NEGATIVE RESULT",
            **repaired,
            "observable_shortlist_candidate_coverage": repaired.get("observable_shortlist_candidate_coverage", "NOT_REPORTED"),
            "calibration_advancement_target": "NOT_MET",
        },
    ]

    policy = {
        "policy_version": "CIRCUITSAGE-HMAC-V2-CORE-DISPOSITION-v1",
        "stage": STAGE,
        "status": "FROZEN",
        "experiment_outcome": "PARTIAL SUCCESS; DETECTION ACCEPTED, LOCALIZATION ADVANCEMENT NOT MET",
        "detector_disposition": {
            "component": "EXACT GOLDEN-REFERENCE ANOMALY GATE",
            "status": "PRESERVED / ACCEPTED WITHIN FROZEN HMAC EXPERIMENT",
            "fault_free_false_alarm_rate": repaired["fault_free_false_alarm_rate"],
            "observable_detection_recall": repaired["observable_detection_recall"],
            "limitations": [
                "only detects behavior that differs from the committed golden responses",
                "cannot prove that a normal-compatible or unactivated fault is absent",
                "not independently validated on another chip or unseen fault family",
            ],
        },
        "locator_disposition": {
            "status": "FAILED ADVANCEMENT / RESEARCH-ONLY",
            "best_observed_calibration_locator": original_id,
            "best_observed_source_stage": "12A-1D",
            "repair_candidate": repaired_id,
            "repair_outcome": "REJECTED FOR ADVANCEMENT",
            "required_targets": TARGETS,
            "reason": "neither original nor repaired locator met the frozen catalog-scale localization targets",
            "exact_location_claim_authorized": False,
        },
        "v1_linkage": "PRESERVED AS FROZEN READ-ONLY CANDIDATE SUPPORT; NOT MODIFIED",
        "locked_partition_disposition": {
            "DEV_SITE_TEST": "REMAINS LOCKED / NOT OPENED",
            "VALIDATION": "NOT ACCESSED",
            "HOLDOUT": "NOT ACCESSED / BLOCKED",
        },
        "prohibitions": [
            "DO NOT PRESENT V2 AS A VALIDATED EXACT-SITE LOCATOR",
            "DO NOT OPEN DEV_SITE_TEST FOR THIS FAILED CALIBRATION BRANCH",
            "DO NOT MODIFY THE FROZEN V1 RELEASE",
            "DO NOT RETRAIN OR TUNE AGAIN UNDER STAGE 12A-1D",
        ],
        "future_work": "start a separately contracted V2.1 locator redesign if more localization research is desired",
        "automatic_continuation": "NONE",
    }

    core_lock = {
        "lock_version": "CIRCUITSAGE-HMAC-V2-CORE-LOCK-v1",
        "stage": STAGE,
        "status": "PASS",
        "lifecycle": "COMPLETED AND FROZEN TIMEBOXED RESEARCH EXPERIMENT",
        "v2_core_result": "PARTIAL SUCCESS",
        "detector": "FROZEN ACCEPTED COMPONENT WITH DOCUMENTED SCOPE",
        "locator": "FROZEN NEGATIVE RESULT / NOT ACCEPTED FOR ADVANCEMENT",
        "best_observed_locator_candidate": original_id,
        "best_observed_locator_metrics": original,
        "r2_selected_candidate": repaired_id,
        "r2_selected_metrics": repaired,
        "r2_minus_original_metric_deltas": deltas,
        "calibration_advancement_target": "NOT_MET",
        "locked_dev_site_test_evaluation_authorized": False,
        "unknown_fault_detection_claim": "SUPPORTED ONLY FOR OBSERVABLE DEVIATIONS IN THE FROZEN HMAC TEST REGIME",
        "exact_site_localization_claim": "NOT SUPPORTED",
        "v1_model_deserialized_by_stage": False,
        "v1_model_modified": False,
        "model_training_calls": 0,
        "model_inference_calls": 0,
        "dev_site_test_opened": False,
        "validation_access_count": 0,
        "holdout_access_count": 0,
        "disposition_policy": None,
    }

    report = f"""# CircuitSage-HMAC V2 Core disposition — Stage {STAGE}

## Outcome

The one-week V2 Core experiment is complete and frozen as a partial success. The exact golden-reference detector passed its frozen calibration criteria: zero fault-free false alarms and 100% recall for observable deviations. Catalog-scale localization did not pass.

## Locator comparison

| Metric | Original 12A-1D | R2 repair | R2 − original | Required |
|---|---:|---:|---:|---:|
| Observable MRR | {original['observable_mean_reciprocal_rank']:.8f} | {repaired['observable_mean_reciprocal_rank']:.8f} | {deltas['observable_mean_reciprocal_rank']:+.8f} | reported/selection metric |
| Unique-signature top-1 site | {original['unique_signature_top1_site_accuracy']:.8f} | {repaired['unique_signature_top1_site_accuracy']:.8f} | {deltas['unique_signature_top1_site_accuracy']:+.8f} | ≥ 0.80 |
| Observable top-5 site | {original['observable_top5_site_accuracy']:.8f} | {repaired['observable_top5_site_accuracy']:.8f} | {deltas['observable_top5_site_accuracy']:+.8f} | ≥ 0.80 |
| Candidate coverage at top 50 | {original['observable_candidate_coverage_top50']:.8f} | {repaired['observable_candidate_coverage_top50']:.8f} | {deltas['observable_candidate_coverage_top50']:+.8f} | ≥ 0.95 |

R2 did not repair the locator and performed below the original locator on the primary ranking measures. The R2 artifact is retained as a reproducible negative result. The original 12A-1D candidate is recorded only as the best observed calibration reference; it is not promoted to a validated exact-site locator.

## Safe capability statement

V2 can flag an observable response mismatch against the committed golden reference within this frozen OpenTitan HMAC SA0/SA1 experiment. It cannot reliably identify the exact physical fault site across the 45,678-instance catalog. A no-anomaly result also cannot rule out an invisible, unactivated, or normal-compatible fault.

DEV_SITE_TEST was not opened because the calibration advancement gate failed. VALIDATION and HOLDOUT were not accessed. V1 remains frozen and unchanged.

## Future work

Any additional locator work should begin as a new V2.1 contract with a redesigned retrieval objective and a new calibration plan. It must not silently continue tuning this frozen branch or use locked partitions for model selection.
"""

    frozen_write(COMPARISON, csv_bytes(rows))
    frozen_write(REPORT, report.encode())
    frozen_write(POLICY, canonical_json(policy))
    core_lock["disposition_policy"] = record(POLICY)
    core_lock["comparison"] = record(COMPARISON)
    core_lock["report"] = record(REPORT)
    frozen_write(CORE_LOCK, canonical_json(core_lock))

    outputs = {rel(path): record(path) for path in (COMPARISON, REPORT, POLICY, CORE_LOCK)}
    manifest = {
        "manifest_version": "CIRCUITSAGE-HMAC-V2-CORE-DISPOSITION-MANIFEST-v1",
        "stage": STAGE,
        "status": "PASS",
        "input_evidence": evidence,
        "outputs": outputs,
        "model_training_calls": 0,
        "model_inference_calls": 0,
        "model_objects_deserialized": 0,
        "dev_site_test_opened": False,
        "validation_access_count": 0,
        "holdout_access_count": 0,
    }
    frozen_write(MANIFEST, canonical_json(manifest))
    audit = {
        "audit_version": "CIRCUITSAGE-HMAC-V2-CORE-FREEZE-v1",
        "stage": STAGE,
        "status": "PASS",
        "disposition_status": "FROZEN",
        "v2_core_status": "COMPLETED AND FROZEN TIMEBOXED RESEARCH EXPERIMENT",
        "detector_disposition": "ACCEPTED WITHIN FROZEN TEST REGIME",
        "locator_disposition": "ADVANCEMENT NOT MET / RESEARCH-ONLY",
        "best_observed_locator": original_id,
        "repair_locator": repaired_id,
        "repair_vs_original": "REGRESSED ON PRIMARY LOCALIZATION METRICS",
        "calibration_advancement_target": "NOT_MET",
        "dev_site_test": "LOCKED / NOT OPENED / NOT AUTHORIZED",
        "validation_access_count": 0,
        "holdout_access_count": 0,
        "v1_model_deserialized": False,
        "v1_model_modified": False,
        "new_training_performed": False,
        "new_inference_performed": False,
        "independent_generalization": "NOT ESTABLISHED",
        "production_readiness": "NOT ESTABLISHED",
        "manifest": record(MANIFEST),
        "core_lock": record(CORE_LOCK),
        "automatic_continuation": "NONE",
        "next_gate": "OPTIONAL STAGE 12B-1A — V2.1 LOCATOR REDESIGN CONTRACT; LOCKED EVALUATION REMAINS BLOCKED",
    }
    frozen_write(AUDIT, canonical_json(audit))

    for path in (POLICY, CORE_LOCK, MANIFEST, AUDIT):
        require(canonical_json(load_json(path)) == path.read_bytes(), f"deterministic JSON replay: {path.name}")
    require(csv_bytes(rows) == COMPARISON.read_bytes(), "deterministic comparison CSV replay")
    require(REPORT.read_bytes() == report.encode(), "deterministic report replay")

    print(f"\nSTAGE {STAGE} — LOCATOR REPAIR DISPOSITION AND V2 CORE FREEZE")
    print(f"{'Status':<43}: PASS")
    print(f"{'Disposition status':<43}: FROZEN")
    print(f"{'V2 Core lifecycle':<43}: COMPLETED AND FROZEN RESEARCH EXPERIMENT")
    print(f"{'V2 Core outcome':<43}: PARTIAL SUCCESS")
    print(f"{'Detector disposition':<43}: ACCEPTED / PRESERVED")
    print(f"{'Locator disposition':<43}: ADVANCEMENT NOT MET / RESEARCH-ONLY")
    print(f"{'Best observed locator':<43}: {original_id}")
    print(f"{'Best calibration observable MRR':<43}: {original['observable_mean_reciprocal_rank']:.8f}")
    print(f"{'R2 repaired locator':<43}: {repaired_id}")
    print(f"{'R2 calibration observable MRR':<43}: {repaired['observable_mean_reciprocal_rank']:.8f}")
    print(f"{'R2 minus original MRR':<43}: {deltas['observable_mean_reciprocal_rank']:+.8f}")
    print(f"{'Calibration advancement target':<43}: NOT_MET")
    print(f"{'Exact-site localization claim':<43}: NOT SUPPORTED")
    print(f"{'DEV_SITE_TEST':<43}: LOCKED / NOT OPENED / NOT AUTHORIZED")
    print(f"{'VALIDATION / HOLDOUT access':<43}: 0 / 0")
    print(f"{'Training / inference calls':<43}: 0 / 0")
    print(f"{'V1 model deserialized / modified':<43}: NO / NO")
    print(f"{'Policy':<43}: {POLICY}")
    print(f"{'Policy SHA':<43}: {sha256(POLICY)}")
    print(f"{'Core lock':<43}: {CORE_LOCK}")
    print(f"{'Core lock SHA':<43}: {sha256(CORE_LOCK)}")
    print(f"{'Report':<43}: {REPORT}")
    print(f"{'Report SHA':<43}: {sha256(REPORT)}")
    print(f"{'Manifest':<43}: {MANIFEST}")
    print(f"{'Manifest SHA':<43}: {sha256(MANIFEST)}")
    print(f"{'Audit':<43}: {AUDIT}")
    print(f"{'Audit SHA':<43}: {sha256(AUDIT)}")
    print(f"{'Next gate':<43}: OPTIONAL V2.1 LOCATOR REDESIGN; LOCKED EVALUATION REMAINS BLOCKED")


if __name__ == "__main__":
    main()
