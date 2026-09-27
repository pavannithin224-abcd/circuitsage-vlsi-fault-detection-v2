#!/usr/bin/env python3
"""Stage 12B-1E: locked V2.1 DEV_SITE_TEST evaluation authorization freeze.

This is an authorization-only gate. It verifies the frozen Stage 12B-1D
training/selection chain and emits the immutable rules for one locked
DEV_SITE_TEST evaluation. It does not load feature matrices, deserialize a
model, run inference, select a threshold, or access VALIDATION/HOLDOUT.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import shutil
from pathlib import Path
from typing import Any


STAGE = "12B-1E"
ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "config/v2_1"
RESULT = ROOT / "results/circuitsage_hmac_v2_12b1"

TRAIN_SOURCE = ROOT / "stage_12b1d_v2_1_locator_train.py"
TRAIN_DIR = RESULT / "v2_1_training_12b1d"
TRAINED_BUNDLE = TRAIN_DIR / "circuitsage_hmac_v2_1_trained_candidate_bundle_12b1d.npz"
METRICS_CSV = TRAIN_DIR / "circuitsage_hmac_v2_1_candidate_calibration_metrics_12b1d.csv"
METRICS_JSON = TRAIN_DIR / "circuitsage_hmac_v2_1_candidate_calibration_metrics_12b1d.json"
SELECTED_METADATA = TRAIN_DIR / "circuitsage_hmac_v2_1_selected_locator_metadata_12b1d.json"
CALIBRATION_OUTPUTS = TRAIN_DIR / "circuitsage_hmac_v2_1_selected_calibration_outputs_12b1d.npz"
SELECTION_LOCK = TRAIN_DIR / "circuitsage_hmac_v2_1_selection_lock_12b1d.json"
TRAIN_MANIFEST = RESULT / "circuitsage_hmac_v2_1_training_manifest_12b1d.json"
TRAIN_AUDIT = RESULT / "circuitsage_hmac_v2_1_training_calibration_freeze_12b1d.json"

ACCEPTANCE = CONFIG / "circuitsage_hmac_v2_1_locator_acceptance_contract_12b1a.json"
PRIOR_AUTH = CONFIG / "circuitsage_hmac_v2_1_training_authorization_12b1c.json"
V2_CORE_FREEZE = ROOT / "results/circuitsage_hmac_v2_12a1/circuitsage_hmac_v2_core_freeze_12a1d_r3.json"
V1_LOCK = ROOT / "config/diagnostic_model/hmac_final_diagnostic_model_lock_11d2d.json"

EVALUATION_CONTRACT = CONFIG / "circuitsage_hmac_v2_1_dev_site_test_evaluation_contract_12b1e.json"
AUTHORIZATION = CONFIG / "circuitsage_hmac_v2_1_dev_site_test_evaluation_authorization_12b1e.json"
EXECUTION_PLAN = RESULT / "circuitsage_hmac_v2_1_dev_site_test_execution_plan_12b1e.csv"
PREFLIGHT = RESULT / "circuitsage_hmac_v2_1_dev_site_test_preflight_12b1e.json"
MANIFEST = RESULT / "circuitsage_hmac_v2_1_dev_site_test_authorization_manifest_12b1e.json"
AUDIT = RESULT / "circuitsage_hmac_v2_1_dev_site_test_authorization_freeze_12b1e.json"

PINNED = {
    TRAIN_SOURCE: "4600c2c95831869e20e98f9278ec5f697612a1604083d1ea3c378e9c46107c80",
    ACCEPTANCE: "899b2ff099da20b4631308962c3052e16c7cf875ae296f5b2f6da825d6c6c12a",
}

SITES = 22839
FAULTS = 45678
DEV_SITE_TEST_SITES = 3426
DEV_SITE_TEST_SAMPLES = 438528
MIN_FREE_GIB = 2.0


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


def record_path(item: dict[str, Any]) -> Path:
    value = item.get("path")
    require(isinstance(value, str) and value, "artifact record path")
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def verify_record(item: dict[str, Any], label: str) -> Path:
    path = record_path(item)
    require(path.is_file(), f"missing {label}: {rel(path)}")
    require(sha256(path) == item.get("sha256"), f"{label} SHA changed: {path.name}")
    if "bytes" in item:
        require(path.stat().st_size == int(item["bytes"]), f"{label} size changed: {path.name}")
    return path


def require_canonical(path: Path) -> None:
    require(canonical_json(load_json(path)) == path.read_bytes(), f"canonical replay: {path.name}")


def verify_manifest(manifest: dict[str, Any], label: str) -> None:
    for section in ("input_evidence", "outputs"):
        entries = manifest.get(section)
        require(isinstance(entries, dict), f"{label} {section}")
        for item in entries.values():
            if isinstance(item, dict) and isinstance(item.get("path"), str):
                verify_record(item, f"{label} {section}")


def verify_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    print("STAGE 12B-1E — LOCKED V2.1 DEV_SITE_TEST EVALUATION AUTHORIZATION")
    print("FROZEN INPUT VERIFICATION")
    evidence: dict[str, Any] = {}
    for path, expected in PINNED.items():
        require(path.is_file(), f"missing frozen input: {rel(path)}")
        require(sha256(path) == expected, f"frozen input SHA: {rel(path)}")
        evidence[rel(path)] = record(path)
        print(f"  {path.name:<86}: OK", flush=True)

    required = (
        TRAINED_BUNDLE, METRICS_CSV, METRICS_JSON, SELECTED_METADATA,
        CALIBRATION_OUTPUTS, SELECTION_LOCK, TRAIN_MANIFEST, TRAIN_AUDIT,
        PRIOR_AUTH, V2_CORE_FREEZE, V1_LOCK,
    )
    for path in required:
        require(path.is_file(), f"missing frozen prerequisite: {rel(path)}")
        evidence[rel(path)] = record(path)
        print(f"  {path.name:<86}: OK", flush=True)

    train_manifest = load_json(TRAIN_MANIFEST)
    train_audit = load_json(TRAIN_AUDIT)
    selection = load_json(SELECTION_LOCK)
    selected_metadata = load_json(SELECTED_METADATA)
    metrics = load_json(METRICS_JSON)
    acceptance = load_json(ACCEPTANCE)
    prior_auth = load_json(PRIOR_AUTH)
    v2_core = load_json(V2_CORE_FREEZE)

    for path in (METRICS_JSON, SELECTED_METADATA, SELECTION_LOCK, TRAIN_MANIFEST,
                 TRAIN_AUDIT, ACCEPTANCE, PRIOR_AUTH, V2_CORE_FREEZE, V1_LOCK):
        require_canonical(path)
    verify_manifest(train_manifest, "Stage 12B-1D manifest")

    require(train_manifest.get("status") == "PASS", "Stage 12B-1D manifest status")
    require(train_manifest.get("training_status") == "FROZEN", "Stage 12B-1D training status")
    require(train_manifest.get("selection_status") == "FROZEN", "Stage 12B-1D selection status")
    require(train_manifest.get("fit_partition") == "DEV_TRAIN ONLY", "Stage 12B-1D fit partition")
    require(train_manifest.get("selection_partition") == "DEV_CALIBRATION ONLY", "Stage 12B-1D selection partition")
    require(train_manifest.get("dev_site_test_opened") is False, "Stage 12B-1D DEV_SITE_TEST state")
    require(train_manifest.get("validation_access_count") == 0, "Stage 12B-1D VALIDATION access")
    require(train_manifest.get("holdout_access_count") == 0, "Stage 12B-1D HOLDOUT access")

    require(train_audit.get("status") == "PASS", "Stage 12B-1D audit status")
    require(train_audit.get("training_status") == "FROZEN", "Stage 12B-1D audit training status")
    require(train_audit.get("model_selection_status") == "FROZEN", "Stage 12B-1D audit selection status")
    require(train_audit.get("calibration_advancement_target") == "PASS", "Stage 12B-1D calibration advancement target")
    require(train_audit.get("dev_site_test") == "LOCKED / NOT OPENED", "Stage 12B-1D locked test state")
    require(train_audit.get("validation_access_count") == 0, "Stage 12B-1D audit VALIDATION access")
    require(train_audit.get("holdout_access_count") == 0, "Stage 12B-1D audit HOLDOUT access")
    verify_record(train_audit["selection_lock"], "Stage 12B-1D selection lock")
    verify_record(train_audit["manifest"], "Stage 12B-1D manifest")

    require(selection.get("status") == "PASS", "selection lock status")
    require(selection.get("training_status") == "FROZEN", "selection lock training status")
    require(selection.get("selection_status") == "FROZEN", "selection lock selection status")
    require(selection.get("calibration_advancement_target") == "PASS", "selection advancement target")
    require(selection.get("dev_site_test_opened") is False, "selection test state")
    require(selection.get("validation_access_count") == 0, "selection VALIDATION access")
    require(selection.get("holdout_access_count") == 0, "selection HOLDOUT access")
    require(selection.get("v1_model_modified") is False, "V1 modification state")
    require(selection.get("v2_core_modified") is False, "V2 Core modification state")
    verify_record(selection["trained_candidate_bundle"], "selected trained bundle")
    verify_record(selection["selected_metadata"], "selected metadata")
    verify_record(selection["calibration_outputs"], "calibration outputs")

    selected_id = selection.get("selected_candidate_id")
    require(isinstance(selected_id, str) and selected_id, "selected candidate ID")
    require(selected_id == selected_metadata.get("selected_candidate_id"), "selected candidate binding")
    require(selected_metadata.get("calibration_advancement_target") == "PASS", "selected metadata target")
    require(metrics.get("status") == "PASS", "calibration metrics status")
    require(metrics.get("evaluation_scope") == "CLOSED-CATALOG CONSISTENCY; NOT INDEPENDENT GENERALIZATION", "calibration scope")
    require(acceptance.get("status") == "FROZEN", "acceptance contract status")
    require(prior_auth.get("dev_site_test_evaluation") == "NOT AUTHORIZED", "prior site-test authorization")
    require(v2_core.get("status") == "PASS", "V2 Core freeze status")

    selected_metrics = selection.get("selected_metrics")
    require(isinstance(selected_metrics, dict), "selected metrics")
    checks = selected_metrics.get("acceptance_checks")
    require(isinstance(checks, dict) and checks and all(value is True for value in checks.values()), "frozen acceptance checks")
    print("  Frozen selection, advancement criteria and partition history                         : PASS")
    return evidence, selection, acceptance


def plan_csv(selected_id: str) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=(
        "step", "partition", "operation", "fitting_allowed", "threshold_change_allowed",
        "expected_sites", "expected_samples", "execution", "status_before_execution",
    ), lineterminator="\n")
    writer.writeheader()
    rows = [
        (1, "FROZEN INPUTS", "verify hashes and semantic locks", "NO", "NO", 0, 0, "SEQUENTIAL", "AUTHORIZED"),
        (2, "DEV_SITE_TEST", f"locked inference with {selected_id}", "NO", "NO", DEV_SITE_TEST_SITES, DEV_SITE_TEST_SAMPLES, "SEQUENTIAL", "AUTHORIZED / NOT STARTED"),
        (3, "DEV_SITE_TEST", "ambiguity-aware localization metrics and site bootstrap", "NO", "NO", DEV_SITE_TEST_SITES, DEV_SITE_TEST_SAMPLES, "SEQUENTIAL", "AUTHORIZED / NOT STARTED"),
        (4, "OUTPUTS", "freeze predictions, comparison, manifest and audit", "NO", "NO", 0, 0, "SEQUENTIAL", "AUTHORIZED / NOT STARTED"),
    ]
    for row in rows:
        writer.writerow(dict(zip(writer.fieldnames, row)))
    return output.getvalue().encode()


def self_test() -> None:
    payload = {"b": 2, "a": 1}
    require(canonical_json(payload) == canonical_json({"a": 1, "b": 2}), "JSON replay canary")
    require(plan_csv("TEST") == plan_csv("TEST"), "plan replay canary")
    print("Stage 12B-1E self-test: PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return

    require(ROOT.name == "vlsi_fault_detection_v2", f"run from project root, not {ROOT}")
    for path in (EVALUATION_CONTRACT, AUTHORIZATION, EXECUTION_PLAN, PREFLIGHT, MANIFEST, AUDIT):
        require(not path.exists(), f"Stage {STAGE} output already exists: {rel(path)}")

    evidence, selection, acceptance = verify_inputs()
    selected_id = str(selection["selected_candidate_id"])
    selected_metrics = selection["selected_metrics"]
    free_gib = shutil.disk_usage(ROOT).free / (1024 ** 3)
    require(free_gib >= MIN_FREE_GIB, f"free disk {free_gib:.2f} GiB below {MIN_FREE_GIB:.2f} GiB")

    evaluation_contract = {
        "contract_version": "CIRCUITSAGE-HMAC-V2.1-LOCKED-SITE-TEST-EVALUATION-v1",
        "stage": STAGE,
        "status": "FROZEN",
        "purpose": "one locked DEV_SITE_TEST evaluation of the frozen V2.1 locator",
        "scientific_scope": "CLOSED-CATALOG HMAC SINGLE PERSISTENT SA0/SA1",
        "selected_candidate_id": selected_id,
        "selected_candidate_bundle": record(TRAINED_BUNDLE),
        "selection_lock": record(SELECTION_LOCK),
        "calibration_metrics": selected_metrics,
        "test_partition": "DEV_SITE_TEST",
        "expected_sites": DEV_SITE_TEST_SITES,
        "expected_samples": DEV_SITE_TEST_SAMPLES,
        "catalog_fault_instances": FAULTS,
        "catalog_physical_sites": SITES,
        "execution": "SEQUENTIAL",
        "parallel_workers": 1,
        "model_fitting_calls_allowed": 0,
        "threshold_or_selection_changes_allowed": 0,
        "calibration_or_test_feedback_changes_allowed": 0,
        "required_metrics": [
            "fault-free false-alarm rate", "observable detection recall",
            "exact-signature candidate-set coverage", "unique-signature top-1 site accuracy",
            "observable candidate-set coverage", "ambiguous false-unique rate",
            "observable mean reciprocal rank", "top-5 and top-10 set accuracy",
            "candidate-set size distribution", "site-bootstrap confidence intervals",
        ],
        "comparison_policy": "compare with frozen V2 Core locator and preserve V1 support unchanged",
        "acceptance_contract": record(ACCEPTANCE),
        "interpretation_limit": "closed-catalog test; not independent-chip or production generalization",
        "validation_access": "PROHIBITED",
        "holdout_access": "PROHIBITED",
    }
    authorization = {
        "authorization_version": "CIRCUITSAGE-HMAC-V2.1-SITE-TEST-AUTHORIZATION-v1",
        "stage": STAGE,
        "status": "FROZEN",
        "selected_candidate_id": selected_id,
        "dev_site_test_inference": "AUTHORIZED ONCE UNDER LOCKED CONTRACT / NOT STARTED",
        "dev_site_test_evaluation": "AUTHORIZED ONCE UNDER LOCKED CONTRACT / NOT STARTED",
        "model_training": "PROHIBITED",
        "candidate_selection_changes": "PROHIBITED",
        "threshold_changes": "PROHIBITED",
        "validation_access": "NOT AUTHORIZED",
        "holdout_access": "NOT AUTHORIZED",
        "v1_model": "FROZEN READ-ONLY / MODIFICATION PROHIBITED",
        "v2_core": "FROZEN READ-ONLY / MODIFICATION PROHIBITED",
        "next_authorized_stage": "12B-1F",
        "authorization_consumed": False,
    }
    preflight = {
        "preflight_version": "CIRCUITSAGE-HMAC-V2.1-SITE-TEST-PREFLIGHT-v1",
        "stage": STAGE,
        "status": "PASS",
        "selected_candidate_id": selected_id,
        "calibration_advancement_target": "PASS",
        "frozen_acceptance_checks": selection["selected_metrics"]["acceptance_checks"],
        "available_disk_gib": round(free_gib, 6),
        "minimum_disk_gib": MIN_FREE_GIB,
        "model_objects_deserialized": 0,
        "model_fitting_calls": 0,
        "inference_calls": 0,
        "dev_site_test_opened": False,
        "validation_access_count": 0,
        "holdout_access_count": 0,
    }

    frozen_write(EVALUATION_CONTRACT, canonical_json(evaluation_contract))
    frozen_write(AUTHORIZATION, canonical_json(authorization))
    frozen_write(EXECUTION_PLAN, plan_csv(selected_id))
    frozen_write(PREFLIGHT, canonical_json(preflight))
    manifest = {
        "manifest_version": "CIRCUITSAGE-HMAC-V2.1-SITE-TEST-AUTHORIZATION-MANIFEST-v1",
        "stage": STAGE,
        "status": "PASS",
        "input_evidence": evidence,
        "outputs": {rel(path): record(path) for path in (
            EVALUATION_CONTRACT, AUTHORIZATION, EXECUTION_PLAN, PREFLIGHT
        )},
        "model_objects_deserialized": 0,
        "model_fitting_calls": 0,
        "inference_calls": 0,
        "dev_site_test_opened": False,
        "validation_access_count": 0,
        "holdout_access_count": 0,
    }
    frozen_write(MANIFEST, canonical_json(manifest))
    audit = {
        "audit_version": "CIRCUITSAGE-HMAC-V2.1-SITE-TEST-AUTHORIZATION-FREEZE-v1",
        "stage": STAGE,
        "status": "PASS",
        "authorization_status": "FROZEN",
        "evaluation_contract_status": "FROZEN",
        "selected_candidate_id": selected_id,
        "calibration_advancement_target": "PASS / VERIFIED",
        "dev_site_test": "AUTHORIZED / NOT OPENED",
        "model_training": "PROHIBITED",
        "model_fitting_calls": 0,
        "threshold_changes": 0,
        "model_objects_deserialized": 0,
        "inference_calls": 0,
        "validation_access_count": 0,
        "holdout_access_count": 0,
        "v1_modified": False,
        "v2_core_modified": False,
        "evaluation_contract": record(EVALUATION_CONTRACT),
        "authorization": record(AUTHORIZATION),
        "execution_plan": record(EXECUTION_PLAN),
        "preflight": record(PREFLIGHT),
        "manifest": record(MANIFEST),
        "next_gate": "STAGE 12B-1F — LOCKED V2.1 DEV_SITE_TEST DETECTION/LOCALIZATION EVALUATION",
    }
    frozen_write(AUDIT, canonical_json(audit))

    for path in (EVALUATION_CONTRACT, AUTHORIZATION, PREFLIGHT, MANIFEST, AUDIT):
        require_canonical(path)
    require(plan_csv(selected_id) == EXECUTION_PLAN.read_bytes(), "execution plan replay")

    print("\nSTAGE 12B-1E — LOCKED V2.1 DEV_SITE_TEST EVALUATION AUTHORIZATION FREEZE")
    print(f"{'Status':<43}: PASS")
    print(f"{'Authorization status':<43}: FROZEN")
    print(f"{'Evaluation contract status':<43}: FROZEN")
    print(f"{'Selected candidate':<43}: {selected_id}")
    print(f"{'Calibration advancement':<43}: PASS / VERIFIED")
    print(f"{'DEV_SITE_TEST sites / samples':<43}: {DEV_SITE_TEST_SITES} / {DEV_SITE_TEST_SAMPLES}")
    print(f"{'DEV_SITE_TEST evaluation':<43}: AUTHORIZED / NOT STARTED")
    print(f"{'Model fitting / threshold changes':<43}: 0 / 0")
    print(f"{'Model objects deserialized':<43}: 0")
    print(f"{'Inference calls':<43}: 0")
    print(f"{'VALIDATION / HOLDOUT access':<43}: 0 / 0")
    print(f"{'V1 / V2 Core modified':<43}: NO / NO")
    print(f"{'Evaluation contract':<43}: {EVALUATION_CONTRACT}")
    print(f"{'Evaluation contract SHA':<43}: {sha256(EVALUATION_CONTRACT)}")
    print(f"{'Authorization':<43}: {AUTHORIZATION}")
    print(f"{'Authorization SHA':<43}: {sha256(AUTHORIZATION)}")
    print(f"{'Manifest':<43}: {MANIFEST}")
    print(f"{'Manifest SHA':<43}: {sha256(MANIFEST)}")
    print(f"{'Audit':<43}: {AUDIT}")
    print(f"{'Audit SHA':<43}: {sha256(AUDIT)}")
    print(f"{'Next gate':<43}: STAGE 12B-1F — LOCKED V2.1 DEV_SITE_TEST DETECTION/LOCALIZATION EVALUATION")


if __name__ == "__main__":
    main()
