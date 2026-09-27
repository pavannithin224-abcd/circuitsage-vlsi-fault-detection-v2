#!/usr/bin/env python3
"""Stage 12B-1G: V2.1 result disposition and generalization-readiness freeze.

This stage performs no training or inference. It verifies and interprets the
locked Stage 12B-1F evidence, freezes V2.1 with bounded capability claims, and
authorizes only the creation of a separate V2.2 generalization contract.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
from typing import Any


STAGE = "12B-1G"
ROOT = Path(__file__).resolve().parent
RESULT = ROOT / "results/circuitsage_hmac_v2_12b1"
CONFIG_V21 = ROOT / "config/v2_1"
CONFIG_V22 = ROOT / "config/v2_2"

EVAL_SOURCE = ROOT / "stage_12b1f_locked_v2_1_site_test.py"
EVAL_DIR = RESULT / "v2_1_evaluation_12b1f"
PREDICTIONS = EVAL_DIR / "circuitsage_hmac_v2_1_site_test_predictions_12b1f.npz"
METRICS = EVAL_DIR / "circuitsage_hmac_v2_1_site_test_metrics_12b1f.json"
BREAKDOWNS = EVAL_DIR / "circuitsage_hmac_v2_1_site_test_breakdowns_12b1f.csv"
BOOTSTRAP = EVAL_DIR / "circuitsage_hmac_v2_1_site_bootstrap_12b1f.csv"
COMPARISON = EVAL_DIR / "circuitsage_hmac_v2_1_vs_v2_core_comparison_12b1f.json"
EVALUATION_LOCK = EVAL_DIR / "circuitsage_hmac_v2_1_site_test_evaluation_lock_12b1f.json"
EVAL_MANIFEST = RESULT / "circuitsage_hmac_v2_1_site_test_manifest_12b1f.json"
EVAL_AUDIT = RESULT / "circuitsage_hmac_v2_1_site_test_evaluation_freeze_12b1f.json"

AUTH_AUDIT = RESULT / "circuitsage_hmac_v2_1_dev_site_test_authorization_freeze_12b1e.json"
SELECTION_LOCK = RESULT / "v2_1_training_12b1d/circuitsage_hmac_v2_1_selection_lock_12b1d.json"
V2_CORE_FREEZE = ROOT / "results/circuitsage_hmac_v2_12a1/circuitsage_hmac_v2_core_freeze_12a1d_r3.json"
V1_REMOTE_LOCK = ROOT / "config/release/hmac_github_remote_integrity_lock_11e1k.json"
V1_REMOTE_AUDIT = ROOT / "results/hmac_project_release_11e1k/hmac_github_remote_integrity_freeze_11e1k.json"

DISPOSITION_POLICY = CONFIG_V21 / "circuitsage_hmac_v2_1_final_disposition_policy_12b1g.json"
GENERALIZATION_POLICY = CONFIG_V22 / "circuitsage_hmac_v2_2_generalization_readiness_policy_12b1g.json"
CAPABILITY_REGISTRY_CSV = RESULT / "circuitsage_hmac_v2_1_capability_registry_12b1g.csv"
CAPABILITY_REGISTRY_JSON = RESULT / "circuitsage_hmac_v2_1_capability_registry_12b1g.json"
REPORT = RESULT / "circuitsage_hmac_v2_1_final_report_12b1g.md"
MANIFEST = RESULT / "circuitsage_hmac_v2_1_disposition_manifest_12b1g.json"
AUDIT = RESULT / "circuitsage_hmac_v2_1_disposition_generalization_readiness_freeze_12b1g.json"

PINNED = {
    EVAL_SOURCE: "c7bfab20466403b5d7485383f92862bcb041bddeb1dd916d8965d75da8f6d018",
    PREDICTIONS: "94484c9d1e2419cdfe3841c76c19050888e79296c0acb6c04bebbbfc4722eb33",
    METRICS: "0c7de49c735a0068c5471344e58748f652f4fe5042f9370e96f88aa7bd2f5c10",
    BREAKDOWNS: "62bac56f6641385846ad6a2487f3617ca79d50efa0af8996170d4387c579091f",
    BOOTSTRAP: "624ade1bdb02be0cd0a5b15511f01ad044df42f49efedf35310034672f71fb79",
    COMPARISON: "3ad64c3bba52dc776393904d1425a862e7566013d6ddad7195864042523b1a6a",
    EVALUATION_LOCK: "6ca9417085b894d3dafd97643062757ca0a2e1f892df4a8b9296801bb40bf732",
    EVAL_MANIFEST: "24c7cd8e0b96f1e7e7690d0f374b3ed07215842c6b2eef51a37d4254f5999a55",
    EVAL_AUDIT: "9c1e7f6d323a9e8b5241e418a92351969cbea39fb14b0902346048cf3068759c",
    AUTH_AUDIT: "f8bb0a1e90eb3670ccea41c0b68af1c428842478a08e50416c0f3c53e72a062a",
    SELECTION_LOCK: "b57b6e58732a66fb80eff78c9fee4224f883001a57281f3987cce346d0125b34",
    V1_REMOTE_LOCK: "3d93cd8bb9c3bba47dd665ba6a45ad4e193613c1f6d386e32b3265471d94dd5a",
    V1_REMOTE_AUDIT: "ca949a25697deb4bce317cf1e0e6181a68bca61ecbf111b252a21417aaff47a4",
}

SELECTED = "V21_EXACT_SIGNATURE_SET"


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
    require(isinstance(value, str) and value, "artifact record path")
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def verify_record(item: dict[str, Any], label: str) -> Path:
    path = resolve_record_path(item)
    require(path.is_file(), f"missing {label}: {rel(path)}")
    require(sha256(path) == item.get("sha256"), f"{label} changed: {path.name}")
    if "bytes" in item:
        require(path.stat().st_size == int(item["bytes"]), f"{label} size changed: {path.name}")
    return path


def require_canonical(path: Path) -> None:
    require(canonical_json(load_json(path)) == path.read_bytes(), f"canonical replay: {path.name}")


def verify_inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    print("STAGE 12B-1G — V2.1 RESULT DISPOSITION AND GENERALIZATION-READINESS")
    print("FROZEN INPUT VERIFICATION")
    evidence: dict[str, Any] = {}
    for path, expected in PINNED.items():
        require(path.is_file(), f"missing frozen input: {rel(path)}")
        require(sha256(path) == expected, f"frozen input SHA: {rel(path)}")
        evidence[rel(path)] = record(path)
        print(f"  {path.name:<88}: OK", flush=True)
    require(V2_CORE_FREEZE.is_file(), f"missing frozen input: {rel(V2_CORE_FREEZE)}")
    evidence[rel(V2_CORE_FREEZE)] = record(V2_CORE_FREEZE)
    print(f"  {V2_CORE_FREEZE.name:<88}: OK", flush=True)

    metric_bundle = load_json(METRICS)
    metrics = metric_bundle.get("metrics")
    require(isinstance(metrics, dict), "locked metric object")
    lock = load_json(EVALUATION_LOCK)
    manifest = load_json(EVAL_MANIFEST)
    audit = load_json(EVAL_AUDIT)
    authorization_audit = load_json(AUTH_AUDIT)
    selection = load_json(SELECTION_LOCK)
    v2_core = load_json(V2_CORE_FREEZE)
    v1_lock = load_json(V1_REMOTE_LOCK)
    v1_audit = load_json(V1_REMOTE_AUDIT)
    for path in (METRICS, COMPARISON, EVALUATION_LOCK, EVAL_MANIFEST, EVAL_AUDIT,
                 AUTH_AUDIT, SELECTION_LOCK, V2_CORE_FREEZE, V1_REMOTE_LOCK,
                 V1_REMOTE_AUDIT):
        require_canonical(path)
    for section in ("input_evidence", "outputs"):
        entries = manifest.get(section)
        require(isinstance(entries, dict), f"evaluation manifest {section}")
        for item in entries.values():
            if isinstance(item, dict) and isinstance(item.get("path"), str):
                verify_record(item, f"evaluation manifest {section}")

    require(metric_bundle.get("status") == "PASS", "metrics status")
    require(metric_bundle.get("result_scope") == "CLOSED-CATALOG CONSISTENCY ONLY", "metric scope")
    require(metrics.get("selected_candidate_id") == SELECTED, "metric candidate")
    require(metrics.get("closed_catalog_consistency_only") is True, "closed-catalog flag")
    require(metrics.get("independent_generalization") == "NOT ESTABLISHED", "generalization flag")
    require(metrics.get("dev_site_test_sites") == 3426, "site count")
    require(metrics.get("dev_site_test_fault_instances") == 6852, "fault count")
    require(metrics.get("dev_site_test_samples") == 438528, "sample count")
    require(lock.get("status") == "PASS", "evaluation lock status")
    require(lock.get("evaluation_status") == "FROZEN", "evaluation lock state")
    require(lock.get("dev_site_test_state") == "CONSUMED AND FROZEN", "test consumption")
    require(lock.get("model_fitting_calls") == 0, "model fitting state")
    require(lock.get("threshold_selection_calls") == 0, "threshold state")
    require(lock.get("validation_access_count") == 0, "VALIDATION access")
    require(lock.get("holdout_access_count") == 0, "HOLDOUT access")
    require(audit.get("status") == "PASS", "evaluation audit status")
    require(audit.get("result_scope") == "CLOSED-CATALOG CONSISTENCY ONLY", "audit scope")
    require(authorization_audit.get("status") == "PASS", "authorization audit status")
    require(selection.get("selected_candidate_id") == SELECTED, "selection binding")
    require(v2_core.get("status") == "PASS", "V2 Core freeze status")
    require(v1_lock.get("status") == "PASS", "V1 remote lock status")
    require(v1_audit.get("status") == "PASS", "V1 remote audit status")

    gates = {
        "fault_free_false_alarm_rate_le_0_01": float(metrics["fault_free_false_alarm_rate"]) <= 0.01,
        "observable_detection_recall_ge_0_95": float(metrics["observable_detection_recall"]) >= 0.95,
        "observable_candidate_set_coverage_ge_0_95": float(metrics["observable_candidate_set_coverage"]) >= 0.95,
        "unique_signature_top1_ge_0_95": float(metrics["unique_signature_top1_site_accuracy"]) >= 0.95,
        "ambiguous_false_unique_rate_le_0_01": float(metrics["ambiguous_false_unique_rate"]) <= 0.01,
    }
    require(all(gates.values()), "closed-catalog acceptance gates")
    print("  Locked evidence, bounded acceptance gates and predecessor freezes                         : PASS")
    return evidence, metrics


def capability_rows(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "capability": "observable_fault_detection",
            "status": "SUPPORTED_IN_FROZEN_HMAC_CATALOG",
            "metric": "observable_detection_recall",
            "value": format(float(metrics["observable_detection_recall"]), ".8f"),
            "limitation": "normal-compatible and unactivated faults may remain undetected",
        },
        {
            "capability": "all_injected_fault_detection",
            "status": "PARTIAL",
            "metric": "all_injected_detection_recall",
            "value": format(float(metrics["all_injected_detection_recall"]), ".8f"),
            "limitation": "approximately half of injected faults caused no observable response change",
        },
        {
            "capability": "unique_signature_exact_site_localization",
            "status": "SUPPORTED_IN_FROZEN_HMAC_CATALOG",
            "metric": "unique_signature_top1_site_accuracy",
            "value": format(float(metrics["unique_signature_top1_site_accuracy"]), ".8f"),
            "limitation": "applies only when the response signature maps to one physical site",
        },
        {
            "capability": "all_injected_exact_site_localization",
            "status": "LIMITED",
            "metric": "all_injected_exact_site_rate",
            "value": format(float(metrics["all_injected_exact_site_rate"]), ".8f"),
            "limitation": "ambiguous and normal-compatible responses cannot support exact localization",
        },
        {
            "capability": "ambiguous_candidate_set_localization",
            "status": "SUPPORTED_IN_FROZEN_HMAC_CATALOG",
            "metric": "observable_candidate_set_coverage",
            "value": format(float(metrics["observable_candidate_set_coverage"]), ".8f"),
            "limitation": f"mean {float(metrics['mean_candidate_set_size_observable']):.4f}; maximum {int(metrics['maximum_candidate_set_size_observable'])} candidates",
        },
        {
            "capability": "unseen_signature_or_circuit_generalization",
            "status": "NOT_ESTABLISHED",
            "metric": "independent_generalization",
            "value": "NOT_ESTABLISHED",
            "limitation": "requires separately frozen unseen-response and independent-circuit evaluation",
        },
    ]


def rows_csv(rows: list[dict[str, Any]]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode()


def report_text(metrics: dict[str, Any]) -> str:
    return f"""# CircuitSage-HMAC V2.1 final disposition — Stage {STAGE}

## Outcome

V2.1 is complete and frozen as a closed-catalog HMAC SA0/SA1 research component. The selected `V21_EXACT_SIGNATURE_SET` method correctly detects every observable fault in the locked site-test partition and always retains the true site in the returned catalog candidate set. It also avoids forcing one location when several sites have the same behavior.

This is not a 100% general fault detector. Only **{float(metrics['all_injected_detection_recall']) * 100:.2f}%** of all injected DEV_SITE_TEST faults produced an observable response change. Exact-site localization across all injected faults was **{float(metrics['all_injected_exact_site_rate']) * 100:.2f}%**. An observable query returned an average of **{float(metrics['mean_candidate_set_size_observable']):.2f}** possible sites, with a maximum of **{int(metrics['maximum_candidate_set_size_observable'])}**.

## Supported claims

- Observable-fault detection recall: **{float(metrics['observable_detection_recall']):.8f}** within the frozen catalog.
- Fault-free false-alarm rate: **{float(metrics['fault_free_false_alarm_rate']):.8f}** under the exact golden-reference gate.
- Unique-signature top-1 site accuracy: **{float(metrics['unique_signature_top1_site_accuracy']):.8f}**.
- Observable candidate-set coverage: **{float(metrics['observable_candidate_set_coverage']):.8f}**.
- Ambiguous false-unique rate: **{float(metrics['ambiguous_false_unique_rate']):.8f}**.

## Claims that are not supported

- Generalization to an unseen response signature.
- Generalization to a different circuit, synthesis flow, technology library, or fault family.
- Detection of a fault that produces no observable difference under the applied vectors.
- Exact localization when several sites are observationally equivalent.
- Production or silicon readiness.

## V2.2 direction

V2.2 may begin only with a new frozen contract. It should use response-only queries, strict unseen-vector/site/fault/circuit partitions, an open-set detector, graph-constrained candidate generation, full-catalog hard negatives, ambiguity-aware scoring, and adaptive test-vector selection. V1, V2 Core, and V2.1 remain immutable comparators.
"""


def self_test() -> None:
    rows = [{"capability": "x", "status": "y", "metric": "z", "value": "1", "limitation": "n"}]
    require(rows_csv(rows) == rows_csv(rows), "CSV replay canary")
    require(canonical_json({"b": 2, "a": 1}) == canonical_json({"a": 1, "b": 2}), "JSON replay canary")
    print("Stage 12B-1G self-test: PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    require(ROOT.name == "vlsi_fault_detection_v2", f"run from project root, not {ROOT}")
    outputs = (DISPOSITION_POLICY, GENERALIZATION_POLICY, CAPABILITY_REGISTRY_CSV,
               CAPABILITY_REGISTRY_JSON, REPORT, MANIFEST, AUDIT)
    for path in outputs:
        require(not path.exists(), f"Stage {STAGE} output already exists: {rel(path)}")

    evidence, metrics = verify_inputs()
    rows = capability_rows(metrics)
    disposition = {
        "policy_version": "CIRCUITSAGE-HMAC-V2.1-FINAL-DISPOSITION-v1",
        "stage": STAGE, "status": "FROZEN",
        "lifecycle": "COMPLETED AND FROZEN CLOSED-CATALOG RESEARCH COMPONENT",
        "selected_candidate_id": SELECTED,
        "scientific_scope": "OpenTitan HMAC; single persistent SA0/SA1; frozen behavior catalog",
        "closed_catalog_research_target": "MET",
        "independent_generalization_target": "NOT_ESTABLISHED",
        "detector_disposition": "ACCEPTED FOR OBSERVABLE DEVIATIONS IN FROZEN HMAC CATALOG",
        "locator_disposition": "ACCEPTED AS AMBIGUITY-AWARE CANDIDATE-SET LOCATOR",
        "exact_site_claim": "AUTHORIZED ONLY FOR UNIQUE CATALOG SIGNATURES",
        "overall_detection_recall": metrics["all_injected_detection_recall"],
        "overall_exact_site_rate": metrics["all_injected_exact_site_rate"],
        "normal_compatible_fault_instances": metrics["normal_compatible_fault_instances"],
        "dev_site_test": "CONSUMED AND FROZEN",
        "retraining_or_threshold_changes": "PROHIBITED UNDER V2.1",
        "validation_access_count": 0, "holdout_access_count": 0,
        "v1_modified": False, "v2_core_modified": False,
        "evaluation_lock": record(EVALUATION_LOCK),
    }
    generalization = {
        "policy_version": "CIRCUITSAGE-HMAC-V2.2-GENERALIZATION-READINESS-v1",
        "stage": STAGE, "status": "FROZEN",
        "v2_2_contract_creation": "AUTHORIZED",
        "v2_2_dataset_construction": "NOT YET AUTHORIZED",
        "v2_2_model_training": "NOT YET AUTHORIZED",
        "dev_site_test_reopening": "PROHIBITED",
        "validation_access": "NOT AUTHORIZED",
        "holdout_access": "NOT AUTHORIZED",
        "required_generalization_axes": [
            "unseen test vectors", "unseen physical sites", "unseen response signatures",
            "unseen fault families", "independent circuits",
        ],
        "required_architecture_elements": [
            "response-only query encoder", "open-set unknown-signature rejection",
            "graph-constrained candidate generation", "ambiguity-aware set prediction",
            "full-catalog hard-negative training", "adaptive discriminating-vector selection",
        ],
        "immutable_comparators": ["CircuitSage-HMAC V1", "V2 Core", "V2.1"],
        "claims_before_independent_evaluation": "RESEARCH EXPERIMENT ONLY",
        "next_authorized_stage": "12C-1A",
    }
    csv_payload = rows_csv(rows)
    registry_json = {
        "registry_version": "CIRCUITSAGE-HMAC-V2.1-CAPABILITY-REGISTRY-v1",
        "stage": STAGE, "status": "FROZEN", "capabilities": rows,
    }
    frozen_write(DISPOSITION_POLICY, canonical_json(disposition))
    frozen_write(GENERALIZATION_POLICY, canonical_json(generalization))
    frozen_write(CAPABILITY_REGISTRY_CSV, csv_payload)
    frozen_write(CAPABILITY_REGISTRY_JSON, canonical_json(registry_json))
    frozen_write(REPORT, report_text(metrics).encode())
    manifest = {
        "manifest_version": "CIRCUITSAGE-HMAC-V2.1-DISPOSITION-MANIFEST-v1",
        "stage": STAGE, "status": "PASS", "input_evidence": evidence,
        "outputs": {rel(path): record(path) for path in (
            DISPOSITION_POLICY, GENERALIZATION_POLICY, CAPABILITY_REGISTRY_CSV,
            CAPABILITY_REGISTRY_JSON, REPORT
        )},
        "model_training_calls": 0, "model_inference_calls": 0,
        "model_objects_deserialized": 0, "dev_site_test_reopened": False,
        "validation_access_count": 0, "holdout_access_count": 0,
    }
    frozen_write(MANIFEST, canonical_json(manifest))
    audit = {
        "audit_version": "CIRCUITSAGE-HMAC-V2.1-DISPOSITION-GENERALIZATION-READINESS-FREEZE-v1",
        "stage": STAGE, "status": "PASS", "disposition_status": "FROZEN",
        "v2_1_lifecycle": "COMPLETED AND FROZEN CLOSED-CATALOG RESEARCH COMPONENT",
        "closed_catalog_research_target": "MET",
        "independent_generalization": "NOT ESTABLISHED",
        "all_injected_detection_recall": metrics["all_injected_detection_recall"],
        "all_injected_exact_site_rate": metrics["all_injected_exact_site_rate"],
        "mean_observable_candidate_sites": metrics["mean_candidate_set_size_observable"],
        "maximum_observable_candidate_sites": metrics["maximum_candidate_set_size_observable"],
        "v2_2_contract_creation": "AUTHORIZED",
        "v2_2_training": "NOT YET AUTHORIZED",
        "dev_site_test": "CONSUMED AND FROZEN / REOPENING PROHIBITED",
        "validation_access_count": 0, "holdout_access_count": 0,
        "v1_modified": False, "v2_core_modified": False, "v2_1_modified": False,
        "model_training_calls": 0, "model_inference_calls": 0,
        "disposition_policy": record(DISPOSITION_POLICY),
        "generalization_policy": record(GENERALIZATION_POLICY),
        "capability_registry_csv": record(CAPABILITY_REGISTRY_CSV),
        "capability_registry_json": record(CAPABILITY_REGISTRY_JSON),
        "report": record(REPORT), "manifest": record(MANIFEST),
        "next_gate": "STAGE 12C-1A — V2.2 GENERALIZATION ARCHITECTURE, DATA-PARTITION, AND ACCEPTANCE-CONTRACT FREEZE",
    }
    frozen_write(AUDIT, canonical_json(audit))

    for path in (DISPOSITION_POLICY, GENERALIZATION_POLICY, CAPABILITY_REGISTRY_JSON, MANIFEST, AUDIT):
        require(canonical_json(load_json(path)) == path.read_bytes(), f"JSON replay: {path.name}")
    require(csv_payload == CAPABILITY_REGISTRY_CSV.read_bytes(), "CSV replay")
    require(report_text(metrics).encode() == REPORT.read_bytes(), "report replay")

    print("\nSTAGE 12B-1G — V2.1 RESULT DISPOSITION AND GENERALIZATION-READINESS FREEZE")
    print(f"{'Status':<46}: PASS")
    print(f"{'Disposition status':<46}: FROZEN")
    print(f"{'V2.1 lifecycle':<46}: COMPLETED AND FROZEN CLOSED-CATALOG RESEARCH COMPONENT")
    print(f"{'Selected candidate':<46}: {SELECTED}")
    print(f"{'Closed-catalog research target':<46}: MET")
    print(f"{'Independent generalization':<46}: NOT ESTABLISHED")
    print(f"{'Observable detection recall':<46}: {float(metrics['observable_detection_recall']):.8f}")
    print(f"{'All-injected detection recall':<46}: {float(metrics['all_injected_detection_recall']):.8f}")
    print(f"{'Unique-signature exact-site accuracy':<46}: {float(metrics['unique_signature_top1_site_accuracy']):.8f}")
    print(f"{'All-injected exact-site rate':<46}: {float(metrics['all_injected_exact_site_rate']):.8f}")
    print(f"{'Mean / maximum observable candidates':<46}: {float(metrics['mean_candidate_set_size_observable']):.4f} / {int(metrics['maximum_candidate_set_size_observable'])}")
    print(f"{'DEV_SITE_TEST':<46}: CONSUMED AND FROZEN")
    print(f"{'VALIDATION / HOLDOUT access':<46}: 0 / 0")
    print(f"{'V2.2 contract creation':<46}: AUTHORIZED")
    print(f"{'V2.2 dataset / training':<46}: NOT YET AUTHORIZED / NOT YET AUTHORIZED")
    print(f"{'V1 / V2 Core / V2.1 modified':<46}: NO / NO / NO")
    print(f"{'Disposition policy':<46}: {DISPOSITION_POLICY}")
    print(f"{'Disposition policy SHA':<46}: {sha256(DISPOSITION_POLICY)}")
    print(f"{'Generalization policy':<46}: {GENERALIZATION_POLICY}")
    print(f"{'Generalization policy SHA':<46}: {sha256(GENERALIZATION_POLICY)}")
    print(f"{'Report':<46}: {REPORT}")
    print(f"{'Report SHA':<46}: {sha256(REPORT)}")
    print(f"{'Manifest':<46}: {MANIFEST}")
    print(f"{'Manifest SHA':<46}: {sha256(MANIFEST)}")
    print(f"{'Audit':<46}: {AUDIT}")
    print(f"{'Audit SHA':<46}: {sha256(AUDIT)}")
    print(f"{'Next gate':<46}: STAGE 12C-1A — V2.2 GENERALIZATION ARCHITECTURE, DATA-PARTITION, AND ACCEPTANCE-CONTRACT FREEZE")


if __name__ == "__main__":
    main()
