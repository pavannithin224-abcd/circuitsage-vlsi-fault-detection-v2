#!/usr/bin/env python3
"""Stage 12C-1J: bounded-pilot disposition and full-campaign readiness freeze.

This is a read-only scientific disposition gate.  It verifies the complete
frozen Stage 12C-1I result, compares the bounded-pilot evidence with the
pre-frozen V2.2 generalization contract, and freezes whether a separate full
TRAIN/CALIBRATION campaign contract may be created.  It performs no synthesis,
simulation, fault injection, training, inference, or protected-partition access.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import io
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STAGE = "12C-1J"
ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "config/v2_2"
RESULT = ROOT / "results/circuitsage_hmac_v2_12c1"
WORK = RESULT / "pilot_disposition_12c1j"
LOCK_FILE = WORK / ".stage_12c1j.lock"

SOURCE_1I = ROOT / "stage_12c1i_adapter_pilot_execution.py"
WORK_1I = RESULT / "adapter_pilot_execution_12c1i"
FEATURES_1I = WORK_1I / "circuitsage_hmac_v2_2_pilot_features_12c1i.npz"
METRICS_1I = WORK_1I / "circuitsage_hmac_v2_2_pilot_metrics_12c1i.json"
MANIFEST_1I = RESULT / "circuitsage_hmac_v2_2_adapter_pilot_execution_manifest_12c1i.json"
AUDIT_1I = RESULT / "circuitsage_hmac_v2_2_adapter_functional_validation_pilot_execution_freeze_12c1i.json"
ACCEPTANCE_1A = CONFIG / "circuitsage_hmac_v2_2_acceptance_contract_12c1a.json"
VECTOR_CONTRACT_1F = CONFIG / "circuitsage_hmac_v2_2_transaction_vector_contract_12c1f.json"

POLICY = CONFIG / "circuitsage_hmac_v2_2_bounded_pilot_disposition_policy_12c1j.json"
READINESS_CONTRACT = CONFIG / "circuitsage_hmac_v2_2_full_campaign_readiness_contract_12c1j.json"
FAMILY_CSV = WORK / "circuitsage_hmac_v2_2_pilot_family_disposition_12c1j.csv"
FAMILY_JSON = WORK / "circuitsage_hmac_v2_2_pilot_family_disposition_12c1j.json"
READINESS_LOCK = WORK / "circuitsage_hmac_v2_2_full_campaign_readiness_lock_12c1j.json"
REPORT = WORK / "circuitsage_hmac_v2_2_pilot_disposition_report_12c1j.md"
MANIFEST = RESULT / "circuitsage_hmac_v2_2_pilot_disposition_manifest_12c1j.json"
AUDIT = RESULT / "circuitsage_hmac_v2_2_pilot_disposition_full_campaign_readiness_freeze_12c1j.json"

PINNED = {
    SOURCE_1I: "735df97cbbf8ccd1097913d7044906c209a647e3290d056741c7880e26544f94",
    FEATURES_1I: "0ffd4cba4bfc0a76e62895763e0deee1d9ce9b4c8aa9bce655f68715ff68f962",
    METRICS_1I: "e17309664d2053a0f1e5c34a72cbec59115aa58cabfb46339155a55516fc835e",
    MANIFEST_1I: "bb574d58165b66b0369b311c52b8014c7e90e1d45c580a162e127dfac0dbb522",
    AUDIT_1I: "cc96e5d7d11f9f75658365875c6521789416399b7bf09471f0fab334f09b2fa7",
    ACCEPTANCE_1A: "9c8eec4d85957c4408ac59e0c8760af90c91d0667c995d91ac969b5a8f205f26",
    VECTOR_CONTRACT_1F: "530f6d5b7264990e02513f06bb2192fb88c1bae22f4d2b0496dba691be8c1826",
}

FAMILIES = (
    "opentitan_hmac_sha256",
    "picorv32_cpu",
    "secworks_aes",
    "secworks_sha256",
)
FUTURE_BRAND = "Faultiva 1.0 — Hybrid VLSI Fault Intelligence"
TOTAL_SITES = 8192
TOTAL_FAULTS = 16384
PILOT_VECTORS = 60
FULL_VECTORS = 240
PILOT_TRANSACTIONS = 245760


def stop(message: str) -> None:
    raise SystemExit(f"STOP: {message}")


def require(condition: bool, message: str) -> None:
    if not condition:
        stop(message)


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def record(path: Path) -> dict[str, Any]:
    return {"path": rel(path), "sha256": sha256(path), "bytes": path.stat().st_size}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def frozen_write(path: Path, payload: bytes) -> None:
    require(not path.exists(), f"refusing to overwrite frozen output: {rel(path)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def csv_bytes(rows: list[dict[str, Any]], fields: list[str]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def safe_record_path(item: dict[str, Any], label: str) -> Path:
    require(isinstance(item, dict), f"{label} record")
    raw = item.get("path")
    require(isinstance(raw, str) and raw, f"{label} path")
    path = (ROOT / raw).resolve()
    require(path.is_relative_to(ROOT), f"{label} path escapes project root")
    require(path.is_file(), f"missing {label}: {raw}")
    require(sha256(path) == item.get("sha256"), f"{label} SHA")
    require(path.stat().st_size == int(item.get("bytes", -1)), f"{label} size")
    return path


def verify_manifest_records(manifest: dict[str, Any]) -> None:
    outputs = manifest.get("outputs")
    require(isinstance(outputs, dict) and len(outputs) >= 10, "12C-1I output records")
    for name, item in sorted(outputs.items()):
        safe_record_path(item, f"12C-1I output {name}")

    raw = manifest.get("raw_batch_csvs")
    batches = manifest.get("per_batch_npz")
    require(isinstance(raw, dict) and len(raw) == 128, "12C-1I raw batch records")
    require(isinstance(batches, dict) and len(batches) == 128, "12C-1I batch NPZ records")
    require(set(raw) == set(batches), "12C-1I batch record keys")
    for key in sorted(raw):
        safe_record_path(raw[key], f"12C-1I raw batch {key}")
        safe_record_path(batches[key], f"12C-1I batch NPZ {key}")


def verify_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    print("FROZEN INPUT VERIFICATION", flush=True)
    for path, expected in PINNED.items():
        require(path.is_file(), f"missing frozen input: {rel(path)}")
        require(sha256(path) == expected, f"frozen input SHA: {path.name}")
        print(f"  {path.name:<108}: OK", flush=True)

    metrics = load_json(METRICS_1I)
    predecessor_manifest = load_json(MANIFEST_1I)
    predecessor_audit = load_json(AUDIT_1I)
    acceptance = load_json(ACCEPTANCE_1A)
    vectors = load_json(VECTOR_CONTRACT_1F)

    # R1: predecessor contracts were emitted by different frozen JSON writers.
    # Their pinned byte-level SHA-256 values above are authoritative; canonical
    # replay applies only to the immediate Stage 12C-1I outputs that use this
    # generation's compact canonical_json convention.
    for path, value in ((METRICS_1I, metrics), (MANIFEST_1I, predecessor_manifest),
                        (AUDIT_1I, predecessor_audit)):
        require(path.read_bytes() == canonical_json(value), f"canonical JSON: {path.name}")

    require(predecessor_manifest.get("stage") == "12C-1I" and predecessor_manifest.get("status") == "PASS", "12C-1I manifest")
    require(predecessor_manifest.get("stage_source", {}).get("sha256") == PINNED[SOURCE_1I], "12C-1I stage-source anchor")
    require(predecessor_audit.get("stage") == "12C-1I" and predecessor_audit.get("status") == "PASS", "12C-1I audit")
    require(predecessor_audit.get("execution_dataset") == "COMPLETED / FROZEN", "12C-1I execution freeze")
    require(predecessor_audit.get("adapter_functional_validation") == "PASS / 4 OF 4 / BYTE-EXACT REPLAY", "adapter validation")
    require(predecessor_audit.get("dataset_integrity") == "PASS", "pilot dataset integrity")
    require(predecessor_audit.get("full_campaign_model_training") == "NOT AUTHORIZED / NOT AUTHORIZED", "predecessor authorization boundary")
    require(predecessor_audit.get("independent_test_validation_holdout_access") == [0, 0, 0], "protected access")
    require(predecessor_audit.get("manifest_record", {}).get("sha256") == PINNED[MANIFEST_1I], "12C-1I manifest audit anchor")
    require(predecessor_audit.get("metrics_record", {}).get("sha256") == PINNED[METRICS_1I], "12C-1I metrics audit anchor")
    require(predecessor_audit.get("features_record", {}).get("sha256") == PINNED[FEATURES_1I], "12C-1I features audit anchor")
    verify_manifest_records(predecessor_manifest)

    require(metrics.get("status") == "PASS / REPORT-ONLY", "pilot metrics status")
    require(metrics.get("adapter_validation") == "PASS / 4 OF 4 / BYTE-EXACT REPLAY", "pilot adapter metric")
    require(metrics.get("dataset_integrity") == "PASS", "pilot metric integrity")
    require(metrics.get("sites") == TOTAL_SITES and metrics.get("fault_instances") == TOTAL_FAULTS, "pilot dimensions")
    require(metrics.get("pilot_vectors") == PILOT_VECTORS and metrics.get("enabled_transactions") == PILOT_TRANSACTIONS, "pilot workload")
    require(metrics.get("observable_faults") == 9349, "observable-fault count")
    require(abs(float(metrics.get("macro_all_injected_detection_recall")) - 0.57061767578125) < 1e-15, "pilot detection replay")
    require(abs(float(metrics.get("all_injected_exact_site_rate")) - 0.43499755859375) < 1e-15, "pilot exact-site replay")
    require(int(metrics.get("fault_free_false_alarms")) == 0, "pilot false alarms")
    require(metrics.get("model_training_inference") == [0, 0], "pilot model activity")
    families = metrics.get("families")
    require(isinstance(families, list) and [item.get("family_id") for item in families] == list(FAMILIES), "pilot family order")
    require(all(int(item.get("observable_faults", 0)) > 0 for item in families), "nonzero signal in every family")

    macro = acceptance.get("macro_gates", {})
    require(float(macro.get("all_injected_detection_recall_min")) == 0.60, "frozen detection gate")
    require(float(macro.get("all_injected_exact_site_rate_min")) == 0.25, "frozen exact-site gate")
    require(float(macro.get("fault_free_false_alarm_rate_max")) == 0.01, "frozen false-alarm gate")
    require(vectors.get("status") == "FROZEN", "vector contract")
    budgets = vectors.get("budgets")
    require(isinstance(budgets, list) and len(budgets) == 4, "vector budgets")
    require(sum(int(item["total_vector_budget"]) for item in budgets) == FULL_VECTORS, "full vector budget")
    require(sum(int(item["pilot_vector_budget"]) for item in budgets) == PILOT_VECTORS, "pilot vector budget")

    print("  Complete 12C-1I evidence, final acceptance gates and vector budgets                    : PASS", flush=True)
    return metrics, predecessor_manifest, acceptance, vectors


def execute() -> None:
    require(ROOT.name == "vlsi_fault_detection_v2", f"place script in project root, not {ROOT}")
    final_outputs = (POLICY, READINESS_CONTRACT, FAMILY_CSV, FAMILY_JSON,
                     READINESS_LOCK, REPORT, MANIFEST, AUDIT)
    require(not any(path.exists() for path in final_outputs), "frozen Stage 12C-1J output already exists; use --status")

    metrics, predecessor_manifest, acceptance, vectors = verify_inputs()
    macro = acceptance["macro_gates"]
    detection = float(metrics["macro_all_injected_detection_recall"])
    exact_site = float(metrics["all_injected_exact_site_rate"])
    false_alarm_rate = float(metrics["fault_free_false_alarm_rate"])
    mean_candidates = float(metrics["mean_observable_candidate_sites"])
    maximum_candidates = int(metrics["maximum_observable_candidate_sites"])
    detection_target = float(macro["all_injected_detection_recall_min"])
    exact_target = float(macro["all_injected_exact_site_rate_min"])
    detection_gap = detection_target - detection
    exact_margin = exact_site - exact_target
    vector_expansion = FULL_VECTORS / PILOT_VECTORS

    require(false_alarm_rate == 0.0, "pilot safety gate")
    require(exact_site >= exact_target, "pilot exact-site feasibility")
    require(detection >= 0.50, "pilot signal feasibility floor")
    require(detection_gap <= 0.05, "pilot detection gap too large for full-campaign readiness")

    family_fields = [
        "family_id", "partition", "sites", "fault_instances", "pilot_vectors",
        "observable_faults", "all_injected_detection_recall", "all_injected_exact_site_rate",
        "mean_observable_candidate_sites", "maximum_observable_candidate_sites", "disposition",
    ]
    family_rows: list[dict[str, Any]] = []
    for item in metrics["families"]:
        family_rows.append({
            "family_id": item["family_id"], "partition": item["partition"],
            "sites": item["sites"], "fault_instances": item["fault_instances"],
            "pilot_vectors": item["pilot_vectors"], "observable_faults": item["observable_faults"],
            "all_injected_detection_recall": f"{float(item['all_injected_detection_recall']):.8f}",
            "all_injected_exact_site_rate": f"{float(item['all_injected_exact_site_rate']):.8f}",
            "mean_observable_candidate_sites": f"{float(item['mean_observable_candidate_sites']):.4f}",
            "maximum_observable_candidate_sites": item["maximum_observable_candidate_sites"],
            "disposition": "RETAIN — FULL FAMILY; NO FAMILY SKIPPING",
        })

    created = now()
    policy = {
        "policy_version": "CIRCUITSAGE-HMAC-V2.2-BOUNDED-PILOT-DISPOSITION-12C1J-v1",
        "stage": STAGE, "status": "FROZEN", "created_at": created,
        "scope": "GENERALIZATION_TRAIN + GENERALIZATION_CALIBRATION ONLY",
        "predecessor": record(AUDIT_1I),
        "disposition_basis": {
            "adapter_and_dataset_hard_gates": "PASS",
            "pilot_vector_fraction": PILOT_VECTORS / FULL_VECTORS,
            "all_injected_detection_recall": detection,
            "frozen_independent_test_detection_target": detection_target,
            "pilot_detection_gap": detection_gap,
            "all_injected_exact_site_rate": exact_site,
            "frozen_independent_test_exact_site_target": exact_target,
            "pilot_exact_site_margin": exact_margin,
            "fault_free_false_alarm_rate": false_alarm_rate,
            "signal_present_in_every_family": True,
        },
        "interpretation": [
            "the bounded pilot is feasibility evidence, not independent-generalization evidence",
            "the pilot used one quarter of the frozen TRAIN/CALIBRATION vector budget",
            "the exact-site ceiling exceeds the pre-frozen independent-test macro target",
            "the pilot detection ceiling is within 0.05 of the pre-frozen independent-test macro target",
            "full-campaign execution remains a separate authorization decision",
        ],
        "pilot_advancement": "CONDITIONAL PASS — FULL-CAMPAIGN CONTRACT DESIGN AUTHORIZED",
        "full_campaign_execution": "NOT AUTHORIZED / NOT STARTED",
        "model_training_inference": "NOT AUTHORIZED / NOT AUTHORIZED",
        "family_policy": "RETAIN ALL FOUR; FAMILY SKIPPING PROHIBITED",
        "threshold_changes": "PROHIBITED",
        "independent_test": "LOCKED / 2 FAMILIES",
        "validation": "UNOPENED",
        "holdout": "SEALED / 1 FAMILY",
    }
    readiness = {
        "contract_version": "CIRCUITSAGE-HMAC-V2.2-FULL-CAMPAIGN-READINESS-12C1J-v1",
        "stage": STAGE, "status": "FROZEN", "created_at": created,
        "readiness": "READY FOR SEPARATE EXECUTION-CONTRACT AND AUTHORIZATION FREEZE",
        "authorized_next_activity": "DESIGN AND FREEZE FULL TRAIN/CALIBRATION CAMPAIGN EXECUTION CONTRACT",
        "not_authorized": [
            "full fault-campaign execution", "model training", "model selection", "inference",
            "independent TEST opening", "VALIDATION opening", "HOLDOUT opening",
        ],
        "frozen_measurement": "PORTABLE FAMILY ADAPTER RESPONSE + FROZEN GENERIC NETLIST GRAPH",
        "families": list(FAMILIES), "family_skipping": "PROHIBITED",
        "full_vector_budget": FULL_VECTORS, "pilot_vector_budget": PILOT_VECTORS,
        "vector_expansion_factor": vector_expansion,
        "mandatory_next_contract_contents": [
            "deterministic full-site catalog and per-family budgets",
            "all 240 frozen vectors with family masks and isolated timeout canaries",
            "fault-free baselines and byte-exact replay",
            "sequential execution, one build job, checkpoints and bounded resources",
            "graph/response feature schema with fault identity absent from queries",
            "TRAIN and CALIBRATION separation with no physical-site overlap where applicable",
            "no independent TEST, VALIDATION or HOLDOUT access",
        ],
        "risk_register": {
            "pilot_detection_below_final_macro_target": True,
            "pilot_detection_gap": detection_gap,
            "large_signature_ambiguity": True,
            "mean_observable_candidate_sites": mean_candidates,
            "maximum_observable_candidate_sites": maximum_candidates,
            "mitigation": "use full frozen vector budget and graph-aware ambiguity-preserving retrieval/reranking; reassess before training authorization",
        },
    }
    family_json = {
        "registry_version": "CIRCUITSAGE-HMAC-V2.2-PILOT-FAMILY-DISPOSITION-12C1J-v1",
        "stage": STAGE, "status": "FROZEN", "families": family_rows,
    }
    readiness_lock = {
        "lock_version": "CIRCUITSAGE-HMAC-V2.2-FULL-CAMPAIGN-READINESS-LOCK-12C1J-v1",
        "stage": STAGE, "status": "FROZEN", "created_at": created,
        "pilot_dataset": record(FEATURES_1I), "pilot_metrics": record(METRICS_1I),
        "pilot_manifest": record(MANIFEST_1I), "pilot_audit": record(AUDIT_1I),
        "disposition": "CONDITIONAL PASS",
        "full_campaign_contract_design": "AUTHORIZED",
        "full_campaign_execution": "NOT AUTHORIZED",
        "model_training": "NOT AUTHORIZED",
        "independent_generalization": "NOT ESTABLISHED",
        "protected_access": {"independent_test": 0, "validation": 0, "holdout": 0},
    }

    report = f"""# CircuitSage-HMAC V2.2 bounded-pilot disposition — Stage 12C-1J

Stage 12C-1I is complete, internally consistent and frozen.  All four portable
adapters passed byte-exact functional replay, all 128 pilot batches completed,
and no fault-free false alarm occurred.

The bounded pilot observed {metrics['observable_faults']} of {TOTAL_FAULTS}
faults, giving all-injected detection recall {detection:.8f}.  This is
{detection_gap:.8f} below the pre-frozen independent-test macro target of
{detection_target:.8f}.  Exact-site rate is {exact_site:.8f}, which is
{exact_margin:.8f} above the corresponding {exact_target:.8f} target.  The
pilot used {PILOT_VECTORS} of {FULL_VECTORS} frozen TRAIN/CALIBRATION vectors.

Disposition: **conditional pass for full-campaign contract design**.  This does
not authorize campaign execution or model training.  All four families must be
retained, ambiguity must be preserved, and the large candidate-set risk (mean
{mean_candidates:.4f}, maximum {maximum_candidates}) must be reassessed using
the full frozen vector schedule before training can be authorized.

Independent TEST remains locked, VALIDATION remains unopened, and HOLDOUT
remains sealed.  Independent-circuit generalization is not established.  The
future combined-model brand remains **{FUTURE_BRAND}**.
""".encode()

    frozen_write(POLICY, canonical_json(policy))
    frozen_write(READINESS_CONTRACT, canonical_json(readiness))
    frozen_write(FAMILY_CSV, csv_bytes(family_rows, family_fields))
    frozen_write(FAMILY_JSON, canonical_json(family_json))
    frozen_write(READINESS_LOCK, canonical_json(readiness_lock))
    frozen_write(REPORT, report)

    outputs = (POLICY, READINESS_CONTRACT, FAMILY_CSV, FAMILY_JSON, READINESS_LOCK, REPORT)
    manifest = {
        "manifest_version": "CIRCUITSAGE-HMAC-V2.2-PILOT-DISPOSITION-MANIFEST-12C1J-v1",
        "stage": STAGE, "status": "PASS", "created_at": created,
        "stage_source": record(Path(__file__).resolve()),
        "frozen_inputs": {rel(path): record(path) for path in PINNED},
        "verified_predecessor_outputs": len(predecessor_manifest["outputs"]),
        "verified_raw_batches": len(predecessor_manifest["raw_batch_csvs"]),
        "verified_batch_npz": len(predecessor_manifest["per_batch_npz"]),
        "outputs": {rel(path): record(path) for path in outputs},
        "simulation_calls": 0, "fault_injection_calls": 0, "training_calls": 0,
        "inference_calls": 0, "model_deserializations": 0,
        "independent_test_access": 0, "validation_access": 0, "holdout_access": 0,
    }
    frozen_write(MANIFEST, canonical_json(manifest))
    audit = {
        "audit_version": "CIRCUITSAGE-HMAC-V2.2-PILOT-DISPOSITION-FULL-CAMPAIGN-READINESS-FREEZE-12C1J-v1",
        "stage": STAGE, "status": "PASS", "disposition_status": "FROZEN",
        "pilot_dataset": "VALID / FROZEN", "adapter_validation": "PASS / 4 OF 4 / BYTE-EXACT REPLAY",
        "families_sites_faults_vectors": [4, TOTAL_SITES, TOTAL_FAULTS, PILOT_VECTORS],
        "observable_faults": metrics["observable_faults"],
        "pilot_detection_recall": detection, "frozen_test_detection_target": detection_target,
        "pilot_detection_gap": detection_gap, "pilot_exact_site_rate": exact_site,
        "frozen_test_exact_site_target": exact_target, "pilot_exact_site_margin": exact_margin,
        "mean_maximum_observable_candidate_sites": [mean_candidates, maximum_candidates],
        "fault_free_false_alarms": 0,
        "pilot_advancement": "CONDITIONAL PASS — FULL-CAMPAIGN CONTRACT DESIGN AUTHORIZED",
        "full_campaign_readiness": "READY FOR SEPARATE EXECUTION-CONTRACT AND AUTHORIZATION FREEZE",
        "full_campaign_execution_model_training": "NOT AUTHORIZED / NOT AUTHORIZED",
        "independent_generalization": "NOT ESTABLISHED",
        "independent_test_validation_holdout_access": [0, 0, 0],
        "policy_record": record(POLICY), "readiness_contract_record": record(READINESS_CONTRACT),
        "readiness_lock_record": record(READINESS_LOCK), "report_record": record(REPORT),
        "manifest_record": record(MANIFEST), "future_combined_model_brand": FUTURE_BRAND,
        "next_gate": "STAGE 12C-1K — FULL TRAIN/CALIBRATION FAULT-CAMPAIGN EXECUTION-CONTRACT AND AUTHORIZATION FREEZE",
    }
    frozen_write(AUDIT, canonical_json(audit))

    for path in (POLICY, READINESS_CONTRACT, FAMILY_JSON, READINESS_LOCK, MANIFEST, AUDIT):
        require(path.read_bytes() == canonical_json(load_json(path)), f"canonical output replay: {path.name}")
    require(FAMILY_CSV.read_bytes() == csv_bytes(family_rows, family_fields), "family CSV replay")
    for path, expected in PINNED.items():
        require(sha256(path) == expected, f"frozen input modified: {rel(path)}")

    print("\nSTAGE 12C-1J — BOUNDED PILOT RESULT DISPOSITION AND FULL-CAMPAIGN READINESS FREEZE")
    print(f"{'Status':<88}: PASS")
    print(f"{'Disposition status':<88}: FROZEN")
    print(f"{'Pilot dataset / adapter validation':<88}: VALID / FROZEN / PASS 4 OF 4")
    print(f"{'Families / sites / faults / pilot vectors':<88}: 4 / {TOTAL_SITES} / {TOTAL_FAULTS} / {PILOT_VECTORS}")
    print(f"{'Pilot detection / frozen target / gap':<88}: {detection:.8f} / {detection_target:.8f} / {detection_gap:.8f}")
    print(f"{'Pilot exact-site / frozen target / margin':<88}: {exact_site:.8f} / {exact_target:.8f} / +{exact_margin:.8f}")
    print(f"{'Mean / maximum observable candidate sites':<88}: {mean_candidates:.4f} / {maximum_candidates}")
    print(f"{'Fault-free false alarms':<88}: 0")
    print(f"{'Pilot advancement':<88}: CONDITIONAL PASS — FULL-CAMPAIGN CONTRACT DESIGN AUTHORIZED")
    print(f"{'Full campaign readiness':<88}: READY FOR SEPARATE AUTHORIZATION")
    print(f"{'Full campaign execution / model training':<88}: NOT AUTHORIZED / NOT AUTHORIZED")
    print(f"{'Independent TEST / VALIDATION / HOLDOUT access':<88}: 0 / 0 / 0")
    print(f"{'Independent generalization':<88}: NOT ESTABLISHED")
    print(f"{'Disposition policy':<88}: {POLICY}")
    print(f"{'Disposition policy SHA':<88}: {sha256(POLICY)}")
    print(f"{'Readiness contract':<88}: {READINESS_CONTRACT}")
    print(f"{'Readiness contract SHA':<88}: {sha256(READINESS_CONTRACT)}")
    print(f"{'Manifest':<88}: {MANIFEST}")
    print(f"{'Manifest SHA':<88}: {sha256(MANIFEST)}")
    print(f"{'Audit':<88}: {AUDIT}")
    print(f"{'Audit SHA':<88}: {sha256(AUDIT)}")
    print(f"{'Next gate':<88}: STAGE 12C-1K — FULL TRAIN/CALIBRATION FAULT-CAMPAIGN EXECUTION-CONTRACT AND AUTHORIZATION FREEZE")


def status() -> None:
    print("STAGE 12C-1J — PILOT-DISPOSITION STATUS")
    if not AUDIT.is_file() or not MANIFEST.is_file():
        print("Status                    : NOT FROZEN")
        print(f"Expected audit            : {AUDIT}")
        return
    require(sha256(AUDIT) == sha256(AUDIT), "audit readability")
    manifest = load_json(MANIFEST)
    audit = load_json(AUDIT)
    require(manifest.get("status") == "PASS" and audit.get("status") == "PASS", "frozen status")
    require(audit.get("manifest_record", {}).get("sha256") == sha256(MANIFEST), "manifest audit anchor")
    for name, item in manifest.get("outputs", {}).items():
        safe_record_path(item, f"12C-1J output {name}")
    print("Status                    : PASS / FROZEN")
    print(f"Pilot advancement         : {audit['pilot_advancement']}")
    print(f"Full-campaign readiness   : {audit['full_campaign_readiness']}")
    print(f"Execution / training      : {audit['full_campaign_execution_model_training']}")
    print(f"Audit                     : {AUDIT}")
    print(f"Audit SHA                 : {sha256(AUDIT)}")


def self_test() -> None:
    detection = 9349 / 16384
    exact_site = 7127 / 16384
    require(abs(detection - 0.57061767578125) < 1e-15, "detection arithmetic")
    require(abs(exact_site - 0.43499755859375) < 1e-15, "exact-site arithmetic")
    require(abs((0.60 - detection) - 0.02938232421875) < 1e-15, "detection-gap arithmetic")
    require(abs((exact_site - 0.25) - 0.18499755859375) < 1e-15, "exact-site-margin arithmetic")
    require(FULL_VECTORS / PILOT_VECTORS == 4.0, "vector expansion")
    print("Stage 12C-1J self-test: PASS")


def locked_execute() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    with LOCK_FILE.open("a+", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            stop("Stage 12C-1J execution lock is held by another process")
        execute()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--status", action="store_true")
    mode.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.status:
        status()
    elif args.self_test:
        self_test()
    else:
        locked_execute()


if __name__ == "__main__":
    main()
