#!/usr/bin/env python3
"""Stage 12B-2D: adaptive vector selection and full-campaign contract freeze.

Uses only the frozen REPAIR_TRAIN pilot dataset from Stage 12B-2C. Selects at
most 96 vectors in four deterministic rounds, measures detection coverage and
closed-pilot signature ambiguity, and freezes the full repair-campaign
disposition. If the complete 512-vector pool cannot reach the frozen detection
target, the full campaign remains blocked. No simulation or model operation is
performed and no protected partition is accessed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import sys
import zipfile
from pathlib import Path
from typing import Any

try:
    import numpy as np
except ImportError as error:
    raise SystemExit("STOP: activate the project .venv (NumPy required)") from error


STAGE = "12B-2D"
ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "config/v2_1_improvement"
RESULT = ROOT / "results/circuitsage_hmac_v2_12b2"
WORK = RESULT / "adaptive_vector_selection_12b2d"

SOURCE_2C = ROOT / "stage_12b2c_adaptive_vector_pilot_screen.py"
AUDIT_2C = RESULT / "circuitsage_hmac_v2_1_pilot_screening_dataset_integrity_freeze_12b2c.json"
MANIFEST_2C = RESULT / "circuitsage_hmac_v2_1_pilot_dataset_manifest_12b2c.json"
DATASET = RESULT / "pilot_screening_12b2c/circuitsage_hmac_v2_1_pilot_response_dataset_12b2c.npz"
DATASET_SCHEMA = RESULT / "pilot_screening_12b2c/circuitsage_hmac_v2_1_pilot_response_schema_12b2c.json"
PILOT_METRICS = RESULT / "pilot_screening_12b2c/circuitsage_hmac_v2_1_candidate_vector_pilot_metrics_12b2c.csv"

SOURCE_2B = ROOT / "stage_12b2b_adaptive_vector_pilot_authorization.py"
VECTOR_POOL = RESULT / "adaptive_vector_pool_12b2b/circuitsage_hmac_v2_1_candidate_vectors_12b2b.npz"
PILOT_SITES = RESULT / "adaptive_vector_pool_12b2b/circuitsage_hmac_v2_1_pilot_screening_sites_12b2b.csv"
AUDIT_2B = RESULT / "circuitsage_hmac_v2_1_pilot_authorization_freeze_12b2b.json"

VECTOR_POLICY = CONFIG / "circuitsage_hmac_v2_1_adaptive_vector_policy_12b2a.json"
ACCEPTANCE_2A = CONFIG / "circuitsage_hmac_v2_1_repair_acceptance_contract_12b2a.json"

SELECTED_NPZ = WORK / "circuitsage_hmac_v2_1_selected_adaptive_vectors_12b2d.npz"
SELECTION_TRACE = WORK / "circuitsage_hmac_v2_1_adaptive_vector_selection_trace_12b2d.csv"
SELECTION_METRICS = WORK / "circuitsage_hmac_v2_1_adaptive_vector_selection_metrics_12b2d.json"
SELECTION_LOCK = WORK / "circuitsage_hmac_v2_1_adaptive_vector_selection_lock_12b2d.json"
FULL_CAMPAIGN_CONTRACT = CONFIG / "circuitsage_hmac_v2_1_full_repair_campaign_contract_12b2d.json"
MANIFEST = RESULT / "circuitsage_hmac_v2_1_adaptive_selection_manifest_12b2d.json"
AUDIT = RESULT / "circuitsage_hmac_v2_1_adaptive_selection_campaign_contract_freeze_12b2d.json"

PINNED = {
    SOURCE_2C: "88084ec6cf03ffc60b9b5c7d42a895347bdd5702c08dc97c56e758dc91f2efaa",
    AUDIT_2C: "148856642c8968a7d6263cfa18fc6c639b25a6572441724466005f5fa67df268",
    SOURCE_2B: "912878bd37c9ed7e0ca6cb88f31afe5bff5b4715f62579f152842011fe56c0b3",
    VECTOR_POOL: "47dd8944b9519465df19f098c2fb2a6a8823a8736452992cf52c73efcc435d78",
    PILOT_SITES: "87eceaf80a800cf60b8a7e3cb20e0ec3f934158dc35ae3d10f73e8cf9d25313c",
    AUDIT_2B: "95f49d71efcabc0d56673dd389ee50c9e474cee1697b5e2cb039fc2da362bc80",
}

VECTOR_COUNT = 512
MAX_SELECTED = 96
ROUNDS = 4
PER_ROUND = 24
PILOT_SITES_COUNT = 1024
FAULTS = 2048


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


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode()


def load_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing JSON: {rel(path)}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON object required: {rel(path)}")
    return value


def frozen_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.exists(), f"refusing to overwrite frozen output: {rel(path)}")
    temporary = path.with_name(path.name + ".tmp")
    require(not temporary.exists(), f"stale temporary output: {rel(temporary)}")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def record(path: Path) -> dict[str, Any]:
    return {"path": rel(path), "sha256": sha256(path), "bytes": path.stat().st_size}


def resolve_record(item: dict[str, Any]) -> Path:
    value = item.get("path")
    require(isinstance(value, str) and value, "artifact record path")
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def verify_record(item: dict[str, Any], expected: Path, label: str) -> None:
    path = resolve_record(item)
    require(path.resolve() == expected.resolve(), f"{label} path")
    require(path.is_file(), f"missing {label}: {rel(path)}")
    require(item.get("sha256") == sha256(path), f"{label} SHA")
    require(int(item.get("bytes", -1)) == path.stat().st_size, f"{label} size")


def verify_inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    print("STAGE 12B-2D — ADAPTIVE VECTOR SELECTION AND FULL REPAIR-CAMPAIGN CONTRACT")
    print("FROZEN INPUT VERIFICATION")
    for path, expected in PINNED.items():
        require(path.is_file(), f"missing frozen input: {rel(path)}")
        require(sha256(path) == expected, f"frozen input SHA: {rel(path)}")
        print(f"  {path.name:<91}: OK", flush=True)
    audit = load_json(AUDIT_2C)
    require(audit.get("status") == "PASS", "12B-2C status")
    require(audit.get("pilot_execution_status") == "COMPLETED / FROZEN", "pilot execution state")
    require(audit.get("dataset_status") == "FROZEN", "pilot dataset state")
    require(audit.get("batches_verified") == "45/45", "pilot batch verification")
    require(audit.get("candidate_vectors") == VECTOR_COUNT, "candidate-vector count")
    require(audit.get("pilot_sites") == PILOT_SITES_COUNT and audit.get("fault_instances") == FAULTS, "pilot dimensions")
    require(audit.get("missing_duplicate_samples") == [0, 0], "pilot sample integrity")
    require(audit.get("unknown_records") == 0 and audit.get("detected_without_activity") == 0, "pilot semantic integrity")
    require(audit.get("adaptive_vector_selection") == "AUTHORIZED / NOT STARTED", "selection authorization")
    require(audit.get("model_training") == "NOT AUTHORIZED", "model-training boundary")
    require(audit.get("repair_site_test") == "LOCKED / NOT AUTHORIZED", "repair-test boundary")
    require(audit.get("validation_access") == 0 and audit.get("holdout_access") == 0, "protected partition access")
    verify_record(audit["dataset"], DATASET, "pilot dataset")
    verify_record(audit["metrics"], PILOT_METRICS, "pilot metrics")
    verify_record(audit["manifest"], MANIFEST_2C, "pilot manifest")
    manifest = load_json(MANIFEST_2C)
    require(manifest.get("status") == "PASS", "pilot manifest status")
    outputs = manifest.get("outputs")
    require(isinstance(outputs, dict), "pilot manifest outputs")
    verify_record(outputs[rel(DATASET_SCHEMA)], DATASET_SCHEMA, "pilot schema")
    acceptance = load_json(ACCEPTANCE_2A)
    policy = load_json(VECTOR_POLICY)
    require(acceptance.get("status") == "FROZEN", "acceptance status")
    require(policy.get("status") == "FROZEN", "vector-policy status")
    require(policy.get("maximum_selected_vectors") == MAX_SELECTED, "selected-vector budget")
    require(policy.get("selection_rounds") == ROUNDS and policy.get("vectors_per_round") == PER_ROUND, "selection schedule")
    require(policy.get("adaptive_feedback_partition") == "REPAIR_TRAIN ONLY", "selection feedback partition")
    require(policy.get("repair_calibration_feedback") == "PROHIBITED FOR VECTOR SELECTION", "calibration blinding")
    return acceptance, policy


def load_data() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    with np.load(DATASET, allow_pickle=False) as archive:
        required = {
            "activity", "baseline_cycles", "cycles", "detected", "digest_xor",
            "fault_instance_index", "pilot_rank", "site_index", "stuck_value",
            "timed_out", "vector_index",
        }
        require(set(archive.files) == required, "pilot dataset members")
        data = {name: np.asarray(archive[name]).copy() for name in required}
    require(data["detected"].shape == (FAULTS, VECTOR_COUNT), "detected shape")
    require(data["activity"].shape == (FAULTS, VECTOR_COUNT), "activity shape")
    require(data["cycles"].shape == (FAULTS, VECTOR_COUNT), "cycles shape")
    require(data["digest_xor"].shape == (FAULTS, VECTOR_COUNT, 32), "digest shape")
    require(np.array_equal(data["fault_instance_index"], np.arange(FAULTS)), "fault ordering")
    require(np.array_equal(data["vector_index"], np.arange(VECTOR_COUNT)), "vector ordering")
    require(np.array_equal(data["stuck_value"], np.tile([0, 1], PILOT_SITES_COUNT)), "stuck ordering")
    require(np.array_equal(data["pilot_rank"], np.repeat(np.arange(PILOT_SITES_COUNT), 2)), "pilot-rank ordering")
    with np.load(VECTOR_POOL, allow_pickle=False) as archive:
        require(set(archive.files) == {"vector_index", "key_u8", "message_u8"}, "vector-pool members")
        vectors = {name: np.asarray(archive[name]).copy() for name in archive.files}
    require(np.array_equal(vectors["vector_index"], np.arange(VECTOR_COUNT)), "vector-pool ordering")
    return data, vectors


def response_codes(data: dict[str, np.ndarray]) -> np.ndarray:
    """Dense per-vector response IDs; 0 always means no observed deviation."""
    detected = data["detected"].astype(bool)
    codes = np.zeros((FAULTS, VECTOR_COUNT), dtype=np.uint16)
    for vector in range(VECTOR_COUNT):
        active_faults = np.flatnonzero(detected[:, vector])
        if not len(active_faults):
            continue
        rows = np.concatenate([
            data["timed_out"][active_faults, vector, None].astype(np.uint8),
            data["cycles"][active_faults, vector, None].astype("<u2").view(np.uint8).reshape(-1, 2),
            data["digest_xor"][active_faults, vector].astype(np.uint8),
        ], axis=1)
        _, inverse = np.unique(rows, axis=0, return_inverse=True)
        require(int(np.max(inverse, initial=0)) < np.iinfo(np.uint16).max, "response-code range")
        codes[active_faults, vector] = inverse.astype(np.uint16) + 1
    return codes


def update_groups(groups: np.ndarray, vector_codes: np.ndarray) -> np.ndarray:
    pairs = np.column_stack((groups, vector_codes))
    _, inverse = np.unique(pairs, axis=0, return_inverse=True)
    return inverse.astype(np.int32)


def signature_metrics(groups: np.ndarray, visible: np.ndarray, site_index: np.ndarray) -> dict[str, Any]:
    candidate_sizes = np.zeros(FAULTS, dtype=np.int32)
    unique_site = np.zeros(FAULTS, dtype=bool)
    for group in np.unique(groups[visible]):
        members = np.flatnonzero(visible & (groups == group))
        sites = np.unique(site_index[members])
        candidate_sizes[members] = len(sites)
        unique_site[members] = len(sites) == 1
    visible_count = int(np.sum(visible))
    return {
        "observable_fault_instances": visible_count,
        "all_injected_detection_recall": visible_count / FAULTS,
        "observable_candidate_set_coverage": 1.0 if visible_count else 0.0,
        "unique_signature_fault_instances": int(np.sum(unique_site)),
        "unique_signature_top1_site": 1.0 if int(np.sum(unique_site)) else 0.0,
        "all_injected_exact_site_rate": float(np.sum(unique_site)) / FAULTS,
        "mean_observable_candidate_sites": float(np.mean(candidate_sizes[visible])) if visible_count else 0.0,
        "maximum_observable_candidate_sites": int(np.max(candidate_sizes[visible])) if visible_count else 0,
        "observable_signature_groups": int(len(np.unique(groups[visible]))) if visible_count else 0,
    }


def greedy_select(data: dict[str, np.ndarray], codes: np.ndarray) -> tuple[list[int], list[dict[str, Any]], dict[str, Any]]:
    detected = data["detected"].astype(bool)
    site_index = data["site_index"].astype(np.int32)
    visible = np.zeros(FAULTS, dtype=bool)
    groups = np.zeros(FAULTS, dtype=np.int32)
    selected: list[int] = []
    remaining = set(range(VECTOR_COUNT))
    trace: list[dict[str, Any]] = []
    previous_group_count = 1
    for selection_rank in range(MAX_SELECTED):
        best_vector = -1
        best_key: tuple[int, int, int, int] | None = None
        best_groups: np.ndarray | None = None
        for vector in sorted(remaining):
            new_detections = int(np.sum(detected[:, vector] & ~visible))
            candidate_groups = update_groups(groups, codes[:, vector])
            group_gain = int(len(np.unique(candidate_groups[visible | detected[:, vector]]))) - previous_group_count
            individual_detections = int(np.sum(detected[:, vector]))
            score = (new_detections, group_gain, individual_detections, -vector)
            if best_key is None or score > best_key:
                best_key = score
                best_vector = vector
                best_groups = candidate_groups
        require(best_vector >= 0 and best_groups is not None and best_key is not None, "greedy selection candidate")
        selected.append(best_vector)
        remaining.remove(best_vector)
        visible |= detected[:, best_vector]
        groups = best_groups
        previous_group_count = int(len(np.unique(groups[visible]))) if np.any(visible) else 1
        metrics = signature_metrics(groups, visible, site_index)
        trace.append({
            "selection_rank": selection_rank,
            "selection_round": selection_rank // PER_ROUND + 1,
            "rank_within_round": selection_rank % PER_ROUND,
            "vector_index": best_vector,
            "newly_detected_fault_instances": best_key[0],
            "signature_group_gain": best_key[1],
            "individual_detected_fault_instances": best_key[2],
            **metrics,
        })
    return selected, trace, signature_metrics(groups, visible, site_index)


def npy_bytes(array: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.lib.format.write_array(stream, np.ascontiguousarray(array), version=(2, 0), allow_pickle=False)
    return stream.getvalue()


def deterministic_npz(arrays: dict[str, np.ndarray]) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        for name in sorted(arrays):
            require(re.fullmatch(r"[a-z][a-z0-9_]*", name) is not None, f"invalid NPZ member: {name}")
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o600 << 16
            archive.writestr(info, npy_bytes(arrays[name]), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    return stream.getvalue()


def self_test() -> None:
    groups = np.zeros(6, dtype=np.int32)
    groups = update_groups(groups, np.asarray([0, 1, 1, 2, 2, 2], dtype=np.uint16))
    require(len(np.unique(groups)) == 3, "group update canary")
    sample = {"x": np.arange(10, dtype=np.uint8)}
    require(deterministic_npz(sample) == deterministic_npz(sample), "NPZ replay canary")
    print("Stage 12B-2D self-test: PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    require(ROOT.name == "vlsi_fault_detection_v2", f"run from project root, not {ROOT}")
    outputs = (SELECTED_NPZ, SELECTION_TRACE, SELECTION_METRICS, SELECTION_LOCK,
               FULL_CAMPAIGN_CONTRACT, MANIFEST, AUDIT)
    for path in outputs:
        require(not path.exists(), f"Stage {STAGE} output already exists: {rel(path)}")

    acceptance, policy = verify_inputs()
    data, vectors = load_data()
    print("Encoding pilot response signatures...", flush=True)
    codes = response_codes(data)
    print("Selecting 96 vectors in four frozen rounds...", flush=True)
    selected, trace, selected_metrics = greedy_select(data, codes)
    require(len(selected) == MAX_SELECTED and len(set(selected)) == MAX_SELECTED, "selected-vector count")

    detected = data["detected"].astype(bool)
    pool_visible = np.any(detected, axis=1)
    pool_ceiling_count = int(np.sum(pool_visible))
    pool_ceiling = pool_ceiling_count / FAULTS
    targets = acceptance["absolute_targets"]
    detection_target = float(targets["all_injected_detection_recall_min"])
    exact_target = float(targets["all_injected_exact_site_rate_min"])
    mean_target = float(targets["mean_observable_candidate_sites_max"])
    maximum_target = int(targets["maximum_observable_candidate_sites_max"])
    coverage_target = float(targets["observable_candidate_set_coverage_min"])
    detection_feasible = pool_ceiling >= detection_target
    selected_target_pass = (
        selected_metrics["all_injected_detection_recall"] >= detection_target
        and selected_metrics["all_injected_exact_site_rate"] >= exact_target
        and selected_metrics["observable_candidate_set_coverage"] >= coverage_target
        and selected_metrics["mean_observable_candidate_sites"] <= mean_target
        and selected_metrics["maximum_observable_candidate_sites"] <= maximum_target
    )
    full_campaign_authorized = bool(detection_feasible and selected_target_pass)
    disposition = "AUTHORIZED / NOT STARTED" if full_campaign_authorized else "BLOCKED — PILOT ACCEPTANCE NOT MET"

    selected_array = np.asarray(selected, dtype=np.int32)
    selected_payload = deterministic_npz({
        "key_u8": vectors["key_u8"][selected_array],
        "message_u8": vectors["message_u8"][selected_array],
        "selection_rank": np.arange(MAX_SELECTED, dtype=np.int32),
        "selection_round": np.arange(MAX_SELECTED, dtype=np.int32) // PER_ROUND + 1,
        "vector_index": selected_array,
    })
    trace_output = io.StringIO(newline="")
    writer = csv.DictWriter(trace_output, fieldnames=list(trace[0]), lineterminator="\n")
    writer.writeheader(); writer.writerows(trace)
    trace_payload = trace_output.getvalue().encode()
    metrics = {
        "metrics_version": "CIRCUITSAGE-HMAC-V2.1-ADAPTIVE-SELECTION-METRICS-v1",
        "stage": STAGE, "status": "FROZEN", "scope": "REPAIR_TRAIN PILOT ONLY",
        "candidate_vectors": VECTOR_COUNT, "selected_vectors": MAX_SELECTED,
        "selection_rounds": ROUNDS, "vectors_per_round": PER_ROUND,
        "complete_pool_observable_faults": pool_ceiling_count,
        "complete_pool_detection_ceiling": pool_ceiling,
        "frozen_detection_target": detection_target,
        "detection_target_feasible_with_complete_pool": detection_feasible,
        "selected_vector_metrics": selected_metrics,
        "absolute_targets": targets,
        "selected_vector_acceptance": "PASS" if selected_target_pass else "NOT_MET",
        "full_campaign_disposition": disposition,
        "claim_limit": "pilot closed-catalog evidence only; no independent-circuit generalization claim",
    }
    selection_lock = {
        "lock_version": "CIRCUITSAGE-HMAC-V2.1-ADAPTIVE-VECTOR-SELECTION-LOCK-v1",
        "stage": STAGE, "status": "FROZEN", "selection_algorithm": "DETERMINISTIC GREEDY LEXICOGRAPHIC",
        "objective_order": ["new fault detections", "response-signature group gain", "individual detections", "lower vector index"],
        "selected_vector_indices": selected, "selected_vectors": MAX_SELECTED,
        "selected_vectors_sha256": hashlib.sha256(selected_payload).hexdigest(),
        "selection_trace_sha256": hashlib.sha256(trace_payload).hexdigest(),
        "selection_partition": "REPAIR_TRAIN PILOT ONLY",
        "repair_calibration_used": False, "repair_site_test_used": False,
        "original_dev_site_test_used": False, "validation_used": False, "holdout_used": False,
        "reselection": "PROHIBITED AFTER FREEZE",
    }
    campaign_contract = {
        "contract_version": "CIRCUITSAGE-HMAC-V2.1-FULL-REPAIR-CAMPAIGN-12B2D-v1",
        "stage": STAGE, "status": "FROZEN",
        "execution_authorization": disposition,
        "blocking_reason": None if full_campaign_authorized else (
            f"complete 512-vector pilot detection ceiling {pool_ceiling:.8f} is below frozen target {detection_target:.8f}"
            if not detection_feasible else "selected pilot acceptance criteria not met"
        ),
        "selected_vectors": record_placeholder(SELECTED_NPZ, selected_payload),
        "planned_source_partition": "REPAIR_TRAIN ONLY",
        "repair_calibration": "LOCKED UNTIL SEPARATE MODEL-SELECTION CONTRACT",
        "repair_site_test": "LOCKED / NOT AUTHORIZED",
        "original_dev_site_test": "CONSUMED / REOPENING PROHIBITED",
        "validation": "PROHIBITED", "holdout": "PROHIBITED",
        "model_training": "NOT AUTHORIZED BY THIS CONTRACT",
        "target_relaxation": "PROHIBITED",
        "required_next_action_if_blocked": "OBSERVABILITY-CEILING REVIEW AND ALTERNATIVE MEASUREMENT CONTRACT",
    }

    frozen_write(SELECTED_NPZ, selected_payload)
    frozen_write(SELECTION_TRACE, trace_payload)
    frozen_write(SELECTION_METRICS, canonical_json(metrics))
    frozen_write(SELECTION_LOCK, canonical_json(selection_lock))
    frozen_write(FULL_CAMPAIGN_CONTRACT, canonical_json(campaign_contract))
    primary = (SELECTED_NPZ, SELECTION_TRACE, SELECTION_METRICS, SELECTION_LOCK, FULL_CAMPAIGN_CONTRACT)
    manifest = {
        "manifest_version": "CIRCUITSAGE-HMAC-V2.1-ADAPTIVE-SELECTION-MANIFEST-v1",
        "stage": STAGE, "status": "PASS", "pilot_audit": record(AUDIT_2C),
        "outputs": {rel(path): record(path) for path in primary},
        "candidate_vectors": VECTOR_COUNT, "selected_vectors": MAX_SELECTED,
        "complete_pool_detection_ceiling": pool_ceiling,
        "full_campaign_authorized": full_campaign_authorized,
        "simulation_calls": 0, "model_objects_deserialized": 0,
        "training_calls": 0, "inference_calls": 0,
        "repair_calibration_access": 0, "repair_site_test_access": 0,
        "original_dev_site_test_access": 0, "validation_access": 0, "holdout_access": 0,
    }
    frozen_write(MANIFEST, canonical_json(manifest))
    audit = {
        "audit_version": "CIRCUITSAGE-HMAC-V2.1-ADAPTIVE-SELECTION-CAMPAIGN-CONTRACT-FREEZE-v1",
        "stage": STAGE, "status": "PASS",
        "selection_status": "FROZEN", "campaign_contract_status": "FROZEN",
        "candidate_selected_vectors": [VECTOR_COUNT, MAX_SELECTED],
        "complete_pool_observable_faults": f"{pool_ceiling_count}/{FAULTS}",
        "complete_pool_detection_ceiling": pool_ceiling,
        "selected_detection_recall": selected_metrics["all_injected_detection_recall"],
        "selected_exact_site_rate": selected_metrics["all_injected_exact_site_rate"],
        "selected_mean_maximum_candidates": [selected_metrics["mean_observable_candidate_sites"], selected_metrics["maximum_observable_candidate_sites"]],
        "frozen_detection_target": detection_target,
        "detection_target_feasible": detection_feasible,
        "selection_acceptance": "PASS" if selected_target_pass else "NOT_MET",
        "full_repair_campaign": disposition,
        "model_training": "NOT AUTHORIZED",
        "repair_site_test": "LOCKED / NOT AUTHORIZED",
        "original_dev_site_test": "CONSUMED / NOT REOPENED",
        "validation_access": 0, "holdout_access": 0,
        "v1_modified": False, "v2_core_modified": False, "v2_1_modified": False,
        "manifest": record(MANIFEST),
        "next_gate": (
            "STAGE 12B-2E — FULL REPAIR-CAMPAIGN EXECUTION AUTHORIZATION FREEZE"
            if full_campaign_authorized else
            "STAGE 12B-2E — OBSERVABILITY-CEILING REVIEW AND ALTERNATIVE-MEASUREMENT CONTRACT FREEZE"
        ),
    }
    frozen_write(AUDIT, canonical_json(audit))

    require(selected_payload == SELECTED_NPZ.read_bytes(), "selected NPZ replay")
    require(trace_payload == SELECTION_TRACE.read_bytes(), "selection trace replay")
    for path in (SELECTION_METRICS, SELECTION_LOCK, FULL_CAMPAIGN_CONTRACT, MANIFEST, AUDIT):
        require(canonical_json(load_json(path)) == path.read_bytes(), f"canonical JSON replay: {path.name}")

    print("\nSTAGE 12B-2D — ADAPTIVE VECTOR SELECTION AND FULL REPAIR-CAMPAIGN CONTRACT FREEZE")
    print(f"{'Status':<52}: PASS")
    print(f"{'Selection / campaign contract':<52}: FROZEN / FROZEN")
    print(f"{'Candidate / selected vectors':<52}: {VECTOR_COUNT} / {MAX_SELECTED}")
    print(f"{'Selection rounds':<52}: {ROUNDS} x {PER_ROUND}")
    print(f"{'Complete-pool observable faults':<52}: {pool_ceiling_count}/{FAULTS}")
    print(f"{'Complete-pool detection ceiling':<52}: {pool_ceiling:.8f}")
    print(f"{'Frozen detection target':<52}: {detection_target:.8f}")
    print(f"{'Detection target feasible with vector pool':<52}: {'YES' if detection_feasible else 'NO'}")
    print(f"{'Selected-vector detection recall':<52}: {selected_metrics['all_injected_detection_recall']:.8f}")
    print(f"{'Selected-vector exact-site rate':<52}: {selected_metrics['all_injected_exact_site_rate']:.8f}")
    print(f"{'Mean / maximum observable candidates':<52}: {selected_metrics['mean_observable_candidate_sites']:.4f} / {selected_metrics['maximum_observable_candidate_sites']}")
    print(f"{'Selection acceptance':<52}: {'PASS' if selected_target_pass else 'NOT_MET'}")
    print(f"{'Full repair campaign':<52}: {disposition}")
    print(f"{'Simulation / training / inference':<52}: 0 / 0 / 0")
    print(f"{'REPAIR_CALIBRATION / REPAIR_SITE_TEST access':<52}: 0 / 0")
    print(f"{'Original DEV_SITE_TEST / VALIDATION / HOLDOUT':<52}: 0 / 0 / 0")
    print(f"{'Selected vectors':<52}: {SELECTED_NPZ}")
    print(f"{'Selected vectors SHA':<52}: {sha256(SELECTED_NPZ)}")
    print(f"{'Selection metrics':<52}: {SELECTION_METRICS}")
    print(f"{'Selection metrics SHA':<52}: {sha256(SELECTION_METRICS)}")
    print(f"{'Campaign contract':<52}: {FULL_CAMPAIGN_CONTRACT}")
    print(f"{'Campaign contract SHA':<52}: {sha256(FULL_CAMPAIGN_CONTRACT)}")
    print(f"{'Manifest':<52}: {MANIFEST}")
    print(f"{'Manifest SHA':<52}: {sha256(MANIFEST)}")
    print(f"{'Audit':<52}: {AUDIT}")
    print(f"{'Audit SHA':<52}: {sha256(AUDIT)}")
    print(f"{'Next gate':<52}: {audit['next_gate']}")


def record_placeholder(path: Path, payload: bytes) -> dict[str, Any]:
    return {"path": rel(path), "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}


if __name__ == "__main__":
    main()
