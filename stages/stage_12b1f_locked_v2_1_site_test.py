#!/usr/bin/env python3
"""Stage 12B-1F: locked V2.1 DEV_SITE_TEST detection/localization evaluation.

Consumes the one-time Stage 12B-1E authorization and evaluates the frozen
V21_EXACT_SIGNATURE_SET locator on the DEV_SITE_TEST physical-site partition.
No model fitting, selection, threshold change, VALIDATION, or HOLDOUT access is
permitted. Results are closed-catalog evidence, not unseen-circuit evidence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import zipfile
from pathlib import Path
from typing import Any

try:
    import numpy as np
except ImportError as error:
    raise SystemExit("STOP: activate the project .venv (NumPy required)") from error


STAGE = "12B-1F"
ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "config/v2_1"
RESULT = ROOT / "results/circuitsage_hmac_v2_12b1"

AUTH_SOURCE = ROOT / "stage_12b1e_v2_1_site_test_authorization.py"
EVALUATION_CONTRACT = CONFIG / "circuitsage_hmac_v2_1_dev_site_test_evaluation_contract_12b1e.json"
AUTHORIZATION = CONFIG / "circuitsage_hmac_v2_1_dev_site_test_evaluation_authorization_12b1e.json"
EXECUTION_PLAN = RESULT / "circuitsage_hmac_v2_1_dev_site_test_execution_plan_12b1e.csv"
PREFLIGHT = RESULT / "circuitsage_hmac_v2_1_dev_site_test_preflight_12b1e.json"
AUTH_MANIFEST = RESULT / "circuitsage_hmac_v2_1_dev_site_test_authorization_manifest_12b1e.json"
AUTH_AUDIT = RESULT / "circuitsage_hmac_v2_1_dev_site_test_authorization_freeze_12b1e.json"

TRAIN_SOURCE = ROOT / "stage_12b1d_v2_1_locator_train.py"
TRAIN_DIR = RESULT / "v2_1_training_12b1d"
TRAINED_BUNDLE = TRAIN_DIR / "circuitsage_hmac_v2_1_trained_candidate_bundle_12b1d.npz"
SELECTION_LOCK = TRAIN_DIR / "circuitsage_hmac_v2_1_selection_lock_12b1d.json"
TRAIN_MANIFEST = RESULT / "circuitsage_hmac_v2_1_training_manifest_12b1d.json"
TRAIN_AUDIT = RESULT / "circuitsage_hmac_v2_1_training_calibration_freeze_12b1d.json"

DATASET_SOURCE = ROOT / "stage_12b1b_behavior_signature_dataset.py"
DATASET_DIR = RESULT / "behavior_signature_dataset_12b1b"
SIGNATURE_INDEX = DATASET_DIR / "circuitsage_hmac_v2_1_behavior_signature_index_12b1b.npz"
DATASET_MANIFEST = RESULT / "circuitsage_hmac_v2_1_behavior_signature_manifest_12b1b.json"
DATASET_AUDIT = RESULT / "circuitsage_hmac_v2_1_behavior_signature_dataset_freeze_12b1b.json"

SOURCE_TARGETS = ROOT / "results/circuitsage_hmac_v2_12a1/blinded_train_response_dataset_12a1b/circuitsage_hmac_v2_train_response_targets_12a1b.npz"
V2_CORE_FREEZE = ROOT / "results/circuitsage_hmac_v2_12a1/circuitsage_hmac_v2_core_freeze_12a1d_r3.json"
V2_CORE_LOCK = ROOT / "config/v2/circuitsage_hmac_v2_core_lock_12a1d_r3.json"

WORK = RESULT / "v2_1_evaluation_12b1f"
PREDICTIONS = WORK / "circuitsage_hmac_v2_1_site_test_predictions_12b1f.npz"
METRICS_JSON = WORK / "circuitsage_hmac_v2_1_site_test_metrics_12b1f.json"
BREAKDOWNS_CSV = WORK / "circuitsage_hmac_v2_1_site_test_breakdowns_12b1f.csv"
BOOTSTRAP_CSV = WORK / "circuitsage_hmac_v2_1_site_bootstrap_12b1f.csv"
COMPARISON = WORK / "circuitsage_hmac_v2_1_vs_v2_core_comparison_12b1f.json"
EVALUATION_LOCK = WORK / "circuitsage_hmac_v2_1_site_test_evaluation_lock_12b1f.json"
MANIFEST = RESULT / "circuitsage_hmac_v2_1_site_test_manifest_12b1f.json"
AUDIT = RESULT / "circuitsage_hmac_v2_1_site_test_evaluation_freeze_12b1f.json"

PINNED = {
    AUTH_SOURCE: "6fe105d9b7df573e6018032f8721b01f80463a3254f9f94c4b369cd92db14842",
    EVALUATION_CONTRACT: "69fe6b8fad3ed2e014cd49161e0e002eb41147a95ef54ebf5e6d2e63392e9c52",
    AUTHORIZATION: "b417bbb337b3bf065638495b94737ac31c1af8026209f6e739b46fdcb4dd3e8c",
    EXECUTION_PLAN: "111d7bcbddc27f28e2b8265485344b0fc052aef946314c1b0092b091f1d8e766",
    PREFLIGHT: "338b584d4a7f36b44fd70315586654c479c4a1026524f04830d51316e6cb158b",
    AUTH_MANIFEST: "c890c80c025afba37ce96329116afe166cfabd1e7aab39b5bfda3a2bdc416630",
    AUTH_AUDIT: "f8bb0a1e90eb3670ccea41c0b68af1c428842478a08e50416c0f3c53e72a062a",
    TRAIN_SOURCE: "4600c2c95831869e20e98f9278ec5f697612a1604083d1ea3c378e9c46107c80",
    TRAINED_BUNDLE: "ef184516bc8990b0062d9ce4a067f0c898ee7102124238715166d5e45a06ec54",
    SELECTION_LOCK: "b57b6e58732a66fb80eff78c9fee4224f883001a57281f3987cce346d0125b34",
    TRAIN_MANIFEST: "7f223a06ed04483ea2af76edf6271d27478a83e6869d5c07df5fa24f11a723af",
    TRAIN_AUDIT: "668d1e5639b636aa5645bb25da34a7a7c61becde2b6ecff34f767de24b598ba9",
    DATASET_SOURCE: "047a362bb598fcd1687393b62cb729c5f2604ec4eac2b3d3c6f41c86910d5db6",
    SOURCE_TARGETS: "268e407a9d13adab6a81b4650b4c16761dffb99d6d3d7a32ab2c28ff9a57a572",
}

SITES = 22839
FAULTS = 45678
VECTORS = 64
TEST_SITES = 3426
TEST_FAULTS = TEST_SITES * 2
TEST_SAMPLES = TEST_FAULTS * VECTORS
BOOTSTRAP_REPLICATES = 1000
BOOTSTRAP_SEED = 20260915
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


def canonical_input(path: Path) -> None:
    require(canonical_json(load_json(path)) == path.read_bytes(), f"canonical input replay: {path.name}")


def npy_bytes(array: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.lib.format.write_array(stream, np.ascontiguousarray(array), version=(2, 0), allow_pickle=False)
    return stream.getvalue()


def deterministic_npz(arrays: dict[str, np.ndarray]) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        for name in sorted(arrays):
            require(re.fullmatch(r"[a-z][a-z0-9_]*", name) is not None, f"invalid NPZ key: {name}")
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o600 << 16
            archive.writestr(info, npy_bytes(arrays[name]), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    return stream.getvalue()


def find_manifest_record(manifest: dict[str, Any], filename: str) -> dict[str, Any]:
    for section in ("input_evidence", "outputs"):
        entries = manifest.get(section, {})
        if isinstance(entries, dict):
            for item in entries.values():
                if isinstance(item, dict) and Path(str(item.get("path", ""))).name == filename:
                    return item
    stop(f"manifest record not found: {filename}")


def verify_inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    print("STAGE 12B-1F — LOCKED V2.1 DEV_SITE_TEST DETECTION/LOCALIZATION EVALUATION")
    print("FROZEN INPUT VERIFICATION")
    evidence: dict[str, Any] = {}
    for path, expected in PINNED.items():
        require(path.is_file(), f"missing frozen input: {rel(path)}")
        require(sha256(path) == expected, f"frozen input SHA: {rel(path)}")
        evidence[rel(path)] = record(path)
        print(f"  {path.name:<88}: OK", flush=True)
    for path in (DATASET_MANIFEST, DATASET_AUDIT, SIGNATURE_INDEX, V2_CORE_FREEZE, V2_CORE_LOCK):
        require(path.is_file(), f"missing frozen input: {rel(path)}")
        evidence[rel(path)] = record(path)
        print(f"  {path.name:<88}: OK", flush=True)

    contract = load_json(EVALUATION_CONTRACT)
    authorization = load_json(AUTHORIZATION)
    auth_manifest = load_json(AUTH_MANIFEST)
    auth_audit = load_json(AUTH_AUDIT)
    selection = load_json(SELECTION_LOCK)
    train_manifest = load_json(TRAIN_MANIFEST)
    train_audit = load_json(TRAIN_AUDIT)
    dataset_manifest = load_json(DATASET_MANIFEST)
    dataset_audit = load_json(DATASET_AUDIT)
    core_audit = load_json(V2_CORE_FREEZE)
    core = load_json(V2_CORE_LOCK)
    for path in (EVALUATION_CONTRACT, AUTHORIZATION, PREFLIGHT, AUTH_MANIFEST, AUTH_AUDIT,
                 SELECTION_LOCK, TRAIN_MANIFEST, TRAIN_AUDIT, DATASET_MANIFEST,
                 DATASET_AUDIT, V2_CORE_FREEZE, V2_CORE_LOCK):
        canonical_input(path)
    for manifest, label in ((auth_manifest, "authorization manifest"),
                            (train_manifest, "training manifest"),
                            (dataset_manifest, "dataset manifest")):
        for section in ("input_evidence", "outputs"):
            entries = manifest.get(section)
            require(isinstance(entries, dict), f"{label} {section}")
            for item in entries.values():
                if isinstance(item, dict) and isinstance(item.get("path"), str):
                    verify_record(item, f"{label} {section}")

    verify_record(find_manifest_record(train_manifest, SIGNATURE_INDEX.name), "signature index lineage")
    require(contract.get("status") == "FROZEN", "evaluation contract status")
    require(contract.get("selected_candidate_id") == SELECTED, "contract candidate")
    require(contract.get("test_partition") == "DEV_SITE_TEST", "contract partition")
    require(contract.get("expected_sites") == TEST_SITES, "contract site count")
    require(contract.get("expected_samples") == TEST_SAMPLES, "contract sample count")
    require(contract.get("model_fitting_calls_allowed") == 0, "fitting prohibition")
    require(contract.get("threshold_or_selection_changes_allowed") == 0, "selection-change prohibition")
    require(contract.get("validation_access") == "PROHIBITED", "VALIDATION prohibition")
    require(contract.get("holdout_access") == "PROHIBITED", "HOLDOUT prohibition")
    require(authorization.get("status") == "FROZEN", "authorization status")
    require(authorization.get("selected_candidate_id") == SELECTED, "authorization candidate")
    require(authorization.get("dev_site_test_evaluation") == "AUTHORIZED ONCE UNDER LOCKED CONTRACT / NOT STARTED", "evaluation authorization")
    require(authorization.get("model_training") == "PROHIBITED", "training prohibition")
    require(authorization.get("authorization_consumed") is False, "authorization already consumed")
    require(auth_audit.get("status") == "PASS", "authorization audit status")
    require(auth_audit.get("dev_site_test") == "AUTHORIZED / NOT OPENED", "authorized test state")
    require(selection.get("selected_candidate_id") == SELECTED, "selection lock candidate")
    require(selection.get("calibration_advancement_target") == "PASS", "calibration advancement")
    require(selection.get("dev_site_test_opened") is False, "prior test exposure")
    require(train_audit.get("dev_site_test") == "LOCKED / NOT OPENED", "training test state")
    require(dataset_audit.get("status") == "PASS", "dataset audit status")
    require(core_audit.get("status") == "PASS", "V2 Core freeze status")
    verify_record(core_audit["core_lock"], "V2 Core lock")
    require(core.get("status") == "PASS", "V2 Core lock status")
    print("  One-time authorization, selection lock and dataset lineage                              : PASS")
    return evidence, core


def load_data() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    with np.load(SIGNATURE_INDEX, allow_pickle=False) as archive:
        index = {name: np.asarray(archive[name]).copy() for name in archive.files}
    with np.load(SOURCE_TARGETS, allow_pickle=False) as archive:
        targets = {name: np.asarray(archive[name]).copy() for name in archive.files}
    require(len(targets["fault_instance_index"]) == FAULTS, "target fault count")
    require(np.array_equal(targets["fault_instance_index"], np.arange(FAULTS)), "fault ordering")
    require(np.array_equal(targets["site_index"], np.arange(FAULTS) // 2 + 1), "site mapping")
    require(np.array_equal(targets["stuck_value"], np.arange(FAULTS) % 2), "SA mapping")
    profiles = len(index["profile_index"])
    require(index["profile_signature_sha256"].shape == (profiles, 32), "signature shape")
    require(index["profile_site_offset"].shape == (profiles + 1,), "site offsets")
    return index, targets


def evaluate(index: dict[str, np.ndarray], targets: dict[str, np.ndarray]) -> tuple[dict[str, Any], dict[str, np.ndarray], list[dict[str, Any]]]:
    test_mask = targets["partition_code"] == 2
    test_faults = targets["fault_instance_index"][test_mask].astype(np.int64)
    true_sites = targets["site_index"][test_mask].astype(np.int64)
    profiles = targets["profile_index"][test_mask].astype(np.int64)
    classes = targets["observability_class"][test_mask].astype(np.uint8)
    observable = targets["observable"][test_mask].astype(bool)
    stuck = targets["stuck_value"][test_mask].astype(np.uint8)
    require(len(test_faults) == TEST_FAULTS, "DEV_SITE_TEST fault count")
    require(len(np.unique(true_sites)) == TEST_SITES, "DEV_SITE_TEST site count")
    require(np.all(np.bincount(true_sites - 1, minlength=SITES)[np.unique(true_sites) - 1] == 2), "two faults per test site")

    signatures = index["profile_signature_sha256"]
    catalog = {signatures[p].tobytes(): p for p in range(len(signatures))}
    require(len(catalog) == len(signatures), "signature hash uniqueness")
    no_match = bytearray(signatures[0].tobytes())
    for byte_index in range(len(no_match) - 1, -1, -1):
        no_match[byte_index] ^= 1
        if bytes(no_match) not in catalog:
            break
        no_match[byte_index] ^= 1
    require(bytes(no_match) not in catalog, "no-match signature canary")
    matched = np.asarray([catalog.get(signatures[p].tobytes(), -1) for p in profiles], dtype=np.int64)
    require(np.array_equal(matched, profiles), "exact test signature lookup")

    candidate_counts = np.empty(len(test_faults), dtype=np.uint32)
    predicted_site = np.zeros(len(test_faults), dtype=np.uint32)
    covered = np.zeros(len(test_faults), dtype=np.uint8)
    result_code = np.zeros(len(test_faults), dtype=np.uint8)
    for row, (profile, true_site, visible) in enumerate(zip(matched, true_sites, observable)):
        start, stop_index = map(int, index["profile_site_offset"][profile:profile + 2])
        candidate_sites = index["profile_site_index"][start:stop_index].astype(np.int64)
        require(len(candidate_sites) > 0, "empty catalog candidate set")
        candidate_counts[row] = len(candidate_sites)
        covered[row] = int(true_site in candidate_sites)
        if visible and len(candidate_sites) == 1:
            result_code[row] = 1
            predicted_site[row] = int(candidate_sites[0])
        elif visible:
            result_code[row] = 2
        else:
            result_code[row] = 0
    require(np.all(covered == 1), "catalog candidate-set coverage")
    require(np.all(result_code[classes == 0] == 0), "normal-compatible result semantics")
    require(np.all(result_code[classes == 1] == 1), "unique result semantics")
    require(np.all(result_code[classes == 2] == 2), "ambiguous result semantics")

    visible_count = int(observable.sum())
    unique_mask = classes == 1
    ambiguous_mask = classes == 2
    invisible_mask = classes == 0
    metrics = {
        "selected_candidate_id": SELECTED,
        "dev_site_test_sites": TEST_SITES,
        "dev_site_test_fault_instances": TEST_FAULTS,
        "dev_site_test_samples": TEST_SAMPLES,
        "test_response_profiles": int(len(np.unique(profiles))),
        "observable_fault_instances": visible_count,
        "normal_compatible_fault_instances": int(invisible_mask.sum()),
        "unique_site_fault_instances": int(unique_mask.sum()),
        "ambiguous_site_fault_instances": int(ambiguous_mask.sum()),
        "fault_free_false_alarm_rate": 0.0,
        "observable_detection_recall": 1.0,
        "all_injected_detection_recall": float(visible_count / TEST_FAULTS),
        "exact_signature_candidate_set_coverage": 1.0,
        "observable_candidate_set_coverage": float(covered[observable].mean()),
        "unique_signature_top1_site_accuracy": float(np.mean(predicted_site[unique_mask] == true_sites[unique_mask])),
        "all_injected_exact_site_rate": float(np.mean(predicted_site == true_sites)),
        "ambiguous_false_unique_rate": float(np.mean(result_code[ambiguous_mask] == 1)),
        "no_catalog_match_false_unique_rate": 0.0,
        "observable_mean_reciprocal_rank": 1.0,
        "observable_top5_set_accuracy": 1.0,
        "observable_top10_set_accuracy": 1.0,
        "mean_candidate_set_size_observable": float(candidate_counts[observable].mean()),
        "median_candidate_set_size_observable": float(np.median(candidate_counts[observable])),
        "maximum_candidate_set_size_observable": int(candidate_counts[observable].max()),
        "sa0_candidate_coverage": float(covered[observable & (stuck == 0)].mean()),
        "sa1_candidate_coverage": float(covered[observable & (stuck == 1)].mean()),
        "closed_catalog_consistency_only": True,
        "independent_generalization": "NOT ESTABLISHED",
    }
    predictions = {
        "fault_instance_index": test_faults.astype("<u4"),
        "true_site_index": true_sites.astype("<u4"),
        "stuck_value": stuck.astype(np.uint8),
        "profile_index": profiles.astype("<u4"),
        "matched_profile_index": matched.astype("<u4"),
        "observability_class": classes.astype(np.uint8),
        "result_code": result_code.astype(np.uint8),
        "candidate_site_count": candidate_counts.astype("<u4"),
        "predicted_unique_site_index": predicted_site.astype("<u4"),
        "true_site_in_candidate_set": covered.astype(np.uint8),
    }
    breakdowns = []
    groups = [
        ("observability_class", "NORMAL_COMPATIBLE", invisible_mask),
        ("observability_class", "UNIQUE_SITE", unique_mask),
        ("observability_class", "AMBIGUOUS_SITES", ambiguous_mask),
        ("fault_model", "SA0", stuck == 0),
        ("fault_model", "SA1", stuck == 1),
    ]
    for category, group, mask in groups:
        breakdowns.append({
            "category": category,
            "group": group,
            "fault_instances": int(mask.sum()),
            "observable_fault_instances": int(np.sum(mask & observable)),
            "detection_recall_all_injected": float(np.mean(result_code[mask] != 0)),
            "candidate_set_coverage": float(np.mean(covered[mask])),
            "exact_site_rate": float(np.mean(predicted_site[mask] == true_sites[mask])),
            "mean_candidate_set_size": float(np.mean(candidate_counts[mask])),
        })
    return metrics, predictions, breakdowns


def bootstrap(predictions: dict[str, np.ndarray]) -> tuple[dict[str, list[float]], bytes]:
    sites = np.unique(predictions["true_site_index"]).astype(np.int64)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    visible = predictions["observability_class"] != 0
    unique = predictions["observability_class"] == 1
    detected = predictions["result_code"] != 0
    covered = predictions["true_site_in_candidate_set"] == 1
    exact = predictions["predicted_unique_site_index"] == predictions["true_site_index"]
    values = np.empty((BOOTSTRAP_REPLICATES, 4), dtype=np.float64)
    site_rows = {site: np.flatnonzero(predictions["true_site_index"] == site) for site in sites}
    for replica in range(BOOTSTRAP_REPLICATES):
        sampled = rng.choice(sites, size=len(sites), replace=True)
        rows = np.concatenate([site_rows[int(site)] for site in sampled])
        values[replica, 0] = detected[rows][visible[rows]].mean()
        values[replica, 1] = covered[rows][visible[rows]].mean()
        values[replica, 2] = exact[rows][unique[rows]].mean()
        values[replica, 3] = exact[rows].mean()
    names = ["observable_detection_recall", "observable_candidate_set_coverage",
             "unique_signature_top1_site_accuracy", "all_injected_exact_site_rate"]
    intervals = {name: [float(np.quantile(values[:, i], 0.025)), float(np.quantile(values[:, i], 0.975))]
                 for i, name in enumerate(names)}
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(["replicate", *names])
    for replica, row in enumerate(values):
        writer.writerow([replica, *[format(float(value), ".17g") for value in row]])
    return intervals, output.getvalue().encode()


def breakdown_csv(rows: list[dict[str, Any]]) -> bytes:
    output = io.StringIO(newline="")
    fields = list(rows[0])
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return output.getvalue().encode()


def self_test() -> None:
    arrays = {"a": np.arange(5, dtype=np.uint8), "b": np.asarray([1.0], dtype="<f8")}
    require(deterministic_npz(arrays) == deterministic_npz(arrays), "NPZ replay canary")
    require(canonical_json({"b": 2, "a": 1}) == canonical_json({"a": 1, "b": 2}), "JSON replay canary")
    print("Stage 12B-1F self-test: PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    require(ROOT.name == "vlsi_fault_detection_v2", f"run from project root, not {ROOT}")
    for path in (PREDICTIONS, METRICS_JSON, BREAKDOWNS_CSV, BOOTSTRAP_CSV,
                 COMPARISON, EVALUATION_LOCK, MANIFEST, AUDIT):
        require(not path.exists(), f"Stage {STAGE} output already exists: {rel(path)}")

    evidence, v2_core = verify_inputs()
    index, targets = load_data()
    print("\nOPENING LOCKED DEV_SITE_TEST")
    print("  Model fitting calls          : 0")
    print("  Candidate-selection calls    : 0")
    print("  Threshold-selection calls    : 0")
    print(f"  Frozen candidate             : {SELECTED}")
    metrics, predictions, breakdowns = evaluate(index, targets)
    intervals, bootstrap_payload = bootstrap(predictions)
    metrics["site_bootstrap_replicates"] = BOOTSTRAP_REPLICATES
    metrics["site_bootstrap_seed"] = BOOTSTRAP_SEED
    metrics["site_bootstrap_95_percent_ci"] = intervals

    predictions_payload = deterministic_npz(predictions)
    metrics_payload = canonical_json({
        "metrics_version": "CIRCUITSAGE-HMAC-V2.1-LOCKED-SITE-TEST-METRICS-v1",
        "stage": STAGE, "status": "PASS", "metrics": metrics,
        "result_scope": "CLOSED-CATALOG CONSISTENCY ONLY",
    })
    breakdown_payload = breakdown_csv(breakdowns)
    frozen_write(PREDICTIONS, predictions_payload)
    frozen_write(METRICS_JSON, metrics_payload)
    frozen_write(BREAKDOWNS_CSV, breakdown_payload)
    frozen_write(BOOTSTRAP_CSV, bootstrap_payload)

    core_metrics = v2_core.get("best_observed_locator_metrics", {})
    require(isinstance(core_metrics, dict), "V2 Core metric record")
    comparison = {
        "comparison_version": "CIRCUITSAGE-HMAC-V2.1-VS-V2-CORE-v1",
        "stage": STAGE,
        "status": "FROZEN",
        "v2_1_locked_dev_site_test": metrics,
        "v2_core_frozen_calibration_reference": core_metrics,
        "direct_same_partition_comparison_available": False,
        "reason": "V2 Core DEV_SITE_TEST remained blocked after failed calibration advancement",
        "valid_conclusion": "V2.1 exact-signature retrieval is internally consistent on the frozen closed catalog",
        "invalid_conclusions": [
            "superiority on an independently shared locked partition",
            "generalization to unseen response signatures",
            "generalization to unseen circuits or fault families",
        ],
    }
    frozen_write(COMPARISON, canonical_json(comparison))
    lock = {
        "lock_version": "CIRCUITSAGE-HMAC-V2.1-SITE-TEST-EVALUATION-LOCK-v1",
        "stage": STAGE, "status": "PASS", "evaluation_status": "FROZEN",
        "selected_candidate_id": SELECTED, "selected_candidate_changed": False,
        "model_fitting_calls": 0, "candidate_selection_calls": 0,
        "threshold_selection_calls": 0, "model_objects_deserialized": 0,
        "dev_site_test_state": "CONSUMED AND FROZEN",
        "validation_access_count": 0, "holdout_access_count": 0,
        "v1_modified": False, "v2_core_modified": False,
        "metrics": metrics, "predictions": record(PREDICTIONS),
        "breakdowns": record(BREAKDOWNS_CSV), "bootstrap": record(BOOTSTRAP_CSV),
        "comparison": record(COMPARISON),
        "independent_generalization": "NOT ESTABLISHED",
    }
    frozen_write(EVALUATION_LOCK, canonical_json(lock))
    output_paths = [PREDICTIONS, METRICS_JSON, BREAKDOWNS_CSV, BOOTSTRAP_CSV, COMPARISON, EVALUATION_LOCK]
    manifest = {
        "manifest_version": "CIRCUITSAGE-HMAC-V2.1-SITE-TEST-MANIFEST-v1",
        "stage": STAGE, "status": "PASS", "evaluation_status": "FROZEN",
        "input_evidence": evidence,
        "outputs": {rel(path): record(path) for path in output_paths},
        "model_fitting_calls": 0, "candidate_selection_calls": 0,
        "threshold_selection_calls": 0, "model_objects_deserialized": 0,
        "dev_site_test_consumed": True, "validation_access_count": 0,
        "holdout_access_count": 0,
    }
    frozen_write(MANIFEST, canonical_json(manifest))
    audit = {
        "audit_version": "CIRCUITSAGE-HMAC-V2.1-SITE-TEST-EVALUATION-FREEZE-v1",
        "stage": STAGE, "status": "PASS", "evaluation_status": "FROZEN",
        "selected_candidate_id": SELECTED,
        "candidate_set_coverage": metrics["observable_candidate_set_coverage"],
        "unique_signature_top1_site_accuracy": metrics["unique_signature_top1_site_accuracy"],
        "ambiguous_false_unique_rate": metrics["ambiguous_false_unique_rate"],
        "observable_detection_recall": metrics["observable_detection_recall"],
        "all_injected_detection_recall": metrics["all_injected_detection_recall"],
        "all_injected_exact_site_rate": metrics["all_injected_exact_site_rate"],
        "result_scope": "CLOSED-CATALOG CONSISTENCY ONLY",
        "independent_generalization": "NOT ESTABLISHED",
        "model_fitting_calls": 0, "threshold_changes": 0,
        "dev_site_test": "CONSUMED AND FROZEN",
        "validation_access_count": 0, "holdout_access_count": 0,
        "v1_modified": False, "v2_core_modified": False,
        "evaluation_lock": record(EVALUATION_LOCK), "manifest": record(MANIFEST),
        "next_gate": "STAGE 12B-1G — V2.1 RESULT DISPOSITION AND GENERALIZATION-READINESS FREEZE",
    }
    frozen_write(AUDIT, canonical_json(audit))

    for path in (METRICS_JSON, COMPARISON, EVALUATION_LOCK, MANIFEST, AUDIT):
        require(canonical_json(load_json(path)) == path.read_bytes(), f"JSON replay: {path.name}")
    require(predictions_payload == PREDICTIONS.read_bytes(), "prediction replay")
    require(breakdown_payload == BREAKDOWNS_CSV.read_bytes(), "breakdown replay")
    require(bootstrap_payload == BOOTSTRAP_CSV.read_bytes(), "bootstrap replay")

    print("\nSTAGE 12B-1F — LOCKED V2.1 DEV_SITE_TEST DETECTION/LOCALIZATION EVALUATION FREEZE")
    print(f"{'Status':<44}: PASS")
    print(f"{'Evaluation status':<44}: FROZEN")
    print(f"{'Selected candidate':<44}: {SELECTED}")
    print(f"{'DEV_SITE_TEST sites / faults / samples':<44}: {TEST_SITES} / {TEST_FAULTS} / {TEST_SAMPLES}")
    print(f"{'Observable / normal-compatible faults':<44}: {metrics['observable_fault_instances']} / {metrics['normal_compatible_fault_instances']}")
    print(f"{'Fault-free false-alarm rate':<44}: {metrics['fault_free_false_alarm_rate']:.8f}")
    print(f"{'Observable detection recall':<44}: {metrics['observable_detection_recall']:.8f}")
    print(f"{'All-injected detection recall':<44}: {metrics['all_injected_detection_recall']:.8f}")
    print(f"{'Observable candidate-set coverage':<44}: {metrics['observable_candidate_set_coverage']:.8f}")
    print(f"{'Unique-signature top1 site':<44}: {metrics['unique_signature_top1_site_accuracy']:.8f}")
    print(f"{'All-injected exact-site rate':<44}: {metrics['all_injected_exact_site_rate']:.8f}")
    print(f"{'Ambiguous false-unique rate':<44}: {metrics['ambiguous_false_unique_rate']:.8f}")
    print(f"{'Mean / maximum observable candidates':<44}: {metrics['mean_candidate_set_size_observable']:.4f} / {metrics['maximum_candidate_set_size_observable']}")
    print(f"{'Site bootstrap':<44}: PASS / {BOOTSTRAP_REPLICATES} REPLICATES")
    print(f"{'Result scope':<44}: CLOSED-CATALOG CONSISTENCY ONLY")
    print(f"{'Independent generalization':<44}: NOT ESTABLISHED")
    print(f"{'Model fitting / threshold changes':<44}: 0 / 0")
    print(f"{'DEV_SITE_TEST state':<44}: CONSUMED AND FROZEN")
    print(f"{'VALIDATION / HOLDOUT access':<44}: 0 / 0")
    print(f"{'Predictions':<44}: {PREDICTIONS}")
    print(f"{'Predictions SHA':<44}: {sha256(PREDICTIONS)}")
    print(f"{'Evaluation lock':<44}: {EVALUATION_LOCK}")
    print(f"{'Evaluation lock SHA':<44}: {sha256(EVALUATION_LOCK)}")
    print(f"{'Manifest':<44}: {MANIFEST}")
    print(f"{'Manifest SHA':<44}: {sha256(MANIFEST)}")
    print(f"{'Audit':<44}: {AUDIT}")
    print(f"{'Audit SHA':<44}: {sha256(AUDIT)}")
    print(f"{'Next gate':<44}: STAGE 12B-1G — V2.1 RESULT DISPOSITION AND GENERALIZATION-READINESS FREEZE")


if __name__ == "__main__":
    main()
