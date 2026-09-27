#!/usr/bin/env python3
"""Stage 12A-1D-R1: V2 localization failure review and repair contract.

This stage analyzes only frozen Stage 12A-1D artifacts.  It does not train a
model, deserialize V1, open DEV_SITE_TEST, or access VALIDATION/HOLDOUT.  Its
purpose is to explain the failed calibration advancement gate and freeze a
leakage-safe repair contract for Stage 12A-1D-R2.
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

try:
    import numpy as np
except ImportError as error:
    raise SystemExit("STOP: activate the project .venv (NumPy required)") from error


STAGE = "12A-1D-R1"
VERSION = "CIRCUITSAGE-HMAC-V2-LOCALIZATION-FAILURE-REVIEW-v1"
ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "config" / "v2"
RESULT = ROOT / "results" / "circuitsage_hmac_v2_12a1"
SOURCE = ROOT / "stage_12a1d_v2_linked_train.py"
SELECTION_LOCK = RESULT / "v2_training_12a1d/circuitsage_hmac_v2_selection_lock_12a1d.json"
TRAINING_MANIFEST = RESULT / "circuitsage_hmac_v2_training_manifest_12a1d.json"
TRAINING_AUDIT = RESULT / "circuitsage_hmac_v2_training_calibration_freeze_12a1d.json"

WORK = RESULT / "v2_localization_failure_review_12a1d_r1"
DIAGNOSTICS = WORK / "circuitsage_hmac_v2_localization_diagnostics_12a1d_r1.json"
METRICS = WORK / "circuitsage_hmac_v2_localization_failure_metrics_12a1d_r1.csv"
REPORT = WORK / "circuitsage_hmac_v2_localization_failure_review_12a1d_r1.md"
REPAIR_CONTRACT = CONFIG / "circuitsage_hmac_v2_localization_repair_contract_12a1d_r1.json"
MANIFEST = RESULT / "circuitsage_hmac_v2_localization_failure_review_manifest_12a1d_r1.json"
AUDIT = RESULT / "circuitsage_hmac_v2_localization_failure_review_freeze_12a1d_r1.json"

PINNED = {
    SOURCE: "6144da31a28041f045d696d041f147efcfceacccd1fcfecaa52d8561c640020d",
    SELECTION_LOCK: "6fb8d4eb79704652ccb289090e2b249c7600f0db723fdfffe62161e4fe5028b7",
    TRAINING_MANIFEST: "2c3df37e43ae93fb5ff14becf873e270ff59be7a8e30f9032feddfa5880b18fe",
    TRAINING_AUDIT: "c6437b429d8b2f4747dbbdcdcf7b92a7703d75b6428c8fb643d8c75fc3184b96",
}

EXPECTED = {
    "candidate_id": "V2_LINKED_RANKER_SMALL_A1E5",
    "observable_mean_reciprocal_rank": 0.00629651,
    "unique_signature_top1_site_accuracy": 0.00348432,
    "observable_top5_site_accuracy": 0.00406740,
    "observable_candidate_coverage_top50": 0.04154561,
    "sa0_sa1_accuracy_given_top1_site": 0.66666667,
    "fault_free_false_alarm_rate": 0.0,
    "observable_detection_recall": 1.0,
}

TARGETS = {
    "observable_candidate_coverage_top50": 0.95,
    "unique_signature_top1_site_accuracy": 0.80,
    "observable_top5_site_accuracy": 0.80,
    "fault_free_false_alarm_rate": 0.05,
    "observable_detection_recall": 0.90,
}

SITES = 22839
FAULTS = 45678
NEGATIVES_PER_QUERY = 128
TOP_K = 50


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


def artifact(outputs: dict[str, Any], basename: str) -> tuple[Path, dict[str, Any]]:
    matches = []
    for key, item in outputs.items():
        if isinstance(item, dict):
            path = resolve_record_path(item)
            if path.name == basename:
                matches.append((path, item))
    require(len(matches) == 1, f"manifest artifact resolution: {basename}")
    path, item = matches[0]
    require(path.is_file(), f"missing Stage 12A-1D artifact: {rel(path)}")
    require(sha256(path) == item.get("sha256"), f"Stage 12A-1D artifact changed: {path.name}")
    return path, item


def verify_frozen_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    print("STAGE 12A-1D-R1 — LOCALIZATION FAILURE REVIEW")
    print("FROZEN INPUT VERIFICATION")
    evidence: dict[str, Any] = {}
    for path, expected in PINNED.items():
        require(path.is_file(), f"missing frozen input: {rel(path)}")
        require(sha256(path) == expected, f"frozen input SHA mismatch: {rel(path)}")
        evidence[rel(path)] = record(path)
        print(f"  {path.name:<76}: OK", flush=True)

    lock = load_json(SELECTION_LOCK)
    manifest = load_json(TRAINING_MANIFEST)
    audit = load_json(TRAINING_AUDIT)
    require(lock.get("status") == "PASS", "Stage 12A-1D selection status")
    require(lock.get("training_status") == "FROZEN", "Stage 12A-1D training status")
    require(lock.get("selection_status") == "FROZEN", "Stage 12A-1D selection freeze")
    require(lock.get("advancement_target_on_calibration") == "NOT_MET", "expected failed advancement gate")
    require(lock.get("dev_site_test_opened") is False, "DEV_SITE_TEST state")
    require(lock.get("validation_access_count") == 0, "VALIDATION access count")
    require(lock.get("holdout_access_count") == 0, "HOLDOUT access count")
    require(lock.get("v1_model_modified") is False, "V1 model modification state")
    require(audit.get("status") == "PASS", "Stage 12A-1D audit status")
    require(audit.get("calibration_advancement_target") == "NOT_MET", "Stage 12A-1D audit gate")
    require(audit.get("dev_site_test_opened") is False, "Stage 12A-1D audit site-test state")
    require(manifest.get("status") == "PASS", "Stage 12A-1D manifest status")
    outputs = manifest.get("outputs")
    require(isinstance(outputs, dict), "Stage 12A-1D output manifest")
    for item in outputs.values():
        if isinstance(item, dict):
            path = resolve_record_path(item)
            require(path.is_file() and sha256(path) == item.get("sha256"),
                    f"Stage 12A-1D output integrity: {path.name}")
    print("  Stage 12A-1D semantic freeze and output manifest                         : PASS")
    return lock, manifest, audit, evidence


def verify_metrics(lock: dict[str, Any], outputs: dict[str, Any]) -> tuple[dict[str, float], Path]:
    selected = lock.get("selected_metrics")
    require(isinstance(selected, dict), "selected calibration metrics")
    require(lock.get("selected_candidate_id") == EXPECTED["candidate_id"], "selected candidate")
    for key, wanted in EXPECTED.items():
        if key == "candidate_id":
            continue
        require(key in selected, f"selected metric missing: {key}")
        require(math.isclose(float(selected[key]), float(wanted), rel_tol=0.0, abs_tol=5e-9),
                f"selected metric changed: {key}")

    metrics_path, _ = artifact(outputs, "circuitsage_hmac_v2_candidate_calibration_metrics_12a1d.csv")
    with metrics_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    require(len(rows) == 2, "candidate calibration metric rows")
    row = next((item for item in rows if item.get("candidate_id") == EXPECTED["candidate_id"]), None)
    require(row is not None, "selected candidate metrics row")
    for key in EXPECTED:
        if key != "candidate_id":
            require(math.isclose(float(row[key]), float(selected[key]), rel_tol=0.0, abs_tol=1e-12),
                    f"CSV/selection-lock metric agreement: {key}")
    return {key: float(value) for key, value in selected.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)}, metrics_path


def training_history(outputs: dict[str, Any]) -> dict[str, Any]:
    path, _ = artifact(outputs, "circuitsage_hmac_v2_training_history_12a1d.csv")
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    canonical = [row for row in rows
                 if row.get("candidate_id") == EXPECTED["candidate_id"]
                 and row.get("replay", "").lower() in {"false", "0"}]
    replay = [row for row in rows
              if row.get("candidate_id") == EXPECTED["candidate_id"]
              and row.get("replay", "").lower() in {"true", "1"}]
    require(canonical and len(canonical) == len(replay), "selected-candidate history/replay rows")
    first = float(canonical[0]["mean_ranking_loss"])
    last = float(canonical[-1]["mean_ranking_loss"])
    require(np.isfinite([first, last]).all(), "finite training losses")
    exact = all(row == other for row, other in zip(
        [{**r, "replay": "X"} for r in canonical],
        [{**r, "replay": "X"} for r in replay],
    ))
    return {
        "path": rel(path),
        "epochs": len(canonical),
        "first_mean_ranking_loss": first,
        "last_mean_ranking_loss": last,
        "absolute_loss_change": last - first,
        "relative_loss_change": (last - first) / first if first else None,
        "canonical_replay_history_exact": exact,
    }


def ranking_diagnostics(outputs: dict[str, Any]) -> dict[str, Any]:
    path, _ = artifact(outputs, "circuitsage_hmac_v2_selected_calibration_rankings_12a1d.npz")
    with np.load(path, allow_pickle=False) as archive:
        require({"profile_index", "top_fault_instance_index", "top_score"}.issubset(archive.files),
                "calibration ranking members")
        profiles = np.asarray(archive["profile_index"], dtype=np.int64)
        top_faults = np.asarray(archive["top_fault_instance_index"], dtype=np.int64)
        scores = np.asarray(archive["top_score"], dtype=np.float64)
    require(top_faults.ndim == 2 and top_faults.shape[1] == TOP_K, "top-50 ranking shape")
    require(scores.shape == top_faults.shape and len(profiles) == len(top_faults), "ranking array agreement")
    require(np.all((0 <= top_faults) & (top_faults < FAULTS)), "ranked fault range")
    require(np.isfinite(scores).all(), "finite ranking scores")
    top_sites = top_faults[:, 0] // 2
    unique_top_sites, counts = np.unique(top_sites, return_counts=True)
    all_top_sites = np.unique(top_faults // 2)
    margins = scores[:, 0] - scores[:, 1]
    require(np.all(margins >= -1e-7), "top-score ordering")
    return {
        "path": rel(path),
        "calibration_profiles_ranked": int(len(profiles)),
        "unique_top1_fault_instances": int(np.unique(top_faults[:, 0]).size),
        "unique_top1_sites": int(unique_top_sites.size),
        "maximum_top1_site_share": float(counts.max() / len(top_sites)),
        "unique_sites_appearing_anywhere_in_top50": int(all_top_sites.size),
        "catalog_site_coverage_anywhere_in_top50": float(all_top_sites.size / SITES),
        "median_top1_top2_score_margin": float(np.median(margins)),
        "p95_top1_top2_score_margin": float(np.quantile(margins, 0.95)),
    }


def support_diagnostics(outputs: dict[str, Any]) -> dict[str, Any]:
    path, _ = artifact(outputs, "circuitsage_hmac_v2_frozen_v1_support_12a1d.npz")
    with np.load(path, allow_pickle=False) as archive:
        require("v1_probability" in archive.files, "V1 support cache member")
        support = np.asarray(archive["v1_probability"], dtype=np.float32)
    require(support.shape == (FAULTS, 64), "V1 support cache shape")
    require(np.isfinite(support).all() and np.all((support >= 0) & (support <= 1)),
            "V1 support probability range")
    binary = np.packbits(support >= 0.5, axis=1)
    _, binary_counts = np.unique(binary, axis=0, return_counts=True)
    rounded = np.round(support, decimals=3)
    rounded_unique = np.unique(rounded, axis=0).shape[0]
    feature_std = support.std(axis=0, dtype=np.float64)
    return {
        "path": rel(path),
        "shape": [FAULTS, 64],
        "binary_support_patterns_at_0_5": int(binary_counts.size),
        "largest_binary_support_collision_group": int(binary_counts.max()),
        "rounded_3dp_support_patterns": int(rounded_unique),
        "constant_or_near_constant_vector_features": int(np.sum(feature_std < 1e-8)),
        "mean_probability": float(support.mean(dtype=np.float64)),
        "mean_feature_standard_deviation": float(feature_std.mean()),
    }


def source_contract_checks() -> dict[str, Any]:
    text = SOURCE.read_text(encoding="utf-8")
    checks = {
        "training_universe_is_dev_train_only":
            'train_universe = targets["fault_instance_index"][targets["partition_code"] == 0]' in text,
        "negative_sampling_is_random": "rng.choice(universe" in text,
        "same_profile_positives_excluded": "integer not in positive_set" in text,
        "evaluation_scores_full_candidate_catalog": "scores = np.empty(len(c_embed)" in text,
        "candidate_contains_frozen_v1_support": "matrix[:, GRAPH_WIDTH + 2:] = support" in text,
        "evaluation_partition_default_is_calibration": "partition_code: int = 1" in text,
    }
    require(all(checks.values()), "Stage 12A-1D source semantics changed")
    return checks


def csv_bytes(rows: list[dict[str, Any]]) -> bytes:
    fields = ["metric", "observed", "required", "gap_to_requirement", "status"]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def self_test() -> None:
    payload = {"b": 2, "a": [1, 3]}
    require(canonical_json(payload) == canonical_json(payload), "canonical JSON replay")
    rows = [{"metric": "x", "observed": 0.1, "required": 0.2,
             "gap_to_requirement": 0.1, "status": "NOT_MET"}]
    require(csv_bytes(rows) == csv_bytes(rows), "CSV replay")
    require(math.isclose(NEGATIVES_PER_QUERY / FAULTS, 0.0028022243, abs_tol=1e-9),
            "negative-sampling fraction canary")
    print("Stage 12A-1D-R1 self-test: PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return

    require(ROOT.name == "vlsi_fault_detection_v2", f"run from project root, not {ROOT}")
    final_outputs = [DIAGNOSTICS, METRICS, REPORT, REPAIR_CONTRACT, MANIFEST, AUDIT]
    for path in final_outputs:
        require(not path.exists(), f"Stage {STAGE} output already exists: {rel(path)}")

    lock, manifest, prior_audit, evidence = verify_frozen_inputs()
    outputs = manifest["outputs"]
    selected, metrics_path = verify_metrics(lock, outputs)
    history = training_history(outputs)
    rankings = ranking_diagnostics(outputs)
    support = support_diagnostics(outputs)
    source_checks = source_contract_checks()

    failure_rows = [
        {
            "metric": "observable_candidate_coverage_top50",
            "observed": selected["observable_candidate_coverage_top50"],
            "required": TARGETS["observable_candidate_coverage_top50"],
            "gap_to_requirement": TARGETS["observable_candidate_coverage_top50"] - selected["observable_candidate_coverage_top50"],
            "status": "NOT_MET",
        },
        {
            "metric": "unique_signature_top1_site_accuracy",
            "observed": selected["unique_signature_top1_site_accuracy"],
            "required": TARGETS["unique_signature_top1_site_accuracy"],
            "gap_to_requirement": TARGETS["unique_signature_top1_site_accuracy"] - selected["unique_signature_top1_site_accuracy"],
            "status": "NOT_MET",
        },
        {
            "metric": "observable_top5_site_accuracy",
            "observed": selected["observable_top5_site_accuracy"],
            "required": TARGETS["observable_top5_site_accuracy"],
            "gap_to_requirement": TARGETS["observable_top5_site_accuracy"] - selected["observable_top5_site_accuracy"],
            "status": "NOT_MET",
        },
        {
            "metric": "fault_free_false_alarm_rate",
            "observed": selected["fault_free_false_alarm_rate"],
            "required": TARGETS["fault_free_false_alarm_rate"],
            "gap_to_requirement": TARGETS["fault_free_false_alarm_rate"] - selected["fault_free_false_alarm_rate"],
            "status": "PASS",
        },
        {
            "metric": "observable_detection_recall",
            "observed": selected["observable_detection_recall"],
            "required": TARGETS["observable_detection_recall"],
            "gap_to_requirement": selected["observable_detection_recall"] - TARGETS["observable_detection_recall"],
            "status": "PASS",
        },
    ]

    random_scale = {
        "uniform_random_top1_site": 1.0 / SITES,
        "uniform_random_top5_site": 5.0 / SITES,
        "uniform_random_top50_site": 50.0 / SITES,
        "observed_over_random_unique_top1": selected["unique_signature_top1_site_accuracy"] / (1.0 / SITES),
        "observed_over_random_top5": selected["observable_top5_site_accuracy"] / (5.0 / SITES),
        "observed_over_random_top50": selected["observable_candidate_coverage_top50"] / (50.0 / SITES),
        "interpretation": "scale reference only; not a statistical significance test",
    }

    confirmed = [
        "The exact golden-reference detector passed both calibration detection criteria.",
        "All three localization advancement criteria failed by large margins.",
        "Training sampled 128 random negatives from the DEV_TRAIN universe per query, while calibration ranked the full 45,678-fault catalog.",
        "The Stage 12A-1D objective optimized sampled ranking, not full-catalog top-1/top-5 retrieval.",
        "The frozen V1 support vector expresses detectability support across 64 vectors; it is not an exact candidate response generator.",
        "The observed ranker performs above a uniform-ranking scale reference, so useful localization signal exists but is insufficient.",
    ]
    hypotheses = [
        "Random negatives are too easy and too sparse to teach full-catalog discrimination.",
        "The separate response and candidate encoders do not expose enough direct per-vector response/support interactions.",
        "Unseen calibration-site scores are weakly controlled because the training candidate universe is DEV_TRAIN only.",
        "Ambiguous response profiles dilute exact-site learning, although they do not explain the very low unique-signature top-1 result by themselves.",
    ]

    diagnostics = {
        "diagnostic_version": VERSION,
        "stage": STAGE,
        "status": "PASS",
        "review_status": "COMPLETE / FROZEN",
        "prior_stage": "12A-1D",
        "prior_stage_status": "PASS WITH CALIBRATION ADVANCEMENT NOT_MET",
        "selected_candidate": EXPECTED["candidate_id"],
        "selected_metrics": selected,
        "acceptance_failures": [row for row in failure_rows if row["status"] == "NOT_MET"],
        "training_history": history,
        "ranking_diagnostics": rankings,
        "v1_support_diagnostics": support,
        "uniform_random_scale_reference": random_scale,
        "training_to_evaluation_mismatch": {
            "negative_candidates_per_query": NEGATIVES_PER_QUERY,
            "full_fault_catalog": FAULTS,
            "sampled_negative_fraction_per_query": NEGATIVES_PER_QUERY / FAULTS,
            "training_candidate_universe": "DEV_TRAIN FAULT INSTANCES",
            "calibration_ranking_universe": "ALL 45,678 FAULT INSTANCES",
        },
        "source_semantic_checks": source_checks,
        "confirmed_findings": confirmed,
        "root_cause_hypotheses_requiring_ablation": hypotheses,
        "single_root_cause_established": False,
        "models_deserialized": 0,
        "new_training_performed": False,
        "dev_site_test_opened": False,
        "validation_access_count": 0,
        "holdout_access_count": 0,
        "v1_artifacts_modified": False,
    }

    repair_contract = {
        "contract_version": "CIRCUITSAGE-HMAC-V2-LOCALIZATION-REPAIR-CONTRACT-v1",
        "stage": STAGE,
        "status": "FROZEN",
        "purpose": "repair calibration localization without weakening blinding or opening locked partitions",
        "unchanged_components": {
            "detector": "EXACT GOLDEN-REFERENCE GATE",
            "fault_scope": "SINGLE PERSISTENT SA0/SA1",
            "candidate_catalog": FAULTS,
            "v1_hybrid_model": "FROZEN / READ-ONLY SUPPORT",
            "golden_netlist_graph": "FROZEN / READ-ONLY",
            "acceptance_thresholds": TARGETS,
        },
        "authorized_data_use": {
            "DEV_TRAIN": "gradients, hard-negative mining, ablations, and deterministic replay",
            "DEV_CALIBRATION": "candidate selection and acceptance measurement only",
            "DEV_SITE_TEST": "LOCKED / NOT AUTHORIZED UNTIL CALIBRATION ADVANCEMENT PASSES",
            "VALIDATION": "PROHIBITED",
            "HOLDOUT": "PROHIBITED",
            "unknown_fault_identity_as_query_input": "PROHIBITED",
        },
        "required_repair_architecture": {
            "candidate_generation": "response-conditioned shortlist before neural reranking",
            "direct_interactions": [
                "query deviation mask × frozen 64-vector V1 support",
                "absolute response/support difference",
                "cosine and weighted agreement summaries",
            ],
            "graph_conditioning": "retain frozen K3 graph features and add response-conditioned feature crosses",
            "hard_negatives": [
                "highest-scoring incorrect DEV_TRAIN candidates",
                "response-support-near DEV_TRAIN candidates",
                "graph-neighbor DEV_TRAIN candidates",
                "SA0/SA1 sibling candidate",
            ],
            "losses": [
                "site-level multi-positive contrastive retrieval",
                "hard-negative margin or full-shortlist softmax",
                "conditional SA0/SA1 classification after site retrieval",
            ],
            "ambiguity_policy": "rank candidate sets for ambiguous signatures; never force an exact site",
            "shortlist_sizes_to_test": [256, 512, 1024],
            "minimum_ablation_set": [
                "V1-support similarity retrieval without neural reranker",
                "response-conditioned feature-cross ranker",
                "feature-cross ranker plus hard negatives",
                "feature-cross ranker plus hard negatives plus graph features",
            ],
        },
        "execution_limits": {
            "candidate_execution": "SEQUENTIAL",
            "parallel_candidates": 1,
            "maximum_new_trainable_candidates": 4,
            "checkpoint_resume": "REQUIRED",
            "deterministic_replay": "EXACT",
            "one_week_scope_preserved": True,
        },
        "advancement_gate": {
            "all_original_calibration_targets_must_pass": True,
            "acceptance_thresholds": TARGETS,
            "if_not_met": "FREEZE NEGATIVE RESULT; DO NOT OPEN DEV_SITE_TEST",
            "if_met": "authorize a new locked DEV_SITE_TEST evaluation stage",
        },
        "repair_training_authorization": "AUTHORIZED FOR DEV_TRAIN ONLY",
        "locked_evaluation_authorization": "NOT AUTHORIZED BY THIS CONTRACT",
        "retraining_of_v1": "PROHIBITED",
        "threshold_changes_to_v1": "PROHIBITED",
    }

    report = f"""# CircuitSage-HMAC V2 localization failure review — Stage {STAGE}

Stage 12A-1D trained correctly and replayed deterministically, but its detector and locator behaved very differently. The exact golden-reference detector achieved zero false alarms and 100% recall on observable calibration faults. The learned locator did not pass the advancement gate.

## Frozen calibration result

| Metric | Observed | Required | Result |
|---|---:|---:|---|
| Observable candidate coverage at top 50 | {selected['observable_candidate_coverage_top50']:.8f} | {TARGETS['observable_candidate_coverage_top50']:.2f} | NOT MET |
| Unique-signature top-1 site accuracy | {selected['unique_signature_top1_site_accuracy']:.8f} | {TARGETS['unique_signature_top1_site_accuracy']:.2f} | NOT MET |
| Observable top-5 site accuracy | {selected['observable_top5_site_accuracy']:.8f} | {TARGETS['observable_top5_site_accuracy']:.2f} | NOT MET |
| Fault-free false-alarm rate | {selected['fault_free_false_alarm_rate']:.8f} | ≤ {TARGETS['fault_free_false_alarm_rate']:.2f} | PASS |
| Observable detection recall | {selected['observable_detection_recall']:.8f} | ≥ {TARGETS['observable_detection_recall']:.2f} | PASS |

## What is confirmed

The training objective used only 128 randomly sampled DEV_TRAIN negatives per response, about {100 * NEGATIVES_PER_QUERY / FAULTS:.4f}% of the full fault catalog, while calibration ranked all {FAULTS:,} fault instances. The model learned some signal—the observed ranking metrics are above a uniform-ranking scale reference—but it did not learn enough full-catalog separation. The V1 support input describes which of 64 tests are likely to detect each candidate; it does not predict the candidate's exact output response.

These findings establish a training/evaluation mismatch and insufficient retrieval performance. They do not prove one single root cause. The frozen R2 contract therefore requires controlled ablations.

## Approved repair

Stage 12A-1D-R2 may train only on DEV_TRAIN. It must add direct response-to-candidate feature interactions, response-similar and graph-neighbor hard negatives, a shortlist retrieval step, and separate site retrieval from conditional SA0/SA1 prediction. DEV_CALIBRATION remains selection-only. DEV_SITE_TEST, VALIDATION, and HOLDOUT remain closed until every original calibration target passes.

No V1 artifact, threshold, graph, RTL, golden netlist, or frozen Stage 12A-1D output may be changed.
"""

    frozen_write(DIAGNOSTICS, canonical_json(diagnostics))
    frozen_write(METRICS, csv_bytes(failure_rows))
    frozen_write(REPORT, report.encode())
    frozen_write(REPAIR_CONTRACT, canonical_json(repair_contract))

    output_records = {rel(path): record(path) for path in (DIAGNOSTICS, METRICS, REPORT, REPAIR_CONTRACT)}
    review_manifest = {
        "manifest_version": "CIRCUITSAGE-HMAC-V2-LOCALIZATION-FAILURE-REVIEW-MANIFEST-v1",
        "stage": STAGE,
        "status": "PASS",
        "input_evidence": evidence,
        "verified_prior_output_count": len(outputs),
        "prior_metrics_csv": record(metrics_path),
        "outputs": output_records,
        "models_deserialized": 0,
        "new_training_performed": False,
        "dev_site_test_opened": False,
        "validation_access_count": 0,
        "holdout_access_count": 0,
    }
    frozen_write(MANIFEST, canonical_json(review_manifest))
    freeze = {
        "audit_version": "CIRCUITSAGE-HMAC-V2-LOCALIZATION-FAILURE-REVIEW-FREEZE-v1",
        "stage": STAGE,
        "status": "PASS",
        "review_status": "FROZEN / COMPLETE",
        "prior_calibration_advancement": "NOT_MET / PRESERVED",
        "detector_disposition": "PASS / PRESERVED",
        "locator_disposition": "REPAIR REQUIRED BEFORE LOCKED EVALUATION",
        "single_root_cause_established": False,
        "controlled_ablation_required": True,
        "repair_contract_status": "FROZEN",
        "repair_training": "AUTHORIZED FOR DEV_TRAIN ONLY",
        "dev_site_test": "LOCKED / NOT AUTHORIZED",
        "validation": "NOT ACCESSED / PROHIBITED",
        "holdout": "NOT ACCESSED / PROHIBITED",
        "v1_model_modified": False,
        "frozen_inputs_modified": False,
        "repair_contract": record(REPAIR_CONTRACT),
        "diagnostics": record(DIAGNOSTICS),
        "report": record(REPORT),
        "manifest": record(MANIFEST),
        "next_gate": "STAGE 12A-1D-R2 — RESPONSE-CONDITIONED LOCATOR REPAIR TRAINING AND CALIBRATION",
    }
    frozen_write(AUDIT, canonical_json(freeze))

    require(canonical_json(load_json(DIAGNOSTICS)) == DIAGNOSTICS.read_bytes(), "diagnostic JSON replay")
    require(canonical_json(load_json(REPAIR_CONTRACT)) == REPAIR_CONTRACT.read_bytes(), "contract JSON replay")
    require(canonical_json(load_json(MANIFEST)) == MANIFEST.read_bytes(), "manifest JSON replay")
    require(canonical_json(load_json(AUDIT)) == AUDIT.read_bytes(), "audit JSON replay")

    print("\nSTAGE 12A-1D-R1 — V2 LOCALIZATION FAILURE REVIEW AND REPAIR-CONTRACT FREEZE")
    print(f"{'Status':<39}: PASS")
    print(f"{'Review status':<39}: FROZEN / COMPLETE")
    print(f"{'Detector disposition':<39}: PASS / PRESERVED")
    print(f"{'Locator disposition':<39}: REPAIR REQUIRED")
    print(f"{'Calibration advancement':<39}: NOT_MET / PRESERVED")
    print(f"{'Confirmed design mismatch':<39}: 128 SAMPLED NEGATIVES vs 45,678 FULL-CATALOG RANKING")
    print(f"{'Single root cause established':<39}: NO — CONTROLLED ABLATION REQUIRED")
    print(f"{'New model training':<39}: NOT PERFORMED")
    print(f"{'Models deserialized':<39}: 0")
    print(f"{'Repair training':<39}: AUTHORIZED FOR DEV_TRAIN ONLY")
    print(f"{'DEV_SITE_TEST':<39}: LOCKED / NOT AUTHORIZED")
    print(f"{'VALIDATION / HOLDOUT access':<39}: 0 / 0")
    print(f"{'V1 model modified':<39}: NO")
    print(f"{'Diagnostics':<39}: {DIAGNOSTICS}")
    print(f"{'Diagnostics SHA':<39}: {sha256(DIAGNOSTICS)}")
    print(f"{'Repair contract':<39}: {REPAIR_CONTRACT}")
    print(f"{'Repair contract SHA':<39}: {sha256(REPAIR_CONTRACT)}")
    print(f"{'Report':<39}: {REPORT}")
    print(f"{'Report SHA':<39}: {sha256(REPORT)}")
    print(f"{'Manifest':<39}: {MANIFEST}")
    print(f"{'Manifest SHA':<39}: {sha256(MANIFEST)}")
    print(f"{'Audit':<39}: {AUDIT}")
    print(f"{'Audit SHA':<39}: {sha256(AUDIT)}")
    print(f"{'Next gate':<39}: STAGE 12A-1D-R2 — RESPONSE-CONDITIONED LOCATOR REPAIR TRAINING AND CALIBRATION")


if __name__ == "__main__":
    main()
