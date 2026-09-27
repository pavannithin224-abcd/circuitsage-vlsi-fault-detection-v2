#!/usr/bin/env python3
"""Stage 12A-1A: freeze the one-week CircuitSage-HMAC V2 Core contract.

This stage performs no model training, inference, simulation, validation access,
or HOLDOUT access.  It verifies the frozen V1/behavior/graph foundations and
emits deterministic contracts for the linked V2 detector-locator pipeline.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any


STAGE = "12A-1A"
ROOT = Path(__file__).resolve().parent

FROZEN_INPUTS = {
    "config/release/hmac_github_remote_integrity_lock_11e1k.json":
        "3d93cd8bb9c3bba47dd665ba6a45ad4e193613c1f6d386e32b3265471d94dd5a",
    "results/hmac_project_release_11e1k/hmac_github_remote_integrity_freeze_11e1k.json":
        "ca949a25697deb4bce317cf1e0e6181a68bca61ecbf111b252a21417aaff47a4",
    "config/release/hmac_final_project_disposition_11e1h.json":
        "32f27f32e1a6102304e42a8cc9c886702fddaa65ce4081251ecb95edb4f5d6d7",
    "config/diagnostic_model/hmac_final_diagnostic_model_lock_11d2d.json":
        "7985b534c62d93717a179d6d2b20f247a531ec0452003e35f0f65cf640d0b30d",
    "results/hmac_fault_campaign_11d2/hybrid_training_11d2b/hmac_selected_hybrid_model_11d2b.joblib":
        "12fea5eabf4a6c605325b6ce4c1217f6a37cc59750065db8b85c3da14713f6b8",
    "results/hmac_behavior_diagnosis_pilot/hmac_behavior_diagnosis_pilot_disposition_freeze_11e1g.json":
        "3a402e789346bdc99fbe0ef3629031e45d5f4b9556ece5b91de28e5bace1c7a4",
    "results/hmac_behavior_diagnosis_pilot/execution_11e1f/freeze.json":
        "b4a83f640bf0ea024625dc8c2e59d6d30f6048e72b0eb7cb979a911f968a2677",
    "results/hmac_fault_campaign_11d1/graph_dataset_11d1a/hmac_golden_netlist_graph_11d1a.npz":
        "e3c2dd2214b544231186c29d8d9cb5aa6621b4d4b4bc9002150ac4f6c208c052",
    "results/hmac_fault_campaign_11d1/graph_dataset_11d1a/hmac_golden_netlist_graph_schema_11d1a.json":
        "d24c3dd285891e284d20d1adcd93f793bedfd3042f20159a8a486359bc25e393",
    "results/hmac_fault_campaign_11c5/canonical_dataset_11c5c/hmac_fault_campaign_canonical_train_11c5c.csv.gz":
        "dcc79c94fa3a8d592b516e5f5683b1bf007682745103a9419ee20ef20a7ec78f",
}

CONFIG_DIR = ROOT / "config" / "v2"
RESULT_DIR = ROOT / "results" / "circuitsage_hmac_v2_12a1"

SCOPE_PATH = CONFIG_DIR / "circuitsage_hmac_v2_core_scope_contract_12a1a.json"
PIPELINE_PATH = CONFIG_DIR / "circuitsage_hmac_v2_linked_pipeline_contract_12a1a.json"
ACCEPTANCE_PATH = CONFIG_DIR / "circuitsage_hmac_v2_acceptance_contract_12a1a.json"
AUDIT_PATH = RESULT_DIR / "circuitsage_hmac_v2_core_contract_freeze_12a1a.json"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def write_frozen(path: Path, payload: Any) -> None:
    encoded = canonical_json(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(path.read_bytes() == encoded, f"existing frozen output differs: {path}")
        return
    path.write_bytes(encoded)


def valid_json(path: Path) -> None:
    try:
        json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid frozen JSON: {path}: {error}") from error


def main() -> None:
    print("STAGE 12A-1A — CIRCUITSAGE-HMAC V2 CORE CONTRACT")
    print("FROZEN INPUT VERIFICATION")

    verified: dict[str, str] = {}
    for relative, expected in FROZEN_INPUTS.items():
        path = ROOT / relative
        require(path.is_file(), f"missing frozen input: {relative}")
        actual = sha256_file(path)
        require(actual == expected, f"SHA mismatch for {relative}: expected {expected}, actual {actual}")
        if path.suffix == ".json":
            valid_json(path)
        verified[relative] = actual
        print(f"  {path.name:<76}: OK")

    scope = {
        "contract_version": "CIRCUITSAGE-HMAC-V2-CORE-SCOPE-v1",
        "stage": STAGE,
        "status": "FROZEN",
        "project": "CircuitSage-HMAC V2 Core",
        "parent_release": {
            "name": "CircuitSage-HMAC v1.0.0",
            "repository": "https://github.com/pavannithin224-abcd/circuitsage-hmac-fault-detection",
            "tag": "v1.0.0",
            "commit_sha": "7460df3f4cd32e4bcf9f9bc2f294f4d72c6c02b7",
            "state": "IMMUTABLE FROZEN BASELINE",
        },
        "timebox": {
            "duration_calendar_days": 7,
            "estimated_effort_hours": "50-70",
            "freeze_at_end_even_if_targets_fail": True,
        },
        "supported_scope": {
            "circuit": "OpenTitan HMAC-SHA256 msg32 wrapper",
            "fault_models": ["SA0", "SA1"],
            "fault_multiplicity": "ONE PERSISTENT FAULT",
            "physical_sites": 22839,
            "fault_instances": 45678,
            "observation_vectors": "64 FROZEN TRAIN VECTORS",
            "primary_tasks": [
                "BEHAVIOR-BASED UNKNOWN-FAULT DETECTION",
                "RANKED PHYSICAL-SITE LOCALIZATION",
                "FROZEN-V1 CANDIDATE VERIFICATION AND RERANKING",
            ],
        },
        "deferred_scope": [
            "TRANSIENT FAULTS",
            "BRIDGING FAULTS",
            "MULTIPLE SIMULTANEOUS FAULTS",
            "CROSS-CIRCUIT OR INDEPENDENT-CHIP GENERALIZATION",
            "QUANTIZATION",
            "DASHBOARD OR FRONTEND",
            "PRODUCTION OR SILICON-READINESS CLAIMS",
        ],
        "authorizations": {
            "response_dataset_construction": "AUTHORIZED FROM TRAIN PARTITION ONLY",
            "v2_model_architecture_contract": "AUTHORIZED",
            "v2_training": "NOT YET AUTHORIZED",
            "validation_inference": "NOT YET AUTHORIZED",
            "holdout_access": "BLOCKED UNTIL FINAL V2 MODEL FREEZE",
        },
        "prohibitions": [
            "MODIFYING OR RETRAINING V1",
            "USING FAULT IDENTITY, SITE ID, OR FAULT LABEL AS QUERY INPUT",
            "USING POST-SIMULATION OUTCOMES AS DETECTOR INPUT FEATURES",
            "OPENING HOLDOUT BEFORE MODEL, THRESHOLD, AND RANKER LOCK",
            "RETRAINING OR THRESHOLD CHANGES AFTER HOLDOUT ACCESS",
        ],
    }

    pipeline = {
        "contract_version": "CIRCUITSAGE-HMAC-V2-LINKED-PIPELINE-v1",
        "stage": STAGE,
        "status": "FROZEN",
        "user_input": {
            "required": [
                "TEST-VECTOR IDENTIFIERS OR ORDERED VECTOR COMMITMENT",
                "OBSERVED CIRCUIT RESPONSES",
            ],
            "forbidden": [
                "FAULT INSTANCE ID",
                "PHYSICAL SITE ID",
                "FAULT MODEL LABEL SA0/SA1",
                "GROUND-TRUTH DETECTION LABEL",
            ],
        },
        "stages": [
            {
                "order": 1,
                "component": "V2 BEHAVIOR DETECTOR",
                "purpose": "decide whether the response contains an observable anomaly",
            },
            {
                "order": 2,
                "component": "V2 GRAPH LOCATOR",
                "purpose": "rank physical sites and SA0/SA1 candidates from response and topology",
            },
            {
                "order": 3,
                "component": "FROZEN V1 HYBRID FUSION MLP",
                "purpose": "supply a fixed detectability-support feature for each proposed candidate",
                "architecture": "646 -> 128 -> 64 -> 32 -> 1",
                "trainable_parameters": 93185,
                "state": "FROZEN; NO WEIGHT OR THRESHOLD CHANGE",
            },
            {
                "order": 4,
                "component": "V2 CALIBRATED RERANKER",
                "purpose": "combine V2 location evidence with frozen V1 support without naive probability averaging",
            },
        ],
        "required_outputs": [
            "NO_OBSERVED_ANOMALY OR FAULT_DETECTED",
            "DETECTION CONFIDENCE",
            "TOP-1/TOP-5/TOP-10 PHYSICAL-SITE CANDIDATES",
            "SA0/SA1 CANDIDATE TYPE WHEN RESOLVABLE",
            "V1 DETECTABILITY-SUPPORT SCORE",
            "AMBIGUOUS/INVISIBLE/OUT-OF-CATALOG WARNING",
        ],
        "fallback": {
            "uncertain_detection": "RETURN UNCERTAIN; DO NOT FORCE A FAULT",
            "unresolved_location": "RETURN RANKED CANDIDATE SET OR NO-CATALOG-MATCH",
            "v2_runtime_failure": "ALLOW EXPLICIT V1 CANDIDATE-ANALYSIS MODE; NEVER MISLABEL AS UNKNOWN-FAULT DIAGNOSIS",
        },
    }

    acceptance = {
        "contract_version": "CIRCUITSAGE-HMAC-V2-ACCEPTANCE-v1",
        "stage": STAGE,
        "status": "FROZEN",
        "selection_metric": "PHYSICAL-SITE-GROUPED MCC FOR DETECTION; MRR FOR LOCALIZATION",
        "development_rules": {
            "allowed_vectors": "TRAIN ONLY",
            "grouping_unit": "PHYSICAL SITE; SA0 AND SA1 MUST REMAIN TOGETHER",
            "development_labels": "ALLOWED ONLY INSIDE TRAIN/CALIBRATION SCORING",
            "historical_validation": "REPORT-ONLY EVIDENCE; NOT A V2 SELECTION PARTITION",
            "holdout": "SEALED UNTIL STAGE 12A-1F AUTHORIZATION",
        },
        "minimum_advancement_targets": {
            "fault_free_false_alarm_rate_max": 0.05,
            "observable_fault_detection_recall_min": 0.90,
            "observable_candidate_coverage_min": 0.95,
            "unique_signature_top1_site_accuracy_min": 0.80,
            "observable_top5_site_accuracy_min": 0.80,
            "deterministic_replay": "EXACT",
            "unknown_identity_feature_count": 0,
            "post_simulation_target_feature_count": 0,
        },
        "required_reported_metrics": {
            "detection": [
                "MCC", "BALANCED_ACCURACY", "PRECISION", "RECALL", "SPECIFICITY",
                "F1", "PR_AUC", "ROC_AUC", "FALSE_ALARM_RATE",
            ],
            "localization": [
                "TOP1_SITE_ACCURACY", "TOP5_SITE_ACCURACY", "TOP10_SITE_ACCURACY",
                "MEAN_RECIPROCAL_RANK", "CANDIDATE_COVERAGE",
                "MEAN_CANDIDATE_SET_SIZE", "UNRESOLVED_RATE",
            ],
            "breakdowns": ["SA0_VS_SA1", "SITE_CATEGORY", "DRIVER_CELL_TYPE", "OBSERVABILITY_CLASS"],
        },
        "holdout_protocol": {
            "execution_count": 1,
            "prerequisite": "FINAL DETECTOR, LOCATOR, V1 LINK, RERANKER, AND THRESHOLDS FROZEN",
            "after_access": "NO RETRAINING, RESELECTION, FEATURE CHANGES, OR THRESHOLD CHANGES",
            "failure_disposition": "FREEZE AND REPORT NOT_MET; DO NOT TUNE ON HOLDOUT",
        },
        "final_disposition": {
            "deadline": "END OF SEVEN-DAY V2 CORE SPRINT",
            "outcomes": ["PASS_AND_FROZEN", "RESEARCH_RESULT_NOT_MET_AND_FROZEN"],
            "production_claim_allowed": False,
        },
    }

    # The in-memory replay check catches accidental nondeterministic fields.
    for name, payload in (("scope", scope), ("pipeline", pipeline), ("acceptance", acceptance)):
        require(canonical_json(payload) == canonical_json(json.loads(canonical_json(payload))),
                f"deterministic replay failed for {name}")

    write_frozen(SCOPE_PATH, scope)
    write_frozen(PIPELINE_PATH, pipeline)
    write_frozen(ACCEPTANCE_PATH, acceptance)

    output_hashes = {
        str(SCOPE_PATH.relative_to(ROOT)): sha256_file(SCOPE_PATH),
        str(PIPELINE_PATH.relative_to(ROOT)): sha256_file(PIPELINE_PATH),
        str(ACCEPTANCE_PATH.relative_to(ROOT)): sha256_file(ACCEPTANCE_PATH),
    }

    audit = {
        "audit_version": "CIRCUITSAGE-HMAC-V2-CORE-CONTRACT-FREEZE-v1",
        "stage": STAGE,
        "status": "PASS",
        "scope_status": "FROZEN",
        "linked_pipeline_status": "FROZEN",
        "blinding_status": "FROZEN",
        "acceptance_status": "FROZEN",
        "v1_status": "VERIFIED AND PRESERVED",
        "unknown_fault_identity_in_input": "PROHIBITED",
        "training_performed": False,
        "model_inference_performed": False,
        "simulation_performed": False,
        "validation_access_count": 0,
        "holdout_access_count": 0,
        "frozen_inputs": verified,
        "frozen_outputs": output_hashes,
        "next_gate": "STAGE 12A-1B — BLINDED TRAIN-RESPONSE DATASET CONSTRUCTION AND SCHEMA FREEZE",
    }
    write_frozen(AUDIT_PATH, audit)

    print("\nSTAGE 12A-1A — V2 CORE SCOPE, BLINDING, LINKAGE, AND ACCEPTANCE FREEZE")
    print(f"{'Status':<31}: PASS")
    print(f"{'V2 Core scope':<31}: FROZEN")
    print(f"{'One-week timebox':<31}: FROZEN — 7 CALENDAR DAYS")
    print(f"{'Unknown fault identity input':<31}: PROHIBITED")
    print(f"{'V1 hybrid component':<31}: VERIFIED / FROZEN / LINKED")
    print(f"{'V2 detector':<31}: CONTRACTED; NOT TRAINED")
    print(f"{'V2 graph locator':<31}: CONTRACTED; NOT TRAINED")
    print(f"{'V1 role':<31}: CANDIDATE VERIFICATION AND RERANKING SUPPORT")
    print(f"{'Fault scope':<31}: SINGLE PERSISTENT SA0/SA1")
    print(f"{'Validation access':<31}: 0")
    print(f"{'HOLDOUT access':<31}: 0 / BLOCKED")
    print(f"{'V1 artifacts modified':<31}: NO")
    print(f"{'Scope contract':<31}: {SCOPE_PATH}")
    print(f"{'Scope SHA':<31}: {output_hashes[str(SCOPE_PATH.relative_to(ROOT))]}")
    print(f"{'Pipeline contract':<31}: {PIPELINE_PATH}")
    print(f"{'Pipeline SHA':<31}: {output_hashes[str(PIPELINE_PATH.relative_to(ROOT))]}")
    print(f"{'Acceptance contract':<31}: {ACCEPTANCE_PATH}")
    print(f"{'Acceptance SHA':<31}: {output_hashes[str(ACCEPTANCE_PATH.relative_to(ROOT))]}")
    print(f"{'Audit':<31}: {AUDIT_PATH}")
    print(f"{'Audit SHA':<31}: {sha256_file(AUDIT_PATH)}")
    print(f"{'Next gate':<31}: STAGE 12A-1B — BLINDED TRAIN-RESPONSE DATASET CONSTRUCTION AND SCHEMA FREEZE")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:  # concise stage-style failure reporting
        print(f"STOP: {error}", file=sys.stderr)
        raise SystemExit(1)
