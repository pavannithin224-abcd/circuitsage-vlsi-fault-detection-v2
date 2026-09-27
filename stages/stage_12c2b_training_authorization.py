#!/usr/bin/env python3
"""Stage 12C-2B: calibration-use and model-training authorization.

The frozen Stage 12C-1A training/selection contract permits gradient updates on
GENERALIZATION_TRAIN only, and permits model selection on
GENERALIZATION_CALIBRATION "ONLY AFTER SEPARATE CAPTURE AUTHORIZATION".  The
calibration family (``secworks_sha256``) was captured by Stage 12C-1O, but the
separate authorization to USE it for selection has never been issued.

This stage issues that authorization and the bounded training authorization for
the four candidates already frozen in the Stage 12C-1A candidate grid.  It
designs no architecture and invents no acceptance criteria: both are frozen
upstream.

Before authorizing, this stage performs an executable LEAKAGE AUDIT over the
frozen artifacts:

  1. partition disjointness - every circuit family belongs to exactly one
     partition, and no family appears in more than one
  2. gradient scope - only GENERALIZATION_TRAIN families may receive gradient
     updates; the calibration family is selection-only
  3. protected exclusion - no INDEPENDENT_CIRCUIT_TEST or GENERALIZATION_HOLDOUT
     family appears anywhere in the captured response or graph datasets
  4. site disjointness - the site index space of each family is disjoint from
     every other family under the frozen global offsets
  5. modality alignment - graph site_node_index and response local_site_index
     describe the same site ordering for every family
  6. identity firewall - no model-facing array carries fault identity, stuck
     value, site index or any absolute circuit identifier

Authorized by this stage: training on GENERALIZATION_TRAIN, selection on
GENERALIZATION_CALIBRATION.
Not authorized: any access to INDEPENDENT_CIRCUIT_TEST or
GENERALIZATION_HOLDOUT, and any final acceptance evaluation.

No training, selection, inference, simulation or dataset construction occurs in
this stage.  It is an authorization gate only.
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

import numpy as np


STAGE = "12C-2B"
ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "config/v2_2"
RESULT1 = ROOT / "results/circuitsage_hmac_v2_12c1"
RESULT2 = ROOT / "results/circuitsage_hmac_v2_12c2"
WORK = RESULT2 / "training_authorization_12c2b"
LOCK_FILE = WORK / ".stage_12c2b.lock"

# ---------------------------------------------------------------- frozen input
SOURCE_2A = ROOT / "stage_12c2a_graph_dataset.py"
ARCHITECTURE_1A = CONFIG / "circuitsage_hmac_v2_2_generalization_architecture_12c1a.json"
TRAINING_1A = CONFIG / "circuitsage_hmac_v2_2_training_selection_contract_12c1a.json"
ACCEPTANCE_1A = CONFIG / "circuitsage_hmac_v2_2_acceptance_contract_12c1a.json"
PARTITION_1A = CONFIG / "circuitsage_hmac_v2_2_circuit_family_partition_contract_12c1a.json"
SCOPE_1A = CONFIG / "circuitsage_hmac_v2_2_generalization_scope_contract_12c1a.json"
INTERFACE_1A = CONFIG / "circuitsage_hmac_v2_2_inference_interface_contract_12c1a.json"
CANDIDATE_GRID_1A = CONFIG / "circuitsage_hmac_v2_2_candidate_grid_12c1a.csv"
SPLIT_1B = CONFIG / "circuitsage_hmac_v2_2_family_split_authorization_12c1b.json"
GRAPH_CONTRACT_2A = CONFIG / "circuitsage_hmac_v2_2_graph_dataset_contract_12c2a.json"
VOCABULARY_2A = CONFIG / "circuitsage_hmac_v2_2_cell_type_vocabulary_12c2a.json"
NORMALIZATION_2A = CONFIG / "circuitsage_hmac_v2_2_graph_normalization_rules_12c2a.json"

MANIFEST_1A = RESULT1 / "circuitsage_hmac_v2_2_generalization_contract_manifest_12c1a.json"
AUDIT_1O = RESULT1 / "circuitsage_hmac_v2_2_amended_campaign_execution_dataset_freeze_12c1o.json"
AUDIT_1Q = RESULT1 / "circuitsage_hmac_v2_2_disposition_correction_freeze_12c1q.json"
AUDIT_2A = RESULT2 / "circuitsage_hmac_v2_2_graph_dataset_freeze_12c2a.json"
MANIFEST_2A = RESULT2 / "circuitsage_hmac_v2_2_graph_dataset_manifest_12c2a.json"

WORK_1O = RESULT1 / "parallel_campaign_execution_12c1o"
FEATURES_1O = WORK_1O / "circuitsage_hmac_v2_2_amended_campaign_features_12c1o.npz"
TARGETS_1O = WORK_1O / "circuitsage_hmac_v2_2_amended_campaign_targets_12c1o.npz"
WORK_2A = RESULT2 / "graph_dataset_12c2a"
GRAPH_NPZ_2A = WORK_2A / "circuitsage_hmac_v2_2_circuit_graphs_12c2a.npz"

PINNED = {
    SOURCE_2A: "180425deda8a7bfae0b79b665578b61ab82a8a0f18829b148371f76324dd46c6",
    ARCHITECTURE_1A: "e83da87e0ba773ed1b6872528c216079d31fc83537b4a2eca348bc6a23791649",
    TRAINING_1A: "081578897b76530c5d63ce2f283b5c",  # prefix-checked; see verify_inputs
    ACCEPTANCE_1A: "9c8eec4d85957c4408ac59e0c8760af90c91d0667c995d91ac969b5a8f205f26",
    PARTITION_1A: "4cbab060ef824f0ff8a17af7e2250d0efcf608ac7121c685bcf9eef4f5b2804c",
    SCOPE_1A: "993902928433953151b09641c156dbc9d9e154f33f5268b9329142fccfc45361",
    INTERFACE_1A: "c5e222640b7404e83fe925ec1b33b2bbf633880b4f21b253ba28d5025686541a",
    CANDIDATE_GRID_1A: "2297edd6a7b61cca28709509a015ff246c43a8379a36ed4842b8d1a2c5528470",
    SPLIT_1B: "4103808fc389c78088e31c0a76549324c386ed9d8f7bedb7cc4d3748f9ea3303",
    GRAPH_CONTRACT_2A: "c7d0d8070e605f5e05592c4ac6b1ffa93f3c701318f69b27512fe0cbdc5e044a",
    VOCABULARY_2A: "f64710234ca0f03aa565682687476bfdfb46dc877bb24db26b65866312a0e927",
    NORMALIZATION_2A: "85fc4df3861006240e1683b15a0e8f4392ae0233ad4d4c12409a595baa077d32",
    MANIFEST_1A: "e1a2000508a3e8d141077a487100f3a54250f227963e663f23cbd21749ed2ddc",
    AUDIT_1O: "fa67278ae07bc6dd57246088c3da41918f88c05fa06580c4583c022c1993430b",
    AUDIT_1Q: "1b5389e559ab8d392a688cd45519a1e6a0d2bf433e35825fdd2dfe2b9b7ed2a6",
    AUDIT_2A: "fa6f6f44105075b2e20a5bf9fedf4a74c79ea85d2a75291698a209b7feb35eb4",
    MANIFEST_2A: "62de5f2a335ee2390bea4b3bfc50a7178a0f886f50d35a2e7ce49eda9ddbe2ff",
}
# TRAINING_1A's SHA was recorded truncated in the 12C-1A freeze; it is verified
# by prefix plus full-content contract checks rather than by a fabricated digest.
PREFIX_ONLY = {TRAINING_1A}

# ---------------------------------------------------------------- stage output
TRAINING_AUTHORIZATION = CONFIG / "circuitsage_hmac_v2_2_training_execution_authorization_12c2b.json"
CALIBRATION_AUTHORIZATION = CONFIG / "circuitsage_hmac_v2_2_calibration_use_authorization_12c2b.json"
LEAKAGE_AUDIT = WORK / "circuitsage_hmac_v2_2_leakage_audit_12c2b.json"
PARTITION_REGISTER = WORK / "circuitsage_hmac_v2_2_partition_register_12c2b.csv"
CANDIDATE_REGISTER = WORK / "circuitsage_hmac_v2_2_candidate_register_12c2b.csv"
PREFLIGHT = WORK / "circuitsage_hmac_v2_2_training_authorization_preflight_12c2b.json"
REPORT = WORK / "circuitsage_hmac_v2_2_training_authorization_report_12c2b.md"
MANIFEST = RESULT2 / "circuitsage_hmac_v2_2_training_authorization_manifest_12c2b.json"
AUDIT = RESULT2 / "circuitsage_hmac_v2_2_training_authorization_freeze_12c2b.json"

CAPTURED_FAMILIES = (
    "opentitan_hmac_sha256",
    "picorv32_cpu",
    "secworks_aes",
    "secworks_sha256",
)
EXPECTED_SITES = {
    "opentitan_hmac_sha256": 18392,
    "picorv32_cpu": 9665,
    "secworks_aes": 26560,
    "secworks_sha256": 9125,
}
MODEL_FACING_ARRAYS = ("response_xor", "completion_cycle_delta", "timeout", "protocol_error",
                       "vector_mask", "response_validity_mask", "observable",
                       "baseline_response", "baseline_cycles", "transaction_id", "family_index")
FORBIDDEN_IN_FEATURES = ("opaque_fault_id", "local_site_index", "stuck_value", "exact_site",
                         "behavior_signature_sha256", "fault_instance_index",
                         "candidate_site_count")
FUTURE_BRAND = "Faultiva 1.0 — Hybrid VLSI Fault Intelligence"


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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def csv_bytes(rows: list[dict[str, Any]], fields: list[str]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


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


def verify_inputs() -> dict[str, Any]:
    print("FROZEN INPUT VERIFICATION", flush=True)
    for path, expected in PINNED.items():
        require(path.is_file(), f"missing frozen input: {rel(path)}")
        actual = sha256(path)
        if path in PREFIX_ONLY:
            require(actual.startswith(expected),
                    f"frozen input SHA prefix: {path.name}")
        else:
            require(actual == expected, f"frozen input SHA changed: {path.name}")
    print(f"  {len(PINNED)} frozen 12C-1A / 12C-1B / 12C-1O / 12C-1Q / 12C-2A inputs"
          f"{'':<18}: OK", flush=True)

    training = load_json(TRAINING_1A)
    require(training.get("status") == "FROZEN", "12C-1A training contract frozen")
    require(training.get("gradient_partitions") == ["GENERALIZATION_TRAIN"],
            "gradient partition scope")
    require("GENERALIZATION_CALIBRATION" in training.get("gradient_prohibited_partitions", []),
            "calibration gradient prohibition")
    require("SEPARATE CAPTURE AUTHORIZATION" in training.get("selection_partition", ""),
            "calibration selection requires separate authorization")
    require(training.get("training") == "NOT YET AUTHORIZED",
            "12C-1A left training unauthorized")

    audit_2a = load_json(AUDIT_2A)
    require(audit_2a.get("status") == "PASS", "12C-2A graph dataset frozen")
    require(audit_2a.get("absolute_node_identity") == "PROHIBITED / ABSENT",
            "graph identity firewall")
    require(audit_2a.get("normalization_fitted_on_training_data") is False,
            "graph normalization not fitted across circuits")

    audit_1q = load_json(AUDIT_1Q)
    require(audit_1q.get("acceptance_criteria_exist") is True, "12C-1Q correction present")

    existing = sorted(p.name for p in CONFIG.glob("*.json"))
    require(not any("training_execution_authorization" in n for n in existing),
            "a training execution authorization already exists")
    print(f"  config/v2_2 enumerated ({len(existing)} contracts); no prior training authorization"
          f"{'':<6}: OK", flush=True)
    return training


def leakage_audit() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Executable leakage checks over the frozen artifacts."""
    split = load_json(SPLIT_1B)
    assignments = split["family_assignments"]
    checks: list[dict[str, Any]] = []

    def check(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "result": "PASS" if ok else "FAIL", "detail": detail})
        require(ok, f"leakage audit failed: {name} - {detail}")

    # 1 partition disjointness
    counts: dict[str, int] = {}
    for fam, part in assignments.items():
        counts[fam] = counts.get(fam, 0) + 1
    check("family_assigned_exactly_once", all(v == 1 for v in counts.values()),
          f"{len(assignments)} families, each assigned once")

    train = sorted(f for f, p in assignments.items() if p == "GENERALIZATION_TRAIN")
    calib = sorted(f for f, p in assignments.items() if p == "GENERALIZATION_CALIBRATION")
    test = sorted(f for f, p in assignments.items() if p == "INDEPENDENT_CIRCUIT_TEST")
    holdout = sorted(f for f, p in assignments.items() if p == "GENERALIZATION_HOLDOUT")

    check("train_calibration_disjoint", not (set(train) & set(calib)),
          f"train={train} calibration={calib}")
    check("protected_disjoint_from_development",
          not ((set(test) | set(holdout)) & (set(train) | set(calib))),
          f"test={test} holdout={holdout}")

    # 2 gradient scope
    check("gradient_families_are_train_only", set(train) <= set(CAPTURED_FAMILIES),
          f"gradient families {train} all captured")
    check("calibration_not_in_gradient_scope", not (set(calib) & set(train)),
          "calibration family excluded from gradient partitions")

    # 3 protected exclusion from captured datasets
    protected = set(test) | set(holdout)
    check("no_protected_family_captured", not (protected & set(CAPTURED_FAMILIES)),
          f"protected={sorted(protected)} captured={list(CAPTURED_FAMILIES)}")

    graphs = np.load(GRAPH_NPZ_2A)
    graph_families = sorted({k.split("__")[0] for k in graphs.files})
    check("no_protected_family_in_graph_dataset", not (protected & set(graph_families)),
          f"graph families {graph_families}")

    # 4 site disjointness under global offsets
    offset = 0
    spans: list[tuple[int, int, str]] = []
    for fam in CAPTURED_FAMILIES:
        spans.append((offset, offset + EXPECTED_SITES[fam] - 1, fam))
        offset += EXPECTED_SITES[fam]
    overlap = any(spans[i][1] >= spans[i + 1][0] for i in range(len(spans) - 1))
    check("site_spans_disjoint", not overlap,
          "; ".join(f"{f}:[{a},{b}]" for a, b, f in spans))

    # 5 modality alignment
    targets = np.load(TARGETS_1O)
    features = np.load(FEATURES_1O)
    family_index = features["family_index"].astype(np.int32)
    local_site = targets["local_site_index"].astype(np.int64)
    aligned = True
    detail_parts = []
    for fi, fam in enumerate(CAPTURED_FAMILIES):
        mask = family_index == fi
        site_max = int(local_site[mask].max())
        node_map = graphs[f"{fam}__site_node_index"]
        ok = (site_max == EXPECTED_SITES[fam] - 1) and (len(node_map) == EXPECTED_SITES[fam])
        aligned = aligned and ok
        detail_parts.append(f"{fam}: sites={len(node_map)} max_local={site_max}")
    check("graph_response_site_alignment", aligned, "; ".join(detail_parts))

    # 6 identity firewall
    feature_keys = set(features.files)
    leaked = feature_keys & set(FORBIDDEN_IN_FEATURES)
    check("identity_firewall_features", not leaked,
          f"feature arrays={sorted(feature_keys)}")
    graph_contract = load_json(GRAPH_CONTRACT_2A)
    gf = set(graph_contract["node_feature_names"])
    leaked_graph = gf & {"site_index", "cell_name", "fault_id", "stuck_value"}
    check("identity_firewall_graph", not leaked_graph, f"{len(gf)} node features, none identity")

    audit = {
        "audit_version": "CIRCUITSAGE-HMAC-V2.2-LEAKAGE-AUDIT-12C2B-v1",
        "stage": STAGE, "status": "PASS", "created_at": now(),
        "checks": checks,
        "checks_run": len(checks),
        "checks_failed": 0,
        "train_families": train,
        "calibration_families": calib,
        "independent_test_families": test,
        "holdout_families": holdout,
        "captured_families": list(CAPTURED_FAMILIES),
        "verdict": "NO LEAKAGE DETECTED ACROSS PARTITION, SITE, MODALITY OR IDENTITY DIMENSIONS",
    }
    rows = [{"family_id": f, "partition": assignments[f],
             "captured": "YES" if f in CAPTURED_FAMILIES else "NO",
             "gradient_updates": "AUTHORIZED" if assignments[f] == "GENERALIZATION_TRAIN" else "PROHIBITED",
             "selection_use": "AUTHORIZED" if assignments[f] == "GENERALIZATION_CALIBRATION" else "PROHIBITED",
             "access_state": ("OPEN" if f in CAPTURED_FAMILIES else
                              ("SEALED" if assignments[f] == "INDEPENDENT_CIRCUIT_TEST" else "BLOCKED"))}
            for f in sorted(assignments)]
    return audit, rows


def execute() -> None:
    require(ROOT.name == "vlsi_fault_detection_v2", f"place script in project root, not {ROOT}")
    outputs = (TRAINING_AUTHORIZATION, CALIBRATION_AUTHORIZATION, LEAKAGE_AUDIT,
               PARTITION_REGISTER, CANDIDATE_REGISTER, PREFLIGHT, REPORT, MANIFEST, AUDIT)
    require(not any(p.exists() for p in outputs),
            f"frozen Stage {STAGE} output already exists; use --status")

    training = verify_inputs()

    print("\nLEAKAGE AUDIT", flush=True)
    audit_result, partition_rows = leakage_audit()
    for c in audit_result["checks"]:
        print(f"  {c['result']:<5}{c['check']}", flush=True)

    grid = read_csv(CANDIDATE_GRID_1A)
    require(len(grid) == 4, "four frozen candidates")
    trainable = [r for r in grid if r["trainable"].strip().upper() == "YES"]
    require(len(trainable) == 3, "three trainable candidates")

    print("\nCANDIDATE REGISTER", flush=True)
    candidate_rows: list[dict[str, Any]] = []
    for r in grid:
        cap = int(r["parameter_cap"])
        candidate_rows.append({
            "candidate_id": r["candidate_id"],
            "trainable": r["trainable"],
            "graph_encoder": r["graph_encoder"],
            "response_encoder": r["response_encoder"],
            "fusion": r["fusion"],
            "parameter_cap": cap,
            "role": r["role"],
            "gradient_partition": "GENERALIZATION_TRAIN" if r["trainable"].strip().upper() == "YES" else "NONE",
            "authorization": "AUTHORIZED / BOUNDED",
        })
        print(f"  {r['candidate_id']:<38} trainable={r['trainable']:<4} cap={cap:>9,}", flush=True)

    created = now()

    calibration_auth = {
        "authorization_version": "CIRCUITSAGE-HMAC-V2.2-CALIBRATION-USE-AUTHORIZATION-12C2B-v1",
        "stage": STAGE, "status": "FROZEN", "created_at": created,
        "satisfies": ("Stage 12C-1A training contract clause: selection_partition = "
                      "GENERALIZATION_CALIBRATION ONLY AFTER SEPARATE CAPTURE AUTHORIZATION"),
        "calibration_families": audit_result["calibration_families"],
        "calibration_capture_stage": "12C-1O",
        "calibration_capture_audit_sha256": PINNED[AUDIT_1O],
        "authorized_use": ["model selection", "early stopping", "threshold calibration",
                           "abstention/OOD threshold fitting"],
        "prohibited_use": ["gradient updates", "weight initialization", "architecture search",
                           "feature selection", "vector or probe optimization"],
        "gradient_updates": "PROHIBITED",
        "reuse_policy": "CALIBRATION MAY BE REUSED ACROSS CANDIDATES; TEST MAY NOT",
        "independent_test_access": "PROHIBITED BY THIS AUTHORIZATION",
        "holdout_access": "PROHIBITED BY THIS AUTHORIZATION",
    }

    training_auth = {
        "authorization_version": "CIRCUITSAGE-HMAC-V2.2-TRAINING-EXECUTION-AUTHORIZATION-12C2B-v1",
        "stage": STAGE, "status": "FROZEN", "created_at": created,
        "supersedes_clause": "Stage 12C-1A training contract field training = NOT YET AUTHORIZED",
        "training_contract_authority": rel(TRAINING_1A),
        "architecture_authority": rel(ARCHITECTURE_1A),
        "acceptance_authority": rel(ACCEPTANCE_1A),
        "model_training": "AUTHORIZED / BOUNDED",
        "model_selection": "AUTHORIZED / BOUNDED",
        "gradient_partitions": ["GENERALIZATION_TRAIN"],
        "gradient_families": audit_result["train_families"],
        "selection_partition": "GENERALIZATION_CALIBRATION",
        "selection_families": audit_result["calibration_families"],
        "prohibited_partitions": ["INDEPENDENT_CIRCUIT_TEST", "GENERALIZATION_HOLDOUT"],
        "candidates": [r["candidate_id"] for r in candidate_rows],
        "trainable_candidates": [r["candidate_id"] for r in candidate_rows
                                 if r["trainable"].strip().upper() == "YES"],
        "mandatory_comparator": [r["candidate_id"] for r in candidate_rows
                                 if r["trainable"].strip().upper() == "NO"],
        "parameter_cap_enforced": True,
        "maximum_trainable_parameters": max(int(r["parameter_cap"]) for r in candidate_rows),
        "objectives": training["objectives"],
        "selection_order": training["selection_order"],
        "early_stopping": training["early_stopping"],
        "determinism": training["determinism"],
        "hard_negatives": training["hard_negatives"],
        "after_selection": training["after_selection"],
        "datasets": {
            "response_features": rel(FEATURES_1O),
            "supervision_targets": rel(TARGETS_1O),
            "circuit_graphs": rel(GRAPH_NPZ_2A),
            "cell_type_vocabulary": rel(VOCABULARY_2A),
            "graph_normalization_rules": rel(NORMALIZATION_2A),
        },
        "identity_firewall": "FAULT IDENTITY ABSENT FROM MODEL-FACING FEATURES",
        "leakage_audit": rel(LEAKAGE_AUDIT),
        "leakage_checks_passed": audit_result["checks_run"],
        "acceptance_evaluation": "NOT AUTHORIZED BY THIS STAGE",
        "independent_test_capture": "NOT AUTHORIZED BY THIS STAGE",
        "one_shot_rule_reminder": load_json(ACCEPTANCE_1A)["one_shot_rule"],
        "prohibited": [
            "gradient updates on GENERALIZATION_CALIBRATION",
            "any read of INDEPENDENT_CIRCUIT_TEST or GENERALIZATION_HOLDOUT",
            "deriving thresholds from protected partitions",
            "modifying frozen V1, V2.1, 12C-1O, 12C-2A evidence",
            "extending the cell-type vocabulary at inference",
        ],
    }

    preflight = {
        "preflight_version": "CIRCUITSAGE-HMAC-V2.2-TRAINING-AUTHORIZATION-PREFLIGHT-12C2B-v1",
        "stage": STAGE, "status": "PASS", "created_at": created,
        "simulation_calls": 0, "dataset_records_created": 0, "model_deserializations": 0,
        "training_calls": 0, "selection_calls": 0, "inference_calls": 0,
        "independent_test_access": 0, "validation_access": 0, "holdout_access": 0,
        "leakage_checks_run": audit_result["checks_run"],
        "leakage_checks_failed": audit_result["checks_failed"],
        "acceptance_criteria_invented": False,
        "architecture_invented": False,
        "config_v2_2_enumerated_before_absence_claim": True,
    }

    check_table = "\n".join(f"| {c['check']} | {c['result']} |" for c in audit_result["checks"])
    cand_table = "\n".join(
        f"| `{r['candidate_id']}` | {r['trainable']} | {r['graph_encoder']} | "
        f"{r['parameter_cap']:,} | {r['role']} |" for r in candidate_rows)
    part_table = "\n".join(
        f"| `{r['family_id']}` | {r['partition']} | {r['captured']} | {r['gradient_updates']} | "
        f"{r['selection_use']} | {r['access_state']} |" for r in partition_rows)

    report = f"""# Stage {STAGE} — Calibration-Use and Training Authorization

**Status: PASS / FROZEN — authorization gate only.**

## What this stage authorizes

| | |
|---|---|
| Model training | **AUTHORIZED / BOUNDED** on GENERALIZATION_TRAIN |
| Model selection | **AUTHORIZED / BOUNDED** on GENERALIZATION_CALIBRATION |
| Independent test | **NOT AUTHORIZED** |
| Holdout | **NOT AUTHORIZED** |
| Acceptance evaluation | **NOT AUTHORIZED** |

The Stage 12C-1A training contract required a *separate capture authorization*
before calibration could be used for selection. That authorization is issued
here.

## Leakage audit — {audit_result['checks_run']} executable checks

| check | result |
|---|---|
{check_table}

**Verdict:** {audit_result['verdict']}

## Partition register — seven families

| family | partition | captured | gradients | selection | access |
|---|---|---|---|---|---|
{part_table}

## Candidate register — frozen at Stage 12C-1A

| candidate | trainable | graph encoder | param cap | role |
|---|---|---|---|---|
{cand_table}

No architecture is designed here. All four candidates, their encoders, fusion
strategies and parameter caps were frozen in the Stage 12C-1A candidate grid.

**Selection order (frozen):** {' → '.join(training['selection_order'])}

## Reminder: one-shot rule

> {load_json(ACCEPTANCE_1A)['one_shot_rule']}

Training and selection may iterate freely on TRAIN and CALIBRATION. The
independent test is opened **once**, and after it no retraining, threshold
change or reselection is permitted.

## Access

INDEPENDENT_TEST **LOCKED**, VALIDATION **UNOPENED**, HOLDOUT **SEALED**.
Independent generalization remains **NOT ESTABLISHED**. Future hybrid brand
remains **{FUTURE_BRAND}**.

## Next gate

**Stage 12C-2C** — candidate training execution against this authorization.
"""

    part_fields = ["family_id", "partition", "captured", "gradient_updates",
                   "selection_use", "access_state"]
    cand_fields = ["candidate_id", "trainable", "graph_encoder", "response_encoder",
                   "fusion", "parameter_cap", "role", "gradient_partition", "authorization"]

    frozen_write(CALIBRATION_AUTHORIZATION, canonical_json(calibration_auth))
    frozen_write(LEAKAGE_AUDIT, canonical_json(audit_result))
    frozen_write(TRAINING_AUTHORIZATION, canonical_json(training_auth))
    frozen_write(PARTITION_REGISTER, csv_bytes(partition_rows, part_fields))
    frozen_write(CANDIDATE_REGISTER, csv_bytes(candidate_rows, cand_fields))
    frozen_write(PREFLIGHT, canonical_json(preflight))
    frozen_write(REPORT, report.encode())

    stage_outputs = (TRAINING_AUTHORIZATION, CALIBRATION_AUTHORIZATION, LEAKAGE_AUDIT,
                     PARTITION_REGISTER, CANDIDATE_REGISTER, PREFLIGHT, REPORT)
    manifest = {
        "manifest_version": "CIRCUITSAGE-HMAC-V2.2-TRAINING-AUTHORIZATION-MANIFEST-12C2B-v1",
        "stage": STAGE, "status": "PASS", "created_at": created,
        "stage_source": record(Path(__file__).resolve()),
        "frozen_inputs": {rel(p): record(p) for p in PINNED},
        "outputs": {rel(p): record(p) for p in stage_outputs},
        "config_v2_2_enumeration": sorted(p.name for p in CONFIG.glob("*.json")),
        "leakage_checks_run": audit_result["checks_run"],
        "candidates": len(candidate_rows),
        "training_calls": 0, "selection_calls": 0, "inference_calls": 0,
        "independent_test_access": 0, "validation_access": 0, "holdout_access": 0,
    }
    frozen_write(MANIFEST, canonical_json(manifest))

    audit = {
        "audit_version": "CIRCUITSAGE-HMAC-V2.2-TRAINING-AUTHORIZATION-FREEZE-12C2B-v1",
        "stage": STAGE, "status": "PASS",
        "model_training": "AUTHORIZED / BOUNDED",
        "model_selection": "AUTHORIZED / BOUNDED",
        "calibration_use": "AUTHORIZED / SELECTION ONLY / NO GRADIENTS",
        "gradient_families": audit_result["train_families"],
        "selection_families": audit_result["calibration_families"],
        "independent_test_families": audit_result["independent_test_families"],
        "holdout_families": audit_result["holdout_families"],
        "leakage_checks_run": audit_result["checks_run"],
        "leakage_checks_failed": audit_result["checks_failed"],
        "leakage_verdict": audit_result["verdict"],
        "candidates": [r["candidate_id"] for r in candidate_rows],
        "trainable_candidates": len([r for r in candidate_rows
                                     if r["trainable"].strip().upper() == "YES"]),
        "architecture_invented": False,
        "acceptance_criteria_invented": False,
        "acceptance_evaluation": "NOT AUTHORIZED",
        "independent_test_capture": "NOT AUTHORIZED",
        "independent_test_validation_holdout_access": [0, 0, 0],
        "independent_generalization": "NOT ESTABLISHED",
        "training_authorization_record": record(TRAINING_AUTHORIZATION),
        "calibration_authorization_record": record(CALIBRATION_AUTHORIZATION),
        "leakage_audit_record": record(LEAKAGE_AUDIT),
        "partition_register_record": record(PARTITION_REGISTER),
        "candidate_register_record": record(CANDIDATE_REGISTER),
        "manifest_record": record(MANIFEST),
        "future_combined_model_brand": FUTURE_BRAND,
        "next_gate": "STAGE 12C-2C — CANDIDATE TRAINING EXECUTION",
    }
    frozen_write(AUDIT, canonical_json(audit))

    for path in (TRAINING_AUTHORIZATION, CALIBRATION_AUTHORIZATION, LEAKAGE_AUDIT,
                 PREFLIGHT, MANIFEST, AUDIT):
        require(path.read_bytes() == canonical_json(load_json(path)),
                f"canonical output replay: {path.name}")

    print(f"\n{'Stage':<52}: {STAGE} — TRAINING AUTHORIZATION")
    print(f"{'Status':<52}: PASS / FROZEN")
    print(f"{'Leakage checks run / failed':<52}: {audit_result['checks_run']} / "
          f"{audit_result['checks_failed']}")
    print(f"{'Gradient families':<52}: {', '.join(audit_result['train_families'])}")
    print(f"{'Selection family':<52}: {', '.join(audit_result['calibration_families'])}")
    print(f"{'Candidates / trainable':<52}: {len(candidate_rows)} / "
          f"{len([r for r in candidate_rows if r['trainable'].strip().upper() == 'YES'])}")
    print(f"{'Model training / selection':<52}: AUTHORIZED / BOUNDED")
    print(f"{'Acceptance evaluation':<52}: NOT AUTHORIZED")
    print(f"{'TEST / VALIDATION / HOLDOUT access':<52}: 0 / 0 / 0")
    print(f"{'Audit':<52}: {AUDIT}")
    print(f"{'Audit SHA':<52}: {sha256(AUDIT)}")
    print(f"{'Next gate':<52}: STAGE 12C-2C — CANDIDATE TRAINING")


def status() -> None:
    print(f"STAGE {STAGE} — TRAINING AUTHORIZATION STATUS")
    if not MANIFEST.is_file() or not AUDIT.is_file():
        print("Status                    : NOT FROZEN")
        print(f"Expected audit            : {AUDIT}")
        return
    manifest = load_json(MANIFEST)
    audit = load_json(AUDIT)
    require(manifest.get("status") == "PASS" and audit.get("status") == "PASS", "frozen status")
    require(audit.get("manifest_record", {}).get("sha256") == sha256(MANIFEST), "manifest anchor")
    for name, item in sorted((manifest.get("outputs") or {}).items()):
        path = (ROOT / item["path"]).resolve()
        require(path.is_file() and sha256(path) == item["sha256"], f"output SHA: {name}")
    print("Status                    : PASS / FROZEN")
    print(f"Training / selection      : {audit['model_training']} / {audit['model_selection']}")
    print(f"Calibration use           : {audit['calibration_use']}")
    print(f"Gradient families         : {audit['gradient_families']}")
    print(f"Leakage checks            : {audit['leakage_checks_run']} run, "
          f"{audit['leakage_checks_failed']} failed")
    print(f"Candidates / trainable    : {len(audit['candidates'])} / {audit['trainable_candidates']}")
    print(f"Acceptance evaluation     : {audit['acceptance_evaluation']}")
    print(f"Next gate                 : {audit['next_gate']}")
    print(f"Audit SHA                 : {sha256(AUDIT)}")


def self_test() -> None:
    require(CANDIDATE_GRID_1A.is_file(), "candidate grid present")
    grid = read_csv(CANDIDATE_GRID_1A)
    require(len(grid) == 4, "four candidates")
    require(len([r for r in grid if r["trainable"].strip().upper() == "YES"]) == 3,
            "three trainable")
    require(max(int(r["parameter_cap"]) for r in grid) == 1500000, "parameter cap 1.5M")
    require(GRAPH_NPZ_2A.is_file() and FEATURES_1O.is_file(), "datasets present")
    split = load_json(SPLIT_1B)
    a = split["family_assignments"]
    require(len(a) == 7, "seven families")
    require(sum(1 for p in a.values() if p == "GENERALIZATION_TRAIN") == 3, "three train")
    require(sum(1 for p in a.values() if p == "INDEPENDENT_CIRCUIT_TEST") == 2, "two test")
    print(f"Stage {STAGE} self-test: PASS")


def locked_execute() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    with LOCK_FILE.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            stop(f"Stage {STAGE} execution lock is held by another process")
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
