#!/usr/bin/env python3
"""Stage 12C-1Q: disposition correction — V2.2 acceptance contract exists.

Stage 12C-1P froze a REPORT-ONLY disposition of the V2.2 multi-circuit campaign
and recorded the justification:

    disposition_mode = "REPORT-ONLY / NO FROZEN ADVANCEMENT TARGET EXISTS"
    advancement_decision = "NOT EVALUATED / NO FROZEN CRITERIA EXIST"
    advancement_target_defined = false

That justification is FACTUALLY INCORRECT.  Stage 12C-1A froze
``circuitsage_hmac_v2_2_acceptance_contract_12c1a.json``, which defines ten
macro gates, a four-metric per-circuit floor, a mandatory two-circuit test
population, a one-shot evaluation rule and a failure disposition.  The
contract was present and frozen before Stage 12C-1P executed; it was not
located during that stage's input survey.

This stage records the correction.  It does NOT modify Stage 12C-1P: that
artifact is frozen evidence and its measured values are independently
verified here as correct.  Only the stated justification was wrong.

Corrected position:

  * V2.2 acceptance criteria EXIST and are frozen at Stage 12C-1A.
  * The criteria are scoped to INDEPENDENT_CIRCUIT_TEST circuits, which have
    not been captured, so no acceptance evaluation was possible at 12C-1P.
  * Stage 12C-1P's refusal to declare PASS / NOT_MET was therefore the correct
    ACTION, reached through an incorrect REASON.
  * All numeric values reported by Stage 12C-1P are re-verified here as
    accurate against the frozen Stage 12C-1O metrics.

This stage also surfaces the full seven-family corpus.  The Stage 12C-1O
campaign covered four families (three TRAIN, one CALIBRATION).  Two
INDEPENDENT_CIRCUIT_TEST families (``ibex_cpu``, ``secworks_chacha``) remain
sealed and one GENERALIZATION_HOLDOUT family (``serv_cpu``) remains blocked.
Acceptance is evaluated on the locked test families only.

No simulation, fault injection, dataset construction, model loading, training,
selection or inference occurs in this stage.  No protected partition is
accessed.  Independent TEST remains LOCKED, VALIDATION UNOPENED, HOLDOUT
SEALED.
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


STAGE = "12C-1Q"
ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "config/v2_2"
RESULT = ROOT / "results/circuitsage_hmac_v2_12c1"
WORK = RESULT / "disposition_correction_12c1q"
LOCK_FILE = WORK / ".stage_12c1q.lock"

# ---------------------------------------------------------------- frozen input
SOURCE_1P = ROOT / "stage_12c1p_campaign_disposition.py"
ACCEPTANCE_1A = CONFIG / "circuitsage_hmac_v2_2_acceptance_contract_12c1a.json"
PARTITION_1A = CONFIG / "circuitsage_hmac_v2_2_circuit_family_partition_contract_12c1a.json"
SPLIT_1B = CONFIG / "circuitsage_hmac_v2_2_family_split_authorization_12c1b.json"
AUDIT_1A = RESULT / "circuitsage_hmac_v2_2_generalization_contract_freeze_12c1a.json"
AUDIT_1B = RESULT / "circuitsage_hmac_v2_2_corpus_license_split_authorization_freeze_12c1b.json"
AUDIT_1P = RESULT / "circuitsage_hmac_v2_2_campaign_disposition_freeze_12c1p.json"
MANIFEST_1P = RESULT / "circuitsage_hmac_v2_2_campaign_disposition_manifest_12c1p.json"

WORK_1O = RESULT / "parallel_campaign_execution_12c1o"
METRICS_1O = WORK_1O / "circuitsage_hmac_v2_2_amended_campaign_metrics_12c1o.json"
AUDIT_1O = RESULT / "circuitsage_hmac_v2_2_amended_campaign_execution_dataset_freeze_12c1o.json"

PINNED = {
    SOURCE_1P: "f513f3bc0b0146c7d86aa8d8d59256d5af656b78cc6ac266ec05aeb8641ee35c",
    ACCEPTANCE_1A: "9c8eec4d85957c4408ac59e0c8760af90c91d0667c995d91ac969b5a8f205f26",
    PARTITION_1A: "4cbab060ef824f0ff8a17af7e2250d0efcf608ac7121c685bcf9eef4f5b2804c",
    SPLIT_1B: "4103808fc389c78088e31c0a76549324c386ed9d8f7bedb7cc4d3748f9ea3303",
    AUDIT_1A: "4e16a3d498e237c312fdef5565300b8ffc37dde75caa6def92c1651514006ae3",
    AUDIT_1B: "08d9b0b0835f91948188c40daed46bc0237bed18fedcc25a02035b5c16b4f359",
    AUDIT_1P: "e0040f8f316e71a24cca232ed18e602d7d2454dd7761db3fd7cf02f1e3aa5548",
    MANIFEST_1P: "4d215b26ac1b02d0169c7abbfd2d27b337074d04e2427c18d04b8d81202b5522",
    AUDIT_1O: "fa67278ae07bc6dd57246088c3da41918f88c05fa06580c4583c022c1993430b",
}

# ---------------------------------------------------------------- stage output
CORRECTION_RECORD = WORK / "circuitsage_hmac_v2_2_disposition_correction_12c1q.json"
ACCEPTANCE_SCOPE = CONFIG / "circuitsage_hmac_v2_2_acceptance_scope_clarification_12c1q.json"
CORPUS_STATUS = WORK / "circuitsage_hmac_v2_2_corpus_partition_status_12c1q.csv"
GATE_REGISTER = WORK / "circuitsage_hmac_v2_2_acceptance_gate_register_12c1q.csv"
REVERIFICATION = WORK / "circuitsage_hmac_v2_2_12c1p_value_reverification_12c1q.json"
PREFLIGHT = WORK / "circuitsage_hmac_v2_2_correction_preflight_12c1q.json"
REPORT = WORK / "circuitsage_hmac_v2_2_disposition_correction_report_12c1q.md"
MANIFEST = RESULT / "circuitsage_hmac_v2_2_disposition_correction_manifest_12c1q.json"
AUDIT = RESULT / "circuitsage_hmac_v2_2_disposition_correction_freeze_12c1q.json"

CAMPAIGN_FAMILIES = (
    "opentitan_hmac_sha256",
    "picorv32_cpu",
    "secworks_aes",
    "secworks_sha256",
)
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
        require(sha256(path) == expected,
                f"frozen input SHA changed: {path.name}")
        print(f"  {path.name:<86}: OK", flush=True)

    acceptance = load_json(ACCEPTANCE_1A)
    require(acceptance.get("status") == "FROZEN", "12C-1A acceptance contract frozen")
    require(isinstance(acceptance.get("macro_gates"), dict) and acceptance["macro_gates"],
            "macro gates present")
    require(isinstance(acceptance.get("per_circuit_floor"), dict) and acceptance["per_circuit_floor"],
            "per-circuit floor present")

    audit_1p = load_json(AUDIT_1P)
    require(audit_1p.get("status") == "PASS", "12C-1P freeze status")
    # The precise incorrect assertions this stage corrects.
    require(audit_1p.get("advancement_target_defined") is False,
            "12C-1P recorded advancement_target_defined=false (the claim under correction)")
    require("NO FROZEN CRITERIA EXIST" in str(audit_1p.get("advancement_decision", "")),
            "12C-1P recorded 'NO FROZEN CRITERIA EXIST' (the claim under correction)")
    print("  Stage 12C-1A acceptance contract located and frozen"
          f"{'':<34}: CONFIRMED", flush=True)
    print("  Stage 12C-1P incorrect justification located"
          f"{'':<41}: CONFIRMED", flush=True)
    return acceptance


def reverify_12c1p_values() -> dict[str, Any]:
    """Confirm every numeric value 12C-1P reported is accurate."""
    metrics = load_json(METRICS_1O)
    audit_1p = load_json(AUDIT_1P)
    families = {f["family_id"]: f for f in metrics["families"]}

    checks: list[dict[str, Any]] = []

    def check(name: str, reported: Any, actual: Any) -> None:
        ok = (abs(float(reported) - float(actual)) < 1e-9
              if isinstance(reported, (int, float)) else reported == actual)
        checks.append({"quantity": name, "reported_12c1p": reported,
                       "actual_12c1o": actual, "agrees": bool(ok)})

    check("aggregate_detection_recall",
          audit_1p["aggregate_detection_recall"], metrics["all_injected_detection_recall"])
    check("aggregate_exact_site_rate",
          audit_1p["aggregate_exact_site_rate"], metrics["all_injected_exact_site_rate"])
    detections = [float(families[f]["all_injected_detection_recall"]) for f in CAMPAIGN_FAMILIES]
    check("per_family_detection_minimum", audit_1p["per_family_detection_minimum"], min(detections))
    check("per_family_detection_maximum", audit_1p["per_family_detection_maximum"], max(detections))
    check("fault_free_false_alarms",
          audit_1p["fault_free_false_alarms"], metrics["fault_free_false_alarms"])

    disagreements = [c for c in checks if not c["agrees"]]
    return {
        "reverification_version": "CIRCUITSAGE-HMAC-V2.2-12C1P-VALUE-REVERIFICATION-12C1Q-v1",
        "stage": STAGE, "status": "PASS" if not disagreements else "FAIL",
        "created_at": now(),
        "checks": checks,
        "disagreements": len(disagreements),
        "verdict": ("ALL STAGE 12C-1P NUMERIC VALUES ARE ACCURATE; ONLY THE STATED "
                    "JUSTIFICATION WAS INCORRECT"
                    if not disagreements else "NUMERIC DISAGREEMENT DETECTED"),
    }


def build_corpus_status() -> list[dict[str, Any]]:
    split = load_json(SPLIT_1B)
    assignments = split["family_assignments"]
    captured = set(CAMPAIGN_FAMILIES)
    rows: list[dict[str, Any]] = []
    for family_id in sorted(assignments):
        partition = assignments[family_id]
        if family_id in captured:
            state = "CAPTURED / FROZEN (Stage 12C-1O)"
            access = "OPEN FOR TRAINING AND CALIBRATION"
        elif partition == "INDEPENDENT_CIRCUIT_TEST":
            state = "NOT CAPTURED"
            access = "SEALED UNTIL ONE COMMITTED FINAL EVALUATION"
        elif partition == "GENERALIZATION_HOLDOUT":
            state = "NOT CAPTURED"
            access = "BLOCKED / EXCLUDED FROM V2.2 ENTIRELY"
        else:
            state = "NOT CAPTURED"
            access = "SEE PARTITION CONTRACT"
        rows.append({
            "family_id": family_id,
            "partition": partition,
            "campaign_state": state,
            "access_policy": access,
            "in_12c1o_campaign": "YES" if family_id in captured else "NO",
            "acceptance_evaluated_here": "NO",
        })
    return rows


def build_gate_register(acceptance: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, value in sorted(acceptance["macro_gates"].items()):
        rows.append({
            "gate_class": "MACRO", "gate_name": name, "threshold": value,
            "evaluated_on": "INDEPENDENT_CIRCUIT_TEST (both circuits)",
            "status": "NOT EVALUABLE — TEST NOT CAPTURED",
        })
    for name, value in sorted(acceptance["per_circuit_floor"].items()):
        rows.append({
            "gate_class": "PER_CIRCUIT_FLOOR", "gate_name": name, "threshold": value,
            "evaluated_on": "EACH INDEPENDENT_CIRCUIT_TEST CIRCUIT INDIVIDUALLY",
            "status": "NOT EVALUABLE — TEST NOT CAPTURED",
        })
    return rows


def execute() -> None:
    require(ROOT.name == "vlsi_fault_detection_v2", f"place script in project root, not {ROOT}")
    outputs = (CORRECTION_RECORD, ACCEPTANCE_SCOPE, CORPUS_STATUS, GATE_REGISTER,
               REVERIFICATION, PREFLIGHT, REPORT, MANIFEST, AUDIT)
    require(not any(p.exists() for p in outputs),
            f"frozen Stage {STAGE} output already exists; use --status")

    acceptance = verify_inputs()

    print("\nRE-VERIFYING STAGE 12C-1P NUMERIC VALUES", flush=True)
    reverification = reverify_12c1p_values()
    for c in reverification["checks"]:
        mark = "OK " if c["agrees"] else "XX "
        print(f"  {mark}{c['quantity']:<40}: {c['reported_12c1p']}", flush=True)
    require(reverification["status"] == "PASS",
            "Stage 12C-1P numeric values disagree with Stage 12C-1O metrics")

    print("\nCORPUS PARTITION STATUS", flush=True)
    corpus = build_corpus_status()
    for r in corpus:
        print(f"  {r['family_id']:<24} {r['partition']:<28} {r['campaign_state']}", flush=True)

    gates = build_gate_register(acceptance)
    created = now()

    correction = {
        "correction_version": "CIRCUITSAGE-HMAC-V2.2-DISPOSITION-CORRECTION-12C1Q-v1",
        "stage": STAGE, "status": "FROZEN", "created_at": created,
        "corrects_stage": "12C-1P",
        "corrects_audit_sha256": PINNED[AUDIT_1P],
        "corrected_artifact_modified": False,
        "correction_class": "STATED JUSTIFICATION INCORRECT; MEASURED VALUES CORRECT",
        "incorrect_assertions": [
            {"field": "disposition_mode",
             "recorded": "REPORT-ONLY / NO FROZEN ADVANCEMENT TARGET EXISTS",
             "correct": "REPORT-ONLY / ACCEPTANCE CRITERIA EXIST BUT ARE SCOPED TO "
                        "INDEPENDENT_CIRCUIT_TEST, WHICH IS NOT CAPTURED"},
            {"field": "advancement_decision",
             "recorded": "NOT EVALUATED / NO FROZEN CRITERIA EXIST",
             "correct": "NOT EVALUATED / CRITERIA EXIST BUT TEST POPULATION NOT CAPTURED"},
            {"field": "advancement_target_defined",
             "recorded": False,
             "correct": True},
        ],
        "authoritative_acceptance_contract": rel(ACCEPTANCE_1A),
        "authoritative_acceptance_contract_sha256": PINNED[ACCEPTANCE_1A],
        "acceptance_contract_stage": "12C-1A",
        "macro_gate_count": len(acceptance["macro_gates"]),
        "per_circuit_floor_count": len(acceptance["per_circuit_floor"]),
        "action_assessment": (
            "Stage 12C-1P declined to declare PASS or NOT_MET. That ACTION was correct: the "
            "acceptance gates are scoped to INDEPENDENT_CIRCUIT_TEST circuits that have not been "
            "captured. The REASON recorded for that action was wrong."
        ),
        "numeric_values_assessment": "ALL ACCURATE / RE-VERIFIED IN THIS STAGE",
        "root_cause": (
            "The Stage 12C-1P input survey inspected the campaign and disposition lineage but did "
            "not enumerate config/v2_2 contract artifacts frozen at Stage 12C-1A."
        ),
        "preventive_rule": (
            "Any stage that asserts the absence of a contract, criterion or artifact must first "
            "enumerate config/v2_2 and the full stage audit chain, and must cite the enumeration "
            "in its own manifest."
        ),
    }

    scope = {
        "clarification_version": "CIRCUITSAGE-HMAC-V2.2-ACCEPTANCE-SCOPE-12C1Q-v1",
        "stage": STAGE, "status": "FROZEN", "created_at": created,
        "acceptance_criteria_exist": True,
        "acceptance_contract": rel(ACCEPTANCE_1A),
        "acceptance_contract_sha256": PINNED[ACCEPTANCE_1A],
        "scientific_claim": acceptance["scientific_claim"],
        "evaluation_population": acceptance["mandatory_test_population"],
        "one_shot_rule": acceptance["one_shot_rule"],
        "prediction_commitment": acceptance["prediction_commitment"],
        "failure_disposition": acceptance["failure_disposition"],
        "holdout_rule": acceptance["holdout"],
        "macro_gates": acceptance["macro_gates"],
        "per_circuit_floor": acceptance["per_circuit_floor"],
        "required_comparators": acceptance["required_comparators"],
        "uncertainty_method": acceptance["uncertainty"],
        "gates_applicable_to_train_calibration": False,
        "gates_applicable_to_12c1o_campaign": False,
        "rationale": (
            "TRAIN and CALIBRATION metrics measure dataset and retrieval properties, not "
            "independent generalization. Applying acceptance gates to them would constitute "
            "evaluating on development data."
        ),
        "independent_test_families": [r["family_id"] for r in corpus
                                      if r["partition"] == "INDEPENDENT_CIRCUIT_TEST"],
        "holdout_families": [r["family_id"] for r in corpus
                             if r["partition"] == "GENERALIZATION_HOLDOUT"],
        "independent_test_access": "LOCKED / NOT CAPTURED",
        "holdout_access": "SEALED / BLOCKED FOR V2.2",
    }

    preflight = {
        "preflight_version": "CIRCUITSAGE-HMAC-V2.2-CORRECTION-PREFLIGHT-12C1Q-v1",
        "stage": STAGE, "status": "PASS", "created_at": created,
        "simulation_calls": 0, "fault_injection_calls": 0, "dataset_records_created": 0,
        "model_deserializations": 0, "training_calls": 0, "selection_calls": 0,
        "inference_calls": 0,
        "independent_test_access": 0, "validation_access": 0, "holdout_access": 0,
        "frozen_12c1p_outputs_modified": False,
        "frozen_12c1o_outputs_modified": False,
        "frozen_12c1a_contracts_modified": False,
        "acceptance_criteria_invented": False,
    }

    macro_rows = "\n".join(
        f"| {r['gate_name']} | {r['threshold']} |"
        for r in gates if r["gate_class"] == "MACRO")
    floor_rows = "\n".join(
        f"| {r['gate_name']} | {r['threshold']} |"
        for r in gates if r["gate_class"] == "PER_CIRCUIT_FLOOR")
    corpus_rows = "\n".join(
        f"| `{r['family_id']}` | {r['partition']} | {r['campaign_state']} | {r['access_policy']} |"
        for r in corpus)

    report = f"""# Stage {STAGE} — Disposition Correction

**Status: PASS / FROZEN — correction record, no re-evaluation.**

## What is being corrected

Stage 12C-1P recorded that no frozen V2.2 advancement target exists:

```
disposition_mode           : REPORT-ONLY / NO FROZEN ADVANCEMENT TARGET EXISTS
advancement_decision       : NOT EVALUATED / NO FROZEN CRITERIA EXIST
advancement_target_defined : false
```

**This is incorrect.** Stage 12C-1A froze
`config/v2_2/circuitsage_hmac_v2_2_acceptance_contract_12c1a.json`
(SHA `{PINNED[ACCEPTANCE_1A]}`)
containing **{len(acceptance['macro_gates'])} macro gates** and a
**{len(acceptance['per_circuit_floor'])}-metric per-circuit floor**. It was
frozen before Stage 12C-1P executed.

## What remains correct

Stage 12C-1P declined to declare PASS or NOT_MET. **That action was correct.**
The acceptance gates are scoped to `INDEPENDENT_CIRCUIT_TEST` circuits, which
have not been captured. Applying them to TRAIN/CALIBRATION data would be
evaluating on development data.

All numeric values reported by Stage 12C-1P were re-verified against the frozen
Stage 12C-1O metrics in this stage: **{reverification['disagreements']} disagreements**.

Stage 12C-1P is **not modified**. Its measured evidence stands.

## The frozen acceptance contract

**Claim:** {acceptance['scientific_claim']}

**Macro gates** (evaluated on both locked test circuits):

| gate | threshold |
|---|---|
{macro_rows}

**Per-circuit floor** (each test circuit individually):

| gate | threshold |
|---|---|
{floor_rows}

**One-shot rule:** {acceptance['one_shot_rule']}

**Failure disposition:** {acceptance['failure_disposition']}

## Full corpus — seven families

| family | partition | campaign state | access |
|---|---|---|---|
{corpus_rows}

The Stage 12C-1O campaign covered **four** of seven families. Acceptance is
decided on the two sealed `INDEPENDENT_CIRCUIT_TEST` families, and the
`GENERALIZATION_HOLDOUT` family remains blocked even if V2.2 passes.

## Root cause and preventive rule

**Root cause:** the Stage 12C-1P input survey inspected the campaign and
disposition lineage but did not enumerate `config/v2_2` contract artifacts
frozen at Stage 12C-1A.

**Preventive rule (frozen here):** any stage asserting the *absence* of a
contract, criterion or artifact must first enumerate `config/v2_2` and the full
stage audit chain, and must cite that enumeration in its own manifest.

## Access

No protected partition was accessed. Independent TEST remains **LOCKED**,
VALIDATION **UNOPENED**, HOLDOUT **SEALED**. Independent generalization remains
**NOT ESTABLISHED**. The future hybrid brand remains **{FUTURE_BRAND}**.

## Next gate

**Stage 12C-2A** — V2.2 model contract and training authorization, written
against the corrected understanding: the target is the Stage 12C-1A acceptance
contract, evaluated once on `ibex_cpu` and `secworks_chacha`.
"""

    corpus_fields = ["family_id", "partition", "campaign_state", "access_policy",
                     "in_12c1o_campaign", "acceptance_evaluated_here"]
    gate_fields = ["gate_class", "gate_name", "threshold", "evaluated_on", "status"]

    frozen_write(CORRECTION_RECORD, canonical_json(correction))
    frozen_write(ACCEPTANCE_SCOPE, canonical_json(scope))
    frozen_write(CORPUS_STATUS, csv_bytes(corpus, corpus_fields))
    frozen_write(GATE_REGISTER, csv_bytes(gates, gate_fields))
    frozen_write(REVERIFICATION, canonical_json(reverification))
    frozen_write(PREFLIGHT, canonical_json(preflight))
    frozen_write(REPORT, report.encode())

    stage_outputs = (CORRECTION_RECORD, ACCEPTANCE_SCOPE, CORPUS_STATUS, GATE_REGISTER,
                     REVERIFICATION, PREFLIGHT, REPORT)
    manifest = {
        "manifest_version": "CIRCUITSAGE-HMAC-V2.2-DISPOSITION-CORRECTION-MANIFEST-12C1Q-v1",
        "stage": STAGE, "status": "PASS", "created_at": created,
        "stage_source": record(Path(__file__).resolve()),
        "frozen_inputs": {rel(p): record(p) for p in PINNED},
        "outputs": {rel(p): record(p) for p in stage_outputs},
        "config_v2_2_enumeration": sorted(p.name for p in CONFIG.glob("*.json")),
        "corrects_stage": "12C-1P",
        "corrected_artifact_modified": False,
        "simulation_calls": 0, "training_calls": 0, "inference_calls": 0,
        "independent_test_access": 0, "validation_access": 0, "holdout_access": 0,
    }
    frozen_write(MANIFEST, canonical_json(manifest))

    audit = {
        "audit_version": "CIRCUITSAGE-HMAC-V2.2-DISPOSITION-CORRECTION-FREEZE-12C1Q-v1",
        "stage": STAGE, "status": "PASS",
        "correction_class": "STATED JUSTIFICATION INCORRECT; MEASURED VALUES CORRECT",
        "corrects_stage": "12C-1P",
        "corrects_audit_sha256": PINNED[AUDIT_1P],
        "corrected_artifact_modified": False,
        "acceptance_criteria_exist": True,
        "acceptance_contract_stage": "12C-1A",
        "acceptance_contract_sha256": PINNED[ACCEPTANCE_1A],
        "macro_gate_count": len(acceptance["macro_gates"]),
        "per_circuit_floor_count": len(acceptance["per_circuit_floor"]),
        "gates_applicable_to_12c1o_campaign": False,
        "12c1p_action_correct": True,
        "12c1p_reason_correct": False,
        "12c1p_numeric_disagreements": reverification["disagreements"],
        "corpus_family_count": len(corpus),
        "campaign_family_count": len(CAMPAIGN_FAMILIES),
        "independent_test_families": scope["independent_test_families"],
        "holdout_families": scope["holdout_families"],
        "independent_test_validation_holdout_access": [0, 0, 0],
        "independent_generalization": "NOT ESTABLISHED",
        "frozen_12c1p_outputs_modified": False,
        "frozen_12c1a_contracts_modified": False,
        "correction_record": record(CORRECTION_RECORD),
        "acceptance_scope_record": record(ACCEPTANCE_SCOPE),
        "corpus_status_record": record(CORPUS_STATUS),
        "gate_register_record": record(GATE_REGISTER),
        "reverification_record": record(REVERIFICATION),
        "manifest_record": record(MANIFEST),
        "future_combined_model_brand": FUTURE_BRAND,
        "next_gate": "STAGE 12C-2A — V2.2 MODEL CONTRACT AND TRAINING AUTHORIZATION",
    }
    frozen_write(AUDIT, canonical_json(audit))

    for path in (CORRECTION_RECORD, ACCEPTANCE_SCOPE, REVERIFICATION, PREFLIGHT,
                 MANIFEST, AUDIT):
        require(path.read_bytes() == canonical_json(load_json(path)),
                f"canonical output replay: {path.name}")

    print(f"\n{'Stage':<52}: {STAGE} — DISPOSITION CORRECTION")
    print(f"{'Status':<52}: PASS / FROZEN")
    print(f"{'Corrects':<52}: 12C-1P (not modified)")
    print(f"{'Acceptance criteria exist':<52}: YES — Stage 12C-1A")
    print(f"{'Macro gates / per-circuit floors':<52}: "
          f"{len(acceptance['macro_gates'])} / {len(acceptance['per_circuit_floor'])}")
    print(f"{'12C-1P action / reason':<52}: CORRECT / INCORRECT")
    print(f"{'12C-1P numeric disagreements':<52}: {reverification['disagreements']}")
    print(f"{'Corpus families / captured':<52}: {len(corpus)} / {len(CAMPAIGN_FAMILIES)}")
    print(f"{'Independent test (sealed)':<52}: {', '.join(scope['independent_test_families'])}")
    print(f"{'Holdout (blocked)':<52}: {', '.join(scope['holdout_families'])}")
    print(f"{'TEST / VALIDATION / HOLDOUT access':<52}: 0 / 0 / 0")
    print(f"{'Audit':<52}: {AUDIT}")
    print(f"{'Audit SHA':<52}: {sha256(AUDIT)}")
    print(f"{'Next gate':<52}: STAGE 12C-2A — MODEL CONTRACT")


def status() -> None:
    print(f"STAGE {STAGE} — DISPOSITION CORRECTION STATUS")
    if not MANIFEST.is_file() or not AUDIT.is_file():
        print("Status                    : NOT FROZEN")
        print(f"Expected audit            : {AUDIT}")
        return
    manifest = load_json(MANIFEST)
    audit = load_json(AUDIT)
    require(manifest.get("status") == "PASS" and audit.get("status") == "PASS", "frozen status")
    require(audit.get("manifest_record", {}).get("sha256") == sha256(MANIFEST), "manifest anchor")
    print("Status                    : PASS / FROZEN")
    print(f"Corrects                  : {audit['corrects_stage']} (modified: {audit['corrected_artifact_modified']})")
    print(f"Acceptance criteria exist : {audit['acceptance_criteria_exist']}")
    print(f"Macro / floor gates       : {audit['macro_gate_count']} / {audit['per_circuit_floor_count']}")
    print(f"12C-1P action / reason    : {audit['12c1p_action_correct']} / {audit['12c1p_reason_correct']}")
    print(f"Numeric disagreements     : {audit['12c1p_numeric_disagreements']}")
    print(f"Independent test families : {audit['independent_test_families']}")
    print(f"Holdout families          : {audit['holdout_families']}")
    print(f"Next gate                 : {audit['next_gate']}")
    print(f"Audit SHA                 : {sha256(AUDIT)}")


def self_test() -> None:
    require(ACCEPTANCE_1A.is_file(), "12C-1A acceptance contract present")
    acceptance = load_json(ACCEPTANCE_1A)
    require(len(acceptance["macro_gates"]) == 10, "ten macro gates")
    require(len(acceptance["per_circuit_floor"]) == 4, "four per-circuit floors")
    split = load_json(SPLIT_1B)
    assignments = split["family_assignments"]
    require(len(assignments) == 7, "seven circuit families")
    test = [f for f, p in assignments.items() if p == "INDEPENDENT_CIRCUIT_TEST"]
    require(sorted(test) == ["ibex_cpu", "secworks_chacha"], "two sealed test families")
    holdout = [f for f, p in assignments.items() if p == "GENERALIZATION_HOLDOUT"]
    require(holdout == ["serv_cpu"], "one holdout family")
    gates = build_gate_register(acceptance)
    require(len(gates) == 14, "fourteen registered gates")
    require(all(g["status"].startswith("NOT EVALUABLE") for g in gates),
            "no gate evaluated in this stage")
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
