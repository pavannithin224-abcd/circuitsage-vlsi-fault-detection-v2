#!/usr/bin/env python3
"""Stage 12C-3A: independent test capture authorization.

Stage 12C-1D authorized elaboration and generic synthesis for TRAIN and
CALIBRATION families ONLY, and explicitly recorded
``locked_test_families: [ibex_cpu, secworks_chacha]``.  No frozen artifact in
this project authorizes opening those trees.  This stage creates that
authorization explicitly, so that the act of unsealing is itself a recorded,
reviewable decision rather than a side effect of running a capture script.

Authorized by this stage
------------------------
  * elaboration, portable-adapter construction, lint and generic synthesis of
    the two INDEPENDENT_CIRCUIT_TEST families
  * site enumeration and fault-catalogue derivation for those two families
  * fault-injection instrumentation and response capture for those two families
  * nothing else

Explicitly NOT authorized by this stage
---------------------------------------
  * any access to GENERALIZATION_HOLDOUT (``serv_cpu``) - it remains sealed by
    the user's decision, recorded here, so that one untouched circuit survives
    the one-shot evaluation
  * model training, retraining, threshold change, candidate reselection
  * acceptance evaluation - that requires its own stage after capture
  * any modification of a frozen artifact

Why the holdout stays sealed
----------------------------
The frozen acceptance contract requires "at least two circuit families absent
from training, calibration and architecture tuning".  ``ibex_cpu`` and
``secworks_chacha`` satisfy that requirement by themselves.  ``serv_cpu`` is a
third, separate seal that the contract never asks for.  Once a circuit is
captured it cannot be re-sealed, so the holdout is retained as a genuine reserve
for any post-hoc question a reviewer may raise.

Toolchain
---------
Stage 12C-1D pinned the exact executables and versions used to produce every
upstream artifact.  This stage verifies those exact binaries at those exact
paths, because a capture produced by a different toolchain would not be
comparable to the frozen development evidence.

One-shot warning
----------------
Capture is irreversible.  After Stage 12C-3B executes, the frozen acceptance
contract permits ONE locked evaluation with no retraining, threshold change or
reselection afterward.  Stage 12C-2H already froze per-circuit predictions,
including a predicted overall outcome of NOT MET, so the result cannot be
retrofitted either way.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import platform
import subprocess
from pathlib import Path
from typing import Any

import stage_12c2c_candidate_training as base


STAGE = "12C-3A"
ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "config/v2_2"
RESULT1 = ROOT / "results/circuitsage_hmac_v2_12c1"
RESULT3 = ROOT / "results/circuitsage_hmac_v2_12c3"
WORK = RESULT3 / "test_capture_authorization_12c3a"
LOCK_FILE = WORK / ".stage_12c3a.lock"

ACCEPTANCE_1A = CONFIG / "circuitsage_hmac_v2_2_acceptance_contract_12c1a.json"
SPLIT_1B = CONFIG / "circuitsage_hmac_v2_2_family_split_authorization_12c1b.json"
AUTH_1D = CONFIG / "circuitsage_hmac_v2_2_train_calibration_synthesis_authorization_12c1d.json"
PREDICTIONS_2H = CONFIG / "circuitsage_hmac_v2_2_sealed_circuit_predictions_12c2h.json"
BOUND_2L = CONFIG / "circuitsage_hmac_v2_2_structural_equivalence_bound_12c2l.json"
AUDIT_2H = ROOT / "results/circuitsage_hmac_v2_12c2/circuitsage_hmac_v2_2_prediction_freeze_12c2h.json"
AUDIT_2L = ROOT / "results/circuitsage_hmac_v2_12c2/circuitsage_hmac_v2_2_equivalence_bound_freeze_12c2l.json"
SOURCE_2L = ROOT / "stage_12c2l_equivalence_bound.py"
SOURCE_2H = ROOT / "stage_12c2h_prediction_freeze.py"

REGISTRY_1C = (RESULT1 / "multicircuit_corpus_12c1c"
               / "circuitsage_hmac_v2_2_acquired_family_registry_12c1c.csv")
ARCHIVES_1C = RESULT1 / "multicircuit_corpus_12c1c/archives"

PINNED = {
    ACCEPTANCE_1A: "9c8eec4d85957c4408ac59e0c8760af90c91d0667c995d91ac969b5a8f205f26",
    SPLIT_1B: "4103808fc389c78088e31c0a76549324c386ed9d8f7bedb7cc4d3748f9ea3303",
    AUTH_1D: "d57bdf8735c67ccb9c1467051774471e214ab2884f88cb789dbd79510896a684",
    PREDICTIONS_2H: "26d4e114dd2cfbcf1708448edf971039ecda7cbf62e1f7b15bdfcb78e2d89cf3",
    BOUND_2L: "aac6920932ced491bf9569e52853258b407882aefa498dcded825109cc3c62e8",
    AUDIT_2H: "a31e283264f9c6795429fbe442ff4258f589636760eaad295dd3d91946bd2273",
    AUDIT_2L: "bfd1df4d20a74c7bbb0832fc59f7decb1da6e76fd937ecb046d425666c6a969a",
    SOURCE_2L: "bb71f56e5d2cc0b25b4aad4a90465edd218383f4307426d4a7ad88886060994d",
    SOURCE_2H: "b7665f3a3ebdcccef296653f5fd9279b0fa529d68d612282524b3a5be21cf006",
}

AUTHORIZATION = CONFIG / "circuitsage_hmac_v2_2_independent_test_capture_authorization_12c3a.json"
TOOLCHAIN = WORK / "circuitsage_hmac_v2_2_toolchain_verification_12c3a.json"
SEAL_RECORD = WORK / "circuitsage_hmac_v2_2_holdout_seal_record_12c3a.json"
SOURCE_INTEGRITY = WORK / "circuitsage_hmac_v2_2_test_source_integrity_12c3a.csv"
PREFLIGHT = WORK / "circuitsage_hmac_v2_2_capture_preflight_12c3a.json"
ONESHOT = WORK / "circuitsage_hmac_v2_2_one_shot_acknowledgement_12c3a.json"
REPORT = WORK / "circuitsage_hmac_v2_2_test_capture_authorization_report_12c3a.md"
MANIFEST = RESULT3 / "circuitsage_hmac_v2_2_test_capture_authorization_manifest_12c3a.json"
AUDIT = RESULT3 / "circuitsage_hmac_v2_2_test_capture_authorization_freeze_12c3a.json"

TEST_FAMILIES = ("ibex_cpu", "secworks_chacha")
HOLDOUT_FAMILY = "serv_cpu"
FUTURE_BRAND = base.FUTURE_BRAND

stop, require, now = base.stop, base.require, base.now
canonical_json, sha256, rel = base.canonical_json, base.sha256, base.rel
record, load_json, read_csv, csv_bytes = base.record, base.load_json, base.read_csv, base.csv_bytes
frozen_write = base.frozen_write


def verify_toolchain() -> dict[str, Any]:
    """The pinned executables at the pinned paths, or refuse."""
    spec = load_json(AUTH_1D)["required_tools"]
    out: dict[str, Any] = {}
    for tool, meta in sorted(spec.items()):
        exe = Path(meta["executable"])
        want = meta["version"]
        require(exe.is_file(), f"pinned {tool} missing at {exe}")
        proc = subprocess.run([str(exe), "--version"], capture_output=True, text=True)
        got = (proc.stdout or proc.stderr).strip().splitlines()[0]
        require(got == want,
                f"{tool} version mismatch\n  pinned: {want}\n  found : {got}")
        out[tool] = {"executable": str(exe), "pinned_version": want,
                     "observed_version": got, "match": True,
                     "sha256": sha256(exe)}
        print(f"  {tool:<10} {got}", flush=True)
    return out


def verify_sources() -> list[dict[str, Any]]:
    """Sealed test archives must match the 12C-1C registry exactly."""
    rows = read_csv(REGISTRY_1C)
    out = []
    for r in rows:
        fam = r["family_id"]
        if fam not in TEST_FAMILIES:
            continue
        require(r["partition"] == "INDEPENDENT_CIRCUIT_TEST",
                f"{fam} must be INDEPENDENT_CIRCUIT_TEST, got {r['partition']}")
        archive = next(ARCHIVES_1C.glob(f"{fam}-*.tar.gz"), None)
        require(archive is not None, f"archive missing for {fam}")
        actual = sha256(archive)
        require(actual == r["archive_sha256"],
                f"{fam} archive SHA drifted since 12C-1C")
        out.append({
            "family_id": fam, "partition": r["partition"],
            "design_class": r["design_class"], "repository": r["repository"],
            "revision": r["revision"], "spdx_license": r["spdx_license"],
            "rtl_file_count": r["rtl_file_count"], "source_bytes": r["source_bytes"],
            "archive": rel(archive), "archive_sha256": actual,
            "registry_match": "YES",
        })
        print(f"  {fam:<18} {r['design_class']:<22} rev {r['revision'][:12]} "
              f"archive OK", flush=True)
    require(len(out) == len(TEST_FAMILIES), "both test families verified")
    return out


def verify_holdout_untouched() -> dict[str, Any]:
    """serv_cpu must have source only - no capture artifact anywhere."""
    capture_markers = ("full_campaign", "parallel_campaign", "raw_batches",
                       "graph_dataset", "adapter_pilot", "site_eligibility",
                       "candidate_training", "gate_measurement", "reranking",
                       "test_capture")
    permitted = ("multicircuit_corpus_12c1c",)
    leaked, allowed = [], 0
    for hit in ROOT.glob(f"**/*{HOLDOUT_FAMILY}*"):
        if not hit.is_file():
            continue
        p = rel(hit)
        if any(m in p for m in permitted):
            allowed += 1
            continue
        if any(m in p for m in capture_markers):
            leaked.append(p)
    require(not leaked, f"holdout capture artifacts found: {leaked[:4]}")
    print(f"  {HOLDOUT_FAMILY:<18} sealed; {allowed} source-only files, "
          f"0 capture artifacts", flush=True)
    return {"family_id": HOLDOUT_FAMILY, "partition": "GENERALIZATION_HOLDOUT",
            "seal_status": "SEALED", "capture_authorized": False,
            "source_only_files": allowed, "capture_artifacts": 0}


def execute() -> None:
    require(ROOT.name == "vlsi_fault_detection_v2", f"place script in project root, not {ROOT}")
    outputs = (AUTHORIZATION, TOOLCHAIN, SEAL_RECORD, SOURCE_INTEGRITY,
               PREFLIGHT, ONESHOT, REPORT, MANIFEST, AUDIT)
    require(not any(p.exists() for p in outputs),
            f"frozen Stage {STAGE} output exists; use --status")

    print("FROZEN INPUT VERIFICATION", flush=True)
    for path, expected in PINNED.items():
        require(path.is_file(), f"missing frozen input: {rel(path)}")
        require(sha256(path) == expected, f"frozen input SHA changed: {path.name}")
    print(f"  {len(PINNED)} frozen inputs (1A/1B/1D contracts, 12C-2H, 12C-2L)"
          f"{'':<8}: OK", flush=True)

    acceptance = load_json(ACCEPTANCE_1A)
    split = load_json(SPLIT_1B)
    auth1d = load_json(AUTH_1D)
    preds = load_json(PREDICTIONS_2H)

    assign = split["family_assignments"]
    for fam in TEST_FAMILIES:
        require(assign.get(fam) == "INDEPENDENT_CIRCUIT_TEST",
                f"{fam} partition mismatch")
    require(assign.get(HOLDOUT_FAMILY) == "GENERALIZATION_HOLDOUT",
            "holdout partition mismatch")
    require(sorted(auth1d["locked_test_families"]) == sorted(TEST_FAMILIES),
            "12C-1D locked-test-family list must match this stage")
    require(auth1d["sealed_holdout_families"] == [HOLDOUT_FAMILY],
            "12C-1D holdout seal must match")
    require(len(TEST_FAMILIES) >= 2,
            "acceptance contract requires at least two independent test families")
    print(f"  partitions confirmed; 12C-1D locked exactly {list(TEST_FAMILIES)}"
          f"{'':<3}: OK", flush=True)

    existing = sorted(p.name for p in CONFIG.glob("*.json"))
    require(not any("independent_test_capture_authorization" in n for n in existing),
            "a test capture authorization already exists")
    print(f"  config/v2_2 enumerated ({len(existing)} contracts); none prior"
          f"{'':<13}: OK", flush=True)

    print("\nTOOLCHAIN VERIFICATION (contract-pinned paths)", flush=True)
    tools = verify_toolchain()

    print("\nSEALED TEST SOURCE INTEGRITY", flush=True)
    sources = verify_sources()

    print("\nHOLDOUT SEAL", flush=True)
    seal = verify_holdout_untouched()

    print("\nPREDICTIONS ALREADY FROZEN FOR THESE CIRCUITS", flush=True)
    pred_index = {p["family_id"]: p for p in preds["predictions"]}
    for fam in TEST_FAMILIES:
        require(fam in pred_index, f"no frozen prediction for {fam}")
        p = pred_index[fam]
        lo, hi = p["predicted_exact_site_band"]
        print(f"  {fam:<18} exact_site [{lo:.2f}, {hi:.2f}]  "
              f"floor={p['predicted_meets_floor_0_15']:<3} conf={p['confidence']}",
              flush=True)
    require(preds["frozen_before_capture"] is True, "predictions frozen before capture")
    require(preds["sealed_data_read"] is False, "12C-2H read no sealed data")

    created = now()
    floor = acceptance["per_circuit_floor"]

    authorization = {
        "authorization_version": "CIRCUITSAGE-HMAC-V2.2-INDEPENDENT-TEST-CAPTURE-AUTHORIZATION-12C3A-v1",
        "stage": STAGE, "status": "FROZEN", "created_at": created,
        "authorized_families": list(TEST_FAMILIES),
        "authorized_partitions": ["INDEPENDENT_CIRCUIT_TEST"],
        "authorized_next_action": (
            "ELABORATE, BUILD PORTABLE ADAPTERS, LINT, GENERICALLY SYNTHESIZE, "
            "ENUMERATE SITES, INSTRUMENT FOR FAULT INJECTION, AND CAPTURE RESPONSES "
            "FOR THE TWO INDEPENDENT TEST FAMILIES ONLY"),
        "fault_injection": "AUTHORIZED FOR THE TWO INDEPENDENT TEST FAMILIES ONLY",
        "response_dataset_creation": "AUTHORIZED FOR THE TWO INDEPENDENT TEST FAMILIES ONLY",
        "sealed_holdout_families": [HOLDOUT_FAMILY],
        "holdout_capture": "NOT AUTHORIZED",
        "holdout_rationale": (
            "the acceptance contract requires at least two independent test families; "
            "ibex_cpu and secworks_chacha satisfy it, so the holdout is retained as an "
            "untouched reserve because a captured circuit cannot be re-sealed"),
        "model_training_inference": "NOT AUTHORIZED / NOT AUTHORIZED",
        "retraining_threshold_change_reselection": "PROHIBITED",
        "acceptance_evaluation": "NOT AUTHORIZED BY THIS STAGE",
        "execution": "SEQUENTIAL PER FAMILY / CHECKPOINT REQUIRED / PARALLEL BATCHES PERMITTED",
        "parallel_batch_justification": (
            "Stage 12C-1O established batch independence: batches are pure functions of "
            "(binary, site range, vectors) with private output paths and a serialized "
            "checkpoint write"),
        "failure_rule": (
            "FREEZE FAILURE DIAGNOSTIC; DO NOT MODIFY UPSTREAM RTL, DO NOT OPEN THE "
            "HOLDOUT, DO NOT RELAX ACCEPTANCE CRITERIA"),
        "required_tools": {k: {"executable": v["executable"],
                               "version": v["pinned_version"]} for k, v in tools.items()},
        "required_checks": [
            "source-tree SHA before and after every family",
            "dependency resolution without source edits",
            "portable-wrapper lint and elaboration clean",
            "fault-free baseline byte-equality across batches",
            "per-batch replay verification",
            "prediction-before-truth: capture must not consult 12C-2H bands",
        ],
        "test_validation_holdout_access_before_this_stage": [0, 0, 0],
        "next_stage": "12C-3B INDEPENDENT TEST CAPTURE EXECUTION",
    }

    oneshot = {
        "acknowledgement_version": "CIRCUITSAGE-HMAC-V2.2-ONE-SHOT-ACKNOWLEDGEMENT-12C3A-v1",
        "stage": STAGE, "created_at": created,
        "one_shot_rule": acceptance["one_shot_rule"],
        "irreversible": True,
        "consequence": (
            "after capture the acceptance evaluation may be performed exactly once; no "
            "retraining, threshold change or candidate reselection is permitted afterward"),
        "per_circuit_floor": floor,
        "frozen_prediction_overall": preds["predicted_overall_acceptance"],
        "frozen_prediction_per_circuit": {
            fam: pred_index[fam]["predicted_meets_floor_0_15"] for fam in TEST_FAMILIES},
        "expected_outcome_stated_before_capture": (
            "NOT MET, driven by the ibex_cpu per-circuit exact-site floor"),
        "user_decision": "PROCEED WITH TWO TEST FAMILIES; HOLD THE HOLDOUT SEALED",
        "evidence_that_result_cannot_be_retrofitted": [
            "12C-2H froze per-circuit predicted bands before capture existed",
            "12C-2L froze the structural bound before capture existed",
            "acceptance criteria were frozen in 12C-1A before any model existed",
        ],
    }

    preflight = {
        "preflight_version": "CIRCUITSAGE-HMAC-V2.2-CAPTURE-PREFLIGHT-12C3A-v1",
        "stage": STAGE, "status": "PASS", "created_at": created,
        "capture_performed_by_this_stage": False,
        "synthesis_calls": 0, "simulation_calls": 0, "fault_injections": 0,
        "training_calls": 0, "selection_calls": 0,
        "sealed_test_trees_opened": 0,
        "holdout_trees_opened": 0,
        "toolchain_matches_pinned": True,
        "test_archives_match_registry": True,
        "predictions_frozen_before_capture": True,
        "acceptance_criteria_invented": False,
        "acceptance_criteria_relaxed": False,
    }

    src_fields = list(sources[0].keys())
    pred_rows = "\n".join(
        f"| `{fam}` | {pred_index[fam]['predicted_circuit_class']} | "
        f"[{pred_index[fam]['predicted_exact_site_band'][0]:.2f}, "
        f"{pred_index[fam]['predicted_exact_site_band'][1]:.2f}] | "
        f"{pred_index[fam]['predicted_meets_floor_0_15']} | "
        f"{pred_index[fam]['confidence']} |" for fam in TEST_FAMILIES)
    src_rows = "\n".join(
        f"| `{s['family_id']}` | {s['design_class']} | `{s['revision'][:12]}` | "
        f"{s['spdx_license']} | {s['rtl_file_count']} | OK |" for s in sources)
    tool_rows = "\n".join(
        f"| `{k}` | `{v['observed_version']}` | MATCH |" for k, v in tools.items())

    report = f"""# Stage {STAGE} — Independent Test Capture Authorization

**Status: PASS / FROZEN — authorization only. No capture performed.**

## Why this stage exists

Stage 12C-1D authorized synthesis for TRAIN and CALIBRATION families only, and
recorded `locked_test_families: {list(TEST_FAMILIES)}`. Nothing in the frozen
record authorizes opening those trees. This stage creates that authorization
explicitly, so unsealing is a recorded decision rather than a side effect.

## Authorized

- elaboration, adapter construction, lint, generic synthesis
- site enumeration and fault-catalogue derivation
- fault-injection instrumentation and response capture

…for **`{TEST_FAMILIES[0]}`** and **`{TEST_FAMILIES[1]}`** only.

## Not authorized

| item | status |
|---|---|
| `{HOLDOUT_FAMILY}` capture | **NOT AUTHORIZED — remains SEALED** |
| model training / inference | NOT AUTHORIZED |
| retraining, threshold change, reselection | **PROHIBITED** |
| acceptance evaluation | requires its own stage |

`{HOLDOUT_FAMILY}` stays sealed deliberately: the contract requires *at least
two* independent test families, which the two above satisfy. A captured circuit
cannot be re-sealed, so one untouched circuit is retained as a reserve.

## Toolchain — contract-pinned

| tool | version | |
|---|---|---|
{tool_rows}

Verified at the exact paths pinned in 12C-1D. A capture produced by a different
toolchain would not be comparable to the frozen development evidence.

## Sealed test sources

| family | class | revision | license | RTL files | archive |
|---|---|---|---|---|---|
{src_rows}

## Predictions already frozen for these circuits

| circuit | predicted class | predicted exact-site | meets floor 0.15 | confidence |
|---|---|---|---|---|
{pred_rows}

**Frozen predicted overall acceptance: {preds['predicted_overall_acceptance']}.**

## One-shot acknowledgement

> {acceptance['one_shot_rule']}

Capture is **irreversible**. Expected outcome, stated before capture: **NOT
MET**, driven by the `ibex_cpu` per-circuit exact-site floor of
{floor['all_injected_exact_site_rate_min']}.

The result cannot be retrofitted in either direction: acceptance criteria were
frozen in 12C-1A before any model existed, predictions in 12C-2H before capture
existed, and the structural bound in 12C-2L.

## Access counters before this stage

TEST / VALIDATION / HOLDOUT = **0 / 0 / 0**

## Next stage

**12C-3B** — independent test capture execution. Independent generalization
remains **NOT ESTABLISHED** until evaluated. Future hybrid brand remains
**{FUTURE_BRAND}**.
"""

    frozen_write(AUTHORIZATION, canonical_json(authorization))
    frozen_write(TOOLCHAIN, canonical_json({
        "toolchain_version": "CIRCUITSAGE-HMAC-V2.2-TOOLCHAIN-VERIFICATION-12C3A-v1",
        "stage": STAGE, "created_at": created, "tools": tools,
        "all_match_pinned": True}))
    frozen_write(SEAL_RECORD, canonical_json({
        "seal_version": "CIRCUITSAGE-HMAC-V2.2-HOLDOUT-SEAL-RECORD-12C3A-v1",
        "stage": STAGE, "created_at": created, "holdout": seal,
        "decision": "HOLD SEALED", "decided_by": "USER",
        "rationale": authorization["holdout_rationale"]}))
    frozen_write(SOURCE_INTEGRITY, csv_bytes(sources, src_fields))
    frozen_write(PREFLIGHT, canonical_json(preflight))
    frozen_write(ONESHOT, canonical_json(oneshot))
    frozen_write(REPORT, report.encode())

    stage_outputs = (AUTHORIZATION, TOOLCHAIN, SEAL_RECORD, SOURCE_INTEGRITY,
                     PREFLIGHT, ONESHOT, REPORT)
    manifest = {
        "manifest_version": "CIRCUITSAGE-HMAC-V2.2-TEST-CAPTURE-AUTHORIZATION-MANIFEST-12C3A-v1",
        "stage": STAGE, "status": "PASS", "created_at": created,
        "stage_source": record(Path(__file__).resolve()),
        "frozen_inputs": {rel(p): record(p) for p in PINNED},
        "outputs": {rel(p): record(p) for p in stage_outputs},
        "authorized_families": list(TEST_FAMILIES),
        "sealed_holdout_families": [HOLDOUT_FAMILY],
        "capture_performed": False,
        "synthesis_calls": 0, "simulation_calls": 0,
        "independent_test_access": 0, "validation_access": 0, "holdout_access": 0,
    }
    frozen_write(MANIFEST, canonical_json(manifest))

    audit = {
        "audit_version": "CIRCUITSAGE-HMAC-V2.2-TEST-CAPTURE-AUTHORIZATION-FREEZE-12C3A-v1",
        "stage": STAGE, "status": "PASS",
        "authorized_families": list(TEST_FAMILIES),
        "authorized_partitions": ["INDEPENDENT_CIRCUIT_TEST"],
        "holdout_family": HOLDOUT_FAMILY,
        "holdout_capture_authorized": False,
        "holdout_seal_status": "SEALED",
        "toolchain_matches_pinned": True,
        "toolchain": {k: v["observed_version"] for k, v in tools.items()},
        "test_archives_match_registry": True,
        "capture_performed": False,
        "fault_injection_authorized_for": list(TEST_FAMILIES),
        "model_training_authorized": False,
        "retraining_threshold_change_reselection": "PROHIBITED",
        "acceptance_evaluation": "NOT AUTHORIZED BY THIS STAGE",
        "one_shot_rule": acceptance["one_shot_rule"],
        "frozen_predicted_overall_acceptance": preds["predicted_overall_acceptance"],
        "predictions_frozen_before_capture": True,
        "independent_test_validation_holdout_access": [0, 0, 0],
        "independent_generalization": "NOT ESTABLISHED",
        "authorization_record": record(AUTHORIZATION),
        "toolchain_record": record(TOOLCHAIN),
        "seal_record": record(SEAL_RECORD),
        "one_shot_record": record(ONESHOT),
        "manifest_record": record(MANIFEST),
        "future_combined_model_brand": FUTURE_BRAND,
        "next_gate": "STAGE 12C-3B — INDEPENDENT TEST CAPTURE EXECUTION",
    }
    frozen_write(AUDIT, canonical_json(audit))

    print(f"\n{'Stage':<52}: {STAGE} — TEST CAPTURE AUTHORIZATION")
    print(f"{'Status':<52}: PASS / FROZEN")
    print(f"{'Authorized families':<52}: {', '.join(TEST_FAMILIES)}")
    print(f"{'Holdout ' + HOLDOUT_FAMILY:<52}: SEALED / NOT AUTHORIZED")
    print(f"{'Toolchain matches 12C-1D pins':<52}: YES")
    print(f"{'Capture performed by this stage':<52}: NO")
    print(f"{'Frozen predicted acceptance':<52}: "
          f"{preds['predicted_overall_acceptance']}")
    print(f"{'TEST / VALIDATION / HOLDOUT access':<52}: 0 / 0 / 0")
    print(f"{'Audit SHA':<52}: {sha256(AUDIT)}")
    print(f"{'Next gate':<52}: STAGE 12C-3B — CAPTURE EXECUTION")


def status() -> None:
    print(f"STAGE {STAGE} — TEST CAPTURE AUTHORIZATION STATUS")
    if not (MANIFEST.is_file() and AUDIT.is_file()):
        print("Status                    : NOT FROZEN")
        return
    a = load_json(AUDIT)
    print("Status                    : PASS / FROZEN")
    print(f"Authorized families       : {a['authorized_families']}")
    print(f"Holdout                   : {a['holdout_family']} = {a['holdout_seal_status']}")
    print(f"Toolchain                 : {a['toolchain']}")
    print(f"Predicted acceptance      : {a['frozen_predicted_overall_acceptance']}")
    print(f"Audit SHA                 : {sha256(AUDIT)}")


def self_test() -> None:
    require(len(TEST_FAMILIES) == 2, "exactly two independent test families")
    require(HOLDOUT_FAMILY not in TEST_FAMILIES, "holdout is not a test family")
    a = load_json(AUTH_1D)
    require(sorted(a["locked_test_families"]) == sorted(TEST_FAMILIES),
            "12C-1D lock list agreement")
    require(a["sealed_holdout_families"] == [HOLDOUT_FAMILY], "12C-1D holdout agreement")
    p = load_json(PREDICTIONS_2H)
    have = {x["family_id"] for x in p["predictions"]}
    require(set(TEST_FAMILIES) <= have, "predictions exist for both test families")
    print(f"Stage {STAGE} self-test: PASS")


def locked_execute() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    with LOCK_FILE.open("a+", encoding="utf-8") as h:
        try:
            fcntl.flock(h.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            stop(f"Stage {STAGE} execution lock is held by another process")
        execute()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    m = p.add_mutually_exclusive_group()
    m.add_argument("--status", action="store_true")
    m.add_argument("--self-test", action="store_true")
    a = p.parse_args()
    if a.status:
        status()
    elif a.self_test:
        self_test()
    else:
        locked_execute()


if __name__ == "__main__":
    main()
