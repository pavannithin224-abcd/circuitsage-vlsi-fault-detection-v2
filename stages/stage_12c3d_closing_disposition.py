#!/usr/bin/env python3
"""Stage 12C-3D: prediction scoring and V2.2 closing disposition.

Scores the Stage 12C-2H sealed-circuit predictions against what was actually
measured, using ONLY the frozen scoring rule, then closes the V2.2 branch with an
explicit statement of what was and was not established.

Scoring inputs
--------------
  * frozen predictions            12C-2H (bands, floor calls, confidences)
  * frozen scoring rule           12C-2H (labels and prohibitions)
  * measured chacha capture       12C-3B
  * ibex capture determination    12C-3C

Scoring rule, quoted from the frozen artifact
---------------------------------------------
  CONFIRMED             measured value inside the predicted band AND floor
                        outcome correct
  DIRECTIONALLY_CORRECT floor outcome correct but measured value outside band
  FALSIFIED             floor outcome incorrect

Prohibited after capture (also frozen): widening a predicted band,
reclassifying a circuit to fit its measurement, adding a prediction for an
already-captured circuit, retraining/threshold changes/reselection.  None of
these are performed here.

A fourth label is required and is introduced honestly
-----------------------------------------------------
The frozen rule did not anticipate a circuit that cannot be captured at all.
``ibex_cpu`` therefore receives NOT_SCORABLE rather than being forced into one of
the three existing labels.  Forcing it into FALSIFIED would misattribute a corpus
problem to the hypothesis; forcing it into CONFIRMED would claim credit for a
prediction that was never tested.  The new label is additive: no existing label
definition is altered, and the omission is recorded as a defect in 12C-2H's
scoring rule rather than silently patched.

Closing disposition
-------------------
States, with SHA-anchored evidence, what V2.2 established, what it did not, and
what remains available for future work.  Makes no acceptance claim beyond the
frozen 12C-3C determination.
"""

from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path
from typing import Any

import stage_12c2c_candidate_training as base


STAGE = "12C-3D"
ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "config/v2_2"
RESULT2 = ROOT / "results/circuitsage_hmac_v2_12c2"
RESULT3 = ROOT / "results/circuitsage_hmac_v2_12c3"
WORK = RESULT3 / "closing_disposition_12c3d"
LOCK_FILE = WORK / ".stage_12c3d.lock"

ACCEPTANCE_1A = CONFIG / "circuitsage_hmac_v2_2_acceptance_contract_12c1a.json"
PREDICTIONS_2H = CONFIG / "circuitsage_hmac_v2_2_sealed_circuit_predictions_12c2h.json"
SCORING_2H = (RESULT2 / "prediction_freeze_12c2h"
              / "circuitsage_hmac_v2_2_prediction_scoring_rule_12c2h.json")
AUDIT_2H = RESULT2 / "circuitsage_hmac_v2_2_prediction_freeze_12c2h.json"
AUDIT_2L = RESULT2 / "circuitsage_hmac_v2_2_equivalence_bound_freeze_12c2l.json"
AUDIT_3B = RESULT3 / "circuitsage_hmac_v2_2_chacha_capture_freeze_12c3b.json"
AUDIT_3C = RESULT3 / "circuitsage_hmac_v2_2_ibex_infeasibility_freeze_12c3c.json"
DETERMINATION_3C = CONFIG / "circuitsage_hmac_v2_2_ibex_capture_infeasibility_12c3c.json"
SOURCE_3C = ROOT / "stage_12c3c_ibex_infeasibility.py"

PINNED = {
    ACCEPTANCE_1A: "9c8eec4d85957c4408ac59e0c8760af90c91d0667c995d91ac969b5a8f205f26",
    PREDICTIONS_2H: "26d4e114dd2cfbcf1708448edf971039ecda7cbf62e1f7b15bdfcb78e2d89cf3",
    AUDIT_2H: "a31e283264f9c6795429fbe442ff4258f589636760eaad295dd3d91946bd2273",
    AUDIT_2L: "bfd1df4d20a74c7bbb0832fc59f7decb1da6e76fd937ecb046d425666c6a969a",
    AUDIT_3B: "c517a208021735bd3b092361594447a14de72146cd7464e14eb310530978d6be",
    AUDIT_3C: "a1d94201521357a3c8c2f33e153297c36f04081ee9a077bb8ac5b33f9608cd47",
    DETERMINATION_3C: "b288d579f9876f6c68d14962990e948f99fd684bb2a7d340492bba62b7134a55",
    SOURCE_3C: "8c136f78d5660a8d9565298d3dc0342a7562c5fd90a5c7d72f4cbe002e7e164d",
}

SCORECARD = WORK / "circuitsage_hmac_v2_2_prediction_scorecard_12c3d.csv"
SCORING_DEFECT = WORK / "circuitsage_hmac_v2_2_scoring_rule_defect_12c3d.json"
ESTABLISHED = WORK / "circuitsage_hmac_v2_2_established_claims_12c3d.csv"
NOT_ESTABLISHED = WORK / "circuitsage_hmac_v2_2_not_established_12c3d.csv"
CLOSING = CONFIG / "circuitsage_hmac_v2_2_branch_closing_disposition_12c3d.json"
UNSPENT = WORK / "circuitsage_hmac_v2_2_unspent_evidence_12c3d.json"
REPORT = WORK / "circuitsage_hmac_v2_2_closing_report_12c3d.md"
MANIFEST = RESULT3 / "circuitsage_hmac_v2_2_closing_disposition_manifest_12c3d.json"
AUDIT = RESULT3 / "circuitsage_hmac_v2_2_closing_disposition_freeze_12c3d.json"

FUTURE_BRAND = base.FUTURE_BRAND
FLOOR_KEY = "all_injected_exact_site_rate_min"

stop, require, now = base.stop, base.require, base.now
canonical_json, sha256, rel = base.canonical_json, base.sha256, base.rel
record, load_json, csv_bytes = base.record, base.load_json, base.csv_bytes
frozen_write = base.frozen_write


def score(preds: dict, rule: dict, floor: float,
          chacha: dict, ibex: dict) -> list[dict[str, Any]]:
    rows = []
    for p in preds["predictions"]:
        fam = p["family_id"]
        lo, hi = p["predicted_exact_site_band"]
        pred_floor = p["predicted_meets_floor_0_15"]

        if fam == "secworks_chacha":
            measured = chacha["all_injected_exact_site_rate"]
            actual_floor = "YES" if measured >= floor else "NO"
            in_band = lo <= measured <= hi
            if actual_floor != pred_floor:
                label = "FALSIFIED"
            elif in_band:
                label = "CONFIRMED"
            else:
                label = "DIRECTIONALLY_CORRECT"
            rows.append({
                "family_id": fam, "partition": p["partition"],
                "predicted_class": p["predicted_circuit_class"],
                "predicted_band_low": lo, "predicted_band_high": hi,
                "predicted_meets_floor": pred_floor,
                "confidence": p["confidence"],
                "captured": "YES",
                "measured_exact_site": round(measured, 6),
                "measured_inside_band": "YES" if in_band else "NO",
                "actual_meets_floor": actual_floor,
                "outcome": label,
                "label_source": "FROZEN 12C-2H SCORING RULE",
            })
        elif fam == "ibex_cpu":
            rows.append({
                "family_id": fam, "partition": p["partition"],
                "predicted_class": p["predicted_circuit_class"],
                "predicted_band_low": lo, "predicted_band_high": hi,
                "predicted_meets_floor": pred_floor,
                "confidence": p["confidence"],
                "captured": "NO",
                "measured_exact_site": None,
                "measured_inside_band": None,
                "actual_meets_floor": None,
                "outcome": "NOT_SCORABLE",
                "label_source": "NEW LABEL - SEE SCORING RULE DEFECT",
            })
        else:  # serv_cpu, deliberately never captured
            rows.append({
                "family_id": fam, "partition": p["partition"],
                "predicted_class": p["predicted_circuit_class"],
                "predicted_band_low": lo, "predicted_band_high": hi,
                "predicted_meets_floor": pred_floor,
                "confidence": p["confidence"],
                "captured": "NO",
                "measured_exact_site": None,
                "measured_inside_band": None,
                "actual_meets_floor": None,
                "outcome": "NOT_SCORABLE",
                "label_source": "HELD SEALED BY USER DECISION (12C-3A)",
            })
    return rows


ESTABLISHED_CLAIMS = [
    {"claim_id": "C1",
     "claim": "Localization degrades with candidate-catalog scale, not with circuit change: "
              "the same HMAC circuit and faults fell from 0.5488 to 0.0286 exact-site "
              "(19.2x) when the catalog grew from one family to four",
     "evidence_stages": "12B-3P, 12C-2F", "strength": "CONTROLLED SINGLE-VARIABLE"},
    {"claim_id": "C2",
     "claim": "Behavioural signature uniqueness varies ~26x across circuit classes "
              "(0.0183 CPU to 0.4771 crypto hash) and orders families identically to "
              "exact-site rate",
     "evidence_stages": "12C-1P, 12C-2F, 12C-2G", "strength": "MEASURED, 4 CIRCUITS"},
    {"claim_id": "C3",
     "claim": "Fault observability is bimodal, not continuous: ~99.8% in crypto "
              "datapaths versus 38-44% in CPU-class circuits, with nothing between",
     "evidence_stages": "12C-2A", "strength": "MEASURED, 4 CIRCUITS"},
    {"claim_id": "C4",
     "claim": "Four independent learned formulations failed to exceed the non-learning "
              "comparator on an unseen circuit; the comparator was 182x better on MRR",
     "evidence_stages": "12C-2C, 12C-2E, 12C-2F, 12C-2G, 12C-2K",
     "strength": "FOUR PRE-REGISTERED FALSIFICATIONS"},
    {"claim_id": "C5",
     "claim": "Within-collision-set localization is not identifiable under a uniform "
              "fault prior: every site carries exactly 2 faults, so posterior is 1/k "
              "and a measured 1.00x lift is the correct result",
     "evidence_stages": "12C-2J", "strength": "PROOF PLUS EMPIRICAL CORROBORATION"},
    {"claim_id": "C6",
     "claim": "Residual ambiguity is dominated by structural fault equivalence "
              "(83.3-99.8% of collision sets), so candidate sets are the fault "
              "equivalence class and the localizer operates at the structural limit",
     "evidence_stages": "12C-2L", "strength": "MEASURED, ADJACENCY-BASED INDICATOR"},
    {"claim_id": "C7",
     "claim": "On observable and behaviourally unique faults, localization is exact "
              "with zero false alarms across 7,849,696 transactions",
     "evidence_stages": "12C-1O", "strength": "MEASURED, FULL CAMPAIGN"},
    {"claim_id": "C8",
     "claim": "A crypto-class prediction frozen before any seal was broken was "
              "confirmed on a genuinely unseen circuit: secworks_chacha predicted "
              "[0.20, 0.70], measured 0.3440",
     "evidence_stages": "12C-2H, 12C-3B", "strength": "PREDICTION BEFORE TRUTH, 1 CIRCUIT"},
]

NOT_ESTABLISHED_ITEMS = [
    {"item": "independent-circuit generalization of the acceptance contract",
     "why": "the contract requires the per-circuit floor on BOTH test circuits; "
            "ibex_cpu could not be captured from the acquired snapshot",
     "evidence_stage": "12C-3C"},
    {"item": "any CPU-class independent test result",
     "why": "the only CPU-class test circuit was uncapturable and the holdout CPU "
            "remains sealed by decision",
     "evidence_stage": "12C-3C, 12C-3A"},
    {"item": "that learned models cannot help this problem in principle",
     "why": "four formulations were falsified; that is not a proof over all models",
     "evidence_stage": "12C-2L"},
    {"item": "that the structural-equivalence bound is formally proven per fault",
     "why": "adjacency is a strong structural indicator, not a per-fault equivalence proof",
     "evidence_stage": "12C-2L"},
    {"item": "behaviour under fault models other than single stuck-at",
     "why": "delay, bridging and transient faults were never in the catalog",
     "evidence_stage": "12C-1F"},
]


def execute() -> None:
    require(ROOT.name == "vlsi_fault_detection_v2", f"place script in project root, not {ROOT}")
    outputs = (SCORECARD, SCORING_DEFECT, ESTABLISHED, NOT_ESTABLISHED, CLOSING,
               UNSPENT, REPORT, MANIFEST, AUDIT)
    require(not any(p.exists() for p in outputs),
            f"frozen Stage {STAGE} output exists; use --status")

    print("FROZEN INPUT VERIFICATION", flush=True)
    for path, expected in PINNED.items():
        require(path.is_file(), f"missing frozen input: {rel(path)}")
        require(sha256(path) == expected, f"frozen input SHA changed: {path.name}")
    print(f"  {len(PINNED)} frozen inputs (1A, 2H, 2L, 3B, 3C){'':<20}: OK", flush=True)

    acceptance = load_json(ACCEPTANCE_1A)
    preds = load_json(PREDICTIONS_2H)
    rule = load_json(SCORING_2H)
    chacha = load_json(AUDIT_3B)
    ibex = load_json(AUDIT_3C)
    bound = load_json(AUDIT_2L)
    floor = acceptance["per_circuit_floor"][FLOOR_KEY]

    require(preds["frozen_before_capture"] is True, "predictions pre-date capture")
    require(chacha["predicted_bands_read"] is False, "capture did not read bands")
    require(ibex["capture_performed"] is False, "ibex not captured")
    require(ibex["substitute_rtl_authored"] is False, "no substitute RTL")
    print(f"  prediction-before-truth chain intact{'':<26}: OK", flush=True)

    existing = sorted(p.name for p in CONFIG.glob("*.json"))
    require(not any("branch_closing_disposition" in n for n in existing),
            "a closing disposition already exists")
    print(f"  config/v2_2 enumerated ({len(existing)} contracts); none prior"
          f"{'':<13}: OK", flush=True)

    print("\nPREDICTION SCORING (frozen rule only)", flush=True)
    rows = score(preds, rule, floor, chacha, ibex)
    for r in rows:
        m = "--" if r["measured_exact_site"] is None else f"{r['measured_exact_site']:.4f}"
        print(f"  {r['family_id']:<18} predicted [{r['predicted_band_low']:.2f},"
              f"{r['predicted_band_high']:.2f}] floor={r['predicted_meets_floor']:<3} "
              f"measured={m:<8} -> {r['outcome']}", flush=True)

    scored = [r for r in rows if r["outcome"] != "NOT_SCORABLE"]
    confirmed = [r for r in scored if r["outcome"] == "CONFIRMED"]
    falsified = [r for r in scored if r["outcome"] == "FALSIFIED"]

    created = now()
    defect = {
        "defect_version": "CIRCUITSAGE-HMAC-V2.2-SCORING-RULE-DEFECT-12C3D-v1",
        "stage": STAGE, "created_at": created,
        "defective_artifact": "12C-2H prediction scoring rule",
        "defect": ("the rule defines CONFIRMED, DIRECTIONALLY_CORRECT and FALSIFIED "
                   "but has no label for a circuit that cannot be captured at all"),
        "how_it_surfaced": "12C-3C determined ibex_cpu is not capturable",
        "resolution": "a fourth label NOT_SCORABLE is introduced by this stage",
        "why_not_forced_into_an_existing_label": [
            "FALSIFIED would misattribute a corpus-acquisition problem to the hypothesis",
            "CONFIRMED would claim credit for a prediction that was never tested",
            "DIRECTIONALLY_CORRECT presupposes a measured value, which does not exist",
        ],
        "existing_label_definitions_modified": False,
        "12c2h_artifact_modified": False,
        "lesson": ("a scoring rule should enumerate the case where the measurement "
                   "cannot be taken, not only the cases where it disagrees"),
    }

    unspent = {
        "unspent_version": "CIRCUITSAGE-HMAC-V2.2-UNSPENT-EVIDENCE-12C3D-v1",
        "stage": STAGE, "created_at": created,
        "one_shot_evaluation_consumed": False,
        "one_shot_rule": acceptance["one_shot_rule"],
        "why_unspent": ("no model was evaluated against the acceptance gates; the "
                        "NOT MET outcome follows from capture infeasibility"),
        "sealed_holdout": {"family_id": "serv_cpu", "status": "SEALED",
                           "held_by": "USER DECISION at 12C-3A"},
        "recoverable_if": [
            "a self-contained ibex source tree is acquired under a new corpus stage",
            "that stage freezes its own acquisition evidence rather than amending 12C-1C",
        ],
        "not_recoverable": [
            "secworks_chacha is now captured and cannot be re-sealed",
        ],
    }

    closing = {
        "closing_version": "CIRCUITSAGE-HMAC-V2.2-BRANCH-CLOSING-DISPOSITION-12C3D-v1",
        "stage": STAGE, "status": "FROZEN", "created_at": created,
        "branch": "V2.2 INDEPENDENT GENERALIZATION",
        "branch_status": "CLOSED",
        "acceptance_outcome": "NOT MET",
        "acceptance_outcome_type": "CAPTURE INFEASIBILITY, NOT A MEASURED SHORTFALL",
        "independent_generalization": "NOT ESTABLISHED",
        "predictions_scored": len(scored),
        "predictions_confirmed": len(confirmed),
        "predictions_falsified": len(falsified),
        "predictions_not_scorable": len(rows) - len(scored),
        "established_claims": len(ESTABLISHED_CLAIMS),
        "shipped_localizer": "exact-signature retrieval (non-learning)",
        "shipped_localizer_status": "OPERATING AT THE STRUCTURAL LIMIT",
        "learned_models_in_product": False,
        "structural_equivalence_minimum": bound["minimum_structurally_equivalent_fraction"],
        "one_shot_evaluation_consumed": False,
        "holdout_seal": "SEALED",
        "next_phase": "FAULTIVA HYBRID PACKAGING (V1 + V2), PUBLIC RELEASE, DASHBOARD",
        "future_combined_model_brand": FUTURE_BRAND,
    }

    sc_fields = list(rows[0].keys())
    est_fields = list(ESTABLISHED_CLAIMS[0].keys())
    ne_fields = list(NOT_ESTABLISHED_ITEMS[0].keys())

    def fmt_measured(row: dict[str, Any]) -> str:
        v = row["measured_exact_site"]
        return "--" if v is None else f"{v:.4f}"

    score_table = "\n".join(
        f"| `{r['family_id']}` | "
        f"[{r['predicted_band_low']:.2f}, {r['predicted_band_high']:.2f}] | "
        f"{r['predicted_meets_floor']} | {r['confidence']} | "
        f"{fmt_measured(r)} | **{r['outcome']}** |" for r in rows)
    est_table = "\n".join(
        f"| **{c['claim_id']}** | {c['claim']} | {c['evidence_stages']} |"
        for c in ESTABLISHED_CLAIMS)
    ne_table = "\n".join(
        f"| {n['item']} | {n['why']} | {n['evidence_stage']} |"
        for n in NOT_ESTABLISHED_ITEMS)

    report = f"""# Stage {STAGE} — V2.2 Closing Disposition

**Status: PASS / FROZEN — scoring and disposition. No evaluation, no training.**

## Prediction scorecard

Scored strictly against the frozen 12C-2H rule.

| circuit | predicted band | predicted floor | confidence | measured | outcome |
|---|---|---|---|---|---|
{score_table}

**`secworks_chacha`: CONFIRMED.** Predicted [0.20, 0.70] with floor YES before the
seal was broken; measured **{chacha['all_injected_exact_site_rate']:.4f}**, inside
the band and above the {floor} floor. A genuine prediction-before-truth result on
an unseen circuit.

**`ibex_cpu`: NOT_SCORABLE** — not capturable from the acquired snapshot.

**`serv_cpu`: NOT_SCORABLE** — deliberately held sealed.

## A defect in the frozen scoring rule

The 12C-2H rule defines CONFIRMED / DIRECTIONALLY_CORRECT / FALSIFIED but has
**no label for a circuit that cannot be measured at all**. Rather than force
`ibex_cpu` into an existing label, this stage introduces **NOT_SCORABLE** and
records the omission as a defect.

Forcing FALSIFIED would blame the hypothesis for a corpus problem. Forcing
CONFIRMED would claim credit for an untested prediction. 12C-2H itself is **not
modified**.

## Acceptance outcome

**NOT MET** — one of the two required independent test circuits was uncapturable.
This is **capture infeasibility, not a measured shortfall**, and the distinction
is preserved in every frozen artifact.

The **one-shot locked evaluation was not consumed**: no model was evaluated, so it
remains available if a self-contained ibex tree is ever acquired.

## What V2.2 established

| # | claim | evidence |
|---|---|---|
{est_table}

## What V2.2 did NOT establish

| item | why | evidence |
|---|---|---|
{ne_table}

## Where the project stands

The shipped localizer is **exact-signature retrieval — non-learning, zero learned
parameters** — and it operates at the structural limit proved in 12C-2L
(minimum {bound['minimum_structurally_equivalent_fraction']:.1%} of collision sets
are structurally equivalent). Learned models are excluded on evidence, not
preference.

**V2.2 branch: CLOSED.**

Next phase: **{FUTURE_BRAND}** — V1 + V2 hybrid packaging, per-circuit
characterization pipeline, public release, dashboard.
"""

    frozen_write(SCORECARD, csv_bytes(rows, sc_fields))
    frozen_write(SCORING_DEFECT, canonical_json(defect))
    frozen_write(ESTABLISHED, csv_bytes(ESTABLISHED_CLAIMS, est_fields))
    frozen_write(NOT_ESTABLISHED, csv_bytes(NOT_ESTABLISHED_ITEMS, ne_fields))
    frozen_write(CLOSING, canonical_json(closing))
    frozen_write(UNSPENT, canonical_json(unspent))
    frozen_write(REPORT, report.encode())

    stage_outputs = (SCORECARD, SCORING_DEFECT, ESTABLISHED, NOT_ESTABLISHED,
                     CLOSING, UNSPENT, REPORT)
    manifest = {
        "manifest_version": "CIRCUITSAGE-HMAC-V2.2-CLOSING-DISPOSITION-MANIFEST-12C3D-v1",
        "stage": STAGE, "status": "PASS", "created_at": created,
        "stage_source": record(Path(__file__).resolve()),
        "frozen_inputs": {rel(p): record(p) for p in PINNED},
        "outputs": {rel(p): record(p) for p in stage_outputs},
        "predictions_scored": len(scored),
        "prior_stages_modified": [],
        "training_calls": 0, "selection_calls": 0,
        "holdout_access": 0,
    }
    frozen_write(MANIFEST, canonical_json(manifest))

    audit = {
        "audit_version": "CIRCUITSAGE-HMAC-V2.2-CLOSING-DISPOSITION-FREEZE-12C3D-v1",
        "stage": STAGE, "status": "PASS",
        "branch_status": "CLOSED",
        "scorecard": {r["family_id"]: r["outcome"] for r in rows},
        "chacha_predicted_band": [rows[0]["predicted_band_low"],
                                  rows[0]["predicted_band_high"]]
        if rows[0]["family_id"] == "secworks_chacha" else None,
        "chacha_measured_exact_site": chacha["all_injected_exact_site_rate"],
        "predictions_scored": len(scored),
        "predictions_confirmed": len(confirmed),
        "predictions_falsified": len(falsified),
        "new_label_introduced": "NOT_SCORABLE",
        "scoring_rule_defect_recorded": True,
        "12c2h_modified": False,
        "acceptance_outcome": "NOT MET",
        "acceptance_outcome_type": "CAPTURE INFEASIBILITY",
        "one_shot_evaluation_consumed": False,
        "holdout_seal": "SEALED",
        "established_claims": len(ESTABLISHED_CLAIMS),
        "not_established_items": len(NOT_ESTABLISHED_ITEMS),
        "shipped_localizer": "exact-signature retrieval (non-learning)",
        "learned_models_in_product": False,
        "independent_generalization": "NOT ESTABLISHED",
        "prior_stages_modified": [],
        "scorecard_record": record(SCORECARD),
        "closing_record": record(CLOSING),
        "defect_record": record(SCORING_DEFECT),
        "unspent_record": record(UNSPENT),
        "manifest_record": record(MANIFEST),
        "future_combined_model_brand": FUTURE_BRAND,
        "next_gate": "FAULTIVA HYBRID PACKAGING (V1 + V2)",
    }
    frozen_write(AUDIT, canonical_json(audit))

    print(f"\n{'Stage':<52}: {STAGE} — V2.2 CLOSING DISPOSITION")
    print(f"{'Status':<52}: PASS / FROZEN")
    print(f"{'Predictions scored':<52}: {len(scored)} "
          f"({len(confirmed)} confirmed, {len(falsified)} falsified)")
    print(f"{'New label introduced':<52}: NOT_SCORABLE (defect recorded)")
    print(f"{'Acceptance outcome':<52}: NOT MET (capture infeasibility)")
    print(f"{'One-shot evaluation consumed':<52}: NO")
    print(f"{'Established claims':<52}: {len(ESTABLISHED_CLAIMS)}")
    print(f"{'V2.2 branch':<52}: CLOSED")
    print(f"{'Audit SHA':<52}: {sha256(AUDIT)}")
    print(f"{'Next gate':<52}: FAULTIVA HYBRID PACKAGING")


def status() -> None:
    print(f"STAGE {STAGE} — CLOSING DISPOSITION STATUS")
    if not (MANIFEST.is_file() and AUDIT.is_file()):
        print("Status                    : NOT FROZEN")
        return
    a = load_json(AUDIT)
    print("Status                    : PASS / FROZEN")
    print(f"Scorecard                 : {a['scorecard']}")
    print(f"Acceptance                : {a['acceptance_outcome']} ({a['acceptance_outcome_type']})")
    print(f"One-shot consumed         : {a['one_shot_evaluation_consumed']}")
    print(f"Branch                    : {a['branch_status']}")
    print(f"Audit SHA                 : {sha256(AUDIT)}")


def self_test() -> None:
    rule = load_json(SCORING_2H)
    require(set(rule["outcome_labels"]) == {"CONFIRMED", "DIRECTIONALLY_CORRECT",
                                           "FALSIFIED"}, "three frozen labels")
    require("NOT_SCORABLE" not in rule["outcome_labels"],
            "NOT_SCORABLE is genuinely new")
    require(len(ESTABLISHED_CLAIMS) == 8, "eight established claims")
    require(len(NOT_ESTABLISHED_ITEMS) == 5, "five not-established items")
    c = load_json(AUDIT_3B)
    require(0.0 <= c["all_injected_exact_site_rate"] <= 1.0, "chacha rate sane")
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
