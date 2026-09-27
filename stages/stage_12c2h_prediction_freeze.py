#!/usr/bin/env python3
"""Stage 12C-2H: sealed-circuit prediction freeze (prediction before truth).

Stages 12C-2C, 12C-2E and 12C-2G tested three independent learned formulations
for localising faults on an unseen circuit.  All three failed to transfer:

    12C-2C  retrieval, uniform random negatives      no transfer
    12C-2E  retrieval, topology-stratified negatives no transfer
    12C-2G  graph-constrained set reranking          lift 1.00x (exactly none)

The surviving explanation is informational rather than architectural: faults
whose observable behaviour is identical cannot be separated by any function of
that behaviour, and the fraction of behaviourally unique faults varies by more
than an order of magnitude across circuit classes.

This stage converts that explanation into FALSIFIABLE PREDICTIONS about circuits
that have never been captured, and freezes them BEFORE capture exists.

Why this stage must precede Stage 12C-3A
----------------------------------------
The frozen acceptance contract permits ONE evaluation on the sealed circuits.
Once they are captured, a prediction can no longer be made in advance - only a
description after the fact.  Recording predictions now converts the sealed
capture from a measurement into a genuine test of a stated hypothesis.  The
option is available exactly once and expires the moment capture begins.

What is predicted
-----------------
For each uncaptured circuit (INDEPENDENT_CIRCUIT_TEST and GENERALIZATION_HOLDOUT)
this stage records a predicted circuit class, a predicted all-injected exact-site
band, the predicted per-circuit floor outcome, the reasoning, and the specific
observation that would FALSIFY the prediction.

This stage reads NO sealed data.  It reads only frozen development-partition
evidence and the frozen contracts.  The sealed circuits are named in the frozen
split authorisation; naming a circuit is not accessing it.

Prohibited here: capture, simulation, synthesis, training, selection, acceptance
evaluation, and any read of an INDEPENDENT_CIRCUIT_TEST or GENERALIZATION_HOLDOUT
artifact.
"""

from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path
from typing import Any

import stage_12c2c_candidate_training as base


STAGE = "12C-2H"
ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "config/v2_2"
RESULT2 = ROOT / "results/circuitsage_hmac_v2_12c2"
WORK = RESULT2 / "prediction_freeze_12c2h"
LOCK_FILE = WORK / ".stage_12c2h.lock"

SPLIT_1B = CONFIG / "circuitsage_hmac_v2_2_family_split_authorization_12c1b.json"
ACCEPTANCE_1A = CONFIG / "circuitsage_hmac_v2_2_acceptance_contract_12c1a.json"
SOURCE_2G = ROOT / "stage_12c2g_graph_reranking.py"
AUDIT_2G = RESULT2 / "circuitsage_hmac_v2_2_graph_reranking_freeze_12c2g.json"
MANIFEST_2G = RESULT2 / "circuitsage_hmac_v2_2_graph_reranking_manifest_12c2g.json"
FORMULATION_2G = CONFIG / "circuitsage_hmac_v2_2_reranking_formulation_12c2g.json"
AUDIT_2F = RESULT2 / "circuitsage_hmac_v2_2_gate_measurement_freeze_12c2f.json"

PINNED = {
    ACCEPTANCE_1A: "9c8eec4d85957c4408ac59e0c8760af90c91d0667c995d91ac969b5a8f205f26",
    SPLIT_1B: "4103808fc389c78088e31c0a76549324c386ed9d8f7bedb7cc4d3748f9ea3303",
    SOURCE_2G: "726ce7403261b2c1675047d8a8fecbeffab09bd8ac2b44330f49c2f95b695ae2",
    AUDIT_2G: "bbff14cdbcf3b53ee6a3cfd3753fc205e7f72a0606d885eac4be18f7f8a37144",
    MANIFEST_2G: "fb3de597caf7810e82bb33f4d36d7c4031525b68ce3bd92b7725e48befe3bb0a",
    FORMULATION_2G: "e7a01baf910128491a535b7e672b75e73d93a424fcb561b034ea98a04ab66740",
    AUDIT_2F: "aaa862b7ec4128eb8a3aa80233c8661ab799f1599ea1fed216bae49f98a66fb7",
}

PREDICTIONS = CONFIG / "circuitsage_hmac_v2_2_sealed_circuit_predictions_12c2h.json"
HYPOTHESIS = WORK / "circuitsage_hmac_v2_2_bounding_hypothesis_12c2h.json"
EVIDENCE = WORK / "circuitsage_hmac_v2_2_supporting_evidence_12c2h.csv"
FALSIFIERS = WORK / "circuitsage_hmac_v2_2_falsification_conditions_12c2h.csv"
SCORING_RULE = WORK / "circuitsage_hmac_v2_2_prediction_scoring_rule_12c2h.json"
PREFLIGHT = WORK / "circuitsage_hmac_v2_2_prediction_preflight_12c2h.json"
REPORT = WORK / "circuitsage_hmac_v2_2_prediction_report_12c2h.md"
MANIFEST = RESULT2 / "circuitsage_hmac_v2_2_prediction_freeze_manifest_12c2h.json"
AUDIT = RESULT2 / "circuitsage_hmac_v2_2_prediction_freeze_12c2h.json"

FUTURE_BRAND = base.FUTURE_BRAND
stop, require, now = base.stop, base.require, base.now
canonical_json, sha256, rel = base.canonical_json, base.sha256, base.rel
record, load_json, csv_bytes = base.record, base.load_json, base.csv_bytes
frozen_write = base.frozen_write

# Measured development evidence, quoted from frozen artifacts (12C-1P / 12C-2F / 12C-2G).
OBSERVED = [
    {"family_id": "secworks_sha256", "circuit_class": "CRYPTO_HASH",
     "partition": "GENERALIZATION_CALIBRATION", "signature_uniqueness": 0.4771,
     "detection_recall": 0.9467, "exact_site_baseline": 0.4516,
     "exact_site_reranked": 0.6385, "meets_floor_0_15": "YES"},
    {"family_id": "secworks_aes", "circuit_class": "CRYPTO_BLOCK",
     "partition": "GENERALIZATION_TRAIN", "signature_uniqueness": 0.3783,
     "detection_recall": 0.9714, "exact_site_baseline": 0.3675,
     "exact_site_reranked": 0.6359, "meets_floor_0_15": "YES"},
    {"family_id": "opentitan_hmac_sha256", "circuit_class": "CRYPTO_MAC_WRAPPED",
     "partition": "GENERALIZATION_TRAIN", "signature_uniqueness": 0.1070,
     "detection_recall": 0.2676, "exact_site_baseline": 0.0286,
     "exact_site_reranked": 0.0785, "meets_floor_0_15": "NO"},
    {"family_id": "picorv32_cpu", "circuit_class": "CPU_RISCV",
     "partition": "GENERALIZATION_TRAIN", "signature_uniqueness": 0.0183,
     "detection_recall": 0.2459, "exact_site_baseline": 0.0045,
     "exact_site_reranked": 0.0404, "meets_floor_0_15": "NO"},
]

PREDICTED = [
    {
        "family_id": "ibex_cpu",
        "partition": "INDEPENDENT_CIRCUIT_TEST",
        "predicted_circuit_class": "CPU_RISCV",
        "class_basis": "32-bit RISC-V integer core, same class as picorv32_cpu",
        "predicted_signature_uniqueness_band": [0.00, 0.10],
        "predicted_detection_recall_band": [0.15, 0.45],
        "predicted_exact_site_band": [0.00, 0.12],
        "predicted_meets_floor_0_15": "NO",
        "confidence": "HIGH",
        "reasoning": (
            "The only measured CPU (picorv32) shows 1.8% signature uniqueness and "
            "0.0045 baseline exact-site. ibex is a comparable RISC-V integer core with "
            "a deeper pipeline and more architectural state, which increases the number "
            "of faults masked before reaching an observable output. No learned "
            "formulation has recovered transferable discrimination inside a collision "
            "set (12C-2G lift 1.00x), so the exact-site rate should remain governed by "
            "signature uniqueness."),
        "falsified_if": "measured all_injected_exact_site_rate >= 0.15",
        "would_also_challenge": (
            "signature uniqueness above 0.25 would contradict the claim that CPU-class "
            "circuits are structurally ambiguous under this test scheme"),
    },
    {
        "family_id": "secworks_chacha",
        "partition": "INDEPENDENT_CIRCUIT_TEST",
        "predicted_circuit_class": "CRYPTO_ARX_STREAM",
        "class_basis": "ARX stream cipher, datapath-dominated like secworks_aes",
        "predicted_signature_uniqueness_band": [0.20, 0.60],
        "predicted_detection_recall_band": [0.80, 1.00],
        "predicted_exact_site_band": [0.20, 0.70],
        "predicted_meets_floor_0_15": "YES",
        "confidence": "MEDIUM",
        "reasoning": (
            "Both measured crypto datapath circuits (AES 0.3783, SHA256 0.4771 signature "
            "uniqueness) exceed the floor comfortably. ChaCha is an add-rotate-xor "
            "datapath whose state is propagated to the output every round, which is the "
            "structural property those two share. Confidence is MEDIUM rather than HIGH "
            "because ChaCha's quarter-round replication may create more symmetric "
            "duplicate structures than AES, and duplicate structure is exactly what "
            "produces signature collisions."),
        "falsified_if": "measured all_injected_exact_site_rate < 0.15",
        "would_also_challenge": (
            "detection recall below 0.60 would indicate an observability regime unlike "
            "either measured crypto circuit"),
    },
    {
        "family_id": "serv_cpu",
        "partition": "GENERALIZATION_HOLDOUT",
        "predicted_circuit_class": "CPU_RISCV_BIT_SERIAL",
        "class_basis": "bit-serial RISC-V core; CPU class but an extreme topology",
        "predicted_signature_uniqueness_band": [0.00, 0.15],
        "predicted_detection_recall_band": [0.10, 0.50],
        "predicted_exact_site_band": [0.00, 0.12],
        "predicted_meets_floor_0_15": "NO",
        "confidence": "LOW",
        "reasoning": (
            "SERV is bit-serial: it reuses one datapath slice across many cycles, so a "
            "single site is exercised repeatedly and its effect may appear in more "
            "output cycles than in a parallel core. That could RAISE uniqueness relative "
            "to picorv32, which is why confidence is LOW. Recorded explicitly because a "
            "surprise here would be informative rather than embarrassing: it would "
            "identify temporal reuse, not circuit family, as the governing variable."),
        "falsified_if": "measured all_injected_exact_site_rate >= 0.15",
        "would_also_challenge": (
            "a bit-serial core outperforming a parallel core would reframe the bounding "
            "variable from circuit class to datapath reuse"),
    },
]


def verify_inputs() -> dict[str, Any]:
    print("FROZEN INPUT VERIFICATION", flush=True)
    for path, expected in PINNED.items():
        require(path.is_file(), f"missing frozen input: {rel(path)}")
        require(sha256(path) == expected, f"frozen input SHA changed: {path.name}")
    print(f"  {len(PINNED)} frozen inputs (contracts + 12C-2F + 12C-2G evidence)"
          f"{'':<13}: OK", flush=True)

    split = load_json(SPLIT_1B)
    assign = split["family_assignments"]
    sealed = {f for f, p in assign.items()
              if p in ("INDEPENDENT_CIRCUIT_TEST", "GENERALIZATION_HOLDOUT")}
    predicted = {p["family_id"] for p in PREDICTED}
    require(predicted == sealed,
            f"prediction set must cover exactly the sealed families: {sorted(sealed)}")
    for p in PREDICTED:
        require(assign[p["family_id"]] == p["partition"],
                f"partition mismatch for {p['family_id']}")
    print(f"  prediction set == sealed families {sorted(sealed)}"
          f"{'':<10}: OK", flush=True)

    # Prove no CAPTURE artifact exists for any sealed family. Source archives
    # acquired in Stage 12C-1C are permitted: fetching public RTL is not
    # measuring it. What must not exist is campaign evidence - responses,
    # signatures, graphs, batches or metrics - for a sealed circuit.
    capture_markers = ("full_campaign", "parallel_campaign", "raw_batches",
                       "graph_dataset", "adapter_pilot", "site_eligibility",
                       "candidate_training", "gate_measurement", "reranking")
    permitted_markers = ("multicircuit_corpus_12c1c",)
    leaked = []
    for fam in sealed:
        for hit in ROOT.glob(f"**/*{fam}*"):
            if not hit.is_file():
                continue
            p = rel(hit)
            if any(m in p for m in permitted_markers):
                continue
            if any(m in p for m in capture_markers):
                leaked.append(p)
    require(not leaked, f"sealed CAPTURE artifacts present: {leaked[:5]}")
    permitted = sum(1 for fam in sealed for h in ROOT.glob(f"**/*{fam}*")
                    if h.is_file() and any(m in rel(h) for m in permitted_markers))
    print(f"  no capture artifact for any sealed family "
          f"({permitted} source-only files allowed){'':<3}: OK", flush=True)
    return load_json(ACCEPTANCE_1A)


def execute() -> None:
    require(ROOT.name == "vlsi_fault_detection_v2", f"place script in project root, not {ROOT}")
    outputs = (PREDICTIONS, HYPOTHESIS, EVIDENCE, FALSIFIERS, SCORING_RULE,
               PREFLIGHT, REPORT, MANIFEST, AUDIT)
    require(not any(p.exists() for p in outputs),
            f"frozen Stage {STAGE} output exists; use --status")

    acceptance = verify_inputs()
    created = now()

    print("\nBOUNDING HYPOTHESIS", flush=True)
    hypothesis = {
        "hypothesis_version": "CIRCUITSAGE-HMAC-V2.2-BOUNDING-HYPOTHESIS-12C2H-v1",
        "stage": STAGE, "status": "FROZEN", "created_at": created,
        "statement": (
            "Under a fixed test scheme, the all-injected exact-site rate achievable by "
            "behaviour-signature fault localisation is bounded by the fraction of faults "
            "whose observable response is unique. That fraction is a property of the "
            "circuit's structure and observability, not of the model."),
        "corollary": (
            "No function of the observed response can separate faults whose observed "
            "responses are identical. Improvements must come from the test scheme or "
            "additional observation points, not from model capacity."),
        "supporting_falsifications": [
            {"stage": "12C-2C", "formulation": "retrieval, uniform random negatives",
             "outcome": "no transfer to unseen circuit"},
            {"stage": "12C-2E", "formulation": "retrieval, topology-stratified hard negatives",
             "outcome": "no transfer to unseen circuit"},
            {"stage": "12C-2F", "formulation": "measured against non-learning comparator",
             "outcome": "comparator MRR 0.6745 vs trained 0.0037 (182x)"},
            {"stage": "12C-2G", "formulation": "graph-constrained reranking within collision sets",
             "outcome": "calibration lift exactly 1.00x over random-within-set"},
        ],
        "not_claimed": [
            "that no model could ever exceed this bound",
            "that the bound is proven rather than strongly evidenced",
            "any result on circuits that have not been captured",
        ],
        "status_of_generalization": "NOT ESTABLISHED",
    }
    for s in hypothesis["supporting_falsifications"]:
        print(f"  {s['stage']:<8} {s['outcome']}", flush=True)

    print("\nPREDICTIONS (frozen before capture exists)", flush=True)
    for p in PREDICTED:
        lo, hi = p["predicted_exact_site_band"]
        print(f"  {p['family_id']:<18} {p['predicted_circuit_class']:<24} "
              f"exact_site [{lo:.2f}, {hi:.2f}]  floor={p['predicted_meets_floor_0_15']:<3} "
              f"conf={p['confidence']}", flush=True)

    predictions = {
        "prediction_version": "CIRCUITSAGE-HMAC-V2.2-SEALED-PREDICTIONS-12C2H-v1",
        "stage": STAGE, "status": "FROZEN", "created_at": created,
        "frozen_before_capture": True,
        "sealed_data_read": False,
        "per_circuit_floor_threshold": 0.15,
        "predicted_overall_acceptance": "NOT MET",
        "predicted_overall_basis": (
            "the per-circuit floor applies to BOTH independent test circuits; a predicted "
            "ibex_cpu shortfall is sufficient for overall NOT MET regardless of chacha"),
        "predictions": PREDICTED,
    }

    scoring = {
        "scoring_version": "CIRCUITSAGE-HMAC-V2.2-PREDICTION-SCORING-12C2H-v1",
        "stage": STAGE, "status": "FROZEN", "created_at": created,
        "rule": (
            "After Stage 12C-3A capture, each prediction is scored by comparing the "
            "measured all_injected_exact_site_rate against the frozen band and the frozen "
            "floor outcome. Scoring must consult this file, not memory."),
        "outcome_labels": {
            "CONFIRMED": "measured value inside the predicted band and floor outcome correct",
            "DIRECTIONALLY_CORRECT": "floor outcome correct but measured value outside band",
            "FALSIFIED": "floor outcome incorrect",
        },
        "prohibited_after_capture": [
            "widening a predicted band",
            "reclassifying a circuit to fit its measurement",
            "adding a prediction for an already-captured circuit",
            "retraining, threshold changes or reselection",
        ],
        "one_shot_rule": acceptance["one_shot_rule"],
        "honest_reporting_requirement": (
            "a FALSIFIED prediction must be reported as prominently as a CONFIRMED one"),
    }

    preflight = {
        "preflight_version": "CIRCUITSAGE-HMAC-V2.2-PREDICTION-PREFLIGHT-12C2H-v1",
        "stage": STAGE, "status": "PASS", "created_at": created,
        "capture_performed": False, "simulation_calls": 0, "synthesis_calls": 0,
        "training_calls": 0, "selection_calls": 0,
        "independent_test_access": 0, "holdout_access": 0, "validation_access": 0,
        "sealed_artifacts_present_in_tree": 0,
        "acceptance_criteria_invented": False,
        "acceptance_evaluation": "NOT AUTHORIZED",
    }

    ev_fields = list(OBSERVED[0].keys())
    fal_rows = [{
        "family_id": p["family_id"], "partition": p["partition"],
        "predicted_class": p["predicted_circuit_class"],
        "predicted_exact_site_low": p["predicted_exact_site_band"][0],
        "predicted_exact_site_high": p["predicted_exact_site_band"][1],
        "predicted_meets_floor": p["predicted_meets_floor_0_15"],
        "confidence": p["confidence"],
        "falsified_if": p["falsified_if"],
    } for p in PREDICTED]

    obs_table = "\n".join(
        f"| `{o['family_id']}` | {o['circuit_class']} | {o['signature_uniqueness']:.4f} | "
        f"{o['detection_recall']:.4f} | {o['exact_site_reranked']:.4f} | "
        f"{o['meets_floor_0_15']} |" for o in OBSERVED)
    pred_table = "\n".join(
        f"| `{p['family_id']}` | {p['predicted_circuit_class']} | "
        f"[{p['predicted_exact_site_band'][0]:.2f}, {p['predicted_exact_site_band'][1]:.2f}] | "
        f"{p['predicted_meets_floor_0_15']} | {p['confidence']} |" for p in PREDICTED)

    report = f"""# Stage {STAGE} — Sealed-Circuit Prediction Freeze

**Status: PASS / FROZEN — predictions only, no capture, no sealed data read.**

## Why this stage exists and why it is time-critical

The frozen acceptance contract permits **one** evaluation on the sealed
circuits. Once capture happens, predictions can no longer be made in advance —
only descriptions after the fact. Freezing predictions now converts the sealed
capture from a measurement into a **test of a stated hypothesis**. This option
is available exactly once and expires when Stage 12C-3A begins.

## Bounding hypothesis

> Under a fixed test scheme, the all-injected exact-site rate achievable by
> behaviour-signature fault localisation is bounded by the fraction of faults
> whose observable response is unique — a property of circuit structure and
> observability, not of the model.

Three independent learned formulations were tested against this bound and none
exceeded it on an unseen circuit:

| stage | formulation | outcome |
|---|---|---|
| 12C-2C | retrieval, random negatives | no transfer |
| 12C-2E | retrieval, hard negatives | no transfer |
| 12C-2F | vs non-learning comparator | comparator 182x better |
| 12C-2G | graph reranking within collision sets | lift exactly 1.00x |

## Measured evidence (development partitions)

| family | class | signature uniqueness | detection | exact-site | meets floor |
|---|---|---|---|---|---|
{obs_table}

## Predictions — frozen before any sealed circuit is captured

| circuit | predicted class | predicted exact-site | meets floor 0.15 | confidence |
|---|---|---|---|---|
{pred_table}

**Predicted overall acceptance: NOT MET**, because the per-circuit floor applies
to *both* independent test circuits and `ibex_cpu` is predicted to fall short.

### What would falsify each

- **`ibex_cpu`** — exact-site ≥ 0.15. Would show CPU-class ambiguity is not
  governing.
- **`secworks_chacha`** — exact-site < 0.15. Would show crypto-datapath
  circuits are not uniformly favourable.
- **`serv_cpu`** — exact-site ≥ 0.15. Would reframe the bounding variable from
  circuit class to datapath reuse. Confidence is deliberately **LOW** here.

## What this stage does not do

No capture, synthesis, simulation, training, or selection. Zero reads of any
INDEPENDENT_CIRCUIT_TEST or GENERALIZATION_HOLDOUT artifact — verified by
scanning the tree for sealed-family files and finding none. Independent
generalization remains **NOT ESTABLISHED**. Future hybrid brand remains
**{FUTURE_BRAND}**.

## Next gate

**Stage 12C-3A** — independent test capture. Scoring after capture must consult
the frozen scoring rule in this stage, and a falsified prediction must be
reported as prominently as a confirmed one.
"""

    frozen_write(PREDICTIONS, canonical_json(predictions))
    frozen_write(HYPOTHESIS, canonical_json(hypothesis))
    frozen_write(EVIDENCE, csv_bytes(OBSERVED, ev_fields))
    frozen_write(FALSIFIERS, csv_bytes(fal_rows, list(fal_rows[0].keys())))
    frozen_write(SCORING_RULE, canonical_json(scoring))
    frozen_write(PREFLIGHT, canonical_json(preflight))
    frozen_write(REPORT, report.encode())

    stage_outputs = (PREDICTIONS, HYPOTHESIS, EVIDENCE, FALSIFIERS,
                     SCORING_RULE, PREFLIGHT, REPORT)
    manifest = {
        "manifest_version": "CIRCUITSAGE-HMAC-V2.2-PREDICTION-FREEZE-MANIFEST-12C2H-v1",
        "stage": STAGE, "status": "PASS", "created_at": created,
        "stage_source": record(Path(__file__).resolve()),
        "frozen_inputs": {rel(p): record(p) for p in PINNED},
        "outputs": {rel(p): record(p) for p in stage_outputs},
        "predictions_frozen": len(PREDICTED),
        "capture_performed": False,
        "independent_test_access": 0, "holdout_access": 0, "validation_access": 0,
    }
    frozen_write(MANIFEST, canonical_json(manifest))

    audit = {
        "audit_version": "CIRCUITSAGE-HMAC-V2.2-PREDICTION-FREEZE-12C2H-v1",
        "stage": STAGE, "status": "PASS",
        "frozen_before_capture": True,
        "sealed_data_read": False,
        "predictions_frozen": len(PREDICTED),
        "predicted_floor_outcomes": {p["family_id"]: p["predicted_meets_floor_0_15"]
                                     for p in PREDICTED},
        "predicted_exact_site_bands": {p["family_id"]: p["predicted_exact_site_band"]
                                       for p in PREDICTED},
        "prediction_confidence": {p["family_id"]: p["confidence"] for p in PREDICTED},
        "predicted_overall_acceptance": "NOT MET",
        "bounding_hypothesis_record": record(HYPOTHESIS),
        "predictions_record": record(PREDICTIONS),
        "scoring_rule_record": record(SCORING_RULE),
        "falsification_conditions_record": record(FALSIFIERS),
        "manifest_record": record(MANIFEST),
        "capture_performed": False,
        "selection_performed": False,
        "acceptance_evaluation": "NOT AUTHORIZED",
        "independent_test_validation_holdout_access": [0, 0, 0],
        "independent_generalization": "NOT ESTABLISHED",
        "future_combined_model_brand": FUTURE_BRAND,
        "next_gate": "STAGE 12C-3A — INDEPENDENT TEST CAPTURE",
    }
    frozen_write(AUDIT, canonical_json(audit))

    print(f"\n{'Stage':<52}: {STAGE} — SEALED-CIRCUIT PREDICTION FREEZE")
    print(f"{'Status':<52}: PASS / FROZEN")
    print(f"{'Predictions frozen before capture':<52}: {len(PREDICTED)}")
    print(f"{'Sealed data read':<52}: NO")
    print(f"{'Predicted overall acceptance':<52}: NOT MET")
    print(f"{'TEST / HOLDOUT / VALIDATION access':<52}: 0 / 0 / 0")
    print(f"{'Audit SHA':<52}: {sha256(AUDIT)}")
    print(f"{'Next gate':<52}: STAGE 12C-3A — INDEPENDENT TEST CAPTURE")


def status() -> None:
    print(f"STAGE {STAGE} — PREDICTION FREEZE STATUS")
    if not (MANIFEST.is_file() and AUDIT.is_file()):
        print("Status                    : NOT FROZEN")
        return
    a = load_json(AUDIT)
    print("Status                    : PASS / FROZEN")
    print(f"Frozen before capture     : {a['frozen_before_capture']}")
    print(f"Predicted floor outcomes  : {a['predicted_floor_outcomes']}")
    print(f"Predicted bands           : {a['predicted_exact_site_bands']}")
    print(f"Confidence                : {a['prediction_confidence']}")
    print(f"Predicted acceptance      : {a['predicted_overall_acceptance']}")
    print(f"Audit SHA                 : {sha256(AUDIT)}")


def self_test() -> None:
    for p in PREDICTED:
        lo, hi = p["predicted_exact_site_band"]
        require(0.0 <= lo < hi <= 1.0, f"valid band for {p['family_id']}")
        require(p["confidence"] in ("LOW", "MEDIUM", "HIGH"), "confidence label")
        require(p["predicted_meets_floor_0_15"] in ("YES", "NO"), "floor label")
        require(len(p["reasoning"]) > 120, f"substantive reasoning for {p['family_id']}")
        require(p["falsified_if"], f"falsifier stated for {p['family_id']}")
        meets = p["predicted_meets_floor_0_15"] == "YES"
        require(meets == (hi >= 0.15), f"band consistent with floor call for {p['family_id']}")
    require(len({p["family_id"] for p in PREDICTED}) == len(PREDICTED), "unique families")
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
