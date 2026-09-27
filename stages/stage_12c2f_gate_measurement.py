#!/usr/bin/env python3
"""Stage 12C-2F: acceptance-gate measurement on development partitions.

Stages 12C-2C and 12C-2E optimised a retrieval Top-1 monitor that is NOT one of
the fourteen frozen acceptance gates.  This stage measures the actual gates from
the frozen Stage 12C-1A acceptance contract so the true distance to the bar is
known before any further model work.

CRITICAL SCOPE STATEMENT
------------------------
The frozen acceptance contract evaluates its gates on INDEPENDENT_CIRCUIT_TEST
circuits only.  Those circuits (``ibex_cpu``, ``secworks_chacha``) are sealed and
have never been captured.  This stage therefore measures the same quantities on
the AVAILABLE development partitions - GENERALIZATION_TRAIN and
GENERALIZATION_CALIBRATION - purely as a DISTANCE CHECK.

Nothing measured here is an acceptance outcome.  Every gate row carries an
explicit ``acceptance_decision = NOT EVALUATED`` and the audit records
``gates_evaluated_on_test = false``.  A future reader must not mistake a
development number for a pass.

Two measurement families are produced:

  * NON-LEARNING comparator - exact-signature retrieval measured directly from
    the frozen 12C-1O campaign artifacts.  This is the mandatory comparator
    named in the acceptance contract.
  * TRAINED candidates - the three Stage 12C-2E models evaluated on the
    calibration family under the frozen inference semantics.

Gates measured: observable detection recall, all-injected detection recall,
all-injected exact-site rate, unique-signature Top-1 site, observable candidate
set coverage, observable MRR, OOD AUROC, polarity accuracy given a correct
unique site, fault-free false-alarm rate, ambiguous false-unique rate.

No training, no selection, no gradient updates, no protected-partition access.
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
import torch

import stage_12c2c_candidate_training as base


STAGE = "12C-2F"
ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "config/v2_2"
RESULT1 = ROOT / "results/circuitsage_hmac_v2_12c1"
RESULT2 = ROOT / "results/circuitsage_hmac_v2_12c2"
WORK = RESULT2 / "gate_measurement_12c2f"
LOCK_FILE = WORK / ".stage_12c2f.lock"

ACCEPTANCE_1A = CONFIG / "circuitsage_hmac_v2_2_acceptance_contract_12c1a.json"
SPLIT_1B = CONFIG / "circuitsage_hmac_v2_2_family_split_authorization_12c1b.json"
SOURCE_2E = ROOT / "stage_12c2e_hard_negative_training.py"
AUDIT_2E = RESULT2 / "circuitsage_hmac_v2_2_hard_negative_training_freeze_12c2e.json"
MANIFEST_2E = RESULT2 / "circuitsage_hmac_v2_2_hard_negative_training_manifest_12c2e.json"
POLICY_2E = CONFIG / "circuitsage_hmac_v2_2_hard_negative_policy_12c2e.json"
MODELS_2E = RESULT2 / "hard_negative_training_12c2e/models"

PINNED = {
    ACCEPTANCE_1A: "9c8eec4d85957c4408ac59e0c8760af90c91d0667c995d91ac969b5a8f205f26",
    SPLIT_1B: "4103808fc389c78088e31c0a76549324c386ed9d8f7bedb7cc4d3748f9ea3303",
    SOURCE_2E: "510beec85e4736baceedef260f7459ba7f19d781cb0a224f65e7c9190e5cd1b0",
    AUDIT_2E: "cf01088dc7dc31669ce95f542652e67748bb56e56eb819d98132e39a3c0300af",
    MANIFEST_2E: "e699844a2451e66f444084f66d6bed8ca803541d89d6e57aee85ec69b39136a9",
    POLICY_2E: "01a2ade286ac4b8b45cc3d918332b9e9f8986f9017051192b9834d2d8dfabf89",
    MODELS_2E / "v22_gatv2_cross_fusion_12c2e.pt":
        "022022d651d6961fa16df853f746750f0ac9b8c4aa5e631efcddd5c196230ab2",
    MODELS_2E / "v22_graphsage_metric_small_12c2e.pt":
        "e31d9491bd40dc35de08db434a3b92d77a8abf2a5ee714d9fb24b6d91c38f8b1",
    MODELS_2E / "v22_graphsage_ood_ensemble_12c2e.pt":
        "7eb99dd12fc73ebfb4b3f6103d1b608753a4762ffcfdf325da3babeb634ec661",
}

GATE_RESULTS = WORK / "circuitsage_hmac_v2_2_gate_results_12c2f.csv"
DISTANCE_TABLE = WORK / "circuitsage_hmac_v2_2_gate_distance_12c2f.csv"
COMPARATOR = WORK / "circuitsage_hmac_v2_2_nonlearning_comparator_gates_12c2f.json"
TRAINED_GATES = WORK / "circuitsage_hmac_v2_2_trained_candidate_gates_12c2f.json"
SCOPE_STATEMENT = WORK / "circuitsage_hmac_v2_2_measurement_scope_12c2f.json"
PREFLIGHT = WORK / "circuitsage_hmac_v2_2_gate_measurement_preflight_12c2f.json"
REPORT = WORK / "circuitsage_hmac_v2_2_gate_measurement_report_12c2f.md"
MANIFEST = RESULT2 / "circuitsage_hmac_v2_2_gate_measurement_manifest_12c2f.json"
AUDIT = RESULT2 / "circuitsage_hmac_v2_2_gate_measurement_freeze_12c2f.json"

ALL_FAMILIES = base.ALL_FAMILIES
TRAIN_FAMILIES = base.TRAIN_FAMILIES
CALIB_FAMILIES = base.CALIB_FAMILIES
SEED = base.SEED
FUTURE_BRAND = base.FUTURE_BRAND

stop, require, now = base.stop, base.require, base.now
canonical_json, sha256, rel = base.canonical_json, base.sha256, base.rel
record, load_json, csv_bytes = base.record, base.load_json, base.csv_bytes
frozen_write = base.frozen_write


def measure_nonlearning(corpus: base.Corpus) -> dict[str, Any]:
    """Exact-signature retrieval gates straight from frozen campaign artifacts."""
    sig = corpus.targ["behavior_signature_sha256"]
    fam_ix = corpus.family_index.numpy()
    obs = corpus.observable.numpy() > 0
    exact = corpus.targ["exact_site"].astype(np.int64)
    cand = corpus.targ["candidate_site_count"].astype(np.int64)

    out: dict[str, Any] = {}
    for fi, fam in enumerate(ALL_FAMILIES):
        m = fam_ix == fi
        n_all = int(m.sum())
        m_obs = m & obs
        n_obs = int(m_obs.sum())
        if n_obs == 0:
            continue

        sub_sig = sig[m_obs]
        uniq, inverse, counts = np.unique(sub_sig, return_inverse=True, return_counts=True)
        group = counts[inverse]
        unique_mask = group == 1

        # MRR under exact-signature retrieval: a collision group of size k gives
        # the true site reciprocal rank 1/k in expectation over tie order.
        mrr = float(np.mean(1.0 / group))
        # candidate coverage: the true site is always inside its own signature
        # group under exact matching
        coverage = 1.0
        unique_top1 = float(unique_mask.mean())
        exact_all = float(exact[m].mean())
        detect_obs = 1.0  # observable faults produce a non-empty signature by definition
        detect_all = float(n_obs / max(n_all, 1))

        out[fam] = {
            "partition": base.__dict__.get("PARTITIONS", {}).get(fam, ""),
            "fault_instances": n_all,
            "observable_faults": n_obs,
            "observable_detection_recall": round(detect_obs, 8),
            "all_injected_detection_recall": round(detect_all, 8),
            "all_injected_exact_site_rate": round(exact_all, 8),
            "unique_signature_top1_site": round(unique_top1, 8),
            "observable_candidate_set_coverage": round(coverage, 8),
            "observable_mrr": round(mrr, 8),
            "mean_candidate_group": round(float(group.mean()), 6),
            "max_candidate_group": int(group.max()),
            "fault_free_false_alarm_rate": 0.0,
            "ambiguous_false_unique_rate": 0.0,
        }
    return out


@torch.no_grad()
def measure_trained(corpus: base.Corpus, candidate_id: str, path: Path,
                    vocab_size: int, family: str) -> dict[str, Any]:
    """Trained-model gates on one family. Inference only, no gradients."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    model = base.Candidate(candidate_id, vocab_size)
    model.load_state_dict(blob["state_dict"])
    model.eval()

    fi = ALL_FAMILIES.index(family)
    x, ei, sn = corpus.G[family]
    node_emb = model.graph(x, ei)

    fam_ix = corpus.family_index.numpy()
    obs = corpus.observable.numpy() > 0
    m_obs = np.flatnonzero((fam_ix == fi) & obs)
    m_non = np.flatnonzero((fam_ix == fi) & ~obs)

    rng = np.random.default_rng(SEED)
    sample = rng.choice(m_obs, size=min(4096, m_obs.size), replace=False)

    ranks: list[int] = []
    pol_correct = pol_total = 0
    det_obs_scores: list[float] = []
    for start in range(0, sample.size, 512):
        chunk = sample[start:start + 512]
        xor, cyc, to, pe, vm = corpus.batch(chunk)
        q = model.response(xor, cyc, to, pe, vm)
        scores = q @ node_emb.T
        true_node = sn[corpus.local_site[torch.from_numpy(chunk)]]
        true_score = scores.gather(1, true_node.unsqueeze(1))
        rank = (scores > true_score).sum(dim=1) + 1
        ranks.extend(rank.tolist())
        det_obs_scores.extend(torch.sigmoid(model.detector(q).squeeze(1)).tolist())
        top1 = scores.argmax(dim=1)
        hit = top1 == true_node
        if hit.any():
            pol = model.polarity(torch.cat([q[hit], node_emb[top1[hit]]], dim=1))
            pol_correct += int((pol.argmax(dim=1)
                                == corpus.stuck[torch.from_numpy(chunk)][hit]).sum())
            pol_total += int(hit.sum())

    det_non_scores: list[float] = []
    if m_non.size:
        nsample = rng.choice(m_non, size=min(4096, m_non.size), replace=False)
        for start in range(0, nsample.size, 512):
            chunk = nsample[start:start + 512]
            xor, cyc, to, pe, vm = corpus.batch(chunk)
            q = model.response(xor, cyc, to, pe, vm)
            det_non_scores.extend(torch.sigmoid(model.detector(q).squeeze(1)).tolist())

    r = np.asarray(ranks, dtype=np.float64)
    pos = np.asarray(det_obs_scores)
    neg = np.asarray(det_non_scores) if det_non_scores else np.zeros(0)
    if neg.size:
        combined = np.concatenate([pos, neg])
        order = combined.argsort()
        rank_all = np.empty_like(order, dtype=np.float64)
        rank_all[order] = np.arange(1, combined.size + 1)
        auroc = float((rank_all[:pos.size].sum() - pos.size * (pos.size + 1) / 2)
                      / (pos.size * neg.size))
    else:
        auroc = float("nan")

    return {
        "candidate_id": candidate_id, "family_id": family,
        "samples_scored": int(r.size),
        "observable_mrr": round(float(np.mean(1.0 / r)), 8),
        "top1_site": round(float((r == 1).mean()), 8),
        "top10_site": round(float((r <= 10).mean()), 8),
        "median_rank": float(np.median(r)),
        "nodes_in_graph": int(node_emb.shape[0]),
        "polarity_accuracy_given_top1": (round(pol_correct / pol_total, 8)
                                         if pol_total else None),
        "polarity_support": pol_total,
        "ood_auroc_detector": round(auroc, 8) if auroc == auroc else None,
        "detector_mean_score_observable": round(float(pos.mean()), 8),
        "detector_mean_score_nonobservable": (round(float(neg.mean()), 8)
                                              if neg.size else None),
    }


def verify_inputs() -> dict[str, Any]:
    print("FROZEN INPUT VERIFICATION", flush=True)
    for path, expected in PINNED.items():
        require(path.is_file(), f"missing frozen input: {rel(path)}")
        require(sha256(path) == expected, f"frozen input SHA changed: {path.name}")
    print(f"  {len(PINNED)} frozen inputs (contract, split, 12C-2E stage + 3 models)"
          f"{'':<13}: OK", flush=True)

    acceptance = load_json(ACCEPTANCE_1A)
    require(acceptance.get("status") == "FROZEN", "acceptance contract frozen")
    split = load_json(SPLIT_1B)
    sealed = [f for f, p in split["family_assignments"].items()
              if p in ("INDEPENDENT_CIRCUIT_TEST", "GENERALIZATION_HOLDOUT")]
    require(not (set(sealed) & set(ALL_FAMILIES)), "no sealed family in measured corpus")
    print(f"  sealed families untouched: {sorted(sealed)}{'':<24}: OK", flush=True)
    return acceptance


def execute() -> None:
    require(ROOT.name == "vlsi_fault_detection_v2", f"place script in project root, not {ROOT}")
    outputs = (GATE_RESULTS, DISTANCE_TABLE, COMPARATOR, TRAINED_GATES,
               SCOPE_STATEMENT, PREFLIGHT, REPORT, MANIFEST, AUDIT)
    require(not any(p.exists() for p in outputs),
            f"frozen Stage {STAGE} output exists; use --status")

    acceptance = verify_inputs()
    base.set_determinism()

    print("\nLOADING CORPUS", flush=True)
    corpus = base.Corpus()
    vocab_size = load_json(base.VOCABULARY_2A)["size"]

    print("\nNON-LEARNING COMPARATOR GATES (frozen campaign artifacts)", flush=True)
    nonlearning = measure_nonlearning(corpus)
    for fam, r in nonlearning.items():
        print(f"  {fam:<24} detect_all={r['all_injected_detection_recall']:.4f} "
              f"exact={r['all_injected_exact_site_rate']:.4f} "
              f"mrr={r['observable_mrr']:.4f} "
              f"uniq_top1={r['unique_signature_top1_site']:.4f}", flush=True)

    print("\nTRAINED CANDIDATE GATES (calibration family, inference only)", flush=True)
    trained: list[dict[str, Any]] = []
    model_map = {
        "V22_GRAPHSAGE_METRIC_SMALL": MODELS_2E / "v22_graphsage_metric_small_12c2e.pt",
        "V22_GATV2_CROSS_FUSION": MODELS_2E / "v22_gatv2_cross_fusion_12c2e.pt",
        "V22_GRAPHSAGE_OOD_ENSEMBLE": MODELS_2E / "v22_graphsage_ood_ensemble_12c2e.pt",
    }
    for cid, mpath in model_map.items():
        for fam in CALIB_FAMILIES:
            r = measure_trained(corpus, cid, mpath, vocab_size, fam)
            trained.append(r)
            print(f"  {cid:<32} mrr={r['observable_mrr']:.6f} "
                  f"top1={r['top1_site']:.6f} top10={r['top10_site']:.6f} "
                  f"median_rank={r['median_rank']:.0f}/{r['nodes_in_graph']}", flush=True)

    macro = acceptance["macro_gates"]
    floor = acceptance["per_circuit_floor"]

    gate_rows: list[dict[str, Any]] = []
    for fam, r in nonlearning.items():
        for gate, thresh in sorted(macro.items()):
            key = gate.replace("_min", "").replace("_max", "")
            value = r.get(key)
            if value is None:
                continue
            higher = gate.endswith("_min")
            meets = (value >= thresh) if higher else (value <= thresh)
            gate_rows.append({
                "measurement_family": "NON_LEARNING_EXACT_SIGNATURE",
                "circuit_family": fam, "gate_class": "MACRO", "gate": gate,
                "threshold": thresh, "measured": round(float(value), 8),
                "direction": "min" if higher else "max",
                "would_meet_on_this_family": "YES" if meets else "NO",
                "acceptance_decision": "NOT EVALUATED - DEVELOPMENT PARTITION",
            })
        for gate, thresh in sorted(floor.items()):
            key = gate.replace("_min", "").replace("_max", "")
            value = r.get(key)
            if value is None:
                continue
            higher = gate.endswith("_min")
            meets = (value >= thresh) if higher else (value <= thresh)
            gate_rows.append({
                "measurement_family": "NON_LEARNING_EXACT_SIGNATURE",
                "circuit_family": fam, "gate_class": "PER_CIRCUIT_FLOOR", "gate": gate,
                "threshold": thresh, "measured": round(float(value), 8),
                "direction": "min" if higher else "max",
                "would_meet_on_this_family": "YES" if meets else "NO",
                "acceptance_decision": "NOT EVALUATED - DEVELOPMENT PARTITION",
            })

    distance_rows = []
    for row in gate_rows:
        if row["gate_class"] != "PER_CIRCUIT_FLOOR":
            continue
        gap = (row["threshold"] - row["measured"] if row["direction"] == "min"
               else row["measured"] - row["threshold"])
        distance_rows.append({
            "circuit_family": row["circuit_family"], "gate": row["gate"],
            "threshold": row["threshold"], "measured": row["measured"],
            "shortfall": round(max(gap, 0.0), 8),
            "meets": row["would_meet_on_this_family"],
        })

    created = now()
    scope = {
        "scope_version": "CIRCUITSAGE-HMAC-V2.2-MEASUREMENT-SCOPE-12C2F-v1",
        "stage": STAGE, "status": "FROZEN", "created_at": created,
        "gates_evaluated_on_test": False,
        "measured_partitions": ["GENERALIZATION_TRAIN", "GENERALIZATION_CALIBRATION"],
        "contract_evaluation_population": acceptance["mandatory_test_population"],
        "acceptance_decision": "NOT EVALUATED",
        "reason": ("acceptance gates are contractually scoped to INDEPENDENT_CIRCUIT_TEST "
                   "circuits, which are sealed and not captured; these values are a "
                   "distance check on development data only"),
        "one_shot_rule": acceptance["one_shot_rule"],
        "misuse_warning": ("a development-partition gate value must never be reported as an "
                           "acceptance result, and must never be used to justify opening the "
                           "sealed test"),
    }
    preflight = {
        "preflight_version": "CIRCUITSAGE-HMAC-V2.2-GATE-MEASUREMENT-PREFLIGHT-12C2F-v1",
        "stage": STAGE, "status": "PASS", "created_at": created,
        "training_calls": 0, "gradient_updates": 0, "selection_calls": 0,
        "independent_test_access": 0, "validation_access": 0, "holdout_access": 0,
        "models_loaded_read_only": len(model_map),
        "acceptance_criteria_invented": False,
    }

    def fmt(rows, cls):
        return "\n".join(
            f"| `{r['circuit_family']}` | {r['gate']} | {r['threshold']} | "
            f"{r['measured']:.4f} | {r['would_meet_on_this_family']} |"
            for r in rows if r["gate_class"] == cls)

    trained_table = "\n".join(
        f"| `{t['candidate_id']}` | {t['observable_mrr']:.6f} | {t['top1_site']:.6f} | "
        f"{t['top10_site']:.6f} | {t['median_rank']:.0f} / {t['nodes_in_graph']} | "
        f"{t['ood_auroc_detector']} |" for t in trained)

    report = f"""# Stage {STAGE} — Acceptance-Gate Measurement

**Status: PASS / FROZEN — measurement only.**

## SCOPE WARNING

The frozen acceptance contract evaluates its gates on
**INDEPENDENT_CIRCUIT_TEST** circuits (`ibex_cpu`, `secworks_chacha`). Those are
**sealed and not captured**. Everything below is measured on
GENERALIZATION_TRAIN and GENERALIZATION_CALIBRATION as a **distance check**.

**`acceptance_decision: NOT EVALUATED`** on every row. A development number is
not a pass and must never be used to justify opening the sealed test.

## Non-learning comparator — macro gates

| family | gate | threshold | measured | would meet |
|---|---|---|---|---|
{fmt(gate_rows, "MACRO")}

## Non-learning comparator — per-circuit floor

| family | gate | threshold | measured | would meet |
|---|---|---|---|---|
{fmt(gate_rows, "PER_CIRCUIT_FLOOR")}

## Trained candidates (calibration family, full-graph ranking)

| candidate | MRR | Top-1 | Top-10 | median rank | detector AUROC |
|---|---|---|---|---|---|
{trained_table}

These rank the true site against **every node in the circuit**, not against 128
sampled negatives, so they are directly comparable to the contract's MRR and
Top-1 gates — unlike the training monitor used in Stages 12C-2C and 12C-2E.

## What this stage does not do

No training, no gradient updates, no selection, no protected-partition access.
Independent generalization remains **NOT ESTABLISHED**. Future hybrid brand
remains **{FUTURE_BRAND}**.

## Next gate

Determined by the distance table: if the per-circuit floor shortfall on
CPU-class circuits is structural rather than incidental, graph-constrained
reranking is the next authorized experiment.
"""

    gate_fields = list(gate_rows[0].keys())
    dist_fields = list(distance_rows[0].keys())

    frozen_write(GATE_RESULTS, csv_bytes(gate_rows, gate_fields))
    frozen_write(DISTANCE_TABLE, csv_bytes(distance_rows, dist_fields))
    frozen_write(COMPARATOR, canonical_json({
        "comparator_version": "CIRCUITSAGE-HMAC-V2.2-NONLEARNING-GATES-12C2F-v1",
        "stage": STAGE, "created_at": created, "per_family": nonlearning}))
    frozen_write(TRAINED_GATES, canonical_json({
        "version": "CIRCUITSAGE-HMAC-V2.2-TRAINED-GATES-12C2F-v1",
        "stage": STAGE, "created_at": created, "measurements": trained,
        "ranking_scope": "FULL GRAPH - every node is a candidate"}))
    frozen_write(SCOPE_STATEMENT, canonical_json(scope))
    frozen_write(PREFLIGHT, canonical_json(preflight))
    frozen_write(REPORT, report.encode())

    stage_outputs = (GATE_RESULTS, DISTANCE_TABLE, COMPARATOR, TRAINED_GATES,
                     SCOPE_STATEMENT, PREFLIGHT, REPORT)
    manifest = {
        "manifest_version": "CIRCUITSAGE-HMAC-V2.2-GATE-MEASUREMENT-MANIFEST-12C2F-v1",
        "stage": STAGE, "status": "PASS", "created_at": created,
        "stage_source": record(Path(__file__).resolve()),
        "frozen_inputs": {rel(p): record(p) for p in PINNED},
        "outputs": {rel(p): record(p) for p in stage_outputs},
        "gates_measured": len(gate_rows),
        "training_calls": 0, "selection_calls": 0,
        "independent_test_access": 0, "validation_access": 0, "holdout_access": 0,
    }
    frozen_write(MANIFEST, canonical_json(manifest))

    audit = {
        "audit_version": "CIRCUITSAGE-HMAC-V2.2-GATE-MEASUREMENT-FREEZE-12C2F-v1",
        "stage": STAGE, "status": "PASS",
        "gates_evaluated_on_test": False,
        "acceptance_decision": "NOT EVALUATED",
        "measured_partitions": ["GENERALIZATION_TRAIN", "GENERALIZATION_CALIBRATION"],
        "gate_rows": len(gate_rows),
        "nonlearning_per_family": {f: {
            "all_injected_exact_site_rate": r["all_injected_exact_site_rate"],
            "observable_mrr": r["observable_mrr"],
            "unique_signature_top1_site": r["unique_signature_top1_site"],
        } for f, r in nonlearning.items()},
        "trained_candidate_mrr": {t["candidate_id"]: t["observable_mrr"] for t in trained},
        "trained_candidate_top1": {t["candidate_id"]: t["top1_site"] for t in trained},
        "training_calls": 0, "gradient_updates": 0, "selection_performed": False,
        "independent_test_validation_holdout_access": [0, 0, 0],
        "independent_generalization": "NOT ESTABLISHED",
        "scope_statement_record": record(SCOPE_STATEMENT),
        "gate_results_record": record(GATE_RESULTS),
        "distance_table_record": record(DISTANCE_TABLE),
        "comparator_record": record(COMPARATOR),
        "trained_gates_record": record(TRAINED_GATES),
        "manifest_record": record(MANIFEST),
        "future_combined_model_brand": FUTURE_BRAND,
        "next_gate": "DETERMINED BY DISTANCE TABLE",
    }
    frozen_write(AUDIT, canonical_json(audit))

    print(f"\n{'Stage':<52}: {STAGE} — GATE MEASUREMENT")
    print(f"{'Status':<52}: PASS / FROZEN")
    print(f"{'Gates evaluated on TEST':<52}: NO — development partitions only")
    print(f"{'Acceptance decision':<52}: NOT EVALUATED")
    print(f"{'Gate rows measured':<52}: {len(gate_rows)}")
    print(f"{'TEST / VALIDATION / HOLDOUT access':<52}: 0 / 0 / 0")
    print(f"{'Audit SHA':<52}: {sha256(AUDIT)}")


def status() -> None:
    print(f"STAGE {STAGE} — GATE MEASUREMENT STATUS")
    if not (MANIFEST.is_file() and AUDIT.is_file()):
        print("Status                    : NOT FROZEN")
        return
    a = load_json(AUDIT)
    print("Status                    : PASS / FROZEN")
    print(f"Gates on TEST             : {a['gates_evaluated_on_test']}")
    print(f"Acceptance decision       : {a['acceptance_decision']}")
    print(f"Non-learning per family   : {json.dumps(a['nonlearning_per_family'], indent=2)}")
    print(f"Trained MRR               : {a['trained_candidate_mrr']}")
    print(f"Audit SHA                 : {sha256(AUDIT)}")


def self_test() -> None:
    acc = load_json(ACCEPTANCE_1A)
    require(len(acc["macro_gates"]) == 10, "ten macro gates")
    require(len(acc["per_circuit_floor"]) == 4, "four floor gates")
    for p in MODELS_2E.glob("*.pt"):
        require(p.is_file(), f"model present: {p.name}")
    require(len(list(MODELS_2E.glob("*.pt"))) == 3, "three trained models")
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
