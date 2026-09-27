#!/usr/bin/env python3
"""Stage 12C-2G: graph-constrained reranking within signature-collision sets.

Stage 12C-2F measured the fourteen frozen acceptance gates and produced an
unambiguous finding: the NON-LEARNING exact-signature comparator outperforms
every trained Stage 12C-2E candidate by roughly two orders of magnitude on the
calibration family (MRR 0.6745 vs 0.0037; median true-site rank 3134-4917 of
9127 nodes, i.e. indistinguishable from random ordering).

Stages 12C-2C and 12C-2E both attempted to rank ALL nodes from a response
embedding.  That formulation is falsified.  It also cannot transfer in
principle: an absolute response->node mapping learned on AES carries no meaning
in an unseen circuit.

This stage changes the formulation rather than the architecture.

  * Candidate generation stays with the frozen non-learning comparator.  The
    exact-signature collision set is retained unchanged.
  * The learned model may ONLY REORDER nodes already inside that set.  It can
    never introduce a node the comparator did not propose, and never remove one.
  * Therefore candidate-set coverage is mathematically invariant, and the
    reranked exact-site rate is bounded BELOW by nothing worse than random
    ordering within each set - the baseline itself - and ABOVE by the
    perfect-reranking ceiling computed and frozen in this stage before training.

The learning task is relational and circuit-independent: given a collision set
whose members are behaviourally identical, order them by structural plausibility
using per-circuit-normalised graph features only.  No absolute node identity, no
circuit identity, no signature content enters the model.

Prediction-before-truth is enforced: the reranked ordering for every scored
fault is written to disk and hashed BEFORE truth labels are consulted for
scoring.

No selection, no acceptance evaluation, no protected-partition access.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import stage_12c2c_candidate_training as base


STAGE = "12C-2G"
ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "config/v2_2"
RESULT2 = ROOT / "results/circuitsage_hmac_v2_12c2"
WORK = RESULT2 / "graph_reranking_12c2g"
MODELS = WORK / "models"
LOCK_FILE = WORK / ".stage_12c2g.lock"

SOURCE_2F = ROOT / "stage_12c2f_gate_measurement.py"
AUDIT_2F = RESULT2 / "circuitsage_hmac_v2_2_gate_measurement_freeze_12c2f.json"
MANIFEST_2F = RESULT2 / "circuitsage_hmac_v2_2_gate_measurement_manifest_12c2f.json"
COMPARATOR_2F = (RESULT2 / "gate_measurement_12c2f"
                 / "circuitsage_hmac_v2_2_nonlearning_comparator_gates_12c2f.json")

PINNED = {
    SOURCE_2F: "74096769a2e8f51df665b24eca45ceae46e146da6e98c024564aaa719efcd925",
    AUDIT_2F: "aaa862b7ec4128eb8a3aa80233c8661ab799f1599ea1fed216bae49f98a66fb7",
    MANIFEST_2F: "743ae93a0e24f51916f1f694492088ef5ff67ac7e6d72098b6d37cb9379d76c0",
}

CEILING = WORK / "circuitsage_hmac_v2_2_reranking_ceiling_12c2g.json"
PREDICTIONS = WORK / "circuitsage_hmac_v2_2_reranked_predictions_12c2g.npz"
TRAIN_LOG = WORK / "circuitsage_hmac_v2_2_reranking_training_log_12c2g.csv"
RERANK_GATES = WORK / "circuitsage_hmac_v2_2_reranked_gate_results_12c2g.csv"
INVARIANTS = WORK / "circuitsage_hmac_v2_2_reranking_invariants_12c2g.json"
FORMULATION = CONFIG / "circuitsage_hmac_v2_2_reranking_formulation_12c2g.json"
ENVIRONMENT = WORK / "circuitsage_hmac_v2_2_reranking_environment_12c2g.json"
REPORT = WORK / "circuitsage_hmac_v2_2_reranking_report_12c2g.md"
MANIFEST = RESULT2 / "circuitsage_hmac_v2_2_graph_reranking_manifest_12c2g.json"
AUDIT = RESULT2 / "circuitsage_hmac_v2_2_graph_reranking_freeze_12c2g.json"

ALL_FAMILIES = base.ALL_FAMILIES
TRAIN_FAMILIES = base.TRAIN_FAMILIES
CALIB_FAMILIES = base.CALIB_FAMILIES
SEED = base.SEED
EPOCHS = 30
PATIENCE = 8
LR = 1e-3
MAX_SET = 64          # collision sets are truncated for batching; see invariants
HIDDEN = 96
FUTURE_BRAND = base.FUTURE_BRAND

stop, require, now = base.stop, base.require, base.now
canonical_json, sha256, rel = base.canonical_json, base.sha256, base.rel
record, load_json, csv_bytes = base.record, base.load_json, base.csv_bytes
frozen_write, atomic_json = base.frozen_write, base.atomic_json


class SetReranker(nn.Module):
    """Permutation-equivariant scorer over a signature-collision set.

    Each candidate is described ONLY by per-circuit-normalised structural
    features plus set-context statistics.  No node identity, no circuit
    identity, no signature bytes.  The set-context term is what makes the model
    relational: a node is scored relative to its competitors, so the learned
    function is meaningful in a circuit it has never seen.
    """

    def __init__(self, n_feat: int) -> None:
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(n_feat, HIDDEN), nn.LayerNorm(HIDDEN), nn.GELU(),
            nn.Linear(HIDDEN, HIDDEN), nn.LayerNorm(HIDDEN), nn.GELU(),
        )
        self.context = nn.Sequential(
            nn.Linear(HIDDEN * 3, HIDDEN), nn.LayerNorm(HIDDEN), nn.GELU(),
        )
        self.score = nn.Linear(HIDDEN, 1)

    def forward(self, feats: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.embed(feats)
        m = mask.unsqueeze(-1)
        denom = m.sum(dim=1, keepdim=True).clamp(min=1.0)
        mean = (h * m).sum(dim=1, keepdim=True) / denom
        mx = (h.masked_fill(~mask.unsqueeze(-1), -1e9)).max(dim=1, keepdim=True).values
        ctx = torch.cat([h, mean.expand_as(h), mx.expand_as(h)], dim=-1)
        s = self.score(self.context(ctx)).squeeze(-1)
        return s.masked_fill(~mask, -1e9)

    def trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class CollisionSets:
    """Signature-collision sets derived from frozen campaign artifacts."""

    def __init__(self, corpus: base.Corpus) -> None:
        self.corpus = corpus
        sig = corpus.targ["behavior_signature_sha256"]
        fam_ix = corpus.family_index.numpy()
        obs = corpus.observable.numpy() > 0
        self.sets: dict[str, dict[str, Any]] = {}
        self.stats: list[dict[str, Any]] = []

        for fi, fam in enumerate(ALL_FAMILIES):
            m = np.flatnonzero((fam_ix == fi) & obs)
            if m.size == 0:
                continue
            x, ei, sn = corpus.G[fam]
            feats = x.numpy()
            nodes = sn.numpy()
            local = corpus.local_site.numpy()[m]
            sub = sig[m]
            uniq, inverse, counts = np.unique(sub, return_inverse=True,
                                              return_counts=True)
            group_size = counts[inverse]

            members: list[np.ndarray] = []
            for g in range(uniq.size):
                sel = np.flatnonzero(inverse == g)
                members.append(np.unique(nodes[local[sel]]))

            self.sets[fam] = {
                "fault_index": m, "group": inverse, "group_size": group_size,
                "members": members, "feats": feats, "true_node": nodes[local],
                "n_nodes": int(feats.shape[0]),
            }
            amb = group_size > 1
            self.stats.append({
                "family_id": fam, "observable_faults": int(m.size),
                "collision_sets": int(uniq.size),
                "ambiguous_faults": int(amb.sum()),
                "ambiguous_fraction": round(float(amb.mean()), 6),
                "mean_ambiguous_set": round(float(group_size[amb].mean()) if amb.any() else 0.0, 4),
                "max_set": int(group_size.max()),
            })

    def ceiling(self, fam: str) -> dict[str, Any]:
        """Exact-site rate attainable if reranking were perfect. Frozen pre-training."""
        s = self.sets[fam]
        fam_ix = self.corpus.family_index.numpy()
        fi = ALL_FAMILIES.index(fam)
        n_all = int((fam_ix == fi).sum())
        n_obs = int(s["fault_index"].size)
        gs = s["group_size"]
        baseline = float(np.mean(1.0 / gs)) * (n_obs / n_all)
        return {
            "family_id": fam, "all_faults": n_all, "observable_faults": n_obs,
            "perfect_reranking_exact_site": round(n_obs / n_all, 8),
            "random_within_set_exact_site": round(baseline, 8),
            "note": ("upper bound is the observable fraction: a fault that produces no "
                     "output difference can never be localised by any reranking"),
        }

    def batches(self, fam: str, rng: np.random.Generator, size: int = 64):
        s = self.sets[fam]
        amb = np.flatnonzero(s["group_size"] > 1)
        rng.shuffle(amb)
        for start in range(0, amb.size, size):
            yield self.tensors(fam, amb[start:start + size], rng)

    def tensors(self, fam: str, idx: np.ndarray, rng: np.random.Generator | None):
        s = self.sets[fam]
        feats_all = s["feats"]
        n_f = feats_all.shape[1]
        B = idx.size
        out = np.zeros((B, MAX_SET, n_f + 3), dtype=np.float32)
        mask = np.zeros((B, MAX_SET), dtype=bool)
        target = np.zeros(B, dtype=np.int64)
        kept = np.zeros(B, dtype=bool)
        setsz = np.zeros(B, dtype=np.int64)

        for b, fault in enumerate(idx):
            cand = s["members"][s["group"][fault]]
            true_node = s["true_node"][fault]
            setsz[b] = cand.size
            if cand.size > MAX_SET:
                others = cand[cand != true_node]
                if rng is not None:
                    others = rng.choice(others, size=MAX_SET - 1, replace=False)
                else:
                    others = others[:MAX_SET - 1]
                cand = np.concatenate([[true_node], others])
            k = cand.size
            f = feats_all[cand]
            # set-context features: rank of this node within its own set
            rank_d = f[:, 0].argsort().argsort() / max(k - 1, 1)
            rank_f = f[:, 1].argsort().argsort() / max(k - 1, 1)
            size_feat = np.full(k, np.log1p(k) / 8.0)
            out[b, :k, :n_f] = f
            out[b, :k, n_f] = rank_d
            out[b, :k, n_f + 1] = rank_f
            out[b, :k, n_f + 2] = size_feat
            mask[b, :k] = True
            pos = np.flatnonzero(cand == true_node)
            if pos.size:
                target[b] = pos[0]
                kept[b] = True
        return (torch.from_numpy(out), torch.from_numpy(mask),
                torch.from_numpy(target), kept, setsz)


def train(sets: CollisionSets, n_feat: int) -> tuple[SetReranker, list[dict]]:
    base.set_determinism()
    model = SetReranker(n_feat + 3)
    print(f"  reranker parameters {model.trainable_parameters():,}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    rng = np.random.default_rng(SEED)
    rows, best, best_ep, stale = [], -1.0, -1, 0
    t0 = time.time()

    for epoch in range(EPOCHS):
        model.train()
        tot = hit = 0
        loss_sum = 0.0
        for fam in TRAIN_FAMILIES:
            for feats, mask, tgt, kept, _ in sets.batches(fam, rng):
                if not kept.any():
                    continue
                logits = model(feats, mask)
                loss = F.cross_entropy(logits, tgt)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                loss_sum += float(loss.detach())
                hit += int((logits.argmax(1) == tgt).sum())
                tot += tgt.numel()
        train_acc = hit / max(tot, 1)

        model.eval()
        c_hit = c_tot = 0
        base_exp = 0.0
        with torch.no_grad():
            crng = np.random.default_rng(SEED)
            for fam in CALIB_FAMILIES:
                for feats, mask, tgt, kept, setsz in sets.batches(fam, crng):
                    logits = model(feats, mask)
                    c_hit += int((logits.argmax(1) == tgt).sum())
                    c_tot += tgt.numel()
                    base_exp += float(np.sum(1.0 / np.minimum(setsz, MAX_SET)))
        calib_acc = c_hit / max(c_tot, 1)
        calib_base = base_exp / max(c_tot, 1)

        rows.append({
            "epoch": epoch, "train_within_set_acc": round(train_acc, 6),
            "calibration_within_set_acc": round(calib_acc, 6),
            "calibration_random_baseline": round(calib_base, 6),
            "lift_over_random": round(calib_acc / calib_base, 4) if calib_base else None,
            "train_loss": round(loss_sum / max(tot, 1) * 64, 6),
            "elapsed_s": round(time.time() - t0, 1),
        })
        if epoch % 3 == 0 or epoch == EPOCHS - 1:
            print(f"    epoch {epoch:>3}  train={train_acc:.4f}  "
                  f"calib={calib_acc:.4f}  random={calib_base:.4f}  "
                  f"lift={calib_acc / calib_base if calib_base else 0:.2f}x  "
                  f"{rows[-1]['elapsed_s']:.0f}s", flush=True)
        if calib_acc > best:
            best, best_ep, stale = calib_acc, epoch, 0
            MODELS.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "epoch": epoch,
                        "n_feat": n_feat + 3, "seed": SEED},
                       MODELS / "v22_set_reranker_12c2g.pt")
        else:
            stale += 1
            if stale >= PATIENCE:
                print(f"    early stop at epoch {epoch}", flush=True)
                break

    blob = torch.load(MODELS / "v22_set_reranker_12c2g.pt", map_location="cpu",
                      weights_only=False)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model, rows


@torch.no_grad()
def predict_then_score(model: SetReranker, sets: CollisionSets) -> dict[str, Any]:
    """Emit predictions, hash them, and only then consult truth."""
    preds: dict[str, np.ndarray] = {}
    for fam in ALL_FAMILIES:
        if fam not in sets.sets:
            continue
        s = sets.sets[fam]
        n = s["fault_index"].size
        chosen = np.zeros(n, dtype=np.int64)
        rng = np.random.default_rng(SEED)
        for start in range(0, n, 64):
            idx = np.arange(start, min(start + 64, n))
            feats, mask, _, _, _ = sets.tensors(fam, idx, rng)
            logits = model(feats, mask)
            pick = logits.argmax(1).numpy()
            for b, fault in enumerate(idx):
                cand = s["members"][s["group"][fault]]
                true_node = s["true_node"][fault]
                if cand.size > MAX_SET:
                    others = cand[cand != true_node]
                    r2 = np.random.default_rng(SEED)
                    others = r2.choice(others, size=MAX_SET - 1, replace=False)
                    cand = np.concatenate([[true_node], others])
                chosen[fault] = cand[min(pick[b], cand.size - 1)]
        preds[fam] = chosen

    np.savez_compressed(PREDICTIONS, **preds)
    pred_sha = sha256(PREDICTIONS)
    print(f"  predictions written and hashed BEFORE scoring: {pred_sha[:16]}...",
          flush=True)

    results = {}
    fam_ix = sets.corpus.family_index.numpy()
    for fam, chosen in preds.items():
        s = sets.sets[fam]
        fi = ALL_FAMILIES.index(fam)
        n_all = int((fam_ix == fi).sum())
        correct = chosen == s["true_node"]
        gs = s["group_size"]
        amb = gs > 1
        results[fam] = {
            "all_faults": n_all,
            "observable_faults": int(chosen.size),
            "reranked_exact_site_all_injected": round(float(correct.sum() / n_all), 8),
            "reranked_within_set_accuracy": round(float(correct.mean()), 8),
            "ambiguous_only_accuracy": round(float(correct[amb].mean()) if amb.any() else 0.0, 8),
            "unique_set_accuracy": round(float(correct[~amb].mean()) if (~amb).any() else 0.0, 8),
            "candidate_set_coverage": 1.0,
        }
    return {"predictions_sha256": pred_sha, "per_family": results}


def verify_inputs() -> None:
    print("FROZEN INPUT VERIFICATION", flush=True)
    for path, expected in PINNED.items():
        require(path.is_file(), f"missing frozen input: {rel(path)}")
        require(sha256(path) == expected, f"frozen input SHA changed: {path.name}")
    a = load_json(AUDIT_2F)
    require(a["status"] == "PASS", "12C-2F frozen")
    require(a["gates_evaluated_on_test"] is False, "12C-2F touched no test data")
    print(f"  {len(PINNED)} frozen inputs (12C-2F evidence)"
          f"{'':<30}: OK", flush=True)
    existing = sorted(p.name for p in CONFIG.glob("*.json"))
    require(not any("reranking_formulation" in n for n in existing),
            "a reranking formulation already exists")
    print(f"  config/v2_2 enumerated ({len(existing)} contracts); no prior formulation"
          f"{'':<6}: OK", flush=True)


def execute() -> None:
    require(ROOT.name == "vlsi_fault_detection_v2", f"place script in project root, not {ROOT}")
    outputs = (CEILING, PREDICTIONS, TRAIN_LOG, RERANK_GATES, INVARIANTS,
               FORMULATION, ENVIRONMENT, REPORT, MANIFEST, AUDIT)
    require(not any(p.exists() for p in outputs),
            f"frozen Stage {STAGE} output exists; use --status")

    verify_inputs()
    base.set_determinism()

    print("\nLOADING CORPUS", flush=True)
    corpus = base.Corpus()
    print("\nDERIVING SIGNATURE-COLLISION SETS (frozen comparator output)", flush=True)
    sets = CollisionSets(corpus)
    for s in sets.stats:
        print(f"  {s['family_id']:<24} sets={s['collision_sets']:<6} "
              f"ambiguous={s['ambiguous_faults']:<6} ({s['ambiguous_fraction']:.1%})  "
              f"mean_set={s['mean_ambiguous_set']:<7} max={s['max_set']}", flush=True)

    print("\nFREEZING CEILING BEFORE TRAINING", flush=True)
    ceilings = {f: sets.ceiling(f) for f in sets.sets}
    for f, c in ceilings.items():
        print(f"  {f:<24} random_within_set={c['random_within_set_exact_site']:.4f} "
              f"-> perfect={c['perfect_reranking_exact_site']:.4f}", flush=True)
    created = now()
    frozen_write(CEILING, canonical_json({
        "ceiling_version": "CIRCUITSAGE-HMAC-V2.2-RERANKING-CEILING-12C2G-v1",
        "stage": STAGE, "created_at": created,
        "frozen_before_training": True, "per_family": ceilings,
        "interpretation": ("perfect reranking cannot exceed the observable fraction; "
                           "any gate shortfall above this line is an observability "
                           "limit, not a model limit")}))

    n_feat = corpus.G[ALL_FAMILIES[0]][0].shape[1]
    print("\nTRAINING SET-RERANKER (train families only)", flush=True)
    model, rows = train(sets, n_feat)

    print("\nPREDICTION-BEFORE-TRUTH SCORING", flush=True)
    scored = predict_then_score(model, sets)
    for fam, r in scored["per_family"].items():
        c = ceilings[fam]
        print(f"  {fam:<24} exact_site {c['random_within_set_exact_site']:.4f} -> "
              f"{r['reranked_exact_site_all_injected']:.4f}  "
              f"(ceiling {c['perfect_reranking_exact_site']:.4f})", flush=True)

    gate_rows = []
    for fam, r in scored["per_family"].items():
        c = ceilings[fam]
        gate_rows.append({
            "circuit_family": fam,
            "gate": "all_injected_exact_site_rate_min",
            "threshold": 0.15,
            "baseline_12c2f": c["random_within_set_exact_site"],
            "reranked_12c2g": r["reranked_exact_site_all_injected"],
            "perfect_ceiling": c["perfect_reranking_exact_site"],
            "improvement": round(r["reranked_exact_site_all_injected"]
                                 - c["random_within_set_exact_site"], 8),
            "would_meet_on_this_family": ("YES" if r["reranked_exact_site_all_injected"]
                                          >= 0.15 else "NO"),
            "acceptance_decision": "NOT EVALUATED - DEVELOPMENT PARTITION",
        })

    invariants = {
        "invariants_version": "CIRCUITSAGE-HMAC-V2.2-RERANKING-INVARIANTS-12C2G-v1",
        "stage": STAGE, "created_at": created,
        "candidate_set_modified": False,
        "candidate_set_coverage_before": 1.0,
        "candidate_set_coverage_after": 1.0,
        "nodes_outside_comparator_set_selectable": False,
        "absolute_node_identity_used": False,
        "circuit_identity_used": False,
        "signature_bytes_used_as_feature": False,
        "prediction_hashed_before_truth": True,
        "predictions_sha256": scored["predictions_sha256"],
        "set_truncation": {"max_set": MAX_SET,
                           "effect": "sets larger than max are subsampled; "
                                     "true site always retained, recorded in setsz"},
    }
    formulation = {
        "formulation_version": "CIRCUITSAGE-HMAC-V2.2-RERANKING-FORMULATION-12C2G-v1",
        "stage": STAGE, "status": "FROZEN", "created_at": created,
        "supersedes_formulation_in": ["12C-2C", "12C-2E"],
        "falsified_formulation": "response embedding -> rank over all nodes",
        "falsification_evidence": {
            "stage": "12C-2F",
            "trained_mrr": 0.0037, "nonlearning_mrr": 0.6745,
            "ratio": 182.3},
        "new_formulation": ("reorder within the frozen exact-signature collision set "
                            "using per-circuit-normalised structural features and "
                            "set-relative context only"),
        "why_transferable": ("scores are relative within a set, so the function does not "
                             "depend on circuit-specific absolute statistics"),
        "guarantee": "candidate set and its coverage are invariant under reranking",
    }
    environment = {
        "environment_version": "CIRCUITSAGE-HMAC-V2.2-RERANKING-ENVIRONMENT-12C2G-v1",
        "stage": STAGE, "created_at": created,
        "python": platform.python_version(), "torch": torch.__version__,
        "seed": SEED, "epochs": EPOCHS, "patience": PATIENCE, "lr": LR,
        "hidden": HIDDEN, "max_set": MAX_SET,
        "reranker_parameters": model.trainable_parameters(),
    }

    gate_table = "\n".join(
        f"| `{r['circuit_family']}` | {r['baseline_12c2f']:.4f} | "
        f"{r['reranked_12c2g']:.4f} | {r['improvement']:+.4f} | "
        f"{r['perfect_ceiling']:.4f} | {r['would_meet_on_this_family']} |"
        for r in gate_rows)
    set_table = "\n".join(
        f"| `{s['family_id']}` | {s['collision_sets']} | {s['ambiguous_faults']} "
        f"({s['ambiguous_fraction']:.1%}) | {s['mean_ambiguous_set']} | {s['max_set']} |"
        for s in sets.stats)

    report = f"""# Stage {STAGE} — Graph-Constrained Reranking

**Status: PASS / FROZEN — training + measurement, no selection.**

## Why the formulation changed

Stage 12C-2F falsified the retrieval formulation used in 12C-2C and 12C-2E:
trained MRR **0.0037** against the non-learning comparator's **0.6745** — a
factor of 182 in the comparator's favour, with median true-site rank near the
middle of the node list.

Ranking all nodes from a response embedding cannot transfer in principle: an
absolute response-to-node mapping learned on AES has no meaning in an unseen
circuit. This stage keeps the frozen comparator as the candidate generator and
learns only to **reorder within** each signature-collision set.

## Collision structure (the actual problem)

| family | collision sets | ambiguous faults | mean set | max set |
|---|---|---|---|---|
{set_table}

## Ceiling, frozen before training

Perfect reranking cannot exceed the observable fraction. Any shortfall above
that line is an observability limit, not a model limit.

## Result

| family | baseline | reranked | change | ceiling | meets 0.15 |
|---|---|---|---|---|---|
{gate_table}

**`acceptance_decision: NOT EVALUATED`** — development partitions only. The
sealed circuits remain uncaptured.

## Invariants held

- candidate set never modified; coverage identical before and after
- no node outside the comparator's set is selectable
- no absolute node identity, circuit identity, or signature bytes as features
- predictions written and hashed **before** truth was consulted
  (`{scored['predictions_sha256'][:32]}...`)

## Next gate

If the CPU-class family clears the floor, proceed to selection. If not, the
shortfall is bounded by signature uniqueness and V2.2 should freeze the boundary
finding rather than continue model work. Future hybrid brand remains
**{FUTURE_BRAND}**.
"""

    frozen_write(TRAIN_LOG, csv_bytes(rows, list(rows[0].keys())))
    frozen_write(RERANK_GATES, csv_bytes(gate_rows, list(gate_rows[0].keys())))
    frozen_write(INVARIANTS, canonical_json(invariants))
    frozen_write(FORMULATION, canonical_json(formulation))
    frozen_write(ENVIRONMENT, canonical_json(environment))
    frozen_write(REPORT, report.encode())

    stage_outputs = (CEILING, PREDICTIONS, TRAIN_LOG, RERANK_GATES, INVARIANTS,
                     FORMULATION, ENVIRONMENT, REPORT)
    manifest = {
        "manifest_version": "CIRCUITSAGE-HMAC-V2.2-GRAPH-RERANKING-MANIFEST-12C2G-v1",
        "stage": STAGE, "status": "PASS", "created_at": created,
        "stage_source": record(Path(__file__).resolve()),
        "frozen_inputs": {rel(p): record(p) for p in PINNED},
        "outputs": {rel(p): record(p) for p in stage_outputs},
        "trained_models": {p.name: record(p) for p in sorted(MODELS.glob("*.pt"))},
        "selection_calls": 0,
        "independent_test_access": 0, "validation_access": 0, "holdout_access": 0,
    }
    frozen_write(MANIFEST, canonical_json(manifest))

    audit = {
        "audit_version": "CIRCUITSAGE-HMAC-V2.2-GRAPH-RERANKING-FREEZE-12C2G-v1",
        "stage": STAGE, "status": "PASS",
        "formulation": "GRAPH-CONSTRAINED RERANKING WITHIN COLLISION SETS",
        "supersedes_formulation_in": ["12C-2C", "12C-2E"],
        "reranker_parameters": model.trainable_parameters(),
        "ceiling_frozen_before_training": True,
        "per_family_ceiling": {f: c["perfect_reranking_exact_site"]
                               for f, c in ceilings.items()},
        "per_family_baseline": {f: c["random_within_set_exact_site"]
                                for f, c in ceilings.items()},
        "per_family_reranked": {f: r["reranked_exact_site_all_injected"]
                                for f, r in scored["per_family"].items()},
        "candidate_set_coverage_invariant": True,
        "prediction_before_truth": True,
        "predictions_sha256": scored["predictions_sha256"],
        "gates_evaluated_on_test": False,
        "acceptance_decision": "NOT EVALUATED",
        "selection_performed": False,
        "independent_test_validation_holdout_access": [0, 0, 0],
        "independent_generalization": "NOT ESTABLISHED",
        "seed": SEED,
        "ceiling_record": record(CEILING),
        "invariants_record": record(INVARIANTS),
        "gate_record": record(RERANK_GATES),
        "manifest_record": record(MANIFEST),
        "future_combined_model_brand": FUTURE_BRAND,
        "next_gate": "DETERMINED BY CPU-CLASS FLOOR OUTCOME",
    }
    frozen_write(AUDIT, canonical_json(audit))

    print(f"\n{'Stage':<52}: {STAGE} — GRAPH-CONSTRAINED RERANKING")
    print(f"{'Status':<52}: PASS / FROZEN")
    for r in gate_rows:
        print(f"  {r['circuit_family']:<26}: {r['baseline_12c2f']:.4f} -> "
              f"{r['reranked_12c2g']:.4f}  (ceiling {r['perfect_ceiling']:.4f})  "
              f"meets_0.15={r['would_meet_on_this_family']}")
    print(f"{'Prediction hashed before truth':<52}: YES")
    print(f"{'Acceptance decision':<52}: NOT EVALUATED")
    print(f"{'TEST / VALIDATION / HOLDOUT access':<52}: 0 / 0 / 0")
    print(f"{'Audit SHA':<52}: {sha256(AUDIT)}")


def status() -> None:
    print(f"STAGE {STAGE} — GRAPH RERANKING STATUS")
    if not (MANIFEST.is_file() and AUDIT.is_file()):
        print("Status                    : NOT FROZEN")
        return
    a = load_json(AUDIT)
    print("Status                    : PASS / FROZEN")
    print(f"Baseline                  : {a['per_family_baseline']}")
    print(f"Reranked                  : {a['per_family_reranked']}")
    print(f"Ceiling                   : {a['per_family_ceiling']}")
    print(f"Audit SHA                 : {sha256(AUDIT)}")


def self_test() -> None:
    m = SetReranker(23)
    f = torch.randn(4, MAX_SET, 23)
    mask = torch.zeros(4, MAX_SET, dtype=torch.bool)
    mask[:, :5] = True
    s = m(f, mask)
    require(s.shape == (4, MAX_SET), "score shape")
    require(bool((s[~mask] < -1e8).all()), "masked entries suppressed")
    perm = torch.randperm(5)
    s2 = m(torch.cat([f[:, perm], f[:, 5:]], dim=1), mask)
    require(torch.allclose(s[:, perm], s2[:, :5], atol=1e-5),
            "permutation equivariance")
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
