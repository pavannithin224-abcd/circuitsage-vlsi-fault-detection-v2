#!/usr/bin/env python3
"""Stage 12C-2E: contract-compliant retraining with topology-stratified hard negatives.

Stage 12C-2C trained the three frozen candidates but sampled contrastive
negatives UNIFORMLY AT RANDOM over graph nodes.  The frozen Stage 12C-1A
training/selection contract requires:

    hard_negatives: "full-catalog or deterministic topology-stratified
                     negatives within TRAIN circuits only"

Uniform random negatives violate that clause.  A random node is trivially
separable from the true fault node, so the retrieval objective is too easy and
never forces fine-grained discrimination - which is precisely the capability
cross-circuit transfer requires.  Stage 12C-2C is therefore a valid frozen
record of a CONTRACT-NONCOMPLIANT run, not evidence that the approach fails.

Stage 12C-2C is NOT modified.  It is retained as the ablation baseline, and this
stage reports the paired comparison so the contribution of hard negatives is
isolated rather than assumed.

Hard-negative construction (deterministic, TRAIN circuits only):

  * structural stratum - nodes sharing the anchor's cell-type index and whose
    (depth_to_output, fan_out) fall in the same quantile bucket as the anchor
  * signature stratum - sites whose frozen behaviour signature collides with
    the anchor's signature (the true ambiguity set the model must separate)
  * a bounded uniform remainder so the partition function stays well behaved

Strata are derived ONLY from GENERALIZATION_TRAIN families and from frozen
artifacts.  No calibration, test or holdout information enters negative
selection.  Selection remains unperformed; this stage trains only.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import io
import json
import os
import platform
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

import stage_12c2c_candidate_training as base


STAGE = "12C-2E"
ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "config/v2_2"
RESULT2 = ROOT / "results/circuitsage_hmac_v2_12c2"
WORK = RESULT2 / "hard_negative_training_12c2e"
MODELS = WORK / "models"
LOCK_FILE = WORK / ".stage_12c2e.lock"
CHECKPOINT = WORK / "circuitsage_hmac_v2_2_training_checkpoint_12c2e.json"

SOURCE_2C = ROOT / "stage_12c2c_candidate_training.py"
AUDIT_2C = RESULT2 / "circuitsage_hmac_v2_2_candidate_training_freeze_12c2c.json"
MANIFEST_2C = RESULT2 / "circuitsage_hmac_v2_2_candidate_training_manifest_12c2c.json"
SUMMARY_2C = (RESULT2 / "candidate_training_12c2c"
              / "circuitsage_hmac_v2_2_candidate_summary_12c2c.csv")

PINNED = dict(base.PINNED)
PINNED.update({
    SOURCE_2C: "72694dfce4795750533503ed0b6e2fdb58007f873ae6aca7c57a27aa70b6e12b",
    AUDIT_2C: "5cc4dc864fb36cc96c058838f84de9532f9e33e7eb57494eb8fd3e1513e11267",
    MANIFEST_2C: "e75c23bf0e66ee126d32bbed5cb66603937907bd674d4928a5c32ab973af0a5e",
})
PREFIX_ONLY: set[Path] = set()

NEGATIVE_POLICY = CONFIG / "circuitsage_hmac_v2_2_hard_negative_policy_12c2e.json"
TRAINING_LOG = WORK / "circuitsage_hmac_v2_2_training_log_12c2e.csv"
CANDIDATE_SUMMARY = WORK / "circuitsage_hmac_v2_2_candidate_summary_12c2e.csv"
ABLATION = WORK / "circuitsage_hmac_v2_2_hard_negative_ablation_12c2e.json"
STRATA_SUMMARY = WORK / "circuitsage_hmac_v2_2_negative_strata_summary_12c2e.csv"
ENVIRONMENT = WORK / "circuitsage_hmac_v2_2_training_environment_12c2e.json"
PREFLIGHT = WORK / "circuitsage_hmac_v2_2_training_preflight_12c2e.json"
REPORT = WORK / "circuitsage_hmac_v2_2_training_report_12c2e.md"
MANIFEST = RESULT2 / "circuitsage_hmac_v2_2_hard_negative_training_manifest_12c2e.json"
AUDIT = RESULT2 / "circuitsage_hmac_v2_2_hard_negative_training_freeze_12c2e.json"

ALL_FAMILIES = base.ALL_FAMILIES
TRAIN_FAMILIES = base.TRAIN_FAMILIES
CALIB_FAMILIES = base.CALIB_FAMILIES

SEED = base.SEED
EPOCHS = 40
BATCH = base.BATCH
NEGATIVES = 128
STRUCTURAL_FRACTION = 0.50
SIGNATURE_FRACTION = 0.30
UNIFORM_FRACTION = 0.20
PATIENCE = 10
DEPTH_BUCKETS = 8
FANOUT_BUCKETS = 4
FUTURE_BRAND = base.FUTURE_BRAND

stop = base.stop
require = base.require
now = base.now
canonical_json = base.canonical_json
sha256 = base.sha256
rel = base.rel
record = base.record
load_json = base.load_json
read_csv = base.read_csv
csv_bytes = base.csv_bytes
frozen_write = base.frozen_write
atomic_json = base.atomic_json


class HardNegativeSampler:
    """Deterministic topology-stratified negatives, TRAIN circuits only."""

    def __init__(self, corpus: base.Corpus) -> None:
        self.corpus = corpus
        self.strata: dict[str, dict[str, Any]] = {}
        self.stats: list[dict[str, Any]] = []
        names = base.__dict__  # noqa: F841  (kept for clarity of provenance)
        fname = list(load_json(base.GRAPH_CONTRACT_2A)["node_feature_names"])
        d_ix = fname.index("depth_to_output")
        f_ix = fname.index("fan_out")
        t_ix = fname.index("type_index")

        sig_all = corpus.targ["behavior_signature_sha256"]
        for fam in TRAIN_FAMILIES:
            fi = ALL_FAMILIES.index(fam)
            x, ei, sn = corpus.G[fam]
            nodes = x.shape[0]
            depth = x[:, d_ix].numpy()
            fan = x[:, f_ix].numpy()
            typ = x[:, t_ix].numpy().astype(np.int64)

            d_edge = np.quantile(depth, np.linspace(0, 1, DEPTH_BUCKETS + 1)[1:-1])
            f_edge = np.quantile(fan, np.linspace(0, 1, FANOUT_BUCKETS + 1)[1:-1])
            d_bucket = np.digitize(depth, d_edge)
            f_bucket = np.digitize(fan, f_edge)
            key = (typ.astype(np.int64) * (DEPTH_BUCKETS + 1) * (FANOUT_BUCKETS + 1)
                   + d_bucket * (FANOUT_BUCKETS + 1) + f_bucket)

            order = np.argsort(key, kind="stable")
            sorted_key = key[order]
            boundaries = np.flatnonzero(np.r_[True, sorted_key[1:] != sorted_key[:-1]])
            groups: dict[int, np.ndarray] = {}
            for gi, start in enumerate(boundaries):
                end = boundaries[gi + 1] if gi + 1 < len(boundaries) else len(order)
                groups[int(sorted_key[start])] = order[start:end]

            # signature collision groups over this family's observable faults
            mask = (corpus.family_index.numpy() == fi) & (corpus.observable.numpy() > 0)
            idxs = np.flatnonzero(mask)
            sigs = sig_all[idxs]
            sites = corpus.local_site.numpy()[idxs]
            sig_to_nodes: dict[bytes, np.ndarray] = {}
            uniq, inverse = np.unique(sigs, return_inverse=True)
            node_arr = sn.numpy()
            for u in range(len(uniq)):
                members = sites[inverse == u]
                if members.size > 1:
                    sig_to_nodes[uniq[u]] = np.unique(node_arr[members])

            self.strata[fam] = {
                "node_key": key, "groups": groups, "nodes": nodes,
                "sig_to_nodes": sig_to_nodes,
            }
            collide = sum(v.size for v in sig_to_nodes.values())
            self.stats.append({
                "family_id": fam, "nodes": int(nodes),
                "structural_strata": int(len(groups)),
                "mean_stratum_size": round(float(np.mean([g.size for g in groups.values()])), 3),
                "max_stratum_size": int(max(g.size for g in groups.values())),
                "signature_collision_groups": int(len(sig_to_nodes)),
                "nodes_in_collision_groups": int(collide),
            })

    def sample(self, fam: str, anchor_nodes: np.ndarray, anchor_sites: np.ndarray,
               generator: np.random.Generator) -> torch.Tensor:
        s = self.strata[fam]
        n_struct = int(NEGATIVES * STRUCTURAL_FRACTION)
        n_sig = int(NEGATIVES * SIGNATURE_FRACTION)
        n_unif = NEGATIVES - n_struct - n_sig

        pool: list[np.ndarray] = []
        # structural: same type + depth/fanout bucket as sampled anchors
        pick = generator.choice(anchor_nodes, size=min(8, anchor_nodes.size), replace=True)
        cand = [s["groups"].get(int(s["node_key"][p]), np.empty(0, dtype=np.int64))
                for p in pick]
        cand = [c for c in cand if c.size]
        if cand:
            merged = np.concatenate(cand)
            pool.append(generator.choice(merged, size=n_struct, replace=True))
        else:
            pool.append(generator.integers(0, s["nodes"], size=n_struct))

        # signature-collision: the true ambiguity set
        sig_nodes = s["sig_to_nodes"]
        if sig_nodes:
            keys = list(sig_nodes.keys())
            chosen = generator.choice(len(keys), size=min(6, len(keys)), replace=True)
            merged = np.concatenate([sig_nodes[keys[c]] for c in chosen])
            pool.append(generator.choice(merged, size=n_sig, replace=True))
        else:
            pool.append(generator.integers(0, s["nodes"], size=n_sig))

        pool.append(generator.integers(0, s["nodes"], size=n_unif))
        neg = np.concatenate(pool)
        # never let a negative equal its own positive
        neg = np.where(np.isin(neg, anchor_nodes),
                       (neg + 1) % s["nodes"], neg)
        return torch.from_numpy(neg.astype(np.int64))


def contrastive_step_hard(model: base.Candidate, corpus: base.Corpus,
                          sampler: HardNegativeSampler, idx: torch.Tensor,
                          generator: np.random.Generator):
    fam_id = int(corpus.family_index[idx[0]])
    fam = ALL_FAMILIES[fam_id]
    same = idx[corpus.family_index[idx] == fam_id]
    if same.numel() < 8:
        return torch.zeros((), requires_grad=True), {}

    x, ei, sn = corpus.G[fam]
    node_emb = model.graph(x, ei)
    xor, cyc, to, pe, vm = corpus.batch(same.numpy())
    q = model.response(xor, cyc, to, pe, vm)

    pos_nodes = sn[corpus.local_site[same]]
    pos = node_emb[pos_nodes]
    neg_nodes = sampler.sample(fam, pos_nodes.numpy(),
                               corpus.local_site[same].numpy(), generator)
    neg = node_emb[neg_nodes]

    logits = torch.cat([(q * pos).sum(1, keepdim=True), q @ neg.T], dim=1) / 0.07
    retrieval = F.cross_entropy(logits, torch.zeros(q.shape[0], dtype=torch.long))

    obs = corpus.observable[same]
    detection = F.binary_cross_entropy_with_logits(
        model.detector(q).squeeze(1), obs)
    polarity = F.cross_entropy(
        model.polarity(torch.cat([q, pos], dim=1)), corpus.stuck[same])
    ood_scores = torch.stack([h(q).squeeze(1) for h in model.ood], dim=0)
    ood = ood_scores.var(dim=0).mean() if model.ensemble > 1 else ood_scores.mean() * 0.0

    loss = retrieval + 0.5 * detection + 0.3 * polarity + 0.05 * ood
    with torch.no_grad():
        top1 = float((logits.argmax(dim=1) == 0).float().mean())
    return loss, {"retrieval": float(retrieval.detach()),
                  "detection": float(detection.detach()),
                  "polarity": float(polarity.detach()), "top1": top1}


def train_candidate(candidate_id: str, cap: int, corpus: base.Corpus,
                    sampler: HardNegativeSampler, vocab_size: int) -> dict[str, Any]:
    base.set_determinism()
    model = base.Candidate(candidate_id, vocab_size)
    params = model.trainable_parameters()
    print(f"\n  {candidate_id}", flush=True)
    print(f"    parameters {params:,} / cap {cap:,}", flush=True)
    require(params <= cap, f"{candidate_id} exceeds cap: {params} > {cap}")

    opt = torch.optim.AdamW(model.parameters(), lr=base.LR,
                            weight_decay=base.WEIGHT_DECAY)
    train_pool = corpus.pool(TRAIN_FAMILIES, observable_only=True)
    calib_pool = corpus.pool(CALIB_FAMILIES, observable_only=True)
    steps = train_pool.numel() // BATCH
    generator = np.random.default_rng(SEED)
    rows: list[dict[str, Any]] = []
    best, best_epoch, stale = -1.0, -1, 0
    start = time.time()

    for epoch in range(EPOCHS):
        model.train()
        order = torch.from_numpy(generator.permutation(train_pool.numel()))
        agg: dict[str, float] = {}
        for s in range(steps):
            sel = train_pool[order[s * BATCH:(s + 1) * BATCH]]
            loss, st = contrastive_step_hard(model, corpus, sampler, sel, generator)
            if not st:
                continue
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            for k, v in st.items():
                agg[k] = agg.get(k, 0.0) + v
        n = max(steps, 1)
        mon = base.monitor_calibration(model, corpus, calib_pool,
                                       np.random.default_rng(SEED + epoch))
        row = {
            "candidate_id": candidate_id, "epoch": epoch,
            "train_retrieval_loss": round(agg.get("retrieval", 0.0) / n, 6),
            "train_top1": round(agg.get("top1", 0.0) / n, 6),
            "calibration_top1": round(mon["calibration_top1"], 6),
            "calibration_detection_acc": round(mon["calibration_detection_acc"], 6),
            "elapsed_s": round(time.time() - start, 1),
        }
        rows.append(row)
        if epoch % 5 == 0 or epoch == EPOCHS - 1:
            print(f"    epoch {epoch:>3}  train_top1={row['train_top1']:.4f}  "
                  f"calib_top1={row['calibration_top1']:.4f}  "
                  f"{row['elapsed_s']:.0f}s", flush=True)
        if row["calibration_top1"] > best:
            best, best_epoch, stale = row["calibration_top1"], epoch, 0
            MODELS.mkdir(parents=True, exist_ok=True)
            torch.save({"candidate_id": candidate_id, "epoch": epoch,
                        "state_dict": model.state_dict(), "parameters": params,
                        "seed": SEED, "negatives": "TOPOLOGY_STRATIFIED_HARD"},
                       MODELS / f"{candidate_id.lower()}_12c2e.pt")
        else:
            stale += 1
            if stale >= PATIENCE:
                print(f"    early stop at epoch {epoch}", flush=True)
                break

    return {"candidate_id": candidate_id, "parameters": params, "parameter_cap": cap,
            "cap_respected": True, "epochs_run": len(rows), "best_epoch": best_epoch,
            "best_calibration_top1": round(best, 6),
            "final_train_top1": rows[-1]["train_top1"] if rows else 0.0,
            "wall_clock_s": round(time.time() - start, 1), "rows": rows}


def verify_inputs() -> list[dict[str, str]]:
    print("FROZEN INPUT VERIFICATION", flush=True)
    for path, expected in PINNED.items():
        require(path.is_file(), f"missing frozen input: {rel(path)}")
        actual = sha256(path)
        if path in base.PINNED and path in getattr(base, "PREFIX_ONLY", set()):
            require(actual.startswith(expected), f"SHA prefix: {path.name}")
        else:
            require(actual == expected, f"frozen input SHA changed: {path.name}")
    print(f"  {len(PINNED)} frozen inputs incl. Stage 12C-2C evidence"
          f"{'':<28}: OK", flush=True)

    audit_2c = load_json(AUDIT_2C)
    require(audit_2c.get("status") == "PASS", "12C-2C frozen")
    require(audit_2c.get("selection_performed") is False, "12C-2C performed no selection")

    contract = load_json(base.TRAINING_1A)
    require("topology-stratified" in contract["hard_negatives"],
            "contract requires topology-stratified negatives")

    existing = sorted(p.name for p in CONFIG.glob("*.json"))
    require(not any("hard_negative_policy" in n for n in existing),
            "a hard-negative policy already exists")
    print(f"  config/v2_2 enumerated ({len(existing)} contracts); no prior policy"
          f"{'':<14}: OK", flush=True)
    return read_csv(base.CANDIDATE_GRID_1A)


def execute() -> None:
    require(ROOT.name == "vlsi_fault_detection_v2", f"place script in project root, not {ROOT}")
    outputs = (NEGATIVE_POLICY, TRAINING_LOG, CANDIDATE_SUMMARY, ABLATION,
               STRATA_SUMMARY, ENVIRONMENT, PREFLIGHT, REPORT, MANIFEST, AUDIT)
    require(not any(p.exists() for p in outputs),
            f"frozen Stage {STAGE} output exists; use --status")

    grid = verify_inputs()
    base.set_determinism()

    print("\nLOADING CORPUS", flush=True)
    corpus = base.Corpus()
    vocab_size = load_json(base.VOCABULARY_2A)["size"]

    print("\nBUILDING TOPOLOGY-STRATIFIED NEGATIVE STRATA (TRAIN ONLY)", flush=True)
    sampler = HardNegativeSampler(corpus)
    for s in sampler.stats:
        print(f"  {s['family_id']:<24} strata={s['structural_strata']:<6} "
              f"mean={s['mean_stratum_size']:<8} sig_groups={s['signature_collision_groups']:<6} "
              f"nodes_in_collisions={s['nodes_in_collision_groups']}", flush=True)

    print("\nCANDIDATE TRAINING (hard negatives)", flush=True)
    summaries, all_rows = [], []
    for r in grid:
        if r["trainable"].strip().upper() != "YES":
            continue
        s = train_candidate(r["candidate_id"], int(r["parameter_cap"]),
                            corpus, sampler, vocab_size)
        all_rows.extend(s.pop("rows"))
        summaries.append(s)
        atomic_json(CHECKPOINT, {"stage": STAGE, "status": "RUNNING",
                                 "completed": [x["candidate_id"] for x in summaries],
                                 "updated_at": now()})

    prev = {r["candidate_id"]: float(r["best_calibration_top1"])
            for r in read_csv(SUMMARY_2C)}
    ablation_rows = []
    for s in summaries:
        before = prev.get(s["candidate_id"], 0.0)
        after = s["best_calibration_top1"]
        ablation_rows.append({
            "candidate_id": s["candidate_id"],
            "calibration_top1_random_negatives_12c2c": before,
            "calibration_top1_hard_negatives_12c2e": after,
            "absolute_change": round(after - before, 6),
            "relative_change": round((after - before) / before, 4) if before > 0 else None,
        })

    created = now()
    policy = {
        "policy_version": "CIRCUITSAGE-HMAC-V2.2-HARD-NEGATIVE-POLICY-12C2E-v1",
        "stage": STAGE, "status": "FROZEN", "created_at": created,
        "satisfies_clause": load_json(base.TRAINING_1A)["hard_negatives"],
        "violated_by_stage": "12C-2C (uniform random negatives)",
        "negatives_per_anchor": NEGATIVES,
        "composition": {"structural_stratified": STRUCTURAL_FRACTION,
                        "signature_collision": SIGNATURE_FRACTION,
                        "bounded_uniform": UNIFORM_FRACTION},
        "structural_stratum": ("same cell-type index and same (depth_to_output, fan_out) "
                               f"quantile bucket; {DEPTH_BUCKETS} depth x {FANOUT_BUCKETS} fanout"),
        "signature_stratum": "nodes whose frozen behaviour signature collides with the anchor",
        "derived_from_partitions": ["GENERALIZATION_TRAIN"],
        "calibration_test_holdout_used_in_strata": False,
        "determinism": f"numpy default_rng(seed={SEED}), deterministic strata construction",
        "self_exclusion": "a negative equal to its own positive node is displaced",
    }
    environment = {
        "environment_version": "CIRCUITSAGE-HMAC-V2.2-TRAINING-ENVIRONMENT-12C2E-v1",
        "stage": STAGE, "created_at": created,
        "python": platform.python_version(), "torch": torch.__version__,
        "platform": platform.platform(), "cuda_available": torch.cuda.is_available(),
        "threads": torch.get_num_threads(), "seed": SEED, "epochs": EPOCHS,
        "batch": BATCH, "negatives": NEGATIVES, "patience": PATIENCE,
    }
    ablation = {
        "ablation_version": "CIRCUITSAGE-HMAC-V2.2-HARD-NEGATIVE-ABLATION-12C2E-v1",
        "stage": STAGE, "status": "REPORT-ONLY", "created_at": created,
        "comparison": "12C-2C uniform random negatives vs 12C-2E topology-stratified hard negatives",
        "controlled": ["architecture", "parameter caps", "seed", "partitions",
                       "optimizer", "learning rate", "batch size", "negative count"],
        "varied": ["negative sampling policy"],
        "rows": ablation_rows,
        "interpretation_rule": ("this isolates the contribution of the contracted negative "
                                "policy; it does not by itself establish generalization"),
    }
    preflight = {
        "preflight_version": "CIRCUITSAGE-HMAC-V2.2-TRAINING-PREFLIGHT-12C2E-v1",
        "stage": STAGE, "status": "PASS", "created_at": created,
        "gradient_families": list(TRAIN_FAMILIES),
        "monitoring_families": list(CALIB_FAMILIES),
        "calibration_gradient_updates": 0,
        "independent_test_access": 0, "validation_access": 0, "holdout_access": 0,
        "negative_strata_use_protected_data": False,
        "architecture_invented": False, "acceptance_criteria_invented": False,
        "parameter_caps_enforced": True, "selection_performed": False,
        "stage_12c2c_modified": False,
    }

    table = "\n".join(
        f"| `{r['candidate_id']}` | {r['calibration_top1_random_negatives_12c2c']:.6f} | "
        f"{r['calibration_top1_hard_negatives_12c2e']:.6f} | "
        f"{r['absolute_change']:+.6f} |" for r in ablation_rows)
    strata_table = "\n".join(
        f"| `{s['family_id']}` | {s['structural_strata']} | {s['mean_stratum_size']} | "
        f"{s['signature_collision_groups']} | {s['nodes_in_collision_groups']} |"
        for s in sampler.stats)

    report = f"""# Stage {STAGE} — Hard-Negative Retraining

**Status: PASS / FROZEN — training only, no selection.**

## Why this stage exists

Stage 12C-2C sampled contrastive negatives uniformly at random. The frozen
Stage 12C-1A contract requires *"full-catalog or deterministic
topology-stratified negatives within TRAIN circuits only"*. Uniform random
negatives violate that clause and make the retrieval task trivially easy.

Stage 12C-2C is **not modified**. It is retained as the ablation baseline.

## Negative strata (TRAIN circuits only)

| family | structural strata | mean size | signature collision groups | nodes in collisions |
|---|---|---|---|---|
{strata_table}

Composition per anchor: {int(STRUCTURAL_FRACTION*100)}% structural,
{int(SIGNATURE_FRACTION*100)}% signature-collision, {int(UNIFORM_FRACTION*100)}% bounded uniform.

## Paired ablation

| candidate | random negatives (12C-2C) | hard negatives (12C-2E) | change |
|---|---|---|---|
{table}

Everything except the negative policy is held fixed: architecture, caps, seed,
partitions, optimizer, learning rate, batch size and negative count.

## Contract compliance

- gradients on GENERALIZATION_TRAIN only
- negative strata derived from TRAIN artifacts only; calibration, test and
  holdout contribute nothing to negative selection
- calibration used for monitoring and early stopping only
- parameter caps enforced before the first optimizer step

## What this stage does NOT establish

Independent generalization remains **NOT ESTABLISHED**. No selection, no
acceptance evaluation, no protected-partition access. Future hybrid brand
remains **{FUTURE_BRAND}**.

## Next gate

**Stage 12C-2D** — model selection on GENERALIZATION_CALIBRATION, if and only if
a candidate is meaningfully above chance.
"""

    log_fields = ["candidate_id", "epoch", "train_retrieval_loss", "train_top1",
                  "calibration_top1", "calibration_detection_acc", "elapsed_s"]
    sum_fields = ["candidate_id", "parameters", "parameter_cap", "cap_respected",
                  "epochs_run", "best_epoch", "best_calibration_top1",
                  "final_train_top1", "wall_clock_s"]
    abl_fields = list(ablation_rows[0].keys())
    strata_fields = list(sampler.stats[0].keys())

    frozen_write(NEGATIVE_POLICY, canonical_json(policy))
    frozen_write(TRAINING_LOG, csv_bytes(all_rows, log_fields))
    frozen_write(CANDIDATE_SUMMARY, csv_bytes(summaries, sum_fields))
    frozen_write(ABLATION, canonical_json(ablation))
    frozen_write(STRATA_SUMMARY, csv_bytes(sampler.stats, strata_fields))
    frozen_write(ENVIRONMENT, canonical_json(environment))
    frozen_write(PREFLIGHT, canonical_json(preflight))
    frozen_write(REPORT, report.encode())

    stage_outputs = (NEGATIVE_POLICY, TRAINING_LOG, CANDIDATE_SUMMARY, ABLATION,
                     STRATA_SUMMARY, ENVIRONMENT, PREFLIGHT, REPORT)
    manifest = {
        "manifest_version": "CIRCUITSAGE-HMAC-V2.2-HARD-NEGATIVE-TRAINING-MANIFEST-12C2E-v1",
        "stage": STAGE, "status": "PASS", "created_at": created,
        "stage_source": record(Path(__file__).resolve()),
        "frozen_inputs": {rel(p): record(p) for p in PINNED},
        "outputs": {rel(p): record(p) for p in stage_outputs},
        "trained_models": {p.name: record(p) for p in sorted(MODELS.glob("*.pt"))},
        "candidates_trained": len(summaries),
        "calibration_gradient_updates": 0,
        "independent_test_access": 0, "validation_access": 0, "holdout_access": 0,
    }
    frozen_write(MANIFEST, canonical_json(manifest))

    audit = {
        "audit_version": "CIRCUITSAGE-HMAC-V2.2-HARD-NEGATIVE-TRAINING-FREEZE-12C2E-v1",
        "stage": STAGE, "status": "PASS",
        "corrects_contract_violation_in": "12C-2C",
        "stage_12c2c_modified": False,
        "negative_policy": "TOPOLOGY-STRATIFIED HARD NEGATIVES",
        "candidates_trained": len(summaries),
        "parameter_counts": {s["candidate_id"]: s["parameters"] for s in summaries},
        "best_calibration_top1": {s["candidate_id"]: s["best_calibration_top1"]
                                  for s in summaries},
        "best_epoch": {s["candidate_id"]: s["best_epoch"] for s in summaries},
        "ablation": ablation_rows,
        "parameter_caps_enforced": True,
        "calibration_gradient_updates": 0,
        "negative_strata_use_protected_data": False,
        "selection_performed": False,
        "acceptance_evaluation": "NOT AUTHORIZED",
        "independent_test_validation_holdout_access": [0, 0, 0],
        "independent_generalization": "NOT ESTABLISHED",
        "seed": SEED,
        "negative_policy_record": record(NEGATIVE_POLICY),
        "ablation_record": record(ABLATION),
        "candidate_summary_record": record(CANDIDATE_SUMMARY),
        "manifest_record": record(MANIFEST),
        "future_combined_model_brand": FUTURE_BRAND,
        "next_gate": "STAGE 12C-2D — MODEL SELECTION ON CALIBRATION",
    }
    frozen_write(AUDIT, canonical_json(audit))

    print(f"\n{'Stage':<52}: {STAGE} — HARD-NEGATIVE RETRAINING")
    print(f"{'Status':<52}: PASS / FROZEN")
    for r in ablation_rows:
        print(f"  {r['candidate_id']:<40}: "
              f"{r['calibration_top1_random_negatives_12c2c']:.4f} -> "
              f"{r['calibration_top1_hard_negatives_12c2e']:.4f} "
              f"({r['absolute_change']:+.4f})")
    print(f"{'Selection performed':<52}: NO")
    print(f"{'TEST / VALIDATION / HOLDOUT access':<52}: 0 / 0 / 0")
    print(f"{'Audit SHA':<52}: {sha256(AUDIT)}")
    print(f"{'Next gate':<52}: STAGE 12C-2D — SELECTION")


def status() -> None:
    print(f"STAGE {STAGE} — HARD-NEGATIVE TRAINING STATUS")
    if MANIFEST.is_file() and AUDIT.is_file():
        a = load_json(AUDIT)
        print("Status                    : PASS / FROZEN")
        print(f"Best calibration Top-1    : {a['best_calibration_top1']}")
        print(f"Best epoch                : {a['best_epoch']}")
        print(f"Ablation                  : {a['ablation']}")
        print(f"Audit SHA                 : {sha256(AUDIT)}")
        return
    if CHECKPOINT.is_file():
        print(f"Status                    : RUNNING  {load_json(CHECKPOINT)}")
        return
    print("Status                    : NOT STARTED")


def self_test() -> None:
    require(abs(STRUCTURAL_FRACTION + SIGNATURE_FRACTION + UNIFORM_FRACTION - 1.0) < 1e-9,
            "negative fractions sum to 1")
    require(SUMMARY_2C.is_file(), "12C-2C summary present for ablation")
    prev = read_csv(SUMMARY_2C)
    require(len(prev) == 3, "three baseline candidates")
    contract = load_json(base.TRAINING_1A)
    require("topology-stratified" in contract["hard_negatives"], "contract clause")
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
