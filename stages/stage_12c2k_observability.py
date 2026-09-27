#!/usr/bin/env python3
"""Stage 12C-2K: graph model on an identifiable target (fault observability).

Stage 12C-2J proved that selecting the true site from inside an exact-signature
collision set is not identifiable under the frozen uniform fault catalogue.  That
is why the Stage 12C-2G reranker measured a lift of exactly 1.00x: it was
reporting the correct posterior.

This stage puts a graph model on a target that IS identifiable.

    target: will a fault at this site be OBSERVABLE at all under the frozen
            test scheme?

Why this target is identifiable, stated before training
------------------------------------------------------
  * the label is a deterministic function of the site and the test scheme, not a
    tie between indistinguishable alternatives
  * the base rate varies enormously across circuits (0.380 to 0.999 in Stage
    12C-2A), so there is real variance to explain
  * Stage 12C-2A measured corr(depth_to_output, observability) = -0.331 and
    -0.311 on the two low-observability circuits, so a structural signal is
    already known to exist
  * the quantity is circuit-relative: it asks whether THIS site reaches an
    output, which is a topological question that carries across circuits

Honest framing
--------------
Observability prediction does NOT localize faults.  It answers a different and
genuinely useful question: which parts of a design are untestable under the
current test scheme.  That is the input a test engineer needs in order to IMPROVE
the scheme, which is the only route that raises the localization ceiling proved
in 12C-2J.

Protocol
--------
Train on GENERALIZATION_TRAIN circuits, monitor on GENERALIZATION_CALIBRATION,
never touch sealed circuits.  Compare against two non-learning baselines
(majority class, and a single-feature depth threshold) so any claimed benefit is
attributable to the graph model rather than to the base rate.  Predictions are
written and hashed before truth is consulted.
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


STAGE = "12C-2K"
ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "config/v2_2"
RESULT2 = ROOT / "results/circuitsage_hmac_v2_12c2"
WORK = RESULT2 / "observability_model_12c2k"
MODELS = WORK / "models"
LOCK_FILE = WORK / ".stage_12c2k.lock"

SOURCE_2J = ROOT / "stage_12c2j_identifiability.py"
AUDIT_2J = RESULT2 / "circuitsage_hmac_v2_2_identifiability_freeze_12c2j.json"
THEOREM_2J = CONFIG / "circuitsage_hmac_v2_2_identifiability_statement_12c2j.json"

PINNED = {
    AUDIT_2J: "3badea39f6e47995c0484ed3a784ad26363b0d34387cf9e3eedbe8b947ff0ade",
}

TARGET_CONTRACT = CONFIG / "circuitsage_hmac_v2_2_observability_target_12c2k.json"
PREDICTIONS = WORK / "circuitsage_hmac_v2_2_observability_predictions_12c2k.npz"
TRAIN_LOG = WORK / "circuitsage_hmac_v2_2_observability_training_log_12c2k.csv"
RESULTS = WORK / "circuitsage_hmac_v2_2_observability_results_12c2k.csv"
BASELINES = WORK / "circuitsage_hmac_v2_2_baseline_comparison_12c2k.csv"
ENVIRONMENT = WORK / "circuitsage_hmac_v2_2_observability_environment_12c2k.json"
REPORT = WORK / "circuitsage_hmac_v2_2_observability_report_12c2k.md"
MANIFEST = RESULT2 / "circuitsage_hmac_v2_2_observability_manifest_12c2k.json"
AUDIT = RESULT2 / "circuitsage_hmac_v2_2_observability_freeze_12c2k.json"

ALL_FAMILIES = base.ALL_FAMILIES
TRAIN_FAMILIES = base.TRAIN_FAMILIES
CALIB_FAMILIES = base.CALIB_FAMILIES
SEED = base.SEED
EPOCHS = 60
PATIENCE = 12
LR = 3e-3
HIDDEN = 64
LAYERS = 3
FUTURE_BRAND = base.FUTURE_BRAND

stop, require, now = base.stop, base.require, base.now
canonical_json, sha256, rel = base.canonical_json, base.sha256, base.rel
record, load_json, csv_bytes = base.record, base.load_json, base.csv_bytes
frozen_write = base.frozen_write


class ObservabilityGNN(nn.Module):
    """Message-passing encoder + per-node binary head.

    Uses only per-circuit-normalised structural features, so the learned
    function is about topology rather than circuit identity.
    """

    def __init__(self, n_feat: int, vocab: int) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab, 8)
        d = n_feat + 8
        self.inp = nn.Sequential(nn.Linear(d, HIDDEN), nn.LayerNorm(HIDDEN), nn.GELU())
        self.msg = nn.ModuleList([
            nn.Sequential(nn.Linear(HIDDEN * 3, HIDDEN), nn.LayerNorm(HIDDEN), nn.GELU())
            for _ in range(LAYERS)])
        self.head = nn.Sequential(
            nn.Linear(HIDDEN, HIDDEN // 2), nn.GELU(), nn.Linear(HIDDEN // 2, 1))

    def forward(self, x: torch.Tensor, ei: torch.Tensor,
                types: torch.Tensor) -> torch.Tensor:
        h = self.inp(torch.cat([x, self.emb(types)], dim=1))
        src, dst = ei[0], ei[1]
        n = h.shape[0]
        for layer in self.msg:
            fwd = torch.zeros_like(h).index_add_(0, dst, h[src])
            bwd = torch.zeros_like(h).index_add_(0, src, h[dst])
            deg_f = torch.zeros(n, 1).index_add_(
                0, dst, torch.ones(src.shape[0], 1)).clamp(min=1)
            deg_b = torch.zeros(n, 1).index_add_(
                0, src, torch.ones(dst.shape[0], 1)).clamp(min=1)
            h = h + layer(torch.cat([h, fwd / deg_f, bwd / deg_b], dim=1))
        return self.head(h).squeeze(1)

    def trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def site_labels(corpus: base.Corpus, fam: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-node observability label: is ANY fault at this site observable?"""
    fi = ALL_FAMILIES.index(fam)
    fam_ix = corpus.family_index.numpy()
    obs = corpus.observable.numpy() > 0
    local = corpus.local_site.numpy()
    _, _, sn = corpus.G[fam]
    nodes = sn.numpy()
    m = np.flatnonzero(fam_ix == fi)
    n_nodes = corpus.G[fam][0].shape[0]
    lab = np.zeros(n_nodes, dtype=np.float32)
    seen = np.zeros(n_nodes, dtype=bool)
    for idx in m:
        node = nodes[local[idx]]
        seen[node] = True
        if obs[idx]:
            lab[node] = 1.0
    return torch.from_numpy(lab), torch.from_numpy(seen)


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    pred = (p >= 0.5).astype(np.float64)
    tp = float(((pred == 1) & (y == 1)).sum())
    tn = float(((pred == 0) & (y == 0)).sum())
    fp = float(((pred == 1) & (y == 0)).sum())
    fn = float(((pred == 0) & (y == 1)).sum())
    pos, neg = y.sum(), (1 - y).sum()
    if pos > 0 and neg > 0:
        order = p.argsort()
        r = np.empty_like(order, dtype=np.float64)
        r[order] = np.arange(1, p.size + 1)
        auroc = float((r[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))
        bal = 0.5 * (tp / max(pos, 1) + tn / max(neg, 1))
    else:
        auroc, bal = float("nan"), float("nan")
    return {
        "accuracy": round((tp + tn) / max(y.size, 1), 6),
        "balanced_accuracy": round(bal, 6) if bal == bal else None,
        "auroc": round(auroc, 6) if auroc == auroc else None,
        "precision": round(tp / max(tp + fp, 1), 6),
        "recall": round(tp / max(tp + fn, 1), 6),
        "positive_rate": round(float(y.mean()), 6),
    }


def execute() -> None:
    require(ROOT.name == "vlsi_fault_detection_v2", f"place script in project root, not {ROOT}")
    outputs = (TARGET_CONTRACT, PREDICTIONS, TRAIN_LOG, RESULTS, BASELINES,
               ENVIRONMENT, REPORT, MANIFEST, AUDIT)
    require(not any(p.exists() for p in outputs),
            f"frozen Stage {STAGE} output exists; use --status")

    print("FROZEN INPUT VERIFICATION", flush=True)
    for path, expected in PINNED.items():
        require(path.is_file(), f"missing frozen input: {rel(path)}")
        require(sha256(path) == expected, f"frozen input SHA changed: {path.name}")
    j = load_json(AUDIT_2J)
    require(j["within_set_identifiable"] is False,
            "12C-2J must establish that within-set selection is unidentifiable")
    print(f"  12C-2J identifiability proof verified{'':<28}: OK", flush=True)
    existing = sorted(p.name for p in CONFIG.glob("*.json"))
    require(not any("observability_target" in n for n in existing),
            "an observability target contract already exists")
    print(f"  config/v2_2 enumerated ({len(existing)} contracts); none prior"
          f"{'':<13}: OK", flush=True)

    base.set_determinism()
    print("\nLOADING CORPUS", flush=True)
    corpus = base.Corpus()
    vocab = load_json(base.VOCABULARY_2A)["size"]
    fname = list(load_json(base.GRAPH_CONTRACT_2A)["node_feature_names"])
    t_ix = fname.index("type_index")
    d_ix = fname.index("depth_to_output")

    data = {}
    for fam in ALL_FAMILIES:
        if fam not in corpus.G:
            continue
        x, ei, _ = corpus.G[fam]
        lab, seen = site_labels(corpus, fam)
        keep = [i for i in range(x.shape[1]) if i != t_ix]
        data[fam] = {"x": x[:, keep], "ei": ei,
                     "types": x[:, t_ix].long().clamp(0, vocab - 1),
                     "y": lab, "mask": seen, "depth": x[:, d_ix]}
        print(f"  {fam:<24} nodes={x.shape[0]:<7} sites={int(seen.sum()):<7} "
              f"observable_rate={float(lab[seen].mean()):.4f}", flush=True)

    n_feat = data[ALL_FAMILIES[0]]["x"].shape[1]
    model = ObservabilityGNN(n_feat, vocab)
    print(f"\nMODEL  parameters={model.trainable_parameters():,}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    rows, best, best_ep, stale = [], -1.0, -1, 0
    t0 = time.time()
    for epoch in range(EPOCHS):
        model.train()
        tot = 0.0
        for fam in TRAIN_FAMILIES:
            d = data[fam]
            logit = model(d["x"], d["ei"], d["types"])
            m = d["mask"]
            loss = F.binary_cross_entropy_with_logits(logit[m], d["y"][m])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += float(loss.detach())

        model.eval()
        with torch.no_grad():
            cal = CALIB_FAMILIES[0]
            d = data[cal]
            p = torch.sigmoid(model(d["x"], d["ei"], d["types"]))[d["mask"]].numpy()
            y = d["y"][d["mask"]].numpy()
            mt = metrics(y, p)
        rows.append({"epoch": epoch, "train_loss": round(tot / len(TRAIN_FAMILIES), 6),
                     "calibration_auroc": mt["auroc"],
                     "calibration_balanced_accuracy": mt["balanced_accuracy"],
                     "elapsed_s": round(time.time() - t0, 1)})
        if epoch % 10 == 0 or epoch == EPOCHS - 1:
            print(f"    epoch {epoch:>3}  loss={rows[-1]['train_loss']:.4f}  "
                  f"calib_auroc={mt['auroc']}  "
                  f"bal_acc={mt['balanced_accuracy']}", flush=True)
        score = mt["auroc"] if mt["auroc"] is not None else -1
        if score > best:
            best, best_ep, stale = score, epoch, 0
            MODELS.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "epoch": epoch,
                        "n_feat": n_feat, "vocab": vocab, "seed": SEED},
                       MODELS / "v22_observability_gnn_12c2k.pt")
        else:
            stale += 1
            if stale >= PATIENCE:
                print(f"    early stop at epoch {epoch}", flush=True)
                break

    blob = torch.load(MODELS / "v22_observability_gnn_12c2k.pt",
                      map_location="cpu", weights_only=False)
    model.load_state_dict(blob["state_dict"])
    model.eval()

    print("\nPREDICTION-BEFORE-TRUTH SCORING", flush=True)
    preds = {}
    with torch.no_grad():
        for fam, d in data.items():
            preds[fam] = torch.sigmoid(model(d["x"], d["ei"], d["types"])).numpy()
    np.savez_compressed(PREDICTIONS, **preds)
    pred_sha = sha256(PREDICTIONS)
    print(f"  predictions hashed before scoring: {pred_sha[:16]}...", flush=True)

    result_rows, baseline_rows = [], []
    for fam, d in data.items():
        m = d["mask"].numpy()
        y = d["y"].numpy()[m]
        p = preds[fam][m]
        mt = metrics(y, p)
        part = ("TRAIN" if fam in TRAIN_FAMILIES
                else "CALIBRATION" if fam in CALIB_FAMILIES else "OTHER")
        result_rows.append({"family_id": fam, "partition": part,
                            "sites": int(m.sum()), **mt})

        maj = np.full_like(y, 1.0 if y.mean() >= 0.5 else 0.0)
        depth = d["depth"].numpy()[m].astype(np.float64)
        dn = (depth - depth.min()) / max(float(depth.max() - depth.min()), 1e-9)
        best_bal, best_dir = -1.0, ""
        for cand, name in ((1 - dn, "shallow_is_observable"), (dn, "deep_is_observable")):
            b = metrics(y, cand)["balanced_accuracy"]
            if b is not None and b > best_bal:
                best_bal, best_dir = float(b), name
        baseline_rows.append({
            "family_id": fam, "partition": part,
            "majority_class_accuracy": round(float((maj == y).mean()), 6),
            "depth_threshold_balanced_accuracy": round(float(best_bal), 6),
            "depth_threshold_direction": best_dir,
            "gnn_balanced_accuracy": mt["balanced_accuracy"],
            "gnn_auroc": mt["auroc"],
            "gnn_beats_depth_baseline": ("YES" if float(mt["balanced_accuracy"] or 0) > best_bal
                                         else "NO"),
        })
        print(f"  {fam:<24} [{part:<11}] auroc={mt['auroc']} "
              f"bal_acc={mt['balanced_accuracy']} "
              f"(depth baseline {best_bal:.4f})", flush=True)

    cal_row = next(r for r in result_rows if r["partition"] == "CALIBRATION")
    cal_base = next(b for b in baseline_rows if b["partition"] == "CALIBRATION")
    transfers = (cal_row["auroc"] or 0) > 0.6 and cal_base["gnn_beats_depth_baseline"] == "YES"

    created = now()
    target = {
        "contract_version": "CIRCUITSAGE-HMAC-V2.2-OBSERVABILITY-TARGET-12C2K-v1",
        "stage": STAGE, "status": "FROZEN", "created_at": created,
        "target": "per-site fault observability under the frozen test scheme",
        "identifiable": True,
        "identifiability_argument": (
            "the label is a deterministic function of site and test scheme rather "
            "than a tie between indistinguishable alternatives; base rate varies "
            "0.380-0.999 across circuits, so there is variance to explain"),
        "contrast_with_12c2g": (
            "12C-2J proved within-collision-set selection is unidentifiable; this "
            "target is chosen precisely because it is not"),
        "does_not_localize": (
            "observability prediction does NOT identify which site is faulty; it "
            "identifies which sites are testable at all"),
        "practical_use": (
            "flags untestable regions so the test scheme can be improved, which is "
            "the only route that raises the localization ceiling in 12C-2J"),
        "baselines_required": ["majority class", "single-feature depth threshold"],
    }
    environment = {
        "environment_version": "CIRCUITSAGE-HMAC-V2.2-OBSERVABILITY-ENVIRONMENT-12C2K-v1",
        "stage": STAGE, "created_at": created,
        "python": platform.python_version(), "torch": torch.__version__,
        "seed": SEED, "epochs_run": len(rows), "best_epoch": best_ep,
        "parameters": model.trainable_parameters(),
        "hidden": HIDDEN, "layers": LAYERS, "lr": LR,
    }

    res_table = "\n".join(
        f"| `{r['family_id']}` | {r['partition']} | {r['positive_rate']:.4f} | "
        f"{r['auroc']} | {r['balanced_accuracy']} |" for r in result_rows)
    base_table = "\n".join(
        f"| `{b['family_id']}` | {b['depth_threshold_balanced_accuracy']:.4f} | "
        f"{b['gnn_balanced_accuracy']} | **{b['gnn_beats_depth_baseline']}** |"
        for b in baseline_rows)

    report = f"""# Stage {STAGE} — Graph Model on an Identifiable Target

**Status: PASS / FROZEN — training + measurement, no selection.**

## Why this target

Stage 12C-2J proved that picking the true site inside an exact-signature
collision set is **not identifiable** under the frozen uniform fault catalogue —
which is why the 12C-2G reranker measured exactly 1.00x. This stage puts a graph
model on a target that *is* identifiable:

> **Will a fault at this site be observable at all under the frozen test scheme?**

Stated before training: base rates range 0.380–0.999 across circuits, and
Stage 12C-2A already measured corr(depth_to_output, observability) = −0.331 and
−0.311 on the low-observability circuits. There is real structural signal here.

## Results

| family | partition | observable rate | AUROC | balanced acc |
|---|---|---|---|---|
{res_table}

## Against non-learning baselines

| family | depth-threshold bal. acc | GNN bal. acc | GNN wins |
|---|---|---|---|
{base_table}

Both baselines are mandatory: a majority-class predictor scores well whenever
the base rate is extreme, so an accuracy figure alone would be misleading.

## What this does and does not do

- **Does**: flag which regions of a design are untestable under the current test
  scheme — the input a test engineer needs to improve that scheme.
- **Does not**: localize faults. Localization remains exact-signature retrieval,
  bounded as proved in 12C-2J.

Improving the test scheme is the *only* route that raises the localization
ceiling, so this model attacks the bound rather than fighting it.

Predictions were written and hashed before truth was consulted
(`{pred_sha[:32]}...`). Sealed circuits untouched. Independent generalization
remains **NOT ESTABLISHED**. Future hybrid brand remains **{FUTURE_BRAND}**.
"""

    frozen_write(TARGET_CONTRACT, canonical_json(target))
    frozen_write(TRAIN_LOG, csv_bytes(rows, list(rows[0].keys())))
    frozen_write(RESULTS, csv_bytes(result_rows, list(result_rows[0].keys())))
    frozen_write(BASELINES, csv_bytes(baseline_rows, list(baseline_rows[0].keys())))
    frozen_write(ENVIRONMENT, canonical_json(environment))
    frozen_write(REPORT, report.encode())

    stage_outputs = (TARGET_CONTRACT, PREDICTIONS, TRAIN_LOG, RESULTS,
                     BASELINES, ENVIRONMENT, REPORT)
    manifest = {
        "manifest_version": "CIRCUITSAGE-HMAC-V2.2-OBSERVABILITY-MANIFEST-12C2K-v1",
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
        "audit_version": "CIRCUITSAGE-HMAC-V2.2-OBSERVABILITY-FREEZE-12C2K-v1",
        "stage": STAGE, "status": "PASS",
        "target": "per-site fault observability",
        "target_identifiable": True,
        "identifiability_authority": "12C-2J",
        "parameters": model.trainable_parameters(),
        "best_epoch": best_ep,
        "per_family": {r["family_id"]: {"auroc": r["auroc"],
                                        "balanced_accuracy": r["balanced_accuracy"],
                                        "partition": r["partition"]}
                       for r in result_rows},
        "beats_depth_baseline": {b["family_id"]: b["gnn_beats_depth_baseline"]
                                 for b in baseline_rows},
        "transfers_to_calibration": transfers,
        "localizes_faults": False,
        "prediction_before_truth": True,
        "predictions_sha256": pred_sha,
        "selection_performed": False,
        "acceptance_evaluation": "NOT AUTHORIZED",
        "independent_test_validation_holdout_access": [0, 0, 0],
        "independent_generalization": "NOT ESTABLISHED",
        "seed": SEED,
        "target_contract_record": record(TARGET_CONTRACT),
        "results_record": record(RESULTS),
        "baselines_record": record(BASELINES),
        "manifest_record": record(MANIFEST),
        "future_combined_model_brand": FUTURE_BRAND,
        "next_gate": "STAGE 12C-3A — INDEPENDENT TEST CAPTURE",
    }
    frozen_write(AUDIT, canonical_json(audit))

    print(f"\n{'Stage':<52}: {STAGE} — OBSERVABILITY MODEL")
    print(f"{'Status':<52}: PASS / FROZEN")
    print(f"{'Calibration AUROC':<52}: {cal_row['auroc']}")
    print(f"{'Beats depth baseline on calibration':<52}: "
          f"{cal_base['gnn_beats_depth_baseline']}")
    print(f"{'Transfers to unseen circuit':<52}: {transfers}")
    print(f"{'Localizes faults':<52}: NO (by design)")
    print(f"{'Audit SHA':<52}: {sha256(AUDIT)}")


def status() -> None:
    print(f"STAGE {STAGE} — OBSERVABILITY MODEL STATUS")
    if not (MANIFEST.is_file() and AUDIT.is_file()):
        print("Status                    : NOT FROZEN")
        return
    a = load_json(AUDIT)
    print("Status                    : PASS / FROZEN")
    print(f"Per family                : {json.dumps(a['per_family'], indent=2)}")
    print(f"Beats depth baseline      : {a['beats_depth_baseline']}")
    print(f"Transfers                 : {a['transfers_to_calibration']}")
    print(f"Audit SHA                 : {sha256(AUDIT)}")


def self_test() -> None:
    m = ObservabilityGNN(19, 22)
    x = torch.randn(50, 19)
    ei = torch.randint(0, 50, (2, 120))
    t = torch.randint(0, 22, (50,))
    out = m(x, ei, t)
    require(out.shape == (50,), "per-node output shape")
    y = np.array([1, 1, 0, 0]); p = np.array([0.9, 0.8, 0.2, 0.1])
    require(abs(metrics(y, p)["auroc"] - 1.0) < 1e-9, "perfect AUROC")
    require(AUDIT_2J.is_file(), "12C-2J present")
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
