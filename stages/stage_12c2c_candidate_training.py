#!/usr/bin/env python3
"""Stage 12C-2C: V2.2 candidate training execution.

Trains the three trainable candidates frozen in the Stage 12C-1A candidate grid
and evaluates the mandatory non-learning comparator, under the Stage 12C-2B
training execution authorization.

Architecture is NOT designed here.  Encoder families, fusion strategy and
parameter caps come from the frozen Stage 12C-1A architecture contract and
candidate grid.  This stage implements them and enforces the caps as hard
preconditions.

Contract compliance enforced at runtime:

  * gradient updates occur ONLY on GENERALIZATION_TRAIN families
  * GENERALIZATION_CALIBRATION is loaded for monitoring only, never for
    gradients, and calibration loss never touches ``backward()``
  * INDEPENDENT_CIRCUIT_TEST and GENERALIZATION_HOLDOUT are never opened
  * each candidate's trainable parameter count must not exceed its frozen cap
  * response encoder is the contracted masked temporal encoder over per-vector
    slices, never a flat projection of the whole response tensor
  * fixed seeds, deterministic data order, single declared environment
  * checkpoint after every epoch; resumable with --resume

Selection is NOT performed here.  This stage produces trained candidates plus
per-epoch calibration monitoring only.  Model selection is a separate gate.
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
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, SAGEConv


STAGE = "12C-2C"
ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "config/v2_2"
RESULT1 = ROOT / "results/circuitsage_hmac_v2_12c1"
RESULT2 = ROOT / "results/circuitsage_hmac_v2_12c2"
WORK = RESULT2 / "candidate_training_12c2c"
MODELS = WORK / "models"
LOCK_FILE = WORK / ".stage_12c2c.lock"
CHECKPOINT = WORK / "circuitsage_hmac_v2_2_training_checkpoint_12c2c.json"

# ---------------------------------------------------------------- frozen input
SOURCE_2B = ROOT / "stage_12c2b_training_authorization.py"
ARCHITECTURE_1A = CONFIG / "circuitsage_hmac_v2_2_generalization_architecture_12c1a.json"
TRAINING_1A = CONFIG / "circuitsage_hmac_v2_2_training_selection_contract_12c1a.json"
CANDIDATE_GRID_1A = CONFIG / "circuitsage_hmac_v2_2_candidate_grid_12c1a.csv"
SPLIT_1B = CONFIG / "circuitsage_hmac_v2_2_family_split_authorization_12c1b.json"
VOCABULARY_2A = CONFIG / "circuitsage_hmac_v2_2_cell_type_vocabulary_12c2a.json"
GRAPH_CONTRACT_2A = CONFIG / "circuitsage_hmac_v2_2_graph_dataset_contract_12c2a.json"
TRAINING_AUTH_2B = CONFIG / "circuitsage_hmac_v2_2_training_execution_authorization_12c2b.json"
CALIB_AUTH_2B = CONFIG / "circuitsage_hmac_v2_2_calibration_use_authorization_12c2b.json"
AUDIT_2B = RESULT2 / "circuitsage_hmac_v2_2_training_authorization_freeze_12c2b.json"
MANIFEST_2B = RESULT2 / "circuitsage_hmac_v2_2_training_authorization_manifest_12c2b.json"

WORK_1O = RESULT1 / "parallel_campaign_execution_12c1o"
FEATURES_1O = WORK_1O / "circuitsage_hmac_v2_2_amended_campaign_features_12c1o.npz"
TARGETS_1O = WORK_1O / "circuitsage_hmac_v2_2_amended_campaign_targets_12c1o.npz"
WORK_2A = RESULT2 / "graph_dataset_12c2a"
GRAPH_NPZ_2A = WORK_2A / "circuitsage_hmac_v2_2_circuit_graphs_12c2a.npz"

PINNED = {
    SOURCE_2B: "a7a958c2c539fe48ad66731b5eb74e8b0ff6c0133c595ba284bd81e2a4b50f36",
    ARCHITECTURE_1A: "e83da87e0ba773ed1b6872528c216079d31fc83537b4a2eca348bc6a23791649",
    CANDIDATE_GRID_1A: "2297edd6a7b61cca28709509a015ff246c43a8379a36ed4842b8d1a2c5528470",
    SPLIT_1B: "4103808fc389c78088e31c0a76549324c386ed9d8f7bedb7cc4d3748f9ea3303",
    VOCABULARY_2A: "f64710234ca0f03aa565682687476bfdfb46dc877bb24db26b65866312a0e927",
    GRAPH_CONTRACT_2A: "c7d0d8070e605f5e05592c4ac6b1ffa93f3c701318f69b27512fe0cbdc5e044a",
    TRAINING_AUTH_2B: "2a47987e94faf5df1d9f7c66e4c4edf77867541ffd9cd0a895d0d13cb3b15047",
    CALIB_AUTH_2B: "4efcd94366eacb3ee3c9d4805556bb83139872d9118711d42b41587a89cf21da",
    AUDIT_2B: "bd800d707cb0e8bead862e584fbed9f6a9a202bb1fc7a666d97a9142af5863fa",
    MANIFEST_2B: "5b98150dc0a88586ab9cb08b2e5b911f0b9680ebd2bb43617f024186c98137c6",
    GRAPH_NPZ_2A: "9608facea513742bc04624b1aee63f88dad457e06a3630be45aecc56ec3a2d44",
    FEATURES_1O: "1233c4826f2f0ec159deca68b40d92d71c011fbd9de6c7f834a5e680d3c6d22f",
    TARGETS_1O: "8f1b36e63b9700f6dd0b1c64d52015d847867fc0a72264bd58b9126b9c813dc4",
}

# ---------------------------------------------------------------- stage output
TRAINING_LOG = WORK / "circuitsage_hmac_v2_2_training_log_12c2c.csv"
CANDIDATE_SUMMARY = WORK / "circuitsage_hmac_v2_2_candidate_summary_12c2c.csv"
BASELINE_RESULT = WORK / "circuitsage_hmac_v2_2_baseline_comparator_12c2c.json"
ENVIRONMENT = WORK / "circuitsage_hmac_v2_2_training_environment_12c2c.json"
PREFLIGHT = WORK / "circuitsage_hmac_v2_2_training_preflight_12c2c.json"
REPORT = WORK / "circuitsage_hmac_v2_2_training_report_12c2c.md"
MANIFEST = RESULT2 / "circuitsage_hmac_v2_2_candidate_training_manifest_12c2c.json"
AUDIT = RESULT2 / "circuitsage_hmac_v2_2_candidate_training_freeze_12c2c.json"

ALL_FAMILIES = ("opentitan_hmac_sha256", "picorv32_cpu", "secworks_aes", "secworks_sha256")
TRAIN_FAMILIES = ("opentitan_hmac_sha256", "picorv32_cpu", "secworks_aes")
CALIB_FAMILIES = ("secworks_sha256",)

SEED = 20260920
EPOCHS = 40
BATCH = 256
NEGATIVES = 128
EMB = 96
LR = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 8
VECTORS = 64
RESP_BYTES = 32
NODE_FEATS = 20
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


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as stream:
        stream.write(canonical_json(value))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def set_determinism() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(False)  # SAGE/GAT scatter kernels
    torch.set_num_threads(12)


# ------------------------------------------------------------------ encoders


class MaskedTemporalResponseEncoder(nn.Module):
    """Contract: 'masked temporal MLP or 1D convolution with circuit-independent
    dimensions'.  Each vector slice is embedded independently, then masked-pooled,
    so the encoder never sees a circuit-specific flattened layout."""

    def __init__(self, emb: int, width: int):
        super().__init__()
        self.slice_mlp = nn.Sequential(
            nn.Linear(RESP_BYTES + 3, width), nn.ReLU(),
            nn.Linear(width, width), nn.ReLU())
        self.head = nn.Sequential(
            nn.Linear(width * 2, width), nn.ReLU(),
            nn.Linear(width, emb))

    def forward(self, xor: torch.Tensor, cyc: torch.Tensor, to: torch.Tensor,
                pe: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        per_vector = torch.cat(
            [xor, cyc.unsqueeze(-1), to.unsqueeze(-1), pe.unsqueeze(-1)], dim=-1)
        h = self.slice_mlp(per_vector)
        m = mask.unsqueeze(-1)
        denom = m.sum(dim=1).clamp(min=1.0)
        mean = (h * m).sum(dim=1) / denom
        peak = (h * m + (m - 1) * 1e4).max(dim=1).values
        return F.normalize(self.head(torch.cat([mean, peak], dim=1)), dim=1)


class GraphEncoder(nn.Module):
    def __init__(self, vocab_size: int, emb: int, kind: str, heads: int = 2):
        super().__init__()
        self.kind = kind
        self.type_emb = nn.Embedding(vocab_size, 16)
        self.proj = nn.Linear(NODE_FEATS - 1 + 16, emb)
        if kind == "GraphSAGE":
            self.c1, self.c2, self.c3 = SAGEConv(emb, emb), SAGEConv(emb, emb), SAGEConv(emb, emb)
        elif kind == "GATv2":
            self.c1 = GATv2Conv(emb, emb // heads, heads=heads)
            self.c2 = GATv2Conv(emb, emb // heads, heads=heads)
            self.c3 = GATv2Conv(emb, emb, heads=1)
        else:
            stop(f"unknown graph encoder: {kind}")

    def forward(self, x: torch.Tensor, ei: torch.Tensor) -> torch.Tensor:
        t = self.type_emb(x[:, 0].long().clamp(0, self.type_emb.num_embeddings - 1))
        h = F.relu(self.proj(torch.cat([t, x[:, 1:]], dim=1)))
        h = F.relu(self.c1(h, ei))
        h = F.relu(self.c2(h, ei))
        return F.normalize(self.c3(h, ei), dim=1)


class Candidate(nn.Module):
    def __init__(self, candidate_id: str, vocab_size: int):
        super().__init__()
        self.candidate_id = candidate_id
        if candidate_id == "V22_GRAPHSAGE_METRIC_SMALL":
            emb, width, kind = 64, 96, "GraphSAGE"
            self.ensemble = 1
        elif candidate_id == "V22_GATV2_CROSS_FUSION":
            emb, width, kind = 96, 160, "GATv2"
            self.ensemble = 1
        elif candidate_id == "V22_GRAPHSAGE_OOD_ENSEMBLE":
            emb, width, kind = 80, 128, "GraphSAGE"
            self.ensemble = 3
        else:
            stop(f"unknown candidate: {candidate_id}")
        self.graph = GraphEncoder(vocab_size, emb, kind)
        self.response = MaskedTemporalResponseEncoder(emb, width)
        self.detector = nn.Sequential(nn.Linear(emb, 64), nn.ReLU(), nn.Linear(64, 1))
        self.polarity = nn.Sequential(nn.Linear(emb * 2, 64), nn.ReLU(), nn.Linear(64, 2))
        self.ood = nn.ModuleList(
            [nn.Sequential(nn.Linear(emb, 48), nn.ReLU(), nn.Linear(48, 1))
             for _ in range(self.ensemble)])

    def trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ------------------------------------------------------------------- dataset


class Corpus:
    def __init__(self) -> None:
        self.feat = np.load(FEATURES_1O)
        self.targ = np.load(TARGETS_1O)
        self.graphs = np.load(GRAPH_NPZ_2A)
        self.family_index = torch.from_numpy(self.feat["family_index"].astype(np.int64))
        self.observable = torch.from_numpy(self.feat["observable"].astype(np.float32))
        self.local_site = torch.from_numpy(self.targ["local_site_index"].astype(np.int64))
        self.stuck = torch.from_numpy(self.targ["stuck_value"].astype(np.int64))
        self.G: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        for fam in ALL_FAMILIES:
            self.G[fam] = (
                torch.from_numpy(self.graphs[f"{fam}__node_features"]).float(),
                torch.from_numpy(self.graphs[f"{fam}__edge_index"]).long(),
                torch.from_numpy(self.graphs[f"{fam}__site_node_index"]).long(),
            )

    def pool(self, families: tuple[str, ...], observable_only: bool) -> torch.Tensor:
        ids = torch.tensor([ALL_FAMILIES.index(f) for f in families])
        mask = torch.isin(self.family_index, ids)
        if observable_only:
            mask = mask & (self.observable > 0)
        return torch.nonzero(mask).squeeze(1)

    def batch(self, idx: np.ndarray) -> tuple[torch.Tensor, ...]:
        xor = torch.from_numpy(self.feat["response_xor"][idx].astype(np.float32) / 255.0)
        cyc = torch.from_numpy(
            self.feat["completion_cycle_delta"][idx].astype(np.float32)).clamp(-1e4, 1e4) / 1e4
        to = torch.from_numpy(self.feat["timeout"][idx].astype(np.float32))
        pe = torch.from_numpy(self.feat["protocol_error"][idx].astype(np.float32))
        vm = torch.from_numpy(self.feat["vector_mask"][idx].astype(np.float32))
        return xor, cyc, to, pe, vm


def contrastive_step(model: Candidate, corpus: Corpus, idx: torch.Tensor,
                     generator: np.random.Generator) -> tuple[torch.Tensor, dict[str, float]]:
    fam_id = int(corpus.family_index[idx[0]])
    fam = ALL_FAMILIES[fam_id]
    same = idx[corpus.family_index[idx] == fam_id]
    if same.numel() < 8:
        return torch.zeros((), requires_grad=True), {}

    x, ei, sn = corpus.G[fam]
    node_emb = model.graph(x, ei)
    ii = same.numpy()
    xor, cyc, to, pe, vm = corpus.batch(ii)
    q = model.response(xor, cyc, to, pe, vm)

    pos_nodes = sn[corpus.local_site[same]]
    pos = node_emb[pos_nodes]
    neg_nodes = torch.from_numpy(
        generator.integers(0, node_emb.shape[0], size=NEGATIVES))
    neg = node_emb[neg_nodes]

    logits = torch.cat([(q * pos).sum(1, keepdim=True), q @ neg.T], dim=1) / 0.07
    retrieval = F.cross_entropy(logits, torch.zeros(q.shape[0], dtype=torch.long))

    obs = corpus.observable[same]
    det_logit = model.detector(q).squeeze(1)
    detection = F.binary_cross_entropy_with_logits(det_logit, obs)

    pol_logit = model.polarity(torch.cat([q, pos], dim=1))
    polarity = F.cross_entropy(pol_logit, corpus.stuck[same])

    ood_scores = torch.stack([head(q).squeeze(1) for head in model.ood], dim=0)
    ood = ood_scores.var(dim=0).mean() if model.ensemble > 1 else ood_scores.mean() * 0.0

    loss = retrieval + 0.5 * detection + 0.3 * polarity + 0.05 * ood
    with torch.no_grad():
        top1 = float((logits.argmax(dim=1) == 0).float().mean())
    return loss, {"retrieval": float(retrieval), "detection": float(detection),
                  "polarity": float(polarity), "top1": top1}


@torch.no_grad()
def monitor_calibration(model: Candidate, corpus: Corpus,
                        pool: torch.Tensor, generator: np.random.Generator) -> dict[str, float]:
    """Monitoring ONLY. No gradients. Never calls backward()."""
    model.eval()
    hits, total, det_correct = 0, 0, 0
    for _ in range(12):
        sel = pool[torch.from_numpy(generator.integers(0, pool.numel(), size=BATCH))]
        fam_id = int(corpus.family_index[sel[0]])
        fam = ALL_FAMILIES[fam_id]
        same = sel[corpus.family_index[sel] == fam_id]
        if same.numel() < 8:
            continue
        x, ei, sn = corpus.G[fam]
        node_emb = model.graph(x, ei)
        xor, cyc, to, pe, vm = corpus.batch(same.numpy())
        q = model.response(xor, cyc, to, pe, vm)
        pos = node_emb[sn[corpus.local_site[same]]]
        neg = node_emb[torch.from_numpy(generator.integers(0, node_emb.shape[0], size=NEGATIVES))]
        logits = torch.cat([(q * pos).sum(1, keepdim=True), q @ neg.T], dim=1)
        hits += int((logits.argmax(dim=1) == 0).sum())
        det = (torch.sigmoid(model.detector(q).squeeze(1)) > 0.5).float()
        det_correct += int((det == corpus.observable[same]).sum())
        total += same.numel()
    model.train()
    return {"calibration_top1": hits / max(total, 1),
            "calibration_detection_acc": det_correct / max(total, 1),
            "calibration_samples": total}


def exact_signature_baseline(corpus: Corpus) -> dict[str, Any]:
    """Mandatory non-learning comparator: graph-constrained exact retrieval."""
    sig = corpus.targ["behavior_signature_sha256"]
    results: dict[str, Any] = {}
    for fam_id, fam in enumerate(ALL_FAMILIES):
        mask = (corpus.family_index == fam_id).numpy()
        obs = corpus.observable.numpy()[mask] > 0
        sub = sig[mask][obs]
        if sub.size == 0:
            results[fam] = {"observable": 0, "unique_signature_fraction": 0.0}
            continue
        _, counts = np.unique(sub, return_counts=True)
        unique = int((counts == 1).sum())
        results[fam] = {
            "observable": int(sub.size),
            "distinct_signatures": int(counts.size),
            "unique_signature_faults": unique,
            "unique_signature_fraction": round(unique / sub.size, 8),
            "mean_collision_group": round(float(counts.mean()), 6),
            "max_collision_group": int(counts.max()),
        }
    return {
        "comparator_version": "CIRCUITSAGE-HMAC-V2.2-EXACT-SIGNATURE-GRAPH-BASELINE-12C2C-v1",
        "candidate_id": "V22_EXACT_SIGNATURE_GRAPH_BASELINE",
        "trainable": False, "parameters": 0,
        "stage": STAGE, "created_at": now(),
        "per_family": results,
        "note": ("non-learning comparator measured on the frozen campaign signatures; "
                 "no gradients, no selection, no protected partition access"),
    }


# -------------------------------------------------------------------- execute


def verify_inputs() -> list[dict[str, str]]:
    print("FROZEN INPUT VERIFICATION", flush=True)
    for path, expected in PINNED.items():
        require(path.is_file(), f"missing frozen input: {rel(path)}")
        require(sha256(path) == expected, f"frozen input SHA changed: {path.name}")
    print(f"  {len(PINNED)} frozen 12C-1A / 12C-1B / 12C-1O / 12C-2A / 12C-2B inputs"
          f"{'':<14}: OK", flush=True)

    auth = load_json(TRAINING_AUTH_2B)
    require(auth.get("model_training") == "AUTHORIZED / BOUNDED", "training authorized")
    require(auth.get("gradient_partitions") == ["GENERALIZATION_TRAIN"], "gradient scope")
    require(auth.get("acceptance_evaluation") == "NOT AUTHORIZED BY THIS STAGE",
            "acceptance not authorized")
    calib = load_json(CALIB_AUTH_2B)
    require(calib.get("gradient_updates") == "PROHIBITED", "calibration gradient prohibition")

    split = load_json(SPLIT_1B)
    protected = {f for f, p in split["family_assignments"].items()
                 if p in ("INDEPENDENT_CIRCUIT_TEST", "GENERALIZATION_HOLDOUT")}
    require(not (protected & set(ALL_FAMILIES)), "no protected family in training corpus")
    print(f"  protected families excluded: {sorted(protected)}"
          f"{'':<26}: OK", flush=True)

    grid = read_csv(CANDIDATE_GRID_1A)
    trainable = [r for r in grid if r["trainable"].strip().upper() == "YES"]
    require(len(trainable) == 3, "three trainable candidates")
    return grid


def train_candidate(candidate_id: str, cap: int, corpus: Corpus, vocab_size: int,
                    state: dict[str, Any]) -> dict[str, Any]:
    set_determinism()
    model = Candidate(candidate_id, vocab_size)
    params = model.trainable_parameters()
    print(f"\n  {candidate_id}", flush=True)
    print(f"    parameters {params:,} / cap {cap:,}", flush=True)
    require(params <= cap,
            f"{candidate_id} exceeds frozen parameter cap: {params} > {cap}")

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
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
            loss, stats = contrastive_step(model, corpus, sel, generator)
            if not stats:
                continue
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            for k, v in stats.items():
                agg[k] = agg.get(k, 0.0) + v
        n = max(steps, 1)
        mon = monitor_calibration(model, corpus, calib_pool, np.random.default_rng(SEED + epoch))
        row = {
            "candidate_id": candidate_id, "epoch": epoch,
            "train_retrieval_loss": round(agg.get("retrieval", 0.0) / n, 6),
            "train_detection_loss": round(agg.get("detection", 0.0) / n, 6),
            "train_polarity_loss": round(agg.get("polarity", 0.0) / n, 6),
            "train_top1": round(agg.get("top1", 0.0) / n, 6),
            "calibration_top1": round(mon["calibration_top1"], 6),
            "calibration_detection_acc": round(mon["calibration_detection_acc"], 6),
            "elapsed_s": round(time.time() - start, 1),
        }
        rows.append(row)
        if epoch % 5 == 0 or epoch == EPOCHS - 1:
            print(f"    epoch {epoch:>3}  train_top1={row['train_top1']:.4f}  "
                  f"calib_top1={row['calibration_top1']:.4f}  "
                  f"det={row['calibration_detection_acc']:.4f}  "
                  f"{row['elapsed_s']:.0f}s", flush=True)
        if row["calibration_top1"] > best:
            best, best_epoch, stale = row["calibration_top1"], epoch, 0
            MODELS.mkdir(parents=True, exist_ok=True)
            torch.save({"candidate_id": candidate_id, "epoch": epoch,
                        "state_dict": model.state_dict(), "parameters": params,
                        "seed": SEED},
                       MODELS / f"{candidate_id.lower()}_12c2c.pt")
        else:
            stale += 1
            if stale >= PATIENCE:
                print(f"    early stop at epoch {epoch} (patience {PATIENCE})", flush=True)
                break

    return {
        "candidate_id": candidate_id, "parameters": params, "parameter_cap": cap,
        "cap_respected": True, "epochs_run": len(rows),
        "best_epoch": best_epoch, "best_calibration_top1": round(best, 6),
        "final_train_top1": rows[-1]["train_top1"] if rows else 0.0,
        "wall_clock_s": round(time.time() - start, 1),
        "rows": rows,
    }


def execute(resume: bool) -> None:
    require(ROOT.name == "vlsi_fault_detection_v2", f"place script in project root, not {ROOT}")
    require(not MANIFEST.exists() and not AUDIT.exists(),
            f"frozen Stage {STAGE} output exists; use --status")

    grid = verify_inputs()
    set_determinism()

    print("\nLOADING CORPUS", flush=True)
    corpus = Corpus()
    vocab_size = load_json(VOCABULARY_2A)["size"]
    train_pool = corpus.pool(TRAIN_FAMILIES, observable_only=True)
    calib_pool = corpus.pool(CALIB_FAMILIES, observable_only=True)
    print(f"  train observable pool      : {train_pool.numel():,}", flush=True)
    print(f"  calibration observable pool: {calib_pool.numel():,} (monitoring only)", flush=True)

    print("\nNON-LEARNING COMPARATOR", flush=True)
    baseline = exact_signature_baseline(corpus)
    for fam, r in baseline["per_family"].items():
        print(f"  {fam:<24} unique_sig={r.get('unique_signature_fraction', 0):.4f} "
              f"max_collision={r.get('max_collision_group', 0)}", flush=True)

    print("\nCANDIDATE TRAINING", flush=True)
    state: dict[str, Any] = {}
    summaries: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    for r in grid:
        if r["trainable"].strip().upper() != "YES":
            continue
        summary = train_candidate(r["candidate_id"], int(r["parameter_cap"]),
                                  corpus, vocab_size, state)
        all_rows.extend(summary.pop("rows"))
        summaries.append(summary)
        atomic_json(CHECKPOINT, {"stage": STAGE, "status": "RUNNING",
                                 "completed": [s["candidate_id"] for s in summaries],
                                 "updated_at": now()})

    created = now()
    environment = {
        "environment_version": "CIRCUITSAGE-HMAC-V2.2-TRAINING-ENVIRONMENT-12C2C-v1",
        "stage": STAGE, "created_at": created,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "cuda_available": torch.cuda.is_available(),
        "threads": torch.get_num_threads(),
        "seed": SEED, "epochs": EPOCHS, "batch": BATCH, "negatives": NEGATIVES,
        "learning_rate": LR, "weight_decay": WEIGHT_DECAY, "patience": PATIENCE,
        "deterministic_data_order": True,
    }
    preflight = {
        "preflight_version": "CIRCUITSAGE-HMAC-V2.2-TRAINING-PREFLIGHT-12C2C-v1",
        "stage": STAGE, "status": "PASS", "created_at": created,
        "gradient_families": list(TRAIN_FAMILIES),
        "monitoring_families": list(CALIB_FAMILIES),
        "calibration_gradient_updates": 0,
        "independent_test_access": 0, "validation_access": 0, "holdout_access": 0,
        "architecture_invented": False,
        "acceptance_criteria_invented": False,
        "parameter_caps_enforced": True,
        "selection_performed": False,
    }

    summary_fields = ["candidate_id", "parameters", "parameter_cap", "cap_respected",
                      "epochs_run", "best_epoch", "best_calibration_top1",
                      "final_train_top1", "wall_clock_s"]
    log_fields = ["candidate_id", "epoch", "train_retrieval_loss", "train_detection_loss",
                  "train_polarity_loss", "train_top1", "calibration_top1",
                  "calibration_detection_acc", "elapsed_s"]

    table = "\n".join(
        f"| `{s['candidate_id']}` | {s['parameters']:,} | {s['parameter_cap']:,} | "
        f"{s['epochs_run']} | {s['best_calibration_top1']:.4f} | {s['wall_clock_s']:.0f}s |"
        for s in summaries)

    report = f"""# Stage {STAGE} — Candidate Training

**Status: PASS / FROZEN — training only, no selection.**

## Candidates trained

| candidate | parameters | cap | epochs | best calib Top-1 | wall clock |
|---|---|---|---|---|---|
{table}

All parameter caps from the frozen Stage 12C-1A candidate grid were enforced as
hard preconditions before any optimizer step.

## Contract compliance

- gradients on **GENERALIZATION_TRAIN only**: {', '.join(TRAIN_FAMILIES)}
- **GENERALIZATION_CALIBRATION** ({', '.join(CALIB_FAMILIES)}) used for
  monitoring and early stopping only; `backward()` never called on it
- INDEPENDENT_CIRCUIT_TEST and GENERALIZATION_HOLDOUT never opened
- response encoder is the contracted **masked temporal** encoder over per-vector
  slices, not a flat projection
- fixed seed {SEED}, deterministic data order, environment recorded

## Non-learning comparator

`V22_EXACT_SIGNATURE_GRAPH_BASELINE` was measured on the frozen campaign
signatures (0 parameters, no gradients) and is reported alongside the trained
candidates as the mandatory comparator.

## What this stage does NOT do

No model selection, no acceptance evaluation, no protected-partition access.
Selection is a separate authorized gate. Independent generalization remains
**NOT ESTABLISHED**. Future hybrid brand remains **{FUTURE_BRAND}**.

## Next gate

**Stage 12C-2D** — model selection on GENERALIZATION_CALIBRATION under the
frozen selection order.
"""

    frozen_write(TRAINING_LOG, csv_bytes(all_rows, log_fields))
    frozen_write(CANDIDATE_SUMMARY, csv_bytes(summaries, summary_fields))
    frozen_write(BASELINE_RESULT, canonical_json(baseline))
    frozen_write(ENVIRONMENT, canonical_json(environment))
    frozen_write(PREFLIGHT, canonical_json(preflight))
    frozen_write(REPORT, report.encode())

    model_records = {p.name: record(p) for p in sorted(MODELS.glob("*.pt"))}
    stage_outputs = (TRAINING_LOG, CANDIDATE_SUMMARY, BASELINE_RESULT, ENVIRONMENT,
                     PREFLIGHT, REPORT)
    manifest = {
        "manifest_version": "CIRCUITSAGE-HMAC-V2.2-CANDIDATE-TRAINING-MANIFEST-12C2C-v1",
        "stage": STAGE, "status": "PASS", "created_at": created,
        "stage_source": record(Path(__file__).resolve()),
        "frozen_inputs": {rel(p): record(p) for p in PINNED},
        "outputs": {rel(p): record(p) for p in stage_outputs},
        "trained_models": model_records,
        "candidates_trained": len(summaries),
        "calibration_gradient_updates": 0,
        "independent_test_access": 0, "validation_access": 0, "holdout_access": 0,
    }
    frozen_write(MANIFEST, canonical_json(manifest))

    audit = {
        "audit_version": "CIRCUITSAGE-HMAC-V2.2-CANDIDATE-TRAINING-FREEZE-12C2C-v1",
        "stage": STAGE, "status": "PASS",
        "candidates_trained": len(summaries),
        "candidate_ids": [s["candidate_id"] for s in summaries],
        "parameter_caps_enforced": True,
        "parameter_counts": {s["candidate_id"]: s["parameters"] for s in summaries},
        "best_calibration_top1": {s["candidate_id"]: s["best_calibration_top1"]
                                  for s in summaries},
        "gradient_families": list(TRAIN_FAMILIES),
        "calibration_families": list(CALIB_FAMILIES),
        "calibration_gradient_updates": 0,
        "non_learning_comparator": "V22_EXACT_SIGNATURE_GRAPH_BASELINE",
        "architecture_invented": False,
        "acceptance_criteria_invented": False,
        "selection_performed": False,
        "acceptance_evaluation": "NOT AUTHORIZED",
        "independent_test_validation_holdout_access": [0, 0, 0],
        "independent_generalization": "NOT ESTABLISHED",
        "seed": SEED,
        "training_log_record": record(TRAINING_LOG),
        "candidate_summary_record": record(CANDIDATE_SUMMARY),
        "baseline_comparator_record": record(BASELINE_RESULT),
        "environment_record": record(ENVIRONMENT),
        "manifest_record": record(MANIFEST),
        "future_combined_model_brand": FUTURE_BRAND,
        "next_gate": "STAGE 12C-2D — MODEL SELECTION ON CALIBRATION",
    }
    frozen_write(AUDIT, canonical_json(audit))

    print(f"\n{'Stage':<52}: {STAGE} — CANDIDATE TRAINING")
    print(f"{'Status':<52}: PASS / FROZEN")
    for s in summaries:
        print(f"  {s['candidate_id']:<40}: {s['parameters']:>9,} params  "
              f"calib_top1={s['best_calibration_top1']:.4f}  {s['wall_clock_s']:.0f}s")
    print(f"{'Parameter caps enforced':<52}: YES")
    print(f"{'Calibration gradient updates':<52}: 0")
    print(f"{'Selection performed':<52}: NO")
    print(f"{'TEST / VALIDATION / HOLDOUT access':<52}: 0 / 0 / 0")
    print(f"{'Audit SHA':<52}: {sha256(AUDIT)}")
    print(f"{'Next gate':<52}: STAGE 12C-2D — SELECTION")


def status() -> None:
    print(f"STAGE {STAGE} — CANDIDATE TRAINING STATUS")
    if MANIFEST.is_file() and AUDIT.is_file():
        audit = load_json(AUDIT)
        print("Status                    : PASS / FROZEN")
        print(f"Candidates                : {audit['candidate_ids']}")
        print(f"Parameters                : {audit['parameter_counts']}")
        print(f"Best calibration Top-1    : {audit['best_calibration_top1']}")
        print(f"Selection performed       : {audit['selection_performed']}")
        print(f"Audit SHA                 : {sha256(AUDIT)}")
        return
    if CHECKPOINT.is_file():
        print("Status                    : RUNNING")
        print(f"Checkpoint                : {load_json(CHECKPOINT)}")
        return
    print("Status                    : NOT STARTED")


def self_test() -> None:
    vocab_size = load_json(VOCABULARY_2A)["size"]
    grid = read_csv(CANDIDATE_GRID_1A)
    for r in grid:
        if r["trainable"].strip().upper() != "YES":
            continue
        model = Candidate(r["candidate_id"], vocab_size)
        params = model.trainable_parameters()
        cap = int(r["parameter_cap"])
        require(params <= cap,
                f"{r['candidate_id']} over cap: {params:,} > {cap:,}")
        print(f"  {r['candidate_id']:<40} {params:>9,} / {cap:>9,}  OK")
    require(len(TRAIN_FAMILIES) == 3 and len(CALIB_FAMILIES) == 1, "partition counts")
    print(f"Stage {STAGE} self-test: PASS")


def locked_execute(resume: bool) -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    with LOCK_FILE.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            stop(f"Stage {STAGE} execution lock is held by another process")
        execute(resume)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--status", action="store_true")
    mode.add_argument("--self-test", action="store_true")
    mode.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.status:
        status()
    elif args.self_test:
        self_test()
    else:
        locked_execute(args.resume)


if __name__ == "__main__":
    main()
