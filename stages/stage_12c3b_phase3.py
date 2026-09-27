#!/usr/bin/env python3
"""Stage 12C-3B phase 3: full chacha capture, signature derivation, freeze.

Runs all 20,222 faults (10,111 sites x SA0/SA1) across the 64-vector plan in
parallel batches, captures the fault-free baseline, derives behaviour signatures
and per-circuit metrics, then freezes.

Batch independence follows the Stage 12C-1O equivalence argument: each batch is a
pure function of (binary, site range, vector plan) writing to its own private
path.

Prediction-before-truth: predictions are written and hashed BEFORE metrics are
compared to anything. The Stage 12C-2H bands are never read here.
"""
from __future__ import annotations

import argparse
import collections
import fcntl
import hashlib
import json
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

import stage_12c2c_candidate_training as base
import stage_12c3b_chacha_capture as p1
import stage_12c3b_phase2 as p2


STAGE = "12C-3B"
FAMILY = p1.FAMILY
ROOT = p1.ROOT
WORK = p1.WORK
BUILD = p1.BUILD
RAW = p1.RAW
SIM = p2.SIM
N_VEC = 64
SITES_PER_BATCH = 128

stop, require, now = base.stop, base.require, base.now
canonical_json, sha256, rel = base.canonical_json, base.sha256, base.rel
record, load_json, read_csv, csv_bytes = base.record, base.load_json, base.read_csv, base.csv_bytes
frozen_write, atomic_json = base.frozen_write, base.atomic_json

BASELINE = RAW / "baseline.txt"
CHECKPOINT = p1.CHECKPOINT


def run_batch(args):
    bid, start, count = args
    RAW.mkdir(parents=True, exist_ok=True)
    out = RAW / f"batch_{bid:04d}.txt"
    if out.exists() and out.stat().st_size > 0:
        return bid, "CACHED", 0.0
    t0 = time.time()
    r = subprocess.run([str(SIM), f"+CSV={out}", f"+SITE_START={start}",
                        f"+SITE_COUNT={count}", f"+BATCH_ID={bid}"],
                       capture_output=True, text=True, cwd=str(BUILD))
    dt = time.time() - t0
    if r.returncode != 0 or not out.exists():
        (RAW / f"err_{bid:04d}.log").write_text((r.stdout or "") + (r.stderr or ""))
        return bid, f"FAIL rc={r.returncode}", dt
    if "V22_3B_BATCH_RESULT=PASS" not in (r.stdout or ""):
        return bid, "FAIL no-token", dt
    return bid, "OK", dt


def capture(workers: int, n_sites: int) -> list[dict[str, Any]]:
    print("BASELINE (fault-free)", flush=True)
    if not (BASELINE.exists() and BASELINE.stat().st_size > 0):
        r = subprocess.run([str(SIM), f"+CSV={BASELINE}", "+SITE_START=-1",
                            "+SITE_COUNT=0", "+BATCH_ID=0"],
                           capture_output=True, text=True, cwd=str(BUILD))
        require(r.returncode == 0 and BASELINE.is_file(), "baseline capture")
    brows = [l.split() for l in BASELINE.read_text().splitlines() if l.strip()]
    require(len(brows) == N_VEC, f"baseline rows {len(brows)} != {N_VEC}")
    print(f"  {len(brows)} vectors, {len(set(r[3] for r in brows))} distinct responses",
          flush=True)

    batches = []
    bid = 1
    for start in range(0, n_sites, SITES_PER_BATCH):
        count = min(SITES_PER_BATCH, n_sites - start)
        batches.append((bid, start, count))
        bid += 1
    print(f"\nFAULT CAPTURE  {len(batches)} batches x {SITES_PER_BATCH} sites, "
          f"{workers} workers", flush=True)

    t0 = time.time()
    done = fails = 0
    integrity = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(run_batch, b): b for b in batches}
        for fu in as_completed(futs):
            b = futs[fu]
            b_id, status, dt = fu.result()
            done += 1
            if status.startswith("FAIL"):
                fails += 1
                print(f"  batch {b_id} {status}", flush=True)
            f = RAW / f"batch_{b_id:04d}.txt"
            integrity.append({
                "batch_id": b_id, "site_start": b[1], "site_count": b[2],
                "faults": b[2] * 2, "rows": sum(1 for _ in f.open()) if f.is_file() else 0,
                "expected_rows": b[2] * 2 * N_VEC,
                "raw_sha256": sha256(f) if f.is_file() else "",
                "status": "PASS" if status in ("OK", "CACHED") else status,
            })
            if done % 20 == 0:
                el = time.time() - t0
                eta = (len(batches) - done) / (done / el) if el else 0
                print(f"  {done:>4}/{len(batches)}  elapsed={el/60:.1f}m  "
                      f"eta={eta/60:.1f}m  fails={fails}", flush=True)
                atomic_json(CHECKPOINT, {"stage": STAGE, "status": "RUNNING",
                                         "batches_done": done,
                                         "batches_total": len(batches),
                                         "updated_at": now()})
    print(f"  capture complete in {(time.time()-t0)/60:.1f} min, {fails} failures",
          flush=True)
    require(fails == 0, f"{fails} batches failed")
    for row in integrity:
        require(row["rows"] == row["expected_rows"],
                f"batch {row['batch_id']} row count {row['rows']} != {row['expected_rows']}")
    return integrity


def derive(n_sites: int) -> tuple[list[dict], dict[str, Any], str]:
    brows = [l.split() for l in BASELINE.read_text().splitlines() if l.strip()]
    basemap = {int(r[2]): r[3] for r in brows}

    per: dict[tuple[int, int], list[tuple[int, str]]] = collections.defaultdict(list)
    for f in sorted(RAW.glob("batch_*.txt")):
        for line in f.open():
            p = line.split()
            if len(p) < 4:
                continue
            per[(int(p[0]), int(p[1]))].append((int(p[2]), p[3]))

    require(len(per) == n_sites * 2,
            f"faults captured {len(per)} != {n_sites*2}")

    rows = []
    sig_index: dict[str, list[tuple[int, int]]] = collections.defaultdict(list)
    observable = 0
    for (site, stuck), vals in sorted(per.items()):
        vals.sort()
        require(len(vals) == N_VEC, f"fault ({site},{stuck}) has {len(vals)} vectors")
        obs = any(basemap[t] != d for t, d in vals)
        sig = hashlib.sha256("|".join(f"{t}:{d}" for t, d in vals).encode()).hexdigest()
        if obs:
            observable += 1
            sig_index[sig].append((site, stuck))
        rows.append({"site_rank": site, "stuck_value": stuck,
                     "observable": int(obs), "signature_sha256": sig})

    # PREDICTION BEFORE TRUTH: hash the signature table before metrics
    sig_bytes = csv_bytes(rows, ["site_rank", "stuck_value", "observable",
                                 "signature_sha256"])
    pred_sha = hashlib.sha256(sig_bytes).hexdigest()

    for r in rows:
        n = len(sig_index.get(r["signature_sha256"], []))
        r["candidate_set_size"] = n if r["observable"] else 0

    n_faults = len(rows)
    uniq = sum(1 for r in rows if r["observable"] and r["candidate_set_size"] == 1)
    amb = [r["candidate_set_size"] for r in rows
           if r["observable"] and r["candidate_set_size"] > 1]
    metrics = {
        "family_id": FAMILY,
        "partition": "INDEPENDENT_CIRCUIT_TEST",
        "sites": n_sites,
        "faults": n_faults,
        "vectors": N_VEC,
        "transactions": n_faults * N_VEC,
        "observable_faults": observable,
        "observable_fraction": round(observable / n_faults, 6),
        "distinct_signatures": len(sig_index),
        "unique_signature_faults": uniq,
        "signature_uniqueness": round(uniq / max(observable, 1), 6),
        "all_injected_exact_site_rate": round(uniq / n_faults, 6),
        "observable_candidate_set_coverage": 1.0,
        "observable_detection_recall": 1.0,
        "all_injected_detection_recall": round(observable / n_faults, 6),
        "fault_free_false_alarm_rate": 0.0,
        "mean_ambiguous_set": round(float(np.mean(amb)) if amb else 0.0, 4),
        "max_candidate_set": max((r["candidate_set_size"] for r in rows), default=0),
        "signature_table_sha256_before_metrics": pred_sha,
    }
    return rows, metrics, pred_sha


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()

    require(p1.SITE_TABLE.is_file(), "phase 1 frozen")
    require(SIM.is_file(), "phase 2 simulator built")
    outputs = (p1.SIGNATURES, p1.BATCH_INTEGRITY, p1.METRICS, p1.REPORT,
               p1.MANIFEST, p1.AUDIT)
    require(not any(p.exists() for p in outputs),
            "phase 3 outputs already frozen; use --status")

    sites = read_csv(p1.SITE_TABLE)
    n_sites = len(sites)
    print(f"STAGE {STAGE} PHASE 3 — {FAMILY}")
    print(f"  sites={n_sites:,}  faults={n_sites*2:,}  "
          f"transactions={n_sites*2*N_VEC:,}\n")

    integrity = capture(a.workers, n_sites)
    print("\nSIGNATURE DERIVATION")
    rows, metrics, pred_sha = derive(n_sites)
    print(f"  signature table hashed before metrics: {pred_sha[:16]}...")
    for k in ("observable_fraction", "signature_uniqueness",
              "all_injected_exact_site_rate", "distinct_signatures",
              "mean_ambiguous_set", "max_candidate_set"):
        print(f"  {k:<34} {metrics[k]}")

    created = now()
    frozen_write(p1.SIGNATURES, csv_bytes(rows, list(rows[0].keys())))
    frozen_write(p1.BATCH_INTEGRITY, csv_bytes(integrity, list(integrity[0].keys())))
    frozen_write(p1.METRICS, canonical_json({
        "metrics_version": "CIRCUITSAGE-HMAC-V2.2-CHACHA-CAPTURE-METRICS-12C3B-v1",
        "stage": STAGE, "created_at": created,
        "prediction_before_truth": True,
        "predicted_bands_read": False,
        **metrics}))

    synth = load_json(p1.SYNTH_METRICS)
    report = f"""# Stage {STAGE} — Independent Test Capture: `{FAMILY}`

**Status: PASS / FROZEN — capture and measurement only. No evaluation.**

## What was captured

| | |
|---|---|
| partition | INDEPENDENT_CIRCUIT_TEST |
| authorized by | 12C-3A |
| cells | {synth['cells']:,} |
| sites | {metrics['sites']:,} |
| faults (SA0+SA1) | {metrics['faults']:,} |
| vectors (frozen 12C-1F rule) | {metrics['vectors']} |
| transactions | {metrics['transactions']:,} |

## Measured

| metric | value |
|---|---|
| observable fraction | {metrics['observable_fraction']:.4f} |
| distinct signatures | {metrics['distinct_signatures']:,} |
| signature uniqueness (of observable) | {metrics['signature_uniqueness']:.4f} |
| **all-injected exact-site rate** | **{metrics['all_injected_exact_site_rate']:.4f}** |
| observable candidate-set coverage | {metrics['observable_candidate_set_coverage']:.4f} |
| fault-free false-alarm rate | {metrics['fault_free_false_alarm_rate']:.4f} |
| mean ambiguous set | {metrics['mean_ambiguous_set']} |
| max candidate set | {metrics['max_candidate_set']:,} |

## Integrity

- every batch row count matched its expected value exactly
- fault-free baseline captured separately and used as the reference
- signature table hashed **before** any metric was computed
  (`{pred_sha[:32]}...`)
- Stage 12C-2H predicted bands were **not read** by this stage

## Not done here

No acceptance evaluation, no scoring against predictions, no model inference.
`ibex_cpu` was not opened. `serv_cpu` remains **SEALED**.

## Next gate

**12C-3C** — `ibex_cpu` capture. Scoring against the frozen 12C-2H predictions
happens only after both test circuits are captured. Future hybrid brand remains
**{base.FUTURE_BRAND}**.
"""
    frozen_write(p1.REPORT, report.encode())

    stage_outputs = (p1.VECTOR_PLAN, p1.SYNTH_METRICS, p1.SITE_TABLE,
                     p1.FAULT_CATALOG, p1.SIGNATURES, p1.BATCH_INTEGRITY,
                     p1.METRICS, p1.SEAL_ASSERT, p1.PREFLIGHT, p1.REPORT)
    manifest = {
        "manifest_version": "CIRCUITSAGE-HMAC-V2.2-CHACHA-CAPTURE-MANIFEST-12C3B-v1",
        "stage": STAGE, "status": "PASS", "created_at": created,
        "stage_source": record(Path(p1.__file__).resolve()),
        "phase2_source": record(Path(p2.__file__).resolve()),
        "phase3_source": record(Path(__file__).resolve()),
        "frozen_inputs": {rel(p): record(p) for p in p1.PINNED},
        "outputs": {rel(p): record(p) for p in stage_outputs},
        "batches": len(integrity),
        "holdout_access": 0, "sibling_test_access": 0,
    }
    frozen_write(p1.MANIFEST, canonical_json(manifest))

    audit = {
        "audit_version": "CIRCUITSAGE-HMAC-V2.2-CHACHA-CAPTURE-FREEZE-12C3B-v1",
        "stage": STAGE, "status": "PASS",
        "family_id": FAMILY, "partition": "INDEPENDENT_CIRCUIT_TEST",
        "authorized_by": "12C-3A",
        "sites": metrics["sites"], "faults": metrics["faults"],
        "vectors": metrics["vectors"], "transactions": metrics["transactions"],
        "observable_fraction": metrics["observable_fraction"],
        "signature_uniqueness": metrics["signature_uniqueness"],
        "all_injected_exact_site_rate": metrics["all_injected_exact_site_rate"],
        "observable_candidate_set_coverage": metrics["observable_candidate_set_coverage"],
        "observable_detection_recall": metrics["observable_detection_recall"],
        "all_injected_detection_recall": metrics["all_injected_detection_recall"],
        "fault_free_false_alarm_rate": metrics["fault_free_false_alarm_rate"],
        "max_candidate_set": metrics["max_candidate_set"],
        "batches": len(integrity), "batch_failures": 0,
        "prediction_before_truth": True,
        "predicted_bands_read": False,
        "signature_table_sha256_before_metrics": pred_sha,
        "acceptance_evaluation": "NOT PERFORMED",
        "scoring_against_predictions": "NOT PERFORMED",
        "ibex_cpu_opened": False,
        "serv_cpu_seal": "SEALED",
        "independent_generalization": "NOT ESTABLISHED",
        "metrics_record": record(p1.METRICS),
        "signatures_record": record(p1.SIGNATURES),
        "manifest_record": record(p1.MANIFEST),
        "future_combined_model_brand": base.FUTURE_BRAND,
        "next_gate": "STAGE 12C-3C — IBEX_CPU CAPTURE",
    }
    frozen_write(p1.AUDIT, canonical_json(audit))

    print(f"\n{'Stage':<52}: {STAGE} — CHACHA CAPTURE COMPLETE")
    print(f"{'Status':<52}: PASS / FROZEN")
    print(f"{'Exact-site rate (measured)':<52}: "
          f"{metrics['all_injected_exact_site_rate']:.4f}")
    print(f"{'Observable fraction':<52}: {metrics['observable_fraction']:.4f}")
    print(f"{'Scoring against predictions':<52}: NOT PERFORMED")
    print(f"{'ibex_cpu opened':<52}: NO")
    print(f"{'serv_cpu':<52}: SEALED")
    print(f"{'Audit SHA':<52}: {sha256(p1.AUDIT)}")


if __name__ == "__main__":
    main()
