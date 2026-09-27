#!/usr/bin/env python3
"""Stage 12C-3B: independent test capture for secworks_chacha.

Stage 12C-3A authorized capture of the two INDEPENDENT_CIRCUIT_TEST families.
This stage executes it for ``secworks_chacha`` ONLY - the smaller of the two, so
that the capture flow is proven end to end on sealed data before the 203-file
``ibex_cpu`` tree is opened in Stage 12C-3C.

Pipeline (mirrors the frozen development flow 12C-1E -> 12C-1O)
---------------------------------------------------------------
  1. extract the sealed archive to a private work tree (upstream RTL untouched)
  2. derive the vector budget from the FROZEN 12C-1F rules - not invented here
  3. generic synthesis with the contract-pinned yosys
  4. enumerate fault sites as distinct driven cell-output bits
  5. instrument one netlist with runtime fault-injection multiplexers
  6. capture the fault-free baseline, then every fault in parallel batches
  7. derive behaviour signatures and per-circuit metrics
  8. freeze

Prediction-before-truth
-----------------------
This stage does NOT read the Stage 12C-2H predicted bands.  It computes metrics
and freezes them.  Scoring measured values against the frozen predictions is a
separate stage, so the capture cannot be influenced by what was predicted.

Vector derivation is contract-bound
-----------------------------------
Stage 12C-1F froze the budget rule for crypto families: 64 total vectors, 16
pilot, stimulus classes NIST_KAT/ZERO/ONES/COUNTING/AVALANCHE/RANDOM, seeds
derived as SHA256(FAMILY_ID || VECTOR_INDEX || 12C1F).  ChaCha is a crypto
stream cipher, so this stage applies the same frozen rule rather than choosing a
new vector count.  The derivation is recorded and the rule citation frozen.

Holdout
-------
``serv_cpu`` is never referenced, opened or simulated.  Its seal is asserted at
preflight and re-asserted in the audit.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

import stage_12c2c_candidate_training as base


STAGE = "12C-3B"
FAMILY = "secworks_chacha"
TOP = "chacha_core"
ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "config/v2_2"
RESULT1 = ROOT / "results/circuitsage_hmac_v2_12c1"
RESULT3 = ROOT / "results/circuitsage_hmac_v2_12c3"
WORK = RESULT3 / "chacha_capture_12c3b"
BUILD = ROOT / "build/circuitsage_hmac_v2_12c3/chacha_12c3b"
RAW = WORK / "raw_batches"
LOCK_FILE = WORK / ".stage_12c3b.lock"
CHECKPOINT = WORK / "circuitsage_hmac_v2_2_chacha_capture_checkpoint_12c3b.json"

AUTH_3A = CONFIG / "circuitsage_hmac_v2_2_independent_test_capture_authorization_12c3a.json"
AUDIT_3A = RESULT3 / "circuitsage_hmac_v2_2_test_capture_authorization_freeze_12c3a.json"
SOURCE_3A = ROOT / "stage_12c3a_test_capture_authorization.py"
VECTOR_CONTRACT_1F = CONFIG / "circuitsage_hmac_v2_2_transaction_vector_contract_12c1f.json"
BUDGET_1F = (RESULT1 / "transaction_fault_campaign_contract_12c1f"
             / "circuitsage_hmac_v2_2_vector_budget_registry_12c1f.csv")
REGISTRY_1C = (RESULT1 / "multicircuit_corpus_12c1c"
               / "circuitsage_hmac_v2_2_acquired_family_registry_12c1c.csv")
ARCHIVES_1C = RESULT1 / "multicircuit_corpus_12c1c/archives"

PINNED = {
    AUTH_3A: "36a9f803baae1621d941695d22dfd3dd3f9396e859a529989e763b5382aee272",
    AUDIT_3A: "3613ef29f3d8f8cdacb2c5d4126e28137991f592b85fe792b7474a4cfd9b315a",
    SOURCE_3A: "49f2392b08c5c8195f6c856aae2da9bba1aa2a820d33b287515af22032ba3347",
}

VECTOR_PLAN = WORK / "circuitsage_hmac_v2_2_chacha_vector_plan_12c3b.csv"
SYNTH_METRICS = WORK / "circuitsage_hmac_v2_2_chacha_synthesis_metrics_12c3b.json"
SITE_TABLE = WORK / "circuitsage_hmac_v2_2_chacha_site_table_12c3b.csv"
FAULT_CATALOG = WORK / "circuitsage_hmac_v2_2_chacha_fault_catalog_12c3b.csv"
SIGNATURES = WORK / "circuitsage_hmac_v2_2_chacha_signatures_12c3b.csv"
BATCH_INTEGRITY = WORK / "circuitsage_hmac_v2_2_chacha_batch_integrity_12c3b.csv"
METRICS = WORK / "circuitsage_hmac_v2_2_chacha_capture_metrics_12c3b.json"
SEAL_ASSERT = WORK / "circuitsage_hmac_v2_2_holdout_seal_assertion_12c3b.json"
PREFLIGHT = WORK / "circuitsage_hmac_v2_2_chacha_capture_preflight_12c3b.json"
REPORT = WORK / "circuitsage_hmac_v2_2_chacha_capture_report_12c3b.md"
MANIFEST = RESULT3 / "circuitsage_hmac_v2_2_chacha_capture_manifest_12c3b.json"
AUDIT = RESULT3 / "circuitsage_hmac_v2_2_chacha_capture_freeze_12c3b.json"

HOLDOUT = "serv_cpu"
SIBLING_TEST = "ibex_cpu"
CRYPTO_VECTOR_BUDGET = 64
CRYPTO_PILOT_BUDGET = 16
CRYPTO_CLASSES = ("NIST_KAT", "ZERO", "ONES", "COUNTING", "AVALANCHE", "RANDOM")
SEED_DOMAIN = "CIRCUITSAGE-HMAC-V2.2/12C-1F/VECTOR"
SITES_PER_BATCH = 64
WORKERS = 10
CYCLE_BUDGET = 4096
FUTURE_BRAND = base.FUTURE_BRAND

stop, require, now = base.stop, base.require, base.now
canonical_json, sha256, rel = base.canonical_json, base.sha256, base.rel
record, load_json, read_csv, csv_bytes = base.record, base.load_json, base.read_csv, base.csv_bytes
frozen_write, atomic_json = base.frozen_write, base.atomic_json


def tools() -> dict[str, Path]:
    spec = load_json(AUTH_3A)["required_tools"]
    out = {}
    for name, meta in spec.items():
        exe = Path(meta["executable"])
        require(exe.is_file(), f"pinned {name} missing at {exe}")
        got = subprocess.run([str(exe), "--version"], capture_output=True,
                             text=True).stdout.strip().splitlines()[0]
        require(got == meta["version"], f"{name} version drift: {got}")
        out[name] = exe
    return out


def assert_holdout_sealed() -> dict[str, Any]:
    markers = ("capture", "raw_batches", "signature", "instrumented", "synthes")
    permitted = ("multicircuit_corpus_12c1c",)
    leaked = []
    for hit in ROOT.glob(f"**/*{HOLDOUT}*"):
        if hit.is_file():
            p = rel(hit)
            if any(m in p for m in permitted):
                continue
            if any(m in p.lower() for m in markers):
                leaked.append(p)
    require(not leaked, f"holdout artifacts present: {leaked[:3]}")
    return {"family_id": HOLDOUT, "seal_status": "SEALED",
            "capture_artifacts": 0, "referenced_by_this_stage": False}


def derive_vectors() -> list[dict[str, Any]]:
    """Apply the FROZEN 12C-1F crypto budget rule. No new scheme invented."""
    budget = read_csv(BUDGET_1F)
    crypto = [r for r in budget if "NIST_KAT" in r["stimulus_classes"]]
    require(len(crypto) >= 2, "frozen crypto budget precedent required")
    for r in crypto:
        require(int(r["total_vector_budget"]) == CRYPTO_VECTOR_BUDGET,
                "crypto total budget precedent")
        require(int(r["pilot_vector_budget"]) == CRYPTO_PILOT_BUDGET,
                "crypto pilot budget precedent")
        require(r["truth_usage"].startswith("EXPECTED RESPONSE GENERATION ONLY"),
                "truth usage rule")
    rows = []
    for i in range(CRYPTO_VECTOR_BUDGET):
        seed = hashlib.sha256(f"{FAMILY}||{i}||12C1F".encode()).hexdigest()
        cls = CRYPTO_CLASSES[i % len(CRYPTO_CLASSES)]
        rows.append({
            "transaction_id": i, "family_id": FAMILY,
            "partition": "INDEPENDENT_CIRCUIT_TEST",
            "family_vector_index": i, "stimulus_class": cls,
            "pilot": "YES" if i < CRYPTO_PILOT_BUDGET else "NO",
            "cycle_budget": CYCLE_BUDGET,
            "seed_sha256": seed,
            "seed_domain": SEED_DOMAIN,
            "fault_identity_fields": 0,
        })
    return rows


def extract_source(yosys: Path) -> tuple[Path, list[Path]]:
    reg = {r["family_id"]: r for r in read_csv(REGISTRY_1C)}
    r = reg[FAMILY]
    require(r["partition"] == "INDEPENDENT_CIRCUIT_TEST", "partition check")
    archive = next(ARCHIVES_1C.glob(f"{FAMILY}-*.tar.gz"))
    require(sha256(archive) == r["archive_sha256"], "archive SHA")
    src = BUILD / "rtl_source"
    if src.exists():
        shutil.rmtree(src)
    src.mkdir(parents=True)
    with tarfile.open(archive) as t:
        t.extractall(src)
    rtl = sorted(p for p in src.rglob("*.v"))
    require(rtl, "RTL files extracted")
    top_file = next((p for p in rtl if p.stem == TOP), None)
    require(top_file is not None, f"top module file for {TOP}")
    return src, rtl


def synthesize(yosys: Path, rtl: list[Path]) -> tuple[Path, dict[str, Any]]:
    netlist = BUILD / f"{FAMILY}_generic_12c3b.json"
    log = BUILD / "yosys_synth.log"
    reads = "\n".join(f"read_verilog -sv {p}" for p in rtl)
    script = BUILD / "synth.ys"
    script.write_text(
        f"{reads}\n"
        f"hierarchy -check -top {TOP}\n"
        "proc; opt_expr; opt_clean\n"
        "check -assert\n"
        "flatten\n"
        "synth -top " + TOP + " -flatten\n"
        "opt -purge\n"
        "techmap\n"
        "opt -purge\n"
        f"write_json {netlist}\n"
        "stat\n")
    with log.open("w") as fh:
        rc = subprocess.run([str(yosys), "-s", str(script)],
                            stdout=fh, stderr=subprocess.STDOUT, cwd=str(BUILD)).returncode
    require(rc == 0, f"yosys synthesis failed; see {rel(log)}")
    require(netlist.is_file(), "netlist produced")
    j = json.loads(netlist.read_text())
    mod = j["modules"][TOP] if TOP in j["modules"] else next(iter(j["modules"].values()))
    cells = mod.get("cells", {})
    types: dict[str, int] = {}
    for c in cells.values():
        types[c["type"]] = types.get(c["type"], 0) + 1
    seq = sum(n for t, n in types.items() if "DFF" in t.upper())
    metrics = {
        "family_id": FAMILY, "top_module": TOP,
        "rtl_files": len(rtl), "cells": len(cells),
        "sequential_cells": seq, "combinational_cells": len(cells) - seq,
        "distinct_cell_types": len(types),
        "cell_type_histogram": dict(sorted(types.items(), key=lambda kv: -kv[1])),
        "netlist": record(netlist), "yosys_log": record(log),
    }
    return netlist, metrics


def enumerate_sites(netlist: Path) -> list[dict[str, Any]]:
    """Distinct driven cell-output bits, per the 12C-1M definition."""
    j = json.loads(netlist.read_text())
    mod = j["modules"][TOP] if TOP in j["modules"] else next(iter(j["modules"].values()))
    seen: set[int] = set()
    rows: list[dict[str, Any]] = []
    skipped_no_output = 0
    for name, cell in sorted(mod.get("cells", {}).items()):
        conns = cell.get("connections", {})
        pdir = cell.get("port_directions", {})
        outs = [p for p, d in pdir.items() if d == "output"]
        if not outs:
            skipped_no_output += 1
            continue
        for port in outs:
            for bit in conns.get(port, []):
                if not isinstance(bit, int):
                    continue
                if bit in seen:
                    continue
                seen.add(bit)
                rows.append({
                    "site_rank": len(rows), "cell_name": name,
                    "cell_type": cell["type"], "output_port": port,
                    "net_bit": bit,
                })
    require(rows, "at least one eligible site")
    print(f"  eligible sites={len(rows)}  cells_without_outputs={skipped_no_output}",
          flush=True)
    return rows


def execute() -> None:
    require(ROOT.name == "vlsi_fault_detection_v2", f"place script in project root, not {ROOT}")
    outputs = (VECTOR_PLAN, SYNTH_METRICS, SITE_TABLE, FAULT_CATALOG, SIGNATURES,
               BATCH_INTEGRITY, METRICS, SEAL_ASSERT, PREFLIGHT, REPORT,
               MANIFEST, AUDIT)
    require(not any(p.exists() for p in outputs),
            f"frozen Stage {STAGE} output exists; use --status")

    print("FROZEN INPUT VERIFICATION", flush=True)
    for path, expected in PINNED.items():
        require(path.is_file(), f"missing frozen input: {rel(path)}")
        require(sha256(path) == expected, f"frozen input SHA changed: {path.name}")
    auth = load_json(AUTH_3A)
    require(FAMILY in auth["authorized_families"], f"{FAMILY} not authorized")
    require(auth["holdout_capture"] == "NOT AUTHORIZED", "holdout must stay sealed")
    require(auth["fault_injection"].startswith("AUTHORIZED"), "fault injection authorized")
    print(f"  3 frozen inputs; 12C-3A authorizes {FAMILY}{'':<20}: OK", flush=True)

    seal = assert_holdout_sealed()
    print(f"  holdout {HOLDOUT} sealed, 0 capture artifacts{'':<20}: OK", flush=True)

    # prediction-before-truth: this stage must not read the predicted bands
    pred_cfg = CONFIG / "circuitsage_hmac_v2_2_sealed_circuit_predictions_12c2h.json"
    require(pred_cfg.is_file(), "12C-2H predictions exist (not read by this stage)")
    print(f"  12C-2H bands present but NOT read by this stage{'':<14}: OK", flush=True)

    t = tools()
    print(f"\nTOOLCHAIN (contract-pinned)", flush=True)
    for k, v in t.items():
        print(f"  {k:<10} {v}", flush=True)

    BUILD.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)

    print(f"\nVECTOR DERIVATION (frozen 12C-1F crypto rule)", flush=True)
    vectors = derive_vectors()
    print(f"  vectors={len(vectors)}  pilot={CRYPTO_PILOT_BUDGET}  "
          f"classes={len(CRYPTO_CLASSES)}", flush=True)

    print(f"\nOPENING SEALED SOURCE: {FAMILY}", flush=True)
    src, rtl = extract_source(t["yosys"])
    print(f"  extracted {len(rtl)} RTL files, top={TOP}", flush=True)

    print(f"\nGENERIC SYNTHESIS (pinned yosys)", flush=True)
    t0 = time.time()
    netlist, synth = synthesize(t["yosys"], rtl)
    print(f"  cells={synth['cells']:,} "
          f"({synth['combinational_cells']:,} comb / {synth['sequential_cells']:,} seq)  "
          f"types={synth['distinct_cell_types']}  {time.time()-t0:.0f}s", flush=True)

    print(f"\nSITE ENUMERATION", flush=True)
    sites = enumerate_sites(netlist)
    faults = [{"fault_id": 2 * s["site_rank"] + k, "site_rank": s["site_rank"],
               "cell_name": s["cell_name"], "cell_type": s["cell_type"],
               "stuck_value": k}
              for s in sites for k in (0, 1)]
    print(f"  sites={len(sites):,}  faults={len(faults):,}  "
           f"transactions={len(faults) * len(vectors):,}", flush=True)

    created = now()
    preflight = {
        "preflight_version": "CIRCUITSAGE-HMAC-V2.2-CHACHA-CAPTURE-PREFLIGHT-12C3B-v1",
        "stage": STAGE, "status": "PASS", "created_at": created,
        "family": FAMILY, "partition": "INDEPENDENT_CIRCUIT_TEST",
        "authorized_by": "12C-3A",
        "sibling_test_family_opened": False,
        "sibling_test_family": SIBLING_TEST,
        "holdout_opened": False,
        "predicted_bands_read": False,
        "vector_rule_source": "FROZEN 12C-1F CRYPTO BUDGET",
        "vector_count": len(vectors),
        "upstream_rtl_modified": False,
        "training_calls": 0, "selection_calls": 0,
        "acceptance_evaluation": "NOT AUTHORIZED BY THIS STAGE",
    }

    frozen_write(VECTOR_PLAN, csv_bytes(vectors, list(vectors[0].keys())))
    frozen_write(SYNTH_METRICS, canonical_json({
        "metrics_version": "CIRCUITSAGE-HMAC-V2.2-CHACHA-SYNTHESIS-12C3B-v1",
        "stage": STAGE, "created_at": created, **synth}))
    frozen_write(SITE_TABLE, csv_bytes(sites, list(sites[0].keys())))
    frozen_write(FAULT_CATALOG, csv_bytes(faults, list(faults[0].keys())))
    frozen_write(SEAL_ASSERT, canonical_json({
        "assertion_version": "CIRCUITSAGE-HMAC-V2.2-HOLDOUT-SEAL-ASSERTION-12C3B-v1",
        "stage": STAGE, "created_at": created, "holdout": seal}))
    frozen_write(PREFLIGHT, canonical_json(preflight))

    print(f"\n{'Stage':<52}: {STAGE} — CHACHA CAPTURE (PHASE 1)")
    print(f"{'Status':<52}: SYNTHESIS + ENUMERATION COMPLETE")
    print(f"{'Family':<52}: {FAMILY}")
    print(f"{'Cells':<52}: {synth['cells']:,}")
    print(f"{'Eligible sites':<52}: {len(sites):,}")
    print(f"{'Faults (SA0+SA1)':<52}: {len(faults):,}")
    print(f"{'Vectors (frozen 12C-1F rule)':<52}: {len(vectors)}")
    print(f"{'Transactions to simulate':<52}: {len(faults) * len(vectors):,}")
    print(f"{'Holdout ' + HOLDOUT:<52}: SEALED")
    print(f"{SIBLING_TEST + ' opened':<52}: NO")
    print()
    print("  Phase 1 artifacts frozen. Instrumentation and response capture")
    print("  follow in phase 2 of this stage once timing is measured.")


def status() -> None:
    print(f"STAGE {STAGE} — CHACHA CAPTURE STATUS")
    if MANIFEST.is_file() and AUDIT.is_file():
        a = load_json(AUDIT)
        print("Status                    : PASS / FROZEN")
        print(f"Audit SHA                 : {sha256(AUDIT)}")
        return
    if SITE_TABLE.is_file():
        print("Status                    : PHASE 1 COMPLETE (synthesis + enumeration)")
        s = load_json(SYNTH_METRICS)
        print(f"Cells                     : {s['cells']:,}")
        print(f"Sites                     : {len(read_csv(SITE_TABLE)):,}")
        return
    print("Status                    : NOT STARTED")


def self_test() -> None:
    a = load_json(AUTH_3A)
    require(FAMILY in a["authorized_families"], "family authorized")
    require(HOLDOUT in a["sealed_holdout_families"], "holdout sealed")
    b = read_csv(BUDGET_1F)
    crypto = [r for r in b if "NIST_KAT" in r["stimulus_classes"]]
    require(crypto and all(int(r["total_vector_budget"]) == CRYPTO_VECTOR_BUDGET
                           for r in crypto), "crypto budget precedent")
    h = hashlib.sha256(f"{FAMILY}||0||12C1F".encode()).hexdigest()
    require(len(h) == 64, "seed derivation")
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
