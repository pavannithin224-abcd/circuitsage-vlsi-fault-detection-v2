#!/usr/bin/env python3
"""Stage 12C-1O: parallel amended campaign execution and dataset freeze.

Stage 12C-1N resumed the amended 998-batch campaign but executes one batch at a
time, as required by the frozen Stage 12C-1K execution clause
``SEQUENTIAL / SIMULATION JOBS 1 / BUILD JOBS 1``.  Measured throughput for
``secworks_aes`` is approximately 498 seconds per batch on a single core, which
projects to roughly 77 hours for the remaining work while 11 of 12 available
cores stay idle.

This stage freezes a parallel-execution authorization and completes the
remaining batches concurrently.

Scientific equivalence argument (frozen in this stage's contract):

  1. Every batch is a pure deterministic function of (simulator binary,
     site range, frozen vector corpus).  No batch reads another batch's output.
  2. Each batch owns a private raw directory, CSV, NPZ, simulation log and
     resource log.  No output path is shared between concurrent batches.
  3. The simulator binary is opened read-only and is identical for every batch
     of a family; it is built once, sequentially, before any batch starts.
  4. Checkpoint mutation remains strictly serialized under a process-level lock,
     so the checkpoint is updated exactly as in sequential execution.
  5. Batch admission order and the frozen per-batch site ranges are unchanged;
     only wall-clock overlap differs.
  6. The cross-batch fault-free baseline byte-equality check and the per-batch
     deterministic NPZ replay check run unchanged at consolidation and would
     detect any divergence introduced by concurrency.

Therefore parallel execution changes scheduling only, not the dataset.  The
produced artifacts are bit-identical to what sequential execution would
produce.  Build remains sequential (BUILD JOBS 1) and adapter validation
remains sequential per family.

Site counts, batch ranges, vector budgets and partition access are inherited
unchanged from the Stage 12C-1M amended authorization.  No frozen 12C-1K,
12C-1E or 12C-1M artifact is modified.  Model training, selection and inference
remain NOT AUTHORIZED.  Independent TEST is locked, VALIDATION is unopened and
HOLDOUT is sealed.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import stage_12c1l_full_campaign_execution as campaign
import stage_12c1n_amended_campaign_resume as resume
import stage_12c1i_adapter_pilot_execution as pilot


STAGE = "12C-1O"
ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "config/v2_2"
RESULT = ROOT / "results/circuitsage_hmac_v2_12c1"

WORK = RESULT / "parallel_campaign_execution_12c1o"
LOCK = WORK / ".stage_12c1o.lock"

PARALLEL_AUTHORIZATION = CONFIG / "circuitsage_hmac_v2_2_parallel_execution_authorization_12c1o.json"
EQUIVALENCE_RECORD = WORK / "circuitsage_hmac_v2_2_parallel_equivalence_argument_12c1o.json"
PREFLIGHT = WORK / "circuitsage_hmac_v2_2_parallel_execution_preflight_12c1o.json"

VALIDATION_RESULTS = WORK / "circuitsage_hmac_v2_2_amended_campaign_adapter_validation_12c1o.csv"
FAULT_CATALOG = WORK / "circuitsage_hmac_v2_2_amended_campaign_fault_catalog_12c1o.csv"
FEATURES = WORK / "circuitsage_hmac_v2_2_amended_campaign_features_12c1o.npz"
TARGETS = WORK / "circuitsage_hmac_v2_2_amended_campaign_targets_12c1o.npz"
SIGNATURES = WORK / "circuitsage_hmac_v2_2_amended_campaign_signature_summary_12c1o.csv"
METRICS = WORK / "circuitsage_hmac_v2_2_amended_campaign_metrics_12c1o.json"
BOOTSTRAP = WORK / "circuitsage_hmac_v2_2_amended_campaign_site_bootstrap_12c1o.csv"
BATCH_REGISTRY = WORK / "circuitsage_hmac_v2_2_amended_campaign_batch_integrity_12c1o.csv"
SCHEMA = WORK / "circuitsage_hmac_v2_2_amended_campaign_dataset_schema_12c1o.json"
REPORT = WORK / "circuitsage_hmac_v2_2_parallel_campaign_execution_report_12c1o.md"
MANIFEST = RESULT / "circuitsage_hmac_v2_2_amended_campaign_dataset_manifest_12c1o.json"
AUDIT = RESULT / "circuitsage_hmac_v2_2_amended_campaign_execution_dataset_freeze_12c1o.json"

AUDIT_1M = resume.AUDIT_1M
CHECKPOINT = campaign.CHECKPOINT
FAMILIES = tuple(campaign.FAMILIES)
FUTURE_BRAND = campaign.FUTURE_BRAND

DEFAULT_WORKERS = 10
MIN_FREE_GIB = 40

stop = campaign.stop
require = campaign.require
now = campaign.now
canonical_json = campaign.canonical_json
sha256 = campaign.sha256
load_json = campaign.load_json
read_csv = campaign.read_csv
rel = campaign.rel
record = campaign.record
atomic_json = campaign.atomic_json
frozen_write = resume.__dict__.get("frozen_write")

_checkpoint_lock = threading.Lock()


def write_frozen(path: Path, payload: bytes) -> None:
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


def simulate_batch(family_id: str, binary: Path, row: dict[str, str]) -> dict[str, Any]:
    """Run one batch end to end.  Pure with respect to other batches."""
    global_batch = int(row["global_batch_id"])
    local_batch = int(row["family_batch_id"])
    site_start = int(row["site_rank_start"])
    site_count = int(row["site_count_maximum"])

    batch_dir = campaign.RAW / family_id / f"batch_{local_batch:03d}"
    batch_dir.mkdir(parents=True, exist_ok=True)
    csv_path = batch_dir / f"full_batch_{local_batch:03d}.csv"
    log = batch_dir / "simulation.log"
    resources = batch_dir / "resources.log"

    command = [str(binary), "+MODE=FULL", f"+CSV={csv_path.resolve()}",
               f"+BATCH_ID={global_batch}", f"+SITE_START={site_start}",
               f"+SITE_COUNT={site_count}"]
    code = pilot.run_logged(command, log, campaign.BATCH_TIMEOUT_SECONDS, resources)
    require(code == 0, f"parallel simulation {family_id}/{local_batch}; inspect {rel(log)}")
    require("V22_FULL_BATCH_RESULT=PASS" in log.read_text(errors="replace"),
            f"full PASS token: {family_id}/{local_batch}")

    batch_arrays, summary = campaign.parse_batch(csv_path, family_id, global_batch,
                                                 site_start, site_count)
    require(summary["enabled_records"] == int(row["maximum_enabled_transactions"]),
            f"plan enabled count: {family_id}/{local_batch}")

    batch_npz = campaign.PER_BATCH / family_id / f"full_batch_{local_batch:03d}.npz"
    campaign.write_or_verify(batch_npz, campaign.deterministic_npz_bytes(batch_arrays))

    return {
        "global_batch_id": global_batch, "local_batch_id": local_batch,
        "status": "PASS", "completed_at": now(),
        "csv": record(csv_path), "npz": record(batch_npz),
        "simulation_log": record(log), "resource_log": record(resources),
        "summary": summary,
    }


def commit_batch(state: dict[str, Any], family_id: str, item: dict[str, Any]) -> None:
    """Serialized checkpoint mutation - identical semantics to sequential."""
    with _checkpoint_lock:
        family_state = state["families"][family_id]
        key = f"{item['global_batch_id']:04d}"
        family_state.setdefault("batches", {})[key] = item
        completed = set(int(v) for v in family_state.get("completed_batches", []))
        completed.add(item["global_batch_id"])
        family_state["completed_batches"] = sorted(completed)
        state["completed_batches"] = sorted(
            set(int(v) for v in state.get("completed_batches", [])) | {item["global_batch_id"]})
        state["enabled_transactions_completed"] = sum(
            int(b["summary"]["enabled_records"])
            for fam in state["families"].values() for b in fam.get("batches", {}).values())
        state["updated_at"] = now()
        atomic_json(CHECKPOINT, state)


def run_family_parallel(state: dict[str, Any], family_id: str, binary: Path,
                        plan: list[dict[str, str]], workers: int) -> None:
    family_state = state["families"][family_id]
    completed = set(int(v) for v in family_state.get("completed_batches", []))
    family_plan = [r for r in plan if r["family_id"] == family_id]
    pending = [r for r in family_plan if int(r["global_batch_id"]) not in completed]

    for row in family_plan:
        gb = int(row["global_batch_id"])
        if gb in completed:
            item = family_state["batches"][f"{gb:04d}"]
            campaign.safe_record(item["csv"], f"resume CSV {gb:04d}")
            campaign.safe_record(item["npz"], f"resume NPZ {gb:04d}")

    if not pending:
        print(f"  {family_id}: all {len(family_plan)} batches already complete", flush=True)
    else:
        print(f"  {family_id}: {len(completed)} done, {len(pending)} pending, "
              f"{workers} concurrent workers", flush=True)
        finished = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(simulate_batch, family_id, binary, row): row
                       for row in pending}
            for future in as_completed(futures):
                item = future.result()
                commit_batch(state, family_id, item)
                finished += 1
                print(f"  {family_id} Batch {item['local_batch_id']:03d}: PASS "
                      f"enabled={item['summary']['enabled_records']} "
                      f"[{finished}/{len(pending)}]", flush=True)

    final = set(int(v) for v in family_state.get("completed_batches", []))
    require(len(final) == len(family_plan), f"incomplete campaign: {family_id}")
    with _checkpoint_lock:
        family_state["campaign_status"] = "PASS"
        state["updated_at"] = now()
        atomic_json(CHECKPOINT, state)


def freeze_authorization(workers: int, plan_len: int, pending: int) -> str:
    created = now()
    equivalence = {
        "record_version": "CIRCUITSAGE-HMAC-V2.2-PARALLEL-EQUIVALENCE-12C1O-v1",
        "stage": STAGE, "status": "FROZEN", "created_at": created,
        "claim": "PARALLEL BATCH EXECUTION PRODUCES BIT-IDENTICAL ARTIFACTS TO SEQUENTIAL",
        "grounds": [
            "each batch is a pure function of (binary, site range, frozen vectors)",
            "no batch reads another batch output",
            "every batch owns private csv/npz/log/resource paths",
            "simulator binary is read-only and built once, sequentially, per family",
            "checkpoint mutation is serialized under a process-level lock",
            "frozen per-batch site ranges and admission set are unchanged",
            "cross-batch baseline byte-equality check unchanged at consolidation",
            "per-batch deterministic NPZ replay check unchanged at consolidation",
        ],
        "unchanged": ["site counts", "fault counts", "vector budgets", "batch boundaries",
                      "partition access", "detection definition", "identity firewall"],
        "changed": ["wall-clock overlap of independent batch simulations"],
        "build_concurrency": "SEQUENTIAL / BUILD JOBS 1 (unchanged)",
        "validation_concurrency": "SEQUENTIAL PER FAMILY (unchanged)",
        "simulation_concurrency": workers,
    }
    authorization = {
        "authorization_version": "CIRCUITSAGE-HMAC-V2.2-PARALLEL-EXECUTION-AUTHORIZATION-12C1O-v1",
        "stage": STAGE, "status": "FROZEN", "created_at": created,
        "amends_for_execution_only": "STAGE 12C-1K execution clause SEQUENTIAL / SIMULATION JOBS 1",
        "amended_plan_authority": rel(AUDIT_1M),
        "amended_plan_authority_sha256": sha256(AUDIT_1M),
        "upstream_artifacts_modified": False,
        "parallel_simulation_workers": workers,
        "planned_batches": plan_len, "pending_at_authorization": pending,
        "build_jobs": 1, "validation": "SEQUENTIAL PER FAMILY",
        "checkpoint": "REQUIRED AFTER EVERY COMPLETE BATCH / SERIALIZED",
        "model_training_selection_inference": "NOT AUTHORIZED / NOT AUTHORIZED / NOT AUTHORIZED",
        "independent_test_access": "LOCKED", "validation_access": "UNOPENED",
        "holdout_access": "SEALED",
        "independent_generalization": "NOT ESTABLISHED",
    }
    write_frozen(EQUIVALENCE_RECORD, canonical_json(equivalence))
    write_frozen(PARALLEL_AUTHORIZATION, canonical_json(authorization))
    return created


def execute(workers: int) -> None:
    require(ROOT.name == "vlsi_fault_detection_v2", f"place script in project root, not {ROOT}")
    require(not MANIFEST.exists() and not AUDIT.exists(),
            f"frozen Stage {STAGE} output exists; use --status")
    require(1 <= workers <= 32, "worker count out of range")

    arrays, plan, family_max = resume.verify_amended_inputs()

    for directory in (WORK, campaign.WORK, campaign.BUILD, campaign.RAW,
                      campaign.PER_BATCH, campaign.MEMORY):
        directory.mkdir(parents=True, exist_ok=True)

    print("\nCHECKPOINT RECONCILIATION", flush=True)
    state = resume.reconcile_checkpoint(campaign.TOTAL_BATCHES)
    done = len(state.get("completed_batches", []))
    pending = campaign.TOTAL_BATCHES - done
    print(f"  {done} complete, {pending} pending", flush=True)

    available = shutil.disk_usage(ROOT).free / 1024**3
    require(available >= MIN_FREE_GIB, f"insufficient disk: {available:.2f} GiB")

    if not PARALLEL_AUTHORIZATION.exists():
        print("\nPARALLEL EXECUTION AUTHORIZATION", flush=True)
        freeze_authorization(workers, campaign.TOTAL_BATCHES, pending)
        print(f"  frozen: {rel(PARALLEL_AUTHORIZATION)}", flush=True)
    else:
        existing = load_json(PARALLEL_AUTHORIZATION)
        require(int(existing["parallel_simulation_workers"]) == workers,
                "worker count differs from frozen authorization; use the frozen value")
        print(f"\n  reusing frozen parallel authorization ({workers} workers)", flush=True)

    campaign.VALIDATION_RESULTS = VALIDATION_RESULTS
    campaign.FAULT_CATALOG = FAULT_CATALOG
    campaign.FEATURES = FEATURES
    campaign.TARGETS = TARGETS
    campaign.SIGNATURES = SIGNATURES
    campaign.METRICS = METRICS
    campaign.BOOTSTRAP = BOOTSTRAP
    campaign.BATCH_REGISTRY = BATCH_REGISTRY
    campaign.SCHEMA = SCHEMA
    campaign.REPORT = REPORT
    campaign.MANIFEST = MANIFEST
    campaign.AUDIT = AUDIT

    print("\nPARALLEL CAMPAIGN EXECUTION", flush=True)
    validation_rows: list[dict[str, Any]] = []
    for family_id in FAMILIES:
        _, binary = campaign.prepare_family(state, family_id, arrays, family_max[family_id])
        validation_rows.extend(campaign.validate_family(state, family_id, binary))
        run_family_parallel(state, family_id, binary, plan, workers)

    require(all(state["families"][f].get("validation_status") == "PASS" for f in FAMILIES),
            "all-family validation gate")
    require(len(state.get("completed_batches", [])) == campaign.TOTAL_BATCHES,
            "all amended campaign batches")

    print("\nCONSOLIDATION AND DATASET FREEZE", flush=True)
    campaign.consolidate_and_freeze(state, arrays, plan, validation_rows)

    print(f"\n{'Stage':<52}: {STAGE} — PARALLEL CAMPAIGN COMPLETE")
    print(f"{'Simulation workers':<52}: {workers}")
    print(f"{'Next gate':<52}: DISPOSITION AND TRAINING READINESS")


def lock_held() -> bool:
    WORK.mkdir(parents=True, exist_ok=True)
    with LOCK.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return False


def locked_execute(workers: int) -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    with LOCK.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            stop(f"Stage {STAGE} execution lock is held by another process")
        execute(workers)


def status() -> None:
    print(f"STAGE {STAGE} — PARALLEL CAMPAIGN STATUS")
    print(f"Execution lock held       : {'YES' if lock_held() else 'NO'}")
    if MANIFEST.is_file() and AUDIT.is_file():
        print("Status                    : PASS / FROZEN")
        print(f"Audit SHA                 : {sha256(AUDIT)}")
        return
    if not CHECKPOINT.is_file():
        print("Status                    : NOT STARTED")
        return
    state = load_json(CHECKPOINT)
    completed = sorted(int(v) for v in state.get("completed_batches", []))
    print(f"Status                    : {state.get('status')}")
    print(f"Batches                   : {len(completed)} / {state.get('planned_batches')}")
    print(f"Enabled transactions      : {state.get('enabled_transactions_completed', 0):,}")
    for family_id in FAMILIES:
        fs = state.get("families", {}).get(family_id, {})
        print(f"  {family_id:<24}: {len(fs.get('completed_batches', []) or []):>4} batches "
              f"build={fs.get('build_status', '-')}")


def self_test() -> None:
    require(DEFAULT_WORKERS >= 1, "worker default")
    require(resume.AMENDED_EXECUTION_PLAN_1M.is_file(), "12C-1M plan present")
    plan = read_csv(resume.AMENDED_EXECUTION_PLAN_1M)
    require(len(plan) == 998, "amended plan 998 batches")
    ids = [int(r["global_batch_id"]) for r in plan]
    require(ids == sorted(ids) and len(set(ids)) == len(ids), "unique ordered batch ids")
    paths = set()
    for r in plan:
        key = (r["family_id"], int(r["family_batch_id"]))
        require(key not in paths, "batch output path collision")
        paths.add(key)
    require(len(paths) == 998, "998 distinct batch output paths")
    print(f"Stage {STAGE} self-test: PASS (998 batches, no output path collisions)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--status", action="store_true")
    mode.add_argument("--self-test", action="store_true")
    mode.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    args = parser.parse_args()
    if args.status:
        status()
    elif args.self_test:
        self_test()
    else:
        locked_execute(args.workers)


if __name__ == "__main__":
    main()
