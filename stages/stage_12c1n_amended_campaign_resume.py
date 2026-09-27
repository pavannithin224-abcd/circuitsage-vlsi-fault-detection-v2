#!/usr/bin/env python3
"""Stage 12C-1N: amended full TRAIN/CALIBRATION campaign resume and dataset freeze.

Stage 12C-1L halted before starting ``secworks_aes`` because the frozen Stage
12C-1K ceiling was derived from Yosys generic cell counts while the campaign
derives fault sites from distinct driven cell-output bits.  Stage 12C-1M froze
the deterministic site-eligibility discovery and an amended 998-batch execution
plan that stays at or below every frozen 12C-1K ceiling.

This stage resumes the campaign against that amended plan.  It imports the
frozen Stage 12C-1L execution module and reuses its netlist instrumentation,
adapter validation, batch runner, batch parser and consolidation logic
unmodified.  Only the authoritative plan source, the family budget and the
derived totals are redirected to the Stage 12C-1M amended artifacts, and the
final dataset artifacts are attributed to this stage.

The live Stage 12C-1L checkpoint is continued, not discarded: the 440 batches
already completed for ``opentitan_hmac_sha256`` and ``picorv32_cpu`` are
re-verified by hash and skipped, never re-simulated.

Model training, selection and inference remain NOT AUTHORIZED.  Independent
TEST is locked, VALIDATION is unopened and HOLDOUT is sealed.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import stage_12c1l_full_campaign_execution as campaign
import stage_12c1i_adapter_pilot_execution as pilot


STAGE = "12C-1N"
ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "config/v2_2"
RESULT = ROOT / "results/circuitsage_hmac_v2_12c1"

# Stage 12C-1M amended authority ------------------------------------------------
WORK_1M = RESULT / "site_eligibility_discovery_12c1m"
DISCOVERY_CONTRACT_1M = CONFIG / "circuitsage_hmac_v2_2_site_eligibility_discovery_contract_12c1m.json"
AMENDED_AUTHORIZATION_1M = CONFIG / "circuitsage_hmac_v2_2_amended_full_campaign_execution_authorization_12c1m.json"
AMENDED_FAMILY_BUDGET_1M = WORK_1M / "circuitsage_hmac_v2_2_amended_full_campaign_family_budget_12c1m.csv"
AMENDED_EXECUTION_PLAN_1M = WORK_1M / "circuitsage_hmac_v2_2_amended_full_campaign_execution_plan_12c1m.csv"
CONTINUITY_1M = WORK_1M / "circuitsage_hmac_v2_2_completed_batch_continuity_12c1m.json"
MANIFEST_1M = RESULT / "circuitsage_hmac_v2_2_site_eligibility_discovery_manifest_12c1m.json"
AUDIT_1M = RESULT / "circuitsage_hmac_v2_2_site_eligibility_discovery_freeze_12c1m.json"
SOURCE_1M = ROOT / "stage_12c1m_site_eligibility_discovery.py"

# Reused Stage 12C-1L working tree (live checkpoint, not frozen evidence) -------
WORK_1L = campaign.WORK
CHECKPOINT_1L = campaign.CHECKPOINT

# Stage 12C-1N dataset outputs --------------------------------------------------
WORK = RESULT / "amended_campaign_execution_12c1n"
LOCK = WORK / ".stage_12c1n.lock"
VALIDATION_RESULTS = WORK / "circuitsage_hmac_v2_2_amended_campaign_adapter_validation_12c1n.csv"
FAULT_CATALOG = WORK / "circuitsage_hmac_v2_2_amended_campaign_fault_catalog_12c1n.csv"
FEATURES = WORK / "circuitsage_hmac_v2_2_amended_campaign_features_12c1n.npz"
TARGETS = WORK / "circuitsage_hmac_v2_2_amended_campaign_targets_12c1n.npz"
SIGNATURES = WORK / "circuitsage_hmac_v2_2_amended_campaign_signature_summary_12c1n.csv"
METRICS = WORK / "circuitsage_hmac_v2_2_amended_campaign_metrics_12c1n.json"
BOOTSTRAP = WORK / "circuitsage_hmac_v2_2_amended_campaign_site_bootstrap_12c1n.csv"
BATCH_REGISTRY = WORK / "circuitsage_hmac_v2_2_amended_campaign_batch_integrity_12c1n.csv"
SCHEMA = WORK / "circuitsage_hmac_v2_2_amended_campaign_dataset_schema_12c1n.json"
REPORT = WORK / "circuitsage_hmac_v2_2_amended_campaign_execution_report_12c1n.md"
MANIFEST = RESULT / "circuitsage_hmac_v2_2_amended_campaign_dataset_manifest_12c1n.json"
AUDIT = RESULT / "circuitsage_hmac_v2_2_amended_campaign_execution_dataset_freeze_12c1n.json"

PINNED_1M = {
    SOURCE_1M: "REQUIRED",  # resolved at runtime from the 12C-1M manifest
}

FAMILIES = tuple(campaign.FAMILIES)
BATCH_SITES = campaign.BATCH_SITES
FULL_VECTORS = dict(campaign.FULL_VECTORS)
FUTURE_BRAND = campaign.FUTURE_BRAND

stop = campaign.stop
require = campaign.require
now = campaign.now
canonical_json = campaign.canonical_json
sha256 = campaign.sha256
load_json = campaign.load_json
read_csv = campaign.read_csv
rel = campaign.rel
record = campaign.record
safe_record = campaign.safe_record
atomic_json = campaign.atomic_json


def verify_amended_inputs() -> tuple[dict[str, np.ndarray], list[dict[str, str]], dict[str, int]]:
    """Replacement for campaign.verify_inputs() bound to the 12C-1M amendment.

    Performs the same frozen-lineage verification as Stage 12C-1L, then swaps
    the authoritative plan/budget to the Stage 12C-1M amended artifacts and
    re-derives the module-level totals the campaign machinery reads.
    """
    print("FROZEN INPUT VERIFICATION (12C-1L lineage)", flush=True)
    for path, expected in campaign.PINNED.items():
        require(path.is_file(), f"missing frozen input: {rel(path)}")
        require(sha256(path) == expected, f"frozen input SHA: {path.name}")
    print(f"  {len(campaign.PINNED)} frozen 12C-1K / 12C-1E / 12C-1G / 12C-1I inputs        : OK", flush=True)

    print("FROZEN INPUT VERIFICATION (12C-1M amendment)", flush=True)
    manifest_1m = load_json(MANIFEST_1M)
    audit_1m = load_json(AUDIT_1M)
    require(manifest_1m.get("status") == "PASS" and audit_1m.get("status") == "PASS", "12C-1M freeze")
    require(audit_1m.get("manifest_record", {}).get("sha256") == sha256(MANIFEST_1M), "12C-1M manifest anchor")
    require(audit_1m.get("ceiling_exceeded") is False, "12C-1M ceiling guarantee")
    require(audit_1m.get("frozen_12c1k_outputs_modified") is False, "12C-1M upstream immutability")
    require(audit_1m.get("amended_campaign_execution") == "AUTHORIZED / RESUMABLE",
            "12C-1M amended authorization")
    require(audit_1m.get("model_training_selection_inference")
            == "NOT AUTHORIZED / NOT AUTHORIZED / NOT AUTHORIZED", "model boundary")
    require(audit_1m.get("independent_test_validation_holdout_access") == [0, 0, 0], "protected access")
    for name, item in sorted((manifest_1m.get("outputs") or {}).items()):
        safe_record(item, f"12C-1M output {name}")
    print(f"  Stage 12C-1M amended authorization and {len(manifest_1m.get('outputs') or {})} outputs      : OK",
          flush=True)

    amended_authorization = load_json(AMENDED_AUTHORIZATION_1M)
    require(amended_authorization.get("amended_full_train_calibration_campaign")
            == "AUTHORIZED / RESUMABLE", "amended authorization object")
    require(amended_authorization.get("original_authorization_modified") is False,
            "original 12C-1K authorization immutability")

    discovery = load_json(DISCOVERY_CONTRACT_1M)
    require(discovery.get("status") == "FROZEN", "12C-1M discovery contract")
    require(discovery.get("ceiling_exceeded") is False, "12C-1M contract ceiling guarantee")
    require(discovery.get("network_requirement") == "NONE / OFFLINE EXECUTION", "offline contract")

    # ---- swap in the amended plan ------------------------------------------
    family_rows = read_csv(AMENDED_FAMILY_BUDGET_1M)
    plan = read_csv(AMENDED_EXECUTION_PLAN_1M)
    require([row["family_id"] for row in family_rows] == list(FAMILIES), "amended family budget order")

    amended_sites = {row["family_id"]: int(row["maximum_sites"]) for row in family_rows}
    require(set(amended_sites) == set(FAMILIES), "amended family identifiers")
    total_sites = sum(amended_sites.values())
    total_faults = total_sites * 2
    total_batches = sum(math.ceil(amended_sites[f] / BATCH_SITES) for f in FAMILIES)
    total_enabled = sum(amended_sites[f] * 2 * FULL_VECTORS[f] for f in FAMILIES)

    require(int(discovery["amended_sites"]) == total_sites, "contract/budget site agreement")
    require(int(discovery["amended_planned_batches"]) == total_batches, "contract/budget batch agreement")
    require(len(plan) == total_batches, "amended plan batch count")
    require([int(r["global_batch_id"]) for r in plan] == list(range(total_batches)),
            "amended plan global order")

    # never exceed the frozen 12C-1K ceilings
    frozen_budget = read_csv(campaign.FAMILY_BUDGET)
    frozen_ceiling = {r["family_id"]: int(r["maximum_sites"]) for r in frozen_budget}
    for family_id in FAMILIES:
        require(amended_sites[family_id] <= frozen_ceiling[family_id],
                f"amended sites exceed frozen 12C-1K ceiling: {family_id}")
        require(int(next(r for r in family_rows if r["family_id"] == family_id)["full_vectors"])
                == FULL_VECTORS[family_id], f"amended vector budget: {family_id}")

    for family_id in FAMILIES:
        rows = [r for r in plan if r["family_id"] == family_id]
        require(len(rows) == math.ceil(amended_sites[family_id] / BATCH_SITES),
                f"amended family batches: {family_id}")
        require(sum(int(r["site_count_maximum"]) for r in rows) == amended_sites[family_id],
                f"amended plan sites: {family_id}")
        require(sum(int(r["maximum_enabled_transactions"]) for r in rows)
                == amended_sites[family_id] * 2 * FULL_VECTORS[family_id],
                f"amended plan transactions: {family_id}")

    # ---- rebind the campaign module totals ---------------------------------
    campaign.EXPECTED_SITES.clear()
    campaign.EXPECTED_SITES.update(amended_sites)
    campaign.TOTAL_SITES = total_sites
    campaign.TOTAL_FAULTS = total_faults
    campaign.TOTAL_BATCHES = total_batches
    campaign.TOTAL_ENABLED = total_enabled

    print(f"  Amended plan: {total_sites:,} sites / {total_faults:,} faults / "
          f"{total_batches} batches / {total_enabled:,} enabled       : OK", flush=True)

    arrays = campaign.load_vectors()
    manifest_1e = load_json(campaign.MANIFEST_1E)
    for family_id in FAMILIES:
        source = campaign.WORK_1E / "families" / family_id / f"{family_id}_generic_12c1e.json"
        require(source.is_file(), f"missing 12C-1E netlist: {family_id}")
    campaign.verify_manifest_outputs(manifest_1e, "12C-1E")

    family_max = {f: amended_sites[f] for f in FAMILIES}
    return arrays, plan, family_max


def reconcile_checkpoint(total_batches: int) -> dict[str, Any]:
    """Continue the live 12C-1L checkpoint under the amended plan."""
    require(CHECKPOINT_1L.is_file(), "no Stage 12C-1L checkpoint to resume")
    state = load_json(CHECKPOINT_1L)
    require(state.get("status") == "RUNNING", "checkpoint is not in RUNNING state")
    require(state.get("authorization_sha256") == campaign.PINNED[campaign.AUDIT_1K],
            "checkpoint authorization anchor")

    continuity = load_json(CONTINUITY_1M)
    require(continuity.get("status") == "PASS", "12C-1M continuity proof")
    require(int(continuity.get("resimulation_required", -1)) == 0, "continuity resimulation guarantee")
    verified = int(continuity.get("completed_batches_verified", -1))
    completed = sorted(int(v) for v in state.get("completed_batches", []))
    # The 12C-1M continuity proof certifies that the batches completed at
    # discovery time keep identical identity under the amended plan.  Batches
    # completed AFTER that proof were produced under the amended plan itself,
    # so the invariant is "at least the certified count", not exact equality.
    require(len(completed) >= verified,
            f"checkpoint lost completed batches: {len(completed)} < 12C-1M verified {verified}")
    require(len(completed) <= total_batches,
            f"checkpoint has more batches ({len(completed)}) than the amended plan ({total_batches})")

    if int(state.get("planned_batches", -1)) != total_batches:
        state["planned_batches"] = total_batches
        state["amended_by"] = "STAGE 12C-1M SITE-ELIGIBILITY DISCOVERY"
        state["amended_authorization_sha256"] = sha256(AUDIT_1M)
        state["updated_at"] = now()
        atomic_json(CHECKPOINT_1L, state)
        print(f"  checkpoint planned_batches reconciled to {total_batches} under 12C-1M authority",
              flush=True)
    return state


def execute(resume: bool) -> None:
    require(ROOT.name == "vlsi_fault_detection_v2", f"place script in project root, not {ROOT}")
    require(not MANIFEST.exists() and not AUDIT.exists(),
            f"frozen Stage {STAGE} output exists; use --status")

    arrays, plan, family_max = verify_amended_inputs()

    for directory in (WORK, WORK_1L, campaign.BUILD, campaign.RAW,
                      campaign.PER_BATCH, campaign.MEMORY):
        directory.mkdir(parents=True, exist_ok=True)

    print("\nCHECKPOINT RECONCILIATION", flush=True)
    state = reconcile_checkpoint(campaign.TOTAL_BATCHES)
    done = len(state.get("completed_batches", []))
    print(f"  resuming with {done} completed batches preserved; "
          f"{campaign.TOTAL_BATCHES - done} remaining", flush=True)

    # redirect the consolidation outputs to this stage
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

    print("\nCAMPAIGN EXECUTION", flush=True)
    validation_rows: list[dict[str, Any]] = []
    for family_id in FAMILIES:
        _, binary = campaign.prepare_family(state, family_id, arrays, family_max[family_id])
        validation_rows.extend(campaign.validate_family(state, family_id, binary))
        campaign.run_batches(state, family_id, binary, plan)

    require(all(state["families"][f].get("validation_status") == "PASS" for f in FAMILIES),
            "all-family validation gate")
    require(len(state.get("completed_batches", [])) == campaign.TOTAL_BATCHES,
            "all amended campaign batches")

    print("\nCONSOLIDATION AND DATASET FREEZE", flush=True)
    campaign.consolidate_and_freeze(state, arrays, plan, validation_rows)
    print(f"\n{'Stage':<52}: {STAGE} — AMENDED CAMPAIGN COMPLETE")
    print(f"{'Amended authority':<52}: STAGE 12C-1M")
    print(f"{'Next gate':<52}: STAGE 12C-1O — DISPOSITION AND TRAINING READINESS")


def lock_held() -> bool:
    WORK.mkdir(parents=True, exist_ok=True)
    with LOCK.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return False


def locked_execute(resume: bool) -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    with LOCK.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            stop(f"Stage {STAGE} execution lock is held by another process")
        execute(resume)


def status() -> None:
    print(f"STAGE {STAGE} — AMENDED CAMPAIGN STATUS")
    print(f"Execution lock held       : {'YES' if lock_held() else 'NO'}")
    if MANIFEST.is_file() and AUDIT.is_file():
        manifest, audit = load_json(MANIFEST), load_json(AUDIT)
        require(manifest.get("status") == "PASS" and audit.get("status") == "PASS", "frozen status")
        print("Status                    : PASS / FROZEN")
        print(f"Audit                     : {AUDIT}")
        print(f"Audit SHA                 : {sha256(AUDIT)}")
        return
    if not CHECKPOINT_1L.is_file():
        print("Status                    : NOT STARTED")
        return
    state = load_json(CHECKPOINT_1L)
    completed = sorted(int(v) for v in state.get("completed_batches", []))
    planned = int(state.get("planned_batches", 0))
    print(f"Status                    : {state.get('status')}")
    print(f"Batches                   : {len(completed)} / {planned}")
    print(f"Enabled transactions      : {state.get('enabled_transactions_completed', 0):,}")
    print(f"Updated                   : {state.get('updated_at')}")
    for family_id in FAMILIES:
        fs = state.get("families", {}).get(family_id, {})
        n = len(fs.get("completed_batches", []) or [])
        print(f"  {family_id:<24}: {n:>4} batches  build={fs.get('build_status', '-')}  "
              f"campaign={fs.get('campaign_status', '-')}")


def self_test() -> None:
    require(campaign.BATCH_SITES == 64, "batch size")
    require(set(FAMILIES) == {"opentitan_hmac_sha256", "picorv32_cpu",
                              "secworks_aes", "secworks_sha256"}, "family set")
    require(math.ceil(26560 / 64) == 415 and math.ceil(9125 / 64) == 143, "amended batch arithmetic")
    require(AMENDED_EXECUTION_PLAN_1M.is_file(), "12C-1M amended plan present")
    require(AUDIT_1M.is_file(), "12C-1M audit present")
    plan = read_csv(AMENDED_EXECUTION_PLAN_1M)
    require(len(plan) == 998, "amended plan is 998 batches")
    sites = sum(int(r["site_count_maximum"]) for r in plan)
    require(sites == 63742, "amended plan total sites")
    print(f"Stage {STAGE} self-test: PASS")


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
        locked_execute(resume=True)


if __name__ == "__main__":
    main()
