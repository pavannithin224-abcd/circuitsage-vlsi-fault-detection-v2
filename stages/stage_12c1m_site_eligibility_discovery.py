#!/usr/bin/env python3
"""Stage 12C-1M: site-eligibility discovery and amended campaign authorization.

During Stage 12C-1L execution the campaign halted before starting
``secworks_aes`` with::

    STOP: secworks_aes has only 26560 eligible driven bits; 26565 required

Root cause.  Stage 12C-1K derived each family's site ceiling from the Yosys
generic **cell count**.  Stage 12C-1L derives fault sites from **distinct
driven cell-output bits**.  Yosys emits ``$scopeinfo`` hierarchy-annotation
pseudo-cells that are counted as cells but expose no ports at all, so they can
never carry a cell-output SA0/SA1 fault.  Two families therefore expose fewer
eligible sites than their frozen ceiling.

This is a physical property of the frozen netlists discovered at execution
time.  Stage 12C-1K was correct given the information available when it was
frozen; it is NOT defective and is NOT modified here.  This stage records the
newly discovered constraint as new, forward-dated evidence and freezes an
amended family budget and execution plan as NEW artifacts.

Authority.  The frozen Stage 12C-1K contract already anticipates this outcome:
the execution stage "must derive the exact eligible site catalog
deterministically; it may reduce this ceiling for contracted exclusions but may
never exceed it", and the dataset policy states "DERIVE ELIGIBLE SITES
DETERMINISTICALLY; ACTUAL COUNT MAY NOT EXCEED FROZEN MAXIMUM".  Every amended
family count produced here is at or below its frozen ceiling.

Completed work.  Stage 12C-1L already completed 440 batches for two families.
This stage proves those batches retain identical global batch identifiers,
site ranges, fault index ranges and vector counts under the amended plan, so no
completed batch is invalidated and no simulation is repeated.

No simulation, fault injection, netlist instrumentation, dataset construction,
model loading, training, selection or inference occurs in this stage.
Independent TEST is locked, VALIDATION is unopened and HOLDOUT is sealed.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import io
import json
import math
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import stage_12c1i_adapter_pilot_execution as pilot


STAGE = "12C-1M"
ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "config/v2_2"
RESULT = ROOT / "results/circuitsage_hmac_v2_12c1"
WORK = RESULT / "site_eligibility_discovery_12c1m"
LOCK_FILE = WORK / ".stage_12c1m.lock"

# ---------------------------------------------------------------- frozen input
SOURCE_1K = ROOT / "stage_12c1k_full_campaign_authorization.py"
SOURCE_1I = ROOT / "stage_12c1i_adapter_pilot_execution.py"
SOURCE_1L = ROOT / "stage_12c1l_full_campaign_execution.py"
EXECUTION_CONTRACT_1K = CONFIG / "circuitsage_hmac_v2_2_full_campaign_execution_contract_12c1k.json"
AUTHORIZATION_1K = CONFIG / "circuitsage_hmac_v2_2_full_campaign_execution_authorization_12c1k.json"
DATASET_SCHEMA_1K = CONFIG / "circuitsage_hmac_v2_2_full_campaign_dataset_schema_12c1k.json"
AUTH_WORK_1K = RESULT / "full_campaign_authorization_12c1k"
FAMILY_BUDGET_1K = AUTH_WORK_1K / "circuitsage_hmac_v2_2_full_campaign_family_budget_12c1k.csv"
EXECUTION_PLAN_1K = AUTH_WORK_1K / "circuitsage_hmac_v2_2_full_campaign_execution_plan_12c1k.csv"
MANIFEST_1K = RESULT / "circuitsage_hmac_v2_2_full_campaign_authorization_manifest_12c1k.json"
AUDIT_1K = RESULT / "circuitsage_hmac_v2_2_full_campaign_execution_authorization_freeze_12c1k.json"

WORK_1E = RESULT / "train_calibration_synthesis_12c1e"
MANIFEST_1E = RESULT / "circuitsage_hmac_v2_2_elaboration_synthesis_manifest_12c1e.json"
AUDIT_1E = RESULT / "circuitsage_hmac_v2_2_elaboration_wrapper_generic_synthesis_freeze_12c1e.json"
AUDIT_1I = RESULT / "circuitsage_hmac_v2_2_adapter_functional_validation_pilot_execution_freeze_12c1i.json"

WORK_1L = RESULT / "full_campaign_execution_12c1l"
CHECKPOINT_1L = WORK_1L / "circuitsage_hmac_v2_2_full_campaign_checkpoint_12c1l.json"

PINNED = {
    SOURCE_1K: "bb2c58384489b043fef4978f8f6cd4b1618c6abd9adfefa7c71a1091bdf67ab8",
    SOURCE_1I: "735df97cbbf8ccd1097913d7044906c209a647e3290d056741c7880e26544f94",
    SOURCE_1L: "3c2d405ccf534459e5fd5ac83fc2ae459d6506278d93ea8811189b5256830814",
    EXECUTION_CONTRACT_1K: "cff6747c9bb469db89d3f5a1ca2f5f2ebb7f9a38a7b04b3280cd9dec0166713d",
    AUTHORIZATION_1K: "c872be0cd7a12241c2c62a010da47e556608e201b0e2744bc8a19483392f68e5",
    DATASET_SCHEMA_1K: "c3e2fccd7d036e6a7deb0ca9c89cf5eaedec86e944ac2ed14f2ac8f4f0ed8723",
    EXECUTION_PLAN_1K: "7e4f253eee3ebd5be204bdb352ab38b25a110f41d0db5451a8e1027ac12f8ee8",
    MANIFEST_1K: "a53fe05b665ae1d633d2f5e1da1cf0c2986572eda03cf0763048808f41b702bf",
    AUDIT_1K: "faf2ad35559e8547d913f6859cf3f1ab2a3e846f2fae134585b77f7bd60c68e1",
    MANIFEST_1E: "4d3d606e174ae06589e0e537486a15e042521fe9a02d49f9b4e1c596198476b6",
    AUDIT_1E: "3ac4f454173166ee6848e2617c7072742665e7166345ce5ffdb837f574d59220",
    AUDIT_1I: "cc96e5d7d11f9f75658365875c6521789416399b7bf09471f0fab334f09b2fa7",
}

# ---------------------------------------------------------------- stage output
DISCOVERY_CONTRACT = CONFIG / "circuitsage_hmac_v2_2_site_eligibility_discovery_contract_12c1m.json"
AMENDED_AUTHORIZATION = CONFIG / "circuitsage_hmac_v2_2_amended_full_campaign_execution_authorization_12c1m.json"
EXCLUSION_REGISTRY = WORK / "circuitsage_hmac_v2_2_site_exclusion_registry_12c1m.csv"
ELIGIBILITY_SUMMARY = WORK / "circuitsage_hmac_v2_2_site_eligibility_summary_12c1m.csv"
AMENDED_FAMILY_BUDGET_CSV = WORK / "circuitsage_hmac_v2_2_amended_full_campaign_family_budget_12c1m.csv"
AMENDED_FAMILY_BUDGET_JSON = WORK / "circuitsage_hmac_v2_2_amended_full_campaign_family_budget_12c1m.json"
AMENDED_EXECUTION_PLAN = WORK / "circuitsage_hmac_v2_2_amended_full_campaign_execution_plan_12c1m.csv"
CONTINUITY = WORK / "circuitsage_hmac_v2_2_completed_batch_continuity_12c1m.json"
PREFLIGHT = WORK / "circuitsage_hmac_v2_2_site_eligibility_discovery_preflight_12c1m.json"
REPORT = WORK / "circuitsage_hmac_v2_2_site_eligibility_discovery_report_12c1m.md"
MANIFEST = RESULT / "circuitsage_hmac_v2_2_site_eligibility_discovery_manifest_12c1m.json"
AUDIT = RESULT / "circuitsage_hmac_v2_2_site_eligibility_discovery_freeze_12c1m.json"

FAMILIES = (
    "opentitan_hmac_sha256",
    "picorv32_cpu",
    "secworks_aes",
    "secworks_sha256",
)
TOPS = dict(pilot.TOPS)
PARTITIONS = dict(pilot.PARTITIONS)
FULL_VECTORS = {
    "opentitan_hmac_sha256": 64,
    "picorv32_cpu": 48,
    "secworks_aes": 64,
    "secworks_sha256": 64,
}
BATCH_SITES = 64
MIN_FREE_GIB = 40
FUTURE_BRAND = "Faultiva 1.0 — Hybrid VLSI Fault Intelligence"

ELIGIBILITY_RULE = (
    "A fault site is a distinct driven cell-output net bit: for every cell in "
    "the frozen 12C-1E generic netlist top module, in deterministic sorted "
    "order, for every port whose port_direction is 'output', every integer net "
    "bit with id >= 2 that has not already been claimed by an earlier "
    "cell/port. Constant bits 0/1 and already-claimed net bits are not sites."
)


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


def safe_record(item: dict[str, Any], label: str) -> Path:
    require(isinstance(item, dict), f"{label} record")
    raw = item.get("path")
    require(isinstance(raw, str) and raw, f"{label} path")
    path = (ROOT / raw).resolve()
    require(path.is_relative_to(ROOT), f"{label} path escapes project root")
    require(path.is_file(), f"missing {label}: {raw}")
    require(sha256(path) == item.get("sha256"), f"{label} SHA")
    require(path.stat().st_size == int(item.get("bytes", -1)), f"{label} size")
    return path


def verify_manifest_outputs(manifest: dict[str, Any], label: str, minimum: int) -> None:
    outputs = manifest.get("outputs")
    require(isinstance(outputs, dict) and len(outputs) >= minimum, f"{label} outputs")
    for name, item in sorted(outputs.items()):
        safe_record(item, f"{label} output {name}")


# ----------------------------------------------------------------- derivation


def netlist_path(family_id: str) -> Path:
    return WORK_1E / "families" / family_id / f"{family_id}_generic_12c1e.json"


def classify_cells(family_id: str) -> tuple[int, int, list[dict[str, Any]]]:
    """Return (eligible_site_count, total_cells, excluded_cell_rows).

    Implements ELIGIBILITY_RULE exactly as the frozen Stage 12C-1I
    enumerate_sites() does, and additionally reports which cells contribute no
    site and why.
    """
    design = load_json(netlist_path(family_id))
    modules = design.get("modules", {})
    require(set(modules) == {TOPS[family_id]}, f"flattened top module: {family_id}")
    module = modules[TOPS[family_id]]
    cells = module.get("cells", {})

    eligible = 0
    used_bits: set[int] = set()
    excluded: list[dict[str, Any]] = []
    for cell_name in sorted(cells):
        cell = cells[cell_name]
        directions = cell.get("port_directions", {})
        connections = cell.get("connections", {})
        output_ports = [p for p in sorted(connections) if directions.get(p) == "output"]
        gained = 0
        saw_constant = False
        saw_duplicate = False
        for port_name in output_ports:
            for bit in connections[port_name]:
                if not isinstance(bit, int):
                    continue
                if bit < 2:
                    saw_constant = True
                    continue
                if bit in used_bits:
                    saw_duplicate = True
                    continue
                used_bits.add(bit)
                gained += 1
        eligible += gained
        if gained == 0:
            if not output_ports:
                reason = "NO_OUTPUT_PORT"
            elif saw_constant and saw_duplicate:
                reason = "CONSTANT_AND_DUPLICATE_ONLY"
            elif saw_duplicate:
                reason = "DUPLICATE_NET_BITS_ONLY"
            elif saw_constant:
                reason = "CONSTANT_NET_BITS_ONLY"
            else:
                reason = "NO_INTEGER_OUTPUT_BITS"
            excluded.append({
                "family_id": family_id,
                "partition": PARTITIONS[family_id],
                "cell_name": cell_name,
                "cell_type": str(cell.get("type", "")),
                "output_port_count": len(output_ports),
                "exclusion_reason": reason,
                "eligible_sites_contributed": 0,
            })
    return eligible, len(cells), excluded


def confirm_enumerator_equivalence(family_id: str, expected: int) -> str:
    """Prove the derived count matches the frozen 12C-1I enumerator exactly.

    Calling the frozen enumerator with SITES_PER_FAMILY set to the derived
    count makes it return at precisely the last eligible bit, so a matching
    length, contiguous ranks and a stable catalog digest prove rule identity
    without modifying or re-implementing the frozen enumerator.
    """
    saved = pilot.SITES_PER_FAMILY
    try:
        pilot.SITES_PER_FAMILY = expected
        sites = pilot.enumerate_sites(family_id, netlist_path(family_id))
    finally:
        pilot.SITES_PER_FAMILY = saved
    require(len(sites) == expected, f"enumerator equivalence count: {family_id}")
    require([s["site_rank"] for s in sites] == list(range(expected)),
            f"enumerator site-rank continuity: {family_id}")
    digest = hashlib.sha256()
    for site in sites:
        digest.update(f"{site['opaque_site_id']}\0{site['cell_name']}\0"
                      f"{site['output_port']}\0{site['output_bit_index']}\0"
                      f"{site['net_bit_id']}\n".encode())
    return digest.hexdigest()


def build_amended_plan(eligible: dict[str, int]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    family_rows: list[dict[str, Any]] = []
    plan_rows: list[dict[str, Any]] = []
    global_batch = 0
    global_site_offset = 0
    global_fault_offset = 0
    for family_id in FAMILIES:
        sites = eligible[family_id]
        faults = sites * 2
        vector_count = FULL_VECTORS[family_id]
        batches = math.ceil(sites / BATCH_SITES)
        enabled = faults * vector_count
        baseline = batches * vector_count
        family_rows.append({
            "family_id": family_id, "partition": PARTITIONS[family_id],
            "maximum_sites": sites, "maximum_faults": faults,
            "full_vectors": vector_count, "site_batches": batches,
            "baseline_records": baseline, "maximum_enabled_transactions": enabled,
            "maximum_total_records": baseline + enabled,
            "campaign_status": "AMENDED / AUTHORIZED",
        })
        for local_batch in range(batches):
            site_start = local_batch * BATCH_SITES
            site_count = min(BATCH_SITES, sites - site_start)
            fault_count = site_count * 2
            plan_rows.append({
                "global_batch_id": global_batch, "family_id": family_id,
                "partition": PARTITIONS[family_id], "family_batch_id": local_batch,
                "site_rank_start": site_start, "site_count_maximum": site_count,
                "global_site_offset": global_site_offset + site_start,
                "fault_index_start": global_fault_offset + site_start * 2,
                "fault_count_maximum": fault_count, "vector_count": vector_count,
                "baseline_records": vector_count,
                "maximum_enabled_transactions": fault_count * vector_count,
                "checkpoint_after_batch": "REQUIRED", "status": "AMENDED / AUTHORIZED",
            })
            global_batch += 1
        global_site_offset += sites
        global_fault_offset += faults
    require(global_site_offset == sum(eligible.values()), "amended plan site total")
    require(global_fault_offset == sum(eligible.values()) * 2, "amended plan fault total")
    return family_rows, plan_rows


def completed_batches_by_family() -> dict[str, list[int]]:
    if not CHECKPOINT_1L.is_file():
        return {}
    state = load_json(CHECKPOINT_1L)
    out: dict[str, list[int]] = {}
    for family_id, family_state in (state.get("families") or {}).items():
        done = sorted(int(v) for v in (family_state.get("completed_batches") or []))
        if done:
            out[family_id] = done
    return out


# -------------------------------------------------------------------- execute


def verify_inputs() -> None:
    print("FROZEN INPUT VERIFICATION", flush=True)
    for path, expected in PINNED.items():
        require(path.is_file(), f"missing frozen input: {rel(path)}")
        require(sha256(path) == expected,
                f"frozen input SHA changed (upstream evidence must be byte-identical): {path.name}")
        print(f"  {path.name:<96}: OK", flush=True)

    manifest_1k = load_json(MANIFEST_1K)
    audit_1k = load_json(AUDIT_1K)
    manifest_1e = load_json(MANIFEST_1E)
    audit_1e = load_json(AUDIT_1E)
    audit_1i = load_json(AUDIT_1I)

    require(manifest_1k.get("status") == "PASS" and audit_1k.get("status") == "PASS", "12C-1K freeze")
    require(audit_1k.get("manifest_record", {}).get("sha256") == sha256(MANIFEST_1K), "12C-1K manifest anchor")
    verify_manifest_outputs(manifest_1k, "12C-1K", 8)
    require(manifest_1e.get("status") == "PASS" and audit_1e.get("status") == "PASS", "12C-1E freeze")
    require(audit_1e.get("deterministic_replay") == "PASS / BYTE-EXACT / 4 OF 4", "netlist replay")
    verify_manifest_outputs(manifest_1e, "12C-1E", 1)
    require(audit_1i.get("adapter_functional_validation") == "PASS / 4 OF 4 / BYTE-EXACT REPLAY",
            "adapter validation lineage")
    print("  Stage 12C-1K / 12C-1E / 12C-1I / 12C-1L frozen evidence                                   : PASS",
          flush=True)


def execute() -> None:
    require(ROOT.name == "vlsi_fault_detection_v2", f"place script in project root, not {ROOT}")
    outputs = (DISCOVERY_CONTRACT, AMENDED_AUTHORIZATION, EXCLUSION_REGISTRY, ELIGIBILITY_SUMMARY,
               AMENDED_FAMILY_BUDGET_CSV, AMENDED_FAMILY_BUDGET_JSON, AMENDED_EXECUTION_PLAN,
               CONTINUITY, PREFLIGHT, REPORT, MANIFEST, AUDIT)
    require(not any(path.exists() for path in outputs),
            f"frozen Stage {STAGE} output already exists; use --status")

    verify_inputs()

    frozen_family_rows = read_csv(FAMILY_BUDGET_1K)
    frozen_plan = read_csv(EXECUTION_PLAN_1K)
    require([row["family_id"] for row in frozen_family_rows] == list(FAMILIES), "frozen family order")
    frozen_ceiling = {row["family_id"]: int(row["maximum_sites"]) for row in frozen_family_rows}
    for row in frozen_family_rows:
        require(int(row["full_vectors"]) == FULL_VECTORS[row["family_id"]],
                f"frozen vector budget: {row['family_id']}")

    print("\nDETERMINISTIC SITE-ELIGIBILITY DERIVATION", flush=True)
    eligible: dict[str, int] = {}
    exclusion_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    digests: dict[str, str] = {}
    for family_id in FAMILIES:
        count, cells, excluded = classify_cells(family_id)
        digest = confirm_enumerator_equivalence(family_id, count)
        ceiling = frozen_ceiling[family_id]
        require(count <= ceiling,
                f"eligible sites exceed frozen 12C-1K ceiling ({count} > {ceiling}): {family_id}")
        require(count > 0, f"no eligible sites: {family_id}")
        eligible[family_id] = count
        digests[family_id] = digest
        exclusion_rows.extend(excluded)
        summary_rows.append({
            "family_id": family_id, "partition": PARTITIONS[family_id],
            "top_module": TOPS[family_id], "generic_cells": cells,
            "frozen_ceiling_sites_12c1k": ceiling, "eligible_sites_12c1m": count,
            "reduction": ceiling - count, "excluded_cells": len(excluded),
            "frozen_batches_12c1k": math.ceil(ceiling / BATCH_SITES),
            "amended_batches_12c1m": math.ceil(count / BATCH_SITES),
            "site_catalog_sha256": digest,
            "ceiling_respected": "YES",
        })
        print(f"  {family_id:<24} cells={cells:<7} eligible={count:<7} ceiling={ceiling:<7} "
              f"reduction={ceiling - count:<3} excluded={len(excluded)}", flush=True)

    total_eligible = sum(eligible.values())
    total_ceiling = sum(frozen_ceiling.values())
    require(total_eligible <= total_ceiling, "amended total exceeds frozen total ceiling")

    family_rows, plan_rows = build_amended_plan(eligible)

    # -- completed Stage 12C-1L work must survive the amendment untouched ------
    print("\nCOMPLETED-BATCH CONTINUITY", flush=True)
    completed = completed_batches_by_family()
    frozen_by_id = {int(row["global_batch_id"]): row for row in frozen_plan}
    amended_by_id = {int(row["global_batch_id"]): row for row in plan_rows}
    continuity_families: list[dict[str, Any]] = []
    checked = 0
    for family_id, done in sorted(completed.items()):
        require(eligible[family_id] == frozen_ceiling[family_id],
                f"family has completed batches but amended site count changed: {family_id}")
        for global_batch in done:
            frozen_row = frozen_by_id.get(global_batch)
            amended_row = amended_by_id.get(global_batch)
            require(frozen_row is not None and amended_row is not None,
                    f"completed batch missing from plan: {global_batch}")
            for field in ("family_id", "family_batch_id", "site_rank_start",
                          "site_count_maximum", "vector_count", "fault_index_start",
                          "global_site_offset", "maximum_enabled_transactions"):
                require(str(frozen_row[field]) == str(amended_row[field]),
                        f"completed batch {global_batch} field '{field}' changed by amendment")
            checked += 1
        continuity_families.append({
            "family_id": family_id, "completed_batches": len(done),
            "global_batch_first": done[0], "global_batch_last": done[-1],
            "sites_unchanged": True,
        })
        print(f"  {family_id:<24} {len(done):>4} completed batches verified identical "
              f"(global {done[0]}..{done[-1]})", flush=True)
    if not completed:
        print("  no Stage 12C-1L checkpoint found; nothing to preserve", flush=True)

    available_gib = shutil.disk_usage(ROOT).free / 1024**3
    require(available_gib >= MIN_FREE_GIB,
            f"insufficient disk: {available_gib:.2f} GiB; {MIN_FREE_GIB} GiB required")

    total_batches = len(plan_rows)
    total_baselines = sum(int(r["baseline_records"]) for r in family_rows)
    total_enabled = sum(int(r["maximum_enabled_transactions"]) for r in family_rows)
    total_records = total_baselines + total_enabled
    created = now()

    family_fields = ["family_id", "partition", "maximum_sites", "maximum_faults", "full_vectors",
                     "site_batches", "baseline_records", "maximum_enabled_transactions",
                     "maximum_total_records", "campaign_status"]
    plan_fields = ["global_batch_id", "family_id", "partition", "family_batch_id",
                   "site_rank_start", "site_count_maximum", "global_site_offset",
                   "fault_index_start", "fault_count_maximum", "vector_count",
                   "baseline_records", "maximum_enabled_transactions",
                   "checkpoint_after_batch", "status"]
    exclusion_fields = ["family_id", "partition", "cell_name", "cell_type",
                        "output_port_count", "exclusion_reason", "eligible_sites_contributed"]
    summary_fields = ["family_id", "partition", "top_module", "generic_cells",
                      "frozen_ceiling_sites_12c1k", "eligible_sites_12c1m", "reduction",
                      "excluded_cells", "frozen_batches_12c1k", "amended_batches_12c1m",
                      "site_catalog_sha256", "ceiling_respected"]

    discovery_contract = {
        "contract_version": "CIRCUITSAGE-HMAC-V2.2-SITE-ELIGIBILITY-DISCOVERY-12C1M-v1",
        "stage": STAGE, "status": "FROZEN", "created_at": created,
        "discovered_during": "STAGE 12C-1L FULL-CAMPAIGN EXECUTION",
        "discovery": ("secworks_aes exposes 26560 eligible driven cell-output bits; the frozen "
                      "12C-1K ceiling of 26565 was derived from the Yosys generic cell count"),
        "root_cause": ("Yosys $scopeinfo hierarchy-annotation pseudo-cells are counted as cells "
                       "but expose no ports, so they cannot carry a cell-output SA0/SA1 fault"),
        "upstream_stage_defective": False,
        "upstream_mutation": "PROHIBITED / NONE PERFORMED",
        "amendment_type": "CONTRACTED SITE EXCLUSION / CEILING REDUCTION ONLY",
        "authority": ("12C-1K fault_catalog_policy: DERIVE ELIGIBLE SITES DETERMINISTICALLY; "
                      "ACTUAL COUNT MAY NOT EXCEED FROZEN MAXIMUM"),
        "eligibility_rule": ELIGIBILITY_RULE,
        "families": list(FAMILIES), "family_skipping": "PROHIBITED",
        "frozen_ceiling_sites": total_ceiling, "amended_sites": total_eligible,
        "reduction_sites": total_ceiling - total_eligible,
        "amended_faults": total_eligible * 2,
        "frozen_planned_batches": len(frozen_plan), "amended_planned_batches": total_batches,
        "ceiling_exceeded": False,
        "site_batch_size": BATCH_SITES,
        "baseline_records": total_baselines,
        "maximum_enabled_transactions": total_enabled,
        "maximum_total_records": total_records,
        "execution": "SEQUENTIAL / SIMULATION JOBS 1 / BUILD JOBS 1",
        "checkpoint": "REQUIRED AFTER EVERY COMPLETE BATCH",
        "completed_batch_policy": "PRESERVE; IDENTICAL GLOBAL IDS AND SITE RANGES; NEVER RESIMULATE",
        "network_requirement": "NONE / OFFLINE EXECUTION",
    }
    amended_authorization = {
        "authorization_version": "CIRCUITSAGE-HMAC-V2.2-AMENDED-FULL-CAMPAIGN-EXECUTION-AUTHORIZATION-12C1M-v1",
        "stage": STAGE, "status": "FROZEN", "created_at": created,
        "amends_for_execution_only": rel(AUTHORIZATION_1K),
        "original_authorization_sha256": PINNED[AUTHORIZATION_1K],
        "original_audit_sha256": PINNED[AUDIT_1K],
        "original_authorization_modified": False,
        "amended_full_train_calibration_campaign": "AUTHORIZED / RESUMABLE",
        "families": list(FAMILIES),
        "amended_sites_faults_batches": [total_eligible, total_eligible * 2, total_batches],
        "model_training_selection_inference": "NOT AUTHORIZED / NOT AUTHORIZED / NOT AUTHORIZED",
        "independent_test_access": "LOCKED", "validation_access": "UNOPENED", "holdout_access": "SEALED",
        "independent_generalization": "NOT ESTABLISHED",
    }
    family_json = {
        "registry_version": "CIRCUITSAGE-HMAC-V2.2-AMENDED-FULL-CAMPAIGN-FAMILY-BUDGET-12C1M-v1",
        "stage": STAGE, "status": "FROZEN", "families": family_rows,
        "amended_totals": {
            "sites": total_eligible, "faults": total_eligible * 2,
            "vectors": sum(FULL_VECTORS.values()), "batches": total_batches,
            "enabled_transactions": total_enabled, "records": total_records,
        },
        "frozen_12c1k_totals": {
            "sites": total_ceiling, "faults": total_ceiling * 2,
            "batches": len(frozen_plan),
        },
    }
    continuity = {
        "continuity_version": "CIRCUITSAGE-HMAC-V2.2-COMPLETED-BATCH-CONTINUITY-12C1M-v1",
        "stage": STAGE, "status": "PASS", "created_at": created,
        "checkpoint_present": CHECKPOINT_1L.is_file(),
        "completed_batches_verified": checked,
        "families": continuity_families,
        "resimulation_required": 0,
        "verdict": "ALL COMPLETED BATCHES RETAIN IDENTICAL IDENTITY UNDER THE AMENDED PLAN",
    }
    preflight = {
        "preflight_version": "CIRCUITSAGE-HMAC-V2.2-SITE-ELIGIBILITY-DISCOVERY-PREFLIGHT-12C1M-v1",
        "stage": STAGE, "status": "PASS", "created_at": created,
        "available_disk_gib": round(available_gib, 2), "minimum_disk_gib": MIN_FREE_GIB,
        "simulation_calls": 0, "fault_injection_calls": 0, "netlist_instrumentation_calls": 0,
        "dataset_records_created": 0, "model_deserializations": 0,
        "training_calls": 0, "inference_calls": 0,
        "independent_test_access": 0, "validation_access": 0, "holdout_access": 0,
        "frozen_12c1k_outputs_modified": False,
        "frozen_12c1e_netlists_modified": False,
        "frozen_12c1l_batches_modified": False,
    }

    reduction_lines = "\n".join(
        f"| `{r['family_id']}` | {r['generic_cells']} | {r['frozen_ceiling_sites_12c1k']} | "
        f"{r['eligible_sites_12c1m']} | {r['reduction']} | {r['excluded_cells']} | "
        f"{r['frozen_batches_12c1k']} → {r['amended_batches_12c1m']} |"
        for r in summary_rows
    )
    excluded_lines = "\n".join(
        f"| `{r['family_id']}` | `{r['cell_name']}` | `{r['cell_type']}` | {r['exclusion_reason']} |"
        for r in exclusion_rows
    ) or "| — | — | — | none |"

    report = f"""# Stage {STAGE} — Site-Eligibility Discovery

**Status: PASS / FROZEN — discovery and re-authorization only, no execution.**

## What was discovered

During Stage 12C-1L execution the campaign halted before starting
`secworks_aes`:

```
STOP: secworks_aes has only 26560 eligible driven bits; 26565 required
```

Stage 12C-1K derived each family's site ceiling from the Yosys **generic cell
count**. Stage 12C-1L derives fault sites from **distinct driven cell-output
bits**. Yosys emits `$scopeinfo` hierarchy-annotation pseudo-cells that are
counted as cells but expose no ports at all, so they can never carry a
cell-output SA0/SA1 fault.

This is a physical property of the frozen netlists that became visible only at
execution time. **Stage 12C-1K is not defective** and is not modified by this
stage; it was correct given the information available when it was frozen.

## Authority to re-authorize

The frozen Stage 12C-1K contract anticipates this outcome:

> `fault_catalog_policy`: DERIVE ELIGIBLE SITES DETERMINISTICALLY; ACTUAL
> COUNT MAY NOT EXCEED FROZEN MAXIMUM

Reducing the ceiling for a contracted exclusion is permitted; exceeding it is
not. Every amended family count is **at or below** its frozen ceiling.

## Eligibility rule

{ELIGIBILITY_RULE}

## Amended budget

| family | generic cells | frozen ceiling | eligible sites | reduction | excluded cells | batches |
|---|---|---|---|---|---|---|
{reduction_lines}

Frozen total: **{total_ceiling}** sites / **{len(frozen_plan)}** batches.
Amended total: **{total_eligible}** sites / **{total_batches}** batches.
Reduction: **{total_ceiling - total_eligible}** sites. Ceiling exceeded: **NO**.

## Excluded cells

| family | cell | type | reason |
|---|---|---|---|
{excluded_lines}

## Completed-work continuity

Stage 12C-1L completed batches verified unchanged: **{checked}**.
Re-simulation required: **0**.

Families with completed batches retain identical site counts, global batch
identifiers, site ranges, fault index ranges and vector counts under the
amended plan.

## What this stage did not do

No simulation, fault injection, netlist instrumentation, dataset construction,
model loading, training, selection or inference. Stage 12C-1K, 12C-1E and
12C-1L artifacts were verified byte-identical and never modified. Independent
TEST remains locked, VALIDATION unopened, HOLDOUT sealed. Independent
generalization remains **NOT ESTABLISHED**. The future hybrid brand remains
**{FUTURE_BRAND}**.

## Next gate

**Stage 12C-1N** — resume the full TRAIN/CALIBRATION campaign against the
amended plan frozen here. Model training, selection and inference remain
**NOT AUTHORIZED**.
""".encode()

    frozen_write(DISCOVERY_CONTRACT, canonical_json(discovery_contract))
    frozen_write(AMENDED_AUTHORIZATION, canonical_json(amended_authorization))
    frozen_write(EXCLUSION_REGISTRY, csv_bytes(exclusion_rows, exclusion_fields))
    frozen_write(ELIGIBILITY_SUMMARY, csv_bytes(summary_rows, summary_fields))
    frozen_write(AMENDED_FAMILY_BUDGET_CSV, csv_bytes(family_rows, family_fields))
    frozen_write(AMENDED_FAMILY_BUDGET_JSON, canonical_json(family_json))
    frozen_write(AMENDED_EXECUTION_PLAN, csv_bytes(plan_rows, plan_fields))
    frozen_write(CONTINUITY, canonical_json(continuity))
    frozen_write(PREFLIGHT, canonical_json(preflight))
    frozen_write(REPORT, report)

    stage_outputs = (DISCOVERY_CONTRACT, AMENDED_AUTHORIZATION, EXCLUSION_REGISTRY,
                     ELIGIBILITY_SUMMARY, AMENDED_FAMILY_BUDGET_CSV, AMENDED_FAMILY_BUDGET_JSON,
                     AMENDED_EXECUTION_PLAN, CONTINUITY, PREFLIGHT, REPORT)
    manifest = {
        "manifest_version": "CIRCUITSAGE-HMAC-V2.2-SITE-ELIGIBILITY-DISCOVERY-MANIFEST-12C1M-v1",
        "stage": STAGE, "status": "PASS", "created_at": created,
        "stage_source": record(Path(__file__).resolve()),
        "frozen_inputs": {rel(path): record(path) for path in PINNED},
        "outputs": {rel(path): record(path) for path in stage_outputs},
        "families": list(FAMILIES),
        "frozen_ceiling_sites": total_ceiling, "amended_sites": total_eligible,
        "amended_faults": total_eligible * 2, "amended_planned_batches": total_batches,
        "site_catalog_sha256": digests,
        "completed_batches_verified": checked,
        "simulation_calls": 0, "fault_injection_calls": 0, "dataset_records_created": 0,
        "model_deserializations": 0, "training_calls": 0, "inference_calls": 0,
        "independent_test_access": 0, "validation_access": 0, "holdout_access": 0,
    }
    frozen_write(MANIFEST, canonical_json(manifest))
    audit = {
        "audit_version": "CIRCUITSAGE-HMAC-V2.2-SITE-ELIGIBILITY-DISCOVERY-FREEZE-12C1M-v1",
        "stage": STAGE, "status": "PASS",
        "discovered_during": "STAGE 12C-1L FULL-CAMPAIGN EXECUTION",
        "amendment_type": "CONTRACTED SITE EXCLUSION / CEILING REDUCTION ONLY",
        "upstream_stage": "12C-1K", "upstream_audit_sha256": PINNED[AUDIT_1K],
        "upstream_stage_defective": False,
        "frozen_12c1k_outputs_modified": False,
        "frozen_12c1e_netlists_modified": False,
        "frozen_12c1l_batches_modified": False,
        "ceiling_exceeded": False,
        "frozen_amended_sites": [total_ceiling, total_eligible],
        "reduction_sites": total_ceiling - total_eligible,
        "frozen_amended_batches": [len(frozen_plan), total_batches],
        "excluded_cell_count": len(exclusion_rows),
        "excluded_cell_types": sorted({r["cell_type"] for r in exclusion_rows}),
        "completed_batches_verified": checked,
        "resimulation_required": 0,
        "amended_campaign_execution": "AUTHORIZED / RESUMABLE",
        "model_training_selection_inference": "NOT AUTHORIZED / NOT AUTHORIZED / NOT AUTHORIZED",
        "independent_test_validation_holdout_access": [0, 0, 0],
        "independent_generalization": "NOT ESTABLISHED",
        "discovery_contract_record": record(DISCOVERY_CONTRACT),
        "amended_authorization_record": record(AMENDED_AUTHORIZATION),
        "amended_family_budget_record": record(AMENDED_FAMILY_BUDGET_CSV),
        "amended_execution_plan_record": record(AMENDED_EXECUTION_PLAN),
        "exclusion_registry_record": record(EXCLUSION_REGISTRY),
        "continuity_record": record(CONTINUITY),
        "preflight_record": record(PREFLIGHT),
        "manifest_record": record(MANIFEST),
        "future_combined_model_brand": FUTURE_BRAND,
        "next_gate": "STAGE 12C-1N — RESUME FULL CAMPAIGN AGAINST THE AMENDED PLAN",
    }
    frozen_write(AUDIT, canonical_json(audit))

    for path in (DISCOVERY_CONTRACT, AMENDED_AUTHORIZATION, AMENDED_FAMILY_BUDGET_JSON,
                 CONTINUITY, PREFLIGHT, MANIFEST, AUDIT):
        require(path.read_bytes() == canonical_json(load_json(path)), f"canonical output replay: {path.name}")

    print(f"\n{'Stage':<52}: {STAGE} — SITE-ELIGIBILITY DISCOVERY")
    print(f"{'Status':<52}: PASS / FROZEN")
    print(f"{'Frozen ceiling / amended sites':<52}: {total_ceiling} / {total_eligible} "
          f"(reduction {total_ceiling - total_eligible})")
    print(f"{'Ceiling exceeded':<52}: NO")
    print(f"{'Frozen / amended batches':<52}: {len(frozen_plan)} / {total_batches}")
    print(f"{'Excluded cells':<52}: {len(exclusion_rows)} "
          f"({', '.join(sorted({r['cell_type'] for r in exclusion_rows})) or 'none'})")
    print(f"{'Completed batches preserved / resimulated':<52}: {checked} / 0")
    print(f"{'12C-1K / 12C-1E / 12C-1L modified':<52}: NO / NO / NO")
    print(f"{'Training / selection / inference':<52}: NOT AUTHORIZED")
    print(f"{'TEST / VALIDATION / HOLDOUT access':<52}: 0 / 0 / 0")
    print(f"{'Audit':<52}: {AUDIT}")
    print(f"{'Audit SHA':<52}: {sha256(AUDIT)}")
    print(f"{'Next gate':<52}: STAGE 12C-1N — RESUME AGAINST AMENDED PLAN")


def status() -> None:
    print(f"STAGE {STAGE} — SITE-ELIGIBILITY DISCOVERY STATUS")
    if not MANIFEST.is_file() or not AUDIT.is_file():
        print("Status                    : NOT FROZEN")
        print(f"Expected audit            : {AUDIT}")
        return
    manifest = load_json(MANIFEST)
    audit = load_json(AUDIT)
    require(manifest.get("status") == "PASS" and audit.get("status") == "PASS", "frozen status")
    require(audit.get("manifest_record", {}).get("sha256") == sha256(MANIFEST), "manifest audit anchor")
    verify_manifest_outputs(manifest, STAGE, 10)
    print("Status                    : PASS / FROZEN")
    print(f"Frozen / amended sites    : {audit['frozen_amended_sites']}")
    print(f"Frozen / amended batches  : {audit['frozen_amended_batches']}")
    print(f"Ceiling exceeded          : {audit['ceiling_exceeded']}")
    print(f"Excluded cells            : {audit['excluded_cell_count']} {audit['excluded_cell_types']}")
    print(f"Completed batches kept    : {audit['completed_batches_verified']}")
    print(f"Upstream modified         : {audit['frozen_12c1k_outputs_modified']}")
    print(f"Next gate                 : {audit['next_gate']}")
    print(f"Audit                     : {AUDIT}")
    print(f"Audit SHA                 : {sha256(AUDIT)}")


def self_test() -> None:
    require(math.ceil(26560 / BATCH_SITES) == 415, "AES amended batch arithmetic")
    require(math.ceil(26565 / BATCH_SITES) == 416, "AES frozen batch arithmetic")
    require(math.ceil(9125 / BATCH_SITES) == 143, "SHA256 amended batch arithmetic")
    require(math.ceil(9127 / BATCH_SITES) == 143, "SHA256 frozen batch arithmetic")
    require(math.ceil(18392 / BATCH_SITES) == 288, "HMAC batch arithmetic")
    require(math.ceil(9665 / BATCH_SITES) == 152, "picorv32 batch arithmetic")
    sample = {"opentitan_hmac_sha256": 18392, "picorv32_cpu": 9665,
              "secworks_aes": 26560, "secworks_sha256": 9125}
    rows, plan = build_amended_plan(sample)
    require(len(plan) == 288 + 152 + 415 + 143, "amended plan batch total")
    require(sum(int(r["maximum_sites"]) for r in rows) == 63742, "amended site total")
    first = [r for r in plan if r["family_id"] == "opentitan_hmac_sha256"]
    require(first[0]["global_batch_id"] == 0 and first[-1]["global_batch_id"] == 287,
            "opentitan global batch range preserved")
    second = [r for r in plan if r["family_id"] == "picorv32_cpu"]
    require(second[0]["global_batch_id"] == 288 and second[-1]["global_batch_id"] == 439,
            "picorv32 global batch range preserved")
    require(sum(int(r["site_count_maximum"]) for r in plan if r["family_id"] == "secworks_aes") == 26560,
            "AES plan site total")
    require(sum(int(r["site_count_maximum"]) for r in plan if r["family_id"] == "secworks_sha256") == 9125,
            "SHA256 plan site total")
    print(f"Stage {STAGE} self-test: PASS")


def locked_execute() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    with LOCK_FILE.open("a+", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            stop(f"Stage {STAGE} execution lock is held by another process")
        execute()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--status", action="store_true")
    mode.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.status:
        status()
    elif args.self_test:
        self_test()
    else:
        locked_execute()


if __name__ == "__main__":
    main()
