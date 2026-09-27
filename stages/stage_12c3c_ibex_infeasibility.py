#!/usr/bin/env python3
"""Stage 12C-3C: ibex_cpu capture infeasibility determination.

Stage 12C-3A authorized capture of both INDEPENDENT_CIRCUIT_TEST families.
Stage 12C-3B captured ``secworks_chacha`` successfully.  This stage records,
with reproducible evidence, that ``ibex_cpu`` CANNOT be captured from the
acquired snapshot without authoring substitute RTL - and refuses to author it.

What was attempted
------------------
Four escalating elaboration attempts with the contract-pinned toolchain:

  1. yosys legacy Verilog frontend on the ``ibex_core.f`` file list
     -> syntax error: the frontend rejects a ternary in a function return
        (``prim_util_pkg.sv:44``)
  2. verilator lint with the same file list
     -> missing verification-only include ``dv_fcov_macros.svh``
  3. verilator lint + no-op verification-macro stubs
     -> the ``.f`` file list is incomplete: ``ibex_cheriot_pkg``,
        ``ibex_csr``, ``prim_cipher_pkg`` referenced but not listed
  4. ``read_slang`` (full SystemVerilog frontend) over every file in the
     archive, packages first
     -> SystemVerilog parses correctly, but the design references packages and
        macros that DO NOT EXIST anywhere in the acquired snapshot

Root cause
----------
The Stage 12C-1C acquisition captured a 382 KB ibex snapshot at revision
e9f55342edbd27e9e17a0e41b1c95a81abb5eac8.  That snapshot is not self-contained
for gate-level synthesis: it references OpenTitan infrastructure that lives in
the wider lowRISC repository, not in the ibex tree.  Missing dependencies
recorded below.

Why substitute RTL was refused
------------------------------
Authoring stand-in implementations for ``lc_ctrl_pkg``, the RAM primitive
packages and ``PRIM_FLOP_SPARSE_FSM`` would mean injecting faults into logic
written by this project rather than into ibex.  Any resulting per-circuit
measurement would partly characterise the substitute, and it would enter a
ONE-SHOT sealed evaluation where it could never be corrected.  The frozen
handoff rule "never improve the appearance of the result by changing the
evidence" applies directly.

Consequence for acceptance
--------------------------
The frozen Stage 12C-1A contract requires the per-circuit floor to be met on
BOTH independent test circuits.  One of the two cannot be captured from the
acquired evidence, so the acceptance outcome is NOT MET - determined by
capture infeasibility, not by a measured shortfall.  Stage 12C-2H had already
frozen a predicted overall outcome of NOT MET before any seal was broken.

This stage performs no synthesis, no simulation and no evaluation.  It records
the attempts, the missing dependencies, and the disposition.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import subprocess
import tarfile
from pathlib import Path
from typing import Any

import stage_12c2c_candidate_training as base


STAGE = "12C-3C"
FAMILY = "ibex_cpu"
ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "config/v2_2"
RESULT1 = ROOT / "results/circuitsage_hmac_v2_12c1"
RESULT3 = ROOT / "results/circuitsage_hmac_v2_12c3"
WORK = RESULT3 / "ibex_infeasibility_12c3c"
LOCK_FILE = WORK / ".stage_12c3c.lock"

AUTH_3A = CONFIG / "circuitsage_hmac_v2_2_independent_test_capture_authorization_12c3a.json"
AUDIT_3A = RESULT3 / "circuitsage_hmac_v2_2_test_capture_authorization_freeze_12c3a.json"
AUDIT_3B = RESULT3 / "circuitsage_hmac_v2_2_chacha_capture_freeze_12c3b.json"
MANIFEST_3B = RESULT3 / "circuitsage_hmac_v2_2_chacha_capture_manifest_12c3b.json"
SOURCE_3B = ROOT / "stage_12c3b_chacha_capture.py"
REGISTRY_1C = (RESULT1 / "multicircuit_corpus_12c1c"
               / "circuitsage_hmac_v2_2_acquired_family_registry_12c1c.csv")
ARCHIVES_1C = RESULT1 / "multicircuit_corpus_12c1c/archives"
ACCEPTANCE_1A = CONFIG / "circuitsage_hmac_v2_2_acceptance_contract_12c1a.json"
PREDICTIONS_2H = CONFIG / "circuitsage_hmac_v2_2_sealed_circuit_predictions_12c2h.json"

PINNED = {
    AUTH_3A: "36a9f803baae1621d941695d22dfd3dd3f9396e859a529989e763b5382aee272",
    AUDIT_3A: "3613ef29f3d8f8cdacb2c5d4126e28137991f592b85fe792b7474a4cfd9b315a",
    AUDIT_3B: "c517a208021735bd3b092361594447a14de72146cd7464e14eb310530978d6be",
    MANIFEST_3B: "87a18d8bb71f672a47a8b085f767ef14e0b8ca50515030d29f9c8a96e56b9ac2",
    SOURCE_3B: "91192d9e6ec9d840307543d124fda21f87cb77f9dbd5e13b1c39ca203e7ba151",
    ACCEPTANCE_1A: "9c8eec4d85957c4408ac59e0c8760af90c91d0667c995d91ac969b5a8f205f26",
    PREDICTIONS_2H: "26d4e114dd2cfbcf1708448edf971039ecda7cbf62e1f7b15bdfcb78e2d89cf3",
}

ATTEMPTS = WORK / "circuitsage_hmac_v2_2_ibex_elaboration_attempts_12c3c.csv"
MISSING = WORK / "circuitsage_hmac_v2_2_ibex_missing_dependencies_12c3c.csv"
SNAPSHOT = WORK / "circuitsage_hmac_v2_2_ibex_snapshot_inventory_12c3c.json"
DETERMINATION = CONFIG / "circuitsage_hmac_v2_2_ibex_capture_infeasibility_12c3c.json"
REFUSAL = WORK / "circuitsage_hmac_v2_2_substitute_rtl_refusal_12c3c.json"
DISPOSITION = WORK / "circuitsage_hmac_v2_2_acceptance_disposition_12c3c.json"
REPORT = WORK / "circuitsage_hmac_v2_2_ibex_infeasibility_report_12c3c.md"
MANIFEST = RESULT3 / "circuitsage_hmac_v2_2_ibex_infeasibility_manifest_12c3c.json"
AUDIT = RESULT3 / "circuitsage_hmac_v2_2_ibex_infeasibility_freeze_12c3c.json"

HOLDOUT = "serv_cpu"
FUTURE_BRAND = base.FUTURE_BRAND

stop, require, now = base.stop, base.require, base.now
canonical_json, sha256, rel = base.canonical_json, base.sha256, base.rel
record, load_json, read_csv, csv_bytes = base.record, base.load_json, base.read_csv, base.csv_bytes
frozen_write = base.frozen_write

ELABORATION_ATTEMPTS = [
    {"attempt": 1, "frontend": "yosys legacy Verilog frontend",
     "file_list": "ibex_core.f (17 files) + 7 prim files",
     "defines": "SYNTHESIS, YOSYS",
     "outcome": "FAIL",
     "blocker": "prim_util_pkg.sv:44 syntax error: ternary in function return "
                "'return (value == 1) ? 1 : $clog2(value);'",
     "blocker_class": "TOOLCHAIN FRONTEND LIMITATION",
     "resolvable_without_authoring_rtl": "YES - use read_slang instead"},
    {"attempt": 2, "frontend": "verilator --lint-only",
     "file_list": "ibex_core.f (17 files) + 3 prim packages",
     "defines": "SYNTHESIS",
     "outcome": "FAIL",
     "blocker": "ibex_controller.sv:12 cannot find include 'dv_fcov_macros.svh'; "
                "16 downstream errors from undefined DV_FCOV_SIGNAL",
     "blocker_class": "MISSING VERIFICATION-ONLY INCLUDE",
     "resolvable_without_authoring_rtl": "YES - no-op macro stub adds no logic"},
    {"attempt": 3, "frontend": "verilator --lint-only",
     "file_list": "ibex_core.f + prim packages + no-op verification stubs",
     "defines": "SYNTHESIS",
     "outcome": "FAIL",
     "blocker": "ibex_core.f is incomplete: prim_cipher_pkg, ibex_cheriot_pkg "
                "types (cheriot_op_t, cheriot_cap_field_e, ...) referenced but not listed",
     "blocker_class": "INCOMPLETE VENDOR FILE LIST",
     "resolvable_without_authoring_rtl": "YES - supply all archive files"},
    {"attempt": 4, "frontend": "yosys read_slang (full SystemVerilog frontend)",
     "file_list": "all 194 .sv in archive, 15 packages first, + stubs",
     "defines": "SYNTHESIS",
     "outcome": "FAIL",
     "blocker": "SystemVerilog parses correctly, but design references packages "
                "and macros absent from the acquired snapshot",
     "blocker_class": "SNAPSHOT NOT SELF-CONTAINED",
     "resolvable_without_authoring_rtl": "NO - requires authoring substitute RTL "
                                         "or re-acquiring a wider source tree"},
]

MISSING_DEPS = [
    {"dependency": "lc_ctrl_pkg", "kind": "SystemVerilog package",
     "upstream_home": "OpenTitan hw/ip/lc_ctrl",
     "role": "life-cycle controller encoding",
     "substitutable_without_changing_semantics": "NO"},
    {"dependency": "prim_ram_1p_pkg", "kind": "SystemVerilog package",
     "upstream_home": "OpenTitan hw/ip/prim",
     "role": "single-port RAM configuration types",
     "substitutable_without_changing_semantics": "NO"},
    {"dependency": "prim_ram_1r1w_pkg", "kind": "SystemVerilog package",
     "upstream_home": "OpenTitan hw/ip/prim",
     "role": "1R1W RAM configuration types",
     "substitutable_without_changing_semantics": "NO"},
    {"dependency": "PRIM_FLOP_SPARSE_FSM", "kind": "compiler macro",
     "upstream_home": "OpenTitan prim_flop_macros.sv",
     "role": "sparse-encoded FSM register instantiation",
     "substitutable_without_changing_semantics": "NO"},
    {"dependency": "RVFI", "kind": "global define",
     "upstream_home": "ibex formal verification harness",
     "role": "RISC-V formal interface gate",
     "substitutable_without_changing_semantics": "PARTIAL - define only"},
    {"dependency": "prim_buf / prim_flop / prim_clock_gating / prim_ram_1p",
     "kind": "technology primitive modules",
     "upstream_home": "OpenTitan per-target prim libraries",
     "role": "buffers, flops, clock gates, memory macros",
     "substitutable_without_changing_semantics": "NO"},
    {"dependency": "dv_fcov_macros.svh", "kind": "verification include",
     "upstream_home": "OpenTitan dv/sv/dv_utils",
     "role": "functional-coverage macros",
     "substitutable_without_changing_semantics": "YES - verification-only, "
                                                 "no-op stub adds no logic"},
]


def inventory() -> dict[str, Any]:
    reg = {r["family_id"]: r for r in read_csv(REGISTRY_1C)}
    r = reg[FAMILY]
    archive = next(ARCHIVES_1C.glob(f"{FAMILY}-*.tar.gz"))
    require(sha256(archive) == r["archive_sha256"], "ibex archive SHA")
    with tarfile.open(archive) as t:
        names = t.getnames()
    def count(suffix): return sum(1 for n in names if n.endswith(suffix))
    return {
        "family_id": FAMILY, "partition": r["partition"],
        "repository": r["repository"], "revision": r["revision"],
        "spdx_license": r["spdx_license"],
        "archive": rel(archive), "archive_sha256": r["archive_sha256"],
        "archive_bytes": int(r["archive_bytes"]),
        "entries": len(names),
        "systemverilog_modules": count(".sv"),
        "systemverilog_headers": count(".svh"),
        "fusesoc_core_files": count(".core"),
        "file_lists": count(".f"),
        "rtl_directories": sorted({str(Path(n).parent) for n in names
                                   if n.endswith(".sv")}),
        "self_contained_for_gate_level_synthesis": False,
    }


def execute() -> None:
    require(ROOT.name == "vlsi_fault_detection_v2", f"place script in project root, not {ROOT}")
    outputs = (ATTEMPTS, MISSING, SNAPSHOT, DETERMINATION, REFUSAL,
               DISPOSITION, REPORT, MANIFEST, AUDIT)
    require(not any(p.exists() for p in outputs),
            f"frozen Stage {STAGE} output exists; use --status")

    print("FROZEN INPUT VERIFICATION", flush=True)
    for path, expected in PINNED.items():
        require(path.is_file(), f"missing frozen input: {rel(path)}")
        require(sha256(path) == expected, f"frozen input SHA changed: {path.name}")
    auth = load_json(AUTH_3A)
    require(FAMILY in auth["authorized_families"], f"{FAMILY} was authorized")
    b = load_json(AUDIT_3B)
    require(b["status"] == "PASS", "12C-3B frozen")
    require(b["family_id"] == "secworks_chacha", "12C-3B is the chacha capture")
    print(f"  {len(PINNED)} frozen inputs; 12C-3B captured "
          f"{b['family_id']}{'':<12}: OK", flush=True)

    existing = sorted(p.name for p in CONFIG.glob("*.json"))
    require(not any("ibex_capture_infeasibility" in n for n in existing),
            "an infeasibility determination already exists")
    print(f"  config/v2_2 enumerated ({len(existing)} contracts); none prior"
          f"{'':<13}: OK", flush=True)

    # no capture artifact for ibex must exist
    leaked = [rel(p) for p in ROOT.glob(f"**/*{FAMILY}*")
              if p.is_file() and any(m in rel(p) for m in
                                     ("raw_batches", "signatures", "instrumented",
                                      "_capture_", "obj_dir"))]
    require(not leaked, f"unexpected ibex capture artifacts: {leaked[:3]}")
    print(f"  no ibex capture artifact exists{'':<31}: OK", flush=True)

    print("\nSNAPSHOT INVENTORY", flush=True)
    inv = inventory()
    print(f"  revision {inv['revision'][:12]}  {inv['archive_bytes']//1024} KB  "
          f"{inv['systemverilog_modules']} .sv  "
          f"{inv['systemverilog_headers']} .svh", flush=True)

    print("\nELABORATION ATTEMPTS", flush=True)
    for a in ELABORATION_ATTEMPTS:
        print(f"  {a['attempt']}. {a['frontend']:<44} {a['outcome']}", flush=True)
        print(f"     {a['blocker_class']}", flush=True)

    print("\nMISSING DEPENDENCIES", flush=True)
    for d in MISSING_DEPS:
        print(f"  {d['dependency']:<52} {d['kind']}", flush=True)

    acceptance = load_json(ACCEPTANCE_1A)
    preds = load_json(PREDICTIONS_2H)
    created = now()

    determination = {
        "determination_version": "CIRCUITSAGE-HMAC-V2.2-IBEX-CAPTURE-INFEASIBILITY-12C3C-v1",
        "stage": STAGE, "status": "FROZEN", "created_at": created,
        "family_id": FAMILY, "partition": "INDEPENDENT_CIRCUIT_TEST",
        "capture_authorized_by": "12C-3A",
        "capture_performed": False,
        "determination": "NOT CAPTURABLE FROM THE ACQUIRED SNAPSHOT",
        "root_cause": (
            "the Stage 12C-1C ibex snapshot is not self-contained for gate-level "
            "synthesis; it references OpenTitan infrastructure packages, macros and "
            "technology primitives that are absent from the acquired tree"),
        "elaboration_attempts": len(ELABORATION_ATTEMPTS),
        "frontends_tried": sorted({a["frontend"] for a in ELABORATION_ATTEMPTS}),
        "systemverilog_parses_correctly": True,
        "blocker_is_syntax": False,
        "blocker_is_missing_source": True,
        "missing_dependency_count": len(MISSING_DEPS),
        "resolvable_without_authoring_rtl": False,
        "toolchain_at_fault": False,
        "upstream_rtl_modified": False,
        "remediation_options_not_taken": [
            "author substitute RTL for the missing packages and primitives",
            "re-acquire a wider OpenTitan source tree (changes the frozen corpus)",
            "capture a reduced ibex_core configuration (a different circuit from "
            "the one predicted on in 12C-2H)",
        ],
    }

    refusal = {
        "refusal_version": "CIRCUITSAGE-HMAC-V2.2-SUBSTITUTE-RTL-REFUSAL-12C3C-v1",
        "stage": STAGE, "created_at": created,
        "refused_action": "authoring substitute RTL for missing ibex dependencies",
        "why": (
            "faults would be injected into logic written by this project rather than "
            "into ibex, so any per-circuit measurement would partly characterise the "
            "substitute"),
        "aggravating_factor": (
            "the acceptance evaluation is ONE-SHOT; a measurement contaminated by "
            "substitute logic could never be corrected afterward"),
        "governing_rule": (
            "never improve the appearance of the result by changing the evidence"),
        "what_was_stubbed_and_why_it_is_acceptable": {
            "dv_fcov_macros.svh": "functional-coverage macros, verification-only, "
                                  "expand to nothing when coverage is disabled",
            "formal_tb_frag.svh": "formal-verification fragment, verification-only",
        },
        "stubs_added_design_logic": False,
        "stubs_used_in_any_frozen_measurement": False,
    }

    chacha_pred = next(p for p in preds["predictions"]
                       if p["family_id"] == "secworks_chacha")
    disposition = {
        "disposition_version": "CIRCUITSAGE-HMAC-V2.2-ACCEPTANCE-DISPOSITION-12C3C-v1",
        "stage": STAGE, "created_at": created,
        "per_circuit_floor_requirement": (
            "the frozen 12C-1A contract requires the per-circuit floor to be met on "
            "BOTH independent test circuits"),
        "independent_test_circuits": ["secworks_chacha", "ibex_cpu"],
        "secworks_chacha": {
            "captured": True,
            "all_injected_exact_site_rate": b["all_injected_exact_site_rate"],
            "observable_fraction": b["observable_fraction"],
            "floor_threshold": acceptance["per_circuit_floor"]["all_injected_exact_site_rate_min"],
            "meets_exact_site_floor":
                b["all_injected_exact_site_rate"] >=
                acceptance["per_circuit_floor"]["all_injected_exact_site_rate_min"],
            "frozen_predicted_band": chacha_pred["predicted_exact_site_band"],
            "frozen_predicted_floor": chacha_pred["predicted_meets_floor_0_15"],
        },
        "ibex_cpu": {
            "captured": False,
            "reason": "NOT CAPTURABLE FROM THE ACQUIRED SNAPSHOT",
        },
        "acceptance_outcome": "NOT MET",
        "acceptance_outcome_basis": (
            "one of the two required independent test circuits could not be captured "
            "from the acquired evidence without authoring substitute RTL; the "
            "contract's both-circuits requirement is therefore unsatisfied"),
        "outcome_type": "CAPTURE INFEASIBILITY, NOT A MEASURED SHORTFALL",
        "predicted_before_any_seal_was_broken": preds["predicted_overall_acceptance"],
        "prediction_and_outcome_agree": preds["predicted_overall_acceptance"] == "NOT MET",
        "note_on_agreement": (
            "12C-2H predicted NOT MET for a different reason - a measured ibex "
            "shortfall - so the agreement is on the outcome label only, not on the "
            "mechanism. This distinction must be preserved in any report."),
        "one_shot_evaluation_consumed": False,
        "one_shot_rationale": (
            "no acceptance evaluation was performed against a model; the outcome "
            "follows from capture infeasibility, so the locked evaluation remains "
            "unspent"),
        "holdout_status": "SEALED",
    }

    att_fields = list(ELABORATION_ATTEMPTS[0].keys())
    dep_fields = list(MISSING_DEPS[0].keys())
    att_table = "\n".join(
        f"| {a['attempt']} | {a['frontend']} | {a['blocker_class']} | "
        f"{a['resolvable_without_authoring_rtl']} |" for a in ELABORATION_ATTEMPTS)
    dep_table = "\n".join(
        f"| `{d['dependency']}` | {d['kind']} | {d['upstream_home']} | "
        f"{d['substitutable_without_changing_semantics']} |" for d in MISSING_DEPS)

    report = f"""# Stage {STAGE} — `{FAMILY}` Capture Infeasibility

**Status: PASS / FROZEN — determination only. No synthesis, simulation or evaluation.**

## Determination

**`{FAMILY}` is NOT CAPTURABLE from the acquired snapshot.**

The Stage 12C-1C acquisition captured a {inv['archive_bytes']//1024} KB ibex tree at
revision `{inv['revision'][:12]}` containing {inv['systemverilog_modules']} SystemVerilog
modules. That snapshot is **not self-contained for gate-level synthesis**: it
references OpenTitan infrastructure that lives in the wider lowRISC repository.

## Elaboration attempts

| # | frontend | blocker class | fixable without authoring RTL |
|---|---|---|---|
{att_table}

The SystemVerilog **parses correctly** under `read_slang`. The blocker is missing
source, not syntax, and not the toolchain.

## Missing dependencies

| dependency | kind | upstream home | substitutable |
|---|---|---|---|
{dep_table}

## Substitute RTL was refused

Authoring stand-ins for `lc_ctrl_pkg`, the RAM packages and
`PRIM_FLOP_SPARSE_FSM` would mean **injecting faults into logic written by this
project rather than into ibex**. The resulting measurement would partly
characterise the substitute — inside a ONE-SHOT evaluation where it could never
be corrected.

Two verification-only stubs were used during probing (`dv_fcov_macros.svh`,
`formal_tb_frag.svh`). They expand to nothing, add no design logic, and appear in
**no frozen measurement**.

## Acceptance disposition

| circuit | captured | exact-site | meets 0.15 floor |
|---|---|---|---|
| `secworks_chacha` | YES | **{b['all_injected_exact_site_rate']:.4f}** | **YES** |
| `{FAMILY}` | **NO** | — | — |

**Acceptance outcome: NOT MET** — because one of the two required independent
test circuits could not be captured, not because of a measured shortfall.

Stage 12C-2H predicted NOT MET before any seal was broken. The outcome label
agrees, but **the mechanism differs**: 12C-2H predicted a measured ibex shortfall,
whereas the actual cause is capture infeasibility. That distinction must be
preserved wherever this result is reported.

## What remains unspent

- the **one-shot locked evaluation** was not consumed: no model was evaluated
- `{HOLDOUT}` remains **SEALED**
- `secworks_chacha` capture stands as valid, frozen evidence

## Next gate

Prediction scoring for `secworks_chacha` and the V2.2 closing disposition.
Independent generalization remains **NOT ESTABLISHED**. Future hybrid brand
remains **{FUTURE_BRAND}**.
"""

    frozen_write(ATTEMPTS, csv_bytes(ELABORATION_ATTEMPTS, att_fields))
    frozen_write(MISSING, csv_bytes(MISSING_DEPS, dep_fields))
    frozen_write(SNAPSHOT, canonical_json({
        "inventory_version": "CIRCUITSAGE-HMAC-V2.2-IBEX-SNAPSHOT-INVENTORY-12C3C-v1",
        "stage": STAGE, "created_at": created, **inv}))
    frozen_write(DETERMINATION, canonical_json(determination))
    frozen_write(REFUSAL, canonical_json(refusal))
    frozen_write(DISPOSITION, canonical_json(disposition))
    frozen_write(REPORT, report.encode())

    stage_outputs = (ATTEMPTS, MISSING, SNAPSHOT, DETERMINATION, REFUSAL,
                     DISPOSITION, REPORT)
    manifest = {
        "manifest_version": "CIRCUITSAGE-HMAC-V2.2-IBEX-INFEASIBILITY-MANIFEST-12C3C-v1",
        "stage": STAGE, "status": "PASS", "created_at": created,
        "stage_source": record(Path(__file__).resolve()),
        "frozen_inputs": {rel(p): record(p) for p in PINNED},
        "outputs": {rel(p): record(p) for p in stage_outputs},
        "capture_performed": False,
        "synthesis_calls": 0, "simulation_calls": 0, "fault_injections": 0,
        "substitute_rtl_authored": False,
        "upstream_rtl_modified": False,
        "holdout_access": 0,
    }
    frozen_write(MANIFEST, canonical_json(manifest))

    audit = {
        "audit_version": "CIRCUITSAGE-HMAC-V2.2-IBEX-INFEASIBILITY-FREEZE-12C3C-v1",
        "stage": STAGE, "status": "PASS",
        "family_id": FAMILY,
        "determination": "NOT CAPTURABLE FROM THE ACQUIRED SNAPSHOT",
        "capture_performed": False,
        "elaboration_attempts": len(ELABORATION_ATTEMPTS),
        "systemverilog_parses_correctly": True,
        "blocker_is_missing_source": True,
        "missing_dependencies": [d["dependency"] for d in MISSING_DEPS],
        "substitute_rtl_authored": False,
        "verification_only_stubs_used_in_probing": ["dv_fcov_macros.svh",
                                                    "formal_tb_frag.svh"],
        "stubs_in_frozen_measurement": False,
        "acceptance_outcome": "NOT MET",
        "acceptance_outcome_type": "CAPTURE INFEASIBILITY, NOT MEASURED SHORTFALL",
        "predicted_outcome_12c2h": "NOT MET",
        "prediction_mechanism_differs_from_actual": True,
        "chacha_exact_site_rate": load_json(AUDIT_3B)["all_injected_exact_site_rate"],
        "chacha_meets_floor": True,
        "one_shot_evaluation_consumed": False,
        "holdout_seal": "SEALED",
        "upstream_rtl_modified": False,
        "independent_generalization": "NOT ESTABLISHED",
        "determination_record": record(DETERMINATION),
        "refusal_record": record(REFUSAL),
        "disposition_record": record(DISPOSITION),
        "manifest_record": record(MANIFEST),
        "future_combined_model_brand": FUTURE_BRAND,
        "next_gate": "PREDICTION SCORING AND V2.2 CLOSING DISPOSITION",
    }
    frozen_write(AUDIT, canonical_json(audit))

    print(f"\n{'Stage':<52}: {STAGE} — IBEX CAPTURE INFEASIBILITY")
    print(f"{'Status':<52}: PASS / FROZEN")
    print(f"{'Determination':<52}: NOT CAPTURABLE")
    print(f"{'SystemVerilog parses correctly':<52}: YES")
    print(f"{'Blocker':<52}: MISSING SOURCE, NOT SYNTAX")
    print(f"{'Substitute RTL authored':<52}: NO (refused)")
    print(f"{'Acceptance outcome':<52}: NOT MET (capture infeasibility)")
    print(f"{'One-shot evaluation consumed':<52}: NO")
    print(f"{'Holdout ' + HOLDOUT:<52}: SEALED")
    print(f"{'Audit SHA':<52}: {sha256(AUDIT)}")


def status() -> None:
    print(f"STAGE {STAGE} — IBEX INFEASIBILITY STATUS")
    if not (MANIFEST.is_file() and AUDIT.is_file()):
        print("Status                    : NOT FROZEN")
        return
    a = load_json(AUDIT)
    print("Status                    : PASS / FROZEN")
    print(f"Determination             : {a['determination']}")
    print(f"Missing dependencies      : {a['missing_dependencies']}")
    print(f"Acceptance outcome        : {a['acceptance_outcome']}")
    print(f"Outcome type              : {a['acceptance_outcome_type']}")
    print(f"One-shot consumed         : {a['one_shot_evaluation_consumed']}")
    print(f"Audit SHA                 : {sha256(AUDIT)}")


def self_test() -> None:
    require(len(ELABORATION_ATTEMPTS) == 4, "four documented attempts")
    require(all(a["outcome"] == "FAIL" for a in ELABORATION_ATTEMPTS),
            "all attempts failed")
    require(ELABORATION_ATTEMPTS[-1]["resolvable_without_authoring_rtl"].startswith("NO"),
            "final attempt establishes infeasibility")
    require(len(MISSING_DEPS) >= 5, "missing dependencies enumerated")
    b = load_json(AUDIT_3B)
    require(b["family_id"] == "secworks_chacha", "chacha capture present")
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
