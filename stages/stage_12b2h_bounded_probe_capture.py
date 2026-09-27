#!/usr/bin/env python3
"""Stage 12B-2H: bounded probe-capture execution and dataset freeze.

Executes the Stage 12B-2G-authorized, REPAIR_TRAIN-only probe campaign on
1,024 pilot sites (SA0/SA1), 96 frozen vectors, and one fixed 64-bit probe
bank.  Derived observer netlists are created in an isolated build directory;
the frozen RTL and canonical fault-batch netlists are never modified.

The run is sequential and checkpoint/resume capable.  It freezes deterministic
per-batch and consolidated NPZ datasets.  Fault identity is retained only in
separate target/metadata arrays and is never placed in the behavior feature
arrays.  No model is loaded, trained, selected, calibrated, or evaluated.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import hmac
import io
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import numpy as np
except ImportError as error:
    raise SystemExit("STOP: activate the project .venv (NumPy required)") from error


STAGE = "12B-2H"
ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "config/v2_1_improvement"
RESULT = ROOT / "results/circuitsage_hmac_v2_12b2"
WORK = RESULT / "probe_capture_12b2h"
RAW = WORK / "raw_batches"
PER_BATCH = WORK / "per_batch_npz"
BUILD = ROOT / "build/circuitsage_hmac_v2_12b2/probe_capture_12b2h"
CHECKPOINT = WORK / "circuitsage_hmac_v2_1_probe_capture_checkpoint_12b2h.json"
DATASET = WORK / "circuitsage_hmac_v2_1_probe_response_features_12b2h.npz"
TARGETS = WORK / "circuitsage_hmac_v2_1_probe_response_targets_12b2h.npz"
METRICS = WORK / "circuitsage_hmac_v2_1_probe_observability_metrics_12b2h.json"
SCHEMA = WORK / "circuitsage_hmac_v2_1_probe_response_schema_12b2h.json"
MANIFEST = RESULT / "circuitsage_hmac_v2_1_probe_capture_dataset_manifest_12b2h.json"
AUDIT = RESULT / "circuitsage_hmac_v2_1_probe_capture_dataset_integrity_freeze_12b2h.json"

SOURCE_2G = ROOT / "stage_12b2g_probe_capture_authorization.py"
AUTH_DIR = RESULT / "probe_capture_authorization_12b2g"
EXECUTION_PLAN = AUTH_DIR / "circuitsage_hmac_v2_1_probe_capture_execution_plan_12b2g.csv"
PREFLIGHT_2G = AUTH_DIR / "circuitsage_hmac_v2_1_probe_capture_preflight_12b2g.json"
EXECUTION_CONTRACT = CONFIG / "circuitsage_hmac_v2_1_probe_capture_execution_contract_12b2g.json"
AUTHORIZATION = CONFIG / "circuitsage_hmac_v2_1_probe_capture_execution_authorization_12b2g.json"
MANIFEST_2G = RESULT / "circuitsage_hmac_v2_1_probe_capture_authorization_manifest_12b2g.json"
AUDIT_2G = RESULT / "circuitsage_hmac_v2_1_probe_capture_authorization_freeze_12b2g.json"

PROBE_DIR = RESULT / "probe_discovery_12b2f"
PROBE_BANK_JSON = PROBE_DIR / "circuitsage_hmac_v2_1_global_probe_bank_12b2f.json"
PROBE_BANK_CSV = PROBE_DIR / "circuitsage_hmac_v2_1_global_probe_bank_12b2f.csv"
CONSISTENCY_2F = PROBE_DIR / "circuitsage_hmac_v2_1_cross_batch_probe_consistency_12b2f.json"
SELECTED_VECTORS = RESULT / "adaptive_vector_selection_12b2d/circuitsage_hmac_v2_1_selected_adaptive_vectors_12b2d.npz"
PILOT_SITES = RESULT / "adaptive_vector_pool_12b2b/circuitsage_hmac_v2_1_pilot_screening_sites_12b2b.csv"

PINNED = {
    SOURCE_2G: "6e23c74e93382204be76ad787711ca705202eb389743945582deebf6de1f81c1",
    PROBE_BANK_JSON: "23cba70a6fa199bb59c62b84ad2546603903abc68cd9d5e7eb7e2e15e5f95632",
    PROBE_BANK_CSV: "e6abe61682aaaccbaea8bbd6d90f1036e1bac8835a5999590bc545c396d5ab51",
    CONSISTENCY_2F: "9d9b349ed7f615867a24e911676d626dc6c612774457a6d2b8b5422ea14d794c",
    SELECTED_VECTORS: "be9df0a3a70e61b54fc793439328bbcc307b9373fd881b0e97d500919ec27bff",
    PILOT_SITES: "87eceaf80a800cf60b8a7e3cb20e0ec3f934158dc35ae3d10f73e8cf9d25313c",
}

BATCHES = 45
SITES = 1024
FAULTS = 2048
VECTORS = 96
PROBES = 64
SNAPSHOTS = 16
SAMPLE_CYCLES = [0, 1, 2, 4, 8, 16, 32, 64, 96, 128, 160, 192, 224, 256, 320]
ENABLED_TRANSACTIONS = FAULTS * VECTORS
TIMEOUT_CYCLES = 2000
EXPECTED_BASELINE_CYCLES = 343
MIN_FREE_GIB = 5
CSV_FIELDS = [
    "batch_id", "run_type", "site_id", "selector", "stuck_value",
    "vector_rank", "source_vector_index", "cycles", "timed_out",
    "busy_first_cycle", "busy_last_cycle", "done_cycle", "unknown",
    "expected_digest", "actual_digest", "snapshot_hex", "toggle_hex",
]


def stop(message: str) -> None:
    raise SystemExit(f"STOP: {message}")


def require(condition: bool, message: str) -> None:
    if not condition:
        stop(message)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode()


def load_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing JSON: {rel(path)}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON object required: {rel(path)}")
    return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(canonical_json(value))
    os.replace(temporary, path)


def frozen_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.exists(), f"refusing to overwrite frozen output: {rel(path)}")
    temporary = path.with_name(path.name + ".tmp")
    require(not temporary.exists(), f"stale temporary output: {rel(temporary)}")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def record(path: Path) -> dict[str, Any]:
    return {"path": rel(path), "sha256": sha256(path), "bytes": path.stat().st_size}


def resolve_record(item: dict[str, Any]) -> Path:
    value = item.get("path")
    require(isinstance(value, str) and value, "artifact record path")
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def verify_record(item: dict[str, Any], expected: Path, label: str) -> None:
    require(isinstance(item, dict), f"{label} record")
    path = resolve_record(item)
    require(path.resolve() == expected.resolve(), f"{label} path")
    require(path.is_file(), f"missing {label}: {rel(path)}")
    require(item.get("sha256") == sha256(path), f"{label} SHA")
    require(int(item.get("bytes", -1)) == path.stat().st_size, f"{label} size")


def verify_stage_2g_outputs() -> tuple[dict[str, Any], dict[str, Any]]:
    for path in (EXECUTION_PLAN, PREFLIGHT_2G, EXECUTION_CONTRACT, AUTHORIZATION, MANIFEST_2G, AUDIT_2G):
        require(path.is_file(), f"Stage 12B-2G is not complete; missing {rel(path)}")
    audit = load_json(AUDIT_2G)
    manifest = load_json(MANIFEST_2G)
    authorization = load_json(AUTHORIZATION)
    contract = load_json(EXECUTION_CONTRACT)
    require(audit.get("status") == "PASS", "12B-2G audit status")
    require(audit.get("authorization_status") == "FROZEN", "12B-2G authorization status")
    require(audit.get("bounded_probe_capture") == "AUTHORIZED / NOT STARTED", "capture authorization")
    require(audit.get("enabled_transactions") == ENABLED_TRANSACTIONS, "authorized transaction count")
    require(audit.get("identity_independent_schedule") == "FROZEN / PASS", "identity-independent schedule")
    require(audit.get("fault_selector_value_raw_as_features") == "PROHIBITED", "leakage boundary")
    require(audit.get("repair_site_test") == "LOCKED / NOT AUTHORIZED", "repair site-test boundary")
    require(audit.get("validation_access") == 0 and audit.get("holdout_access") == 0, "protected partition access")
    verify_record(audit["manifest"], MANIFEST_2G, "12B-2G manifest")
    require(manifest.get("status") == "PASS", "12B-2G manifest status")
    outputs = manifest.get("outputs")
    require(isinstance(outputs, dict), "12B-2G output registry")
    for path in (EXECUTION_PLAN, PREFLIGHT_2G, EXECUTION_CONTRACT, AUTHORIZATION):
        verify_record(outputs[rel(path)], path, f"12B-2G {path.name}")
    require(authorization.get("bounded_probe_capture") == "AUTHORIZED / NOT STARTED", "authorization state")
    require(authorization.get("model_training") == "NOT AUTHORIZED", "model-training boundary")
    require(authorization.get("repair_calibration") == "LOCKED / NOT AUTHORIZED", "repair-calibration boundary")
    require(authorization.get("repair_site_test") == "LOCKED / NOT AUTHORIZED", "repair-test boundary")
    require(authorization.get("validation") == "PROHIBITED" and authorization.get("holdout") == "PROHIBITED", "protected partitions")
    require(contract.get("status") == "FROZEN" and contract.get("scope") == "REPAIR_TRAIN PILOT ONLY", "execution scope")
    require(contract.get("selected_vectors") == VECTORS and contract.get("probe_bits") == PROBES, "contract dimensions")
    require(contract.get("snapshots_per_transaction") == SNAPSHOTS, "snapshot count")
    require(contract.get("sample_cycles_after_start") == SAMPLE_CYCLES, "sample schedule")
    require(contract.get("checkpoint_resume") == "REQUIRED AFTER EACH CANONICAL BATCH", "checkpoint requirement")
    return audit, contract


def verify_inputs() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    print("STAGE 12B-2H — BOUNDED PROBE-CAPTURE EXECUTION")
    print("FROZEN INPUT VERIFICATION")
    for path, expected in PINNED.items():
        require(path.is_file(), f"missing frozen input: {rel(path)}")
        require(sha256(path) == expected, f"frozen input SHA: {rel(path)}")
        print(f"  {path.name:<91}: OK", flush=True)
    _, contract = verify_stage_2g_outputs()
    with PROBE_BANK_CSV.open(newline="", encoding="utf-8") as stream:
        probes = list(csv.DictReader(stream))
    require(len(probes) == PROBES, "probe-bank size")
    require([int(row["probe_bit"]) for row in probes] == list(range(PROBES)), "probe-bank ordering")
    require(len({int(row["bit_id"]) for row in probes}) == PROBES, "probe bit-ID uniqueness")
    require(shutil.which("yosys") is not None and shutil.which("verilator") is not None, "toolchain unavailable")
    require(shutil.disk_usage(ROOT).free >= MIN_FREE_GIB * 1024**3, f"less than {MIN_FREE_GIB} GiB free disk")
    print("  Stage 12B-2G authorization, leakage controls and protected partitions                 : PASS")
    return contract, probes


def load_vectors() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(SELECTED_VECTORS, allow_pickle=False) as archive:
        required = {"key_u8", "message_u8", "selection_rank", "selection_round", "vector_index"}
        require(set(archive.files) == required, "selected-vector NPZ members")
        keys = np.asarray(archive["key_u8"], dtype=np.uint8)
        messages = np.asarray(archive["message_u8"], dtype=np.uint8)
        source_indices = np.asarray(archive["vector_index"], dtype=np.int32)
        ranks = np.asarray(archive["selection_rank"], dtype=np.int32)
    require(keys.shape == (VECTORS, 32) and messages.shape == (VECTORS, 32), "selected-vector shapes")
    require(np.array_equal(ranks, np.arange(VECTORS, dtype=np.int32)), "selected-vector rank order")
    require(len(set(source_indices.tolist())) == VECTORS, "selected source-vector uniqueness")
    digests = np.vstack([
        np.frombuffer(hmac.new(keys[i].tobytes(), messages[i].tobytes(), hashlib.sha256).digest(), dtype=np.uint8)
        for i in range(VECTORS)
    ])
    return keys, messages, source_indices, digests


def find_sites(value: Any) -> list[dict[str, Any]] | None:
    if isinstance(value, dict):
        for child in value.values():
            found = find_sites(child)
            if found is not None:
                return found
    if isinstance(value, list):
        if value and isinstance(value[0], dict) and {"fault_site_id", "selector_code"} <= set(value[0]):
            return value
        for child in value:
            found = find_sites(child)
            if found is not None:
                return found
    return None


def load_pilot_plan() -> tuple[list[dict[str, Any]], dict[int, list[dict[str, Any]]]]:
    with PILOT_SITES.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    require(len(rows) == SITES, "pilot-site row count")
    require([int(row["pilot_rank"]) for row in rows] == list(range(SITES)), "pilot-site rank order")
    wanted = {row["site_id"]: row for row in rows}
    require(len(wanted) == SITES, "duplicate pilot site")
    batches: dict[int, list[dict[str, Any]]] = {}
    found: set[str] = set()
    consistency = load_json(CONSISTENCY_2F)
    entries = consistency.get("batches")
    require(isinstance(entries, list) and len(entries) == BATCHES, "12B-2F consistency entries")
    for batch_id, entry in enumerate(entries):
        key = f"{batch_id:03d}"
        require(entry.get("batch_id") == batch_id and entry.get("probe_bank_exact") is True, f"Batch {key} consistency")
        json_path = ROOT / f"build/hmac_fault_batches_canonical_11c4g/batch_{key}/opentitan_hmac_sha256_msg32_faultbatch{key}.json"
        mapping_path = ROOT / f"results/hmac_fault_campaign_11c4/canonical_batches/batch_{key}/hmac_fault_batch_{key}_mapping.json"
        verify_record(entry["canonical_json"], json_path, f"Batch {key} canonical JSON")
        verify_record(entry["mapping"], mapping_path, f"Batch {key} mapping")
        mapping_sites = find_sites(load_json(mapping_path))
        expected = 311 if batch_id == 44 else 512
        require(mapping_sites is not None and len(mapping_sites) == expected, f"Batch {key} mapping count")
        require([int(site["selector_code"]) for site in mapping_sites] == list(range(expected)), f"Batch {key} selector order")
        selected: list[dict[str, Any]] = []
        for site in mapping_sites:
            site_id = str(site["fault_site_id"])
            if site_id in wanted:
                require(site_id not in found, f"pilot site mapped twice: {site_id}")
                found.add(site_id)
                selected.append({
                    "site_id": site_id,
                    "site_index": int(wanted[site_id]["site_index"]),
                    "pilot_rank": int(wanted[site_id]["pilot_rank"]),
                    "selector": int(site["selector_code"]),
                    "batch_id": batch_id,
                })
        require(selected, f"Batch {key} has no authorized pilot site")
        batches[batch_id] = sorted(selected, key=lambda row: row["pilot_rank"])
    require(found == set(wanted), f"unmapped pilot sites: {len(set(wanted) - found)}")
    ordered = sorted((row for values in batches.values() for row in values), key=lambda row: row["pilot_rank"])
    require(len(ordered) == SITES and len(batches) == BATCHES, "pilot execution coverage")
    return ordered, batches


def memory_file(path: Path, rows: np.ndarray) -> None:
    path.write_text("".join(row.tobytes().hex() + "\n" for row in rows), encoding="ascii")


def sv_quote(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def observer_json(canonical: Path, module_name: str, probe_bits: list[int]) -> bytes:
    design = copy.deepcopy(load_json(canonical))
    modules = design.get("modules")
    require(isinstance(modules, dict) and set(modules) == {module_name}, f"{module_name} module set")
    module = modules[module_name]
    require("probe_o" not in module.get("ports", {}) and "probe_o" not in module.get("netnames", {}), "probe_o collision")
    available: set[int] = set()
    for net in module.get("netnames", {}).values():
        available.update(bit for bit in net.get("bits", []) if isinstance(bit, int))
    require(set(probe_bits) <= available, f"{module_name} missing probe bits")
    module["ports"]["probe_o"] = {"direction": "output", "bits": probe_bits}
    module["netnames"]["probe_o"] = {"hide_name": 0, "bits": probe_bits, "attributes": {}}
    return canonical_json(design)


def sample_case_lines() -> str:
    lines = []
    for slot, cycle in enumerate(SAMPLE_CYCLES[1:], start=1):
        lines.append(f"        {cycle}: begin snapshot_work[{slot}]=probe_o; snapshot_written[{slot}]=1; end")
    return "\n".join(lines)


def generate_testbench(batch_id: int, selected: list[dict[str, Any]], csv_path: Path,
                       key_mem: Path, message_mem: Path, digest_mem: Path,
                       source_index_mem: Path) -> str:
    key = f"{batch_id:03d}"
    top = f"tb_v21_probe_capture_batch_{key}"
    dut = f"opentitan_hmac_sha256_msg32_faultbatch{key}"
    site_init: list[str] = []
    for index, site in enumerate(selected):
        site_init += [
            f"    selected_selectors[{index}] = 9'd{site['selector']};",
            f"    selected_site_ids[{index}] = \"{site['site_id']}\";",
        ]
    return f'''`timescale 1ns/1ps
module {top};
  localparam integer BATCH_ID={batch_id}, VECTOR_COUNT={VECTORS}, SITE_COUNT={len(selected)};
  localparam integer PROBE_BITS={PROBES}, SNAPSHOTS={SNAPSHOTS};
  localparam integer TIMEOUT_CYCLES={TIMEOUT_CYCLES}, EXPECTED_BASELINE_CYCLES={EXPECTED_BASELINE_CYCLES};
  logic clk_i=0,rst_ni=0,start_i=0,busy_o,done_o,fault_enable_i=0,fault_value_i=0,fault_raw_o;
  logic [8:0] fault_selector_i='0; logic [255:0] key_i='0,message_i='0,digest_o;
  logic [PROBE_BITS-1:0] probe_o,prev_probe,delta_probe;
  logic [255:0] key_vectors[0:VECTOR_COUNT-1],message_vectors[0:VECTOR_COUNT-1],digest_vectors[0:VECTOR_COUNT-1];
  logic [31:0] source_vector_indices[0:VECTOR_COUNT-1];
  integer selected_selectors[0:SITE_COUNT-1]; string selected_site_ids[0:SITE_COUNT-1];
  logic [63:0] snapshot_work[0:SNAPSHOTS-1]; integer snapshot_written[0:SNAPSHOTS-1];
  integer toggle_work[0:PROBE_BITS-1];
  logic [63:0] baseline_snapshot[0:VECTOR_COUNT-1][0:SNAPSHOTS-1];
  integer baseline_toggle[0:VECTOR_COUNT-1][0:PROBE_BITS-1];
  integer baseline_cycles[0:VECTOR_COUNT-1],baseline_busy_first[0:VECTOR_COUNT-1];
  integer baseline_busy_last[0:VECTOR_COUNT-1],baseline_done_cycle[0:VECTOR_COUNT-1];
  integer csv_fd,site_index,vector_rank,stuck_index,s,t,cycles_result;
  integer busy_first_result,busy_last_result,done_cycle_result;
  integer baseline_runs=0,enabled_runs=0,unknown_runs=0,baseline_failures=0;
  logic timeout_result,unknown_result,stuck_value; logic [255:0] digest_result;
  {dut} dut(.clk_i(clk_i),.rst_ni(rst_ni),.start_i(start_i),.key_i(key_i),.message_i(message_i),
    .busy_o(busy_o),.done_o(done_o),.digest_o(digest_o),.fault_enable_i(fault_enable_i),
    .fault_selector_i(fault_selector_i),.fault_value_i(fault_value_i),.fault_raw_o(fault_raw_o),.probe_o(probe_o));
  always #5 clk_i=~clk_i;
  task automatic reset_dut; begin
    start_i=0; fault_enable_i=0; fault_selector_i='0; fault_value_i=0; rst_ni=0;
    repeat(4) @(posedge clk_i); @(negedge clk_i); rst_ni=1; repeat(2) @(posedge clk_i);
  end endtask
  task automatic run_transaction(input integer vi,input logic en,input integer selector,input logic forced,
      output integer cycles,output logic timed_out,output logic [255:0] measured_digest,
      output integer busy_first,output integer busy_last,output integer done_cycle,output logic unknown_seen); begin
    reset_dut();
    for(s=0;s<SNAPSHOTS;s=s+1) begin snapshot_work[s]='0; snapshot_written[s]=0; end
    for(t=0;t<PROBE_BITS;t=t+1) toggle_work[t]=0;
    @(negedge clk_i); key_i=key_vectors[vi]; message_i=message_vectors[vi];
    fault_selector_i=selector; fault_value_i=forced; fault_enable_i=en;
    @(negedge clk_i); start_i=1; @(negedge clk_i); start_i=0; #1;
    cycles=0; busy_first=-1; busy_last=-1; done_cycle=-1; unknown_seen=0;
    snapshot_work[0]=probe_o; snapshot_written[0]=1; prev_probe=probe_o;
    if($isunknown({{probe_o,digest_o,done_o,busy_o}})) unknown_seen=1;
    while(!done_o && cycles<TIMEOUT_CYCLES) begin
      @(posedge clk_i); #1; cycles=cycles+1;
      delta_probe=probe_o^prev_probe;
      for(t=0;t<PROBE_BITS;t=t+1) begin
        if(delta_probe[t] && toggle_work[t]<65535) toggle_work[t]=toggle_work[t]+1;
      end
      prev_probe=probe_o;
      if(busy_o) begin if(busy_first<0) busy_first=cycles; busy_last=cycles; end
      if(done_o && done_cycle<0) done_cycle=cycles;
      case(cycles)
{sample_case_lines()}
        default: begin end
      endcase
      if($isunknown({{probe_o,digest_o,done_o,busy_o}})) unknown_seen=1;
    end
    timed_out=!done_o; measured_digest=digest_o;
    for(s=0;s<SNAPSHOTS-1;s=s+1) if(!snapshot_written[s]) snapshot_work[s]=probe_o;
    snapshot_work[SNAPSHOTS-1]=probe_o; snapshot_written[SNAPSHOTS-1]=1;
    if($isunknown({{probe_o,digest_o,done_o,busy_o}})) unknown_seen=1;
    fault_enable_i=0; @(posedge clk_i);
  end endtask
  task automatic write_trace_fields(input integer vi,input logic compare_baseline); begin
    for(s=0;s<SNAPSHOTS;s=s+1) begin
      if(compare_baseline) $fwrite(csv_fd,"%016h",snapshot_work[s]^baseline_snapshot[vi][s]);
      else $fwrite(csv_fd,"%016h",snapshot_work[s]);
    end
    $fwrite(csv_fd,",");
    for(t=0;t<PROBE_BITS;t=t+1) $fwrite(csv_fd,"%04x",toggle_work[t]);
    $fwrite(csv_fd,"\\n");
  end endtask
  initial begin
    $readmemh("{sv_quote(str(key_mem))}",key_vectors);
    $readmemh("{sv_quote(str(message_mem))}",message_vectors);
    $readmemh("{sv_quote(str(digest_mem))}",digest_vectors);
    $readmemh("{sv_quote(str(source_index_mem))}",source_vector_indices);
{chr(10).join(site_init)}
    csv_fd=$fopen("{sv_quote(str(csv_path))}","w"); if(csv_fd==0) $fatal(1,"CSV open failed");
    $fdisplay(csv_fd,"{','.join(CSV_FIELDS)}");
    for(vector_rank=0;vector_rank<VECTOR_COUNT;vector_rank=vector_rank+1) begin
      run_transaction(vector_rank,0,0,0,cycles_result,timeout_result,digest_result,busy_first_result,busy_last_result,done_cycle_result,unknown_result);
      baseline_cycles[vector_rank]=cycles_result; baseline_busy_first[vector_rank]=busy_first_result;
      baseline_busy_last[vector_rank]=busy_last_result; baseline_done_cycle[vector_rank]=done_cycle_result;
      for(s=0;s<SNAPSHOTS;s=s+1) baseline_snapshot[vector_rank][s]=snapshot_work[s];
      for(t=0;t<PROBE_BITS;t=t+1) baseline_toggle[vector_rank][t]=toggle_work[t];
      baseline_runs=baseline_runs+1; if(unknown_result) unknown_runs=unknown_runs+1;
      if(timeout_result||digest_result!==digest_vectors[vector_rank]||cycles_result!=EXPECTED_BASELINE_CYCLES) baseline_failures=baseline_failures+1;
      $fwrite(csv_fd,"%0d,BASELINE,BASELINE,-1,-1,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%064h,%064h,",BATCH_ID,vector_rank,source_vector_indices[vector_rank],cycles_result,timeout_result,busy_first_result,busy_last_result,done_cycle_result,unknown_result,digest_vectors[vector_rank],digest_result);
      write_trace_fields(vector_rank,0);
    end
    for(site_index=0;site_index<SITE_COUNT;site_index=site_index+1) begin
      for(stuck_index=0;stuck_index<2;stuck_index=stuck_index+1) begin
        stuck_value=(stuck_index!=0);
        for(vector_rank=0;vector_rank<VECTOR_COUNT;vector_rank=vector_rank+1) begin
          run_transaction(vector_rank,1,selected_selectors[site_index],stuck_value,cycles_result,timeout_result,digest_result,busy_first_result,busy_last_result,done_cycle_result,unknown_result);
          enabled_runs=enabled_runs+1; if(unknown_result) unknown_runs=unknown_runs+1;
          $fwrite(csv_fd,"%0d,ENABLED,%s,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%064h,%064h,",BATCH_ID,selected_site_ids[site_index],selected_selectors[site_index],stuck_index,vector_rank,source_vector_indices[vector_rank],cycles_result,timeout_result,busy_first_result,busy_last_result,done_cycle_result,unknown_result,digest_vectors[vector_rank],digest_result);
          write_trace_fields(vector_rank,1);
        end
      end
    end
    $fclose(csv_fd);
    $display("V21_PROBE_BATCH=%0d",BATCH_ID); $display("V21_PROBE_BASELINES=%0d",baseline_runs);
    $display("V21_PROBE_ENABLED=%0d",enabled_runs); $display("V21_PROBE_UNKNOWN=%0d",unknown_runs);
    if(baseline_runs==VECTOR_COUNT&&enabled_runs==SITE_COUNT*2*VECTOR_COUNT&&unknown_runs==0&&baseline_failures==0) begin
      $display("V21_PROBE_BATCH_RESULT=PASS"); $finish;
    end else $fatal(1,"V21_PROBE_BATCH_RESULT=FAIL baseline_failures=%0d",baseline_failures);
  end
endmodule
'''


def run_logged(command: list[str], log: Path, timeout: int, resource: Path | None = None) -> int:
    actual = list(command)
    if resource is not None:
        actual = ["/usr/bin/time", "-v", "-o", str(resource)] + actual
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as stream:
        try:
            return subprocess.run(actual, stdout=stream, stderr=subprocess.STDOUT, timeout=timeout, check=False).returncode
        except subprocess.TimeoutExpired:
            stream.write(f"\nTIMEOUT_SECONDS={timeout}\n")
            return 124


def parse_hex_vector(value: str, count: int, width: int, dtype: Any) -> np.ndarray:
    require(len(value) == count * width and re.fullmatch(r"[0-9a-fA-F]+", value) is not None, "trace hex encoding")
    return np.asarray([int(value[i * width:(i + 1) * width], 16) for i in range(count)], dtype=dtype)


def npy_bytes(array: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.lib.format.write_array(stream, np.ascontiguousarray(array), version=(2, 0), allow_pickle=False)
    return stream.getvalue()


def deterministic_npz(arrays: dict[str, np.ndarray]) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        for name in sorted(arrays):
            require(re.fullmatch(r"[a-z][a-z0-9_]*", name) is not None, f"invalid NPZ member: {name}")
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o600 << 16
            archive.writestr(info, npy_bytes(arrays[name]), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    return stream.getvalue()


def parse_batch_csv(path: Path, batch_id: int, sites: list[dict[str, Any]],
                    source_indices: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    local_faults = len(sites) * 2
    baseline_snapshot = np.zeros((VECTORS, SNAPSHOTS), dtype=np.uint64)
    baseline_toggle = np.zeros((VECTORS, PROBES), dtype=np.uint16)
    baseline_cycles = np.zeros(VECTORS, dtype=np.uint16)
    baseline_control = np.zeros((VECTORS, 3), dtype=np.int16)
    snapshot_xor = np.zeros((local_faults, VECTORS, SNAPSHOTS), dtype=np.uint64)
    toggle_count = np.zeros((local_faults, VECTORS, PROBES), dtype=np.uint16)
    cycles = np.zeros((local_faults, VECTORS), dtype=np.uint16)
    control = np.zeros((local_faults, VECTORS, 3), dtype=np.int16)
    timed_out = np.zeros((local_faults, VECTORS), dtype=np.uint8)
    digest_xor = np.zeros((local_faults, VECTORS, 32), dtype=np.uint8)
    seen_baseline = np.zeros(VECTORS, dtype=np.uint8)
    seen = np.zeros((local_faults, VECTORS), dtype=np.uint8)
    site_to_local = {site["site_id"]: index for index, site in enumerate(sites)}
    unknown = baseline_failures = 0
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames == CSV_FIELDS, f"CSV header Batch {batch_id:03d}")
        for row in reader:
            require(int(row["batch_id"]) == batch_id, "CSV batch ID")
            rank = int(row["vector_rank"])
            require(0 <= rank < VECTORS and int(row["source_vector_index"]) == int(source_indices[rank]), "vector mapping")
            unknown += int(row["unknown"])
            snapshots = parse_hex_vector(row["snapshot_hex"], SNAPSHOTS, 16, np.uint64)
            toggles = parse_hex_vector(row["toggle_hex"], PROBES, 4, np.uint16)
            row_cycles = int(row["cycles"])
            row_control = [int(row["busy_first_cycle"]), int(row["busy_last_cycle"]), int(row["done_cycle"])]
            if row["run_type"] == "BASELINE":
                require(seen_baseline[rank] == 0 and row["site_id"] == "BASELINE", "duplicate/invalid baseline")
                seen_baseline[rank] = 1
                baseline_snapshot[rank] = snapshots
                baseline_toggle[rank] = toggles
                baseline_cycles[rank] = row_cycles
                baseline_control[rank] = row_control
                baseline_failures += int(row["timed_out"] != "0" or row_cycles != EXPECTED_BASELINE_CYCLES or row["expected_digest"].lower() != row["actual_digest"].lower())
            elif row["run_type"] == "ENABLED":
                require(row["site_id"] in site_to_local, "unexpected pilot site")
                stuck = int(row["stuck_value"])
                require(stuck in (0, 1), "stuck value")
                fault = site_to_local[row["site_id"]] * 2 + stuck
                require(seen[fault, rank] == 0, "duplicate enabled sample")
                seen[fault, rank] = 1
                snapshot_xor[fault, rank] = snapshots
                toggle_count[fault, rank] = toggles
                cycles[fault, rank] = row_cycles
                control[fault, rank] = row_control
                timed_out[fault, rank] = int(row["timed_out"])
                expected = bytes.fromhex(row["expected_digest"])
                actual = bytes.fromhex(row["actual_digest"])
                require(len(expected) == 32 and len(actual) == 32, "digest encoding")
                digest_xor[fault, rank] = np.frombuffer(bytes(a ^ b for a, b in zip(expected, actual)), dtype=np.uint8)
            else:
                stop("unknown CSV run type")
    require(np.all(seen_baseline == 1), f"baseline coverage Batch {batch_id:03d}")
    require(np.all(seen == 1), f"enabled coverage Batch {batch_id:03d}")
    require(unknown == 0 and baseline_failures == 0, f"semantic failures Batch {batch_id:03d}")
    toggle_delta = toggle_count.astype(np.int32) - baseline_toggle[None, :, :].astype(np.int32)
    control_delta = control.astype(np.int32) - baseline_control[None, :, :].astype(np.int32)
    cycle_delta = cycles.astype(np.int32) - baseline_cycles[None, :].astype(np.int32)
    external_detected = ((timed_out != 0) | (cycle_delta != 0) | np.any(digest_xor != 0, axis=2)).astype(np.uint8)
    probe_effect = (np.any(snapshot_xor != 0, axis=2) | np.any(toggle_delta != 0, axis=2) | np.any(control_delta != 0, axis=2)).astype(np.uint8)
    fault_indices = np.asarray([site["pilot_rank"] * 2 + stuck for site in sites for stuck in (0, 1)], dtype=np.int32)
    arrays = {
        "baseline_control_timeline": baseline_control,
        "baseline_cycles": baseline_cycles,
        "baseline_probe_snapshots": baseline_snapshot,
        "baseline_probe_toggle_count": baseline_toggle,
        "control_timeline_delta": control_delta.astype(np.int16),
        "cycle_delta": cycle_delta.astype(np.int16),
        "cycles": cycles,
        "digest_xor": digest_xor,
        "external_detected": external_detected,
        "fault_instance_index": fault_indices,
        "probe_effect": probe_effect,
        "probe_snapshot_xor": snapshot_xor,
        "probe_toggle_count": toggle_count,
        "probe_toggle_delta": toggle_delta.astype(np.int16),
        "source_vector_index": source_indices.astype(np.int32),
        "timed_out": timed_out,
        "vector_rank": np.arange(VECTORS, dtype=np.int32),
    }
    summary = {
        "baseline_records": VECTORS,
        "enabled_records": local_faults * VECTORS,
        "fault_instances": local_faults,
        "unknown_records": unknown,
        "baseline_failures": baseline_failures,
        "externally_detected_faults": int(np.sum(np.any(external_detected != 0, axis=1))),
        "probe_effect_faults": int(np.sum(np.any(probe_effect != 0, axis=1))),
        "probe_only_faults": int(np.sum(np.any((probe_effect != 0) & (external_detected == 0), axis=1))),
    }
    return arrays, summary


def execute(resume: bool) -> None:
    require(not AUDIT.exists() and not MANIFEST.exists() and not DATASET.exists(), f"Stage {STAGE} is already frozen")
    _, probes = verify_inputs()
    keys, messages, source_indices, digests = load_vectors()
    pilot_rows, batches = load_pilot_plan()
    probe_bits = [int(row["bit_id"]) for row in probes]
    WORK.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    PER_BATCH.mkdir(parents=True, exist_ok=True)
    BUILD.mkdir(parents=True, exist_ok=True)
    key_mem = WORK / "selected_keys_12b2h.mem"
    message_mem = WORK / "selected_messages_12b2h.mem"
    digest_mem = WORK / "selected_expected_digests_12b2h.mem"
    source_index_mem = WORK / "selected_source_vector_indices_12b2h.mem"
    memory_payloads = {
        key_mem: "".join(row.tobytes().hex() + "\n" for row in keys).encode(),
        message_mem: "".join(row.tobytes().hex() + "\n" for row in messages).encode(),
        digest_mem: "".join(row.tobytes().hex() + "\n" for row in digests).encode(),
        source_index_mem: "".join(f"{int(value):08x}\n" for value in source_indices).encode(),
    }
    for path, payload in memory_payloads.items():
        if path.exists():
            require(path.read_bytes() == payload, f"resume memory mismatch: {path.name}")
        else:
            path.write_bytes(payload)

    if resume:
        state = load_json(CHECKPOINT)
        require(state.get("stage") == STAGE and state.get("status") == "RUNNING", "resume checkpoint state")
        require(state.get("authorization_sha256") == sha256(AUTHORIZATION), "resume authorization")
        completed = set(int(value) for value in state.get("completed_batches", []))
    else:
        require(not CHECKPOINT.exists(), "checkpoint exists; use --resume after verifying the interruption")
        completed: set[int] = set()
        state = {
            "checkpoint_version": "CIRCUITSAGE-HMAC-V2.1-PROBE-CAPTURE-CHECKPOINT-v1",
            "stage": STAGE, "status": "RUNNING", "created_at": now(), "updated_at": now(),
            "authorization_sha256": sha256(AUTHORIZATION),
            "selected_vectors_sha256": sha256(SELECTED_VECTORS),
            "probe_bank_sha256": sha256(PROBE_BANK_JSON),
            "pilot_sites_sha256": sha256(PILOT_SITES),
            "planned_batches": list(range(BATCHES)), "completed_batches": [], "batches": {},
        }
        atomic_json(CHECKPOINT, state)

    yosys = shutil.which("yosys")
    verilator = shutil.which("verilator")
    require(yosys is not None and verilator is not None, "toolchain disappeared")
    print(f"Canonical batches      : {BATCHES}")
    print(f"Pilot sites / faults   : {SITES} / {FAULTS}")
    print(f"Vectors / probes       : {VECTORS} / {PROBES}")
    print(f"Enabled transactions   : {ENABLED_TRANSACTIONS}")
    print("Execution              : SEQUENTIAL; build jobs=1")
    print(f"Checkpoint             : {CHECKPOINT}")

    consistency = load_json(CONSISTENCY_2F)["batches"]
    for batch_id in range(BATCHES):
        key = f"{batch_id:03d}"
        selected = batches[batch_id]
        if batch_id in completed:
            item = state["batches"].get(key, {})
            csv_path = Path(item.get("csv", ""))
            batch_npz = Path(item.get("batch_npz", ""))
            require(csv_path.is_file() and sha256(csv_path) == item.get("csv_sha256"), f"resume CSV Batch {key}")
            require(batch_npz.is_file() and sha256(batch_npz) == item.get("batch_npz_sha256"), f"resume NPZ Batch {key}")
            parse_batch_csv(csv_path, batch_id, selected, source_indices)
            print(f"Batch {key}: RESUME SKIP — VERIFIED", flush=True)
            continue
        batch_raw = RAW / f"batch_{key}"
        batch_build = BUILD / f"batch_{key}"
        batch_raw.mkdir(parents=True, exist_ok=True)
        batch_build.mkdir(parents=True, exist_ok=True)
        canonical_json_path = ROOT / f"build/hmac_fault_batches_canonical_11c4g/batch_{key}/opentitan_hmac_sha256_msg32_faultbatch{key}.json"
        mapping_path = ROOT / f"results/hmac_fault_campaign_11c4/canonical_batches/batch_{key}/hmac_fault_batch_{key}_mapping.json"
        verify_record(consistency[batch_id]["canonical_json"], canonical_json_path, f"Batch {key} canonical JSON")
        verify_record(consistency[batch_id]["mapping"], mapping_path, f"Batch {key} mapping")
        module = f"opentitan_hmac_sha256_msg32_faultbatch{key}"
        derived_json = batch_build / f"{module}_probe.json"
        derived_verilog = batch_build / f"{module}_probe.v"
        observer_payload = observer_json(canonical_json_path, module, probe_bits)
        if derived_json.exists():
            require(derived_json.read_bytes() == observer_payload, f"derived observer mismatch Batch {key}")
        else:
            derived_json.write_bytes(observer_payload)
        yosys_log = batch_raw / "yosys_observer.log"
        print(f"Batch {key}: DERIVE OBSERVER sites={len(selected)}", flush=True)
        yosys_command = [yosys, "-p", f"read_json {derived_json}; hierarchy -check -top {module}; write_verilog -noattr {derived_verilog}"]
        require(run_logged(yosys_command, yosys_log, 900) == 0, f"Batch {key} observer generation")
        require(derived_verilog.is_file() and derived_verilog.stat().st_size > 0, f"Batch {key} observer Verilog")
        csv_path = batch_raw / f"circuitsage_hmac_v2_1_probe_capture_batch_{key}.csv"
        tb = batch_build / f"tb_v21_probe_capture_batch_{key}.sv"
        tb_payload = generate_testbench(batch_id, selected, csv_path.resolve(), key_mem.resolve(), message_mem.resolve(), digest_mem.resolve(), source_index_mem.resolve()).encode()
        if tb.exists():
            require(tb.read_bytes() == tb_payload, f"testbench mismatch Batch {key}")
        else:
            tb.write_bytes(tb_payload)
        top = f"tb_v21_probe_capture_batch_{key}"
        obj_dir = batch_build / "obj_dir"
        build_log = batch_raw / "verilator_build.log"
        print(f"Batch {key}: BUILD", flush=True)
        command = [verilator, "--binary", "--timing", "--assert", "-Wall", "-Wno-fatal", "-j", "1", "--Mdir", str(obj_dir), "--top-module", top, str(derived_verilog), str(tb)]
        require(run_logged(command, build_log, 1800) == 0, f"Batch {key} build failed")
        binary = obj_dir / f"V{top}"
        require(binary.is_file() and os.access(binary, os.X_OK), f"Batch {key} binary missing")
        simulation_log = batch_raw / "simulation.log"
        resource_log = batch_raw / "resources.log"
        print(f"Batch {key}: SIMULATE", flush=True)
        require(run_logged([str(binary)], simulation_log, 14400, resource_log) == 0, f"Batch {key} simulation failed")
        require("V21_PROBE_BATCH_RESULT=PASS" in simulation_log.read_text(errors="replace"), f"Batch {key} PASS token")
        arrays, summary = parse_batch_csv(csv_path, batch_id, selected, source_indices)
        batch_npz = PER_BATCH / f"circuitsage_hmac_v2_1_probe_capture_batch_{key}.npz"
        frozen_write(batch_npz, deterministic_npz(arrays))
        item = {
            "status": "PASS", "batch_id": batch_id, "completed_at": now(), "sites": len(selected),
            "canonical_json": record(canonical_json_path), "mapping": record(mapping_path),
            "derived_observer_json": record(derived_json), "derived_observer_verilog": record(derived_verilog),
            "testbench": record(tb), "binary": record(binary), "yosys_log": record(yosys_log),
            "build_log": record(build_log), "simulation_log": record(simulation_log), "resource_log": record(resource_log),
            "csv": str(csv_path), "csv_sha256": sha256(csv_path), "csv_bytes": csv_path.stat().st_size,
            "batch_npz": str(batch_npz), "batch_npz_sha256": sha256(batch_npz), "batch_npz_bytes": batch_npz.stat().st_size,
            "summary": summary,
        }
        state["batches"][key] = item
        completed.add(batch_id)
        state["completed_batches"] = sorted(completed)
        state["updated_at"] = now()
        atomic_json(CHECKPOINT, state)
        print(f"Batch {key}: PASS enabled={summary['enabled_records']} probe_effect_faults={summary['probe_effect_faults']}", flush=True)

    require(completed == set(range(BATCHES)), "incomplete capture batches")
    freeze_dataset(state, pilot_rows, batches, source_indices)


def freeze_dataset(state: dict[str, Any], pilot_rows: list[dict[str, Any]],
                   batches: dict[int, list[dict[str, Any]]], source_indices: np.ndarray) -> None:
    print("CONSOLIDATING AND VERIFYING PROBE DATASET", flush=True)
    snapshot_xor = np.zeros((FAULTS, VECTORS, SNAPSHOTS), dtype=np.uint64)
    toggle_count = np.zeros((FAULTS, VECTORS, PROBES), dtype=np.uint16)
    toggle_delta = np.zeros((FAULTS, VECTORS, PROBES), dtype=np.int16)
    control_delta = np.zeros((FAULTS, VECTORS, 3), dtype=np.int16)
    cycle_delta = np.zeros((FAULTS, VECTORS), dtype=np.int16)
    digest_xor = np.zeros((FAULTS, VECTORS, 32), dtype=np.uint8)
    timed_out = np.zeros((FAULTS, VECTORS), dtype=np.uint8)
    external_detected = np.zeros((FAULTS, VECTORS), dtype=np.uint8)
    probe_effect = np.zeros((FAULTS, VECTORS), dtype=np.uint8)
    seen = np.zeros((FAULTS, VECTORS), dtype=np.uint8)
    baseline_snapshots = np.zeros((BATCHES, VECTORS, SNAPSHOTS), dtype=np.uint64)
    baseline_toggles = np.zeros((BATCHES, VECTORS, PROBES), dtype=np.uint16)
    baseline_cycles = np.zeros((BATCHES, VECTORS), dtype=np.uint16)
    baseline_control = np.zeros((BATCHES, VECTORS, 3), dtype=np.int16)
    raw_records: dict[str, Any] = {}
    batch_records: dict[str, Any] = {}
    baseline_records = enabled_records = 0
    for batch_id in range(BATCHES):
        key = f"{batch_id:03d}"
        item = state["batches"][key]
        csv_path = Path(item["csv"])
        batch_npz = Path(item["batch_npz"])
        require(sha256(csv_path) == item["csv_sha256"], f"checkpoint/CSV Batch {key}")
        require(sha256(batch_npz) == item["batch_npz_sha256"], f"checkpoint/NPZ Batch {key}")
        parsed, summary = parse_batch_csv(csv_path, batch_id, batches[batch_id], source_indices)
        require(deterministic_npz(parsed) == batch_npz.read_bytes(), f"deterministic batch replay Batch {key}")
        local_faults = parsed["fault_instance_index"]
        require(np.all(seen[local_faults] == 0), f"duplicate cross-batch fault Batch {key}")
        seen[local_faults] = 1
        snapshot_xor[local_faults] = parsed["probe_snapshot_xor"]
        toggle_count[local_faults] = parsed["probe_toggle_count"]
        toggle_delta[local_faults] = parsed["probe_toggle_delta"]
        control_delta[local_faults] = parsed["control_timeline_delta"]
        cycle_delta[local_faults] = parsed["cycle_delta"]
        digest_xor[local_faults] = parsed["digest_xor"]
        timed_out[local_faults] = parsed["timed_out"]
        external_detected[local_faults] = parsed["external_detected"]
        probe_effect[local_faults] = parsed["probe_effect"]
        baseline_snapshots[batch_id] = parsed["baseline_probe_snapshots"]
        baseline_toggles[batch_id] = parsed["baseline_probe_toggle_count"]
        baseline_cycles[batch_id] = parsed["baseline_cycles"]
        baseline_control[batch_id] = parsed["baseline_control_timeline"]
        baseline_records += summary["baseline_records"]
        enabled_records += summary["enabled_records"]
        raw_records[key] = record(csv_path)
        batch_records[key] = record(batch_npz)
    require(np.all(seen == 1), "missing consolidated fault/vector samples")
    require(baseline_records == BATCHES * VECTORS and enabled_records == ENABLED_TRANSACTIONS, "record totals")
    require(np.all(baseline_cycles == EXPECTED_BASELINE_CYCLES), "baseline latency replay")
    # Golden observer behavior must be identical across all canonical batches.
    require(np.all(baseline_snapshots == baseline_snapshots[0:1]), "cross-batch baseline snapshot replay")
    require(np.all(baseline_toggles == baseline_toggles[0:1]), "cross-batch baseline toggle replay")
    require(np.all(baseline_control == baseline_control[0:1]), "cross-batch baseline control replay")

    site_index = np.repeat(np.asarray([int(row["site_index"]) for row in pilot_rows], dtype=np.int32), 2)
    stuck_value = np.tile(np.asarray([0, 1], dtype=np.uint8), SITES)
    # Feature payload deliberately excludes fault/site identity and stuck value.
    feature_arrays = {
        "baseline_control_timeline": baseline_control[0],
        "baseline_cycles": baseline_cycles[0],
        "baseline_probe_snapshots": baseline_snapshots[0],
        "baseline_probe_toggle_count": baseline_toggles[0],
        "control_timeline_delta": control_delta,
        "cycle_delta": cycle_delta,
        "digest_xor": digest_xor,
        "external_detected": external_detected,
        "probe_effect": probe_effect,
        "probe_snapshot_xor": snapshot_xor,
        "probe_toggle_count": toggle_count,
        "probe_toggle_delta": toggle_delta,
        "source_vector_index": source_indices.astype(np.int32),
        "timed_out": timed_out,
        "vector_rank": np.arange(VECTORS, dtype=np.int32),
    }
    target_arrays = {
        "fault_instance_index": np.arange(FAULTS, dtype=np.int32),
        "pilot_rank": np.repeat(np.arange(SITES, dtype=np.int32), 2),
        "site_index": site_index,
        "stuck_value": stuck_value,
    }
    feature_payload = deterministic_npz(feature_arrays)
    target_payload = deterministic_npz(target_arrays)
    frozen_write(DATASET, feature_payload)
    frozen_write(TARGETS, target_payload)
    externally_observable = np.any(external_detected != 0, axis=1)
    probe_observable = np.any(probe_effect != 0, axis=1)
    combined_observable = externally_observable | probe_observable
    probe_rescued = (~externally_observable) & probe_observable
    metrics = {
        "metrics_version": "CIRCUITSAGE-HMAC-V2.1-PROBE-OBSERVABILITY-12B2H-v1",
        "stage": STAGE, "status": "FROZEN",
        "fault_instances": FAULTS, "selected_vectors": VECTORS, "probe_bits": PROBES,
        "external_observable_faults": int(np.sum(externally_observable)),
        "probe_observable_faults": int(np.sum(probe_observable)),
        "combined_observable_faults": int(np.sum(combined_observable)),
        "probe_rescued_external_invisible_faults": int(np.sum(probe_rescued)),
        "external_detection_ceiling": float(np.mean(externally_observable)),
        "combined_measurement_detection_ceiling": float(np.mean(combined_observable)),
        "frozen_target": 0.70,
        "target_feasible_on_pilot_with_probes": bool(np.mean(combined_observable) >= 0.70),
        "interpretation": "REPAIR_TRAIN PILOT MEASUREMENT FEASIBILITY; NOT INDEPENDENT MODEL ACCURACY",
    }
    frozen_write(METRICS, canonical_json(metrics))
    schema = {
        "schema_version": "CIRCUITSAGE-HMAC-V2.1-PROBE-RESPONSE-DATASET-12B2H-v1",
        "stage": STAGE, "status": "FROZEN", "format": "DETERMINISTIC NPZ",
        "features": {name: {"dtype": str(value.dtype), "shape": list(value.shape)} for name, value in sorted(feature_arrays.items())},
        "targets": {name: {"dtype": str(value.dtype), "shape": list(value.shape)} for name, value in sorted(target_arrays.items())},
        "sample_key": ["fault_instance_index", "vector_rank"],
        "sample_cycles_after_start": SAMPLE_CYCLES,
        "final_snapshot": "DONE_OR_TIMEOUT",
        "early_completion_padding": "UNREACHED FIXED-CYCLE SLOTS COPY FINAL PROBE STATE",
        "probe_snapshot_representation": "EXACT 64-BIT ENABLED XOR MATCHED GOLDEN",
        "probe_toggle_representation": "UINT16 SATURATING COUNT PLUS SIGNED DELTA FROM MATCHED GOLDEN",
        "feature_identity_exclusion": "site_index, fault_instance_index, pilot_rank, stuck_value and selector are absent from feature NPZ",
        "fault_identity_location": "SEPARATE TARGET NPZ FOR SCORING ONLY",
        "source_partition": "REPAIR_TRAIN PILOT ONLY",
        "model_training_or_inference": False,
        "repair_site_test_rows": 0, "original_dev_site_test_rows": 0,
        "validation_rows": 0, "holdout_rows": 0,
    }
    frozen_write(SCHEMA, canonical_json(schema))
    state["status"] = "PASS"; state["finished_at"] = now(); state["updated_at"] = now()
    atomic_json(CHECKPOINT, state)
    manifest = {
        "manifest_version": "CIRCUITSAGE-HMAC-V2.1-PROBE-CAPTURE-DATASET-MANIFEST-v1",
        "stage": STAGE, "status": "PASS", "authorization": record(AUTHORIZATION),
        "checkpoint": record(CHECKPOINT), "raw_batch_csvs": raw_records,
        "per_batch_npz": batch_records,
        "outputs": {rel(path): record(path) for path in (DATASET, TARGETS, METRICS, SCHEMA)},
        "canonical_batches": BATCHES, "pilot_sites": SITES, "fault_instances": FAULTS,
        "selected_vectors": VECTORS, "probe_bits": PROBES, "snapshots": SNAPSHOTS,
        "baseline_records": baseline_records, "enabled_records": enabled_records,
        "missing_samples": int(np.sum(seen == 0)), "duplicate_samples": 0,
        "unknown_records": 0, "baseline_failures": 0,
        "model_objects_deserialized": 0, "training_calls": 0, "inference_calls": 0,
        "repair_calibration_access": 0, "repair_site_test_access": 0,
        "original_dev_site_test_access": 0, "validation_access": 0, "holdout_access": 0,
        "frozen_rtl_modified": False, "canonical_netlists_modified": False,
    }
    frozen_write(MANIFEST, canonical_json(manifest))
    audit = {
        "audit_version": "CIRCUITSAGE-HMAC-V2.1-PROBE-CAPTURE-DATASET-INTEGRITY-FREEZE-v1",
        "stage": STAGE, "status": "PASS",
        "capture_execution": "COMPLETED / FROZEN", "dataset_status": "FROZEN",
        "batches_verified": f"{BATCHES}/{BATCHES}", "pilot_sites": SITES,
        "fault_instances": FAULTS, "selected_vectors": VECTORS,
        "probe_bits_snapshots": [PROBES, SNAPSHOTS],
        "baseline_enabled_records": [baseline_records, enabled_records],
        "external_observable_faults": int(np.sum(externally_observable)),
        "probe_observable_faults": int(np.sum(probe_observable)),
        "combined_observable_faults": int(np.sum(combined_observable)),
        "probe_rescued_external_invisible_faults": int(np.sum(probe_rescued)),
        "combined_measurement_detection_ceiling": float(np.mean(combined_observable)),
        "frozen_detection_target": 0.70,
        "target_feasibility": "PASS" if np.mean(combined_observable) >= 0.70 else "NOT_MET",
        "missing_duplicate_samples": [0, 0], "unknown_records": 0, "baseline_failures": 0,
        "deterministic_dataset_replay": "PASS / EXACT",
        "identity_fields_in_feature_npz": 0,
        "fault_selector_value_raw_as_features": "PROHIBITED / ABSENT",
        "model_training_inference": "0 / 0",
        "repair_calibration": "LOCKED / NOT ACCESSED",
        "repair_site_test": "LOCKED / NOT ACCESSED",
        "original_dev_site_test": "CONSUMED / NOT REOPENED",
        "validation_access": 0, "holdout_access": 0,
        "frozen_rtl_modified": False, "canonical_netlists_modified": False,
        "features": record(DATASET), "targets": record(TARGETS), "metrics": record(METRICS),
        "manifest": record(MANIFEST),
        "next_gate": "STAGE 12B-2I — PROBE-MEASUREMENT FEASIBILITY DISPOSITION AND REPAIR-TRAINING CONTRACT FREEZE",
    }
    frozen_write(AUDIT, canonical_json(audit))
    require(feature_payload == DATASET.read_bytes() and target_payload == TARGETS.read_bytes(), "exact NPZ replay")
    for path in (METRICS, SCHEMA, MANIFEST, AUDIT):
        require(canonical_json(load_json(path)) == path.read_bytes(), f"canonical JSON replay: {path.name}")
    for path, expected in PINNED.items():
        require(sha256(path) == expected, f"frozen input changed: {rel(path)}")

    print("\nSTAGE 12B-2H — BOUNDED PROBE-CAPTURE EXECUTION AND DATASET INTEGRITY FREEZE")
    print(f"{'Status':<60}: PASS")
    print(f"{'Capture execution / dataset':<60}: COMPLETED / FROZEN")
    print(f"{'Batches verified':<60}: {BATCHES}/{BATCHES}")
    print(f"{'Pilot sites / fault instances':<60}: {SITES} / {FAULTS}")
    print(f"{'Selected vectors / probes / snapshots':<60}: {VECTORS} / {PROBES} / {SNAPSHOTS}")
    print(f"{'Baseline / enabled records':<60}: {baseline_records} / {enabled_records}")
    print(f"{'External observable faults':<60}: {int(np.sum(externally_observable))}/{FAULTS}")
    print(f"{'Probe observable faults':<60}: {int(np.sum(probe_observable))}/{FAULTS}")
    print(f"{'Combined observable faults':<60}: {int(np.sum(combined_observable))}/{FAULTS}")
    print(f"{'Probe-rescued externally invisible faults':<60}: {int(np.sum(probe_rescued))}")
    print(f"{'Combined measurement detection ceiling':<60}: {float(np.mean(combined_observable)):.8f}")
    print(f"{'Frozen 0.70 target feasibility':<60}: {'PASS' if np.mean(combined_observable) >= 0.70 else 'NOT_MET'}")
    print(f"{'Missing / duplicate / unknown / baseline failures':<60}: 0 / 0 / 0 / 0")
    print(f"{'Identity fields in behavior feature NPZ':<60}: 0")
    print(f"{'Model training / inference':<60}: 0 / 0")
    print(f"{'REPAIR_SITE_TEST / DEV_SITE_TEST / VALIDATION / HOLDOUT':<60}: 0 / 0 / 0 / 0")
    print(f"{'Frozen RTL / canonical netlists modified':<60}: NO / NO")
    print(f"{'Features':<60}: {DATASET}")
    print(f"{'Features SHA':<60}: {sha256(DATASET)}")
    print(f"{'Targets':<60}: {TARGETS}")
    print(f"{'Targets SHA':<60}: {sha256(TARGETS)}")
    print(f"{'Metrics':<60}: {METRICS}")
    print(f"{'Metrics SHA':<60}: {sha256(METRICS)}")
    print(f"{'Manifest':<60}: {MANIFEST}")
    print(f"{'Manifest SHA':<60}: {sha256(MANIFEST)}")
    print(f"{'Audit':<60}: {AUDIT}")
    print(f"{'Audit SHA':<60}: {sha256(AUDIT)}")
    print(f"{'Next gate':<60}: STAGE 12B-2I — PROBE-MEASUREMENT FEASIBILITY DISPOSITION AND REPAIR-TRAINING CONTRACT FREEZE")


def status() -> None:
    print("STAGE 12B-2H — PROBE-CAPTURE STATUS")
    if AUDIT.is_file():
        audit = load_json(AUDIT)
        print(f"Status                    : {audit.get('status')} / FROZEN")
        print(f"Batches verified          : {audit.get('batches_verified')}")
        print(f"Combined observable faults: {audit.get('combined_observable_faults')}/{FAULTS}")
        print(f"Detection ceiling         : {audit.get('combined_measurement_detection_ceiling')}")
        print(f"Target feasibility        : {audit.get('target_feasibility')}")
        print(f"Audit                     : {AUDIT}")
        print(f"Audit SHA                 : {sha256(AUDIT)}")
        return
    if not CHECKPOINT.is_file():
        print("Status                    : NOT STARTED")
        return
    state = load_json(CHECKPOINT)
    completed = state.get("completed_batches", [])
    planned = state.get("planned_batches", [])
    print(f"Status                    : {state.get('status')}")
    print(f"Completed batches         : {len(completed)}/{len(planned)}")
    if completed:
        print(f"Latest completed batch    : {max(completed):03d}")
    print(f"Checkpoint                : {CHECKPOINT}")
    print(f"Checkpoint SHA            : {sha256(CHECKPOINT)}")


def self_test() -> None:
    require(len(SAMPLE_CYCLES) + 1 == SNAPSHOTS, "snapshot schedule")
    require(SAMPLE_CYCLES == sorted(set(SAMPLE_CYCLES)), "sample-cycle ordering")
    require(ENABLED_TRANSACTIONS == 196608, "enabled transaction count")
    values = np.asarray([0, 1, 0xFFFFFFFFFFFFFFFF], dtype=np.uint64)
    encoded = "".join(f"{int(value):016x}" for value in values)
    require(np.array_equal(parse_hex_vector(encoded, 3, 16, np.uint64), values), "trace parser")
    arrays = {"a": np.arange(8, dtype=np.uint8), "b": np.arange(3, dtype=np.int32)}
    require(deterministic_npz(arrays) == deterministic_npz(arrays), "deterministic NPZ")
    print("Stage 12B-2H self-test: PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--status", action="store_true")
    mode.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    elif args.status:
        status()
    else:
        require(ROOT.name == "vlsi_fault_detection_v2", f"run from project root, not {ROOT}")
        execute(resume=args.resume)


if __name__ == "__main__":
    main()
