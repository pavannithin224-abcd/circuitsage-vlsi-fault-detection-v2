#!/usr/bin/env python3
"""Stage 12B-2C: adaptive-vector pilot screening and dataset integrity freeze.

Executes the Stage 12B-2B-authorized 512-vector campaign on 1,024 REPAIR_TRAIN
pilot sites (SA0 and SA1), using the frozen 45 canonical fault-batch netlists.
The run is sequential and checkpoint/resume capable. On complete success it
consolidates the response data, verifies exact sample coverage, and freezes the
pilot dataset and manifest. No model is loaded, trained, selected, or evaluated.
"""

from __future__ import annotations

import argparse
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
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import numpy as np
except ImportError as error:
    raise SystemExit("STOP: activate the project .venv (NumPy required)") from error


STAGE = "12B-2C"
ROOT = Path(__file__).resolve().parent
RESULT = ROOT / "results/circuitsage_hmac_v2_12b2"
POOL_DIR = RESULT / "adaptive_vector_pool_12b2b"
WORK = RESULT / "pilot_screening_12b2c"
RAW = WORK / "raw_batches"
BUILD = ROOT / "build/circuitsage_hmac_v2_12b2/pilot_screening_12b2c"
CHECKPOINT = WORK / "circuitsage_hmac_v2_1_pilot_checkpoint_12b2c.json"
DATASET = WORK / "circuitsage_hmac_v2_1_pilot_response_dataset_12b2c.npz"
METRICS_CSV = WORK / "circuitsage_hmac_v2_1_candidate_vector_pilot_metrics_12b2c.csv"
SCHEMA = WORK / "circuitsage_hmac_v2_1_pilot_response_schema_12b2c.json"
MANIFEST = RESULT / "circuitsage_hmac_v2_1_pilot_dataset_manifest_12b2c.json"
AUDIT = RESULT / "circuitsage_hmac_v2_1_pilot_screening_dataset_integrity_freeze_12b2c.json"

SOURCE_2B = ROOT / "stage_12b2b_adaptive_vector_pilot_authorization.py"
VECTOR_NPZ = POOL_DIR / "circuitsage_hmac_v2_1_candidate_vectors_12b2b.npz"
VECTOR_SCHEMA = POOL_DIR / "circuitsage_hmac_v2_1_candidate_vector_schema_12b2b.json"
PILOT_SITES = POOL_DIR / "circuitsage_hmac_v2_1_pilot_screening_sites_12b2b.csv"
EXECUTION_CONTRACT = ROOT / "config/v2_1_improvement/circuitsage_hmac_v2_1_pilot_screening_execution_contract_12b2b.json"
AUTHORIZATION = ROOT / "config/v2_1_improvement/circuitsage_hmac_v2_1_pilot_screening_authorization_12b2b.json"
MANIFEST_2B = RESULT / "circuitsage_hmac_v2_1_pilot_authorization_manifest_12b2b.json"
AUDIT_2B = RESULT / "circuitsage_hmac_v2_1_pilot_authorization_freeze_12b2b.json"

GENERATION_INDEX = ROOT / "results/hmac_fault_campaign_11c4/hmac_canonical_batch_generation_index_11c4g_a1.json"
SEMANTIC_FREEZE = ROOT / "results/hmac_fault_campaign_11c4/hmac_all_batch_semantic_manifest_freeze_11c4g_a2.json"

PINNED = {
    SOURCE_2B: "912878bd37c9ed7e0ca6cb88f31afe5bff5b4715f62579f152842011fe56c0b3",
    VECTOR_NPZ: "47dd8944b9519465df19f098c2fb2a6a8823a8736452992cf52c73efcc435d78",
    VECTOR_SCHEMA: "893126561b49f4e0dc9e6edb4b083c37c72f7fd2020dcbe8148ea5edb0b2ffd3",
    PILOT_SITES: "87eceaf80a800cf60b8a7e3cb20e0ec3f934158dc35ae3d10f73e8cf9d25313c",
    EXECUTION_CONTRACT: "fc518fba8cdb6748a5c6baba291dee71432b36fe03fc3be0af80a70e74093ac0",
    AUTHORIZATION: "7b40f4df13a7d2c07168e7a43b7017dad1e33c1c91cc25eed8bd4fde94556f26",
    MANIFEST_2B: "c3fdc621bec4ffe51e98f4c63a190f33f7f0db3cdc41dfa49c6ee55947169bb0",
    AUDIT_2B: "95f49d71efcabc0d56673dd389ee50c9e474cee1697b5e2cb039fc2da362bc80",
    GENERATION_INDEX: "7ab1f5bac5fa9d635dfc04a6bd52a826d4dbd6c622bcc9c03186921c6371cd6a",
    SEMANTIC_FREEZE: "a586b8ce233f10e10d3d7db052fe249ae8a570165c0ee87f8eb4f507ca1ca7f0",
}

VECTOR_COUNT = 512
PILOT_SITE_COUNT = 1024
FAULT_COUNT = 2048
ENABLED_RECORDS = VECTOR_COUNT * FAULT_COUNT
TIMEOUT_CYCLES = 2000
EXPECTED_BASELINE_CYCLES = 343
CSV_FIELDS = [
    "batch_id", "run_type", "site_id", "selector", "stuck_value",
    "vector_index", "cycles", "baseline_cycles", "timed_out", "activity",
    "detected", "unknown", "expected_digest", "actual_digest",
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


def verify_inputs() -> None:
    print("STAGE 12B-2C — ADAPTIVE VECTOR PILOT-SCREENING EXECUTION")
    print("FROZEN INPUT VERIFICATION")
    for path, expected in PINNED.items():
        require(path.is_file(), f"missing frozen input: {rel(path)}")
        require(sha256(path) == expected, f"frozen input SHA: {rel(path)}")
        print(f"  {path.name:<91}: OK", flush=True)
    authorization = load_json(AUTHORIZATION)
    audit = load_json(AUDIT_2B)
    contract = load_json(EXECUTION_CONTRACT)
    require(authorization.get("status") == "FROZEN", "authorization status")
    require(authorization.get("pilot_screening") == "AUTHORIZED / NOT STARTED", "pilot authorization")
    require(authorization.get("full_repair_campaign") == "NOT AUTHORIZED", "full-campaign boundary")
    require(authorization.get("model_training") == "NOT AUTHORIZED", "training boundary")
    require(authorization.get("repair_site_test") == "LOCKED / NOT AUTHORIZED", "repair test boundary")
    require(authorization.get("validation") == "PROHIBITED" and authorization.get("holdout") == "PROHIBITED", "protected partitions")
    require(audit.get("status") == "PASS", "12B-2B audit status")
    require(audit.get("pilot_screening") == "AUTHORIZED / NOT STARTED", "12B-2B execution state")
    require(contract.get("source_partition") == "REPAIR_TRAIN ONLY", "pilot source")
    require(contract.get("pilot_sites") == PILOT_SITE_COUNT, "pilot-site count")
    require(contract.get("pilot_fault_instances") == FAULT_COUNT, "pilot-fault count")
    require(contract.get("checkpoint_resume") == "REQUIRED", "checkpoint contract")
    require(shutil.which("verilator") is not None, "verilator is not in PATH")
    require(shutil.disk_usage(ROOT).free >= 5 * 1024**3, "less than 5 GiB free disk")


def load_vectors() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(VECTOR_NPZ, allow_pickle=False) as archive:
        require(set(archive.files) == {"vector_index", "key_u8", "message_u8"}, "candidate-vector NPZ members")
        indices = np.asarray(archive["vector_index"], dtype=np.int32)
        keys = np.asarray(archive["key_u8"], dtype=np.uint8)
        messages = np.asarray(archive["message_u8"], dtype=np.uint8)
    require(np.array_equal(indices, np.arange(VECTOR_COUNT, dtype=np.int32)), "vector ordering")
    require(keys.shape == (VECTOR_COUNT, 32) and messages.shape == (VECTOR_COUNT, 32), "vector shapes")
    digests = np.vstack([
        np.frombuffer(hmac.new(keys[i].tobytes(), messages[i].tobytes(), hashlib.sha256).digest(), dtype=np.uint8)
        for i in range(VECTOR_COUNT)
    ])
    return indices, keys, messages, digests


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
    require(len(rows) == PILOT_SITE_COUNT, "pilot-site row count")
    require([int(row["pilot_rank"]) for row in rows] == list(range(PILOT_SITE_COUNT)), "pilot ranks")
    wanted = {row["site_id"]: row for row in rows}
    require(len(wanted) == PILOT_SITE_COUNT, "duplicate pilot site")
    batches: dict[int, list[dict[str, Any]]] = {}
    found: set[str] = set()
    for batch_id in range(45):
        key = f"{batch_id:03d}"
        mapping_path = ROOT / f"results/hmac_fault_campaign_11c4/canonical_batches/batch_{key}/hmac_fault_batch_{key}_mapping.json"
        require(mapping_path.is_file(), f"missing canonical mapping Batch {key}")
        sites = find_sites(load_json(mapping_path))
        expected_sites = 311 if batch_id == 44 else 512
        require(sites is not None and len(sites) == expected_sites, f"mapping site count Batch {key}")
        require([int(site["selector_code"]) for site in sites] == list(range(expected_sites)), f"selector order Batch {key}")
        selected: list[dict[str, Any]] = []
        for site in sites:
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
        if selected:
            batches[batch_id] = sorted(selected, key=lambda row: row["pilot_rank"])
    require(found == set(wanted), f"unmapped pilot sites: {len(set(wanted) - found)}")
    require(sum(map(len, batches.values())) == PILOT_SITE_COUNT, "mapped pilot total")
    return sorted((row for values in batches.values() for row in values), key=lambda row: row["pilot_rank"]), batches


def memory_file(path: Path, rows: np.ndarray) -> None:
    text = "".join(row.tobytes().hex() + "\n" for row in rows)
    path.write_text(text, encoding="ascii")


def sv_quote(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def generate_testbench(batch_id: int, selected: list[dict[str, Any]], csv_path: Path,
                       key_mem: Path, message_mem: Path, digest_mem: Path) -> str:
    key = f"{batch_id:03d}"
    top = f"tb_v21_pilot_batch_{key}"
    dut = f"opentitan_hmac_sha256_msg32_faultbatch{key}"
    site_init = []
    for index, site in enumerate(selected):
        site_init.extend([
            f"    selected_selectors[{index}] = 9'd{site['selector']};",
            f"    selected_site_ids[{index}] = \"{site['site_id']}\";",
        ])
    return f'''`timescale 1ns/1ps
module {top};
  localparam integer BATCH_ID={batch_id}, VECTOR_COUNT={VECTOR_COUNT}, SITE_COUNT={len(selected)};
  localparam integer TIMEOUT_CYCLES={TIMEOUT_CYCLES}, EXPECTED_BASELINE_CYCLES={EXPECTED_BASELINE_CYCLES};
  logic clk_i=0,rst_ni=0,start_i=0,busy_o,done_o,fault_enable_i=0,fault_value_i=0,fault_raw_o;
  logic [8:0] fault_selector_i='0; logic [255:0] key_i='0,message_i='0,digest_o;
  logic monitor_clear=1,activity_latched=0;
  logic [255:0] key_vectors[0:VECTOR_COUNT-1],message_vectors[0:VECTOR_COUNT-1],digest_vectors[0:VECTOR_COUNT-1];
  integer selected_selectors[0:SITE_COUNT-1]; string selected_site_ids[0:SITE_COUNT-1];
  integer baseline_cycles[0:VECTOR_COUNT-1],csv_fd;
  integer baseline_runs=0,enabled_runs=0,unknown_runs=0,detected_without_activity=0;
  integer baseline_digest_mismatches=0,baseline_latency_mismatches=0,baseline_timeouts=0;
  integer site_index,vector_index,stuck_index,cycles_result;
  logic timeout_result,activity_result,unknown_result,detected_result,stuck_value;
  logic [255:0] digest_result;
  {dut} dut(.clk_i(clk_i),.rst_ni(rst_ni),.start_i(start_i),.key_i(key_i),.message_i(message_i),.busy_o(busy_o),.done_o(done_o),.digest_o(digest_o),.fault_enable_i(fault_enable_i),.fault_selector_i(fault_selector_i),.fault_value_i(fault_value_i),.fault_raw_o(fault_raw_o));
  always #5 clk_i=~clk_i;
  always @(monitor_clear or fault_enable_i or fault_value_i or fault_raw_o) begin
    if(monitor_clear) activity_latched=0;
    else if(fault_enable_i && (fault_raw_o !== fault_value_i)) activity_latched=1;
  end
  task automatic reset_dut; begin
    start_i=0; fault_enable_i=0; fault_selector_i='0; fault_value_i=0; monitor_clear=1; rst_ni=0;
    repeat(4) @(posedge clk_i); @(negedge clk_i); rst_ni=1; repeat(2) @(posedge clk_i);
  end endtask
  task automatic run_transaction(input integer vi,input logic en,input integer selector,input logic forced,output integer cycles,output logic timed_out,output logic [255:0] measured_digest,output logic activity_seen,output logic unknown_seen); begin
    reset_dut(); @(negedge clk_i); key_i=key_vectors[vi]; message_i=message_vectors[vi];
    fault_selector_i=selector; fault_value_i=forced; monitor_clear=0; fault_enable_i=en;
    @(negedge clk_i); start_i=1; @(negedge clk_i); start_i=0; cycles=0;
    while(!done_o && cycles<TIMEOUT_CYCLES) begin @(posedge clk_i); #1; cycles=cycles+1; end
    timed_out=!done_o; measured_digest=digest_o; activity_seen=activity_latched;
    unknown_seen=$isunknown({{digest_o,done_o,busy_o,fault_raw_o}});
    fault_enable_i=0; monitor_clear=1; @(posedge clk_i);
  end endtask
  initial begin
    $readmemh("{sv_quote(str(key_mem))}",key_vectors);
    $readmemh("{sv_quote(str(message_mem))}",message_vectors);
    $readmemh("{sv_quote(str(digest_mem))}",digest_vectors);
{chr(10).join(site_init)}
    csv_fd=$fopen("{sv_quote(str(csv_path))}","w"); if(csv_fd==0) $fatal(1,"CSV open failed");
    $fdisplay(csv_fd,"{','.join(CSV_FIELDS)}");
    for(vector_index=0;vector_index<VECTOR_COUNT;vector_index=vector_index+1) begin
      run_transaction(vector_index,0,0,0,cycles_result,timeout_result,digest_result,activity_result,unknown_result);
      baseline_cycles[vector_index]=cycles_result; baseline_runs=baseline_runs+1;
      if(timeout_result) baseline_timeouts=baseline_timeouts+1;
      if(digest_result!==digest_vectors[vector_index]) baseline_digest_mismatches=baseline_digest_mismatches+1;
      if(cycles_result!=EXPECTED_BASELINE_CYCLES) baseline_latency_mismatches=baseline_latency_mismatches+1;
      if(unknown_result) unknown_runs=unknown_runs+1;
      $fdisplay(csv_fd,"%0d,BASELINE,BASELINE,-1,-1,%0d,%0d,%0d,%0d,0,0,%0d,%064h,%064h",BATCH_ID,vector_index,cycles_result,EXPECTED_BASELINE_CYCLES,timeout_result,unknown_result,digest_vectors[vector_index],digest_result);
    end
    for(site_index=0;site_index<SITE_COUNT;site_index=site_index+1) begin
      for(stuck_index=0;stuck_index<2;stuck_index=stuck_index+1) begin
        stuck_value=(stuck_index!=0);
        for(vector_index=0;vector_index<VECTOR_COUNT;vector_index=vector_index+1) begin
          run_transaction(vector_index,1,selected_selectors[site_index],stuck_value,cycles_result,timeout_result,digest_result,activity_result,unknown_result);
          detected_result=timeout_result||(digest_result!==digest_vectors[vector_index])||(cycles_result!=baseline_cycles[vector_index]); enabled_runs=enabled_runs+1;
          if(unknown_result) unknown_runs=unknown_runs+1;
          if(detected_result&&!activity_result) detected_without_activity=detected_without_activity+1;
          $fdisplay(csv_fd,"%0d,ENABLED,%s,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%064h,%064h",BATCH_ID,selected_site_ids[site_index],selected_selectors[site_index],stuck_index,vector_index,cycles_result,baseline_cycles[vector_index],timeout_result,activity_result,detected_result,unknown_result,digest_vectors[vector_index],digest_result);
        end
      end
    end
    $fclose(csv_fd);
    $display("V21_PILOT_BATCH=%0d",BATCH_ID); $display("V21_PILOT_BASELINES=%0d",baseline_runs);
    $display("V21_PILOT_ENABLED=%0d",enabled_runs); $display("V21_PILOT_UNKNOWN=%0d",unknown_runs);
    $display("V21_PILOT_DETECTED_WITHOUT_ACTIVITY=%0d",detected_without_activity);
    if(baseline_runs==VECTOR_COUNT&&enabled_runs==SITE_COUNT*2*VECTOR_COUNT&&unknown_runs==0&&detected_without_activity==0&&baseline_digest_mismatches==0&&baseline_latency_mismatches==0&&baseline_timeouts==0) begin
      $display("V21_PILOT_BATCH_RESULT=PASS"); $finish;
    end else $fatal(1,"V21_PILOT_BATCH_RESULT=FAIL");
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


def validate_batch_csv(path: Path, batch_id: int, sites: list[dict[str, Any]]) -> dict[str, Any]:
    expected_enabled = len(sites) * 2 * VECTOR_COUNT
    baseline = enabled = unknown = detected_without_activity = 0
    seen: set[tuple[str, int, int]] = set()
    wanted = {site["site_id"] for site in sites}
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames == CSV_FIELDS, f"CSV header Batch {batch_id:03d}")
        for row in reader:
            require(int(row["batch_id"]) == batch_id, "CSV batch ID")
            unknown += int(row["unknown"])
            if row["run_type"] == "BASELINE":
                baseline += 1
                require(row["timed_out"] == "0", "baseline timeout")
                require(row["expected_digest"].lower() == row["actual_digest"].lower(), "baseline digest")
            elif row["run_type"] == "ENABLED":
                enabled += 1
                require(row["site_id"] in wanted, "unexpected pilot site")
                identity = (row["site_id"], int(row["stuck_value"]), int(row["vector_index"]))
                require(identity not in seen, "duplicate enabled sample")
                seen.add(identity)
                detected_without_activity += int(row["detected"] == "1" and row["activity"] == "0")
            else:
                stop("unknown CSV run type")
    require(baseline == VECTOR_COUNT, f"baseline count Batch {batch_id:03d}")
    require(enabled == expected_enabled and len(seen) == expected_enabled, f"enabled coverage Batch {batch_id:03d}")
    require(unknown == 0, f"unknown records Batch {batch_id:03d}")
    require(detected_without_activity == 0, f"detected-without-activity Batch {batch_id:03d}")
    return {
        "baseline_records": baseline, "enabled_records": enabled,
        "total_records": baseline + enabled, "unknown_records": unknown,
        "detected_without_activity": detected_without_activity,
    }


def execute(resume: bool) -> None:
    require(not AUDIT.exists() and not MANIFEST.exists() and not DATASET.exists(), "Stage 12B-2C is already frozen")
    verify_inputs()
    _, keys, messages, digests = load_vectors()
    pilot_rows, batches = load_pilot_plan()
    batch_ids = sorted(batches)
    require(len(batch_ids) > 0, "empty pilot batch set")
    WORK.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    BUILD.mkdir(parents=True, exist_ok=True)
    key_mem = WORK / "candidate_keys_12b2c.mem"
    message_mem = WORK / "candidate_messages_12b2c.mem"
    digest_mem = WORK / "candidate_expected_digests_12b2c.mem"
    for path, values in ((key_mem, keys), (message_mem, messages), (digest_mem, digests)):
        if path.exists():
            expected = "".join(row.tobytes().hex() + "\n" for row in values).encode()
            require(path.read_bytes() == expected, f"resume memory-file mismatch: {path.name}")
        else:
            memory_file(path, values)

    if resume:
        state = load_json(CHECKPOINT)
        require(state.get("stage") == STAGE and state.get("status") == "RUNNING", "resume checkpoint state")
        require(state.get("authorization_sha256") == PINNED[AUTHORIZATION], "resume authorization")
        require(state.get("candidate_vectors_sha256") == PINNED[VECTOR_NPZ], "resume vector pool")
        completed = set(int(value) for value in state.get("completed_batches", []))
    else:
        require(not CHECKPOINT.exists(), "checkpoint exists; use --resume after verifying the interruption")
        completed: set[int] = set()
        state = {
            "checkpoint_version": "CIRCUITSAGE-HMAC-V2.1-PILOT-CHECKPOINT-v1",
            "stage": STAGE, "status": "RUNNING", "created_at": now(), "updated_at": now(),
            "authorization_sha256": PINNED[AUTHORIZATION],
            "candidate_vectors_sha256": PINNED[VECTOR_NPZ],
            "pilot_sites_sha256": PINNED[PILOT_SITES],
            "planned_batches": batch_ids, "completed_batches": [], "batches": {},
        }
        atomic_json(CHECKPOINT, state)

    verilator = shutil.which("verilator")
    require(verilator is not None, "verilator disappeared from PATH")
    print(f"Pilot batches          : {len(batch_ids)} canonical batches")
    print(f"Pilot sites / faults   : {PILOT_SITE_COUNT} / {FAULT_COUNT}")
    print(f"Candidate vectors      : {VECTOR_COUNT}")
    print(f"Enabled records        : {ENABLED_RECORDS}")
    print("Execution              : SEQUENTIAL; build jobs=1")
    print(f"Checkpoint             : {CHECKPOINT}")

    for batch_id in batch_ids:
        key = f"{batch_id:03d}"
        selected = batches[batch_id]
        if batch_id in completed:
            item = state["batches"].get(key, {})
            csv_path = Path(item.get("csv", ""))
            require(csv_path.is_file() and sha256(csv_path) == item.get("csv_sha256"), f"resume evidence Batch {key}")
            validate_batch_csv(csv_path, batch_id, selected)
            print(f"Batch {key}: RESUME SKIP — VERIFIED", flush=True)
            continue
        batch_result = RAW / f"batch_{key}"
        batch_build = BUILD / f"batch_{key}"
        batch_result.mkdir(parents=True, exist_ok=True)
        batch_build.mkdir(parents=True, exist_ok=True)
        netlist = ROOT / f"build/hmac_fault_batches_canonical_11c4g/batch_{key}/opentitan_hmac_sha256_msg32_faultbatch{key}.v"
        mapping = ROOT / f"results/hmac_fault_campaign_11c4/canonical_batches/batch_{key}/hmac_fault_batch_{key}_mapping.json"
        require(netlist.is_file(), f"missing canonical netlist Batch {key}")
        tb = batch_build / f"tb_v21_pilot_batch_{key}.sv"
        csv_path = batch_result / f"circuitsage_hmac_v2_1_pilot_batch_{key}.csv"
        tb_payload = generate_testbench(batch_id, selected, csv_path.resolve(), key_mem.resolve(), message_mem.resolve(), digest_mem.resolve()).encode()
        if tb.exists():
            require(tb.read_bytes() == tb_payload, f"existing testbench mismatch Batch {key}")
        else:
            tb.write_bytes(tb_payload)
        top = f"tb_v21_pilot_batch_{key}"
        obj_dir = batch_build / "obj_dir"
        build_log = batch_result / "verilator_build.log"
        print(f"Batch {key}: BUILD sites={len(selected)}", flush=True)
        command = [
            verilator, "--binary", "--timing", "--assert", "-Wall", "-Wno-fatal", "-j", "1",
            "--Mdir", str(obj_dir), "--top-module", top, str(netlist), str(tb),
        ]
        rc = run_logged(command, build_log, 1800)
        require(rc == 0, f"Batch {key} build failed rc={rc}")
        binary = obj_dir / f"V{top}"
        require(binary.is_file() and os.access(binary, os.X_OK), f"Batch {key} binary missing")
        simulation_log = batch_result / "simulation.log"
        resource_log = batch_result / "resources.log"
        print(f"Batch {key}: SIMULATE", flush=True)
        rc = run_logged([str(binary)], simulation_log, 14400, resource_log)
        require(rc == 0, f"Batch {key} simulation failed rc={rc}")
        require("V21_PILOT_BATCH_RESULT=PASS" in simulation_log.read_text(errors="replace"), f"Batch {key} PASS token")
        summary = validate_batch_csv(csv_path, batch_id, selected)
        item = {
            "status": "PASS", "batch_id": batch_id, "completed_at": now(), "sites": len(selected),
            "netlist": record(netlist), "mapping": record(mapping), "testbench": record(tb),
            "binary": record(binary), "csv": str(csv_path), "csv_sha256": sha256(csv_path),
            "csv_bytes": csv_path.stat().st_size, "build_log": record(build_log),
            "simulation_log": record(simulation_log), "resource_log": record(resource_log),
            "summary": summary,
        }
        state["batches"][key] = item
        completed.add(batch_id)
        state["completed_batches"] = sorted(completed)
        state["updated_at"] = now()
        atomic_json(CHECKPOINT, state)
        print(f"Batch {key}: PASS records={summary['total_records']} csv_sha256={item['csv_sha256']}", flush=True)

    require(completed == set(batch_ids), "incomplete pilot batches")
    freeze_dataset(state, pilot_rows, batches, digests)


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


def freeze_dataset(state: dict[str, Any], pilot_rows: list[dict[str, Any]],
                   batches: dict[int, list[dict[str, Any]]], digests: np.ndarray) -> None:
    print("CONSOLIDATING AND VERIFYING PILOT DATASET", flush=True)
    site_to_rank = {row["site_id"]: row["pilot_rank"] for row in pilot_rows}
    site_indices = np.asarray([row["site_index"] for row in pilot_rows], dtype=np.int32)
    timeout = np.zeros((FAULT_COUNT, VECTOR_COUNT), dtype=np.uint8)
    activity = np.zeros_like(timeout)
    detected = np.zeros_like(timeout)
    cycles = np.zeros((FAULT_COUNT, VECTOR_COUNT), dtype=np.uint16)
    digest_xor = np.zeros((FAULT_COUNT, VECTOR_COUNT, 32), dtype=np.uint8)
    seen = np.zeros((FAULT_COUNT, VECTOR_COUNT), dtype=np.uint8)
    baseline_cycles: np.ndarray | None = None
    baseline_records = enabled_records = unknown_records = detected_without_activity = 0
    raw_records: dict[str, Any] = {}
    for batch_id in sorted(batches):
        key = f"{batch_id:03d}"
        item = state["batches"][key]
        csv_path = Path(item["csv"])
        require(sha256(csv_path) == item["csv_sha256"], f"checkpoint/CSV agreement Batch {key}")
        local_baseline = np.zeros(VECTOR_COUNT, dtype=np.uint16)
        with csv_path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            require(reader.fieldnames == CSV_FIELDS, f"freeze CSV header Batch {key}")
            for row in reader:
                vector = int(row["vector_index"])
                if row["run_type"] == "BASELINE":
                    baseline_records += 1
                    local_baseline[vector] = int(row["cycles"])
                    require(row["expected_digest"].lower() == row["actual_digest"].lower(), "golden digest mismatch")
                else:
                    enabled_records += 1
                    rank = int(site_to_rank[row["site_id"]])
                    fault = rank * 2 + int(row["stuck_value"])
                    require(seen[fault, vector] == 0, "duplicate consolidated sample")
                    seen[fault, vector] = 1
                    timeout[fault, vector] = int(row["timed_out"])
                    activity[fault, vector] = int(row["activity"])
                    detected[fault, vector] = int(row["detected"])
                    cycles[fault, vector] = int(row["cycles"])
                    unknown_records += int(row["unknown"])
                    detected_without_activity += int(row["detected"] == "1" and row["activity"] == "0")
                    expected = bytes.fromhex(row["expected_digest"])
                    actual = bytes.fromhex(row["actual_digest"])
                    digest_xor[fault, vector] = np.frombuffer(bytes(a ^ b for a, b in zip(expected, actual)), dtype=np.uint8)
        require(np.all(local_baseline > 0), f"baseline vector coverage Batch {key}")
        if baseline_cycles is None:
            baseline_cycles = local_baseline
        else:
            require(np.array_equal(baseline_cycles, local_baseline), f"cross-batch baseline replay Batch {key}")
        raw_records[key] = record(csv_path)
    require(baseline_cycles is not None, "no baseline data")
    require(baseline_records == len(batches) * VECTOR_COUNT, "total baseline records")
    require(enabled_records == ENABLED_RECORDS, "total enabled records")
    require(np.all(seen == 1), "missing pilot samples")
    require(unknown_records == 0 and detected_without_activity == 0, "semantic integrity counters")
    require(np.all(baseline_cycles == EXPECTED_BASELINE_CYCLES), "baseline latency agreement")
    require(np.array_equal((np.any(digest_xor != 0, axis=2) | (cycles != baseline_cycles[None, :]) | (timeout != 0)).astype(np.uint8), detected), "detection replay")

    fault_site_index = np.repeat(site_indices, 2)
    stuck_value = np.tile(np.asarray([0, 1], dtype=np.uint8), PILOT_SITE_COUNT)
    arrays = {
        "activity": activity, "baseline_cycles": baseline_cycles,
        "cycles": cycles, "detected": detected, "digest_xor": digest_xor,
        "fault_instance_index": np.arange(FAULT_COUNT, dtype=np.int32),
        "pilot_rank": np.repeat(np.arange(PILOT_SITE_COUNT, dtype=np.int32), 2),
        "site_index": fault_site_index, "stuck_value": stuck_value,
        "timed_out": timeout, "vector_index": np.arange(VECTOR_COUNT, dtype=np.int32),
    }
    dataset_payload = deterministic_npz(arrays)
    frozen_write(DATASET, dataset_payload)

    metric_rows: list[dict[str, Any]] = []
    for vector in range(VECTOR_COUNT):
        det = detected[:, vector].astype(bool)
        act = activity[:, vector].astype(bool)
        signatures = set()
        for fault in np.flatnonzero(det):
            signatures.add(hashlib.sha256(
                bytes([int(timeout[fault, vector])])
                + int(cycles[fault, vector]).to_bytes(2, "little")
                + digest_xor[fault, vector].tobytes()
            ).digest())
        metric_rows.append({
            "vector_index": vector,
            "activated_fault_instances": int(np.sum(act)),
            "detected_fault_instances": int(np.sum(det)),
            "new_observable_candidates": int(np.sum(det)),
            "distinct_detected_response_signatures": len(signatures),
            "sa0_detected": int(np.sum(det[0::2])),
            "sa1_detected": int(np.sum(det[1::2])),
            "timeout_fault_instances": int(np.sum(timeout[:, vector])),
        })
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(metric_rows[0]), lineterminator="\n")
    writer.writeheader(); writer.writerows(metric_rows)
    metrics_payload = output.getvalue().encode()
    frozen_write(METRICS_CSV, metrics_payload)
    observable = np.any(detected != 0, axis=1)
    activated = np.any(activity != 0, axis=1)
    schema = {
        "schema_version": "CIRCUITSAGE-HMAC-V2.1-PILOT-RESPONSE-DATASET-12B2C-v1",
        "stage": STAGE, "status": "FROZEN", "format": "DETERMINISTIC NPZ",
        "arrays": {name: {"dtype": str(value.dtype), "shape": list(value.shape)} for name, value in sorted(arrays.items())},
        "sample_key": ["fault_instance_index", "vector_index"],
        "fault_instance_mapping": "fault_instance_index = pilot_rank * 2 + stuck_value",
        "target_or_fault_identity_used_as_simulation_input": "FAULT INJECTION CONTROL ONLY; NEVER MODEL QUERY INPUT",
        "model_features_constructed": False, "model_training_or_inference": False,
        "source_partition": "REPAIR_TRAIN ONLY", "validation_or_holdout_rows": 0,
    }
    frozen_write(SCHEMA, canonical_json(schema))
    state["status"] = "PASS"; state["finished_at"] = now(); state["updated_at"] = now()
    atomic_json(CHECKPOINT, state)
    manifest = {
        "manifest_version": "CIRCUITSAGE-HMAC-V2.1-PILOT-DATASET-MANIFEST-v1",
        "stage": STAGE, "status": "PASS",
        "authorization": record(AUTHORIZATION), "checkpoint": record(CHECKPOINT),
        "raw_batch_csvs": raw_records,
        "outputs": {rel(path): record(path) for path in (DATASET, METRICS_CSV, SCHEMA)},
        "canonical_batches": len(batches), "pilot_sites": PILOT_SITE_COUNT,
        "fault_instances": FAULT_COUNT, "candidate_vectors": VECTOR_COUNT,
        "baseline_records": baseline_records, "enabled_records": enabled_records,
        "total_records": baseline_records + enabled_records,
        "observable_fault_instances": int(np.sum(observable)),
        "activated_fault_instances": int(np.sum(activated)),
        "unknown_records": unknown_records, "detected_without_activity": detected_without_activity,
        "missing_samples": int(np.sum(seen == 0)), "duplicate_samples": 0,
        "model_objects_deserialized": 0, "training_calls": 0, "inference_calls": 0,
        "repair_site_test_access": 0, "original_dev_site_test_access": 0,
        "validation_access": 0, "holdout_access": 0,
    }
    frozen_write(MANIFEST, canonical_json(manifest))
    audit = {
        "audit_version": "CIRCUITSAGE-HMAC-V2.1-PILOT-SCREENING-DATASET-INTEGRITY-FREEZE-v1",
        "stage": STAGE, "status": "PASS",
        "pilot_execution_status": "COMPLETED / FROZEN",
        "dataset_status": "FROZEN", "batches_verified": f"{len(batches)}/{len(batches)}",
        "candidate_vectors": VECTOR_COUNT, "pilot_sites": PILOT_SITE_COUNT,
        "fault_instances": FAULT_COUNT, "enabled_records": enabled_records,
        "observable_fault_instances": int(np.sum(observable)),
        "activated_fault_instances": int(np.sum(activated)),
        "missing_duplicate_samples": [0, 0], "unknown_records": 0,
        "detected_without_activity": 0, "deterministic_dataset_replay": "PASS / EXACT",
        "adaptive_vector_selection": "AUTHORIZED / NOT STARTED",
        "full_repair_campaign": "NOT AUTHORIZED", "model_training": "NOT AUTHORIZED",
        "repair_site_test": "LOCKED / NOT AUTHORIZED",
        "original_dev_site_test": "CONSUMED / NOT REOPENED",
        "validation_access": 0, "holdout_access": 0,
        "v1_modified": False, "v2_core_modified": False, "v2_1_modified": False,
        "dataset": record(DATASET), "metrics": record(METRICS_CSV),
        "manifest": record(MANIFEST),
        "next_gate": "STAGE 12B-2D — ADAPTIVE VECTOR SELECTION AND FULL REPAIR-CAMPAIGN CONTRACT FREEZE",
    }
    frozen_write(AUDIT, canonical_json(audit))
    require(dataset_payload == DATASET.read_bytes(), "exact dataset replay")
    for path in (SCHEMA, MANIFEST, AUDIT):
        require(canonical_json(load_json(path)) == path.read_bytes(), f"canonical JSON replay: {path.name}")

    print("\nSTAGE 12B-2C — ADAPTIVE VECTOR PILOT-SCREENING EXECUTION AND DATASET INTEGRITY FREEZE")
    print(f"{'Status':<48}: PASS")
    print(f"{'Pilot execution / dataset':<48}: COMPLETED / FROZEN")
    print(f"{'Canonical batches':<48}: {len(batches)}")
    print(f"{'Candidate vectors':<48}: {VECTOR_COUNT}")
    print(f"{'Pilot sites / fault instances':<48}: {PILOT_SITE_COUNT} / {FAULT_COUNT}")
    print(f"{'Baseline / enabled records':<48}: {baseline_records} / {enabled_records}")
    print(f"{'Observable / activated faults':<48}: {int(np.sum(observable))} / {int(np.sum(activated))}")
    print(f"{'Missing / duplicate samples':<48}: 0 / 0")
    print(f"{'Unknown / detected without activity':<48}: 0 / 0")
    print(f"{'Model training / inference':<48}: 0 / 0")
    print(f"{'Repair SITE_TEST':<48}: LOCKED / NOT AUTHORIZED")
    print(f"{'VALIDATION / HOLDOUT access':<48}: 0 / 0")
    print(f"{'Dataset':<48}: {DATASET}")
    print(f"{'Dataset SHA':<48}: {sha256(DATASET)}")
    print(f"{'Vector metrics':<48}: {METRICS_CSV}")
    print(f"{'Vector metrics SHA':<48}: {sha256(METRICS_CSV)}")
    print(f"{'Checkpoint':<48}: {CHECKPOINT}")
    print(f"{'Checkpoint SHA':<48}: {sha256(CHECKPOINT)}")
    print(f"{'Manifest':<48}: {MANIFEST}")
    print(f"{'Manifest SHA':<48}: {sha256(MANIFEST)}")
    print(f"{'Audit':<48}: {AUDIT}")
    print(f"{'Audit SHA':<48}: {sha256(AUDIT)}")
    print(f"{'Next gate':<48}: STAGE 12B-2D — ADAPTIVE VECTOR SELECTION AND FULL REPAIR-CAMPAIGN CONTRACT FREEZE")


def status() -> None:
    print("STAGE 12B-2C — PILOT-SCREENING STATUS")
    if AUDIT.is_file():
        audit = load_json(AUDIT)
        print(f"Status                 : {audit.get('status')} / FROZEN")
        print(f"Batches verified       : {audit.get('batches_verified')}")
        print(f"Observable faults      : {audit.get('observable_fault_instances')}/{FAULT_COUNT}")
        print(f"Audit                  : {AUDIT}")
        print(f"Audit SHA              : {sha256(AUDIT)}")
        return
    if not CHECKPOINT.is_file():
        print("Status                 : NOT STARTED")
        return
    state = load_json(CHECKPOINT)
    completed = state.get("completed_batches", [])
    planned = state.get("planned_batches", [])
    print(f"Status                 : {state.get('status')}")
    print(f"Completed batches      : {len(completed)}/{len(planned)}")
    if completed:
        print(f"Latest completed batch : {max(completed):03d}")
    print(f"Checkpoint             : {CHECKPOINT}")
    print(f"Checkpoint SHA         : {sha256(CHECKPOINT)}")


def self_test() -> None:
    arrays = {"a": np.arange(8, dtype=np.uint8), "b": np.arange(3, dtype=np.int32)}
    require(deterministic_npz(arrays) == deterministic_npz(arrays), "deterministic NPZ canary")
    require(len(hmac.new(b"k" * 32, b"m" * 32, hashlib.sha256).digest()) == 32, "HMAC canary")
    print("Stage 12B-2C self-test: PASS")


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
