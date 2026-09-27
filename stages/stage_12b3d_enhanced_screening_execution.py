#!/usr/bin/env python3
"""Stage 12B-3D: enhanced-measurement bounded-screen execution and freeze.

Runs the Stage 12B-3C-authorized REPAIR_TRAIN-only screen.  A passive union
observer captures all three frozen candidates in one physical simulation per
transaction; the resulting traces are projected into the three precommitted
candidate views.  This keeps execution below the authorized maximum while
preserving candidate-equivalent observations.

The campaign is sequential and checkpoint/resume capable.  No model is loaded,
trained, calibrated or evaluated, and no protected partition is opened.
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


STAGE = "12B-3D"
ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "config/v2_1_improvement"
RESULT = ROOT / "results/circuitsage_hmac_v2_12b3"
WORK = RESULT / "enhanced_screening_execution_12b3d"
RAW = WORK / "raw_batches"
PER_BATCH = WORK / "per_batch_npz"
BUILD = ROOT / "build/circuitsage_hmac_v2_12b3/enhanced_screening_12b3d"
CHECKPOINT = WORK / "circuitsage_hmac_v2_1_enhanced_screening_checkpoint_12b3d.json"
TARGETS = WORK / "circuitsage_hmac_v2_1_enhanced_screening_targets_12b3d.npz"
METRICS_CSV = WORK / "circuitsage_hmac_v2_1_enhanced_screening_candidate_metrics_12b3d.csv"
METRICS_JSON = WORK / "circuitsage_hmac_v2_1_enhanced_screening_candidate_metrics_12b3d.json"
BOOTSTRAP = WORK / "circuitsage_hmac_v2_1_enhanced_screening_site_bootstrap_12b3d.csv"
SCHEMA = WORK / "circuitsage_hmac_v2_1_enhanced_screening_dataset_schema_12b3d.json"
MANIFEST = RESULT / "circuitsage_hmac_v2_1_enhanced_screening_dataset_manifest_12b3d.json"
AUDIT = RESULT / "circuitsage_hmac_v2_1_enhanced_screening_execution_dataset_freeze_12b3d.json"

SOURCE_3C = ROOT / "stage_12b3c_enhanced_screening_authorization.py"
AUTH_DIR = RESULT / "enhanced_screening_authorization_12b3c"
SCREEN_SITES = AUTH_DIR / "circuitsage_hmac_v2_1_enhanced_screening_sites_12b3c.csv"
SCREEN_VECTORS = AUTH_DIR / "circuitsage_hmac_v2_1_enhanced_screening_vectors_12b3c.npz"
SCREEN_SCHEMA_3C = AUTH_DIR / "circuitsage_hmac_v2_1_enhanced_screening_schema_12b3c.json"
EXECUTION_PLAN = AUTH_DIR / "circuitsage_hmac_v2_1_enhanced_screening_execution_plan_12b3c.csv"
PREFLIGHT_3C = AUTH_DIR / "circuitsage_hmac_v2_1_enhanced_screening_preflight_12b3c.json"
EXECUTION_CONTRACT = CONFIG / "circuitsage_hmac_v2_1_enhanced_screening_execution_contract_12b3c.json"
AUTHORIZATION = CONFIG / "circuitsage_hmac_v2_1_enhanced_screening_authorization_12b3c.json"
MANIFEST_3C = RESULT / "circuitsage_hmac_v2_1_enhanced_screening_authorization_manifest_12b3c.json"
AUDIT_3C = RESULT / "circuitsage_hmac_v2_1_enhanced_screening_authorization_freeze_12b3c.json"

DISCOVERY_DIR = RESULT / "enhanced_probe_discovery_12b3b"
PROBE_BANK_CSV = DISCOVERY_DIR / "circuitsage_hmac_v2_1_enhanced_probe_banks_12b3b.csv"
CONSISTENCY_3B = DISCOVERY_DIR / "circuitsage_hmac_v2_1_enhanced_cross_batch_consistency_12b3b.json"

PINNED = {
    SOURCE_3C: "4fa30e82324382a8063cd9b49e4dd5c5db93458d269f6f3984983eee64d1e94f",
    SCREEN_SITES: "497626404598a616bd6bf9b71a8876a4ff1c4071db2fa242c6b86a0bb8d8dcea",
    SCREEN_VECTORS: "9bfed2ad7046b842f3dc1b8899c2b683937fdca2c87f5b1e3c1c4c953817b5e4",
    SCREEN_SCHEMA_3C: "338b7a52446f24166e33682205806f68da3ea64ff3f5ed841acdb8663f7a2e92",
    EXECUTION_PLAN: "b6fd254f1e673a192fbedcb150de2d1782e636a13d47ba28536757cb0b9887cb",
    PREFLIGHT_3C: "31c40035f6caad9cc700a089fb3dd80074a40f840ec384508a7de0da07ee45e9",
    EXECUTION_CONTRACT: "e694b6cac9f50e5f56db9fcefe6c0e6bd66980c447dcd0f23530a68c9ce9fea1",
    AUTHORIZATION: "c89a6676dd3b8c46f8fa876c9fce3d317d047aa975a5fabe4081474330ab6397",
    MANIFEST_3C: "9a59b2570637c5458d19bf3efd4288f2ccae578ab68bece39469981f11a0aa9d",
    AUDIT_3C: "bf809b7c2ad1978b51264c912b742471cfa1ffa26fa6122f53ea69665622d8d2",
    PROBE_BANK_CSV: "d7f9b984d48db4e9a8c242e1ee4e859ef5f321d6d5546b901bde569d5009eae2",
    CONSISTENCY_3B: "6c94117035b979135657d2b384b11db40a7c63029f34df365f8f34360d4bbb20",
}

CANDIDATES = {
    "EM_TOPOLOGY_4X64_T24": {"tag": "topology", "probe_bits": 256, "snapshots": 24},
    "EM_STATE_CHECKPOINT_2X64_T32": {"tag": "state", "probe_bits": 128, "snapshots": 32},
    "EM_TESTPOINT_4X64_T16": {"tag": "testpoint", "probe_bits": 256, "snapshots": 16},
}
FEATURES = {name: WORK / f"circuitsage_hmac_v2_1_{spec['tag']}_screening_features_12b3d.npz"
            for name, spec in CANDIDATES.items()}
BATCHES = 45
SITES = 256
FAULTS = 512
VECTORS = 48
LOGICAL_ENABLED_PER_CANDIDATE = FAULTS * VECTORS
LOGICAL_ENABLED_TOTAL = LOGICAL_ENABLED_PER_CANDIDATE * 3
PHYSICAL_ENABLED = FAULTS * VECTORS
TIMEOUT_CYCLES = 2000
EXPECTED_BASELINE_CYCLES = 343
MIN_FREE_GIB = 10
BOOTSTRAP_REPLICATES = 1000
FROZEN_TARGET = 0.70

# Slots 0..30 are fixed cycles; slot 31 is DONE_OR_TIMEOUT.
UNION_CYCLES = [0, 1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64, 72, 80,
                88, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256,
                272, 288, 304, 320]
CANDIDATE_SLOTS = {
    "EM_TOPOLOGY_4X64_T24": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 17, 19, 21, 23, 25, 27, 29, 30, 31],
    "EM_STATE_CHECKPOINT_2X64_T32": list(range(32)),
    "EM_TESTPOINT_4X64_T16": [0, 1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 31],
}
CSV_FIELDS = [
    "batch_id", "run_type", "site_id", "selector", "stuck_value", "vector_rank",
    "source_vector_index", "cycles", "timed_out", "busy_first_cycle", "busy_last_cycle",
    "done_cycle", "unknown", "expected_digest", "actual_digest", "snapshot_hex", "toggle_hex",
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
    path = resolve_record(item)
    require(path.resolve() == expected.resolve(), f"{label} path")
    require(path.is_file(), f"missing {label}: {rel(path)}")
    require(item.get("sha256") == sha256(path), f"{label} SHA")
    require(int(item.get("bytes", -1)) == path.stat().st_size, f"{label} size")


def csv_payload(rows: list[dict[str, Any]]) -> bytes:
    require(bool(rows), "CSV rows")
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader(); writer.writerows(rows)
    return output.getvalue().encode()


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
            info.compress_type = zipfile.ZIP_DEFLATED; info.create_system = 3; info.external_attr = 0o600 << 16
            archive.writestr(info, npy_bytes(arrays[name]), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    return stream.getvalue()


def memory_file(path: Path, rows: np.ndarray) -> None:
    path.write_text("".join(row.tobytes().hex() + "\n" for row in rows), encoding="ascii")


def sv_quote(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def run_logged(command: list[str], log: Path, timeout: int, resource: Path | None = None) -> int:
    actual = (["/usr/bin/time", "-v", "-o", str(resource)] if resource is not None else []) + list(command)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as stream:
        try:
            return subprocess.run(actual, stdout=stream, stderr=subprocess.STDOUT, timeout=timeout, check=False).returncode
        except subprocess.TimeoutExpired:
            stream.write(f"\nTIMEOUT_SECONDS={timeout}\n")
            return 124


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


def verify_inputs() -> tuple[dict[str, list[dict[str, str]]], dict[int, dict[str, Any]]]:
    print("STAGE 12B-3D — ENHANCED MEASUREMENT BOUNDED-SCREENING EXECUTION")
    print("FROZEN INPUT VERIFICATION")
    for path, expected in PINNED.items():
        require(path.is_file(), f"missing frozen input: {rel(path)}")
        require(sha256(path) == expected, f"frozen input SHA: {rel(path)}")
        print(f"  {path.name:<91}: OK", flush=True)
    audit = load_json(AUDIT_3C)
    authorization = load_json(AUTHORIZATION)
    contract = load_json(EXECUTION_CONTRACT)
    manifest = load_json(MANIFEST_3C)
    require(audit.get("status") == "PASS" and audit.get("authorization_status") == "FROZEN", "12B-3C freeze")
    require(audit.get("bounded_three_candidate_screening") == "AUTHORIZED / NOT STARTED", "screen authorization")
    require(audit.get("screen_sites_faults_vectors") == [SITES, FAULTS, VECTORS], "screen dimensions")
    require(audit.get("maximum_total_enabled_transactions") == LOGICAL_ENABLED_TOTAL, "logical transaction budget")
    require(audit.get("fault_identity_in_query") == "PROHIBITED", "identity boundary")
    require(audit.get("model_training") == "NOT AUTHORIZED", "training boundary")
    require(audit.get("repair_site_test") == "LOCKED / NOT ACCESSED", "repair test boundary")
    require(audit.get("validation_access") == 0 and audit.get("holdout_access") == 0, "protected access")
    require(authorization.get("bounded_three_candidate_screening") == "AUTHORIZED / NOT STARTED", "authorization object")
    require(authorization.get("model_training") == "NOT AUTHORIZED", "authorization training boundary")
    require(contract.get("status") == "FROZEN" and contract.get("scope") == "REPAIR_TRAIN BOUNDED SCREEN ONLY", "execution scope")
    require(contract.get("candidate_ids") == list(CANDIDATES), "candidate order")
    require(contract.get("checkpoint_resume") == "REQUIRED AFTER EACH CANDIDATE/BATCH PAIR", "checkpoint rule")
    require(manifest.get("status") == "PASS", "3C manifest")
    outputs = manifest.get("outputs")
    require(isinstance(outputs, dict), "3C output registry")
    for path in (SCREEN_SITES, SCREEN_VECTORS, SCREEN_SCHEMA_3C, EXECUTION_PLAN, PREFLIGHT_3C, EXECUTION_CONTRACT, AUTHORIZATION):
        item = outputs.get(rel(path)); require(isinstance(item, dict), f"3C record {path.name}")
        verify_record(item, path, f"3C {path.name}")
    require(shutil.which("yosys") is not None and shutil.which("verilator") is not None, "toolchain unavailable")
    require(shutil.disk_usage(ROOT).free >= MIN_FREE_GIB * 1024**3, f"less than {MIN_FREE_GIB} GiB free disk")

    with PROBE_BANK_CSV.open(newline="", encoding="utf-8") as stream:
        probe_rows = list(csv.DictReader(stream))
    candidates = {name: [] for name in CANDIDATES}
    for row in probe_rows:
        require(row["candidate_id"] in candidates, "probe candidate ID")
        candidates[row["candidate_id"]].append(row)
    for name, spec in CANDIDATES.items():
        rows = candidates[name]
        require(len(rows) == spec["probe_bits"], f"{name} probe count")
        require([int(row["probe_bit"]) for row in rows] == list(range(spec["probe_bits"])), f"{name} probe order")
    consistency = load_json(CONSISTENCY_3B)
    entries = consistency.get("batches")
    require(consistency.get("status") == "PASS" and isinstance(entries, list) and len(entries) == BATCHES, "3B consistency")
    batch_entries = {int(item["batch_id"]): item for item in entries}
    print("  Authorization, leakage boundaries, candidate banks and toolchain                         : PASS")
    return candidates, batch_entries


def load_vectors() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(SCREEN_VECTORS, allow_pickle=False) as archive:
        required = {"key_u8", "message_u8", "selection_rank", "selection_round", "vector_index"}
        require(set(archive.files) == required, "screen-vector NPZ members")
        keys = np.asarray(archive["key_u8"], dtype=np.uint8)
        messages = np.asarray(archive["message_u8"], dtype=np.uint8)
        source_indices = np.asarray(archive["vector_index"], dtype=np.int32)
        ranks = np.asarray(archive["selection_rank"], dtype=np.int32)
    require(keys.shape == (VECTORS, 32) and messages.shape == (VECTORS, 32), "screen-vector shapes")
    require(ranks.tolist() == list(range(VECTORS)), "screen-vector ranks")
    require(len(set(source_indices.tolist())) == VECTORS, "screen-vector uniqueness")
    digests = np.vstack([
        np.frombuffer(hmac.new(keys[index].tobytes(), messages[index].tobytes(), hashlib.sha256).digest(), dtype=np.uint8)
        for index in range(VECTORS)
    ])
    return keys, messages, source_indices, digests


def load_site_plan(batch_entries: dict[int, dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[int, list[dict[str, Any]]]]:
    with SCREEN_SITES.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    require(len(rows) == SITES and [int(row["screen_rank"]) for row in rows] == list(range(SITES)), "screen-site ordering")
    wanted = {row["site_id"]: row for row in rows}
    require(len(wanted) == SITES, "duplicate screen site")
    batches: dict[int, list[dict[str, Any]]] = {}
    found: set[str] = set()
    for batch_id in range(BATCHES):
        entry = batch_entries[batch_id]
        json_path = resolve_record(entry["canonical_json"])
        mapping_path = resolve_record(entry["mapping"])
        verify_record(entry["canonical_json"], json_path, f"Batch {batch_id:03d} JSON")
        verify_record(entry["mapping"], mapping_path, f"Batch {batch_id:03d} mapping")
        mapping_sites = find_sites(load_json(mapping_path))
        expected = 311 if batch_id == 44 else 512
        require(mapping_sites is not None and len(mapping_sites) == expected, f"Batch {batch_id:03d} mapping count")
        selected: list[dict[str, Any]] = []
        for site in mapping_sites:
            site_id = str(site["fault_site_id"])
            if site_id in wanted:
                require(site_id not in found, f"screen site mapped twice: {site_id}")
                found.add(site_id)
                selected.append({
                    "site_id": site_id, "site_index": int(wanted[site_id]["site_index"]),
                    "screen_rank": int(wanted[site_id]["screen_rank"]),
                    "selector": int(site["selector_code"]), "batch_id": batch_id,
                })
        require(selected, f"Batch {batch_id:03d} has no authorized screen site")
        batches[batch_id] = sorted(selected, key=lambda row: row["screen_rank"])
    require(found == set(wanted) and len(batches) == BATCHES, "screen-site mapping coverage")
    ordered = sorted((row for values in batches.values() for row in values), key=lambda row: row["screen_rank"])
    return ordered, batches


def union_probe_plan(candidates: dict[str, list[dict[str, str]]]) -> tuple[list[int], dict[str, list[int]]]:
    union: list[int] = []
    lookup: dict[int, int] = {}
    positions: dict[str, list[int]] = {}
    for name in CANDIDATES:
        positions[name] = []
        for row in candidates[name]:
            bit = int(row["bit_id"])
            if bit not in lookup:
                lookup[bit] = len(union); union.append(bit)
            positions[name].append(lookup[bit])
    require(len(union) <= 640 and len(union) >= 256, "union probe width")
    return union, positions


def observer_json(canonical: Path, module_name: str, probe_bits: list[int]) -> bytes:
    design = copy.deepcopy(load_json(canonical))
    modules = design.get("modules")
    require(isinstance(modules, dict) and set(modules) == {module_name}, f"{module_name} module set")
    module = modules[module_name]
    require("probe_o" not in module.get("ports", {}) and "probe_o" not in module.get("netnames", {}), "probe_o collision")
    module["ports"]["probe_o"] = {"direction": "output", "bits": probe_bits}
    module["netnames"]["probe_o"] = {"hide_name": 0, "bits": probe_bits, "attributes": {}}
    return canonical_json(design)


def sample_case_lines() -> str:
    return "\n".join(
        f"        {cycle}: begin snapshot_work[{slot}]=probe_o; snapshot_written[{slot}]=1; end"
        for slot, cycle in enumerate(UNION_CYCLES[1:], start=1)
    )


def generate_testbench(batch_id: int, selected: list[dict[str, Any]], csv_path: Path,
                       key_mem: Path, message_mem: Path, digest_mem: Path,
                       source_index_mem: Path, union_bits: int) -> str:
    key = f"{batch_id:03d}"; top = f"tb_v21_enhanced_screen_batch_{key}"
    dut = f"opentitan_hmac_sha256_msg32_faultbatch{key}"
    hex_width = (union_bits + 3) // 4
    site_init: list[str] = []
    for index, site in enumerate(selected):
        site_init += [f"    selected_selectors[{index}] = 9'd{site['selector']};",
                      f"    selected_site_ids[{index}] = \"{site['site_id']}\";"]
    return f'''`timescale 1ns/1ps
module {top};
  localparam integer BATCH_ID={batch_id}, VECTOR_COUNT={VECTORS}, SITE_COUNT={len(selected)};
  localparam integer PROBE_BITS={union_bits}, SNAPSHOTS=32;
  localparam integer TIMEOUT_CYCLES={TIMEOUT_CYCLES}, EXPECTED_BASELINE_CYCLES={EXPECTED_BASELINE_CYCLES};
  logic clk_i=0,rst_ni=0,start_i=0,busy_o,done_o,fault_enable_i=0,fault_value_i=0,fault_raw_o;
  logic [8:0] fault_selector_i='0; logic [255:0] key_i='0,message_i='0,digest_o;
  logic [PROBE_BITS-1:0] probe_o,prev_probe,delta_probe;
  logic [255:0] key_vectors[0:VECTOR_COUNT-1],message_vectors[0:VECTOR_COUNT-1],digest_vectors[0:VECTOR_COUNT-1];
  logic [31:0] source_vector_indices[0:VECTOR_COUNT-1];
  integer selected_selectors[0:SITE_COUNT-1]; string selected_site_ids[0:SITE_COUNT-1];
  logic [PROBE_BITS-1:0] snapshot_work[0:SNAPSHOTS-1]; integer snapshot_written[0:SNAPSHOTS-1];
  integer toggle_work[0:PROBE_BITS-1];
  logic [PROBE_BITS-1:0] baseline_snapshot[0:VECTOR_COUNT-1][0:SNAPSHOTS-1];
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
      @(posedge clk_i); #1; cycles=cycles+1; delta_probe=probe_o^prev_probe;
      for(t=0;t<PROBE_BITS;t=t+1) if(delta_probe[t] && toggle_work[t]<65535) toggle_work[t]=toggle_work[t]+1;
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
      if(compare_baseline) $fwrite(csv_fd,"%0{hex_width}h",snapshot_work[s]^baseline_snapshot[vi][s]);
      else $fwrite(csv_fd,"%0{hex_width}h",snapshot_work[s]);
    end
    $fwrite(csv_fd,","); for(t=0;t<PROBE_BITS;t=t+1) $fwrite(csv_fd,"%04x",toggle_work[t]);
    $fwrite(csv_fd,"\\n");
  end endtask
  initial begin
    $readmemh("{sv_quote(str(key_mem))}",key_vectors); $readmemh("{sv_quote(str(message_mem))}",message_vectors);
    $readmemh("{sv_quote(str(digest_mem))}",digest_vectors); $readmemh("{sv_quote(str(source_index_mem))}",source_vector_indices);
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
    for(site_index=0;site_index<SITE_COUNT;site_index=site_index+1) for(stuck_index=0;stuck_index<2;stuck_index=stuck_index+1) begin
      stuck_value=(stuck_index!=0);
      for(vector_rank=0;vector_rank<VECTOR_COUNT;vector_rank=vector_rank+1) begin
        run_transaction(vector_rank,1,selected_selectors[site_index],stuck_value,cycles_result,timeout_result,digest_result,busy_first_result,busy_last_result,done_cycle_result,unknown_result);
        enabled_runs=enabled_runs+1; if(unknown_result) unknown_runs=unknown_runs+1;
        $fwrite(csv_fd,"%0d,ENABLED,%s,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%064h,%064h,",BATCH_ID,selected_site_ids[site_index],selected_selectors[site_index],stuck_index,vector_rank,source_vector_indices[vector_rank],cycles_result,timeout_result,busy_first_result,busy_last_result,done_cycle_result,unknown_result,digest_vectors[vector_rank],digest_result);
        write_trace_fields(vector_rank,1);
      end
    end
    $fclose(csv_fd); $display("V21_ENHANCED_SCREEN_BATCH=%0d",BATCH_ID);
    $display("V21_ENHANCED_SCREEN_BASELINES=%0d",baseline_runs); $display("V21_ENHANCED_SCREEN_ENABLED=%0d",enabled_runs);
    if(baseline_runs==VECTOR_COUNT&&enabled_runs==SITE_COUNT*2*VECTOR_COUNT&&unknown_runs==0&&baseline_failures==0) begin
      $display("V21_ENHANCED_SCREEN_BATCH_RESULT=PASS"); $finish;
    end else $fatal(1,"V21_ENHANCED_SCREEN_BATCH_RESULT=FAIL baseline_failures=%0d",baseline_failures);
  end
endmodule
'''


def snapshot_bits(text: str, union_width: int) -> np.ndarray:
    padded = text if len(text) % 2 == 0 else "0" + text
    require(re.fullmatch(r"[0-9a-fA-F]+", padded) is not None, "snapshot hex encoding")
    bits_msb = np.unpackbits(np.frombuffer(bytes.fromhex(padded), dtype=np.uint8), bitorder="big")[-union_width:]
    require(bits_msb.size == union_width, "snapshot bit width")
    return bits_msb[::-1].copy()


def pack_candidate(bits_lsb: np.ndarray, positions: list[int]) -> np.ndarray:
    return np.packbits(bits_lsb[np.asarray(positions, dtype=np.int32)], bitorder="little")


def parse_batch_csv(path: Path, batch_id: int, sites: list[dict[str, Any]], source_indices: np.ndarray,
                    union_width: int, positions: dict[str, list[int]]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    local_faults = len(sites) * 2
    hex_width = (union_width + 3) // 4
    common = {
        "baseline_cycles": np.zeros(VECTORS, dtype=np.uint16),
        "baseline_control_timeline": np.zeros((VECTORS, 3), dtype=np.int16),
        "cycles": np.zeros((local_faults, VECTORS), dtype=np.uint16),
        "control_timeline": np.zeros((local_faults, VECTORS, 3), dtype=np.int16),
        "timed_out": np.zeros((local_faults, VECTORS), dtype=np.uint8),
        "digest_xor": np.zeros((local_faults, VECTORS, 32), dtype=np.uint8),
    }
    candidate_arrays: dict[str, dict[str, np.ndarray]] = {}
    for name, spec in CANDIDATES.items():
        tag = spec["tag"]; count = spec["probe_bits"]; snapshots = spec["snapshots"]
        candidate_arrays[name] = {
            f"{tag}_baseline_probe_snapshots": np.zeros((VECTORS, snapshots, count // 8), dtype=np.uint8),
            f"{tag}_baseline_probe_toggle_count": np.zeros((VECTORS, count), dtype=np.uint16),
            f"{tag}_probe_snapshot_xor": np.zeros((local_faults, VECTORS, snapshots, count // 8), dtype=np.uint8),
            f"{tag}_probe_toggle_count": np.zeros((local_faults, VECTORS, count), dtype=np.uint16),
        }
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
            snapshot_text = row["snapshot_hex"]
            require(len(snapshot_text) == 32 * hex_width, "snapshot trace length")
            union_snapshots = [snapshot_bits(snapshot_text[index * hex_width:(index + 1) * hex_width], union_width) for index in range(32)]
            toggle_text = row["toggle_hex"]
            require(len(toggle_text) == union_width * 4 and re.fullmatch(r"[0-9a-fA-F]+", toggle_text) is not None, "toggle trace")
            union_toggles = np.asarray([int(toggle_text[index * 4:(index + 1) * 4], 16) for index in range(union_width)], dtype=np.uint16)
            row_cycles = int(row["cycles"])
            row_control = [int(row["busy_first_cycle"]), int(row["busy_last_cycle"]), int(row["done_cycle"])]
            if row["run_type"] == "BASELINE":
                require(seen_baseline[rank] == 0 and row["site_id"] == "BASELINE", "duplicate/invalid baseline")
                seen_baseline[rank] = 1
                common["baseline_cycles"][rank] = row_cycles
                common["baseline_control_timeline"][rank] = row_control
                for name, spec in CANDIDATES.items():
                    tag = spec["tag"]
                    for out_slot, union_slot in enumerate(CANDIDATE_SLOTS[name]):
                        candidate_arrays[name][f"{tag}_baseline_probe_snapshots"][rank, out_slot] = pack_candidate(union_snapshots[union_slot], positions[name])
                    candidate_arrays[name][f"{tag}_baseline_probe_toggle_count"][rank] = union_toggles[positions[name]]
                baseline_failures += int(row["timed_out"] != "0" or row_cycles != EXPECTED_BASELINE_CYCLES or row["expected_digest"].lower() != row["actual_digest"].lower())
            elif row["run_type"] == "ENABLED":
                require(row["site_id"] in site_to_local, "unexpected screen site")
                stuck = int(row["stuck_value"]); require(stuck in (0, 1), "stuck value")
                fault = site_to_local[row["site_id"]] * 2 + stuck
                require(seen[fault, rank] == 0, "duplicate enabled sample")
                seen[fault, rank] = 1
                common["cycles"][fault, rank] = row_cycles
                common["control_timeline"][fault, rank] = row_control
                common["timed_out"][fault, rank] = int(row["timed_out"])
                expected = bytes.fromhex(row["expected_digest"]); actual = bytes.fromhex(row["actual_digest"])
                require(len(expected) == 32 and len(actual) == 32, "digest encoding")
                common["digest_xor"][fault, rank] = np.frombuffer(bytes(a ^ b for a, b in zip(expected, actual)), dtype=np.uint8)
                for name, spec in CANDIDATES.items():
                    tag = spec["tag"]
                    for out_slot, union_slot in enumerate(CANDIDATE_SLOTS[name]):
                        candidate_arrays[name][f"{tag}_probe_snapshot_xor"][fault, rank, out_slot] = pack_candidate(union_snapshots[union_slot], positions[name])
                    candidate_arrays[name][f"{tag}_probe_toggle_count"][fault, rank] = union_toggles[positions[name]]
            else:
                stop("unknown CSV run type")
    require(np.all(seen_baseline == 1) and np.all(seen == 1), f"sample coverage Batch {batch_id:03d}")
    require(unknown == 0 and baseline_failures == 0, f"semantic failures Batch {batch_id:03d}")
    arrays: dict[str, np.ndarray] = {
        "baseline_cycles": common["baseline_cycles"],
        "baseline_control_timeline": common["baseline_control_timeline"],
        "control_timeline_delta": (common["control_timeline"].astype(np.int32) - common["baseline_control_timeline"][None, :, :]).astype(np.int16),
        "cycle_delta": (common["cycles"].astype(np.int32) - common["baseline_cycles"][None, :]).astype(np.int16),
        "cycles": common["cycles"], "digest_xor": common["digest_xor"], "timed_out": common["timed_out"],
        "fault_instance_index": np.asarray([site["screen_rank"] * 2 + stuck for site in sites for stuck in (0, 1)], dtype=np.int32),
        "source_vector_index": source_indices.astype(np.int32), "vector_rank": np.arange(VECTORS, dtype=np.int32),
    }
    external = ((arrays["timed_out"] != 0) | (arrays["cycle_delta"] != 0) | np.any(arrays["digest_xor"] != 0, axis=2)).astype(np.uint8)
    arrays["external_detected"] = external
    candidate_summary: dict[str, Any] = {}
    for name, spec in CANDIDATES.items():
        tag = spec["tag"]; values = candidate_arrays[name]
        toggles = values[f"{tag}_probe_toggle_count"]
        baseline_toggles = values[f"{tag}_baseline_probe_toggle_count"]
        toggle_delta = (toggles.astype(np.int32) - baseline_toggles[None, :, :].astype(np.int32)).astype(np.int16)
        probe_effect = (np.any(values[f"{tag}_probe_snapshot_xor"] != 0, axis=(2, 3)) |
                        np.any(toggle_delta != 0, axis=2) | np.any(arrays["control_timeline_delta"] != 0, axis=2)).astype(np.uint8)
        arrays.update(values)
        arrays[f"{tag}_probe_toggle_delta"] = toggle_delta
        arrays[f"{tag}_probe_effect"] = probe_effect
        candidate_summary[name] = {
            "probe_effect_faults": int(np.sum(np.any(probe_effect != 0, axis=1))),
            "combined_observable_faults": int(np.sum(np.any((external != 0) | (probe_effect != 0), axis=1))),
        }
    summary = {
        "baseline_records": VECTORS, "enabled_records": local_faults * VECTORS,
        "fault_instances": local_faults, "unknown_records": unknown, "baseline_failures": baseline_failures,
        "externally_detected_faults": int(np.sum(np.any(external != 0, axis=1))),
        "candidates": candidate_summary,
    }
    return arrays, summary


def execute(resume: bool) -> None:
    require(not AUDIT.exists() and not MANIFEST.exists(), f"Stage {STAGE} is already frozen")
    candidates, batch_entries = verify_inputs()
    keys, messages, source_indices, digests = load_vectors()
    site_rows, batches = load_site_plan(batch_entries)
    union_bits, positions = union_probe_plan(candidates)
    WORK.mkdir(parents=True, exist_ok=True); RAW.mkdir(parents=True, exist_ok=True)
    PER_BATCH.mkdir(parents=True, exist_ok=True); BUILD.mkdir(parents=True, exist_ok=True)
    key_mem = WORK / "screen_keys_12b3d.mem"; message_mem = WORK / "screen_messages_12b3d.mem"
    digest_mem = WORK / "screen_expected_digests_12b3d.mem"; source_mem = WORK / "screen_source_indices_12b3d.mem"
    memory_payloads = {
        key_mem: "".join(row.tobytes().hex() + "\n" for row in keys).encode(),
        message_mem: "".join(row.tobytes().hex() + "\n" for row in messages).encode(),
        digest_mem: "".join(row.tobytes().hex() + "\n" for row in digests).encode(),
        source_mem: "".join(f"{int(value):08x}\n" for value in source_indices).encode(),
    }
    for path, payload in memory_payloads.items():
        if path.exists(): require(path.read_bytes() == payload, f"memory-file mismatch: {path.name}")
        else: path.write_bytes(payload)
    if CHECKPOINT.exists():
        require(resume, "checkpoint exists; rerun with --resume")
        state = load_json(CHECKPOINT)
        require(state.get("stage") == STAGE and state.get("union_probe_bits") == len(union_bits), "checkpoint contract")
    else:
        require(not resume, "--resume requested without checkpoint")
        state = {
            "checkpoint_version": "CIRCUITSAGE-HMAC-V2.1-ENHANCED-SCREEN-CHECKPOINT-12B3D-v1",
            "stage": STAGE, "status": "RUNNING", "started_at": now(), "updated_at": now(),
            "candidate_ids": list(CANDIDATES), "planned_batches": list(range(BATCHES)),
            "completed_batches": [], "completed_candidate_batch_pairs": [],
            "union_probe_bits": len(union_bits), "physical_enabled_transactions": PHYSICAL_ENABLED,
            "logical_candidate_evaluations": LOGICAL_ENABLED_TOTAL, "batches": {},
        }
        atomic_json(CHECKPOINT, state)
    completed = set(int(value) for value in state.get("completed_batches", []))
    yosys = shutil.which("yosys"); verilator = shutil.which("verilator")
    require(yosys is not None and verilator is not None, "toolchain unavailable")
    for batch_id in range(BATCHES):
        if batch_id in completed:
            item = state["batches"][f"{batch_id:03d}"]
            require(sha256(Path(item["csv"])) == item["csv_sha256"], f"resume CSV Batch {batch_id:03d}")
            require(sha256(Path(item["batch_npz"])) == item["batch_npz_sha256"], f"resume NPZ Batch {batch_id:03d}")
            print(f"Batch {batch_id:03d}: CHECKPOINT PASS", flush=True)
            continue
        key = f"{batch_id:03d}"; selected = batches[batch_id]
        entry = batch_entries[batch_id]
        canonical = resolve_record(entry["canonical_json"]); mapping = resolve_record(entry["mapping"])
        module = f"opentitan_hmac_sha256_msg32_faultbatch{key}"
        batch_raw = RAW / f"batch_{key}"; batch_build = BUILD / f"batch_{key}"
        batch_raw.mkdir(parents=True, exist_ok=True); batch_build.mkdir(parents=True, exist_ok=True)
        derived_json = batch_build / f"{module}_enhanced_union.json"
        derived_verilog = batch_build / f"{module}_enhanced_union.v"
        observer_payload = observer_json(canonical, module, union_bits)
        if derived_json.exists(): require(derived_json.read_bytes() == observer_payload, f"observer mismatch Batch {key}")
        else: derived_json.write_bytes(observer_payload)
        yosys_log = batch_raw / "yosys_observer.log"
        print(f"Batch {key}: DERIVE UNION OBSERVER sites={len(selected)} bits={len(union_bits)}", flush=True)
        require(run_logged([yosys, "-p", f"read_json {derived_json}; hierarchy -check -top {module}; write_verilog -noattr {derived_verilog}"], yosys_log, 900) == 0, f"Batch {key} observer generation")
        csv_path = batch_raw / f"circuitsage_hmac_v2_1_enhanced_screen_batch_{key}.csv"
        tb = batch_build / f"tb_v21_enhanced_screen_batch_{key}.sv"
        tb_payload = generate_testbench(batch_id, selected, csv_path.resolve(), key_mem.resolve(), message_mem.resolve(), digest_mem.resolve(), source_mem.resolve(), len(union_bits)).encode()
        if tb.exists(): require(tb.read_bytes() == tb_payload, f"testbench mismatch Batch {key}")
        else: tb.write_bytes(tb_payload)
        top = f"tb_v21_enhanced_screen_batch_{key}"; obj_dir = batch_build / "obj_dir"
        build_log = batch_raw / "verilator_build.log"
        print(f"Batch {key}: BUILD", flush=True)
        command = [verilator, "--binary", "--timing", "--assert", "-Wall", "-Wno-fatal", "-j", "1", "--Mdir", str(obj_dir), "--top-module", top, str(derived_verilog), str(tb)]
        require(run_logged(command, build_log, 1800) == 0, f"Batch {key} build failed")
        binary = obj_dir / f"V{top}"; require(binary.is_file() and os.access(binary, os.X_OK), f"Batch {key} binary missing")
        simulation_log = batch_raw / "simulation.log"; resource_log = batch_raw / "resources.log"
        print(f"Batch {key}: SIMULATE", flush=True)
        require(run_logged([str(binary)], simulation_log, 14400, resource_log) == 0, f"Batch {key} simulation failed")
        require("V21_ENHANCED_SCREEN_BATCH_RESULT=PASS" in simulation_log.read_text(errors="replace"), f"Batch {key} PASS token")
        arrays, summary = parse_batch_csv(csv_path, batch_id, selected, source_indices, len(union_bits), positions)
        batch_npz = PER_BATCH / f"circuitsage_hmac_v2_1_enhanced_screen_batch_{key}.npz"
        frozen_write(batch_npz, deterministic_npz(arrays))
        item = {
            "status": "PASS", "batch_id": batch_id, "completed_at": now(), "sites": len(selected),
            "canonical_json": record(canonical), "mapping": record(mapping),
            "derived_observer_json": record(derived_json), "derived_observer_verilog": record(derived_verilog),
            "testbench": record(tb), "binary": record(binary), "yosys_log": record(yosys_log),
            "build_log": record(build_log), "simulation_log": record(simulation_log), "resource_log": record(resource_log),
            "csv": str(csv_path), "csv_sha256": sha256(csv_path), "csv_bytes": csv_path.stat().st_size,
            "batch_npz": str(batch_npz), "batch_npz_sha256": sha256(batch_npz), "batch_npz_bytes": batch_npz.stat().st_size,
            "summary": summary,
        }
        state["batches"][key] = item; completed.add(batch_id)
        state["completed_batches"] = sorted(completed)
        state["completed_candidate_batch_pairs"] = [f"{name}::{value:03d}" for value in sorted(completed) for name in CANDIDATES]
        state["updated_at"] = now(); atomic_json(CHECKPOINT, state)
        print(f"Batch {key}: PASS enabled={summary['enabled_records']}", flush=True)
    require(completed == set(range(BATCHES)), "incomplete screening batches")
    freeze_dataset(state, site_rows, batches, source_indices, positions)


def freeze_dataset(state: dict[str, Any], site_rows: list[dict[str, Any]],
                   batches: dict[int, list[dict[str, Any]]], source_indices: np.ndarray,
                   positions: dict[str, list[int]]) -> None:
    print("CONSOLIDATING AND VERIFYING THREE-CANDIDATE SCREEN DATASETS", flush=True)
    common = {
        "control_timeline_delta": np.zeros((FAULTS, VECTORS, 3), dtype=np.int16),
        "cycle_delta": np.zeros((FAULTS, VECTORS), dtype=np.int16),
        "digest_xor": np.zeros((FAULTS, VECTORS, 32), dtype=np.uint8),
        "timed_out": np.zeros((FAULTS, VECTORS), dtype=np.uint8),
        "external_detected": np.zeros((FAULTS, VECTORS), dtype=np.uint8),
    }
    candidate_data: dict[str, dict[str, np.ndarray]] = {}
    baseline_data: dict[str, dict[str, np.ndarray]] = {}
    for name, spec in CANDIDATES.items():
        tag = spec["tag"]; count = spec["probe_bits"]; snapshots = spec["snapshots"]
        candidate_data[name] = {
            "probe_snapshot_xor": np.zeros((FAULTS, VECTORS, snapshots, count // 8), dtype=np.uint8),
            "probe_toggle_count": np.zeros((FAULTS, VECTORS, count), dtype=np.uint16),
            "probe_toggle_delta": np.zeros((FAULTS, VECTORS, count), dtype=np.int16),
            "probe_effect": np.zeros((FAULTS, VECTORS), dtype=np.uint8),
        }
        baseline_data[name] = {
            "probe_snapshots": np.zeros((BATCHES, VECTORS, snapshots, count // 8), dtype=np.uint8),
            "probe_toggle_count": np.zeros((BATCHES, VECTORS, count), dtype=np.uint16),
        }
    baseline_cycles = np.zeros((BATCHES, VECTORS), dtype=np.uint16)
    baseline_control = np.zeros((BATCHES, VECTORS, 3), dtype=np.int16)
    seen = np.zeros((FAULTS, VECTORS), dtype=np.uint8)
    raw_records: dict[str, Any] = {}; batch_records: dict[str, Any] = {}
    baseline_records = enabled_records = 0
    for batch_id in range(BATCHES):
        key = f"{batch_id:03d}"; item = state["batches"][key]
        csv_path = Path(item["csv"]); batch_npz = Path(item["batch_npz"])
        require(sha256(csv_path) == item["csv_sha256"], f"checkpoint/CSV Batch {key}")
        require(sha256(batch_npz) == item["batch_npz_sha256"], f"checkpoint/NPZ Batch {key}")
        # Reparse raw evidence and require byte-exact deterministic NPZ replay.
        arrays, summary = parse_batch_csv(csv_path, batch_id, batches[batch_id], source_indices,
                                          int(state["union_probe_bits"]), positions)
        require(deterministic_npz(arrays) == batch_npz.read_bytes(), f"deterministic batch replay Batch {key}")
        local_faults = arrays["fault_instance_index"]
        require(np.all(seen[local_faults] == 0), f"duplicate cross-batch fault Batch {key}")
        seen[local_faults] = 1
        for field in common:
            common[field][local_faults] = arrays[field]
        baseline_cycles[batch_id] = arrays["baseline_cycles"]
        baseline_control[batch_id] = arrays["baseline_control_timeline"]
        for name, spec in CANDIDATES.items():
            tag = spec["tag"]
            candidate_data[name]["probe_snapshot_xor"][local_faults] = arrays[f"{tag}_probe_snapshot_xor"]
            candidate_data[name]["probe_toggle_count"][local_faults] = arrays[f"{tag}_probe_toggle_count"]
            candidate_data[name]["probe_toggle_delta"][local_faults] = arrays[f"{tag}_probe_toggle_delta"]
            candidate_data[name]["probe_effect"][local_faults] = arrays[f"{tag}_probe_effect"]
            baseline_data[name]["probe_snapshots"][batch_id] = arrays[f"{tag}_baseline_probe_snapshots"]
            baseline_data[name]["probe_toggle_count"][batch_id] = arrays[f"{tag}_baseline_probe_toggle_count"]
        baseline_records += summary["baseline_records"]; enabled_records += summary["enabled_records"]
        raw_records[key] = record(csv_path); batch_records[key] = record(batch_npz)
    require(np.all(seen == 1), "missing consolidated fault/vector samples")
    require(baseline_records == BATCHES * VECTORS and enabled_records == PHYSICAL_ENABLED, "physical record totals")
    require(np.all(baseline_cycles == EXPECTED_BASELINE_CYCLES), "baseline latency replay")
    require(np.all(baseline_cycles == baseline_cycles[0:1]) and np.all(baseline_control == baseline_control[0:1]), "cross-batch control replay")
    for name in CANDIDATES:
        require(np.all(baseline_data[name]["probe_snapshots"] == baseline_data[name]["probe_snapshots"][0:1]), f"{name} baseline snapshot replay")
        require(np.all(baseline_data[name]["probe_toggle_count"] == baseline_data[name]["probe_toggle_count"][0:1]), f"{name} baseline toggle replay")

    feature_records: dict[str, Any] = {}
    metrics_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    external_fault = np.any(common["external_detected"] != 0, axis=1)
    rng = np.random.default_rng(12031203)
    bootstrap_indices = rng.integers(0, SITES, size=(BOOTSTRAP_REPLICATES, SITES), dtype=np.int32)
    for name, spec in CANDIDATES.items():
        data = candidate_data[name]
        probe_fault = np.any(data["probe_effect"] != 0, axis=1)
        combined_fault = external_fault | probe_fault
        rescued = (~external_fault) & probe_fault
        per_site = combined_fault.reshape(SITES, 2).mean(axis=1)
        bootstrap_values = per_site[bootstrap_indices].mean(axis=1)
        for replicate, value in enumerate(bootstrap_values.tolist()):
            bootstrap_rows.append({"candidate_id": name, "replicate": replicate, "combined_detection_recall": f"{value:.10f}"})
        lower, upper = np.percentile(bootstrap_values, [2.5, 97.5]).tolist()
        feature_arrays = {
            "baseline_control_timeline": baseline_control[0], "baseline_cycles": baseline_cycles[0],
            "baseline_probe_snapshots": baseline_data[name]["probe_snapshots"][0],
            "baseline_probe_toggle_count": baseline_data[name]["probe_toggle_count"][0],
            "control_timeline_delta": common["control_timeline_delta"], "cycle_delta": common["cycle_delta"],
            "digest_xor": common["digest_xor"], "external_detected": common["external_detected"],
            "probe_effect": data["probe_effect"], "probe_snapshot_xor": data["probe_snapshot_xor"],
            "probe_toggle_count": data["probe_toggle_count"], "probe_toggle_delta": data["probe_toggle_delta"],
            "source_vector_index": source_indices.astype(np.int32), "timed_out": common["timed_out"],
            "vector_rank": np.arange(VECTORS, dtype=np.int32),
        }
        payload = deterministic_npz(feature_arrays); frozen_write(FEATURES[name], payload)
        feature_records[name] = record(FEATURES[name])
        combined_recall = float(np.mean(combined_fault))
        metrics_rows.append({
            "candidate_id": name, "probe_bits": spec["probe_bits"], "snapshots": spec["snapshots"],
            "external_observable_faults": int(np.sum(external_fault)),
            "probe_observable_faults": int(np.sum(probe_fault)),
            "probe_rescued_faults": int(np.sum(rescued)),
            "combined_observable_faults": int(np.sum(combined_fault)),
            "combined_detection_recall": f"{combined_recall:.8f}",
            "site_bootstrap_ci_low": f"{lower:.8f}", "site_bootstrap_ci_high": f"{upper:.8f}",
            "fault_free_false_alarms": 0, "fault_free_baseline_records": baseline_records,
            "frozen_detection_target": f"{FROZEN_TARGET:.8f}",
            "acceptance": "PASS" if combined_recall >= FROZEN_TARGET else "NOT_MET",
        })
    target_arrays = {
        "fault_instance_index": np.arange(FAULTS, dtype=np.int32),
        "screen_rank": np.repeat(np.arange(SITES, dtype=np.int32), 2),
        "site_index": np.repeat(np.asarray([int(row["site_index"]) for row in site_rows], dtype=np.int32), 2),
        "stuck_value": np.tile(np.asarray([0, 1], dtype=np.uint8), SITES),
    }
    target_payload = deterministic_npz(target_arrays); frozen_write(TARGETS, target_payload)
    metrics_payload = csv_payload(metrics_rows); bootstrap_payload = csv_payload(bootstrap_rows)
    frozen_write(METRICS_CSV, metrics_payload); frozen_write(BOOTSTRAP, bootstrap_payload)
    passing = [row["candidate_id"] for row in metrics_rows if row["acceptance"] == "PASS"]
    metrics_json = {
        "metrics_version": "CIRCUITSAGE-HMAC-V2.1-ENHANCED-SCREENING-METRICS-12B3D-v1",
        "stage": STAGE, "status": "FROZEN", "scope": "REPAIR_TRAIN BOUNDED SCREEN ONLY",
        "screen_sites": SITES, "fault_instances": FAULTS, "vectors": VECTORS,
        "physical_enabled_simulations": PHYSICAL_ENABLED,
        "logical_candidate_evaluations": LOGICAL_ENABLED_TOTAL,
        "passive_union_observer_equivalence": "ONE TRANSACTION; THREE NON-PERTURBING FROZEN PROJECTIONS",
        "frozen_detection_target": FROZEN_TARGET, "fault_free_false_alarm_target": 0.0,
        "site_bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "candidate_metrics": metrics_rows, "passing_candidates": passing,
        "maximum_advancing_candidates": 1,
        "advancement_selection": "NOT PERFORMED; SEPARATE DISPOSITION REQUIRED",
        "interpretation": "REPAIR_TRAIN SCREENING FEASIBILITY; NOT INDEPENDENT GENERALIZATION ACCURACY",
    }
    frozen_write(METRICS_JSON, canonical_json(metrics_json))
    schema = {
        "schema_version": "CIRCUITSAGE-HMAC-V2.1-ENHANCED-SCREENING-DATASET-12B3D-v1",
        "stage": STAGE, "status": "FROZEN", "format": "THREE DETERMINISTIC FEATURE NPZ + ONE TARGET NPZ",
        "candidate_features": feature_records,
        "targets": {name: {"dtype": str(value.dtype), "shape": list(value.shape)} for name, value in sorted(target_arrays.items())},
        "sample_key": ["fault_instance_index", "vector_rank"],
        "union_fixed_cycles_after_start": UNION_CYCLES, "final_union_snapshot": "DONE_OR_TIMEOUT",
        "candidate_union_slots": CANDIDATE_SLOTS,
        "snapshot_representation": "PACKED-LITTLE-BIT-ORDER ENABLED XOR MATCHED GOLDEN",
        "toggle_representation": "UINT16 COUNT PLUS SIGNED DELTA FROM MATCHED GOLDEN",
        "identity_exclusion": "site/fault/stuck/selector absent from all feature NPZ files",
        "identity_location": "SEPARATE TARGET NPZ FOR SCORING ONLY",
        "source_partition": "REPAIR_TRAIN BOUNDED SCREEN ONLY",
        "model_training_or_inference": False, "repair_site_test_rows": 0,
        "original_dev_site_test_rows": 0, "validation_rows": 0, "holdout_rows": 0,
    }
    frozen_write(SCHEMA, canonical_json(schema))
    state["status"] = "PASS"; state["finished_at"] = now(); state["updated_at"] = now(); atomic_json(CHECKPOINT, state)
    manifest = {
        "manifest_version": "CIRCUITSAGE-HMAC-V2.1-ENHANCED-SCREENING-DATASET-MANIFEST-12B3D-v1",
        "stage": STAGE, "status": "PASS", "authorization": record(AUTHORIZATION),
        "checkpoint": record(CHECKPOINT), "raw_batch_csvs": raw_records, "per_batch_npz": batch_records,
        "candidate_feature_datasets": feature_records,
        "outputs": {rel(path): record(path) for path in (TARGETS, METRICS_CSV, METRICS_JSON, BOOTSTRAP, SCHEMA)},
        "canonical_batches": BATCHES, "screen_sites": SITES, "fault_instances": FAULTS,
        "selected_vectors": VECTORS, "baseline_records": baseline_records,
        "physical_enabled_records": enabled_records, "logical_candidate_evaluations": LOGICAL_ENABLED_TOTAL,
        "missing_samples": int(np.sum(seen == 0)), "duplicate_samples": 0,
        "unknown_records": 0, "baseline_failures": 0, "fault_free_false_alarms": 0,
        "model_objects_deserialized": 0, "training_calls": 0, "inference_calls": 0,
        "repair_calibration_access": 0, "repair_site_test_access": 0,
        "original_dev_site_test_access": 0, "validation_access": 0, "holdout_access": 0,
        "frozen_rtl_modified": False, "canonical_netlists_modified": False,
    }
    frozen_write(MANIFEST, canonical_json(manifest))
    audit = {
        "audit_version": "CIRCUITSAGE-HMAC-V2.1-ENHANCED-SCREENING-EXECUTION-DATASET-FREEZE-12B3D-v1",
        "stage": STAGE, "status": "PASS", "screening_execution": "COMPLETED / FROZEN",
        "dataset_status": "FROZEN", "batches_verified": f"{BATCHES}/{BATCHES}",
        "candidate_batch_pairs": f"{BATCHES * 3}/{BATCHES * 3}",
        "screen_sites_faults_vectors": [SITES, FAULTS, VECTORS],
        "physical_enabled_simulations": PHYSICAL_ENABLED,
        "logical_candidate_evaluations": LOGICAL_ENABLED_TOTAL,
        "candidate_results": {row["candidate_id"]: {
            "combined_observable_faults": row["combined_observable_faults"],
            "combined_detection_recall": float(row["combined_detection_recall"]),
            "site_bootstrap_ci": [float(row["site_bootstrap_ci_low"]), float(row["site_bootstrap_ci_high"])],
            "fault_free_false_alarms": 0, "acceptance": row["acceptance"],
        } for row in metrics_rows},
        "passing_candidates": passing, "advancement_selection": "NOT PERFORMED",
        "missing_duplicate_unknown_baseline_failures": [0, 0, 0, 0],
        "deterministic_dataset_replay": "PASS / EXACT",
        "identity_fields_in_feature_npz": 0, "fault_selector_value_raw_as_features": "PROHIBITED / ABSENT",
        "model_training_inference": "0 / 0", "repair_calibration": "LOCKED / NOT ACCESSED",
        "repair_site_test": "LOCKED / NOT ACCESSED", "original_dev_site_test": "CONSUMED / NOT REOPENED",
        "validation_access": 0, "holdout_access": 0, "frozen_rtl_modified": False,
        "canonical_netlists_modified": False, "metrics": record(METRICS_JSON), "manifest": record(MANIFEST),
        "next_gate": "STAGE 12B-3E — ENHANCED MEASUREMENT SCREENING DISPOSITION AND FULL-CAPTURE READINESS FREEZE",
    }
    frozen_write(AUDIT, canonical_json(audit))
    require(target_payload == TARGETS.read_bytes() and metrics_payload == METRICS_CSV.read_bytes(), "final dataset replay")
    for path in (METRICS_JSON, SCHEMA, MANIFEST, AUDIT):
        require(canonical_json(load_json(path)) == path.read_bytes(), f"canonical JSON replay: {path.name}")
    for path, expected in PINNED.items():
        require(sha256(path) == expected, f"frozen input changed: {rel(path)}")

    print("\nSTAGE 12B-3D — ENHANCED MEASUREMENT BOUNDED-SCREENING EXECUTION AND DATASET FREEZE")
    print(f"{'Status':<66}: PASS")
    print(f"{'Screening execution / dataset':<66}: COMPLETED / FROZEN")
    print(f"{'Batches / candidate-batch pairs':<66}: {BATCHES}/{BATCHES} / {BATCHES * 3}/{BATCHES * 3}")
    print(f"{'Screen sites / faults / vectors':<66}: {SITES} / {FAULTS} / {VECTORS}")
    print(f"{'Physical simulations / logical candidate evaluations':<66}: {PHYSICAL_ENABLED} / {LOGICAL_ENABLED_TOTAL}")
    for row in metrics_rows:
        print(f"{row['candidate_id']:<66}: recall={row['combined_detection_recall']} rescued={row['probe_rescued_faults']} acceptance={row['acceptance']}")
    print(f"{'Passing candidates':<66}: {passing if passing else 'NONE'}")
    print(f"{'Fault-free false alarms':<66}: 0")
    print(f"{'Missing / duplicate / unknown / baseline failures':<66}: 0 / 0 / 0 / 0")
    print(f"{'Model training / inference':<66}: 0 / 0")
    print(f"{'REPAIR_SITE_TEST / DEV_SITE_TEST / VALIDATION / HOLDOUT':<66}: 0 / 0 / 0 / 0")
    print(f"{'Metrics':<66}: {METRICS_JSON}")
    print(f"{'Metrics SHA':<66}: {sha256(METRICS_JSON)}")
    print(f"{'Manifest':<66}: {MANIFEST}")
    print(f"{'Manifest SHA':<66}: {sha256(MANIFEST)}")
    print(f"{'Audit':<66}: {AUDIT}")
    print(f"{'Audit SHA':<66}: {sha256(AUDIT)}")
    print(f"{'Next gate':<66}: STAGE 12B-3E — ENHANCED MEASUREMENT SCREENING DISPOSITION AND FULL-CAPTURE READINESS FREEZE")


def status() -> None:
    print("STAGE 12B-3D — ENHANCED-SCREENING STATUS")
    if AUDIT.is_file():
        audit = load_json(AUDIT)
        print(f"Status                    : {audit.get('status')} / FROZEN")
        print(f"Batches verified          : {audit.get('batches_verified')}")
        print(f"Candidate-batch pairs     : {audit.get('candidate_batch_pairs')}")
        for name, result in audit.get("candidate_results", {}).items():
            print(f"{name:<26}: recall={result.get('combined_detection_recall')} acceptance={result.get('acceptance')}")
        print(f"Audit                     : {AUDIT}")
        print(f"Audit SHA                 : {sha256(AUDIT)}")
        return
    if not CHECKPOINT.is_file():
        print("Status                    : NOT STARTED")
        return
    state = load_json(CHECKPOINT); completed = state.get("completed_batches", [])
    print(f"Status                    : {state.get('status')}")
    print(f"Completed batches         : {len(completed)}/{BATCHES}")
    print(f"Candidate-batch pairs     : {len(state.get('completed_candidate_batch_pairs', []))}/{BATCHES * 3}")
    if completed: print(f"Latest completed batch    : {max(completed):03d}")
    print(f"Checkpoint                : {CHECKPOINT}")
    print(f"Checkpoint SHA            : {sha256(CHECKPOINT)}")


def self_test() -> None:
    require(len(UNION_CYCLES) + 1 == 32 and UNION_CYCLES == sorted(set(UNION_CYCLES)), "union schedule")
    require({name: len(slots) for name, slots in CANDIDATE_SLOTS.items()} == {name: spec["snapshots"] for name, spec in CANDIDATES.items()}, "candidate schedules")
    bits = snapshot_bits("1f", 5); require(bits.tolist() == [1, 1, 1, 1, 1], "snapshot parser")
    require(LOGICAL_ENABLED_TOTAL == 73728 and PHYSICAL_ENABLED == 24576, "transaction counts")
    arrays = {"a": np.arange(8, dtype=np.uint8), "b": np.arange(3, dtype=np.int32)}
    require(deterministic_npz(arrays) == deterministic_npz(arrays), "deterministic NPZ")
    print("Stage 12B-3D self-test: PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--status", action="store_true")
    mode.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test: self_test()
    elif args.status: status()
    else:
        require(ROOT.name == "vlsi_fault_detection_v2", f"run from project root, not {ROOT}")
        execute(resume=args.resume)


if __name__ == "__main__":
    main()
