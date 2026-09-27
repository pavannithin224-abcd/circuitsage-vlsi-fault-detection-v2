#!/usr/bin/env python3
"""Stage 12B-2F: simulation-only probe discovery and feasibility freeze.

Selects one fixed 64-bit topology-only probe bank from the frozen golden graph,
verifies that every selected original net bit exists in all 45 canonical fault
batch JSON netlists, creates one non-invasive copied Batch-000 observer canary,
and runs Yosys/Verilator structural lint. It never executes a simulation or
captures a response, and it never modifies frozen RTL or canonical netlists.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


STAGE = "12B-2F"
ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "config/v2_1_improvement"
RESULT = ROOT / "results/circuitsage_hmac_v2_12b2"
WORK = RESULT / "probe_discovery_12b2f"
CANARY = WORK / "observer_canary"

SOURCE_2E = ROOT / "stage_12b2e_observability_ceiling_contract.py"
REVIEW_2E = RESULT / "observability_ceiling_review_12b2e/circuitsage_hmac_v2_1_observability_ceiling_review_12b2e.md"
REGISTRY_2E = RESULT / "observability_ceiling_review_12b2e/circuitsage_hmac_v2_1_alternative_measurement_registry_12b2e.csv"
ENVIRONMENT_2E = RESULT / "observability_ceiling_review_12b2e/circuitsage_hmac_v2_1_instrumentation_environment_12b2e.json"
ARCHITECTURE_2E = CONFIG / "circuitsage_hmac_v2_1_alternative_measurement_architecture_12b2e.json"
PROBE_CONTRACT_2E = CONFIG / "circuitsage_hmac_v2_1_probe_discovery_contract_12b2e.json"
ACCEPTANCE_2E = CONFIG / "circuitsage_hmac_v2_1_instrumentation_feasibility_acceptance_12b2e.json"
MANIFEST_2E = RESULT / "circuitsage_hmac_v2_1_observability_ceiling_manifest_12b2e.json"
AUDIT_2E = RESULT / "circuitsage_hmac_v2_1_observability_ceiling_alternative_measurement_freeze_12b2e.json"

GRAPH_ROOT = ROOT / "results/hmac_fault_campaign_11d1/graph_dataset_11d1a"
GRAPH_MANIFEST = ROOT / "results/hmac_fault_campaign_11d1/hmac_golden_netlist_graph_manifest_11d1a.json"
NODE_TABLE = GRAPH_ROOT / "hmac_golden_netlist_graph_nodes_11d1a.csv"
GENERATION_INDEX = ROOT / "results/hmac_fault_campaign_11c4/hmac_canonical_batch_generation_index_11c4g_a1.json"
SEMANTIC_FREEZE = ROOT / "results/hmac_fault_campaign_11c4/hmac_all_batch_semantic_manifest_freeze_11c4g_a2.json"

PROBE_BANK_CSV = WORK / "circuitsage_hmac_v2_1_global_probe_bank_12b2f.csv"
PROBE_BANK_JSON = WORK / "circuitsage_hmac_v2_1_global_probe_bank_12b2f.json"
CONSISTENCY = WORK / "circuitsage_hmac_v2_1_cross_batch_probe_consistency_12b2f.json"
CANARY_JSON = CANARY / "opentitan_hmac_sha256_msg32_faultbatch000_probe_canary.json"
CANARY_VERILOG = CANARY / "opentitan_hmac_sha256_msg32_faultbatch000_probe_canary.v"
CANARY_TB = CANARY / "tb_v21_probe_canary.sv"
YOSYS_LOG = CANARY / "yosys_observer_canary.log"
VERILATOR_LOG = CANARY / "verilator_observer_lint.log"
CAPTURE_CONTRACT = CONFIG / "circuitsage_hmac_v2_1_probe_capture_contract_12b2f.json"
MANIFEST = RESULT / "circuitsage_hmac_v2_1_probe_discovery_manifest_12b2f.json"
AUDIT = RESULT / "circuitsage_hmac_v2_1_probe_discovery_instrumentation_feasibility_freeze_12b2f.json"

PINNED = {
    SOURCE_2E: "f0f4e38c0278c1c1e4a583f496b8eefc1716c95c1b9215feffc8e203b339f4df",
    REVIEW_2E: "fa7da13382b0d40b2d3bb092d4dfe833548d02d8a7fef5f0427a626b5e664327",
    REGISTRY_2E: "c23f4e606e88075640f4f2afdb5ad19f377e28536a85f0e56f427de8487e70ea",
    ENVIRONMENT_2E: "eb6cc18f97ae9141304fdaa15ac15457bef8ad523ae04b04ef3ff119dcf6709e",
    ARCHITECTURE_2E: "44766190f8ee24bbe8e315e96efc2a86f93c64b36657da2ca94a56408ceb2f0d",
    PROBE_CONTRACT_2E: "f54df9f04ef70391893c01c9804f854abd87d92987a751b9a467106bde63987a",
    ACCEPTANCE_2E: "824e3a65ad38a63c3b5a59219ad938976ee3c73b72b2ef4f6095416e1edc0483",
    MANIFEST_2E: "1059332439ef13d634fb655a8379dfd0d04a392493ec7632b17aacbe8819e6a8",
    AUDIT_2E: "5165e617f75717233b5b6619861071e6c6b0a848fce35fc69309ac06b573fef1",
    GRAPH_MANIFEST: "b719e33941460ce9c0186c29c18f4585adf044ae3afd0493ad40214f42b9828c",
    GENERATION_INDEX: "7ab1f5bac5fa9d635dfc04a6bd52a826d4dbd6c622bcc9c03186921c6371cd6a",
    SEMANTIC_FREEZE: "a586b8ce233f10e10d3d7db052fe249ae8a570165c0ee87f8eb4f507ca1ca7f0",
}

PROBE_BITS = 64
STATE_PROBES = 32
COMBINATIONAL_PROBES = 32
BATCHES = 45
ORIGINAL_CELLS = 22839
FORBIDDEN_PATTERN = re.compile(r"fault|inject|selector|decode|rawmux|site_raw", re.IGNORECASE)


def stop(message: str) -> None:
    raise SystemExit(f"STOP: {message}")


def require(condition: bool, message: str) -> None:
    if not condition:
        stop(message)


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


def verify_inputs() -> None:
    print("STAGE 12B-2F — SIMULATION-ONLY PROBE DISCOVERY AND INSTRUMENTATION FEASIBILITY")
    print("FROZEN INPUT VERIFICATION")
    for path, expected in PINNED.items():
        require(path.is_file(), f"missing frozen input: {rel(path)}")
        require(sha256(path) == expected, f"frozen input SHA: {rel(path)}")
        print(f"  {path.name:<91}: OK", flush=True)
    audit = load_json(AUDIT_2E)
    contract = load_json(PROBE_CONTRACT_2E)
    architecture = load_json(ARCHITECTURE_2E)
    require(audit.get("status") == "PASS", "12B-2E status")
    require(audit.get("probe_discovery") == "AUTHORIZED / NOT STARTED", "probe-discovery authorization")
    require(audit.get("probe_capture") == "NOT AUTHORIZED", "capture boundary")
    require(audit.get("model_training") == "NOT AUTHORIZED", "training boundary")
    require(audit.get("fault_selector_value_raw_as_features") == "PROHIBITED", "leakage boundary")
    require(audit.get("repair_site_test") == "LOCKED / NOT AUTHORIZED", "repair test boundary")
    require(audit.get("validation_access") == 0 and audit.get("holdout_access") == 0, "protected partition access")
    require(contract.get("authorization") == "STATIC PROBE DISCOVERY AND TESTBENCH-LINT ONLY", "discovery scope")
    require(contract.get("probe_discovery_limit_bits") == PROBE_BITS, "probe budget")
    require(contract.get("next_capture_requires_separate_authorization") is True, "capture authorization boundary")
    require(architecture.get("probe_selection") == "ONE GLOBAL FROZEN SET; NEVER CONDITIONED ON QUERY FAULT IDENTITY", "global-probe rule")
    yosys = shutil.which("yosys")
    verilator = shutil.which("verilator")
    require(yosys is not None, "yosys is not in PATH")
    require(verilator is not None, "verilator is not in PATH")
    print("  Authorization, leakage controls, protected partitions and toolchain                       : PASS")


def graph_nodes() -> list[dict[str, str]]:
    manifest = load_json(GRAPH_MANIFEST)
    require(manifest.get("status") == "PASS", "graph manifest status")
    outputs = manifest.get("outputs")
    require(isinstance(outputs, dict) and isinstance(outputs.get("node_table"), dict), "graph node-table record")
    verify_record(outputs["node_table"], NODE_TABLE, "graph node table")
    with NODE_TABLE.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    require(len(rows) == ORIGINAL_CELLS, "graph node count")
    require([int(row["node_index"]) for row in rows] == list(range(ORIGINAL_CELLS)), "graph node ordering")
    require(len({int(row["bit_id"]) for row in rows}) == ORIGINAL_CELLS, "graph bit-ID uniqueness")
    return rows


def score(row: dict[str, str]) -> tuple[int, int, int, int, int]:
    return (
        int(row["out_degree_branches"]),
        int(row["cell_fanout"]),
        int(row["out_neighbor_count"]),
        int(row["in_degree_branches"]),
        -int(row["site_index"]),
    )


def select_probes(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    clean = [row for row in rows if not FORBIDDEN_PATTERN.search("|".join([
        row["site_id"], row["driver_cell"], row["net_name"], row["output_port"],
    ]))]
    state = sorted((row for row in clean if row["site_category"] == "SEQUENTIAL_STATE_STEM"), key=score, reverse=True)
    combinational = sorted((row for row in clean if row["site_category"] == "COMBINATIONAL_LOGIC_STEM"), key=score, reverse=True)
    require(len(state) >= STATE_PROBES and len(combinational) >= COMBINATIONAL_PROBES, "probe candidate classes")
    chosen = [(row, "HIGH_FANOUT_SEQUENTIAL") for row in state[:STATE_PROBES]]
    chosen += [(row, "HIGH_FANOUT_COMBINATIONAL") for row in combinational[:COMBINATIONAL_PROBES]]
    chosen.sort(key=lambda item: (item[1], -score(item[0])[0], int(item[0]["site_index"])))
    probes: list[dict[str, Any]] = []
    for probe_bit, (row, rationale) in enumerate(chosen):
        require(not FORBIDDEN_PATTERN.search("|".join(row.values())), f"forbidden probe candidate: {row['site_id']}")
        probes.append({
            "probe_bit": probe_bit,
            "site_id": row["site_id"],
            "site_index": int(row["site_index"]),
            "node_index": int(row["node_index"]),
            "bit_id": int(row["bit_id"]),
            "driver_cell": row["driver_cell"],
            "driver_cell_type": row["driver_cell_type"],
            "site_category": row["site_category"],
            "net_name": row["net_name"],
            "cell_fanout": int(row["cell_fanout"]),
            "in_degree_branches": int(row["in_degree_branches"]),
            "out_degree_branches": int(row["out_degree_branches"]),
            "selection_rationale": rationale,
        })
    require(len(probes) == PROBE_BITS, "probe-bank width")
    require(len({probe["bit_id"] for probe in probes}) == PROBE_BITS, "probe bit uniqueness")
    require(len({probe["site_id"] for probe in probes}) == PROBE_BITS, "probe site uniqueness")
    return probes


def all_integer_bits(module: dict[str, Any]) -> set[int]:
    bits: set[int] = set()
    for port in module.get("ports", {}).values():
        bits.update(bit for bit in port.get("bits", []) if isinstance(bit, int))
    for net in module.get("netnames", {}).values():
        bits.update(bit for bit in net.get("bits", []) if isinstance(bit, int))
    for cell in module.get("cells", {}).values():
        for connection in cell.get("connections", {}).values():
            bits.update(bit for bit in connection if isinstance(bit, int))
    return bits


def batch_json_path(batch_id: int) -> Path:
    key = f"{batch_id:03d}"
    return ROOT / f"build/hmac_fault_batches_canonical_11c4g/batch_{key}/opentitan_hmac_sha256_msg32_faultbatch{key}.json"


def batch_mapping_path(batch_id: int) -> Path:
    key = f"{batch_id:03d}"
    return ROOT / f"results/hmac_fault_campaign_11c4/canonical_batches/batch_{key}/hmac_fault_batch_{key}_mapping.json"


def verify_batches(probes: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    wanted = {probe["bit_id"] for probe in probes}
    consistency: list[dict[str, Any]] = []
    canary_design: dict[str, Any] | None = None
    canary_module_name = ""
    for batch_id in range(BATCHES):
        key = f"{batch_id:03d}"
        json_path = batch_json_path(batch_id)
        mapping_path = batch_mapping_path(batch_id)
        require(json_path.is_file(), f"missing canonical Batch {key} JSON")
        require(mapping_path.is_file(), f"missing canonical Batch {key} mapping")
        design = load_json(json_path)
        module_name = f"opentitan_hmac_sha256_msg32_faultbatch{key}"
        modules = design.get("modules")
        require(isinstance(modules, dict) and set(modules) == {module_name}, f"Batch {key} module set")
        module = modules[module_name]
        require(isinstance(module, dict), f"Batch {key} module object")
        ports = module.get("ports", {})
        required_ports = {"clk_i", "rst_ni", "start_i", "key_i", "message_i", "busy_o", "done_o", "digest_o", "fault_enable_i", "fault_selector_i", "fault_value_i", "fault_raw_o"}
        require(required_ports <= set(ports), f"Batch {key} port contract")
        present = wanted & all_integer_bits(module)
        require(present == wanted, f"Batch {key} missing global probe bits: {len(wanted - present)}")
        cells = module.get("cells", {})
        require(isinstance(cells, dict) and len(cells) >= ORIGINAL_CELLS, f"Batch {key} cell count")
        mapping = load_json(mapping_path)
        expected_sites = 311 if batch_id == 44 else 512
        require(mapping.get("valid_site_count") == expected_sites, f"Batch {key} mapping count")
        consistency.append({
            "batch_id": batch_id, "module": module_name,
            "canonical_json": record(json_path), "mapping": record(mapping_path),
            "cells": len(cells), "probe_bits_present": len(present),
            "probe_bank_exact": True,
        })
        if batch_id == 0:
            canary_design = copy.deepcopy(design)
            canary_module_name = module_name
    require(canary_design is not None, "Batch 000 canary source")
    return consistency, canary_design, canary_module_name


def create_canary(design: dict[str, Any], module_name: str, probes: list[dict[str, Any]]) -> bytes:
    module = design["modules"][module_name]
    require("probe_o" not in module["ports"] and "probe_o" not in module["netnames"], "probe_o name collision")
    bits = [probe["bit_id"] for probe in probes]
    module["ports"]["probe_o"] = {"direction": "output", "bits": bits}
    module["netnames"]["probe_o"] = {"hide_name": 0, "bits": bits, "attributes": {}}
    return canonical_json(design)


def run_logged(command: list[str], log: Path, timeout: int) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as stream:
        try:
            return subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=timeout, check=False).returncode
        except subprocess.TimeoutExpired:
            stream.write(f"\nTIMEOUT_SECONDS={timeout}\n")
            return 124


def testbench_text(module_name: str) -> str:
    return f'''`timescale 1ns/1ps
module tb_v21_probe_canary;
  logic clk_i=0,rst_ni=0,start_i=0,fault_enable_i=0,fault_value_i=0;
  logic [8:0] fault_selector_i='0;
  logic [255:0] key_i='0,message_i='0;
  wire busy_o,done_o,fault_raw_o;
  wire [255:0] digest_o;
  wire [{PROBE_BITS - 1}:0] probe_o;
  {module_name} dut(
    .clk_i(clk_i),.rst_ni(rst_ni),.start_i(start_i),.key_i(key_i),.message_i(message_i),
    .busy_o(busy_o),.done_o(done_o),.digest_o(digest_o),
    .fault_enable_i(fault_enable_i),.fault_selector_i(fault_selector_i),
    .fault_value_i(fault_value_i),.fault_raw_o(fault_raw_o),.probe_o(probe_o));
  always #5 clk_i=~clk_i;
  initial begin #20; $finish; end
endmodule
'''


def self_test() -> None:
    sample = [
        {"site_id": "HMAC-STEM-000001", "driver_cell": "cell", "net_name": "n", "output_port": "Y"},
        {"site_id": "HMAC-STEM-000002", "driver_cell": "$fault$inject", "net_name": "n", "output_port": "Y"},
    ]
    require(FORBIDDEN_PATTERN.search("|".join(sample[0].values())) is None, "clean-name canary")
    require(FORBIDDEN_PATTERN.search("|".join(sample[1].values())) is not None, "forbidden-name canary")
    print("Stage 12B-2F self-test: PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    require(ROOT.name == "vlsi_fault_detection_v2", f"run from project root, not {ROOT}")
    outputs = (PROBE_BANK_CSV, PROBE_BANK_JSON, CONSISTENCY, CANARY_JSON,
               CANARY_VERILOG, CANARY_TB, YOSYS_LOG, VERILATOR_LOG,
               CAPTURE_CONTRACT, MANIFEST, AUDIT)
    for path in outputs:
        require(not path.exists(), f"Stage {STAGE} output already exists: {rel(path)}")

    verify_inputs()
    rows = graph_nodes()
    probes = select_probes(rows)
    consistency_rows, canary_design, canary_module = verify_batches(probes)
    probe_fields = list(probes[0])
    csv_output = io.StringIO(newline="")
    writer = csv.DictWriter(csv_output, fieldnames=probe_fields, lineterminator="\n")
    writer.writeheader(); writer.writerows(probes)
    probe_csv_payload = csv_output.getvalue().encode()
    probe_json = {
        "probe_bank_version": "CIRCUITSAGE-HMAC-V2.1-GLOBAL-PROBE-BANK-12B2F-v1",
        "stage": STAGE, "status": "FROZEN", "probe_bits": PROBE_BITS,
        "selection_source": "FROZEN GOLDEN GRAPH TOPOLOGY ONLY",
        "selection_policy": {
            "sequential_state_probes": STATE_PROBES,
            "combinational_probes": COMBINATIONAL_PROBES,
            "ranking": ["out_degree_branches", "cell_fanout", "out_neighbor_count", "in_degree_branches", "lower site_index"],
            "partition_labels_used": False, "pilot_response_values_used": False,
            "fault_identity_conditioning": False,
        },
        "forbidden_names_regex": FORBIDDEN_PATTERN.pattern,
        "probe_site_ids": [probe["site_id"] for probe in probes],
        "probe_bit_ids": [probe["bit_id"] for probe in probes],
        "probe_csv_sha256": hashlib.sha256(probe_csv_payload).hexdigest(),
    }
    consistency = {
        "consistency_version": "CIRCUITSAGE-HMAC-V2.1-CROSS-BATCH-PROBE-CONSISTENCY-v1",
        "stage": STAGE, "status": "PASS", "canonical_batches": BATCHES,
        "probe_bits": PROBE_BITS, "all_batches_exact": True,
        "canonical_sources_modified": False, "batches": consistency_rows,
    }
    canary_payload = create_canary(canary_design, canary_module, probes)
    frozen_write(PROBE_BANK_CSV, probe_csv_payload)
    frozen_write(PROBE_BANK_JSON, canonical_json(probe_json))
    frozen_write(CONSISTENCY, canonical_json(consistency))
    frozen_write(CANARY_JSON, canary_payload)
    frozen_write(CANARY_TB, testbench_text(canary_module).encode())

    yosys = shutil.which("yosys")
    verilator = shutil.which("verilator")
    require(yosys is not None and verilator is not None, "toolchain unavailable")
    yosys_command = [
        yosys, "-p",
        f"read_json {CANARY_JSON}; hierarchy -check -top {canary_module}; write_verilog -noattr {CANARY_VERILOG}",
    ]
    print("Observer canary: YOSYS STRUCTURAL CHECK", flush=True)
    require(run_logged(yosys_command, YOSYS_LOG, 600) == 0, "observer canary Yosys check")
    require(CANARY_VERILOG.is_file() and CANARY_VERILOG.stat().st_size > 0, "observer canary Verilog")
    print("Observer canary: VERILATOR LINT", flush=True)
    verilator_command = [
        verilator, "--lint-only", "--timing", "-Wall", "-Wno-fatal",
        "--top-module", "tb_v21_probe_canary", str(CANARY_VERILOG), str(CANARY_TB),
    ]
    require(run_logged(verilator_command, VERILATOR_LOG, 600) == 0, "observer canary Verilator lint")

    capture_contract = {
        "contract_version": "CIRCUITSAGE-HMAC-V2.1-PROBE-CAPTURE-12B2F-v1",
        "stage": STAGE, "status": "FROZEN",
        "global_probe_bank": record(PROBE_BANK_JSON),
        "probe_bank_width": PROBE_BITS,
        "capture_authorization": "NOT YET AUTHORIZED",
        "allowed_future_scope": "REPAIR_TRAIN PILOT SITES ONLY",
        "planned_measurements": ["M1_CONTROL_TIMELINE", "M2_ARCH_STATE_SKETCH"],
        "fixed_event_schedule": ["transaction_start", "message_drain", "compression_boundaries", "transaction_done"],
        "maximum_snapshots_per_transaction": 16,
        "probe_representation": "DOMAIN-SEPARATED HASHED SNAPSHOTS PLUS TOGGLE COUNTS",
        "forbidden_features": ["fault_selector_i", "fault_enable_i", "fault_value_i", "fault_raw_o", "site ID", "fault instance ID"],
        "observer_implementation": "DERIVED SIMULATION NETLIST COPY; FROZEN SOURCES UNCHANGED",
        "model_training": "NOT AUTHORIZED",
        "repair_calibration": "LOCKED", "repair_site_test": "LOCKED",
        "original_dev_site_test": "CONSUMED / REOPENING PROHIBITED",
        "validation": "PROHIBITED", "holdout": "PROHIBITED",
        "next_stage_requires_separate_authorization": True,
    }
    frozen_write(CAPTURE_CONTRACT, canonical_json(capture_contract))
    primary = (PROBE_BANK_CSV, PROBE_BANK_JSON, CONSISTENCY, CANARY_JSON,
               CANARY_VERILOG, CANARY_TB, YOSYS_LOG, VERILATOR_LOG, CAPTURE_CONTRACT)
    manifest = {
        "manifest_version": "CIRCUITSAGE-HMAC-V2.1-PROBE-DISCOVERY-MANIFEST-v1",
        "stage": STAGE, "status": "PASS", "stage_12b2e_audit": record(AUDIT_2E),
        "outputs": {rel(path): record(path) for path in primary},
        "probe_bits": PROBE_BITS, "canonical_batches_checked": BATCHES,
        "cross_batch_probe_consistency": "PASS / EXACT",
        "yosys_structural_check": "PASS", "verilator_lint": "PASS",
        "simulation_binaries_executed": 0, "response_values_captured": 0,
        "model_objects_deserialized": 0, "training_calls": 0, "inference_calls": 0,
        "repair_calibration_access": 0, "repair_site_test_access": 0,
        "original_dev_site_test_access": 0, "validation_access": 0, "holdout_access": 0,
        "frozen_rtl_modified": False, "canonical_netlists_modified": False,
    }
    frozen_write(MANIFEST, canonical_json(manifest))
    audit = {
        "audit_version": "CIRCUITSAGE-HMAC-V2.1-PROBE-DISCOVERY-INSTRUMENTATION-FEASIBILITY-FREEZE-v1",
        "stage": STAGE, "status": "PASS",
        "probe_discovery_status": "FROZEN / COMPLETE",
        "global_probe_bank": f"{PROBE_BITS} BITS / TOPOLOGY ONLY / IDENTITY INDEPENDENT",
        "sequential_combinational_probes": [STATE_PROBES, COMBINATIONAL_PROBES],
        "cross_batch_consistency": f"PASS / {BATCHES}/{BATCHES}",
        "observer_canary": "YOSYS PASS / VERILATOR LINT PASS / NOT EXECUTED",
        "probe_capture": "NOT AUTHORIZED",
        "model_training": "NOT AUTHORIZED",
        "fault_selector_value_raw_as_features": "PROHIBITED",
        "repair_site_test": "LOCKED / NOT AUTHORIZED",
        "original_dev_site_test": "CONSUMED / NOT REOPENED",
        "validation_access": 0, "holdout_access": 0,
        "frozen_rtl_modified": False, "golden_netlist_modified": False,
        "canonical_batch_netlists_modified": False,
        "manifest": record(MANIFEST),
        "next_gate": "STAGE 12B-2G — BOUNDED PROBE-CAPTURE EXECUTION AUTHORIZATION FREEZE",
    }
    frozen_write(AUDIT, canonical_json(audit))

    require(probe_csv_payload == PROBE_BANK_CSV.read_bytes(), "probe CSV replay")
    require(canary_payload == CANARY_JSON.read_bytes(), "canary JSON replay")
    for path in (PROBE_BANK_JSON, CONSISTENCY, CAPTURE_CONTRACT, MANIFEST, AUDIT):
        require(canonical_json(load_json(path)) == path.read_bytes(), f"canonical JSON replay: {path.name}")
    for path, expected in PINNED.items():
        require(sha256(path) == expected, f"frozen input changed during stage: {rel(path)}")

    print("\nSTAGE 12B-2F — SIMULATION-ONLY PROBE DISCOVERY AND INSTRUMENTATION FEASIBILITY FREEZE")
    print(f"{'Status':<56}: PASS")
    print(f"{'Probe discovery status':<56}: FROZEN / COMPLETE")
    print(f"{'Global probe bank':<56}: {PROBE_BITS} BITS")
    print(f"{'Sequential / combinational probes':<56}: {STATE_PROBES} / {COMBINATIONAL_PROBES}")
    print(f"{'Selection inputs':<56}: GOLDEN GRAPH TOPOLOGY ONLY")
    print(f"{'Fault identity-conditioned selection':<56}: NO")
    print(f"{'Cross-batch consistency':<56}: PASS / {BATCHES}/{BATCHES}")
    print(f"{'Yosys structural check / Verilator lint':<56}: PASS / PASS")
    print(f"{'Simulation executed / response values captured':<56}: 0 / 0")
    print(f"{'Probe capture / model training':<56}: NOT AUTHORIZED / NOT AUTHORIZED")
    print(f"{'Fault selector/value/raw diagnostic features':<56}: PROHIBITED")
    print(f"{'REPAIR_SITE_TEST / original DEV_SITE_TEST':<56}: LOCKED / CONSUMED")
    print(f"{'VALIDATION / HOLDOUT access':<56}: 0 / 0")
    print(f"{'Frozen RTL / canonical netlists modified':<56}: NO / NO")
    print(f"{'Probe bank':<56}: {PROBE_BANK_CSV}")
    print(f"{'Probe bank SHA':<56}: {sha256(PROBE_BANK_CSV)}")
    print(f"{'Consistency':<56}: {CONSISTENCY}")
    print(f"{'Consistency SHA':<56}: {sha256(CONSISTENCY)}")
    print(f"{'Capture contract':<56}: {CAPTURE_CONTRACT}")
    print(f"{'Capture contract SHA':<56}: {sha256(CAPTURE_CONTRACT)}")
    print(f"{'Manifest':<56}: {MANIFEST}")
    print(f"{'Manifest SHA':<56}: {sha256(MANIFEST)}")
    print(f"{'Audit':<56}: {AUDIT}")
    print(f"{'Audit SHA':<56}: {sha256(AUDIT)}")
    print(f"{'Next gate':<56}: STAGE 12B-2G — BOUNDED PROBE-CAPTURE EXECUTION AUTHORIZATION FREEZE")


if __name__ == "__main__":
    main()
