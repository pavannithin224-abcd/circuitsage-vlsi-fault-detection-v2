#!/usr/bin/env python3
"""Stage 12B-3B: enhanced probe-bank discovery and structural freeze.

Builds three fixed, identity-independent global probe candidates from the
frozen golden graph and aggregated REPAIR_TRAIN observability evidence.  It
verifies every selected bit in all 45 canonical fault-batch netlists and runs
Yosys/Verilator checks on isolated Batch-000 observer canaries.  It performs no
simulation, response capture, model operation, or protected-partition access.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


STAGE = "12B-3B"
ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "config/v2_1_improvement"
RESULT_2 = ROOT / "results/circuitsage_hmac_v2_12b2"
RESULT = ROOT / "results/circuitsage_hmac_v2_12b3"
WORK = RESULT / "enhanced_probe_discovery_12b3b"
CANARY_ROOT = WORK / "observer_canaries"

SOURCE_3A = ROOT / "stage_12b3a_enhanced_measurement_architecture.py"
ARCHITECTURE_3A = CONFIG / "circuitsage_hmac_v2_1_enhanced_measurement_architecture_12b3a.json"
BUDGET_3A = CONFIG / "circuitsage_hmac_v2_1_enhanced_measurement_budget_policy_12b3a.json"
ACCEPTANCE_3A = CONFIG / "circuitsage_hmac_v2_1_enhanced_measurement_acceptance_contract_12b3a.json"
ARCH_WORK_3A = RESULT / "enhanced_measurement_architecture_12b3a"
CANDIDATE_GRID_3A = ARCH_WORK_3A / "circuitsage_hmac_v2_1_enhanced_measurement_candidate_grid_12b3a.csv"
SCHEDULE_3A = ARCH_WORK_3A / "circuitsage_hmac_v2_1_enhanced_measurement_screening_schedule_12b3a.csv"
REPORT_3A = ARCH_WORK_3A / "circuitsage_hmac_v2_1_enhanced_measurement_architecture_report_12b3a.md"
ENVIRONMENT_3A = ARCH_WORK_3A / "circuitsage_hmac_v2_1_enhanced_measurement_environment_12b3a.json"
MANIFEST_3A = RESULT / "circuitsage_hmac_v2_1_enhanced_measurement_architecture_manifest_12b3a.json"
AUDIT_3A = RESULT / "circuitsage_hmac_v2_1_enhanced_measurement_architecture_budget_freeze_12b3a.json"

GRAPH_ROOT = ROOT / "results/hmac_fault_campaign_11d1/graph_dataset_11d1a"
GRAPH_MANIFEST = ROOT / "results/hmac_fault_campaign_11d1/hmac_golden_netlist_graph_manifest_11d1a.json"
NODE_TABLE = GRAPH_ROOT / "hmac_golden_netlist_graph_nodes_11d1a.csv"
GRAPH_NPZ = GRAPH_ROOT / "hmac_golden_netlist_graph_11d1a.npz"
CONSISTENCY_2F = RESULT_2 / "probe_discovery_12b2f/circuitsage_hmac_v2_1_cross_batch_probe_consistency_12b2f.json"
AUDIT_2H = RESULT_2 / "circuitsage_hmac_v2_1_probe_capture_dataset_integrity_freeze_12b2h.json"
FEATURES_2H = RESULT_2 / "probe_capture_12b2h/circuitsage_hmac_v2_1_probe_response_features_12b2h.npz"
TARGETS_2H = RESULT_2 / "probe_capture_12b2h/circuitsage_hmac_v2_1_probe_response_targets_12b2h.npz"

PROBE_BANK_CSV = WORK / "circuitsage_hmac_v2_1_enhanced_probe_banks_12b3b.csv"
PROBE_BANK_JSON = WORK / "circuitsage_hmac_v2_1_enhanced_probe_banks_12b3b.json"
CONSISTENCY = WORK / "circuitsage_hmac_v2_1_enhanced_cross_batch_consistency_12b3b.json"
STRUCTURAL_METRICS = WORK / "circuitsage_hmac_v2_1_enhanced_structural_metrics_12b3b.csv"
SCREENING_CONTRACT = CONFIG / "circuitsage_hmac_v2_1_enhanced_probe_screening_contract_12b3b.json"
MANIFEST = RESULT / "circuitsage_hmac_v2_1_enhanced_probe_discovery_manifest_12b3b.json"
AUDIT = RESULT / "circuitsage_hmac_v2_1_enhanced_probe_discovery_structural_freeze_12b3b.json"

PINNED = {
    SOURCE_3A: "4081bc37d9bb36cb25ece22edcaaae689f61d6c5be9aed3e14808b9b7263dbbe",
    GRAPH_MANIFEST: "b719e33941460ce9c0186c29c18f4585adf044ae3afd0493ad40214f42b9828c",
    CONSISTENCY_2F: "9d9b349ed7f615867a24e911676d626dc6c612774457a6d2b8b5422ea14d794c",
    AUDIT_2H: "cc68202cc1ef92d32319c43ca1d8afd2c9bb324ac24fb9125e630d50a7403d91",
}

BATCHES = 45
NODES = 22839
PILOT_FAULTS = 2048
INVISIBLE_FAULTS = 1021
SCREEN_SITES = 256
SCREEN_FAULTS = 512
SCREEN_VECTORS = 48
FROZEN_TARGET = 0.70
FORBIDDEN = re.compile(r"fault|inject|selector|decode|rawmux|site_raw", re.I)
CONTROL = re.compile(r"state|ctrl|control|round|valid|ready|busy|done|start|sha|msg|message|key|digest", re.I)

CANDIDATE_SPECS = {
    "EM_TOPOLOGY_4X64_T24": {"family": "MULTI_BANK_TOPOLOGY", "bits": 256, "banks": 4, "snapshots": 24},
    "EM_STATE_CHECKPOINT_2X64_T32": {"family": "SEQUENTIAL_STATE_CHECKPOINT", "bits": 128, "banks": 2, "snapshots": 32},
    "EM_TESTPOINT_4X64_T16": {"family": "SIMULATION_ONLY_TEST_POINT", "bits": 256, "banks": 4, "snapshots": 16},
}


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


def csv_payload(rows: list[dict[str, Any]]) -> bytes:
    require(bool(rows), "CSV rows")
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode()


def verify_inputs() -> list[dict[str, str]]:
    print("STAGE 12B-3B — ENHANCED PROBE-BANK DISCOVERY AND STRUCTURAL FEASIBILITY")
    print("FROZEN INPUT VERIFICATION")
    for path, expected in PINNED.items():
        require(path.is_file(), f"missing frozen input: {rel(path)}")
        require(sha256(path) == expected, f"frozen input SHA: {rel(path)}")
        print(f"  {path.name:<91}: OK", flush=True)
    for path in (ARCHITECTURE_3A, BUDGET_3A, ACCEPTANCE_3A, CANDIDATE_GRID_3A,
                 SCHEDULE_3A, REPORT_3A, ENVIRONMENT_3A, MANIFEST_3A, AUDIT_3A):
        require(path.is_file(), f"missing Stage 12B-3A output: {rel(path)}")
    audit = load_json(AUDIT_3A)
    manifest = load_json(MANIFEST_3A)
    architecture = load_json(ARCHITECTURE_3A)
    budget = load_json(BUDGET_3A)
    acceptance = load_json(ACCEPTANCE_3A)
    require(audit.get("status") == "PASS", "12B-3A status")
    require(audit.get("architecture_status") == "FROZEN" and audit.get("budget_status") == "FROZEN", "3A architecture/budget")
    require(audit.get("acceptance_status") == "FROZEN", "3A acceptance")
    require(audit.get("candidate_architectures") == 3, "3A candidate count")
    require(audit.get("maximum_banks_bits_snapshots") == [4, 64, 32], "3A maximum budget")
    require(audit.get("fault_identity_in_query") == "PROHIBITED", "identity boundary")
    require(audit.get("probe_discovery") == "NOT YET AUTHORIZED", "discovery entry state")
    require(audit.get("simulation_capture_training") == "0 / 0 / 0 — NOT AUTHORIZED", "execution boundary")
    require(audit.get("repair_site_test") == "LOCKED / NOT ACCESSED", "repair test boundary")
    require(audit.get("validation_access") == 0 and audit.get("holdout_access") == 0, "protected access")
    require(manifest.get("status") == "PASS", "3A manifest status")
    outputs = manifest.get("outputs")
    require(isinstance(outputs, dict), "3A output records")
    for path in (ARCHITECTURE_3A, BUDGET_3A, ACCEPTANCE_3A, CANDIDATE_GRID_3A,
                 SCHEDULE_3A, REPORT_3A, ENVIRONMENT_3A):
        item = outputs.get(rel(path))
        require(isinstance(item, dict), f"3A record: {path.name}")
        verify_record(item, path, f"3A {path.name}")
    require(architecture.get("candidate_ids") == list(CANDIDATE_SPECS), "candidate IDs")
    require(budget.get("maximum_candidates") == 3, "candidate budget")
    require(budget.get("screening", {}).get("maximum_total_enabled_transactions") == 73728, "screening budget")
    require(acceptance.get("screening_gate", {}).get("combined_all_injected_detection_recall_min") == FROZEN_TARGET, "frozen target")
    with CANDIDATE_GRID_3A.open(newline="", encoding="utf-8") as stream:
        grid = list(csv.DictReader(stream))
    require([row["candidate_id"] for row in grid] == list(CANDIDATE_SPECS), "candidate-grid order")
    for row in grid:
        spec = CANDIDATE_SPECS[row["candidate_id"]]
        require(int(row["concurrent_observation_bits"]) == spec["bits"], f"{row['candidate_id']} width")
        require(int(row["probe_banks"]) == spec["banks"], f"{row['candidate_id']} banks")
        require(int(row["snapshots"]) == spec["snapshots"], f"{row['candidate_id']} snapshots")
        require(row["fault_conditioned"] == "NO", f"{row['candidate_id']} identity independence")
    require(shutil.which("yosys") is not None, "yosys is not in PATH")
    require(shutil.which("verilator") is not None, "verilator is not in PATH")
    print("  Stage 12B-3A contracts, authorization boundary and toolchain                         : PASS")
    return grid


def load_graph() -> tuple[list[dict[str, str]], np.ndarray]:
    manifest = load_json(GRAPH_MANIFEST)
    require(manifest.get("status") == "PASS", "graph manifest status")
    outputs = manifest.get("outputs")
    require(isinstance(outputs, dict), "graph output records")
    verify_record(outputs["node_table"], NODE_TABLE, "graph node table")
    verify_record(outputs["graph_npz"], GRAPH_NPZ, "graph NPZ")
    with NODE_TABLE.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    require(len(rows) == NODES, "graph node count")
    require([int(row["node_index"]) for row in rows] == list(range(NODES)), "graph node order")
    with np.load(GRAPH_NPZ, allow_pickle=False) as data:
        require("edge_index" in data.files, "edge_index")
        edge_index = data["edge_index"].astype(np.int32, copy=True)
    require(edge_index.ndim == 2 and edge_index.shape[0] == 2, "edge_index shape")
    return rows, edge_index


def clean_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    clean = []
    for row in rows:
        names = "|".join(row.get(key, "") for key in ("site_id", "driver_cell", "net_name", "output_port"))
        if FORBIDDEN.search(names) is None:
            clean.append(row)
    require(len(clean) >= 256, "clean probe candidates")
    return clean


def topology_key(row: dict[str, str]) -> tuple[int, int, int, int, int, int]:
    return (int(row["out_degree_branches"]), int(row["cell_fanout"]), int(row["out_neighbor_count"]),
            int(row["in_degree_branches"]), int(row["in_neighbor_count"]), -int(row["site_index"]))


def take(pool: list[dict[str, str]], count: int, used: set[int], rationale: str) -> list[tuple[dict[str, str], str, float]]:
    chosen = []
    for row in pool:
        bit = int(row["bit_id"])
        if bit in used:
            continue
        used.add(bit)
        chosen.append((row, rationale, 0.0))
        if len(chosen) == count:
            break
    require(len(chosen) == count, f"candidate pool for {rationale}")
    return chosen


def select_topology(rows: list[dict[str, str]]) -> list[tuple[dict[str, str], str, float]]:
    clean = clean_rows(rows)
    ranked = sorted(clean, key=topology_key, reverse=True)
    state = [row for row in ranked if row["site_category"] == "SEQUENTIAL_STATE_STEM"]
    control = [row for row in ranked if CONTROL.search("|".join(row.values()))]
    combinational = [row for row in ranked if row["site_category"] == "COMBINATIONAL_LOGIC_STEM"]
    boundary = sorted(clean, key=lambda row: (int(row["is_primary_output_stem"]),) + topology_key(row), reverse=True)
    used: set[int] = set()
    selected = take(state, 64, used, "TOPOLOGY_SEQUENTIAL_STATE")
    selected += take(control + ranked, 64, used, "TOPOLOGY_CONTROL")
    selected += take(combinational + ranked, 64, used, "TOPOLOGY_HIGH_FANOUT")
    selected += take(boundary + ranked, 64, used, "TOPOLOGY_BOUNDARY")
    return selected


def select_state(rows: list[dict[str, str]]) -> list[tuple[dict[str, str], str, float]]:
    clean = clean_rows(rows)
    ranked = sorted(clean, key=topology_key, reverse=True)
    state = [row for row in ranked if row["site_category"] == "SEQUENTIAL_STATE_STEM"]
    control = [row for row in ranked if CONTROL.search("|".join(row.values()))]
    used: set[int] = set()
    selected = take(state, 96, used, "STATE_CHECKPOINT")
    selected += take(control + ranked, 32, used, "CONTROL_CHECKPOINT")
    return selected


def repair_train_scores(rows: list[dict[str, str]], edges: np.ndarray) -> tuple[np.ndarray, int]:
    audit = load_json(AUDIT_2H)
    require(audit.get("status") == "PASS", "12B-2H status")
    for key, expected in (("features", FEATURES_2H), ("targets", TARGETS_2H)):
        item = audit.get(key)
        require(isinstance(item, dict), f"12B-2H {key} record")
        verify_record(item, expected, f"12B-2H {key}")
    with np.load(FEATURES_2H, allow_pickle=False) as features:
        external = features["external_detected"].astype(np.uint8, copy=True)
    with np.load(TARGETS_2H, allow_pickle=False) as targets:
        site_index = targets["site_index"].astype(np.int32, copy=True)
        fault_index = targets["fault_instance_index"].astype(np.int32, copy=True)
    require(external.shape == (PILOT_FAULTS, 96), "REPAIR_TRAIN external-detection shape")
    require(site_index.shape == (PILOT_FAULTS,) and fault_index.tolist() == list(range(PILOT_FAULTS)), "REPAIR_TRAIN target order")
    invisible = ~np.any(external != 0, axis=1)
    require(int(np.sum(invisible)) == INVISIBLE_FAULTS, "externally invisible fault count")
    site_to_node = {int(row["site_index"]): int(row["node_index"]) for row in rows}
    adjacency: list[list[int]] = [[] for _ in range(NODES)]
    for source, destination in edges.T.tolist():
        if destination not in adjacency[source]:
            adjacency[source].append(destination)
    scores = np.zeros(NODES, dtype=np.float64)
    for site in site_index[invisible].tolist():
        source = site_to_node[int(site)]
        scores[source] += 2.0
        for one_hop in adjacency[source]:
            scores[one_hop] += 1.0
            for two_hop in adjacency[one_hop]:
                scores[two_hop] += 0.5
    return scores, int(np.sum(invisible))


def select_testpoints(rows: list[dict[str, str]], scores: np.ndarray) -> list[tuple[dict[str, str], str, float]]:
    clean = clean_rows(rows)
    ranked = sorted(clean, key=lambda row: (float(scores[int(row["node_index"])]),
                    int(row["site_category"] == "SEQUENTIAL_STATE_STEM")) + topology_key(row), reverse=True)
    used: set[int] = set()
    chosen = take(ranked, 256, used, "AGGREGATED_REPAIR_TRAIN_COVERAGE")
    return [(row, rationale, float(scores[int(row["node_index"])])) for row, rationale, _ in chosen]


def normalize_candidate(candidate_id: str, selected: list[tuple[dict[str, str], str, float]]) -> list[dict[str, Any]]:
    spec = CANDIDATE_SPECS[candidate_id]
    require(len(selected) == spec["bits"], f"{candidate_id} width")
    rows: list[dict[str, Any]] = []
    for probe_bit, (row, rationale, aggregate_score) in enumerate(selected):
        require(FORBIDDEN.search("|".join(row.values())) is None, f"forbidden probe: {row['site_id']}")
        rows.append({
            "candidate_id": candidate_id, "probe_bit": probe_bit,
            "bank_index": probe_bit // 64, "bank_bit": probe_bit % 64,
            "site_id": row["site_id"], "site_index": int(row["site_index"]),
            "node_index": int(row["node_index"]), "bit_id": int(row["bit_id"]),
            "driver_cell": row["driver_cell"], "driver_cell_type": row["driver_cell_type"],
            "site_category": row["site_category"], "net_name": row["net_name"],
            "cell_fanout": int(row["cell_fanout"]),
            "in_degree_branches": int(row["in_degree_branches"]),
            "out_degree_branches": int(row["out_degree_branches"]),
            "aggregated_repair_train_score": f"{aggregate_score:.6f}",
            "selection_rationale": rationale,
        })
    require(len({row["bit_id"] for row in rows}) == len(rows), f"{candidate_id} bit uniqueness")
    require(len({row["site_id"] for row in rows}) == len(rows), f"{candidate_id} site uniqueness")
    return rows


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


def verify_batches(candidates: dict[str, list[dict[str, Any]]]) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    prior = load_json(CONSISTENCY_2F)
    require(prior.get("status") == "PASS" and prior.get("canonical_batches") == BATCHES, "12B-2F batch registry")
    prior_batches = prior.get("batches")
    require(isinstance(prior_batches, list) and len(prior_batches) == BATCHES, "12B-2F batch records")
    union_bits = {row["bit_id"] for values in candidates.values() for row in values}
    consistency: list[dict[str, Any]] = []
    canary_design: dict[str, Any] | None = None
    canary_module = ""
    for batch_id, prior_row in enumerate(prior_batches):
        require(prior_row.get("batch_id") == batch_id, f"Batch {batch_id:03d} registry order")
        json_record = prior_row.get("canonical_json")
        mapping_record = prior_row.get("mapping")
        require(isinstance(json_record, dict) and isinstance(mapping_record, dict), f"Batch {batch_id:03d} records")
        json_path = resolve_record(json_record)
        mapping_path = resolve_record(mapping_record)
        verify_record(json_record, json_path, f"Batch {batch_id:03d} canonical JSON")
        verify_record(mapping_record, mapping_path, f"Batch {batch_id:03d} mapping")
        design = load_json(json_path)
        module_name = f"opentitan_hmac_sha256_msg32_faultbatch{batch_id:03d}"
        modules = design.get("modules")
        require(isinstance(modules, dict) and set(modules) == {module_name}, f"Batch {batch_id:03d} module set")
        module = modules[module_name]
        present = all_integer_bits(module)
        require(union_bits <= present, f"Batch {batch_id:03d} missing enhanced probe bits: {len(union_bits - present)}")
        candidate_counts = {candidate: len({row["bit_id"] for row in rows} & present) for candidate, rows in candidates.items()}
        require(all(candidate_counts[name] == CANDIDATE_SPECS[name]["bits"] for name in CANDIDATE_SPECS), f"Batch {batch_id:03d} candidate consistency")
        consistency.append({
            "batch_id": batch_id, "module": module_name,
            "canonical_json": json_record, "mapping": mapping_record,
            "candidate_probe_bits_present": candidate_counts,
            "all_candidates_exact": True,
        })
        if batch_id == 0:
            canary_design = copy.deepcopy(design)
            canary_module = module_name
    require(canary_design is not None, "Batch 000 canary source")
    return consistency, canary_design, canary_module


def create_canary(design: dict[str, Any], module_name: str, rows: list[dict[str, Any]]) -> bytes:
    copied = copy.deepcopy(design)
    module = copied["modules"][module_name]
    require("probe_o" not in module["ports"] and "probe_o" not in module["netnames"], "probe_o collision")
    bits = [row["bit_id"] for row in rows]
    module["ports"]["probe_o"] = {"direction": "output", "bits": bits}
    module["netnames"]["probe_o"] = {"hide_name": 0, "bits": bits, "attributes": {}}
    return canonical_json(copied)


def safe_name(candidate_id: str) -> str:
    return candidate_id.lower()


def testbench_text(module_name: str, width: int, top: str) -> str:
    return f'''`timescale 1ns/1ps
module {top};
  logic clk_i=0,rst_ni=0,start_i=0,fault_enable_i=0,fault_value_i=0;
  logic [8:0] fault_selector_i='0;
  logic [255:0] key_i='0,message_i='0;
  wire busy_o,done_o,fault_raw_o;
  wire [255:0] digest_o;
  wire [{width - 1}:0] probe_o;
  {module_name} dut(
    .clk_i(clk_i),.rst_ni(rst_ni),.start_i(start_i),.key_i(key_i),.message_i(message_i),
    .busy_o(busy_o),.done_o(done_o),.digest_o(digest_o),
    .fault_enable_i(fault_enable_i),.fault_selector_i(fault_selector_i),
    .fault_value_i(fault_value_i),.fault_raw_o(fault_raw_o),.probe_o(probe_o));
  always #5 clk_i=~clk_i;
  initial begin #20; $finish; end
endmodule
'''


def run_logged(command: list[str], log: Path, timeout: int = 900) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as stream:
        try:
            return subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                  timeout=timeout, check=False).returncode
        except subprocess.TimeoutExpired:
            stream.write(f"\nTIMEOUT_SECONDS={timeout}\n")
            return 124


def structural_checks(design: dict[str, Any], module_name: str,
                      candidates: dict[str, list[dict[str, Any]]]) -> tuple[list[dict[str, Any]], list[Path]]:
    yosys = shutil.which("yosys")
    verilator = shutil.which("verilator")
    require(yosys is not None and verilator is not None, "toolchain unavailable")
    metrics: list[dict[str, Any]] = []
    outputs: list[Path] = []
    for candidate_id, rows in candidates.items():
        candidate_dir = CANARY_ROOT / safe_name(candidate_id)
        json_path = candidate_dir / "opentitan_hmac_sha256_msg32_faultbatch000_observer.json"
        verilog_path = candidate_dir / "opentitan_hmac_sha256_msg32_faultbatch000_observer.v"
        top = "tb_" + safe_name(candidate_id)
        tb_path = candidate_dir / f"{top}.sv"
        yosys_log = candidate_dir / "yosys_structural_check.log"
        verilator_log = candidate_dir / "verilator_lint.log"
        frozen_write(json_path, create_canary(design, module_name, rows))
        frozen_write(tb_path, testbench_text(module_name, len(rows), top).encode())
        print(f"{candidate_id}: YOSYS STRUCTURAL CHECK", flush=True)
        command = [yosys, "-p", f"read_json {json_path}; hierarchy -check -top {module_name}; write_verilog -noattr {verilog_path}"]
        require(run_logged(command, yosys_log) == 0, f"{candidate_id} Yosys structural check")
        require(verilog_path.is_file() and verilog_path.stat().st_size > 0, f"{candidate_id} observer Verilog")
        print(f"{candidate_id}: VERILATOR LINT", flush=True)
        command = [verilator, "--lint-only", "--timing", "-Wall", "-Wno-fatal", "--top-module", top,
                   str(verilog_path), str(tb_path)]
        require(run_logged(command, verilator_log) == 0, f"{candidate_id} Verilator lint")
        spec = CANDIDATE_SPECS[candidate_id]
        metrics.append({
            "candidate_id": candidate_id, "family": spec["family"], "probe_banks": spec["banks"],
            "probe_bits": spec["bits"], "snapshots": spec["snapshots"],
            "cross_batch_presence": "45/45", "yosys_structural_check": "PASS",
            "verilator_lint": "PASS", "canary_executed": "NO", "response_values_captured": 0,
        })
        outputs.extend((json_path, verilog_path, tb_path, yosys_log, verilator_log))
    return metrics, outputs


def self_test() -> None:
    require(sum(spec["bits"] for spec in CANDIDATE_SPECS.values()) == 640, "candidate-width canary")
    require([spec["banks"] for spec in CANDIDATE_SPECS.values()] == [4, 2, 4], "bank-count canary")
    require(FORBIDDEN.search("$fault$inject") is not None and FORBIDDEN.search("hmac_state_q") is None, "name-filter canary")
    print("Stage 12B-3B self-test: PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    require(ROOT.name == "vlsi_fault_detection_v2", f"run from project root, not {ROOT}")
    core_outputs = (PROBE_BANK_CSV, PROBE_BANK_JSON, CONSISTENCY, STRUCTURAL_METRICS,
                    SCREENING_CONTRACT, MANIFEST, AUDIT)
    for path in core_outputs:
        require(not path.exists(), f"Stage {STAGE} output already exists: {rel(path)}")

    verify_inputs()
    graph_rows, edges = load_graph()
    repair_scores, invisible_count = repair_train_scores(graph_rows, edges)
    candidates = {
        "EM_TOPOLOGY_4X64_T24": normalize_candidate("EM_TOPOLOGY_4X64_T24", select_topology(graph_rows)),
        "EM_STATE_CHECKPOINT_2X64_T32": normalize_candidate("EM_STATE_CHECKPOINT_2X64_T32", select_state(graph_rows)),
        "EM_TESTPOINT_4X64_T16": normalize_candidate("EM_TESTPOINT_4X64_T16", select_testpoints(graph_rows, repair_scores)),
    }
    combined_rows = [row for candidate_id in CANDIDATE_SPECS for row in candidates[candidate_id]]
    require(len(combined_rows) == 640, "total candidate probe entries")
    consistency_rows, canary_design, canary_module = verify_batches(candidates)

    csv_bytes = csv_payload(combined_rows)
    bank_json = {
        "probe_bank_version": "CIRCUITSAGE-HMAC-V2.1-ENHANCED-PROBE-BANKS-12B3B-v1",
        "stage": STAGE, "status": "FROZEN", "candidate_count": 3,
        "candidate_order": list(CANDIDATE_SPECS),
        "selection_sources": ["FROZEN GOLDEN GRAPH TOPOLOGY", "AGGREGATED REPAIR_TRAIN OBSERVABILITY"],
        "repair_train_externally_invisible_faults": invisible_count,
        "partition_labels_used": False, "query_fault_identity_used": False,
        "fault_selector_value_raw_used": False, "forbidden_names_regex": FORBIDDEN.pattern,
        "candidates": {
            candidate_id: {
                **spec, "site_ids": [row["site_id"] for row in candidates[candidate_id]],
                "bit_ids": [row["bit_id"] for row in candidates[candidate_id]],
            } for candidate_id, spec in CANDIDATE_SPECS.items()
        },
        "probe_csv_sha256": hashlib.sha256(csv_bytes).hexdigest(),
    }
    consistency = {
        "consistency_version": "CIRCUITSAGE-HMAC-V2.1-ENHANCED-CROSS-BATCH-CONSISTENCY-12B3B-v1",
        "stage": STAGE, "status": "PASS", "canonical_batches": BATCHES,
        "candidate_count": 3, "all_batches_all_candidates_exact": True,
        "canonical_sources_modified": False, "batches": consistency_rows,
    }
    frozen_write(PROBE_BANK_CSV, csv_bytes)
    frozen_write(PROBE_BANK_JSON, canonical_json(bank_json))
    frozen_write(CONSISTENCY, canonical_json(consistency))
    metrics, canary_outputs = structural_checks(canary_design, canary_module, candidates)
    metrics_bytes = csv_payload(metrics)
    frozen_write(STRUCTURAL_METRICS, metrics_bytes)

    screening_contract = {
        "contract_version": "CIRCUITSAGE-HMAC-V2.1-ENHANCED-PROBE-SCREENING-12B3B-v1",
        "stage": STAGE, "status": "FROZEN",
        "probe_banks": record(PROBE_BANK_JSON), "cross_batch_consistency": record(CONSISTENCY),
        "candidate_count": 3, "screening_authorization": "NOT YET AUTHORIZED",
        "allowed_future_partition": "REPAIR_TRAIN BOUNDED SCREEN ONLY",
        "screening_sites": SCREEN_SITES, "screening_fault_instances": SCREEN_FAULTS,
        "screening_vectors": SCREEN_VECTORS, "maximum_enabled_transactions": 73728,
        "frozen_detection_target": FROZEN_TARGET, "fault_free_false_alarm_rate_max": 0.0,
        "fault_identity_in_query": "PROHIBITED", "candidate_probe_schedules": "GLOBAL AND FROZEN",
        "model_training": "NOT AUTHORIZED", "repair_calibration": "LOCKED", "repair_site_test": "LOCKED",
        "original_dev_site_test": "CONSUMED / REOPENING PROHIBITED",
        "validation": "PROHIBITED", "holdout": "PROHIBITED",
        "next_stage_requires_separate_authorization": True,
    }
    frozen_write(SCREENING_CONTRACT, canonical_json(screening_contract))
    primary = [PROBE_BANK_CSV, PROBE_BANK_JSON, CONSISTENCY, STRUCTURAL_METRICS, SCREENING_CONTRACT] + canary_outputs
    manifest = {
        "manifest_version": "CIRCUITSAGE-HMAC-V2.1-ENHANCED-PROBE-DISCOVERY-MANIFEST-12B3B-v1",
        "stage": STAGE, "status": "PASS", "stage_12b3a_audit": record(AUDIT_3A),
        "outputs": {rel(path): record(path) for path in primary},
        "candidate_count": 3, "candidate_probe_entries": 640,
        "repair_train_externally_invisible_faults": invisible_count,
        "canonical_batches_checked": BATCHES, "cross_batch_consistency": "PASS / EXACT",
        "yosys_checks_passed": 3, "verilator_lints_passed": 3,
        "simulation_binaries_executed": 0, "response_values_captured": 0,
        "model_objects_deserialized": 0, "training_calls": 0, "inference_calls": 0,
        "repair_calibration_access": 0, "repair_site_test_access": 0,
        "original_dev_site_test_access": 0, "validation_access": 0, "holdout_access": 0,
        "frozen_rtl_modified": False, "canonical_netlists_modified": False,
    }
    frozen_write(MANIFEST, canonical_json(manifest))
    audit = {
        "audit_version": "CIRCUITSAGE-HMAC-V2.1-ENHANCED-PROBE-DISCOVERY-STRUCTURAL-FREEZE-12B3B-v1",
        "stage": STAGE, "status": "PASS", "probe_discovery_status": "FROZEN / COMPLETE",
        "candidate_structural_feasibility": "PASS / 3 OF 3",
        "candidate_probe_bits": {name: spec["bits"] for name, spec in CANDIDATE_SPECS.items()},
        "candidate_probe_entries": 640, "repair_train_externally_invisible_faults": invisible_count,
        "fault_identity_conditioned_selection": False, "partition_labels_used": False,
        "cross_batch_consistency": f"PASS / {BATCHES}/{BATCHES}",
        "yosys_structural_checks": "PASS / 3 OF 3", "verilator_lints": "PASS / 3 OF 3",
        "canary_execution_response_capture": "0 / 0", "bounded_screening": "NOT YET AUTHORIZED",
        "model_training": "NOT AUTHORIZED", "fault_selector_value_raw_as_features": "PROHIBITED",
        "repair_calibration": "LOCKED / NOT ACCESSED", "repair_site_test": "LOCKED / NOT ACCESSED",
        "original_dev_site_test": "CONSUMED / NOT REOPENED", "validation_access": 0, "holdout_access": 0,
        "frozen_rtl_modified": False, "canonical_netlists_modified": False,
        "probe_bank": record(PROBE_BANK_JSON), "consistency": record(CONSISTENCY),
        "screening_contract": record(SCREENING_CONTRACT), "manifest": record(MANIFEST),
        "next_gate": "STAGE 12B-3C — ENHANCED MEASUREMENT BOUNDED-SCREENING AUTHORIZATION FREEZE",
    }
    frozen_write(AUDIT, canonical_json(audit))

    require(csv_bytes == PROBE_BANK_CSV.read_bytes(), "probe CSV replay")
    require(metrics_bytes == STRUCTURAL_METRICS.read_bytes(), "structural metrics replay")
    for path in (PROBE_BANK_JSON, CONSISTENCY, SCREENING_CONTRACT, MANIFEST, AUDIT):
        require(canonical_json(load_json(path)) == path.read_bytes(), f"canonical JSON replay: {path.name}")
    for path, expected in PINNED.items():
        require(sha256(path) == expected, f"frozen input changed: {rel(path)}")

    print("\nSTAGE 12B-3B — ENHANCED PROBE-BANK DISCOVERY AND STRUCTURAL FEASIBILITY FREEZE")
    print(f"{'Status':<64}: PASS")
    print(f"{'Probe discovery status':<64}: FROZEN / COMPLETE")
    print(f"{'Candidate structural feasibility':<64}: PASS / 3 OF 3")
    print(f"{'Topology / state / test-point bits':<64}: 256 / 128 / 256")
    print(f"{'Total candidate probe entries':<64}: 640")
    print(f"{'REPAIR_TRAIN invisible faults used in aggregate':<64}: {invisible_count}")
    print(f"{'Fault identity-conditioned selection':<64}: NO")
    print(f"{'Cross-batch consistency':<64}: PASS / {BATCHES}/{BATCHES}")
    print(f"{'Yosys structural checks / Verilator lints':<64}: PASS 3/3 / PASS 3/3")
    print(f"{'Simulation / response capture / model training':<64}: 0 / 0 / 0")
    print(f"{'Bounded screening':<64}: NOT YET AUTHORIZED")
    print(f"{'REPAIR_CALIBRATION / REPAIR_SITE_TEST':<64}: LOCKED / LOCKED")
    print(f"{'Original DEV_SITE_TEST / VALIDATION / HOLDOUT':<64}: CONSUMED / 0 / 0")
    print(f"{'Probe banks':<64}: {PROBE_BANK_JSON}")
    print(f"{'Probe banks SHA':<64}: {sha256(PROBE_BANK_JSON)}")
    print(f"{'Consistency':<64}: {CONSISTENCY}")
    print(f"{'Consistency SHA':<64}: {sha256(CONSISTENCY)}")
    print(f"{'Screening contract':<64}: {SCREENING_CONTRACT}")
    print(f"{'Screening contract SHA':<64}: {sha256(SCREENING_CONTRACT)}")
    print(f"{'Manifest':<64}: {MANIFEST}")
    print(f"{'Manifest SHA':<64}: {sha256(MANIFEST)}")
    print(f"{'Audit':<64}: {AUDIT}")
    print(f"{'Audit SHA':<64}: {sha256(AUDIT)}")
    print(f"{'Next gate':<64}: STAGE 12B-3C — ENHANCED MEASUREMENT BOUNDED-SCREENING AUTHORIZATION FREEZE")


if __name__ == "__main__":
    main()
