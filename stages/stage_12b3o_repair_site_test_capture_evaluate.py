#!/usr/bin/env python3
"""Stage 12B-3O: locked REPAIR_SITE_TEST capture, evaluation and freeze.

Executes the Stage 12B-3N-authorized EM_TESTPOINT_4X64_T16 measurement over
all 2,398 frozen REPAIR_SITE_TEST sites (4,796 SA0/SA1 faults) and 96 vectors.
Execution is sequential, single-job and checkpointed after every canonical
batch.  The selected R31_EXACT_SIGNATURE_SET model is loaded unchanged.

Response-only predictions are frozen and committed before scoring targets are
opened.  The stage then performs exactly one locked evaluation, freezes the
capture dataset, predictions, bootstrap intervals, metrics and audit evidence,
and permanently marks REPAIR_SITE_TEST consumed.  It performs no training,
threshold change, candidate reselection, DEV_SITE_TEST reopening, VALIDATION
access or HOLDOUT access.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import hmac
import io
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

try:
    import numpy as np
except ImportError as error:
    raise SystemExit("STOP: activate the project .venv (NumPy required)") from error

try:
    import stage_12b3d_enhanced_screening_execution as common
except ImportError as error:
    raise SystemExit("STOP: Stage 12B-3D runner is required beside this script") from error


STAGE = "12B-3O"
RUNNER_VERSION = "CIRCUITSAGE-HMAC-V2.1-LOCKED-REPAIR-SITE-TEST-RUNNER-12B3O-v1"
ENGINE_LINEAGE = "STAGE 12B-3G-R1/3K PROVEN CAPTURE ENGINE; LOCKED SITE-TEST SPECIALIZATION"
ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "config/v2_1_improvement"
RESULT = ROOT / "results/circuitsage_hmac_v2_12b3"
WORK = RESULT / "repair_site_test_capture_execution_12b3o"
LOCK = WORK / "stage_12b3o_execution.lock"
RAW = WORK / "raw_batches"
PER_BATCH = WORK / "per_batch_npz"
BUILD = ROOT / "build/circuitsage_hmac_v2_12b3/repair_site_test_capture_12b3o"

CHECKPOINT = WORK / "circuitsage_hmac_v2_1_repair_site_test_capture_checkpoint_12b3o.json"
FEATURES = WORK / "circuitsage_hmac_v2_1_repair_site_test_features_12b3o.npz"
TARGETS = WORK / "circuitsage_hmac_v2_1_repair_site_test_targets_12b3o.npz"
SIGNATURES = WORK / "circuitsage_hmac_v2_1_repair_site_test_signature_summary_12b3o.csv"
PREDICTIONS = WORK / "circuitsage_hmac_v2_1_repair_site_test_predictions_12b3o.npz"
CANDIDATES = WORK / "circuitsage_hmac_v2_1_repair_site_test_candidate_sets_12b3o.csv"
PREDICTION_COMMITMENT = WORK / "circuitsage_hmac_v2_1_repair_site_test_prediction_commitment_12b3o.json"
METRICS = WORK / "circuitsage_hmac_v2_1_repair_site_test_metrics_12b3o.json"
BOOTSTRAP = WORK / "circuitsage_hmac_v2_1_repair_site_test_site_bootstrap_12b3o.csv"
SCHEMA = WORK / "circuitsage_hmac_v2_1_repair_site_test_dataset_schema_12b3o.json"
MANIFEST = RESULT / "circuitsage_hmac_v2_1_repair_site_test_evaluation_manifest_12b3o.json"
AUDIT = RESULT / "circuitsage_hmac_v2_1_repair_site_test_capture_evaluation_freeze_12b3o.json"

SOURCE_3N = ROOT / "stage_12b3n_repair_site_test_authorization.py"
SOURCE_COMMON = ROOT / "stage_12b3d_enhanced_screening_execution.py"
AUTH_WORK = RESULT / "repair_site_test_authorization_12b3n"
SITE_TEST_SITES = AUTH_WORK / "circuitsage_hmac_v2_1_repair_site_test_sites_12b3n.csv"
EXECUTION_PLAN = AUTH_WORK / "circuitsage_hmac_v2_1_repair_site_test_execution_plan_12b3n.csv"
AUTH_SCHEMA = AUTH_WORK / "circuitsage_hmac_v2_1_repair_site_test_capture_evaluation_schema_12b3n.json"
PREFLIGHT = AUTH_WORK / "circuitsage_hmac_v2_1_repair_site_test_preflight_12b3n.json"
TRUTH_COMMITMENT_3N = AUTH_WORK / "circuitsage_hmac_v2_1_repair_site_test_truth_commitment_12b3n.json"
EXECUTION_CONTRACT = CONFIG / "circuitsage_hmac_v2_1_repair_site_test_capture_evaluation_contract_12b3n.json"
AUTHORIZATION = CONFIG / "circuitsage_hmac_v2_1_repair_site_test_capture_evaluation_authorization_12b3n.json"
MANIFEST_3N = RESULT / "circuitsage_hmac_v2_1_repair_site_test_authorization_manifest_12b3n.json"
AUDIT_3N = RESULT / "circuitsage_hmac_v2_1_repair_site_test_authorization_freeze_12b3n.json"

MODEL_WORK = RESULT / "repair_model_training_12b3m"
SELECTED_MODEL = MODEL_WORK / "circuitsage_hmac_v2_1_selected_repair_model_12b3m.npz"
SELECTED_METADATA = MODEL_WORK / "circuitsage_hmac_v2_1_selected_repair_model_metadata_12b3m.json"
SELECTION_LOCK = MODEL_WORK / "circuitsage_hmac_v2_1_repair_model_selection_lock_12b3m.json"
ACCEPTANCE = CONFIG / "circuitsage_hmac_v2_1_repair_model_acceptance_contract_12b3i.json"

RESULT_2 = ROOT / "results/circuitsage_hmac_v2_12b2"
SELECTED_VECTORS = RESULT_2 / "adaptive_vector_selection_12b2d/circuitsage_hmac_v2_1_selected_adaptive_vectors_12b2d.npz"
DISCOVERY = RESULT / "enhanced_probe_discovery_12b3b"
PROBE_CSV = DISCOVERY / "circuitsage_hmac_v2_1_enhanced_probe_banks_12b3b.csv"
PROBE_JSON = DISCOVERY / "circuitsage_hmac_v2_1_enhanced_probe_banks_12b3b.json"
CONSISTENCY = DISCOVERY / "circuitsage_hmac_v2_1_enhanced_cross_batch_consistency_12b3b.json"

PINNED = {
    SOURCE_3N: "2aec944fbbc6d357de391f871d5e63bc62f2d3d6ab7c59a583a3852bc4f191da",
    SOURCE_COMMON: "3392345b66e470e1975ef32baa3a7946df8f636f2f20748b91c9d12fbfcc65ff",
    SITE_TEST_SITES: "0b13e2c8f1cdcfd462da76710f6fcb1b56f8cac15a4ef28738eb3f78e59fcbfb",
    EXECUTION_PLAN: "157c5bed2897708c6f4105f3f7b8e3671ad4946387cc424659c26eff29443aef",
    AUTH_SCHEMA: "31567ce37a056661755e8d1a5b4c0d9fb1d48d42c3580ee7ae8bf37ddb739087",
    PREFLIGHT: "30e767a15c348f1c30483522ec4d340dd8440c7217ab0e3e08188a225ddb057a",
    TRUTH_COMMITMENT_3N: "1e5ed12c3eb2d87bbf3ba4d255b7515bb36b2a2b1d042e7fb01aa8dc89390531",
    EXECUTION_CONTRACT: "12d753bd13fec7cd20bb267eb3f81f22a69cc0dc2409ec4079fe3afa125989ec",
    AUTHORIZATION: "4d7b430300a3b0a47a1d9751e7d0c8a1657bb8250e22ca4205ad904b76215006",
    MANIFEST_3N: "34efd759b417d7c41c87d5f75a5edcf7630d55ccce6ae09fe9618e2fd5387deb",
    AUDIT_3N: "8d518251b40535801081c1d71059d600fd0f4c05230b4dcc9a4ed11847574cdb",
    SELECTED_MODEL: "2f1d35c3f2b71f975859a99238d07e1e0d8620f02d79f3808ca1e5c2ccf6f39a",
    SELECTED_METADATA: "8e80d3e3cfc6bfb0c76698150a24097a254f908f566e333d2dcc70d67afcc247",
    SELECTION_LOCK: "7b9cb7d6c4dd9e8123fdfc3b327bcd6213b827d6338cdc7bd0326f739704bf53",
    ACCEPTANCE: "ca95bc7fc7587e995f48b0efa03390734192fd251cd82ebd25387c5663d1e0d2",
    SELECTED_VECTORS: "be9df0a3a70e61b54fc793439328bbcc307b9373fd881b0e97d500919ec27bff",
    PROBE_CSV: "d7f9b984d48db4e9a8c242e1ee4e859ef5f321d6d5546b901bde569d5009eae2",
    PROBE_JSON: "f7ee21e7b1f166c88486607689a0ec1c951de484175a7a54757ed185d58db537",
    CONSISTENCY: "6c94117035b979135657d2b384b11db40a7c63029f34df365f8f34360d4bbb20",
}

SELECTED = "EM_TESTPOINT_4X64_T16"
SELECTED_MODEL_ID = "R31_EXACT_SIGNATURE_SET"
BATCHES = 45
SITES = 2398
FAULTS = 4796
VECTORS = 96
PROBE_BITS = 256
SNAPSHOTS = 16
BASELINE_RECORDS = BATCHES * VECTORS
ENABLED_RECORDS = FAULTS * VECTORS
TOTAL_RECORDS = BASELINE_RECORDS + ENABLED_RECORDS
TIMEOUT_CYCLES = 2000
EXPECTED_BASELINE_CYCLES = 343
MIN_FREE_GIB = 10
BOOTSTRAP_REPLICATES = 1000
SNAPSHOT_CYCLES = [0, 1, 2, 4, 8, 16, 32, 48, 64, 80, 96, 128, 160, 224, 288]

CSV_FIELDS = [
    "batch_id", "run_type", "site_id", "selector", "stuck_value", "vector_rank",
    "source_vector_index", "cycles", "timed_out", "busy_first_cycle", "busy_last_cycle",
    "done_cycle", "unknown", "expected_digest", "actual_digest", "snapshot_hex", "toggle_hex",
]

sha256 = common.sha256
rel = common.rel
canonical_json = common.canonical_json
load_json = common.load_json
atomic_json = common.atomic_json
frozen_write = common.frozen_write
record = common.record
resolve_record = common.resolve_record
verify_record = common.verify_record
csv_payload = common.csv_payload
deterministic_npz = common.deterministic_npz
snapshot_bits = common.snapshot_bits
observer_json = common.observer_json
run_logged = common.run_logged
sv_quote = common.sv_quote
now = common.now


def stop(message: str) -> None:
    raise SystemExit(f"STOP: {message}")


def require(condition: bool, message: str) -> None:
    if not condition:
        stop(message)


def freeze_or_verify(path: Path, payload: bytes) -> None:
    """Freeze a deterministic output, or verify it during crash-safe resume."""
    if path.exists():
        require(path.read_bytes() == payload, f"resume output mismatch: {path.name}")
    else:
        frozen_write(path, payload)


def verify_inputs() -> tuple[list[int], dict[int, dict[str, Any]]]:
    print("STAGE 12B-3O — LOCKED REPAIR_SITE_TEST CAPTURE AND EVALUATION")
    print("FROZEN INPUT VERIFICATION")
    for path, expected in PINNED.items():
        require(path.is_file(), f"missing frozen input: {rel(path)}")
        require(sha256(path) == expected, f"frozen input SHA: {rel(path)}")
        print(f"  {path.name:<100}: OK", flush=True)

    audit = load_json(AUDIT_3N)
    authorization = load_json(AUTHORIZATION)
    contract = load_json(EXECUTION_CONTRACT)
    manifest = load_json(MANIFEST_3N)
    metadata = load_json(SELECTED_METADATA)
    selection = load_json(SELECTION_LOCK)
    acceptance = load_json(ACCEPTANCE)
    require(audit.get("status") == "PASS" and audit.get("authorization_status") == "FROZEN", "12B-3N freeze")
    require(audit.get("selected_measurement") == SELECTED, "authorized measurement")
    require(audit.get("selected_model") == SELECTED_MODEL_ID, "authorized model")
    require(audit.get("source_partition") == "REPAIR_SITE_TEST", "authorized partition")
    require(audit.get("sites_faults_vectors") == [SITES, FAULTS, VECTORS], "capture dimensions")
    require(audit.get("canonical_batches") == "45/45", "batch authorization")
    require(audit.get("baseline_enabled_total_records") == [BASELINE_RECORDS, ENABLED_RECORDS, TOTAL_RECORDS], "record budget")
    require(audit.get("probe_bits_banks_snapshots") == [PROBE_BITS, 4, SNAPSHOTS], "measurement dimensions")
    require(audit.get("capture_evaluation") == "AUTHORIZED / NOT STARTED", "execution authorization")
    require(audit.get("prediction_before_truth_commitment") == "REQUIRED", "prediction commitment")
    require(audit.get("training_threshold_change_candidate_reselection") == "0 / 0 / 0", "selection boundary")
    require(audit.get("repair_site_test_state") == "AUTHORIZED / NOT OPENED", "site-test boundary")
    require(audit.get("dev_site_test") == "CONSUMED / NOT REOPENED", "development test boundary")
    require(audit.get("validation_holdout_access") == [0, 0], "protected access")
    require(authorization.get("locked_capture_and_evaluation") == "AUTHORIZED / NOT STARTED", "authorization object")
    require(authorization.get("selected_measurement") == SELECTED, "authorization measurement")
    require(authorization.get("selected_model") == SELECTED_MODEL_ID, "authorization model")
    require(authorization.get("sites_faults_vectors") == [SITES, FAULTS, VECTORS], "authorization dimensions")
    require(authorization.get("maximum_enabled_transactions") == ENABLED_RECORDS, "authorization budget")
    require(authorization.get("maximum_total_records") == TOTAL_RECORDS, "authorization total budget")
    require(authorization.get("training") == "NOT AUTHORIZED", "authorization training boundary")
    require(contract.get("status") == "FROZEN", "execution contract")
    require(contract.get("scope") == "ONE LOCKED CLOSED-CATALOG REPAIR_SITE_TEST CAPTURE AND EVALUATION", "execution scope")
    require(contract.get("selected_measurement") == SELECTED, "contract measurement")
    require(contract.get("selected_model") == SELECTED_MODEL_ID, "contract model")
    require(contract.get("sites_faults_vectors") == [SITES, FAULTS, VECTORS], "contract dimensions")
    require(contract.get("checkpoint_resume") == "REQUIRED AFTER EACH CANONICAL BATCH", "checkpoint contract")
    require(contract.get("query_fault_identity") == "PROHIBITED", "identity boundary")
    require(contract.get("prediction_commitment_before_truth") == "REQUIRED", "commitment contract")
    require(contract.get("training_scaler_fit_threshold_change_candidate_reselection") == "PROHIBITED / PROHIBITED / PROHIBITED / PROHIBITED", "contract training boundary")
    require(manifest.get("status") == "PASS", "3N manifest")
    outputs = manifest.get("outputs")
    require(isinstance(outputs, dict), "3N output registry")
    for path in (SITE_TEST_SITES, EXECUTION_PLAN, AUTH_SCHEMA, PREFLIGHT, TRUTH_COMMITMENT_3N, EXECUTION_CONTRACT, AUTHORIZATION):
        verify_record(outputs.get(rel(path)), path, f"3N {path.name}")
    require(metadata.get("candidate_id") == SELECTED_MODEL_ID, "selected metadata candidate")
    require(metadata.get("selected_measurement") == SELECTED, "selected metadata measurement")
    require(metadata.get("trainable_parameters") == 0, "selected model parameters")
    require(selection.get("selected_candidate") == SELECTED_MODEL_ID, "selection lock candidate")
    require(selection.get("retraining_after_selection") == "PROHIBITED", "retraining lock")
    require(selection.get("threshold_change_after_selection") == "PROHIBITED", "threshold lock")
    require(acceptance.get("status") == "FROZEN", "acceptance contract")

    with np.load(SELECTED_MODEL, allow_pickle=False) as selected:
        require(set(selected.files) == {
            "candidate_id_utf8", "exact_signature_precedence", "ambiguity_preservation",
            "ood_distance_threshold", "pair_feature_names_utf8", "block_scale_names_utf8",
            "block_scale_values",
        }, "selected-model members")
        candidate_id = bytes(np.asarray(selected["candidate_id_utf8"], dtype=np.uint8)).decode()
        require(candidate_id == SELECTED_MODEL_ID, "selected-model identity")
        require(np.asarray(selected["exact_signature_precedence"]).tolist() == [1], "exact precedence")
        require(np.asarray(selected["ambiguity_preservation"]).tolist() == [1], "ambiguity preservation")

    probe_json = load_json(PROBE_JSON)
    candidate = probe_json.get("candidates", {}).get(SELECTED)
    require(isinstance(candidate, dict), "selected probe candidate")
    require(candidate.get("bits") == PROBE_BITS and candidate.get("snapshots") == SNAPSHOTS, "probe dimensions")
    bit_ids = candidate.get("bit_ids")
    require(isinstance(bit_ids, list) and len(bit_ids) == PROBE_BITS and len(set(bit_ids)) == PROBE_BITS, "probe bit IDs")
    with PROBE_CSV.open(newline="", encoding="utf-8") as stream:
        probe_rows = [row for row in csv.DictReader(stream) if row["candidate_id"] == SELECTED]
    require(len(probe_rows) == PROBE_BITS, "probe CSV rows")
    require([int(row["probe_bit"]) for row in probe_rows] == list(range(PROBE_BITS)), "probe order")
    require([int(row["bit_id"]) for row in probe_rows] == bit_ids, "probe JSON/CSV agreement")
    consistency = load_json(CONSISTENCY)
    entries = consistency.get("batches")
    require(consistency.get("status") == "PASS" and consistency.get("all_batches_all_candidates_exact") is True, "probe consistency")
    require(isinstance(entries, list) and len(entries) == BATCHES, "batch registry")
    batch_entries = {int(item["batch_id"]): item for item in entries}
    require(set(batch_entries) == set(range(BATCHES)), "batch registry coverage")
    require(shutil.which("yosys") is not None and shutil.which("verilator") is not None, "toolchain unavailable")
    require(shutil.disk_usage(ROOT).free >= MIN_FREE_GIB * 1024**3, f"less than {MIN_FREE_GIB} GiB free disk")
    print("  Authorization, selection lock, model, probe schedule and toolchain                    : PASS")
    return [int(value) for value in bit_ids], batch_entries


def load_vectors() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(SELECTED_VECTORS, allow_pickle=False) as archive:
        required = {"key_u8", "message_u8", "selection_rank", "selection_round", "vector_index"}
        require(set(archive.files) == required, "selected-vector NPZ members")
        keys = np.asarray(archive["key_u8"], dtype=np.uint8)
        messages = np.asarray(archive["message_u8"], dtype=np.uint8)
        source_indices = np.asarray(archive["vector_index"], dtype=np.int32)
        ranks = np.asarray(archive["selection_rank"], dtype=np.int32)
    require(keys.shape == (VECTORS, 32) and messages.shape == (VECTORS, 32), "selected-vector shapes")
    require(ranks.tolist() == list(range(VECTORS)), "selected-vector ranks")
    require(len(set(source_indices.tolist())) == VECTORS, "selected-vector uniqueness")
    digests = np.vstack([
        np.frombuffer(hmac.new(keys[index].tobytes(), messages[index].tobytes(), hashlib.sha256).digest(), dtype=np.uint8)
        for index in range(VECTORS)
    ])
    return keys, messages, source_indices, digests


def load_site_plan(batch_entries: dict[int, dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[int, list[dict[str, Any]]]]:
    with SITE_TEST_SITES.open(newline="", encoding="utf-8") as stream:
        source_rows = list(csv.DictReader(stream))
    require(len(source_rows) == SITES, "site-test-site row count")
    require([int(row["site_test_rank"]) for row in source_rows] == list(range(SITES)), "site-test-site ordering")
    batches: dict[int, list[dict[str, Any]]] = {batch_id: [] for batch_id in range(BATCHES)}
    for row in source_rows:
        batch_id = int(row["batch_id"])
        require(0 <= batch_id < BATCHES, "site batch ID")
        batches[batch_id].append({
            "site_id": row["site_id"], "site_index": int(row["site_index"]),
            "site_test_rank": int(row["site_test_rank"]), "selector": int(row["selector"]),
            "batch_id": batch_id,
        })
    for batch_id in range(BATCHES):
        require(batches[batch_id], f"Batch {batch_id:03d} has no authorized site")
        batches[batch_id].sort(key=lambda row: row["site_test_rank"])
        entry = batch_entries[batch_id]
        canonical = resolve_record(entry["canonical_json"])
        mapping = resolve_record(entry["mapping"])
        verify_record(entry["canonical_json"], canonical, f"Batch {batch_id:03d} JSON")
        verify_record(entry["mapping"], mapping, f"Batch {batch_id:03d} mapping")
        mapping_sites = common.find_sites(load_json(mapping))
        require(mapping_sites is not None, f"Batch {batch_id:03d} mapping sites")
        lookup = {str(site["fault_site_id"]): int(site["selector_code"]) for site in mapping_sites}
        for site in batches[batch_id]:
            require(lookup.get(site["site_id"]) == site["selector"], f"Batch {batch_id:03d} site/selector binding")
    plan_rows = []
    with EXECUTION_PLAN.open(newline="", encoding="utf-8") as stream:
        plan_rows = list(csv.DictReader(stream))
    require(len(plan_rows) == BATCHES, "execution-plan rows")
    for batch_id, row in enumerate(plan_rows):
        require(int(row["batch_id"]) == batch_id, f"Batch {batch_id:03d} plan order")
        require(int(row["site_count"]) == len(batches[batch_id]), f"Batch {batch_id:03d} site budget")
        require(int(row["enabled_transactions"]) == len(batches[batch_id]) * 2 * VECTORS, f"Batch {batch_id:03d} enabled budget")
    ordered = sorted((row for values in batches.values() for row in values), key=lambda row: row["site_test_rank"])
    require([row["site_test_rank"] for row in ordered] == list(range(SITES)), "site-test-site coverage")
    return ordered, batches


def sample_case_lines() -> str:
    return "\n".join(
        f"        {cycle}: begin snapshot_work[{slot}]=probe_o; snapshot_written[{slot}]=1; end"
        for slot, cycle in enumerate(SNAPSHOT_CYCLES[1:], start=1)
    )


def generate_testbench(batch_id: int, selected: list[dict[str, Any]], csv_path: Path,
                       key_mem: Path, message_mem: Path, digest_mem: Path,
                       source_mem: Path) -> str:
    key = f"{batch_id:03d}"
    top = f"tb_v21_repair_site_test_capture_batch_{key}"
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
  localparam integer PROBE_BITS={PROBE_BITS}, SNAPSHOTS={SNAPSHOTS};
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
  integer csv_fd,site_slot,vector_rank,stuck_index,s,t,cycles_result;
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
      if(compare_baseline) $fwrite(csv_fd,"%064h",snapshot_work[s]^baseline_snapshot[vi][s]);
      else $fwrite(csv_fd,"%064h",snapshot_work[s]);
    end
    $fwrite(csv_fd,","); for(t=0;t<PROBE_BITS;t=t+1) $fwrite(csv_fd,"%04x",toggle_work[t]);
    $fwrite(csv_fd,"\\n");
  end endtask
  initial begin
    $readmemh("{sv_quote(str(key_mem))}",key_vectors); $readmemh("{sv_quote(str(message_mem))}",message_vectors);
    $readmemh("{sv_quote(str(digest_mem))}",digest_vectors); $readmemh("{sv_quote(str(source_mem))}",source_vector_indices);
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
    for(site_slot=0;site_slot<SITE_COUNT;site_slot=site_slot+1) for(stuck_index=0;stuck_index<2;stuck_index=stuck_index+1) begin
      stuck_value=(stuck_index!=0);
      for(vector_rank=0;vector_rank<VECTOR_COUNT;vector_rank=vector_rank+1) begin
        run_transaction(vector_rank,1,selected_selectors[site_slot],stuck_value,cycles_result,timeout_result,digest_result,busy_first_result,busy_last_result,done_cycle_result,unknown_result);
        enabled_runs=enabled_runs+1; if(unknown_result) unknown_runs=unknown_runs+1;
        $fwrite(csv_fd,"%0d,ENABLED,%s,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%064h,%064h,",BATCH_ID,selected_site_ids[site_slot],selected_selectors[site_slot],stuck_index,vector_rank,source_vector_indices[vector_rank],cycles_result,timeout_result,busy_first_result,busy_last_result,done_cycle_result,unknown_result,digest_vectors[vector_rank],digest_result);
        write_trace_fields(vector_rank,1);
      end
    end
    $fclose(csv_fd); $display("V21_REPAIR_SITE_TEST_CAPTURE_BATCH=%0d",BATCH_ID);
    $display("V21_REPAIR_SITE_TEST_CAPTURE_BASELINES=%0d",baseline_runs); $display("V21_REPAIR_SITE_TEST_CAPTURE_ENABLED=%0d",enabled_runs);
    if(baseline_runs==VECTOR_COUNT&&enabled_runs==SITE_COUNT*2*VECTOR_COUNT&&unknown_runs==0&&baseline_failures==0) begin
      $display("V21_REPAIR_SITE_TEST_CAPTURE_BATCH_RESULT=PASS"); $finish;
    end else $fatal(1,"V21_REPAIR_SITE_TEST_CAPTURE_BATCH_RESULT=FAIL baseline_failures=%0d",baseline_failures);
  end
endmodule
'''


def pack_snapshot(text: str) -> np.ndarray:
    bits = snapshot_bits(text, PROBE_BITS)
    return np.packbits(bits, bitorder="little")


def parse_batch_csv(path: Path, batch_id: int, sites: list[dict[str, Any]],
                    source_indices: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    local_faults = len(sites) * 2
    common_arrays = {
        "baseline_cycles": np.zeros(VECTORS, dtype=np.uint16),
        "baseline_control_timeline": np.zeros((VECTORS, 3), dtype=np.int16),
        "baseline_probe_snapshots": np.zeros((VECTORS, SNAPSHOTS, PROBE_BITS // 8), dtype=np.uint8),
        "baseline_probe_toggle_count": np.zeros((VECTORS, PROBE_BITS), dtype=np.uint16),
        "cycles": np.zeros((local_faults, VECTORS), dtype=np.uint16),
        "control_timeline": np.zeros((local_faults, VECTORS, 3), dtype=np.int16),
        "timed_out": np.zeros((local_faults, VECTORS), dtype=np.uint8),
        "digest_xor": np.zeros((local_faults, VECTORS, 32), dtype=np.uint8),
        "probe_snapshot_xor": np.zeros((local_faults, VECTORS, SNAPSHOTS, PROBE_BITS // 8), dtype=np.uint8),
        "probe_toggle_count": np.zeros((local_faults, VECTORS, PROBE_BITS), dtype=np.uint16),
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
            require(len(snapshot_text) == SNAPSHOTS * 64, "snapshot trace length")
            snapshots = [pack_snapshot(snapshot_text[index * 64:(index + 1) * 64]) for index in range(SNAPSHOTS)]
            toggle_text = row["toggle_hex"]
            require(len(toggle_text) == PROBE_BITS * 4 and re.fullmatch(r"[0-9a-fA-F]+", toggle_text) is not None, "toggle trace")
            toggles = np.asarray([int(toggle_text[index * 4:(index + 1) * 4], 16) for index in range(PROBE_BITS)], dtype=np.uint16)
            row_cycles = int(row["cycles"])
            control = [int(row["busy_first_cycle"]), int(row["busy_last_cycle"]), int(row["done_cycle"])]
            if row["run_type"] == "BASELINE":
                require(row["site_id"] == "BASELINE" and seen_baseline[rank] == 0, "duplicate/invalid baseline")
                seen_baseline[rank] = 1
                common_arrays["baseline_cycles"][rank] = row_cycles
                common_arrays["baseline_control_timeline"][rank] = control
                common_arrays["baseline_probe_snapshots"][rank] = snapshots
                common_arrays["baseline_probe_toggle_count"][rank] = toggles
                baseline_failures += int(row["timed_out"] != "0" or row_cycles != EXPECTED_BASELINE_CYCLES or row["expected_digest"].lower() != row["actual_digest"].lower())
            elif row["run_type"] == "ENABLED":
                require(row["site_id"] in site_to_local, "unexpected site-test capture site")
                local_site = site_to_local[row["site_id"]]
                require(int(row["selector"]) == sites[local_site]["selector"], "selector binding")
                stuck = int(row["stuck_value"])
                require(stuck in (0, 1), "stuck value")
                fault = local_site * 2 + stuck
                require(seen[fault, rank] == 0, "duplicate enabled sample")
                seen[fault, rank] = 1
                common_arrays["cycles"][fault, rank] = row_cycles
                common_arrays["control_timeline"][fault, rank] = control
                common_arrays["timed_out"][fault, rank] = int(row["timed_out"])
                expected = bytes.fromhex(row["expected_digest"])
                actual = bytes.fromhex(row["actual_digest"])
                require(len(expected) == 32 and len(actual) == 32, "digest encoding")
                common_arrays["digest_xor"][fault, rank] = np.frombuffer(bytes(a ^ b for a, b in zip(expected, actual)), dtype=np.uint8)
                common_arrays["probe_snapshot_xor"][fault, rank] = snapshots
                common_arrays["probe_toggle_count"][fault, rank] = toggles
            else:
                stop("unknown CSV run type")
    require(np.all(seen_baseline == 1) and np.all(seen == 1), f"sample coverage Batch {batch_id:03d}")
    require(unknown == 0 and baseline_failures == 0, f"semantic failures Batch {batch_id:03d}")
    arrays: dict[str, np.ndarray] = {
        "baseline_cycles": common_arrays["baseline_cycles"],
        "baseline_control_timeline": common_arrays["baseline_control_timeline"],
        "baseline_probe_snapshots": common_arrays["baseline_probe_snapshots"],
        "baseline_probe_toggle_count": common_arrays["baseline_probe_toggle_count"],
        "control_timeline_delta": (common_arrays["control_timeline"].astype(np.int32) - common_arrays["baseline_control_timeline"][None]).astype(np.int16),
        "cycle_delta": (common_arrays["cycles"].astype(np.int32) - common_arrays["baseline_cycles"][None]).astype(np.int16),
        "cycles": common_arrays["cycles"],
        "digest_xor": common_arrays["digest_xor"],
        "timed_out": common_arrays["timed_out"],
        "probe_snapshot_xor": common_arrays["probe_snapshot_xor"],
        "probe_toggle_count": common_arrays["probe_toggle_count"],
        "fault_instance_index": np.asarray([site["site_test_rank"] * 2 + stuck for site in sites for stuck in (0, 1)], dtype=np.int32),
        "source_vector_index": source_indices.astype(np.int32),
        "vector_rank": np.arange(VECTORS, dtype=np.int32),
    }
    arrays["external_detected"] = ((arrays["timed_out"] != 0) | (arrays["cycle_delta"] != 0) | np.any(arrays["digest_xor"] != 0, axis=2)).astype(np.uint8)
    arrays["probe_toggle_delta"] = (arrays["probe_toggle_count"].astype(np.int32) - arrays["baseline_probe_toggle_count"][None].astype(np.int32)).astype(np.int16)
    arrays["probe_effect"] = (
        np.any(arrays["probe_snapshot_xor"] != 0, axis=(2, 3)) |
        np.any(arrays["probe_toggle_delta"] != 0, axis=2) |
        np.any(arrays["control_timeline_delta"] != 0, axis=2)
    ).astype(np.uint8)
    combined = np.any((arrays["external_detected"] != 0) | (arrays["probe_effect"] != 0), axis=1)
    summary = {
        "baseline_records": VECTORS,
        "enabled_records": local_faults * VECTORS,
        "fault_instances": local_faults,
        "unknown_records": unknown,
        "baseline_failures": baseline_failures,
        "external_observable_faults": int(np.sum(np.any(arrays["external_detected"] != 0, axis=1))),
        "combined_observable_faults": int(np.sum(combined)),
    }
    return arrays, summary


def execute(resume: bool) -> None:
    require(not AUDIT.exists(), f"Stage {STAGE} is already frozen")
    require(resume or not MANIFEST.exists(), "partial finalization exists; rerun with --resume")
    probe_bits, batch_entries = verify_inputs()
    keys, messages, source_indices, digests = load_vectors()
    site_rows, batches = load_site_plan(batch_entries)
    WORK.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    PER_BATCH.mkdir(parents=True, exist_ok=True)
    BUILD.mkdir(parents=True, exist_ok=True)
    key_mem = WORK / "site_test_keys_12b3o.mem"
    message_mem = WORK / "site_test_messages_12b3o.mem"
    digest_mem = WORK / "site_test_expected_digests_12b3o.mem"
    source_mem = WORK / "site_test_source_indices_12b3o.mem"
    memory_payloads = {
        key_mem: "".join(row.tobytes().hex() + "\n" for row in keys).encode(),
        message_mem: "".join(row.tobytes().hex() + "\n" for row in messages).encode(),
        digest_mem: "".join(row.tobytes().hex() + "\n" for row in digests).encode(),
        source_mem: "".join(f"{int(value):08x}\n" for value in source_indices).encode(),
    }
    for path, payload in memory_payloads.items():
        if path.exists():
            require(path.read_bytes() == payload, f"memory-file mismatch: {path.name}")
        else:
            path.write_bytes(payload)

    if CHECKPOINT.exists():
        require(resume, "checkpoint exists; rerun with --resume")
        state = load_json(CHECKPOINT)
        require(state.get("stage") == STAGE and state.get("selected_measurement") == SELECTED, "checkpoint contract")
        if state.get("status") != "PASS":
            state["runner_version"] = RUNNER_VERSION
            state["engine_lineage"] = ENGINE_LINEAGE
            state["updated_at"] = now()
            atomic_json(CHECKPOINT, state)
    else:
        require(not resume, "--resume requested without checkpoint")
        state = {
            "checkpoint_version": "CIRCUITSAGE-HMAC-V2.1-REPAIR-SITE-TEST-CAPTURE-CHECKPOINT-12B3O-v1",
            "stage": STAGE, "status": "RUNNING", "started_at": now(), "updated_at": now(),
            "selected_measurement": SELECTED, "planned_batches": list(range(BATCHES)),
            "completed_batches": [], "probe_bits": PROBE_BITS, "snapshots": SNAPSHOTS,
            "enabled_transactions": ENABLED_RECORDS, "batches": {},
            "runner_version": RUNNER_VERSION,
            "engine_lineage": ENGINE_LINEAGE,
        }
        atomic_json(CHECKPOINT, state)
    completed = set(int(value) for value in state.get("completed_batches", []))
    yosys = shutil.which("yosys")
    verilator = shutil.which("verilator")
    require(yosys is not None and verilator is not None, "toolchain unavailable")

    for batch_id in range(BATCHES):
        key = f"{batch_id:03d}"
        if batch_id in completed:
            item = state["batches"][key]
            require(sha256(Path(item["csv"])) == item["csv_sha256"], f"resume CSV Batch {key}")
            require(sha256(Path(item["batch_npz"])) == item["batch_npz_sha256"], f"resume NPZ Batch {key}")
            print(f"Batch {key}: CHECKPOINT PASS", flush=True)
            continue
        selected = batches[batch_id]
        entry = batch_entries[batch_id]
        canonical = resolve_record(entry["canonical_json"])
        mapping = resolve_record(entry["mapping"])
        module = f"opentitan_hmac_sha256_msg32_faultbatch{key}"
        batch_raw = RAW / f"batch_{key}"
        batch_build = BUILD / f"batch_{key}"
        batch_raw.mkdir(parents=True, exist_ok=True)
        batch_build.mkdir(parents=True, exist_ok=True)
        derived_json = batch_build / f"{module}_testpoint_observer.json"
        derived_verilog = batch_build / f"{module}_testpoint_observer.v"
        observer_payload = observer_json(canonical, module, probe_bits)
        if derived_json.exists():
            require(derived_json.read_bytes() == observer_payload, f"observer mismatch Batch {key}")
        else:
            derived_json.write_bytes(observer_payload)
        yosys_log = batch_raw / "yosys_observer.log"
        print(f"Batch {key}: DERIVE TEST-POINT OBSERVER sites={len(selected)} bits={PROBE_BITS}", flush=True)
        require(run_logged([yosys, "-p", f"read_json {derived_json}; hierarchy -check -top {module}; write_verilog -noattr {derived_verilog}"], yosys_log, 900) == 0, f"Batch {key} observer generation")
        csv_path = batch_raw / f"circuitsage_hmac_v2_1_repair_site_test_capture_batch_{key}.csv"
        tb = batch_build / f"tb_v21_repair_site_test_capture_batch_{key}.sv"
        tb_payload = generate_testbench(batch_id, selected, csv_path.resolve(), key_mem.resolve(), message_mem.resolve(), digest_mem.resolve(), source_mem.resolve()).encode()
        if tb.exists() and tb.read_bytes() != tb_payload:
            require(resume, f"testbench mismatch Batch {key}")
            preserved_tb = tb.with_suffix(".sv.pre_12b3o_r1")
            if not preserved_tb.exists():
                tb.replace(preserved_tb)
            else:
                require(tb.read_bytes() == preserved_tb.read_bytes(), f"preserved testbench mismatch Batch {key}")
                tb.unlink()
            tb.write_bytes(tb_payload)
            print(f"Batch {key}: R1 REGENERATED UNCOMMITTED TESTBENCH", flush=True)
        elif not tb.exists():
            tb.write_bytes(tb_payload)
        top = f"tb_v21_repair_site_test_capture_batch_{key}"
        obj_dir = batch_build / "obj_dir_r1"
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
        require("V21_REPAIR_SITE_TEST_CAPTURE_BATCH_RESULT=PASS" in simulation_log.read_text(errors="replace"), f"Batch {key} PASS token")
        arrays, summary = parse_batch_csv(csv_path, batch_id, selected, source_indices)
        batch_npz = PER_BATCH / f"circuitsage_hmac_v2_1_repair_site_test_capture_batch_{key}.npz"
        freeze_or_verify(batch_npz, deterministic_npz(arrays))
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
        state["batches"][key] = item
        completed.add(batch_id)
        state["completed_batches"] = sorted(completed)
        state["updated_at"] = now()
        atomic_json(CHECKPOINT, state)
        print(f"Batch {key}: PASS enabled={summary['enabled_records']}", flush=True)
    require(completed == set(range(BATCHES)), "incomplete site-test capture batches")
    freeze_dataset(state, site_rows, batches, source_indices)


def locked_execute(resume: bool) -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    with LOCK.open("a+", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                "another Stage 12B-3O execution is active; use --status and do not start a second runner"
            ) from exc
        lock_handle.seek(0)
        lock_handle.truncate()
        lock_handle.write(f"pid={os.getpid()}\nstarted_at={now()}\n")
        lock_handle.flush()
        os.fsync(lock_handle.fileno())
        execute(resume=resume)


def execution_lock_held() -> bool:
    if not LOCK.is_file():
        return False
    with LOCK.open("a+", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
    return False


def behavior_signature(arrays: dict[str, np.ndarray], fault: int) -> str:
    digest = hashlib.sha256()
    for field in ("timed_out", "cycle_delta", "control_timeline_delta", "digest_xor", "probe_snapshot_xor", "probe_toggle_delta"):
        value = np.ascontiguousarray(arrays[field][fault])
        digest.update(field.encode() + b"\0")
        digest.update(str(value.dtype).encode() + b"\0")
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def freeze_dataset(state: dict[str, Any], site_rows: list[dict[str, Any]],
                   batches: dict[int, list[dict[str, Any]]], source_indices: np.ndarray) -> None:
    print("CONSOLIDATING AND VERIFYING LOCKED REPAIR_SITE_TEST DATASET", flush=True)
    arrays: dict[str, np.ndarray] = {
        "control_timeline_delta": np.zeros((FAULTS, VECTORS, 3), dtype=np.int16),
        "cycle_delta": np.zeros((FAULTS, VECTORS), dtype=np.int16),
        "digest_xor": np.zeros((FAULTS, VECTORS, 32), dtype=np.uint8),
        "timed_out": np.zeros((FAULTS, VECTORS), dtype=np.uint8),
        "external_detected": np.zeros((FAULTS, VECTORS), dtype=np.uint8),
        "probe_effect": np.zeros((FAULTS, VECTORS), dtype=np.uint8),
        "probe_snapshot_xor": np.zeros((FAULTS, VECTORS, SNAPSHOTS, PROBE_BITS // 8), dtype=np.uint8),
        "probe_toggle_count": np.zeros((FAULTS, VECTORS, PROBE_BITS), dtype=np.uint16),
        "probe_toggle_delta": np.zeros((FAULTS, VECTORS, PROBE_BITS), dtype=np.int16),
    }
    baseline_cycles = np.zeros((BATCHES, VECTORS), dtype=np.uint16)
    baseline_control = np.zeros((BATCHES, VECTORS, 3), dtype=np.int16)
    baseline_snapshots = np.zeros((BATCHES, VECTORS, SNAPSHOTS, PROBE_BITS // 8), dtype=np.uint8)
    baseline_toggles = np.zeros((BATCHES, VECTORS, PROBE_BITS), dtype=np.uint16)
    seen = np.zeros((FAULTS, VECTORS), dtype=np.uint8)
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
        local, summary = parse_batch_csv(csv_path, batch_id, batches[batch_id], source_indices)
        require(deterministic_npz(local) == batch_npz.read_bytes(), f"deterministic batch replay Batch {key}")
        fault_indices = local["fault_instance_index"]
        require(np.all(seen[fault_indices] == 0), f"duplicate cross-batch fault Batch {key}")
        seen[fault_indices] = 1
        for field in arrays:
            arrays[field][fault_indices] = local[field]
        baseline_cycles[batch_id] = local["baseline_cycles"]
        baseline_control[batch_id] = local["baseline_control_timeline"]
        baseline_snapshots[batch_id] = local["baseline_probe_snapshots"]
        baseline_toggles[batch_id] = local["baseline_probe_toggle_count"]
        baseline_records += summary["baseline_records"]
        enabled_records += summary["enabled_records"]
        raw_records[key] = record(csv_path)
        batch_records[key] = record(batch_npz)
    require(np.all(seen == 1), "missing consolidated samples")
    require(baseline_records == BASELINE_RECORDS and enabled_records == ENABLED_RECORDS, "record totals")
    require(np.all(baseline_cycles == EXPECTED_BASELINE_CYCLES), "baseline latency replay")
    require(np.all(baseline_cycles == baseline_cycles[0:1]), "cross-batch baseline cycles")
    require(np.all(baseline_control == baseline_control[0:1]), "cross-batch baseline control")
    require(np.all(baseline_snapshots == baseline_snapshots[0:1]), "cross-batch baseline snapshots")
    require(np.all(baseline_toggles == baseline_toggles[0:1]), "cross-batch baseline toggles")

    feature_arrays = {
        "baseline_control_timeline": baseline_control[0],
        "baseline_cycles": baseline_cycles[0],
        "baseline_probe_snapshots": baseline_snapshots[0],
        "baseline_probe_toggle_count": baseline_toggles[0],
        **arrays,
        "source_vector_index": source_indices.astype(np.int32),
        "vector_rank": np.arange(VECTORS, dtype=np.int32),
    }
    feature_payload = deterministic_npz(feature_arrays)
    freeze_or_verify(FEATURES, feature_payload)

    # Phase 1: response-only inference.  The site labels below belong to the
    # frozen closed catalog; the query's injected identity is never read as a
    # feature.  Exact ties are returned in full.
    print("RUNNING LOCKED RESPONSE-ONLY EXACT-SIGNATURE INFERENCE", flush=True)
    require(sha256(SELECTED_MODEL) == PINNED[SELECTED_MODEL], "model identity at inference")
    external_fault = np.any(arrays["external_detected"] != 0, axis=1)
    probe_fault = np.any(arrays["probe_effect"] != 0, axis=1)
    observable = external_fault | probe_fault
    rescued = (~external_fault) & probe_fault
    signatures = [behavior_signature(arrays, fault) for fault in range(FAULTS)]
    catalog_site = np.repeat(np.asarray([int(row["site_index"]) for row in site_rows], dtype=np.int32), 2)
    catalog_stuck = np.tile(np.asarray([0, 1], dtype=np.uint8), SITES)
    signature_groups: dict[str, list[int]] = {}
    for fault in np.flatnonzero(observable):
        signature_groups.setdefault(signatures[int(fault)], []).append(int(fault))

    candidate_count = np.zeros(FAULTS, dtype=np.int32)
    output_class = np.zeros(FAULTS, dtype=np.uint8)  # 0 normal-compatible, 1 unique, 2 ambiguous
    candidate_rows: list[dict[str, Any]] = []
    for query in range(FAULTS):
        members = signature_groups.get(signatures[query], []) if observable[query] else []
        sites = sorted({int(catalog_site[index]) for index in members})
        pairs = sorted({(int(catalog_site[index]), int(catalog_stuck[index])) for index in members})
        candidate_count[query] = len(sites)
        output_class[query] = 0 if not observable[query] else (1 if len(sites) == 1 else 2)
        candidate_rows.append({
            "query_rank": query,
            "output_class": ("NO_OBSERVED_ANOMALY" if output_class[query] == 0 else
                             "UNIQUE_SITE_IN_CATALOG" if output_class[query] == 1 else
                             "AMBIGUOUS_SITES_IN_CATALOG"),
            "behavior_signature_sha256": signatures[query] if observable[query] else "NORMAL_COMPATIBLE",
            "candidate_site_count": len(sites),
            "candidate_site_indices": ";".join(str(value) for value in sites),
            "candidate_site_polarities": ";".join(f"{site}:SA{stuck}" for site, stuck in pairs),
        })

    prediction_arrays = {
        "query_rank": np.arange(FAULTS, dtype=np.int32),
        "observable": observable.astype(np.uint8),
        "output_class": output_class,
        "candidate_site_count": candidate_count,
        "behavior_signature_sha256": np.vstack([
            np.frombuffer(bytes.fromhex(value), dtype=np.uint8) for value in signatures
        ]),
    }
    prediction_payload = deterministic_npz(prediction_arrays)
    candidate_payload = csv_payload(candidate_rows)
    freeze_or_verify(PREDICTIONS, prediction_payload)
    freeze_or_verify(CANDIDATES, candidate_payload)
    if PREDICTION_COMMITMENT.exists():
        prediction_commitment = load_json(PREDICTION_COMMITMENT)
        require(prediction_commitment.get("status") == "FROZEN BEFORE SCORING", "prediction commitment status")
        require(prediction_commitment.get("predictions") == record(PREDICTIONS), "prediction commitment binding")
        require(prediction_commitment.get("candidate_sets") == record(CANDIDATES), "candidate commitment binding")
    else:
        prediction_commitment = {
            "commitment_version": "CIRCUITSAGE-HMAC-V2.1-REPAIR-SITE-TEST-PREDICTION-COMMITMENT-12B3O-v1",
            "stage": STAGE,
            "status": "FROZEN BEFORE SCORING",
            "created_at": now(),
            "selected_measurement": SELECTED,
            "selected_model": SELECTED_MODEL_ID,
            "model": record(SELECTED_MODEL),
            "feature_dataset": record(FEATURES),
            "predictions": record(PREDICTIONS),
            "candidate_sets": record(CANDIDATES),
            "query_identity_fields_used": 0,
            "scoring_targets_opened_before_commitment": 0,
            "training_calls": 0,
            "threshold_changes": 0,
            "candidate_reselections": 0,
        }
        frozen_write(PREDICTION_COMMITMENT, canonical_json(prediction_commitment))
    if state.get("status") != "PASS":
        state["phase"] = "PREDICTIONS_COMMITTED_BEFORE_TRUTH"
        state["prediction_commitment"] = record(PREDICTION_COMMITMENT)
        state["updated_at"] = now()
        atomic_json(CHECKPOINT, state)
    committed_prediction_sha = sha256(PREDICTIONS)
    committed_candidates_sha = sha256(CANDIDATES)

    # Phase 2: only after the immutable prediction commitment exists, open the
    # ordered scoring truth and compute the single authorized evaluation.
    print("PREDICTIONS COMMITTED — OPENING SCORING TRUTH", flush=True)
    authorized_truth = load_json(TRUTH_COMMITMENT_3N)
    truth_payload = ("\n".join(
        f"{row['site_test_rank']}|{row['site_id']}|{row['site_index']}|{row['batch_id']}|{row['selector']}|SA0|SA1"
        for row in site_rows
    ) + "\n").encode()
    require(hashlib.sha256(truth_payload).hexdigest() == authorized_truth["commitment_sha256"], "truth commitment opening")
    true_site = catalog_site.copy()
    true_stuck = catalog_stuck.copy()
    target_arrays = {
        "fault_instance_index": np.arange(FAULTS, dtype=np.int32),
        "site_test_rank": np.repeat(np.arange(SITES, dtype=np.int32), 2),
        "site_index": true_site,
        "stuck_value": true_stuck,
    }
    target_payload = deterministic_npz(target_arrays)
    freeze_or_verify(TARGETS, target_payload)

    true_site_covered = np.zeros(FAULTS, dtype=np.uint8)
    exact_site = np.zeros(FAULTS, dtype=np.uint8)
    polarity_resolved = np.zeros(FAULTS, dtype=np.uint8)
    ambiguous = (candidate_count > 1).astype(np.uint8)
    signature_rows: list[dict[str, Any]] = []
    for fault in range(FAULTS):
        members = signature_groups.get(signatures[fault], []) if observable[fault] else []
        sites = {int(catalog_site[index]) for index in members}
        polarities = {int(catalog_stuck[index]) for index in members if int(catalog_site[index]) == int(true_site[fault])}
        true_site_covered[fault] = int(observable[fault] and int(true_site[fault]) in sites)
        exact_site[fault] = int(observable[fault] and sites == {int(true_site[fault])})
        polarity_resolved[fault] = int(exact_site[fault] and polarities == {int(true_stuck[fault])})
        signature_rows.append({
            "fault_instance_index": fault,
            "site_test_rank": fault // 2,
            "site_index": int(true_site[fault]),
            "stuck_value": int(true_stuck[fault]),
            "observable": int(observable[fault]),
            "candidate_site_count": int(candidate_count[fault]),
            "true_site_covered": int(true_site_covered[fault]),
            "exact_site": int(exact_site[fault]),
            "polarity_resolved": int(polarity_resolved[fault]),
            "behavior_signature_sha256": signatures[fault] if observable[fault] else "NORMAL_COMPATIBLE",
        })

    observed = np.flatnonzero(observable)
    unique = observed[candidate_count[observed] == 1]
    observable_counts = candidate_count[observed]
    detection_recall = float(np.mean(observable))
    coverage = float(np.mean(true_site_covered[observed])) if len(observed) else 0.0
    unique_top1 = float(np.mean(exact_site[unique])) if len(unique) else 0.0
    exact_rate = float(np.mean(exact_site))
    mean_candidates = float(np.mean(observable_counts)) if len(observable_counts) else float("inf")
    max_candidates = int(np.max(observable_counts)) if len(observable_counts) else SITES
    polarity_denominator = max(1, int(np.sum(exact_site[unique])))
    polarity_accuracy = float(np.sum(polarity_resolved[unique]) / polarity_denominator)

    rng = np.random.default_rng(12031317)
    bootstrap_indices = rng.integers(0, SITES, size=(BOOTSTRAP_REPLICATES, SITES), dtype=np.int32)
    per_site_detection = observable.reshape(SITES, 2).mean(axis=1)
    per_site_exact = exact_site.reshape(SITES, 2).mean(axis=1)
    per_site_coverage = np.ones(SITES, dtype=np.float64)
    for site_rank in range(SITES):
        rows = np.asarray([site_rank * 2, site_rank * 2 + 1], dtype=np.int32)
        visible = rows[observable[rows]]
        per_site_coverage[site_rank] = float(np.mean(true_site_covered[visible])) if len(visible) else 1.0
    detection_bootstrap = per_site_detection[bootstrap_indices].mean(axis=1)
    exact_bootstrap = per_site_exact[bootstrap_indices].mean(axis=1)
    coverage_bootstrap = per_site_coverage[bootstrap_indices].mean(axis=1)
    bootstrap_rows = [{
        "replicate": index,
        "detection_recall": f"{detection_bootstrap[index]:.10f}",
        "all_injected_exact_site_rate": f"{exact_bootstrap[index]:.10f}",
        "observable_candidate_set_coverage": f"{coverage_bootstrap[index]:.10f}",
    } for index in range(BOOTSTRAP_REPLICATES)]
    intervals = {
        "detection_recall": np.percentile(detection_bootstrap, [2.5, 97.5]).tolist(),
        "all_injected_exact_site_rate": np.percentile(exact_bootstrap, [2.5, 97.5]).tolist(),
        "observable_candidate_set_coverage": np.percentile(coverage_bootstrap, [2.5, 97.5]).tolist(),
    }

    frozen_targets = load_json(EXECUTION_CONTRACT)["frozen_acceptance_targets"]
    checks = {
        "fault_free_false_alarm_rate": 0.0 <= float(frozen_targets["fault_free_false_alarm_rate_max"]),
        "combined_detection_recall": detection_recall >= float(frozen_targets["combined_detection_recall_min"]),
        "candidate_set_coverage": coverage >= float(frozen_targets["observable_candidate_set_coverage_min"]),
        "unique_signature_top1_site": unique_top1 >= float(frozen_targets["unique_signature_top1_site_min"]),
        "ambiguous_false_unique_rate": 0.0 <= float(frozen_targets["ambiguous_false_unique_rate_max"]),
        "all_injected_exact_site_rate": exact_rate >= float(frozen_targets["all_injected_exact_site_rate_min"]),
        "mean_observable_candidate_sites": mean_candidates <= float(frozen_targets["mean_observable_candidate_sites_max"]),
        "maximum_observable_candidate_sites": max_candidates <= int(frozen_targets["maximum_observable_candidate_sites_max"]),
        "polarity_accuracy": polarity_accuracy >= float(frozen_targets["polarity_accuracy_given_correct_unique_site_min"]),
    }
    criterion_map = {name: "PASS" if value else "NOT_MET" for name, value in checks.items()}
    acceptance_status = "PASS" if all(checks.values()) else "NOT_MET"

    signature_payload = csv_payload(signature_rows)
    bootstrap_payload = csv_payload(bootstrap_rows)
    freeze_or_verify(SIGNATURES, signature_payload)
    freeze_or_verify(BOOTSTRAP, bootstrap_payload)
    metrics = {
        "metrics_version": "CIRCUITSAGE-HMAC-V2.1-LOCKED-REPAIR-SITE-TEST-METRICS-12B3O-v1",
        "stage": STAGE, "status": "FROZEN",
        "scope": "ONE LOCKED CLOSED-CATALOG REPAIR_SITE_TEST CAPTURE AND EVALUATION",
        "selected_measurement": SELECTED,
        "selected_model": SELECTED_MODEL_ID,
        "sites": SITES, "fault_instances": FAULTS, "vectors": VECTORS,
        "external_observable_faults": int(np.sum(external_fault)),
        "probe_observable_faults": int(np.sum(probe_fault)),
        "probe_rescued_faults": int(np.sum(rescued)),
        "combined_observable_faults": int(np.sum(observable)),
        "normal_compatible_faults": int(np.sum(~observable)),
        "fault_free_false_alarms": 0,
        "fault_free_false_alarm_rate": 0.0,
        "combined_all_injected_detection_recall": detection_recall,
        "observable_candidate_set_coverage": coverage,
        "unique_signature_top1_site": unique_top1,
        "ambiguous_false_unique_rate": 0.0,
        "all_injected_exact_site_rate": exact_rate,
        "observable_mrr": coverage,
        "mean_observable_candidate_sites": mean_candidates,
        "maximum_observable_candidate_sites": max_candidates,
        "polarity_accuracy_given_correct_unique_site": polarity_accuracy,
        "observable_faults": int(len(observed)),
        "unique_signature_faults": int(len(unique)),
        "ambiguous_signature_faults": int(np.sum(ambiguous)),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_ci_95": intervals,
        "acceptance_targets": frozen_targets,
        "criteria": criterion_map,
        "locked_evaluation_acceptance": acceptance_status,
        "prediction_commitment_before_truth": "PASS / VERIFIED",
        "training_calls": 0,
        "threshold_changes": 0,
        "candidate_reselections": 0,
        "independent_circuit_generalization": "NOT ESTABLISHED",
        "interpretation": "CLOSED-CATALOG SITE-HELD-OUT CONSISTENCY; TEST CATALOG CONSTRUCTION AND QUERY USE THE SAME FROZEN CAPTURE COHORT; NOT INDEPENDENT-CIRCUIT GENERALIZATION",
    }
    freeze_or_verify(METRICS, canonical_json(metrics))
    schema = {
        "schema_version": "CIRCUITSAGE-HMAC-V2.1-REPAIR-SITE-TEST-DATASET-12B3O-v1",
        "stage": STAGE, "status": "FROZEN",
        "format": "DETERMINISTIC FEATURE/TARGET/PREDICTION NPZ + CANDIDATE/SIGNATURE/BOOTSTRAP CSV",
        "features": {name: {"dtype": str(value.dtype), "shape": list(value.shape)} for name, value in sorted(feature_arrays.items())},
        "targets": {name: {"dtype": str(value.dtype), "shape": list(value.shape)} for name, value in sorted(target_arrays.items())},
        "predictions": {name: {"dtype": str(value.dtype), "shape": list(value.shape)} for name, value in sorted(prediction_arrays.items())},
        "sample_key": ["fault_instance_index", "vector_rank"],
        "snapshot_cycles_after_start": SNAPSHOT_CYCLES,
        "final_snapshot": "DONE_OR_TIMEOUT",
        "identity_exclusion": "site/fault/stuck/selector absent from feature NPZ",
        "prediction_commitment_before_scoring_truth": True,
        "source_partition": "REPAIR_SITE_TEST — CONSUMED AND FROZEN",
        "repair_train_rows": 0, "repair_calibration_rows": 0,
        "repair_site_test_enabled_rows": ENABLED_RECORDS,
        "dev_site_test_rows": 0, "validation_rows": 0, "holdout_rows": 0,
    }
    freeze_or_verify(SCHEMA, canonical_json(schema))

    if state.get("status") != "PASS":
        state["status"] = "PASS"
        state["phase"] = "CAPTURE_EVALUATION_FROZEN"
        state["finished_at"] = now()
        state["updated_at"] = now()
        state["prediction_commitment"] = record(PREDICTION_COMMITMENT)
        atomic_json(CHECKPOINT, state)
    manifest = {
        "manifest_version": "CIRCUITSAGE-HMAC-V2.1-REPAIR-SITE-TEST-EVALUATION-MANIFEST-12B3O-v1",
        "stage": STAGE, "status": "PASS", "authorization": record(AUTHORIZATION),
        "engine_lineage": ENGINE_LINEAGE,
        "checkpoint": record(CHECKPOINT), "raw_batch_csvs": raw_records, "per_batch_npz": batch_records,
        "outputs": {rel(path): record(path) for path in (
            FEATURES, TARGETS, SIGNATURES, PREDICTIONS, CANDIDATES,
            PREDICTION_COMMITMENT, METRICS, BOOTSTRAP, SCHEMA,
        )},
        "canonical_batches": BATCHES, "sites": SITES, "fault_instances": FAULTS,
        "vectors": VECTORS, "baseline_records": baseline_records, "enabled_records": enabled_records,
        "total_records": baseline_records + enabled_records,
        "missing_duplicate_unknown_baseline_failures": [int(np.sum(seen == 0)), 0, 0, 0],
        "prediction_commitment_before_truth": True,
        "model_objects_deserialized": 1, "training_calls": 0, "inference_calls": FAULTS,
        "threshold_changes": 0, "candidate_reselections": 0,
        "repair_train_access": 0, "repair_calibration_access": 0,
        "repair_site_test_sites_accessed": SITES,
        "repair_site_test_enabled_records": enabled_records,
        "dev_site_test_access": 0, "validation_access": 0, "holdout_access": 0,
        "frozen_rtl_modified": False, "canonical_netlists_modified": False,
    }
    freeze_or_verify(MANIFEST, canonical_json(manifest))
    audit = {
        "audit_version": "CIRCUITSAGE-HMAC-V2.1-REPAIR-SITE-TEST-CAPTURE-EVALUATION-FREEZE-12B3O-v1",
        "stage": STAGE, "status": "PASS",
        "capture_evaluation": "COMPLETED / FROZEN",
        "dataset_status": "FROZEN",
        "selected_measurement_model": [SELECTED, SELECTED_MODEL_ID],
        "batches_verified": f"{BATCHES}/{BATCHES}",
        "sites_faults_vectors": [SITES, FAULTS, VECTORS],
        "baseline_enabled_total_records": [baseline_records, enabled_records, baseline_records + enabled_records],
        "external_rescued_combined_observable_faults": [int(np.sum(external_fault)), int(np.sum(rescued)), int(np.sum(observable))],
        "combined_detection_recall": detection_recall,
        "observable_candidate_set_coverage": coverage,
        "unique_signature_top1_site": unique_top1,
        "all_injected_exact_site_rate": exact_rate,
        "mean_maximum_observable_candidate_sites": [mean_candidates, max_candidates],
        "polarity_accuracy_given_correct_unique_site": polarity_accuracy,
        "fault_free_false_alarms": 0,
        "locked_evaluation_acceptance": acceptance_status,
        "prediction_commitment_before_truth": "PASS / VERIFIED",
        "prediction_commitment": record(PREDICTION_COMMITMENT),
        "truth_commitment": record(TRUTH_COMMITMENT_3N),
        "deterministic_capture_prediction_scoring_replay": "PASS / EXACT",
        "training_threshold_change_candidate_reselection": [0, 0, 0],
        "model_objects_deserialized_inference_calls": [1, FAULTS],
        "repair_site_test": "CONSUMED / FROZEN / DO NOT REOPEN",
        "dev_site_test": "CONSUMED / NOT REOPENED",
        "validation_holdout_access": [0, 0],
        "independent_generalization": "NOT ESTABLISHED",
        "metrics": record(METRICS), "manifest": record(MANIFEST),
        "next_gate": "STAGE 12B-3P — REPAIR_SITE_TEST RESULT DISPOSITION AND V2.1 IMPROVEMENT FREEZE",
    }
    freeze_or_verify(AUDIT, canonical_json(audit))

    require(feature_payload == FEATURES.read_bytes(), "feature NPZ replay")
    require(target_payload == TARGETS.read_bytes(), "target NPZ replay")
    require(prediction_payload == PREDICTIONS.read_bytes(), "prediction NPZ replay")
    require(candidate_payload == CANDIDATES.read_bytes(), "candidate CSV replay")
    require(signature_payload == SIGNATURES.read_bytes(), "signature CSV replay")
    require(bootstrap_payload == BOOTSTRAP.read_bytes(), "bootstrap CSV replay")
    require(sha256(PREDICTIONS) == committed_prediction_sha, "prediction changed after commitment")
    require(sha256(CANDIDATES) == committed_candidates_sha, "candidate sets changed after commitment")
    for path in (PREDICTION_COMMITMENT, METRICS, SCHEMA, MANIFEST, AUDIT):
        require(canonical_json(load_json(path)) == path.read_bytes(), f"canonical JSON replay: {path.name}")
    for path, expected in PINNED.items():
        require(sha256(path) == expected, f"frozen input changed: {rel(path)}")

    print("\nSTAGE 12B-3O — LOCKED REPAIR_SITE_TEST CAPTURE, EVALUATION, AND DATASET FREEZE")
    print(f"{'Status':<76}: PASS")
    print(f"{'Capture / evaluation / dataset':<76}: COMPLETED / COMPLETED / FROZEN")
    print(f"{'Selected measurement / model':<76}: {SELECTED} / {SELECTED_MODEL_ID}")
    print(f"{'Batches / sites / faults / vectors':<76}: {BATCHES}/{BATCHES} / {SITES} / {FAULTS} / {VECTORS}")
    print(f"{'Baseline / enabled / total records':<76}: {baseline_records} / {enabled_records} / {baseline_records + enabled_records}")
    print(f"{'External / rescued / combined observable faults':<76}: {int(np.sum(external_fault))} / {int(np.sum(rescued))} / {int(np.sum(observable))}")
    print(f"{'Combined all-injected detection recall':<76}: {detection_recall:.8f}")
    print(f"{'Detection 95% site-bootstrap CI':<76}: [{intervals['detection_recall'][0]:.8f}, {intervals['detection_recall'][1]:.8f}]")
    print(f"{'Observable candidate-set coverage':<76}: {coverage:.8f}")
    print(f"{'Unique-signature top1 site':<76}: {unique_top1:.8f}")
    print(f"{'All-injected exact-site rate':<76}: {exact_rate:.8f}")
    print(f"{'Exact-site 95% site-bootstrap CI':<76}: [{intervals['all_injected_exact_site_rate'][0]:.8f}, {intervals['all_injected_exact_site_rate'][1]:.8f}]")
    print(f"{'Mean / maximum observable candidate sites':<76}: {mean_candidates:.4f} / {max_candidates}")
    print(f"{'Polarity accuracy | correct unique site':<76}: {polarity_accuracy:.8f}")
    print(f"{'Fault-free false alarms':<76}: 0")
    print(f"{'Prediction commitment before scoring truth':<76}: PASS / VERIFIED")
    print(f"{'Locked evaluation acceptance':<76}: {acceptance_status}")
    print(f"{'Training / threshold changes / candidate reselection':<76}: 0 / 0 / 0")
    print(f"{'DEV_SITE_TEST / VALIDATION / HOLDOUT access':<76}: 0 / 0 / 0")
    print(f"{'Predictions':<76}: {PREDICTIONS}")
    print(f"{'Predictions SHA':<76}: {sha256(PREDICTIONS)}")
    print(f"{'Metrics':<76}: {METRICS}")
    print(f"{'Metrics SHA':<76}: {sha256(METRICS)}")
    print(f"{'Manifest':<76}: {MANIFEST}")
    print(f"{'Manifest SHA':<76}: {sha256(MANIFEST)}")
    print(f"{'Audit':<76}: {AUDIT}")
    print(f"{'Audit SHA':<76}: {sha256(AUDIT)}")
    print(f"{'Next gate':<76}: STAGE 12B-3P — REPAIR_SITE_TEST RESULT DISPOSITION AND V2.1 IMPROVEMENT FREEZE")


def status() -> None:
    print("STAGE 12B-3O — LOCKED REPAIR_SITE_TEST STATUS")
    print(f"Execution lock held       : {'YES' if execution_lock_held() else 'NO'}")
    if AUDIT.is_file():
        audit = load_json(AUDIT)
        print(f"Status                    : {audit.get('status')} / FROZEN")
        print(f"Batches verified          : {audit.get('batches_verified')}")
        print(f"Detection recall          : {audit.get('combined_detection_recall')}")
        print(f"Candidate-set coverage    : {audit.get('observable_candidate_set_coverage')}")
        print(f"Exact-site rate           : {audit.get('all_injected_exact_site_rate')}")
        print(f"Evaluation acceptance     : {audit.get('locked_evaluation_acceptance')}")
        print(f"Prediction commitment     : {audit.get('prediction_commitment_before_truth')}")
        print(f"Audit                     : {AUDIT}")
        print(f"Audit SHA                 : {sha256(AUDIT)}")
        return
    if not CHECKPOINT.is_file():
        print("Status                    : NOT STARTED")
        return
    state = load_json(CHECKPOINT)
    completed = state.get("completed_batches", [])
    print(f"Status                    : {state.get('status')}")
    print(f"Completed batches         : {len(completed)}/{BATCHES}")
    if completed:
        print(f"Latest completed batch    : {max(completed):03d}")
    print(f"Checkpoint                : {CHECKPOINT}")
    print(f"Checkpoint SHA            : {sha256(CHECKPOINT)}")


def self_test() -> None:
    require(len(SNAPSHOT_CYCLES) == SNAPSHOTS - 1, "snapshot schedule")
    require(SNAPSHOT_CYCLES == sorted(set(SNAPSHOT_CYCLES)), "snapshot ordering")
    require(BASELINE_RECORDS == 4320 and ENABLED_RECORDS == 460416 and TOTAL_RECORDS == 464736, "record counts")
    require(SITES * 2 == FAULTS, "site/fault arithmetic")
    bits = snapshot_bits("1f", 5)
    require(bits.tolist() == [1, 1, 1, 1, 1], "snapshot parser")
    sample = {"a": np.arange(8, dtype=np.uint8), "b": np.arange(3, dtype=np.int32)}
    require(deterministic_npz(sample) == deterministic_npz(sample), "deterministic NPZ")
    generated = generate_testbench(
        0,
        [{"selector": 1, "site_id": "SELFTEST"}],
        *(Path("/tmp") / name for name in ("out.csv", "key.mem", "message.mem", "digest.mem", "source.mem")),
    )
    require('$fwrite(csv_fd,"\\n");' in generated, "SystemVerilog newline escape")
    require('$fwrite(csv_fd,"\n");' not in generated, "no literal newline inside SystemVerilog string")
    print("Stage 12B-3O self-test: PASS")


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
        locked_execute(resume=args.resume)


if __name__ == "__main__":
    main()
