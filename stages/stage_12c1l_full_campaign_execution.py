#!/usr/bin/env python3
"""Stage 12C-1L: full TRAIN/CALIBRATION campaign execution and dataset freeze.

Consumes only the frozen Stage 12C-1K authorization, the four frozen Stage
12C-1E generic netlists, and the frozen Stage 12C-1G vector corpus.  It builds
one full-site instrumented simulator per authorized family, validates every
family vector twice, executes the 999 frozen 64-site campaign batches, and
freezes separate model-facing feature and supervision-only target artifacts.

Execution is offline, sequential, checkpointed after every complete batch,
and resumable with ``--resume``.  TEST, VALIDATION, and HOLDOUT remain closed.
No model is loaded, trained, selected, or evaluated in this stage.
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
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import stage_12c1i_adapter_pilot_execution as pilot


STAGE = "12C-1L"
ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "config/v2_2"
RESULT = ROOT / "results/circuitsage_hmac_v2_12c1"
WORK = RESULT / "full_campaign_execution_12c1l"
BUILD = ROOT / "build/circuitsage_hmac_v2_12c1/full_campaign_12c1l"
RAW = WORK / "raw_batches"
PER_BATCH = WORK / "per_batch_npz"
MEMORY = WORK / "vector_memory"
LOCK = WORK / ".stage_12c1l.lock"

SOURCE_1K = ROOT / "stage_12c1k_full_campaign_authorization.py"
SOURCE_1I = ROOT / "stage_12c1i_adapter_pilot_execution.py"
EXECUTION_CONTRACT = CONFIG / "circuitsage_hmac_v2_2_full_campaign_execution_contract_12c1k.json"
AUTHORIZATION = CONFIG / "circuitsage_hmac_v2_2_full_campaign_execution_authorization_12c1k.json"
DATASET_CONTRACT = CONFIG / "circuitsage_hmac_v2_2_full_campaign_dataset_schema_12c1k.json"
AUTH_WORK = RESULT / "full_campaign_authorization_12c1k"
FAMILY_BUDGET = AUTH_WORK / "circuitsage_hmac_v2_2_full_campaign_family_budget_12c1k.csv"
EXECUTION_PLAN = AUTH_WORK / "circuitsage_hmac_v2_2_full_campaign_execution_plan_12c1k.csv"
PREFLIGHT_1K = AUTH_WORK / "circuitsage_hmac_v2_2_full_campaign_preflight_12c1k.json"
MANIFEST_1K = RESULT / "circuitsage_hmac_v2_2_full_campaign_authorization_manifest_12c1k.json"
AUDIT_1K = RESULT / "circuitsage_hmac_v2_2_full_campaign_execution_authorization_freeze_12c1k.json"

WORK_1G = RESULT / "portable_adapter_vector_generation_12c1g"
VECTORS_1G = WORK_1G / "circuitsage_hmac_v2_2_deterministic_transaction_vectors_12c1g.npz"
WORK_1E = RESULT / "train_calibration_synthesis_12c1e"
MANIFEST_1E = RESULT / "circuitsage_hmac_v2_2_elaboration_synthesis_manifest_12c1e.json"
AUDIT_1E = RESULT / "circuitsage_hmac_v2_2_elaboration_wrapper_generic_synthesis_freeze_12c1e.json"
AUDIT_1I = RESULT / "circuitsage_hmac_v2_2_adapter_functional_validation_pilot_execution_freeze_12c1i.json"

CHECKPOINT = WORK / "circuitsage_hmac_v2_2_full_campaign_checkpoint_12c1l.json"
VALIDATION_RESULTS = WORK / "circuitsage_hmac_v2_2_full_campaign_adapter_validation_12c1l.csv"
FAULT_CATALOG = WORK / "circuitsage_hmac_v2_2_full_campaign_fault_catalog_12c1l.csv"
FEATURES = WORK / "circuitsage_hmac_v2_2_full_campaign_features_12c1l.npz"
TARGETS = WORK / "circuitsage_hmac_v2_2_full_campaign_targets_12c1l.npz"
SIGNATURES = WORK / "circuitsage_hmac_v2_2_full_campaign_signature_summary_12c1l.csv"
METRICS = WORK / "circuitsage_hmac_v2_2_full_campaign_metrics_12c1l.json"
BOOTSTRAP = WORK / "circuitsage_hmac_v2_2_full_campaign_site_bootstrap_12c1l.csv"
BATCH_REGISTRY = WORK / "circuitsage_hmac_v2_2_full_campaign_batch_integrity_12c1l.csv"
SCHEMA = WORK / "circuitsage_hmac_v2_2_full_campaign_dataset_schema_12c1l.json"
REPORT = WORK / "circuitsage_hmac_v2_2_full_campaign_execution_report_12c1l.md"
MANIFEST = RESULT / "circuitsage_hmac_v2_2_full_campaign_dataset_manifest_12c1l.json"
AUDIT = RESULT / "circuitsage_hmac_v2_2_full_campaign_execution_dataset_freeze_12c1l.json"

PINNED = {
    SOURCE_1K: "bb2c58384489b043fef4978f8f6cd4b1618c6abd9adfefa7c71a1091bdf67ab8",
    SOURCE_1I: "735df97cbbf8ccd1097913d7044906c209a647e3290d056741c7880e26544f94",
    EXECUTION_CONTRACT: "cff6747c9bb469db89d3f5a1ca2f5f2ebb7f9a38a7b04b3280cd9dec0166713d",
    AUTHORIZATION: "c872be0cd7a12241c2c62a010da47e556608e201b0e2744bc8a19483392f68e5",
    EXECUTION_PLAN: "7e4f253eee3ebd5be204bdb352ab38b25a110f41d0db5451a8e1027ac12f8ee8",
    MANIFEST_1K: "a53fe05b665ae1d633d2f5e1da1cf0c2986572eda03cf0763048808f41b702bf",
    AUDIT_1K: "faf2ad35559e8547d913f6859cf3f1ab2a3e846f2fae134585b77f7bd60c68e1",
    VECTORS_1G: "0d8fce6c73c5d484fc9880ef97c0663b6dbfc9cbda73fcbd413f1970f418fa7b",
    MANIFEST_1E: "4d3d606e174ae06589e0e537486a15e042521fe9a02d49f9b4e1c596198476b6",
    AUDIT_1E: "3ac4f454173166ee6848e2617c7072742665e7166345ce5ffdb837f574d59220",
    AUDIT_1I: "cc96e5d7d11f9f75658365875c6521789416399b7bf09471f0fab334f09b2fa7",
}

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
# Bootstrap values are used only by the dependency-free self-test.  During a
# real execution verify_inputs() replaces them with the authoritative values
# from the SHA-anchored Stage 12C-1K family-budget registry.
EXPECTED_SITES = {
    "opentitan_hmac_sha256": 18392,
    "picorv32_cpu": 9665,
    "secworks_aes": 17000,
    "secworks_sha256": 18692,
}
BATCH_SITES = 64
TOTAL_SITES = sum(EXPECTED_SITES.values())
TOTAL_FAULTS = TOTAL_SITES * 2
TOTAL_VECTORS = sum(FULL_VECTORS.values())
TOTAL_BATCHES = sum(math.ceil(value / BATCH_SITES) for value in EXPECTED_SITES.values())
TOTAL_ENABLED = sum(EXPECTED_SITES[key] * 2 * FULL_VECTORS[key] for key in FAMILIES)
MAX_VECTORS = max(FULL_VECTORS.values())
MIN_FREE_GIB = 40
BOOTSTRAP_REPLICATES = 1000
BUILD_TIMEOUT_SECONDS = 21600
BATCH_TIMEOUT_SECONDS = 86400
FUTURE_BRAND = "Faultiva 1.0 — Hybrid VLSI Fault Intelligence"

CSV_FIELDS = list(pilot.CSV_FIELDS)
FAULT_FIELDS = list(pilot.FAULT_FIELDS)
BATCH_FIELDS = [
    "global_batch_id", "family_id", "partition", "family_batch_id",
    "site_rank_start", "site_count", "fault_count", "vector_count",
    "baseline_records", "enabled_records", "observable_faults",
    "raw_csv", "raw_csv_sha256", "batch_npz", "batch_npz_sha256", "status",
]


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


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as stream:
        stream.write(canonical_json(value))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def write_or_verify(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(path.read_bytes() == payload, f"generated-file replay mismatch: {rel(path)}")
        return
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def frozen_write(path: Path, payload: bytes) -> None:
    require(not path.exists(), f"refusing to overwrite frozen output: {rel(path)}")
    write_or_verify(path, payload)


def deterministic_npz_bytes(arrays: dict[str, np.ndarray]) -> bytes:
    return pilot.deterministic_npz(arrays)


def write_npz_or_verify(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Stream a deterministic ZIP_STORED NPZ without duplicating it in RAM."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with np.load(path, allow_pickle=False) as archive:
            require(set(archive.files) == set(arrays), f"NPZ members: {rel(path)}")
            for name in sorted(arrays):
                require(np.array_equal(archive[name], arrays[name]), f"NPZ replay: {rel(path)}::{name}")
        return
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
            for name in sorted(arrays):
                info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_STORED
                info.create_system = 3
                info.external_attr = 0o600 << 16
                with archive.open(info, "w", force_zip64=True) as member:
                    np.lib.format.write_array(member, np.ascontiguousarray(arrays[name]), allow_pickle=False)
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


def verify_manifest_outputs(manifest: dict[str, Any], label: str) -> None:
    outputs = manifest.get("outputs")
    require(isinstance(outputs, dict) and outputs, f"{label} outputs")
    for name, item in sorted(outputs.items()):
        safe_record(item, f"{label} output {name}")


def load_vectors() -> dict[str, np.ndarray]:
    with np.load(VECTORS_1G, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    required = {
        "family_ids", "family_index", "partition_index", "transaction_id",
        "family_vector_index", "pilot_mask", "stimulus_class", "payload",
        "payload_valid_byte_mask", "control", "request_sha256",
    }
    require(set(arrays) == required, "12C-1G vector members")
    require(arrays["payload"].shape == (TOTAL_VECTORS, 256), "vector payload shape")
    require(arrays["control"].shape == (TOTAL_VECTORS, 8), "vector control shape")
    require(arrays["family_ids"].tolist() == [value.encode() for value in FAMILIES], "vector family order")
    for family_number, family_id in enumerate(FAMILIES):
        indices = np.flatnonzero(arrays["family_index"] == family_number)
        require(indices.size == FULL_VECTORS[family_id], f"full vectors: {family_id}")
        require(np.array_equal(arrays["family_vector_index"][indices], np.arange(indices.size)), f"vector order: {family_id}")
        require(int(np.sum(arrays["control"][indices, 7] == 1)) == 1, f"timeout canary: {family_id}")
        require(int(arrays["control"][indices[-1], 7]) == 1, f"last-vector timeout canary: {family_id}")
    return arrays


def verify_inputs() -> tuple[dict[str, np.ndarray], list[dict[str, str]], dict[str, int]]:
    print("STAGE 12C-1L — FULL TRAIN/CALIBRATION FAULT-CAMPAIGN EXECUTION")
    print("FROZEN INPUT VERIFICATION")
    for path, expected in PINNED.items():
        require(path.is_file(), f"missing frozen input: {rel(path)}")
        require(sha256(path) == expected, f"frozen input SHA: {path.name}")
        print(f"  {path.name:<112}: OK", flush=True)

    contract = load_json(EXECUTION_CONTRACT)
    authorization = load_json(AUTHORIZATION)
    dataset = load_json(DATASET_CONTRACT)
    manifest_1k = load_json(MANIFEST_1K)
    audit_1k = load_json(AUDIT_1K)
    manifest_1e = load_json(MANIFEST_1E)
    audit_1e = load_json(AUDIT_1E)
    audit_1i = load_json(AUDIT_1I)
    for path, value in ((EXECUTION_CONTRACT, contract), (AUTHORIZATION, authorization),
                        (DATASET_CONTRACT, dataset), (MANIFEST_1K, manifest_1k),
                        (AUDIT_1K, audit_1k)):
        require(path.read_bytes() == canonical_json(value), f"canonical predecessor JSON: {path.name}")
    require(manifest_1k.get("status") == "PASS" and audit_1k.get("status") == "PASS", "12C-1K freeze")
    require(manifest_1k.get("stage_source", {}).get("sha256") == PINNED[SOURCE_1K], "12C-1K source anchor")
    require(audit_1k.get("manifest_record", {}).get("sha256") == PINNED[MANIFEST_1K], "12C-1K manifest anchor")
    require(audit_1k.get("full_campaign_execution") == "AUTHORIZED / NOT STARTED", "full-campaign authorization")
    require(audit_1k.get("model_training_selection_inference") == "NOT AUTHORIZED / NOT AUTHORIZED / NOT AUTHORIZED", "model boundary")
    require(audit_1k.get("independent_test_validation_holdout_access") == [0, 0, 0], "protected access")
    require(contract.get("status") == "FROZEN" and contract.get("planned_batches") == TOTAL_BATCHES, "execution contract")
    require(contract.get("maximum_enabled_transactions") == TOTAL_ENABLED, "enabled-transaction ceiling")
    require(contract.get("network_requirement") == "NONE / OFFLINE EXECUTION", "offline contract")
    require(authorization.get("full_train_calibration_campaign") == "AUTHORIZED / NOT STARTED", "authorization object")
    require(dataset.get("identity_firewall") == "FAULT IDENTITY ABSENT FROM MODEL-FACING FEATURES", "identity firewall")
    verify_manifest_outputs(manifest_1k, "12C-1K")
    require(manifest_1e.get("status") == "PASS" and audit_1e.get("status") == "PASS", "12C-1E freeze")
    require(audit_1e.get("deterministic_replay") == "PASS / BYTE-EXACT / 4 OF 4", "netlist replay")
    require(audit_1i.get("adapter_functional_validation") == "PASS / 4 OF 4 / BYTE-EXACT REPLAY", "adapter validation lineage")
    verify_manifest_outputs(manifest_1e, "12C-1E")

    family_rows = read_csv(FAMILY_BUDGET)
    plan = read_csv(EXECUTION_PLAN)
    require([row["family_id"] for row in family_rows] == list(FAMILIES), "family budget order")
    require(len(plan) == TOTAL_BATCHES, "execution-plan batch count")
    expected_global = list(range(TOTAL_BATCHES))
    require([int(row["global_batch_id"]) for row in plan] == expected_global, "execution-plan global order")
    family_max: dict[str, int] = {}
    authoritative_sites = {row["family_id"]: int(row["maximum_sites"]) for row in family_rows}
    require(set(authoritative_sites) == set(FAMILIES), "family-budget identifiers")
    require(sum(authoritative_sites.values()) == TOTAL_SITES, "authoritative site total")
    require(sum(authoritative_sites[family] * 2 * FULL_VECTORS[family] for family in FAMILIES) == TOTAL_ENABLED,
            "authoritative enabled-transaction total")
    EXPECTED_SITES.clear()
    EXPECTED_SITES.update(authoritative_sites)
    for row in family_rows:
        family_id = row["family_id"]
        require(int(row["maximum_faults"]) == EXPECTED_SITES[family_id] * 2, f"fault budget: {family_id}")
        require(int(row["full_vectors"]) == FULL_VECTORS[family_id], f"vector budget: {family_id}")
        family_max[family_id] = int(row["maximum_sites"])
    for family_id in FAMILIES:
        rows = [row for row in plan if row["family_id"] == family_id]
        require(len(rows) == math.ceil(EXPECTED_SITES[family_id] / BATCH_SITES), f"family batches: {family_id}")
        require(sum(int(row["site_count_maximum"]) for row in rows) == EXPECTED_SITES[family_id], f"plan sites: {family_id}")
        require(sum(int(row["maximum_enabled_transactions"]) for row in rows) == EXPECTED_SITES[family_id] * 2 * FULL_VECTORS[family_id], f"plan transactions: {family_id}")

    arrays = load_vectors()
    manifest_outputs = manifest_1e["outputs"]
    for family_id in FAMILIES:
        family_dir = WORK_1E / "families" / family_id
        for path in (family_dir / "family_result_12c1e.json",
                     family_dir / f"{family_id}_generic_12c1e.json",
                     family_dir / f"{family_id}_generic_12c1e.v"):
            item = manifest_outputs.get(rel(path))
            require(isinstance(item, dict), f"12C-1E manifest record: {family_id}/{path.name}")
            safe_record(item, f"frozen netlist {family_id}/{path.name}")
    require(shutil.which("yosys") is not None, "Yosys not found")
    require(shutil.which("verilator") is not None, "Verilator not found")
    require(Path("/usr/bin/time").is_file(), "/usr/bin/time not found")
    free_gib = shutil.disk_usage(ROOT).free / 1024**3
    require(free_gib >= MIN_FREE_GIB, f"insufficient disk: {free_gib:.2f} GiB; {MIN_FREE_GIB} GiB required")
    print("  Authorization, plan, full vectors, four netlists and protected partitions                      : PASS", flush=True)
    return arrays, plan, family_max


def family_indices(arrays: dict[str, np.ndarray], family_id: str) -> np.ndarray:
    return np.flatnonzero(arrays["family_index"] == FAMILIES.index(family_id))


def vector_material(arrays: dict[str, np.ndarray], family_id: str) -> dict[str, Any]:
    indices = family_indices(arrays, family_id)
    payloads = arrays["payload"][indices].astype(np.uint8, copy=False)
    controls = arrays["control"][indices].astype(np.uint32, copy=False)
    expected: list[bytes] = []
    for slot in range(indices.size):
        if int(controls[slot, 7]) == 1:
            expected.append(bytes(32))
        else:
            expected.append(pilot.expected_response(family_id, payloads[slot].tobytes(), controls[slot]))
    return {
        "indices": indices.astype(np.int32), "payload": payloads, "control": controls,
        "transaction_id": arrays["transaction_id"][indices].astype(np.int32), "expected": expected,
    }


def configure_pilot_helpers(family_id: str, site_count: int) -> None:
    pilot.ROOT = ROOT
    pilot.MEMORY = MEMORY
    pilot.SITES_PER_FAMILY = site_count
    pilot.PILOT_VECTORS = dict(FULL_VECTORS)
    pilot.VALIDATION_VECTORS = dict(FULL_VECTORS)


def full_testbench(family_id: str, site_count: int) -> str:
    configure_pilot_helpers(family_id, site_count)
    text = pilot.generate_testbench(family_id)
    text = text.replace("_12c1i", "_12c1l")
    text = text.replace('mode=="PILOT"', 'mode=="FULL"')
    text = text.replace("V22_PILOT_FAMILY", "V22_FULL_FAMILY")
    text = text.replace("V22_PILOT_BATCH_RESULT", "V22_FULL_BATCH_RESULT")
    old = "if(timeout_result||protocol_result||unknown_result||(response_result!==expected_vectors[vi])) failures=failures+1;"
    new = "if(control7[vi]!=0)begin if(!timeout_result||unknown_result||protocol_result)failures=failures+1;end else if(timeout_result||protocol_result||unknown_result||(response_result!==expected_vectors[vi]))failures=failures+1;"
    require(text.count(old) == 2, f"full-testbench baseline guard: {family_id}")
    position = text.rfind(old)
    require(position >= 0, f"full-testbench campaign branch: {family_id}")
    return text[:position] + new + text[position + len(old):]


def full_hmac_adapter_sv() -> str:
    """Extend the validated pilot adapter to contracted partial-byte messages."""
    text = pilot.hmac_adapter_sv()
    declaration = "  logic [31:0] user_fifo_data,hmac_fifo_data,selected_fifo_data;"
    require(text.count(declaration) == 1, "HMAC user-mask declaration anchor")
    text = text.replace(
        declaration,
        declaration + " logic [3:0] user_fifo_mask;",
        1,
    )
    data_assignment = "  assign user_fifo_data=message_q[2047-(msg_word_index*32)-:32];"
    require(text.count(data_assignment) == 1, "HMAC user-mask assignment anchor")
    text = text.replace(
        data_assignment,
        data_assignment + "\n"
        "  always_comb begin\n"
        "    if((msg_word_index+1<message_word_count)||(message_length_q[1:0]==0)) user_fifo_mask=4'hf;\n"
        "    else case(message_length_q[1:0]) 2'd1:user_fifo_mask=4'h8;2'd2:user_fifo_mask=4'hc;default:user_fifo_mask=4'he;endcase\n"
        "  end",
        1,
    )
    fifo_write = "fifo_mask_mem[fifo_wptr]<=4'hf;"
    require(text.count(fifo_write) == 1, "HMAC FIFO mask-write anchor")
    text = text.replace(fifo_write, "fifo_mask_mem[fifo_wptr]<=hmac_fifo_push?4'hf:user_fifo_mask;", 1)
    word_count = "message_word_count<=message_length_i>>2;"
    require(text.count(word_count) == 1, "HMAC message-word-count anchor")
    text = text.replace(word_count, "message_word_count<=(message_length_i+16'd3)>>2;", 1)
    return text


def verify_record(item: dict[str, Any], label: str) -> Path:
    return safe_record(item, label)


def prepare_family(state: dict[str, Any], family_id: str, arrays: dict[str, np.ndarray], site_count: int) -> tuple[list[dict[str, Any]], Path]:
    family_state = state["families"].setdefault(family_id, {})
    source_json = WORK_1E / "families" / family_id / f"{family_id}_generic_12c1e.json"
    configure_pilot_helpers(family_id, site_count)
    sites = pilot.enumerate_sites(family_id, source_json)
    require(len(sites) == site_count, f"eligible site count: {family_id}")
    if family_state.get("build_status") == "PASS":
        for key in ("derived_json", "derived_verilog", "testbench", "binary"):
            verify_record(family_state[key], f"resume {family_id} {key}")
        print(f"{family_id}: BUILD CHECKPOINT PASS", flush=True)
        return sites, ROOT / family_state["binary"]["path"]

    print(f"{family_id}: DERIVE {site_count:,}-SITE INSTRUMENTED NETLIST", flush=True)
    family_build = BUILD / family_id
    family_raw = RAW / family_id
    family_memory = MEMORY / family_id
    for directory in (family_build, family_raw, family_memory):
        directory.mkdir(parents=True, exist_ok=True)
    derived_json = family_build / f"{family_id}_full_instrumented_12c1l.json"
    derived_verilog = family_build / f"{family_id}_full_instrumented_12c1l.v"
    write_or_verify(derived_json, pilot.instrument_netlist(family_id, source_json, sites))
    yosys_log = family_raw / "yosys_instrumented_netlist.log"
    yosys = shutil.which("yosys")
    require(yosys is not None, "Yosys unavailable")
    command_text = f"read_json {derived_json.resolve()}; hierarchy -check -top {TOPS[family_id]}; write_verilog -noattr {derived_verilog.resolve()}"
    require(pilot.run_logged([yosys, "-p", command_text], yosys_log, BUILD_TIMEOUT_SECONDS) == 0,
            f"{family_id} instrumented netlist generation; inspect {rel(yosys_log)}")
    require(derived_verilog.is_file() and derived_verilog.stat().st_size > 0, f"instrumented Verilog: {family_id}")

    material = vector_material(arrays, family_id)
    for path, payload in pilot.memory_payloads(family_id, material, family_memory).items():
        write_or_verify(path, payload)
    testbench = family_build / f"tb_v22_{family_id}_12c1l.sv"
    write_or_verify(testbench, full_testbench(family_id, site_count).encode())

    sources: list[Path] = [derived_verilog]
    include_files: list[Path] = []
    include_dirs: list[Path] = []
    if family_id == "opentitan_hmac_sha256":
        require(pilot.HMAC_BRIDGE_SOURCE.is_file(), f"missing HMAC bridge: {rel(pilot.HMAC_BRIDGE_SOURCE)}")
        require(sha256(pilot.HMAC_BRIDGE_SOURCE) == pilot.HMAC_BRIDGE_SHA256, "HMAC bridge SHA")
        bridge = family_build / "v22_hmac_bridge_12c1l.sv"
        adapter = family_build / "v22_hmac_adapter_12c1l.sv"
        write_or_verify(bridge, pilot.transform_hmac_bridge(pilot.HMAC_BRIDGE_SOURCE.read_text(encoding="utf-8")).encode())
        write_or_verify(adapter, full_hmac_adapter_sv().encode())
        prim_sources = [pilot.find_unique_source(family_id, name) for name in (
            "prim_sha2_pkg.sv", "prim_sha2_pad.sv", "prim_sha2.sv", "prim_sha2_32.sv"
        )]
        prim_assert = pilot.find_unique_source(family_id, "prim_assert.sv")
        include_files = [prim_assert]
        include_dirs = [prim_assert.parent]
        sources = [prim_sources[0], derived_verilog, *prim_sources[1:], bridge, adapter]

    top = f"tb_v22_{family_id}_12c1l"
    obj_dir = family_build / "obj_dir"
    build_log = family_raw / "verilator_build.log"
    verilator = shutil.which("verilator")
    require(verilator is not None, "Verilator unavailable")
    command = [
        verilator, "--binary", "--timing", "--assert", "-Wall", "-Wno-fatal",
        "-Wno-DECLFILENAME", "-Wno-PINMISSING", "--error-limit", "0", "-j", "1",
        "--Mdir", str(obj_dir), "--top-module", top,
        *[f"-I{path}" for path in include_dirs], *[str(path) for path in sources], str(testbench),
    ]
    print(f"{family_id}: BUILD", flush=True)
    require(pilot.run_logged(command, build_log, BUILD_TIMEOUT_SECONDS) == 0,
            f"{family_id} Verilator build; inspect {rel(build_log)}")
    binary = obj_dir / f"V{top}"
    require(binary.is_file() and os.access(binary, os.X_OK), f"simulation binary: {family_id}")
    family_state.update({
        "build_status": "PASS", "built_at": now(), "sites": len(sites),
        "source_json": record(source_json), "derived_json": record(derived_json),
        "derived_verilog": record(derived_verilog), "testbench": record(testbench),
        "yosys_log": record(yosys_log), "build_log": record(build_log), "binary": record(binary),
        "compile_sources": [record(path) for path in sources],
        "include_files": [record(path) for path in include_files],
    })
    state["updated_at"] = now()
    atomic_json(CHECKPOINT, state)
    return sites, binary


def parse_validation(path: Path, family_id: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = read_csv(path)
    count = FULL_VECTORS[family_id]
    require(len(rows) == count, f"validation row count: {family_id}")
    parsed: list[dict[str, Any]] = []
    failures = 0
    for index, row in enumerate(rows):
        require(row["family_id"] == family_id and row["mode"] == "VALIDATE", f"validation identity: {family_id}/{index}")
        require(row["run_type"] == "BASELINE" and int(row["vector_rank"]) == index, f"validation order: {family_id}/{index}")
        timeout, protocol, unknown = int(row["timed_out"]), int(row["protocol_error"]), int(row["unknown"])
        is_canary = index == count - 1
        passed = unknown == 0 and protocol == 0 and ((timeout == 1) if is_canary else (timeout == 0 and row["actual_response"] == row["expected_response"]))
        failures += int(not passed)
        parsed.append({
            "family_id": family_id, "partition": PARTITIONS[family_id], "vector_rank": index,
            "transaction_id": int(row["transaction_id"]), "timeout_canary": "YES" if is_canary else "NO",
            "cycles": int(row["cycles"]), "timed_out": timeout, "protocol_error": protocol,
            "unknown": unknown, "expected_response": row["expected_response"],
            "actual_response": row["actual_response"], "result": "PASS" if passed else "FAIL",
        })
    require(failures == 0, f"{family_id} full-vector validation failed ({failures}); inspect {rel(path)}")
    return parsed, {"family_id": family_id, "status": "PASS", "vectors": count,
                    "timeout_canaries": 1, "failures": 0}


def validate_family(state: dict[str, Any], family_id: str, binary: Path) -> list[dict[str, Any]]:
    family_state = state["families"][family_id]
    if family_state.get("validation_status") == "PASS":
        first = verify_record(family_state["validation_csv_a"], f"resume validation {family_id} A")
        second = verify_record(family_state["validation_csv_b"], f"resume validation {family_id} B")
        require(first.read_bytes() == second.read_bytes(), f"validation replay: {family_id}")
        rows, _ = parse_validation(first, family_id)
        print(f"{family_id}: FULL-VECTOR VALIDATION CHECKPOINT PASS", flush=True)
        return rows
    family_raw = RAW / family_id
    outputs: list[Path] = []
    parsed_rows: list[list[dict[str, Any]]] = []
    for replay in ("a", "b"):
        csv_path = family_raw / f"full_vector_validation_{replay}.csv"
        log = family_raw / f"full_vector_validation_{replay}.log"
        resources = family_raw / f"full_vector_validation_{replay}_resources.log"
        print(f"{family_id}: FULL-VECTOR VALIDATION REPLAY {replay.upper()}", flush=True)
        require(pilot.run_logged([str(binary), "+MODE=VALIDATE", f"+CSV={csv_path.resolve()}"],
                                 log, BATCH_TIMEOUT_SECONDS, resources) == 0,
                f"{family_id} validation replay {replay}; inspect {rel(log)}")
        require("V22_ADAPTER_VALIDATION_RESULT=PASS" in log.read_text(errors="replace"), f"validation PASS token: {family_id}")
        parsed, _ = parse_validation(csv_path, family_id)
        outputs.append(csv_path)
        parsed_rows.append(parsed)
    require(outputs[0].read_bytes() == outputs[1].read_bytes(), f"byte-exact full-vector replay: {family_id}")
    family_state.update({
        "validation_status": "PASS", "validated_at": now(),
        "validation_csv_a": record(outputs[0]), "validation_csv_b": record(outputs[1]),
        "validation_replay": "PASS / BYTE-EXACT / ALL FULL VECTORS",
    })
    state["updated_at"] = now()
    atomic_json(CHECKPOINT, state)
    return parsed_rows[0]


def parse_batch(path: Path, family_id: str, batch_id: int, site_start: int,
                site_count: int) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    rows = read_csv(path)
    vectors = FULL_VECTORS[family_id]
    require(len(rows) == vectors + site_count * 2 * vectors, f"full row count: {family_id}/{batch_id}")
    baseline_rows = rows[:vectors]
    baseline_response = np.zeros((vectors, 32), dtype=np.uint8)
    baseline_cycles = np.zeros(vectors, dtype=np.int32)
    baseline_timeout = np.zeros(vectors, dtype=np.uint8)
    for vector, row in enumerate(baseline_rows):
        require(row["family_id"] == family_id and row["mode"] == "FULL", "full baseline identity")
        require(row["run_type"] == "BASELINE" and int(row["vector_rank"]) == vector, "full baseline order")
        timeout, protocol, unknown = int(row["timed_out"]), int(row["protocol_error"]), int(row["unknown"])
        is_canary = vector == vectors - 1
        require(unknown == 0 and protocol == 0, "full baseline protocol")
        require((timeout == 1) if is_canary else (timeout == 0 and row["actual_response"] == row["expected_response"]), "full baseline oracle")
        baseline_response[vector] = np.frombuffer(bytes.fromhex(row["actual_response"]), dtype=np.uint8)
        baseline_cycles[vector] = int(row["cycles"])
        baseline_timeout[vector] = timeout

    faults = site_count * 2
    response_xor = np.zeros((faults, vectors, 32), dtype=np.uint8)
    cycle_delta = np.zeros((faults, vectors), dtype=np.int32)
    timed_out = np.zeros((faults, vectors), dtype=np.uint8)
    protocol_error = np.zeros((faults, vectors), dtype=np.uint8)
    seen = np.zeros((faults, vectors), dtype=np.uint8)
    unknown_count = 0
    for row in rows[vectors:]:
        require(row["family_id"] == family_id and row["mode"] == "FULL" and row["run_type"] == "ENABLED", "full enabled identity")
        site, stuck, vector = int(row["site_rank"]), int(row["stuck_value"]), int(row["vector_rank"])
        require(site_start <= site < site_start + site_count and stuck in (0, 1) and 0 <= vector < vectors, "full enabled coordinates")
        fault = (site - site_start) * 2 + stuck
        require(seen[fault, vector] == 0, "duplicate full record")
        seen[fault, vector] = 1
        actual = np.frombuffer(bytes.fromhex(row["actual_response"]), dtype=np.uint8)
        response_xor[fault, vector] = actual ^ baseline_response[vector]
        cycle_delta[fault, vector] = int(row["cycles"]) - int(baseline_cycles[vector])
        timed_out[fault, vector] = int(row["timed_out"])
        protocol_error[fault, vector] = int(row["protocol_error"])
        unknown_count += int(row["unknown"])
    require(np.all(seen == 1), f"full sample coverage: {family_id}/{batch_id}")
    require(unknown_count == 0, f"unknown response records: {family_id}/{batch_id}")
    detected = (np.any(response_xor != 0, axis=2) | (cycle_delta != 0) |
                (timed_out != baseline_timeout[None, :]) | (protocol_error != 0)).astype(np.uint8)
    arrays = {
        "response_xor": response_xor, "cycle_delta": cycle_delta,
        "timed_out": timed_out, "protocol_error": protocol_error, "detected": detected,
        "site_rank": np.repeat(np.arange(site_start, site_start + site_count, dtype=np.int32), 2),
        "stuck_value": np.tile(np.asarray([0, 1], dtype=np.uint8), site_count),
        "baseline_response": baseline_response, "baseline_cycles": baseline_cycles,
    }
    summary = {
        "family_id": family_id, "batch_id": batch_id, "site_start": site_start,
        "sites": site_count, "faults": faults, "full_vectors": vectors,
        "baseline_records": vectors, "enabled_records": faults * vectors,
        "observable_faults": int(np.sum(np.any(detected != 0, axis=1))),
        "unknown_records": 0, "baseline_failures": 0,
    }
    return arrays, summary


def run_batches(state: dict[str, Any], family_id: str, binary: Path,
                plan: list[dict[str, str]]) -> None:
    family_state = state["families"][family_id]
    completed = set(int(value) for value in family_state.get("completed_batches", []))
    family_state.setdefault("batches", {})
    family_plan = [row for row in plan if row["family_id"] == family_id]
    for row in family_plan:
        global_batch = int(row["global_batch_id"])
        local_batch = int(row["family_batch_id"])
        site_start = int(row["site_rank_start"])
        site_count = int(row["site_count_maximum"])
        key = f"{global_batch:04d}"
        if global_batch in completed:
            item = family_state["batches"][key]
            verify_record(item["csv"], f"resume full CSV {key}")
            verify_record(item["npz"], f"resume full NPZ {key}")
            print(f"{family_id} Batch {local_batch:03d}/{len(family_plan)-1:03d}: CHECKPOINT PASS", flush=True)
            continue
        batch_dir = RAW / family_id / f"batch_{local_batch:03d}"
        batch_dir.mkdir(parents=True, exist_ok=True)
        csv_path = batch_dir / f"full_batch_{local_batch:03d}.csv"
        log = batch_dir / "simulation.log"
        resources = batch_dir / "resources.log"
        print(f"{family_id} Batch {local_batch:03d}: SIMULATE sites={site_start}-{site_start + site_count - 1}", flush=True)
        command = [str(binary), "+MODE=FULL", f"+CSV={csv_path.resolve()}",
                   f"+BATCH_ID={global_batch}", f"+SITE_START={site_start}", f"+SITE_COUNT={site_count}"]
        require(pilot.run_logged(command, log, BATCH_TIMEOUT_SECONDS, resources) == 0,
                f"full simulation {family_id}/{local_batch}; inspect {rel(log)}")
        require("V22_FULL_BATCH_RESULT=PASS" in log.read_text(errors="replace"), f"full PASS token: {family_id}/{local_batch}")
        batch_arrays, summary = parse_batch(csv_path, family_id, global_batch, site_start, site_count)
        require(summary["enabled_records"] == int(row["maximum_enabled_transactions"]), f"plan enabled count: {family_id}/{local_batch}")
        batch_npz = PER_BATCH / family_id / f"full_batch_{local_batch:03d}.npz"
        write_or_verify(batch_npz, deterministic_npz_bytes(batch_arrays))
        family_state["batches"][key] = {
            "status": "PASS", "completed_at": now(), "global_batch_id": global_batch,
            "local_batch_id": local_batch, "csv": record(csv_path), "npz": record(batch_npz),
            "simulation_log": record(log), "resource_log": record(resources), "summary": summary,
        }
        completed.add(global_batch)
        family_state["completed_batches"] = sorted(completed)
        state["completed_batches"] = sorted(set(int(value) for value in state.get("completed_batches", [])) | {global_batch})
        state["enabled_transactions_completed"] = sum(
            int(item["summary"]["enabled_records"])
            for value in state["families"].values() for item in value.get("batches", {}).values()
        )
        state["updated_at"] = now()
        atomic_json(CHECKPOINT, state)
        print(f"{family_id} Batch {local_batch:03d}: PASS enabled={summary['enabled_records']}", flush=True)
    require(len(completed) == len(family_plan), f"incomplete full campaign: {family_id}")
    family_state["campaign_status"] = "PASS"
    state["updated_at"] = now()
    atomic_json(CHECKPOINT, state)


def signature_digest(family_id: str, response_xor: np.ndarray, cycle_delta: np.ndarray,
                     timed_out: np.ndarray, protocol_error: np.ndarray) -> str:
    return pilot.signature_digest(family_id, response_xor, cycle_delta, timed_out, protocol_error)


def percentile_interval(values: np.ndarray) -> tuple[float, float]:
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def consolidate_and_freeze(state: dict[str, Any], arrays_1g: dict[str, np.ndarray],
                           plan: list[dict[str, str]], validation_rows: list[dict[str, Any]]) -> None:
    require(len(state.get("completed_batches", [])) == TOTAL_BATCHES, "full campaign batches incomplete")
    response_xor = np.zeros((TOTAL_FAULTS, MAX_VECTORS, 32), dtype=np.uint8)
    response_validity = np.zeros((TOTAL_FAULTS, MAX_VECTORS, 32), dtype=np.uint8)
    cycle_delta = np.zeros((TOTAL_FAULTS, MAX_VECTORS), dtype=np.int32)
    timed_out = np.zeros((TOTAL_FAULTS, MAX_VECTORS), dtype=np.uint8)
    protocol_error = np.zeros((TOTAL_FAULTS, MAX_VECTORS), dtype=np.uint8)
    detected = np.zeros((TOTAL_FAULTS, MAX_VECTORS), dtype=np.uint8)
    vector_mask = np.zeros((TOTAL_FAULTS, MAX_VECTORS), dtype=np.uint8)
    baseline_response = np.zeros((len(FAMILIES), MAX_VECTORS, 32), dtype=np.uint8)
    baseline_cycles = np.zeros((len(FAMILIES), MAX_VECTORS), dtype=np.int32)
    transaction_ids = np.full((len(FAMILIES), MAX_VECTORS), -1, dtype=np.int32)
    family_index = np.empty(TOTAL_FAULTS, dtype=np.uint8)
    local_site_index = np.empty(TOTAL_FAULTS, dtype=np.int32)
    stuck_value = np.empty(TOTAL_FAULTS, dtype=np.uint8)
    opaque_ids: list[bytes] = []
    catalog_rows: list[dict[str, Any]] = []
    batch_rows: list[dict[str, Any]] = []

    validity_by_family = {
        "opentitan_hmac_sha256": np.ones(32, dtype=np.uint8),
        "picorv32_cpu": np.asarray([1] * 20 + [0] * 12, dtype=np.uint8),
        "secworks_aes": np.asarray([0] * 16 + [1] * 16, dtype=np.uint8),
        "secworks_sha256": np.ones(32, dtype=np.uint8),
    }
    family_offsets: dict[str, int] = {}
    running_fault = 0
    for family_number, family_id in enumerate(FAMILIES):
        family_offsets[family_id] = running_fault
        sites = EXPECTED_SITES[family_id]
        faults = sites * 2
        family_index[running_fault:running_fault + faults] = family_number
        local_site_index[running_fault:running_fault + faults] = np.repeat(np.arange(sites, dtype=np.int32), 2)
        stuck_value[running_fault:running_fault + faults] = np.tile(np.asarray([0, 1], dtype=np.uint8), sites)
        running_fault += faults
    require(running_fault == TOTAL_FAULTS, "fault offset total")

    for family_number, family_id in enumerate(FAMILIES):
        vectors = FULL_VECTORS[family_id]
        site_count = EXPECTED_SITES[family_id]
        configure_pilot_helpers(family_id, site_count)
        sites = pilot.enumerate_sites(family_id, WORK_1E / "families" / family_id / f"{family_id}_generic_12c1e.json")
        indices = family_indices(arrays_1g, family_id)
        transaction_ids[family_number, :vectors] = arrays_1g["transaction_id"][indices].astype(np.int32)
        base = family_offsets[family_id]
        for site in sites:
            for stuck in (0, 1):
                local_fault = int(site["site_rank"]) * 2 + stuck
                global_fault = base + local_fault
                opaque = f"F{family_number}-{local_fault:06d}-{hashlib.sha256((site['opaque_site_id'] + ':' + str(stuck)).encode()).hexdigest()[:12]}"
                opaque_ids.append(opaque.encode())
                catalog_rows.append({
                    "fault_instance_index": global_fault, "opaque_fault_id": opaque,
                    "family_id": family_id, "partition": PARTITIONS[family_id],
                    "family_index": family_number, "site_rank": site["site_rank"], "stuck_value": stuck,
                    "cell_name": site["cell_name"], "cell_type": site["cell_type"],
                    "output_port": site["output_port"], "output_bit_index": site["output_bit_index"],
                    "net_bit_id": site["net_bit_id"],
                })

        reference_response: np.ndarray | None = None
        reference_cycles: np.ndarray | None = None
        for row in (item for item in plan if item["family_id"] == family_id):
            global_batch = int(row["global_batch_id"])
            local_batch = int(row["family_batch_id"])
            site_start = int(row["site_rank_start"])
            batch_sites = int(row["site_count_maximum"])
            key = f"{global_batch:04d}"
            item = state["families"][family_id]["batches"][key]
            csv_path = verify_record(item["csv"], f"full CSV {key}")
            npz_path = verify_record(item["npz"], f"full NPZ {key}")
            local, summary = parse_batch(csv_path, family_id, global_batch, site_start, batch_sites)
            require(deterministic_npz_bytes(local) == npz_path.read_bytes(), f"batch NPZ replay: {family_id}/{local_batch}")
            require(summary == item["summary"], f"checkpoint summary replay: {family_id}/{local_batch}")
            if reference_response is None:
                reference_response, reference_cycles = local["baseline_response"], local["baseline_cycles"]
            else:
                require(np.array_equal(reference_response, local["baseline_response"]), f"cross-batch response baseline: {family_id}")
                require(np.array_equal(reference_cycles, local["baseline_cycles"]), f"cross-batch cycle baseline: {family_id}")
            global_start = base + site_start * 2
            count = batch_sites * 2
            response_xor[global_start:global_start + count, :vectors] = local["response_xor"]
            cycle_delta[global_start:global_start + count, :vectors] = local["cycle_delta"]
            timed_out[global_start:global_start + count, :vectors] = local["timed_out"]
            protocol_error[global_start:global_start + count, :vectors] = local["protocol_error"]
            detected[global_start:global_start + count, :vectors] = local["detected"]
            vector_mask[global_start:global_start + count, :vectors] = 1
            response_validity[global_start:global_start + count, :vectors] = validity_by_family[family_id][None, None, :]
            batch_rows.append({
                "global_batch_id": global_batch, "family_id": family_id, "partition": PARTITIONS[family_id],
                "family_batch_id": local_batch, "site_rank_start": site_start, "site_count": batch_sites,
                "fault_count": count, "vector_count": vectors, "baseline_records": vectors,
                "enabled_records": summary["enabled_records"], "observable_faults": summary["observable_faults"],
                "raw_csv": rel(csv_path), "raw_csv_sha256": sha256(csv_path),
                "batch_npz": rel(npz_path), "batch_npz_sha256": sha256(npz_path), "status": "PASS",
            })
        require(reference_response is not None and reference_cycles is not None, f"family baseline: {family_id}")
        baseline_response[family_number, :vectors] = reference_response
        baseline_cycles[family_number, :vectors] = reference_cycles

    require(len(catalog_rows) == TOTAL_FAULTS and len(set(opaque_ids)) == TOTAL_FAULTS, "fault catalog integrity")
    require(len(batch_rows) == TOTAL_BATCHES, "batch registry count")
    require(int(vector_mask.sum()) == TOTAL_ENABLED, "enabled transaction matrix")
    observable = np.any(detected != 0, axis=1)
    candidate_count = np.zeros(TOTAL_FAULTS, dtype=np.int32)
    exact_site = np.zeros(TOTAL_FAULTS, dtype=np.uint8)
    signature_values = ["NORMAL_COMPATIBLE"] * TOTAL_FAULTS
    signature_rows: list[dict[str, Any]] = []
    family_metrics: list[dict[str, Any]] = []
    for family_number, family_id in enumerate(FAMILIES):
        vectors = FULL_VECTORS[family_id]
        start = family_offsets[family_id]
        end = start + EXPECTED_SITES[family_id] * 2
        groups: dict[str, list[int]] = {}
        for fault in range(start, end):
            if observable[fault]:
                signature = signature_digest(family_id, response_xor[fault, :vectors], cycle_delta[fault, :vectors], timed_out[fault, :vectors], protocol_error[fault, :vectors])
                signature_values[fault] = signature
                groups.setdefault(signature, []).append(fault)
        for signature, members in sorted(groups.items()):
            candidate_sites = sorted({int(local_site_index[index]) for index in members})
            for fault in members:
                candidate_count[fault] = len(candidate_sites)
                exact_site[fault] = int(candidate_sites == [int(local_site_index[fault])])
            signature_rows.append({
                "family_id": family_id, "partition": PARTITIONS[family_id],
                "signature_sha256": signature, "fault_instances": len(members),
                "candidate_sites": len(candidate_sites), "site_rank_min": min(candidate_sites),
                "site_rank_max": max(candidate_sites),
            })
        local_observable = observable[start:end]
        obs_counts = candidate_count[start:end][local_observable]
        family_metrics.append({
            "family_id": family_id, "partition": PARTITIONS[family_id],
            "sites": EXPECTED_SITES[family_id], "fault_instances": end - start,
            "full_vectors": vectors, "enabled_transactions": (end - start) * vectors,
            "observable_faults": int(local_observable.sum()),
            "all_injected_detection_recall": float(local_observable.mean()),
            "unique_signature_faults": int(np.sum((candidate_count[start:end] == 1) & local_observable)),
            "ambiguous_signature_faults": int(np.sum((candidate_count[start:end] > 1) & local_observable)),
            "all_injected_exact_site_rate": float(exact_site[start:end].mean()),
            "mean_observable_candidate_sites": float(obs_counts.mean()) if obs_counts.size else 0.0,
            "maximum_observable_candidate_sites": int(obs_counts.max()) if obs_counts.size else 0,
            "fault_free_false_alarms": 0,
        })

    observable_counts = candidate_count[observable]
    rng = np.random.default_rng(12030112)
    site_detection = observable.reshape(TOTAL_SITES, 2).mean(axis=1)
    site_exact = exact_site.reshape(TOTAL_SITES, 2).mean(axis=1)
    bootstrap_detection = np.zeros(BOOTSTRAP_REPLICATES, dtype=np.float64)
    bootstrap_exact = np.zeros(BOOTSTRAP_REPLICATES, dtype=np.float64)
    bootstrap_rows: list[dict[str, Any]] = []
    for replicate in range(BOOTSTRAP_REPLICATES):
        sample = rng.integers(0, TOTAL_SITES, size=TOTAL_SITES)
        bootstrap_detection[replicate] = float(site_detection[sample].mean())
        bootstrap_exact[replicate] = float(site_exact[sample].mean())
        bootstrap_rows.append({
            "replicate": replicate,
            "all_injected_detection_recall": f"{bootstrap_detection[replicate]:.8f}",
            "all_injected_exact_site_rate": f"{bootstrap_exact[replicate]:.8f}",
        })
    detection_ci = percentile_interval(bootstrap_detection)
    exact_ci = percentile_interval(bootstrap_exact)
    metrics = {
        "metrics_version": "CIRCUITSAGE-HMAC-V2.2-FULL-CAMPAIGN-METRICS-12C1L-v1",
        "stage": STAGE, "status": "PASS / REPORT-ONLY", "created_at": now(),
        "scope": "GENERALIZATION_TRAIN + GENERALIZATION_CALIBRATION FULL CAMPAIGN",
        "families": family_metrics, "sites": TOTAL_SITES, "fault_instances": TOTAL_FAULTS,
        "full_vectors": TOTAL_VECTORS, "enabled_transactions": TOTAL_ENABLED,
        "observable_faults": int(observable.sum()),
        "all_injected_detection_recall": float(observable.mean()),
        "detection_site_bootstrap_95_ci": list(detection_ci),
        "observable_candidate_set_coverage": 1.0 if bool(observable.any()) else 0.0,
        "unique_signature_top1_site": 1.0 if bool(np.any(candidate_count == 1)) else 0.0,
        "all_injected_exact_site_rate": float(exact_site.mean()),
        "exact_site_bootstrap_95_ci": list(exact_ci),
        "mean_observable_candidate_sites": float(observable_counts.mean()) if observable_counts.size else 0.0,
        "maximum_observable_candidate_sites": int(observable_counts.max()) if observable_counts.size else 0,
        "ambiguous_false_unique_rate": 0.0, "fault_free_false_alarms": 0,
        "fault_free_false_alarm_rate": 0.0, "dataset_integrity": "PASS",
        "model_training_selection_inference": [0, 0, 0],
        "independent_test_validation_holdout_access": [0, 0, 0],
        "independent_generalization": "NOT ESTABLISHED",
    }

    features_arrays = {
        "family_index": family_index, "vector_mask": vector_mask,
        "response_validity_mask": response_validity, "response_xor": response_xor,
        "completion_cycle_delta": cycle_delta, "timeout": timed_out,
        "protocol_error": protocol_error, "observable": observable.astype(np.uint8),
        "baseline_response": baseline_response, "baseline_cycles": baseline_cycles,
        "transaction_id": transaction_ids,
    }
    targets_arrays = {
        "fault_instance_index": np.arange(TOTAL_FAULTS, dtype=np.int32),
        "family_index": family_index, "local_site_index": local_site_index,
        "stuck_value": stuck_value, "opaque_fault_id": np.asarray(opaque_ids, dtype="S40"),
        "candidate_site_count": candidate_count, "exact_site": exact_site,
        "behavior_signature_sha256": np.asarray([value.encode() for value in signature_values], dtype="S64"),
    }
    schema = {
        "schema_version": "CIRCUITSAGE-HMAC-V2.2-FULL-CAMPAIGN-DATASET-12C1L-v1",
        "stage": STAGE, "status": "FROZEN", "features": sorted(features_arrays),
        "targets": sorted(targets_arrays), "sample_axis": "fault_instance",
        "vector_axis": "family-specific full vector schedule padded to 64 with vector_mask=0",
        "response_bytes": 32, "fault_identity_in_features": "PROHIBITED / ABSENT",
        "truth_separation": "site, stuck value, opaque fault ID and signatures exist only in target/catalog artifacts",
        "partitions": {"GENERALIZATION_TRAIN": list(FAMILIES[:3]), "GENERALIZATION_CALIBRATION": [FAMILIES[3]]},
        "calibration_gradient_updates": "PROHIBITED",
        "graph_binding": "Stage 12C-1E frozen family netlists",
        "protected_partitions": {"independent_test": "LOCKED", "validation": "UNOPENED", "holdout": "SEALED"},
    }

    validation_fields = ["family_id", "partition", "vector_rank", "transaction_id", "timeout_canary",
                         "cycles", "timed_out", "protocol_error", "unknown", "expected_response",
                         "actual_response", "result"]
    signature_fields = ["family_id", "partition", "signature_sha256", "fault_instances",
                        "candidate_sites", "site_rank_min", "site_rank_max"]
    bootstrap_fields = ["replicate", "all_injected_detection_recall", "all_injected_exact_site_rate"]
    write_or_verify(VALIDATION_RESULTS, csv_bytes(validation_rows, validation_fields))
    write_or_verify(FAULT_CATALOG, csv_bytes(catalog_rows, FAULT_FIELDS))
    write_npz_or_verify(FEATURES, features_arrays)
    write_npz_or_verify(TARGETS, targets_arrays)
    write_or_verify(SIGNATURES, csv_bytes(signature_rows, signature_fields))
    write_or_verify(METRICS, canonical_json(metrics))
    write_or_verify(BOOTSTRAP, csv_bytes(bootstrap_rows, bootstrap_fields))
    write_or_verify(BATCH_REGISTRY, csv_bytes(batch_rows, BATCH_FIELDS))
    write_or_verify(SCHEMA, canonical_json(schema))
    report = f"""# CircuitSage-HMAC V2.2 full TRAIN/CALIBRATION campaign — Stage 12C-1L

The complete authorized campaign executed {TOTAL_BATCHES} checkpointed batches
covering {TOTAL_SITES} sites, {TOTAL_FAULTS} persistent SA0/SA1 faults and
{TOTAL_ENABLED} enabled fault/vector transactions.  All four full-vector
adapter schedules passed byte-exact fault-free replay.

The report-only TRAIN/CALIBRATION all-injected detection recall is
{metrics['all_injected_detection_recall']:.8f}, and exact-site rate is
{metrics['all_injected_exact_site_rate']:.8f}.  These results authorize no
scientific claim by themselves; model-training readiness requires a separate
disposition gate.  Independent TEST remains locked, VALIDATION unopened, and
HOLDOUT sealed.  Independent-circuit generalization is not established.

The future combined-model brand remains **{FUTURE_BRAND}**.
""".encode()
    write_or_verify(REPORT, report)

    state["status"] = "PASS / FROZEN"
    state["campaign_execution"] = f"COMPLETED / {TOTAL_BATCHES} OF {TOTAL_BATCHES} BATCHES"
    state["dataset_integrity"] = "PASS"
    state["metrics"] = metrics
    state["updated_at"] = now()
    atomic_json(CHECKPOINT, state)
    stage_outputs = [CHECKPOINT, VALIDATION_RESULTS, FAULT_CATALOG, FEATURES, TARGETS,
                     SIGNATURES, METRICS, BOOTSTRAP, BATCH_REGISTRY, SCHEMA, REPORT]
    manifest = {
        "manifest_version": "CIRCUITSAGE-HMAC-V2.2-FULL-CAMPAIGN-DATASET-MANIFEST-12C1L-v1",
        "stage": STAGE, "status": "PASS", "created_at": now(),
        "stage_source": record(Path(__file__).resolve()),
        "frozen_inputs": {rel(path): record(path) for path in PINNED},
        "outputs": {rel(path): record(path) for path in stage_outputs},
        "families": list(FAMILIES), "sites": TOTAL_SITES, "fault_instances": TOTAL_FAULTS,
        "full_vectors": TOTAL_VECTORS, "enabled_transactions": TOTAL_ENABLED,
        "simulation_batches": TOTAL_BATCHES, "dataset_records": TOTAL_FAULTS,
        "model_deserializations": 0, "training_calls": 0, "selection_calls": 0, "inference_calls": 0,
        "independent_test_access": 0, "validation_access": 0, "holdout_access": 0,
    }
    frozen_write(MANIFEST, canonical_json(manifest))
    audit = {
        "audit_version": "CIRCUITSAGE-HMAC-V2.2-FULL-CAMPAIGN-EXECUTION-DATASET-FREEZE-12C1L-v1",
        "stage": STAGE, "status": "PASS", "execution_dataset": "COMPLETED / FROZEN",
        "families_train_calibration": "4 / TRAIN 3 / CALIBRATION 1",
        "sites_faults_full_vectors": [TOTAL_SITES, TOTAL_FAULTS, TOTAL_VECTORS],
        "simulation_batches": f"{TOTAL_BATCHES}/{TOTAL_BATCHES}",
        "enabled_transactions": TOTAL_ENABLED, "observable_faults": metrics["observable_faults"],
        "all_injected_detection_recall": metrics["all_injected_detection_recall"],
        "detection_site_bootstrap_95_ci": metrics["detection_site_bootstrap_95_ci"],
        "all_injected_exact_site_rate": metrics["all_injected_exact_site_rate"],
        "exact_site_bootstrap_95_ci": metrics["exact_site_bootstrap_95_ci"],
        "mean_maximum_observable_candidate_sites": [metrics["mean_observable_candidate_sites"], metrics["maximum_observable_candidate_sites"]],
        "fault_free_false_alarms": 0, "dataset_integrity": "PASS",
        "model_training_selection_inference": "NOT AUTHORIZED / NOT AUTHORIZED / NOT AUTHORIZED",
        "independent_test_validation_holdout_access": [0, 0, 0],
        "independent_generalization": "NOT ESTABLISHED",
        "fault_identity_in_features": "PROHIBITED / ABSENT",
        "frozen_rtl_netlists_vectors_modified": [False, False, False],
        "features_record": record(FEATURES), "targets_record": record(TARGETS),
        "metrics_record": record(METRICS), "batch_registry_record": record(BATCH_REGISTRY),
        "manifest_record": record(MANIFEST), "future_combined_model_brand": FUTURE_BRAND,
        "next_gate": "STAGE 12C-1M — FULL-CAMPAIGN RESULT DISPOSITION AND MODEL-TRAINING READINESS FREEZE",
    }
    frozen_write(AUDIT, canonical_json(audit))
    for path in (METRICS, SCHEMA, MANIFEST, AUDIT):
        require(path.read_bytes() == canonical_json(load_json(path)), f"canonical output replay: {path.name}")
    for path, expected in PINNED.items():
        require(sha256(path) == expected, f"frozen input modified: {rel(path)}")

    print("\nSTAGE 12C-1L — FULL TRAIN/CALIBRATION FAULT-CAMPAIGN EXECUTION AND DATASET FREEZE")
    print(f"{'Status':<94}: PASS")
    print(f"{'Execution / dataset':<94}: COMPLETED / FROZEN")
    print(f"{'Families / sites / faults / full vectors':<94}: 4 / {TOTAL_SITES} / {TOTAL_FAULTS} / {TOTAL_VECTORS}")
    print(f"{'Simulation batches / enabled transactions':<94}: {TOTAL_BATCHES}/{TOTAL_BATCHES} / {TOTAL_ENABLED}")
    print(f"{'Observable faults / detection recall':<94}: {metrics['observable_faults']} / {metrics['all_injected_detection_recall']:.8f}")
    print(f"{'Detection 95% site-bootstrap CI':<94}: [{detection_ci[0]:.8f}, {detection_ci[1]:.8f}]")
    print(f"{'All-injected exact-site rate':<94}: {metrics['all_injected_exact_site_rate']:.8f}")
    print(f"{'Exact-site 95% site-bootstrap CI':<94}: [{exact_ci[0]:.8f}, {exact_ci[1]:.8f}]")
    print(f"{'Mean / maximum observable candidate sites':<94}: {metrics['mean_observable_candidate_sites']:.4f} / {metrics['maximum_observable_candidate_sites']}")
    print(f"{'Fault-free false alarms':<94}: 0")
    print(f"{'Model training / selection / inference':<94}: NOT AUTHORIZED / NOT AUTHORIZED / NOT AUTHORIZED")
    print(f"{'Independent TEST / VALIDATION / HOLDOUT access':<94}: 0 / 0 / 0")
    print(f"{'Independent generalization':<94}: NOT ESTABLISHED")
    print(f"{'Features':<94}: {FEATURES}")
    print(f"{'Features SHA':<94}: {sha256(FEATURES)}")
    print(f"{'Metrics':<94}: {METRICS}")
    print(f"{'Metrics SHA':<94}: {sha256(METRICS)}")
    print(f"{'Manifest':<94}: {MANIFEST}")
    print(f"{'Manifest SHA':<94}: {sha256(MANIFEST)}")
    print(f"{'Audit':<94}: {AUDIT}")
    print(f"{'Audit SHA':<94}: {sha256(AUDIT)}")
    print(f"{'Next gate':<94}: STAGE 12C-1M — FULL-CAMPAIGN RESULT DISPOSITION AND MODEL-TRAINING READINESS FREEZE")


def execute(resume: bool) -> None:
    require(ROOT.name == "vlsi_fault_detection_v2", f"place script in project root, not {ROOT}")
    require(not MANIFEST.exists() and not AUDIT.exists(), "frozen Stage 12C-1L output exists; use --status")
    arrays, plan, family_max = verify_inputs()
    for directory in (WORK, BUILD, RAW, PER_BATCH, MEMORY):
        directory.mkdir(parents=True, exist_ok=True)
    if CHECKPOINT.exists():
        require(resume, "checkpoint exists; rerun with --resume")
        state = load_json(CHECKPOINT)
        require(state.get("stage") == STAGE and state.get("status") == "RUNNING", "checkpoint state")
        require(state.get("authorization_sha256") == PINNED[AUDIT_1K], "checkpoint authorization anchor")
    else:
        require(not resume, "--resume requested without checkpoint")
        state = {
            "checkpoint_version": "CIRCUITSAGE-HMAC-V2.2-FULL-CAMPAIGN-CHECKPOINT-12C1L-v1",
            "stage": STAGE, "status": "RUNNING", "started_at": now(), "updated_at": now(),
            "authorization_sha256": PINNED[AUDIT_1K], "authorized_families": list(FAMILIES),
            "planned_batches": TOTAL_BATCHES, "completed_batches": [],
            "enabled_transactions_completed": 0, "families": {},
            "independent_test_access": 0, "validation_access": 0, "holdout_access": 0,
        }
        atomic_json(CHECKPOINT, state)

    validation_rows: list[dict[str, Any]] = []
    for family_id in FAMILIES:
        _, binary = prepare_family(state, family_id, arrays, family_max[family_id])
        validation_rows.extend(validate_family(state, family_id, binary))
        run_batches(state, family_id, binary, plan)
    require(all(state["families"][family].get("validation_status") == "PASS" for family in FAMILIES), "all-family validation gate")
    require(len(state.get("completed_batches", [])) == TOTAL_BATCHES, "all campaign batches")
    consolidate_and_freeze(state, arrays, plan, validation_rows)


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
            stop("Stage 12C-1L execution lock is held by another process")
        execute(resume)


def status() -> None:
    print("STAGE 12C-1L — FULL-CAMPAIGN STATUS")
    held = lock_held()
    print(f"Execution lock held       : {'YES' if held else 'NO'}")
    if MANIFEST.is_file() and AUDIT.is_file():
        manifest, audit = load_json(MANIFEST), load_json(AUDIT)
        require(manifest.get("status") == "PASS" and audit.get("status") == "PASS", "frozen status")
        require(audit.get("manifest_record", {}).get("sha256") == sha256(MANIFEST), "manifest audit anchor")
        verify_manifest_outputs(manifest, "12C-1L")
        print("Status                    : PASS / FROZEN")
        print(f"Batches verified          : {audit['simulation_batches']}")
        print(f"Faults / transactions     : {TOTAL_FAULTS} / {TOTAL_ENABLED}")
        print(f"Detection recall          : {audit['all_injected_detection_recall']:.8f}")
        print(f"Exact-site rate           : {audit['all_injected_exact_site_rate']:.8f}")
        print(f"Audit                     : {AUDIT}")
        print(f"Audit SHA                 : {sha256(AUDIT)}")
        return
    if not CHECKPOINT.is_file():
        print("Status                    : NOT STARTED")
        print(f"Checkpoint                : {CHECKPOINT}")
        return
    state = load_json(CHECKPOINT)
    completed = sorted(int(value) for value in state.get("completed_batches", []))
    built = [family for family in FAMILIES if state.get("families", {}).get(family, {}).get("build_status") == "PASS"]
    validated = [family for family in FAMILIES if state.get("families", {}).get(family, {}).get("validation_status") == "PASS"]
    print(f"Status                    : {state.get('status')}")
    print(f"Families built            : {len(built)}/4")
    print(f"Adapters validated        : {len(validated)}/4")
    print(f"Completed batches         : {len(completed)}/{TOTAL_BATCHES} ({100.0*len(completed)/TOTAL_BATCHES:.2f}%)")
    print(f"Enabled transactions      : {int(state.get('enabled_transactions_completed', 0))}/{TOTAL_ENABLED}")
    if completed:
        print(f"Latest completed batch    : {completed[-1]:04d}")
    print(f"Checkpoint                : {CHECKPOINT}")
    print(f"Checkpoint SHA            : {sha256(CHECKPOINT)}")


def self_test() -> None:
    require(TOTAL_SITES == 63749 and TOTAL_FAULTS == 127498, "campaign dimensions")
    require(TOTAL_BATCHES == 999 and TOTAL_VECTORS == 240, "batch/vector dimensions")
    require(TOTAL_ENABLED == 7850592, "enabled transaction arithmetic")
    require(sum(math.ceil(value / BATCH_SITES) for value in EXPECTED_SITES.values()) == TOTAL_BATCHES, "batch arithmetic")
    sample = {"a": np.arange(5, dtype=np.uint8), "b": np.arange(3, dtype=np.int32)}
    path = ROOT / ".stage_12c1l_selftest.npz"
    path.unlink(missing_ok=True)
    try:
        write_npz_or_verify(path, sample)
        write_npz_or_verify(path, sample)
        with np.load(path, allow_pickle=False) as archive:
            require(np.array_equal(archive["a"], sample["a"]), "streamed NPZ")
    finally:
        path.unlink(missing_ok=True)
    require(np.allclose(percentile_interval(np.arange(100, dtype=np.float64)), (2.475, 96.525)), "percentile helper")
    print("Stage 12C-1L self-test: PASS")


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
        locked_execute(resume=args.resume)


if __name__ == "__main__":
    main()
