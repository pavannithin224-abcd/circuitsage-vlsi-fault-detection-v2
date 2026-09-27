#!/usr/bin/env python3
"""Stage 12C-1H: adapter-validation and pilot-campaign authorization freeze.

Verifies the frozen Stage 12C-1G logical adapters and deterministic vectors,
the Stage 12C-1E synthesized TRAIN/CALIBRATION netlists, and the Stage 12C-1F
fault-campaign contract.  It then freezes the executable adapter functional-
validation plan, bounded pilot fault-campaign plan, acceptance policy, and
authorization for the next execution gate.

This authorization stage performs no RTL simulation, response generation,
fault injection, dataset construction, training, inference, or access to
INDEPENDENT_TEST, VALIDATION, or HOLDOUT.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


STAGE = "12C-1H"
ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "config/v2_2"
RESULT = ROOT / "results/circuitsage_hmac_v2_12c1"
WORK = RESULT / "adapter_pilot_authorization_12c1h"

SOURCE_1G = ROOT / "stage_12c1g_portable_adapter_vector_generation.py"
WORK_1G = RESULT / "portable_adapter_vector_generation_12c1g"
ADAPTERS_1G = WORK_1G / "circuitsage_hmac_v2_2_portable_adapter_registry_12c1g.json"
ADAPTER_MODULE_1G = WORK_1G / "circuitsage_hmac_v2_2_portable_adapters_12c1g.py"
VECTORS_1G = WORK_1G / "circuitsage_hmac_v2_2_deterministic_transaction_vectors_12c1g.npz"
VECTOR_INVENTORY_1G = WORK_1G / "circuitsage_hmac_v2_2_deterministic_vector_inventory_12c1g.csv"
VECTOR_SCHEMA_1G = WORK_1G / "circuitsage_hmac_v2_2_deterministic_vector_schema_12c1g.json"
REPLAY_1G = WORK_1G / "circuitsage_hmac_v2_2_adapter_vector_replay_12c1g.json"
MANIFEST_1G = RESULT / "circuitsage_hmac_v2_2_portable_adapter_vector_manifest_12c1g.json"
AUDIT_1G = RESULT / "circuitsage_hmac_v2_2_portable_adapter_vector_generation_freeze_12c1g.json"

WORK_1E = RESULT / "train_calibration_synthesis_12c1e"
MANIFEST_1E = RESULT / "circuitsage_hmac_v2_2_elaboration_synthesis_manifest_12c1e.json"
AUDIT_1E = RESULT / "circuitsage_hmac_v2_2_elaboration_wrapper_generic_synthesis_freeze_12c1e.json"

CAMPAIGN_1F = CONFIG / "circuitsage_hmac_v2_2_fault_campaign_contract_12c1f.json"
CAMPAIGN_BUDGET_1F = RESULT / "transaction_fault_campaign_contract_12c1f/circuitsage_hmac_v2_2_campaign_budget_registry_12c1f.csv"
GENERALIZATION_ACCEPTANCE_1A = CONFIG / "circuitsage_hmac_v2_2_acceptance_contract_12c1a.json"

PINNED = {
    SOURCE_1G: "f73c33107b2271785bcc2e6bf1e41deebb797617eb8a11602845c6a40b2a7bb7",
    ADAPTERS_1G: "296b15254debb27f336511ee2d6f1db1745f668444866f7d0ccdee35dab91fb2",
    VECTORS_1G: "0d8fce6c73c5d484fc9880ef97c0663b6dbfc9cbda73fcbd413f1970f418fa7b",
    MANIFEST_1G: "a987b1c13767143a2ffdf0561a93ce352615e5bbe4c81cc77a4a2801e62f108e",
    AUDIT_1G: "4f079aa98c97751bdd2215104df9839b1523b4a4a65df1bc3a6fedbd6e7fd82d",
    MANIFEST_1E: "4d3d606e174ae06589e0e537486a15e042521fe9a02d49f9b4e1c596198476b6",
    AUDIT_1E: "3ac4f454173166ee6848e2617c7072742665e7166345ce5ffdb837f574d59220",
    CAMPAIGN_1F: "5e3590b376e4b61a18ae802bcc73a4a17fd5978a94b5aa045aa947d793280d85",
    CAMPAIGN_BUDGET_1F: "46a5fe2abb62051c7aa2daaf2c8503acd7fceacb29b907b5a6a5d8a56663d7c1",
    GENERALIZATION_ACCEPTANCE_1A: "9c8eec4d85957c4408ac59e0c8760af90c91d0667c995d91ac969b5a8f205f26",
}

VALIDATION_CONTRACT = CONFIG / "circuitsage_hmac_v2_2_adapter_functional_validation_contract_12c1h.json"
PILOT_CONTRACT = CONFIG / "circuitsage_hmac_v2_2_pilot_campaign_execution_contract_12c1h.json"
ACCEPTANCE_CONTRACT = CONFIG / "circuitsage_hmac_v2_2_adapter_pilot_acceptance_contract_12c1h.json"
AUTHORIZATION = CONFIG / "circuitsage_hmac_v2_2_adapter_pilot_execution_authorization_12c1h.json"
EXECUTION_PLAN = WORK / "circuitsage_hmac_v2_2_adapter_pilot_execution_plan_12c1h.csv"
VALIDATION_REGISTRY = WORK / "circuitsage_hmac_v2_2_adapter_validation_registry_12c1h.json"
PREFLIGHT = WORK / "circuitsage_hmac_v2_2_adapter_pilot_preflight_12c1h.json"
REPORT = WORK / "circuitsage_hmac_v2_2_adapter_pilot_authorization_report_12c1h.md"
MANIFEST = RESULT / "circuitsage_hmac_v2_2_adapter_pilot_authorization_manifest_12c1h.json"
AUDIT = RESULT / "circuitsage_hmac_v2_2_adapter_pilot_authorization_freeze_12c1h.json"

AUTHORIZED = (
    "opentitan_hmac_sha256",
    "picorv32_cpu",
    "secworks_aes",
    "secworks_sha256",
)
FUTURE_BRAND = "Faultiva 1.0 — Hybrid VLSI Fault Intelligence"
MIN_FREE_GIB = 20

FAMILY_REQUIREMENTS = {
    "opentitan_hmac_sha256": {
        "partition": "GENERALIZATION_TRAIN",
        "top": "hmac_core",
        "oracle": "PYTHON HMAC-SHA256 REFERENCE + FROZEN OPENTITAN PROTOCOL",
        "protocol": "RESET, START, MESSAGE ACCEPT, PROCESS, DIGEST VALID, TIMEOUT",
        "required_classes": "ZERO,ONES,COUNTING,ONEHOT,RANDOM,BOUNDARY_LENGTH",
    },
    "picorv32_cpu": {
        "partition": "GENERALIZATION_TRAIN",
        "top": "picorv32",
        "oracle": "FROZEN BOUNDED RV32I INTERPRETER + ARCHITECTURAL SIGNATURE",
        "protocol": "RESET, MEMORY VALID/READY, INSTRUCTION/DATA ACCESS, TRAP, CYCLE BOUND",
        "required_classes": "ALU,BRANCH,LOAD_STORE,CSR,INTERRUPT,MIXED",
    },
    "secworks_aes": {
        "partition": "GENERALIZATION_TRAIN",
        "top": "aes_core",
        "oracle": "FIPS-197/NIST AES KNOWN-ANSWER VECTORS",
        "protocol": "RESET, READY, INIT/NEXT, KEY LENGTH, ENC/DEC, RESULT VALID, TIMEOUT",
        "required_classes": "NIST_KAT,ZERO,ONES,COUNTING,AVALANCHE,RANDOM",
    },
    "secworks_sha256": {
        "partition": "GENERALIZATION_CALIBRATION",
        "top": "sha256_core",
        "oracle": "PYTHON HASHLIB SHA-256 + NIST KNOWN-ANSWER VECTORS",
        "protocol": "RESET, READY, INIT/NEXT, BLOCK SEQUENCE, DIGEST VALID, TIMEOUT",
        "required_classes": "NIST_KAT,ZERO,ONES,COUNTING,AVALANCHE,RANDOM",
    },
}


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


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def record(path: Path) -> dict[str, Any]:
    return {"path": rel(path), "sha256": sha256(path), "bytes": path.stat().st_size}


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


def csv_bytes(rows: list[dict[str, Any]], fields: list[str]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def command_version(command: list[str]) -> str:
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, check=False, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as exc:
        stop(f"tool version command failed: {' '.join(command)}: {exc}")
    require(result.returncode == 0, f"tool version command: {' '.join(command)}")
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    require(bool(lines), f"empty tool version: {' '.join(command)}")
    return lines[0]


def verify_output_manifest(manifest: dict[str, Any], label: str) -> None:
    outputs = manifest.get("outputs")
    require(isinstance(outputs, dict) and outputs, f"{label} manifest outputs")
    for key, item in outputs.items():
        require(isinstance(item, dict), f"{label} output record: {key}")
        path = ROOT / str(item.get("path", ""))
        require(path.is_file(), f"missing {label} output: {item.get('path')}")
        require(rel(path) == item.get("path"), f"{label} output path: {key}")
        require(sha256(path) == item.get("sha256"), f"{label} output SHA: {rel(path)}")
        require(path.stat().st_size == int(item.get("bytes", -1)), f"{label} output size: {rel(path)}")


def verify_manifest_item(manifest: dict[str, Any], path: Path, label: str) -> None:
    item = manifest.get("outputs", {}).get(rel(path))
    require(isinstance(item, dict), f"missing {label} manifest record: {rel(path)}")
    require(item.get("sha256") == sha256(path), f"{label} manifest SHA: {rel(path)}")
    require(int(item.get("bytes", -1)) == path.stat().st_size, f"{label} manifest size: {rel(path)}")


def load_vectors() -> dict[str, np.ndarray]:
    with np.load(VECTORS_1G, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    required = {
        "family_ids", "family_index", "partition_index", "transaction_id",
        "family_vector_index", "pilot_mask", "stimulus_class", "payload",
        "payload_valid_byte_mask", "control", "request_sha256",
    }
    require(set(arrays) == required, "12C-1G vector array names")
    require(arrays["payload"].shape == (240, 256), "payload shape")
    require(arrays["payload_valid_byte_mask"].shape == (240, 256), "payload mask shape")
    require(arrays["control"].shape == (240, 8), "control shape")
    require(arrays["request_sha256"].shape == (240, 32), "request hash shape")
    require(arrays["pilot_mask"].shape == (240,) and int(arrays["pilot_mask"].sum()) == 60, "pilot mask")
    require(arrays["family_ids"].tolist() == [item.encode() for item in AUTHORIZED], "family order")
    require(int(np.sum(arrays["partition_index"] == 0)) == 176, "TRAIN vector count")
    require(int(np.sum(arrays["partition_index"] == 1)) == 64, "CALIBRATION vector count")

    for index in range(240):
        family_id = AUTHORIZED[int(arrays["family_index"][index])]
        payload = arrays["payload"][index].astype(np.uint8, copy=False).tobytes()
        mask = arrays["payload_valid_byte_mask"][index].astype(np.uint8, copy=False).tobytes()
        control = arrays["control"][index].astype("<u4", copy=False).tobytes()
        material = family_id.encode() + b"\x00" + payload + mask + control
        require(hashlib.sha256(material).digest() == arrays["request_sha256"][index].tobytes(),
                f"request hash replay: vector {index}")
    return arrays


def top_port_summary(netlist_json: Path, expected_top: str) -> dict[str, Any]:
    document = load_json(netlist_json)
    modules = document.get("modules", {})
    require(len(modules) == 1, f"flattened netlist module count: {rel(netlist_json)}")
    top, module = next(iter(modules.items()))
    require(top == expected_top, f"frozen top mismatch: expected {expected_top}, got {top}")
    ports = module.get("ports", {})
    require(isinstance(ports, dict) and ports, f"top ports: {expected_top}")
    inputs = sum(len(item.get("bits", [])) for item in ports.values() if item.get("direction") == "input")
    outputs = sum(len(item.get("bits", [])) for item in ports.values() if item.get("direction") == "output")
    return {
        "top_module": top,
        "port_count": len(ports),
        "input_bits": inputs,
        "output_bits": outputs,
        "ports": [
            {"name": name, "direction": item.get("direction"), "width": len(item.get("bits", []))}
            for name, item in sorted(ports.items())
        ],
    }


def verify_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, np.ndarray], list[dict[str, str]], dict[str, dict[str, Any]]]:
    print("STAGE 12C-1H — TRAIN/CALIBRATION ADAPTER FUNCTIONAL-VALIDATION AND PILOT-CAMPAIGN AUTHORIZATION FREEZE")
    print("FROZEN INPUT VERIFICATION")
    for path, expected in PINNED.items():
        require(path.is_file(), f"missing frozen input: {rel(path)}")
        require(sha256(path) == expected, f"frozen input SHA: {rel(path)}")
        print(f"  {path.name:<112}: OK")

    manifest_1g = load_json(MANIFEST_1G)
    audit_1g = load_json(AUDIT_1G)
    manifest_1e = load_json(MANIFEST_1E)
    audit_1e = load_json(AUDIT_1E)
    campaign = load_json(CAMPAIGN_1F)
    acceptance = load_json(GENERALIZATION_ACCEPTANCE_1A)
    adapters = load_json(ADAPTERS_1G)
    campaign_rows = read_csv(CAMPAIGN_BUDGET_1F)

    require(manifest_1g.get("status") == "PASS" and audit_1g.get("status") == "PASS", "12C-1G freeze")
    require(audit_1g.get("adapter_generation") == "COMPLETED / FROZEN / 4 OF 4", "adapter generation")
    require(audit_1g.get("vector_generation") == "COMPLETED / FROZEN / 240 TOTAL / 60 PILOT", "vector generation")
    require(audit_1g.get("deterministic_replay") == "PASS / BYTE-EXACT", "12C-1G replay")
    require(audit_1g.get("rtl_functional_validation") == "NOT PERFORMED / NOT AUTHORIZED", "validation boundary")
    require(audit_1g.get("independent_test_validation_holdout_access") == [0, 0, 0], "12C-1G protected access")
    require(manifest_1g.get("stage_source", {}).get("sha256") == PINNED[SOURCE_1G], "12C-1G stage source")
    verify_output_manifest(manifest_1g, "12C-1G")

    require(manifest_1e.get("status") == "PASS" and audit_1e.get("status") == "PASS", "12C-1E freeze")
    require(audit_1e.get("execution_status") == "COMPLETED / FROZEN", "12C-1E lifecycle")
    require(audit_1e.get("generic_synthesis") == "PASS / 4 OF 4", "12C-1E synthesis")
    require(audit_1e.get("deterministic_replay") == "PASS / BYTE-EXACT / 4 OF 4", "12C-1E replay")
    require(audit_1e.get("independent_test_validation_holdout_access") == [0, 0, 0], "12C-1E protected access")
    verify_output_manifest(manifest_1e, "12C-1E")

    require(campaign.get("status") == "FROZEN", "campaign contract status")
    require(campaign.get("fault_model") == "SINGLE PERSISTENT CELL-OUTPUT SA0/SA1", "fault model")
    require(campaign.get("execution") == "NOT AUTHORIZED", "predecessor campaign boundary")
    require(acceptance.get("status") == "FROZEN", "generalization acceptance")
    require(acceptance.get("holdout") == "REMAINS BLOCKED EVEN IF V2.2 PASSES", "HOLDOUT contract")
    require(adapters.get("status") == "FROZEN" and len(adapters.get("families", [])) == 4, "adapter registry")
    require(adapters.get("functional_validation") == "NOT PERFORMED / SEPARATE GATE REQUIRED", "functional boundary")
    require([row["family_id"] for row in campaign_rows] == list(AUTHORIZED), "campaign family order")

    arrays = load_vectors()
    inventory = read_csv(VECTOR_INVENTORY_1G)
    require(len(inventory) == 240, "vector inventory records")
    require(sum(row["pilot"] == "YES" for row in inventory) == 60, "pilot inventory records")
    require(sum(row["timeout_canary"] == "YES" for row in inventory) == 4, "timeout canaries")
    require(all(row["fault_identity_fields"] == "0" for row in inventory), "vector identity firewall")
    compile(ADAPTER_MODULE_1G.read_text(encoding="utf-8"), str(ADAPTER_MODULE_1G), "exec")

    family_results: dict[str, dict[str, Any]] = {}
    for family_id in AUTHORIZED:
        result_path = WORK_1E / "families" / family_id / "family_result_12c1e.json"
        require(result_path.is_file(), f"family result: {family_id}")
        verify_manifest_item(manifest_1e, result_path, "12C-1E")
        result = load_json(result_path)
        require(result.get("status") == "PASS" and result.get("family_id") == family_id, f"family result: {family_id}")
        require(result.get("top_module") == FAMILY_REQUIREMENTS[family_id]["top"], f"family top: {family_id}")
        require(result.get("deterministic_netlist_replay") == "PASS / BYTE-EXACT", f"netlist replay: {family_id}")
        netlist_json = WORK_1E / "families" / family_id / f"{family_id}_generic_12c1e.json"
        netlist_verilog = WORK_1E / "families" / family_id / f"{family_id}_generic_12c1e.v"
        verify_manifest_item(manifest_1e, netlist_json, "12C-1E")
        verify_manifest_item(manifest_1e, netlist_verilog, "12C-1E")
        result["port_summary_12c1h"] = top_port_summary(netlist_json, result["top_module"])
        family_results[family_id] = result

    print("  Four adapters, 240 vectors, 60-vector pilot, four netlists and protected partitions              : PASS")
    return manifest_1g, manifest_1e, arrays, campaign_rows, family_results


def execute() -> None:
    require(ROOT.name == "vlsi_fault_detection_v2", f"place script in project root, not {ROOT}")
    outputs = (VALIDATION_CONTRACT, PILOT_CONTRACT, ACCEPTANCE_CONTRACT, AUTHORIZATION,
               EXECUTION_PLAN, VALIDATION_REGISTRY, PREFLIGHT, REPORT, MANIFEST, AUDIT)
    for path in outputs:
        require(not path.exists(), f"Stage {STAGE} output already exists: {rel(path)}")

    manifest_1g, manifest_1e, arrays, campaign_rows, family_results = verify_inputs()
    timestamp = now()
    verilator = shutil.which("verilator")
    yosys = shutil.which("yosys")
    require(verilator is not None, "Verilator not found")
    require(yosys is not None, "Yosys not found")
    disk = shutil.disk_usage(ROOT)
    free_gib = disk.free / (1024 ** 3)
    require(free_gib >= MIN_FREE_GIB, f"insufficient disk: {free_gib:.2f} GiB available; {MIN_FREE_GIB} GiB required")

    budget_map = {row["family_id"]: row for row in campaign_rows}
    adapter_map = {item["family_id"]: item for item in load_json(ADAPTERS_1G)["families"]}
    plan_rows: list[dict[str, Any]] = []
    validation_items: list[dict[str, Any]] = []
    pilot_counts: dict[str, int] = {}
    validation_counts: dict[str, int] = {}

    for family_index, family_id in enumerate(AUTHORIZED):
        requirement = FAMILY_REQUIREMENTS[family_id]
        budget = budget_map[family_id]
        result = family_results[family_id]
        port_summary = result["port_summary_12c1h"]
        vector_count = int(np.sum((arrays["family_index"] == family_index) & arrays["pilot_mask"]))
        require(vector_count == int(budget["pilot_vector_budget"]), f"pilot-vector alignment: {family_id}")
        timeout_canaries = int(np.sum(
            (arrays["family_index"] == family_index) & (arrays["control"][:, 7] == 1)
        ))
        timeout_canaries_in_pilot = int(np.sum(
            (arrays["family_index"] == family_index) & arrays["pilot_mask"]
            & (arrays["control"][:, 7] == 1)
        ))
        require(timeout_canaries == 1 and timeout_canaries_in_pilot == 0,
                f"isolated timeout canary: {family_id}")
        validation_vectors = vector_count + timeout_canaries
        pilot_counts[family_id] = vector_count
        validation_counts[family_id] = validation_vectors
        pilot_faults = int(budget["pilot_fault_budget"])
        maximum_transactions = int(budget["maximum_pilot_transactions"])
        require(maximum_transactions == pilot_faults * vector_count, f"pilot transaction arithmetic: {family_id}")
        netlist_json = WORK_1E / "families" / family_id / f"{family_id}_generic_12c1e.json"
        netlist_verilog = WORK_1E / "families" / family_id / f"{family_id}_generic_12c1e.v"
        descriptor = ROOT / adapter_map[family_id]["descriptor"]
        require(descriptor.is_file() and sha256(descriptor) == adapter_map[family_id]["descriptor_sha256"],
                f"adapter descriptor: {family_id}")
        plan_rows.append({
            "execution_order": family_index,
            "family_id": family_id,
            "partition": requirement["partition"],
            "top_module": requirement["top"],
            "adapter_descriptor": rel(descriptor),
            "netlist_json": rel(netlist_json),
            "netlist_verilog": rel(netlist_verilog),
            "fault_free_validation_vectors": validation_vectors,
            "pilot_vectors": vector_count,
            "pilot_fault_budget": pilot_faults,
            "maximum_pilot_transactions": maximum_transactions,
            "execution_mode": "SEQUENTIAL / BUILD JOBS 1",
            "authorization": "FUNCTIONAL VALIDATION THEN CONDITIONAL PILOT",
        })
        validation_items.append({
            "family_id": family_id,
            "partition": requirement["partition"],
            "top_module": requirement["top"],
            "reference_oracle": requirement["oracle"],
            "protocol_checks": requirement["protocol"],
            "required_stimulus_classes": requirement["required_classes"],
            "fault_free_vectors": validation_vectors,
            "pilot_vectors": vector_count,
            "timeout_canaries": timeout_canaries,
            "back_to_back_vectors": int(np.sum(
                (arrays["family_index"] == family_index) & arrays["pilot_mask"] & (arrays["control"][:, 6] == 1)
            )),
            "netlist_ports": port_summary,
            "validation_status": "AUTHORIZED / NOT STARTED",
        })

    total_pilot_faults = sum(int(row["pilot_fault_budget"]) for row in campaign_rows)
    total_transactions = sum(int(row["maximum_pilot_transactions"]) for row in campaign_rows)
    require(total_pilot_faults == 16384, "aggregate pilot fault budget")
    require(total_transactions == 245760, "aggregate pilot transaction budget")
    require(sum(pilot_counts.values()) == 60, "aggregate pilot vectors")
    require(sum(validation_counts.values()) == 64, "aggregate validation vectors")

    validation_contract = {
        "contract_version": "CIRCUITSAGE-HMAC-V2.2-ADAPTER-FUNCTIONAL-VALIDATION-12C1H-v1",
        "stage": STAGE,
        "status": "FROZEN",
        "scope": "FOUR TRAIN/CALIBRATION FAMILIES ONLY",
        "inputs": {
            "adapters": record(ADAPTERS_1G),
            "adapter_module": record(ADAPTER_MODULE_1G),
            "vectors": record(VECTORS_1G),
            "vector_inventory": record(VECTOR_INVENTORY_1G),
        },
        "execution_order": [
            "compile frozen source/netlist and generated harness",
            "run reset-only canary",
            "run family known-answer canary",
            "run every family pilot vector fault-free",
            "rerun complete fault-free validation byte-exact",
            "freeze golden responses and adapter-validation results",
            "unlock bounded pilot faults only if every hard gate passes",
        ],
        "hard_gates": [
            "all four adapters compile and bind to the frozen family top",
            "reset, handshake, ready/valid, timeout and completion behavior pass",
            "known-answer/reference-oracle results pass",
            "all 60 pilot vectors complete or produce their contracted timeout-canary status",
            "fault-free replay is byte-exact",
            "no unknown X/Z bit is accepted as a valid response bit",
            "no fault identity, site, polarity or supervision truth enters a model-facing record",
            "upstream RTL, synthesized netlists, adapters and vectors remain byte-identical",
        ],
        "families": validation_items,
        "simulation": "AUTHORIZED NEXT GATE / NOT STARTED",
        "response_generation": "AUTHORIZED NEXT GATE / NOT STARTED",
    }

    pilot_contract = {
        "contract_version": "CIRCUITSAGE-HMAC-V2.2-BOUNDED-PILOT-FAULT-CAMPAIGN-12C1H-v1",
        "stage": STAGE,
        "status": "FROZEN",
        "fault_model": "SINGLE PERSISTENT CELL-OUTPUT SA0/SA1",
        "execution_precondition": "ALL ADAPTER FUNCTIONAL-VALIDATION HARD GATES PASS",
        "families": plan_rows,
        "fault_catalog": {
            "ordering": "DETERMINISTIC FAMILY, HIERARCHICAL CELL, OUTPUT BIT, SA0 THEN SA1",
            "maximum_total_faults": total_pilot_faults,
            "opaque_fault_ids": True,
            "fault_identity_in_query": "PROHIBITED",
            "one_fault_per_run": True,
        },
        "baseline_policy": "ONE FROZEN FAULT-FREE GOLDEN RESPONSE PER FAMILY/VECTOR BEFORE INJECTION",
        "execution": "SEQUENTIAL / BUILD JOBS 1 / CHECKPOINT REQUIRED",
        "maximum_enabled_transactions": total_transactions,
        "checkpoint_frequency": "AFTER EVERY FAMILY-BATCH",
        "resume_policy": "VERIFY COMPLETE BATCH HASHES; NEVER APPEND TO PARTIAL BATCH",
        "full_campaign": "NOT AUTHORIZED",
        "training": "NOT AUTHORIZED",
    }

    acceptance_contract = {
        "contract_version": "CIRCUITSAGE-HMAC-V2.2-ADAPTER-PILOT-ACCEPTANCE-12C1H-v1",
        "stage": STAGE,
        "status": "FROZEN",
        "adapter_hard_gates": {
            "families_passed": "4 OF 4",
            "known_answer_failures_max": 0,
            "protocol_failures_max": 0,
            "missing_duplicate_unknown_records_max": 0,
            "valid_xz_response_bits_max": 0,
            "fault_free_replay": "BYTE-EXACT",
        },
        "pilot_dataset_hard_gates": {
            "families_completed": "4 OF 4",
            "fault_catalog_duplicates_max": 0,
            "missing_duplicate_unknown_records_max": 0,
            "fault_free_baseline_failures_max": 0,
            "identity_leakage_fields_max": 0,
            "checkpoint_and_final_replay": "BYTE-EXACT",
        },
        "pilot_metrics_report_only": [
            "per-family and macro observable ceiling",
            "per-family and macro all-injected detection recall",
            "unique and ambiguous response-signature rates",
            "candidate-set sizes and exact-site ceilings",
            "timeouts and protocol-error observability",
        ],
        "pilot_advancement_threshold": "NOT PRE-DECLARED — SEPARATE DISPOSITION REQUIRED BEFORE FULL CAMPAIGN",
        "independent_generalization_thresholds": record(GENERALIZATION_ACCEPTANCE_1A),
        "failure_policy": "FREEZE FAILURE; DO NOT SKIP FAMILY; DO NOT OPEN TEST OR HOLDOUT",
    }

    authorization = {
        "authorization_version": "CIRCUITSAGE-HMAC-V2.2-ADAPTER-PILOT-EXECUTION-AUTHORIZATION-12C1H-v1",
        "stage": STAGE,
        "status": "FROZEN",
        "adapter_functional_validation": "AUTHORIZED / NOT STARTED",
        "golden_response_generation": "AUTHORIZED ONLY AS PART OF FAULT-FREE VALIDATION",
        "bounded_pilot_campaign": "CONDITIONALLY AUTHORIZED AFTER ADAPTER VALIDATION PASS",
        "authorized_families": list(AUTHORIZED),
        "authorized_partitions": ["GENERALIZATION_TRAIN", "GENERALIZATION_CALIBRATION"],
        "fault_free_validation_vectors": 64,
        "pilot_vectors": 60,
        "pilot_fault_budget": total_pilot_faults,
        "maximum_pilot_transactions": total_transactions,
        "full_campaign": "NOT AUTHORIZED",
        "dataset_consolidation": "AUTHORIZED ONLY FOR PILOT EXECUTION OUTPUTS",
        "model_training_inference": "NOT AUTHORIZED / NOT AUTHORIZED",
        "independent_test": "LOCKED / 2 FAMILIES",
        "holdout": "SEALED / 1 FAMILY",
        "protected_access": {"independent_test": 0, "validation": 0, "holdout": 0},
    }

    plan_fields = ["execution_order", "family_id", "partition", "top_module", "adapter_descriptor",
                   "netlist_json", "netlist_verilog", "fault_free_validation_vectors", "pilot_vectors",
                   "pilot_fault_budget", "maximum_pilot_transactions", "execution_mode", "authorization"]
    plan_payload = csv_bytes(plan_rows, plan_fields)
    validation_registry = {
        "registry_version": "CIRCUITSAGE-HMAC-V2.2-ADAPTER-VALIDATION-REGISTRY-12C1H-v1",
        "stage": STAGE,
        "status": "FROZEN",
        "families": validation_items,
    }
    preflight = {
        "preflight_version": "CIRCUITSAGE-HMAC-V2.2-ADAPTER-PILOT-PREFLIGHT-12C1H-v1",
        "stage": STAGE,
        "status": "PASS",
        "created_at": timestamp,
        "python": sys.version.splitlines()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "verilator": command_version([verilator, "--version"]),
        "yosys": command_version([yosys, "-V"]),
        "available_disk_gib": round(free_gib, 2),
        "minimum_disk_gib": MIN_FREE_GIB,
        "execution_mode": "SEQUENTIAL",
        "build_jobs": 1,
        "checkpoint_required": True,
        "adapter_simulation_calls": 0,
        "fault_injection_calls": 0,
        "response_records_created": 0,
        "dataset_records_created": 0,
        "training_calls": 0,
        "inference_calls": 0,
        "independent_test_access": 0,
        "validation_access": 0,
        "holdout_access": 0,
    }
    report = f"""# CircuitSage-HMAC V2.2 Adapter/Pilot Authorization — Stage 12C-1H

Stage 12C-1H verified the frozen four-family adapter/vector bundle, all four
byte-exact Stage 12C-1E synthesized netlists, and the bounded Stage 12C-1F
SA0/SA1 campaign budget.

The next execution gate is authorized to functionally bind and validate all
four adapters using the 60-vector pilot subset. Only after every adapter hard
gate passes may it execute the bounded pilot campaign of at most
{total_pilot_faults} faults and {total_transactions} enabled transactions.

No simulation, response generation, fault injection, dataset construction,
training, or inference occurred in this authorization stage. Independent TEST
remains locked and HOLDOUT remains sealed. Pilot metrics are report-only and
cannot by themselves establish independent-circuit generalization.

The future hybrid release name remains **{FUTURE_BRAND}**.
""".encode()

    frozen_write(VALIDATION_CONTRACT, canonical_json(validation_contract))
    frozen_write(PILOT_CONTRACT, canonical_json(pilot_contract))
    frozen_write(ACCEPTANCE_CONTRACT, canonical_json(acceptance_contract))
    frozen_write(AUTHORIZATION, canonical_json(authorization))
    frozen_write(EXECUTION_PLAN, plan_payload)
    frozen_write(VALIDATION_REGISTRY, canonical_json(validation_registry))
    frozen_write(PREFLIGHT, canonical_json(preflight))
    frozen_write(REPORT, report)

    stage_outputs = [VALIDATION_CONTRACT, PILOT_CONTRACT, ACCEPTANCE_CONTRACT, AUTHORIZATION,
                     EXECUTION_PLAN, VALIDATION_REGISTRY, PREFLIGHT, REPORT]
    manifest = {
        "manifest_version": "CIRCUITSAGE-HMAC-V2.2-ADAPTER-PILOT-AUTHORIZATION-MANIFEST-12C1H-v1",
        "stage": STAGE,
        "status": "PASS",
        "created_at": timestamp,
        "stage_source": record(Path(__file__).resolve()),
        "frozen_inputs": {rel(path): record(path) for path in PINNED},
        "stage_12c1g_outputs_verified": len(manifest_1g["outputs"]),
        "stage_12c1e_outputs_verified": len(manifest_1e["outputs"]),
        "outputs": {rel(path): record(path) for path in stage_outputs},
        "authorized_families": list(AUTHORIZED),
        "adapter_functional_validation_authorized": True,
        "bounded_pilot_campaign_conditionally_authorized": True,
        "full_campaign_authorized": False,
        "fault_free_validation_vectors": 64,
        "pilot_vectors": 60,
        "pilot_fault_budget": total_pilot_faults,
        "maximum_pilot_transactions": total_transactions,
        "simulation_calls": 0,
        "fault_injection_calls": 0,
        "response_records_created": 0,
        "dataset_records_created": 0,
        "models_deserialized": 0,
        "training_calls": 0,
        "inference_calls": 0,
        "independent_test_access": 0,
        "validation_access": 0,
        "holdout_access": 0,
    }
    frozen_write(MANIFEST, canonical_json(manifest))

    audit = {
        "audit_version": "CIRCUITSAGE-HMAC-V2.2-ADAPTER-PILOT-AUTHORIZATION-FREEZE-12C1H-v1",
        "stage": STAGE,
        "status": "PASS",
        "authorization_status": "FROZEN",
        "adapter_functional_validation": "AUTHORIZED / NOT STARTED",
        "bounded_pilot_campaign": "CONDITIONALLY AUTHORIZED / NOT STARTED",
        "full_campaign": "NOT AUTHORIZED",
        "families": "4 / TRAIN 3 / CALIBRATION 1",
        "fault_free_validation_vectors": 64,
        "pilot_vectors": 60,
        "pilot_fault_budget": total_pilot_faults,
        "maximum_pilot_transactions": total_transactions,
        "execution_build_jobs_checkpoint": "SEQUENTIAL / 1 / REQUIRED",
        "simulation_fault_injection_response_dataset": [0, 0, 0, 0],
        "training_inference": [0, 0],
        "independent_test": "LOCKED / 2 FAMILIES",
        "holdout": "SEALED / 1 FAMILY",
        "independent_test_validation_holdout_access": [0, 0, 0],
        "upstream_rtl_netlists_adapters_vectors_modified": [False, False, False, False],
        "validation_contract_record": record(VALIDATION_CONTRACT),
        "pilot_contract_record": record(PILOT_CONTRACT),
        "acceptance_contract_record": record(ACCEPTANCE_CONTRACT),
        "authorization_record": record(AUTHORIZATION),
        "execution_plan_record": record(EXECUTION_PLAN),
        "preflight_record": record(PREFLIGHT),
        "report_record": record(REPORT),
        "manifest_record": record(MANIFEST),
        "future_combined_model_brand": FUTURE_BRAND,
        "next_gate": "STAGE 12C-1I — TRAIN/CALIBRATION ADAPTER FUNCTIONAL-VALIDATION AND BOUNDED PILOT FAULT-CAMPAIGN EXECUTION FREEZE",
    }
    frozen_write(AUDIT, canonical_json(audit))

    for path in (VALIDATION_CONTRACT, PILOT_CONTRACT, ACCEPTANCE_CONTRACT, AUTHORIZATION,
                 VALIDATION_REGISTRY, PREFLIGHT, MANIFEST, AUDIT):
        require(path.read_bytes() == canonical_json(load_json(path)), f"canonical output replay: {path.name}")
    require(EXECUTION_PLAN.read_bytes() == plan_payload, "execution-plan CSV replay")
    for path, expected in PINNED.items():
        require(sha256(path) == expected, f"frozen input modified: {rel(path)}")

    print("\nSTAGE 12C-1H — TRAIN/CALIBRATION ADAPTER FUNCTIONAL-VALIDATION AND PILOT-CAMPAIGN AUTHORIZATION FREEZE")
    print(f"{'Status':<86}: PASS")
    print(f"{'Authorization status':<86}: FROZEN")
    print(f"{'Adapter functional validation':<86}: AUTHORIZED / NOT STARTED")
    print(f"{'Bounded pilot campaign':<86}: CONDITIONALLY AUTHORIZED / NOT STARTED")
    print(f"{'Full campaign / model training':<86}: NOT AUTHORIZED / NOT AUTHORIZED")
    print(f"{'TRAIN / CALIBRATION families':<86}: 3 / 1")
    print(f"{'Fault-free validation vectors':<86}: 64 — 60 PILOT + 4 ISOLATED TIMEOUT CANARIES")
    print(f"{'Pilot fault budget / maximum enabled transactions':<86}: {total_pilot_faults} / {total_transactions}")
    print(f"{'Execution / build jobs / checkpoint':<86}: SEQUENTIAL / 1 / REQUIRED")
    print(f"{'Available / minimum disk':<86}: {free_gib:.2f} / {MIN_FREE_GIB} GiB")
    print(f"{'Simulation / response generation / fault injection / dataset records':<86}: 0 / 0 / 0 / 0")
    print(f"{'Training / inference':<86}: 0 / 0")
    print(f"{'Independent TEST / HOLDOUT':<86}: LOCKED 2 / SEALED 1")
    print(f"{'Independent TEST / VALIDATION / HOLDOUT access':<86}: 0 / 0 / 0")
    print(f"{'Validation contract':<86}: {VALIDATION_CONTRACT}")
    print(f"{'Validation contract SHA':<86}: {sha256(VALIDATION_CONTRACT)}")
    print(f"{'Pilot contract':<86}: {PILOT_CONTRACT}")
    print(f"{'Pilot contract SHA':<86}: {sha256(PILOT_CONTRACT)}")
    print(f"{'Authorization':<86}: {AUTHORIZATION}")
    print(f"{'Authorization SHA':<86}: {sha256(AUTHORIZATION)}")
    print(f"{'Manifest':<86}: {MANIFEST}")
    print(f"{'Manifest SHA':<86}: {sha256(MANIFEST)}")
    print(f"{'Audit':<86}: {AUDIT}")
    print(f"{'Audit SHA':<86}: {sha256(AUDIT)}")
    print(f"{'Next gate':<86}: STAGE 12C-1I — TRAIN/CALIBRATION ADAPTER FUNCTIONAL-VALIDATION AND BOUNDED PILOT FAULT-CAMPAIGN EXECUTION FREEZE")


def status() -> None:
    print("STAGE 12C-1H — ADAPTER/PILOT AUTHORIZATION STATUS")
    if not MANIFEST.is_file() or not AUDIT.is_file():
        print("Status                    : NOT FROZEN")
        print(f"Audit                     : {AUDIT}")
        return
    manifest = load_json(MANIFEST)
    audit = load_json(AUDIT)
    require(manifest.get("status") == "PASS" and audit.get("status") == "PASS", "freeze status")
    for item in manifest.get("outputs", {}).values():
        path = ROOT / item["path"]
        require(path.is_file(), f"missing frozen output: {item['path']}")
        require(sha256(path) == item["sha256"], f"frozen output SHA: {item['path']}")
        require(path.stat().st_size == int(item["bytes"]), f"frozen output size: {item['path']}")
    require(sha256(MANIFEST) == audit["manifest_record"]["sha256"], "manifest audit record")
    print("Status                    : PASS / FROZEN")
    print("Adapter validation        : AUTHORIZED / NOT STARTED")
    print("Pilot campaign            : CONDITIONALLY AUTHORIZED / NOT STARTED")
    print("Validation vectors        : 64 — 60 PILOT + 4 TIMEOUT CANARIES")
    print("Pilot vectors / faults    : 60 / 16384")
    print("Maximum transactions      : 245760")
    print(f"Audit                     : {AUDIT}")
    print(f"Audit SHA                 : {sha256(AUDIT)}")


def self_test() -> None:
    require(tuple(sorted(FAMILY_REQUIREMENTS)) == AUTHORIZED, "family identity set")
    require(sum(item["partition"] == "GENERALIZATION_TRAIN" for item in FAMILY_REQUIREMENTS.values()) == 3,
            "TRAIN family count")
    require(sum(item["partition"] == "GENERALIZATION_CALIBRATION" for item in FAMILY_REQUIREMENTS.values()) == 1,
            "CALIBRATION family count")
    require(canonical_json({"z": 1, "a": 2}) == b'{"a":2,"z":1}\n', "canonical JSON")
    print("Stage 12C-1H self-test: PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--execute", action="store_true")
    action.add_argument("--status", action="store_true")
    action.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.execute:
        execute()
    elif args.status:
        status()
    else:
        self_test()


if __name__ == "__main__":
    main()
