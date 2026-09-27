#!/usr/bin/env python3
"""Stage 12C-1G: portable adapter and deterministic vector generation freeze.

This stage verifies the frozen Stage 12C-1F contracts, generates one portable
logical transaction adapter descriptor for each authorized TRAIN/CALIBRATION
circuit family, generates the contracted deterministic request-vector corpus,
performs byte-exact generation replay, and freezes all generated artifacts.

It performs no RTL simulation, functional response generation, fault
injection, dataset construction, model training, inference, or access to the
independent TEST, VALIDATION, or HOLDOUT partitions.  Adapter-to-RTL functional
validation and pilot-campaign authorization require a separate next gate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


STAGE = "12C-1G"
ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "config/v2_2"
RESULT = ROOT / "results/circuitsage_hmac_v2_12c1"
PREV_WORK = RESULT / "transaction_fault_campaign_contract_12c1f"
WORK = RESULT / "portable_adapter_vector_generation_12c1g"
ADAPTER_DIR = WORK / "adapters"

SOURCE_1F = ROOT / "stage_12c1f_transaction_fault_campaign_contract.py"
LOG_1F = RESULT / "stage_12c1f_20260918_123017.log"
RESOURCE_1F = RESULT / "stage_12c1f_resources_20260918_123017.log"
TRANSACTION_CONTRACT = CONFIG / "circuitsage_hmac_v2_2_portable_transaction_contract_12c1f.json"
VECTOR_CONTRACT = CONFIG / "circuitsage_hmac_v2_2_transaction_vector_contract_12c1f.json"
CAMPAIGN_CONTRACT = CONFIG / "circuitsage_hmac_v2_2_fault_campaign_contract_12c1f.json"
RESPONSE_SCHEMA_1F = CONFIG / "circuitsage_hmac_v2_2_response_schema_contract_12c1f.json"
ACCESS_POLICY_1F = CONFIG / "circuitsage_hmac_v2_2_campaign_partition_access_policy_12c1f.json"
FAMILY_CSV_1F = PREV_WORK / "circuitsage_hmac_v2_2_family_transaction_registry_12c1f.csv"
FAMILY_JSON_1F = PREV_WORK / "circuitsage_hmac_v2_2_family_transaction_registry_12c1f.json"
VECTOR_BUDGET_1F = PREV_WORK / "circuitsage_hmac_v2_2_vector_budget_registry_12c1f.csv"
CAMPAIGN_BUDGET_1F = PREV_WORK / "circuitsage_hmac_v2_2_campaign_budget_registry_12c1f.csv"
REPORT_1F = PREV_WORK / "circuitsage_hmac_v2_2_transaction_campaign_contract_report_12c1f.md"
MANIFEST_1F = RESULT / "circuitsage_hmac_v2_2_transaction_campaign_contract_manifest_12c1f.json"
AUDIT_1F = RESULT / "circuitsage_hmac_v2_2_transaction_vector_fault_campaign_contract_freeze_12c1f.json"

PINNED = {
    SOURCE_1F: "36ef0adbff6a0ae31ec97d681b22aa734ba956e325bcc7cf2d325d259158864a",
    LOG_1F: "65c7e92cc39d9580cb0c8c45659013ec8f9672933dd8448a8350a074b7c5bfe6",
    RESOURCE_1F: "509202b8cb7e57a3d669608ab4e28feee4900b0af6fa3a3200c196c3807aa912",
    TRANSACTION_CONTRACT: "2c4de1e97b3f29cc7eaa83faffc6b5252b813f7e738f04165c4b5866698f7b00",
    VECTOR_CONTRACT: "530f6d5b7264990e02513f06bb2192fb88c1bae22f4d2b0496dba691be8c1826",
    CAMPAIGN_CONTRACT: "5e3590b376e4b61a18ae802bcc73a4a17fd5978a94b5aa045aa947d793280d85",
    RESPONSE_SCHEMA_1F: "e433599a185823808fd02e6f9b792f148caf75d2f98c77f2061620f046945f2e",
    ACCESS_POLICY_1F: "107bb7b9c09e8f77c792883b2a6215e3fafed909089152e2bc4d8ffb2a095dc5",
    FAMILY_CSV_1F: "efc9843eb7a3a7b92c7bd6747e31a21c751ef8cf28142c00746025df367e4f60",
    FAMILY_JSON_1F: "87ea205312051176e8c61be85b40caccbc21c20deb8893e5b0d19f60050f1697",
    VECTOR_BUDGET_1F: "d42ba825868101a0bb1d38a348bb704914c5f34a65e4d3b399c7d34b52e6ace0",
    CAMPAIGN_BUDGET_1F: "46a5fe2abb62051c7aa2daaf2c8503acd7fceacb29b907b5a6a5d8a56663d7c1",
    REPORT_1F: "b308ea1ccf2e6605ac6397af3498301d277a3a29ee48883dde4901ccfc3d07e9",
    MANIFEST_1F: "b5b57d0f3b7e9ed1df9589cb4ea39270d060bc3228299be946a4672ca8eea1e9",
    AUDIT_1F: "2e90d0c48b33d390d522b6d08f5d3a45eb084b7922c7779198ee372056244ca1",
}

ADAPTER_REGISTRY_CSV = WORK / "circuitsage_hmac_v2_2_portable_adapter_registry_12c1g.csv"
ADAPTER_REGISTRY_JSON = WORK / "circuitsage_hmac_v2_2_portable_adapter_registry_12c1g.json"
ADAPTER_MODULE = WORK / "circuitsage_hmac_v2_2_portable_adapters_12c1g.py"
VECTORS = WORK / "circuitsage_hmac_v2_2_deterministic_transaction_vectors_12c1g.npz"
VECTOR_INVENTORY = WORK / "circuitsage_hmac_v2_2_deterministic_vector_inventory_12c1g.csv"
VECTOR_SCHEMA = WORK / "circuitsage_hmac_v2_2_deterministic_vector_schema_12c1g.json"
REPLAY = WORK / "circuitsage_hmac_v2_2_adapter_vector_replay_12c1g.json"
REPORT = WORK / "circuitsage_hmac_v2_2_portable_adapter_vector_generation_report_12c1g.md"
MANIFEST = RESULT / "circuitsage_hmac_v2_2_portable_adapter_vector_manifest_12c1g.json"
AUDIT = RESULT / "circuitsage_hmac_v2_2_portable_adapter_vector_generation_freeze_12c1g.json"

AUTHORIZED = (
    "opentitan_hmac_sha256",
    "picorv32_cpu",
    "secworks_aes",
    "secworks_sha256",
)
FUTURE_BRAND = "Faultiva 1.0 — Hybrid VLSI Fault Intelligence"
SEED_DOMAIN = b"CIRCUITSAGE-HMAC-V2.2/12C-1F/VECTOR"
PAYLOAD_BYTES = 256
CONTROL_WORDS = 8

EXPECTED = {
    "opentitan_hmac_sha256": {
        "partition": "GENERALIZATION_TRAIN",
        "operation": "HMAC_SHA256_FIXED_AND_VARIABLE_MESSAGE",
        "vectors": 64,
        "pilot": 16,
        "classes": ("ZERO", "ONES", "COUNTING", "ONEHOT", "RANDOM", "BOUNDARY_LENGTH"),
        "payload_interpretation": "key[0:32] || message[32:32+message_length]",
        "output_interpretation": "digest_256,done,timeout,error",
    },
    "picorv32_cpu": {
        "partition": "GENERALIZATION_TRAIN",
        "operation": "BOUNDED_RISCV_PROGRAM_EXECUTION",
        "vectors": 48,
        "pilot": 12,
        "classes": ("ALU", "BRANCH", "LOAD_STORE", "CSR", "INTERRUPT", "MIXED"),
        "payload_interpretation": "little-endian RV32I instruction image",
        "output_interpretation": "architectural_signature,memory_signature,trap,retired_count,timeout,error",
    },
    "secworks_aes": {
        "partition": "GENERALIZATION_TRAIN",
        "operation": "AES_BLOCK_ENCRYPT_DECRYPT",
        "vectors": 64,
        "pilot": 16,
        "classes": ("NIST_KAT", "ZERO", "ONES", "COUNTING", "AVALANCHE", "RANDOM"),
        "payload_interpretation": "key_256[0:32] || block_128[32:48]",
        "output_interpretation": "result_block_128,ready,valid,timeout,error",
    },
    "secworks_sha256": {
        "partition": "GENERALIZATION_CALIBRATION",
        "operation": "SHA256_BLOCK_SEQUENCE",
        "vectors": 64,
        "pilot": 16,
        "classes": ("NIST_KAT", "ZERO", "ONES", "COUNTING", "AVALANCHE", "RANDOM"),
        "payload_interpretation": "one or more packed 512-bit message blocks",
        "output_interpretation": "digest_256,ready,digest_valid,timeout,error",
    },
}


def stop(message: str) -> None:
    raise SystemExit(f"STOP: {message}")


def require(condition: bool, message: str) -> None:
    if not condition:
        stop(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def npy_bytes(array: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.lib.format.write_array(stream, np.asarray(array), allow_pickle=False)
    return stream.getvalue()


def deterministic_npz(arrays: dict[str, np.ndarray]) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in sorted(arrays):
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o600 << 16
            archive.writestr(info, npy_bytes(arrays[name]), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    return stream.getvalue()


def expand(seed: bytes, length: int) -> bytes:
    output = bytearray()
    counter = 0
    while len(output) < length:
        output.extend(hashlib.sha256(seed + counter.to_bytes(4, "big")).digest())
        counter += 1
    return bytes(output[:length])


def seed_for(family_id: str, index: int) -> bytes:
    return b"\x00".join((SEED_DOMAIN, family_id.encode(), str(index).encode()))


def sha256_padding_block(message: bytes) -> bytes:
    require(len(message) <= 55, "single SHA-256 padding block message limit")
    padding = message + b"\x80"
    padding += b"\x00" * (56 - len(padding))
    return padding + (len(message) * 8).to_bytes(8, "big")


def cpu_program(stimulus_class: str, random_bytes: bytes, index: int) -> bytes:
    # Valid, bounded RV32I instruction words.  The pilot harness supplies the
    # memory handshake and termination policy at the next gate.
    instructions = {
        "ALU": [0x00100093, 0x00208113, 0x0020C1B3, 0x00310233],
        "BRANCH": [0x00000093, 0x00100113, 0x00208463, 0x00108093],
        "LOAD_STORE": [0x00002083, 0x00102223, 0x00402103, 0x00202423],
        "CSR": [0xC00020F3, 0xC0202173, 0xC01021F3, 0x00000013],
        "INTERRUPT": [0x00000013, 0x00000013, 0x00000013, 0x00000013],
        "MIXED": [0x00100093, 0x00208113, 0x00208463, 0x00102223],
    }[stimulus_class]
    words = list(instructions)
    for offset in range(0, 48, 4):
        immediate = int.from_bytes(random_bytes[offset:offset + 2], "little") & 0x7FF
        rd = 1 + ((index + offset // 4) % 15)
        words.append((immediate << 20) | (rd << 7) | 0x13)  # ADDI rd,x0,imm
    words.append(0x00100073)  # EBREAK termination marker
    return b"".join(word.to_bytes(4, "little") for word in words)


def make_request(family_id: str, index: int, stimulus_class: str) -> tuple[bytes, list[int], int]:
    random_data = expand(seed_for(family_id, index), PAYLOAD_BYTES)
    payload = bytearray(PAYLOAD_BYTES)
    control = [0] * CONTROL_WORDS
    control[5] = int.from_bytes(hashlib.sha256(seed_for(family_id, index)).digest()[:4], "big")
    control[6] = 1 if index % 8 == 7 else 0  # legal back-to-back schedule request
    control[7] = 1 if index == int(EXPECTED[family_id]["vectors"]) - 1 else 0  # timeout canary

    if family_id == "opentitan_hmac_sha256":
        key = random_data[:32]
        lengths = (0, 1, 31, 32, 55, 64, 96, 128)
        message_length = lengths[index % len(lengths)] if stimulus_class == "BOUNDARY_LENGTH" else 32
        message_length = min(message_length, PAYLOAD_BYTES - 32)
        message = random_data[32:32 + message_length]
        if stimulus_class == "ZERO":
            key, message = bytes(32), bytes(message_length)
        elif stimulus_class == "ONES":
            key, message = bytes([0xFF]) * 32, bytes([0xFF]) * message_length
        elif stimulus_class == "COUNTING":
            key = bytes(range(32)); message = bytes(i & 0xFF for i in range(message_length))
        elif stimulus_class == "ONEHOT":
            key = bytes(32); temp = bytearray(message_length or 1); temp[index % len(temp)] = 1 << (index % 8); message = bytes(temp)
        payload[:32] = key
        payload[32:32 + message_length] = message
        control[0] = 32 + message_length
        control[1] = 32
        control[2] = message_length
        control[3] = 4096
        control[4] = 256
        valid = 32 + message_length

    elif family_id == "picorv32_cpu":
        program = cpu_program(stimulus_class, random_data, index)
        payload[:len(program)] = program
        control[0] = len(program)
        control[1] = 0  # initial PC
        control[2] = 8192
        control[3] = index if stimulus_class == "INTERRUPT" else 0xFFFFFFFF
        control[4] = 32
        valid = len(program)

    elif family_id == "secworks_aes":
        key = random_data[:32]
        block = random_data[32:48]
        key_bits = 128 if index % 2 == 0 else 256
        direction = index % 2
        if stimulus_class == "NIST_KAT":
            key = bytes.fromhex("000102030405060708090a0b0c0d0e0f") + bytes(16)
            block = bytes.fromhex("00112233445566778899aabbccddeeff")
            key_bits = 128; direction = 0
        elif stimulus_class == "ZERO":
            key, block = bytes(32), bytes(16)
        elif stimulus_class == "ONES":
            key, block = bytes([0xFF]) * 32, bytes([0xFF]) * 16
        elif stimulus_class == "COUNTING":
            key, block = bytes(range(32)), bytes(range(16))
        elif stimulus_class == "AVALANCHE":
            key = bytes(32); temp = bytearray(16); temp[(index // 8) % 16] = 1 << (index % 8); block = bytes(temp)
        payload[:32] = key
        payload[32:48] = block
        control[0] = 48
        control[1] = key_bits
        control[2] = direction
        control[3] = 2048
        valid = 48

    elif family_id == "secworks_sha256":
        block = random_data[:64]
        if stimulus_class == "NIST_KAT":
            block = sha256_padding_block(b"abc")
        elif stimulus_class == "ZERO":
            block = bytes(64)
        elif stimulus_class == "ONES":
            block = bytes([0xFF]) * 64
        elif stimulus_class == "COUNTING":
            block = bytes(range(64))
        elif stimulus_class == "AVALANCHE":
            temp = bytearray(64); temp[(index // 8) % 64] = 1 << (index % 8); block = bytes(temp)
        payload[:64] = block
        control[0] = 64
        control[1] = 1  # block count
        control[2] = 1  # init
        control[3] = 4096
        control[4] = 256
        valid = 64
    else:
        stop(f"unsupported family: {family_id}")

    # Schedule seed makes every complete logical request unique while leaving
    # the intended payload class intact.
    return bytes(payload), control, valid


def adapter_module_text() -> str:
    return '''#!/usr/bin/env python3
"""Frozen Stage 12C-1G logical transaction adapters.

These functions decode a family-neutral request into a family transaction.
They do not inject faults, expose fault identity, simulate RTL, or predict a
response.  The next gate binds and functionally validates them against frozen
TRAIN/CALIBRATION netlists.
"""

from __future__ import annotations

FAMILIES = (
    "opentitan_hmac_sha256", "picorv32_cpu", "secworks_aes", "secworks_sha256"
)


def _check(family_id, payload, control):
    if family_id not in FAMILIES:
        raise ValueError("unregistered family")
    if not isinstance(payload, (bytes, bytearray)) or len(payload) != 256:
        raise ValueError("payload must contain exactly 256 bytes")
    if len(control) != 8:
        raise ValueError("control must contain exactly 8 uint32-compatible words")
    if any(int(value) < 0 or int(value) > 0xFFFFFFFF for value in control):
        raise ValueError("control word outside uint32 range")


def adapt_request(family_id, transaction_id, payload, control):
    _check(family_id, payload, control)
    p = bytes(payload)
    c = tuple(int(value) for value in control)
    common = {"family_id": family_id, "transaction_id": int(transaction_id),
              "cycle_budget": c[3], "schedule_seed": c[5],
              "back_to_back": bool(c[6]), "timeout_canary": bool(c[7])}
    if family_id == "opentitan_hmac_sha256":
        common.update(operation="HMAC_SHA256_FIXED_AND_VARIABLE_MESSAGE",
                      key=p[:32], message=p[32:32+c[2]], message_length=c[2], sha_mode=c[4])
    elif family_id == "picorv32_cpu":
        common.update(operation="BOUNDED_RISCV_PROGRAM_EXECUTION",
                      program_image=p[:c[0]], initial_pc=c[1], cycle_budget=c[2],
                      interrupt_cycle=c[3])
    elif family_id == "secworks_aes":
        common.update(operation="AES_BLOCK_ENCRYPT_DECRYPT", key=p[:32], block=p[32:48],
                      key_length=c[1], direction=c[2])
    else:
        common.update(operation="SHA256_BLOCK_SEQUENCE", message_blocks=p[:c[0]],
                      block_count=c[1], init=c[2])
    return common


def model_query_fields(transaction, observed_response, validity_mask, golden_difference,
                       completion_cycle_delta, timeout=False, protocol_error=False):
    prohibited = {"fault_id", "site_id", "cell_name", "net_name", "stuck_value",
                  "target_site", "target_polarity", "raw_injection_selector"}
    if prohibited.intersection(transaction):
        raise ValueError("fault identity or truth leaked into transaction")
    return {"family_id": transaction["family_id"],
            "transaction_id": transaction["transaction_id"],
            "observed_response": bytes(observed_response),
            "response_validity_mask": bytes(validity_mask),
            "golden_difference": bytes(golden_difference),
            "completion_cycle_delta": int(completion_cycle_delta),
            "timeout": bool(timeout), "protocol_error": bool(protocol_error)}
'''


def verify_manifest_outputs(manifest: dict[str, Any]) -> None:
    outputs = manifest.get("outputs")
    require(isinstance(outputs, dict) and outputs, "12C-1F manifest output records")
    for key, item in outputs.items():
        require(isinstance(item, dict), f"invalid 12C-1F manifest item: {key}")
        path = ROOT / str(item.get("path", ""))
        require(path.is_file(), f"missing 12C-1F output: {item.get('path')}")
        require(rel(path) == item.get("path"), f"12C-1F manifest path: {key}")
        require(sha256(path) == item.get("sha256"), f"12C-1F manifest SHA: {rel(path)}")
        require(path.stat().st_size == int(item.get("bytes", -1)), f"12C-1F manifest size: {rel(path)}")


def verify_inputs() -> tuple[dict[str, Any], list[dict[str, str]], list[dict[str, str]]]:
    print("STAGE 12C-1G — TRAIN/CALIBRATION PORTABLE ADAPTER AND DETERMINISTIC VECTOR GENERATION FREEZE")
    print("FROZEN INPUT VERIFICATION")
    for path, expected in PINNED.items():
        require(path.is_file(), f"missing frozen input: {rel(path)}")
        require(sha256(path) == expected, f"frozen input SHA: {rel(path)}")
        print(f"  {path.name:<112}: OK")

    manifest = load_json(MANIFEST_1F)
    audit = load_json(AUDIT_1F)
    transaction = load_json(TRANSACTION_CONTRACT)
    vectors = load_json(VECTOR_CONTRACT)
    access = load_json(ACCESS_POLICY_1F)
    family_rows = read_csv(FAMILY_CSV_1F)
    vector_rows = read_csv(VECTOR_BUDGET_1F)

    require(manifest.get("status") == "PASS", "12C-1F manifest status")
    require(audit.get("status") == "PASS", "12C-1F audit status")
    require(audit.get("contracts") == "FROZEN", "12C-1F contracts")
    require(audit.get("vector_budgets") == "FROZEN / 240 TOTAL / 60 PILOT", "12C-1F vector budget")
    require(audit.get("fault_identity_input") == "PROHIBITED", "12C-1F identity firewall")
    require(audit.get("adapter_vector_generation") == "AUTHORIZED NEXT GATE / NOT STARTED", "generation authorization")
    require(audit.get("independent_test") == "LOCKED / 2 FAMILIES", "TEST lock")
    require(audit.get("holdout") == "SEALED / 1 FAMILY", "HOLDOUT seal")
    require(audit.get("independent_test_validation_holdout_access") == [0, 0, 0], "protected access")
    require(transaction.get("adapter_generation") == "AUTHORIZED NEXT GATE / NOT PERFORMED", "adapter authorization")
    require(vectors.get("vector_generation") == "AUTHORIZED NEXT GATE / NOT PERFORMED", "vector authorization")
    require(access.get("INDEPENDENT_CIRCUIT_TEST", "").startswith("LOCKED"), "TEST access policy")
    require(access.get("GENERALIZATION_HOLDOUT", "").startswith("SEALED"), "HOLDOUT access policy")
    verify_manifest_outputs(manifest)

    require([row["family_id"] for row in family_rows] == list(AUTHORIZED), "family registry order")
    require([row["family_id"] for row in vector_rows] == list(AUTHORIZED), "vector registry order")
    by_family = {row["family_id"]: row for row in family_rows}
    by_budget = {row["family_id"]: row for row in vector_rows}
    for family_id in AUTHORIZED:
        expected = EXPECTED[family_id]
        family = by_family[family_id]
        budget = by_budget[family_id]
        require(family["partition"] == expected["partition"], f"partition: {family_id}")
        require(family["operation"] == expected["operation"], f"operation: {family_id}")
        require(family["fault_identity_input"] == "PROHIBITED", f"identity firewall: {family_id}")
        require(int(budget["total_vector_budget"]) == expected["vectors"], f"vector count: {family_id}")
        require(int(budget["pilot_vector_budget"]) == expected["pilot"], f"pilot count: {family_id}")
        require(tuple(budget["stimulus_classes"].split(",")) == expected["classes"], f"classes: {family_id}")
    print("  Contract authorization, four family budgets, identity firewall and partition locks             : PASS")
    return manifest, family_rows, vector_rows


def generate() -> tuple[dict[str, np.ndarray], list[dict[str, Any]], list[dict[str, Any]], dict[str, bytes]]:
    family_index: list[int] = []
    partition_index: list[int] = []
    transaction_ids: list[int] = []
    local_indices: list[int] = []
    pilot_mask: list[bool] = []
    stimulus_classes: list[bytes] = []
    payloads: list[np.ndarray] = []
    payload_valid_masks: list[np.ndarray] = []
    controls: list[np.ndarray] = []
    request_hashes: list[np.ndarray] = []
    inventory: list[dict[str, Any]] = []
    adapters: list[dict[str, Any]] = []
    descriptor_payloads: dict[str, bytes] = {}
    transaction_id = 0

    for family_number, family_id in enumerate(AUTHORIZED):
        spec = EXPECTED[family_id]
        adapter = {
            "adapter_version": "CIRCUITSAGE-HMAC-V2.2-PORTABLE-LOGICAL-ADAPTER-12C1G-v1",
            "stage": STAGE,
            "status": "GENERATED / FROZEN / FUNCTIONAL VALIDATION DEFERRED",
            "family_id": family_id,
            "partition": spec["partition"],
            "operation": spec["operation"],
            "payload_bytes": PAYLOAD_BYTES,
            "control_words": CONTROL_WORDS,
            "payload_interpretation": spec["payload_interpretation"],
            "output_interpretation": spec["output_interpretation"],
            "request_decoder": "circuitsage_hmac_v2_2_portable_adapters_12c1g.adapt_request",
            "fault_identity_input": "PROHIBITED / ABSENT",
            "normal_faulty_schedule": "IDENTICAL BY CONTRACT",
            "upstream_rtl_modified": False,
            "synthesized_netlist_modified": False,
            "rtl_binding": "NOT PERFORMED — SEPARATE FUNCTIONAL-VALIDATION GATE REQUIRED",
            "simulation_calls": 0,
        }
        descriptor = ADAPTER_DIR / f"{family_id}_portable_adapter_12c1g.json"
        payload = canonical_json(adapter)
        descriptor_payloads[rel(descriptor)] = payload
        adapters.append({
            "family_id": family_id,
            "partition": spec["partition"],
            "operation": spec["operation"],
            "payload_bytes": PAYLOAD_BYTES,
            "control_words": CONTROL_WORDS,
            "total_vectors": spec["vectors"],
            "pilot_vectors": spec["pilot"],
            "fault_identity_input": "PROHIBITED / ABSENT",
            "generation_status": "GENERATED / FROZEN",
            "functional_validation": "DEFERRED",
            "descriptor": rel(descriptor),
            "descriptor_sha256": sha256_bytes(payload),
        })

        seen: set[str] = set()
        for index in range(int(spec["vectors"])):
            stimulus_class = spec["classes"][index % len(spec["classes"])]
            payload, control, valid_bytes = make_request(family_id, index, stimulus_class)
            require(0 <= valid_bytes <= PAYLOAD_BYTES, f"valid payload bytes: {family_id}/{index}")
            validity = np.zeros(PAYLOAD_BYTES, dtype=np.uint8)
            validity[:valid_bytes] = 1
            control_array = np.asarray(control, dtype=np.uint32)
            request_material = (family_id.encode() + b"\x00" + payload + validity.tobytes()
                                + control_array.astype("<u4", copy=False).tobytes())
            request_digest = hashlib.sha256(request_material).digest()
            request_hex = request_digest.hex()
            require(request_hex not in seen, f"duplicate logical request: {family_id}/{index}")
            seen.add(request_hex)

            is_pilot = index < int(spec["pilot"])
            family_index.append(family_number)
            partition_index.append(0 if spec["partition"] == "GENERALIZATION_TRAIN" else 1)
            transaction_ids.append(transaction_id)
            local_indices.append(index)
            pilot_mask.append(is_pilot)
            stimulus_classes.append(stimulus_class.encode())
            payloads.append(np.frombuffer(payload, dtype=np.uint8).copy())
            payload_valid_masks.append(validity)
            controls.append(control_array)
            request_hashes.append(np.frombuffer(request_digest, dtype=np.uint8).copy())
            inventory.append({
                "transaction_id": transaction_id,
                "family_id": family_id,
                "partition": spec["partition"],
                "family_vector_index": index,
                "stimulus_class": stimulus_class,
                "pilot": "YES" if is_pilot else "NO",
                "valid_payload_bytes": valid_bytes,
                "cycle_budget": control[2] if family_id == "picorv32_cpu" else control[3],
                "request_sha256": request_hex,
                "schedule_mode": "BACK_TO_BACK" if control[6] else "STANDARD",
                "timeout_canary": "YES" if control[7] else "NO",
                "fault_identity_fields": 0,
            })
            transaction_id += 1

    arrays = {
        "family_ids": np.asarray(AUTHORIZED, dtype="S32"),
        "family_index": np.asarray(family_index, dtype=np.uint8),
        "partition_index": np.asarray(partition_index, dtype=np.uint8),
        "transaction_id": np.asarray(transaction_ids, dtype=np.uint64),
        "family_vector_index": np.asarray(local_indices, dtype=np.uint16),
        "pilot_mask": np.asarray(pilot_mask, dtype=np.bool_),
        "stimulus_class": np.asarray(stimulus_classes, dtype="S32"),
        "payload": np.stack(payloads).astype(np.uint8, copy=False),
        "payload_valid_byte_mask": np.stack(payload_valid_masks).astype(np.uint8, copy=False),
        "control": np.stack(controls).astype(np.uint32, copy=False),
        "request_sha256": np.stack(request_hashes).astype(np.uint8, copy=False),
    }
    return arrays, inventory, adapters, descriptor_payloads


def output_paths(descriptor_payloads: dict[str, bytes]) -> list[Path]:
    descriptors = [ROOT / name for name in sorted(descriptor_payloads)]
    return descriptors + [ADAPTER_REGISTRY_CSV, ADAPTER_REGISTRY_JSON, ADAPTER_MODULE,
                           VECTORS, VECTOR_INVENTORY, VECTOR_SCHEMA, REPLAY, REPORT]


def execute() -> None:
    require(ROOT.name == "vlsi_fault_detection_v2", f"place script in project root, not {ROOT}")
    for path in (ADAPTER_REGISTRY_CSV, ADAPTER_REGISTRY_JSON, ADAPTER_MODULE, VECTORS,
                 VECTOR_INVENTORY, VECTOR_SCHEMA, REPLAY, REPORT, MANIFEST, AUDIT):
        require(not path.exists(), f"Stage {STAGE} output already exists: {rel(path)}")

    previous_manifest, _, _ = verify_inputs()
    timestamp = utc_now()
    arrays, inventory, adapters, descriptor_payloads = generate()
    arrays_replay, inventory_replay, adapters_replay, descriptors_replay = generate()

    vector_payload = deterministic_npz(arrays)
    require(vector_payload == deterministic_npz(arrays_replay), "deterministic vector NPZ replay")
    require(inventory == inventory_replay, "deterministic inventory replay")
    require(adapters == adapters_replay, "deterministic adapter registry replay")
    require(descriptor_payloads == descriptors_replay, "deterministic adapter descriptor replay")
    require(len(inventory) == 240, "total vector count")
    require(sum(row["pilot"] == "YES" for row in inventory) == 60, "pilot vector count")
    require(int(np.sum(arrays["partition_index"] == 0)) == 176, "TRAIN vector count")
    require(int(np.sum(arrays["partition_index"] == 1)) == 64, "CALIBRATION vector count")

    adapter_fields = ["family_id", "partition", "operation", "payload_bytes", "control_words",
                      "total_vectors", "pilot_vectors", "fault_identity_input", "generation_status",
                      "functional_validation", "descriptor", "descriptor_sha256"]
    inventory_fields = ["transaction_id", "family_id", "partition", "family_vector_index",
                        "stimulus_class", "pilot", "valid_payload_bytes", "cycle_budget",
                        "request_sha256", "schedule_mode", "timeout_canary", "fault_identity_fields"]
    adapter_csv_payload = csv_bytes(adapters, adapter_fields)
    inventory_payload = csv_bytes(inventory, inventory_fields)
    adapter_json = {
        "registry_version": "CIRCUITSAGE-HMAC-V2.2-PORTABLE-ADAPTER-REGISTRY-12C1G-v1",
        "stage": STAGE,
        "status": "FROZEN",
        "adapter_scope": "LOGICAL TRANSACTION DECODING; RTL BINDING DEFERRED",
        "families": adapters,
        "functional_validation": "NOT PERFORMED / SEPARATE GATE REQUIRED",
        "simulation_calls": 0,
        "fault_injection_calls": 0,
    }
    adapter_json_payload = canonical_json(adapter_json)
    module_payload = adapter_module_text().encode()
    compile(module_payload.decode(), str(ADAPTER_MODULE), "exec")

    schema = {
        "schema_version": "CIRCUITSAGE-HMAC-V2.2-DETERMINISTIC-VECTOR-SCHEMA-12C1G-v1",
        "stage": STAGE,
        "status": "FROZEN",
        "seed_domain": SEED_DOMAIN.decode(),
        "ordering": "AUTHORIZED FAMILY ORDER THEN ASCENDING FAMILY VECTOR INDEX",
        "arrays": {
            name: {"dtype": str(array.dtype), "shape": list(array.shape)}
            for name, array in arrays.items()
        },
        "control_words": {
            "0": "payload_length_bytes",
            "1": "family-specific key length, initial PC, or block count",
            "2": "family-specific message length, direction, cycle budget, or init",
            "3": "contracted timeout/cycle budget except PicoRV32 interrupt schedule is encoded separately by adapter",
            "4": "family mode or architectural width",
            "5": "deterministic schedule seed",
            "6": "legal back-to-back schedule flag",
            "7": "timeout-canary flag; excluded from accepted functional-vector metrics",
        },
        "model_input_identity_firewall": "FAULT IDENTITY, SITE, POLARITY AND TRUTH FIELDS ABSENT",
        "expected_response_fields": "ABSENT — RESPONSE GENERATION NOT AUTHORIZED",
        "test_holdout_vectors": "NOT GENERATED / NOT ACCESSED",
    }
    schema_payload = canonical_json(schema)
    replay = {
        "replay_version": "CIRCUITSAGE-HMAC-V2.2-ADAPTER-VECTOR-REPLAY-12C1G-v1",
        "stage": STAGE,
        "status": "PASS",
        "adapter_descriptors": "PASS / BYTE-EXACT / 4 OF 4",
        "adapter_registry_csv": "PASS / BYTE-EXACT",
        "adapter_registry_json": "PASS / BYTE-EXACT",
        "adapter_module": "PASS / SOURCE-EXACT / PYTHON-COMPILE-PASS",
        "vector_npz": "PASS / BYTE-EXACT",
        "vector_inventory": "PASS / BYTE-EXACT",
        "total_vectors": 240,
        "pilot_vectors": 60,
        "duplicate_requests_within_family": 0,
        "fault_identity_fields": 0,
    }
    replay_payload = canonical_json(replay)
    report = f"""# CircuitSage-HMAC V2.2 Portable Adapters and Vectors — Stage 12C-1G

Stage 12C-1G generated and froze four family-specific logical transaction
adapters and 240 deterministic TRAIN/CALIBRATION request vectors. The first 60
contracted vectors form the family-stratified pilot subset.

Every artifact passed byte-exact in-memory replay. Fault identity, fault site,
stuck polarity and supervision truth are absent. No expected responses were
generated because RTL execution is not authorized in this gate.

These adapters decode the common transaction envelope into HMAC, bounded
PicoRV32 program, AES-block, and SHA-256-block requests. They have not yet been
bound to or functionally validated against RTL. That validation and any pilot
campaign require a separate authorization stage.

Independent TEST remains locked, HOLDOUT remains sealed, and the future hybrid
release name remains **{FUTURE_BRAND}**.
""".encode()

    for name, payload in sorted(descriptor_payloads.items()):
        frozen_write(ROOT / name, payload)
    frozen_write(ADAPTER_REGISTRY_CSV, adapter_csv_payload)
    frozen_write(ADAPTER_REGISTRY_JSON, adapter_json_payload)
    frozen_write(ADAPTER_MODULE, module_payload)
    frozen_write(VECTORS, vector_payload)
    frozen_write(VECTOR_INVENTORY, inventory_payload)
    frozen_write(VECTOR_SCHEMA, schema_payload)
    frozen_write(REPLAY, replay_payload)
    frozen_write(REPORT, report)

    # Compile from the frozen source in memory without importing, executing, or
    # creating an unmanifested __pycache__ artifact.
    compile(ADAPTER_MODULE.read_text(encoding="utf-8"), str(ADAPTER_MODULE), "exec")
    generated_outputs = output_paths(descriptor_payloads)
    manifest = {
        "manifest_version": "CIRCUITSAGE-HMAC-V2.2-PORTABLE-ADAPTER-VECTOR-MANIFEST-12C1G-v1",
        "stage": STAGE,
        "status": "PASS",
        "created_at": timestamp,
        "stage_source": record(Path(__file__).resolve()),
        "frozen_inputs": {rel(path): record(path) for path in PINNED},
        "predecessor_output_count_verified": len(previous_manifest["outputs"]),
        "outputs": {rel(path): record(path) for path in generated_outputs},
        "authorized_families": list(AUTHORIZED),
        "train_families": 3,
        "calibration_families": 1,
        "adapter_descriptors_generated": 4,
        "vector_records_created": 240,
        "pilot_vector_records": 60,
        "expected_response_records_created": 0,
        "simulation_calls": 0,
        "fault_injection_calls": 0,
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
        "audit_version": "CIRCUITSAGE-HMAC-V2.2-PORTABLE-ADAPTER-VECTOR-GENERATION-FREEZE-12C1G-v1",
        "stage": STAGE,
        "status": "PASS",
        "adapter_generation": "COMPLETED / FROZEN / 4 OF 4",
        "adapter_scope": "PORTABLE LOGICAL TRANSACTION ADAPTERS",
        "rtl_functional_validation": "NOT PERFORMED / NOT AUTHORIZED",
        "vector_generation": "COMPLETED / FROZEN / 240 TOTAL / 60 PILOT",
        "deterministic_replay": "PASS / BYTE-EXACT",
        "train_calibration_families": "3 / 1",
        "fault_identity_in_vectors": "ABSENT / PROHIBITED",
        "expected_responses": "0 / NOT AUTHORIZED",
        "simulation_fault_injection_dataset": [0, 0, 0],
        "training_inference": [0, 0],
        "independent_test": "LOCKED / 2 FAMILIES",
        "holdout": "SEALED / 1 FAMILY",
        "independent_test_validation_holdout_access": [0, 0, 0],
        "upstream_rtl_modified": False,
        "synthesized_netlists_modified": False,
        "future_combined_model_brand": FUTURE_BRAND,
        "adapter_registry_record": record(ADAPTER_REGISTRY_JSON),
        "adapter_module_record": record(ADAPTER_MODULE),
        "vectors_record": record(VECTORS),
        "vector_inventory_record": record(VECTOR_INVENTORY),
        "vector_schema_record": record(VECTOR_SCHEMA),
        "replay_record": record(REPLAY),
        "report_record": record(REPORT),
        "manifest_record": record(MANIFEST),
        "next_gate": "STAGE 12C-1H — TRAIN/CALIBRATION ADAPTER FUNCTIONAL-VALIDATION AND PILOT-CAMPAIGN AUTHORIZATION FREEZE",
    }
    frozen_write(AUDIT, canonical_json(audit))

    # Final byte-level replay from the frozen files.
    require(VECTORS.read_bytes() == deterministic_npz(generate()[0]), "frozen vector replay")
    require(ADAPTER_REGISTRY_CSV.read_bytes() == adapter_csv_payload, "frozen adapter CSV replay")
    require(ADAPTER_REGISTRY_JSON.read_bytes() == canonical_json(load_json(ADAPTER_REGISTRY_JSON)), "adapter JSON canonical replay")
    require(VECTOR_INVENTORY.read_bytes() == inventory_payload, "frozen vector inventory replay")
    require(VECTOR_SCHEMA.read_bytes() == canonical_json(load_json(VECTOR_SCHEMA)), "schema canonical replay")
    require(REPLAY.read_bytes() == canonical_json(load_json(REPLAY)), "replay canonical replay")
    for name, payload in descriptor_payloads.items():
        require((ROOT / name).read_bytes() == payload, f"frozen descriptor replay: {name}")
    for path, expected in PINNED.items():
        require(sha256(path) == expected, f"frozen predecessor modified: {rel(path)}")

    print("\nSTAGE 12C-1G — TRAIN/CALIBRATION PORTABLE ADAPTER AND DETERMINISTIC VECTOR GENERATION FREEZE")
    print(f"{'Status':<82}: PASS")
    print(f"{'Adapter generation':<82}: COMPLETED / FROZEN / 4 OF 4")
    print(f"{'Adapter scope':<82}: PORTABLE LOGICAL TRANSACTION ADAPTERS")
    print(f"{'RTL functional validation':<82}: NOT PERFORMED / NOT AUTHORIZED")
    print(f"{'TRAIN / CALIBRATION families':<82}: 3 / 1")
    print(f"{'Total / pilot vectors':<82}: 240 / 60")
    print(f"{'Deterministic replay':<82}: PASS / BYTE-EXACT")
    print(f"{'Fault identity in vectors':<82}: ABSENT / PROHIBITED")
    print(f"{'Expected response generation':<82}: 0 / NOT AUTHORIZED")
    print(f"{'Simulation / fault injection / dataset records':<82}: 0 / 0 / 0")
    print(f"{'Training / inference':<82}: 0 / 0")
    print(f"{'Independent TEST / HOLDOUT':<82}: LOCKED 2 / SEALED 1")
    print(f"{'Independent TEST / VALIDATION / HOLDOUT access':<82}: 0 / 0 / 0")
    print(f"{'Adapters':<82}: {ADAPTER_REGISTRY_JSON}")
    print(f"{'Adapters SHA':<82}: {sha256(ADAPTER_REGISTRY_JSON)}")
    print(f"{'Vectors':<82}: {VECTORS}")
    print(f"{'Vectors SHA':<82}: {sha256(VECTORS)}")
    print(f"{'Manifest':<82}: {MANIFEST}")
    print(f"{'Manifest SHA':<82}: {sha256(MANIFEST)}")
    print(f"{'Audit':<82}: {AUDIT}")
    print(f"{'Audit SHA':<82}: {sha256(AUDIT)}")
    print(f"{'Next gate':<82}: STAGE 12C-1H — TRAIN/CALIBRATION ADAPTER FUNCTIONAL-VALIDATION AND PILOT-CAMPAIGN AUTHORIZATION FREEZE")


def status() -> None:
    print("STAGE 12C-1G — PORTABLE-ADAPTER/VECTOR STATUS")
    if not AUDIT.is_file() or not MANIFEST.is_file():
        print("Status                    : NOT FROZEN")
        print(f"Audit                     : {AUDIT}")
        return
    audit = load_json(AUDIT)
    manifest = load_json(MANIFEST)
    require(audit.get("status") == "PASS", "audit status")
    require(manifest.get("status") == "PASS", "manifest status")
    for item in manifest.get("outputs", {}).values():
        path = ROOT / item["path"]
        require(path.is_file(), f"missing frozen output: {item['path']}")
        require(sha256(path) == item["sha256"], f"frozen output SHA: {item['path']}")
        require(path.stat().st_size == int(item["bytes"]), f"frozen output size: {item['path']}")
    require(sha256(MANIFEST) == audit["manifest_record"]["sha256"], "manifest audit SHA")
    print("Status                    : PASS / FROZEN")
    print("Adapters                  : 4/4")
    print("Total / pilot vectors     : 240 / 60")
    print("Deterministic replay      : PASS / BYTE-EXACT")
    print("RTL functional validation : NOT PERFORMED")
    print(f"Audit                     : {AUDIT}")
    print(f"Audit SHA                 : {sha256(AUDIT)}")


def self_test() -> None:
    require(tuple(sorted(EXPECTED)) == AUTHORIZED, "family identity set")
    require(sum(int(item["vectors"]) for item in EXPECTED.values()) == 240, "total vectors")
    require(sum(int(item["pilot"]) for item in EXPECTED.values()) == 60, "pilot vectors")
    require(sum(item["partition"] == "GENERALIZATION_TRAIN" for item in EXPECTED.values()) == 3, "TRAIN families")
    require(deterministic_npz({"b": np.arange(3), "a": np.arange(2, dtype=np.uint8)}) ==
            deterministic_npz({"b": np.arange(3), "a": np.arange(2, dtype=np.uint8)}), "NPZ replay")
    arrays, inventory, adapters, descriptors = generate()
    require(arrays["payload"].shape == (240, PAYLOAD_BYTES), "payload shape")
    require(arrays["control"].shape == (240, CONTROL_WORDS), "control shape")
    require(len(inventory) == 240 and len(adapters) == 4 and len(descriptors) == 4, "generated counts")
    require(len({row["request_sha256"] for row in inventory}) == 240, "global request digest uniqueness")
    compile(adapter_module_text(), "<adapter-module>", "exec")
    print("Stage 12C-1G self-test: PASS")


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
