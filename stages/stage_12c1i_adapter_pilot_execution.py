#!/usr/bin/env python3
"""Stage 12C-1I: adapter validation and bounded pilot execution freeze.

This gate verifies the frozen Stage 12C-1H authorization, functionally binds
all four TRAIN/CALIBRATION adapters to their frozen synthesized netlists,
executes deterministic fault-free RTL validation twice, and only then runs the
authorized bounded single-persistent-SA0/SA1 pilot campaign.

Execution is resumable and sequential.  One instrumented binary is built per
family; simulation is checkpointed after every 64-site batch.  Fault identity
is written only to the separate target/catalog artifacts and is never present
in the model-facing feature bundle.  INDEPENDENT_TEST, VALIDATION and HOLDOUT
remain inaccessible, and this stage performs no model training or inference.
"""

from __future__ import annotations

import argparse
import copy
import csv
import fcntl
import hashlib
import hmac
import io
import json
import os
import shutil
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


STAGE = "12C-1I"
ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "config/v2_2"
RESULT = ROOT / "results/circuitsage_hmac_v2_12c1"
WORK = RESULT / "adapter_pilot_execution_12c1i"
BUILD = ROOT / "build/circuitsage_hmac_v2_12c1/adapter_pilot_12c1i"
RAW = WORK / "raw_batches"
PER_BATCH = WORK / "per_batch_npz"
MEMORY = WORK / "vector_memory"

SOURCE_1H = ROOT / "stage_12c1h_adapter_pilot_authorization.py"
VALIDATION_CONTRACT = CONFIG / "circuitsage_hmac_v2_2_adapter_functional_validation_contract_12c1h.json"
PILOT_CONTRACT = CONFIG / "circuitsage_hmac_v2_2_pilot_campaign_execution_contract_12c1h.json"
ACCEPTANCE_CONTRACT = CONFIG / "circuitsage_hmac_v2_2_adapter_pilot_acceptance_contract_12c1h.json"
AUTHORIZATION = CONFIG / "circuitsage_hmac_v2_2_adapter_pilot_execution_authorization_12c1h.json"
AUTH_WORK = RESULT / "adapter_pilot_authorization_12c1h"
EXECUTION_PLAN = AUTH_WORK / "circuitsage_hmac_v2_2_adapter_pilot_execution_plan_12c1h.csv"
VALIDATION_REGISTRY = AUTH_WORK / "circuitsage_hmac_v2_2_adapter_validation_registry_12c1h.json"
PREFLIGHT_1H = AUTH_WORK / "circuitsage_hmac_v2_2_adapter_pilot_preflight_12c1h.json"
MANIFEST_1H = RESULT / "circuitsage_hmac_v2_2_adapter_pilot_authorization_manifest_12c1h.json"
AUDIT_1H = RESULT / "circuitsage_hmac_v2_2_adapter_pilot_authorization_freeze_12c1h.json"

WORK_1G = RESULT / "portable_adapter_vector_generation_12c1g"
VECTORS_1G = WORK_1G / "circuitsage_hmac_v2_2_deterministic_transaction_vectors_12c1g.npz"
ADAPTERS_1G = WORK_1G / "circuitsage_hmac_v2_2_portable_adapter_registry_12c1g.json"
WORK_1E = RESULT / "train_calibration_synthesis_12c1e"
MANIFEST_1E = RESULT / "circuitsage_hmac_v2_2_elaboration_synthesis_manifest_12c1e.json"
AUDIT_1E = RESULT / "circuitsage_hmac_v2_2_elaboration_wrapper_generic_synthesis_freeze_12c1e.json"
CAMPAIGN_BUDGET_1F = RESULT / "transaction_fault_campaign_contract_12c1f/circuitsage_hmac_v2_2_campaign_budget_registry_12c1f.csv"
HMAC_BRIDGE_SOURCE = ROOT / "rtl/hmac/opentitan_hmac_sha256_bridge.sv"
HMAC_BRIDGE_SHA256 = "82b8bcff3fcd37e0231ccd2463c7a8d44a596625f746ee7a6006fa0c1b9d5356"

CHECKPOINT = WORK / "circuitsage_hmac_v2_2_adapter_pilot_checkpoint_12c1i.json"
LOCK = WORK / ".stage_12c1i.lock"
VALIDATION_RESULTS = WORK / "circuitsage_hmac_v2_2_adapter_functional_validation_results_12c1i.csv"
FAULT_CATALOG = WORK / "circuitsage_hmac_v2_2_pilot_fault_catalog_12c1i.csv"
FEATURES = WORK / "circuitsage_hmac_v2_2_pilot_features_12c1i.npz"
TARGETS = WORK / "circuitsage_hmac_v2_2_pilot_targets_12c1i.npz"
SIGNATURES = WORK / "circuitsage_hmac_v2_2_pilot_signature_summary_12c1i.csv"
METRICS = WORK / "circuitsage_hmac_v2_2_pilot_metrics_12c1i.json"
BOOTSTRAP = WORK / "circuitsage_hmac_v2_2_pilot_site_bootstrap_12c1i.csv"
SCHEMA = WORK / "circuitsage_hmac_v2_2_pilot_dataset_schema_12c1i.json"
REPORT = WORK / "circuitsage_hmac_v2_2_adapter_pilot_execution_report_12c1i.md"
MANIFEST = RESULT / "circuitsage_hmac_v2_2_adapter_pilot_execution_manifest_12c1i.json"
AUDIT = RESULT / "circuitsage_hmac_v2_2_adapter_functional_validation_pilot_execution_freeze_12c1i.json"

PINNED = {
    SOURCE_1H: "fdaf67b91279011bfcdde24dc829607ce5b9692eab9ab32396f454eb37a01e21",
    VALIDATION_CONTRACT: "64c892805d7ee53a50465ad7ac63d43307190f2689ed016a04286404ca175cd2",
    PILOT_CONTRACT: "1e0a3846dea192a0520c59d5a24ac4b884a0e4e52e590937b9c6268b14195693",
    AUTHORIZATION: "f55175877bc35e496c528a5f6d49c278e683e9acf1567f9c6381aa9cc8d294f3",
    MANIFEST_1H: "ceda76b97764dfbfaaf2d81f284d66169b78d1ee31542fb252c51f6b03bade8f",
    AUDIT_1H: "3b29e70c8d8fab32b29c729f6168f4bbadf078e6359dd5542cf200aea286b810",
    VECTORS_1G: "0d8fce6c73c5d484fc9880ef97c0663b6dbfc9cbda73fcbd413f1970f418fa7b",
    ADAPTERS_1G: "296b15254debb27f336511ee2d6f1db1745f668444866f7d0ccdee35dab91fb2",
    MANIFEST_1E: "4d3d606e174ae06589e0e537486a15e042521fe9a02d49f9b4e1c596198476b6",
    AUDIT_1E: "3ac4f454173166ee6848e2617c7072742665e7166345ce5ffdb837f574d59220",
    CAMPAIGN_BUDGET_1F: "46a5fe2abb62051c7aa2daaf2c8503acd7fceacb29b907b5a6a5d8a56663d7c1",
}

AUTHORIZED = (
    "opentitan_hmac_sha256",
    "picorv32_cpu",
    "secworks_aes",
    "secworks_sha256",
)
TOPS = {
    "opentitan_hmac_sha256": "hmac_core",
    "picorv32_cpu": "picorv32",
    "secworks_aes": "aes_core",
    "secworks_sha256": "sha256_core",
}
PARTITIONS = {
    "opentitan_hmac_sha256": "GENERALIZATION_TRAIN",
    "picorv32_cpu": "GENERALIZATION_TRAIN",
    "secworks_aes": "GENERALIZATION_TRAIN",
    "secworks_sha256": "GENERALIZATION_CALIBRATION",
}
PILOT_VECTORS = {
    "opentitan_hmac_sha256": 16,
    "picorv32_cpu": 12,
    "secworks_aes": 16,
    "secworks_sha256": 16,
}
VALIDATION_VECTORS = {key: value + 1 for key, value in PILOT_VECTORS.items()}
SITES_PER_FAMILY = 2048
FAULTS_PER_FAMILY = SITES_PER_FAMILY * 2
TOTAL_SITES = SITES_PER_FAMILY * 4
TOTAL_FAULTS = FAULTS_PER_FAMILY * 4
BATCH_SITES = 64
BATCHES_PER_FAMILY = SITES_PER_FAMILY // BATCH_SITES
TOTAL_BATCHES = BATCHES_PER_FAMILY * 4
TOTAL_PILOT_VECTORS = sum(PILOT_VECTORS.values())
MAX_PILOT_TRANSACTIONS = sum(FAULTS_PER_FAMILY * value for value in PILOT_VECTORS.values())
MIN_FREE_GIB = 20
BOOTSTRAP_REPLICATES = 1000
FUTURE_BRAND = "Faultiva 1.0 — Hybrid VLSI Fault Intelligence"

CSV_FIELDS = [
    "family_id", "mode", "batch_id", "run_type", "site_rank", "stuck_value",
    "vector_rank", "transaction_id", "cycles", "timed_out", "protocol_error",
    "unknown", "expected_response", "actual_response",
]
FAULT_FIELDS = [
    "fault_instance_index", "opaque_fault_id", "family_id", "partition",
    "family_index", "site_rank", "stuck_value", "cell_name", "cell_type",
    "output_port", "output_bit_index", "net_bit_id",
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


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


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


def write_or_verify(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(path.read_bytes() == payload, f"generated-file replay mismatch: {path}")
    else:
        path.write_bytes(payload)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as stream:
        stream.write(canonical_json(value))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def csv_bytes(rows: list[dict[str, Any]], fields: list[str]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def deterministic_npz(arrays: dict[str, np.ndarray]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as archive:
        for name in sorted(arrays):
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, np.ascontiguousarray(arrays[name]), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o600 << 16
            archive.writestr(info, buffer.getvalue())
    return output.getvalue()


def run_logged(command: list[str], log: Path, timeout: int, resource_log: Path | None = None) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    actual = command
    if resource_log is not None:
        resource_log.parent.mkdir(parents=True, exist_ok=True)
        actual = ["/usr/bin/time", "-v", "-o", str(resource_log), *command]
    with log.open("wb") as stream:
        try:
            result = subprocess.run(actual, stdout=stream, stderr=subprocess.STDOUT,
                                    cwd=ROOT, check=False, timeout=timeout)
        except subprocess.TimeoutExpired:
            stream.write(f"\nTIMEOUT after {timeout} seconds\n".encode())
            return 124
    return int(result.returncode)


def verify_manifest_outputs(manifest: dict[str, Any], label: str) -> None:
    outputs = manifest.get("outputs")
    require(isinstance(outputs, dict) and outputs, f"{label} manifest outputs")
    for item in outputs.values():
        require(isinstance(item, dict), f"{label} manifest item")
        path = ROOT / str(item.get("path", ""))
        require(path.is_file(), f"missing {label} output: {item.get('path')}")
        require(sha256(path) == item.get("sha256"), f"{label} output SHA: {item.get('path')}")
        require(path.stat().st_size == int(item.get("bytes", -1)), f"{label} output size: {item.get('path')}")


def load_vectors() -> dict[str, np.ndarray]:
    with np.load(VECTORS_1G, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    required = {
        "family_ids", "family_index", "partition_index", "transaction_id",
        "family_vector_index", "pilot_mask", "stimulus_class", "payload",
        "payload_valid_byte_mask", "control", "request_sha256",
    }
    require(set(arrays) == required, "12C-1G vector members")
    require(arrays["payload"].shape == (240, 256), "vector payload shape")
    require(arrays["control"].shape == (240, 8), "vector control shape")
    require(int(arrays["pilot_mask"].sum()) == TOTAL_PILOT_VECTORS, "pilot-vector count")
    require(arrays["family_ids"].tolist() == [item.encode() for item in AUTHORIZED], "family order")
    return arrays


# Minimal, dependency-free AES oracle.  State bytes use the FIPS-197 column-major order.
AES_SBOX = (
    0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
    0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
    0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
    0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
    0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
    0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
    0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
    0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
    0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
    0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
    0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
    0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
    0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
    0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
    0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
    0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16,
)
AES_INV_SBOX = tuple(AES_SBOX.index(value) for value in range(256))
AES_RCON = (0x00,0x01,0x02,0x04,0x08,0x10,0x20,0x40,0x80,0x1b,0x36)


def aes_mul(a: int, b: int) -> int:
    result = 0
    for _ in range(8):
        if b & 1:
            result ^= a
        a = ((a << 1) ^ (0x11B if a & 0x80 else 0)) & 0xFF
        b >>= 1
    return result


def aes_expand_key(key: bytes) -> list[bytes]:
    nk = len(key) // 4
    require(nk in (4, 8), "AES oracle key length")
    nr = nk + 6
    words = [list(key[index:index + 4]) for index in range(0, len(key), 4)]
    for index in range(nk, 4 * (nr + 1)):
        temp = words[index - 1].copy()
        if index % nk == 0:
            temp = temp[1:] + temp[:1]
            temp = [AES_SBOX[value] for value in temp]
            temp[0] ^= AES_RCON[index // nk]
        elif nk > 6 and index % nk == 4:
            temp = [AES_SBOX[value] for value in temp]
        words.append([words[index - nk][slot] ^ temp[slot] for slot in range(4)])
    return [bytes(sum(words[4 * rnd:4 * rnd + 4], [])) for rnd in range(nr + 1)]


def aes_shift_rows(state: list[int], inverse: bool = False) -> list[int]:
    output = state.copy()
    for row in range(4):
        values = [state[row + 4 * col] for col in range(4)]
        shift = (-row if inverse else row) % 4
        values = values[shift:] + values[:shift]
        for col in range(4):
            output[row + 4 * col] = values[col]
    return output


def aes_mix_columns(state: list[int], inverse: bool = False) -> list[int]:
    matrix = ((14,11,13,9),(9,14,11,13),(13,9,14,11),(11,13,9,14)) if inverse else ((2,3,1,1),(1,2,3,1),(1,1,2,3),(3,1,1,2))
    output = [0] * 16
    for col in range(4):
        source = state[4 * col:4 * col + 4]
        for row in range(4):
            value = 0
            for slot in range(4):
                value ^= aes_mul(matrix[row][slot], source[slot])
            output[4 * col + row] = value
    return output


def aes_block(key: bytes, block: bytes, decrypt: bool) -> bytes:
    keys = aes_expand_key(key)
    state = list(block)
    add = lambda values, round_key: [a ^ b for a, b in zip(values, round_key)]
    if not decrypt:
        state = add(state, keys[0])
        for round_index in range(1, len(keys) - 1):
            state = [AES_SBOX[value] for value in state]
            state = aes_shift_rows(state)
            state = aes_mix_columns(state)
            state = add(state, keys[round_index])
        state = [AES_SBOX[value] for value in state]
        state = aes_shift_rows(state)
        return bytes(add(state, keys[-1]))
    state = add(state, keys[-1])
    for round_index in range(len(keys) - 2, 0, -1):
        state = aes_shift_rows(state, True)
        state = [AES_INV_SBOX[value] for value in state]
        state = add(state, keys[round_index])
        state = aes_mix_columns(state, True)
    state = aes_shift_rows(state, True)
    state = [AES_INV_SBOX[value] for value in state]
    return bytes(add(state, keys[0]))


SHA256_K = (
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2,
)
SHA256_IV = (0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19)


def ror(value: int, count: int) -> int:
    return ((value >> count) | (value << (32 - count))) & 0xFFFFFFFF


def sha256_compress(block: bytes) -> bytes:
    require(len(block) == 64, "SHA-256 oracle block length")
    words = [int.from_bytes(block[offset:offset + 4], "big") for offset in range(0, 64, 4)]
    for index in range(16, 64):
        s0 = ror(words[index - 15], 7) ^ ror(words[index - 15], 18) ^ (words[index - 15] >> 3)
        s1 = ror(words[index - 2], 17) ^ ror(words[index - 2], 19) ^ (words[index - 2] >> 10)
        words.append((words[index - 16] + s0 + words[index - 7] + s1) & 0xFFFFFFFF)
    a, b, c, d, e, f, g, h_value = SHA256_IV
    for index in range(64):
        s1 = ror(e, 6) ^ ror(e, 11) ^ ror(e, 25)
        choice = (e & f) ^ ((~e) & g)
        temp1 = (h_value + s1 + choice + SHA256_K[index] + words[index]) & 0xFFFFFFFF
        s0 = ror(a, 2) ^ ror(a, 13) ^ ror(a, 22)
        majority = (a & b) ^ (a & c) ^ (b & c)
        temp2 = (s0 + majority) & 0xFFFFFFFF
        h_value, g, f, e, d, c, b, a = g, f, e, (d + temp1) & 0xFFFFFFFF, c, b, a, (temp1 + temp2) & 0xFFFFFFFF
    values = [(x + y) & 0xFFFFFFFF for x, y in zip(SHA256_IV, (a,b,c,d,e,f,g,h_value))]
    return b"".join(value.to_bytes(4, "big") for value in values)


def sign_extend(value: int, bits: int) -> int:
    sign = 1 << (bits - 1)
    return (value & (sign - 1)) - (value & sign)


def cpu_oracle(program: bytes, instruction_limit: int) -> bytes:
    memory = bytearray(4096)
    memory[:len(program)] = program
    registers = [0] * 32
    pc = 0
    stores = 0
    last_address = 0
    last_data = 0
    signature = 0
    trap = False
    for _ in range(min(instruction_limit, 8192)):
        if pc + 4 > len(memory):
            trap = True
            break
        instruction = int.from_bytes(memory[pc:pc + 4], "little")
        opcode = instruction & 0x7F
        rd = (instruction >> 7) & 31
        funct3 = (instruction >> 12) & 7
        rs1 = (instruction >> 15) & 31
        rs2 = (instruction >> 20) & 31
        next_pc = (pc + 4) & 0xFFFFFFFF
        if instruction == 0x00100073:
            trap = True
            break
        if opcode == 0x13 and funct3 == 0:  # ADDI
            registers[rd] = (registers[rs1] + sign_extend(instruction >> 20, 12)) & 0xFFFFFFFF
        elif opcode == 0x33:
            if funct3 == 0:
                registers[rd] = (registers[rs1] + registers[rs2]) & 0xFFFFFFFF
            elif funct3 == 4:
                registers[rd] = registers[rs1] ^ registers[rs2]
        elif opcode == 0x63 and funct3 == 0:  # BEQ
            immediate = (((instruction >> 31) & 1) << 12) | (((instruction >> 7) & 1) << 11) | (((instruction >> 25) & 0x3F) << 5) | (((instruction >> 8) & 0xF) << 1)
            if registers[rs1] == registers[rs2]:
                next_pc = (pc + sign_extend(immediate, 13)) & 0xFFFFFFFF
        elif opcode == 0x03 and funct3 == 2:  # LW
            address = (registers[rs1] + sign_extend(instruction >> 20, 12)) & 0xFFFFFFFF
            registers[rd] = int.from_bytes(memory[address:address + 4], "little") if address + 4 <= len(memory) else 0
        elif opcode == 0x23 and funct3 == 2:  # SW
            immediate = ((instruction >> 25) << 5) | ((instruction >> 7) & 0x1F)
            address = (registers[rs1] + sign_extend(immediate, 12)) & 0xFFFFFFFF
            data = registers[rs2]
            if address + 4 <= len(memory):
                memory[address:address + 4] = data.to_bytes(4, "little")
            stores += 1
            last_address, last_data = address, data
            signature ^= ((address * 0x9E3779B1) ^ data) & 0xFFFFFFFF
        elif opcode == 0x73:  # Read-only counter CSR; value is intentionally not part of the signature.
            registers[rd] = 0
        else:
            trap = True
            break
        registers[0] = 0
        pc = next_pc
    response = bytearray(32)
    response[0:4] = (1 if trap else 0).to_bytes(4, "big")
    response[4:8] = stores.to_bytes(4, "big")
    response[8:12] = last_address.to_bytes(4, "big")
    response[12:16] = last_data.to_bytes(4, "big")
    response[16:20] = signature.to_bytes(4, "big")
    return bytes(response)


def expected_response(family_id: str, payload: bytes, control: np.ndarray) -> bytes:
    if family_id == "opentitan_hmac_sha256":
        return hmac.new(payload[:32], payload[32:32 + int(control[2])], hashlib.sha256).digest()
    if family_id == "secworks_aes":
        key_length = 16 if int(control[1]) == 128 else 32
        result = aes_block(payload[:key_length], payload[32:48], decrypt=bool(int(control[2])))
        return bytes(16) + result
    if family_id == "secworks_sha256":
        return sha256_compress(payload[:64])
    require(family_id == "picorv32_cpu", f"response oracle family: {family_id}")
    return cpu_oracle(payload[:int(control[0])], int(control[2]))


def verify_inputs() -> tuple[dict[str, np.ndarray], dict[str, int]]:
    print("STAGE 12C-1I — ADAPTER FUNCTIONAL-VALIDATION AND BOUNDED PILOT EXECUTION")
    print("FROZEN INPUT VERIFICATION")
    for path, expected in PINNED.items():
        require(path.is_file(), f"missing frozen input: {rel(path)}")
        require(sha256(path) == expected, f"frozen input SHA: {rel(path)}")
        print(f"  {path.name:<112}: OK", flush=True)

    manifest_1h = load_json(MANIFEST_1H)
    audit_1h = load_json(AUDIT_1H)
    authorization = load_json(AUTHORIZATION)
    validation = load_json(VALIDATION_CONTRACT)
    pilot = load_json(PILOT_CONTRACT)
    acceptance = load_json(ACCEPTANCE_CONTRACT)
    manifest_1e = load_json(MANIFEST_1E)
    audit_1e = load_json(AUDIT_1E)
    require(manifest_1h.get("status") == "PASS" and audit_1h.get("status") == "PASS", "12C-1H freeze")
    require(audit_1h.get("authorization_status") == "FROZEN", "12C-1H authorization status")
    require(audit_1h.get("adapter_functional_validation") == "AUTHORIZED / NOT STARTED", "adapter validation authorization")
    require(audit_1h.get("bounded_pilot_campaign") == "CONDITIONALLY AUTHORIZED / NOT STARTED", "pilot authorization")
    require(audit_1h.get("pilot_fault_budget") == TOTAL_FAULTS, "pilot fault budget")
    require(audit_1h.get("maximum_pilot_transactions") == MAX_PILOT_TRANSACTIONS, "pilot transaction budget")
    require(audit_1h.get("independent_test_validation_holdout_access") == [0, 0, 0], "protected access")
    require(authorization.get("status") == "FROZEN", "authorization object")
    require(authorization.get("bounded_pilot_campaign", "").startswith("CONDITIONALLY AUTHORIZED"), "conditional pilot authorization")
    require(tuple(authorization.get("authorized_families", [])) == AUTHORIZED, "authorized family order")
    require(validation.get("status") == "FROZEN" and len(validation.get("families", [])) == 4, "validation contract")
    require(pilot.get("status") == "FROZEN" and pilot.get("fault_model") == "SINGLE PERSISTENT CELL-OUTPUT SA0/SA1", "pilot contract")
    require(acceptance.get("status") == "FROZEN", "acceptance contract")
    require(acceptance.get("adapter_hard_gates", {}).get("families_passed") == "4 OF 4", "adapter hard gate")
    require(manifest_1e.get("status") == "PASS" and audit_1e.get("status") == "PASS", "12C-1E freeze")
    require(audit_1e.get("deterministic_replay") == "PASS / BYTE-EXACT / 4 OF 4", "12C-1E replay")
    verify_manifest_outputs(manifest_1h, "12C-1H")
    verify_manifest_outputs(manifest_1e, "12C-1E")

    plan = read_csv(EXECUTION_PLAN)
    budgets = read_csv(CAMPAIGN_BUDGET_1F)
    require([row["family_id"] for row in plan] == list(AUTHORIZED), "execution plan family order")
    require([row["family_id"] for row in budgets] == list(AUTHORIZED), "campaign budget family order")
    budget_map: dict[str, int] = {}
    for family_id, row in zip(AUTHORIZED, budgets):
        require(int(row["pilot_vector_budget"]) == PILOT_VECTORS[family_id], f"pilot vectors: {family_id}")
        require(int(row["pilot_fault_budget"]) == FAULTS_PER_FAMILY, f"pilot faults: {family_id}")
        require(int(row["maximum_pilot_transactions"]) == FAULTS_PER_FAMILY * PILOT_VECTORS[family_id], f"pilot transactions: {family_id}")
        budget_map[family_id] = int(row["pilot_fault_budget"])

    arrays = load_vectors()
    manifest_outputs = manifest_1e["outputs"]
    for family_id in AUTHORIZED:
        family_dir = WORK_1E / "families" / family_id
        result_path = family_dir / "family_result_12c1e.json"
        json_path = family_dir / f"{family_id}_generic_12c1e.json"
        verilog_path = family_dir / f"{family_id}_generic_12c1e.v"
        for path in (result_path, json_path, verilog_path):
            item = manifest_outputs.get(rel(path))
            require(isinstance(item, dict), f"12C-1E manifest record: {family_id}/{path.name}")
            require(path.is_file() and sha256(path) == item.get("sha256"), f"frozen netlist artifact: {family_id}/{path.name}")
        result = load_json(result_path)
        require(result.get("status") == "PASS" and result.get("top_module") == TOPS[family_id], f"family result: {family_id}")
        document = load_json(json_path)
        require(set(document.get("modules", {})) == {TOPS[family_id]}, f"flattened top module: {family_id}")

    require(shutil.which("yosys") is not None, "Yosys not found")
    require(shutil.which("verilator") is not None, "Verilator not found")
    require(Path("/usr/bin/time").is_file(), "/usr/bin/time not found")
    free_gib = shutil.disk_usage(ROOT).free / 1024**3
    require(free_gib >= MIN_FREE_GIB, f"insufficient disk: {free_gib:.2f} GiB; {MIN_FREE_GIB} GiB required")
    print("  Authorization, four frozen netlists, vectors, budgets and protected partitions                 : PASS", flush=True)
    return arrays, budget_map


def integer_bits(value: Any) -> list[int]:
    found: list[int] = []
    if isinstance(value, int):
        found.append(value)
    elif isinstance(value, list):
        for item in value:
            found.extend(integer_bits(item))
    elif isinstance(value, dict):
        for item in value.values():
            found.extend(integer_bits(item))
    return found


def maximum_net_bit(module: dict[str, Any]) -> int:
    values: list[int] = []
    for port in module.get("ports", {}).values():
        values.extend(bit for bit in port.get("bits", []) if isinstance(bit, int))
    for cell in module.get("cells", {}).values():
        for connection in cell.get("connections", {}).values():
            values.extend(bit for bit in connection if isinstance(bit, int))
    for netname in module.get("netnames", {}).values():
        values.extend(bit for bit in netname.get("bits", []) if isinstance(bit, int))
    return max(values, default=1)


def enumerate_sites(family_id: str, source_json: Path) -> list[dict[str, Any]]:
    design = load_json(source_json)
    module = design["modules"][TOPS[family_id]]
    sites: list[dict[str, Any]] = []
    used_bits: set[int] = set()
    for cell_name in sorted(module.get("cells", {})):
        cell = module["cells"][cell_name]
        directions = cell.get("port_directions", {})
        for port_name in sorted(cell.get("connections", {})):
            if directions.get(port_name) != "output":
                continue
            for bit_index, bit in enumerate(cell["connections"][port_name]):
                if not isinstance(bit, int) or bit < 2 or bit in used_bits:
                    continue
                used_bits.add(bit)
                site_rank = len(sites)
                identity = f"{family_id}\0{cell_name}\0{port_name}\0{bit_index}\0{bit}".encode()
                sites.append({
                    "site_rank": site_rank,
                    "opaque_site_id": f"S{AUTHORIZED.index(family_id)}-{site_rank:04d}-{hashlib.sha256(identity).hexdigest()[:12]}",
                    "cell_name": cell_name,
                    "cell_type": str(cell.get("type", "")),
                    "output_port": port_name,
                    "output_bit_index": bit_index,
                    "net_bit_id": bit,
                })
                if len(sites) == SITES_PER_FAMILY:
                    return sites
    stop(f"{family_id} has only {len(sites)} eligible driven bits; {SITES_PER_FAMILY} required")


def instrument_netlist(family_id: str, source_json: Path, sites: list[dict[str, Any]]) -> bytes:
    design = copy.deepcopy(load_json(source_json))
    module = design["modules"][TOPS[family_id]]
    require(all(name not in module.get("ports", {}) for name in ("fi_enable_i", "fi_site_onehot_i", "fi_stuck_value_i")), f"fault-port collision: {family_id}")
    maximum = maximum_net_bit(module)
    enable_bit = maximum + 1
    stuck_bit = maximum + 2
    onehot_bits = list(range(maximum + 3, maximum + 3 + len(sites)))
    next_bit = maximum + 3 + len(sites)
    module.setdefault("ports", {})["fi_enable_i"] = {"direction": "input", "bits": [enable_bit]}
    module["ports"]["fi_stuck_value_i"] = {"direction": "input", "bits": [stuck_bit]}
    module["ports"]["fi_site_onehot_i"] = {"direction": "input", "bits": onehot_bits}
    netnames = module.setdefault("netnames", {})
    netnames["fi_enable_i"] = {"hide_name": 0, "bits": [enable_bit], "attributes": {}}
    netnames["fi_stuck_value_i"] = {"hide_name": 0, "bits": [stuck_bit], "attributes": {}}
    netnames["fi_site_onehot_i"] = {"hide_name": 0, "bits": onehot_bits, "attributes": {}}
    cells = module.setdefault("cells", {})
    for site in sites:
        rank = int(site["site_rank"])
        cell = cells[site["cell_name"]]
        connection = cell["connections"][site["output_port"]]
        old_bit = connection[int(site["output_bit_index"])]
        require(old_bit == int(site["net_bit_id"]), f"site replay: {family_id}/{rank}")
        raw_bit, gate_bit = next_bit, next_bit + 1
        next_bit += 2
        connection[int(site["output_bit_index"])] = raw_bit
        cells[f"$v22fi_and${rank}"] = {
            "hide_name": 1, "type": "$_AND_", "parameters": {}, "attributes": {},
            "port_directions": {"A": "input", "B": "input", "Y": "output"},
            "connections": {"A": [enable_bit], "B": [onehot_bits[rank]], "Y": [gate_bit]},
        }
        cells[f"$v22fi_mux${rank}"] = {
            "hide_name": 1, "type": "$_MUX_", "parameters": {}, "attributes": {},
            "port_directions": {"A": "input", "B": "input", "S": "input", "Y": "output"},
            "connections": {"A": [raw_bit], "B": [stuck_bit], "S": [gate_bit], "Y": [old_bit]},
        }
    return canonical_json(design)


def family_vector_indices(arrays: dict[str, np.ndarray], family_id: str) -> tuple[np.ndarray, np.ndarray]:
    family_number = AUTHORIZED.index(family_id)
    all_indices = np.flatnonzero(arrays["family_index"] == family_number)
    pilot = all_indices[arrays["pilot_mask"][all_indices].astype(bool)]
    canaries = all_indices[arrays["control"][all_indices, 7] == 1]
    require(pilot.size == PILOT_VECTORS[family_id], f"pilot subset: {family_id}")
    require(canaries.size == 1 and canaries[0] not in set(pilot.tolist()), f"timeout canary: {family_id}")
    return pilot, np.concatenate((pilot, canaries))


def vector_material(arrays: dict[str, np.ndarray], family_id: str) -> dict[str, Any]:
    pilot, validation = family_vector_indices(arrays, family_id)
    payloads = arrays["payload"][validation].astype(np.uint8, copy=False)
    controls = arrays["control"][validation].astype(np.uint32, copy=False)
    expected = []
    for slot in range(len(validation)):
        if int(controls[slot, 7]) == 1:
            expected.append(bytes(32))
        else:
            expected.append(expected_response(family_id, payloads[slot].tobytes(), controls[slot]))
    return {
        "pilot_indices": pilot.astype(np.int32),
        "validation_indices": validation.astype(np.int32),
        "payload": payloads,
        "control": controls,
        "transaction_id": arrays["transaction_id"][validation].astype(np.int32),
        "expected": expected,
    }


def memory_payloads(family_id: str, material: dict[str, Any], directory: Path) -> dict[Path, bytes]:
    payloads: np.ndarray = material["payload"]
    controls: np.ndarray = material["control"]
    outputs: dict[Path, bytes] = {
        directory / "payload.mem": "".join(row.tobytes().hex() + "\n" for row in payloads).encode(),
        directory / "expected.mem": "".join(value.hex() + "\n" for value in material["expected"]).encode(),
        directory / "transaction.mem": "".join(f"{int(value):08x}\n" for value in material["transaction_id"]).encode(),
    }
    for index in range(8):
        outputs[directory / f"control{index}.mem"] = "".join(f"{int(value):08x}\n" for value in controls[:, index]).encode()
    if family_id in ("opentitan_hmac_sha256", "secworks_aes"):
        outputs[directory / "key.mem"] = "".join(row[:32].tobytes().hex() + "\n" for row in payloads).encode()
    if family_id == "opentitan_hmac_sha256":
        outputs[directory / "message.mem"] = "".join((row[32:].tobytes() + bytes(32)).hex() + "\n" for row in payloads).encode()
    elif family_id == "secworks_aes":
        outputs[directory / "block.mem"] = "".join(row[32:48].tobytes().hex() + "\n" for row in payloads).encode()
    elif family_id == "secworks_sha256":
        outputs[directory / "block.mem"] = "".join(row[:64].tobytes().hex() + "\n" for row in payloads).encode()
    elif family_id == "picorv32_cpu":
        outputs[directory / "program.mem"] = "".join(row.tobytes().hex() + "\n" for row in payloads).encode()
    return outputs


def sv_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "\\\\").replace('"', '\\"')


def transform_hmac_bridge(source: str) -> str:
    require(source.count("module opentitan_hmac_sha256_bridge") == 1, "HMAC bridge module declaration")
    require(source.count("hmac_core u_hmac_core (") == 1, "HMAC bridge core instance")
    source = source.replace("module opentitan_hmac_sha256_bridge", "module v22_hmac_bridge", 1)
    marker = "    input  logic rst_ni,\n"
    require(source.count(marker) == 1, "HMAC bridge reset port")
    source = source.replace(marker, marker + f"\n    input logic fi_enable_i,\n    input logic [{SITES_PER_FAMILY - 1}:0] fi_site_onehot_i,\n    input logic fi_stuck_value_i,\n", 1)
    instance = "        .rst_ni                   (rst_ni),\n"
    # R1: both HMAC and SHA have this reset connection. Modify only HMAC.
    core_start = source.index("hmac_core u_hmac_core (")
    core_end = source.find(");", core_start)
    require(core_end >= 0, "HMAC bridge core instance terminator")
    core = source[core_start:core_end]
    require(core.count(instance) == 1, "HMAC bridge reset connection")
    core = core.replace(instance, instance + "        .fi_enable_i              (fi_enable_i),\n        .fi_site_onehot_i          (fi_site_onehot_i),\n        .fi_stuck_value_i          (fi_stuck_value_i),\n", 1)
    source = source[:core_start] + core + source[core_end:]
    return source


def hmac_adapter_sv() -> str:
    return f'''module v22_hmac_adapter(
  input logic clk_i,input logic rst_ni,input logic start_i,
  input logic [255:0] key_i,input logic [2047:0] message_i,input logic [15:0] message_length_i,
  input logic fi_enable_i,input logic [{SITES_PER_FAMILY - 1}:0] fi_site_onehot_i,input logic fi_stuck_value_i,
  output logic busy_o,output logic done_o,output logic [255:0] digest_o);
  localparam integer FIFO_DEPTH=64;
  logic [255:0] key_q; logic [2047:0] message_q; logic [15:0] message_length_q;
  logic [6:0] message_word_count,msg_word_index,msg_words_consumed;
  logic [1023:0] secret_key; assign secret_key={{key_q,768'b0}};
  logic bridge_hash_done,bridge_hmac_idle,bridge_sha_idle,bridge_hash_running,bridge_digest_on_blk;
  logic [255:0] bridge_digest; logic bridge_start,bridge_process;
  logic bridge_fifo_rvalid; logic [31:0] bridge_fifo_rdata; logic [3:0] bridge_fifo_rmask; logic bridge_fifo_rready;
  logic hmac_fifo_wsel,hmac_fifo_wvalid; logic [3:0] hmac_fifo_wdata_sel; logic hmac_fifo_wready;
  logic [31:0] fifo_data_mem[0:FIFO_DEPTH-1]; logic [3:0] fifo_mask_mem[0:FIFO_DEPTH-1];
  logic [5:0] fifo_wptr,fifo_rptr; logic [6:0] fifo_count; logic fifo_empty,fifo_full;
  assign fifo_empty=(fifo_count==0); assign fifo_full=(fifo_count==FIFO_DEPTH);
  assign bridge_fifo_rvalid=!fifo_empty; assign bridge_fifo_rdata=fifo_data_mem[fifo_rptr];
  assign bridge_fifo_rmask=fifo_mask_mem[fifo_rptr]; assign hmac_fifo_wready=!fifo_full;
  typedef enum logic [2:0] {{ST_IDLE,ST_START,ST_WRITE,ST_DRAIN,ST_PROCESS,ST_WAIT}} state_e;
  state_e state_q; logic user_fifo_push,user_message_pop,hmac_fifo_push,fifo_push,fifo_pop;
  logic [31:0] user_fifo_data,hmac_fifo_data,selected_fifo_data;
  assign user_fifo_data=message_q[2047-(msg_word_index*32)-:32];
  always_comb begin
    case(hmac_fifo_wdata_sel[2:0])
      3'd0:hmac_fifo_data=bridge_digest[255:224]; 3'd1:hmac_fifo_data=bridge_digest[223:192];
      3'd2:hmac_fifo_data=bridge_digest[191:160]; 3'd3:hmac_fifo_data=bridge_digest[159:128];
      3'd4:hmac_fifo_data=bridge_digest[127:96]; 3'd5:hmac_fifo_data=bridge_digest[95:64];
      3'd6:hmac_fifo_data=bridge_digest[63:32]; default:hmac_fifo_data=bridge_digest[31:0];
    endcase
  end
  assign hmac_fifo_push=hmac_fifo_wsel&&hmac_fifo_wvalid&&!fifo_full;
  assign user_fifo_push=(state_q==ST_WRITE)&&!hmac_fifo_wsel&&!fifo_full;
  assign fifo_push=hmac_fifo_push||user_fifo_push;
  assign selected_fifo_data=hmac_fifo_push?hmac_fifo_data:user_fifo_data;
  assign fifo_pop=bridge_fifo_rvalid&&bridge_fifo_rready;
  assign user_message_pop=fifo_pop&&((state_q==ST_WRITE)||(state_q==ST_DRAIN));
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if(!rst_ni) begin fifo_wptr<='0;fifo_rptr<='0;fifo_count<='0; end else begin
      if((state_q==ST_IDLE)&&start_i) begin fifo_wptr<='0;fifo_rptr<='0;fifo_count<='0; end else begin
        if(fifo_push) begin fifo_data_mem[fifo_wptr]<=selected_fifo_data;fifo_mask_mem[fifo_wptr]<=4'hf;fifo_wptr<=fifo_wptr+1'b1;end
        if(fifo_pop) fifo_rptr<=fifo_rptr+1'b1;
        case({{fifo_push,fifo_pop}}) 2'b10:fifo_count<=fifo_count+1'b1;2'b01:fifo_count<=fifo_count-1'b1;default:fifo_count<=fifo_count;endcase
      end
    end
  end
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if(!rst_ni) begin state_q<=ST_IDLE;key_q<='0;message_q<='0;message_length_q<='0;message_word_count<='0;
      msg_word_index<='0;msg_words_consumed<='0;bridge_start<=0;bridge_process<=0;busy_o<=0;done_o<=0;digest_o<='0;end
    else begin bridge_start<=0;bridge_process<=0;done_o<=0;
      if(user_message_pop&&(msg_words_consumed<message_word_count)) msg_words_consumed<=msg_words_consumed+1'b1;
      case(state_q)
        ST_IDLE:begin busy_o<=0;if(start_i) begin key_q<=key_i;message_q<=message_i;message_length_q<=message_length_i;
          message_word_count<=message_length_i>>2;msg_word_index<=0;msg_words_consumed<=0;busy_o<=1;state_q<=ST_START;end end
        ST_START:begin bridge_start<=1;state_q<=ST_WRITE;end
        ST_WRITE:if(user_fifo_push) begin if(msg_word_index+1>=message_word_count) begin msg_word_index<=0;state_q<=ST_DRAIN;end else msg_word_index<=msg_word_index+1'b1;end
        ST_DRAIN:if((msg_words_consumed==message_word_count)&&fifo_empty) state_q<=ST_PROCESS;
        ST_PROCESS:begin bridge_process<=1;state_q<=ST_WAIT;end
        ST_WAIT:if(bridge_hash_done) begin digest_o<=bridge_digest;busy_o<=0;done_o<=1;state_q<=ST_IDLE;end
        default:state_q<=ST_IDLE;
      endcase
    end
  end
  v22_hmac_bridge u_bridge(.clk_i(clk_i),.rst_ni(rst_ni),.fi_enable_i(fi_enable_i),
    .fi_site_onehot_i(fi_site_onehot_i),.fi_stuck_value_i(fi_stuck_value_i),.secret_key_i(secret_key),
    .reg_hash_start_i(bridge_start),.reg_hash_stop_i(1'b0),.reg_hash_continue_i(1'b0),.reg_hash_process_i(bridge_process),
    .fifo_rvalid_i(bridge_fifo_rvalid),.fifo_rdata_i(bridge_fifo_rdata),.fifo_rmask_i(bridge_fifo_rmask),
    .fifo_rready_o(bridge_fifo_rready),.fifo_wsel_o(hmac_fifo_wsel),.fifo_wvalid_o(hmac_fifo_wvalid),
    .fifo_wdata_sel_o(hmac_fifo_wdata_sel),.fifo_wready_i(hmac_fifo_wready),
    .message_length_i({{45'b0,message_length_q,3'b0}}),.hash_done_o(bridge_hash_done),
    .hmac_idle_o(bridge_hmac_idle),.sha_idle_o(bridge_sha_idle),.hash_running_o(bridge_hash_running),
    .digest_on_blk_o(bridge_digest_on_blk),.digest_o(bridge_digest));
endmodule
'''


def common_testbench(family_id: str, declarations: str, setup: str, transaction_task: str) -> str:
    top = f"tb_v22_{family_id}_12c1i"
    count = VALIDATION_VECTORS[family_id]
    pilot = PILOT_VECTORS[family_id]
    memory_dir = MEMORY / family_id
    read_common = "\n".join([
        f'    $readmemh("{sv_path(memory_dir / "expected.mem")}",expected_vectors);',
        f'    $readmemh("{sv_path(memory_dir / "transaction.mem")}",transaction_ids);',
        *[f'    $readmemh("{sv_path(memory_dir / f"control{index}.mem")}",control{index});' for index in range(8)],
    ])
    return f'''`timescale 1ns/1ps
module {top};
  localparam integer VALIDATION_COUNT={count},PILOT_COUNT={pilot},SITE_COUNT={SITES_PER_FAMILY};
  logic clk=0; always #5 clk=~clk;
  logic fi_enable=0,fi_stuck=0; logic [SITE_COUNT-1:0] fi_onehot='0;
  logic [255:0] expected_vectors[0:VALIDATION_COUNT-1]; logic [31:0] transaction_ids[0:VALIDATION_COUNT-1];
  logic [31:0] control0[0:VALIDATION_COUNT-1],control1[0:VALIDATION_COUNT-1],control2[0:VALIDATION_COUNT-1],control3[0:VALIDATION_COUNT-1];
  logic [31:0] control4[0:VALIDATION_COUNT-1],control5[0:VALIDATION_COUNT-1],control6[0:VALIDATION_COUNT-1],control7[0:VALIDATION_COUNT-1];
  {declarations}
  integer csv_fd,vi,site,stuck,batch_id,site_start,site_count,failures=0;
  integer cycles_result; logic timeout_result,protocol_result,unknown_result; logic [255:0] response_result;
  string mode,csv_path;
  {transaction_task}
  task automatic emit(input string run_type,input integer site_value,input integer stuck_value); begin
    $fwrite(csv_fd,"{family_id},%s,%0d,%s,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%064h,%064h\\n",
      mode,batch_id,run_type,site_value,stuck_value,vi,transaction_ids[vi],cycles_result,timeout_result,
      protocol_result,unknown_result,expected_vectors[vi],response_result);
  end endtask
  initial begin
{read_common}
{setup}
    if(!$value$plusargs("MODE=%s",mode)) $fatal(1,"MODE plusarg required");
    if(!$value$plusargs("CSV=%s",csv_path)) $fatal(1,"CSV plusarg required");
    batch_id=-1;site_start=0;site_count=0;
    void'($value$plusargs("BATCH_ID=%d",batch_id));void'($value$plusargs("SITE_START=%d",site_start));void'($value$plusargs("SITE_COUNT=%d",site_count));
    csv_fd=$fopen(csv_path,"w");if(csv_fd==0)$fatal(1,"CSV open failed");
    $fdisplay(csv_fd,"{','.join(CSV_FIELDS)}");
    if(mode=="VALIDATE") begin
      for(vi=0;vi<VALIDATION_COUNT;vi=vi+1) begin
        run_transaction(vi,0,0,0,cycles_result,timeout_result,protocol_result,unknown_result,response_result);emit("BASELINE",-1,-1);
        if(control7[vi]!=0) begin if(!timeout_result||unknown_result||protocol_result) failures=failures+1;end
        else if(timeout_result||protocol_result||unknown_result||(response_result!==expected_vectors[vi])) failures=failures+1;
      end
      $fclose(csv_fd);$display("V22_ADAPTER_VALIDATION_FAMILY={family_id}");
      if(failures==0)begin $display("V22_ADAPTER_VALIDATION_RESULT=PASS");$finish;end
      else $fatal(1,"V22_ADAPTER_VALIDATION_RESULT=FAIL failures=%0d",failures);
    end else if(mode=="PILOT") begin
      for(vi=0;vi<PILOT_COUNT;vi=vi+1) begin
        run_transaction(vi,0,0,0,cycles_result,timeout_result,protocol_result,unknown_result,response_result);emit("BASELINE",-1,-1);
        if(timeout_result||protocol_result||unknown_result||(response_result!==expected_vectors[vi])) failures=failures+1;
      end
      for(site=site_start;site<site_start+site_count;site=site+1)for(stuck=0;stuck<2;stuck=stuck+1)for(vi=0;vi<PILOT_COUNT;vi=vi+1)begin
        run_transaction(vi,1,site,stuck,cycles_result,timeout_result,protocol_result,unknown_result,response_result);emit("ENABLED",site,stuck);
        if(unknown_result) failures=failures+1;
      end
      $fclose(csv_fd);$display("V22_PILOT_FAMILY={family_id} BATCH=%0d",batch_id);
      if(failures==0)begin $display("V22_PILOT_BATCH_RESULT=PASS");$finish;end
      else $fatal(1,"V22_PILOT_BATCH_RESULT=FAIL failures=%0d",failures);
    end else $fatal(1,"Unknown MODE");
  end
endmodule
'''


def hmac_testbench() -> str:
    memory_dir = MEMORY / "opentitan_hmac_sha256"
    declarations = f'''logic rst_n=0,start=0,busy,done;logic[255:0]key,digest;logic[2047:0]message;
  logic[15:0]message_length;logic[255:0]key_vectors[0:VALIDATION_COUNT-1];
  logic[2047:0]message_vectors[0:VALIDATION_COUNT-1];
  v22_hmac_adapter dut(.clk_i(clk),.rst_ni(rst_n),.start_i(start),.key_i(key),.message_i(message),
    .message_length_i(message_length),.fi_enable_i(fi_enable),.fi_site_onehot_i(fi_onehot),
    .fi_stuck_value_i(fi_stuck),.busy_o(busy),.done_o(done),.digest_o(digest));'''
    setup = f'''    $readmemh("{sv_path(memory_dir / 'key.mem')}",key_vectors);
    $readmemh("{sv_path(memory_dir / 'message.mem')}",message_vectors);'''
    task = '''task automatic run_transaction(input integer vector,input logic inject,input integer selected,input logic forced,
    output integer cycles,output logic timed_out,output logic protocol_error,output logic unknown_seen,output logic[255:0]response);begin
    start=0;fi_enable=0;fi_onehot='0;fi_stuck=forced;key='0;message='0;message_length=0;
    if((control6[vector]==0)||(vector==0))begin rst_n=0;repeat(5)@(posedge clk);@(negedge clk);rst_n=1;repeat(2)@(posedge clk);end
    else begin rst_n=1;repeat(2)@(posedge clk);end
    cycles=0;protocol_error=0;unknown_seen=0;response='0;
    if(control7[vector]!=0)begin repeat(16)@(posedge clk);cycles=16;timed_out=1;end else begin
      key=key_vectors[vector];message=message_vectors[vector];message_length=control2[vector][15:0];
      fi_enable=inject;fi_stuck=forced;if(inject)fi_onehot[selected]=1'b1;
      @(negedge clk);start=1;@(negedge clk);start=0;
      while(!done&&cycles<control3[vector])begin @(posedge clk);#1;cycles=cycles+1;if($isunknown({done,busy,digest}))unknown_seen=1;end
      timed_out=!done;response=digest;if(done&&busy)protocol_error=1;
    end fi_enable=0;fi_onehot='0;@(posedge clk);
  end endtask'''
    return common_testbench("opentitan_hmac_sha256", declarations, setup, task)


def aes_testbench() -> str:
    memory_dir = MEMORY / "secworks_aes"
    declarations = f'''logic reset_n=0,encdec=0,init=0,next_signal=0,ready,keylen=0,result_valid;
  logic[255:0]key,key_vectors[0:VALIDATION_COUNT-1];logic[127:0]block,result,block_vectors[0:VALIDATION_COUNT-1];
  aes_core dut(.clk(clk),.reset_n(reset_n),.encdec(encdec),.init(init),.next(next_signal),.ready(ready),
    .key(key),.keylen(keylen),.block(block),.result(result),.result_valid(result_valid),
    .fi_enable_i(fi_enable),.fi_site_onehot_i(fi_onehot),.fi_stuck_value_i(fi_stuck));'''
    setup = f'''    $readmemh("{sv_path(memory_dir / 'key.mem')}",key_vectors);
    $readmemh("{sv_path(memory_dir / 'block.mem')}",block_vectors);'''
    task = '''task automatic run_transaction(input integer vector,input logic inject,input integer selected,input logic forced,
    output integer cycles,output logic timed_out,output logic protocol_error,output logic unknown_seen,output logic[255:0]response);begin
    init=0;next_signal=0;fi_enable=0;fi_onehot='0;fi_stuck=forced;cycles=0;protocol_error=0;unknown_seen=0;response='0;
    if((control6[vector]==0)||(vector==0))begin reset_n=0;repeat(5)@(posedge clk);@(negedge clk);reset_n=1;repeat(2)@(posedge clk);end
    else begin reset_n=1;repeat(2)@(posedge clk);end
    if(control7[vector]!=0)begin repeat(16)@(posedge clk);cycles=16;timed_out=1;end else begin
      key=key_vectors[vector];block=block_vectors[vector];keylen=(control1[vector]==256);encdec=(control2[vector]==0);
      fi_enable=inject;fi_stuck=forced;if(inject)fi_onehot[selected]=1'b1;
      @(negedge clk);init=1;@(negedge clk);init=0;
      while(!ready&&cycles<control3[vector])begin @(posedge clk);#1;cycles=cycles+1;if($isunknown({ready,result_valid,result}))unknown_seen=1;end
      if(ready)begin @(negedge clk);next_signal=1;@(negedge clk);next_signal=0;end
      while(!result_valid&&cycles<control3[vector])begin @(posedge clk);#1;cycles=cycles+1;if($isunknown({ready,result_valid,result}))unknown_seen=1;end
      timed_out=!result_valid;response={128'b0,result};if(result_valid&&!ready)protocol_error=1;
    end fi_enable=0;fi_onehot='0;@(posedge clk);
  end endtask'''
    return common_testbench("secworks_aes", declarations, setup, task)


def sha_testbench() -> str:
    memory_dir = MEMORY / "secworks_sha256"
    declarations = f'''logic reset_n=0,init=0,next_signal=0,mode_sha=1,ready,digest_valid;
  logic[511:0]block,block_vectors[0:VALIDATION_COUNT-1];logic[255:0]digest;
  sha256_core dut(.clk(clk),.reset_n(reset_n),.init(init),.next(next_signal),.mode(mode_sha),.block(block),
    .ready(ready),.digest(digest),.digest_valid(digest_valid),.fi_enable_i(fi_enable),
    .fi_site_onehot_i(fi_onehot),.fi_stuck_value_i(fi_stuck));'''
    setup = f'''    $readmemh("{sv_path(memory_dir / 'block.mem')}",block_vectors);'''
    task = '''task automatic run_transaction(input integer vector,input logic inject,input integer selected,input logic forced,
    output integer cycles,output logic timed_out,output logic protocol_error,output logic unknown_seen,output logic[255:0]response);begin
    init=0;next_signal=0;mode_sha=1;fi_enable=0;fi_onehot='0;fi_stuck=forced;cycles=0;protocol_error=0;unknown_seen=0;response='0;
    if((control6[vector]==0)||(vector==0))begin reset_n=0;repeat(5)@(posedge clk);@(negedge clk);reset_n=1;repeat(2)@(posedge clk);end
    else begin reset_n=1;repeat(2)@(posedge clk);end
    if(control7[vector]!=0)begin repeat(16)@(posedge clk);cycles=16;timed_out=1;end else begin
      block=block_vectors[vector];fi_enable=inject;fi_stuck=forced;if(inject)fi_onehot[selected]=1'b1;
      if(!ready)protocol_error=1;@(negedge clk);init=1;@(negedge clk);init=0;
      while(!digest_valid&&cycles<control3[vector])begin @(posedge clk);#1;cycles=cycles+1;if($isunknown({ready,digest_valid,digest}))unknown_seen=1;end
      timed_out=!digest_valid;response=digest;
    end fi_enable=0;fi_onehot='0;@(posedge clk);
  end endtask'''
    return common_testbench("secworks_sha256", declarations, setup, task)


def cpu_testbench() -> str:
    memory_dir = MEMORY / "picorv32_cpu"
    declarations = f'''logic resetn=0,trap,mem_valid,mem_instr,mem_ready;logic[31:0]mem_addr,mem_wdata,mem_rdata;
  logic[3:0]mem_wstrb;logic mem_la_read,mem_la_write;logic[31:0]mem_la_addr,mem_la_wdata;logic[3:0]mem_la_wstrb;
  logic pcpi_valid;logic[31:0]pcpi_insn,pcpi_rs1,pcpi_rs2;logic[31:0]irq=0,eoi;logic trace_valid;logic[35:0]trace_data;
  logic[2047:0]program_vectors[0:VALIDATION_COUNT-1];logic[31:0]program_memory[0:63];integer active_word;
  always_comb begin mem_ready=mem_valid;active_word=mem_addr[31:2];if(active_word>=0&&active_word<64)mem_rdata=program_memory[active_word];else mem_rdata=0;end
  picorv32 dut(.clk(clk),.resetn(resetn),.trap(trap),.mem_valid(mem_valid),.mem_instr(mem_instr),.mem_ready(mem_ready),
    .mem_addr(mem_addr),.mem_wdata(mem_wdata),.mem_wstrb(mem_wstrb),.mem_rdata(mem_rdata),
    .mem_la_read(mem_la_read),.mem_la_write(mem_la_write),.mem_la_addr(mem_la_addr),.mem_la_wdata(mem_la_wdata),.mem_la_wstrb(mem_la_wstrb),
    .pcpi_valid(pcpi_valid),.pcpi_insn(pcpi_insn),.pcpi_rs1(pcpi_rs1),.pcpi_rs2(pcpi_rs2),
    .pcpi_wr(1'b0),.pcpi_rd(32'b0),.pcpi_wait(1'b0),.pcpi_ready(1'b0),.irq(irq),.eoi(eoi),
    .trace_valid(trace_valid),.trace_data(trace_data),.fi_enable_i(fi_enable),.fi_site_onehot_i(fi_onehot),.fi_stuck_value_i(fi_stuck));'''
    setup = f'''    $readmemh("{sv_path(memory_dir / 'program.mem')}",program_vectors);'''
    task = '''task automatic run_transaction(input integer vector,input logic inject,input integer selected,input logic forced,
    output integer cycles,output logic timed_out,output logic protocol_error,output logic unknown_seen,output logic[255:0]response);
    integer word_index,store_count;logic[31:0]raw_word,last_address,last_data,store_signature;
    logic captured_valid;logic[31:0]captured_address,captured_data;logic[3:0]captured_strb;
    begin resetn=0;fi_enable=0;fi_onehot='0;fi_stuck=forced;irq=0;cycles=0;protocol_error=0;unknown_seen=0;response='0;
      store_count=0;last_address=0;last_data=0;store_signature=0;
      for(word_index=0;word_index<64;word_index=word_index+1)begin raw_word=program_vectors[vector][2047-(word_index*32)-:32];program_memory[word_index]={raw_word[7:0],raw_word[15:8],raw_word[23:16],raw_word[31:24]};end
      repeat(5)@(posedge clk);@(negedge clk);resetn=1;
      if(control7[vector]!=0)begin repeat(16)@(posedge clk);cycles=16;timed_out=1;end else begin
        fi_enable=inject;fi_stuck=forced;if(inject)fi_onehot[selected]=1'b1;
        while(!trap&&cycles<control2[vector])begin
          @(negedge clk);captured_valid=mem_valid;captured_address=mem_addr;captured_data=mem_wdata;captured_strb=mem_wstrb;
          if(control3[vector]!=32'hffffffff&&cycles==control3[vector])irq[0]=1;else irq=0;
          @(posedge clk);#1;cycles=cycles+1;
          if(captured_valid&&captured_strb!=0)begin
            if(captured_address[31:2]<64)begin
              if(captured_strb[0])program_memory[captured_address[31:2]][7:0]=captured_data[7:0];
              if(captured_strb[1])program_memory[captured_address[31:2]][15:8]=captured_data[15:8];
              if(captured_strb[2])program_memory[captured_address[31:2]][23:16]=captured_data[23:16];
              if(captured_strb[3])program_memory[captured_address[31:2]][31:24]=captured_data[31:24];
            end else protocol_error=1;
            store_count=store_count+1;last_address=captured_address;last_data=captured_data;
            store_signature=store_signature^((captured_address*32'h9e3779b1)^captured_data);
          end
          if($isunknown({trap,mem_valid,mem_addr,mem_wdata,mem_wstrb}))unknown_seen=1;
        end
        timed_out=!trap;response={{trap?32'd1:32'd0,store_count[31:0],last_address,last_data,store_signature,96'b0}};
      end fi_enable=0;fi_onehot='0;irq=0;@(posedge clk);
    end endtask'''
    return common_testbench("picorv32_cpu", declarations, setup, task)


def generate_testbench(family_id: str) -> str:
    return {
        "opentitan_hmac_sha256": hmac_testbench,
        "picorv32_cpu": cpu_testbench,
        "secworks_aes": aes_testbench,
        "secworks_sha256": sha_testbench,
    }[family_id]()


def find_unique_source(family_id: str, filename: str) -> Path:
    source_root = RESULT / "multicircuit_corpus_12c1c/sources" / family_id / "rtl_source"
    require(source_root.is_dir(), f"corpus source root: {family_id}")
    matches = sorted(source_root.rglob(filename))
    require(len(matches) == 1, f"expected one {filename} for {family_id}; found {len(matches)}")
    return matches[0]


def verify_record_item(item: dict[str, Any], label: str) -> None:
    path = ROOT / str(item.get("path", ""))
    require(path.is_file(), f"missing {label}: {item.get('path')}")
    require(sha256(path) == item.get("sha256"), f"{label} SHA")
    require(path.stat().st_size == int(item.get("bytes", -1)), f"{label} size")


def prepare_family(state: dict[str, Any], family_id: str, arrays: dict[str, np.ndarray]) -> tuple[list[dict[str, Any]], Path]:
    family_state = state["families"].setdefault(family_id, {})
    if family_state.get("build_status") == "PASS":
        for key in ("derived_json", "derived_verilog", "testbench", "binary"):
            verify_record_item(family_state[key], f"resume {family_id} {key}")
        for index, item in enumerate(family_state.get("compile_sources", [])):
            verify_record_item(item, f"resume {family_id} compile source {index}")
        for index, item in enumerate(family_state.get("include_files", [])):
            verify_record_item(item, f"resume {family_id} include file {index}")
        return enumerate_sites(family_id, WORK_1E / "families" / family_id / f"{family_id}_generic_12c1e.json"), ROOT / family_state["binary"]["path"]

    print(f"{family_id}: DERIVE 2,048-SITE INSTRUMENTED NETLIST", flush=True)
    family_build = BUILD / family_id
    family_raw = RAW / family_id
    family_memory = MEMORY / family_id
    family_build.mkdir(parents=True, exist_ok=True)
    family_raw.mkdir(parents=True, exist_ok=True)
    family_memory.mkdir(parents=True, exist_ok=True)
    source_json = WORK_1E / "families" / family_id / f"{family_id}_generic_12c1e.json"
    sites = enumerate_sites(family_id, source_json)
    derived_json = family_build / f"{family_id}_pilot_instrumented_12c1i.json"
    derived_verilog = family_build / f"{family_id}_pilot_instrumented_12c1i.v"
    write_or_verify(derived_json, instrument_netlist(family_id, source_json, sites))
    yosys_log = family_raw / "yosys_instrumented_netlist.log"
    yosys = shutil.which("yosys")
    require(yosys is not None, "Yosys unavailable")
    command_text = f"read_json {derived_json.resolve()}; hierarchy -check -top {TOPS[family_id]}; write_verilog -noattr {derived_verilog.resolve()}"
    require(run_logged([yosys, "-p", command_text], yosys_log, 1800) == 0, f"{family_id} instrumented netlist generation; inspect {rel(yosys_log)}")
    require(derived_verilog.is_file() and derived_verilog.stat().st_size > 0, f"instrumented Verilog: {family_id}")

    material = vector_material(arrays, family_id)
    for path, payload in memory_payloads(family_id, material, family_memory).items():
        write_or_verify(path, payload)
    testbench = family_build / f"tb_v22_{family_id}_12c1i.sv"
    write_or_verify(testbench, generate_testbench(family_id).encode())

    sources: list[Path] = [derived_verilog]
    include_files: list[Path] = []
    include_dirs: list[Path] = []
    if family_id == "opentitan_hmac_sha256":
        require(HMAC_BRIDGE_SOURCE.is_file(), f"missing validated HMAC bridge: {rel(HMAC_BRIDGE_SOURCE)}")
        require(sha256(HMAC_BRIDGE_SOURCE) == HMAC_BRIDGE_SHA256, "validated HMAC bridge SHA")
        bridge = family_build / "v22_hmac_bridge_12c1i.sv"
        adapter = family_build / "v22_hmac_adapter_12c1i.sv"
        write_or_verify(bridge, transform_hmac_bridge(HMAC_BRIDGE_SOURCE.read_text(encoding="utf-8")).encode())
        write_or_verify(adapter, hmac_adapter_sv().encode())
        prim_sources = [find_unique_source(family_id, name) for name in (
            "prim_sha2_pkg.sv", "prim_sha2_pad.sv", "prim_sha2.sv", "prim_sha2_32.sv"
        )]
        # R2: prim_sha2_pad.sv includes prim_assert.sv. Keep the frozen source
        # untouched and expose its existing directory to the Verilator frontend.
        prim_assert = find_unique_source(family_id, "prim_assert.sv")
        include_files = [prim_assert]
        include_dirs = [prim_assert.parent]
        sources = [prim_sources[0], derived_verilog, *prim_sources[1:], bridge, adapter]

    top = f"tb_v22_{family_id}_12c1i"
    obj_dir = family_build / "obj_dir"
    build_log = family_raw / "verilator_build.log"
    verilator = shutil.which("verilator")
    require(verilator is not None, "Verilator unavailable")
    compile_command = [
        verilator, "--binary", "--timing", "--assert", "-Wall", "-Wno-fatal",
        "-Wno-DECLFILENAME", "-Wno-PINMISSING", "--error-limit", "0", "-j", "1",
        "--Mdir", str(obj_dir), "--top-module", top,
        *[f"-I{path}" for path in include_dirs],
        *[str(path) for path in sources], str(testbench),
    ]
    print(f"{family_id}: BUILD", flush=True)
    require(run_logged(compile_command, build_log, 3600) == 0, f"{family_id} Verilator build; inspect {rel(build_log)}")
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
    expected_count = VALIDATION_VECTORS[family_id]
    require(len(rows) == expected_count, f"validation row count: {family_id}")
    parsed: list[dict[str, Any]] = []
    failures = 0
    for index, row in enumerate(rows):
        require(row["family_id"] == family_id and row["mode"] == "VALIDATE", f"validation identity: {family_id}/{index}")
        require(row["run_type"] == "BASELINE" and int(row["vector_rank"]) == index, f"validation ordering: {family_id}/{index}")
        timeout = int(row["timed_out"])
        protocol = int(row["protocol_error"])
        unknown = int(row["unknown"])
        is_canary = index == expected_count - 1
        passed = unknown == 0 and protocol == 0 and ((timeout == 1) if is_canary else (timeout == 0 and row["actual_response"] == row["expected_response"]))
        failures += int(not passed)
        parsed.append({
            "family_id": family_id, "partition": PARTITIONS[family_id], "vector_rank": index,
            "transaction_id": int(row["transaction_id"]), "timeout_canary": "YES" if is_canary else "NO",
            "cycles": int(row["cycles"]), "timed_out": timeout, "protocol_error": protocol,
            "unknown": unknown, "expected_response": row["expected_response"],
            "actual_response": row["actual_response"], "result": "PASS" if passed else "FAIL",
        })
    require(failures == 0, f"{family_id} adapter functional validation failed ({failures} records); inspect {rel(path)}")
    return parsed, {"family_id": family_id, "status": "PASS", "vectors": expected_count,
                    "pilot_vectors": PILOT_VECTORS[family_id], "timeout_canaries": 1,
                    "known_answer_failures": 0, "protocol_failures": 0, "unknown_records": 0}


def validate_family(state: dict[str, Any], family_id: str, binary: Path) -> list[dict[str, Any]]:
    family_state = state["families"][family_id]
    if family_state.get("validation_status") == "PASS":
        for key in ("validation_csv_a", "validation_csv_b"):
            verify_record_item(family_state[key], f"resume {family_id} {key}")
        path = ROOT / family_state["validation_csv_a"]["path"]
        rows, _ = parse_validation(path, family_id)
        require((ROOT / family_state["validation_csv_b"]["path"]).read_bytes() == path.read_bytes(), f"resume validation replay: {family_id}")
        print(f"{family_id}: VALIDATION CHECKPOINT PASS", flush=True)
        return rows

    family_raw = RAW / family_id
    csv_a = family_raw / "adapter_validation_replay_a.csv"
    csv_b = family_raw / "adapter_validation_replay_b.csv"
    records: list[list[dict[str, Any]]] = []
    summaries = []
    for replay, csv_path in (("a", csv_a), ("b", csv_b)):
        log = family_raw / f"adapter_validation_replay_{replay}.log"
        resources = family_raw / f"adapter_validation_replay_{replay}_resources.log"
        print(f"{family_id}: FAULT-FREE VALIDATION REPLAY {replay.upper()}", flush=True)
        command = [str(binary), "+MODE=VALIDATE", f"+CSV={csv_path.resolve()}"]
        require(run_logged(command, log, 14400, resources) == 0, f"{family_id} validation replay {replay}; inspect {rel(log)}")
        require("V22_ADAPTER_VALIDATION_RESULT=PASS" in log.read_text(errors="replace"), f"{family_id} validation PASS token")
        parsed, summary = parse_validation(csv_path, family_id)
        records.append(parsed)
        summaries.append(summary)
    require(csv_a.read_bytes() == csv_b.read_bytes(), f"byte-exact fault-free replay: {family_id}")
    family_state.update({
        "validation_status": "PASS", "validated_at": now(), "validation_summary": summaries[0],
        "validation_csv_a": record(csv_a), "validation_csv_b": record(csv_b),
        "validation_replay": "PASS / BYTE-EXACT",
    })
    state["updated_at"] = now()
    atomic_json(CHECKPOINT, state)
    print(f"{family_id}: VALIDATION PASS / BYTE-EXACT", flush=True)
    return records[0]


def parse_pilot_batch(path: Path, family_id: str, batch_id: int, site_start: int,
                      site_count: int) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    rows = read_csv(path)
    vectors = PILOT_VECTORS[family_id]
    expected_rows = vectors + site_count * 2 * vectors
    require(len(rows) == expected_rows, f"pilot row count: {family_id}/{batch_id}")
    baseline_rows = rows[:vectors]
    baseline_response = np.zeros((vectors, 32), dtype=np.uint8)
    baseline_cycles = np.zeros(vectors, dtype=np.int32)
    baseline_timeout = np.zeros(vectors, dtype=np.uint8)
    for vector, row in enumerate(baseline_rows):
        require(row["family_id"] == family_id and row["mode"] == "PILOT", "pilot baseline family/mode")
        require(row["run_type"] == "BASELINE" and int(row["vector_rank"]) == vector, "pilot baseline order")
        require(int(row["unknown"]) == 0 and int(row["protocol_error"]) == 0 and int(row["timed_out"]) == 0, "pilot baseline protocol")
        require(row["actual_response"] == row["expected_response"], "pilot baseline oracle")
        baseline_response[vector] = np.frombuffer(bytes.fromhex(row["actual_response"]), dtype=np.uint8)
        baseline_cycles[vector] = int(row["cycles"])
        baseline_timeout[vector] = int(row["timed_out"])

    local_faults = site_count * 2
    response_xor = np.zeros((local_faults, vectors, 32), dtype=np.uint8)
    cycle_delta = np.zeros((local_faults, vectors), dtype=np.int32)
    timed_out = np.zeros((local_faults, vectors), dtype=np.uint8)
    protocol_error = np.zeros((local_faults, vectors), dtype=np.uint8)
    seen = np.zeros((local_faults, vectors), dtype=np.uint8)
    unknown = 0
    for row in rows[vectors:]:
        require(row["run_type"] == "ENABLED", "pilot enabled run type")
        site = int(row["site_rank"])
        stuck = int(row["stuck_value"])
        vector = int(row["vector_rank"])
        require(site_start <= site < site_start + site_count and stuck in (0, 1) and 0 <= vector < vectors, "pilot enabled coordinates")
        fault = (site - site_start) * 2 + stuck
        require(seen[fault, vector] == 0, "duplicate pilot record")
        seen[fault, vector] = 1
        actual = np.frombuffer(bytes.fromhex(row["actual_response"]), dtype=np.uint8)
        response_xor[fault, vector] = actual ^ baseline_response[vector]
        cycle_delta[fault, vector] = int(row["cycles"]) - int(baseline_cycles[vector])
        timed_out[fault, vector] = int(row["timed_out"])
        protocol_error[fault, vector] = int(row["protocol_error"])
        unknown += int(row["unknown"])
    require(np.all(seen == 1), f"pilot sample coverage: {family_id}/{batch_id}")
    require(unknown == 0, f"pilot unknown response records: {family_id}/{batch_id}")
    detected = (
        np.any(response_xor != 0, axis=2) |
        (cycle_delta != 0) |
        (timed_out != baseline_timeout[None, :]) |
        (protocol_error != 0)
    ).astype(np.uint8)
    arrays = {
        "response_xor": response_xor,
        "cycle_delta": cycle_delta,
        "timed_out": timed_out,
        "protocol_error": protocol_error,
        "detected": detected,
        "site_rank": np.repeat(np.arange(site_start, site_start + site_count, dtype=np.int32), 2),
        "stuck_value": np.tile(np.asarray([0, 1], dtype=np.uint8), site_count),
        "baseline_response": baseline_response,
        "baseline_cycles": baseline_cycles,
    }
    summary = {
        "family_id": family_id, "batch_id": batch_id, "site_start": site_start,
        "sites": site_count, "faults": local_faults, "pilot_vectors": vectors,
        "baseline_records": vectors, "enabled_records": local_faults * vectors,
        "observable_faults": int(np.sum(np.any(detected != 0, axis=1))),
        "unknown_records": 0, "baseline_failures": 0,
    }
    return arrays, summary


def run_pilot_batches(state: dict[str, Any], family_id: str, binary: Path) -> None:
    family_state = state["families"][family_id]
    completed = set(int(value) for value in family_state.get("completed_batches", []))
    family_state.setdefault("batches", {})
    for local_batch in range(BATCHES_PER_FAMILY):
        global_batch = AUTHORIZED.index(family_id) * BATCHES_PER_FAMILY + local_batch
        key = f"{global_batch:03d}"
        site_start = local_batch * BATCH_SITES
        if global_batch in completed:
            item = family_state["batches"][key]
            verify_record_item(item["csv"], f"resume pilot CSV {key}")
            verify_record_item(item["npz"], f"resume pilot NPZ {key}")
            print(f"{family_id} Batch {local_batch:02d}/{BATCHES_PER_FAMILY - 1:02d}: CHECKPOINT PASS", flush=True)
            continue
        batch_dir = RAW / family_id / f"batch_{local_batch:02d}"
        batch_dir.mkdir(parents=True, exist_ok=True)
        csv_path = batch_dir / f"pilot_batch_{local_batch:02d}.csv"
        log = batch_dir / "simulation.log"
        resources = batch_dir / "resources.log"
        print(f"{family_id} Batch {local_batch:02d}: SIMULATE sites={site_start}-{site_start + BATCH_SITES - 1}", flush=True)
        command = [
            str(binary), "+MODE=PILOT", f"+CSV={csv_path.resolve()}", f"+BATCH_ID={global_batch}",
            f"+SITE_START={site_start}", f"+SITE_COUNT={BATCH_SITES}",
        ]
        require(run_logged(command, log, 43200, resources) == 0, f"pilot simulation {family_id}/{local_batch}; inspect {rel(log)}")
        require("V22_PILOT_BATCH_RESULT=PASS" in log.read_text(errors="replace"), f"pilot PASS token: {family_id}/{local_batch}")
        batch_arrays, summary = parse_pilot_batch(csv_path, family_id, global_batch, site_start, BATCH_SITES)
        batch_npz = PER_BATCH / family_id / f"pilot_batch_{local_batch:02d}.npz"
        batch_npz.parent.mkdir(parents=True, exist_ok=True)
        write_or_verify(batch_npz, deterministic_npz(batch_arrays))
        item = {
            "status": "PASS", "completed_at": now(), "global_batch_id": global_batch,
            "local_batch_id": local_batch, "csv": record(csv_path), "npz": record(batch_npz),
            "simulation_log": record(log), "resource_log": record(resources), "summary": summary,
        }
        family_state["batches"][key] = item
        completed.add(global_batch)
        family_state["completed_batches"] = sorted(completed)
        state["completed_batches"] = sorted(set(int(value) for value in state.get("completed_batches", [])) | {global_batch})
        state["updated_at"] = now()
        atomic_json(CHECKPOINT, state)
        print(f"{family_id} Batch {local_batch:02d}: PASS enabled={summary['enabled_records']}", flush=True)
    require(len(completed) == BATCHES_PER_FAMILY, f"incomplete family pilot: {family_id}")
    family_state["pilot_status"] = "PASS"
    state["updated_at"] = now()
    atomic_json(CHECKPOINT, state)


def signature_digest(family_id: str, response_xor: np.ndarray, cycle_delta: np.ndarray,
                     timed_out: np.ndarray, protocol_error: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(family_id.encode() + b"\0")
    digest.update(np.ascontiguousarray(response_xor).tobytes())
    digest.update(np.ascontiguousarray(cycle_delta.astype("<i4", copy=False)).tobytes())
    digest.update(np.ascontiguousarray(timed_out).tobytes())
    digest.update(np.ascontiguousarray(protocol_error).tobytes())
    return digest.hexdigest()


def consolidate_and_freeze(state: dict[str, Any], arrays_1g: dict[str, np.ndarray],
                           validation_rows: list[dict[str, Any]]) -> None:
    require(len(state.get("completed_batches", [])) == TOTAL_BATCHES, "pilot batches incomplete")
    response_xor = np.zeros((TOTAL_FAULTS, 16, 32), dtype=np.uint8)
    response_validity = np.zeros((TOTAL_FAULTS, 16, 32), dtype=np.uint8)
    cycle_delta = np.zeros((TOTAL_FAULTS, 16), dtype=np.int32)
    timed_out = np.zeros((TOTAL_FAULTS, 16), dtype=np.uint8)
    protocol_error = np.zeros((TOTAL_FAULTS, 16), dtype=np.uint8)
    detected = np.zeros((TOTAL_FAULTS, 16), dtype=np.uint8)
    vector_mask = np.zeros((TOTAL_FAULTS, 16), dtype=np.uint8)
    baseline_response = np.zeros((4, 16, 32), dtype=np.uint8)
    baseline_cycles = np.zeros((4, 16), dtype=np.int32)
    transaction_ids = np.full((4, 16), -1, dtype=np.int32)
    family_index = np.repeat(np.arange(4, dtype=np.uint8), FAULTS_PER_FAMILY)
    local_site_index = np.tile(np.repeat(np.arange(SITES_PER_FAMILY, dtype=np.int32), 2), 4)
    stuck_value = np.tile(np.tile(np.asarray([0, 1], dtype=np.uint8), SITES_PER_FAMILY), 4)
    opaque_ids: list[bytes] = []
    catalog_rows: list[dict[str, Any]] = []
    raw_records: dict[str, Any] = {}
    batch_records: dict[str, Any] = {}

    validity_by_family = {
        "opentitan_hmac_sha256": np.ones(32, dtype=np.uint8),
        "picorv32_cpu": np.asarray([1] * 20 + [0] * 12, dtype=np.uint8),
        "secworks_aes": np.asarray([0] * 16 + [1] * 16, dtype=np.uint8),
        "secworks_sha256": np.ones(32, dtype=np.uint8),
    }

    for family_number, family_id in enumerate(AUTHORIZED):
        vectors = PILOT_VECTORS[family_id]
        sites = enumerate_sites(family_id, WORK_1E / "families" / family_id / f"{family_id}_generic_12c1e.json")
        pilot_indices, _ = family_vector_indices(arrays_1g, family_id)
        transaction_ids[family_number, :vectors] = arrays_1g["transaction_id"][pilot_indices].astype(np.int32)
        family_base = family_number * FAULTS_PER_FAMILY
        for site in sites:
            for stuck in (0, 1):
                local_fault = int(site["site_rank"]) * 2 + stuck
                global_fault = family_base + local_fault
                opaque = f"F{family_number}-{local_fault:04d}-{hashlib.sha256((site['opaque_site_id'] + ':' + str(stuck)).encode()).hexdigest()[:12]}"
                opaque_ids.append(opaque.encode())
                catalog_rows.append({
                    "fault_instance_index": global_fault, "opaque_fault_id": opaque,
                    "family_id": family_id, "partition": PARTITIONS[family_id],
                    "family_index": family_number, "site_rank": site["site_rank"], "stuck_value": stuck,
                    "cell_name": site["cell_name"], "cell_type": site["cell_type"],
                    "output_port": site["output_port"], "output_bit_index": site["output_bit_index"],
                    "net_bit_id": site["net_bit_id"],
                })

        reference_baseline_response: np.ndarray | None = None
        reference_baseline_cycles: np.ndarray | None = None
        for local_batch in range(BATCHES_PER_FAMILY):
            global_batch = family_number * BATCHES_PER_FAMILY + local_batch
            key = f"{global_batch:03d}"
            item = state["families"][family_id]["batches"][key]
            csv_path = ROOT / item["csv"]["path"]
            npz_path = ROOT / item["npz"]["path"]
            verify_record_item(item["csv"], f"pilot CSV {key}")
            verify_record_item(item["npz"], f"pilot NPZ {key}")
            local, summary = parse_pilot_batch(csv_path, family_id, global_batch, local_batch * BATCH_SITES, BATCH_SITES)
            require(deterministic_npz(local) == npz_path.read_bytes(), f"deterministic batch replay: {family_id}/{local_batch}")
            if reference_baseline_response is None:
                reference_baseline_response = local["baseline_response"]
                reference_baseline_cycles = local["baseline_cycles"]
            else:
                require(np.array_equal(reference_baseline_response, local["baseline_response"]), f"cross-batch response baseline: {family_id}")
                require(np.array_equal(reference_baseline_cycles, local["baseline_cycles"]), f"cross-batch cycle baseline: {family_id}")
            local_start = local_batch * BATCH_SITES * 2
            global_start = family_base + local_start
            count = BATCH_SITES * 2
            response_xor[global_start:global_start + count, :vectors] = local["response_xor"]
            cycle_delta[global_start:global_start + count, :vectors] = local["cycle_delta"]
            timed_out[global_start:global_start + count, :vectors] = local["timed_out"]
            protocol_error[global_start:global_start + count, :vectors] = local["protocol_error"]
            detected[global_start:global_start + count, :vectors] = local["detected"]
            vector_mask[global_start:global_start + count, :vectors] = 1
            response_validity[global_start:global_start + count, :vectors] = validity_by_family[family_id][None, None, :]
            raw_records[key] = record(csv_path)
            batch_records[key] = record(npz_path)
            require(summary == item["summary"], f"checkpoint summary replay: {family_id}/{local_batch}")
        require(reference_baseline_response is not None and reference_baseline_cycles is not None, f"family baseline: {family_id}")
        baseline_response[family_number, :vectors] = reference_baseline_response
        baseline_cycles[family_number, :vectors] = reference_baseline_cycles

    require(len(catalog_rows) == TOTAL_FAULTS and len(opaque_ids) == TOTAL_FAULTS, "fault catalog dimensions")
    require(len(set(opaque_ids)) == TOTAL_FAULTS, "opaque fault ID uniqueness")
    require(int(vector_mask.sum()) == MAX_PILOT_TRANSACTIONS, "enabled transaction matrix")

    observable = np.any(detected != 0, axis=1)
    candidate_count = np.zeros(TOTAL_FAULTS, dtype=np.int32)
    exact_site = np.zeros(TOTAL_FAULTS, dtype=np.uint8)
    signature_values = ["NORMAL_COMPATIBLE"] * TOTAL_FAULTS
    signature_rows: list[dict[str, Any]] = []
    family_metrics: list[dict[str, Any]] = []
    for family_number, family_id in enumerate(AUTHORIZED):
        vectors = PILOT_VECTORS[family_id]
        start = family_number * FAULTS_PER_FAMILY
        end = start + FAULTS_PER_FAMILY
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
                "family_id": family_id, "signature_sha256": signature,
                "fault_instances": len(members), "candidate_sites": len(candidate_sites),
                "site_rank_min": min(candidate_sites), "site_rank_max": max(candidate_sites),
            })
        obs_counts = candidate_count[start:end][observable[start:end]]
        observed_faults = int(observable[start:end].sum())
        family_metrics.append({
            "family_id": family_id, "partition": PARTITIONS[family_id], "sites": SITES_PER_FAMILY,
            "fault_instances": FAULTS_PER_FAMILY, "pilot_vectors": vectors,
            "enabled_transactions": FAULTS_PER_FAMILY * vectors,
            "observable_faults": observed_faults,
            "all_injected_detection_recall": observed_faults / FAULTS_PER_FAMILY,
            "unique_signature_faults": int(np.sum((candidate_count[start:end] == 1) & observable[start:end])),
            "ambiguous_signature_faults": int(np.sum((candidate_count[start:end] > 1) & observable[start:end])),
            "all_injected_exact_site_rate": float(exact_site[start:end].mean()),
            "mean_observable_candidate_sites": float(obs_counts.mean()) if obs_counts.size else 0.0,
            "maximum_observable_candidate_sites": int(obs_counts.max()) if obs_counts.size else 0,
            "fault_free_false_alarms": 0,
        })

    observable_counts = candidate_count[observable]
    metrics = {
        "metrics_version": "CIRCUITSAGE-HMAC-V2.2-BOUNDED-PILOT-METRICS-12C1I-v1",
        "stage": STAGE, "status": "PASS / REPORT-ONLY", "created_at": now(),
        "scope": "GENERALIZATION_TRAIN + GENERALIZATION_CALIBRATION BOUNDED PILOT ONLY",
        "families": family_metrics, "sites": TOTAL_SITES, "fault_instances": TOTAL_FAULTS,
        "pilot_vectors": TOTAL_PILOT_VECTORS, "enabled_transactions": MAX_PILOT_TRANSACTIONS,
        "observable_faults": int(observable.sum()),
        "macro_all_injected_detection_recall": float(observable.mean()),
        "observable_candidate_set_coverage": 1.0 if bool(observable.any()) else 0.0,
        "unique_signature_top1_site": 1.0 if bool(np.any(candidate_count == 1)) else 0.0,
        "all_injected_exact_site_rate": float(exact_site.mean()),
        "mean_observable_candidate_sites": float(observable_counts.mean()) if observable_counts.size else 0.0,
        "maximum_observable_candidate_sites": int(observable_counts.max()) if observable_counts.size else 0,
        "ambiguous_false_unique_rate": 0.0, "fault_free_false_alarms": 0,
        "fault_free_false_alarm_rate": 0.0, "adapter_validation": "PASS / 4 OF 4 / BYTE-EXACT REPLAY",
        "dataset_integrity": "PASS", "pilot_advancement": "NOT DECIDED — SEPARATE DISPOSITION REQUIRED",
        "independent_generalization": "NOT ESTABLISHED", "model_training_inference": [0, 0],
    }

    rng = np.random.default_rng(12030109)
    site_detection = observable.reshape(4, SITES_PER_FAMILY, 2).mean(axis=2).reshape(-1)
    site_exact = exact_site.reshape(4, SITES_PER_FAMILY, 2).mean(axis=2).reshape(-1)
    bootstrap_rows: list[dict[str, Any]] = []
    for replicate in range(BOOTSTRAP_REPLICATES):
        sample = rng.integers(0, TOTAL_SITES, size=TOTAL_SITES)
        bootstrap_rows.append({
            "replicate": replicate,
            "all_injected_detection_recall": f"{float(site_detection[sample].mean()):.8f}",
            "all_injected_exact_site_rate": f"{float(site_exact[sample].mean()):.8f}",
        })

    features_arrays = {
        "family_index": family_index,
        "vector_mask": vector_mask,
        "response_validity_mask": response_validity,
        "response_xor": response_xor,
        "completion_cycle_delta": cycle_delta,
        "timeout": timed_out,
        "protocol_error": protocol_error,
        "observable": observable.astype(np.uint8),
        "baseline_response": baseline_response,
        "baseline_cycles": baseline_cycles,
        "pilot_transaction_id": transaction_ids,
    }
    targets_arrays = {
        "fault_instance_index": np.arange(TOTAL_FAULTS, dtype=np.int32),
        "family_index": family_index,
        "local_site_index": local_site_index,
        "stuck_value": stuck_value,
        "opaque_fault_id": np.asarray(opaque_ids, dtype="S32"),
        "candidate_site_count": candidate_count,
        "exact_site": exact_site,
        "behavior_signature_sha256": np.asarray([value.encode() for value in signature_values], dtype="S64"),
    }
    schema = {
        "schema_version": "CIRCUITSAGE-HMAC-V2.2-BOUNDED-PILOT-DATASET-12C1I-v1",
        "stage": STAGE, "status": "FROZEN", "features": sorted(features_arrays),
        "targets": sorted(targets_arrays), "sample_axis": "fault_instance",
        "vector_axis": "family-specific pilot vector padded to 16 with vector_mask=0",
        "response_bytes": 32, "fault_identity_in_features": "PROHIBITED / ABSENT",
        "truth_separation": "cell/site/stuck/opaque fault IDs exist only in target NPZ and catalog CSV",
        "graph_binding": "Stage 12C-1E frozen family netlist graph; construction deferred to subsequent full-dataset gate",
        "protected_partitions": {"independent_test": "LOCKED", "validation": "UNOPENED", "holdout": "SEALED"},
    }

    validation_fields = [
        "family_id", "partition", "vector_rank", "transaction_id", "timeout_canary", "cycles",
        "timed_out", "protocol_error", "unknown", "expected_response", "actual_response", "result",
    ]
    signature_fields = ["family_id", "signature_sha256", "fault_instances", "candidate_sites", "site_rank_min", "site_rank_max"]
    bootstrap_fields = ["replicate", "all_injected_detection_recall", "all_injected_exact_site_rate"]
    frozen_write(VALIDATION_RESULTS, csv_bytes(validation_rows, validation_fields))
    frozen_write(FAULT_CATALOG, csv_bytes(catalog_rows, FAULT_FIELDS))
    frozen_write(FEATURES, deterministic_npz(features_arrays))
    frozen_write(TARGETS, deterministic_npz(targets_arrays))
    frozen_write(SIGNATURES, csv_bytes(signature_rows, signature_fields))
    frozen_write(METRICS, canonical_json(metrics))
    frozen_write(BOOTSTRAP, csv_bytes(bootstrap_rows, bootstrap_fields))
    frozen_write(SCHEMA, canonical_json(schema))

    report = f"""# CircuitSage-HMAC V2.2 bounded pilot — Stage 12C-1I

All four TRAIN/CALIBRATION adapters passed real RTL functional validation,
including independent known-answer/reference-oracle checks and byte-exact
fault-free replay.  The conditionally authorized pilot was then executed for
{TOTAL_FAULTS} single persistent SA0/SA1 faults and
{MAX_PILOT_TRANSACTIONS} enabled transactions.

The report-only pilot all-injected detection recall is
{metrics['macro_all_injected_detection_recall']:.8f}; the all-injected exact-site
ceiling is {metrics['all_injected_exact_site_rate']:.8f}.  These are bounded
TRAIN/CALIBRATION pilot results, not evidence of independent-circuit
generalization.  A separate disposition gate must decide whether a full
campaign is justified.

No model was trained or deserialized.  INDEPENDENT_TEST remains locked,
VALIDATION remains unopened, and HOLDOUT remains sealed.  The future hybrid
release name remains **{FUTURE_BRAND}**.
""".encode()
    frozen_write(REPORT, report)

    state["status"] = "PASS / FROZEN"
    state["adapter_validation"] = "PASS / 4 OF 4 / BYTE-EXACT"
    state["pilot_execution"] = "COMPLETED / 128 OF 128 BATCHES"
    state["updated_at"] = now()
    state["metrics"] = metrics
    atomic_json(CHECKPOINT, state)

    stage_outputs = [CHECKPOINT, VALIDATION_RESULTS, FAULT_CATALOG, FEATURES, TARGETS,
                     SIGNATURES, METRICS, BOOTSTRAP, SCHEMA, REPORT]
    manifest = {
        "manifest_version": "CIRCUITSAGE-HMAC-V2.2-ADAPTER-PILOT-EXECUTION-MANIFEST-12C1I-v1",
        "stage": STAGE, "status": "PASS", "created_at": now(),
        "stage_source": record(Path(__file__).resolve()),
        "frozen_inputs": {rel(path): record(path) for path in PINNED},
        "outputs": {rel(path): record(path) for path in stage_outputs},
        "raw_batch_csvs": raw_records, "per_batch_npz": batch_records,
        "families": list(AUTHORIZED), "adapter_validation_families": 4,
        "sites": TOTAL_SITES, "fault_instances": TOTAL_FAULTS,
        "pilot_vectors": TOTAL_PILOT_VECTORS, "enabled_transactions": MAX_PILOT_TRANSACTIONS,
        "simulation_batches": TOTAL_BATCHES, "model_training_calls": 0,
        "inference_calls": 0, "model_deserializations": 0,
        "independent_test_access": 0, "validation_access": 0, "holdout_access": 0,
    }
    frozen_write(MANIFEST, canonical_json(manifest))
    audit = {
        "audit_version": "CIRCUITSAGE-HMAC-V2.2-ADAPTER-FUNCTIONAL-VALIDATION-PILOT-EXECUTION-FREEZE-12C1I-v1",
        "stage": STAGE, "status": "PASS", "execution_dataset": "COMPLETED / FROZEN",
        "adapter_functional_validation": "PASS / 4 OF 4 / BYTE-EXACT REPLAY",
        "bounded_pilot_campaign": "COMPLETED / FROZEN", "families": "4 / TRAIN 3 / CALIBRATION 1",
        "sites_faults_vectors": [TOTAL_SITES, TOTAL_FAULTS, TOTAL_PILOT_VECTORS],
        "enabled_transactions": MAX_PILOT_TRANSACTIONS, "simulation_batches": f"{TOTAL_BATCHES}/{TOTAL_BATCHES}",
        "observable_faults": metrics["observable_faults"],
        "macro_all_injected_detection_recall": metrics["macro_all_injected_detection_recall"],
        "all_injected_exact_site_rate": metrics["all_injected_exact_site_rate"],
        "mean_maximum_observable_candidate_sites": [metrics["mean_observable_candidate_sites"], metrics["maximum_observable_candidate_sites"]],
        "fault_free_false_alarms": 0, "dataset_integrity": "PASS",
        "pilot_advancement": "NOT DECIDED / SEPARATE DISPOSITION REQUIRED",
        "full_campaign_model_training": "NOT AUTHORIZED / NOT AUTHORIZED",
        "independent_generalization": "NOT ESTABLISHED",
        "independent_test": "LOCKED / 2 FAMILIES", "holdout": "SEALED / 1 FAMILY",
        "independent_test_validation_holdout_access": [0, 0, 0],
        "fault_identity_in_features": "PROHIBITED / ABSENT",
        "frozen_rtl_netlists_vectors_modified": [False, False, False],
        "metrics_record": record(METRICS), "features_record": record(FEATURES),
        "targets_record": record(TARGETS), "manifest_record": record(MANIFEST),
        "future_combined_model_brand": FUTURE_BRAND,
        "next_gate": "STAGE 12C-1J — BOUNDED PILOT RESULT DISPOSITION AND FULL-CAMPAIGN READINESS FREEZE",
    }
    frozen_write(AUDIT, canonical_json(audit))

    for path in (METRICS, SCHEMA, MANIFEST, AUDIT):
        require(path.read_bytes() == canonical_json(load_json(path)), f"canonical output replay: {path.name}")
    require(VALIDATION_RESULTS.read_bytes() == csv_bytes(validation_rows, validation_fields), "validation CSV replay")
    require(FAULT_CATALOG.read_bytes() == csv_bytes(catalog_rows, FAULT_FIELDS), "fault catalog replay")
    require(FEATURES.read_bytes() == deterministic_npz(features_arrays), "feature NPZ replay")
    require(TARGETS.read_bytes() == deterministic_npz(targets_arrays), "target NPZ replay")
    for path, expected in PINNED.items():
        require(sha256(path) == expected, f"frozen input modified: {rel(path)}")

    print("\nSTAGE 12C-1I — TRAIN/CALIBRATION ADAPTER FUNCTIONAL-VALIDATION AND BOUNDED PILOT FAULT-CAMPAIGN EXECUTION FREEZE")
    print(f"{'Status':<88}: PASS")
    print(f"{'Adapter functional validation':<88}: PASS / 4 OF 4 / BYTE-EXACT REPLAY")
    print(f"{'Pilot execution / dataset':<88}: COMPLETED / FROZEN")
    print(f"{'Families / sites / faults / pilot vectors':<88}: 4 / {TOTAL_SITES} / {TOTAL_FAULTS} / {TOTAL_PILOT_VECTORS}")
    print(f"{'Simulation batches / enabled transactions':<88}: {TOTAL_BATCHES}/{TOTAL_BATCHES} / {MAX_PILOT_TRANSACTIONS}")
    print(f"{'Observable faults / detection recall':<88}: {metrics['observable_faults']} / {metrics['macro_all_injected_detection_recall']:.8f}")
    print(f"{'All-injected exact-site rate':<88}: {metrics['all_injected_exact_site_rate']:.8f}")
    print(f"{'Mean / maximum observable candidate sites':<88}: {metrics['mean_observable_candidate_sites']:.4f} / {metrics['maximum_observable_candidate_sites']}")
    print(f"{'Fault-free false alarms':<88}: 0")
    print(f"{'Pilot advancement':<88}: NOT DECIDED — SEPARATE DISPOSITION REQUIRED")
    print(f"{'Full campaign / model training':<88}: NOT AUTHORIZED / NOT AUTHORIZED")
    print(f"{'Independent TEST / VALIDATION / HOLDOUT access':<88}: 0 / 0 / 0")
    print(f"{'Independent generalization':<88}: NOT ESTABLISHED")
    print(f"{'Features':<88}: {FEATURES}")
    print(f"{'Features SHA':<88}: {sha256(FEATURES)}")
    print(f"{'Metrics':<88}: {METRICS}")
    print(f"{'Metrics SHA':<88}: {sha256(METRICS)}")
    print(f"{'Manifest':<88}: {MANIFEST}")
    print(f"{'Manifest SHA':<88}: {sha256(MANIFEST)}")
    print(f"{'Audit':<88}: {AUDIT}")
    print(f"{'Audit SHA':<88}: {sha256(AUDIT)}")
    print(f"{'Next gate':<88}: STAGE 12C-1J — BOUNDED PILOT RESULT DISPOSITION AND FULL-CAMPAIGN READINESS FREEZE")


def execute(resume: bool) -> None:
    require(ROOT.name == "vlsi_fault_detection_v2", f"place script in project root, not {ROOT}")
    final_outputs = (VALIDATION_RESULTS, FAULT_CATALOG, FEATURES, TARGETS, SIGNATURES,
                     METRICS, BOOTSTRAP, SCHEMA, REPORT, MANIFEST, AUDIT)
    require(not any(path.exists() for path in final_outputs), "frozen Stage 12C-1I output already exists; use --status")
    arrays, _ = verify_inputs()
    WORK.mkdir(parents=True, exist_ok=True)
    BUILD.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    PER_BATCH.mkdir(parents=True, exist_ok=True)
    MEMORY.mkdir(parents=True, exist_ok=True)

    if CHECKPOINT.exists():
        require(resume, "checkpoint exists; rerun with --resume")
        state = load_json(CHECKPOINT)
        require(state.get("stage") == STAGE and state.get("status") == "RUNNING", "checkpoint state")
    else:
        require(not resume, "--resume requested without checkpoint")
        state = {
            "checkpoint_version": "CIRCUITSAGE-HMAC-V2.2-ADAPTER-PILOT-CHECKPOINT-12C1I-v1",
            "stage": STAGE, "status": "RUNNING", "started_at": now(), "updated_at": now(),
            "authorized_families": list(AUTHORIZED), "planned_batches": TOTAL_BATCHES,
            "completed_batches": [], "families": {},
            "independent_test_access": 0, "validation_access": 0, "holdout_access": 0,
        }
        atomic_json(CHECKPOINT, state)

    binaries: dict[str, Path] = {}
    for family_id in AUTHORIZED:
        _, binaries[family_id] = prepare_family(state, family_id, arrays)

    validation_rows: list[dict[str, Any]] = []
    for family_id in AUTHORIZED:
        validation_rows.extend(validate_family(state, family_id, binaries[family_id]))
    require(all(state["families"][family_id].get("validation_status") == "PASS" for family_id in AUTHORIZED), "all-family adapter gate")
    state["adapter_validation"] = "PASS / 4 OF 4 / BYTE-EXACT"
    state["pilot_unlock"] = "AUTHORIZED PRECONDITION SATISFIED"
    state["updated_at"] = now()
    atomic_json(CHECKPOINT, state)
    print("ALL FOUR ADAPTERS: PASS — BOUNDED PILOT UNLOCKED", flush=True)

    for family_id in AUTHORIZED:
        run_pilot_batches(state, family_id, binaries[family_id])
    consolidate_and_freeze(state, arrays, validation_rows)


def locked_execute(resume: bool) -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    with LOCK.open("a+", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            stop("Stage 12C-1I execution lock is held by another process")
        execute(resume)


def status() -> None:
    print("STAGE 12C-1I — ADAPTER/PILOT EXECUTION STATUS")
    if MANIFEST.is_file() and AUDIT.is_file():
        manifest = load_json(MANIFEST)
        audit = load_json(AUDIT)
        require(manifest.get("status") == "PASS" and audit.get("status") == "PASS", "frozen status")
        for item in manifest.get("outputs", {}).values():
            verify_record_item(item, "frozen output")
        require(sha256(MANIFEST) == audit["manifest_record"]["sha256"], "audit manifest record")
        print("Status                    : PASS / FROZEN")
        print(f"Adapter validation        : {audit['adapter_functional_validation']}")
        print(f"Pilot batches             : {audit['simulation_batches']}")
        print(f"Faults / transactions     : {TOTAL_FAULTS} / {MAX_PILOT_TRANSACTIONS}")
        print(f"Detection recall          : {audit['macro_all_injected_detection_recall']:.8f}")
        print(f"Exact-site rate           : {audit['all_injected_exact_site_rate']:.8f}")
        print(f"Audit                     : {AUDIT}")
        print(f"Audit SHA                 : {sha256(AUDIT)}")
        return
    if not CHECKPOINT.is_file():
        print("Status                    : NOT STARTED")
        print(f"Checkpoint                : {CHECKPOINT}")
        return
    state = load_json(CHECKPOINT)
    completed = state.get("completed_batches", [])
    passed_validation = [family for family in AUTHORIZED if state.get("families", {}).get(family, {}).get("validation_status") == "PASS"]
    built = [family for family in AUTHORIZED if state.get("families", {}).get(family, {}).get("build_status") == "PASS"]
    print(f"Status                    : {state.get('status')}")
    print(f"Families built            : {len(built)}/4")
    print(f"Adapters validated        : {len(passed_validation)}/4")
    print(f"Completed pilot batches   : {len(completed)}/{TOTAL_BATCHES}")
    if completed:
        print(f"Latest completed batch    : {max(completed):03d}")
    print(f"Checkpoint                : {CHECKPOINT}")
    print(f"Checkpoint SHA            : {sha256(CHECKPOINT)}")


def self_test() -> None:
    require(TOTAL_FAULTS == 16384 and TOTAL_SITES == 8192, "pilot fault dimensions")
    require(TOTAL_PILOT_VECTORS == 60 and MAX_PILOT_TRANSACTIONS == 245760, "pilot transaction dimensions")
    key128 = bytes.fromhex("000102030405060708090a0b0c0d0e0f")
    plaintext = bytes.fromhex("00112233445566778899aabbccddeeff")
    ciphertext = bytes.fromhex("69c4e0d86a7b0430d8cdb78070b4c55a")
    require(aes_block(key128, plaintext, False) == ciphertext, "AES-128 encryption oracle")
    require(aes_block(key128, ciphertext, True) == plaintext, "AES-128 decryption oracle")
    key256 = bytes(range(32))
    cipher256 = bytes.fromhex("8ea2b7ca516745bfeafc49904b496089")
    require(aes_block(key256, plaintext, False) == cipher256, "AES-256 encryption oracle")
    abc_block = b"abc" + b"\x80" + bytes(52) + (24).to_bytes(8, "big")
    require(sha256_compress(abc_block).hex() == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad", "SHA-256 compression oracle")
    program = (0x00100093).to_bytes(4, "little") + (0x00100073).to_bytes(4, "little")
    require(cpu_oracle(program, 16)[:4] == (1).to_bytes(4, "big"), "RV32I trap oracle")
    tiny = {
        "creator": "self-test", "modules": {"tiny": {"attributes": {},
            "ports": {"a": {"direction": "input", "bits": [2]}, "y": {"direction": "output", "bits": [3]}},
            "cells": {"gate": {"hide_name": 0, "type": "$_NOT_", "parameters": {}, "attributes": {},
                "port_directions": {"A": "input", "Y": "output"}, "connections": {"A": [2], "Y": [3]}}},
            "netnames": {"a": {"hide_name": 0, "bits": [2], "attributes": {}}, "y": {"hide_name": 0, "bits": [3], "attributes": {}}}}},
    }
    temp = Path("/tmp/stage_12c1i_selftest.json")
    temp.write_bytes(canonical_json(tiny))
    original_top = TOPS["secworks_aes"]
    try:
        TOPS["secworks_aes"] = "tiny"
        try:
            sites = enumerate_sites("secworks_aes", temp)
        except SystemExit as exc:
            require("has only 1 eligible" in str(exc), "site-budget self-test")
            sites = [{"site_rank": 0, "cell_name": "gate", "output_port": "Y", "output_bit_index": 0, "net_bit_id": 3}]
        rewritten = json.loads(instrument_netlist("secworks_aes", temp, sites))
        require("$v22fi_mux$0" in rewritten["modules"]["tiny"]["cells"], "fault-mux rewrite self-test")
    except SystemExit as exc:
        stop(f"netlist rewrite self-test: {exc}")
    finally:
        TOPS["secworks_aes"] = original_top
        temp.unlink(missing_ok=True)
    # Exercise the netlist rewrite with an isolated one-site copy.
    design = copy.deepcopy(tiny)
    module = design["modules"]["tiny"]
    maximum = max(integer_bits(module))
    require(maximum == 3 and sites[0]["net_bit_id"] == 3, "netlist-site self-test")
    generated = common_testbench("secworks_aes", "", "", "task automatic run_transaction(input integer a,input logic b,input integer c,input logic d,output integer e,output logic f,output logic g,output logic h,output logic[255:0]i);begin e=0;f=0;g=0;h=0;i=0;end endtask")
    require('$fwrite(csv_fd,"secworks_aes,%s' in generated, "testbench CSV writer")
    require("\\n\"" in generated and '\n"' not in generated.replace("\\n\"", ""), "SystemVerilog newline escaping")
    sample = {"a": np.arange(8, dtype=np.uint8), "b": np.arange(3, dtype=np.int32)}
    require(deterministic_npz(sample) == deterministic_npz(sample), "deterministic NPZ")
    print("Stage 12C-1I self-test: PASS")


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
