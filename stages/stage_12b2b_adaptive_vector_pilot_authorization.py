#!/usr/bin/env python3
"""Stage 12B-2B: adaptive vector pool and pilot-screening authorization freeze.

Verifies the frozen Stage 12B-2A improvement contract, deterministically creates
the contracted 512-vector stimulus pool and 1,024-site REPAIR_TRAIN pilot plan,
and freezes authorization for pilot screening only. It does not simulate a
fault, generate response labels, deserialize a model, train, or run inference.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import struct
import sys
import zipfile
from pathlib import Path
from typing import Any

try:
    import numpy as np
except ImportError as error:
    raise SystemExit("STOP: activate the project .venv (NumPy required)") from error


STAGE = "12B-2B"
ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "config/v2_1_improvement"
RESULT = ROOT / "results/circuitsage_hmac_v2_12b2"
POOL_DIR = RESULT / "adaptive_vector_pool_12b2b"

SOURCE = ROOT / "stage_12b2a_v2_1_improvement_contract.py"
SCOPE = CONFIG / "circuitsage_hmac_v2_1_improvement_scope_contract_12b2a.json"
PARTITION = CONFIG / "circuitsage_hmac_v2_1_repair_partition_contract_12b2a.json"
VECTOR_POLICY = CONFIG / "circuitsage_hmac_v2_1_adaptive_vector_policy_12b2a.json"
ARCHITECTURE = CONFIG / "circuitsage_hmac_v2_1_repair_architecture_contract_12b2a.json"
ACCEPTANCE = CONFIG / "circuitsage_hmac_v2_1_repair_acceptance_contract_12b2a.json"
CANDIDATE_GRID = CONFIG / "circuitsage_hmac_v2_1_repair_candidate_grid_12b2a.csv"
REPAIR_SPLIT = RESULT / "circuitsage_hmac_v2_1_repair_site_split_12b2a.csv"
ENVIRONMENT_2A = RESULT / "circuitsage_hmac_v2_1_improvement_environment_12b2a.json"
MANIFEST_2A = RESULT / "circuitsage_hmac_v2_1_improvement_contract_manifest_12b2a.json"
AUDIT_2A = RESULT / "circuitsage_hmac_v2_1_improvement_contract_freeze_12b2a.json"

VECTOR_NPZ = POOL_DIR / "circuitsage_hmac_v2_1_candidate_vectors_12b2b.npz"
VECTOR_SCHEMA = POOL_DIR / "circuitsage_hmac_v2_1_candidate_vector_schema_12b2b.json"
PILOT_SITES = POOL_DIR / "circuitsage_hmac_v2_1_pilot_screening_sites_12b2b.csv"
EXECUTION_CONTRACT = CONFIG / "circuitsage_hmac_v2_1_pilot_screening_execution_contract_12b2b.json"
AUTHORIZATION = CONFIG / "circuitsage_hmac_v2_1_pilot_screening_authorization_12b2b.json"
MANIFEST = RESULT / "circuitsage_hmac_v2_1_pilot_authorization_manifest_12b2b.json"
AUDIT = RESULT / "circuitsage_hmac_v2_1_pilot_authorization_freeze_12b2b.json"

SOURCE_SHA = "26d7250da239fc667e497e616c680eba0ef364fe2880a748006f0996c3d01ae9"
VECTOR_COUNT = 512
PILOT_SITE_COUNT = 1024
PILOT_FAULT_COUNT = 2048
SEED = 20260916
DOMAIN = "CIRCUITSAGE-HMAC-V2.1-ADAPTIVE-DEVELOPMENT-VECTORS-12B2A-v1"
KEY_BYTES = 32
MESSAGE_BYTES = 32

EXPECTED_2A_OUTPUTS = {
    SCOPE, PARTITION, VECTOR_POLICY, ARCHITECTURE, ACCEPTANCE,
    CANDIDATE_GRID, REPAIR_SPLIT, ENVIRONMENT_2A,
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
    return str(path.relative_to(ROOT))


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode()


def load_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing input: {rel(path)}")
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


def verify_record(item: dict[str, Any], expected_path: Path) -> None:
    require(item.get("path") == rel(expected_path), f"manifest path: {rel(expected_path)}")
    require(expected_path.is_file(), f"missing frozen 12B-2A output: {rel(expected_path)}")
    require(item.get("sha256") == sha256(expected_path), f"manifest SHA: {rel(expected_path)}")
    require(item.get("bytes") == expected_path.stat().st_size, f"manifest size: {rel(expected_path)}")


def verify_2a() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    print("STAGE 12B-2B — ADAPTIVE VECTOR POOL AND PILOT-SCREENING AUTHORIZATION")
    print("FROZEN INPUT VERIFICATION")
    require(SOURCE.is_file() and sha256(SOURCE) == SOURCE_SHA, "Stage 12B-2A source SHA")
    print(f"  {SOURCE.name:<91}: OK")
    manifest = load_json(MANIFEST_2A)
    audit = load_json(AUDIT_2A)
    require(manifest.get("stage") == "12B-2A" and manifest.get("status") == "PASS", "12B-2A manifest status")
    require(audit.get("stage") == "12B-2A" and audit.get("status") == "PASS", "12B-2A audit status")
    require(audit.get("scope_status") == "FROZEN", "12B-2A scope freeze")
    require(audit.get("partition_contract_status") == "FROZEN", "12B-2A partition freeze")
    require(audit.get("vector_policy_status") == "FROZEN", "12B-2A vector-policy freeze")
    require(audit.get("acceptance_status") == "FROZEN", "12B-2A acceptance freeze")
    require(audit.get("adaptive_pilot_generation") == "AUTHORIZED / NOT STARTED", "pilot-generation authorization")
    require(audit.get("model_training") == "NOT YET AUTHORIZED", "training boundary")
    require(audit.get("repair_site_test") == "LOCKED / NOT AUTHORIZED", "repair-site-test boundary")
    require(audit.get("original_dev_site_test") == "CONSUMED / NOT REOPENED", "original test protection")
    require(audit.get("validation_access_count") == 0 and audit.get("holdout_access_count") == 0, "protected partition access")
    outputs = manifest.get("outputs")
    require(isinstance(outputs, dict), "12B-2A manifest outputs")
    expected_keys = {rel(path) for path in EXPECTED_2A_OUTPUTS}
    require(set(outputs) == expected_keys, "12B-2A manifest output set")
    for path in sorted(EXPECTED_2A_OUTPUTS):
        verify_record(outputs[rel(path)], path)
        print(f"  {path.name:<91}: OK")
    manifest_record = audit.get("manifest")
    require(isinstance(manifest_record, dict), "12B-2A audit manifest record")
    verify_record(manifest_record, MANIFEST_2A)
    print(f"  {MANIFEST_2A.name:<91}: OK")
    print(f"  {AUDIT_2A.name:<91}: PASS / FROZEN")

    policy = load_json(VECTOR_POLICY)
    partition = load_json(PARTITION)
    require(policy.get("status") == "FROZEN", "vector policy status")
    require(policy.get("candidate_vector_domain") == DOMAIN, "vector domain")
    require(policy.get("candidate_vector_seed") == SEED, "vector seed")
    require(policy.get("candidate_vectors") == VECTOR_COUNT, "candidate-vector count")
    require(policy.get("key_bytes") == KEY_BYTES and policy.get("message_bytes") == MESSAGE_BYTES, "vector widths")
    require(policy.get("pilot_screening_sites") == PILOT_SITE_COUNT, "pilot-site count")
    require(policy.get("pilot_screening_fault_instances") == PILOT_FAULT_COUNT, "pilot-fault count")
    require(policy.get("pilot_site_source") == "REPAIR_TRAIN ONLY", "pilot source")
    require(policy.get("held_out_vector_payloads_used") == 0, "held-out vector use")
    require(partition.get("status") == "FROZEN", "partition contract status")
    require(partition.get("repair_site_test_policy") == "LOCK UNTIL A SEPARATE AUTHORIZATION AFTER MODEL SELECTION", "repair test lock")
    return manifest, audit, policy


def derive_vector(index: int, label: bytes, size: int) -> bytes:
    output = bytearray()
    counter = 0
    while len(output) < size:
        payload = (
            DOMAIN.encode("utf-8")
            + struct.pack("<Q", SEED)
            + struct.pack("<I", index)
            + label
            + struct.pack("<I", counter)
        )
        output.extend(hashlib.sha256(payload).digest())
        counter += 1
    return bytes(output[:size])


def build_vectors() -> dict[str, np.ndarray]:
    indices = np.arange(VECTOR_COUNT, dtype=np.int32)
    keys = np.vstack([
        np.frombuffer(derive_vector(i, b"key", KEY_BYTES), dtype=np.uint8)
        for i in range(VECTOR_COUNT)
    ])
    messages = np.vstack([
        np.frombuffer(derive_vector(i, b"message", MESSAGE_BYTES), dtype=np.uint8)
        for i in range(VECTOR_COUNT)
    ])
    require(len({row.tobytes() for row in keys}) == VECTOR_COUNT, "duplicate candidate keys")
    require(len({row.tobytes() for row in messages}) == VECTOR_COUNT, "duplicate candidate messages")
    return {"vector_index": indices, "key_u8": keys, "message_u8": messages}


def npy_bytes(array: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.lib.format.write_array(stream, array, allow_pickle=False)
    return stream.getvalue()


def deterministic_npz(arrays: dict[str, np.ndarray]) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in sorted(arrays):
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            info.create_system = 3
            archive.writestr(info, npy_bytes(arrays[name]), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    return stream.getvalue()


def build_pilot_sites() -> tuple[bytes, list[dict[str, str]]]:
    with REPAIR_SPLIT.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        expected = {
            "site_id", "site_index", "repair_partition", "repair_partition_rank",
            "repair_split_hash_sha256", "source_partition",
        }
        require(set(reader.fieldnames or []) == expected, "repair split header")
        rows = [row for row in reader if row["repair_partition"] == "REPAIR_TRAIN"]
    require(len(rows) == 11191, "REPAIR_TRAIN site count")
    rows.sort(key=lambda row: int(row["repair_partition_rank"]))
    pilot = rows[:PILOT_SITE_COUNT]
    require([int(row["repair_partition_rank"]) for row in pilot] == list(range(PILOT_SITE_COUNT)), "pilot rank continuity")
    require(all(row["source_partition"] == "DEV_TRAIN" for row in pilot), "pilot lineage")
    output_rows = [{
        "pilot_rank": str(rank),
        "site_id": row["site_id"],
        "site_index": row["site_index"],
        "repair_partition_rank": row["repair_partition_rank"],
        "fault_polarities": "SA0|SA1",
    } for rank, row in enumerate(pilot)]
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(output_rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(output_rows)
    return output.getvalue().encode(), output_rows


def self_test() -> None:
    first = derive_vector(0, b"key", 32)
    require(len(first) == 32, "vector derivation width")
    require(first == derive_vector(0, b"key", 32), "vector derivation replay")
    arrays = build_vectors()
    require(deterministic_npz(arrays) == deterministic_npz(arrays), "NPZ replay")
    print("Stage 12B-2B self-test: PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    require(ROOT.name == "vlsi_fault_detection_v2", f"run from project root, not {ROOT}")
    outputs = (VECTOR_NPZ, VECTOR_SCHEMA, PILOT_SITES, EXECUTION_CONTRACT, AUTHORIZATION, MANIFEST, AUDIT)
    for path in outputs:
        require(not path.exists(), f"Stage {STAGE} output already exists: {rel(path)}")

    manifest_2a, audit_2a, policy = verify_2a()
    arrays = build_vectors()
    vector_payload = deterministic_npz(arrays)
    pilot_payload, pilot_rows = build_pilot_sites()
    vector_commitment = hashlib.sha256(vector_payload).hexdigest()
    pilot_commitment = hashlib.sha256(pilot_payload).hexdigest()

    schema = {
        "schema_version": "CIRCUITSAGE-HMAC-V2.1-CANDIDATE-VECTORS-12B2B-v1",
        "stage": STAGE, "status": "FROZEN", "format": "DETERMINISTIC NPZ",
        "arrays": {
            "vector_index": {"dtype": "int32", "shape": [VECTOR_COUNT]},
            "key_u8": {"dtype": "uint8", "shape": [VECTOR_COUNT, KEY_BYTES]},
            "message_u8": {"dtype": "uint8", "shape": [VECTOR_COUNT, MESSAGE_BYTES]},
        },
        "derivation": policy["derivation"], "domain": DOMAIN, "seed": SEED,
        "candidate_vectors": VECTOR_COUNT, "response_or_label_columns": 0,
        "validation_or_holdout_payloads": 0,
    }
    execution = {
        "contract_version": "CIRCUITSAGE-HMAC-V2.1-PILOT-SCREENING-EXECUTION-v1",
        "stage": STAGE, "status": "FROZEN",
        "candidate_vector_pool": record_placeholder(VECTOR_NPZ, vector_payload),
        "pilot_site_plan": record_placeholder(PILOT_SITES, pilot_payload),
        "pilot_sites": PILOT_SITE_COUNT, "fault_polarities": ["SA0", "SA1"],
        "pilot_fault_instances": PILOT_FAULT_COUNT,
        "source_partition": "REPAIR_TRAIN ONLY",
        "execution": "SEQUENTIAL", "parallel_batches": 1, "build_jobs": 1,
        "checkpoint_resume": "REQUIRED", "deterministic_replay": "REQUIRED / EXACT",
        "allowed_outputs": ["golden response", "faulty response", "activation", "detection signature", "timing status"],
        "prohibited_actions": [
            "model training", "model selection", "model inference", "adaptive selection using REPAIR_CALIBRATION",
            "REPAIR_SITE_TEST access", "original DEV_SITE_TEST access", "VALIDATION access", "HOLDOUT access",
        ],
    }
    authorization = {
        "authorization_version": "CIRCUITSAGE-HMAC-V2.1-PILOT-SCREENING-AUTHORIZATION-v1",
        "stage": STAGE, "status": "FROZEN",
        "candidate_vector_generation": "COMPLETED AND FROZEN",
        "pilot_screening": "AUTHORIZED / NOT STARTED",
        "pilot_scope": {"vectors": VECTOR_COUNT, "sites": PILOT_SITE_COUNT, "fault_instances": PILOT_FAULT_COUNT},
        "full_repair_campaign": "NOT AUTHORIZED",
        "adaptive_final_vector_selection": "NOT AUTHORIZED UNTIL PILOT DATASET INTEGRITY FREEZE",
        "model_training": "NOT AUTHORIZED",
        "repair_site_test": "LOCKED / NOT AUTHORIZED",
        "original_dev_site_test": "CONSUMED / REOPENING PROHIBITED",
        "validation": "PROHIBITED", "holdout": "PROHIBITED",
        "v1_v2_core_v2_1_modification": "PROHIBITED",
        "vector_commitment_sha256": vector_commitment,
        "pilot_site_commitment_sha256": pilot_commitment,
    }

    frozen_write(VECTOR_NPZ, vector_payload)
    frozen_write(VECTOR_SCHEMA, canonical_json(schema))
    frozen_write(PILOT_SITES, pilot_payload)
    frozen_write(EXECUTION_CONTRACT, canonical_json(execution))
    frozen_write(AUTHORIZATION, canonical_json(authorization))
    primary = (VECTOR_NPZ, VECTOR_SCHEMA, PILOT_SITES, EXECUTION_CONTRACT, AUTHORIZATION)
    manifest = {
        "manifest_version": "CIRCUITSAGE-HMAC-V2.1-PILOT-AUTHORIZATION-MANIFEST-v1",
        "stage": STAGE, "status": "PASS",
        "stage_12b2a_manifest": record(MANIFEST_2A),
        "stage_12b2a_audit": record(AUDIT_2A),
        "outputs": {rel(path): record(path) for path in primary},
        "candidate_vectors": VECTOR_COUNT, "pilot_sites": len(pilot_rows),
        "pilot_fault_instances": PILOT_FAULT_COUNT,
        "simulation_calls": 0, "response_payloads_generated": 0,
        "model_objects_deserialized": 0, "training_calls": 0, "inference_calls": 0,
        "repair_site_test_access": 0, "original_dev_site_test_access": 0,
        "validation_access": 0, "holdout_access": 0,
    }
    frozen_write(MANIFEST, canonical_json(manifest))
    audit = {
        "audit_version": "CIRCUITSAGE-HMAC-V2.1-PILOT-AUTHORIZATION-FREEZE-v1",
        "stage": STAGE, "status": "PASS",
        "candidate_vector_pool_status": "FROZEN",
        "pilot_site_plan_status": "FROZEN",
        "execution_contract_status": "FROZEN",
        "authorization_status": "FROZEN",
        "pilot_screening": "AUTHORIZED / NOT STARTED",
        "candidate_vectors": VECTOR_COUNT, "pilot_sites": PILOT_SITE_COUNT,
        "pilot_fault_instances": PILOT_FAULT_COUNT,
        "candidate_vector_commitment": vector_commitment,
        "pilot_site_commitment": pilot_commitment,
        "model_training": "NOT AUTHORIZED", "full_campaign": "NOT AUTHORIZED",
        "repair_site_test": "LOCKED / NOT AUTHORIZED",
        "original_dev_site_test": "CONSUMED / NOT REOPENED",
        "validation_access": 0, "holdout_access": 0,
        "v1_modified": False, "v2_core_modified": False, "v2_1_modified": False,
        "manifest": record(MANIFEST),
        "next_gate": "STAGE 12B-2C — ADAPTIVE VECTOR PILOT-SCREENING EXECUTION AND DATASET INTEGRITY FREEZE",
    }
    frozen_write(AUDIT, canonical_json(audit))

    require(vector_payload == VECTOR_NPZ.read_bytes(), "vector NPZ replay")
    require(pilot_payload == PILOT_SITES.read_bytes(), "pilot CSV replay")
    with np.load(VECTOR_NPZ, allow_pickle=False) as loaded:
        require(set(loaded.files) == set(arrays), "NPZ member set")
        for name, expected in arrays.items():
            require(np.array_equal(loaded[name], expected), f"NPZ array replay: {name}")
    for path in (VECTOR_SCHEMA, EXECUTION_CONTRACT, AUTHORIZATION, MANIFEST, AUDIT):
        require(canonical_json(load_json(path)) == path.read_bytes(), f"JSON replay: {path.name}")

    print("\nSTAGE 12B-2B — ADAPTIVE TEST-VECTOR CANDIDATE POOL AND PILOT-SCREENING AUTHORIZATION FREEZE")
    print(f"{'Status':<48}: PASS")
    print(f"{'Candidate vector pool':<48}: FROZEN — {VECTOR_COUNT}")
    print(f"{'Key / message bytes':<48}: {KEY_BYTES} / {MESSAGE_BYTES}")
    print(f"{'Pilot sites / fault instances':<48}: {PILOT_SITE_COUNT} / {PILOT_FAULT_COUNT}")
    print(f"{'Pilot source':<48}: REPAIR_TRAIN ONLY")
    print(f"{'Pilot screening':<48}: AUTHORIZED / NOT STARTED")
    print(f"{'Simulation / response generation':<48}: 0 / 0")
    print(f"{'Full campaign / model training':<48}: NOT AUTHORIZED / NOT AUTHORIZED")
    print(f"{'Repair SITE_TEST':<48}: LOCKED / NOT AUTHORIZED")
    print(f"{'Original DEV_SITE_TEST':<48}: CONSUMED / NOT REOPENED")
    print(f"{'VALIDATION / HOLDOUT access':<48}: 0 / 0")
    print(f"{'V1 / V2 Core / V2.1 modified':<48}: NO / NO / NO")
    print(f"{'Candidate vectors':<48}: {VECTOR_NPZ}")
    print(f"{'Candidate vectors SHA':<48}: {sha256(VECTOR_NPZ)}")
    print(f"{'Pilot sites':<48}: {PILOT_SITES}")
    print(f"{'Pilot sites SHA':<48}: {sha256(PILOT_SITES)}")
    print(f"{'Execution contract':<48}: {EXECUTION_CONTRACT}")
    print(f"{'Execution contract SHA':<48}: {sha256(EXECUTION_CONTRACT)}")
    print(f"{'Authorization':<48}: {AUTHORIZATION}")
    print(f"{'Authorization SHA':<48}: {sha256(AUTHORIZATION)}")
    print(f"{'Manifest':<48}: {MANIFEST}")
    print(f"{'Manifest SHA':<48}: {sha256(MANIFEST)}")
    print(f"{'Audit':<48}: {AUDIT}")
    print(f"{'Audit SHA':<48}: {sha256(AUDIT)}")
    print(f"{'Next gate':<48}: STAGE 12B-2C — ADAPTIVE VECTOR PILOT-SCREENING EXECUTION AND DATASET INTEGRITY FREEZE")


def record_placeholder(path: Path, payload: bytes) -> dict[str, Any]:
    return {"path": rel(path), "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}


if __name__ == "__main__":
    main()
