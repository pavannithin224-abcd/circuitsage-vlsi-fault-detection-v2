#!/usr/bin/env python3
"""Stage 12A-1B: blinded TRAIN-response dataset and schema freeze.

Builds factorized, deterministic NPZ artifacts from the already-frozen 11E-1B
behavior catalog. Response-only model inputs are kept separate from fault/site
targets. No simulation, model loading, training, validation, or HOLDOUT access
is performed.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import sqlite3
import struct
import sys
import zipfile
from pathlib import Path
from typing import Any

try:
    import numpy as np
except ImportError as error:
    print("STOP: NumPy is required in the existing project environment", file=sys.stderr)
    raise SystemExit(1) from error


STAGE = "12A-1B"
ROOT = Path(__file__).resolve().parent
SITES = 22839
FAULTS = 45678
VECTORS = 64
TOKEN_BYTES = 35
PROFILE_BYTES = VECTORS * TOKEN_BYTES
OBSERVATION_HEADER = ["vector_slot", "vector_id", "actual_digest", "cycles", "timed_out"]
SPLIT_HEADER = ["site_id", "site_index", "partition", "partition_rank", "split_hash_sha256"]
PARTITION_CODE = {"DEV_TRAIN": 0, "DEV_CALIBRATION": 1, "DEV_SITE_TEST": 2}

PINNED_INPUTS = {
    "stage_12a1a_v2_core_contract.py":
        "1fca14eb7384794baa5b8a0bc68aa2124f3a92fbd1f8d80f63984e96e39f9b9e",
    "config/v2/circuitsage_hmac_v2_core_scope_contract_12a1a.json":
        "38118c0ba90c7d2af6ef8a01fcf7bd1e86ef13929fa7766e640db1ee08d482e6",
    "config/v2/circuitsage_hmac_v2_linked_pipeline_contract_12a1a.json":
        "312fde4cb5ecc7819418e4affe88d1b07c7dbe5cbdeab5af92f14403ffd7427d",
    "config/v2/circuitsage_hmac_v2_acceptance_contract_12a1a.json":
        "3943601adcdd603e78b07699317066b3a187d22d1f7f9b71106aab4feed95c93",
    "results/circuitsage_hmac_v2_12a1/circuitsage_hmac_v2_core_contract_freeze_12a1a.json":
        "b0d1a0c2bebbe0a4579dad5fca0efec01b8d569926cb4cdab1efc9e9a086fa75",
    "results/hmac_behavior_diagnosis_pilot/baseline_11e1b/freeze.json":
        "a6f7312b9cd8a137da711133cd425a706c1705a84eb444df4a45eb2e3129ee71",
    "results/hmac_fault_campaign_11c5/hmac_fault_site_group_split_11c5d.csv":
        "602fa310547f68d1e9b5ceb6f148d6b125c69df0589929f9edfde9d264922c61",
    "results/hmac_fault_campaign_11c5/hmac_fault_site_group_split_manifest_11c5d.json":
        "1c96b7c7c7f5301e70037aaa9a0d18effd28cae6bf15e9f85cc85f0c413473fb",
}

BASELINE_DIR = ROOT / "results/hmac_behavior_diagnosis_pilot/baseline_11e1b"
CATALOG_PATH = BASELINE_DIR / "catalog.sqlite3"
NORMAL_PATH = BASELINE_DIR / "normal.csv"
SPLIT_PATH = ROOT / "results/hmac_fault_campaign_11c5/hmac_fault_site_group_split_11c5d.csv"

RESULT_DIR = ROOT / "results/circuitsage_hmac_v2_12a1"
DATASET_DIR = RESULT_DIR / "blinded_train_response_dataset_12a1b"
FEATURES_PATH = DATASET_DIR / "circuitsage_hmac_v2_train_response_features_12a1b.npz"
TARGETS_PATH = DATASET_DIR / "circuitsage_hmac_v2_train_response_targets_12a1b.npz"
SCHEMA_PATH = DATASET_DIR / "circuitsage_hmac_v2_train_response_schema_12a1b.json"
MANIFEST_PATH = RESULT_DIR / "circuitsage_hmac_v2_train_response_manifest_12a1b.json"
AUDIT_PATH = RESULT_DIR / "circuitsage_hmac_v2_train_response_dataset_freeze_12a1b.json"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_frozen(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(path.read_bytes() == data, f"existing frozen output differs: {path}")
        return
    temporary = path.with_name(path.name + ".tmp")
    require(not temporary.exists(), f"stale temporary output exists: {temporary}")
    temporary.write_bytes(data)
    temporary.replace(path)


def npy_bytes(array: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.lib.format.write_array(
        stream,
        np.ascontiguousarray(array),
        version=(2, 0),
        allow_pickle=False,
    )
    return stream.getvalue()


def deterministic_npz(arrays: dict[str, np.ndarray]) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        for name in sorted(arrays):
            require(re.fullmatch(r"[a-z][a-z0-9_]*", name) is not None, f"invalid NPZ key: {name}")
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o600 << 16
            archive.writestr(
                info,
                npy_bytes(arrays[name]),
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            )
    return stream.getvalue()


def verify_pinned_inputs() -> dict[str, str]:
    verified = {}
    print("FROZEN INPUT VERIFICATION")
    for relative, expected in PINNED_INPUTS.items():
        path = ROOT / relative
        require(path.is_file(), f"missing frozen input: {relative}")
        actual = sha256_file(path)
        require(actual == expected, f"SHA mismatch for {relative}: expected {expected}, actual {actual}")
        if path.suffix == ".json":
            load_json(path)
        verified[relative] = actual
        print(f"  {path.name:<78}: OK")
    return verified


def verify_baseline_bundle() -> tuple[dict[str, Any], dict[str, str]]:
    freeze = load_json(BASELINE_DIR / "freeze.json")
    require(freeze.get("stage") == "11E-1B", "behavior baseline stage")
    require(freeze.get("status") == "PASS", "behavior baseline status")
    outputs = freeze.get("outputs")
    require(isinstance(outputs, dict), "behavior baseline output records")
    required = {"catalog.sqlite3", "normal.csv", "summary.json", "checks.json"}
    require(required <= set(outputs), "behavior baseline required outputs")

    hashes: dict[str, str] = {}
    for name, record in outputs.items():
        require(isinstance(record, dict), f"invalid baseline record: {name}")
        relative = record.get("path")
        expected = record.get("sha256")
        require(isinstance(relative, str) and isinstance(expected, str), f"invalid baseline binding: {name}")
        path = ROOT / relative
        require(path.is_file(), f"missing baseline output: {relative}")
        actual = sha256_file(path)
        require(actual == expected, f"baseline output SHA mismatch: {relative}")
        hashes[relative] = actual
    return freeze, hashes


def read_normal_profile() -> tuple[bytes, np.ndarray]:
    tokens = bytearray()
    vector_ids = []
    with NORMAL_PATH.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames == OBSERVATION_HEADER, "normal observation header")
        for expected_slot, row in enumerate(reader):
            require(expected_slot < VECTORS, "extra normal observation row")
            slot = int(row["vector_slot"])
            require(slot == expected_slot, "normal observation slot order")
            vector_ids.append(int(row["vector_id"]))
            timeout = int(row["timed_out"])
            cycles = int(row["cycles"])
            require(timeout in (0, 1), "normal timeout value")
            require(0 <= cycles <= 2000, "normal cycle range")
            digest_text = row["actual_digest"]
            require(timeout == 0, "normal profile unexpectedly timed out")
            require(re.fullmatch(r"[0-9a-fA-F]{64}", digest_text) is not None, "normal digest")
            tokens.extend(struct.pack(">BH", timeout, cycles))
            tokens.extend(bytes.fromhex(digest_text))
    require(len(vector_ids) == VECTORS, "normal observation count")
    require(len(set(vector_ids)) == VECTORS, "duplicate normal vector ID")
    require(len(tokens) == PROFILE_BYTES, "normal profile width")
    return bytes(tokens), np.asarray(vector_ids, dtype=np.uint64)


def read_site_partitions() -> tuple[np.ndarray, dict[str, int]]:
    codes = np.full(SITES + 1, 255, dtype=np.uint8)
    counts = {name: 0 for name in PARTITION_CODE}
    with SPLIT_PATH.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames == SPLIT_HEADER, "site split header")
        for row in reader:
            site_index = int(row["site_index"])
            site_id = row["site_id"]
            partition = row["partition"]
            require(1 <= site_index <= SITES, "site split index range")
            require(site_id == f"HMAC-STEM-{site_index:06d}", "site split ID/index binding")
            require(partition in PARTITION_CODE, "unknown site split partition")
            require(codes[site_index] == 255, "duplicate site split record")
            codes[site_index] = PARTITION_CODE[partition]
            counts[partition] += 1
    require(np.all(codes[1:] != 255), "missing site split record")
    require(counts == {"DEV_TRAIN": 15987, "DEV_CALIBRATION": 3426, "DEV_SITE_TEST": 3426},
            "site partition counts")
    return codes, counts


def open_catalog() -> sqlite3.Connection:
    require(CATALOG_PATH.is_file(), "behavior catalog missing")
    connection = sqlite3.connect(CATALOG_PATH.resolve().as_uri() + "?mode=ro", uri=True)
    connection.execute("PRAGMA query_only=ON")
    connection.execute("PRAGMA trusted_schema=OFF")
    require(connection.execute("PRAGMA integrity_check").fetchone() == ("ok",), "catalog integrity")
    return connection


def construct_arrays(normal: bytes, vector_ids: np.ndarray, site_partitions: np.ndarray):
    connection = open_catalog()
    try:
        metadata = dict(connection.execute("SELECT key,value FROM metadata"))
        require(metadata.get("version") == "HMAC-BEHAVIOR-DETECTOR-LOCATOR-v1", "catalog version")
        profile_rows = connection.execute(
            "SELECT profile_id,response FROM profiles ORDER BY profile_id"
        ).fetchall()
        require(profile_rows, "empty behavior catalog")
        profile_count = len(profile_rows)
        require([int(row[0]) for row in profile_rows] == list(range(profile_count)),
                "catalog profile IDs are not contiguous")

        fault_rows = connection.execute(
            "SELECT fault_instance,profile_id FROM faults ORDER BY fault_instance"
        ).fetchall()
        require(len(fault_rows) == FAULTS, "catalog fault membership count")
        require([int(row[0]) for row in fault_rows] == list(range(FAULTS)),
                "catalog fault-instance order")

        profile_index = np.empty(FAULTS, dtype=np.int32)
        members_by_profile: list[list[int]] = [[] for _ in range(profile_count)]
        for fault_instance, profile_id in fault_rows:
            fault_instance = int(fault_instance)
            profile_id = int(profile_id)
            require(0 <= profile_id < profile_count, "fault profile reference range")
            profile_index[fault_instance] = profile_id
            members_by_profile[profile_id].append(fault_instance)
    finally:
        connection.close()

    normal_matrix = np.frombuffer(normal, dtype=np.uint8).reshape(VECTORS, TOKEN_BYTES)
    normal_cycles = ((normal_matrix[:, 1].astype(np.uint16) << 8) |
                     normal_matrix[:, 2].astype(np.uint16))
    normal_digest = normal_matrix[:, 3:]

    profile_count = len(profile_rows)
    timeout = np.empty((profile_count, VECTORS), dtype=np.uint8)
    latency_delta = np.empty((profile_count, VECTORS), dtype=np.int16)
    digest_xor = np.empty((profile_count, VECTORS, 32), dtype=np.uint8)
    digest_hamming = np.empty((profile_count, VECTORS), dtype=np.uint16)
    deviation = np.empty((profile_count, VECTORS), dtype=np.uint8)
    profile_observable = np.empty(profile_count, dtype=np.uint8)
    profile_candidate_sites = np.empty(profile_count, dtype=np.int32)
    profile_partition_mask = np.zeros(profile_count, dtype=np.uint8)
    profile_member_count = np.empty(profile_count, dtype=np.int32)

    bit_count = np.asarray([int(value).bit_count() for value in range(256)], dtype=np.uint8)
    normal_profile_index = None

    print("\nCONSTRUCTING FACTORIZED RESPONSE FEATURES", flush=True)
    for index, (profile_id, response) in enumerate(profile_rows):
        response = bytes(response)
        require(len(response) == PROFILE_BYTES, f"profile width: {profile_id}")
        matrix = np.frombuffer(response, dtype=np.uint8).reshape(VECTORS, TOKEN_BYTES)
        current_timeout = matrix[:, 0]
        require(np.all(current_timeout <= 1), f"profile timeout encoding: {profile_id}")
        cycles = ((matrix[:, 1].astype(np.uint16) << 8) | matrix[:, 2].astype(np.uint16))
        delta32 = cycles.astype(np.int32) - normal_cycles.astype(np.int32)
        require(np.all((-32768 <= delta32) & (delta32 <= 32767)), "latency delta range")
        current_xor = np.bitwise_xor(matrix[:, 3:], normal_digest)
        current_hamming = bit_count[current_xor].sum(axis=1, dtype=np.uint16)
        current_deviation = (
            (current_timeout != 0) |
            (cycles != normal_cycles) |
            np.any(current_xor != 0, axis=1)
        ).astype(np.uint8)

        timeout[index] = current_timeout
        latency_delta[index] = delta32.astype(np.int16)
        digest_xor[index] = current_xor
        digest_hamming[index] = current_hamming
        deviation[index] = current_deviation

        observable = int(response != normal)
        require(observable == int(np.any(current_deviation)), f"profile anomaly semantics: {profile_id}")
        profile_observable[index] = observable
        if not observable:
            require(normal_profile_index is None, "multiple exact normal profiles")
            normal_profile_index = index

        members = members_by_profile[index]
        require(members, f"profile without fault members: {profile_id}")
        sites = {member // 2 + 1 for member in members}
        profile_member_count[index] = len(members)
        profile_candidate_sites[index] = len(sites)
        mask = 0
        for site in sites:
            mask |= 1 << int(site_partitions[site])
        profile_partition_mask[index] = mask

        if (index + 1) % 2500 == 0:
            print(f"  Processed profiles: {index + 1:,}/{profile_count:,}", flush=True)

    require(normal_profile_index is not None, "normal-compatible profile absent")

    fault_instance_index = np.arange(FAULTS, dtype=np.int32)
    site_index = fault_instance_index // 2 + 1
    stuck_value = (fault_instance_index % 2).astype(np.uint8)
    partition_code = site_partitions[site_index].astype(np.uint8)
    observable = profile_observable[profile_index]
    candidate_site_count = profile_candidate_sites[profile_index]
    observability_class = np.where(
        observable == 0,
        0,
        np.where(candidate_site_count == 1, 1, 2),
    ).astype(np.uint8)
    cross_partition_profile = (
        (profile_partition_mask[profile_index] & (profile_partition_mask[profile_index] - 1)) != 0
    ).astype(np.uint8)

    require(int(np.sum(observable)) == 22930, "observable fault count")
    require(int(np.sum(observable == 0)) == 22748, "normal-compatible fault count")
    require(int(np.sum(observability_class == 1)) == 9171, "unique-site fault count")
    require(int(np.sum(observability_class == 2)) == 13759, "ambiguous-site fault count")
    require(np.all(partition_code[0::2] == partition_code[1::2]), "SA0/SA1 site separation")

    features = {
        "profile_deviation": deviation,
        "profile_digest_hamming": digest_hamming,
        "profile_digest_xor": digest_xor,
        "profile_latency_delta": latency_delta,
        "profile_timeout": timeout,
        "vector_id": vector_ids,
    }
    targets = {
        "candidate_site_count": candidate_site_count.astype(np.int32),
        "cross_partition_profile": cross_partition_profile,
        "fault_instance_index": fault_instance_index,
        "observable": observable.astype(np.uint8),
        "observability_class": observability_class,
        "partition_code": partition_code,
        "profile_index": profile_index.astype(np.int32),
        "site_index": site_index.astype(np.int32),
        "stuck_value": stuck_value,
    }
    statistics = {
        "distinct_response_profiles": profile_count,
        "normal_profile_index": int(normal_profile_index),
        "normal_profile_fault_members": int(profile_member_count[normal_profile_index]),
        "observable_profiles": int(np.sum(profile_observable)),
        "normal_compatible_profiles": int(np.sum(profile_observable == 0)),
        "cross_partition_profiles": int(np.sum(
            (profile_partition_mask & (profile_partition_mask - 1)) != 0
        )),
        "fault_instances_with_cross_partition_profile": int(np.sum(cross_partition_profile)),
        "maximum_candidate_site_count": int(np.max(profile_candidate_sites)),
    }
    return features, targets, statistics


def describe_arrays(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    return {
        name: {"dtype": str(array.dtype), "shape": list(array.shape)}
        for name, array in sorted(arrays.items())
    }


def main() -> None:
    print("STAGE 12A-1B — BLINDED TRAIN-RESPONSE DATASET CONSTRUCTION")
    verified_inputs = verify_pinned_inputs()
    _, baseline_hashes = verify_baseline_bundle()
    normal, vector_ids = read_normal_profile()
    site_partitions, partition_counts = read_site_partitions()
    features, targets, statistics = construct_arrays(normal, vector_ids, site_partitions)

    print("\nWRITING DETERMINISTIC FACTORIZED NPZ ARTIFACTS", flush=True)
    feature_bytes = deterministic_npz(features)
    target_bytes = deterministic_npz(targets)
    require(feature_bytes == deterministic_npz(features), "feature NPZ deterministic replay")
    require(target_bytes == deterministic_npz(targets), "target NPZ deterministic replay")
    write_frozen(FEATURES_PATH, feature_bytes)
    write_frozen(TARGETS_PATH, target_bytes)

    schema = {
        "schema_version": "CIRCUITSAGE-HMAC-V2-TRAIN-RESPONSE-SCHEMA-v1",
        "stage": STAGE,
        "status": "FROZEN",
        "storage": "FACTORIZED DETERMINISTIC NPZ",
        "observation_contract": {
            "vectors": VECTORS,
            "token_bytes_per_vector": TOKEN_BYTES,
            "raw_fields": ["timed_out", "cycles", "actual_digest"],
            "golden_reference_operation": "observed response minus/XOR frozen normal response",
        },
        "model_input_file": str(FEATURES_PATH.relative_to(ROOT)),
        "model_input_arrays": describe_arrays(features),
        "target_file": str(TARGETS_PATH.relative_to(ROOT)),
        "target_arrays": describe_arrays(targets),
        "flattened_response_feature_count": 2304,
        "feature_components": {
            "timeout": 64,
            "latency_delta": 64,
            "digest_xor_bytes": 2048,
            "digest_hamming_distance": 64,
            "deviation_indicator": 64,
        },
        "join_rule": "targets.profile_index selects one response profile; profile_index is metadata, not a model feature",
        "partition_codes": {"0": "DEV_TRAIN", "1": "DEV_CALIBRATION", "2": "DEV_SITE_TEST"},
        "observability_class_codes": {"0": "NORMAL_COMPATIBLE", "1": "UNIQUE_SITE", "2": "AMBIGUOUS_SITES"},
        "forbidden_model_inputs": [
            "fault_instance_index", "site_index", "stuck_value", "profile_index",
            "candidate_site_count", "observability_class", "partition_code",
            "activity", "detected", "fault_enable", "selector", "batch_id",
        ],
        "localization_policy": {
            "candidate labels remain in the separate target artifact": True,
            "cross_partition_profile must be reported as a breakdown": True,
            "catalog membership may not be consulted during locked inference": True,
        },
    }
    write_frozen(SCHEMA_PATH, canonical_json(schema))

    artifacts = {
        str(FEATURES_PATH.relative_to(ROOT)): {
            "sha256": sha256_file(FEATURES_PATH), "bytes": FEATURES_PATH.stat().st_size,
        },
        str(TARGETS_PATH.relative_to(ROOT)): {
            "sha256": sha256_file(TARGETS_PATH), "bytes": TARGETS_PATH.stat().st_size,
        },
        str(SCHEMA_PATH.relative_to(ROOT)): {
            "sha256": sha256_file(SCHEMA_PATH), "bytes": SCHEMA_PATH.stat().st_size,
        },
    }
    manifest = {
        "manifest_version": "CIRCUITSAGE-HMAC-V2-TRAIN-RESPONSE-MANIFEST-v1",
        "stage": STAGE,
        "status": "PASS",
        "dataset_status": "FROZEN",
        "schema_status": "FROZEN",
        "source_partition": "TRAIN ONLY",
        "physical_sites": SITES,
        "fault_instances": FAULTS,
        "response_vectors": VECTORS,
        "observable_fault_instances": 22930,
        "normal_compatible_fault_instances": 22748,
        "unique_site_fault_instances": 9171,
        "ambiguous_site_fault_instances": 13759,
        "partition_site_counts": partition_counts,
        "statistics": statistics,
        "artifacts": artifacts,
        "input_evidence": {**verified_inputs, **baseline_hashes},
        "privacy": {
            "features_contain_fault_identity": False,
            "targets_contain_private_development_labels": True,
            "public_release_authorized": False,
        },
    }
    write_frozen(MANIFEST_PATH, canonical_json(manifest))

    audit = {
        "audit_version": "CIRCUITSAGE-HMAC-V2-TRAIN-RESPONSE-FREEZE-v1",
        "stage": STAGE,
        "status": "PASS",
        "dataset_status": "FROZEN",
        "schema_status": "FROZEN",
        "exact_npz_replay": "PASS",
        "unknown_fault_identity_in_features": "NO",
        "fault_and_site_targets_separate": "YES",
        "sa0_sa1_site_separations": 0,
        "model_training_performed": False,
        "model_objects_deserialized": 0,
        "simulation_performed": False,
        "validation_access_count": 0,
        "holdout_access_count": 0,
        "v1_model_modified": False,
        "frozen_inputs_modified": False,
        "manifest": {
            "path": str(MANIFEST_PATH.relative_to(ROOT)),
            "sha256": sha256_file(MANIFEST_PATH),
        },
        "next_gate": "STAGE 12A-1C — V2 DETECTOR/LOCATOR ARCHITECTURE AND TRAINING-CONTRACT FREEZE",
    }
    write_frozen(AUDIT_PATH, canonical_json(audit))

    print("\nSTAGE 12A-1B — BLINDED TRAIN-RESPONSE DATASET AND SCHEMA FREEZE")
    print(f"{'Status':<35}: PASS")
    print(f"{'Dataset status':<35}: FROZEN")
    print(f"{'Schema status':<35}: FROZEN")
    print(f"{'Source partition':<35}: TRAIN ONLY")
    print(f"{'Response profiles':<35}: {statistics['distinct_response_profiles']}")
    print(f"{'Fault instances':<35}: {FAULTS}")
    print(f"{'Observable / normal-compatible':<35}: 22930 / 22748")
    print(f"{'Unique / ambiguous site':<35}: 9171 / 13759")
    print(f"{'Response feature dimensions':<35}: 2304")
    print(f"{'Unknown identity in features':<35}: NO")
    print(f"{'Fault/site targets separate':<35}: YES")
    print(f"{'SA0/SA1 site separations':<35}: 0")
    print(f"{'Cross-partition profiles':<35}: {statistics['cross_partition_profiles']} (REPORT BREAKDOWN REQUIRED)")
    print(f"{'Exact NPZ replay':<35}: PASS")
    print(f"{'Model training':<35}: NOT PERFORMED")
    print(f"{'Validation / HOLDOUT access':<35}: 0 / 0")
    print(f"{'Features':<35}: {FEATURES_PATH}")
    print(f"{'Features SHA':<35}: {sha256_file(FEATURES_PATH)}")
    print(f"{'Targets':<35}: {TARGETS_PATH}")
    print(f"{'Targets SHA':<35}: {sha256_file(TARGETS_PATH)}")
    print(f"{'Schema':<35}: {SCHEMA_PATH}")
    print(f"{'Schema SHA':<35}: {sha256_file(SCHEMA_PATH)}")
    print(f"{'Manifest':<35}: {MANIFEST_PATH}")
    print(f"{'Manifest SHA':<35}: {sha256_file(MANIFEST_PATH)}")
    print(f"{'Audit':<35}: {AUDIT_PATH}")
    print(f"{'Audit SHA':<35}: {sha256_file(AUDIT_PATH)}")
    print(f"{'Next gate':<35}: STAGE 12A-1C — V2 DETECTOR/LOCATOR ARCHITECTURE AND TRAINING-CONTRACT FREEZE")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nSTOP: interrupted; frozen inputs were not modified", file=sys.stderr)
        raise SystemExit(130)
    except Exception as error:
        print(f"STOP: {error}", file=sys.stderr)
        raise SystemExit(1)
