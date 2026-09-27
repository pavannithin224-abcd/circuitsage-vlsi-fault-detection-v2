#!/usr/bin/env python3
"""Stage 12B-1B: behavior-signature index and ambiguity-aware dataset freeze.

Builds a deterministic closed-catalog response-signature index and set-valued
localization targets from the already-frozen TRAIN-vector response dataset.
No model is loaded, trained, or evaluated.  DEV_SITE_TEST query rows are not
emitted; VALIDATION and HOLDOUT are not accessed.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import sys
import zipfile
from pathlib import Path
from typing import Any

try:
    import numpy as np
except ImportError as error:
    raise SystemExit("STOP: activate the project .venv (NumPy required)") from error


STAGE = "12B-1B"
ROOT = Path(__file__).resolve().parent
V2_RESULT = ROOT / "results/circuitsage_hmac_v2_12a1"
RESULT = ROOT / "results/circuitsage_hmac_v2_12b1"
CONFIG = ROOT / "config/v2_1"

CONTRACT_SOURCE = ROOT / "stage_12b1a_v2_1_locator_redesign_contract.py"
ARCHITECTURE = CONFIG / "circuitsage_hmac_v2_1_locator_architecture_12b1a.json"
TRAINING_CONTRACT = CONFIG / "circuitsage_hmac_v2_1_locator_training_contract_12b1a.json"
ACCEPTANCE = CONFIG / "circuitsage_hmac_v2_1_locator_acceptance_contract_12b1a.json"
CANDIDATE_GRID = CONFIG / "circuitsage_hmac_v2_1_locator_candidate_grid_12b1a.csv"
CONTRACT_ENVIRONMENT = RESULT / "circuitsage_hmac_v2_1_environment_12b1a.json"
CONTRACT_MANIFEST = RESULT / "circuitsage_hmac_v2_1_contract_manifest_12b1a.json"
CONTRACT_AUDIT = RESULT / "circuitsage_hmac_v2_1_locator_redesign_contract_freeze_12b1a.json"

SOURCE_SCRIPT = ROOT / "stage_12a1b_blinded_train_response_dataset.py"
SOURCE_DIR = V2_RESULT / "blinded_train_response_dataset_12a1b"
SOURCE_FEATURES = SOURCE_DIR / "circuitsage_hmac_v2_train_response_features_12a1b.npz"
SOURCE_TARGETS = SOURCE_DIR / "circuitsage_hmac_v2_train_response_targets_12a1b.npz"
SOURCE_SCHEMA = SOURCE_DIR / "circuitsage_hmac_v2_train_response_schema_12a1b.json"
SOURCE_MANIFEST = V2_RESULT / "circuitsage_hmac_v2_train_response_manifest_12a1b.json"
SOURCE_AUDIT = V2_RESULT / "circuitsage_hmac_v2_train_response_dataset_freeze_12a1b.json"

WORK = RESULT / "behavior_signature_dataset_12b1b"
SIGNATURE_INDEX = WORK / "circuitsage_hmac_v2_1_behavior_signature_index_12b1b.npz"
AMBIGUITY_DATASET = WORK / "circuitsage_hmac_v2_1_ambiguity_aware_dataset_12b1b.npz"
SCHEMA = WORK / "circuitsage_hmac_v2_1_behavior_signature_schema_12b1b.json"
STATISTICS = WORK / "circuitsage_hmac_v2_1_behavior_signature_statistics_12b1b.json"
MANIFEST = RESULT / "circuitsage_hmac_v2_1_behavior_signature_manifest_12b1b.json"
AUDIT = RESULT / "circuitsage_hmac_v2_1_behavior_signature_dataset_freeze_12b1b.json"

PINNED = {
    CONTRACT_SOURCE: "8a5a211c5adc073bf3d67560d50bd4457f181a0c19ca7e2f136c9f905bb6f979",
    ARCHITECTURE: "370fa36bfa952a1fe54d1b7b7f4e27950692387849d94aa73b733de4c45b02fe",
    TRAINING_CONTRACT: "36d6da41f359dee47d146866601655ed46b39d941ef79dcbd48cb689d3260e67",
    ACCEPTANCE: "899b2ff099da20b4631308962c3052e16c7cf875ae296f5b2f6da825d6c6c12a",
    CANDIDATE_GRID: "c159aa527d6720093fa3578d0d4e0042d1b0c2138535ac01e567efdddd600653",
    CONTRACT_ENVIRONMENT: "6bed97f508754f9c4782a2ee507b78272034ca92d7d9d24d729b57195294c1bc",
    CONTRACT_MANIFEST: "eae8cf0d02155c8fbf3e736307ffca79f13d2268442c10a49a945477d7e62577",
    CONTRACT_AUDIT: "95ccfb93fcc11c5b87b21d0a7fd869a0562c42175ec6aa60310ca20ee272fa01",
    SOURCE_SCRIPT: "d0dd96765a10f9821c284b21f6d757771710b46064016c0a1506fbd97d237032",
    SOURCE_FEATURES: "09f5dded67a6f82d14f02b4de23765b66ff954eb6a9779817c8b7286dc6ac2b3",
    SOURCE_TARGETS: "268e407a9d13adab6a81b4650b4c16761dffb99d6d3d7a32ab2c28ff9a57a572",
    SOURCE_SCHEMA: "5743d02ad5a8067aef331e955b6e91acd599b066956b0e572c7d25b2c1f98032",
    SOURCE_MANIFEST: "dd667b287285e0fb6cca6930de8415bc97b2a8b4213840205aa7b4b842dbdbca",
    SOURCE_AUDIT: "e4b5e6f3aa60c2f49668ed5105549898729af5985e5e8db7d8eabc1e86aaf222",
}

SITES = 22839
FAULTS = 45678
VECTORS = 64
PARTITION_NAMES = {0: "DEV_TRAIN", 1: "DEV_CALIBRATION", 2: "DEV_SITE_TEST"}
SIGNATURE_DOMAIN = b"CIRCUITSAGE-HMAC-V2.1-RESPONSE-SIGNATURE-v1\0"


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


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON object required: {rel(path)}")
    return value


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode()


def frozen_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.exists(), f"refusing to overwrite frozen output: {rel(path)}")
    temporary = path.with_name(path.name + ".tmp")
    require(not temporary.exists(), f"stale temporary output: {rel(temporary)}")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def record(path: Path) -> dict[str, Any]:
    return {"path": rel(path), "sha256": sha256(path), "bytes": path.stat().st_size}


def npy_bytes(array: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.lib.format.write_array(stream, np.ascontiguousarray(array), version=(2, 0), allow_pickle=False)
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
            archive.writestr(info, npy_bytes(arrays[name]), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    return stream.getvalue()


def describe(arrays: dict[str, np.ndarray]) -> dict[str, dict[str, Any]]:
    return {
        name: {"dtype": str(array.dtype), "shape": list(array.shape)}
        for name, array in sorted(arrays.items())
    }


def resolve_record_path(item: dict[str, Any]) -> Path:
    value = item.get("path")
    require(isinstance(value, str) and value, "artifact path record")
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def verify_record(item: dict[str, Any], label: str) -> None:
    path = resolve_record_path(item)
    require(path.is_file(), f"missing {label}: {rel(path)}")
    require(sha256(path) == item.get("sha256"), f"{label} changed: {path.name}")
    if "bytes" in item:
        require(path.stat().st_size == int(item["bytes"]), f"{label} size changed: {path.name}")


def verify_inputs() -> dict[str, Any]:
    print("STAGE 12B-1B — BEHAVIOR-SIGNATURE INDEX AND AMBIGUITY-AWARE DATASET")
    print("FROZEN INPUT VERIFICATION")
    evidence: dict[str, Any] = {}
    for path, expected in PINNED.items():
        require(path.is_file(), f"missing frozen input: {rel(path)}")
        require(sha256(path) == expected, f"frozen input SHA: {rel(path)}")
        evidence[rel(path)] = record(path)
        print(f"  {path.name:<82}: OK", flush=True)

    contract_audit = load_json(CONTRACT_AUDIT)
    contract_manifest = load_json(CONTRACT_MANIFEST)
    training = load_json(TRAINING_CONTRACT)
    source_manifest = load_json(SOURCE_MANIFEST)
    source_audit = load_json(SOURCE_AUDIT)

    require(contract_audit.get("status") == "PASS", "12B-1A audit status")
    require(contract_audit.get("architecture_status") == "FROZEN", "12B-1A architecture freeze")
    require(contract_audit.get("training_contract_status") == "FROZEN", "12B-1A training-contract freeze")
    require(contract_audit.get("model_training") == "NOT STARTED / NOT AUTHORIZED BY THIS STAGE", "12B-1A training state")
    require(contract_audit.get("dev_site_test") == "LOCKED / NOT OPENED / NOT AUTHORIZED", "12B-1A site-test state")
    require(contract_audit.get("validation_access_count") == 0, "12B-1A VALIDATION access")
    require(contract_audit.get("holdout_access_count") == 0, "12B-1A HOLDOUT access")
    require(training.get("authorizations", {}).get("next_stage_signature_dataset_construction") == "AUTHORIZED FROM DEV_TRAIN ONLY", "12B-1A dataset authorization")
    require(training.get("authorizations", {}).get("model_training") == "NOT YET AUTHORIZED", "12B-1A model authorization")
    require(source_manifest.get("status") == "PASS", "source manifest status")
    require(source_manifest.get("source_partition") == "TRAIN ONLY", "source vector partition")
    require(source_manifest.get("fault_instances") == FAULTS, "source fault count")
    require(source_manifest.get("physical_sites") == SITES, "source site count")
    require(source_manifest.get("response_vectors") == VECTORS, "source vector count")
    require(source_audit.get("status") == "PASS", "source audit status")

    for section in ("input_evidence", "outputs"):
        entries = contract_manifest.get(section)
        require(isinstance(entries, dict), f"12B-1A manifest {section}")
        for item in entries.values():
            if isinstance(item, dict) and isinstance(item.get("path"), str):
                verify_record(item, f"12B-1A manifest {section}")
    print("  Contract lineage, dataset authorization and source semantics                  : PASS")
    return evidence


def load_source() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    with np.load(SOURCE_FEATURES, allow_pickle=False) as archive:
        required = {
            "profile_timeout", "profile_latency_delta", "profile_digest_xor",
            "profile_digest_hamming", "profile_deviation", "vector_id",
        }
        require(required.issubset(archive.files), "source response members")
        features = {name: np.asarray(archive[name]).copy() for name in required}
    profiles = len(features["profile_timeout"])
    require(features["profile_timeout"].shape == (profiles, VECTORS), "timeout shape")
    require(features["profile_latency_delta"].shape == (profiles, VECTORS), "latency shape")
    require(features["profile_digest_xor"].shape == (profiles, VECTORS, 32), "digest XOR shape")
    require(features["profile_digest_hamming"].shape == (profiles, VECTORS), "hamming shape")
    require(features["profile_deviation"].shape == (profiles, VECTORS), "deviation shape")
    require(features["vector_id"].shape == (VECTORS,), "vector ID shape")

    with np.load(SOURCE_TARGETS, allow_pickle=False) as archive:
        required = {
            "candidate_site_count", "cross_partition_profile", "fault_instance_index",
            "observable", "observability_class", "partition_code", "profile_index",
            "site_index", "stuck_value",
        }
        require(required.issubset(archive.files), "source target members")
        targets = {name: np.asarray(archive[name]).copy() for name in required}
    require(all(len(value) == FAULTS for value in targets.values()), "target array lengths")
    require(np.array_equal(targets["fault_instance_index"], np.arange(FAULTS)), "fault ordering")
    require(np.array_equal(targets["site_index"], np.arange(FAULTS) // 2 + 1), "site mapping")
    require(np.array_equal(targets["stuck_value"], np.arange(FAULTS) % 2), "SA mapping")
    require(set(np.unique(targets["partition_code"]).tolist()) == {0, 1, 2}, "partition codes")
    require(int(np.sum(targets["observable"])) == 22930, "observable fault count")
    require(int(np.sum(targets["observability_class"] == 1)) == 9171, "unique-site fault count")
    require(int(np.sum(targets["observability_class"] == 2)) == 13759, "ambiguous-site fault count")
    require(int(np.min(targets["profile_index"])) == 0, "profile index minimum")
    require(int(np.max(targets["profile_index"])) == profiles - 1, "profile index maximum")
    return features, targets


def signature_digest(vector_commitment: bytes, features: dict[str, np.ndarray], profile: int) -> bytes:
    digest = hashlib.sha256()
    digest.update(SIGNATURE_DOMAIN)
    digest.update(vector_commitment)
    fields = (
        (b"timeout\0", np.asarray(features["profile_timeout"][profile], dtype=np.uint8)),
        (b"latency_delta_i16le\0", np.asarray(features["profile_latency_delta"][profile], dtype="<i2")),
        (b"digest_xor\0", np.asarray(features["profile_digest_xor"][profile], dtype=np.uint8)),
    )
    for label, array in fields:
        digest.update(label)
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.digest()


def offsets_from_counts(counts: np.ndarray) -> np.ndarray:
    offsets = np.zeros(len(counts) + 1, dtype=np.uint32)
    offsets[1:] = np.cumsum(counts, dtype=np.uint64).astype(np.uint32)
    return offsets


def build(features: dict[str, np.ndarray], targets: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    profiles = len(features["profile_timeout"])
    vector_ids = np.asarray(features["vector_id"], dtype="<u8")
    vector_commitment = hashlib.sha256(b"CIRCUITSAGE-HMAC-V2.1-VECTOR-ORDER-v1\0" + vector_ids.tobytes()).digest()

    print("\nBUILDING CANONICAL RESPONSE-SIGNATURE INDEX", flush=True)
    signatures = np.empty((profiles, 32), dtype=np.uint8)
    for profile in range(profiles):
        signatures[profile] = np.frombuffer(signature_digest(vector_commitment, features, profile), dtype=np.uint8)
        if (profile + 1) % 2500 == 0 or profile + 1 == profiles:
            print(f"  signatures {profile + 1:,}/{profiles:,}", flush=True)
    signature_keys = signatures.view(np.dtype((np.void, 32))).reshape(-1)
    require(len(np.unique(signature_keys)) == profiles, "signature collision or non-distinct source profiles")
    signature_order = np.argsort(signature_keys, kind="stable").astype(np.uint32)

    profile_index = targets["profile_index"].astype(np.int64)
    fault_order = np.argsort(profile_index, kind="stable").astype(np.uint32)
    fault_counts = np.bincount(profile_index, minlength=profiles).astype(np.uint32)
    require(np.all(fault_counts > 0), "profile without fault member")
    fault_offsets = offsets_from_counts(fault_counts)
    require(int(fault_offsets[-1]) == FAULTS, "profile fault offset closure")

    site_chunks: list[np.ndarray] = []
    profile_site_counts = np.empty(profiles, dtype=np.uint32)
    profile_observable = np.empty(profiles, dtype=np.uint8)
    profile_class = np.empty(profiles, dtype=np.uint8)
    profile_pure_partition = np.full(profiles, 255, dtype=np.uint8)
    cross_partition_profiles = 0
    for profile in range(profiles):
        members = fault_order[fault_offsets[profile]:fault_offsets[profile + 1]].astype(np.int64)
        sites = np.unique(targets["site_index"][members]).astype(np.uint32)
        partitions = np.unique(targets["partition_code"][members]).astype(np.uint8)
        observable_values = np.unique(targets["observable"][members])
        class_values = np.unique(targets["observability_class"][members])
        candidate_counts = np.unique(targets["candidate_site_count"][members])
        cross_values = np.unique(targets["cross_partition_profile"][members])
        require(len(observable_values) == len(class_values) == len(candidate_counts) == len(cross_values) == 1, "profile target consistency")
        require(int(candidate_counts[0]) == len(sites), "profile candidate-site count")
        require(int(cross_values[0]) == int(len(partitions) > 1), "cross-partition flag")
        profile_site_counts[profile] = len(sites)
        profile_observable[profile] = int(observable_values[0])
        profile_class[profile] = int(class_values[0])
        if len(partitions) == 1:
            profile_pure_partition[profile] = int(partitions[0])
        else:
            cross_partition_profiles += 1
        site_chunks.append(sites)
    profile_site_offsets = offsets_from_counts(profile_site_counts)
    profile_site_indices = np.concatenate(site_chunks).astype(np.uint32)
    require(int(profile_site_offsets[-1]) == len(profile_site_indices), "profile site offset closure")

    query_profiles = np.flatnonzero(
        (profile_observable == 1) &
        ((profile_pure_partition == 0) | (profile_pure_partition == 1))
    ).astype(np.uint32)
    query_partitions = profile_pure_partition[query_profiles].astype(np.uint8)
    require(not np.any(query_partitions == 2), "DEV_SITE_TEST query emitted")

    query_fault_chunks: list[np.ndarray] = []
    query_site_chunks: list[np.ndarray] = []
    query_fault_counts = np.empty(len(query_profiles), dtype=np.uint32)
    query_site_counts = np.empty(len(query_profiles), dtype=np.uint32)
    for row, profile_u32 in enumerate(query_profiles):
        profile = int(profile_u32)
        faults = fault_order[fault_offsets[profile]:fault_offsets[profile + 1]].astype(np.uint32)
        sites = profile_site_indices[profile_site_offsets[profile]:profile_site_offsets[profile + 1]].astype(np.uint32)
        require(np.all(targets["partition_code"][faults] == query_partitions[row]), "query partition purity")
        query_fault_counts[row] = len(faults)
        query_site_counts[row] = len(sites)
        query_fault_chunks.append(faults)
        query_site_chunks.append(sites)
    query_fault_offsets = offsets_from_counts(query_fault_counts)
    query_site_offsets = offsets_from_counts(query_site_counts)
    query_fault_indices = np.concatenate(query_fault_chunks).astype(np.uint32) if query_fault_chunks else np.empty(0, dtype=np.uint32)
    query_site_indices = np.concatenate(query_site_chunks).astype(np.uint32) if query_site_chunks else np.empty(0, dtype=np.uint32)

    require(np.all(profile_class[profile_observable == 0] == 0), "normal-compatible class semantics")
    require(np.all(profile_class[(profile_observable == 1) & (profile_site_counts == 1)] == 1), "unique-site class semantics")
    require(np.all(profile_class[(profile_observable == 1) & (profile_site_counts > 1)] == 2), "ambiguous-site class semantics")

    index = {
        "fault_instance_index_by_profile": fault_order.astype("<u4"),
        "profile_candidate_fault_count": fault_counts.astype("<u4"),
        "profile_candidate_site_count": profile_site_counts.astype("<u4"),
        "profile_deviation": features["profile_deviation"].astype(np.uint8),
        "profile_digest_hamming": features["profile_digest_hamming"].astype("<u2"),
        "profile_fault_offset": fault_offsets.astype("<u4"),
        "profile_index": np.arange(profiles, dtype="<u4"),
        "profile_latency_delta": features["profile_latency_delta"].astype("<i2"),
        "profile_observability_class": profile_class.astype(np.uint8),
        "profile_observable": profile_observable.astype(np.uint8),
        "profile_pure_partition_code": profile_pure_partition.astype(np.uint8),
        "profile_signature_sha256": signatures.astype(np.uint8),
        "profile_signature_sort_order": signature_order.astype("<u4"),
        "profile_site_index": profile_site_indices.astype("<u4"),
        "profile_site_offset": profile_site_offsets.astype("<u4"),
        "profile_timeout": features["profile_timeout"].astype(np.uint8),
        "vector_commitment_sha256": np.frombuffer(vector_commitment, dtype=np.uint8).copy(),
        "vector_id": vector_ids,
    }
    dataset = {
        "query_candidate_fault_index": query_fault_indices.astype("<u4"),
        "query_candidate_fault_offset": query_fault_offsets.astype("<u4"),
        "query_candidate_site_index": query_site_indices.astype("<u4"),
        "query_candidate_site_offset": query_site_offsets.astype("<u4"),
        "query_candidate_site_count": query_site_counts.astype("<u4"),
        "query_observability_class": profile_class[query_profiles].astype(np.uint8),
        "query_partition_code": query_partitions.astype(np.uint8),
        "query_profile_index": query_profiles.astype("<u4"),
        "query_signature_sha256": signatures[query_profiles].astype(np.uint8),
    }

    stats = {
        "distinct_response_profiles": profiles,
        "signature_hashes": profiles,
        "signature_hash_collisions": 0,
        "catalog_fault_instances": FAULTS,
        "catalog_physical_sites": SITES,
        "catalog_profile_site_memberships": len(profile_site_indices),
        "observable_fault_instances": int(np.sum(targets["observable"])),
        "normal_compatible_fault_instances": int(np.sum(targets["observable"] == 0)),
        "unique_site_fault_instances": int(np.sum(targets["observability_class"] == 1)),
        "ambiguous_site_fault_instances": int(np.sum(targets["observability_class"] == 2)),
        "cross_partition_profiles_excluded_from_queries": cross_partition_profiles,
        "train_query_profiles": int(np.sum(query_partitions == 0)),
        "calibration_query_profiles": int(np.sum(query_partitions == 1)),
        "dev_site_test_query_profiles": 0,
        "query_profiles_total": len(query_profiles),
        "query_unique_site_profiles": int(np.sum(profile_class[query_profiles] == 1)),
        "query_ambiguous_site_profiles": int(np.sum(profile_class[query_profiles] == 2)),
        "maximum_catalog_candidate_sites": int(np.max(profile_site_counts)),
        "vector_commitment_sha256": vector_commitment.hex(),
    }
    return index, dataset, stats


def self_test() -> None:
    features = {
        "profile_timeout": np.zeros((3, VECTORS), dtype=np.uint8),
        "profile_latency_delta": np.zeros((3, VECTORS), dtype=np.int16),
        "profile_digest_xor": np.zeros((3, VECTORS, 32), dtype=np.uint8),
    }
    features["profile_digest_xor"][1, 0, 0] = 1
    features["profile_digest_xor"][2, 0, 0] = 2
    commitment = hashlib.sha256(b"vectors").digest()
    hashes = [signature_digest(commitment, features, index) for index in range(3)]
    require(len(set(hashes)) == 3, "signature uniqueness canary")
    counts = np.asarray([2, 0, 3], dtype=np.uint32)
    require(np.array_equal(offsets_from_counts(counts), np.asarray([0, 2, 2, 5], dtype=np.uint32)), "offset canary")
    arrays = {"b": np.arange(3, dtype=np.uint8), "a": np.arange(2, dtype=np.uint32)}
    require(deterministic_npz(arrays) == deterministic_npz(arrays), "NPZ replay canary")
    require(canonical_json({"b": 2, "a": 1}) == canonical_json({"a": 1, "b": 2}), "JSON replay canary")
    print("Stage 12B-1B self-test: PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return

    require(ROOT.name == "vlsi_fault_detection_v2", f"run from project root, not {ROOT}")
    for path in (SIGNATURE_INDEX, AMBIGUITY_DATASET, SCHEMA, STATISTICS, MANIFEST, AUDIT):
        require(not path.exists(), f"Stage {STAGE} output already exists: {rel(path)}")

    evidence = verify_inputs()
    features, targets = load_source()
    index, dataset, stats = build(features, targets)

    print("\nWRITING DETERMINISTIC DATASET ARTIFACTS", flush=True)
    index_bytes = deterministic_npz(index)
    dataset_bytes = deterministic_npz(dataset)
    require(index_bytes == deterministic_npz(index), "signature-index NPZ replay")
    require(dataset_bytes == deterministic_npz(dataset), "ambiguity-dataset NPZ replay")
    frozen_write(SIGNATURE_INDEX, index_bytes)
    frozen_write(AMBIGUITY_DATASET, dataset_bytes)

    schema = {
        "schema_version": "CIRCUITSAGE-HMAC-V2.1-BEHAVIOR-SIGNATURE-SCHEMA-v1",
        "stage": STAGE,
        "status": "FROZEN",
        "storage": "DETERMINISTIC NPZ",
        "signature_algorithm": {
            "hash": "SHA-256",
            "domain_separator": SIGNATURE_DOMAIN.decode("ascii").rstrip("\0"),
            "vector_commitment": "SHA-256(domain || ordered uint64-little-endian vector IDs)",
            "profile_fields_in_order": [
                "profile_timeout as uint8[64]",
                "profile_latency_delta as int16-little-endian[64]",
                "profile_digest_xor as uint8[64,32]",
            ],
            "collision_policy": "STOP; never merge profiles solely on a colliding digest",
        },
        "signature_index_file": rel(SIGNATURE_INDEX),
        "signature_index_arrays": describe(index),
        "ambiguity_dataset_file": rel(AMBIGUITY_DATASET),
        "ambiguity_dataset_arrays": describe(dataset),
        "query_join_rule": "query_profile_index selects response features; it is metadata and not a model feature",
        "candidate_set_rule": "offset[i]:offset[i+1] gives every observationally equivalent catalog fault/site",
        "partition_codes": {str(key): value for key, value in PARTITION_NAMES.items()},
        "partition_policy": {
            "DEV_TRAIN": "query rows emitted; fitting allowed only after next authorization",
            "DEV_CALIBRATION": "query rows emitted; selection-only labels",
            "DEV_SITE_TEST": "zero query rows emitted",
            "cross_partition_profiles": "excluded from fitting and selection queries",
            "VALIDATION": "not accessed",
            "HOLDOUT": "not accessed",
        },
        "identity_boundary": {
            "candidate catalog IDs": "allowed only as retrieval output and set-valued targets",
            "query identity features": 0,
            "ground-truth fault/site/type in query X": False,
        },
        "claim_boundary": "closed-catalog consistency dataset; not independent-chip or unseen-fault generalization evidence",
    }
    frozen_write(SCHEMA, canonical_json(schema))
    frozen_write(STATISTICS, canonical_json({
        "statistics_version": "CIRCUITSAGE-HMAC-V2.1-BEHAVIOR-SIGNATURE-STATISTICS-v1",
        "stage": STAGE,
        "status": "FROZEN",
        **stats,
    }))

    manifest = {
        "manifest_version": "CIRCUITSAGE-HMAC-V2.1-BEHAVIOR-SIGNATURE-MANIFEST-v1",
        "stage": STAGE,
        "status": "PASS",
        "dataset_status": "FROZEN",
        "schema_status": "FROZEN",
        "input_evidence": evidence,
        "outputs": {
            rel(path): record(path)
            for path in (SIGNATURE_INDEX, AMBIGUITY_DATASET, SCHEMA, STATISTICS)
        },
        "statistics": stats,
        "model_objects_deserialized": 0,
        "model_training_calls": 0,
        "model_inference_calls": 0,
        "dev_site_test_query_rows": 0,
        "validation_access_count": 0,
        "holdout_access_count": 0,
    }
    frozen_write(MANIFEST, canonical_json(manifest))
    audit = {
        "audit_version": "CIRCUITSAGE-HMAC-V2.1-BEHAVIOR-SIGNATURE-DATASET-FREEZE-v1",
        "stage": STAGE,
        "status": "PASS",
        "signature_index_status": "FROZEN",
        "ambiguity_aware_dataset_status": "FROZEN",
        "schema_status": "FROZEN",
        "deterministic_replay": "PASS / EXACT",
        "signature_hash_collisions": 0,
        "catalog_fault_instances": FAULTS,
        "catalog_physical_sites": SITES,
        "train_query_profiles": stats["train_query_profiles"],
        "calibration_query_profiles": stats["calibration_query_profiles"],
        "cross_partition_profiles_excluded": stats["cross_partition_profiles_excluded_from_queries"],
        "dev_site_test": "LOCKED / ZERO QUERY ROWS / NOT OPENED",
        "validation_access_count": 0,
        "holdout_access_count": 0,
        "unknown_identity_in_query_features": False,
        "models_deserialized": 0,
        "training_performed": False,
        "inference_performed": False,
        "v1_modified": False,
        "v2_core_modified": False,
        "signature_index": record(SIGNATURE_INDEX),
        "ambiguity_dataset": record(AMBIGUITY_DATASET),
        "schema": record(SCHEMA),
        "statistics": record(STATISTICS),
        "manifest": record(MANIFEST),
        "next_gate": "STAGE 12B-1C — V2.1 LOCATOR TRAINING AUTHORIZATION AND EXECUTION-CONTRACT FREEZE",
    }
    frozen_write(AUDIT, canonical_json(audit))

    for path in (SCHEMA, STATISTICS, MANIFEST, AUDIT):
        require(canonical_json(load_json(path)) == path.read_bytes(), f"deterministic JSON replay: {path.name}")
    require(deterministic_npz(index) == SIGNATURE_INDEX.read_bytes(), "frozen signature index replay")
    require(deterministic_npz(dataset) == AMBIGUITY_DATASET.read_bytes(), "frozen ambiguity dataset replay")

    print("\nSTAGE 12B-1B — BEHAVIOR-SIGNATURE INDEX AND AMBIGUITY-AWARE DATASET FREEZE")
    print(f"{'Status':<43}: PASS")
    print(f"{'Signature index status':<43}: FROZEN")
    print(f"{'Ambiguity-aware dataset status':<43}: FROZEN")
    print(f"{'Schema status':<43}: FROZEN")
    print(f"{'Distinct response signatures':<43}: {stats['distinct_response_profiles']}")
    print(f"{'Signature collisions':<43}: 0")
    print(f"{'Catalog fault instances / sites':<43}: {FAULTS} / {SITES}")
    print(f"{'TRAIN query profiles':<43}: {stats['train_query_profiles']}")
    print(f"{'DEV_CALIBRATION query profiles':<43}: {stats['calibration_query_profiles']}")
    print(f"{'Cross-partition profiles excluded':<43}: {stats['cross_partition_profiles_excluded_from_queries']}")
    print(f"{'Unique / ambiguous query profiles':<43}: {stats['query_unique_site_profiles']} / {stats['query_ambiguous_site_profiles']}")
    print(f"{'Maximum candidate sites':<43}: {stats['maximum_catalog_candidate_sites']}")
    print(f"{'Unknown identity in query features':<43}: NO")
    print(f"{'Model training / inference':<43}: NOT PERFORMED / NOT PERFORMED")
    print(f"{'DEV_SITE_TEST query rows':<43}: 0")
    print(f"{'VALIDATION / HOLDOUT access':<43}: 0 / 0")
    print(f"{'V1 / V2 Core modified':<43}: NO / NO")
    print(f"{'Deterministic replay':<43}: PASS / EXACT")
    print(f"{'Signature index':<43}: {SIGNATURE_INDEX}")
    print(f"{'Signature index SHA':<43}: {sha256(SIGNATURE_INDEX)}")
    print(f"{'Ambiguity dataset':<43}: {AMBIGUITY_DATASET}")
    print(f"{'Ambiguity dataset SHA':<43}: {sha256(AMBIGUITY_DATASET)}")
    print(f"{'Schema':<43}: {SCHEMA}")
    print(f"{'Schema SHA':<43}: {sha256(SCHEMA)}")
    print(f"{'Manifest':<43}: {MANIFEST}")
    print(f"{'Manifest SHA':<43}: {sha256(MANIFEST)}")
    print(f"{'Audit':<43}: {AUDIT}")
    print(f"{'Audit SHA':<43}: {sha256(AUDIT)}")
    print(f"{'Next gate':<43}: STAGE 12B-1C — V2.1 LOCATOR TRAINING AUTHORIZATION AND EXECUTION-CONTRACT FREEZE")


if __name__ == "__main__":
    main()
