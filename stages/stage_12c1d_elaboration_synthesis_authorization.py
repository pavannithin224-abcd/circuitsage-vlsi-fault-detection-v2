#!/usr/bin/env python3
"""Stage 12C-1D: multi-circuit elaboration and synthesis authorization freeze.

Verifies the frozen Stage 12C-1C source corpus and creates deterministic
TRAIN/CALIBRATION build recipes, a portable wrapper contract, partition access
rules, and a bounded authorization for the next elaboration/synthesis stage.
No RTL is elaborated or synthesized here. Independent TEST and HOLDOUT remain
locked, and no simulation, fault injection, dataset, training or inference is
performed.
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
import stat
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


STAGE = "12C-1D"
ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "config/v2_2"
RESULT = ROOT / "results/circuitsage_hmac_v2_12c1"
CORPUS = RESULT / "multicircuit_corpus_12c1c"
SOURCE_ROOT = CORPUS / "sources"
ARCHIVE_ROOT = CORPUS / "archives"
WORK = RESULT / "elaboration_synthesis_authorization_12c1d"

SOURCE_1C = ROOT / "stage_12c1c_multicircuit_corpus_acquire.py"
CHECKPOINT_1C = CORPUS / "circuitsage_hmac_v2_2_corpus_acquisition_checkpoint_12c1c.json"
INVENTORY_1C = CORPUS / "circuitsage_hmac_v2_2_corpus_file_inventory_12c1c.csv"
ACQUIRED_CSV_1C = CORPUS / "circuitsage_hmac_v2_2_acquired_family_registry_12c1c.csv"
ACQUIRED_JSON_1C = CORPUS / "circuitsage_hmac_v2_2_acquired_family_registry_12c1c.json"
LICENSE_CSV_1C = CORPUS / "circuitsage_hmac_v2_2_verified_license_registry_12c1c.csv"
LICENSE_JSON_1C = CORPUS / "circuitsage_hmac_v2_2_verified_license_registry_12c1c.json"
INTEGRITY_1C = CORPUS / "circuitsage_hmac_v2_2_corpus_integrity_lock_12c1c.json"
REPORT_1C = CORPUS / "circuitsage_hmac_v2_2_corpus_acquisition_report_12c1c.md"
MANIFEST_1C = RESULT / "circuitsage_hmac_v2_2_corpus_integrity_manifest_12c1c.json"
AUDIT_1C = RESULT / "circuitsage_hmac_v2_2_rtl_acquisition_license_corpus_freeze_12c1c.json"

ARCHIVES_1C = {
    "ibex_cpu": ARCHIVE_ROOT / "ibex_cpu-e9f55342edbd27e9e17a0e41b1c95a81abb5eac8.tar.gz",
    "opentitan_hmac_sha256": ARCHIVE_ROOT / "opentitan_hmac_sha256-83fc48ed3a727399056772d12be8c7d4a8a276f0.tar.gz",
    "picorv32_cpu": ARCHIVE_ROOT / "picorv32_cpu-ef203c2b0a3fb793280f5114941416c425c5b461.tar.gz",
    "secworks_aes": ARCHIVE_ROOT / "secworks_aes-80dc4718e1dcbbdb4b0dd1bdb393d8f7b98981dc.tar.gz",
    "secworks_chacha": ARCHIVE_ROOT / "secworks_chacha-7eaba360df9fed9fc2db98d5f3df81cf01e5b604.tar.gz",
    "secworks_sha256": ARCHIVE_ROOT / "secworks_sha256-837c5cc396f001d18f2c765721c585716eb439ae.tar.gz",
    "serv_cpu": ARCHIVE_ROOT / "serv_cpu-f200eb2ed7b69ac1c6b8eddd47654522aeee5ce8.tar.gz",
}

PINNED = {
    SOURCE_1C: "c5b3d9822ad5ec70e9de5905fcadfeb411fd8d02a31848cd7fa5e64553c8b9a4",
    CHECKPOINT_1C: "0b0af44ffa0846e19e1ab3ff51ee29ba6735933b70d31652b981143b8e54f7b0",
    INVENTORY_1C: "f5925c3fa9c31bdf7120ae5339c3bd93bc03c60267005493bdea8bf7d7cf1170",
    ACQUIRED_CSV_1C: "a06354e5989a1aafdf3db4e277a68a1bab029e4437cd1c6ecc9b227385f6b55c",
    ACQUIRED_JSON_1C: "9551255ae3ea3a2041e70c62e051b6704bb1295de22548df393393da94b691ee",
    LICENSE_CSV_1C: "abd558b2daf3bc0752c9945b30212ed63c323e4c6e4d6fb5be3b13aa978e3bad",
    LICENSE_JSON_1C: "3c41bea7762dd7e5e28bb042420303229a24ee8d916118ba750bd565f0b2fd93",
    INTEGRITY_1C: "a7f91e43ef913cfddb6647951b07043604ad35a96328d4a208e12160956f43cc",
    REPORT_1C: "62c3a27cdc6b3b447e7c611b15d497d9c86a3302678cbc6f60a6802fdf8ccea5",
    MANIFEST_1C: "a1f060e852df19505b28c1a02fbe35328e819ee264fc260727862812685e8bd0",
    AUDIT_1C: "9d6a0a3f52ce3dad2b52cc0f638f9ded2a09b19d4fa191d9cf25aa3f3f38e21a",
    ARCHIVES_1C["ibex_cpu"]: "9e242ea141cdf452ca3e755066641f6d2b12b0940fd641354dafda41d1ebaccf",
    ARCHIVES_1C["opentitan_hmac_sha256"]: "bfd663fb51b76a826f72687070dbc3abed347f708af8304474e5fc8f6823e3a6",
    ARCHIVES_1C["picorv32_cpu"]: "47b07d5160b5d79d5110158a4d1d4b5aabffbbcf33836d86af39b58fb13008aa",
    ARCHIVES_1C["secworks_aes"]: "7362d259009dd2cc028316507921ac3605bb7e038c801776123a50ad427a9f9b",
    ARCHIVES_1C["secworks_chacha"]: "fda07f1b1010023e26d6d0b3e8b9d65520ff36eb18b3521abe1e99fb55afe88a",
    ARCHIVES_1C["secworks_sha256"]: "ad50bede5dfbf2f17da74f00864cc86b25e9b42f8533077cfd01625e877bf3fa",
    ARCHIVES_1C["serv_cpu"]: "0d45ca2b62f3abdcd3cb6ed4862c0d312fff4b5d05fef7a4d240bde10d19a446",
}

WRAPPER_CONTRACT = CONFIG / "circuitsage_hmac_v2_2_portable_wrapper_contract_12c1d.json"
ACCESS_POLICY = CONFIG / "circuitsage_hmac_v2_2_elaboration_partition_access_policy_12c1d.json"
SYNTH_AUTH = CONFIG / "circuitsage_hmac_v2_2_train_calibration_synthesis_authorization_12c1d.json"
RECIPE_CSV = WORK / "circuitsage_hmac_v2_2_elaboration_recipe_registry_12c1d.csv"
RECIPE_JSON = WORK / "circuitsage_hmac_v2_2_elaboration_recipe_registry_12c1d.json"
SOURCE_LISTS = WORK / "source_lists"
WRAPPER_REGISTRY = WORK / "circuitsage_hmac_v2_2_portable_wrapper_registry_12c1d.csv"
ENVIRONMENT = WORK / "circuitsage_hmac_v2_2_elaboration_environment_12c1d.json"
PREFLIGHT = WORK / "circuitsage_hmac_v2_2_elaboration_synthesis_preflight_12c1d.json"
REPORT = WORK / "circuitsage_hmac_v2_2_elaboration_synthesis_authorization_report_12c1d.md"
MANIFEST = RESULT / "circuitsage_hmac_v2_2_elaboration_synthesis_authorization_manifest_12c1d.json"
AUDIT = RESULT / "circuitsage_hmac_v2_2_elaboration_wrapper_synthesis_authorization_freeze_12c1d.json"

EXPECTED_PARTITIONS = {
    "GENERALIZATION_TRAIN": 3,
    "GENERALIZATION_CALIBRATION": 1,
    "INDEPENDENT_CIRCUIT_TEST": 2,
    "GENERALIZATION_HOLDOUT": 1,
}
AUTHORIZED_PARTITIONS = {"GENERALIZATION_TRAIN", "GENERALIZATION_CALIBRATION"}
RTL_SUFFIXES = {".v", ".sv", ".vh", ".svh"}
FUTURE_BRAND = "Faultiva 1.0 — Hybrid VLSI Fault Intelligence"


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
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


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


def record(path: Path) -> dict[str, Any]:
    return {"path": rel(path), "sha256": sha256(path), "bytes": path.stat().st_size}


def csv_bytes(rows: list[dict[str, Any]], fields: list[str]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def iter_files(path: Path) -> Iterable[Path]:
    for item in sorted(path.rglob("*"), key=lambda p: p.relative_to(path).as_posix()):
        if item.is_symlink():
            stop(f"symlink in frozen source tree: {item}")
        if item.is_file():
            yield item


def tree_digest(path: Path) -> str:
    digest = hashlib.sha256()
    for item in iter_files(path):
        relative = item.relative_to(path).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256(item).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(item.stat().st_size).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def tool_version(name: str, arguments: list[str]) -> dict[str, Any]:
    executable = shutil.which(name)
    require(executable is not None, f"required tool not found: {name}")
    result = subprocess.run(
        [executable, *arguments],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
        check=False,
    )
    require(result.returncode == 0, f"tool version check failed: {name}")
    return {"executable": executable, "version": result.stdout.strip().splitlines()[0]}


def verify_manifest_record(item: Any, path: Path) -> None:
    require(isinstance(item, dict), f"missing manifest record: {rel(path)}")
    require(item.get("path") == rel(path), f"manifest path: {rel(path)}")
    require(item.get("sha256") == sha256(path), f"manifest SHA: {rel(path)}")
    require(int(item.get("bytes", -1)) == path.stat().st_size, f"manifest size: {rel(path)}")


def verify_inputs() -> tuple[list[dict[str, Any]], dict[str, list[dict[str, str]]]]:
    print("STAGE 12C-1D — MULTI-CIRCUIT ELABORATION, PORTABLE-WRAPPER, AND SYNTHESIS-AUTHORIZATION FREEZE")
    print("FROZEN INPUT VERIFICATION")
    for path, expected in PINNED.items():
        require(path.is_file(), f"missing frozen input: {rel(path)}")
        require(sha256(path) == expected, f"frozen input SHA: {rel(path)}")
        print(f"  {path.name:<112}: OK")

    manifest = load_json(MANIFEST_1C)
    audit = load_json(AUDIT_1C)
    integrity = load_json(INTEGRITY_1C)
    acquired = load_json(ACQUIRED_JSON_1C)
    licenses = load_json(LICENSE_JSON_1C)
    checkpoint = load_json(CHECKPOINT_1C)

    require(manifest.get("status") == "PASS", "12C-1C manifest status")
    require(manifest.get("families_acquired") == 7, "acquired family count")
    require(manifest.get("licenses_verified") == 7, "verified license count")
    require(manifest.get("partition_counts") == EXPECTED_PARTITIONS, "partition counts")
    require([manifest.get("rtl_elaboration_calls"), manifest.get("synthesis_calls"), manifest.get("simulation_calls"), manifest.get("fault_injection_calls")] == [0, 0, 0, 0], "prior tool activity")
    require([manifest.get("dataset_records_created"), manifest.get("training_calls"), manifest.get("inference_calls")] == [0, 0, 0], "prior ML activity")
    require([manifest.get("independent_test_truth_access"), manifest.get("validation_access"), manifest.get("holdout_truth_access")] == [0, 0, 0], "prior protected access")
    require(audit.get("status") == "PASS", "12C-1C audit status")
    require(audit.get("acquisition_dataset_status") == "COMPLETED / FROZEN", "corpus freeze")
    require(audit.get("holdout") == "SEALED / NOT SEMANTICALLY OPENED", "holdout seal")
    require(audit.get("future_combined_model_brand") == FUTURE_BRAND, "brand reservation")
    require(integrity.get("status") == "FROZEN", "integrity lock")
    require(integrity.get("family_count") == 7, "integrity family count")
    require(checkpoint.get("status") == "PASS / FROZEN", "acquisition checkpoint")
    require(acquired.get("status") == "FROZEN" and len(acquired.get("families", [])) == 7, "acquired registry")
    require(licenses.get("status") == "FROZEN" and len(licenses.get("licenses", [])) == 7, "license registry")

    output_records = manifest.get("outputs")
    require(isinstance(output_records, dict), "12C-1C output records")
    for path in (CHECKPOINT_1C, INVENTORY_1C, ACQUIRED_CSV_1C, ACQUIRED_JSON_1C,
                 LICENSE_CSV_1C, LICENSE_JSON_1C, INTEGRITY_1C, REPORT_1C,
                 *ARCHIVES_1C.values()):
        verify_manifest_record(output_records.get(rel(path)), path)

    family_rows = acquired["families"]
    family_ids = {row["family_id"] for row in family_rows}
    require(family_ids == set(ARCHIVES_1C), "family identity set")
    archive_hashes = integrity.get("family_archive_sha256")
    tree_hashes = integrity.get("family_tree_sha256")
    require(isinstance(archive_hashes, dict) and isinstance(tree_hashes, dict), "integrity mappings")
    aggregate = hashlib.sha256()
    for row in sorted(family_rows, key=lambda item: item["family_id"]):
        family_id = row["family_id"]
        source_dir = SOURCE_ROOT / family_id
        require(source_dir.is_dir(), f"source tree missing: {family_id}")
        calculated_tree = tree_digest(source_dir)
        require(calculated_tree == row["tree_sha256"] == tree_hashes[family_id], f"tree integrity: {family_id}")
        require(sha256(ARCHIVES_1C[family_id]) == row["archive_sha256"] == archive_hashes[family_id], f"archive integrity: {family_id}")
        aggregate.update(family_id.encode())
        aggregate.update(b"\0")
        aggregate.update(calculated_tree.encode())
        aggregate.update(b"\0")
        aggregate.update(row["archive_sha256"].encode())
        aggregate.update(b"\0")
    require(aggregate.hexdigest() == integrity.get("aggregate_corpus_sha256"), "aggregate corpus integrity")

    inventory_by_family: dict[str, list[dict[str, str]]] = {family_id: [] for family_id in family_ids}
    with INVENTORY_1C.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            require(row.get("family_id") in inventory_by_family, "inventory family")
            inventory_by_family[row["family_id"]].append(row)
    print("  Frozen source trees, archives, licenses, partitions and protected-access locks              : PASS")
    return family_rows, inventory_by_family


def main() -> None:
    require(ROOT.name == "vlsi_fault_detection_v2", f"run from project root, not {ROOT}")
    outputs = (WRAPPER_CONTRACT, ACCESS_POLICY, SYNTH_AUTH, RECIPE_CSV, RECIPE_JSON,
               WRAPPER_REGISTRY, ENVIRONMENT, PREFLIGHT, REPORT, MANIFEST, AUDIT)
    for path in outputs:
        require(not path.exists(), f"Stage {STAGE} output already exists: {rel(path)}")
    require(not SOURCE_LISTS.exists(), f"Stage {STAGE} source lists already exist: {rel(SOURCE_LISTS)}")

    family_rows, inventory = verify_inputs()
    timestamp = now()
    tools = {
        "yosys": tool_version("yosys", ["-V"]),
        "verilator": tool_version("verilator", ["--version"]),
    }

    wrapper_contract = {
        "contract_version": "CIRCUITSAGE-HMAC-V2.2-PORTABLE-WRAPPER-CONTRACT-12C1D-v1",
        "stage": STAGE,
        "status": "FROZEN",
        "purpose": "CIRCUIT-INDEPENDENT TRANSACTION, OBSERVATION AND FAULT-CAMPAIGN BOUNDARY",
        "required_logical_ports": {
            "clock": "single normalized rising-edge experiment clock",
            "reset": "active polarity recorded per adapter; wrapper exports normalized active-high reset",
            "request": "start one pre-registered transaction/vector",
            "input_payload": "fixed-width or masked packed input derived without fault identity",
            "input_validity_mask": "declares meaningful payload bits",
            "response_payload": "packed architectural/external response",
            "response_validity_mask": "declares meaningful response bits",
            "busy_done_timeout": "normalized completion and timeout state",
        },
        "adapter_rules": [
            "one adapter per family; upstream RTL remains byte-identical",
            "no site ID, fault selector, stuck value or partition label at inference-facing ports",
            "constant parameters and reset sequence frozen before fault-campaign construction",
            "response fields must be externally observable or pre-registered probes",
            "unsupported or non-terminating transactions produce explicit masks/timeouts",
            "wrapper may not reveal golden-vs-faulty truth through metadata",
        ],
        "synthesis_boundary": "wrapper plus pinned upstream RTL only",
        "test_vector_status": "NOT DESIGNED OR AUTHORIZED BY THIS STAGE",
        "fault_injection_status": "NOT AUTHORIZED",
    }

    access_policy = {
        "policy_version": "CIRCUITSAGE-HMAC-V2.2-ELABORATION-PARTITION-ACCESS-12C1D-v1",
        "stage": STAGE,
        "status": "FROZEN",
        "GENERALIZATION_TRAIN": "ELABORATION, WRAPPER VALIDATION AND GENERIC SYNTHESIS AUTHORIZED IN NEXT GATE",
        "GENERALIZATION_CALIBRATION": "ELABORATION, WRAPPER VALIDATION AND GENERIC SYNTHESIS AUTHORIZED IN NEXT GATE; NO MODEL GRADIENTS",
        "INDEPENDENT_CIRCUIT_TEST": "LOCKED; NO ELABORATION, WRAPPER TUNING OR SYNTHESIS",
        "GENERALIZATION_HOLDOUT": "SEALED; NO SEMANTIC ACCESS, ELABORATION, WRAPPER WORK OR SYNTHESIS",
        "test_unlock_condition": "SELECTED V2.2 MODEL, FEATURE SCHEMA, WRAPPERS, VECTORS, PROBES AND THRESHOLDS MUST BE FROZEN FIRST",
        "holdout_unlock_condition": "NOT AUTHORIZED BY THE V2.2 DEVELOPMENT PROGRAM",
        "reassignment": "PROHIBITED",
    }

    family_map = {row["family_id"]: row for row in family_rows}
    authorized_ids = sorted(row["family_id"] for row in family_rows if row["partition"] in AUTHORIZED_PARTITIONS)
    locked_test_ids = sorted(row["family_id"] for row in family_rows if row["partition"] == "INDEPENDENT_CIRCUIT_TEST")
    holdout_ids = sorted(row["family_id"] for row in family_rows if row["partition"] == "GENERALIZATION_HOLDOUT")
    require(len(authorized_ids) == 4 and len(locked_test_ids) == 2 and len(holdout_ids) == 1, "authorization partition counts")

    recipe_rows: list[dict[str, Any]] = []
    wrapper_rows: list[dict[str, Any]] = []
    source_list_records: dict[str, dict[str, Any]] = {}
    for family_id in sorted(family_map):
        row = family_map[family_id]
        metadata_path = SOURCE_ROOT / family_id / "CORPUS_METADATA.json"
        metadata = load_json(metadata_path)
        require(metadata.get("family_id") == family_id and metadata.get("revision") == row["revision"], f"family metadata: {family_id}")
        authorized = row["partition"] in AUTHORIZED_PARTITIONS
        rtl_rows = [item for item in inventory[family_id] if Path(item["relative_path"]).suffix.lower() in RTL_SUFFIXES]
        require(len(rtl_rows) == int(row["rtl_file_count"]), f"RTL inventory count: {family_id}")
        source_list_path = ""
        source_list_sha = ""
        if authorized:
            source_paths = [str((SOURCE_ROOT / family_id / item["relative_path"]).resolve()) for item in sorted(rtl_rows, key=lambda item: item["relative_path"])]
            payload = ("\n".join(source_paths) + "\n").encode()
            path = SOURCE_LISTS / f"{family_id}_sources_12c1d.f"
            frozen_write(path, payload)
            source_list_path = rel(path)
            source_list_sha = sha256(path)
            source_list_records[family_id] = record(path)
        recipe_rows.append({
            "family_id": family_id,
            "partition": row["partition"],
            "candidate_top": metadata["candidate_top"],
            "language": "SYSTEMVERILOG" if any(Path(item["relative_path"]).suffix.lower() in {".sv", ".svh"} for item in rtl_rows) else "VERILOG-2001",
            "rtl_files": len(rtl_rows),
            "source_list": source_list_path,
            "source_list_sha256": source_list_sha,
            "elaboration": "AUTHORIZED NEXT GATE" if authorized else "LOCKED",
            "generic_synthesis": "AUTHORIZED NEXT GATE" if authorized else "LOCKED",
            "source_order": "RESOLVE PACKAGES/INCLUDES DETERMINISTICALLY IN 12C-1E" if authorized else "NOT OPENED",
            "upstream_mutation": "PROHIBITED",
        })
        wrapper_rows.append({
            "family_id": family_id,
            "partition": row["partition"],
            "candidate_top": metadata["candidate_top"],
            "adapter_status": "DESIGN/VALIDATION AUTHORIZED NEXT GATE" if authorized else "LOCKED",
            "logical_interface": "PORTABLE_WRAPPER_12C1D-v1" if authorized else "UNAVAILABLE",
            "fault_identity_ports": "PROHIBITED",
            "upstream_rtl_modified": "NO",
        })

    recipe_fields = ["family_id", "partition", "candidate_top", "language", "rtl_files", "source_list", "source_list_sha256", "elaboration", "generic_synthesis", "source_order", "upstream_mutation"]
    wrapper_fields = ["family_id", "partition", "candidate_top", "adapter_status", "logical_interface", "fault_identity_ports", "upstream_rtl_modified"]
    recipe_payload = csv_bytes(recipe_rows, recipe_fields)
    wrapper_payload = csv_bytes(wrapper_rows, wrapper_fields)

    recipe_json = {
        "registry_version": "CIRCUITSAGE-HMAC-V2.2-ELABORATION-RECIPE-REGISTRY-12C1D-v1",
        "stage": STAGE,
        "status": "FROZEN",
        "recipes": recipe_rows,
        "source_lists": source_list_records,
        "authorized_families": authorized_ids,
        "locked_test_families": locked_test_ids,
        "sealed_holdout_families": holdout_ids,
    }

    synth_auth = {
        "authorization_version": "CIRCUITSAGE-HMAC-V2.2-TRAIN-CALIBRATION-SYNTHESIS-AUTHORIZATION-12C1D-v1",
        "stage": STAGE,
        "status": "FROZEN",
        "authorized_next_action": "ELABORATE, BUILD PORTABLE ADAPTERS, LINT, AND GENERICALLY SYNTHESIZE TRAIN/CALIBRATION FAMILIES ONLY",
        "authorized_families": authorized_ids,
        "authorized_partitions": sorted(AUTHORIZED_PARTITIONS),
        "locked_test_families": locked_test_ids,
        "sealed_holdout_families": holdout_ids,
        "required_tools": tools,
        "execution": "SEQUENTIAL / ONE FAMILY AT A TIME / CHECKPOINT REQUIRED",
        "required_checks": [
            "source-tree SHA before and after every family",
            "language/package/include dependency resolution without source edits",
            "portable-wrapper lint and elaboration",
            "generic Yosys synthesis and zero-error structural check",
            "topology statistics and deterministic netlist replay",
            "wrapper smoke using fault-free transactions only",
        ],
        "fault_injection": "NOT AUTHORIZED",
        "response_dataset_creation": "NOT AUTHORIZED",
        "model_training_inference": "NOT AUTHORIZED / NOT AUTHORIZED",
        "test_validation_holdout_access": [0, 0, 0],
        "failure_rule": "FREEZE FAILURE DIAGNOSTIC; DO NOT MODIFY UPSTREAM RTL OR OPEN LOCKED PARTITIONS",
    }

    environment = {
        "environment_version": "CIRCUITSAGE-HMAC-V2.2-ELABORATION-ENVIRONMENT-12C1D-v1",
        "stage": STAGE,
        "status": "FROZEN",
        "created_at": timestamp,
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "tools": tools,
        "rtl_semantic_parsing": 0,
        "elaboration_calls": 0,
        "synthesis_calls": 0,
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
    preflight = {
        "preflight_version": "CIRCUITSAGE-HMAC-V2.2-ELABORATION-SYNTHESIS-PREFLIGHT-12C1D-v1",
        "stage": STAGE,
        "status": "PASS",
        "corpus_integrity": "PASS / 7 OF 7",
        "authorized_families": len(authorized_ids),
        "locked_test_families": len(locked_test_ids),
        "sealed_holdout_families": len(holdout_ids),
        "source_lists": len(source_list_records),
        "required_tools": "PASS",
        "wrapper_contract": "FROZEN",
        "upstream_rtl_modified": False,
        "elaboration_or_synthesis_executed": False,
    }
    report = f"""# CircuitSage-HMAC V2.2 Elaboration and Synthesis Authorization — Stage 12C-1D

The seven-family source corpus and every deterministic archive replayed against
the Stage 12C-1C integrity lock. Four families are authorized for the next
gate: three TRAIN circuits and one CALIBRATION circuit.

The portable wrapper contract normalizes clock/reset, request, payload,
validity, response, completion and timeout behavior without exposing fault
identity. Upstream RTL must remain byte-identical. Wrapper behavior and build
recipes may be developed only for TRAIN and CALIBRATION.

Both INDEPENDENT TEST circuits remain locked against elaboration, wrapper
tuning and synthesis. The HOLDOUT remains sealed. This stage performed no
elaboration, synthesis, simulation, fault injection, dataset construction,
training or inference.

**{FUTURE_BRAND}** remains reserved for the eventual completed hybrid release.
""".encode()

    frozen_write(WRAPPER_CONTRACT, canonical_json(wrapper_contract))
    frozen_write(ACCESS_POLICY, canonical_json(access_policy))
    frozen_write(SYNTH_AUTH, canonical_json(synth_auth))
    frozen_write(RECIPE_CSV, recipe_payload)
    frozen_write(RECIPE_JSON, canonical_json(recipe_json))
    frozen_write(WRAPPER_REGISTRY, wrapper_payload)
    frozen_write(ENVIRONMENT, canonical_json(environment))
    frozen_write(PREFLIGHT, canonical_json(preflight))
    frozen_write(REPORT, report)

    stage_outputs = [WRAPPER_CONTRACT, ACCESS_POLICY, SYNTH_AUTH, RECIPE_CSV,
                     RECIPE_JSON, WRAPPER_REGISTRY, ENVIRONMENT, PREFLIGHT,
                     REPORT, *[SOURCE_LISTS / f"{family_id}_sources_12c1d.f" for family_id in authorized_ids]]
    manifest = {
        "manifest_version": "CIRCUITSAGE-HMAC-V2.2-ELABORATION-SYNTHESIS-AUTHORIZATION-MANIFEST-12C1D-v1",
        "stage": STAGE,
        "status": "PASS",
        "created_at": timestamp,
        "frozen_inputs": {rel(path): record(path) for path in PINNED},
        "outputs": {rel(path): record(path) for path in stage_outputs},
        "authorized_families": authorized_ids,
        "locked_test_families": locked_test_ids,
        "sealed_holdout_families": holdout_ids,
        "elaboration_calls": 0,
        "synthesis_calls": 0,
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
        "audit_version": "CIRCUITSAGE-HMAC-V2.2-ELABORATION-WRAPPER-SYNTHESIS-AUTHORIZATION-FREEZE-12C1D-v1",
        "stage": STAGE,
        "status": "PASS",
        "authorization_status": "FROZEN",
        "corpus_integrity": "PASS / 7 OF 7",
        "wrapper_contract": "FROZEN",
        "build_recipes": "FROZEN / 7 FAMILIES",
        "train_calibration_elaboration_synthesis": "AUTHORIZED / NOT STARTED / 4 FAMILIES",
        "independent_test": "LOCKED / 2 FAMILIES",
        "holdout": "SEALED / 1 FAMILY",
        "upstream_rtl_modified": False,
        "elaboration_synthesis_simulation_fault_injection": [0, 0, 0, 0],
        "dataset_training_inference": [0, 0, 0],
        "independent_test_validation_holdout_access": [0, 0, 0],
        "future_combined_model_brand": FUTURE_BRAND,
        "wrapper_contract_record": record(WRAPPER_CONTRACT),
        "access_policy_record": record(ACCESS_POLICY),
        "synthesis_authorization_record": record(SYNTH_AUTH),
        "recipe_registry_record": record(RECIPE_JSON),
        "environment_record": record(ENVIRONMENT),
        "preflight_record": record(PREFLIGHT),
        "report_record": record(REPORT),
        "manifest_record": record(MANIFEST),
        "next_gate": "STAGE 12C-1E — TRAIN/CALIBRATION MULTI-CIRCUIT ELABORATION, WRAPPER VALIDATION, AND GENERIC-SYNTHESIS EXECUTION FREEZE",
    }
    frozen_write(AUDIT, canonical_json(audit))

    for path in (WRAPPER_CONTRACT, ACCESS_POLICY, SYNTH_AUTH, RECIPE_JSON,
                 ENVIRONMENT, PREFLIGHT, MANIFEST, AUDIT):
        require(canonical_json(load_json(path)) == path.read_bytes(), f"canonical output replay: {path.name}")
    require(RECIPE_CSV.read_bytes() == recipe_payload, "recipe CSV replay")
    require(WRAPPER_REGISTRY.read_bytes() == wrapper_payload, "wrapper registry replay")
    require(REPORT.read_bytes() == report, "report replay")
    for path, expected in PINNED.items():
        require(sha256(path) == expected, f"frozen input modified: {rel(path)}")

    print("\nSTAGE 12C-1D — MULTI-CIRCUIT ELABORATION, PORTABLE-WRAPPER, AND SYNTHESIS-AUTHORIZATION FREEZE")
    print(f"{'Status':<76}: PASS")
    print(f"{'Authorization / wrapper contract':<76}: FROZEN / FROZEN")
    print(f"{'Corpus integrity':<76}: PASS / 7 OF 7")
    print(f"{'Authorized TRAIN / CALIBRATION families':<76}: 3 / 1")
    print(f"{'Authorized family IDs':<76}: {authorized_ids}")
    print(f"{'Independent TEST':<76}: LOCKED / 2 FAMILIES")
    print(f"{'HOLDOUT':<76}: SEALED / 1 FAMILY")
    print(f"{'Yosys':<76}: {tools['yosys']['version']}")
    print(f"{'Verilator':<76}: {tools['verilator']['version']}")
    print(f"{'Elaboration / synthesis / simulation / fault injection':<76}: 0 / 0 / 0 / 0")
    print(f"{'Dataset construction / training / inference':<76}: NOT AUTHORIZED / NOT AUTHORIZED / NOT AUTHORIZED")
    print(f"{'Independent TEST / VALIDATION / HOLDOUT access':<76}: 0 / 0 / 0")
    print(f"{'Wrapper contract':<76}: {WRAPPER_CONTRACT}")
    print(f"{'Wrapper contract SHA':<76}: {sha256(WRAPPER_CONTRACT)}")
    print(f"{'Synthesis authorization':<76}: {SYNTH_AUTH}")
    print(f"{'Synthesis authorization SHA':<76}: {sha256(SYNTH_AUTH)}")
    print(f"{'Recipe registry':<76}: {RECIPE_JSON}")
    print(f"{'Recipe registry SHA':<76}: {sha256(RECIPE_JSON)}")
    print(f"{'Manifest':<76}: {MANIFEST}")
    print(f"{'Manifest SHA':<76}: {sha256(MANIFEST)}")
    print(f"{'Audit':<76}: {AUDIT}")
    print(f"{'Audit SHA':<76}: {sha256(AUDIT)}")
    print(f"{'Next gate':<76}: STAGE 12C-1E — TRAIN/CALIBRATION MULTI-CIRCUIT ELABORATION, WRAPPER VALIDATION, AND GENERIC-SYNTHESIS EXECUTION FREEZE")


def self_test() -> None:
    require(len(ARCHIVES_1C) == 7, "archive count")
    require(sum(EXPECTED_PARTITIONS.values()) == 7, "partition arithmetic")
    require(AUTHORIZED_PARTITIONS == {"GENERALIZATION_TRAIN", "GENERALIZATION_CALIBRATION"}, "authorization scope")
    require(canonical_json({"z": 1, "a": 2}) == b'{"a":2,"z":1}\n', "canonical JSON")
    print("Stage 12C-1D self-test: PASS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    arguments = parser.parse_args()
    if arguments.self_test:
        self_test()
    else:
        main()
