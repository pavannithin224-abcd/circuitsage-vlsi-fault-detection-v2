#!/usr/bin/env python3
"""Stage 12C-1B: multi-circuit corpus and family-split authorization freeze.

Verifies the frozen Stage 12C-1A V2.2 generalization contract, then freezes a
seven-family, circuit-disjoint RTL source registry, immutable upstream revision
commitments, license/provenance policy, partition assignment, and a bounded
source-acquisition authorization.  This stage performs no network access,
downloads, RTL parsing, synthesis, simulation, fault injection, dataset
construction, model loading, training, inference, or protected evaluation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STAGE = "12C-1B"
ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "config/v2_2"
RESULT = ROOT / "results/circuitsage_hmac_v2_12c1"
WORK = RESULT / "multicircuit_corpus_authorization_12c1b"

SOURCE_1A = ROOT / "stage_12c1a_v2_2_generalization_contract.py"
LOG_1A = RESULT / "stage_12c1a_20260918_004844.log"
RESOURCE_1A = RESULT / "stage_12c1a_resources_20260918_004844.log"
SCOPE_1A = CONFIG / "circuitsage_hmac_v2_2_generalization_scope_contract_12c1a.json"
ARCH_1A = CONFIG / "circuitsage_hmac_v2_2_generalization_architecture_12c1a.json"
PARTITION_1A = CONFIG / "circuitsage_hmac_v2_2_circuit_family_partition_contract_12c1a.json"
INTERFACE_1A = CONFIG / "circuitsage_hmac_v2_2_inference_interface_contract_12c1a.json"
TRAINING_1A = CONFIG / "circuitsage_hmac_v2_2_training_selection_contract_12c1a.json"
ACCEPTANCE_1A = CONFIG / "circuitsage_hmac_v2_2_acceptance_contract_12c1a.json"
GRID_1A = CONFIG / "circuitsage_hmac_v2_2_candidate_grid_12c1a.csv"
ENV_1A = RESULT / "circuitsage_hmac_v2_2_contract_environment_12c1a.json"
REPORT_1A = RESULT / "circuitsage_hmac_v2_2_generalization_contract_report_12c1a.md"
MANIFEST_1A = RESULT / "circuitsage_hmac_v2_2_generalization_contract_manifest_12c1a.json"
AUDIT_1A = RESULT / "circuitsage_hmac_v2_2_generalization_contract_freeze_12c1a.json"

SOURCE_POLICY = CONFIG / "circuitsage_hmac_v2_2_corpus_source_license_policy_12c1b.json"
SPLIT_AUTH = CONFIG / "circuitsage_hmac_v2_2_family_split_authorization_12c1b.json"
ACQUISITION = CONFIG / "circuitsage_hmac_v2_2_corpus_acquisition_contract_12c1b.json"
FAMILY_CSV = WORK / "circuitsage_hmac_v2_2_circuit_family_registry_12c1b.csv"
FAMILY_JSON = WORK / "circuitsage_hmac_v2_2_circuit_family_registry_12c1b.json"
LICENSE_CSV = WORK / "circuitsage_hmac_v2_2_license_provenance_registry_12c1b.csv"
LICENSE_JSON = WORK / "circuitsage_hmac_v2_2_license_provenance_registry_12c1b.json"
COMMITMENTS = WORK / "circuitsage_hmac_v2_2_source_revision_commitments_12c1b.json"
PREFLIGHT = WORK / "circuitsage_hmac_v2_2_corpus_authorization_preflight_12c1b.json"
REPORT = WORK / "circuitsage_hmac_v2_2_corpus_authorization_report_12c1b.md"
MANIFEST = RESULT / "circuitsage_hmac_v2_2_corpus_authorization_manifest_12c1b.json"
AUDIT = RESULT / "circuitsage_hmac_v2_2_corpus_license_split_authorization_freeze_12c1b.json"

PINNED = {
    SOURCE_1A: "e26905e8cfe4a0e3852b355afb6d09c96ca5192fa9075674829adf85a11cdb5f",
    LOG_1A: "bbed701646821d829582bb649d38c88fcd5a297ec9e52a2fb6542375c67d1a48",
    RESOURCE_1A: "b138496691329e110891567ad5e145127cc1522de1917214ef1ee9d246167939",
    SCOPE_1A: "993902928433953151b09641c156dbc9d9e154f33f5268b9329142fccfc45361",
    ARCH_1A: "e83da87e0ba773ed1b6872528c216079d31fc83537b4a2eca348bc6a23791649",
    PARTITION_1A: "4cbab060ef824f0ff8a17af7e2250d0efcf608ac7121c685bcf9eef4f5b2804c",
    INTERFACE_1A: "c5e222640b7404e83fe925ec1b33b2bbf633880b4f21b253ba28d5025686541a",
    TRAINING_1A: "081578897b76530c5d63ce2f283b5c7c19d9c5aaf9d7c1caf421e9ee5a564433",
    ACCEPTANCE_1A: "9c8eec4d85957c4408ac59e0c8760af90c91d0667c995d91ac969b5a8f205f26",
    GRID_1A: "2297edd6a7b61cca28709509a015ff246c43a8379a36ed4842b8d1a2c5528470",
    ENV_1A: "c95ddb00f9789c67a31c82b80e9a3f2c52b9393e135a7cf49eadd7c5ae6dd26e",
    REPORT_1A: "fa65512854a96adffb163f26853605127e34d96ae7153678e98ffc85acabe704",
    MANIFEST_1A: "e1a2000508a3e8d141077a487100f3a54250f227963e663f23cbd21749ed2ddc",
    AUDIT_1A: "4e16a3d498e237c312fdef5565300b8ffc37dde75caa6def92c1651514006ae3",
}

FUTURE_BRAND = "Faultiva 1.0 — Hybrid VLSI Fault Intelligence"
SPLIT_DOMAIN = "CIRCUITSAGE-HMAC-V2.2-CIRCUIT-FAMILY-SPLIT-12C1A-v1"
PARTITION_COUNTS = {
    "GENERALIZATION_TRAIN": 3,
    "GENERALIZATION_CALIBRATION": 1,
    "INDEPENDENT_CIRCUIT_TEST": 2,
    "GENERALIZATION_HOLDOUT": 1,
}

# Immutable revisions were resolved before this authorization was created.
# No remote is contacted by this script. Stage 12C-1C must independently
# resolve each revision, download it, hash the archive and verify its license.
FAMILIES = [
    {
        "family_id": "opentitan_hmac_sha256",
        "display_name": "OpenTitan HMAC-SHA256",
        "design_class": "CRYPTO_HASH_MAC",
        "partition": "GENERALIZATION_TRAIN",
        "repository": "https://github.com/lowRISC/opentitan",
        "revision": "83fc48ed3a727399056772d12be8c7d4a8a276f0",
        "spdx_license": "Apache-2.0",
        "license_path": "LICENSE",
        "rtl_scope": "hw/ip/hmac/rtl; required hw/ip/prim/rtl dependencies only",
        "candidate_top": "hmac_core",
        "prior_project_exposure": "YES — FORCED TO TRAIN",
    },
    {
        "family_id": "secworks_aes",
        "display_name": "secworks AES",
        "design_class": "CRYPTO_BLOCK_CIPHER",
        "partition": "GENERALIZATION_TRAIN",
        "repository": "https://github.com/secworks/aes",
        "revision": "80dc4718e1dcbbdb4b0dd1bdb393d8f7b98981dc",
        "spdx_license": "BSD-2-Clause",
        "license_path": "LICENSE",
        "rtl_scope": "src/rtl",
        "candidate_top": "aes_core",
        "prior_project_exposure": "NO",
    },
    {
        "family_id": "picorv32_cpu",
        "display_name": "PicoRV32",
        "design_class": "RISC_V_CPU",
        "partition": "GENERALIZATION_TRAIN",
        "repository": "https://github.com/YosysHQ/picorv32",
        "revision": "ef203c2b0a3fb793280f5114941416c425c5b461",
        "spdx_license": "ISC",
        "license_path": "COPYING",
        "rtl_scope": "picorv32.v",
        "candidate_top": "picorv32",
        "prior_project_exposure": "NO",
    },
    {
        "family_id": "secworks_sha256",
        "display_name": "secworks SHA-256",
        "design_class": "CRYPTO_HASH",
        "partition": "GENERALIZATION_CALIBRATION",
        "repository": "https://github.com/secworks/sha256",
        "revision": "837c5cc396f001d18f2c765721c585716eb439ae",
        "spdx_license": "BSD-2-Clause",
        "license_path": "LICENSE",
        "rtl_scope": "src/rtl",
        "candidate_top": "sha256_core",
        "prior_project_exposure": "NO",
    },
    {
        "family_id": "ibex_cpu",
        "display_name": "Ibex RISC-V Core",
        "design_class": "RISC_V_CPU",
        "partition": "INDEPENDENT_CIRCUIT_TEST",
        "repository": "https://github.com/lowRISC/ibex",
        "revision": "e9f55342edbd27e9e17a0e41b1c95a81abb5eac8",
        "spdx_license": "Apache-2.0",
        "license_path": "LICENSE",
        "rtl_scope": "rtl; required vendor/lowrisc_ip/ip/prim dependencies only",
        "candidate_top": "ibex_top",
        "prior_project_exposure": "NO",
    },
    {
        "family_id": "secworks_chacha",
        "display_name": "secworks ChaCha",
        "design_class": "CRYPTO_STREAM_CIPHER",
        "partition": "INDEPENDENT_CIRCUIT_TEST",
        "repository": "https://github.com/secworks/chacha",
        "revision": "7eaba360df9fed9fc2db98d5f3df81cf01e5b604",
        "spdx_license": "BSD-2-Clause",
        "license_path": "LICENSE",
        "rtl_scope": "src/rtl",
        "candidate_top": "chacha_core",
        "prior_project_exposure": "NO",
    },
    {
        "family_id": "serv_cpu",
        "display_name": "SERV Bit-Serial RISC-V",
        "design_class": "RISC_V_CPU",
        "partition": "GENERALIZATION_HOLDOUT",
        "repository": "https://github.com/olofk/serv",
        "revision": "f200eb2ed7b69ac1c6b8eddd47654522aeee5ce8",
        "spdx_license": "ISC",
        "license_path": "LICENSE",
        "rtl_scope": "rtl",
        "candidate_top": "serv_top",
        "prior_project_exposure": "NO",
    },
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


def predecessor_canonical_json(value: Any) -> bytes:
    """Replay the indented serializer frozen by Stage 12C-1A."""
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode()


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


def verify_record(item: Any, path: Path, label: str) -> None:
    require(isinstance(item, dict), f"missing manifest record: {label}")
    require(item.get("path") == rel(path), f"manifest path mismatch: {label}")
    require(item.get("sha256") == sha256(path), f"manifest SHA mismatch: {label}")
    require(int(item.get("bytes", -1)) == path.stat().st_size, f"manifest size mismatch: {label}")


def family_commitment(row: dict[str, str]) -> str:
    payload = "\0".join([
        SPLIT_DOMAIN,
        row["family_id"],
        row["repository"],
        row["revision"],
        row["partition"],
        row["spdx_license"],
    ]).encode()
    return hashlib.sha256(payload).hexdigest()


def validate_families() -> None:
    require(len(FAMILIES) == 7, "exactly seven circuit families required")
    require(len({row["family_id"] for row in FAMILIES}) == 7, "family IDs must be unique")
    require(len({(row["repository"], row["revision"]) for row in FAMILIES}) == 7, "source revisions must be unique")
    counts = {name: 0 for name in PARTITION_COUNTS}
    allowed_licenses = {"Apache-2.0", "BSD-2-Clause", "ISC"}
    for row in FAMILIES:
        require(row["partition"] in counts, f"unknown partition: {row['family_id']}")
        counts[row["partition"]] += 1
        require(re.fullmatch(r"[0-9a-f]{40}", row["revision"]) is not None, f"invalid revision: {row['family_id']}")
        require(row["repository"].startswith("https://github.com/"), f"non-HTTPS repository: {row['family_id']}")
        require(row["spdx_license"] in allowed_licenses, f"unapproved license: {row['family_id']}")
    require(counts == PARTITION_COUNTS, f"partition counts: {counts}")
    require(next(row for row in FAMILIES if row["family_id"] == "opentitan_hmac_sha256")["partition"] == "GENERALIZATION_TRAIN", "previously exposed HMAC must be TRAIN")
    test_rows = [row for row in FAMILIES if row["partition"] == "INDEPENDENT_CIRCUIT_TEST"]
    require(all(row["prior_project_exposure"] == "NO" for row in test_rows), "test family prior-exposure firewall")
    require(len({row["design_class"] for row in FAMILIES}) >= 5, "design-class diversity")


def verify_inputs() -> None:
    print("STAGE 12C-1B — MULTI-CIRCUIT RTL CORPUS, LICENSE-PROVENANCE, AND FAMILY-SPLIT AUTHORIZATION FREEZE")
    print("FROZEN INPUT VERIFICATION")
    for path, expected in PINNED.items():
        require(path.is_file(), f"missing frozen input: {rel(path)}")
        require(sha256(path) == expected, f"frozen input SHA: {rel(path)}")
        print(f"  {path.name:<108}: OK")

    scope = load_json(SCOPE_1A)
    partition = load_json(PARTITION_1A)
    acceptance = load_json(ACCEPTANCE_1A)
    manifest = load_json(MANIFEST_1A)
    audit = load_json(AUDIT_1A)

    require(scope.get("status") == "FROZEN", "12C-1A scope status")
    require(scope.get("query_fault_identity") == "PROHIBITED", "query identity prohibition")
    require(scope.get("future_combined_release_brand") == FUTURE_BRAND, "future brand reservation")
    require(partition.get("minimum_independent_circuit_families") == 7, "minimum family count")
    require(partition.get("minimum_family_counts") == PARTITION_COUNTS, "partition count contract")
    require(partition.get("split_hash_domain") == SPLIT_DOMAIN, "split hash domain")
    require(acceptance.get("holdout") == "REMAINS BLOCKED EVEN IF V2.2 PASSES", "holdout lock")
    require(manifest.get("status") == "PASS", "12C-1A manifest status")
    require(manifest.get("dataset_construction_authorized") is False, "12C-1A dataset boundary")
    require(manifest.get("training_authorized") is False, "12C-1A training boundary")
    require(manifest.get("evaluation_authorized") is False, "12C-1A evaluation boundary")
    require(audit.get("status") == "PASS", "12C-1A audit status")
    require(audit.get("architecture_status") == "FROZEN", "architecture freeze")
    require(audit.get("partition_contract_status") == "FROZEN", "partition freeze")
    require(audit.get("dataset_construction") == "NOT AUTHORIZED", "dataset remains blocked")
    require(audit.get("validation_holdout_access") == [0, 0], "protected access")

    predecessor_outputs = manifest.get("outputs")
    require(isinstance(predecessor_outputs, dict), "12C-1A output registry")
    for path in (SCOPE_1A, ARCH_1A, PARTITION_1A, INTERFACE_1A, TRAINING_1A,
                 ACCEPTANCE_1A, GRID_1A, ENV_1A, REPORT_1A):
        verify_record(predecessor_outputs.get(rel(path)), path, path.name)

    for path in (SCOPE_1A, ARCH_1A, PARTITION_1A, INTERFACE_1A, TRAINING_1A,
                 ACCEPTANCE_1A, ENV_1A, MANIFEST_1A, AUDIT_1A):
        require(predecessor_canonical_json(load_json(path)) == path.read_bytes(), f"canonical predecessor JSON: {path.name}")
    print("  V2.2 scope, partitions, acceptance, protected locks and brand reservation             : PASS")


def self_test() -> None:
    validate_families()
    commitments = [family_commitment(row) for row in FAMILIES]
    require(len(set(commitments)) == 7, "commitments must be unique")
    require(all(re.fullmatch(r"[0-9a-f]{64}", value) for value in commitments), "commitment format")
    sample = {"z": 1, "a": [2, 3]}
    require(canonical_json(sample) == canonical_json(json.loads(canonical_json(sample))), "canonical JSON")
    print("Stage 12C-1B self-test: PASS")


def main() -> None:
    require(ROOT.name == "vlsi_fault_detection_v2", f"run from project root, not {ROOT}")
    outputs = (SOURCE_POLICY, SPLIT_AUTH, ACQUISITION, FAMILY_CSV, FAMILY_JSON,
               LICENSE_CSV, LICENSE_JSON, COMMITMENTS, PREFLIGHT, REPORT, MANIFEST, AUDIT)
    for path in outputs:
        require(not path.exists(), f"Stage {STAGE} output already exists: {rel(path)}")

    validate_families()
    verify_inputs()
    timestamp = now()

    enriched = []
    for row in FAMILIES:
        item = dict(row)
        item["archive_url"] = f"{row['repository']}/archive/{row['revision']}.tar.gz"
        item["family_split_commitment"] = family_commitment(row)
        item["acquisition_status"] = "AUTHORIZED / NOT STARTED"
        item["synthesis_status"] = "NOT AUTHORIZED"
        item["dataset_status"] = "NOT AUTHORIZED"
        enriched.append(item)

    source_policy = {
        "policy_version": "CIRCUITSAGE-HMAC-V2.2-CORPUS-SOURCE-LICENSE-POLICY-12C1B-v1",
        "stage": STAGE,
        "status": "FROZEN",
        "eligible_license_spdx": ["Apache-2.0", "BSD-2-Clause", "ISC"],
        "source_requirements": [
            "public HTTPS repository",
            "full immutable 40-hex commit",
            "license file present at pinned revision",
            "RTL file inventory and recursive SHA-256 tree",
            "upstream provenance and citation retained",
            "submodules and vendored dependencies separately licensed and pinned",
        ],
        "redistribution_rule": "NO UPSTREAM RTL MAY ENTER A PUBLIC RELEASE UNTIL ITS PINNED LICENSE, NOTICE, ATTRIBUTION AND DERIVATIVE OBLIGATIONS PASS A SEPARATE RELEASE REVIEW",
        "restricted_material_rule": "PROPRIETARY, CONFIDENTIAL, NDA, EXPORT-RESTRICTED, NON-REDISTRIBUTABLE OR UNKNOWN-LICENSE MATERIAL IS PROHIBITED",
        "generated_artifact_rule": "GENERATED NETLISTS AND DATASETS INHERIT SOURCE REVIEW REQUIREMENTS AND ARE PRIVATE BY DEFAULT",
        "legal_boundary": "SPDX/FILE REVIEW IS TECHNICAL PROVENANCE CONTROL, NOT LEGAL ADVICE",
        "network_access": "NOT PERFORMED BY THIS STAGE",
    }

    split_auth = {
        "authorization_version": "CIRCUITSAGE-HMAC-V2.2-FAMILY-SPLIT-AUTHORIZATION-12C1B-v1",
        "stage": STAGE,
        "status": "FROZEN",
        "split_unit": "CIRCUIT FAMILY",
        "split_domain": SPLIT_DOMAIN,
        "partition_counts": PARTITION_COUNTS,
        "assignment_method": "LICENSE/PROVENANCE ELIGIBILITY, PRIOR-EXPOSURE FIREWALL, DESIGN-CLASS STRATIFICATION, THEN SHA-256 COMMITMENT",
        "prior_exposure_rule": "OPENTITAN HMAC WAS USED TO DESIGN EARLIER STAGES AND IS FORCED INTO GENERALIZATION_TRAIN",
        "test_rule": "INDEPENDENT TEST FAMILIES HAVE NO PRIOR PROJECT EXPOSURE AND MAY NOT INFORM ARCHITECTURE, FEATURES, VECTORS, PROBES, THRESHOLDS OR MODEL SELECTION",
        "holdout_rule": "SERV FAMILY IS REGISTERED BUT REMAINS SEALED; ACQUISITION MAY VERIFY ARCHIVE/LICENSE HASHES ONLY",
        "family_assignments": {row["family_id"]: row["partition"] for row in enriched},
        "family_commitments": {row["family_id"]: row["family_split_commitment"] for row in enriched},
        "reassignment": "PROHIBITED AFTER THIS FREEZE; A SOURCE FAILURE REQUIRES A NEW VERSIONED CONTRACT, NOT SILENT SUBSTITUTION",
    }

    acquisition = {
        "contract_version": "CIRCUITSAGE-HMAC-V2.2-CORPUS-ACQUISITION-12C1B-v1",
        "stage": STAGE,
        "status": "FROZEN",
        "authorized_next_action": "DOWNLOAD PINNED SOURCE ARCHIVES; VERIFY COMMIT, LICENSE, NOTICE, INVENTORY, DEPENDENCIES AND RECURSIVE HASHES; FREEZE CORPUS",
        "authorized_families": [row["family_id"] for row in enriched],
        "required_controls": [
            "one immutable archive per registered family",
            "fail closed on revision, license, dependency or path mismatch",
            "record archive SHA-256 and extracted-tree SHA-256",
            "normalize no RTL and alter no upstream source",
            "store TEST and HOLDOUT under access-marked directories",
            "emit zero training examples and zero fault responses",
        ],
        "network_domains": ["github.com"],
        "maximum_download_bytes_per_family": 1073741824,
        "rtl_elaboration": "NOT AUTHORIZED",
        "synthesis_simulation_fault_injection": "NOT AUTHORIZED / NOT AUTHORIZED / NOT AUTHORIZED",
        "dataset_construction": "NOT AUTHORIZED",
        "model_training_inference": "NOT AUTHORIZED / NOT AUTHORIZED",
        "independent_test_truth_access": 0,
        "holdout_semantic_access": 0,
    }

    family_fields = [
        "family_id", "display_name", "design_class", "partition", "repository",
        "revision", "spdx_license", "license_path", "rtl_scope", "candidate_top",
        "prior_project_exposure", "archive_url", "family_split_commitment",
        "acquisition_status", "synthesis_status", "dataset_status",
    ]
    family_payload = csv_bytes(enriched, family_fields)
    family_json = {
        "registry_version": "CIRCUITSAGE-HMAC-V2.2-CIRCUIT-FAMILY-REGISTRY-12C1B-v1",
        "stage": STAGE,
        "status": "FROZEN",
        "family_count": len(enriched),
        "partition_counts": PARTITION_COUNTS,
        "families": enriched,
    }

    license_rows = []
    for row in enriched:
        license_rows.append({
            "family_id": row["family_id"],
            "repository": row["repository"],
            "revision": row["revision"],
            "spdx_license": row["spdx_license"],
            "license_path": row["license_path"],
            "provenance_status": "REGISTERED / REMOTE BYTE VERIFICATION PENDING 12C-1C",
            "redistribution_status": "CONDITIONALLY ELIGIBLE / FINAL RELEASE REVIEW REQUIRED",
        })
    license_fields = ["family_id", "repository", "revision", "spdx_license", "license_path", "provenance_status", "redistribution_status"]
    license_payload = csv_bytes(license_rows, license_fields)
    license_json = {
        "registry_version": "CIRCUITSAGE-HMAC-V2.2-LICENSE-PROVENANCE-REGISTRY-12C1B-v1",
        "stage": STAGE,
        "status": "FROZEN",
        "licenses": license_rows,
        "license_files_read": 0,
        "remote_verification": "PENDING STAGE 12C-1C",
    }

    commitments = {
        "commitment_version": "CIRCUITSAGE-HMAC-V2.2-SOURCE-REVISION-COMMITMENTS-12C1B-v1",
        "stage": STAGE,
        "status": "FROZEN",
        "split_domain": SPLIT_DOMAIN,
        "algorithm": "SHA-256(domain NUL family_id NUL repository NUL revision NUL partition NUL SPDX)",
        "commitments": {row["family_id"]: row["family_split_commitment"] for row in enriched},
    }

    preflight = {
        "preflight_version": "CIRCUITSAGE-HMAC-V2.2-CORPUS-AUTHORIZATION-PREFLIGHT-12C1B-v1",
        "stage": STAGE,
        "status": "PASS",
        "created_at": timestamp,
        "families": len(enriched),
        "design_classes": len({row["design_class"] for row in enriched}),
        "partition_counts": PARTITION_COUNTS,
        "immutable_revisions": True,
        "approved_spdx_only": True,
        "prior_exposure_firewall": True,
        "network_calls": 0,
        "downloads": 0,
        "rtl_files_read": 0,
        "license_files_read": 0,
        "synthesis_calls": 0,
        "simulation_calls": 0,
        "fault_injection_calls": 0,
        "dataset_payloads_read": 0,
        "models_deserialized": 0,
        "training_calls": 0,
        "inference_calls": 0,
        "validation_access": 0,
        "holdout_access": 0,
    }

    report = f"""# CircuitSage-HMAC V2.2 Multi-Circuit Corpus Authorization — Stage 12C-1B

## Frozen corpus plan

Seven independent RTL families are registered across five design classes.
The fixed split is 3 TRAIN, 1 CALIBRATION, 2 locked INDEPENDENT TEST, and 1
blocked HOLDOUT. OpenTitan HMAC is forced into TRAIN because it informed the
earlier project; it cannot be used as evidence of unseen-circuit performance.

## License and provenance boundary

Only Apache-2.0, BSD-2-Clause, and ISC sources are registered. This stage
records upstream repositories, immutable commits, license paths and expected
SPDX identifiers. It does not download or redistribute upstream RTL. Stage
12C-1C must verify the actual archive and license bytes, dependency licenses,
notices, file inventory, and recursive hashes. Any mismatch fails closed.

## Current authorization

The next stage may download the seven pinned source archives solely to verify
provenance and freeze the corpus. Elaboration, synthesis, simulation, fault
injection, vector/probe optimization, dataset creation, model training,
inference, and protected truth access remain prohibited.

## Naming boundary

**{FUTURE_BRAND}** remains reserved for the eventual completed V1+V2 hybrid.
The V2.2 research component is not yet Faultiva.
""".encode()

    frozen_write(SOURCE_POLICY, canonical_json(source_policy))
    frozen_write(SPLIT_AUTH, canonical_json(split_auth))
    frozen_write(ACQUISITION, canonical_json(acquisition))
    frozen_write(FAMILY_CSV, family_payload)
    frozen_write(FAMILY_JSON, canonical_json(family_json))
    frozen_write(LICENSE_CSV, license_payload)
    frozen_write(LICENSE_JSON, canonical_json(license_json))
    frozen_write(COMMITMENTS, canonical_json(commitments))
    frozen_write(PREFLIGHT, canonical_json(preflight))
    frozen_write(REPORT, report)

    artifact_outputs = [SOURCE_POLICY, SPLIT_AUTH, ACQUISITION, FAMILY_CSV,
                        FAMILY_JSON, LICENSE_CSV, LICENSE_JSON, COMMITMENTS,
                        PREFLIGHT, REPORT]
    manifest = {
        "manifest_version": "CIRCUITSAGE-HMAC-V2.2-CORPUS-AUTHORIZATION-MANIFEST-12C1B-v1",
        "stage": STAGE,
        "status": "PASS",
        "created_at": timestamp,
        "frozen_inputs": {rel(path): record(path) for path in PINNED},
        "outputs": {rel(path): record(path) for path in artifact_outputs},
        "family_count": len(enriched),
        "partition_counts": PARTITION_COUNTS,
        "source_acquisition_authorized": True,
        "dataset_construction_authorized": False,
        "training_authorized": False,
        "evaluation_authorized": False,
        "network_calls": 0,
        "downloads": 0,
        "rtl_files_read": 0,
        "dataset_payloads_read": 0,
        "models_deserialized": 0,
        "training_calls": 0,
        "inference_calls": 0,
        "validation_access": 0,
        "holdout_access": 0,
    }
    frozen_write(MANIFEST, canonical_json(manifest))

    audit = {
        "audit_version": "CIRCUITSAGE-HMAC-V2.2-CORPUS-LICENSE-SPLIT-AUTHORIZATION-FREEZE-12C1B-v1",
        "stage": STAGE,
        "status": "PASS",
        "authorization_status": "FROZEN",
        "corpus_registry": "FROZEN — 7 CIRCUIT FAMILIES / 5 DESIGN CLASSES",
        "license_provenance_registry": "FROZEN / REMOTE BYTE VERIFICATION PENDING",
        "family_split": "FROZEN — 3 TRAIN / 1 CALIBRATION / 2 TEST / 1 HOLDOUT",
        "immutable_revision_commitments": "FROZEN",
        "source_acquisition": "AUTHORIZED / NOT STARTED",
        "rtl_elaboration_synthesis_simulation_fault_injection": "0 / 0 / 0 / 0",
        "dataset_construction_model_training_inference": "0 / 0 / 0",
        "independent_test_access": 0,
        "validation_holdout_access": [0, 0],
        "prior_exposure_firewall": "PASS — OPENTITAN HMAC FORCED TO TRAIN",
        "independent_generalization": "NOT YET EVALUATED",
        "future_combined_model_brand": FUTURE_BRAND,
        "source_policy": record(SOURCE_POLICY),
        "split_authorization": record(SPLIT_AUTH),
        "acquisition_contract": record(ACQUISITION),
        "family_registry_csv": record(FAMILY_CSV),
        "family_registry_json": record(FAMILY_JSON),
        "license_registry_csv": record(LICENSE_CSV),
        "license_registry_json": record(LICENSE_JSON),
        "source_commitments": record(COMMITMENTS),
        "preflight": record(PREFLIGHT),
        "report": record(REPORT),
        "manifest": record(MANIFEST),
        "next_gate": "STAGE 12C-1C — MULTI-CIRCUIT RTL ACQUISITION, LICENSE VERIFICATION, AND CORPUS-INTEGRITY FREEZE",
    }
    frozen_write(AUDIT, canonical_json(audit))

    for path in (SOURCE_POLICY, SPLIT_AUTH, ACQUISITION, FAMILY_JSON,
                 LICENSE_JSON, COMMITMENTS, PREFLIGHT, MANIFEST, AUDIT):
        require(canonical_json(load_json(path)) == path.read_bytes(), f"canonical output replay: {path.name}")
    require(FAMILY_CSV.read_bytes() == family_payload, "family CSV deterministic replay")
    require(LICENSE_CSV.read_bytes() == license_payload, "license CSV deterministic replay")
    require(REPORT.read_bytes() == report, "report deterministic replay")
    for path, expected in PINNED.items():
        require(sha256(path) == expected, f"frozen input modified: {rel(path)}")

    print("\nSTAGE 12C-1B — MULTI-CIRCUIT RTL CORPUS, LICENSE-PROVENANCE, AND FAMILY-SPLIT AUTHORIZATION FREEZE")
    print(f"{'Status':<72}: PASS")
    print(f"{'Authorization status':<72}: FROZEN")
    print(f"{'Circuit families / design classes':<72}: {len(enriched)} / {len({row['design_class'] for row in enriched})}")
    print(f"{'TRAIN / CALIBRATION / TEST / HOLDOUT':<72}: 3 / 1 / 2 / 1")
    print(f"{'Approved SPDX licenses':<72}: Apache-2.0 / BSD-2-Clause / ISC")
    print(f"{'Immutable source revisions':<72}: 7/7")
    print(f"{'Prior-exposure firewall':<72}: PASS — OPENTITAN HMAC FORCED TO TRAIN")
    print(f"{'Source acquisition':<72}: AUTHORIZED / NOT STARTED")
    print(f"{'RTL/download/network activity':<72}: 0 / 0 / 0")
    print(f"{'Elaboration / synthesis / simulation / fault injection':<72}: 0 / 0 / 0 / 0")
    print(f"{'Dataset construction / training / inference':<72}: NOT AUTHORIZED / NOT AUTHORIZED / NOT AUTHORIZED")
    print(f"{'Independent TEST / VALIDATION / HOLDOUT access':<72}: 0 / 0 / 0")
    print(f"{'Future combined-model brand':<72}: {FUTURE_BRAND}")
    print(f"{'Family registry':<72}: {FAMILY_CSV}")
    print(f"{'Family registry SHA':<72}: {sha256(FAMILY_CSV)}")
    print(f"{'License registry':<72}: {LICENSE_CSV}")
    print(f"{'License registry SHA':<72}: {sha256(LICENSE_CSV)}")
    print(f"{'Acquisition contract':<72}: {ACQUISITION}")
    print(f"{'Acquisition contract SHA':<72}: {sha256(ACQUISITION)}")
    print(f"{'Manifest':<72}: {MANIFEST}")
    print(f"{'Manifest SHA':<72}: {sha256(MANIFEST)}")
    print(f"{'Audit':<72}: {AUDIT}")
    print(f"{'Audit SHA':<72}: {sha256(AUDIT)}")
    print(f"{'Next gate':<72}: STAGE 12C-1C — MULTI-CIRCUIT RTL ACQUISITION, LICENSE VERIFICATION, AND CORPUS-INTEGRITY FREEZE")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    arguments = parser.parse_args()
    if arguments.self_test:
        self_test()
    else:
        main()
