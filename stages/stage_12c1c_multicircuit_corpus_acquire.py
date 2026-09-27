#!/usr/bin/env python3
"""Stage 12C-1C: acquire and freeze the V2.2 multi-circuit RTL corpus.

This resumable networked stage fetches the seven Stage 12C-1B repositories at
their immutable commits, verifies root license evidence, copies only approved
RTL/provenance scopes, builds deterministic local source archives, and freezes
file-level and tree-level integrity records.  It performs no elaboration,
synthesis, simulation, fault injection, dataset construction, model loading,
training, inference, or protected truth access.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


STAGE = "12C-1C"
ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "config/v2_2"
RESULT = ROOT / "results/circuitsage_hmac_v2_12c1"
PREV_WORK = RESULT / "multicircuit_corpus_authorization_12c1b"
WORK = RESULT / "multicircuit_corpus_12c1c"
SOURCE_ROOT = WORK / "sources"
ARCHIVE_ROOT = WORK / "archives"
FETCH_ROOT = WORK / "fetch_work"
LOG_ROOT = WORK / "fetch_logs"

SOURCE_1B = ROOT / "stage_12c1b_multicircuit_corpus_authorization.py"
SOURCE_POLICY_1B = CONFIG / "circuitsage_hmac_v2_2_corpus_source_license_policy_12c1b.json"
SPLIT_AUTH_1B = CONFIG / "circuitsage_hmac_v2_2_family_split_authorization_12c1b.json"
ACQUISITION_1B = CONFIG / "circuitsage_hmac_v2_2_corpus_acquisition_contract_12c1b.json"
FAMILY_CSV_1B = PREV_WORK / "circuitsage_hmac_v2_2_circuit_family_registry_12c1b.csv"
FAMILY_JSON_1B = PREV_WORK / "circuitsage_hmac_v2_2_circuit_family_registry_12c1b.json"
LICENSE_CSV_1B = PREV_WORK / "circuitsage_hmac_v2_2_license_provenance_registry_12c1b.csv"
LICENSE_JSON_1B = PREV_WORK / "circuitsage_hmac_v2_2_license_provenance_registry_12c1b.json"
COMMITMENTS_1B = PREV_WORK / "circuitsage_hmac_v2_2_source_revision_commitments_12c1b.json"
PREFLIGHT_1B = PREV_WORK / "circuitsage_hmac_v2_2_corpus_authorization_preflight_12c1b.json"
REPORT_1B = PREV_WORK / "circuitsage_hmac_v2_2_corpus_authorization_report_12c1b.md"
MANIFEST_1B = RESULT / "circuitsage_hmac_v2_2_corpus_authorization_manifest_12c1b.json"
AUDIT_1B = RESULT / "circuitsage_hmac_v2_2_corpus_license_split_authorization_freeze_12c1b.json"

CHECKPOINT = WORK / "circuitsage_hmac_v2_2_corpus_acquisition_checkpoint_12c1c.json"
FILE_INVENTORY = WORK / "circuitsage_hmac_v2_2_corpus_file_inventory_12c1c.csv"
ACQUIRED_CSV = WORK / "circuitsage_hmac_v2_2_acquired_family_registry_12c1c.csv"
ACQUIRED_JSON = WORK / "circuitsage_hmac_v2_2_acquired_family_registry_12c1c.json"
LICENSE_CSV = WORK / "circuitsage_hmac_v2_2_verified_license_registry_12c1c.csv"
LICENSE_JSON = WORK / "circuitsage_hmac_v2_2_verified_license_registry_12c1c.json"
INTEGRITY = WORK / "circuitsage_hmac_v2_2_corpus_integrity_lock_12c1c.json"
REPORT = WORK / "circuitsage_hmac_v2_2_corpus_acquisition_report_12c1c.md"
MANIFEST = RESULT / "circuitsage_hmac_v2_2_corpus_integrity_manifest_12c1c.json"
AUDIT = RESULT / "circuitsage_hmac_v2_2_rtl_acquisition_license_corpus_freeze_12c1c.json"

PINNED = {
    SOURCE_1B: "e511c25c1b8f5e9551376686977386f2ef9b06872a7e9f7369ddf16be80397d3",
    FAMILY_CSV_1B: "7789aab69b83d846a12252751418d3b11189bc49823e89791d870c470d4e1991",
    LICENSE_CSV_1B: "b2bca6a70bad664c62cdf6127de3fe2e7185b92d745c7782a1175d4fb4d3f8c3",
    ACQUISITION_1B: "a811315801d0532945c72a8f8e2773432a63dede43135f507702b316f256e140",
    MANIFEST_1B: "a1f3734554d87ab2c42daa675ca003916fe9428fc9abae42383e7eeb6ad75727",
    AUDIT_1B: "08d9b0b0835f91948188c40daed46bc0237bed18fedcc25a02035b5c16b4f359",
}

EXPECTED_PARTITIONS = {
    "GENERALIZATION_TRAIN": 3,
    "GENERALIZATION_CALIBRATION": 1,
    "INDEPENDENT_CIRCUIT_TEST": 2,
    "GENERALIZATION_HOLDOUT": 1,
}
FUTURE_BRAND = "Faultiva 1.0 — Hybrid VLSI Fault Intelligence"
MAX_BYTES_PER_FAMILY = 1_073_741_824
RTL_SUFFIXES = {".v", ".sv", ".vh", ".svh"}
SOURCE_SELECTORS = {
    "opentitan_hmac_sha256": ["hw/ip/hmac/rtl", "hw/ip/prim/rtl"],
    "secworks_aes": ["src/rtl"],
    "picorv32_cpu": ["picorv32.v"],
    "secworks_sha256": ["src/rtl"],
    "ibex_cpu": ["rtl", "vendor/lowrisc_ip/ip/prim/rtl"],
    "secworks_chacha": ["src/rtl"],
    "serv_cpu": ["rtl"],
}
LICENSE_MARKERS = {
    "Apache-2.0": ["apache license", "version 2.0"],
    "BSD-2-Clause": ["redistribution and use in source and binary forms", "this software is provided"],
    "ISC": ["permission to use, copy, modify, and/or distribute", "the software is provided"],
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
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def frozen_write(path: Path, payload: bytes) -> None:
    require(not path.exists(), f"refusing to overwrite frozen output: {rel(path)}")
    atomic_write(path, payload)


def record(path: Path) -> dict[str, Any]:
    return {"path": rel(path), "sha256": sha256(path), "bytes": path.stat().st_size}


def csv_bytes(rows: list[dict[str, Any]], fields: list[str]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def safe_manifest_path(value: str) -> Path:
    path = ROOT / value
    resolved = path.resolve()
    require(resolved == ROOT or ROOT in resolved.parents, f"manifest path escapes project: {value}")
    return path


def verify_record(item: Any, expected_path: Path | None = None) -> Path:
    require(isinstance(item, dict), "invalid manifest record")
    require(isinstance(item.get("path"), str), "manifest record path")
    path = safe_manifest_path(item["path"])
    if expected_path is not None:
        require(path.resolve() == expected_path.resolve(), f"unexpected manifest path: {item['path']}")
    require(path.is_file(), f"missing manifest artifact: {item['path']}")
    require(sha256(path) == item.get("sha256"), f"manifest SHA: {item['path']}")
    require(path.stat().st_size == int(item.get("bytes", -1)), f"manifest size: {item['path']}")
    return path


def directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def iter_files(path: Path) -> Iterable[Path]:
    for item in sorted(path.rglob("*"), key=lambda p: p.relative_to(path).as_posix()):
        if item.is_symlink():
            stop(f"symlink prohibited in frozen corpus: {item}")
        if item.is_file():
            yield item


def tree_digest(path: Path) -> tuple[str, list[dict[str, Any]]]:
    digest = hashlib.sha256()
    rows: list[dict[str, Any]] = []
    for item in iter_files(path):
        relative = item.relative_to(path).as_posix()
        file_hash = sha256(item)
        size = item.stat().st_size
        mode = stat.S_IMODE(item.stat().st_mode)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        rows.append({"relative_path": relative, "sha256": file_hash, "bytes": size, "mode": f"{mode:04o}"})
    return digest.hexdigest(), rows


def run_git(arguments: list[str], cwd: Path, log_path: Path) -> str:
    environment = os.environ.copy()
    environment["GIT_TERMINAL_PROMPT"] = "0"
    command = ["git", "-c", "core.hooksPath=/dev/null", "-c", "core.autocrlf=false", *arguments]
    result = subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=1800,
        check=False,
    )
    with log_path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(f"$ {' '.join(command)}\n")
        stream.write(result.stdout)
        if result.stdout and not result.stdout.endswith("\n"):
            stream.write("\n")
        stream.write(f"return_code={result.returncode}\n")
    require(result.returncode == 0, f"git command failed for {cwd.name}; inspect {rel(log_path)}")
    return result.stdout.strip()


def copy_selected(repo: Path, destination: Path, family_id: str, license_path: str) -> list[str]:
    selectors = SOURCE_SELECTORS.get(family_id)
    require(selectors is not None, f"no approved source selector: {family_id}")
    copied: list[str] = []
    for selector in selectors:
        source = repo / selector
        require(source.exists(), f"approved RTL scope missing: {family_id}/{selector}")
        if source.is_symlink():
            stop(f"approved RTL scope is symlink: {family_id}/{selector}")
        if source.is_file():
            targets = [source]
        else:
            targets = list(iter_files(source))
        for item in targets:
            relative = item.relative_to(repo)
            target = destination / "rtl_source" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(item, target)
            os.chmod(target, stat.S_IMODE(item.stat().st_mode))
            copied.append(relative.as_posix())

    provenance_names = {license_path, "README.md", "NOTICE", ".gitmodules"}
    provenance_names.update(item.name for item in repo.glob("*.core") if item.is_file())
    for name in sorted(provenance_names):
        source = repo / name
        if not source.is_file() or source.is_symlink():
            continue
        target = destination / "provenance" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        os.chmod(target, 0o644)
        copied.append(f"provenance/{name}")
    return copied


def license_evidence(path: Path, expected_spdx: str) -> tuple[str, list[str]]:
    require(path.is_file(), f"license file missing: {path}")
    require(path.stat().st_size <= 2_000_000, f"license file unexpectedly large: {path}")
    text = path.read_text(encoding="utf-8", errors="replace").lower()
    markers = LICENSE_MARKERS.get(expected_spdx)
    require(markers is not None, f"unsupported SPDX expectation: {expected_spdx}")
    missing = [marker for marker in markers if marker not in text]
    require(not missing, f"license marker mismatch for {expected_spdx}: {missing}")
    return sha256(path), markers


def scan_spdx(source_root: Path, expected_spdx: str, sealed: bool) -> tuple[list[str], int]:
    if sealed:
        return [], 0
    identifiers: set[str] = set()
    scanned = 0
    pattern = re.compile(r"SPDX-License-Identifier:\s*([^\s*]+)", re.IGNORECASE)
    for path in iter_files(source_root):
        if path.suffix.lower() not in RTL_SUFFIXES or path.stat().st_size > 10_000_000:
            continue
        scanned += 1
        text = path.read_text(encoding="utf-8", errors="replace")
        identifiers.update(match.group(1) for match in pattern.finditer(text))
    conflicting = sorted(value for value in identifiers if value != expected_spdx)
    require(not conflicting, f"conflicting RTL SPDX identifiers: {conflicting}")
    return sorted(identifiers), scanned


def top_module_present(source_root: Path, module_name: str, sealed: bool) -> bool | None:
    if sealed:
        return None
    pattern = re.compile(rf"\bmodule\s+{re.escape(module_name)}\b")
    for path in iter_files(source_root):
        if path.suffix.lower() not in RTL_SUFFIXES or path.stat().st_size > 20_000_000:
            continue
        if pattern.search(path.read_text(encoding="utf-8", errors="replace")):
            return True
    return False


def deterministic_archive(source: Path, destination: Path, root_name: str) -> None:
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with temporary.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
                with tarfile.open(fileobj=zipped, mode="w", format=tarfile.PAX_FORMAT) as archive:
                    for path in iter_files(source):
                        relative = path.relative_to(source).as_posix()
                        info = archive.gettarinfo(str(path), arcname=f"{root_name}/{relative}")
                        info.uid = 0
                        info.gid = 0
                        info.uname = ""
                        info.gname = ""
                        info.mtime = 0
                        info.mode = stat.S_IMODE(path.stat().st_mode)
                        with path.open("rb") as stream:
                            archive.addfile(info, stream)
            raw.flush()
            os.fsync(raw.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def load_checkpoint() -> dict[str, Any]:
    if not CHECKPOINT.exists():
        return {
            "checkpoint_version": "CIRCUITSAGE-HMAC-V2.2-CORPUS-ACQUISITION-CHECKPOINT-12C1C-v1",
            "stage": STAGE,
            "status": "RUNNING",
            "created_at": now(),
            "updated_at": now(),
            "completed": {},
        }
    checkpoint = load_json(CHECKPOINT)
    require(checkpoint.get("stage") == STAGE, "checkpoint stage")
    require(checkpoint.get("status") in {"RUNNING", "PASS / FROZEN"}, "checkpoint status")
    require(isinstance(checkpoint.get("completed"), dict), "checkpoint completed map")
    return checkpoint


def save_checkpoint(checkpoint: dict[str, Any]) -> None:
    checkpoint["updated_at"] = now()
    atomic_write(CHECKPOINT, canonical_json(checkpoint))


def verify_predecessor() -> list[dict[str, Any]]:
    print("FROZEN INPUT VERIFICATION")
    for path, expected in PINNED.items():
        require(path.is_file(), f"missing frozen input: {rel(path)}")
        require(sha256(path) == expected, f"frozen input SHA: {rel(path)}")
        print(f"  {path.name:<108}: OK")

    manifest = load_json(MANIFEST_1B)
    audit = load_json(AUDIT_1B)
    require(manifest.get("status") == "PASS", "12C-1B manifest status")
    require(manifest.get("source_acquisition_authorized") is True, "source acquisition authorization")
    require(manifest.get("dataset_construction_authorized") is False, "dataset boundary")
    require(manifest.get("training_authorized") is False, "training boundary")
    require(manifest.get("evaluation_authorized") is False, "evaluation boundary")
    require(manifest.get("partition_counts") == EXPECTED_PARTITIONS, "partition counts")
    require(audit.get("status") == "PASS", "12C-1B audit status")
    require(audit.get("authorization_status") == "FROZEN", "12C-1B authorization freeze")
    require(audit.get("source_acquisition") == "AUTHORIZED / NOT STARTED", "acquisition state")
    require(audit.get("independent_test_access") == 0, "test access")
    require(audit.get("validation_holdout_access") == [0, 0], "protected access")
    require(audit.get("future_combined_model_brand") == FUTURE_BRAND, "brand reservation")

    outputs = manifest.get("outputs")
    require(isinstance(outputs, dict) and len(outputs) == 10, "12C-1B output registry")
    expected = [SOURCE_POLICY_1B, SPLIT_AUTH_1B, ACQUISITION_1B, FAMILY_CSV_1B,
                FAMILY_JSON_1B, LICENSE_CSV_1B, LICENSE_JSON_1B, COMMITMENTS_1B,
                PREFLIGHT_1B, REPORT_1B]
    for path in expected:
        verify_record(outputs.get(rel(path)), path)

    family_registry = load_json(FAMILY_JSON_1B)
    require(family_registry.get("status") == "FROZEN", "family registry status")
    require(family_registry.get("family_count") == 7, "family count")
    families = family_registry.get("families")
    require(isinstance(families, list) and len(families) == 7, "family rows")
    require({row.get("family_id") for row in families} == set(SOURCE_SELECTORS), "family identity set")
    counts = {name: 0 for name in EXPECTED_PARTITIONS}
    for row in families:
        require(re.fullmatch(r"[0-9a-f]{40}", str(row.get("revision", ""))) is not None, f"revision: {row.get('family_id')}")
        require(row.get("repository", "").startswith("https://github.com/"), f"repository: {row.get('family_id')}")
        require(row.get("partition") in counts, f"partition: {row.get('family_id')}")
        counts[row["partition"]] += 1
    require(counts == EXPECTED_PARTITIONS, "family partition replay")
    return families


def family_is_complete(row: dict[str, Any], completed: dict[str, Any]) -> bool:
    family_id = row["family_id"]
    item = completed.get(family_id)
    if not isinstance(item, dict):
        return False
    source_dir = SOURCE_ROOT / family_id
    archive = ARCHIVE_ROOT / f"{family_id}-{row['revision']}.tar.gz"
    metadata = source_dir / "CORPUS_METADATA.json"
    if not source_dir.is_dir() or not archive.is_file() or not metadata.is_file():
        return False
    tree_hash, _ = tree_digest(source_dir)
    return (
        item.get("tree_sha256") == tree_hash
        and item.get("archive_sha256") == sha256(archive)
        and item.get("revision") == row["revision"]
    )


def acquire_family(row: dict[str, Any], checkpoint: dict[str, Any]) -> None:
    family_id = row["family_id"]
    completed = checkpoint["completed"]
    if family_is_complete(row, completed):
        print(f"  {family_id:<32}: VERIFIED / SKIP")
        return

    repo = FETCH_ROOT / family_id
    destination = SOURCE_ROOT / family_id
    archive = ARCHIVE_ROOT / f"{family_id}-{row['revision']}.tar.gz"
    log_path = LOG_ROOT / f"{family_id}_git.log"
    for target in (repo, destination):
        if target.exists():
            shutil.rmtree(target)
    archive.unlink(missing_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.unlink(missing_ok=True)
    repo.mkdir(parents=True)

    print(f"  {family_id:<32}: FETCH {row['revision'][:12]}")
    run_git(["init", "-q"], repo, log_path)
    run_git(["remote", "add", "origin", row["repository"]], repo, log_path)
    run_git(["fetch", "--quiet", "--depth", "1", "origin", row["revision"]], repo, log_path)
    fetched = run_git(["rev-parse", "FETCH_HEAD"], repo, log_path).splitlines()[-1]
    require(fetched == row["revision"], f"fetched revision mismatch: {family_id}")
    run_git(["checkout", "--quiet", "--detach", "FETCH_HEAD"], repo, log_path)
    head = run_git(["rev-parse", "HEAD"], repo, log_path).splitlines()[-1]
    require(head == row["revision"], f"checked-out revision mismatch: {family_id}")

    license_source = repo / row["license_path"]
    license_hash, markers = license_evidence(license_source, row["spdx_license"])
    destination.mkdir(parents=True)
    copied = copy_selected(repo, destination, family_id, row["license_path"])
    rtl_root = destination / "rtl_source"
    rtl_files = [path for path in iter_files(rtl_root) if path.suffix.lower() in RTL_SUFFIXES]
    require(rtl_files, f"no RTL files selected: {family_id}")
    sealed = row["partition"] == "GENERALIZATION_HOLDOUT"
    spdx_ids, spdx_scanned = scan_spdx(rtl_root, row["spdx_license"], sealed)
    top_present = top_module_present(rtl_root, row["candidate_top"], sealed)
    if not sealed:
        require(top_present is True, f"candidate top module not found: {family_id}/{row['candidate_top']}")

    preliminary_hash, preliminary_rows = tree_digest(destination)
    metadata = {
        "metadata_version": "CIRCUITSAGE-HMAC-V2.2-FAMILY-CORPUS-METADATA-12C1C-v1",
        "stage": STAGE,
        "family_id": family_id,
        "display_name": row["display_name"],
        "partition": row["partition"],
        "repository": row["repository"],
        "revision": row["revision"],
        "spdx_license": row["spdx_license"],
        "license_path": row["license_path"],
        "license_sha256": license_hash,
        "license_markers_verified": markers,
        "selected_scopes": SOURCE_SELECTORS[family_id],
        "selected_files_before_metadata": len(preliminary_rows),
        "selected_tree_sha256_before_metadata": preliminary_hash,
        "rtl_files": len(rtl_files),
        "rtl_spdx_identifiers": spdx_ids,
        "rtl_spdx_files_scanned": spdx_scanned,
        "candidate_top": row["candidate_top"],
        "candidate_top_present": top_present,
        "holdout_sealed": sealed,
        "copied_registry_entries": len(copied),
        "git_submodules_initialized": False,
        "synthesis_or_simulation": False,
    }
    atomic_write(destination / "CORPUS_METADATA.json", canonical_json(metadata))
    tree_hash, file_rows = tree_digest(destination)
    require(sum(item["bytes"] for item in file_rows) <= MAX_BYTES_PER_FAMILY, f"selected corpus exceeds family limit: {family_id}")
    deterministic_archive(destination, archive, family_id)
    require(archive.stat().st_size <= MAX_BYTES_PER_FAMILY, f"archive exceeds family limit: {family_id}")

    completed[family_id] = {
        "family_id": family_id,
        "partition": row["partition"],
        "repository": row["repository"],
        "revision": row["revision"],
        "spdx_license": row["spdx_license"],
        "license_sha256": license_hash,
        "source_directory": rel(destination),
        "file_count": len(file_rows),
        "rtl_file_count": len(rtl_files),
        "source_bytes": sum(item["bytes"] for item in file_rows),
        "tree_sha256": tree_hash,
        "archive": rel(archive),
        "archive_sha256": sha256(archive),
        "archive_bytes": archive.stat().st_size,
        "fetch_log": rel(log_path),
        "fetch_log_sha256": sha256(log_path),
        "top_module_check": "SEALED / NOT READ" if sealed else "PASS",
        "completed_at": now(),
    }
    save_checkpoint(checkpoint)
    shutil.rmtree(repo)
    print(f"  {family_id:<32}: PASS files={len(file_rows)} rtl={len(rtl_files)}")


def finalize(families: list[dict[str, Any]], checkpoint: dict[str, Any]) -> None:
    require(len(checkpoint["completed"]) == 7, "all seven families must complete")
    final_outputs = (FILE_INVENTORY, ACQUIRED_CSV, ACQUIRED_JSON, LICENSE_CSV,
                     LICENSE_JSON, INTEGRITY, REPORT, MANIFEST, AUDIT)
    for path in final_outputs:
        require(not path.exists(), f"final output already exists: {rel(path)}")

    acquired_rows: list[dict[str, Any]] = []
    license_rows: list[dict[str, Any]] = []
    inventory_rows: list[dict[str, Any]] = []
    family_map = {row["family_id"]: row for row in families}
    for family_id in sorted(family_map):
        row = family_map[family_id]
        require(family_is_complete(row, checkpoint["completed"]), f"completed family integrity: {family_id}")
        item = checkpoint["completed"][family_id]
        acquired_rows.append({
            "family_id": family_id,
            "partition": row["partition"],
            "design_class": row["design_class"],
            "repository": row["repository"],
            "revision": row["revision"],
            "spdx_license": row["spdx_license"],
            "file_count": item["file_count"],
            "rtl_file_count": item["rtl_file_count"],
            "source_bytes": item["source_bytes"],
            "tree_sha256": item["tree_sha256"],
            "archive_sha256": item["archive_sha256"],
            "archive_bytes": item["archive_bytes"],
        })
        license_rows.append({
            "family_id": family_id,
            "partition": row["partition"],
            "spdx_license": row["spdx_license"],
            "license_path": row["license_path"],
            "license_sha256": item["license_sha256"],
            "root_license_markers": "PASS",
            "rtl_spdx_review": "SEALED / DEFERRED" if row["partition"] == "GENERALIZATION_HOLDOUT" else "PASS / NO CONFLICTS",
            "redistribution": "CONDITIONALLY ELIGIBLE / FINAL RELEASE REVIEW REQUIRED",
        })
        source_dir = SOURCE_ROOT / family_id
        _, rows = tree_digest(source_dir)
        for file_row in rows:
            inventory_rows.append({"family_id": family_id, **file_row})

    acquired_fields = ["family_id", "partition", "design_class", "repository", "revision", "spdx_license", "file_count", "rtl_file_count", "source_bytes", "tree_sha256", "archive_sha256", "archive_bytes"]
    license_fields = ["family_id", "partition", "spdx_license", "license_path", "license_sha256", "root_license_markers", "rtl_spdx_review", "redistribution"]
    inventory_fields = ["family_id", "relative_path", "sha256", "bytes", "mode"]
    acquired_payload = csv_bytes(acquired_rows, acquired_fields)
    license_payload = csv_bytes(license_rows, license_fields)
    inventory_payload = csv_bytes(inventory_rows, inventory_fields)

    acquired_json = {
        "registry_version": "CIRCUITSAGE-HMAC-V2.2-ACQUIRED-FAMILY-REGISTRY-12C1C-v1",
        "stage": STAGE,
        "status": "FROZEN",
        "families": acquired_rows,
    }
    license_json = {
        "registry_version": "CIRCUITSAGE-HMAC-V2.2-VERIFIED-LICENSE-REGISTRY-12C1C-v1",
        "stage": STAGE,
        "status": "FROZEN",
        "licenses": license_rows,
        "technical_review_only": True,
    }
    aggregate = hashlib.sha256()
    for row in acquired_rows:
        aggregate.update(row["family_id"].encode())
        aggregate.update(b"\0")
        aggregate.update(row["tree_sha256"].encode())
        aggregate.update(b"\0")
        aggregate.update(row["archive_sha256"].encode())
        aggregate.update(b"\0")
    integrity = {
        "lock_version": "CIRCUITSAGE-HMAC-V2.2-CORPUS-INTEGRITY-LOCK-12C1C-v1",
        "stage": STAGE,
        "status": "FROZEN",
        "family_count": len(acquired_rows),
        "partition_counts": EXPECTED_PARTITIONS,
        "aggregate_corpus_sha256": aggregate.hexdigest(),
        "family_tree_sha256": {row["family_id"]: row["tree_sha256"] for row in acquired_rows},
        "family_archive_sha256": {row["family_id"]: row["archive_sha256"] for row in acquired_rows},
        "license_sha256": {row["family_id"]: next(item["license_sha256"] for item in license_rows if item["family_id"] == row["family_id"]) for row in acquired_rows},
        "holdout_family": "serv_cpu",
        "holdout_state": "SEALED / NO RTL SEMANTIC INSPECTION / NO SYNTHESIS",
        "mutation": "PROHIBITED; ANY CHANGE REQUIRES A NEW VERSIONED CORPUS",
    }
    report = f"""# CircuitSage-HMAC V2.2 RTL Corpus Freeze — Stage 12C-1C

Seven repositories were fetched at the exact revisions frozen by Stage 12C-1B.
Only pre-authorized RTL and provenance scopes were retained. Root license
evidence, selected file inventories, deterministic source archives, per-family
tree hashes and an aggregate corpus commitment are frozen.

The HOLDOUT family remains sealed: its bytes were copied and hashed, but its
RTL was not semantically scanned, elaborated, simulated, synthesized, or used
for feature creation. No circuit was synthesized and no fault data, training
data, model predictions, validation results, or HOLDOUT truth were produced.

Licensing status is technical provenance verification, not legal advice. Any
future public release still requires a final license, notice, attribution,
restricted-material and generated-artifact review.

**{FUTURE_BRAND}** remains reserved for the eventual completed hybrid release.
""".encode()

    frozen_write(FILE_INVENTORY, inventory_payload)
    frozen_write(ACQUIRED_CSV, acquired_payload)
    frozen_write(ACQUIRED_JSON, canonical_json(acquired_json))
    frozen_write(LICENSE_CSV, license_payload)
    frozen_write(LICENSE_JSON, canonical_json(license_json))
    frozen_write(INTEGRITY, canonical_json(integrity))
    frozen_write(REPORT, report)

    checkpoint["status"] = "PASS / FROZEN"
    checkpoint["completed_at"] = now()
    checkpoint["aggregate_corpus_sha256"] = integrity["aggregate_corpus_sha256"]
    save_checkpoint(checkpoint)

    outputs = [CHECKPOINT, FILE_INVENTORY, ACQUIRED_CSV, ACQUIRED_JSON,
               LICENSE_CSV, LICENSE_JSON, INTEGRITY, REPORT]
    for row in acquired_rows:
        outputs.append(ARCHIVE_ROOT / f"{row['family_id']}-{row['revision']}.tar.gz")
    manifest = {
        "manifest_version": "CIRCUITSAGE-HMAC-V2.2-CORPUS-INTEGRITY-MANIFEST-12C1C-v1",
        "stage": STAGE,
        "status": "PASS",
        "created_at": now(),
        "frozen_predecessor": {rel(path): record(path) for path in PINNED},
        "outputs": {rel(path): record(path) for path in outputs},
        "families_acquired": 7,
        "licenses_verified": 7,
        "partition_counts": EXPECTED_PARTITIONS,
        "network_fetches": 7,
        "rtl_elaboration_calls": 0,
        "synthesis_calls": 0,
        "simulation_calls": 0,
        "fault_injection_calls": 0,
        "dataset_records_created": 0,
        "models_deserialized": 0,
        "training_calls": 0,
        "inference_calls": 0,
        "independent_test_truth_access": 0,
        "validation_access": 0,
        "holdout_truth_access": 0,
    }
    frozen_write(MANIFEST, canonical_json(manifest))

    audit = {
        "audit_version": "CIRCUITSAGE-HMAC-V2.2-RTL-ACQUISITION-LICENSE-CORPUS-FREEZE-12C1C-v1",
        "stage": STAGE,
        "status": "PASS",
        "acquisition_dataset_status": "COMPLETED / FROZEN",
        "families_revisions": "7/7",
        "license_root_evidence": "PASS 7/7",
        "source_tree_and_archive_hashes": "FROZEN 7/7",
        "partition_counts": EXPECTED_PARTITIONS,
        "holdout": "SEALED / NOT SEMANTICALLY OPENED",
        "rtl_elaboration_synthesis_simulation_fault_injection": [0, 0, 0, 0],
        "dataset_training_inference": [0, 0, 0],
        "independent_test_truth_validation_holdout_truth_access": [0, 0, 0],
        "independent_generalization": "NOT YET EVALUATED",
        "future_combined_model_brand": FUTURE_BRAND,
        "corpus_integrity_lock": record(INTEGRITY),
        "acquired_family_registry": record(ACQUIRED_JSON),
        "verified_license_registry": record(LICENSE_JSON),
        "file_inventory": record(FILE_INVENTORY),
        "report": record(REPORT),
        "manifest": record(MANIFEST),
        "next_gate": "STAGE 12C-1D — MULTI-CIRCUIT ELABORATION, PORTABLE-WRAPPER, AND SYNTHESIS-AUTHORIZATION FREEZE",
    }
    frozen_write(AUDIT, canonical_json(audit))

    for path in (ACQUIRED_JSON, LICENSE_JSON, INTEGRITY, MANIFEST, AUDIT):
        require(canonical_json(load_json(path)) == path.read_bytes(), f"canonical output replay: {path.name}")
    require(ACQUIRED_CSV.read_bytes() == acquired_payload, "acquired CSV replay")
    require(LICENSE_CSV.read_bytes() == license_payload, "license CSV replay")
    require(FILE_INVENTORY.read_bytes() == inventory_payload, "inventory replay")
    for path, expected in PINNED.items():
        require(sha256(path) == expected, f"frozen predecessor modified: {rel(path)}")

    total_files = sum(int(row["file_count"]) for row in acquired_rows)
    total_rtl = sum(int(row["rtl_file_count"]) for row in acquired_rows)
    total_source_bytes = sum(int(row["source_bytes"]) for row in acquired_rows)
    print("\nSTAGE 12C-1C — MULTI-CIRCUIT RTL ACQUISITION, LICENSE VERIFICATION, AND CORPUS-INTEGRITY FREEZE")
    print(f"{'Status':<74}: PASS")
    print(f"{'Acquisition / corpus':<74}: COMPLETED / FROZEN")
    print(f"{'Families / immutable revisions':<74}: 7 / 7")
    print(f"{'TRAIN / CALIBRATION / TEST / HOLDOUT':<74}: 3 / 1 / 2 / 1")
    print(f"{'Root license evidence':<74}: PASS / 7 OF 7")
    print(f"{'Selected files / RTL files / source bytes':<74}: {total_files} / {total_rtl} / {total_source_bytes}")
    print(f"{'Tree hashes / deterministic archives':<74}: FROZEN 7/7 / FROZEN 7/7")
    print(f"{'HOLDOUT':<74}: SEALED / NOT SEMANTICALLY OPENED")
    print(f"{'Elaboration / synthesis / simulation / fault injection':<74}: 0 / 0 / 0 / 0")
    print(f"{'Dataset records / training / inference':<74}: 0 / 0 / 0")
    print(f"{'Independent TEST truth / VALIDATION / HOLDOUT truth access':<74}: 0 / 0 / 0")
    print(f"{'Aggregate corpus SHA':<74}: {integrity['aggregate_corpus_sha256']}")
    print(f"{'Integrity lock':<74}: {INTEGRITY}")
    print(f"{'Integrity lock SHA':<74}: {sha256(INTEGRITY)}")
    print(f"{'Manifest':<74}: {MANIFEST}")
    print(f"{'Manifest SHA':<74}: {sha256(MANIFEST)}")
    print(f"{'Audit':<74}: {AUDIT}")
    print(f"{'Audit SHA':<74}: {sha256(AUDIT)}")
    print(f"{'Next gate':<74}: STAGE 12C-1D — MULTI-CIRCUIT ELABORATION, PORTABLE-WRAPPER, AND SYNTHESIS-AUTHORIZATION FREEZE")


def execute() -> None:
    require(ROOT.name == "vlsi_fault_detection_v2", f"run from project root, not {ROOT}")
    require(shutil.which("git") is not None, "git executable not found")
    if AUDIT.exists():
        stop(f"Stage {STAGE} is already frozen: {rel(AUDIT)}")
    print("STAGE 12C-1C — MULTI-CIRCUIT RTL ACQUISITION")
    families = verify_predecessor()
    checkpoint = load_checkpoint()
    if checkpoint.get("status") == "PASS / FROZEN":
        stop("checkpoint is frozen but final audit is missing; inspect outputs before any repair")
    for row in families:
        acquire_family(row, checkpoint)
    finalize(families, checkpoint)


def status() -> None:
    if AUDIT.is_file():
        audit = load_json(AUDIT)
        print("STAGE 12C-1C — CORPUS-ACQUISITION STATUS")
        print(f"Status                    : {audit.get('status')} / FROZEN")
        print(f"Families                  : {audit.get('families_revisions')}")
        print(f"License evidence          : {audit.get('license_root_evidence')}")
        print(f"Audit                     : {AUDIT}")
        print(f"Audit SHA                 : {sha256(AUDIT)}")
        return
    if not CHECKPOINT.exists():
        print("STAGE 12C-1C — CORPUS-ACQUISITION STATUS")
        print("Status                    : NOT STARTED")
        print("Completed families        : 0/7")
        return
    checkpoint = load_checkpoint()
    print("STAGE 12C-1C — CORPUS-ACQUISITION STATUS")
    print(f"Status                    : {checkpoint.get('status')}")
    print(f"Completed families        : {len(checkpoint.get('completed', {}))}/7")
    print(f"Checkpoint                : {CHECKPOINT}")
    print(f"Checkpoint SHA            : {sha256(CHECKPOINT)}")


def self_test() -> None:
    require(len(SOURCE_SELECTORS) == 7, "selector count")
    require(set(LICENSE_MARKERS) == {"Apache-2.0", "BSD-2-Clause", "ISC"}, "license set")
    require(sum(EXPECTED_PARTITIONS.values()) == 7, "partition arithmetic")
    require(canonical_json({"z": 1, "a": 2}) == b'{"a":2,"z":1}\n', "canonical JSON")
    print("Stage 12C-1C self-test: PASS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--execute", action="store_true")
    action.add_argument("--resume", action="store_true")
    action.add_argument("--status", action="store_true")
    action.add_argument("--self-test", action="store_true")
    arguments = parser.parse_args()
    if arguments.status:
        status()
    elif arguments.self_test:
        self_test()
    else:
        execute()
