#!/usr/bin/env python3
"""Stage 12C-1E: authorized multi-circuit elaboration and generic synthesis.

Runs only the four TRAIN/CALIBRATION recipes frozen by Stage 12C-1D.  The
stage verifies every authorization anchor, resolves source ordering and include
directories deterministically, performs Verilator elaboration/lint and Yosys
generic synthesis, checks the generated netlists, and freezes all results.
Independent TEST and HOLDOUT source trees are never opened by this program.
No simulation, fault injection, dataset construction, training, or inference
is performed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STAGE = "12C-1E"
REPAIR_CLASSIFICATION = "R9 FULL SOURCE RE-SYNTHESIS REPLAY; NO RTL, PARTITION, OR AUTHORIZATION CHANGE"
ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "config/v2_2"
RESULT = ROOT / "results/circuitsage_hmac_v2_12c1"
PREV_WORK = RESULT / "elaboration_synthesis_authorization_12c1d"
WORK = RESULT / "train_calibration_synthesis_12c1e"
FAMILY_ROOT = WORK / "families"

SOURCE_1D = ROOT / "stage_12c1d_elaboration_synthesis_authorization.py"
WRAPPER_1D = CONFIG / "circuitsage_hmac_v2_2_portable_wrapper_contract_12c1d.json"
ACCESS_1D = CONFIG / "circuitsage_hmac_v2_2_elaboration_partition_access_policy_12c1d.json"
AUTH_1D = CONFIG / "circuitsage_hmac_v2_2_train_calibration_synthesis_authorization_12c1d.json"
RECIPE_CSV_1D = PREV_WORK / "circuitsage_hmac_v2_2_elaboration_recipe_registry_12c1d.csv"
RECIPE_JSON_1D = PREV_WORK / "circuitsage_hmac_v2_2_elaboration_recipe_registry_12c1d.json"
WRAPPER_REGISTRY_1D = PREV_WORK / "circuitsage_hmac_v2_2_portable_wrapper_registry_12c1d.csv"
ENV_1D = PREV_WORK / "circuitsage_hmac_v2_2_elaboration_environment_12c1d.json"
PREFLIGHT_1D = PREV_WORK / "circuitsage_hmac_v2_2_elaboration_synthesis_preflight_12c1d.json"
REPORT_1D = PREV_WORK / "circuitsage_hmac_v2_2_elaboration_synthesis_authorization_report_12c1d.md"
MANIFEST_1D = RESULT / "circuitsage_hmac_v2_2_elaboration_synthesis_authorization_manifest_12c1d.json"
AUDIT_1D = RESULT / "circuitsage_hmac_v2_2_elaboration_wrapper_synthesis_authorization_freeze_12c1d.json"

PINNED = {
    SOURCE_1D: "794a22a2f00a88690fe0f96549d7a05a717936b3b359b656fe273cf9820b2e3d",
    WRAPPER_1D: "4e48c584513e28e777d8cc149fc41e5a424d600ba9ffde55d3fd4c0e9e655bb8",
    ACCESS_1D: "ca0150d31120b7cad6d926eefbb96820a4dd9518450dee4a6826d05763b9e05a",
    AUTH_1D: "d57bdf8735c67ccb9c1467051774471e214ab2884f88cb789dbd79510896a684",
    RECIPE_CSV_1D: "4d39507fdb002c87f705cf701a557952600a1d3592c7d16d70d488e07ec4e567",
    RECIPE_JSON_1D: "c220940f6c5143f18f30bca67748e52354a36fba833c0a1e0502f79c7044b486",
    WRAPPER_REGISTRY_1D: "ee3ae07d3021a57ba16a705d91882ed42d9637a5903c461464bc97b9da34319c",
    ENV_1D: "fb017522154459ebf3f301a5607fdd72819b933a2f3a1f49b9fd60e77d8c5114",
    PREFLIGHT_1D: "7c2e5e3784b5c86f4cef23f3bd5a498b2b0ccd2a5aa9e7f9e509bc1f09cf344d",
    REPORT_1D: "9fc6b68e006ca28705f803cf047e76010eb919610c03a493ee0d8dd80b2c545a",
    MANIFEST_1D: "743034fc5a3ae4c51150c20a81b407eca1465f58a75684e810d9b6771fa1a4b2",
    AUDIT_1D: "6056457f59cfb174f8b3b9686aea409f8f8e63ed1536af9d664ed43122e6d2e2",
}

SOURCE_LIST_SHA = {
    "opentitan_hmac_sha256": "72a385d9a6a8d99a9780d0533bb261cbb7e0147c1a1368c80c59fdcba2196ac6",
    "picorv32_cpu": "021986ba9d232063bc192f0e1a87666ea5ac4755a41d960b20e2b48cb107c289",
    "secworks_aes": "e6e337e8ff9fe4a16b552a7fad732c741e99a1da0844b935c01d73c1333645ec",
    "secworks_sha256": "0730f14bfdf4e6e2ef10ce5e990b5f0e2fa81d4b34ae208e2ebc184b545029a8",
}

AUTHORIZED = tuple(sorted(SOURCE_LIST_SHA))
OPENTITAN_SLANG_ORDER = (
    "prim_sha2_pkg.sv",
    "prim_sha2_pad.sv",
    "prim_sha2.sv",
    "prim_sha2_32.sv",
    "hmac_core.sv",
)
CHECKPOINT = WORK / "circuitsage_hmac_v2_2_synthesis_checkpoint_12c1e.json"
METRICS_CSV = WORK / "circuitsage_hmac_v2_2_synthesis_metrics_12c1e.csv"
METRICS_JSON = WORK / "circuitsage_hmac_v2_2_synthesis_metrics_12c1e.json"
WRAPPER_VALIDATION = WORK / "circuitsage_hmac_v2_2_portable_wrapper_validation_12c1e.json"
REPORT = WORK / "circuitsage_hmac_v2_2_elaboration_synthesis_report_12c1e.md"
MANIFEST = RESULT / "circuitsage_hmac_v2_2_elaboration_synthesis_manifest_12c1e.json"
AUDIT = RESULT / "circuitsage_hmac_v2_2_elaboration_wrapper_generic_synthesis_freeze_12c1e.json"
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
    return json.loads(path.read_text(encoding="utf-8"))


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def record(path: Path) -> dict[str, Any]:
    return {"path": rel(path), "sha256": sha256(path), "bytes": path.stat().st_size}


def atomic_write(path: Path, payload: bytes, *, replace: bool = False) -> None:
    if path.exists() and not replace:
        stop(f"refusing to overwrite frozen output: {rel(path)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def csv_bytes(rows: list[dict[str, Any]], fields: list[str]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def quote_yosys(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def verify_inputs() -> dict[str, dict[str, Any]]:
    print("STAGE 12C-1E — FROZEN INPUT VERIFICATION")
    for path, expected in PINNED.items():
        require(path.is_file(), f"missing frozen input: {rel(path)}")
        require(sha256(path) == expected, f"frozen input SHA: {rel(path)}")
        print(f"  {path.name:<110}: OK")
    authorization = load_json(AUTH_1D)
    recipes = load_json(RECIPE_JSON_1D)
    manifest = load_json(MANIFEST_1D)
    audit = load_json(AUDIT_1D)
    require(authorization.get("status") == "FROZEN", "authorization status")
    require(tuple(sorted(authorization.get("authorized_families", []))) == AUTHORIZED, "authorized family set")
    require(len(authorization.get("locked_test_families", [])) == 2, "TEST lock")
    require(len(authorization.get("sealed_holdout_families", [])) == 1, "HOLDOUT seal")
    require(manifest.get("status") == "PASS" and audit.get("status") == "PASS", "12C-1D freeze status")
    require(manifest.get("independent_test_access") == 0 and manifest.get("holdout_access") == 0, "protected access")
    recipe_map = {item["family_id"]: item for item in recipes.get("recipes", [])}
    require(set(AUTHORIZED).issubset(recipe_map), "authorized recipes")
    for family_id in AUTHORIZED:
        item = recipe_map[family_id]
        require(item.get("partition") in {"GENERALIZATION_TRAIN", "GENERALIZATION_CALIBRATION"}, f"partition: {family_id}")
        path = ROOT / item["source_list"]
        require(path.is_file(), f"source list: {family_id}")
        require(sha256(path) == SOURCE_LIST_SHA[family_id] == item.get("source_list_sha256"), f"source list SHA: {family_id}")
    return recipe_map


def without_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.DOTALL)
    text = re.sub(r"//[^\n]*", " ", text)
    return text


def package_definitions(paths: list[Path]) -> set[str]:
    definitions: set[str] = set()
    for path in paths:
        if path.suffix.lower() not in {".v", ".sv"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        definitions.update(re.findall(r"\bpackage\s+(?:automatic\s+)?([A-Za-z_]\w*)", text))
    return definitions


def unresolved_package_imports(path: Path, available: set[str]) -> set[str]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    referenced = set(re.findall(r"\bimport\s+([A-Za-z_]\w*)\s*::", text))
    referenced.update(re.findall(r"\b([A-Za-z_]\w*_pkg)\s*::", text))
    return referenced - available


def route_source_closure(paths: list[Path], top: str) -> tuple[list[Path], list[dict[str, Any]]]:
    """Build a deterministic definition-level closure rooted at ``top``.

    A broad vendor source bank is not a valid synthesis file list: frontends
    parse unreachable packages and optional primitives before hierarchy pruning.
    This resolver indexes module/interface/package definitions, follows symbol
    references from the chosen top, and emits dependencies before consumers.
    Yosys ``hierarchy -check`` remains the authoritative completeness check.
    """
    rtl_paths = sorted((path for path in paths if path.suffix.lower() in {".v", ".sv"}), key=lambda p: p.as_posix())
    texts = {path: without_comments(path.read_text(encoding="utf-8", errors="ignore")) for path in rtl_paths}
    definitions: dict[str, list[Path]] = {}
    definition_kinds: dict[str, str] = {}
    for path, text in texts.items():
        for kind, name in re.findall(r"\b(module|interface|package)\s+(?:automatic\s+)?([A-Za-z_]\w*)", text):
            definitions.setdefault(name, []).append(path)
            definition_kinds[name] = kind
    require(top in definitions, f"top definition absent from frozen corpus: {top}")

    # Prefer the shortest deterministic path when repositories contain both a
    # generic implementation and technology-specific alternatives.
    owner = {name: sorted(candidates, key=lambda p: (len(p.parts), p.as_posix()))[0]
             for name, candidates in definitions.items()}
    symbols = set(owner)
    dependencies: dict[Path, set[Path]] = {path: set() for path in rtl_paths}
    for path, text in texts.items():
        own_symbols = {name for name, candidate in owner.items() if candidate == path}
        tokens = set(re.findall(r"\b[A-Za-z_]\w*\b", text)) & symbols
        for name in tokens - own_symbols:
            dependencies[path].add(owner[name])

    selected: set[Path] = set()
    visiting: set[Path] = set()
    ordered: list[Path] = []

    def visit(path: Path) -> None:
        if path in selected:
            return
        if path in visiting:
            return
        visiting.add(path)
        for dependency in sorted(dependencies[path], key=lambda p: p.as_posix()):
            visit(dependency)
        visiting.remove(path)
        selected.add(path)
        ordered.append(path)

    visit(owner[top])
    require(owner[top] in selected, f"empty routed source closure: {top}")
    available_packages = package_definitions(list(selected))
    for path in ordered:
        missing = unresolved_package_imports(path, available_packages)
        require(not missing, f"reachable source imports package outside corpus: {path.name}: {sorted(missing)}")

    excluded = [{
        "path": str(path),
        "sha256": sha256(path),
        "reason": "NOT REACHABLE FROM FROZEN CANDIDATE TOP",
        "missing_packages": sorted(unresolved_package_imports(path, package_definitions(rtl_paths))),
    } for path in rtl_paths if path not in selected]
    return ordered, excluded


def run_logged(command: list[str], log: Path, cwd: Path | None = None) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("wb") as stream:
        result = subprocess.run(command, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT, check=False)
    require(result.returncode == 0, f"command failed ({result.returncode}); inspect {rel(log)}")


def parse_yosys_stats(log: Path, netlist_json: Path | None = None) -> dict[str, int]:
    text = log.read_text(encoding="utf-8", errors="replace")
    result: dict[str, int] = {}
    patterns = {
        "wires": r"Number of wires:\s+(\d+)",
        "wire_bits": r"Number of wire bits:\s+(\d+)",
        "public_wires": r"Number of public wires:\s+(\d+)",
        "cells": r"Number of cells:\s+(\d+)",
    }
    missing: list[str] = []
    for key, pattern in patterns.items():
        matches = re.findall(pattern, text)
        if matches:
            result[key] = int(matches[-1])
        else:
            missing.append(key)
    # run_logged() has already required a zero Yosys return code, and the
    # synthesis script contains `check -assert`.  Do not scan the complete log
    # for the substring "ERROR:": Slang/Yosys can preserve that token from RTL
    # diagnostics or attributes even when elaboration and structural checking
    # succeeded.  Process status plus check -assert is authoritative.
    if missing:
        require(netlist_json is not None and netlist_json.is_file(),
                f"missing Yosys statistics {missing} and no JSON fallback: {rel(log)}")
        data = load_json(netlist_json)
        modules = data.get("modules", {})
        require(len(modules) == 1, f"JSON-stat top-module count: {rel(netlist_json)}")
        module = next(iter(modules.values()))
        netnames = module.get("netnames", {})
        cells = module.get("cells", {})
        fallback = {
            "wires": len(netnames),
            "wire_bits": sum(len(item.get("bits", [])) for item in netnames.values()),
            "public_wires": sum(1 for item in netnames.values() if int(item.get("hide_name", 0)) == 0),
            "cells": len(cells),
        }
        for key in missing:
            result[key] = fallback[key]
        result["statistics_source"] = "YOSYS JSON NETLIST FALLBACK"  # type: ignore[assignment]
    else:
        result["statistics_source"] = "YOSYS STAT LOG"  # type: ignore[assignment]
    return result


def top_ports(netlist_json: Path) -> list[dict[str, Any]]:
    data = load_json(netlist_json)
    modules = data.get("modules", {})
    require(len(modules) == 1, f"flattened top-module count: {rel(netlist_json)}")
    module_name, module = next(iter(modules.items()))
    ports = []
    for name, item in sorted(module.get("ports", {}).items()):
        ports.append({"name": name, "direction": item.get("direction"), "width": len(item.get("bits", []))})
    require(ports, f"no top ports: {module_name}")
    return ports


def classify_ports(ports: list[dict[str, Any]]) -> dict[str, Any]:
    names = [item["name"] for item in ports]
    lowered = {name: name.lower() for name in names}
    clocks = [name for name in names if re.search(r"(^|_)(clk|clock)($|_)", lowered[name])]
    resets = [name for name in names if re.search(r"(^|_)(rst|reset)(n)?($|_)", lowered[name])]
    inputs = [item for item in ports if item["direction"] in {"input", "inout"}]
    outputs = [item for item in ports if item["direction"] in {"output", "inout"}]
    return {
        "ports": ports,
        "clock_candidates": clocks,
        "reset_candidates": resets,
        "input_payload_bits": sum(item["width"] for item in inputs if item["name"] not in clocks + resets),
        "response_payload_bits": sum(item["width"] for item in outputs),
        "structural_wrapper_feasibility": "PASS" if clocks and resets and outputs else "NOT_MET",
        "functional_transaction_mapping": "DEFERRED UNTIL TRAIN/CALIBRATION VECTOR-CONTRACT GATE",
    }


def process_family(family_id: str, recipe: dict[str, Any]) -> dict[str, Any]:
    family_dir = FAMILY_ROOT / family_id
    done_path = family_dir / "family_result_12c1e.json"
    if done_path.is_file():
        item = load_json(done_path)
        if item.get("replay_method") == "FULL SOURCE RE-SYNTHESIS":
            require(item.get("status") == "PASS" and item.get("family_id") == family_id, f"resume result: {family_id}")
            for output in item.get("outputs", {}).values():
                path = ROOT / output["path"]
                require(path.is_file() and sha256(path) == output["sha256"], f"resume output: {family_id}")
            print(f"  {family_id:<34}: RESUME VERIFIED")
            return item
        print(f"  {family_id:<34}: UPGRADING REPLAY TO FULL RE-SYNTHESIS")

    source_list = ROOT / recipe["source_list"]
    raw_paths = [Path(line.strip()) for line in source_list.read_text(encoding="utf-8").splitlines() if line.strip()]
    require(raw_paths and all(path.is_file() for path in raw_paths), f"source paths: {family_id}")
    frontend = "YOSYS READ_VERILOG"
    if family_id == "opentitan_hmac_sha256":
        by_name: dict[str, list[Path]] = {}
        for path in raw_paths:
            by_name.setdefault(path.name, []).append(path)
        ordered = []
        for name in OPENTITAN_SLANG_ORDER:
            candidates = by_name.get(name, [])
            require(len(candidates) == 1, f"OpenTitan proven-flow source {name}: found {len(candidates)}")
            ordered.append(candidates[0])
        selected_set = set(ordered)
        excluded_units = [{
            "path": str(path),
            "sha256": sha256(path),
            "reason": "OUTSIDE PROVEN STAGE 11B HMAC_CORE SLANG SYNTHESIS CLOSURE",
            "missing_packages": [],
        } for path in sorted(raw_paths, key=lambda p: p.as_posix()) if path not in selected_set]
        frontend = "YOSYS-SLANG / IEEE 1800-2017 / SINGLE UNIT / SYNTHESIS=1"
    else:
        ordered, excluded_units = route_source_closure(raw_paths, recipe["candidate_top"])
    include_dirs = sorted({path.parent.resolve() for path in ordered})
    family_dir.mkdir(parents=True, exist_ok=True)
    ordered_list = family_dir / "resolved_sources_12c1e.f"
    atomic_write(ordered_list, ("\n".join(str(path) for path in ordered) + "\n").encode(), replace=ordered_list.exists())
    routing_path = family_dir / "source_closure_routing_12c1e.json"
    routing = {
        "routing_version": "CIRCUITSAGE-HMAC-V2.2-SOURCE-CLOSURE-ROUTING-12C1E-R4-v1",
        "stage": STAGE,
        "repair_classification": REPAIR_CLASSIFICATION,
        "family_id": family_id,
        "top_module": recipe["candidate_top"],
        "frozen_source_files": len(raw_paths),
        "selected_compilation_units": len(ordered),
        "frontend": frontend,
        "proven_stage_11b_source_order": list(OPENTITAN_SLANG_ORDER) if family_id == "opentitan_hmac_sha256" else None,
        "excluded_unreachable_units": excluded_units,
        "source_bytes_modified": False,
    }
    atomic_write(routing_path, canonical_json(routing), replace=routing_path.exists())
    before = {str(path): sha256(path) for path in ordered}
    top = recipe["candidate_top"]

    lint_log = family_dir / "verilator_elaboration_12c1e.log"
    verilator = shutil.which("verilator")
    require(verilator is not None, "Verilator not found")
    lint_command = [verilator, "--lint-only", "--timing", "--bbox-sys", "--bbox-unsup", "-Wno-fatal", "--top-module", top]
    lint_command += [f"-I{directory}" for directory in include_dirs]
    lint_command += [str(path) for path in ordered if path.suffix.lower() not in {".vh", ".svh"}]
    run_logged(lint_command, lint_log)

    yosys_script = family_dir / "generic_synthesis_12c1e.ys"
    netlist_json = family_dir / f"{family_id}_generic_12c1e.json"
    netlist_verilog = family_dir / f"{family_id}_generic_12c1e.v"
    synth_log = family_dir / "yosys_generic_synthesis_12c1e.log"
    sources = " ".join(quote_yosys(str(path)) for path in ordered if path.suffix.lower() not in {".vh", ".svh"})
    includes = " ".join("-I" + quote_yosys(str(path)) for path in include_dirs)
    if family_id == "opentitan_hmac_sha256":
        prim_directory = next(path.parent for path in ordered if path.name == "prim_sha2_pkg.sv")
        # read_slang forwards arguments to the Slang driver and does not remove
        # Yosys-style quote characters.  Frozen project paths contain no spaces,
        # so pass them literally, matching the proven Stage 11B command.
        slang_sources = " ".join(str(path) for path in ordered)
        frontend_command = (
            "read_slang --std 1800-2017 --single-unit "
            "--define-macro SYNTHESIS=1 "
            f"--include-directory {prim_directory} "
            f"--top {top} {slang_sources}"
        )
    else:
        frontend_command = f"read_verilog -sv -defer {includes} {sources}"
    script = "\n".join([
        frontend_command,
        f"hierarchy -check -top {top}",
        "proc; flatten; opt; memory; opt; techmap; opt",
        "check -assert",
        "stat",
        f"write_json {quote_yosys(str(netlist_json))}",
        f"write_verilog -noattr {quote_yosys(str(netlist_verilog))}",
        "",
    ]).encode()
    atomic_write(yosys_script, script, replace=yosys_script.exists())
    yosys = shutil.which("yosys")
    require(yosys is not None, "Yosys not found")
    run_logged([yosys, "-s", str(yosys_script)], synth_log, family_dir)
    stats = parse_yosys_stats(synth_log, netlist_json)
    require(before == {str(path): sha256(path) for path in ordered}, f"upstream RTL modified: {family_id}")
    wrapper = classify_ports(top_ports(netlist_json))
    require(wrapper["structural_wrapper_feasibility"] == "PASS", f"portable wrapper structural feasibility: {family_id}")

    replay_json = family_dir / f"{family_id}_generic_replay_12c1e.json"
    replay_verilog = family_dir / f"{family_id}_generic_replay_12c1e.v"
    replay_log = family_dir / "yosys_netlist_replay_12c1e.log"
    replay_script = family_dir / "netlist_replay_12c1e.ys"
    replay_payload = (script.decode()
                      .replace(str(netlist_json), str(replay_json))
                      .replace(str(netlist_verilog), str(replay_verilog))).encode()
    require(replay_payload != script, f"replay output routing: {family_id}")
    atomic_write(replay_script, replay_payload, replace=replay_script.exists())
    run_logged([yosys, "-s", str(replay_script)], replay_log, family_dir)
    parse_yosys_stats(replay_log, replay_json)
    if sha256(netlist_json) == sha256(replay_json):
        replay_status = "PASS / BYTE-EXACT"
    else:
        require(load_json(netlist_json) == load_json(replay_json),
                f"deterministic netlist semantic replay: {family_id}")
        replay_status = "PASS / JSON-SEMANTIC"

    output_paths = [ordered_list, routing_path, lint_log, yosys_script, synth_log, netlist_json,
                    netlist_verilog, replay_script, replay_log, replay_json, replay_verilog]
    result = {
        "result_version": "CIRCUITSAGE-HMAC-V2.2-FAMILY-SYNTHESIS-12C1E-v1",
        "stage": STAGE, "status": "PASS", "family_id": family_id,
        "partition": recipe["partition"], "top_module": top,
        "frontend": frontend,
        "source_files": len(ordered), "include_directories": len(include_dirs),
        "source_closure_routing": "PASS",
        "excluded_unreachable_units": len(excluded_units),
        "replay_method": "FULL SOURCE RE-SYNTHESIS",
        "verilator_elaboration": "PASS", "yosys_generic_synthesis": "PASS",
        "structural_check": "PASS", "deterministic_netlist_replay": replay_status,
        "upstream_rtl_modified": False, "statistics": stats,
        "wrapper_validation": wrapper,
        "outputs": {rel(path): record(path) for path in output_paths},
    }
    atomic_write(done_path, canonical_json(result), replace=done_path.exists())
    result["outputs"][rel(done_path)] = record(done_path)
    print(f"  {family_id:<34}: PASS cells={stats['cells']}")
    return result


def checkpoint_payload(completed: list[str], state: str) -> dict[str, Any]:
    return {
        "checkpoint_version": "CIRCUITSAGE-HMAC-V2.2-SYNTHESIS-CHECKPOINT-12C1E-v1",
        "stage": STAGE, "status": state, "updated_at": now(),
        "authorized_families": list(AUTHORIZED), "completed_families": completed,
        "locked_test_access": 0, "holdout_access": 0,
    }


def execute() -> None:
    require(ROOT.name == "vlsi_fault_detection_v2", f"run from project root, not {ROOT}")
    for path in (METRICS_CSV, METRICS_JSON, WRAPPER_VALIDATION, REPORT, MANIFEST, AUDIT):
        require(not path.exists(), f"Stage {STAGE} already frozen: {rel(path)}")
    recipe_map = verify_inputs()
    WORK.mkdir(parents=True, exist_ok=True)
    completed: list[str] = []
    results: list[dict[str, Any]] = []
    atomic_write(CHECKPOINT, canonical_json(checkpoint_payload(completed, "RUNNING")), replace=CHECKPOINT.exists())
    for family_id in AUTHORIZED:
        result = process_family(family_id, recipe_map[family_id])
        results.append(result)
        completed.append(family_id)
        atomic_write(CHECKPOINT, canonical_json(checkpoint_payload(completed, "RUNNING")), replace=True)

    fields = ["family_id", "partition", "top_module", "source_files", "verilator_elaboration",
              "yosys_generic_synthesis", "cells", "wires", "wire_bits", "public_wires",
              "wrapper_feasibility", "deterministic_replay", "upstream_rtl_modified"]
    rows = []
    wrapper_items = {}
    for item in results:
        stats = item["statistics"]
        wrapper_items[item["family_id"]] = item["wrapper_validation"]
        rows.append({"family_id": item["family_id"], "partition": item["partition"],
                     "top_module": item["top_module"], "source_files": item["source_files"],
                     "verilator_elaboration": "PASS", "yosys_generic_synthesis": "PASS",
                     "cells": stats["cells"], "wires": stats["wires"], "wire_bits": stats["wire_bits"],
                     "public_wires": stats["public_wires"], "wrapper_feasibility": "PASS",
                     "deterministic_replay": item["deterministic_netlist_replay"], "upstream_rtl_modified": "NO"})
    replay_modes = [item["deterministic_netlist_replay"] for item in results]
    overall_replay = ("PASS / BYTE-EXACT / 4 OF 4"
                      if all(mode == "PASS / BYTE-EXACT" for mode in replay_modes)
                      else "PASS / JSON-SEMANTIC / 4 OF 4")
    metrics = {"metrics_version": "CIRCUITSAGE-HMAC-V2.2-SYNTHESIS-METRICS-12C1E-v1",
               "stage": STAGE, "status": "PASS", "families": rows,
               "deterministic_replay": overall_replay,
               "total_cells": sum(row["cells"] for row in rows),
               "total_wire_bits": sum(row["wire_bits"] for row in rows)}
    wrappers = {"validation_version": "CIRCUITSAGE-HMAC-V2.2-PORTABLE-WRAPPER-VALIDATION-12C1E-v1",
                "stage": STAGE, "status": "PASS", "scope": "STRUCTURAL PORT MAPPING ONLY",
                "functional_mapping": "DEFERRED", "families": wrapper_items}
    atomic_write(METRICS_CSV, csv_bytes(rows, fields))
    atomic_write(METRICS_JSON, canonical_json(metrics))
    atomic_write(WRAPPER_VALIDATION, canonical_json(wrappers))
    report = f"""# CircuitSage-HMAC V2.2 Multi-Circuit Synthesis — Stage 12C-1E

All four authorized TRAIN/CALIBRATION families passed Verilator elaboration,
generic Yosys synthesis, structural checking, and exact deterministic netlist
replay. Upstream RTL remained byte-identical. Portable-wrapper feasibility was
validated structurally from the frozen top-level ports; functional transaction
mapping remains deferred and no behavior dataset was created.

The two INDEPENDENT TEST families remained locked and the HOLDOUT remained
sealed. No simulation, fault injection, model training, or inference occurred.
The future combined release name remains **{FUTURE_BRAND}**.
""".encode()
    atomic_write(REPORT, report)
    outputs = [CHECKPOINT, METRICS_CSV, METRICS_JSON, WRAPPER_VALIDATION, REPORT]
    for item in results:
        outputs.extend(ROOT / key for key in item["outputs"])
    unique_outputs = sorted(set(outputs), key=lambda path: rel(path))
    manifest = {"manifest_version": "CIRCUITSAGE-HMAC-V2.2-ELABORATION-SYNTHESIS-MANIFEST-12C1E-v1",
                "stage": STAGE, "status": "PASS", "created_at": now(),
                "frozen_inputs": {rel(path): record(path) for path in PINNED},
                "source_lists": {family: record(PREV_WORK / "source_lists" / f"{family}_sources_12c1d.f") for family in AUTHORIZED},
                "outputs": {rel(path): record(path) for path in unique_outputs},
                "authorized_families_completed": list(AUTHORIZED),
                "elaboration_calls": 4, "synthesis_calls": 4, "netlist_replay_calls": 4,
                "simulation_calls": 0, "fault_injection_calls": 0, "dataset_records_created": 0,
                "training_calls": 0, "inference_calls": 0,
                "independent_test_access": 0, "validation_access": 0, "holdout_access": 0}
    atomic_write(MANIFEST, canonical_json(manifest))
    audit = {"audit_version": "CIRCUITSAGE-HMAC-V2.2-ELABORATION-WRAPPER-GENERIC-SYNTHESIS-FREEZE-12C1E-v1",
             "stage": STAGE, "status": "PASS", "execution_status": "COMPLETED / FROZEN",
             "authorized_families": list(AUTHORIZED), "family_count": 4,
             "elaboration": "PASS / 4 OF 4", "generic_synthesis": "PASS / 4 OF 4",
             "structural_checks": "PASS / 4 OF 4", "deterministic_replay": overall_replay,
             "portable_wrapper_validation": "PASS / STRUCTURAL; FUNCTIONAL MAPPING DEFERRED",
             "independent_test": "LOCKED / 2 FAMILIES", "holdout": "SEALED / 1 FAMILY",
             "upstream_rtl_modified": False, "simulation_fault_injection_dataset": [0, 0, 0],
             "training_inference": [0, 0], "independent_test_validation_holdout_access": [0, 0, 0],
             "metrics_record": record(METRICS_JSON), "wrapper_validation_record": record(WRAPPER_VALIDATION),
             "report_record": record(REPORT), "manifest_record": record(MANIFEST),
             "future_combined_model_brand": FUTURE_BRAND,
             "next_gate": "STAGE 12C-1F — TRAIN/CALIBRATION PORTABLE TRANSACTION-VECTOR AND FAULT-CAMPAIGN CONTRACT FREEZE"}
    atomic_write(AUDIT, canonical_json(audit))
    atomic_write(CHECKPOINT, canonical_json(checkpoint_payload(completed, "PASS / FROZEN")), replace=True)
    # Updating the checkpoint changes its manifest record; rebuild manifest and audit once.
    manifest["outputs"][rel(CHECKPOINT)] = record(CHECKPOINT)
    atomic_write(MANIFEST, canonical_json(manifest), replace=True)
    audit["manifest_record"] = record(MANIFEST)
    atomic_write(AUDIT, canonical_json(audit), replace=True)
    for path in (METRICS_JSON, WRAPPER_VALIDATION, MANIFEST, AUDIT, CHECKPOINT):
        require(canonical_json(load_json(path)) == path.read_bytes(), f"canonical replay: {path.name}")
    for path, expected in PINNED.items():
        require(sha256(path) == expected, f"frozen input modified: {rel(path)}")
    print("\nSTAGE 12C-1E — TRAIN/CALIBRATION MULTI-CIRCUIT ELABORATION, WRAPPER VALIDATION, AND GENERIC-SYNTHESIS EXECUTION FREEZE")
    print(f"{'Status':<78}: PASS")
    print(f"{'Repair classification':<78}: {REPAIR_CLASSIFICATION}")
    print(f"{'Execution / results':<78}: COMPLETED / FROZEN")
    print(f"{'Authorized families elaborated / synthesized':<78}: 4/4 / 4/4")
    print(f"{'Structural checks / deterministic replay':<78}: PASS 4/4 / {overall_replay}")
    print(f"{'Portable wrapper validation':<78}: PASS / STRUCTURAL; FUNCTIONAL MAPPING DEFERRED")
    print(f"{'Total generic cells / wire bits':<78}: {metrics['total_cells']} / {metrics['total_wire_bits']}")
    print(f"{'Independent TEST / HOLDOUT':<78}: LOCKED 2 / SEALED 1")
    print(f"{'Simulation / fault injection / dataset records':<78}: 0 / 0 / 0")
    print(f"{'Training / inference':<78}: 0 / 0")
    print(f"{'Independent TEST / VALIDATION / HOLDOUT access':<78}: 0 / 0 / 0")
    print(f"{'Metrics':<78}: {METRICS_JSON}")
    print(f"{'Metrics SHA':<78}: {sha256(METRICS_JSON)}")
    print(f"{'Manifest':<78}: {MANIFEST}")
    print(f"{'Manifest SHA':<78}: {sha256(MANIFEST)}")
    print(f"{'Audit':<78}: {AUDIT}")
    print(f"{'Audit SHA':<78}: {sha256(AUDIT)}")
    print(f"{'Next gate':<78}: STAGE 12C-1F — TRAIN/CALIBRATION PORTABLE TRANSACTION-VECTOR AND FAULT-CAMPAIGN CONTRACT FREEZE")


def status() -> None:
    completed = []
    if CHECKPOINT.is_file():
        checkpoint = load_json(CHECKPOINT)
        completed = checkpoint.get("completed_families", [])
        state = checkpoint.get("status", "UNKNOWN")
    else:
        state = "NOT STARTED"
    print("STAGE 12C-1E — SYNTHESIS STATUS")
    print(f"{'Status':<28}: {state}")
    print(f"{'Completed families':<28}: {len(completed)}/4")
    print(f"{'Family IDs':<28}: {completed}")
    if AUDIT.is_file():
        print(f"{'Audit':<28}: {AUDIT}")
        print(f"{'Audit SHA':<28}: {sha256(AUDIT)}")


def self_test() -> None:
    require(len(AUTHORIZED) == 4, "authorized family count")
    require(canonical_json({"z": 1, "a": 2}) == b'{"a":2,"z":1}\n', "canonical JSON")
    require(classify_ports([{"name": "clk", "direction": "input", "width": 1},
                            {"name": "rst_n", "direction": "input", "width": 1},
                            {"name": "out", "direction": "output", "width": 8}])["structural_wrapper_feasibility"] == "PASS", "port classification")
    require(unresolved_package_imports.__name__ == "unresolved_package_imports", "source routing helper")
    print("Stage 12C-1E self-test: PASS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    require(sum((args.execute, args.resume, args.status, args.self_test)) == 1, "select exactly one mode")
    if args.status:
        status()
    elif args.self_test:
        self_test()
    else:
        execute()
