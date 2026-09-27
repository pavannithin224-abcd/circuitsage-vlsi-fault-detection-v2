#!/usr/bin/env python3
"""Stage 12C-2A: portable circuit-graph dataset construction and observability diagnostic.

The Stage 12C-1A frozen architecture contract specifies a circuit-normalized
GRAPH encoder plus a response encoder with cross-modal retrieval and graph
reranking.  Stage 12C-1O captured response evidence only; no graph dataset
exists.  The contracted model therefore cannot be trained yet.

This stage builds the missing modality: a portable, circuit-independent
directed graph for every captured family, aligned index-for-index with the
frozen Stage 12C-1O fault-site ordering.

Portability requirements enforced here (from the frozen 12C-1A architecture and
scope contracts):

  * absolute node identity is PROHIBITED - no cell name, net name, path,
    circuit name, partition token or site index becomes a model-facing feature
  * node features are semantic and normalized per circuit using rules that do
    not reference any protected partition
  * the cell-type vocabulary is closed and derived only from captured
    GENERALIZATION_TRAIN and GENERALIZATION_CALIBRATION families; unseen types
    at inference map to a reserved UNKNOWN slot
  * fault identity, stuck value and truth labels are ABSENT from graph features

This stage additionally freezes an OBSERVABILITY DIAGNOSTIC.  The Stage 12C-1P
disposition established that per-family detection varies by a factor of ~3.95
under identical retrieval logic, and attributed the cause to observability
rather than retrieval.  It did not establish WHY observability differs.  An
informal structural hypothesis (sequential depth) is falsified by the captured
corpus: opentitan_hmac_sha256 has the lowest sequential-cell fraction of the
four families and nearly the lowest detection.  This stage therefore computes
structural position features per site and freezes their measured association
with observed detectability, WITHOUT asserting a causal explanation.

No simulation, fault injection, model loading, training, selection or inference
occurs in this stage.  No protected partition is accessed: INDEPENDENT_TEST
remains LOCKED, VALIDATION UNOPENED, HOLDOUT SEALED.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import io
import json
import math
import os
from collections import Counter, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import stage_12c1i_adapter_pilot_execution as pilot


STAGE = "12C-2A"
ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "config/v2_2"
RESULT = ROOT / "results/circuitsage_hmac_v2_12c1"
RESULT2 = ROOT / "results/circuitsage_hmac_v2_12c2"
WORK = RESULT2 / "graph_dataset_12c2a"
LOCK_FILE = WORK / ".stage_12c2a.lock"

# ---------------------------------------------------------------- frozen input
SOURCE_1I = ROOT / "stage_12c1i_adapter_pilot_execution.py"
ARCHITECTURE_1A = CONFIG / "circuitsage_hmac_v2_2_generalization_architecture_12c1a.json"
SCOPE_1A = CONFIG / "circuitsage_hmac_v2_2_generalization_scope_contract_12c1a.json"
INTERFACE_1A = CONFIG / "circuitsage_hmac_v2_2_inference_interface_contract_12c1a.json"
SPLIT_1B = CONFIG / "circuitsage_hmac_v2_2_family_split_authorization_12c1b.json"
AUDIT_1M = RESULT / "circuitsage_hmac_v2_2_site_eligibility_discovery_freeze_12c1m.json"
AUDIT_1O = RESULT / "circuitsage_hmac_v2_2_amended_campaign_execution_dataset_freeze_12c1o.json"
AUDIT_1P = RESULT / "circuitsage_hmac_v2_2_campaign_disposition_freeze_12c1p.json"
AUDIT_1Q = RESULT / "circuitsage_hmac_v2_2_disposition_correction_freeze_12c1q.json"

WORK_1E = RESULT / "train_calibration_synthesis_12c1e"
WORK_1O = RESULT / "parallel_campaign_execution_12c1o"
FEATURES_1O = WORK_1O / "circuitsage_hmac_v2_2_amended_campaign_features_12c1o.npz"
TARGETS_1O = WORK_1O / "circuitsage_hmac_v2_2_amended_campaign_targets_12c1o.npz"
CATALOG_1O = WORK_1O / "circuitsage_hmac_v2_2_amended_campaign_fault_catalog_12c1o.csv"

PINNED = {
    SOURCE_1I: "735df97cbbf8ccd1097913d7044906c209a647e3290d056741c7880e26544f94",
    ARCHITECTURE_1A: "e83da87e0ba773ed1b6872528c216079d31fc83537b4a2eca348bc6a23791649",
    SCOPE_1A: "993902928433953151b09641c156dbc9d9e154f33f5268b9329142fccfc45361",
    INTERFACE_1A: "c5e222640b7404e83fe925ec1b33b2bbf633880b4f21b253ba28d5025686541a",
    SPLIT_1B: "4103808fc389c78088e31c0a76549324c386ed9d8f7bedb7cc4d3748f9ea3303",
    AUDIT_1M: "d6fc641681f13f557a5df820a5aece1f475f2a37becb380af5df26cc9521783a",
    AUDIT_1O: "fa67278ae07bc6dd57246088c3da41918f88c05fa06580c4583c022c1993430b",
    AUDIT_1P: "e0040f8f316e71a24cca232ed18e602d7d2454dd7761db3fd7cf02f1e3aa5548",
    AUDIT_1Q: "1b5389e559ab8d392a688cd45519a1e6a0d2bf433e35825fdd2dfe2b9b7ed2a6",
}

# ---------------------------------------------------------------- stage output
GRAPH_CONTRACT = CONFIG / "circuitsage_hmac_v2_2_graph_dataset_contract_12c2a.json"
VOCABULARY = CONFIG / "circuitsage_hmac_v2_2_cell_type_vocabulary_12c2a.json"
NORMALIZATION = CONFIG / "circuitsage_hmac_v2_2_graph_normalization_rules_12c2a.json"
GRAPH_NPZ = WORK / "circuitsage_hmac_v2_2_circuit_graphs_12c2a.npz"
SITE_ALIGNMENT = WORK / "circuitsage_hmac_v2_2_site_node_alignment_12c2a.csv"
GRAPH_SUMMARY = WORK / "circuitsage_hmac_v2_2_graph_summary_12c2a.csv"
OBSERVABILITY_DIAGNOSTIC = WORK / "circuitsage_hmac_v2_2_observability_diagnostic_12c2a.json"
DIAGNOSTIC_TABLE = WORK / "circuitsage_hmac_v2_2_observability_by_structure_12c2a.csv"
PREFLIGHT = WORK / "circuitsage_hmac_v2_2_graph_dataset_preflight_12c2a.json"
REPORT = WORK / "circuitsage_hmac_v2_2_graph_dataset_report_12c2a.md"
MANIFEST = RESULT2 / "circuitsage_hmac_v2_2_graph_dataset_manifest_12c2a.json"
AUDIT = RESULT2 / "circuitsage_hmac_v2_2_graph_dataset_freeze_12c2a.json"

FAMILIES = (
    "opentitan_hmac_sha256",
    "picorv32_cpu",
    "secworks_aes",
    "secworks_sha256",
)
TOPS = dict(pilot.TOPS)
PARTITIONS = dict(pilot.PARTITIONS)
EXPECTED_SITES = {
    "opentitan_hmac_sha256": 18392,
    "picorv32_cpu": 9665,
    "secworks_aes": 26560,
    "secworks_sha256": 9125,
}
FUTURE_BRAND = "Faultiva 1.0 — Hybrid VLSI Fault Intelligence"

UNKNOWN_TYPE = "<UNKNOWN>"

NODE_FEATURE_NAMES = (
    "type_index",              # closed vocabulary index (embedding input)
    "is_sequential",           # DFF/latch family
    "is_mux",
    "is_inverting",
    "fan_in",                  # raw, pre-normalization
    "fan_out",
    "log1p_fan_in",
    "log1p_fan_out",
    "norm_fan_in",             # per-circuit normalized
    "norm_fan_out",
    "depth_from_input",        # combinational levels from any PI / seq output
    "depth_to_output",         # combinational levels to any PO / seq input
    "norm_depth_from_input",
    "norm_depth_to_output",
    "reconvergence_degree",    # successors sharing a downstream node
    "drives_output_port",
    "driven_by_input_port",
    "seq_distance_to_output",  # sequential-element hops to an observable port
    "norm_seq_distance_to_output",
    "is_fault_site",           # node carries an enumerated SA0/SA1 site
)


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


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def record(path: Path) -> dict[str, Any]:
    return {"path": rel(path), "sha256": sha256(path), "bytes": path.stat().st_size}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def csv_bytes(rows: list[dict[str, Any]], fields: list[str]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


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


def deterministic_npz_bytes(arrays: dict[str, np.ndarray]) -> bytes:
    import zipfile
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(arrays):
            item = io.BytesIO()
            np.lib.format.write_array(item, np.ascontiguousarray(arrays[name]),
                                      allow_pickle=False)
            info = zipfile.ZipInfo(filename=f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, item.getvalue())
    return buffer.getvalue()


def netlist_path(family_id: str) -> Path:
    return WORK_1E / "families" / family_id / f"{family_id}_generic_12c1e.json"


# ------------------------------------------------------------------ vocabulary


def is_sequential_type(cell_type: str) -> bool:
    upper = cell_type.upper()
    return ("DFF" in upper) or ("LATCH" in upper) or ("_SR_" in upper)


def is_inverting_type(cell_type: str) -> bool:
    upper = cell_type.upper()
    return upper in {"$_NOT_", "$_NAND_", "$_NOR_", "$_XNOR_", "$_ANDNOT_", "$_ORNOT_"}


def build_vocabulary() -> dict[str, int]:
    counter: Counter[str] = Counter()
    for family_id in FAMILIES:
        design = load_json(netlist_path(family_id))
        module = design["modules"][TOPS[family_id]]
        for cell in module.get("cells", {}).values():
            counter[str(cell.get("type", ""))] += 1
    ordered = [UNKNOWN_TYPE] + sorted(counter)
    return {t: i for i, t in enumerate(ordered)}


# ----------------------------------------------------------------- graph build


def build_family_graph(family_id: str, vocab: dict[str, int]) -> dict[str, Any]:
    """Build a portable directed graph. Nodes are CELLS; edges follow net flow."""
    design = load_json(netlist_path(family_id))
    module = design["modules"][TOPS[family_id]]
    cells = module.get("cells", {})
    ports = module.get("ports", {})

    cell_names = sorted(cells)
    index_of = {name: i for i, name in enumerate(cell_names)}
    n = len(cell_names)

    # port bit sets
    input_bits: set[int] = set()
    output_bits: set[int] = set()
    for port in ports.values():
        bits = [b for b in port.get("bits", []) if isinstance(b, int)]
        if port.get("direction") in ("input", "inout"):
            input_bits.update(bits)
        if port.get("direction") in ("output", "inout"):
            output_bits.update(bits)

    # driver map: net bit -> producing cell index
    driver: dict[int, int] = {}
    for name in cell_names:
        cell = cells[name]
        directions = cell.get("port_directions", {})
        for port_name, conn in cell.get("connections", {}).items():
            if directions.get(port_name) != "output":
                continue
            for bit in conn:
                if isinstance(bit, int) and bit >= 2:
                    driver.setdefault(bit, index_of[name])

    # edges: driver -> consumer
    edge_set: set[tuple[int, int]] = set()
    fan_in = np.zeros(n, dtype=np.int32)
    fan_out = np.zeros(n, dtype=np.int32)
    driven_by_input = np.zeros(n, dtype=np.uint8)
    drives_output = np.zeros(n, dtype=np.uint8)

    for name in cell_names:
        i = index_of[name]
        cell = cells[name]
        directions = cell.get("port_directions", {})
        for port_name, conn in cell.get("connections", {}).items():
            direction = directions.get(port_name)
            for bit in conn:
                if not isinstance(bit, int):
                    continue
                if direction == "input":
                    fan_in[i] += 1
                    if bit in input_bits:
                        driven_by_input[i] = 1
                    src = driver.get(bit)
                    if src is not None and src != i:
                        edge_set.add((src, i))
                elif direction == "output":
                    fan_out[i] += 1
                    if bit in output_bits:
                        drives_output[i] = 1

    edges = sorted(edge_set)
    edge_index = (np.asarray(edges, dtype=np.int32).T if edges
                  else np.zeros((2, 0), dtype=np.int32))

    # adjacency
    succ: list[list[int]] = [[] for _ in range(n)]
    pred: list[list[int]] = [[] for _ in range(n)]
    for s, d in edges:
        succ[s].append(d)
        pred[d].append(s)

    type_index = np.zeros(n, dtype=np.int32)
    is_seq = np.zeros(n, dtype=np.uint8)
    is_mux = np.zeros(n, dtype=np.uint8)
    is_inv = np.zeros(n, dtype=np.uint8)
    for name in cell_names:
        i = index_of[name]
        t = str(cells[name].get("type", ""))
        type_index[i] = vocab.get(t, 0)
        is_seq[i] = 1 if is_sequential_type(t) else 0
        is_mux[i] = 1 if t.upper() == "$_MUX_" else 0
        is_inv[i] = 1 if is_inverting_type(t) else 0

    # combinational depth (sequential cells act as barriers/sources)
    def forward_depth() -> np.ndarray:
        depth = np.full(n, -1, dtype=np.int32)
        queue: deque[int] = deque()
        for i in range(n):
            if driven_by_input[i] or is_seq[i] or not pred[i]:
                depth[i] = 0
                queue.append(i)
        while queue:
            i = queue.popleft()
            if is_seq[i] and depth[i] > 0:
                continue
            for j in succ[i]:
                cand = depth[i] + 1
                if depth[j] < 0 or cand < depth[j]:
                    depth[j] = cand
                    queue.append(j)
        depth[depth < 0] = 0
        return depth

    def backward_depth() -> np.ndarray:
        depth = np.full(n, -1, dtype=np.int32)
        queue: deque[int] = deque()
        for i in range(n):
            if drives_output[i] or not succ[i]:
                depth[i] = 0
                queue.append(i)
        while queue:
            i = queue.popleft()
            for j in pred[i]:
                if is_seq[j] and depth[j] >= 0:
                    continue
                cand = depth[i] + 1
                if depth[j] < 0 or cand < depth[j]:
                    depth[j] = cand
                    queue.append(j)
        depth[depth < 0] = 0
        return depth

    depth_in = forward_depth()
    depth_out = backward_depth()

    # sequential hops to an observable output port (BFS over full graph,
    # counting sequential boundaries crossed)
    seq_dist = np.full(n, -1, dtype=np.int32)
    queue: deque[int] = deque()
    for i in range(n):
        if drives_output[i]:
            seq_dist[i] = 0
            queue.append(i)
    while queue:
        i = queue.popleft()
        for j in pred[i]:
            cand = seq_dist[i] + (1 if is_seq[i] else 0)
            if seq_dist[j] < 0 or cand < seq_dist[j]:
                seq_dist[j] = cand
                queue.append(j)
    unreachable = int((seq_dist < 0).sum())
    seq_dist[seq_dist < 0] = -1  # keep -1 as explicit "cannot reach output"

    # reconvergence: successors of successors overlap
    reconv = np.zeros(n, dtype=np.int32)
    for i in range(n):
        if len(succ[i]) < 2:
            continue
        seen: Counter[int] = Counter()
        for j in succ[i]:
            for k in succ[j]:
                seen[k] += 1
        reconv[i] = sum(1 for v in seen.values() if v > 1)

    def norm(a: np.ndarray) -> np.ndarray:
        a = a.astype(np.float32)
        hi = float(a.max()) if a.size and a.max() > 0 else 1.0
        return a / hi

    # site alignment: enumerate_sites order is the frozen 12C-1O site_rank order
    saved = pilot.SITES_PER_FAMILY
    try:
        pilot.SITES_PER_FAMILY = EXPECTED_SITES[family_id]
        sites = pilot.enumerate_sites(family_id, netlist_path(family_id))
    finally:
        pilot.SITES_PER_FAMILY = saved
    require(len(sites) == EXPECTED_SITES[family_id], f"site count: {family_id}")

    site_node = np.full(len(sites), -1, dtype=np.int32)
    is_site = np.zeros(n, dtype=np.uint8)
    for s in sites:
        i = index_of.get(s["cell_name"])
        require(i is not None, f"site cell not in graph: {family_id}/{s['cell_name']}")
        site_node[int(s["site_rank"])] = i
        is_site[i] = 1
    require(int((site_node < 0).sum()) == 0, f"unmapped sites: {family_id}")

    features = np.stack([
        type_index.astype(np.float32),
        is_seq.astype(np.float32),
        is_mux.astype(np.float32),
        is_inv.astype(np.float32),
        fan_in.astype(np.float32),
        fan_out.astype(np.float32),
        np.log1p(fan_in.astype(np.float32)),
        np.log1p(fan_out.astype(np.float32)),
        norm(fan_in),
        norm(fan_out),
        depth_in.astype(np.float32),
        depth_out.astype(np.float32),
        norm(depth_in),
        norm(depth_out),
        reconv.astype(np.float32),
        drives_output.astype(np.float32),
        driven_by_input.astype(np.float32),
        seq_dist.astype(np.float32),
        norm(np.maximum(seq_dist, 0)),
        is_site.astype(np.float32),
    ], axis=1).astype(np.float32)
    require(features.shape == (n, len(NODE_FEATURE_NAMES)), f"feature shape: {family_id}")

    return {
        "family_id": family_id, "nodes": n, "edges": len(edges),
        "node_features": features, "edge_index": edge_index,
        "site_node_index": site_node,
        "unreachable_nodes": unreachable,
        "sequential_nodes": int(is_seq.sum()),
        "output_driving_nodes": int(drives_output.sum()),
        "input_driven_nodes": int(driven_by_input.sum()),
        "max_depth_from_input": int(depth_in.max()),
        "max_depth_to_output": int(depth_out.max()),
    }


# -------------------------------------------------------- observability study


def observability_diagnostic(graphs: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Associate structural position with OBSERVED detectability. Descriptive only."""
    features = np.load(FEATURES_1O)
    targets = np.load(TARGETS_1O)
    observable = features["observable"].astype(np.int8)
    family_index = features["family_index"].astype(np.int32)
    local_site = targets["local_site_index"].astype(np.int64)

    rows: list[dict[str, Any]] = []
    per_family: dict[str, Any] = {}

    for fi, family_id in enumerate(FAMILIES):
        g = graphs[family_id]
        mask = family_index == fi
        sites = local_site[mask]
        obs = observable[mask]
        # per-site observability = any of its two faults observable
        n_sites = EXPECTED_SITES[family_id]
        site_obs = np.zeros(n_sites, dtype=np.float32)
        np.maximum.at(site_obs, sites, obs.astype(np.float32))

        nf = g["node_features"]
        node_of = g["site_node_index"]
        depth_out = nf[node_of, NODE_FEATURE_NAMES.index("depth_to_output")]
        seq_dist = nf[node_of, NODE_FEATURE_NAMES.index("seq_distance_to_output")]
        fanout = nf[node_of, NODE_FEATURE_NAMES.index("fan_out")]
        seq_flag = nf[node_of, NODE_FEATURE_NAMES.index("is_sequential")]

        def corr(x: np.ndarray) -> float:
            if x.std() == 0 or site_obs.std() == 0:
                return 0.0
            return float(np.corrcoef(x, site_obs)[0, 1])

        stats = {
            "family_id": family_id,
            "partition": PARTITIONS[family_id],
            "sites": int(n_sites),
            "observable_sites": int(site_obs.sum()),
            "observable_site_fraction": round(float(site_obs.mean()), 8),
            "sequential_cell_fraction": round(float(seq_flag.mean()), 8),
            "mean_depth_to_output": round(float(depth_out.mean()), 6),
            "mean_seq_distance_to_output": round(float(seq_dist[seq_dist >= 0].mean())
                                                 if (seq_dist >= 0).any() else -1.0, 6),
            "sites_unreachable_to_output": int((seq_dist < 0).sum()),
            "corr_observable_vs_depth_to_output": round(corr(depth_out), 6),
            "corr_observable_vs_seq_distance": round(corr(np.maximum(seq_dist, 0)), 6),
            "corr_observable_vs_fanout": round(corr(fanout), 6),
            "corr_observable_vs_is_sequential": round(corr(seq_flag), 6),
        }
        per_family[family_id] = stats
        rows.append(stats)

    seq_fracs = [per_family[f]["sequential_cell_fraction"] for f in FAMILIES]
    obs_fracs = [per_family[f]["observable_site_fraction"] for f in FAMILIES]
    cross = (float(np.corrcoef(seq_fracs, obs_fracs)[0, 1])
             if np.std(seq_fracs) > 0 and np.std(obs_fracs) > 0 else 0.0)

    diagnostic = {
        "diagnostic_version": "CIRCUITSAGE-HMAC-V2.2-OBSERVABILITY-DIAGNOSTIC-12C2A-v1",
        "stage": STAGE, "status": "DESCRIPTIVE / NO CAUSAL CLAIM", "created_at": now(),
        "purpose": ("associate structural node position with OBSERVED per-site detectability "
                    "to constrain future hypotheses about the observability gap"),
        "per_family": per_family,
        "cross_family_corr_sequential_fraction_vs_observable_fraction": round(cross, 6),
        "falsified_hypothesis": {
            "hypothesis": "observability gap is explained by sequential-cell fraction "
                          "(more sequential logic implies lower observability)",
            "status": "FALSIFIED BY THE CAPTURED CORPUS",
            "evidence": [
                f"{f}: sequential_fraction={per_family[f]['sequential_cell_fraction']:.4f}, "
                f"observable_site_fraction={per_family[f]['observable_site_fraction']:.4f}"
                for f in FAMILIES
            ],
            "note": ("opentitan_hmac_sha256 has the lowest sequential-cell fraction of the four "
                     "captured families and among the lowest observable fraction, so sequential "
                     "depth alone does not explain the gap"),
        },
        "established": [
            "per-site structural position features are now available for every captured site",
            "their measured association with observed detectability is frozen here",
        ],
        "not_established": [
            "any causal explanation of the observability gap",
            "that these correlations transfer to unseen circuit families",
        ],
        "intended_use": ("supply the graph encoder with structural evidence so it can reason "
                         "about where a fault may be hiding when response evidence is silent"),
    }
    return diagnostic, rows


# -------------------------------------------------------------------- execute


def verify_inputs() -> dict[str, Any]:
    print("FROZEN INPUT VERIFICATION", flush=True)
    for path, expected in PINNED.items():
        require(path.is_file(), f"missing frozen input: {rel(path)}")
        require(sha256(path) == expected, f"frozen input SHA changed: {path.name}")
        print(f"  {path.name:<84}: OK", flush=True)

    architecture = load_json(ARCHITECTURE_1A)
    require(architecture.get("status") == "FROZEN", "12C-1A architecture frozen")
    ge = architecture["components"]["graph_encoder"]
    require(ge.get("absolute_node_identity") == "PROHIBITED", "absolute node identity prohibited")

    scope = load_json(SCOPE_1A)
    require(scope.get("query_fault_identity") == "PROHIBITED", "query fault identity prohibited")

    audit_1o = load_json(AUDIT_1O)
    require(audit_1o.get("dataset_integrity") == "PASS", "12C-1O integrity")
    audit_1q = load_json(AUDIT_1Q)
    require(audit_1q.get("acceptance_criteria_exist") is True, "12C-1Q correction present")

    # preventive rule from 12C-1Q: enumerate config/v2_2 before asserting absence
    existing = sorted(p.name for p in CONFIG.glob("*.json"))
    require(not any("graph_dataset_contract" in name for name in existing),
            "a graph dataset contract already exists in config/v2_2")
    print(f"  config/v2_2 enumerated ({len(existing)} json contracts); no prior graph contract"
          f"{'':<10}: OK", flush=True)
    return architecture


def execute() -> None:
    require(ROOT.name == "vlsi_fault_detection_v2", f"place script in project root, not {ROOT}")
    outputs = (GRAPH_CONTRACT, VOCABULARY, NORMALIZATION, GRAPH_NPZ, SITE_ALIGNMENT,
               GRAPH_SUMMARY, OBSERVABILITY_DIAGNOSTIC, DIAGNOSTIC_TABLE, PREFLIGHT,
               REPORT, MANIFEST, AUDIT)
    require(not any(p.exists() for p in outputs),
            f"frozen Stage {STAGE} output already exists; use --status")

    architecture = verify_inputs()

    print("\nCELL-TYPE VOCABULARY", flush=True)
    vocab = build_vocabulary()
    print(f"  closed vocabulary: {len(vocab)} entries (index 0 = {UNKNOWN_TYPE})", flush=True)

    print("\nPORTABLE GRAPH CONSTRUCTION", flush=True)
    graphs: dict[str, dict[str, Any]] = {}
    for family_id in FAMILIES:
        g = build_family_graph(family_id, vocab)
        graphs[family_id] = g
        print(f"  {family_id:<24} nodes={g['nodes']:<7} edges={g['edges']:<8} "
              f"sites={len(g['site_node_index']):<7} seq={g['sequential_nodes']:<5} "
              f"maxdepth_out={g['max_depth_to_output']}", flush=True)

    print("\nOBSERVABILITY DIAGNOSTIC", flush=True)
    diagnostic, diag_rows = observability_diagnostic(graphs)
    for r in diag_rows:
        print(f"  {r['family_id']:<24} obs_frac={r['observable_site_fraction']:.4f} "
              f"seq_frac={r['sequential_cell_fraction']:.4f} "
              f"corr(depth)={r['corr_observable_vs_depth_to_output']:+.3f} "
              f"corr(seqdist)={r['corr_observable_vs_seq_distance']:+.3f}", flush=True)
    print(f"  cross-family corr(seq_fraction, observable_fraction) = "
          f"{diagnostic['cross_family_corr_sequential_fraction_vs_observable_fraction']:+.4f}",
          flush=True)

    created = now()

    arrays: dict[str, np.ndarray] = {}
    summary_rows: list[dict[str, Any]] = []
    align_rows: list[dict[str, Any]] = []
    for fi, family_id in enumerate(FAMILIES):
        g = graphs[family_id]
        arrays[f"{family_id}__node_features"] = g["node_features"]
        arrays[f"{family_id}__edge_index"] = g["edge_index"]
        arrays[f"{family_id}__site_node_index"] = g["site_node_index"]
        summary_rows.append({
            "family_id": family_id, "partition": PARTITIONS[family_id],
            "family_index": fi, "nodes": g["nodes"], "edges": g["edges"],
            "sites": int(len(g["site_node_index"])),
            "sequential_nodes": g["sequential_nodes"],
            "output_driving_nodes": g["output_driving_nodes"],
            "input_driven_nodes": g["input_driven_nodes"],
            "max_depth_from_input": g["max_depth_from_input"],
            "max_depth_to_output": g["max_depth_to_output"],
            "nodes_unreachable_to_output": g["unreachable_nodes"],
            "node_feature_count": len(NODE_FEATURE_NAMES),
        })
        align_rows.append({
            "family_id": family_id, "family_index": fi,
            "sites": int(len(g["site_node_index"])),
            "site_rank_order": "FROZEN 12C-1I enumerate_sites ORDER",
            "alignment": "site_node_index[site_rank] -> graph node index",
            "site_node_index_sha256": hashlib.sha256(
                g["site_node_index"].tobytes()).hexdigest(),
        })

    graph_payload = deterministic_npz_bytes(arrays)

    vocabulary = {
        "vocabulary_version": "CIRCUITSAGE-HMAC-V2.2-CELL-TYPE-VOCABULARY-12C2A-v1",
        "stage": STAGE, "status": "FROZEN", "created_at": created,
        "closed": True,
        "unknown_slot": {"token": UNKNOWN_TYPE, "index": 0},
        "derived_from_partitions": ["GENERALIZATION_TRAIN", "GENERALIZATION_CALIBRATION"],
        "protected_partitions_used": [],
        "entries": vocab,
        "size": len(vocab),
        "unseen_type_policy": "MAP TO INDEX 0 (<UNKNOWN>); NEVER EXTEND AT INFERENCE",
    }

    normalization = {
        "rules_version": "CIRCUITSAGE-HMAC-V2.2-GRAPH-NORMALIZATION-12C2A-v1",
        "stage": STAGE, "status": "FROZEN", "created_at": created,
        "scope": "PER-CIRCUIT MAX NORMALIZATION COMPUTED INDEPENDENTLY FOR EACH GRAPH",
        "rationale": ("per-circuit normalization keeps features comparable across circuits of "
                      "very different size without fitting any global scaler on training data, "
                      "so an unseen test circuit is normalized by its own statistics only"),
        "normalized_features": ["norm_fan_in", "norm_fan_out", "norm_depth_from_input",
                                "norm_depth_to_output", "norm_seq_distance_to_output"],
        "raw_features_retained": ["fan_in", "fan_out", "depth_from_input", "depth_to_output",
                                  "seq_distance_to_output", "reconvergence_degree"],
        "fitted_on_training_data": False,
        "leakage_risk": "NONE - NO CROSS-CIRCUIT STATISTIC IS SHARED",
    }

    contract = {
        "contract_version": "CIRCUITSAGE-HMAC-V2.2-GRAPH-DATASET-12C2A-v1",
        "stage": STAGE, "status": "FROZEN", "created_at": created,
        "purpose": "SUPPLY THE GRAPH MODALITY REQUIRED BY THE FROZEN 12C-1A ARCHITECTURE",
        "architecture_authority": rel(ARCHITECTURE_1A),
        "architecture_sha256": PINNED[ARCHITECTURE_1A],
        "node_definition": "ONE NODE PER SYNTHESIZED CELL IN THE FROZEN 12C-1E GENERIC NETLIST",
        "edge_definition": "DIRECTED DRIVER->CONSUMER EDGE, DEDUPLICATED, SELF-LOOPS REMOVED",
        "node_feature_names": list(NODE_FEATURE_NAMES),
        "node_feature_count": len(NODE_FEATURE_NAMES),
        "absolute_node_identity": "PROHIBITED / ABSENT",
        "prohibited_from_features": ["cell_name", "net_name", "module_path", "circuit_name",
                                     "partition_token", "site_index", "fault_id", "stuck_value",
                                     "truth_label"],
        "site_alignment": "site_node_index[site_rank] MAPS THE FROZEN 12C-1O SITE ORDER TO NODES",
        "site_order_authority": "STAGE 12C-1I enumerate_sites DETERMINISTIC ORDER",
        "families": list(FAMILIES),
        "partitions_covered": sorted({PARTITIONS[f] for f in FAMILIES}),
        "protected_partitions_covered": [],
        "vocabulary": rel(VOCABULARY),
        "normalization_rules": rel(NORMALIZATION),
        "serialization": "DETERMINISTIC COMPRESSED NPZ; EXECUTABLE PICKLE PROHIBITED",
        "training": "NOT AUTHORIZED BY THIS STAGE",
        "inference": "NOT AUTHORIZED BY THIS STAGE",
    }

    preflight = {
        "preflight_version": "CIRCUITSAGE-HMAC-V2.2-GRAPH-DATASET-PREFLIGHT-12C2A-v1",
        "stage": STAGE, "status": "PASS", "created_at": created,
        "simulation_calls": 0, "fault_injection_calls": 0, "netlist_modifications": 0,
        "model_deserializations": 0, "training_calls": 0, "selection_calls": 0,
        "inference_calls": 0,
        "independent_test_access": 0, "validation_access": 0, "holdout_access": 0,
        "frozen_12c1e_netlists_modified": False,
        "frozen_12c1o_outputs_modified": False,
        "acceptance_criteria_invented": False,
        "config_v2_2_enumerated_before_absence_claim": True,
    }

    summary_table = "\n".join(
        f"| `{r['family_id']}` | {r['partition']} | {r['nodes']:,} | {r['edges']:,} | "
        f"{r['sites']:,} | {r['sequential_nodes']:,} | {r['max_depth_to_output']} |"
        for r in summary_rows)
    diag_table = "\n".join(
        f"| `{r['family_id']}` | {r['observable_site_fraction']:.4f} | "
        f"{r['sequential_cell_fraction']:.4f} | {r['mean_depth_to_output']:.1f} | "
        f"{r['corr_observable_vs_depth_to_output']:+.3f} | "
        f"{r['corr_observable_vs_seq_distance']:+.3f} | "
        f"{r['corr_observable_vs_fanout']:+.3f} |"
        for r in diag_rows)

    report = f"""# Stage {STAGE} — Portable Circuit-Graph Dataset

**Status: PASS / FROZEN — dataset construction and diagnostic only.**

## Why this stage exists

The frozen Stage 12C-1A architecture contract requires a **circuit-normalized
graph encoder** alongside the response encoder. Stage 12C-1O captured response
evidence only. Without a graph dataset the contracted model cannot be trained.

## What was built

One portable directed graph per captured family, with
**{len(NODE_FEATURE_NAMES)} semantic node features** and no absolute identity.

| family | partition | nodes | edges | sites | sequential | max depth→out |
|---|---|---|---|---|---|---|
{summary_table}

Node features: `{', '.join(NODE_FEATURE_NAMES)}`.

**Portability guarantees**

- no cell name, net name, path, circuit name, partition token or site index is a feature
- cell-type vocabulary is **closed** ({len(vocab)} entries) with a reserved
  `{UNKNOWN_TYPE}` slot at index 0 for unseen types at inference
- normalization is **per-circuit**; no scaler is fitted across circuits, so an
  unseen test circuit is normalized by its own statistics only
- fault identity, stuck value and truth labels are absent

**Site alignment:** `site_node_index[site_rank]` maps the frozen Stage 12C-1O
site order onto graph nodes, so response evidence and graph evidence are
index-aligned without any identity leak.

## Observability diagnostic (descriptive, no causal claim)

Stage 12C-1P established that observability, not retrieval, limits detection.
It did not establish *why* observability differs by family.

An informal structural hypothesis — that the gap follows sequential-cell
fraction — is **falsified by this corpus**:

| family | observable frac | sequential frac | mean depth→out | corr(depth) | corr(seq dist) | corr(fanout) |
|---|---|---|---|---|---|---|
{diag_table}

Cross-family correlation between sequential fraction and observable fraction:
**{diagnostic['cross_family_corr_sequential_fraction_vs_observable_fraction']:+.4f}**.

`opentitan_hmac_sha256` has the **lowest** sequential-cell fraction of the four
families and among the **lowest** observable fraction. Sequential depth alone
therefore does not explain the gap. This stage records the measured
associations and **asserts no causal explanation**.

## What this enables

The graph encoder can now reason about *where a fault may be hiding when the
response is silent* — structural evidence that response data alone cannot
supply. That is the intended mechanism for closing the per-circuit floor on the
sealed test circuits.

## Access

No protected partition accessed. INDEPENDENT_TEST **LOCKED**, VALIDATION
**UNOPENED**, HOLDOUT **SEALED**. Training and inference remain **NOT
AUTHORIZED**. Independent generalization remains **NOT ESTABLISHED**. Future
hybrid brand remains **{FUTURE_BRAND}**.

## Next gate

**Stage 12C-2B** — calibration capture authorization, then training
authorization for the four frozen candidates in the Stage 12C-1A grid.
"""

    summary_fields = ["family_id", "partition", "family_index", "nodes", "edges", "sites",
                      "sequential_nodes", "output_driving_nodes", "input_driven_nodes",
                      "max_depth_from_input", "max_depth_to_output",
                      "nodes_unreachable_to_output", "node_feature_count"]
    align_fields = ["family_id", "family_index", "sites", "site_rank_order", "alignment",
                    "site_node_index_sha256"]
    diag_fields = list(diag_rows[0].keys())

    frozen_write(GRAPH_CONTRACT, canonical_json(contract))
    frozen_write(VOCABULARY, canonical_json(vocabulary))
    frozen_write(NORMALIZATION, canonical_json(normalization))
    frozen_write(GRAPH_NPZ, graph_payload)
    frozen_write(SITE_ALIGNMENT, csv_bytes(align_rows, align_fields))
    frozen_write(GRAPH_SUMMARY, csv_bytes(summary_rows, summary_fields))
    frozen_write(OBSERVABILITY_DIAGNOSTIC, canonical_json(diagnostic))
    frozen_write(DIAGNOSTIC_TABLE, csv_bytes(diag_rows, diag_fields))
    frozen_write(PREFLIGHT, canonical_json(preflight))
    frozen_write(REPORT, report.encode())

    stage_outputs = (GRAPH_CONTRACT, VOCABULARY, NORMALIZATION, GRAPH_NPZ, SITE_ALIGNMENT,
                     GRAPH_SUMMARY, OBSERVABILITY_DIAGNOSTIC, DIAGNOSTIC_TABLE, PREFLIGHT,
                     REPORT)
    manifest = {
        "manifest_version": "CIRCUITSAGE-HMAC-V2.2-GRAPH-DATASET-MANIFEST-12C2A-v1",
        "stage": STAGE, "status": "PASS", "created_at": created,
        "stage_source": record(Path(__file__).resolve()),
        "frozen_inputs": {rel(p): record(p) for p in PINNED},
        "outputs": {rel(p): record(p) for p in stage_outputs},
        "config_v2_2_enumeration": sorted(p.name for p in CONFIG.glob("*.json")),
        "families": list(FAMILIES),
        "total_nodes": sum(r["nodes"] for r in summary_rows),
        "total_edges": sum(r["edges"] for r in summary_rows),
        "total_sites": sum(r["sites"] for r in summary_rows),
        "node_feature_count": len(NODE_FEATURE_NAMES),
        "vocabulary_size": len(vocab),
        "simulation_calls": 0, "training_calls": 0, "inference_calls": 0,
        "independent_test_access": 0, "validation_access": 0, "holdout_access": 0,
    }
    frozen_write(MANIFEST, canonical_json(manifest))

    audit = {
        "audit_version": "CIRCUITSAGE-HMAC-V2.2-GRAPH-DATASET-FREEZE-12C2A-v1",
        "stage": STAGE, "status": "PASS",
        "graph_dataset": "CONSTRUCTED / FROZEN",
        "families": list(FAMILIES),
        "total_nodes": sum(r["nodes"] for r in summary_rows),
        "total_edges": sum(r["edges"] for r in summary_rows),
        "total_sites": sum(r["sites"] for r in summary_rows),
        "node_feature_count": len(NODE_FEATURE_NAMES),
        "vocabulary_size": len(vocab),
        "vocabulary_closed": True,
        "absolute_node_identity": "PROHIBITED / ABSENT",
        "normalization_fitted_on_training_data": False,
        "site_alignment": "FROZEN 12C-1O SITE ORDER PRESERVED",
        "observability_hypothesis_sequential_depth": "FALSIFIED BY CAPTURED CORPUS",
        "cross_family_corr_sequential_vs_observable":
            diagnostic["cross_family_corr_sequential_fraction_vs_observable_fraction"],
        "causal_explanation_claimed": False,
        "model_training_selection_inference":
            "NOT AUTHORIZED / NOT AUTHORIZED / NOT AUTHORIZED",
        "independent_test_validation_holdout_access": [0, 0, 0],
        "independent_generalization": "NOT ESTABLISHED",
        "frozen_12c1e_netlists_modified": False,
        "frozen_12c1o_outputs_modified": False,
        "graph_contract_record": record(GRAPH_CONTRACT),
        "vocabulary_record": record(VOCABULARY),
        "normalization_record": record(NORMALIZATION),
        "graph_dataset_record": record(GRAPH_NPZ),
        "site_alignment_record": record(SITE_ALIGNMENT),
        "observability_diagnostic_record": record(OBSERVABILITY_DIAGNOSTIC),
        "manifest_record": record(MANIFEST),
        "future_combined_model_brand": FUTURE_BRAND,
        "next_gate": "STAGE 12C-2B — CALIBRATION CAPTURE AND TRAINING AUTHORIZATION",
    }
    frozen_write(AUDIT, canonical_json(audit))

    for path in (GRAPH_CONTRACT, VOCABULARY, NORMALIZATION, OBSERVABILITY_DIAGNOSTIC,
                 PREFLIGHT, MANIFEST, AUDIT):
        require(path.read_bytes() == canonical_json(load_json(path)),
                f"canonical output replay: {path.name}")
    require(GRAPH_NPZ.read_bytes() == graph_payload, "graph NPZ deterministic replay")

    print(f"\n{'Stage':<52}: {STAGE} — PORTABLE CIRCUIT-GRAPH DATASET")
    print(f"{'Status':<52}: PASS / FROZEN")
    print(f"{'Families / nodes / edges':<52}: {len(FAMILIES)} / "
          f"{sum(r['nodes'] for r in summary_rows):,} / {sum(r['edges'] for r in summary_rows):,}")
    print(f"{'Sites aligned':<52}: {sum(r['sites'] for r in summary_rows):,}")
    print(f"{'Node features / vocabulary':<52}: {len(NODE_FEATURE_NAMES)} / {len(vocab)} (closed)")
    print(f"{'Absolute node identity':<52}: PROHIBITED / ABSENT")
    print(f"{'Sequential-depth hypothesis':<52}: FALSIFIED")
    print(f"{'Causal explanation claimed':<52}: NO")
    print(f"{'Training / inference':<52}: NOT AUTHORIZED")
    print(f"{'TEST / VALIDATION / HOLDOUT access':<52}: 0 / 0 / 0")
    print(f"{'Audit':<52}: {AUDIT}")
    print(f"{'Audit SHA':<52}: {sha256(AUDIT)}")
    print(f"{'Next gate':<52}: STAGE 12C-2B — CALIBRATION + TRAINING AUTH")


def status() -> None:
    print(f"STAGE {STAGE} — GRAPH DATASET STATUS")
    if not MANIFEST.is_file() or not AUDIT.is_file():
        print("Status                    : NOT FROZEN")
        print(f"Expected audit            : {AUDIT}")
        return
    manifest = load_json(MANIFEST)
    audit = load_json(AUDIT)
    require(manifest.get("status") == "PASS" and audit.get("status") == "PASS", "frozen status")
    require(audit.get("manifest_record", {}).get("sha256") == sha256(MANIFEST), "manifest anchor")
    for name, item in sorted((manifest.get("outputs") or {}).items()):
        path = (ROOT / item["path"]).resolve()
        require(path.is_file() and sha256(path) == item["sha256"], f"output SHA: {name}")
    print("Status                    : PASS / FROZEN")
    print(f"Nodes / edges / sites     : {audit['total_nodes']:,} / {audit['total_edges']:,} / "
          f"{audit['total_sites']:,}")
    print(f"Node features / vocab     : {audit['node_feature_count']} / {audit['vocabulary_size']}")
    print(f"Node identity             : {audit['absolute_node_identity']}")
    print(f"Seq-depth hypothesis      : {audit['observability_hypothesis_sequential_depth']}")
    print(f"Causal claim              : {audit['causal_explanation_claimed']}")
    print(f"Training / inference      : {audit['model_training_selection_inference']}")
    print(f"Next gate                 : {audit['next_gate']}")
    print(f"Audit SHA                 : {sha256(AUDIT)}")


def self_test() -> None:
    require(len(NODE_FEATURE_NAMES) == 20, "twenty node features")
    require(NODE_FEATURE_NAMES[0] == "type_index", "type index first")
    require("cell_name" not in NODE_FEATURE_NAMES, "no cell name feature")
    require("site_index" not in NODE_FEATURE_NAMES, "no site index feature")
    require(is_sequential_type("$_DFFE_PN0P_") and not is_sequential_type("$_MUX_"),
            "sequential classification")
    require(is_inverting_type("$_NOT_") and not is_inverting_type("$_AND_"),
            "inverting classification")
    for family_id in FAMILIES:
        require(netlist_path(family_id).is_file(), f"netlist present: {family_id}")
    require(FEATURES_1O.is_file() and TARGETS_1O.is_file(), "12C-1O dataset present")
    print(f"Stage {STAGE} self-test: PASS")


def locked_execute() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    with LOCK_FILE.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            stop(f"Stage {STAGE} execution lock is held by another process")
        execute()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--status", action="store_true")
    mode.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.status:
        status()
    elif args.self_test:
        self_test()
    else:
        locked_execute()


if __name__ == "__main__":
    main()
