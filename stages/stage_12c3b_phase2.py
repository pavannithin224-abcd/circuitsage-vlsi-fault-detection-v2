#!/usr/bin/env python3
"""Stage 12C-3B phase 2: chacha instrumentation, capture, signature derivation.

Phase 1 (already frozen) synthesized secworks_chacha, enumerated 10,111 sites and
derived the 20,222-fault catalogue plus the 64-vector plan from the frozen 12C-1F
crypto rule.

This phase:
  1. instruments the netlist with runtime fault-injection multiplexers, using the
     SAME mechanism as Stage 12C-1I (per-site output MUX selected by
     AND(fi_enable_i, fi_site_onehot_i[k]), fed by fi_stuck_value_i)
  2. converts to Verilog with the pinned yosys
  3. builds a self-contained testbench and compiles with the pinned Verilator
  4. captures the fault-free baseline, then every fault in parallel batches
  5. derives behaviour signatures and per-circuit metrics
  6. freezes

Prediction-before-truth is preserved: the 12C-2H predicted bands are never read.
serv_cpu is never referenced. ibex_cpu is not opened.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

import stage_12c2c_candidate_training as base
import stage_12c3b_chacha_capture as p1


STAGE = "12C-3B"
FAMILY = p1.FAMILY
TOP = p1.TOP
ROOT = p1.ROOT
WORK = p1.WORK
BUILD = p1.BUILD
RAW = p1.RAW

SITES_CSV = p1.SITE_TABLE
FAULTS_CSV = p1.FAULT_CATALOG
VECTORS_CSV = p1.VECTOR_PLAN
NETLIST = BUILD / f"{FAMILY}_generic_12c3b.json"

INSTRUMENTED_JSON = BUILD / f"{FAMILY}_instrumented_12c3b.json"
INSTRUMENTED_V = BUILD / f"{FAMILY}_instrumented_12c3b.v"
TB = BUILD / f"tb_v22_{FAMILY}_12c3b.sv"
OBJ = BUILD / "obj_dir"
SIM = OBJ / f"Vtb_v22_{FAMILY}_12c3b"

stop, require, now = base.stop, base.require, base.now
canonical_json, sha256, rel = base.canonical_json, base.sha256, base.rel
record, load_json, read_csv, csv_bytes = base.record, base.load_json, base.read_csv, base.csv_bytes
frozen_write, atomic_json = base.frozen_write, base.atomic_json

WORKERS = 10
SITES_PER_BATCH = 256


def max_net_bit(module: dict) -> int:
    m = 1
    for coll in (module.get("ports", {}), module.get("netnames", {})):
        for spec in coll.values():
            for b in spec.get("bits", []):
                if isinstance(b, int):
                    m = max(m, b)
    for cell in module.get("cells", {}).values():
        for bits in cell.get("connections", {}).values():
            for b in bits:
                if isinstance(b, int):
                    m = max(m, b)
    return m


def instrument(sites: list[dict]) -> None:
    """Same mechanism as 12C-1I: one netlist, runtime-selectable single fault."""
    design = copy.deepcopy(load_json(NETLIST))
    module = design["modules"][TOP] if TOP in design["modules"] else \
        next(iter(design["modules"].values()))
    for name in ("fi_enable_i", "fi_site_onehot_i", "fi_stuck_value_i"):
        require(name not in module.get("ports", {}), f"port collision: {name}")

    mx = max_net_bit(module)
    enable_bit, stuck_bit = mx + 1, mx + 2
    onehot = list(range(mx + 3, mx + 3 + len(sites)))
    nxt = mx + 3 + len(sites)

    module.setdefault("ports", {})["fi_enable_i"] = {"direction": "input", "bits": [enable_bit]}
    module["ports"]["fi_stuck_value_i"] = {"direction": "input", "bits": [stuck_bit]}
    module["ports"]["fi_site_onehot_i"] = {"direction": "input", "bits": onehot}
    nn = module.setdefault("netnames", {})
    nn["fi_enable_i"] = {"hide_name": 0, "bits": [enable_bit], "attributes": {}}
    nn["fi_stuck_value_i"] = {"hide_name": 0, "bits": [stuck_bit], "attributes": {}}
    nn["fi_site_onehot_i"] = {"hide_name": 0, "bits": onehot, "attributes": {}}

    cells = module["cells"]
    for s in sites:
        rank = int(s["site_rank"])
        cell = cells[s["cell_name"]]
        conn = cell["connections"][s["output_port"]]
        idx = conn.index(int(s["net_bit"]))
        old = conn[idx]
        raw, gate = nxt, nxt + 1
        nxt += 2
        conn[idx] = raw
        cells[f"$v22fi_and${rank}"] = {
            "hide_name": 1, "type": "$_AND_", "parameters": {}, "attributes": {},
            "port_directions": {"A": "input", "B": "input", "Y": "output"},
            "connections": {"A": [enable_bit], "B": [onehot[rank]], "Y": [gate]}}
        cells[f"$v22fi_mux${rank}"] = {
            "hide_name": 1, "type": "$_MUX_", "parameters": {}, "attributes": {},
            "port_directions": {"A": "input", "B": "input", "S": "input", "Y": "output"},
            "connections": {"A": [raw], "B": [stuck_bit], "S": [gate], "Y": [old]}}

    INSTRUMENTED_JSON.write_bytes(canonical_json(design))
    print(f"  instrumented {len(sites):,} sites -> {rel(INSTRUMENTED_JSON)}", flush=True)


def to_verilog(yosys: Path) -> None:
    log = BUILD / "yosys_write_verilog.log"
    cmd = (f"read_json {INSTRUMENTED_JSON.resolve()}; "
           f"hierarchy -check -top {TOP}; "
           f"write_verilog -noattr {INSTRUMENTED_V.resolve()}")
    with log.open("w") as fh:
        rc = subprocess.run([str(yosys), "-p", cmd], stdout=fh,
                            stderr=subprocess.STDOUT, cwd=str(BUILD)).returncode
    require(rc == 0, f"write_verilog failed; see {rel(log)}")
    require(INSTRUMENTED_V.is_file(), "instrumented verilog produced")
    print(f"  verilog: {INSTRUMENTED_V.stat().st_size/1e6:.1f} MB", flush=True)


def port_widths() -> dict[str, int]:
    j = load_json(NETLIST)
    mod = j["modules"][TOP] if TOP in j["modules"] else next(iter(j["modules"].values()))
    return {p: len(v["bits"]) for p, v in mod["ports"].items()}, \
           {p: v["direction"] for p, v in mod["ports"].items()}


def write_tb(n_sites: int) -> None:
    widths, dirs = port_widths()
    ins = [(p, w) for p, w in widths.items() if dirs[p] == "input"
           and not p.startswith("fi_")]
    outs = [(p, w) for p, w in widths.items() if dirs[p] == "output"]
    clk = next((p for p, _ in ins if "clk" in p.lower()), None)
    rst = next((p for p, _ in ins if "rst" in p.lower() or "reset" in p.lower()), None)
    stim = [(p, w) for p, w in ins if p not in (clk, rst)]

    decl = "\n".join(f"  logic [{w-1}:0] {p};" if w > 1 else f"  logic {p};"
                     for p, w in ins)
    odecl = "\n".join(f"  logic [{w-1}:0] {p};" if w > 1 else f"  logic {p};"
                      for p, w in outs)
    conns = ",\n".join(f"    .{p}({p})" for p, _ in ins + outs)
    drive = "\n".join(
        f"        {p} <= seed[{min(w,32)-1}:0] ^ {w}'(t*{i+7});" for i, (p, w) in enumerate(stim))
    digest = " ^ ".join(f"{{{'32' if w < 32 else str(w)}'(0), {p}}}" if w < 32 else p
                        for p, w in outs) if outs else "32'h0"

    TB.write_text(f"""// Stage 12C-3B testbench for {FAMILY} (generated; upstream RTL untouched)
`timescale 1ns/1ps
module tb_v22_{FAMILY}_12c3b;
  localparam int N_SITES = {n_sites};
  localparam int N_VEC   = 64;

{decl}
{odecl}
  logic fi_enable_i = 1'b0;
  logic [N_SITES-1:0] fi_site_onehot_i = '0;
  logic fi_stuck_value_i = 1'b0;

  {TOP} dut (
{conns},
    .fi_enable_i(fi_enable_i),
    .fi_site_onehot_i(fi_site_onehot_i),
    .fi_stuck_value_i(fi_stuck_value_i)
  );

  int    fout;
  string csv;
  int    site_start, site_count, batch_id;
  int    mode_enable;
  longint unsigned seed, acc;

  always #5 {clk if clk else "dummy_clk"} = ~{clk if clk else "dummy_clk"};

  task automatic run_vectors(input int s_rank, input int s_val);
    for (int t = 0; t < N_VEC; t++) begin
      seed = 64'h9E3779B97F4A7C15 * (t + 1);
{drive}
      repeat (24) @(posedge {clk if clk else "dummy_clk"});
      acc = 0;
      acc = acc ^ longint'({digest});
      $fdisplay(fout, "%0d %0d %0d %0h", s_rank, s_val, t, acc);
    end
  endtask

  initial begin
    if (!$value$plusargs("CSV=%s", csv))        $fatal(1, "Missing +CSV");
    if (!$value$plusargs("SITE_START=%d", site_start)) site_start = -1;
    if (!$value$plusargs("SITE_COUNT=%d", site_count)) site_count = 0;
    if (!$value$plusargs("BATCH_ID=%d", batch_id))     batch_id = 0;
    fout = $fopen(csv, "w");
    if (fout == 0) $fatal(1, "cannot open CSV");

    {rst if rst else "dummy_rst"} = 1'b0;
    repeat (8) @(posedge {clk if clk else "dummy_clk"});
    {rst if rst else "dummy_rst"} = 1'b1;
    repeat (4) @(posedge {clk if clk else "dummy_clk"});

    if (site_start < 0) begin
      fi_enable_i = 1'b0;
      fi_site_onehot_i = '0;
      run_vectors(-1, -1);
    end else begin
      for (int k = 0; k < site_count; k++) begin
        for (int sv = 0; sv < 2; sv++) begin
          fi_enable_i = 1'b1;
          fi_stuck_value_i = sv[0];
          fi_site_onehot_i = '0;
          fi_site_onehot_i[site_start + k] = 1'b1;
          {rst if rst else "dummy_rst"} = 1'b0;
          repeat (4) @(posedge {clk if clk else "dummy_clk"});
          {rst if rst else "dummy_rst"} = 1'b1;
          repeat (2) @(posedge {clk if clk else "dummy_clk"});
          run_vectors(site_start + k, sv);
        end
      end
    end
    $fclose(fout);
    $display("V22_3B_BATCH_RESULT=PASS batch=%0d", batch_id);
    $finish;
  end
endmodule
""")
    print(f"  testbench: {TB.stat().st_size} bytes "
          f"(clk={clk}, rst={rst}, stimulus_ports={len(stim)}, outputs={len(outs)})",
          flush=True)


def build_sim(verilator: Path) -> None:
    log = BUILD / "verilator_build.log"
    cmd = [str(verilator), "--binary", "--timing", "-Wno-fatal", "-Wno-DECLFILENAME",
           "-Wno-PINMISSING", "-Wno-WIDTH", "-Wno-UNOPTFLAT", "-Wno-CASEINCOMPLETE",
           "-Wno-BLKANDNBLK", "-Wno-MULTIDRIVEN", "-Wno-SELRANGE", "-Wno-LATCH",
           "--error-limit", "0", "-j", "10",
           "--Mdir", str(OBJ), "--top-module", f"tb_v22_{FAMILY}_12c3b",
           str(INSTRUMENTED_V), str(TB)]
    t0 = time.time()
    with log.open("w") as fh:
        rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT,
                            cwd=str(BUILD)).returncode
    if rc != 0 or not SIM.is_file():
        tail = log.read_text()[-2500:]
        stop(f"Verilator build failed (rc={rc}).\n{tail}")
    print(f"  simulator built in {time.time()-t0:.0f}s", flush=True)


def run_batch(args):
    bid, start, count = args
    RAW.mkdir(parents=True, exist_ok=True)
    out = RAW / f"batch_{bid:04d}.txt"
    if out.exists() and out.stat().st_size > 0:
        return bid, "CACHED", 0.0
    t0 = time.time()
    r = subprocess.run([str(SIM), f"+CSV={out}", f"+SITE_START={start}",
                        f"+SITE_COUNT={count}", f"+BATCH_ID={bid}"],
                       capture_output=True, text=True, cwd=str(BUILD))
    dt = time.time() - t0
    if r.returncode != 0 or not out.exists():
        (RAW / f"err_{bid:04d}.log").write_text((r.stdout or "") + (r.stderr or ""))
        return bid, f"FAIL rc={r.returncode}", dt
    if "V22_3B_BATCH_RESULT=PASS" not in (r.stdout or ""):
        return bid, "FAIL no-token", dt
    return bid, "OK", dt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=("build", "smoke", "full"), default="build")
    ap.add_argument("--workers", type=int, default=WORKERS)
    a = ap.parse_args()

    require(SITES_CSV.is_file(), "phase 1 must be frozen first")
    sites = read_csv(SITES_CSV)
    print(f"STAGE {STAGE} PHASE 2 — {FAMILY}")
    print(f"  sites={len(sites):,}")

    tools = p1.tools()
    yosys, verilator = tools["yosys"], tools["verilator"]

    if a.phase in ("build", "smoke", "full"):
        if not INSTRUMENTED_JSON.is_file():
            print("\nINSTRUMENTATION")
            instrument(sites)
        if not INSTRUMENTED_V.is_file():
            print("\nWRITE VERILOG (pinned yosys)")
            to_verilog(yosys)
        if not TB.is_file():
            print("\nTESTBENCH")
            write_tb(len(sites))
        if not SIM.is_file():
            print("\nVERILATOR BUILD (pinned)")
            build_sim(verilator)
        print(f"\n  simulator: {SIM}")

    if a.phase == "smoke":
        print("\nSMOKE: baseline + 1 batch of 8 sites")
        t0 = time.time()
        bid, st, dt = run_batch((9000, 0, 8))
        print(f"  batch(8 sites) {st} in {dt:.1f}s")
        f = RAW / "batch_9000.txt"
        if f.is_file():
            n = sum(1 for _ in f.open())
            print(f"  rows={n}  (expect 8 sites x 2 polarities x 64 vectors = 1024)")
            print("  first 2 rows:")
            for i, line in enumerate(f.open()):
                if i >= 2:
                    break
                print(f"    {line.rstrip()}")
            per_fault = dt / 16 if dt else 0
            total = per_fault * len(sites) * 2
            print(f"\n  measured {per_fault:.3f}s/fault")
            print(f"  projected full: {total/60:.0f} min sequential, "
                  f"{total/60/a.workers:.0f} min with {a.workers} workers")


if __name__ == "__main__":
    main()
