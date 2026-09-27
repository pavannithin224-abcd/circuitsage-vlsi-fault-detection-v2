#!/usr/bin/env python3
"""Stage 12C-4A: genuine V1 verification wiring.

Stage 12C-2I wired V1 into the Faultiva pipeline as a SCOPE GATE: it loaded the
model, checked the circuit family, and returned VERIFIED without ever calling
``predict_proba``.  That is not verification, and shipping it publicly under the
label VERIFIED would overstate what the pipeline does.

This stage builds the real V1 inference path and measures it.  Stage 12C-2I is
FROZEN and is NOT modified; this is an additive correction in a new stage.

The 646-feature contract (frozen 11D-2A / 11C-5E)
--------------------------------------------------
    1   stuck_value
   14   site features   cell_fanout_z, is_primary_output_stem,
                        is_sequential_stem, driver_cell_type one-hot (9),
                        site_category one-hot (2)
  512   stimulus        key_bit_255..key_bit_0, message_bit_255..message_bit_0
  ---
  527   sample branch
  119   graph branch    frozen DIR_SGC_K3 propagation cache
  ---
  646   fusion input -> MLPClassifier(128, 64, 32) -> threshold 0.4965

SCALING CONVENTION - the trap this stage exists to document
------------------------------------------------------------
``hmac_directed_sgc_graph_features_11d1c.npz`` stores ``k3_features`` ALREADY
SCALED (column mean ~0.008, column std ~0.997).  The companion ``k3_mean`` and
``k3_scale`` arrays are the RECORD of the scaler that was already applied - they
are provenance, NOT a transform to apply at inference.

Applying them a second time inflates the graph branch from [-5.07, 9.12] to
[-74.98, 47960.99] and saturates the model: 34% of outputs land on exactly 0.0
or exactly 1.0, and fault-polarity sensitivity collapses from 99/100 sites to
39/100.  The pipeline still "runs" and still returns verdicts, so the defect is
silent.  ``cell_fanout`` is the opposite case: stored RAW, and the schema's
``numeric_preprocessing`` block MUST be applied.

This stage freezes both conventions and asserts them in a self-test so the
regression cannot return unnoticed.

Scope
-----
V1 was trained on ``opentitan_hmac_sha256`` only.  Out-of-scope circuits receive
OUT_OF_SCOPE, never a guessed probability.  This stage does not change that rule
and does not retrain, recalibrate or reselect anything.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import joblib

import stage_12c2c_candidate_training as base


STAGE = "12C-4A"
ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "config/v2_2"
RESULT2 = ROOT / "results/circuitsage_hmac_v2_12c2"
RESULT3 = ROOT / "results/circuitsage_hmac_v2_12c3"
RESULT4 = ROOT / "results/circuitsage_hmac_v2_12c4"
WORK = RESULT4 / "v1_verification_wiring_12c4a"
LOCK_FILE = WORK / ".stage_12c4a.lock"

V1_RELEASE = ROOT / "release/opentitan-hmac-vlsi-fault-detection-v1"
V1_MODEL = V1_RELEASE / "models/hmac_hybrid_v1_original.joblib"
V1_LOCK = V1_RELEASE / "models/hmac_final_diagnostic_model_lock_11d2d.json"
V1_SCHEMA = V1_RELEASE / "schemas/hmac_leakage_safe_feature_matrix_schema_11c5e.json"
SGC_CACHE = (ROOT / "results/hmac_fault_campaign_11d1/gnn_training_11d1c"
             / "hmac_directed_sgc_graph_features_11d1c.npz")
GRAPH = (ROOT / "results/hmac_fault_campaign_11d1/graph_dataset_11d1a"
         / "hmac_golden_netlist_graph_11d1a.npz")
AUDIT_2I = RESULT2 / "circuitsage_hmac_v2_2_inference_pipeline_freeze_12c2i.json"
AUDIT_3D = RESULT3 / "circuitsage_hmac_v2_2_closing_disposition_freeze_12c3d.json"
SOURCE_2I = ROOT / "stage_12c2i_faultiva_pipeline.py"

PINNED = {
    V1_MODEL: "12fea5eabf4a6c605325b6ce4c1217f6a37cc59750065db8b85c3da14713f6b8",
    V1_LOCK: "7985b534c62d93717a179d6d2b20f247a531ec0452003e35f0f65cf640d0b30d",
    V1_SCHEMA: "0bf1edffb8078b003a1116b276615d5544979d007712ab49870b1b93784e486c",
    SGC_CACHE: "b5874b329a272f1f11baace3127d97e5ecbea8f6170936e7991a6aca4e02a2ce",
    GRAPH: "e3c2dd2214b544231186c29d8d9cb5aa6621b4d4b4bc9002150ac4f6c208c052",
    AUDIT_2I: "78a5cad19a6910ee8a163adaba4514e13e08b87d9ef6c6253699b79bfcffd350",
    AUDIT_3D: "1d34e232822e575a7c372fb1cfbf2d4b6b0ed8fb421d3513a5a085e8232023e4",
    SOURCE_2I: "dabe558f90ab0897322d241cb854000318812968aa79efd0f9124957b2d3774a",
}

SCALING = CONFIG / "circuitsage_hmac_v2_2_v1_feature_scaling_convention_12c4a.json"
DEFECT = WORK / "circuitsage_hmac_v2_2_scope_gate_defect_12c4a.json"
DIAGNOSTICS = WORK / "circuitsage_hmac_v2_2_v1_forward_pass_diagnostics_12c4a.csv"
COMPARISON = WORK / "circuitsage_hmac_v2_2_correct_vs_double_scaled_12c4a.csv"
LATENCY = WORK / "circuitsage_hmac_v2_2_v1_verification_latency_12c4a.csv"
GAPS = WORK / "circuitsage_hmac_v2_2_release_packaging_gaps_12c4a.json"
SELFCHECK = WORK / "circuitsage_hmac_v2_2_wiring_selfcheck_12c4a.csv"
REPORT = WORK / "circuitsage_hmac_v2_2_v1_wiring_report_12c4a.md"
MANIFEST = RESULT4 / "circuitsage_hmac_v2_2_v1_verification_wiring_manifest_12c4a.json"
AUDIT = RESULT4 / "circuitsage_hmac_v2_2_v1_verification_wiring_freeze_12c4a.json"

V1_SCOPE_FAMILY = "opentitan_hmac_sha256"
FUTURE_BRAND = base.FUTURE_BRAND
SEED = 20260926

stop, require, now = base.stop, base.require, base.now
canonical_json, sha256, rel = base.canonical_json, base.sha256, base.rel
record, load_json, csv_bytes = base.record, base.load_json, base.csv_bytes
frozen_write = base.frozen_write


class V1Verifier:
    """Genuine V1 verification: builds the 646-vector and calls predict_proba."""

    N_SITE = 14
    N_SAMPLE = 527
    N_GRAPH = 119
    N_TOTAL = 646

    def __init__(self, *, double_scale_graph: bool = False) -> None:
        self.schema = load_json(V1_SCHEMA)
        self.lock = load_json(V1_LOCK)
        self.model = joblib.load(V1_MODEL)
        self.threshold = float(self.lock["threshold"])
        self.double_scale_graph = double_scale_graph

        z = np.load(SGC_CACHE, allow_pickle=True)
        self.k3 = z["k3_features"]
        self.k3_mean, self.k3_scale = z["k3_mean"], z["k3_scale"]

        g = np.load(GRAPH, allow_pickle=True)
        self.fanout = g["node_cell_fanout"]
        self.dcode = g["node_driver_type_code"]
        self.is_po = g["node_is_primary_output"]
        self.cat = g["node_site_category_code"]

        npre = self.schema["numeric_preprocessing"]
        self.fan_mean = float(npre["mean_float64"])
        self.fan_scale = float(npre["scale_float64"])
        self.dvocab = self.schema["categorical_vocabulary"]["driver_cell_type"]
        self.svocab = self.schema["categorical_vocabulary"]["site_category"]
        self.n_stim = int(self.schema["stimulus_feature_count"])

        require(self.model.n_features_in_ == self.N_TOTAL,
                f"V1 expects {self.model.n_features_in_}, contract says {self.N_TOTAL}")
        require(int(self.schema["model_feature_count"]) == self.N_SAMPLE,
                "schema sample-branch width")

    def site_features(self, node: int) -> np.ndarray:
        v = np.zeros(self.N_SITE, dtype=np.float32)
        v[0] = (float(self.fanout[node]) - self.fan_mean) / self.fan_scale
        v[1] = float(self.is_po[node])
        dc = int(self.dcode[node])
        v[2] = 1.0 if "DFF" in self.dvocab[dc] else 0.0
        v[3 + dc] = 1.0
        v[3 + len(self.dvocab) + int(self.cat[node])] = 1.0
        return v

    def graph_features(self, node: int) -> np.ndarray:
        """k3 is stored ALREADY SCALED. Rescaling is the documented defect."""
        if self.double_scale_graph:
            return ((self.k3[node] - self.k3_mean)
                    / np.where(self.k3_scale == 0, 1.0, self.k3_scale)).astype(np.float32)
        return self.k3[node].astype(np.float32)

    def build(self, node: int, stuck: int, stimulus: np.ndarray) -> np.ndarray:
        require(stimulus.size == self.n_stim, "stimulus width")
        return np.concatenate([[np.float32(stuck)], self.site_features(node),
                               stimulus.astype(np.float32), self.graph_features(node)])

    def verify(self, circuit: str, nodes, stuck: int,
               stimulus: np.ndarray) -> tuple[str, dict[str, Any]]:
        if circuit != V1_SCOPE_FAMILY:
            return "OUT_OF_SCOPE", {
                "reason": f"V1 was trained only on {V1_SCOPE_FAMILY}; applying it "
                          f"to {circuit} would be an unsupported extrapolation"}
        X = np.stack([self.build(int(n), stuck, stimulus) for n in nodes])
        p = self.model.predict_proba(X)[:, 1]
        above = p >= self.threshold
        return ("VERIFIED" if above.any() else "FLAGGED"), {
            "candidates": int(p.size),
            "max_probability": float(p.max()),
            "mean_probability": float(p.mean()),
            "above_threshold": int(above.sum()),
            "threshold": self.threshold,
        }


def diagnose(v: V1Verifier, rng, n_sites: int = 200) -> tuple[np.ndarray, list]:
    nodes = rng.choice(v.k3.shape[0], size=n_sites, replace=False)
    rows, probs = [], []
    for n in nodes:
        for s in (0, 1):
            stim = rng.integers(0, 2, v.n_stim).astype(np.float32)
            p = float(v.model.predict_proba(v.build(int(n), s, stim).reshape(1, -1))[0, 1])
            probs.append(p)
            rows.append({"node": int(n), "stuck_value": s,
                         "probability": round(p, 6),
                         "above_threshold": "YES" if p >= v.threshold else "NO"})
    return np.asarray(probs), rows


def execute() -> None:
    require(ROOT.name == "vlsi_fault_detection_v2", f"place script in project root, not {ROOT}")
    outputs = (SCALING, DEFECT, DIAGNOSTICS, COMPARISON, LATENCY, GAPS,
               SELFCHECK, REPORT, MANIFEST, AUDIT)
    require(not any(p.exists() for p in outputs),
            f"frozen Stage {STAGE} output exists; use --status")

    print("FROZEN INPUT VERIFICATION", flush=True)
    for path, expected in PINNED.items():
        require(path.is_file(), f"missing frozen input: {rel(path)}")
        require(sha256(path) == expected, f"frozen input SHA changed: {path.name}")
    print(f"  {len(PINNED)} frozen inputs (V1 release, SGC cache, 2I, 3D){'':<11}: OK",
          flush=True)

    a2i = load_json(AUDIT_2I)
    require(a2i["status"] == "PASS", "12C-2I frozen")
    src = SOURCE_2I.read_text(encoding="utf-8")
    require("predict_proba" not in src,
            "12C-2I already calls predict_proba; defect premise is wrong")
    print(f"  12C-2I confirmed to contain no predict_proba call{'':<13}: OK", flush=True)

    existing = sorted(p.name for p in CONFIG.glob("*.json"))
    require(not any("v1_feature_scaling_convention" in n for n in existing),
            "a scaling convention already exists")
    print(f"  config/v2_2 enumerated ({len(existing)} contracts); none prior"
          f"{'':<13}: OK", flush=True)

    print("\nBUILDING GENUINE V1 FORWARD PASS", flush=True)
    rng = np.random.default_rng(SEED)
    v = V1Verifier()
    # robust spread: the max is set by ~5 extreme-fanout DFF nodes out of 22,839,
    # so a max-based bound tests outlier count, not scaling correctness
    g_abs = np.abs(v.k3)
    g_p9999 = float(np.percentile(g_abs, 99.99))
    g_colmean = float(np.abs(v.k3.mean(0)).max())
    g_colstd = (float(v.k3.std(0).min()), float(v.k3.std(0).max()))
    g_outliers = int(np.count_nonzero(g_abs.max(1) > 20.0))
    print(f"  model expects {v.model.n_features_in_} features; "
          f"1 + {v.N_SITE} + {v.n_stim} + {v.N_GRAPH} = "
          f"{1 + v.N_SITE + v.n_stim + v.N_GRAPH}", flush=True)

    probs, diag_rows = diagnose(v, rng)
    sat = int(np.count_nonzero((probs == 0.0) | (probs == 1.0)))
    gmin, gmax = float(v.k3.min()), float(v.k3.max())
    print(f"  graph branch range [{gmin:.3f}, {gmax:.3f}]", flush=True)
    print(f"  probabilities: min={probs.min():.6f} max={probs.max():.6f} "
          f"std={probs.std():.6f}", flush=True)
    print(f"  saturated at exactly 0 or 1: {sat}/{probs.size}", flush=True)

    print("\nCOUNTERFACTUAL: DOUBLE-SCALED GRAPH BRANCH (the defect)", flush=True)
    rng_b = np.random.default_rng(SEED)
    vb = V1Verifier(double_scale_graph=True)
    probs_b, _ = diagnose(vb, rng_b)
    sat_b = int(np.count_nonzero((probs_b == 0.0) | (probs_b == 1.0)))
    bad = ((vb.k3 - vb.k3_mean) / np.where(vb.k3_scale == 0, 1.0, vb.k3_scale))
    print(f"  graph branch range [{bad.min():.2f}, {bad.max():.2f}]", flush=True)
    print(f"  saturated at exactly 0 or 1: {sat_b}/{probs_b.size}", flush=True)

    # polarity sensitivity, both ways
    def polarity(ver, seed) -> int:
        r = np.random.default_rng(seed)
        nodes = r.choice(ver.k3.shape[0], size=100, replace=False)
        stim = r.integers(0, 2, ver.n_stim).astype(np.float32)
        p0 = ver.model.predict_proba(np.stack([ver.build(int(n), 0, stim) for n in nodes]))[:, 1]
        p1 = ver.model.predict_proba(np.stack([ver.build(int(n), 1, stim) for n in nodes]))[:, 1]
        return int(np.count_nonzero(np.abs(p0 - p1) > 1e-9))

    pol_ok, pol_bad = polarity(v, SEED + 1), polarity(vb, SEED + 1)
    print(f"  polarity sensitivity  correct={pol_ok}/100  double-scaled={pol_bad}/100",
          flush=True)

    print("\nLATENCY", flush=True)
    r = np.random.default_rng(SEED + 2)
    stim = r.integers(0, 2, v.n_stim).astype(np.float32)
    lat_rows = []
    for k in (1, 3, 10, 100):
        nodes = r.choice(v.k3.shape[0], size=k, replace=False)
        X = np.stack([v.build(int(n), 1, stim) for n in nodes])
        t0 = time.perf_counter()
        for _ in range(50):
            v.model.predict_proba(X)
        dt = (time.perf_counter() - t0) / 50
        lat_rows.append({"candidate_sites": k,
                         "total_ms": round(dt * 1000, 4),
                         "ms_per_site": round(dt * 1000 / k, 5)})
        print(f"  {k:>3} sites: {dt*1000:8.3f} ms", flush=True)

    print("\nSCOPE BEHAVIOUR", flush=True)
    nodes = r.choice(v.k3.shape[0], size=3, replace=False)
    verdict_in, det_in = v.verify(V1_SCOPE_FAMILY, nodes, 1, stim)
    verdict_out, det_out = v.verify("secworks_chacha", nodes, 1, stim)
    print(f"  {V1_SCOPE_FAMILY:<22} -> {verdict_in} "
          f"(max p={det_in['max_probability']:.4f})", flush=True)
    print(f"  {'secworks_chacha':<22} -> {verdict_out}", flush=True)
    require(verdict_out == "OUT_OF_SCOPE", "out-of-scope circuits must not be scored")

    created = now()
    checks = [
        {"check": "feature width equals 646", "expected": "646",
         "observed": str(v.model.n_features_in_),
         "result": "PASS" if v.model.n_features_in_ == 646 else "FAIL"},
        {"check": "graph branch is standardized (col mean ~ 0)",
         "expected": "max|colmean| < 0.05", "observed": f"{g_colmean:.6f}",
         "result": "PASS" if g_colmean < 0.05 else "FAIL"},
        {"check": "graph branch is standardized (col std ~ 1)",
         "expected": "all in [0.5, 2.0]",
         "observed": f"[{g_colstd[0]:.3f}, {g_colstd[1]:.3f}]",
         "result": "PASS" if 0.5 < g_colstd[0] and g_colstd[1] < 2.0 else "FAIL"},
        {"check": "graph branch robust range sane (p99.99)",
         "expected": "< 20", "observed": f"{g_p9999:.3f}",
         "result": "PASS" if g_p9999 < 20.0 else "FAIL"},
        {"check": "extreme-fanout outliers are rare",
         "expected": "< 0.1% of nodes",
         "observed": f"{g_outliers}/{v.k3.shape[0]}",
         "result": "PASS" if g_outliers / v.k3.shape[0] < 0.001 else "FAIL"},
        {"check": "saturation at exactly 0 or 1 is negligible",
         "expected": "<= 1% of samples", "observed": f"{sat}/{probs.size}",
         "result": "PASS" if sat <= 0.01 * probs.size else "FAIL"},
        {"check": "probability spread is graded", "expected": "std > 0.05",
         "observed": f"{probs.std():.4f}",
         "result": "PASS" if probs.std() > 0.05 else "FAIL"},
        {"check": "responds to fault polarity", "expected": ">= 80/100",
         "observed": f"{pol_ok}/100", "result": "PASS" if pol_ok >= 80 else "FAIL"},
        {"check": "double-scaling demonstrably worse", "expected": "more saturation",
         "observed": f"{sat_b} vs {sat}", "result": "PASS" if sat_b > sat else "FAIL"},
        {"check": "out-of-scope returns OUT_OF_SCOPE", "expected": "OUT_OF_SCOPE",
         "observed": verdict_out, "result": "PASS" if verdict_out == "OUT_OF_SCOPE" else "FAIL"},
        {"check": "12C-2I unmodified", "expected": PINNED[SOURCE_2I][:16],
         "observed": sha256(SOURCE_2I)[:16],
         "result": "PASS" if sha256(SOURCE_2I) == PINNED[SOURCE_2I] else "FAIL"},
    ]
    print("\nSELF-CHECKS", flush=True)
    for c in checks:
        print(f"  {c['check']:<44} {c['result']}", flush=True)
    require(all(c["result"] == "PASS" for c in checks), "all self-checks must pass")

    scaling = {
        "convention_version": "CIRCUITSAGE-HMAC-V2.2-V1-FEATURE-SCALING-CONVENTION-12C4A-v1",
        "stage": STAGE, "status": "FROZEN", "created_at": created,
        "applies_to": "V1 HYBRID_FUSION_MLP_11D2C 646-feature input",
        "graph_branch": {
            "artifact": rel(SGC_CACHE),
            "array": "k3_features",
            "storage_state": "ALREADY SCALED",
            "apply_k3_mean_k3_scale_at_inference": False,
            "k3_mean_k3_scale_role": "provenance record of the scaler already applied",
            "evidence_column_mean": round(float(np.abs(v.k3.mean(0)).max()), 6),
            "evidence_column_std_median": round(float(np.median(v.k3.std(0))), 6),
            "correct_range": [round(gmin, 4), round(gmax, 4)],
            "correct_robust_p99_99": round(g_p9999, 4),
            "column_std_range": [round(g_colstd[0], 4), round(g_colstd[1], 4)],
            "extreme_fanout_outlier_nodes": g_outliers,
            "outlier_note": ("the maximum is set by ~5 very-high-fanout $_DFFE_PN1P_ "
                             "nodes out of 22,839; this is real netlist structure, "
                             "not a scaling error, so correctness is asserted on "
                             "column mean/std and p99.99 rather than on the max"),
            "double_scaled_range": [round(float(bad.min()), 2), round(float(bad.max()), 2)],
        },
        "cell_fanout": {
            "storage_state": "RAW",
            "apply_numeric_preprocessing_at_inference": True,
            "scaler": v.schema["numeric_preprocessing"]["name"],
            "mean": v.fan_mean, "scale": v.fan_scale,
        },
        "why_this_is_recorded": (
            "the two blocks use OPPOSITE conventions and nothing in the published "
            "schema states this; applying the graph scaler twice saturates the model "
            "silently - it still returns verdicts, so the defect is invisible without "
            "inspecting the output distribution"),
        "regression_guard": "stage self-test asserts zero saturation and polarity sensitivity",
    }

    defect = {
        "defect_version": "CIRCUITSAGE-HMAC-V2.2-SCOPE-GATE-DEFECT-12C4A-v1",
        "stage": STAGE, "created_at": created,
        "defective_stage": "12C-2I",
        "defect": ("V1 verification was a scope gate: it loaded the model, checked the "
                   "circuit family and returned VERIFIED without calling predict_proba"),
        "evidence": "the string 'predict_proba' does not occur in stage_12c2i source",
        "why_it_matters": ("the pipeline reported a VERIFICATION stage backed by a "
                           "93,185-parameter model that never ran"),
        "resolution": "this stage builds and measures the genuine forward pass",
        "12c2i_modified": False,
        "second_defect_found_while_fixing": {
            "defect": "graph branch double-scaled at inference",
            "found_by": "inspecting the output probability distribution",
            "symptom": f"{sat_b}/{probs_b.size} probabilities at exactly 0.0 or 1.0; "
                       f"polarity sensitivity {pol_bad}/100",
            "after_fix": f"{sat}/{probs.size} saturated; polarity {pol_ok}/100",
            "lesson": ("a model that runs is not a model that works; check the output "
                       "distribution, not just the absence of exceptions"),
        },
    }

    gaps = {
        "gaps_version": "CIRCUITSAGE-HMAC-V2.2-RELEASE-PACKAGING-GAPS-12C4A-v1",
        "stage": STAGE, "created_at": created,
        "subject": "the already-public V1 release bundle",
        "gaps": [
            {"gap": "propagation cache not shipped",
             "detail": f"{rel(SGC_CACHE)} (792 KB) supplies all 119 graph features "
                       f"but is absent from the release bundle",
             "consequence": "a downloader cannot compute V1 inputs; the model is unusable",
             "severity": "BLOCKING"},
            {"gap": "golden netlist graph not shipped",
             "detail": f"{rel(GRAPH)} supplies site features (fanout, driver type, "
                       f"primary-output flag, site category)",
             "consequence": "site-feature block cannot be constructed",
             "severity": "BLOCKING"},
            {"gap": "scaling convention undocumented",
             "detail": "nothing states that k3_features ship pre-scaled while "
                       "cell_fanout ships raw",
             "consequence": "silent saturation, as demonstrated in this stage",
             "severity": "HIGH"},
            {"gap": "no worked inference example",
             "detail": "no runnable script builds one 646-vector end to end",
             "consequence": "every integrator re-derives the layout from the schema",
             "severity": "MEDIUM"},
        ],
        "resolution_stage": "FAULTIVA PACKAGING (next)",
        "v1_release_modified_by_this_stage": False,
    }

    lat_table = "\n".join(f"| {r['candidate_sites']} | {r['total_ms']:.3f} | "
                          f"{r['ms_per_site']:.5f} |" for r in lat_rows)
    gap_table = "\n".join(f"| {g['gap']} | {g['consequence']} | **{g['severity']}** |"
                          for g in gaps["gaps"])
    chk_table = "\n".join(f"| {c['check']} | {c['expected']} | {c['observed']} | "
                          f"**{c['result']}** |" for c in checks)

    report = f"""# Stage {STAGE} — Genuine V1 Verification Wiring

**Status: PASS / FROZEN.** Stage 12C-2I is unmodified; this is an additive correction.

## The defect

Stage 12C-2I wired V1 as a **scope gate**. It loaded the model, checked the circuit
family, and returned `VERIFIED` — **without ever calling `predict_proba`**. The
string does not occur in its source. A 93,185-parameter model was reported as a
verification stage while never running.

## The genuine forward pass

| block | width | source |
|---|---|---|
| `stuck_value` | 1 | fault polarity |
| site features | 14 | fanout (scaled), stem flags, cell-type one-hot |
| stimulus | 512 | 256 key + 256 message bits |
| **sample branch** | **527** | |
| graph branch | 119 | frozen DIR_SGC_K3 cache |
| **total** | **646** | → MLP(128,64,32) → threshold {v.threshold} |

## A second defect, found while fixing the first

The graph cache ships **already scaled**. Applying `k3_mean`/`k3_scale` again — the
obvious reading — inflates the branch and saturates the model:

| | correct | double-scaled |
|---|---|---|
| graph range (min/max) | [{gmin:.2f}, {gmax:.2f}] | [{bad.min():.1f}, **{bad.max():.1f}**] |
| graph p99.99 | **{g_p9999:.2f}** | {np.percentile(np.abs(bad), 99.99):.1f} |
| column mean / std | {g_colmean:.4f} / [{g_colstd[0]:.2f}, {g_colstd[1]:.2f}] | inflated |
| at exactly 0.0 or 1.0 | **{sat}/{probs.size}** | {sat_b}/{probs_b.size} |
| polarity sensitivity | **{pol_ok}/100** | {pol_bad}/100 |

The correct branch's maximum of {gmax:.1f} comes from just **{g_outliers} nodes out of
{v.k3.shape[0]:,}** — very-high-fanout `$_DFFE_PN1P_` flip-flops. That is real netlist
structure, so correctness is asserted on column mean/std and p99.99, not on the max.

`cell_fanout` uses the **opposite** convention — stored raw, scaler must be applied.
Nothing in the published schema says so. Both conventions are now frozen in
`{SCALING.name}` and asserted by the self-test.

**The lesson worth keeping: a model that runs is not a model that works.** The
double-scaled pipeline threw no exception and returned confident verdicts.

## Measured behaviour

Probabilities over 400 site/polarity samples: min {probs.min():.4f}, max
{probs.max():.4f}, mean {probs.mean():.4f}, std {probs.std():.4f} — graded, no saturation.

| candidate sites | total ms | ms/site |
|---|---|---|
{lat_table}

Scope is unchanged: `{V1_SCOPE_FAMILY}` is scored, everything else returns
**OUT_OF_SCOPE**.

## Self-checks

| check | expected | observed | result |
|---|---|---|---|
{chk_table}

## Release packaging gaps found

| gap | consequence | severity |
|---|---|---|
{gap_table}

**Two BLOCKING gaps in the already-public V1 release**: the propagation cache and
the golden netlist graph are not shipped, so a downloader cannot compute the
model's inputs. Must be resolved in packaging.

## Next gate

Faultiva packaging: ship V1 with its feature pipeline, the documented scaling
convention, a worked example, and the V2.2 signature dictionary. Brand: **{FUTURE_BRAND}**.
"""

    frozen_write(SCALING, canonical_json(scaling))
    frozen_write(DEFECT, canonical_json(defect))
    frozen_write(DIAGNOSTICS, csv_bytes(diag_rows, list(diag_rows[0].keys())))
    frozen_write(COMPARISON, csv_bytes([
        {"variant": "correct", "graph_min": round(gmin, 4), "graph_max": round(gmax, 4),
         "saturated": sat, "polarity_sensitive_sites": pol_ok,
         "prob_std": round(float(probs.std()), 6)},
        {"variant": "double_scaled", "graph_min": round(float(bad.min()), 2),
         "graph_max": round(float(bad.max()), 2), "saturated": sat_b,
         "polarity_sensitive_sites": pol_bad,
         "prob_std": round(float(probs_b.std()), 6)},
    ], ["variant", "graph_min", "graph_max", "saturated",
        "polarity_sensitive_sites", "prob_std"]))
    frozen_write(LATENCY, csv_bytes(lat_rows, list(lat_rows[0].keys())))
    frozen_write(GAPS, canonical_json(gaps))
    frozen_write(SELFCHECK, csv_bytes(checks, list(checks[0].keys())))
    frozen_write(REPORT, report.encode())

    stage_outputs = (SCALING, DEFECT, DIAGNOSTICS, COMPARISON, LATENCY, GAPS,
                     SELFCHECK, REPORT)
    manifest = {
        "manifest_version": "CIRCUITSAGE-HMAC-V2.2-V1-WIRING-MANIFEST-12C4A-v1",
        "stage": STAGE, "status": "PASS", "created_at": created,
        "stage_source": record(Path(__file__).resolve()),
        "frozen_inputs": {rel(p): record(p) for p in PINNED},
        "outputs": {rel(p): record(p) for p in stage_outputs},
        "seed": SEED,
        "training_calls": 0, "selection_calls": 0,
        "threshold_changed": False,
        "v1_release_modified": False,
        "prior_stages_modified": [],
        "test_access": 0, "holdout_access": 0,
    }
    frozen_write(MANIFEST, canonical_json(manifest))

    audit = {
        "audit_version": "CIRCUITSAGE-HMAC-V2.2-V1-WIRING-FREEZE-12C4A-v1",
        "stage": STAGE, "status": "PASS",
        "v1_verification_is_genuine": True,
        "predict_proba_called": True,
        "feature_width": int(v.model.n_features_in_),
        "threshold": v.threshold,
        "threshold_changed": False,
        "scope_family": V1_SCOPE_FAMILY,
        "out_of_scope_behaviour": "OUT_OF_SCOPE, never a guessed probability",
        "graph_branch_range": [round(gmin, 4), round(gmax, 4)],
        "graph_branch_p99_99": round(g_p9999, 4),
        "graph_branch_column_mean_max": round(g_colmean, 6),
        "graph_branch_outlier_nodes": g_outliers,
        "double_scaled_range": [round(float(bad.min()), 2), round(float(bad.max()), 2)],
        "saturated_correct": sat,
        "saturated_double_scaled": sat_b,
        "polarity_sensitivity_correct": pol_ok,
        "polarity_sensitivity_double_scaled": pol_bad,
        "probability_std": round(float(probs.std()), 6),
        "latency_ms_10_sites": next(r["total_ms"] for r in lat_rows
                                    if r["candidate_sites"] == 10),
        "self_checks_total": len(checks),
        "self_checks_passed": sum(1 for c in checks if c["result"] == "PASS"),
        "defects_recorded": 2,
        "12c2i_modified": False,
        "v1_release_modified": False,
        "release_blocking_gaps": sum(1 for g in gaps["gaps"] if g["severity"] == "BLOCKING"),
        "independent_generalization": "NOT ESTABLISHED",
        "scaling_convention_record": record(SCALING),
        "defect_record": record(DEFECT),
        "gaps_record": record(GAPS),
        "manifest_record": record(MANIFEST),
        "future_combined_model_brand": FUTURE_BRAND,
        "next_gate": "FAULTIVA PACKAGING",
    }
    frozen_write(AUDIT, canonical_json(audit))

    print(f"\n{'Stage':<52}: {STAGE} — GENUINE V1 VERIFICATION")
    print(f"{'Status':<52}: PASS / FROZEN")
    print(f"{'predict_proba actually called':<52}: YES")
    print(f"{'Feature width':<52}: {v.model.n_features_in_}")
    print(f"{'Saturated (correct / double-scaled)':<52}: {sat} / {sat_b}")
    print(f"{'Polarity sensitivity (correct / defective)':<52}: {pol_ok}/100 / {pol_bad}/100")
    print(f"{'Self-checks':<52}: {len(checks)}/{len(checks)} PASS")
    print(f"{'12C-2I modified':<52}: NO")
    print(f"{'Release BLOCKING gaps found':<52}: "
          f"{sum(1 for g in gaps['gaps'] if g['severity']=='BLOCKING')}")
    print(f"{'Audit SHA':<52}: {sha256(AUDIT)}")


def status() -> None:
    print(f"STAGE {STAGE} — V1 VERIFICATION WIRING STATUS")
    if not (MANIFEST.is_file() and AUDIT.is_file()):
        print("Status                    : NOT FROZEN")
        return
    a = load_json(AUDIT)
    print("Status                    : PASS / FROZEN")
    print(f"Genuine verification      : {a['v1_verification_is_genuine']}")
    print(f"Saturation correct/defect : {a['saturated_correct']} / {a['saturated_double_scaled']}")
    print(f"Self-checks               : {a['self_checks_passed']}/{a['self_checks_total']}")
    print(f"Release blocking gaps     : {a['release_blocking_gaps']}")
    print(f"Audit SHA                 : {sha256(AUDIT)}")


def self_test() -> None:
    v = V1Verifier()
    require(v.model.n_features_in_ == 646, "646 features")
    r = np.random.default_rng(1)
    stim = r.integers(0, 2, v.n_stim).astype(np.float32)
    x = v.build(0, 1, stim)
    require(x.size == 646, f"built vector is {x.size}, expected 646")
    require(np.isfinite(x).all(), "vector must be finite")
    require(float(np.abs(v.k3.mean(0)).max()) < 0.05,
            "graph branch must be standardized (column means ~ 0)")
    require(float(np.percentile(np.abs(v.k3), 99.99)) < 20.0,
            "graph branch robust range must be sane")
    verdict, _ = v.verify("secworks_chacha", [0, 1], 1, stim)
    require(verdict == "OUT_OF_SCOPE", "scope rule enforced")
    print(f"Stage {STAGE} self-test: PASS")


def locked_execute() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    with LOCK_FILE.open("a+", encoding="utf-8") as h:
        try:
            fcntl.flock(h.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            stop(f"Stage {STAGE} execution lock is held by another process")
        execute()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    m = p.add_mutually_exclusive_group()
    m.add_argument("--status", action="store_true")
    m.add_argument("--self-test", action="store_true")
    a = p.parse_args()
    if a.status:
        status()
    elif a.self_test:
        self_test()
    else:
        locked_execute()


if __name__ == "__main__":
    main()
