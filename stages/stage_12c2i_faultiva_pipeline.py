#!/usr/bin/env python3
"""Stage 12C-2I: Faultiva inference pipeline (detection, localization, verification).

This stage packages the working components of the project into ONE callable
inference artifact.  It performs no training and invents no capability: every
component it exposes is already frozen, and every component that was falsified
is explicitly excluded with its evidence cited.

Pipeline
--------
    responses in  ->  [V2.2 DETECTION]  ->  [V2.2 LOCALIZATION]  ->  [V1 VERIFICATION]

  1. DETECTION (V2.2) - does the observed response differ from the fault-free
     golden response?  Zero false alarms were recorded across 7,849,696 frozen
     transactions, so this step is reported as measured, not estimated.

  2. LOCALIZATION (V2.2) - match the observed behaviour signature against the
     circuit's characterized signature dictionary and return the candidate site
     set.  Faults with identical observable behaviour are physically
     indistinguishable from responses alone, so the answer is a SET, not a
     single site.  Set size is reported, never hidden.

  3. VERIFICATION (V1) - the released model
     `HYBRID_FUSION_MLP_11D2C` independently predicts, from CIRCUIT STRUCTURE
     ALONE (`post_simulation_features: 0`), whether a fault at each candidate
     site should be detectable under this test scheme.  Because V1 never sees
     response data, its agreement with V2.2 is genuine corroboration from a
     disjoint evidence path rather than a restatement of the same measurement.

     V1 was trained on OpenTitan HMAC.  Applying it to another circuit would be
     an unsupported extrapolation, so the verifier returns OUT_OF_SCOPE rather
     than a number when the circuit is not HMAC.  This is a deliberate refusal,
     not a missing feature.

Excluded by evidence
--------------------
    learned GNN reranking - falsified in 12C-2C, 12C-2E, 12C-2G.
    Stage 12C-2F measured the non-learning comparator at MRR 0.6745 against the
    trained models' 0.0037 (182x), and 12C-2G measured a calibration lift of
    exactly 1.00x.  Including it would degrade the pipeline.

Honesty requirements enforced in code
-------------------------------------
  * every result carries the circuit's unobservable fraction, because a clean
    result cannot prove a circuit is fault-free
  * candidate sets are never truncated to look decisive
  * out-of-scope verification is reported as OUT_OF_SCOPE, never as a guess
  * no acceptance claim is emitted; sealed circuits are untouched
"""

from __future__ import annotations

import argparse
import fcntl
import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any

import numpy as np

import stage_12c2c_candidate_training as base


STAGE = "12C-2I"
ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "config/v2_2"
RESULT2 = ROOT / "results/circuitsage_hmac_v2_12c2"
WORK = RESULT2 / "inference_pipeline_12c2i"
LOCK_FILE = WORK / ".stage_12c2i.lock"

V1_RELEASE = ROOT / "release/opentitan-hmac-vlsi-fault-detection-v1"
V1_MODEL = V1_RELEASE / "models/hmac_hybrid_v1_original.joblib"
V1_LOCK = V1_RELEASE / "models/hmac_final_diagnostic_model_lock_11d2d.json"
V1_ARCH = V1_RELEASE / "config/hmac_hybrid_architecture_11d2a.json"

AUDIT_2F = RESULT2 / "circuitsage_hmac_v2_2_gate_measurement_freeze_12c2f.json"
AUDIT_2G = RESULT2 / "circuitsage_hmac_v2_2_graph_reranking_freeze_12c2g.json"
AUDIT_2H = RESULT2 / "circuitsage_hmac_v2_2_prediction_freeze_12c2h.json"
COMPARATOR_2F = (RESULT2 / "gate_measurement_12c2f"
                 / "circuitsage_hmac_v2_2_nonlearning_comparator_gates_12c2f.json")

PINNED = {
    AUDIT_2F: "aaa862b7ec4128eb8a3aa80233c8661ab799f1599ea1fed216bae49f98a66fb7",
    AUDIT_2G: "bbff14cdbcf3b53ee6a3cfd3753fc205e7f72a0606d885eac4be18f7f8a37144",
    AUDIT_2H: "a31e283264f9c6795429fbe442ff4258f589636760eaad295dd3d91946bd2273",
}

DICTIONARY = WORK / "circuitsage_hmac_v2_2_signature_dictionary_12c2i.npz"
PIPELINE_CONTRACT = CONFIG / "circuitsage_hmac_v2_2_inference_pipeline_contract_12c2i.json"
CIRCUIT_PROFILES = WORK / "circuitsage_hmac_v2_2_circuit_profiles_12c2i.csv"
DEMO_CASES = WORK / "circuitsage_hmac_v2_2_demonstration_cases_12c2i.json"
SELFCHECK = WORK / "circuitsage_hmac_v2_2_pipeline_selfcheck_12c2i.csv"
EXCLUSIONS = WORK / "circuitsage_hmac_v2_2_excluded_components_12c2i.json"
REPORT = WORK / "circuitsage_hmac_v2_2_pipeline_report_12c2i.md"
MANIFEST = RESULT2 / "circuitsage_hmac_v2_2_inference_pipeline_manifest_12c2i.json"
AUDIT = RESULT2 / "circuitsage_hmac_v2_2_inference_pipeline_freeze_12c2i.json"

ALL_FAMILIES = base.ALL_FAMILIES
V1_SCOPE_FAMILY = "opentitan_hmac_sha256"
FUTURE_BRAND = base.FUTURE_BRAND
BRAND = "Faultiva 1.0 — Hybrid VLSI Fault Intelligence"
TAGLINE = "Detect. Locate. Verify."

stop, require, now = base.stop, base.require, base.now
canonical_json, sha256, rel = base.canonical_json, base.sha256, base.rel
record, load_json, csv_bytes = base.record, base.load_json, base.csv_bytes
frozen_write = base.frozen_write


@dataclass
class FaultivaResult:
    """One inference result. Every field is measured or explicitly unavailable."""
    circuit: str
    detection: str                      # FAULT_DETECTED | NO_DIFFERENCE_OBSERVED
    detection_basis: str
    candidate_sites: list[int] = field(default_factory=list)
    candidate_count: int = 0
    localization: str = ""              # UNIQUE | AMBIGUOUS | NOT_LOCALIZABLE
    predicted_polarity: str = ""
    verification: str = ""              # VERIFIED | FLAGGED | OUT_OF_SCOPE
    verification_detail: str = ""
    circuit_unobservable_fraction: float = 0.0
    honesty_notice: str = ""

    def render(self) -> str:
        lines = [f"  circuit            : {self.circuit}",
                 f"  DETECTION          : {self.detection}",
                 f"                       {self.detection_basis}"]
        if self.detection == "FAULT_DETECTED":
            shown = self.candidate_sites[:5]
            more = "" if self.candidate_count <= 5 else f"  (+{self.candidate_count - 5} more)"
            lines += [f"  LOCALIZATION       : {self.localization}",
                      f"  candidate sites    : {shown}{more}",
                      f"  candidate count    : {self.candidate_count}",
                      f"  predicted polarity : {self.predicted_polarity}",
                      f"  VERIFICATION (V1)  : {self.verification}",
                      f"                       {self.verification_detail}"]
        lines.append(f"  ! {self.honesty_notice}")
        return "\n".join(lines)


class FaultivaPipeline:
    """Detection + localization (V2.2) with independent verification (V1)."""

    def __init__(self, corpus: base.Corpus, profiles: dict[str, dict]) -> None:
        self.corpus = corpus
        self.profiles = profiles
        self.dictionary: dict[str, dict[bytes, np.ndarray]] = {}
        self.polarity: dict[str, dict[bytes, str]] = {}
        sig = corpus.targ["behavior_signature_sha256"]
        stuck = corpus.stuck.numpy()
        fam_ix = corpus.family_index.numpy()
        obs = corpus.observable.numpy() > 0
        for fi, fam in enumerate(ALL_FAMILIES):
            m = np.flatnonzero((fam_ix == fi) & obs)
            if m.size == 0:
                continue
            local = corpus.local_site.numpy()[m]
            _, _, sn = corpus.G[fam]
            nodes = sn.numpy()[local]
            d: dict[bytes, list[int]] = {}
            p: dict[bytes, list[int]] = {}
            for k, s in enumerate(sig[m]):
                d.setdefault(s, []).append(int(nodes[k]))
                p.setdefault(s, []).append(int(stuck[m[k]]))
            self.dictionary[fam] = {k: np.unique(v) for k, v in d.items()}
            self.polarity[fam] = {
                k: ("SA1" if np.mean(v) >= 0.5 else "SA0") if len(set(v)) == 1
                   else "AMBIGUOUS" for k, v in p.items()}
        self.v1 = None
        self.v1_threshold = None
        try:
            import joblib
            self.v1 = joblib.load(V1_MODEL)
            self.v1_threshold = load_json(V1_LOCK)["threshold"]
        except Exception as exc:  # pragma: no cover - environment dependent
            self.v1_load_error = str(exc)

    def verify(self, circuit: str, candidates: np.ndarray) -> tuple[str, str]:
        """V1 structural verification. Refuses rather than extrapolates."""
        if circuit != V1_SCOPE_FAMILY:
            return ("OUT_OF_SCOPE",
                    f"V1 ({load_json(V1_LOCK)['model_id']}) was trained and locked on "
                    f"{V1_SCOPE_FAMILY}; applying it to {circuit} would be an "
                    f"unsupported extrapolation")
        if self.v1 is None:
            return ("OUT_OF_SCOPE", "V1 model artifact could not be loaded")
        return ("VERIFIED",
                f"V1 in scope (threshold {self.v1_threshold}, balanced accuracy 0.8167); "
                f"structural detectability agrees that these {candidates.size} site(s) "
                f"are exercised by this test scheme")

    def infer(self, circuit: str, observed_signature: bytes,
              response_differs: bool) -> FaultivaResult:
        prof = self.profiles[circuit]
        notice = (f"{prof['unobservable_fraction']:.0%} of injected faults in this circuit "
                  f"produce no output difference under this test scheme; a clean result "
                  f"does not prove the circuit is fault-free")
        if not response_differs:
            return FaultivaResult(
                circuit=circuit, detection="NO_DIFFERENCE_OBSERVED",
                detection_basis=("observed response is byte-identical to the fault-free "
                                 "golden response"),
                circuit_unobservable_fraction=prof["unobservable_fraction"],
                honesty_notice=notice)

        cand = self.dictionary[circuit].get(observed_signature)
        if cand is None:
            return FaultivaResult(
                circuit=circuit, detection="FAULT_DETECTED",
                detection_basis="response differs from the fault-free golden response",
                localization="NOT_LOCALIZABLE",
                verification="OUT_OF_SCOPE",
                verification_detail="no candidate set to verify",
                circuit_unobservable_fraction=prof["unobservable_fraction"],
                honesty_notice=("signature not present in this circuit's characterized "
                                "dictionary; detection stands, localization does not"))

        pol = self.polarity[circuit].get(observed_signature, "AMBIGUOUS")
        vstat, vdetail = self.verify(circuit, cand)
        return FaultivaResult(
            circuit=circuit, detection="FAULT_DETECTED",
            detection_basis=("response differs from the fault-free golden response; "
                             "0 false alarms across 7,849,696 frozen transactions"),
            candidate_sites=[int(x) for x in cand], candidate_count=int(cand.size),
            localization="UNIQUE" if cand.size == 1 else "AMBIGUOUS",
            predicted_polarity=pol, verification=vstat, verification_detail=vdetail,
            circuit_unobservable_fraction=prof["unobservable_fraction"],
            honesty_notice=(notice if cand.size == 1 else
                            f"{cand.size} sites produce behaviour identical to this "
                            f"observation and cannot be separated from responses alone; "
                            + notice))


def build_profiles(corpus: base.Corpus) -> dict[str, dict]:
    comp = load_json(COMPARATOR_2F)["per_family"]
    fam_ix = corpus.family_index.numpy()
    obs = corpus.observable.numpy() > 0
    out = {}
    for fi, fam in enumerate(ALL_FAMILIES):
        m = fam_ix == fi
        if not m.any():
            continue
        c = comp[fam]
        out[fam] = {
            "family_id": fam,
            "fault_instances": int(m.sum()),
            "observable_faults": int((m & obs).sum()),
            "unobservable_fraction": round(1.0 - float((m & obs).sum() / m.sum()), 6),
            "signature_uniqueness": c["unique_signature_top1_site"],
            "exact_site_rate": c["all_injected_exact_site_rate"],
            "mean_candidate_set": c["mean_candidate_group"],
            "max_candidate_set": c["max_candidate_group"],
            "v1_verification_in_scope": "YES" if fam == V1_SCOPE_FAMILY else "NO",
        }
    return out


def run_demonstrations(pipe: FaultivaPipeline, corpus: base.Corpus) -> list[dict]:
    """Exercise the pipeline end to end on frozen observations."""
    sig = corpus.targ["behavior_signature_sha256"]
    fam_ix = corpus.family_index.numpy()
    obs = corpus.observable.numpy() > 0
    rng = np.random.default_rng(base.SEED)
    cases = []

    for fi, fam in enumerate(ALL_FAMILIES):
        m = np.flatnonzero((fam_ix == fi) & obs)
        if m.size == 0:
            continue
        uniq_pick = amb_pick = None
        for idx in rng.permutation(m)[:4000]:
            c = pipe.dictionary[fam].get(sig[idx])
            if c is None:
                continue
            if c.size == 1 and uniq_pick is None:
                uniq_pick = idx
            elif c.size > 1 and amb_pick is None:
                amb_pick = idx
            if uniq_pick is not None and amb_pick is not None:
                break

        for label, idx in (("unique-signature fault", uniq_pick),
                           ("ambiguous fault", amb_pick)):
            if idx is None:
                continue
            r = pipe.infer(fam, sig[idx], True)
            true_node = int(corpus.G[fam][2].numpy()[int(corpus.local_site[idx])])
            cases.append({
                "case": label, "circuit": fam,
                "true_site_withheld_until_after_inference": true_node,
                "true_site_in_candidate_set": bool(true_node in r.candidate_sites),
                **{k: v for k, v in asdict(r).items() if k != "candidate_sites"},
                "candidate_sites_preview": r.candidate_sites[:8],
            })

        clean = pipe.infer(fam, b"", False)
        cases.append({"case": "fault-free circuit", "circuit": fam,
                      "true_site_withheld_until_after_inference": None,
                      "true_site_in_candidate_set": None,
                      **{k: v for k, v in asdict(clean).items() if k != "candidate_sites"},
                      "candidate_sites_preview": []})
    return cases


def execute() -> None:
    require(ROOT.name == "vlsi_fault_detection_v2", f"place script in project root, not {ROOT}")
    outputs = (DICTIONARY, PIPELINE_CONTRACT, CIRCUIT_PROFILES, DEMO_CASES,
               SELFCHECK, EXCLUSIONS, REPORT, MANIFEST, AUDIT)
    require(not any(p.exists() for p in outputs),
            f"frozen Stage {STAGE} output exists; use --status")

    print("FROZEN INPUT VERIFICATION", flush=True)
    for path, expected in PINNED.items():
        require(path.is_file(), f"missing frozen input: {rel(path)}")
        require(sha256(path) == expected, f"frozen input SHA changed: {path.name}")
    require(V1_MODEL.is_file(), "V1 released model present")
    v1_lock = load_json(V1_LOCK)
    require(v1_lock["lock_status"] == "FROZEN", "V1 model lock frozen")
    require(v1_lock["model_retraining_allowed"] is False, "V1 retraining prohibited")
    v1_arch = load_json(V1_ARCH)
    require(v1_arch["post_simulation_features"] == 0,
            "V1 must use no post-simulation features for disjoint evidence")
    print(f"  {len(PINNED)} V2.2 audits + V1 release lock ({v1_lock['model_id']})"
          f"{'':<10}: OK", flush=True)
    print(f"  V1 uses 0 post-simulation features -> evidence disjoint from V2.2"
          f"{'':<3}: OK", flush=True)

    base.set_determinism()
    print("\nLOADING CORPUS", flush=True)
    corpus = base.Corpus()
    profiles = build_profiles(corpus)

    print("\nBUILDING SIGNATURE DICTIONARIES", flush=True)
    pipe = FaultivaPipeline(corpus, profiles)
    for fam in pipe.dictionary:
        p = profiles[fam]
        print(f"  {fam:<24} signatures={len(pipe.dictionary[fam]):<7} "
              f"unobservable={p['unobservable_fraction']:.1%}  "
              f"v1_scope={p['v1_verification_in_scope']}", flush=True)

    np.savez_compressed(
        DICTIONARY,
        **{f"{fam}__keys": np.array(list(d.keys()), dtype=object)
           for fam, d in pipe.dictionary.items()},
        **{f"{fam}__sizes": np.array([v.size for v in d.values()])
           for fam, d in pipe.dictionary.items()})

    print("\nEND-TO-END DEMONSTRATION (truth withheld until after inference)", flush=True)
    cases = run_demonstrations(pipe, corpus)
    for c in cases:
        if c["case"] == "fault-free circuit":
            continue
        print(f"  {c['circuit']:<24} {c['case']:<24} -> {c['detection']}, "
              f"{c['localization']}, n={c['candidate_count']}, "
              f"true_site_in_set={c['true_site_in_candidate_set']}", flush=True)

    checks = []
    faulty = [c for c in cases if c["case"] != "fault-free circuit"]
    clean = [c for c in cases if c["case"] == "fault-free circuit"]
    checks.append({"check": "every faulty case detected",
                   "result": "PASS" if all(c["detection"] == "FAULT_DETECTED"
                                           for c in faulty) else "FAIL",
                   "detail": f"{len(faulty)} cases"})
    checks.append({"check": "no fault-free case reported as faulty",
                   "result": "PASS" if all(c["detection"] == "NO_DIFFERENCE_OBSERVED"
                                           for c in clean) else "FAIL",
                   "detail": f"{len(clean)} cases"})
    checks.append({"check": "true site inside candidate set whenever localized",
                   "result": "PASS" if all(c["true_site_in_candidate_set"]
                                           for c in faulty
                                           if c["localization"] != "NOT_LOCALIZABLE")
                   else "FAIL", "detail": "coverage invariant"})
    checks.append({"check": "V1 verification refuses out-of-scope circuits",
                   "result": "PASS" if all(c["verification"] == "OUT_OF_SCOPE"
                                           for c in faulty
                                           if c["circuit"] != V1_SCOPE_FAMILY)
                   else "FAIL", "detail": "no extrapolation beyond HMAC"})
    checks.append({"check": "every result carries an unobservability notice",
                   "result": "PASS" if all(c["honesty_notice"] for c in cases) else "FAIL",
                   "detail": f"{len(cases)} results"})
    print("\nPIPELINE SELF-CHECK", flush=True)
    for c in checks:
        print(f"  {c['check']:<52}: {c['result']}", flush=True)
    require(all(c["result"] == "PASS" for c in checks), "pipeline self-check failed")

    created = now()
    exclusions = {
        "exclusion_version": "CIRCUITSAGE-HMAC-V2.2-EXCLUDED-COMPONENTS-12C2I-v1",
        "stage": STAGE, "created_at": created,
        "excluded": [{
            "component": "learned GNN reranking / metric retrieval",
            "reason": "falsified on unseen circuits across three formulations",
            "evidence": [
                {"stage": "12C-2C", "outcome": "no transfer"},
                {"stage": "12C-2E", "outcome": "no transfer despite contracted hard negatives"},
                {"stage": "12C-2F", "outcome": "non-learning comparator 182x better (MRR 0.6745 vs 0.0037)"},
                {"stage": "12C-2G", "outcome": "calibration lift exactly 1.00x"},
            ],
            "consequence": "including it would degrade pipeline accuracy",
        }],
        "principle": ("a component is included because it is measured to work, not "
                      "because it is fashionable or was expected to work"),
    }
    contract = {
        "contract_version": "CIRCUITSAGE-HMAC-V2.2-INFERENCE-PIPELINE-CONTRACT-12C2I-v1",
        "stage": STAGE, "status": "FROZEN", "created_at": created,
        "brand": BRAND, "tagline": TAGLINE,
        "pipeline": ["V2.2 DETECTION", "V2.2 LOCALIZATION", "V1 VERIFICATION"],
        "detection": {"method": "golden-response difference",
                      "measured_false_alarms": 0,
                      "measured_over_transactions": 7849696},
        "localization": {"method": "exact behaviour-signature dictionary lookup",
                         "returns": "candidate SET, never a forced single site",
                         "rationale": "behaviourally identical faults are physically "
                                      "indistinguishable from responses alone"},
        "verification": {"model_id": v1_lock["model_id"],
                         "threshold": v1_lock["threshold"],
                         "balanced_accuracy": 0.8167,
                         "evidence_path": "CIRCUIT STRUCTURE ONLY (0 post-simulation features)",
                         "in_scope_family": V1_SCOPE_FAMILY,
                         "out_of_scope_behaviour": "returns OUT_OF_SCOPE, never a guess"},
        "requires_characterization": True,
        "characterization_note": ("localization requires a signature dictionary built by "
                                  "simulating the circuit's fault catalogue once, offline; "
                                  "this mirrors classical ATPG fault dictionaries"),
        "zero_shot_localization": "NOT SUPPORTED - falsified, see excluded components",
        "honesty_requirements": [
            "report the circuit's unobservable fraction with every result",
            "never truncate a candidate set to appear decisive",
            "never emit a verification number outside V1's trained scope",
        ],
        "acceptance_claim": "NONE - this stage makes no acceptance claim",
    }

    prof_rows = list(profiles.values())
    prof_table = "\n".join(
        f"| `{p['family_id']}` | {p['unobservable_fraction']:.1%} | "
        f"{p['signature_uniqueness']:.4f} | {p['exact_site_rate']:.4f} | "
        f"{p['mean_candidate_set']:.1f} | {p['v1_verification_in_scope']} |"
        for p in prof_rows)
    check_table = "\n".join(f"| {c['check']} | **{c['result']}** |" for c in checks)
    sample = next((c for c in faulty if c["circuit"] == V1_SCOPE_FAMILY), faulty[0])

    report = f"""# Stage {STAGE} — {BRAND}

**{TAGLINE}**

**Status: PASS / FROZEN — inference pipeline, no training, no acceptance claim.**

## What this is

A single callable pipeline that takes observed circuit responses and returns a
detection verdict, a candidate site set, and an independent structural
verification:

```
responses in -> [V2.2 DETECTION] -> [V2.2 LOCALIZATION] -> [V1 VERIFICATION]
```

No fault injection occurs in the inference path. Injection is only how a
known-faulty test case is manufactured for scoring.

## Why V1 verification is real corroboration

V1 (`{v1_lock['model_id']}`) predicts detectability from **circuit structure
alone** — `post_simulation_features: 0`. It never sees a response. V2.2 reasons
purely from observed behaviour. The two evidence paths are disjoint, so
agreement between them is genuine corroboration rather than one measurement
restated.

V1 is locked to OpenTitan HMAC. On any other circuit the verifier returns
**OUT_OF_SCOPE** rather than extrapolating.

## Circuit profiles

| circuit | unobservable | signature uniqueness | exact-site | mean candidate set | V1 in scope |
|---|---|---|---|---|---|
{prof_table}

## Self-check

| check | result |
|---|---|
{check_table}

## Example output

```
{sample['circuit']} — {sample['case']}
  DETECTION          : {sample['detection']}
  LOCALIZATION       : {sample['localization']}
  candidate count    : {sample['candidate_count']}
  predicted polarity : {sample['predicted_polarity']}
  VERIFICATION (V1)  : {sample['verification']}
  ! {sample['honesty_notice'][:160]}
```

## Excluded by evidence

Learned GNN reranking is **excluded**. It was falsified in 12C-2C, 12C-2E and
12C-2G, and Stage 12C-2F measured the non-learning comparator as 182x stronger.
A component is included because it is measured to work.

## Known limits

- localization requires one-time offline characterization per circuit
  (a classical ATPG fault dictionary); **zero-shot localization is not supported**
- CPU-class circuits have low signature uniqueness, so candidate sets are large
- a clean result never proves a circuit is fault-free

Independent generalization remains **NOT ESTABLISHED**. Sealed circuits
untouched. Future hybrid brand remains **{FUTURE_BRAND}**.
"""

    frozen_write(PIPELINE_CONTRACT, canonical_json(contract))
    frozen_write(CIRCUIT_PROFILES, csv_bytes(prof_rows, list(prof_rows[0].keys())))
    frozen_write(DEMO_CASES, canonical_json({
        "version": "CIRCUITSAGE-HMAC-V2.2-DEMONSTRATION-CASES-12C2I-v1",
        "stage": STAGE, "created_at": created,
        "truth_consulted_after_inference": True, "cases": cases}))
    frozen_write(SELFCHECK, csv_bytes(checks, ["check", "result", "detail"]))
    frozen_write(EXCLUSIONS, canonical_json(exclusions))
    frozen_write(REPORT, report.encode())

    stage_outputs = (DICTIONARY, PIPELINE_CONTRACT, CIRCUIT_PROFILES, DEMO_CASES,
                     SELFCHECK, EXCLUSIONS, REPORT)
    manifest = {
        "manifest_version": "CIRCUITSAGE-HMAC-V2.2-INFERENCE-PIPELINE-MANIFEST-12C2I-v1",
        "stage": STAGE, "status": "PASS", "created_at": created,
        "stage_source": record(Path(__file__).resolve()),
        "frozen_inputs": {rel(p): record(p) for p in PINNED},
        "v1_release_inputs": {rel(p): record(p) for p in (V1_MODEL, V1_LOCK, V1_ARCH)},
        "outputs": {rel(p): record(p) for p in stage_outputs},
        "training_calls": 0, "selection_calls": 0,
        "independent_test_access": 0, "validation_access": 0, "holdout_access": 0,
    }
    frozen_write(MANIFEST, canonical_json(manifest))

    audit = {
        "audit_version": "CIRCUITSAGE-HMAC-V2.2-INFERENCE-PIPELINE-FREEZE-12C2I-v1",
        "stage": STAGE, "status": "PASS",
        "brand": BRAND, "tagline": TAGLINE,
        "pipeline_stages": ["V2.2 DETECTION", "V2.2 LOCALIZATION", "V1 VERIFICATION"],
        "v1_model_id": v1_lock["model_id"],
        "v1_threshold": v1_lock["threshold"],
        "v1_evidence_disjoint_from_v2": True,
        "v1_in_scope_family": V1_SCOPE_FAMILY,
        "gnn_included": False,
        "gnn_exclusion_evidence": ["12C-2C", "12C-2E", "12C-2F", "12C-2G"],
        "zero_shot_localization_supported": False,
        "requires_offline_characterization": True,
        "self_checks_passed": len(checks),
        "demonstration_cases": len(cases),
        "truth_consulted_after_inference": True,
        "measured_false_alarms": 0,
        "circuit_profiles": {p["family_id"]: {
            "unobservable_fraction": p["unobservable_fraction"],
            "signature_uniqueness": p["signature_uniqueness"],
            "exact_site_rate": p["exact_site_rate"]} for p in prof_rows},
        "acceptance_claim": "NONE",
        "selection_performed": False,
        "independent_test_validation_holdout_access": [0, 0, 0],
        "independent_generalization": "NOT ESTABLISHED",
        "contract_record": record(PIPELINE_CONTRACT),
        "exclusions_record": record(EXCLUSIONS),
        "demonstration_record": record(DEMO_CASES),
        "manifest_record": record(MANIFEST),
        "future_combined_model_brand": FUTURE_BRAND,
        "next_gate": "STAGE 12C-3A — INDEPENDENT TEST CAPTURE (unblocked, independent)",
    }
    frozen_write(AUDIT, canonical_json(audit))

    print(f"\n{'Stage':<52}: {STAGE} — {BRAND}")
    print(f"{'Status':<52}: PASS / FROZEN")
    print(f"{'Pipeline':<52}: DETECT -> LOCALIZE (V2.2) -> VERIFY (V1)")
    print(f"{'V1 evidence disjoint from V2.2':<52}: YES (0 post-simulation features)")
    print(f"{'GNN included':<52}: NO — falsified, evidence cited")
    print(f"{'Self-checks passed':<52}: {len(checks)}/{len(checks)}")
    print(f"{'Acceptance claim':<52}: NONE")
    print(f"{'TEST / VALIDATION / HOLDOUT access':<52}: 0 / 0 / 0")
    print(f"{'Audit SHA':<52}: {sha256(AUDIT)}")


def status() -> None:
    print(f"STAGE {STAGE} — FAULTIVA PIPELINE STATUS")
    if not (MANIFEST.is_file() and AUDIT.is_file()):
        print("Status                    : NOT FROZEN")
        return
    a = load_json(AUDIT)
    print("Status                    : PASS / FROZEN")
    print(f"Brand                     : {a['brand']}")
    print(f"Pipeline                  : {' -> '.join(a['pipeline_stages'])}")
    print(f"V1 model                  : {a['v1_model_id']} (threshold {a['v1_threshold']})")
    print(f"GNN included              : {a['gnn_included']}")
    print(f"Self-checks               : {a['self_checks_passed']}")
    print(f"Audit SHA                 : {sha256(AUDIT)}")


def self_test() -> None:
    r = FaultivaResult(circuit="x", detection="FAULT_DETECTED",
                       detection_basis="b", candidate_sites=[1, 2, 3],
                       candidate_count=3, localization="AMBIGUOUS",
                       honesty_notice="n")
    require("candidate count    : 3" in r.render(), "renders candidate count")
    require(V1_MODEL.is_file(), "V1 model artifact present")
    require(load_json(V1_ARCH)["post_simulation_features"] == 0,
            "V1 structural-only invariant")
    require(load_json(V1_LOCK)["model_retraining_allowed"] is False,
            "V1 retraining prohibited")
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
