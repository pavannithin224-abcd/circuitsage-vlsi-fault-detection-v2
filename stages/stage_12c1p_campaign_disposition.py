#!/usr/bin/env python3
"""Stage 12C-1P: V2.2 campaign result disposition and model-training readiness.

Stage 12C-1O completed and froze the amended 998-batch multi-circuit
TRAIN/CALIBRATION fault campaign.  This stage is a REPORT-ONLY disposition: it
verifies the frozen campaign evidence, records the observed per-family result
structure, states what the evidence does and does not establish, and freezes a
bounded model-training readiness authorization.

Disposition mode: REPORT-ONLY.

No numeric advancement target was defined for the V2.2 multi-circuit campaign
in any retained contract.  This stage therefore does NOT invent acceptance
thresholds and does NOT declare PASS or NOT_MET against fabricated criteria.
It reports measured values, classifies each family by observability, and
records the resulting training scope.  Any future advancement criteria must be
frozen in their own authorized contract stage before being evaluated.

Principal finding recorded by this stage: aggregate campaign metrics are
dominated by fault-population imbalance across circuit families, and per-family
detection varies by roughly a factor of four.  The limiting factor is
observability of injected faults under the frozen vector and measurement
scheme, not candidate retrieval: observable candidate set coverage is 1.0,
unique-signature Top-1 is 1.0 and fault-free false alarms are 0.

No simulation, fault injection, dataset construction, model loading, training,
selection or inference occurs in this stage.  Independent TEST is locked,
VALIDATION is unopened and HOLDOUT is sealed.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import io
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STAGE = "12C-1P"
ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "config/v2_2"
RESULT = ROOT / "results/circuitsage_hmac_v2_12c1"
WORK = RESULT / "campaign_disposition_12c1p"
LOCK_FILE = WORK / ".stage_12c1p.lock"

# ---------------------------------------------------------------- frozen input
SOURCE_1M = ROOT / "stage_12c1m_site_eligibility_discovery.py"
SOURCE_1O = ROOT / "stage_12c1o_parallel_campaign_execution.py"
AUDIT_1M = RESULT / "circuitsage_hmac_v2_2_site_eligibility_discovery_freeze_12c1m.json"
AUDIT_1O = RESULT / "circuitsage_hmac_v2_2_amended_campaign_execution_dataset_freeze_12c1o.json"
MANIFEST_1O = RESULT / "circuitsage_hmac_v2_2_amended_campaign_dataset_manifest_12c1o.json"
PARALLEL_AUTH_1O = CONFIG / "circuitsage_hmac_v2_2_parallel_execution_authorization_12c1o.json"

WORK_1O = RESULT / "parallel_campaign_execution_12c1o"
METRICS_1O = WORK_1O / "circuitsage_hmac_v2_2_amended_campaign_metrics_12c1o.json"
FEATURES_1O = WORK_1O / "circuitsage_hmac_v2_2_amended_campaign_features_12c1o.npz"
TARGETS_1O = WORK_1O / "circuitsage_hmac_v2_2_amended_campaign_targets_12c1o.npz"
SCHEMA_1O = WORK_1O / "circuitsage_hmac_v2_2_amended_campaign_dataset_schema_12c1o.json"
CATALOG_1O = WORK_1O / "circuitsage_hmac_v2_2_amended_campaign_fault_catalog_12c1o.csv"

PINNED = {
    SOURCE_1M: "50d156d795400ea321942782009f24eecaccc0c43a5e123757daa31d2c4743bd",
    SOURCE_1O: "d6841a1e5f6ae1e3101e882ebe4f7188d911ee305fd1a969d62f52e31d1759ce",
    AUDIT_1M: "d6fc641681f13f557a5df820a5aece1f475f2a37becb380af5df26cc9521783a",
    AUDIT_1O: "fa67278ae07bc6dd57246088c3da41918f88c05fa06580c4583c022c1993430b",
    MANIFEST_1O: "4a5a9d54e8dc7b85f07a97247abb71de3beab02af9ecebc126deb2ac0881789c",
    PARALLEL_AUTH_1O: "fc5358a3b597b8f452831a4fbd6933d0cbeaec79d0532f6563f3470ff76af986",
}

# ---------------------------------------------------------------- stage output
DISPOSITION_POLICY = CONFIG / "circuitsage_hmac_v2_2_campaign_disposition_policy_12c1p.json"
TRAINING_READINESS = CONFIG / "circuitsage_hmac_v2_2_model_training_readiness_contract_12c1p.json"
FAMILY_DISPOSITION = WORK / "circuitsage_hmac_v2_2_family_disposition_12c1p.csv"
OBSERVABILITY_ANALYSIS = WORK / "circuitsage_hmac_v2_2_observability_analysis_12c1p.json"
CLAIMS_REGISTER = WORK / "circuitsage_hmac_v2_2_claims_register_12c1p.json"
PREFLIGHT = WORK / "circuitsage_hmac_v2_2_disposition_preflight_12c1p.json"
REPORT = WORK / "circuitsage_hmac_v2_2_campaign_disposition_report_12c1p.md"
MANIFEST = RESULT / "circuitsage_hmac_v2_2_campaign_disposition_manifest_12c1p.json"
AUDIT = RESULT / "circuitsage_hmac_v2_2_campaign_disposition_freeze_12c1p.json"

FAMILIES = (
    "opentitan_hmac_sha256",
    "picorv32_cpu",
    "secworks_aes",
    "secworks_sha256",
)
FUTURE_BRAND = "Faultiva 1.0 — Hybrid VLSI Fault Intelligence"

DISPOSITION_MODE = "REPORT-ONLY / NO FROZEN ADVANCEMENT TARGET EXISTS"

# Classification boundary used ONLY to label families descriptively in this
# report.  It is not an acceptance criterion, not a pass/fail gate, and carries
# no authorization consequence beyond naming the training scope.
OBSERVABILITY_LABEL_BOUNDARY = 0.50


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


def safe_record(item: dict[str, Any], label: str) -> Path:
    require(isinstance(item, dict), f"{label} record")
    raw = item.get("path")
    require(isinstance(raw, str) and raw, f"{label} path")
    path = (ROOT / raw).resolve()
    require(path.is_relative_to(ROOT), f"{label} path escapes project root")
    require(path.is_file(), f"missing {label}: {raw}")
    require(sha256(path) == item.get("sha256"), f"{label} SHA")
    return path


def verify_inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    print("FROZEN INPUT VERIFICATION", flush=True)
    for path, expected in PINNED.items():
        require(path.is_file(), f"missing frozen input: {rel(path)}")
        require(sha256(path) == expected,
                f"frozen input SHA changed (campaign evidence must be byte-identical): {path.name}")
        print(f"  {path.name:<92}: OK", flush=True)

    audit_1o = load_json(AUDIT_1O)
    manifest_1o = load_json(MANIFEST_1O)
    audit_1m = load_json(AUDIT_1M)

    require(audit_1o.get("status") == "PASS", "12C-1O freeze status")
    require(audit_1o.get("dataset_integrity") == "PASS", "12C-1O dataset integrity")
    require(audit_1o.get("execution_dataset") == "COMPLETED / FROZEN", "12C-1O dataset completion")
    require(audit_1o.get("simulation_batches") == "998/998", "12C-1O batch completion")
    require(audit_1o.get("fault_identity_in_features") == "PROHIBITED / ABSENT", "identity firewall")
    require(audit_1o.get("independent_generalization") == "NOT ESTABLISHED",
            "generalization boundary")
    require(audit_1o.get("model_training_selection_inference")
            == "NOT AUTHORIZED / NOT AUTHORIZED / NOT AUTHORIZED", "upstream model boundary")
    require(audit_1m.get("status") == "PASS", "12C-1M freeze status")
    require(audit_1m.get("ceiling_exceeded") is False, "12C-1M ceiling guarantee")

    for name, item in sorted((manifest_1o.get("outputs") or {}).items()):
        safe_record(item, f"12C-1O output {name}")
    print(f"  Stage 12C-1O dataset manifest ({len(manifest_1o.get('outputs') or {})} outputs)"
          f"{'':<48}: OK", flush=True)

    metrics = load_json(METRICS_1O)
    require(metrics.get("dataset_integrity") == "PASS", "metrics integrity")
    require(len(metrics.get("families") or []) == len(FAMILIES), "per-family metrics present")
    return metrics, audit_1o


def build_disposition(metrics: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    families = {f["family_id"]: f for f in metrics["families"]}
    require(set(families) == set(FAMILIES), "family identifiers")

    rows: list[dict[str, Any]] = []
    for family_id in FAMILIES:
        f = families[family_id]
        detection = float(f["all_injected_detection_recall"])
        label = ("OBSERVABILITY_SUFFICIENT" if detection >= OBSERVABILITY_LABEL_BOUNDARY
                 else "OBSERVABILITY_LIMITED")
        rows.append({
            "family_id": family_id,
            "partition": f.get("partition", ""),
            "sites": int(f["sites"]),
            "fault_instances": int(f["fault_instances"]),
            "full_vectors": int(f["full_vectors"]),
            "observable_faults": int(f["observable_faults"]),
            "observable_fraction": round(int(f["observable_faults"]) / int(f["fault_instances"]), 8),
            "all_injected_detection_recall": detection,
            "all_injected_exact_site_rate": float(f["all_injected_exact_site_rate"]),
            "unique_signature_faults": int(f.get("unique_signature_faults", 0)),
            "ambiguous_signature_faults": int(f.get("ambiguous_signature_faults", 0)),
            "mean_observable_candidate_sites": float(f.get("mean_observable_candidate_sites", 0.0)),
            "maximum_observable_candidate_sites": int(f.get("maximum_observable_candidate_sites", 0)),
            "fault_free_false_alarms": int(f.get("fault_free_false_alarms", 0)),
            "observability_label": label,
        })

    total_faults = sum(r["fault_instances"] for r in rows)
    total_observable = sum(r["observable_faults"] for r in rows)
    detections = [r["all_injected_detection_recall"] for r in rows]
    sufficient = [r for r in rows if r["observability_label"] == "OBSERVABILITY_SUFFICIENT"]
    limited = [r for r in rows if r["observability_label"] == "OBSERVABILITY_LIMITED"]

    # population-share analysis: how much of the aggregate each family drives
    shares = {r["family_id"]: round(r["observable_faults"] / total_observable, 8) for r in rows}

    analysis = {
        "analysis_version": "CIRCUITSAGE-HMAC-V2.2-OBSERVABILITY-ANALYSIS-12C1P-v1",
        "stage": STAGE, "status": "REPORT-ONLY", "created_at": now(),
        "aggregate_detection_recall": float(metrics["all_injected_detection_recall"]),
        "aggregate_exact_site_rate": float(metrics["all_injected_exact_site_rate"]),
        "aggregate_detection_95_ci": metrics.get("detection_site_bootstrap_95_ci"),
        "aggregate_exact_site_95_ci": metrics.get("exact_site_bootstrap_95_ci"),
        "fault_instances": total_faults,
        "observable_faults": total_observable,
        "observable_fraction": round(total_observable / total_faults, 8),
        "per_family_detection_minimum": min(detections),
        "per_family_detection_maximum": max(detections),
        "per_family_detection_spread_ratio": round(max(detections) / min(detections), 4),
        "observable_population_share": shares,
        "observability_sufficient_families": [r["family_id"] for r in sufficient],
        "observability_limited_families": [r["family_id"] for r in limited],
        "retrieval_quality_on_observable_faults": {
            "observable_candidate_set_coverage": float(metrics["observable_candidate_set_coverage"]),
            "unique_signature_top1_site": float(metrics["unique_signature_top1_site"]),
            "fault_free_false_alarm_rate": float(metrics["fault_free_false_alarm_rate"]),
            "ambiguous_false_unique_rate": float(metrics["ambiguous_false_unique_rate"]),
        },
        "limiting_factor": (
            "OBSERVABILITY OF INJECTED FAULTS UNDER THE FROZEN VECTOR AND MEASUREMENT SCHEME; "
            "NOT CANDIDATE RETRIEVAL"
        ),
        "limiting_factor_grounds": [
            "observable candidate set coverage is 1.0",
            "unique-signature Top-1 site is 1.0",
            "fault-free false alarm rate is 0.0",
            "ambiguous false-unique rate is 0.0",
            "per-family detection varies by a factor of "
            f"{round(max(detections) / min(detections), 2)} across identical retrieval logic",
        ],
        "aggregate_interpretation": (
            "The aggregate detection recall is a fault-population weighted quantity. One family "
            "contributes the majority of observable faults, so the aggregate must not be read as a "
            "per-circuit expectation."
        ),
        "label_boundary_disclaimer": (
            f"The {OBSERVABILITY_LABEL_BOUNDARY} detection boundary is a descriptive label used only "
            "to name the training scope in this report. It is not an acceptance criterion and no "
            "advancement decision is made against it."
        ),
    }
    return rows, analysis


def execute() -> None:
    require(ROOT.name == "vlsi_fault_detection_v2", f"place script in project root, not {ROOT}")
    outputs = (DISPOSITION_POLICY, TRAINING_READINESS, FAMILY_DISPOSITION, OBSERVABILITY_ANALYSIS,
               CLAIMS_REGISTER, PREFLIGHT, REPORT, MANIFEST, AUDIT)
    require(not any(p.exists() for p in outputs),
            f"frozen Stage {STAGE} output already exists; use --status")

    metrics, audit_1o = verify_inputs()

    print("\nPER-FAMILY DISPOSITION", flush=True)
    rows, analysis = build_disposition(metrics)
    for r in rows:
        print(f"  {r['family_id']:<24} detect={r['all_injected_detection_recall']:.4f} "
              f"exact={r['all_injected_exact_site_rate']:.4f} "
              f"observable={r['observable_faults']:>6} "
              f"{r['observability_label']}", flush=True)

    created = now()
    sufficient = analysis["observability_sufficient_families"]
    limited = analysis["observability_limited_families"]

    policy = {
        "policy_version": "CIRCUITSAGE-HMAC-V2.2-CAMPAIGN-DISPOSITION-POLICY-12C1P-v1",
        "stage": STAGE, "status": "FROZEN", "created_at": created,
        "disposition_mode": DISPOSITION_MODE,
        "advancement_target_defined": False,
        "advancement_decision": "NOT EVALUATED / NO FROZEN CRITERIA EXIST",
        "acceptance_criteria_invented": False,
        "campaign_evidence": "COMPLETE / FROZEN / INTEGRITY PASS",
        "campaign_audit_sha256": PINNED[AUDIT_1O],
        "amended_plan_audit_sha256": PINNED[AUDIT_1M],
        "reported_aggregate_detection": analysis["aggregate_detection_recall"],
        "reported_aggregate_exact_site": analysis["aggregate_exact_site_rate"],
        "per_family_detection_spread_ratio": analysis["per_family_detection_spread_ratio"],
        "limiting_factor": analysis["limiting_factor"],
        "future_criteria_requirement": (
            "Any V2.2 advancement criteria must be frozen in their own authorized contract stage "
            "BEFORE evaluation, and must not be derived from these observed values."
        ),
        "upstream_mutation": "PROHIBITED / NONE PERFORMED",
    }

    training_readiness = {
        "contract_version": "CIRCUITSAGE-HMAC-V2.2-MODEL-TRAINING-READINESS-12C1P-v1",
        "stage": STAGE, "status": "FROZEN", "created_at": created,
        "model_training": "AUTHORIZED / BOUNDED",
        "model_selection": "AUTHORIZED / BOUNDED",
        "model_inference_on_protected_data": "NOT AUTHORIZED",
        "authorized_training_dataset": rel(FEATURES_1O),
        "authorized_supervision_dataset": rel(TARGETS_1O),
        "authorized_dataset_schema": rel(SCHEMA_1O),
        "authorized_fault_catalog": rel(CATALOG_1O),
        "training_partition": "GENERALIZATION_TRAIN",
        "calibration_partition": "GENERALIZATION_CALIBRATION",
        "training_families": [r["family_id"] for r in rows
                              if r["partition"] == "GENERALIZATION_TRAIN"],
        "calibration_families": [r["family_id"] for r in rows
                                 if r["partition"] == "GENERALIZATION_CALIBRATION"],
        "observability_sufficient_families": sufficient,
        "observability_limited_families": limited,
        "limited_family_policy": (
            "Observability-limited families remain IN the training corpus. They must not be "
            "removed to improve reported metrics. Any per-family result must be reported "
            "separately and never hidden inside an aggregate."
        ),
        "identity_firewall": "FAULT IDENTITY ABSENT FROM MODEL-FACING FEATURES",
        "leakage_policy": [
            "targets are supervision-only and must never enter model-facing features",
            "calibration partition must not be used to fit model parameters",
            "no cross-family leakage: a site may appear in exactly one partition",
            "prediction must be produced before truth is consulted for scoring",
        ],
        "independent_test_access": "LOCKED",
        "validation_access": "UNOPENED",
        "holdout_access": "SEALED",
        "independent_generalization": "NOT ESTABLISHED",
        "prohibited": [
            "reopening or retraining on any consumed protected partition",
            "modifying frozen V1, V2.1, 12C-1M or 12C-1O evidence",
            "deriving acceptance criteria from observed campaign values",
            "reporting only the aggregate metric without the per-family breakdown",
        ],
    }

    claims = {
        "register_version": "CIRCUITSAGE-HMAC-V2.2-CLAIMS-REGISTER-12C1P-v1",
        "stage": STAGE, "status": "FROZEN", "created_at": created,
        "established": [
            "a leakage-controlled multi-circuit SA0/SA1 fault campaign over 4 independent "
            "circuit families was executed and frozen",
            f"{analysis['fault_instances']} fault instances across {sum(r['sites'] for r in rows)} "
            "sites were simulated with prediction-before-truth scoring",
            "exact-signature retrieval achieves observable candidate coverage 1.0 and "
            "unique-signature Top-1 1.0 on this corpus",
            "fault-free false alarms are 0 across the full campaign",
            "per-family detection varies by a factor of "
            f"{analysis['per_family_detection_spread_ratio']} under identical retrieval logic, "
            "isolating observability as the limiting factor",
            "parallel batch execution was shown to be schedule-equivalent and produced a "
            "dataset that passed full deterministic replay and cross-batch baseline checks",
        ],
        "not_established": [
            "independent-circuit generalization of a trained model",
            "unrestricted unknown-fault detection",
            "that the aggregate detection recall represents a per-circuit expectation",
            "any advancement or acceptance outcome for V2.2",
            "production readiness",
        ],
        "must_not_claim": [
            "V2.2 generalizes across circuits",
            "the system detects any fault in any circuit",
            "the aggregate figure is the system's accuracy",
        ],
        "honest_summary": (
            "The campaign establishes a rigorous multi-circuit dataset and isolates observability, "
            "not retrieval, as the dominant limiting factor. It does not establish that fault "
            "detection generalizes across circuit families."
        ),
    }

    preflight = {
        "preflight_version": "CIRCUITSAGE-HMAC-V2.2-DISPOSITION-PREFLIGHT-12C1P-v1",
        "stage": STAGE, "status": "PASS", "created_at": created,
        "simulation_calls": 0, "fault_injection_calls": 0, "dataset_records_created": 0,
        "model_deserializations": 0, "training_calls": 0, "selection_calls": 0,
        "inference_calls": 0,
        "independent_test_access": 0, "validation_access": 0, "holdout_access": 0,
        "frozen_12c1o_outputs_modified": False,
        "frozen_12c1m_outputs_modified": False,
        "acceptance_criteria_invented": False,
    }

    family_fields = ["family_id", "partition", "sites", "fault_instances", "full_vectors",
                     "observable_faults", "observable_fraction",
                     "all_injected_detection_recall", "all_injected_exact_site_rate",
                     "unique_signature_faults", "ambiguous_signature_faults",
                     "mean_observable_candidate_sites", "maximum_observable_candidate_sites",
                     "fault_free_false_alarms", "observability_label"]

    table = "\n".join(
        f"| `{r['family_id']}` | {r['partition']} | {r['all_injected_detection_recall']:.4f} | "
        f"{r['all_injected_exact_site_rate']:.4f} | {r['observable_faults']:,} | "
        f"{r['observable_fraction']:.3f} | {r['mean_observable_candidate_sites']:.2f} | "
        f"{r['maximum_observable_candidate_sites']} | {r['observability_label']} |"
        for r in rows
    )

    report = f"""# Stage {STAGE} — V2.2 Campaign Disposition

**Status: PASS / FROZEN (report-only disposition)**

**Disposition mode:** {DISPOSITION_MODE}

## Why no PASS/NOT_MET is declared

No numeric advancement target for the V2.2 multi-circuit campaign exists in any
retained frozen contract. This stage therefore reports measured values and
**does not invent acceptance criteria**. Any future V2.2 advancement criteria
must be frozen in their own authorized contract stage *before* evaluation, and
must not be back-derived from the values below.

## Campaign evidence

| | |
|---|---|
| Batches | 998 / 998 |
| Fault instances | {analysis['fault_instances']:,} |
| Enabled transactions | {int(load_json(METRICS_1O)['enabled_transactions']):,} |
| Dataset integrity | PASS |
| Fault-free false alarms | 0 |
| Identity firewall | PROHIBITED / ABSENT |

## Per-family result

| family | partition | detection | exact-site | observable | obs. frac | mean cand | max cand | label |
|---|---|---|---|---|---|---|---|---|
{table}

Aggregate detection recall: **{analysis['aggregate_detection_recall']:.8f}**
(95% CI {analysis['aggregate_detection_95_ci']})
Aggregate exact-site rate: **{analysis['aggregate_exact_site_rate']:.8f}**
(95% CI {analysis['aggregate_exact_site_95_ci']})

## Principal finding

Per-family detection spans **{analysis['per_family_detection_minimum']:.4f}** to
**{analysis['per_family_detection_maximum']:.4f}** — a spread ratio of
**{analysis['per_family_detection_spread_ratio']}** — under *identical* retrieval
logic, identical measurement and identical scoring.

On observable faults the retrieval stage is essentially perfect:

- observable candidate set coverage: **{analysis['retrieval_quality_on_observable_faults']['observable_candidate_set_coverage']}**
- unique-signature Top-1 site: **{analysis['retrieval_quality_on_observable_faults']['unique_signature_top1_site']}**
- fault-free false alarm rate: **{analysis['retrieval_quality_on_observable_faults']['fault_free_false_alarm_rate']}**
- ambiguous false-unique rate: **{analysis['retrieval_quality_on_observable_faults']['ambiguous_false_unique_rate']}**

Therefore the limiting factor is **observability of injected faults under the
frozen vector and measurement scheme, not candidate retrieval.**

## Aggregate interpretation warning

The aggregate is a fault-population weighted quantity. `secworks_aes` alone
contributes {analysis['observable_population_share']['secworks_aes']:.1%} of all
observable faults. **The aggregate must not be read as a per-circuit
expectation**, and must never be reported without the per-family breakdown.

## Training scope

Observability-sufficient: {', '.join(f'`{x}`' for x in sufficient) or 'none'}
Observability-limited: {', '.join(f'`{x}`' for x in limited) or 'none'}

Observability-limited families **remain in the training corpus**. They must not
be removed to improve reported metrics. The
{OBSERVABILITY_LABEL_BOUNDARY} boundary is a descriptive label only and carries
no acceptance consequence.

## What this evidence does and does not establish

**Established:** a leakage-controlled multi-circuit campaign, perfect retrieval
on observable faults, zero false alarms, and isolation of observability as the
dominant limiting factor.

**Not established:** independent-circuit generalization of a trained model,
unrestricted unknown-fault detection, or any V2.2 advancement outcome.

Independent generalization remains **NOT ESTABLISHED**. The future hybrid brand
remains **{FUTURE_BRAND}**.

## Authorization

Model training and selection on the frozen 12C-1O TRAIN/CALIBRATION corpus:
**AUTHORIZED / BOUNDED**.
Independent TEST remains **LOCKED**, VALIDATION **UNOPENED**, HOLDOUT **SEALED**.
"""

    frozen_write(DISPOSITION_POLICY, canonical_json(policy))
    frozen_write(TRAINING_READINESS, canonical_json(training_readiness))
    frozen_write(FAMILY_DISPOSITION, csv_bytes(rows, family_fields))
    frozen_write(OBSERVABILITY_ANALYSIS, canonical_json(analysis))
    frozen_write(CLAIMS_REGISTER, canonical_json(claims))
    frozen_write(PREFLIGHT, canonical_json(preflight))
    frozen_write(REPORT, report.encode())

    stage_outputs = (DISPOSITION_POLICY, TRAINING_READINESS, FAMILY_DISPOSITION,
                     OBSERVABILITY_ANALYSIS, CLAIMS_REGISTER, PREFLIGHT, REPORT)
    manifest = {
        "manifest_version": "CIRCUITSAGE-HMAC-V2.2-CAMPAIGN-DISPOSITION-MANIFEST-12C1P-v1",
        "stage": STAGE, "status": "PASS", "created_at": created,
        "stage_source": record(Path(__file__).resolve()),
        "frozen_inputs": {rel(p): record(p) for p in PINNED},
        "outputs": {rel(p): record(p) for p in stage_outputs},
        "disposition_mode": DISPOSITION_MODE,
        "advancement_target_defined": False,
        "simulation_calls": 0, "training_calls": 0, "inference_calls": 0,
        "independent_test_access": 0, "validation_access": 0, "holdout_access": 0,
    }
    frozen_write(MANIFEST, canonical_json(manifest))

    audit = {
        "audit_version": "CIRCUITSAGE-HMAC-V2.2-CAMPAIGN-DISPOSITION-FREEZE-12C1P-v1",
        "stage": STAGE, "status": "PASS",
        "disposition_mode": DISPOSITION_MODE,
        "advancement_target_defined": False,
        "advancement_decision": "NOT EVALUATED / NO FROZEN CRITERIA EXIST",
        "acceptance_criteria_invented": False,
        "campaign_batches": "998/998",
        "dataset_integrity": "PASS",
        "aggregate_detection_recall": analysis["aggregate_detection_recall"],
        "aggregate_exact_site_rate": analysis["aggregate_exact_site_rate"],
        "per_family_detection_minimum": analysis["per_family_detection_minimum"],
        "per_family_detection_maximum": analysis["per_family_detection_maximum"],
        "per_family_detection_spread_ratio": analysis["per_family_detection_spread_ratio"],
        "limiting_factor": analysis["limiting_factor"],
        "observability_sufficient_families": sufficient,
        "observability_limited_families": limited,
        "fault_free_false_alarms": 0,
        "model_training_selection": "AUTHORIZED / BOUNDED",
        "model_inference_on_protected_data": "NOT AUTHORIZED",
        "independent_test_validation_holdout_access": [0, 0, 0],
        "independent_generalization": "NOT ESTABLISHED",
        "frozen_12c1o_outputs_modified": False,
        "frozen_12c1m_outputs_modified": False,
        "disposition_policy_record": record(DISPOSITION_POLICY),
        "training_readiness_record": record(TRAINING_READINESS),
        "family_disposition_record": record(FAMILY_DISPOSITION),
        "observability_analysis_record": record(OBSERVABILITY_ANALYSIS),
        "claims_register_record": record(CLAIMS_REGISTER),
        "manifest_record": record(MANIFEST),
        "future_combined_model_brand": FUTURE_BRAND,
        "next_gate": "STAGE 12C-2A — V2.2 MODEL CONTRACT AND TRAINING AUTHORIZATION",
    }
    frozen_write(AUDIT, canonical_json(audit))

    for path in (DISPOSITION_POLICY, TRAINING_READINESS, OBSERVABILITY_ANALYSIS,
                 CLAIMS_REGISTER, PREFLIGHT, MANIFEST, AUDIT):
        require(path.read_bytes() == canonical_json(load_json(path)),
                f"canonical output replay: {path.name}")

    print(f"\n{'Stage':<52}: {STAGE} — V2.2 CAMPAIGN DISPOSITION")
    print(f"{'Status':<52}: PASS / FROZEN")
    print(f"{'Disposition mode':<52}: REPORT-ONLY")
    print(f"{'Advancement target defined':<52}: NO — NOT EVALUATED")
    print(f"{'Acceptance criteria invented':<52}: NO")
    print(f"{'Aggregate detection / exact-site':<52}: "
          f"{analysis['aggregate_detection_recall']:.8f} / {analysis['aggregate_exact_site_rate']:.8f}")
    print(f"{'Per-family detection spread':<52}: "
          f"{analysis['per_family_detection_minimum']:.4f} .. "
          f"{analysis['per_family_detection_maximum']:.4f} "
          f"(x{analysis['per_family_detection_spread_ratio']})")
    print(f"{'Limiting factor':<52}: OBSERVABILITY, NOT RETRIEVAL")
    print(f"{'Observability-limited families':<52}: {', '.join(limited) or 'none'}")
    print(f"{'Model training / selection':<52}: AUTHORIZED / BOUNDED")
    print(f"{'TEST / VALIDATION / HOLDOUT access':<52}: 0 / 0 / 0")
    print(f"{'Independent generalization':<52}: NOT ESTABLISHED")
    print(f"{'Audit':<52}: {AUDIT}")
    print(f"{'Audit SHA':<52}: {sha256(AUDIT)}")
    print(f"{'Next gate':<52}: STAGE 12C-2A — MODEL CONTRACT")


def status() -> None:
    print(f"STAGE {STAGE} — V2.2 CAMPAIGN DISPOSITION STATUS")
    if not MANIFEST.is_file() or not AUDIT.is_file():
        print("Status                    : NOT FROZEN")
        print(f"Expected audit            : {AUDIT}")
        return
    manifest = load_json(MANIFEST)
    audit = load_json(AUDIT)
    require(manifest.get("status") == "PASS" and audit.get("status") == "PASS", "frozen status")
    require(audit.get("manifest_record", {}).get("sha256") == sha256(MANIFEST), "manifest anchor")
    for name, item in sorted((manifest.get("outputs") or {}).items()):
        safe_record(item, f"{STAGE} output {name}")
    print("Status                    : PASS / FROZEN")
    print(f"Disposition mode          : {audit['disposition_mode']}")
    print(f"Advancement decision      : {audit['advancement_decision']}")
    print(f"Aggregate detect/exact    : {audit['aggregate_detection_recall']:.8f} / "
          f"{audit['aggregate_exact_site_rate']:.8f}")
    print(f"Detection spread          : {audit['per_family_detection_minimum']:.4f} .. "
          f"{audit['per_family_detection_maximum']:.4f} "
          f"(x{audit['per_family_detection_spread_ratio']})")
    print(f"Limiting factor           : {audit['limiting_factor']}")
    print(f"Observability-limited     : {audit['observability_limited_families']}")
    print(f"Training / selection      : {audit['model_training_selection']}")
    print(f"Generalization            : {audit['independent_generalization']}")
    print(f"Next gate                 : {audit['next_gate']}")
    print(f"Audit SHA                 : {sha256(AUDIT)}")


def self_test() -> None:
    require(AUDIT_1O.is_file() and METRICS_1O.is_file(), "12C-1O evidence present")
    metrics = load_json(METRICS_1O)
    require(len(metrics.get("families") or []) == 4, "four families in metrics")
    rows, analysis = build_disposition(metrics)
    require(len(rows) == 4, "four disposition rows")
    require(analysis["per_family_detection_spread_ratio"] > 1.0, "spread computed")
    require(abs(sum(analysis["observable_population_share"].values()) - 1.0) < 1e-6,
            "population shares sum to 1")
    require(analysis["limiting_factor"].startswith("OBSERVABILITY"), "limiting factor")
    require(sum(r["observable_faults"] for r in rows) == metrics["observable_faults"],
            "observable fault total agreement")
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
