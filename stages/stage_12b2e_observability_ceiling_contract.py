#!/usr/bin/env python3
"""Stage 12B-2E: observability-ceiling review and alternative-measurement freeze.

Reviews the frozen Stage 12B-2D vector-selection result and establishes whether
more external HMAC key/message tests can satisfy the detection target. It then
freezes a leakage-safe, simulation-only instrumentation feasibility contract.
No simulation, probe capture, model loading, training, or inference occurs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any


STAGE = "12B-2E"
ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "config/v2_1_improvement"
RESULT = ROOT / "results/circuitsage_hmac_v2_12b2"
WORK = RESULT / "observability_ceiling_review_12b2e"

SOURCE_2D = ROOT / "stage_12b2d_adaptive_vector_selection_contract.py"
SELECTION_DIR = RESULT / "adaptive_vector_selection_12b2d"
SELECTED_VECTORS = SELECTION_DIR / "circuitsage_hmac_v2_1_selected_adaptive_vectors_12b2d.npz"
SELECTION_TRACE = SELECTION_DIR / "circuitsage_hmac_v2_1_adaptive_vector_selection_trace_12b2d.csv"
SELECTION_METRICS = SELECTION_DIR / "circuitsage_hmac_v2_1_adaptive_vector_selection_metrics_12b2d.json"
SELECTION_LOCK = SELECTION_DIR / "circuitsage_hmac_v2_1_adaptive_vector_selection_lock_12b2d.json"
CAMPAIGN_CONTRACT_2D = CONFIG / "circuitsage_hmac_v2_1_full_repair_campaign_contract_12b2d.json"
MANIFEST_2D = RESULT / "circuitsage_hmac_v2_1_adaptive_selection_manifest_12b2d.json"
AUDIT_2D = RESULT / "circuitsage_hmac_v2_1_adaptive_selection_campaign_contract_freeze_12b2d.json"

GRAPH = ROOT / "results/hmac_fault_campaign_11d1/graph_dataset_11d1a/hmac_golden_netlist_graph_11d1a.npz"
GRAPH_SCHEMA = ROOT / "results/hmac_fault_campaign_11d1/graph_dataset_11d1a/hmac_golden_netlist_graph_schema_11d1a.json"
GRAPH_AUDIT = ROOT / "results/hmac_fault_campaign_11d1/hmac_golden_netlist_graph_topology_integrity_freeze_11d1a.json"

REVIEW = WORK / "circuitsage_hmac_v2_1_observability_ceiling_review_12b2e.md"
MEASUREMENT_REGISTRY = WORK / "circuitsage_hmac_v2_1_alternative_measurement_registry_12b2e.csv"
ARCHITECTURE = CONFIG / "circuitsage_hmac_v2_1_alternative_measurement_architecture_12b2e.json"
PROBE_CONTRACT = CONFIG / "circuitsage_hmac_v2_1_probe_discovery_contract_12b2e.json"
ACCEPTANCE = CONFIG / "circuitsage_hmac_v2_1_instrumentation_feasibility_acceptance_12b2e.json"
ENVIRONMENT = WORK / "circuitsage_hmac_v2_1_instrumentation_environment_12b2e.json"
MANIFEST = RESULT / "circuitsage_hmac_v2_1_observability_ceiling_manifest_12b2e.json"
AUDIT = RESULT / "circuitsage_hmac_v2_1_observability_ceiling_alternative_measurement_freeze_12b2e.json"

PINNED = {
    SOURCE_2D: "ee125219771c1495608b0fef1bcf188fb63c3c492b141c0a927a0d7c7742ebce",
    SELECTED_VECTORS: "be9df0a3a70e61b54fc793439328bbcc307b9373fd881b0e97d500919ec27bff",
    SELECTION_TRACE: "6191a9e133f93eb66f52d70aa97991b7655782485341d4df23e66fcd4398555f",
    SELECTION_METRICS: "69dad070e994e8808e19373e34f47abeb3b7633b16f59bf6142f40ca855ae139",
    SELECTION_LOCK: "963fa242559e6681b8c3f58d3de103f1b638bc4476425c7a80776c6b435597b1",
    CAMPAIGN_CONTRACT_2D: "b81ef49913f7235a120545240529996c159c55d45c9412ef217599e41e8f2598",
    MANIFEST_2D: "6f5d171a6c72eca8cb578b4e5a17f4ab269ea632ba33a34a75d5a8b62eae84a2",
    AUDIT_2D: "83c39f566ca45d5dbc58a7b5a54adfae6b83a0fb2ee2115cd2ae562bbb2348f1",
    GRAPH: "e3c2dd2214b544231186c29d8d9cb5aa6621b4d4b4bc9002150ac4f6c208c052",
    GRAPH_SCHEMA: "d24c3dd285891e284d20d1adcd93f793bedfd3042f20159a8a486359bc25e393",
    GRAPH_AUDIT: "4b9aef6468b467350667338373c28dc5797e3f59ce989af71a18369f086dfeb9",
}

PILOT_FAULTS = 2048
OBSERVABLE_FAULTS = 1027
INVISIBLE_FAULTS = 1021
DETECTION_CEILING = OBSERVABLE_FAULTS / PILOT_FAULTS
ORIGINAL_TARGET = 0.70
SELECTED_EXACT_SITE = 0.41748047
SELECTED_MEAN_CANDIDATES = 6.0925
SELECTED_MAX_CANDIDATES = 69


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


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode()


def load_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing JSON: {rel(path)}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON object required: {rel(path)}")
    return value


def frozen_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.exists(), f"refusing to overwrite frozen output: {rel(path)}")
    temporary = path.with_name(path.name + ".tmp")
    require(not temporary.exists(), f"stale temporary output: {rel(temporary)}")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def record(path: Path) -> dict[str, Any]:
    return {"path": rel(path), "sha256": sha256(path), "bytes": path.stat().st_size}


def verify_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    print("STAGE 12B-2E — OBSERVABILITY-CEILING REVIEW AND ALTERNATIVE-MEASUREMENT CONTRACT")
    print("FROZEN INPUT VERIFICATION")
    for path, expected in PINNED.items():
        require(path.is_file(), f"missing frozen input: {rel(path)}")
        require(sha256(path) == expected, f"frozen input SHA: {rel(path)}")
        print(f"  {path.name:<91}: OK", flush=True)
    metrics = load_json(SELECTION_METRICS)
    campaign = load_json(CAMPAIGN_CONTRACT_2D)
    audit = load_json(AUDIT_2D)
    graph_audit = load_json(GRAPH_AUDIT)
    require(metrics.get("status") == "FROZEN", "selection metrics status")
    require(metrics.get("complete_pool_observable_faults") == OBSERVABLE_FAULTS, "observable count")
    require(abs(float(metrics.get("complete_pool_detection_ceiling")) - DETECTION_CEILING) < 1e-12, "detection ceiling")
    require(metrics.get("detection_target_feasible_with_complete_pool") is False, "ceiling feasibility")
    require(metrics.get("selected_vector_acceptance") == "NOT_MET", "selection disposition")
    require(campaign.get("status") == "FROZEN", "campaign contract status")
    require(campaign.get("execution_authorization") == "BLOCKED — PILOT ACCEPTANCE NOT MET", "campaign block")
    require(campaign.get("target_relaxation") == "PROHIBITED", "target protection")
    require(audit.get("status") == "PASS", "12B-2D audit status")
    require(audit.get("full_repair_campaign") == "BLOCKED — PILOT ACCEPTANCE NOT MET", "12B-2D full-campaign state")
    require(audit.get("model_training") == "NOT AUTHORIZED", "training boundary")
    require(audit.get("repair_site_test") == "LOCKED / NOT AUTHORIZED", "repair test boundary")
    require(audit.get("validation_access") == 0 and audit.get("holdout_access") == 0, "protected partition access")
    require(graph_audit.get("status") == "PASS", "graph integrity")
    print("  Vector ceiling, full-campaign block, graph integrity and protected partitions             : PASS")
    return metrics, campaign, audit


def measurement_registry_bytes() -> bytes:
    rows = [
        {
            "measurement_id": "M0_EXTERNAL_IO_FINAL",
            "tier": 0, "source": "EXTERNAL INTERFACE",
            "signals": "digest_o|done_o|busy_o|completion_cycles|timeout",
            "representation": "EXACT FINAL OUTPUT AND TIMING",
            "silicon_access": "POTENTIALLY EXTERNAL",
            "identity_leakage_risk": "NONE", "pilot_priority": 0,
            "disposition": "FROZEN REFERENCE; CEILING ESTABLISHED",
        },
        {
            "measurement_id": "M1_CONTROL_TIMELINE",
            "tier": 1, "source": "EXTERNAL CONTROL TIMELINE",
            "signals": "busy_o|done_o sampled each cycle",
            "representation": "EVENT TRANSITIONS AND RUN-LENGTHS",
            "silicon_access": "EXTERNAL IF PINS EXPOSED",
            "identity_leakage_risk": "NONE", "pilot_priority": 1,
            "disposition": "AUTHORIZED FOR FEASIBILITY",
        },
        {
            "measurement_id": "M2_ARCH_STATE_SKETCH",
            "tier": 2, "source": "SIMULATION-ONLY INTERNAL ARCHITECTURAL STATE",
            "signals": "round counter|compression state|message schedule boundaries",
            "representation": "DOMAIN-SEPARATED HASHED SNAPSHOTS AT FIXED EVENTS",
            "silicon_access": "REQUIRES DFT OR EMBEDDED MONITOR",
            "identity_leakage_risk": "MEDIUM; STATIC GLOBAL PROBE SET REQUIRED", "pilot_priority": 2,
            "disposition": "AUTHORIZED FOR PROBE DISCOVERY ONLY",
        },
        {
            "measurement_id": "M3_GRAPH_CONE_TOGGLE_SKETCH",
            "tier": 3, "source": "SIMULATION-ONLY GRAPH-SELECTED INTERNAL NETS",
            "signals": "fixed global probe bank selected without query fault identity",
            "representation": "PER-WINDOW TOGGLE COUNTS AND HASH SKETCH",
            "silicon_access": "REQUIRES DFT OR EMBEDDED MONITOR",
            "identity_leakage_risk": "HIGH UNLESS SELECTION IS GLOBAL AND FROZEN", "pilot_priority": 3,
            "disposition": "CONTRACTED; NOT YET AUTHORIZED FOR CAPTURE",
        },
        {
            "measurement_id": "X_FAULT_SELECTOR_OR_RAW",
            "tier": 99, "source": "INJECTION CONTROL",
            "signals": "fault_selector_i|fault_enable_i|fault_value_i|fault_raw_o",
            "representation": "PROHIBITED",
            "silicon_access": "EXPERIMENT-ONLY",
            "identity_leakage_risk": "CRITICAL", "pilot_priority": 99,
            "disposition": "FORBIDDEN AS DIAGNOSTIC FEATURE OR PROBE-SELECTION INPUT",
        },
    ]
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader(); writer.writerows(rows)
    return output.getvalue().encode()


def report_text() -> str:
    return f"""# CircuitSage-HMAC V2.1 observability-ceiling review

## Disposition

Stage 12B-2D screened all 512 deterministic key/message vectors on 2,048
REPAIR_TRAIN pilot fault instances. Exactly {OBSERVABLE_FAULTS} faults produced
an externally visible digest, timeout, or completion-latency deviation. The
remaining {INVISIBLE_FAULTS} faults were indistinguishable from golden behavior
under every candidate vector.

The complete-pool detection ceiling is therefore {DETECTION_CEILING:.8f}, below
the frozen {ORIGINAL_TARGET:.8f} target. More ranking, retraining, or choosing a
different subset of the same vectors cannot make an externally invisible fault
detectable. The full repair campaign remains blocked and the target is not
relaxed.

## What did improve

The selected 96-vector subset preserved all observable pilot faults, increased
the all-injected exact-site rate to {SELECTED_EXACT_SITE:.8f}, and reduced the
mean/maximum observable candidate-site counts to {SELECTED_MEAN_CANDIDATES:.4f}
and {SELECTED_MAX_CANDIDATES}. These are closed-pilot results, not independent
generalization evidence.

## Next experiment

The authorized next step is a bounded feasibility study using fixed,
fault-identity-independent measurements. External control timing is evaluated
first. Simulation-only architectural-state sketches may then be tested on the
same REPAIR_TRAIN pilot sites. Any internal measurement would require DFT,
trace hardware, or embedded monitors before it could be used on silicon.

Injection controls—including the selector, forced value, enable signal, and
selected raw-net monitor—are forbidden as model inputs. They exist solely to
create and score simulated faults. No claim of production or independent-chip
readiness is made.
"""


def self_test() -> None:
    require(OBSERVABLE_FAULTS + INVISIBLE_FAULTS == PILOT_FAULTS, "fault-count canary")
    require(abs(DETECTION_CEILING - 0.50146484375) < 1e-15, "ceiling canary")
    require(measurement_registry_bytes() == measurement_registry_bytes(), "registry replay")
    print("Stage 12B-2E self-test: PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    require(ROOT.name == "vlsi_fault_detection_v2", f"run from project root, not {ROOT}")
    outputs = (REVIEW, MEASUREMENT_REGISTRY, ARCHITECTURE, PROBE_CONTRACT,
               ACCEPTANCE, ENVIRONMENT, MANIFEST, AUDIT)
    for path in outputs:
        require(not path.exists(), f"Stage {STAGE} output already exists: {rel(path)}")

    metrics, campaign_2d, audit_2d = verify_inputs()
    registry_payload = measurement_registry_bytes()
    review_payload = report_text().encode()
    architecture = {
        "architecture_version": "CIRCUITSAGE-HMAC-V2.1-ALTERNATIVE-MEASUREMENT-v1",
        "stage": STAGE, "status": "FROZEN",
        "purpose": "test whether identity-independent temporal/internal observations expose externally invisible HMAC SA0/SA1 faults",
        "measurement_tiers": [
            {"tier": 0, "id": "EXTERNAL_IO_FINAL", "status": "FROZEN CEILING REFERENCE"},
            {"tier": 1, "id": "CONTROL_TIMELINE", "status": "AUTHORIZED FOR FEASIBILITY"},
            {"tier": 2, "id": "ARCH_STATE_SKETCH", "status": "PROBE DISCOVERY AUTHORIZED; CAPTURE NOT YET AUTHORIZED"},
            {"tier": 3, "id": "GRAPH_CONE_TOGGLE_SKETCH", "status": "CONTRACTED; NOT AUTHORIZED"},
        ],
        "probe_selection": "ONE GLOBAL FROZEN SET; NEVER CONDITIONED ON QUERY FAULT IDENTITY",
        "maximum_internal_probe_bits": 64,
        "maximum_event_snapshots_per_transaction": 16,
        "maximum_toggle_windows_per_transaction": 16,
        "sketch_hash": "SHA256 WITH FROZEN DOMAIN SEPARATION",
        "query_inputs": ["key", "message", "observed external behavior", "authorized fixed-probe measurements"],
        "forbidden_query_inputs": ["fault site ID", "fault instance ID", "batch selector", "stuck value", "fault_raw_o", "fault_enable_i"],
        "deployment_boundary": {
            "external_only": "CURRENT 0.50146484 PILOT DETECTION CEILING",
            "simulation_instrumented": "RESEARCH FEASIBILITY ONLY",
            "silicon": "REQUIRES SEPARATELY DESIGNED AND VALIDATED DFT/MONITOR HARDWARE",
        },
    }
    probe_contract = {
        "contract_version": "CIRCUITSAGE-HMAC-V2.1-PROBE-DISCOVERY-12B2E-v1",
        "stage": STAGE, "status": "FROZEN",
        "authorization": "STATIC PROBE DISCOVERY AND TESTBENCH-LINT ONLY",
        "source_netlists": "FROZEN CANONICAL GOLDEN/FAULT-BATCH NETLISTS",
        "source_graph": record(GRAPH),
        "allowed_partition": "REPAIR_TRAIN PILOT SITES ONLY",
        "allowed_operations": [
            "enumerate architecture-stable internal signals",
            "map signals to frozen graph nodes",
            "choose one global probe bank without fault labels from evaluation partitions",
            "generate and lint non-invasive simulation observers",
        ],
        "prohibited_operations": [
            "modify RTL or synthesized netlists", "capture probe responses", "train or fit any model",
            "use injection selector/raw monitor as a feature", "condition probe bank on the injected site",
            "open REPAIR_CALIBRATION", "open REPAIR_SITE_TEST", "reopen original DEV_SITE_TEST",
            "open VALIDATION", "open HOLDOUT",
        ],
        "probe_discovery_limit_bits": 64,
        "probe_discovery_output": "NAMES/HASHES/MAPPINGS ONLY; NO RESPONSE VALUES",
        "next_capture_requires_separate_authorization": True,
    }
    acceptance = {
        "acceptance_version": "CIRCUITSAGE-HMAC-V2.1-INSTRUMENTATION-FEASIBILITY-12B2E-v1",
        "stage": STAGE, "status": "FROZEN",
        "pilot_partition": "REPAIR_TRAIN PILOT ONLY",
        "feasibility_targets": {
            "total_detection_recall_min": 0.60,
            "absolute_detection_gain_over_external_min": 0.10,
            "newly_detected_external_invisible_faults_min": 205,
            "all_injected_exact_site_rate_min": 0.35,
            "mean_observable_candidate_sites_max": 50.0,
            "maximum_observable_candidate_sites_max": 500,
            "fault_free_false_alarm_rate_max": 0.01,
            "identity_leakage_violations": 0,
        },
        "final_repair_target_preserved": {"all_injected_detection_recall_min": ORIGINAL_TARGET},
        "advancement_rule": "ALL FEASIBILITY TARGETS PASS AND IDENTITY-LEAKAGE AUDIT PASS",
        "target_relaxation": "PROHIBITED AFTER CAPTURE",
        "claim_limit": "simulation-instrumented HMAC SA0/SA1 feasibility; independent generalization and silicon readiness not established",
    }
    environment = {
        "environment_version": "CIRCUITSAGE-HMAC-V2.1-INSTRUMENTATION-ENVIRONMENT-v1",
        "stage": STAGE, "status": "FROZEN", "python": sys.version.split()[0],
        "platform": platform.platform(), "cpu_count": os.cpu_count(),
        "internet_required": False, "gpu_required": False,
        "estimated_probe_discovery_time": "5-30 MINUTES",
        "estimated_future_capture_time": "2-12 HOURS IF SEPARATELY AUTHORIZED",
    }

    frozen_write(REVIEW, review_payload)
    frozen_write(MEASUREMENT_REGISTRY, registry_payload)
    frozen_write(ARCHITECTURE, canonical_json(architecture))
    frozen_write(PROBE_CONTRACT, canonical_json(probe_contract))
    frozen_write(ACCEPTANCE, canonical_json(acceptance))
    frozen_write(ENVIRONMENT, canonical_json(environment))
    primary = (REVIEW, MEASUREMENT_REGISTRY, ARCHITECTURE, PROBE_CONTRACT, ACCEPTANCE, ENVIRONMENT)
    manifest = {
        "manifest_version": "CIRCUITSAGE-HMAC-V2.1-OBSERVABILITY-CEILING-MANIFEST-v1",
        "stage": STAGE, "status": "PASS", "stage_12b2d_audit": record(AUDIT_2D),
        "outputs": {rel(path): record(path) for path in primary},
        "pilot_faults": PILOT_FAULTS, "externally_observable_faults": OBSERVABLE_FAULTS,
        "externally_invisible_faults": INVISIBLE_FAULTS,
        "external_detection_ceiling": DETECTION_CEILING,
        "simulation_calls": 0, "probe_response_values_captured": 0,
        "model_objects_deserialized": 0, "training_calls": 0, "inference_calls": 0,
        "repair_calibration_access": 0, "repair_site_test_access": 0,
        "original_dev_site_test_access": 0, "validation_access": 0, "holdout_access": 0,
    }
    frozen_write(MANIFEST, canonical_json(manifest))
    audit = {
        "audit_version": "CIRCUITSAGE-HMAC-V2.1-OBSERVABILITY-CEILING-ALTERNATIVE-MEASUREMENT-FREEZE-v1",
        "stage": STAGE, "status": "PASS",
        "review_status": "FROZEN / COMPLETE",
        "external_io_detection_ceiling": DETECTION_CEILING,
        "frozen_detection_target": ORIGINAL_TARGET,
        "external_target_feasible": False,
        "selected_localization_gain": "PRESERVED",
        "full_repair_campaign": "BLOCKED / NOT AUTHORIZED",
        "alternative_measurement_architecture": "FROZEN",
        "probe_discovery": "AUTHORIZED / NOT STARTED",
        "probe_capture": "NOT AUTHORIZED",
        "model_training": "NOT AUTHORIZED",
        "identity_leakage_controls": "FROZEN / REQUIRED",
        "fault_selector_value_raw_as_features": "PROHIBITED",
        "repair_site_test": "LOCKED / NOT AUTHORIZED",
        "original_dev_site_test": "CONSUMED / NOT REOPENED",
        "validation_access": 0, "holdout_access": 0,
        "v1_modified": False, "v2_core_modified": False, "v2_1_modified": False,
        "rtl_modified": False, "golden_netlist_modified": False,
        "manifest": record(MANIFEST),
        "next_gate": "STAGE 12B-2F — SIMULATION-ONLY PROBE DISCOVERY AND INSTRUMENTATION FEASIBILITY FREEZE",
    }
    frozen_write(AUDIT, canonical_json(audit))

    require(review_payload == REVIEW.read_bytes(), "review replay")
    require(registry_payload == MEASUREMENT_REGISTRY.read_bytes(), "registry replay")
    for path in (ARCHITECTURE, PROBE_CONTRACT, ACCEPTANCE, ENVIRONMENT, MANIFEST, AUDIT):
        require(canonical_json(load_json(path)) == path.read_bytes(), f"canonical JSON replay: {path.name}")

    print("\nSTAGE 12B-2E — OBSERVABILITY-CEILING REVIEW AND ALTERNATIVE-MEASUREMENT CONTRACT FREEZE")
    print(f"{'Status':<55}: PASS")
    print(f"{'Review / alternative architecture':<55}: FROZEN / FROZEN")
    print(f"{'External observable / invisible faults':<55}: {OBSERVABLE_FAULTS} / {INVISIBLE_FAULTS}")
    print(f"{'External-I/O detection ceiling':<55}: {DETECTION_CEILING:.8f}")
    print(f"{'Frozen final detection target':<55}: {ORIGINAL_TARGET:.8f} / PRESERVED")
    print(f"{'External-only target feasibility':<55}: NO")
    print(f"{'Selected exact-site / mean / max candidates':<55}: {SELECTED_EXACT_SITE:.8f} / {SELECTED_MEAN_CANDIDATES:.4f} / {SELECTED_MAX_CANDIDATES}")
    print(f"{'Full repair campaign':<55}: BLOCKED / NOT AUTHORIZED")
    print(f"{'Probe discovery':<55}: AUTHORIZED / NOT STARTED")
    print(f"{'Probe capture / model training':<55}: NOT AUTHORIZED / NOT AUTHORIZED")
    print(f"{'Fault selector/value/raw diagnostic features':<55}: PROHIBITED")
    print(f"{'REPAIR_SITE_TEST / original DEV_SITE_TEST':<55}: LOCKED / CONSUMED")
    print(f"{'VALIDATION / HOLDOUT access':<55}: 0 / 0")
    print(f"{'Frozen RTL / golden netlist modified':<55}: NO / NO")
    print(f"{'Review':<55}: {REVIEW}")
    print(f"{'Review SHA':<55}: {sha256(REVIEW)}")
    print(f"{'Measurement registry':<55}: {MEASUREMENT_REGISTRY}")
    print(f"{'Measurement registry SHA':<55}: {sha256(MEASUREMENT_REGISTRY)}")
    print(f"{'Architecture':<55}: {ARCHITECTURE}")
    print(f"{'Architecture SHA':<55}: {sha256(ARCHITECTURE)}")
    print(f"{'Probe contract':<55}: {PROBE_CONTRACT}")
    print(f"{'Probe contract SHA':<55}: {sha256(PROBE_CONTRACT)}")
    print(f"{'Acceptance':<55}: {ACCEPTANCE}")
    print(f"{'Acceptance SHA':<55}: {sha256(ACCEPTANCE)}")
    print(f"{'Manifest':<55}: {MANIFEST}")
    print(f"{'Manifest SHA':<55}: {sha256(MANIFEST)}")
    print(f"{'Audit':<55}: {AUDIT}")
    print(f"{'Audit SHA':<55}: {sha256(AUDIT)}")
    print(f"{'Next gate':<55}: STAGE 12B-2F — SIMULATION-ONLY PROBE DISCOVERY AND INSTRUMENTATION FEASIBILITY FREEZE")


if __name__ == "__main__":
    main()
