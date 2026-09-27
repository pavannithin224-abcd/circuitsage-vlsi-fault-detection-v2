# Stage 12C-2C — Candidate Training

**Status: PASS / FROZEN — training only, no selection.**

## Candidates trained

| candidate | parameters | cap | epochs | best calib Top-1 | wall clock |
|---|---|---|---|---|---|
| `V22_GRAPHSAGE_METRIC_SMALL` | 80,708 | 750,000 | 10 | 0.0156 | 1767s |
| `V22_GATV2_CROSS_FUSION` | 182,052 | 1,500,000 | 9 | 0.0150 | 1550s |
| `V22_GRAPHSAGE_OOD_ENSEMBLE` | 133,702 | 1,500,000 | 9 | 0.0186 | 1049s |

All parameter caps from the frozen Stage 12C-1A candidate grid were enforced as
hard preconditions before any optimizer step.

## Contract compliance

- gradients on **GENERALIZATION_TRAIN only**: opentitan_hmac_sha256, picorv32_cpu, secworks_aes
- **GENERALIZATION_CALIBRATION** (secworks_sha256) used for
  monitoring and early stopping only; `backward()` never called on it
- INDEPENDENT_CIRCUIT_TEST and GENERALIZATION_HOLDOUT never opened
- response encoder is the contracted **masked temporal** encoder over per-vector
  slices, not a flat projection
- fixed seed 20260920, deterministic data order, environment recorded

## Non-learning comparator

`V22_EXACT_SIGNATURE_GRAPH_BASELINE` was measured on the frozen campaign
signatures (0 parameters, no gradients) and is reported alongside the trained
candidates as the mandatory comparator.

## What this stage does NOT do

No model selection, no acceptance evaluation, no protected-partition access.
Selection is a separate authorized gate. Independent generalization remains
**NOT ESTABLISHED**. Future hybrid brand remains **Faultiva 1.0 — Hybrid VLSI Fault Intelligence**.

## Next gate

**Stage 12C-2D** — model selection on GENERALIZATION_CALIBRATION under the
frozen selection order.
