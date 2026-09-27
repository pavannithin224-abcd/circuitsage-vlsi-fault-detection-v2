# Stage 12C-1P — V2.2 Campaign Disposition

**Status: PASS / FROZEN (report-only disposition)**

**Disposition mode:** REPORT-ONLY / NO FROZEN ADVANCEMENT TARGET EXISTS

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
| Fault instances | 127,484 |
| Enabled transactions | 7,849,696 |
| Dataset integrity | PASS |
| Fault-free false alarms | 0 |
| Identity firewall | PROHIBITED / ABSENT |

## Per-family result

| family | partition | detection | exact-site | observable | obs. frac | mean cand | max cand | label |
|---|---|---|---|---|---|---|---|---|
| `opentitan_hmac_sha256` | GENERALIZATION_TRAIN | 0.2676 | 0.0286 | 9,844 | 0.268 | 141.11 | 1089 | OBSERVABILITY_LIMITED |
| `picorv32_cpu` | GENERALIZATION_TRAIN | 0.2459 | 0.0045 | 4,753 | 0.246 | 51.70 | 307 | OBSERVABILITY_LIMITED |
| `secworks_aes` | GENERALIZATION_TRAIN | 0.9714 | 0.3675 | 51,601 | 0.971 | 14.39 | 98 | OBSERVABILITY_SUFFICIENT |
| `secworks_sha256` | GENERALIZATION_CALIBRATION | 0.9467 | 0.4516 | 17,277 | 0.947 | 3.80 | 44 | OBSERVABILITY_SUFFICIENT |

Aggregate detection recall: **0.65478805**
(95% CI [0.6514146873333124, 0.6581161165322708])
Aggregate exact-site rate: **0.22673434**
(95% CI [0.22419244767970883, 0.22937074456402373])

## Principal finding

Per-family detection spans **0.2459** to
**0.9714** — a spread ratio of
**3.9506** — under *identical* retrieval
logic, identical measurement and identical scoring.

On observable faults the retrieval stage is essentially perfect:

- observable candidate set coverage: **1.0**
- unique-signature Top-1 site: **1.0**
- fault-free false alarm rate: **0.0**
- ambiguous false-unique rate: **0.0**

Therefore the limiting factor is **observability of injected faults under the
frozen vector and measurement scheme, not candidate retrieval.**

## Aggregate interpretation warning

The aggregate is a fault-population weighted quantity. `secworks_aes` alone
contributes 61.8% of all
observable faults. **The aggregate must not be read as a per-circuit
expectation**, and must never be reported without the per-family breakdown.

## Training scope

Observability-sufficient: `secworks_aes`, `secworks_sha256`
Observability-limited: `opentitan_hmac_sha256`, `picorv32_cpu`

Observability-limited families **remain in the training corpus**. They must not
be removed to improve reported metrics. The
0.5 boundary is a descriptive label only and carries
no acceptance consequence.

## What this evidence does and does not establish

**Established:** a leakage-controlled multi-circuit campaign, perfect retrieval
on observable faults, zero false alarms, and isolation of observability as the
dominant limiting factor.

**Not established:** independent-circuit generalization of a trained model,
unrestricted unknown-fault detection, or any V2.2 advancement outcome.

Independent generalization remains **NOT ESTABLISHED**. The future hybrid brand
remains **Faultiva 1.0 — Hybrid VLSI Fault Intelligence**.

## Authorization

Model training and selection on the frozen 12C-1O TRAIN/CALIBRATION corpus:
**AUTHORIZED / BOUNDED**.
Independent TEST remains **LOCKED**, VALIDATION **UNOPENED**, HOLDOUT **SEALED**.
