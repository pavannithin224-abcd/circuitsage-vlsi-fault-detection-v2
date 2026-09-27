# Stage 12C-2B — Calibration-Use and Training Authorization

**Status: PASS / FROZEN — authorization gate only.**

## What this stage authorizes

| | |
|---|---|
| Model training | **AUTHORIZED / BOUNDED** on GENERALIZATION_TRAIN |
| Model selection | **AUTHORIZED / BOUNDED** on GENERALIZATION_CALIBRATION |
| Independent test | **NOT AUTHORIZED** |
| Holdout | **NOT AUTHORIZED** |
| Acceptance evaluation | **NOT AUTHORIZED** |

The Stage 12C-1A training contract required a *separate capture authorization*
before calibration could be used for selection. That authorization is issued
here.

## Leakage audit — 11 executable checks

| check | result |
|---|---|
| family_assigned_exactly_once | PASS |
| train_calibration_disjoint | PASS |
| protected_disjoint_from_development | PASS |
| gradient_families_are_train_only | PASS |
| calibration_not_in_gradient_scope | PASS |
| no_protected_family_captured | PASS |
| no_protected_family_in_graph_dataset | PASS |
| site_spans_disjoint | PASS |
| graph_response_site_alignment | PASS |
| identity_firewall_features | PASS |
| identity_firewall_graph | PASS |

**Verdict:** NO LEAKAGE DETECTED ACROSS PARTITION, SITE, MODALITY OR IDENTITY DIMENSIONS

## Partition register — seven families

| family | partition | captured | gradients | selection | access |
|---|---|---|---|---|---|
| `ibex_cpu` | INDEPENDENT_CIRCUIT_TEST | NO | PROHIBITED | PROHIBITED | SEALED |
| `opentitan_hmac_sha256` | GENERALIZATION_TRAIN | YES | AUTHORIZED | PROHIBITED | OPEN |
| `picorv32_cpu` | GENERALIZATION_TRAIN | YES | AUTHORIZED | PROHIBITED | OPEN |
| `secworks_aes` | GENERALIZATION_TRAIN | YES | AUTHORIZED | PROHIBITED | OPEN |
| `secworks_chacha` | INDEPENDENT_CIRCUIT_TEST | NO | PROHIBITED | PROHIBITED | SEALED |
| `secworks_sha256` | GENERALIZATION_CALIBRATION | YES | PROHIBITED | AUTHORIZED | OPEN |
| `serv_cpu` | GENERALIZATION_HOLDOUT | NO | PROHIBITED | PROHIBITED | BLOCKED |

## Candidate register — frozen at Stage 12C-1A

| candidate | trainable | graph encoder | param cap | role |
|---|---|---|---|---|
| `V22_EXACT_SIGNATURE_GRAPH_BASELINE` | NO | NONE | 0 | mandatory non-learning comparator |
| `V22_GRAPHSAGE_METRIC_SMALL` | YES | GraphSAGE 3-layer | 750,000 | primary compact candidate |
| `V22_GATV2_CROSS_FUSION` | YES | GATv2 3-layer | 1,500,000 | higher-capacity candidate |
| `V22_GRAPHSAGE_OOD_ENSEMBLE` | YES | GraphSAGE shared encoder | 1,500,000 | uncertainty-focused candidate |

No architecture is designed here. All four candidates, their encoders, fusion
strategies and parameter caps were frozen in the Stage 12C-1A candidate grid.

**Selection order (frozen):** safety → OOD/false alarms → circuit-macro detection → candidate coverage → MRR/exact site → simplicity

## Reminder: one-shot rule

> ONE LOCKED TEST EVALUATION; NO RETRAINING, THRESHOLD CHANGE, OR RESELECTION AFTERWARD

Training and selection may iterate freely on TRAIN and CALIBRATION. The
independent test is opened **once**, and after it no retraining, threshold
change or reselection is permitted.

## Access

INDEPENDENT_TEST **LOCKED**, VALIDATION **UNOPENED**, HOLDOUT **SEALED**.
Independent generalization remains **NOT ESTABLISHED**. Future hybrid brand
remains **Faultiva 1.0 — Hybrid VLSI Fault Intelligence**.

## Next gate

**Stage 12C-2C** — candidate training execution against this authorization.
