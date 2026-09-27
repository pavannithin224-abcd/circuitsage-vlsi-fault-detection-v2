# Stage 12C-1Q — Disposition Correction

**Status: PASS / FROZEN — correction record, no re-evaluation.**

## What is being corrected

Stage 12C-1P recorded that no frozen V2.2 advancement target exists:

```
disposition_mode           : REPORT-ONLY / NO FROZEN ADVANCEMENT TARGET EXISTS
advancement_decision       : NOT EVALUATED / NO FROZEN CRITERIA EXIST
advancement_target_defined : false
```

**This is incorrect.** Stage 12C-1A froze
`config/v2_2/circuitsage_hmac_v2_2_acceptance_contract_12c1a.json`
(SHA `9c8eec4d85957c4408ac59e0c8760af90c91d0667c995d91ac969b5a8f205f26`)
containing **10 macro gates** and a
**4-metric per-circuit floor**. It was
frozen before Stage 12C-1P executed.

## What remains correct

Stage 12C-1P declined to declare PASS or NOT_MET. **That action was correct.**
The acceptance gates are scoped to `INDEPENDENT_CIRCUIT_TEST` circuits, which
have not been captured. Applying them to TRAIN/CALIBRATION data would be
evaluating on development data.

All numeric values reported by Stage 12C-1P were re-verified against the frozen
Stage 12C-1O metrics in this stage: **0 disagreements**.

Stage 12C-1P is **not modified**. Its measured evidence stands.

## The frozen acceptance contract

**Claim:** INDEPENDENT-CIRCUIT GENERALIZATION IS ESTABLISHED ONLY IF EVERY MANDATORY GATE PASSES ON BOTH LOCKED TEST CIRCUITS

**Macro gates** (evaluated on both locked test circuits):

| gate | threshold |
|---|---|
| all_injected_detection_recall_min | 0.6 |
| all_injected_exact_site_rate_min | 0.25 |
| ambiguous_false_unique_rate_max | 0.01 |
| fault_free_false_alarm_rate_max | 0.01 |
| observable_candidate_set_coverage_min | 0.9 |
| observable_detection_recall_min | 0.9 |
| observable_mrr_min | 0.3 |
| ood_auroc_min | 0.75 |
| polarity_accuracy_given_correct_unique_site_min | 0.8 |
| unique_signature_top1_site_min | 0.7 |

**Per-circuit floor** (each test circuit individually):

| gate | threshold |
|---|---|
| all_injected_exact_site_rate_min | 0.15 |
| fault_free_false_alarm_rate_max | 0.02 |
| observable_candidate_set_coverage_min | 0.8 |
| observable_detection_recall_min | 0.8 |

**One-shot rule:** ONE LOCKED TEST EVALUATION; NO RETRAINING, THRESHOLD CHANGE, OR RESELECTION AFTERWARD

**Failure disposition:** FREEZE NOT_ESTABLISHED; DO NOT REOPEN TEST; DESIGN A NEW TRAIN/CALIBRATION-ONLY STUDY

## Full corpus — seven families

| family | partition | campaign state | access |
|---|---|---|---|
| `ibex_cpu` | INDEPENDENT_CIRCUIT_TEST | NOT CAPTURED | SEALED UNTIL ONE COMMITTED FINAL EVALUATION |
| `opentitan_hmac_sha256` | GENERALIZATION_TRAIN | CAPTURED / FROZEN (Stage 12C-1O) | OPEN FOR TRAINING AND CALIBRATION |
| `picorv32_cpu` | GENERALIZATION_TRAIN | CAPTURED / FROZEN (Stage 12C-1O) | OPEN FOR TRAINING AND CALIBRATION |
| `secworks_aes` | GENERALIZATION_TRAIN | CAPTURED / FROZEN (Stage 12C-1O) | OPEN FOR TRAINING AND CALIBRATION |
| `secworks_chacha` | INDEPENDENT_CIRCUIT_TEST | NOT CAPTURED | SEALED UNTIL ONE COMMITTED FINAL EVALUATION |
| `secworks_sha256` | GENERALIZATION_CALIBRATION | CAPTURED / FROZEN (Stage 12C-1O) | OPEN FOR TRAINING AND CALIBRATION |
| `serv_cpu` | GENERALIZATION_HOLDOUT | NOT CAPTURED | BLOCKED / EXCLUDED FROM V2.2 ENTIRELY |

The Stage 12C-1O campaign covered **four** of seven families. Acceptance is
decided on the two sealed `INDEPENDENT_CIRCUIT_TEST` families, and the
`GENERALIZATION_HOLDOUT` family remains blocked even if V2.2 passes.

## Root cause and preventive rule

**Root cause:** the Stage 12C-1P input survey inspected the campaign and
disposition lineage but did not enumerate `config/v2_2` contract artifacts
frozen at Stage 12C-1A.

**Preventive rule (frozen here):** any stage asserting the *absence* of a
contract, criterion or artifact must first enumerate `config/v2_2` and the full
stage audit chain, and must cite that enumeration in its own manifest.

## Access

No protected partition was accessed. Independent TEST remains **LOCKED**,
VALIDATION **UNOPENED**, HOLDOUT **SEALED**. Independent generalization remains
**NOT ESTABLISHED**. The future hybrid brand remains **Faultiva 1.0 — Hybrid VLSI Fault Intelligence**.

## Next gate

**Stage 12C-2A** — V2.2 model contract and training authorization, written
against the corrected understanding: the target is the Stage 12C-1A acceptance
contract, evaluated once on `ibex_cpu` and `secworks_chacha`.
