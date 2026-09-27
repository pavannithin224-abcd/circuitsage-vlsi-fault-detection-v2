# Stage 12C-2F — Acceptance-Gate Measurement

**Status: PASS / FROZEN — measurement only.**

## SCOPE WARNING

The frozen acceptance contract evaluates its gates on
**INDEPENDENT_CIRCUIT_TEST** circuits (`ibex_cpu`, `secworks_chacha`). Those are
**sealed and not captured**. Everything below is measured on
GENERALIZATION_TRAIN and GENERALIZATION_CALIBRATION as a **distance check**.

**`acceptance_decision: NOT EVALUATED`** on every row. A development number is
not a pass and must never be used to justify opening the sealed test.

## Non-learning comparator — macro gates

| family | gate | threshold | measured | would meet |
|---|---|---|---|---|
| `opentitan_hmac_sha256` | all_injected_detection_recall_min | 0.6 | 0.2676 | NO |
| `opentitan_hmac_sha256` | all_injected_exact_site_rate_min | 0.25 | 0.0286 | NO |
| `opentitan_hmac_sha256` | ambiguous_false_unique_rate_max | 0.01 | 0.0000 | YES |
| `opentitan_hmac_sha256` | fault_free_false_alarm_rate_max | 0.01 | 0.0000 | YES |
| `opentitan_hmac_sha256` | observable_candidate_set_coverage_min | 0.9 | 1.0000 | YES |
| `opentitan_hmac_sha256` | observable_detection_recall_min | 0.9 | 1.0000 | YES |
| `opentitan_hmac_sha256` | observable_mrr_min | 0.3 | 0.2544 | NO |
| `opentitan_hmac_sha256` | unique_signature_top1_site_min | 0.7 | 0.1070 | NO |
| `picorv32_cpu` | all_injected_detection_recall_min | 0.6 | 0.2459 | NO |
| `picorv32_cpu` | all_injected_exact_site_rate_min | 0.25 | 0.0045 | NO |
| `picorv32_cpu` | ambiguous_false_unique_rate_max | 0.01 | 0.0000 | YES |
| `picorv32_cpu` | fault_free_false_alarm_rate_max | 0.01 | 0.0000 | YES |
| `picorv32_cpu` | observable_candidate_set_coverage_min | 0.9 | 1.0000 | YES |
| `picorv32_cpu` | observable_detection_recall_min | 0.9 | 1.0000 | YES |
| `picorv32_cpu` | observable_mrr_min | 0.3 | 0.0989 | NO |
| `picorv32_cpu` | unique_signature_top1_site_min | 0.7 | 0.0183 | NO |
| `secworks_aes` | all_injected_detection_recall_min | 0.6 | 0.9714 | YES |
| `secworks_aes` | all_injected_exact_site_rate_min | 0.25 | 0.3675 | YES |
| `secworks_aes` | ambiguous_false_unique_rate_max | 0.01 | 0.0000 | YES |
| `secworks_aes` | fault_free_false_alarm_rate_max | 0.01 | 0.0000 | YES |
| `secworks_aes` | observable_candidate_set_coverage_min | 0.9 | 1.0000 | YES |
| `secworks_aes` | observable_detection_recall_min | 0.9 | 1.0000 | YES |
| `secworks_aes` | observable_mrr_min | 0.3 | 0.5463 | YES |
| `secworks_aes` | unique_signature_top1_site_min | 0.7 | 0.3783 | NO |
| `secworks_sha256` | all_injected_detection_recall_min | 0.6 | 0.9467 | YES |
| `secworks_sha256` | all_injected_exact_site_rate_min | 0.25 | 0.4516 | YES |
| `secworks_sha256` | ambiguous_false_unique_rate_max | 0.01 | 0.0000 | YES |
| `secworks_sha256` | fault_free_false_alarm_rate_max | 0.01 | 0.0000 | YES |
| `secworks_sha256` | observable_candidate_set_coverage_min | 0.9 | 1.0000 | YES |
| `secworks_sha256` | observable_detection_recall_min | 0.9 | 1.0000 | YES |
| `secworks_sha256` | observable_mrr_min | 0.3 | 0.6745 | YES |
| `secworks_sha256` | unique_signature_top1_site_min | 0.7 | 0.4771 | NO |

## Non-learning comparator — per-circuit floor

| family | gate | threshold | measured | would meet |
|---|---|---|---|---|
| `opentitan_hmac_sha256` | all_injected_exact_site_rate_min | 0.15 | 0.0286 | NO |
| `opentitan_hmac_sha256` | fault_free_false_alarm_rate_max | 0.02 | 0.0000 | YES |
| `opentitan_hmac_sha256` | observable_candidate_set_coverage_min | 0.8 | 1.0000 | YES |
| `opentitan_hmac_sha256` | observable_detection_recall_min | 0.8 | 1.0000 | YES |
| `picorv32_cpu` | all_injected_exact_site_rate_min | 0.15 | 0.0045 | NO |
| `picorv32_cpu` | fault_free_false_alarm_rate_max | 0.02 | 0.0000 | YES |
| `picorv32_cpu` | observable_candidate_set_coverage_min | 0.8 | 1.0000 | YES |
| `picorv32_cpu` | observable_detection_recall_min | 0.8 | 1.0000 | YES |
| `secworks_aes` | all_injected_exact_site_rate_min | 0.15 | 0.3675 | YES |
| `secworks_aes` | fault_free_false_alarm_rate_max | 0.02 | 0.0000 | YES |
| `secworks_aes` | observable_candidate_set_coverage_min | 0.8 | 1.0000 | YES |
| `secworks_aes` | observable_detection_recall_min | 0.8 | 1.0000 | YES |
| `secworks_sha256` | all_injected_exact_site_rate_min | 0.15 | 0.4516 | YES |
| `secworks_sha256` | fault_free_false_alarm_rate_max | 0.02 | 0.0000 | YES |
| `secworks_sha256` | observable_candidate_set_coverage_min | 0.8 | 1.0000 | YES |
| `secworks_sha256` | observable_detection_recall_min | 0.8 | 1.0000 | YES |

## Trained candidates (calibration family, full-graph ranking)

| candidate | MRR | Top-1 | Top-10 | median rank | detector AUROC |
|---|---|---|---|---|---|
| `V22_GRAPHSAGE_METRIC_SMALL` | 0.002718 | 0.000732 | 0.003662 | 4184 / 9127 | 0.01273119 |
| `V22_GATV2_CROSS_FUSION` | 0.003679 | 0.000488 | 0.004883 | 3134 / 9127 | 0.01228105 |
| `V22_GRAPHSAGE_OOD_ENSEMBLE` | 0.003413 | 0.001709 | 0.003906 | 4917 / 9127 | 0.0128421 |

These rank the true site against **every node in the circuit**, not against 128
sampled negatives, so they are directly comparable to the contract's MRR and
Top-1 gates — unlike the training monitor used in Stages 12C-2C and 12C-2E.

## What this stage does not do

No training, no gradient updates, no selection, no protected-partition access.
Independent generalization remains **NOT ESTABLISHED**. Future hybrid brand
remains **Faultiva 1.0 — Hybrid VLSI Fault Intelligence**.

## Next gate

Determined by the distance table: if the per-circuit floor shortfall on
CPU-class circuits is structural rather than incidental, graph-constrained
reranking is the next authorized experiment.
