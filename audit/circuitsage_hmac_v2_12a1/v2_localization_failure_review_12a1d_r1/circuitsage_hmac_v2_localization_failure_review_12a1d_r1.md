# CircuitSage-HMAC V2 localization failure review — Stage 12A-1D-R1

Stage 12A-1D trained correctly and replayed deterministically, but its detector and locator behaved very differently. The exact golden-reference detector achieved zero false alarms and 100% recall on observable calibration faults. The learned locator did not pass the advancement gate.

## Frozen calibration result

| Metric | Observed | Required | Result |
|---|---:|---:|---|
| Observable candidate coverage at top 50 | 0.04154561 | 0.95 | NOT MET |
| Unique-signature top-1 site accuracy | 0.00348432 | 0.80 | NOT MET |
| Observable top-5 site accuracy | 0.00406740 | 0.80 | NOT MET |
| Fault-free false-alarm rate | 0.00000000 | ≤ 0.05 | PASS |
| Observable detection recall | 1.00000000 | ≥ 0.90 | PASS |

## What is confirmed

The training objective used only 128 randomly sampled DEV_TRAIN negatives per response, about 0.2802% of the full fault catalog, while calibration ranked all 45,678 fault instances. The model learned some signal—the observed ranking metrics are above a uniform-ranking scale reference—but it did not learn enough full-catalog separation. The V1 support input describes which of 64 tests are likely to detect each candidate; it does not predict the candidate's exact output response.

These findings establish a training/evaluation mismatch and insufficient retrieval performance. They do not prove one single root cause. The frozen R2 contract therefore requires controlled ablations.

## Approved repair

Stage 12A-1D-R2 may train only on DEV_TRAIN. It must add direct response-to-candidate feature interactions, response-similar and graph-neighbor hard negatives, a shortlist retrieval step, and separate site retrieval from conditional SA0/SA1 prediction. DEV_CALIBRATION remains selection-only. DEV_SITE_TEST, VALIDATION, and HOLDOUT remain closed until every original calibration target passes.

No V1 artifact, threshold, graph, RTL, golden netlist, or frozen Stage 12A-1D output may be changed.
