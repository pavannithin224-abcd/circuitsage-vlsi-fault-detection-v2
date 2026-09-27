# CircuitSage-HMAC V2 Core disposition — Stage 12A-1D-R3

## Outcome

The one-week V2 Core experiment is complete and frozen as a partial success. The exact golden-reference detector passed its frozen calibration criteria: zero fault-free false alarms and 100% recall for observable deviations. Catalog-scale localization did not pass.

## Locator comparison

| Metric | Original 12A-1D | R2 repair | R2 − original | Required |
|---|---:|---:|---:|---:|
| Observable MRR | 0.00629651 | 0.00321161 | -0.00308490 | reported/selection metric |
| Unique-signature top-1 site | 0.00348432 | 0.00000000 | -0.00348432 | ≥ 0.80 |
| Observable top-5 site | 0.00406740 | 0.00290529 | -0.00116212 | ≥ 0.80 |
| Candidate coverage at top 50 | 0.04154561 | 0.02382336 | -0.01772225 | ≥ 0.95 |

R2 did not repair the locator and performed below the original locator on the primary ranking measures. The R2 artifact is retained as a reproducible negative result. The original 12A-1D candidate is recorded only as the best observed calibration reference; it is not promoted to a validated exact-site locator.

## Safe capability statement

V2 can flag an observable response mismatch against the committed golden reference within this frozen OpenTitan HMAC SA0/SA1 experiment. It cannot reliably identify the exact physical fault site across the 45,678-instance catalog. A no-anomaly result also cannot rule out an invisible, unactivated, or normal-compatible fault.

DEV_SITE_TEST was not opened because the calibration advancement gate failed. VALIDATION and HOLDOUT were not accessed. V1 remains frozen and unchanged.

## Future work

Any additional locator work should begin as a new V2.1 contract with a redesigned retrieval objective and a new calibration plan. It must not silently continue tuning this frozen branch or use locked partitions for model selection.
