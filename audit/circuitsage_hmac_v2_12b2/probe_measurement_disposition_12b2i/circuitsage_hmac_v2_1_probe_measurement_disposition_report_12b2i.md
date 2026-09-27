# CircuitSage-HMAC V2.1 Probe-Measurement Disposition

## Result

Stage 12B-2H produced a valid and complete probe-response dataset, but the
measurement objective was not met. External behavior exposed
1027 of 2048 pilot fault instances. Adding the fixed
64-bit probe bank exposed 1030 faults, an increase of only
3 faults (0.00146484
absolute recall).

The combined ceiling is 0.50292969, below the frozen target of
0.70000000 by 0.19707031. This is a measurement limitation, not a
dataset-integrity or simulator failure.

## Disposition

- The Stage 12B-2H dataset is valid, frozen, and retained as a negative result.
- Training a repair detector or locator from this measurement system is blocked.
- The adaptive-vector localization gain remains preserved for observable pilot faults:
  exact-site rate 0.41748047, mean candidate set 6.0925, maximum 69.
- REPAIR_CALIBRATION, REPAIR_SITE_TEST, original DEV_SITE_TEST, VALIDATION and
  HOLDOUT remain unopened or locked under their existing contracts.

## Authorized follow-up

Only an enhanced-measurement architecture and budget contract may be created.
It must compare at least three identity-independent designs using multi-bank
topology probes, sequential-state checkpoints, or controlled simulation-only
test points. Capture and training still require separate authorization.

The 0.70 detection target is unchanged. If the redesigned measurement still
cannot reach that ceiling on REPAIR_TRAIN, model training must remain blocked.
