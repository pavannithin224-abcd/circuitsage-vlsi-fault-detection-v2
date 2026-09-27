# CircuitSage-HMAC V2.1 Repair-Model Contract — Stage 12B-3I

## Purpose

This contract defines a small CPU-feasible localization repair around the
frozen `EM_TESTPOINT_4X64_T16` behavior measurements. Detection remains a
deterministic anomaly gate. The trainable portion is limited to approximate
retrieval and reranking when an exact catalog signature is unavailable.

## Scientific guardrails

- Fault identity, site, selector and stuck value are forbidden query inputs.
- Exact matches take precedence over learned scores.
- Physically indistinguishable sites remain an explicit candidate set.
- A learned model cannot declare one site unique when the complete signatures
  of multiple sites are identical.
- REPAIR_TRAIN is the only fitting partition.
- REPAIR_CALIBRATION is reserved for selection and confidence calibration.
- REPAIR_SITE_TEST remains sealed for one final evaluation.
- Original DEV_SITE_TEST, VALIDATION and HOLDOUT remain prohibited.

## Current frozen measurement

- Detection recall: 0.75878906
- All-injected exact-site rate: 0.67871094
- Mean / maximum observable candidates: 1.9595 / 36
- False alarms: 0

These are measurement-consistency results, not trained-model accuracy.

## Next action

Stage 12B-3J may create a bounded authorization for capturing the same enhanced
measurement on the already-frozen REPAIR_CALIBRATION partition. It may not
train a model or open REPAIR_SITE_TEST.
