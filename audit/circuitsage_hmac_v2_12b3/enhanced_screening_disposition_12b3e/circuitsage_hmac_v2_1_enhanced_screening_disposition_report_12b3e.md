# CircuitSage-HMAC V2.1 Enhanced-Screening Disposition

## Result

The frozen Stage 12B-3D bounded screen is valid and complete. Of the three
precommitted candidates, only `EM_TESTPOINT_4X64_T16` met the frozen screening rule.
It reached combined all-injected detection recall 0.75976562,
rescued 142 faults beyond the matched external observation, and
produced zero fault-free false alarms. Its 95% site-bootstrap interval is
[0.71875000, 0.80078125] using 1000
frozen replicates.

The two topology/state candidates each reached only 0.50000000 recall and are
rejected but preserved as negative screening results. The selected candidate's
point-recall margin over the unchanged 0.70000000 target is 0.05976562; its
gap over the best rejected candidate is 0.25976562.

## Scientific boundary

This was a stratified 256-site, 512-fault REPAIR_TRAIN screen. It establishes
that the test-point measurement is worth a full capture; it does not establish
its full-cohort recall, localization performance, model accuracy, unseen-fault
generalization, unseen-circuit generalization, or production readiness.

## Disposition

- `EM_TESTPOINT_4X64_T16` is selected and frozen as the sole advancing measurement.
- A later 1,024-site / 2,048-fault / 96-vector REPAIR_TRAIN capture is ready
  for a separate authorization stage (196608 enabled transactions).
- This stage does not authorize that execution and does not authorize model
  training.
- The full capture must still meet recall >=0.70, all-injected exact-site rate
  >=0.35, mean observable candidate sites <=50, maximum <=500, and zero
  fault-free false alarms.
- REPAIR_CALIBRATION, REPAIR_SITE_TEST, the consumed DEV_SITE_TEST,
  VALIDATION and HOLDOUT remain locked or prohibited.
