# CircuitSage-HMAC V2.1 Full-Capture Disposition — Stage 12B-3H

## Decision

The enhanced measurement target is **met**. `EM_TESTPOINT_4X64_T16` is retained
as the frozen measurement winner for repair-model contract design.

## Frozen result

- Cohort: 1,024 REPAIR_TRAIN sites, 2,048 injected faults, 96 vectors
- Combined detection recall: 75.87890625%
- All-injected exact-site rate: 67.87109375%
- Probe-rescued faults: 527
- Mean / maximum observable candidate sites: 1.9595 / 36
- Fault-free false alarms: 0
- Remaining normal-compatible faults: 494

## Interpretation

These values measure deterministic consistency inside the known HMAC SA0/SA1
catalog using the frozen REPAIR_TRAIN cohort. They are not trained-model
accuracy and do not establish performance on unseen circuits, unseen fault
types, physical silicon, or out-of-catalog behavior.

## Disposition

Stage 12B-3I may define the repair-model architecture, group-disjoint data
usage, candidate grid, objectives and stopping rules. Training remains blocked
until a separate authorization verifies that contract. REPAIR_SITE_TEST,
DEV_SITE_TEST, VALIDATION and HOLDOUT remain closed.
