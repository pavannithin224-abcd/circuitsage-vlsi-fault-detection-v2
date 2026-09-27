# CircuitSage-HMAC V2.1 Enhanced Measurement Architecture

## Why this stage exists

The previous fixed 64-bit probe bank raised pilot observability from 1027 to
1030 of 2048 faults. Its 0.50292969 ceiling remained
well below the frozen 0.70000000 target. Training was therefore
correctly blocked.

## Frozen candidates

1. **EM_TOPOLOGY_4X64_T24** — four topology-stratified banks covering state,
   control, high-fanout logic and boundary logic.
2. **EM_STATE_CHECKPOINT_2X64_T32** — dense sequential-state and control
   checkpoints with the longest temporal schedule.
3. **EM_TESTPOINT_4X64_T16** — simulation-only global test points selected from
   topology plus aggregated REPAIR_TRAIN observability evidence.

Every candidate uses a single global schedule. The query never includes a
fault selector, fault value, site identity or fault-instance identity.

## Cost control

All candidates must first pass structural discovery and lint. A later,
separately authorized screen is limited to 256 sites,
512 faults and 48 vectors. At most one candidate may
advance to a full 2048-fault REPAIR_TRAIN capture.

No candidate may advance unless it reaches 0.70 combined detection with zero
fault-free false alarms. The target will not be lowered to force progression.

## Boundaries

This stage authorizes no discovery, simulation, capture or training. The
REPAIR_CALIBRATION and REPAIR_SITE_TEST partitions remain locked. The consumed
original DEV_SITE_TEST cannot be reopened, and VALIDATION/HOLDOUT remain
prohibited.
