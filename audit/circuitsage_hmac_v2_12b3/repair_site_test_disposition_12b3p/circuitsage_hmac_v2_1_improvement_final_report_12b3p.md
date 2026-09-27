# CircuitSage-HMAC V2.1 Improvement Final Disposition — Stage 12B-3P

## Decision

The V2.1 improvement experiment is **completed and frozen**, but its locked
final acceptance is **NOT_MET**. The Stage 12B-3O result is retained as a
useful closed-catalog research baseline. The consumed REPAIR_SITE_TEST must not
be reopened for tuning, retraining, threshold changes, or candidate reselection.

## Frozen final result

- Measurement / model: `EM_TESTPOINT_4X64_T16` / `R31_EXACT_SIGNATURE_SET`
- Cohort: 2,398 sites, 4,796 injected SA0/SA1 faults, 96 vectors
- Combined detection recall: 69.16180150% (target 70%; gap 0.83819850%)
- All-injected exact-site rate: 54.87906589% (target 60%; gap 5.12093411%)
- Observable candidate-set coverage: 100.00000000%
- Unique-signature top-1 site accuracy: 100.00000000%
- Mean / maximum observable candidate sites: 3.7561 / 84
- Maximum-candidate excess over target: 20
- Probe-rescued faults: 895
- Fault-free false alarms: 0

## What improved

Relative to the frozen Stage 12B-1F reference, detection recall increased from
50.01459428% to 69.16180150%, exact-site rate increased from
19.38120257% to 54.87906589%, and mean observable candidates decreased
from 141.6268 to 3.7561. The cohorts differ, so this
is a bounded engineering comparison rather than a paired superiority test.

## Unmet criteria

1. Combined detection recall: 0.69161802 < 0.70000000
2. All-injected exact-site rate: 0.54879066 < 0.60000000
3. Maximum observable candidate sites: 84 > 64

## Scientific boundary

The result measures closed-catalog, physical-site-held-out consistency for the
frozen HMAC SA0/SA1 experiment. It does not establish independent-circuit
generalization, unseen fault-family performance, physical-silicon readiness,
or production readiness.

## Next gate

Stage 12C-1A may define a separate V2.2 generalization architecture,
data-partition plan and acceptance contract. This disposition does not
authorize V2.2 dataset construction or training. The name
**Faultiva 1.0 — Hybrid VLSI Fault Intelligence** remains reserved for the eventual completed V1+V2 hybrid
release and is not assigned to this V2.1 component.
