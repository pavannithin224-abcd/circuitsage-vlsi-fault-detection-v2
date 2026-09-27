# CircuitSage-HMAC V2.2 Elaboration and Synthesis Authorization — Stage 12C-1D

The seven-family source corpus and every deterministic archive replayed against
the Stage 12C-1C integrity lock. Four families are authorized for the next
gate: three TRAIN circuits and one CALIBRATION circuit.

The portable wrapper contract normalizes clock/reset, request, payload,
validity, response, completion and timeout behavior without exposing fault
identity. Upstream RTL must remain byte-identical. Wrapper behavior and build
recipes may be developed only for TRAIN and CALIBRATION.

Both INDEPENDENT TEST circuits remain locked against elaboration, wrapper
tuning and synthesis. The HOLDOUT remains sealed. This stage performed no
elaboration, synthesis, simulation, fault injection, dataset construction,
training or inference.

**Faultiva 1.0 — Hybrid VLSI Fault Intelligence** remains reserved for the eventual completed hybrid release.
