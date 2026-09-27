# CircuitSage-HMAC V2.2 Multi-Circuit Corpus Authorization — Stage 12C-1B

## Frozen corpus plan

Seven independent RTL families are registered across five design classes.
The fixed split is 3 TRAIN, 1 CALIBRATION, 2 locked INDEPENDENT TEST, and 1
blocked HOLDOUT. OpenTitan HMAC is forced into TRAIN because it informed the
earlier project; it cannot be used as evidence of unseen-circuit performance.

## License and provenance boundary

Only Apache-2.0, BSD-2-Clause, and ISC sources are registered. This stage
records upstream repositories, immutable commits, license paths and expected
SPDX identifiers. It does not download or redistribute upstream RTL. Stage
12C-1C must verify the actual archive and license bytes, dependency licenses,
notices, file inventory, and recursive hashes. Any mismatch fails closed.

## Current authorization

The next stage may download the seven pinned source archives solely to verify
provenance and freeze the corpus. Elaboration, synthesis, simulation, fault
injection, vector/probe optimization, dataset creation, model training,
inference, and protected truth access remain prohibited.

## Naming boundary

**Faultiva 1.0 — Hybrid VLSI Fault Intelligence** remains reserved for the eventual completed V1+V2 hybrid.
The V2.2 research component is not yet Faultiva.
