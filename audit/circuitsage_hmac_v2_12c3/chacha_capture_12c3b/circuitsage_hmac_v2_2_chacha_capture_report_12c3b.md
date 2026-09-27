# Stage 12C-3B — Independent Test Capture: `secworks_chacha`

**Status: PASS / FROZEN — capture and measurement only. No evaluation.**

## What was captured

| | |
|---|---|
| partition | INDEPENDENT_CIRCUIT_TEST |
| authorized by | 12C-3A |
| cells | 10,111 |
| sites | 10,111 |
| faults (SA0+SA1) | 20,222 |
| vectors (frozen 12C-1F rule) | 64 |
| transactions | 1,294,208 |

## Measured

| metric | value |
|---|---|
| observable fraction | 0.9988 |
| distinct signatures | 9,137 |
| signature uniqueness (of observable) | 0.3445 |
| **all-injected exact-site rate** | **0.3440** |
| observable candidate-set coverage | 1.0000 |
| fault-free false-alarm rate | 0.0000 |
| mean ambiguous set | 3305.815 |
| max candidate set | 6,614 |

## Integrity

- every batch row count matched its expected value exactly
- fault-free baseline captured separately and used as the reference
- signature table hashed **before** any metric was computed
  (`aec3da6923d6e7a7644326fa7287dbae...`)
- Stage 12C-2H predicted bands were **not read** by this stage

## Not done here

No acceptance evaluation, no scoring against predictions, no model inference.
`ibex_cpu` was not opened. `serv_cpu` remains **SEALED**.

## Next gate

**12C-3C** — `ibex_cpu` capture. Scoring against the frozen 12C-2H predictions
happens only after both test circuits are captured. Future hybrid brand remains
**Faultiva 1.0 — Hybrid VLSI Fault Intelligence**.
