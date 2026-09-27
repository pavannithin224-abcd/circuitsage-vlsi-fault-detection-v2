# Stage 12C-3A — Independent Test Capture Authorization

**Status: PASS / FROZEN — authorization only. No capture performed.**

## Why this stage exists

Stage 12C-1D authorized synthesis for TRAIN and CALIBRATION families only, and
recorded `locked_test_families: ['ibex_cpu', 'secworks_chacha']`. Nothing in the frozen
record authorizes opening those trees. This stage creates that authorization
explicitly, so unsealing is a recorded decision rather than a side effect.

## Authorized

- elaboration, adapter construction, lint, generic synthesis
- site enumeration and fault-catalogue derivation
- fault-injection instrumentation and response capture

…for **`ibex_cpu`** and **`secworks_chacha`** only.

## Not authorized

| item | status |
|---|---|
| `serv_cpu` capture | **NOT AUTHORIZED — remains SEALED** |
| model training / inference | NOT AUTHORIZED |
| retraining, threshold change, reselection | **PROHIBITED** |
| acceptance evaluation | requires its own stage |

`serv_cpu` stays sealed deliberately: the contract requires *at least
two* independent test families, which the two above satisfy. A captured circuit
cannot be re-sealed, so one untouched circuit is retained as a reserve.

## Toolchain — contract-pinned

| tool | version | |
|---|---|---|
| `verilator` | `Verilator 5.051 devel rev v5.050-222-gf6f6f8404 (mod)` | MATCH |
| `yosys` | `Yosys 0.68+106 (git sha1 c92678eb2-dirty, Release, Clang /usr/bin/clang++ 21.1.8)` | MATCH |

Verified at the exact paths pinned in 12C-1D. A capture produced by a different
toolchain would not be comparable to the frozen development evidence.

## Sealed test sources

| family | class | revision | license | RTL files | archive |
|---|---|---|---|---|---|
| `ibex_cpu` | RISC_V_CPU | `e9f55342edbd` | Apache-2.0 | 203 | OK |
| `secworks_chacha` | CRYPTO_STREAM_CIPHER | `7eaba360df9f` | BSD-2-Clause | 3 | OK |

## Predictions already frozen for these circuits

| circuit | predicted class | predicted exact-site | meets floor 0.15 | confidence |
|---|---|---|---|---|
| `ibex_cpu` | CPU_RISCV | [0.00, 0.12] | NO | HIGH |
| `secworks_chacha` | CRYPTO_ARX_STREAM | [0.20, 0.70] | YES | MEDIUM |

**Frozen predicted overall acceptance: NOT MET.**

## One-shot acknowledgement

> ONE LOCKED TEST EVALUATION; NO RETRAINING, THRESHOLD CHANGE, OR RESELECTION AFTERWARD

Capture is **irreversible**. Expected outcome, stated before capture: **NOT
MET**, driven by the `ibex_cpu` per-circuit exact-site floor of
0.15.

The result cannot be retrofitted in either direction: acceptance criteria were
frozen in 12C-1A before any model existed, predictions in 12C-2H before capture
existed, and the structural bound in 12C-2L.

## Access counters before this stage

TEST / VALIDATION / HOLDOUT = **0 / 0 / 0**

## Next stage

**12C-3B** — independent test capture execution. Independent generalization
remains **NOT ESTABLISHED** until evaluated. Future hybrid brand remains
**Faultiva 1.0 — Hybrid VLSI Fault Intelligence**.
