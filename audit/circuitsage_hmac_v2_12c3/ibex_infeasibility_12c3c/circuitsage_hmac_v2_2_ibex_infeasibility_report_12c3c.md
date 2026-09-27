# Stage 12C-3C — `ibex_cpu` Capture Infeasibility

**Status: PASS / FROZEN — determination only. No synthesis, simulation or evaluation.**

## Determination

**`ibex_cpu` is NOT CAPTURABLE from the acquired snapshot.**

The Stage 12C-1C acquisition captured a 382 KB ibex tree at
revision `e9f55342edbd` containing 194 SystemVerilog
modules. That snapshot is **not self-contained for gate-level synthesis**: it
references OpenTitan infrastructure that lives in the wider lowRISC repository.

## Elaboration attempts

| # | frontend | blocker class | fixable without authoring RTL |
|---|---|---|---|
| 1 | yosys legacy Verilog frontend | TOOLCHAIN FRONTEND LIMITATION | YES - use read_slang instead |
| 2 | verilator --lint-only | MISSING VERIFICATION-ONLY INCLUDE | YES - no-op macro stub adds no logic |
| 3 | verilator --lint-only | INCOMPLETE VENDOR FILE LIST | YES - supply all archive files |
| 4 | yosys read_slang (full SystemVerilog frontend) | SNAPSHOT NOT SELF-CONTAINED | NO - requires authoring substitute RTL or re-acquiring a wider source tree |

The SystemVerilog **parses correctly** under `read_slang`. The blocker is missing
source, not syntax, and not the toolchain.

## Missing dependencies

| dependency | kind | upstream home | substitutable |
|---|---|---|---|
| `lc_ctrl_pkg` | SystemVerilog package | OpenTitan hw/ip/lc_ctrl | NO |
| `prim_ram_1p_pkg` | SystemVerilog package | OpenTitan hw/ip/prim | NO |
| `prim_ram_1r1w_pkg` | SystemVerilog package | OpenTitan hw/ip/prim | NO |
| `PRIM_FLOP_SPARSE_FSM` | compiler macro | OpenTitan prim_flop_macros.sv | NO |
| `RVFI` | global define | ibex formal verification harness | PARTIAL - define only |
| `prim_buf / prim_flop / prim_clock_gating / prim_ram_1p` | technology primitive modules | OpenTitan per-target prim libraries | NO |
| `dv_fcov_macros.svh` | verification include | OpenTitan dv/sv/dv_utils | YES - verification-only, no-op stub adds no logic |

## Substitute RTL was refused

Authoring stand-ins for `lc_ctrl_pkg`, the RAM packages and
`PRIM_FLOP_SPARSE_FSM` would mean **injecting faults into logic written by this
project rather than into ibex**. The resulting measurement would partly
characterise the substitute — inside a ONE-SHOT evaluation where it could never
be corrected.

Two verification-only stubs were used during probing (`dv_fcov_macros.svh`,
`formal_tb_frag.svh`). They expand to nothing, add no design logic, and appear in
**no frozen measurement**.

## Acceptance disposition

| circuit | captured | exact-site | meets 0.15 floor |
|---|---|---|---|
| `secworks_chacha` | YES | **0.3440** | **YES** |
| `ibex_cpu` | **NO** | — | — |

**Acceptance outcome: NOT MET** — because one of the two required independent
test circuits could not be captured, not because of a measured shortfall.

Stage 12C-2H predicted NOT MET before any seal was broken. The outcome label
agrees, but **the mechanism differs**: 12C-2H predicted a measured ibex shortfall,
whereas the actual cause is capture infeasibility. That distinction must be
preserved wherever this result is reported.

## What remains unspent

- the **one-shot locked evaluation** was not consumed: no model was evaluated
- `serv_cpu` remains **SEALED**
- `secworks_chacha` capture stands as valid, frozen evidence

## Next gate

Prediction scoring for `secworks_chacha` and the V2.2 closing disposition.
Independent generalization remains **NOT ESTABLISHED**. Future hybrid brand
remains **Faultiva 1.0 — Hybrid VLSI Fault Intelligence**.
