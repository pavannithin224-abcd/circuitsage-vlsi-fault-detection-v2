# Stage 12C-1M — Site-Eligibility Discovery

**Status: PASS / FROZEN — discovery and re-authorization only, no execution.**

## What was discovered

During Stage 12C-1L execution the campaign halted before starting
`secworks_aes`:

```
STOP: secworks_aes has only 26560 eligible driven bits; 26565 required
```

Stage 12C-1K derived each family's site ceiling from the Yosys **generic cell
count**. Stage 12C-1L derives fault sites from **distinct driven cell-output
bits**. Yosys emits `$scopeinfo` hierarchy-annotation pseudo-cells that are
counted as cells but expose no ports at all, so they can never carry a
cell-output SA0/SA1 fault.

This is a physical property of the frozen netlists that became visible only at
execution time. **Stage 12C-1K is not defective** and is not modified by this
stage; it was correct given the information available when it was frozen.

## Authority to re-authorize

The frozen Stage 12C-1K contract anticipates this outcome:

> `fault_catalog_policy`: DERIVE ELIGIBLE SITES DETERMINISTICALLY; ACTUAL
> COUNT MAY NOT EXCEED FROZEN MAXIMUM

Reducing the ceiling for a contracted exclusion is permitted; exceeding it is
not. Every amended family count is **at or below** its frozen ceiling.

## Eligibility rule

A fault site is a distinct driven cell-output net bit: for every cell in the frozen 12C-1E generic netlist top module, in deterministic sorted order, for every port whose port_direction is 'output', every integer net bit with id >= 2 that has not already been claimed by an earlier cell/port. Constant bits 0/1 and already-claimed net bits are not sites.

## Amended budget

| family | generic cells | frozen ceiling | eligible sites | reduction | excluded cells | batches |
|---|---|---|---|---|---|---|
| `opentitan_hmac_sha256` | 18392 | 18392 | 18392 | 0 | 0 | 288 → 288 |
| `picorv32_cpu` | 9665 | 9665 | 9665 | 0 | 0 | 152 → 152 |
| `secworks_aes` | 26565 | 26565 | 26560 | 5 | 5 | 416 → 415 |
| `secworks_sha256` | 9127 | 9127 | 9125 | 2 | 2 | 143 → 143 |

Frozen total: **63749** sites / **999** batches.
Amended total: **63742** sites / **998** batches.
Reduction: **7** sites. Ceiling exceeded: **NO**.

## Excluded cells

| family | cell | type | reason |
|---|---|---|---|
| `secworks_aes` | `dec_block` | `$scopeinfo` | NO_OUTPUT_PORT |
| `secworks_aes` | `dec_block.inv_sbox_inst` | `$scopeinfo` | NO_OUTPUT_PORT |
| `secworks_aes` | `enc_block` | `$scopeinfo` | NO_OUTPUT_PORT |
| `secworks_aes` | `keymem` | `$scopeinfo` | NO_OUTPUT_PORT |
| `secworks_aes` | `sbox_inst` | `$scopeinfo` | NO_OUTPUT_PORT |
| `secworks_sha256` | `k_constants_inst` | `$scopeinfo` | NO_OUTPUT_PORT |
| `secworks_sha256` | `w_mem_inst` | `$scopeinfo` | NO_OUTPUT_PORT |

## Completed-work continuity

Stage 12C-1L completed batches verified unchanged: **440**.
Re-simulation required: **0**.

Families with completed batches retain identical site counts, global batch
identifiers, site ranges, fault index ranges and vector counts under the
amended plan.

## What this stage did not do

No simulation, fault injection, netlist instrumentation, dataset construction,
model loading, training, selection or inference. Stage 12C-1K, 12C-1E and
12C-1L artifacts were verified byte-identical and never modified. Independent
TEST remains locked, VALIDATION unopened, HOLDOUT sealed. Independent
generalization remains **NOT ESTABLISHED**. The future hybrid brand remains
**Faultiva 1.0 — Hybrid VLSI Fault Intelligence**.

## Next gate

**Stage 12C-1N** — resume the full TRAIN/CALIBRATION campaign against the
amended plan frozen here. Model training, selection and inference remain
**NOT AUTHORIZED**.
