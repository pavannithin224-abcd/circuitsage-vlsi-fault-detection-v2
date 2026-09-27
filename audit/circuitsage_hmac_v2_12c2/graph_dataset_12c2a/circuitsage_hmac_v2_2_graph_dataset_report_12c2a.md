# Stage 12C-2A — Portable Circuit-Graph Dataset

**Status: PASS / FROZEN — dataset construction and diagnostic only.**

## Why this stage exists

The frozen Stage 12C-1A architecture contract requires a **circuit-normalized
graph encoder** alongside the response encoder. Stage 12C-1O captured response
evidence only. Without a graph dataset the contracted model cannot be trained.

## What was built

One portable directed graph per captured family, with
**20 semantic node features** and no absolute identity.

| family | partition | nodes | edges | sites | sequential | max depth→out |
|---|---|---|---|---|---|---|
| `opentitan_hmac_sha256` | GENERALIZATION_TRAIN | 18,392 | 35,416 | 18,392 | 75 | 22 |
| `picorv32_cpu` | GENERALIZATION_TRAIN | 9,665 | 21,225 | 9,665 | 1,601 | 35 |
| `secworks_aes` | GENERALIZATION_TRAIN | 26,565 | 50,547 | 26,560 | 2,469 | 27 |
| `secworks_sha256` | GENERALIZATION_CALIBRATION | 9,127 | 19,062 | 9,125 | 1,033 | 22 |

Node features: `type_index, is_sequential, is_mux, is_inverting, fan_in, fan_out, log1p_fan_in, log1p_fan_out, norm_fan_in, norm_fan_out, depth_from_input, depth_to_output, norm_depth_from_input, norm_depth_to_output, reconvergence_degree, drives_output_port, driven_by_input_port, seq_distance_to_output, norm_seq_distance_to_output, is_fault_site`.

**Portability guarantees**

- no cell name, net name, path, circuit name, partition token or site index is a feature
- cell-type vocabulary is **closed** (22 entries) with a reserved
  `<UNKNOWN>` slot at index 0 for unseen types at inference
- normalization is **per-circuit**; no scaler is fitted across circuits, so an
  unseen test circuit is normalized by its own statistics only
- fault identity, stuck value and truth labels are absent

**Site alignment:** `site_node_index[site_rank]` maps the frozen Stage 12C-1O
site order onto graph nodes, so response evidence and graph evidence are
index-aligned without any identity leak.

## Observability diagnostic (descriptive, no causal claim)

Stage 12C-1P established that observability, not retrieval, limits detection.
It did not establish *why* observability differs by family.

An informal structural hypothesis — that the gap follows sequential-cell
fraction — is **falsified by this corpus**:

| family | observable frac | sequential frac | mean depth→out | corr(depth) | corr(seq dist) | corr(fanout) |
|---|---|---|---|---|---|---|
| `opentitan_hmac_sha256` | 0.3801 | 0.0041 | 11.5 | -0.331 | +0.065 | +0.000 |
| `picorv32_cpu` | 0.4393 | 0.1656 | 9.6 | -0.311 | +0.073 | +0.000 |
| `secworks_aes` | 0.9985 | 0.0930 | 12.1 | +0.003 | -0.007 | +0.000 |
| `secworks_sha256` | 0.9980 | 0.1132 | 11.1 | +0.041 | +0.054 | +0.000 |

Cross-family correlation between sequential fraction and observable fraction:
**+0.2254**.

`opentitan_hmac_sha256` has the **lowest** sequential-cell fraction of the four
families and among the **lowest** observable fraction. Sequential depth alone
therefore does not explain the gap. This stage records the measured
associations and **asserts no causal explanation**.

## What this enables

The graph encoder can now reason about *where a fault may be hiding when the
response is silent* — structural evidence that response data alone cannot
supply. That is the intended mechanism for closing the per-circuit floor on the
sealed test circuits.

## Access

No protected partition accessed. INDEPENDENT_TEST **LOCKED**, VALIDATION
**UNOPENED**, HOLDOUT **SEALED**. Training and inference remain **NOT
AUTHORIZED**. Independent generalization remains **NOT ESTABLISHED**. Future
hybrid brand remains **Faultiva 1.0 — Hybrid VLSI Fault Intelligence**.

## Next gate

**Stage 12C-2B** — calibration capture authorization, then training
authorization for the four frozen candidates in the Stage 12C-1A grid.
