# Stage 12C-2L — Structural-Equivalence Bound

**Status: PASS / FROZEN — measurement and disposition, no training.**

## What this closes

Stage 12C-2J proved within-collision-set selection is not identifiable. One
escape route remained: if collision sets were an artifact of a weak **test
scheme**, richer vectors would shrink them and the ceiling would move.

This stage tested that route before writing any model — and closed it.

## Collision sets are structurally equivalent

| family | ambiguous sets | structurally equivalent | partial | potentially splittable |
|---|---|---|---|---|
| `opentitan_hmac_sha256` | 1,451 | **99.8%** | 0.1% | 0.1% |
| `picorv32_cpu` | 383 | **83.3%** | 15.1% | 1.6% |
| `secworks_aes` | 8,665 | **89.6%** | 7.2% | 3.2% |
| `secworks_sha256` | 3,411 | **99.7%** | 0.1% | 0.2% |

Members of these sets lie **adjacent on the same signal path**. A stuck-at fault
on a gate output and the same stuck-at on the net it drives are the same
physical fault — identical under *every* input vector. This is the classical
**fault-collapsing** equivalence relation used in ATPG since the 1960s.

## Bound on test-scheme enrichment

| family | current exact-site | optimistic upper bound | floor | reachable |
|---|---|---|---|---|
| `opentitan_hmac_sha256` | 0.0681 | 0.0681 | 0.15 | **NO** |
| `picorv32_cpu` | 0.0243 | 0.0249 | 0.15 | **NO** |
| `secworks_aes` | 0.5306 | 0.5410 | 0.15 | **YES** |
| `secworks_sha256` | 0.6385 | 0.6392 | 0.15 | **YES** |

Even assuming *perfect* vector design that resolves every potentially-splittable
set, the per-circuit floor stays out of reach on CPU-class circuits.

## Complete falsification record

| stage | formulation | outcome |
|---|---|---|
| 12C-2C | metric retrieval, uniform random negatives | no transfer to unseen circuit |
| 12C-2E | metric retrieval, topology-stratified hard negatives | no transfer; contract violation fixed |
| 12C-2F | measurement against non-learning comparator | comparator 182x better (MRR 0.6745 vs 0.0037) |
| 12C-2G | graph-constrained reranking within collision sets | lift exactly 1.00x |
| 12C-2J | identifiability analysis | PROVED NOT IDENTIFIABLE |
| 12C-2K | message-passing GNN on observability | calibration AUROC 0.348, below random; depth threshold 0.768 beats it |

Four formulations were predicted by the author to work. All four failed. Each
prediction and failure is recorded rather than omitted.

## The positive result

The localizer is **not underperforming — it is operating at the structural
limit.**

A candidate set is not a hedge. It *is* the fault equivalence class, which is
precisely what commercial ATPG reports after fault collapsing. Narrowing 36,784
sites to 3 mutually-equivalent ones is a complete answer, not a partial one.

Reporting a single site would be **less** correct.

## Disposition

**V2.2 learned-model work is closed.** Remaining V2.2 work is capture and
evaluation only:

- `12C-3A` independent test capture
- `12C-3B` one-shot locked evaluation and prediction scoring

Then V1+V2 hybrid packaging as **Faultiva 1.0 — Hybrid VLSI Fault Intelligence**, public release with a
per-circuit characterization pipeline, and the dashboard.

Stages 12C-2G, 12C-2J and 12C-2K are **not modified**. Independent
generalization remains **NOT ESTABLISHED**. Sealed circuits untouched.
