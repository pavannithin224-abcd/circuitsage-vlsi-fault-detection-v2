# Stage 12C-2G — Graph-Constrained Reranking

**Status: PASS / FROZEN — training + measurement, no selection.**

## Why the formulation changed

Stage 12C-2F falsified the retrieval formulation used in 12C-2C and 12C-2E:
trained MRR **0.0037** against the non-learning comparator's **0.6745** — a
factor of 182 in the comparator's favour, with median true-site rank near the
middle of the node list.

Ranking all nodes from a response embedding cannot transfer in principle: an
absolute response-to-node mapping learned on AES has no meaning in an unseen
circuit. This stage keeps the frozen comparator as the candidate generator and
learns only to **reorder within** each signature-collision set.

## Collision structure (the actual problem)

| family | collision sets | ambiguous faults | mean set | max set |
|---|---|---|---|---|
| `opentitan_hmac_sha256` | 2504 | 8791 (89.3%) | 177.6193 | 1230 |
| `picorv32_cpu` | 470 | 4666 (98.2%) | 55.4496 | 344 |
| `secworks_aes` | 28188 | 32078 (62.2%) | 22.5493 | 98 |
| `secworks_sha256` | 11653 | 9035 (52.3%) | 6.3485 | 44 |

## Ceiling, frozen before training

Perfect reranking cannot exceed the observable fraction. Any shortfall above
that line is an observability limit, not a model limit.

## Result

| family | baseline | reranked | change | ceiling | meets 0.15 |
|---|---|---|---|---|---|
| `opentitan_hmac_sha256` | 0.0681 | 0.0785 | +0.0105 | 0.2676 | NO |
| `picorv32_cpu` | 0.0243 | 0.0404 | +0.0161 | 0.2459 | NO |
| `secworks_aes` | 0.5306 | 0.6359 | +0.1053 | 0.9714 | YES |
| `secworks_sha256` | 0.6385 | 0.6385 | +0.0000 | 0.9467 | YES |

**`acceptance_decision: NOT EVALUATED`** — development partitions only. The
sealed circuits remain uncaptured.

## Invariants held

- candidate set never modified; coverage identical before and after
- no node outside the comparator's set is selectable
- no absolute node identity, circuit identity, or signature bytes as features
- predictions written and hashed **before** truth was consulted
  (`9ab9f663d11df47e6c896ec930516801...`)

## Next gate

If the CPU-class family clears the floor, proceed to selection. If not, the
shortfall is bounded by signature uniqueness and V2.2 should freeze the boundary
finding rather than continue model work. Future hybrid brand remains
**Faultiva 1.0 — Hybrid VLSI Fault Intelligence**.
