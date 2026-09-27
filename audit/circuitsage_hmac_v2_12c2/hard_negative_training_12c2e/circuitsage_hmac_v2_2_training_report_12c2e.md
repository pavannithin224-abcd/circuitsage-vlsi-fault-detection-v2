# Stage 12C-2E — Hard-Negative Retraining

**Status: PASS / FROZEN — training only, no selection.**

## Why this stage exists

Stage 12C-2C sampled contrastive negatives uniformly at random. The frozen
Stage 12C-1A contract requires *"full-catalog or deterministic
topology-stratified negatives within TRAIN circuits only"*. Uniform random
negatives violate that clause and make the retrieval task trivially easy.

Stage 12C-2C is **not modified**. It is retained as the ablation baseline.

## Negative strata (TRAIN circuits only)

| family | structural strata | mean size | signature collision groups | nodes in collisions |
|---|---|---|---|---|
| `opentitan_hmac_sha256` | 42 | 437.905 | 1451 | 8650 |
| `picorv32_cpu` | 66 | 146.439 | 383 | 4621 |
| `secworks_aes` | 44 | 603.75 | 8665 | 32055 |

Composition per anchor: 50% structural,
30% signature-collision, 20% bounded uniform.

## Paired ablation

| candidate | random negatives (12C-2C) | hard negatives (12C-2E) | change |
|---|---|---|---|
| `V22_GRAPHSAGE_METRIC_SMALL` | 0.015625 | 0.019206 | +0.003581 |
| `V22_GATV2_CROSS_FUSION` | 0.014974 | 0.027344 | +0.012370 |
| `V22_GRAPHSAGE_OOD_ENSEMBLE` | 0.018555 | 0.021484 | +0.002929 |

Everything except the negative policy is held fixed: architecture, caps, seed,
partitions, optimizer, learning rate, batch size and negative count.

## Contract compliance

- gradients on GENERALIZATION_TRAIN only
- negative strata derived from TRAIN artifacts only; calibration, test and
  holdout contribute nothing to negative selection
- calibration used for monitoring and early stopping only
- parameter caps enforced before the first optimizer step

## What this stage does NOT establish

Independent generalization remains **NOT ESTABLISHED**. No selection, no
acceptance evaluation, no protected-partition access. Future hybrid brand
remains **Faultiva 1.0 — Hybrid VLSI Fault Intelligence**.

## Next gate

**Stage 12C-2D** — model selection on GENERALIZATION_CALIBRATION, if and only if
a candidate is meaningfully above chance.
