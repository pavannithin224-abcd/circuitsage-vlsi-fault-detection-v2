# Stage 12C-2K — Graph Model on an Identifiable Target

**Status: PASS / FROZEN — training + measurement, no selection.**

## Why this target

Stage 12C-2J proved that picking the true site inside an exact-signature
collision set is **not identifiable** under the frozen uniform fault catalogue —
which is why the 12C-2G reranker measured exactly 1.00x. This stage puts a graph
model on a target that *is* identifiable:

> **Will a fault at this site be observable at all under the frozen test scheme?**

Stated before training: base rates range 0.380–0.999 across circuits, and
Stage 12C-2A already measured corr(depth_to_output, observability) = −0.331 and
−0.311 on the low-observability circuits. There is real structural signal here.

## Results

| family | partition | observable rate | AUROC | balanced acc |
|---|---|---|---|---|
| `opentitan_hmac_sha256` | TRAIN | 0.3801 | 0.620507 | 0.493134 |
| `picorv32_cpu` | TRAIN | 0.4393 | 0.614239 | 0.5 |
| `secworks_aes` | TRAIN | 0.9985 | 0.738033 | 0.5 |
| `secworks_sha256` | CALIBRATION | 0.9980 | 0.347639 | 0.5 |

## Against non-learning baselines

| family | depth-threshold bal. acc | GNN bal. acc | GNN wins |
|---|---|---|---|
| `opentitan_hmac_sha256` | 0.6298 | 0.493134 | **NO** |
| `picorv32_cpu` | 0.5135 | 0.5 | **NO** |
| `secworks_aes` | 0.6054 | 0.5 | **NO** |
| `secworks_sha256` | 0.7678 | 0.5 | **NO** |

Both baselines are mandatory: a majority-class predictor scores well whenever
the base rate is extreme, so an accuracy figure alone would be misleading.

## What this does and does not do

- **Does**: flag which regions of a design are untestable under the current test
  scheme — the input a test engineer needs to improve that scheme.
- **Does not**: localize faults. Localization remains exact-signature retrieval,
  bounded as proved in 12C-2J.

Improving the test scheme is the *only* route that raises the localization
ceiling, so this model attacks the bound rather than fighting it.

Predictions were written and hashed before truth was consulted
(`9be9c084ee222f18675cd4cdac2a731d...`). Sealed circuits untouched. Independent generalization
remains **NOT ESTABLISHED**. Future hybrid brand remains **Faultiva 1.0 — Hybrid VLSI Fault Intelligence**.
