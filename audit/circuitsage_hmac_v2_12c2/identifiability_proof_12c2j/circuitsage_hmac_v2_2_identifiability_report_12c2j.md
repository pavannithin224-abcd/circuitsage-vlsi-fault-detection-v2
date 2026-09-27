# Stage 12C-2J — Identifiability of Within-Collision-Set Localization

**Status: PASS / FROZEN — measurement and correction, no training.**

## What this stage corrects

Stage 12C-2G measured a calibration lift of **exactly 1.00x** for a reranker
selecting the true site inside an exact-signature collision set, and that was
recorded as a falsification of the learned approach.

**That interpretation was wrong.** The task is not hard — it is *not
identifiable*. A lift of exactly 1.00x is the **correct** answer.

## The argument

1. Every site carries exactly **two** faults (SA0, SA1) → the prior over sites
   is exactly uniform. *Measured below, not assumed.*
2. A collision set is *defined* as the faults whose observable response is
   identical → the likelihood is constant across its members.
3. Uniform prior x constant likelihood = uniform posterior → every member has
   posterior exactly **1/k**.
4. Therefore no function of the observation can rank one member above another.

## 1. Fault-prior uniformity

| family | sites | faults/site min | max | uniform |
|---|---|---|---|---|
| `opentitan_hmac_sha256` | 18,392 | 2 | 2 | **YES** |
| `picorv32_cpu` | 9,665 | 2 | 2 | **YES** |
| `secworks_aes` | 26,560 | 2 | 2 | **YES** |
| `secworks_sha256` | 9,125 | 2 | 2 | **YES** |

## 2. Within-set separability

| family | ambiguous sets | mean faults/set | mean distinct sites/set | max deviation from uniform |
|---|---|---|---|---|
| `opentitan_hmac_sha256` | 1,451 | 6.0586 | 5.9614 | 0.0040 |
| `picorv32_cpu` | 383 | 12.1828 | 12.0653 | 0.0023 |
| `secworks_aes` | 8,665 | 3.702 | 3.6994 | 0.0152 |
| `secworks_sha256` | 3,411 | 2.6488 | 2.6482 | 0.0072 |

The true site's percentile within its own collision set is indistinguishable
from 0.5 on every circuit and every structural feature tested. Collisions are
genuinely **across sites** (mean distinct sites per set > 1.5), not SA0/SA1
pairs of a single site.

## Consequence

A selector scoring **above** 1.00x under this catalogue would indicate label
leakage or a catalogue-ordering artifact — not skill. Stage 12C-2G's reranker
behaved correctly.

## How localization can actually improve

- **Enrich the test scheme** — more or better vectors, additional observation
  points. This shrinks collision sets and attacks the bound itself.
- **Predict observability or set size**, where structural information
  demonstrably exists (Stage 12C-2K).
- **Report candidate sets honestly** rather than forcing a single site.

## Lesson recorded

> Before attributing a null result to a model, verify that the target is
> identifiable from the inputs provided.

Stage 12C-2G is **not modified**. Independent generalization remains **NOT
ESTABLISHED**. Future hybrid brand remains **Faultiva 1.0 — Hybrid VLSI Fault Intelligence**.
