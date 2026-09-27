# Stage 12C-4A — Genuine V1 Verification Wiring

**Status: PASS / FROZEN.** Stage 12C-2I is unmodified; this is an additive correction.

## The defect

Stage 12C-2I wired V1 as a **scope gate**. It loaded the model, checked the circuit
family, and returned `VERIFIED` — **without ever calling `predict_proba`**. The
string does not occur in its source. A 93,185-parameter model was reported as a
verification stage while never running.

## The genuine forward pass

| block | width | source |
|---|---|---|
| `stuck_value` | 1 | fault polarity |
| site features | 14 | fanout (scaled), stem flags, cell-type one-hot |
| stimulus | 512 | 256 key + 256 message bits |
| **sample branch** | **527** | |
| graph branch | 119 | frozen DIR_SGC_K3 cache |
| **total** | **646** | → MLP(128,64,32) → threshold 0.4965 |

## A second defect, found while fixing the first

The graph cache ships **already scaled**. Applying `k3_mean`/`k3_scale` again — the
obvious reading — inflates the branch and saturates the model:

| | correct | double-scaled |
|---|---|---|
| graph range (min/max) | [-5.27, 126.44] | [-75.0, **47961.0**] |
| graph p99.99 | **12.61** | 2369.9 |
| column mean / std | 0.0085 / [0.84, 1.18] | inflated |
| at exactly 0.0 or 1.0 | **1/400** | 143/400 |
| polarity sensitivity | **100/100** | 48/100 |

The correct branch's maximum of 126.4 comes from just **5 nodes out of
22,839** — very-high-fanout `$_DFFE_PN1P_` flip-flops. That is real netlist
structure, so correctness is asserted on column mean/std and p99.99, not on the max.

`cell_fanout` uses the **opposite** convention — stored raw, scaler must be applied.
Nothing in the published schema says so. Both conventions are now frozen in
`circuitsage_hmac_v2_2_v1_feature_scaling_convention_12c4a.json` and asserted by the self-test.

**The lesson worth keeping: a model that runs is not a model that works.** The
double-scaled pipeline threw no exception and returned confident verdicts.

## Measured behaviour

Probabilities over 400 site/polarity samples: min 0.0000, max
1.0000, mean 0.5436, std 0.3925 — graded, no saturation.

| candidate sites | total ms | ms/site |
|---|---|---|
| 1 | 0.225 | 0.22469 |
| 3 | 0.381 | 0.12687 |
| 10 | 0.256 | 0.02564 |
| 100 | 2.042 | 0.02042 |

Scope is unchanged: `opentitan_hmac_sha256` is scored, everything else returns
**OUT_OF_SCOPE**.

## Self-checks

| check | expected | observed | result |
|---|---|---|---|
| feature width equals 646 | 646 | 646 | **PASS** |
| graph branch is standardized (col mean ~ 0) | max|colmean| < 0.05 | 0.008466 | **PASS** |
| graph branch is standardized (col std ~ 1) | all in [0.5, 2.0] | [0.837, 1.183] | **PASS** |
| graph branch robust range sane (p99.99) | < 20 | 12.608 | **PASS** |
| extreme-fanout outliers are rare | < 0.1% of nodes | 5/22839 | **PASS** |
| saturation at exactly 0 or 1 is negligible | <= 1% of samples | 1/400 | **PASS** |
| probability spread is graded | std > 0.05 | 0.3925 | **PASS** |
| responds to fault polarity | >= 80/100 | 100/100 | **PASS** |
| double-scaling demonstrably worse | more saturation | 143 vs 1 | **PASS** |
| out-of-scope returns OUT_OF_SCOPE | OUT_OF_SCOPE | OUT_OF_SCOPE | **PASS** |
| 12C-2I unmodified | dabe558f90ab0897 | dabe558f90ab0897 | **PASS** |

## Release packaging gaps found

| gap | consequence | severity |
|---|---|---|
| propagation cache not shipped | a downloader cannot compute V1 inputs; the model is unusable | **BLOCKING** |
| golden netlist graph not shipped | site-feature block cannot be constructed | **BLOCKING** |
| scaling convention undocumented | silent saturation, as demonstrated in this stage | **HIGH** |
| no worked inference example | every integrator re-derives the layout from the schema | **MEDIUM** |

**Two BLOCKING gaps in the already-public V1 release**: the propagation cache and
the golden netlist graph are not shipped, so a downloader cannot compute the
model's inputs. Must be resolved in packaging.

## Next gate

Faultiva packaging: ship V1 with its feature pipeline, the documented scaling
convention, a worked example, and the V2.2 signature dictionary. Brand: **Faultiva 1.0 — Hybrid VLSI Fault Intelligence**.
