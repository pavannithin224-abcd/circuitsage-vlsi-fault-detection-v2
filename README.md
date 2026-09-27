<div align="center">

# CircuitSage V2

### Behaviour-Signature Fault Localization in VLSI Circuits — Method and Audit Trail

**A pre-registered study of whether learned models can localize stuck-at faults across unseen circuits.**
**They could not. This repository documents why, and what the actual limit is.**

<p>
<img alt="status" src="https://img.shields.io/badge/status-research%20artifact-blue?style=flat-square">
<img alt="license" src="https://img.shields.io/badge/license-Apache--2.0-green?style=flat-square">
<img alt="stages" src="https://img.shields.io/badge/frozen%20stages-36-orange?style=flat-square">
<img alt="falsified" src="https://img.shields.io/badge/formulations%20falsified-6-red?style=flat-square">
<img alt="generalization" src="https://img.shields.io/badge/generalization-NOT%20ESTABLISHED-lightgrey?style=flat-square">
</p>

<sub>Study branch of the **CircuitSage** programme · [V1 (HMAC detection)](https://github.com/pavannithin224-abcd/circuitsage-hmac-fault-detection) · Faultiva (shipping model) — released separately</sub>

</div>

---

> [!IMPORTANT]
> **This repository is a laboratory notebook, not a product.** It contains no installable
> model. If you want the working fault detector, use **Faultiva** — this repository
> exists so that Faultiva's claims can be audited.

---

## The result in one table

Same circuit. Same faults. Same method. Only the candidate catalogue changed.

| candidate catalogue | exact-site localization rate |
|---|---|
| single circuit (V1 scope) | **0.5488** |
| four circuits (V2 scope) | **0.0286** |
| | **19.2× degradation** |

Localization accuracy was never a property of the model. It was a property of **how many
candidates could produce the same observable behaviour**. Once that was measured properly,
every learned formulation collapsed to chance — and a method with **zero learned
parameters** outperformed all of them by ~182× on MRR.

---

## What was measured

Four circuits, full stuck-at campaigns, **7,849,696 transactions**, zero false alarms.

| circuit | class | faults | observable | exact-site | MRR | structurally equivalent |
|---|---|---|---|---|---|---|
| `secworks_sha256` | crypto hash | 18,250 | 94.7% | **0.4516** | **0.6745** | 99.71% |
| `secworks_aes` | crypto block | 53,120 | 97.1% | **0.3675** | 0.5463 | 89.58% |
| `opentitan_hmac_sha256` | crypto + control | 36,784 | 26.8% | 0.0286 | 0.2544 | **99.79%** |
| `picorv32_cpu` | RISC-V CPU | 19,330 | 24.6% | 0.0045 | 0.0989 | 83.29% |

Then, on a **sealed circuit never seen during development**:

| | |
|---|---|
| circuit | `secworks_chacha` (ARX stream cipher) |
| prediction, frozen **before** the seal was broken | exact-site ∈ [0.20, 0.70], meets floor |
| measured | **0.3440** |
| outcome | **CONFIRMED** |

The signature table was hashed before any metric was computed, and the capture stage was
mechanically prevented from reading the predicted bands.

---

## Why the learned models failed — and why that was the correct answer

Six formulations were pre-registered and falsified:

| formulation | parameters | outcome |
|---|---|---|
| `GRAPHSAGE_METRIC_SMALL` | 80,708 | no transfer to unseen circuits |
| `GATV2_CROSS_FUSION` | 182,052 | no transfer; apparent gain within noise |
| `GRAPHSAGE_OOD_ENSEMBLE` | 133,702 | no transfer |
| set reranker | 40,033 | lift exactly **1.00×** at every epoch |
| observability GNN | 41,649 | AUROC **0.348** — below random |
| hard-negative retraining | — | endpoint identical to random negatives |

The reranker's 1.00× lift looked like a failure. It was not. Subsequent analysis proved the
task is **not identifiable**: every site carries exactly **2** faults (SA0, SA1), so the
within-set posterior is 1/k regardless of model capacity. The true site sits at depth
percentile 0.4961–0.5083 against a uniform expectation of 0.5000 — no feature separates it.

**A lift of exactly 1.00× is the mathematically correct result.** The model was asked an
impossible question.

### The real limit

| circuit | structurally equivalent | current exact-site | optimistic ceiling |
|---|---|---|---|
| `opentitan_hmac_sha256` | 99.79% | 0.0681 | **0.0681** |
| `picorv32_cpu` | 83.29% | 0.0243 | **0.0249** |
| `secworks_aes` | 89.58% | 0.5306 | 0.5410 |
| `secworks_sha256` | 99.71% | 0.6385 | 0.6392 |

Between **83.3% and 99.8%** of ambiguous candidate sets are *structurally equivalent* — the
faults inside them are indistinguishable from any response, by any method. A perfect
reranker would move HMAC from 0.0681 to 0.0681.

> **Defensible claim.** Behaviour-signature localization resolves faults to their
> **equivalence class** — the finest resolution any response-based method can achieve. The
> residual ambiguity is structural, not methodological, and matches the classical
> fault-collapsing result.

---

## Acceptance outcome: NOT MET

The pre-registered contract required the per-circuit exact-site floor (≥ 0.15) on **both**
independent test circuits.

| circuit | captured | exact-site | meets floor |
|---|---|---|---|
| `secworks_chacha` | yes | 0.3440 | **yes** |
| `ibex_cpu` | **no** | — | — |

`ibex_cpu` could not be captured: the acquired snapshot is not self-contained for gate-level
synthesis (missing `lc_ctrl_pkg`, RAM primitive packages, `PRIM_FLOP_SPARSE_FSM`, technology
primitives). Authoring substitute RTL was **refused** — faults would then be injected into
logic written by this project rather than into ibex, inside a one-shot evaluation where the
contamination could never be corrected.

**The outcome is capture infeasibility, not a measured shortfall.** The one-shot locked
evaluation remains **unconsumed**.

Accordingly, independent-circuit generalization is reported as **NOT ESTABLISHED**. One
crypto-class circuit met its floor on a prediction frozen in advance; the CPU-class circuit
that would have tested the harder half of the hypothesis could not be measured at all. One
confirmed prediction on one circuit is genuine evidence — and it is one data point.

---

## Repository layout

```
stages/     76 stage scripts — the method, every one independently runnable
config/     66 pre-registered contracts — acceptance criteria frozen before measurement
audit/      176 freeze/manifest JSONs + 123 metric CSVs — the hash-anchored chain
corpus/     6 pinned RTL archives + acquisition registry
rtl/ tb/    21 HDL sources — wrappers and fault-injection testbenches
```

Every stage script supports `--status` (read back a frozen result) and `--self-test`
(validate preconditions without executing).

```bash
python3 stages/stage_12c2l_equivalence_bound.py --status
python3 stages/stage_12c3b_chacha_capture.py --self-test
```

---

## Methodological controls

These are the reason this repository exists.

| control | mechanism |
|---|---|
| **Append-only evidence** | no frozen stage is ever edited; corrections become new stages |
| **Prediction before truth** | sealed-circuit predictions frozen before any seal was broken; capture stages mechanically blocked from reading them |
| **One-shot evaluation** | the locked test may be opened once — it remains unconsumed |
| **Protected-partition accounting** | every stage records test/validation/holdout access counts; all are 0 |
| **Toolchain pinning** | Verilator `5.051 devel rev v5.050-222-gf6f6f8404 (mod)`, Yosys `0.68+106 (git sha1 c92678eb2-dirty)`; re-simulation reproduces frozen batches byte-identically |
| **Sealed holdout** | `serv_cpu` remains sealed by decision and is **not redistributed here** |
| **Disclosed defects** | four defects found during development are recorded in-repo, not silently fixed |

### Defects on the record

| defect | resolution |
|---|---|
| V1 verification was wired as a scope gate that never called `predict_proba` | fixed in a new stage; the original is preserved unmodified |
| Graph features double-scaled, saturating 34% of outputs to exactly 0 or 1 | fixed; convention frozen and asserted by regression self-test |
| Shipped dictionary stored candidate *counts* but not candidate *sites* | fixed; 127,484 site entries exported |
| Prediction scoring rule had no label for an uncapturable circuit | `NOT_SCORABLE` added; original rule unmodified, omission recorded as a defect |

---

## Reproducibility — and its honest limit

> [!WARNING]
> **Reproducibility here is partial, and you should know exactly how.**
>
> **What you can verify now:** every stage's logic, every pre-registered contract, every
> metric CSV, and the internal consistency of the audit chain — all 512 files ship with
> their recorded hashes.
>
> **What you cannot verify without rerunning:** the raw simulation data is **3.7 GB across
> 4,827 batch files** and is not redistributable at this size. Many audit hashes reference
> those inputs. To re-verify them independently you must rerun the campaigns with the pinned
> toolchain — hours to days of CPU time.
>
> **What is impossible to re-verify:** `ibex_cpu` cannot be characterized from the pinned
> snapshot at all (see above). That is a property of the upstream source, not of this work.

Claiming "fully reproducible" would undercut the exact rigour this repository documents. It
is *inspectable* and *rerunnable*, not *re-verifiable offline*.

---

## Relationship to the other repositories

| repository | what it is | audience |
|---|---|---|
| **V1** | HMAC fault detection — a trained classifier, 93,185 parameters | people wanting the original model |
| **V2** *(here)* | the generalization study — method, contracts, audit trail, negative results | reviewers, examiners, researchers |
| **Faultiva** | the shipping hybrid — detection + localization + verification in one package | people wanting a working tool |

Faultiva's model card cites audit hashes from this repository. That is what this repository
is for: making those citations checkable.

---

## Citation

If this work informs yours, please cite it. See [`CITATION.cff`](CITATION.cff) — GitHub
renders a ready-made citation from it via **Cite this repository**.

```bibtex
@software{circuitsage_v2_2026,
  title   = {CircuitSage V2: Behaviour-Signature Fault Localization is Bounded
             by Structural Fault Equivalence},
  author  = {Nithin, Pavan},
  year    = {2026},
  version = {2.2.0},
  url     = {https://github.com/pavannithin224-abcd/circuitsage-vlsi-fault-detection-v2},
  note    = {Pre-registered study; independent-circuit generalization not established}
}
```

---

## License

Apache-2.0 — see [`LICENSE`](LICENSE). Chosen to match V1 and for its explicit patent grant.

Third-party RTL redistributed under `corpus/` retains its **original upstream licensing**
(Apache-2.0, ISC, BSD-2-Clause). See [`THIRD_PARTY_NOTICE.md`](THIRD_PARTY_NOTICE.md) for
per-project attribution and pinned revisions. Apache-2.0 covers only the contributions
authored here.

---

<div align="center">
<sub>

**Negative results, reported as results.**

Six falsified formulations, one impossibility proof, one structural bound, and an acceptance
outcome of NOT MET — published rather than buried, because the bound is the finding.

</sub>
</div>
