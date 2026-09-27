# Stage 12C-3D — V2.2 Closing Disposition

**Status: PASS / FROZEN — scoring and disposition. No evaluation, no training.**

## Prediction scorecard

Scored strictly against the frozen 12C-2H rule.

| circuit | predicted band | predicted floor | confidence | measured | outcome |
|---|---|---|---|---|---|
| `ibex_cpu` | [0.00, 0.12] | NO | HIGH | -- | **NOT_SCORABLE** |
| `secworks_chacha` | [0.20, 0.70] | YES | MEDIUM | 0.3440 | **CONFIRMED** |
| `serv_cpu` | [0.00, 0.12] | NO | LOW | -- | **NOT_SCORABLE** |

**`secworks_chacha`: CONFIRMED.** Predicted [0.20, 0.70] with floor YES before the
seal was broken; measured **0.3440**, inside
the band and above the 0.15 floor. A genuine prediction-before-truth result on
an unseen circuit.

**`ibex_cpu`: NOT_SCORABLE** — not capturable from the acquired snapshot.

**`serv_cpu`: NOT_SCORABLE** — deliberately held sealed.

## A defect in the frozen scoring rule

The 12C-2H rule defines CONFIRMED / DIRECTIONALLY_CORRECT / FALSIFIED but has
**no label for a circuit that cannot be measured at all**. Rather than force
`ibex_cpu` into an existing label, this stage introduces **NOT_SCORABLE** and
records the omission as a defect.

Forcing FALSIFIED would blame the hypothesis for a corpus problem. Forcing
CONFIRMED would claim credit for an untested prediction. 12C-2H itself is **not
modified**.

## Acceptance outcome

**NOT MET** — one of the two required independent test circuits was uncapturable.
This is **capture infeasibility, not a measured shortfall**, and the distinction
is preserved in every frozen artifact.

The **one-shot locked evaluation was not consumed**: no model was evaluated, so it
remains available if a self-contained ibex tree is ever acquired.

## What V2.2 established

| # | claim | evidence |
|---|---|---|
| **C1** | Localization degrades with candidate-catalog scale, not with circuit change: the same HMAC circuit and faults fell from 0.5488 to 0.0286 exact-site (19.2x) when the catalog grew from one family to four | 12B-3P, 12C-2F |
| **C2** | Behavioural signature uniqueness varies ~26x across circuit classes (0.0183 CPU to 0.4771 crypto hash) and orders families identically to exact-site rate | 12C-1P, 12C-2F, 12C-2G |
| **C3** | Fault observability is bimodal, not continuous: ~99.8% in crypto datapaths versus 38-44% in CPU-class circuits, with nothing between | 12C-2A |
| **C4** | Four independent learned formulations failed to exceed the non-learning comparator on an unseen circuit; the comparator was 182x better on MRR | 12C-2C, 12C-2E, 12C-2F, 12C-2G, 12C-2K |
| **C5** | Within-collision-set localization is not identifiable under a uniform fault prior: every site carries exactly 2 faults, so posterior is 1/k and a measured 1.00x lift is the correct result | 12C-2J |
| **C6** | Residual ambiguity is dominated by structural fault equivalence (83.3-99.8% of collision sets), so candidate sets are the fault equivalence class and the localizer operates at the structural limit | 12C-2L |
| **C7** | On observable and behaviourally unique faults, localization is exact with zero false alarms across 7,849,696 transactions | 12C-1O |
| **C8** | A crypto-class prediction frozen before any seal was broken was confirmed on a genuinely unseen circuit: secworks_chacha predicted [0.20, 0.70], measured 0.3440 | 12C-2H, 12C-3B |

## What V2.2 did NOT establish

| item | why | evidence |
|---|---|---|
| independent-circuit generalization of the acceptance contract | the contract requires the per-circuit floor on BOTH test circuits; ibex_cpu could not be captured from the acquired snapshot | 12C-3C |
| any CPU-class independent test result | the only CPU-class test circuit was uncapturable and the holdout CPU remains sealed by decision | 12C-3C, 12C-3A |
| that learned models cannot help this problem in principle | four formulations were falsified; that is not a proof over all models | 12C-2L |
| that the structural-equivalence bound is formally proven per fault | adjacency is a strong structural indicator, not a per-fault equivalence proof | 12C-2L |
| behaviour under fault models other than single stuck-at | delay, bridging and transient faults were never in the catalog | 12C-1F |

## Where the project stands

The shipped localizer is **exact-signature retrieval — non-learning, zero learned
parameters** — and it operates at the structural limit proved in 12C-2L
(minimum 83.3% of collision sets
are structurally equivalent). Learned models are excluded on evidence, not
preference.

**V2.2 branch: CLOSED.**

Next phase: **Faultiva 1.0 — Hybrid VLSI Fault Intelligence** — V1 + V2 hybrid packaging, per-circuit
characterization pipeline, public release, dashboard.
