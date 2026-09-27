# Stage 12C-2H — Sealed-Circuit Prediction Freeze

**Status: PASS / FROZEN — predictions only, no capture, no sealed data read.**

## Why this stage exists and why it is time-critical

The frozen acceptance contract permits **one** evaluation on the sealed
circuits. Once capture happens, predictions can no longer be made in advance —
only descriptions after the fact. Freezing predictions now converts the sealed
capture from a measurement into a **test of a stated hypothesis**. This option
is available exactly once and expires when Stage 12C-3A begins.

## Bounding hypothesis

> Under a fixed test scheme, the all-injected exact-site rate achievable by
> behaviour-signature fault localisation is bounded by the fraction of faults
> whose observable response is unique — a property of circuit structure and
> observability, not of the model.

Three independent learned formulations were tested against this bound and none
exceeded it on an unseen circuit:

| stage | formulation | outcome |
|---|---|---|
| 12C-2C | retrieval, random negatives | no transfer |
| 12C-2E | retrieval, hard negatives | no transfer |
| 12C-2F | vs non-learning comparator | comparator 182x better |
| 12C-2G | graph reranking within collision sets | lift exactly 1.00x |

## Measured evidence (development partitions)

| family | class | signature uniqueness | detection | exact-site | meets floor |
|---|---|---|---|---|---|
| `secworks_sha256` | CRYPTO_HASH | 0.4771 | 0.9467 | 0.6385 | YES |
| `secworks_aes` | CRYPTO_BLOCK | 0.3783 | 0.9714 | 0.6359 | YES |
| `opentitan_hmac_sha256` | CRYPTO_MAC_WRAPPED | 0.1070 | 0.2676 | 0.0785 | NO |
| `picorv32_cpu` | CPU_RISCV | 0.0183 | 0.2459 | 0.0404 | NO |

## Predictions — frozen before any sealed circuit is captured

| circuit | predicted class | predicted exact-site | meets floor 0.15 | confidence |
|---|---|---|---|---|
| `ibex_cpu` | CPU_RISCV | [0.00, 0.12] | NO | HIGH |
| `secworks_chacha` | CRYPTO_ARX_STREAM | [0.20, 0.70] | YES | MEDIUM |
| `serv_cpu` | CPU_RISCV_BIT_SERIAL | [0.00, 0.12] | NO | LOW |

**Predicted overall acceptance: NOT MET**, because the per-circuit floor applies
to *both* independent test circuits and `ibex_cpu` is predicted to fall short.

### What would falsify each

- **`ibex_cpu`** — exact-site ≥ 0.15. Would show CPU-class ambiguity is not
  governing.
- **`secworks_chacha`** — exact-site < 0.15. Would show crypto-datapath
  circuits are not uniformly favourable.
- **`serv_cpu`** — exact-site ≥ 0.15. Would reframe the bounding variable from
  circuit class to datapath reuse. Confidence is deliberately **LOW** here.

## What this stage does not do

No capture, synthesis, simulation, training, or selection. Zero reads of any
INDEPENDENT_CIRCUIT_TEST or GENERALIZATION_HOLDOUT artifact — verified by
scanning the tree for sealed-family files and finding none. Independent
generalization remains **NOT ESTABLISHED**. Future hybrid brand remains
**Faultiva 1.0 — Hybrid VLSI Fault Intelligence**.

## Next gate

**Stage 12C-3A** — independent test capture. Scoring after capture must consult
the frozen scoring rule in this stage, and a falsified prediction must be
reported as prominently as a confirmed one.
