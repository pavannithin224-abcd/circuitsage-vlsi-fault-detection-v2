# Stage 12C-2I — Faultiva 1.0 — Hybrid VLSI Fault Intelligence

**Detect. Locate. Verify.**

**Status: PASS / FROZEN — inference pipeline, no training, no acceptance claim.**

## What this is

A single callable pipeline that takes observed circuit responses and returns a
detection verdict, a candidate site set, and an independent structural
verification:

```
responses in -> [V2.2 DETECTION] -> [V2.2 LOCALIZATION] -> [V1 VERIFICATION]
```

No fault injection occurs in the inference path. Injection is only how a
known-faulty test case is manufactured for scoring.

## Why V1 verification is real corroboration

V1 (`HYBRID_FUSION_MLP_11D2C`) predicts detectability from **circuit structure
alone** — `post_simulation_features: 0`. It never sees a response. V2.2 reasons
purely from observed behaviour. The two evidence paths are disjoint, so
agreement between them is genuine corroboration rather than one measurement
restated.

V1 is locked to OpenTitan HMAC. On any other circuit the verifier returns
**OUT_OF_SCOPE** rather than extrapolating.

## Circuit profiles

| circuit | unobservable | signature uniqueness | exact-site | mean candidate set | V1 in scope |
|---|---|---|---|---|---|
| `opentitan_hmac_sha256` | 73.2% | 0.1070 | 0.0286 | 158.7 | YES |
| `picorv32_cpu` | 75.4% | 0.0183 | 0.0045 | 54.5 | NO |
| `secworks_aes` | 2.9% | 0.3783 | 0.3675 | 14.4 | NO |
| `secworks_sha256` | 5.3% | 0.4771 | 0.4516 | 3.8 | NO |

## Self-check

| check | result |
|---|---|
| every faulty case detected | **PASS** |
| no fault-free case reported as faulty | **PASS** |
| true site inside candidate set whenever localized | **PASS** |
| V1 verification refuses out-of-scope circuits | **PASS** |
| every result carries an unobservability notice | **PASS** |

## Example output

```
opentitan_hmac_sha256 — unique-signature fault
  DETECTION          : FAULT_DETECTED
  LOCALIZATION       : UNIQUE
  candidate count    : 1
  predicted polarity : SA0
  VERIFICATION (V1)  : VERIFIED
  ! 73% of injected faults in this circuit produce no output difference under this test scheme; a clean result does not prove the circuit is fault-free
```

## Excluded by evidence

Learned GNN reranking is **excluded**. It was falsified in 12C-2C, 12C-2E and
12C-2G, and Stage 12C-2F measured the non-learning comparator as 182x stronger.
A component is included because it is measured to work.

## Known limits

- localization requires one-time offline characterization per circuit
  (a classical ATPG fault dictionary); **zero-shot localization is not supported**
- CPU-class circuits have low signature uniqueness, so candidate sets are large
- a clean result never proves a circuit is fault-free

Independent generalization remains **NOT ESTABLISHED**. Sealed circuits
untouched. Future hybrid brand remains **Faultiva 1.0 — Hybrid VLSI Fault Intelligence**.
