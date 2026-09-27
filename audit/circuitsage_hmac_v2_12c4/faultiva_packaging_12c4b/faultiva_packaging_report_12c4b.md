# Stage 12C-4B — Faultiva Single-Bundle Packaging

**Status: PASS / FROZEN.** One package: V1 and V2 ship together.

## The three gaps this closes

| # | gap | consequence before |
|---|---|---|
| 1 | propagation cache not shipped | 119 of 646 features uncomputable |
| 2 | golden netlist graph not shipped | 14 site features uncomputable |
| 3 | **dictionary stored counts, not sites** | could say "3 candidates", not *which* 3 |

Gap 3 was the serious one. The frozen 12C-2I dictionary holds `__keys` and
`__sizes` only; `demo_faultiva.py` worked solely because it rebuilt the mapping
from the full frozen corpus, which is not distributable. **A shipped dictionary
that cannot name candidate sites cannot localize.**

Now exported as a CSR mapping with full 32-byte digests — no truncation, so no
collision risk.

| circuit | signatures | site entries | uniquely localizable | largest set |
|---|---|---|---|---|
| `opentitan_hmac_sha256` | 2,505 | 36,784 | 42.0% | 26,940 |
| `picorv32_cpu` | 471 | 19,330 | 18.5% | 14,577 |
| `secworks_aes` | 28,189 | 53,120 | 69.3% | 1,519 |
| `secworks_sha256` | 11,654 | 18,250 | 70.7% | 973 |

## Bundle

`release/faultiva-v1.0` — **19 files, 4.62 MB**

```
models/     V1 MLP + lock + schema + signature dictionary
data/       golden netlist graph + directed-SGC propagation cache
faultiva/   feature_pipeline.py  pipeline.py  __init__.py
examples/   predict_one.py  end_to_end.py
docs/       HONEST_LIMITS.md  FEATURE_SCALING_CONVENTION.json
```

The scaling convention is **applied inside `FeaturePipeline`**, so integrators
never have to know that `cell_fanout` ships raw while `k3_features` ship
pre-scaled. That asymmetry caused a silent saturation defect during development.

## Clean-room verification

Bundle copied to a temporary directory, project tree removed from `PYTHONPATH`:

| check | result | time |
|---|---|---|
| `examples/predict_one.py` | PASS | 1.565s |
| `examples/end_to_end.py` | PASS | 1.534s |
| `sha256sum -c SHA256SUMS` | PASS | 0.0s |

## Self-checks

| check | observed | result |
|---|---|---|
| bundle contains the V1 model | present | **PASS** |
| bundle contains the propagation cache (gap 1) | present | **PASS** |
| bundle contains the netlist graph (gap 2) | present | **PASS** |
| dictionary carries candidate SITES (gap 3) | 127,484 entries | **PASS** |
| scaling convention shipped | present | **PASS** |
| honest limits shipped | present | **PASS** |
| examples run in a clean room | 3/3 | **PASS** |
| SHA256SUMS covers every file | 18 | **PASS** |
| V1 release directory unmodified | 12fea5eabf4a6c60 | **PASS** |
| no separate V1 download required | True | **PASS** |

## Honesty surface

`docs/HONEST_LIMITS.md` ships with the package and states plainly: independent
generalization **NOT ESTABLISHED**, acceptance **NOT MET** by capture
infeasibility, four learned models falsified, localization resolves to an
equivalence class, V1 is HMAC-only, single stuck-at faults only.

## Next gate

Public release to GitHub, then the dashboard.
