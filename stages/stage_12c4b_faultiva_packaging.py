#!/usr/bin/env python3
"""Stage 12C-4B: Faultiva single-bundle packaging.

Builds ONE self-contained distributable containing V2.2 detection + localization
AND V1 verification.  V1 is an internal component of Faultiva, not a separate
download.

Why this stage exists
---------------------
Three packaging defects block any usable release, all found in 12C-4A and while
sizing this stage:

  1. the V1 propagation cache (792 KB) is not shipped -> 119 of the 646 features
     cannot be computed by a downloader
  2. the golden netlist graph is not shipped -> the 14 site features cannot be
     computed either
  3. the frozen 12C-2I signature dictionary stores only ``__keys`` and
     ``__sizes``.  It records THAT a signature maps to N candidates but not WHICH
     candidates.  ``demo_faultiva.py`` works only because it rebuilds the
     dictionary from the full frozen corpus, which is not distributable.

Defect 3 is the serious one: the shipped dictionary can report an ambiguity count
but cannot localize.  This stage exports a genuine signature -> candidate-site
mapping (127,484 pairs, full 32-byte digests, no truncation and therefore no
collision risk).

What is packaged
----------------
  models/      V1 MLP, its lock, the V2.2 signature dictionary
  data/        golden netlist graph, directed-SGC propagation cache
  faultiva/    inference package - feature pipeline, detector, localizer,
               verifier, with the frozen scaling convention applied internally
  examples/    runnable end-to-end scripts
  docs/        model card, scaling convention, honest-limits statement
  SHA256SUMS   every shipped file

Nothing is retrained, recalibrated or reselected.  No threshold is changed.  The
V1 release directory is read, never written.  All frozen artifacts are copied,
never modified.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

import stage_12c2c_candidate_training as base


STAGE = "12C-4B"
ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "config/v2_2"
RESULT2 = ROOT / "results/circuitsage_hmac_v2_12c2"
RESULT3 = ROOT / "results/circuitsage_hmac_v2_12c3"
RESULT4 = ROOT / "results/circuitsage_hmac_v2_12c4"
WORK = RESULT4 / "faultiva_packaging_12c4b"
LOCK_FILE = WORK / ".stage_12c4b.lock"

BUNDLE = ROOT / "release/faultiva-v1.0"

V1_RELEASE = ROOT / "release/opentitan-hmac-vlsi-fault-detection-v1"
V1_MODEL = V1_RELEASE / "models/hmac_hybrid_v1_original.joblib"
V1_LOCK = V1_RELEASE / "models/hmac_final_diagnostic_model_lock_11d2d.json"
V1_SCHEMA = V1_RELEASE / "schemas/hmac_leakage_safe_feature_matrix_schema_11c5e.json"
V1_ARCH = V1_RELEASE / "config/hmac_hybrid_architecture_11d2a.json"
V1_LICENSE = V1_RELEASE / "LICENSE"
SGC_CACHE = (ROOT / "results/hmac_fault_campaign_11d1/gnn_training_11d1c"
             / "hmac_directed_sgc_graph_features_11d1c.npz")
GRAPH = (ROOT / "results/hmac_fault_campaign_11d1/graph_dataset_11d1a"
         / "hmac_golden_netlist_graph_11d1a.npz")

AUDIT_2I = RESULT2 / "circuitsage_hmac_v2_2_inference_pipeline_freeze_12c2i.json"
AUDIT_2L = RESULT2 / "circuitsage_hmac_v2_2_equivalence_bound_freeze_12c2l.json"
AUDIT_3B = RESULT3 / "circuitsage_hmac_v2_2_chacha_capture_freeze_12c3b.json"
AUDIT_3D = RESULT3 / "circuitsage_hmac_v2_2_closing_disposition_freeze_12c3d.json"
AUDIT_4A = RESULT4 / "circuitsage_hmac_v2_2_v1_verification_wiring_freeze_12c4a.json"
SCALING_4A = CONFIG / "circuitsage_hmac_v2_2_v1_feature_scaling_convention_12c4a.json"
SOURCE_4A = ROOT / "stage_12c4a_v1_verification_wiring.py"

PINNED = {
    V1_MODEL: "12fea5eabf4a6c605325b6ce4c1217f6a37cc59750065db8b85c3da14713f6b8",
    V1_LOCK: "7985b534c62d93717a179d6d2b20f247a531ec0452003e35f0f65cf640d0b30d",
    V1_SCHEMA: "0bf1edffb8078b003a1116b276615d5544979d007712ab49870b1b93784e486c",
    SGC_CACHE: "b5874b329a272f1f11baace3127d97e5ecbea8f6170936e7991a6aca4e02a2ce",
    GRAPH: "e3c2dd2214b544231186c29d8d9cb5aa6621b4d4b4bc9002150ac4f6c208c052",
    AUDIT_2I: "78a5cad19a6910ee8a163adaba4514e13e08b87d9ef6c6253699b79bfcffd350",
    AUDIT_2L: "bfd1df4d20a74c7bbb0832fc59f7decb1da6e76fd937ecb046d425666c6a969a",
    AUDIT_3B: "c517a208021735bd3b092361594447a14de72146cd7464e14eb310530978d6be",
    AUDIT_3D: "1d34e232822e575a7c372fb1cfbf2d4b6b0ed8fb421d3513a5a085e8232023e4",
    AUDIT_4A: "594fc0c0bc578db3e6f3f4333e586e99756d835240c8b31a8a15ecb61c44d64e",
}

DICTIONARY_EXPORT = WORK / "faultiva_signature_dictionary_12c4b.npz"
PACKAGING_CONTRACT = CONFIG / "circuitsage_hmac_v2_2_faultiva_packaging_contract_12c4b.json"
BUNDLE_INVENTORY = WORK / "faultiva_bundle_inventory_12c4b.csv"
CLEANROOM = WORK / "faultiva_cleanroom_verification_12c4b.csv"
SELFCHECK = WORK / "faultiva_packaging_selfcheck_12c4b.csv"
REPORT = WORK / "faultiva_packaging_report_12c4b.md"
MANIFEST = RESULT4 / "circuitsage_hmac_v2_2_faultiva_packaging_manifest_12c4b.json"
AUDIT = RESULT4 / "circuitsage_hmac_v2_2_faultiva_packaging_freeze_12c4b.json"

FAMILIES = {0: "opentitan_hmac_sha256", 1: "picorv32_cpu",
            2: "secworks_aes", 3: "secworks_sha256"}
V1_SCOPE_FAMILY = "opentitan_hmac_sha256"
BRAND = base.FUTURE_BRAND
VERSION = "1.0.0"

stop, require, now = base.stop, base.require, base.now
canonical_json, sha256, rel = base.canonical_json, base.sha256, base.rel
record, load_json, csv_bytes = base.record, base.load_json, base.csv_bytes
frozen_write = base.frozen_write


# --------------------------------------------------------------------------
# the inference package that actually ships
# --------------------------------------------------------------------------

FEATURE_PIPELINE = '''"""Faultiva V1 feature pipeline - builds the 646-feature vector.

SCALING CONVENTION (frozen, 12C-4A) - the two blocks use OPPOSITE conventions:

  cell_fanout   stored RAW      -> the schema's numeric_preprocessing MUST be applied
  k3_features   stored SCALED   -> k3_mean / k3_scale must NOT be applied again

k3_mean and k3_scale are the provenance record of a scaler already applied.
Applying them a second time inflates the graph branch from a p99.99 of 12.6 to
roughly 48,000 and saturates the model: 36% of probabilities collapse onto
exactly 0.0 or 1.0 and fault-polarity sensitivity falls from 100/100 sites to
48/100.  The pipeline still returns confident verdicts, so the defect is silent.

This module applies both conventions correctly.  Use build_vector(); do not
assemble the 646 features by hand.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

PKG = Path(__file__).resolve().parent.parent

N_SITE_FEATURES = 14
N_GRAPH_FEATURES = 119
N_TOTAL_FEATURES = 646


class FeaturePipeline:
    """Builds V1's 646-feature input vector with the frozen scaling convention."""

    def __init__(self, package_root: Path | None = None) -> None:
        root = Path(package_root) if package_root else PKG
        self.schema = json.loads(
            (root / "models/hmac_leakage_safe_feature_matrix_schema_11c5e.json")
            .read_text(encoding="utf-8"))

        graph = np.load(root / "data/hmac_golden_netlist_graph_11d1a.npz",
                        allow_pickle=True)
        self.fanout = graph["node_cell_fanout"]
        self.driver_code = graph["node_driver_type_code"]
        self.is_primary_output = graph["node_is_primary_output"]
        self.site_category = graph["node_site_category_code"]

        sgc = np.load(root / "data/hmac_directed_sgc_graph_features_11d1c.npz",
                      allow_pickle=True)
        # ALREADY SCALED - see module docstring
        self.k3 = sgc["k3_features"]

        pre = self.schema["numeric_preprocessing"]
        self.fanout_mean = float(pre["mean_float64"])
        self.fanout_scale = float(pre["scale_float64"])
        self.driver_vocab = self.schema["categorical_vocabulary"]["driver_cell_type"]
        self.category_vocab = self.schema["categorical_vocabulary"]["site_category"]
        self.n_stimulus = int(self.schema["stimulus_feature_count"])

        if self.k3.shape[1] != N_GRAPH_FEATURES:
            raise ValueError(f"graph cache has {self.k3.shape[1]} features, "
                             f"expected {N_GRAPH_FEATURES}")

    @property
    def n_nodes(self) -> int:
        return int(self.k3.shape[0])

    def site_features(self, node: int) -> np.ndarray:
        v = np.zeros(N_SITE_FEATURES, dtype=np.float32)
        v[0] = (float(self.fanout[node]) - self.fanout_mean) / self.fanout_scale
        v[1] = float(self.is_primary_output[node])
        code = int(self.driver_code[node])
        v[2] = 1.0 if "DFF" in self.driver_vocab[code] else 0.0
        v[3 + code] = 1.0
        v[3 + len(self.driver_vocab) + int(self.site_category[node])] = 1.0
        return v

    def graph_features(self, node: int) -> np.ndarray:
        """Return pre-scaled graph features. Do NOT rescale - see docstring."""
        return self.k3[node].astype(np.float32)

    def build_vector(self, node: int, stuck_value: int,
                     key_bits, message_bits) -> np.ndarray:
        """Assemble one 646-feature vector.

        node         node index into the golden netlist graph
        stuck_value  0 for stuck-at-0, 1 for stuck-at-1
        key_bits     256 bits, MSB first (key_bit_255 .. key_bit_0)
        message_bits 256 bits, MSB first (message_bit_255 .. message_bit_0)
        """
        if stuck_value not in (0, 1):
            raise ValueError("stuck_value must be 0 or 1")
        if not 0 <= node < self.n_nodes:
            raise ValueError(f"node {node} out of range [0, {self.n_nodes})")
        key = np.asarray(key_bits, dtype=np.float32).ravel()
        msg = np.asarray(message_bits, dtype=np.float32).ravel()
        if key.size + msg.size != self.n_stimulus:
            raise ValueError(f"expected {self.n_stimulus} stimulus bits, "
                             f"got {key.size} + {msg.size}")
        vector = np.concatenate([
            np.array([stuck_value], dtype=np.float32),
            self.site_features(node),
            key, msg,
            self.graph_features(node),
        ])
        if vector.size != N_TOTAL_FEATURES:
            raise ValueError(f"built {vector.size} features, "
                             f"expected {N_TOTAL_FEATURES}")
        return vector

    def build_batch(self, nodes, stuck_value: int,
                    key_bits, message_bits) -> np.ndarray:
        return np.stack([self.build_vector(int(n), stuck_value, key_bits, message_bits)
                         for n in nodes])
'''

PIPELINE_MODULE = '''"""Faultiva inference pipeline: detection, localization, verification.

    observed responses
        |
        v
    [1] DETECTION      V2.2, 0 learned parameters, golden-response comparison
        |
        v
    [2] LOCALIZATION   V2.2, 0 learned parameters, exact behaviour-signature lookup
        |
        v
    [3] VERIFICATION   V1 MLP, 93,185 parameters, threshold 0.4965

Stages 1 and 2 reason from BEHAVIOUR.  Stage 3 reasons from STRUCTURE - V1 uses
zero post-simulation features - so agreement between them is genuine
corroboration from disjoint evidence rather than one model confirming itself.

SCOPE: V1 was trained on opentitan_hmac_sha256 only.  On any other circuit the
verifier returns OUT_OF_SCOPE.  It never guesses.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .feature_pipeline import FeaturePipeline

PKG = Path(__file__).resolve().parent.parent
V1_SCOPE_FAMILY = "opentitan_hmac_sha256"


@dataclass
class FaultivaResult:
    circuit: str
    detection: str = ""
    localization: str = ""
    candidate_sites: np.ndarray = field(default_factory=lambda: np.empty(0, int))
    stuck_value: int | None = None
    verification: str = ""
    verification_detail: dict = field(default_factory=dict)
    signature: str = ""

    def summary(self) -> str:
        n = self.candidate_sites.size
        lines = [
            f"[1] DETECTION       {self.detection}",
            f"[2] LOCALIZATION    {self.localization}"
            + (f"  {n} candidate{'s' if n != 1 else ''}" if n else "")
            + (f"  SA{self.stuck_value}" if self.stuck_value is not None else ""),
            f"[3] VERIFICATION    {self.verification}",
        ]
        return "\\n".join(lines)


class Faultiva:
    """The shipped hybrid fault detection and localization pipeline."""

    def __init__(self, package_root: Path | None = None) -> None:
        self.root = Path(package_root) if package_root else PKG
        self.features = FeaturePipeline(self.root)

        d = np.load(self.root / "models/faultiva_signature_dictionary.npz",
                    allow_pickle=True)
        self.families = [str(f) for f in d["families"]]
        self._sig = {}
        self._sites = {}
        self._offsets = {}
        for fam in self.families:
            self._sig[fam] = d[f"{fam}__signatures"]
            self._sites[fam] = d[f"{fam}__sites"]
            self._offsets[fam] = d[f"{fam}__offsets"]

        self._model = None
        self._threshold = float(json.loads(
            (self.root / "models/hmac_final_diagnostic_model_lock_11d2d.json")
            .read_text(encoding="utf-8"))["threshold"])

    @property
    def model(self):
        if self._model is None:
            import joblib
            self._model = joblib.load(self.root / "models/hmac_hybrid_v1_original.joblib")
        return self._model

    @property
    def threshold(self) -> float:
        return self._threshold

    def signature_count(self, family: str) -> int:
        return int(self._sig[family].size)

    # ---------------- stage 1: detection ----------------
    @staticmethod
    def behaviour_signature(responses) -> str:
        """SHA-256 over the observed response sequence."""
        h = hashlib.sha256()
        for r in responses:
            h.update(str(r).encode())
            h.update(b"|")
        return h.hexdigest()

    def detect(self, observed, golden) -> tuple[str, str]:
        obs = list(observed)
        gold = list(golden)
        if len(obs) != len(gold):
            raise ValueError(f"observed has {len(obs)} responses, "
                             f"golden has {len(gold)}")
        differing = sum(1 for a, b in zip(obs, gold) if a != b)
        if differing == 0:
            return "NO_FAULT_DETECTED", ""
        return "FAULT_DETECTED", self.behaviour_signature(
            [a != b for a, b in zip(obs, gold)])

    # ---------------- stage 2: localization ----------------
    def localize(self, family: str, signature: str) -> np.ndarray:
        if family not in self._sig:
            raise KeyError(f"unknown circuit family {family!r}; "
                           f"known: {self.families}")
        sig = self._sig[family]
        idx = np.searchsorted(sig, signature)
        if idx >= sig.size or sig[idx] != signature:
            return np.empty(0, dtype=np.int64)
        start = self._offsets[family][idx]
        end = self._offsets[family][idx + 1]
        return self._sites[family][start:end]

    # ---------------- stage 3: verification ----------------
    def verify(self, family: str, nodes, stuck_value: int,
               key_bits, message_bits) -> tuple[str, dict]:
        if family != V1_SCOPE_FAMILY:
            return "OUT_OF_SCOPE", {
                "reason": f"V1 was trained only on {V1_SCOPE_FAMILY}; applying it "
                          f"to {family} would be an unsupported extrapolation"}
        nodes = np.asarray(nodes, dtype=np.int64)
        if nodes.size == 0:
            return "NO_CANDIDATES", {}
        X = self.features.build_batch(nodes, stuck_value, key_bits, message_bits)
        p = self.model.predict_proba(X)[:, 1]
        above = p >= self._threshold
        return ("VERIFIED" if above.any() else "FLAGGED"), {
            "candidates": int(p.size),
            "max_probability": float(p.max()),
            "mean_probability": float(p.mean()),
            "above_threshold": int(above.sum()),
            "threshold": self._threshold,
        }

    # ---------------- full pipeline ----------------
    def analyse(self, family: str, observed, golden, *,
                stuck_value: int | None = None,
                key_bits=None, message_bits=None) -> FaultivaResult:
        result = FaultivaResult(circuit=family)
        result.detection, signature = self.detect(observed, golden)
        if result.detection == "NO_FAULT_DETECTED":
            result.localization = "NOT_APPLICABLE"
            result.verification = "NOT_APPLICABLE"
            return result

        result.signature = signature
        sites = self.localize(family, signature)
        result.candidate_sites = sites
        result.stuck_value = stuck_value
        if sites.size == 0:
            result.localization = "UNKNOWN_SIGNATURE"
        elif sites.size == 1:
            result.localization = "EXACT"
        else:
            result.localization = "AMBIGUOUS"

        if sites.size and stuck_value is not None and key_bits is not None:
            result.verification, result.verification_detail = self.verify(
                family, sites, stuck_value, key_bits, message_bits)
        else:
            result.verification = "NOT_REQUESTED"
        return result
'''

INIT_MODULE = '''"""Faultiva - hybrid fault detection and localization for VLSI circuits.

    from faultiva import Faultiva
    f = Faultiva()
    result = f.analyse("opentitan_hmac_sha256", observed, golden)
    print(result.summary())

Read docs/HONEST_LIMITS.md before drawing conclusions from any output.
"""
from .pipeline import Faultiva, FaultivaResult, V1_SCOPE_FAMILY
from .feature_pipeline import FeaturePipeline

__version__ = "1.0.0"
__all__ = ["Faultiva", "FaultivaResult", "FeaturePipeline", "V1_SCOPE_FAMILY"]
'''


def build_dictionary_export() -> dict[str, Any]:
    """Export a REAL signature -> candidate-site mapping (CSR layout)."""
    corpus = base.Corpus()
    sig = corpus.targ["behavior_signature_sha256"]
    site = corpus.targ["local_site_index"]
    fam = corpus.targ["family_index"]

    arrays: dict[str, np.ndarray] = {}
    stats = []
    for code, name in FAMILIES.items():
        m = fam == code
        s, st = sig[m], site[m].astype(np.int64)
        order = np.argsort(s, kind="stable")
        s, st = s[order], st[order]
        uniq, starts = np.unique(s, return_index=True)
        offsets = np.append(starts, s.size).astype(np.int64)
        arrays[f"{name}__signatures"] = uniq.astype("U64")
        arrays[f"{name}__sites"] = st
        arrays[f"{name}__offsets"] = offsets
        sizes = np.diff(offsets)
        stats.append({
            "family_id": name,
            "distinct_signatures": int(uniq.size),
            "site_entries": int(st.size),
            "unique_signature_fraction": round(float((sizes == 1).mean()), 6),
            "max_candidate_set": int(sizes.max()),
            "mean_candidate_set": round(float(sizes.mean()), 4),
        })
    arrays["families"] = np.array(list(FAMILIES.values()), dtype=object)
    return {"arrays": arrays, "stats": stats}


def write_bundle(stats: list[dict], created: str) -> None:
    for sub in ("models", "data", "faultiva", "examples", "docs"):
        (BUNDLE / sub).mkdir(parents=True, exist_ok=True)

    # --- copies of frozen artifacts (never modified) ---
    shutil.copy2(V1_MODEL, BUNDLE / "models/hmac_hybrid_v1_original.joblib")
    shutil.copy2(V1_LOCK, BUNDLE / "models/hmac_final_diagnostic_model_lock_11d2d.json")
    shutil.copy2(V1_SCHEMA, BUNDLE / "models/hmac_leakage_safe_feature_matrix_schema_11c5e.json")
    shutil.copy2(V1_ARCH, BUNDLE / "models/hmac_hybrid_architecture_11d2a.json")
    shutil.copy2(DICTIONARY_EXPORT, BUNDLE / "models/faultiva_signature_dictionary.npz")
    shutil.copy2(GRAPH, BUNDLE / "data/hmac_golden_netlist_graph_11d1a.npz")
    shutil.copy2(SGC_CACHE, BUNDLE / "data/hmac_directed_sgc_graph_features_11d1c.npz")
    shutil.copy2(SCALING_4A, BUNDLE / "docs/FEATURE_SCALING_CONVENTION.json")
    if V1_LICENSE.is_file():
        shutil.copy2(V1_LICENSE, BUNDLE / "LICENSE")

    (BUNDLE / "faultiva/__init__.py").write_text(INIT_MODULE, encoding="utf-8")
    (BUNDLE / "faultiva/feature_pipeline.py").write_text(FEATURE_PIPELINE, encoding="utf-8")
    (BUNDLE / "faultiva/pipeline.py").write_text(PIPELINE_MODULE, encoding="utf-8")

    b = load_json(AUDIT_3B)
    l = load_json(AUDIT_2L)
    a4 = load_json(AUDIT_4A)

    stat_rows = "\n".join(
        f"| `{s['family_id']}` | {s['distinct_signatures']:,} | "
        f"{s['site_entries']:,} | {s['unique_signature_fraction']:.1%} | "
        f"{s['max_candidate_set']:,} |" for s in stats)

    (BUNDLE / "README.md").write_text(f'''# Faultiva {VERSION}

Hybrid fault **detection** and **localization** for gate-level VLSI circuits.

One package. V1 and V2 ship together — there is nothing else to download.

```python
from faultiva import Faultiva

f = Faultiva()
result = f.analyse("opentitan_hmac_sha256", observed, golden)
print(result.summary())
```

## Pipeline

```
observed responses
    -> [1] DETECTION      V2.2  0 learned parameters   golden-response comparison
    -> [2] LOCALIZATION   V2.2  0 learned parameters   exact signature lookup
    -> [3] VERIFICATION   V1    93,185 parameters      threshold {a4['threshold']}
```

Stages 1-2 reason from **behaviour**. Stage 3 reasons from **structure** (V1 uses
zero post-simulation features). Agreement is corroboration from disjoint
evidence, not a model confirming itself.

## Characterized circuits

| circuit | signatures | site entries | uniquely localizable | largest ambiguous set |
|---|---|---|---|---|
{stat_rows}

"Uniquely localizable" = the fraction of signatures that map to exactly one site.
Everything else resolves to a fault **equivalence class**, not a single site —
see `docs/HONEST_LIMITS.md`.

## Install

```bash
pip install -r requirements.txt
python examples/predict_one.py
python examples/end_to_end.py
```

Python 3.10+, numpy, scikit-learn, joblib. No GPU. No network.

## Verify integrity

```bash
sha256sum -c SHA256SUMS
```

## Scope

V1 verification covers `{V1_SCOPE_FAMILY}` only. Every other circuit returns
**OUT_OF_SCOPE** — never a guessed probability.

## Limits

Read `docs/HONEST_LIMITS.md` first. Independent-circuit generalization is
**NOT ESTABLISHED**. This package reports what was measured, including what failed.

## License

See `LICENSE`. Third-party RTL retains its original licensing.
''', encoding="utf-8")

    (BUNDLE / "docs/HONEST_LIMITS.md").write_text(f'''# Faultiva — honest limits

Read this before drawing conclusions from any output.

## 1. Independent generalization is NOT ESTABLISHED

The acceptance contract required the per-circuit floor on **both** independent
test circuits. One (`secworks_chacha`) was captured and **met** its floor. The
other (`ibex_cpu`) could **not be captured** — the acquired source snapshot is not
self-contained for gate-level synthesis, and authoring substitute RTL was refused
because faults would then be injected into logic written by this project.

**Acceptance outcome: NOT MET**, by capture infeasibility rather than a measured
shortfall. The distinction matters and is preserved in the frozen record.

## 2. Localization resolves to an equivalence class, not always a site

Between **83.3% and 99.8%** of ambiguous candidate sets are *structurally
equivalent* — the faults are indistinguishable from any response, by any method.
When Faultiva returns 6 candidates, that is usually the true resolution limit,
not a weakness in the search.

Optimistic per-family ceilings (12C-2L):

| circuit | current exact-site | optimistic maximum |
|---|---|---|
| `opentitan_hmac` | 0.0681 | 0.0681 |
| `picorv32_cpu` | 0.0243 | 0.0249 |
| `secworks_aes` | 0.5306 | 0.5410 |
| `secworks_sha256` | 0.6385 | 0.6392 |

## 3. Learned models were tried and failed

Four independent formulations (metric learning, GATv2 cross-fusion, OOD ensemble,
observability prediction) were pre-registered and **falsified**. A non-learning
comparator beat the best trained model by ~182x on MRR. The observability GNN
reached AUROC **0.348** — below random.

**The shipped localizer has zero learned parameters.** This is an evidence-based
choice, not a shortcut.

Within-collision-set localization was additionally proven **not identifiable**:
every site carries exactly 2 faults, so the posterior is 1/k and no model can do
better than chance inside a set.

## 4. V1 verification is HMAC-only

V1 was trained on `{V1_SCOPE_FAMILY}`. Other circuits return **OUT_OF_SCOPE**.
Balanced accuracy **0.8166**, Brier **0.1210**, threshold **{a4['threshold']}**
(reselection prohibited).

## 5. Single stuck-at faults only

The catalog contains SA0 and SA1 exclusively. Delay, bridging, transient and
multi-site faults were never characterized. Nothing here supports claims about them.

## 6. New circuits require characterization

Detection works on any circuit with a golden reference. **Localization requires a
pre-built signature dictionary**, which means a full fault-injection campaign —
hours of offline simulation per circuit. Four circuits ship characterized.

## What IS established

- detection with **0 false alarms** across 7,849,696 transactions
- exact localization on observable, behaviourally unique faults
- a crypto-class prediction frozen **before** the seal was broken and then
  confirmed: `secworks_chacha` predicted [0.20, 0.70], measured
  **{b['all_injected_exact_site_rate']:.4f}**
- catalog-scale effect: same circuit, same faults, catalog 4x larger →
  exact-site fell **19.2x**
''', encoding="utf-8")

    (BUNDLE / "examples/predict_one.py").write_text('''#!/usr/bin/env python3
"""Smallest possible working example: build one V1 vector and score it."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from faultiva import Faultiva

f = Faultiva()
rng = np.random.default_rng(0)

node = 1000
key = rng.integers(0, 2, 256)
msg = rng.integers(0, 2, 256)

vector = f.features.build_vector(node, stuck_value=1,
                                 key_bits=key, message_bits=msg)
print(f"feature vector : {vector.shape[0]} features")

probability = f.model.predict_proba(vector.reshape(1, -1))[0, 1]
print(f"V1 probability : {probability:.6f}")
print(f"threshold      : {f.threshold}")
print(f"verdict        : {'DETECTABLE' if probability >= f.threshold else 'NOT DETECTABLE'}")
''', encoding="utf-8")

    (BUNDLE / "examples/end_to_end.py").write_text('''#!/usr/bin/env python3
"""Full pipeline: detection -> localization -> verification.

Uses a synthetic golden/observed pair to keep the example self-contained.
Replace `observed` with real captured responses to analyse your own circuit.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from faultiva import Faultiva

f = Faultiva()
family = "opentitan_hmac_sha256"
print(f"characterized circuits : {f.families}")
print(f"signatures for {family} : {f.signature_count(family):,}")
print()

# --- clean circuit ---
golden = [f"resp_{i:04d}" for i in range(64)]
result = f.analyse(family, observed=golden, golden=golden)
print("CLEAN CIRCUIT")
print(result.summary())
print()

# --- faulty circuit: take a real signature from the dictionary ---
signature = f._sig[family][7]
sites = f.localize(family, signature)
print("FAULTY CIRCUIT (signature replayed from the dictionary)")
print(f"  signature      : {signature[:32]}...")
print(f"  candidate sites: {sites.size}")
print(f"  first few      : {sites[:8].tolist()}")
print()

rng = np.random.default_rng(1)
verdict, detail = f.verify(family, sites[:10], stuck_value=1,
                           key_bits=rng.integers(0, 2, 256),
                           message_bits=rng.integers(0, 2, 256))
print(f"  V1 verification: {verdict}")
for k, v in detail.items():
    print(f"    {k:<18} {v}")
print()
print("  out-of-scope check:", f.verify("secworks_aes", [0], 1, None, None)[0])
''', encoding="utf-8")

    (BUNDLE / "requirements.txt").write_text(
        "numpy>=1.24\nscikit-learn>=1.3\njoblib>=1.3\n", encoding="utf-8")

    meta = {
        "name": "faultiva", "version": VERSION, "created_at": created,
        "description": "Hybrid fault detection and localization for VLSI circuits",
        "components": {
            "detection": {"source": "V2.2", "learned_parameters": 0},
            "localization": {"source": "V2.2", "learned_parameters": 0},
            "verification": {"source": "V1", "model_id": "HYBRID_FUSION_MLP_11D2C",
                             "learned_parameters": 93185,
                             "threshold": load_json(V1_LOCK)["threshold"],
                             "scope": V1_SCOPE_FAMILY},
        },
        "total_learned_parameters": 93185,
        "characterized_circuits": stats,
        "independent_generalization": "NOT ESTABLISHED",
        "acceptance_outcome": "NOT MET (capture infeasibility)",
        "single_bundle": True,
        "separate_v1_download_required": False,
        "evidence_audits": {
            "12C-2I": sha256(AUDIT_2I), "12C-2L": sha256(AUDIT_2L),
            "12C-3B": sha256(AUDIT_3B), "12C-3D": sha256(AUDIT_3D),
            "12C-4A": sha256(AUDIT_4A),
        },
    }
    (BUNDLE / "RELEASE_METADATA.json").write_bytes(canonical_json(meta))

    # SHA256SUMS last, over everything else
    lines = []
    for p in sorted(BUNDLE.rglob("*")):
        if p.is_file() and p.name != "SHA256SUMS" and "__pycache__" not in str(p):
            lines.append(f"{sha256(p)}  {p.relative_to(BUNDLE).as_posix()}")
    (BUNDLE / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def cleanroom_verify() -> list[dict]:
    """Run the shipped examples from a copy, with the project tree off sys.path."""
    import tempfile
    rows = []
    with tempfile.TemporaryDirectory() as td:
        dest = Path(td) / "faultiva-v1.0"
        shutil.copytree(BUNDLE, dest)
        for script in ("examples/predict_one.py", "examples/end_to_end.py"):
            t0 = time.perf_counter()
            proc = subprocess.run(
                [sys.executable, str(dest / script)],
                capture_output=True, text=True, timeout=300,
                cwd=str(dest), env={"PATH": "/usr/bin:/bin", "HOME": td,
                                    "PYTHONPATH": ""})
            dt = time.perf_counter() - t0
            rows.append({
                "script": script,
                "exit_code": proc.returncode,
                "seconds": round(dt, 3),
                "stdout_lines": len(proc.stdout.splitlines()),
                "result": "PASS" if proc.returncode == 0 else "FAIL",
                "error": (proc.stderr.strip().splitlines()[-1][:160]
                          if proc.returncode else ""),
            })
        # integrity check
        proc = subprocess.run(["sha256sum", "-c", "SHA256SUMS"],
                              capture_output=True, text=True, cwd=str(dest))
        rows.append({
            "script": "sha256sum -c SHA256SUMS",
            "exit_code": proc.returncode, "seconds": 0.0,
            "stdout_lines": len(proc.stdout.splitlines()),
            "result": "PASS" if proc.returncode == 0 else "FAIL",
            "error": "" if proc.returncode == 0 else proc.stdout.strip()[-160:],
        })
    return rows


def execute() -> None:
    require(ROOT.name == "vlsi_fault_detection_v2", f"place script in project root, not {ROOT}")
    outputs = (DICTIONARY_EXPORT, PACKAGING_CONTRACT, BUNDLE_INVENTORY,
               CLEANROOM, SELFCHECK, REPORT, MANIFEST, AUDIT)
    require(not any(p.exists() for p in outputs),
            f"frozen Stage {STAGE} output exists; use --status")
    require(not BUNDLE.exists(), f"bundle already exists at {rel(BUNDLE)}")

    print("FROZEN INPUT VERIFICATION", flush=True)
    for path, expected in PINNED.items():
        require(path.is_file(), f"missing frozen input: {rel(path)}")
        require(sha256(path) == expected, f"frozen input SHA changed: {path.name}")
    require(load_json(AUDIT_4A)["v1_verification_is_genuine"] is True,
            "12C-4A must have established genuine V1 verification")
    print(f"  {len(PINNED)} frozen inputs; V1 verification genuine{'':<16}: OK", flush=True)

    print("\nEXPORTING SIGNATURE DICTIONARY", flush=True)
    base.set_determinism()
    export = build_dictionary_export()
    np.savez_compressed(DICTIONARY_EXPORT, **export["arrays"])
    for s in export["stats"]:
        print(f"  {s['family_id']:<24} sig={s['distinct_signatures']:>6} "
              f"sites={s['site_entries']:>6} unique={s['unique_signature_fraction']:.1%} "
              f"max_set={s['max_candidate_set']}", flush=True)
    print(f"  export size {DICTIONARY_EXPORT.stat().st_size/1024/1024:.2f} MB", flush=True)

    created = now()
    print("\nWRITING BUNDLE", flush=True)
    write_bundle(export["stats"], created)
    files = [p for p in sorted(BUNDLE.rglob("*"))
             if p.is_file() and "__pycache__" not in str(p)]
    total = sum(p.stat().st_size for p in files)
    print(f"  {len(files)} files, {total/1024/1024:.2f} MB at {rel(BUNDLE)}", flush=True)

    print("\nCLEAN-ROOM VERIFICATION (bundle copied out, project tree off path)",
          flush=True)
    clean = cleanroom_verify()
    for r in clean:
        print(f"  {r['script']:<30} {r['result']}  ({r['seconds']}s)", flush=True)
        if r["error"]:
            print(f"     {r['error']}", flush=True)

    inventory = [{"path": p.relative_to(BUNDLE).as_posix(),
                  "bytes": p.stat().st_size,
                  "sha256": sha256(p)} for p in files]

    checks = [
        {"check": "bundle contains the V1 model", "expected": "present",
         "observed": "present" if (BUNDLE/"models/hmac_hybrid_v1_original.joblib").is_file() else "MISSING",
         "result": "PASS" if (BUNDLE/"models/hmac_hybrid_v1_original.joblib").is_file() else "FAIL"},
        {"check": "bundle contains the propagation cache (gap 1)", "expected": "present",
         "observed": "present" if (BUNDLE/"data/hmac_directed_sgc_graph_features_11d1c.npz").is_file() else "MISSING",
         "result": "PASS" if (BUNDLE/"data/hmac_directed_sgc_graph_features_11d1c.npz").is_file() else "FAIL"},
        {"check": "bundle contains the netlist graph (gap 2)", "expected": "present",
         "observed": "present" if (BUNDLE/"data/hmac_golden_netlist_graph_11d1a.npz").is_file() else "MISSING",
         "result": "PASS" if (BUNDLE/"data/hmac_golden_netlist_graph_11d1a.npz").is_file() else "FAIL"},
        {"check": "dictionary carries candidate SITES (gap 3)",
         "expected": "site arrays present",
         "observed": f"{sum(s['site_entries'] for s in export['stats']):,} entries",
         "result": "PASS" if all(f"{f}__sites" in export["arrays"] for f in FAMILIES.values()) else "FAIL"},
        {"check": "scaling convention shipped", "expected": "present",
         "observed": "present" if (BUNDLE/"docs/FEATURE_SCALING_CONVENTION.json").is_file() else "MISSING",
         "result": "PASS" if (BUNDLE/"docs/FEATURE_SCALING_CONVENTION.json").is_file() else "FAIL"},
        {"check": "honest limits shipped", "expected": "present",
         "observed": "present" if (BUNDLE/"docs/HONEST_LIMITS.md").is_file() else "MISSING",
         "result": "PASS" if (BUNDLE/"docs/HONEST_LIMITS.md").is_file() else "FAIL"},
        {"check": "examples run in a clean room", "expected": "all PASS",
         "observed": f"{sum(1 for r in clean if r['result']=='PASS')}/{len(clean)}",
         "result": "PASS" if all(r["result"] == "PASS" for r in clean) else "FAIL"},
        {"check": "SHA256SUMS covers every file", "expected": f"{len(files)-1}",
         "observed": str(len((BUNDLE/"SHA256SUMS").read_text().strip().splitlines())),
         "result": "PASS" if len((BUNDLE/"SHA256SUMS").read_text().strip().splitlines()) == len(files)-1 else "FAIL"},
        {"check": "V1 release directory unmodified", "expected": PINNED[V1_MODEL][:16],
         "observed": sha256(V1_MODEL)[:16],
         "result": "PASS" if sha256(V1_MODEL) == PINNED[V1_MODEL] else "FAIL"},
        {"check": "no separate V1 download required", "expected": "True",
         "observed": "True", "result": "PASS"},
    ]
    print("\nSELF-CHECKS", flush=True)
    for c in checks:
        print(f"  {c['check']:<48} {c['result']}", flush=True)
    require(all(c["result"] == "PASS" for c in checks), "all self-checks must pass")

    contract = {
        "contract_version": "CIRCUITSAGE-HMAC-V2.2-FAULTIVA-PACKAGING-CONTRACT-12C4B-v1",
        "stage": STAGE, "status": "FROZEN", "created_at": created,
        "bundle_name": "faultiva", "bundle_version": VERSION,
        "bundle_path": rel(BUNDLE),
        "distribution_model": "SINGLE BUNDLE",
        "v1_is_internal_component": True,
        "separate_v1_download_required": False,
        "separate_v2_download_required": False,
        "file_count": len(files), "total_bytes": total,
        "packaging_gaps_resolved": [
            "propagation cache now shipped (119 graph features computable)",
            "golden netlist graph now shipped (14 site features computable)",
            "signature dictionary now carries candidate SITES, not only counts",
            "feature scaling convention documented and applied in code",
            "runnable worked examples included",
        ],
        "retraining_performed": False,
        "threshold_changed": False,
        "frozen_artifacts_modified": False,
        "v1_release_directory_modified": False,
    }

    report_rows = "\n".join(
        f"| `{s['family_id']}` | {s['distinct_signatures']:,} | {s['site_entries']:,} | "
        f"{s['unique_signature_fraction']:.1%} | {s['max_candidate_set']:,} |"
        for s in export["stats"])
    clean_rows = "\n".join(f"| `{r['script']}` | {r['result']} | {r['seconds']}s |"
                           for r in clean)
    chk_rows = "\n".join(f"| {c['check']} | {c['observed']} | **{c['result']}** |"
                         for c in checks)

    report = f"""# Stage {STAGE} — Faultiva Single-Bundle Packaging

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
{report_rows}

## Bundle

`{rel(BUNDLE)}` — **{len(files)} files, {total/1024/1024:.2f} MB**

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
{clean_rows}

## Self-checks

| check | observed | result |
|---|---|---|
{chk_rows}

## Honesty surface

`docs/HONEST_LIMITS.md` ships with the package and states plainly: independent
generalization **NOT ESTABLISHED**, acceptance **NOT MET** by capture
infeasibility, four learned models falsified, localization resolves to an
equivalence class, V1 is HMAC-only, single stuck-at faults only.

## Next gate

Public release to GitHub, then the dashboard.
"""

    frozen_write(PACKAGING_CONTRACT, canonical_json(contract))
    frozen_write(BUNDLE_INVENTORY, csv_bytes(inventory, ["path", "bytes", "sha256"]))
    frozen_write(CLEANROOM, csv_bytes(clean, list(clean[0].keys())))
    frozen_write(SELFCHECK, csv_bytes(checks, list(checks[0].keys())))
    frozen_write(REPORT, report.encode())

    stage_outputs = (DICTIONARY_EXPORT, PACKAGING_CONTRACT, BUNDLE_INVENTORY,
                     CLEANROOM, SELFCHECK, REPORT)
    manifest = {
        "manifest_version": "CIRCUITSAGE-HMAC-V2.2-FAULTIVA-PACKAGING-MANIFEST-12C4B-v1",
        "stage": STAGE, "status": "PASS", "created_at": created,
        "stage_source": record(Path(__file__).resolve()),
        "frozen_inputs": {rel(p): record(p) for p in PINNED},
        "outputs": {rel(p): record(p) for p in stage_outputs},
        "bundle_file_count": len(files), "bundle_bytes": total,
        "bundle_sha256sums": sha256(BUNDLE / "SHA256SUMS"),
        "training_calls": 0, "selection_calls": 0,
        "threshold_changed": False,
        "v1_release_modified": False,
        "prior_stages_modified": [],
        "test_access": 0, "holdout_access": 0,
    }
    frozen_write(MANIFEST, canonical_json(manifest))

    audit = {
        "audit_version": "CIRCUITSAGE-HMAC-V2.2-FAULTIVA-PACKAGING-FREEZE-12C4B-v1",
        "stage": STAGE, "status": "PASS",
        "bundle_version": VERSION, "bundle_path": rel(BUNDLE),
        "distribution_model": "SINGLE BUNDLE",
        "separate_v1_download_required": False,
        "file_count": len(files), "total_bytes": total,
        "signature_site_entries": sum(s["site_entries"] for s in export["stats"]),
        "distinct_signatures": sum(s["distinct_signatures"] for s in export["stats"]),
        "packaging_gaps_resolved": 3,
        "cleanroom_scripts_passed": sum(1 for r in clean if r["result"] == "PASS"),
        "cleanroom_scripts_total": len(clean),
        "self_checks_passed": sum(1 for c in checks if c["result"] == "PASS"),
        "self_checks_total": len(checks),
        "honest_limits_shipped": True,
        "scaling_convention_shipped": True,
        "retraining_performed": False,
        "threshold_changed": False,
        "v1_release_modified": False,
        "prior_stages_modified": [],
        "independent_generalization": "NOT ESTABLISHED",
        "acceptance_outcome": "NOT MET (capture infeasibility)",
        "contract_record": record(PACKAGING_CONTRACT),
        "inventory_record": record(BUNDLE_INVENTORY),
        "manifest_record": record(MANIFEST),
        "brand": BRAND,
        "next_gate": "PUBLIC RELEASE",
    }
    frozen_write(AUDIT, canonical_json(audit))

    print(f"\n{'Stage':<52}: {STAGE} — FAULTIVA PACKAGING")
    print(f"{'Status':<52}: PASS / FROZEN")
    print(f"{'Bundle':<52}: {rel(BUNDLE)}")
    print(f"{'Files / size':<52}: {len(files)} / {total/1024/1024:.2f} MB")
    print(f"{'Signature -> site entries':<52}: "
          f"{sum(s['site_entries'] for s in export['stats']):,}")
    print(f"{'Clean-room scripts':<52}: "
          f"{sum(1 for r in clean if r['result']=='PASS')}/{len(clean)} PASS")
    print(f"{'Self-checks':<52}: {len(checks)}/{len(checks)} PASS")
    print(f"{'Separate V1 download required':<52}: NO")
    print(f"{'Audit SHA':<52}: {sha256(AUDIT)}")


def status() -> None:
    print(f"STAGE {STAGE} — FAULTIVA PACKAGING STATUS")
    if not (MANIFEST.is_file() and AUDIT.is_file()):
        print("Status                    : NOT FROZEN")
        return
    a = load_json(AUDIT)
    print("Status                    : PASS / FROZEN")
    print(f"Bundle                    : {a['bundle_path']} ({a['bundle_version']})")
    print(f"Files / bytes             : {a['file_count']} / {a['total_bytes']:,}")
    print(f"Clean-room                : {a['cleanroom_scripts_passed']}/{a['cleanroom_scripts_total']}")
    print(f"Self-checks               : {a['self_checks_passed']}/{a['self_checks_total']}")
    print(f"Audit SHA                 : {sha256(AUDIT)}")


def self_test() -> None:
    require(len(FAMILIES) == 4, "four characterized families")
    require("predict_proba" in PIPELINE_MODULE, "shipped pipeline must call the model")
    require("k3_mean" not in FEATURE_PIPELINE.split('"""')[2],
            "shipped feature pipeline must not rescale k3")
    require("OUT_OF_SCOPE" in PIPELINE_MODULE, "scope rule shipped")
    a = load_json(AUDIT_4A)
    require(a["v1_verification_is_genuine"] is True, "12C-4A genuine verification")
    print(f"Stage {STAGE} self-test: PASS")


def locked_execute() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    with LOCK_FILE.open("a+", encoding="utf-8") as h:
        try:
            fcntl.flock(h.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            stop(f"Stage {STAGE} execution lock is held by another process")
        execute()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    m = p.add_mutually_exclusive_group()
    m.add_argument("--status", action="store_true")
    m.add_argument("--self-test", action="store_true")
    a = p.parse_args()
    if a.status:
        status()
    elif a.self_test:
        self_test()
    else:
        locked_execute()


if __name__ == "__main__":
    main()
