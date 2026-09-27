#!/usr/bin/env python3
"""Faultiva 1.0 - interactive demonstration.

Run an unknown observed response through the frozen pipeline:

    responses in -> [V2.2 DETECTION] -> [V2.2 LOCALIZATION] -> [V1 VERIFICATION]

Truth is never shown to the pipeline. It is revealed only AFTER the verdict, so
what you see on screen is a genuine prediction, not a replay.

Usage
-----
    python3 demo_faultiva.py                     # one random case per circuit
    python3 demo_faultiva.py --circuit picorv32_cpu --cases 5
    python3 demo_faultiva.py --clean             # include fault-free circuits
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import stage_12c2c_candidate_training as base
import stage_12c2i_faultiva_pipeline as fv


BOLD, DIM, RESET = "\033[1m", "\033[2m", "\033[0m"
GREEN, YELLOW, RED, CYAN = "\033[32m", "\033[33m", "\033[31m", "\033[36m"


def banner() -> None:
    print()
    print(f"{BOLD}{CYAN}  ╔════════════════════════════════════════════════════════════╗{RESET}")
    print(f"{BOLD}{CYAN}  ║   FAULTIVA 1.0  —  Hybrid VLSI Fault Intelligence           ║{RESET}")
    print(f"{BOLD}{CYAN}  ║   Detect. Locate. Verify.                                   ║{RESET}")
    print(f"{BOLD}{CYAN}  ╚════════════════════════════════════════════════════════════╝{RESET}")
    print(f"{DIM}    V2.2 detection + localization  ·  V1 structural verification{RESET}")
    print(f"{DIM}    stage 12C-2I · frozen · no acceptance claim{RESET}")
    print()


def show(result, true_site, circuit_profile) -> None:
    det = result.detection
    colour = GREEN if det == "FAULT_DETECTED" else DIM
    print(f"  {BOLD}circuit{RESET}             {circuit_profile['family_id']}")
    print(f"  {BOLD}observation{RESET}         response vector captured (truth withheld)")
    print()
    print(f"  {BOLD}[1] DETECTION{RESET}       {colour}{BOLD}{det}{RESET}")
    print(f"      {DIM}{result.detection_basis}{RESET}")

    if det == "FAULT_DETECTED":
        loc_colour = GREEN if result.localization == "UNIQUE" else YELLOW
        print()
        print(f"  {BOLD}[2] LOCALIZATION{RESET}    {loc_colour}{BOLD}{result.localization}{RESET}")
        preview = result.candidate_sites[:6]
        extra = "" if result.candidate_count <= 6 else f" (+{result.candidate_count - 6} more)"
        print(f"      candidate sites   {preview}{extra}")
        print(f"      candidate count   {BOLD}{result.candidate_count}{RESET}")
        print(f"      polarity          {result.predicted_polarity}")
        print()
        vcol = GREEN if result.verification == "VERIFIED" else DIM
        print(f"  {BOLD}[3] VERIFICATION{RESET}    {vcol}{BOLD}{result.verification}{RESET}")
        print(f"      {DIM}{result.verification_detail}{RESET}")

    print()
    print(f"  {YELLOW}!{RESET} {DIM}{result.honesty_notice}{RESET}")

    if true_site is not None and det == "FAULT_DETECTED":
        inside = true_site in result.candidate_sites
        mark = f"{GREEN}CORRECT{RESET}" if inside else f"{RED}MISSED{RESET}"
        print()
        print(f"  {DIM}--- truth revealed after the verdict ---{RESET}")
        print(f"  true faulty site    {true_site}   -> {mark}")
        if inside and result.candidate_count > 1:
            print(f"  {DIM}narrowed from {circuit_profile['fault_instances']:,} possible "
                  f"sites to {result.candidate_count}{RESET}")
    print(f"  {DIM}{'─' * 62}{RESET}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--circuit", choices=base.ALL_FAMILIES, help="restrict to one circuit")
    ap.add_argument("--cases", type=int, default=1, help="faulty cases per circuit")
    ap.add_argument("--clean", action="store_true", help="also show a fault-free circuit")
    ap.add_argument("--seed", type=int, default=None, help="pick different cases")
    a = ap.parse_args()

    banner()
    print(f"{DIM}  loading frozen artifacts ...{RESET}", flush=True)
    base.set_determinism()
    corpus = base.Corpus()
    profiles = fv.build_profiles(corpus)
    pipe = fv.FaultivaPipeline(corpus, profiles)
    print(f"{DIM}  ready — {sum(len(d) for d in pipe.dictionary.values()):,} "
          f"characterized signatures across {len(pipe.dictionary)} circuits{RESET}")
    print()

    sig = corpus.targ["behavior_signature_sha256"]
    fam_ix = corpus.family_index.numpy()
    obs = corpus.observable.numpy() > 0
    rng = np.random.default_rng(a.seed if a.seed is not None else base.SEED)
    families = [a.circuit] if a.circuit else list(pipe.dictionary)

    for fam in families:
        fi = base.ALL_FAMILIES.index(fam)
        pool = np.flatnonzero((fam_ix == fi) & obs)
        picks = rng.permutation(pool)[:a.cases]
        for idx in picks:
            result = pipe.infer(fam, sig[idx], True)
            true_site = int(corpus.G[fam][2].numpy()[int(corpus.local_site[idx])])
            show(result, true_site, profiles[fam])
        if a.clean:
            show(pipe.infer(fam, b"", False), None, profiles[fam])

    print()
    print(f"{DIM}  Localization returns a candidate SET because behaviourally identical{RESET}")
    print(f"{DIM}  faults cannot be separated from responses alone. Set size is always{RESET}")
    print(f"{DIM}  reported. A clean result never proves a circuit is fault-free.{RESET}")
    print()


if __name__ == "__main__":
    main()
