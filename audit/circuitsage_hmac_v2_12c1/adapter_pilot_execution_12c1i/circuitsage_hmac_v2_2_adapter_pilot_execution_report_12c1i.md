# CircuitSage-HMAC V2.2 bounded pilot — Stage 12C-1I

All four TRAIN/CALIBRATION adapters passed real RTL functional validation,
including independent known-answer/reference-oracle checks and byte-exact
fault-free replay.  The conditionally authorized pilot was then executed for
16384 single persistent SA0/SA1 faults and
245760 enabled transactions.

The report-only pilot all-injected detection recall is
0.57061768; the all-injected exact-site
ceiling is 0.43499756.  These are bounded
TRAIN/CALIBRATION pilot results, not evidence of independent-circuit
generalization.  A separate disposition gate must decide whether a full
campaign is justified.

No model was trained or deserialized.  INDEPENDENT_TEST remains locked,
VALIDATION remains unopened, and HOLDOUT remains sealed.  The future hybrid
release name remains **Faultiva 1.0 — Hybrid VLSI Fault Intelligence**.
