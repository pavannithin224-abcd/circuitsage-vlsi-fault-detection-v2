# CircuitSage-HMAC V2.2 Portable Adapters and Vectors — Stage 12C-1G

Stage 12C-1G generated and froze four family-specific logical transaction
adapters and 240 deterministic TRAIN/CALIBRATION request vectors. The first 60
contracted vectors form the family-stratified pilot subset.

Every artifact passed byte-exact in-memory replay. Fault identity, fault site,
stuck polarity and supervision truth are absent. No expected responses were
generated because RTL execution is not authorized in this gate.

These adapters decode the common transaction envelope into HMAC, bounded
PicoRV32 program, AES-block, and SHA-256-block requests. They have not yet been
bound to or functionally validated against RTL. That validation and any pilot
campaign require a separate authorization stage.

Independent TEST remains locked, HOLDOUT remains sealed, and the future hybrid
release name remains **Faultiva 1.0 — Hybrid VLSI Fault Intelligence**.
