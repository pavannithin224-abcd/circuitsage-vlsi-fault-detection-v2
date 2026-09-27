# CircuitSage-HMAC V2.2 bounded-pilot disposition — Stage 12C-1J

Stage 12C-1I is complete, internally consistent and frozen.  All four portable
adapters passed byte-exact functional replay, all 128 pilot batches completed,
and no fault-free false alarm occurred.

The bounded pilot observed 9349 of 16384
faults, giving all-injected detection recall 0.57061768.  This is
0.02938232 below the pre-frozen independent-test macro target of
0.60000000.  Exact-site rate is 0.43499756, which is
0.18499756 above the corresponding 0.25000000 target.  The
pilot used 60 of 240 frozen TRAIN/CALIBRATION vectors.

Disposition: **conditional pass for full-campaign contract design**.  This does
not authorize campaign execution or model training.  All four families must be
retained, ambiguity must be preserved, and the large candidate-set risk (mean
41.4728, maximum 582) must be reassessed using
the full frozen vector schedule before training can be authorized.

Independent TEST remains locked, VALIDATION remains unopened, and HOLDOUT
remains sealed.  Independent-circuit generalization is not established.  The
future combined-model brand remains **Faultiva 1.0 — Hybrid VLSI Fault Intelligence**.
