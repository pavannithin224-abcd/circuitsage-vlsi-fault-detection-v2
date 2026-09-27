# CircuitSage-HMAC V2.2 Transaction and Campaign Contract — Stage 12C-1F

All four frozen TRAIN/CALIBRATION netlists passed predecessor integrity replay.
This stage defines a shared request/response envelope while preserving each
circuit's legal protocol: HMAC messages, bounded RISC-V programs, AES blocks,
and SHA-256 block sequences.

Fault campaigns are restricted to single persistent SA0/SA1 faults and must
begin with bounded pilots. Fault identity, site and polarity are prohibited
from model-facing inputs. Localization is ambiguity-aware. The two independent
TEST families remain locked and the HOLDOUT family remains sealed.

No adapters, vectors, simulations, faults, datasets or models were created.
The future combined release remains **Faultiva 1.0 — Hybrid VLSI Fault Intelligence**.
