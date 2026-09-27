# CircuitSage-HMAC V2.1 observability-ceiling review

## Disposition

Stage 12B-2D screened all 512 deterministic key/message vectors on 2,048
REPAIR_TRAIN pilot fault instances. Exactly 1027 faults produced
an externally visible digest, timeout, or completion-latency deviation. The
remaining 1021 faults were indistinguishable from golden behavior
under every candidate vector.

The complete-pool detection ceiling is therefore 0.50146484, below
the frozen 0.70000000 target. More ranking, retraining, or choosing a
different subset of the same vectors cannot make an externally invisible fault
detectable. The full repair campaign remains blocked and the target is not
relaxed.

## What did improve

The selected 96-vector subset preserved all observable pilot faults, increased
the all-injected exact-site rate to 0.41748047, and reduced the
mean/maximum observable candidate-site counts to 6.0925
and 69. These are closed-pilot results, not independent
generalization evidence.

## Next experiment

The authorized next step is a bounded feasibility study using fixed,
fault-identity-independent measurements. External control timing is evaluated
first. Simulation-only architectural-state sketches may then be tested on the
same REPAIR_TRAIN pilot sites. Any internal measurement would require DFT,
trace hardware, or embedded monitors before it could be used on silicon.

Injection controls—including the selector, forced value, enable signal, and
selected raw-net monitor—are forbidden as model inputs. They exist solely to
create and score simulated faults. No claim of production or independent-chip
readiness is made.
