# CircuitSage-HMAC V2.1 final disposition — Stage 12B-1G

## Outcome

V2.1 is complete and frozen as a closed-catalog HMAC SA0/SA1 research component. The selected `V21_EXACT_SIGNATURE_SET` method correctly detects every observable fault in the locked site-test partition and always retains the true site in the returned catalog candidate set. It also avoids forcing one location when several sites have the same behavior.

This is not a 100% general fault detector. Only **50.01%** of all injected DEV_SITE_TEST faults produced an observable response change. Exact-site localization across all injected faults was **19.38%**. An observable query returned an average of **141.63** possible sites, with a maximum of **1668**.

## Supported claims

- Observable-fault detection recall: **1.00000000** within the frozen catalog.
- Fault-free false-alarm rate: **0.00000000** under the exact golden-reference gate.
- Unique-signature top-1 site accuracy: **1.00000000**.
- Observable candidate-set coverage: **1.00000000**.
- Ambiguous false-unique rate: **0.00000000**.

## Claims that are not supported

- Generalization to an unseen response signature.
- Generalization to a different circuit, synthesis flow, technology library, or fault family.
- Detection of a fault that produces no observable difference under the applied vectors.
- Exact localization when several sites are observationally equivalent.
- Production or silicon readiness.

## V2.2 direction

V2.2 may begin only with a new frozen contract. It should use response-only queries, strict unseen-vector/site/fault/circuit partitions, an open-set detector, graph-constrained candidate generation, full-catalog hard negatives, ambiguity-aware scoring, and adaptive test-vector selection. V1, V2 Core, and V2.1 remain immutable comparators.
