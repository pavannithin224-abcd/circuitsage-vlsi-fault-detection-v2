# CircuitSage-HMAC V2.2 Generalization Contract — Stage 12C-1A

## Purpose

V2.2 is a separate research branch intended to test whether a portable
graph-and-behavior model can detect and localize persistent SA0/SA1 faults on
circuit families that are absent from training. It does not reopen or repair
the consumed V2.1 test sets.

## Frozen design

- Minimum independent circuit families: 7
- Family split: 3 TRAIN, 1 CALIBRATION, 2 locked TEST, 1 blocked HOLDOUT
- Model: circuit-normalized graph encoder + response encoder + metric retrieval/reranking
- Query fault identity: prohibited
- Ambiguous and unknown/OOD outputs: mandatory
- Online learning: disabled
- V1: optional frozen verification support only
- V2.1: immutable closed-catalog comparator

## Claim boundary

Passing calibration is not generalization. Independent-circuit generalization
may be claimed only after predictions are committed and every mandatory macro
and per-circuit gate passes on at least two locked circuit families. HOLDOUT
remains blocked even after a V2.2 pass.

## Current authorization

This stage authorizes contracts only. It does not authorize downloading RTL,
constructing datasets, synthesizing circuits, injecting faults, selecting
probes, training models, or evaluating protected partitions.

## Naming

**Faultiva 1.0 — Hybrid VLSI Fault Intelligence** remains reserved for the eventual completed V1+V2 hybrid
release. The V2.2 research component must not use that release identity yet.
