# ADR-004 — Pretrained models before custom training

- **Status:** Accepted
- **Date:** 2026-08-25

## Context

Neural separation architectures (Conv-TasNet, DPRNN, SepFormer, TF-GridNet)
all have public pretrained checkpoints. Training each from scratch would cost
GPU-weeks before producing a single comparable number. The project's principle
is "benchmark before optimizing".

## Decision

The initial benchmark (Phases 2-4) uses pretrained models as-is — no
fine-tuning. Custom training is explicitly deferred until the pretrained
baseline matrix exists and its failure modes are documented.

## Alternatives considered

- **Fine-tune every model on the synthetic benchmark set first** — rejected for
  now: confounds architecture comparison with training-budget differences.
- **Train only the most promising architecture** — deferred to post-benchmark;
  choosing it requires the benchmark results.

## Consequences

- Fast time-to-first-results; all architectures compared under equal (zero)
  training investment.
- Known limitation: pretrained models may mismatch our synthetic conditions
  (speaker counts beyond training range, noise levels); such mismatches are
  findings, not bugs, and are documented per model.
- Checkpoint provenance (source, version hash) is recorded in experiment JSONs
  per ADR-008; fine-tuning later becomes an ADR of its own.
