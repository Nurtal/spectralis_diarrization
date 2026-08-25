# ADR-006 — Speaker embeddings for speaker attribution

- **Status:** Accepted
- **Date:** 2026-08-25

## Context

Blind separators output unlabeled sources; assigning each output to the correct
speaker is a separate problem. Additionally, conditioning a separator on a
target-speaker representation can turn "separate N sources" into "extract this
specific speaker", which may improve attribution (RQ5).

## Decision

Speaker attribution relies on pretrained speaker embeddings (e.g., ECAPA-TDNN /
x-vector family). Embeddings serve two roles:

1. **Attribution metric** — cosine similarity between separated outputs and
   ground-truth speaker references quantifies assignment correctness.
2. **Conditioning input** — embeddings extracted from clean segments identified
   by diarization condition target-speaker extraction (Phase 6).

## Alternatives considered

- **Permutation-matching only (no embeddings)** — retained as baseline scoring,
  but cannot measure attribution confidence or support conditioning.
- **Training a joint diarize-and-separate end-to-end model** — deferred: out of
  scope before the modular pipeline is benchmarked.

## Consequences

- Depends on embedding-extractor quality; the encoder choice is recorded in
  experiment metadata like any other component.
- Embeddings extracted from mixed/noisy segments may degrade conditioning;
  robustness to degraded enrollment is an explicit Phase 6 study.
- Attribution errors become measurable independently of separation quality.
