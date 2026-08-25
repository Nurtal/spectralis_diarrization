# ADR-002 — Synthetic mixtures for controlled benchmarking

- **Status:** Accepted
- **Date:** 2026-08-25

## Context

Evaluating diarization and separation requires knowing the ground truth: which
speaker produced which signal, and when. Real recordings rarely come with
exact per-speaker references, and their overlap ratios/SNR cannot be controlled.

## Decision

The first benchmark is built on synthetic mixtures: clean single-speaker
utterances are mixed programmatically with controlled number of speakers,
overlap ratio, SNR, duration, and optional reverberation. Ground truth (clean
per-speaker signals + segment annotations) is exact by construction.

## Alternatives considered

- **Real corpora with approximate ground truth (e.g., AMI meetings)** — deferred:
  useful as a later robustness check, but approximate labels would contaminate
  the first comparisons.
- **Simulated rooms only (no real utterances)** — rejected: synthetic mixing of
  real speech keeps source acoustics realistic while controlling arrangement.

## Consequences

- Objective metrics are trustworthy; every mixture has perfect references.
- Known risk: models tuned on synthetic conditions may overfit them; a real-data
  validation pass is planned before any final conclusion.
- Dataset generation must be seeded and versioned for reproducibility (ADR-008).
