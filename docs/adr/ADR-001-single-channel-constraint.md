# ADR-001 — Single-channel audio as the primary constraint

- **Status:** Accepted
- **Date:** 2026-08-25

## Context

The target recordings come from a single microphone. Multi-microphone arrays
would provide spatial cues (beamforming, time-difference of arrival) that make
separation dramatically easier, but they are not available in the intended use
cases (single-device capture, existing mono recordings, telephony).

## Decision

All approaches are benchmarked under a strict single-channel constraint: one
waveform in, no spatial information, regardless of how many speakers are mixed.

## Alternatives considered

- **Multi-channel benchmark as well** — rejected for the initial phase: it
  doubles the experimental matrix and dilutes focus; spatial methods are a
  different research problem.
- **Allowing auxiliary modalities (video)** — rejected: changes the problem
  entirely.

## Consequences

- Separation quality ceiling is lower than multi-channel; this is accepted and
  is precisely what makes the benchmark interesting.
- Approaches relying on beamforming or array geometry are out of scope.
- Results are comparable across approaches because the input constraint is
  identical for all.
