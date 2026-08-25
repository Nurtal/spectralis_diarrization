# ADR-005 — Overlap-aware separation

- **Status:** Accepted
- **Date:** 2026-08-25

## Context

In conversational audio, much of the recording contains a single active
speaker. Running a blind separator over everything risks introducing artifacts
into already-clean regions and wastes compute. The README identifies
overlap-aware processing as a major benchmark configuration (RQ4).

## Decision

A hybrid pipeline configuration is a first-class citizen of the benchmark:
diarization detects overlapping regions; non-overlapping regions pass through
untouched; only overlaps go through separation; outputs are recombined with
proper boundary handling and mapped back to speaker labels.

## Alternatives considered

- **Always separate everything** — kept as the comparison baseline, not the
  recommended pipeline.
- **VAD-only gating (ignore silence)** — insufficient: VAD does not distinguish
  one-speaker from two-speaker regions.

## Consequences

- Requires reliable overlap detection from diarization (evaluated in Phase 2).
- Adds engineering complexity: region stitching, crossfades, speaker reassignment.
- Expected benefit is quantified directly by RQ4 (quality delta vs compute delta);
  if selective processing loses more quality than it saves compute, that is a
  legitimate benchmark outcome.
