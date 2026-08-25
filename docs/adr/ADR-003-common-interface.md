# ADR-003 — Common interface for diarizers and separators

- **Status:** Accepted
- **Date:** 2026-08-25

## Context

The project will implement many diarization and separation approaches. Without
a shared contract, each model would need bespoke glue code, comparisons would
risk unfair harness differences, and pipelines could not swap components.

## Decision

Every implementation exposes the same interface:

```python
class Separator:
    def separate(self, audio, sample_rate, num_speakers=None): ...

class Diarizer:
    def diarize(self, audio, sample_rate): ...
```

The benchmark harness calls only these interfaces. Model-specific behavior
(checkpoints, chunking, fixed speaker counts) lives inside each implementation
or its configuration, never in the harness.

## Alternatives considered

- **Per-model harness scripts** — rejected: invites measurement inconsistencies
  and duplicated evaluation code.
- **Framework-specific abstractions (e.g., SpeechBrain/Asteroid base classes)** —
  rejected: couples the project to one framework's release cycle; thin wrappers
  over such frameworks are fine behind our own interface.

## Consequences

- Adding an approach means writing one adapter class + config.
- Unfair-comparison bugs are confined to adapters, which are small and reviewable.
- Interfaces may need extension (e.g., returning embeddings); extensions must
  keep existing implementations valid.
