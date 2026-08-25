# ADR-008 — Experiment reproducibility requirements

- **Status:** Accepted
- **Date:** 2026-08-25

## Context

Benchmark conclusions are only trustworthy if any reported number can be
regenerated. Research code commonly rots: configs drift, datasets change,
checkpoints get updated silently. This project treats reproducibility as a hard
requirement, not a nice-to-have.

## Decision

Every experiment run must record, in a machine-readable JSON result file:

- model identifier and checkpoint/version hash;
- dataset version and generation seed;
- random seed(s);
- sample rate and preprocessing parameters;
- number of speakers, overlap ratio, SNR, duration;
- hardware (device type, GPU model where applicable);
- inference time;
- all computed metrics.

Additionally:

- Datasets regenerate byte-for-byte from seed + version (Phase 1 acceptance criterion).
- Result JSONs are committed; tables/plots regenerate from them via one command.
- Re-running a committed config must reproduce its stored metrics within stated tolerance.

## Alternatives considered

- **Notebook-based experiments** — rejected: poor diffability and automation.
- **External tracking service (MLflow/W&B)** — optional later addition; the JSON
  contract stays the source of truth so the project does not depend on a service.

## Consequences

- Slight overhead per experiment (config + metadata plumbing) — paid once in Phase 0.
- Storage grows with every run; acceptable for text JSONs.
- Enables Phase 7 aggregation and public release of complete, checkable results.
