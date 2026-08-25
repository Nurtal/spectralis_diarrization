# ADR-007 — Standard benchmark metrics

- **Status:** Accepted
- **Date:** 2026-08-25

## Context

Approaches must be compared objectively. Different papers emphasize different
metrics, making cross-paper comparison unreliable; the project therefore fixes
its own consistent metric set, applied identically to every approach.

## Decision

Primary and secondary metrics:

| Area | Primary | Secondary |
|---|---|---|
| Source separation | SI-SDR | SDR, SIR, SAR |
| Speech quality | PESQ | STOI |
| Diarization | DER | JER, missed/false-alarm speech, overlap detection accuracy |
| Speaker attribution | Identification accuracy | Embedding cosine similarity, cluster purity |
| Downstream utility | WER via ASR | CER |
| Cost | Wall-clock inference time | Hardware recorded per run |

Separation outputs are matched to references using permutation-invariant
scoring. Every reported number comes from the standardized experiment JSON
(see ADR-008).

## Alternatives considered

- **Perceptual subjective listening tests** — deferred: valuable but not
  scalable during automated benchmarking.
- **MOS prediction networks** — considered optional add-on, not a core metric.

## Consequences

- Metric libraries are pinned versions so scores are comparable over time.
- SI-SDR alone does not decide winners; attribution, DER, WER, and runtime all
  weigh in ("better waveform" must translate to downstream benefit, RQ9).
- Adding a metric later appends columns; it never invalidates prior runs since
  raw results are stored.
