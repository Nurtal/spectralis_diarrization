# Changelog

All notable changes to the Single-Channel Speech Separation Benchmark.

## 0.1.0 — 2026-08-25

First working benchmark. All roadmap phases (M0-M7) have their core implemented.

### Infrastructure
- uv-managed Python package (`src/benchmark`) with locked dependencies;
  optional groups: `neural` (torch/speechbrain), `viz` (matplotlib)
- CLI: `diarize`, `separate`, `evaluate`, `compare`, `report`
- GitHub Actions CI (ruff + pytest); 151 tests

### Data
- Synthetic mixture generator: N speakers, controlled overlap ratio
  (analytic chain placement), SNR-controlled noise, synthetic RIR
  reverberation, byte-for-byte reproducible from a seed
- Manifest format with per-mixture ground truth and speaker alignment
- LibriSpeech test-clean prep script; first benchmark corpora committed
  as manifests/results only (audio gitignored)

### Models
- Common `Diarizer`/`Separator`/encoder interfaces (ADR-003)
- Baselines: energy VAD, blind NMF (multiplicative updates, k-means++
  component clustering, Wiener masks)
- Pretrained adapters: SepFormer/SepFormer3 (validated end-to-end on CPU),
  pyannote diarization and ECAPA embeddings (token-gated)

### Evaluation
- DER with FA/missed/confusion decomposition, JER, overlap detection P/R/F1
- SI-SDR with permutation-invariant pairing, simplified bss_eval SDR/SIR/SAR
- Hybrid selective pipeline (ADR-005): overlap detection → selective
  separation → spectral or embedding attribution → crossfaded reassembly
- Enrollment policy from solo segments + robustness curves
- Analysis: aggregation, markdown tables, Pareto frontier, worst-K catalog,
  report generator with RQ1-RQ9 status

### First findings (real speech, LibriSpeech subset)
- SepFormer leads: SI-SDR ≈ 1.5 dB / SIR ≈ 26 dB vs near-zero for blind NMF
- Neural separation degrades gracefully at 3 speakers; NMF collapses
- Hybrid selective pipelines inherit the separator's artifacts inside overlaps
- Selective processing cuts inference cost vs full-recording separation (RQ4)
