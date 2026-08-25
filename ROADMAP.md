# ROADMAP

> Detailed, actionable roadmap for the Single-Channel Speech Separation Benchmark.
> The high-level vision lives in [README.md](README.md). This document defines the
> execution order, the tasks, and the acceptance criteria for each phase.

## Status

| Phase | Milestone | Status |
|---|---|---|
| Phase 0 — Project Infrastructure | M0 | ✅ Done (2026-08-25) |
| Phase 1 — Dataset Generator | M1 | 🟨 In progress (generator done; corpus download + reverb pending) |
| Phase 2 — Diarization Baseline | M2 | 🔲 Not started |
| Phase 3 — Classical Separation | M3 | 🔲 Not started |
| Phase 4 — Neural Separation | M4 | 🔲 Not started |
| Phase 5 — Hybrid Pipelines | M5 | 🔲 Not started |
| Phase 6 — Speaker-Conditioned Separation | M6 | 🔲 Not started |
| Phase 7 — Benchmark Analysis | M7 | 🔲 Not started |

Statuses: 🔲 not started · 🟨 in progress · ✅ done · ⏸️ blocked.

---

## Execution Order

```text
M0 (infra)
  └── M1 (dataset)
        ├── M2 (diarization baseline)
        │     └── M5 (hybrid pipelines) ──┐
        ├── M3 (classical separation) ────┤
        └── M4 (neural separation) ───────┴── M7 (analysis)
              └── M6 (speaker-conditioned) ──┘
```

- **Strict order:** M0 → M1. Nothing else can start before these exist.
- **Parallelizable:** M2, M3, M4 are independent once M1 is done.
- M5 requires M2 + at least one separator (M3 or M4).
- M6 requires M2 (for speaker embeddings from clean segments) and benefits from M4.
- M7 closes the loop and consumes results from all previous phases.

Each phase ends with a review: acceptance criteria checked, ADRs updated if
decisions changed, results committed before moving on.

---

# Phase 0 — Project Infrastructure

**Goal:** establish the skeleton every later phase builds on. No models, no data —
just structure, interfaces, tooling, and reproducibility plumbing.

**Prerequisites:** none.

### Tasks

- [x] Python project scaffold (`pyproject.toml`, package under `src/`)
- [x] Environment management (locked dependencies; document Python version)
- [x] Repository layout matching the README architecture proposal
      (`diarization/`, `separation/`, `evaluation/`, `datasets/`, `experiments/`)
      — Phase 0 keeps flat modules (`audio`, `interfaces`, `datasets`, `config`,
      `results`, `registry`, `cli`); split into subpackages when they grow
- [x] Audio loading utilities (read/write WAV/FLAC, resampling to a common rate)
- [x] Common `Diarizer` base class (`diarize(audio, sample_rate) -> segments`)
- [x] Common `Separator` base class (`separate(audio, sample_rate, num_speakers=None) -> sources`)
- [x] Dataset interface (iteration over mixtures + ground truth + metadata)
- [x] Experiment configuration format (YAML configs with seed, model, dataset params)
- [x] Result storage (one machine-readable JSON per experiment run)
- [x] Basic CLI entry point (`python -m benchmark {diarize,separate,evaluate,compare}`)
- [x] Test framework + CI running lint and tests on every push

**Acceptance criteria**

- A fresh clone can install the environment with one documented command. ✅ (`uv sync`)
- `python -m benchmark --help` runs. ✅
- A dummy `Separator` and `Diarizer` implementation pass the interface tests. ✅
- An experiment config produces an empty-but-valid result JSON. ✅

---

# Phase 1 — Dataset Generator

**Goal:** produce synthetic mixtures with exact ground truth, covering the full
experimental matrix (speakers, overlap ratio, SNR, duration).

**Prerequisites:** Phase 0 complete.

### Tasks

- [ ] Select and download clean single-speaker speech corpora (e.g., LibriSpeech,
      VCTK); record dataset versions in metadata — *generator accepts any
      `<speaker_id>/<utterance>.wav` tree; actual corpus download pending*
- [x] Utterance-level index of source clips (speaker id, duration, gender where available)
- [x] Mixture generator: N speakers (2, 3, 4+), controlled overlap ratio
      (0–90 %), random or scripted placement
- [x] Ground truth export: per-speaker clean wav + RTTM-style segment annotations
- [x] SNR control: additive noise at {clean, 20, 10, 5, 0 dB}; noise corpus selection
      — *white Gaussian noise for now; dedicated noise corpus pending*
- [ ] Reverberation support (simulated RIRs), optional per configuration
- [x] Duration control: 10 s / 30 s / 60 s / 5 min / 10 min mixtures
- [x] Deterministic generation from a seed (regenerating a dataset id reproduces it byte-for-byte)
- [x] Metadata manifest per mixture (speakers, overlap ratio, SNR, sources, seed)

**Acceptance criteria**

- Generating a 2-speaker, 50 % overlap, 20 dB mixture from a fixed seed is reproducible. ✅
- Every generated mixture ships with ground-truth wavs and segment annotations loadable by the Phase 0 dataset interface. ✅
- Generation speed makes a benchmark set of ≥100 mixtures practical. ✅ (~125 mixtures/s on CPU)

---

# Phase 2 — Diarization Baseline

**Goal:** answer *who speaks when* without separation, and quantify how much
overlap degrades diarization. Establishes the baseline all pipelines improve on.

**Prerequisites:** Phase 1 complete.

### Tasks

- [ ] VAD integration (voice activity detection as a standalone module)
- [ ] pyannote.audio diarizer implementing the common `Diarizer` interface
- [ ] DER evaluation on the synthetic benchmark set
- [ ] JER evaluation
- [ ] Overlap-region detection accuracy (does it flag overlapping spans?)
- [ ] Speaker confusion analysis (missed speech / false alarm speech breakdown)
- [ ] Visualization tools (timeline plots: ground truth vs hypothesis, overlap heatmaps)

**Acceptance criteria**

- DER/JER reported for at least {2 spk × overlap 0/25/50/75 %} configurations.
- Results stored as JSON per experiment (model version, seed, hardware recorded).
- Timeline visualizations generated automatically for a sample of mixtures.

---

# Phase 3 — Classical Separation

**Goal:** implement the STFT + NMF classical baseline and measure how far
signal processing alone goes, using the same metrics as everything else.

**Prerequisites:** Phase 1 complete.

### Tasks

- [ ] STFT/iSTFT utilities (shared with later TF-domain approaches)
- [ ] NMF factorization with configurable number of components
- [ ] Component-to-source assignment strategy (clustering on spectral basis)
- [ ] SI-SDR evaluation against ground truth sources
- [ ] SDR / SIR / SAR evaluation
- [ ] PESQ / STOI where applicable
- [ ] Comparison table: NMF vs diarization-only baseline

**Acceptance criteria**

- NMF runs through the standard `Separator` interface with no special-casing.
- SI-SDR/SDR/SIR/SAR reported for {2 spk, 50 % overlap, clean} at minimum.
- Documented expectation check: quantify the time-frequency overlap limitation.

---

# Phase 4 — Neural Separation

**Goal:** integrate pretrained neural separators under identical conditions and
benchmark them head-to-head. Pretrained only, no fine-tuning yet (ADR-004).

**Prerequisites:** Phase 1 complete (can run in parallel with Phases 2-3).

### Tasks

- [ ] Conv-TasNet (pretrained) behind the common `Separator` interface
- [ ] DPRNN (pretrained) behind the same interface
- [ ] SepFormer (pretrained) behind the same interface
- [ ] TF-GridNet (pretrained) behind the same interface
- [ ] Standardized inference harness (same batching, sample rate, chunking policy for all)
- [ ] Inference-time measurement protocol (hardware logged, wall-clock per mixture)
- [ ] Unknown-speaker-count handling policy defined per model (fixed-N vs estimated)
- [ ] Benchmark across the experimental matrix: speakers × overlap × SNR × duration
- [ ] Permutation-invariant evaluation (match outputs to ground-truth speakers)

**Acceptance criteria**

- All four models evaluated on identical mixtures with identical metrics.
- SI-SDR, SDR/SIR/SAR, inference time reported per experimental cell.
- Failure modes documented (e.g., behavior at 4+ speakers, low SNR).

---

# Phase 5 — Hybrid Pipelines

**Goal:** evaluate complete diarization → separation pipelines, especially
overlap-aware processing (separate only what needs separating).

**Prerequisites:** Phase 2 complete + at least one working separator (Phase 3 or 4).

### Tasks

- [ ] Pipeline orchestration: diarization output drives separation input
- [ ] Overlap detection from diarization output
- [ ] Selective separation of overlap regions only
- [ ] Recombination of untouched non-overlap audio with separated overlap audio
- [ ] Crossfade/stitching quality checks (no audible seams at region boundaries)
- [ ] Speaker assignment: map separated sources back to diarized speaker labels
- [ ] Full-pipeline DER + SI-SDR evaluation (end-to-end, not per-module)
- [ ] Cost comparison: selective vs full-recording separation (compute saved vs quality lost)

**Acceptance criteria**

- Hybrid pipeline evaluated end-to-end on the benchmark matrix.
- Direct comparison: full separation vs overlap-aware separation
  (quality delta vs inference-time delta) answering RQ4.
- Speaker attribution accuracy measured for the assembled output.

---

# Phase 6 — Speaker-Conditioned Separation

**Goal:** test whether conditioning separation on a target-speaker embedding
improves attribution compared with blind separation.

**Prerequisites:** Phase 2 (embeddings from diarized clean segments); ideally Phase 4.

### Tasks

- [ ] Speaker embedding extraction (pretrained encoder, e.g., ECAPA/x-vector)
- [ ] Target-speaker extraction model (pretrained personal/conditioned separator)
- [ ] Embedding sourcing policy: clean segment selection from diarization output
- [ ] Speaker attribution evaluation (identification accuracy, cosine similarity, cluster purity)
- [ ] Blind vs conditioned comparison on identical mixtures
- [ ] Robustness study: embeddings extracted from noisy/reverberant segments
- [ ] Robustness study: embeddings from very short enrollment segments

**Acceptance criteria**

- Conditioned separation evaluated on the shared benchmark subset.
- Attribution metrics compared directly against blind separation (answers RQ5).
- Degradation curve documented as embedding quality/enrollment length decreases.

---

# Phase 7 — Benchmark Analysis

**Goal:** aggregate everything into publishable tables, plots, and conclusions.
No new models — analysis only.

**Prerequisites:** Phases 2-6 have produced result JSONs.

### Tasks

- [ ] Result aggregation across all experiment JSONs
- [ ] Benchmark table generation (README "Results" format, filled in)
- [ ] Plots: quality vs number of speakers (RQ6), quality vs overlap (RQ7),
      robustness curves for noise/reverberation (RQ8)
- [ ] Quality-vs-cost Pareto analysis (SI-SDR vs runtime)
- [ ] SI-SDR ↔ WER correlation analysis (RQ9), if ASR evaluation was included
- [ ] Failure case catalog (worst-K mixtures per approach, with audio inspection notes)
- [ ] Written benchmark report summarizing findings per research question
- [ ] Public release of results, configs, and generation seeds

**Acceptance criteria**

- Every research question RQ1-RQ9 has either an answer or an explicit
  "not answerable this cycle" note with reason.
- All tables/plots regenerate from committed result files via one command.
- The final report states which architecture wins on which axis — or that no
  single winner exists, with evidence.

---

## Cross-Cutting Rules (apply to every phase)

1. **Same conditions for everyone.** Any approach entering a comparison uses the
   same mixtures, metrics, and harness. No exceptions, no side channels.
2. **Machine-readable results.** Every experiment writes the standardized JSON
   (model, checkpoint, dataset version, seed, parameters, hardware, timing, metrics).
3. **Reproducibility.** Seed + config must regenerate any reported number.
4. **ADR discipline.** Significant technical choices get an ADR
   ([docs/adr/](docs/adr/)); revisiting a decision updates the ADR rather than
   silently diverging.
5. **Benchmark before optimizing.** No fine-tuning, no custom training until the
   pretrained baseline matrix (Phases 2-4) exists.

---

## Deferred (explicitly out of scope until after M7)

- Conversation grouping / reconstruction / transcription
- Fine-tuning or training custom separation models
- Multilingual and real-world recording evaluation
- Streaming / real-time operation
