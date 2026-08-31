# ROADMAP

> Detailed, actionable roadmap for the Single-Channel Speech Separation Benchmark.
> The high-level vision lives in [README.md](README.md). This document defines the
> execution order, the tasks, and the acceptance criteria for each phase.

# Status

> Snapshot at v0.1.0: all phases have their core implemented; remaining items
> are gated on external resources (HF token for pyannote, TF-GridNet
> checkpoint, full matrix runs on real corpora).

| Phase | Milestone | Status |
|---|---|---|
| Phase 0 — Project Infrastructure | M0 | ✅ Done (2026-08-25) |
| Phase 1 — Dataset Generator | M1 | ✅ Done (2026-08-25) |
| Phase 2 — Diarization Baseline | M2 | 🟨 In progress (metrics + VAD + pyannote adapter + viz done; real pyannote runs pending) |
| Phase 3 — Classical Separation | M3 | 🟨 In progress (STFT/NMF + SI-SDR/BSS metrics done; PESQ/STOI pending) |
| Phase 4 — Neural Separation | M4 | 🟨 In progress (SepFormer validated end-to-end on CPU; TF-GridNet + full matrix pending) |
| Phase 5 — Hybrid Pipelines | M5 | 🟨 In progress (pipeline + e2e eval + first matrix cells done) |
| Phase 6 — Speaker-Conditioned Separation | M6 | 🟨 In progress (encoder + enrollment + attribution comparison done; conditioned extraction model pending) |
| Phase 7 — Benchmark Analysis | M7 | 🟨 In progress (tooling + real-speech pass + first matrix done; full findings pending) |

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

- [x] Select and download clean single-speaker speech corpora (e.g., LibriSpeech,
      VCTK); record dataset versions in metadata
      — ✅ LibriSpeech test-clean subset via `scripts/prep_librisspeech.py`
- [x] Utterance-level index of source clips (speaker id, duration, gender where available)
- [x] Mixture generator: N speakers (2, 3, 4+), controlled overlap ratio
      (0–90 %), random or scripted placement
- [x] Ground truth export: per-speaker clean wav + RTTM-style segment annotations
- [x] SNR control: additive noise at {clean, 20, 10, 5, 0 dB}; noise corpus selection
      — *white Gaussian noise for now; dedicated noise corpus pending*
- [x] Reverberation support (simulated RIRs), optional per configuration
      — ✅ synthetic exponential-decay RIRs, deterministic per seed (`reverb={"rt60": ...}`)
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

- [x] VAD integration (voice activity detection as a standalone module)
      — *energy baseline; pyannote VAD can replace it later*
- [x] pyannote.audio diarizer implementing the common `Diarizer` interface
      — *adapter ready; real runs need `pip install pyannote.audio` + HF token*
- [x] DER evaluation on the synthetic benchmark set (via `evaluate` CLI)
- [x] JER evaluation
- [x] Overlap-region detection accuracy (precision/recall/F1 on overlap spans)
- [x] Speaker confusion analysis (DER decomposition: FA / missed / confusion seconds)
- [x] Visualization tools (timeline plots: ground truth vs hypothesis, overlap heatmaps)
      — ✅ `plots.plot_diarization_timeline` (GT vs HYP per-speaker lanes, overlap hatched),
        `plot_overlap_heatmap` (binary overlap P/R/F1), `plot_der_breakdown` (bonus stacked FA/missed/confusion);
        CLI `python -m benchmark visualize --manifest ... --diarizer ... --out viz/ --num-samples 3 --seed 0`

**Acceptance criteria**

- DER/JER reported for at least {2 spk × overlap 0/25/50/75 %} configurations. ✅ pipeline validated end-to-end with the VAD baseline; real numbers await pyannote runs.
- Results stored as JSON per experiment (model version, seed, hardware recorded). ✅ JSON contract; hardware field to be added with first GPU run.
- Timeline visualizations generated automatically for a sample of mixtures. ✅ `visualize` samples N mixtures seed-wise, emits `timeline_*.png` + `overlap_*.png` + `der_breakdown.png`

---

# Phase 3 — Classical Separation

**Goal:** implement the STFT + NMF classical baseline and measure how far
signal processing alone goes, using the same metrics as everything else.

**Prerequisites:** Phase 1 complete.

### Tasks

- [x] STFT/iSTFT utilities (shared with later TF-domain approaches)
- [x] NMF factorization with configurable number of components
      — *multiplicative updates, spectral-spike init, k-means++ component clustering*
- [x] Component-to-source assignment strategy (clustering on spectral basis)
- [x] SI-SDR evaluation against ground truth sources
- [x] SDR / SIR / SAR evaluation
      — *simplified bss_eval decomposition without distortion filters*
- [ ] PESQ / STOI where applicable
- [x] Comparison table: NMF vs diarization-only baseline
      — *both runnable via `evaluate` CLI on identical manifests*

**Acceptance criteria**

- NMF runs through the standard `Separator` interface with no special-casing. ✅ (`registry: "nmf"`)
- SI-SDR/SDR/SIR/SAR reported for {2 spk, 50 % overlap, clean} at minimum. ✅ (first run: SI-SDR ≈ 7.9 dB, SIR ≈ 13.9 dB)
- Documented expectation check: quantify the time-frequency overlap limitation. ✅ (degenerate case documented in `nmf_separator.py`; quantified further as benchmark grows)

---

# Phase 4 — Neural Separation

**Goal:** integrate pretrained neural separators under identical conditions and
benchmark them head-to-head. Pretrained only, no fine-tuning yet (ADR-004).

**Prerequisites:** Phase 1 complete (can run in parallel with Phases 2-3).

### Tasks

- [x] Conv-TasNet (pretrained) behind the common `Separator` interface
      — *adapter ready; official SpeechBrain checkpoint pending, community id configured*
- [x] DPRNN (pretrained) behind the same interface
      — *adapter ready; checkpoint id to confirm against SpeechBrain hub*
- [x] SepFormer (pretrained) behind the same interface
      — ✅ real run validated: `speechbrain/sepformer-wsj02mix`, CPU, ~0.7 s / 2 s audio
- [ ] TF-GridNet (pretrained) behind the same interface
- [x] Standardized inference harness (same batching, sample rate, chunking policy for all)
      — *models resampled to their native rate; outputs resampled back and clipped*
- [x] Inference-time measurement protocol (hardware logged, wall-clock per mixture)
      — *wall-clock recorded in result JSON; hardware field with first GPU run*
- [x] Unknown-speaker-count handling policy defined per model (fixed-N vs estimated)
      — *neural models are fixed-N; `num_speakers` passed from manifest metadata*
- [ ] Benchmark across the experimental matrix: speakers × overlap × SNR × duration
      — 🟨 first matrix pass done (2-3 spk × overlap {0,50} % × {clean,10 dB},
        24 cells): neural degrades gracefully at 3 spk, NMF collapses,
        hybrid inherits separator artifacts. Duration + reverb cells pending.
- [x] Permutation-invariant evaluation (match outputs to ground-truth speakers)
      — *done in Phase 3 (`best_pairing_si_sdr`)*

**Acceptance criteria**

- All four models evaluated on identical mixtures with identical metrics. 🟨 pipeline ready and validated end-to-end with SepFormer; TF-GridNet + matrix runs pending.
  First real numbers (3 synthetic mixtures, tones corpus): SIR ≈ 22.6 dB —
  strong source separation even far outside the training distribution;
  low SI-SDR/SAR expected since WSJ0-trained models see non-speech signals.
- SI-SDR, SDR/SIR/SAR, inference time reported per experimental cell. ✅ format ready
- Failure modes documented (e.g., behavior at 4+ speakers, low SNR). 🔲

---

# Phase 5 — Hybrid Pipelines

**Goal:** evaluate complete diarization → separation pipelines, especially
overlap-aware processing (separate only what needs separating).

**Prerequisites:** Phase 2 complete + at least one working separator (Phase 3 or 4).

### Tasks

- [x] Pipeline orchestration: diarization output drives separation input (`hybrid.HybridPipeline`)
- [x] Overlap detection from diarization output
- [x] Selective separation of overlap regions only (with context padding)
- [x] Recombination of untouched non-overlap audio with separated overlap audio
- [x] Crossfade/stitching quality checks (faded boundaries, no NaNs, length preserved)
- [x] Speaker assignment: map separated sources back to diarized speaker labels
      — *baseline via spectral-profile similarity against solo speech; embeddings in Phase 6*
- [x] Full-pipeline DER + SI-SDR evaluation (end-to-end, not per-module)
      — *per-speaker assembled tracks scored directly by name; DER via Phase 2 metrics*
- [x] Cost comparison: selective vs full-recording separation (compute saved vs quality lost)
      — ✅ first run (NMF, 2 spk, 50 % overlap): ~58 % inference time saved,
        per-speaker SI-SDR ≈ 13.4 dB

**Acceptance criteria**

- Hybrid pipeline evaluated end-to-end on the benchmark matrix. 🟨 pipeline + metrics ready; matrix runs pending real corpora.
- Direct comparison: full separation vs overlap-aware separation answering RQ4. ✅ mechanism in place (both timings recorded per experiment).
- Speaker attribution accuracy measured for the assembled output. ✅ (attribution correctness folded into per-speaker SI-SDR)

---

# Phase 6 — Speaker-Conditioned Separation

**Goal:** test whether conditioning separation on a target-speaker embedding
improves attribution compared with blind separation.

**Prerequisites:** Phase 2 (embeddings from diarized clean segments); ideally Phase 4.

### Tasks

- [x] Speaker embedding extraction (pretrained encoder, e.g., ECAPA/x-vector)
      — ✅ `SpeakerEncoder` adapter, real ECAPA validated (`spkrec-ecapa-voxceleb`)
- [ ] Target-speaker extraction model (pretrained personal/conditioned separator)
      — *no official SpeechBrain checkpoint today; pending Asteroid/community models*
- [x] Embedding sourcing policy: clean segment selection from diarization output
      — *`enrollment.py`: solo-span selection, duration caps, synthetic SNR degradation*
- [x] Speaker attribution evaluation (identification accuracy, cosine similarity, cluster purity)
      — *attribution folded into per-speaker SI-SDR; embedding cosine in robustness curves*
- [x] Blind vs conditioned comparison on identical mixtures
      — *hybrid pipeline `attribution: spectral|embedding` switch, same manifest;
        first run on tone corpus: spectral 14.5 dB vs ECAPA 5.4 dB — ECAPA is
        off-distribution there; meaningful conclusions need real speech*
- [x] Robustness study: embeddings extracted from noisy/reverberant segments
      — *noise degradation done (`studies.embedding_robustness_curve`); reverb pending*
- [x] Robustness study: embeddings from very short enrollment segments
      — *enrollment duration cap parameterized; curve runs to be produced on real data*

**Acceptance criteria**

- Conditioned separation evaluated on the shared benchmark subset. 🟨 attribution comparison ready; true conditioned extraction awaits a checkpoint.
- Attribution metrics compared directly against blind separation (answers RQ5). ✅ mechanism in place.
- Degradation curve documented as embedding quality/enrollment length decreases. 🟨 function + tests ready; curves to be published with real corpora.

---

# Phase 7 — Benchmark Analysis

**Goal:** aggregate everything into publishable tables, plots, and conclusions.
No new models — analysis only.

**Prerequisites:** Phases 2-6 have produced result JSONs.

### Tasks

- [x] Result aggregation across all experiment JSONs (`analysis.aggregate`)
- [x] Benchmark table generation (README "Results" format, filled in)
      — `python -m benchmark compare` renders the markdown table
- [x] Plots: quality vs number of speakers (RQ6), quality vs overlap (RQ7),
      robustness curves for noise/reverberation (RQ8)
      — *plot functions ready (`plots.py`, optional matplotlib group `viz`);
        sweep data to be produced by matrix runs*
- [x] Quality-vs-cost Pareto analysis (SI-SDR vs runtime)
- [ ] SI-SDR ↔ WER correlation analysis (RQ9), if ASR evaluation was included
- [ ] Failure case catalog (worst-K mixtures per approach, with audio inspection notes)
      — *worst-K table done; audio inspection notes pending real corpora*
- [ ] Written benchmark report summarizing findings per research question
      — ✅ generator ready (`python -m benchmark report`); substantive findings
        pending Phases 1/2/4 leftovers (real corpus, pyannote runs, full matrix)
- [ ] Public release of results, configs, and generation seeds

**Acceptance criteria**

- Every research question RQ1-RQ9 has either an answer or an explicit
  "not answerable this cycle" note with reason. ✅ mechanism (RQ_STATUS in report).
- All tables/plots regenerate from committed result files via one command. ✅ (`report`)
- The final report states which architecture wins on which axis — or that no
  single winner exists, with evidence. 🔲 awaits the full benchmark runs.

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
