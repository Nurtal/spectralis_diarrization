# Single-Channel Speech Separation Benchmark

> **Benchmarking approaches for diarization and voice isolation from a single microphone recording containing overlapping speech.**

## Overview

This project investigates how well different approaches can **identify and isolate individual speakers from a single-channel audio recording** containing multiple simultaneous speakers.

The initial scope is deliberately limited:

```text
                    SINGLE MICROPHONE
                           |
                           v
                    Mixed audio signal
                           |
                           v
                    +--------------+
                    | Diarization  |
                    +--------------+
                           |
                           v
                    +--------------+
                    |   Isolation  |
                    | / Separation |
                    +--------------+
                           |
             +-------------+-------------+
             |             |             |
             v             v             v
         Speaker A     Speaker B     Speaker C
```

### Out of scope for the first phase

The project does **not** attempt to reconstruct complete conversations yet.

In particular, we will not initially solve:

- conversation grouping;
- conversational turn reconstruction;
- semantic conversation assignment;
- dialogue reconstruction;
- conversation-level transcription.

These will be addressed in a later phase once reliable speaker diarization and isolation have been benchmarked.

---

# Objectives

The project has four main objectives:

1. **Implement multiple approaches** to speaker diarization and speech separation.
2. Build a **common benchmark dataset and evaluation pipeline**.
3. Compare approaches using objective audio and speaker metrics.
4. Identify which architecture provides the best trade-off between:
   - separation quality;
   - speaker attribution;
   - computational cost;
   - robustness to overlapping speech.

The goal is not to assume that one architecture is best.

The goal is to **implement and benchmark them all under identical conditions**.

---

# Problem Definition

The microphone records a mixture:

\[
x(t) = \sum_{i=1}^{N}s_i(t) + n(t)
\]

where:

- `x(t)` = observed microphone signal;
- `s_i(t)` = speech signal produced by speaker `i`;
- `n(t)` = background noise.

We want to estimate:

\[
\hat{s}_1(t), \hat{s}_2(t), ..., \hat{s}_N(t)
\]

and determine which estimated source corresponds to which speaker.

The number of speakers may be unknown.

The difficult case is overlapping speech:

```text
Speaker A:  ────────────────
Speaker B:        ────────────────
Speaker C:              ───────────
                  ↑
             overlapping speech
```

Because the system uses **one microphone**, there is no spatial information available.

Therefore, the project focuses on **single-channel speech separation**.

---

# Scope

## Phase 1 — Diarization

Determine:

> **Who speaks when?**

Example:

```text
00:00 ── 00:04    Speaker A
00:04 ── 00:06    Speaker B
00:06 ── 00:09    Speaker A + Speaker B
00:09 ── 00:12    Speaker C
```

---

## Phase 2 — Voice Isolation / Separation

Determine:

> **What audio signal belongs to each speaker?**

Example:

```text
mixed.wav
    |
    +── speaker_A.wav
    +── speaker_B.wav
    +── speaker_C.wav
```

This phase is the core research problem.

---

## Phase 3 — Benchmark

Every approach must run against the same:

- input mixtures;
- speaker configurations;
- overlap ratios;
- SNR levels;
- evaluation metrics.

---

## Later phases

The following are intentionally postponed:

```text
Speaker isolation
        |
        v
Conversation grouping
        |
        v
Conversation reconstruction
        |
        v
Conversation transcription
```

---

# Approaches

The repository will implement several families of approaches.

## 1. Diarization Baseline

First establish a baseline using speaker diarization without explicit source separation.

Candidate:

- `pyannote.audio`

Pipeline:

```text
audio
  |
  v
VAD / diarization
  |
  v
speaker segments
```

This establishes how much can be achieved before attempting actual separation.

### Expected limitation

When speakers overlap:

```text
Speaker A + Speaker B
```

the system knows that two people are speaking but does not necessarily reconstruct two clean signals.

---

# 2. STFT + NMF

Classical time-frequency source separation.

```text
audio
  |
  v
STFT
  |
  v
spectrogram
  |
  v
NMF
  |
  +── component 1
  +── component 2
  +── component N
```

We will use this as a classical baseline.

### Advantages

- simple;
- interpretable;
- relatively cheap;
- useful reference point.

### Expected limitation

Human voices overlap heavily in the time-frequency domain.

---

# 3. Conv-TasNet

A neural time-domain speech separation architecture.

```text
mixed waveform
      |
      v
 Conv-TasNet
      |
      +── source 1
      +── source 2
      +── source N
```

The pretrained model will initially be used without fine-tuning.

---

# 4. DPRNN

Dual-Path RNN architecture for speech separation.

The benchmark will evaluate its performance on:

- short mixtures;
- long mixtures;
- different numbers of speakers;
- different overlap ratios.

---

# 5. SepFormer

Transformer-based speech separation.

This is expected to be one of the strongest initial pretrained baselines.

```text
mixed audio
     |
     v
 SepFormer
     |
     +── speaker 1
     +── speaker 2
     +── speaker N
```

---

# 6. TF-GridNet

Time-frequency neural separation architecture.

This approach will be benchmarked independently rather than assuming that waveform-domain approaches are always superior.

---

# 7. Overlap-Aware Separation

Instead of separating the complete recording:

```text
audio
  |
  v
diarization
  |
  +───────────────+
  |               |
no overlap      overlap
  |               |
original       separator
  |               |
  +───────+───────+
          |
          v
      output
```

Only regions containing overlapping speech are passed to the separation model.

This approach is important because:

- non-overlapping speech is already clean;
- separation can introduce artifacts;
- inference cost is reduced;
- the difficult parts of the recording receive specialized processing.

This will be a major benchmark configuration.

---

# 8. Speaker-Conditioned Separation

Instead of asking the model to blindly separate every source, provide a representation of the target speaker.

```text
                +----------------+
mixed audio --->|                |
                |  separation    |----> target speaker
speaker         |     model      |
embedding ----->|                |
                +----------------+
```

The speaker embedding can be extracted from a clean segment of the same recording.

Pipeline:

```text
mixed audio
     |
     v
diarization
     |
     v
clean speaker segment
     |
     v
speaker embedding
     |
     +------------------+
     |                  |
     v                  v
mixed audio       speaker embedding
     \                  /
      \                /
       v              v
        conditioned separation
                 |
                 v
          target speaker
```

This approach is particularly interesting because the diarization system can provide the information needed to guide the separation model.

---

# 9. Hybrid Diarization + Separation

The benchmark will also evaluate complete pipelines rather than isolated models.

Example:

```text
                 mixed audio
                     |
                     v
               diarization
                     |
              +------+------+
              |             |
         non-overlap      overlap
              |             |
              |        separation
              |             |
              +------+------+
                     |
                     v
              speaker matching
                     |
                     v
             isolated speakers
```

This is likely to be more representative of the final application than benchmarking separation models alone.

---

# Benchmark Dataset

## Synthetic mixtures

The first benchmark will use synthetic mixtures because the ground truth is known exactly.

Example:

```text
speaker_A.wav ──────┐
                    |
speaker_B.wav ──────+──> mixture.wav
                    |
speaker_C.wav ──────┘
```

Ground truth remains available:

```text
mixture.wav

ground_truth/
├── speaker_A.wav
├── speaker_B.wav
└── speaker_C.wav
```

This allows objective evaluation.

---

# Experimental Variables

Each experiment should explicitly define:

### Number of speakers

```text
2
3
4
5+
```

### Overlap ratio

```text
0%
10%
25%
50%
75%
90%
```

### Signal-to-noise ratio

```text
clean
20 dB
10 dB
5 dB
0 dB
```

### Recording duration

```text
10 s
30 s
60 s
5 min
10 min
```

### Acoustic conditions

Where possible:

- clean;
- background noise;
- reverberation;
- different speaker genders;
- different speaking rates;
- different accents/languages.

---

# Metrics

A model must not be judged solely by how good the waveform sounds.

## Source Separation

### SI-SDR

Scale-Invariant Signal-to-Distortion Ratio.

Primary metric for comparing reconstructed sources.

### SDR

Signal-to-Distortion Ratio.

### SIR

Signal-to-Interference Ratio.

Measures how much unwanted speech remains.

### SAR

Signal-to-Artifacts Ratio.

Measures artifacts introduced by the separation system.

---

# Speech Quality

Where appropriate:

- PESQ;
- STOI.

These provide complementary measures of perceptual and intelligibility quality.

---

# Diarization

For diarization:

- DER — Diarization Error Rate;
- JER — Jaccard Error Rate.

Also evaluate:

- speaker confusion;
- missed speech;
- false alarm speech;
- overlap detection accuracy.

---

# Speaker Attribution

A separated signal is not useful if it is assigned to the wrong person.

Evaluate:

- speaker identification accuracy;
- speaker embedding cosine similarity;
- cluster purity;
- speaker assignment accuracy.

---

# ASR-Based Evaluation

Use ASR as an additional downstream metric.

```text
ground truth speaker
        |
        v
ground truth transcript
        |
        +-------------+
                      |
                 WER comparison
                      |
separated audio ----> ASR
```

Measure:

- WER;
- CER where relevant.

The key question is:

> Does better waveform separation actually produce better speech recognition?

---

# Conversation-Independent Metrics

At this stage, **conversation reconstruction is not evaluated**.

We only care about:

```text
Who is speaking?
        +
Can we isolate their voice?
```

---

# Benchmark Architecture

The project should enforce a common interface for all approaches.

```python
class Separator:
    def separate(
        self,
        audio,
        sample_rate,
        num_speakers=None,
    ):
        ...
```

Diarization:

```python
class Diarizer:
    def diarize(
        self,
        audio,
        sample_rate,
    ):
        ...
```

Each implementation must expose the same interface.

Example:

```text
src/
├── diarization/
│   ├── base.py
│   ├── pyannote.py
│   └── ...
│
├── separation/
│   ├── base.py
│   ├── nmf.py
│   ├── conv_tasnet.py
│   ├── dprnn.py
│   ├── sepformer.py
│   ├── tf_gridnet.py
│   └── speaker_conditioned.py
│
├── evaluation/
│   ├── separation_metrics.py
│   ├── diarization_metrics.py
│   ├── speaker_metrics.py
│   └── asr_metrics.py
│
├── datasets/
│   ├── generation.py
│   └── loaders.py
│
└── experiments/
    └── ...
```

---

# Experiment Tracking

Every experiment should produce a machine-readable result.

Example:

```json
{
  "model": "sepformer",
  "num_speakers": 3,
  "overlap_ratio": 0.50,
  "snr_db": 10,
  "duration_seconds": 60,
  "si_sdr": 12.4,
  "sdr": 13.1,
  "sir": 17.2,
  "sar": 15.8,
  "wer": 0.18,
  "inference_time_seconds": 4.2
}
```

This allows results to be compared programmatically.

---

# Reproducibility

Every experiment must record:

- model;
- model version/checkpoint;
- dataset version;
- random seed;
- sample rate;
- number of speakers;
- overlap ratio;
- SNR;
- hardware;
- inference time;
- metrics.

The objective is to make every benchmark result reproducible.

---

# Experimental Matrix

The initial benchmark should progressively build this matrix:

| Approach | 2 spk | 3 spk | 4 spk | 50% overlap | 75% overlap | Noise |
|---|---:|---:|---:|---:|---:|---:|
| Diarization baseline | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| STFT + NMF | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Conv-TasNet | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| DPRNN | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| SepFormer | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| TF-GridNet | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Overlap-aware | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Speaker-conditioned | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

The matrix will expand as the project matures.

---

# Roadmap

> The detailed, actionable roadmap (tasks, acceptance criteria, execution order)
> lives in [ROADMAP.md](ROADMAP.md). Architectural decisions are recorded in
> [docs/adr/](docs/adr/).

## Phase 0 — Project Infrastructure

- [ ] Repository structure
- [ ] Environment management
- [ ] Audio loading utilities
- [ ] Dataset interface
- [ ] Experiment configuration
- [ ] Result storage
- [ ] Basic CLI
- [ ] ADR framework

---

## Phase 1 — Dataset Generator

- [ ] Select clean speech datasets
- [ ] Generate 2-speaker mixtures
- [ ] Generate 3-speaker mixtures
- [ ] Generate 4+ speaker mixtures
- [ ] Control overlap ratio
- [ ] Control SNR
- [ ] Add noise
- [ ] Add reverberation
- [ ] Store ground truth metadata

---

## Phase 2 — Diarization Baseline

- [ ] Implement VAD
- [ ] Integrate pyannote
- [ ] Evaluate DER
- [ ] Evaluate JER
- [ ] Evaluate overlap detection
- [ ] Produce visualization tools

---

## Phase 3 — Classical Separation

- [ ] Implement STFT
- [ ] Implement NMF
- [ ] Evaluate SI-SDR
- [ ] Evaluate SDR/SIR/SAR
- [ ] Compare against diarization baseline

---

## Phase 4 — Neural Separation

- [ ] Conv-TasNet
- [ ] DPRNN
- [ ] SepFormer
- [ ] TF-GridNet
- [ ] Standardized inference interface
- [ ] Benchmark all models

---

## Phase 5 — Hybrid Pipelines

- [ ] Diarization → separation
- [ ] Overlap detection
- [ ] Separate overlap regions only
- [ ] Recombine separated and untouched regions
- [ ] Speaker assignment

---

## Phase 6 — Speaker-Conditioned Separation

- [ ] Speaker embedding extraction
- [ ] Target speaker extraction
- [ ] Evaluate speaker attribution
- [ ] Compare with blind separation
- [ ] Test robustness to noisy embeddings

---

## Phase 7 — Benchmark Analysis

- [ ] Aggregate results
- [ ] Generate benchmark tables
- [ ] Generate plots
- [ ] Compare quality vs inference cost
- [ ] Identify failure cases
- [ ] Publish benchmark results

---

# ADRs

The project uses **Architectural Decision Records (ADRs)** for significant technical decisions.

Initial ADRs:

- `ADR-001` — Single-channel audio as the primary constraint
- `ADR-002` — Synthetic mixtures for controlled benchmarking
- `ADR-003` — Common interface for diarizers and separators
- `ADR-004` — Pretrained models before custom training
- `ADR-005` — Overlap-aware separation
- `ADR-006` — Speaker embeddings for speaker attribution
- `ADR-007` — Standard benchmark metrics
- `ADR-008` — Experiment reproducibility requirements

Each ADR should document:

```text
Context
Decision
Alternatives considered
Consequences
```

---

# CLI Concept

The project should eventually expose a simple CLI.

### Run diarization

```bash
python -m benchmark diarize \
    --input audio.wav \
    --output results/
```

### Run separation

```bash
python -m benchmark separate \
    --model sepformer \
    --input audio.wav \
    --output results/
```

### Run benchmark

```bash
python -m benchmark evaluate \
    --config configs/sepformer.yaml
```

### Compare models

```bash
python -m benchmark compare \
    --results results/
```

---

# Results

First real-speech pass (LibriSpeech test-clean subset, 2 speakers, 8 s mixtures,
clean, overlap sweep, oracle diarization for hybrid pipelines):

| Model | SI-SDR ↑ | SIR ↑ | SAR ↑ |
|---|---:|---:|---:|
| NMF (blind) | -0.11 | 3.3 | 7.0 |
| SepFormer (pretrained WSJ0) | **1.46** | **26.0** | 1.6 |
| Hybrid selective (NMF + spectral attr.) | -1.57 | 74.4 | -1.6 |

Full details: [data/benchmarks/real_speech_v1/report.md](data/benchmarks/real_speech_v1/report.md).
Early findings: pretrained neural separation generalizes across corpora far
better than blind NMF; hybrid selective pipelines inherit the separator's
artifacts inside overlaps, so separator quality dominates end-to-end quality.

No model should be declared the winner until it has been evaluated under the same conditions as the others.

---

# Research Questions

The benchmark should progressively answer:

### RQ1

How well can single-channel diarization handle simultaneous conversations?

### RQ2

How much does explicit speech separation improve speaker attribution?

### RQ3

Which pretrained separation architecture performs best for single-channel conversational speech?

### RQ4

Does separating only overlapping segments improve the quality/compute trade-off?

### RQ5

Does speaker conditioning improve speaker attribution compared with blind separation?

### RQ6

How does performance degrade as the number of simultaneous speakers increases?

### RQ7

How does performance degrade as overlap increases?

### RQ8

How robust are the approaches to noise and reverberation?

### RQ9

Does higher SI-SDR actually translate into lower ASR WER?

---

# Design Philosophy

The project follows three principles.

## 1. Benchmark before optimizing

Do not assume that a more sophisticated model is better.

Measure it.

## 2. Separate concerns

Diarization, separation, speaker attribution and evaluation should remain independent modules.

## 3. Reproducibility over demos

A spectacular result on one recording is not enough.

Every approach must be evaluated on a controlled benchmark.

---

# Future Work

Once speaker isolation is reliable, the project can expand toward:

```text
Single microphone
       |
       v
Diarization
       |
       v
Voice separation
       |
       v
Speaker attribution
       |
       v
Conversation grouping
       |
       v
Conversation reconstruction
       |
       v
Per-conversation ASR
```

Potential future research directions:

- conversation-conditioned separation;
- semantic conversation grouping;
- end-to-end conversation reconstruction;
- custom fine-tuned separation models;
- multilingual evaluation;
- real-world meeting/crowd recordings.

These are intentionally **not part of the initial benchmark**.

---

# License

TBD.

---

## Status

🚧 **Research / Benchmark — Early Development**

The project is currently focused on building the dataset, baseline implementations and evaluation framework.

### Quick start

```bash
uv sync                      # install environment (one command)
uv run python -m benchmark --help
uv run pytest tests/         # run the test suite
uv sync --group neural       # optional: torch + speechbrain (neural separators)
uv sync --group viz          # optional: matplotlib (plots)
uv run python -m benchmark report --results results --out report.md
```

The objective is to establish a rigorous benchmark before developing a custom end-to-end system.
