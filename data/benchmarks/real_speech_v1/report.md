# Benchmark Report

## Results

| Model | si_sdr ↑ | sdr ↑ | sir ↑ | sar ↑ | der ↓ | jer ↓ |
|---|---|---|---|---|---|---|
| hybrid | -2.25 | -2.25 | 71.12 | -2.25 | - | - |
| nmf | -0.10 | -0.10 | 3.41 | 6.95 | - | - |
| pyannote_real | - | - | - | - | 0.12 | 0.18 |
| sepformer | 1.51 | 1.51 | 25.97 | 1.64 | - | - |

## Quality / cost Pareto (SI-SDR vs runtime)

_No runtime data recorded yet._

## Worst cases (SI-SDR)

| Model | Experiment | SI-SDR |
|---|---|---|
| hybrid | hybrid_oracle_sepformer_spectral_ovl00 | -11.89 |
| hybrid | hybrid_oracle_sepformer_embedding_ovl25 | -7.71 |
| hybrid | hybrid_spectral_ovl75 | -6.16 |
| hybrid | hybrid_spectral_ovl75 | -6.16 |
| hybrid | hybrid_oracle_sepformer_embedding_ovl00 | -4.94 |

## Research questions

- **RQ1** — diarization on simultaneous conversations — DER/JER per overlap config
- **RQ2** — separation vs attribution improvement — hybrid vs diarization-only runs
- **RQ3** — best pretrained architecture — benchmark table
- **RQ4** — overlap-aware selective separation trade-off — selective vs full timings
- **RQ5** — speaker conditioning vs blind attribution — spectral vs embedding runs
- **RQ6** — degradation with speaker count — speaker-count sweep (pending)
- **RQ7** — degradation with overlap ratio — overlap sweep (pending)
- **RQ8** — noise/reverberation robustness — SNR/RIR sweeps (pending)
- **RQ9** — SI-SDR vs WER correlation — requires ASR evaluation (not implemented)

_Generated from result JSONs; regenerate with `python -m benchmark report`._
