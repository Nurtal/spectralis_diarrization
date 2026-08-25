# Benchmark Report

## Results

| Model | si_sdr ↑ | sdr ↑ | sir ↑ | sar ↑ | der ↓ | jer ↓ |
|---|---|---|---|---|---|---|
| hybrid | -1.57 | -1.57 | 74.40 | -1.57 | - | - |
| nmf | -0.11 | -0.11 | 3.32 | 7.00 | - | - |
| sepformer | 1.46 | 1.46 | 26.03 | 1.59 | - | - |

## Quality / cost Pareto (SI-SDR vs runtime)

_No runtime data recorded yet._

## Worst cases (SI-SDR)

| Model | Experiment | SI-SDR |
|---|---|---|
| hybrid | hybrid_spectral_ovl75 | -6.16 |
| hybrid | hybrid_embedding_ovl00 | -2.06 |
| hybrid | hybrid_embedding_ovl25 | -1.78 |
| hybrid | hybrid_embedding_ovl75 | -1.59 |
| hybrid | hybrid_embedding_ovl50 | -0.99 |

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
