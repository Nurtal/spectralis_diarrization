# Benchmark Report

## Results

| Model | si_sdr ↑ | sdr ↑ | sir ↑ | sar ↑ | der ↓ | jer ↓ |
|---|---|---|---|---|---|---|
| hybrid | -4.31 | -4.31 | 92.47 | -4.31 | - | - |
| nmf | -2.62 | -2.62 | 0.77 | 4.88 | - | - |
| sepformer | 1.42 | 1.42 | 26.93 | 1.52 | - | - |
| sepformer3 | 0.21 | 0.21 | 14.52 | 1.55 | - | - |

## Quality / cost Pareto (SI-SDR vs runtime)

_No runtime data recorded yet._

## Worst cases (SI-SDR)

| Model | Experiment | SI-SDR |
|---|---|---|
| hybrid | hybrid_3spk_ovl50_10db | -13.67 |
| hybrid | hybrid_3spk_ovl50_clean | -8.23 |
| nmf | nmf_3spk_ovl50_clean | -4.97 |
| hybrid | hybrid_3spk_ovl00_10db | -4.90 |
| hybrid | hybrid_3spk_ovl00_clean | -4.63 |

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
