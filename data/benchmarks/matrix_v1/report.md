# Benchmark Report

## Results

| Model | si_sdr ↑ | sdr ↑ | sir ↑ | sar ↑ | pesq ↑ | stoi ↑ | der ↓ | jer ↓ |
|---|---|---|---|---|---|---|---|---|
| hybrid | -4.31 | -4.31 | 92.47 | -4.31 | - | - | - | - |
| nmf | -2.84 | -2.84 | 0.59 | 4.81 | 1.87 | 0.69 | - | - |
| sepformer | 0.98 | 0.98 | 24.11 | 1.94 | 1.36 | 0.87 | - | - |
| sepformer3 | 0.21 | 0.21 | 14.52 | 1.55 | - | - | - | - |

## Quality / cost Pareto (SI-SDR vs runtime)

_No runtime data recorded yet._

## Worst cases (SI-SDR)

| Model | Experiment | SI-SDR |
|---|---|---|
| hybrid | hybrid_3spk_ovl50_10db | -13.67 |
| hybrid | hybrid_3spk_ovl50_10db | -13.67 |
| hybrid | hybrid_3spk_ovl50_clean | -8.23 |
| hybrid | hybrid_3spk_ovl50_clean | -8.23 |
| nmf | nmf_4spk_ovl00_clean | -5.50 |

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
