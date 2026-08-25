"""Synthetic mixture generation with exact ground truth (ADR-002).

Speakers are placed sequentially on a timeline; consecutive utterances are
shifted so they overlap by approximately the requested overlap ratio. Noise is
added at a requested SNR. Everything is driven by a single seed so datasets
regenerate identically (ADR-008).
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from benchmark.audio import load_audio, save_audio
from benchmark.interfaces import Segment

AUDIO_EXTENSIONS = {".wav", ".flac"}
OVERLAP_GRID_STEP_S = 0.01


@dataclass(frozen=True)
class ClipInfo:
    path: Path
    speaker_id: str
    duration: float
    sample_rate: int


@dataclass
class MixtureResult:
    mixture: np.ndarray
    sources: list
    segments: list
    sample_rate: int
    snr_db: float | None

    def measured_overlap_ratio(self):
        """Fraction of speaking time where two or more speakers are active."""
        if not self.segments:
            return 0.0
        step = OVERLAP_GRID_STEP_S
        end = max(s.end for s in self.segments)
        grid = np.arange(0.0, end, step) + step / 2
        active = np.zeros(len(grid), dtype=np.int32)
        for seg in self.segments:
            active[(grid >= seg.start) & (grid < seg.end)] += 1
        speaking = active >= 1
        overlapped = active >= 2
        total_speech = speaking.sum()
        if total_speech == 0:
            return 0.0
        return float(overlapped.sum() / total_speech)


def index_clips(root, exclude=None):
    """Index audio clips organized as <speaker_id>/<utterance>.wav.

    Paths under `exclude` are ignored (used to keep generated datasets out of
    the source corpus when the output lives inside the corpus directory).
    """
    root = Path(root)
    exclude = Path(exclude).resolve() if exclude else None
    clips = []
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in AUDIO_EXTENSIONS:
            continue
        if exclude is not None and exclude in path.resolve().parents:
            continue
        speaker_dir = path.parent
        if speaker_dir == root:
            continue
        audio, sr = load_audio(path)
        clips.append(
            ClipInfo(
                path=path,
                speaker_id=speaker_dir.name,
                duration=len(audio) / sr,
                sample_rate=sr,
            )
        )
    if not clips:
        raise ValueError(f"no clips found under speaker directories in {root}")
    return clips


def _pick(clips, num_speakers, rng):
    by_speaker = {}
    for c in clips:
        by_speaker.setdefault(c.speaker_id, []).append(c)
    speakers = sorted(by_speaker)
    if len(speakers) < num_speakers:
        raise ValueError(f"requested {num_speakers} speakers but corpus has only {len(speakers)}")
    chosen = rng.choice(speakers, size=num_speakers, replace=False)
    return [by_speaker[s][int(rng.integers(len(by_speaker[s])))] for s in sorted(chosen)]


def _place(durations, overlap_ratio):
    """Sequential chain placement solving exactly for the target overlap ratio.

    The union speaking time is U = sum(d) - t where t is the total overlapped
    time; the ratio is r = t / U, hence t = r * sum(d) / (1 + r). That time is
    spread over the N-1 adjacent gaps proportionally to min(d_i, d_{i+1}).
    """
    total_speech = float(sum(durations))
    if len(durations) == 1 or total_speech <= 0:
        return [0.0] * len(durations)
    target_overlap = overlap_ratio * total_speech / (1.0 + overlap_ratio)
    mins = [min(a, b) for a, b in zip(durations, durations[1:])]
    min_sum = float(sum(mins))
    starts = [0.0]
    for i in range(len(durations) - 1):
        gap = target_overlap * mins[i] / min_sum
        gap = min(gap, 0.95 * mins[i])
        starts.append(starts[-1] + durations[i] - gap)
    return starts


def synthetic_rir(sample_rate, rt60=0.4, seed=0):
    """Deterministic exponential-decay noise RIR for a target RT60.

    Simple room simulation: a strong direct tap followed by exponentially
    decaying diffuse noise. Not a physical model, but sufficient to study
    robustness of separation approaches to reverberation.
    """
    rng = np.random.default_rng(seed)
    n = max(int(rt60 * sample_rate), 2)
    t = np.arange(n) / sample_rate
    decay = np.exp(-6.9 * t / rt60)  # -60 dB at t = rt60
    ir = rng.normal(0.0, 1.0, size=n) * decay
    ir[0] += 5.0  # dominant direct path
    norm = np.linalg.norm(ir)
    return (ir / norm).astype(np.float32)


def _reverberate(audio, sample_rate, reverb, seed):
    from scipy.signal import fftconvolve

    rir = synthetic_rir(sample_rate, rt60=reverb["rt60"], seed=seed)
    # keep the full reverb tail: it bleeds past the clip end on purpose
    wet = fftconvolve(audio, rir).astype(np.float32)
    peak = np.abs(wet).max()
    if peak > 1.0:
        wet /= peak
    return wet


def generate_mixture(clips, num_speakers, duration, overlap_ratio, snr_db=None, seed=0,
                     reverb=None):
    """Generate one mixture. Returns a MixtureResult with exact ground truth."""
    rng = np.random.default_rng(seed)
    chosen = _pick(list(clips), num_speakers, rng)

    durations = []
    loaded = []
    for clip in chosen:
        audio, sr = load_audio(clip.path)
        if sr != chosen[0].sample_rate:
            raise ValueError(f"inconsistent sample rate in {clip.path}")
        loaded.append(audio.astype(np.float32))
        durations.append(len(audio) / sr)

    starts = _place(durations, overlap_ratio)
    # real-speech guard: clips can be longer than the target duration; clamp
    # every start so each segment fits entirely inside the mixture window
    starts = [
        max(0.0, min(start, duration - d)) if d < duration else max(0.0, duration - d - 1e-3)
        for start, d in zip(starts, durations)
    ]

    total_samples = int(duration * chosen[0].sample_rate)
    sources = []
    segments = []
    for clip, audio, start in zip(chosen, loaded, starts):
        if reverb is not None:
            audio = _reverberate(audio, chosen[0].sample_rate, reverb,
                                 seed=int(rng.integers(2**31)))
        buf = np.zeros(total_samples, dtype=np.float32)
        i0 = int(start * chosen[0].sample_rate)
        i1 = min(i0 + len(audio), total_samples)
        if i0 < total_samples:
            buf[i0:i1] = audio[: i1 - i0]
        np.clip(buf, -1.0, 1.0, out=buf)
        sources.append(buf)
        segments.append(
            Segment(
                start=start,
                end=min(start + clip.duration, duration),
                speaker=clip.speaker_id,
            )
        )

    mixture = np.clip(sum(sources), -1.0, 1.0)
    if snr_db is not None:
        mixture, _ = mix_with_noise(mixture, snr_db=snr_db, seed=int(rng.integers(2**31)))

    return MixtureResult(
        mixture=mixture,
        sources=sources,
        segments=segments,
        sample_rate=chosen[0].sample_rate,
        snr_db=snr_db,
    )


def mix_with_noise(speech, snr_db, seed):
    """Add white Gaussian noise at the requested SNR.

    Returns (mixture, noise); noise is None when snr_db is None.
    """
    speech = np.asarray(speech, dtype=np.float32)
    if snr_db is None:
        return speech.copy(), None
    rng = np.random.default_rng(seed)
    signal_power = float(np.mean(speech**2))
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = rng.normal(0.0, np.sqrt(noise_power), size=len(speech)).astype(np.float32)
    mixture = np.clip(speech + noise, -1.0, 1.0)
    return mixture, noise


def generate_dataset(
    clips_dir,
    output_dir,
    num_mixtures,
    durations,
    speaker_counts,
    overlap_ratios,
    snr_values,
    seed,
    version=None,
    reverb=None,
):
    """Generate a full benchmark dataset and its manifest.json.

    Parameter lists are cycled across mixtures using the seeded RNG. Returns
    the path to the written manifest.
    """
    output_dir = Path(output_dir)
    clips_root = Path(clips_dir).resolve()
    out_resolved = output_dir.resolve()
    if clips_root in out_resolved.parents or out_resolved in clips_root.parents:
        raise ValueError(
            "clips_dir and output_dir must not contain each other; "
            "generated files must never be indexed as source clips"
        )
    clips = index_clips(clips_dir, exclude=output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    version = version or f"synthetic-v1_seed{seed}"

    rng = np.random.default_rng(seed + 1)
    entries = []
    for i in range(num_mixtures):
        duration = float(rng.choice(durations))
        num_speakers = int(rng.choice(speaker_counts))
        overlap = float(rng.choice(overlap_ratios))
        snr = rng.choice(snr_values)
        snr = None if snr is None else float(snr)

        result = generate_mixture(
            clips,
            num_speakers=num_speakers,
            duration=duration,
            overlap_ratio=overlap,
            snr_db=snr,
            seed=int(rng.integers(2**31)),
            reverb=reverb,
        )

        mix_dir = output_dir / f"mix_{i:04d}"
        mix_dir.mkdir(exist_ok=True)
        save_audio(mix_dir / "mixture.wav", result.mixture, result.sample_rate)
        source_names = []
        for j, source in enumerate(result.sources):
            name = f"source_{j}.wav"
            save_audio(mix_dir / name, source, result.sample_rate)
            source_names.append(name)

        def rel(p):
            return p.relative_to(output_dir).as_posix()

        entries.append(
            {
                "id": f"mix_{i:04d}",
                "mixture": rel(mix_dir / "mixture.wav"),
                "sources": [rel(mix_dir / n) for n in source_names],
                "source_speakers": [s.speaker for s in result.segments],
                "segments": [
                    {"start": s.start, "end": s.end, "speaker": s.speaker} for s in result.segments
                ],
                "metadata": {
                    "num_speakers": num_speakers,
                    "overlap_ratio": overlap,
                    "snr_db": snr,
                    "duration_seconds": duration,
                    "reverb": reverb,
                },
            }
        )

    manifest_path = output_dir / "manifest.json"
    manifest = {
        "version": version,
        "sample_rate": _first_sr(clips),
        "seed": seed,
        "mixtures": entries,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest_path


def _first_sr(clips):
    return clips[0].sample_rate
