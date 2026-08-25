import json

import numpy as np
import pytest

from benchmark.audio import save_audio
from benchmark.datasets import ManifestDataset
from benchmark.generator import (
    generate_dataset,
    generate_mixture,
    index_clips,
    mix_with_noise,
)

SR = 8000


@pytest.fixture
def clips_dir(tmp_path):
    """Three speakers, two clips each, distinct amplitudes to tell them apart."""
    rng = np.random.default_rng(7)
    clips_root = tmp_path / "clips_root"
    for spk in ("spk1", "spk2", "spk3"):
        d = clips_root / spk
        d.mkdir(parents=True)
        for i in range(2):
            dur = 1.0 + 0.5 * i + {"spk1": 0.0, "spk2": 0.25, "spk3": 0.5}[spk]
            n = int(dur * SR)
            tone = (0.2 * np.sin(2 * np.pi * (200 + 100 * i) * np.arange(n) / SR)).astype(
                np.float32
            )
            save_audio(d / f"utt{i}.wav", tone, SR)
            _ = rng  # keep rng unused-but-scoped
    return tmp_path / "clips_root"


class TestIndexClips:
    def test_finds_clips_with_speaker_from_parent_dir(self, clips_dir):
        clips = index_clips(clips_dir)
        assert len(clips) == 6
        speakers = {c.speaker_id for c in clips}
        assert speakers == {"spk1", "spk2", "spk3"}

    def test_durations_measured(self, clips_dir):
        clips = index_clips(clips_dir)
        utt = next(c for c in clips if c.speaker_id == "spk1" and c.path.name == "utt0.wav")
        assert utt.duration == pytest.approx(1.0)
        assert utt.sample_rate == SR

    def test_ignores_wav_outside_speaker_dirs(self, clips_dir):
        save_audio(clips_dir / "loose.wav", np.zeros(100, dtype=np.float32), SR)
        clips = index_clips(clips_dir)
        assert all(c.path.parent != clips_dir for c in clips)

    def test_empty_dir_raises(self, tmp_path):
        with pytest.raises(ValueError, match="no clips"):
            index_clips(tmp_path)


class TestGenerateMixture:
    def test_returns_expected_shapes_and_count(self, clips_dir):
        clips = index_clips(clips_dir)
        result = generate_mixture(clips, num_speakers=2, duration=4.0, overlap_ratio=0.5, seed=0)
        assert result.sample_rate == SR
        assert len(result.sources) == 2
        total = int(4.0 * SR)
        assert len(result.mixture) == total
        assert all(len(s) == total for s in result.sources)

    def test_segments_within_duration_and_distinct_speakers(self, clips_dir):
        clips = index_clips(clips_dir)
        result = generate_mixture(clips, num_speakers=3, duration=10.0, overlap_ratio=0.0, seed=1)
        speakers = {seg.speaker for seg in result.segments}
        assert len(speakers) == 3
        for seg in result.segments:
            assert 0.0 <= seg.start < seg.end <= 10.0 + 1e-6

    def test_clean_mixture_is_sum_of_sources(self, clips_dir):
        clips = index_clips(clips_dir)
        result = generate_mixture(
            clips, num_speakers=2, duration=4.0, overlap_ratio=0.3, snr_db=None, seed=2
        )
        expected = sum(result.sources)
        # sources are clipped to [-1, 1] before summing; mixture too
        np.testing.assert_allclose(result.mixture, np.clip(expected, -1.0, 1.0), atol=1e-6)

    @pytest.mark.parametrize("target", [0.0, 0.25, 0.5])
    def test_overlap_ratio_respected(self, clips_dir, target):
        clips = index_clips(clips_dir)
        result = generate_mixture(
            clips, num_speakers=2, duration=30.0, overlap_ratio=target, seed=3
        )
        measured = result.measured_overlap_ratio()
        assert measured == pytest.approx(target, abs=0.15)


class TestMixWithNoise:
    def test_zero_snr_means_no_noise(self):
        speech = np.full(16000, 0.5, dtype=np.float32)
        mixed, noise = mix_with_noise(speech, snr_db=None, seed=0)
        np.testing.assert_array_equal(mixed, speech)
        assert noise is None

    def test_noise_power_matches_requested_snr(self):
        rng_check = np.random.default_rng(11)
        speech = rng_check.uniform(-0.3, 0.3, size=48000).astype(np.float32)
        snr_db = 20.0
        mixed, noise = mix_with_noise(speech, snr_db=snr_db, seed=5)
        p_sig = float(np.mean(speech**2))
        p_noise = float(np.mean(noise**2))
        measured_snr = 10 * np.log10(p_sig / p_noise)
        assert measured_snr == pytest.approx(snr_db, abs=0.1)
        np.testing.assert_allclose(mixed, speech + noise, atol=1e-7)

    def test_deterministic_given_seed(self):
        speech = np.full(4000, 0.1, dtype=np.float32)
        a, _ = mix_with_noise(speech, snr_db=10, seed=9)
        b, _ = mix_with_noise(speech, snr_db=10, seed=9)
        np.testing.assert_array_equal(a, b)


class TestDeterminism:
    def test_same_seed_same_output(self, clips_dir):
        clips = index_clips(clips_dir)
        r1 = generate_mixture(
            clips, num_speakers=2, duration=4.0, overlap_ratio=0.5, snr_db=20, seed=42
        )
        r2 = generate_mixture(
            clips, num_speakers=2, duration=4.0, overlap_ratio=0.5, snr_db=20, seed=42
        )
        np.testing.assert_array_equal(r1.mixture, r2.mixture)
        assert r1.segments == r2.segments

    def test_different_seed_different_placement(self, clips_dir):
        clips = index_clips(clips_dir)
        r1 = generate_mixture(clips, num_speakers=3, duration=8.0, overlap_ratio=0.5, seed=1)
        r2 = generate_mixture(clips, num_speakers=3, duration=8.0, overlap_ratio=0.5, seed=2)
        assert r1.segments != r2.segments


class TestGenerateDataset:
    def test_writes_files_loadable_by_manifest_dataset(self, clips_dir, tmp_path):
        out = tmp_path / "bench_v1"
        manifest_path = generate_dataset(
            clips_dir,
            output_dir=out,
            num_mixtures=3,
            durations=[4.0],
            speaker_counts=[2],
            overlap_ratios=[0.5],
            snr_values=[20],
            seed=123,
        )

        ds = ManifestDataset(manifest_path)
        assert len(ds) == 3
        m = ds[0]
        assert m.metadata["num_speakers"] == 2
        assert m.metadata["overlap_ratio"] == 0.5
        assert m.metadata["snr_db"] == 20
        audio, sr = m.load_mixture()
        assert sr == SR and len(audio) > 0
        assert m.num_sources == 2
        src_audio, _ = m.load_source(0)
        assert len(src_audio) == len(audio)

    def test_manifest_records_seed_and_version(self, clips_dir, tmp_path):
        out = tmp_path / "b"
        path = generate_dataset(
            clips_dir,
            output_dir=out,
            num_mixtures=1,
            durations=[2.0],
            speaker_counts=[2],
            overlap_ratios=[0.0],
            snr_values=[None],
            seed=55,
        )
        raw = json.loads(path.read_text())
        assert raw["seed"] == 55
        assert raw["version"]

    def test_reproducible_byte_for_byte(self, clips_dir, tmp_path):
        kwargs = dict(
            num_mixtures=2,
            durations=[3.0],
            speaker_counts=[2],
            overlap_ratios=[0.5],
            snr_values=[10],
            seed=77,
        )
        out_a = tmp_path / "a"
        out_b = tmp_path / "bb"
        pa = generate_dataset(clips_dir, output_dir=out_a, **kwargs)
        pb = generate_dataset(clips_dir, output_dir=out_b, **kwargs)

        mixtures = sorted(p.name for p in out_a.rglob("*") if p.suffix in {".wav", ".json"})
        mixtures_b = sorted(p.name for p in out_b.rglob("*") if p.suffix in {".wav", ".json"})
        assert mixtures == mixtures_b

        a1, _ = ManifestDataset(pa)[0].load_mixture()
        b1, _ = ManifestDataset(pb)[0].load_mixture()
        np.testing.assert_array_equal(a1, b1)

    def test_requests_more_speakers_than_available_raises(self, clips_dir, tmp_path):
        with pytest.raises(ValueError, match="speakers"):
            generate_mixture(
                index_clips(clips_dir), num_speakers=5, duration=4.0, overlap_ratio=0.0, seed=0
            )
