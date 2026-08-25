import numpy as np
import pytest

from benchmark.hybrid import HybridPipeline, spectral_profile
from benchmark.interfaces import Segment, Separator

SR = 8000


def tone(freq, start_s, end_s, dur_s=4.0, amp=0.3, sr=SR):
    x = np.zeros(int(dur_s * sr), dtype=np.float32)
    i0, i1 = int(start_s * sr), int(end_s * sr)
    t = np.arange(i1 - i0) / sr
    x[i0:i1] = amp * np.sin(2 * np.pi * freq * t)
    return x


class OracleDiarizer:
    """Perfect diarizer for controlled tests."""

    def __init__(self, segments):
        self.segments = segments

    def diarize(self, audio, sample_rate):
        return list(self.segments)


class NMFishSeparator(Separator):
    """Frequency-band splitter: bins < 1000 Hz -> source 0, else source 1.

    Good enough to validate the pipeline mechanics on disjoint tones.
    """

    def separate(self, audio, sample_rate, num_speakers=None):
        from benchmark.stft import istft, stft

        audio = self._validate_audio(audio, sample_rate)
        spec, params = stft(audio, sample_rate)
        freqs = np.fft.rfftfreq(params.n_fft, 1 / sample_rate)
        sources = []
        for band in (freqs < 1000.0, freqs >= 1000.0):
            masked = np.zeros_like(spec)
            masked[band, :] = spec[band, :]
            y, _ = istft(masked, params, length=len(audio))
            sources.append(y)
        return sources


@pytest.fixture
def scenario():
    a = tone(200.0, 0.5, 2.5)
    b = tone(2400.0, 1.5, 3.5)
    mixture = np.clip(a + b, -1.0, 1.0)
    segments = [Segment(0.5, 2.5, "A"), Segment(1.5, 3.5, "B")]
    return {
        "mixture": mixture,
        "sources": {"A": a, "B": b},
        "segments": segments,
    }


@pytest.fixture
def pipeline(scenario):
    return HybridPipeline(
        diarizer=OracleDiarizer(scenario["segments"]),
        separator=NMFishSeparator(),
    )


class TestOverlapDetection:
    def test_overlap_region_found(self, scenario, pipeline):
        result = pipeline.process(scenario["mixture"], SR)
        assert len(result.overlap_regions) == 1
        start, end = result.overlap_regions[0]
        assert start == pytest.approx(1.5, abs=0.05)
        assert end == pytest.approx(2.5, abs=0.05)

    def test_no_overlap_when_single_speaker(self, scenario):
        pipe = HybridPipeline(
            diarizer=OracleDiarizer([Segment(0.0, 2.0, "A")]),
            separator=NMFishSeparator(),
        )
        result = pipe.process(np.zeros(SR, dtype=np.float32), SR)
        assert result.overlap_regions == []


class TestSelectiveReassembly:
    def test_non_overlap_audio_untouched(self, scenario, pipeline):
        result = pipeline.process(scenario["mixture"], SR)
        # identical strictly before the padded/faded separation zone
        # (overlap starts at 1.5 s, pad is 0.1 s)
        safe_end = int(1.38 * SR)
        np.testing.assert_array_equal(result.output[:safe_end], scenario["mixture"][:safe_end])

    def test_output_length_preserved(self, scenario, pipeline):
        result = pipeline.process(scenario["mixture"], SR)
        assert len(result.output) == len(scenario["mixture"])

    def test_no_nans_or_explosions(self, scenario, pipeline):
        result = pipeline.process(scenario["mixture"], SR)
        assert np.isfinite(result.output).all()
        assert np.abs(result.output).max() <= 1.0 + 1e-6


class TestAttribution:
    def test_sources_assigned_to_correct_speakers(self, scenario, pipeline):
        result = pipeline.process(scenario["mixture"], SR)
        region = result.assignments[result.overlap_regions[0]]
        assert set(region.values()) == {"A", "B"}
        # check by spectral content of attributed tracks
        prof_a = spectral_profile(result.tracks["A"], SR)
        prof_b = spectral_profile(result.tracks["B"], SR)
        freqs = np.fft.rfftfreq(512, 1 / SR)
        assert prof_a[np.argmax(prof_a)] > 0 and freqs[int(np.argmax(prof_a))] < 1000
        assert freqs[int(np.argmax(prof_b))] > 2000

    def test_tracks_reconstruct_clean_speakers(self, scenario, pipeline):
        from benchmark.separation_metrics import best_pairing_si_sdr

        result = pipeline.process(scenario["mixture"], SR)
        refs = [scenario["sources"]["A"], scenario["sources"]["B"]]
        ests = [result.tracks["A"], result.tracks["B"]]
        score = best_pairing_si_sdr(refs, ests)
        # solo regions are copied verbatim; overlap separation is near-perfect
        # on disjoint bands, so assembled tracks should be very clean
        assert score > 15.0


class TestTimings:
    def test_records_selective_and_full_times(self, scenario, pipeline):
        result = pipeline.process(scenario["mixture"], SR, compare_full=True)
        assert result.selective_time >= 0.0
        assert result.full_time >= 0.0
