import numpy as np
import pytest

from benchmark.interfaces import Diarizer, Segment, Separator


class TestSegment:
    def test_holds_timing_and_speaker(self):
        seg = Segment(start=0.0, end=4.0, speaker="A")
        assert seg.start == 0.0
        assert seg.end == 4.0
        assert seg.speaker == "A"

    def test_rejects_end_before_start(self):
        with pytest.raises(ValueError, match="end"):
            Segment(start=4.0, end=2.0, speaker="A")


class TestSeparator:
    def test_cannot_instantiate_abstract_base(self):
        with pytest.raises(TypeError):
            Separator()

    def test_subclass_must_implement_separate(self):
        class Incomplete(Separator):
            pass

        with pytest.raises(TypeError):
            Incomplete()

    def test_working_subclass_returns_sources(self):
        class Identity(Separator):
            def separate(self, audio, sample_rate, num_speakers=None):
                return [audio.copy()]

        audio = np.zeros(100, dtype=np.float32)
        sources = Identity().separate(audio, 16000)
        assert len(sources) == 1
        assert sources[0].shape == audio.shape


class TestDiarizer:
    def test_cannot_instantiate_abstract_base(self):
        with pytest.raises(TypeError):
            Diarizer()

    def test_working_subclass_returns_segments(self):
        class Fixed(Diarizer):
            def diarize(self, audio, sample_rate):
                return [Segment(0.0, 2.0, "spk0")]

        segments = Fixed().diarize(np.zeros(32000, dtype=np.float32), 16000)
        assert segments == [Segment(0.0, 2.0, "spk0")]
