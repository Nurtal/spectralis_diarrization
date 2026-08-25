import numpy as np

from benchmark.vad import energy_vad

SR = 16000


def tone(duration_s, freq=440, amp=0.3):
    t = np.arange(int(duration_s * SR)) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


class TestEnergyVad:
    def test_detects_tone_rejects_silence(self):
        audio = np.concatenate([np.zeros(SR, dtype=np.float32), tone(1.0)])
        segments = energy_vad(audio, SR)
        assert len(segments) == 1
        assert 0.9 <= segments[0].start < 1.1
        assert 1.9 < segments[0].end <= 2.01
        assert segments[0].speaker == "speech"

    def test_alternating_silence_and_speech_two_segments(self):
        gap = np.zeros(int(0.3 * SR), dtype=np.float32)
        audio = np.concatenate([tone(0.5), gap, gap, tone(0.5)])
        segments = energy_vad(audio, SR)
        assert len(segments) == 2
        assert segments[0].end <= 0.7
        assert segments[1].start >= 1.0

    def test_all_silence_gives_no_segments(self):
        segments = energy_vad(np.zeros(SR, dtype=np.float32), SR)
        assert segments == []

    def test_very_short_bursts_filtered_by_min_duration(self):
        click = tone(0.02)
        silence = np.zeros(SR, dtype=np.float32)
        audio = np.concatenate([click, silence])
        segments = energy_vad(audio, SR, min_speech_s=0.1)
        assert segments == []
