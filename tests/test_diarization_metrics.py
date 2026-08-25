from benchmark.interfaces import Segment
from benchmark.metrics import (
    der,
    jer,
    overlap_detection_scores,
)


def S(*spans):
    return [Segment(start=a, end=b, speaker=s) for a, b, s in spans]


class TestDer:
    def test_perfect_match_is_zero(self):
        ref = S((0, 1, "A"), (1, 2, "B"))
        hyp = S((0, 1, "X"), (1, 2, "Y"))
        result = der(ref, hyp)
        assert result["der"] == 0.0
        assert result["missed"] == 0.0
        assert result["false_alarm"] == 0.0
        assert result["confusion"] == 0.0

    def test_label_invariant(self):
        ref = S((0, 1, "A"), (1, 2, "B"))
        hyp = S((0, 1, "zzz"), (1, 2, "aaa"))
        assert der(ref, hyp)["der"] == pytest_approx(0.0)

    def test_half_missed_speech(self):
        ref = S((0, 2, "A"))
        hyp = S((0, 1, "anything"))
        result = der(ref, hyp)
        assert result["der"] == pytest_approx(0.5)
        assert result["missed"] == pytest_approx(1.0)

    def test_false_alarm_and_missed(self):
        # ref speaks 1s; hyp speaks 3s covering it: 1s correct + 2s FA
        ref = S((1, 2, "A"))
        hyp = S((0, 3, "H"))
        result = der(ref, hyp)
        assert result["der"] == pytest_approx(2.0)
        assert result["false_alarm"] == pytest_approx(2.0)
        assert result["missed"] == pytest_approx(0.0)

    def test_missed_when_hypothesis_covers_only_part(self):
        # hyp covers middle half of a 2s reference
        ref = S((0, 2, "A"))
        hyp = S((0.5, 1.5, "H"))
        result = der(ref, hyp)
        assert result["missed"] == pytest_approx(1.0)
        assert result["false_alarm"] == pytest_approx(0.0)
        assert result["der"] == pytest_approx(0.5)

    def test_unmapped_hypothesis_counts_as_confusion(self):
        # two sequential hyp labels over one ref speaker: second is confusion
        ref = S((0, 1, "A"))
        hyp = S((0, 0.5, "A'"), (0.5, 1, "B'"))
        result = der(ref, hyp)
        assert result["confusion"] == pytest_approx(0.5)
        assert result["der"] == pytest_approx(0.5)

    def test_overlap_region_scored_once_per_ref_time(self):
        # A and B speak simultaneously for 1s; hyp only has one label
        ref = S((0, 1, "A"), (0, 1, "B"))
        hyp = S((0, 1, "only"))
        result = der(ref, hyp)
        # total ref speech time = 2s; one mapped correctly, other missed
        assert result["total"] == pytest_approx(2.0)
        assert result["missed"] == pytest_approx(1.0)
        assert result["der"] == pytest_approx(0.5)

    def test_no_reference_speech_returns_nan_der(self):
        result = der([], S((0, 1, "A")))
        assert result["total"] == 0.0

    def test_empty_hypothesis_all_missed(self):
        ref = S((0, 4, "A"), (2, 6, "B"))
        result = der(ref, [])
        assert result["der"] == pytest_approx(1.0)
        assert result["missed"] == pytest_approx(8.0)


class TestOverlapDetection:
    def test_perfect_overlap_detection(self):
        ref = S((0, 1, "A"), (0, 1, "B"), (2, 3, "A"))
        hyp = S((0, 1, "x"), (0, 1, "y"), (2, 3, "z"))
        scores = overlap_detection_scores(ref, hyp)
        assert scores["precision"] == pytest_approx(1.0)
        assert scores["recall"] == pytest_approx(1.0)

    def test_single_label_hyp_misses_all_overlaps(self):
        ref = S((0, 1, "A"), (0, 1, "B"))
        hyp = S((0, 1, "one"))
        scores = overlap_detection_scores(ref, hyp)
        assert scores["recall"] == pytest_approx(0.0)

    def test_precision_recall_on_partial_detection(self):
        # ref overlap on [0,2]; hyp detects overlap only on [0,1] and
        # falsely on [3,4]
        ref = S((0, 2, "A"), (0, 2, "B"))
        hyp = S((0, 1, "a"), (0, 1, "b"), (3, 4, "c"), (3, 4, "d"))
        scores = overlap_detection_scores(ref, hyp)
        assert scores["precision"] == pytest_approx(0.5)
        assert scores["recall"] == pytest_approx(0.5)

    def test_no_overlaps_anywhere_gives_zero_scores_not_crash(self):
        ref = S((0, 1, "A"))
        hyp = S((0, 1, "B"))
        scores = overlap_detection_scores(ref, hyp)
        assert scores["f1"] == 0.0


class TestJer:
    def test_perfect_match_is_zero(self):
        ref = S((0, 1, "A"), (1, 2, "B"))
        hyp = S((0, 1, "X"), (1, 2, "Y"))
        assert jer(ref, hyp) == pytest_approx(0.0)

    def test_half_covered_reference(self):
        ref = S((0, 2, "A"))
        hyp = S((0, 1, "H"))
        # J = 1s / 2s -> error 0.5
        assert jer(ref, hyp) == pytest_approx(0.5)

    def test_label_invariant(self):
        ref = S((0, 1, "A"), (1, 2, "B"))
        hyp = S((1, 2, "zzz"), (0, 1, "aaa"))
        assert jer(ref, hyp) == pytest_approx(0.0)

    def test_uncovered_speaker_counts_full_error(self):
        ref = S((0, 1, "A"), (1, 2, "B"))
        hyp = S((0, 1, "only"))  # single label mapped to A; B uncovered
        # A: J=1 -> err 0 ; B: no hyp time -> err 1 ; mean = 0.5
        assert jer(ref, hyp) == pytest_approx(0.5)


def pytest_approx(value):
    from pytest import approx

    return approx(value, abs=1e-6)
