"""Hybrid diarization + selective separation pipeline (Phase 5, ADR-005).

Diarization drives everything: overlapping regions are detected from its
output, only those regions are separated, separated sources are attributed
back to speakers by spectral similarity against their solo speech, and the
output is reassembled with short crossfades.
"""

import time
from dataclasses import dataclass, field

import numpy as np

from benchmark.stft import stft

DEFAULT_CROSSFADE_S = 0.05
DEFAULT_PAD_S = 0.1


@dataclass
class HybridResult:
    output: np.ndarray
    segments: list
    overlap_regions: list
    assignments: dict = field(default_factory=dict)
    tracks: dict = field(default_factory=dict)
    selective_time: float = 0.0
    full_time: float | None = None


def spectral_profile(audio, sample_rate, n_fft=512):
    """Mean STFT magnitude of a signal, L2-normalized.

    Short signals are zero-padded so profiles always share the same
    dimensionality and stay comparable.
    """
    audio = np.asarray(audio, dtype=np.float32)
    if len(audio) < n_fft:
        audio = np.pad(audio, (0, n_fft - len(audio)))
    spec, _ = stft(audio, sample_rate, n_fft=n_fft)
    profile = np.abs(spec).mean(axis=1)
    norm = np.linalg.norm(profile)
    return profile / norm if norm > 0 else profile


class HybridPipeline:
    def __init__(
        self,
        diarizer,
        separator,
        crossfade_s=DEFAULT_CROSSFADE_S,
        pad_s=DEFAULT_PAD_S,
        attribution="spectral",
        encoder=None,
        max_enrollment_s=10.0,
    ):
        if attribution not in ("spectral", "embedding"):
            raise ValueError(f"unknown attribution mode: {attribution}")
        if attribution == "embedding" and encoder is None:
            raise ValueError("attribution='embedding' requires an encoder")
        self.diarizer = diarizer
        self.separator = separator
        self.crossfade_s = crossfade_s
        self.pad_s = pad_s
        self.attribution = attribution
        self.encoder = encoder
        self.max_enrollment_s = max_enrollment_s

    def process(self, audio, sample_rate, num_speakers=None, compare_full=False):
        audio = np.asarray(audio, dtype=np.float32)
        self._source_audio = audio
        self._source_sr = sample_rate
        segments = self.diarizer.diarize(audio, sample_rate)

        overlap_regions = self._overlap_regions(segments)
        output = audio.copy()
        tracks = {spk: np.zeros_like(audio) for spk in {s.speaker for s in segments}}
        assignments = {}
        selective_time = 0.0

        # solo regions: verbatim copy into each speaker's track
        for spk in tracks:
            mask = self._speaker_mask(segments, spk, len(audio), sample_rate, exclusive=True)
            idx = np.flatnonzero(mask)
            tracks[spk][idx] = audio[idx]

        for start, end in overlap_regions:
            i0 = max(0, int((start - self.pad_s) * sample_rate))
            i1 = min(len(audio), int((end + self.pad_s) * sample_rate))
            if i1 - i0 < int(0.05 * sample_rate):
                continue

            chunk = audio[i0:i1]
            t0 = time.perf_counter()
            sources = self.separator.separate(chunk, sample_rate, num_speakers=num_speakers)
            selective_time += time.perf_counter() - t0

            speakers_here = sorted({s.speaker for s in segments if s.start < end and s.end > start})
            mapping = self._attribute(
                sources,
                chunk,
                sample_rate,
                segments,
                speakers_here,
                overlap_regions,
                i0 / sample_rate,
                i1 / sample_rate,
            )
            assignments[(round(start, 3), round(end, 3))] = {i: spk for i, spk in mapping.items()}

            combined = np.zeros(i1 - i0, dtype=np.float32)
            # attribute into per-speaker tracks only over the true overlap
            # span (padding exists for separation context, not for assembly)
            j0 = max(i0, int(start * sample_rate))
            j1 = min(i1, int(end * sample_rate))
            for src_idx, spk in mapping.items():
                tracks[spk][j0:j1] += sources[src_idx][j0 - i0 : j1 - i0]
                combined += sources[src_idx]
            self._blend(output, combined, i0, i1, sample_rate)

        full_time = None
        if compare_full:
            t0 = time.perf_counter()
            self.separator.separate(audio, sample_rate, num_speakers=num_speakers)
            full_time = time.perf_counter() - t0

        return HybridResult(
            output=output,
            segments=segments,
            overlap_regions=overlap_regions,
            assignments=assignments,
            tracks=tracks,
            selective_time=selective_time,
            full_time=full_time,
        )

    @staticmethod
    def _overlap_regions(segments, step=0.01):
        if not segments:
            return []
        end = max(s.end for s in segments)
        grid = np.arange(0.0, end, step) + step / 2
        active = np.zeros(len(grid), dtype=np.int32)
        for seg in segments:
            active[(grid >= seg.start) & (grid < seg.end)] += 1
        overlapped = active >= 2
        regions = []
        inside = False
        for point, is_ovr in zip(grid, overlapped):
            if is_ovr and not inside:
                regions.append([point - step / 2, None])
                inside = True
            elif not is_ovr and inside:
                regions[-1][1] = point - step / 2
                inside = False
        if inside:
            regions[-1][1] = float(end)
        return [(round(a, 3), round(b, 3)) for a, b in regions]

    @staticmethod
    def _speaker_mask(segments, speaker, length, sr, exclusive=False):
        mask = np.zeros(length, dtype=bool)
        others = []
        if exclusive:
            others = [(s.start, s.end) for s in segments if s.speaker != speaker]
        for seg in segments:
            if seg.speaker != speaker:
                continue
            i0, i1 = int(seg.start * sr), min(length, int(seg.end * sr))
            span = np.zeros(length, dtype=bool)
            span[i0:i1] = True
            for o0, o1 in others:
                j0, j1 = max(0, int(o0 * sr)), min(length, int(o1 * sr))
                span[max(0, j0) : j1] = False
            mask |= span
        return mask

    def _attribute(
        self, sources, chunk, sr, segments, speakers, overlap_regions, region_start, region_end
    ):
        """Map each separated source index to a speaker. Two strategies:
        - spectral: similarity against the speaker's non-overlapped reference
          speech (Phase 5 baseline);
        - embedding: speaker-encoder embeddings of sources vs enrollment audio
          extracted from clean solo segments (ADR-006, Phase 6).
        """
        if self.attribution == "embedding":
            sims = self._embedding_similarities(sources, sr, segments, speakers)
        else:
            sims = self._spectral_similarities(
                sources,
                chunk,
                sr,
                segments,
                speakers,
                overlap_regions,
                region_start,
                region_end,
            )

        mapping = {}
        used_sources, used_speakers = set(), set()
        for (i, spk), score in sorted(sims.items(), key=lambda kv: -kv[1]):
            if i in used_sources or spk in used_speakers:
                continue
            mapping[i] = spk
            used_sources.add(i)
            used_speakers.add(spk)
        for i in range(len(sources)):
            if i not in mapping:
                best = max(speakers, key=lambda spk: sims[(i, spk)])
                mapping[i] = best
        return mapping

    def _enrollment_embeddings(self, segments, speakers, source_audio, source_sr):
        """Per-speaker enrollment embeddings from clean solo segments (cached)."""
        from benchmark.enrollment import (
            enrollment_audio,
            select_enrollment_segments,
        )

        cache = getattr(self, "_enrollment_cache", None)
        if cache is None or cache.get("_key") != id(segments):
            cache = {"_key": id(segments)}
            self._enrollment_cache = cache

        embeddings = {}
        for spk in speakers:
            if spk in cache:
                embeddings[spk] = cache[spk]
                continue
            spans = select_enrollment_segments(segments, spk, max_duration=self.max_enrollment_s)
            clip = enrollment_audio(source_audio, source_sr, spans)
            emb = self.encoder.encode(clip, source_sr) if len(clip) else None
            cache[spk] = emb
            embeddings[spk] = emb
        return embeddings

    def _embedding_similarities(self, sources, sr, segments, speakers):
        enrollments = self._enrollment_embeddings(
            segments, speakers, self._source_audio, self._source_sr
        )
        sims = {}
        for i, src in enumerate(sources):
            src_emb = self.encoder.encode(src, sr)
            for spk in speakers:
                emb = enrollments.get(spk)
                sims[(i, spk)] = float(np.dot(src_emb, emb)) if emb is not None else -1.0
        return sims

    def _spectral_similarities(
        self, sources, chunk, sr, segments, speakers, overlap_regions, region_start, region_end
    ):
        refs = {}
        for spk in speakers:
            spans = []
            for seg in segments:
                if seg.speaker != spk:
                    continue
                s, e = seg.start, seg.end
                for o0, o1 in overlap_regions:
                    if o1 <= region_start or o0 >= region_end:
                        continue  # unrelated region
                    if o0 < e and o1 > s:
                        if o0 > s:
                            spans.append((s, o0))
                        s = max(s, o1)
                if e > s:
                    spans.append((s, e))
            profiles = [
                spectral_profile(
                    chunk[
                        int(max(s, region_start) * sr - region_start * sr) : int(
                            min(e, region_end) * sr - region_start * sr
                        )
                    ],
                    sr,
                )
                for s, e in spans
                if min(e, region_end) > max(s, region_start)
            ]
            refs[spk] = profiles

        sims = {}
        for i, src in enumerate(sources):
            prof = spectral_profile(src, sr)
            for spk, profiles in refs.items():
                sims[(i, spk)] = max((float(np.dot(prof, p)) for p in profiles), default=-1.0)
        return sims

    def _blend(self, output, replacement, i0, i1, sr):
        fade = min(int(self.crossfade_s * sr), (i1 - i0) // 2)
        seg_len = i1 - i0
        if fade == 0:
            output[i0:i1] = replacement
            return
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        output[i0 : i0 + fade] = output[i0 : i0 + fade] * (1 - ramp) + replacement[:fade] * ramp
        output[i0 + fade : i1 - fade] = replacement[fade : seg_len - fade]
        output[i1 - fade : i1] = (
            output[i1 - fade : i1] * (1 - ramp[::-1]) + replacement[seg_len - fade :] * ramp[::-1]
        )
