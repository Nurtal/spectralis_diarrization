"""Adapter exposing pyannote.audio speaker diarization behind the common interface.

Requires `pip install pyannote.audio` and a HuggingFace token accepted for
the model (see https://huggingface.co/pyannote/speaker-diarization-3.1).
"""

import os

import numpy as np

from benchmark.interfaces import Diarizer, Segment

DEFAULT_MODEL = "pyannote/speaker-diarization-3.1"


class PyannoteDiarizer(Diarizer):
    def __init__(self, model=DEFAULT_MODEL, hf_token=None, device=None):
        try:
            from pyannote.audio import Pipeline
        except ImportError as e:
            raise RuntimeError(
                "pyannote.audio is not installed. Install it with "
                "`pip install pyannote.audio torch torchaudio` to use this diarizer."
            ) from e

        token = hf_token or os.environ.get("HF_TOKEN")
        if token is None and not _model_cached(model):
            raise RuntimeError(
                "No HuggingFace token found. Accept the model conditions at "
                f"https://huggingface.co/{model} then run `hf auth login`, "
                "or pass hf_token=... / set HF_TOKEN."
            )

        self._pipeline = Pipeline.from_pretrained(model, token=token)
        if device is not None:
            self._pipeline.to(device)

    def diarize(self, audio, sample_rate):
        import torch

        audio = self._validate_audio(audio, sample_rate)
        waveform = torch.from_numpy(np.asarray(audio)).float().unsqueeze(0)
        annotation = self._pipeline({"waveform": waveform, "sample_rate": int(sample_rate)})
        # pyannote 4.x returns DiarizeOutput with speaker_diarization attribute
        diarization = getattr(annotation, "speaker_diarization", annotation)
        return [
            Segment(start=turn.start, end=turn.end, speaker=speaker)
            for turn, _, speaker in diarization.itertracks(yield_label=True)
        ]


def _model_cached(model_id):
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    return try_to_load_from_cache(model_id, "config.yaml") is not None
