"""Speaker embedding extraction behind a common interface (ADR-006).

ECAPA-TDNN via SpeechBrain is the reference encoder; any duck-typed model
with `encode_batch` can be injected for tests.
"""

import numpy as np

from benchmark.interfaces import Segment  # noqa: F401 (kept for API symmetry)

DEFAULT_ECAPA_ID = "speechbrain/spkrec-ecapa-voxceleb"
ENCODE_SAMPLE_RATE = 16000


class SpeakerEncoder:
    def __init__(self, model=None, model_id=DEFAULT_ECAPA_ID, device="cpu"):
        if model is not None:
            self._model = model
            return

        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError as e:
            raise RuntimeError(
                "speechbrain is not installed. Install it with "
                "`pip install torch torchaudio speechbrain` to extract speaker embeddings."
            ) from e

        import tempfile

        savedir = tempfile.mkdtemp(prefix="spectralis-encoder-")
        hparams = {
            "source": model_id,
            "run_opts": {"device": device},
        }
        self._model = EncoderClassifier.from_hparams(savedir=savedir, **hparams)

    def encode(self, audio, sample_rate):
        """Return an L2-normalized embedding as float32 1-D array."""
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim != 1:
            raise ValueError("audio must be a 1D mono waveform")
        if len(audio) == 0:
            raise ValueError("cannot encode empty audio")

        if sample_rate != ENCODE_SAMPLE_RATE:
            from benchmark.neural_separator import _resample

            audio = _resample(audio, sample_rate, ENCODE_SAMPLE_RATE)

        embedding = self._compute_embedding(audio)
        embedding = np.asarray(embedding, dtype=np.float32).reshape(-1)
        norm = np.linalg.norm(embedding)
        return embedding / norm if norm > 0 else embedding

    def _compute_embedding(self, audio):
        """Duck-typed dispatch: injected encoders expose `encode`, SpeechBrain
        models expose `encode_batch` (torch tensors)."""
        if hasattr(self._model, "encode") and not hasattr(self._model, "encode_batch"):
            return self._model.encode(audio, ENCODE_SAMPLE_RATE)
        return self._encode_batch(audio)

    def _encode_batch(self, audio):
        import torch

        waveform = torch.from_numpy(audio).float().unsqueeze(0)
        with torch.no_grad():
            return self._model.encode_batch(waveform).squeeze().cpu().numpy()
