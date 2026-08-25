"""Adapter exposing pretrained neural separators behind the common interface.

SpeechBrain checkpoints are loaded lazily; the heavy dependencies (torch,
speechbrain) are only required when actually constructing a separator without
an injected model. Model outputs are resampled back to the caller's rate.
"""

import numpy as np

from benchmark.interfaces import Separator

DEFAULT_MODEL_SAMPLE_RATE = 8000


class SpeechBrainSeparator(Separator):
    """Separator backed by a SpeechBrain pretrained model.

    Pass `model=` (any object with `separate_batch(waveform) -> [1, T, N]`)
    for testing, or `model_id=` to download a HuggingFace checkpoint.
    """

    def __init__(
        self,
        model=None,
        model_id=None,
        model_sample_rate=DEFAULT_MODEL_SAMPLE_RATE,
        device="cpu",
        hf_token=None,
    ):
        if model is not None:
            self._model = model
            self.model_sample_rate = int(getattr(model, "sample_rate", model_sample_rate))
            return

        try:
            from speechbrain.inference.separation import SepformerSeparation
        except ImportError as e:
            raise RuntimeError(
                "speechbrain is not installed. Install it with "
                "`pip install torch torchaudio speechbrain` to use neural separators."
            ) from e

        if model_id is None:
            raise ValueError("either `model` or `model_id` must be provided")

        import os
        import tempfile

        savedir = tempfile.mkdtemp(prefix="spectralis-model-")
        if hf_token:
            # speechbrain/huggingface_hub pick the token up from the environment
            os.environ.setdefault("HF_TOKEN", hf_token)
        self._model = SepformerSeparation.from_hparams(
            source=model_id,
            savedir=savedir,
            overrides={"device": device} if device != "cpu" else None,
        )
        self.model_sample_rate = int(model_sample_rate)

    def separate(self, audio, sample_rate, num_speakers=None):
        from benchmark.audio import load_audio

        audio = self._validate_audio(audio, sample_rate)

        model_sr = self.model_sample_rate
        if sample_rate != model_sr:
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
                from benchmark.audio import save_audio

                save_audio(tmp.name, audio, sample_rate)
                audio, _ = load_audio(tmp.name, sample_rate=model_sr)

        batch_in = np.asarray(audio, dtype=np.float32)[None, :]
        try:
            import torch
        except ImportError:
            estimates = np.asarray(self._model.separate_batch(batch_in))
        else:
            with torch.no_grad():
                estimates = self._model.separate_batch(torch.from_numpy(batch_in))

        batch = np.asarray(estimates).squeeze(0).astype(np.float32)
        sources = []
        for i in range(batch.shape[-1]):
            source = batch[:, i]
            if sample_rate != model_sr:
                source = _resample(source, model_sr, sample_rate)
            # neural outputs are unbounded; keep them within the pipeline's
            # float-audio contract
            source = np.clip(source, -1.0, 1.0)
            sources.append(np.ascontiguousarray(source, dtype=np.float32))
        return sources


def _resample(x, sr_in, sr_out):
    from math import gcd

    from scipy.signal import resample_poly

    g = gcd(int(sr_in), int(sr_out))
    return resample_poly(x, sr_out // g, sr_in // g).astype(np.float32)
