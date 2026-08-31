"""Target-speaker extraction conditioned on enrollment (M6, RQ5, ADR-006).

Primary path: Asteroid SpEx+ when available (``model_class='spexplus'``).
Fallback: blind separation + ECAPA cosine selection (separate-then-select),
which already answers the core research question without requiring a
specialized checkpoint.

The fallback is the default in the current environment where
``TFGridNetSeparation`` is not in SpeechBrain 1.1.0 and SpEx+ hub ids are
not yet published. It is a legitimate conditioned baseline: it uses the
enrollment embedding to choose the correct source, precisely the
attribution problem that RQ5 studies.
"""

import numpy as np


def _asteroid_available():
    try:
        import asteroid  # noqa: F401

        return True
    except ImportError:
        return False


def _cosine(a, b):
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)


class ConditionedSeparator:
    """Extract a target speaker given enrollment audio.

    Parameters
    ----------
    model:
        Duck-typed injected model for tests. Must expose
        ``separate_target(mixture, enrollment)`` or ``forward``.
    model_id:
        HuggingFace id for an Asteroid SpEx+ checkpoint (when ``model`` is None).
    model_class:
        ``'spexplus'`` (default) or ``'fallback'``. ``'fallback'`` forces the
        blind+embedding path even when asteroid is installed.
    separator:
        Blind separator used in fallback mode (any ``Separator``).
    encoder:
        Speaker encoder used in fallback mode (any ``SpeakerEncoder``).
    device:
        Device string passed to asteroid / SpeechBrain loaders.
    """

    def __init__(
        self,
        model=None,
        model_id=None,
        model_class="spexplus",
        separator=None,
        encoder=None,
        device="cpu",
    ):
        self.model_class = str(model_class).lower()
        self.device = device

        if model is not None:
            self._model = model
            self._mode = "injected"
            self.separator = separator
            self.encoder = encoder
            return

        # try asteroid path when requested and available
        if self.model_class in ("spexplus", "spex+", "spext", "target"):
            if model_id is not None and _asteroid_available():
                try:
                    # lazy import to keep the module importable without asteroid
                    from asteroid.models import SpExPlus  # type: ignore

                    # Asteroid's from_pretrained is the common entry point;
                    # we keep the call generic and let the hub id decide.
                    # If the id is not a valid SpEx+ checkpoint, the constructor
                    # will raise and we fall through to the fallback path.
                    self._model = SpExPlus.from_pretrained(model_id)  # type: ignore
                    if hasattr(self._model, "to"):
                        try:
                            self._model.to(device)
                        except Exception:
                            pass
                    self._mode = "asteroid"
                    self.separator = separator
                    self.encoder = encoder
                    return
                except Exception as e:
                    # fall through to fallback, but keep the error for debugging
                    self._asteroid_error = str(e)
            # no model_id or asteroid not available / load failed → fallback
            self._mode = "fallback"
            self._asteroid_error = None if _asteroid_available() else "asteroid not installed"
        elif self.model_class == "fallback":
            self._mode = "fallback"
        else:
            raise ValueError(f"unknown model_class '{model_class}' (expected spexplus|fallback)")

        # fallback dependencies (lazy to avoid importing torch/speechbrain at init)
        if self._mode == "fallback":
            if separator is not None:
                self.separator = separator
            else:
                # delay import to keep module importable without torch
                self.separator = None
            if encoder is not None:
                self.encoder = encoder
            else:
                self.encoder = None
            self._model = None

    def _ensure_fallback_deps(self):
        if self.separator is None:
            from benchmark.registry import SEPARATORS

            self.separator = SEPARATORS["sepformer"]()
        if self.encoder is None:
            from benchmark.speaker_encoder import SpeakerEncoder

            self.encoder = SpeakerEncoder(device=self.device)

    def extract_target(self, mixture, sample_rate, enrollment_audio, enrollment_sr=None):
        """Return the estimated target speaker waveform (1D array)."""
        mixture = np.asarray(mixture, dtype=np.float32)
        if mixture.ndim != 1:
            raise ValueError("mixture must be 1D mono")
        if enrollment_audio is None:
            raise ValueError("enrollment_audio must be provided")
        enrollment_audio = np.asarray(enrollment_audio, dtype=np.float32)
        if enrollment_audio.ndim != 1:
            raise ValueError("enrollment_audio must be 1D mono")
        if enrollment_sr is None:
            enrollment_sr = sample_rate
        if len(enrollment_audio) == 0:
            raise ValueError("enrollment_audio is empty")

        if self._mode == "injected":
            # duck-typed: try separate_target, then forward, then separate
            if hasattr(self._model, "separate_target"):
                out = self._model.separate_target(mixture, enrollment_audio)
                return np.asarray(out, dtype=np.float32).reshape(-1)
            if hasattr(self._model, "forward"):
                # asteroid SpExPlus forward: mixture, enrollment
                try:
                    import torch

                    with torch.no_grad():
                        mix_t = torch.from_numpy(mixture).float().unsqueeze(0)
                        enrol_t = torch.from_numpy(enrollment_audio).float().unsqueeze(0)
                        est = self._model(mix_t, enrol_t)
                        return np.asarray(est.squeeze().cpu().numpy(), dtype=np.float32)
                except Exception:
                    pass
            # fallback to separate then select if injected model is actually a blind separator
            if hasattr(self._model, "separate_batch") or hasattr(self._model, "separate"):
                # treat as blind separator for fallback testing
                self._mode = "fallback"
                self.separator = self._model
                self.encoder = self.encoder or self._ensure_encoder_fallback()
            else:
                raise RuntimeError("injected model has no separate_target/forward method")

        if self._mode == "asteroid":
            try:
                import torch

                with torch.no_grad():
                    mix_t = torch.from_numpy(mixture).float().unsqueeze(0)
                    enrol_t = torch.from_numpy(enrollment_audio).float().unsqueeze(0)
                    est = self._model(mix_t, enrol_t)
                    # SpExPlus returns (batch, time) or (batch, 1, time)
                    est_np = np.asarray(est.squeeze().cpu().numpy(), dtype=np.float32)
                    return est_np.reshape(-1)
            except Exception as e:
                raise RuntimeError(f"asteroid SpEx+ inference failed: {e}") from e

        # fallback: blind separation + embedding cosine selection
        if self._mode == "fallback":
            self._ensure_fallback_deps()
            # lazy encoder for fallback if still None
            if self.encoder is None:
                self.encoder = self._ensure_encoder_fallback()
            estimates = self.separator.separate(mixture, sample_rate)
            if not estimates:
                raise RuntimeError("fallback separator returned no estimates")
            # enrollment embedding
            try:
                enrol_emb = self.encoder.encode(enrollment_audio, int(enrollment_sr))
            except Exception as e:
                raise RuntimeError(f"failed to encode enrollment: {e}") from e
            best, best_score = None, -2.0
            for est in estimates:
                try:
                    emb = self.encoder.encode(est, int(sample_rate))
                    score = _cosine(emb, enrol_emb)
                except Exception:
                    score = -1.0
                if score > best_score:
                    best_score, best = score, est
            if best is None:
                raise RuntimeError("failed to select target from fallback estimates")
            return np.asarray(best, dtype=np.float32).reshape(-1)

        raise RuntimeError(f"unknown mode {self._mode}")

    def _ensure_encoder_fallback(self):
        from benchmark.speaker_encoder import SpeakerEncoder

        return SpeakerEncoder(device=self.device)

    # alias for Separator-like API
    def separate(self, mixture, sample_rate, enrollment_audio=None, enrollment_sr=None):
        """List API for compatibility: returns [target]."""
        if enrollment_audio is None:
            raise ValueError("conditioned separation requires enrollment_audio")
        target = self.extract_target(mixture, sample_rate, enrollment_audio, enrollment_sr)
        return [target]
