"""Audio modality input handler: speech-to-text + a best-effort signal check.

Transcription uses faster-whisper (CPU, "tiny" model — chosen over "base"
for import-time load cost; swap `_MODEL_SIZE` if accuracy matters more
than startup time for your deployment). The frequency-domain check is
explicitly a heuristic: it flags WAV uploads with unusual high-frequency
energy (a rough proxy for ultrasonic/out-of-band audio-injection
attempts), not a validated defense. Treat a flag as "worth a second look
downstream", not as proof of an attack.
"""

import tempfile
import wave
from pathlib import Path

import numpy as np
from fastapi import HTTPException
from faster_whisper import WhisperModel
from scipy import signal as scipy_signal

from app.input_layer.base import InputHandler, InputResult

_MODEL_SIZE = "tiny"

# Loaded once at import time (per the design: a singleton, not re-loaded
# per request) so the ~75MB model weight is paid once per process, not
# once per request. This does mean importing this module needs network
# access on a cold cache and adds a few seconds to process startup.
_model = WhisperModel(_MODEL_SIZE, device="cpu", compute_type="int8")

# Above this frequency we don't expect meaningful speech energy; a
# suspicious upload deliberately embedding content here would show up as
# an elevated share of total spectral energy in this band.
_OUT_OF_BAND_HZ = 15_000
_OUT_OF_BAND_ENERGY_RATIO_THRESHOLD = 0.05


class AudioInputHandler(InputHandler):
    """Transcribes audio bytes to text via faster-whisper."""

    async def process(self, raw: bytes) -> InputResult:
        """Transcribe `raw` audio bytes.

        Raises:
            HTTPException: 400, if `raw` cannot be decoded as audio.
        """
        signal_check = self._signal_check(raw)

        tmp_path = self._write_temp_file(raw)
        try:
            segments, _info = _model.transcribe(str(tmp_path), beam_size=5)
            segments = list(segments)
        except Exception as exc:  # faster-whisper/av raise various decode errors
            raise HTTPException(
                status_code=400, detail="Could not decode audio input."
            ) from exc
        finally:
            tmp_path.unlink(missing_ok=True)

        text = "".join(segment.text for segment in segments).strip()
        confidence = self._confidence_from_segments(segments)

        return InputResult(
            text=text,
            modality="audio",
            confidence=confidence,
            metadata={"signal_check": signal_check},
        )

    @staticmethod
    def _write_temp_file(raw: bytes) -> Path:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(raw)
            return Path(f.name)

    @staticmethod
    def _confidence_from_segments(segments: list) -> float | None:
        """Map whisper's avg_logprob (roughly -1..0, higher is better) to [0, 1].

        This is a rough, undocumented-by-whisper heuristic, not a
        calibrated probability — treat it as a relative signal, not an
        exact confidence percentage.
        """
        if not segments:
            return None

        avg_logprob = sum(s.avg_logprob for s in segments) / len(segments)
        return max(0.0, min(1.0, 1.0 + avg_logprob))

    @staticmethod
    def _signal_check(raw: bytes) -> dict:
        """Best-effort frequency-domain check on WAV-encoded input.

        Returns a dict describing what was checked, always — including
        when the check couldn't run (non-WAV input, corrupt header) —
        so the caller can see whether the flag is meaningful or just
        "not applicable".
        """
        import io

        try:
            with wave.open(io.BytesIO(raw), "rb") as wav_file:
                sample_rate = wav_file.getframerate()
                n_frames = wav_file.getnframes()
                sample_width = wav_file.getsampwidth()
                n_channels = wav_file.getnchannels()
                frames = wav_file.readframes(n_frames)
        except (wave.Error, EOFError):
            return {"performed": False, "reason": "input is not a readable WAV file"}

        if sample_width != 2 or n_frames == 0:
            return {"performed": False, "reason": "unsupported sample width or empty audio"}

        samples = np.frombuffer(frames, dtype=np.int16)
        if n_channels > 1:
            samples = samples.reshape(-1, n_channels).mean(axis=1)

        if len(samples) < 2:
            return {"performed": False, "reason": "too few samples"}

        # Welch's method (scipy.signal): power spectral density estimate,
        # sturdier against noise than a single raw FFT bin-by-bin.
        freqs, psd = scipy_signal.welch(samples.astype(np.float64), fs=sample_rate)

        total_energy = float(psd.sum())
        if total_energy == 0:
            return {"performed": True, "flagged": False, "out_of_band_energy_ratio": 0.0}

        out_of_band_energy = float(psd[freqs > _OUT_OF_BAND_HZ].sum())
        ratio = out_of_band_energy / total_energy

        return {
            "performed": True,
            "flagged": ratio > _OUT_OF_BAND_ENERGY_RATIO_THRESHOLD,
            "out_of_band_energy_ratio": round(ratio, 4),
        }
