"""Audio helpers: listing and post-processing (trim + normalize)."""

from pathlib import Path

import numpy as np
import soundfile as sf

from tts.paths import REFERENCE_AUDIO_DIR, INPUT_AUDIO_DIR, OUTPUT_AUDIO_DIR

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma"}

TRIM_THRESHOLD = 0.02
TRIM_PAD_SECONDS = 0.1
NORMALIZE_PEAK = 0.8


def ensure_output_dir():
    OUTPUT_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_AUDIO_DIR


def list_reference_audio():
    return _list_audio(REFERENCE_AUDIO_DIR)


def list_input_audio():
    return _list_audio(INPUT_AUDIO_DIR)


def list_output_audio():
    return _list_audio(OUTPUT_AUDIO_DIR)


def _list_audio(directory):
    directory = Path(directory)
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in AUDIO_EXTENSIONS)


def trim_trailing_silence(data, sr, threshold=TRIM_THRESHOLD, pad_seconds=TRIM_PAD_SECONDS):
    """Cut trailing silence, keeping `pad_seconds` of padding after the last loud sample."""
    if len(data) == 0:
        return data
    abs_data = np.abs(data)
    if abs_data.ndim > 1:
        abs_data = abs_data.max(axis=1)
    loud = np.nonzero(abs_data > threshold)[0]
    if len(loud) == 0:
        return data
    end = loud[-1] + int(pad_seconds * sr)
    return data[: min(end, len(data))]


def normalize_peak(data, target=NORMALIZE_PEAK):
    """Scale so the maximum absolute peak equals `target` (-2 dB). No-op if silent."""
    peak = np.abs(data).max() if len(data) else 0.0
    if peak <= 0.0:
        return data
    return data * (target / peak)


def polish_audio(input_path, output_path=None):
    """Trim trailing silence and normalize peak volume, then save as 16-bit PCM WAV.

    Overwrites the input file when output_path is None.
    Returns the output Path.
    """
    input_path = Path(input_path)
    output_path = Path(output_path) if output_path else input_path

    data, sr = sf.read(str(input_path), dtype="float32")
    data = trim_trailing_silence(data, sr)
    data = normalize_peak(data)
    sf.write(str(output_path), data, sr, subtype="PCM_16")
    return output_path


def detect_static_noise(audio, sample_rate):
    """Return True if the signal looks like static/noise rather than speech.

    Heuristics:
    1. RMS < 0.01 and duration > 0.5 s -> silence or pure static.
    2. Spectral flatness (geometric/arithmetic mean of power spectrum)
       > 0.5 -> white noise/static (speech is typically < 0.3).
    3. Zero-crossing rate > 0.5 -> ultra high-frequency noise.
    """
    audio = np.asarray(audio, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if len(audio) == 0:
        return True
    duration = len(audio) / sample_rate
    if duration <= 0.5:
        return False

    rms = float(np.sqrt(np.mean(audio ** 2)))
    if rms < 0.01:
        return True

    windowed = audio * np.hanning(len(audio))
    spec = np.abs(np.fft.rfft(windowed)) ** 2
    spec = spec[spec > 1e-12]
    if spec.size == 0:
        return True
    flatness = float(np.exp(np.mean(np.log(spec))) / (np.mean(spec) + 1e-12))

    signs = np.signbit(audio)
    if len(signs) > 1:
        zcr = float(np.mean(np.diff(signs.astype(np.int8)) != 0))
    else:
        zcr = 0.0

    if flatness > 0.5 or zcr > 0.5:
        return True
    return False