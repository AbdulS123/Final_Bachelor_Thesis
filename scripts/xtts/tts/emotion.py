"""Emotion and style control V1 for XTTS generation.

Emotion approximation via:
1. Generation parameter presets (temperature/top_k/top_p/penalties).
2. Speed/pitch/volume/lowpass post-processing (librosa + scipy).
3. Optional emotional reference clips from data/audio/reference/emotions/.

Neutral is untouched: it uses the locked "austrian" default and only passes
through the normal audio polish.
"""

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfiltfilt

ROOT = Path(__file__).resolve().parent.parent
EMOTION_REFERENCE_DIR = ROOT / "data" / "audio" / "reference" / "emotions"

NEUTRAL_GEN_PARAMS = {
    "temperature": 0.75,
    "top_k": 62,
    "top_p": 0.86,
    "repetition_penalty": 1.09,
    "length_penalty": 1.05,
}

_NEUTRAL_POST = {
    "speed_rate": 1.0,
    "pitch_semitones": 0.0,
    "volume_scale": 1.0,
    "lowpass_hz": None,
}

EMOTION_PRESETS = {
    "neutral": {
        **NEUTRAL_GEN_PARAMS,
        **_NEUTRAL_POST,
        "note": "New body_single voice with balanced-natural parameters.",
    },
    "happy": {
        "temperature": 0.82, "top_k": 65, "top_p": 0.88,
        "repetition_penalty": 1.10, "length_penalty": 0.95,
        "speed_rate": 1.03, "pitch_semitones": 0.0,
        "volume_scale": 1.03, "lowpass_hz": None,
        "note": "Natural happiness from generation, minimal post-processing.",
    },
    "sad": {
        "temperature": 0.72, "top_k": 55, "top_p": 0.82,
        "repetition_penalty": 1.10, "length_penalty": 1.15,
        "speed_rate": 0.95, "pitch_semitones": 0.0,
        "volume_scale": 0.95, "lowpass_hz": None,
        "note": "Natural sadness from generation, very subtle slowdown.",
    },
    "angry": {
        "temperature": 0.82, "top_k": 65, "top_p": 0.88,
        "repetition_penalty": 1.15, "length_penalty": 0.95,
        "speed_rate": 1.05, "pitch_semitones": 0.0,
        "volume_scale": 1.08, "lowpass_hz": None,
        "note": "Natural anger from generation, slightly faster/louder.",
    },
    "excited": {
        "temperature": 0.85, "top_k": 70, "top_p": 0.90,
        "repetition_penalty": 1.10, "length_penalty": 0.90,
        "speed_rate": 1.05, "pitch_semitones": 0.0,
        "volume_scale": 1.05, "lowpass_hz": None,
        "note": "Natural excitement from generation, slightly faster.",
    },
    "calm": {
        "temperature": 0.68, "top_k": 50, "top_p": 0.80,
        "repetition_penalty": 1.05, "length_penalty": 1.10,
        "speed_rate": 0.97, "pitch_semitones": 0.0,
        "volume_scale": 0.95, "lowpass_hz": None,
        "note": "Calm and relaxed.",
    },
    "whisper": {
        "temperature": 0.68, "top_k": 50, "top_p": 0.80,
        "repetition_penalty": 1.05, "length_penalty": 1.10,
        "speed_rate": 0.95, "pitch_semitones": 0.0,
        "volume_scale": 0.70, "lowpass_hz": 4000,
        "note": "Soft whisper via lowpass and volume reduction only.",
    },
    "fast": {
        "temperature": 0.70, "top_k": 60, "top_p": 0.85,
        "repetition_penalty": 1.10, "length_penalty": 1.00,
        "speed_rate": 1.12, "pitch_semitones": 0.0,
        "volume_scale": 1.0, "lowpass_hz": None,
        "note": "Faster speech, no pitch change.",
    },
    "slow": {
        "temperature": 0.70, "top_k": 60, "top_p": 0.85,
        "repetition_penalty": 1.10, "length_penalty": 1.10,
        "speed_rate": 0.88, "pitch_semitones": 0.0,
        "volume_scale": 1.0, "lowpass_hz": None,
        "note": "Slower speech, no pitch change.",
    },
    "laugh": {
        "temperature": 0.80, "top_k": 70, "top_p": 0.90,
        "repetition_penalty": 1.10, "length_penalty": 0.95,
        "speed_rate": 1.06, "pitch_semitones": 1.0,
        "volume_scale": 1.05, "lowpass_hz": None,
        "reference": "laugh",
        "text_override": "Hahaha!",
        "note": "Laugh is experimental. Uses laugh/ clips if present, else happy approximation.",
    },
    "cry": {
        "temperature": 0.65, "top_k": 50, "top_p": 0.80,
        "repetition_penalty": 1.10, "length_penalty": 1.20,
        "speed_rate": 0.90, "pitch_semitones": -1.0,
        "volume_scale": 0.92, "lowpass_hz": None,
        "reference": "cry",
        "text_override": "Oh nein... schluchz...",
        "note": "Cry is experimental. Uses cry/ clips if present, else sad approximation.",
    },
}


def list_emotions():
    return sorted(EMOTION_PRESETS)


def get_emotion_preset(name):
    return EMOTION_PRESETS.get((name or "neutral").strip().lower(), EMOTION_PRESETS["neutral"])


def emotion_clip_dir(emotion):
    return EMOTION_REFERENCE_DIR / emotion


def best_emotion_clip(emotion):
    """Return the best WAV in data/audio/reference/emotions/<emotion>/ or None."""
    d = emotion_clip_dir(emotion)
    if not d.is_dir():
        return None
    files = sorted(d.glob("*.wav"))
    if not files:
        return None
    best = None
    best_score = -1.0
    for p in files:
        try:
            data, sr = sf.read(str(p), dtype="float32")
            if data.ndim > 1:
                data = data.mean(axis=1)
            dur = len(data) / sr
            if not (2.0 <= dur <= 20.0):
                continue
            if float(np.abs(data).max()) >= 0.99:
                continue
            rms = np.sqrt((data ** 2).mean())
            dur_score = max(0.0, 1.0 - abs(dur - 8.0) / 8.0)
            score = dur_score + min(float(rms), 0.5)
            if score > best_score:
                best_score = score
                best = p
        except Exception:
            continue
    return best


def normalize_peak(audio, peak=0.8):
    audio = np.asarray(audio, dtype="float32")
    m = float(np.abs(audio).max())
    if m > 1e-9 and m != peak:
        audio = audio * (peak / m)
    return audio


def apply_emotion_audio(input_path, emotion_name, output_path=None,
                        speed=None, pitch=None):
    """Post-process a WAV for an emotion.

    Steps (in order): time stretch, pitch shift, lowpass, volume, normalize.
    speed/pitch kwargs override the preset values.
    Neutral with no overrides only re-normalizes the peak (like polish).
    Default: overwrite input_path.
    """
    input_path = Path(input_path)
    output_path = Path(output_path) if output_path else input_path
    preset = get_emotion_preset(emotion_name)

    speed_rate = preset["speed_rate"] if speed is None else float(speed)
    semitones = preset["pitch_semitones"] if pitch is None else float(pitch)

    data, sr = sf.read(str(input_path), dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)

    changed = abs(speed_rate - 1.0) > 1e-6 or abs(semitones) > 1e-6 \
        or preset["lowpass_hz"] is not None or preset["volume_scale"] != 1.0

    if abs(speed_rate - 1.0) > 1e-6:
        data = librosa.effects.time_stretch(data, rate=speed_rate)
    if abs(semitones) > 1e-6:
        data = librosa.effects.pitch_shift(data, sr=sr, n_steps=semitones)
    if preset["lowpass_hz"]:
        sos = butter(4, preset["lowpass_hz"] / (sr / 2.0), btype="low",
                     output="sos")
        data = sosfiltfilt(sos, data, padlen=None)
    if preset["volume_scale"] != 1.0:
        data = data.astype("float32") * float(preset["volume_scale"])
    data = normalize_peak(data, 0.8)

    sf.write(str(output_path), data.astype("float32"), sr, subtype="PCM_16")
    return str(output_path)