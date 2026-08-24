"""XTTS-v2 backend for German/Austrian voice cloning.

- Golden Speaker Embedding (models/xtts/speaker_embedding_golden.pth) is the
  locked voice source; no random reference clips are scanned at generation.
- German text normalizer (tts/text_norm_de.py) is applied to every sentence.
- Emotion control (tts/emotion.py): parameter presets + speed/pitch/volume/
  lowpass post-processing. Default emotion is "neutral" = the locked voice.
"""

import os
import time
from pathlib import Path

import numpy as np

from tts.backends.base import BaseTTSBackend

MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
DEFAULT_LANGUAGE = "de"
DEFAULT_WARMUP_TEXT = "Hallo. Dies ist ein Test."
GOLDEN_DIR = Path(
    os.environ.get(
        "XTTS_PRIVATE_MODEL_DIR",
        str(Path(__file__).resolve().parent.parent.parent / "models" / "xtts"),
    )
)
GOLDEN_PTH = GOLDEN_DIR / "speaker_embedding_golden.pth"
GOLDEN_WAV = GOLDEN_DIR / "golden_reference.wav"

DEFAULT_GEN_PARAMS = {
    "temperature": 0.75,
    "top_k": 62,
    "top_p": 0.86,
    "repetition_penalty": 1.09,
    "length_penalty": 1.05,
    "enable_text_splitting": True,
}


class XTTSBackend(BaseTTSBackend):
    name = "xtts"

    def __init__(self, config=None):
        self.config = config or {}
        self._model = None
        self.load_time = None
        self.last_generation_time = None
        self.device_used = None
        self.error = None
        self.is_warmed_up = False
        self.main_speaker_wav = None
        self.golden_reference_wav = None
        self.gpt_cond_latent = None
        self.speaker_embedding = None

    def _normalize(self, text):
        try:
            from tts.text_norm_de import normalize_german

            return normalize_german(text)
        except Exception:
            return text

    def _load_golden(self):
        """Load cached conditioning latents + golden wav path."""
        self.golden_reference_wav = str(GOLDEN_WAV) if GOLDEN_WAV.is_file() else None
        if not GOLDEN_PTH.is_file():
            print("Golden embedding .pth not found; falling back to raw wav path.")
            return
        try:
            import torch

            data = torch.load(GOLDEN_PTH, map_location="cpu")
            self.gpt_cond_latent = data["gpt_cond_latent"]
            self.speaker_embedding = data["speaker_embedding"]
            self.golden_source = data.get("source_clip", "unknown")
            print(f"Golden embedding loaded from {GOLDEN_PTH.name} "
                  f"(source: {self.golden_source})")
        except Exception as exc:
            self.gpt_cond_latent = None
            self.speaker_embedding = None
            print(f"Golden embedding load failed ({exc}); using raw wav.")

    def load(self):
        """Load XTTS model once and keep it in memory."""
        if self._model is not None:
            return self
        try:
            from TTS.api import TTS
        except ImportError as exc:
            self.error = f"TTS package not installed: {exc}"
            raise RuntimeError(self.error)

        import os

        os.environ.setdefault("COQUI_TOS_AGREED", "1")

        t0 = time.time()
        try:
            model = TTS(model_name=MODEL_NAME, gpu=True)
            device = "cuda"
        except Exception:
            try:
                model = TTS(model_name=MODEL_NAME, gpu=False)
                device = "cpu"
            except Exception as exc:
                self.error = str(exc)
                self.device_used = None
                raise

        self._model = model
        self.device_used = device
        self.load_time = time.time() - t0
        self._load_golden()
        self._move_latents_to_device()
        return self

    def _move_latents_to_device(self):
        if self.gpt_cond_latent is None or self.speaker_embedding is None:
            return
        dev = self._model.synthesizer.tts_model.device
        self.gpt_cond_latent = self.gpt_cond_latent.to(dev)
        self.speaker_embedding = self.speaker_embedding.to(dev)

    def _latents_from_wav(self, wav_path):
        """Compute conditioning latents from an emotional reference clip."""
        import soundfile as sf
        import torch

        wav_dir = Path(wav_path)
        data, sr = sf.read(str(wav_dir), dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        tts_model = self._model.synthesizer.tts_model
        audio = torch.from_numpy(np.ascontiguousarray(data)).unsqueeze(0).to(tts_model.device)
        gpt = tts_model.get_gpt_cond_latents(audio, sr)
        emb = tts_model.get_speaker_embedding(audio, sr)
        return gpt, emb

    def set_main_speaker(self, wav_path):
        p = Path(wav_path)
        if p.is_file():
            self.main_speaker_wav = str(p)
        return self.main_speaker_wav

    def warmup(self, text=None, output_dir=None):
        """Warm up: load and do one generation to prime GPU/CPU caches."""
        self.load()
        if self._model is None:
            return
        if self.is_warmed_up:
            return
        text = text or DEFAULT_WARMUP_TEXT
        output_dir = Path(output_dir or Path.cwd() / "outputs")
        output_dir.mkdir(parents=True, exist_ok=True)
        out = output_dir / "_warmup.wav"
        try:
            self.synthesize(text, output_path=str(out),
                            speaker_wav=self.main_speaker_wav,
                            language=DEFAULT_LANGUAGE)
            self.is_warmed_up = True
        finally:
            if out.exists():
                try:
                    out.unlink()
                except OSError:
                    pass

    def _emotion_prepare(self, text, emotion, params, speaker_wav):
        """Resolve emotion preset -> (gen_text, settings, latent_pair or None).

        - laugh/cry: use emotional reference clip if present (temporary
          speaker latents), else fall back to happy/sad params and append the
          text imitation.
        - preset gen params override backend defaults; explicit params win.
        """
        from tts.emotion import best_emotion_clip, get_emotion_preset

        preset = get_emotion_preset(emotion)
        settings = dict(DEFAULT_GEN_PARAMS)
        for key in ("temperature", "top_k", "top_p",
                    "repetition_penalty", "length_penalty"):
            if preset.get(key) is not None:
                settings[key] = preset[key]
        settings.update(params)

        gen_text = self._normalize(text)
        temp_latents = None
        if preset.get("reference") and not speaker_wav:
            clip = best_emotion_clip(preset["reference"])
            if clip is not None:
                temp_latents = self._latents_from_wav(clip)
                print(f"Emotion reference clip: {clip.name}")
            elif preset.get("text_override"):
                gen_text = f"{gen_text} {preset['text_override']}"
                print(f"No emotional clip for '{emotion}'; approximation used.")
        return gen_text, settings, temp_latents

    def _generate_wav(self, text, language, settings, output_path, temp_latents):
        """Generate WAV via cached latents (or temp emotion latents / wav)."""
        if temp_latents is not None:
            gpt, emb = temp_latents
        elif self.gpt_cond_latent is not None and self.speaker_embedding is not None:
            gpt, emb = self.gpt_cond_latent, self.speaker_embedding
        else:
            speaker = self.golden_reference_wav or self.main_speaker_wav
            self._model.tts_to_file(
                text=text, file_path=str(output_path),
                speaker_wav=speaker, language=language, **settings)
            return

        result = self._model.synthesizer.tts_model.inference(
            text=text, language=language,
            gpt_cond_latent=gpt, speaker_embedding=emb,
            **settings,
        )
        wav = result["wav"]
        if hasattr(wav, "detach"):
            wav = wav.detach().cpu().numpy()
        sr = int(getattr(self._model.synthesizer, "output_sample_rate", 24000))
        import soundfile as sf

        sf.write(str(output_path), np.asarray(wav).reshape(-1), sr, subtype="PCM_16")

    def _duration_anomalous(self, text, seconds, sr):
        """Detect XTTS length blow-ups (autoregressive looping)."""
        expected = len(text) / 13.0 + 1.5
        if seconds > max(expected * 3.0, 8.0):
            return True
        return False

    def synthesize(self, text, output_path=None, speaker_wav=None,
                   language=DEFAULT_LANGUAGE, emotion="neutral",
                   speed=None, pitch=None, **params):
        """Synthesize text to polished WAV using XTTS-v2.

        Order: normalize -> emotion preset -> XTTS generation with preset
        parameters -> static/duration retry with safe parameters -> normal
        polish -> emotion post-processing -> final peak normalize. Default
        emotion "neutral" = locked voice, no audio changes. speed/pitch
        override the emotion preset values.
        """
        self.load()
        if self._model is None:
            raise RuntimeError("XTTS not loaded.")
        if not text or not text.strip():
            raise ValueError("Empty text.")

        gen_text, settings, temp_latents = self._emotion_prepare(
            text, emotion, params, speaker_wav)
        output_path = Path(output_path or "tts_output.wav").resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        last_issue = None
        static_final = False
        safe_settings = dict(DEFAULT_GEN_PARAMS)
        try:
            for attempt in range(3):
                from tts.audio import detect_static_noise

                import soundfile as sf

                use_settings = dict(settings) if attempt == 0 else dict(safe_settings)
                if attempt > 0 and last_issue == "duration":
                    use_settings["temperature"] = min(
                        1.05, use_settings.get("temperature", 0.7) + 0.08)
                    use_settings["repetition_penalty"] = max(
                        use_settings.get("repetition_penalty", 1.0), 1.2)
                self._generate_wav(gen_text, language, use_settings,
                                   output_path, temp_latents)
                sr = int(getattr(self._model.synthesizer,
                                 "output_sample_rate", 24000))
                data, _ = sf.read(str(output_path), dtype="float32")
                duration = len(data) / sr
                static_here = detect_static_noise(data, sr)
                if static_here:
                    issue = "static"
                elif self._duration_anomalous(gen_text, duration, sr):
                    issue = "duration"
                else:
                    issue = None
                if issue is None:
                    break
                static_final = static_here
                if attempt < 2:
                    if issue == "static":
                        print("Static detected, retrying with safe parameters...")
                    else:
                        print(f"Anomalous duration ({duration:.1f}s), retrying "
                              f"generation ({attempt + 1}/3)...")
                    last_issue = issue
            else:
                if static_final:
                    print("Static detected in final output; returning audio anyway.")
        except RuntimeError as exc:
            if self.device_used == "cuda" and ("out of memory" in str(exc).lower()
                                               or "cuda" in str(exc).lower()):
                print("GPU generation failed, retrying on CPU once...")
                try:
                    from TTS.api import TTS

                    self._model = TTS(model_name=MODEL_NAME, gpu=False)
                    self.device_used = "cpu"
                    self.load_time = None
                    self._move_latents_to_device()
                    self._generate_wav(gen_text, language, settings,
                                       output_path, temp_latents)
                except Exception as exc2:
                    self.error = str(exc2)
                    raise
            else:
                raise

        try:
            from tts.audio import polish_audio

            polish_audio(output_path)
        except Exception as exc:
            print(f"Audio polish skipped ({exc}); raw output kept.")

        try:
            from tts.emotion import apply_emotion_audio

            preset_applied = (emotion or "neutral").strip().lower()
            apply_emotion_audio(str(output_path), preset_applied,
                                speed=speed, pitch=pitch)
        except Exception as exc:
            print(f"Emotion post-processing skipped ({exc}).")

        self.last_generation_time = time.time() - t0
        return str(output_path)

    def info(self):
        return {
            "backend": self.name,
            "model": MODEL_NAME,
            "loaded": self._model is not None,
            "device": self.device_used,
            "load_time": self.load_time,
            "last_generation_time": self.last_generation_time,
            "warmed_up": self.is_warmed_up,
            "golden_embedding": self.gpt_cond_latent is not None,
            "golden_source": getattr(self, "golden_source", None),
            "error": self.error,
        }