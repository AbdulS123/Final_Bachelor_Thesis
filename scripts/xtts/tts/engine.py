"""TTS engine with singleton backend loading.

Engine holds a single loaded backend (XTTS-v2 by default) so that a
persistent server can keep the model in memory across requests.
"""

import time
from pathlib import Path

from tts.config import load_config
from tts.backends import BACKENDS
from tts.paths import OUTPUT_AUDIO_DIR, REFERENCE_AUDIO_DIR, ROOT_DIR

DEFAULT_LANGUAGE = "de"
DEFAULT_BACKEND = "xtts"


def _load_models_yaml():
    models_file = ROOT_DIR / "config" / "models.yaml"
    try:
        import yaml

        with open(models_file, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def _read_main_voice():
    path = REFERENCE_AUDIO_DIR / "selected_main_voice.txt"
    if path.is_file():
        content = path.read_text(encoding="utf-8").strip()
        if content:
            return content
    return None


class TTSEngine:
    def __init__(self, config=None, backend=None, language=DEFAULT_LANGUAGE):
        self.config = config or load_config()
        self.models_config = _load_models_yaml()
        self.backend_name = backend or self.models_config.get(
            "selected_backend", DEFAULT_BACKEND
        )
        self.language = language
        model_cfg = self.models_config.get("backends", {}).get(self.backend_name, {})
        self.model = model_cfg.get("model_name", None)
        self.backend = None
        self.loaded = False
        self.load_time = None
        self.last_generation_time = None
        self.device = None
        self.message = "Engine created, backend not loaded."
        self.main_speaker = _read_main_voice()

    def get_backend(self):
        """Return the backend instance, creating it if needed."""
        if self.backend is None:
            cls = BACKENDS.get(self.backend_name)
            if cls is None:
                raise RuntimeError(f"Backend '{self.backend_name}' is not registered.")
            self.backend = cls(config=self.config)
            self.backend.set_main_speaker(self.main_speaker)
        return self.backend

    def load(self):
        """Load the backend (load once, keep in memory) and set up speaker."""
        backend = self.get_backend()
        if self.loaded:
            return self
        if self.main_speaker:
            backend.set_main_speaker(self.main_speaker)
        backend.load()
        self.loaded = True
        self.load_time = backend.load_time
        self.device = backend.device_used
        self.message = "Backend loaded."
        return self

    def warmup(self, text=None):
        backend = self.get_backend()
        backend.warmup(text=text, output_dir=OUTPUT_AUDIO_DIR)
        self.last_generation_time = backend.last_generation_time
        return self

    def synthesize(self, text, output_path=None, language=None, speaker_wav=None,
                   emotion="neutral", speed=None, pitch=None, **params):
        backend = self.get_backend()
        if not self.loaded:
            self.load()
        language = language or self.language or DEFAULT_LANGUAGE
        if output_path is None:
            stamp = time.strftime("%Y%m%d_%H%M%S")
            output_path = OUTPUT_AUDIO_DIR / f"tts_{stamp}.wav"
        out = backend.synthesize(text, output_path=output_path,
                                 speaker_wav=speaker_wav, language=language,
                                 emotion=emotion, speed=speed, pitch=pitch,
                                 **params)
        self.last_generation_time = backend.last_generation_time
        return out

    def info(self):
        status = "loaded" if self.loaded else ("backend_unavailable" if self.backend is not None and (self.backend.error is not None) else "not_loaded")
        if self.loaded:
            status = "loaded"
        elif self.backend is not None and self.backend.error:
            status = "error"
        return {
            "status": status,
            "backend": self.backend_name,
            "language": self.language,
            "model": self.model,
            "loaded": self.loaded,
            "device": self.device,
            "load_time": self.load_time,
            "last_generation_time": self.last_generation_time,
            "main_speaker": self.main_speaker,
            "message": self.message,
        }