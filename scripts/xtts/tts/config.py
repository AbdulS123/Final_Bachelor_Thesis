"""Load config/default.yaml safely with a built-in fallback."""

from pathlib import Path

from tts.paths import CONFIG_DIR, OUTPUT_AUDIO_DIR, REFERENCE_AUDIO_DIR, INPUT_AUDIO_DIR

DEFAULT_CONFIG = {
    "project_name": "tts-studio",
    "selected_backend": "xtts",
    "backends": {
        "xtts": {"enabled": True, "model_path": "models/xtts", "device": "auto"},
    },
    "audio": {
        "sample_rate": 24000,
        "output_dir": "data/audio/outputs",
        "reference_dir": "data/audio/reference",
        "input_dir": "data/audio/inputs",
    },
    "logging": {"level": "INFO", "file": "logs/tts.log"},
}


def _deep_merge(base, override):
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path=None):
    """Load the YAML config, falling back to defaults if pyyaml is missing or fails."""
    cfg = _deep_merge(DEFAULT_CONFIG, None)
    config_file = Path(path) if path else CONFIG_DIR / "default.yaml"

    if not config_file.is_file():
        return cfg

    try:
        import yaml

        with open(config_file, "r", encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh)
        if isinstance(loaded, dict):
            cfg = _deep_merge(cfg, loaded)
    except ImportError:
        pass
    except Exception:
        pass
    return cfg


def ensure_dirs(config=None):
    """Create runtime directories from the config."""
    cfg = config or load_config()
    dirs = [
        OUTPUT_AUDIO_DIR,
        REFERENCE_AUDIO_DIR,
        INPUT_AUDIO_DIR,
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    return cfg
