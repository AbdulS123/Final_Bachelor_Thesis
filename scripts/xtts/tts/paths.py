"""Project path definitions."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT_DIR / "config"
DATA_DIR = ROOT_DIR / "data"
AUDIO_DIR = DATA_DIR / "audio"
REFERENCE_AUDIO_DIR = AUDIO_DIR / "reference"
INPUT_AUDIO_DIR = AUDIO_DIR / "inputs"
OUTPUT_AUDIO_DIR = AUDIO_DIR / "outputs"
MODELS_DIR = ROOT_DIR / "models"
QWEN3_MODEL_DIR = MODELS_DIR / "qwen3"
COSYVOICE_MODEL_DIR = MODELS_DIR / "cosyvoice"
LOGS_DIR = ROOT_DIR / "logs"
