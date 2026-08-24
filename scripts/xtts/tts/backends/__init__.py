"""Backend registry for the published XTTS evaluation subset."""

from tts.backends.xtts import XTTSBackend

BACKENDS = {"xtts": XTTSBackend}
