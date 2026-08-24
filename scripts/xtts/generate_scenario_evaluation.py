"""FINAL scenario-linked XTTS evaluation - generation (exactly 6 runs).

Uses the unchanged final XTTS configuration (models/xtts golden embedding,
TTS 0.22.0, tts_models/multilingual/multi-dataset/xtts_v2, language de,
fixed inference parameters). Timing methodology identical to
scripts/latency_benchmark.py: wall-clock around engine.synthesize().

Behavior:
- exactly 2 texts x 3 independent runs = 6 outputs, fixed order
- no cherry-picking, no manual regeneration, no removal of failures
- engine-internal automatic retries (duration guard / static guard) are
  recorded (initial result -> retry -> final result, reason)
- every run logged below XTTS_OUTPUT_DIR/07_LOGS/xtts_scenario_generation.log
"""

import contextlib
import io
import json
import os
import sys
import time
from pathlib import Path

import soundfile as sf

# Publication paths. The code and evaluation logic are unchanged; private
# conditioning material is supplied separately through XTTS_PRIVATE_MODEL_DIR.
PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = Path(os.environ.get("XTTS_OUTPUT_DIR", str(PROJECT_ROOT / "run_output"))).resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from tts.config import load_config
from tts.engine import TTSEngine
from tts.text_norm_de import normalize_german

SCENARIOS = [
    ("S1", "executive", "Ich bin gerade zwischen zwei Terminen und die Zahlung muss heute noch bearbeitet werden. Die schriftliche Bestätigung reiche ich direkt danach nach."),
    ("S2", "customer", "Ich stehe gerade in der Apotheke und die Bestätigung in der App funktioniert nicht. Bitte helfen Sie mir mit der gestoppten Zahlung."),
]
RUNS = [1, 2, 3]

AUDIO_DIR = OUTPUT_ROOT / "05_AUDIO"
LOGS_DIR = OUTPUT_ROOT / "07_LOGS"
LOG_FILE = LOGS_DIR / "xtts_scenario_generation.log"


def word_count(text):
    return len([w for w in text.split() if w.strip()])


def main():
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log = open(LOG_FILE, "w", encoding="utf-8")
    tee = Tee(sys.stdout, log)

    tee.write("FINAL scenario-linked technical evaluation - generation log")
    tee.write(f"project: {PROJECT_ROOT}")
    tee.write(f"python: {sys.version.split()[0]}")
    try:
        import TTS, torch
        tee.write(f"TTS {TTS.__version__}, torch {torch.__version__}, "
                  f"cuda={torch.cuda.is_available()} "
                  f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''}")
    except Exception as exc:
        tee.write(f"env info failed: {exc}")

    tee.write("loading engine (final config)...")
    engine = TTSEngine(config=load_config())
    engine.load()
    tee.write(f"model load time: {engine.load_time:.2f} s (device: {engine.device})")

    engine.warmup()
    tee.write(f"warmup generation: {engine.last_generation_time:.3f} s "
              f"(not part of the measured runs)")

    records = []
    for sid, name, text in SCENARIOS:
        norm = normalize_german(text)
        tee.write("")
        tee.write(f"== {sid} ({name}) ==")
        tee.write(f"  original   : {text}")
        tee.write(f"  normalized : {norm}")
        tee.write(f"  words      : {word_count(norm)}")
        tee.write(f"  guard      : duration anomalous if > "
                  f"{max((len(norm)/13.0+1.5)*3.0, 8.0):.2f}s")
        for run in RUNS:
            out = AUDIO_DIR / f"{sid}_XTTS_run{run:02d}.wav"
            capture = io.StringIO()
            tee.write(f"-- {out.name} --")
            start = time.time()
            with contextlib.redirect_stdout(capture):
                engine.synthesize(text, output_path=str(out), language="de")
            gen_time = time.time() - start
            captured = capture.getvalue()
            if captured.strip():
                tee.write("    [engine log] " + captured.strip().replace("\n", "\n    [engine log] "))
            dur = sf.info(str(out)).duration
            rtf = gen_time / dur if dur > 0 else 0.0

            retry_events = []
            for line in captured.splitlines():
                l = line.strip()
                if "Anomalous duration" in l:
                    retry_events.append({"type": "duration_guard", "message": l})
                elif "Static detected" in l:
                    retry_events.append({"type": "static_guard", "message": l})
                elif "retrying" in l and retry_events and l not in [e["message"] for e in retry_events]:
                    retry_events[-1]["message"] += f" | {l}"

            status = "CLEAN"
            if any("duration_guard" == e["type"] for e in retry_events):
                status = "ANOMALY_RETRY"
            elif any("static_guard" == e["type"] for e in retry_events):
                status = "STATIC_RETRY"

            records.append({
                "scenario": sid, "scenario_name": name, "run": run,
                "input_text": text, "normalized_text": norm,
                "generation_time_s": round(gen_time, 4),
                "output_duration_s": round(dur, 4),
                "rtf": round(rtf, 4),
                "model_already_loaded": True,
                "anomaly_status": status,
                "retry_events": retry_events,
                "output_filename": out.name,
            })
            tee.write(f"    gen {gen_time:.3f} s | audio {dur:.2f} s | "
                      f"RTF {rtf:.2f} | status {status} | "
                      f"retries {len(retry_events)}")

    with open(LOGS_DIR / "timing_records.json", "w", encoding="utf-8") as fh:
        json.dump(records, fh, ensure_ascii=False, indent=2)

    tee.write("")
    tee.write("SUMMARY")
    for r in records:
        tee.write(f"{r['scenario']} run{r['run']}: gen {r['generation_time_s']} s, "
                  f"audio {r['output_duration_s']} s, RTF {r['rtf']}, "
                  f"status {r['anomaly_status']}, retries {len(r['retry_events'])}")
    log.close()
    print("wrote", LOG_FILE)
    print("wrote", LOGS_DIR / "timing_records.json")
    return 0


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            try:
                st.write(s + "\n")
            except Exception:
                pass
        for st in self.streams:
            try:
                st.flush()
            except Exception:
                pass


if __name__ == "__main__":
    sys.exit(main())