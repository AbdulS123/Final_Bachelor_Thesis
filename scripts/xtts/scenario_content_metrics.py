"""Content preservation metrics (WER/CER) for the 6 scenario outputs.

Identical finalized ASR protocol as the STS/XTTS evaluation:
faster-whisper large-v3-turbo (cached), beam_size=5, temperature=0,
language=de, condition_on_previous_text=False, no VAD.
Ground truth = the normalized text actually supplied to XTTS.
"""

import csv
import json
import os
import sys
from pathlib import Path

import jiwer

# Publication path. Set XTTS_OUTPUT_DIR to the private evaluation workspace.
OUTPUT_ROOT = Path(os.environ.get("XTTS_OUTPUT_DIR", str(Path(__file__).resolve().parent / "run_output"))).resolve()

STAGE = OUTPUT_ROOT
AUDIO_DIR = STAGE / "05_AUDIO"
METRICS_DIR = STAGE / "02_METRICS"
LOGS_DIR = STAGE / "07_LOGS"

ASR = "mobiuslabsgmbh/faster-whisper-large-v3-turbo"
FILES = [
    ("S1", 1), ("S1", 2), ("S1", 3),
    ("S2", 1), ("S2", 2), ("S2", 3),
]


def normalize_for_wer(text):
    t = jiwer.RemovePunctuation()(text)
    t = jiwer.ToLowerCase()(t)
    return t


def main():
    from faster_whisper import WhisperModel

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    records = json.loads((LOGS_DIR / "timing_records.json").read_text(encoding="utf-8"))
    norm = {r["output_filename"]: r["normalized_text"] for r in records}

    print("loading ASR:", ASR)
    model = WhisperModel(ASR, device="cuda", compute_type="float16")
    rows = []
    for sid, run in FILES:
        name = f"{sid}_XTTS_run{run:02d}.wav"
        path = AUDIO_DIR / name
        ref = norm[name]
        segs, _ = model.transcribe(
            str(path), beam_size=5, temperature=0, language="de",
            condition_on_previous_text=False, vad_filter=False,
        )
        hyp = " ".join(s.text.strip() for s in segs).strip()
        ref_n = normalize_for_wer(ref)
        hyp_n = normalize_for_wer(hyp)
        wm = jiwer.process_words(ref_n, hyp_n)
        cm = jiwer.process_characters(ref_n, hyp_n)
        wer = wm.wer
        cer = cm.cer
        rows.append({
            "output_filename": name, "scenario": sid, "run": run,
            "normalized_ground_truth": ref,
            "asr_transcript": hyp,
            "wer": round(wer, 4), "cer": round(cer, 4),
            "substitutions": wm.substitutions,
            "deletions": wm.deletions,
            "insertions": wm.insertions,
        })
        print(f"{name}: WER={wer:.4f} CER={cer:.4f} sub={wm.substitutions} "
              f"del={wm.deletions} ins={wm.insertions}")
        print(f"   REF: {ref}")
        print(f"   HYP: {hyp}")

    with open(METRICS_DIR / "xtts_scenario_content.csv", "w", newline="",
              encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote", METRICS_DIR / "xtts_scenario_content.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())