"""CAMPPlus target-speaker embedding similarity for the 6 scenario outputs.

Identical finalized methodology and reference set as the XTTS metrics
package (tts_chapter7_metrics_FINAL): FunASR CAMPPlus class
(funasr.models.campplus.model.CAMPPlus) + locally cached official
checkpoint campplus_cn_common.bin (192-dim, 16 kHz fbank).
Target set = the same eight private authentic reference clips used in the final evaluation.

Descriptive only: no threshold, no identity decision, no claim about human
recognition. Absolute values must not be compared numerically with any
other evaluation using a different reference set.
"""

import csv
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from funasr.models.campplus.model import CAMPPlus
from funasr.models.campplus.utils import extract_feature
from funasr.utils.load_utils import load_audio_text_image_video

# Publication paths. Private reference audio and the CAMPPlus checkpoint are
# intentionally not distributed. Set these environment variables locally.
def required_env(name):
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Set the {name} environment variable before running this script.")
    return value

OUTPUT_ROOT = Path(os.environ.get("XTTS_OUTPUT_DIR", str(Path(__file__).resolve().parent / "run_output"))).resolve()
PRIVATE_REFERENCE_DIR = Path(required_env("XTTS_PRIVATE_REFERENCE_DIR"))
CAMPPLUS_CHECKPOINT = Path(required_env("CAMPPLUS_CHECKPOINT"))

STAGE = OUTPUT_ROOT
AUDIO_DIR = STAGE / "05_AUDIO"
METRICS_DIR = STAGE / "02_METRICS"
CKPT = CAMPPLUS_CHECKPOINT

# Same finalized target set as the XTTS metrics package
TARGET_CLIPS = [
    "audio1_chunk_046.wav",
    "audio3_chunk_074.wav",
    "audio3_chunk_026.wav",
    "audio1_chunk_079.wav",
    "audio1_chunk_045.wav",
    "audio2_chunk_058.wav",
    "audio1_chunk_081.wav",
    "audio2_chunk_052.wav",
]

FILES = [
    ("S1", 1), ("S1", 2), ("S1", 3),
    ("S2", 1), ("S2", 2), ("S2", 3),
]


def embed(model, path):
    wav = load_audio_text_image_video(str(path), fs=16000, audio_fs=16000, data_type="sound")
    speech, speech_lengths, speech_times = extract_feature([wav])
    with torch.no_grad():
        e = model.forward(speech.float().cuda())[0].detach().cpu().numpy().reshape(-1)
    e = e / (np.linalg.norm(e) + 1e-12)
    return e


def main():
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    print("loading CAMPPlus (funasr, campplus_cn_common.bin)")
    model = CAMPPlus()
    sd = torch.load(str(CKPT), map_location="cpu", weights_only=False)
    model.load_state_dict(sd)
    model.eval().cuda()

    ref_dir = PRIVATE_REFERENCE_DIR
    embs = np.stack([embed(model, ref_dir / c) for c in TARGET_CLIPS])
    centroid = embs.mean(axis=0)
    centroid = centroid / (np.linalg.norm(centroid) + 1e-12)
    loo = []
    for i in range(len(embs)):
        c = embs[np.arange(len(embs)) != i].mean(axis=0)
        c = c / (np.linalg.norm(c) + 1e-12)
        loo.append(float(np.dot(embs[i], c)))
    loo = np.array(loo)
    print(f"target set: n={len(embs)} LOO mean={loo.mean():.4f} "
          f"std={loo.std():.4f} min={loo.min():.4f} max={loo.max():.4f}")

    rows = []
    for sid, run in FILES:
        name = f"{sid}_XTTS_run{run:02d}.wav"
        path = AUDIO_DIR / name
        sim = float(np.dot(embed(model, path), centroid))
        dur = round(sf.info(str(path)).duration, 3)
        rows.append({
            "output_filename": name, "scenario": sid, "run": run,
            "output_duration_s": dur,
            "campplus_similarity": round(sim, 4),
            "reference_set": "fixed eight-clip private XTTS target reference set",
        })
        print(f"{name}: sim={sim:.4f} dur={dur}s")

    with open(METRICS_DIR / "xtts_scenario_similarity.csv", "w", newline="",
              encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote", METRICS_DIR / "xtts_scenario_similarity.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())