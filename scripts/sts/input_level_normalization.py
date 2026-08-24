"""make_normalized_sources.py - INPUT-LEVEL SENSITIVITY CHECK, step 1.

Creates gain-only copies of the six scenario originals at approximately -28 LUFS
integrated. CONSTANT GAIN ONLY: no compression, dynamic normalization, limiting,
denoising, EQ, trimming, enhancement, gating, or resynthesis. Content, timing,
sample rate (48 kHz), channels (mono), and encoding (PCM16) are preserved.

- Integrated loudness: ITU-R BS.1770-4 via pyloudnorm (Meter(48000), standard K-weighting).
- RMS: whole-file RMS in dBFS (10*log10(mean(x^2))), reported as context.
- Peak: dBFS.
- Gain applied: gain_db = -28 - original_LUFS, capped so the output peak never
  exceeds -0.1 dBFS (no clipping). If capped, the resulting LUFS is documented.
- Output: WAV PCM16 48 kHz mono to SOURCE_AUDIO_DIRECTORY_NORMALIZED.
- CSV: input_level_normalization.csv (this folder) with the required columns.
"""
import csv
import hashlib
import os

import numpy as np
import pyloudnorm as pyln
import soundfile as sf

ORIG_DIR = os.environ.get("SOURCE_AUDIO_DIRECTORY")
NORM_DIR = os.environ.get("SOURCE_AUDIO_DIRECTORY_NORMALIZED")
INPUT_LEVEL_WS = os.environ.get("INPUT_LEVEL_WS")
if not ORIG_DIR or not NORM_DIR or not INPUT_LEVEL_WS:
    raise RuntimeError(
        "Set SOURCE_AUDIO_DIRECTORY, SOURCE_AUDIO_DIRECTORY_NORMALIZED, and INPUT_LEVEL_WS before running this script."
    )
OUT_CSV = os.path.join(INPUT_LEVEL_WS, "01_NORMALIZED_SOURCE", "input_level_normalization.csv")

TARGET_LUFS = -28.0
PEAK_CEILING_DB = -0.1

FILES = [
    "S1_executive_take01.wav", "S1_executive_take02.wav", "S1_executive_take03.wav",
    "S2_customer_take01.wav", "S2_customer_take02.wav", "S2_customer_take03.wav",
]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rms_db(x):
    return 10.0 * np.log10(np.mean(x ** 2) + 1e-12)


def main():
    rows = []
    for name in FILES:
        src = os.path.join(ORIG_DIR, name)
        dst = os.path.join(NORM_DIR, name)
        x, sr = sf.read(src, dtype="float32", always_2d=False)
        info = sf.info(src)
        dur = info.frames / info.samplerate
        meter = pyln.Meter(sr)
        lufs_orig = meter.integrated_loudness(x)
        peak_orig = float(np.max(np.abs(x)))
        peak_db_orig = 20.0 * np.log10(peak_orig + 1e-12)

        gain_db = TARGET_LUFS - lufs_orig
        capped = False
        if peak_db_orig + gain_db > PEAK_CEILING_DB:
            gain_db = PEAK_CEILING_DB - peak_db_orig
            capped = True
        y = x * float(10.0 ** (gain_db / 20.0))
        peak_db_new = 20.0 * np.log10(float(np.max(np.abs(y))) + 1e-12)
        lufs_new = meter.integrated_loudness(y)
        sf.write(dst, y, sr, subtype="PCM_16")
        info_new = sf.info(dst)

        rows.append({
            "filename": name,
            "original_LUFS": round(lufs_orig, 2),
            "normalized_LUFS": round(lufs_new, 2),
            "original_peak_dBFS": round(peak_db_orig, 2),
            "normalized_peak_dBFS": round(peak_db_new, 2),
            "gain_applied_dB": round(gain_db, 2),
            "duration_before": round(dur, 3),
            "duration_after": round(info_new.frames / info_new.samplerate, 3),
            "SHA256_original": sha256(src),
            "SHA256_normalized": sha256(dst),
            "original_RMS_dBFS": round(rms_db(x), 2),
            "target_reached": ("yes" if not capped else "no (peak ceiling; resulting LUFS documented)"),
        })
        print("%-24s LUFS %.2f -> %.2f  peak %.2f -> %.2f dBFS  gain %+.2f dB  %s"
              % (name, lufs_orig, lufs_new, peak_db_orig, peak_db_new, gain_db,
                 "capped" if capped else "target reached"))

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("WROTE", OUT_CSV)


if __name__ == "__main__":
    main()