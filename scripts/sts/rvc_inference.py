"""rvc_scenario_run.py - run the FIXED RVC_Audit8 configuration over the 6 scenario
sources in ONE process (model + index loaded once) and write per-file timing.

Same code path as tools/rvc/infer/cli.py main(): VC(config).get_vc(model) then
vc_single(..., pitch=2, f0_method="harvest", index_rate=0.75, protect=0.33).
"""
import os
import sys
import time
from pathlib import Path

import soundfile as sf

def required_env(name):
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Set the {name} environment variable before running this script.")
    return value

RVC_ROOT = required_env("RVC_ROOT")
SCENARIO_WS = required_env("SCENARIO_WS")
os.chdir(RVC_ROOT)
sys.path.insert(0, RVC_ROOT)
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("weight_root", required_env("RVC_MODEL_DIR"))
os.environ.setdefault("index_root", os.environ.get("RVC_LOGS_DIR", required_env("RVC_INDEX_ROOT")))
os.environ.setdefault("outside_index_root", required_env("RVC_INDEX_ROOT"))
os.environ.setdefault("rmvpe_root", required_env("RVC_RMVPE_ROOT"))

MODEL = "RVC_Audit8_e800_s52800.pth"
INDEX = os.environ.get("RVC_INDEX_FILE", "added_IVF2416_Flat_nprobe_1_RVC_Audit8_v2.index")
PITCH = 2
F0_METHOD = "harvest"
INDEX_RATE = 0.75
PROTECT = 0.33
RESAMPLE_SR = 0
RMS_MIX_RATE = 1.0
SPEAKER_ID = 0

SRC_DIR = os.path.join(SCENARIO_WS, "01_SOURCE")
OUT_DIR = os.path.join(SCENARIO_WS, "02_RVC")

INPUTS = {
    "S1_executive_take01.wav": "S1_RVC_take01.wav",
    "S1_executive_take02.wav": "S1_RVC_take02.wav",
    "S1_executive_take03.wav": "S1_RVC_take03.wav",
    "S2_customer_take01.wav": "S2_RVC_take01.wav",
    "S2_customer_take02.wav": "S2_RVC_take02.wav",
    "S2_customer_take03.wav": "S2_RVC_take03.wav",
}

import csv
import json

def create_config():
    from configs.config import Config
    original_argv = sys.argv[:]
    sys.argv = [sys.argv[0]]
    try:
        return Config()
    finally:
        sys.argv = original_argv

def main():
    from i18n.i18n import I18nAuto
    from infer.vc.modules import VC

    config = create_config()
    print("device=%s dtype=%s" % (config.device, config.dtype))
    vc = VC(config)
    t_load0 = time.time()
    vc.get_vc(MODEL)
    t_load1 = time.time()
    print("model loaded in %.1fs" % (t_load1 - t_load0), flush=True)

    rows = []
    for src_name, out_name in INPUTS.items():
        src_path = os.path.join(SRC_DIR, src_name)
        out_path = os.path.join(OUT_DIR, out_name)
        src_dur = sf.info(src_path).frames / sf.info(src_path).samplerate
        t0 = time.time()
        status, result = vc.vc_single(SPEAKER_ID, src_path, PITCH, F0_METHOD, INDEX,
                                      INDEX_RATE, RESAMPLE_SR, RMS_MIX_RATE, PROTECT)
        dt = time.time() - t0
        if not result or result[0] is None or result[1] is None:
            print("%s FAILED status=%r" % (src_name, status))
            rows.append({"file": out_name, "source_file": src_name,
                         "processing_time_s": round(dt, 3), "source_duration_s": round(src_dur, 3),
                         "output_duration_s": "", "rtf": "", "status": str(status),
                         "output_sr": "", "model_loaded": True, "preprocessing_included": True})
            continue
        out_sr, audio = result
        sf.write(out_path, audio, out_sr)
        out_dur = len(audio) / out_sr
        rows.append({"file": out_name, "source_file": src_name,
                     "processing_time_s": round(dt, 3), "source_duration_s": round(src_dur, 3),
                     "output_duration_s": round(out_dur, 3), "rtf": round(dt / src_dur, 4),
                     "status": str(status), "output_sr": out_sr,
                     "model_loaded": True, "preprocessing_included": True})
        print("%s -> %s  %.1fs  out=%.2fs  RTF=%.2f" % (src_name, out_name, dt, out_dur, dt / src_dur),
              flush=True)

    timing_csv = os.path.join(OUT_DIR, "rvc_timing.csv")
    with open(timing_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(OUT_DIR, "rvc_run_config.json"), "w", encoding="utf-8") as f:
        json.dump({"model": MODEL, "index": INDEX, "pitch": PITCH, "f0_method": F0_METHOD,
                   "index_rate": INDEX_RATE, "protect": PROTECT, "resample_sr": RESAMPLE_SR,
                   "rms_mix_rate": RMS_MIX_RATE, "speaker_id": SPEAKER_ID,
                   "model_load_s": round(t_load1 - t_load0, 3)}, f, indent=2)
    print("WROTE", timing_csv)

if __name__ == "__main__":
    main()