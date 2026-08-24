"""seedvc_scenario_run.py - run the FIXED Seed-VC configuration (100 steps, cfg 1.0,
temp 0.7, top_p 0.7, rep_pen 1.5, AR-EOS min_tokens 0.97x) over the 6 scenario
sources in ONE process (model loaded once) and write per-file timing.

Exact code path of the locked-evidence runner exec3_run.py.
"""
import csv
import json
import os
import sys
import time

import soundfile as sf
import torch

def required_env(name):
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Set the {name} environment variable before running this script.")
    return value

REPO = required_env("SEEDVC_REPO")
SCENARIO_WS = required_env("SCENARIO_WS")
sys.path.insert(0, REPO)
os.chdir(REPO)

import yaml
from omegaconf import DictConfig
from hydra.utils import instantiate
import librosa

SRC_DIR = os.path.join(SCENARIO_WS, "01_SOURCE")
OUT_DIR = os.path.join(SCENARIO_WS, "03_SEEDVC")
REF = required_env("SEEDVC_TARGET_REFERENCE")
STEPS = 100

INPUTS = {
    "S1_executive_take01.wav": "S1_SEEDVC_take01.wav",
    "S1_executive_take02.wav": "S1_SEEDVC_take02.wav",
    "S1_executive_take03.wav": "S1_SEEDVC_take03.wav",
    "S2_customer_take01.wav": "S2_SEEDVC_take01.wav",
    "S2_customer_take02.wav": "S2_SEEDVC_take02.wav",
    "S2_customer_take03.wav": "S2_SEEDVC_take03.wav",
}

cfg = DictConfig(yaml.safe_load(open("configs/v2/vc_wrapper.yaml", "r")))
vc = instantiate(cfg)
vc.load_checkpoints()
vc.to("cuda")
vc.eval()
vc.setup_ar_caches(max_batch_size=1, max_seq_len=4096, dtype=torch.float32, device=torch.device("cuda"))


@torch.no_grad()
def convert(src, ref, steps=100, cfg_rate=1.0, top_p=0.7, temperature=0.7, min_tokens=0):
    device = torch.device("cuda")
    source_wave = librosa.load(src, sr=vc.sr)[0]
    target_wave = librosa.load(ref, sr=vc.sr)[0]
    source_wave_tensor = torch.tensor(source_wave).unsqueeze(0).to(device)
    target_wave_tensor = torch.tensor(target_wave).unsqueeze(0).to(device)
    source_wave_16k = librosa.resample(source_wave, orig_sr=vc.sr, target_sr=16000)
    target_wave_16k = librosa.resample(target_wave, orig_sr=vc.sr, target_sr=16000)
    source_wave_16k_tensor = torch.tensor(source_wave_16k).unsqueeze(0).to(device)
    target_wave_16k_tensor = torch.tensor(target_wave_16k).unsqueeze(0).to(device)
    source_mel = vc.mel_fn(source_wave_tensor)
    target_mel = vc.mel_fn(target_wave_tensor)
    source_mel_len = source_mel.size(2)
    target_mel_len = target_mel.size(2)
    with torch.autocast(device_type=device.type, dtype=torch.float32):
        _, source_content_indices, _ = vc.content_extractor_wide(source_wave_16k_tensor, [source_wave_16k.size])
        _, target_content_indices, _ = vc.content_extractor_wide(target_wave_16k_tensor, [target_wave_16k.size])
        _, source_narrow_indices, _ = vc.content_extractor_narrow(source_wave_16k_tensor,
            [source_wave_16k.size], ssl_model=vc.content_extractor_wide.ssl_model)
        _, target_narrow_indices, _ = vc.content_extractor_narrow(target_wave_16k_tensor,
            [target_wave_16k.size], ssl_model=vc.content_extractor_wide.ssl_model)
        src_narrow_reduced, src_narrow_len = vc.duration_reduction_func(source_narrow_indices[0], 1)
        tgt_narrow_reduced, tgt_narrow_len = vc.duration_reduction_func(target_narrow_indices[0], 1)
        ar_cond = vc.ar_length_regulator(torch.cat([tgt_narrow_reduced, src_narrow_reduced], dim=0)[None])[0]
        expected_tokens = int(source_content_indices.size(-1))
        if min_tokens == 0:
            min_tokens = int(expected_tokens * 0.97)
        ar_out = vc.ar.generate(ar_cond, target_content_indices, top_p=top_p, temperature=temperature,
                                repetition_penalty=1.5, min_tokens=min_tokens)
        ar_out_mel_len = torch.LongTensor([int(source_mel_len / expected_tokens * ar_out.size(-1) * 1.0)]).to(device)
        target_style = vc.compute_style(target_wave_16k_tensor)
        cond, _ = vc.cfm_length_regulator(ar_out, ylens=torch.LongTensor([ar_out_mel_len]).to(device))
        prompt_condition, _ = vc.cfm_length_regulator(target_content_indices, ylens=torch.LongTensor([target_mel_len]).to(device))
        cat_condition = torch.cat([prompt_condition, cond], dim=1)
        vc_mel = vc.cfm.inference(cat_condition, torch.LongTensor([cat_condition.size(1)]).to(device),
                                  target_mel, target_style, n_timesteps=steps,
                                  inference_cfg_rate=[cfg_rate, cfg_rate])
    vc_mel = vc_mel[:, :, target_mel_len:]
    vc_wave = vc.vocoder(vc_mel.float()).squeeze()
    return vc_wave.cpu().numpy()


def main():
    rows = []
    for src_name, out_name in INPUTS.items():
        src_path = os.path.join(SRC_DIR, src_name)
        out_path = os.path.join(OUT_DIR, out_name)
        src_dur = librosa.get_duration(path=src_path)
        t0 = time.time()
        wav = convert(src_path, REF, steps=STEPS)
        dt = time.time() - t0
        out_dur = len(wav) / vc.sr
        sf.write(out_path, wav, vc.sr)
        rows.append({"file": out_name, "source_file": src_name,
                     "processing_time_s": round(dt, 3), "source_duration_s": round(src_dur, 3),
                     "output_duration_s": round(out_dur, 3), "rtf": round(dt / src_dur, 4),
                     "status": "ok", "output_sr": vc.sr,
                     "model_loaded": True, "preprocessing_included": True})
        print("%s -> %s  %.1fs  out=%.2fs  RTF=%.2f" % (src_name, out_name, dt, out_dur, dt / src_dur),
              flush=True)

    timing_csv = os.path.join(OUT_DIR, "seedvc_timing.csv")
    with open(timing_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(OUT_DIR, "seedvc_run_config.json"), "w", encoding="utf-8") as f:
        json.dump({"repo": REPO, "wrapper": "configs/v2/vc_wrapper.yaml",
                   "reference": REF, "steps": STEPS, "cfg_rate": 1.0, "top_p": 0.7,
                   "temperature": 0.7, "repetition_penalty": 1.5,
                   "min_tokens": "int(expected_tokens*0.97)"}, f, indent=2)
    print("WROTE", timing_csv)


if __name__ == "__main__":
    main()