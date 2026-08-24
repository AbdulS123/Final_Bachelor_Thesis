"""CAMPPlus similarity measurements for the final STS scenario evaluation.

The final target centroid is derived from the same ten private, L2-normalized
target embeddings used in the evaluation. The private embeddings and the
resulting centroid vector are not written to or distributed with this repository.
"""
import csv
import json
import os

import librosa
import numpy as np
import soundfile as sf
import torch

import sys

def required_env(name):
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Set the {name} environment variable before running this script.")
    return value

REPO = required_env("SEEDVC_REPO")
WS = required_env("SCENARIO_WS")
PRIVATE_EMBEDDINGS_JSON = required_env("STS_PRIVATE_TARGET_EMBEDDINGS_JSON")
os.chdir(REPO)
sys.path.insert(0, REPO)
from modules.campplus.DTDNN import CAMPPlus  # noqa: E402

EMB_JSON = PRIVATE_EMBEDDINGS_JSON
OUT_CSV = os.path.join(WS, "04_METRICS", "sts_scenario_similarity.csv")

CKPT = os.path.join(REPO, "checkpoints", "models--funasr--campplus", "snapshots",
                    "e4b6ede7ce16997aff4ae69fbca1f0175e2afede", "campplus_cn_common.bin")

FILES = []
for take in (1, 2, 3):
    FILES.append(("S1", take, "S1_executive_take%02d.wav" % take, "SOURCE",
                  os.path.join(WS, "01_SOURCE", "S1_executive_take%02d.wav" % take)))
    FILES.append(("S2", take, "S2_customer_take%02d.wav" % take, "SOURCE",
                  os.path.join(WS, "01_SOURCE", "S2_customer_take%02d.wav" % take)))
for take in (1, 2, 3):
    FILES.append(("S1", take, "S1_RVC_take%02d.wav" % take, "RVC_Audit8_fixed",
                  os.path.join(WS, "02_RVC", "S1_RVC_take%02d.wav" % take)))
    FILES.append(("S2", take, "S2_RVC_take%02d.wav" % take, "RVC_Audit8_fixed",
                  os.path.join(WS, "02_RVC", "S2_RVC_take%02d.wav" % take)))
for take in (1, 2, 3):
    FILES.append(("S1", take, "S1_SEEDVC_take%02d.wav" % take, "SeedVC_fixed",
                  os.path.join(WS, "03_SEEDVC", "S1_SEEDVC_take%02d.wav" % take)))
    FILES.append(("S2", take, "S2_SEEDVC_take%02d.wav" % take, "SeedVC_fixed",
                  os.path.join(WS, "03_SEEDVC", "S2_SEEDVC_take%02d.wav" % take)))

device = "cuda" if torch.cuda.is_available() else "cpu"
model = CAMPPlus(feat_dim=80, embedding_size=192)
model.load_state_dict(torch.load(CKPT, map_location="cpu"))
model.eval().to(device)


def embed(path):
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    if sr != 16000:
        x = librosa.resample(x, orig_sr=sr, target_sr=16000).astype(np.float32)
    spec = librosa.stft(x, n_fft=512, hop_length=160, win_length=512, center=True)
    mag = np.abs(spec)
    mel = librosa.filters.mel(sr=16000, n_fft=512, n_mels=80, fmin=0, fmax=8000)
    m = (mel @ mag).T
    m = np.log(np.clip(m, 1e-5, None))
    t = torch.from_numpy(m[None]).float().to(device)
    with torch.no_grad():
        e = model(t)[0].cpu().numpy()
    return e / (np.linalg.norm(e) + 1e-9)


def cos(a, b):
    return float(np.dot(a, b))


def main():
    locked = json.load(open(EMB_JSON, encoding="utf-8"))
    names = list(locked["target_embeddings"].keys())
    cent = sum(np.array(locked["target_embeddings"][n]) for n in names)
    cent = cent / np.linalg.norm(cent)

    src_emb = {}
    rows = []
    for scenario, take, name, kind, path in FILES:
        e = embed(path)
        d = sf.info(path)
        row = {"file": name, "model": kind, "scenario": scenario, "take": take,
               "campplus_similarity_to_target_centroid": round(cos(e, cent), 4),
               "duration_s": round(d.frames / d.samplerate, 3)}
        if kind == "SOURCE":
            src_emb[(scenario, take)] = e
            row["similarity_to_source"] = ""
            row["delta_similarity_vs_source"] = ""
        else:
            s2s = round(cos(e, src_emb[(scenario, take)]), 4)
            row["similarity_to_source"] = s2s
            row["delta_similarity_vs_source"] = round(
                cos(e, cent) - cos(src_emb[(scenario, take)], cent), 4)
        rows.append(row)

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print("%-22s %s to_target=%.4f %s" % (
            r["file"], r["model"],
            r["campplus_similarity_to_target_centroid"],
            ("to_source=%.4f delta=%.4f" % (r["similarity_to_source"], r["delta_similarity_vs_source"]))
            if r["model"] != "SOURCE" else "(source)"))
    print("WROTE", OUT_CSV)


if __name__ == "__main__":
    main()