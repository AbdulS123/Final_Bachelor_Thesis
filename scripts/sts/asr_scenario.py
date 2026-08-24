"""asr_scenario.py - fixed-protocol ASR (faster-whisper large-v3-turbo, de,
beam_size=5, temperature=0.0, vad_filter=False, condition_on_previous_text=True,
word_timestamps=True) over all 18 scenario files (6 sources + 6 RVC + 6 Seed-VC),
then WER/CER against the fixed intended texts (the scenario scripts).

Raw WER/CER stays primary. delta_wer/cer_vs_source are context-only columns.
Normalization is identical to wer_cer_final.py in the LOCKED common-metrics stage
(NFKC, ae/oe/ue/ss, lower, strip non a-z0-9).
"""
import csv
import json
import os
import re
import unicodedata

from faster_whisper import WhisperModel

WS = os.environ.get("SCENARIO_WS")
if not WS:
    raise RuntimeError("Set the SCENARIO_WS environment variable before running this script.")
SOURCE_DIR = os.path.join(WS, "01_SOURCE")
RVC_DIR = os.path.join(WS, "02_RVC")
SEEDVC_DIR = os.path.join(WS, "03_SEEDVC")
METRICS_DIR = os.path.join(WS, "04_METRICS")
TRANSCRIPTS_JSON = os.path.join(METRICS_DIR, "transcripts_scenario.json")
CONTENT_CSV = os.path.join(METRICS_DIR, "sts_scenario_content.csv")

INTENDED = {
    "S1": ("Ich bin gerade zwischen zwei Terminen und die Zahlung muss heute noch "
           "bearbeitet werden. Die schriftliche Bestätigung reiche ich direkt danach nach."),
    "S2": ("Ich stehe gerade in der Apotheke und die Bestätigung in der App funktioniert "
           "nicht. Bitte helfen Sie mir mit der gestoppten Zahlung."),
}

PUNCT = re.compile(r"[^a-z0-9 ]")

FILES = []
for take in (1, 2, 3):
    FILES.append(("S1", take, "S1_executive_take%02d.wav" % take, "SOURCE"))
    FILES.append(("S2", take, "S2_customer_take%02d.wav" % take, "SOURCE"))
for take in (1, 2, 3):
    FILES.append(("S1", take, "S1_RVC_take%02d.wav" % take, "RVC_Audit8_fixed"))
    FILES.append(("S2", take, "S2_RVC_take%02d.wav" % take, "RVC_Audit8_fixed"))
for take in (1, 2, 3):
    FILES.append(("S1", take, "S1_SEEDVC_take%02d.wav" % take, "SeedVC_fixed"))
    FILES.append(("S2", take, "S2_SEEDVC_take%02d.wav" % take, "SeedVC_fixed"))

MODEL_NAME = "large-v3-turbo"


def norm(text):
    t = unicodedata.normalize("NFKC", text)
    t = t.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue").replace("ß", "ss")
    t = t.lower()
    t = PUNCT.sub("", t)
    return [w for w in t.split()]


def lev(a, b):
    prev = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] != b[j - 1]))
        prev = cur
    return prev[-1]


def wer_details(ref, hyp):
    n = len(ref)
    if n == 0:
        return 1.0, 0, 0, 0, 0
    d = lev(ref, hyp)
    m = len(ref) + 1
    mat = [[0] * (len(hyp) + 1) for _ in range(m)]
    for i in range(m):
        mat[i][0] = i
    for j in range(len(hyp) + 1):
        mat[0][j] = j
    for i in range(1, m):
        for j in range(1, len(hyp) + 1):
            mat[i][j] = min(mat[i - 1][j] + 1, mat[i][j - 1] + 1,
                            mat[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]))
    i, j = len(ref), len(hyp)
    subs = dels = ins = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and mat[i][j] == mat[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            subs += ref[i - 1] != hyp[j - 1]
            i, j = i - 1, j - 1
        elif i > 0 and mat[i][j] == mat[i - 1][j] + 1:
            dels += 1
            i -= 1
        else:
            ins += 1
            j -= 1
    return d, subs, dels, ins, n


def cer(ref_tok, hyp_tok):
    r = " ".join(ref_tok)
    h = " ".join(hyp_tok)
    d = lev(list(r), list(h))
    return d / max(1, len(r)), d, len(r)


def main():
    model = WhisperModel(MODEL_NAME, device="cuda", compute_type="float16")
    results = {}
    for scenario, take, name, kind in FILES:
        if kind == "SOURCE":
            path = os.path.join(SOURCE_DIR, name)
        elif kind == "RVC_Audit8_fixed":
            path = os.path.join(RVC_DIR, name)
        else:
            path = os.path.join(SEEDVC_DIR, name)
        segments, info = model.transcribe(
            path, language="de", beam_size=5, temperature=0.0,
            vad_filter=False, condition_on_previous_text=True,
            word_timestamps=True,
        )
        text = "".join(s.text for s in segments).strip()
        results[name] = {"text": text, "language": info.language}
        print(f"{name}: {text}")

    with open(TRANSCRIPTS_JSON, "w", encoding="utf-8") as f:
        json.dump({"model": MODEL_NAME, "protocol": {
            "language": "de", "beam_size": 5, "temperature": 0.0,
            "vad_filter": False, "condition_on_previous_text": True,
            "word_timestamps": True}, "files": results}, f, ensure_ascii=False, indent=2)

    refs = {k: norm(v) for k, v in INTENDED.items()}
    rows = []
    baseline = {}
    for scenario, take, name, kind in FILES:
        hyp = norm(results[name]["text"])
        d, subs, dels, ins, n = wer_details(refs[scenario], hyp)
        c, cd, cr = cer(refs[scenario], hyp)
        row = {"file": name, "model": kind, "scenario": scenario, "take": take,
               "wer": round(d / n, 4), "cer": round(c, 4),
               "subs": subs, "dels": dels, "ins": ins, "ref_words": n,
               "hypothesis": results[name]["text"]}
        if kind == "SOURCE":
            baseline[(scenario, take)] = (d / n, c)
            row["delta_wer_vs_source"] = 0.0
            row["delta_cer_vs_source"] = 0.0
        else:
            bw, bc = baseline[(scenario, take)]
            row["delta_wer_vs_source"] = round(d / n - bw, 4)
            row["delta_cer_vs_source"] = round(c - bc, 4)
        rows.append(row)

    with open(CONTENT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("WROTE", CONTENT_CSV, len(rows), "rows")


if __name__ == "__main__":
    main()