"""failures_scenario.py - objective failure classification for the 12 scenario
outputs, based on ASR transcripts + durations ONLY (no listening).

Attribution principle: outputs are compared against the SOURCE turbo transcript
(not against the intended script text), so deviations that the speaker already
made in the source (e.g. spoken "dir") are not attributed to the conversion.
Token matching is substring-aware: a source token counts as present if it equals
or is contained in an output token (handles merged words such as
'zwischenzeittermin' without double-counting them as missing AND inserted).

Heuristics (all documented in the failures CSV and the report):
- conversion crash        : output file missing or 0 bytes (process-level failure)
- truncation              : output_duration < 0.75 x source_duration
- abnormal duration       : |output_duration/source_duration - 1| > 0.25
- repeated word/phrase    : a consecutive duplicate n-gram (n>=3) in the output
                            normalized transcript
- missing content         : >= 3 source tokens absent (not contained) in output
- inserted content        : >= 2 output tokens absent (not contained) in source
- malformed ending        : last source token not contained in the last 3 output
                            tokens
- none                    : no flag

These are OBJECTIVE FAILURE heuristics based on ASR evidence; they are not a
substitute for listening (author listening remains the final authority).
"""
import csv
import json
import os
import re
import unicodedata

WS = os.environ.get("SCENARIO_WS")
if not WS:
    raise RuntimeError("Set the SCENARIO_WS environment variable before running this script.")
TRANSCRIPTS = os.path.join(WS, "04_METRICS", "transcripts_scenario.json")
OUT = os.path.join(WS, "05_FAILURES", "sts_scenario_failures.csv")

RVC_TIMING = os.path.join(WS, "02_RVC", "rvc_timing.csv")
SV_TIMING = os.path.join(WS, "03_SEEDVC", "seedvc_timing.csv")

PUNCT = re.compile(r"[^a-z0-9 ]")

PAIRS = []
for take in (1, 2, 3):
    PAIRS.append(("S1_executive_take%02d.wav" % take, "S1_RVC_take%02d.wav" % take, "RVC_Audit8_fixed", "S1", take))
    PAIRS.append(("S2_customer_take%02d.wav" % take, "S2_RVC_take%02d.wav" % take, "RVC_Audit8_fixed", "S2", take))
    PAIRS.append(("S1_executive_take%02d.wav" % take, "S1_SEEDVC_take%02d.wav" % take, "SeedVC_fixed", "S1", take))
    PAIRS.append(("S2_customer_take%02d.wav" % take, "S2_SEEDVC_take%02d.wav" % take, "SeedVC_fixed", "S2", take))


def norm(text):
    t = unicodedata.normalize("NFKC", text)
    t = t.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue").replace("ß", "ss")
    t = t.lower()
    t = PUNCT.sub("", t)
    return t.split()


def ngram_repeat(tokens, n=3):
    for i in range(len(tokens) - n + 1):
        window = tokens[i:i + n]
        for j in range(i + n, len(tokens) - n + 1):
            if tokens[j:j + n] == window:
                return " ".join(window)
    return None


def contains(tok, others):
    return any(tok == o or tok in o or o in tok for o in others)


def load_timing(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[row["file"]] = row
    return out


def main():
    trans = json.load(open(TRANSCRIPTS, encoding="utf-8"))["files"]
    timing = {}
    timing.update(load_timing(RVC_TIMING))
    timing.update(load_timing(SV_TIMING))

    rows = []
    for src_fn, out_fn, model, scenario, take in PAIRS:
        src_tok = norm(trans[src_fn]["text"])
        out_tok = norm(trans[out_fn]["text"])
        t = timing[out_fn]
        src_dur = float(t["source_duration_s"])
        out_dur = float(t["output_duration_s"])
        dur_ratio = out_dur / src_dur
        out_path = os.path.join(WS, "02_RVC" if "RVC" in out_fn else "03_SEEDVC", out_fn)

        flags = []
        evidence = []
        if not os.path.isfile(out_path) or os.path.getsize(out_path) == 0:
            flags.append("conversion crash")
            evidence.append("output file missing or empty")
        if dur_ratio < 0.75:
            flags.append("truncation")
            evidence.append("duration %.2fs vs source %.2fs (ratio %.2f)" % (out_dur, src_dur, dur_ratio))
        rep = ngram_repeat(out_tok, 3)
        if rep:
            flags.append("repeated word/phrase")
            evidence.append("repeated phrase: '%s'" % rep)
        missing = [w for w in src_tok if not contains(w, out_tok)]
        if len(missing) >= 3:
            flags.append("missing linguistic content")
            evidence.append("%d source tokens absent: %s" % (len(missing), " ".join(missing[:8])))
        inserted = [w for w in out_tok if not contains(w, src_tok)]
        if len(inserted) >= 2:
            flags.append("inserted linguistic content")
            evidence.append("%d output tokens absent from source: %s" % (len(inserted), " ".join(inserted[:8])))
        if src_tok and out_tok and not contains(src_tok[-1], out_tok[-3:]):
            flags.append("malformed ending")
            evidence.append("source ends '%s', output ends '%s'" % (src_tok[-1], " ".join(out_tok[-3:])))
        if abs(dur_ratio - 1.0) > 0.25:
            flags.append("abnormal duration")
            evidence.append("duration ratio %.2f" % dur_ratio)

        if not flags:
            primary = "none"
            secondary = []
        else:
            primary = flags[0]
            secondary = flags[1:]

        rows.append({
            "file": out_fn, "system": model, "scenario": scenario, "take": take,
            "primary_failure": primary, "secondary_failures": "; ".join(secondary) or "",
            "evidence": "; ".join(evidence) or "",
            "source_duration_s": src_dur, "output_duration_s": out_dur,
            "duration_ratio": round(dur_ratio, 3),
            "heuristics_note": "ASR-based objective heuristics; final authority = author listening",
        })

    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print("%-22s %-32s %s" % (r["file"], r["primary_failure"], r["evidence"][:140]))
    print("WROTE", OUT)


if __name__ == "__main__":
    main()