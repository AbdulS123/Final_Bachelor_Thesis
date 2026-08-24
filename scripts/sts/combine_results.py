"""combine_results.py - build 06_RESULTS/sts_scenario_evaluation.csv (12 rows: 6 RVC +
6 Seed-VC) from the content, similarity, timing, and failure tables, plus the
descriptive per-combination summary (n=3, mean/min/max; no significance tests,
no ranking).
"""
import csv
import json
import os

WS = os.environ.get("SCENARIO_WS")
if not WS:
    raise RuntimeError("Set the SCENARIO_WS environment variable before running this script.")
CONTENT = os.path.join(WS, "04_METRICS", "sts_scenario_content.csv")
SIMILAR = os.path.join(WS, "04_METRICS", "sts_scenario_similarity.csv")
FAILURES = os.path.join(WS, "05_FAILURES", "sts_scenario_failures.csv")
RVC_TIMING = os.path.join(WS, "02_RVC", "rvc_timing.csv")
SV_TIMING = os.path.join(WS, "03_SEEDVC", "seedvc_timing.csv")
OUT_CSV = os.path.join(WS, "06_RESULTS", "sts_scenario_evaluation.csv")
OUT_SUMMARY = os.path.join(WS, "06_RESULTS", "descriptive_summary.md")


def load(path, key):
    out = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[row[key]] = row
    return out


def main():
    content = load(CONTENT, "file")
    similar = load(SIMILAR, "file")
    failures = load(FAILURES, "file")
    timing = {}
    timing.update(load(RVC_TIMING, "file"))
    timing.update(load(SV_TIMING, "file"))

    rows = []
    for fn in sorted(os.listdir(os.path.join(WS, "02_RVC"))) + sorted(os.listdir(os.path.join(WS, "03_SEEDVC"))):
        if not fn.endswith(".wav"):
            continue
        c = content[fn]
        s = similar[fn]
        t = timing[fn]
        f = failures[fn]
        rows.append({
            "file": fn,
            "system": c["model"],
            "scenario": c["scenario"],
            "take": c["take"],
            "wer_vs_intended": c["wer"],
            "cer_vs_intended": c["cer"],
            "delta_wer_vs_source": c["delta_wer_vs_source"],
            "delta_cer_vs_source": c["delta_cer_vs_source"],
            "subs_dels_ins": "%s/%s/%s" % (c["subs"], c["dels"], c["ins"]),
            "campplus_similarity_to_target": s["campplus_similarity_to_target_centroid"],
            "similarity_to_source": s["similarity_to_source"],
            "delta_similarity_vs_source": s["delta_similarity_vs_source"],
            "processing_time_s": t["processing_time_s"],
            "output_duration_s": t["output_duration_s"],
            "source_duration_s": t["source_duration_s"],
            "rtf": t["rtf"],
            "model_loaded": t["model_loaded"],
            "preprocessing_included": t["preprocessing_included"],
            "primary_failure": f["primary_failure"],
            "secondary_failures": f["secondary_failures"],
        })

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    def stats(items, key):
        vals = [float(r[key]) for r in items]
        return min(vals), max(vals), sum(vals) / len(vals)

    combos = [
        ("RVC / S1", [r for r in rows if r["system"] == "RVC_Audit8_fixed" and r["scenario"] == "S1"]),
        ("RVC / S2", [r for r in rows if r["system"] == "RVC_Audit8_fixed" and r["scenario"] == "S2"]),
        ("Seed-VC / S1", [r for r in rows if r["system"] == "SeedVC_fixed" and r["scenario"] == "S1"]),
        ("Seed-VC / S2", [r for r in rows if r["system"] == "SeedVC_fixed" and r["scenario"] == "S2"]),
    ]

    lines = [
        "# Descriptive summary (n=3 per combination; descriptive only — no significance tests, no ranking)",
        "",
        "Per-metric mean / min / max over the 3 takes of each combination. RTF =",
        "processing_time / source_duration with the model already loaded; a conversion",
        "throughput indicator, not conversational latency.",
        "",
        "| Combination | metric | mean | min | max |",
        "| --- | --- | --- | --- | --- |",
    ]
    for label, items in combos:
        for metric in ["wer_vs_intended", "cer_vs_intended", "delta_wer_vs_source",
                       "campplus_similarity_to_target", "delta_similarity_vs_source",
                       "rtf"]:
            lo, hi, mean = stats(items, metric)
            lines.append("| %s | %s | %.4f | %.4f | %.4f |" % (label, metric, mean, lo, hi))
        fails = [r["primary_failure"] for r in items if r["primary_failure"] != "none"]
        lines.append("| %s | failures (primary, n) | %s | | |" % (
            label, ", ".join(sorted(set(fails))) if fails else "none"))
    with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("WROTE", OUT_CSV, len(rows), "rows")
    print("WROTE", OUT_SUMMARY)


if __name__ == "__main__":
    main()