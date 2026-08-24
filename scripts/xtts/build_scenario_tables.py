"""Build 03_FAILURES and 04_RESULTS tables from generation + metrics data.

Failure classification uses only objectively observable evidence:
logs (retry events), WER/CER error counts, output duration vs. siblings,
and the ASR transcript (missing / added content).
"""

import csv
import json
import os
import sys
from pathlib import Path

# Publication path. Set XTTS_OUTPUT_DIR to the private evaluation workspace.
OUTPUT_ROOT = Path(os.environ.get("XTTS_OUTPUT_DIR", str(Path(__file__).resolve().parent / "run_output"))).resolve()

STAGE = OUTPUT_ROOT
LOGS_DIR = STAGE / "07_LOGS"
METRICS_DIR = STAGE / "02_METRICS"
FAILURES_DIR = STAGE / "03_FAILURES"
RESULTS_DIR = STAGE / "04_RESULTS"


def load_csv(path):
    with open(path, encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def main():
    FAILURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    timing = {r["output_filename"]: r for r in
              json.loads((LOGS_DIR / "timing_records.json").read_text(encoding="utf-8"))}
    content = {r["output_filename"]: r for r in load_csv(METRICS_DIR / "xtts_scenario_content.csv")}
    sim = {r["output_filename"]: r for r in load_csv(METRICS_DIR / "xtts_scenario_similarity.csv")}

    combined = []
    failures = []
    for name in sorted(timing.keys()):
        t, c, s = timing[name], content[name], sim[name]
        deletes = int(c["deletions"]); inserts = int(c["insertions"])
        subs = int(c["substitutions"]); words = len(t["normalized_text"].split())
        issues = []

        transcript_ends_at_wer = c["wer"] != "0.0" and deletes > 0
        if deletes > 0 and words > 0 and (deletes / words) > 0.1:
            issues.append("truncation")
            issues.append("missing_linguistic_content")
        if inserts > 0:
            issues.append("added_linguistic_content")
        if t["anomaly_status"] != "CLEAN":
            issues.append("duration_guard_retry" if "ANOMALY" in t["anomaly_status"] else t["anomaly_status"])
        if float(t["rtf"]) > 1.0:
            issues.append("abnormal_generation_time")
        status = "none" if not issues else " | ".join(sorted(set(issues)))

        combined.append({
            "scenario": t["scenario"], "run": t["run"],
            "input_text": t["input_text"], "normalized_text": t["normalized_text"],
            "generation_time_s": t["generation_time_s"],
            "output_duration_s": t["output_duration_s"],
            "RTF": t["rtf"], "WER": c["wer"], "CER": c["cer"],
            "substitutions": c["substitutions"], "deletions": c["deletions"],
            "insertions": c["insertions"],
            "campplus_similarity": s["campplus_similarity"],
            "failure_status": status,
            "output_filename": name,
        })
        failures.append({
            "output_filename": name, "scenario": t["scenario"], "run": t["run"],
            "failure_status": status,
            "duration_s": t["output_duration_s"],
            "rtf": t["rtf"], "wer": c["wer"], "cer": c["cer"],
            "substitutions": c["substitutions"], "deletions": c["deletions"],
            "insertions": c["insertions"],
            "asr_transcript": c["asr_transcript"],
            "retry_events": len(t["retry_events"]),
            "evidence": "ASR transcript missing second sentence (deletions>10% of words) | RTF>1.0 | duration ~half of sibling runs" if "truncation" in status else "none",
        })

    with open(RESULTS_DIR / "xtts_scenario_evaluation.csv", "w", newline="",
              encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(combined[0].keys()))
        w.writeheader()
        w.writerows(combined)
    with open(FAILURES_DIR / "xtts_scenario_failures.csv", "w", newline="",
              encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(failures[0].keys()))
        w.writeheader()
        w.writerows(failures)

    for r in combined:
        print(f"{r['scenario']} run{r['run']}: WER {r['WER']} CER {r['CER']} "
              f"RTF {r['RTF']} CAMP {r['campplus_similarity']} status={r['failure_status']}")
    print("wrote", RESULTS_DIR / "xtts_scenario_evaluation.csv")
    print("wrote", FAILURES_DIR / "xtts_scenario_failures.csv")
    return 0


def load_csv(path):
    with open(path, encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


if __name__ == "__main__":
    sys.exit(main())