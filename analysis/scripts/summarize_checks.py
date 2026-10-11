#!/usr/bin/env python
"""Did the reproduction match the paper?  Summarise every `*_checks.csv` written by the pipeline.

    python scripts/summarize_checks.py --results results            # exit code 1 if any published number was not reproduced

Each analysis stage compares its numbers with the ones printed in the paper (config/paper_values.yaml) and writes `<stage>_checks.csv`
(columns: check, got, expected, ok).  This lists, per file, how many matched, prints the mismatches, and warns about stages that wrote no checks.
"""
import argparse
import glob
import os
import sys

import pandas as pd

EXPECTED_STAGES = {          # stage folder -> what it covers
    "classification": "Table S1, Table S4, Fig. 4A/B, Fig. S3",
    "habituation": "Section 1.7, Fig. 5",
    "pac": "Section 1.1, Fig. 2",
    "fooof": "Section 1.2, Table S2, Fig. S2",
    "breath": "Section 1.3, Table S3",
    "timing": "Section 1.8, Tables S5-S6",
    "subject28": "SI 1.4, Table S7",
    "cluster_perm": "Section 1.5, Fig. S4",
    "fig3": "Fig. 3",
    "word_scoring": "Fig. 4C/D, Section 1.6, SI 3.1",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--max-listed", type=int, default=15, help="mismatches to print per file")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.results, "**", "*_checks.csv"), recursive=True))
    if not files:
        sys.exit(f"No *_checks.csv found under {a.results}/ -- run the pipeline first (and keep checks.enabled: true).")
    total = ok = 0
    seen = set()
    print(f"{'file':58s} {'matched':>12s}")
    mismatches = []
    for f in files:
        df = pd.read_csv(f)
        stage = os.path.basename(os.path.dirname(f))
        seen.add(stage)
        n, k = len(df), int(df["ok"].sum()) if len(df) else 0
        total += n
        ok += k
        print(f"{os.path.relpath(f, a.results):58s} {k:5d} / {n:<5d}" + ("" if k == n else "   <-- mismatches"))
        for _, r in df[~df["ok"].astype(bool)].iterrows():
            mismatches.append((os.path.relpath(f, a.results), r["check"], r["got"], r["expected"]))
    print(f"{'TOTAL':58s} {ok:5d} / {total:<5d}")
    for f, c, g, e in mismatches[: a.max_listed * 4]:
        print(f"  [MISMATCH] {f}: {c}: got {g}, expected {e}")
    if len(mismatches) > a.max_listed * 4:
        print(f"  ... and {len(mismatches) - a.max_listed * 4} more")
    missing = [f"{s} ({w})" for s, w in EXPECTED_STAGES.items() if s not in seen]
    if missing:
        print("\nNo checks found for: " + "; ".join(missing) + "  (stage not run, or checks disabled)")
    sys.exit(0 if ok == total and total > 0 else 1)


if __name__ == "__main__":
    main()
