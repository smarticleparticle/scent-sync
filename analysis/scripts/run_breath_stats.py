#!/usr/bin/env python
"""Section 1.3 / Table S3: do breathing dynamics differ between odor conditions?

First inhale and first exhale after odor onset are detected in every trial (notebook cell 37), averaged within subject and condition,
and compared across the four conditions with a one-way repeated-measures ANOVA (inhale, exhale and total breath duration).

    python scripts/run_breath_stats.py --epochs data/preprocessed/epochs_eeg_respiration.npy --info-dir data/preprocessed/info \
        --params results/breath/params.yaml --out-dir results/breath
"""
import argparse
import glob
import os
import re
import warnings

import numpy as np
import pandas as pd
import yaml

from olfaction_eeg import breath
from olfaction_eeg.checks import Checks

warnings.filterwarnings("ignore")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", required=True, help="the ORIGINAL epochs file (EEG + respiration epochs)")
    ap.add_argument("--info-dir", required=True)
    ap.add_argument("--params", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--strict", action="store_true")
    a = ap.parse_args()

    p = yaml.safe_load(open(a.params))
    paper = (p.get("paper") or {}).get("table_s3", {})
    tol = (p.get("paper") or {}).get("tolerance", {})
    subjects = p["normosmia"]
    os.makedirs(a.out_dir, exist_ok=True)

    # subjects that have an info file (the notebook used this as a membership filter)
    ids = {m.group(1): [] for f in glob.glob(os.path.join(a.info_dir, "subject*_info.fif"))
           for m in [re.search(r"subject(\d+)_info\.fif$", f)] if m}
    epochs = np.load(a.epochs, allow_pickle=True).item()

    breath_data = breath.extract_first_breath_data(epochs, ids, conditions=breath.CONDITIONS, store_eeg=False)
    df_breath = breath.breath_table(breath_data, subjects)
    df_subj_cond = breath.subject_condition_means(df_breath)
    df_breath.to_csv(os.path.join(a.out_dir, "breath_trials.csv"), index=False)

    desc = breath.descriptives(df_subj_cond)
    desc.columns = [f"{m}_{s}" for m, s in desc.columns]
    desc.reindex(breath.CONDITIONS).to_csv(os.path.join(a.out_dir, "table_S3_descriptives.csv"))
    complete = breath.complete_cases(df_subj_cond)
    rows = []
    for metric in breath.METRICS:
        rows.append({"metric": metric, **breath.rm_anova_oneway(complete, metric)})
    anova = pd.DataFrame(rows)
    anova.to_csv(os.path.join(a.out_dir, "table_S3_anova.csv"), index=False)

    print(f"\nsubjects included: {sorted(df_breath['subject'].unique(), key=int)}  (n = {df_breath['subject'].nunique()}, "
          f"complete in all 4 conditions: {complete['subject'].nunique()})")
    print(desc.reindex(breath.CONDITIONS).round(3).to_string())
    print(anova[["metric", "n", "df1", "df2", "F", "p_unc", "ng2", "p_gg"]].round(4).to_string(index=False))

    # ---------------- checks against the paper
    checks = Checks(enabled=bool(p.get("checks_enabled", True) and paper))
    if checks.enabled:
        checks.number("Table S3: subjects in the repeated-measures ANOVA", int(anova["n"].iloc[0]), paper.get("n_subjects"), 0)
        short = {"inhale": "inhale_duration", "exhale": "exhale_duration", "breath": "breath_duration"}
        for cond, want in (paper.get("means_sd") or {}).items():
            for k, col in short.items():
                m, sd = want[k]
                checks.number(f"Table S3 {cond} {k}: mean", desc.loc[cond, f"{col}_mean"], m, 0.006)
                checks.number(f"Table S3 {cond} {k}: SD", desc.loc[cond, f"{col}_std"], sd, 0.006)
        for k, want in (paper.get("anova") or {}).items():
            r = anova[anova["metric"] == short[k]].iloc[0]
            checks.number(f"ANOVA {k}: F", r["F"], want["F"], 0.006)
            checks.number(f"ANOVA {k}: p", r["p_unc"], want["p"], tol.get("p", 0.002))
            checks.number(f"ANOVA {k}: generalized eta squared", r["ng2"], want["ng2"], 0.001)
            checks.number(f"ANOVA {k}: df2", r["df2"], want["df"][1], 0)
    checks.report(os.path.join(a.out_dir, "breath_checks.csv"), strict=a.strict)


if __name__ == "__main__":
    main()
