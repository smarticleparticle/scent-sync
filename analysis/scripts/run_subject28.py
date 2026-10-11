#!/usr/bin/env python
"""Subject-28 sensitivity: SI Section 1.4 (breathing rate vs. trial retention) and SI Table S7 (AUC with / without subject 28).

Replaces analysis-notebook cells 8-9 (rejected-trial breakdown of subject 28), 52-53 (Spearman: breathing rate vs. % trials kept)
and 58 (Table S7).  Inputs: the ORIGINAL epochs file (as for Table S3), the info folder, and the paper's phase-aligned ERSP file.

    python scripts/run_subject28.py --epochs data/preprocessed/epochs_eeg_respiration.npy --info-dir data/preprocessed/info \
        --phase results_paper/ersp/phase_aligned_fullbreath_1deg.npy --params results_paper/subject28/params.yaml \
        --out-dir results_paper/subject28
"""
import argparse
import glob
import os
import re
import warnings

import numpy as np
import pandas as pd
import yaml

from olfaction_eeg import breath, subject28 as s28
from olfaction_eeg.checks import Checks

warnings.filterwarnings("ignore")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", required=True, help="the ORIGINAL epochs file (EEG + respiration epochs)")
    ap.add_argument("--info-dir", required=True)
    ap.add_argument("--phase", required=True, help="phase-aligned ERSP .npy (the paper's own file, or the regenerated one)")
    ap.add_argument("--params", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--strict", action="store_true")
    a = ap.parse_args()

    p = yaml.safe_load(open(a.params))
    paper = (p.get("paper") or {}).get("subject28", {})
    tol = (p.get("paper") or {}).get("tolerance", {})
    cfg = p["subject28"]
    subjects, odors, drop = p["normosmia"], p["odors"], str(cfg["subject"])
    os.makedirs(a.out_dir, exist_ok=True)

    epochs = np.load(a.epochs, allow_pickle=True).item()
    phase = np.load(a.phase, allow_pickle=True).item()

    # ------------------------------------------------------------------ SI 1.4a: why were subject 28's trials rejected? (cells 8-9)
    skip, detail = s28.diagnose_subject(epochs, drop)
    cyc = s28.rejected_cycle_breakdown(detail)
    detail.to_csv(os.path.join(a.out_dir, "subject28_trial_diagnosis.csv"), index=False)
    pd.DataFrame([{"reason": k, "n_trials": v} for k, v in skip.items()]).to_csv(
        os.path.join(a.out_dir, "subject28_skip_reasons.csv"), index=False)
    print(f"\n=== Subject {drop}: ERSP quality-control breakdown (all conditions) ===")
    for k, v in skip.items():
        print(f"  {k:38s}: {v}")
    print(f"  rejected for too few cycles: {cyc['n_rejected']}; detected cycles {cyc['cycle_counts']}")
    print(f"  would pass at 2 cycles: {cyc['would_pass_at_2_cycles']}; still fail: {cyc['still_fail_at_2_cycles']}")

    # ------------------------------------------------------------------ SI 1.4b: breathing rate vs trial retention (cells 52-53)
    ids = {m.group(1): [] for f in glob.glob(os.path.join(a.info_dir, "subject*_info.fif"))
           for m in [re.search(r"subject(\d+)_info\.fif$", f)] if m}
    breath_data = breath.extract_first_breath_data(epochs, ids, conditions=breath.CONDITIONS, store_eeg=False)
    df_breath = breath.breath_table(breath_data, subjects)
    ret = s28.trial_retention(epochs, breath_data, subjects)
    stats = s28.subject_breath_stats(df_breath)
    merged = stats.merge(ret, on="subject").sort_values("breaths_per_16s").reset_index(drop=True)
    merged.to_csv(os.path.join(a.out_dir, "retention_by_subject.csv"), index=False)
    cols = ["subject", "mean_breath_sec", "breaths_per_16s", "total_attempted", "total_kept", "pct_kept"]
    print("\n=== Breathing rate vs. % trials kept (first-breath criterion) ===")
    print(merged[cols].round(4).to_string(index=False))
    corr = s28.retention_vs_rate(merged, drop)
    print(f"\nSpearman (all {corr['n_all']}): r={corr['r_all']:.3f}, p={corr['p_all']:.4f}")
    print(f"Without subject {drop} ({corr['n_without']}): r={corr['r_without']:.3f}, p={corr['p_without']:.4f}")
    pd.DataFrame([corr]).to_csv(os.path.join(a.out_dir, "retention_spearman.csv"), index=False)

    # ------------------------------------------------------------------ Table S7 (cell 58)
    t7 = s28.auc_without_subject(phase, subjects, odors, drop_subject=drop, C=cfg["C"],
                                 trial_start=cfg["trial_start"], trial_end=cfg["trial_end"])
    t7.to_csv(os.path.join(a.out_dir, "table_S7.csv"), index=False)
    print("\n=== Table S7 ===\n" + t7.round(4).to_string(index=False))

    # ------------------------------------------------------------------ checks
    checks = Checks(enabled=bool(p.get("checks_enabled", True) and paper))
    if checks.enabled:
        for odor, want in (paper.get("table_s7") or {}).items():
            r = t7[t7.odor == odor].iloc[0]
            checks.number(f"Table S7 {odor}: AUC with subject {drop}", r["auc_with_subj28"], want[0], tol.get("auc", 0.001))
            checks.number(f"Table S7 {odor}: AUC without subject {drop}", r["auc_without_subj28"], want[1], tol.get("auc", 0.001))
            checks.number(f"Table S7 {odor}: delta AUC", r["delta"], want[2], 0.0015)
            checks.less_than(f"SI 1.4 {odor}: |delta AUC|", abs(r["delta"]), paper.get("max_abs_delta_auc", 0.005))
        sp = paper.get("spearman") or {}
        checks.number("SI 1.4 Spearman r (all subjects)", corr["r_all"], (sp.get("all") or {}).get("r"), 0.005)
        checks.number("SI 1.4 Spearman p (all subjects)", corr["p_all"], (sp.get("all") or {}).get("p"), tol.get("p", 0.002))
        checks.number("SI 1.4 Spearman r (without 28)", corr["r_without"], (sp.get("without") or {}).get("r"), 0.005)
        checks.number("SI 1.4 Spearman p (without 28)", corr["p_without"], (sp.get("without") or {}).get("p"), tol.get("p", 0.003))
        row = merged[merged["subject"] == drop].iloc[0]
        checks.number("SI 1.4 subject 28 % trials kept (first-breath criterion)", row["pct_kept"], paper.get("pct_kept_28"), 0.05)
        checks.number("SI 1.4 subject 28 trials attempted", row["total_attempted"], paper.get("attempted_28"), 0)
        checks.number("SI 1.4 subject 28 trials rejected by the ERSP quality control", cyc["n_rejected"], paper.get("rejected_28"), 0)
        checks.number("SI 1.4 subject 28 valid trials under the ERSP quality control", skip["valid"], paper.get("valid_28_ersp_qc"), 0)
        checks.number("SI 1.4 rejected trials with 0-1 cycles", cyc["n_with_0_or_1_cycles"], paper.get("rejected_with_0_or_1_cycles"), 0)
        checks.number("SI 1.4 rejected trials that would pass at 2 cycles", cyc["would_pass_at_2_cycles"], paper.get("would_pass_at_2_cycles"), 0)
        # the full table printed by notebook cell 52 (which subject-level breathing-rate definition reproduces it?)
        tab = paper.get("notebook_cell52_table") or {}
        for variant, col_t, col_r in (("trial-weighted", "mean_breath_sec", "breaths_per_16s"),
                                      ("mean of condition means", "mean_breath_sec_condmean", "breaths_per_16s_condmean")):
            ok = 0
            for s, want in tab.items():
                m = merged[merged["subject"] == str(s)]
                if len(m) and abs(float(m[col_t].iloc[0]) - want[0]) <= 0.0006 and abs(float(m[col_r].iloc[0]) - want[1]) <= 0.0006:
                    ok += 1
            if tab:
                print(f"subject-level breathing rate, definition '{variant}': {ok}/{len(tab)} subjects match notebook cell 52")
        for s, want in tab.items():
            m = merged[merged["subject"] == str(s)]
            if not len(m):
                checks.text(f"cell-52 table: subject {s} present", "missing", "present")
                continue
            m = m.iloc[0]
            checks.number(f"cell-52 table {s}: mean breath (s)", m["mean_breath_sec"], want[0], 0.0006)
            checks.number(f"cell-52 table {s}: breaths per 16 s", m["breaths_per_16s"], want[1], 0.0006)
            checks.number(f"cell-52 table {s}: trials attempted", m["total_attempted"], want[2], 0)
            checks.number(f"cell-52 table {s}: trials kept", m["total_kept"], want[3], 0)
            checks.number(f"cell-52 table {s}: % kept", m["pct_kept"], want[4], 0.001)
    checks.report(os.path.join(a.out_dir, "subject28_checks.csv"), strict=a.strict)


if __name__ == "__main__":
    main()
