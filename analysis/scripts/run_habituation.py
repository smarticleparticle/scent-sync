#!/usr/bin/env python
"""Stage 3: olfactory habituation (paper Section 1.7, Fig. 5).

Replaces analysis-notebook cells 76, 77, 81 and the statistics behind cell 84.
The phase-locked (RPL-ERSP, 4-bin) classifier is run separately on EARLY (trials 1-3) and LATE (trials 6-8)
trial blocks.  For every odor, and pooled over odors:
  * AUC per block and the paired permutation test on the early-minus-late AUC (subjects with both blocks),
  * Benjamini-Hochberg FDR across the 4 tests (3 odors + pooled)  <- notebook cell 77 had the raw p-values typed in
  * bootstrap 95% CIs (resampling subjects) for the Fig. 5A error bars,
  * feature-selection frequencies per block (Fig. 5B).

    python scripts/run_habituation.py --phase P --params results/habituation/params.yaml --out-dir results/habituation
"""
import argparse
import os
import pickle
import warnings

import numpy as np
import pandas as pd
import yaml

from olfaction_eeg import classify as clf
from olfaction_eeg.checks import Checks
from olfaction_eeg.features import extract_binned_features

warnings.filterwarnings("ignore", category=FutureWarning)


def block_result(ersp, odor, subjects, x_bins, trials, C):
    df = extract_binned_features(ersp, "water", odor, subjects, freqs=ersp["freqs"], x_bins=x_bins, x=ersp["x"],
                                 trial_start=trials[0], trial_end=trials[1])
    if df is None:
        raise RuntimeError(f"No usable data for water vs {odor}, trials {trials}")
    return clf.nested_loso_selk_logreg(df, C=C, verbose=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True)
    ap.add_argument("--params", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    p = yaml.safe_load(open(args.params))
    h, odors, subjects = p["habituation"], p["odors"], p["normosmia"]
    x_bins = p["phase_strategies"][p["phase_main_strategy"]]
    paper = p.get("paper") or {}
    tol = paper.get("tolerance", {})
    checks = Checks(enabled=bool(p.get("checks_enabled", True) and paper.get("habituation")))
    os.makedirs(args.out_dir, exist_ok=True)

    phase_ersp = np.load(args.phase, allow_pickle=True).item()

    per_odor = {}
    for odor in odors:
        print(f"\n=== {odor}: habituation ===", flush=True)
        res_early = block_result(phase_ersp, odor, subjects, x_bins, h["early_trials"], p["C"])
        res_late = block_result(phase_ersp, odor, subjects, x_bins, h["late_trials"], p["C"])
        perm = clf.paired_permutation_auc_delta(res_early, res_late, n_perm=h["n_perm"], seed=h["seed"])
        ci_early = clf.bootstrap_auc_ci(res_early, n_boot=h["n_boot"], seed=h["seed"])[1:3]
        ci_late = clf.bootstrap_auc_ci(res_late, n_boot=h["n_boot"], seed=h["seed"])[1:3]
        print(f"early AUC {res_early['auc']:.4f} | late AUC {res_late['auc']:.4f} | "
              f"(common subjects n={perm['n_subjects']}: {perm['observed_auc_a']:.4f} -> {perm['observed_auc_b']:.4f}, "
              f"delta {perm['observed_delta']:+.4f}, p={perm['p_value']:.4f})", flush=True)
        per_odor[odor] = {"res_early": res_early, "res_late": res_late, "perm": perm,
                          "ci_early": ci_early, "ci_late": ci_late}

    pooled = clf.pooled_permutation_auc_delta(
        {o: {"early": per_odor[o]["res_early"], "late": per_odor[o]["res_late"]} for o in odors},
        "early", "late", n_perm=h["n_perm"], seed=h["seed"])
    p_raw = [per_odor[o]["perm"]["p_value"] for o in odors] + [pooled["p_value"]]
    p_fdr = clf.fdr_bh(p_raw)
    print(f"\npooled: n={pooled['n_units']} delta={pooled['observed_delta']:+.4f} p={pooled['p_value']:.4f}")

    # ------------------------------------------------------------------ summary table
    rows = []
    for i, o in enumerate(odors):
        r = per_odor[o]
        rows.append({"odor": o, "n_subjects_with_both_blocks": r["perm"]["n_subjects"],
                     "early_auc_all_subjects": r["res_early"]["auc"], "late_auc_all_subjects": r["res_late"]["auc"],
                     "early_ci_lo": r["ci_early"][0], "early_ci_hi": r["ci_early"][1],
                     "late_ci_lo": r["ci_late"][0], "late_ci_hi": r["ci_late"][1],
                     "early_auc_common_subjects": r["perm"]["observed_auc_a"],
                     "late_auc_common_subjects": r["perm"]["observed_auc_b"],
                     "delta_auc": r["perm"]["observed_delta"], "p_raw": p_raw[i], "p_fdr": p_fdr[i]})
    rows.append({"odor": "pooled", "n_subjects_with_both_blocks": pooled["n_units"],
                 "early_auc_common_subjects": pooled["auc_a_pooled"], "late_auc_common_subjects": pooled["auc_b_pooled"],
                 "delta_auc": pooled["observed_delta"], "p_raw": p_raw[-1], "p_fdr": p_fdr[-1]})
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(args.out_dir, "habituation_summary.csv"), index=False)
    print("\n" + summary.round(4).to_string(index=False))

    # ------------------------------------------------------------------ feature selection (Fig. 5B)
    early_by_odor = {o: per_odor[o]["res_early"] for o in odors}
    late_by_odor = {o: per_odor[o]["res_late"] for o in odors}
    fs = {}
    for zero in (False, True):
        tag = "incl_zeros" if zero else "present_only"
        e = clf.mean_selection_frequency(early_by_odor, zero)
        l = clf.mean_selection_frequency(late_by_odor, zero)
        for feat in sorted(set(e) | set(l)):
            fs.setdefault(feat, {"feature": feat})
            fs[feat][f"early_{tag}"] = e.get(feat, 0.0)
            fs[feat][f"late_{tag}"] = l.get(feat, 0.0)
    fs_df = pd.DataFrame(list(fs.values()))
    for o in odors:
        for blk, by in (("early", early_by_odor), ("late", late_by_odor)):
            d = by[o]["selection_frequency"].set_index("feature")["selected_frac"]
            fs_df[f"{o}_{blk}"] = fs_df["feature"].map(d).fillna(0.0)
    fs_df = fs_df.sort_values("early_present_only", ascending=False)
    fs_df.to_csv(os.path.join(args.out_dir, "habituation_feature_selection.csv"), index=False)

    with open(os.path.join(args.out_dir, "habituation_results.pkl"), "wb") as f:
        pickle.dump({"per_odor": per_odor, "pooled": pooled, "p_raw": p_raw, "p_fdr": p_fdr, "params": p}, f)

    # ------------------------------------------------------------------ checks against the paper
    hv = paper.get("habituation", {})
    if checks.enabled:
        for i, o in enumerate(odors):
            want = hv.get("per_odor", {}).get(o, {})
            r = per_odor[o]
            checks.number(f"habituation {o}: n subjects", r["perm"]["n_subjects"], want.get("n"), 0)
            checks.number(f"habituation {o}: delta AUC", r["perm"]["observed_delta"], want.get("delta_auc"), tol.get("auc", 0.001) * 1.5)
            checks.number(f"habituation {o}: p (FDR)", p_fdr[i], want.get("p_fdr"), tol.get("p", 0.002))
            checks.number(f"habituation {o}: early AUC (common subjects, Fig. 5 caption)", r["perm"]["observed_auc_a"], want.get("early_auc_common"), tol.get("auc", 0.001) * 1.5)
            checks.number(f"habituation {o}: late AUC (common subjects, Fig. 5 caption)", r["perm"]["observed_auc_b"], want.get("late_auc_common"), tol.get("auc", 0.001) * 1.5)
        pw = hv.get("pooled", {})
        checks.number("habituation pooled: n", pooled["n_units"], pw.get("n"), 0)
        checks.number("habituation pooled: delta AUC", pooled["observed_delta"], pw.get("delta_auc"), tol.get("auc", 0.001) * 1.5)
        checks.less_than("habituation pooled: p (FDR)", p_fdr[-1], pw.get("p_fdr_below"))
        zero = bool(h["fig5b_count_unselected_as_zero"])
        e = clf.mean_selection_frequency(early_by_odor, zero)
        l = clf.mean_selection_frequency(late_by_odor, zero)
        for key, want in (hv.get("selection_drop_pct") or {}).items():
            band, _, binname = key.partition("_")
            feat = f"{band}_{binname}"
            drop = 100 * (e.get(feat, 0) - l.get(feat, 0)) / e[feat] if e.get(feat, 0) > 0 else float("nan")
            checks.number(f"habituation Fig 5B: drop in selection of {feat} (%)", drop, want, 1.0)
    checks.report(os.path.join(args.out_dir, "habituation_checks.csv"), strict=args.strict)


if __name__ == "__main__":
    main()
