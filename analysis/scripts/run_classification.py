#!/usr/bin/env python
"""Stage 2: odor-vs-water classification, phase-locked (RPL-ERSP) vs time-locked ERSP.

Produces the numbers behind paper Fig. 4A/B, Fig. S3, Table S1 and Table S4
(replaces analysis-notebook cells 3, 4, 5, 12, 22, 26, 57, 59).  For every odor:

  1. nested leave-one-subject-out classification for every candidate binning of
       RPL-ERSP (1/2/4 phase bins), "Traditional" ERSP (2-s window, 1/2/4 time bins) and
       "Longer" ERSP (3.78-s window, 1/2/4 time bins)                              -> Table S1
  2. the best binning of each time-locked window by AUC (this replaces the hand-copied
     `best_time_strategies` dict of the notebook); RPL uses the 4-bin version throughout
  3. paired permutation tests on per-subject decision-value gaps, RPL vs Traditional and RPL vs Longer
     (best of 3 binnings), Benjamini-Hochberg FDR over the six comparisons          -> Table S4
  4. selection-frequency tables                                                      -> Fig. 4A, Fig. S3
  5. the per-subject RPL predictions used by the word-embedding analysis             -> neural_predictions_early.csv
  6. comparison of everything above with the numbers printed in the paper            -> classification_checks.csv

    python scripts/run_classification.py --phase P --time-short S --time-long L \
        --params results/classification/params.yaml --out-dir results/classification
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
REPS = [("traditional", "2s"), ("longer", "3.78s"), ("rpl", "RPL")]


def features(ersp, odor, subjects, x_bins, p):
    df = extract_binned_features(
        ersp, "water", odor, subjects,
        freqs=ersp["freqs"], x_bins=x_bins, x=ersp["x"],
        trial_start=p["trial_start"], trial_end=p["trial_end"],
    )
    if df is None:
        raise RuntimeError(f"No usable data for water vs {odor} with bins {list(x_bins)}")
    return df


def search_strategies(ersp, strategies, odor, subjects, p, label):
    """Run every candidate binning; best = highest AUC (first wins ties), as in notebook cells 3/5/59."""
    results, rows, best_name, best_auc = {}, [], None, -1.0
    for name, x_bins in strategies.items():
        res = clf.nested_loso_selk_logreg(features(ersp, odor, subjects, x_bins, p), C=p["C"], verbose=False)
        results[name] = res
        rows.append({"odor": odor, "window": label, "strategy": name, "auc": res["auc"], "acc": res["acc"]})
        if res["auc"] > best_auc:
            best_auc, best_name = res["auc"], name
    for r in rows:
        r["is_best"] = (r["strategy"] == best_name)
    return best_name, results, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, help="phase-aligned ERSP .npy")
    ap.add_argument("--time-short", required=True, help="time-locked ERSP, 2-s window .npy")
    ap.add_argument("--time-long", required=True, help="time-locked ERSP, 3.78-s window .npy")
    ap.add_argument("--params", required=True, help="params.yaml written by the workflow")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--strict", action="store_true", help="exit non-zero if a published value is not reproduced")
    args = ap.parse_args()

    p = yaml.safe_load(open(args.params))
    odors, subjects = p["odors"], p["normosmia"]
    paper = p.get("paper") or {}
    tol = paper.get("tolerance", {})
    checks = Checks(enabled=bool(p.get("checks_enabled", True) and paper))
    os.makedirs(args.out_dir, exist_ok=True)

    phase_ersp = np.load(args.phase, allow_pickle=True).item()
    short_ersp = np.load(args.time_short, allow_pickle=True).item()
    long_ersp = np.load(args.time_long, allow_pickle=True).item()
    main_phase = p["phase_main_strategy"]
    fig_long = p["time_378s_comparison_strategy"]

    per_odor, search_rows = {}, []
    for odor in odors:
        print(f"\n=== {odor} ===", flush=True)
        _, all_phase, rows_phase = search_strategies(phase_ersp, p["phase_strategies"], odor, subjects, p, "RPL")
        best_2s, all_2s, rows_2s = search_strategies(short_ersp, p["time_strategies_2s"], odor, subjects, p, "2s")
        best_378, all_378, rows_378 = search_strategies(long_ersp, p["time_strategies_378s"], odor, subjects, p, "3.78s")
        search_rows += rows_phase + rows_2s + rows_378

        res_phase = all_phase[main_phase]
        res_trad = all_2s[best_2s]
        res_long_best = all_378[best_378]          # Table S4 ("Longer": best of 1/2/4 bins)
        res_long_fig = all_378[fig_long]           # Fig. 4 (4-bin for every odor)
        print(f"RPL {main_phase} AUC {res_phase['auc']:.4f} | Traditional best {best_2s} ({res_trad['auc']:.4f}) "
              f"| Longer best {best_378} ({res_long_best['auc']:.4f})", flush=True)

        per_odor[odor] = {
            "phase": res_phase, "phase_strategy": main_phase,
            "time_2s": res_trad, "time_2s_strategy": best_2s,
            "time_378s": res_long_fig, "time_378s_strategy": fig_long,
            "time_378s_best": res_long_best, "time_378s_best_strategy": best_378,
            "all_phase": all_phase, "all_time_2s": all_2s, "all_time_378s": all_378,
        }

    # ------------------------------------------------------------------ Table S4: decision-value-gap tests
    g = p["gap_test"]
    rows_s4 = []
    for odor in odors:
        r = per_odor[odor]
        gaps_phase = clf.get_decision_gaps(r["phase"])
        for comparison, other in [("phase_vs_traditional", r["time_2s"]), ("phase_vs_longer", r["time_378s_best"])]:
            gaps_other = clf.get_decision_gaps(other)
            common = sorted(set(gaps_phase) & set(gaps_other))
            diff, p_raw, lo, hi, outside = clf.paired_gap_permutation_test(
                [gaps_phase[s] for s in common], [gaps_other[s] for s in common],
                n_permutations=g["n_perm"], seed=g["seed"], ci=g["ci"])
            rows_s4.append({"odor": odor, "comparison": comparison, "n": len(common), "observed_diff": diff,
                            "null_lo": lo, "null_hi": hi, "p_raw": p_raw, "outside_null": outside})
    table_s4 = pd.DataFrame(rows_s4)
    table_s4["p_fdr"] = clf.fdr_bh(table_s4["p_raw"].values)
    print("\nTable S4\n" + table_s4.round(4).to_string(index=False))

    # ------------------------------------------------------------------ Table S1 (wide) + long search table
    search = pd.DataFrame(search_rows)
    wide = search.pivot_table(index="odor", columns=["window", "strategy"], values="auc").reindex(odors)
    wide.columns = [f"{w}: {s}" for w, s in wide.columns]
    wide.reset_index().to_csv(os.path.join(args.out_dir, "table_S1.csv"), index=False)
    search.to_csv(os.path.join(args.out_dir, "strategy_search.csv"), index=False)
    table_s4.to_csv(os.path.join(args.out_dir, "table_S4.csv"), index=False)

    # ------------------------------------------------------------------ summary
    summary = pd.DataFrame([{
        "odor": o,
        "rpl_strategy": per_odor[o]["phase_strategy"], "rpl_auc": per_odor[o]["phase"]["auc"],
        "traditional_strategy": per_odor[o]["time_2s_strategy"], "traditional_auc": per_odor[o]["time_2s"]["auc"],
        "longer_best_strategy": per_odor[o]["time_378s_best_strategy"], "longer_best_auc": per_odor[o]["time_378s_best"]["auc"],
        f"longer_{fig_long}_auc": per_odor[o]["time_378s"]["auc"],
        "n_subjects": len(np.unique(per_odor[o]["phase"]["subject"])),
    } for o in odors])
    summary.to_csv(os.path.join(args.out_dir, "classification_summary.csv"), index=False)

    # per-subject RPL predictions for the word-embedding analysis (was notebook cell 4 -> /tmp/neural_predictions_early.csv)
    pd.concat([pd.DataFrame({"subject": per_odor[o]["phase"]["subject"], "label": per_odor[o]["phase"]["y_true"],
                             "y_proba": per_odor[o]["phase"]["y_proba"], "odor": o}) for o in odors],
              ignore_index=True).to_csv(os.path.join(args.out_dir, "neural_predictions_early.csv"), index=False)

    with open(os.path.join(args.out_dir, "classification_results.pkl"), "wb") as f:
        pickle.dump({"per_odor": per_odor, "table_s4": table_s4, "params": p}, f)

    # ------------------------------------------------------------------ checks against the paper
    if checks.enabled:
        t1 = paper.get("table_s1", {})
        for odor in odors:
            for rep, label in REPS:
                for strat, want in t1.get(odor, {}).get(rep, {}).items():
                    got = search[(search.odor == odor) & (search.window == label) & (search.strategy == strat)]["auc"].values[0]
                    checks.number(f"Table S1 {odor} {rep} {strat}", got, want, tol.get("auc", 0.001))
        for row in paper.get("table_s4", []):
            m = table_s4[(table_s4.odor == row["odor"]) & (table_s4.comparison == row["comparison"])].iloc[0]
            tag = f"Table S4 {row['odor']} {row['comparison']}"
            checks.number(f"{tag}: n", m["n"], row["n"], 0)
            checks.number(f"{tag}: observed diff", m["observed_diff"], row["observed_diff"], tol.get("auc", 0.001))
            checks.number(f"{tag}: p_raw", m["p_raw"], row["p_raw"], tol.get("p", 0.002))
            checks.number(f"{tag}: p_fdr", m["p_fdr"], row["p_fdr"], tol.get("p", 0.002))
        phase_bins = list(p["phase_strategies"][main_phase])
        time_bins = list(p["time_strategies_378s"][fig_long])
        bands = ["Beta", "Alpha", "Theta"]
        mean_phase = np.mean([clf.selection_matrix(per_odor[o]["phase"], phase_bins) for o in odors], axis=0)
        mean_time = np.mean([clf.selection_matrix(per_odor[o]["time_378s"], time_bins) for o in odors], axis=0)
        checks.matrix("Fig 4A RPL-ERSP (%)", mean_phase, paper.get("fig4a_phase"), tol.get("pct_fig4a", 0.15), bands, phase_bins)
        checks.matrix("Fig 4A time-locked (%)", mean_time, paper.get("fig4a_time"), tol.get("pct_fig4a", 0.15), bands, time_bins)
        for odor in odors:
            checks.matrix(f"Fig S3 {odor} (%)", clf.selection_matrix(per_odor[odor]["phase"], phase_bins),
                          paper.get("fig_s3", {}).get(odor), tol.get("pct_figS3", 1.0), bands, phase_bins)
    checks.report(os.path.join(args.out_dir, "classification_checks.csv"), strict=args.strict)


if __name__ == "__main__":
    main()
