#!/usr/bin/env python
"""Section 1.5 / SI Fig. S4: cluster-based permutation tests of the odor-minus-water ERSPs (theta x first half, alpha x second half),
in the 2-s time-locked, 3.78-s time-locked and phase-locked representations, plus two band-width sensitivity tests.

Replaces analysis-notebook cells 117-128.  Takes the three ERSP files (the paper's own, or regenerated ones).

    python scripts/run_cluster_perm.py --phase results_paper/ersp/phase_aligned_fullbreath_1deg.npy \
        --time-short results_paper/ersp/time_aligned_short.npy --time-long results_paper/ersp/time_aligned_long.npy \
        --params results_paper/cluster_perm/params.yaml --out-dir results_paper/cluster_perm --fig-dir figures_paper
"""
import argparse
import os
import pickle
import warnings

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import yaml

from olfaction_eeg import cluster_perm as cp
from olfaction_eeg.checks import Checks
from olfaction_eeg.cluster_figure import make_figS4

warnings.filterwarnings("ignore")
BANDS = (("theta", "theta_result"), ("alpha", "alpha_result"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True)
    ap.add_argument("--time-short", required=True)
    ap.add_argument("--time-long", required=True)
    ap.add_argument("--params", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--fig-dir", required=True)
    ap.add_argument("--n-perm", type=int, default=None, help="override the number of permutations (published p-values use 1000; checks are then skipped)")
    ap.add_argument("--strict", action="store_true")
    a = ap.parse_args()

    p = yaml.safe_load(open(a.params))
    cfg, paper = p["cluster_perm"], (p.get("paper") or {}).get("cluster_perm", {})
    n_perm = a.n_perm or cfg["n_perm"]
    published = n_perm == 1000
    os.makedirs(a.out_dir, exist_ok=True)
    os.makedirs(a.fig_dir, exist_ok=True)

    phase = np.load(a.phase, allow_pickle=True).item()
    short = np.load(a.time_short, allow_pickle=True).item()
    long_ = np.load(a.time_long, allow_pickle=True).item()
    subjects, odors = p["normosmia"], p["odors"]
    kw = dict(trial_start=cfg["trial_start"], trial_end=cfg["trial_end"], n_perm=n_perm)

    results = cp.run_all(phase, short, long_, subjects, odors, theta_range=tuple(cfg["theta"]), alpha_range=tuple(cfg["alpha"]),
                         bin_factor=cfg["bin_factor"], **kw)

    summary, clusters = [], []
    for run, res in results.items():
        for band, key in BANDS:
            sub = res[key]
            b = cp.best_cluster(sub)
            summary.append({"run": run, "band": band, "n_clusters": int(sub["n_clusters"]),
                            "best_p": b["p_value"] if b else 1.0, "best_n_pixels": int(b["n_pixels"]) if b else 0,
                            "best_mass": float(b["mass"]) if b else 0.0})
            for c in sub["clusters"]:
                clusters.append({"run": run, "band": band, "cluster_id": c["cluster_id"], "mass": float(c["mass"]),
                                 "p_value": c["p_value"], "n_pixels": int(c["n_pixels"])})
    summary = pd.DataFrame(summary)
    summary.to_csv(os.path.join(a.out_dir, "cluster_perm_summary.csv"), index=False)
    pd.DataFrame(clusters).to_csv(os.path.join(a.out_dir, "cluster_perm_clusters.csv"), index=False)

    # ---- band-width sensitivity of the phase-locked test (notebook cells 124-125)
    sens_rows = []
    for s in cfg.get("sensitivity", []):
        res = cp.run_region(phase, subjects, odors, s["name"], s["freq_range"], s["phase_range"], s["tail"],
                            bin_factor=cfg["bin_factor"], **kw)
        b = cp.best_cluster(res)
        sens_rows.append({"name": s["name"], "freq_range": str(s["freq_range"]), "phase_range": str(s["phase_range"]), "tail": s["tail"],
                          "n_clusters": int(res["n_clusters"]), "best_p": b["p_value"] if b else 1.0,
                          "best_n_pixels": int(b["n_pixels"]) if b else 0})
    sens = pd.DataFrame(sens_rows)
    sens.to_csv(os.path.join(a.out_dir, "cluster_perm_sensitivity.csv"), index=False)

    with open(os.path.join(a.out_dir, "cluster_perm_results.pkl"), "wb") as f:
        pickle.dump({"results": results, "summary": summary, "params": p, "n_perm": n_perm}, f)
    fig, vmax = make_figS4(results, out_prefix=os.path.join(a.fig_dir, "figS4_cluster_perm"))

    print("\n=== Section 1.5 / Fig. S4: best cluster per run and band ===")
    print(summary.round(4).to_string(index=False))
    if len(sens):
        print("\n=== band-width sensitivity (phase-locked) ===")
        print(sens.round(4).to_string(index=False))
    print(f"\nshared colour scale: |t| max = {vmax:.3f}; permutations: {n_perm}")

    # ---- checks
    checks = Checks(enabled=bool(p.get("checks_enabled", True) and paper and published))
    if not published:
        print("\n(n_perm != 1000: published p-values not checked)")
    if checks.enabled:
        ptol = paper.get("p_tol", 0.0006)
        for run, bands in (paper.get("runs") or {}).items():
            for band, want in bands.items():
                r = summary[(summary.run == run) & (summary.band == band)].iloc[0]
                checks.number(f"Fig S4 / 1.5 {run} {band}: best p", r["best_p"], want["p"], ptol)
                if "n_pixels" in want:
                    checks.number(f"Fig S4 / 1.5 {run} {band}: n pixels", r["best_n_pixels"], want["n_pixels"], 0)
        for name, want in (paper.get("sensitivity") or {}).items():
            r = sens[sens["name"] == name]
            if len(r):
                checks.number(f"sensitivity {name}: best p", r["best_p"].iloc[0], want["p"], ptol)
        checks.number("Fig S4 shared colour scale |t| max", vmax, paper.get("global_vmax"), 0.0006)
    checks.report(os.path.join(a.out_dir, "cluster_perm_checks.csv"), strict=a.strict)


if __name__ == "__main__":
    main()
