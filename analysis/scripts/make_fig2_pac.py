#!/usr/bin/env python
"""Fig. 2 and the numbers of paper Section 1.1: whole-head PAC statistics, cluster tests, topoplots.

    python scripts/make_fig2_pac.py --pac-results pac_results_humidity.npy --recordings-meta ..._meta.npy \
        --info-fif data/preprocessed/info/subject22_info.fif --params results/pac/params.yaml \
        --out-dir results/pac --fig-dir figures
"""
import argparse
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import mne
import numpy as np
import pandas as pd
import yaml

from olfaction_eeg import compat, pac_figure, pac_stats
from olfaction_eeg.checks import Checks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pac-results", required=True)
    ap.add_argument("--recordings-meta", required=True)
    ap.add_argument("--info-fif", required=True)
    ap.add_argument("--params", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--fig-dir", required=True)
    ap.add_argument("--strict", action="store_true")
    a = ap.parse_args()

    p = yaml.safe_load(open(a.params))
    paper = p.get("paper") or {}
    mne.set_log_level("WARNING")
    pac_stats.MIN_CHANNELS_WHOLEHEAD = pac_figure.MIN_CHANNELS_WHOLEHEAD = p["pac"]["min_channels_wholehead"]
    pac_stats.N_PERMUTATIONS_CLUSTER = pac_figure.N_PERMUTATIONS_CLUSTER = p["pac"]["n_permutations_cluster"]

    pac_results = np.load(a.pac_results, allow_pickle=True).item()
    meta = compat.load_pickled_dict(a.recordings_meta)
    subject_info = pac_figure.load_plot_info(a.info_fif)

    fig, clusters, df_obs, group_stats = pac_figure.build_fig2(pac_results, meta, subject_info)
    os.makedirs(a.out_dir, exist_ok=True)
    os.makedirs(a.fig_dir, exist_ok=True)
    fig.savefig(os.path.join(a.fig_dir, "fig2_pac.pdf"))
    fig.savefig(os.path.join(a.fig_dir, "fig2_pac.png"), dpi=300, bbox_inches="tight", facecolor="white")

    effects = pac_stats.effect_sizes(df_obs)
    df_obs.to_csv(os.path.join(a.out_dir, "pac_subject_level.csv"), index=False)
    table = group_stats.merge(effects[["band", "cohens_d"]], on="band")
    table.to_csv(os.path.join(a.out_dir, "pac_group_stats.csv"), index=False)
    rows = [{"band": b, "cluster": j, "p_value": c["p_value"], "n_channels": c["n_channels"],
             "primary_region": c.get("primary_region"), "channels": " ".join(c["channels"])}
            for b, r in clusters.items() for j, c in enumerate(r["significant_clusters"], start=1)]
    pd.DataFrame(rows, columns=["band", "cluster", "p_value", "n_channels", "primary_region", "channels"]).to_csv(
        os.path.join(a.out_dir, "pac_clusters.csv"), index=False)
    with open(os.path.join(a.out_dir, "pac_stats_results.pkl"), "wb") as f:
        pickle.dump({"group_stats": group_stats, "df_obs": df_obs, "effects": effects,
                     "clusters": {b: {k: v for k, v in r.items() if k not in ("adjacency", "data_matrix", "H0")}
                                  for b, r in clusters.items()}}, f)
    print(table.round(4).to_string(index=False))

    # ---------------- checks against the published numbers
    pv = paper.get("pac", {})
    tol = paper.get("tolerance", {})
    checks = Checks(enabled=bool(p.get("checks_enabled", True) and pv))
    if checks.enabled:
        checks.number("PAC: n subjects", int(table["n_subjects"].max()), pv.get("n_subjects"), 0)
        for band, want in (pv.get("whole_head") or {}).items():
            row = table[table["band"] == band].iloc[0]
            checks.number(f"PAC {band}: p (FDR)", row["p_fdr"], want.get("p"), tol.get("p", 0.002))
            checks.number(f"PAC {band}: Cohen's d", row["cohens_d"], want.get("d"), 0.01)
        for band, want in (pv.get("clusters") or {}).items():
            got = sorted([(c["p_value"], c["n_channels"]) for c in clusters[band]["significant_clusters"]])
            checks.number(f"PAC {band}: number of significant clusters", len(got), len(want), 0)
            for j, (g, w) in enumerate(zip(got, sorted((x["p"], x["n"]) for x in want)), start=1):
                checks.number(f"PAC {band} cluster {j}: p", g[0], w[0], tol.get("p", 0.002))
                checks.number(f"PAC {band} cluster {j}: n channels", g[1], w[1], 0)
    checks.report(os.path.join(a.out_dir, "pac_checks.csv"), strict=a.strict)


if __name__ == "__main__":
    main()
