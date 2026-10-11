#!/usr/bin/env python
"""Fig. 4A/B (selection-frequency heatmaps + ROC curves) and Fig. S3 (per-odor selection frequency).

Fig. 4C/D (odor-recognition scoring) come from the word-embedding analysis, which is not ported yet.
    python scripts/make_fig4.py --results results/classification/classification_results.pkl --out-dir figures
"""
import argparse
import os
import pickle

import matplotlib
matplotlib.use("Agg")

from olfaction_eeg import figures


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    R = pickle.load(open(args.results, "rb"))
    p, per_odor = R["params"], R["per_odor"]
    phase_bins = list(p["phase_strategies"][p["phase_main_strategy"]])
    time_bins = list(p["time_strategies_378s"][p["time_378s_comparison_strategy"]])
    os.makedirs(args.out_dir, exist_ok=True)

    fig4 = figures.build_fig4_AB(per_odor, phase_bins, time_bins)
    for ext in ("png", "pdf"):
        fig4.savefig(os.path.join(args.out_dir, f"fig4_AB_classification.{ext}"), dpi=300, bbox_inches="tight")

    figS3 = figures.build_figS3(figures.figS3_data(per_odor, phase_bins))
    for ext in ("png", "pdf"):
        figS3.savefig(os.path.join(args.out_dir, f"figS3_per_odor_selection.{ext}"), dpi=300, bbox_inches="tight")
    print(f"Saved Fig. 4A/B and Fig. S3 to {args.out_dir}")


if __name__ == "__main__":
    main()
