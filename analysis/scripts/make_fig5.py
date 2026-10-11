#!/usr/bin/env python
"""Fig. 5: olfactory habituation (AUC early vs late; feature selection early vs late).

    python scripts/make_fig5.py --results results/habituation/habituation_results.pkl --out-dir figures
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
    h = R["params"]["habituation"]
    os.makedirs(args.out_dir, exist_ok=True)
    fig = figures.build_fig5(R["per_odor"], count_unselected_as_zero=h["fig5b_count_unselected_as_zero"],
                             robust_threshold=h["fig5b_robust_threshold"])
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(args.out_dir, f"fig5_habituation.{ext}"), bbox_inches="tight")
    print(f"Saved Fig. 5 to {args.out_dir}")


if __name__ == "__main__":
    main()
