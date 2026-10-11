#!/usr/bin/env python
"""Phase-locked (RPL-ERSP) classification AUCs from one or more phase-aligned ERSP files, versus paper Table S1.

Needs ONLY the phase-aligned file (no time-locked ERSPs), so it runs in a few minutes.  Use it to
  * check the classification code against the paper on the paper's own ERSP file, and
  * see whether a regenerated ERSP file changes the AUCs.

    python scripts/check_rpl_from_ersp.py --ersp PAPER_phase_aligned.npy results/ersp/phase_aligned_fullbreath_1deg.npy
"""
import argparse
import os
import sys
import warnings

import numpy as np
import yaml

warnings.filterwarnings("ignore")
from olfaction_eeg import classify as clf  # noqa: E402
from olfaction_eeg.features import extract_binned_features  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ersp", nargs="+", required=True, help="phase-aligned ERSP .npy file(s)")
    ap.add_argument("--config", default=os.path.join(HERE, "..", "config", "config.yaml"))
    ap.add_argument("--paper", default=os.path.join(HERE, "..", "config", "paper_values.yaml"))
    a = ap.parse_args()
    cfg, paper = yaml.safe_load(open(a.config)), yaml.safe_load(open(a.paper))
    C = cfg["classification"]
    odors, subjects = cfg["odors"], cfg["subjects"]["normosmia"]
    table = {}
    for path in a.ersp:
        ersp = np.load(path, allow_pickle=True).item()
        print(f"\n=== {path} ===", flush=True)
        for odor in odors:
            for name, bins in C["phase_strategies"].items():
                df = extract_binned_features(ersp, "water", odor, subjects, freqs=ersp["freqs"], x_bins=bins, x=ersp["x"],
                                             trial_start=C["trial_start"], trial_end=C["trial_end"])
                res = clf.nested_loso_selk_logreg(df, C=C["C"], verbose=False)
                table[(path, odor, name)] = res["auc"]
                want = paper["table_s1"][odor]["rpl"][name]
                print(f"  {odor:11s} {name:13s} AUC {res['auc']:.4f}   paper {want:.3f}   diff {res['auc'] - want:+.4f}", flush=True)
    print("\n=== summary: largest |AUC - Table S1| per file ===")
    for path in a.ersp:
        diffs = [abs(table[(path, o, n)] - paper["table_s1"][o]["rpl"][n]) for o in odors for n in C["phase_strategies"]]
        diffs4 = [abs(table[(path, o, "4-bin")] - paper["table_s1"][o]["rpl"]["4-bin"]) for o in odors]
        print(f"  {os.path.basename(path)[:60]:60s} all 9: {max(diffs):.4f}   4-bin (the reported ones): {max(diffs4):.4f}")
    print("\n(Table S1 prints 3 decimals, so differences below 0.001 are a match.)")


if __name__ == "__main__":
    sys.exit(main())
