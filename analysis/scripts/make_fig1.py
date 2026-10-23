#!/usr/bin/env python
"""Fig. 1 for the journal as one vector PDF: the hand-made olfactometer / participant illustration + the respiration trace drawn from data.

    python scripts/make_fig1.py --trial-file data/preprocessed/fig1_example_trial.npy --out-prefix figures/fig1

Writes figures/fig1.pdf (180 x 96 mm, the width of a full-page PNAS figure). The illustration (figure_sources/figure1_left_infographic.pdf, exported from the editable .svg next to it)
is placed unchanged. The printed size of every text of the trace is reported: the journal asks for 6-12 pt. Without --trial-file a PLACEHOLDER trace is drawn (layout preview only, marked "PREVIEW").
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import numpy as np

from olfaction_eeg import fig1_compose as fc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--left-pdf", default="figure_sources/figure1_left_infographic.pdf")
    ap.add_argument("--trial-file", default=None)
    ap.add_argument("--trial-idx", type=int, default=0)
    ap.add_argument("--out-prefix", required=True, help="the figure is written to <out-prefix>.pdf")
    ap.add_argument("--graph-width-mm", type=float, default=fc.GRAPH_W_MM, help="width of the plot area (default: %(default)s)")
    ap.add_argument("--graph-height-mm", type=float, default=fc.GRAPH_H_MM, help="height of the plot area (default: %(default)s)")
    ap.add_argument("--graph-bottom-mm", type=float, default=fc.GRAPH_BOTTOM_MM, help="distance of the plot's lower edge from the bottom of the page (default: %(default)s of 96 mm)")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out_prefix) or ".", exist_ok=True)
    if a.trial_file:
        d = np.load(a.trial_file, allow_pickle=True).item()
        trial, placeholder = d["trials"][a.trial_idx], False
    else:
        t = np.arange(70) / 8.757
        trial, placeholder = 77 + 4.5 * np.sin(2 * np.pi * (t - 2.2) / 4.6) - 1.2 * np.sin(2 * np.pi * (t - 2.2) / 9), True
        print("No --trial-file: drawing a PLACEHOLDER trace (layout preview only).")
    out = f"{a.out_prefix}.pdf"
    findings = fc.compose_pdf(a.left_pdf, trial, out, trial_idx=a.trial_idx, placeholder=placeholder,
                              graph=(fc.GRAPH_LEFT_MM, a.graph_width_mm, a.graph_height_mm, a.graph_bottom_mm))
    rows, small, big = fc.summarize_findings(findings)
    print("\nPrinted text sizes of the plot (pt) at the final page size (the illustration has no live text):")
    for pt in sorted(rows):
        flag = "   <-- outside 6-12 pt" if (pt < fc.MIN_PT or pt > fc.MAX_PT) else ""
        print(f"  {pt:5.1f}  {len(rows[pt]):3d} x  {', '.join(sorted(set(rows[pt])))[:70]}{flag}")
    print(f"\n{out}  {os.path.getsize(out) / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
