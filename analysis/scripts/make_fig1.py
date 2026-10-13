#!/usr/bin/env python
"""Fig. 1 for the journal: the hand-made olfactometer / participant illustration (figure_sources/figure1_left_infographic.svg) + the respiration trace drawn from data.

    python scripts/make_fig1.py --trial-file data/preprocessed/fig1_example_trial.npy --out-prefix figures/fig1

Writes figures/fig1.svg (editable: open it in Inkscape), and, if Inkscape or CairoSVG is available, fig1.pdf / fig1.eps / fig1.png (600 dpi preview).  The page is 180 x 96 mm
(full-page PNAS width).  The illustration is placed unchanged.  Printed text sizes are reported: the journal asks for 6-12 pt.  Without --trial-file a PLACEHOLDER trace is drawn
(layout preview only, marked "PREVIEW").
Options: --drop-tiny-text removes text that would print below 6 pt (the digits and "L/min" on the flow meters, "UNO" / "Arduino" on the board).
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import numpy as np

from olfaction_eeg import fig1_compose as fc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--left-svg", default="figure_sources/figure1_left_infographic.svg")
    ap.add_argument("--trial-file", default=None)
    ap.add_argument("--trial-idx", type=int, default=0)
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--formats", default="svg,pdf,eps,png")
    ap.add_argument("--png-dpi", type=int, default=600)
    ap.add_argument("--inkscape", default=None, help="path of the Inkscape executable (found automatically if it is on the PATH or in /Applications)")
    ap.add_argument("--graph-width-mm", type=float, default=fc.GRAPH_W_MM, help="width of the plot area (default: %(default)s)")
    ap.add_argument("--graph-height-mm", type=float, default=fc.GRAPH_H_MM, help="height of the plot area (default: %(default)s; the original figure was about 2.6 : 1)")
    ap.add_argument("--graph-bottom-mm", type=float, default=fc.GRAPH_BOTTOM_MM, help="distance of the plot's lower edge from the bottom of the page (default: %(default)s of 96 mm)")
    ap.add_argument("--drop-tiny-text", action="store_true")
    ap.add_argument("--text-to-path", action="store_true", help="with Inkscape: convert all text to outlines in the PDF / EPS (no fonts to include)")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out_prefix) or ".", exist_ok=True)
    if a.trial_file:
        d = np.load(a.trial_file, allow_pickle=True).item()
        trial, placeholder = d["trials"][a.trial_idx], False
    else:
        t = np.arange(70) / 8.757
        trial, placeholder = 77 + 4.5 * np.sin(2 * np.pi * (t - 2.2) / 4.6) - 1.2 * np.sin(2 * np.pi * (t - 2.2) / 9), True
        print("No --trial-file: drawing a PLACEHOLDER trace (layout preview only).")
    svg = f"{a.out_prefix}.svg"
    findings = fc.compose_svg(a.left_svg, trial, svg, trial_idx=a.trial_idx, placeholder=placeholder, drop_tiny_text=a.drop_tiny_text,
                              graph=(fc.GRAPH_LEFT_MM, a.graph_width_mm, a.graph_height_mm, a.graph_bottom_mm))
    rows, small, big = fc.summarize_findings(findings)
    print("\nPrinted text sizes (pt) at the final page size:")
    for pt in sorted(rows):
        items = rows[pt]
        flag = "   <-- below 6 pt" if pt < fc.MIN_PT else ("   <-- above 12 pt" if pt > fc.MAX_PT else "")
        print(f"  {pt:5.1f}  {len(items):3d} x  {', '.join(sorted(set(items)))[:70]}{flag}")
    if small:
        print(f"\n{small} text item(s) would print below 6 pt (the journal minimum). They are unreadable at that size; run again with --drop-tiny-text to remove them, or select them in"
              " Inkscape and use Path > Object to Path.")
    formats = [f for f in a.formats.split(",") if f]
    made = fc.export(svg, a.out_prefix, formats=[f for f in formats if f != "svg"], png_dpi=a.png_dpi, inkscape=a.inkscape, text_to_path=a.text_to_path)
    print()
    for f in formats:
        p = svg if f == "svg" else made.get(f)
        print(f"{f:4s} {p + '  ' + format(os.path.getsize(p) / 1e6, '.2f') + ' MB' if p else 'NOT written (needs Inkscape or CairoSVG: pip install cairosvg)'}")
    if any(v is None for v in made.values()):
        print("\nTo get the PDF / EPS without the script: open the .svg in Inkscape, File > Save a Copy... > PDF (or EPS); leave 'Embed fonts' on, or convert text to paths.")


if __name__ == "__main__":
    main()
