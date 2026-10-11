#!/usr/bin/env python
"""SI Fig. S2: example FOOOF fits with / without an oscillatory peak, per band.  Replaces analysis-notebook cell 106.

Works from the spectra cache (psd_cache.npy), i.e. from the public deposit; no continuous EEG is needed.

    python scripts/make_figS2.py --psd-cache data/preprocessed/psd_cache.npy --params results/fooof/params.yaml \
        --out-dir results/fooof --fig-dir figures
"""
import argparse
import json
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import numpy as np
import yaml

from olfaction_eeg import fooof_gating as fg
from olfaction_eeg.checks import Checks
from olfaction_eeg.fig_s2 import make_figS2

warnings.filterwarnings("ignore")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--psd-cache", required=True)
    ap.add_argument("--params", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--fig-dir", required=True)
    ap.add_argument("--strict", action="store_true")
    a = ap.parse_args()

    p = yaml.safe_load(open(a.params))
    f = p["fooof"]
    bands = {k: tuple(v) for k, v in p["pac"]["freq_bands"].items()}
    settings = {**f["settings"], "peak_threshold": float(f["default_threshold"])}
    paper = (p.get("paper") or {}).get("fig_s2", {})
    os.makedirs(a.out_dir, exist_ok=True)
    os.makedirs(a.fig_dir, exist_ok=True)

    cache = np.load(a.psd_cache, allow_pickle=True).item()
    from olfaction_eeg.fig_s2 import manual_examples
    absent = [(k, v) for k, v in manual_examples.items() if v[1] not in cache["psd"].get(v[0], {})]
    if absent:
        raise SystemExit(f"Fig. S2 shows hand-picked channels that are not in this spectra cache: {absent}. "
                         "It is meant for the paper's own psd_cache.npy (subjects 23, 29, 32, 33, 34).")
    gate = fg.build_peak_gate(cache, bands, settings)
    fig, chosen = make_figS2(gate, cache, bands, settings=settings, out_prefix=os.path.join(a.fig_dir, "figS2_fooof_examples"))

    rows = {f"{b}_{'peak' if w else 'nopeak'}": list(v) for (b, w), v in chosen.items()}
    json.dump(rows, open(os.path.join(a.out_dir, "figS2_examples.json"), "w"), indent=2)
    print("\nFig. S2 examples (subject, channel):")
    for k, v in rows.items():
        print(f"  {k:18s}: subject {v[0]}, channel {v[1]}")

    checks = Checks(enabled=bool(p.get("checks_enabled", True) and paper))
    if checks.enabled:
        for key, want in (paper.get("examples") or {}).items():
            checks.text(f"Fig S2 {key}: subject, channel", ",".join(rows.get(key, ["missing"])), ",".join(want))
        # every panel's label must agree with the peak gate used for Section 1.2
        for (band, want_peak), (subj, ch) in chosen.items():
            checks.text(f"Fig S2 {band} {subj}/{ch}: peak gate says peak present = {want_peak}",
                        gate[subj][ch][band], want_peak)
    checks.report(os.path.join(a.out_dir, "figS2_checks.csv"), strict=a.strict)


if __name__ == "__main__":
    main()
