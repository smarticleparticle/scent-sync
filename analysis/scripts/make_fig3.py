#!/usr/bin/env python
"""Fig. 3: respiratory phase-locked ERSPs (schematic, two example subjects, grand averages with respiration overlays).

Replaces analysis-notebook cell 60 (main figure).  Needs the ORIGINAL epochs file (respiration of every trial; ~1 min to detect the first breath), the
info folder, the paper's phase-aligned ERSP and the 3.78-s time-locked ERSP.

    python scripts/make_fig3.py --epochs data/preprocessed/epochs_eeg_respiration.npy --info-dir data/preprocessed/info \
        --phase results_paper/ersp/phase_aligned_fullbreath_1deg.npy --time-long results_paper/ersp/time_aligned_long.npy \
        --params results_paper/fig3/params.yaml --out-dir results_paper/fig3 --fig-dir figures_paper
"""
import argparse
import glob
import json
import os
import re
import warnings

import matplotlib
matplotlib.use("Agg")
import numpy as np
import yaml

from olfaction_eeg import breath
from olfaction_eeg.checks import Checks
from olfaction_eeg.fig3 import create_main_figure_rearranged

warnings.filterwarnings("ignore")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", required=True, help="the ORIGINAL epochs file")
    ap.add_argument("--info-dir", required=True)
    ap.add_argument("--phase", required=True)
    ap.add_argument("--time-long", required=True)
    ap.add_argument("--params", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--fig-dir", required=True)
    ap.add_argument("--strict", action="store_true")
    a = ap.parse_args()

    p = yaml.safe_load(open(a.params))
    cfg, paper = p["fig3"], (p.get("paper") or {}).get("fig3", {})
    os.makedirs(a.out_dir, exist_ok=True)
    os.makedirs(a.fig_dir, exist_ok=True)

    missing = [s for s in cfg["subjects_individual"] if s not in p["normosmia"]]
    if missing:
        raise SystemExit(f"fig3.subjects_individual {missing} are not among subjects.normosmia {p['normosmia']}; set fig3.subjects_individual in the config")
    ids = {m.group(1): [] for f in glob.glob(os.path.join(a.info_dir, "subject*_info.fif"))
           for m in [re.search(r"subject(\d+)_info\.fif$", f)] if m}
    epochs = np.load(a.epochs, allow_pickle=True).item()
    breath_data = breath.extract_first_breath_data(epochs, ids, conditions=breath.CONDITIONS, store_eeg=False)
    del epochs
    phase = np.load(a.phase, allow_pickle=True).item()
    time_long = np.load(a.time_long, allow_pickle=True).item()

    pdf = os.path.join(a.fig_dir, "fig3_ersps.pdf")
    fig = create_main_figure_rearranged(
        phase, time_long, breath_data,
        subjects_individual=cfg["subjects_individual"], subjects_grand=p["normosmia"], odor_list=p["odors"],
        n_trials=cfg["n_trials"], freq_min=4, freq_max=40, time_max_long=3.78, percentile=99, save_path=pdf)
    info = fig.fig3_info
    json.dump({k: (v if not isinstance(v, (np.floating, np.integer)) else float(v)) for k, v in info.items()},
              open(os.path.join(a.out_dir, "fig3_info.json"), "w"), indent=2, default=float)
    print("\nFig. 3 numbers:", {k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v) for k, v in info.items()})

    checks = Checks(enabled=bool(p.get("checks_enabled", True) and paper))
    if checks.enabled:
        tol = paper.get("clim_tol", 0.0006)
        checks.number("Fig 3 colour limit, panels B-C (|vmax|)", info["vmax_subj"], paper.get("vmax_subj"), tol)
        checks.number("Fig 3 colour limit, panels D-E (|vmax|)", info["vmax_grand"], paper.get("vmax_grand"), tol)
        checks.number("Fig 3 subjects in the phase-locked grand average", info["n_grand_phase"], paper.get("n_grand"), 0)
        checks.number("Fig 3 subjects in the time-locked grand average", info["n_grand_time"], paper.get("n_grand"), 0)
        checks.number("Fig 3 subjects with a respiration trace", info["n_resp_traces"], paper.get("n_grand"), 0)
        checks.text("Fig 3 subjects shown in panels B-C", ",".join(info["subjects_individual"]), ",".join(paper.get("subjects_individual", [])) or None)
    checks.report(os.path.join(a.out_dir, "fig3_checks.csv"), strict=a.strict)


if __name__ == "__main__":
    main()
