#!/usr/bin/env python
"""Section 1.8 / SI Tables S5-S6: did the odor reach the nostrils by inhalation onset?

    python scripts/run_timing.py --epochs-v2 data/preprocessed/epochs_v2_for_timing.npy \
        --continuous data/preprocessed/continuous_respiration_v2.npy --params results/timing/params.yaml --out-dir results/timing
(--epochs-v2 / --continuous also accept the full v2 epochs file / full_respiration_EEG_v2.npy.)
"""
import argparse
import os
import warnings

import numpy as np
import pandas as pd
import yaml

from olfaction_eeg import compat, timing
from olfaction_eeg.checks import Checks

warnings.filterwarnings("ignore")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs-v2", required=True)
    ap.add_argument("--continuous", required=True)
    ap.add_argument("--params", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--strict", action="store_true")
    a = ap.parse_args()

    p = yaml.safe_load(open(a.params))
    cfg = p["timing"]
    pv = (p.get("paper") or {}).get("timing", {})
    tol = (p.get("paper") or {}).get("tolerance", {})
    os.makedirs(a.out_dir, exist_ok=True)

    epochs_v2 = np.load(a.epochs_v2, allow_pickle=True).item()
    continuous = compat.load_pickled_dict(a.continuous)["respiration"]
    df = timing.compute_trial_timing(epochs_v2, continuous, cfg)
    df.to_csv(os.path.join(a.out_dir, "timing_trials.csv"), index=False)

    s5 = timing.lag_table(df)
    s5.to_csv(os.path.join(a.out_dir, "table_S5_lag_by_subject.csv"))
    segments = cfg["segments"]
    transit = timing.transit_time_seconds(cfg["flow_rate_Lmin"], segments)
    s6 = timing.flow_sensitivity_table(df, cfg["flow_rate_Lmin"], segments, np.asarray(cfg["pct_errors"], dtype=float))
    s6.to_csv(os.path.join(a.out_dir, "table_S6_flow_sensitivity.csv"), index=False)
    m = timing.margin_summary(df)
    pd.DataFrame([{**{k: v for k, v in m.items() if k != "late_delays_s"}, "late_delays_s": " ".join(f"{x:.3f}" for x in m["late_delays_s"]),
                   "transit_time_computed_s": transit["total_transit_time_sec"]}]).to_csv(os.path.join(a.out_dir, "timing_summary.csv"), index=False)

    print(f"\ntransit time through the tubing: {transit['total_transit_time_sec']:.4f} s  "
          f"({' + '.join(f'{s['transit_time_sec']:.3f}' for s in transit['segments'])}); the classification uses {cfg['transit_time_measured']} s")
    print(f"{m['n_arrived_in_time']}/{m['n_trials']} trials ({m['pct_in_time']:.2f}%) had the odor at the nostrils by inhale onset; "
          f"late: {m['n_late']} trials, delays {m['late_delays_s']} s; median margin {m['median_margin_s']:.2f} s, 5th percentile {m['p5_margin_s']:.2f} s")
    print(s5.to_string()); print(s6.to_string(index=False))

    checks = Checks(enabled=bool(p.get("checks_enabled", True) and pv))
    if checks.enabled:
        checks.number("transit time, total (s)", transit["total_transit_time_sec"], pv.get("transit_time_s"), 0.001)
        for seg, (name, want) in zip(transit["segments"], (pv.get("transit_segments_s") or {}).items()):
            checks.number(f"transit time, segment '{name}' (s)", seg["transit_time_sec"], want, 0.001)
        checks.number("number of trials analysed", m["n_trials"], pv.get("n_trials"), 0)
        checks.number("trials with odor in time", m["n_arrived_in_time"], pv.get("n_arrived_in_time"), 0)
        checks.number("trials with odor late", m["n_late"], pv.get("n_late"), 0)
        checks.number("median margin (s)", m["median_margin_s"], pv.get("median_margin_s"), 0.006)
        checks.number("5th percentile margin (s)", m["p5_margin_s"], pv.get("p5_margin_s"), 0.006)
        for i, (got, want) in enumerate(zip(m["late_delays_s"], sorted(pv.get("late_delays_s") or []))):
            checks.number(f"late trial {i + 1}: delay after onset (s)", got, want, 0.006)
        for subj, (mean, sd, n) in (pv.get("table_s5") or {}).items():
            r = s5.loc[subj]
            checks.number(f"Table S5 subject {subj}: mean lag", r["mean_lag_sec"], mean, 0.001)
            checks.number(f"Table S5 subject {subj}: SD", r["sd_lag_sec"], sd, 0.001)
            checks.number(f"Table S5 subject {subj}: n trials", r["n_trials"], n, 0)
        for pct, (flow, tt, nc, pres) in (pv.get("table_s6") or {}).items():
            r = s6[np.isclose(s6["pct_error_in_flow_rate"], float(pct))].iloc[0]
            checks.number(f"Table S6 {pct}%: flow rate", r["assumed_flow_rate_Lmin"], flow, 0.0001)
            checks.number(f"Table S6 {pct}%: transit time", r["transit_time_sec"], tt, 0.001)
            checks.number(f"Table S6 {pct}%: trials late", r["n_contaminated"], nc, 0)
            checks.number(f"Table S6 {pct}%: % odor present", r["pct_odor_present_at_onset"], pres, 0.006)
    checks.report(os.path.join(a.out_dir, "timing_checks.csv"), strict=a.strict)


if __name__ == "__main__":
    main()
