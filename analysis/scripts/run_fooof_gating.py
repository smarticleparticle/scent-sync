#!/usr/bin/env python
"""Section 1.2 / Table S2: is respiration-EEG PAC carried by genuine oscillations?

(1) fraction of channels with a FOOOF-detected peak per band, (2) group PAC restricted to channels with a peak, at four
peak-detection thresholds (Table S2) and compared with the all-channel result, (3) inhale-vs-exhale aperiodic offset/exponent.
Works from the spectra cache (make_psd_cache.py) and the PAC results.

    python scripts/run_fooof_gating.py --psd-cache psd_cache.npy --pac-results pac_results.npy --params results/fooof/params.yaml --out-dir results/fooof
"""
import argparse
import os

import numpy as np
import pandas as pd
import yaml

from olfaction_eeg import fooof_gating as fg
from olfaction_eeg.checks import Checks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--psd-cache", required=True)
    ap.add_argument("--pac-results", required=True)
    ap.add_argument("--params", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--strict", action="store_true")
    a = ap.parse_args()

    p = yaml.safe_load(open(a.params))
    f = p["fooof"]
    bands = {k: tuple(v) for k, v in p["pac"]["freq_bands"].items()}
    subjects = p["normosmia"]
    cache = np.load(a.psd_cache, allow_pickle=True).item()
    pac_results = np.load(a.pac_results, allow_pickle=True).item()
    os.makedirs(a.out_dir, exist_ok=True)
    base = dict(f["settings"])

    gates, rows_s2 = {}, []
    for thr in f["thresholds"]:
        gate = fg.build_peak_gate(cache, bands, {**base, "peak_threshold": float(thr)})
        gates[float(thr)] = gate
        stats = fg.run_gated_stats(pac_results, gate, bands, min_channels=f["min_channels"], n_perm=f["n_perm"], seed=f["seed"])
        for b in bands:
            rows_s2.append({"threshold_sd": float(thr), "band": b, **stats[b]})
        print(f"threshold {thr} SD done", flush=True)
    s2 = pd.DataFrame(rows_s2)
    s2.to_csv(os.path.join(a.out_dir, "table_S2_long.csv"), index=False)
    s2.assign(cell=lambda d: d.apply(lambda r: f"{r['d']:.2f} ({int(r['n'])})", axis=1)).pivot(
        index="threshold_sd", columns="band", values="cell")[list(bands)].to_csv(os.path.join(a.out_dir, "table_S2.csv"))

    dflt = float(f["default_threshold"])
    prev = fg.peak_prevalence(gates[dflt], bands)
    pd.DataFrame([{"band": b, **v} for b, v in prev.items()]).to_csv(os.path.join(a.out_dir, "peak_prevalence.csv"), index=False)
    comp = {b: fg.compare_gated_vs_ungated(pac_results, gates[dflt], b, min_channels=f["min_channels"], n_perm=f["n_perm"], seed=f["seed"]) for b in bands}
    comp_df = pd.DataFrame([{"band": b, **v} for b, v in comp.items()])
    comp_df["p_gated_fdr"] = fg.fdr_bh(comp_df["p_gated"].values)
    comp_df.to_csv(os.path.join(a.out_dir, "gated_vs_all_channels.csv"), index=False)
    ap_res = fg.aperiodic_inhale_exhale(cache, subjects, n_perm=f["n_perm"], seed=f["seed"])
    pd.DataFrame([{k: v for k, v in ap_res.items() if not k.startswith("subject_")}]).to_csv(
        os.path.join(a.out_dir, "aperiodic_inhale_exhale.csv"), index=False)
    print(prev, "\n", comp_df.round(3).to_string(index=False), "\n", {k: v for k, v in ap_res.items() if not k.startswith("subject_")})

    # ---------------- checks against the paper
    pv = (p.get("paper") or {}).get("fooof", {})
    tol = (p.get("paper") or {}).get("tolerance", {})
    checks = Checks(enabled=bool(p.get("checks_enabled", True) and pv))
    if checks.enabled:
        for b, want in (pv.get("peak_prevalence_pct") or {}).items():
            checks.number(f"FOOOF peak prevalence {b} (%)", prev[b]["pct"], want, 0.1)
        checks.number("FOOOF channel-subject instances", prev["alpha"]["n"], pv.get("n_instances"), 0)
        for b, want in (pv.get("gated_default") or {}).items():
            r = comp_df[comp_df.band == b].iloc[0]
            checks.number(f"gated PAC {b}: n subjects", r["n_gated"], want.get("n"), 0)
            checks.number(f"gated PAC {b}: Cohen's d", r["d_gated"], want.get("d"), 0.01)
            checks.number(f"gated PAC {b}: p (FDR)", r["p_gated_fdr"], want.get("p"), tol.get("p", 0.002))
        for b, want in (pv.get("paired_gated_vs_all_p") or {}).items():
            checks.number(f"gated vs all channels, paired p, {b}", comp_df[comp_df.band == b].iloc[0]["p_paired"], want, tol.get("p", 0.002))
        for thr, bands_want in (pv.get("table_s2") or {}).items():
            for b, (d_want, n_want) in bands_want.items():
                r = s2[(s2.threshold_sd == float(thr)) & (s2.band == b)].iloc[0]
                checks.number(f"Table S2 {thr} SD {b}: d", r["d"], d_want, 0.01)
                checks.number(f"Table S2 {thr} SD {b}: n", r["n"], n_want, 0)
        checks.number("Table S2: beta p (FDR) at 1.5 SD", s2[(s2.threshold_sd == 1.5) & (s2.band == "beta")].iloc[0]["p_fdr"], pv.get("table_s2_beta_p_fdr_at_1_5"), tol.get("p", 0.002))
        ap_want = pv.get("aperiodic") or {}
        checks.number("aperiodic exponent (inhale-exhale): d", ap_res["exponent_d"], (ap_want.get("exponent") or {}).get("d"), 0.01)
        checks.number("aperiodic exponent: p", ap_res["exponent_p"], (ap_want.get("exponent") or {}).get("p"), tol.get("p", 0.002))
        checks.number("aperiodic offset: p", ap_res["offset_p"], (ap_want.get("offset") or {}).get("p"), tol.get("p", 0.002))
    checks.report(os.path.join(a.out_dir, "fooof_checks.csv"), strict=a.strict)


if __name__ == "__main__":
    main()
