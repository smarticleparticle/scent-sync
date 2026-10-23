#!/usr/bin/env python
"""Does the PAC computation reproduce your saved PAC results?  (a few channels of one subject, minutes instead of hours)

Recomputes the first N usable channels of one subject, in the original processing order, with the original seed, and compares
with the saved results:
  * MI_observed is deterministic -> must match exactly for ANY subject/channel if the computation is the same;
  * the surrogate distribution (MI_null_distribution) matches too if the original run was one sequential pass seeded once
    (np.random.seed(42), subjects in config order) and the chosen subject is the FIRST one processed.

    python scripts/check_pac_subset.py --recordings full_respiration_EEG_humidity.npy --pac-results pac_results_humidity....npy \
        --params results_paper/pac/params.yaml --n-channels 3
"""
import argparse
import sys
from fractions import Fraction

import mne
import numpy as np
import yaml
from scipy.signal import resample_poly

from olfaction_eeg import compat, pac


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recordings", required=True)
    ap.add_argument("--pac-results", required=True)
    ap.add_argument("--params", required=True)
    ap.add_argument("--subject", default=None, help="default: the first subject of the config that has saved results")
    ap.add_argument("--n-channels", type=int, default=3)
    a = ap.parse_args()
    p = yaml.safe_load(open(a.params))
    cfg = p["pac"]
    mne.set_log_level("WARNING")

    saved = np.load(a.pac_results, allow_pickle=True).item()
    recordings = compat.load_pickled_dict(a.recordings)
    in_order = [s for s in p["normosmia"] if s in saved["subject_level"]]
    subject = a.subject or in_order[0]
    is_first = subject == in_order[0]
    print(f"subject {subject} ({'the first' if is_first else 'NOT the first'} subject of the original pass); saved settings: "
          f"{ {k: saved['metadata'].get(k) for k in ('n_surrogates', 'min_shift_sec')} }")
    if saved["metadata"].get("n_surrogates") != cfg["n_surrogates"]:
        print(f"WARNING: config n_surrogates={cfg['n_surrogates']} differs from the saved file's {saved['metadata'].get('n_surrogates')}")

    raw = recordings["EEG"][subject]
    resp = recordings["respiration"][subject]
    bad = recordings["metadata"][subject]["bad_channels"]
    eeg = raw.get_data()
    freq_bands = {k: tuple(v) for k, v in cfg["freq_bands"].items()}
    ratio = Fraction(cfg["sfreq_eeg"] / cfg["sfreq_resp"]).limit_denominator(1000)
    print("respiration quality check passes:", pac.check_respiration_quality(resample_poly(resp, ratio.numerator, ratio.denominator), cfg["sfreq_eeg"], tuple(cfg["resp_band"])))

    np.random.seed(cfg["seed"])
    n_done, n_obs_ok, n_null_ok, n_total = 0, 0, 0, 0
    for ch_idx, ch in enumerate(raw.ch_names):
        if ch in bad or n_done >= a.n_channels:
            continue
        res = pac.compute_pac_continuous_linearized_direct(
            eeg_signal=eeg[ch_idx, :], resp_signal=resp, sfreq_eeg=cfg["sfreq_eeg"], sfreq_resp=cfg["sfreq_resp"],
            resp_band=tuple(cfg["resp_band"]), freq_bands=freq_bands, n_phase_bins=cfg["n_phase_bins"],
            n_surrogates=cfg["n_surrogates"], min_shift_sec=cfg["min_shift_sec"],
            min_cycle_duration=cfg["min_cycle_duration"], max_cycle_duration=cfg["max_cycle_duration"], verbose=False)
        n_done += 1
        for band in freq_bands:
            s = (saved["subject_level"][subject].get(ch) or {}).get(band)
            r = res.get(band)
            if s is None or r is None:
                print(f"  {ch} {band}: saved {'missing' if s is None else 'present'}, recomputed {'missing' if r is None else 'present'}")
                continue
            n_total += 1
            obs_ok = bool(np.isclose(s["MI_observed"], r["MI_observed"], rtol=1e-9, atol=0))
            null_ok = bool(np.array_equal(np.asarray(s["MI_null_distribution"]), np.asarray(r["MI_null_distribution"])))
            n_obs_ok += obs_ok
            n_null_ok += null_ok
            print(f"  {ch:5s} {band:10s} MI_observed saved {s['MI_observed']:.6f} / now {r['MI_observed']:.6f} {'OK' if obs_ok else 'DIFFERENT'} | "
                  f"surrogates {'identical' if null_ok else 'different'}")
    print(f"\n=== {n_obs_ok}/{n_total} observed MI identical; {n_null_ok}/{n_total} surrogate distributions identical ===")
    if n_total and n_obs_ok == n_total:
        print("The PAC computation (phase, filtering, Hilbert amplitude, modulation index) reproduces your saved results.")
        if n_null_ok == n_total:
            print("Even the random surrogates are identical: the saved file came from a sequential pass seeded with the config seed.")
        else:
            print("Surrogates differ: expected unless this subject/channel was the first of a pass seeded once; observed values are what matter.")
    else:
        print("The observed MI differs: the computation (or its parameters / the input recordings) is not the one that made the saved file.")


if __name__ == "__main__":
    sys.exit(main())
