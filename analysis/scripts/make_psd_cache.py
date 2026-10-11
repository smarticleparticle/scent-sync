#!/usr/bin/env python
"""Welch spectra of every channel (for the FOOOF oscillatory gating) -- the only step that needs the continuous EEG.

The result is a few MB and part of the data deposit; Section 1.2 / Table S2 can be regenerated from it alone.

    python scripts/make_psd_cache.py --recordings data/preprocessed/full_respiration_EEG_humidity.npy \
        --pac-results data/preprocessed/pac_results_humidity.npy --params results/fooof/params.yaml --out data/preprocessed/psd_cache.npy
"""
import argparse
import os

import mne
import numpy as np
import yaml

from olfaction_eeg import compat
from olfaction_eeg import fooof_gating as fg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recordings", required=True)
    ap.add_argument("--pac-results", required=True)
    ap.add_argument("--params", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    p = yaml.safe_load(open(a.params))
    mne.set_log_level("WARNING")
    recordings = compat.load_pickled_dict(a.recordings)
    pac_results = np.load(a.pac_results, allow_pickle=True).item()
    cache = fg.compute_psd_cache(recordings, pac_results, p["normosmia"], sfreq_eeg=p["pac"]["sfreq_eeg"],
                                 sfreq_resp=p["pac"]["sfreq_resp"], resp_band=tuple(p["pac"]["resp_band"]))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.save(a.out, cache, allow_pickle=True)
    n = sum(len(v) for v in cache["psd"].values())
    print(f"Saved {a.out}: spectra of {n} channels in {len(cache['psd'])} subjects")


if __name__ == "__main__":
    main()
