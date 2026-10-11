#!/usr/bin/env python
"""Stage 5a: respiration-EEG phase-amplitude coupling for every channel / band / subject (SLOW: hours on real data).

Replaces the loop that was commented out in analysis-notebook cells 89/90.  The result file is what Fig. 2 and the
oscillatory-gating analyses read; you can place a previously computed `pac_results_humidity*.npy` at the output path
and Snakemake will skip this step.

    python scripts/run_pac.py --recordings data/preprocessed/full_respiration_EEG_humidity.npy \
        --params results/pac/params.yaml --out data/preprocessed/pac_results_humidity.npy
"""
import argparse
import os
import sys

import mne
import numpy as np
import yaml

from olfaction_eeg import compat, pac


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recordings", required=True)
    ap.add_argument("--params", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    p = yaml.safe_load(open(a.params))
    mne.set_log_level("WARNING")
    recordings = compat.load_pickled_dict(a.recordings)
    subjects = [s for s in p["normosmia"] if s in recordings["EEG"]]
    print(f"PAC for {len(subjects)} subjects, {p['pac']['n_surrogates']} surrogates, seed mode {p['pac']['seed_mode']}", flush=True)
    results = pac.run_pac(recordings, subjects, p["pac"], seed_mode=p["pac"]["seed_mode"])
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.save(a.out, results, allow_pickle=True)
    print(f"Saved {a.out}: {len(results['subject_level'])} subjects")


if __name__ == "__main__":
    main()
