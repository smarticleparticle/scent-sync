#!/usr/bin/env python
"""Small input files for the odor-arrival timing analysis (both are megabytes instead of gigabytes; part of the data deposit).

  --out-epochs      respiration epochs + event timestamps + EEG array shapes of the v2 epochs file (no EEG)
  --out-continuous  continuous respiration (+ event times, metadata) of full_respiration_EEG_v2.npy (no continuous EEG)

    python scripts/make_timing_inputs.py --epochs-v2 epochs_eeg_respiration_withtimestamps_v2.npy --full-recordings full_respiration_EEG_v2.npy \
        --out-epochs data/preprocessed/epochs_v2_for_timing.npy --out-continuous data/preprocessed/continuous_respiration_v2.npy
"""
import argparse
import os

import numpy as np

from olfaction_eeg import compat, deposit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs-v2", required=True)
    ap.add_argument("--full-recordings", required=True)
    ap.add_argument("--out-epochs", required=True)
    ap.add_argument("--out-continuous", required=True)
    a = ap.parse_args()
    epochs = np.load(a.epochs_v2, allow_pickle=True).item()
    full = compat.load_pickled_dict(a.full_recordings)
    for out in (a.out_epochs, a.out_continuous):
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    np.save(a.out_epochs, deposit.extract_v2_for_timing(epochs), allow_pickle=True)
    np.save(a.out_continuous, deposit.extract_continuous_respiration(full), allow_pickle=True)
    print(f"Saved {a.out_epochs} ({os.path.getsize(a.out_epochs) / 1e6:.1f} MB) and {a.out_continuous} ({os.path.getsize(a.out_continuous) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
