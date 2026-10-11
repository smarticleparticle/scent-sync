#!/usr/bin/env python
"""Stage 0a: EEGLAB-preprocessed EEG + olfactometer humidity CSV  ->  epoched EEG/respiration/timestamps.

Replaces notebook cell 2 of epochs_eeg_respiration_withtimestamps.ipynb (output:
epochs_eeg_respiration_withtimestamps_v2.npy).  All paths and per-subject corrections come from the
params file written by the workflow (config.yaml -> recordings).

    python scripts/make_epochs.py --params results/recordings/params.yaml --out data/preprocessed/epochs_..._v2.npy
"""
import argparse
import os

import mne
import numpy as np
import yaml

from olfaction_eeg import recordings as rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    p = yaml.safe_load(open(args.params))
    mne.set_log_level("WARNING")
    dataset = rec.build_epochs(p["subjects"], p["eeglab_fmt"], p["csv_fmt"], p["cfg"], verbose=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.save(args.out, dataset)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
