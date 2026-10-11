#!/usr/bin/env python
"""Stage 0b: continuous, offset-aligned EEG + respiration (one entry per subject).

Replaces notebook cells 5 and 17 of epochs_eeg_respiration_withtimestamps.ipynb:
    --variant zscore         -> full_respiration_EEG_v2.npy        (clipped + z-scored humidity; timing analyses)
    --variant minmax_smooth  -> full_respiration_EEG_humidity.npy  (min-max + smoothed humidity; PAC / FOOOF)

CAUTION: the saved file contains pickled MNE RawArray objects, so loading it needs a compatible MNE
(your analysis notebook patches sys.modules['mne.io.array.array'] for exactly this reason).
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
    ap.add_argument("--variant", required=True, choices=["zscore", "minmax_smooth"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    p = yaml.safe_load(open(args.params))
    mne.set_log_level("WARNING")
    dataset = rec.build_full_recordings(p["subjects"], p["eeglab_fmt"], p["csv_fmt"], p["cfg"],
                                        args.variant, verbose=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.save(args.out, dataset)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
