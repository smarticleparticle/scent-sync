#!/usr/bin/env python
"""Extract channel names / electrode positions / bad channels from a (huge) full_respiration_EEG_*.npy.

The PAC cluster statistics and topoplots only need these, not the multi-GB continuous recordings.
    python scripts/make_recordings_meta.py --recordings data/preprocessed/full_respiration_EEG_humidity.npy --out ..._meta.npy
"""
import argparse

import numpy as np

from olfaction_eeg import compat
from olfaction_eeg import recordings as rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recordings", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    full = compat.load_pickled_dict(a.recordings)
    np.save(a.out, rec.make_recordings_meta(full))
    print(f"Saved {a.out} ({len(full['EEG'])} subjects)")


if __name__ == "__main__":
    main()
