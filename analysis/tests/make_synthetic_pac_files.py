"""Write the files of the synthetic PAC test (config/test_pac.yaml) into the given directory.

    python tests/make_synthetic_pac_files.py tests/data/pac
"""
import os
import sys

import mne
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import make_synthetic_pac as msp  # noqa: E402


def main(out_dir, subjects=("22", "23", "24", "26"), n_ch=6):
    os.makedirs(os.path.join(out_dir, "info"), exist_ok=True)
    recs = msp.make_recordings(subjects=subjects, n_ch=n_ch, duration_s=240, seed=3)
    np.save(os.path.join(out_dir, "full_respiration_EEG_humidity.npy"), recs, allow_pickle=True)
    mne.io.write_info(os.path.join(out_dir, "info", "subject22_info.fif"), msp.make_info(n_ch, seed=99), overwrite=True)
    print(f"Synthetic PAC test files written to {out_dir}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "tests/data/pac")
