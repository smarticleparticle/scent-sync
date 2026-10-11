"""generate_ersp.py: n_jobs>1 must give exactly the serial result (and must not leak the time-window override)."""
import contextlib
import io
import os
import sys

import numpy as np
import pytest

pytest.importorskip("mne")
pytest.importorskip("pywt")
pytest.importorskip("fire")
HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
import make_synthetic_raw as msr  # noqa: E402

from olfaction_eeg import recordings as rec  # noqa: E402


def same(a, b):
    if isinstance(a, dict):
        return set(a) == set(b) and all(same(a[k], b[k]) for k in a)
    return np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True)


@pytest.fixture(scope="module")
def epochs_and_info(tmp_path_factory):
    d = tmp_path_factory.mktemp("ersp")
    subjects = ["22", "24"]
    old = msr.SUBJECTS
    msr.SUBJECTS = subjects
    try:
        msr.main(str(d / "raw"), str(d / "info"), n_trials=3)
    finally:
        msr.SUBJECTS = old
    cfg = dict(tmin=-1.0, tmax=15.0, resp_sfreq=8.75714285714, resp_clip=(0, 200), minmax_smooth_window=70,
               bad_channels=msr.TEST_BAD_CHANNELS, offsets=msr.TEST_OFFSETS, csv_drop_first_rows={},
               drop_last_kept_epoch={}, shift_kept_indices={}, drop_first_trial=[])
    with contextlib.redirect_stdout(io.StringIO()):
        ds = rec.build_epochs(subjects, str(d / "raw/eeglab/{subject}/{subject}_EEG_preprocessed.mat"),
                              str(d / "raw/olfactometer/OlfactometerSubject{subject}.csv"), cfg, verbose=False)
    np.save(d / "epochs.npy", ds, allow_pickle=True)
    return str(d / "epochs.npy"), str(d / "info"), d


def run(mod, ep, info, out, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        mod.run(epochs_file=ep, info_dir=info, out_file=str(out), exclude_subjects=("30",), **kw)
    return np.load(out, allow_pickle=True).item()


def test_workers_equal_serial_and_no_window_leak(epochs_and_info):
    import generate_ersp as mod
    ep, info, d = epochs_and_info
    serial = run(mod, ep, info, d / "s.npy", phase_aligned=True, n_jobs=1)
    parallel = run(mod, ep, info, d / "p.npy", phase_aligned=True, n_jobs=2)
    assert same(serial, parallel)
    t2 = run(mod, ep, info, d / "t2.npy", phase_aligned=False, time_window_end=2.0, n_jobs=1)
    t_default = run(mod, ep, info, d / "t3.npy", phase_aligned=False, n_jobs=2)      # must NOT inherit the 2-s window
    assert len(t2["x"]) == 625 and len(t_default["x"]) == 1070
    assert list(serial["water"]) == list(parallel["water"])                            # same subject order
