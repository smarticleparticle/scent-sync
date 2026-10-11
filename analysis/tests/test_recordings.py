"""Unit tests for the raw -> epochs stage.  Needs mne + h5py (skipped otherwise).
    pytest -q tests/test_recordings.py
"""
import os
import sys

import numpy as np
import pytest

pytest.importorskip("mne")
pytest.importorskip("h5py")
sys.path.insert(0, os.path.dirname(__file__))
import make_synthetic_raw as msr  # noqa: E402

from olfaction_eeg import recordings as rec  # noqa: E402

CFG = dict(tmin=-1.0, tmax=15.0, resp_sfreq=8.75714285714, resp_clip=(0, 200), minmax_smooth_window=70,
           bad_channels=msr.TEST_BAD_CHANNELS, offsets=msr.TEST_OFFSETS, csv_drop_first_rows={"27": 73},
           drop_last_kept_epoch={"24": ["peppermint"], "28": ["oregano"], "35": ["citrus"]},
           shift_kept_indices={"27": {"peppermint": -1}}, drop_first_trial=["27"])


@pytest.fixture(scope="module")
def raw_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("raw")
    msr.main(str(d))
    return str(d)


@pytest.fixture(scope="module")
def epochs(raw_dir):
    return rec.build_epochs(msr.SUBJECTS, raw_dir + "/eeglab/{subject}/{subject}_EEG_preprocessed.mat",
                            raw_dir + "/olfactometer/OlfactometerSubject{subject}.csv", CFG, verbose=False)


def test_remove_peaks_interpolates_spikes_and_edges():
    x = np.array([5.0, 10.0, 1000.0, 20.0, -3.0, 30.0, 999.0])
    out = rec.remove_peaks(x, 0, 200)
    assert out[2] == pytest.approx(15.0)          # between 10 and 20
    assert out[4] == pytest.approx(25.0)          # between 20 and 30
    assert out[6] == 30.0                         # trailing spike -> last valid value
    assert rec.remove_peaks(np.array([1000.0, 4.0, 6.0]))[0] == 4.0   # leading spike -> first valid value


def test_remove_peaks_treats_nan_as_outlier():
    out = rec.remove_peaks(np.array([1.0, np.nan, 3.0]))
    assert out[1] == pytest.approx(2.0)


def test_scalar_helper_handles_1x1_arrays():
    assert rec._scalar(np.array([[250.0]])) == 250.0   # float(np.array([[...]])) fails on numpy >= 2


def test_loader(raw_dir):
    e = rec.load_eeglab_mat(raw_dir + "/eeglab/22/22_EEG_preprocessed.mat")
    assert e["data"].shape[0] == msr.N_CH and e["srate"] == msr.SRATE and e["ch_names"][0] == "E1"
    assert set(np.unique(e["events"][:, 2])) <= {3, 4, 5, 6, 7}        # unknown DIN99 dropped
    assert e["pos"].max() < 0.2                                         # mm -> m


def test_csv_drop_first_rows(raw_dir):
    path = raw_dir + "/olfactometer/OlfactometerSubject27.csv"
    assert (rec.read_olfactometer_csv(path, "27", None) == 7777.0).sum() == 73     # junk rows present
    assert not (rec.read_olfactometer_csv(path, "27", {"27": 73}) == 7777.0).any()  # and removed


def test_epochs_internally_consistent(epochs):
    for cond in rec.CONDITION_ORDER:
        for subj in msr.SUBJECTS:
            n = {k: epochs[k][cond][subj].shape[0] for k in ("EEG", "respiration", "timestamps")}
            assert len(set(n.values())) == 1, (cond, subj, n)
            assert epochs["EEG"][cond][subj].shape[1] == msr.N_CH
            assert epochs["respiration"][cond][subj].shape[1] == int(round(16 * msr.RESP_SFREQ))


def test_corrections_applied(epochs):
    # synthetic recordings end 10 s after the last event => last epoch is cut; 4 trials/condition scheduled
    counts = {s: sum(epochs["EEG"][c][s].shape[0] for c in rec.CONDITION_ORDER) for s in msr.SUBJECTS}
    assert counts["27"] < counts["22"]      # subject 27 loses trial 0 of every condition


def test_no_nan_in_respiration_after_cleaning(epochs):
    # subject 23's humidity trace has missing samples; remove_peaks must interpolate them
    for cond in rec.CONDITION_ORDER:
        assert not np.isnan(epochs["respiration"][cond]["23"]).any()
