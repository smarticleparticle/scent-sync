"""audit_deposit.py flags clock times, date objects and MNE info fields, and stays quiet on clean files."""
import datetime as dt
import os
import subprocess
import sys

import mne
import numpy as np

ROOT = os.path.dirname(os.path.dirname(__file__))


def audit(root, *args):
    return subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "audit_deposit.py"), "--root", str(root), *args], capture_output=True, text=True)


def test_clean_deposit_is_quiet(tmp_path):
    (tmp_path / "info").mkdir()
    info = mne.create_info(["E1", "E2"], 250.0, "eeg")
    mne.io.write_info(str(tmp_path / "info" / "subject22_info.fif"), info)
    np.save(tmp_path / "clean.npy", {"respiration": {"water": {"22": np.zeros((3, 10))}}, "timestamps": {"water": {"22": np.array([1.5, 20.2, 40.1])}}}, allow_pickle=True)
    r = audit(tmp_path)
    assert r.returncode == 0
    assert "UNIX CLOCK TIMES" not in r.stdout and "[CHECK] subject22_info.fif" not in r.stdout
    assert "none of meas_date" in r.stdout                              # the info file carries none of the identifying fields


def test_flags_clock_times_dates_and_info_fields(tmp_path):
    (tmp_path / "info").mkdir()
    info = mne.create_info(["E1"], 250.0, "eeg")
    with info._unlock():
        info["meas_date"] = dt.datetime(2025, 3, 4, 12, 30, tzinfo=dt.timezone.utc)
        info["experimenter"] = "Someone"
        info["meas_id"] = {"version": 1, "machid": np.array([1, 2], dtype=np.int32), "secs": 1741091400, "usecs": 0}
    mne.io.write_info(str(tmp_path / "info" / "subject22_info.fif"), info)
    np.save(tmp_path / "t.npy", {"timestamps": {"water": np.array([1741091400.0, 1741091420.0])}, "when": dt.datetime(2025, 3, 4), "note": "John's session"}, allow_pickle=True)
    r = audit(tmp_path)
    out = r.stdout
    assert "UNIX CLOCK TIMES" in out and "2025-03-04" in out
    assert "date/time object" in out and "[text]" in out and "John's session" in out
    assert "meas_date" in out and "experimenter" in out and "meas_id=2025-03-04" in out
    assert "item(s) to look at" in out
