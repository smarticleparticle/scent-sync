"""Files pickled by an OLD MNE (RawArray under mne.io.array.array) must load in a new MNE through the compat helper."""
import os
import subprocess
import sys

import pytest

pytest.importorskip("mne")
SRC = os.path.join(os.path.dirname(__file__), "..", "src")


def test_old_mne_pickle_loads_with_shim(tmp_path):
    f = str(tmp_path / "old.npy")
    writer = f'''
import sys, numpy as np, mne
mne.set_log_level("ERROR")
sys.modules["mne.io.array.array"] = sys.modules[mne.io.RawArray.__module__]
mne.io.RawArray.__module__ = "mne.io.array.array"
raw = mne.io.RawArray(np.zeros((2, 50)), mne.create_info(["E1", "E2"], 250.0, "eeg"), verbose=False)
np.save({f!r}, dict(EEG=dict(s=raw)), allow_pickle=True)
'''
    subprocess.run([sys.executable, "-W", "ignore", "-c", writer], check=True)
    plain = subprocess.run([sys.executable, "-W", "ignore", "-c", f"import numpy as np; np.load({f!r}, allow_pickle=True)"],
                           capture_output=True, text=True)
    if plain.returncode == 0:
        pytest.skip("this MNE still provides mne.io.array.array, nothing to shim")
    assert "mne.io.array.array" in plain.stderr
    fixed = subprocess.run([sys.executable, "-W", "ignore", "-c",
                            f"import sys; sys.path.insert(0, {SRC!r}); from olfaction_eeg import compat; "
                            f"d = compat.load_pickled_dict({f!r}); print(d['EEG']['s'].get_data().shape)"],
                           capture_output=True, text=True)
    assert fixed.returncode == 0 and "(2, 50)" in fixed.stdout, fixed.stderr
