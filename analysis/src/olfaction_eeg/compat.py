"""Compatibility helpers for loading pickled MNE objects written by an OLDER MNE.

The continuous-recordings files (full_respiration_EEG_*.npy) contain pickled ``mne.io.RawArray`` objects.  MNE moved that class
(``mne.io.array.array`` -> ``mne.io.array._array``), so a file written by an old MNE fails to load in a newer one with
``ModuleNotFoundError: No module named 'mne.io.array.array'``.  The analysis notebook worked around it with the same one-line
alias that ``install_mne_pickle_shim`` installs.
"""
import importlib
import sys

import numpy as np


def install_mne_pickle_shim():
    """Alias the old module path to wherever RawArray lives now (no-op if the old path still exists)."""
    try:
        importlib.import_module("mne.io.array.array")
        return False
    except ImportError:
        import mne
        sys.modules["mne.io.array.array"] = sys.modules[mne.io.RawArray.__module__]
        return True


def load_pickled_dict(path):
    """np.load(path, allow_pickle=True).item() with the MNE pickle alias installed first."""
    install_mne_pickle_shim()
    return np.load(path, allow_pickle=True).item()
