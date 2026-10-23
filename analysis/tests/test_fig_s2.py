"""SI Fig. S2 (FOOOF example panels) from a spectra cache: synthetic checks."""
import os
import subprocess
import sys

import numpy as np
import yaml

from olfaction_eeg import fooof_gating as fg
from olfaction_eeg.fig_s2 import find_example_channel, manual_examples, make_figS2

BANDS = {"theta": (4, 8), "alpha": (8, 12), "beta": (12, 30), "low_gamma": (30, 40)}
SETTINGS = {"peak_width_limits": [1, 12], "max_n_peaks": 6, "min_peak_height": 0, "aperiodic_mode": "fixed", "peak_threshold": 2.0}


def spectrum(rng, peaks=()):
    f = np.arange(1, 45.01, 0.25)
    logp = -11.0 - 1.0 * np.log10(f) * 1.0
    for c, h, w in peaks:
        logp = logp + h * np.exp(-0.5 * ((f - c) / w) ** 2)
    return f, 10 ** (logp + 0.01 * rng.normal(size=len(f)))


def make_cache():
    rng = np.random.default_rng(0)
    subs = ["22", "23", "29", "32", "33", "34"]
    chans = ["E1", "E3", "E5", "E14", "E58", "E70", "E99", "E111"] + [f"E{i}" for i in range(120, 126)]
    cache = {"psd": {}}
    for s in subs:
        cache["psd"][s] = {}
        for k, c in enumerate(chans):
            peaks = [(10.0, 0.8, 1.2)] if k % 2 == 0 else []
            if k % 3 == 0:
                peaks.append((20.0, 0.5, 2.0))
            cache["psd"][s][c] = spectrum(rng, peaks)
    return cache


def test_find_example_channel_returns_first_match_in_gate_order():
    gate = {"A": {"c1": {"alpha": False}, "c2": {"alpha": True}}, "B": {"c3": {"alpha": True}}}
    assert find_example_channel(gate, "alpha", True) == ("A", "c2")
    assert find_example_channel(gate, "alpha", False) == ("A", "c1")
    assert find_example_channel(gate, "theta", True) is None
    assert find_example_channel(gate, "alpha", True, max_search=1) is None          # search budget is respected


def test_figS2_builds_eight_panels():
    cache = make_cache()
    gate = fg.build_peak_gate(cache, BANDS, SETTINGS)
    fig, chosen = make_figS2(gate, cache, BANDS, settings=SETTINGS)
    assert len(fig.axes) == 8 and len(chosen) == 8
    for key, val in manual_examples.items():
        assert chosen[key] == val                                                   # hand-picked panels are used as given
    assert ("alpha", True) in chosen and ("beta", False) in chosen                  # the two searched panels were found


def test_figS2_script_end_to_end(tmp_path):
    cache = make_cache()
    np.save(tmp_path / "psd.npy", cache)
    params = {"fooof": {"settings": {k: v for k, v in SETTINGS.items() if k != "peak_threshold"}, "default_threshold": 2.0},
              "pac": {"freq_bands": {k: list(v) for k, v in BANDS.items()}}, "paper": {}, "checks_enabled": False}
    yaml.safe_dump(params, open(tmp_path / "params.yaml", "w"))
    root = os.path.dirname(os.path.dirname(__file__))
    r = subprocess.run([sys.executable, os.path.join(root, "scripts", "make_figS2.py"), "--psd-cache", str(tmp_path / "psd.npy"),
                        "--params", str(tmp_path / "params.yaml"), "--out-dir", str(tmp_path / "out"), "--fig-dir", str(tmp_path / "fig")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "fig" / "figS2_fooof_examples.pdf").exists() and (tmp_path / "out" / "figS2_examples.json").exists()
