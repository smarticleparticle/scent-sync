"""FOOOF gating: unit tests on tiny synthetic data (equivalence with the notebook cells was checked separately)."""
import os
import sys

import numpy as np
import pytest

pytest.importorskip("mne")
pytest.importorskip("fooof")
sys.path.insert(0, os.path.dirname(__file__))
import make_synthetic_pac as msp  # noqa: E402

from olfaction_eeg import fooof_gating as fg  # noqa: E402

BANDS = {"theta": (4, 8), "alpha": (8, 12), "beta": (12, 30), "low_gamma": (30, 40)}
SUBJECTS = ["22", "23", "24"]


@pytest.fixture(scope="module")
def data():
    recs = msp.make_recordings(subjects=SUBJECTS, n_ch=5, duration_s=120, seed=2)
    pac = msp.make_pac_results(SUBJECTS, msp.make_info(5, seed=99), n_surrogates=30, seed=1)
    cache = fg.compute_psd_cache(recs, pac, SUBJECTS)
    return recs, pac, cache


def test_cache_structure_and_size(data):
    _, _, cache = data
    assert set(cache["psd"]) == set(SUBJECTS) and set(cache["inhale_exhale"]) <= set(SUBJECTS)
    freqs, psd = cache["psd"]["22"]["E1"]
    assert freqs.min() >= 1 and freqs.max() <= 45 and freqs.shape == psd.shape


def test_alpha_peak_detected_not_theta(data):
    _, _, cache = data
    prev = fg.peak_prevalence(fg.build_peak_gate(cache, BANDS, fg.DEFAULT_SETTINGS), BANDS)
    assert prev["alpha"]["pct"] > 90 and prev["theta"]["pct"] < 20
    assert prev["alpha"]["n"] == 15                                # 3 subjects x 5 channels


def test_gating_respects_min_channels(data):
    _, pac, cache = data
    gate = fg.build_peak_gate(cache, BANDS, fg.DEFAULT_SETTINGS)
    assert len(fg.subject_level_zscore(pac, "alpha", gate, gated=True, min_channels=3)) == 3
    assert fg.subject_level_zscore(pac, "alpha", gate, gated=True, min_channels=6) == {}      # only 5 channels exist
    assert len(fg.subject_level_zscore(pac, "alpha", gated=False, min_channels=3)) == 3


def test_stricter_threshold_never_adds_peaks(data):
    _, _, cache = data
    n_peaks = []
    for thr in (1.5, 2.0, 3.0):
        g = fg.build_peak_gate(cache, BANDS, {**fg.DEFAULT_SETTINGS, "peak_threshold": thr})
        n_peaks.append(sum(v[b] for s in g.values() for v in s.values() for b in BANDS))
    assert n_peaks[0] >= n_peaks[1] >= n_peaks[2]


def test_gated_stats_fdr_not_below_raw(data):
    _, pac, cache = data
    gate = fg.build_peak_gate(cache, BANDS, fg.DEFAULT_SETTINGS)
    out = fg.run_gated_stats(pac, gate, BANDS, min_channels=3, n_perm=200)
    for b in BANDS:
        p, q = out[b].get("p", np.nan), out[b]["p_fdr"]
        assert np.isnan(p) or q >= p - 1e-12


def test_aperiodic_inhale_exhale_runs(data):
    _, _, cache = data
    res = fg.aperiodic_inhale_exhale(cache, SUBJECTS, n_perm=200)
    assert res["n_subjects"] == 3 and 0 <= res["exponent_p"] <= 1
