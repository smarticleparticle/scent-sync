"""Unit tests for the PAC modules (tiny synthetic data; the heavy equivalence checks ran against the notebook)."""
import os
import sys

import numpy as np
import pytest

mne = pytest.importorskip("mne")
sys.path.insert(0, os.path.dirname(__file__))
import make_synthetic_pac as msp  # noqa: E402

from olfaction_eeg import pac, pac_figure, pac_stats  # noqa: E402


def test_modulation_index_uniform_vs_concentrated():
    phase = np.linspace(0, 2 * np.pi, 20000, endpoint=False)
    mi_flat, _ = pac.compute_modulation_index(phase, np.ones_like(phase))
    mi_peaked, mean_amp = pac.compute_modulation_index(phase, 1 + 0.9 * np.cos(phase - np.pi))
    assert mi_flat == pytest.approx(0, abs=1e-9)
    assert mi_peaked > 0.01
    pref = pac.compute_preferred_phase(mean_amp)                                      # angle in (-pi, pi]
    assert abs(np.angle(np.exp(1j * (pref - np.pi)))) < 0.2                          # amplitude peaks at pi (circular distance)


def test_modulation_index_ignores_nan_phase():
    phase = np.linspace(0, 2 * np.pi, 1000, endpoint=False)
    amp = 1 + 0.5 * np.cos(phase)
    phase_nan = phase.copy()
    phase_nan[::3] = np.nan
    mi_all, _ = pac.compute_modulation_index(phase, amp)
    mi_nan, _ = pac.compute_modulation_index(phase_nan, amp)
    assert abs(mi_all - mi_nan) < 5e-3


def test_surrogate_keeps_phase_nan_mask():
    np.random.seed(0)
    n = 5000
    phase = np.linspace(0, 2 * np.pi, n)
    phase[:100] = np.nan
    amp = np.random.rand(n)
    sur = pac.create_surrogate_amplitude_with_nan_mask(amp, phase, sfreq=100, min_shift_sec=5)
    assert np.array_equal(np.isnan(sur), np.isnan(phase))                  # surrogate NaNs exactly where phase is NaN
    assert np.allclose(np.sort(sur[~np.isnan(sur)]), np.sort(amp[~np.isnan(phase)]))   # a circular shift permutes values


def test_respiratory_phase_covers_cycle():
    sfreq = 250
    t = np.arange(0, 120, 1 / sfreq)
    resp = np.sin(2 * np.pi * t / 4.0) + 0.01 * np.random.default_rng(0).normal(size=t.size)
    phase = pac.compute_linearized_respiratory_phase_direct(resp, sfreq, resp_band=(0.1, 0.4))
    valid = phase[~np.isnan(phase)]
    assert valid.min() >= 0 and valid.max() <= 2 * np.pi
    assert len(valid) > 0.7 * len(phase)


def test_run_pac_is_deterministic_and_skips_bad_channels():
    recs = msp.make_recordings(subjects=["22"], n_ch=3, duration_s=200, seed=5, bad_channels={"22": ["E2"]})
    cfg = dict(sfreq_eeg=250, sfreq_resp=8.75714285714, resp_band=[0.1, 0.4], freq_bands={"alpha": [8, 12]},
               n_phase_bins=18, n_surrogates=8, min_shift_sec=30.0, min_cycle_duration=2.0, max_cycle_duration=8.0, seed=1)
    a = pac.run_pac(recs, ["22"], cfg, verbose=False)
    b = pac.run_pac(recs, ["22"], cfg, verbose=False)
    assert set(a["subject_level"]["22"]) == {"E1", "E3"}
    for ch in a["subject_level"]["22"]:
        ra, rb = a["subject_level"]["22"][ch]["alpha"], b["subject_level"]["22"][ch]["alpha"]
        assert np.array_equal(ra["MI_null_distribution"], rb["MI_null_distribution"])


def test_group_stats_fdr_never_below_raw_p():
    import pandas as pd
    rng = np.random.default_rng(0)
    rows = [{"subject_id": str(s), "band": b, "z": rng.normal(loc=mu)}
            for b, mu in [("theta", 0.0), ("alpha", 1.0), ("beta", 0.5), ("low_gamma", 1.5)] for s in range(12)]
    gs = pac_stats.compute_group_stats(pd.DataFrame(rows))
    assert (gs["p_fdr"] >= gs["p_for_fdr"] - 1e-12).all()
    assert gs.set_index("band").loc["low_gamma", "p_perm"] < gs.set_index("band").loc["theta", "p_perm"]


def test_cohens_d():
    assert pac_stats.cohens_d_onesample([1, 2, 3, 4]) == pytest.approx(2.5 / np.std([1, 2, 3, 4], ddof=1), abs=1e-9)
    assert np.isnan(pac_stats.cohens_d_onesample([1.0]))


def test_plot_info_scaling_and_rotation(tmp_path):
    info = msp.make_info(n_ch=10, seed=3)
    f = tmp_path / "subject22_info.fif"
    mne.io.write_info(str(f), info)
    out = pac_figure.load_plot_info(str(f))
    ch = out.get_montage().get_positions()["ch_pos"]
    radii = np.linalg.norm(np.array(list(ch.values())), axis=1)
    assert radii.mean() == pytest.approx(0.095, rel=0.01)                  # rescaled to a ~9.5 cm head
    # a +x point maps to +y after the 90 degree rotation about z
    orig = info.get_montage().get_positions()
    pos0 = orig["lpa"]; pos1 = out.get_montage().get_positions()["lpa"]
    assert pos1[1] == pytest.approx(pos0[0] * 0.095 / 0.00991496, rel=1e-6)
