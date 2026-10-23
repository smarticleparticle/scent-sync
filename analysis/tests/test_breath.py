"""Breath-duration analysis: first-breath detection and the repeated-measures ANOVA."""
import numpy as np
import pandas as pd
import pytest

from olfaction_eeg import breath


def fake_epochs(period=4.0, n_trials=3, n_subjects=2):
    sf_eeg, sf_resp = 250, 8.75714285714
    n_eeg, n_resp = int(16 * sf_eeg) + 1, int(16 * sf_resp)
    t = np.arange(n_resp) / sf_resp
    out = {"EEG": {}, "respiration": {}}
    for c in breath.CONDITIONS:
        out["EEG"][c], out["respiration"][c] = {}, {}
        for s in range(n_subjects):
            # trough at odor onset (t = 1 s), peak one half-period later
            resp = -np.cos(2 * np.pi * (t - 1.0) / period) + 0.01 * np.random.default_rng(s).normal(size=n_resp)
            out["respiration"][c][str(s)] = np.tile(resp, (n_trials, 1))
            out["EEG"][c][str(s)] = np.zeros((n_trials, 3, n_eeg))
    return out


def test_first_breath_durations_follow_the_respiratory_period():
    ds = fake_epochs(period=4.0)
    bd = breath.extract_first_breath_data(ds, {"0": [], "1": []}, store_eeg=False)
    df = breath.breath_table(bd, ["0", "1"])
    assert len(df) == 4 * 2 * 3                                   # every trial passes quality control
    assert df["inhale_duration"].between(1.8, 2.2).all() and df["exhale_duration"].between(1.8, 2.2).all()
    assert np.allclose(df["breath_duration"], df["inhale_duration"] + df["exhale_duration"])
    assert bd["water"]["0"][0]["inhale"]["eeg_data"] is None       # store_eeg=False skips the slices


def test_subject_filter_and_flat_trials_are_dropped():
    ds = fake_epochs()
    ds["respiration"]["water"]["0"][1] = 0.0                       # flat trace -> bad respiration -> trial skipped
    bd = breath.extract_first_breath_data(ds, {"0": [], "1": []}, store_eeg=False)
    assert 1 not in bd["water"]["0"] and 0 in bd["water"]["0"]
    assert set(breath.breath_table(bd, ["1"])["subject"]) == {"1"}


def test_rm_anova_known_example():
    # textbook example: 4 subjects x 3 conditions with a clear condition effect and no error variance beyond subjects
    X = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0], [4.0, 5.0, 6.0]])
    df = pd.DataFrame([{"subject": str(i), "condition": f"c{j}", "x": X[i, j]} for i in range(4) for j in range(3)])
    r = breath.rm_anova_oneway(df, "x")
    assert (r["df1"], r["df2"]) == (2, 6) and r["SS_effect"] == pytest.approx(8.0) and r["SS_error"] == pytest.approx(0.0, abs=1e-12)


def test_rm_anova_matches_pingouin():
    pg = pytest.importorskip("pingouin")
    rng = np.random.default_rng(3)
    X = rng.normal(2, 0.3, size=(15, 1)) + rng.normal(0, 0.2, size=(15, 4)) + np.array([0, 0.05, 0.1, 0.0])
    df = pd.DataFrame([{"subject": str(i), "condition": f"c{j}", "x": X[i, j]} for i in range(15) for j in range(4)])
    a = pg.rm_anova(data=df, dv="x", within="condition", subject="subject", detailed=True)
    m = breath.rm_anova_oneway(df, "x")
    assert m["F"] == pytest.approx(a.loc[0, "F"]) and m["p_unc"] == pytest.approx(a.loc[0, "p_unc"]) and m["ng2"] == pytest.approx(a.loc[0, "ng2"])
