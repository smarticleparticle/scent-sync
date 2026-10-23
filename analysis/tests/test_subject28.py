"""Subject-28 sensitivity (SI 1.4, Table S7): synthetic checks of the helpers and of the script end to end."""
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

from olfaction_eeg import breath, subject28 as s28

sys.path.insert(0, os.path.dirname(__file__))
import make_synthetic_ersp as mse  # noqa: E402

SF_RESP, SF_EEG = 8.75714285714, 250


def fake_epochs(subjects=("22", "23", "28"), periods=None, n_trials=9, flat_every=None, seed=0):
    rng = np.random.default_rng(seed)
    n_eeg, n_resp = int(16 * SF_EEG) + 1, int(16 * SF_RESP)
    t = np.arange(n_resp) / SF_RESP
    periods = periods or {s: 3.6 + 0.4 * i for i, s in enumerate(subjects)}
    out = {"EEG": {}, "respiration": {}}
    for c in breath.CONDITIONS:
        out["EEG"][c], out["respiration"][c] = {}, {}
        for s in subjects:
            resp = []
            for k in range(n_trials):
                r = -np.cos(2 * np.pi * (t - 1.0) / periods[s]) + 0.02 * rng.normal(size=n_resp)
                if flat_every and s == "28" and k % flat_every == 0:
                    r = -np.cos(2 * np.pi * (t - 1.0) / 12.0) + 0.02 * rng.normal(size=n_resp)   # very slow breathing -> 1-2 cycles
                resp.append(r)
            out["respiration"][c][s] = np.array(resp)
            out["EEG"][c][s] = np.zeros((n_trials, 3, n_eeg))
    return out


def test_diagnose_subject_counts_add_up_and_cycle_breakdown():
    ep = fake_epochs(flat_every=3)
    skip, det = s28.diagnose_subject(ep, "28")
    assert sum(skip.values()) == len(det) == 4 * 9
    cyc = s28.rejected_cycle_breakdown(det)
    assert cyc["n_rejected"] == skip["insufficient_oscillations"] > 0
    assert cyc["would_pass_at_2_cycles"] + cyc["still_fail_at_2_cycles"] == cyc["n_rejected"]
    assert cyc["n_with_0_or_1_cycles"] == cyc["still_fail_at_2_cycles"]


def test_retention_and_breathing_rate_table():
    subs = ("22", "23", "24", "26", "27", "28", "29", "31")
    ep = fake_epochs(subjects=subs, periods={s: 3.2 + 0.5 * i for i, s in enumerate(subs)}, flat_every=2)
    bd = breath.extract_first_breath_data(ep, {s: [] for s in subs}, store_eeg=False)
    df_breath = breath.breath_table(bd, list(subs))
    ret = s28.trial_retention(ep, bd, list(subs))
    stats = s28.subject_breath_stats(df_breath)
    merged = stats.merge(ret, on="subject")
    assert len(merged) == len(subs)
    assert np.allclose(merged["breaths_per_16s"], 16.0 / merged["mean_breath_sec"])
    row = merged[merged.subject == "28"].iloc[0]
    assert row["pct_kept"] == pytest.approx(100 * row["total_kept"] / row["total_attempted"]) and row["pct_kept"] < 100
    corr = s28.retention_vs_rate(merged, "28")
    assert corr["n_all"] == 8 and corr["n_without"] == 7 and -1 <= corr["r_all"] <= 1


def test_auc_without_subject_shapes_and_delta():
    ersp = mse.make_one("phase", mse.NORMOSMIA, np.random.default_rng(2))
    t7 = s28.auc_without_subject(ersp, mse.NORMOSMIA, ["peppermint", "oregano", "citrus"])
    assert list(t7["odor"]) == ["peppermint", "oregano", "citrus"]
    assert np.allclose(t7["delta"], t7["auc_without_subj28"] - t7["auc_with_subj28"])
    # peppermint has no subject-28 data in the synthetic set, so dropping 28 cannot change anything
    assert t7.loc[0, "delta"] == 0.0


def test_script_end_to_end(tmp_path):
    subs = list(mse.NORMOSMIA)
    ep = fake_epochs(subjects=subs, periods={s: 3.2 + 0.15 * i for i, s in enumerate(subs)}, n_trials=6, flat_every=2)
    np.save(tmp_path / "epochs.npy", ep)
    info = tmp_path / "info"
    info.mkdir()
    for s in subs:
        (info / f"subject{s}_info.fif").write_bytes(b"")           # only the file names are used
    np.save(tmp_path / "phase.npy", mse.make_one("phase", subs, np.random.default_rng(0)))
    params = {"subject28": {"subject": "28", "C": 1.0, "trial_start": 0, "trial_end": 3},
              "odors": ["peppermint", "oregano", "citrus"], "normosmia": subs, "paper": {}, "checks_enabled": False}
    yaml.safe_dump(params, open(tmp_path / "params.yaml", "w"))
    root = os.path.dirname(os.path.dirname(__file__))
    r = subprocess.run([sys.executable, os.path.join(root, "scripts", "run_subject28.py"), "--epochs", str(tmp_path / "epochs.npy"),
                        "--info-dir", str(info), "--phase", str(tmp_path / "phase.npy"), "--params", str(tmp_path / "params.yaml"),
                        "--out-dir", str(tmp_path / "out")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    out = tmp_path / "out"
    for f in ["table_S7.csv", "retention_by_subject.csv", "retention_spearman.csv", "subject28_trial_diagnosis.csv",
              "subject28_skip_reasons.csv", "subject28_checks.csv"]:
        assert (out / f).exists(), f
    assert len(pd.read_csv(out / "table_S7.csv")) == 3
