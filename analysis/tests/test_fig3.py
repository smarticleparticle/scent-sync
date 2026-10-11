"""Fig. 3 (phase-locked ERSP figure) and the Fig. 1 example respiration trace: synthetic checks."""
import os
import subprocess
import sys

import numpy as np
import pytest
import yaml

from olfaction_eeg import breath
from olfaction_eeg.fig3 import create_main_figure_rearranged, get_averaged_respiration_all_odors

SF_RESP, SF_EEG = 8.75714285714, 250
SUBS = ["33", "40", "22"]
ODORS = ["peppermint", "oregano", "citrus"]


def ersp_dict(kind, subjects, rng, n_trials=4):
    """Same grid as the real files: 100 log-spaced frequencies; 360 phase bins or a 250-Hz time axis (-0.5 to 3.78 s)."""
    x = (np.arange(360) + 0.5) if kind == "phase" else np.linspace(-0.5, 3.78, 1070)
    out = {c: {s: {t: rng.normal(scale=0.3, size=(100, len(x))) for t in range(n_trials)} for s in subjects}
           for c in ["water"] + ODORS}
    out["freqs"], out["x"] = np.geomspace(2, 40, 100), x
    return out


def epochs(subjects, rng, n_trials=6):
    n_eeg, n_resp = int(16 * SF_EEG) + 1, int(16 * SF_RESP)
    t = np.arange(n_resp) / SF_RESP
    ep = {"EEG": {}, "respiration": {}}
    for c in breath.CONDITIONS:
        ep["EEG"][c], ep["respiration"][c] = {}, {}
        for i, s in enumerate(subjects):
            period = 3.4 + 0.35 * i
            ep["respiration"][c][s] = np.array([-np.cos(2 * np.pi * (t - 1.0) / period) + 0.02 * rng.normal(size=n_resp)
                                                for _ in range(n_trials)])
            ep["EEG"][c][s] = np.zeros((n_trials, 2, n_eeg))
    return ep


def test_averaged_respiration_has_unit_variance_and_matching_phase_axis():
    rng = np.random.default_rng(0)
    bd = breath.extract_first_breath_data(epochs(SUBS, rng), {s: [] for s in SUBS}, store_eeg=False)
    sig, time, phase, inh, exh = get_averaged_respiration_all_odors(bd, "33", ODORS, 3)
    assert len(sig) == len(time) == len(phase) and abs(sig.std() - 1) < 1e-3 and abs(sig.mean()) < 1e-6
    assert phase[0] == 0 and phase.max() < 360 and inh > 0 and exh > 0
    assert get_averaged_respiration_all_odors(bd, "99", ODORS, 3)[0] is None


def test_fig3_builds_and_reports_its_numbers(tmp_path):
    rng = np.random.default_rng(1)
    phase, tl = ersp_dict("phase", SUBS, rng), ersp_dict("long", SUBS, rng)
    bd = breath.extract_first_breath_data(epochs(SUBS, rng), {s: [] for s in SUBS}, store_eeg=False)
    fig = create_main_figure_rearranged(phase, tl, bd, subjects_individual=["33", "40"], subjects_grand=SUBS, odor_list=ODORS,
                                        n_trials=3, save_path=str(tmp_path / "fig3.pdf"))
    info = fig.fig3_info
    assert (tmp_path / "fig3.pdf").exists()
    assert info["n_grand_phase"] == info["n_grand_time"] == info["n_resp_traces"] == 3
    assert info["subjects_individual"] == ["33", "40"] and info["vmax_subj"] > 0 and info["vmax_grand"] > 0


def test_fig3_script_end_to_end(tmp_path):
    rng = np.random.default_rng(2)
    np.save(tmp_path / "epochs.npy", epochs(SUBS, rng))
    np.save(tmp_path / "phase.npy", ersp_dict("phase", SUBS, rng))
    np.save(tmp_path / "long.npy", ersp_dict("long", SUBS, rng))
    info = tmp_path / "info"
    info.mkdir()
    for s in SUBS:
        (info / f"subject{s}_info.fif").write_bytes(b"")
    yaml.safe_dump({"fig3": {"subjects_individual": ["33", "40"], "n_trials": 3}, "odors": ODORS, "normosmia": SUBS,
                    "paper": {}, "checks_enabled": False}, open(tmp_path / "params.yaml", "w"))
    root = os.path.dirname(os.path.dirname(__file__))
    r = subprocess.run([sys.executable, os.path.join(root, "scripts", "make_fig3.py"), "--epochs", str(tmp_path / "epochs.npy"),
                        "--info-dir", str(info), "--phase", str(tmp_path / "phase.npy"), "--time-long", str(tmp_path / "long.npy"),
                        "--params", str(tmp_path / "params.yaml"), "--out-dir", str(tmp_path / "out"), "--fig-dir", str(tmp_path / "fig")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "fig" / "fig3_ersps.pdf").exists() and (tmp_path / "fig" / "fig3_ersps.png").exists()
    assert (tmp_path / "out" / "fig3_info.json").exists()


def test_fig1_trace_plot_and_script(tmp_path):
    from olfaction_eeg import fig1_resp
    t = np.arange(int(8 * SF_RESP)) / SF_RESP
    trial = 70 + 3 * np.sin(2 * np.pi * (t - 1) / 4.0)                 # relative humidity (%), 4-s breaths
    fig = fig1_resp.plot_example_trial(trial, trial_idx=0, save_path=str(tmp_path / "x.png"))
    assert (tmp_path / "x.png").exists() and fig.axes[0].get_ylabel() == "Relative Humidity (%)"
    np.save(tmp_path / "trial.npy", {"subject": "22", "trials": np.array([trial, trial + 1]), "resp_sfreq": SF_RESP, "pre_ttl": 4, "post_ttl": 4})
    root = os.path.dirname(os.path.dirname(__file__))
    r = subprocess.run([sys.executable, os.path.join(root, "scripts", "make_fig1_resp.py"), "plot", "--trial-file", str(tmp_path / "trial.npy"),
                        "--out-prefix", str(tmp_path / "o" / "fig1"), "--trial-idx", "1"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "o" / "fig1.pdf").exists()
