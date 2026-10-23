"""Cluster-based permutation tests of the ERSPs (Section 1.5, Fig. S4): synthetic checks of the helpers and of the script end to end."""
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import yaml

from olfaction_eeg import cluster_perm as cp

sys.path.insert(0, os.path.dirname(__file__))
import make_synthetic_ersp as mse  # noqa: E402

SUBS = ["22", "23", "24", "26", "27", "29", "31", "32"]
ODORS = ["peppermint", "oregano", "citrus"]


def test_restrict_to_region_masks():
    freqs, bins = np.array([2.0, 4.0, 6.0, 8.0, 10.0]), np.arange(10) + 0.5
    fm, bm = cp.restrict_to_region(None, (4, 8), (0, 5), freqs, bins)
    assert list(fm) == [False, True, True, True, False] and bm.sum() == 5 and bins[bm].max() == 4.5


def test_coarsen_bins_averages_consecutive_bins():
    arr = np.arange(2 * 3 * 8, dtype=float).reshape(2, 3, 8)
    out = cp.coarsen_bins({"a": arr}, bin_factor=4)["a"]
    assert out.shape == (2, 3, 2) and np.allclose(out[..., 0], arr[..., :4].mean(axis=2))
    assert np.array_equal(cp.coarsen_bins({"a": arr}, bin_factor=1)["a"], arr)


def test_pooling_and_trial_range():
    ersp = mse.make_one("phase", SUBS, np.random.default_rng(0))
    st = cp.build_trial_stack_pooled_odors(ersp, ODORS, SUBS, trial_start=0, trial_end=3)
    assert set(st) == set(SUBS) and all(a.shape[0] <= 9 and a.shape[1:] == (24, 360) for a in st.values())


def _stacks(effect, seed=0):
    rng = np.random.default_rng(seed)
    odor = {s: rng.normal(size=(6, 10, 40)) + effect for s in SUBS}
    water = {s: rng.normal(size=(6, 10, 40)) for s in SUBS}
    return odor, water


def test_region_test_finds_an_injected_effect_and_not_noise():
    ones = np.ones(10, bool), np.ones(40, bool)
    odor, water = _stacks(effect=1.0)
    res = cp.cluster_permutation_test_2d_region(odor, water, *ones, n_perm=99, tail="positive")
    best = cp.best_cluster(res)
    assert best is not None and best["p_value"] <= 0.02 and best["n_pixels"] > 100
    # the opposite tail finds no suprathreshold pixels at all on a purely positive effect
    assert cp.cluster_permutation_test_2d_region(odor, water, *ones, n_perm=99, tail="negative")["n_clusters"] == 0
    odor0, water0 = _stacks(effect=0.0, seed=4)
    res0 = cp.cluster_permutation_test_2d_region(odor0, water0, *ones, n_perm=99, tail="positive")
    b0 = cp.best_cluster(res0)
    assert b0 is None or b0["p_value"] > 0.02


def test_p_values_are_k_plus_one_over_nperm_plus_one():
    odor, water = _stacks(effect=0.3, seed=2)
    res = cp.cluster_permutation_test_2d_region(odor, water, np.ones(10, bool), np.ones(40, bool), n_perm=49, tail="positive")
    for c in res["clusters"]:
        assert abs(c["p_value"] * 50 - round(c["p_value"] * 50)) < 1e-9


def test_script_end_to_end(tmp_path):
    rng = np.random.default_rng(1)
    for kind, name in (("phase", "phase.npy"), ("long", "long.npy"), ("short", "short.npy")):
        np.save(tmp_path / name, mse.make_one(kind, SUBS, rng, missing=(), short_subject="__none__"))
    params = {"cluster_perm": {"run": True, "n_perm": 20, "trial_start": 0, "trial_end": 3, "theta": [4, 8], "alpha": [8, 12],
                               "bin_factor": 1,
                               "sensitivity": [{"name": "theta_3_9_inhale", "freq_range": [3, 9], "phase_range": [0, 180], "tail": "positive"}]},
              "odors": ODORS, "normosmia": SUBS, "paper": {}, "checks_enabled": False}
    yaml.safe_dump(params, open(tmp_path / "params.yaml", "w"))
    root = os.path.dirname(os.path.dirname(__file__))
    r = subprocess.run([sys.executable, os.path.join(root, "scripts", "run_cluster_perm.py"), "--phase", str(tmp_path / "phase.npy"),
                        "--time-short", str(tmp_path / "short.npy"), "--time-long", str(tmp_path / "long.npy"),
                        "--params", str(tmp_path / "params.yaml"), "--out-dir", str(tmp_path / "out"), "--fig-dir", str(tmp_path / "fig")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    summ = pd.read_csv(tmp_path / "out" / "cluster_perm_summary.csv")
    assert len(summ) == 6 and set(summ["band"]) == {"theta", "alpha"}
    assert (tmp_path / "fig" / "figS4_cluster_perm.pdf").exists() and (tmp_path / "out" / "cluster_perm_sensitivity.csv").exists()
