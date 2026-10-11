"""Fast unit tests (no real data needed):  pytest -q tests/test_classify.py"""
import numpy as np
import pandas as pd
import pytest

from olfaction_eeg import classify as clf


def make_df(seed=0, n_sub=10, n_feat=6):
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_sub):
        base = rng.normal(size=n_feat)
        for lab in (0, 1):
            f = base + rng.normal(size=n_feat)
            f[:2] += 0.8 * lab
            rows.append({**{f"f{i}": f[i] for i in range(n_feat)}, "subject": str(s), "label": lab})
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def res():
    return clf.nested_loso_selk_logreg(make_df(), verbose=False)


def test_output_contract(res):
    for k in ["auc", "acc", "y_true", "y_proba", "y_pred", "subject",
              "selected_features_per_fold", "selected_k_per_fold", "selection_frequency"]:
        assert k in res
    assert len(res["subject"]) == len(res["y_true"]) == 20
    assert 0 <= res["auc"] <= 1


def test_every_subject_has_one_water_one_odor(res):
    assert len(clf.get_subject_probs(res)) == 10


def test_deterministic(res):
    again = clf.nested_loso_selk_logreg(make_df(), verbose=False)
    assert np.array_equal(res["y_proba"], again["y_proba"])


def test_bootstrap_ci_brackets_auc_and_is_seeded(res):
    auc, lo, hi, boots = clf.bootstrap_auc_ci(res, n_boot=200, seed=1)
    assert lo <= auc <= hi or np.isclose(lo, auc) or np.isclose(hi, auc)
    assert np.array_equal(boots, clf.bootstrap_auc_ci(res, n_boot=200, seed=1)[3])


def test_permutation_pvalue_valid(res):
    other = clf.nested_loso_selk_logreg(make_df(seed=5), verbose=False)
    p = clf.paired_permutation_auc_delta(res, other, n_perm=200)["p_value"]
    assert 0 < p <= 1


def test_pooled_matches_single_odor_when_one_odor(res):
    other = clf.nested_loso_selk_logreg(make_df(seed=5), verbose=False)
    single = clf.paired_permutation_auc_delta(res, other, n_perm=300, seed=3)
    pooled = clf.pooled_permutation_auc_delta({"x": {"a": res, "b": other}}, "a", "b", n_perm=300, seed=3)
    assert np.isclose(single["observed_delta"], pooled["observed_delta"])


# ---------------------------------------------------------------------------
# Table S4 test, FDR, selection-frequency helpers
# ---------------------------------------------------------------------------
def test_fdr_bh_known_values():
    # the raw habituation p-values of the paper (notebook cell 77) -> reported 0.022 / 0.040 / 0.022 / <0.001
    adj = clf.fdr_bh([0.0132, 0.0398, 0.0162, 0.0002])
    assert np.allclose(adj, [0.0216, 0.0398, 0.0216, 0.0008])
    assert np.all(adj >= np.array([0.0132, 0.0398, 0.0162, 0.0002]))        # never smaller than the raw p


def test_fdr_bh_matches_statsmodels():
    sm = pytest.importorskip("statsmodels.stats.multitest")
    rng = np.random.default_rng(0)
    for _ in range(200):
        p = np.round(rng.random(rng.integers(1, 10)), 2)
        assert np.allclose(clf.fdr_bh(p), sm.multipletests(p, method="fdr_bh")[1], atol=1e-12)


def test_gap_permutation_test(res):
    other = clf.nested_loso_selk_logreg(make_df(seed=5), verbose=False)
    ga, gb = clf.get_decision_gaps(res), clf.get_decision_gaps(other)
    common = sorted(set(ga) & set(gb))
    diff, p, lo, hi, outside = clf.paired_gap_permutation_test([ga[s] for s in common], [gb[s] for s in common],
                                                               n_permutations=500, seed=0)
    assert 0 <= p <= 1 and lo < hi
    assert outside == (diff < lo or diff > hi)
    again = clf.paired_gap_permutation_test([ga[s] for s in common], [gb[s] for s in common], n_permutations=500, seed=0)
    assert again[1] == p                                                    # seeded => reproducible
    flipped = clf.paired_gap_permutation_test([gb[s] for s in common], [ga[s] for s in common], n_permutations=500, seed=0)
    assert np.isclose(flipped[0], -diff)                                    # antisymmetric observed difference


def _fake_selection(**fracs):
    return {"selection_frequency": pd.DataFrame({"feature": list(fracs), "selected_frac": list(fracs.values())})}


def test_mean_selection_frequency_two_conventions():
    # per-odor selection of theta 0-90 deg: 100%, 94%, never (absent) -- Fig. 5B vs Fig. 4A convention
    odors = {"a": _fake_selection(theta_p0_90=1.00), "b": _fake_selection(theta_p0_90=0.94), "c": _fake_selection(alpha_p180_270=1.0)}
    present_only = clf.mean_selection_frequency(odors, count_unselected_as_zero=False)
    with_zeros = clf.mean_selection_frequency(odors, count_unselected_as_zero=True)
    assert present_only["theta_p0_90"] == pytest.approx(0.97)
    assert with_zeros["theta_p0_90"] == pytest.approx(0.6467, abs=1e-3)
    assert present_only["alpha_p180_270"] == with_zeros["alpha_p180_270"] * 3 == 1.0   # selected in 1 of 3 odors only


def test_selection_matrix_layout():
    res = _fake_selection(theta_p0_90=1.0, alpha_p270_360=0.5, beta_p90_180=0.25)
    m = clf.selection_matrix(res, ["p0_90", "p90_180", "p180_270", "p270_360"])      # rows: beta, alpha, theta
    assert m.shape == (3, 4)
    assert m[2, 0] == 100 and m[1, 3] == 50 and m[0, 1] == 25 and m.sum() == 175
