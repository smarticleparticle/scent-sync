"""Figure builders run and respond to their options (fast: uses fabricated LOSO-result objects)."""
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from olfaction_eeg import figures

BANDS = ["theta", "alpha", "beta"]
PHASE = ["p0_90", "p90_180", "p180_270", "p270_360"]
TIME = ["t0_0.945", "t0.945_1.89", "t1.89_2.835", "t2.835_3.78"]


def fake_result(rng, sel, n_subj=17):
    y_true = np.tile([0, 1], n_subj)
    y_proba = np.clip(0.4 + 0.2 * y_true + rng.normal(scale=0.2, size=2 * n_subj), 0, 1)
    df = pd.DataFrame({"feature": list(sel), "selected_frac": list(sel.values())})
    return {"y_true": y_true, "y_proba": y_proba, "auc": roc_auc_score(y_true, y_proba), "selection_frequency": df}


def all_features(bins, value):
    return {f"{b}_{p}": value for b in BANDS for p in bins}


@pytest.fixture(scope="module")
def per_odor():
    rng = np.random.default_rng(0)
    return {o: {"phase": fake_result(rng, all_features(PHASE, 0.5)), "time_2s": fake_result(rng, all_features(TIME, 0.2)),
                "time_378s": fake_result(rng, all_features(TIME, 0.3))} for o in figures.ODORS}


def test_fig4_AB_and_S3_build(per_odor):
    fig = figures.build_fig4_AB(per_odor, PHASE, TIME)
    assert len(fig.axes) == 2 + 1 + 3                         # two heatmaps, colorbar, three ROC panels
    assert [ax.get_title() for ax in fig.axes[3:]] == ["Peppermint vs Water", "Oregano vs Water", "Grapefruit vs Water"]
    s3 = figures.build_figS3(figures.figS3_data(per_odor, PHASE))
    assert len(s3.axes) == 3 + 1
    assert figures.figS3_data(per_odor, PHASE)["Peppermint"].shape == (3, 4)


def hab_results(rng, early_sel, late_sel):
    return {o: {"res_early": fake_result(rng, early_sel[o]), "res_late": fake_result(rng, late_sel[o]),
                "ci_early": (0.0, 1.0), "ci_late": (0.0, 1.0)} for o in figures.ODORS}


def test_fig5_selection_convention_changes_which_features_pass_threshold():
    # paper scenario for theta 0-90 deg in early trials: peppermint 100%, oregano 94%, grapefruit never selected
    early = {"peppermint": {"theta_p0_90": 1.0, "alpha_p180_270": 1.0},
             "oregano": {"theta_p0_90": 0.94, "alpha_p180_270": 1.0},
             "citrus": {"alpha_p180_270": 1.0}}
    late = {o: {"alpha_p180_270": 0.5} for o in figures.ODORS}
    rng = np.random.default_rng(1)
    present_only = figures.build_fig5(hab_results(rng, early, late), count_unselected_as_zero=False)
    with_zeros = figures.build_fig5(hab_results(rng, early, late), count_unselected_as_zero=True)
    n_bars = lambda fig: len(fig.axes[1].patches) // 2          # Early + Late bar per feature
    assert n_bars(present_only) == 2                            # 0.97 >= 0.75: theta 0-90 is shown
    assert n_bars(with_zeros) == 1                              # 0.647 < 0.75: it drops out
