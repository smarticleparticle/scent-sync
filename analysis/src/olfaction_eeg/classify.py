"""Nested leave-one-subject-out classification and the statistics built on it.

Consolidates what used to live in three places:
  * analysis.nested_loso_selk_logreg            (src/analysis.py)
  * nested_loso_with_subject_tracking           (notebook cell 22) -- a copy of the
    above that also tracked which subject each prediction belonged to.  The
    merged function below returns the union of both outputs (the only addition
    is the ``subject`` key), so there is now exactly one implementation.
  * bootstrap / discrimination / permutation helpers (notebook cells 22, 23, 25, 76)
    -- cell 23 and cell 76's ``pooled_permutation_auc_delta_generic`` were the same
    function; they are now one.
"""
from collections import Counter

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _make_pipe(C):
    return Pipeline([
        ("scaler", StandardScaler()),
        # penalty="l2" is the default; leaving it implicit avoids a deprecation
        # warning in scikit-learn >= 1.8 and is numerically identical.
        ("logreg", LogisticRegression(
            solver="liblinear", C=C,
            class_weight="balanced", max_iter=2000,
        )),
    ])


# ---------------------------------------------------------------------------
# Core classifier
# ---------------------------------------------------------------------------
def nested_loso_selk_logreg(df, C=1.0, verbose=True):
    """
    Nested LOSO for top-k feature selection (no leakage).

    df columns:
      - subject (group id)
      - label   (0/1)
      - n_trials (optional, ignored)
      - remaining columns are features

    Inner loop: on TRAIN subjects only, feature importance = mean |logreg coef|
                across inner LOSO folds; k is chosen by inner-loop AUC.
    Outer loop: fit on all outer-train subjects using the top-k features,
                evaluate on the held-out subject.

    Returns a dict with predictions, AUC/accuracy, per-fold selected features and k,
    feature-selection frequency, and the ``subject`` of every prediction.
    """
    feature_cols = [c for c in df.columns if c not in ["subject", "label", "n_trials"]]
    n_features = len(feature_cols)

    X_all = df[feature_cols].values
    y_all = df["label"].values
    groups_all = df["subject"].values

    outer_logo = LeaveOneGroupOut()

    y_true_all, y_proba_all, y_pred_all, subject_all = [], [], [], []
    selected_features_per_fold = []
    selected_k_per_fold = []
    importance_per_fold = []

    for outer_tr, outer_te in outer_logo.split(X_all, y_all, groups=groups_all):
        X_tr, X_te = X_all[outer_tr], X_all[outer_te]
        y_tr, y_te = y_all[outer_tr], y_all[outer_te]
        g_tr = groups_all[outer_tr]
        g_te = groups_all[outer_te]

        # Inner LOSO: rank features and choose k using TRAIN subjects only
        inner_logo = LeaveOneGroupOut()
        inner_abs_coefs = []

        ks = np.arange(n_features)
        y_pred_proba_ks = np.zeros((len(X_tr), n_features))
        y_true_ks = np.zeros((len(X_tr), n_features))

        for inner_tr, inner_te in inner_logo.split(X_tr, y_tr, groups=g_tr):
            X_inner_tr, y_inner_tr = X_tr[inner_tr], y_tr[inner_tr]
            X_inner_te, y_inner_te = X_tr[inner_te], y_tr[inner_te]

            pipe = _make_pipe(C)
            pipe.fit(X_inner_tr, y_inner_tr)
            abs_coefs = np.abs(pipe.named_steps["logreg"].coef_[0])
            inner_abs_coefs.append(abs_coefs)
            abs_sort = np.argsort(abs_coefs)[::-1]
            for k in ks:
                idx = abs_sort[:k + 1]
                pipe.fit(X_inner_tr[:, idx], y_inner_tr)
                y_proba = pipe.predict_proba(X_inner_te[:, idx])[:, 1]
                y_pred_proba_ks[inner_te, k] = y_proba
                y_true_ks[inner_te, k] = y_inner_te

        auc_k = [roc_auc_score(y1, y2) for y1, y2 in zip(y_true_ks.T, y_pred_proba_ks.T)]
        k_sel = np.argmax(auc_k) + 1

        inner_abs_coefs = np.vstack(inner_abs_coefs)       # (n_inner_folds, n_features)
        importance = inner_abs_coefs.mean(axis=0)           # mean |coef| across inner folds
        topk_idx = np.argsort(importance)[::-1][:k_sel]
        topk_features = [feature_cols[i] for i in topk_idx]
        importance_per_fold.append(importance)
        selected_features_per_fold.append(topk_features)
        selected_k_per_fold.append(k_sel)

        if verbose:
            print(f"Selected {k_sel} features with AUC {auc_k[k_sel - 1]}: {topk_features}")

        # Fit on outer-train using ONLY the selected features
        final_pipe = _make_pipe(C)
        final_pipe.fit(X_tr[:, topk_idx], y_tr)
        proba = final_pipe.predict_proba(X_te[:, topk_idx])[:, 1]
        pred = final_pipe.predict(X_te[:, topk_idx])

        y_true_all.extend(y_te)
        y_proba_all.extend(proba)
        y_pred_all.extend(pred)
        subject_all.extend(g_te)

    y_true_all = np.array(y_true_all)
    y_proba_all = np.array(y_proba_all)
    y_pred_all = np.array(y_pred_all)
    subject_all = np.array(subject_all)

    auc = roc_auc_score(y_true_all, y_proba_all)
    acc = accuracy_score(y_true_all, y_pred_all)

    feat_counts = Counter(f for fold_feats in selected_features_per_fold for f in fold_feats)
    freq_df = pd.DataFrame({
        "feature": list(feat_counts.keys()),
        "selected_count": list(feat_counts.values()),
    }).sort_values("selected_count", ascending=False)
    freq_df["selected_frac"] = freq_df["selected_count"] / len(selected_features_per_fold)

    return {
        "auc": auc,
        "acc": acc,
        "y_true": y_true_all,
        "y_proba": y_proba_all,
        "y_pred": y_pred_all,
        "subject": subject_all,
        "selected_features_per_fold": selected_features_per_fold,
        "selected_k_per_fold": np.array(selected_k_per_fold),
        "importance_per_fold": np.array(importance_per_fold),
        "selection_frequency": freq_df,
    }


# ---------------------------------------------------------------------------
# Per-subject pairs (each subject contributes exactly one water + one odor row)
# ---------------------------------------------------------------------------
def get_subject_probs(result):
    """Extract (water_proba, odor_proba) pair per subject from a LOSO result."""
    df = pd.DataFrame({"subject": result["subject"], "y_true": result["y_true"],
                       "y_proba": result["y_proba"]})
    out = {}
    for subj, g in df.groupby("subject"):
        water = g[g.y_true == 0]["y_proba"]
        odor = g[g.y_true == 1]["y_proba"]
        if len(water) == 1 and len(odor) == 1:
            out[subj] = (water.values[0], odor.values[0])
    return out


def compute_aggregated_auc_from_pairs(pairs):
    y_true, y_proba = [], []
    for _, (w, o) in pairs.items():
        y_true += [0, 1]
        y_proba += [w, o]
    return roc_auc_score(y_true, y_proba)


def compute_subject_discrimination(result):
    """1 if the odor row's probability exceeded the water row's, else 0, per subject."""
    df = pd.DataFrame({"subject": result["subject"], "y_true": result["y_true"],
                       "y_proba": result["y_proba"]})
    discrimination = {}
    for subj, group in df.groupby("subject"):
        water = group[group["y_true"] == 0]["y_proba"]
        odor = group[group["y_true"] == 1]["y_proba"]
        if len(water) == 1 and len(odor) == 1:
            discrimination[subj] = int(odor.values[0] > water.values[0])
    return discrimination


# ---------------------------------------------------------------------------
# Uncertainty / comparison statistics
# ---------------------------------------------------------------------------
def bootstrap_auc_ci(result, n_boot=2000, ci=95, seed=42):
    """
    Bootstrap CI on the aggregated AUC by resampling SUBJECTS with replacement
    (preserving each subject's water/odor pair).

    Returns (auc, lower, upper, boot_aucs).
    (Same RNG stream and arithmetic as the original notebook version; per-subject
    arrays are pre-split instead of re-filtering a DataFrame on every draw.)
    """
    rng = np.random.default_rng(seed)
    subjects = np.unique(result["subject"])
    y_true = np.asarray(result["y_true"])
    y_proba = np.asarray(result["y_proba"])
    subj_arr = np.asarray(result["subject"])
    per_subj = {s: (y_true[subj_arr == s], y_proba[subj_arr == s]) for s in subjects}

    boot_aucs = []
    for _ in range(n_boot):
        sampled = rng.choice(subjects, size=len(subjects), replace=True)
        yt = np.concatenate([per_subj[s][0] for s in sampled])
        yp = np.concatenate([per_subj[s][1] for s in sampled])
        if len(np.unique(yt)) < 2:
            continue
        boot_aucs.append(roc_auc_score(yt, yp))

    boot_aucs = np.array(boot_aucs)
    lower = np.percentile(boot_aucs, (100 - ci) / 2)
    upper = np.percentile(boot_aucs, 100 - (100 - ci) / 2)
    return result["auc"], lower, upper, boot_aucs


def paired_discrimination_test(disc_a, disc_b):
    """
    Paired comparison of two per-subject discrimination dicts (e.g. phase vs time),
    restricted to subjects present in both (Wilcoxon signed-rank on paired binary
    outcomes).
    """
    common_subjects = sorted(set(disc_a.keys()) & set(disc_b.keys()))
    a_vals = np.array([disc_a[s] for s in common_subjects])
    b_vals = np.array([disc_b[s] for s in common_subjects])

    n_a_only = np.sum((a_vals == 1) & (b_vals == 0))
    n_b_only = np.sum((a_vals == 0) & (b_vals == 1))
    n_both = np.sum((a_vals == 1) & (b_vals == 1))
    n_neither = np.sum((a_vals == 0) & (b_vals == 0))

    diffs = a_vals - b_vals
    if np.all(diffs == 0):
        stat, p = np.nan, 1.0
    else:
        stat, p = wilcoxon(diffs)

    return {
        "n_subjects": len(common_subjects),
        "a_discriminated_pct": 100 * a_vals.mean(),
        "b_discriminated_pct": 100 * b_vals.mean(),
        "n_a_only": n_a_only, "n_b_only": n_b_only,
        "n_both": n_both, "n_neither": n_neither,
        "wilcoxon_stat": stat, "p_value": p,
    }


def paired_permutation_auc_delta(res_a, res_b, n_perm=5000, seed=42):
    """
    Paired permutation test on the AUC delta between two models evaluated on the
    same subjects: under the null, swapping each subject's model-A/model-B
    assignment should not systematically change the delta.
    """
    rng = np.random.default_rng(seed)
    probs_a = get_subject_probs(res_a)
    probs_b = get_subject_probs(res_b)
    common = sorted(set(probs_a) & set(probs_b))

    auc_a_obs = compute_aggregated_auc_from_pairs({s: probs_a[s] for s in common})
    auc_b_obs = compute_aggregated_auc_from_pairs({s: probs_b[s] for s in common})
    observed_delta = auc_a_obs - auc_b_obs

    null_deltas = []
    for _ in range(n_perm):
        perm_a, perm_b = {}, {}
        for s in common:
            if rng.random() < 0.5:
                perm_a[s], perm_b[s] = probs_a[s], probs_b[s]
            else:
                perm_a[s], perm_b[s] = probs_b[s], probs_a[s]
        null_deltas.append(
            compute_aggregated_auc_from_pairs(perm_a) - compute_aggregated_auc_from_pairs(perm_b)
        )

    null_deltas = np.array(null_deltas)
    p_value = (np.sum(np.abs(null_deltas) >= np.abs(observed_delta)) + 1) / (n_perm + 1)

    return {
        "observed_auc_a": auc_a_obs, "observed_auc_b": auc_b_obs,
        "observed_delta": observed_delta, "null_deltas": null_deltas,
        "p_value": p_value, "n_subjects": len(common),
    }


def pooled_permutation_auc_delta(results_by_odor, key_a, key_b, n_perm=5000, seed=42):
    """
    Pool (subject, odor) pairs across all odors into ONE permutation test.

    results_by_odor = {odor: {key_a: res, key_b: res, ...}}
    (Replaces notebook cell 23's version [key_a='res_phase', key_b='res_time'] and
    cell 76's ``pooled_permutation_auc_delta_generic`` [e.g. 'res_early', 'res_late'].)
    """
    rng = np.random.default_rng(seed)

    all_pairs_a, all_pairs_b = {}, {}
    for odor, res in results_by_odor.items():
        probs_a = get_subject_probs(res[key_a])
        probs_b = get_subject_probs(res[key_b])
        common = set(probs_a) & set(probs_b)
        for s in common:
            key = f"{s}_{odor}"
            all_pairs_a[key] = probs_a[s]
            all_pairs_b[key] = probs_b[s]

    keys = sorted(all_pairs_a.keys())
    auc_a_obs = compute_aggregated_auc_from_pairs(all_pairs_a)
    auc_b_obs = compute_aggregated_auc_from_pairs(all_pairs_b)
    observed_delta = auc_a_obs - auc_b_obs

    null_deltas = []
    for _ in range(n_perm):
        perm_a, perm_b = {}, {}
        for k in keys:
            if rng.random() < 0.5:
                perm_a[k], perm_b[k] = all_pairs_a[k], all_pairs_b[k]
            else:
                perm_a[k], perm_b[k] = all_pairs_b[k], all_pairs_a[k]
        null_deltas.append(
            compute_aggregated_auc_from_pairs(perm_a) - compute_aggregated_auc_from_pairs(perm_b)
        )

    null_deltas = np.array(null_deltas)
    p_value = (np.sum(np.abs(null_deltas) >= np.abs(observed_delta)) + 1) / (n_perm + 1)

    return {
        "auc_a_pooled": auc_a_obs, "auc_b_pooled": auc_b_obs,
        "observed_delta": observed_delta, "p_value": p_value, "n_units": len(keys),
    }


# ---------------------------------------------------------------------------
# Decision-value-gap comparison (paper Table S4) and multiple-comparison correction
# ---------------------------------------------------------------------------
def get_decision_gaps(result):
    """Continuous per-subject decision-value gap: P(odor | odor trial) - P(odor | water trial).

    (Notebook cell 57.)  Only subjects with exactly one water and one odor row are kept.
    """
    df = pd.DataFrame({"subject": result["subject"], "y_true": result["y_true"],
                       "y_proba": result["y_proba"]})
    gaps = {}
    for subj, group in df.groupby("subject"):
        water = group.loc[group["y_true"] == 0, "y_proba"]
        odor = group.loc[group["y_true"] == 1, "y_proba"]
        if len(water) == 1 and len(odor) == 1:
            gaps[subj] = odor.values[0] - water.values[0]
    return gaps


def paired_gap_permutation_test(vals_a, vals_b, n_permutations=5000, seed=0, ci=90):
    """Paired permutation test on the MEAN difference of two per-subject value sets.

    Each iteration swaps a subject's two values with probability 0.5.  Returns
    (observed_diff, p_value, null_lo, null_hi, outside_null_interval); the p-value is
    mean(|null| >= |observed|), exactly as in notebook cell 57 (default seed 0).
    """
    rng = np.random.default_rng(seed)
    vals_a, vals_b = np.asarray(vals_a, dtype=float), np.asarray(vals_b, dtype=float)
    observed_diff = vals_a.mean() - vals_b.mean()
    null_diffs = np.zeros(n_permutations)
    for i in range(n_permutations):
        swap = rng.random(len(vals_a)) < 0.5
        a = np.where(swap, vals_b, vals_a)
        b = np.where(swap, vals_a, vals_b)
        null_diffs[i] = a.mean() - b.mean()
    p_value = (np.abs(null_diffs) >= np.abs(observed_diff)).mean()

    lower_pct = (100 - ci) / 2
    upper_pct = 100 - lower_pct
    null_lo, null_hi = np.percentile(null_diffs, [lower_pct, upper_pct])
    outside_null_interval = (observed_diff < null_lo) or (observed_diff > null_hi)
    return observed_diff, p_value, null_lo, null_hi, outside_null_interval


def fdr_bh(pvals):
    """Benjamini-Hochberg FDR-adjusted p-values (identical to statsmodels' multipletests(method='fdr_bh'))."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adj_sorted = p[order] * n / np.arange(1, n + 1)
    adj_sorted = np.minimum.accumulate(adj_sorted[::-1])[::-1]
    adj = np.empty(n)
    adj[order] = np.clip(adj_sorted, 0, 1)
    return adj


# ---------------------------------------------------------------------------
# Selection-frequency tables (paper Fig. 4A, Fig. 5B, Fig. S3)
# ---------------------------------------------------------------------------
def selection_matrix(result, bin_names, bands=("beta", "alpha", "theta")):
    """(len(bands), len(bin_names)) array of selection frequency in % (0 where a feature was never selected).

    Feature names look like "theta_p0_90" / "alpha_t0.945_1.89" (band + bin name from the config).
    """
    mat = np.zeros((len(bands), len(bin_names)))
    for _, row in result["selection_frequency"].iterrows():
        band, _, bin_name = row["feature"].partition("_")
        if band in bands and bin_name in bin_names:
            mat[bands.index(band), list(bin_names).index(bin_name)] = row["selected_frac"] * 100
    return mat


def mean_selection_frequency(results_by_odor, count_unselected_as_zero=False):
    """feature -> mean selection frequency (0-1) across odors.

    results_by_odor: {odor: LOSO result}.  A result's ``selection_frequency`` table only lists features that were
    selected at least once, so two conventions exist:

      count_unselected_as_zero=False  average only over odors where the feature appears  (notebook cell 84, i.e. the
                                      published Fig. 5B)
      count_unselected_as_zero=True   treat an odor where it never appears as 0           (convention of Fig. 4A)

    They agree for features selected in every odor and differ for the rest (e.g. 0.97 vs 0.65 for theta 0-90 deg).
    """
    per_feat = {}
    for res in results_by_odor.values():
        df = res["selection_frequency"]
        for feat, frac in zip(df["feature"], df["selected_frac"]):
            per_feat.setdefault(feat, []).append(frac)
    n_odors = len(results_by_odor)
    if count_unselected_as_zero:
        return {f: float(np.sum(v) / n_odors) for f, v in per_feat.items()}
    return {f: float(np.mean(v)) for f, v in per_feat.items()}
