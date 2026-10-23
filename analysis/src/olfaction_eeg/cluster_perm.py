"""Cluster-based permutation tests of the ERSPs (paper Section 1.5, SI Fig. S4).

Ported from analysis-notebook cells 117 (``coarsen_bins``), 118 (``build_trial_stack_pooled_odors``), 121 (``restrict_to_region``,
``cluster_permutation_test_2d_region``), 123-127 (driver) and 128 (figure).  The five functions below are the notebook's, extracted from its
source and unchanged (only the imports they relied on as notebook globals -- ``np``, ``ndimage``, ``scipy_stats`` -- are explicit here).

What is tested (Methods, "Cluster-Based Permutation Testing of RPL-ERSP and Time-Locked ERSP"): odor-minus-water difference maps, pooled over the
three odors and averaged over trials 1-3 within each subject, one-sided one-sample t-test per frequency x phase (or time) pixel (p < 0.05),
contiguous suprathreshold pixels form clusters (signed t-mass), significance against the null distribution of the maximum cluster mass from
1000 within-subject shuffles of the odor/water trial labels (seed 42).  Theta enhancement (4-8 Hz, first half of the breath / window) and alpha
suppression (8-12 Hz, second half) are tested separately, in three representations:
    1_traditional_2s   2-s time-locked window (not shown in the paper's figure)
    2_longer_5050      3.78-s time-locked window ("Longer"), split at the midpoint of the post-stimulus window
    3_phase_locked     RPL-ERSP (0-360 deg), split at 180 deg
"""
import numpy as np
from scipy import ndimage
from scipy import stats as scipy_stats


# ---------------------------------------------------------------------------------------------------------
# notebook cell 117, verbatim (coarsen_bins)
# ---------------------------------------------------------------------------------------------------------
def coarsen_bins(trial_stack_dict, bin_factor=10):
    """Average every `bin_factor` consecutive phase bins together."""
    out = {}
    for s, arr in trial_stack_dict.items():
        n_trials, n_freqs, n_bins = arr.shape
        n_new_bins = n_bins // bin_factor
        reshaped = arr[:, :, :n_new_bins * bin_factor].reshape(n_trials, n_freqs, n_new_bins, bin_factor)
        out[s] = reshaped.mean(axis=3)
    return out


# ---------------------------------------------------------------------------------------------------------
# notebook cell 118, verbatim
# ---------------------------------------------------------------------------------------------------------
def build_trial_stack_pooled_odors(results_dict, odor_conditions, subjects,
                                       trial_start=None, trial_end=None):
    """
    Pool trials across multiple odor conditions into one stack per subject,
    matching how Figure 3D's grand average is computed (pooled across odors).
    """
    out = {}
    for s in subjects:
        subj_trials = []
        for cond in odor_conditions:
            if s not in results_dict[cond]:
                continue
            trials = results_dict[cond][s]
            trial_keys = sorted(trials.keys())
            if trial_start is not None or trial_end is not None:
                trial_keys = trial_keys[trial_start:trial_end]
            subj_trials.extend([trials[tk] for tk in trial_keys])
        if len(subj_trials) > 0:
            out[s] = np.stack(subj_trials, axis=0)  # (n_trials_pooled, n_freqs, n_bins)
    return out


# ---------------------------------------------------------------------------------------------------------
# notebook cell 121, verbatim
# ---------------------------------------------------------------------------------------------------------
def restrict_to_region(t_map_shape, freq_range, phase_range, freqs_axis, bins_axis):
    """
    freqs_axis: array of actual frequency values matching t_map's freq dimension
    bins_axis: array of actual phase/time values matching t_map's bin dimension
    """
    freq_mask = (freqs_axis >= freq_range[0]) & (freqs_axis <= freq_range[1])
    bin_mask = (bins_axis >= phase_range[0]) & (bins_axis <= phase_range[1])
    return freq_mask, bin_mask


def cluster_permutation_test_2d_region(odor_trials_by_subject, water_trials_by_subject,
                                          freq_mask, bin_mask, n_perm=1000,
                                          alpha_threshold=0.05, tail='positive', seed=42):
    """
    Same as cluster_permutation_test_2d_onesided, but restricts BOTH the
    observed test and the null permutations to a specific sub-region
    (e.g., theta x inhale, or alpha/beta x exhale) before finding clusters.
    """
    rng = np.random.default_rng(seed)
    subjects = sorted(set(odor_trials_by_subject) & set(water_trials_by_subject))

    def compute_diff_map(odor_dict, water_dict):
        diffs = []
        for s in subjects:
            odor_avg = np.nanmean(odor_dict[s], axis=0)
            water_avg = np.nanmean(water_dict[s], axis=0)
            diff = odor_avg - water_avg
            diffs.append(diff[np.ix_(freq_mask, bin_mask)])  # restrict to region HERE
        return np.stack(diffs, axis=0)

    def tmap_and_clusters(diff_stack, thresh, tail):
        n_subj = diff_stack.shape[0]
        t_map, _ = scipy_stats.ttest_1samp(diff_stack, 0, axis=0)
        t_crit = scipy_stats.t.ppf(1 - thresh, df=n_subj - 1)

        if tail == 'positive':
            sig_mask = t_map > t_crit
            cluster_stat = lambda mask: np.sum(t_map[mask])
        else:
            sig_mask = t_map < -t_crit
            cluster_stat = lambda mask: np.sum(-t_map[mask])

        labeled, n_clusters = ndimage.label(sig_mask)
        cluster_masses = [cluster_stat(labeled == c) for c in range(1, n_clusters + 1)]
        return t_map, labeled, n_clusters, cluster_masses

    diff_obs = compute_diff_map(odor_trials_by_subject, water_trials_by_subject)
    t_map_obs, labeled_obs, n_clusters_obs, masses_obs = tmap_and_clusters(diff_obs, alpha_threshold, tail)

    if n_clusters_obs == 0:
        return {'t_map': t_map_obs, 'labeled': labeled_obs, 'clusters': [], 'n_clusters': 0}

    null_max_masses = []
    for _ in range(n_perm):
        perm_odor, perm_water = {}, {}
        for s in subjects:
            combined = np.concatenate([odor_trials_by_subject[s], water_trials_by_subject[s]], axis=0)
            n_odor = odor_trials_by_subject[s].shape[0]
            perm_idx = rng.permutation(combined.shape[0])
            perm_odor[s] = combined[perm_idx[:n_odor]]
            perm_water[s] = combined[perm_idx[n_odor:]]
        diff_perm = compute_diff_map(perm_odor, perm_water)
        _, _, _, masses_perm = tmap_and_clusters(diff_perm, alpha_threshold, tail)
        null_max_masses.append(max(masses_perm) if masses_perm else 0)

    null_max_masses = np.array(null_max_masses)
    cluster_results = []
    for c_idx, mass in enumerate(masses_obs, start=1):
        p_val = (np.sum(null_max_masses >= mass) + 1) / (n_perm + 1)
        cluster_results.append({'cluster_id': c_idx, 'mass': mass, 'p_value': p_val,
                                  'n_pixels': np.sum(labeled_obs == c_idx)})

    return {'t_map': t_map_obs, 'labeled': labeled_obs, 'n_clusters': n_clusters_obs,
            'clusters': cluster_results}


# ---------------------------------------------------------------------------------------------------------
# notebook cell 127, verbatim (run_cluster_test_suite prints a short summary, as in the notebook)
# ---------------------------------------------------------------------------------------------------------
def run_cluster_test_suite(odor_data, water_data, freqs_axis, bins_axis,
                              theta_range, alpha_range, split_point,
                              n_perm=1000, freq_buffer=0.0, label=""):
    freq_mask_theta, bin_mask_first = restrict_to_region(
        None, freq_range=(theta_range[0]-freq_buffer, theta_range[1]+freq_buffer),
        phase_range=(bins_axis.min(), split_point),
        freqs_axis=freqs_axis, bins_axis=bins_axis
    )
    theta_result = cluster_permutation_test_2d_region(
        odor_data, water_data, freq_mask_theta, bin_mask_first, n_perm=n_perm, tail='positive'
    )

    freq_mask_alpha, bin_mask_second = restrict_to_region(
        None, freq_range=(alpha_range[0]-freq_buffer, alpha_range[1]+freq_buffer),
        phase_range=(split_point, bins_axis.max()),
        freqs_axis=freqs_axis, bins_axis=bins_axis
    )
    alpha_result = cluster_permutation_test_2d_region(
        odor_data, water_data, freq_mask_alpha, bin_mask_second, n_perm=n_perm, tail='negative'
    )

    best_theta_p = min([c['p_value'] for c in theta_result['clusters']], default=1.0)
    best_alpha_p = min([c['p_value'] for c in alpha_result['clusters']], default=1.0)

    print(f"\n{'='*60}\n{label}\n{'='*60}")
    print(f"  Best theta x first-half p-value:  {best_theta_p:.4f}")
    print(f"  Best alpha x second-half p-value: {best_alpha_p:.4f}")

    return {'theta_result': theta_result, 'alpha_result': alpha_result,
            'best_theta_p': best_theta_p, 'best_alpha_p': best_alpha_p,
            'freq_mask_theta': freq_mask_theta, 'bin_mask_first': bin_mask_first,
            'freq_mask_alpha': freq_mask_alpha, 'bin_mask_second': bin_mask_second,
            'freqs_axis': freqs_axis, 'bins_axis': bins_axis}

def restrict_trials_to_posttime(trial_stack_dict, post_stim_mask):
    return {s: arr[:, :, post_stim_mask] for s, arr in trial_stack_dict.items()}


def restrict_trials_to_posttime(trial_stack_dict, post_stim_mask):
    """notebook cell 127, verbatim"""
    return {s: arr[:, :, post_stim_mask] for s, arr in trial_stack_dict.items()}


# ---------------------------------------------------------------------------------------------------------
# driver (notebook cells 122-127)
# ---------------------------------------------------------------------------------------------------------
def best_cluster(sub_result):
    """The cluster with the smallest p (first one on ties), as used for the figure title; None if there is no cluster."""
    if not sub_result["clusters"]:
        return None
    return min(sub_result["clusters"], key=lambda c: c["p_value"])


def run_all(phase_ersp, time_short_ersp, time_long_ersp, subjects, odors, *, trial_start=0, trial_end=3, n_perm=1000,
            theta_range=(4, 8), alpha_range=(8, 12), bin_factor=1, freq_buffer=0.0):
    """Runs 1-3 of notebook cell 127.  Returns {run_name: output of run_cluster_test_suite}."""
    out = {}
    # ---- RUN 1: traditional 2-s window, 50/50 split
    for name, ersp, label in (("1_traditional_2s", time_short_ersp, "RUN 1: Traditional 2s window, 50/50 split"),
                              ("2_longer_5050", time_long_ersp, "RUN 2: 3.78s window, plain 50/50 split")):
        odor = build_trial_stack_pooled_odors(ersp, odors, subjects, trial_start=trial_start, trial_end=trial_end)
        water = build_trial_stack_pooled_odors(ersp, ["water"], subjects, trial_start=trial_start, trial_end=trial_end)
        time_axis, freqs_axis = ersp["x"], ersp["freqs"]
        post_mask = time_axis >= 0
        time_post = time_axis[post_mask]
        out[name] = run_cluster_test_suite(
            restrict_trials_to_posttime(odor, post_mask), restrict_trials_to_posttime(water, post_mask), freqs_axis, time_post,
            theta_range=theta_range, alpha_range=alpha_range,
            split_point=time_post.min() + 0.5 * (time_post.max() - time_post.min()),
            n_perm=n_perm, freq_buffer=freq_buffer, label=label)
    # ---- RUN 3: phase-locked (RPL-ERSP), early trials; coarsening factor 1 = no coarsening (cells 122-123)
    odor = coarsen_bins(build_trial_stack_pooled_odors(phase_ersp, odors, subjects, trial_start=trial_start, trial_end=trial_end),
                        bin_factor=bin_factor)
    water = coarsen_bins(build_trial_stack_pooled_odors(phase_ersp, ["water"], subjects, trial_start=trial_start, trial_end=trial_end),
                         bin_factor=bin_factor)
    n_bins = next(iter(odor.values())).shape[2]
    bins_axis = np.arange(n_bins) * bin_factor + (bin_factor / 2)
    out["3_phase_locked"] = run_cluster_test_suite(
        odor, water, phase_ersp["freqs"], bins_axis, theta_range=theta_range, alpha_range=alpha_range, split_point=180,
        n_perm=n_perm, freq_buffer=freq_buffer, label="RUN 3: Phase-locked (RPL-ERSP)")
    return out


def run_region(phase_ersp, subjects, odors, name, freq_range, phase_range, tail, *, trial_start=0, trial_end=3, n_perm=1000,
               bin_factor=1):
    """One extra region test on the phase-locked ERSP (notebook cells 124-126: wider theta / alpha-beta bands).  Returns the best p."""
    odor = coarsen_bins(build_trial_stack_pooled_odors(phase_ersp, odors, subjects, trial_start=trial_start, trial_end=trial_end),
                        bin_factor=bin_factor)
    water = coarsen_bins(build_trial_stack_pooled_odors(phase_ersp, ["water"], subjects, trial_start=trial_start, trial_end=trial_end),
                         bin_factor=bin_factor)
    n_bins = next(iter(odor.values())).shape[2]
    bins_axis = np.arange(n_bins) * bin_factor + (bin_factor / 2)
    fm, bm = restrict_to_region(None, freq_range=tuple(freq_range), phase_range=tuple(phase_range),
                                freqs_axis=phase_ersp["freqs"], bins_axis=bins_axis)
    res = cluster_permutation_test_2d_region(odor, water, fm, bm, n_perm=n_perm, tail=tail)
    return res
