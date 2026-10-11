"""Respiration-EEG PAC -- group statistics (paper Section 1.1, Fig. 2).

Ported from analysis notebook cell 96 (functions extracted from the notebook source) and cell 105 (Cohen's d).
Change: statsmodels' FDR call replaced by classify.fdr_bh (verified identical).  Subject-level z-scores compare each
subject's whole-head mean MI with the mean of the channel-averaged surrogate distribution; the group test is a one-sided
sign-flip permutation test (50,000 flips, seed 42) FDR-corrected across the four bands; spatial clusters come from MNE's
cluster-based permutation test on the channel z-scores (1000 permutations, seed 42).
"""
import numpy as np
import pandas as pd
import scipy.stats as scipy_stats
import mne
from mne.channels import find_ch_adjacency
from mne.stats import permutation_cluster_1samp_test

from .classify import fdr_bh

# ---- configuration of the notebook cell (unchanged) ---------------------------------------------------------------
RNG = np.random.default_rng(42)

BANDS_ORDER = ['theta', 'alpha', 'beta', 'low_gamma']
FREQ_BANDS = {
    'theta': (4, 8),
    'alpha': (8, 12),
    'beta': (12, 30),
    'low_gamma': (30, 40),
}
BAND_COLORS = {
    'theta': '#1f77b4',
    'alpha': '#ff7f0e',
    'beta': '#2ca02c',
    'low_gamma': '#d62728'
}

# Figure knobs
ROWB_VLIM = (-2, 6)
N_PERMUTATIONS_CLUSTER = 1000
CLUSTER_TAIL = 1  # one-sided positive

# Whole-head violin knobs
MIN_CHANNELS_WHOLEHEAD = 20
NULL_PER_SUBJECT_FOR_PLOT = 1000
GROUP_P_SOURCE = 'perm'  # 'perm' recommended; can use 'ttest' or 'wilcoxon'

ROI_MAPPING = {
    'frontal': ['E11', 'E12', 'E5', 'E6', 'E13', 'E20', 'E21', 'E29', 'E30',
                'E106', 'E111', 'E112', 'E7', 'E4', 'E118', 'E3', 'E124', 'E19', 'E22'],
    'central': ['E7', 'E31', 'E55', 'E80', 'E106', 'E129', 'E36', 'E104', 'E37'],
    'parietal': ['E52', 'E53', 'E60', 'E61', 'E62', 'E67', 'E72', 'E77', 'E78',
                 'E79', 'E85', 'E86', 'E87', 'E93'],
    'occipital': ['E70', 'E71', 'E75', 'E76', 'E82', 'E83', 'E84', 'E88', 'E89',
                  'E90', 'E91', 'E94', 'E95', 'E96', 'E97'],
    'temporal': ['E33', 'E34', 'E39', 'E40', 'E45', 'E46', 'E50', 'E56', 'E57',
                 'E58', 'E63', 'E64', 'E68', 'E73', 'E108', 'E109', 'E114',
                 'E115', 'E120', 'E121', 'E122', 'E116', 'E110', 'E102', 'E100']
}


def compute_subject_wholehead_stats_with_ids(pac_results, bands_order=BANDS_ORDER, min_channels=20):
    subj_level = pac_results['subject_level']
    n_surrogates_expected = pac_results['metadata']['n_surrogates']

    rows = []
    null_by_subj_band = {}

    for subj_id, ch_dict in subj_level.items():
        ch_names = list(ch_dict.keys())
        if len(ch_names) == 0:
            continue

        for band in bands_order:
            mi_obs_list = []
            mi_null_mat = []

            for ch in ch_names:
                res = ch_dict[ch].get(band, None)
                if res is None:
                    continue
                if 'MI_observed' not in res or 'MI_null_distribution' not in res:
                    continue

                mi_obs = float(res['MI_observed'])
                mi_null = np.asarray(res['MI_null_distribution'], dtype=float)

                if mi_null.ndim != 1 or len(mi_null) != n_surrogates_expected:
                    continue

                mi_obs_list.append(mi_obs)
                mi_null_mat.append(mi_null)

            n_ch = len(mi_obs_list)
            if n_ch < min_channels:
                continue

            mi_obs_list = np.asarray(mi_obs_list, dtype=float)
            mi_null_mat = np.asarray(mi_null_mat, dtype=float)

            mi_subj_obs = mi_obs_list.mean()
            mi_subj_null = mi_null_mat.mean(axis=0)

            null_mu = mi_subj_null.mean()
            null_sd = mi_subj_null.std(ddof=1) + 1e-12
            z_subj = (mi_subj_obs - null_mu) / null_sd

            p_subj = (np.sum(mi_subj_null >= mi_subj_obs) + 1.0) / (len(mi_subj_null) + 1.0)

            rows.append(dict(
                subject_id=subj_id,
                band=band,
                n_channels=n_ch,
                MI_obs=mi_subj_obs,
                MI_null_mean=null_mu,
                MI_null_std=null_sd,
                z=z_subj,
                p_subject=p_subj
            ))

            null_by_subj_band[(subj_id, band)] = mi_subj_null

    return pd.DataFrame(rows), null_by_subj_band


def sign_flip_permutation_pvalue(values, n_perm=50000, alternative='greater', seed=42):
    rng = np.random.default_rng(seed)
    x = np.asarray(values, float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 3:
        return np.nan

    obs = x.mean()
    signs = rng.choice([-1, 1], size=(n_perm, n), replace=True)
    perm_means = (signs * x).mean(axis=1)

    if alternative == 'greater':
        p = (np.sum(perm_means >= obs) + 1) / (n_perm + 1)
    elif alternative == 'less':
        p = (np.sum(perm_means <= obs) + 1) / (n_perm + 1)
    else:
        p = (np.sum(np.abs(perm_means) >= np.abs(obs)) + 1) / (n_perm + 1)
    return p


def compute_group_stats(df_obs, bands_order=BANDS_ORDER, p_source='perm'):
    rows = []
    for band in bands_order:
        d = df_obs[df_obs['band'] == band]
        z = d['z'].to_numpy()
        z = z[~np.isnan(z)]
        n = len(z)
        if n < 3:
            continue

        t_stat, p_t = scipy_stats.ttest_1samp(z, 0.0, alternative='greater')
        try:
            w_stat, p_w = scipy_stats.wilcoxon(z, alternative='greater', zero_method='wilcox')
        except Exception:
            w_stat, p_w = np.nan, np.nan

        p_perm = sign_flip_permutation_pvalue(z, n_perm=50000, alternative='greater', seed=42)

        if p_source == 'ttest':
            p_for_fdr = p_t
        elif p_source == 'wilcoxon':
            p_for_fdr = p_w
        else:
            p_for_fdr = p_perm

        rows.append(dict(
            band=band,
            n_subjects=n,
            mean_z=float(np.mean(z)),
            sem_z=float(scipy_stats.sem(z)),
            p_ttest=float(p_t),
            p_wilcoxon=float(p_w) if not np.isnan(p_w) else np.nan,
            p_perm=float(p_perm),
            p_for_fdr=float(p_for_fdr)
        ))

    out = pd.DataFrame(rows)
    if len(out) == 0:
        return out

    p_fdr = fdr_bh(out['p_for_fdr'].values)       # Benjamini-Hochberg (the notebook used statsmodels' implementation)
    reject = p_fdr <= 0.05
    out['p_fdr'] = p_fdr
    out['sig_fdr_05'] = reject
    return out


def p_to_stars(p):
    if p < 1e-3: return '***'
    if p < 1e-2: return '**'
    if p < 0.05: return '*'
    return 'n.s.'


def assign_primary_regions_inplace(cluster_result):
    for cl in cluster_result.get('significant_clusters', []):
        region_counts = {}
        for roi_name, roi_channels in ROI_MAPPING.items():
            overlap = set(cl['channels']) & set(roi_channels)
            if overlap:
                region_counts[roi_name] = len(overlap)
        if region_counts:
            cl['primary_region'] = max(region_counts, key=region_counts.get)
        else:
            cl['primary_region'] = 'unknown'


def run_spatial_cluster_permutation_pac(pac_results, full_data_dict, band_name,
                                        n_permutations=1000, threshold=None):
    """
    YOUR exact cluster permutation test version:
    - subject×channel matrix of z_score
    - filter channels with >=50% coverage
    - fill remaining NaNs with 0
    - adjacency from find_ch_adjacency
    - permutation_cluster_1samp_test tail=1
    - assign primary_region via ROI_MAPPING
    """
    first_subject = list(pac_results['subject_level'].keys())[0]
    raw_example = full_data_dict['EEG'][first_subject]

    all_channels = raw_example.ch_names
    metadata = full_data_dict['metadata'][first_subject]
    bad_channels = metadata['bad_channels']
    good_channels = [ch for ch in all_channels if ch not in bad_channels]

    n_subjects = len(pac_results['subject_level'].keys())
    n_channels = len(good_channels)

    data_matrix = np.full((n_subjects, n_channels), np.nan)

    for subj_idx, subject_id in enumerate(pac_results['subject_level'].keys()):
        for ch_idx, ch_name in enumerate(good_channels):
            if ch_name in pac_results['subject_level'][subject_id]:
                if band_name in pac_results['subject_level'][subject_id][ch_name]:
                    result = pac_results['subject_level'][subject_id][ch_name][band_name]
                    if result is not None:
                        data_matrix[subj_idx, ch_idx] = result['z_score']

    n_valid_per_channel = np.sum(~np.isnan(data_matrix), axis=0)

    valid_channel_mask = n_valid_per_channel >= (n_subjects * 0.5)
    data_matrix = data_matrix[:, valid_channel_mask]
    valid_channels = [good_channels[i] for i in range(len(good_channels)) if valid_channel_mask[i]]

    # Fill remaining NaNs with 0 (your choice)
    data_matrix = np.nan_to_num(data_matrix, nan=0.0)

    info_subset = mne.pick_info(raw_example.info,
                                [raw_example.ch_names.index(ch) for ch in valid_channels])
    adjacency, _ = find_ch_adjacency(info_subset, ch_type='eeg')

    if threshold is None:
        threshold = scipy_stats.t.ppf(1 - 0.05, n_subjects - 1)

    T_obs, clusters, cluster_pv, H0 = permutation_cluster_1samp_test(
        data_matrix,
        threshold=threshold,
        n_permutations=n_permutations,
        tail=CLUSTER_TAIL,
        adjacency=adjacency,
        n_jobs=1,
        seed=42,
        verbose=False,
        out_type='indices'
    )

    significant_clusters = []
    for i, (cluster, p_val) in enumerate(zip(clusters, cluster_pv)):
        cluster_channels_idx = cluster[0] if isinstance(cluster, tuple) else cluster
        cluster_channels = [valid_channels[idx] for idx in cluster_channels_idx]
        cluster_t_values = T_obs[cluster_channels_idx]
        cluster_mass = np.sum(cluster_t_values)

        if p_val < 0.05:
            significant_clusters.append({
                'cluster_id': i,
                'p_value': float(p_val),
                'channels': cluster_channels,
                'channel_indices': cluster_channels_idx,
                't_values': cluster_t_values,
                'cluster_mass': float(cluster_mass),
                'n_channels': int(len(cluster_channels))
            })

    out = {
        'T_obs': T_obs,
        'clusters': clusters,
        'cluster_pv': cluster_pv,
        'H0': H0,
        'significant_clusters': significant_clusters,
        'valid_channels': valid_channels,
        'adjacency': adjacency,
        'data_matrix': data_matrix,
        'threshold': float(threshold)
    }
    assign_primary_regions_inplace(out)
    return out


def cohens_d_onesample(x):
    """One-sample Cohen's d = mean / sd (notebook cell 105)."""
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return np.nan
    return x.mean() / (x.std(ddof=1) + 1e-12)


def effect_sizes(df_obs, bands_order=BANDS_ORDER):
    """Per-band effect size table of the whole-head subject z-scores (notebook cell 105)."""
    rows = []
    for band in bands_order:
        z = df_obs.loc[df_obs["band"] == band, "z"].to_numpy()
        rows.append({"band": band, "n": int(np.sum(~np.isnan(z))), "mean_z": float(np.nanmean(z)),
                     "sd_z": float(np.nanstd(z, ddof=1)), "cohens_d": float(cohens_d_onesample(z))})
    return pd.DataFrame(rows)
