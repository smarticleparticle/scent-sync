"""Oscillatory basis of the PAC effects (paper Section 1.2, Fig. S2, Table S2).

Ported from analysis-notebook cells 100-104.  The analysis asks whether respiration-EEG PAC survives restricting every
subject's whole-head average to channels that show a genuine oscillatory peak (FOOOF) in the band, and whether the
aperiodic background itself differs between inhalation and exhalation.

Design: the only step that needs the multi-GB continuous EEG is the Welch spectrum of each channel.  ``compute_psd_cache``
therefore computes all spectra once and stores them in a small file (a few MB); everything else (FOOOF fits, gating,
statistics) works from that cache, so it can be re-run from the public data deposit.

Functions ``fit_fooof``, ``compute_psd`` and ``band_has_peak`` are the notebook's (cell 101); the rest re-implements
cells 100-104 with explicit arguments instead of notebook globals (verified identical on synthetic data).
"""
from fractions import Fraction

import numpy as np
from fooof import FOOOF
from scipy.signal import resample_poly, welch

from .classify import fdr_bh
from .pac import compute_linearized_respiratory_phase_direct
from .pac_stats import cohens_d_onesample, sign_flip_permutation_pvalue  # noqa: F401

DEFAULT_SETTINGS = dict(peak_width_limits=[1, 12], max_n_peaks=6, min_peak_height=0, peak_threshold=2.0,
                        aperiodic_mode="fixed")


# ---------------------------------------------------------------------------------------------------------
# notebook cell 101 (verbatim)
# ---------------------------------------------------------------------------------------------------------
def fit_fooof(freqs, psd, freq_range=(1, 45), settings=None):
    if settings is None:
        settings = dict(peak_width_limits=[1, 12], max_n_peaks=6,
                        min_peak_height=0, peak_threshold=2.0,
                        aperiodic_mode='fixed')
    fm = FOOOF(**settings, verbose=False)
    fm.fit(freqs, psd, freq_range)
    return fm


def compute_psd(eeg_signal, sfreq, fmin=1, fmax=45, nperseg_sec=4):
    nperseg = min(int(nperseg_sec * sfreq), len(eeg_signal))
    freqs, psd = welch(eeg_signal, fs=sfreq, nperseg=nperseg)
    mask = (freqs >= fmin) & (freqs <= fmax)
    return freqs[mask], psd[mask]


def band_has_peak(fm, band):
    peaks = fm.peak_params_
    return any(band[0] <= p[0] <= band[1] for p in peaks)


# ---------------------------------------------------------------------------------------------------------
# spectra cache (the only step that needs the continuous EEG)
# ---------------------------------------------------------------------------------------------------------
def compute_psd_cache(recordings, pac_results, subjects, sfreq_eeg=250, sfreq_resp=8.75714285714, resp_band=(0.1, 0.4)):
    """Welch spectra for the peak gating (every channel with PAC results) and for the inhale/exhale aperiodic test.

    Returns {'psd': {subject: {channel: (freqs, psd)}},                      # 4-s Welch windows, 1-45 Hz
             'inhale_exhale': {subject: {channel_index: None | (f_in, psd_in, f_ex, psd_ex)}}}   # 1-s windows
    """
    cache = {"psd": {}, "inhale_exhale": {}}
    for subject_id in subjects:
        if subject_id in pac_results["subject_level"]:
            raw_eeg = recordings["EEG"][subject_id]
            eeg_data = raw_eeg.get_data()
            cache["psd"][subject_id] = {}
            for ch_idx, ch_name in enumerate(raw_eeg.ch_names):
                if ch_name not in pac_results["subject_level"][subject_id]:
                    continue
                cache["psd"][subject_id][ch_name] = compute_psd(eeg_data[ch_idx, :], sfreq_eeg)

    for subject_id in subjects:
        if subject_id not in recordings["EEG"]:
            continue
        raw_eeg = recordings["EEG"][subject_id]
        resp_signal = recordings["respiration"][subject_id]
        eeg_data = raw_eeg.get_data()
        ratio = Fraction(sfreq_eeg / sfreq_resp).limit_denominator(1000)
        resp_resampled = resample_poly(resp_signal, ratio.numerator, ratio.denominator)
        min_len = min(eeg_data.shape[1], len(resp_resampled))
        resp_resampled = resp_resampled[:min_len]
        resp_phase = compute_linearized_respiratory_phase_direct(resp_resampled, sfreq_eeg, resp_band=resp_band)
        if resp_phase is None:
            continue
        inhale_mask = (resp_phase >= 0) & (resp_phase < np.pi)
        exhale_mask = (resp_phase >= np.pi) & (resp_phase < 2 * np.pi)
        cache["inhale_exhale"][subject_id] = {}
        for ch_idx in range(eeg_data.shape[0]):
            eeg_channel = eeg_data[ch_idx, :min_len]
            eeg_inhale, eeg_exhale = eeg_channel[inhale_mask], eeg_channel[exhale_mask]
            if eeg_inhale.sum() == 0 or eeg_exhale.sum() == 0:
                cache["inhale_exhale"][subject_id][ch_idx] = None
                continue
            f_in, psd_in = compute_psd(eeg_inhale, sfreq_eeg, nperseg_sec=1)
            f_ex, psd_ex = compute_psd(eeg_exhale, sfreq_eeg, nperseg_sec=1)
            cache["inhale_exhale"][subject_id][ch_idx] = (f_in, psd_in, f_ex, psd_ex)
    return cache


# ---------------------------------------------------------------------------------------------------------
# peak gating (cells 100/101)
# ---------------------------------------------------------------------------------------------------------
def build_peak_gate(psd_cache, freq_bands, settings):
    """{subject: {channel: {band: has_peak}}} -- channels whose FOOOF fit failed are left out (as in the notebook)."""
    gate = {}
    for subject_id, channels in psd_cache["psd"].items():
        gate[subject_id] = {}
        for ch_name, (freqs, psd) in channels.items():
            try:
                fm = fit_fooof(freqs, psd, settings=settings)
            except Exception:
                continue
            gate[subject_id][ch_name] = {band_name: band_has_peak(fm, tuple(band)) for band_name, band in freq_bands.items()}
    return gate


def peak_prevalence(gate, freq_bands):
    """% of channel-subject instances with a peak per band, and the number of instances."""
    out = {}
    for band in freq_bands:
        flags = [g[ch][band] for g in gate.values() for ch in g]
        out[band] = {"pct": 100 * float(np.mean(flags)), "n": len(flags)}
    return out


# ---------------------------------------------------------------------------------------------------------
# gated group statistics (cells 102/104)
# ---------------------------------------------------------------------------------------------------------
def subject_level_zscore(pac_results, band_name, gate_dict=None, gated=False, min_channels=20):
    """Per-subject PAC z: whole-head mean MI vs the channel-averaged surrogate distribution (optionally gated)."""
    subject_z = {}
    for subject_id, chans in pac_results["subject_level"].items():
        obs_vals, null_arrays = [], []
        for ch_name, band_results in chans.items():
            r = band_results.get(band_name)
            if r is None:
                continue
            if gated and not gate_dict.get(subject_id, {}).get(ch_name, {}).get(band_name, False):
                continue
            obs_vals.append(r["MI_observed"])
            null_arrays.append(r["MI_null_distribution"])
        if len(obs_vals) < min_channels:
            continue
        subject_obs_MI = np.mean(obs_vals)
        subject_null_dist = np.stack(null_arrays, axis=0).mean(axis=0)
        null_mu = subject_null_dist.mean()
        null_sd = subject_null_dist.std(ddof=1) + 1e-12
        subject_z[subject_id] = (subject_obs_MI - null_mu) / null_sd
    return subject_z


def run_gated_stats(pac_results, gate_dict, freq_bands, min_channels=20, n_perm=50000, seed=42):
    """Gated group test per band: n subjects, Cohen's d, one-sided sign-flip p, FDR p (BH across bands)."""
    out, p_vals = {}, []
    for band_name in freq_bands:
        z = subject_level_zscore(pac_results, band_name, gate_dict=gate_dict, gated=True, min_channels=min_channels)
        vals = np.array(list(z.values()))
        if len(vals) < 3:
            out[band_name] = dict(n=len(vals), d=np.nan, p=np.nan)
            p_vals.append(1.0)
            continue
        p = sign_flip_permutation_pvalue(vals, n_perm=n_perm, alternative="greater", seed=seed)
        out[band_name] = dict(n=len(vals), d=vals.mean() / vals.std(ddof=1), p=p)
        p_vals.append(p)
    for band_name, p_adj in zip(freq_bands, fdr_bh(p_vals)):
        out[band_name]["p_fdr"] = p_adj
    return out


def compare_gated_vs_ungated(pac_results, gate_dict, band_name, min_channels=20, n_perm=50000, seed=42):
    """Notebook cell 104: all-channel vs gated group tests and their paired difference (sign-flip permutation)."""
    z_all = subject_level_zscore(pac_results, band_name, gated=False, min_channels=min_channels)
    z_gated = subject_level_zscore(pac_results, band_name, gate_dict=gate_dict, gated=True, min_channels=min_channels)
    subjects_all, subjects_gated = sorted(z_all), sorted(z_gated)
    vals_all = np.array([z_all[s] for s in subjects_all])
    vals_gated = np.array([z_gated[s] for s in subjects_gated])
    p_all = sign_flip_permutation_pvalue(vals_all, n_perm=n_perm, alternative="greater", seed=seed)
    p_gated = sign_flip_permutation_pvalue(vals_gated, n_perm=n_perm, alternative="greater", seed=seed)
    common = sorted(set(subjects_all) & set(subjects_gated))
    paired_diff = np.array([z_gated[s] for s in common]) - np.array([z_all[s] for s in common])
    p_paired = sign_flip_permutation_pvalue(paired_diff, n_perm=n_perm, alternative="two-sided", seed=seed)
    return dict(n_all=len(subjects_all), d_all=vals_all.mean() / vals_all.std(ddof=1), p_all=p_all,
                n_gated=len(subjects_gated), d_gated=vals_gated.mean() / vals_gated.std(ddof=1), p_gated=p_gated,
                n_paired=len(common), mean_paired_diff=paired_diff.mean(), p_paired=p_paired)


# ---------------------------------------------------------------------------------------------------------
# inhale vs exhale aperiodic parameters (cell 103 + 104)
# ---------------------------------------------------------------------------------------------------------
def aperiodic_inhale_exhale(psd_cache, subjects, n_perm=50000, seed=42):
    """Per-subject mean (inhale - exhale) FOOOF offset and exponent over channels, and sign-flip tests (two-sided)."""
    offset_diffs, exponent_diffs = [], []
    for subject_id in subjects:
        if subject_id not in psd_cache["inhale_exhale"]:
            continue
        ch_off_in, ch_off_ex, ch_exp_in, ch_exp_ex = [], [], [], []
        for _, entry in psd_cache["inhale_exhale"][subject_id].items():
            if entry is None:
                continue
            f_in, psd_in, f_ex, psd_ex = entry
            try:
                fm_in = fit_fooof(f_in, psd_in)
                fm_ex = fit_fooof(f_ex, psd_ex)
            except Exception:
                continue
            (off_in, exp_in), (off_ex, exp_ex) = fm_in.aperiodic_params_, fm_ex.aperiodic_params_
            ch_off_in.append(off_in); ch_off_ex.append(off_ex)
            ch_exp_in.append(exp_in); ch_exp_ex.append(exp_ex)
        if ch_off_in:
            offset_diffs.append(np.mean(ch_off_in) - np.mean(ch_off_ex))
            exponent_diffs.append(np.mean(ch_exp_in) - np.mean(ch_exp_ex))
    off, exp = np.array(offset_diffs), np.array(exponent_diffs)
    return dict(
        n_subjects=len(off),
        offset_mean_diff=float(off.mean()), offset_d=float(off.mean() / off.std(ddof=1)),
        offset_p=sign_flip_permutation_pvalue(off, n_perm=n_perm, alternative="two-sided", seed=seed),
        exponent_mean_diff=float(exp.mean()), exponent_d=float(exp.mean() / exp.std(ddof=1)),
        exponent_p=sign_flip_permutation_pvalue(exp, n_perm=n_perm, alternative="two-sided", seed=seed),
        subject_offset_diffs=offset_diffs, subject_exponent_diffs=exponent_diffs)
