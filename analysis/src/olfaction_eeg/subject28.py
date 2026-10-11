"""Subject-28 sensitivity: SI Section 1.4 (breathing rate vs. trial retention) and SI Table S7 (classification without subject 28).

Ported from analysis-notebook cells 8 (``diagnose_subject``, verbatim), 9 (rejected-trial breakdown), 52-53 (retention vs. breathing rate,
Spearman) and 58 (Table S7).  ``subject_breath_stats`` (mean breath duration per subject) is NOT defined anywhere in the notebook as saved
(only used in cell 52); it is reconstructed here from ``breath_data``/Table S3 logic and checked against the table printed by cell 52.

IMPORTANT (two different trial-quality criteria are mixed in SI 1.4, as in the notebook):
  * ``trial_retention`` (53.7 % for subject 28 = 22/41) uses the FIRST-BREATH detection of Table S3 (>= 2 respiratory cycles).
  * ``diagnose_subject`` (36 rejected, 5 valid of 41) uses the ERSP quality control of the Methods (>= 3 cycles + landmarks).
"""
import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import spearmanr

from olfaction_eeg.breath import bandpass_filter, is_bad_respiration_raw, sufficient_oscillations, CONDITIONS
from olfaction_eeg.features import extract_binned_features
from olfaction_eeg import classify as clf

PHASE_BINS = {"p0_90": (0, 90), "p90_180": (90, 180), "p180_270": (180, 270), "p270_360": (270, 360)}


# ---------------------------------------------------------------------------------------------------------
# notebook cell 8, verbatim
# ---------------------------------------------------------------------------------------------------------
def diagnose_subject(epochs_eeg_respiration, subject_id,
                       sfreq_eeg=250, sfreq_resp=8.75714285714,
                       resp_band=(0.1, 0.4), resp_filter_order=2,
                       accept_window=(-0.4, 3.0), reject_window=(-1.0, -0.4),
                       prestim_duration=1.0):
    from fractions import Fraction

    ratio = Fraction(sfreq_eeg / sfreq_resp).limit_denominator(1000)
    up, down = ratio.numerator, ratio.denominator

    min_samples = int(2.0 * sfreq_eeg)
    max_samples = int(8.0 * sfreq_eeg)
    stimulus_idx = int(prestim_duration * sfreq_eeg)

    conditions = ['water', 'peppermint', 'oregano', 'citrus']

    skip_reasons = {
        'bad_resp_raw': 0, 'insufficient_oscillations': 0,
        'phase_none_too_few_peaks': 0, 'phase_none_no_peak_in_accept_window': 0,
        'phase_none_duration_out_of_range': 0, 'phase_none_other': 0, 'valid': 0,
    }
    detail_rows = []

    for condition in conditions:
        if subject_id not in epochs_eeg_respiration['EEG'][condition]:
            continue

        n_trials = epochs_eeg_respiration['EEG'][condition][subject_id].shape[0]
        n_times_eeg = epochs_eeg_respiration['EEG'][condition][subject_id].shape[2]

        for trial_idx in range(n_trials):
            resp_trial = np.squeeze(epochs_eeg_respiration['respiration'][condition][subject_id][trial_idx])
            row = {'condition': condition, 'trial': trial_idx}

            if is_bad_respiration_raw(resp_trial):
                skip_reasons['bad_resp_raw'] += 1
                row['reason'] = 'bad_resp_raw'
                row['std'] = np.std(resp_trial)
                detail_rows.append(row)
                continue

            if not sufficient_oscillations(resp_trial, fs=sfreq_resp):
                skip_reasons['insufficient_oscillations'] += 1
                peaks_raw, _ = signal.find_peaks(resp_trial, distance=int(0.5*sfreq_resp), prominence=0.05)
                row['reason'] = 'insufficient_oscillations'
                row['n_peaks_native_rate'] = len(peaks_raw)
                detail_rows.append(row)
                continue

            resp_filtered = bandpass_filter(resp_trial, resp_band[0], resp_band[1], sfreq_resp, resp_filter_order)
            resp_filtered = (resp_filtered - np.mean(resp_filtered)) / (np.std(resp_filtered) + 1e-12)
            resp_upsampled = signal.resample_poly(resp_filtered, up, down)
            if len(resp_upsampled) > n_times_eeg:
                resp_upsampled = resp_upsampled[:n_times_eeg]
            else:
                resp_upsampled = np.pad(resp_upsampled, (0, n_times_eeg - len(resp_upsampled)), mode='edge')

            peaks, _ = signal.find_peaks(resp_upsampled, distance=int(0.5*sfreq_eeg), prominence=0.05)
            if len(peaks) < 2:
                skip_reasons['phase_none_too_few_peaks'] += 1
                row['reason'] = 'too_few_peaks'
                row['n_peaks_upsampled'] = len(peaks)
                detail_rows.append(row)
                continue

            accept_start_idx = stimulus_idx + int(accept_window[0] * sfreq_eeg)
            accept_end_idx = stimulus_idx + int(accept_window[1] * sfreq_eeg)
            peaks_in_accept = peaks[(peaks >= accept_start_idx) & (peaks <= accept_end_idx)]

            if len(peaks_in_accept) == 0:
                skip_reasons['phase_none_no_peak_in_accept_window'] += 1
                row['reason'] = 'no_peak_in_accept_window'
                row['nearest_peak_time_sec'] = (peaks[0] - stimulus_idx)/sfreq_eeg if len(peaks) > 0 else None
                detail_rows.append(row)
                continue

            peak1_idx = peaks_in_accept[0]
            peaks_after_first = peaks[peaks > peak1_idx]
            if len(peaks_after_first) == 0:
                skip_reasons['phase_none_other'] += 1
                row['reason'] = 'no_second_peak'
                detail_rows.append(row)
                continue

            peak2_idx = peaks_after_first[0]
            breath_duration_samples = peak2_idx - peak1_idx

            if breath_duration_samples < min_samples or breath_duration_samples > max_samples:
                skip_reasons['phase_none_duration_out_of_range'] += 1
                row['reason'] = 'duration_out_of_range'
                row['breath_duration_sec'] = breath_duration_samples / sfreq_eeg
                detail_rows.append(row)
                continue

            skip_reasons['valid'] += 1
            row['reason'] = 'valid'
            detail_rows.append(row)

    trial_counts_global = {}
    trial_counts_global[('all', subject_id)] = {"valid": skip_reasons['valid'],
                                                    "skipped": sum(v for k,v in skip_reasons.items() if k != 'valid')}

    detail_df = pd.DataFrame(detail_rows)
    return skip_reasons, detail_df


# ---------------------------------------------------------------------------------------------------------
# notebook cell 9: how many rejected trials would pass a relaxed 2-cycle threshold?
# ---------------------------------------------------------------------------------------------------------
def rejected_cycle_breakdown(detail_df):
    """Rejected-for-insufficient-oscillations trials: number of detected respiratory cycles (native rate) and the 2-cycle counterfactual."""
    if "reason" not in detail_df or "n_peaks_native_rate" not in detail_df:        # nothing was rejected for this reason
        return {"n_rejected": 0, "cycle_counts": {}, "would_pass_at_2_cycles": 0, "still_fail_at_2_cycles": 0,
                "n_with_0_or_1_cycles": 0}
    rej = detail_df[detail_df["reason"] == "insufficient_oscillations"]
    counts = rej["n_peaks_native_rate"].value_counts().sort_index()
    return {
        "n_rejected": int(len(rej)),
        "cycle_counts": {float(k): int(v) for k, v in counts.items()},
        "would_pass_at_2_cycles": int((rej["n_peaks_native_rate"] >= 2).sum()),
        "still_fail_at_2_cycles": int((rej["n_peaks_native_rate"] < 2).sum()),
        "n_with_0_or_1_cycles": int((rej["n_peaks_native_rate"] < 2).sum()),
    }


# ---------------------------------------------------------------------------------------------------------
# notebook cell 52: trial retention per subject (first-breath criterion) and breathing rate
# ---------------------------------------------------------------------------------------------------------
def trial_retention(epochs, breath_data, subjects):
    """One row per subject: trials attempted (all conditions) and kept by the first-breath detection (Table S3 criterion)."""
    rows = []
    for condition in CONDITIONS:
        for subject in epochs["EEG"][condition].keys():
            rows.append({"subject": subject, "condition": condition,
                         "n_attempted": epochs["EEG"][condition][subject].shape[0],
                         "n_kept": len(breath_data[condition].get(subject, {}))})
    df = pd.DataFrame(rows)
    out = (df.groupby("subject").agg(total_attempted=("n_attempted", "sum"), total_kept=("n_kept", "sum")).reset_index())
    out["pct_kept"] = 100 * out["total_kept"] / out["total_attempted"]
    return out[out["subject"].isin(subjects)].reset_index(drop=True)


def subject_breath_stats(df_breath, window_sec=16.0):
    """Mean first-breath duration per subject and the implied number of breaths per 16-s epoch.

    ``mean_breath_sec``: mean of ``breath_duration`` (inhale + exhale) over ALL kept trials of the subject (trial-weighted).
    ``mean_breath_sec_condmean``: the same but averaging the four condition means (an equally plausible reading of the notebook);
    the checks report which one reproduces the table printed by cell 52.
    """
    g = df_breath.groupby("subject")["breath_duration"].mean().rename("mean_breath_sec")
    g2 = (df_breath.groupby(["subject", "condition"])["breath_duration"].mean().groupby("subject").mean()
          .rename("mean_breath_sec_condmean"))
    out = pd.concat([g, g2], axis=1).reset_index()
    out["breaths_per_16s"] = window_sec / out["mean_breath_sec"]
    out["breaths_per_16s_condmean"] = window_sec / out["mean_breath_sec_condmean"]
    return out


def retention_vs_rate(merged, drop_subject="28"):
    """Spearman correlation of breathing rate with % trials kept: all subjects and without ``drop_subject`` (cells 52-53)."""
    r, p = spearmanr(merged["breaths_per_16s"], merged["pct_kept"])
    rest = merged[merged["subject"] != drop_subject]
    r2, p2 = spearmanr(rest["breaths_per_16s"], rest["pct_kept"])
    return {"r_all": float(r), "p_all": float(p), "n_all": int(len(merged)),
            "r_without": float(r2), "p_without": float(p2), "n_without": int(len(rest))}


# ---------------------------------------------------------------------------------------------------------
# notebook cell 58: Table S7
# ---------------------------------------------------------------------------------------------------------
def auc_without_subject(phase_ersp, subjects, odors, drop_subject="28", C=1.0, trial_start=0, trial_end=3, x_bins=None):
    """Phase-locked (4-bin) early-trial AUC with and without ``drop_subject``, per odor."""
    x_bins = x_bins or PHASE_BINS
    kept = [s for s in subjects if s != drop_subject]
    rows = []
    for odor in odors:
        auc = {}
        for tag, subj in (("with", subjects), ("without", kept)):
            df = extract_binned_features(phase_ersp, "water", odor, subj, freqs=phase_ersp["freqs"], x_bins=x_bins,
                                         x=phase_ersp["x"], trial_start=trial_start, trial_end=trial_end)
            auc[tag] = clf.nested_loso_selk_logreg(df, C=C, verbose=False)["auc"]
        rows.append({"odor": odor, "auc_with_subj28": auc["with"], "auc_without_subj28": auc["without"],
                     "delta": auc["without"] - auc["with"]})
    return pd.DataFrame(rows)
