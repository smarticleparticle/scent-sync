"""First-breath detection and breath-duration statistics (paper Section 1.3, Table S3).

Ported from analysis-notebook cells 34 (three helper functions), 37 (``extract_first_breath_data``), 46 (tables) and 50
(repeated-measures ANOVA).  The helper functions and ``extract_first_breath_data`` are the notebook's, extracted from its source;
the only change is the option ``store_eeg=False`` (skip the EEG slices; Table S3 needs only durations).  pingouin's
``rm_anova`` is replaced by ``rm_anova_oneway`` (verified identical to pingouin).

NOTE: this first-breath definition (first peak after stimulus onset, troughs before and after it, >= 2 respiratory cycles, inhale and
exhale between 1 and 4 s) is specific to this analysis; it is not the quality control or the landmarks used for the ERSPs.
"""
import numpy as np
import pandas as pd
from scipy import signal
from scipy import stats

CONDITIONS = ["water", "peppermint", "oregano", "citrus"]
METRICS = ["inhale_duration", "exhale_duration", "breath_duration"]


def bandpass_filter(data, lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs
    b, a = signal.butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return signal.filtfilt(b, a, data)


def is_bad_respiration_raw(resp_signal):
    return np.std(resp_signal) < 1e-3 or np.isnan(resp_signal).all()


def sufficient_oscillations(resp_signal, fs, min_cycles=3):
    inhale_peaks, _ = signal.find_peaks(resp_signal, distance=int(0.5 * fs), prominence=0.05)
    return len(inhale_peaks) >= min_cycles


def extract_first_breath_data(epochs_eeg_respiration, all_subject_ch_names,
                               conditions=['water', 'peppermint', 'oregano', 'citrus'],
                               sfreq_eeg=250, sfreq_resp=8.75714285714,
                               prestim_duration=1.0,
                               min_duration=1.0,
                               max_duration=4.0,
                               store_eeg=True):
    """
    Extract EEG data for FIRST INHALE and FIRST EXHALE after odor onset.
    
    Following: Zelano et al. (2016) J Neuroscience - respiratory gating approach
    
    Returns:
    --------
    breath_data : dict
        [condition][subject][trial]['inhale'/'exhale']['eeg_data']
    """
    
    from fractions import Fraction
    
    breath_data = {cond: {} for cond in conditions}
    
    ratio = Fraction(sfreq_eeg / sfreq_resp).limit_denominator(1000)
    up, down = ratio.numerator, ratio.denominator
    
    stimulus_idx = int(prestim_duration * sfreq_eeg)
    min_samples = int(min_duration * sfreq_eeg)
    max_samples = int(max_duration * sfreq_eeg)
    
    print(f"\n{'='*70}")
    print("EXTRACTING FIRST BREATH DATA (INHALE & EXHALE)")
    print(f"{'='*70}")
    
    for condition in conditions:
        
        for subject in epochs_eeg_respiration['EEG'][condition].keys():
            
            if subject not in all_subject_ch_names:
                continue
            
            breath_data[condition][subject] = {}
            
            n_trials = epochs_eeg_respiration['EEG'][condition][subject].shape[0]
            n_channels = epochs_eeg_respiration['EEG'][condition][subject].shape[1]
            n_times_eeg = epochs_eeg_respiration['EEG'][condition][subject].shape[2]
            
            for trial_idx in range(n_trials):
                
                eeg_trial = epochs_eeg_respiration['EEG'][condition][subject][trial_idx]
                resp_trial = epochs_eeg_respiration['respiration'][condition][subject][trial_idx]
                resp_trial = np.squeeze(resp_trial)
                
                # Basic quality checks (reuse your existing functions)
                if is_bad_respiration_raw(resp_trial):
                    continue
                
                if not sufficient_oscillations(resp_trial, fs=sfreq_resp, min_cycles=2):
                    continue
                
                # Filter respiration
                resp_filtered = bandpass_filter(resp_trial, 0.1, 0.4, sfreq_resp, 2)
                resp_filtered = (resp_filtered - np.mean(resp_filtered)) / (np.std(resp_filtered) + 1e-12)
                
                # Upsample to match EEG
                resp_upsampled = signal.resample_poly(resp_filtered, up, down)
                if len(resp_upsampled) > n_times_eeg:
                    resp_upsampled = resp_upsampled[:n_times_eeg]
                else:
                    resp_upsampled = np.pad(resp_upsampled, (0, n_times_eeg - len(resp_upsampled)), mode='edge')
                
                # Find first breath after stimulus
                search_start = stimulus_idx
                search_end = min(stimulus_idx + int(6 * sfreq_eeg), n_times_eeg)
                
                if search_end - search_start < min_samples:
                    continue
                
                # Find peak (inhale→exhale transition) in search window
                resp_segment = resp_upsampled[search_start:search_end]
                
                from scipy.signal import find_peaks
                peaks, _ = find_peaks(resp_segment, distance=int(1.5 * sfreq_eeg))
                
                if len(peaks) < 1:
                    continue
                
                # First peak after stimulus
                first_peak_relative = peaks[0]
                first_peak_absolute = search_start + first_peak_relative
                
                # Find trough BEFORE peak (start of inhale)
                trough_search_start = max(0, first_peak_absolute - int(4 * sfreq_eeg))
                trough_segment = resp_upsampled[trough_search_start:first_peak_absolute]
                
                if len(trough_segment) < min_samples:
                    continue
                
                trough_before_relative = np.argmin(trough_segment)
                trough_before_absolute = trough_search_start + trough_before_relative
                
                # INHALE: trough → peak
                inhale_start = trough_before_absolute
                inhale_end = first_peak_absolute
                inhale_duration = inhale_end - inhale_start
                
                # Find trough AFTER peak (end of exhale)
                trough_search_end = min(n_times_eeg, first_peak_absolute + int(4 * sfreq_eeg))
                trough_segment_after = resp_upsampled[first_peak_absolute:trough_search_end]
                
                if len(trough_segment_after) < min_samples:
                    continue
                
                trough_after_relative = np.argmin(trough_segment_after)
                trough_after_absolute = first_peak_absolute + trough_after_relative
                
                # EXHALE: peak → trough
                exhale_start = first_peak_absolute
                exhale_end = trough_after_absolute
                exhale_duration = exhale_end - exhale_start
                
                # Quality checks
                if inhale_duration < min_samples or inhale_duration > max_samples:
                    continue
                
                if exhale_duration < min_samples or exhale_duration > max_samples:
                    continue
                
                # Extract EEG data
                inhale_eeg = eeg_trial[:, inhale_start:inhale_end] if store_eeg else None
                exhale_eeg = eeg_trial[:, exhale_start:exhale_end] if store_eeg else None
                
                # Store
                breath_data[condition][subject][trial_idx] = {
                    'inhale': {
                        'eeg_data': inhale_eeg,
                        'resp_data': resp_upsampled[inhale_start:inhale_end],
                        'duration_samples': inhale_duration,
                        'duration_seconds': inhale_duration / sfreq_eeg,
                    },
                    'exhale': {
                        'eeg_data': exhale_eeg,
                        'resp_data': resp_upsampled[exhale_start:exhale_end],
                        'duration_samples': exhale_duration,
                        'duration_seconds': exhale_duration / sfreq_eeg,
                    },
                    'channel_names': all_subject_ch_names[subject]
                }
    
    # Summary
    print(f"\nData extraction complete:")
    for condition in conditions:
        total_trials = sum(len(breath_data[condition][s]) for s in breath_data[condition].keys())
        n_subjects = len(breath_data[condition].keys())
        print(f"  {condition:12s}: {n_subjects} subjects, {total_trials} trials")
    
    return breath_data


# ---------------------------------------------------------------------------------------------------------
# notebook cell 46: tables
# ---------------------------------------------------------------------------------------------------------
def breath_table(breath_data, subjects, conditions=CONDITIONS):
    """One row per subject x condition x trial (only the given subjects)."""
    rows = []
    for condition in conditions:
        for subject, trials in breath_data[condition].items():
            if subject not in subjects:
                continue
            for trial_idx, trial_data in trials.items():
                inhale_dur = trial_data['inhale']['duration_seconds']
                exhale_dur = trial_data['exhale']['duration_seconds']
                rows.append({'subject': subject, 'condition': condition, 'trial': trial_idx,
                             'inhale_duration': inhale_dur, 'exhale_duration': exhale_dur,
                             'breath_duration': inhale_dur + exhale_dur})
    return pd.DataFrame(rows)


def subject_condition_means(df_breath):
    """Average across trials within each subject x condition (repeated measures need one value per cell)."""
    return (df_breath.groupby(['subject', 'condition'], as_index=False)
            .agg(inhale_duration=('inhale_duration', 'mean'), exhale_duration=('exhale_duration', 'mean'),
                 breath_duration=('breath_duration', 'mean'), n_trials=('trial', 'count')))


def descriptives(df_subj_cond):
    """Mean and SD (across subjects) per condition and metric."""
    return df_subj_cond.groupby('condition')[METRICS].agg(['mean', 'std'])


# ---------------------------------------------------------------------------------------------------------
# notebook cell 50: one-way repeated-measures ANOVA (replaces pingouin.rm_anova)
# ---------------------------------------------------------------------------------------------------------
def complete_cases(df_subj_cond, n_conditions=4):
    counts = df_subj_cond.groupby('subject')['condition'].nunique()
    return df_subj_cond[df_subj_cond['subject'].isin(counts[counts == n_conditions].index)].copy()


def rm_anova_oneway(df, dv, within='condition', subject='subject'):
    """One-way repeated-measures ANOVA on complete cases.

    Returns dict: n, df1, df2, SS_effect, SS_error, SS_subject, F, p_unc, ng2 (generalized eta squared =
    SS_effect / (SS_effect + SS_error + SS_subject)), eps (Greenhouse-Geisser epsilon), p_gg.
    """
    wide = df.pivot(index=subject, columns=within, values=dv).dropna()
    X = wide.to_numpy(float)
    n, k = X.shape
    grand = X.mean()
    ss_effect = n * ((X.mean(axis=0) - grand) ** 2).sum()
    ss_subject = k * ((X.mean(axis=1) - grand) ** 2).sum()
    ss_total = ((X - grand) ** 2).sum()
    ss_error = ss_total - ss_effect - ss_subject
    df1, df2 = k - 1, (n - 1) * (k - 1)
    F = (ss_effect / df1) / (ss_error / df2)
    S = np.cov(X, rowvar=False)
    Sc = S - S.mean(axis=0, keepdims=True) - S.mean(axis=1, keepdims=True) + S.mean()
    eps = (np.trace(Sc) ** 2) / ((k - 1) * (Sc ** 2).sum())
    return dict(n=n, df1=df1, df2=df2, SS_effect=ss_effect, SS_error=ss_error, SS_subject=ss_subject, F=F,
                p_unc=float(stats.f.sf(F, df1, df2)), ng2=ss_effect / (ss_effect + ss_error + ss_subject),
                eps=float(eps), p_gg=float(stats.f.sf(F, df1 * eps, df2 * eps)))
