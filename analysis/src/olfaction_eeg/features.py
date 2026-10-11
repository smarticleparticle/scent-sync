"""Feature extraction: bin ERSP arrays into (frequency band x phase/time bin) features.

(Formerly format.py; renamed so it no longer shadows Python's builtin `format`.)
"""
import pandas as pd
import numpy as np

def extract_binned_features(
    results,
    cond_a,
    cond_b,
    used_subjects,
    freqs,
    x_bins,
    x,
    trial_start=0,
    trial_end=4,
    # n_early=4,
    # phase_edges_deg=None,
    freq_bands=None,
    respiratory_features = None
):
    """
    Create dataframe with ONLY EEG features.
    Uses first n_early trials averaged per subject.
    """
    if freq_bands is None:
        freq_bands = {
            "theta": (4, 8),
            "alpha": (8, 12),
            "beta": (12, 30),
        }

    def extract_subject_features(condition):
        rows = []

        for subj in used_subjects:
            if subj not in results[condition]:
                continue

            trials = results[condition][subj]
            trial_keys = sorted(trials.keys())

            # if len(trial_keys) < n_early:
            #     continue

            # Get first n_early trials
            early_keys = trial_keys[trial_start:trial_end]
            if len(early_keys) == 0:
                continue

            # Average EEG data across early trials
            trial_list = []
            for tk in early_keys:
                trial_data = trials[tk]
                trial_list.append(trial_data)

            if len(trial_list) == 0:
                continue

            avg_data = np.mean(np.stack(trial_list, axis=0), axis=0)

            # Extract EEG features
            features = {}
            for freq_name, (f_low, f_high) in freq_bands.items():
                freq_mask = (freqs >= f_low) & (freqs <= f_high)

                for bin_name, (bin_low, bin_high) in x_bins.items():
                    bin_mask = (x >= bin_low) & (x < bin_high)
                    roi_power = avg_data[freq_mask, :][:, bin_mask]
                    features[f"{freq_name}_{bin_name}"] = np.nanmean(roi_power)
                    
            if respiratory_features is not None:
                if subj in respiratory_features[condition]:
                    resp_trials = respiratory_features[condition][subj]
                    
                    # Average respiratory features across early trials
                    resp_values = {}
                    for tk in early_keys:
                        if tk in resp_trials:
                            for feat_name, feat_val in resp_trials[tk].items():
                                # All features in respiratory_features are already filtered
                                # to only include duration_inhale1
                                if feat_name not in resp_values:
                                    resp_values[feat_name] = []
                                resp_values[feat_name].append(feat_val)
                    
                    for feat_name, values in resp_values.items():
                        features[f"resp_{feat_name}"] = np.nanmean(values)

            features["subject"] = subj
            rows.append(features)

        return pd.DataFrame(rows)

    # Extract for both conditions
    df_a = extract_subject_features(cond_a)
    df_b = extract_subject_features(cond_b)

    if len(df_a) == 0 or len(df_b) == 0:
        return None

    # Keep only common subjects
    common = sorted(set(df_a["subject"]).intersection(set(df_b["subject"])))
    df_a = df_a[df_a["subject"].isin(common)].copy()
    df_b = df_b[df_b["subject"].isin(common)].copy()

    df_a["label"] = 0
    df_b["label"] = 1

    df = pd.concat([df_a, df_b], axis=0).reset_index(drop=True)

    return df
