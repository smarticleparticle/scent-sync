"""Odor-arrival timing (paper Section 1.8, SI Tables S5-S6).

Odor delivery is triggered on exhalation detection and needs ~1.1 s to travel through the tubing, so the question is whether the odor
had reached the nostrils by the time inhalation (0 degrees of the respiratory phase) began.  For every trial that passes the real ERSP
quality control, the lag between the exhalation trigger (the preceding trough in the CONTINUOUS respiration trace) and the detected
inhale onset is measured and compared with the transit time through the tubing.

Ported from analysis-notebook cells 40 (tubing transit time), 42 (per-trial timing), 44 (per-subject lag; its trial loop selects exactly the
same trials as cell 42) and 49 (sensitivity to flow-rate error).  ``segment_volume_mL``, ``transit_time_seconds``,
``find_preceding_trough_continuous`` and ``get_resp_phase_landmark_normalized`` are the notebook's functions, extracted from its source;
the trial loop is cell 42's, parameterised.
"""
from fractions import Fraction

import numpy as np
import pandas as pd
from scipy import signal

from .breath import bandpass_filter, is_bad_respiration_raw, sufficient_oscillations


def segment_volume_mL(inner_diameter_mm, length):
    """
    Volume of a cylindrical tube segment, in mL.
    inner_diameter_mm: inner diameter in mm
    length: length in meters
    """
    radius_m = (inner_diameter_mm / 1000) / 2
    volume_m3 = np.pi * radius_m**2 * length
    volume_mL = volume_m3 * 1e6
    return volume_mL


def transit_time_seconds(flow_rate_Lmin, segments):
    """
    Compute total transit time through one or more tubing segments in series.

    flow_rate_Lmin: volumetric flow rate, in L/min (assumed constant through all segments)
    segments: list of dicts, each with:
        'name': str, label for this segment
        'inner_diameter_mm': float
        'length_m': float

    Returns: dict with per-segment volumes/times and total transit time.
    """
    flow_mL_per_sec = flow_rate_Lmin * 1000 / 60

    results = []
    total_volume_mL = 0
    total_time_sec = 0

    for seg in segments:
        vol = segment_volume_mL(seg['inner_diameter_mm'], seg['length_m'])
        t = vol / flow_mL_per_sec
        results.append({
            'name': seg['name'],
            'inner_diameter_mm': seg['inner_diameter_mm'],
            'length_m': seg['length_m'],
            'volume_mL': vol,
            'transit_time_sec': t
        })
        total_volume_mL += vol
        total_time_sec += t

    return {
        'flow_rate_Lmin': flow_rate_Lmin,
        'flow_mL_per_sec': flow_mL_per_sec,
        'segments': results,
        'total_volume_mL': total_volume_mL,
        'total_transit_time_sec': total_time_sec
    }


def find_preceding_trough_continuous(resp_continuous, resp_sfreq, target_time_sec,
                                        lookback_sec=8.0, prominence_frac=0.05,
                                        min_dist_sec=1.0):
    target_idx = int(round(target_time_sec * resp_sfreq))
    lookback_samples = int(lookback_sec * resp_sfreq)
    search_start = max(0, target_idx - lookback_samples)
    if target_idx <= search_start:
        return None
    segment = resp_continuous[search_start:target_idx]
    local_range = np.nanmax(segment) - np.nanmin(segment)
    if local_range <= 0 or np.isnan(local_range):
        return None
    troughs, _ = signal.find_peaks(-segment, distance=int(min_dist_sec * resp_sfreq),
                                     prominence=prominence_frac * local_range)
    if len(troughs) == 0:
        return None
    return search_start + troughs[-1]


def get_resp_phase_landmark_normalized(
    resp_signal,
    fs,
    stimulus_idx=250,
    accept_window=(-0.4, 3.0),
    reject_window=(-1.0, -0.4),
    min_samples=500,  # 2s * 250Hz
    max_samples=2000,  # 8s * 250Hz
    verbose=False,
):
    """
    Find respiratory phase using landmark-based normalization.
    Returns phase_deg, breath_start_idx, breath_end_idx or (None, None, None) if bad.
    """
    peaks, _ = signal.find_peaks(resp_signal, distance=int(0.5 * fs), prominence=0.05)
    troughs, _ = signal.find_peaks(-resp_signal, distance=int(0.5 * fs), prominence=0.05)

    if len(peaks) < 2:
        return None, None, None

    accept_start_idx = stimulus_idx + int(accept_window[0] * fs)
    accept_end_idx = stimulus_idx + int(accept_window[1] * fs)
    reject_start_idx = stimulus_idx + int(reject_window[0] * fs)
    reject_end_idx = stimulus_idx + int(reject_window[1] * fs)

    peaks_in_accept = peaks[(peaks >= accept_start_idx) & (peaks <= accept_end_idx)]

    if len(peaks_in_accept) == 0:
        if len(peaks) > 0 and reject_start_idx <= peaks[0] < reject_end_idx:
            return None, None, None
        return None, None, None

    peak1_idx = peaks_in_accept[0]

    peaks_after_first = peaks[peaks > peak1_idx]
    if len(peaks_after_first) == 0:
        return None, None, None
    peak2_idx = peaks_after_first[0]

    breath_duration_samples = peak2_idx - peak1_idx
    if breath_duration_samples < min_samples or breath_duration_samples > max_samples:
        return None, None, None

    troughs_between = troughs[(troughs > peak1_idx) & (troughs < peak2_idx)]

    if len(troughs_between) == 0:
        search_segment = resp_signal[peak1_idx:peak2_idx]
        if len(search_segment) == 0:
            return None, None, None
        trough_idx = peak1_idx + np.argmin(search_segment)

        if verbose:
            print(f"    ⚠️  Fallback used: no trough detected, using minimum at idx {trough_idx}")
    else:
        trough_idx = troughs_between[0]
        if verbose:
            print(f"    ✓ Trough detected normally at idx {trough_idx}")

    inhale_length = trough_idx - peak1_idx
    exhale_length = peak2_idx - trough_idx

    if inhale_length <= 0 or exhale_length <= 0:
        return None, None, None

    phase_deg = np.full(len(resp_signal), np.nan)
    phase_deg[peak1_idx:trough_idx] = np.linspace(0, 180, inhale_length, endpoint=False)
    phase_deg[trough_idx:peak2_idx] = np.linspace(180, 360, exhale_length, endpoint=False)

    return phase_deg, peak1_idx, peak2_idx


def compute_trial_timing(epochs_v2, continuous_respiration, cfg, subjects=None):
    """One row per trial that passes the real QC (notebook cell 42).

    epochs_v2: dict with 'respiration' and 'timestamps' [condition][subject] and either 'EEG' (arrays) or 'eeg_shapes'
               ([condition][subject] -> (n_trials, n_channels, n_times)); only the EEG epoch LENGTH is used.
    continuous_respiration: {subject: continuous respiration trace} (full_respiration_EEG_v2['respiration']).
    """
    sfreq_eeg, sfreq_resp = cfg["sfreq_eeg"], cfg["sfreq_resp"]
    resp_band, resp_filter_order = tuple(cfg["resp_band"]), cfg["resp_filter_order"]
    accept_window, reject_window = tuple(cfg["accept_window"]), tuple(cfg["reject_window"])
    stimulus_idx = int(cfg["prestim_duration"] * sfreq_eeg)
    min_samples, max_samples = int(cfg["min_duration"] * sfreq_eeg), int(cfg["max_duration"] * sfreq_eeg)
    transit_time_measured = cfg["transit_time_measured"]
    ratio = Fraction(sfreq_eeg / sfreq_resp).limit_denominator(1000)
    up, down = ratio.numerator, ratio.denominator

    rows = []
    for condition in ['water', 'peppermint', 'oregano', 'citrus']:
        for subject in epochs_v2['respiration'][condition].keys():
            if subject not in continuous_respiration or (subjects is not None and subject not in subjects):
                continue
            resp_data = epochs_v2['respiration'][condition][subject]
            timestamps = epochs_v2['timestamps'][condition][subject]
            if 'EEG' in epochs_v2:
                n_trials, n_channels, n_times_eeg = epochs_v2['EEG'][condition][subject].shape
            else:
                n_trials, n_channels, n_times_eeg = epochs_v2['eeg_shapes'][condition][subject]
            resp_continuous = continuous_respiration[subject]

            for trial_idx in range(n_trials):
                resp_trial = np.squeeze(resp_data[trial_idx])
                stim_time_sec = timestamps[trial_idx]

                if is_bad_respiration_raw(resp_trial):
                    continue
                if not sufficient_oscillations(resp_trial, fs=sfreq_resp):
                    continue

                resp_filtered = bandpass_filter(resp_trial, resp_band[0], resp_band[1], sfreq_resp, resp_filter_order)
                resp_filtered = (resp_filtered - np.mean(resp_filtered)) / (np.std(resp_filtered) + 1e-12)

                resp_upsampled = signal.resample_poly(resp_filtered, up, down)
                if len(resp_upsampled) > n_times_eeg:
                    resp_upsampled = resp_upsampled[:n_times_eeg]
                else:
                    resp_upsampled = np.pad(resp_upsampled, (0, n_times_eeg - len(resp_upsampled)), mode='edge')

                phase_deg, peak1_idx, peak2_idx = get_resp_phase_landmark_normalized(
                    resp_upsampled, sfreq_eeg, stimulus_idx=stimulus_idx,
                    accept_window=accept_window, reject_window=reject_window,
                    min_samples=min_samples, max_samples=max_samples, verbose=False
                )
                if phase_deg is None:
                    continue

                offset_from_ttl_sec = (peak1_idx - stimulus_idx) / sfreq_eeg

                troughs_between = signal.find_peaks(-resp_upsampled, distance=int(0.5 * sfreq_eeg), prominence=0.05)[0]
                troughs_between = troughs_between[(troughs_between > peak1_idx) & (troughs_between < peak2_idx)]
                if len(troughs_between) == 0:
                    trough_idx = peak1_idx + np.argmin(resp_upsampled[peak1_idx:peak2_idx])
                else:
                    trough_idx = troughs_between[0]
                actual_inhale_duration_sec = (trough_idx - peak1_idx) / sfreq_eeg
                if actual_inhale_duration_sec <= 0:
                    continue

                absolute_peak1_time = stim_time_sec + (peak1_idx - stimulus_idx) / sfreq_eeg

                trough_idx_continuous = find_preceding_trough_continuous(resp_continuous, sfreq_resp, absolute_peak1_time)
                if trough_idx_continuous is None:
                    continue

                absolute_peak1_idx_continuous = int(round(absolute_peak1_time * sfreq_resp))
                lag_sec = (absolute_peak1_idx_continuous - trough_idx_continuous) / sfreq_resp
                if lag_sec <= 0:
                    continue

                arrival_relative = transit_time_measured - lag_sec
                if arrival_relative <= 0:
                    timing = 'arrived_before_real_onset'
                elif arrival_relative <= actual_inhale_duration_sec:
                    timing = 'arrived_during_real_inhale'
                else:
                    timing = 'arrived_after_real_inhale_ended'

                rows.append({'subject': subject, 'condition': condition, 'trial': trial_idx, 'lag_sec': lag_sec,
                             'offset_from_ttl_sec': offset_from_ttl_sec,
                             'actual_inhale_duration_sec': actual_inhale_duration_sec,
                             'arrival_relative': arrival_relative, 'timing': timing})
    return pd.DataFrame(rows)


def lag_table(df_timing):
    """SI Table S5: per-subject mean / SD of the exhale-to-inhale lag and number of trials (notebook cell 44)."""
    return (df_timing.groupby('subject')['lag_sec']
            .agg(mean_lag_sec='mean', sd_lag_sec='std', n_trials='count').round(3).sort_index())


def margin_summary(df_timing):
    """Odor-arrival margin before inhale onset (notebook cell 48): counts, median and 5th percentile of the passing trials."""
    margin = -df_timing['arrival_relative']
    good, bad = margin[margin >= 0], margin[margin < 0]
    return {"n_trials": len(df_timing), "n_arrived_in_time": int(len(good)), "n_late": int(len(bad)),
            "pct_in_time": 100 * len(good) / len(df_timing), "median_margin_s": float(good.median()),
            "p5_margin_s": float(np.percentile(good, 5)), "min_margin_s": float(good.min()),
            "late_delays_s": sorted((-bad).round(4).tolist())}


def flow_sensitivity_table(df_timing, flow_rate_Lmin, segments, pct_errors):
    """SI Table S6: how many trials would be 'late' if the measured flow rate were off by a given percentage (notebook cell 49)."""
    rows = []
    for pct_err in pct_errors:
        fr = flow_rate_Lmin * (1 + pct_err / 100)
        tt = transit_time_seconds(fr, segments)['total_transit_time_sec']
        arrival_relative_sens = tt - df_timing['lag_sec']
        n_contaminated = int((arrival_relative_sens > 0).sum())
        rows.append({'pct_error_in_flow_rate': pct_err, 'assumed_flow_rate_Lmin': round(fr, 4),
                     'transit_time_sec': round(tt, 4), 'n_contaminated': n_contaminated,
                     'pct_odor_present_at_onset': round(100 * (1 - n_contaminated / len(df_timing)), 2)})
    return pd.DataFrame(rows)
