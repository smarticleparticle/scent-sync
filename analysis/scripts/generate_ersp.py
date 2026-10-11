import numpy as np
import pywt
import matplotlib.pyplot as plt
from fractions import Fraction
from scipy.signal import butter, filtfilt, resample_poly, hilbert, find_peaks
from itertools import groupby
from operator import itemgetter
import os
import mne
import fire
import io
import sys
import contextlib
import multiprocessing
import concurrent.futures

# Parameters
sfreq_eeg = 250
sfreq_resp = 8.75714285714
resp_band = (0.1, 0.4)
resp_filter_order = 2
wavelet = "cmor1.0-1.0"
n_freqs = 100
freqs = np.geomspace(2, 40, n_freqs)
scales = pywt.frequency2scale(wavelet, freqs / sfreq_eeg)

# Phase bins - FULL BREATH (0-360°) with 1-degree bins
phase_step_deg = 1
phase_bin_edges = np.arange(0, 360 + phase_step_deg, phase_step_deg)
n_phase_bins = len(phase_bin_edges) - 1
phase_bin_centers = (phase_bin_edges[:-1] + phase_bin_edges[1:]) / 2

# BASELINE PERIOD BEFORE BREATH CYCLE
baseline_duration = 0.5  # 0.5 seconds before 0° (inhale onset)
baseline_samples = int(baseline_duration * sfreq_eeg)

# Sniff-triggered ERSP parameters
ersp_start = -0.5
ersp_end = 3.78
ersp_samples_start = int(ersp_start * sfreq_eeg)
ersp_samples_end = int(ersp_end * sfreq_eeg)
times = np.arange(ersp_samples_start, ersp_samples_end) / sfreq_eeg
n_time_points = ersp_samples_end - ersp_samples_start
_DEFAULT_ERSP_END = ersp_end          # remembered so that run(time_window_end=None) always restores the default window

# Resampling ratio
ratio = Fraction(sfreq_eeg / sfreq_resp).limit_denominator(1000)
up, down = ratio.numerator, ratio.denominator

# Minimum and maximum samples for FULL breath (2-8 seconds)
min_samples = int(2.0 * sfreq_eeg)
max_samples = int(8.0 * sfreq_eeg)

# Timing windows for peak selection relative to stimulus
accept_window = (-0.4, 3.0)
reject_window = (-1.0, -0.4)

# CRITICAL: Stimulus timing in epoched data
prestim_duration = 1.0
stimulus_idx = int(prestim_duration * sfreq_eeg)

# Defaults only; the Snakemake workflow passes explicit paths (see run()).
data_path = "data"
info_dir = "data/preprocessed/info"

roi_channels = [
    "E8", "E9", "E10", "E1", "E2", "E3", "E4", "E123", "E124", "E11", "E14",
    "E15", "E16", "E118", "E5", "E17", "E12", "E19", "E18", "E22", "E21",
    "E117", "E33", "E34", "E38", "E39", "E40", "E43", "E44", "E45", "E46",
    "E49", "E50", "E51", "E56", "E57", "E58", "E63", "E96", "E97", "E99",
    "E100", "E101", "E102", "E108", "E109", "E107", "E113", "E114", "E115",
    "E116", "E120", "E121", "E122",
]


# --------------------------------------------------------------------------
# Function definitions
# --------------------------------------------------------------------------
def bandpass_filter(data, lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    return filtfilt(b, a, data)


def is_bad_respiration_raw(resp_signal):
    return np.std(resp_signal) < 1e-3 or np.isnan(resp_signal).all()


def sufficient_oscillations(resp_signal, fs, min_cycles=3):
    inhale_peaks, _ = find_peaks(resp_signal, distance=int(0.5 * fs), prominence=0.05)
    return len(inhale_peaks) >= min_cycles


def get_resp_phase_landmark_normalized(
    resp_signal,
    fs,
    stimulus_idx=250,
    accept_window=(-0.4, 3.0),
    reject_window=(-1.0, -0.4),
    verbose=False,
):
    from scipy.signal import find_peaks

    peaks, _ = find_peaks(resp_signal, distance=int(0.5 * fs), prominence=0.05)
    troughs, _ = find_peaks(-resp_signal, distance=int(0.5 * fs), prominence=0.05)

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
            print(
                f"    ⚠️  Fallback used: no trough detected, using minimum at idx {trough_idx}"
            )
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


def is_bad_respiration(resp_signal):
    return np.std(resp_signal) < 1e-3 or np.isnan(resp_signal).any()


def find_inhalation_onset(resp_signal, fs):
    from scipy.signal import find_peaks

    peaks, _ = find_peaks(resp_signal, distance=int(0.5 * fs), prominence=0.05)

    if len(peaks) == 0:
        return None

    return peaks[0]


def compute_subject_condition_phase_aligned_with_baseline(
    eeg_data,
    resp_data,
    subj_id,
    cond_name,
    trial_counts_global,
    roi_indices,
    shared_roi_channels=None,
    save_channels=False,
):
    """
    Process all trials for one subject in one condition.
    Analyzes FULL BREATH CYCLES (0-360°) AND saves a pre-breath baseline period.

    NEW: Returns both phase-aligned data AND baseline period for normalization

    Parameters:
    -----------
    save_channels : bool
        If True, returns dict with per-channel data
        If False, returns dict with channel-averaged data

    Returns:
    --------
    dict with keys:
        'phase_data': phase-binned ERSP (as before)
        'baseline_data': baseline ERSP from -0.5s before breath onset
    """
    n_trials, n_channels, n_times_eeg = eeg_data.shape

    if save_channels:
        per_channel_results_phase = {}
        per_channel_results_baseline = {}
    else:
        per_trial_results_phase = {}
        per_trial_results_baseline = {}

    n_skipped = 0
    n_valid = 0

    if not roi_indices:
        print(f"    Subject {subj_id} has no shared ROI channels. Skipping.")
        return None

    for trial in range(n_trials):
        resp_trial = np.squeeze(resp_data[trial])

        if is_bad_respiration_raw(resp_trial):
            n_skipped += 1
            continue
        if not sufficient_oscillations(resp_trial, fs=sfreq_resp):
            n_skipped += 1
            continue

        resp_filtered = bandpass_filter(
            resp_trial, resp_band[0], resp_band[1], sfreq_resp, resp_filter_order
        )
        resp_filtered = (resp_filtered - np.mean(resp_filtered)) / (
            np.std(resp_filtered) + 1e-12
        )

        resp_upsampled = resample_poly(resp_filtered, up, down)
        if len(resp_upsampled) > n_times_eeg:
            resp_upsampled = resp_upsampled[:n_times_eeg]
        else:
            resp_upsampled = np.pad(
                resp_upsampled, (0, n_times_eeg - len(resp_upsampled)), mode="edge"
            )

        phase_deg, breath_start_idx, breath_end_idx = (
            get_resp_phase_landmark_normalized(
                resp_upsampled,
                sfreq_eeg,
                stimulus_idx=stimulus_idx,
                accept_window=accept_window,
                reject_window=reject_window,
                verbose=(trial == 0),
            )
        )
        if phase_deg is None:
            n_skipped += 1
            continue

        # Check if we have enough data for baseline
        baseline_start_idx = breath_start_idx - baseline_samples
        if baseline_start_idx < 0:
            n_skipped += 1
            continue

        segment_indices = np.arange(breath_start_idx, breath_end_idx)
        phase_segment = phase_deg[segment_indices]

        # Baseline indices
        baseline_indices = np.arange(baseline_start_idx, breath_start_idx)

        pad_len = 250

        if save_channels:
            # Save per-channel data
            for idx, ch_idx in enumerate(roi_indices):
                eeg_ch_data = eeg_data[trial, ch_idx, :]
                padded = np.pad(eeg_ch_data, (pad_len, pad_len), mode="reflect")
                cwtmatr, _ = pywt.cwt(
                    padded, scales, wavelet, sampling_period=1 / sfreq_eeg
                )
                cwtmatr = cwtmatr[:, pad_len:-pad_len]
                
                # PHASE-BINNED DATA
                channel_power = np.abs(cwtmatr[:, segment_indices]) ** 2
                channel_power = np.log10(channel_power + 1e-20)

                binned_power = np.full((n_freqs, n_phase_bins), np.nan, dtype=float)
                for bin_idx in range(n_phase_bins):
                    bin_start = phase_bin_edges[bin_idx]
                    bin_end = phase_bin_edges[bin_idx + 1]
                    mask = (phase_segment >= bin_start) & (phase_segment < bin_end)
                    if np.any(mask):
                        binned_power[:, bin_idx] = np.mean(
                            channel_power[:, mask], axis=1
                        )

                # BASELINE DATA (average over time)
                baseline_power = np.abs(cwtmatr[:, baseline_indices]) ** 2
                baseline_power = np.log10(baseline_power + 1e-20)
                baseline_avg = np.mean(baseline_power, axis=1)  # Average across time

                # Get actual channel name
                if shared_roi_channels is not None:
                    channel_name = shared_roi_channels[idx]
                else:
                    channel_name = f"ch_{ch_idx}"

                if channel_name not in per_channel_results_phase:
                    per_channel_results_phase[channel_name] = {}
                    per_channel_results_baseline[channel_name] = {}

                per_channel_results_phase[channel_name][trial] = binned_power
                per_channel_results_baseline[channel_name][trial] = baseline_avg

        else:
            # Average across ROI channels
            trial_power = np.zeros((n_freqs, len(segment_indices)))
            trial_baseline_power = np.zeros((n_freqs, len(baseline_indices)))
            
            for ch_idx in roi_indices:
                eeg_ch_data = eeg_data[trial, ch_idx, :]
                padded = np.pad(eeg_ch_data, (pad_len, pad_len), mode="reflect")
                cwtmatr, _ = pywt.cwt(
                    padded, scales, wavelet, sampling_period=1 / sfreq_eeg
                )
                cwtmatr = cwtmatr[:, pad_len:-pad_len]
                
                trial_power += np.abs(cwtmatr[:, segment_indices]) ** 2
                trial_baseline_power += np.abs(cwtmatr[:, baseline_indices]) ** 2

            trial_power /= len(roi_indices)
            trial_power = np.log10(trial_power + 1e-20)

            trial_baseline_power /= len(roi_indices)
            trial_baseline_power = np.log10(trial_baseline_power + 1e-20)

            # Bin phase data
            binned_power = np.full((n_freqs, n_phase_bins), np.nan, dtype=float)
            for bin_idx in range(n_phase_bins):
                bin_start = phase_bin_edges[bin_idx]
                bin_end = phase_bin_edges[bin_idx + 1]
                mask = (phase_segment >= bin_start) & (phase_segment < bin_end)
                if np.any(mask):
                    binned_power[:, bin_idx] = np.mean(trial_power[:, mask], axis=1)

            # Average baseline across time
            baseline_avg = np.mean(trial_baseline_power, axis=1)

            per_trial_results_phase[trial] = binned_power
            per_trial_results_baseline[trial] = baseline_avg

        n_valid += 1

    trial_counts_global[(cond_name, subj_id)] = {"valid": n_valid, "skipped": n_skipped}

    if save_channels:
        if per_channel_results_phase:
            return {
                'phase_data': per_channel_results_phase,
                'baseline_data': per_channel_results_baseline
            }
        else:
            return None
    else:
        if per_trial_results_phase:
            return {
                'phase_data': per_trial_results_phase,
                'baseline_data': per_trial_results_baseline
            }
        else:
            return None


def compute_sniff_triggered_ersp(
    eeg_data,
    resp_data,
    subj_id,
    cond_name,
    trial_counts_global,
    roi_indices,
    resp_band=(0.1, 0.4),
    shared_roi_channels=None,
    save_channels=False,
):
    """
    Compute ERSP time-locked to INHALATION ONSET.
    (Keep original function unchanged)
    """
    n_trials, n_channels, n_samples_eeg = eeg_data.shape

    if save_channels:
        per_channel_results = {}
    else:
        per_trial_results = {}

    n_skipped = 0
    n_valid = 0

    if not roi_indices:
        return None

    for trial in range(n_trials):
        resp_trial = np.squeeze(resp_data[trial])

        if is_bad_respiration(resp_trial):
            n_skipped += 1
            continue

        resp_filtered = bandpass_filter(
            resp_trial, resp_band[0], resp_band[1], sfreq_resp, order=2
        )
        resp_filtered = (resp_filtered - np.mean(resp_filtered)) / (
            np.std(resp_filtered) + 1e-12
        )

        resp_upsampled = resample_poly(resp_filtered, up, down)
        if len(resp_upsampled) > n_samples_eeg:
            resp_upsampled = resp_upsampled[:n_samples_eeg]
        else:
            resp_upsampled = np.pad(
                resp_upsampled, (0, n_samples_eeg - len(resp_upsampled)), mode="edge"
            )

        inhale_onset_sample = find_inhalation_onset(resp_upsampled, sfreq_eeg)
        if inhale_onset_sample is None:
            n_skipped += 1
            continue

        extract_start = inhale_onset_sample + ersp_samples_start
        extract_end = inhale_onset_sample + ersp_samples_end

        if extract_start < 0 or extract_end > n_samples_eeg:
            n_skipped += 1
            continue

        pad_len = 250

        if save_channels:
            for idx, ch_idx in enumerate(roi_indices):
                eeg_ch_data = eeg_data[trial, ch_idx, :]
                padded = np.pad(eeg_ch_data, (pad_len, pad_len), mode="reflect")
                cwtmatr, _ = pywt.cwt(
                    padded, scales, wavelet, sampling_period=1 / sfreq_eeg
                )
                cwtmatr = cwtmatr[:, pad_len:-pad_len]

                power_full = np.abs(cwtmatr) ** 2
                power_window = power_full[:, extract_start:extract_end]
                ersp = np.log10(power_window + 1e-20)

                if np.any(~np.isfinite(ersp)):
                    continue

                if shared_roi_channels is not None:
                    channel_name = shared_roi_channels[idx]
                else:
                    channel_name = f"ch_{ch_idx}"

                if channel_name not in per_channel_results:
                    per_channel_results[channel_name] = {}

                per_channel_results[channel_name][trial] = ersp
        else:
            trial_power = np.zeros((n_freqs, n_time_points))
            for ch_idx in roi_indices:
                eeg_ch_data = eeg_data[trial, ch_idx, :]
                padded = np.pad(eeg_ch_data, (pad_len, pad_len), mode="reflect")
                cwtmatr, _ = pywt.cwt(
                    padded, scales, wavelet, sampling_period=1 / sfreq_eeg
                )
                cwtmatr = cwtmatr[:, pad_len:-pad_len]

                power_full = np.abs(cwtmatr) ** 2
                power_window = power_full[:, extract_start:extract_end]
                trial_power += power_window

            trial_power /= len(roi_indices)
            ersp = np.log10(trial_power + 1e-20)

            if np.any(~np.isfinite(ersp)):
                n_skipped += 1
                continue

            per_trial_results[trial] = ersp

        n_valid += 1

    trial_counts_global[(cond_name, subj_id)] = {"valid": n_valid, "skipped": n_skipped}

    if save_channels:
        return per_channel_results if per_channel_results else None
    else:
        return per_trial_results if per_trial_results else None
    
def compute_subject_condition_phase_aligned(
    eeg_data, resp_data, subj_id, cond_name, trial_counts_global, roi_indices,
    shared_roi_channels=None, save_channels=False
):
    n_trials, n_channels, n_times_eeg = eeg_data.shape
    per_trial_results = {}
    n_skipped = 0
    n_valid = 0

    # NEW: track skip reasons
    skip_reasons = {'bad_resp_raw': 0, 'insufficient_osc': 0, 'phase_none': 0}

    if not roi_indices:
        print(f"    Subject {subj_id} has no shared ROI channels. Skipping.")
        return None

    for trial in range(n_trials):
        resp_trial = np.squeeze(resp_data[trial])

        if is_bad_respiration_raw(resp_trial):
            n_skipped += 1
            skip_reasons['bad_resp_raw'] += 1
            continue
        if not sufficient_oscillations(resp_trial, fs=sfreq_resp):
            n_skipped += 1
            skip_reasons['insufficient_osc'] += 1
            continue

        resp_filtered = bandpass_filter(
            resp_trial, resp_band[0], resp_band[1], sfreq_resp, resp_filter_order
        )
        resp_filtered = (resp_filtered - np.mean(resp_filtered)) / (
            np.std(resp_filtered) + 1e-12
        )

        resp_upsampled = resample_poly(resp_filtered, up, down)
        if len(resp_upsampled) > n_times_eeg:
            resp_upsampled = resp_upsampled[:n_times_eeg]
        else:
            resp_upsampled = np.pad(
                resp_upsampled, (0, n_times_eeg - len(resp_upsampled)), mode="edge"
            )

        phase_deg, breath_start_idx, breath_end_idx = (
            get_resp_phase_landmark_normalized(
                resp_upsampled,
                sfreq_eeg,
                stimulus_idx=stimulus_idx,
                accept_window=accept_window,
                reject_window=reject_window,
                verbose=(trial == 0),
            )
        )
        if phase_deg is None:
            n_skipped += 1
            skip_reasons['phase_none'] += 1
            continue

        segment_indices = np.arange(breath_start_idx, breath_end_idx)
        phase_segment = phase_deg[segment_indices]

        trial_power = np.zeros((n_freqs, len(segment_indices)))
        pad_len = 250

        for ch_idx in roi_indices:
            eeg_ch_data = eeg_data[trial, ch_idx, :]
            padded = np.pad(eeg_ch_data, (pad_len, pad_len), mode="reflect")
            cwtmatr, _ = pywt.cwt(padded, scales, wavelet, sampling_period=1 / sfreq_eeg)
            cwtmatr = cwtmatr[:, pad_len:-pad_len]
            trial_power += np.abs(cwtmatr[:, segment_indices]) ** 2

        trial_power /= len(roi_indices)
        trial_power = np.log10(trial_power + 1e-20)

        binned_power = np.full((n_freqs, n_phase_bins), np.nan, dtype=float)
        for bin_idx in range(n_phase_bins):
            bin_start = phase_bin_edges[bin_idx]
            bin_end = phase_bin_edges[bin_idx + 1]
            mask = (phase_segment >= bin_start) & (phase_segment < bin_end)
            if np.any(mask):
                binned_power[:, bin_idx] = np.mean(trial_power[:, mask], axis=1)

        per_trial_results[trial] = binned_power
        n_valid += 1

    # NEW: print skip breakdown
    print(f"    Skip breakdown for {subj_id}/{cond_name}: {skip_reasons} (total trials: {n_trials})")

    trial_counts_global[(cond_name, subj_id)] = {"valid": n_valid, "skipped": n_skipped}

    if not per_trial_results:
        return None
    return per_trial_results

def compute_subject_condition_phase_aligned_with_baseline_for_MI(
    eeg_data,
    resp_data,
    subj_id,
    cond_name,
    trial_counts_global,
    roi_indices,
    shared_roi_channels=None,
    save_channels=False,
):
    """
    Process all trials for one subject in one condition.
    
    NEW: Saves RAW EEG and RESPIRATION from baseline period for MI null testing.
    
    Returns:
    --------
    dict with keys:
        'phase_data': phase-binned ERSP (task period)
        'baseline_data': baseline ERSP (averaged across time)
        'baseline_eeg_raw': RAW EEG during baseline (for MI calculation)
        'baseline_resp_raw': RAW respiration during baseline (for MI calculation)
        'task_eeg_raw': RAW EEG during task breath cycle
        'task_resp_raw': RAW respiration during task breath cycle
    """
    n_trials, n_channels, n_times_eeg = eeg_data.shape

    if save_channels:
        per_channel_results_phase = {}
        per_channel_results_baseline = {}
        per_channel_results_baseline_eeg_raw = {}
        per_channel_results_baseline_resp_raw = {}
        per_channel_results_task_eeg_raw = {}
        per_channel_results_task_resp_raw = {}
    else:
        per_trial_results_phase = {}
        per_trial_results_baseline = {}
        per_trial_results_baseline_eeg_raw = {}
        per_trial_results_baseline_resp_raw = {}
        per_trial_results_task_eeg_raw = {}
        per_trial_results_task_resp_raw = {}

    n_skipped = 0
    n_valid = 0

    if not roi_indices:
        print(f"    Subject {subj_id} has no shared ROI channels. Skipping.")
        return None

    for trial in range(n_trials):
        resp_trial = np.squeeze(resp_data[trial])

        if is_bad_respiration_raw(resp_trial):
            n_skipped += 1
            continue
        if not sufficient_oscillations(resp_trial, fs=sfreq_resp):
            n_skipped += 1
            continue

        # Filter respiration
        resp_filtered = bandpass_filter(
            resp_trial, resp_band[0], resp_band[1], sfreq_resp, resp_filter_order
        )
        resp_filtered = (resp_filtered - np.mean(resp_filtered)) / (
            np.std(resp_filtered) + 1e-12
        )

        # Upsample to EEG rate
        resp_upsampled = resample_poly(resp_filtered, up, down)
        if len(resp_upsampled) > n_times_eeg:
            resp_upsampled = resp_upsampled[:n_times_eeg]
        else:
            resp_upsampled = np.pad(
                resp_upsampled, (0, n_times_eeg - len(resp_upsampled)), mode="edge"
            )

        # Get phase
        phase_deg, breath_start_idx, breath_end_idx = (
            get_resp_phase_landmark_normalized(
                resp_upsampled,
                sfreq_eeg,
                stimulus_idx=stimulus_idx,
                accept_window=accept_window,
                reject_window=reject_window,
                verbose=(trial == 0),
            )
        )
        if phase_deg is None:
            n_skipped += 1
            continue

        # Check if we have enough data for baseline
        baseline_start_idx = breath_start_idx - baseline_samples
        if baseline_start_idx < 0:
            n_skipped += 1
            continue

        # Define segments
        segment_indices = np.arange(breath_start_idx, breath_end_idx)
        phase_segment = phase_deg[segment_indices]
        baseline_indices = np.arange(baseline_start_idx, breath_start_idx)

        # Extract RAW respiration signals
        task_resp_raw = resp_upsampled[segment_indices]  # RAW resp during task
        baseline_resp_raw = resp_upsampled[baseline_indices]  # RAW resp during baseline

        pad_len = 250

        if save_channels:
            # Save per-channel data
            for idx, ch_idx in enumerate(roi_indices):
                eeg_ch_data = eeg_data[trial, ch_idx, :]
                
                # Extract RAW EEG (no wavelet yet)
                task_eeg_raw = eeg_ch_data[segment_indices]
                baseline_eeg_raw = eeg_ch_data[baseline_indices]
                
                # Compute wavelets for ERSP
                padded = np.pad(eeg_ch_data, (pad_len, pad_len), mode="reflect")
                cwtmatr, _ = pywt.cwt(
                    padded, scales, wavelet, sampling_period=1 / sfreq_eeg
                )
                cwtmatr = cwtmatr[:, pad_len:-pad_len]
                
                # PHASE-BINNED DATA (task)
                channel_power = np.abs(cwtmatr[:, segment_indices]) ** 2
                channel_power = np.log10(channel_power + 1e-20)

                binned_power = np.full((n_freqs, n_phase_bins), np.nan, dtype=float)
                for bin_idx in range(n_phase_bins):
                    bin_start = phase_bin_edges[bin_idx]
                    bin_end = phase_bin_edges[bin_idx + 1]
                    mask = (phase_segment >= bin_start) & (phase_segment < bin_end)
                    if np.any(mask):
                        binned_power[:, bin_idx] = np.mean(
                            channel_power[:, mask], axis=1
                        )

                # BASELINE ERSP (averaged across time)
                baseline_power = np.abs(cwtmatr[:, baseline_indices]) ** 2
                baseline_power = np.log10(baseline_power + 1e-20)
                baseline_avg = np.mean(baseline_power, axis=1)

                # Get channel name
                if shared_roi_channels is not None:
                    channel_name = shared_roi_channels[idx]
                else:
                    channel_name = f"ch_{ch_idx}"

                if channel_name not in per_channel_results_phase:
                    per_channel_results_phase[channel_name] = {}
                    per_channel_results_baseline[channel_name] = {}
                    per_channel_results_baseline_eeg_raw[channel_name] = {}
                    per_channel_results_baseline_resp_raw[channel_name] = {}
                    per_channel_results_task_eeg_raw[channel_name] = {}
                    per_channel_results_task_resp_raw[channel_name] = {}

                per_channel_results_phase[channel_name][trial] = binned_power
                per_channel_results_baseline[channel_name][trial] = baseline_avg
                per_channel_results_baseline_eeg_raw[channel_name][trial] = baseline_eeg_raw
                per_channel_results_baseline_resp_raw[channel_name][trial] = baseline_resp_raw
                per_channel_results_task_eeg_raw[channel_name][trial] = task_eeg_raw
                per_channel_results_task_resp_raw[channel_name][trial] = task_resp_raw

        else:
            # Average across ROI channels
            trial_power = np.zeros((n_freqs, len(segment_indices)))
            trial_baseline_power = np.zeros((n_freqs, len(baseline_indices)))
            trial_task_eeg_raw = np.zeros(len(segment_indices))
            trial_baseline_eeg_raw = np.zeros(len(baseline_indices))
            
            for ch_idx in roi_indices:
                eeg_ch_data = eeg_data[trial, ch_idx, :]
                
                # Accumulate RAW EEG
                trial_task_eeg_raw += eeg_ch_data[segment_indices]
                trial_baseline_eeg_raw += eeg_ch_data[baseline_indices]
                
                # Compute wavelets
                padded = np.pad(eeg_ch_data, (pad_len, pad_len), mode="reflect")
                cwtmatr, _ = pywt.cwt(
                    padded, scales, wavelet, sampling_period=1 / sfreq_eeg
                )
                cwtmatr = cwtmatr[:, pad_len:-pad_len]
                
                trial_power += np.abs(cwtmatr[:, segment_indices]) ** 2
                trial_baseline_power += np.abs(cwtmatr[:, baseline_indices]) ** 2

            # Average across channels
            trial_power /= len(roi_indices)
            trial_power = np.log10(trial_power + 1e-20)

            trial_baseline_power /= len(roi_indices)
            trial_baseline_power = np.log10(trial_baseline_power + 1e-20)
            
            trial_task_eeg_raw /= len(roi_indices)
            trial_baseline_eeg_raw /= len(roi_indices)

            # Bin phase data
            binned_power = np.full((n_freqs, n_phase_bins), np.nan, dtype=float)
            for bin_idx in range(n_phase_bins):
                bin_start = phase_bin_edges[bin_idx]
                bin_end = phase_bin_edges[bin_idx + 1]
                mask = (phase_segment >= bin_start) & (phase_segment < bin_end)
                if np.any(mask):
                    binned_power[:, bin_idx] = np.mean(trial_power[:, mask], axis=1)

            # Average baseline across time
            baseline_avg = np.mean(trial_baseline_power, axis=1)

            per_trial_results_phase[trial] = binned_power
            per_trial_results_baseline[trial] = baseline_avg
            per_trial_results_baseline_eeg_raw[trial] = trial_baseline_eeg_raw
            per_trial_results_baseline_resp_raw[trial] = baseline_resp_raw  # Same for all channels
            per_trial_results_task_eeg_raw[trial] = trial_task_eeg_raw
            per_trial_results_task_resp_raw[trial] = task_resp_raw  # Same for all channels

        n_valid += 1

    trial_counts_global[(cond_name, subj_id)] = {"valid": n_valid, "skipped": n_skipped}

    if save_channels:
        if per_channel_results_phase:
            return {
                'phase_data': per_channel_results_phase,
                'baseline_data': per_channel_results_baseline,
                'baseline_eeg_raw': per_channel_results_baseline_eeg_raw,
                'baseline_resp_raw': per_channel_results_baseline_resp_raw,
                'task_eeg_raw': per_channel_results_task_eeg_raw,
                'task_resp_raw': per_channel_results_task_resp_raw
            }
        else:
            return None
    else:
        if per_trial_results_phase:
            return {
                'phase_data': per_trial_results_phase,
                'baseline_data': per_trial_results_baseline,
                'baseline_eeg_raw': per_trial_results_baseline_eeg_raw,
                'baseline_resp_raw': per_trial_results_baseline_resp_raw,
                'task_eeg_raw': per_trial_results_task_eeg_raw,
                'task_resp_raw': per_trial_results_task_resp_raw
            }
        else:
            return None
        



def _apply_time_window(time_window_end):
    """Set the end of the sniff-triggered window (module globals) -- EVERY call sets it explicitly (None = the 3.78 s default),
    so repeated run() calls in one process cannot inherit an earlier call's window.  Also re-applied inside worker processes."""
    global ersp_end, ersp_samples_end, times, n_time_points
    ersp_end = _DEFAULT_ERSP_END if time_window_end is None else float(time_window_end)
    ersp_samples_end = int(ersp_end * sfreq_eeg)
    times = np.arange(ersp_samples_start, ersp_samples_end) / sfreq_eeg
    n_time_points = ersp_samples_end - ersp_samples_start


def _compute_one(task, capture=False):
    """One (condition, subject) block.  Returns (results, trial_counts, log_text); log_text is only filled when capture=True."""
    _apply_time_window(task["time_window_end"])
    buf = io.StringIO()
    counts = {}
    with (contextlib.redirect_stdout(buf) if capture else contextlib.nullcontext()):
        print(f"\n=== condition {task['cond']} / subject {task['subj']} ===")
        print(f"    Using {len(task['shared'])} ROI channels")
        fn = {"time": compute_sniff_triggered_ersp,
              "phase": compute_subject_condition_phase_aligned,
              "phase_baseline": compute_subject_condition_phase_aligned_with_baseline,
              "phase_baseline_mi": compute_subject_condition_phase_aligned_with_baseline_for_MI}[task["kind"]]
        results = fn(task["eeg"], task["resp"], task["subj"], task["cond"], counts, task["roi"],
                     shared_roi_channels=task["shared"], save_channels=task["save_channels"])
    return results, counts, buf.getvalue()


def _run_tasks(tasks, n_jobs):
    n_jobs = max(1, int(n_jobs))
    if n_jobs == 1 or len(tasks) <= 1:
        return [_compute_one(t, capture=False) for t in tasks]
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, "1")           # one thread per worker: the parallelism is across blocks
    print(f"Computing {len(tasks)} subject x condition blocks with {n_jobs} worker processes", flush=True)
    results = [None] * len(tasks)
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs, mp_context=multiprocessing.get_context("spawn")) as ex:
        futures = {ex.submit(_compute_one, t, True): i for i, t in enumerate(tasks)}
        for done, fut in enumerate(concurrent.futures.as_completed(futures), start=1):
            i = futures[fut]
            results[i] = fut.result()                 # re-raises a worker's exception
            print(f"[{done}/{len(tasks)}] finished {tasks[i]['cond']} / subject {tasks[i]['subj']}", flush=True)
    return results


def run(phase_aligned=True, save_channels=False, with_baseline=False, baseline_for_mi=False,
        epochs_file=None, info_dir=info_dir, out_file=None, exclude_subjects=("30",),
        time_window_end=None, n_jobs=1):
    """
    Parameters:
    -----------
    epochs_file : str, optional
        Path to the preprocessed epochs .npy ({'EEG': {cond: {subj: ...}}, 'respiration': ...}).
        Default: <data_path>/epochs_eeg_respiration_withtimestamps_v2.npy
    info_dir : str
        Directory containing subject<ID>_info.fif files (used for channel names).
    out_file : str, optional
        Where to save the result. Default: legacy file name inside <data_path>.
    exclude_subjects : iterable
        Subject IDs skipped entirely (default: subject 30, bad respiration data).
    n_jobs : int
        Number of worker processes; each (condition, subject) block is independent, so the result is identical for any value.
        Default 1 = serial.  Memory: each worker holds one block (about 50 MB) plus wavelet buffers.
    time_window_end : float, optional
        End of the sniff-triggered (time-locked) ERSP window in seconds after inhale onset
        (start stays at -0.5 s).  Default = module constant ersp_end (3.78 s, the average first-breath
        duration).  Use 2.0 for the "traditional" 2-s time-locked ERSP.  Only affects phase_aligned=False.
    phase_aligned : bool
        If True, compute phase-aligned analysis. If False, compute sniff-triggered ERSP.
    save_channels : bool
        If True, save per-channel data. If False, average across channels.
    with_baseline : bool
        If True, use new function that saves baseline period (only for phase_aligned=True)
    baseline_for_mi : bool
        If True, save RAW EEG and respiration from baseline for MI null testing
    """
    # Optionally override the time-locked window end (same effect as editing ersp_end at the top of the file)
    _apply_time_window(time_window_end)
    try:
        sys.stdout.reconfigure(line_buffering=True)      # live log lines when stdout is redirected to a file
    except Exception:
        pass

    # Get subject IDs
    try:
        subject_ids = sorted(
            [
                f.split("_")[0].replace("subject", "")
                for f in os.listdir(info_dir)
                if f.endswith("_info.fif")
            ]
        )
    except FileNotFoundError:
        print(f"Error: The directory '{info_dir}' was not found.")
        subject_ids = []

    # Load channel names
    all_subject_ch_names = {}
    all_ch_sets = []
    if subject_ids:
        print(f"Found info for subjects: {subject_ids}")
        for sid in subject_ids:
            info_path = os.path.join(info_dir, f"subject{sid}_info.fif")
            info = mne.io.read_info(info_path, verbose=False)
            all_subject_ch_names[sid] = info["ch_names"]
            all_ch_sets.append(set(info["ch_names"]))

        shared_roi_channels = sorted(list(set(roi_channels).intersection(*all_ch_sets)))
        print("-" * 50)
        print(
            f"Found {len(shared_roi_channels)} shared ROI channels across all subjects."
        )
        print("These channels will be used for analysis:", shared_roi_channels)
        print("-" * 50)
    else:
        print("No subject info files found. Cannot proceed with analysis.")
        shared_roi_channels = []

    # epochs_eeg_respiration = np.load(
    #     os.path.join(data_path, "epochs_eeg_respiration.npy"), allow_pickle=True
    # ).item()
    if epochs_file is None:
        epochs_file = os.path.join(data_path, "epochs_eeg_respiration_withtimestamps_v2.npy")
    epochs_eeg_respiration = np.load(epochs_file, allow_pickle=True).item()

    # fire turns "--exclude_subjects=30" into an int and "30,31" into a tuple
    if isinstance(exclude_subjects, (str, int)):
        exclude_subjects = (exclude_subjects,)
    exclude_subjects = {str(x) for x in exclude_subjects}

    out = {}
    trial_counts_global = {}

    if shared_roi_channels:
        if phase_aligned:
            kind = "phase_baseline_mi" if baseline_for_mi else ("phase_baseline" if with_baseline else "phase")
        else:
            kind = "time"
        tasks = []
        for cond_name in epochs_eeg_respiration["EEG"].keys():
            out[cond_name] = {}
            for subject_id in epochs_eeg_respiration["EEG"][cond_name].keys():
                if subject_id not in all_subject_ch_names:
                    print(f"  Skipping subject {subject_id} (no info file found)")
                    continue
                if str(subject_id) in exclude_subjects:
                    print(f"  Skipping subject {subject_id} (excluded)")
                    continue
                current_subject_ch_list = all_subject_ch_names[subject_id]
                tasks.append(dict(
                    cond=cond_name, subj=subject_id, kind=kind, shared=shared_roi_channels,
                    eeg=epochs_eeg_respiration["EEG"][cond_name][subject_id],
                    resp=epochs_eeg_respiration["respiration"][cond_name][subject_id],
                    roi=[current_subject_ch_list.index(ch) for ch in shared_roi_channels],
                    save_channels=save_channels, time_window_end=time_window_end))

        for task, (results, counts, log_text) in zip(tasks, _run_tasks(tasks, n_jobs)):
            if log_text:
                print(log_text, end="")
            trial_counts_global.update(counts)
            if results is None:
                print(f"    No valid trials for subject {task['subj']}")
                continue
            out[task["cond"]][task["subj"]] = results

    print("\n=== Trial Summary ===")
    for (cond, subj), counts in trial_counts_global.items():
        print(
            f"  {cond} - Subj {subj}: valid={counts['valid']}  skipped={counts['skipped']}"
        )

    out["freqs"] = freqs

    if phase_aligned:
        out["x"] = phase_bin_centers
        if baseline_for_mi:
            if save_channels:
                filename = "phase_aligned_with_baseline_raw_for_mi_perchannel.npy"
            else:
                filename = "phase_aligned_with_baseline_raw_for_mi.npy"
        elif with_baseline:
            if save_channels:
                filename = "phase_aligned_with_baseline_perchannel.npy"
            else:
                filename = "phase_aligned_with_baseline.npy"
        else:
            if save_channels:
                filename = "phase_aligned_results_fullbreath_frontal_temporal_1deg_trials_perchannel.npy"
            else:
                #filename = "phase_aligned_results_fullbreath_frontal_temporal_1deg_trials.npy"
                filename = "phase_aligned_results_fullbreath_frontal_temporal_1deg_trials_v2.npy"

        save_to = out_file if out_file is not None else os.path.join(data_path, filename)
        os.makedirs(os.path.dirname(os.path.abspath(save_to)), exist_ok=True)
        np.save(save_to, dict(out))
        print(f"\nSaved to: {save_to}")
    else:
        out["x"] = times
        if save_channels:
            filename = "timealigned_ersp_results_avgbreath_perchannel.npy"
        else:
            filename = "timealigned_ersp_results_avgbreath.npy"

        save_to = out_file if out_file is not None else os.path.join(data_path, filename)
        os.makedirs(os.path.dirname(os.path.abspath(save_to)), exist_ok=True)
        np.save(save_to, dict(out))
        print(f"\nSaved to: {save_to}")

if __name__ == "__main__":
    fire.Fire()