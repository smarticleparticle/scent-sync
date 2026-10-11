"""Respiration-EEG phase-amplitude coupling (PAC) -- computation.

Ported from analysis notebook cell 89.  The functions below are the notebook's, unchanged (extracted from the
notebook source); only the module header, the configuration (now config.yaml -> pac) and the driver at the bottom are new.
The driver (`run_pac`) is the loop that was commented out in cells 89/90.

Method (paper Methods, "Phase Amplitude Coupling"): respiratory phase is linearised between detected breath landmarks
(0 -> pi inhale, pi -> 2pi exhale; invalid breaths = NaN), EEG band amplitude comes from the Hilbert transform, and the
modulation index (MI, Tort et al. 2010) is tested against 1000 circular-shift surrogates.

RANDOM NUMBERS: the surrogates use NumPy's *global* generator (`np.random.randint`), seeded once.  The saved results
therefore depend on the order in which subjects, channels and bands are processed.  ``run_pac(seed_mode="notebook")``
reproduces the notebook's single sequential pass exactly; ``seed_mode="per_subject"`` re-seeds for every subject so that
subjects can be computed independently (different surrogate draws, statistically equivalent).
"""
import time
from fractions import Fraction

import numpy as np
from scipy import stats as sp_stats  # noqa: F401  (kept from the notebook header)
from scipy.signal import butter, filtfilt, find_peaks, hilbert, resample_poly


def check_respiration_quality(resp_signal, sfreq, resp_band=(0.1, 0.5), 
                              skip_start_sec=60):
    """
    Quality control check for respiratory signal.
    
    NEW: Skips initial settling period and checks RAW signal for breathing.
    
    Parameters:
    -----------
    skip_start_sec : float
        Skip this many seconds at start (default 60s for sensor settling)
    """
    from scipy.fft import fft, fftfreq
    
    # SKIP INITIAL TRANSIENT
    skip_samples = int(skip_start_sec * sfreq)
    if len(resp_signal) <= skip_samples:
        print(f"    ⚠️  FAIL: Signal too short (only {len(resp_signal)/sfreq:.1f}s)")
        return False
    
    resp_signal_trimmed = resp_signal[skip_samples:]
    
    print(f"    Skipping first {skip_start_sec}s (sensor settling)")
    print(f"    Using {len(resp_signal_trimmed)/sfreq:.1f}s for quality check")
    
    # Filter
    filtered = bandpass_filter(resp_signal_trimmed, resp_band[0], resp_band[1], sfreq, order=2)
    
    # =========================================================================
    # Check 1: Signal amplitude on RAW (not filtered) signal
    # =========================================================================
    # Use RAW signal to check amplitude (filtering might remove slow oscillations)
    raw_std = np.std(resp_signal_trimmed)
    raw_range = np.percentile(resp_signal_trimmed, 95) - np.percentile(resp_signal_trimmed, 5)
    
    if raw_std < 1e-10:  # Extremely conservative threshold
        print(f"    ⚠️  FAIL: Raw signal essentially flat (std={raw_std:.2e})")
        return False
    
    # Check that signal has variation (very lenient for humidity)
    if raw_range < raw_std * 0.1:  # Even more lenient than before
        print(f"    ⚠️  FAIL: Raw signal has no dynamic range")
        return False
    
    print(f"    ✓ Raw signal: std={raw_std:.2e}, range={raw_range:.2e}")
    
    # =========================================================================
    # Check 2: Detrend before filtering (CRITICAL for humidity!)
    # =========================================================================
    from scipy.signal import detrend
    
    # Remove slow drift before checking respiratory band
    resp_detrended = detrend(resp_signal_trimmed, type='linear')
    
    # Now filter the detrended signal
    filtered = bandpass_filter(resp_detrended, resp_band[0], resp_band[1], sfreq, order=2)
    
    filtered_std = np.std(filtered)
    filtered_range = np.percentile(filtered, 95) - np.percentile(filtered, 5)
    
    print(f"    After detrend + filter: std={filtered_std:.2e}, range={filtered_range:.2e}")
    
    # Check filtered signal has SOME variation
    if filtered_std < 1e-10:
        print(f"    ⚠️  FAIL: No respiratory oscillations after filtering")
        return False
    
    # =========================================================================
    # Check 3: Peak detection (on detrended + filtered)
    # =========================================================================
    filtered_norm = (filtered - np.mean(filtered)) / (filtered_std + 1e-12)
    peaks, _ = find_peaks(filtered_norm, distance=int(2.0 * sfreq), prominence=0.5)  # Lowered prominence
    
    duration_min = len(resp_signal_trimmed) / sfreq / 60
    breathing_rate = 60 * len(peaks) / (len(resp_signal_trimmed) / sfreq)
    
    if breathing_rate < 5 or breathing_rate > 35:  # Wider range
        print(f"    ⚠️  FAIL: Breathing rate out of range ({breathing_rate:.1f} bpm)")
        return False
    
    # Adaptive minimum based on duration
    min_expected_breaths = duration_min * 5  # 5 bpm minimum
    if len(peaks) < max(10, min_expected_breaths * 0.3):
        print(f"    ⚠️  FAIL: Too few breaths detected ({len(peaks)} peaks in {duration_min:.1f} min)")
        return False
    
    # =========================================================================
    # Check 4: Spectral peak (on detrended signal)
    # =========================================================================
    freqs = fftfreq(len(resp_detrended), 1/sfreq)
    fft_vals = fft(resp_detrended)
    power = np.abs(fft_vals)**2
    
    resp_range_mask = (freqs >= resp_band[0]) & (freqs <= resp_band[1])
    
    if np.sum(resp_range_mask) == 0:
        print(f"    ⚠️  FAIL: No power in respiratory range")
        return False
    
    peak_freq = freqs[resp_range_mask][np.argmax(power[resp_range_mask])]
    peak_bpm = peak_freq * 60
    
    if peak_bpm < 6 or peak_bpm > 30:
        print(f"    ⚠️  FAIL: Spectral peak outside normal range ({peak_bpm:.1f} bpm)")
        return False
    
    # All checks passed
    print(f"    ✓ PASS: {len(peaks)} breaths, rate={breathing_rate:.1f} bpm, spectral peak={peak_bpm:.1f} bpm")
    return True


def bandpass_filter(data, low_freq, high_freq, sfreq, order=3):
    """Bandpass filter using Butterworth filter"""
    nyquist = sfreq / 2.0
    low = low_freq / nyquist
    high = high_freq / nyquist
    low = np.clip(low, 0.001, 0.999)
    high = np.clip(high, 0.001, 0.999)
    
    if low >= high:
        raise ValueError(f"Low freq {low_freq} must be < high freq {high_freq}")
    
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, padlen=3*(max(len(b), len(a))-1))


def compute_linearized_respiratory_phase_direct(
    resp_signal, sfreq, resp_band=(0.1, 0.5),
    min_duration=2.0, max_duration=8.0, 
    verbose=False):
    """
    Compute respiratory phase using direct peak detection with breath-by-breath validation.
    
    NOTE: Assumes signal has already been trimmed (e.g., initial transient removed upstream)
    
    Invalid breaths → NaN phase
    """
    
    # DETREND (remove slow drift)
    from scipy.signal import detrend
    resp_detrended = detrend(resp_signal, type='linear')
    
    # Bandpass filter
    filtered = bandpass_filter(resp_detrended, resp_band[0], resp_band[1], sfreq, order=2)
    
    # Signal quality checks
    signal_std = np.std(filtered)
    
    if signal_std < 1e-10:
        if verbose:
            print(f"    ⚠️  FAIL: Signal essentially flat after filtering (std={signal_std:.2e})")
        return None
    
    # Normalize
    filtered_norm = (filtered - np.mean(filtered)) / signal_std
    
    # Find peaks and troughs
    min_distance = int(min_duration * 0.5 * sfreq)
    peaks, _ = find_peaks(filtered_norm, distance=min_distance, prominence=0.5)
    troughs, _ = find_peaks(-filtered_norm, distance=min_distance, prominence=0.5)
    
    if len(peaks) < 2:
        if verbose:
            print(f"    ⚠️  Insufficient peaks detected ({len(peaks)})")
        return None
    
    # Initialize phase array
    phase = np.full(len(filtered), np.nan)
    
    # Convert duration limits to samples
    min_samples = int(min_duration * sfreq)
    max_samples = int(max_duration * sfreq)
    
    # Process each breath cycle
    valid_cycles = 0
    
    for i in range(len(peaks) - 1):
        peak_start = peaks[i]
        peak_end = peaks[i + 1]
        
        cycle_duration = peak_end - peak_start
        if cycle_duration < min_samples or cycle_duration > max_samples:
            continue
        
        troughs_between = troughs[(troughs > peak_start) & (troughs < peak_end)]
        
        if len(troughs_between) == 0:
            search_segment = filtered_norm[peak_start:peak_end]
            if len(search_segment) == 0:
                continue
            trough = peak_start + np.argmin(search_segment)
        else:
            trough = troughs_between[0]
        
        inhale_length = trough - peak_start
        exhale_length = peak_end - trough
        
        if inhale_length <= 0 or exhale_length <= 0:
            continue
        
        # LINEAR PHASE ASSIGNMENT
        phase[peak_start:trough] = np.linspace(0, np.pi, inhale_length, endpoint=False)
        phase[trough:peak_end] = np.linspace(np.pi, 2*np.pi, exhale_length, endpoint=False)
        
        valid_cycles += 1
    
    if valid_cycles == 0:
        if verbose:
            print("    ⚠️  No valid breath cycles after duration filtering")
        return None
    
    if verbose:
        total_samples = len(filtered)
        valid_samples = np.sum(~np.isnan(phase))
        rejection_rate = 100 * (1 - valid_samples / total_samples)
        
        print(f"    Signal quality: std={signal_std:.2e} (after detrend+filter)")
        print(f"    Detected peaks: {len(peaks)}, troughs: {len(troughs)}")
        print(f"    Valid cycles: {valid_cycles}/{len(peaks)-1} ({100*valid_cycles/(len(peaks)-1):.1f}%)")
        print(f"    Valid phase samples: {valid_samples}/{total_samples} ({100*valid_samples/total_samples:.1f}%)")
        print(f"    Rejection rate: {rejection_rate:.1f}%")
        
        if rejection_rate > 30:
            print(f"    ⚠️  WARNING: High rejection rate!")
    
    return phase


def get_instantaneous_amplitude(signal, sfreq, band):
    """Extract instantaneous amplitude using Hilbert transform"""
    filtered = bandpass_filter(signal, band[0], band[1], sfreq, order=3)
    analytic_signal = hilbert(filtered)
    amplitude = np.abs(analytic_signal)
    return amplitude


def compute_modulation_index(phase, amplitude, n_bins=18):
    """
    Compute Modulation Index (MI) as per Tort et al. 2010.
    Handles NaN values in phase.
    """
    phase_bins_edges = np.linspace(0, 2 * np.pi, n_bins + 1)
    
    # Remove NaN values
    valid_mask = ~np.isnan(phase)
    phase_valid = phase[valid_mask]
    amplitude_valid = amplitude[valid_mask]
    
    if len(phase_valid) == 0:
        return 0.0, np.zeros(n_bins)
    
    # Compute mean amplitude in each phase bin
    mean_amp = np.zeros(n_bins)
    for i in range(n_bins):
        in_bin = (phase_valid >= phase_bins_edges[i]) & (phase_valid < phase_bins_edges[i + 1])
        if np.sum(in_bin) > 0:
            mean_amp[i] = np.mean(amplitude_valid[in_bin])
        else:
            mean_amp[i] = 0.0
    
    # Normalize to probability distribution
    if np.sum(mean_amp) > 0:
        p = mean_amp / np.sum(mean_amp)
    else:
        return 0.0, mean_amp
    
    # Shannon entropy
    p_nonzero = p[p > 0]
    if len(p_nonzero) == 0:
        return 0.0, mean_amp
    
    H = -np.sum(p_nonzero * np.log(p_nonzero))
    H_max = np.log(n_bins)
    MI = (H_max - H) / H_max
    
    return MI, mean_amp


def compute_preferred_phase(mean_amp_per_bin):
    """Compute preferred phase from amplitude distribution"""
    n_bins = len(mean_amp_per_bin)
    phase_centers = np.linspace(0, 2*np.pi, n_bins, endpoint=False) + (np.pi/n_bins)
    z = np.sum(mean_amp_per_bin * np.exp(1j * phase_centers))
    preferred_phase = np.angle(z)
    return preferred_phase


def create_surrogate_amplitude_with_nan_mask(amplitude, phase, sfreq, min_shift_sec=10.0):
    """
    Create surrogate amplitude that RESPECTS the phase NaN mask.
    
    CRITICAL: Ensures both observed and surrogate use THE SAME valid phase indices.
    """
    # Find valid phase indices
    valid_mask = ~np.isnan(phase)
    valid_indices = np.where(valid_mask)[0]
    
    if len(valid_indices) < 2:
        raise ValueError("Insufficient valid phase samples")
    
    min_shift_samples = int(min_shift_sec * sfreq)
    n_valid = len(valid_indices)
    
    if n_valid < min_shift_samples * 2:
        raise ValueError(f"Not enough valid samples for {min_shift_sec}s shift")
    
    # Extract amplitude at valid indices
    valid_amplitudes = amplitude[valid_indices]
    
    # Circular shift within valid sample space
    shift_amount = np.random.randint(min_shift_samples, n_valid - min_shift_samples)
    shifted_valid_amplitudes = np.roll(valid_amplitudes, shift_amount)
    
    # Create surrogate with NaNs in same places as phase
    amplitude_surrogate = np.full_like(amplitude, np.nan)
    amplitude_surrogate[valid_indices] = shifted_valid_amplitudes
    
    return amplitude_surrogate


def compute_pac_continuous_linearized_direct(
    eeg_signal, resp_signal, sfreq_eeg, sfreq_resp,
    resp_band, freq_bands, 
    n_phase_bins=18, n_surrogates=200, min_shift_sec=10.0,
    min_cycle_duration=2.0, max_cycle_duration=8.0, 
    skip_start_sec=60, verbose=False):
    """
    Compute PAC using linearized phase with direct peak detection and NaN handling.
    
    NEW: Skips initial transient period.
    """
    
    # 1. Resample respiration
    ratio = Fraction(sfreq_eeg / sfreq_resp).limit_denominator(1000)
    resp_resampled = resample_poly(resp_signal, ratio.numerator, ratio.denominator)
    
    # Trim to match
    min_len = min(len(eeg_signal), len(resp_resampled))
    eeg_signal = eeg_signal[:min_len]
    resp_resampled = resp_resampled[:min_len]
    
    # **NEW: Skip initial transient period from BOTH signals**
    skip_samples_eeg = int(skip_start_sec * sfreq_eeg)
    
    if len(eeg_signal) <= skip_samples_eeg:
        if verbose:
            print(f"    ⚠️  Signal too short after skipping {skip_start_sec}s")
        return {band_name: None for band_name in freq_bands.keys()}
    
    # Trim both signals to skip initial period
    eeg_signal = eeg_signal[skip_samples_eeg:]
    resp_resampled = resp_resampled[skip_samples_eeg:]
    
    # 2. Extract respiratory phase (now operates on already-trimmed signal)
    resp_phase = compute_linearized_respiratory_phase_direct(
        resp_resampled, 
        sfreq_eeg, 
        resp_band=resp_band,
        min_duration=min_cycle_duration,
        max_duration=max_cycle_duration,
        verbose=verbose
    )
    
    if resp_phase is None:
        return {band_name: None for band_name in freq_bands.keys()}
    
    # **VERIFY DIMENSIONS MATCH**
    if len(resp_phase) != len(eeg_signal):
        if verbose:
            print(f"    ⚠️  Dimension mismatch: EEG={len(eeg_signal)}, phase={len(resp_phase)}")
        return {band_name: None for band_name in freq_bands.keys()}
    
    # Count valid samples
    n_valid_phases = np.sum(~np.isnan(resp_phase))
    if n_valid_phases < sfreq_eeg * 10:  # Need at least 10 seconds
        return {band_name: None for band_name in freq_bands.keys()}
    
    # 3. Process each EEG frequency band
    results = {}
    
    for band_name, (low_freq, high_freq) in freq_bands.items():
        
        # Extract EEG amplitude
        eeg_amplitude = get_instantaneous_amplitude(eeg_signal, sfreq_eeg, 
                                                     (low_freq, high_freq))
        
        # Compute observed MI
        MI_obs, mean_amp_per_bin_obs = compute_modulation_index(
            resp_phase, eeg_amplitude, n_bins=n_phase_bins
        )
        
        # Compute observed preferred phase
        preferred_phase_obs = compute_preferred_phase(mean_amp_per_bin_obs)
        
        # Generate null distribution WITH NaN HANDLING
        MI_null = np.zeros(n_surrogates)
        surrogate_preferred_phases = np.zeros(n_surrogates)
        surrogate_mean_amp_bins = []
        
        for i in range(n_surrogates):
            # KEY: Use NaN-aware surrogate creation
            amplitude_surrogate = create_surrogate_amplitude_with_nan_mask(
                eeg_amplitude, resp_phase, sfreq_eeg, min_shift_sec=min_shift_sec
            )
            
            # Compute MI for surrogate (NaNs in same places as observed)
            MI_null[i], mean_amp_surrogate = compute_modulation_index(
                resp_phase, amplitude_surrogate, n_bins=n_phase_bins
            )
            
            surrogate_preferred_phases[i] = compute_preferred_phase(mean_amp_surrogate)
            surrogate_mean_amp_bins.append(mean_amp_surrogate)
        
        # Statistical testing
        p_value = np.mean(MI_null >= MI_obs)
        z_score = (MI_obs - np.mean(MI_null)) / (np.std(MI_null) + 1e-10)
        effect_size = (MI_obs - np.mean(MI_null)) / (np.mean(MI_null) + 1e-10)
        
        # Store results
        results[band_name] = {
            'MI_observed': MI_obs,
            'MI_null_mean': np.mean(MI_null),
            'MI_null_std': np.std(MI_null),
            'MI_null_distribution': MI_null,
            'MI_null_ci_95': np.percentile(MI_null, [2.5, 97.5]),
            'p_value': p_value,
            'z_score': z_score,
            'effect_size': effect_size,
            'significant_p05': p_value < 0.05,
            'significant_p01': p_value < 0.01,
            'mean_amp_per_phase_bin': mean_amp_per_bin_obs,
            'preferred_phase_observed': preferred_phase_obs,
            'surrogate_preferred_phases': surrogate_preferred_phases,
            'surrogate_mean_amp_bins': np.array(surrogate_mean_amp_bins),
            'n_samples': len(eeg_signal),
            'n_valid_phase_samples': n_valid_phases,
            'duration_sec': len(eeg_signal) / sfreq_eeg,
            'phase_method': 'linearized_peak_to_peak_direct_with_validation'
        }
    
    return results


# =============================================================================
# DRIVER  (the loop that was commented out in notebook cells 89/90)
# =============================================================================
def run_pac(recordings, subjects, cfg, seed_mode="notebook", verbose=True):
    """PAC for every channel and band of every subject.

    recordings : dict from full_respiration_EEG_humidity.npy ('EEG' RawArray per subject, 'respiration', 'metadata')
    cfg        : dict with sfreq_eeg, sfreq_resp, resp_band, freq_bands, n_phase_bins, n_surrogates, min_shift_sec,
                 min_cycle_duration, max_cycle_duration, seed
    Returns the same nested dict the notebook saved as pac_results_humidity*.npy:
        {'subject_level': {subject: {channel: {band: result}}}, 'metadata': {...}}
    """
    sfreq_eeg, sfreq_resp = cfg["sfreq_eeg"], cfg["sfreq_resp"]
    resp_band, freq_bands = tuple(cfg["resp_band"]), {k: tuple(v) for k, v in cfg["freq_bands"].items()}
    n_phase_bins, n_surrogates = cfg["n_phase_bins"], cfg["n_surrogates"]
    min_shift_sec = cfg["min_shift_sec"]
    min_cycle_duration, max_cycle_duration = cfg["min_cycle_duration"], cfg["max_cycle_duration"]

    results = {
        "subject_level": {},
        "metadata": {
            "resp_band": resp_band, "freq_bands": freq_bands, "n_phase_bins": n_phase_bins,
            "n_surrogates": n_surrogates, "min_shift_sec": min_shift_sec,
            "min_cycle_duration": min_cycle_duration, "max_cycle_duration": max_cycle_duration,
            "sfreq_eeg": sfreq_eeg, "sfreq_resp": sfreq_resp,
            "phase_method": "linearized_peak_to_peak_direct_with_validation",
        },
    }
    if seed_mode == "notebook":
        np.random.seed(cfg["seed"])          # seeded once, as when notebook cell 89 was run

    ratio = Fraction(sfreq_eeg / sfreq_resp).limit_denominator(1000)
    t0 = time.time()
    for subject_id in subjects:
        if seed_mode == "per_subject":
            np.random.seed(cfg["seed"] + int(subject_id))
        elif seed_mode != "notebook":
            raise ValueError(f"unknown seed_mode {seed_mode!r}")
        raw_eeg = recordings["EEG"][subject_id]
        resp_signal = recordings["respiration"][subject_id]
        metadata = recordings["metadata"][subject_id]

        if verbose:
            print(f"\n=== Subject {subject_id} ===", flush=True)
        resp_resampled_check = resample_poly(resp_signal, ratio.numerator, ratio.denominator)
        if not check_respiration_quality(resp_resampled_check, sfreq_eeg, resp_band):
            print(f"  EXCLUDING subject {subject_id}: poor respiratory signal quality")
            continue

        eeg_data = raw_eeg.get_data()
        ch_names = raw_eeg.ch_names
        results["subject_level"][subject_id] = {}
        n_ok = n_fail = 0
        for ch_idx, ch_name in enumerate(ch_names):
            if ch_name in metadata["bad_channels"]:
                continue
            try:
                channel_results = compute_pac_continuous_linearized_direct(
                    eeg_signal=eeg_data[ch_idx, :], resp_signal=resp_signal,
                    sfreq_eeg=sfreq_eeg, sfreq_resp=sfreq_resp, resp_band=resp_band, freq_bands=freq_bands,
                    n_phase_bins=n_phase_bins, n_surrogates=n_surrogates, min_shift_sec=min_shift_sec,
                    min_cycle_duration=min_cycle_duration, max_cycle_duration=max_cycle_duration,
                    verbose=(ch_idx == 0))
                if any(channel_results[band] is not None for band in freq_bands):
                    results["subject_level"][subject_id][ch_name] = channel_results
                    n_ok += 1
                else:
                    n_fail += 1
            except Exception as e:
                print(f"  WARNING: channel {ch_name} failed: {e}")
                n_fail += 1
        if verbose:
            print(f"  done: {n_ok} channels ({n_fail} failed), {time.time() - t0:.0f}s elapsed", flush=True)
    return results
