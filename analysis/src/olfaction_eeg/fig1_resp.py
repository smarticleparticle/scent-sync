"""Fig. 1 (right panel): one example respiration trace (nasal humidity) around the TTL pulse, with inhale and exhale marked.

Ported from analysis-notebook cell 115.  ``bandpass_filter``, ``remove_peaks``, ``extract_values_and_text`` are the notebook's, unchanged.
Edits to ``extract_respiratory_data``: the two hard-coded file paths and the per-subject EEG/respiration offset table are now arguments (the table is
``recordings.eeg_resp_offset_samples`` in config/config.yaml; it is identical to the notebook's).  The plotting code, which ran at the top level of
the notebook cell, is now ``plot_example_trial`` (same statements; the cell's matplotlib font settings are applied with ``rc_context``).

The extraction needs the raw EEGLAB file and the olfactometer CSV (not deposited); it yields a tiny file (one trial's humidity snippet) that IS
deposited, so the panel can be redrawn without the raw data.
"""
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks, resample_poly
from fractions import Fraction

SMALLEST_SIZE = 10
SMALL_SIZE = 12
MED_SIZE = 14
BIG_SIZE = 16
RC = {"font.size": SMALL_SIZE, "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "axes.titlesize": SMALL_SIZE,
      "axes.labelsize": SMALL_SIZE, "legend.fontsize": SMALLEST_SIZE, "xtick.labelsize": SMALLEST_SIZE, "ytick.labelsize": SMALLEST_SIZE}


def bandpass_filter(data, lowcut, highcut, fs, order=2):
    """Bandpass filter for respiration"""
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    return filtfilt(b, a, data)

def remove_peaks(data, lower_bound=0, upper_bound=101):
    mask = (data >= lower_bound) & (data <= upper_bound)
    outlier_indices = np.where(~mask)[0]
    smoothed_data = data.copy()
    for idx in outlier_indices:
        left = idx - 1
        right = idx + 1
        while left >= 0 and not mask[left]: left -= 1
        while right < len(data) and not mask[right]: right += 1
        if left >= 0 and right < len(data):
            smoothed_data[idx] = np.interp(idx, [left, right], [data[left], data[right]])
        elif left >= 0: smoothed_data[idx] = data[left]
        elif right < len(data): smoothed_data[idx] = data[right]
    return smoothed_data

def extract_values_and_text(row):
    row_str = str(row.iloc[0])
    parts = row_str.split(',')
    temperature = humidity = pressure = None
    non_numeric_text = "N/A"
    numeric_values = []
    for part in parts:
        part = part.strip()
        try:
            numeric_values.append(float(part))
        except ValueError:
            if part and not part.replace(':', '').replace('.', '').isdigit():
                non_numeric_text = part
    if len(numeric_values) >= 1: temperature = float(numeric_values[0])
    if len(numeric_values) >= 2: humidity = float(numeric_values[1])
    if len(numeric_values) >= 3: pressure = float(numeric_values[2])
    return pd.Series({'Temperature': temperature, 'Humidity': humidity, 
                     'Pressure': pressure, 'Text': non_numeric_text})


def extract_respiratory_data(subject_id, resp_file, file_path, offset_map):
    """Extract QC-filtered respiratory trials for one subject"""
    
    olfact_breath = pd.read_csv(resp_file)
    olfact_breath[['Temperature', 'Humidity', 'Pressure', 'Text']] = \
        olfact_breath.apply(extract_values_and_text, axis=1)
    
    if subject_id == '27':
        olfact_breath = olfact_breath.iloc[73:].reset_index(drop=True)
    
    cleaned_humidity = remove_peaks(olfact_breath['Humidity'])
    smooth_humidity = cleaned_humidity.rolling(window=5, center=True).mean().bfill().ffill()
    
    with h5py.File(file_path, 'r') as f:
        eeg = f['EEG']
        srate = np.array(eeg['srate']).item()
        events_group = f['EEG/event']
        latencies = []
        event_ids = []
        
        for event_name in events_group:
            if event_name == 'latency':
                for lat_ref in events_group['latency']:
                    latency = np.array(f[lat_ref[0]]).item() - 1
                    latencies.append(int(latency))
            if event_name == 'code':
                for code_data_ref in events_group['code']:
                    code_data = f[code_data_ref[0]]
                    code_bytes = code_data[()]
                    code_str = ''.join(chr(c) for c in code_bytes.flatten()).strip()
                    event_ids.append(code_str)
        
        event_dict = {'DIN4': 4, 'DIN5': 5, 'DIN6': 6, 'DIN7': 7, 'DIN3': 3}
        event_ids_int = [event_dict.get(event, -1) for event in event_ids]
        events_array = np.array([[lat, 0, eid] for lat, eid in zip(latencies, event_ids_int) if eid != -1], dtype=int)
    
    
    offset = offset_map.get(subject_id, 0)
    adjusted_times_sec = (events_array[:, 0] - offset) / srate
    valid_mask = adjusted_times_sec >= 0
    valid_times = adjusted_times_sec[valid_mask]
    
    resp_sfreq = 8.75714285714
    resp_band = (0.1, 0.4)
    pre_ttl = 4
    post_ttl = 4
    
    accept_window = (-0.4, 3.0)
    reject_window = (-1.0, -0.4)
    
    ratio = Fraction(srate / resp_sfreq).limit_denominator(1000)
    up, down = ratio.numerator, ratio.denominator
    
    all_trials = []
    
    for ttl_time in valid_times:
        ttl_idx = int(ttl_time * resp_sfreq)
        start_idx = ttl_idx - int(pre_ttl * resp_sfreq)
        end_idx = ttl_idx + int(post_ttl * resp_sfreq)
        
        if start_idx < 0 or end_idx >= len(smooth_humidity):
            continue
        
        snippet = smooth_humidity.iloc[start_idx:end_idx].values
        
        if np.any(np.isnan(snippet)) or np.std(snippet) < 1e-3:
            continue
        
        resp_filtered = bandpass_filter(snippet, resp_band[0], resp_band[1], resp_sfreq, order=2)
        resp_filtered = (resp_filtered - np.mean(resp_filtered)) / (np.std(resp_filtered) + 1e-12)
        
        resp_upsampled = resample_poly(resp_filtered, up, down)
        n_samples_needed = int((pre_ttl + post_ttl) * srate)
        if len(resp_upsampled) > n_samples_needed:
            resp_upsampled = resp_upsampled[:n_samples_needed]
        else:
            resp_upsampled = np.pad(resp_upsampled, (0, n_samples_needed - len(resp_upsampled)), mode='edge')
        
        peaks, _ = find_peaks(resp_upsampled, distance=int(0.5 * srate), prominence=0.05)
        
        ttl_idx_in_trial = int(pre_ttl * srate)
        accept_start = ttl_idx_in_trial + int(accept_window[0] * srate)
        accept_end = ttl_idx_in_trial + int(accept_window[1] * srate)
        reject_start = ttl_idx_in_trial + int(reject_window[0] * srate)
        reject_end = ttl_idx_in_trial + int(reject_window[1] * srate)
        
        peaks_in_accept = peaks[(peaks >= accept_start) & (peaks <= accept_end)]
        if len(peaks_in_accept) == 0:
            continue
        
        peaks_in_reject = peaks[(peaks >= reject_start) & (peaks < reject_end)]
        if len(peaks_in_reject) > 0:
            continue
        
        all_trials.append(snippet)
    
    return np.array(all_trials) if len(all_trials) > 0 else None


def plot_example_trial(trial, trial_idx=0, pre_ttl=4, post_ttl=4, save_path=None):
    """Notebook cell 115, plotting part.  ``trial``: humidity snippet (pre_ttl + post_ttl seconds at 8.757 Hz)."""
    with plt.rc_context(RC):
        # time axis: your snippets are pre_ttl + post_ttl seconds long at resp_sfreq
        resp_sfreq = 8.75714285714
        pre_ttl, post_ttl = 4, 4
        t = np.linspace(-pre_ttl, post_ttl, len(trial), endpoint=False)

        # --- small figure ---
        fig, ax = plt.subplots(figsize=(4.2, 2.6), dpi=200)

        ax.plot(t, trial, color='black', linewidth=2)
        ax.axvline(0, color='gray', linestyle='--', linewidth=1)  # TTL marker
        ax.axvline(0, color='gray', linestyle='--', linewidth=1)  # TTL marker
        ttl_idx = np.argmin(np.abs(t - 0))

        pre = trial[:ttl_idx]
        pre_t = t[:ttl_idx]

        # INHALE: last PEAK before TTL
        peaks_pre, _ = find_peaks(pre, distance=3)
        if len(peaks_pre) > 0:
            i = peaks_pre[-1]
            x_in, y_in = pre_t[i], pre[i]
            ax.annotate(
                "Inhale",
                xy=(x_in, y_in),
                xytext=(x_in - 0.4, y_in + 0.25),
                fontsize=SMALL_SIZE,
                fontweight="bold",
                color="red",
                ha="center"
            )

        # EXHALE: last TROUGH before TTL
        troughs_pre, _ = find_peaks(-pre, distance=3)   # troughs of pre
        if len(troughs_pre) > 0:
            j = troughs_pre[-1]
            x_ex, y_ex = pre_t[j], pre[j]
            ax.annotate(
                "Exhale",
                xy=(x_ex, y_ex),
                xytext=(x_ex + 0.6, y_ex - 1.1),
                fontsize=SMALL_SIZE,
                fontweight="bold",
                color="red",
                ha="center"
            )
        ax.set_title(f"Respiration \nSubject 1, Trial {trial_idx+1}",
                     fontsize=BIG_SIZE, fontweight='bold')
        ax.set_xlabel("Time from TTL pulse (s)", fontweight='bold')
        ax.set_ylabel("Relative Humidity (%)", fontweight='bold')

        ax.grid(alpha=0.2, linestyle=':', linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig
