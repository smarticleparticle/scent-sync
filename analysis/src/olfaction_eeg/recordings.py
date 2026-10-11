"""Raw (EEGLAB-preprocessed EEG + olfactometer humidity sensor) -> aligned recordings and epochs.

Consolidates four near-identical copies of the same code from the preprocessing notebook
(epochs_eeg_respiration_withtimestamps.ipynb, cells 2, 5, 12, 17):

  * ``load_eeglab_mat``            EEGLAB v7.3 .mat loader        (identical in all four cells)
  * ``extract_values_and_text`` / ``read_olfactometer_csv``  olfactometer CSV parsing
  * ``remove_peaks``               outlier interpolation          (cell 5; in cell 2 it was nested
                                   after a ``return`` and therefore never defined -- the notebook
                                   only worked because an earlier session had defined it)
  * ``build_epochs``               cell 2   -> epochs_eeg_respiration_withtimestamps_v2.npy
  * ``build_full_recordings``      cell 5   -> full_respiration_EEG_v2.npy              (variant "zscore")
                                   cell 17  -> full_respiration_EEG_humidity.npy        (variant "minmax_smooth")

Every per-subject hand correction that used to be hardcoded in the loop (bad channels, EEG/respiration
offsets, trimmed CSV rows, dropped/shifted epochs) is passed in through ``cfg`` (see config.yaml).

Scalar conversions use ``.item()`` rather than ``float(np.array(...))``, which is an error on
numpy >= 2.
"""
from datetime import datetime

import h5py
import mne
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d

CONDITION_NAMES = {4: "water", 5: "citrus", 6: "oregano", 7: "peppermint"}
CONDITION_ORDER = ["water", "citrus", "oregano", "peppermint"]
EVENT_DICT = {"DIN4": 4, "DIN5": 5, "DIN6": 6, "DIN7": 7, "DIN3": 3}  # DIN3 = start marker


# ---------------------------------------------------------------------------
# Olfactometer CSV  (humidity sensor used as respiration signal)
# ---------------------------------------------------------------------------
def extract_values_and_text(row):
    """Extract temperature, humidity, pressure, text, and time from a CSV row."""
    row_str = str(row.iloc[0])
    parts = row_str.split(",")

    temperature = None
    humidity = None
    pressure = None
    non_numeric_text = "N/A"
    time_value = None

    numeric_values = []
    for part in parts:
        part = part.strip()
        try:
            numeric_values.append(float(part))
        except ValueError:
            if part and not part.replace(":", "").replace(".", "").isdigit():
                non_numeric_text = part
            elif ":" in part:
                try:
                    time_value = datetime.strptime(part, "%H:%M:%S").time()
                except ValueError:
                    pass

    if len(numeric_values) >= 1:
        temperature = float(numeric_values[0])
    if len(numeric_values) >= 2:
        humidity = float(numeric_values[1])
    if len(numeric_values) >= 3:
        pressure = float(numeric_values[2])

    return pd.Series({
        "Temperature": temperature,
        "Humidity": humidity,
        "Pressure": pressure,
        "Text": non_numeric_text,
        "Time": time_value,
    })


def read_olfactometer_csv(path, subject_id, drop_first_rows=None):
    """Read one subject's olfactometer CSV; returns the humidity trace (np.ndarray).

    ``drop_first_rows``: {subject: n}  rows to discard at the start (e.g. {"27": 73}).
    """
    olfact_breath = pd.read_csv(path)
    olfact_breath[["Temperature", "Humidity", "Pressure", "Text", "Time"]] = \
        olfact_breath.apply(extract_values_and_text, axis=1)
    n_drop = (drop_first_rows or {}).get(str(subject_id))
    if n_drop:
        olfact_breath = olfact_breath.iloc[n_drop:].reset_index(drop=True)
    return olfact_breath["Humidity"].values


def remove_peaks(data, lower_bound=0, upper_bound=200):
    """Replace out-of-range samples by linear interpolation between the nearest valid neighbours."""
    data = np.asarray(data, dtype=float)
    mask = (data >= lower_bound) & (data <= upper_bound)
    outlier_indices = np.where(~mask)[0]
    cleaned = data.copy()
    for idx in outlier_indices:
        left = idx - 1
        while left >= 0 and not mask[left]:
            left -= 1
        right = idx + 1
        while right < len(data) and not mask[right]:
            right += 1
        if left >= 0 and right < len(data):
            cleaned[idx] = np.interp(idx, [left, right], [data[left], data[right]])
        elif left >= 0:
            cleaned[idx] = data[left]
        elif right < len(data):
            cleaned[idx] = data[right]
    return cleaned


def respiration_zscore(humidity, clip=(0, 200)):
    """Original-pipeline respiration: clip outliers (interpolated) then z-score.  (cells 2 / 5)"""
    smoothed = remove_peaks(humidity, lower_bound=clip[0], upper_bound=clip[1])
    return (smoothed - np.mean(smoothed)) / np.std(smoothed)


def respiration_minmax_smooth(humidity, window_size=70):
    """Min-max normalise (NaN-aware), interpolate NaNs, uniform-smooth.  (cell 17, used for PAC)"""
    humidity_min = np.nanmin(humidity)
    humidity_max = np.nanmax(humidity)
    norm_humidity = (humidity - humidity_min) / (humidity_max - humidity_min)
    nan_mask = np.isnan(norm_humidity)
    if np.any(nan_mask):
        valid_idx = np.where(~nan_mask)[0]
        valid_values = norm_humidity[~nan_mask]
        norm_humidity = np.interp(np.arange(len(norm_humidity)), valid_idx, valid_values)
    return uniform_filter1d(norm_humidity, size=window_size, mode="nearest")


# ---------------------------------------------------------------------------
# EEGLAB v7.3 .mat
# ---------------------------------------------------------------------------
def _scalar(x):
    """h5py gives 1x1 arrays for MATLAB scalars; float(np.array(...)) fails on numpy >= 2."""
    return np.asarray(x).reshape(-1)[0].item()


def load_eeglab_mat(file_path):
    """Load an EEGLAB (MATLAB v7.3 / HDF5) .mat file.

    Returns dict(data=(n_ch, n_samples) in volts, srate, ch_names, pos=(n_ch, 3) in metres,
                 events=(n_events, 3) int array [latency_sample, 0, code]).
    Only events DIN3..DIN7 are kept.
    """
    with h5py.File(file_path, "r") as f:
        eeg = f["EEG"]
        data = np.array(eeg["data"])
        data = data.T / 1e6  # microvolts -> volts
        srate = _scalar(eeg["srate"])

        chanlocs = eeg["chanlocs"]
        n_channels = chanlocs["labels"].shape[0]

        ch_names, pos = [], []
        for i in range(n_channels):
            label_ref = chanlocs["labels"][i][0]
            label_data = f[label_ref][()]
            ch_names.append("".join(chr(c) for c in label_data.flatten()).strip())

            def get_coord(field):
                if field in chanlocs:
                    try:
                        ref = chanlocs[field][i][0]
                        return _scalar(f[ref])
                    except Exception:
                        return 0.0
                return 0.0

            pos.append([get_coord("X"), get_coord("Y"), get_coord("Z")])

        events_group = f["EEG/event"]
        latencies, event_ids = [], []
        for event_name in events_group:
            if "latency" == event_name:
                for lat_ref in events_group["latency"]:
                    latency = _scalar(f[lat_ref[0]]) - 1  # MATLAB is 1-based
                    latencies.append(int(latency))
            if "code" == event_name:
                for code_data_ref in events_group["code"]:
                    code_bytes = f[code_data_ref[0]][()]
                    event_ids.append("".join(chr(c) for c in code_bytes.flatten()).strip())

        event_ids_int = [EVENT_DICT.get(e, -1) for e in event_ids]
        events_array = np.array(
            [[lat, 0, eid] for lat, eid in zip(latencies, event_ids_int) if eid != -1], dtype=int)

    return {"data": data, "srate": srate, "ch_names": ch_names,
            "pos": np.array(pos) / 1000.0, "events": events_array}


def make_raw(eeg, bads=None, data=None):
    """Build an MNE RawArray (with the EEGLAB electrode positions) from ``load_eeglab_mat`` output."""
    montage = mne.channels.make_dig_montage(ch_pos=dict(zip(eeg["ch_names"], eeg["pos"])), coord_frame="head")
    info = mne.create_info(ch_names=eeg["ch_names"], sfreq=eeg["srate"], ch_types=["eeg"] * len(eeg["ch_names"]))
    info.set_montage(montage)
    raw = mne.io.RawArray(eeg["data"] if data is None else data, info, verbose=False)
    if bads:
        raw.info["bads"] = bads
    return raw, info


def _align_events(events_array, offset, srate):
    """Shift event times by the EEG/respiration offset and drop events that fall before time 0."""
    adjusted_times_sec = (events_array[:, 0] - offset) / srate
    event_codes = events_array[:, 2]
    invalid_mask = adjusted_times_sec < 0
    valid_mask = ~invalid_mask
    return adjusted_times_sec[valid_mask], event_codes[valid_mask], invalid_mask


# ---------------------------------------------------------------------------
# Epoching  (notebook cell 2)
# ---------------------------------------------------------------------------
def epoch_respiration(resp_data, resp_sfreq, event_times_sec, tmin=-1.0, tmax=15.0, verbose=True):
    """Epoch respiration data around event times (events too close to the edges are skipped)."""
    n_samples = len(resp_data)
    start_offset = int(np.round(tmin * resp_sfreq))
    stop_offset = int(np.round(tmax * resp_sfreq))

    epochs, valid_times, valid_indices = [], [], []
    for idx, t in enumerate(event_times_sec):
        center_sample = int(np.round(t * resp_sfreq))
        start = center_sample + start_offset
        stop = center_sample + stop_offset
        if start <= 0 or stop > n_samples:
            if verbose:
                print(f"Skipping event at {t}s (index {idx}) - out of bounds")
            continue
        epochs.append(resp_data[start:stop])
        valid_times.append(t)
        valid_indices.append(idx)
    return np.array(epochs), valid_times, valid_indices


def build_subject_epochs(subject_id, eeg, resp, cfg, verbose=True):
    """Epoch one subject.  Returns {"respiration": {cond: arr}, "EEG": {cond: arr}, "timestamps": {cond: arr}}.

    cfg keys used: tmin, tmax, resp_sfreq, bad_channels, offsets, drop_last_kept_epoch,
                   shift_kept_indices, drop_first_trial.
    """
    srate, resp_sfreq = eeg["srate"], cfg["resp_sfreq"]
    tmin, tmax = cfg["tmin"], cfg["tmax"]
    raw, _ = make_raw(eeg, bads=cfg["bad_channels"].get(subject_id))

    offset = cfg["offsets"].get(subject_id, 0)
    adjusted_times_sec, event_codes, invalid_mask = _align_events(eeg["events"], offset, srate)
    if verbose:
        print(f"Using offset: {offset} samples; removed {int(invalid_mask.sum())} events before offset")

    events_by_condition = {}
    for code in np.unique(event_codes):
        events_by_condition[code] = adjusted_times_sec[event_codes == code]

    # Per-condition EEG epochs (mne drops epochs that run past the end of the recording)
    kept_indices_by_cond, epoch_by_cond = {}, {}
    for cond_code, cond_name in CONDITION_NAMES.items():
        cond_times_sec = events_by_condition.get(cond_code, np.array([]))
        if len(cond_times_sec) == 0:
            continue
        cond_event_samples = np.round(cond_times_sec * srate + offset).astype(int)
        cond_events_array = np.column_stack((
            cond_event_samples,
            np.zeros(len(cond_event_samples), dtype=int),
            np.full(len(cond_event_samples), cond_code, dtype=int),
        ))
        cond_epoch = mne.Epochs(raw, cond_events_array, event_id={cond_name: cond_code},
                                tmin=tmin, tmax=tmax, baseline=None, preload=True, verbose=False)
        kept_indices_by_cond[cond_name] = cond_epoch.selection
        epoch_by_cond[cond_name] = cond_epoch

    # Respiration epochs, matched to the kept EEG epochs
    kept_resp, kept_timestamps = {}, {}
    for cond_code, cond_name in CONDITION_NAMES.items():
        if cond_code not in events_by_condition:
            if verbose:
                print(f"No events found for {cond_name}, skipping...")
            continue

        resp_epochs_all, valid_times_all, _ = epoch_respiration(
            resp, resp_sfreq, events_by_condition[cond_code], tmin=tmin, tmax=tmax, verbose=verbose)

        kept_indices = kept_indices_by_cond.get(cond_name, [])

        # ---- documented hand corrections (were hardcoded `if subject_id == ...` blocks) ----
        if cond_name in cfg["drop_last_kept_epoch"].get(subject_id, []):
            kept_indices = kept_indices[:-1]
        if cond_name in cfg["shift_kept_indices"].get(subject_id, {}):
            shift = cfg["shift_kept_indices"][subject_id][cond_name]
            kept_indices = [x + shift for x in kept_indices]

        if len(kept_indices) == 0:
            print(f"ERROR: No kept indices for {cond_name} (subject {subject_id})")
            continue

        if max(kept_indices) >= len(resp_epochs_all):
            print(f"WARNING: kept_indices out of bounds for {cond_name} (subject {subject_id})")
            while len(kept_indices) > 0 and max(kept_indices) >= len(resp_epochs_all):
                kept_indices = kept_indices[:-1]
            if len(kept_indices) == 0:
                print(f"ERROR: Could not match {cond_name} - skipping this condition")
                continue

        kept_resp[cond_name] = resp_epochs_all[kept_indices]
        kept_timestamps[cond_name] = np.array([valid_times_all[i] for i in kept_indices])

    out = {"respiration": {}, "EEG": {}, "timestamps": {}}
    for cond in kept_resp:
        out["respiration"][cond] = kept_resp[cond]
        out["timestamps"][cond] = kept_timestamps[cond]

    for cond_name in CONDITION_ORDER:
        if cond_name in kept_resp:
            eeg_data = epoch_by_cond[cond_name].get_data()
            resp_length = kept_resp[cond_name].shape[0]
            if eeg_data.shape[0] > resp_length:
                eeg_data = eeg_data[:resp_length, :, :]
            out["EEG"][cond_name] = eeg_data
        elif cond_name in epoch_by_cond:
            out["EEG"][cond_name] = epoch_by_cond[cond_name].get_data()

    if subject_id in cfg["drop_first_trial"]:
        for cond in CONDITION_ORDER:
            if cond in out["EEG"] and len(out["EEG"][cond]) > 0:
                out["EEG"][cond] = out["EEG"][cond][1:]
                if cond in out["respiration"]:
                    out["respiration"][cond] = out["respiration"][cond][1:]
                    out["timestamps"][cond] = out["timestamps"][cond][1:]
    return out


def build_epochs(subjects, eeglab_path_fmt, csv_path_fmt, cfg, verbose=True):
    """Epoch all subjects -> dict identical in structure to epochs_eeg_respiration_withtimestamps_v2.npy."""
    dataset = {k: {cond: {} for cond in CONDITION_ORDER} for k in ("respiration", "EEG", "timestamps")}
    for subject_id in subjects:
        if verbose:
            print(f"\n=== Subject {subject_id} ===")
        humidity = read_olfactometer_csv(csv_path_fmt.format(subject=subject_id), subject_id, cfg["csv_drop_first_rows"])
        resp = respiration_zscore(humidity, cfg["resp_clip"])
        eeg = load_eeglab_mat(eeglab_path_fmt.format(subject=subject_id))
        res = build_subject_epochs(subject_id, eeg, resp, cfg, verbose=verbose)
        for key in dataset:
            for cond, arr in res[key].items():
                dataset[key][cond][subject_id] = arr
        for cond in CONDITION_ORDER:
            r = dataset["respiration"][cond].get(subject_id)
            e = dataset["EEG"][cond].get(subject_id)
            t = dataset["timestamps"][cond].get(subject_id)
            n = [None if x is None else x.shape[0] for x in (r, e, t)]
            flag = "OK" if (None not in n and len(set(n)) == 1) else "MISMATCH/MISSING"
            if verbose or flag != "OK":
                print(f"  [{flag}] {cond}: resp={n[0]} eeg={n[1]} timestamps={n[2]}")
    return dataset


# ---------------------------------------------------------------------------
# Continuous aligned recordings  (notebook cells 5 and 17)
# ---------------------------------------------------------------------------
def build_full_recordings(subjects, eeglab_path_fmt, csv_path_fmt, cfg, variant, verbose=True):
    """Continuous, offset-aligned EEG + respiration per subject.

    variant "zscore"        = notebook cell 5  (full_respiration_EEG_v2.npy; timing analyses)
    variant "minmax_smooth" = notebook cell 17 (full_respiration_EEG_humidity.npy; PAC / FOOOF)

    NOTE: ["EEG"][subject] is a pickled MNE RawArray (as in the notebook); loading such a file requires
    a compatible MNE version -- one reason to record versions (results/versions.txt).
    """
    resp_sfreq = cfg["resp_sfreq"]
    out = {"respiration": {}, "EEG": {}, "event_times": {}, "metadata": {}}
    for subject_id in subjects:
        if verbose:
            print(f"\n=== Subject {subject_id} ({variant}) ===")
        humidity = read_olfactometer_csv(csv_path_fmt.format(subject=subject_id), subject_id, cfg["csv_drop_first_rows"])
        if variant == "zscore":
            resp = respiration_zscore(humidity, cfg["resp_clip"])
        elif variant == "minmax_smooth":
            resp = respiration_minmax_smooth(humidity, cfg["minmax_smooth_window"])
        else:
            raise ValueError(f"unknown variant {variant!r}")

        eeg = load_eeglab_mat(eeglab_path_fmt.format(subject=subject_id))
        srate = eeg["srate"]
        raw, info = make_raw(eeg, bads=cfg["bad_channels"].get(subject_id))

        offset = cfg["offsets"].get(subject_id, 0)
        adjusted_times_sec, event_codes, invalid_mask = _align_events(eeg["events"], offset, srate)

        eeg_data_aligned = eeg["data"][:, offset:]
        resp_duration_in_eeg_samples = int(len(resp) * (srate / resp_sfreq))
        min_duration_samples = min(eeg_data_aligned.shape[1], resp_duration_in_eeg_samples)
        eeg_data_truncated = eeg_data_aligned[:, :min_duration_samples]
        resp_duration_samples = int(min_duration_samples / (srate / resp_sfreq))
        resp_truncated = resp[:resp_duration_samples]

        raw_aligned = mne.io.RawArray(eeg_data_truncated, info, verbose=False)
        raw_aligned.info["bads"] = raw.info["bads"]
        out["respiration"][subject_id] = resp_truncated
        out["EEG"][subject_id] = raw_aligned

        max_time_sec = min_duration_samples / srate
        valid_event_mask = adjusted_times_sec < max_time_sec
        t_trunc = adjusted_times_sec[valid_event_mask]
        c_trunc = event_codes[valid_event_mask]

        by_cond = {CONDITION_NAMES[code]: t_trunc[c_trunc == code] for code in [4, 5, 6, 7]}
        out["event_times"][subject_id] = {
            "water": by_cond["water"], "citrus": by_cond["citrus"],
            "oregano": by_cond["oregano"], "peppermint": by_cond["peppermint"],
            "all_events_sec": t_trunc, "all_event_codes": c_trunc,
        }
        out["metadata"][subject_id] = {
            "resp_sfreq": resp_sfreq,
            "eeg_sfreq": srate,
            "offset_samples": offset,
            "offset_seconds": offset / srate,
            "resp_duration_sec": len(resp_truncated) / resp_sfreq,
            "eeg_duration_sec": eeg_data_truncated.shape[1] / srate,
            "matched_duration_sec": min_duration_samples / srate,
            "bad_channels": raw_aligned.info["bads"],
            "n_resp_samples": len(resp_truncated),
            "n_eeg_samples": eeg_data_truncated.shape[1],
            "n_eeg_channels": len(eeg["ch_names"]),
            "n_events_total": len(t_trunc),
            "n_events_removed_before_offset": np.sum(invalid_mask),
            "n_events_removed_after_truncation": np.sum(~valid_event_mask),
        }
        if verbose:
            print(f"respiration {len(resp_truncated)} samples, EEG {eeg_data_truncated.shape[1]} samples, "
                  f"{len(t_trunc)} valid events")
    return out


# ---------------------------------------------------------------------------
# Light-weight extract of the continuous recordings (channel names, electrode positions, bad channels)
# ---------------------------------------------------------------------------
class RecordingStub:
    """Stands in for an mne RawArray where only ``ch_names`` and ``info`` are needed (e.g. PAC cluster tests, topoplots)."""

    def __init__(self, info):
        self.info = info
        self.ch_names = list(info["ch_names"])


def make_recordings_meta(full_recordings):
    """{'EEG': {subject: RecordingStub}, 'metadata': {...}} from a full_respiration_EEG_*.npy dict (a few KB per subject)."""
    return {"EEG": {s: RecordingStub(raw.info.copy()) for s, raw in full_recordings["EEG"].items()},
            "metadata": full_recordings["metadata"]}
