"""Generate SYNTHETIC raw inputs for the epoching stage (numbers are meaningless; formats are real).

  <out>/eeglab/<subject>/<subject>_EEG_preprocessed.mat   EEGLAB v7.3 (HDF5) with chanlocs + DIN events
  <out>/olfactometer/OlfactometerSubject<subject>.csv     humidity-sensor record, one quoted record per row
  <info_dir>/subject<subject>_info.fif                     MNE Info (channel names) as used by generate_ersp.py

It deliberately contains the awkward cases the real pipeline has to handle: events before the
EEG/respiration offset, a last epoch that runs past the end of the EEG recording, sensor spikes
(removed by remove_peaks), extra leading CSV rows (subject 27), and (optionally) NaN humidity samples.

Usage:  python tests/make_synthetic_raw.py tests/data/raw tests/data/preprocessed/info --n-trials 9
"""
import argparse
import os

import h5py
import mne
import numpy as np

SRATE = 250.0
RESP_SFREQ = 8.75714285714
N_CH = 4
CH_NAMES = [f"E{i}" for i in range(1, N_CH + 1)]
CODES = {"water": "DIN4", "citrus": "DIN5", "oregano": "DIN6", "peppermint": "DIN7"}

# values used by the synthetic test config (config/test.yaml) and the equivalence checks
TEST_OFFSETS = {"22": 500, "23": 250, "24": 0, "27": 125, "28": 375, "35": 750}
TEST_BAD_CHANNELS = {"22": ["E3"], "23": [], "24": ["E4"], "27": ["E2"], "28": [], "35": ["E1"]}
SUBJECTS = ["22", "23", "24", "27", "28", "35"]


def _chars(s):
    return np.array([[ord(c)] for c in s], dtype=np.uint16)


def write_eeglab_mat(path, data_uV, srate, ch_names, positions_mm, latencies_1based, codes):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as f:
        refs = f.create_group("#refs#")
        eeg = f.create_group("EEG")
        eeg.create_dataset("data", data=data_uV.T)                  # MATLAB stores (n_samples, n_ch)
        eeg.create_dataset("srate", data=np.array([[srate]]))
        chan = eeg.create_group("chanlocs")
        n = len(ch_names)
        lab = chan.create_dataset("labels", shape=(n, 1), dtype=h5py.ref_dtype)
        for i, name in enumerate(ch_names):
            lab[i, 0] = refs.create_dataset(f"lab{i}", data=_chars(name)).ref
        for k, field in enumerate("XYZ"):
            ds = chan.create_dataset(field, shape=(n, 1), dtype=h5py.ref_dtype)
            for i in range(n):
                ds[i, 0] = refs.create_dataset(f"{field}{i}", data=np.array([[positions_mm[i][k]]])).ref
        ev = eeg.create_group("event")
        lat = ev.create_dataset("latency", shape=(len(latencies_1based), 1), dtype=h5py.ref_dtype)
        cod = ev.create_dataset("code", shape=(len(codes), 1), dtype=h5py.ref_dtype)
        for i, (l, c) in enumerate(zip(latencies_1based, codes)):
            lat[i, 0] = refs.create_dataset(f"lat{i}", data=np.array([[float(l)]])).ref
            cod[i, 0] = refs.create_dataset(f"cod{i}", data=_chars(c)).ref


def make_subject(out_dir, subject, offset, rng, n_trials=4, spacing_s=20.0, nan_humidity=False):
    # event schedule (adjusted time, i.e. relative to the offset), shuffled conditions
    conds = np.repeat(list(CODES), n_trials)
    rng.shuffle(conds)
    t_events = 5.0 + spacing_s * np.arange(len(conds))
    last = t_events[-1]

    # EEG ends only 10 s after the last event => its 15-s epoch is cut off and dropped by MNE
    n_samples = int((offset / SRATE + last + 10.0) * SRATE)
    data_uV = rng.normal(scale=10.0, size=(N_CH, n_samples))
    pos_mm = rng.uniform(-90, 90, size=(N_CH, 3))

    latencies = list((offset + t_events * SRATE).astype(int))
    codes = [CODES[c] for c in conds]
    # events that fall BEFORE the offset (must be dropped) + start marker + an unknown code
    latencies = [10, 60, 20] + latencies
    codes = ["DIN4", "DIN6", "DIN3"] + codes
    latencies.append(latencies[-1] + 5)
    codes.append("DIN99")
    write_eeglab_mat(os.path.join(out_dir, "eeglab", subject, f"{subject}_EEG_preprocessed.mat"),
                     data_uV, SRATE, CH_NAMES, pos_mm, np.array(latencies) + 1, codes)

    # humidity record: breathing-like oscillation + spikes + (optional NaNs); covers every event's epoch
    n_resp = int((last + 16.5) * RESP_SFREQ)
    t = np.arange(n_resp) / RESP_SFREQ
    hum = 45.0 + 8.0 * np.sin(2 * np.pi * t / 4.0) + rng.normal(scale=0.8, size=n_resp)
    spike_idx = rng.choice(n_resp, size=6, replace=False)
    hum[spike_idx] = rng.choice([1000.0, -50.0, 101167.23], size=6)
    rows = []
    if subject == "27":                                     # 73 junk rows the pipeline must skip
        rows += [f'"0.0,{7777.0},0.0,JUNK,00:00:00"'] * 73
    for k, h in enumerate(hum):
        if nan_humidity and k % 97 == 5:
            rows.append(f'"23.5"')                           # record without a humidity value -> NaN
        else:
            rows.append(f'"23.5,{h:.3f},1013.2,N/A,12:{(k // 3600) % 60:02d}:{(k // 60) % 60:02d}"')
    os.makedirs(os.path.join(out_dir, "olfactometer"), exist_ok=True)
    with open(os.path.join(out_dir, "olfactometer", f"OlfactometerSubject{subject}.csv"), "w") as fh:
        fh.write("record\n" + "\n".join(rows) + "\n")


def main(out_dir, info_dir=None, seed=0, nan_subject="23", n_trials=4):
    rng = np.random.default_rng(seed)
    for s in SUBJECTS:
        make_subject(out_dir, s, TEST_OFFSETS[s], rng, n_trials=n_trials, nan_humidity=(s == nan_subject))
    if info_dir:
        os.makedirs(info_dir, exist_ok=True)
        for s in SUBJECTS:
            info = mne.create_info(CH_NAMES, SRATE, "eeg")
            mne.io.write_info(os.path.join(info_dir, f"subject{s}_info.fif"), info, overwrite=True)
    print(f"Synthetic raw data written to {out_dir} (subjects {SUBJECTS}; NaN humidity in {nan_subject})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("info_dir", nargs="?")
    ap.add_argument("--n-trials", type=int, default=4, help="trials per condition (use >= 9 for the habituation analysis)")
    a = ap.parse_args()
    main(a.out_dir, a.info_dir, n_trials=a.n_trials)
