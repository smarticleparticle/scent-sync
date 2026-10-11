"""SYNTHETIC inputs for the PAC stage (numbers are meaningless; structures match the real files).

  make_recordings(...)    continuous EEG (mne RawArray) + humidity-like respiration + metadata, i.e. the structure of
                          full_respiration_EEG_humidity.npy; alpha amplitude is modulated by respiratory phase.
  make_pac_results(...)   ready-made PAC results (structure of pac_results_humidity*.npy) with spatially clustered
                          coupling, for testing the group statistics and Fig. 2 without the hours-long computation.
  make_info(...)          an MNE Info with a montage of the right (tiny) scale incl. fiducials, like subject22_info.fif.
"""
import mne
import numpy as np

SFREQ_EEG = 250.0
SFREQ_RESP = 8.75714285714
BANDS = ["theta", "alpha", "beta", "low_gamma"]


def sphere_positions(n, rng, radius=0.00991):
    v = rng.normal(size=(n, 3))
    v[:, 2] = np.abs(v[:, 2])                      # upper hemisphere
    return radius * v / np.linalg.norm(v, axis=1, keepdims=True)


def make_info(n_ch=30, seed=0, radius=0.00991):
    rng = np.random.default_rng(seed)
    names = [f"E{i}" for i in range(1, n_ch + 1)]
    pos = sphere_positions(n_ch, rng, radius)
    info = mne.create_info(names, SFREQ_EEG, "eeg")
    montage = mne.channels.make_dig_montage(
        ch_pos=dict(zip(names, pos)), nasion=[0.0, radius, 0.0], lpa=[-radius, 0.0, 0.0], rpa=[radius, 0.0, 0.0],
        coord_frame="head")
    info.set_montage(montage)
    return info


def make_recordings(subjects=("22", "23"), n_ch=3, duration_s=240.0, seed=0, bad_channels=None):
    rng = np.random.default_rng(seed)
    out = {"EEG": {}, "respiration": {}, "metadata": {}}
    n_eeg, n_resp = int(duration_s * SFREQ_EEG), int(duration_s * SFREQ_RESP)
    for s in subjects:
        t_resp = np.arange(n_resp) / SFREQ_RESP
        period = rng.uniform(3.6, 4.6)
        resp = 0.5 + 0.4 * np.sin(2 * np.pi * t_resp / period) + rng.normal(scale=0.02, size=n_resp)
        t = np.arange(n_eeg) / SFREQ_EEG
        phase = 2 * np.pi * t / period
        data = rng.normal(scale=1e-5, size=(n_ch, n_eeg))
        data += (1 + 0.6 * np.sin(phase))[None, :] * 1e-5 * np.sin(2 * np.pi * 10 * t)[None, :]      # alpha follows breathing
        out["EEG"][s] = mne.io.RawArray(data, make_info(n_ch, seed=99), verbose=False)
        out["respiration"][s] = resp
        out["metadata"][s] = {"bad_channels": list((bad_channels or {}).get(s, [])), "resp_sfreq": SFREQ_RESP,
                              "eeg_sfreq": SFREQ_EEG}
    return out


def make_pac_results(subjects, info, n_surrogates=60, seed=0, none_fraction=0.03):
    """Spatially structured synthetic PAC results.  alpha / low_gamma couple in 'frontal' channels (y > 0)."""
    rng = np.random.default_rng(seed)
    pos = np.array([info["chs"][i]["loc"][:3] for i in range(len(info["ch_names"]))])
    front = pos[:, 1] > 0.3 * np.abs(pos[:, 1]).max()
    strength = {"theta": 0.0, "alpha": 1.4, "beta": 0.5, "low_gamma": 1.0}
    results = {"subject_level": {}, "metadata": {"n_surrogates": n_surrogates, "resp_band": (0.1, 0.4),
                                                 "sfreq_eeg": SFREQ_EEG, "sfreq_resp": SFREQ_RESP}}
    for s in subjects:
        subj_gain = rng.uniform(0.3, 1.7)
        results["subject_level"][s] = {}
        for ci, ch in enumerate(info["ch_names"]):
            band_res = {}
            for band in BANDS:
                if rng.random() < none_fraction:
                    band_res[band] = None
                    continue
                null = rng.normal(0.010, 0.002, size=n_surrogates)
                eff = strength[band] * subj_gain * (1.0 if front[ci] else 0.15)
                mi_obs = null.mean() + (eff + rng.normal(scale=0.8)) * null.std()
                band_res[band] = {"MI_observed": float(mi_obs), "MI_null_distribution": null,
                                  "z_score": float((mi_obs - null.mean()) / (null.std() + 1e-10))}
            results["subject_level"][s][ch] = band_res
    return results
