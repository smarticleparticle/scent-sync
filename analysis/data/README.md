# Data layout

Nothing in this folder is committed.  Paths are set in `config/config.yaml -> paths`.

```
data/
├── raw/                                     # only needed to RE-CREATE the epochs (stage 0)
│   ├── eeglab/<ID>/<ID>_EEG_preprocessed.mat      EEGLAB (MATLAB v7.3/HDF5) file after your EEGLAB preprocessing
│   └── olfactometer/OlfactometerSubject<ID>.csv   olfactometer humidity-sensor record
└── preprocessed/                            # what is shared; stage 0 writes epochs here, or you drop it in
    ├── epochs_eeg_respiration.npy                 the epochs file the paper's ERSPs were computed from (deposit)
    ├── epochs_eeg_respiration_withtimestamps_v2.npy   rebuilt from raw by stage 0 (not deposited; a different file, see docs/DETAILS.md)
    ├── epochs_v2_for_timing.npy, continuous_respiration_v2.npy   small, no EEG (NOT in the public deposit; odor-arrival timing, on request)
    ├── fig1_example_trial.npy                     one subject's respiration snippets (deposit; Fig. 1 right panel)
    ├── pac_results_slim.npy, full_respiration_EEG_humidity_meta.npy, psd_cache.npy   derived, small (deposit; Fig. 2, Table S2, Fig. S2)
    ├── full_respiration_EEG_v2.npy, full_respiration_EEG_humidity.npy   continuous EEG (NOT deposited; only to recompute PAC / spectra)
    └── info/subject<ID>_info.fif                  one MNE Info per subject (channel names / montage)
(results/ersp/ holds the three ERSP files; the deposit ships them, see DEPOSIT.md)
```

**If you only have the preprocessed epochs file**, put it at `data/preprocessed/epochs_...v2.npy`; the
workflow then skips stage 0 and starts from the ERSPs.  (If the raw files are missing *and* the epochs
file is missing, Snakemake names exactly which raw files it would need.)

## epochs_eeg_respiration_withtimestamps_v2.npy
`np.load(..., allow_pickle=True).item()` -> dict:
```
{"EEG":         {condition: {subject_id: array(n_trials, n_channels, n_samples)}},   # 250 Hz, odor onset at sample 250
 "respiration": {condition: {subject_id: array(n_trials, n_resp_samples)}},          # 8.757 Hz, z-scored humidity
 "timestamps":  {condition: {subject_id: array(n_trials)}}}                          # event time (s) in the aligned recording
```
conditions: `water`, `citrus`, `oregano`, `peppermint`; subject ids are strings (`"22"`).  Epoch window
-1 to +15 s around the DIN event.

## info/
**Include the info file of every subject in the epochs file, including excluded ones.**
`generate_ersp.py` takes the ROI channels common to *all* `*_info.fif` files in the folder, so adding or
removing a file can change which channels enter every ERSP.

## full_respiration_EEG_*.npy
dict with `respiration`, `EEG` (**pickled MNE RawArray** per subject), `event_times`, `metadata`.  Loading needs
a compatible MNE version (see `results/versions.txt`).

## pac_results_humidity.npy  (+ full_respiration_EEG_humidity_meta.npy)
`pac_results['subject_level'][subject][channel][band]` = dict with `MI_observed`, `MI_null_distribution` (1000 surrogates),
`z_score`, ...; `pac_results['metadata']` holds the settings.  Computed by `snakemake pac_compute` (hours) or provided.
The `_meta` file is derived from `full_respiration_EEG_humidity.npy` (channel names, electrode info, bad channels) so that
the multi-GB recordings are not needed for the statistics and Fig 2.  `info/subject22_info.fif` supplies the montage for the topoplots.

## psd_cache.npy
Welch spectra (1-45 Hz) of every channel with PAC results, plus inhale/exhale spectra; made by `snakemake psd_cache` (needs the
continuous EEG) and shipped, so that Section 1.2 / Table S2 can be regenerated without it. See ../DEPOSIT.md for the full deposit.

## ERSP files (generated into results/ersp/)
dict: `{condition: {subject_id: {trial_index: array(n_freqs, n_x)}}, "freqs": ..., "x": ...}`
where `x` is phase-bin centres (deg) or time (s).
