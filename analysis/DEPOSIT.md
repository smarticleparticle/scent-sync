# Data deposit

Deposited in the Stanford Digital Repository (link to be added). It contains what is needed to regenerate every figure and table of the paper. The raw EEG, the olfactometer files, the continuous EEG and the whole-session respiration recordings (used only
for the odor-arrival timing check, Section 1.8 and Tables S5-S6) are not included (available on reasonable request).

## Contents (about 4.7 GB)

| file | what it is | used for |
|---|---|---|
| `epochs_eeg_respiration.npy` | preprocessed 128-channel EEG and respiration, cut into trials around each odor or water delivery, per participant and condition | the data itself; breath durations (Table S3), Fig. 3, and recomputing the ERSPs |
| `ersp/` (3 files) | trial-level time-frequency maps (ERSPs) computed from the EEG | odor classification and habituation (Table S1, S4, Figs. 3-5, S3, S4) |
| `pac_results_slim.npy` | respiration-EEG coupling (PAC) results per participant, channel and frequency band | Fig. 2, Section 1.1 |
| `psd_cache.npy` | power spectra per channel | Section 1.2, Table S2, Fig. S2 |
| `full_respiration_EEG_humidity_meta.npy` | channel names and each participant's list of bad (interpolated) channels | Fig. 2 and Section 1.1 (the statistics use only the good channels) |
| `fig1_example_trial.npy` | respiration (nasal humidity) snippets of one participant | the example trace in Fig. 1 |
| `info/subject<ID>_info.fif` (17 files) | electrode names and positions (the same standard cap layout for every participant), and that participant's list of bad (interpolated) channels | channel selection; Fig. 2 topographies |
| `MANIFEST.sha256` | SHA-256 checksum of every file | `scripts/verify_data.py` |

## Participants and privacy
* The deposit contains data from the **17 participants analyzed in the paper**, identified only by integer codes (the codes used in the paper's supplement). Other participants are not included.
* There are **no names, demographics, questionnaires, smell-test scores, photographs, or dates, clock times or event timestamps of testing.**
  File metadata is scrubbed: the montage files contain no date, experimenter or subject fields, and all files carry one neutral modification date.
* Check a copy yourself with `python scripts/audit_deposit.py --root <folder> --config config/config.yaml --include-large`.
* The `.npy` files are Python dictionaries (pickles): load only files whose checksum matches `MANIFEST.sha256`.

## Using it
Put the files in `data/preprocessed/` (keep `info/` as a subfolder), copy the three files of `ersp/` to `results/ersp/`, then follow the README. If your download tool does not keep subfolders, the deposit also
exists as the top-level files plus `info.zip` and `ersp.zip`: unzip them inside `data/preprocessed/` first.

## Building it (authors)
`python scripts/make_deposit.py --config config/config.yaml --out-dir deposit_final --include-ersp --copy-all --analyzed-only` collects the files, keeps only the analyzed participants, sets neutral file dates and
writes the manifest. If your files are not at the paths of `config/config.yaml`, point the script to them with `--ersp-dir`, `--pac-results`, `--recordings-dir`, `--psd-cache`, `--epochs`.
