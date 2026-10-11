# Data deposit: what is shared, what is computed, what can be regenerated

**License of the deposit:** CC-BY-NC-4.0.

The raw EEG (EEGLAB `.mat` files), the olfactometer CSVs and the continuous preprocessed EEG are **not** deposited
(file size; available on request). Everything needed to regenerate every figure and table is: the deposit holds the
preprocessed epochs plus small *derived* files whose creation step needs the undeposited data. All code that created the
derived files is in this repository and was verified against the original notebooks on synthetic data.

## Files (sizes marked ~ are estimates; `python scripts/make_deposit.py` prints the real ones)

| file | contents | size | regenerates |
|---|---|---|---|
| the epochs file the paper's ERSPs were computed from (the original `epochs_eeg_respiration.npy`; the rebuilt `..._v2.npy` is a different file and does not reproduce the paper's ERSPs) | preprocessed EEG epochs and respiration epochs | 3.36 GB | stages 1-4 (ERSP, classification, habituation, Fig 3-5, S3-S4, Tables S1, S4) and the breath-duration analysis (Table S3) |
| `info/subject<ID>_info.fif` | channel names and electrode montage | 18 KB each | ERSP channel selection; Fig 2 topoplots |
| `epochs_v2_for_timing.npy`, `continuous_respiration_v2.npy` | respiration epochs + event timestamps (v2 epochs, no EEG) and continuous respiration (no EEG) | a few MB each | odor-arrival timing (Section 1.8; Tables S5, S6) |
| `fig1_example_trial.npy` | one subject's QC-passed humidity snippets | ~ 30 kB | Fig. 1, right panel (example respiration trace) |
| `continuous_respiration_humidity.npy` | continuous respiration, event times, metadata - **no EEG** | ~ a few MB | PAC respiratory phase |
| `pac_results_slim.npy` | PAC results (observed MI, 1000-surrogate null, z) per subject, channel, band; the unused per-surrogate arrays are dropped | ~ 70 MB (full file ~ 1.3 GB) | Fig 2, Section 1.1 |
| `full_respiration_EEG_humidity_meta.npy` | channel names, electrode info, bad channels | < 1 MB | Fig 2 (cluster adjacency, topoplots) |
| `psd_cache.npy` | Welch spectra of every channel, and inhale/exhale spectra | ~ 10 MB | Section 1.2, Table S2 |
| `ersp/*.npy` (**recommended**) | the three ERSP result files the paper's numbers were computed from | ~ 1 GB | skips the wavelet step **and is needed to reproduce Tables S1/S4, Figs 3-5 and S4 exactly**: ERSPs recomputed from the epochs agree only approximately (AUCs within ~0.004) |
| `MANIFEST.sha256` | SHA-256 and size of every file | - | `scripts/verify_data.py` |

## What a downloader computes vs. what is already there

| step | from the deposit | time (indicative) |
|---|---|---|
| epochs from raw EEG | **not possible** (raw not deposited) - the epochs file is provided | - |
| ERSP (wavelet) | computed from the epochs, **or** provided in `ersp/` | the slow step of stages 1-4: a wavelet transform per channel and trial per file (about 0.1 s each on the slow CPU I tested; not measured on your data) |
| classification, habituation, statistics, figures | computed | minutes to tens of minutes |
| PAC surrogate test | **provided** (`pac_results_slim.npy`); recomputing needs the continuous EEG | hours (on request) |
| Welch spectra for FOOOF | **provided** (`psd_cache.npy`); recomputing needs the continuous EEG | minutes |

## Archive layout
Some archives (e.g. Zenodo's web upload, as far as we know) list files flat and do not keep subfolders. The deposit is therefore published as the top-level files
plus two zip files for the subfolders:

```
epochs_eeg_respiration.npy, epochs_v2_for_timing.npy, continuous_respiration_v2.npy, continuous_respiration_humidity.npy,
fig1_example_trial.npy, full_respiration_EEG_humidity_meta.npy, pac_results_slim.npy, psd_cache.npy, MANIFEST.sha256
info.zip      -> unzip inside data/preprocessed/   (creates info/)
ersp.zip      -> unzip inside data/preprocessed/   (creates ersp/; then copy its three files to results/ersp/)
```
Unzip first, then `python scripts/verify_data.py` checks everything against `MANIFEST.sha256`. (To build the two zip files from `deposit/`:
`cd deposit && zip -r info.zip info && zip -r ersp.zip ersp`, and upload everything except the `info/` and `ersp/` folders themselves.)

## Using the deposit

```bash
# 1. clone the repository, put the deposit files in data/preprocessed/ (info/ is a subfolder) and the ersp/ files in results/ersp/
python scripts/verify_data.py --manifest data/preprocessed/MANIFEST.sha256 --root data/preprocessed
# 2. dry run: shows which steps will run (none of them needs raw or continuous EEG)
snakemake -n --configfile config/deposit.yaml
# 3. reproduce everything; 4. summarise how many published numbers were matched
snakemake --configfile config/deposit.yaml -c4 --keep-going
python scripts/summarize_checks.py --results results
```
`config/deposit.yaml` differs from the default configuration only in that the PAC results are read from the slimmed file `pac_results_slim.npy`
and that the free-response analysis (needs `sentence-transformers` and a one-time model download) is switched on.
The expensive steps (`make_epochs`, `make_full_recordings`, `ersp_*`, `pac_compute`, `psd_cache`, `recordings_meta`) run
**only if their output file is missing** (or with `--forcerun <rule>`), never because file timestamps differ after a download
or a `git clone`. Cheap steps (classification, statistics, figures) re-run whenever their inputs, parameters or code change.

## For the authors: building the deposit
```bash
snakemake psd_cache                       # needs the continuous EEG, once
python scripts/make_deposit.py --config config/config.yaml --out-dir deposit --include-ersp --copy-all
```
With `--copy-all`, `deposit/` is the complete folder to upload (about 3.6 GB with the epochs file and the ERSP files); without it, large existing files are only referenced in the manifest.
If your files are not at the paths of `config/config.yaml`, point the script at them with `--ersp-dir`, `--pac-results`, `--recordings-dir`, `--psd-cache`, `--epochs`.

## Data-availability statement
The statement submitted to PNAS lists "analysis code and scripts, olfactometer hardware design and build instructions,
respiration data, and preprocessed EEG recordings". The files above add **derived intermediate results** (slimmed PAC
results, a spectra cache, channel metadata), and the PAC analysis starts from continuous preprocessed EEG that is *not* in
the deposit. Consider adding to the statement, e.g.: "Derived intermediate results (PAC results, power-spectra cache) are
deposited so that all figures and tables can be regenerated; the continuous preprocessed EEG from which they were computed
is available on request." That wording is a suggestion - adjust it to what you actually deposit.
