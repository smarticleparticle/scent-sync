# Respiratory Phase Orchestrates Human Olfactory Cortical Dynamics: analysis code

Code and instructions to reproduce the analyses, tables and figures of

> Hathaway AY, Coleman TP. *Respiratory Phase Orchestrates Human Olfactory Cortical Dynamics.* PNAS (2026). DOI: `TODO`

and of the respiration-synchronized olfactometer paper's EEG analyses (respiratory phase-locked ERSPs, odor decoding, habituation,
respiration-EEG coupling, free-response odor identification). The hardware design of the olfactometer is described in the paper and its supplement.
The workflow is managed with [Snakemake](https://snakemake.readthedocs.io): every step declares its inputs and outputs, only stale steps re-run, and
`snakemake -n` shows what would happen before anything runs. **Every number the pipeline produces is compared with the number printed in the paper**
(`config/paper_values.yaml`), and one command summarises how many matched.

## Reproduce the paper from the data deposit

You need the preprocessed data deposit (DOI: `TODO`; contents and checksums in [DEPOSIT.md](DEPOSIT.md)). The raw EEG is *not* needed.

```bash
git clone https://github.com/TODO/TODO.git && cd TODO
conda env create -f environment.yml && conda activate olfaction-eeg      # Python 3.13; also installs this package

# put the deposit files in data/preprocessed/ (info/ is a subfolder) and the three ERSP files in results/ersp/
python scripts/verify_data.py --manifest data/preprocessed/MANIFEST.sha256 --root data/preprocessed

snakemake -n --configfile config/deposit.yaml                        # dry run: lists the steps; none needs raw or continuous EEG
snakemake --configfile config/deposit.yaml -c4 --keep-going          # reproduce everything
python scripts/summarize_checks.py --results results                 # how many published numbers were matched?
```
Figures go to `figures/`, tables and statistics to `results/<stage>/`. `-c4` is the number of cores; the (optional) wavelet step is parallel across
subject x condition blocks and gives the same result for any number of cores.

**What to expect.** With the deposited ERSP files, the checks of all stages compare with the published values: about 560 numbers
(Tables S1-S7, every AUC and p-value of Figs. 2-5, S3, S4, Sections 1.1-1.8, SI 1.4 and 3.1). The stages need minutes each, except:
* the **ERSP step** (hours; skipped when `results/ersp/*.npy` are present). ERSPs recomputed from the epochs agree with the paper's only
  approximately (power within ~3%, AUCs within ~0.004), so a few third-decimal checks can then differ; use the deposited ERSP files for exact numbers.
* the **PAC surrogate test** (hours; the deposit ships its result, `pac_results_slim.npy`). Surrogate draws depend on the order of processing and cannot be
  regenerated bit for bit from the paper's seed; the deposited result is the one behind Fig. 2.
* the **free-response analysis** (`word_scoring`) needs `sentence-transformers` (installed by `environment.yml`) and downloads the language model
  `all-mpnet-base-v2` once (about 420 MB; the exact commit used for the paper is pinned in `config/config.yaml`). To skip it:
  `snakemake --configfile config/deposit.yaml config/no_word_scoring.yaml -c4 --keep-going` (the second file overrides the first).

Individual items: `snakemake --configfile config/deposit.yaml -c4 <rule>`, with `<rule>` one of `classification`, `habituation`, `fig2`, `fooof_gating`,
`breath_stats`, `timing`, `subject28`, `cluster_perm`, `fig3`, `fig1_resp`, `figS2`, `word_scoring` (see the table below).

## Test the installation without any data
```bash
pytest -q tests                                                                   # unit tests, ~2 min
python tests/make_synthetic_raw.py tests/data/raw tests/data/preprocessed/info --n-trials 9
snakemake --configfile config/test.yaml -c3                                       # synthetic raw EEG -> epochs -> ERSP -> every stage -> figures, ~15 min
```
The synthetic run checks that the code runs end to end; it cannot reproduce numbers (checks are disabled there).
The PAC chain has its own synthetic test: `python tests/make_synthetic_pac_files.py tests/data/pac && snakemake fig2 fooof_gating --configfile config/test_pac.yaml -c1`.

## Starting from the raw data
Stage 0 (`make_epochs`, `make_full_recordings`) rebuilds the epochs and continuous recordings from the EEGLAB-preprocessed files and the olfactometer CSVs
(not deposited; available on request). Put them in `data/raw/` as described in [data/README.md](data/README.md) and run `snakemake -c4` with the default
configuration. The rebuilt epochs file (`..._v2.npy`) is not identical to the one the paper's ERSPs were computed from (see DEPOSIT.md), so use the deposited
file for the published numbers.

## Data availability
Raw EEG is not deposited; the deposit holds the preprocessed epochs plus small derived files so that **every figure and table can be regenerated without
the raw or continuous EEG**. What is shared, what is computed, and how to verify a download: [DEPOSIT.md](DEPOSIT.md). Layout of `data/`: [data/README.md](data/README.md).

## How the analysis is organized

```
raw EEGLAB .mat + olfactometer .csv --[0 make_epochs]--> epochs_...v2.npy --[1 generate_ersp]--> 3 ERSP files --+--[2 classification]--> Table S1, S4; Fig 4 data
        (data/raw, optional)                              (shareable)                           phase/3.78s/2s   |                          neural_predictions_early.csv
                                                                                                                 '--[3 habituation]-----> Section 1.7, Fig 5 data
                                                                                  results --[4 fig4, fig5]--> Fig 4A/B, Fig S3, Fig 5
full_respiration_EEG_humidity.npy --[5 pac_compute, slow]--> pac_results --[5 fig2]--> Fig 2, Section 1.1
                                    '--[6 psd_cache]--> psd_cache (few MB) --[6 fooof_gating]--> Section 1.2, Table S2
```

| Stage | Rule(s) | Output (paper item) | Replaces |
|---|---|---|---|
| 0. epochs | `make_epochs` | `epochs_..._v2.npy` | preprocessing notebook cell 2 |
| 0. recordings | `make_full_recordings`, `recordings` | `full_respiration_EEG_{v2,humidity}.npy` | preprocessing notebook cells 5, 17 |
| 1. ERSP | `ersp_phase_aligned`, `ersp_time_aligned` (long = 3.78 s, short = 2 s) | `results/ersp/*.npy` | `generate_ersp.py` (as is) |
| 2. classification | `classification` | **Table S1**, **Table S4**, Fig 4A/S3 data, `neural_predictions_early.csv` | analysis notebook cells 3, 4, 5, 12, 22, 26, 57, 59 |
| 3. habituation | `habituation` | **Section 1.7**, Fig 5 data | analysis notebook cells 76, 77, 81 |
| 4. figures | `fig4`, `fig5` | **Fig 4A/B, Fig S3, Fig 5** | analysis notebook cells 29, 31, 84 |
| 5. PAC | `pac_compute` (slow), `recordings_meta`, `fig2` | **Section 1.1, Fig 2** (whole-head z, FDR p, Cohen's d, cluster tests, topoplots) | analysis notebook cells 85-98, 105 |
| 6. FOOOF gating | `psd_cache`, `fooof_gating` | **Section 1.2, Table S2** (peak prevalence, gated PAC at 4 thresholds, inhale/exhale aperiodic) | analysis notebook cells 100-104 |
| 8. odor-arrival timing | `timing_inputs`, `timing` | **Section 1.8, SI Tables S5-S6** (lag by subject, flow-rate sensitivity, 703/708 trials) | analysis notebook cells 40, 42, 44, 48, 49 |
| 9. subject 28 | `subject28` | **SI 1.4** (why subject 28 loses trials; breathing rate vs. trial retention, Spearman) and **Table S7** (AUC with / without subject 28) | analysis notebook cells 8-9, 52-53, 58 |
| 10. cluster permutation | `cluster_perm` | **Section 1.5, Fig. S4** (theta x first half and alpha x second half cluster tests; 2-s, 3.78-s and phase-locked ERSP; wider-band sensitivity) | analysis notebook cells 117-128 |
| 11. Fig. 3 | `fig3` | **Fig. 3** (schematic, two example subjects, grand-average phase-locked and time-locked ERSPs with respiration overlays) | analysis notebook cell 60 |
| 12. Fig. 1 trace | `fig1_trial`, `fig1_resp` | **Fig. 1, right panel** (example nasal-humidity trace with inhale / exhale); the one-trial snippet file is deposited | analysis notebook cell 115 |
| 13. Fig. S2 | `figS2` | **SI Fig. S2** (example FOOOF fits with / without a peak in theta, alpha, beta, low gamma), drawn from the spectra cache | analysis notebook cell 106 |
| 14. free responses | `word_scoring` | **Fig. 4C/D, Section 1.6, SI 3.1** (GoodScents + word-embedding score of the free descriptions, comparison between odors, neural-behavior correlation); needs `sentence-transformers` | preprocessing notebook cells 21-27, 29-30 |
| 7. breath durations | `breath_stats` | **Section 1.3, Table S3** (first inhale/exhale per condition, repeated-measures ANOVA) | analysis notebook cells 34, 37, 46, 50 |

* **All parameters live in `config/config.yaml`** - including every per-subject hand correction that used to be buried
  in notebook cells. `results/*/params.yaml` records what a run used.
* **Self-check against the paper.** Each analysis stage writes a `*_checks.csv` comparing its numbers with the published
  ones (`config/paper_values.yaml`: all 27 AUCs of Table S1, Table S4, Fig 4A, Fig S3, Section 1.7). `checks.strict: true`
  makes a mismatch fail the run.
* **Code changes trigger re-runs**: each rule lists the scripts/modules it runs.
* **Everything is seeded.** `results/versions.txt` records the software versions of each run.
* Values that used to be typed in by hand are now computed: the best time-locked binning per odor, the habituation
  FDR-corrected p-values (notebook cell 77), the Fig S3 heatmap numbers (notebook cell 31).

### Code layout
```
Snakefile                  workflow definition
config/                    config.yaml, paper_values.yaml (published numbers), test.yaml (synthetic smoke test)
scripts/                   one script per workflow step (also runnable on their own); print_versions.py, compare_npy_dicts.py
src/olfaction_eeg/         recordings.py (raw -> epochs), features.py, classify.py (classifier + statistics),
                           figures.py (Fig 4A/B, S3, 5), pac.py / pac_stats.py / pac_figure.py (PAC, Fig 2), checks.py
envs/                      archival (original `mne` env) and word-embedding environments
tests/                     unit tests + synthetic data generators
```


* **All parameters live in `config/config.yaml`** - paths, subject lists, bin definitions, seeds, and every per-subject correction. `results/*/params.yaml`
  records what a run used.
* **Self-check against the paper.** Each analysis stage writes a `*_checks.csv` comparing its numbers with the published ones; `checks.strict: true`
  makes a mismatch fail the run.
* **Code changes trigger re-runs**: each rule lists the scripts/modules it runs. Expensive steps run only when their output file is missing.
* **Everything is seeded.** `results/versions.txt` records the software versions of each run.

## Implementation notes that matter when reading the paper
* **Two trial-quality criteria.** The breath durations of Table S3 and the example traces of Fig. 3 use a first-breath detection (first peak after odor onset,
  troughs before and after, >= 2 respiratory cycles, inhale and exhale 1-4 s); the ERSP analyses use the quality control described in the Methods (>= 3 cycles
  and an inhalation peak between -0.4 and 3 s).
* **Two permutation tests.** The decision-value gaps of Table S4 and the AUC differences of the habituation analysis use different permutation tests; each is implemented
  where the paper uses it.
* **PAC surrogates** use NumPy's global random generator, so results depend on the processing order; `pac.seed_mode: notebook` reproduces one sequential pass,
  `per_subject` allows parallel subjects with different (statistically equivalent) draws.
* **Free-response scoring** builds the odor profiles from GoodScents entries listed in `config/goodscents_selected.csv` (the first three name matches of each search term;
  the three source tables are pinned by SHA-256 in `config/goodscents_selected_source.json`), so no internet access is needed for that part.

## Reproduction status
The complete pipeline was run from the data deposit alone (no raw or continuous EEG), in a fresh environment built from `environment.yml` (macOS, Apple silicon, Python 3.13), and
**all 561 published numbers it checks were reproduced** (`python scripts/summarize_checks.py --results results`):

| stage | checks | covers |
|---|---|---|
| `classification` | 111 / 111 | Table S1, Table S4, Fig. 4A/B, Fig. S3 |
| `habituation` | 20 / 20 | Section 1.7, Fig. 5 |
| `pac` | 19 / 19 | Section 1.1, Fig. 2 |
| `fooof` | 55 / 55 + 16 / 16 | Section 1.2, Table S2; Fig. S2 |
| `breath` | 37 / 37 | Section 1.3, Table S3 |
| `timing` | 108 / 108 | Section 1.8, Tables S5-S6 |
| `subject28` | 107 / 107 | SI 1.4, Table S7 |
| `cluster_perm` | 13 / 13 | Section 1.5, Fig. S4 |
| `fig3` | 6 / 6 | Fig. 3 |
| `word_scoring` | 69 / 69 | Fig. 4C/D, Section 1.6, SI 3.1 |

This used the deposited ERSP files and the deposited PAC result; recomputing either from the epochs / continuous EEG agrees only approximately (see "What to expect" above).
On other platforms the FOOOF-based numbers of Section 1.2 can differ in the last digit (see "Environments").

## Verification of the code
All checks compare the refactored code with the **original notebook code run on identical inputs** (synthetic data; the
originals are the notebook cells / `generate_ersp.py`, executed with only file paths patched). They are not in the repo
because the notebooks are not.

| Check | Result |
|---|---|
| merged classifier vs `analysis.py` + the notebook's cell-22 copy; bootstrap, paired and pooled permutation tests (53 checks) | identical, incl. null distributions |
| `generate_ersp.py` (phase-aligned, 3.78 s, 2 s windows) | identical |
| epoching (cell 2), continuous recordings (cells 5, 17), incl. subjects 24/27/28/35 corrections and a NaN-humidity subject | identical, in old **and** modern stacks |
| Table S1: all 27 strategy AUCs vs the notebook's functions (cells 3/5/12/59 logic) | identical |
| Table S4 (n, observed difference, raw p, 90% null interval, FDR p) vs notebook cell 57 run verbatim | identical |
| Fig 4A matrices vs cell 29's `extract_phase_data` / `extract_time_data` | identical |
| Habituation: predictions, per-odor and pooled permutation tests, FDR (vs statsmodels), bootstrap CIs vs cells 76/81 | identical |
| Fig 4A/B vs cell 29, Fig S3 vs cell 31, Fig 5 vs cell 84 (rendered images) | **pixel-identical** |
| PAC computation: the notebook's commented-out driver loop (cell 89) vs `run_pac`, incl. every surrogate draw | identical |
| FOOOF peak gating, Table S2 sweep, gated-vs-all comparisons, inhale/exhale aperiodic tests (cells 100-104) from the spectra cache | identical |
| deposit-only regeneration: Fig 2 and Table S2 from the slim PAC file + spectra cache + metadata (no recordings) | identical to the full-data run |
| PAC group statistics (sign-flip p, FDR, Cohen's d), MNE cluster tests, `subject_info` montage (cells 93-95), Fig 2 vs cells 96/105 | identical; Fig 2 **pixel-identical** |
| Section 1.5: cluster tests (all clusters, masses, p-values, t-maps, labels) of the 2-s, 3.78-s and phase-locked ERSPs vs notebook cells 117-127; Fig. S4 vs cell 128 | identical; Fig. S4 **pixel-identical** |
| Fig. 3 (every panel) vs notebook cell 60; Fig. 1 example trace (extracted trials and plot) vs cell 115 on synthetic raw files | trial arrays identical; both figures **pixel-identical** |
| Fig. S2 (cell 106: raw EEG -> Welch spectrum -> FOOOF -> plot) vs the cache-based module, incl. the two panels chosen by search | **pixel-identical** |
| Free-response scoring (GoodScents search -> empirical profiles -> synonym expansion -> combined profiles -> scores of all 51 responses), permutation tests and Fig. 4C/D vs notebook cells 21-27, with a deterministic stand-in for the embedding model | identical; Fig. 4C/D **pixel-identical** |
| `pytest tests` | passes |

What synthetic data cannot show: that these reproduce the numbers on the *real* recordings. That is what the `*_checks.csv`
files and `scripts/compare_npy_dicts.py` are for.


## Environments
`environment.yml` follows the Python 3.13 environment the main analysis was run in. A single modern environment is enough for every stage: on synthetic data the
stage-0 code gives bit-identical output under the old stack (numpy 1.26 / MNE 1.8 / pandas 2.1) and a modern one (numpy 2.4 / MNE 1.13 / pandas 3.0).
`envs/original_epoching_env.yml` is kept for the record; `envs/embeddings.yml` is a stand-alone environment for the word-embedding stage only.
`fooof` is deprecated in favour of `specparam`; the version used for the paper is pinned.
The scientific packages are installed with **pip** on purpose: with the conda-forge builds of numpy and scipy (same version numbers), FOOOF's fits differ in the 5th-6th digit,
which flips two borderline channels in the oscillation gating of Section 1.2 and moves one reported number (beta gated-vs-all paired p: 0.282 instead of 0.288; nothing else changed).
On other platforms (e.g. Linux with OpenBLAS) a few FOOOF-based numbers may likewise differ in the last digit; `summarize_checks.py` lists every deviation.

## Repository layout
```
Snakefile            the workflow (stages 0-14)       config/        parameters, paper values, example configs (deposit.yaml, test.yaml, ...)
src/olfaction_eeg/   analysis modules                  scripts/       command-line entry points of the stages
tests/               unit tests + synthetic data       data/, results/, figures/   inputs and generated outputs (not committed)
DEPOSIT.md           the data deposit                  CITATION.cff   how to cite
```

## Citation, license, contact
Please cite the paper (DOI `TODO`) and the data deposit (DOI `TODO`); see `CITATION.cff`. License: `TODO`. Contact: alisayh@stanford.edu (analysis),
toddcol@stanford.edu (correspondence).
