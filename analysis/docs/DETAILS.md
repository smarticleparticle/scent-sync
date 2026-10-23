# Details

Notes for people who want to look under the hood. None of this is needed to reproduce the paper: see the README.

## How the analysis is organized
Each step is a Snakemake rule (`Snakefile`); settings are in `config/config.yaml`, the published numbers each stage is compared with in `config/paper_values.yaml`.

| stage | what it does | paper items |
|---|---|---|
| ERSP | time-frequency maps of the epochs, aligned to the respiratory phase or to time | input of the classification stages |
| classification | odor vs. water decoding with leave-one-subject-out validation | Tables S1, S4; Fig. 4A/B, S3 |
| habituation | early vs. late trials | Section 1.7, Fig. 5 |
| PAC, FOOOF gating | respiration-EEG coupling, and whether it reflects an oscillation | Fig. 2, Sections 1.1-1.2, Table S2, Fig. S2 |
| breath durations, participant 28 | respiration timing checks | Tables S3, S7, Section 1.3, SI 1.4 |
| cluster tests | theta and alpha clusters in the ERSPs | Section 1.5, Fig. S4 |
| free-response scoring | how well participants described each odor, with a language model | Fig. 4C/D, Section 1.6, SI 3.1 |
| figures | Figs. 1, 3 and the rest | |

## Not in the public deposit
The odor-arrival timing check (Section 1.8, Tables S5-S6) measures, for each trial, the time between the start of exhalation (which triggered the odor valve) and the next inhalation, using the whole-session respiration
recordings. These recordings are available on request; the code is `scripts/run_timing.py` and is switched on with `timing: run: true`.

## Recomputing the ERSP and PAC results
The deposit includes the ERSP and PAC results behind the paper. Both can be recomputed with this code (`scripts/generate_ersp.py` from the epochs; the PAC step from the continuous EEG, which is not deposited),
but the numbers then agree only closely, not exactly: recomputed ERSPs differ slightly (power within about 3%, classification AUCs within about 0.004), and the PAC step uses random time shifts that cannot be reproduced
bit for bit. The conclusions are unchanged. Use the deposited files to reproduce the printed numbers exactly.

## Environment
`environment.yml` installs the scientific packages with pip on purpose: with the conda-forge builds of numpy and scipy, FOOOF's peak detection differs in the last digits and one reported p-value
(gated vs. all-channel PAC, beta) moves from 0.288 to 0.282. On other platforms a few FOOOF-based numbers may likewise differ in the last digit; `summarize_checks.py` lists every deviation.

## Starting from the raw data
Stage 0 (`make_epochs`, `make_full_recordings`) rebuilds the epochs and continuous recordings from the EEGLAB-preprocessed files and the olfactometer CSVs (not deposited; available on request); see `data/README.md`.
The rebuilt epochs file is not identical to the one the paper's ERSPs were computed from, so use the deposited file for the published numbers.

## Fig. 1
`figure_sources/figure1_left_infographic.svg` is the editable drawing of the olfactometer and participant (all text and shapes are outlines). The `.pdf` next to it is its export (Inkscape: File > Save a Copy > PDF),
which `scripts/make_fig1.py` places unchanged beside the respiration trace; re-export it if you edit the drawing. `figure1_left_infographic_live_text.svg` is an earlier version with editable text.

## Repository layout
```
Snakefile            the workflow                      config/        settings, published values, example configs (deposit.yaml, test.yaml, ...)
src/olfaction_eeg/   analysis modules                  scripts/       command-line entry points of the stages
tests/               unit tests, synthetic data        figure_sources/  the illustration of Fig. 1
DEPOSIT.md           the data deposit                  data/, results/, figures/   inputs and outputs (not committed)
```
