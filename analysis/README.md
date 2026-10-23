# Respiratory phase and human olfactory cortical dynamics: analysis code

Code to reproduce the analyses, tables and figures of

> Hathaway AY, Coleman TP. *Respiratory Phase Orchestrates Human Olfactory Cortical Dynamics.* PNAS (2026).

The analysis is a [Snakemake](https://snakemake.readthedocs.io) workflow. Each stage compares its results with the numbers printed in the paper, and `scripts/summarize_checks.py` reports how many matched.

## Quick start

```bash
# 1. environment (Python 3.13)
conda env create -f environment.yml && conda activate olfaction-eeg

# 2. data: download the deposit (Stanford Digital Repository: https://doi.org/10.25740/gs394xq8865) into data/preprocessed/ and copy its three ERSP files into results/ersp/
python scripts/verify_data.py --manifest data/preprocessed/MANIFEST.sha256 --root data/preprocessed

# 3. run everything and check it against the paper
snakemake --configfile config/deposit.yaml -c4 --keep-going
python scripts/summarize_checks.py --results results
```
Figures are written as PDFs to `figures/`, tables and statistics to `results/`. The last stage (free-response scoring) downloads a language model once (about 420 MB);
to skip it, add `config/no_word_scoring.yaml` after `config/deposit.yaml` in the second command.

## Data
The raw EEG is not included. The deposit holds the preprocessed EEG and respiration of the 17 participants analyzed in the paper, plus the derived results needed to regenerate every figure and table
(contents and privacy: [DEPOSIT.md](DEPOSIT.md)).

## Figures
| paper | file in `figures/` |
|---|---|
| Fig. 1 | `fig1.pdf` (the illustration in `figure_sources/` plus the respiration trace) |
| Fig. 2 | `fig2_pac.pdf` |
| Fig. 3 | `fig3_ersps.pdf` |
| Fig. 4 | `fig4_AB_classification.pdf`, `fig4CD_recognition.pdf` |
| Fig. 5 | `fig5_habituation.pdf` |
| Figs. S2-S4 | `figS2_fooof_examples.pdf`, `figS3_per_odor_selection.pdf`, `figS4_cluster_perm.pdf` |

## Testing
```bash
pytest -q tests                                                  # unit tests
python tests/make_synthetic_raw.py tests/data/raw tests/data/preprocessed/info --n-trials 9
snakemake --configfile config/test.yaml -c3                      # whole pipeline on synthetic data
```

## More
* Olfactometer hardware and firmware: the `hardware/` and `firmware/` folders of this repository.
* How the analysis is organized and how to recompute the intermediate files: [docs/DETAILS.md](docs/DETAILS.md).

## Citation, license, contact
See `CITATION.cff` and `LICENSE`. Contact: alisayh@stanford.edu.
