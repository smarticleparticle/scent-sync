# scent-sync

An open-source, low-cost olfactometer that delivers odors in step with natural breathing, together with the EEG and respiration analysis code of

> Hathaway AY, Coleman TP. *Respiratory Phase Orchestrates Human Olfactory Cortical Dynamics.* PNAS (2026), in press.

An Arduino reads nasal humidity, detects the start of each breath, and releases the odor so that it reaches the nose at the next inhalation, without disturbing normal breathing. A TTL pulse marks the moment in the EEG recording.


## What is in this repository

| folder | contents |
|---|---|
| `hardware/` | CAD files (STL) for the printed parts, the wiring schematic, and the bill of materials |
| `firmware/` | the Arduino sketch |
| `documentation/` | the build guide |
| `analysis/` | the analysis code that reproduces the paper's tables and figures from the data deposit (start with [`analysis/README.md`](analysis/README.md)) |

## Data

The preprocessed EEG and respiration data (17 participants) are deposited in the Stanford Digital Repository: https://doi.org/10.25740/gs394xq8865 (license CC BY-NC 4.0). The raw EEG is available on reasonable request.

## License and citation

The hardware, firmware and documentation are covered by the MIT license in `LICENSE`; the analysis code by `analysis/LICENSE`. The data deposit carries the license shown on its repository record.
To cite this work, see `CITATION.cff`.
