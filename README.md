# scent-sync

Open-source, respiration-synchronized olfactometer for precise scent delivery in neuroscience research, with the EEG / respiration analysis code of

> Hathaway AY, Coleman TP. *Respiratory Phase Orchestrates Human Olfactory Cortical Dynamics.* PNAS (2026).

## What is here
| folder | contents |
|---|---|
| `hardware/` | CAD files (STL), schematic, bill of materials |
| `firmware/` | Arduino sketch (`olfactometer-sync-2026.ino`) |
| `documentation/` | build guide |
| `analysis/` | analysis code that reproduces the paper's tables and figures from the data deposit (start with [`analysis/README.md`](analysis/README.md)) |

## License and citation
Hardware, firmware and documentation: MIT (see `LICENSE`). Analysis code: see `analysis/LICENSE`. The data deposit has its own license (CC BY-NC 4.0). To cite, see `CITATION.cff`.
