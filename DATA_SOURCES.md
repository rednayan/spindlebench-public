# Data sources

No raw data are stored in this repository. Each dataset is downloaded by the user from its source
and keeps its own licence. This page gives:
- the licence and the citations each source requires;
- what this repository shares from it.

Licence readings for derived outputs are ours, not legal advice. Open questions are marked.

| dataset | source | licence | download |
|---|---|---|---|
| DREAMS Spindles / K-complexes | [Zenodo 2650142](https://zenodo.org/records/2650142) (Devuyst et al., Univ. Mons TCTS / ULB–CHU Charleroi) | CC BY-NC-ND 3.0 Unported | manual: unzip into `<data>/DatabaseSpindles`, `<data>/DatabaseKcomplexes` |
| NAP EEG | [OSF chav7](https://osf.io/chav7/) (Mei et al.) | BSD 3-Clause on the OSF project (whether it is meant to cover the EEG files is unconfirmed) | `spindlebench download nap` |
| MASS SS2 annotations | [Borealis doi:10.5683/SP3/Y889CS](https://doi.org/10.5683/SP3/Y889CS) | CC0, with a requirement to cite O'Reilly et al. 2014 | manual: `<data>/MASS_data/doi-10.5683-sp3-y889cs` |
| MASS SS2 signals | Borealis doi:10.5683/SP3/K26LXJ | restricted (request + ethics approval) | not used |
| BOAS | [OpenNeuro ds005555](https://doi.org/10.18112/openneuro.ds005555.v1.1.3) | CC0 | `spindlebench download boas` |
| HMC | [PhysioNet hmc-sleep-staging 1.1](https://doi.org/10.13026/t79q-fr32) | CC BY 4.0 | `spindlebench download hmc` |
| Sleep-EDF Expanded | [PhysioNet sleep-edfx 1.0.0](https://physionet.org/content/sleep-edfx/1.0.0/) | Open Data Commons Attribution 1.0 | `spindlebench download sleepedf` |

## Required citations

- **DREAMS:** Devuyst S., Dutoit T., Stenuit P., Kerkhofs M. (2011), automatic sleep spindle detection,
  EMBC; the DREAMS databases on Zenodo (record 2650142).
- **NAP EEG:** Mei N., Grossberg M. D., Ng K., Navarro K. T., Ellmore T. M. (2018), *Data in Brief* 18, 1513–1519;
  and Mei et al. (2017).
- **MASS:** O'Reilly C., Gosselin N., Carrier J., Nielsen T. (2014), *J Sleep Res* 23(6), 628–635,
  doi:10.1111/jsr.12169.
- **BOAS:** the OpenNeuro dataset ds005555 v1.1.3 (doi:10.18112/openneuro.ds005555.v1.1.3) and its paper.
- **HMC:** Alvarez-Estevez D., Rijsman R. M. (2021), *PLoS ONE* 16(8): e0256111; the PhysioNet record
  doi:10.13026/t79q-fr32; Goldberger et al. (2000), PhysioNet.
- **Sleep-EDF Expanded:** Kemp B. et al. (2000), *IEEE Trans Biomed Eng* 47(9), 1185–1194; Goldberger et al. (2000).

## What this repository shares

| source | aggregate scores | detector outputs | expert-annotation tables | signals (incl. injected) |
|---|---|---|---|---|
| BOAS, HMC, Sleep-EDF | yes | yes | yes | SpindleInject frozen set, with attribution |
| MASS SS2 annotations | yes | — | yes | no (signals restricted) |
| NAP | yes | yes (BSD 3-Clause notice and citation kept) | once the licence scope is confirmed | recipe only until confirmed |
| DREAMS | yes | **no**: rebuilt locally (and never `dreams_auto`, which is part of the database) | **no**: rebuilt locally from the user's own download | **no**: recipe only |

DREAMS's no-derivatives clause forbids publishing adaptations of the database. The repository therefore
ships code that rebuilds DREAMS-derived tables from each user's own copy, never the tables themselves:
detector outputs, per-event tables and per-sample labels on DREAMS are left out, and mixed tables keep
only their rows from other datasets (`results/NOT_INCLUDED.md` lists what is left out).
For SpindleInject on DREAMS backgrounds it ships only:
- the injection event table;
- the generator;
- SHA-256 checksums of the inputs and outputs.

Users with DREAMS access regenerate the data and can verify it against those checksums.

## Detector weights

Learned detectors are run with their authors' weights, which are not redistributed here (see each
`detectors/<name>/manifest.json`).
- **SUMO / SUMOv2 and OpenSpindleNet:** MIT.
- **SSSM:** declares no licence, so its weights are used for benchmarking only.
