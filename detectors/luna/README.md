# Luna SPINDLES (detector id `luna_spindles`)

Purcell S. M. et al. *Characterizing sleep spindles in 11,630 individuals from the National Sleep
Research Resource.* Nat Commun 8, 15930 (2017). doi:10.1038/ncomms15930. Software: Luna,
[remnrem/luna-base](https://github.com/remnrem/luna-base), called through the Python bindings
[lunapi](https://pypi.org/project/lunapi/) 1.7.0. GPL-3.0, so this adapter is GPL-3.0 as well.

```
cd detectors/luna && uv sync          # Python 3.12, lunapi 1.7.0 (Windows wheels on PyPI)
cd ../.. && detectors/luna/.venv/Scripts/python detectors/luna/run.py \
    --datasets dreams_spindles,nap --channels C3,F3,F4,C4,O1,O2,Cz
```

## Settings
`SPINDLES sig=EEG per-spindle`, i.e. the program defaults. Read from `spindles/spindles.cpp` in
luna-base, because the documentation site failed its TLS check on 2026-10-30:
- one complex Morlet wavelet at **13.5 Hz**, 7 cycles;
- core threshold **4.5 x the whole-signal mean** of the 0.1 s smoothed coefficient (`th`), flanks at 2 x (`th2`);
- duration 0.5-3 s; spindles closer than 0.5 s merged; Luna's own spindle QC (failed spindles are removed).

Purcell et al. ran two wavelets (11 and 15 Hz) on whole-night N2. The single default wavelet is used
here so the row is the program as shipped. Nothing was tuned.

## Input
The cached signal in µV at its native rate, trimmed to whole seconds (Luna stores 1 s EDF records).
Stage-free: no epoch mask, like every stage-free row. The threshold is relative to the whole-signal
mean, so on 30-min DREAMS excerpts and naps it is set by a shorter, wake-containing record than the
whole nights it was designed for.

## Runtime (14-core CPU)
DREAMS (8 excerpts, 1 channel) plus NAP (36 naps x 7 channels): about 5 min.

## Sanity check (pooled F1, N2, `experiments/rq1_benchmark/evaluate.py`)
| reference | protocol | P | R | F1 |
|---|---|---|---|---|
| DREAMS expert 1 | onset ±0.5 s | 0.76 | 0.52 | 0.62 |
| DREAMS expert 2 | onset ±0.5 s | 0.75 | 0.63 | 0.69 |
| NAP expert | centre ±0.5 s | 0.47 | 0.27 | 0.34 |

Above YASA defaults (0.42 on DREAMS), so units and timing are right.
