# Mölle et al. (2011), written from the paper

`run.py` implements the fast-spindle detector of Mölle, Bergmann, Marshall & Born (2011), Sleep 34(10),
from its Methods section (`@molle2011` in `docs/literature/references.bib`). It is a check on
wonambi's implementation (`wonambi_moelle2011`), which uses a fixed 12–15 Hz band and wonambi's own
threshold base. Runs in the main `.venv` (numpy, scipy, pandas).

Recipe:
1. Individual fast-spindle peak: the frequency with the largest Welch power in 12–15 Hz.
2. Band-pass at peak ± 1.5 Hz (FIR, 1-s taps, zero phase).
3. RMS in a sliding 200-ms window, then smoothed by a 200-ms moving average.
4. Threshold: 1.5 × SD of the band-passed signal (factor `k`, recalibrated in `experiments/rq4_adaptation/adaptation_moelle.py`).
5. A spindle is a run of the smoothed RMS above threshold lasting 0.5–3 s.

The paper takes the peak and the SD over NonREM sleep. Two ids are written:

| id | peak and SD taken over | ranked? |
|---|---|---|
| `moelle2011_faithful` | whole recording | yes (stage-free like every ranked detector) |
| `moelle2011_faithful_nrem` | expert-scored N2 + N3 (whole recording if < 1 min) | no: uses expert staging |

Results (N2, main protocol): DREAMS 0.60 / 0.62 (stage-free / NREM baseline), NAP 0.34 / 0.34,
against 0.63 and 0.38 for wonambi's implementation. Implementation choices for one published method
move F1 by 0.03–0.04.
