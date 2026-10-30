# Wonambi spindle detectors

> Written from the code after the adapter agent was cut off by a network error. The
> sanity-check numbers come from `results/benchmark/summary.csv` (regenerate with `experiments/rq1_benchmark/evaluate.py`).

**Source.** [wonambi](https://github.com/wonambi-python/wonambi) 7.17 (`wonambi.detect.DetectSpindle`),
GPL-3.0 (PyPI; the repository LICENSE file says BSD-3). Cite as `@wonambi` in
`docs/literature/references.bib`. It contains wonambi's own Python implementations of 11 published
rule-based spindle methods. Each runs with **wonambi 7.17's preset values for that method, unstaged on
the whole recording**. These presets are wonambi's interpretation of each paper, not the authors'
published code or defaults. For Moelle2011, for example, the paper (`@molle2011`) uses an individual
peak ±1.5 Hz band and a 1.5 SD threshold on NREM, while wonambi presets a fixed 12–15 Hz band. Label
results as "wonambi's implementation of X", e.g. "Mölle 2011 (wonambi)". None is tuned on our labels.

| detector id | method |
|---|---|
| `wonambi_ferrarelli2007` | Ferrarelli et al. 2007 |
| `wonambi_moelle2011` | Mölle et al. 2011 (`@molle2011`) |
| `wonambi_nir2011` | Nir et al. 2011 |
| `wonambi_wamsley2012` | Wamsley et al. 2012 |
| `wonambi_martin2013` | Martin et al. 2013 |
| `wonambi_ray2015` | Ray et al. 2015 |
| `wonambi_lacourse2018` | Lacourse et al. 2018/2019 (A7), wonambi's port; compare with `detectors/a7` |
| `wonambi_fasst`, `wonambi_fasst2` | FASST toolbox variants |
| `wonambi_ucsd` | UCSD method |
| `wonambi_concordia` | Concordia method |

**Environment.** Own uv env (Python 3.12, wonambi 7.17, scipy 1.12, numpy 1.26). wonambi imports
`scipy.signal.tukey`, which was removed in scipy 1.13, so it cannot run in the main `.venv`.

```
detectors/wonambi/.venv/Scripts/python detectors/wonambi/run.py [--datasets dreams_spindles,nap] [--channels C3,...] [--methods Ferrarelli2007,...]
```

**Input handling.**

- The cached µV signal at its native rate, run on the whole recording without staging (as for
  `yasa_default`).
- DREAMS excerpt 3 (50 Hz) is upsampled to 100 Hz first, so that no method's filter exceeds
  Nyquist (Ray2015 band-passes 0.3–35 Hz).
- If a method raises on a recording, the error is printed and an empty event list is written for it.

**Coverage.**

- DREAMS spindles: all 8 excerpts.
- NAP: all 36 naps on C3, plus some extra channels (F3, F4, O1, Cz in the files checked).

**Caveats.**

- Some of these methods were designed for whole-night N2/N3 data on specific montages. Running them
  unstaged on 30-min clinical excerpts and naps is part of the benchmark condition, not the
  authors' intended use.
- Events can exceed a method's duration limit. wonambi checks duration before a second
  `_merge_close(min_interval)`, so merged events can be longer than the cap. With
  `duration=(0.4, 1.3)`, 7.3% of `wonambi_fasst` events on DREAMS and 11.5% on NAP C3 are longer
  than 1.3 s (up to 4.75 s and 12.06 s); for `wonambi_fasst2` it is 12.5% / 17.5% (up to 12.4 s). This is
  wonambi's behaviour and is left as is; no post-merge duration cap is applied.
