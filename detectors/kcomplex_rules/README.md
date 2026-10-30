# Rule-based K-complex detectors

> Written from the code after the adapter agent was cut off by a network error.

No published rule-based K-complex detector runs natively in Python: Spinky and DETOKS are MATLAB
(`docs/literature/detectors_review.md`). This folder provides two.

## `spinky_kc`: Python port of Spinky's K-complex detector

**Source.** Lajnef et al. 2017, *Front. Neuroinform.* 11:15. Code: https://github.com/TarekLaj/SPINKY
(commit a2cb3293, GPL-3.0), which bundles Selesnick's TQWT toolbox. `spinky.py` ports the K-complex
path, with this pipeline for each 30-s epoch:

1. Zero-phase FIR band-pass 0.2–45 Hz (fir1, order 105, filtfilt).
2. Morphological component analysis with the dual-Q TQWT: Q1 = 5.5, r1 = 3, Q2 = 1, r2 = 3,
   θ = 0.5, µ = 0.005, 500 iterations. This splits the signal into an oscillatory and a
   transient component; K-complexes come from the transient one.
3. The transient component is demeaned and filtered with a causal FIR band-pass at 0.8–4 Hz
   (order 100). Its group delay of 50 samples is subtracted from the event onsets.
4. Troughs count as K-complexes when they are:
   - at least 1.5 s apart
   - at or below the threshold
   - more than 1 s into the epoch
   - deeper than 2.5 × the filtered value 1 s earlier
5. Each event's extent comes from `kc_extent`: from the zero crossing before the trough to the end
   of the following positive half-wave. This onset and duration convention is our addition;
   Spinky itself reports only a position.

**Implementation.** The TQWT is computed with precomputed frequency-domain gather maps,
vectorised over epochs. `check_spinky.py` checks the port against a literal transcription of the
MATLAB `afb`/`sfb`/`lps` cascade (forward transform, perfect reconstruction, wavelet norms).
**It has not been checked against MATLAB output on real data.** MATLAB R2025b is installed on this
machine, so that check is possible; it is still TODO.

**Threshold.** Spinky has no fixed default threshold. The paper recommends a per-subject ROC
threshold from about 10 minutes of expert-scored pages; its manual-entry example is −60. We use a
fixed **−35 µV** for every recording and do not tune it on our labels. This value is an author
choice: the paper's per-subject thresholds (its Fig. 5B) appear only in the figure, so "centre of
the per-subject optimal thresholds" could not be checked against the text. A −60 µV sensitivity run
has not been done. Per-subject calibration is the RQ4 experiment, not the benchmark condition.

## `yasa_sw_kc`: YASA slow-wave detector as a K-complex proxy

This is **not a published K-complex detector**. It is YASA's `sw_detect` with its published
defaults:

- 0.3–1.5 Hz band
- negative peak 40–200 µV
- positive peak 10–150 µV
- peak-to-peak 75–350 µV
- negative phase 0.3–1.5 s, positive phase 0.1–1 s

Its 75 µV peak-to-peak lower bound is the AASM slow-wave (N3) amplitude criterion; the AASM
K-complex definition has no amplitude threshold (TODO(verify) against the manual). Every detected
negative-positive wave is written as a `kcomplex`. It serves as a transparent amplitude-rule
baseline.

## Running

Both detectors run on the whole recording without staging:

```
.venv/Scripts/python detectors/kcomplex_rules/run.py [--datasets dreams_kc,nap] [--channels C3,...] [--detectors spinky_kc,yasa_sw_kc] [--workers N]
```

**Coverage.** DREAMS K-complexes: all 10 excerpts. NAP: all 36 naps; `spinky_kc` on C3 only,
`yasa_sw_kc` on C3, Cz, F3 and F4.
