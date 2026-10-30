# Protocol decisions

Facts about the datasets that change F1 if handled differently, and the choice made for each.
The scoring protocols themselves are code: `spindlebench.protocols.benchmark` (`benchmark.v1`) and
`spindlebench.protocols.nap_points` (`nap_points.v1`). Changes are logged in `CHANGELOG.md`.

**DREAMS**
- **Expert 1 stopped scoring at about 1000 s in all 8 excerpts.** The last marks fall between 878 and 996 s. Every comparison against expert 1 is therefore scored on [0, 1000) s. Expert 2 alone is scored on the full 1800 s.
- **The hypnograms use Rechtschaffen & Kales codes in 5-second epochs.** The codes are 5 W, 4 REM, 3 S1, 2 S2, 1 S3, 0 S4. This was checked against band power: code 0 has the most delta and code 3 the most theta/alpha.
- **N2 scope:** an event counts if its midpoint falls in an N2 epoch.
- **Matching:** onset within ±0.5 s, or IoU ≥ 0.2. Expert 2 always gives a fixed 1 s duration, so onset is the fairer rule against them.
- **Excerpt 4:** the hypnogram scores the first ~340 s as wake, while expert 1 and YASA both find dense spindles there. The staging in that excerpt is suspect.

**NAP EEG** (Mei, Grossberg, Ng, Navarro & Ellmore; OSF `chav7`)
- **Coverage:** there are 40 raw recordings and 46 annotation files. 36 recordings have both. 30 remain after the exclusions listed in the Methods of Mei et al. 2017 (`mei2017spindles` in the reference list; subjects 26 and 27, subject 12 day 1, subject 20 day 2).
- **One expert per recording.** There is no second scorer, so there's no inter-rater ceiling like on DREAMS.
- **The expert marks sit inside the spindle, near its centre, not at its onset.** The sigma envelope peaks about 0.1 s after each mark. The median offset is −0.37 s from the YASA onset and +0.03 s from the YASA centre (`results/nap/annotation_alignment.png`). The benchmark protocol (`benchmark.v1`) therefore matches each mark against the centre of the detection, within ±0.5 s; onset matching and tolerances of 1.0 and 1.5 s are also reported. Since 2026-10-28 the primary NAP score is `nap_points.v1`: centre tolerance swept over 0.25–1.5 s with the integrated F1, affiliation precision/recall (Huet et al. 2022), a containment check and a random baseline (`experiments/nap_scoring/`).
- **Signal:** the actiCHamp amplifier is DC-coupled, so the data is band-passed (0.3–40 Hz in `experiments/yasa_sweeps/nap_pipeline.py`; 0.1–45 Hz in the shared cache used by every other detector). It is then re-referenced to linked mastoids (x − TP10/2; TP9 was the recording reference) and resampled to 200 Hz.
- **Label cleanup:** labels were typed by hand in several spellings ("Marker:Markon:2", "Makon: 2", "spind;e", "K-Complex"). The parser handles all but 3 junk labels.
