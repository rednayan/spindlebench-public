# How this project fits together

A map of the whole project for someone seeing it for the first time: where the data come from, what
happens to them, which detectors run, how they are scored, and how the label-free student is built and
tested. Every box names the file or folder that does the work, so each diagram can be followed into
the code.

For a gentler, illustrated walk through the same story (real EEG, the teachers' actual votes, the measured
numbers), open [`docs/visual_guide.html`](docs/visual_guide.html) in a browser first.

Read it top to bottom. Part A is the benchmark (existing detectors scored against human experts).
Part B is the student, a new detector trained without any expert labels and then scored with Part A's
code. The full technical specification of the student (equations, tensor shapes, parameter counts) is
in [`docs/student_architecture.md`](docs/student_architecture.md).

**Contents**
1. [The project in one picture](#1-the-project-in-one-picture)
2. [Datasets and the cache](#2-datasets-and-the-cache)
3. [Part A: the benchmark](#3-part-a-the-benchmark)
4. [Part B: the label-free student](#4-part-b-the-label-free-student)
5. [Where things live](#5-where-things-live)
6. [Status and key numbers](#6-status-and-key-numbers)
7. [Glossary](#7-glossary)

---

## 1. The project in one picture

**What a sleep spindle is.** A short burst (0.3–2.5 s) of 11–16 Hz waxing-waning activity in the EEG
during light (N2) sleep. A spindle *detector* takes a night of EEG and returns a list of events
(onset, duration). Experts mark spindles by eye, and they disagree with each other.

**The two questions.**
- **Part A (benchmark):** how good are existing detectors, and how much does the answer depend on
  which expert, which matching rule and which sleep stages you score against?
- **Part B (student):** can a network trained only on the agreement of four rule-based detectors, on
  nights no expert ever marked, match detectors trained on expert labels?

```mermaid
flowchart LR
  raw[("Public sleep EEG<br/>DREAMS · NAP · MASS · BOAS<br/>HMC · CinC 2018 · Sleep-EDF")]
  cache[("Standard cache<br/>cache/manifest.csv<br/>signals + labels")]
  raw -->|"spindlebench download / cache"| cache

  subgraph A["Part A · Benchmark"]
    det["~30 existing detectors<br/>detectors/&lt;name&gt;/run.py"]
    score["One scoring code<br/>benchmark.v1 · nap_points.v1"]
    rq["RQ1–RQ4 results<br/>results/benchmark, validity, ..."]
    det --> score --> rq
  end

  subgraph B["Part B · Label-free student"]
    teach["4 rule teachers on<br/>401 unlabelled nights"]
    targ["Consensus targets<br/>no expert labels"]
    stud["Student network<br/>experiments/student/"]
    teach --> targ --> stud
  end

  cache --> det
  cache --> teach
  stud -->|"same detection format"| score
  experts[("Expert labels<br/>DREAMS · NAP only")] --> score
```

Part B never sees an expert label. Its only contact with the expert-labelled datasets is at the very
end, when its detections go through the same scoring code as every other detector.

---

## 2. Datasets and the cache

Each dataset has a role. Experts marked spindles only on DREAMS and NAP, so those two are the
**gold sets**: they are used for scoring and never for training the student. The others have no
spindle labels (only sleep stages) and serve as unlabelled material.

```mermaid
flowchart TB
  subgraph gold["Gold sets · expert spindle labels · scoring only"]
    dreams["DREAMS spindles<br/>8 excerpts × 30 min<br/>expert 1 + expert 2"]
    dkc["DREAMS K-complexes<br/>10 excerpts"]
    nap["NAP EEG<br/>36 naps, 17 people<br/>1 expert, point marks"]
  end
  subgraph ref["Reference annotations"]
    mass["MASS SS2<br/>annotations only<br/>(inter-rater ceiling)"]
  end
  subgraph free["Unlabelled · sleep stages only"]
    boas["BOAS · 20 nights"]
    hmc["HMC · 151 nights · 1,144 h"]
    cinc["CinC 2018 subset<br/>250 nights · 1,925 h"]
    sedf["Sleep-EDF<br/>(pilot only: Fpz-Cz)"]
  end

  gold & ref & free -->|"src/spindlebench/data/download_*.py"| rawdir[("raw files in datasets/<br/>DatabaseSpindles/, NapEEG/, HMC/, CINC2018/, ...")]
  rawdir -->|"spindlebench cache<br/>data/export_cache.py"| c

  subgraph c["cache/ · one format for every detector"]
    man["manifest.csv<br/>dataset, rec, subject, sf,<br/>default_channel, windows, excluded"]
    sig["signals/&lt;ds&gt;/&lt;rec&gt;.npz<br/>data [channels × samples] µV, sf, ch_names"]
    lab["labels/&lt;ds&gt;/&lt;rec&gt;__hypno.csv<br/>labels/&lt;ds&gt;/&lt;rec&gt;__expert1__spindle.csv<br/>labels/&lt;ds&gt;/&lt;rec&gt;__artefact.csv"]
  end
```

Facts about the data that change scores if handled differently (expert 1 on DREAMS stops at 1,000 s;
DREAMS hypnograms use R&K codes in 5-s epochs; NAP marks sit at the spindle's centre, not its onset)
are listed in [`docs/protocol.md`](docs/protocol.md).

---

## 3. Part A: the benchmark

### 3.1 Detector pass

Every detector, rule-based or learned, is wrapped in an **adapter** that reads the cache and writes one
CSV per recording in the same format. Learned detectors run in their own Python environments because
their dependencies conflict.

```mermaid
flowchart LR
  cache[("cache/")] --> common["detectors/common.py<br/>iterate recordings,<br/>default channel,<br/>write + provenance"]

  subgraph rules["Rule-based"]
    a7["a7 · A7 (Lacourse 2019)"]
    yasa["yasa · YASA default / tuned"]
    won["wonambi · 12 methods<br/>(Mölle, UCSD, Lacourse, ...)"]
    luna["luna · Luna SPINDLES"]
    mo["moelle · faithful Mölle 2011"]
  end
  subgraph learned["Learned (own environments)"]
    sumo["sumo · SUMO, SUMOv2"]
    s5["s5 · S5"]
    osn["openspindlenet"]
    sssm["sssm"]
  end

  common --> rules & learned
  rules & learned --> out[("results/detections/&lt;ds&gt;/&lt;id&gt;/&lt;rec&gt;.csv<br/>onset, duration, channel, event, score")]
  out --> chk["tests/test_detectors.py<br/>checks every file against the contract<br/>(src/spindlebench/detections.py)"]
```

### 3.2 Scoring

One function scores every detector against every reference under every protocol, so a ranking can be
checked for how much it depends on those choices.

```mermaid
flowchart TB
  det[("detections")] --> clip["clip to the reference's window<br/>(DREAMS expert 1: first 1,000 s)"]
  refs[("expert labels<br/>expert1, expert2,<br/>union, intersection")] --> clip
  clip --> scope{"scope"}
  scope -->|"all"| m
  scope -->|"N2: event midpoint<br/>in an expert N2 epoch"| m
  m["matching rule × tolerance<br/>onset ±0.25/0.5/1/1.5 s<br/>centre ±0.25/0.5/1/1.5 s<br/>IoU ≥ 0.1/0.2/0.3/0.5"]
  m --> hung["one-to-one maximum matching<br/>(Hungarian)<br/>metrics/events.py"]
  hung --> f1["TP · FP · FN → precision, recall, F1<br/>per recording and pooled"]
  f1 --> main["primary protocol<br/>DREAMS: onset ±0.5 s vs expert 1, N2<br/>NAP: centre ±0.5 s vs expert, N2<br/>+ nap_points.v1 tolerance sweep"]
```

Code: `experiments/rq1_benchmark/evaluate.py` → `src/spindlebench/protocols/benchmark.py`
(`benchmark.v1`). Any change to a rule that can move F1 gets a new protocol version and a line in
`CHANGELOG.md`.

### 3.3 The research questions

```mermaid
flowchart LR
  scores[("per-recording scores<br/>results/benchmark/")]
  scores --> rq1["RQ1 · ranking<br/>experiments/rq1_benchmark"]
  scores --> rq2["RQ2 · validity: does the ranking<br/>survive other references,<br/>tolerances, matchers?<br/>experiments/rq2_validity"]
  scores --> rq3["RQ3 · channels, stage, montage<br/>experiments/rq3_channels"]
  scores --> rq4["RQ4 · cheapest adaptation to NAP<br/>(tuning, fine-tuning SUMO)<br/>experiments/rq4_adaptation"]
  raters["experiments/raters<br/>expert-vs-expert ceiling"] --> rq2
  inj["experiments/injection<br/>SpindleInject: synthetic spindles<br/>with known boundaries"] --> rq2
```

---

## 4. Part B: the label-free student

### 4.1 The idea

Four rule-based detectors ("teachers") disagree with each other (pairwise F1 0.34–0.57). Each makes
systematic errors: YASA misses many spindles, UCSD over-calls. A plain student trained on their vote
learns their shared mistakes. The student here has a second, training-only part that estimates, for
every teacher and every 10 ms, how likely that teacher is to be right. The loss then separates "what
is really there" from "how each teacher errs".

```mermaid
flowchart LR
  eeg["EEG"] --> T1["A7"] & T2["Luna"] & T3["UCSD"] & T4["YASA"]
  T1 & T2 & T3 & T4 --> votes["4 yes/no votes<br/>per 10 ms"]
  eeg --> trunk["student trunk<br/>p(t) = P(true spindle)"]
  eeg --> noise["noise branch<br/>A_r(t): how teacher r errs"]
  trunk & noise --> expl["explain each teacher's vote:<br/>q_r = A_rᵀ · p"]
  votes --> loss["loss = Σ_r CE(vote_r, q_r)<br/>+ λ · mean_r trace(A_r), λ = 0.01"]
  expl --> loss
  trunk -->|"at inference: only this"| out["spindle events"]
```

### 4.2 The whole pipeline, in order

```mermaid
flowchart TB
  subgraph S1["Step 1 · teachers and targets (done)"]
    direction TB
    a["1a · artefact mask<br/>spindlebench artefacts"] --> b["1b · teachers on clean windows<br/>detectors/*/run.py --clean"]
    b --> c["1c · splits by subject<br/>splits.py"]
    c --> d["1d · preprocessing<br/>prepare.py"]
    d --> e["1e · rasterise teacher events<br/>rasterise.py"]
    e --> f["1f · label models<br/>label_models.py fit"]
    f --> g["1g · consensus as a detector<br/>on the gold sets<br/>label_models.py gold"]
  end
  subgraph S2["Step 2 · model and label-free checks (done, pilot running)"]
    direction TB
    h["2a · model, losses, decoders<br/>model.py · losses.py · decode.py"] --> i["2b · inject synthetic spindles<br/>into validation nights<br/>build_val_injection.py"]
    i --> j["2c · teachers on injected nights<br/>→ identifiability check<br/>score_val_injection.py"]
    j --> k["2d · timed runs + pilot<br/>train.py --arm student_soft"]
  end
  subgraph S3["Steps 3–4 · training grid (next)"]
    direction TB
    l["CPU track: trunk S, 42 runs"] --> m2["GPU track: trunk M, 140 runs"]
  end
  subgraph S5["Steps 5–6 · selection, freeze, one gold-set run"]
    direction TB
    n["selection.py → frozen.json<br/>(label-free criteria only)"] --> o["infer.py on DREAMS + NAP<br/>(refuses to run before the freeze)"]
    o --> p["evaluate.py, same as Part A"]
  end
  S1 --> S2 --> S3 --> S5
```

### 4.3 Teacher pass on the unlabelled corpus (step 1)

Artefacts (movement, electrode pops) inflate the whole-night thresholds of relative-power detectors.
Before rejection, Luna and UCSD found almost nothing on 46% and 42% of CinC nights. So every teacher
runs on artefact-free windows only.

```mermaid
flowchart LR
  sig[("cache/signals<br/>HMC 151 + CinC 250 nights<br/>C3-M2")] --> hp["0.3 Hz high-pass"]
  hyp[("public hypnogram<br/>30-s epochs")] --> art
  hp --> art["YASA art_detect<br/>5-s windows, log-SD z-scored<br/>per stage, flag |z| > 3<br/>(artefacts.v1, ~1% flagged)"]
  art --> mask[("cache/labels/&lt;ds&gt;/&lt;rec&gt;__artefact.csv")]
  sig --> join["keep clean windows,<br/>join end to end"]
  mask --> join
  join --> t1["A7"] & t2["Luna"] & t3["Wonambi UCSD"] & t4["YASA"]
  t1 & t2 & t3 & t4 --> back["map events back to<br/>recording time; drop events<br/>within 1 s of a join"]
  back --> store[("results/detections_clean/<br/>&lt;ds&gt;/&lt;teacher&gt;/&lt;rec&gt;.csv")]
```

The same teachers, run the same way, also go on DREAMS and NAP. That gives the "consensus as a
detector" rows in step 1g; no expert label is read to produce them.

### 4.4 From teacher events to training targets (step 1)

```mermaid
flowchart TB
  ev[("teacher events<br/>detections_clean")] --> ras["rasterise at 100 Hz<br/>y_r(t) = 1 inside teacher r's events"]
  hyp[("hypnogram")] --> val["valid v(t) = N2 ∧ clean<br/>∧ > 1 s from a flagged window"]
  art[("artefact mask")] --> val
  ras --> code["one byte per sample<br/>bits 0–3: the 4 teachers<br/>bit 4: valid<br/>work/arrays/&lt;ds&gt;/&lt;rec&gt;__code.npy"]
  val --> code
  code --> pat["count the 16 patterns<br/>(which teachers fired)<br/>over valid training samples<br/>teachers/patterns.csv"]
  pat --> lm["label models, each a 16-entry table<br/>P(spindle | pattern)"]
  lm --> mv["mv: ≥ 2 of 4"]
  lm --> soft["soft: k / 4"]
  lm --> ds["ds: Dawid–Skene (crowd-kit)"]
  lm --> fs["fs: FlyingSquid"]
  ev --> agr["corpus agreement table<br/>teachers/agreement.csv<br/>pairwise F1 0.34–0.57"]
```

Splits are by subject (seed 2026): 10% of subjects per site are **validation nights** (15 HMC + 25 CinC)
and are used for every model-selection decision. The rest are training nights, with nested 10% ⊂ 30%
⊂ 100% subsets for the data-scale arms. The CPU track uses the 30% subset (109 nights, 336 h of valid N2).

### 4.5 Student architecture

```mermaid
flowchart TB
  x["x(t) · 40-s crop · 100 Hz<br/>SUMOv2 preprocessing"]
  subgraph A["Trunk (stream A) · runs at inference"]
    enc["U-Net encoder<br/>S: SUMOv2's network, 70 k params<br/>M: 4 levels, 1.5 M params"] --> bott["bottleneck"]
    bott --> dec["U-Net decoder"]
  end
  x --> enc
  dec --> h1["head 1<br/>p(t) = P(spindle)"]
  dec --> h2["head 2<br/>b(t) = P(boundary)"]
  subgraph B["Stream B · 56 params · training only"]
    g["8 learnable Gabor filters<br/>(centre frequency, width)"] --> e["envelope |·|²<br/>+ Gaussian low-pass"] --> pc["sPCEN:<br/>envelope relative to<br/>its own running level"]
  end
  x --> g
  pc --> cat["concatenate"]
  bott -->|"up-sample to 100 Hz"| cat
  subgraph N["Teacher-noise branch · training only"]
    nb["3 dilated convs<br/>→ 4 teachers × 2 × 2 logits<br/>→ row softmax"]
  end
  cat --> nb
  nb --> A_r["A_r(t): confusion matrix<br/>of teacher r at sample t<br/>init: 0.85 on the diagonal"]
  h1 --> q["q_r(t) = A_r(t)ᵀ π(t)"]
  A_r --> q
  q --> L1["Σ_r cross-entropy(y_r, q_r)<br/>+ λ · mean_r trace A_r"]
  h2 --> L2["boundary BCE<br/>+ segment CDF loss"]
```

Design rule: stream B computes what the teachers threshold (sigma power relative to its recent level).
It feeds **only** the noise branch, so the network can explain a teacher's mistake without the trunk
learning to copy it. Trunk S is byte-for-byte SUMOv2's U-Net, trained from scratch, so "student vs
SUMOv2" compares teacher labels with expert labels on the same network.

### 4.6 One training run (step 2 onwards)

```mermaid
sequenceDiagram
  autonumber
  participant S as CropSampler<br/>(data.py)
  participant Aug as augment
  participant M as Student<br/>(model.py)
  participant L as losses.py
  participant O as Adam
  participant V as ValSet<br/>(validate.py)
  loop every pass (≈ 3,241 batches of 12 on the CPU track)
    S->>Aug: 40-s crops inside valid N2 runs ≥ 40 s<br/>signal + teacher code
    Aug->>M: scale ×[0.5,2], sign flip, 1% noise
    M->>L: logits, A_r(t), b(t)
    L->>L: mask = valid ∧ central 30 s
    L->>O: noisy-annotator loss (+ boundary, segment)
    O->>M: update weights<br/>(noise branch frozen for passes 1–2)
  end
  M->>V: whole-night inference on 40 validation nights
  V->>V: D0 events vs 4 teachers, mv, ds<br/>→ 6 F1s, plausibility (N2 rate, duration)
  V-->>M: Borda rank over passes → keep the best, stop after 5 passes without improvement
```

Arms are switches of this one code path: turning the noise branch off gives `student_soft`, a hard
majority-vote target gives `student_mv`, one teacher's mask gives a per-teacher student (section 14 of
the architecture document).

### 4.7 Choosing a model without expert labels (steps 2–5)

```mermaid
flowchart TB
  vn[("40 unlabelled<br/>validation nights")]
  vn --> agree["agreement criterion<br/>F1 against each teacher, mv, ds<br/>→ Borda count"]
  vn --> inj["inject synthetic spindles<br/>4 copies, −3 to 12 dB SNR<br/>exact boundaries known"]
  inj --> teachers_inj["teachers on injected nights<br/>→ recall by SNR<br/>(identifiability check)"]
  inj --> stud_inj["students on injected nights<br/>→ threshold, decoder,<br/>boundary snapping"]
  vn --> plaus["plausibility filter<br/>N2 rate 1–8 /min,<br/>median duration 0.5–1.2 s<br/>(Wake/REM ratios reported only)"]
  agree & stud_inj & plaus --> sel["selection.py"]
  sel --> frozen[("selection/frozen.json<br/>committed before any gold-set run")]
  frozen --> infer["infer.py: DREAMS + NAP, once"]
  guard["data.guard(): every training,<br/>validation and selection call<br/>refuses DREAMS and NAP"] -.-> sel
```

### 4.8 Decoding probabilities into events

```mermaid
flowchart LR
  p["p(t), 100 Hz"] --> d0["D0 · threshold τ<br/>→ runs → merge short gaps<br/>→ keep 0.3–2.5 s<br/>(SUMOv2's rule, default)"]
  p --> d1["D1 · SEBB change points:<br/>step filter, peaks = onsets,<br/>troughs = offsets, merge, threshold"]
  p --> d2["D2 · semi-Markov Viterbi:<br/>best segmentation with<br/>spindles 0.3–2.5 s, gaps ≥ 0.1 s"]
  b["b(t) boundary head"] --> snap["+ snap onsets / offsets<br/>to b(t) peaks within ±0.2 s"]
  d0 & d1 & d2 --> snap
  snap --> ev[("detections, same CSV<br/>format as Part A")]
```

### 4.9 The student pass on the gold sets (step 6, after the freeze)

```mermaid
flowchart LR
  frozen[("frozen.json")] --> infer["infer.py"]
  gold[("DREAMS 8 excerpts<br/>NAP 36 naps")] --> prep["same preprocessing<br/>(SUMOv2)"] --> infer
  infer --> det[("results/student/detections/<br/>&lt;ds&gt;/student_*/&lt;rec&gt;.csv")]
  det --> eval["experiments/rq1_benchmark/evaluate.py<br/>(Part A, unchanged)"]
  experts[("expert labels")] --> eval
  eval --> table["student vs teachers vs consensus<br/>vs SUMOv2 vs expert-vs-expert"]
```

---

## 5. Where things live

```mermaid
flowchart LR
  root["repository"] --> src["src/spindlebench/<br/>package: paths, cache readers,<br/>matching, protocols, provenance,<br/>SpindleInject generator, artefacts"]
  root --> detd["detectors/<br/>one adapter per detector,<br/>manifest.json, own env"]
  root --> exp["experiments/<br/>one folder per study"]
  exp --> stu["experiments/student/<br/>data, model, losses, decode,<br/>label_models, validate, train,<br/>build/score_val_injection"]
  root --> res["results/<br/>tables, figures, provenance<br/>(large arrays not in git)"]
  res --> rs["results/student/<br/>teachers/, consensus/, injection/,<br/>work/ (splits, targets, runs)"]
  root --> tst["tests/<br/>235 tests: metrics, detectors,<br/>scripts, student"]
  root --> docs["docs/<br/>protocol.md, development.md,<br/>student_architecture.md"]
```

| want to… | look at |
|---|---|
| run or add a detector | `detectors/INTERFACE.md`, `detectors/common.py` |
| change how detections are scored | `src/spindlebench/protocols/benchmark.py` (bump `VERSION`, log in `CHANGELOG.md`) |
| see every study and its commands | `experiments/README.md` |
| understand the student in detail | `docs/student_architecture.md` |
| reproduce the student's steps | `experiments/student/README.md` |
| know what may be published | `DATA_SOURCES.md` |

---

## 6. Status and key numbers

| item | result |
|---|---|
| Part A, DREAMS (N2, onset ±0.5 s vs expert 1) | SUMOv2 0.69 first; lead not resolved |
| Part A, NAP (N2, centre ±0.5 s) | Mölle 0.38 vs DREAMS-tuned YASA 0.37; lead not resolved |
| Teachers disagree on the corpus | pairwise event F1 0.34–0.57 |
| Consensus as a detector (no expert labels) | majority vote DREAMS 0.725, NAP 0.362; best single teacher 0.669 / 0.352 |
| Identifiability (injected spindles) | mean teacher recall > 0.5 only from 9 dB SNR; at 3 dB, 72% of spindles are found by no teacher |
| Trunk S | reproduces SUMOv2's outputs exactly with SUMOv2's weights; trained here from scratch |
| Pilot `student_soft` | running; F1 vs majority vote 0.70 at pass 3 |
| Next | CPU-track training grid (42 runs), then selection and the single gold-set run |

---

## 7. Glossary

| term | meaning |
|---|---|
| N2 | light non-REM sleep, where spindles occur; scored in 30-s epochs |
| gold set | dataset with expert spindle labels (DREAMS, NAP); used only for scoring |
| teacher | one of the four rule-based detectors whose output trains the student |
| consensus / `mv` | majority vote of the teachers (≥ 2 of 4) |
| label model | a way to combine teacher votes into a target (mv, soft, Dawid–Skene, FlyingSquid) |
| valid samples | N2, artefact-free, more than 1 s from an artefact window |
| confusion matrix A_r(t) | P(teacher r says spindle / no spindle ∣ truth) at sample t |
| trace penalty | λ · mean over teachers of (A_r[0,0] + A_r[1,1]), λ = 0.01 (Tanno et al. 2019); pushes teachers to be modelled as noisy, so the trunk takes what they agree on. Too large a λ flips the matrices (amendment 8); a run stops if the teachers are modelled as worse than chance |
| identifiability | the condition under which truth and teacher error can be separated: mean teacher recall > 0.5 |
| injection | adding synthetic spindles with known boundaries to real EEG, to measure recall without experts |
| SNR | 20·log10 of spindle RMS over the background's 11–16 Hz RMS in ±5 s |
| Borda count | ranking checkpoints by the sum of their ranks over several criteria |
| freeze | committing `frozen.json` with the chosen model before anything touches DREAMS or NAP |
| trunk S / M | the small (SUMOv2's) and the larger U-Net; CPU and GPU tracks |
| D0 / D1 / D2 | the three decoders from p(t) to events |
| protocol version | `benchmark.v1`, `nap_points.v1`, `artefacts.v1`: any change that can move F1 gets a new one |
