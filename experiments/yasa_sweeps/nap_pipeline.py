"""
NAP EEG (Mei, Grossberg, Ng, Navarro & Ellmore, OSF chav7): dataset analysis + YASA spindle benchmark.

    uv run python experiments/yasa_sweeps/nap_pipeline.py [NapEEG] [results/nap] [--describe-only] [--no-sweep]

Data
  64-ch actiCHamp, 1 kHz, ~30 min naps, recorded against TP9 (left mastoid).
  One expert per recording (subjects 11+): sleep stages in 30-s epochs
  (Markon/Markoff), spindle *onsets* only (criterion: visible in >= 3 of
  F3 F4 C3 C4 O1 O2), K-complex onsets. Labels are hand-typed and messy
  ("Marker:Markon:2", "Makon: 2", "spind;e", "K-Complex") -> parse_annotations.

Protocol (kept as close to DREAMS as the annotations allow)
  signal      band-passed 0.3-40 Hz (DC-coupled amplifier), re-referenced to
              linked mastoids (x - TP10/2), resampled to 200 Hz
  detectors   YASA defaults, YASA loosened, YASA with the thresholds that won the
              DREAMS sweep (transfer), YASA tuned on NAP by leave-one-subject-out
              (C3 only); all on single channels C3 and Cz; plus a 6-channel
              consensus (spindle onsets within 1 s on >= k of the 6 scored channels)
  matching    one-to-one, expert mark vs detection centre within 0.5 s (main);
              also vs detection onset, and tolerances 1.0 / 1.5 s. No IoU: the
              reference has no durations. The marks sit at the spindle peak
              (sigma envelope peaks ~0.1 s after them), not at the onset.
  scope       whole nap and N2 only (event onset inside an expert N2 epoch)
  exclusions  as in Mei et al. 2017: subjects 26, 27; subject 12 day 1;
              subject 20 day 2 (reported separately, not dropped from the description)
"""
import itertools
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from scipy.signal import butter, hilbert, sosfiltfilt

from spindlebench import paths, plotting, yasa_utils
from spindlebench.data.nap_annotations import EXCLUDED, hypnogram, parse_annotations
from spindlebench.data.nap_annotations import recordings as _recordings
from spindlebench.metrics import events as em
from spindlebench import provenance  # noqa: E402

provenance.track(__file__)

mne.set_log_level("ERROR")
args = [a for a in sys.argv[1:] if not a.startswith("--")]
ROOT = Path(args[0]) if args else paths.raw("nap")
OUT = Path(args[1]) if len(args) > 1 else (paths.RESULTS / "nap")
DESCRIBE_ONLY = "--describe-only" in sys.argv
SWEEP = "--no-sweep" not in sys.argv
OUT.mkdir(parents=True, exist_ok=True)

EPOCH = 30.0
SF = 200.0
SCORED = ["F3", "F4", "C3", "C4", "O1", "O2"]
SINGLE = ["C3", "Cz"]
TOLS = [0.5, 1.0, 1.5]
# expert marks sit near the spindle peak, not its onset (see annotation_alignment.png),
# so the main protocol compares the mark with the detection centre
ANCHORS = ["center", "onset"]


def n2_mask(hyp):
    def f(t):
        idx = np.clip((t // EPOCH).astype(int), 0, len(hyp) - 1)
        return hyp[idx] == "N2"
    return f


def recordings():
    return _recordings(ROOT)


def raw_ready(vhdr):
    return vhdr is not None and vhdr.with_suffix(".eeg").exists()


# ---- 1. describe ------------------------------------------------------------
recs = recordings()
desc = []
for r in recs:
    row = dict(subject=r["subject"], day=r["day"], wm_load=r["load"],
               has_annotations=r["ann"] is not None, has_raw=r["vhdr"] is not None,
               excluded_mei2017=(r["subject"], r["day"]) in EXCLUDED)
    if r["ann"] is not None:
        a = parse_annotations(r["ann"])
        r["a"] = a
        dur = max([t for t, _ in a["stages"]] + list(a["spindles"]) + [0]) + EPOCH
        hyp = hypnogram(a["stages"], dur, a["stage_offs"])
        r["hyp"] = hyp
        mins = {s: (hyp == s).sum() * EPOCH / 60 for s in ["W", "N1", "N2", "N3", "REM", "?"]}
        sp_n2 = len(a["spindles"][n2_mask(hyp)(a["spindles"])]) if len(a["spindles"]) else 0
        row.update(scored_min=sum(v for k, v in mins.items() if k != "?"),
                   **{f"min_{k}": v for k, v in mins.items()},
                   n_spindles=len(a["spindles"]), n_spindles_N2=sp_n2,
                   spindle_density_N2_per_min=sp_n2 / mins["N2"] if mins["N2"] else np.nan,
                   n_kcomplexes=len(a["kcomplexes"]), n_unparsed=len(a["unparsed"]))
    desc.append(row)
desc = pd.DataFrame(desc)
desc.to_csv(OUT / "dataset_description.csv", index=False)
pd.set_option("display.width", 220, "display.precision", 2, "display.max_rows", 100)
print(desc.to_string(index=False))
paired = desc[desc.has_annotations & desc.has_raw]
print(f"\nannotation files {desc.has_annotations.sum()}, raw recordings {desc.has_raw.sum()}, "
      f"paired {len(paired)}, paired & not excluded {(~paired.excluded_mei2017).sum()}")
inc = paired[~paired.excluded_mei2017]
print(f"included: {inc.n_spindles.sum()} spindles ({inc.n_spindles_N2.sum()} in N2), "
      f"{inc.n_kcomplexes.sum()} K-complexes, N2 {inc.min_N2.sum():.0f} min of {inc.scored_min.sum():.0f} scored min, "
      f"N2 density {inc.n_spindles_N2.sum() / inc.min_N2.sum():.2f}/min")
unparsed = [u for r in recs if "a" in r for u in r["a"]["unparsed"]]
if unparsed:
    print("unparsed labels:", pd.Series(unparsed).value_counts().to_dict())

# stage composition figure
stage_cols = {"W": "#c3c2b7", "N1": "#9cc3ee", "N2": plotting.C_YASA, "N3": "#173f75", "REM": plotting.C_E1, "?": "#ffffff"}
d = desc[desc.has_annotations].sort_values(["subject", "day"])
fig, ax = plt.subplots(figsize=(12, max(4, 0.2 * len(d))))
left = np.zeros(len(d))
for s in ["W", "N1", "N2", "N3", "REM"]:
    ax.barh(np.arange(len(d)), d[f"min_{s}"], left=left, color=stage_cols[s], label=s, edgecolor="white", lw=0.6)
    left += d[f"min_{s}"].values
ax.set_yticks(np.arange(len(d)), [f"{s}/d{dd}{' ×' if e else ''}" for s, dd, e in zip(d.subject, d.day, d.excluded_mei2017)], fontsize=8)
ax.invert_yaxis()
ax.set_xlabel("minutes (expert-scored 30-s epochs)  ·  × = excluded in Mei et al. 2017")
ax.legend(frameon=False, ncol=5, loc="lower right", fontsize=8)
plotting.style(ax)
fig.tight_layout()
fig.savefig(OUT / "stage_composition.png", dpi=170)
plt.close(fig)

if DESCRIBE_ONLY:
    sys.exit()


# ---- 2. YASA benchmark ------------------------------------------------------
def linked_mastoid(vhdr):
    """channels in µV at SF Hz, referenced to (TP9 + TP10) / 2 = x - TP10/2 (TP9 is the recording ref)"""
    raw = mne.io.read_raw_brainvision(vhdr, preload=True)
    raw.pick(SCORED + ["Cz", "TP10"])
    raw.filter(0.3, 40.0)  # actiCHamp is DC-coupled: remove drift (std was 100s-1000s of µV)
    raw.resample(SF)
    x = raw.get_data() * 1e6
    names = raw.ch_names
    tp10 = x[names.index("TP10")]
    return {c: x[names.index(c)] - tp10 / 2 for c in SCORED + ["Cz"]}


def consensus(per_ch, k, tol=1.0):
    """onsets detected on >= k channels within tol s (greedy clustering of all onsets)"""
    ev = sorted((o, d, c) for c, arr in per_ch.items() for o, d in arr)
    out, i = [], 0
    while i < len(ev):
        j = i
        while j + 1 < len(ev) and ev[j + 1][0] - ev[i][0] <= tol:
            j += 1
        cluster = ev[i:j + 1]
        if len({c for _, _, c in cluster}) >= k:
            out.append((np.median([o for o, _, _ in cluster]), np.median([d for _, d, _ in cluster])))
        i = j + 1
    return np.array(out) if out else em.empty()


def centred(ev):
    """events re-anchored at their midpoint"""
    out = ev.copy()
    if len(out):
        out[:, 0] += out[:, 1] / 2
    return out


def dreams_best():
    p = paths.RESULTS / "dreams/sweep_expert1_onset.csv"
    if not p.exists():
        return None
    s = pd.read_csv(p).groupby(["rel_pow", "corr", "rms"])[["tp", "fp", "fn"]].sum()
    best = (2 * s.tp / (2 * s.tp + s.fp + s.fn)).idxmax()
    return dict(zip(["rel_pow", "corr", "rms"], best))


DREAMS_BEST = dreams_best()
print("\nDREAMS-tuned thresholds (transfer):", DREAMS_BEST)
configs = {"yasa_default": yasa_utils.YASA_DEFAULT, "yasa_loose": yasa_utils.YASA_LOOSE}
if DREAMS_BEST:
    configs["yasa_dreams_tuned"] = DREAMS_BEST
GRID = yasa_utils.THRESH_GRID
combos = list(itertools.product(*GRID.values()))

todo = [r for r in recs if r["ann"] is not None and raw_ready(r["vhdr"])]
print(f"running YASA on {len(todo)} paired recordings with raw data on disk")
rows, sweep, lags, erp = [], [], {a: [] for a in ANCHORS}, []
sos = butter(4, [11, 16], btype="band", fs=SF, output="sos")
t0 = time.time()
for r in todo:
    key = dict(subject=r["subject"], day=r["day"], excluded=(r["subject"], r["day"]) in EXCLUDED)
    sig = linked_mastoid(r["vhdr"])
    dur = len(sig["C3"]) / SF
    hyp = r["hyp"]
    ref = np.c_[r["a"]["spindles"], np.zeros(len(r["a"]["spindles"]))]  # onsets only
    ref = em.clip(ref, 0, dur)

    dets = {}
    for name, th in configs.items():
        per_ch = {c: yasa_utils.run_yasa(sig[c], SF, th) for c in SCORED + ["Cz"]}
        for c in SINGLE:
            dets[(name, c)] = per_ch[c]
        for k in (1, 3):
            dets[(name, f"6ch>={k}")] = consensus({c: per_ch[c] for c in SCORED}, k)

    for (name, chan), det in dets.items():
        for scope in ("all", "N2"):
            dd, rr = (em.in_mask(det, n2_mask(hyp)), em.in_mask(ref, n2_mask(hyp))) if scope == "N2" else (det, ref)
            for anchor in ANCHORS:
                da = centred(dd) if anchor == "center" else dd
                for tol in TOLS:
                    tp, fp, fn, pairs = em.match(da, rr, "onset", tol)
                    p, rc, f = em.prf(tp, fp, fn)
                    rows.append({**key, "detector": name, "channel": chan, "scope": scope, "anchor": anchor,
                                 "tol": tol, "tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": rc, "f1": f})
                    if name == "yasa_loose" and chan == "C3" and scope == "all" and tol == 1.0:
                        lags[anchor] += [da[i, 0] - rr[j, 0] for i, j in pairs]

    # annotation alignment: sigma envelope (C3) around expert onsets vs random N2 times
    env = np.abs(hilbert(sosfiltfilt(sos, sig["C3"])))
    win = np.arange(int(-2 * SF), int(3 * SF))
    on = (ref[:, 0] * SF).astype(int)
    on = on[(on + win[0] >= 0) & (on + win[-1] < len(env))]
    if len(on):
        erp.append(env[on[:, None] + win[None, :]].mean(0) / np.median(env))

    if SWEEP:  # C3 only, main protocol: detection centre vs expert mark, tol 0.5
        for rp, co, rm in combos:
            det = yasa_utils.run_yasa(sig["C3"], SF, {"rel_pow": rp, "corr": co, "rms": rm})
            tp, fp, fn, _ = em.match(centred(det), ref, "onset", 0.5)
            dn, rn = em.in_mask(det, n2_mask(hyp)), em.in_mask(ref, n2_mask(hyp))
            tp2, fp2, fn2, _ = em.match(centred(dn), rn, "onset", 0.5)
            sweep.append({**key, "rel_pow": rp, "corr": co, "rms": rm, "tp": tp, "fp": fp, "fn": fn,
                          "tp_N2": tp2, "fp_N2": fp2, "fn_N2": fn2})
    print(f"  suj{r['subject']} d{r['day']}: {len(ref)} ref spindles  ({time.time() - t0:.0f} s)", flush=True)

per = pd.DataFrame(rows)

# leave-one-subject-out tuning on NAP itself (C3), evaluated like the fixed configs
if SWEEP and len(sweep):
    sweep = pd.DataFrame(sweep)
    sweep.to_csv(OUT / "sweep_C3_center05.csv", index=False)
    keys = ["rel_pow", "corr", "rms"]
    inc_sw = sweep[~sweep.excluded]
    chosen = []
    for subj in sorted(sweep.subject.unique()):
        tr = inc_sw[inc_sw.subject != subj].groupby(keys)[["tp", "fp", "fn"]].sum()
        best = (2 * tr.tp / (2 * tr.tp + tr.fp + tr.fn)).idxmax()
        chosen.append(dict(held_out_subject=subj, **dict(zip(keys, best))))
        for _, s in sweep[(sweep.subject == subj) & (sweep[keys] == pd.Series(best, index=keys)).all(axis=1)].iterrows():
            for scope, sfx in (("all", ""), ("N2", "_N2")):
                tp, fp, fn = s["tp" + sfx], s["fp" + sfx], s["fn" + sfx]
                p, rc, f = em.prf(tp, fp, fn)
                rows.append(dict(subject=s.subject, day=s.day, excluded=s.excluded, detector="yasa_nap_tuned_loso",
                                 channel="C3", scope=scope, anchor="center", tol=0.5, tp=tp, fp=fp, fn=fn,
                                 precision=p, recall=rc, f1=f))
    pd.DataFrame(chosen).to_csv(OUT / "loso_chosen_thresholds.csv", index=False)
    pooled = inc_sw.groupby(keys)[["tp", "fp", "fn"]].sum()
    pooled["f1"] = 2 * pooled.tp / (2 * pooled.tp + pooled.fp + pooled.fn)
    print("best NAP thresholds (in-sample):", dict(zip(keys, pooled.f1.idxmax())), f"F1 {pooled.f1.max():.3f}")
    per = pd.DataFrame(rows)

per.to_csv(OUT / "per_recording.csv", index=False)
summ = em.summarise(per[~per.excluded], ["detector", "channel", "scope", "anchor", "tol"])
summ.to_csv(OUT / "summary_included.csv", index=False)
em.summarise(per, ["detector", "channel", "scope", "anchor", "tol"]).to_csv(OUT / "summary_all_recordings.csv", index=False)
print("\n=== included recordings, main protocol: detection centre vs expert mark, tol 0.5 s ===")
print(summ.query("anchor == 'center' and tol == 0.5").to_string(index=False))
print("\n=== literal onset reading vs centre, C3, N2 ===")
print(summ.query("channel == 'C3' and scope == 'N2'").to_string(index=False))

# ---- figures ----------------------------------------------------------------
# F1 distribution per detector (C3, N2, tol 0.5) - one dot per recording
sub = per[(~per.excluded) & (per.channel == "C3") & (per.scope == "N2") & (per.anchor == "center") & (per.tol == 0.5)]
order = [d for d in ["yasa_default", "yasa_loose", "yasa_dreams_tuned", "yasa_nap_tuned_loso"] if d in sub.detector.unique()]
labels = {"yasa_default": "defaults", "yasa_loose": "loosened", "yasa_dreams_tuned": "DREAMS-tuned\n(transfer)",
          "yasa_nap_tuned_loso": "NAP-tuned\n(LOSO)"}
fig, ax = plt.subplots(figsize=(7, 4))
rng = np.random.default_rng(0)
for k, d in enumerate(order):
    v = sub[sub.detector == d].f1.dropna().values
    ax.scatter(k + rng.uniform(-0.15, 0.15, len(v)), v, s=22, color=plotting.C_YASA, alpha=0.55, edgecolor="white", lw=0.5)
    ax.hlines(np.median(v), k - 0.28, k + 0.28, color=plotting.INK, lw=2)
    ax.text(k + 0.32, np.median(v), f"{np.median(v):.2f}", va="center", fontsize=9, color=plotting.INK)
ax.set_xticks(range(len(order)), [labels[d] for d in order])
ax.set_ylabel("F1 per recording (C3, N2, centre ±0.5 s)\nbar = median")
ax.set_ylim(0, 1)
ax.yaxis.grid(True, color=plotting.GRID, lw=0.8)
ax.set_axisbelow(True)
plotting.style(ax)
fig.tight_layout()
fig.savefig(OUT / "f1_by_detector.png", dpi=170)
plt.close(fig)

# alignment: average sigma envelope around expert onsets, and YASA-expert onset lag
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.6))
if erp:
    tt = np.arange(int(-2 * SF), int(3 * SF)) / SF
    m = np.mean(erp, 0)
    a1.plot(tt, m, color=plotting.C_YASA, lw=2)
    a1.axvline(0, color=plotting.INK2, ls="--", lw=0.8)
    a1.set_xlabel("time from expert mark (s)")
    a1.set_ylabel("11-16 Hz envelope (× recording median)")
    a1.set_title("Sigma envelope around expert spindle marks (C3)", fontsize=10)
    plotting.style(a1)
if lags["onset"]:
    bins = np.arange(-1, 1.01, 0.1)
    for anchor, c in (("onset", "#9cc3ee"), ("center", plotting.C_YASA)):
        a2.hist(lags[anchor], bins=bins, color=c, histtype="step", lw=2,
                label=f"YASA {anchor} − mark, median {np.median(lags[anchor]):+.2f} s")
    a2.set_xlabel("time difference (s)")
    a2.set_ylabel("matched pairs")
    a2.set_title("YASA loosened on C3 vs expert marks (matched within 1 s)", fontsize=10)
    a2.legend(frameon=False, fontsize=8)
    plotting.style(a2)
fig.tight_layout()
fig.savefig(OUT / "annotation_alignment.png", dpi=170)
plt.close(fig)
print(f"\nwrote {OUT}/")
