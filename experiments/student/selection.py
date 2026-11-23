"""
Label-free model selection and the freeze (plan section 6; docs/student_architecture.md, section 13).

    uv run python selection.py score [--runs a,b] [--workers 10] [--device auto|cpu|cuda]
                                                                    per finished run: injected + validation nights;
                                                                    the network runs on --device (auto: the GPU
                                                                    when torch sees one), decoding on the CPU
    uv run python selection.py freeze [--track cpu]                 -> results/student/selection/frozen.json
                                                                    (frozen_<track>.json for other tracks)
Runs are read from data.RUNS (SPINDLEBENCH_RUNS, default <repo>/../runs/student). freeze considers the runs of
one track only (default cpu), so GPU-track runs never enter the CPU-track headline choice.

score, per run (its Borda-selected pass, done.json):
  injected validation nights (4 copies of the 40 nights, build_val_injection.py), every decoder setting:
      injected F1 at IoU >= 0.2 per SNR level, with one precision over all considered detections
      (F1_s = 2 P R_s / (P + R_s)); F1_avg = mean over the six SNR levels; fragmentation = mean number
      of detections overlapping an injected event that has one; median signed onset / offset error and
      mean IoU of matched events; induced false alarms per quiet minute
  validation nights (unaltered), every D0 threshold: the six agreement F1s, densities, plausibility
  -> results/student/selection/scores/<run>__injection.csv, <run>__validation.csv
Decoder grid: D0 tau in 0.15..0.70 (step 0.05) for every run; for the headline family (student_rules4*)
also D1 (step 0.32/0.48/0.64 s x merge abs 0.15/0.2/0.3 or rel 1.5/2/3 x confidence 0.3..0.6), D2, and
boundary snapping on the best D0, D1 and D2 setting when head 2 exists.

freeze (amendment 10, 2026-11-09):
  1. every run at one fixed decoder setting, D0 with tau 0.5 and no snapping (FIXED_SETTING); a run whose
     setting fails plausibility on the validation nights (N2 density 1-8 per min, median N2 duration
     0.5-1.2 s; decision 2026-11-04) is excluded. The setting the injection criterion would have chosen
     (highest injected F1_avg, ties -> lower fragmentation; plausibility of D1 / D2 / snapped settings =
     that of the D0 setting with the nearest detection count) is recorded as `injection_choice`, reported only;
  2. headline: student_rules4 fixA, every seed not excluded (HEADLINE). Amendment 9's lambda rule is
     retired; its table (mean injected F1, false alarms, agreement Borda, Wake/N2, REM/N2 per lambda, at
     the fixed setting) is reported under `lambda_ablation`; a family with no surviving run has seeds 0.
     Runs stopped by the label-swap guard have no selected pass and are listed under `collapsed`;
  3. writes frozen.json (CPU track) or frozen_<track>.json: every run with checkpoint (path relative to
     data.RUNS, sha256), pass, setting and criteria values, the headline runs, git commit. infer.py refuses
     to run unless the file is committed.
"""
import argparse
import hashlib
import json
import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import torch

import data
import decode
import rasterise
import score_val_injection as SVI
import train
import validate
from spindlebench import paths, provenance
from spindlebench.metrics.events import match, prf

SEL = data.STUDY / "selection"
SCORES = SEL / "scores"
INJ_ARRAYS = data.WORK / "arrays_inj"
VARIANTS = (0, 1, 2, 3)
TAUS = tuple(round(t, 2) for t in np.arange(0.15, 0.701, 0.05))
D1_GRID = [(st, mode, g) for st in (0.32, 0.48, 0.64) for mode, gs in (("abs", (0.15, 0.2, 0.3)), ("rel", (1.5, 2.0, 3.0)))
           for g in gs]
D1_LAMS = (0.3, 0.4, 0.5, 0.6)
IOU = 0.2


# ------------------------------------------------------------------ model of a run

def resolve_device(device):
    """auto -> cuda when torch sees a GPU, else cpu"""
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda: no CUDA GPU visible to torch")
    return device


def setup_device(device):
    """deterministic inference, as in train.py; one CPU thread per worker for the decoding"""
    if device == "cuda":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)


def load_run(name, device="cpu"):
    run = data.RUNS / name
    cfg = json.loads((run / "config.json").read_text())
    done = json.loads((run / "done.json").read_text())
    a = argparse.Namespace(**cfg["args"])
    a.trunk = cfg["track_settings"]["trunk"]
    net = train.build(cfg["arm"], a, cfg["prior"])
    ck = run / f"pass{done['best_pass']}.pt"
    net.load_state_dict(torch.load(ck, map_location="cpu"))
    net.to(device).eval()
    return net, cfg, done, ck


def sha256(path):
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


# ------------------------------------------------------------------ injected nights

def injected_x(v, ds, rec):
    """SUMOv2-preprocessed injected night (100 Hz float32 .npy, cached)"""
    f = INJ_ARRAYS / f"v{v}" / ds / f"{rec}.npy"
    if not f.exists():
        z = np.load(paths.CACHE.parent / "cache_student_inj" / f"v{v}" / "signals" / ds / f"{rec}.npz")
        f.parent.mkdir(parents=True, exist_ok=True)
        np.save(f, data.preprocess(z["data"][0].astype(float), float(z["sf"])))
    return np.load(f, mmap_mode="r")


def iou_hits(det, tr):
    """one-to-one maximum matching at IoU >= 0.2, solved per cluster of overlapping intervals.
    det, tr: [n, 2] onset/offset s -> (matched truth index -> det index)"""
    out = {}
    if not len(det) or not len(tr):
        return out
    iv = np.r_[det, tr]
    lab = np.r_[np.zeros(len(det), int), np.ones(len(tr), int)]
    idx = np.r_[np.arange(len(det)), np.arange(len(tr))]
    o = np.argsort(iv[:, 0], kind="stable")
    iv, lab, idx = iv[o], lab[o], idx[o]
    reach = np.maximum.accumulate(iv[:, 1])
    cut = np.r_[0, np.flatnonzero(iv[1:, 0] > reach[:-1]) + 1, len(iv)]
    for a, b in zip(cut[:-1], cut[1:]):
        d, t = idx[a:b][lab[a:b] == 0], idx[a:b][lab[a:b] == 1]
        if len(d) and len(t):
            dd, tt = det[d], tr[t]
            _, _, _, pairs = match(np.c_[dd[:, 0], dd[:, 1] - dd[:, 0]], np.c_[tt[:, 0], tt[:, 1] - tt[:, 0]], "iou", IOU)
            for i, j in pairs:
                out[int(t[j])] = int(d[i])
    return out


def score_setting(nights, events_fn):
    """nights: list of (truth DataFrame, region mask, quiet minutes, p, b); events_fn(p, b) -> [n, 3]"""
    hit_snr, fa, n_det, frag, on_err, off_err, ious, quiet = {}, 0, 0, [], [], [], [], 0.0
    for ev, region, qmin, p, b in nights:
        det = events_fn(p, b)
        det = det[:, :2] if len(det) else np.zeros((0, 2))
        c = det.mean(1) if len(det) else np.zeros(0)
        i = np.round(c * data.FS).astype(int)
        keep = (i >= 0) & (i < len(region))
        keep[keep] = region[i[keep]]
        det = det[keep]
        tr = np.c_[ev.onset, ev.onset + ev.duration]
        m = iou_hits(det, tr)
        n_det += len(det)
        fa += len(det) - len(m)
        quiet += qmin
        for j, s in enumerate(ev.snr_db):
            hit_snr.setdefault(s, [0, 0])
            hit_snr[s][1] += 1
            if j in m:
                hit_snr[s][0] += 1
                d = det[m[j]]
                on_err.append(d[0] - tr[j, 0])
                off_err.append(d[1] - tr[j, 1])
                inter = max(0.0, min(d[1], tr[j, 1]) - max(d[0], tr[j, 0]))
                ious.append(inter / (max(d[1], tr[j, 1]) - min(d[0], tr[j, 0])))
        if len(det) and len(tr):
            ov = (det[:, None, 0] < tr[None, :, 1]) & (det[:, None, 1] > tr[None, :, 0])
            k = ov.sum(0)
            frag += list(k[k > 0])
    tp = sum(h for h, _ in hit_snr.values())
    P = tp / n_det if n_det else 0.0
    row = dict(precision=P, false_alarms_per_quiet_min=fa / quiet if quiet else np.nan,
               fragmentation=float(np.mean(frag)) if frag else np.nan,
               onset_err_median=float(np.median(on_err)) if on_err else np.nan,
               offset_err_median=float(np.median(off_err)) if off_err else np.nan,
               iou_mean=float(np.mean(ious)) if ious else np.nan, detections=n_det)
    f1s = []
    for s in sorted(hit_snr):
        R = hit_snr[s][0] / hit_snr[s][1]
        f = 2 * P * R / (P + R) if (P + R) else 0.0
        row[f"recall_{s:g}dB"] = R
        row[f"f1_{s:g}dB"] = f
        f1s.append(f)
    row["f1_avg"] = float(np.mean(f1s))
    return row


def score_run(name, quick=False, device="cpu"):
    """quick: variant 0 and the first 2 validation nights only, written to scores_quick/ (a code check).
    device: where the network runs (cpu or cuda); decoding and matching run on the CPU either way"""
    setup_device(device)
    t0 = time.time()
    net, cfg, done, ck = load_run(name, device)
    headline = cfg["arm"].startswith("student_rules4")
    has_b = net.boundary is not None
    sp = data.splits()
    val = sp[sp.role == "val"]
    if quick:
        val = val.head(2)
    nights = []
    for v in (VARIANTS[:1] if quick else VARIANTS):
        for ds, rec in zip(val.dataset, val.rec):
            ev, q = SVI.load_truth(v, ds, rec)
            x = injected_x(v, ds, rec)
            region = SVI.scored_region(ev, q, len(x))
            qmin = np.unpackbits(q)[:len(x)].sum() / data.FS / 60
            p, b = validate.predict_night(net, x)
            nights.append((ev, region, qmin, p, b if has_b else None))
    rows = []
    for t in TAUS:
        rows.append(dict(decoder="D0", tau=t, snap=False, **score_setting(nights, lambda p, b, t=t: decode.d0(p, t))))
    if headline:
        for st, mode, g in D1_GRID:
            boxes = [decode.sebb_boxes(p, st, g, mode) for _, _, _, p, _ in nights]
            for lam in D1_LAMS:
                it = iter(boxes)

                def d1(p, b, lam=lam, it=it):
                    bx = next(it)
                    if not len(bx):
                        return bx
                    dur = bx[:, 1] - bx[:, 0]
                    return bx[(bx[:, 2] > lam) & (dur >= decode.MIN_DUR) & (dur <= decode.MAX_DUR)]
                rows.append(dict(decoder="D1", step=st, merge=mode, gamma=g, lam=lam, snap=False,
                                 **score_setting(nights, d1)))
        d2_ev = [decode.d2(p) for _, _, _, p, _ in nights]
        it2 = iter(d2_ev)
        rows.append(dict(decoder="D2", snap=False, **score_setting(nights, lambda p, b: next(it2))))
        if has_b:
            df = pd.DataFrame(rows)
            for dec in ("D0", "D1", "D2"):
                best = df[df.decoder == dec].sort_values(["f1_avg", "fragmentation"], ascending=[False, True]).iloc[0]
                if dec == "D0":
                    fn = lambda p, b, t=best.tau: decode.snap(decode.d0(p, t), b)
                elif dec == "D1":
                    fn = lambda p, b, r=best: decode.snap(decode.d1(p, r.step, r.gamma, r.merge, r.lam), b)
                else:
                    it3 = iter(d2_ev)
                    fn = lambda p, b, it3=it3: decode.snap(next(it3), b)
                extra = {k: best[k] for k in ("tau", "step", "merge", "gamma", "lam") if k in best and pd.notna(best[k])}
                rows.append(dict(decoder=dec, snap=True, **extra, **score_setting(nights, fn)))
    inj = pd.DataFrame(rows)
    # validation nights: agreement and plausibility per D0 threshold
    V = validate.ValSet()
    if quick:
        V.nights = V.nights[:2]
    vrows = []
    preds = [validate.predict_night(net, nt["x"])[0] for nt in V.nights]
    for t in TAUS:
        tp = {r: np.zeros(3) for r in validate.REFS}
        cnt = {s: 0 for s in validate.STAGE}
        durs = []
        for nt, p in zip(V.nights, preds):
            ev = decode.d0(p, t)
            c = validate.centres(ev)
            cv = validate.keep_valid(c, nt["valid"])
            for r in validate.REFS:
                k = rasterise.match_tp(cv, nt["ref"][r])
                tp[r] += [k, len(cv) - k, len(nt["ref"][r]) - k]
            i = np.clip(np.round(c * data.FS).astype(int), 0, len(nt["stage"]) - 1)
            ok = nt["clean"][i]
            for s, k in validate.STAGE.items():
                cnt[s] += int((ok & (nt["stage"][i] == k)).sum())
            sel = ok & (nt["stage"][i] == validate.STAGE["N2"])
            durs.append(ev[sel, 1] - ev[sel, 0])
        mins = {s: sum(nt["mins"][s] for nt in V.nights) for s in cnt}
        dens = {s: cnt[s] / mins[s] for s in cnt}
        d = np.concatenate(durs)
        row = dict(tau=t, **{f"f1_{r}": prf(*tp[r])[2] for r in validate.REFS},
                   **{f"per_min_{s}": dens[s] for s in dens}, W_over_N2=dens["W"] / dens["N2"] if dens["N2"] else np.nan,
                   REM_over_N2=dens["REM"] / dens["N2"] if dens["N2"] else np.nan,
                   dur_median=float(np.median(d)) if len(d) else np.nan)
        row["plausible"] = bool(1.0 <= dens["N2"] <= 8.0 and 0.5 <= row["dur_median"] <= 1.2)
        vrows.append(row)
    out = SEL / "scores_quick" if quick else SCORES
    out.mkdir(parents=True, exist_ok=True)
    inj.insert(0, "run", name)
    inj.to_csv(out / f"{name}__injection.csv", index=False)
    pd.DataFrame(vrows).assign(run=name).to_csv(out / f"{name}__validation.csv", index=False)
    (out / f"{name}__meta.json").write_text(json.dumps(dict(
        run=name, device=device, torch=torch.__version__, best_pass=done["best_pass"], sha256=sha256(ck),
        seconds=round(time.time() - t0, 1)), indent=1))
    return name, len(inj)


def run_track(p):
    c = p / "config.json"
    return json.loads(c.read_text()).get("track", "cpu") if c.exists() else None


def finished_runs(track=None):
    """runs whose done.json names the selected pass (not a launcher's claim placeholder); of one track if given"""
    out = []
    for p in sorted(data.RUNS.glob("*")):
        f = p / "done.json"
        if f.exists() and json.loads(f.read_text() or "{}").get("best_pass") is not None:
            if track is None or run_track(p) == track:
                out.append(p.name)
    return out


def cmd_score(runs, workers, device):
    provenance.track(__file__, protocols=["artefacts.v1"])
    runs = runs or [r for r in finished_runs() if not (SCORES / f"{r}__validation.csv").exists()]
    print(f"{time.strftime('%d-%m %H:%M')} scoring {len(runs)} runs on {workers} workers, network on {device}",
          flush=True)
    with ProcessPoolExecutor(workers) as ex:
        for name, n in ex.map(score_run, runs, [False] * len(runs), [device] * len(runs)):
            print(time.strftime("%H:%M"), name, n, "settings", flush=True)


# ------------------------------------------------------------------ freeze

def git_commit():
    r = lambda *a: subprocess.run(["git", "-C", str(paths.ROOT), *a], capture_output=True, text=True).stdout.strip()
    return dict(commit=r("rev-parse", "HEAD"), dirty=bool(r("status", "--porcelain", "--untracked-files=no")))


def choose(name):
    inj = pd.read_csv(SCORES / f"{name}__injection.csv")
    val = pd.read_csv(SCORES / f"{name}__validation.csv").set_index("tau")
    ok_tau = set(val.index[val.plausible])

    def plausible(r):
        if r.decoder == "D0" and not r.snap:
            return r.tau in ok_tau
        # nearest D0 setting by detection count on the injected nights (a proxy for N2 density)
        d0 = inj[(inj.decoder == "D0") & (~inj.snap)]
        near = d0.iloc[(d0.detections - r.detections).abs().argsort().iloc[0]]
        return near.tau in ok_tau
    inj["plausible"] = [plausible(r) for r in inj.itertuples()]
    cand = inj[inj.plausible]
    if cand.empty:
        return None
    best = cand.sort_values(["f1_avg", "fragmentation"], ascending=[False, True]).iloc[0]
    tau_for_val = best.tau if best.decoder == "D0" and not best.snap else \
        inj[(inj.decoder == "D0") & (~inj.snap)].iloc[
            (inj[(inj.decoder == "D0") & (~inj.snap)].detections - best.detections).abs().argsort().iloc[0]].tau
    v = val.loc[tau_for_val]
    setting = {k: (best[k].item() if hasattr(best[k], "item") else best[k])
               for k in ("decoder", "tau", "step", "merge", "gamma", "lam", "snap") if k in best and pd.notna(best[k])}
    return dict(setting=setting, injected={k: float(best[k]) for k in best.index
                                           if k.startswith(("f1_", "recall_")) or k in ("precision", "fragmentation",
                                                                                         "onset_err_median", "offset_err_median",
                                                                                         "iou_mean", "false_alarms_per_quiet_min")},
                validation={k: (float(v[k]) if not isinstance(v[k], (bool, np.bool_)) else bool(v[k]))
                            for k in v.index if k != "run"}, validation_tau=float(tau_for_val))


INJ_KEYS = ("precision", "fragmentation", "onset_err_median", "offset_err_median", "iou_mean",
            "false_alarms_per_quiet_min", "detections")


def at_setting(name, setting=None):
    """amendment 10: every run is frozen at one fixed decoder setting (default FIXED_SETTING), chosen
    before the freeze and the same for all runs; its injected and validation scores are recorded, and the
    run is excluded if that setting fails plausibility on the validation nights. -> (dict, plausible)"""
    s = dict(FIXED_SETTING if setting is None else setting)
    inj = pd.read_csv(SCORES / f"{name}__injection.csv")
    val = pd.read_csv(SCORES / f"{name}__validation.csv")
    i = inj[(inj.decoder == s["decoder"]) & (inj.snap == s["snap"]) & (inj.tau.round(2) == round(s["tau"], 2))]
    v = val[val.tau.round(2) == round(s["tau"], 2)]
    if len(i) != 1 or len(v) != 1:
        raise ValueError(f"{name}: the scores have no single row for {s}")
    i, v = i.iloc[0], v.iloc[0]
    return dict(setting=s, injected={k: float(i[k]) for k in i.index if k.startswith(("f1_", "recall_")) or k in INJ_KEYS},
                validation={k: (float(v[k]) if not isinstance(v[k], (bool, np.bool_)) else bool(v[k]))
                            for k in v.index if k != "run"}, validation_tau=float(s["tau"])), bool(v.plausible)


# amendment 10 (2026-11-09, before any gold-set run): the decoder is no longer chosen by the injection
# criterion, which picked the lowest threshold of the grid (tau 0.15) for every run and ranked the most
# over-detecting runs first; every run is frozen at D0, tau 0.5, no snapping (the decision rule the students
# were trained for). The headline is student_rules4 fixA (noise tables anchored at Dawid-Skene, lambda 0);
# the lambda rule of amendment 9 is retired and its table reported only.
FIXED_SETTING = dict(decoder="D0", tau=0.5, snap=False)
DECODER_RULE = "every run at D0, tau 0.5, no snapping (amendment 10); injection_choice = the old criterion, reported only"
HEADLINE = dict(arm="student_rules4", tag="fixA")
HEADLINE_RULE = ("amendment 10: student_rules4 fixA (noise tables anchored at the Dawid-Skene matrices, "
                 "per-sample correction bounded by 0.5 logit, lambda 0), every seed not excluded")

LAM_FAMILIES = {"0": "lam0", "0.01": "", "0.1": "lam0.1"}   # lambda -> run tag of student_rules4
LAM_RULE = "max mean injected f1_avg over seeds; tie -> min mean false_alarms_per_quiet_min (amendment 9)"


def headline_lambda(out, fam):
    """amendment 9: lambda by the injection criterion; agreement Borda and Wake/REM ratios reported only"""
    rows = []
    for k, rs in fam.items():
        inj = pd.DataFrame([out[r]["injected"] for r in rs]).mean()
        val = pd.DataFrame([out[r]["validation"] for r in rs]).mean()
        rows.append(dict(lam=k, seeds=len(rs), inj_f1_avg=float(inj["f1_avg"]),
                         false_alarms_per_quiet_min=float(inj["false_alarms_per_quiet_min"]),
                         W_over_N2=float(val["W_over_N2"]), REM_over_N2=float(val["REM_over_N2"]),
                         per_min_N2=float(val["per_min_N2"]),
                         **{f"f1_{ref}": float(val[f"f1_{ref}"]) for ref in validate.REFS}))
    df = pd.DataFrame(rows)
    df["agreement_borda"] = sum(df[f"f1_{ref}"].rank() for ref in validate.REFS)
    best = df.sort_values(["inj_f1_avg", "false_alarms_per_quiet_min"], ascending=[False, True]).iloc[0]
    return best.lam, df.to_dict("records")


def cmd_freeze(track="cpu"):
    provenance.track(__file__, protocols=["artefacts.v1"])
    runs = [r for r in finished_runs(track) if (SCORES / f"{r}__validation.csv").exists()]
    out = {}
    for r in runs:
        cfg = json.loads((data.RUNS / r / "config.json").read_text())
        done = json.loads((data.RUNS / r / "done.json").read_text())
        ck = data.RUNS / r / f"pass{done['best_pass']}.pt"
        c, ok = at_setting(r)
        out[r] = dict(arm=cfg["arm"], tag=cfg["args"].get("tag", ""), seed=cfg["seed"], track=cfg["track"],
                      checkpoint=ck.relative_to(data.RUNS).as_posix(), sha256=sha256(ck),
                      best_pass=done["best_pass"], excluded=not ok, **c, injection_choice=choose(r))
    collapsed = {}
    for p in sorted(data.RUNS.glob("*/done.json")):
        if run_track(p.parent) != track:
            continue
        d = json.loads(p.read_text() or "{}")
        if d.get("collapsed"):
            collapsed[p.parent.name] = d["collapsed"]
    head = [r for r, v in out.items() if v["arm"] == HEADLINE["arm"] and v["tag"] == HEADLINE["tag"] and not v["excluded"]]
    # amendment 9's lambda table, at the fixed setting, for the report only; a family without a surviving
    # run (all seeds collapsed or excluded) is listed with seeds 0
    fam = {k: [r for r, v in out.items() if v["arm"] == "student_rules4" and v["tag"] == tag and not v["excluded"]]
           for k, tag in LAM_FAMILIES.items()}
    live = {k: rs for k, rs in fam.items() if rs}
    _, lam_table = headline_lambda(out, live) if live else (None, [])
    lam_table += [dict(lam=k, seeds=0) for k, rs in fam.items() if not rs]
    frozen = dict(runs=out, headline=dict(**HEADLINE, rule=HEADLINE_RULE, runs=head),
                  decoder_rule=DECODER_RULE, lambda_ablation=dict(rule_retired=LAM_RULE, table=lam_table),
                  collapsed=collapsed,
                  criteria="plan 6.1-6.4 with amendment 10; plausibility = N2 density 1-8 /min and median duration "
                           "0.5-1.2 s at the fixed setting",
                  track=track, device=sorted({json.loads((data.RUNS / r / "config.json").read_text())["device"]
                                              for r in runs}),
                  git=git_commit(), protocols=["benchmark.v1", "nap_points.v1", "artefacts.v1"])
    SEL.mkdir(parents=True, exist_ok=True)
    target = SEL / ("frozen.json" if track == "cpu" else f"frozen_{track}.json")
    target.write_text(json.dumps(frozen, indent=1, default=str))
    summary = pd.DataFrame([dict(run=r, decoder=v["setting"]["decoder"], tau=v["setting"]["tau"],
                                 snap=v["setting"]["snap"], inj_f1=v["injected"]["f1_avg"],
                                 precision=v["injected"]["precision"], f1_mv=v["validation"]["f1_mv"],
                                 per_min_N2=v["validation"]["per_min_N2"], excluded=v["excluded"],
                                 headline=r in head) for r, v in out.items()])
    summary.to_csv(SEL / ("summary.csv" if track == "cpu" else f"summary_{track}.csv"), index=False)
    print(summary.round(3).to_string(index=False))
    print(pd.DataFrame(lam_table).round(3).to_string(index=False))
    print(f"headline: {HEADLINE['arm']} {HEADLINE['tag']}, {len(head)} runs; collapsed: {len(collapsed)}")
    print(f"wrote {target}; commit it before infer.py")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["score", "freeze"])
    ap.add_argument("--runs", default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--track", default="cpu", help="freeze: the track whose runs are frozen")
    ap.add_argument("--quick", action="store_true", help="code check: 1 variant, 2 nights, scores_quick/")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                    help="score: where the network runs; auto = cuda when torch sees a GPU")
    a = ap.parse_args()
    device = resolve_device(a.device)
    if a.cmd == "score" and a.quick:
        provenance.track(__file__, protocols=["artefacts.v1"])
        for r in a.runs.split(","):
            print(score_run(r, quick=True, device=device))
    elif a.cmd == "score":
        cmd_score(a.runs.split(",") if a.runs else None, a.workers, device)
    else:
        cmd_freeze(a.track)


if __name__ == "__main__":
    main()
