"""
Step 1: Label-free faint-spindle trade-off benchmark (3 dB & 6 dB injected recall vs.
Wake detection rate AND quiet-N2 induced false-alarm rate).

No expert label is read. Evaluates everything on the 40 unlabelled validation nights
and their injected variants (v0..v3) under the exact same IoU >= 0.2 matching rule
used by selection.py.

    python eval_faint_curve.py                           # baselines only (teachers, MV, frozen GPU arms)
    python eval_faint_curve.py --runs <run1,run2>        # overlay full tau-sweep curves for specific runs
    python eval_faint_curve.py --runs <run> --variants 0 # fast 1-variant sweep for quick pilots

Outputs -> results/student/faint_curve/
    baselines.csv        teachers, majority vote, and all GPU arms (tau 0.5 and 0.15 from frozen_gpu.json)
    sweeps.csv           full tau-sweep rows for any --runs requested (incl. phase-randomised N2 surrogate FA)
    faint_tradeoff.png   trade-off plot: (6 dB, 3 dB) recall vs. (Wake/min, Quiet-N2 FA/min, Surrogate-N2 FA/min)
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import data
import decode
import score_val_injection as SVI
import selection
import train
import validate

OUT = data.STUDY / "faint_curve"
SWEEP_TAUS = (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70, 0.80)
SURR_MIN_LEN = 3000      # 30 s minimum contiguous clean N2 segment for phase-randomised surrogate
SURR_EDGE = 500          # 5 s edge margin excluded from scoring (avoids FFT wrap-around & RF edge)
SURR_SEED = 20260101


def _events_in_stages(ev, stage, clean):
    """ev [n, >=2] onset/offset s -> counts per stage dict, N2 durations"""
    counts = {s: 0 for s in validate.STAGE}
    if not len(ev):
        return counts, np.zeros(0)
    c = (ev[:, 0] + ev[:, 1]) / 2
    i = np.clip(np.round(c * data.FS).astype(int), 0, len(stage) - 1)
    ok = clean[i]
    for s, code in validate.STAGE.items():
        counts[s] = int((ok & (stage[i] == code)).sum())
    sel = ok & (stage[i] == validate.STAGE["N2"])
    return counts, ev[sel, 1] - ev[sel, 0]


def _load_teacher_ev_csv(path, channel=None):
    d = pd.read_csv(path, dtype={"channel": str})
    d = d[d.event == "spindle"]
    if channel is not None:
        d = d[d.channel == str(channel)]
    on = d.onset.to_numpy(float)
    du = np.nan_to_num(d.duration.to_numpy(float))
    if not len(on):
        return np.zeros((0, 2), float)
    return np.c_[on, on + du]


def _mv_events_from_teachers(ev_by_teacher, n):
    """Majority vote (>= 2 of 4 teachers) on the 100 Hz grid, decoded with D0."""
    cnt = np.zeros(n, np.uint8)
    for ev in ev_by_teacher:
        if len(ev):
            cnt += data.events_to_mask(ev[:, 0], ev[:, 1] - ev[:, 0], n)
    out = decode.d0((cnt >= 2).astype(np.float32), 0.5)
    return out[:, :2] if len(out) else np.zeros((0, 2), float)


def score_teachers_and_mv(variants=selection.VARIANTS):
    """Score the 4 teachers and their majority vote on both unaltered validation nights
    (Wake/N2/REM densities) and injected validation nights (IoU >= 0.2 recall & quiet-N2 FA)."""
    sp = data.splits()
    val = sp[sp.role == "val"].reset_index(drop=True)
    man = data.manifest().set_index(["dataset", "rec"])
    names = (*data.TEACHERS, "majority_vote")

    # 1. Unaltered validation nights -> per_min_W, per_min_N2, per_min_REM, dur_median
    val_counts = {nm: {s: 0 for s in validate.STAGE} for nm in names}
    val_durs = {nm: [] for nm in names}
    val_mins = {s: 0.0 for s in validate.STAGE}

    for ds, rec in zip(val.dataset, val.rec):
        row = man.loc[(ds, rec)]
        n = data.n_samples(row)
        stage = validate.stage_codes(data.stage_per_sample(ds, rec, n))
        clean = data.clean_per_sample(ds, rec, n)
        for s, code in validate.STAGE.items():
            val_mins[s] += ((stage == code) & clean).sum() / data.FS / 60
        evs = []
        for t in data.TEACHERS:
            ev = _load_teacher_ev_csv(data.TEACHER_STORE / ds / t / f"{rec}.csv", row.default_channel)
            evs.append(ev)
            c, d = _events_in_stages(ev, stage, clean)
            for s in validate.STAGE:
                val_counts[t][s] += c[s]
            val_durs[t].append(d)
        mv_ev = _mv_events_from_teachers(evs, n)
        c, d = _events_in_stages(mv_ev, stage, clean)
        for s in validate.STAGE:
            val_counts["majority_vote"][s] += c[s]
        val_durs["majority_vote"].append(d)

    # 2. Injected validation nights -> IoU >= 0.2 recall per SNR and quiet-N2 FA/min
    hit_snr = {nm: {} for nm in names}
    fa_tot = {nm: 0 for nm in names}
    ndet_tot = {nm: 0 for nm in names}
    quiet_tot = 0.0

    for v in variants:
        store = SVI.OUT / f"v{v}" / "detections"
        for ds, rec in zip(val.dataset, val.rec):
            row = man.loc[(ds, rec)]
            n = data.n_samples(row)
            ev_tr, q = SVI.load_truth(v, ds, rec)
            region = SVI.scored_region(ev_tr, q, n)
            qmin = np.unpackbits(q)[:n].sum() / data.FS / 60
            quiet_tot += qmin
            tr = np.c_[ev_tr.onset, ev_tr.onset + ev_tr.duration]

            evs = []
            for t in data.TEACHERS:
                det = _load_teacher_ev_csv(store / ds / t / f"{rec}.csv")
                evs.append(det)
            all_det = dict(zip(data.TEACHERS, evs))
            all_det["majority_vote"] = _mv_events_from_teachers(evs, n)

            for nm, det in all_det.items():
                if len(det):
                    c = det.mean(1)
                    idx = np.round(c * data.FS).astype(int)
                    keep = (idx >= 0) & (idx < len(region))
                    keep[keep] = region[idx[keep]]
                    det = det[keep]
                m = selection.iou_hits(det, tr)
                ndet_tot[nm] += len(det)
                fa_tot[nm] += len(det) - len(m)
                for j, s in enumerate(ev_tr.snr_db):
                    hit_snr[nm].setdefault(float(s), [0, 0])
                    hit_snr[nm][float(s)][1] += 1
                    if j in m:
                        hit_snr[nm][float(s)][0] += 1

    rows = []
    for nm in names:
        tp = sum(h for h, _ in hit_snr[nm].values())
        P = tp / ndet_tot[nm] if ndet_tot[nm] else 0.0
        dens = {s: val_counts[nm][s] / val_mins[s] if val_mins[s] else np.nan for s in validate.STAGE}
        d = np.concatenate(val_durs[nm]) if val_durs[nm] else np.zeros(0)
        r = dict(
            group="consensus" if nm == "majority_vote" else "teacher",
            name=nm,
            tau=np.nan,
            precision=P,
            false_alarms_per_quiet_min=fa_tot[nm] / quiet_tot if quiet_tot else np.nan,
            per_min_W=dens["W"],
            per_min_N2=dens["N2"],
            per_min_REM=dens["REM"],
            W_over_N2=dens["W"] / dens["N2"] if dens["N2"] else np.nan,
            dur_median=float(np.median(d)) if len(d) else np.nan,
        )
        for s in sorted(hit_snr[nm]):
            r[f"recall_{s:g}dB"] = hit_snr[nm][s][0] / hit_snr[nm][s][1]
        rows.append(r)
    return pd.DataFrame(rows)


def load_frozen_gpu_baselines():
    """Extract mean across seeds for all 13 GPU arms at both tau=0.5 and injection_choice (tau=0.15)."""
    f = selection.SEL / "frozen_gpu.json"
    d = json.loads(f.read_text())["runs"]
    rows = []
    for k, v in d.items():
        arm = "_".join(k.split("_")[:-1])
        for label, inj, val, tau in [
            ("frozen", v["injected"], v["validation"], v["setting"]["tau"]),
            ("low_tau", v["injection_choice"]["injected"], v["injection_choice"]["validation"],
             v["injection_choice"]["setting"].get("tau", v["injection_choice"].get("validation_tau", np.nan))),
        ]:
            if v["injection_choice"]["setting"]["decoder"] != "D0" and label == "low_tau":
                continue
            r = dict(
                group=f"gpu_{label}",
                name=arm,
                tau=float(tau),
                precision=inj["precision"],
                false_alarms_per_quiet_min=inj["false_alarms_per_quiet_min"],
                per_min_W=val["per_min_W"],
                per_min_N2=val["per_min_N2"],
                per_min_REM=val["per_min_REM"],
                W_over_N2=val["W_over_N2"],
                dur_median=val["dur_median"],
            )
            for s in (-3, 0, 3, 6, 9, 12):
                r[f"recall_{s}dB"] = inj[f"recall_{s}dB"]
            rows.append(r)
    df = pd.DataFrame(rows)
    return df.groupby(["group", "name"], as_index=False).mean(numeric_only=True)


def phase_randomize(seg, rng):
    """Phase-randomised surrogate of a 1-D signal: preserves exact power spectrum, mean, and variance,
    destroying transient coherent bursts (spindles)."""
    n = len(seg)
    X = np.fft.rfft(np.asarray(seg, dtype=np.float64))
    hi = len(X) - 1 if n % 2 == 0 else len(X)
    if hi > 1:
        phi = rng.uniform(0.0, 2.0 * np.pi, hi - 1)
        X[1:hi] *= np.exp(1j * phi)
    return np.fft.irfft(X, n=n).astype(np.float32)


def surrogate_n2_fa(net, V, taus):
    """Detections per minute on phase-randomised clean N2 segments of the validation nights.
    Any detection here is a genuine false alarm on an exact N2 power spectrum (cannot be cheated by
    a Wake-vs-N2 stage classifier)."""
    counts = {float(t): 0 for t in taus}
    total_min = 0.0
    for k, nt in enumerate(V.nights):
        x = nt["x"]
        v = nt["valid"]
        d = np.diff(np.r_[0, v.astype(np.int8), 0])
        starts, stops = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
        rng = np.random.default_rng(SURR_SEED + k * 997)
        for a, b in zip(starts, stops):
            L = int(b - a)
            if L < SURR_MIN_LEN:
                continue
            surr = phase_randomize(x[a:b], rng)
            p, _ = validate.predict_night(net, surr)
            core_lo, core_hi = SURR_EDGE / data.FS, (L - SURR_EDGE) / data.FS
            total_min += (core_hi - core_lo) / 60.0
            for t in taus:
                ev = decode.d0(p, t)
                if len(ev):
                    c = (ev[:, 0] + ev[:, 1]) / 2.0
                    counts[float(t)] += int(((c >= core_lo) & (c <= core_hi)).sum())
    return {t: (counts[t] / total_min if total_min > 0 else np.nan) for t in counts}


def _load_run_pass(run_name, pass_num=None, device="cpu"):
    """Load either the Borda-selected pass (pass_num=None) or a specific pass<k>.pt."""
    if pass_num is None:
        return selection.load_run(run_name, device)
    run = data.RUNS / run_name
    cfg = json.loads((run / "config.json").read_text())
    a = argparse.Namespace(**cfg["args"])
    a.trunk = cfg["track_settings"]["trunk"]
    net = train.build(cfg["arm"], a, cfg["prior"])
    ck = run / f"pass{pass_num}.pt"
    net.load_state_dict(torch.load(ck, map_location="cpu"))
    net.to(device).eval()
    return net, cfg, {"best_pass": pass_num}, ck


def sweep_run(run_name, variants=selection.VARIANTS, taus=SWEEP_TAUS, device="auto",
              pass_num=None, threads=None, max_nights=None):
    """Run full tau-sweep on validation + injected + phase-randomised N2 surrogate nights."""
    dev = selection.resolve_device(device)
    selection.setup_device(dev)
    if threads is not None and dev == "cpu":
        torch.set_num_threads(threads)
    net, cfg, done, ck = _load_run_pass(run_name, pass_num=pass_num, device=dev)
    label = f"{run_name}@p{done['best_pass']}" if pass_num is not None else run_name
    sp = data.splits()
    val = sp[sp.role == "val"]
    if max_nights is not None:
        val = val.head(max_nights)

    # Injected predictions
    inj_nights = []
    for v in variants:
        for ds, rec in zip(val.dataset, val.rec):
            ev, q = SVI.load_truth(v, ds, rec)
            x = selection.injected_x(v, ds, rec)
            region = SVI.scored_region(ev, q, len(x))
            qmin = np.unpackbits(q)[:len(x)].sum() / data.FS / 60
            p, _ = validate.predict_night(net, x)
            inj_nights.append((ev, region, qmin, p, None))

    # Unaltered validation predictions + phase-randomised N2 surrogates
    V = validate.ValSet()
    if max_nights is not None:
        V.nights = V.nights[:max_nights]
    val_preds = [validate.predict_night(net, nt["x"])[0] for nt in V.nights]
    surr_fa = surrogate_n2_fa(net, V, taus)
    mins = {s: sum(nt["mins"][s] for nt in V.nights) for s in validate.STAGE}

    rows = []
    for t in taus:
        inj_r = selection.score_setting(inj_nights, lambda p, b, t=t: decode.d0(p, t))
        cnt = {s: 0 for s in validate.STAGE}
        durs = []
        for nt, p in zip(V.nights, val_preds):
            ev = decode.d0(p, t)
            c, d = _events_in_stages(ev, nt["stage"], nt["clean"])
            for s in validate.STAGE:
                cnt[s] += c[s]
            durs.append(d)
        dens = {s: cnt[s] / mins[s] if mins[s] else np.nan for s in validate.STAGE}
        d_all = np.concatenate(durs) if durs else np.zeros(0)
        r = dict(
            group="sweep",
            name=label,
            tau=float(t),
            precision=inj_r["precision"],
            false_alarms_per_quiet_min=inj_r["false_alarms_per_quiet_min"],
            surr_n2_per_min=surr_fa[float(t)],
            per_min_W=dens["W"],
            per_min_N2=dens["N2"],
            per_min_REM=dens["REM"],
            W_over_N2=dens["W"] / dens["N2"] if dens["N2"] else np.nan,
            dur_median=float(np.median(d_all)) if len(d_all) else np.nan,
        )
        for s in (-3, 0, 3, 6, 9, 12):
            r[f"recall_{s}dB"] = inj_r[f"recall_{s}dB"]
        rows.append(r)
    return pd.DataFrame(rows)


def plot_tradeoff(base_df, sweep_df, out_path):
    has_surr = sweep_df is not None and len(sweep_df) and "surr_n2_per_min" in sweep_df.columns
    xcols = [
        ("per_min_W", "Wake detections / min (clean Wake)"),
        ("false_alarms_per_quiet_min", "Quiet-N2 false alarms / min (injected nights)"),
    ]
    if has_surr:
        xcols.append(("surr_n2_per_min", "Phase-randomised N2 surrogate FA / min"))
    ncols = len(xcols)
    fig, axes = plt.subplots(2, ncols, figsize=(6.0 * ncols, 9.2), dpi=200)
    snrs = [("recall_6dB", "6 dB injected recall"), ("recall_3dB", "3 dB injected recall")]

    key_arms = {
        "student_rules4_fixA_gpu": ("Headline (rules4_fixA)", "#d62728", "o"),
        "student_ds_gpu": ("Plain student (DS)", "#1f77b4", "s"),
        "student_mv_gpu": ("Plain student (MV)", "#2ca02c", "^"),
        "student_rules4_lam0_gpu": ("rules4 (lam=0)", "#9467bd", "D"),
    }

    for r_i, (ycol, ylabel) in enumerate(snrs):
        for c_i, (xcol, xlabel) in enumerate(xcols):
            ax = axes[r_i, c_i]

            if xcol in base_df.columns:
                # Teachers
                tdf = base_df[base_df.group == "teacher"]
                ax.scatter(tdf[xcol], tdf[ycol], s=65, marker="x", color="#555555", zorder=4, label="Single teachers")
                for row in tdf.itertuples():
                    short = row.name.replace("wonambi_", "").replace("_spindles", "").replace("_default", "")
                    ax.annotate(short, (getattr(row, xcol), getattr(row, ycol)),
                                textcoords="offset points", xytext=(5, 3), fontsize=8, color="#444444")

                # Majority vote (star)
                mv = base_df[base_df.name == "majority_vote"].iloc[0]
                ax.scatter([mv[xcol]], [mv[ycol]], s=180, marker="*", color="black", zorder=6,
                           label="Teachers' majority vote")
                ax.axvline(mv[xcol], color="black", ls=":", lw=1.0, alpha=0.5)
                ax.axhline(mv[ycol], color="black", ls=":", lw=1.0, alpha=0.5)

                # Key GPU arms: connect tau=0.5 to tau=0.15
                for arm, (lab, col, mk) in key_arms.items():
                    sub = base_df[base_df.name == arm].sort_values("tau", ascending=False)
                    if len(sub):
                        ax.plot(sub[xcol], sub[ycol], color=col, ls="--", lw=1.2, alpha=0.65)
                        f0 = sub[sub.group == "gpu_frozen"]
                        f1 = sub[sub.group == "gpu_low_tau"]
                        if len(f0):
                            ax.scatter(f0[xcol], f0[ycol], s=70, marker=mk, color=col, zorder=5, label=f"{lab} (τ=0.5)")
                        if len(f1):
                            ax.scatter(f1[xcol], f1[ycol], s=45, marker=mk, facecolors="none", edgecolors=col,
                                       lw=1.4, zorder=5)
            else:
                mv = base_df[base_df.name == "majority_vote"].iloc[0]
                ax.axhline(mv[ycol], color="black", ls=":", lw=1.0, alpha=0.5, label="MV recall level")

            # Sweep curves for any requested runs
            if sweep_df is not None and len(sweep_df):
                for nm, g in sweep_df.groupby("name"):
                    g = g.sort_values("tau", ascending=False)
                    ax.plot(g[xcol], g[ycol], "-o", lw=2.0, ms=4, zorder=7, label=f"Sweep: {nm}")

            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ymax = max(0.65, base_df[ycol].max() * 1.15)
            if sweep_df is not None and len(sweep_df):
                ymax = max(ymax, sweep_df[ycol].max() * 1.12)
            ax.set_ylim(0, min(1.0, ymax))
            ax.grid(True, alpha=0.25)
            if r_i == 0 and c_i == ncols - 1:
                ax.legend(fontsize=8, loc="lower right", framealpha=0.9)

    fig.suptitle("Step 1 Label-Free Bar: Faint-Spindle Recall vs. Wake, Quiet-N2 & Surrogate-N2 False Alarms\n"
                 "(Target zone: above & left; dotted lines = teachers' majority vote)",
                 fontsize=11, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="", help="comma-separated run names to sweep over tau (optional :pass suffix)")
    ap.add_argument("--variants", default="0,1,2,3", help="injected variants to include")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--threads", type=int, default=4, help="CPU threads for inference")
    ap.add_argument("--max-nights", type=int, default=None, help="cap validation nights (quick check)")
    ap.add_argument("--reuse-baselines", action="store_true", help="load existing baselines.csv instead of recomputing")
    a = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    variants = tuple(int(v) for v in a.variants.split(",") if v.strip())

    if a.reuse_baselines and (OUT / "baselines.csv").exists():
        base_df = pd.read_csv(OUT / "baselines.csv")
    else:
        print("Scoring teachers and majority vote on validation + injected nights...", flush=True)
        t_df = score_teachers_and_mv(variants)
        g_df = load_frozen_gpu_baselines()
        base_df = pd.concat([t_df, g_df], ignore_index=True)
        base_df.to_csv(OUT / "baselines.csv", index=False)

    sweep_df = None
    if a.runs.strip():
        parts = []
        for spec in [x.strip() for x in a.runs.split(",") if x.strip()]:
            if ":" in spec:
                r, pstr = spec.split(":", 1)
                pnum = int(pstr)
            else:
                r, pnum = spec, None
            print(f"Sweeping run {r} (pass={pnum or 'best'}, variants={variants})...", flush=True)
            parts.append(sweep_run(r, variants=variants, device=a.device, pass_num=pnum,
                                   threads=a.threads, max_nights=a.max_nights))
        sweep_df = pd.concat(parts, ignore_index=True)
        sweep_df.to_csv(OUT / "sweeps.csv", index=False)
        cols = ["name", "tau", "recall_3dB", "recall_6dB", "per_min_W",
                "false_alarms_per_quiet_min", "surr_n2_per_min", "per_min_N2"]
        print("\n=== Sweeps (incl. phase-randomised N2 surrogate FA/min) ===")
        print(sweep_df[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    plot_tradeoff(base_df, sweep_df, OUT / "faint_tradeoff.png")
    print(f"\nSaved: {OUT / 'baselines.csv'}"
          + (f", {OUT / 'sweeps.csv'}" if sweep_df is not None else "")
          + f", {OUT / 'faint_tradeoff.png'}")


if __name__ == "__main__":
    main()
