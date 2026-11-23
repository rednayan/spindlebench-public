"""
Label-free checkpoint evaluation on the unlabelled validation nights (plan 6.1, 6.4).

    ValSet(track)            the validation nights (both sites), their reference events, staging, masks
    ValSet.evaluate(model)   whole-night inference -> six agreement F1s + plausibility statistics

Agreement (6.1): event F1, centres within +-0.5 s, one-to-one maximum matching, pooled over the
validation nights, on events whose centre is valid (N2 of the public hypnogram, clean), of the
student's D0 events (tau = 0.5) against each of the six references: the four teachers, the majority
vote `mv` and the Dawid-Skene `ds` consensus (16-pattern tables of label_models.py, decoded with D0).
Plausibility (6.4): student events per minute in clean W, N2, N3 and REM, and their median duration.
`plausible` = N2 density 1-8 per min and median N2 duration 0.5-1.2 s; W/N2 and REM/N2 are reported only.
No expert label is read.
"""
import numpy as np
import pandas as pd
import torch

import data
import decode
import label_models
import rasterise
from spindlebench.metrics.events import prf

REFS = (*data.TEACHERS, "mv", "ds")
CHUNK, CONTEXT = 60_000, 3_000     # 10-min chunks with 30 s of context on each side
STAGE = {"W": 0, "N2": 2, "N3": 3, "REM": 4}   # int8 codes (N1 = 1, unscored = -1)


def stage_codes(st):
    out = np.full(len(st), -1, np.int8)
    for name, k in {"W": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4}.items():
        out[st == name] = k
    return out


def predict_night(model, x):
    """P(spindle) and boundary b over a whole night, in chunks (eval mode, no grad)"""
    model.eval()
    device = next(model.parameters()).device      # the model's device (cpu or cuda)
    n = len(x)
    p, b = np.zeros(n, np.float32), np.zeros(n, np.float32)
    for a in range(0, n, CHUNK):
        lo, hi = max(a - CONTEXT, 0), min(a + CHUNK + CONTEXT, n)
        xt = torch.from_numpy(np.ascontiguousarray(x[lo:hi], dtype=np.float32))[None].to(device)
        pc, bc = model.predict(xt)
        e = min(a + CHUNK, n)
        p[a:e] = pc[0, a - lo:e - lo].cpu().numpy()
        if bc is not None:
            b[a:e] = bc[0, a - lo:e - lo].cpu().numpy()
    return p, b


def centres(ev):
    return (ev[:, 0] + ev[:, 1]) / 2 if len(ev) else np.zeros(0)


def keep_valid(c, valid):
    i = np.round(c * data.FS).astype(int)
    ok = (i >= 0) & (i < len(valid))
    return np.sort(c[ok][valid[i[ok]]])


class ValSet:
    def __init__(self, sites=data.SITES):
        sp = data.splits()
        val = sp[(sp.role == "val") & sp.dataset.isin(sites)]
        man = data.manifest().set_index(["dataset", "rec"])
        tables = label_models.load_targets()
        self.nights = []
        for ds, rec in zip(val.dataset, val.rec):
            x, code = data.open_night(ds, rec)
            code = np.asarray(code)
            valid = (code & data.VALID_BIT) > 0
            pat = code & (data.VALID_BIT - 1)
            stage = stage_codes(data.stage_per_sample(ds, rec, len(code)))
            clean = data.clean_per_sample(ds, rec, len(code))
            ref = {}
            for t in data.TEACHERS:
                on, du = data.load_teacher_events(ds, rec, t, man.loc[(ds, rec)].default_channel)
                ref[t] = keep_valid(on + np.nan_to_num(du) / 2, valid)
            for k in ("mv", "ds"):
                ref[k] = keep_valid(centres(decode.d0(tables[k][pat], 0.5)), valid)
            mins = {s: ((stage == STAGE[s]) & clean).sum() / data.FS / 60 for s in STAGE}
            self.nights.append(dict(dataset=ds, rec=rec, x=x, valid=valid, stage=stage, clean=clean, ref=ref,
                                    mins=mins))

    def evaluate(self, model, tau=0.5, keep_p=False):
        tp = {r: np.zeros(3) for r in REFS}
        counts = {s: 0 for s in STAGE}
        durs, ps = [], {}
        for nt in self.nights:
            p, b = predict_night(model, nt["x"])
            if keep_p:
                ps[(nt["dataset"], nt["rec"])] = (p, b)
            ev = decode.d0(p, tau)
            c = centres(ev)
            cv = keep_valid(c, nt["valid"])
            for r in REFS:
                k = rasterise.match_tp(cv, nt["ref"][r])
                tp[r] += [k, len(cv) - k, len(nt["ref"][r]) - k]
            i = np.clip(np.round(c * data.FS).astype(int), 0, len(nt["stage"]) - 1)
            ok = nt["clean"][i]
            for s in counts:
                counts[s] += int((ok & (nt["stage"][i] == STAGE[s])).sum())
            sel = ok & (nt["stage"][i] == STAGE["N2"])
            durs.append(ev[sel, 1] - ev[sel, 0])
        mins = {s: sum(nt["mins"][s] for nt in self.nights) for s in counts}
        out = {f"f1_{r}": prf(*tp[r])[2] for r in REFS}
        out["min_teacher_f1"] = float(min(out[f"f1_{t}"] for t in data.TEACHERS))
        dens = {s: counts[s] / mins[s] if mins[s] else np.nan for s in counts}
        d = np.concatenate(durs) if durs else np.zeros(0)
        out.update({f"per_min_{s}": dens[s] for s in dens},
                   W_over_N2=dens["W"] / dens["N2"] if dens["N2"] else np.nan,
                   REM_over_N2=dens["REM"] / dens["N2"] if dens["N2"] else np.nan,
                   dur_median=float(np.median(d)) if len(d) else np.nan)
        # filter (decision 2026-11-04): N2 density and duration only; W/N2 and REM/N2 are reported, not
        # filtered, because the teachers' own majority vote has W/N2 0.38 and REM/N2 0.11 on these nights
        out["plausible"] = bool(1.0 <= dens["N2"] <= 8.0 and 0.5 <= out["dur_median"] <= 1.2)
        return (out, ps) if keep_p else out


def borda(metrics, refs=REFS, min_pass=3):
    """Borda count over passes: each of the six F1s ranks the passes (best = highest rank); the score is
    the sum of the ranks. Passes before min_pass are not eligible. -> (best pass, scores Series)"""
    df = pd.DataFrame(metrics)
    df = df[df["pass"] >= min_pass]
    if df.empty:
        return None, pd.Series(dtype=float)
    score = sum(df[f"f1_{r}"].rank(method="average") for r in refs)
    score.index = df["pass"]
    return int(score.idxmax()), score
