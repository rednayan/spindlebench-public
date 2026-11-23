"""
Zero-retraining decoder comparison (#4) on saved probability traces (results/student/work/gold_p/).

Diagnostic comparison of D0 (tau=0.50 frozen), D0 (tau=0.40, 0.35), Hysteresis (separates peak
confidence tau_hi from boundary extent tau_lo), D1 (SEBBs change-point), and D2 (Viterbi) across
all 5 seeds of student_rules4_fixA_gpu and student_ds_gpu on DREAMS (E1 onset 0.5 & IoU 0.2) and NAP
(integrated F1).

    python eval_decoders_gold.py

Outputs -> results/student/decoder_study/summary.csv
"""
import numpy as np
import pandas as pd

import data
import decode
from spindlebench import paths
from spindlebench.data import nap as nap_data
from spindlebench.data.cache import labels_dir
from spindlebench.metrics import events as em
from spindlebench.protocols import nap_points as npt

OUT = data.STUDY / "decoder_study"
GOLD_P = data.WORK / "gold_p"


def hysteresis(p, tau_hi=0.50, tau_lo=0.25):
    """Dual-threshold hysteresis decoding: an event's extent is the contiguous run where p > tau_lo,
    kept only if its peak reaches >= tau_hi; then SUMOv2 merge/filter."""
    p = np.asarray(p, float)
    s, e = decode._runs(p > tau_lo)
    if not len(s):
        return np.zeros((0, 3))
    keep = [k for k, (a, b) in enumerate(zip(s, e)) if p[a:b].max() >= tau_hi]
    if not keep:
        return np.zeros((0, 3))
    ev = np.c_[s[keep], e[keep]].astype(float) / decode.FS
    ev = decode._sumo().merge_and_filter_spindles(ev)
    return decode._score(p, ev)


def load_dreams_meta():
    man = pd.read_csv(paths.CACHE / "manifest.csv")
    man = man[man.dataset == "dreams_spindles"].reset_index(drop=True)
    meta = []
    for r in man.itertuples():
        h = pd.read_csv(labels_dir() / "dreams_spindles" / f"{r.rec}__hypno.csv")
        ep = float(h.epoch_start.iloc[1] - h.epoch_start.iloc[0])
        ref = pd.read_csv(labels_dir() / "dreams_spindles" / f"{r.rec}__expert1__spindle.csv")
        ron = ref.onset.to_numpy(float)
        rdu = np.nan_to_num(ref.duration.to_numpy(float), nan=1.0)
        rc = ron + rdu / 2.0
        st_r = h.stage.to_numpy().astype(str)[np.clip((rc // ep).astype(int), 0, len(h) - 1)]
        rkeep = (ron < 1000.0) & (st_r == "N2")
        meta.append(dict(rec=r.rec, stages=h.stage.to_numpy().astype(str), ep=ep,
                         ref=np.c_[ron[rkeep], rdu[rkeep]]))
    return meta


def score_dreams(run, dec_fn, meta):
    tp_on = fp_on = fn_on = 0
    tp_io = fp_io = fn_io = 0
    for m in meta:
        z = np.load(GOLD_P / run / "dreams_spindles" / f"{m['rec']}.npz")
        p = z["p"].astype(np.float64)
        ev = dec_fn(p)
        if len(ev):
            on, off = ev[:, 0], ev[:, 1]
            c = (on + off) / 2.0
            st = m["stages"][np.clip((c // m["ep"]).astype(int), 0, len(m["stages"]) - 1)]
            keep = (on < 1000.0) & (st == "N2")
            det = np.c_[on[keep], (off - on)[keep]]
        else:
            det = np.zeros((0, 2))
        a, b, c_, _ = em.match(det, m["ref"], "onset", 0.5)
        tp_on += a; fp_on += b; fn_on += c_
        a, b, c_, _ = em.match(det, m["ref"], "iou", 0.2)
        tp_io += a; fp_io += b; fn_io += c_
    p_on, r_on, f1_on = em.prf(tp_on, fp_on, fn_on)
    _, _, f1_io = em.prf(tp_io, fp_io, fn_io)
    return dict(dreams_P=p_on, dreams_R=r_on, dreams_F1=f1_on, dreams_IoU02=f1_io)


def score_nap(run, dec_fn, infos, subjects):
    res = []
    for info in infos:
        z = np.load(GOLD_P / run / "nap" / f"{info['rec']}.npz")
        p = z["p"].astype(np.float64)
        ev = dec_fn(p)
        if len(ev):
            on, off = ev[:, 0], ev[:, 1]
            keep = nap_data.in_n2(info, (on + off) / 2.0)
            on, off = on[keep], off[keep]
        else:
            on, off = np.zeros(0), np.zeros(0)
        res.append((info["subject"], npt.score_nap(on, off, info["marks"], info["T"])))
    C, aff = npt.tensors(res, subjects)
    F, integ, _, _, _ = npt.summarise(C, aff)
    return dict(nap_int_F1=float(integ[0]), nap_F1_05=float(F[0, 1]))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dmeta = load_dreams_meta()
    infos = [nap_data.nap_info(r) for r in nap_data.naps().itertuples()]
    subjects = sorted({i["subject"] for i in infos})

    decoders = [
        ("D0_tau0.50 (frozen)", lambda p: decode.d0(p, 0.50)),
        ("D0_tau0.40", lambda p: decode.d0(p, 0.40)),
        ("D0_tau0.35", lambda p: decode.d0(p, 0.35)),
        ("Hyst_0.50_0.25", lambda p: hysteresis(p, 0.50, 0.25)),
        ("Hyst_0.45_0.20", lambda p: hysteresis(p, 0.45, 0.20)),
        ("D1_SEBB_lam0.50", lambda p: decode.d1(p, 0.48, 2.0, "rel", 0.50)),
        ("D1_SEBB_lam0.40", lambda p: decode.d1(p, 0.48, 2.0, "rel", 0.40)),
        ("D2_Viterbi", lambda p: decode.d2(p)),
    ]

    arms = ["student_rules4_fixA_gpu", "student_ds_gpu"]
    rows = []
    for arm in arms:
        for dec_name, fn in decoders:
            seed_rows = []
            for s in range(5):
                run = f"{arm}_s{s}"
                r = score_dreams(run, fn, dmeta)
                r.update(score_nap(run, fn, infos, subjects))
                seed_rows.append(r)
            mean_r = pd.DataFrame(seed_rows).mean().to_dict()
            rows.append(dict(arm=arm, decoder=dec_name, **mean_r))
            print(f"{arm:24s} | {dec_name:20s} | DREAMS F1={mean_r['dreams_F1']:.3f} "
                  f"(P={mean_r['dreams_P']:.3f}, R={mean_r['dreams_R']:.3f}, IoU={mean_r['dreams_IoU02']:.3f}) | "
                  f"NAP int={mean_r['nap_int_F1']:.3f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "summary.csv", index=False)
    print(f"\nSaved -> {OUT / 'summary.csv'}")


if __name__ == "__main__":
    main()
