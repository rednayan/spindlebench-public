"""
Step 1: preprocess every included NAP nap (C3) and the 8 DREAMS excerpts for SUMOv2, with
  - real targets (NAP: +-0.4 s around marks; DREAMS: expert-1 intervals, valid 0-1000 s only),
  - self-training targets (SUMOv2's zero-shot detections from results/detections),
  - N_VARIANTS injected copies (injection/generator.py, raw uV, then the same preprocessing),
    each with its event mask and its valid mask (quiet N2 + injected events),
and write the DREAMS leave-one-excerpt-out folds. NAP reuses the RQ4 folds.

    detectors/sumo/.venv-sumov2/Scripts/python.exe experiments/injection/sumo_inject_ft/prepare.py
"""
import numpy as np
import pandas as pd

import lib as L
from generator import Params, eligible_mask, inject

SEED = 0


def prepare(row, k):
    out = L.prep_path(row.dataset, row.rec)
    if out.exists():
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    z = np.load(L.common.signal_path(row.path, L.ROOT / "cache" / "manifest.csv"))
    ch = [str(c) for c in z["ch_names"]]
    x = z["data"][ch.index(row.default_channel)].astype(float)
    sf = float(z["sf"])
    y = L.zs.preprocess_sumov2(x, sf).astype(np.float32)
    n = len(y)
    params = Params()
    elig, n2_raw, used = eligible_mask(row.dataset, row.rec, row.default_channel, len(x), sf, params)
    n2 = L.resample_mask(n2_raw, sf, n)
    h = pd.read_csv(L.ROOT / "cache" / "labels" / row.dataset / f"{row.rec}__hypno.csv")
    d = dict(y=y, n2=n2.astype(np.int8), epoch=float(h.epoch_start.iloc[1] - h.epoch_start.iloc[0]),
             quiet_detectors="+".join(used))
    if row.dataset == "nap":
        marks = pd.read_csv(L.ROOT / "cache" / "labels" / "nap" / f"{row.rec}__expert__spindle.csv").onset.to_numpy(float)
        d.update(marks=marks, real_mask=L.to_mask(n, marks - L.HALF_WIDTH, np.full(len(marks), 2 * L.HALF_WIDTH)),
                 real_valid=np.ones(n, np.int8))
    else:
        lab = pd.read_csv(L.ROOT / "cache" / "labels" / row.dataset / f"{row.rec}__expert1__spindle.csv")
        d.update(ref_on=lab.onset.to_numpy(float), ref_du=lab.duration.to_numpy(float),
                 real_mask=L.to_mask(n, lab.onset, lab.duration),
                 real_valid=(np.arange(n) < 1000 * L.FS).astype(np.int8))
    zs_ev = pd.read_csv(L.ROOT / "results" / "detections" / row.dataset / "sumov2" / f"{row.rec}.csv")
    zs_ev = zs_ev[zs_ev.channel.astype(str) == row.default_channel]
    d["self_mask"] = L.to_mask(n, zs_ev.onset, zs_ev.duration)
    for v in range(L.N_VARIANTS):
        xi, ev = inject(x, sf, elig, params, seed=100_000 * v + 1000 * k + 7)
        d[f"y{v}"] = L.zs.preprocess_sumov2(xi, sf).astype(np.float32)
        d[f"inj{v}"] = L.to_mask(n, ev.onset, ev.duration)
        d[f"valid{v}"] = (L.resample_mask(elig, sf, n) | d[f"inj{v}"].astype(bool)).astype(np.int8)
        d[f"inj_ev{v}"] = ev[["onset", "duration", "snr_db", "freq"]].to_numpy(float)
    np.savez_compressed(out, **d)


def dreams_folds():
    recs = L.recordings("dreams_spindles").rec.tolist()
    rng = np.random.default_rng(SEED)
    rows = []
    for k, test in enumerate(recs):
        rest = list(rng.permutation([r for r in recs if r != test]))
        rows += [dict(fold=k, rec=test, role="test")]
        rows += [dict(fold=k, rec=r, role="val") for r in rest[:2]]
        rows += [dict(fold=k, rec=r, role="train") for r in rest[2:]]
    f = pd.DataFrame(rows)
    f.to_csv(L.WORK / "folds_dreams.csv", index=False)
    return f


def main():
    rows = pd.concat([L.recordings("nap"), L.recordings("dreams_spindles")])
    for k, row in enumerate(rows.itertuples()):
        prepare(row, k)
        print("prepared", row.dataset, row.rec, flush=True)
    dreams_folds()
    nap = pd.read_csv(L.ROOT / "results" / "rq4" / "work" / "folds.csv")
    nap.to_csv(L.WORK / "folds_nap.csv", index=False)


if __name__ == "__main__":
    main()
