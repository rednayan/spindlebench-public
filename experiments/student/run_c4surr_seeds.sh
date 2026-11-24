#!/usr/bin/env bash
set -uo pipefail

PY="uv run --no-group cpu --group cuda python"
RUNS="../../../runs/student"

for s in 1 2 3 4; do
  echo "=== Training c4surr seed $s (15 passes) ==="
  if [ ! -f "$RUNS/student_faint_pu_c4surr_gpu_s${s}/done.json" ]; then
    $PY train.py --arm student_faint_pu --mc-witness --surr-n2-frac 0.20 --wake-rem-frac 0.15 --seed "$s" --track gpu --passes 15 --tag c4surr || exit 1
  else
    echo "  (seed $s already done, skipping)"
  fi
done

echo "=== 5-seed 3-axis validation curve (40 val nights, v0..v3) ==="
$PY eval_faint_curve.py --reuse-baselines --device cuda --runs student_faint_pu_c4surr_gpu_s0:15,student_faint_pu_c4surr_gpu_s1:15,student_faint_pu_c4surr_gpu_s2:15,student_faint_pu_c4surr_gpu_s3:15,student_faint_pu_c4surr_gpu_s4:15
cp -f ../../results/student/faint_curve/sweeps.csv ../../results/student/faint_curve/sweeps_c4surr_5seeds.csv 2>/dev/null || true

echo "=== 5-seed DREAMS (E1, E2, IoU) & NAP evaluation ==="
$PY - << 'PYEOF' | tee ../../results/student/faint_curve/c4surr_5seeds_gold.txt
import numpy as np
import pandas as pd
import data
import decode
import eval_faint_curve as efc
import validate
from spindlebench import paths
from spindlebench.data import nap as nap_data
from spindlebench.data.cache import labels_dir, signal_path
from spindlebench.metrics import events as em
from spindlebench.protocols import nap_points as npt

man = pd.read_csv(paths.CACHE / "manifest.csv")
dm = man[man.dataset == "dreams_spindles"].reset_index(drop=True)
d_recs = []
for r in dm.itertuples():
    z = np.load(signal_path(r.path, paths.CACHE / "manifest.csv"))
    ch = [str(c) for c in z["ch_names"]]
    x = data.preprocess(z["data"][ch.index(str(r.default_channel))].astype(float), float(z["sf"]))
    h = pd.read_csv(labels_dir() / "dreams_spindles" / f"{r.rec}__hypno.csv")
    ep = float(h.epoch_start.iloc[1] - h.epoch_start.iloc[0])
    st = h.stage.to_numpy().astype(str)
    refs = {}
    for ex in ("expert1", "expert2"):
        rf = labels_dir() / "dreams_spindles" / f"{r.rec}__{ex}__spindle.csv"
        if rf.exists():
            ref = pd.read_csv(rf)
            ron, rdu = ref.onset.to_numpy(float), np.nan_to_num(ref.duration.to_numpy(float), nan=1.0)
            rk = (ron < 1000.0) & (st[np.clip(((ron + rdu / 2) // ep).astype(int), 0, len(st) - 1)] == "N2")
            refs[ex] = np.c_[ron[rk], rdu[rk]]
    d_recs.append((x, st, ep, refs))

nm = man[man.dataset == "nap"].set_index("rec")
infos = [nap_data.nap_info(r) for r in nap_data.naps().itertuples()]
subjs = sorted({i["subject"] for i in infos})
n_xs = []
for info in infos:
    r = nm.loc[info["rec"]]
    z = np.load(signal_path(r.path, paths.CACHE / "manifest.csv"))
    ch = [str(c) for c in z["ch_names"]]
    n_xs.append(data.preprocess(z["data"][ch.index("C3")].astype(float), float(z["sf"])))

rows = []
for s in range(5):
    run = f"student_faint_pu_c4surr_gpu_s{s}"
    net, *_ = efc._load_run_pass(run, pass_num=15, device="cuda")
    dp = [validate.predict_night(net, x)[0] for x, *_ in d_recs]
    np_ = [validate.predict_night(net, x)[0] for x in n_xs]
    for tau in (0.50, 0.70, 0.75, 0.80):
        tpo = fpo = fno = tpi = fpi = fni = tp2 = fp2 = fn2 = 0
        for p, (_, st, ep, refs) in zip(dp, d_recs):
            ev = decode.d0(p, tau)
            if len(ev):
                on, off = ev[:, 0], ev[:, 1]
                k = (on < 1000.0) & (st[np.clip((((on + off) / 2) // ep).astype(int), 0, len(st) - 1)] == "N2")
                det = np.c_[on[k], (off - on)[k]]
            else:
                det = np.zeros((0, 2))
            a, b, c, _ = em.match(det, refs["expert1"], "onset", 0.5)
            tpo += a
            fpo += b
            fno += c
            a, b, c, _ = em.match(det, refs["expert1"], "iou", 0.2)
            tpi += a
            fpi += b
            fni += c
            if "expert2" in refs:
                a, b, c, _ = em.match(det, refs["expert2"], "onset", 0.5)
                tp2 += a
                fp2 += b
                fn2 += c
        P, R, F1 = em.prf(tpo, fpo, fno)
        _, _, IoU = em.prf(tpi, fpi, fni)
        _, _, F1_E2 = em.prf(tp2, fp2, fn2)
        nres = []
        for p, info in zip(np_, infos):
            ev = decode.d0(p, tau)
            if len(ev):
                on, off = ev[:, 0], ev[:, 1]
                k = nap_data.in_n2(info, (on + off) / 2)
                on, off = on[k], off[k]
            else:
                on, off = np.zeros(0), np.zeros(0)
            nres.append((info["subject"], npt.score_nap(on, off, info["marks"], info["T"])))
        C, aff = npt.tensors(nres, subjs)
        F, integ, *_ = npt.summarise(C, aff)
        rows.append(dict(seed=s, tau=tau, dreams_P=P, dreams_R=R, dreams_F1=F1, dreams_IoU=IoU,
                         dreams_E2=F1_E2, nap_int_F1=float(integ[0]), nap_F1_05=float(F[0, 1])))

df = pd.DataFrame(rows)
df.to_csv("../../results/student/faint_curve/c4surr_5seeds_gold.csv", index=False)
print("=== Per-seed results ===")
print(df.round(3).to_string(index=False))
print("\n=== 5-seed mean [min - max] ===")
for tau, g in df.groupby("tau"):
    print(f"tau={tau:.2f} | DREAMS E1={g.dreams_F1.mean():.3f} [{g.dreams_F1.min():.3f}-{g.dreams_F1.max():.3f}] "
          f"(P={g.dreams_P.mean():.3f}, R={g.dreams_R.mean():.3f}, IoU={g.dreams_IoU.mean():.3f}, E2={g.dreams_E2.mean():.3f}) | "
          f"NAP int={g.nap_int_F1.mean():.3f} [{g.nap_int_F1.min():.3f}-{g.nap_int_F1.max():.3f}]")
PYEOF

echo "=== Copying logs and pushing to origin/pilot-results ==="
for s in 0 1 2 3 4; do
  r="student_faint_pu_c4surr_gpu_s${s}"
  mkdir -p "../../results/student/faint_curve/logs/$r"
  cp -f "$RUNS/$r/config.json" "../../results/student/faint_curve/logs/$r/" 2>/dev/null || true
  cp -f "$RUNS/$r/done.json" "../../results/student/faint_curve/logs/$r/" 2>/dev/null || true
  cp -f "$RUNS/$r/metrics.csv" "../../results/student/faint_curve/logs/$r/" 2>/dev/null || true
done

cd ../..
git add -f results/student/faint_curve/
git commit --allow-empty -m "results: c4surr full 5-seed validation + DREAMS/NAP gold scores"
git push origin HEAD:refs/heads/pilot-results --force
