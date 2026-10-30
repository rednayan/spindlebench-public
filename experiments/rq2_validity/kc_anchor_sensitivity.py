"""
Sensitivity of the K-complex scores to where an event is anchored in time.

    uv run python experiments/rq2_validity/kc_anchor_sensitivity.py

NAP K-complex marks are neither onsets nor centres: the most negative trough of the 0.3-4 Hz signal
lies a median of ~0.4 s before the mark, and that lag varies by nap (0 to -1 s). So besides the onset
and centre rules, every detection and every mark is re-anchored to its trough:
  detection  most negative 0.3-4 Hz sample within [onset, onset + max(duration, 0.5 s)]
  NAP mark   most negative sample within mark ± 1.5 s
  DREAMS     most negative sample within the marked extent ± 0.25 s
and matched one-to-one within ±0.5 and ±1.0 s. Same scoring window, N2 rule (event midpoint in N2)
and included recordings as experiments/rq1_benchmark/evaluate.py. Output: results/sensitivity/kc_anchor.csv (pooled F1).
"""
import json

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

from spindlebench.data.cache import signal_path
from spindlebench import paths
from spindlebench.metrics.events import EPS
from spindlebench import provenance  # noqa: E402

provenance.track(__file__)

DETS = ["spinky_kc", "yasa_sw_kc", "sssm", "sssm_v2", "sssm_strict", "dreams_auto"]
REF = {"nap": "expert", "dreams_kc": "expert1"}
OUT = paths.RESULTS / "sensitivity"


def n_match(a, b, tol):
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if not len(a) or not len(b):
        return 0
    ok = np.abs(a[:, None] - b[None, :]) <= tol + EPS
    return int((maximum_bipartite_matching(csr_matrix(ok.astype(np.int8)), perm_type="column") >= 0).sum())


def trough(x, sf, t0, t1):
    i0, i1 = max(int(round(t0 * sf)), 0), min(int(round(t1 * sf)), len(x))
    return (i0 + np.argmin(x[i0:i1])) / sf if i1 > i0 else np.nan


def main():
    man = pd.read_csv(paths.CACHE / "manifest.csv")
    rows = []
    for ds, refname in REF.items():
        for _, r in man[(man.dataset == ds) & (~man.excluded)].iterrows():
            z = np.load(signal_path(r.path))
            ch = [str(c) for c in z["ch_names"]]
            sf = float(z["sf"])
            xf = sosfiltfilt(butter(2, [0.3, 4], btype="band", fs=sf, output="sos"),
                             z["data"][ch.index(r.default_channel)].astype(float))
            h = pd.read_csv(paths.CACHE / "labels" / f"{ds}/{r.rec}__hypno.csv")
            ep = float(h.epoch_start.iloc[1] - h.epoch_start.iloc[0])
            st = h.stage.to_numpy()
            isn2 = lambda t: st[np.clip((np.asarray(t) // ep).astype(int), 0, len(st) - 1)] == "N2"
            t0, t1 = json.loads(r.windows)[refname]
            ref = pd.read_csv(paths.CACHE / "labels" / f"{ds}/{r.rec}__{refname}__kcomplex.csv")[["onset", "duration"]].to_numpy(float)
            ref = ref[(ref[:, 0] >= t0) & (ref[:, 0] < t1)]
            rmid = ref[:, 0] + np.nan_to_num(ref[:, 1]) / 2
            rtr = np.array([trough(xf, sf, m - 1.5, m + 1.5) for m in ref[:, 0]] if ds == "nap" else
                           [trough(xf, sf, o - 0.25, o + d + 0.25) for o, d in ref]).reshape(-1)
            for det in DETS:
                f = (paths.RESULTS / "detections") / ds / det / f"{r.rec}.csv"
                if not f.exists():
                    continue
                d = pd.read_csv(f)
                d = d[(d.event == "kcomplex") & (d.channel.astype(str) == str(r.default_channel))]
                D = d[["onset", "duration"]].to_numpy(float)
                D = D[(D[:, 0] >= t0) & (D[:, 0] < t1)]
                dmid = D[:, 0] + np.nan_to_num(D[:, 1]) / 2
                dtr = np.array([trough(xf, sf, o, o + max(du, 0.5)) for o, du in D]).reshape(-1)
                for scope in ("all", "N2"):
                    kd = isn2(dmid) if scope == "N2" else np.ones(len(D), bool)
                    kr = isn2(rmid) if scope == "N2" else np.ones(len(ref), bool)
                    for rule, (a, b) in {"onset": (D[kd, 0], ref[kr, 0]), "center": (dmid[kd], rmid[kr]),
                                         "trough": (dtr[kd], rtr[kr])}.items():
                        for tol in (0.5, 1.0):
                            rows.append(dict(dataset=ds, detector=det, rec=r.rec, scope=scope, rule=rule, tol=tol,
                                             tp=n_match(a, b, tol), n_det=len(a), n_ref=len(b)))
        print(ds, "done", flush=True)
    df = pd.DataFrame(rows)
    g = df.groupby(["dataset", "scope", "rule", "tol", "detector"])[["tp", "n_det", "n_ref"]].sum().reset_index()
    g["F1"] = 2 * g.tp / (g.n_det + g.n_ref)
    OUT.mkdir(parents=True, exist_ok=True)
    g.to_csv(OUT / "kc_anchor.csv", index=False)
    pd.set_option("display.width", 200)
    print(g[g.scope == "N2"].pivot_table(index=["dataset", "detector"], columns=["rule", "tol"], values="F1")
          .round(3).to_string())


if __name__ == "__main__":
    main()
