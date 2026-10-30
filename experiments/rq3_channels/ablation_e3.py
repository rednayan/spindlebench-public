"""
E3 (RQ3) on the benchmark data: how does spindle detection change with the number of
channels the detector sees, and with the sleep stages it is evaluated on?

    uv run python experiments/rq3_channels/ablation_e3.py detect      # YASA on every EEG channel of every NAP recording (~20 min)
    uv run python experiments/rq3_channels/ablation_e3.py analyse     # fuse channel sets, score, plot

Channel sets (NAP, linked-mastoid reference):
  C3, Cz                       single channel (the benchmark condition)
  C3+C4                        central pair
  clinical6  F3 F4 C3 C4 O1 O2 the montage the NAP expert scored on
  central9   C1 C2 C3 C4 Cz FC1 FC2 CP1 CP2
  all        every EEG channel (EOG and the TP10 reference excluded)
Detectors: YASA (defaults, DREAMS-tuned) on every EEG channel; plus Mölle 2011 and wonambi's
A7 and SUMOv2 from the shared store, which were run on the 7 scored channels only (sets
outside those channels are skipped; a run channel without detections is an empty set).
Fusion: the detector runs per channel; onsets from all channels in the set are clustered
(within 1 s); a cluster is a spindle when at least k channels fired. k is a fraction q
of the set size (q in 0 [any channel], 0.1, 0.25, 0.5, 0.75), chosen per held-out
subject on the other subjects (leave-one-subject-out). Scored centre ±0.5 s vs the
expert inside the scored window, whole nap and N2 only, recordings excluded in Mei et
al. 2017 left out. Gains over C3 get a paired subject-bootstrap 95% CI.
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from spindlebench.data.cache import signal_path
from spindlebench import paths, plotting, yasa_utils
from spindlebench.metrics import events as em

sys.path.insert(0, str(paths.ROOT / "detectors"))
from common import NAP_CHANNELS  # noqa: E402  channels the store detectors were run on
from spindlebench import provenance  # noqa: E402

provenance.track(__file__)

OUT = paths.RESULTS / "e3"
DET = OUT / "detections"
CONFIGS = {"yasa_default": yasa_utils.YASA_DEFAULT}
SETS = {
    "C3": ["C3"], "Cz": ["Cz"], "C3+C4": ["C3", "C4"],
    "clinical6": ["F3", "F4", "C3", "C4", "O1", "O2"],
    "central9": ["C1", "C2", "C3", "C4", "Cz", "FC1", "FC2", "CP1", "CP2"],
}
QS = [0.0, 0.1, 0.25, 0.5, 0.75]
NOT_EEG = {"LOc", "ROc", "TP10"}


def dreams_best():
    s = pd.read_csv(paths.RESULTS / "dreams/sweep_expert1_onset.csv").groupby(["rel_pow", "corr", "rms"])[["tp", "fp", "fn"]].sum()
    return dict(zip(["rel_pow", "corr", "rms"], map(float, (2 * s.tp / (2 * s.tp + s.fp + s.fn)).idxmax())))


CONFIGS["yasa_dreams_tuned"] = dreams_best()
# detectors from the shared store (results/detections/nap), fused only over the channels they were run on
STORE = ["wonambi_moelle2011", "wonambi_lacourse2018", "sumov2"]


def detect_one(path_rec):
    path, rec = path_rec
    z = np.load(signal_path(path))
    names = [str(c) for c in z["ch_names"]]
    sf = float(z["sf"])
    for cfg, th in CONFIGS.items():
        f = DET / cfg / f"{rec}.csv"
        if f.exists():
            continue
        rows = []
        for i, ch in enumerate(names):
            if ch in NOT_EEG:
                continue
            for o, d in yasa_utils.run_yasa(z["data"][i].astype(float), sf, th):
                rows.append((o, d, ch))
        f.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows, columns=["onset", "duration", "channel"]).to_csv(f, index=False)
    return rec


def fuse(det, channels, k, tol=1.0):
    """cluster onsets over the channel set; keep clusters seen on >= k channels"""
    d = det[det.channel.isin(channels)].sort_values("onset").to_numpy()
    out, i = [], 0
    while i < len(d):
        j = i
        while j + 1 < len(d) and d[j + 1][0] - d[i][0] <= tol:
            j += 1
        c = d[i:j + 1]
        if len({x[2] for x in c}) >= k:
            out.append((np.median(c[:, 0].astype(float)), np.median(c[:, 1].astype(float))))
        i = j + 1
    return np.array(out, float) if out else em.empty()


def analyse():
    m = pd.read_csv(paths.CACHE / "manifest.csv").query("dataset == 'nap'")
    rows = []
    for _, r in m.iterrows():
        t0, t1 = json.loads(r.windows)["expert"]
        ref = em.clip(pd.read_csv(paths.CACHE / "labels" / f"nap/{r.rec}__expert__spindle.csv")[["onset", "duration"]].to_numpy(float), t0, t1)
        h = pd.read_csv(paths.CACHE / "labels" / f"nap/{r.rec}__hypno.csv").stage.to_numpy()
        is_n2 = lambda t: h[np.clip((t // 30).astype(int), 0, len(h) - 1)] == "N2"
        names = [str(c) for c in np.load(signal_path(r.path))["ch_names"]]
        sets = dict(SETS, all=[c for c in names if c not in NOT_EEG])
        sources = {cfg: DET / cfg for cfg in CONFIGS}
        sources.update({d: (paths.RESULTS / "detections/nap") / d for d in STORE
                        if ((paths.RESULTS / "detections/nap") / d / f"{r.rec}.csv").exists()})
        for cfg, folder in sources.items():
            det = pd.read_csv(folder / f"{r.rec}.csv")
            det = det[det.event == "spindle"] if "event" in det.columns else det
            for sname, chans in sets.items():
                if cfg in STORE and not set(chans) <= set(NAP_CHANNELS):
                    continue  # channel set not covered by this detector's run
                for q in QS:
                    k = max(1, int(np.ceil(q * len(chans))))
                    ev = em.clip(fuse(det, chans, k), t0, t1)
                    for scope in ("all", "N2"):
                        d, rr = (em.in_mask(ev, is_n2), em.in_mask(ref, is_n2)) if scope == "N2" else (ev, ref)
                        dc = d.copy()
                        if len(dc):
                            dc[:, 0] += dc[:, 1] / 2
                        tp, fp, fn, _ = em.match(dc, rr, "onset", 0.5)
                        rows.append(dict(rec=r.rec, subject=r.subject, excluded=r.excluded, config=cfg, set=sname,
                                         n_channels=len(chans), q=q, k=k, scope=scope, tp=tp, fp=fp, fn=fn))
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "channel_ablation_per_recording.csv", index=False)
    inc = res[~res.excluded]

    def f1(g):
        return 2 * g.tp.sum() / (2 * g.tp.sum() + g.fp.sum() + g.fn.sum())

    # fixed-q table and LOSO-chosen q
    fixed = inc.groupby(["config", "scope", "set", "n_channels", "q"]).apply(f1, include_groups=False).rename("F1").reset_index()
    fixed.to_csv(OUT / "channel_ablation_fixed_q.csv", index=False)
    loso, held = [], []
    for (cfg, scope, sname), g in inc.groupby(["config", "scope", "set"]):
        chosen = []
        for subj in g.subject.unique():
            tr = g[g.subject != subj].groupby("q").apply(f1, include_groups=False)
            q = tr.idxmax()
            chosen.append(q)
            te = g[(g.subject == subj) & (g.q == q)]
            held.append(dict(config=cfg, scope=scope, set=sname, subject=subj, tp=te.tp.sum(), fp=te.fp.sum(), fn=te.fn.sum()))
        h = pd.DataFrame(held[-len(chosen):])
        loso.append(dict(config=cfg, scope=scope, set=sname, n_channels=g.n_channels.iloc[0],
                         n_rec=len(g) // len(QS), F1_loso=f1(h), q_chosen=pd.Series(chosen).mode().iloc[0]))
    loso = pd.DataFrame(loso).sort_values(["config", "scope", "n_channels"])
    # gain over C3 per config and scope: paired bootstrap over subjects (both naps together)
    held = pd.DataFrame(held).set_index(["config", "scope", "set", "subject"])
    rng = np.random.default_rng(0)
    lo, hi = [], []
    for _, r in loso.iterrows():
        a, b = held.loc[(r.config, r.scope, r.set)], held.loc[(r.config, r.scope, "C3")]
        subj = a.index.intersection(b.index).to_numpy()
        a, b = a.loc[subj].to_numpy(float), b.loc[subj].to_numpy(float)
        idx = rng.integers(0, len(subj), (2000, len(subj)))
        sa, sb = a[idx].sum(1), b[idx].sum(1)  # (boot, [tp, fp, fn])
        f = lambda x: 2 * x[:, 0] / (2 * x[:, 0] + x[:, 1] + x[:, 2])
        d = f(sa) - f(sb)
        lo.append(np.percentile(d, 2.5))
        hi.append(np.percentile(d, 97.5))
    loso["gain_vs_C3"] = loso.F1_loso - loso.set_index(["config", "scope"]).index.map(
        loso[loso.set == "C3"].set_index(["config", "scope"]).F1_loso)
    loso["gain_lo"], loso["gain_hi"] = lo, hi
    loso.to_csv(OUT / "channel_ablation_loso.csv", index=False)
    print(loso.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    order = ["C3", "Cz", "C3+C4", "clinical6", "central9", "all"]
    nch = loso.groupby("set").n_channels.first()
    # Okabe-Ito colours plus a distinct marker per config, so series stay apart without colour
    style = {"yasa_default": ("#56B4E9", "o"), "yasa_dreams_tuned": ("#0072B2", "s"),
             "wonambi_moelle2011": ("#D55E00", "^"), "wonambi_lacourse2018": ("#009E73", "D"),
             "sumov2": ("#CC79A7", "v")}
    for ax, scope in zip(axes, ["N2", "all"]):
        for cfg in loso.config.unique():
            c, mk = style.get(cfg, ("#000000", "x"))
            s = loso[(loso.config == cfg) & (loso.scope == scope)].set_index("set").reindex(order)
            ax.plot(range(len(order)), s.F1_loso, "-", marker=mk, color=c, lw=2, ms=7, label=cfg)
        ax.set_xticks(range(len(order)), [f"{o}\n({int(nch[o])} ch)" for o in order], fontsize=8)
        ax.set_title(f"scope: {scope}", fontsize=10)
        ax.yaxis.grid(True, color=plotting.GRID, lw=0.8)
        plotting.style(ax)
    axes[0].set_ylabel("pooled F1 (centre ±0.5 s, LOSO-chosen vote)")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "channel_ablation.png", dpi=170)
    print(f"wrote {OUT}/")


if __name__ == "__main__":
    if sys.argv[1:] == ["detect"]:
        m = pd.read_csv(paths.CACHE / "manifest.csv").query("dataset == 'nap'")
        with ProcessPoolExecutor(6) as ex:
            for rec in ex.map(detect_one, zip(m.path, m.rec)):
                print("done", rec, flush=True)
    else:
        analyse()
