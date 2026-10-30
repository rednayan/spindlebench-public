"""
Human ceiling from MASS SS2 annotations (plan_nov_dec.md, workstream A). No signals are needed.

    uv run python experiments/raters/mass_raters.py

MASS SS2 annotations (Borealis doi:10.5683/SP3/Y889CS; cite O'Reilly et al. 2014, J Sleep Res,
doi:10.1111/jsr.12169): 19 young adults, whole nights, C3, R&K stages in 20 s epochs; spindles
scored by expert 1 (all 19) and expert 2 (15). Files in MASS_data/ (not in git).

For the 15 people with both experts, expert 2 is scored against expert 1 (and the reverse; F1 is
symmetric) exactly as a detector would be:
  - IoU >= 0.2, one-to-one (the interval convention of A7/MODA/SUMO; spindlebench.metrics.events.match)
  - onset +-0.5 s (the DREAMS primary rule)
  - NAP-style point scoring: expert 1's spindle centres become point marks, expert 2's intervals are
    the "detections", scored with nap_scoring (centre tolerance sweep 0.25-1.5 s + integrated F1,
    affiliation P/R). This makes the human ceiling directly comparable with the NAP numbers.
Scope: N2 (event midpoint in an R&K stage-2 epoch). CIs: bootstrap over people (2000, seed 0).
Also: density per N2 minute, durations, and timing offsets of matched pairs.
-> results/mass_raters/{per_subject.csv, summary.csv, offsets.csv}
"""
import re

import mne
import numpy as np
import pandas as pd

from spindlebench import paths
from spindlebench.metrics import events as em
from spindlebench.protocols import nap_points as npt
from spindlebench import provenance  # noqa: E402
from spindlebench.protocols import nap_points  # noqa: E402

provenance.track(__file__, protocols=[nap_points.VERSION])

ROOT = paths.ROOT

MASS = paths.raw("mass_ss2_annotations")
OUT = paths.RESULTS / "mass_raters"
EPOCH = 20.0
mne.set_log_level("ERROR")


def events(path):
    a = mne.read_annotations(path)
    e = np.c_[a.onset, a.duration].astype(float)
    return e[np.argsort(e[:, 0])]


def stages(subj):
    a = mne.read_annotations(MASS / f"{subj} Base.edf")
    return [(o, d, re.sub("Sleep stage ", "", s)) for o, d, s in zip(a.onset, a.duration, a.description)]


def n2_fn(st):
    on = np.array([o for o, _, s in st if s == "2"])
    du = np.array([d for _, d, s in st if s == "2"])
    return lambda t: np.array([bool(((on <= x) & (x < on + du)).any()) for x in np.atleast_1d(t)])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    subs = sorted({p.name.split(" ")[0] for p in MASS.glob("*Spindles_E2.edf")})
    rows, offs = [], []
    for s in subs:
        st = stages(s)
        in_n2 = n2_fn(st)
        n2_min = sum(d for _, d, x in st if x == "2") / 60
        e1, e2 = events(MASS / f"{s} Spindles_E1.edf"), events(MASS / f"{s} Spindles_E2.edf")
        e1 = e1[in_n2(e1[:, 0] + e1[:, 1] / 2)]
        e2 = e2[in_n2(e2[:, 0] + e2[:, 1] / 2)]
        row = dict(subject=s, n2_min=n2_min, n_e1=len(e1), n_e2=len(e2),
                   density_e1=len(e1) / n2_min, density_e2=len(e2) / n2_min,
                   dur_e1=np.median(e1[:, 1]), dur_e2=np.median(e2[:, 1]))
        for rule, tol in [("iou", 0.2), ("onset", 0.5)]:
            tp, fp, fn, pairs = em.match(e2, e1, rule, tol)
            row.update({f"{rule}_tp": tp, f"{rule}_fp": fp, f"{rule}_fn": fn})
            if rule == "iou":
                for i, j in pairs:
                    offs.append(dict(subject=s, onset_diff=e2[i, 0] - e1[j, 0],
                                     centre_diff=(e2[i, 0] + e2[i, 1] / 2) - (e1[j, 0] + e1[j, 1] / 2),
                                     dur_e1=e1[j, 1], dur_e2=e2[i, 1]))
        # NAP-style: expert-1 centres as points, expert-2 intervals as detections
        T = max(st[-1][0] + st[-1][1], e1[:, 0].max() + 5, e2[:, 0].max() + 5)
        sc = npt.score_nap(e2[:, 0], e2[:, 0] + e2[:, 1], e1[:, 0] + e1[:, 1] / 2, T)
        for k in npt.KEYS:
            row.update({f"pt_{k}_tp": sc[k][0], f"pt_{k}_fp": sc[k][1], f"pt_{k}_fn": sc[k][2]})
        row.update(aff_P=sc["aff"][0], aff_R=sc["aff"][1])
        rows.append(row)
        print(s, f"E1 {len(e1)} E2 {len(e2)} in N2 ({n2_min:.0f} min)", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "per_subject.csv", index=False)
    o = pd.DataFrame(offs)
    o.to_csv(OUT / "offsets.csv", index=False)

    rng = np.random.default_rng(0)
    W = rng.multinomial(len(d), np.full(len(d), 1 / len(d)), size=2000).astype(float)

    def pooled_f1(prefix, w=None):
        c = d[[f"{prefix}_tp", f"{prefix}_fp", f"{prefix}_fn"]].to_numpy(float)
        tot = c.sum(0) if w is None else w @ c
        tp, fp, fn = (tot[..., 0], tot[..., 1], tot[..., 2])
        return 2 * tp / (2 * tp + fp + fn)

    summ = []
    for name, prefix in [("IoU >= 0.2", "iou"), ("onset +-0.5 s", "onset")] + \
                        [(f"point, centre +-{t} s", f"pt_tau{t}") for t in npt.TAUS] + [("point, containment", "pt_contain")]:
        f, fb = pooled_f1(prefix), pooled_f1(prefix, W)
        summ.append(dict(rule=name, F1=f, lo=np.percentile(fb, 2.5), hi=np.percentile(fb, 97.5)))
    F = np.array([pooled_f1(f"pt_tau{t}") for t in npt.TAUS])
    Fb = np.array([pooled_f1(f"pt_tau{t}", W) for t in npt.TAUS]).T
    integ, integb = npt.integrated(F), npt.integrated(Fb)
    summ.append(dict(rule="point, integrated F1 (0.25-1.5 s)", F1=integ, lo=np.percentile(integb, 2.5),
                     hi=np.percentile(integb, 97.5)))
    P, R = d.aff_P.mean(), d.aff_R.mean()
    summ.append(dict(rule="point, affiliation F1", F1=2 * P * R / (P + R), lo=np.nan, hi=np.nan))
    s = pd.DataFrame(summ)
    s.round(4).to_csv(OUT / "summary.csv", index=False)

    pd.set_option("display.width", 200)
    print(s.round(3).to_string(index=False))
    print(f"\ndensity per N2 min, median: E1 {d.density_e1.median():.2f}, E2 {d.density_e2.median():.2f}; "
          f"E2/E1 ratio median {(d.density_e2 / d.density_e1).median():.2f} (range {(d.density_e2 / d.density_e1).min():.2f}-"
          f"{(d.density_e2 / d.density_e1).max():.2f})")
    print(f"duration median: E1 {d.dur_e1.median():.2f} s, E2 {d.dur_e2.median():.2f} s")
    q = o.drop(columns="subject").quantile([0.25, 0.5, 0.75])
    print(f"matched pairs (IoU>=0.2, n={len(o)}): onset diff median {q.onset_diff[0.5]:+.2f} s "
          f"(IQR {q.onset_diff[0.75] - q.onset_diff[0.25]:.2f}); centre diff median {q.centre_diff[0.5]:+.2f} s "
          f"(IQR {q.centre_diff[0.75] - q.centre_diff[0.25]:.2f})")
    per = d.assign(f1_iou=2 * d.iou_tp / (2 * d.iou_tp + d.iou_fp + d.iou_fn))
    print(f"per-subject F1 (IoU>=0.2): median {per.f1_iou.median():.2f}, range {per.f1_iou.min():.2f}-{per.f1_iou.max():.2f}")


if __name__ == "__main__":
    main()
