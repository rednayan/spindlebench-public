"""
RQ2 figure: pooled F1 as a function of matching strictness, per detector, as in the
RED/SEED papers' F1-vs-IoU curves. Reads results/benchmark/per_recording.csv.

    uv run python experiments/rq2_validity/protocol_curves.py      # -> results/validity/protocol_curves_<dataset>.png

Left panel: onset (or centre, on NAP) tolerance 0.25-1.5 s. Right panel (DREAMS only): IoU
threshold 0.1-0.5. Crossing curves mean the detector ranking depends on the strictness chosen.
N2 scope (the RQ1 primary protocol). Only detectors that cover every recording are drawn; the
five best under the main protocol are labelled (colour, marker and an end label, so they stay
apart without colour; * = in-sample on this dataset), the others are grey.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from spindlebench import paths, plotting, registry
from spindlebench import provenance  # noqa: E402

provenance.track(__file__)

per = pd.read_csv(paths.RESULTS / "benchmark/per_recording.csv")
per = per[~per.excluded & ~per.detector.isin(registry.USES_EXPERT_STAGING)]
OUT = paths.RESULTS / "validity"
PANELS = {"dreams_spindles": [("expert1", "onset", "onset tolerance (s)"), ("expert1", "iou", "IoU threshold")],
          "dreams_kc": [("expert1", "onset", "onset tolerance (s)"), ("expert1", "iou", "IoU threshold")],
          "nap": [("expert", "center", "centre tolerance (s)")]}
EVENT = {"dreams_spindles": "spindle", "dreams_kc": "kcomplex", "nap": "spindle"}
HIGHLIGHT = [("#0072B2", "o"), ("#D55E00", "s"), ("#009E73", "^"), ("#CC79A7", "D"), ("#E69F00", "v")]  # Okabe-Ito


def pooled(g):
    return 2 * g.tp.sum() / (2 * g.tp.sum() + g.fp.sum() + g.fn.sum())


for ds, panels in PANELS.items():
    g = per[(per.dataset == ds) & (per.event == EVENT[ds]) & (per.scope == "N2")]
    cover = g.groupby("detector").rec.nunique()
    g = g[g.detector.isin(cover[cover == g.rec.nunique()].index)]
    if g.empty:
        continue
    curves = g.groupby(["reference", "rule", "detector", "tol"]).apply(pooled, include_groups=False).rename("F1").reset_index()
    ref0, rule0, _ = panels[0]
    main_tol = 0.5
    main = curves.query("reference == @ref0 and rule == @rule0 and tol == @main_tol").set_index("detector").F1
    top = list(main.sort_values(ascending=False).index[:5])
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 4), squeeze=False)
    for ax, (ref, rule, xlabel) in zip(axes[0], panels):
        c = curves.query("reference == @ref and rule == @rule")
        for det, d in c.groupby("detector"):
            if det in top:
                continue
            ax.plot(d.tol, d.F1, color="#c3c2b7", lw=1, zorder=1)
        for det, (col, mk) in zip(top, HIGHLIGHT):
            d = c[c.detector == det].sort_values("tol")
            name = det + (" *" if det in registry.IN_SAMPLE.get(ds, ()) else "")
            ax.plot(d.tol, d.F1, "-", marker=mk, color=col, lw=2, ms=5, label=name, zorder=2)
            ax.annotate(name, (d.tol.iloc[-1], d.F1.iloc[-1]), xytext=(4, 0), textcoords="offset points",
                        fontsize=7, color=col, va="center")
        ax.set_xlabel(xlabel)
        ax.set_ylim(0, 1)
        ax.yaxis.grid(True, color=plotting.GRID, lw=0.8)
        plotting.style(ax)
    axes[0][0].set_ylabel(f"pooled F1 vs {ref0} (N2)")
    axes[0][0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"{ds}: F1 vs matching strictness (grey: other detectors)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / f"protocol_curves_{ds}.png", dpi=170)
    plt.close(fig)
    print("wrote", OUT / f"protocol_curves_{ds}.png")
