"""
RQ4 "cheapest adequate adaptation", on a label-budget axis.

    uv run python experiments/rq4_adaptation/rq4_label_budget.py        (after experiments/rq4_adaptation/adaptation_ladder.py)

x = labelled NAP subjects a rung needs per held-out subject (0 = no NAP labels; per-subject
calibration uses the subject's own other nap, plotted as 1); y = pooled N2 F1 with a subject-bootstrap
95% CI, on the same 26 naps for every rung (results/rq4/ladder_summary.csv, F1_common).

Adequacy target: the best detector that needs no NAP labels at all (the zero-budget column). A rung
whose CI reaches that F1 is not distinguishable from it (a weak criterion with 13 subjects); a rung
that clearly beats it would justify its label cost. Oracle thresholds are
tuned on the test nap, so they are a ceiling, not a rung.
Writes results/rq4/label_budget.csv and label_budget.png.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from spindlebench import paths, plotting
from spindlebench import provenance  # noqa: E402

provenance.track(__file__)

OUT = paths.RESULTS / "rq4"
# rung -> (label, NAP subjects labelled, family); SUMO trains on 17 - test (3-4) - val (2) = 11-12
RUNGS = {
    "yasa_default": ("YASA default", 0, "yasa"), "yasa_transfer": ("YASA DREAMS-tuned", 0, "yasa"),
    "moelle_default": ("Mölle, paper recipe", 0, "moelle"), "wonambi_moelle2011": ("Mölle, wonambi", 0, "moelle"),
    "sumo_zeroshot": ("SUMO zero-shot", 0, "sumo"), "sumov2": ("SUMOv2 zero-shot", 0, "sumo"),
    "yasa_per_subject": ("YASA per subject", 1, "yasa"), "moelle_per_subject": ("Mölle per subject", 1, "moelle"),
    "sumo_ft_n1": ("SUMO ft, 1 subj.", 1, "sumo"), "sumo_ft_n3": ("SUMO ft, 3 subj.", 3, "sumo"),
    "sumo_ft_n6": ("SUMO ft, 6 subj.", 6, "sumo"), "sumo_ft_all_3seed": ("SUMO ft, all", 11.5, "sumo"),
    "sumo_ft_all_lr5e-3_3seed": ("SUMO ft lr 5e-3, all", 11.5, "sumo"),
    "sumo_scratch_3seed": ("SUMO scratch, all", 11.5, "sumo"),
    "yasa_population": ("YASA population", 16, "yasa"), "moelle_population": ("Mölle population", 16, "moelle"),
}
COLOR = {"yasa": plotting.C_YASA, "sumo": plotting.C_E1, "moelle": "#52514e"}


def main():
    s = pd.read_csv(OUT / "ladder_summary.csv").query("scope == 'N2'").set_index("rung")
    t = pd.DataFrame([dict(rung=r, label=l, subjects=n, family=f, F1=s.F1_common[r], lo=s.common_lo[r],
                           hi=s.common_hi[r]) for r, (l, n, f) in RUNGS.items() if r in s.index])
    target = t[t.subjects == 0].F1.max()
    t["ci_reaches_target"] = t.hi >= target  # weak: marginal CIs are wide with 13 subjects
    t.sort_values(["subjects", "F1"]).to_csv(OUT / "label_budget.csv", index=False)
    print(f"target (best zero-label rung): {target:.3f} ({t.loc[t[t.subjects == 0].F1.idxmax(), 'label']})")
    print(t.sort_values(["subjects", "F1"]).round(3).to_string(index=False))

    # one row per rung, grouped by label budget (cheapest at the top)
    t = t.sort_values(["subjects", "F1"], ascending=[True, False]).reset_index(drop=True)
    y = t.index.to_numpy() + t.subjects.rank(method="dense").to_numpy() * 0.6  # gap between budget groups
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axvline(target, color=plotting.INK2, ls="--", lw=1)
    ax.text(target + 0.005, -0.4, "best with no NAP labels", fontsize=8, color=plotting.INK2)
    for fam, g in t.groupby("family"):
        ax.errorbar(g.F1, y[g.index], xerr=[g.F1 - g.lo, g.hi - g.F1], fmt="o", color=COLOR[fam], ms=5, capsize=2,
                    lw=1, label={"yasa": "YASA thresholds", "moelle": "Mölle 2011", "sumo": "SUMO"}[fam])
    budget = lambda r: ("1 nap, own" if r.rung.endswith("per_subject") else
                        {0: "0", 1: "1", 3: "3", 6: "6", 11.5: "11-12", 16: "16"}[r.subjects])
    ax.set_yticks(y, [f"{r.label}  [{budget(r)}]" for r in t.itertuples()], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("pooled F1, N2 (same 26 naps, C3, centre ±0.5 s), subject-bootstrap 95% CI")
    ax.set_title("[n] = labelled NAP subjects needed per held-out subject", fontsize=9, color=plotting.INK2, loc="left")
    ax.set_xlim(0, 0.55)
    ax.xaxis.grid(True, color=plotting.GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    plotting.style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "label_budget.png", dpi=170)
    print(f"wrote {OUT}/label_budget.*")


if __name__ == "__main__":
    main()
