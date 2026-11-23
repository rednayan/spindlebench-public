"""
Groundwork (#2 from next_steps.md): Test non-vote label-free selection criteria across all 65 GPU runs.

Compares candidate label-free scores (computable strictly from validation nights, injected validation
nights, and unpaired MASS-SS2 E1 label statistics) against gold-set rank (DREAMS E1, DREAMS E2, NAP)
using Spearman rho and top-1 / top-5 selection regret.

    python eval_selection_criteria.py

Outputs -> results/student/selection_study/
    criteria_comparison.csv
    run_scores.csv
    selection_scatter.png
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import data
import selection
from spindlebench import paths

OUT = data.STUDY / "selection_study"


def load_mass_e1_stats():
    """Unpaired MASS-SS2 E1 target density (events/N2 min) and median duration (s)."""
    f = paths.RESULTS / "mass_raters" / "per_subject.csv"
    df = pd.read_csv(f)
    return dict(
        density_e1_median=float(df["density_e1"].median()),
        density_e1_mean=float(df["density_e1"].mean()),
        density_e1_iqr=float(df["density_e1"].quantile(0.75) - df["density_e1"].quantile(0.25)),
        dur_e1_median=float(df["dur_e1"].median()),
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    mass = load_mass_e1_stats()
    print("MASS-SS2 E1 unpaired priors:", {k: round(v, 3) for k, v in mass.items()})

    frozen = json.loads((selection.SEL / "frozen_gpu.json").read_text())["runs"]
    dreams = pd.read_csv(data.STUDY / "report" / "dreams.csv")
    d_e1 = dreams[dreams.rule == "E1 onset 0.5 (primary)"].set_index("detector")["F1"]
    d_e2 = dreams[dreams.rule == "E2 onset 0.5"].set_index("detector")["F1"]
    nap = pd.read_csv(data.STUDY / "report" / "nap.csv").set_index("detector")["F1_integrated"]

    rows = []
    for run, v in frozen.items():
        if run not in d_e1.index:
            continue
        inj, val = v["injected"], v["validation"]
        f_teachers = [val["f1_a7"], val["f1_luna_spindles"], val["f1_wonambi_ucsd"], val["f1_yasa_default"]]
        n2 = val["per_min_N2"]
        w_n2 = val["W_over_N2"]
        dur = val["dur_median"]
        r6 = inj["recall_6dB"]
        r3 = inj["recall_3dB"]
        prec = inj["precision"]
        fa = inj["false_alarms_per_quiet_min"]
        f1_6 = inj["f1_6dB"]
        f1_avg = inj["f1_avg"]

        # Candidate label-free selection scores (higher = better)
        # 1. Existing vote-copying baseline
        s_mv = val["f1_mv"]
        # 2. Mean / min agreement across the 4 individual teachers (no vote)
        s_mean_t = float(np.mean(f_teachers))
        s_min_t = float(np.min(f_teachers))
        # 3. Raw injected F1 (old Amendment 9/10 criterion)
        s_inj_f1 = f1_avg
        # 4. 6 dB injected F1 alone
        s_f1_6 = f1_6
        # 5. FA- and Wake-penalised faint recall (purely label-free, no teacher agreement)
        s_faint_pen = r6 - 0.10 * val["per_min_W"] - 0.25 * fa
        # 6. Stage-specificity x precision x 6 dB recall (no teacher agreement)
        s_spec_r6 = r6 * prec * max(0.0, 1.0 - w_n2)
        # 7. Pure MASS-E1 plausibility (density + duration distance, no signal/teachers)
        s_mass_only = -(abs(n2 - mass["density_e1_median"]) / max(mass["density_e1_iqr"], 0.5)
                        + abs(dur - mass["dur_e1_median"]) / 0.15)
        # 8. Combined non-vote: 6 dB F1 + MASS-E1 density anchor + Wake specificity
        s_f1_6_mass = f1_6 - 0.08 * abs(n2 - mass["density_e1_median"]) - 0.15 * w_n2
        # 9. Combined non-vote: 3+6 dB recall + precision + MASS-E1 + Wake (no teacher agreement at all!)
        s_composite = 0.5 * (r3 + r6) * prec - 0.05 * abs(n2 - mass["density_e1_median"]) - 0.10 * w_n2 - 0.10 * fa

        rows.append(dict(
            run=run,
            arm="_".join(run.split("_")[:-1]),
            dreams_e1=float(d_e1[run]),
            dreams_e2=float(d_e2[run]) if run in d_e2.index else np.nan,
            nap_f1=float(nap[run]) if run in nap.index else np.nan,
            val_f1_mv=s_mv,
            mean_teacher_f1=s_mean_t,
            min_teacher_f1=s_min_t,
            inj_f1_avg=s_inj_f1,
            inj_f1_6dB=s_f1_6,
            faint_pen_wake_fa=s_faint_pen,
            spec_x_prec_x_r6=s_spec_r6,
            mass_e1_plaus=s_mass_only,
            f1_6dB_mass_wake=s_f1_6_mass,
            composite_nonvote=s_composite,
        ))

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "run_scores.csv", index=False)

    criteria = [
        ("val_f1_mv", "Vote agreement (current, copies vote)"),
        ("mean_teacher_f1", "Mean of 4 teachers"),
        ("min_teacher_f1", "Min of 4 teachers"),
        ("inj_f1_avg", "Injected F1 avg (old)"),
        ("inj_f1_6dB", "Injected 6 dB F1"),
        ("faint_pen_wake_fa", "6 dB recall - Wake - QuietN2 FA"),
        ("spec_x_prec_x_r6", "6 dB recall × Prec × (1 - W/N2)"),
        ("mass_e1_plaus", "MASS-E1 density + duration only"),
        ("f1_6dB_mass_wake", "6 dB F1 + MASS-E1 + W/N2"),
        ("composite_nonvote", "Composite non-vote (3&6dB + Prec + MASS + W/N2)"),
    ]

    best_e1 = df["dreams_e1"].max()
    summary = []
    for col, desc in criteria:
        r_e1, _ = spearmanr(df[col], df["dreams_e1"])
        r_e2, _ = spearmanr(df[col], df["dreams_e2"], nan_policy="omit")
        r_nap, _ = spearmanr(df[col], df["nap_f1"], nan_policy="omit")
        top1 = df.loc[df[col].idxmax()]
        top5 = df.nlargest(5, col)
        summary.append(dict(
            criterion=col,
            description=desc,
            rho_dreams_e1=r_e1,
            rho_dreams_e2=r_e2,
            rho_nap=r_nap,
            top1_run=top1["run"],
            top1_dreams_e1=top1["dreams_e1"],
            top5_mean_dreams_e1=top5["dreams_e1"].mean(),
            top1_regret_e1=best_e1 - top1["dreams_e1"],
            top5_mean_nap=top5["nap_f1"].mean(),
        ))
    sdf = pd.DataFrame(summary).sort_values("rho_dreams_e1", ascending=False)
    sdf.to_csv(OUT / "criteria_comparison.csv", index=False)

    print("\n=== Non-Vote Selection Criteria across all 65 GPU runs ===")
    show = sdf[["criterion", "rho_dreams_e1", "rho_dreams_e2", "rho_nap",
                "top1_dreams_e1", "top5_mean_dreams_e1", "top1_run"]]
    print(show.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # Plot top 4 criteria vs DREAMS E1 F1
    top_cols = ["val_f1_mv", "f1_6dB_mass_wake", "spec_x_prec_x_r6", "mass_e1_plaus"]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), dpi=200, sharey=True)
    for ax, col in zip(axes, top_cols):
        rho = spearmanr(df[col], df["dreams_e1"]).statistic
        is_hl = df["arm"] == "student_rules4_fixA_gpu"
        ax.scatter(df.loc[~is_hl, col], df.loc[~is_hl, "dreams_e1"], alpha=0.65, s=30, label="Other 12 arms")
        ax.scatter(df.loc[is_hl, col], df.loc[is_hl, "dreams_e1"], color="#d62728", s=55, label="Headline (fixA)")
        ax.set_title(f"{col}\nSpearman ρ = {rho:.3f}", fontsize=9.5)
        ax.set_xlabel("Selection score")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("DREAMS E1 F1")
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Label-free selection criteria vs. DREAMS E1 F1 (65 GPU runs)", fontsize=11, y=1.01)
    fig.tight_layout()
    fig.savefig(OUT / "selection_scatter.png", bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
