"""
Step 4: turn the saved held-out probabilities of every trained rung into detection files, average
the 5 fold models on DREAMS, and plot the training curves.

    detectors/sumo/.venv-sumo/Scripts/python.exe experiments/rq4_adaptation/sumo_finetune/assemble.py

NAP: each included nap gets the detections of the model of the fold where it is held out, at that
model's validation threshold.
DREAMS (forgetting check, never in training): the 5 fold models' p(spindle) are averaged per sample
(each fold model sees the identical preprocessed excerpt), then thresholded at the mean of the 5
fold thresholds; written as <id> under dreams_spindles. `<id>_t05` uses the published 0.5 instead.
`<id>_fold<k>` is fold model k alone at its own validation threshold (no ensembling).
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import ftlib as L


def main():
    m = L.manifest()
    infos = [json.loads(p.read_text()) for p in sorted((L.WORK / "models").glob("*/fold*.json"))]
    infos = [i for i in infos if "_pilot" not in i["id"] and "_smoke" not in i["id"]]
    tab = pd.DataFrame(infos)
    ids = sorted(tab.id.unique())
    summary = []
    for did in ids:
        t = tab[tab.id == did].sort_values("fold")
        if len(t) != 5:
            print("incomplete", did, list(t.fold))
            continue
        pdir = L.WORK / "prob" / did
        for i in t.itertuples():
            for r in i.test_recs:
                L.write_detections("nap", did, r, L.events(np.load(pdir / f"nap__{r}.npy"), i.threshold), "C3")
        if (pdir / "dreams_spindles__excerpt1__fold0.npy").exists():
            thr = float(t.threshold.mean())
            for row in m[m.dataset == "dreams_spindles"].itertuples():
                p = np.mean([np.load(pdir / f"dreams_spindles__{row.rec}__fold{k}.npy") for k in range(5)], axis=0)
                L.write_detections("dreams_spindles", did, row.rec, L.events(p, thr), row.default_channel)
                L.write_detections("dreams_spindles", did + "_t05", row.rec, L.events(p, 0.5), row.default_channel)
                # single fold models at their own thresholds: separates NAP training from 5-model ensembling
                for i in t.itertuples():
                    pk = np.load(pdir / f"dreams_spindles__{row.rec}__fold{i.fold}.npy")
                    L.write_detections("dreams_spindles", f"{did}_fold{i.fold}", row.rec, L.events(pk, i.threshold),
                                       row.default_channel)
        summary.append(dict(id=did, thresholds=" ".join(f"{x:.3f}" for x in t.threshold),
                            best_epochs=" ".join(str(x) for x in t.best_epoch),
                            epochs_run=" ".join(str(x) for x in t.epochs_run),
                            val_f1=" ".join(f"{x:.3f}" for x in t.val_f1),
                            train_n2_min=" ".join(f"{x:.0f}" for x in t.train_n2_minutes),
                            train_min_total=round(t.train_seconds.sum() / 60, 1),
                            train_min_max=round(t.train_seconds.max() / 60, 1)))
    s = pd.DataFrame(summary)
    s.to_csv(L.WORK / "training_summary.csv", index=False)
    print(s.to_string(index=False))
    plot_curves(ids)


def plot_curves(ids):
    out = L.HERE / "curves"
    out.mkdir(exist_ok=True)
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#8e5bd0", "#c9a227"]
    for did in ids:
        hs = [(k, L.WORK / "models" / did / f"fold{k}_history.csv") for k in range(5)]
        hs = [(k, pd.read_csv(p)) for k, p in hs if p.exists() and p.with_name(f"fold{k}.json").exists()]
        if not hs:
            continue
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
        for (k, h), c in zip(hs, colors):
            ax[0].plot(h.epoch, h.train_loss, color=c, lw=1.3, label=f"fold {k}")
            ax[1].plot(h.epoch, h.val_f1, color=c, lw=1.3, label=f"fold {k}")
            ax[1].plot(h.epoch, h.val_f1_at05, color=c, lw=0.8, ls=":")
            info = json.loads((L.WORK / "models" / did / f"fold{k}.json").read_text())
            ax[1].plot(info["best_epoch"], info["val_f1"], "o", color=c, ms=5)
        ax[0].set(xlabel="epoch (25 steps x 12 crops of 60 s)", ylabel="train loss (generalised Dice)", title=did)
        ax[1].set(xlabel="epoch", ylabel="val F1 (N2, centre ±0.5 s)",
                  title="solid: best threshold, dotted: 0.5; dot = kept epoch")
        ax[1].set_ylim(0, 0.7)
        for a in ax:
            a.spines[["top", "right"]].set_visible(False)
            a.grid(True, color="#e4e3df", lw=0.8)
        ax[1].legend(fontsize=8, frameon=False)
        fig.tight_layout()
        fig.savefig(out / f"{did}.png", dpi=130)
        plt.close(fig)


if __name__ == "__main__":
    main()
