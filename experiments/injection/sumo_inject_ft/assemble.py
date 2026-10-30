"""
Step 4: baselines and detections for every arm, in results/injection/detections/<dataset>/<id>/.

    detectors/sumo/.venv-sumov2/Scripts/python.exe experiments/injection/sumo_inject_ft/assemble.py
    uv run python experiments/rq1_benchmark/evaluate.py --store results/injection/detections --datasets nap,dreams_spindles --out results/injection/benchmark

Baselines (released weights):
  sumov2_zeroshot   threshold 0.5 (= published argmax; checked against results/detections/*/sumov2)
  sumov2_recal      threshold = best pooled F1 on the real labels of the fold's train + val recordings (arm A1)
  sumov2_injrecal   threshold = best F1 on injected events of the same recordings (label-free A1)
Trained arms: fold model on its test recordings at its validation threshold.
NAP-trained arms on DREAMS (`<id>_x`): mean of the 5 fold probabilities at the mean fold threshold.
"""
import json

import numpy as np
import pandas as pd

import lib as L
from train import split

NFOLDS = {"nap": 5, "dreams_spindles": 8}


def channel(ds, rec):
    m = L.manifest()
    return m[(m.dataset == ds) & (m.rec == rec)].default_channel.iloc[0]


def baselines():
    model = L.make_model()
    for ds, nf in NFOLDS.items():
        recs = L.recordings(ds).rec.tolist()
        prep = {r: L.load_prep(ds, r) for r in recs}
        prob = {r: L.predict_prob(model, prep[r]["y"]) for r in recs}
        prob_i = {r: L.predict_prob(model, prep[r]["y0"]) for r in recs}
        for r in recs:
            L.write_detections(ds, "sumov2_zeroshot", r, L.events(prob[r], 0.5), channel(ds, r))
        thr = []
        for k in range(nf):
            tr, va, te = split(ds, k)
            cr = sum(L.counts(prob[r], *L.real_scope(prep[r], ds)) for r in tr + va)
            ci = sum(L.counts(prob_i[r], *L.inj_scope(prep[r], 0)) for r in tr + va)
            t_r, _ = L.best_threshold(cr)
            t_i, _ = L.best_threshold(ci)
            thr.append(dict(dataset=ds, fold=k, recal=t_r, injrecal=t_i))
            for r in te:
                L.write_detections(ds, "sumov2_recal", r, L.events(prob[r], t_r), channel(ds, r))
                L.write_detections(ds, "sumov2_injrecal", r, L.events(prob[r], t_i), channel(ds, r))
        print(pd.DataFrame(thr).to_string(), flush=True)
        pd.DataFrame(thr).to_csv(L.WORK / f"baseline_thresholds_{ds}.csv", index=False)


def trained():
    rows = []
    for mdir in sorted((L.WORK / "models").iterdir()):
        oid = mdir.name
        if oid.endswith("_pilot"):
            continue
        for ds, nf in NFOLDS.items():
            infos = [json.loads((mdir / f"{ds}_fold{k}.json").read_text()) for k in range(nf)
                     if (mdir / f"{ds}_fold{k}.json").exists()]
            if not infos:
                continue
            if len(infos) < nf:
                print(f"skip {oid} {ds}: {len(infos)}/{nf} folds")
                continue
            pdir = L.WORK / "prob" / oid
            for info in infos:
                thr = info["threshold"] if info["threshold"] is not None else 0.5
                for r in info["test_recs"]:
                    p1 = np.load(pdir / f"{ds}__{r}.npy")
                    L.write_detections(ds, oid, r, L.events(p1, thr), channel(ds, r))
                rows.append(dict(id=oid, dataset=ds, fold=info["fold"], threshold=info["threshold"],
                                 best_epoch=info["best_epoch"], epochs_run=info["epochs_run"], val_f1=info["val_f1"],
                                 train_seconds=info["train_seconds"]))
            if ds == "nap":
                thr = float(np.mean([i["threshold"] for i in infos if i["threshold"] is not None]))
                for r in L.recordings("dreams_spindles").rec:
                    p1 = np.mean([np.load(pdir / f"dreams_spindles__{r}__fold{k}.npy") for k in range(nf)], axis=0)
                    L.write_detections("dreams_spindles", f"{oid}_x", r, L.events(p1, thr), channel("dreams_spindles", r))
    s = pd.DataFrame(rows)
    s.to_csv(L.WORK / "training_summary.csv", index=False)
    if len(s):
        print(s.groupby(["id", "dataset"])[["threshold", "best_epoch", "val_f1", "train_seconds"]].mean().round(3).to_string())


if __name__ == "__main__":
    baselines()
    trained()
