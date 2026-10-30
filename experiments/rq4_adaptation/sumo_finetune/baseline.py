"""
Step 2: threshold rungs with the released SUMO weights (no training).

    detectors/sumo/.venv-sumo/Scripts/python.exe experiments/rq4_adaptation/sumo_finetune/baseline.py

  sumo_zeroshot  threshold 0.5 (== the published softmax argmax); must equal results/detections/*/sumo
  sumo_recal     threshold maximising pooled N2 F1 (centre +-0.5 s) on all non-test subjects of the
                 fold (train + val: the released model was trained on none of them)
  sumo_persubj   threshold maximising F1 on the same subject's other included nap; subjects with a
                 single included nap, or whose other nap has no N2 (suj19_d2), fall back to sumo_recal
DREAMS: sumo_zeroshot, and sumo_recal with the mean of the 5 fold thresholds (operating point
transferred from NAP; separates the threshold shift from the weight change in sumo_ft_all).
"""
import time

import numpy as np
import pandas as pd

import ftlib as L


def main():
    import torch
    torch.set_num_threads(8)
    t0 = time.time()
    model = L.make_model("released")
    inc = L.nap_included()
    m = L.manifest()
    dreams = m[m.dataset == "dreams_spindles"]
    prob, prep = {}, {}
    for row in inc.itertuples():
        prep[row.rec] = L.load_prep("nap", row.rec)
        prob[row.rec] = L.predict_prob(model, prep[row.rec]["y"])
    counts = {r: L.counts_over_thresholds(prob[r], prep[r]) for r in prob}
    f = L.folds()
    rec_subj = dict(zip(inc.rec, inc.subject))
    thr_rows = []
    for row in inc.itertuples():
        L.write_detections("nap", "sumo_zeroshot", row.rec, L.events(prob[row.rec], 0.5), "C3")
    for k in range(5):
        fk = f[f.fold == k]
        test_s = set(fk[fk.role == "test"].subject)
        calib = [r for r in prob if rec_subj[r] not in test_s]
        thr, f1c = L.best_threshold(sum(counts[r] for r in calib))
        thr_rows.append(dict(rung="sumo_recal", fold=k, rec="*", threshold=thr, calib_f1=f1c, calib=";".join(calib)))
        for r in prob:
            if rec_subj[r] not in test_s:
                continue
            L.write_detections("nap", "sumo_recal", r, L.events(prob[r], thr), "C3")
            other = [o for o in prob if rec_subj[o] == rec_subj[r] and o != r]
            pt, pf = (L.best_threshold(counts[other[0]]) if other else (None, np.nan))
            src = other[0] if pt is not None else "fallback:sumo_recal"
            pt = thr if pt is None else pt
            thr_rows.append(dict(rung="sumo_persubj", fold=k, rec=r, threshold=pt, calib_f1=pf, calib=src))
            L.write_detections("nap", "sumo_persubj", r, L.events(prob[r], pt), "C3")
    thr = pd.DataFrame(thr_rows)
    mean_thr = float(thr[thr.rung == "sumo_recal"].threshold.mean())
    for row in dreams.itertuples():
        p1 = L.predict_prob(model, L.load_prep("dreams_spindles", row.rec)["y"])
        np.save(L.WORK / "prep" / "dreams_spindles" / f"{row.rec}__released_prob.npy", p1.astype(np.float32))
        L.write_detections("dreams_spindles", "sumo_zeroshot", row.rec, L.events(p1, 0.5), row.default_channel)
        L.write_detections("dreams_spindles", "sumo_recal", row.rec, L.events(p1, mean_thr), row.default_channel)
    thr = pd.concat([thr, pd.DataFrame([dict(rung="sumo_recal", fold="dreams", rec="*", threshold=mean_thr,
                                             calib="mean of the 5 fold thresholds")])])
    thr.to_csv(L.WORK / "thresholds_baseline.csv", index=False)
    print(thr.to_string(index=False))

    # sanity check: sumo_zeroshot == the benchmark's sumo detections
    bad = 0
    for row in pd.concat([inc, dreams]).itertuples():
        a = pd.read_csv(L.DET / row.dataset / "sumo_zeroshot" / f"{row.rec}.csv")
        b = pd.read_csv(L.ROOT / "results" / "detections" / row.dataset / "sumo" / f"{row.rec}.csv")
        b = b[b.channel.astype(str) == str(row.default_channel)].reset_index(drop=True)
        same = len(a) == len(b) and np.allclose(a[["onset", "duration"]].to_numpy(float), b[["onset", "duration"]].to_numpy(float), atol=0.011) \
            and np.allclose(a.score.to_numpy(float), b.score.to_numpy(float), atol=2e-4)
        if not same:
            bad += 1
            print("MISMATCH", row.dataset, row.rec, len(a), len(b))
    print(f"zero-shot reproduction: {bad} mismatching recordings of {len(inc) + len(dreams)}")
    print(f"baseline done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
