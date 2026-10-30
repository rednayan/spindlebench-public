"""
Step 3: train one SUMO model for one fold and write its held-out probabilities.

    detectors/sumo/.venv-sumo/Scripts/python.exe experiments/rq4_adaptation/sumo_finetune/train.py --fold 0 --rung ft_all --seed 0

rung   ft_n1 / ft_n3 / ft_n6 / ft_all : released MODA weights, fine-tuned on the first 1 / 3 / 6 / all
                                        training subjects of the fold (fixed order in folds.csv -> nested)
       scratch                         : same architecture, random init (seeded), all training subjects

Training follows SUMO's own training step (sumo/model/sumo.py: training_step): GeneralizedDiceLoss on
the dense logits (no moving average), Adam (betas 0.9/0.999, eps 1e-8, no weight decay), batch 12,
BatchNorm in train mode. Input = random 60-s crops inside expert-N2 runs of the training naps, cut
from the whole-nap preprocessed signal (filters + 100 Hz + whole-recording z-score, i.e. the same
signal the model sees at inference). Target = +-0.4 s around each expert mark.
One "epoch" = a fixed number of optimiser steps (default 25 x 12 crops = 5 h of crops), so the
budget is the same for every rung. After every epoch the model runs stage-free on the whole
validation naps; val metric = best pooled F1 over the threshold grid (N2 scope, centre +-0.5 s).
Early stopping on that metric; the kept model is the best epoch >= 1 and its best threshold is the
operating point used on the test naps.
"""
import argparse
import time

import numpy as np
import pandas as pd

import ftlib as L

CROP = 6000  # 60 s @ 100 Hz


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--rung", required=True, choices=["ft_n1", "ft_n3", "ft_n6", "ft_all", "scratch"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=None, help="default: 1e-3 fine-tune, 5e-3 scratch")
    p.add_argument("--max-epochs", type=int, default=100)
    p.add_argument("--steps", type=int, default=25, help="optimiser steps per epoch")
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--batch", type=int, default=12)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--tag", default="", help="suffix for pilot runs")
    p.add_argument("--no-test", action="store_true", help="pilot: do not write test probabilities")
    return p.parse_args()


def det_id(rung, seed, tag=""):
    return f"sumo_{rung}" + (f"_s{seed}" if seed else "") + tag


class Sampler:
    """uniform random 60-s crops over all valid N2 positions of the training naps"""

    def __init__(self, recs, rng):
        self.rng = rng
        self.y, self.m, self.starts = [], [], []
        for r in recs:
            p = L.load_prep("nap", r)
            for a, b in L.n2_runs(p["n2"], CROP):
                self.y.append(p["y"])
                self.m.append(p["mask"])
                self.starts.append((a, b - CROP))
        w = np.array([b - a + 1 for a, b in self.starts], float)
        self.w = w / w.sum()
        self.n2_minutes = (w + CROP - 1).sum() / L.FS / 60

    def batch(self, n):
        import torch
        k = self.rng.choice(len(self.starts), size=n, p=self.w)
        xs, ms = [], []
        for i in k:
            a, b = self.starts[i]
            s = int(self.rng.integers(a, b + 1))
            xs.append(self.y[i][s:s + CROP])
            ms.append(self.m[i][s:s + CROP])
        return torch.from_numpy(np.stack(xs)).float(), torch.from_numpy(np.stack(ms).astype(np.int64))


def validate(model, val):
    tot = sum(L.counts_over_thresholds(L.predict_prob(model, p["y"]), p) for p in val.values())
    thr, f1 = L.best_threshold(tot)
    f05 = float(L.f1_of(tot)[np.argmin(np.abs(L.THRESHOLDS - 0.5))])
    return thr, f1, f05


def main():
    a = parse()
    import torch
    torch.set_num_threads(a.threads)
    torch.manual_seed(1000 * a.seed + a.fold)
    rng = np.random.default_rng(1000 * a.seed + a.fold)
    did = det_id(a.rung, a.seed, a.tag)
    out = L.WORK / "models" / did
    out.mkdir(parents=True, exist_ok=True)

    f = L.folds()
    fk = f[f.fold == a.fold]
    recs = lambda s: sum([r.split(";") for r in s.recs], [])  # noqa: E731
    train_s = fk[fk.role == "train"].sort_values("train_rank")
    n = {"ft_n1": 1, "ft_n3": 3, "ft_n6": 6}.get(a.rung, len(train_s))
    train_s = train_s.iloc[:n]
    train_recs = recs(train_s)
    val_recs = recs(fk[fk.role == "val"])
    test_recs = recs(fk[fk.role == "test"])
    assert not set(train_recs) & set(val_recs) and not set(train_recs + val_recs) & set(test_recs)

    init = "random" if a.rung == "scratch" else "released"
    lr = a.lr if a.lr is not None else (5e-3 if init == "random" else 1e-3)
    model = L.make_model(init)
    cfg = L.get_config()
    crit = cfg.loss(**cfg.loss_params)
    params = dict(cfg.optimizer_params)
    params["lr"] = lr
    opt = cfg.optimizer(model.parameters(), **params)
    sampler = Sampler(train_recs, rng)
    val = {r: L.load_prep("nap", r) for r in val_recs}

    t0 = time.time()
    thr, f1, f05 = validate(model, val)
    hist = [dict(epoch=0, step=0, train_loss=np.nan, val_f1=f1, val_thr=thr, val_f1_at05=f05, sec=time.time() - t0)]
    print(did, a.fold, "epoch 0", f"val F1 {f1:.3f} @ {thr}", flush=True)
    best, best_state, bad = -1.0, None, 0
    for ep in range(1, a.max_epochs + 1):
        model.train()
        losses = []
        for _ in range(a.steps):
            x, y = sampler.batch(a.batch)
            opt.zero_grad()
            loss = crit(model.dense_logits(x), y)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        thr, f1, f05 = validate(model, val)
        hist.append(dict(epoch=ep, step=ep * a.steps, train_loss=float(np.mean(losses)), val_f1=f1, val_thr=thr,
                         val_f1_at05=f05, sec=time.time() - t0))
        print(did, a.fold, f"epoch {ep}", f"loss {np.mean(losses):.4f}", f"val F1 {f1:.3f} @ {thr}",
              f"({time.time() - t0:.0f}s)", flush=True)
        if f1 > best + 1e-9:
            best, bad = f1, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_ep, best_thr = ep, thr
        else:
            bad += 1
            if bad >= a.patience:
                break
    train_sec = time.time() - t0
    pd.DataFrame(hist).to_csv(out / f"fold{a.fold}_history.csv", index=False)
    torch.save({"state_dict": best_state}, out / f"fold{a.fold}.pt")
    model.load_state_dict(best_state)
    info = dict(id=did, fold=a.fold, rung=a.rung, seed=a.seed, init=init, lr=lr, batch=a.batch, crop_s=CROP / L.FS,
                steps_per_epoch=a.steps, max_epochs=a.max_epochs, patience=a.patience, epochs_run=len(hist) - 1,
                best_epoch=best_ep, threshold=best_thr, val_f1=best, train_subjects=[int(s) for s in train_s.subject],
                train_recs=train_recs, val_recs=val_recs, test_recs=test_recs, train_n2_minutes=sampler.n2_minutes,
                train_seconds=round(train_sec, 1), threads=a.threads)
    if not a.no_test:
        pdir = L.WORK / "prob" / did
        pdir.mkdir(parents=True, exist_ok=True)
        for r in test_recs:
            np.save(pdir / f"nap__{r}.npy", L.predict_prob(model, L.load_prep("nap", r)["y"]).astype(np.float32))
        if a.rung in ("ft_all", "scratch"):  # forgetting check on DREAMS (never trained on)
            m = L.manifest()
            for r in m[m.dataset == "dreams_spindles"].rec:
                np.save(pdir / f"dreams_spindles__{r}__fold{a.fold}.npy",
                        L.predict_prob(model, L.load_prep("dreams_spindles", r)["y"]).astype(np.float32))
    L.save_json(out / f"fold{a.fold}.json", info)
    print("DONE", did, a.fold, f"best epoch {best_ep}, val F1 {best:.3f} @ {best_thr}, {train_sec:.0f}s", flush=True)


if __name__ == "__main__":
    main()
