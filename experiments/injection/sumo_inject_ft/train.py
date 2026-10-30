"""
Step 2: fine-tune SUMOv2 for one fold and one arm; write held-out probabilities.

    detectors/sumo/.venv-sumov2/Scripts/python.exe experiments/injection/sumo_inject_ft/train.py --dataset nap --fold 0 --arm inj --seed 0

arm  real : expert targets (NAP +-0.4 s around marks; DREAMS expert-1 intervals, loss only 0-1000 s);
            early stopping / threshold on real validation labels                          (plan arm D)
     self : SUMOv2's own zero-shot detections as targets; selection on injected validation events (arm B)
     inj  : injected spindles as targets, loss only on quiet N2 + injected events; selection on
            injected validation events (arm C). The same run also tracks real-label validation and
            keeps that model separately as `inj_rv` (arm C').

Same recipe as the SUMO RQ4 runs: released weights, Adam lr 1e-3, batch 12, random 60-s crops from
expert-N2 runs >= 60 s of the training recordings, epoch = 25 steps, max 100 epochs, patience 15,
kept epoch >= 1. SUMOv2's own loss (generalised Dice on dense logits; masked, see lib.masked_gdl) and
its amplitude augmentation (scale x U(0.5, 1) or x U(1, 2)).
"""
import argparse
import time

import numpy as np
import pandas as pd

import lib as L

CROP = 60 * L.FS


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["nap", "dreams_spindles"])
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--arm", required=True, choices=["real", "self", "inj"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max-epochs", type=int, default=100)
    p.add_argument("--steps", type=int, default=25)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--batch", type=int, default=12)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--tag", default="")
    p.add_argument("--no-test", action="store_true")
    return p.parse_args()


def det_id(arm, seed, tag=""):
    return f"sumov2_{arm}" + (f"_s{seed}" if seed else "") + tag


def split(dataset, fold):
    if dataset == "nap":
        f = pd.read_csv(L.WORK / "folds_nap.csv")
        f = f[f.fold == fold]
        get = lambda role: sum([r.split(";") for r in f[f.role == role].recs], [])  # noqa: E731
    else:
        f = pd.read_csv(L.WORK / "folds_dreams.csv")
        f = f[f.fold == fold]
        get = lambda role: f[f.role == role].rec.tolist()  # noqa: E731
    return get("train"), get("val"), get("test")


class Sampler:
    def __init__(self, dataset, recs, arm, rng):
        self.rng, self.arm = rng, arm
        self.items = []
        for r in recs:
            p = L.load_prep(dataset, r)
            for a, b in L.runs(p["n2"], CROP):
                self.items.append((p, a, b - CROP))
        w = np.array([b - a + 1 for _, a, b in self.items], float)
        self.w = w / w.sum()
        self.n2_minutes = (w + CROP - 1).sum() / L.FS / 60

    def batch(self, n):
        import torch
        xs, ts, vs = [], [], []
        for i in self.rng.choice(len(self.items), size=n, p=self.w):
            p, a, b = self.items[i]
            s = int(self.rng.integers(a, b + 1))
            sl = slice(s, s + CROP)
            if self.arm == "inj":
                v = int(self.rng.integers(L.N_VARIANTS))
                x, t, val = p[f"y{v}"][sl], p[f"inj{v}"][sl], p[f"valid{v}"][sl]
            elif self.arm == "self":
                x, t, val = p["y"][sl], p["self_mask"][sl], np.ones(CROP, np.int8)
            else:
                x, t, val = p["y"][sl], p["real_mask"][sl], p["real_valid"][sl]
            down = self.rng.random() < 0.5                       # SUMOv2 augmentation, range 0.5
            x = x * (self.rng.uniform(0.5, 1) if down else self.rng.uniform(1, 2))
            xs.append(x), ts.append(t), vs.append(val)
        f = lambda a, dt: torch.from_numpy(np.stack(a).astype(dt))  # noqa: E731
        return f(xs, np.float32), f(ts, np.int64), f(vs, np.float32)


def validate(model, dataset, val):
    """-> dict with label-free (injected) and real-label validation: (thr, F1) each"""
    c_inj, c_real = 0, 0
    for p in val.values():
        ref, keep, anchor = L.inj_scope(p, 0)
        c_inj = c_inj + L.counts(L.predict_prob(model, p["y0"]), ref, keep, anchor)
        ref, keep, anchor = L.real_scope(p, dataset)
        c_real = c_real + L.counts(L.predict_prob(model, p["y"]), ref, keep, anchor)
    ti, fi = L.best_threshold(c_inj)
    tr, fr = L.best_threshold(c_real)
    return dict(inj_thr=ti, inj_f1=fi, inj_f1_05=L.f1_at(c_inj), real_thr=tr, real_f1=fr, real_f1_05=L.f1_at(c_real))


def main():
    a = parse()
    import torch
    torch.set_num_threads(a.threads)
    torch.manual_seed(1000 * a.seed + a.fold)
    rng = np.random.default_rng(1000 * a.seed + a.fold)
    train_recs, val_recs, test_recs = split(a.dataset, a.fold)
    assert not set(train_recs) & set(val_recs) and not set(train_recs + val_recs) & set(test_recs)
    did = det_id(a.arm, a.seed, a.tag)
    tag = f"{a.dataset}_fold{a.fold}"
    out = L.WORK / "models" / did
    out.mkdir(parents=True, exist_ok=True)

    model = L.make_model()
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    sampler = Sampler(a.dataset, train_recs, a.arm, rng)
    val = {r: L.load_prep(a.dataset, r) for r in val_recs}
    sel = "real" if a.arm == "real" else "inj"     # selection signal of this arm
    keeps = {sel: [-1.0, None, None, None]}       # best F1, state, epoch, threshold
    if a.arm == "inj":
        keeps["real"] = [-1.0, None, None, None]  # arm C'

    t0 = time.time()
    v = validate(model, a.dataset, val)
    hist = [dict(epoch=0, train_loss=np.nan, sec=0.0, **v)]
    print(did, tag, "epoch 0", v, flush=True)
    bad = 0
    for ep in range(1, a.max_epochs + 1):
        model.train()
        losses = []
        for _ in range(a.steps):
            x, t, valid = sampler.batch(a.batch)
            opt.zero_grad()
            loss = L.masked_gdl(model.dense_logits(x), t, valid)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        v = validate(model, a.dataset, val)
        hist.append(dict(epoch=ep, train_loss=float(np.mean(losses)), sec=time.time() - t0, **v))
        print(did, tag, f"epoch {ep} loss {np.mean(losses):.4f}",
              f"inj F1 {v['inj_f1']:.3f}@{v['inj_thr']} real F1 {v['real_f1']:.3f}@{v['real_thr']}",
              f"({time.time() - t0:.0f}s)", flush=True)
        improved = False
        for k, slot in keeps.items():
            if v[f"{k}_f1"] > slot[0] + 1e-9:
                keeps[k] = [v[f"{k}_f1"], {kk: vv.detach().clone() for kk, vv in model.state_dict().items()},
                            ep, v[f"{k}_thr"]]
                improved |= k == sel
        bad = 0 if improved else bad + 1
        if bad >= a.patience:
            break
    sec = time.time() - t0
    pd.DataFrame(hist).to_csv(out / f"{tag}_history.csv", index=False)

    outputs = {did: keeps[sel]}
    if a.arm == "inj":
        outputs[det_id("inj_rv", a.seed, a.tag)] = keeps["real"]
    for oid, (f1, state, ep, thr) in outputs.items():
        od = L.WORK / "models" / oid
        od.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": state}, od / f"{tag}.pt")
        model.load_state_dict(state)
        info = dict(id=oid, dataset=a.dataset, fold=a.fold, arm=a.arm, seed=a.seed, lr=a.lr, best_epoch=ep,
                    threshold=thr, val_f1=f1, epochs_run=len(hist) - 1, train_recs=train_recs, val_recs=val_recs,
                    test_recs=test_recs, train_n2_minutes=sampler.n2_minutes, train_seconds=round(sec, 1))
        if not a.no_test:
            pdir = L.WORK / "prob" / oid
            pdir.mkdir(parents=True, exist_ok=True)
            for r in test_recs:
                np.save(pdir / f"{a.dataset}__{r}.npy", L.predict_prob(model, L.load_prep(a.dataset, r)["y"]).astype(np.float32))
            if a.dataset == "nap":   # cross-dataset: NAP-trained fold models on DREAMS
                for r in L.recordings("dreams_spindles").rec:
                    np.save(pdir / f"dreams_spindles__{r}__fold{a.fold}.npy",
                            L.predict_prob(model, L.load_prep("dreams_spindles", r)["y"]).astype(np.float32))
        L.save_json(od / f"{tag}.json", info)
        print("DONE", oid, tag, f"best epoch {ep}, val F1 {f1:.3f} @ {thr}, {sec:.0f}s", flush=True)


if __name__ == "__main__":
    main()
