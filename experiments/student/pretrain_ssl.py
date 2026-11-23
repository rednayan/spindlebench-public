"""
Step 3 (Stage 1): SNR-invariant self-supervised pretraining of the EXACT same TrunkM / TrunkS
backbone (no architectural change to the trunk, for a clean apples-to-apples comparison).

Trains the trunk with VICReg (Bardes et al. 2022) across clean vs. faint-mixed (same-night quiet N2)
views + an auxiliary amplitude-invariant relative-spectrum denoising head (predicting the clean view's
relative band powers from the faint view). No expert or teacher labels enter the SSL objective
(teachers are used only to locate quiet N2 tiles for background mixing).

    python pretrain_ssl.py --seed 0 --track gpu --passes 15
    python pretrain_ssl.py --seed 0 --track cpu --subsets 10 --max-batches 20 --tag smoke

After pretraining, probe the frozen encoder with train.py:
    python train.py --arm student_ds --init-from ../runs/student/<ssl_run> --freeze-trunk --passes 5 --track gpu
    python train.py --arm student_faint_pu --init-from ../runs/student/<ssl_run> --freeze-trunk --passes 5 --track gpu
"""
import argparse
import json
import os
import platform
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import data
import model as M
import score_val_injection as SVI
import selection
import train
from spindlebench import provenance

BANDS = (
    (0.5, 4.0),    # delta
    (4.0, 8.0),    # theta
    (8.0, 10.5),   # alpha
    (10.5, 13.0),  # slow sigma
    (13.0, 16.0),  # fast sigma
    (16.0, 30.0),  # beta
)
SPEC_SMOOTH = 41   # 0.41 s moving average for local relative band power


class SSLWrapper(nn.Module):
    """Wraps an unchanged Student trunk with a 2-layer 1x1 conv VICReg projector and a 1x1 conv
    relative-spectrum head (both discarded after pretraining)."""

    def __init__(self, trunk_name="M", proj_dim=128):
        super().__init__()
        self.trunk = M.TrunkS() if trunk_name == "S" else M.TrunkM()
        c = self.trunk.out_channels
        self.proj = nn.Sequential(
            nn.Conv1d(c, proj_dim, 1, bias=False),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(),
            nn.Conv1d(proj_dim, proj_dim, 1),
        )
        self.spec_head = nn.Conv1d(c, len(BANDS), 1)

    def encode(self, x):
        _, feat, _, _ = self.trunk(x)
        return feat

    def forward(self, x):
        feat = self.encode(x)
        return feat, self.proj(feat), self.spec_head(feat)


@torch.no_grad()
def relative_band_powers(x, fs=100):
    """Amplitude-invariant local relative power across BANDS [N, len(BANDS), T], summing to 1."""
    T = x.shape[-1]
    X = torch.fft.rfft(x.float(), dim=-1)
    freqs = torch.fft.rfftfreq(T, d=1.0 / fs, device=x.device)
    pows = []
    for lo, hi in BANDS:
        fmask = (freqs >= lo) & (freqs <= hi)
        xb = torch.fft.irfft(X * fmask, n=T, dim=-1)
        p = F.avg_pool1d((xb ** 2).unsqueeze(1), SPEC_SMOOTH, stride=1, padding=SPEC_SMOOTH // 2)[:, 0]
        pows.append(p)
    P = torch.stack(pows, dim=1).clamp_min(1e-8)
    return P / P.sum(dim=1, keepdim=True)


def vicreg_loss(z1, z2, sim_w=25.0, std_w=25.0, cov_w=1.0, stride=5):
    """VICReg on temporal embeddings [N, D, T_core], subsampled by `stride`."""
    a = z1[:, :, ::stride].transpose(1, 2).reshape(-1, z1.shape[1])   # [M, D]
    b = z2[:, :, ::stride].transpose(1, 2).reshape(-1, z2.shape[1])
    inv = F.mse_loss(a, b)

    a_c = a - a.mean(dim=0, keepdim=True)
    b_c = b - b.mean(dim=0, keepdim=True)
    std_a = torch.sqrt(a_c.var(dim=0, unbiased=False) + 1e-4)
    std_b = torch.sqrt(b_c.var(dim=0, unbiased=False) + 1e-4)
    var = 0.5 * (F.relu(1.0 - std_a).mean() + F.relu(1.0 - std_b).mean())

    m, d = a_c.shape
    cov_a = (a_c.T @ a_c) / max(m - 1, 1)
    cov_b = (b_c.T @ b_c) / max(m - 1, 1)
    off = ~torch.eye(d, dtype=torch.bool, device=a.device)
    cov = 0.5 * ((cov_a[off] ** 2).sum() / d + (cov_b[off] ** 2).sum() / d)

    loss = sim_w * inv + std_w * var + cov_w * cov
    parts = dict(
        inv=inv.detach(),
        var=var.detach(),
        cov=cov.detach(),
        mean_std=0.5 * (std_a.mean() + std_b.mean()).detach(),
    )
    return loss, parts


def _auroc(pos_scores, neg_scores):
    """Exact rank-based AUROC without sklearn dependency."""
    s = np.r_[pos_scores, neg_scores]
    y = np.r_[np.ones(len(pos_scores)), np.zeros(len(neg_scores))]
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    n_pos, n_neg = len(pos_scores), len(neg_scores)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


@torch.no_grad()
def embedding_diagnostic(ssl_net, device, max_nights=8, variant=0):
    """Zero-label diagnostic on injected validation nights:
    Computes the centroid of 12 dB injected spindles and quiet N2 background in the trunk's 32-D
    feature space, then scores how well cos(z, mu_12dB) - cos(z, mu_bg) separates 12 dB, 6 dB, and
    3 dB spindles from quiet N2 background (AUROC, no classifier trained)."""
    ssl_net.eval()
    sp = data.splits()
    val = sp[sp.role == "val"].head(max_nights)
    rng = np.random.default_rng(2026)
    by_snr = {12.0: [], 6.0: [], 3.0: []}
    bg_vecs = []

    for ds, rec in zip(val.dataset, val.rec):
        ev, q = SVI.load_truth(variant, ds, rec)
        x = selection.injected_x(variant, ds, rec)
        n = len(x)
        # Encode in 10-min chunks (as validate.predict_night)
        feats = np.zeros((ssl_net.trunk.out_channels, n), dtype=np.float32)
        chunk, ctx = 60_000, 3_000
        for a in range(0, n, chunk):
            lo, hi = max(a - ctx, 0), min(a + chunk + ctx, n)
            xt = torch.from_numpy(np.ascontiguousarray(x[lo:hi], dtype=np.float32))[None].to(device)
            f = ssl_net.encode(xt)[0, :, a - lo:min(a + chunk, n) - lo].cpu().numpy()
            feats[:, a:min(a + chunk, n)] = f

        for row in ev.itertuples():
            if row.snr_db in by_snr:
                c = int(round((row.onset + row.duration / 2.0) * data.FS))
                if 25 <= c < n - 25:
                    by_snr[row.snr_db].append(feats[:, c - 20:c + 21].mean(axis=1))

        qmask = np.unpackbits(q)[:n].astype(bool)
        for row in ev.itertuples():
            i0 = max(int((row.onset - 1.0) * data.FS), 0)
            i1 = min(int((row.onset + row.duration + 1.0) * data.FS), n)
            qmask[i0:i1] = False
        qidx = np.flatnonzero(qmask)
        if len(qidx) > 200:
            pick = rng.choice(qidx, size=min(400, len(qidx)), replace=False)
            for c in pick:
                if 25 <= c < n - 25:
                    bg_vecs.append(feats[:, c - 20:c + 21].mean(axis=1))

    if not by_snr[12.0] or not bg_vecs:
        return {}
    def norm(v):
        v = np.asarray(v, dtype=np.float64)
        return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-8)

    v12 = norm(by_snr[12.0])
    vbg = norm(bg_vecs)
    mu12 = norm(v12.mean(axis=0))
    mubg = norm(vbg.mean(axis=0))
    bg_sc = vbg @ mu12 - vbg @ mubg
    out = {}
    for s in (12.0, 6.0, 3.0):
        if by_snr[s]:
            vs = norm(by_snr[s])
            sc = vs @ mu12 - vs @ mubg
            out[f"diag_auroc_{int(s)}dB"] = _auroc(sc, bg_sc)
    return out


def make_views(xb_n2, qx, xb_wr, rng, alpha_lo, alpha_hi):
    """Build paired views (x1 clean, x2 faint/perturbed):
      - N2 crops: x1 = sign-flipped clean; x2 = quiet-N2 mixed + amplitude-scaled + tiny noise
      - Wake/REM crops: x1 = sign-flipped clean; x2 = amplitude-scaled + tiny noise"""
    sign_n2 = np.where(rng.random(len(xb_n2)) < 0.5, -1.0, 1.0).astype(np.float32)[:, None]
    x1_n2 = xb_n2 * sign_n2
    x2_n2, _ = data.mix_faint(x1_n2, qx, rng, prob=1.0, alpha_lo=alpha_lo, alpha_hi=alpha_hi)
    # Amplitude scale x2 (preserving sign so phase alignment with x1 matches sample-for-sample)
    down = rng.random(len(x2_n2)) < 0.5
    f = np.where(down, rng.uniform(0.5, 1.0, len(x2_n2)), rng.uniform(1.0, 2.0, len(x2_n2))).astype(np.float32)[:, None]
    x2_n2 = x2_n2 * f + 0.02 * x2_n2.std(1, keepdims=True) * rng.standard_normal(x2_n2.shape).astype(np.float32)

    if xb_wr is not None and len(xb_wr):
        sign_wr = np.where(rng.random(len(xb_wr)) < 0.5, -1.0, 1.0).astype(np.float32)[:, None]
        x1_wr = xb_wr * sign_wr
        down_w = rng.random(len(x1_wr)) < 0.5
        fw = np.where(down_w, rng.uniform(0.5, 1.0, len(x1_wr)), rng.uniform(1.0, 2.0, len(x1_wr))).astype(np.float32)[:, None]
        x2_wr = x1_wr * fw + 0.02 * x1_wr.std(1, keepdims=True) * rng.standard_normal(x1_wr.shape).astype(np.float32)
        return np.concatenate([x1_n2, x1_wr], axis=0), np.concatenate([x2_n2, x2_wr], axis=0)
    return x1_n2, x2_n2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--track", default="gpu", choices=list(train.TRACKS))
    ap.add_argument("--device", default=None, choices=["cpu", "cuda"])
    ap.add_argument("--subsets", default=None)
    ap.add_argument("--passes", type=int, default=15)
    ap.add_argument("--max-batches", type=int, default=None)
    ap.add_argument("--alpha-lo", type=float, default=0.45)
    ap.add_argument("--alpha-hi", type=float, default=0.75)
    ap.add_argument("--wake-rem-frac", type=float, default=0.25)
    ap.add_argument("--spec-weight", type=float, default=0.25,
                    help="weight of relative-spectrum denoising loss (clean spectrum predicted from faint view)")
    ap.add_argument("--no-diag", action="store_true", help="skip per-pass injection embedding diagnostic")
    ap.add_argument("--diag-nights", type=int, default=8)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    provenance.track(__file__, protocols=["artefacts.v1"])

    tr = dict(train.TRACKS[a.track])
    subsets = tuple(a.subsets.split(",")) if a.subsets else tr["subsets"]
    device = a.device or tr["device"]
    if device == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("--device cuda: no CUDA GPU visible")
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    torch.set_num_threads(a.threads)
    torch.use_deterministic_algorithms(True)
    rng = np.random.default_rng([data.SEED + 77, a.seed])

    sp = data.splits()
    trn = sp[(sp.role == "train") & sp.subset.isin(subsets)]
    for ds in trn.dataset:
        data.guard(ds)
    recs = list(zip(trn.dataset, trn.rec))
    sampler = data.CropSampler(recs)
    quiet_sampler = data.QuietN2TileSampler(sampler)
    wr_sampler = data.WakeRemSampler(recs) if a.wake_rem_frac > 0 else None

    ssl_net = SSLWrapper(trunk_name=tr["trunk"]).to(device)
    opt = torch.optim.AdamW(ssl_net.parameters(), lr=tr["lr"], weight_decay=1e-4)
    per_pass = sampler.crops_per_pass() // tr["batch"]
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.passes * per_pass, eta_min=1e-5)

    name = data.run_name("ssl_vicreg", a.tag, a.seed, a.track)
    out = data.RUNS / name
    out.mkdir(parents=True, exist_ok=True)
    cfg = dict(arm="ssl_vicreg", seed=a.seed, track=a.track, args=vars(a), track_settings=tr,
               subsets=subsets, passes=a.passes, batches_per_pass=per_pass, nights=len(trn),
               trunk_params=M.count_parameters(ssl_net.trunk), total_params=M.count_parameters(ssl_net),
               device=device, host=platform.node(), git=train.git_state())
    (out / "config.json").write_text(json.dumps(cfg, indent=1, default=str))
    print(json.dumps({k: cfg[k] for k in ("arm", "seed", "nights", "batches_per_pass", "trunk_params")}), flush=True)

    c0, c1 = data.CORE_LO, data.CORE_LO + data.CORE
    metrics, done = [], 0
    for k in range(1, a.passes + 1):
        ssl_net.train()
        t0, run, b0 = time.time(), {}, done
        for _ in range(per_pass):
            n_wr = int(round(tr["batch"] * a.wake_rem_frac)) if wr_sampler is not None else 0
            n_n2 = tr["batch"] - n_wr
            xb_n2, _, nights_n2 = sampler.batch_with_nights(rng, n_n2)
            qx = quiet_sampler.sample_for_nights(nights_n2, rng)
            xb_wr = wr_sampler.batch(rng, n_wr) if n_wr > 0 else None
            x1_np, x2_np = make_views(xb_n2, qx, xb_wr, rng, a.alpha_lo, a.alpha_hi)

            x1 = torch.from_numpy(x1_np).to(device, non_blocking=True)
            x2 = torch.from_numpy(x2_np).to(device, non_blocking=True)
            _, z1, _ = ssl_net(x1)
            _, z2, sp2 = ssl_net(x2)

            v_loss, parts = vicreg_loss(z1[:, :, c0:c1], z2[:, :, c0:c1])
            loss = v_loss
            if a.spec_weight > 0:
                tgt_sp = relative_band_powers(x1, fs=data.FS)[:, :, c0:c1]
                pred_sp = F.softmax(sp2[:, :, c0:c1], dim=1)
                sp_loss = F.mse_loss(pred_sp, tgt_sp)
                parts["spec"] = sp_loss.detach()
                loss = loss + a.spec_weight * sp_loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()

            for key, v in {"loss": loss.detach(), **parts}.items():
                run[key] = run.get(key, 0.0) + float(v)
            done += 1
            if a.max_batches and done >= a.max_batches:
                break

        n_b = max(done - b0, 1)
        row = {"pass": k, "batches": done, "pass_s": time.time() - t0,
               **{f"train_{key}": v / n_b for key, v in run.items()}}
        # Save in exact Student format (trunk.* keys) so train.py --init-from loads it directly!
        sd = {f"trunk.{key}": v.cpu() for key, v in ssl_net.trunk.state_dict().items()}
        torch.save(sd, out / f"pass{k}.pt")

        if not a.no_diag:
            row.update(embedding_diagnostic(ssl_net, device, max_nights=a.diag_nights))
        metrics.append(row)
        pd.DataFrame(metrics).to_csv(out / "metrics.csv", index=False)
        print(json.dumps({key: (round(v, 4) if isinstance(v, float) else v) for key, v in row.items()}), flush=True)
        if a.max_batches and done >= a.max_batches:
            break

    # Pick pass with highest 6 dB zero-shot embedding AUROC (or last pass if --no-diag)
    df = pd.DataFrame(metrics)
    best = int(df.loc[df["diag_auroc_6dB"].idxmax(), "pass"]) if "diag_auroc_6dB" in df.columns else int(df["pass"].iloc[-1])
    (out / "done.json").write_text(json.dumps(dict(best_pass=best, passes_run=len(metrics))))
    print(f"Done. Selected pass: {best} -> {out}", flush=True)


if __name__ == "__main__":
    main()
