"""
One training run of the student (plan sections 4.5-4.7, 7).

    uv run python train.py --arm student_rules4 --seed 0 [--track cpu|gpu] [--device cpu|cuda]
                           [--lam 0.01] [--max-batches N] [--passes P] [--no-val] [--tag x]   (from experiments/student)

Track = the settings (trunk, data, batch, optimiser); device = where they run. Each track has a default
device (cpu -> cpu, gpu -> cuda) and --device overrides it per run; the device is recorded in config.json.
On CUDA the run is made deterministic (CUBLAS_WORKSPACE_CONFIG, deterministic algorithms, no cuDNN
benchmark), but a GPU run is not bit-identical to a CPU run of the same seed.

-> data.RUNS/<arm>[_<tag>][_<track>]_s<seed>/   (data.RUNS = SPINDLEBENCH_RUNS, default <repo>/../runs/student)
     config.json     every setting, seed, git commit and dirty flag, data counts, parameter table
     metrics.csv     one row per pass: losses, the six agreement F1s and plausibility on validation nights
     pass<k>.pt      model state after pass k
     timing.json     seconds per batch, per pass and for validation
     done.json       the Borda-selected pass; or best_pass null and `collapsed` when the noise branch swapped
                     the labels (amendment 8: the run is stopped, it can never be selected)
Arms (the switches of one code path; section 7 of the plan):
    student_soft    soft target k/4, CE, no noise branch, no stream B, no head 2
    student_mv      hard majority vote, CE
    student_ds      Dawid-Skene posterior (soft), CE
    student_<t>     one teacher's mask, CE (t in a7, luna, ucsd, yasa)
    student_rules4  noisy-annotator loss over the four teachers with the per-sample noise branch
                    (stream B + trunk bottleneck), trace weight lam, head 2 (boundary BCE + segment CDF)
  ablation flags: --noise global|none, --no-stream-b, --stream-b-into-trunk (trunk M), --no-pcen,
                  --no-trunk-features, --no-boundary, --gdl (generalised Dice on mv), --bigru (trunk M)
  rules4-fix pilot: --anchor ds [--anchor-delta D] (noise tables = Dawid-Skene matrices + a correction
                  bounded by +-D logits), --init-from <run folder> (warm start of the trunk)
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import data
import label_models
import losses
import model as M
import validate
from spindlebench import paths, provenance

TEACHER_ARMS = {"student_a7": "a7", "student_luna": "luna_spindles", "student_ucsd": "wonambi_ucsd",
                "student_yasa": "yasa_default"}
TRACKS = {
    "cpu": dict(device="cpu", trunk="S", subsets=("10", "30"), batch=12, passes=20, patience=5, opt="adam", lr=1e-3, wd=0.0,
                seeds=(0, 1, 2)),
    "gpu": dict(device="cuda", trunk="M", subsets=("10", "30", "100"), batch=32, passes=30, patience=8, opt="adamw", lr=1e-3,
                wd=1e-4, seeds=(0, 1, 2, 3, 4)),
}
WARMUP_PASSES = 2          # [ours] noise branch frozen at its diagonal initialisation; segment loss off
LAMBDA_B, LAMBDA_S = 1.0, 1.0   # [ours] weights of the boundary BCE and the segment CDF loss
MIN_PASS = 3
SWAP = 1.0                 # amendment 8: stop when the probability-weighted mean CM over teachers has
                           # diag0 + diag1 <= 1, i.e. the teachers are modelled as no better than chance
                           # (Youden's J <= 0): the label-swapped side of the likelihood
DOMINANCE = 0.5            # diagonal dominance of the mean CM (Tanno et al. 2019, Theorem 2): reported per
                           # pass, not a stop rule (true teacher recall is < 0.5 below 9 dB, section 13.3)
HEARTBEAT = 50             # batches between progress records (logging only; no effect on training)


def heartbeat(out, **state):
    """progress record for the dashboard: heartbeat.json (overwritten atomically)"""
    state["time"] = time.time()
    tmp = out / "heartbeat.json.tmp"
    tmp.write_text(json.dumps(state, default=float))
    tmp.replace(out / "heartbeat.json")


def log_batches(out, row):
    """one row per HEARTBEAT batches: pass, batch, mean losses over the window, seconds per batch"""
    f = out / "batches.csv"
    new = not f.exists()
    cols = ["pass", "batch", "global_batch", "loss", "ce", "trace", "boundary", "segment", "diag0", "diag1",
            "s_per_batch"]
    with open(f, "a", encoding="utf-8") as fh:
        if new:
            fh.write(",".join(cols) + "\n")
        vals = [row.get(c, "") for c in cols]
        fh.write(",".join(f"{v:.6g}" if isinstance(v, float) else str(v) for v in vals) + "\n")


def collapsed(row, warm):
    """amendment 8: after warm-up, the pass means of the probability-weighted diagonal of the mean CM over
    teachers; a label swap (diag0 + diag1 <= SWAP, teachers worse than chance) -> {diag0, diag1}, else None.
    Adds `dominant` (both > DOMINANCE) to the row as a reported diagnostic."""
    if "train_diag0" not in row:
        return None
    d = dict(diag0=float(row["train_diag0"]), diag1=float(row["train_diag1"]))
    row["dominant"] = min(d.values()) > DOMINANCE
    if warm:
        return None
    return d if sum(d.values()) <= SWAP else None


def git_state():
    def g(*a):
        try:
            return subprocess.run(["git", "-C", str(paths.ROOT), *a], capture_output=True, text=True).stdout.strip()
        except OSError:
            return ""
    return dict(commit=g("rev-parse", "HEAD"), dirty=bool(g("status", "--porcelain", "--untracked-files=no")))


def build(arm, a, prior):
    rules4 = arm.startswith("student_rules4")
    kw = dict(trunk=a.trunk, bigru=a.bigru, prior=prior)
    if rules4:
        noise = None if a.noise == "none" else a.noise
        kw.update(noise=noise, stream_b=not a.no_stream_b, pcen=not a.no_pcen,
                  noise_trunk_features=not a.no_trunk_features, stream_b_into_trunk=a.stream_b_into_trunk,
                  boundary=not a.no_boundary)
        if getattr(a, "anchor", "none") == "ds":   # (runs from before the option have no `anchor`)
            kw.update(noise_anchor=label_models.ds_confusion(), anchor_delta=a.anchor_delta)
    else:
        kw.update(noise=None, stream_b=False, boundary=False)
    return M.Student(**kw)


def targets(arm, pat, Y, tables):
    """per-sample target for the plain and per-teacher arms"""
    if arm == "student_soft":
        return tables["soft"][pat]
    if arm == "student_mv":
        return tables["mv"][pat]
    if arm in ("student_ds", "student_faint_pu"):
        return tables["ds"][pat]
    if arm in TEACHER_ARMS:
        return Y[:, data.TEACHERS.index(TEACHER_ARMS[arm])].astype(np.float32)
    raise ValueError(arm)


def step_loss_faint_pu(net, a, xb_n2, xb_orig, cb_n2, xb_wr, tables, device="cpu", mc_n2=None):
    """Loss for student_faint_pu: N2 crops (with optional faint-mixing, narrow PU ignore mask on
    unmixed xb_orig, and optional C4/O2 multi-channel witness) + clean Wake/REM crops (y = 0)."""
    Y, valid, pat = data.decode_code(cb_n2)
    m_n2 = valid.copy()
    m_n2[:, :data.CORE_LO] = False
    m_n2[:, data.CORE_LO + data.CORE:] = False

    def t(arr):
        return torch.from_numpy(arr).to(device, non_blocking=True)

    mt_n2 = t(m_n2.astype(np.float32))
    s_n2 = t(np.asarray(tables["ds"][pat], np.float32))
    Yt = t(Y)
    keep = torch.ones_like(mt_n2, dtype=torch.bool)
    parts = {}

    if not a.no_pu_mask:
        keep = losses.pu_keep_mask(t(xb_orig), Yt, fs=data.FS, sigma_mult=a.pu_sigma_mult)

    if mc_n2 is not None:
        mct = t(mc_n2)
        n_votes = Yt.sum(dim=1)
        # Recovered real faint spindle on C3: C4 clear >= 9 dB (bit 0) & C3 faint >= 3 dB (bit 3) & not O2 alpha (bit 2)
        recov = (n_votes == 0) & ((mct & 1) > 0) & ((mct & 8) > 0) & ((mct & 4) == 0)
        # Suspected teacher FP on C3: O2 alpha burst (bit 2) or unilateral (no C4 support, bit 1 == 0)
        veto = (n_votes >= 2) & (((mct & 4) > 0) | ((mct & 2) == 0))
        s_n2 = torch.where(recov, torch.full_like(s_n2, 0.90), s_n2)
        keep = (keep | recov) & (~veto)
        core_valid = mt_n2 > 0.5
        denom = max(int(core_valid.sum().item()), 1)
        parts["mc_recov"] = float((recov & core_valid).sum().item() / denom)
        parts["mc_veto"] = float((veto & core_valid).sum().item() / denom)

    core_valid = mt_n2 > 0.5
    parts["pu_masked"] = float(((~keep) & core_valid).sum().item() / max(int(core_valid.sum().item()), 1))
    mt_n2 = mt_n2 * keep.float()

    if xb_wr is not None and len(xb_wr):
        x = t(np.concatenate([xb_n2, xb_wr], axis=0))
        m_wr = np.zeros((len(xb_wr), xb_wr.shape[1]), dtype=np.float32)
        m_wr[:, data.CORE_LO:data.CORE_LO + data.CORE] = 1.0
        mt = torch.cat([mt_n2, t(m_wr)], dim=0)
        s = torch.cat([s_n2, torch.zeros((len(xb_wr), xb_wr.shape[1]), dtype=torch.float32, device=device)], dim=0)
    else:
        x, mt, s = t(xb_n2), mt_n2, s_n2

    out = net(x, need_noise=False)
    loss = losses.soft_ce(out["logits"], s, mt)
    return loss, parts


def step_loss(net, arm, a, xb, cb, tables, warm, device="cpu"):
    Y, valid, pat = data.decode_code(cb)
    m = valid.copy()
    m[:, :data.CORE_LO] = False
    m[:, data.CORE_LO + data.CORE:] = False

    def t(arr):                                   # numpy batch -> tensor on the run's device
        return torch.from_numpy(arr).to(device, non_blocking=True)

    x = t(xb)
    mt = t(m.astype(np.float32))
    out = net(x, need_noise=arm.startswith("student_rules4"))
    parts = {}
    if arm.startswith("student_rules4"):
        if "A" in out:
            loss, parts = losses.noisy_annotator(out["logits"], out["A"], t(Y), mt, a.lam)
        else:   # noise branch off: the rules4 arm degenerates to soft-target CE (= student_soft)
            loss = losses.soft_ce(out["logits"], t(tables["soft"][pat].astype(np.float32)), mt)
        if a.gdl:
            loss = losses.gdl(out["logits"], t(tables["mv"][pat].astype(np.int64)), mt)
        if "b_logit" in out:
            mv = tables["mv"][pat].astype(np.int8)
            ev = losses.consensus_events(mv, valid)
            bt = t(losses.boundary_targets(ev, *mv.shape))
            lb = losses.boundary_bce(out["b_logit"], bt, mt)
            parts["boundary"] = lb.detach()
            loss = loss + LAMBDA_B * lb
            if not warm:
                ls = losses.segment_cdf(out["p"], ev)
                parts["segment"] = ls.detach()
                loss = loss + LAMBDA_S * ls
    elif a.gdl:
        loss = losses.gdl(out["logits"], t(tables["mv"][pat].astype(np.int64)), mt)
    else:
        s = targets(arm, pat, Y, tables)
        loss = losses.soft_ce(out["logits"], t(np.asarray(s, np.float32)), mt)
    return loss, parts


def set_noise_trainable(net, on):
    for mod in (net.noise, net.stream_b if not net.stream_b_into_trunk else None):
        if mod is not None:
            for prm in mod.parameters():
                prm.requires_grad_(on)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--track", default="cpu", choices=list(TRACKS))
    ap.add_argument("--device", default=None, choices=["cpu", "cuda"], help="default: the track's device")
    ap.add_argument("--lam", type=float, default=0.01, help="trace weight (Tanno et al. 2019; amendment 8)")
    ap.add_argument("--noise", default="local", choices=["local", "global", "none"])
    ap.add_argument("--no-stream-b", action="store_true")
    ap.add_argument("--stream-b-into-trunk", action="store_true")
    ap.add_argument("--no-pcen", action="store_true")
    ap.add_argument("--no-trunk-features", action="store_true")
    ap.add_argument("--no-boundary", action="store_true")
    ap.add_argument("--gdl", action="store_true")
    ap.add_argument("--bigru", action="store_true")
    ap.add_argument("--subsets", default=None, help="override the track's data subsets, e.g. 10 or 10,30,100")
    ap.add_argument("--passes", type=int, default=None)
    ap.add_argument("--max-batches", type=int, default=None, help="stop after N batches (timing runs)")
    ap.add_argument("--stop-pass", type=int, default=None,
                    help="stop after pass N, keeping the schedule of --passes (pilots: passes 1..N as in a full run)")
    ap.add_argument("--no-val", action="store_true")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--tag", default="")
    ap.add_argument("--anchor", default="none", choices=["none", "ds"],
                    help="rules4 local noise branch: start each teacher's table at its Dawid-Skene matrix and "
                         "bound the per-sample correction (rules4-fix pilot)")
    ap.add_argument("--anchor-delta", type=float, default=1.0, help="bound of the correction, in logits (0: fixed)")
    ap.add_argument("--init-from", default=None,
                    help="a finished run folder: load its trunk weights at its selected pass (warm start)")
    ap.add_argument("--freeze-trunk", action="store_true",
                    help="freeze trunk backbone (train only trunk.dense linear head; for SSL probing)")
    ap.add_argument("--faint-mix", type=float, default=0.5,
                    help="student_faint_pu: fraction of N2 crops mixed with same-night quiet N2")
    ap.add_argument("--faint-alpha-lo", type=float, default=0.45, help="student_faint_pu: min mixing alpha")
    ap.add_argument("--faint-alpha-hi", type=float, default=0.75, help="student_faint_pu: max mixing alpha")
    ap.add_argument("--pu-sigma-mult", type=float, default=2.0,
                    help="student_faint_pu: sigma-envelope multiplier over crop median for PU ignore mask")
    ap.add_argument("--no-pu-mask", action="store_true", help="student_faint_pu: disable PU ignore mask")
    ap.add_argument("--wake-rem-frac", type=float, default=0.25,
                    help="student_faint_pu: fraction of batch from clean Wake/REM (y = 0)")
    ap.add_argument("--mc-witness", action="store_true",
                    help="student_faint_pu: use precomputed C4-M1 contralateral faint-spindle recovery & O2-M1 alpha veto")
    ap.add_argument("--no-borda-stop", action="store_true",
                    help="disable Borda vote-agreement early stopping (always on for student_faint_pu)")
    a = ap.parse_args()
    provenance.track(__file__, protocols=["artefacts.v1"])
    tr = dict(TRACKS[a.track])
    a.trunk = tr["trunk"]
    subsets = tuple(a.subsets.split(",")) if a.subsets else tr["subsets"]
    passes = a.passes or tr["passes"]
    if (a.arm not in ("student_soft", "student_mv", "student_ds", "student_faint_pu", *TEACHER_ARMS)
            and not a.arm.startswith("student_rules4")):
        raise SystemExit(f"unknown arm {a.arm}")

    device = a.device or tr["device"]
    if device == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("--device cuda: no CUDA GPU visible to torch (CPU build installed? see README, GPU)")
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # deterministic cuBLAS (PyTorch notes)
        torch.backends.cudnn.benchmark = False
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    torch.set_num_threads(a.threads)
    torch.use_deterministic_algorithms(True)
    rng = np.random.default_rng([data.SEED, a.seed])

    sp = data.splits()
    trn = sp[(sp.role == "train") & sp.subset.isin(subsets)]
    for ds in trn.dataset:
        data.guard(ds)
    recs = list(zip(trn.dataset, trn.rec))
    faint_pu = a.arm == "student_faint_pu"
    sampler = data.CropSampler(recs, use_witness=(faint_pu and a.mc_witness))
    quiet_sampler = data.QuietN2TileSampler(sampler) if (faint_pu and a.faint_mix > 0) else None
    wr_sampler = data.WakeRemSampler(recs) if (faint_pu and a.wake_rem_frac > 0) else None
    tables = label_models.load_targets()
    pat_counts = pd.read_csv(data.STUDY / "teachers" / "patterns.csv")
    pc = pat_counts[pat_counts.rec.isin(trn.rec)].groupby("pattern")["count"].sum().reindex(range(16), fill_value=0)
    prior = float((pc.to_numpy() * tables["mv"]).sum() / pc.sum())     # pi: mv spindle fraction (SEED init)

    net = build(a.arm, a, prior).to(device)
    init = None
    if a.init_from:   # warm start: the trunk (detector) of a finished run at its selected pass
        src = Path(a.init_from)
        bp = json.loads((src / "done.json").read_text())["best_pass"]
        sd = torch.load(src / f"pass{bp}.pt", map_location=device)
        trunk = {k[len("trunk."):]: v for k, v in sd.items() if k.startswith("trunk.")}
        net.trunk.load_state_dict(trunk)   # strict: same trunk and track
        init = dict(run=src.name, best_pass=bp, tensors=len(trunk))
    if a.freeze_trunk:
        for prm in net.trunk.parameters():
            prm.requires_grad_(False)
        for prm in net.trunk.dense.parameters():
            prm.requires_grad_(True)
        dense = net.trunk.dense[0] if isinstance(net.trunk.dense, torch.nn.Sequential) else net.trunk.dense
        torch.nn.init.kaiming_uniform_(dense.weight, a=5 ** 0.5)
        net.init_priors(prior)
    params = M.parameter_table(net)
    trainable = [p for p in net.parameters() if p.requires_grad]
    opt = (torch.optim.Adam(trainable, lr=tr["lr"]) if tr["opt"] == "adam"
           else torch.optim.AdamW(trainable, lr=tr["lr"], weight_decay=tr["wd"]))
    per_pass = sampler.crops_per_pass() // tr["batch"]
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=passes * per_pass, eta_min=1e-5)
             if tr["opt"] == "adamw" else None)

    name = data.run_name(a.arm, a.tag, a.seed, a.track)
    out = data.RUNS / name
    out.mkdir(parents=True, exist_ok=True)
    cfg = dict(arm=a.arm, seed=a.seed, track=a.track, args=vars(a), track_settings=tr, subsets=subsets,
               passes=passes, batches_per_pass=per_pass, nights=len(trn), valid_h=sampler.valid_s / 3600,
               sampled_h=sampler.sampled_s / 3600, short_runs=sampler.short_runs, short_h=sampler.short_s / 3600,
               prior=prior, warmup_passes=WARMUP_PASSES, lambda_b=LAMBDA_B, lambda_s=LAMBDA_S,
               boundary_half=losses.BOUNDARY_HALF, segment_margin=losses.SEGMENT_MARGIN, parameters=params,
               init_from=init,
               noise_anchor=label_models.ds_confusion().tolist() if a.anchor == "ds" else None,
               device=device, gpu=torch.cuda.get_device_name(0) if device == "cuda" else None,
               host=platform.node(), cpu=platform.processor(), torch=torch.__version__, python=sys.version.split()[0], git=git_state(),
               command=" ".join(sys.argv))
    (out / "config.json").write_text(json.dumps(cfg, indent=1, default=str))
    print(json.dumps({k: cfg[k] for k in ("arm", "seed", "nights", "batches_per_pass", "parameters")}, default=str),
          flush=True)

    val = None if a.no_val else validate.ValSet()
    rules4 = a.arm.startswith("student_rules4")
    metrics, timing, done = [], dict(batch_s=[], pass_s=[], val_s=[]), 0
    for k in range(1, passes + 1):
        warm = rules4 and k <= WARMUP_PASSES
        if rules4:
            set_noise_trainable(net, not warm)
        net.train()
        if a.freeze_trunk:
            net.trunk.eval()
            net.trunk.dense.train()
        t0, run, b0 = time.time(), {}, done
        win, win_t = {}, time.time()
        heartbeat(out, phase="train", pass_=k, batch=0, batches_per_pass=per_pass, passes=passes, warm=warm)
        for _ in range(per_pass):
            tb = time.time()
            if faint_pu:
                n_wr = int(round(tr["batch"] * a.wake_rem_frac)) if wr_sampler is not None else 0
                n_n2 = tr["batch"] - n_wr
                if a.mc_witness:
                    xb_n2, cb_n2, nights_n2, mc_n2 = sampler.batch_with_nights(rng, n_n2, return_mc=True)
                else:
                    xb_n2, cb_n2, nights_n2 = sampler.batch_with_nights(rng, n_n2)
                    mc_n2 = None
                xb_orig = xb_n2
                if quiet_sampler is not None:
                    qx = quiet_sampler.sample_for_nights(nights_n2, rng)
                    xb_n2, _ = data.mix_faint(xb_n2, qx, rng, prob=a.faint_mix,
                                              alpha_lo=a.faint_alpha_lo, alpha_hi=a.faint_alpha_hi)
                xb_n2 = data.augment(xb_n2, rng)
                xb_wr = data.augment(wr_sampler.batch(rng, n_wr), rng) if n_wr > 0 else None
                loss, parts = step_loss_faint_pu(net, a, xb_n2, xb_orig, cb_n2, xb_wr, tables, device, mc_n2=mc_n2)
            else:
                xb, cb = sampler.batch(rng, tr["batch"])
                xb = data.augment(xb, rng)
                loss, parts = step_loss(net, a.arm, a, xb, cb, tables, warm, device)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if sched is not None:
                sched.step()
            for key, v in {"loss": loss.detach(), **parts}.items():
                run[key] = run.get(key, 0.0) + float(v)
            timing["batch_s"].append(time.time() - tb)
            done += 1
            for key, v in {"loss": loss.detach(), **parts}.items():
                win[key] = win.get(key, 0.0) + float(v)
            if (done - b0) % HEARTBEAT == 0:
                sb = (time.time() - win_t) / HEARTBEAT
                log_batches(out, {"pass": k, "batch": done - b0, "global_batch": done,
                                  **{key: v / HEARTBEAT for key, v in win.items()}, "s_per_batch": sb})
                heartbeat(out, phase="train", pass_=k, batch=done - b0, batches_per_pass=per_pass, passes=passes,
                          warm=warm, s_per_batch=sb, loss=win["loss"] / HEARTBEAT)
                win, win_t = {}, time.time()
            if a.max_batches and done >= a.max_batches:
                break
        n_b = max(done - b0, 1)
        row = {"pass": k, "batches": done, **{f"train_{key}": v / n_b for key, v in run.items()}}
        timing["pass_s"].append(time.time() - t0)
        torch.save({key: v.cpu() for key, v in net.state_dict().items()}, out / f"pass{k}.pt")   # loads anywhere
        if val is not None:
            tv = time.time()
            heartbeat(out, phase="validate", pass_=k, batch=per_pass, batches_per_pass=per_pass, passes=passes,
                      warm=warm)
            row.update(val.evaluate(net))
            timing["val_s"].append(time.time() - tv)
        metrics.append(row)
        (out / "timing.json").write_text(json.dumps(dict(
            mean_batch_s=float(np.mean(timing["batch_s"])), median_batch_s=float(np.median(timing["batch_s"])),
            batches=len(timing["batch_s"]), pass_s=timing["pass_s"], val_s=timing["val_s"]), indent=1))
        lost = collapsed(row, warm)
        pd.DataFrame(metrics).to_csv(out / "metrics.csv", index=False)
        print(json.dumps({key: (round(v, 4) if isinstance(v, float) else v) for key, v in row.items()}), flush=True)
        if lost:
            print(f"stopped: label swap (teachers modelled as worse than chance) at pass {k} {lost}", flush=True)
            (out / "done.json").write_text(json.dumps(dict(best_pass=None, passes_run=len(metrics),
                                                           collapsed=dict(pass_=k, **lost, rule=f"diag0 + diag1 <= {SWAP}"))))
            sys.exit(3)
        if (a.max_batches and done >= a.max_batches) or (a.stop_pass and k >= a.stop_pass):
            break
        if val is not None and not (a.no_borda_stop or faint_pu or a.freeze_trunk):
            best, _ = validate.borda(metrics, min_pass=MIN_PASS)
            if best is not None and k - best >= tr["patience"]:
                print(f"early stop: best pass {best}", flush=True)
                break
    best, score = validate.borda(metrics, min_pass=MIN_PASS) if val is not None else (None, None)
    (out / "done.json").write_text(json.dumps(dict(best_pass=best, passes_run=len(metrics),
                                                   borda=None if score is None else score.to_dict())))


if __name__ == "__main__":
    main()
