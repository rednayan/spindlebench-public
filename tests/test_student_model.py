"""experiments/student, step 2: model, losses, decoders. Needs torch: run in the student environment,
    uv run --project experiments/student pytest tests/test_student_model.py
(skipped in the main environment)."""
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

from spindlebench import paths  # noqa: E402

sys.path.insert(0, str(paths.ROOT / "experiments" / "student"))
import decode  # noqa: E402
import losses  # noqa: E402
import model as M  # noqa: E402

FIXTURE = paths.RESULTS / "student" / "work" / "fixtures" / "sumov2_reference.npz"


def test_trunk_s_is_sumov2():
    if not FIXTURE.exists():
        pytest.skip("no SUMOv2 reference (exported from detectors/sumo/.venv-sumov2)")
    z = np.load(FIXTURE)
    t = M.TrunkS()
    t.load_state_dict({k[4:]: torch.from_numpy(z[k]) for k in z.files if k.startswith("sd__")})
    t.eval()
    with torch.no_grad():
        logits = t(torch.from_numpy(z["x"]))[0]
    assert np.abs(logits.numpy() - z["dense"]).max() < 1e-5
    assert np.abs(t.postprocess(logits).numpy() - z["out"]).max() < 1e-5


def test_parameter_counts():
    m = M.Student()
    tab = M.parameter_table(m)
    assert tab["stream_b"] == 56
    assert tab["trunk"] == 70338          # SUMOv2's U-Net
    assert tab["head2"] == 17
    assert M.count_parameters(M.Student(noise=None, stream_b=False, boundary=False)) == 70338


def test_confusion_matrices_row_stochastic_and_diagonal_init():
    m = M.Student()
    A = m(torch.randn(2, 4000))["A"]
    assert torch.allclose(A.sum(3), torch.ones_like(A.sum(3)), atol=1e-6)
    assert torch.allclose(A[:, :, 0, 0], torch.full_like(A[:, :, 0, 0], M.DIAG_INIT), atol=1e-6)
    g = M.Student(noise="global")(torch.randn(1, 4000))["A"]
    assert torch.allclose(g.sum(3), torch.ones_like(g.sum(3)), atol=1e-6)


def test_identity_noise_equals_soft_target_ce():
    torch.manual_seed(0)
    logits = torch.randn(3, 2, 500)
    Y = (torch.rand(3, 4, 500) < 0.3).long()
    m = (torch.rand(3, 500) < 0.8).float()
    eye = torch.eye(2)[None, None, :, :, None].expand(3, 4, 2, 2, 500)
    loss, parts = losses.noisy_annotator(logits, eye, Y, m, lam=0.0)
    soft = losses.soft_ce(logits, Y.float().mean(1), m)
    assert abs(float(loss) / 4 - float(soft)) < 1e-6


def test_trace_term_value():
    """trace of the mean CM = mean over teachers of tr A_r (Tanno et al. 2019, Listing 1), with the diagonals"""
    A = torch.full((1, 4, 2, 2, 10), 0.5)
    A[:, 0, 0, 0], A[:, 0, 0, 1] = 0.9, 0.1          # teacher 0: specificity 0.9
    _, parts = losses.noisy_annotator(torch.zeros(1, 2, 10), A, torch.zeros(1, 4, 10).long(), torch.ones(1, 10), 1.0)
    assert abs(float(parts["trace"]) - (1.4 + 3 * 1.0) / 4) < 1e-6
    assert abs(float(parts["diag0"]) - (0.9 + 3 * 0.5) / 4) < 1e-6
    assert abs(float(parts["diag1"]) - 0.5) < 1e-6


def test_collapse_guard():
    import train
    row = {"train_diag0": 0.999, "train_diag1": 0.431}          # rules4 s0 pass 3 at lambda 0.01: kept
    assert train.collapsed(row, warm=False) is None and row["dominant"] is False
    assert train.collapsed({"train_diag0": 0.001, "train_diag1": 0.04}, warm=True) is None     # warm-up: frozen
    assert train.collapsed({"train_diag0": 0.009, "train_diag1": 0.004}, warm=False) == dict(diag0=0.009, diag1=0.004)
    assert train.collapsed({"train_loss": 0.05}, warm=False) is None                           # plain arms


def test_diagonals_are_weighted_by_the_class_probability():
    """A[1,1] where the student sees no spindle must not count (per-sample matrices)"""
    logits = torch.tensor([[[9.0, -9.0], [-9.0, 9.0]]]).expand(1, 2, 2).clone()   # t0: no spindle, t1: spindle
    A = torch.zeros(1, 4, 2, 2, 2)
    A[..., 0, 0, :], A[..., 0, 1, :] = 0.95, 0.05
    A[..., 1, 1, 0], A[..., 1, 0, 0] = 0.0, 1.0                  # spindle row at t0: irrelevant
    A[..., 1, 1, 1], A[..., 1, 0, 1] = 0.7, 0.3                  # spindle row at t1: recall 0.7
    _, parts = losses.noisy_annotator(logits, A, torch.zeros(1, 4, 2).long(), torch.ones(1, 2), 0.01)
    assert abs(float(parts["diag1"]) - 0.7) < 1e-3 and abs(float(parts["diag0"]) - 0.95) < 1e-3


def test_gabor_frequencies_stay_in_range():
    sb = M.StreamB()
    with torch.no_grad():
        sb.f_raw.copy_(torch.tensor([-50.0, 50, 0, 3, -3, 10, -10, 1]))
    f = sb.centre_frequencies()
    assert float(f.min()) >= 4.0 and float(f.max()) <= 24.0


def test_fft_filters_equal_direct_convolution():
    torch.manual_seed(1)
    sb = M.StreamB()
    x = torch.randn(2, 3000)
    re, im = sb.gabor()
    xr = F.conv1d(F.pad(x[:, None], (sb.GABOR_HALF,) * 2, mode="reflect"), re)
    xi = F.conv1d(F.pad(x[:, None], (sb.GABOR_HALF,) * 2, mode="reflect"), im)
    ref = F.conv1d(F.pad(xr ** 2 + xi ** 2, (sb.LP_HALF,) * 2, mode="replicate"), sb.lowpass(), groups=8)
    e = sb.envelope(x)
    assert float((e - ref).abs().max() / ref.abs().max()) < 1e-5
    s = torch.sigmoid(sb.s_raw)[:, None]
    k = torch.arange(sb.EMA_LEN - 1, -1, -1, dtype=e.dtype)[None]
    h = s * (1 - s) ** k
    ref_m = F.conv1d(F.pad(e, (sb.EMA_LEN - 1, 0), mode="replicate"), (h / h.sum(1, keepdim=True))[:, None], groups=8)
    assert float(((sb.smoother(e) - ref_m).abs() / ref_m.abs().clamp_min(1e-6)).max()) < 1e-4


def test_pcen_smoother_is_the_recursion():
    sb = M.StreamB()
    e = torch.rand(1, 8, 600) + 0.1
    m = sb.smoother(e)[0].detach().numpy()
    s = torch.sigmoid(sb.s_raw).detach().numpy()
    ev = e[0].numpy()
    rec = np.empty_like(ev)
    rec[:, 0] = ev[:, 0]
    for t in range(1, ev.shape[1]):
        rec[:, t] = (1 - s) * rec[:, t - 1] + s * ev[:, t]
    # truncation at EMA_LEN and the E(0) pre-roll make the two agree up to (1 - s)^EMA_LEN
    assert np.abs(m - rec).max() / ev.max() < 5e-3


def test_segment_cdf_is_zero_for_flat_and_positive_for_skewed():
    p = torch.zeros(1, 400)
    p[0, 100:200] = 0.8
    ev = [[(100, 200)]]
    assert float(losses.segment_cdf(p, ev)) < 1e-10
    q = p.clone()
    q[0, 100:200] = torch.linspace(0.1, 0.9, 100)
    assert float(losses.segment_cdf(q, ev)) > 1e-3


def test_consensus_events_skip_edges_and_invalid():
    mv = np.zeros((1, 500), np.int8)
    mv[0, 0:50] = 1          # touches the crop start: skipped
    mv[0, 100:180] = 1       # kept
    mv[0, 300:320] = 1       # 0.2 s: skipped
    valid = np.ones((1, 500), bool)
    assert losses.consensus_events(mv, valid) == [[(100, 180)]]
    valid[0, 150] = False
    assert losses.consensus_events(mv, valid) == [[]]


def test_d0_matches_sumov2_postprocessing():
    rng = np.random.default_rng(0)
    p = np.clip(rng.random(5000) ** 4 + 0.5 * (np.sin(np.arange(5000) / 40) > 0.9), 0, 1)
    ev = decode.d0(p, 0.5)
    s, e = decode._runs(p > 0.5)
    ref = decode._sumo().merge_and_filter_spindles(np.c_[s, e].astype(float) / 100)
    assert np.allclose(ev[:, :2], ref[:, :2])


def test_d2_durations_and_windowing_exact():
    rng = np.random.default_rng(3)
    p = np.clip(rng.beta(0.4, 4, 30000), 0, 1)
    for i in range(500, 29000, 1500):
        p[i:i + rng.integers(10, 300)] = rng.uniform(0.4, 0.95)
    ev = decode.d2(p)
    dur = ev[:, 1] - ev[:, 0]
    assert len(ev) and (dur >= 0.3 - 1e-9).all() and (dur <= 2.5 + 1e-9).all()
    assert (ev[1:, 0] - ev[:-1, 1] >= 0.1 - 1e-9).all()
    assert np.array_equal(decode._d2_core(p) / 100, ev[:, :2])


def test_d1_recovers_clean_boxes():
    p = np.full(3000, 0.02)
    p[500:580] = 0.9
    p[1500:1620] = 0.8
    ev = decode.d1(p)
    assert np.allclose(ev[:, :2], [[5.0, 5.8], [15.0, 16.2]])


def test_two_training_steps_are_bit_reproducible():
    def run():
        torch.manual_seed(0)
        m = M.Student()
        opt = torch.optim.Adam(m.parameters(), 1e-3)
        g = torch.Generator().manual_seed(1)
        for _ in range(2):
            x = torch.randn(2, 4000, generator=g)
            Y = (torch.rand(2, 4, 4000, generator=g) < 0.05).long()
            out = m(x)
            loss, _ = losses.noisy_annotator(out["logits"], out["A"], Y, torch.ones(2, 4000), 1.0)
            opt.zero_grad()
            loss.backward()
            opt.step()
        return torch.cat([p.detach().flatten() for p in m.parameters()])
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    assert torch.equal(run(), run())


def _receptive_field(trunk, T=8000, draws=5):
    """span of input samples that affect the decoder features at the centre (union over random inputs,
    since a ReLU that is off for one input hides part of the field)"""
    torch.manual_seed(0)
    trunk.eval()
    hit = torch.zeros(T, dtype=torch.bool)
    for _ in range(draws):
        x = torch.randn(1, T, requires_grad=True)
        trunk(x)[1][0, :, T // 2].sum().backward()
        hit |= x.grad[0].abs() > 0
    g = hit.nonzero().flatten()
    return int(g.max() - g.min() + 1)


def test_receptive_fields():
    assert _receptive_field(M.TrunkS()) == RF_S
    assert _receptive_field(M.TrunkM()) == RF_M
    assert RF_M > RF_S


RF_S, RF_M = 400, 464      # samples: 4.00 s and 4.64 s


def test_infer_refuses_without_a_committed_freeze(tmp_path):
    import infer
    with pytest.raises(infer.NotFrozen, match="does not exist"):
        infer.check_frozen(tmp_path / "frozen.json")
    f = paths.ROOT / "results" / "student" / "selection" / "_test_frozen.json"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text('{"runs": {}}')
    try:
        with pytest.raises(infer.NotFrozen, match="not committed"):
            infer.check_frozen(f)
    finally:
        f.unlink()


def test_headline_lambda_is_chosen_by_injection_not_agreement():
    """amendment 9: lambda 0 agrees best with the teachers, but the injected F1 decides"""
    import selection
    import validate

    def run(tag, inj, fa, agree):
        return dict(arm="student_rules4", tag=tag, excluded=False,
                    injected=dict(f1_avg=inj, false_alarms_per_quiet_min=fa),
                    validation=dict(W_over_N2=0.5, REM_over_N2=0.1, per_min_N2=3.0,
                                    **{f"f1_{r}": agree for r in validate.REFS}))
    out = {"a0": run("lam0", 0.50, 0.3, 0.75), "b0": run("lam0", 0.50, 0.3, 0.74),
           "a1": run("", 0.60, 0.5, 0.65), "b1": run("", 0.60, 0.5, 0.66),
           "a2": run("lam0.1", 0.55, 0.4, 0.60), "b2": run("lam0.1", 0.55, 0.4, 0.61)}
    fam = {k: [r for r, v in out.items() if v["tag"] == t] for k, t in selection.LAM_FAMILIES.items()}
    lam, table = selection.headline_lambda(out, fam)
    assert lam == "0.01"
    t = {r["lam"]: r for r in table}
    assert t["0"]["agreement_borda"] == max(r["agreement_borda"] for r in table)   # reported, not used
    for r in ("a2", "b2"):                     # tie on injected F1 -> fewer false alarms wins
        out[r]["injected"]["f1_avg"] = 0.60
    assert selection.headline_lambda(out, fam)[0] == "0.1"


def test_freeze_uses_the_fixed_decoder_not_the_injection_choice(tmp_path, monkeypatch):
    """amendment 10: every run is frozen at D0, tau 0.5, even when the injection criterion prefers another tau"""
    import pandas as pd
    import selection
    monkeypatch.setattr(selection, "SCORES", tmp_path)
    inj = pd.DataFrame([dict(decoder="D0", tau=t, snap=False, f1_avg=f, precision=p, fragmentation=1.0, detections=d,
                             false_alarms_per_quiet_min=fa)
                        for t, f, p, d, fa in [(0.15, 0.52, 0.6, 900, 2.0), (0.5, 0.43, 0.93, 500, 0.2)]])
    val = pd.DataFrame([dict(tau=t, f1_mv=f, per_min_N2=n, plausible=True)
                        for t, f, n in [(0.15, 0.6, 5.1), (0.5, 0.76, 2.9)]])
    inj.to_csv(tmp_path / "r__injection.csv", index=False)
    val.assign(run="r").to_csv(tmp_path / "r__validation.csv", index=False)
    c, ok = selection.at_setting("r")
    assert ok and c["setting"] == dict(decoder="D0", tau=0.5, snap=False)
    assert c["injected"]["f1_avg"] == 0.43 and c["validation"]["f1_mv"] == 0.76
    assert selection.choose("r")["setting"]["tau"] == 0.15          # the old criterion, reported only
    val.loc[val.tau == 0.5, "plausible"] = False
    val.assign(run="r").to_csv(tmp_path / "r__validation.csv", index=False)
    assert selection.at_setting("r")[1] is False                    # implausible at the fixed setting -> excluded


def test_run_names_keep_the_cpu_grid_and_separate_tracks():
    """the CPU grid keeps its folder names; another track gets a suffix, so its runs never collide"""
    import data
    assert data.run_name("student_rules4", "lam0.1", 2) == "student_rules4_lam0.1_s2"
    assert data.run_name("student_soft", "", 0, "cpu") == "student_soft_s0"
    assert data.run_name("student_soft", "", 0, "gpu") == "student_soft_gpu_s0"


def test_launcher_counts_workers_per_device():
    import run_all
    import train
    assert all(run_all.DEVICE[t] == train.TRACKS[t]["device"] for t in train.TRACKS)
    assert run_all.device_of("--arm student_soft --seed 0 --track cpu") == "cpu"   # runs started before --device
    assert run_all.device_of("--arm student_soft --seed 0 --track gpu") == "cuda"
    assert run_all.device_of("--arm student_soft --seed 0 --track cpu --device cuda") == "cuda"


def test_step_loss_on_the_requested_device():
    """train.step_loss moves the batch to the device; on CUDA the loss matches the CPU loss"""
    import argparse
    import data
    import train
    if not torch.cuda.is_available():
        pytest.skip("no CUDA GPU")
    rng = np.random.default_rng(0)
    n, T = 2, data.CORE_LO * 2 + data.CORE
    xb = rng.standard_normal((n, T)).astype(np.float32)
    cb = np.full((n, T), data.VALID_BIT, dtype=np.uint8)
    tables = {"soft": np.linspace(0, 1, 16, dtype=np.float32)}
    a = argparse.Namespace(trunk="S", bigru=False, gdl=False, lam=0.01)
    torch.manual_seed(0)
    net = train.build("student_soft", a, 0.05)
    l_cpu, _ = train.step_loss(net, "student_soft", a, xb, cb, tables, False, "cpu")
    l_gpu, _ = train.step_loss(net.to("cuda"), "student_soft", a, xb, cb, tables, False, "cuda")
    assert l_gpu.device.type == "cuda"
    assert abs(float(l_cpu) - float(l_gpu)) < 1e-4


def test_restart_archives_interrupted_runs(tmp_path, monkeypatch):
    """a run folder without a finished done.json and without a live process reruns from scratch"""
    import data
    import run_all
    monkeypatch.setattr(data, "RUNS", tmp_path / "runs" / "student")
    monkeypatch.setattr(run_all, "alive_runs", lambda: {"student_ds_gpu_s0"})
    for name, done in [("student_soft_gpu_s0", '{"claimed_by": 1}'), ("student_mv_gpu_s0", '{"best_pass": 4}'),
                       ("student_ds_gpu_s0", '{"claimed_by": 2}')]:
        r = data.RUNS / name
        r.mkdir(parents=True)
        (r / "done.json").write_text(done)
        (r / "pass1.pt").write_text("")
    assert run_all.recover_interrupted(["student_soft_gpu_s0", "student_mv_gpu_s0", "student_ds_gpu_s0"]) == \
        ["student_soft_gpu_s0"]
    archived = list((tmp_path / "runs" / "archive" / "interrupted").glob("student_soft_gpu_s0__*"))
    assert len(archived) == 1 and not list(archived[0].glob("*.pt"))
    assert (data.RUNS / "student_mv_gpu_s0").exists() and (data.RUNS / "student_ds_gpu_s0").exists()
