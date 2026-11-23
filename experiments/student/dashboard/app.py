"""
Live dashboard for the student training grid.

    cd experiments/student/dashboard && uv run streamlit run app.py      -> http://localhost:8501

Reads only files the training writes (<repo>/../runs/student/<run>/, SPINDLEBENCH_RUNS: config.json, metrics.csv,
timing.json, heartbeat.json, batches.csv, done.json; logs in results/student/logs/; selection scores in
results/student/selection/) and the process table (psutil). It never touches DREAMS or NAP and needs no
torch, so it runs in its own small environment next to the training.
"""
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import psutil
import streamlit as st

ROOT = Path(os.environ.get("SPINDLEBENCH_ROOT") or Path(__file__).resolve().parents[3])
RESULTS = Path(os.environ.get("SPINDLEBENCH_RESULTS") or ROOT / "results")
STUDY = RESULTS / "student"
RUNS = Path(os.environ.get("SPINDLEBENCH_RUNS") or ROOT.parent / "runs" / "student").resolve()   # = data.RUNS
LOGS = STUDY / "logs"
SEL = STUDY / "selection"

# the job lists of both tracks, as experiments/student/run_all.py builds them (kept in step by hand);
# budgets as in train.TRACKS (MIN_STOP = MIN_PASS + patience: the earliest pass an early stop can happen)
TRACKS = {
    "cpu": dict(seeds=(0, 1, 2), patience=5, max_pass=20, workers=10, bpp=3241,
                caption="Trunk S (SUMOv2's U-Net), batch 12, seeds 0–2"),
    "gpu": dict(seeds=(0, 1, 2, 3, 4), patience=8, max_pass=30, workers=4, bpp=None,
                caption="Trunk M, batch 32, all training data, seeds 0–4"),
}
ARMS = ["student_rules4", "student_soft", "student_mv", "student_ds", "student_a7", "student_luna",
        "student_ucsd", "student_yasa"]
ABLATIONS = ["lam0", "lam0.1", "noB", "global", "nohead2"]
MIN_PASS = 3
REFS = ["a7", "luna_spindles", "wonambi_ucsd", "yasa_default", "mv", "ds"]
REF_NAME = {"a7": "A7", "luna_spindles": "Luna", "wonambi_ucsd": "UCSD", "yasa_default": "YASA",
            "mv": "majority vote", "ds": "Dawid–Skene"}


def all_jobs(track):
    seeds = TRACKS[track]["seeds"]
    out = [("student_rules4", "", s) for s in seeds]
    out += [(a, "", s) for a in ARMS[1:4] for s in seeds]
    out += [(a, "", s) for a in ARMS[4:] for s in seeds]
    out += [("student_rules4", t, s) for t in ABLATIONS for s in seeds]
    return out


def run_name(arm, tag, seed, track="cpu"):
    """= data.run_name: <arm>[_<tag>][_<track>]_s<seed>; the CPU track has no suffix"""
    return arm + (f"_{tag}" if tag else "") + ("" if track == "cpu" else f"_{track}") + f"_s{seed}"


def gpu_stats():
    """nvidia-smi -> dict (util %, memory used/total GB, temperature, power W), or None without an NVIDIA GPU"""
    import subprocess
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,"
                              "power.draw", "--format=csv,noheader,nounits"], capture_output=True, text=True,
                             timeout=5, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout
        name, util, used, total, temp, power = [x.strip() for x in out.splitlines()[0].split(",")]
        return dict(name=name, util=float(util), used=float(used) / 1024, total=float(total) / 1024,
                    temp=float(temp), power=float(power))
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


def arm_id(arm, tag):
    return arm + (f"_{tag}" if tag else "")


# ------------------------------------------------------------------ look and feel

COLORS = {"student_rules4": "#6250d6", "student_soft": "#2a78d6", "student_mv": "#1baf7a", "student_ds": "#008300",
          "student_a7": "#eb6834", "student_luna": "#e87ba4", "student_ucsd": "#e34948", "student_yasa": "#eda100",
          # ablations: steps of the headline's violet (they are variants of it)
          "student_rules4_lam0": "#3b2f99", "student_rules4_lam0.1": "#8579e3", "student_rules4_noB": "#a99ff0",
          "student_rules4_global": "#4d3fb8", "student_rules4_nohead2": "#c6c0f6"}
LABEL = {"student_rules4": "rules4 (headline)", "student_soft": "soft", "student_mv": "majority vote",
         "student_ds": "Dawid–Skene", "student_a7": "A7 only", "student_luna": "Luna only", "student_ucsd": "UCSD only",
         "student_yasa": "YASA only", "student_rules4_lam0": "rules4 · λ 0", "student_rules4_lam0.1": "rules4 · λ 0.1",
         "student_rules4_noB": "rules4 · no stream B", "student_rules4_global": "rules4 · global matrices",
         "student_rules4_nohead2": "rules4 · no head 2"}
DASH = {0: "solid", 1: "dash", 2: "dot", 3: "dashdot", 4: "longdash"}
FAMILIES = {
    "Headline vs plain students": ["student_rules4", "student_soft", "student_mv", "student_ds"],
    "Per-teacher students": ["student_rules4", "student_a7", "student_luna", "student_ucsd", "student_yasa"],
    "Ablations of the headline": ["student_rules4"] + [f"student_rules4_{t}" for t in ABLATIONS],
    "Everything": list(COLORS),
}
INK, INK2, INK3, GRID, PANEL = "#1a1f2b", "#5b6372", "#8f97a3", "#eceff3", "#ffffff"
STATUS_STYLE = {"running": ("#2a78d6", "RUNNING"), "validating": ("#6250d6", "VALIDATING"),
                "starting": ("#8f97a3", "STARTING"), "done": ("#1a1f2b", "DONE"), "queued": ("#c3c8d0", "QUEUED"),
                "failed": ("#d03b3b", "FAILED"), "stopped": ("#d08a00", "STOPPED")}

CSS = f"""
<style>
  .block-container {{padding-top: 1.6rem; max-width: 1500px}}
  h1, h2, h3 {{letter-spacing: -0.01em}}
  .kpi {{background:{PANEL}; border:1px solid {GRID}; border-radius:12px; padding:14px 16px 12px}}
  .kpi .lab {{font-size:12px; color:{INK3}; text-transform:uppercase; letter-spacing:.06em}}
  .kpi .val {{font-size:28px; font-weight:700; color:{INK}; line-height:1.15; font-variant-numeric: tabular-nums}}
  .kpi .sub {{font-size:12.5px; color:{INK2}; margin-top:2px}}
  .chip {{display:inline-block; font-size:11px; font-weight:700; letter-spacing:.06em; padding:2px 8px;
          border-radius:999px; color:white}}
  .muted {{color:{INK3}; font-size:13px}}
  div[data-testid="stTabs"] button p {{font-size:15px; font-weight:600}}
</style>
"""


def fig_style(fig, title=None, height=300, y_title=None, x_title=None):
    fig.update_layout(template="plotly_white", height=height, margin=dict(l=10, r=10, t=40 if title else 10, b=10),
                      title=dict(text=title, font=dict(size=14, color=INK), x=0.01) if title else None,
                      font=dict(color=INK2, size=12), paper_bgcolor=PANEL, plot_bgcolor=PANEL,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(size=11)),
                      hovermode="x unified")
    fig.update_xaxes(gridcolor=GRID, zeroline=False, title=x_title, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, title=y_title, linecolor=GRID)
    return fig


# ------------------------------------------------------------------ reading the runs

def read_json(p):
    try:
        return json.loads(p.read_text() or "{}")
    except (OSError, ValueError):
        return None


def read_csv(p):
    try:
        return pd.read_csv(p)
    except (OSError, ValueError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return None


def live_processes():
    """train.py processes -> {run name: (pid, cpu %, rss GB)} (the venv launcher and its child count once)"""
    out = {}
    for p in psutil.process_iter(["pid", "name", "cmdline", "memory_info", "cpu_percent"]):
        try:
            cmd = p.info["cmdline"] or []
        except (psutil.Error, KeyError):
            continue
        if not any(c.endswith("train.py") for c in cmd):
            continue
        get = lambda flag: cmd[cmd.index(flag) + 1] if flag in cmd else ""
        name = run_name(get("--arm"), get("--tag"), int(get("--seed") or 0), get("--track") or "cpu")
        rss = (p.info["memory_info"].rss if p.info["memory_info"] else 0) / 1e9
        cpu = p.info["cpu_percent"] or 0.0
        pid, c0, r0 = out.get(name, (p.info["pid"], 0.0, 0.0))
        out[name] = (pid, c0 + cpu, max(r0, rss))
    return out


def borda(m):
    df = m[m["pass"] >= MIN_PASS]
    cols = [f"f1_{r}" for r in REFS if f"f1_{r}" in df]
    if df.empty or not cols:
        return None
    score = sum(df[c].rank(method="average") for c in cols)
    return int(df["pass"].iloc[int(np.argmax(score.to_numpy()))])


def default_s_per_batch(arm):
    return 0.95 if arm.startswith("student_rules4") else 0.42


@st.cache_data(ttl=8)
def collect(tick, track):
    procs = live_processes()
    rows, metrics, batches = [], {}, {}
    now = time.time()
    tk = TRACKS[track]
    PATIENCE, MAX_PASS = tk["patience"], tk["max_pass"]
    MIN_STOP = MIN_PASS + PATIENCE
    for arm, tag, seed in all_jobs(track):
        name = run_name(arm, tag, seed, track)
        d = RUNS / name
        cfg = read_json(d / "config.json") if (d / "config.json").exists() else None
        done = read_json(d / "done.json") if (d / "done.json").exists() else None
        m = read_csv(d / "metrics.csv") if (d / "metrics.csv").exists() else None
        tim = read_json(d / "timing.json") if (d / "timing.json").exists() else None
        hb = read_json(d / "heartbeat.json") if (d / "heartbeat.json").exists() else None
        b = read_csv(d / "batches.csv") if (d / "batches.csv").exists() else None
        log = LOGS / f"train_{name}.out"
        alive = name in procs
        finished = bool(done and "best_pass" in done)
        failed = (not alive and not finished and log.exists() and "Traceback" in log.read_text(errors="ignore")[-4000:])
        if finished:
            status = "done"
        elif failed:
            status = "failed"
        elif alive and cfg is None:
            status = "starting"
        elif alive:
            status = "validating" if hb and hb.get("phase") == "validate" else "running"
        elif cfg is not None:
            status = "stopped"
        else:
            status = "queued"
        passes = int(m["pass"].max()) if m is not None and len(m) else 0
        bpp = (cfg or {}).get("batches_per_pass") or (hb or {}).get("batches_per_pass") or tk["bpp"] or 1300
        spb = (tim or {}).get("mean_batch_s") or (hb or {}).get("s_per_batch") or default_s_per_batch(arm)
        val_s = float(np.mean(tim["val_s"])) if tim and tim.get("val_s") else 360.0
        pass_s = float(np.mean(tim["pass_s"])) if tim and tim.get("pass_s") else spb * bpp
        per_pass = pass_s + val_s
        # progress inside the current pass
        t_pass0 = (d / "metrics.csv").stat().st_mtime if m is not None else ((d / "config.json").stat().st_mtime
                                                                             if cfg else now)
        measured = bool(hb and hb.get("pass_") == passes + 1)
        if status in ("running", "validating", "starting"):
            if measured:
                frac = hb["batch"] / max(hb["batches_per_pass"], 1)
                if hb.get("phase") == "validate":
                    frac = 1.0
            else:
                t0 = (d / "metrics.csv").stat().st_mtime if m is not None else ((d / "config.json").stat().st_mtime
                                                                                 if cfg else now)
                frac = min((now - t0) / max(spb * bpp + val_s, 1), 0.99)
        else:
            frac = 0.0
        best = done.get("best_pass") if finished else (borda(m) if m is not None and len(m) else None)
        since = (passes - best) if best else None
        if finished:
            exp_left = worst_left = 0.0
        else:
            if best is None:
                exp_passes = max(MIN_STOP - passes, 1)
            else:
                exp_passes = max(best + PATIENCE - passes, 1)
            exp_left = (exp_passes - frac) * per_pass if status != "queued" else MIN_STOP * per_pass
            worst_left = (MAX_PASS - passes - frac) * per_pass if status != "queued" else MAX_PASS * per_pass
        f1mv = m["f1_mv"] if m is not None and "f1_mv" in m else None
        rows.append(dict(
            run=name, arm=arm_id(arm, tag), seed=seed, status=status, passes=passes,
            pass_progress=100 * (passes + frac) / MAX_PASS, in_pass=100 * frac,
            in_pass_min=f"{(now - t_pass0) / 60:.0f}" if status in ("running", "validating", "starting") else "",
            progress_src="heartbeat" if measured else ("estimate" if status in ("running", "validating") else ""),
            best_pass=best, since_best=since,
            f1_mv_best=float(m.loc[m["pass"] == best, "f1_mv"].iloc[0]) if best and f1mv is not None and (m["pass"] == best).any() else np.nan,
            f1_mv_last=float(f1mv.iloc[-1]) if f1mv is not None and len(f1mv) else np.nan,
            s_per_batch=spb, min_per_pass=per_pass / 60, eta_h=exp_left / 3600, worst_h=worst_left / 3600,
            cpu=procs.get(name, (0, 0, 0))[1], rss=procs.get(name, (0, 0, 0))[2],
            started=(d / "config.json").stat().st_mtime if cfg else np.nan, warm=(hb or {}).get("warm"),
            heartbeat_age=(now - hb["time"]) if hb else np.nan))
        if m is not None:
            metrics[name] = m.assign(run=name, arm=arm_id(arm, tag), seed=seed)
        if b is not None and len(b):
            batches[name] = b.assign(run=name, arm=arm_id(arm, tag), seed=seed)
    return pd.DataFrame(rows), metrics, batches


def grid_eta(df, workers):
    """list scheduling of what is left onto the workers -> (expected, worst) hours from now"""
    out = []
    for col in ("eta_h", "worst_h"):
        running = sorted(df[df.status.isin(["running", "validating", "starting"])][col].tolist())
        queued = df[df.status == "queued"][col].tolist()
        slots = sorted(running + [0.0] * max(workers - len(running), 0))[:max(workers, len(running))]
        for q in queued:
            slots.sort()
            slots[0] += q
        out.append(max(slots) if slots else 0.0)
    return out


# ------------------------------------------------------------------ page

st.set_page_config(page_title="Student training", page_icon="◉", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)

with st.sidebar:
    _default = os.environ.get("STUDENT_TRACK") or ("gpu" if any(RUNS.glob("*_gpu_s*")) else "cpu")
    track = st.radio("Track", ["gpu", "cpu"], index=["gpu", "cpu"].index(_default), horizontal=True,
                     format_func=str.upper)
    st.markdown(f"### Student · {track.upper()} track")
    st.caption(f"{TRACKS[track]['caption']}, {len(all_jobs(track))} runs")
    refresh = st.select_slider("Refresh every", options=[10, 20, 30, 60, 120], value=30, format_func=lambda s: f"{s} s")
    family = st.radio("Arms to compare", list(FAMILIES), index=0)
    group = st.toggle("Group seeds (mean ± range)", value=False)
    workers = st.number_input("Training workers", 1, 32, TRACKS[track]["workers"], key=f"workers_{track}",
                              help="for the grid finish estimate")
    st.divider()
    st.caption(f"Reading {RUNS.relative_to(ROOT) if RUNS.is_relative_to(ROOT) else RUNS}")


def kpi(col, label, value, sub=""):
    col.markdown(f'<div class="kpi"><div class="lab">{label}</div><div class="val">{value}</div>'
                 f'<div class="sub">{sub}</div></div>', unsafe_allow_html=True)


def chip(status):
    c, t = STATUS_STYLE[status]
    return f'<span class="chip" style="background:{c}">{t}</span>'


def series_for(metrics, arms):
    return {k: v for k, v in metrics.items() if v["arm"].iloc[0] in arms}


def legend_html(arms, metrics):
    present = {v["arm"].iloc[0] for v in metrics.values()}
    items = "".join(f'<span style="display:inline-flex;align-items:center;gap:6px;margin-right:18px">'
                    f'<span style="width:22px;height:3px;background:{COLORS[a]};border-radius:2px"></span>'
                    f'{LABEL[a]}</span>' for a in arms if a in present)
    seeds = "" if group else ('<span class="muted">seed 0 ─── · seed 1 ‒ ‒ ‒ · seed 2 ┄┄┄ · ★ selected pass</span>')
    return f'<div style="font-size:13px;color:{INK2};margin:4px 0 10px">{items}{seeds}</div>'


def line_panel(metrics, arms, col, title, y_title=None, best=None, height=280, y_fmt=None):
    fig = go.Figure()
    data = series_for(metrics, arms)
    if group:
        for arm in arms:
            parts = [v for v in data.values() if v["arm"].iloc[0] == arm and col in v]
            if not parts:
                continue
            g = pd.concat(parts).groupby("pass")[col]
            mean, lo, hi = g.mean(), g.min(), g.max()
            c = COLORS[arm]
            fig.add_trace(go.Scatter(x=list(hi.index) + list(lo.index[::-1]), y=list(hi) + list(lo[::-1]), fill="toself",
                                     fillcolor=c, opacity=0.15, line=dict(width=0), hoverinfo="skip", showlegend=False))
            fig.add_trace(go.Scatter(x=mean.index, y=mean, name=LABEL[arm], line=dict(color=c, width=2.5),
                                     mode="lines+markers", marker=dict(size=6)))
    else:
        for name, v in sorted(data.items()):
            if col not in v:
                continue
            arm, seed = v["arm"].iloc[0], int(v["seed"].iloc[0])
            fig.add_trace(go.Scatter(x=v["pass"], y=v[col], name=f"{LABEL[arm]} · s{seed}",
                                     line=dict(color=COLORS[arm], width=2, dash=DASH[seed]), mode="lines+markers",
                                     marker=dict(size=6)))
            if best is not None and name in best and best[name]:
                r = v[v["pass"] == best[name]]
                if len(r):
                    fig.add_trace(go.Scatter(x=r["pass"], y=r[col], mode="markers", showlegend=False,
                                             marker=dict(size=13, symbol="star", color=COLORS[arm],
                                                         line=dict(color="white", width=1.5)),
                                             hovertemplate="selected pass %{x}<extra></extra>"))
    fig_style(fig, title, height, y_title, "pass")
    fig.update_layout(showlegend=False)
    if y_fmt:
        fig.update_yaxes(tickformat=y_fmt)
    return fig


@st.fragment(run_every=refresh)
def page():
    df, metrics, batches = collect(int(time.time() // 8), track)
    gpu = gpu_stats()
    arms = FAMILIES[family]
    best = dict(zip(df.run, df.best_pass))
    counts = df.status.value_counts()
    exp_h, worst_h = grid_eta(df, workers)
    vm = psutil.virtual_memory()
    cpu = psutil.cpu_percent(interval=0.3)
    if "sys" not in st.session_state:
        st.session_state.sys = []
    st.session_state.sys.append(dict(t=pd.Timestamp.now(), cpu=cpu, mem=vm.percent,
                                     gpu=gpu["util"] if gpu else np.nan,
                                     vram=100 * gpu["used"] / gpu["total"] if gpu else np.nan))
    st.session_state.sys = st.session_state.sys[-720:]

    t1, t2 = st.columns([3, 1])
    t1.markdown("## Label-free spindle student · training")
    t2.markdown(f'<div class="muted" style="text-align:right;padding-top:14px">updated {time.strftime("%H:%M:%S")} · '
                f'every {refresh} s</div>', unsafe_allow_html=True)

    c = st.columns(7 if gpu else 6)
    kpi(c[0], "Runs done", f"{counts.get('done', 0)} / {len(df)}",
        f"{counts.get('running', 0) + counts.get('validating', 0)} running · {counts.get('queued', 0)} queued")
    fin = pd.Timestamp.now() + pd.Timedelta(hours=exp_h)
    kpi(c[1], "Grid finishes", fin.strftime("%a %H:%M"), f"expected · worst case "
        f"{(pd.Timestamp.now() + pd.Timedelta(hours=worst_h)).strftime('%a %H:%M')}")
    top = df.dropna(subset=["f1_mv_best"]).sort_values("f1_mv_best", ascending=False)
    kpi(c[2], "Best F1 vs vote", f"{top.f1_mv_best.iloc[0]:.3f}" if len(top) else "—",
        f"{LABEL.get(top.arm.iloc[0], top.arm.iloc[0])} · s{top.seed.iloc[0]}, pass {int(top.best_pass.iloc[0])}"
        if len(top) else "at the selected pass, D0 τ 0.5")
    failed = counts.get("failed", 0) + counts.get("stopped", 0)
    kpi(c[3], "Problems", f"{failed}", "failed or stopped runs" if failed else "none")
    kpi(c[4], "CPU", f"{cpu:.0f}%", f"{psutil.cpu_count()} logical threads")
    kpi(c[5], "Memory", f"{vm.used / 1e9:.1f} GB", f"of {vm.total / 1e9:.0f} GB · {vm.available / 1e9:.1f} free")
    if gpu:
        kpi(c[6], "GPU", f"{gpu['util']:.0f}%", f"{gpu['used']:.1f} / {gpu['total']:.0f} GB · {gpu['temp']:.0f} °C · "
            f"{gpu['power']:.0f} W")
    st.write("")

    tabs = st.tabs(["Runs", "Charts", "Arms", "Live", "System", "Selection", "Logs"])

    # ---- Runs
    with tabs[0]:
        order = {"failed": 0, "stopped": 1, "validating": 2, "running": 3, "starting": 4, "queued": 6, "done": 5}
        v = df.assign(o=df.status.map(order)).sort_values(["o", "arm", "seed"])
        v = v.assign(Status=[STATUS_STYLE[s][1] for s in v.status], Arm=[LABEL.get(a, a) for a in v.arm],
                     Started=[time.strftime("%H:%M", time.localtime(t)) if t == t else "" for t in v.started],
                     Best=[f"{int(b)}" if b == b and b is not None else "" for b in v.best_pass],
                     Since=[f"{int(x)} / 5" if x == x and x is not None else "" for x in v.since_best])
        st.dataframe(
            v[["run", "Arm", "seed", "Status", "passes", "pass_progress", "in_pass", "in_pass_min", "progress_src",
               "Best", "Since", "f1_mv_best", "f1_mv_last", "min_per_pass", "s_per_batch", "eta_h", "worst_h", "cpu",
               "rss", "Started"]],
            hide_index=True, use_container_width=True, height=min(38 * len(v) + 40, 1100),
            column_config={
                "run": st.column_config.TextColumn("Run", width="medium"),
                "seed": st.column_config.NumberColumn("Seed", width="small"),
                "passes": st.column_config.NumberColumn("Pass", help="passes completed"),
                "pass_progress": st.column_config.ProgressColumn("Run progress", min_value=0, max_value=100,
                                                                 format="%.0f%%", help="of the pass budget"),
                "in_pass": st.column_config.ProgressColumn("Current pass", min_value=0, max_value=100, format="%.0f%%"),
                "in_pass_min": st.column_config.TextColumn("In pass (min)"),
                "progress_src": st.column_config.TextColumn("Source", help="heartbeat = measured; estimate = from "
                                                            "elapsed time and the expected speed (runs started before "
                                                            "the heartbeat existed)"),
                "Best": st.column_config.TextColumn("Best pass", help="Borda over the six agreement F1s, pass ≥ 3"),
                "Since": st.column_config.TextColumn("Since best", help="early stop after 5"),
                "f1_mv_best": st.column_config.NumberColumn("F1 vote @ best", format="%.3f"),
                "f1_mv_last": st.column_config.NumberColumn("F1 vote @ last", format="%.3f"),
                "min_per_pass": st.column_config.NumberColumn("min / pass", format="%.0f"),
                "s_per_batch": st.column_config.NumberColumn("s / batch", format="%.2f"),
                "eta_h": st.column_config.NumberColumn("ETA h", format="%.1f", help="expected, assuming early stop"),
                "worst_h": st.column_config.NumberColumn("worst h", format="%.1f", help="all 20 passes"),
                "cpu": st.column_config.NumberColumn("CPU %", format="%.0f"),
                "rss": st.column_config.NumberColumn("RAM GB", format="%.2f"),
            })
        _tk = TRACKS[track]
        st.caption(f"Run progress is against the {_tk['max_pass']}-pass budget; most runs stop earlier "
                   f"({_tk['patience']} passes without a better Borda score, pass {MIN_PASS + _tk['patience']} at the "
                   "earliest). Current-pass progress comes from the heartbeat when the run "
                   "has one, otherwise from elapsed time and the measured speed.")

    # ---- Charts
    with tabs[1]:
        st.markdown(f"**{family}** · agreement on the 40 unlabelled validation nights, per pass "
                    f"(D0 τ = 0.5, centres ±0.5 s).")
        st.markdown(legend_html(arms, metrics), unsafe_allow_html=True)
        cols = st.columns(2)
        cols[0].plotly_chart(line_panel(metrics, arms, "f1_mv", "F1 vs majority vote", best=best, height=340),
                             use_container_width=True)
        cols[1].plotly_chart(line_panel(metrics, arms, "train_loss", "Training loss (pass mean)", height=340),
                             use_container_width=True)
        cols = st.columns(4)
        for i, r in enumerate(["a7", "luna_spindles", "wonambi_ucsd", "yasa_default"]):
            cols[i].plotly_chart(line_panel(metrics, arms, f"f1_{r}", f"F1 vs {REF_NAME[r]}", best=best, height=260),
                                 use_container_width=True)
        cols = st.columns(3)
        cols[0].plotly_chart(line_panel(metrics, arms, "per_min_N2", "Spindles per N2 minute", best=best, height=260),
                             use_container_width=True)
        cols[1].plotly_chart(line_panel(metrics, arms, "W_over_N2", "Wake / N2 density", best=best, height=260),
                             use_container_width=True)
        cols[2].plotly_chart(line_panel(metrics, arms, "dur_median", "Median N2 duration (s)", best=best, height=260),
                             use_container_width=True)
        if any(a.startswith("student_rules4") for a in arms):
            st.markdown("**Headline-family loss terms**")
            cols = st.columns(4)
            for i, (k, t) in enumerate([("train_ce", "Teacher cross-entropy"), ("train_trace", "Σ trace A_r"),
                                        ("train_boundary", "Boundary BCE"), ("train_segment", "Segment CDF")]):
                cols[i].plotly_chart(line_panel(metrics, [a for a in arms if a.startswith("student_rules4")], k, t,
                                                height=240), use_container_width=True)

    # ---- Arms
    with tabs[2]:
        st.markdown("Each arm at its selected (best) pass. Dots are seeds; the bar is their mean.")
        rows = []
        for name, m in metrics.items():
            bp = best.get(name)
            if not bp or not (m["pass"] == bp).any():
                continue
            r = m[m["pass"] == bp].iloc[0]
            rows.append(dict(run=name, arm=m["arm"].iloc[0], seed=int(m["seed"].iloc[0]),
                             **{f"f1_{x}": r.get(f"f1_{x}", np.nan) for x in REFS},
                             per_min_N2=r.get("per_min_N2"), W_over_N2=r.get("W_over_N2")))
        if rows:
            a = pd.DataFrame(rows)
            ref = st.radio("Reference", REFS, index=4, horizontal=True, format_func=lambda r: REF_NAME[r])
            g = a.groupby("arm")[f"f1_{ref}"].mean().sort_values()
            fig = go.Figure()
            fig.add_trace(go.Bar(y=[LABEL.get(x, x) for x in g.index], x=g.values, orientation="h",
                                 marker=dict(color=[COLORS.get(x, INK3) for x in g.index], opacity=0.35),
                                 hovertemplate="%{y}: mean %{x:.3f}<extra></extra>", showlegend=False))
            for arm, sub in a.groupby("arm"):
                fig.add_trace(go.Scatter(y=[LABEL.get(arm, arm)] * len(sub), x=sub[f"f1_{ref}"], mode="markers",
                                         marker=dict(size=11, color=COLORS.get(arm, INK3), line=dict(color="white", width=1.5)),
                                         text=[f"s{s}" for s in sub.seed], hovertemplate="%{text}: %{x:.3f}<extra></extra>",
                                         showlegend=False))
            fig_style(fig, f"F1 vs {REF_NAME[ref]} at the selected pass", 60 + 44 * len(g))
            fig.update_layout(hovermode="closest")
            st.plotly_chart(fig, use_container_width=True)
            t = a.groupby("arm")[[f"f1_{x}" for x in REFS] + ["per_min_N2", "W_over_N2"]].agg(["mean", "std"])
            t.columns = [f"{REF_NAME.get(c[0][3:], c[0])} {c[1]}" for c in t.columns]
            t.index = [LABEL.get(i, i) for i in t.index]
            st.dataframe(t.style.format("{:.3f}", na_rep=""), use_container_width=True)
        else:
            st.info("No run has reached pass 3 yet; the selected pass is chosen from pass 3 on.")

    # ---- Live
    with tabs[3]:
        live = {k: v for k, v in batches.items()}
        if not live:
            st.info("Per-batch curves appear for runs started after the heartbeat was added (every 50 batches). "
                    "The runs started before it report once per pass.")
        else:
            names = sorted(live)
            pick = st.multiselect("Runs", names, default=names[: min(6, len(names))])
            cols = st.columns(2)
            for i, (k, t) in enumerate([("loss", "Loss (50-batch mean)"), ("s_per_batch", "Seconds per batch")]):
                fig = go.Figure()
                for n in pick:
                    b = live[n]
                    arm, seed = b["arm"].iloc[0], int(b["seed"].iloc[0])
                    x = (b["pass"] - 1) + b["batch"] / b["batch"].max()
                    fig.add_trace(go.Scatter(x=x, y=b[k], name=f"{LABEL.get(arm, arm)} · s{seed}",
                                             line=dict(color=COLORS.get(arm, INK3), width=1.6, dash=DASH[seed])))
                cols[i].plotly_chart(fig_style(fig, t, 320, x_title="pass"), use_container_width=True)

    # ---- System
    with tabs[4]:
        s = pd.DataFrame(st.session_state.sys)
        cols = st.columns(2)
        panels = [("cpu", "CPU %"), ("mem", "Memory %")]
        if gpu:
            cols += st.columns(2)
            panels += [("gpu", "GPU %"), ("vram", "GPU memory %")]
        for i, (k, t) in enumerate(panels):
            fig = go.Figure(go.Scatter(x=s.t, y=s[k], fill="tozeroy", line=dict(color="#6250d6", width=2),
                                       fillcolor="rgba(98,80,214,0.12)"))
            fig_style(fig, t + " (since this page opened)", 260)
            fig.update_yaxes(range=[0, 100])
            cols[i].plotly_chart(fig, use_container_width=True)
        j = df[df.status.isin(["running", "validating", "starting"])][["run", "status", "cpu", "rss", "s_per_batch"]]
        st.dataframe(j.rename(columns={"cpu": "CPU %", "rss": "RAM GB", "s_per_batch": "s / batch"}), hide_index=True,
                     use_container_width=True)
        st.caption("Training runs at below-normal priority, so the machine stays responsive.")

    # ---- Selection
    with tabs[5]:
        scores = sorted((SEL / "scores").glob("*__injection.csv")) if (SEL / "scores").exists() else []
        frozen = read_json(SEL / "frozen.json") if (SEL / "frozen.json").exists() else None
        if not scores:
            st.info("Selection has not run yet. After the grid: `uv run python selection.py score`, then `freeze`. "
                    "This tab then shows injected-spindle recall by SNR, the decoder grid and the frozen choices.")
        else:
            inj = pd.concat([read_csv(f) for f in scores if read_csv(f) is not None])
            snr_cols = sorted([c for c in inj.columns if c.startswith("recall_")], key=lambda c: float(c[7:-2]))
            best_rows = inj.sort_values(["f1_avg", "fragmentation"], ascending=[False, True]).groupby("run").head(1)
            fig = go.Figure()
            for r in best_rows.itertuples():
                arm = re.sub(r"(_gpu)?_s\d+$", "", r.run)
                fig.add_trace(go.Scatter(x=[float(c[7:-2]) for c in snr_cols], y=[getattr(r, c) for c in snr_cols],
                                         name=r.run, line=dict(color=COLORS.get(arm, INK3), width=2)))
            fig_style(fig, "Recall of planted spindles at each run's best setting", 360, "recall", "SNR (dB)")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(best_rows[["run", "decoder", "tau", "snap", "f1_avg", "precision", "fragmentation",
                                    "onset_err_median", "offset_err_median"]].round(3), hide_index=True,
                         use_container_width=True)
            if frozen:
                st.success(f"frozen.json written · headline λ = {frozen['headline'].get('lam')} · "
                           f"git {frozen['git']['commit'][:8]}")

    # ---- Logs
    with tabs[6]:
        names = [r for r in df.run if (LOGS / f"train_{r}.out").exists()]
        if names:
            n = st.selectbox("Run", names, index=0)
            txt = (LOGS / f"train_{n}.out").read_text(errors="ignore").splitlines()[-60:]
            st.code("\n".join(txt) or "(empty)", language="json")


page()
