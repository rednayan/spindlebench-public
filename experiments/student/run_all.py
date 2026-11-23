"""
The training grid of one track (plan section 7): a job list and a pool of single-thread workers.

    uv run python run_all.py --track cpu|gpu [--device cpu|cuda] [--workers N] [--only <substring>] [--dry-run]

Each job is one `train.py` run; a job whose run folder (data.RUNS/<run>) already has done.json is skipped,
so the script can be restarted. On start, a run folder left unfinished with no live process (reboot, killed
launcher) is archived to <runs>/../archive/interrupted/ and the job reruns from scratch. Logs: results/student/logs/train_<run>.out. Order = the plan's order of
execution: headline first, then plain students, per-teacher students, ablations.

Track and device are separate (train.py): --track picks the settings and the job list, --device where the
jobs run (default: the track's own device, cpu -> cpu, gpu -> cuda). CPU and CUDA jobs have separate
worker limits (--workers; default 10 on cpu, 3 on cuda: several small runs share one GPU), so a CPU
launcher and a GPU launcher can run side by side. Examples:
    uv run python run_all.py --track cpu                       the CPU grid on the CPU
    uv run python run_all.py --track cpu --device cuda         the same jobs on the GPU
    uv run python run_all.py --track gpu --workers 2           the GPU track (trunk M, 5 seeds)

CPU track (trunk S, seeds 0, 1, 2), 39 jobs here; the SUMOv2-as-fifth-teacher arm (3 more) needs SUMOv2
run on the corpus and is added once it has been:
    student_rules4 (lambda 0.01, amendment 8)                         3
    student_soft, student_mv, student_ds                              9
    student_a7, student_luna, student_ucsd, student_yasa             12
    ablations of student_rules4: lambda 0, lambda 0.1, stream B off,
    global matrices, head 2 off                                       15
The pool never runs more than --workers training processes on the machine, counting runs started by
other launchers (or left running by a stopped one): a job waits until a slot is free.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import data

SEEDS = {"cpu": (0, 1, 2), "gpu": (0, 1, 2, 3, 4)}
DEVICE = {"cpu": "cpu", "gpu": "cuda"}          # each track's default device (= train.TRACKS[track]["device"])
WORKERS = {"cpu": 10, "cuda": 3}                # default worker limit per device
# training runs below normal priority so the machine stays responsive (they still use every idle cycle)
LOW_PRIORITY = (dict(creationflags=subprocess.BELOW_NORMAL_PRIORITY_CLASS) if os.name == "nt"
                else dict(preexec_fn=lambda: os.nice(10)))
ABLATIONS_CPU = [("lam0", ["--lam", "0"]), ("lam0.1", ["--lam", "0.1"]), ("noB", ["--no-stream-b"]),
                 ("global", ["--noise", "global"]), ("nohead2", ["--no-boundary"])]


def jobs(track):
    seeds = SEEDS[track]
    out = []
    for s in seeds:
        out.append(("student_rules4", "", [], s))
    for arm in ("student_soft", "student_mv", "student_ds"):
        for s in seeds:
            out.append((arm, "", [], s))
    for arm in ("student_a7", "student_luna", "student_ucsd", "student_yasa"):
        for s in seeds:
            out.append((arm, "", [], s))
    for tag, flags in ABLATIONS_CPU:
        for s in seeds:
            out.append(("student_rules4", tag, flags, s))
    return out


PILOT_PASSES = 5      # rules4-fix pilot: the collapse showed at pass 3; passes 1-5 of the full schedule


def pilot_jobs(track):
    """rules4-fix pilot, seeds 0 and 1, all with lam 0 (the trace term pushed the wrong way in the grid):
        fixA  noise tables anchored at the Dawid-Skene matrices, per-sample correction bounded by +-0.5 logits
        fixB  noise tables fixed at the Dawid-Skene matrices (delta 0)
        fixC  fixA with the trunk warm-started from student_ds of the same seed and track (main runs folder:
              SPINDLEBENCH_RUNS_MAIN, default <runs>/../student)"""
    main_runs = Path(os.environ.get("SPINDLEBENCH_RUNS_MAIN") or data.RUNS.parent / "student")
    base = ["--lam", "0", "--anchor", "ds", "--stop-pass", str(PILOT_PASSES)]
    out = []
    for s in (0, 1):
        ds_run = main_runs / data.run_name("student_ds", "", s, track)
        out += [("student_rules4", "fixA", base + ["--anchor-delta", "0.5"], s),
                ("student_rules4", "fixB", base + ["--anchor-delta", "0"], s),
                ("student_rules4", "fixC", base + ["--anchor-delta", "0.5", "--init-from", str(ds_run)], s)]
    return out


def fix_jobs(track):
    """fixA as full runs (every seed of the track, the full schedule with early stop): lam 0, noise tables
    anchored at the Dawid-Skene matrices, per-sample correction bounded by +-0.5 logits"""
    return [("student_rules4", "fixA", ["--lam", "0", "--anchor", "ds", "--anchor-delta", "0.5"], s)
            for s in SEEDS[track]]


def is_done(name):
    """a finished run: done.json with the selected pass (a launcher's claim placeholder has none)"""
    f = data.RUNS / name / "done.json"
    return f.exists() and "best_pass" in json.loads(f.read_text() or "{}")


def claim(name):
    """reserve a job for this launcher: an exclusive placeholder done.json, which other launchers (and the
    first version of this script, which checks only for done.json) skip; refused if the job has started"""
    run = data.RUNS / name
    if (run / "config.json").exists() or (run / "done.json").exists():
        return False
    run.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(run / "done.json", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w") as f:
        f.write(json.dumps(dict(claimed_by=os.getpid(), at=time.strftime("%Y-%m-%d %H:%M"))))
    return True


def device_of(cmdline):
    """the device a train.py command line runs on: --device, else its track's default"""
    args = cmdline.split()
    if "--device" in args[:-1]:
        return args[args.index("--device") + 1]
    return DEVICE[args[args.index("--track") + 1]] if "--track" in args[:-1] else "cpu"


def running_trainings(device):
    """number of train.py runs on `device` alive on this machine (any launcher); a run counts once
    although a venv launcher process and the interpreter both carry its command line"""
    return sum(device_of(c) == device for c in train_cmdlines())


def train_cmdlines():
    """the train.py arguments of every training process alive (once per run)"""
    if os.name == "nt":
        ps = ["powershell", "-NoProfile", "-Command",
              "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | ForEach-Object { $_.CommandLine }"]
    else:
        ps = ["ps", "-eo", "args"]
    lines = subprocess.run(ps, capture_output=True, text=True).stdout.splitlines()
    return {ln.split("train.py", 1)[1].strip() for ln in lines if "train.py" in ln and "--arm" in ln}


def alive_runs():
    """run names of the train.py processes alive on this machine"""
    out = set()
    for c in train_cmdlines():
        a = c.split()
        get = lambda k, d="": a[a.index(k) + 1] if k in a[:-1] else d   # noqa: E731
        out.add(data.run_name(get("--arm"), get("--tag"), get("--seed", "0"), get("--track", "cpu")))
    return out


def recover_interrupted(names):
    """a run folder without a finished done.json and without a live process was cut off (reboot, killed
    launcher): move it to <runs>/../archive/interrupted/<run>__<time>/ without its checkpoints, so the job
    starts again from scratch (a pass checkpoint holds no optimiser or sampler state). -> moved names"""
    alive, moved = alive_runs(), []
    for name in names:
        run = data.RUNS / name
        if not run.exists() or is_done(name) or name in alive:
            continue
        dst = data.RUNS.parent / "archive" / "interrupted" / f"{name}__{time.strftime('%Y%m%d-%H%M')}"
        dst.parent.mkdir(parents=True, exist_ok=True)
        for ck in run.glob("pass*.pt"):
            ck.unlink()
        shutil.move(str(run), str(dst))
        moved.append(name)
    return moved


SLOT = threading.Lock()


def run_one(job, track, device, workers):
    arm, tag, flags, seed = job
    name = data.run_name(arm, tag, seed, track)
    if is_done(name) or (data.RUNS / name / "config.json").exists():   # finished or running elsewhere
        return name, "skipped", 0.0
    cmd = [sys.executable, "-W", "ignore", str(data.here() / "train.py"), "--arm", arm, "--seed", str(seed),
           "--track", track, "--device", device, *flags] + (["--tag", tag] if tag else [])
    log = data.STUDY / "logs" / f"train_{name}.out"
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    with SLOT:                                   # one job at a time waits for a free slot, then starts
        while running_trainings(device) >= workers:
            time.sleep(60)
        if not claim(name):
            return name, "skipped", 0.0
        f = open(log, "w")
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=data.here(), env=env, **LOW_PRIORITY)
        time.sleep(20)                           # until the new process is visible to running_trainings()
    t = time.time()
    with f:
        rc = proc.wait()
    if rc != 0 and not is_done(name):
        (data.RUNS / name / "done.json").unlink(missing_ok=True)   # release the claim
    return name, "ok" if rc == 0 else f"exit {rc}", (time.time() - t) / 3600


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", default="cpu", choices=list(SEEDS))
    ap.add_argument("--device", default=None, choices=list(WORKERS), help="default: the track's device")
    ap.add_argument("--workers", type=int, default=None, help="default: 10 on cpu, 3 on cuda")
    ap.add_argument("--only", default=None, help="run only jobs whose run name contains this")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--reverse", action="store_true", help="take jobs from the end of the queue (a second launcher)")
    ap.add_argument("--first", default=None,
                    help="comma-separated substrings: jobs whose run name contains one move to the front of the "
                         "queue, in this order (the rest keep their order), e.g. lam0_gpu,student_mv,student_ds")
    ap.add_argument("--pilot", action="store_true", help="the rules4-fix pilot jobs instead of the grid "
                    "(set SPINDLEBENCH_RUNS to a separate folder)")
    ap.add_argument("--fix", action="store_true", help="the fixA full runs instead of the grid "
                    "(set SPINDLEBENCH_RUNS to a separate folder)")
    a = ap.parse_args()
    if (a.pilot or a.fix) and data.RUNS.name == "student":
        raise SystemExit("--pilot / --fix: set SPINDLEBENCH_RUNS to a separate folder, e.g. ../runs/student_fix")
    device = a.device or DEVICE[a.track]
    workers = a.workers or WORKERS[device]
    name = lambda j: data.run_name(j[0], j[1], j[3], a.track)   # noqa: E731
    js = pilot_jobs(a.track) if a.pilot else fix_jobs(a.track) if a.fix else jobs(a.track)
    if a.only:
        js = [j for j in js if a.only in name(j)]
    if not a.dry_run:
        for n in recover_interrupted([name(j) for j in js]):
            print(f"{n}: interrupted earlier, archived; it reruns from scratch", flush=True)
    todo = [j for j in js if not is_done(name(j))]
    if a.reverse:
        todo = todo[::-1]
    if a.first:
        keys = [k for k in a.first.split(",") if k]
        todo.sort(key=lambda j: next((i for i, k in enumerate(keys) if k in name(j)), len(keys)))   # stable
    print(f"{len(js)} jobs, {len(js) - len(todo)} done, {len(todo)} to run on {workers} {device} workers "
          f"-> {data.RUNS}", flush=True)
    if a.dry_run:
        for j in todo:
            print(name(j), " ".join(j[2]))
        return
    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(run_one, j, a.track, device, workers) for j in todo]
        for fu in as_completed(futs):
            name, status, h = fu.result()
            print(f"{time.strftime('%H:%M')} {name}: {status} ({h:.1f} h)", flush=True)


if __name__ == "__main__":
    main()
