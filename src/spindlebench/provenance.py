"""
Provenance record written next to every result: which code, environment and settings made it.

    from spindlebench.provenance import write_provenance
    write_provenance(out_dir, config={"draws": 10}, protocols=["nap_points.v1"])

-> <out_dir>/provenance.json with: time (UTC), command line, git commit and whether the working tree
had uncommitted changes, Python and package versions, sha256 of uv.lock, the resolved config and the
protocol versions used. Detector environments write the same record through detectors/common.py
(standard library only, so it also works there).
"""
import hashlib
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

from spindlebench import __version__, paths

PACKAGES = ["numpy", "pandas", "scipy", "mne", "yasa", "matplotlib", "affiliation", "torch", "onnxruntime",
            "wonambi", "sssm", "openspindlenet", "pytorch-lightning", "lightning"]


def git_state(root=None):
    root = root or paths.ROOT

    def git(*a):
        try:
            r = subprocess.run(["git", "-C", str(root), *a], capture_output=True, text=True, timeout=30)
            return r.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""

    commit = git("rev-parse", "HEAD")
    status = git("status", "--porcelain", "--untracked-files=no")
    return {"commit": commit or None, "branch": git("rev-parse", "--abbrev-ref", "HEAD") or None,
            "dirty": bool(status) if commit else None}


def package_versions(names=PACKAGES):
    out = {}
    for n in names:
        try:
            out[n] = metadata.version(n)
        except metadata.PackageNotFoundError:
            pass
    return out


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def scrub(obj, root):
    """paths under the repository become <root>/..., the home folder ~/...: records are published"""
    if isinstance(obj, Path):
        obj = str(obj)
    if isinstance(obj, dict):
        return {k: scrub(v, root) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [scrub(v, root) for v in obj]
    if isinstance(obj, str):
        s = obj.replace("\\", "/")
        for base, name in ((Path(root).resolve().as_posix(), "<root>"), (Path.home().as_posix(), "~")):
            if s.lower().startswith(base.lower()):
                return name + s[len(base):]
        return obj
    return obj


def record(config=None, protocols=None):
    lock = paths.ROOT / "uv.lock"
    return scrub({
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "command": [sys.executable, *sys.argv],
        "spindlebench": __version__,
        "git": git_state(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": package_versions(),
        "uv_lock_sha256": sha256(lock) if lock.exists() else None,
        "protocols": list(protocols or []),
        "config": config or {},
    }, paths.ROOT)


def write_provenance(out_dir, config=None, protocols=None, name="provenance.json"):
    out_dir.mkdir(parents=True, exist_ok=True)
    rec = record(config, protocols)
    (out_dir / name).write_text(json.dumps(rec, indent=2, default=str) + "\n", encoding="utf-8")
    return rec


def track(script, config=None, protocols=None):
    """Call once near the top of an experiment script: when the script finishes without an uncaught
    exception, write results/provenance/<study>/<script>[__<args hash>].json. Runs with arguments get
    their own file (the hash of the arguments), so `ablation_e3.py detect` and `... analyse` do not
    overwrite each other. Worker processes and imports (e.g. from tests) record nothing."""
    import atexit
    import multiprocessing
    import time

    script = Path(script).resolve()
    if multiprocessing.parent_process() is not None or Path(sys.argv[0]).resolve() != script:
        return  # a worker process, or the script was imported (e.g. by a test), not run
    rel = script.relative_to(paths.ROOT).with_suffix("")
    parts = rel.parts[1:] if rel.parts[0] == "experiments" else rel.parts
    args = sys.argv[1:]
    name = parts[-1] + (f"__{hashlib.sha256(' '.join(args).encode()).hexdigest()[:8]}" if args else "")
    out_dir = paths.RESULTS.joinpath("provenance", *parts[:-1])
    state = {"failed": False, "t0": time.time()}
    previous = sys.excepthook

    def hook(*exc):
        state["failed"] = True
        previous(*exc)

    def finish():
        if not state["failed"]:
            cfg = {"args": args, **(config or {})}
            rec = write_provenance(out_dir, cfg, protocols, name=f"{name}.json")
            rec["seconds"] = round(time.time() - state["t0"], 1)
            (out_dir / f"{name}.json").write_text(json.dumps(rec, indent=2, default=str) + "\n", encoding="utf-8")

    sys.excepthook = hook
    atexit.register(finish)
