"""
Download the Nap EEG dataset (Mei, Grossberg, Ng, Navarro & Ellmore, OSF project chav7).

    uv run spindlebench download nap [out_dir] [--annotations-only]   # -> <data>/NapEEG/

Pulls two OSF components into out_dir (default NapEEG/):
  raw/          yb6ft  BrainVision .vhdr/.vmrk/.eeg, 64 ch, 1 kHz (~18.5 GB)
  annotations/  h68a3  "Manual Spindle Files" (manual spindles + stages)
Resumable: files already on disk with the right size are skipped.
"""
import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from spindlebench import paths

COMPONENTS = {"raw": "yb6ft", "annotations": "h68a3"}
ANNOT_FOLDER = "Manual Spindle Files/"


def get_all(url):
    out = []
    while url:
        d = json.load(urllib.request.urlopen(url, timeout=60))
        out += d["data"]
        url = d["links"].get("next")
    return out


def walk(url, prefix=""):
    for x in get_all(url):
        a = x["attributes"]
        if a["kind"] == "folder":
            yield from walk(x["relationships"]["files"]["links"]["related"]["href"], prefix + a["name"] + "/")
        else:
            yield prefix + a["name"], a["size"], x["links"]["download"]


def fetch(job):
    dest, size, url = job
    if dest.exists() and dest.stat().st_size == size:
        return f"skip {dest.name}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    err = "size mismatch"
    for _attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            if tmp.stat().st_size == size:
                tmp.replace(dest)
                return f"ok   {dest.name} ({size / 1e6:.0f} MB)"
        except Exception as e:  # network hiccup: retry
            err = e
    return f"FAIL {dest.name}: {err}"


def main(argv=None):
    if argv is not None:
        sys.argv = [sys.argv[0], *argv]
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    out = Path(args[0]) if args else paths.raw("nap")
    only_annot = "--annotations-only" in sys.argv
    jobs = []
    for sub, node in COMPONENTS.items():
        if only_annot and sub == "raw":
            continue
        for path, size, url in walk(f"https://api.osf.io/v2/nodes/{node}/files/osfstorage/"):
            if sub == "annotations":
                if not path.startswith(ANNOT_FOLDER):
                    continue
                path = path[len(ANNOT_FOLDER):]
            jobs.append((out / sub / path, size, url))
    jobs.sort(key=lambda j: j[1])  # small files first
    print(f"{len(jobs)} files, {sum(j[1] for j in jobs) / 1e9:.1f} GB -> {out}", flush=True)
    with ThreadPoolExecutor(4) as ex:
        for msg in ex.map(fetch, jobs):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
