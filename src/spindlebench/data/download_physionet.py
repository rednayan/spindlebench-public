"""
Download a PhysioNet sleep database, verifying every file against the database's SHA256SUMS.txt.
The checksum list always comes from physionet.org; the files come from PhysioNet's AWS open-data
mirror (s3://physionet-open, about 80x faster here) unless --source physionet.
Files already present with the right checksum are skipped, so the script can be resumed.

    uv run spindlebench download hmc        # -> <data>/HMC/       (15.7 GB)
    uv run spindlebench download sleepedf   # -> <data>/SLEEPEDF/  (sleep-cassette only)

hmc       Haaglanden Medisch Centrum sleep staging database v1.1 (Alvarez-Estevez & Rijsman 2021,
          doi:10.13026/t79q-fr32), CC BY 4.0. 151 whole-night PSGs, AASM staging, no spindle labels.
sleepedf  Sleep-EDF Expanded v1.0.0 (Kemp et al. 2000; Goldberger et al. 2000), Open Data Commons
          Attribution. Only the sleep-cassette study (healthy, ages 25-101, mostly two nights per
          person, R&K staging); the sleep-telemetry (temazepam) study is not downloaded.
Both are label-free for spindles in this benchmark (label-free validity, SpindleInject backgrounds).
"""
import argparse
import hashlib
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from spindlebench import paths

DBS = {
    "hmc": ("hmc-sleep-staging/1.1/", paths.raw("hmc"), lambda f: True),
    "sleepedf": ("sleep-edfx/1.0.0/", paths.raw("sleepedf"),
                 lambda f: f.startswith("sleep-cassette/") or f in ("SC-subjects.xls", "RECORDS", "LICENSE.txt")),
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(base, out, name, digest):
    dst = out / name
    if dst.exists() and sha256(dst) == digest:
        return name, "ok (present)"
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    for attempt in range(4):
        try:
            with urllib.request.urlopen(base + name, timeout=300) as r, open(tmp, "wb") as fh:
                while chunk := r.read(1 << 20):
                    fh.write(chunk)
            if sha256(tmp) != digest:
                raise IOError("checksum mismatch")
            tmp.replace(dst)
            return name, "ok"
        except Exception as e:  # noqa: BLE001
            err = e
            time.sleep(5 * (attempt + 1))
    return name, f"FAILED: {err}"


SOURCES = {"s3": "https://physionet-open.s3.amazonaws.com/", "physionet": "https://physionet.org/files/"}
CHECKSUMS = "https://physionet.org/files/"


def main(argv=None):
    ap = argparse.ArgumentParser(prog="spindlebench download")
    ap.add_argument("db", choices=DBS)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--source", choices=SOURCES, default="s3")
    a = ap.parse_args(argv)
    rel, out, keep = DBS[a.db]
    base = SOURCES[a.source] + rel
    with urllib.request.urlopen(CHECKSUMS + rel + "SHA256SUMS.txt", timeout=120) as r:
        lines = r.read().decode().splitlines()
    files = [(ln.split()[1], ln.split()[0]) for ln in lines if len(ln.split()) == 2 and keep(ln.split()[1])]
    print(f"{a.db}: {len(files)} files from {a.source} -> {out}/", flush=True)
    t0, fails = time.time(), 0
    with ThreadPoolExecutor(a.workers) as ex:
        for i, (name, status) in enumerate(ex.map(lambda f: fetch(base, out, *f), files), 1):
            fails += status.startswith("FAILED")
            if status.startswith("FAILED") or i % 25 == 0 or i == len(files):
                print(f"[{i}/{len(files)}] {name}: {status} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    print(f"done, {fails} failures", flush=True)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
