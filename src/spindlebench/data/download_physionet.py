"""
Download a PhysioNet sleep database, verifying every file against the database's SHA256SUMS.txt.
The checksum list always comes from physionet.org; the files come from PhysioNet's AWS open-data
mirror (s3://physionet-open, about 80x faster here) unless --source physionet.
Files already present with the right checksum are skipped, so the script can be resumed.

    uv run spindlebench download hmc               # -> <data>/HMC/       (15.7 GB)
    uv run spindlebench download sleepedf          # -> <data>/SLEEPEDF/  (sleep-cassette only)
    uv run spindlebench download cinc2018 [--n 250]  # -> <data>/CINC2018/ (~0.13 GB per night)

hmc       Haaglanden Medisch Centrum sleep staging database v1.1 (Alvarez-Estevez & Rijsman 2021,
          doi:10.13026/t79q-fr32), CC BY 4.0. 151 whole-night PSGs, AASM staging, no spindle labels.
sleepedf  Sleep-EDF Expanded v1.0.0 (Kemp et al. 2000; Goldberger et al. 2000), Open Data Commons
          Attribution. Only the sleep-cassette study (healthy, ages 25-101, mostly two nights per
          person, R&K staging); the sleep-telemetry (temazepam) study is not downloaded.
cinc2018  PhysioNet/CinC Challenge 2018 v1.0.0 (Ghassemi et al. 2018, doi:10.13026/6phb-r450),
          Open Data Commons Attribution. 994 MGH training PSGs at 200 Hz with F3-M2 F4-M1 C3-M2
          C4-M1 O1-M2 O2-M1 and AASM staging. Only a subset of the training set: the 715 records of
          people aged 18-64 (age-sex.csv) in a fixed random order (seed 2018), the first --n of them,
          so a larger --n is a superset of a smaller one. Per record the WFDB header, the signal
          .mat and the .arousal annotation file (stages); the v7.3 -arousal.mat copy is skipped.
          The chosen records are listed in <data>/CINC2018/records_used.csv.
All three are label-free for spindles in this benchmark (label-free validity, teacher corpus,
SpindleInject backgrounds).
"""
import argparse
import csv
import hashlib
import io
import random
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from spindlebench import paths

CINC_SEED, CINC_AGES = 2018, (18, 64)


def keep_hmc(files, a):
    return files


def keep_sleepedf(files, a):
    extra = ("SC-subjects.xls", "RECORDS", "LICENSE.txt")
    return [f for f in files if f[0].startswith("sleep-cassette/") or f[0] in extra]


def cinc_records(n):
    """the first n training records of people aged 18-64, in a fixed random order"""
    with urllib.request.urlopen(CHECKSUMS + DBS["cinc2018"][0] + "age-sex.csv", timeout=120) as r:
        rows = list(csv.DictReader(io.StringIO(r.read().decode())))
    rows = sorted((r for r in rows if r["Record"].startswith("tr") and CINC_AGES[0] <= int(r["Age"]) <= CINC_AGES[1]),
                  key=lambda r: r["Record"])
    random.Random(CINC_SEED).shuffle(rows)
    return rows[:n]


def keep_cinc2018(files, a):
    pick = cinc_records(a.n)
    out = paths.raw("cinc2018")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "records_used.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, ["Record", "Sex", "Age"])
        w.writeheader()
        w.writerows(pick)
    want = {f"training/{r['Record']}/{r['Record']}{ext}" for r in pick for ext in (".hea", ".mat", ".arousal")}
    return [f for f in files if f[0] in want or f[0] in ("age-sex.csv", "RECORDS")]


DBS = {
    "hmc": ("hmc-sleep-staging/1.1/", paths.raw("hmc"), keep_hmc),
    "sleepedf": ("sleep-edfx/1.0.0/", paths.raw("sleepedf"), keep_sleepedf),
    "cinc2018": ("challenge-2018/1.0.0/", paths.raw("cinc2018"), keep_cinc2018),
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
    ap.add_argument("--n", type=int, default=250, help="cinc2018 only: number of records")
    a = ap.parse_args(argv)
    rel, out, keep = DBS[a.db]
    base = SOURCES[a.source] + rel
    with urllib.request.urlopen(CHECKSUMS + rel + "SHA256SUMS.txt", timeout=120) as r:
        lines = r.read().decode().splitlines()
    files = keep([(ln.split()[1], ln.split()[0]) for ln in lines if len(ln.split()) == 2], a)
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
