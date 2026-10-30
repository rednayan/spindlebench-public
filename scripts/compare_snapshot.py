"""
Compare result tables with a frozen snapshot (restructure safety check).

    uv run python scripts/compare_snapshot.py [--snapshot _snapshot/2026-10-28] [--only results/nap_scoring] [--tol 0]

The snapshot holds results.tar (every results/*.csv|json|md|txt outside work dirs) and sha256.txt.
Each file in the snapshot is compared with the current file at the same path:
  identical  same bytes
  equal      CSV with the same columns and rows, numbers within --tol (absolute), text identical
  DIFFERENT  anything else (first differing columns are printed)
  missing    file no longer exists
  accepted   listed in <snapshot>/accepted.tsv (path, reason): an intended, documented change
Exit code 1 if any file is DIFFERENT or missing.
"""
import argparse
import hashlib
import io
import sys
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def csv_equal(a, b, tol):
    try:
        x, y = pd.read_csv(io.BytesIO(a)), pd.read_csv(io.BytesIO(b))
    except Exception as e:  # noqa: BLE001
        return False, f"unreadable: {e}"
    if list(x.columns) != list(y.columns):
        return False, f"columns {set(x.columns) ^ set(y.columns)}"
    if len(x) != len(y):
        return False, f"rows {len(x)} vs {len(y)}"
    bad = []
    for c in x.columns:
        if pd.api.types.is_numeric_dtype(x[c]) and pd.api.types.is_numeric_dtype(y[c]):
            u, v = x[c].to_numpy(float), y[c].to_numpy(float)
            same = (np.isnan(u) & np.isnan(v)) | (np.abs(u - v) <= tol)
            if not same.all():
                bad.append(f"{c} (max |d| {np.nanmax(np.abs(u - v)):.3g})")
        elif not x[c].astype(str).equals(y[c].astype(str)):
            bad.append(c)
    return not bad, ", ".join(bad[:5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default="_snapshot/2026-10-28")
    ap.add_argument("--only", action="append", default=[], help="path prefix(es) to check (default: results/)")
    ap.add_argument("--tol", type=float, default=0.0)
    a = ap.parse_args()
    a.only = a.only or ["results/"]
    snap = ROOT / a.snapshot
    counts = {"identical": 0, "equal": 0, "accepted": 0, "DIFFERENT": 0, "missing": 0}
    acc = snap / "accepted.tsv"
    lines = acc.read_text(encoding="utf-8").splitlines() if acc.exists() else []
    accepted = {ln.split("\t")[0] for ln in lines if ln}
    with tarfile.open(snap / "results.tar") as tar:
        for m in tar.getmembers():
            if not m.isfile() or (a.only and not any(m.name.startswith(p) for p in a.only)):
                continue
            old = tar.extractfile(m).read()
            cur = ROOT / m.name
            if not cur.exists():
                counts["missing"] += 1
                print("missing  ", m.name)
                continue
            new = cur.read_bytes()
            if hashlib.sha256(old).digest() == hashlib.sha256(new).digest():
                counts["identical"] += 1
                continue
            if m.name.endswith(".csv"):
                ok, why = csv_equal(old, new, a.tol)
                if ok:
                    counts["equal"] += 1
                    continue
            else:
                ok, why = old.replace(b"\r\n", b"\n") == new.replace(b"\r\n", b"\n"), "text differs"
                if ok:
                    counts["identical"] += 1
                    continue
            if m.name in accepted:
                counts["accepted"] += 1
                continue
            counts["DIFFERENT"] += 1
            print("DIFFERENT", m.name, "-", why)
    print(" ".join(f"{k}={v}" for k, v in counts.items()))
    sys.exit(1 if counts["DIFFERENT"] or counts["missing"] else 0)


if __name__ == "__main__":
    main()
