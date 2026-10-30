"""
Evaluation benchmark, step 1 (plan section 4): injected caches for the detectors.

    .venv/Scripts/python.exe injection/build_bench.py [--variants 2]

For each variant v and each DREAMS spindle excerpt / included NAP nap:
  cache_inj/v<v>/signals/<ds>/<rec>.npz     default channel only, raw uV + injected spindles,
                                            same format as cache/ (data, sf, ch_names)
  cache_inj/v<v>/manifest.csv               same columns as cache/manifest.csv
  results/injection/bench/v<v>/events/<ds>/<rec>.csv    injected events (ground truth)
  results/injection/bench/v<v>/masks/<ds>/<rec>.npz     quiet-N2 mask at native rate (packed bits)
SNR on a fixed grid (each value equally often, random order); frequency, drift and duration as in
Params. Seeds differ from the training variants (experiments/injection/sumo_inject_ft/prepare.py).
"""
import argparse

import numpy as np
import pandas as pd

from spindlebench import paths
from spindlebench.data.cache import signal_path
from spindlebench.synth.generator import Params, eligible_mask, inject
from spindlebench import provenance  # noqa: E402

provenance.track(__file__)

SNR_GRID = np.array([-6.0, -3.0, 0.0, 3.0, 6.0, 9.0, 12.0])
DATASETS = ["dreams_spindles", "nap"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", type=int, default=2)
    a = ap.parse_args()
    m = pd.read_csv(paths.CACHE / "manifest.csv")
    m = m[m.dataset.isin(DATASETS) & ~m.excluded.astype(bool)].reset_index(drop=True)
    p = Params()
    for v in range(a.variants):
        cdir = paths.CACHE_INJ / f"v{v}"
        bdir = paths.RESULTS / "injection" / "bench" / f"v{v}"
        rows = []
        for k, r in enumerate(m.itertuples()):
            z = np.load(signal_path(r.path))
            ch = [str(c) for c in z["ch_names"]]
            x = z["data"][ch.index(r.default_channel)].astype(float)
            sf = float(z["sf"])
            elig, _, used = eligible_mask(r.dataset, r.rec, r.default_channel, len(x), sf, p)
            rng = np.random.default_rng(7_000_000 + 1000 * v + k)
            grid = np.concatenate([rng.permutation(SNR_GRID) for _ in range(200)])
            xi, ev = inject(x, sf, elig, p, seed=9_000_000 + 1000 * v + k, snr_db=grid)
            sp = cdir / "signals" / r.dataset / f"{r.rec}.npz"
            sp.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(sp, data=xi[None].astype(np.float32), sf=sf, ch_names=np.array([r.default_channel]))
            ep = bdir / "events" / r.dataset / f"{r.rec}.csv"
            ep.parent.mkdir(parents=True, exist_ok=True)
            ev.to_csv(ep, index=False)
            mp = bdir / "masks" / r.dataset / f"{r.rec}.npz"
            mp.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(mp, quiet=np.packbits(elig), n=len(elig), sf=sf, quiet_detectors="+".join(used))
            row = r._asdict()
            row.pop("Index")
            row.update(path=sp.relative_to(cdir).as_posix(), n_channels=1)  # relative to the manifest
            rows.append(row)
            print(f"v{v} {r.dataset:16s} {r.rec:10s} injected {len(ev):3d}", flush=True)
        pd.DataFrame(rows)[m.columns].to_csv(cdir / "manifest.csv", index=False)


if __name__ == "__main__":
    main()
