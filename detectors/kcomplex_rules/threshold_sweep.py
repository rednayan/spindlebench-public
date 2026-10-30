"""
Sensitivity of the Spinky K-complex port to its trough threshold. The benchmark uses -35 µV, our
choice: the paper gives no default (it calibrates per subject; its one example value is -60 µV).

    .venv/Scripts/python detectors/kcomplex_rules/threshold_sweep.py [--workers 8]
    uv run python experiments/rq1_benchmark/evaluate.py --store results/sensitivity/detections --datasets dreams_kc,nap \
        --out results/sensitivity/spinky_threshold

Each recording is decomposed once (TQWT-MCA); only kc_detection is repeated per threshold. Writes
results/sensitivity/detections/<dataset>/spinky_kc_m<|threshold|>/<rec>.csv.
"""
import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import common  # noqa: E402
import spinky as sk  # noqa: E402

THRESHOLDS = [-25.0, -30.0, -35.0, -40.0, -45.0, -50.0, -60.0]


def sweep(task):
    x, sf, ch = task
    d = sk.DEF
    L = int(round(d["epoch_s"] * sf))
    n_ep = len(x) // L
    transit, _ = sk.signal_decomposition(x[:n_ep * L].reshape(n_ep, L), sf)
    delay = d["kc_order"] / 2
    out = {t: [] for t in THRESHOLDS}
    for e in range(n_ep):
        for t in THRESHOLDS:
            troughs, k = sk.kc_detection(transit[e], t, sf)
            for i in troughs:
                s, en = sk.kc_extent(k, i)
                out[t].append(dict(onset=max((e * L + s - delay) / sf, 0.0), duration=(en - s + 1) / sf,
                                   channel=ch, event="kcomplex", score=-float(k[i])))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--workers", type=int, default=8)
    extra, rest = ap.parse_known_args()
    sys.argv = [sys.argv[0], "--out", str(common.RESULTS / "sensitivity" / "detections")] + rest
    args = common.parse_args(["dreams_kc", "nap"])
    recs = list(common.iter_recordings(args))
    tasks = [(data[ch_names.index(chans[0])].astype(float), sf, chans[0]) for _, data, sf, ch_names, chans in recs]
    with ProcessPoolExecutor(extra.workers) as pool:
        for (row, *_), res in zip(recs, pool.map(sweep, tasks)):
            n = {t: common.write(args, row, f"spinky_kc_m{abs(int(t))}", rows) for t, rows in res.items()}
            print(row.dataset, row.rec, n, flush=True)
