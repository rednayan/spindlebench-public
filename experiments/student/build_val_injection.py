"""
Injected copies of the unlabelled validation nights (plan 6.2, 6.3): spindles with exact boundaries
and no expert anywhere.

    uv run python build_val_injection.py [--variants 4]      (from experiments/student)

For each variant v and each validation night (splits.csv, role == val; 15 HMC + 25 CinC):
  <cache>/../cache_student_inj/v<v>/
      manifest.csv                        same columns as cache/manifest.csv, one channel
      signals/<ds>/<rec>.npz              default channel, raw uV + injected spindles (data, sf, ch_names)
      labels/<ds>/<rec>__hypno.csv        copied: the public hypnogram
      labels/<ds>/<rec>__artefact.csv     copied: the artefacts.v1 mask of the ORIGINAL night (the teachers
                                          run with --clean on the same windows as before)
  results/student/injection/v<v>/events/<ds>/<rec>.csv   injected events: onset, duration, freq, drift,
                                                         snr_db, amp_ptp, bg_sigma_rms
  results/student/injection/v<v>/quiet/<ds>/<rec>.npy    quiet mask on the 100 Hz grid (packed bits)

Generator: spindlebench.synth.generator.inject with the frozen Params of plan_injection.md
(11-16 Hz start frequency, drift -1..+0.5 Hz, log-normal duration median 0.8 s clipped 0.5-2 s,
6 events per quiet N2 minute, >= 3 s apart). SNR on the plan's grid -3, 0, 3, 6, 9, 12 dB (each value
equally often, random order). Eligible ("quiet") = N2 of the public hypnogram, clean (artefacts.v1 with
the 1-s join guard) and more than 1 s from any event of the four teachers on the original night
(results/detections_clean). Seeds 11,000,000 + 1000 v + k (k = night index), distinct from every
other injection in the repository.
"""
import argparse
import shutil

import numpy as np
import pandas as pd

import data
from spindlebench import paths, provenance
from spindlebench.data.cache import labels_dir, signal_path
from spindlebench.synth.generator import Params, inject, stage_mask

SNR_GRID = np.array([-3.0, 0.0, 3.0, 6.0, 9.0, 12.0])
CACHE_INJ = paths.CACHE.parent / "cache_student_inj"
OUT = data.STUDY / "injection"
SEED_SIG, SEED_GRID = 11_000_000, 12_000_000


def busy_mask(row, n, sf, margin):
    """True within margin s of any teacher event on the original night"""
    m = np.zeros(n, bool)
    for t in data.TEACHERS:
        on, du = data.load_teacher_events(row.dataset, row.rec, t, row.default_channel)
        for o, d in zip(on, np.nan_to_num(du)):
            m[max(int((o - margin) * sf), 0):min(int((o + d + margin) * sf) + 1, n)] = True
    return m


def clean_native(row, n, sf):
    a = pd.read_csv(labels_dir() / row.dataset / f"{row.rec}__artefact.csv")
    ok = np.ones(n, bool)
    for s, e in a.loc[a.artefact == 1, ["start", "end"]].to_numpy():
        ok[max(int(round((s - data.JOIN_GUARD) * sf)), 0):int(round((e + data.JOIN_GUARD) * sf))] = False
    return ok


def eligible(row, n, sf, p):
    return stage_mask(row.dataset, row.rec, n, sf) & clean_native(row, n, sf) & ~busy_mask(row, n, sf, p.quiet_margin)


def to_grid(mask, sf):
    """native-rate mask -> 100 Hz grid (sample i covers native index floor(i * sf / 100))"""
    n = int(np.ceil(len(mask) * data.FS / sf))
    return mask[np.minimum((np.arange(n) * sf / data.FS).astype(int), len(mask) - 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", type=int, default=4)
    a = ap.parse_args()
    provenance.track(__file__, protocols=["artefacts.v1"])
    sp = data.splits()
    val = sp[sp.role == "val"]
    m = data.manifest()
    m = m.merge(val[["dataset", "rec"]], on=["dataset", "rec"]).sort_values(["dataset", "rec"]).reset_index(drop=True)
    p = Params()
    for v in range(a.variants):
        cdir, bdir = CACHE_INJ / f"v{v}", OUT / f"v{v}"
        rows = []
        for k, r in enumerate(m.itertuples()):
            data.guard(r.dataset)
            z = np.load(signal_path(r.path, paths.CACHE / "manifest.csv"))
            ch = [str(c) for c in z["ch_names"]]
            x = z["data"][ch.index(r.default_channel)].astype(float)
            sf = float(z["sf"])
            elig = eligible(r, len(x), sf, p)
            rng = np.random.default_rng(SEED_GRID + 1000 * v + k)
            grid = np.concatenate([rng.permutation(SNR_GRID) for _ in range(2000)])
            xi, ev = inject(x, sf, elig, p, seed=SEED_SIG + 1000 * v + k, snr_db=grid)
            f = cdir / "signals" / r.dataset / f"{r.rec}.npz"
            f.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(f, data=xi[None].astype(np.float32), sf=sf, ch_names=np.array([r.default_channel]))
            lab = cdir / "labels" / r.dataset
            lab.mkdir(parents=True, exist_ok=True)
            for suffix in ("hypno", "artefact"):
                shutil.copyfile(labels_dir() / r.dataset / f"{r.rec}__{suffix}.csv", lab / f"{r.rec}__{suffix}.csv")
            e = bdir / "events" / r.dataset / f"{r.rec}.csv"
            e.parent.mkdir(parents=True, exist_ok=True)
            ev.to_csv(e, index=False)
            q = bdir / "quiet" / r.dataset / f"{r.rec}.npy"
            q.parent.mkdir(parents=True, exist_ok=True)
            np.save(q, np.packbits(to_grid(elig, sf)))
            row = r._asdict()
            row.pop("Index")
            row.update(path=f.relative_to(cdir).as_posix(), n_channels=1)
            rows.append(row)
            print(f"v{v} {r.dataset:9s} {r.rec:10s} quiet {elig.sum() / sf / 60:6.1f} min, injected {len(ev):4d}",
                  flush=True)
        pd.DataFrame(rows)[m.columns].to_csv(cdir / "manifest.csv", index=False)


if __name__ == "__main__":
    main()
