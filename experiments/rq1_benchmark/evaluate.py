"""
Harmonised evaluation of every detector in results/detections/ (RQ1) under every
matching protocol (RQ2).

    uv run python experiments/rq1_benchmark/evaluate.py [--detectors a,b] [--datasets nap,dreams_spindles]
                                                         [--out results/benchmark]

The protocol itself is spindlebench.protocols.benchmark (benchmark.v1).

For every (dataset, recording, detector, event type) on the recording's default channel:
  reference   each annotator; on DREAMS also the union and intersection of the two
              experts (onsets within 1 s), scored on the window both experts covered
  scope       all  /  N2 (event midpoint inside an expert N2 epoch)
  protocol    rule x tolerance:
                onset   |det onset - ref onset|   <= 0.25 / 0.5 / 1.0 / 1.5 s
                center  |det centre - ref centre| <= 0.25 / 0.5 / 1.0 / 1.5 s
                        (ref centre = the mark itself when the label has no duration)
                iou     IoU >= 0.1 / 0.2 / 0.3 / 0.5   (only where labels have durations)
  matching    one-to-one, maximum number of pairs (Hungarian)
Writes per_recording.csv (long format, tp/fp/fn) and summary.csv (pooled P/R/F1 and
mean-of-recordings F1), and prints the main-protocol table:
  DREAMS: onset 0.5 s vs expert 1;  NAP: centre 0.5 s vs the expert (marks sit at the
  spindle peak, see results/nap/annotation_alignment.png); recordings excluded in
  Mei et al. 2017 are left out of the NAP summary.
"""

import argparse
from pathlib import Path

import pandas as pd

from spindlebench import paths
from spindlebench.data.cache import manifest as load_manifest
from spindlebench.metrics.events import summarise
from spindlebench.protocols import benchmark
from spindlebench.protocols.benchmark import MAIN
from spindlebench.registry import IN_SAMPLE, USES_EXPERT_STAGING
from spindlebench import provenance  # noqa: E402

provenance.track(__file__, protocols=[benchmark.VERSION])

STORE = paths.RESULTS / "detections"  # --store overrides (e.g. results/rq4/detections)
OUT = paths.RESULTS / "benchmark"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detectors", default=None)
    ap.add_argument("--datasets", default="dreams_spindles,dreams_kc,nap")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--store", default=str(STORE), help="detections root")
    args = ap.parse_args()
    if args.store != ap.get_default("store") and args.out == ap.get_default("out"):
        ap.error("--store other than the default needs its own --out (it would overwrite the benchmark)")
    store = Path(args.store)
    manifest = load_manifest()
    datasets = args.datasets.split(",")
    if args.detectors:
        detectors = args.detectors.split(",")
    else:
        detectors = sorted({p.name for ds in datasets for p in (store / ds).glob("*") if p.is_dir()})
    per = benchmark.evaluate(manifest, detectors, datasets, store)
    if per.empty:
        print("no detections found for", detectors, datasets)
        return
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = "" if args.detectors is None and args.datasets == ap.get_default("datasets") else "_partial"
    per.to_csv(out / f"per_recording{tag}.csv", index=False)
    keys = ["dataset", "event", "detector", "reference", "scope", "rule", "tol"]
    summ = summarise(per[~per.excluded], keys)
    summ.to_csv(out / f"summary{tag}.csv", index=False)
    pd.set_option("display.width", 220, "display.precision", 3, "display.max_rows", 200)
    for ds, (ref, rule, tol) in MAIN.items():
        s = summ.query("dataset == @ds and reference == @ref and rule == @rule and tol == @tol").copy()
        # flag rows that are not comparable with the stage-free, out-of-sample field
        s["detector"] = [d + (" [expert staging]" if d in USES_EXPERT_STAGING else "")
                         + (" [in-sample]" if d in IN_SAMPLE.get(ds, ()) else "") for d in s.detector]
        if len(s):
            print(f"\n=== {ds}: main protocol ({rule} ±{tol}, vs {ref}) ===")
            print(s.drop(columns=["dataset", "reference", "rule", "tol"])
                  .sort_values(["event", "scope", "F1_pooled"], ascending=[True, True, False]).to_string(index=False))
    print(f"\nwrote {out}/")


if __name__ == "__main__":
    main()
