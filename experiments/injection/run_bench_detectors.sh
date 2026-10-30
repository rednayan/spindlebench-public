#!/usr/bin/env bash
# Evaluation benchmark, step 2: run the benchmark detectors on the injected caches.
#   bash experiments/injection/run_bench_detectors.sh [variants...]   (default: 0 1)
# Each adapter runs unchanged; only --manifest and --out differ from the main benchmark.
set -u
cd "$(dirname "$0")/../.."
unset PYTHONHOME PYTHONPATH
export PYTHONIOENCODING=utf-8
DS=dreams_spindles,nap
for v in "${@:-0 1}"; do
  for v in $v; do
    M=cache_inj/v$v/manifest.csv
    O=results/injection/bench/v$v/detections
    LOG=results/injection/bench/v$v/run.log
    mkdir -p "$O"
    echo "== v$v $(date +%T)" | tee -a "$LOG"
    .venv/Scripts/python.exe detectors/yasa/run.py --manifest $M --out $O --datasets $DS >> "$LOG" 2>&1; echo "yasa $? $(date +%T)" | tee -a "$LOG"
    .venv/Scripts/python.exe detectors/moelle/run.py --manifest $M --out $O --datasets $DS >> "$LOG" 2>&1; echo "moelle $? $(date +%T)" | tee -a "$LOG"
    .venv/Scripts/python.exe detectors/a7/run.py --manifest $M --out $O --datasets $DS >> "$LOG" 2>&1; echo "a7 $? $(date +%T)" | tee -a "$LOG"
    .venv/Scripts/python.exe detectors/sumo/run.py --model sumo,sumov2 --manifest $M --out $O --datasets $DS >> "$LOG" 2>&1; echo "sumo $? $(date +%T)" | tee -a "$LOG"
    detectors/openspindlenet/.venv/Scripts/python.exe detectors/openspindlenet/run.py --manifest $M --out $O --datasets $DS >> "$LOG" 2>&1; echo "osn $? $(date +%T)" | tee -a "$LOG"
  done
done
