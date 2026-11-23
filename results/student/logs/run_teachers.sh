#!/usr/bin/env bash
# Step 1 of notes/plans/student_architecture.md: four rule teachers on HMC + CinC 2018 (default channel C3-M2).
# Shards the 401 recordings; each shard is one process. Logs in results/student/logs/.
cd "$(dirname "$0")/../../.."
L=results/student/logs
recs() { awk -F, -v d="$1" '$1==d{print $2}' cache/manifest.csv; }
shard() { # dataset n_shards i -> comma list
  recs "$1" | awk -v n="$2" -v i="$3" '(NR-1)%n==i' | paste -sd, -; }
run() { # name n_shards cmd...
  local name=$1 n=$2; shift 2
  for ds in hmc cinc2018; do for i in $(seq 0 $((n-1))); do
    r=$(shard $ds $n $i); [ -z "$r" ] && continue
    "$@" --datasets $ds --recs "$r" > $L/${name}_${ds}_$i.log 2>&1 &
  done; done; }
run wonambi 3 detectors/wonambi/.venv/Scripts/python detectors/wonambi/run.py --methods UCSD
run a7      2 uv run python detectors/a7/run.py
run yasa    1 uv run python detectors/yasa/run.py
run luna    1 detectors/luna/.venv/Scripts/python detectors/luna/run.py
wait
echo ALL DONE
