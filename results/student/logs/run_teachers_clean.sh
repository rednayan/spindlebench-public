#!/usr/bin/env bash
# Step 3 of notes/plans/student_architecture.md: the four rule teachers on clean (artefact-masked) windows
# (--clean, artefacts.v1) on HMC + CinC 2018 + DREAMS + NAP, default channel. Only recordings without
# an output file are run, so the script can be restarted. One process per shard; logs in results/student/logs/.
cd "$(dirname "$0")/../../.."
L=results/student/logs
missing() { # dataset detector_id -> recordings with no output yet
  awk -F, -v d="$1" '$1==d{print $2}' cache/manifest.csv | while read r; do
    [ -f results/detections_clean/$1/$2/$r.csv ] || echo "$r"; done; }
run() { # name id n_shards cmd...
  local name=$1 id=$2 n=$3; shift 3
  for ds in hmc cinc2018 dreams_spindles nap; do
    list=$(missing $ds $id)
    for i in $(seq 0 $((n-1))); do
      r=$(echo "$list" | awk -v n="$n" -v i="$i" 'NF && (NR-1)%n==i' | paste -sd, -)
      [ -z "$r" ] && continue
      "$@" --datasets $ds --recs "$r" --clean > $L/clean_${name}_${ds}_$i.log 2>&1 &
    done
  done; }
run wonambi wonambi_ucsd 4 detectors/wonambi/.venv/Scripts/python detectors/wonambi/run.py --methods UCSD
run a7      a7           1 uv run python detectors/a7/run.py
run yasa    yasa_default 2 uv run python detectors/yasa/run.py
run luna    luna_spindles 1 detectors/luna/.venv/Scripts/python detectors/luna/run.py
wait
echo ALL DONE
