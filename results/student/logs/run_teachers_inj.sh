#!/usr/bin/env bash
# Plan 6.3: the four rule teachers on the injected validation nights (experiments/student/build_val_injection.py),
# --clean on the same artefacts.v1 windows as the original nights. Restartable: only nights without an output run.
#   bash results/student/logs/run_teachers_inj.sh [variants...]      (default 0 1 2 3)
cd "$(dirname "$0")/../../.."
unset PYTHONHOME PYTHONPATH
export PYTHONIOENCODING=utf-8
L=results/student/logs
missing() { # manifest out id -> recs (all datasets) with no output yet, as "dataset rec"
  awk -F, 'NR>1{print $1, $2}' "$1" | while read d r; do [ -f "$2/$d/$3/$r.csv" ] || echo "$d $r"; done; }
run() { # v name id n_shards cmd...
  local v=$1 name=$2 id=$3 n=$4; shift 4
  local M=cache_student_inj/v$v/manifest.csv O=results/student/injection/v$v/detections
  mkdir -p "$O"
  for ds in hmc cinc2018; do
    list=$(missing $M $O $id | awk -v d=$ds '$1==d{print $2}')
    for i in $(seq 0 $((n-1))); do
      r=$(echo "$list" | awk -v n="$n" -v i="$i" 'NF && (NR-1)%n==i' | paste -sd, -)
      [ -z "$r" ] && continue
      SPINDLEBENCH_CACHE=cache_student_inj/v$v "$@" --manifest $M --out $O --datasets $ds --recs "$r" --clean \
        > $L/inj_v${v}_${name}_${ds}_$i.log 2>&1 &
    done
  done; }
for v in ${@:-0 1 2 3}; do
  run $v wonambi wonambi_ucsd 1 detectors/wonambi/.venv/Scripts/python detectors/wonambi/run.py --methods UCSD
  run $v a7      a7           1 uv run python detectors/a7/run.py
  run $v yasa    yasa_default 1 uv run python detectors/yasa/run.py
  run $v luna    luna_spindles 1 detectors/luna/.venv/Scripts/python detectors/luna/run.py
done
wait
echo ALL DONE
