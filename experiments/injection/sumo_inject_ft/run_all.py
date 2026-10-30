"""
Step 3: every training job as a parallel single-thread process (resumable: finished jobs are skipped).

    detectors/sumo/.venv-sumov2/Scripts/python.exe experiments/injection/sumo_inject_ft/run_all.py [--workers 11] [--seeds 0,1,2]

NAP: arms real / self / inj x 5 folds x seeds. DREAMS: arms real / inj x 8 folds x seeds.
"""
import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import lib as L
from train import det_id

PLAN = {"nap": (["inj", "self", "real"], 5), "dreams_spindles": (["inj", "real"], 8)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=11)
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--datasets", default="nap,dreams_spindles")
    a = p.parse_args()
    jobs = []
    for seed in [int(s) for s in a.seeds.split(",")]:
        for ds in a.datasets.split(","):
            arms, nf = PLAN[ds]
            for arm in arms:
                for fold in range(nf):
                    if not (L.WORK / "models" / det_id(arm, seed) / f"{ds}_fold{fold}.json").exists():
                        jobs.append((ds, arm, seed, fold))
    logs = L.WORK / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    print(len(jobs), "jobs", flush=True)
    t0 = time.time()

    def run(job):
        ds, arm, seed, fold = job
        name = f"{det_id(arm, seed)}_{ds}_fold{fold}"
        with open(logs / f"{name}.log", "w") as fh:
            r = subprocess.run([sys.executable, str(L.HERE / "train.py"), "--dataset", ds, "--fold", str(fold),
                                "--arm", arm, "--seed", str(seed), "--threads", "1"],
                               stdout=fh, stderr=subprocess.STDOUT, cwd=str(L.HERE))
        print(f"{name}: exit {r.returncode} at {(time.time() - t0) / 60:.1f} min", flush=True)
        return r.returncode

    with ThreadPoolExecutor(a.workers) as ex:
        codes = list(ex.map(run, jobs))
    print(f"all done in {(time.time() - t0) / 60:.1f} min, failures: {sum(c != 0 for c in codes)}")


if __name__ == "__main__":
    main()
