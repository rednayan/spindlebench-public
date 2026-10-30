"""
Runs every training job (5 folds x rungs x seeds) as parallel single-thread processes, then nothing
else (run assemble.py afterwards).

    detectors/sumo/.venv-sumo/Scripts/python.exe experiments/rq4_adaptation/sumo_finetune/run_all.py [--workers 12] [--seeds 0,1,2]

Jobs whose fold<k>.json already exists are skipped, so the script can be resumed.
"""
import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import ftlib as L
from train import det_id

RUNGS = ["scratch", "ft_all", "ft_n6", "ft_n3", "ft_n1"]
SEEDED = ["scratch", "ft_all"]  # rungs repeated with extra seeds


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=11)
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--rungs", default=",".join(RUNGS))
    a = p.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    jobs = []
    for seed in seeds:
        for rung in a.rungs.split(","):
            if seed and rung not in SEEDED:
                continue
            for fold in range(5):
                if not (L.WORK / "models" / det_id(rung, seed) / f"fold{fold}.json").exists():
                    jobs.append((rung, seed, fold, None, ""))
    for seed in seeds:  # lr control: fine-tune at SUMO's own (= scratch) lr 5e-3, same seeds as scratch
        for fold in range(5):
            if not (L.WORK / "models" / det_id("ft_all", seed, "_lr5e-3") / f"fold{fold}.json").exists():
                jobs.append(("ft_all", seed, fold, 5e-3, "_lr5e-3"))
    logs = L.WORK / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    print(len(jobs), "jobs", flush=True)
    t0 = time.time()

    def run(job):
        rung, seed, fold, lr, tag = job
        name = f"{det_id(rung, seed, tag)}_fold{fold}"
        extra = ["--lr", str(lr), "--tag", tag] if lr else []
        with open(logs / f"{name}.log", "w") as fh:
            r = subprocess.run([sys.executable, str(L.HERE / "train.py"), "--fold", str(fold), "--rung", rung,
                                "--seed", str(seed), "--threads", "1"] + extra, stdout=fh, stderr=subprocess.STDOUT,
                               cwd=str(L.HERE))
        print(f"{name}: exit {r.returncode} at {(time.time() - t0) / 60:.1f} min", flush=True)
        return r.returncode

    with ThreadPoolExecutor(a.workers) as ex:
        codes = list(ex.map(run, jobs))
    print(f"all done in {(time.time() - t0) / 60:.1f} min, failures: {sum(c != 0 for c in codes)}")


if __name__ == "__main__":
    main()
