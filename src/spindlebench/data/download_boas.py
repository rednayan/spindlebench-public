"""
Download a subset of the Bitbrain Open Access Sleep (BOAS) dataset, OpenNeuro ds005555 (CC0):
the PSG files of one night per person, for the first N people by recording number.

    uv run spindlebench download boas [--n 20]     # -> <data>/BOAS/ (~155 MB per night)

BOAS (López-Larraz et al.; doi:10.18112/openneuro.ds005555.v1.1.3): 128 nights, 100 healthy adults,
Micromed PSG with F3 F4 C3 C4 O1 O2 against the left mastoid (M1) at 256 Hz, plus a forehead headband
(not downloaded). Labels: consensus of three AASM scorers, 30-s epochs ("stage_hum" in *_psg_events.tsv).
There are no spindle or K-complex events, so BOAS is used for label-free analyses only.
"""
import argparse
import urllib.request
from io import StringIO

import pandas as pd

from spindlebench import paths

S3 = "https://s3.amazonaws.com/openneuro.org/ds005555"
OUT = paths.raw("boas")
FILES = ["acq-psg_eeg.edf", "acq-psg_events.tsv", "acq-psg_channels.tsv", "acq-psg_eeg.json"]


def get(url):
    with urllib.request.urlopen(url, timeout=120) as r:
        return r.read()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    a = ap.parse_args(argv)
    OUT.mkdir(exist_ok=True)
    p = pd.read_csv(StringIO(get(f"{S3}/participants.tsv").decode()), sep="\t")
    p["num"] = p.participant_id.str.removeprefix("sub-").astype(int)
    pick = p.sort_values("num").drop_duplicates("pid").head(a.n)  # one night per person
    pick.to_csv(OUT / "participants_used.tsv", sep="\t", index=False)
    for sub in pick.participant_id:
        for f in FILES:
            dst = OUT / sub / f"{sub}_task-Sleep_{f}"
            if dst.exists():
                continue
            dst.parent.mkdir(exist_ok=True)
            dst.write_bytes(get(f"{S3}/{sub}/eeg/{sub}_task-Sleep_{f}"))
        print(sub, flush=True)
    print(f"{len(pick)} nights in {OUT}/ (ages {pick.age.min()}-{pick.age.max()}, "
          f"{(pick.sex == 'F').sum()} F / {(pick.sex == 'M').sum()} M)")


if __name__ == "__main__":
    main()
