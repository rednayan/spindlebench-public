"""
Command-line entry point.

    uv run spindlebench paths                      where data, cache and results are read and written
    uv run spindlebench download nap|boas|hmc|sleepedf|cinc2018 [options]
    uv run spindlebench cache [--labels-only | --boas-only | --only hmc,cinc2018]
    uv run spindlebench artefacts [--datasets ...] [--recs ...]   shared artefact mask (artefacts.v1)

Experiments are run as scripts: see experiments/README.md.
"""
import argparse

from spindlebench import __version__, paths


def cmd_paths(_):
    print(f"spindlebench {__version__}")
    for name in ["ROOT", "DATA", "CACHE", "RESULTS"]:
        p = getattr(paths, name)
        print(f"  {name:8s} {p}  {'(exists)' if p.exists() else '(missing)'}")
    for name, p in paths.RAW.items():
        print(f"  raw/{name:22s} {'ok' if p.exists() else 'missing'}  {p}")


def cmd_download(a):
    if a.dataset == "nap":
        from spindlebench.data import download_nap as m
        m.main(a.rest)
    elif a.dataset == "boas":
        from spindlebench.data import download_boas as m
        m.main(a.rest)
    else:
        from spindlebench.data import download_physionet as m
        m.main([a.dataset, *a.rest])


def cmd_cache(a):
    from spindlebench.data import export_cache
    export_cache.main(a.rest)


def cmd_artefacts(a):
    from spindlebench.data import artefacts
    artefacts.main(a.rest)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="spindlebench", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("paths", help="show where data, cache and results are read and written").set_defaults(fn=cmd_paths)
    d = sub.add_parser("download", help="download a dataset (checksums verified where the source provides them)")
    d.add_argument("dataset", choices=["nap", "boas", "hmc", "sleepedf", "cinc2018"])
    d.add_argument("rest", nargs=argparse.REMAINDER, help="options passed to the downloader")
    d.set_defaults(fn=cmd_download)
    c = sub.add_parser("cache", help="build the standardised signal and label cache from the raw datasets")
    c.add_argument("rest", nargs=argparse.REMAINDER, help="--labels-only, --boas-only or --only hmc,cinc2018")
    c.set_defaults(fn=cmd_cache)
    r = sub.add_parser("artefacts", help="shared artefact mask per recording (YASA art_detect, std, |z| > 3)")
    r.add_argument("rest", nargs=argparse.REMAINDER, help="--datasets ..., --recs ...")
    r.set_defaults(fn=cmd_artefacts)
    a, extra = ap.parse_known_args(argv)  # REMAINDER does not take a leading --option: pass it on
    if extra and a.cmd == "paths":
        ap.error(f"unrecognized arguments: {' '.join(extra)}")
    a.rest = [*extra, *getattr(a, "rest", [])]
    a.fn(a)


if __name__ == "__main__":
    main()
