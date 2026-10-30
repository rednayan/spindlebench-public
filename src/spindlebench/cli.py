"""
Command-line entry point.

    uv run spindlebench paths                      where data, cache and results are read and written
    uv run spindlebench download nap|boas|hmc|sleepedf [options]
    uv run spindlebench cache [--labels-only | --boas-only]

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


def main(argv=None):
    ap = argparse.ArgumentParser(prog="spindlebench", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("paths", help="show where data, cache and results are read and written").set_defaults(fn=cmd_paths)
    d = sub.add_parser("download", help="download a dataset (checksums verified where the source provides them)")
    d.add_argument("dataset", choices=["nap", "boas", "hmc", "sleepedf"])
    d.add_argument("rest", nargs=argparse.REMAINDER, help="options passed to the downloader")
    d.set_defaults(fn=cmd_download)
    c = sub.add_parser("cache", help="build the standardised signal and label cache from the raw datasets")
    c.add_argument("rest", nargs=argparse.REMAINDER, help="--labels-only or --boas-only")
    c.set_defaults(fn=cmd_cache)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
