"""
Where data, caches and results live. Every script gets its paths from here.

    SPINDLEBENCH_ROOT     repository root (default: the checkout containing this file or the working
                          directory; else the working directory itself)
    SPINDLEBENCH_DATA     raw datasets (default: <root>)
    SPINDLEBENCH_CACHE    standardised signal/label cache (default: <root>/cache)
    SPINDLEBENCH_RESULTS  results tree (default: <root>/results)
Derived caches (injected copies, montage variant) sit next to the cache: <cache>/../cache_inj, ../cache_montage.

Raw datasets sit in <data>/<folder> with the folder names below (as the downloaders write them).
"""
import os
from pathlib import Path


def _find_root():
    if os.environ.get("SPINDLEBENCH_ROOT"):
        return Path(os.environ["SPINDLEBENCH_ROOT"]).resolve()
    for start in (Path(__file__).resolve().parent, Path.cwd().resolve()):
        for d in (start, *start.parents):
            p = d / "pyproject.toml"
            if p.exists() and 'name = "spindlebench"' in p.read_text(encoding="utf-8"):
                return d
    return Path.cwd().resolve()  # installed outside a checkout: data next to where it is run


def _env(name, default):
    v = os.environ.get(name)
    return Path(v).resolve() if v else default


ROOT = _find_root()
DATA = _env("SPINDLEBENCH_DATA", ROOT)
CACHE = _env("SPINDLEBENCH_CACHE", ROOT / "cache")
RESULTS = _env("SPINDLEBENCH_RESULTS", ROOT / "results")
CACHE_INJ = CACHE.parent / "cache_inj"
CACHE_MONTAGE = CACHE.parent / "cache_montage"
THIRD_PARTY = ROOT / "third_party"

RAW = {
    "dreams_spindles": DATA / "DatabaseSpindles",
    "dreams_kc": DATA / "DatabaseKcomplexes",
    "nap": DATA / "NapEEG",
    "boas": DATA / "BOAS",
    "hmc": DATA / "HMC",
    "sleepedf": DATA / "SLEEPEDF",
    "mass_ss2_annotations": DATA / "MASS_data" / "doi-10.5683-sp3-y889cs",
}


def raw(name):
    return RAW[name]
