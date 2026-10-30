"""Adapter conformance: manifests, the detection file contract, provenance records."""
import importlib.util
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from spindlebench import detections, paths
from spindlebench.provenance import write_provenance

# ids in results/detections that no adapter writes: the automatic detections shipped with DREAMS
DATASET_PROVIDED = {"dreams_auto"}


def load_common():
    spec = importlib.util.spec_from_file_location("common", paths.ROOT / "detectors" / "common.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_every_adapter_has_a_valid_manifest():
    ms = detections.manifests()
    adapters = {p.parent.name for p in (paths.ROOT / "detectors").glob("*/run.py")}
    assert adapters == set(ms)
    seen = {}
    for name, m in ms.items():
        assert detections.MANIFEST_KEYS <= set(m), name
        assert m["name"] == name
        assert m["kind"] in {"rule-based", "learned"}
        assert set(m["events"]) <= detections.EVENTS
        assert m["adapter_licence"], name
        for i in m["ids"]:
            assert i not in seen, f"{i} claimed by {seen.get(i)} and {name}"
            seen[i] = name
        lock = m["environment"].get("lock")
        for part in (lock or "").split(" ; "):
            assert part and (paths.ROOT / part).exists(), f"{name}: lock file {part!r} missing"


def test_adapter_writer_uses_the_contract_columns():
    assert load_common().COLUMNS == detections.COLUMNS


def test_every_stored_detector_id_has_an_owner():
    ids = {d.name for ds in (paths.RESULTS / "detections").iterdir() if ds.is_dir() for d in ds.iterdir() if d.is_dir()}
    unowned = {i for i in ids if detections.owner(i) is None} - DATASET_PROVIDED
    assert not unowned


def test_problems_flags_contract_violations():
    ok = pd.DataFrame([[1.0, 0.8, "C3", "spindle", ""]], columns=detections.COLUMNS)
    assert detections.problems(ok, duration=10, channels=["C3"]) == []
    bad = ok.assign(event="spindles", onset=-1.0)
    assert len(detections.problems(bad)) == 2
    assert detections.problems(ok, duration=0.5)
    assert detections.problems(ok.drop(columns="score"))


@pytest.mark.data
def test_stored_detections_follow_the_contract():
    man = paths.CACHE / "manifest.csv"
    if not man.exists():
        pytest.skip("no cache")
    m = pd.read_csv(man).set_index(["dataset", "rec"])
    bad = []
    n = 0
    for f in sorted((paths.RESULTS / "detections").glob("*/*/*.csv")):
        dataset, rec = f.parent.parent.name, f.stem
        if (dataset, rec) not in m.index:
            bad.append(f"{f}: unknown recording")
            continue
        row = m.loc[(dataset, rec)]
        n += 1
        d = detections.read(f)
        if f.parent.name in DATASET_PROVIDED and "score" not in d:  # exported from the DREAMS files: no score column
            d["score"] = None
        for p in detections.problems(d, duration=float(row.duration)):
            bad.append(f"{f.relative_to(paths.RESULTS)}: {p}")
    assert n > 1000
    assert not bad, "\n".join(bad[:20])


def test_provenance_records(tmp_path):
    rec = write_provenance(tmp_path / "run", config={"draws": 10}, protocols=["nap_points.v1"])
    saved = json.loads((tmp_path / "run" / "provenance.json").read_text(encoding="utf-8"))
    assert saved["protocols"] == ["nap_points.v1"] and saved["config"] == {"draws": 10}
    assert rec["git"]["commit"] and "numpy" in rec["packages"]

    common = load_common()
    args = SimpleNamespace(out=str(tmp_path / "results" / "detections"), datasets="nap", channels=None, recs=None)
    common.write(args, SimpleNamespace(dataset="nap", rec="suj11_d1"), "dummy",
                 [dict(onset=1.0, duration=0.5, channel="C3", event="spindle", score=None)])
    p = tmp_path / "results" / "provenance" / "detections" / "nap" / "dummy.json"
    assert json.loads(p.read_text(encoding="utf-8"))["config"]["datasets"] == "nap"
    assert (tmp_path / "results" / "detections" / "nap" / "dummy" / "suj11_d1.csv").exists()


def test_provenance_hides_local_paths(tmp_path):
    from spindlebench.provenance import scrub

    out = scrub({"a": str(paths.ROOT / "results" / "x"), "b": [paths.ROOT / "cache"], "c": "plain", "n": 3},
                paths.ROOT)
    assert out == {"a": "<root>/results/x", "b": ["<root>/cache"], "c": "plain", "n": 3}
    rec = write_provenance(tmp_path / "run")
    assert "Users" not in json.dumps(rec)
