import importlib


def test_environment_overrides(monkeypatch, tmp_path):
    import spindlebench.paths as paths

    monkeypatch.setenv("SPINDLEBENCH_DATA", str(tmp_path / "raw"))
    monkeypatch.setenv("SPINDLEBENCH_RESULTS", str(tmp_path / "out"))
    p = importlib.reload(paths)
    try:
        assert p.DATA == (tmp_path / "raw").resolve()
        assert p.RESULTS == (tmp_path / "out").resolve()
        assert p.raw("nap") == p.DATA / "NapEEG"
        assert (p.ROOT / "pyproject.toml").exists()
    finally:
        monkeypatch.delenv("SPINDLEBENCH_DATA")
        monkeypatch.delenv("SPINDLEBENCH_RESULTS")
        importlib.reload(paths)
