"""Every experiment and adapter script parses and uses no working-directory-relative data paths."""
import ast
import re

import pytest

from spindlebench import paths

def _ours(p):
    """skip detector environments (.venv, .venv-sumo, ...) and upstream clones (detectors/<name>/src)"""
    parts = p.relative_to(paths.ROOT).parts
    in_env = any(x.startswith(".venv") or x == "__pycache__" for x in parts)
    return not in_env and parts[:3] not in (("detectors", "sumo", "src"), ("detectors", "s5", "src"))


def _walk(d):
    for p in d.iterdir():
        if p.is_dir() and _ours(p):
            yield from _walk(p)
        elif p.suffix == ".py" and _ours(p):
            yield p


SCRIPTS = sorted(p for d in ["experiments", "detectors", "scripts"] for p in _walk(paths.ROOT / d))
RELATIVE = re.compile(
    r"""["'](cache|results|cache_inj|cache_montage|NapEEG|DatabaseSpindles|DatabaseKcomplexes|BOAS)/""")


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: str(p.relative_to(paths.ROOT)))
def test_script_parses(path):
    ast.parse(path.read_text(encoding="utf-8"))


# scripts whose path-like strings are glob rules over repository paths, not files they open
PATTERN_RULES = {"scripts/export_public.py"}


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: str(p.relative_to(paths.ROOT)))
def test_no_working_directory_paths(path):
    if path.relative_to(paths.ROOT).as_posix() in PATTERN_RULES:
        pytest.skip("export rules are glob patterns")
    code = ast.parse(path.read_text(encoding="utf-8"))
    doc = ast.get_docstring(code) or ""
    hits = [n.value for n in ast.walk(code) if isinstance(n, ast.Constant) and isinstance(n.value, str)
            and RELATIVE.match(repr(n.value)) and n.value not in doc]
    assert not hits, hits


def test_no_scripts_left_at_the_top_level():
    assert not list(paths.ROOT.glob("*.py"))


@pytest.mark.parametrize("path", [p for p in SCRIPTS if p.relative_to(paths.ROOT).parts[0] == "experiments"
                                  and re.search(r"^(from|import) spindlebench", p.read_text(encoding="utf-8"), re.M)],
                         ids=lambda p: str(p.relative_to(paths.ROOT)))
def test_experiments_record_provenance(path):
    assert "provenance.track(__file__" in path.read_text(encoding="utf-8")
