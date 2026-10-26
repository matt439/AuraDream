"""`tools/run_smoke.py` reads `AGENTS.md`'s port table as the smokes need it.

`test_docs_maps.py` holds every smoke file to a row on its port; this holds
the runner's reading of those rows - the checkpoint each fake claims, its
step, the smokes it runs - so a row the runner would misread fails here
rather than as a smoke against the wrong fake.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _runner():
    spec = importlib.util.spec_from_file_location("run_smoke", ROOT / "tools" / "run_smoke.py")
    module = importlib.util.module_from_spec(spec)
    # Registered first: a dataclass reads its own module from sys.modules.
    sys.modules.setdefault("run_smoke", module)
    spec.loader.exec_module(module)
    return module


def _rows():
    return _runner().parse_table((ROOT / "AGENTS.md").read_text(encoding="utf-8"))


def test_every_smoke_is_on_exactly_one_row():
    named = [smoke for row in _rows() for smoke in row.smokes]
    files = sorted(p.stem for p in (ROOT / "tools").glob("smoke*.json"))
    assert sorted(named) == files


def test_every_row_claims_a_full_checkpoint_id():
    for row in _rows():
        assert "/" in row.model and " " not in row.model and "`" not in row.model, row


def test_short_names_and_steps_resolve():
    rows = {smoke: row for row in _rows() for smoke in row.smokes}
    assert rows["smoke-flux"].model == "black-forest-labs/FLUX.1-Krea-dev"
    assert rows["smoke-animagine"].model == "cagliostrolab/animagine-xl-4.0"
    assert rows["smoke-anima-aesthetic"].model == "circlestone-labs/Anima@aesthetic-v1.1"
    assert rows["smoke"].model == _runner().DEFAULT_MODEL
    assert rows["smoke-paging"].model == _runner().DEFAULT_MODEL
    assert rows["smoke-results"].step == 0.3 and rows["smoke-flux"].step is None
    assert rows["smoke-analysis"].smokes == [
        "smoke-analysis", "smoke-rewrite", "smoke-comparison", "smoke-comparison-batch",
    ]
    assert rows["smoke-model-gallery"].smokes == ["smoke-model-gallery"]
    assert rows["smoke-model-gallery"].separate


def test_the_fake_is_started_on_its_own_gallery():
    row = _runner().Row(7862, "black-forest-labs/FLUX.1-Krea-dev", 0.3, ["smoke-flux"])
    args = row.fake_args(7962, Path("scratch"))
    assert args == [
        "--port", "7962", "--model", "black-forest-labs/FLUX.1-Krea-dev",
        "--data", "scratch", "--step", "0.3",
    ]


def test_smokes_keep_separate_screenshots_after_a_failure(monkeypatch, tmp_path):
    runner = _runner()
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    scratch = tmp_path / "gallery"
    scratch.mkdir()
    make_temp = runner.tempfile.mkdtemp
    monkeypatch.setattr(runner.tempfile, "mkdtemp", lambda **kwargs: (
        str(scratch) if kwargs.get("dir") is None else make_temp(**kwargs)
    ))
    monkeypatch.setattr(runner, "_port_free", lambda port: True)
    monkeypatch.setattr(runner, "_wait_ready", lambda fake, port: True)

    class Fake:
        stopped = False

        def terminate(self):
            self.stopped = True

        def wait(self, **kwargs):
            return 0

    fake = Fake()
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *args, **kwargs: fake)
    destinations = []

    def browser(command, **kwargs):
        destination = Path(kwargs["env"]["LOCAL_DIFFUSION_SCREENSHOT_DIR"])
        destinations.append(destination)
        (destination / "same-name.png").write_bytes(b"retained")
        assert kwargs["env"]["LOCAL_DIFFUSION_PORT_OFFSET"] == "100"
        return subprocess.CompletedProcess(command, len(destinations) - 1, "browser output", "")

    monkeypatch.setattr(runner.subprocess, "run", browser)
    row = runner.Row(7862, "black-forest-labs/FLUX.1-Krea-dev", None, ["first", "second"])
    assert runner.run_row(row, row.smokes, 100, stream=False, artifacts=artifacts) == {"first": 0, "second": 1}
    assert fake.stopped
    assert len(set(destinations)) == 2
    assert all(path.parent == artifacts and (path / "same-name.png").exists() for path in destinations)
    assert (scratch / "fake-server.log").exists()
