"""`tools/run_smoke.py` reads `AGENTS.md`'s port table as the smokes need it.

`test_docs_maps.py` holds every smoke file to a row on its port; this holds
the runner's reading of those rows - the checkpoint each fake claims, its
step, the smokes it runs - so a row the runner would misread fails here
rather than as a smoke against the wrong fake.
"""
from __future__ import annotations

import importlib.util
import io
import json
import subprocess
import sys
from pathlib import Path

import pytest

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

        def poll(self):
            return 0 if self.stopped else None

        def terminate(self):
            self.stopped = True

        def wait(self, **kwargs):
            return 0

    fake = Fake()
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *args, **kwargs: fake)
    destinations = []

    def browser(command, env, log_path):
        destination = Path(env["AURADREAM_SCREENSHOT_DIR"])
        destinations.append(destination)
        (destination / "same-name.png").write_bytes(b"retained")
        assert env["AURADREAM_PORT_OFFSET"] == "100"
        log_path.write_text("browser output", encoding="utf-8")
        return len(destinations) - 1

    monkeypatch.setattr(runner, "_run_browser", browser)
    row = runner.Row(7862, "black-forest-labs/FLUX.1-Krea-dev", None, ["first", "second"])
    assert runner.run_row(row, row.smokes, 100, stream=False, artifacts=artifacts) == {"first": 0, "second": 1}
    assert fake.stopped
    assert len(set(destinations)) == 2
    assert all(path.parent == artifacts and (path / "same-name.png").exists() for path in destinations)
    assert (scratch / "fake-server.log").exists()
    reports = {json.loads((path / "result.json").read_text())["smoke"]: path for path in destinations}
    assert (reports["first"] / "browser.log").read_text() == "browser output"
    assert not (reports["first"] / "fake-server.log").exists()
    assert (reports["second"] / "fake-server.log").exists()


@pytest.fixture
def row_case(monkeypatch, tmp_path):
    from types import SimpleNamespace

    runner = _runner()
    artifacts, scratch = tmp_path / "artifacts", tmp_path / "gallery"
    artifacts.mkdir()
    scratch.mkdir()
    make_temp = runner.tempfile.mkdtemp
    monkeypatch.setattr(runner.tempfile, "mkdtemp", lambda **kwargs: (
        str(scratch) if kwargs.get("dir") is None else make_temp(**kwargs)
    ))
    monkeypatch.setattr(runner, "_port_free", lambda port: True)
    monkeypatch.setattr(runner, "_wait_ready", lambda fake, port: True)

    class Fake:
        stopped = False
        killed = False
        returncode = None
        stubborn = False

        def poll(self):
            return self.returncode

        def terminate(self):
            self.stopped = True

        def kill(self):
            self.killed = True

        def wait(self, timeout=None):
            if self.stubborn and not self.killed:
                raise subprocess.TimeoutExpired("fake", timeout)
            self.returncode = self.returncode or 0
            return self.returncode

    fake = Fake()

    def launch(*args, **kwargs):
        kwargs["stdout"].write("fake diagnostic\n")
        return fake

    monkeypatch.setattr(runner.subprocess, "Popen", launch)

    def browser(command, env, log_path):
        log_path.write_text("browser diagnostic\n", encoding="utf-8")
        return 0

    monkeypatch.setattr(runner, "_run_browser", browser)
    return SimpleNamespace(
        runner=runner, fake=fake, artifacts=artifacts, scratch=scratch,
        row=runner.Row(7862, "black-forest-labs/FLUX.1-Krea-dev", None, ["smoke-flux"]),
    )


def _receipt(case):
    paths = list(case.artifacts.glob("*/result.json"))
    assert len(paths) == 1
    return paths[0].parent, json.loads(paths[0].read_text(encoding="utf-8"))


def _run_case(case):
    return case.runner.run_row(case.row, case.row.smokes, 0, stream=False, artifacts=case.artifacts)


def test_success_retains_output_and_timing_after_gallery_cleanup(row_case):
    assert _run_case(row_case) == {"smoke-flux": 0}
    output, report = _receipt(row_case)
    assert report["result"] == "passed" and report["phase"] == "browser"
    assert report["exit_code"] == 0 and report["duration_seconds"] >= 0
    assert (output / report["browser_log"]).read_text() == "browser diagnostic\n"
    assert report["fake_log"] is None
    assert not row_case.scratch.exists()
    assert row_case.fake.stopped and not row_case.fake.killed


def test_occupied_port_reports_without_starting_or_stopping_a_process(row_case, monkeypatch):
    monkeypatch.setattr(row_case.runner, "_port_free", lambda port: False)
    assert _run_case(row_case) == {"smoke-flux": 2}
    output, report = _receipt(row_case)
    assert report["phase"] == "port" and report["fake_log"] is None
    assert "port 7862 is in use" in report["detail"]
    assert "is in use" in (output / "browser.log").read_text()
    assert not row_case.fake.stopped


def test_failed_fake_readiness_keeps_its_log_and_original_exit_status(row_case, monkeypatch):
    row_case.fake.returncode = 7
    monkeypatch.setattr(row_case.runner, "_wait_ready", lambda fake, port: False)
    assert _run_case(row_case) == {"smoke-flux": 7}
    output, report = _receipt(row_case)
    assert report["phase"] == "fake_start" and report["exit_code"] == 7
    assert (output / report["fake_log"]).read_text() == "fake diagnostic\n"
    assert not row_case.fake.stopped


def test_a_fake_launch_exception_also_writes_a_receipt(row_case, monkeypatch):
    def refuse(*args, **kwargs):
        raise OSError("fake executable unavailable")

    monkeypatch.setattr(row_case.runner.subprocess, "Popen", refuse)
    assert _run_case(row_case) == {"smoke-flux": 2}
    output, report = _receipt(row_case)
    assert report["phase"] == "fake_start"
    assert "fake executable unavailable" in report["detail"]
    assert "fake executable unavailable" in (output / "browser.log").read_text()
    assert (output / "fake-server.log").exists()


def test_a_browser_launch_exception_retains_diagnostics_and_stops_the_fake(row_case, monkeypatch):
    def refuse(*args, **kwargs):
        raise OSError("node executable unavailable")

    monkeypatch.setattr(row_case.runner, "_run_browser", refuse)
    assert _run_case(row_case) == {"smoke-flux": 2}
    output, report = _receipt(row_case)
    assert report["phase"] == "browser_start"
    assert "node executable unavailable" in report["detail"]
    assert "node executable unavailable" in (output / "browser.log").read_text()
    assert (output / "fake-server.log").read_text() == "fake diagnostic\n"
    assert row_case.fake.stopped


@pytest.mark.parametrize("code", [1, 2, 42])
def test_browser_failures_keep_their_exact_exit_status(row_case, monkeypatch, code):
    def fail(command, env, log_path):
        log_path.write_text("failed assertion\n", encoding="utf-8")
        return code

    monkeypatch.setattr(row_case.runner, "_run_browser", fail)
    assert _run_case(row_case) == {"smoke-flux": code}
    output, report = _receipt(row_case)
    assert report["exit_code"] == code and report["result"] == "failed"
    assert (output / "browser.log").read_text() == "failed assertion\n"
    assert (output / "fake-server.log").exists()
    assert row_case.fake.stopped


def test_an_unresponsive_fake_is_killed_and_waited_for(row_case):
    row_case.fake.stubborn = True
    assert _run_case(row_case) == {"smoke-flux": 0}
    assert row_case.fake.stopped and row_case.fake.killed
    assert row_case.fake.returncode == 0


def test_interruption_keeps_partial_output_and_stops_the_fake(row_case, monkeypatch):
    partial = "".join(f"partial output line {index}\n" for index in range(60))

    def interrupt(command, env, log_path):
        log_path.write_text(partial, encoding="utf-8")
        raise KeyboardInterrupt

    monkeypatch.setattr(row_case.runner, "_run_browser", interrupt)
    with pytest.raises(KeyboardInterrupt):
        _run_case(row_case)
    output, report = _receipt(row_case)
    assert report["exit_code"] == 130 and report["phase"] == "interrupted"
    assert (output / "browser.log").read_text() == partial + "\nbrowser interrupted\n"
    assert row_case.fake.stopped


@pytest.mark.parametrize("stubborn", [False, True])
def test_interrupted_browser_gets_cleanup_time_before_the_kill_fallback(monkeypatch, tmp_path, stubborn):
    runner = _runner()

    class Browser:
        pid = 12345
        waits = 0
        signalled = False
        killed = False
        stopped = False

        def poll(self):
            return 0 if self.stopped else None

        def wait(self, timeout=None):
            self.waits += 1
            if self.waits == 1:
                raise KeyboardInterrupt
            if stubborn and not self.killed:
                raise subprocess.TimeoutExpired("browser", timeout)
            self.stopped = True
            return 0

        def send_signal(self, signal):
            self.signalled = True

        def kill(self):
            self.killed = True

    browser = Browser()
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *args, **kwargs: browser)
    tree_commands = []
    monkeypatch.setattr(runner.subprocess, "run", lambda command, **kwargs: tree_commands.append(command))
    with pytest.raises(KeyboardInterrupt):
        runner._run_browser(["node", "fixture"], {}, tmp_path / "browser.log")
    assert browser.signalled and browser.stopped
    assert browser.killed == stubborn
    assert bool(tree_commands) == (stubborn and sys.platform == "win32")
    if tree_commands:
        assert tree_commands == [["taskkill", "/pid", "12345", "/T", "/F"]]


def test_main_writes_a_collectable_summary_and_keeps_failure_status(monkeypatch, tmp_path):
    runner = _runner()

    class Output(io.StringIO):
        def reconfigure(self, **kwargs):
            pass

    monkeypatch.setattr(runner.sys, "stdout", Output())
    monkeypatch.setattr(runner.sys, "argv", ["run_smoke.py", "smoke-flux", "--artifacts", str(tmp_path)])

    def fail(row, smokes, offset, stream, artifacts):
        assert smokes == ["smoke-flux"]
        output = artifacts / "smoke-flux-fixture"
        output.mkdir()
        (output / "result.json").write_text(json.dumps({"smoke": "smoke-flux", "exit_code": 42}))
        return {"smoke-flux": 42}

    monkeypatch.setattr(runner, "run_row", fail)
    assert runner.main() == 1
    summaries = list(tmp_path.glob("*/summary.json"))
    assert len(summaries) == 1
    summary = json.loads(summaries[0].read_text())
    assert summary["duration_seconds"] >= 0
    assert summary["results"] == [{
        "smoke": "smoke-flux", "exit_code": 42, "artifact_dir": "smoke-flux-fixture",
    }]


class _MainOutput(io.StringIO):
    def reconfigure(self, **kwargs):
        pass


def _port_plan_main(monkeypatch, tmp_path, ports, wanted, offset, jobs):
    runner = _runner()
    output = _MainOutput()
    rows = [runner.Row(port, "owner/model", None, [f"smoke-{index}"]) for index, port in enumerate(ports)]
    monkeypatch.setattr(runner, "parse_table", lambda _text: rows)
    monkeypatch.setattr(runner.sys, "stdout", output)
    monkeypatch.setattr(runner.sys, "argv", [
        "run_smoke.py", *[f"smoke-{index}" for index in wanted],
        "--offset", str(offset), "--jobs", str(jobs), "--artifacts", str(tmp_path / "chosen-artifacts"),
    ])
    return runner, output, rows


@pytest.mark.parametrize(("ports", "wanted", "offset", "jobs", "message"), [
    ([7862], [0], -7862, 1, "port 0 is outside 1..65535"),
    ([7862], [0], -7863, 1, "port -1 is outside 1..65535"),
    ([7862], [0], 65536 - 7862, 1, "port 65536 is outside 1..65535"),
    ([7862], [0], 10**100, 1, "is outside 1..65535"),
    ([7862, 7863], [0, 1], 65535 - 7863 - 99, 2, "port 65536 is outside 1..65535"),
    ([7862, 7962], [0, 1], 0, 2, "worker bands overlap on port 7962"),
    ([7862, 8062, 7863], [0, 1, 2], 0, 3, "worker bands overlap on port 8062"),
])
def test_invalid_effective_port_plan_refuses_before_any_startup_effect(
    monkeypatch, tmp_path, ports, wanted, offset, jobs, message,
):
    runner, output, _rows_unused = _port_plan_main(monkeypatch, tmp_path, ports, wanted, offset, jobs)
    attempts = []

    def forbidden(*_args, **_kwargs):
        attempts.append("startup-effect")
        pytest.fail("invalid port plan reached artifact, socket, process or worker startup")

    monkeypatch.setattr(Path, "mkdir", forbidden)
    monkeypatch.setattr(runner.tempfile, "mkdtemp", forbidden)
    monkeypatch.setattr(runner.socket, "socket", forbidden)
    monkeypatch.setattr(runner.subprocess, "Popen", forbidden)
    monkeypatch.setattr(runner.subprocess, "run", forbidden)
    monkeypatch.setattr(runner, "_port_free", forbidden)
    monkeypatch.setattr(runner, "_run_browser", forbidden)
    monkeypatch.setattr(runner, "run_row", forbidden)
    monkeypatch.setattr(runner, "ThreadPoolExecutor", forbidden)
    assert runner.main() == 2
    assert message in output.getvalue()
    assert "artifacts:" not in output.getvalue()
    assert attempts == []
    assert not (tmp_path / "chosen-artifacts").exists()


@pytest.mark.parametrize(("ports", "wanted", "offset", "jobs", "effective_jobs"), [
    ([7861, 7862], [1], -7861, 4, 1),  # Unselected row would use port zero.
    ([7862, 7863], [0], 65535 - 7862, 4, 1),  # Only the selected row and clamped band matter.
    ([7862, 7863], [0, 1], -7861, 2, 2),
    ([7862, 7863], [0, 1], 65535 - 100 - 7863, 2, 2),
    ([7867, 7867], [0, 1], 0, 2, 2),  # The same band cannot run these rows simultaneously.
    ([7862, 7962], [0, 1], 0, 1, 1),  # Serial rows may use ports a worker span apart.
    ([7862, 7963], [0, 1], 0, 2, 2),  # A wide row range need not contain a collision.
    ([7862, 7962], [0], 0, 2, 1),  # An unselected collision partner is irrelevant.
    ([7862], [0], 0, 0, 1),  # Preserve the existing jobs clamp.
])
def test_valid_selected_port_bands_keep_launch_arguments_and_summary(
    monkeypatch, tmp_path, ports, wanted, offset, jobs, effective_jobs,
):
    runner, output, rows = _port_plan_main(monkeypatch, tmp_path, ports, wanted, offset, jobs)
    calls = []
    pools = []
    allowed_bands = [offset + runner.WORKER_SPAN * index for index in range(effective_jobs)]

    class InlinePool:
        def __init__(self, workers):
            pools.append(workers)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def map(self, function, items):
            return map(function, items)

    def run(row, smokes, band, stream, artifacts):
        assert band in allowed_bands
        assert 1 <= row.port + band <= 65535
        assert stream == (effective_jobs == 1)
        calls.append((row, smokes, band))
        destination = artifacts / smokes[0]
        destination.mkdir()
        (destination / "result.json").write_text(json.dumps({"smoke": smokes[0], "exit_code": 0}))
        return {smoke: 0 for smoke in smokes}

    def forbidden(*_args, **_kwargs):
        pytest.fail("mocked launch plan reached an actual socket or process")

    monkeypatch.setattr(runner.socket, "socket", forbidden)
    monkeypatch.setattr(runner.subprocess, "Popen", forbidden)
    monkeypatch.setattr(runner.subprocess, "run", forbidden)
    monkeypatch.setattr(runner, "ThreadPoolExecutor", InlinePool)
    monkeypatch.setattr(runner, "run_row", run)
    assert runner.main() == 0
    assert [(row, smokes) for row, smokes, _band in calls] == [
        (row, row.smokes) for index, row in enumerate(rows) if index in wanted
    ]
    assert pools == ([effective_jobs] if effective_jobs > 1 else [])
    assert len(calls) == len(wanted)
    assert "of " + str(len(wanted)) + " passed" in output.getvalue()
    summaries = list((tmp_path / "chosen-artifacts").glob("*/summary.json"))
    assert len(summaries) == 1
    receipts = json.loads(summaries[0].read_text(encoding="utf-8"))["results"]
    assert {item["smoke"] for item in receipts} == {f"smoke-{index}" for index in wanted}
    assert all(item["exit_code"] == 0 for item in receipts)


def test_list_remains_read_only_even_with_an_unusable_execution_offset(monkeypatch, tmp_path):
    runner, output, _rows_unused = _port_plan_main(monkeypatch, tmp_path, [7862], [], 10**100, 3)
    monkeypatch.setattr(runner.sys, "argv", ["run_smoke.py", "--list", "--offset", str(10**100)])

    def forbidden(*_args, **_kwargs):
        pytest.fail("list attempted execution-plan validation or startup")

    monkeypatch.setattr(runner, "_port_plan_error", forbidden)
    monkeypatch.setattr(Path, "mkdir", forbidden)
    monkeypatch.setattr(runner.tempfile, "mkdtemp", forbidden)
    monkeypatch.setattr(runner, "run_row", forbidden)
    monkeypatch.setattr(runner, "ThreadPoolExecutor", forbidden)
    assert runner.main() == 0
    assert output.getvalue() == "7862  owner/model  smoke-0\n"
