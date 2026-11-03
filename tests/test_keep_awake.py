"""Keeping the PC awake (`backend.keep_awake`): the systemd-inhibit child with a
fake popen and clock (no real systemd), the controller through its asyncio
task, the server's providers, activity middleware and status, the CLI's hold,
the doctor's rows and the two settings."""
from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend import doctor, keep_awake, main
from backend.bundle_manager import BACKUP_PREFIX, PREFIX as BUNDLE_PREFIX
from backend.doctor import PASS, WARN
from backend.keep_awake import Inhibitor, KeepAwake, NoInhibitor, counts_as_activity
from backend.schemas import KeepAwakeStatus
from tests.api_helpers import until
from tests.cli_helpers import cli, isolated_execution_environment  # noqa: F401 - fixtures

# The real factory, before conftest's autouse stub replaces it for each test.
platform_inhibitor = keep_awake.platform_inhibitor

ROOT = Path(__file__).resolve().parents[1]


class Clock:
    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


class Pipe:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class FakeProc:
    """A systemd-inhibit child: `lives` None runs until its stdin closes (or
    a signal), 0 exits at once with `stderr`, N dies N seconds in."""

    def __init__(self, clock: Clock, lives: float | None = None, *, returncode: int = 1,
                 stderr: bytes = b"", ignores_eof: bool = False) -> None:
        self.clock, self.lives, self.stderr_text = clock, lives, stderr
        self.started = clock()
        self.exit_code = returncode
        self.ignores_eof = ignores_eof
        self.stdin, self.stderr = Pipe(), Pipe()
        self.returncode: int | None = None
        self.signals: list[str] = []

    def poll(self) -> int | None:
        if self.returncode is None:
            if self.stdin.closed and not self.ignores_eof:
                self.returncode = 0
            elif self.lives is not None and self.clock() - self.started >= self.lives:
                self.returncode = self.exit_code
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        code = self.poll()
        if code is None:
            raise subprocess.TimeoutExpired("systemd-inhibit", timeout or 0)
        return code

    def communicate(self, timeout: float | None = None) -> tuple[bytes, bytes]:
        return b"", self.stderr_text

    def terminate(self) -> None:
        self.signals.append("terminate")
        self.returncode = -15

    def kill(self) -> None:
        self.signals.append("kill")
        self.returncode = -9


class Popen:
    def __init__(self, *procs: FakeProc) -> None:
        self.procs = list(procs)
        self.calls: list[tuple[list[str], dict]] = []

    def __call__(self, argv, **kwargs):
        self.calls.append((argv, kwargs))
        proc = self.procs.pop(0)
        proc.started = proc.clock()         # its life counts from its start
        return proc


# --- the inhibitor -------------------------------------------------------------

def test_a_live_child_holds_on_a_pipe_only_we_hold_until_released():
    clock = Clock()
    proc = FakeProc(clock)
    popen = Popen(proc)
    inhibitor = Inhibitor("AuraDream is in use", program="/usr/bin/systemd-inhibit", popen=popen, monotonic=clock)
    inhibitor.hold()
    inhibitor.hold()                        # idempotent: one child
    argv, kwargs = popen.calls[0]
    assert len(popen.calls) == 1 and inhibitor.held
    assert argv == ["/usr/bin/systemd-inhibit", "--what=sleep:idle", "--who=AuraDream",
                    "--why=AuraDream is in use", "--mode=block", "cat"]
    assert kwargs["stdin"] == subprocess.PIPE and kwargs["close_fds"] is True
    inhibitor.release()
    assert proc.stdin.closed and proc.signals == [] and not inhibitor.held
    inhibitor.release()                     # idempotent


def test_release_terminates_a_child_that_ignores_the_end_of_its_input():
    clock = Clock()
    proc = FakeProc(clock, ignores_eof=True)
    inhibitor = Inhibitor(popen=Popen(proc), monotonic=clock)
    inhibitor.hold()
    inhibitor.release()
    assert proc.signals == ["terminate"] and proc.stderr.closed


def test_a_child_that_exits_at_once_was_refused_and_waits_before_trying_again():
    clock = Clock()
    popen = Popen(FakeProc(clock, 0, stderr=b"\nFailed to inhibit: Access denied\n"), FakeProc(clock))
    inhibitor = Inhibitor(popen=popen, monotonic=clock)
    with pytest.raises(OSError, match="refused \\(exit code 1\\): Failed to inhibit: Access denied"):
        inhibitor.hold()
    assert not inhibitor.held
    clock.now += Inhibitor.RETRY_AFTER - 1
    with pytest.raises(OSError, match="Access denied"):
        inhibitor.hold()                    # within RETRY_AFTER: not tried again
    assert len(popen.calls) == 1
    clock.now += 2
    inhibitor.hold()
    assert inhibitor.held and len(popen.calls) == 2


def test_a_child_that_dies_soon_counts_as_a_refusal_and_one_that_lived_is_restarted():
    clock = Clock()
    popen = Popen(FakeProc(clock, 10, stderr=b"no logind"), FakeProc(clock, 120), FakeProc(clock))
    inhibitor = Inhibitor(popen=popen, monotonic=clock)
    inhibitor.hold()
    clock.now += 10
    assert not inhibitor.held
    with pytest.raises(OSError, match="stopped at once .*no logind"):
        inhibitor.hold()
    clock.now += Inhibitor.RETRY_AFTER
    inhibitor.hold()                        # the second child, which lives two minutes
    clock.now += 120
    inhibitor.hold()                        # logind restarted, say: a new child, no refusal
    assert inhibitor.held and len(popen.calls) == 3


def test_a_program_that_cannot_start_is_a_refusal():
    def popen(argv, **kwargs):
        raise FileNotFoundError(2, "No such file or directory")
    inhibitor = Inhibitor(popen=popen, monotonic=Clock())
    with pytest.raises(OSError, match="could not be started"):
        inhibitor.hold()


def test_platform_inhibitor_is_systemd_inhibit_on_linux_only_where_it_is_installed():
    found = platform_inhibitor(platform="linux", which=lambda name: "/usr/bin/" + name)
    assert isinstance(found, Inhibitor) and found.program == "/usr/bin/systemd-inhibit"
    assert isinstance(platform_inhibitor(platform="linux", which=lambda name: None), NoInhibitor)
    assert isinstance(platform_inhibitor(platform="win32", which=lambda name: "x"), NoInhibitor)
    nothing = NoInhibitor()
    nothing.hold()
    assert nothing.held is False and nothing.kind == "none"


# --- the controller ------------------------------------------------------------

class Recorder:
    """An inhibitor that records, optionally refusing."""

    kind = "systemd-inhibit"
    RETRY_AFTER = 300

    def __init__(self, refuse: str | None = None) -> None:
        self.held = False
        self.refuse = refuse
        self.holds = 0
        self.thread_ids: set[int] = set()

    def hold(self) -> None:
        self.thread_ids.add(threading.get_ident())
        self.holds += 1
        if self.refuse:
            raise OSError(self.refuse)
        self.held = True

    def release(self) -> None:
        self.held = False


async def _until(predicate, timeout: float = 5.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() > deadline:
            raise AssertionError("timed out")
        await asyncio.sleep(0.01)


def _run(scenario):
    asyncio.run(asyncio.wait_for(scenario(), 30))


def test_a_provider_holds_and_its_end_releases_off_the_loop_thread():
    async def scenario():
        inhibitor = Recorder()
        # An hour between ticks: only a wake can check again here.
        controller = KeepAwake(inhibitor, idle_minutes=0, monotonic=Clock(), interval=3600)
        busy = {"reason": None}
        controller.add_provider(lambda: busy["reason"])
        controller.start()
        await _until(lambda: controller.status()["reasons"] == [] and inhibitor.holds == 0)
        busy["reason"] = "an image job is queued or running"
        controller.wake()
        await _until(lambda: inhibitor.held)
        assert controller.status()["reasons"] == ["an image job is queued or running"]
        assert threading.get_ident() not in inhibitor.thread_ids
        busy["reason"] = None
        controller.wake()
        await _until(lambda: not inhibitor.held)
        await controller.stop()
    _run(scenario)


def test_a_counted_hold_from_a_thread_is_held_at_once_not_at_the_next_tick():
    async def scenario():
        inhibitor = Recorder()
        controller = KeepAwake(inhibitor, idle_minutes=0, monotonic=Clock(), interval=3600)
        controller.start()
        await asyncio.sleep(0)
        entered, leave = threading.Event(), threading.Event()

        def download():
            with controller.busy("loading the model"):
                entered.set()
                leave.wait(10)

        worker = threading.Thread(target=download)
        worker.start()
        await asyncio.to_thread(entered.wait, 5)
        await _until(lambda: inhibitor.held)
        assert controller.status()["reasons"] == ["loading the model"]
        leave.set()
        await asyncio.to_thread(worker.join, 5)
        await _until(lambda: not inhibitor.held)
        await controller.stop()
    _run(scenario)


def test_the_idle_minutes_after_a_request_hold_and_then_let_go():
    async def scenario():
        clock = Clock()
        inhibitor = Recorder()
        controller = KeepAwake(inhibitor, idle_minutes=20, monotonic=clock, interval=3600)
        controller.start()
        # Starting counts as use.
        await _until(lambda: inhibitor.held)
        assert controller.status()["reasons"] == ["in use (a request in the last 20 minutes)"]
        clock.now += 20 * 60
        controller.wake()
        await _until(lambda: not inhibitor.held)
        controller.note_activity()          # wakes the check itself while idle
        await _until(lambda: inhibitor.held)
        await controller.stop()
        assert not inhibitor.held and controller.status()["holding"] is False
    _run(scenario)


def test_a_refusal_is_wanted_not_holding_logged_once_and_never_raised(caplog):
    async def scenario():
        inhibitor = Recorder(refuse="systemd-inhibit refused (exit code 1): Access denied")
        controller = KeepAwake(inhibitor, idle_minutes=0, monotonic=Clock(), interval=3600)
        controller.add_provider(lambda: "a gallery export is running")
        controller.start()
        await _until(lambda: inhibitor.holds >= 1)
        controller.wake()
        await _until(lambda: inhibitor.holds >= 2)
        await asyncio.sleep(0.05)
        status = controller.status()
        assert status["wanted"] is True and status["holding"] is False
        assert "Access denied" in status["detail"]
        await controller.stop()
    with caplog.at_level(logging.WARNING, logger="auradream.keep_awake"):
        _run(scenario)
    assert sum("Could not keep the PC awake" in r.getMessage() for r in caplog.records) == 1


def test_a_failing_provider_is_ignored_and_logged_once(caplog):
    async def scenario():
        inhibitor = Recorder()
        controller = KeepAwake(inhibitor, idle_minutes=0, monotonic=Clock(), interval=3600)

        def broken():
            raise RuntimeError("dictionary changed size during iteration")
        controller.add_provider(broken)
        controller.add_provider(lambda: "switching the checkpoint")
        controller.start()
        await _until(lambda: inhibitor.held)
        controller.wake()
        await asyncio.sleep(0.05)
        assert controller.status()["reasons"] == ["switching the checkpoint"]
        await controller.stop()
    with caplog.at_level(logging.WARNING, logger="auradream.keep_awake"):
        _run(scenario)
    assert sum("busy check failed" in r.getMessage() for r in caplog.records) == 1


def test_status_has_the_documented_shape_and_validates():
    controller = KeepAwake(NoInhibitor(), idle_minutes=0, monotonic=Clock())
    status = controller.status()
    assert set(status) == {"kind", "holding", "wanted", "reasons", "detail", "idle_minutes"}
    assert status == {"kind": "none", "holding": False, "wanted": False, "reasons": [], "detail": None,
                      "idle_minutes": 0}
    KeepAwakeStatus.model_validate(status)
    assert KeepAwakeStatus.model_validate(keep_awake.off_status(20)).kind == "off"


def test_module_busy_is_nothing_until_a_controller_is_installed():
    with keep_awake.busy("loading the model"):
        pass                                # nothing installed: a no-op
    controller = KeepAwake(NoInhibitor(), idle_minutes=0, monotonic=Clock())
    keep_awake.install(controller)
    assert keep_awake.current() is controller
    with keep_awake.busy("loading the model"):
        assert controller.reasons() == ["loading the model"]
    assert controller.reasons() == []
    controller.wake()                       # not started: nothing to wake, no error


def test_hold_while_holds_for_the_block_and_never_raises_a_refusal():
    inhibitor = Recorder()
    with keep_awake.hold_while("rendering", inhibitor=inhibitor):
        assert inhibitor.held
    assert not inhibitor.held
    refused = Recorder(refuse="no")
    with keep_awake.hold_while("rendering", inhibitor=refused):
        pass
    with pytest.raises(RuntimeError), keep_awake.hold_while("rendering", inhibitor=inhibitor):
        raise RuntimeError("the render failed")
    assert not inhibitor.held


# --- what counts as someone using it -------------------------------------------

@pytest.mark.parametrize(("method", "path", "counts"), [
    ("GET", "/", True),
    ("GET", "/api/gallery", True),
    ("POST", "/api/generate", True),
    ("GET", "/api/status", False),
    ("GET", "/api/status/", False),
    ("GET", "/api/jobs", False),
    ("GET", "/api/jobs/abc/stream", False),
    ("POST", "/api/jobs/abc/cancel", True),
    ("GET", BUNDLE_PREFIX + "/status", False),
    ("POST", BUNDLE_PREFIX + "/export", True),
    ("GET", BACKUP_PREFIX, False),
    ("PUT", BACKUP_PREFIX, True),
    ("POST", "/api/masks/" + "a" * 64 + "/keep", False),
    ("POST", "/api/masks", True),
    ("GET", "/static/app.js", False),
    ("GET", "/outputs/thumbnails/1.webp", False),
    ("HEAD", "/", False),
    ("OPTIONS", "/api/generate", False),
])
def test_counts_as_activity(method, path, counts):
    assert counts_as_activity(method, path) is counts


def test_the_polls_named_are_the_routes_the_page_polls():
    # The prefixes are spelled out in keep_awake (stdlib only); held to the routes.
    assert BUNDLE_PREFIX == "/api/gallery/bundles" and BACKUP_PREFIX == "/api/gallery/automatic-backup"
    web = ROOT / "web"
    assert "setInterval(refreshStatus" in (web / "app.js").read_text(encoding="utf-8")
    assert "api('/api/gallery/bundles/status')" in (web / "bundles.js").read_text(encoding="utf-8")
    assert "/keep`" in (web / "inpaint.js").read_text(encoding="utf-8")


# --- the server ------------------------------------------------------------------

def test_the_lifespan_runs_one_controller_and_lets_go_on_the_way_out(fake):
    from fastapi.testclient import TestClient
    from tests.api_helpers import ORIGIN

    with TestClient(main.app, base_url=ORIGIN) as client:
        controller = main._keep_awake
        assert controller is not None and keep_awake.current() is controller
        body = client.get("/api/status").json()
        assert body["keep_awake"]["kind"] == "none"     # conftest: nothing real is held
        assert set(body["keep_awake"]) == {"kind", "holding", "wanted", "reasons", "detail", "idle_minutes"}
        assert body["keep_awake"]["idle_minutes"] == 20
    assert main._keep_awake is None and keep_awake.current() is None


def test_keep_awake_off_starts_nothing_and_says_off(fake, monkeypatch):
    import dataclasses
    from fastapi.testclient import TestClient
    from tests.api_helpers import ORIGIN

    monkeypatch.setattr(main, "settings", dataclasses.replace(main.settings, keep_awake=False))
    made = []
    monkeypatch.setattr(keep_awake, "platform_inhibitor", lambda *a, **k: made.append(1) or NoInhibitor())
    with TestClient(main.app, base_url=ORIGIN) as client:
        assert main._keep_awake is None and keep_awake.current() is None
        assert client.get("/api/status").json()["keep_awake"]["kind"] == "off"
    assert made == []


def test_requests_note_activity_but_the_page_polls_do_not(client, monkeypatch):
    noted = []
    monkeypatch.setattr(main._keep_awake, "note_activity", lambda: noted.append(1))
    client.get("/api/status")
    client.get("/api/jobs")
    client.get(BUNDLE_PREFIX + "/status")
    client.get(BACKUP_PREFIX)
    client.post("/api/masks/" + "0" * 64 + "/keep")
    client.get("/static/app.js")
    client.head("/")
    assert noted == []
    client.get("/")
    client.get("/api/gallery")
    assert len(noted) == 2


def test_a_refused_host_is_never_activity(client, monkeypatch):
    noted = []
    monkeypatch.setattr(main._keep_awake, "note_activity", lambda: noted.append(1))
    response = client.get("/", headers={"host": "evil.example"})
    assert response.status_code == 400 and noted == []


def test_a_running_job_is_a_reason_seen_at_once(client, fake):
    fake.step_seconds = 0.2
    controller = main._keep_awake
    assert controller is not None
    job = client.post("/api/generate", json={"prompt": "a lighthouse", "width": 256, "height": 256,
                                             "steps": 20}).json()["job_id"]
    until(lambda: "an image job is queued or running" in controller.status()["reasons"], 5)
    assert "an image job is queued or running" in client.get("/api/status").json()["keep_awake"]["reasons"]
    assert main._busy_jobs() is not None
    client.post(f"/api/jobs/{job}/cancel")
    until(lambda: main._busy_jobs() is None, 10)


def test_the_load_holds_on_its_own_thread(client, fake, monkeypatch):
    controller = main._keep_awake
    assert controller is not None
    fake.loaded = False
    entered, leave = threading.Event(), threading.Event()
    held_inside = []
    real_load = fake.load

    def slow_load():
        held_inside.append("loading the model" in controller._counted)
        entered.set()
        leave.wait(10)
        real_load()

    monkeypatch.setattr(fake, "load", slow_load)
    client.post("/api/model/load")
    assert entered.wait(5)
    assert held_inside == [True]
    until(lambda: "loading the model" in controller.status()["reasons"], 5)
    assert main._busy_loading() == "loading the model"
    leave.set()
    until(lambda: main._busy_loading() is None and "loading the model" not in controller._counted, 5)


def test_providers_name_the_card_a_switch_and_a_bundle(fake, monkeypatch):
    assert main._busy_card() is None and main._busy_switching() is None and main._busy_bundle() is None
    monkeypatch.setattr(main.jobs, "card", lambda: ("critic", None))
    assert main._busy_card() == "the critic is using the GPU"
    monkeypatch.setattr(main.jobs, "card", lambda: ("job", None))
    assert main._busy_card() is None                    # the job provider names it
    monkeypatch.setattr(main, "_switch_draining", True)
    assert main._busy_switching() == "switching the checkpoint"
    main.bundle_manager.state = {**main.bundle_manager.state, "state": "running", "operation": "automatic_backup"}
    assert main._busy_bundle() == "a gallery automatic backup is running"


def test_the_critic_prepare_holds_while_it_downloads(client, fake, monkeypatch):
    controller = main._keep_awake
    seen = []
    monkeypatch.setattr(main.prompt_critic, "prepare",
                        lambda: seen.append("preparing the prompt critic" in controller._counted))
    client.post("/api/prompt/critique", json={"prompt": "a lighthouse in fog"})
    assert seen == [True]


def test_the_fake_server_switches_keep_awake_off():
    text = (ROOT / "tools" / "fake_server.py").read_text(encoding="utf-8")
    assert 'os.environ["AURADREAM_KEEP_AWAKE"] = "false"' in text


# --- the command line --------------------------------------------------------------

@pytest.mark.parametrize("enabled", [True, False])
def test_a_cli_render_is_held_from_the_load_and_released_at_its_end(cli, monkeypatch, enabled):
    import dataclasses

    from backend import config, generate

    events: list[str] = []

    class Watch(Recorder):
        def hold(self):
            super().hold()
            events.append("hold")

        def release(self):
            super().release()
            events.append("release")

    def made(reason="", **kwargs):
        events.append(reason)
        return Watch()

    def load():
        events.append("load")
        cli.loaded = True

    monkeypatch.setattr(keep_awake, "platform_inhibitor", made)
    monkeypatch.setattr(config, "settings", dataclasses.replace(config.settings, keep_awake=enabled))
    monkeypatch.setattr(cli, "load", load)
    assert generate.main(["a fox", "--width", "256", "--height", "256", "-q"]) == 0
    if enabled:
        assert events == ["AuraDream is rendering from the command line", "hold", "load", "release"]
    else:
        assert events == ["load"]


# --- the doctor ----------------------------------------------------------------------

def _torch(total_gib: float, capability=(8, 6)):
    return SimpleNamespace(
        version=SimpleNamespace(cuda="13.0"),
        cuda=SimpleNamespace(
            is_available=lambda: True,
            get_device_capability=lambda index=0: capability,
            get_device_properties=lambda index=0: SimpleNamespace(total_memory=int(total_gib * 1024 ** 3)),
        ),
    )


def test_vram_under_the_default_checkpoints_peak_warns_with_smaller_checkpoints():
    [check] = doctor.check_vram(_torch(7.6), platform="linux")
    assert check.status == WARN and check.title == "VRAM"
    assert "7.6 GiB" in check.detail and "12.0 GiB allocated" in check.detail and "RTX 5080" in check.detail
    assert "Animagine XL 4.0" in check.fix and "Anima" in check.fix and "FLUX.2 [klein] 4B" in check.fix
    assert "int4" in check.detail and "nvfp4 needs an RTX 50" in check.detail


def test_vram_enough_passes_and_the_int4_note_is_linux_below_blackwell_only():
    [check] = doctor.check_vram(_torch(15.9, (12, 0)), platform="linux")
    assert check.status == PASS and "int4" not in check.detail
    [check] = doctor.check_vram(_torch(23.6, (8, 9)), platform="win32")
    assert check.status == PASS and "int4" not in check.detail
    assert doctor.check_vram(SimpleNamespace(version=SimpleNamespace(cuda=None), cuda=None)) == []


def _ran(code, stderr=""):
    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(returncode=code, stdout="", stderr=stderr)
    return run, calls


def test_keep_awake_trial_passes_when_the_hold_is_allowed():
    run, calls = _ran(0)
    [check] = doctor.check_keep_awake("linux", which=lambda name: "/usr/bin/" + name, run=run, enabled=True)
    assert check.status == PASS and check.title == "Keep awake"
    argv, kwargs = calls[0]
    assert argv == ["/usr/bin/systemd-inhibit", "--what=sleep:idle", "--who=AuraDream doctor", "--why=check",
                    "--mode=block", "true"]
    assert kwargs["timeout"] == doctor.INHIBIT_TRIAL_SECONDS and kwargs["stdin"] == subprocess.DEVNULL


def test_keep_awake_refused_warns_with_the_polkit_rule():
    run, _ = _ran(1, "Failed to inhibit: Access denied")
    [check] = doctor.check_keep_awake("linux", which=lambda name: "/usr/bin/" + name, run=run, enabled=True,
                                      user="matt")
    assert check.status == WARN and "Access denied" in check.detail
    assert "org.freedesktop.login1.inhibit-block-sleep" in check.fix
    assert "org.freedesktop.login1.inhibit-block-idle" in check.fix
    assert 'subject.user == "matt"' in check.fix and "/etc/polkit-1/rules.d/" in check.fix


def test_keep_awake_timeout_missing_off_and_elsewhere():
    def slow(argv, **kwargs):
        raise subprocess.TimeoutExpired(argv, kwargs["timeout"])
    [check] = doctor.check_keep_awake("linux", which=lambda name: "x", run=slow, enabled=True, user="u")
    assert check.status == WARN and "did not finish" in check.detail
    [check] = doctor.check_keep_awake("linux", which=lambda name: None, enabled=True)
    assert check.status == WARN and "not found" in check.detail

    def never(*args, **kwargs):
        raise AssertionError("nothing is run")
    [check] = doctor.check_keep_awake("linux", which=lambda name: "x", run=never, enabled=False)
    assert check.status == PASS and "AURADREAM_KEEP_AWAKE=false" in check.detail
    assert doctor.check_keep_awake("win32", run=never) == []


def test_lan_rows():
    assert doctor.check_lan("127.0.0.1", ["127.0.0.1", "localhost"]) == []
    [check] = doctor.check_lan("0.0.0.0", ["127.0.0.1", "localhost", "my-box", "192.168.1.20"])
    assert check.status == PASS and "my-box, 192.168.1.20" in check.detail
    [check] = doctor.check_lan("0.0.0.0", ["my-box"])
    assert check.status == WARN and "127.0.0.1" in check.detail and "second start" in check.detail
    checks = doctor.check_lan("0.0.0.0", ["*"])
    assert [c.status for c in checks] == [WARN, WARN]
    assert "wildcard (*)" in checks[1].detail and "403" in checks[1].detail
    [check] = doctor.check_lan("0.0.0.0", ["127.0.0.1", "*.local"])
    assert check.status == WARN and "*.local" in check.detail
    # The default allowed hosts on a LAN bind reach no other machine.
    [check] = doctor.check_lan("0.0.0.0", ["127.0.0.1", "localhost"])
    assert check.status == WARN and "400 Invalid host header" in check.detail
    [check] = doctor.check_lan("0.0.0.0", ["127.0.0.1", "localhost", "::1"])
    assert check.status == WARN and "only this machine's names" in check.detail
    # Browsers send lowercase names, so a capitalised entry never matches.
    [check] = doctor.check_lan("0.0.0.0", ["127.0.0.1", "localhost", "My-Box"])
    assert check.status == WARN and "My-Box" in check.detail and "lowercase" in check.detail


def test_the_doctor_runs_the_new_rows():
    assert doctor.check_vram in doctor.CHECKS
    assert doctor.check_keep_awake in doctor.CHECKS and doctor.check_lan in doctor.CHECKS


# --- the settings ----------------------------------------------------------------------

_PRINT = ("from backend.config import settings; "
          "print(settings.keep_awake); print(settings.keep_awake_idle_minutes)")


def _settings(env: dict[str, str], tmp_path) -> subprocess.CompletedProcess:
    clean = {k: v for k, v in os.environ.items() if not k.startswith(("AURADREAM_", "DIFFUSERS_"))}
    base = {"AURADREAM_OUTPUTS_DIR": str(tmp_path / "out"), "AURADREAM_DB_PATH": str(tmp_path / "g.db")}
    return subprocess.run([sys.executable, "-c", _PRINT], env={**clean, **base, **env}, capture_output=True,
                          text=True, cwd=ROOT, timeout=120)


@pytest.mark.parametrize(("env", "expected"), [
    ({}, ["True", "20"]),
    ({"AURADREAM_KEEP_AWAKE": "false", "AURADREAM_KEEP_AWAKE_IDLE_MINUTES": "0"}, ["False", "0"]),
    ({"AURADREAM_KEEP_AWAKE": "on", "AURADREAM_KEEP_AWAKE_IDLE_MINUTES": "45"}, ["True", "45"]),
])
def test_settings_parse(env, expected, tmp_path):
    result = _settings(env, tmp_path)
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == expected


def test_a_negative_idle_minutes_is_refused_by_name(tmp_path):
    result = _settings({"AURADREAM_KEEP_AWAKE_IDLE_MINUTES": "-5"}, tmp_path)
    assert result.returncode != 0 and "AURADREAM_KEEP_AWAKE_IDLE_MINUTES" in result.stderr
