"""The platform holders `backend.keep_awake` keeps the PC awake with: the
systemd-inhibit child with a fake popen and clock (no real systemd), Windows'
power request through powershell.exe under WSL, and Windows' own with its
four calls faked (and once for real, let go at once). The controller, the
server and the doctor that use them are `test_keep_awake.py`'s."""
from __future__ import annotations

import asyncio
import logging
import shutil
import subprocess
import sys
import threading

import pytest

from backend import keep_awake
from backend.keep_awake import Inhibitor, KeepAwake, NoInhibitor, PowerRequest
from backend.schemas import KeepAwakeStatus
from tests.keep_awake_helpers import (
    platform_inhibitor,
    ROOT,
    Clock,
    FakeProc,
    Popen,
    INTEROP,
    _exists,
    WslProc,
    _wsl,
    FakePower,
    _until,
    _run,
)


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
    found = platform_inhibitor(platform="linux", wsl=False, which=lambda name: "/usr/bin/" + name)
    assert isinstance(found, Inhibitor) and found.program == "/usr/bin/systemd-inhibit"
    assert isinstance(platform_inhibitor(platform="linux", wsl=False, which=lambda name: None), NoInhibitor)
    assert isinstance(platform_inhibitor(platform="darwin", which=lambda name: "x"), NoInhibitor)
    nothing = NoInhibitor()
    nothing.hold()
    assert nothing.held is False and nothing.kind == "none" and nothing.detail is None


# --- WSL: Windows' power request through powershell.exe ---------------------------


def test_wsl_is_its_interop_entry_or_microsoft_in_the_kernel_release():
    assert keep_awake.under_wsl(release="5.15.167.4-microsoft-standard-WSL2", exists=_exists())
    assert keep_awake.under_wsl(release="4.4.0-19041-Microsoft", exists=_exists())
    assert keep_awake.under_wsl(release="6.8.0-generic", exists=_exists(keep_awake.WSL_INTEROP[1]))
    assert not keep_awake.under_wsl(release="6.8.0-48-generic", exists=_exists())


def test_windows_powershell_needs_interop_then_the_path_or_the_default_mount():
    found = lambda name: "/mnt/c/WINDOWS/System32/WindowsPowerShell/v1.0/" + name  # noqa: E731
    assert keep_awake.windows_powershell(which=found, exists=_exists()) is None      # interop off
    assert keep_awake.windows_powershell(which=found, exists=_exists(INTEROP)).endswith("/powershell.exe")
    assert keep_awake.windows_powershell(
        which=lambda name: None, exists=_exists(INTEROP, keep_awake.WINDOWS_POWERSHELL),
    ) == keep_awake.WINDOWS_POWERSHELL
    assert keep_awake.windows_powershell(which=lambda name: None, exists=_exists(INTEROP)) is None


def test_under_wsl_the_hold_is_windows_power_request_never_systemd_inhibit():
    everything = lambda name: "/usr/bin/" + name  # noqa: E731
    found = platform_inhibitor(platform="linux", wsl=True, which=everything, exists=_exists(INTEROP))
    assert isinstance(found, keep_awake.WslPowerRequest) and found.kind == "wsl-power-request"
    assert found.program == "/usr/bin/powershell.exe" and not found.held
    # Detected by itself, from the interop entry.
    assert isinstance(platform_inhibitor(platform="linux", which=everything, exists=_exists(INTEROP)),
                      keep_awake.WslPowerRequest)
    # Without interop nothing can reach Windows: nothing is held, and the status says why.
    nothing = platform_inhibitor(platform="linux", wsl=True, which=everything, exists=_exists())
    assert isinstance(nothing, NoInhibitor) and nothing.detail == keep_awake.WSL_NO_HOLD
    assert "never Windows" in nothing.detail


def _script(argv: list[str]) -> str:
    import base64

    assert argv[:5] == ["/mnt/c/ps/powershell.exe", "-NoLogo", "-NoProfile", "-NonInteractive", "-EncodedCommand"]
    return base64.b64decode(argv[5]).decode("utf-16-le")


def test_a_wsl_hold_is_powershell_holding_on_a_pipe_only_we_hold_until_released():
    clock = Clock()
    proc = WslProc(clock)
    request, popen = _wsl(proc)
    request.hold()
    request.hold()                          # idempotent: one child
    argv, kwargs = popen.calls[0]
    assert len(popen.calls) == 1 and request.held and request.windows_pid == "4242"
    assert kwargs["stdin"] == subprocess.PIPE and kwargs["stdout"] == subprocess.PIPE
    assert kwargs["stderr"] == subprocess.DEVNULL and kwargs["close_fds"] is True
    script = _script(argv)
    assert "::Hold('AuraDream is in use')" in script
    # The system only (PowerRequestSystemRequired, 1), never the display (0);
    # cleared and closed once its stdin ends.
    assert "PowerSetRequest(request, 1)" in script and "PowerClearRequest(request, 1)" in script
    assert "PowerSetRequest(request, 0)" not in script and "powercfg" not in script.lower()
    assert "[Console]::In.ReadLine()" in script and "::Let($request)" in script
    request.release()
    assert proc.stdin.closed and proc.signals == [] and proc.stdout.closed and not request.held
    request.release()                       # idempotent


def test_the_wsl_script_quotes_its_reason_as_one_powershell_string():
    script = keep_awake.wsl_hold_script("Matt's\nrender")
    assert "::Hold('Matt''s render')" in script
    assert "-EncodedCommand" not in script and "\"kernel32.dll\"" in script


def test_a_wsl_holder_that_ignores_the_end_of_its_input_is_terminated():
    clock = Clock()
    proc = WslProc(clock, ignores_eof=True)
    request, _ = _wsl(proc)
    request.hold()
    request.release()
    assert proc.signals == ["terminate"] and not request.held


def test_windows_refusal_through_wsl_is_in_windows_words_and_waits_before_asking_again():
    clock = Clock()
    refused = WslProc(clock, b"refused: Windows refused PowerSetRequest (error 5): Access is denied.\r\n")
    request, popen = _wsl(refused, WslProc(clock))
    with pytest.raises(OSError, match=r"^Windows refused PowerSetRequest \(error 5\): Access is denied\.$"):
        request.hold()
    assert not request.held and refused.stdin.closed
    clock.now += keep_awake.WslPowerRequest.RETRY_AFTER - 1
    with pytest.raises(OSError, match="Access is denied"):
        request.hold()
    assert len(popen.calls) == 1
    clock.now += 2
    request.hold()
    assert request.held and len(popen.calls) == 2


def test_a_wsl_holder_that_says_nothing_or_stops_at_once_or_cannot_start_is_a_refusal():
    clock = Clock()
    silent = WslProc(clock, None)
    request, _ = _wsl(silent)
    request.START_WAIT = 0.05
    with pytest.raises(OSError, match="gave no answer within 0.05 s"):
        request.hold()
    assert silent.stdin.closed and silent.returncode is not None and not request.held

    request, _ = _wsl(WslProc(clock, b"", returncode=255))
    with pytest.raises(OSError, match=r"stopped before holding \(exit code 255\)$"):
        request.hold()

    def popen(argv, **kwargs):
        raise OSError(8, "Exec format error")
    request = keep_awake.WslPowerRequest(program="powershell.exe", popen=popen, monotonic=clock)
    with pytest.raises(OSError, match="could not be started through WSL interop: .*Exec format error"):
        request.hold()


def test_a_wsl_holder_that_stops_soon_is_a_refusal_and_one_that_lived_is_restarted():
    clock = Clock()
    request, popen = _wsl(WslProc(clock, lives=10), WslProc(clock, lives=120), WslProc(clock))
    request.hold()
    clock.now += 10
    assert not request.held
    with pytest.raises(OSError, match="stopped holding at once"):
        request.hold()
    clock.now += keep_awake.WslPowerRequest.RETRY_AFTER
    request.hold()
    clock.now += 120
    request.hold()                          # it lived: a new one, no refusal
    assert request.held and len(popen.calls) == 3


def test_the_controller_says_why_nothing_is_held_under_wsl_without_interop():
    controller = KeepAwake(NoInhibitor(keep_awake.WSL_NO_HOLD), monotonic=Clock())
    status = controller.status()
    assert status["kind"] == "none" and status["detail"] == keep_awake.WSL_NO_HOLD
    KeepAwakeStatus.model_validate(status)
    KeepAwakeStatus.model_validate({**status, "kind": "wsl-power-request", "detail": None})


@pytest.mark.skipif(sys.platform != "win32" or not shutil.which("powershell.exe"),
                    reason="Windows' own powershell.exe")
def test_the_wsl_hold_script_holds_and_lets_go_for_real():
    # The script WSL sends, run by Windows' powershell.exe directly: Add-Type
    # compiles, the request is set and, at the end of its input, let go.
    request = keep_awake.WslPowerRequest("AuraDream test: released at once", program=shutil.which("powershell.exe"))
    try:
        request.hold()
        assert request.held and (request.windows_pid or "").isdigit()
    finally:
        proc = request._proc
        request.release()
    assert not request.held and proc is not None and proc.returncode == 0


# --- Windows' power request ------------------------------------------------------


def test_platform_inhibitor_is_a_power_request_on_windows_and_builds_it_without_a_call():
    # On any platform: building one loads no DLL, so a Linux runner can ask.
    found = platform_inhibitor("AuraDream is in use", platform="win32", which=lambda name: None)
    assert isinstance(found, PowerRequest) and found.kind == "power-request"
    assert found.reason == "AuraDream is in use" and found.held is False
    assert isinstance(found._api, keep_awake.Kernel32Power) and found._api._bound is None


def test_a_power_request_is_created_with_the_reason_set_once_and_cleared_then_closed():
    api = FakePower()
    request = PowerRequest("AuraDream is in use", api=api, monotonic=Clock())
    assert not request.held
    request.hold()
    request.hold()                          # idempotent: one handle
    assert request.held
    assert api.calls == [("create", "AuraDream is in use"), ("set", 0x101)]
    request.release()
    assert not request.held and api.calls[2:] == [("clear", 0x101), ("close", 0x101)]
    request.release()                       # idempotent
    assert len(api.calls) == 4
    request.hold()                          # a new handle each time it is wanted again
    assert request.held and api.calls[4:] == [("create", "AuraDream is in use"), ("set", 0x102)]
    request.release()


def test_only_the_idle_sleep_timer_is_asked_for_never_the_display():
    # winnt.h's POWER_REQUEST_TYPE: 0 is the display, 1 the system.
    assert keep_awake._SYSTEM_REQUIRED == 1
    assert keep_awake._CONTEXT_VERSION == 0 and keep_awake._CONTEXT_SIMPLE_STRING == 1
    text = (ROOT / "backend" / "keep_awake.py").read_text(encoding="utf-8")
    assert "SetThreadExecutionState(" not in text and "powercfg" not in text


@pytest.mark.parametrize("failing", ["create", "set"])
def test_a_refused_power_request_holds_nothing_leaks_no_handle_and_waits_before_asking_again(failing):
    clock = Clock()
    api = FakePower(failing)
    request = PowerRequest(api=api, monotonic=clock)
    with pytest.raises(OSError, match=f"Windows refused {failing} \\(error 5\\): Access is denied"):
        request.hold()
    assert not request.held
    # A handle Windows gave before refusing to set it is closed, not kept.
    assert api.named() == (["create"] if failing == "create" else ["create", "set", "close"])
    asked = len(api.calls)
    clock.now += PowerRequest.RETRY_AFTER - 1
    with pytest.raises(OSError, match="Access is denied"):
        request.hold()                      # within RETRY_AFTER: not asked again
    assert len(api.calls) == asked
    api.refuse.clear()
    clock.now += 2
    request.hold()
    assert request.held and api.named()[asked:] == ["create", "set"]
    request.release()                       # nothing is left held
    assert not request.held


def test_a_call_that_is_missing_altogether_is_a_refusal_in_words():
    class Broken(FakePower):
        def create(self, reason: str):
            raise AttributeError("function 'PowerCreateRequest' not found")
    request = PowerRequest(api=Broken(), monotonic=Clock())
    with pytest.raises(OSError, match="PowerCreateRequest"):
        request.hold()
    assert not request.held


def test_release_closes_the_handle_even_where_clearing_is_refused_and_never_raises(caplog):
    api = FakePower("clear", "close")
    request = PowerRequest(api=api, monotonic=Clock())
    request.hold()
    with caplog.at_level(logging.WARNING, logger="auradream.keep_awake"):
        request.release()
    assert not request.held and api.named() == ["create", "set", "clear", "close"]
    assert sum("letting go of the power request" in r.getMessage() for r in caplog.records) == 2


def test_the_controller_holds_and_releases_a_power_request_on_other_threads():
    async def scenario():
        api = FakePower()
        request = PowerRequest(api=api, monotonic=Clock())
        controller = KeepAwake(request, idle_minutes=0, monotonic=Clock(), interval=3600)
        busy = {"reason": None}
        controller.add_provider(lambda: busy["reason"])
        controller.start()
        busy["reason"] = "an image job is queued or running"
        controller.wake()
        await _until(lambda: controller.status()["holding"])
        status = controller.status()
        assert status["kind"] == "power-request" and status["wanted"] is True
        assert status["reasons"] == ["an image job is queued or running"]
        KeepAwakeStatus.model_validate(status)
        busy["reason"] = None
        controller.wake()
        await _until(lambda: not controller.status()["holding"])
        assert api.named() == ["create", "set", "clear", "close"]
        busy["reason"] = "loading the model"
        controller.wake()
        await _until(lambda: controller.status()["holding"])
        await controller.stop()             # the way out lets go too
        assert not request.held and api.named()[-2:] == ["clear", "close"]
        assert threading.get_ident() not in api.thread_ids
    _run(scenario)


def test_a_refused_power_request_is_wanted_not_holding_with_windows_words(caplog):
    async def scenario():
        request = PowerRequest(api=FakePower("set"), monotonic=Clock())
        controller = KeepAwake(request, idle_minutes=0, monotonic=Clock(), interval=3600)
        controller.add_provider(lambda: "a gallery export is running")
        controller.start()
        await _until(lambda: controller.status()["detail"] is not None)
        controller.wake()
        await asyncio.sleep(0.05)
        status = controller.status()
        assert status["wanted"] is True and status["holding"] is False
        assert "Windows refused set" in status["detail"]
        await controller.stop()
    with caplog.at_level(logging.WARNING, logger="auradream.keep_awake"):
        _run(scenario)
    [warning] = [r.getMessage() for r in caplog.records if "Could not keep the PC awake" in r.getMessage()]
    assert "Tried again every 5 minutes" in warning


def test_the_reason_context_is_the_size_and_shape_windows_reads():
    import ctypes

    context = keep_awake._ReasonContext()
    context.version, context.flags = keep_awake._CONTEXT_VERSION, keep_awake._CONTEXT_SIMPLE_STRING
    context.reason.simple = "AuraDream is in use"
    assert context.reason.simple == "AuraDream is in use"
    pointer = ctypes.sizeof(ctypes.c_void_p)
    # Two ULONGs, then a union as large as its localized member (a module,
    # two ULONGs and a pointer), which starts on a pointer boundary.
    assert keep_awake._ReasonContext.reason.offset == max(8, pointer)
    assert ctypes.sizeof(keep_awake._Reason) == 2 * pointer + 8
    assert keep_awake._Reason.simple.offset == 0


class FakeKernel32:
    """kernel32's four entries as `Kernel32Power` calls them, recorded.
    `answers` gives a call's return value; the default is success."""

    def __init__(self, **answers) -> None:
        self.answers = answers
        self.calls: list[tuple] = []

    def PowerCreateRequest(self, reference):  # noqa: N802 - Windows' name
        context = reference._obj            # what byref() points at
        self.calls.append(("PowerCreateRequest", context.version, context.flags, context.reason.simple))
        return self.answers.get("PowerCreateRequest", 0x2A4)

    def PowerSetRequest(self, handle, kind):  # noqa: N802
        self.calls.append(("PowerSetRequest", handle, kind))
        return self.answers.get("PowerSetRequest", 1)

    def PowerClearRequest(self, handle, kind):  # noqa: N802
        self.calls.append(("PowerClearRequest", handle, kind))
        return self.answers.get("PowerClearRequest", 1)

    def CloseHandle(self, handle):  # noqa: N802
        self.calls.append(("CloseHandle", handle))
        return self.answers.get("CloseHandle", 1)


def _kernel32_power(**answers) -> tuple[keep_awake.Kernel32Power, FakeKernel32]:
    api, dll = keep_awake.Kernel32Power(), FakeKernel32(**answers)
    api._bound = dll                        # as if kernel32 were bound: no DLL on any system
    return api, dll


def test_the_kernel32_calls_ask_for_the_system_with_a_simple_reason_on_one_handle():
    api, dll = _kernel32_power()
    request = PowerRequest("AuraDream is in use", api=api, monotonic=Clock())
    request.hold()
    assert request.held
    request.release()
    # Version 0, the simple-string flag, the reason; then the system (1,
    # never the display's 0) set and cleared on the handle create returned.
    assert dll.calls == [("PowerCreateRequest", 0, 1, "AuraDream is in use"), ("PowerSetRequest", 0x2A4, 1),
                         ("PowerClearRequest", 0x2A4, 1), ("CloseHandle", 0x2A4)]


def test_a_null_or_invalid_handle_from_kernel32_is_a_refusal_and_nothing_is_set():
    import ctypes

    for answer in (None, 0, ctypes.c_void_p(-1).value):
        api, dll = _kernel32_power(PowerCreateRequest=answer)
        request = PowerRequest(api=api, monotonic=Clock())
        with pytest.raises(OSError, match="Windows refused PowerCreateRequest"):
            request.hold()
        assert not request.held and [call[0] for call in dll.calls] == ["PowerCreateRequest"]


@pytest.mark.parametrize("call, method", [("PowerSetRequest", "set"), ("PowerClearRequest", "clear"),
                                          ("CloseHandle", "close")])
def test_a_zero_from_kernel32_is_a_refusal_naming_the_call(call, method):
    api, dll = _kernel32_power(**{call: 0})
    with pytest.raises(OSError, match=f"Windows refused {call}"):
        getattr(api, method)(0x2A4)
    assert dll.calls[0][:2] == (call, 0x2A4)


def test_a_handle_kernel32_refuses_to_set_is_closed():
    api, dll = _kernel32_power(PowerSetRequest=0)
    request = PowerRequest(api=api, monotonic=Clock())
    with pytest.raises(OSError, match="Windows refused PowerSetRequest"):
        request.hold()
    assert not request.held
    assert [call[0] for call in dll.calls] == ["PowerCreateRequest", "PowerSetRequest", "CloseHandle"]


@pytest.mark.skipif(sys.platform == "win32", reason="elsewhere only")
def test_the_real_calls_are_refused_in_words_off_windows():
    request = PowerRequest()
    with pytest.raises(OSError, match="Windows"):
        request.hold()
    assert not request.held


@pytest.mark.skipif(sys.platform != "win32", reason="Windows' own calls")
def test_the_real_power_request_is_taken_and_let_go_at_once():
    # The one real call: set and released in the same breath, no power
    # setting read or changed.
    request = PowerRequest("AuraDream test: released at once")
    try:
        request.hold()
        assert request.held
    finally:
        request.release()
    assert not request.held
    request.release()
