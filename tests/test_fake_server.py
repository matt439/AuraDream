"""Cold fake-server argument/data preparation; never start the application."""
from __future__ import annotations

import builtins
import math
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _tool(name, monkeypatch):
    path = ROOT / "tools" / f"{name}.py"
    module = ModuleType(f"_fake_server_test_{name}")
    module.__file__ = str(path)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    # fake_server adds the checkout to sys.path. Keep that change test-local;
    # compile the real source directly so loading it cannot write bytecode.
    monkeypatch.setattr(sys, "path", list(sys.path))
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), module.__dict__)
    return module


@pytest.fixture
def fake_server(monkeypatch):
    return _tool("fake_server", monkeypatch)


def test_fresh_allocates_distinct_empty_retained_directories(fake_server, monkeypatch, tmp_path):
    monkeypatch.setattr(fake_server.tempfile, "tempdir", str(tmp_path))
    shared = tmp_path / "auradream-fake"
    explicit = tmp_path / "chosen" / "nested"
    shared.mkdir()
    explicit.mkdir(parents=True)
    (shared / "keep.txt").write_bytes(b"shared gallery")
    (explicit / "keep.txt").write_bytes(b"chosen gallery")

    args = fake_server._parse_args(["--fresh"])
    first = fake_server._data_directory(args)
    assert first.parent == tmp_path and first.name.startswith("auradream-fake-")
    assert list(first.iterdir()) == []
    (first / "keep.txt").write_bytes(b"first fresh gallery")

    second = fake_server._data_directory(args)
    assert second.parent == tmp_path and second.name.startswith("auradream-fake-")
    assert second != first and list(second.iterdir()) == []
    assert (first / "keep.txt").read_bytes() == b"first fresh gallery"
    assert (shared / "keep.txt").read_bytes() == b"shared gallery"
    assert (explicit / "keep.txt").read_bytes() == b"chosen gallery"


@pytest.mark.parametrize("selection", ["default", "explicit", "relative", "empty", "missing"])
def test_nonfresh_selection_preserves_paths_without_premature_creation(
    fake_server, monkeypatch, tmp_path, selection,
):
    monkeypatch.setattr(fake_server.tempfile, "tempdir", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    shared = tmp_path / "auradream-fake"
    explicit = tmp_path / "chosen" / "nested"
    shared.mkdir()
    explicit.mkdir(parents=True)
    (shared / "keep.txt").write_bytes(b"shared gallery")
    (explicit / "keep.txt").write_bytes(b"chosen gallery")
    cases = {
        "default": ([], shared),
        "explicit": (["--data", str(explicit)], explicit),
        "relative": (["--data", "chosen/nested"], Path("chosen/nested")),
        "empty": (["--data", ""], Path("")),
        "missing": (["--data", "must-not-create"], Path("must-not-create")),
    }
    argv, expected = cases[selection]

    def forbidden(*_args, **_kwargs):
        pytest.fail("nonfresh path selection attempted filesystem creation")

    with monkeypatch.context() as guard:
        guard.setattr(fake_server.tempfile, "mkdtemp", forbidden)
        guard.setattr(Path, "mkdir", forbidden)
        args = fake_server._parse_args(argv)
        assert not args.fresh
        assert fake_server._data_directory(args) == expected
    assert not (tmp_path / "must-not-create").exists()
    assert (shared / "keep.txt").read_bytes() == b"shared gallery"
    assert (explicit / "keep.txt").read_bytes() == b"chosen gallery"


@pytest.mark.parametrize("fresh_first", [False, True])
def test_both_flag_orders_refuse_in_main_before_data_environment_or_backend_work(
    fake_server, monkeypatch, tmp_path, capsys, fresh_first,
):
    selected = tmp_path / "chosen" / "nested"
    selected.mkdir(parents=True)
    sentinel = selected / "keep.txt"
    sentinel.write_bytes(b"existing gallery")
    argv = ["--fresh", "--data", str(selected)] if fresh_first else ["--data", str(selected), "--fresh"]

    def forbidden(*_args, **_kwargs):
        pytest.fail("conflicting arguments reached data/environment/backend preparation")

    class NoEnvironmentWrites(dict):
        def __setitem__(self, _key, _value):
            forbidden()

    original_import = builtins.__import__

    def cold_import(name, *args, **kwargs):
        if name.split(".", 1)[0] in {"PIL", "backend", "uvicorn"} or name == "tests.fakes":
            forbidden()
        return original_import(name, *args, **kwargs)

    with monkeypatch.context() as guard:
        guard.setattr(sys, "argv", ["fake_server.py", *argv])
        guard.setattr(fake_server, "_data_directory", forbidden)
        guard.setattr(fake_server.tempfile, "mkdtemp", forbidden)
        guard.setattr(fake_server.tempfile, "gettempdir", forbidden)
        guard.setattr(fake_server.os, "environ", NoEnvironmentWrites(fake_server.os.environ))
        guard.setattr(fake_server.os, "mkdir", forbidden)
        guard.setattr(fake_server.os, "rmdir", forbidden)
        guard.setattr(fake_server.os, "unlink", forbidden)
        for name in ("mkdir", "write_bytes", "write_text", "unlink", "rmdir"):
            guard.setattr(Path, name, forbidden)
        guard.setattr(builtins, "__import__", cold_import)
        with pytest.raises(SystemExit) as stopped:
            fake_server.main()
    assert stopped.value.code == 2
    output = capsys.readouterr()
    assert output.out == ""
    assert "--data" in output.err and "--fresh" in output.err and "not allowed with argument" in output.err
    assert sentinel.read_bytes() == b"existing gallery"
    assert sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*")) == [
        "chosen", "chosen/nested", "chosen/nested/keep.txt",
    ]


def test_fresh_reports_retained_path_before_first_application_import(fake_server, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(fake_server.tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["fake_server.py", "--fresh"])
    monkeypatch.setattr(fake_server.os, "environ", dict(fake_server.os.environ))
    original_import = builtins.__import__
    attempted = []

    class ApplicationBoundary(Exception):
        pass

    def cold_import(name, *args, **kwargs):
        if name.split(".", 1)[0] in {"PIL", "backend", "uvicorn"} or name == "tests.fakes":
            attempted.append(name)
            raise ApplicationBoundary
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", cold_import)
    with pytest.raises(ApplicationBoundary):
        fake_server.main()
    data = Path(fake_server.os.environ["AURADREAM_DB_PATH"]).parent
    assert attempted == ["PIL"]
    assert data.parent == tmp_path and data.is_dir()
    assert capsys.readouterr().out == f"fresh fake gallery retained: {data}\n"
    assert (data / "loras" / "demo-style.safetensors").is_file()
    assert not (data / "auradream.db").exists()


def test_existing_smoke_row_arguments_parse_without_fresh_or_creating_data(fake_server, monkeypatch, tmp_path):
    runner = _tool("run_smoke", monkeypatch)
    selected = tmp_path / "runner-owned-gallery"
    row = runner.Row(7862, "black-forest-labs/FLUX.1-Krea-dev", 0.3, ["smoke-flux"])
    argv = row.fake_args(7962, selected)
    assert "--fresh" not in argv
    args = fake_server._parse_args(argv)
    assert args.port == 7962 and args.model == row.model and args.step == 0.3
    assert args.data == str(selected) and not args.fresh
    assert fake_server._data_directory(args) == selected
    assert not selected.exists()


class _FactoryPoison:
    def __init__(self, calls):
        self.calls = calls

    def __call__(self):
        self.calls.append("original-factory")
        pytest.fail("fake credential guard called the original factory")

    def __getattr__(self, name):
        self.calls.append(f"original-factory-attribute:{name}")
        pytest.fail("fake credential guard inspected the original factory/cache")


def test_fake_installer_refuses_without_original_factory_cache_platform_or_dll(
    fake_server, monkeypatch,
):
    import ctypes

    calls = []
    credentials = ModuleType("backend.backup_credentials")
    credentials._credential_api = _FactoryPoison(calls)

    def forbidden(*_args, **_kwargs):
        calls.append("platform-or-provider")
        pytest.fail("fake credential installation reached a platform/provider boundary")

    credentials.supported_platform = forbidden
    backend = ModuleType("backend")
    backend.backup_credentials = credentials
    original_import = builtins.__import__

    def cold_import(name, *args, **kwargs):
        if name == "backend":
            return backend
        if name.startswith("backend.") or name == "uvicorn":
            forbidden()
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(ctypes, "WinDLL", forbidden, raising=False)
    monkeypatch.setattr(builtins, "__import__", cold_import)
    fake_server._install_credential_refusal()
    refusal = credentials._credential_api
    assert refusal is fake_server._refuse_credential_api
    for _ in range(2):
        with pytest.raises(RuntimeError, match="^Windows backup credentials are unavailable in the fake server[.]$"):
            refusal()
    fake_server._install_credential_refusal()
    assert credentials._credential_api is refusal
    assert calls == []


@pytest.mark.parametrize("operation", ["read", "setup"])
def test_fake_refusal_stops_real_adapter_before_any_provider(fake_server, monkeypatch, operation):
    from backend import backup_credentials

    calls = []
    refusal_calls = []
    fixed_refusal = fake_server._refuse_credential_api

    def counted_refusal():
        refusal_calls.append("installed-refusal")
        fixed_refusal()

    def forbidden(*_args, **_kwargs):
        calls.append("provider-boundary")
        pytest.fail("refused fake credential operation reached the original provider")

    monkeypatch.setattr(backup_credentials, "_credential_api", _FactoryPoison(calls))
    monkeypatch.setattr(backup_credentials, "supported_platform", forbidden)
    monkeypatch.setattr(backup_credentials, "_last_error", forbidden)
    monkeypatch.setattr(backup_credentials.ctypes, "WinDLL", forbidden, raising=False)
    monkeypatch.setattr(fake_server, "_refuse_credential_api", counted_refusal)
    fake_server._install_credential_refusal()
    assert backup_credentials._credential_api is fake_server._refuse_credential_api
    reference = backup_credentials.new_reference(
        "f78c9158-4e20-497a-8a0b-1b05f3f25784", "ebc02e7d-3c29-47d3-a093-d2a2332b7cd1",
    )
    synthetic_password = "synthetic-test-password-only"
    with pytest.raises(backup_credentials.CredentialError) as refused:
        if operation == "read":
            with backup_credentials.read_secret(reference):
                pytest.fail("fake credential read yielded a secret")
        else:
            backup_credentials.create_credential(reference, synthetic_password)
    assert refused.value.reason == "credential_unusable"
    assert str(refused.value) == backup_credentials._ERROR
    assert refused.value.__cause__ is None and refused.value.__context__ is None
    assert synthetic_password not in str(refused.value)
    assert "fake server" not in str(refused.value)
    assert backup_credentials._credential_api is fake_server._refuse_credential_api
    # _read deliberately catches BaseException; a poison raising by itself
    # would be masked as the expected fixed error. Independently check calls.
    assert calls == []
    assert refusal_calls == ["installed-refusal"]


def test_main_installs_persistent_refusal_after_pillow_before_application_import(
    fake_server, monkeypatch, tmp_path,
):
    import ctypes

    monkeypatch.setattr(fake_server.tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["fake_server.py", "--fresh"])
    monkeypatch.setattr(fake_server.os, "environ", {})
    calls = []
    credentials = ModuleType("backend.backup_credentials")
    credentials._credential_api = _FactoryPoison(calls)
    backend = ModuleType("backend")
    backend.backup_credentials = credentials
    pillow = ModuleType("PIL")
    pillow.Image = pillow.ImageDraw = object()
    original_import = builtins.__import__
    events = []

    class ApplicationBoundary(Exception):
        pass

    def forbidden(*_args, **_kwargs):
        calls.append("provider-or-application")
        pytest.fail("fake main boundary reached a real provider or later application import")

    credentials.supported_platform = forbidden

    def cold_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "PIL":
            assert tuple(fromlist) == ("Image", "ImageDraw")
            events.append("PIL")
            return pillow
        if name == "backend":
            if tuple(fromlist) == ("backup_credentials",):
                events.append("credentials")
                return backend
            if tuple(fromlist) == ("main",):
                events.append("main-boundary")
                assert credentials._credential_api is fake_server._refuse_credential_api
                with pytest.raises(RuntimeError, match="^Windows backup credentials are unavailable in the fake server[.]$"):
                    credentials._credential_api()
                raise ApplicationBoundary
            forbidden()
        if name.startswith("backend.") or name == "uvicorn" or name == "tests.fakes":
            forbidden()
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(ctypes, "WinDLL", forbidden, raising=False)
    monkeypatch.setattr(builtins, "__import__", cold_import)
    with pytest.raises(ApplicationBoundary):
        fake_server.main()
    assert events == ["PIL", "credentials", "main-boundary"]
    assert credentials._credential_api is fake_server._refuse_credential_api
    with pytest.raises(RuntimeError, match="^Windows backup credentials are unavailable in the fake server[.]$"):
        credentials._credential_api()
    data = Path(fake_server.os.environ["AURADREAM_DB_PATH"]).parent
    assert data.parent == tmp_path and data.is_dir()
    assert not (data / "auradream.db").exists()
    assert calls == []


def _forbid_numeric_startup(fake_server, guard):
    """Count attempts as well as raising; no downstream masking can pass."""
    attempts = []

    def forbidden(*_args, **_kwargs):
        attempts.append("startup-effect")
        pytest.fail("numeric CLI validation reached data/environment/application/provider work")

    class NoEnvironmentWrites(dict):
        def __setitem__(self, _key, _value):
            forbidden()

    original_import = builtins.__import__

    def cold_import(name, *args, **kwargs):
        if name.split(".", 1)[0] in {"PIL", "backend", "uvicorn"} or name == "tests.fakes":
            forbidden()
        return original_import(name, *args, **kwargs)

    guard.setattr(fake_server, "_data_directory", forbidden)
    guard.setattr(fake_server, "_install_credential_refusal", forbidden)
    guard.setattr(fake_server.tempfile, "mkdtemp", forbidden)
    guard.setattr(fake_server.tempfile, "gettempdir", forbidden)
    guard.setattr(fake_server.os, "environ", NoEnvironmentWrites(fake_server.os.environ))
    guard.setattr(fake_server.os, "mkdir", forbidden)
    guard.setattr(fake_server.os, "rmdir", forbidden)
    guard.setattr(fake_server.os, "unlink", forbidden)
    for name in ("mkdir", "write_bytes", "write_text", "unlink", "rmdir"):
        guard.setattr(Path, name, forbidden)
    guard.setattr(builtins, "__import__", cold_import)
    return attempts


@pytest.mark.parametrize("data_selection", ["fresh", "explicit"])
@pytest.mark.parametrize(("option", "value", "message"), [
    ("--port", "-1", "must be an integer from 0 to 65535"),
    ("--port", "65536", "must be an integer from 0 to 65535"),
    ("--port", "1.5", "must be an integer from 0 to 65535"),
    *[
        (option, value, "must be finite seconds at least zero")
        for option in ("--step", "--critic-delay")
        for value in ("-0.25", "nan", "inf", "-inf", "1e309")
    ],
])
def test_invalid_numeric_cli_refuses_before_data_or_application_work(
    fake_server, monkeypatch, tmp_path, capsys, data_selection, option, value, message,
):
    selected = tmp_path / "chosen" / "nested"
    selected.mkdir(parents=True)
    sentinel = selected / "keep.txt"
    sentinel.write_bytes(b"existing chosen gallery")
    data_args = ["--fresh"] if data_selection == "fresh" else ["--data", str(selected)]
    argv = ["fake_server.py", *data_args, f"{option}={value}"]
    environment_before = dict(fake_server.os.environ)
    with monkeypatch.context() as guard:
        guard.setattr(sys, "argv", argv)
        attempts = _forbid_numeric_startup(fake_server, guard)
        with pytest.raises(SystemExit) as stopped:
            fake_server.main()
        assert attempts == []
        assert dict(fake_server.os.environ) == environment_before
    assert stopped.value.code == 2
    output = capsys.readouterr()
    assert output.out == ""
    assert f"argument {option}: {message}" in output.err
    assert sentinel.read_bytes() == b"existing chosen gallery"
    assert sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*")) == [
        "chosen", "chosen/nested", "chosen/nested/keep.txt",
    ]


@pytest.mark.parametrize(("value", "expected"), [
    ("0", 0), ("1", 1), ("65535", 65535), ("+7861", 7861), (" 7861 ", 7861),
])
def test_valid_cli_ports_preserve_zero_boundaries_and_integer_spellings(
    fake_server, monkeypatch, value, expected,
):
    with monkeypatch.context() as guard:
        attempts = _forbid_numeric_startup(fake_server, guard)
        args = fake_server._parse_args([f"--port={value}"])
        assert attempts == []
    assert args.port == expected
    assert args.step == 0.0 and args.critic_delay == 0.4


@pytest.mark.parametrize("option", ["--step", "--critic-delay"])
@pytest.mark.parametrize("value", ["0", "-0.0", "0.3", "1e-300", "1e100"])
def test_valid_cli_delays_preserve_zero_sign_and_positive_values(
    fake_server, monkeypatch, option, value,
):
    with monkeypatch.context() as guard:
        attempts = _forbid_numeric_startup(fake_server, guard)
        args = fake_server._parse_args([f"{option}={value}"])
        assert attempts == []
    actual = args.step if option == "--step" else args.critic_delay
    assert actual == float(value)
    assert math.copysign(1, actual) == math.copysign(1, float(value))
    assert args.port == 7861


@pytest.mark.parametrize(("argv", "expected"), [
    (["--port", "0", "--step", "-0.0", "--critic-delay", "0"], (0, -0.0, 0.0)),
    (["--port", "65535", "--step", "0.3", "--critic-delay", "-0.0"], (65535, 0.3, -0.0)),
    (["--port", "1", "--step", "1e100", "--critic-delay", "1e-300"], (1, 1e100, 1e-300)),
    ([], (7861, 0.0, 0.4)),
])
def test_valid_numeric_cli_reaches_the_existing_data_preparation_boundary_unchanged(
    fake_server, monkeypatch, argv, expected,
):
    observed = []

    class PreparationBoundary(Exception):
        pass

    def stop_at_data(args):
        observed.append((args.port, args.step, args.critic_delay))
        raise PreparationBoundary

    with monkeypatch.context() as guard:
        guard.setattr(sys, "argv", ["fake_server.py", *argv])
        attempts = _forbid_numeric_startup(fake_server, guard)
        guard.setattr(fake_server, "_data_directory", stop_at_data)
        with pytest.raises(PreparationBoundary):
            fake_server.main()
        assert attempts == []
    assert observed == [expected]
    assert math.copysign(1, observed[0][1]) == math.copysign(1, expected[1])
    assert math.copysign(1, observed[0][2]) == math.copysign(1, expected[2])
