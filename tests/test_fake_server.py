"""Cold fake-server argument/data preparation; never start the application."""
from __future__ import annotations

import builtins
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
