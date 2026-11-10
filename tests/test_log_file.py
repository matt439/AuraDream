"""The server's log file (`backend/log_file.py`) and `main.run`'s use of it.

The root logger is the process's, so every test here puts back the levels
and handlers it found; `log_file.Installed.close()` is what does it.
"""
from __future__ import annotations

import dataclasses
import io
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest

from backend import log_file, main

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def console():
    """A console handler as `basicConfig` leaves one (no level of its own),
    with the root and the app's loggers at INFO; all put back afterwards."""
    root = logging.getLogger()
    before = [(logger, logger.level) for logger in (
        root, *(logging.getLogger(name) for name in log_file.APP_LOGGERS)
    )]
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    for name in log_file.APP_LOGGERS:
        logging.getLogger(name).setLevel(logging.NOTSET)
    yield stream
    root.removeHandler(handler)
    for logger, level in before:
        logger.setLevel(level)


def _fail(logger: logging.Logger, words: str) -> None:
    """A failure logged the way the job, cell, load and switch sites log
    theirs: one ERROR line, the traceback at DEBUG."""
    try:
        raise RuntimeError(words)
    except RuntimeError as exc:
        logger.error("job failed: %s", exc)
        logger.debug("job traceback", exc_info=True)


def test_the_file_sits_beside_the_database_under_the_new_name(tmp_path):
    assert log_file.path_for(tmp_path / "gallery.db") == tmp_path / "auradream.log"
    # A gallery kept under the old name still gets the new one: nothing wrote
    # a log before the rename, so there is no old file to keep using.
    (tmp_path / "local-diffusion.log").write_text("not ours")
    assert log_file.path_for(tmp_path / "local-diffusion.db").name == "auradream.log"


@pytest.mark.parametrize("name", ["auradream", "backend.jobs", "auradream.keep_awake"])
def test_a_traceback_at_debug_reaches_the_file_and_not_the_console(tmp_path, console, name):
    installed = log_file.install(tmp_path / "auradream.db", "INFO")
    try:
        logger = logging.getLogger(name)
        _fail(logger, "out of memory")
        logger.debug("a dropped pipeline argument")
        logger.info("job queued")
    finally:
        installed.close()
    written = installed.path.read_text(encoding="utf-8")
    assert "job failed: out of memory" in written and "job queued" in written
    assert "Traceback (most recent call last)" in written
    assert "RuntimeError: out of memory" in written
    # DEBUG without a traceback is still DEBUG's alone.
    assert "a dropped pipeline argument" not in written
    # The console: what INFO printed before, nothing more.
    assert console.getvalue() == "ERROR job failed: out of memory\nINFO job queued\n"


def test_a_librarys_debug_stays_out_of_both(tmp_path, console):
    installed = log_file.install(tmp_path / "auradream.db", "INFO")
    try:
        logging.getLogger("PIL.PngImagePlugin").debug("STREAM b'IHDR' 16 13")
    finally:
        installed.close()
    assert installed.path.read_text(encoding="utf-8") == "" and console.getvalue() == ""


def test_at_debug_the_file_has_what_the_console_has(tmp_path, console):
    logging.getLogger().setLevel(logging.DEBUG)
    installed = log_file.install(tmp_path / "auradream.db", "DEBUG")
    try:
        logging.getLogger("backend.jobs").debug("a dropped pipeline argument")
    finally:
        installed.close()
    assert "a dropped pipeline argument" in installed.path.read_text(encoding="utf-8")
    assert "a dropped pipeline argument" in console.getvalue()


def test_close_puts_the_loggers_back_and_lets_the_file_go(tmp_path, console):
    root = logging.getLogger()
    handlers = list(root.handlers)
    levels = [handler.level for handler in handlers]
    app = [logging.getLogger(name).level for name in log_file.APP_LOGGERS]
    installed = log_file.install(tmp_path / "auradream.db", "INFO")
    assert installed.handler in root.handlers
    assert logging.getLogger("backend.jobs").isEnabledFor(logging.DEBUG)
    installed.close()
    installed.close()  # idempotent
    assert root.handlers == handlers and [handler.level for handler in handlers] == levels
    assert [logging.getLogger(name).level for name in log_file.APP_LOGGERS] == app
    logging.getLogger("auradream").error("after close")
    assert "after close" not in installed.path.read_text(encoding="utf-8")
    installed.path.unlink()  # Windows: no handle left on it


def test_the_file_rotates_and_keeps_only_its_backups(tmp_path, console, monkeypatch):
    monkeypatch.setattr(log_file, "MAX_BYTES", 2000)
    installed = log_file.install(tmp_path / "auradream.db", "INFO")
    try:
        for index in range(400):
            logging.getLogger("auradream").info("line %04d %s", index, "x" * 60)
    finally:
        installed.close()
    names = sorted(path.name for path in tmp_path.iterdir())
    assert names == ["auradream.log", *(f"auradream.log.{n}" for n in range(1, log_file.BACKUPS + 1))]
    assert all(path.stat().st_size <= 2000 for path in tmp_path.iterdir())
    assert "line 0399" in installed.path.read_text(encoding="utf-8")


def _refuse_moving(monkeypatch, held: set[str]):
    """Windows refuses the rename of a file another program holds open (a
    console following the log). `held` names the files so held."""
    replace = log_file.os.replace

    def refuse(source, dest):
        if Path(source).name in held:
            raise PermissionError(13, "The process cannot access the file", str(source))
        replace(source, dest)

    monkeypatch.setattr(log_file.os, "replace", refuse)


def _fill(tmp_path, first: int, last: int) -> None:
    for index in range(first, last):
        logging.getLogger("auradream").info("line %04d %s", index, "x" * 60)


def _first_lines(tmp_path) -> dict[str, str]:
    return {
        path.name: path.read_text(encoding="utf-8").splitlines()[0].split("| ")[1][:9]
        for path in sorted(tmp_path.iterdir())
    }


def test_a_refused_rollover_keeps_writing_and_keeps_the_older_files(tmp_path, console, monkeypatch, capsys):
    monkeypatch.setattr(log_file, "MAX_BYTES", 500)
    installed = log_file.install(tmp_path / "auradream.db", "INFO")
    try:
        _fill(tmp_path, 0, 40)
        before = _first_lines(tmp_path)
        assert sorted(before) == ["auradream.log", "auradream.log.1", "auradream.log.2", "auradream.log.3"]
        held = {"auradream.log"}
        _refuse_moving(monkeypatch, held)
        _fill(tmp_path, 40, 80)  # a rollover refused at every one of them
        # Nothing moved: the older files are the ones that were there. (The
        # stdlib's order pushed two of the three out, one per refusal.)
        assert _first_lines(tmp_path) == before
        written = installed.path.read_text(encoding="utf-8")
        assert "line 0040" in written and "line 0079" in written
        held.clear()
        _fill(tmp_path, 80, 81)
        after = _first_lines(tmp_path)
        assert after["auradream.log"] == "line 0080"
        assert after["auradream.log.1"] == before["auradream.log"]
        assert after["auradream.log.2"] == before["auradream.log.1"]
        assert after["auradream.log.3"] == before["auradream.log.2"]
        assert "line 0079" in (tmp_path / "auradream.log.1").read_text(encoding="utf-8")
    finally:
        installed.close()
    # logging reports a handler's own failure on stderr; this is not one.
    assert "Logging error" not in capsys.readouterr().err


def test_a_held_older_file_loses_no_lines(tmp_path, console, monkeypatch, capsys):
    monkeypatch.setattr(log_file, "MAX_BYTES", 500)
    installed = log_file.install(tmp_path / "auradream.db", "INFO")
    try:
        _fill(tmp_path, 0, 40)
        held = {"auradream.log.1"}
        _refuse_moving(monkeypatch, held)
        _fill(tmp_path, 40, 60)
        # The live file was staged and a new one started; the staged one
        # waits, and is not overwritten by the rollovers refused after it.
        assert (tmp_path / "auradream.log.0").exists()
        staged = (tmp_path / "auradream.log.0").read_text(encoding="utf-8")
        held.clear()
        _fill(tmp_path, 60, 61)
    finally:
        installed.close()
    # The staged file took its place as the newest older one, whole, and the
    # live file kept every line written while it waited.
    assert not (tmp_path / "auradream.log.0").exists()
    assert (tmp_path / "auradream.log.1").read_text(encoding="utf-8") == staged
    written = installed.path.read_text(encoding="utf-8")
    assert all(f"line {index:04d}" in staged + written for index in range(40, 61))
    assert "Logging error" not in capsys.readouterr().err


def test_a_file_that_cannot_be_opened_is_a_warning_not_a_stopped_server(tmp_path, console):
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("")
    root = logging.getLogger()
    handlers = list(root.handlers)
    installed = log_file.install(blocker / "auradream.db", "INFO")
    assert installed.handler is None and root.handlers == handlers
    assert "no log file" in console.getvalue()
    # Nothing was opened to DEBUG for a file that is not there.
    assert not logging.getLogger("backend.jobs").isEnabledFor(logging.DEBUG)
    installed.follow(logging.getLogger("uvicorn"))
    installed.close()


@pytest.mark.parametrize("level, expected", [
    ("INFO", "info"), ("DEBUG", "debug"), ("WARNING", "warning"), ("WARN", "warning"),
    ("ERROR", "error"), ("CRITICAL", "critical"), ("FATAL", "critical"), ("NOTSET", "debug"),
    ("debug", "debug"),
])
def test_uvicorn_is_given_its_own_name_for_the_level(level, expected):
    import uvicorn.config

    assert log_file.uvicorn_level(level) == expected
    assert expected in uvicorn.config.LOG_LEVELS


def test_a_fatal_error_is_written_to_the_file_across_a_rollover(tmp_path):
    # In a process of its own: pytest owns this one's faulthandler, and the
    # process has to die. SIGABRT is one of the signals faulthandler dumps on.
    script = textwrap.dedent("""
        import faulthandler, logging, signal, sys
        from pathlib import Path
        from backend import log_file

        assert not faulthandler.is_enabled()
        log_file.MAX_BYTES = 300
        installed = log_file.install(Path(sys.argv[1]) / "auradream.db", "INFO")
        assert faulthandler.is_enabled()
        for index in range(20):
            logging.getLogger("auradream").warning("line %04d %s", index, "x" * 60)
        assert Path(sys.argv[1], "auradream.log.1").exists()
        signal.raise_signal(signal.SIGABRT)
    """)
    environment = {
        key: value for key, value in __import__("os").environ.items()
        if key != "PYTHONFAULTHANDLER"
    }
    result = subprocess.run(
        [sys.executable, "-B", "-c", script, str(tmp_path)], cwd=ROOT, env=environment,
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode != 0 and "AssertionError" not in result.stderr, result.stderr
    written = (tmp_path / "auradream.log").read_text(encoding="utf-8")
    assert "Fatal Python error: Aborted" in written and "line 0019" in written
    # The console is not where it went.
    assert "Fatal Python error" not in result.stderr


def test_an_enabled_faulthandler_is_left_to_its_owner(tmp_path, console):
    import faulthandler

    if not faulthandler.is_enabled():
        pytest.skip("pytest's faulthandler plugin is off")
    installed = log_file.install(tmp_path / "auradream.db", "INFO")
    try:
        assert installed.handler is not None and not installed.handler.owns_faulthandler
    finally:
        installed.close()
    assert faulthandler.is_enabled()


def test_importing_the_server_opens_no_file():
    assert not any(isinstance(handler, RotatingFileHandler) for handler in logging.getLogger().handlers)


def _recording_server(monkeypatch, seen: dict):
    class Recorded:
        should_exit = False
        started = True

        def __init__(self, config):
            seen["config"] = config

        def run(self):
            seen["root"] = [h for h in logging.getLogger().handlers if isinstance(h, RotatingFileHandler)]
            seen["uvicorn"] = [
                h for h in logging.getLogger("uvicorn").handlers if isinstance(h, RotatingFileHandler)
            ]
            main.log.info("serving")
            _fail(logging.getLogger("backend.jobs"), "cell 3 ran out of memory")
            try:
                raise ValueError("an unhandled route")
            except ValueError:
                logging.getLogger("uvicorn.error").exception("Exception in ASGI application")
            logging.getLogger("uvicorn.access").info('127.0.0.1 - "GET /api/status HTTP/1.1" 200')

    monkeypatch.setattr(main, "Server", Recorded)


@pytest.mark.parametrize("level, uvicorn_level", [("INFO", "info"), ("DEBUG", "debug"), ("WARNING", "warning")])
def test_run_writes_the_file_beside_the_database_and_uvicorn_follows_the_level(
    fake, monkeypatch, console, level, uvicorn_level,
):
    seen: dict = {}
    _recording_server(monkeypatch, seen)
    monkeypatch.setattr(main, "settings", dataclasses.replace(main.settings, port=0, log_level=level))
    uvicorn_handlers = list(logging.getLogger("uvicorn").handlers)
    try:
        main.run()
        # uvicorn's level was the hard-coded "info" whatever was configured.
        assert seen["config"].log_level == uvicorn_level
        assert len(seen["root"]) == 1 and seen["uvicorn"] == seen["root"]
        path = main.settings.db_path.parent / "auradream.log"
        written = path.read_text(encoding="utf-8")
        assert "RuntimeError: cell 3 ran out of memory" in written
        assert "ValueError: an unhandled route" in written
        assert ("serving" in written) == (level != "WARNING")
        assert "/api/status" not in written  # a line per poll is not followed
        # Closed when the server stops: the next record goes nowhere new.
        assert not any(isinstance(h, RotatingFileHandler) for h in logging.getLogger().handlers)
        assert not any(isinstance(h, RotatingFileHandler) for h in logging.getLogger("uvicorn").handlers)
    finally:
        # uvicorn.Config configured the process's uvicorn loggers; put the
        # handlers back as they were found.
        logging.getLogger("uvicorn").handlers[:] = uvicorn_handlers
    if level == "INFO":
        assert "cell 3 ran out of memory" in console.getvalue()
        assert "Traceback" not in console.getvalue()


def test_a_second_launch_opens_no_file(fake, monkeypatch):
    monkeypatch.setattr(main, "settings", dataclasses.replace(main.settings, port=7860))
    monkeypatch.setattr(main, "_port_busy", lambda host, port: True)
    monkeypatch.setattr(main, "_auradream_at", lambda url: True)
    with pytest.raises(SystemExit):
        main.run()
    assert not (main.settings.db_path.parent / "auradream.log").exists()


def test_the_log_is_not_gallery_data(fake):
    # No bundle, backup or clear names it; what they take is the database's
    # rows and the files those rows track.
    path = log_file.path_for(main.settings.db_path)
    assert path.parent == main.settings.db_path.parent and path.suffix == ".log"
    sources = [
        "bundles.py", "bundle_archive.py", "bundle_images.py", "bundle_manager.py",
        "backup_coordinator.py", "gallery_delete.py", "storage.py", "clear.py",
    ]
    for name in sources:
        text = (ROOT / "backend" / name).read_text(encoding="utf-8")
        assert "log_file" not in text and ".log" not in text, name
