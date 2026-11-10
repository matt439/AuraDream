"""The server's log file: `auradream.log` beside the gallery database.

The console's log is all there was, and a failure's traceback went to it
only at DEBUG. On Windows a job that failed overnight left nothing once the
console window closed. `install` adds one rotating file that receives what
the console receives and, at any configured level, every record carrying a
traceback; a fatal error in native code (`faulthandler`) is written to the
same file. The console prints what it printed before.

The file holds prompts and paths. It is not gallery data: no bundle, export,
backup or `clear` reads, copies or removes it. It always takes the new name
(nothing made it before the rename), and it is opened by `main.run` alone,
never at import: the CLIs, the tests and the fake server write none.

Stdlib only.
"""
from __future__ import annotations

import faulthandler
import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

from .legacy_names import PREFIX

FORMAT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"
# The file and three older ones (`auradream.log.1` the newest): 20 MB at most.
MAX_BYTES = 5 * 1024 * 1024
BACKUPS = 3
# The app's own loggers: `auradream` and its children, and each module's
# `backend.<name>`. Only these are opened to DEBUG for the file's tracebacks;
# a library's DEBUG stays where the configured level puts it.
APP_LOGGERS = ("auradream", "backend")
# uvicorn's logger for the server itself and an unhandled route's traceback
# (`uvicorn.error` propagates to it). `uvicorn.access`, a line per request,
# is not followed: the page's polls would fill the file.
UVICORN_LOGGER = "uvicorn"
_UVICORN_LEVELS = {
    "CRITICAL": "critical", "FATAL": "critical", "ERROR": "error", "WARNING": "warning",
    "WARN": "warning", "INFO": "info", "DEBUG": "debug", "NOTSET": "debug",
}

log = logging.getLogger("auradream")


def path_for(database: Path) -> Path:
    """`auradream.log` in `database`'s directory."""
    return Path(database).expanduser().parent / f"{PREFIX}.log"


def uvicorn_level(level: str) -> str:
    """uvicorn's name for the configured logging level (`AURADREAM_LOG_LEVEL`).
    uvicorn takes lower-case names and knows neither WARN nor FATAL."""
    return _UVICORN_LEVELS.get(level.upper(), "info")


class _AtLevelOrTraceback(logging.Filter):
    """Pass what the console's level passes, and any record with a traceback."""

    def __init__(self, level: int) -> None:
        super().__init__()
        self.level = level

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno >= self.level or bool(record.exc_info)


class _Handler(RotatingFileHandler):
    """The rotating file, with `faulthandler` kept on the current file.

    `faulthandler` writes to a file descriptor, and a rollover closes the one
    it holds, so it is turned off across the rename and on again after it.
    The handler's own stream is the only handle on the file: a second one
    would make the rename fail on Windows.
    """

    owns_faulthandler = False

    def doRollover(self) -> None:  # noqa: N802 - logging's name
        if self.owns_faulthandler:
            faulthandler.disable()
        if self.stream is not None:
            self.stream.close()
            # As the stdlib's rollover does; its stubs say never None.
            self.stream = None  # pyright: ignore[reportAttributeAccessIssue]
        try:
            self._move_files()
        except OSError:
            # Windows refuses a rename while another program holds the file
            # open (a console following it). Keep appending to the live
            # file; the next record tries again.
            pass
        try:
            self.stream = self._open()
        finally:
            if self.owns_faulthandler and self.stream is not None:
                faulthandler.enable(file=self.stream, all_threads=True)

    def _move_files(self) -> None:
        """The live file becomes `.1` and each older file one older.

        The stdlib moves the older files first, so a refused rename of the
        live file, tried again at every record, pushed two of the three
        older files out and kept nothing in their place. Here the live file
        moves first, to a staging name, and nothing else has changed when
        that is refused. A staged file a refused later step left behind is
        finished first, the live file waiting for the next record.
        """
        base = self.baseFilename
        staged = f"{base}.0"
        if not os.path.exists(staged):
            os.replace(base, staged)
        for index in range(self.backupCount - 1, 0, -1):
            older = f"{base}.{index}"
            if os.path.exists(older):
                os.replace(older, f"{base}.{index + 1}")
        os.replace(staged, f"{base}.1")


class Installed:
    """What `install` changed, undone by `close()` (idempotent)."""

    def __init__(self, path: Path, handler: _Handler | None) -> None:
        self.path = path
        self.handler = handler
        self._loggers: list[logging.Logger] = []
        self._levels: list[tuple[logging.Logger | logging.Handler, int]] = []

    def follow(self, logger: logging.Logger) -> None:
        """Send `logger`'s records to the file too. For a logger that does not
        propagate to the root: uvicorn's, once `uvicorn.Config` has set up its
        logging (which replaces that logger's handlers)."""
        if self.handler is not None and self.handler not in logger.handlers:
            logger.addHandler(self.handler)
            self._loggers.append(logger)

    def _set_level(self, target: logging.Logger | logging.Handler, level: int) -> None:
        self._levels.append((target, target.level))
        target.setLevel(level)

    def close(self) -> None:
        handler, self.handler = self.handler, None
        if handler is None:
            return
        for target, level in reversed(self._levels):
            target.setLevel(level)
        for logger in self._loggers:
            logger.removeHandler(handler)
        if handler.owns_faulthandler:
            faulthandler.disable()
            handler.owns_faulthandler = False
        handler.close()


def install(database: Path, level: str) -> Installed:
    """Open the log file beside `database` and send it the root logger's
    records at `level` and above, plus every record with a traceback.

    The app's loggers are opened to DEBUG so their DEBUG tracebacks exist to
    be written; the console's handlers are held at `level`, so the console
    shows what it showed. `faulthandler` is pointed at the file unless
    something (`-X faulthandler`, `PYTHONFAULTHANDLER`) already enabled it,
    which is left as its owner set it.

    A file that cannot be opened is a warning on the console, never a server
    that does not start: the answer then has no handler and changes nothing.
    """
    path = path_for(database)
    number = logging.getLevelNamesMapping().get(level.upper(), logging.INFO)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = _Handler(path, maxBytes=MAX_BYTES, backupCount=BACKUPS, encoding="utf-8")
    except OSError as exc:
        log.warning("no log file: %s could not be opened (%s)", path, exc)
        return Installed(path, None)
    handler.setFormatter(logging.Formatter(FORMAT))
    handler.addFilter(_AtLevelOrTraceback(number))
    installed = Installed(path, handler)
    root = logging.getLogger()
    for console in root.handlers:
        if console.level == logging.NOTSET:
            # It relied on the loggers' level, which is about to open.
            installed._set_level(console, number)
    for name in APP_LOGGERS:
        logger = logging.getLogger(name)
        if logger.getEffectiveLevel() > logging.DEBUG:
            installed._set_level(logger, logging.DEBUG)
    installed.follow(root)
    if not faulthandler.is_enabled():
        faulthandler.enable(file=handler.stream, all_threads=True)
        handler.owns_faulthandler = True
    return installed
