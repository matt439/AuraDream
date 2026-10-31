"""The browser's remembered loader choice, independent of frozen config imports.

Only loader/startup preferences are persisted, beside the gallery database. The
server launcher reads them before importing config; explicit environment and
command-line values still take precedence. An unreadable execution preference
keeps the browser open but blocks loading/rendering until explicitly repaired.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
from collections.abc import MutableMapping
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .legacy_names import default_db_path, sidecar
from .execution import INVALID_LOADER_CONFIGURATION, ExecutionSpec, canonical_loader_execution
from .model_defaults import OFFLOAD_MODES, QUANT_MODES

log = logging.getLogger(__name__)
_VERSION = 2
_MAX_BYTES = 64 * 1024
_ROOT = Path(__file__).resolve().parent.parent
LOADER_EXECUTION_ERROR_VARIABLE = "AURADREAM_LOADER_EXECUTION_ERROR"
LOADER_EXECUTION_ERROR = INVALID_LOADER_CONFIGURATION


@dataclass(frozen=True)
class StartupPreferences:
    model_id: str
    quantization: str
    offload: str
    load_on_startup: bool
    loader_execution: ExecutionSpec | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.model_id, str)
            or not self.model_id.strip()
            or self.model_id != self.model_id.strip()
            or len(self.model_id) > 4096
            or any(ord(char) < 32 for char in self.model_id)
        ):
            raise ValueError("model_id must be a nonempty checkpoint id or local path")
        if self.quantization not in QUANT_MODES:
            raise ValueError("unknown quantization")
        if self.offload not in OFFLOAD_MODES:
            raise ValueError("unknown offload strategy")
        if type(self.load_on_startup) is not bool:
            raise ValueError("load_on_startup must be a boolean")
        object.__setattr__(self, "loader_execution", canonical_loader_execution(self.loader_execution))

    def environ(self) -> dict[str, str]:
        values = {
            "AURADREAM_MODEL_ID": self.model_id,
            "AURADREAM_QUANT": self.quantization,
            "AURADREAM_OFFLOAD": self.offload,
            "AURADREAM_PRELOAD": "true" if self.load_on_startup else "false",
        }
        if self.loader_execution is not None:
            values.update(
                AURADREAM_COMPILATION=self.loader_execution.compilation or "eager",
                AURADREAM_ATTENTION=self.loader_execution.attention_backend or "native",
            )
        return values


def path_for(db_path: Path) -> Path:
    # An existing local-diffusion.settings.json is kept and rewritten in place.
    return sidecar(db_path, "settings.json")


@dataclass(frozen=True)
class PreferenceSnapshot:
    """Exact bounded prior file bytes, or absence; never a parsed fallback."""

    data: bytes | None = field(repr=False)

    def __post_init__(self) -> None:
        if self.data is not None and (not isinstance(self.data, bytes) or len(self.data) > _MAX_BYTES):
            raise ValueError("Startup preference snapshot is invalid or exceeds its byte bound")


def capture_snapshot(db_path: Path) -> PreferenceSnapshot:
    """Capture before saving; unreadable/oversized data refuses any overwrite."""
    try:
        with path_for(db_path).open("rb") as stream:
            data = stream.read(_MAX_BYTES + 1)
    except FileNotFoundError:
        return PreferenceSnapshot(None)
    return PreferenceSnapshot(data)


def restore_snapshot(db_path: Path, snapshot: PreferenceSnapshot) -> None:
    """Restore exact bytes/absence using the existing switch transaction scope."""
    if not isinstance(snapshot, PreferenceSnapshot):
        raise ValueError("Startup preference snapshot is invalid")
    checked = PreferenceSnapshot(snapshot.data)
    if checked.data is None:
        path_for(db_path).unlink(missing_ok=True)
    else:
        _write_bytes(db_path, checked.data)


@dataclass(frozen=True)
class PreferenceLoad:
    preferences: StartupPreferences | None
    loader_execution_error: str | None = None


def _unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate startup setting")
        result[key] = value
    return result


def _constant(_value: str) -> None:
    raise ValueError("nonfinite startup setting")


def load_result(db_path: Path) -> PreferenceLoad:
    """Read pure metadata; never mutate environment or rewrite an invalid file."""
    path = path_for(db_path)
    payload: object = None
    try:
        with path.open("rb") as stream:
            data = stream.read(_MAX_BYTES + 1)
        if len(data) > _MAX_BYTES:
            raise ValueError("startup preferences exceed their byte bound")
        payload = json.loads(data.decode("utf-8"), object_pairs_hook=_unique, parse_constant=_constant)
        if not isinstance(payload, dict):
            raise ValueError("settings must be a JSON object")
        if type(payload.get("version")) is not int or payload["version"] not in (1, _VERSION):
            raise ValueError("unsupported settings version")
        expected = {"version", "model_id", "quantization", "offload", "load_on_startup"}
        if payload["version"] == 2:
            expected.add("loader_execution")
        if payload.keys() != expected:
            raise ValueError("settings fields do not match the known schema")
        values = {key: value for key, value in payload.items() if key != "version"}
        return PreferenceLoad(StartupPreferences(**values))
    except FileNotFoundError:
        return PreferenceLoad(None)
    except (OSError, UnicodeError, ValueError, TypeError, RecursionError, OverflowError):
        log.warning("Ignoring saved startup settings at %s: unreadable or unsupported preferences", path)
        legacy = (isinstance(payload, dict) and type(payload.get("version")) is int and payload["version"] == 1
                  and not payload.keys() & {"loader_execution", "execution", "compilation", "attention_backend", "step_cache"})
        return PreferenceLoad(None, None if legacy else LOADER_EXECUTION_ERROR)


def load(db_path: Path) -> StartupPreferences | None:
    """Compatibility read; load_result also exposes fail-closed diagnostics."""
    return load_result(db_path).preferences


def save(db_path: Path, preferences: StartupPreferences) -> None:
    """Atomically replace only our preference file; report write failures."""
    # Revalidate even callers that construct an instance unusually; never dump
    # a Settings instance (which contains paths and unrelated configuration).
    values = asdict(preferences)
    checked = StartupPreferences(**values)
    values["loader_execution"] = checked.loader_execution.model_dump(exclude_none=True) if checked.loader_execution is not None else None
    data = json.dumps({"version": _VERSION, **values}, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if len(data.encode("utf-8")) > _MAX_BYTES:
        raise ValueError("startup preferences exceed their byte bound")
    _write_bytes(db_path, data.encode("utf-8"))


def _write_bytes(db_path: Path, data: bytes) -> None:
    """One atomic writer for canonical saves and opaque rollback snapshots."""
    PreferenceSnapshot(data)
    path = path_for(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    owned: tuple[int, int] | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent,
            prefix=".auradream.settings.", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            opened = os.fstat(stream.fileno())
            owned = opened.st_dev, opened.st_ino
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            try:
                value = temporary.lstat()
                if owned is not None and owned[1] and (value.st_dev, value.st_ino) == owned:
                    temporary.unlink()
            except OSError:
                pass


# env.sh supplies Linux-friendly defaults for the server and CLI. It marks
# only values it supplied, so unchanged defaults can yield to a browser choice
# while an operator's explicit environment still wins.
_WSL_MARKERS = {
    "AURADREAM_MODEL_ID": "AURADREAM_WSL_DEFAULT_MODEL_ID",
    "AURADREAM_QUANT": "AURADREAM_WSL_DEFAULT_QUANT",
}
# Each startup variable, the preference it carries and the flag that sets it.
_VARIABLES = (
    ("AURADREAM_MODEL_ID", "model_id", "--model"),
    ("AURADREAM_QUANT", "quantization", "--quant"),
    ("AURADREAM_OFFLOAD", "offload", "--offload"),
    ("AURADREAM_PRELOAD", "load_on_startup", None),
    ("AURADREAM_COMPILATION", "compilation", "--compilation"),
    ("AURADREAM_ATTENTION", "attention_backend", "--attention"),
)
# Set by the launcher only: what a saved choice cannot change on this start.
OVERRIDES_VARIABLE = "AURADREAM_STARTUP_OVERRIDES"


def _explicit(env: MutableMapping[str, str], name: str) -> bool:
    marker = _WSL_MARKERS.get(name)
    return name in env and not (marker is not None and marker in env and env[name] == env[marker])


def explicit_sources(
    env: MutableMapping[str, str] | None = None, flags: dict[str, str | None] | None = None,
) -> dict[str, str]:
    """Each preference an operator set for this start, keyed to what set it.

    `flags` maps a variable to its command-line value; a flag beats the
    environment. env.sh's unchanged marked defaults are not explicit.
    """
    env = os.environ if env is None else env
    flags = flags or {}
    sources = {}
    for name, key, flag in _VARIABLES:
        if flags.get(name) is not None:
            sources[key] = flag
        elif _explicit(env, name):
            sources[key] = name
    return sources


def startup_overrides(env: MutableMapping[str, str] | None = None) -> dict[str, str]:
    """The launcher's record of explicit startup values; none when imported."""
    env = os.environ if env is None else env
    try:
        recorded = json.loads(env.get(OVERRIDES_VARIABLE) or "{}")
    except ValueError:
        return {}
    keys = {key for _, key, _ in _VARIABLES}
    if not isinstance(recorded, dict):
        return {}
    return {
        key: value for key, value in recorded.items()
        if key in keys and isinstance(value, str) and value
    }


def apply_saved_environment(
    env: MutableMapping[str, str] | None = None, *, model: str | None = None,
) -> StartupPreferences | None:
    """Fill absent startup variables, without importing config or changing disk.

    The saved values (checkpoint, quantization, offload, auto-load and loader
    execution) are one choice made for one checkpoint. When the model is
    explicit (`model`, the --model flag applied afterwards, or the
    environment) and is not the saved one, none of them applies: a saved
    BF16/no offload/auto-load for a small model must not load a large one so.
    Returns the preferences applied, for a later choice (the --menu's) to
    check against; None when none were.
    """
    env = os.environ if env is None else env
    db_path = Path(env.get("AURADREAM_DB_PATH", str(default_db_path(_ROOT))))
    result = load_result(db_path)
    preferences = result.preferences
    if result.loader_execution_error is not None:
        env[LOADER_EXECUTION_ERROR_VARIABLE] = LOADER_EXECUTION_ERROR
        return None
    if preferences is None:
        return None
    if preferences.quantization == "nunchaku" and sys.platform != "linux":
        log.warning("Ignoring saved Nunchaku startup settings: Nunchaku requires Linux/WSL")
        if preferences.loader_execution is not None:
            env[LOADER_EXECUTION_ERROR_VARIABLE] = LOADER_EXECUTION_ERROR
        return None
    if model is None and _explicit(env, "AURADREAM_MODEL_ID"):
        model = env["AURADREAM_MODEL_ID"]
    if model is not None and model != preferences.model_id:
        log.warning(
            "Ignoring saved startup settings for %s: this start names %s",
            preferences.model_id, model,
        )
        return None
    for name, value in preferences.environ().items():
        if not _explicit(env, name):
            env[name] = value
    return preferences
