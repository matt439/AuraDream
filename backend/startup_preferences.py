"""The browser's remembered loader choice, independent of frozen config imports.

Only these four preferences are persisted, beside the gallery database. The
server launcher reads them before importing config; explicit environment and
command-line values still take precedence. A bad or newer file never prevents
the browser from opening so its settings can be repaired there.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
from collections.abc import MutableMapping
from dataclasses import asdict, dataclass
from pathlib import Path

from .model_defaults import OFFLOAD_MODES, QUANT_MODES

log = logging.getLogger(__name__)
_VERSION = 1
_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class StartupPreferences:
    model_id: str
    quantization: str
    offload: str
    load_on_startup: bool

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

    def environ(self) -> dict[str, str]:
        return {
            "LOCAL_DIFFUSION_MODEL_ID": self.model_id,
            "LOCAL_DIFFUSION_QUANT": self.quantization,
            "LOCAL_DIFFUSION_OFFLOAD": self.offload,
            "LOCAL_DIFFUSION_PRELOAD": "true" if self.load_on_startup else "false",
        }


def path_for(db_path: Path) -> Path:
    return Path(db_path).expanduser().parent / "local-diffusion.settings.json"


def load(db_path: Path) -> StartupPreferences | None:
    """Read a complete, known schema; ignore a missing or unusable file."""
    path = path_for(db_path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("settings must be a JSON object")
        if type(payload.get("version")) is not int or payload["version"] != _VERSION:
            raise ValueError("unsupported settings version")
        values = {key: value for key, value in payload.items() if key != "version"}
        return StartupPreferences(**values)
    except FileNotFoundError:
        return None
    except (OSError, UnicodeError, ValueError, TypeError) as exc:
        log.warning("Ignoring saved startup settings at %s: %s", path, exc)
        return None


def save(db_path: Path, preferences: StartupPreferences) -> None:
    """Atomically replace only our preference file; report write failures."""
    # Revalidate even callers that construct an instance unusually; never dump
    # a Settings instance (which contains paths and unrelated configuration).
    values = asdict(preferences)
    StartupPreferences(**values)
    path = path_for(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=".local-diffusion.settings.", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump({"version": _VERSION, **values}, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


# env.sh supplies Linux-friendly defaults for the server and CLI. It marks
# only values it supplied, so unchanged defaults can yield to a browser choice
# while an operator's explicit environment still wins.
_WSL_MARKERS = {
    "LOCAL_DIFFUSION_MODEL_ID": "LOCAL_DIFFUSION_WSL_DEFAULT_MODEL_ID",
    "LOCAL_DIFFUSION_QUANT": "LOCAL_DIFFUSION_WSL_DEFAULT_QUANT",
}
# Each startup variable, the preference it carries and the flag that sets it.
_VARIABLES = (
    ("LOCAL_DIFFUSION_MODEL_ID", "model_id", "--model"),
    ("LOCAL_DIFFUSION_QUANT", "quantization", "--quant"),
    ("LOCAL_DIFFUSION_OFFLOAD", "offload", "--offload"),
    ("LOCAL_DIFFUSION_PRELOAD", "load_on_startup", None),
)
# Set by the launcher only: what a saved choice cannot change on this start.
OVERRIDES_VARIABLE = "LOCAL_DIFFUSION_STARTUP_OVERRIDES"


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
) -> None:
    """Fill absent startup variables, without importing config or changing disk.

    The four values are one choice made for one checkpoint. When the model is
    explicit (`model`, the --model flag applied afterwards, or the
    environment) and is not the saved one, none of them applies: a saved
    BF16/no offload/auto-load for a small model must not load a large one so.
    """
    env = os.environ if env is None else env
    db_path = Path(env.get("LOCAL_DIFFUSION_DB_PATH", str(_ROOT / "local-diffusion.db")))
    preferences = load(db_path)
    if preferences is None:
        return
    if preferences.quantization == "nunchaku" and sys.platform != "linux":
        log.warning("Ignoring saved Nunchaku startup settings: Nunchaku requires Linux/WSL")
        return
    if model is None and _explicit(env, "LOCAL_DIFFUSION_MODEL_ID"):
        model = env["LOCAL_DIFFUSION_MODEL_ID"]
    if model is not None and model != preferences.model_id:
        log.warning(
            "Ignoring saved startup settings for %s: this start names %s",
            preferences.model_id, model,
        )
        return
    for name, value in preferences.environ().items():
        if not _explicit(env, name):
            env[name] = value
