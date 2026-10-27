"""Where files written under the app's old name, Local Diffusion, keep working.

The app was renamed AuraDream. Everything it creates now takes the new name;
a gallery, sidecar or cache that already exists under the old one is used
where it is. Nothing here renames, moves or writes anything:

- The gallery database is not renamed. SQLite keeps a live database in three
  files (the `.db`, its `-wal` and `-shm`) and this app adds its own
  `.gallery.lock` beside them, every one named from the database path. A
  rename while another process holds them open is refused on Windows but
  succeeds on Linux (WSL), where it splits one gallery into two and leaves
  the two processes locking different files. Defaulting to the old file when
  it is the only one there keeps the database, its journal, its lock and
  every sidecar named from it on one unchanged path, with no window at all.
- A sidecar beside the database (saved settings, snippets, backup policy and
  receipts) is read and rewritten under the name it already has, for the
  same reason: another process may be writing it, and two names for one
  file would let it read a stale copy.
- The app's folder under HF_HOME (converted checkpoints, the llama.cpp
  critic, upscalers) is shared by every server and CLI process with no
  gallery lock to coordinate them, and a conversion holds its work there for
  minutes; renaming it under a running one costs a re-conversion (minutes,
  and 40-50 GB of RAM). Each subfolder is used where it already is.

So an existing install keeps its old names until its owner renames the set
by hand (with no AuraDream process running); a new one has only new names.
Import-light on purpose: `config.py` and `startup_preferences.py` resolve the
default database before anything else is imported.
"""
from __future__ import annotations

from pathlib import Path

PREFIX = "auradream"
# Frozen from the old name: files already on disk carry it.
LEGACY_PREFIX = "local-diffusion"


def prefer(new: Path, legacy: Path) -> Path:
    """`new`, unless it is absent and `legacy` exists.

    Whichever is chosen is then both read and written, so no process creates
    the new name while the old one holds the data.
    """
    if not new.exists() and legacy.exists():
        return legacy
    return new


def default_db_path(root: Path) -> Path:
    """The database used when AURADREAM_DB_PATH names none.

    `root / "auradream.db"`, or the old `local-diffusion.db` while that is
    the only gallery there. An explicit path is never redirected.
    """
    return prefer(root / f"{PREFIX}.db", root / f"{LEGACY_PREFIX}.db")


def sidecar(database: Path, suffix: str) -> Path:
    """`auradream.<suffix>` beside `database`, or its old-name file there.

    `suffix` is the part after the app prefix and its dot, as in
    `settings.json` or `backup-policy-{database.name}.json`. Applies beside
    any database, the default or an explicit one.
    """
    directory = Path(database).expanduser().parent
    return prefer(directory / f"{PREFIX}.{suffix}", directory / f"{LEGACY_PREFIX}.{suffix}")


def hf_home_dir(name: str) -> Path:
    """`HF_HOME/auradream/<name>`, or the old `HF_HOME/local-diffusion/<name>`
    while only that exists: its converted trees and installs stay valid."""
    from huggingface_hub.constants import HF_HOME

    home = Path(HF_HOME)
    return prefer(home / PREFIX / name, home / LEGACY_PREFIX / name)
