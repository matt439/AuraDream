"""Explicit inactive Windows backup credential setup and user-key recovery check.

Setup never enables automatic backups or retention. A local inactive pending
profile is durable before any vault write. Inspect reads that local profile only,
and recovery uses a password supplied directly by the user, never the vault.
"""
from __future__ import annotations

import argparse
import getpass
import json
from pathlib import Path
import stat
import sys
import tempfile
from uuid import uuid4
import warnings

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend import backup_credentials as credentials  # noqa: E402

_SETUP_ERROR = ("Credential setup did not finish cleanly. Inspect the local inactive profile before another action. "
                "No uncertain credential or profile is automatically deleted, overwritten or retried.")
_RECOVERY_ERROR = "Recovery verification did not succeed. Keep the archive and check the independently retained password; no credential lookup or import occurred."


class SetupError(ValueError):
    pass


def _hidden(prompt: str) -> str:
    # getpass otherwise warns and falls back to echoed stdin. Setup/recovery
    # must fail before that fallback can consume a password visibly.
    with warnings.catch_warnings():
        warnings.simplefilter("error", getpass.GetPassWarning)
        return getpass.getpass(prompt)


def _readable_profile(loaded) -> bool:
    unreadable = (bool(getattr(loaded, "warning", None)) or bool(getattr(loaded, "requires_confirmation", False))) and not getattr(loaded, "legacy", False)
    return loaded.revision is not None and not unreadable


def _check_generation(loaded, new_generation: bool) -> None:
    state = getattr(loaded.settings, "credential_state", None)
    reference = getattr(loaded.settings, "credential_reference", None)
    if not _readable_profile(loaded) or state == "pending":
        raise SetupError("The inactive credential profile needs explicit review before setup can continue.")
    if (state == "ready" or reference is not None) and not new_generation:
        raise SetupError("An existing generation requires explicit --new-generation; old credentials are kept.")


def _setup(args) -> dict:
    from backend import backup_settings, gallery_lock

    if not credentials.supported_platform():
        raise SetupError("Credential setup requires Windows; no profile or vault change was attempted.")
    if not args.allow_os_storage or not args.confirm_independent_recovery:
        raise SetupError("Confirm OS password storage and an independent recovery copy before setup.")
    loaded = backup_settings.load(args.database)
    _check_generation(loaded, args.new_generation)
    assert loaded.revision is not None
    named = args.destination.lstat()
    if (args.destination.is_symlink() or args.destination.is_junction() or not stat.S_ISDIR(named.st_mode)
            or bool(getattr(named, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT)):
        raise SetupError("Choose an existing unlinked destination directory.")
    directory = args.destination.resolve()
    gallery = loaded.settings.gallery_id or str(uuid4())
    # Every setup/rotation gets a new scope/target; never replace an old key.
    reference = credentials.new_reference(gallery, str(uuid4()))
    password: str | None = None
    repeated: str | None = None
    try:
        password = credentials.checked_secret(_hidden("New automatic-backup password (hidden): "))
        repeated = credentials.checked_secret(_hidden("Repeat the new password (hidden): "))
        if password != repeated:
            raise SetupError("The passwords did not match. Nothing was provisioned.")
        with gallery_lock.hold(args.database, wait=gallery_lock.MAINTENANCE_GRACE):
            current = backup_settings.load(args.database)
            _check_generation(current, args.new_generation)
            if current.revision != loaded.revision:
                raise SetupError("The local settings changed. Reload them before setup.")
            assert current.revision is not None
            pending = backup_settings.save_encrypted_profile(
                args.database, revision=current.revision, directory=directory,
                reference=reference, recovery_confirmed=True,
            )
            if pending.revision is None:
                raise SetupError("The inactive pending profile could not be confirmed.")
            credentials.create_credential(reference, password)
            ready = backup_settings.complete_encrypted_profile(args.database, revision=pending.revision, reference=reference)
            if ready.settings.enabled or getattr(ready.settings, "credential_state", None) != "ready":
                raise SetupError("The inactive profile could not be confirmed ready.")
        return {"state": "ready", "enabled": False, "retention_enabled": False,
                "credential_reference": credentials.reference_wire(reference),
                "generation": reference.destination_id, "recovery_password_verified": False}
    finally:
        # Drop our references on mismatch, provider/profile failure and cancel
        # too; this does not promise zeroing caller strings or Python memory.
        password = repeated = None


def _inspect(args) -> dict:
    from backend import backup_settings

    loaded = backup_settings.load(args.database)
    reference = getattr(loaded.settings, "credential_reference", None)
    readable = _readable_profile(loaded)
    return {"state": (getattr(loaded.settings, "credential_state", None) or "not_configured") if readable else "unreadable",
            "settings_readable": readable,
            "credential_reference": credentials.reference_wire(reference) if reference is not None else None,
            "enabled": loaded.settings.enabled, "vault_lookup": False,
            "vault_contents_confirmed": False}


def _recovery(args) -> dict:
    from backend import bundle_archive, bundle_crypto, bundle_leases

    password: str | None = None
    try:
        password = _hidden("Independent recovery password for this archive (hidden): ")
        if not isinstance(password, str) or not password or len(password) > 1024:
            raise SetupError(_RECOVERY_ERROR)
        # Manual archives do not have the credential blob's byte limit. Reuse
        # their existing UTF-8/NFC codec instead of the vault-specific validator.
        bundle_crypto._secret(password)
        with bundle_leases.reading(args.archive) as owner:
            if not bundle_crypto._is_encrypted(args.archive, owner):
                raise SetupError("Choose a password-protected archive for a recovery-password check.")
            # A private temporary database lets the existing verifier validate
            # tracked rows/PNGs too. Other members are read/checked, not installed.
            with tempfile.TemporaryDirectory(prefix=".auradream-recovery-check-") as folder:
                bundle_archive._verify_archive(args.archive, Path(folder), password=password,
                                                extract=lambda _name: False, read_owner=owner)
        return {"state": "verified", "archive": str(args.archive), "credential_lookup": False,
                "imported": False, "user_supplied_recovery_password": True}
    finally:
        password = None


def main(arguments: list[str] | None = None) -> int:
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfigure):
        reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    setup = commands.add_parser("setup", help="save an inactive pending profile, then create one new named Windows credential")
    setup.add_argument("--database", type=Path, required=True)
    setup.add_argument("--destination", type=Path, required=True)
    setup.add_argument("--new-generation", action="store_true", help="explicitly rotate an existing ready profile; keep every old credential")
    setup.add_argument("--allow-os-storage", action="store_true", help="confirm storing this password in Windows Credential Manager")
    setup.add_argument("--confirm-independent-recovery", action="store_true", help="confirm an independent recovery copy of the password is retained outside this PC")
    inspect = commands.add_parser("inspect", help="read local profile metadata only; never open/list the vault")
    inspect.add_argument("--database", type=Path, required=True)
    recovery = commands.add_parser("verify-recovery", help="fully verify one encrypted archive using the user's independent password; no import/vault lookup")
    recovery.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args(arguments)
    try:
        result = _setup(args) if args.command == "setup" else _inspect(args) if args.command == "inspect" else _recovery(args)
    except (Exception, KeyboardInterrupt):
        print(_RECOVERY_ERROR if args.command == "verify-recovery" else _SETUP_ERROR, file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
