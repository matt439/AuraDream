"""Credential Manager operations are entirely fake; no native vault is opened."""
from __future__ import annotations

from contextlib import contextmanager
import ctypes
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import warnings

import pytest

from backend import backup_credentials as credentials
from tools import backup_credential as cli

GALLERY = "11111111-1111-4111-8111-111111111111"
DESTINATION = "22222222-2222-4222-8222-222222222222"
SECRET = "stub-only recovery password"


class FakeApi:
    def __init__(self):
        self.entries = {}
        self.reads, self.writes, self.frees, self.allocations = [], [], [], []
        self.error = 1168
        self.read_error = None
        self.write_error = None
        self.free_error = None
        self.changes = {}
        self.events = []

    def CredReadW(self, target, kind, flags, output):
        self.reads.append((target, kind, flags))
        self.events.append("read")
        if target not in self.entries:
            return 0
        blob = self.entries[target]
        target_buffer = ctypes.create_unicode_buffer(target)
        username = ctypes.create_unicode_buffer(credentials._USERNAME)
        data = (ctypes.c_ubyte * len(blob)).from_buffer_copy(blob)
        item = credentials._Credential(Flags=0, Type=1, TargetName=ctypes.cast(target_buffer, credentials._WidePointer),
                                        CredentialBlobSize=len(blob), CredentialBlob=ctypes.cast(data, credentials._BytePointer),
                                        Persist=2, UserName=ctypes.cast(username, credentials._WidePointer))
        for field, value in self.changes.items():
            setattr(item, field, value)
        pointer = ctypes.pointer(item)
        ctypes.cast(output, ctypes.POINTER(ctypes.POINTER(credentials._Credential)))[0] = pointer
        self.allocations.append((item, pointer, target_buffer, username, data))
        if self.read_error:
            raise self.read_error
        return 1

    def CredWriteW(self, pointer, flags):
        item = ctypes.cast(pointer, ctypes.POINTER(credentials._Credential)).contents
        target = credentials._text(item.TargetName, 256)
        blob = ctypes.string_at(item.CredentialBlob, item.CredentialBlobSize)
        self.writes.append((target, item.Type, item.Persist, flags, blob))
        self.events.append("write")
        if self.write_error:
            raise self.write_error
        self.entries[target] = blob
        return 1

    def CredFree(self, pointer):
        address = ctypes.cast(pointer, ctypes.c_void_p).value
        assert address not in self.frees, "fake buffer freed twice"
        self.frees.append(address)
        self.events.append("free")
        if self.free_error:
            raise self.free_error


@pytest.fixture
def fake(monkeypatch):
    api = FakeApi()
    monkeypatch.setattr(credentials, "_credential_api", lambda: api)
    monkeypatch.setattr(credentials, "_last_error", lambda: api.error)
    monkeypatch.setattr(credentials, "_failed_targets", set())
    monkeypatch.setattr(credentials, "_uncertain_buffers", [])
    return api


def reference():
    return credentials.new_reference(GALLERY, DESTINATION)


def assert_secret_free(error):
    assert SECRET not in str(error) and SECRET not in repr(error)
    assert error.__cause__ is None and error.__context__ is None


def test_reference_is_strict_scope_bound_and_secret_free():
    selected = reference()
    # Frozen from the old name: existing vault entries keep resolving.
    assert selected.target == f"LocalDiffusion/AutomaticBackup/v1/{GALLERY}/{DESTINATION}"
    wire = credentials.reference_wire(selected)
    assert wire["provider"] == "windows-credential-manager-v1"
    assert credentials.read_reference(wire) == selected
    for change in ({"password": SECRET}, {"provider": "plaintext"}, {"target": "another target"},
                   {"gallery_id": "not-a-uuid"}, {"destination_id": "0" * 36}):
        with pytest.raises(credentials.CredentialError) as caught:
            credentials.read_reference({**wire, **change})
        assert_secret_free(caught.value)


def test_import_and_cli_help_are_cold(monkeypatch, capsys):
    def forbidden(*_args, **_kwargs):
        pytest.fail("cold import/help attempted a native credential operation")

    monkeypatch.setattr(ctypes, "WinDLL", forbidden, raising=False)
    module_name = "_cold_backup_credentials_test"
    spec = importlib.util.spec_from_file_location(module_name, Path(credentials.__file__))
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    assert module.new_reference(GALLERY, DESTINATION).target == reference().target
    monkeypatch.setattr(credentials, "_credential_api", forbidden)
    with pytest.raises(SystemExit) as caught:
        cli.main(["--help"])
    assert caught.value.code == 0
    assert "verify-recovery" in capsys.readouterr().out


def test_credential_layout_uses_fixed_width_fields():
    assert ctypes.sizeof(credentials._FileTime) == 8
    assert ctypes.sizeof(credentials._Credential) == (80 if ctypes.sizeof(ctypes.c_void_p) == 8 else 52)


@pytest.mark.parametrize("value", [None, "", "a" * 1025, "\ud800", "😀" * 700])
def test_checked_secret_rejects_invalid_or_oversized_fake_provider_values(value):
    with pytest.raises(credentials.CredentialError) as caught:
        credentials.checked_secret(value)
    assert_secret_free(caught.value)


def test_checked_secret_and_vault_roundtrip_normalize_like_bundle_crypto(fake):
    assert credentials.checked_secret("e\u0301") == "é"
    selected = reference()
    credentials.create_credential(selected, "e\u0301")
    assert fake.reads == [(selected.target, 1, 0)]
    assert fake.writes[0][1:4] == (1, 2, 0)
    assert len(fake.writes[0][4]) <= 2560
    with credentials.read_secret(selected) as password:
        assert password == "é"
    assert len(fake.frees) == 1


@pytest.mark.parametrize("change", [
    {"Flags": 1}, {"Type": 2}, {"Persist": 1}, {"AttributeCount": 1},
    {"CredentialBlobSize": 0}, {"CredentialBlobSize": 2561}, {"CredentialBlob": credentials._BytePointer()},
])
def test_invalid_metadata_is_bounded_and_freed_once(fake, change):
    selected = reference()
    fake.entries[selected.target] = bytes(credentials._blob(selected, SECRET))
    fake.changes = change
    with pytest.raises(credentials.CredentialError) as caught:
        with credentials.read_secret(selected):
            pytest.fail("malformed fake credential yielded")
    assert len(fake.frees) == 1 and fake.writes == []
    assert_secret_free(caught.value)


@pytest.mark.parametrize("kind", ["foreign_scope", "invalid_utf8", "empty_secret", "long_target", "wrong_user"])
def test_foreign_or_malformed_blob_and_metadata_never_escape(fake, kind):
    selected = reference()
    blob = bytes(credentials._blob(selected, SECRET))
    buffer = None
    if kind == "foreign_scope":
        blob = blob.replace(GALLERY.encode(), DESTINATION.encode(), 1)
    elif kind == "invalid_utf8":
        blob = credentials._scope(selected) + b"\xff"
    elif kind == "empty_secret":
        blob = credentials._scope(selected)
    elif kind in {"long_target", "wrong_user"}:
        buffer = ctypes.create_unicode_buffer("x" * 257 if kind == "long_target" else "someone else")
        fake.changes["TargetName" if kind == "long_target" else "UserName"] = ctypes.cast(buffer, credentials._WidePointer)
    fake.entries[selected.target] = blob
    with pytest.raises(credentials.CredentialError) as caught:
        with credentials.read_secret(selected):
            pytest.fail("invalid fake credential yielded")
    assert len(fake.frees) == 1
    assert_secret_free(caught.value)


def test_preexisting_any_credential_is_freed_and_never_overwritten(fake):
    selected = reference()
    fake.entries[selected.target] = b"foreign credential contents"
    fake.changes["Type"] = 99
    with pytest.raises(credentials.CredentialExists) as caught:
        credentials.create_credential(selected, SECRET)
    assert fake.writes == [] and len(fake.frees) == 1
    assert_secret_free(caught.value)


@pytest.mark.parametrize("code,reason", [(1168, "credential_missing"), (1312, "credential_session_unavailable"),
                                         (5, "credential_access_denied"), (12345, "credential_unusable")])
def test_native_error_codes_have_only_fixed_secret_free_reasons(fake, code, reason):
    fake.error = code
    with pytest.raises(credentials.CredentialError) as caught:
        with credentials.read_secret(reference()):
            pytest.fail("failed fake read yielded")
    assert caught.value.reason == reason and fake.frees == []
    assert_secret_free(caught.value)


@pytest.mark.parametrize("operation", ["read", "free", "write"])
def test_provider_exception_messages_and_chains_are_never_returned(fake, operation):
    selected = reference()
    if operation != "write":
        fake.entries[selected.target] = bytes(credentials._blob(selected, SECRET))
    setattr(fake, operation + "_error", RuntimeError(SECRET + " raw provider failure"))
    with pytest.raises(credentials.CredentialError) as caught:
        if operation == "write":
            credentials.create_credential(selected, SECRET)
        else:
            with credentials.read_secret(selected):
                pytest.fail("failed fake read yielded")
    assert_secret_free(caught.value)
    if operation == "free":
        traceback = caught.value.__traceback__
        checked = False
        while traceback is not None:
            if traceback.tb_frame.f_code.co_name == "_read":
                assert traceback.tb_frame.f_locals["secret"] is None
                checked = True
            traceback = traceback.tb_next
        assert checked
    assert len(fake.frees) == (0 if operation == "write" else 1)
    if operation in {"free", "write"}:
        prior = len(fake.reads), len(fake.writes), len(fake.frees)
        with pytest.raises(credentials.CredentialError):
            with credentials.read_secret(selected):
                pytest.fail("uncertain fake target was retried")
        assert prior == (len(fake.reads), len(fake.writes), len(fake.frees))


def test_owned_mutable_buffers_are_wiped_on_success_and_failure(fake, monkeypatch):
    captured = []
    original = credentials._wipe

    def wipe(data):
        original(data)
        captured.append(bytes(data))

    monkeypatch.setattr(credentials, "_wipe", wipe)
    credentials.create_credential(reference(), SECRET)
    with credentials.read_secret(reference()):
        pass
    assert captured and all(not any(data) for data in captured)


def profile(*, state=None, selected=None, revision="old", enabled=False, warning=None, requires_confirmation=False, legacy=False):
    return SimpleNamespace(revision=revision, warning=warning, requires_confirmation=requires_confirmation, legacy=legacy,
                            settings=SimpleNamespace(gallery_id=GALLERY, enabled=enabled,
                                                      credential_state=state, credential_reference=selected))


@pytest.fixture
def tool_stubs(fake, monkeypatch, tmp_path):
    from backend import backup_settings, gallery_lock

    current = [profile()]
    directory = tmp_path / "owned-destination"
    directory.mkdir()
    events = fake.events
    # An explicit fake provider is supported even on Linux CI. Never load a
    # native Windows API merely to qualify this source-only workflow.
    monkeypatch.setattr(credentials, "supported_platform", lambda: True)
    monkeypatch.setattr(backup_settings, "load", lambda _path: current[0])

    @contextmanager
    def hold(*_args, **_kwargs):
        yield

    monkeypatch.setattr(gallery_lock, "hold", hold)
    monkeypatch.setattr(cli.getpass, "getpass", lambda _prompt: SECRET)

    def pending(_database, *, revision, directory, reference, recovery_confirmed):
        assert revision == current[0].revision and recovery_confirmed is True
        assert fake.writes == []
        events.append("pending")
        current[0] = profile(state="pending", selected=reference, revision="pending")
        return current[0]

    def complete(_database, *, revision, reference):
        assert revision == "pending" and current[0].settings.credential_reference == reference
        events.append("complete")
        current[0] = profile(state="ready", selected=reference, revision="ready")
        return current[0]

    monkeypatch.setattr(backup_settings, "save_encrypted_profile", pending)
    monkeypatch.setattr(backup_settings, "complete_encrypted_profile", complete)
    args = ["setup", "--database", str(tmp_path / "owned.db"), "--destination", str(directory),
            "--allow-os-storage", "--confirm-independent-recovery"]
    return current, args, events


def setup_arguments(arguments):
    return SimpleNamespace(database=Path(arguments[arguments.index("--database") + 1]),
                           destination=Path(arguments[arguments.index("--destination") + 1]),
                           new_generation=False, allow_os_storage=True, confirm_independent_recovery=True)


def assert_cleared_cli_frame(error, function, *names):
    traceback = error.__traceback__
    found = False
    while traceback is not None:
        if traceback.tb_frame.f_code is function.__code__:
            assert all(traceback.tb_frame.f_locals[name] is None for name in names)
            found = True
        traceback = traceback.tb_next
    assert found


def test_unsupported_setup_is_cold_before_input_or_pending_profile(tool_stubs, fake, monkeypatch):
    from backend import backup_settings

    _current, arguments, events = tool_stubs
    args = setup_arguments(arguments)
    path = backup_settings.backup_preferences.path_for(args.database)
    defaults = backup_settings.BackupSettings()
    payload = {"version": 2, **{name: getattr(defaults, name) for name in backup_settings._FIELDS - {"version"}}}
    path.write_text(json.dumps(payload))
    before = path.read_bytes()
    names = set(path.parent.iterdir())
    monkeypatch.setattr(credentials, "supported_platform", lambda: False)
    monkeypatch.setattr(cli, "_hidden", lambda *_a: pytest.fail("unsupported setup requested a password"))
    monkeypatch.setattr(backup_settings, "load", lambda *_a: pytest.fail("unsupported setup inspected a profile"))
    with pytest.raises(cli.SetupError, match="requires Windows"):
        cli._setup(args)
    assert path.read_bytes() == before and set(path.parent.iterdir()) == names
    assert events == [] and not fake.reads and not fake.writes and not fake.frees


@pytest.mark.parametrize("phase", ["second_prompt", "mismatch", "pending_save", "vault_write", "ready_save", "cancel"])
def test_setup_failures_clear_own_password_locals_and_preserve_pending(tool_stubs, fake, monkeypatch, phase):
    from backend import backup_settings

    current, arguments, events = tool_stubs
    args = setup_arguments(arguments)
    error = KeyboardInterrupt("injected cancellation") if phase == "cancel" else RuntimeError("injected setup failure")
    count = 0

    def hidden(_prompt):
        nonlocal count
        count += 1
        if count == 2 and phase == "second_prompt":
            raise error
        return "different recovery value" if count == 2 and phase == "mismatch" else SECRET

    def fail(*_args, **_kwargs):
        raise error

    monkeypatch.setattr(cli, "_hidden", hidden)
    if phase == "pending_save":
        monkeypatch.setattr(backup_settings, "save_encrypted_profile", fail)
    elif phase in {"vault_write", "cancel"}:
        monkeypatch.setattr(credentials, "create_credential", fail)
    elif phase == "ready_save":
        monkeypatch.setattr(backup_settings, "complete_encrypted_profile", fail)
    expected = cli.SetupError if phase == "mismatch" else type(error)
    with pytest.raises(expected) as caught:
        cli._setup(args)
    if phase != "mismatch":
        assert caught.value is error
    assert_cleared_cli_frame(caught.value, cli._setup, "password", "repeated")
    if phase in {"vault_write", "ready_save", "cancel"}:
        assert current[0].settings.credential_state == "pending" and not current[0].settings.enabled
    else:
        assert "pending" not in events and not fake.writes


@pytest.mark.parametrize("phase", ["invalid_password", "verifier", "cancel", "hidden_cancel"])
def test_recovery_failures_clear_own_password_and_remain_vault_free(fake, monkeypatch, tmp_path, phase):
    from backend import bundle_archive, bundle_crypto, bundle_leases

    events = []
    error = KeyboardInterrupt("injected recovery cancellation") if "cancel" in phase else RuntimeError("injected verifier failure")

    def hidden(_prompt):
        if phase == "hidden_cancel":
            raise error
        return "\ud800" if phase == "invalid_password" else SECRET

    @contextmanager
    def reading(_path):
        events.append("enter")
        try:
            yield object()
        finally:
            events.append("exit")

    def verify(*_args, **_kwargs):
        raise error

    monkeypatch.setattr(credentials, "supported_platform", lambda: False)
    monkeypatch.setattr(cli, "_hidden", hidden)
    monkeypatch.setattr(bundle_leases, "reading", reading)
    monkeypatch.setattr(bundle_crypto, "_is_encrypted", lambda *_a: True)
    monkeypatch.setattr(bundle_archive, "_verify_archive", verify)
    expected = bundle_crypto.CryptoError if phase == "invalid_password" else type(error)
    with pytest.raises(expected) as caught:
        cli._recovery(SimpleNamespace(archive=tmp_path / "owned.ldbundle"))
    if phase != "invalid_password":
        assert caught.value is error
    assert_cleared_cli_frame(caught.value, cli._recovery, "password")
    assert events == ([] if phase in {"invalid_password", "hidden_cancel"} else ["enter", "exit"])
    assert not fake.reads and not fake.writes and not fake.frees


def test_simulated_setup_records_inactive_intent_before_any_fake_vault_write(tool_stubs, fake, capsys):
    current, args, events = tool_stubs
    assert cli.main(args) == 0
    assert events == ["pending", "read", "write", "complete"]
    assert current[0].settings.credential_state == "ready" and not current[0].settings.enabled
    output = capsys.readouterr()
    assert SECRET not in output.out + output.err
    assert json.loads(output.out)["retention_enabled"] is False


@pytest.mark.parametrize("state,new_generation", [("pending", False), ("pending", True), ("ready", False)])
def test_incomplete_or_unconfirmed_existing_generation_refuses_before_prompt(tool_stubs, monkeypatch, capsys, state, new_generation):
    current, args, events = tool_stubs
    current[0] = profile(state=state, selected=reference())
    monkeypatch.setattr(cli.getpass, "getpass", lambda *_a: pytest.fail("blocked setup prompted"))
    assert cli.main(args + (["--new-generation"] if new_generation else [])) == 1
    assert events == [] and SECRET not in capsys.readouterr().err


def test_echoed_getpass_fallback_is_refused_before_profile_or_vault(tool_stubs, fake, monkeypatch, capsys):
    _current, args, events = tool_stubs

    def unsafe(_prompt):
        warnings.warn("terminal cannot hide input", cli.getpass.GetPassWarning, stacklevel=2)
        pytest.fail("echoed fallback was reached")

    monkeypatch.setattr(cli.getpass, "getpass", unsafe)
    assert cli.main(args) == 1
    assert events == [] and not fake.reads and not fake.writes
    assert SECRET not in capsys.readouterr().err


@pytest.mark.parametrize("requires_confirmation", [False, True])
def test_unreadable_nonlegacy_profile_refuses_setup_and_inspect_reports_unreadable(tool_stubs, fake, monkeypatch, capsys, requires_confirmation):
    current, args, events = tool_stubs
    current[0] = profile(revision="present-damaged-bytes", warning=SECRET, requires_confirmation=requires_confirmation)
    monkeypatch.setattr(cli.getpass, "getpass", lambda *_a: pytest.fail("unreadable profile prompted"))
    assert cli.main(args + ["--new-generation"]) == 1
    assert events == [] and not fake.reads and not fake.writes
    assert SECRET not in capsys.readouterr().err
    database = args[args.index("--database") + 1]
    assert cli.main(["inspect", "--database", database]) == 0
    output = capsys.readouterr()
    assert SECRET not in output.out + output.err
    assert json.loads(output.out)["settings_readable"] is False
    assert json.loads(output.out)["state"] == "unreadable"


def test_ready_rotation_requires_fresh_target_and_retains_old_fake_entry(tool_stubs, fake):
    current, args, _events = tool_stubs
    old = reference()
    fake.entries[old.target] = bytes(credentials._blob(old, SECRET))
    current[0] = profile(state="ready", selected=old)
    assert cli.main(args + ["--new-generation"]) == 0
    assert old.target in fake.entries
    assert current[0].settings.credential_reference.target != old.target


@pytest.mark.parametrize("failure", ["pending_save", "vault_write", "ready_save"])
def test_simulated_provisioning_failures_leave_inactive_intent_and_never_retry(tool_stubs, fake, monkeypatch, capsys, failure):
    from backend import backup_settings

    current, args, events = tool_stubs

    def fail(*_a, **_k):
        raise RuntimeError(SECRET + " injected publication failure")

    if failure == "pending_save":
        monkeypatch.setattr(backup_settings, "save_encrypted_profile", fail)
    elif failure == "vault_write":
        fake.write_error = RuntimeError(SECRET)
    else:
        monkeypatch.setattr(backup_settings, "complete_encrypted_profile", fail)
    assert cli.main(args) == 1
    assert len(fake.writes) == (0 if failure == "pending_save" else 1)
    if failure != "pending_save":
        assert current[0].settings.credential_state == "pending" and not current[0].settings.enabled
        assert cli.main(args + ["--new-generation"]) == 1
        assert len(fake.writes) == 1
    output = capsys.readouterr()
    assert SECRET not in output.out + output.err


def test_inspect_reads_local_metadata_without_any_fake_vault_call(tool_stubs, fake, capsys):
    current, args, _events = tool_stubs
    current[0] = profile(state="pending", selected=reference())
    database = args[args.index("--database") + 1]
    assert cli.main(["inspect", "--database", database]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["state"] == "pending" and result["vault_lookup"] is False
    assert not fake.reads and not fake.writes


@pytest.mark.parametrize("recovery_password", [SECRET, "😀" * 700])
def test_recovery_uses_only_user_password_and_one_borrowed_full_verifier(fake, monkeypatch, tmp_path, capsys, recovery_password):
    from backend import bundle_archive, bundle_crypto, bundle_leases

    owner, events = object(), []

    @contextmanager
    def reading(path):
        events.append(("enter", path))
        yield owner
        events.append(("exit", path))

    def encrypted(path, supplied):
        assert supplied is owner
        return True

    def verify(path, destination, *, password, extract, read_owner):
        assert password == recovery_password and read_owner is owner and destination.is_dir()
        assert not extract("images/fixture.png")
        events.append(("verified", path))

    monkeypatch.setattr(cli.getpass, "getpass", lambda _prompt: recovery_password)
    monkeypatch.setattr(bundle_leases, "reading", reading)
    monkeypatch.setattr(bundle_crypto, "_is_encrypted", encrypted)
    monkeypatch.setattr(bundle_archive, "_verify_archive", verify)
    archive = tmp_path / "synthetic-encrypted.ldbundle"
    assert cli.main(["verify-recovery", "--archive", str(archive)]) == 0
    assert [event[0] for event in events] == ["enter", "verified", "exit"]
    assert not fake.reads and not fake.writes
    output = capsys.readouterr()
    assert SECRET not in output.out + output.err and json.loads(output.out)["imported"] is False


def test_vault_names_are_frozen_from_the_old_name():
    # Renaming these would orphan every credential stored before Auradream.
    assert credentials._USERNAME == "LocalDiffusion"
    assert credentials._MAGIC == b"LocalDiffusion.AutomaticBackup.Credential.v1" + bytes(1)
