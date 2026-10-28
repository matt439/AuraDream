"""Portable setups retain raw intent and require explicit, bounded merge choices."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import uuid

import pytest
from pydantic import ValidationError

from backend import preset_portability as portable, presets
from backend.schemas_preset_portability import (
    PORTABLE_FORMAT, ImportDecision, PortablePreset, PortablePresets,
)
from backend.schemas_presets import MAX_PRESETS, Preset, PresetFields


MAX_BYTES = 8 * 1024 * 1024
MODEL = "example/Exact-Checkpoint"


def fields(name="Study", model_id=MODEL, **changes):
    raw = {
        "mode": "generate", "prompt": "  first line\n\u65e5\u672c\u8a9e \U0001f3a8  ",
        "negative_prompt": "  blurry\r\n lowres  ",
        "steps": None, "guidance_scale": 0, "mu": 0, "true_cfg": None,
        "seed": None, "num_images": 2, "runs": 3,
        **changes,
    }
    return {"name": name, "model_id": model_id, "payload": raw}


def grid_fields(name="Grid", model_id=MODEL, **changes):
    return fields(name, model_id, mode="grid", grid={
        "prompt_axis": "  light = night | day\ncolour = \u9752 | red  ",
        "prompt_mode": "random", "steps_axis": "28, 12", "cfg_axis": "3,0",
        "mu_axis": "1.2,0", "size_axis": "1024x768,768x1024",
        "seeds": 5, "seed_axis": "44,11,22", "budget_minutes": 45,
    }, **changes)


def payload(*items):
    return portable.parse_portable(json.dumps({
        "format": PORTABLE_FORMAT, "version": 1, "items": list(items),
    }, ensure_ascii=False).encode("utf-8"))


def saved(number, *, revision=1, **values):
    return Preset(id=f"{number:032x}", revision=revision, **fields(**values))


def saved_grid(number, *, revision=1, **values):
    return Preset(id=f"{number:032x}", revision=revision, **grid_fields(**values))


def public(item):
    return item.model_dump(include={"name", "model_id", "payload"})


def choose(index, action, **values):
    return ImportDecision(index=index, action=action, **values)


def test_generate_and_grid_export_exact_raw_drafts_and_complete_payloads():
    originals = [saved(1, revision=8), saved_grid(2, model_id="other/Foreign-v2", revision=4)]
    before = [item.model_dump() for item in originals]
    raw = portable.export_portable(originals)
    decoded = json.loads(raw.decode("utf-8"))
    assert decoded == {
        "format": PORTABLE_FORMAT, "version": 1,
        "items": [public(item) for item in originals],
    }
    assert all(set(item) == {"name", "model_id", "payload"} for item in decoded["items"])
    restored = portable.parse_portable(raw)
    assert isinstance(restored, PortablePresets)
    assert [public(item) for item in restored.items] == decoded["items"]
    assert set(decoded["items"][0]["payload"]) == set(type(originals[0].payload).model_fields)
    assert restored.items[0].payload.prompt == originals[0].payload.prompt
    assert restored.items[0].payload.negative_prompt == originals[0].payload.negative_prompt
    assert restored.items[0].payload.seed is None and restored.items[0].payload.steps is None
    assert restored.items[0].payload.guidance_scale == 0 and restored.items[0].payload.mu == 0
    assert restored.items[0].payload.runs == 3 and restored.items[0].payload.num_images == 2
    assert restored.items[1].payload.grid == originals[1].payload.grid
    assert restored.items[1].payload.grid.seed_axis == "44,11,22"
    assert [item.model_dump() for item in originals] == before


def test_empty_file_is_valid_and_plan_keeps_existing_library_without_a_change():
    document = portable.parse_portable(portable.export_portable([]))
    current = [saved(1)]
    plan = portable.plan_merge(current, document)
    assert plan.items == current and plan.additions == [] and plan.rows == []
    assert (plan.add_count, plan.replace_count, plan.skip_count, plan.changed) == (0, 0, 0, False)
    shown = portable.preview(current, document)
    assert not shown.changed and not shown.can_import and shown.reason


@pytest.mark.parametrize("version", [True, False, 1.0, "1", 0, 2, None])
def test_version_is_exact_integer_one(version):
    raw = json.dumps({"format": PORTABLE_FORMAT, "version": version, "items": []}).encode()
    with pytest.raises(portable.PortabilityError):
        portable.parse_portable(raw)


@pytest.mark.parametrize("change", [
    {"format": "auradream.prompt-snippets"}, {"format": PORTABLE_FORMAT + ".next"},
    {"items": {}}, {"items": None}, {"unknown_private_field": "PRIVATE PROMPT MUST NOT BE ECHOED"},
])
def test_envelope_is_exact_and_public_errors_do_not_echo_private_values(change):
    value = {"format": PORTABLE_FORMAT, "version": 1, "items": [], **change}
    with pytest.raises(portable.PortabilityError) as error:
        portable.parse_portable(json.dumps(value).encode())
    assert error.value.status == 422
    assert "PRIVATE PROMPT MUST NOT BE ECHOED" not in str(error.value)


@pytest.mark.parametrize("raw", [
    b"", b"not JSON", b"\xff",
    b'{"a":1,"a":2}', b'{"a":{"b":1,"b":2}}',
    b'{"a":NaN}', b'{"a":Infinity}', b'{"a":-Infinity}',
    b"[" * 17 + b"0" + b"]" * 17,
])
def test_strict_json_refuses_bad_syntax_encoding_duplicate_keys_and_depth(raw):
    with pytest.raises(portable.PortabilityError) as error:
        portable.parse_json(raw)
    assert error.value.status == 422


def test_json_depth_counts_containers_and_ignores_escaped_text():
    assert portable.parse_json(b"[" * 16 + b"0" + b"]" * 16)
    text = '[{{ "quoted" \\ }}]' * 60
    restored = payload(fields(prompt=text))
    assert restored.items[0].payload.prompt == text


def test_raw_byte_limit_is_checked_before_json_and_counts_whitespace():
    raw = portable.export_portable([])
    exact = raw + b" " * (MAX_BYTES - len(raw))
    assert portable.parse_portable(exact).items == []
    with pytest.raises(portable.PortabilityError):
        portable.parse_portable(exact + b" ")


@pytest.mark.parametrize("encoding", ["utf-16", "utf-32"])
def test_portable_is_utf8_only_while_existing_sidecar_read_edit_compatibility_remains(tmp_path, encoding):
    original = saved_grid(1)
    portable_raw = json.dumps({
        "format": PORTABLE_FORMAT, "version": 1, "items": [public(original)],
    }, ensure_ascii=False).encode(encoding)
    with pytest.raises(portable.PortabilityError):
        portable.parse_portable(portable_raw)
    database = tmp_path / "gallery.db"
    path = presets.path_for(database)
    path.write_bytes(json.dumps({
        "version": 1, "items": [original.model_dump()],
    }, ensure_ascii=False).encode(encoding))
    assert presets.read(database) == [original]
    added = presets.change(database, PresetFields(**fields(name="Second")))
    assert added[0] == original and added[1].name == "Second"
    assert presets.read(database) == added
    assert not database.exists()


@pytest.mark.parametrize("encoding", ["utf-16", "utf-32"])
def test_legacy_encoded_duplicate_fields_still_refuse_and_preserve_file(tmp_path, encoding):
    database = tmp_path / "gallery.db"
    path = presets.path_for(database)
    raw = '{"version":1,"version":1,"items":[]}'.encode(encoding)
    path.write_bytes(raw)
    for call in (lambda: presets.read(database), lambda: presets.change(database, PresetFields(**fields()))):
        with pytest.raises(presets.PresetsError, match="left unchanged"):
            call()
        assert path.read_bytes() == raw


@pytest.mark.parametrize("level", ["item", "payload", "grid"])
@pytest.mark.parametrize("key,value", [
    ("id", "0" * 32), ("revision", 1), ("model_revision", 0),
    ("step_cache", None), ("execution", None), ("loader_execution", {}),
    ("reference", []), ("lora", None), ("refine", None), ("mask", None),
    ("edit", None), ("then_refine", None), ("resume", False),
    ("job_id", "private"), ("run_id", "private"), ("path", "../private"),
])
def test_forbidden_identity_conditioning_execution_and_transport_fields_are_not_stripped(level, key, value):
    raw = grid_fields()
    target = raw if level == "item" else raw["payload"] if level == "payload" else raw["payload"]["grid"]
    target[key] = value
    with pytest.raises(portable.PortabilityError):
        payload(raw)


@pytest.mark.parametrize("level", ["payload", "grid"])
def test_checkpoint_identity_is_only_an_item_field(level):
    raw = grid_fields()
    target = raw["payload"] if level == "payload" else raw["payload"]["grid"]
    target["model_id"] = "another/Checkpoint"
    with pytest.raises(portable.PortabilityError):
        payload(raw)


@pytest.mark.parametrize("change", [
    {"name": " "}, {"name": "x" * 81}, {"name": "a\x7fb"}, {"name": "a\nb"},
    {"name": "\ud800"}, {"model_id": " model"}, {"model_id": "model "},
    {"model_id": "\ud800"}, {"model_id": "x" * 513}, {"model_id": True},
])
def test_portable_preserves_existing_name_and_exact_checkpoint_validation(change):
    # Escaped surrogate input must reach validation rather than fail in this
    # test's UTF-8 encoder before the codec is called.
    raw = json.dumps({
        "format": PORTABLE_FORMAT, "version": 1, "items": [{**fields(), **change}],
    }, ensure_ascii=True).encode("utf-8")
    with pytest.raises(portable.PortabilityError):
        portable.parse_portable(raw)


def test_astral_name_bound_is_codepoints_and_raw_text_is_not_normalized():
    document = payload(fields(name="\U0001f3a8" * 80, prompt=" \r\n ", negative_prompt="e\u0301"))
    assert document.items[0].name == "\U0001f3a8" * 80
    assert document.items[0].payload.prompt == " \r\n "
    assert document.items[0].payload.negative_prompt == "e\u0301"
    with pytest.raises(portable.PortabilityError):
        payload(fields(name="\U0001f3a8" * 81))


@pytest.mark.parametrize("patch", [
    {"steps": True}, {"steps": 1.0}, {"width": True}, {"seed": "1"},
    {"guidance_scale": float("nan")}, {"mu": float("inf")},
    {"mode": "grid", "grid": None}, {"mode": "generate", "grid": {}},
    {"prompt": "x" * 4001}, {"negative_prompt": "x" * 4001},
])
def test_payload_types_and_mode_grid_contract_are_authoritative(patch):
    with pytest.raises(portable.PortabilityError):
        payload(fields(**patch))


def test_unvalidated_model_inputs_are_revalidated_before_export_digest_and_planning():
    document = payload(fields())
    bad_payload = document.items[0].payload.model_copy(update={"width": True})
    bad_item = document.items[0].model_copy(update={"payload": bad_payload})
    forged = document.model_copy(update={"items": [bad_item]})
    for call in (
        lambda: portable.payload_digest(forged), lambda: portable.plan_merge([], forged),
        lambda: portable.export_portable([saved(1).model_copy(update={"revision": True})]),
    ):
        with pytest.raises(portable.PortabilityError):
            call()


def test_payload_digest_is_canonical_but_covers_null_zero_modes_order_and_exact_text():
    original = payload(fields())
    complete = original.model_dump()
    reordered = {"items": complete["items"], "version": 1, "format": PORTABLE_FORMAT}
    assert portable.payload_digest(original) == portable.payload_digest(portable.parse_portable(
        json.dumps(reordered, ensure_ascii=True, indent=4).encode(),
    ))
    for different in (
        payload(fields(steps=1)), payload(fields(guidance_scale=None)),
        payload(fields(seed=0)), payload(fields(prompt=original.items[0].payload.prompt + " ")),
        payload(grid_fields()), payload(fields(model_id=MODEL.lower())),
    ):
        assert portable.payload_digest(original) != portable.payload_digest(different)
    first, second = fields("First"), fields("Second")
    assert portable.payload_digest(payload(first, second)) != portable.payload_digest(payload(second, first))


def test_library_revision_covers_order_identity_revision_checkpoint_and_complete_payload():
    first, second = saved(1), saved(2, name="Other")
    current = [first, second]
    revision = portable.collection_revision(current)
    assert revision == portable.collection_revision([item.model_copy(deep=True) for item in current])
    for changed in (
        [], [second, first], [first], [saved(3), second],
        [saved(1, revision=2), second], [saved(1, model_id=MODEL.lower()), second],
        [saved(1, guidance_scale=None), second], [saved(1, seed=0), second],
        [saved(1, prompt=first.payload.prompt + " "), second],
    ):
        assert portable.collection_revision(changed) != revision


def test_default_plan_preserves_duplicate_names_and_checkpoint_case_namespace():
    current = [saved(1), saved(2, prompt="local variant"), saved(3, name="Caf\u00e9")]
    incoming = payload(
        public(current[0]), fields(prompt="incoming"), fields(name="Cafe\u0301"),
        fields(model_id=MODEL.lower()), fields(model_id="unavailable/private-model"),
        grid_fields(name="Study"),
    )
    plan = portable.plan_merge(current, incoming)
    assert [row.action for row in plan.rows] == ["skip", "skip", "skip", "add", "add", "skip"]
    assert plan.rows[0].exact_ids == [current[0].id]
    assert [item.id for item in plan.rows[1].matches] == [current[0].id, current[1].id]
    assert [item.id for item in plan.rows[2].matches] == [current[2].id]
    assert plan.rows[5].exact_ids == []
    assert plan.items == current and len(plan.additions) == 2
    assert (plan.add_count, plan.replace_count, plan.skip_count) == (2, 0, 4)


def test_export_reimport_defaults_to_exact_skip_even_with_legal_duplicate_rows():
    current = [saved(1), saved(2, prompt="variant"), saved(3)]
    plan = portable.plan_merge(current, portable.parse_portable(portable.export_portable(current)))
    assert not plan.changed and plan.skip_count == 3 and not plan.additions
    assert plan.rows[0].exact_ids == [current[0].id, current[2].id]


def test_generate_and_grid_share_conflict_namespace_and_model_identity_never_changes_on_replace():
    current = [saved(1), saved(2, model_id="another/Checkpoint")]
    incoming = payload(grid_fields(name="Study"))
    plan = portable.plan_merge(current, incoming, [choose(0, "replace", item_id=current[0].id, revision=1)])
    assert plan.items[0].id == current[0].id and plan.items[0].revision == 2
    assert plan.items[0].model_id == MODEL and plan.items[0].payload.mode == "grid"
    assert plan.items[1] == current[1]
    with pytest.raises(portable.PortabilityError):
        portable.plan_merge(current, incoming, [choose(0, "replace", item_id=current[1].id, revision=1)])


def test_explicit_exact_replace_is_a_revisioned_edit_and_skip_does_not_change_it():
    current = [saved(1, revision=7)]
    document = payload(public(current[0]))
    plan = portable.plan_merge(current, document, [choose(0, "replace", item_id=current[0].id, revision=7)])
    assert plan.changed and plan.replace_count == 1
    assert plan.items[0] == current[0].model_copy(update={"revision": 8})
    skipped = portable.plan_merge(current, document, [choose(0, "skip")])
    assert not skipped.changed and skipped.items == current


def test_file_indices_and_copy_renames_preserve_every_incoming_duplicate_and_file_order():
    document = payload(fields("First"), fields("STRASSE"), fields("Stra\u00dfe", prompt="variant"))
    plan = portable.plan_merge([], document, [choose(0, "copy", name="  Strasse  ")])
    assert [row.index for row in plan.rows] == [0, 1, 2]
    assert [row.action for row in plan.rows] == ["copy", "add", "add"]
    assert [row.incoming_collisions for row in plan.rows] == [[1, 2], [0, 2], [0, 1]]
    assert [item.name for item in plan.additions] == ["Strasse", "STRASSE", "Stra\u00dfe"]
    assert plan.add_count == 3


def test_replace_positions_and_addition_order_follow_ids_and_file_order_not_choice_order():
    current = [saved(1), saved(2, prompt="second", revision=4), saved(3, name="Untouched")]
    incoming = payload(fields(prompt="replacement"), fields("New"), public(current[0]), fields("Skipped"))
    plan = portable.plan_merge(current, incoming, [
        choose(2, "copy", name="Copied"), choose(0, "replace", item_id=current[1].id, revision=4),
        choose(3, "skip"), choose(1, "add"),
    ])
    assert [item.id for item in plan.items] == [item.id for item in current]
    assert plan.items[0] == current[0] and plan.items[2] == current[2]
    assert plan.items[1].revision == 5 and plan.items[1].payload.prompt == "replacement"
    assert [item.name for item in plan.additions] == ["New", "Copied"]


def test_add_cannot_bypass_local_collision_but_explicit_copy_may_keep_duplicate_name():
    current = [saved(1)]
    document = payload(fields(prompt="different"))
    with pytest.raises(portable.PortabilityError):
        portable.plan_merge(current, document, [choose(0, "add")])
    plan = portable.plan_merge(current, document, [choose(0, "copy")])
    assert plan.items == current and plan.additions[0].name == "Study"


def test_stale_target_and_duplicate_replacement_target_refuse_without_source_mutation():
    current = [saved(1, revision=3)]
    document = payload(fields(prompt="first"), fields(prompt="second"))
    before = ([item.model_dump() for item in current], document.model_dump())
    with pytest.raises(portable.PortabilityError) as error:
        portable.plan_merge(current, document, [choose(0, "replace", item_id=current[0].id, revision=2)])
    assert error.value.status == 409
    with pytest.raises(portable.PortabilityError):
        portable.plan_merge(current, document, [
            choose(index, "replace", item_id=current[0].id, revision=3) for index in range(2)
        ])
    assert ([item.model_dump() for item in current], document.model_dump()) == before


@pytest.mark.parametrize("decisions", [
    [choose(0, "skip"), choose(0, "copy")], [choose(1, "skip")],
])
def test_duplicate_and_out_of_range_choice_indices_are_refused(decisions):
    with pytest.raises(portable.PortabilityError):
        portable.plan_merge([], payload(fields()), decisions)


@pytest.mark.parametrize("value", [
    {"index": True, "action": "skip"}, {"index": 0.0, "action": "skip"},
    {"index": -1, "action": "skip"}, {"index": 100, "action": "skip"},
    {"index": 0, "action": "delete"}, {"index": 0, "action": "replace"},
    {"index": 0, "action": "replace", "item_id": "0" * 32},
    {"index": 0, "action": "replace", "item_id": "../x", "revision": 1},
    {"index": 0, "action": "replace", "item_id": "0" * 32, "revision": True},
    {"index": 0, "action": "skip", "item_id": "0" * 32},
    {"index": 0, "action": "replace", "item_id": "0" * 32, "revision": 1, "name": "Rename"},
    {"index": 0, "action": "copy", "name": " "},
])
def test_choice_fields_are_strict_and_only_copy_accepts_an_edited_name(value):
    with pytest.raises((ValidationError, portable.PortabilityError)):
        choice = ImportDecision.model_validate(value)
        portable.plan_merge([], payload(fields()), [choice])


def test_preview_can_fill_defaults_but_complete_merge_requires_every_choice_and_a_change():
    document = payload(fields("One"), fields("Two"))
    assert portable.plan_merge([], document, [choose(1, "skip")]).add_count == 1
    for decisions in ([], [choose(1, "skip")], [choose(0, "skip"), choose(1, "skip")]):
        with pytest.raises(portable.PortabilityError):
            portable.plan_merge([], document, decisions, require_complete=True)
    plan = portable.plan_merge([], document, [choose(1, "skip"), choose(0, "add")], require_complete=True)
    assert plan.changed and [item.name for item in plan.additions] == ["One"]
    with pytest.raises(portable.PortabilityError):
        portable.plan_merge([saved(1)], payload(), [], require_complete=True)


def test_maximum_collection_is_accepted_but_one_more_and_final_overflow_are_refused():
    entries = [fields(f"Setup {index}") for index in range(MAX_PRESETS)]
    assert len(payload(*entries).items) == MAX_PRESETS
    with pytest.raises(portable.PortabilityError):
        payload(*entries, fields("Extra"))
    current = [saved(index, name=f"Setup {index}") for index in range(MAX_PRESETS)]
    assert not portable.plan_merge(current, payload(public(current[0]))).changed
    with pytest.raises(portable.PortabilityError):
        portable.plan_merge(current, payload(fields("Extra")))
    with pytest.raises(portable.PortabilityError):
        portable.plan_merge(current, payload(public(current[0])), [choose(0, "copy")])


def test_final_size_uses_pretty_sidecar_utf8_not_compact_payload_or_each_input_size():
    def large(name):
        return fields(name, mode="grid", prompt="\x00" * 4000, negative_prompt="\x00" * 4000, grid={
            "prompt_axis": "\x00" * 2000, "size_axis": "\x00" * 2000,
            "seed_axis": "\x00" * 2000, "steps_axis": "\x00" * 200,
            "cfg_axis": "\x00" * 200, "mu_axis": "\x00" * 200,
        })

    current = [Preset(**large(f"Local {index}"), id=f"{index:032x}", revision=1) for index in range(50)]
    document = payload(*(large(f"Incoming {index}") for index in range(50)))
    assert len(portable.export_portable(current)) < MAX_BYTES
    hypothetical = [item.model_dump() for item in current] + [
        {**item.model_dump(), "id": "f" * 32, "revision": 1} for item in document.items
    ]
    assert len((json.dumps({"version": 1, "items": hypothetical}, ensure_ascii=False,
                           indent=2, allow_nan=False) + "\n").encode("utf-8")) > MAX_BYTES
    before = [item.model_dump() for item in current]
    with pytest.raises(portable.PortabilityError):
        portable.plan_merge(current, document)
    assert [item.model_dump() for item in current] == before


def test_planning_allocates_no_ids_or_store_access_and_mutates_no_inputs(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("portable draft planning reached identity or storage work")

    monkeypatch.setattr(uuid, "uuid4", forbidden)
    monkeypatch.setattr(presets, "read", forbidden)
    monkeypatch.setattr(presets, "change", forbidden)
    monkeypatch.setattr(presets, "_write", forbidden)
    current = [saved(1)]
    document = payload(fields(prompt="replacement"), grid_fields("New", model_id="unknown/Unloaded"))
    decisions = [choose(0, "replace", item_id=current[0].id, revision=1)]
    before = ([item.model_dump() for item in current], document.model_dump(), [item.model_dump() for item in decisions])
    plan = portable.plan_merge(current, document, decisions)
    assert plan.changed and plan.items[0].revision == 2
    assert all(isinstance(item, PortablePreset) for item in plan.additions)
    assert all(set(item.model_dump()) == {"name", "model_id", "payload"} for item in plan.additions)
    assert portable.parse_portable(portable.export_portable(current)).items
    assert portable.collection_revision(current) and portable.payload_digest(document)
    assert portable.preview(current, document, decisions).can_import
    assert ([item.model_dump() for item in current], document.model_dump(), [item.model_dump() for item in decisions]) == before


def test_codec_is_cold_in_a_fresh_process_without_application_or_execution_imports(tmp_path):
    # A fresh process makes this independent of the suite's main-app conftest.
    # Deny modules by public import boundary, without resolving their helpers.
    script = r"""
import json
import sys

sys.path.insert(0, sys.argv[1])
blocked = {
    "backend.model_selection", "backend.grid", "backend.preset_validation",
    "backend.main", "backend.jobs", "backend.submission", "backend.backends",
    "backend.backup_credentials", "backend.presets", "backend.storage",
    "torch", "diffusers", "transformers", "huggingface_hub",
}
attempts = []

class ColdBoundary:
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == prefix or fullname.startswith(prefix + ".") for prefix in blocked):
            attempts.append(fullname)
            raise AssertionError("cold setup portability imported " + fullname)
        return None

sys.meta_path.insert(0, ColdBoundary())
from backend import preset_portability as portable
from backend.schemas_preset_portability import ImportDecision, PORTABLE_FORMAT
from backend.schemas_presets import Preset

raw = {"name": "Cold", "model_id": "unknown/Exact-Checkpoint", "payload": {
    "prompt": "  untouched raw\n", "seed": None, "steps": None, "guidance_scale": 0,
    "num_images": 2, "runs": 3,
}}
local = Preset(**raw, id="a" * 32, revision=4)
grid = {"name": "Raw grid", "model_id": "foreign/Unavailable", "payload": {
    "mode": "grid", "prompt": "{name}", "grid": {
        "prompt_axis": "name = blue | red", "prompt_mode": "random", "seed_axis": "44,11,22",
    },
}}
document = portable.parse_portable(json.dumps({
    "format": PORTABLE_FORMAT, "version": 1, "items": [raw, grid],
}).encode("utf-8"))
choices = [ImportDecision(index=0, action="replace", item_id=local.id, revision=4)]
plan = portable.plan_merge([local], document, choices)
assert plan.changed and plan.items[0].revision == 5 and plan.add_count == 1
assert plan.additions[0].payload.grid.seed_axis == "44,11,22"
assert plan.additions[0].payload.seed is None
assert portable.parse_portable(portable.export_portable([local])).items[0].payload.prompt == raw["payload"]["prompt"]
assert portable.collection_revision([local]) and portable.payload_digest(document)
assert portable.preview([local], document, choices).can_import
assert attempts == [], attempts
assert not any(name == prefix or name.startswith(prefix + ".") for name in sys.modules for prefix in blocked)
print("cold setup codec complete")
"""
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", script, str(Path(__file__).resolve().parents[1])],
        cwd=tmp_path, capture_output=True, text=True, timeout=30, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "cold setup codec complete"


def test_selected_native_format_and_filename_are_explicit():
    from backend.schemas_preset_portability import PORTABLE_FILENAME

    assert PORTABLE_FORMAT == "auradream.generation-presets"
    assert PORTABLE_FILENAME == "auradream-generation-setups.json"


def test_constructed_nested_subtypes_and_unchecked_extras_are_rejected_before_serialization():
    from backend.schemas_presets import PresetCreate

    original = payload(grid_fields())
    nested = original.items[0]
    invalid_items = [
        saved(1),
        PresetCreate(**fields(), model_revision=1),
        nested.model_copy(update={"id": "0" * 32}),
        nested.model_copy(update={"payload": nested.payload.model_copy(update={"execution": None})}),
        nested.model_copy(update={"payload": nested.payload.model_copy(update={
            "grid": nested.payload.grid.model_copy(update={"resume": False}),
        })}),
    ]
    for item in invalid_items:
        forged = PortablePresets.model_construct(format=PORTABLE_FORMAT, version=1, items=[item])
        with pytest.raises(portable.PortabilityError):
            portable.validate_payload(forged)
        with pytest.raises(portable.PortabilityError):
            portable.plan_merge([], forged)
    with pytest.raises(portable.PortabilityError):
        portable.plan_merge([], payload(fields()), [choose(0, "add").model_copy(update={"index": True})])


def test_full_defaults_and_minimal_raw_input_have_the_same_validated_digest():
    minimal = {"name": "Defaults", "model_id": MODEL, "payload": {"prompt": "exact raw"}}
    document = payload(minimal)
    assert portable.payload_digest(document) == portable.payload_digest(payload(public(document.items[0])))


def test_final_size_boundary_matches_the_existing_writer_exactly(monkeypatch):
    document = payload(fields())
    item = document.items[0]
    bounded = {**item.model_dump(), "id": "f" * 32, "revision": 1}
    size = len((json.dumps({"version": 1, "items": [bounded]}, ensure_ascii=False,
                           indent=2, allow_nan=False) + "\n").encode("utf-8"))
    portable_size = len(portable.export_portable([Preset(**item.model_dump(), id="a" * 32, revision=1)]))
    assert portable_size < size
    monkeypatch.setattr(portable, "MAX_PORTABLE_BYTES", size)
    assert portable.plan_merge([], document).add_count == 1
    monkeypatch.setattr(portable, "MAX_PORTABLE_BYTES", size - 1)
    with pytest.raises(portable.PortabilityError, match="merged setups file"):
        portable.plan_merge([], document)
