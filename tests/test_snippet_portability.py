"""Portable snippets preserve text and require explicit, bounded merge decisions."""
from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from backend import snippet_portability as portable
from backend.schemas_snippet_portability import ImportDecision, PortableSnippet, PortableSnippets
from backend.schemas_snippets import MAX_SNIPPETS, Snippet


FORMAT = "auradream.prompt-snippets"
MAX_BYTES = 8 * 1024 * 1024


def item(name="Style", text="soft light", target="prompt"):
    return {"name": name, "text": text, "target": target}


def payload(*items):
    return portable.parse_portable(json.dumps({
        "format": FORMAT, "version": 1, "items": list(items),
    }, ensure_ascii=False).encode("utf-8"))


def saved(number, name="Style", text="soft light", target="prompt", revision=1):
    return Snippet(id=f"{number:032x}", revision=revision, **item(name, text, target))


def choose(index, action, **kwargs):
    return ImportDecision(index=index, action=action, **kwargs)


def public_fields(snippet):
    return snippet.model_dump(include={"name", "text", "target"})


def test_export_round_trip_preserves_saved_order_exact_text_and_both_targets():
    collection = [
        saved(1, "Caf\u00e9 \U0001f3a8", "  first line\n\n\u65e5\u672c\u8a9e\t\U0001f308  ", revision=7),
        saved(2, "Negative", "blurry\r\n  lowres  ", "negative", revision=3),
    ]
    before = [snippet.model_dump() for snippet in collection]
    encoded = portable.export_portable(collection)
    assert isinstance(encoded, bytes)
    decoded = json.loads(encoded.decode("utf-8"))
    assert set(decoded) == {"format", "version", "items"}
    assert decoded == {"format": FORMAT, "version": 1, "items": [public_fields(x) for x in collection]}
    restored = portable.parse_portable(encoded)
    assert [public_fields(x) for x in restored.items] == decoded["items"]
    assert [snippet.model_dump() for snippet in collection] == before


def test_empty_export_and_import_are_valid_and_do_not_clear_the_destination():
    empty = portable.parse_portable(portable.export_portable([]))
    assert empty.items == []
    collection = [saved(1)]
    plan = portable.plan_merge(collection, empty)
    assert plan.items == collection and plan.additions == [] and plan.rows == []
    assert (plan.add_count, plan.replace_count, plan.skip_count, plan.changed) == (0, 0, 0, False)


@pytest.mark.parametrize("raw", [
    b"", b"not JSON", b"[]", b"null", b"{}",
    b'{"format":"auradream.prompt-snippets","version":1,"items":[]} trailing',
    b'{"format":"auradream.prompt-snippets","version":1,"items":[],"items":[]}',
    b'{"format":"auradream.prompt-snippets","version":1,"version":1,"items":[]}',
    b'{"format":"auradream.prompt-snippets","version":1,"items":[{"name":"a","name":"b","text":"c","target":"prompt"}]}',
    b'{"format":"auradream.prompt-snippets","version":NaN,"items":[]}',
    b'{"format":"auradream.prompt-snippets","version":Infinity,"items":[]}',
    b'{"format":"auradream.prompt-snippets","version":1,"items":[{"name":"a","text":"\\ud800","target":"prompt"}]}',
    b'\xff', b"[" * 2000 + b"]" * 2000,
])
def test_malformed_portable_bytes_are_fixed_validation_errors(raw):
    with pytest.raises(portable.PortabilityError) as error:
        portable.parse_portable(raw)
    assert error.value.status == 422


@pytest.mark.parametrize("encoding", ["utf-16", "utf-32"])
def test_only_utf8_is_accepted_even_when_json_decoder_supports_other_encodings(encoding):
    document = json.dumps({"format": FORMAT, "version": 1, "items": []})
    with pytest.raises(portable.PortabilityError):
        portable.parse_portable(document.encode(encoding))


@pytest.mark.parametrize("change", [
    {"format": "local-diffusion.prompt-snippets"}, {"format": FORMAT + ".new"},
    {"version": 0}, {"version": 2}, {"version": True}, {"version": 1.0}, {"version": "1"},
    {"unknown": "PERSONAL TEXT MUST NOT BE ECHOED"},
    {"items": {}}, {"items": None},
])
def test_format_version_and_top_level_fields_are_strict(change):
    document = {"format": FORMAT, "version": 1, "items": [], **change}
    with pytest.raises(portable.PortabilityError) as error:
        portable.parse_portable(json.dumps(document).encode("utf-8"))
    assert "PERSONAL TEXT MUST NOT BE ECHOED" not in str(error.value)


@pytest.mark.parametrize("fields", [
    {"name": "Style", "text": "text"},
    item(name=""), item(name=" \t "), item(name="a\nb"), item(name="a" * 81),
    item(text=""), item(text="\n \t"), item(text="x" * 16001),
    item(target="path"), item(target=1), item(name=7), item(text=False),
    {**item(), "id": "0" * 32}, {**item(), "revision": 1},
    {**item(), "path": "../private"}, {**item(), "unknown": "private"},
])
def test_portable_items_reuse_strict_fields_without_accepting_local_identity(fields):
    with pytest.raises(portable.PortabilityError):
        payload(fields)


def test_maximum_name_text_and_collection_are_accepted_but_one_more_item_is_refused():
    entries = [item(name="n" * 80, text="x" * 16000, target="negative") for _ in range(MAX_SNIPPETS)]
    accepted = payload(*entries)
    assert len(accepted.items) == MAX_SNIPPETS
    assert accepted.items[0].text == "x" * 16000
    with pytest.raises(portable.PortabilityError):
        payload(*entries, item())


def test_raw_byte_limit_counts_whitespace_before_json_parsing():
    document = portable.export_portable([])
    exactly_at_limit = document + b" " * (MAX_BYTES - len(document))
    assert portable.parse_portable(exactly_at_limit).items == []
    with pytest.raises(portable.PortabilityError):
        portable.parse_portable(exactly_at_limit + b" ")


def test_braces_and_escaped_quotes_inside_text_are_not_counted_as_json_nesting():
    text = ('[{{{ "quoted" \\ }}}]' * 40) + "\nlast line"
    assert payload(item(text=text)).items[0].text == text


def test_json_depth_boundary_is_checked_before_the_decoder():
    assert portable.parse_json(b"[" * 16 + b"0" + b"]" * 16)
    with pytest.raises(portable.PortabilityError):
        portable.parse_json(b"[" * 17 + b"0" + b"]" * 17)


def test_unvalidated_model_copies_cannot_smuggle_identity_or_invalid_content():
    document = payload(item())
    bad_item = document.items[0].model_copy(update={"target": "path"})
    with pytest.raises(portable.PortabilityError):
        portable.plan_merge([], document.model_copy(update={"items": [bad_item]}))
    with pytest.raises(portable.PortabilityError):
        portable.export_portable([saved(1).model_copy(update={"revision": True})])


def test_payload_digest_is_stable_for_validated_content_and_changes_with_exact_text():
    original = payload(item(name="  Caf\u00e9  ", text="x\ny"))
    equivalent = portable.parse_portable(json.dumps({
        "items": [item(name="Caf\u00e9", text="x\ny")], "version": 1, "format": FORMAT,
    }, ensure_ascii=True, indent=4).encode("utf-8"))
    assert portable.payload_digest(original) == portable.payload_digest(equivalent)
    assert portable.payload_digest(original) != portable.payload_digest(payload(item(name="Caf\u00e9", text="x\ny ")))


def test_collection_revision_covers_order_identity_item_revision_and_content():
    first, second = saved(1), saved(2, "Other")
    original = [first, second]
    revision = portable.collection_revision(original)
    assert isinstance(revision, str) and revision
    assert revision == portable.collection_revision([first.model_copy(deep=True), second.model_copy(deep=True)])
    variants = [
        [second, first], [first], [],
        [saved(3), second], [saved(1, revision=2), second],
        [saved(1, text="soft light "), second],
        [saved(1, name="STYLE"), second], [saved(1, target="negative"), second],
    ]
    assert all(portable.collection_revision(variant) != revision for variant in variants)


def test_default_merge_separates_exact_matches_conflicts_new_names_and_targets():
    collection = [saved(1), saved(2, text="another version"), saved(3, "Caf\u00e9", "local")]
    incoming = payload(
        item(), item(text="changed"), item(name="New"),
        item(name="Cafe\u0301", text="incoming"), item(target="negative"),
    )
    plan = portable.plan_merge(collection, incoming)
    assert [row.action for row in plan.rows] == ["skip", "skip", "add", "skip", "add"]
    assert plan.rows[0].exact_ids == [collection[0].id]
    assert {x.id for x in plan.rows[0].matches} == {collection[0].id, collection[1].id}
    assert plan.rows[1].exact_ids == []
    assert [x.id for x in plan.rows[3].matches] == [collection[2].id]
    assert plan.items == collection
    assert [x.name for x in plan.additions] == ["New", "Style"]
    assert (plan.add_count, plan.replace_count, plan.skip_count, plan.changed) == (2, 0, 3, True)


def test_default_round_trip_skips_every_exact_copy_including_existing_duplicate_names():
    collection = [saved(1), saved(2, text="different"), saved(3)]
    plan = portable.plan_merge(collection, portable.parse_portable(portable.export_portable(collection)))
    assert [row.action for row in plan.rows] == ["skip", "skip", "skip"]
    assert plan.rows[0].exact_ids == [collection[0].id, collection[2].id]
    assert (plan.add_count, plan.replace_count, plan.skip_count, plan.changed) == (0, 0, 3, False)
    assert plan.items == collection and plan.additions == []


def test_incoming_collisions_are_file_indexed_and_never_silently_collapsed():
    incoming = payload(item(name="STRASSE"), item(name="Stra\u00dfe", text="other"), item(name="Strasse", target="negative"))
    plan = portable.plan_merge([], incoming)
    assert [row.index for row in plan.rows] == [0, 1, 2]
    assert [row.incoming_collisions for row in plan.rows] == [[1], [0], []]
    assert [public_fields(x) for x in plan.additions] == [public_fields(x) for x in incoming.items]
    assert plan.add_count == 3


def test_explicit_copy_add_skip_and_replace_preserve_rows_positions_and_addition_order():
    collection = [saved(1), saved(2, text="second", revision=4), saved(3, "Untouched")]
    incoming = payload(item(), item(text="replacement"), item(name="New"), item(name="Skipped"))
    decisions = [
        choose(2, "add"), choose(1, "replace", item_id=collection[1].id, revision=4),
        choose(0, "copy", name="  Copied style  "), choose(3, "skip"),
    ]
    plan = portable.plan_merge(collection, incoming, decisions)
    assert [x.id for x in plan.items] == [x.id for x in collection]
    assert plan.items[0] == collection[0] and plan.items[2] == collection[2]
    assert plan.items[1].text == "replacement" and plan.items[1].revision == 5
    assert [x.name for x in plan.additions] == ["Copied style", "New"]
    assert [row.action for row in plan.rows] == ["copy", "replace", "add", "skip"]
    assert (plan.add_count, plan.replace_count, plan.skip_count, plan.changed) == (2, 1, 1, True)


def test_partial_decisions_keep_other_default_choices():
    collection = [saved(1)]
    incoming = payload(item(), item(name="New"))
    plan = portable.plan_merge(collection, incoming, [choose(0, "copy")])
    assert [row.action for row in plan.rows] == ["copy", "add"]
    assert plan.add_count == 2 and plan.skip_count == 0


def test_add_cannot_overwrite_or_bypass_a_local_name_collision():
    with pytest.raises(portable.PortabilityError):
        portable.plan_merge([saved(1)], payload(item(text="different")), [choose(0, "add")])
    plan = portable.plan_merge([saved(1)], payload(item(text="different")), [choose(0, "copy")])
    assert plan.items == [saved(1)] and plan.additions[0].text == "different"


def test_replacement_selects_one_explicit_candidate_among_duplicate_names():
    collection = [saved(1), saved(2, text="second", revision=6)]
    plan = portable.plan_merge(collection, payload(item(text="incoming")), [
        choose(0, "replace", item_id=collection[1].id, revision=6),
    ])
    assert plan.items[0] == collection[0]
    assert plan.items[1].id == collection[1].id
    assert plan.items[1].revision == 7 and plan.items[1].text == "incoming"
    assert plan.additions == []


@pytest.mark.parametrize("target", [saved(2, "Other"), saved(2, target="negative")])
def test_replace_refuses_an_unrelated_id_or_target(target):
    with pytest.raises(portable.PortabilityError):
        portable.plan_merge([saved(1), target], payload(item(text="incoming")), [
            choose(0, "replace", item_id=target.id, revision=target.revision),
        ])


def test_stale_replacement_is_409_without_mutating_any_source():
    collection = [saved(1, revision=3)]
    incoming = payload(item(text="incoming"))
    before_collection = [x.model_dump() for x in collection]
    before_payload = incoming.model_dump()
    with pytest.raises(portable.PortabilityError) as error:
        portable.plan_merge(collection, incoming, [choose(0, "replace", item_id=collection[0].id, revision=2)])
    assert error.value.status == 409
    assert [x.model_dump() for x in collection] == before_collection
    assert incoming.model_dump() == before_payload


def test_two_incoming_rows_cannot_replace_one_local_item():
    local = saved(1)
    with pytest.raises(portable.PortabilityError):
        portable.plan_merge([local], payload(item(text="first"), item(text="second")), [
            choose(0, "replace", item_id=local.id, revision=1),
            choose(1, "replace", item_id=local.id, revision=1),
        ])


@pytest.mark.parametrize("decisions", [
    [choose(0, "skip"), choose(0, "copy")], [choose(1, "skip")],
])
def test_duplicate_and_out_of_range_decision_indices_are_refused(decisions):
    with pytest.raises(portable.PortabilityError):
        portable.plan_merge([], payload(item()), decisions)


@pytest.mark.parametrize("fields", [
    {"index": -1, "action": "skip"}, {"index": True, "action": "skip"},
    {"index": 0.0, "action": "skip"}, {"index": 0, "action": "delete"},
    {"index": 0, "action": "replace"},
    {"index": 0, "action": "replace", "item_id": "0" * 32},
    {"index": 0, "action": "replace", "item_id": "../x", "revision": 1},
    {"index": 0, "action": "replace", "item_id": "0" * 32, "revision": True},
    {"index": 0, "action": "skip", "item_id": "0" * 32, "revision": 1},
    {"index": 0, "action": "copy", "revision": 1},
    {"index": 0, "action": "copy", "name": " "},
    {"index": 0, "action": "copy", "name": "x" * 81},
    {"index": 0, "action": "copy", "unknown": True},
])
def test_decisions_require_strict_bounded_fields_and_replacement_identity(fields):
    with pytest.raises((ValidationError, portable.PortabilityError)):
        decision = ImportDecision.model_validate(fields)
        portable.plan_merge([], payload(item()), [decision])


def test_final_count_limit_includes_existing_and_all_added_copies():
    collection = [saved(index, name=f"Local {index}") for index in range(MAX_SNIPPETS)]
    plan = portable.plan_merge(collection, payload(public_fields(collection[0])))
    assert not plan.changed
    with pytest.raises(portable.PortabilityError):
        portable.plan_merge(collection, payload(item(name="Extra")))
    with pytest.raises(portable.PortabilityError):
        portable.plan_merge(collection, payload(public_fields(collection[0])), [choose(0, "copy")])


def test_final_byte_limit_counts_escaped_saved_json_even_when_each_input_fits():
    # Every individual collection is below 8 MiB; 100 escaped texts are above it.
    text = "\x00" * 15999 + "x"
    collection = [saved(index, name=f"Local {index}", text=text) for index in range(50)]
    incoming = payload(*(item(name=f"Imported {index}", text=text) for index in range(50)))
    before = [x.model_dump() for x in collection]
    assert len(portable.export_portable(collection)) < MAX_BYTES
    assert len(portable.export_portable([
        saved(index, **public_fields(x)) for index, x in enumerate(incoming.items)
    ])) < MAX_BYTES
    with pytest.raises(portable.PortabilityError):
        portable.plan_merge(collection, incoming)
    assert [x.model_dump() for x in collection] == before


def test_successful_planning_does_not_mutate_source_models_or_forge_new_local_ids():
    collection = [saved(1)]
    incoming = payload(item(text="replacement"), item(name="New"))
    decisions = [choose(0, "replace", item_id=collection[0].id, revision=1)]
    before = ([x.model_dump() for x in collection], incoming.model_dump(), [x.model_dump() for x in decisions])
    plan = portable.plan_merge(collection, incoming, decisions)
    assert isinstance(incoming, PortableSnippets)
    assert all(isinstance(x, PortableSnippet) for x in plan.additions)
    assert all(set(x.model_dump()) == {"name", "text", "target"} for x in plan.additions)
    assert ([x.model_dump() for x in collection], incoming.model_dump(), [x.model_dump() for x in decisions]) == before


def test_explicit_exact_replacement_is_a_revisioned_edit():
    local = saved(1, revision=5)
    plan = portable.plan_merge([local], payload(item()), [
        choose(0, "replace", item_id=local.id, revision=5),
    ])
    assert plan.changed and plan.replace_count == 1
    assert plan.items[0].id == local.id and plan.items[0].revision == 6
    assert public_fields(plan.items[0]) == public_fields(local)
