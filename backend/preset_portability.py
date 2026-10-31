"""Cold raw-setup portability; no store, UUID, model or execution access.

Plans preserve local identities and row positions. Additions remain identity-free;
only `presets.import_collection` assigns ids and persists a plan, under the
preset-path reservation.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import unicodedata

from pydantic import BaseModel

from .schemas_preset_portability import (
    MAX_JSON_DEPTH, MAX_PORTABLE_BYTES, PORTABLE_FORMAT,
    ImportDecision, MergeRowResponse, PortablePreset, PortablePresets, PresetImportPreview,
)
from .schemas_presets import MAX_PRESETS, Preset
from .strict_json import JsonSizeError, JsonSyntaxError, read_utf8_json


class PortabilityError(ValueError):
    def __init__(self, message: str, status: int = 422):
        super().__init__(message)
        self.status = status


def parse_json(raw: bytes, *, max_bytes: int = MAX_PORTABLE_BYTES) -> object:
    try:
        return read_utf8_json(raw, max_bytes=max_bytes, max_depth=MAX_JSON_DEPTH)
    except JsonSizeError as exc:
        raise PortabilityError("The setup file or request is too large.") from exc
    except JsonSyntaxError as exc:
        raise PortabilityError("Choose a valid UTF-8 setup JSON file with no duplicate keys or deep nesting.") from exc


def _encoded(value: object, *, pretty: bool = False) -> bytes:
    try:
        if pretty:
            return (json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode("utf-8")
        return json.dumps(
            value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")
    except (UnicodeError, ValueError, TypeError, RecursionError) as exc:
        raise PortabilityError("The setup data is invalid.") from exc


def _raw_model_data(value: object) -> object:
    # A model serializer can hide subtype/unchecked extra fields. Revalidate
    # its actual data recursively so constructed inputs cannot strip identities
    # or capture intent before the strict portable schema sees them.
    if isinstance(value, BaseModel):
        return _raw_model_data({**vars(value), **(value.model_extra or {})})
    if isinstance(value, dict):
        return {key: _raw_model_data(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_raw_model_data(item) for item in value]
    return value


def validate_payload(value: object) -> PortablePresets:
    try:
        # Revalidate model_construct/model_copy inputs and every nested raw field.
        payload = PortablePresets.model_validate(_raw_model_data(value))
    except (ValueError, TypeError, RecursionError) as exc:
        raise PortabilityError("The portable setup format, version or items are invalid.") from exc
    if len(_encoded(payload.model_dump(), pretty=True)) > MAX_PORTABLE_BYTES:
        raise PortabilityError("The portable setup file would exceed 8 MiB.")
    return payload


def parse_portable(raw: bytes) -> PortablePresets:
    return validate_payload(parse_json(raw))


def _snapshot(items: list[Preset]) -> list[Preset]:
    try:
        if len(items) > MAX_PRESETS or any(not isinstance(item, Preset) for item in items):
            raise ValueError("invalid local collection")
        result = [Preset.model_validate(_raw_model_data(item)) for item in items]
        if len({item.id for item in result}) != len(result):
            raise ValueError("duplicate local ids")
        return result
    except (ValueError, TypeError, RecursionError) as exc:
        raise PortabilityError("The saved setup snapshot is invalid.") from exc


def export_portable(items: list[Preset]) -> bytes:
    payload = validate_payload({
        "format": PORTABLE_FORMAT, "version": 1,
        "items": [item.model_dump(include={"name", "model_id", "payload"}) for item in _snapshot(items)],
    })
    return _encoded(payload.model_dump(), pretty=True)


def payload_digest(payload: PortablePresets) -> str:
    return hashlib.sha256(_encoded(validate_payload(payload).model_dump())).hexdigest()


def collection_revision(items: list[Preset]) -> str:
    return hashlib.sha256(_encoded([item.model_dump() for item in _snapshot(items)])).hexdigest()


def _name_key(item: PortablePreset | Preset) -> tuple[str, str]:
    return unicodedata.normalize("NFC", item.name).casefold(), item.model_id


def _exact(item: PortablePreset | Preset) -> bytes:
    return _encoded(item.model_dump(include={"name", "model_id", "payload"}))


@dataclass(frozen=True)
class MergeRow:
    index: int
    action: str
    preset: PortablePreset
    matches: list[Preset]
    exact_ids: list[str]
    incoming_collisions: list[int]
    item_id: str | None = None
    revision: int | None = None


@dataclass(frozen=True)
class MergePlan:
    items: list[Preset]
    additions: list[PortablePreset]
    rows: list[MergeRow]
    changed: bool

    @property
    def add_count(self) -> int:
        return len(self.additions)

    @property
    def replace_count(self) -> int:
        return sum(row.action == "replace" for row in self.rows)

    @property
    def skip_count(self) -> int:
        return sum(row.action == "skip" for row in self.rows)


def plan_merge(
    items: list[Preset], payload: PortablePresets,
    decisions: list[ImportDecision] | None = None, *, require_complete: bool = False,
) -> MergePlan:
    """Preview defaults are safe; a later commit requires every explicit choice."""
    current = _snapshot(items)
    incoming = validate_payload(payload).items
    choices: dict[int, ImportDecision] = {}
    try:
        if decisions is not None and len(decisions) > MAX_PRESETS:
            raise ValueError("too many choices")
        for value in decisions or []:
            if not isinstance(value, ImportDecision):
                raise ValueError("invalid choice type")
            decision = ImportDecision.model_validate(_raw_model_data(value))
            if decision.index >= len(incoming) or decision.index in choices:
                raise ValueError("invalid file index")
            choices[decision.index] = decision
        if require_complete and len(choices) != len(incoming):
            raise ValueError("every file index needs an explicit choice")
        effective = [PortablePreset.model_validate({
            **item.model_dump(),
            **({"name": choices[index].name} if index in choices and choices[index].name is not None else {}),
        }) for index, item in enumerate(incoming)]
    except (ValueError, TypeError, RecursionError) as exc:
        raise PortabilityError("The setup merge choices are invalid. Preview them again.") from exc

    merged = list(current)
    positions = {item.id: index for index, item in enumerate(current)}
    additions: list[PortablePreset] = []
    rows: list[MergeRow] = []
    replaced: set[str] = set()
    changed = False
    for index, item in enumerate(effective):
        matches = [local for local in current if _name_key(local) == _name_key(item)]
        exact_ids = [local.id for local in matches if _exact(local) == _exact(item)]
        decision = choices.get(index)
        action = decision.action if decision else ("skip" if matches else "add")
        item_id = decision.item_id if decision else None
        revision = decision.revision if decision else None
        if action == "add" and matches:
            raise PortabilityError("A colliding setup needs Skip, Add as copy or an explicit replacement.")
        if action == "replace":
            local = next((candidate for candidate in matches if candidate.id == item_id), None)
            if local is None:
                raise PortabilityError("Choose an existing matching setup to replace.")
            if local.revision != revision:
                raise PortabilityError("That setup changed. Reload the list and preview again.", 409)
            if local.id in replaced:
                raise PortabilityError("Two incoming setups cannot replace the same saved setup.")
            replaced.add(local.id)
            merged[positions[local.id]] = Preset(**item.model_dump(), id=local.id, revision=local.revision + 1)
            changed = True
        elif action in {"add", "copy"}:
            additions.append(item)
            changed = True
        rows.append(MergeRow(
            index=index, action=action, preset=item, matches=matches, exact_ids=exact_ids,
            incoming_collisions=[other for other, candidate in enumerate(effective)
                                 if other != index and _name_key(candidate) == _name_key(item)],
            item_id=item_id, revision=revision,
        ))
    if len(merged) + len(additions) > MAX_PRESETS:
        raise PortabilityError(f"The merged list would exceed {MAX_PRESETS} setups.")
    # Match presets._write's exact insertion order/encoding using same-width
    # placeholders; never import the store or allocate UUIDs during planning.
    bounded = [item.model_dump() for item in merged] + [
        {**item.model_dump(), "id": "0" * 32, "revision": 1} for item in additions
    ]
    if len(_encoded({"version": 1, "items": bounded}, pretty=True)) > MAX_PORTABLE_BYTES:
        raise PortabilityError("The merged setups file would exceed 8 MiB.")
    if require_complete and not changed:
        raise PortabilityError("The selected import changes nothing.")
    return MergePlan(items=merged, additions=additions, rows=rows, changed=changed)


def preview(
    items: list[Preset], payload: PortablePresets, decisions: list[ImportDecision] | None = None,
) -> PresetImportPreview:
    plan = plan_merge(items, payload, decisions)
    return PresetImportPreview(
        collection_revision=collection_revision(items), payload_digest=payload_digest(payload),
        rows=[MergeRowResponse.model_validate(vars(row)) for row in plan.rows],
        add_count=plan.add_count, replace_count=plan.replace_count, skip_count=plan.skip_count,
        changed=plan.changed, can_import=plan.changed,
        reason="" if plan.changed else "The selected import changes nothing.",
    )
