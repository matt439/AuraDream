"""Cold portable snippet codec and merge planning; no storage or model access.

Plans retain local identities and leave additions without IDs. Only the existing
snippet writer may assign identities and persist a plan under its reservation.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import unicodedata

from .schemas_snippet_portability import (
    MAX_JSON_DEPTH, MAX_PORTABLE_BYTES, PORTABLE_FORMAT,
    ImportDecision, PortableSnippet, PortableSnippets,
)
from .schemas_snippets import MAX_SNIPPETS, Snippet


class PortabilityError(ValueError):
    def __init__(self, message: str, status: int = 422):
        super().__init__(message)
        self.status = status


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    raise ValueError("nonfinite JSON number")


def parse_json(raw: bytes, *, max_bytes: int = MAX_PORTABLE_BYTES) -> object:
    """Bound bytes and nesting before JSON parsing, preserving duplicate keys."""
    if len(raw) > max_bytes:
        raise PortabilityError("The snippet file or request is too large.")
    try:
        text = raw.decode("utf-8")
        depth = 0
        quoted = escaped = False
        for char in text:
            if quoted:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    quoted = False
            elif char == '"':
                quoted = True
            elif char in "[{":
                depth += 1
                if depth > MAX_JSON_DEPTH:
                    raise ValueError("JSON nesting is too deep")
            elif char in "]}":
                depth -= 1
        return json.loads(text, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    except (UnicodeError, ValueError, TypeError, RecursionError) as exc:
        raise PortabilityError("Choose a valid UTF-8 snippet JSON file with no duplicate keys or deep nesting.") from exc


def _encoded(value: object, *, pretty: bool = False) -> bytes:
    try:
        if pretty:
            return (json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode("utf-8")
        return json.dumps(
            value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")
    except (UnicodeError, ValueError, TypeError, RecursionError) as exc:
        raise PortabilityError("The snippet data is invalid.") from exc


def validate_payload(value: object) -> PortableSnippets:
    try:
        # Revalidate even a caller-supplied model; model_copy/construct bypass validation.
        if isinstance(value, PortableSnippets):
            value = value.model_dump()
        payload = PortableSnippets.model_validate(value)
    except (ValueError, TypeError, RecursionError) as exc:
        raise PortabilityError("The portable snippet format, version or items are invalid.") from exc
    if len(_encoded(payload.model_dump(), pretty=True)) > MAX_PORTABLE_BYTES:
        raise PortabilityError("The portable snippet file would exceed 8 MiB.")
    return payload


def parse_portable(raw: bytes) -> PortableSnippets:
    return validate_payload(parse_json(raw))


def _snapshot(items: list[Snippet]) -> list[Snippet]:
    try:
        if len(items) > MAX_SNIPPETS:
            raise ValueError("too many snippets")
        result = [Snippet.model_validate(item.model_dump()) for item in items]
        if len({item.id for item in result}) != len(result):
            raise ValueError("duplicate local ids")
        return result
    except (ValueError, TypeError, RecursionError) as exc:
        raise PortabilityError("The saved snippet snapshot is invalid.") from exc


def export_portable(items: list[Snippet]) -> bytes:
    payload = validate_payload({
        "format": PORTABLE_FORMAT, "version": 1,
        "items": [item.model_dump(include={"name", "text", "target"}) for item in _snapshot(items)],
    })
    return _encoded(payload.model_dump(), pretty=True)


def payload_digest(payload: PortableSnippets) -> str:
    return hashlib.sha256(_encoded(validate_payload(payload).model_dump())).hexdigest()


def collection_revision(items: list[Snippet]) -> str:
    return hashlib.sha256(_encoded([item.model_dump() for item in _snapshot(items)])).hexdigest()


def _name_key(item: PortableSnippet | Snippet) -> tuple[str, str]:
    return unicodedata.normalize("NFC", item.name).casefold(), item.target


def _exact(item: PortableSnippet | Snippet) -> tuple[str, str, str]:
    return item.name, item.text, item.target


@dataclass(frozen=True)
class MergeRow:
    index: int
    action: str
    snippet: PortableSnippet
    matches: list[Snippet]
    exact_ids: list[str]
    incoming_collisions: list[int]
    item_id: str | None = None
    revision: int | None = None


@dataclass(frozen=True)
class MergePlan:
    items: list[Snippet]
    additions: list[PortableSnippet]
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
    items: list[Snippet], payload: PortableSnippets,
    decisions: list[ImportDecision] | None = None,
) -> MergePlan:
    """Plan against one snapshot; missing preview choices take safe defaults."""
    current = _snapshot(items)
    incoming = validate_payload(payload).items
    choices: dict[int, ImportDecision] = {}
    try:
        if decisions is not None and len(decisions) > MAX_SNIPPETS:
            raise ValueError("too many choices")
        for value in decisions or []:
            decision = ImportDecision.model_validate(value.model_dump())
            if decision.index >= len(incoming) or decision.index in choices:
                raise ValueError("invalid file index")
            choices[decision.index] = decision
        effective = [PortableSnippet.model_validate({
            **item.model_dump(),
            **({"name": choices[index].name} if index in choices and choices[index].name is not None else {}),
        }) for index, item in enumerate(incoming)]
    except (ValueError, TypeError, RecursionError) as exc:
        raise PortabilityError("The snippet merge choices are invalid. Preview them again.") from exc

    merged = list(current)
    positions = {item.id: index for index, item in enumerate(current)}
    additions: list[PortableSnippet] = []
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
            raise PortabilityError("A colliding snippet needs Skip, Add as copy or an explicit replacement.")
        if action == "replace":
            local = next((local for local in matches if local.id == item_id), None)
            if local is None:
                raise PortabilityError("Choose an existing matching snippet to replace.")
            if local.revision != revision:
                raise PortabilityError("That snippet changed. Reload the list and preview again.", 409)
            if local.id in replaced:
                raise PortabilityError("Two incoming snippets cannot replace the same saved snippet.")
            replaced.add(local.id)
            merged[positions[local.id]] = Snippet(
                **item.model_dump(), id=local.id, revision=local.revision + 1,
            )
            changed = True
        elif action in {"add", "copy"}:
            additions.append(item)
            changed = True
        rows.append(MergeRow(
            index=index, action=action, snippet=item, matches=matches, exact_ids=exact_ids,
            incoming_collisions=[other for other, candidate in enumerate(effective)
                                 if other != index and _name_key(candidate) == _name_key(item)],
            item_id=item_id, revision=revision,
        ))
    if len(merged) + len(additions) > MAX_SNIPPETS:
        raise PortabilityError(f"The merged list would exceed {MAX_SNIPPETS} snippets.")
    # Match snippets._write exactly, using same-width stand-ins without issuing IDs.
    bounded = [item.model_dump() for item in merged] + [
        {**item.model_dump(), "id": "0" * 32, "revision": 1} for item in additions
    ]
    if len(_encoded({"version": 1, "items": bounded}, pretty=True)) > MAX_PORTABLE_BYTES:
        raise PortabilityError("The merged snippets file would exceed 8 MiB.")
    return MergePlan(items=merged, additions=additions, rows=rows, changed=changed)
