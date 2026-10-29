"""Bounded UTF-8 JSON syntax for native portable documents.

Domain callers own schemas, messages and storage compatibility. In particular,
this reader is not the decoder for existing local sidecar files.
"""
from __future__ import annotations

import json


class JsonSizeError(ValueError):
    """The encoded input exceeds the caller's explicit byte bound."""


class JsonSyntaxError(ValueError):
    """The input is not bounded, unambiguous UTF-8 JSON syntax."""


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    raise ValueError("nonfinite JSON number")


def read_utf8_json(raw: bytes, *, max_bytes: int, max_depth: int) -> object:
    """Check encoded bytes/nesting before decoding objects with duplicate hooks."""
    if len(raw) > max_bytes:
        raise JsonSizeError("JSON input exceeds its byte limit.")
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
                if depth > max_depth:
                    raise ValueError("JSON nesting is too deep")
            elif char in "]}":
                depth -= 1
        return json.loads(text, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    except (UnicodeError, ValueError, TypeError, RecursionError) as exc:
        raise JsonSyntaxError("Choose bounded, unambiguous UTF-8 JSON.") from exc
