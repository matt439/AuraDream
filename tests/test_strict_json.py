"""Only syntax and explicit caller limits belong to the shared JSON reader."""
from __future__ import annotations

import pytest

from backend.strict_json import JsonSizeError, JsonSyntaxError, read_utf8_json


def test_byte_and_depth_limits_are_explicit_and_independent():
    assert read_utf8_json(b" {} ", max_bytes=4, max_depth=1) == {}
    with pytest.raises(JsonSizeError):
        read_utf8_json(b" {} ", max_bytes=3, max_depth=1)
    with pytest.raises(JsonSyntaxError):
        read_utf8_json(b"[[]]", max_bytes=4, max_depth=1)
    assert read_utf8_json(b"[[]]", max_bytes=4, max_depth=2) == [[]]


@pytest.mark.parametrize("raw", [
    b'{"x":1,"x":2}', b'{"x":{"y":1,"y":2}}', b'{"x":NaN}',
    b'{"x":Infinity}', b'{"x":-Infinity}', b"{} trailing", b"\xff",
    "{}".encode("utf-16"), "{}".encode("utf-32"),
])
def test_neutral_syntax_errors_do_not_echo_input(raw):
    with pytest.raises(JsonSyntaxError) as error:
        read_utf8_json(raw, max_bytes=1000, max_depth=16)
    assert str(error.value) == "Choose bounded, unambiguous UTF-8 JSON."


def test_text_brackets_escapes_and_empty_objects_keep_json_semantics():
    raw = b'{"text":"[{\\\"}\\\\]", "items":[]}'
    value = read_utf8_json(raw, max_bytes=1000, max_depth=2)
    assert value == {"text": '[{"}\\]', "items": []}
