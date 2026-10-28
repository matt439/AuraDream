"""Malformed embedded JSON stays a named PNG refusal, including file reads."""
from __future__ import annotations

import json

import pytest
from PIL import Image, PngImagePlugin

from backend import png_settings

_KEYS = ("auradream", "local-diffusion", "krea")
_PARAMETERS = "usable fallback prompt\nSteps: 20, Seed: 1, Size: 512x512"


def _nested_json() -> str:
    # A string inside the bounded HTTP body, not a deeply nested request body.
    depth = 10_000
    return '{"unused":' + "[" * depth + "0" + "]" * depth + "}"


@pytest.mark.parametrize("key", _KEYS)
def test_app_metadata_decoder_depth_is_a_named_refusal(key):
    with pytest.raises(png_settings.PngSettingsError, match="metadata in nested.png is not JSON"):
        png_settings.record({key: _nested_json(), "parameters": _PARAMETERS}, "nested.png")


@pytest.mark.parametrize("key", _KEYS)
def test_http_keeps_app_metadata_precedence_when_its_decoder_refuses(client, key):
    response = client.post("/api/png/settings", json={
        "name": "nested.png", "chunks": {key: _nested_json(), "parameters": _PARAMETERS},
    })
    assert response.status_code == 422
    assert response.json() == {"detail": "The AuraDream metadata in nested.png is not JSON."}
    assert client.get("/api/gallery?all_models=true").json()["total"] == 0
    assert client.get("/api/jobs").json() == []


@pytest.mark.parametrize(("chunks", "detail"), [
    ({"workflow": _nested_json(), "prompt": json.dumps({"1": {}})}, "carries a ComfyUI workflow"),
    ({"prompt": _nested_json()}, "carries no settings this app reads"),
    ({"workflow": _nested_json(), "prompt": _nested_json()}, "carries no settings this app reads"),
])
def test_workflow_decoder_depth_does_not_escape_or_hide_another_chunk(client, chunks, detail):
    response = client.post("/api/png/settings", json={"name": "workflow.png", "chunks": chunks})
    assert response.status_code == 422
    assert response.json()["detail"].startswith("workflow.png " + detail)


@pytest.mark.parametrize("key", _KEYS)
def test_owned_png_file_uses_the_same_named_decoder_refusal(tmp_path, key):
    path = tmp_path / "nested.png"
    chunks = PngImagePlugin.PngInfo()
    chunks.add_text(key, _nested_json())
    chunks.add_text("parameters", _PARAMETERS)
    with Image.new("RGB", (1, 1)) as image:
        image.save(path, pnginfo=chunks)
    with pytest.raises(png_settings.PngSettingsError, match="is not JSON"):
        png_settings.image_settings(str(path))
