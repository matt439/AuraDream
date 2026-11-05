"""The server's card facts, and the Nunchaku format a card defaults to (GEN#42).

No GPU: torch is a stand-in, and `card_facts.card` is patched per test
(tests/conftest.py makes every other test's card unknown).
"""
from __future__ import annotations

import io
from types import SimpleNamespace

import pytest

from backend import card_facts, launch, main, model_selection
from backend.config import Settings
from backend.doctor import check_vram
from backend.launch import CHECKPOINTS
from backend.loader_choices import default_format

GIB = 1024 ** 3
RTX_3070 = card_facts.Card(8 * GIB, (8, 6))
RTX_5080 = card_facts.Card(16 * GIB, (12, 0))


def answers(*lines: str):
    """An `ask` that hands out the given answers in turn and refuses to be asked more."""
    queue = list(lines)

    def ask(_prompt: str) -> str:
        assert queue, "the menu asked one question more than the test expected"
        return queue.pop(0)

    return ask


def fake_torch(total: int = 8 * GIB, capability=(8, 6), *, cuda="13.0", available=True, fail=False):
    def properties(index):
        if fail:
            raise RuntimeError("no device")
        assert index == seen["index"]
        return SimpleNamespace(total_memory=total)

    seen = {"index": 0}
    import torch

    module = SimpleNamespace(
        device=torch.device,
        version=SimpleNamespace(cuda=cuda),
        cuda=SimpleNamespace(
            is_available=lambda: available,
            get_device_properties=properties,
            get_device_capability=lambda index: capability,
        ),
    )
    return module, seen


def test_probe_reads_total_memory_and_capability():
    torch, _ = fake_torch()
    assert card_facts.probe("cuda", torch) == RTX_3070


def test_probe_reads_the_named_cuda_index():
    torch, seen = fake_torch(capability=(12, 0), total=16 * GIB)
    seen["index"] = 1
    assert card_facts.probe("cuda:1", torch) == RTX_5080


@pytest.mark.parametrize("device, kwargs", [
    ("cpu", {}),                       # AURADREAM_DEVICE=cpu: no card to ask
    ("cuda", {"cuda": None}),          # a CPU build of torch
    ("cuda", {"available": False}),    # no driver or no card
    ("cuda", {"fail": True}),          # any error reading it
    ("not a device", {}),
])
def test_probe_answers_none_when_the_card_is_unknown(device, kwargs):
    torch, _ = fake_torch(**kwargs)
    assert card_facts.probe(device, torch) is None


def test_the_helpers_read_the_cached_card(monkeypatch):
    monkeypatch.setattr(card_facts, "card", lambda device="cuda": RTX_3070)
    assert card_facts.total_memory() == 8 * GIB and card_facts.capability() == (8, 6)
    monkeypatch.setattr(card_facts, "card", lambda device="cuda": None)
    assert card_facts.total_memory() is None and card_facts.capability() is None


@pytest.mark.parametrize("capability, expected", [
    (None, "nvfp4"), ((7, 5), "int4"), ((8, 6), "int4"), ((8, 9), "int4"),
    ((10, 0), "nvfp4"), ((12, 0), "nvfp4"),
])
def test_default_format_is_int4_below_blackwell_and_nvfp4_otherwise(capability, expected):
    assert default_format(capability) == expected


@pytest.fixture
def linux_nunchaku(monkeypatch):
    monkeypatch.setattr(model_selection, "_selection", (None, 0))
    monkeypatch.setattr(model_selection, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setenv("DIFFUSERS_TRUST_REMOTE_KERNELS", "true")
    monkeypatch.setitem(model_selection.imported_modules, "diffusers.utils.constants", SimpleNamespace(
        DIFFUSERS_TRUST_REMOTE_KERNELS=True,
    ))


def _card(monkeypatch, found, devices=None):
    def card(device="cuda"):
        if devices is not None:
            devices.append(device)
        return found
    monkeypatch.setattr(card_facts, "card", card)


@pytest.mark.parametrize("found, expected", [(RTX_3070, "int4"), (RTX_5080, "nvfp4"), (None, "nvfp4")])
def test_an_unchosen_format_follows_the_card(monkeypatch, linux_nunchaku, found, expected):
    devices: list[str] = []
    _card(monkeypatch, found, devices)
    startup = Settings(model_id=CHECKPOINTS[0].repo, device="cuda:1")
    assert model_selection.format_for(startup) == expected
    assert devices and set(devices) == {"cuda:1"}, "the selected device's card is the one asked"
    assert model_selection.available(startup, "nunchaku") == [
        {"model_id": c.lite_repo(expected), "label": c.name} for c in CHECKPOINTS if c.lite
    ]


@pytest.mark.parametrize("found", [RTX_3070, RTX_5080, None])
def test_a_chosen_format_is_kept_whatever_the_card(monkeypatch, linux_nunchaku, found):
    _card(monkeypatch, found)
    startup = Settings(model_id=CHECKPOINTS[0].repo)
    for fmt in ("nvfp4", "int4"):
        assert model_selection.available(startup, "nunchaku", fmt)[0]["model_id"] == CHECKPOINTS[0].lite_repo(fmt)
        target = model_selection.validate(startup, CHECKPOINTS[0].lite_repo(fmt), quantization="nunchaku")
        model_selection.select(target.model_id, quantization=target.quantization)
        assert model_selection.format_for(startup) == fmt


@pytest.mark.parametrize("found, marked", [(RTX_3070, "int4"), (RTX_5080, "nvfp4"), (None, None)])
def test_the_formats_name_this_cards_once_it_is_known(monkeypatch, linux_nunchaku, found, marked):
    _card(monkeypatch, found)
    formats = model_selection.options(Settings())["formats"]
    assert [f["value"] for f in formats] == ["nvfp4", "int4"]
    for entry in formats:
        assert entry["description"].endswith(" (this card)") == (entry["value"] == marked)
    if marked is None:
        assert [f["description"] for f in formats] == ["RTX 50 (Blackwell)", "RTX 20 to 40"]


def test_status_and_catalogue_carry_the_cards_format(client, monkeypatch, linux_nunchaku):
    _card(monkeypatch, RTX_3070)
    assert client.get("/api/status").json()["format"] == "int4"
    choices = client.get("/api/models", params={"quantization": "nunchaku"}).json()
    assert choices["format"] == "int4"
    assert choices["models"][0]["model_id"] == CHECKPOINTS[0].lite_repo("int4")
    assert "(this card)" in next(f for f in choices["formats"] if f["value"] == "int4")["description"]


def test_the_routes_read_the_card_off_the_event_loop(client, monkeypatch, linux_nunchaku):
    import asyncio

    on_loop: list[bool] = []

    def card(device="cuda"):
        try:
            asyncio.get_running_loop()
            on_loop.append(True)
        except RuntimeError:
            on_loop.append(False)
        return RTX_3070

    monkeypatch.setattr(card_facts, "card", card)
    reads = []
    original = main._read_card

    async def counted():
        reads.append(True)
        await original()

    monkeypatch.setattr(main, "_read_card", counted)
    client.get("/api/models")
    client.post("/api/model/switch", json={"model_id": "not/in-the-catalogue"})
    assert len(reads) == 2, "the catalogue and the switch each read the card first"
    on_loop.clear()
    client.get("/api/status")
    # The status's describe thread reads it before the loop's format default.
    assert on_loop[0] is False


def test_the_menu_defaults_to_the_cards_format(monkeypatch):
    devices: list[str] = []
    _card(monkeypatch, RTX_3070, devices)
    choice = launch.prompt(ask=answers("nunchaku", "", "1", ""), out=io.StringIO(),
                           env={"AURADREAM_DEVICE": "cuda:1"}, linux=True)
    assert choice.model_id == CHECKPOINTS[0].lite_repo("int4")
    assert devices == ["cuda:1"]


def test_the_menu_keeps_nvfp4_on_blackwell_and_an_unknown_card(monkeypatch):
    for found in (RTX_5080, None):
        _card(monkeypatch, found)
        choice = launch.prompt(ask=answers("nunchaku", "", "1", ""), out=io.StringIO(), env={}, linux=True)
        assert choice.model_id == CHECKPOINTS[0].lite_repo("nvfp4")


def test_the_menu_keeps_a_lite_repo_in_the_environment_whatever_the_card(monkeypatch):
    _card(monkeypatch, RTX_3070)
    env = {"AURADREAM_MODEL_ID": CHECKPOINTS[0].lite_repo("nvfp4"), "AURADREAM_QUANT": "nunchaku"}
    choice = launch.prompt(ask=answers("", "", "", ""), out=io.StringIO(), env=env, linux=True)
    assert choice.model_id == env["AURADREAM_MODEL_ID"]


def test_the_menu_asks_no_card_without_nunchaku(monkeypatch):
    devices: list[str] = []
    _card(monkeypatch, RTX_3070, devices)
    launch.prompt(ask=answers("", "", ""), out=io.StringIO(), env={}, linux=True)
    assert devices == []


def test_the_doctor_names_int4_by_the_same_rule():
    torch, _ = fake_torch()
    assert "int4" in check_vram(torch, platform="linux")[0].detail
    torch, _ = fake_torch(capability=(12, 0), total=16 * GIB)
    assert "int4" not in check_vram(torch, platform="linux")[0].detail
