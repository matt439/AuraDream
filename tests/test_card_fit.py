"""The catalogue's fit badge sized from the server's card (GEN#43).

No GPU: `card_memory` is handed to `model_info.details_for` directly, or
`card_facts.card` is patched for the HTTP listing (tests/conftest.py makes
every other test's card unknown).
"""
from __future__ import annotations

import pytest

from backend import card_facts, model_info
from backend.doctor import DEFAULT_PEAK_GIB
from backend.launch import CHECKPOINTS, FORMATS

GIB = 1024 ** 3
# torch's total_memory: the 3070 reports a little under its 8 GB, the 5080
# 16,303 MiB.
RTX_3070 = 7.79 * GIB
RTX_5080 = 16_303 * 1024 ** 2
QUANTS = ("fp8", "int8", "none", "nunchaku", None)
OFFLOADS = ("model", "sequential", "none")

KREA_DEV = "black-forest-labs/FLUX.1-Krea-dev"
KONTEXT = "black-forest-labs/FLUX.1-Kontext-dev"
FLUX_SIZE = {
    KREA_DEV, "black-forest-labs/FLUX.1-dev", "black-forest-labs/FLUX.1-schnell",
    "krea/Krea-2-Turbo", "krea/Krea-2-Raw", "black-forest-labs/FLUX.2-klein-9B",
    "black-forest-labs/FLUX.2-klein-base-9B", "baidu/ERNIE-Image-Turbo", KONTEXT,
}


def _ids():
    for entry in CHECKPOINTS:
        yield entry.repo
        for fmt, _ in FORMATS:
            if entry.lite_repo(fmt):
                yield entry.lite_repo(fmt)


def _bytes(value: float) -> int:
    return int(value)


@pytest.mark.parametrize("card", [None, RTX_5080, 24 * GIB])
def test_an_unknown_or_large_card_keeps_every_row_as_it_was(card):
    # The 5080's catalogue, byte for byte: nothing it lists needs more.
    for model_id in [*_ids(), "local/custom-checkpoint"]:
        for quant in QUANTS:
            for offload in OFFLOADS:
                assert model_info.details_for(
                    model_id, quantization=quant, offload=offload,
                    card_memory=None if card is None else _bytes(card),
                ) == model_info.details_for(model_id, quantization=quant, offload=offload), model_id


def test_an_8_gb_card_flags_the_flux_size_official_rows():
    for entry in CHECKPOINTS:
        detail = model_info.details_for(entry.repo, quantization="fp8", card_memory=_bytes(RTX_3070))
        if entry.repo in FLUX_SIZE:
            assert detail["fit"] == "Over 8 GB", entry.repo
            assert "this card has 7.8 GiB" in detail["fit_note"]
            assert "Allocator peaks leave out the CUDA context and the desktop." in detail["fit_note"]
        elif entry.repo == "Qwen/Qwen-Image":
            assert detail["fit"] == "Over 16 GB"   # its own record, larger still
        else:
            assert detail["fit"] == detail["fit_note"] == "", entry.repo


def test_the_default_quotes_its_measured_peak():
    detail = model_info.details_for(KREA_DEV, quantization="fp8", card_memory=_bytes(RTX_3070))
    assert detail["fit_note"].startswith(
        "FP8 with model offload peaked at 12.00 GiB allocated (12.63 reserved) at 1024×1024 "
        "on the 16 GB RTX 5080 (2026-10-27); this card has 7.8 GiB."
    )
    assert "Sequential offload needs far less" in detail["fit_note"]
    # The doctor warns from the same record.
    assert model_info._CARD_NEEDS[KREA_DEV][0] == DEFAULT_PEAK_GIB


def test_unmeasured_rows_say_so():
    for repo in ("black-forest-labs/FLUX.1-dev", "krea/Krea-2-Raw", "black-forest-labs/FLUX.2-klein-9B",
                 "black-forest-labs/FLUX.2-klein-base-9B"):
        note = model_info.details_for(repo, card_memory=_bytes(RTX_3070))["fit_note"]
        assert note.startswith("FP8 is unmeasured here; "), repo


def test_kontext_keeps_its_qualified_size_after_the_cards_words():
    detail = model_info.details_for(KONTEXT, quantization="fp8", card_memory=_bytes(RTX_3070))
    assert detail["fit"] == "Over 8 GB"
    assert detail["fit_note"].endswith(model_info._FIT[KONTEXT][1])
    assert "Sequential" not in detail["fit_note"]   # held for the editor


@pytest.mark.parametrize("quant", ["fp8", "int8", "none", None])
def test_the_sized_loaders_flag_and_sequential_or_nunchaku_do_not(quant):
    for offload in ("model", "none"):
        assert model_info.details_for(
            KREA_DEV, quantization=quant, offload=offload, card_memory=_bytes(RTX_3070),
        )["fit"] == "Over 8 GB"
    assert model_info.details_for(
        KREA_DEV, quantization=quant, offload="sequential", card_memory=_bytes(RTX_3070),
    )["fit"] == ""
    assert model_info.details_for(
        KREA_DEV, quantization="nunchaku", card_memory=_bytes(RTX_3070),
    )["fit"] == ""


def test_a_card_just_large_enough_is_not_flagged():
    need = model_info._CARD_NEEDS[KREA_DEV][0]
    assert model_info.details_for(KREA_DEV, card_memory=int(need * GIB))["fit"] == ""
    assert model_info.details_for(KREA_DEV, card_memory=int(need * GIB) - 1)["fit"] == "Over 12 GB"


def test_lite_rows_use_the_nvfp4_measurements_and_int4_says_likely():
    lite = {entry.repo: entry for entry in CHECKPOINTS if entry.lite}
    for repo, entry in lite.items():
        nvfp4 = model_info.details_for(entry.lite_repo("nvfp4"), quantization="nunchaku",
                                       card_memory=_bytes(RTX_3070))
        int4 = model_info.details_for(entry.lite_repo("int4"), quantization="nunchaku",
                                      card_memory=_bytes(RTX_3070))
        if repo in model_info._LITE_NEEDS:
            assert nvfp4["fit"] == "Over 8 GB", repo
            assert nvfp4["fit_note"].startswith("The NVFP4 build peaked at ")
            assert int4["fit"] == "Likely over 8 GB", repo
            assert int4["fit_note"].startswith("The INT4 build is unmeasured. The NVFP4 build peaked at ")
            assert "Sequential" not in int4["fit_note"]
        else:
            assert nvfp4["fit"] == int4["fit"] == "", repo
    assert set(model_info._LITE_NEEDS) == {"krea/Krea-2-Turbo", "Qwen/Qwen-Image"}


def test_every_sized_entry_is_in_the_catalogue():
    repos = {entry.repo for entry in CHECKPOINTS}
    assert set(model_info._CARD_NEEDS) <= repos
    assert set(model_info._LITE_NEEDS) <= {entry.repo for entry in CHECKPOINTS if entry.lite}


def test_the_listing_sizes_rows_from_the_servers_card(client, monkeypatch):
    monkeypatch.setattr(card_facts, "card", lambda device="cuda": card_facts.Card(int(RTX_3070), (8, 6)))
    details = client.get("/api/models").json()["details"]
    assert details[KREA_DEV]["fit"] == "Over 8 GB"
    assert details["cagliostrolab/animagine-xl-4.0"]["fit"] == ""
    sequential = client.get("/api/models", params={"offload": "sequential"}).json()["details"]
    assert sequential[KREA_DEV]["fit"] == ""


def test_the_listing_without_a_card_is_unchanged(client):
    details = client.get("/api/models").json()["details"]
    assert details[KREA_DEV]["fit"] == ""
