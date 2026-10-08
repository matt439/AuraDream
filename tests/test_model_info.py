"""Cold picker summaries preserve model identity and source distinctions."""
from types import SimpleNamespace

import pytest

from backend import launch, model_info, model_selection
from backend.launch import CHECKPOINTS, FORMATS
from backend.model_defaults import defaults_for


def test_every_catalogue_selection_has_explicit_compact_facts():
    assert set(model_info._FACTS) == {entry.repo for entry in CHECKPOINTS}
    for entry in CHECKPOINTS:
        detail = model_info.details_for(entry.repo)
        assert 1 <= len(detail["style"].split()) <= 2
        assert len(detail["megapixels"]) <= 12
        assert detail["megapixels"].endswith(" MP")
        assert float(detail["megapixels"].removesuffix(" MP")) > 0
        assert detail["refine"] == ("Yes" if defaults_for(entry.repo)["refine_supported"] else "No")
        assert detail["refine_note"]
        assert "Direct generation, not Refine" in detail["megapixels_note"]
        assert detail["megapixels_label"] in {
            "Max recommended", "Documented max", "Trained max", "Native max",
            "Reference size", "Largest example",
        }
        assert detail["favourite_id"] == entry.repo


@pytest.mark.parametrize("fmt", [value for value, _ in FORMATS])
def test_lite_variants_share_canonical_facts_and_favourites(fmt):
    for entry in CHECKPOINTS:
        if entry.lite:
            lite = model_info.details_for(entry.lite_repo(fmt))
            native = model_info.details_for(entry.repo)
            def shared(detail):
                return {
                    key: value for key, value in detail.items()
                    if not key.startswith(("speed_", "access"))
                }
            assert shared(lite) == shared(native)
            assert lite["access"] == "Open"
            assert "lite-infer" in lite["access_note"]


def test_every_catalogue_entry_reports_its_hugging_face_access():
    words = {"click-through": "Accept licence", "manual": "Needs approval"}
    for entry in CHECKPOINTS:
        detail = model_info.details_for(entry.repo)
        want = next((label for prefix, label in words.items() if entry.gate.startswith(prefix)), "Open")
        assert detail["access"] == want, entry.repo
        assert detail["access_note"]
    assert model_info.details_for("krea/Krea-2-Turbo")["access"] == "Needs approval"
    assert "can take days" in model_info.details_for("krea/Krea-2-Raw")["access_note"]
    assert model_info.details_for("black-forest-labs/FLUX.1-Krea-dev")["access"] == "Accept licence"
    assert model_info.details_for("cagliostrolab/animagine-xl-4.0")["access"] == "Open"
    custom = model_info.details_for("local/custom-checkpoint")
    assert custom["access"] == "Unknown" and custom["access_note"]


def test_anima_releases_keep_separate_favourites_but_link_to_the_real_repo():
    turbo = model_info.details_for("circlestone-labs/Anima")
    aesthetic = model_info.details_for("circlestone-labs/Anima@aesthetic-v1.1")
    assert turbo["favourite_id"] != aesthetic["favourite_id"]
    assert turbo["url"] == aesthetic["url"] == "https://huggingface.co/circlestone-labs/Anima"
    assert turbo["refine"] == aesthetic["refine"] == "No"
    assert "Only a 1.05 MP square" in aesthetic["megapixels_note"]


@pytest.mark.parametrize("model_id", [
    "LyliaEngine/waiIllustriousSDXL_v170",
    "duongve/NetaYume-Lumina-Image-2.0",
    "NewBie-AI/NewBie-image-Exp0.1",
])
def test_example_sizes_never_claim_a_supported_maximum(model_id):
    detail = model_info.details_for(model_id)
    note = detail["megapixels_note"]
    assert "example" in note.lower()
    assert "not a published" in note or "no v4 operating range" in note
    assert detail["megapixels_label"] in {"Reference size", "Largest example"}


def test_custom_selection_has_no_invented_capabilities_or_size_range():
    detail = model_info.details_for("local/custom-checkpoint")
    assert detail["refine"] == "After load"
    assert detail["megapixels"] == "Unknown"
    assert detail["style"] == "Custom"
    assert detail["favourite_id"] == "local/custom-checkpoint"
    assert detail["url"] == ""


def test_size_summaries_distinguish_recommended_documented_and_example_upper_values():
    turbo = model_info.details_for("krea/Krea-2-Turbo")
    wai = model_info.details_for("LyliaEngine/waiIllustriousSDXL_v170")
    flux = model_info.details_for("black-forest-labs/FLUX.1-Krea-dev")
    z_anime = model_info.details_for("SeeSee21/Z-Anime")
    assert (flux["megapixels_label"], flux["megapixels"]) == ("Max recommended", "1.64 MP")
    assert (z_anime["megapixels_label"], z_anime["megapixels"]) == ("Max recommended", "2 MP")
    assert (turbo["megapixels_label"], turbo["megapixels"]) == ("Documented max", "4 MP")
    assert (wai["megapixels_label"], wai["megapixels"]) == ("Reference size", "1.38 MP")
    assert "4.19 MP" in turbo["megapixels_note"]
    assert "1.05–1.38 MP" in wai["megapixels_note"]


@pytest.mark.parametrize(("model_id", "label", "upper_mp"), [
    ("OnomaAIResearch/Illustrious-XL-v2.0", "Native max", "2.25 MP"),
    ("circlestone-labs/Anima", "Documented max", "2.36 MP"),
    ("Qwen/Qwen-Image", "Reference size", "1.76 MP"),
    ("duongve/NetaYume-Lumina-Image-2.0", "Largest example", "1.57 MP"),
    ("NewBie-AI/NewBie-image-Exp0.1", "Largest example", "1.05 MP"),
])
def test_source_upper_bounds_are_not_rounded_down_to_an_app_preset(model_id, label, upper_mp):
    detail = model_info.details_for(model_id)
    assert (detail["megapixels_label"], detail["megapixels"]) == (label, upper_mp)


def test_browser_loader_options_are_short_and_keep_their_explanations(monkeypatch):
    monkeypatch.setattr(model_selection.sys, "platform", "linux")
    options = model_selection.options(SimpleNamespace())
    assert {entry["value"]: entry["label"] for entry in options["quantizations"]} == {
        "fp8": "FP8", "int8": "INT8", "none": "BF16", "nunchaku": "Nunchaku 4-bit",
    }
    assert {entry["value"]: entry["label"] for entry in options["offloads"]} == {
        "model": "Model offload", "sequential": "Sequential", "none": "No offload",
    }
    assert [entry["label"] for entry in options["formats"]] == ["NVFP4", "INT4"]
    assert all(entry["description"] for choices in options.values() for entry in choices)
    descriptions = {
        key: {entry["value"]: entry["description"] for entry in options[key]}
        for key in ("quantizations", "offloads")
    }
    assert "recommended for a 16 GB card" in descriptions["quantizations"]["fp8"]
    assert "recommended for a 16 GB card" in descriptions["offloads"]["model"]
    assert "may not fit 16 GB" in descriptions["quantizations"]["none"]
    assert "24 GB+" in descriptions["offloads"]["none"]
    # The terminal menu prints the same words.
    assert dict(launch.QUANTIZATIONS) == descriptions["quantizations"]
    assert dict(launch.OFFLOADS) == descriptions["offloads"]
    monkeypatch.setattr(model_selection.sys, "platform", "win32")
    assert "nunchaku" not in {
        entry["value"] for entry in model_selection.options(SimpleNamespace())["quantizations"]
    }
