"""Cold picker summaries preserve model identity and source distinctions."""
from types import SimpleNamespace

import pytest

from backend import launch, model_info, model_selection
from backend.launch import CHECKPOINTS, FORMATS
from backend.model_defaults import defaults_for


def test_every_catalogue_selection_has_explicit_compact_facts():
    from backend.instruction_edit import KONTEXT_V1, is_edit_only

    assert KONTEXT_V1.model_id in {entry.repo for entry in CHECKPOINTS}
    assert set(model_info._FACTS) == {entry.repo for entry in CHECKPOINTS}
    for entry in CHECKPOINTS:
        detail = model_info.details_for(entry.repo)
        assert 1 <= len(detail["style"].split()) <= 2
        assert len(detail["megapixels"]) <= 12
        assert detail["megapixels"].endswith(" MP")
        assert float(detail["megapixels"].removesuffix(" MP")) > 0
        assert detail["refine"] == ("Yes" if defaults_for(entry.repo)["refine_supported"] else "No")
        assert detail["refine_note"]
        assert detail["favourite_id"] == entry.repo
        # An editor never generates directly: its size is the one measured.
        if is_edit_only(entry.repo):
            assert "Direct generation" not in detail["megapixels_note"]
            assert detail["megapixels_label"] == "Qualified size"
            continue
        assert "Direct generation, not Refine" in detail["megapixels_note"]
        assert detail["megapixels_label"] in {
            "Max recommended", "Documented max", "Trained max", "Native max",
            "Reference size", "Largest example",
        }


@pytest.mark.parametrize("fmt", [value for value, _ in FORMATS])
def test_lite_variants_share_canonical_facts_and_favourites(fmt):
    for entry in CHECKPOINTS:
        if entry.lite:
            lite = model_info.details_for(entry.lite_repo(fmt))
            native = model_info.details_for(entry.repo)
            def shared(detail):
                return {
                    key: value for key, value in detail.items()
                    if not key.startswith(("speed_", "access", "fit", "download", "cache"))
                }
            assert shared(lite) == shared(native)
            assert lite["access"] == "Open"
            assert "lite-infer" in lite["access_note"]
            assert lite["fit"] == lite["fit_note"] == ""


def test_qwen_image_says_its_torchao_builds_are_larger_than_the_card():
    # Only the 4-bit re-upload fits 16 GB (launch.CHECKPOINTS, verified.md);
    # the official repo's row, which FP8/INT8/BF16 load, must say so.
    official = model_info.details_for("Qwen/Qwen-Image", quantization="fp8")
    assert official["fit"] == "Over 16 GB"
    assert "Nunchaku" in official["fit_note"] and "20B" in official["fit_note"]
    lite = next(entry for entry in CHECKPOINTS if entry.repo == "Qwen/Qwen-Image").lite_repo("nvfp4")
    assert model_info.details_for(lite, quantization="nunchaku")["fit"] == ""
    assert set(model_info._FIT) <= set(model_info._FACTS)
    for entry in CHECKPOINTS:
        if entry.repo not in model_info._FIT:
            assert model_info.details_for(entry.repo)["fit"] == "", entry.repo
    assert model_info.details_for("local/custom-checkpoint")["fit"] == ""


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


def test_anima_preview_keeps_its_own_recommendations_and_measured_sizes():
    preview = model_info.details_for("Gazingstars123/Anima-2.9B")
    assert preview["favourite_id"] == "Gazingstars123/Anima-2.9B"
    assert preview["url"] == "https://huggingface.co/Gazingstars123/Anima-2.9B"
    assert preview["megapixels_label"] == "Max recommended"
    assert preview["megapixels"] == "1.77 MP"
    assert "1536-square" in preview["megapixels_note"] and "iffy" in preview["megapixels_note"]
    assert "1024×1024 and 1152×1536" in preview["megapixels_note"]
    assert "other sizes and loading settings remain unmeasured" in preview["megapixels_note"]
    assert preview["refine"] == "No"


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
    # Both are one byte a weight; INT8 was compared on the card on sixteen
    # checkpoints, not FLUX.1, Krea 2, klein 9B or Qwen-Image, and the line says so.
    assert "same size as FP8" in descriptions["quantizations"]["int8"]
    assert "where compared (not FLUX.1, Krea 2, klein 9B or Qwen-Image)" in descriptions["quantizations"]["int8"]
    assert "softer" not in descriptions["quantizations"]["int8"]
    assert "24 GB+" in descriptions["offloads"]["none"]
    # The terminal menu prints the same words.
    assert dict(launch.QUANTIZATIONS) == descriptions["quantizations"]
    assert dict(launch.OFFLOADS) == descriptions["offloads"]
    monkeypatch.setattr(model_selection.sys, "platform", "win32")
    assert "nunchaku" not in {
        entry["value"] for entry in model_selection.options(SimpleNamespace())["quantizations"]
    }


def test_instruction_edit_facts_on_catalogue_editor_and_other_custom_rows():
    from backend.instruction_edit import INSTRUCTION_EDIT_UNAVAILABLE, KONTEXT_V1
    from backend.launch import CHECKPOINTS

    # Listed last, and the one row that can say anything but No.
    assert CHECKPOINTS[-1].repo == KONTEXT_V1.model_id
    for entry in CHECKPOINTS[:-1]:
        detail = model_info.details_for(entry.repo)
        assert detail["instruction_edit"] == "No"
        assert "FLUX.1 Kontext [dev]" in detail["instruction_edit_note"]
        assert "its row says whether it can run" in detail["instruction_edit_note"]
    editor = model_info.details_for(KONTEXT_V1.model_id)
    assert editor["instruction_edit"] == "Held"
    assert editor["instruction_edit_note"] == INSTRUCTION_EDIT_UNAVAILABLE
    custom = model_info.details_for("local/custom-checkpoint")
    assert custom["instruction_edit"] == "No"
    assert custom["instruction_edit_note"] == ""


def test_the_editor_profile_gives_generate_its_held_words():
    from backend.instruction_edit import INSTRUCTION_EDIT_UNAVAILABLE, KONTEXT_V1
    from backend.model_defaults import defaults_for

    editor = defaults_for(KONTEXT_V1.model_id)
    assert editor["text_to_image_supported"] is False
    # The closed gate's words until a backend answers: never a pointer at
    # Edit by instruction…, which is itself held.
    assert editor["text_to_image_reason"].endswith(INSTRUCTION_EDIT_UNAVAILABLE)
    assert "Use Edit by instruction" not in editor["text_to_image_reason"]
    plain = defaults_for("black-forest-labs/FLUX.1-Krea-dev")
    assert plain["text_to_image_supported"] is True
    assert plain["text_to_image_reason"] is None
