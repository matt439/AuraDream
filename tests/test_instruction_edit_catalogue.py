"""INP#11 stage 4 slices 5-6: FLUX.1 Kontext [dev]'s catalogue entry.

Everything the entry carries - the sourced guide, the explicit profile, the
encoder rows, the compact facts, the download reference and the timing -
kept by its exact id (slice 5), and the entry itself, listed last as an
edit-only checkpoint (slice 6). Nothing here loads, downloads or opens the
edit gate.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from backend import guides, launch, model_benchmarks, model_info, prompt_sources
from backend import prompt_encoding as pe
from backend.instruction_edit import (
    INSTRUCTION_EDIT_NEEDS_MODEL,
    INSTRUCTION_EDIT_TEXT_TO_IMAGE,
    INSTRUCTION_EDIT_UNAVAILABLE,
    KONTEXT_V1,
    text_to_image_refusal,
    text_to_image_words,
)
from backend.model_defaults import _FALLBACK, _profile_for, defaults_for, guide_for, hints_for
from tests.prompt_rules_helpers import run

KONTEXT = KONTEXT_V1.model_id


def test_the_editor_is_listed_last_as_an_edit_only_entry():
    # Last, so the terminal menu's numbers for every other entry stay put;
    # no Lite re-upload, so the Nunchaku menu leaves it out.
    entry = launch.CHECKPOINTS[-1]
    assert entry == launch.Checkpoint("FLUX.1 Kontext [dev]", KONTEXT, "click-through gate")
    assert [checkpoint.repo for checkpoint in launch.CHECKPOINTS].count(KONTEXT) == 1
    assert entry.lite is None and entry.lite_repo("nvfp4") is None
    # The menu's summary says what it is, and no CFG it does not have.
    assert entry.summary() == f"edit only, {KONTEXT_V1.default_steps} steps"
    assert "edit only" not in launch.CHECKPOINTS[0].summary() and "CFG" in launch.CHECKPOINTS[0].summary()
    # The menu's summary column (19 wide) still lines its gates up.
    assert max(len(checkpoint.summary()) for checkpoint in launch.CHECKPOINTS) <= 19


# --- the profile ------------------------------------------------------------


def test_the_profile_is_its_own_not_the_fallbacks():
    profile = _profile_for(KONTEXT)
    assert profile is not _FALLBACK
    assert profile["guide"] is guides.KONTEXT_GUIDE
    # Written out, never spread from the fallback whose generation flags it
    # must not inherit; any case reaches the same exact profile.
    assert _profile_for(KONTEXT.upper()) is profile
    defaults = defaults_for(KONTEXT)
    for flag in (
        "text_to_image_supported", "refine_supported", "inpaint_supported",
        "reference_supported", "upscaler_supported", "negative_prompt_supported",
        "true_cfg_supported", "mu_supported", "instruction_edit_supported",
    ):
        assert defaults[flag] is False, flag
    assert defaults["instruction_edit_candidate"] is True
    assert (defaults["steps"], defaults["guidance_scale"]) == (
        KONTEXT_V1.default_steps, KONTEXT_V1.default_guidance_scale,
    )


def test_the_hints_describe_the_recipe_without_calling_it_unavailable():
    hints = hints_for(KONTEXT)
    for key in ("steps", "guidance_scale", "mu"):
        assert "unavailable" not in hints[key].casefold(), key
    assert "28" in hints["steps"] and "2.5" in hints["guidance_scale"]
    assert "no negative prompt" in hints["guidance_scale"]
    assert hints["prompt"] == guides.KONTEXT_GUIDE["prompt"]
    assert "512x512" in hints["size"] and "square" in hints["size"]


def test_the_status_words_follow_the_backends_answer():
    # Cold: the closed gate's words, never a pointer at a held dialog.
    cold = defaults_for(KONTEXT)
    assert cold["instruction_edit_reason"] == INSTRUCTION_EDIT_UNAVAILABLE
    assert cold["text_to_image_reason"] == text_to_image_refusal(KONTEXT, qualified=False)
    assert "Use Edit by instruction" not in cold["text_to_image_reason"]
    # A backend's answer, passed in, words both the same way the status does.
    ready = defaults_for(KONTEXT, edit_reason=None)
    assert ready["instruction_edit_supported"] is True and ready["instruction_edit_reason"] is None
    assert ready["text_to_image_reason"] == INSTRUCTION_EDIT_TEXT_TO_IMAGE
    held = defaults_for(KONTEXT, edit_reason="a measured reason")
    assert held["instruction_edit_supported"] is False
    assert held["text_to_image_reason"].endswith("instruction editing is held: a measured reason")
    assert text_to_image_words(INSTRUCTION_EDIT_NEEDS_MODEL).endswith(INSTRUCTION_EDIT_NEEDS_MODEL)
    # Never the profile's to grant on a checkpoint that is not the editor.
    plain = defaults_for("black-forest-labs/FLUX.1-Krea-dev", edit_reason=None)
    assert plain["instruction_edit_supported"] is False and plain["text_to_image_reason"] is None


# --- the guide ----------------------------------------------------------------


def test_the_guide_passes_the_catalogues_guide_checks():
    # What test_every_catalogue_entry_has_a_prompting_guide holds every
    # listed checkpoint to, the editor's own values named.
    guide = guide_for(KONTEXT)
    assert guide["examples"] and guide["slots"] == []
    assert set(guide["example_notes"]) <= set(guide["examples"])
    low, high = guide["pixels"]
    assert 0 < low <= high and guide["pixels_word"] == "example"
    assert guide["pixels_source"] == "the authors' edit examples"
    hints = hints_for(KONTEXT)
    assert "Unrecognised" not in hints["prompt"] and "Unrecognised" not in hints["size"]
    assert guides.KONTEXT_GUIDE["analysis"]["url"].startswith("https://docs.bfl.ai/kontext/")


# BFL's Kontext pages, as fetched on 2026-10-29: the edit instructions its
# captions and code spans show, verbatim.
_BFL_INSTRUCTIONS = {
    "Change the car color to red",
    "Remove the object from her face",
    "She is now taking a selfie in the streets of Freiburg, it's a lovely day out.",
    "It's now snowing, everything is covered in snow.",
    "Replace 'joy' with 'BFL'",
}


def test_the_examples_are_bfls_own_edit_instructions():
    assert set(guides.KONTEXT_GUIDE["examples"]) == _BFL_INSTRUCTIONS


@pytest.mark.parametrize("example", guides.KONTEXT_GUIDE["examples"])
def test_bfls_instructions_draw_no_finding(example):
    rules = sorted({finding.rule for finding in run(KONTEXT, example)})
    assert rules == [], (example, rules)


def test_an_unquoted_text_edit_draws_the_quote_rule():
    findings = run(KONTEXT, "Change the sign so it says hello world")
    assert "guide.quote_text" in {finding.rule for finding in findings}


def test_the_critic_cites_bfls_pages_and_never_the_measured_drift():
    sources = prompt_sources.sources_for(KONTEXT)
    assert sources.guide_name == "KONTEXT_GUIDE"
    texts = [sentence.text for sentence in sources.sentences]
    assert guides._KONTEXT_DRIFT not in texts
    assert all(sentence.url.startswith("https://docs.bfl.ai/kontext/") for sentence in sources.sentences)
    assert {excerpt.url for excerpt in sources.excerpts} <= {
        "https://docs.bfl.ai/kontext/kontext_overview",
        "https://docs.bfl.ai/kontext/kontext_image_editing",
    }
    assert all("[pro]" in excerpt.note for excerpt in sources.excerpts)
    assert "does not generate from text" in sources.caveat


# --- the encoders ---------------------------------------------------------------


def test_the_encoders_are_flux_kontexts_at_the_pinned_revision():
    specs = pe.encoders_for(KONTEXT)
    assert [spec.name for spec in specs] == ["CLIP-L", "T5-XXL"]
    for spec in specs:
        assert (spec.source.repo, spec.source.revision) == (KONTEXT, KONTEXT_V1.revision)
        assert spec.file.endswith("pipeline_flux_kontext.py")
    assert specs[0].pooled_only and specs[0].max_length is None
    assert specs[1].max_length == pe.FLUX_T5_SEQUENCE
    # FLUX.1's other checkpoints keep FluxPipeline's rows.
    assert pe.encoders_for("black-forest-labs/FLUX.1-dev")[0].file.endswith("pipeline_flux.py")


# --- the compact facts, as a catalogue row will show them -------------------------


def test_details_for_gives_the_editors_strings(monkeypatch):
    # The measured sample is a Windows one; Linux would see only a reference.
    monkeypatch.setattr(model_benchmarks.sys, "platform", "win32")
    detail = model_info.details_for(KONTEXT, "fp8", "model")
    assert detail["style"] == "Instruction editor"
    assert (detail["megapixels"], detail["megapixels_label"]) == ("0.26 MP", "Qualified size")
    assert "512×512" in detail["megapixels_note"]
    assert "Direct generation" not in detail["megapixels_note"]
    assert "size follows its source" in detail["megapixels_note"]
    assert detail["refine"] == "No"
    assert detail["access"] == "Accept licence"
    assert detail["fit"] == "512×512 only"
    assert "14.5 GB at 512×512, fp8, model offload (2026-10-27)" in detail["fit_note"]
    assert detail["url"] == f"https://huggingface.co/{KONTEXT}"
    assert detail["speed_label"] == "Medium" and detail["speed_basis"] == "Measured"
    assert "512×512 edit" in detail["speed_summary"]
    assert "29.3 s" in detail["speed_note"] and "2026-10-27" in detail["speed_note"]
    # Other setups are references only.
    assert model_info.details_for(KONTEXT, "int8", "model")["speed_label"] == "Reference"
    assert model_info.details_for(KONTEXT, "fp8", "sequential")["speed_label"] == "Unmeasured"


def test_the_editors_row_says_the_live_answer():
    cold = model_info.details_for(KONTEXT)
    assert (cold["instruction_edit"], cold["instruction_edit_note"]) == ("Held", INSTRUCTION_EDIT_UNAVAILABLE)
    held = model_info.details_for(KONTEXT, edit_reason="sequential offload is not measured")
    assert (held["instruction_edit"], held["instruction_edit_note"]) == (
        "Held", "sequential offload is not measured",
    )
    ready = model_info.details_for(KONTEXT, edit_reason=None)
    assert ready["instruction_edit"] == "Yes" and "Edit by instruction" in ready["instruction_edit_note"]
    # A case variant is never the editor's row: held, whatever is passed.
    variant = model_info.details_for(KONTEXT.lower(), edit_reason=None)
    assert variant["instruction_edit"] == "Held"
    # Every other row says where editing runs, whatever the gate answers.
    other = model_info.details_for("black-forest-labs/FLUX.1-Krea-dev", edit_reason=None)
    assert other["instruction_edit"] == "No" and "FLUX.1 Kontext [dev]" in other["instruction_edit_note"]


def test_the_timing_is_the_trials_and_keyed_by_the_exact_id():
    [sample] = model_benchmarks._SAMPLES[KONTEXT]
    assert sample.steps == KONTEXT_V1.default_steps
    assert (sample.quantization, sample.offload, sample.platform) == ("fp8", "model", "win32")
    assert sample.size.startswith("512×512") and sample.warm


# --- the download reference -----------------------------------------------------


def test_the_download_reference_is_the_pinned_component_set():
    data = json.loads(Path(model_info.__file__).with_name("model-downloads.json").read_text(encoding="utf-8"))
    recipe = data["models"][KONTEXT]
    assert recipe["kind"] == "pipeline"
    [selected] = recipe["sources"]
    source = data["sources"][selected["source"]]
    assert source["commit"] == source["revision"] == KONTEXT_V1.revision
    assert selected["source"] == f"{KONTEXT}@{KONTEXT_V1.revision}"
    assert source["repo_id"] == KONTEXT
    # The components the measured download left, never the root single file.
    assert all("/" in name or name == "model_index.json" for name in selected["files"])
    assert not any(re.search(r"flux1-kontext-dev\.safetensors$", name) for name in source["files"])
    assert source["indexes"]["model_index.json"]["_class_name"] == KONTEXT_V1.pipeline_class
    for name in selected["files"]:
        assert source["files"][name] > 0
        if name.endswith(".safetensors.index.json"):
            shards = source["indexes"][name]["weight_map"].values()
            folder = name.rsplit("/", 1)[0]
            assert {f"{folder}/{shard}" for shard in shards} <= set(selected["files"])
    facts = model_info.download_facts(KONTEXT)
    assert facts["download"] == "~33.7 GB"
