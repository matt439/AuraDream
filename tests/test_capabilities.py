"""The capability flags in the Models notes and docs/usage.md's table (GEN#56).

No GPU, weights or Hub: `model_info.capabilities_for` asks each flag of the
place that decides it at run time, by the id alone, and these tests hold
every catalogue entry to those places and the document to the function.
`smoke-model-hover` owns the note as the page shows it.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from backend import latent_preview, model_info
from backend.backends import loading
from backend.instruction_edit import is_edit_only
from backend.launch import CHECKPOINTS, FORMATS
from backend.model_defaults import defaults_for

ROOT = Path(__file__).resolve().parent.parent
KEYS = [key for key, _ in model_info.CAPABILITIES]
KREA_DEV = "black-forest-labs/FLUX.1-Krea-dev"


def _generators():
    return [entry for entry in CHECKPOINTS if not is_edit_only(entry.repo)]


def test_the_flags_are_the_seven_the_picker_did_not_say():
    assert model_info.CAPABILITIES == (
        ("inpaint", "Inpaint"), ("outpaint", "Outpaint"), ("references", "References"),
        ("lora", "LoRA"), ("negative_prompt", "Negative prompt"),
        ("preview", "Live preview"), ("expand", "Expand"),
    )


def test_every_catalogue_entry_answers_every_flag_from_its_own_source():
    # A catalogue entry with no latents named is a failure, as one reaching
    # the fallback profile is.
    assert set(model_info._LATENTS) == {entry.repo for entry in CHECKPOINTS}
    for entry in _generators():
        flags = model_info.capabilities_for(entry.repo)
        assert flags is not None and list(flags) == KEYS, entry.repo
        profile = defaults_for(entry.repo)
        want_inpaint = (
            "Yes" if profile["inpaint_supported"]
            else "Not yet" if profile["inpaint_deferred"] else "No"
        )
        # One gate for both (`generate.py`, `outpaint.js`).
        assert flags["inpaint"] == flags["outpaint"] == want_inpaint, entry.repo
        assert flags["references"] == ("Yes" if profile["reference_supported"] else "No"), entry.repo
        assert flags["negative_prompt"] == ("Yes" if profile["negative_prompt_supported"] else "No"), entry.repo
        assert flags["lora"] == ("Yes" if loading.lora_refusal(entry.repo) is None else "No"), entry.repo
        assert flags["expand"] == ("Yes" if loading.prompt_expander_by_id(entry.repo) else "No"), entry.repo
        mapped = model_info._LATENTS[entry.repo] in latent_preview.FACTORS
        assert flags["preview"] == ("Yes" if mapped else "No"), entry.repo


def test_a_deferred_inpaint_is_not_yet_and_never_a_plain_no():
    words = {entry.repo: model_info.capabilities_for(entry.repo)["inpaint"] for entry in CHECKPOINTS}
    assert {words[repo] for repo in words} == {"Yes", "No", "Not yet"}
    assert words["black-forest-labs/FLUX.2-klein-4B"] == "Not yet"
    assert words["Qwen/Qwen-Image"] == "Not yet"
    assert words[KREA_DEV] == "Yes"
    assert words["krea/Krea-2-Turbo"] == "No"


def test_recorded_answers_per_family():
    """What the docs already record, so a table edit that flips one fails
    here by name: FLUX.2's VAE has no preview map and SDXL's has one
    (`GEN#6`'s SDXL half), klein
    and FLUX 3 take references, the app's own pipelines take no LoRA, and
    only ERNIE and FLUX 3 expand."""
    def flag(repo: str, key: str) -> str:
        flags = model_info.capabilities_for(repo)
        assert flags is not None
        return flags[key]

    for repo in ("black-forest-labs/FLUX.2-klein-9B", "baidu/ERNIE-Image-Turbo",
                 "black-forest-labs/flux-3-image-klein"):
        assert flag(repo, "preview") == "No", repo
    for repo in ("cagliostrolab/animagine-xl-4.0", "LyliaEngine/waiIllustriousSDXL_v170",
                 "Laxhar/noobai-XL-Vpred-1.0",
                 KREA_DEV, "krea/Krea-2-Turbo", "Tongyi-MAI/Z-Image-Turbo", "SeeSee21/Z-Anime",
                 "circlestone-labs/Anima", "duongve/NetaYume-Lumina-Image-2.0",
                 "NewBie-AI/NewBie-image-Exp0.1"):
        assert flag(repo, "preview") == "Yes", repo
    takes_references = {entry.repo for entry in CHECKPOINTS if flag(entry.repo, "references") == "Yes"}
    assert takes_references == {
        "black-forest-labs/FLUX.2-klein-4B", "black-forest-labs/FLUX.2-klein-9B",
        "black-forest-labs/FLUX.2-klein-base-4B", "black-forest-labs/FLUX.2-klein-base-9B",
        "black-forest-labs/flux-3-image-klein", "black-forest-labs/flux-3-image-klein-base",
    }
    no_lora = {entry.repo for entry in _generators() if flag(entry.repo, "lora") == "No"}
    assert no_lora == {
        "circlestone-labs/Anima", "circlestone-labs/Anima@aesthetic-v1.1", "Gazingstars123/Anima-2.9B",
        "NewBie-AI/NewBie-image-Exp0.1",
        "black-forest-labs/flux-3-image-klein", "black-forest-labs/flux-3-image-klein-base",
    }
    expands = {entry.repo for entry in CHECKPOINTS if flag(entry.repo, "expand") == "Yes"}
    assert expands == {
        "baidu/ERNIE-Image-Turbo",
        "black-forest-labs/flux-3-image-klein", "black-forest-labs/flux-3-image-klein-base",
    }
    assert flag("black-forest-labs/FLUX.2-klein-4B", "negative_prompt") == "No"


def test_the_editor_answers_no_to_everything_but_its_preview():
    editor = CHECKPOINTS[-1].repo
    assert is_edit_only(editor)
    flags = model_info.capabilities_for(editor)
    assert flags == {
        "inpaint": "No", "outpaint": "No", "references": "No", "lora": "No",
        "negative_prompt": "No", "preview": "Yes", "expand": "No",
    }


@pytest.mark.parametrize("fmt", [value for value, _ in FORMATS])
def test_a_nunchaku_build_differs_only_in_lora(fmt):
    seen = 0
    for entry in CHECKPOINTS:
        if not entry.lite:
            continue
        seen += 1
        native = model_info.capabilities_for(entry.repo)
        lite = model_info.capabilities_for(entry.lite_repo(fmt))
        assert native is not None and lite is not None
        assert lite["lora"] == "No", entry.repo
        assert {**lite, "lora": native["lora"]} == native, entry.repo
        # The official repo under Nunchaku quantization is the same build.
        assert model_info.capabilities_for(entry.repo, "nunchaku") == lite, entry.repo
    assert seen
    for quantization in (None, "fp8", "int8", "none"):
        assert model_info.capabilities_for(KREA_DEV, quantization)["lora"] == "Yes"


def test_the_expand_rule_by_id_is_the_one_the_backend_and_the_fake_ask():
    from tests.fakes import expands

    assert loading.prompt_expander_by_id("baidu/ERNIE-Image-Turbo")
    assert loading.prompt_expander_by_id("baidu/ERNIE_Image")
    assert loading.prompt_expander_by_id("black-forest-labs/flux-3-image-klein")
    assert not loading.prompt_expander_by_id(KREA_DEV)
    for entry in CHECKPOINTS:
        assert expands(entry.repo) == loading.prompt_expander_by_id(entry.repo)
    source = (ROOT / "backend" / "backends" / "diffusers_backend.py").read_text(encoding="utf-8")
    assert "return loading.prompt_expander_by_id(current_model_id(settings))" in source
    assert '"ernie-image" in' not in source


def test_details_for_says_the_flags_as_one_note_and_no_new_cell():
    before = {
        "style", "refine", "refine_note", "megapixels", "megapixels_label", "megapixels_note",
        "favourite_id", "url", "access", "access_note", "fit", "fit_note",
        "instruction_edit", "instruction_edit_note",
    }
    for entry in CHECKPOINTS:
        detail = model_info.details_for(entry.repo)
        assert before < set(detail)
        new = {key for key in detail if key.startswith("capabilit")}
        assert new == {"capabilities_note"}, entry.repo
        flags = model_info.capabilities_for(entry.repo)
        assert flags is not None
        note = detail["capabilities_note"]
        said = "; ".join(f"{label} {flags[key].lower()}" for key, label in model_info.CAPABILITIES)
        assert note == f"{said}. From the checkpoint's id, before a load.", entry.repo
    krea = model_info.details_for(KREA_DEV)["capabilities_note"]
    assert krea.startswith(
        "Inpaint yes; Outpaint yes; References no; LoRA yes; Negative prompt yes; "
        "Live preview yes; Expand no."
    )
    # The loader being chosen reaches the note: the picker's rows rebuild on it.
    assert "LoRA no;" in model_info.details_for(KREA_DEV, quantization="nunchaku")["capabilities_note"]
    assert "Inpaint not yet; Outpaint not yet;" in model_info.details_for("Qwen/Qwen-Image")["capabilities_note"]


def test_a_custom_id_is_claimed_nothing():
    assert model_info.capabilities_for("local/custom-checkpoint") is None
    assert model_info.details_for("local/custom-checkpoint")["capabilities_note"] == ""


def _table_in(text: str) -> str:
    start, end = model_info.CAPABILITY_TABLE_START, model_info.CAPABILITY_TABLE_END
    assert text.count(start) == 1 and text.count(end) == 1
    return text.split(start, 1)[1].split(end, 1)[0].strip("\n")


def test_usage_md_holds_the_generated_table():
    """Fails while docs/usage.md differs from the catalogue:
    `python tools/capability_table.py --write` regenerates it."""
    usage = (ROOT / "docs" / "usage.md").read_text(encoding="utf-8")
    table = model_info.capability_table()
    assert _table_in(usage) == table
    rows = table.splitlines()
    assert rows[0] == (
        "| Checkpoint | Refine | Inpaint | Outpaint | References | LoRA | Negative prompt | "
        "Live preview | Expand |"
    )
    assert len(rows) == len(CHECKPOINTS) + 2
    for entry, row in zip(CHECKPOINTS, rows[2:], strict=True):
        cells = [cell.strip() for cell in row.strip("|").split("|")]
        flags = model_info.capabilities_for(entry.repo)
        assert flags is not None
        assert cells == [entry.name, model_info.details_for(entry.repo)["refine"], *flags.values()]
        assert set(cells[1:]) <= {"Yes", "No", "Not yet"}


def test_the_tool_rewrites_only_between_its_markers():
    spec = importlib.util.spec_from_file_location("capability_table", ROOT / "tools" / "capability_table.py")
    assert spec is not None and spec.loader is not None
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    start, end = model_info.CAPABILITY_TABLE_START, model_info.CAPABILITY_TABLE_END
    stale = f"before\n\n{start}\n| stale |\n{end}\n\nafter\n"
    fresh = tool.replaced(stale)
    assert fresh == f"before\n\n{start}\n{model_info.capability_table()}\n{end}\n\nafter\n"
    assert tool.replaced(fresh) == fresh
    with pytest.raises(SystemExit):
        tool.replaced("no markers here")
    usage = (ROOT / "docs" / "usage.md").read_text(encoding="utf-8")
    assert tool.replaced(usage) == usage


def test_the_page_says_the_note_with_refine_and_adds_no_column():
    source = (ROOT / "web" / "model-switch.js").read_text(encoding="utf-8")
    assert "return [info.refine_note, info.capabilities_note].filter(Boolean).join(' ');" in source
    assert "modelFactCell('Refine', info.refine, '', modelRefineNote(info))" in source
    assert "['Refine', modelRefineNote(info)]" in source
    assert source.count("modelFactCell('") == 4
