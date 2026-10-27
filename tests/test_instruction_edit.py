"""Dormant edit identity and pixels, held to installed source without models."""
from __future__ import annotations

import ast
import dataclasses
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
from PIL import Image, PngImagePlugin
from pydantic import ValidationError

from backend import instruction_edit as edit


def _proof(**changes):
    return {
        "filename": "source.png", "hash": "ab" * 32, "width": 1024, "height": 1024,
        **changes,
    }


def _stored(**changes):
    return {
        "source": 17, "recipe": "kontext-v1", "edit_source": _proof(),
        **changes,
    }


@pytest.mark.parametrize("source", [None, True, False, 0, -1, 1.0, "1", [], {}])
def test_a_request_never_coerces_a_missing_or_noninteger_gallery_id(source):
    with pytest.raises(ValidationError):
        edit.InstructionEditSpec(source=source, recipe="kontext-v1")


@pytest.mark.parametrize("values", [
    {"source": 1},
    {"source": 1, "recipe": "kontext-v2"},
    {"source": 1, "recipe": "Kontext-v1"},
    {"source": 1, "recipe": "kontext-v1", "strength": 0.5},
    {"source": 1, "recipe": "kontext-v1", "path": "source.png"},
])
def test_request_identity_requires_an_explicit_known_recipe_and_no_extra_pass_fields(values):
    with pytest.raises(ValidationError):
        edit.InstructionEditSpec.model_validate(values)


def test_request_recipe_and_nested_source_proof_are_frozen():
    requested = edit.InstructionEditSpec(source=1, recipe="kontext-v1")
    recorded = edit.StoredInstructionEdit.model_validate(_stored())
    before = edit.stored_json(recorded)
    with pytest.raises(ValidationError, match="frozen"):
        requested.source = 2
    with pytest.raises(ValidationError, match="frozen"):
        recorded.edit_source.hash = "cd" * 32
    with pytest.raises(dataclasses.FrozenInstanceError):
        edit.KONTEXT_V1.revision = "main"
    assert edit.stored_json(recorded) == before


def test_complete_stored_identity_has_a_deterministic_json_spelling():
    recorded = edit.StoredInstructionEdit.model_validate(_stored())
    text = edit.stored_json(recorded)
    expected = (
        '{"edit_source":{"filename":"source.png","hash":"' + "ab" * 32
        + '","height":1024,"width":1024},"recipe":"kontext-v1","source":17}'
    )
    assert text == expected
    assert edit.stored_json(json.loads(text)) == text
    assert edit.stored_from_json(text) == recorded
    assert recorded.request() == edit.InstructionEditSpec(source=17, recipe="kontext-v1")


def test_unresolved_source_keeps_its_proof_and_instruction_edit_identity():
    recorded = edit.StoredInstructionEdit.model_validate(_stored(source=None))
    text = edit.stored_json(recorded)
    assert json.loads(text)["source"] is None
    restored = edit.stored_from_json(text)
    assert restored.recipe == "kontext-v1"
    assert restored.edit_source == edit.InstructionEditSource.model_validate(_proof())
    with pytest.raises(ValueError, match="has not been resolved"):
        restored.request()
    with pytest.raises(ValueError, match="Instruction editing is not available"):
        edit.require_execution_available(restored)


@pytest.mark.parametrize("values", [
    {"source": 1, "recipe": "kontext-v1"},
    {"recipe": "kontext-v1", "edit_source": _proof()},
    _stored(source="17"),
    _stored(source=False),
    _stored(edit_source=None),
    _stored(edit_source=_proof(hash="a" * 16)),
    _stored(edit_source=_proof(hash="AB" * 32)),
    _stored(edit_source=_proof(width="1024")),
    _stored(edit_source=_proof(height=True)),
    _stored(edit_source=_proof(width=0)),
    _stored(edit_source=_proof(extra="ignored?")),
    _stored(strength=0.5),
])
def test_portable_records_refuse_missing_proof_coercion_and_silently_ignored_fields(values):
    with pytest.raises(ValidationError):
        edit.stored_json(values)
    with pytest.raises(ValidationError):
        edit.stored_from_json(json.dumps(values))


@pytest.mark.parametrize("filename", [
    "../source.png", "folder/source.png", "folder\\source.png", "C:source.png", "C:\\source.png",
    "source.png:stream", "source.png ", "source.jpg", ".png", "a\x00.png", "a\n.png",
    "con.png", "LPT1.source.png", "COM².png", "a?.png", "a|b.png",
])
def test_provenance_filename_is_a_bare_png_name_not_a_windows_or_posix_path(filename):
    with pytest.raises(ValidationError):
        edit.InstructionEditSource.model_validate(_proof(filename=filename))


@pytest.mark.parametrize("filename", ["source.png", "2026-10-27_17.source.PNG", "a b.png", "夜.png", ".source.png"])
def test_provenance_accepts_existing_bare_png_names(filename):
    assert edit.InstructionEditSource.model_validate(_proof(filename=filename)).filename == filename


@pytest.mark.parametrize("filename", [".png", "a" * 252 + ".png"])
def test_invalid_filename_length_refuses_before_source_pixel_work(filename, monkeypatch):
    image = Image.new("RGB", (512, 512))

    def forbidden(*args, **kwargs):
        pytest.fail("an invalid source filename must refuse before decoding or hashing pixels")

    monkeypatch.setattr(Image.Image, "convert", forbidden)
    with pytest.raises(ValueError, match="bare PNG basename"):
        edit.descriptor_from_image(filename, image)


@pytest.mark.parametrize("changes", [{"source": True}, {"recipe": "future"}, {"surprise": "discarded?"}])
def test_an_unvalidated_model_copy_cannot_cross_the_request_boundary(changes):
    forged = edit.InstructionEditSpec(source=1, recipe="kontext-v1").model_copy(update=changes)
    with pytest.raises(ValidationError):
        edit.require_execution_available(forged)


def test_unvalidated_copies_cannot_smuggle_unknown_nested_or_stored_fields():
    recorded = edit.StoredInstructionEdit.model_validate(_stored())
    for forged in (
        recorded.model_copy(update={"source": True}),
        recorded.model_copy(update={"extra": "discarded?"}),
        recorded.model_copy(update={"edit_source": recorded.edit_source.model_copy(update={"hash": "a" * 16})}),
        recorded.model_copy(update={"edit_source": recorded.edit_source.model_copy(update={"extra": "discarded?"})}),
    ):
        with pytest.raises(ValidationError):
            edit.stored_json(forged)
        with pytest.raises(ValidationError):
            forged.request()


def test_proof_hashes_full_original_decoded_rgb_pixels_before_resize():
    image = Image.new("RGBA", (3, 2))
    image.putdata([(1, 2, 3, 0), (4, 5, 6, 255), (7, 8, 9, 127)] * 2)
    before = image.tobytes()
    proof = edit.descriptor_from_image("original.png", image)
    assert proof.hash == hashlib.sha256(bytes(range(1, 10)) * 2).hexdigest()
    assert (proof.width, proof.height) == (3, 2)
    prepared = edit.prepare_source(image)
    assert prepared.size == (576, 384)
    assert proof.hash != hashlib.sha256(prepared.tobytes()).hexdigest()
    assert image.mode == "RGBA" and image.tobytes() == before


def test_png_encoding_and_metadata_do_not_change_decoded_source_proof():
    image = Image.new("RGB", (4, 3), (10, 20, 30))
    streams = []
    for compression, note in ((0, "first"), (9, "a different metadata value")):
        stream = io.BytesIO()
        info = PngImagePlugin.PngInfo()
        info.add_text("Comment", note)
        image.save(stream, format="PNG", pnginfo=info, compress_level=compression)
        streams.append(stream.getvalue())
    assert streams[0] != streams[1]
    proofs = []
    for data in streams:
        with Image.open(io.BytesIO(data)) as decoded:
            proofs.append(edit.descriptor_from_image("source.png", decoded))
    assert proofs[0] == proofs[1]


@pytest.mark.parametrize("source_size, expected", [
    ((1024, 1024), (512, 512)),
    ((1200, 800), (576, 384)),
    ((800, 1200), (384, 576)),
    ((1200, 900), (512, 384)),
    ((900, 1200), (384, 512)),
    ((1920, 1080), (512, 288)),
    ((1080, 1920), (288, 512)),
])
def test_supported_source_ratios_have_exact_small_working_and_output_geometry(source_size, expected):
    resolved = edit.geometry(*source_size)
    assert resolved.source_size == source_size
    assert resolved.working_size == resolved.output_size == expected
    assert resolved.width * resolved.height <= 262144
    assert resolved.width * source_size[1] == resolved.height * source_size[0]
    kwargs = resolved.pipeline_kwargs()
    assert kwargs == {
        "width": expected[0], "height": expected[1], "max_area": expected[0] * expected[1],
        "_auto_resize": False, "num_images_per_prompt": 1, "max_sequence_length": 512,
    }
    kwargs["width"] = 2048
    assert resolved.pipeline_kwargs()["width"] == expected[0]
    with pytest.raises(dataclasses.FrozenInstanceError):
        resolved.width = 2048


@pytest.mark.parametrize("source_size", [
    (0, 1024), (-1, 1), (True, 1), (1024.0, 1024), ("1024", 1024),
    (1024, 1023), (1360, 768), (848, 1264), (1, 3), (1, 10000),
])
def test_geometry_refuses_approximate_ratios_and_noninteger_sizes_without_hidden_fitting(source_size):
    with pytest.raises(ValueError):
        edit.geometry(*source_size)


def test_unknown_recipe_and_invented_geometry_cannot_borrow_the_kontext_mapping():
    with pytest.raises(ValueError, match="unsupported"):
        edit.geometry(1024, 1024, recipe="kontext-v2")
    with pytest.raises(ValueError, match="frozen"):
        edit.InstructionEditGeometry(1024, 1024, 1024, 1024)


def test_source_preparation_keeps_every_edge_and_does_not_apply_exif_orientation():
    image = Image.new("RGB", (1024, 768), (0, 0, 0))
    image.paste((255, 0, 0), (0, 0, 512, 384))
    image.paste((0, 255, 0), (512, 0, 1024, 384))
    image.paste((0, 0, 255), (0, 384, 512, 768))
    image.paste((255, 255, 0), (512, 384, 1024, 768))
    image.getexif()[274] = 6  # Would rotate to portrait if transposed.
    before = image.tobytes()
    prepared = edit.prepare_source(image, edit.geometry(*image.size))
    assert prepared.mode == "RGB" and prepared.size == (512, 384)
    assert [prepared.getpixel(point) for point in ((0, 0), (511, 0), (0, 383), (511, 383))] == [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    ]
    # Lanczos filters the boundary; a nearest-neighbour/hidden crop replacement differs.
    assert prepared.getpixel((255, 50)) == (241, 14, 0)
    assert image.size == (1024, 768) and image.tobytes() == before


def test_same_size_rgb_preparation_returns_an_independent_owned_image():
    image = Image.new("RGB", (512, 512), (13, 17, 19))
    prepared = edit.prepare_source(image)
    assert prepared is not image and prepared.tobytes() == image.tobytes()
    image.close()
    assert prepared.getpixel((0, 0)) == (13, 17, 19)
    prepared.putpixel((0, 0), (99, 99, 99))
    assert prepared.getpixel((1, 0)) == (13, 17, 19)


def test_preparation_refuses_a_stale_original_size_even_if_its_aspect_matches(monkeypatch):
    image = Image.new("RGB", (600, 400))

    def forbidden(*args, **kwargs):
        pytest.fail("a stale source must be refused before resizing")

    monkeypatch.setattr(Image.Image, "resize", forbidden)
    with pytest.raises(ValueError, match="dimensions changed"):
        edit.prepare_source(image, edit.geometry(1200, 800))


def test_named_recipe_is_pinned_but_every_execution_remains_unavailable():
    recipe = edit.recipe_for("kontext-v1")
    assert recipe.model_id == "black-forest-labs/FLUX.1-Kontext-dev"
    assert recipe.revision == "24e9dedc4ef646698dc8eb4e18ae2cec3c9fea0d"
    assert recipe.pipeline_class == "FluxKontextPipeline"
    assert (recipe.default_steps, recipe.default_guidance_scale) == (28, 2.5)
    assert (recipe.negative_prompt, recipe.true_cfg, recipe.mu) == ("", None, None)
    with pytest.raises(ValueError, match="Instruction editing is not available"):
        edit.require_execution_available(edit.InstructionEditSpec(source=1, recipe="kontext-v1"))


def _installed_tree(relative):
    distribution = importlib.metadata.distribution("diffusers")
    return ast.parse(Path(distribution.locate_file("diffusers") / relative).read_text(encoding="utf-8"))


def _method(tree, class_name, method_name):
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    return next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == method_name)


def test_frozen_kwargs_match_the_installed_kontext_signature_without_importing_it():
    tree = _installed_tree("pipelines/flux/pipeline_flux_kontext.py")
    call = _method(tree, "FluxKontextPipeline", "__call__")
    names = [argument.arg for argument in call.args.args]
    defaults = dict(zip(names[-len(call.args.defaults):], call.args.defaults))
    assert {"image", "prompt", "generator", "callback_on_step_end"} <= set(names)
    assert not {"strength", "mask_image"} & set(names)
    assert ast.literal_eval(defaults["_auto_resize"]) is True
    assert ast.literal_eval(defaults["max_sequence_length"]) == 512
    assert ast.literal_eval(defaults["num_images_per_prompt"]) == 1
    for key in edit.geometry(1024, 1024).pipeline_kwargs():
        assert key in names
    # The guarded source path must still use preferred sizes only under this flag.
    resizing = next(node for node in ast.walk(call) if isinstance(node, ast.If)
                    and isinstance(node.test, ast.Name) and node.test.id == "_auto_resize")
    assert any(isinstance(node, ast.Name) and node.id == "PREFERRED_KONTEXT_RESOLUTIONS"
               for node in ast.walk(resizing))


def test_explicit_area_prevents_the_installed_output_formula_from_expanding_a_small_edit():
    call = _method(_installed_tree("pipelines/flux/pipeline_flux_kontext.py"), "FluxKontextPipeline", "__call__")
    targets = {"height", "width", "original_height", "original_width", "aspect_ratio", "multiple_of"}
    assignments = [node for node in call.body if isinstance(node, ast.Assign)
                   and {item.id for target in node.targets for item in ast.walk(target) if isinstance(item, ast.Name)}
                   <= targets]
    # Execute only the installed scalar size formula, with no imports or model call.
    assert len(assignments) == 9
    code = compile(ast.Module(body=assignments, type_ignores=[]), "installed-kontext-size-formula", "exec")
    for source in ((1024, 1024), (1200, 800), (1920, 1080)):
        resolved = edit.geometry(*source)
        kwargs = resolved.pipeline_kwargs()
        environment = {**kwargs, "self": SimpleNamespace(vae_scale_factor=8, default_sample_size=128)}
        exec(code, {"__builtins__": {}, "round": round}, environment)
        assert (environment["width"], environment["height"]) == resolved.output_size
    default_area = {
        "width": 512, "height": 512, "max_area": 1024**2,
        "self": SimpleNamespace(vae_scale_factor=8, default_sample_size=128),
    }
    exec(code, {"__builtins__": {}, "round": round}, default_area)
    assert (default_area["width"], default_area["height"]) == (1024, 1024)


def test_installed_source_vae_uses_the_mode_instead_of_sampling_with_another_generator():
    encode = _method(_installed_tree("pipelines/flux/pipeline_flux_kontext.py"),
                     "FluxKontextPipeline", "_encode_vae_image")
    retrievals = [node for node in ast.walk(encode) if isinstance(node, ast.Call)
                  and isinstance(node.func, ast.Name) and node.func.id == "retrieve_latents"]
    assert len(retrievals) == 2
    assert all(any(keyword.arg == "sample_mode" and ast.literal_eval(keyword.value) == "argmax"
                   for keyword in call.keywords) for call in retrievals)


def test_contract_module_import_and_pixel_operations_are_cold_in_a_fresh_process():
    root = str(Path(__file__).resolve().parents[1])
    script = """
import importlib.abc
import sys
sys.path.insert(0, ROOT)
class RefuseModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'diffusers', 'huggingface_hub'} or fullname in {
            'backend.config', 'backend.storage', 'backend.main',
        }:
            raise AssertionError('unexpected warm import: ' + fullname)
sys.meta_path.insert(0, RefuseModels())
from backend import instruction_edit as edit
from PIL import Image
source = Image.new('RGB', (512, 512))
proof = edit.descriptor_from_image('source.png', source)
prepared = edit.prepare_source(source)
assert proof.width == prepared.width == 512
assert not {'torch', 'diffusers', 'huggingface_hub'} & set(sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", "ROOT = " + repr(root) + "\n" + script],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
