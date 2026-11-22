import json
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import pytest
import torch
from PIL import Image

from backend import bundles, reindex
from backend.backends.base import Cancelled
from backend.backends.regional_prompts import prepared
from backend.bundle_common import BundleError
from backend.schemas import GenerateRequest
from backend.spatial_conditioning import PromptRegion, RegionalPrompts
from backend.tiled_refine_contract import MODEL_ID, QUANTIZATION, ROOT_SIZE
from tests.bundle_helpers import png_metadata, use_destination


def test_regional_processors_preserve_global_and_average_overlaps():
    from diffusers.models.attention_processor import Attention

    rand = torch.Generator().manual_seed(42)
    attention = Attention(query_dim=8, cross_attention_dim=8, heads=2, dim_head=4)
    original = attention.processor
    self_processor = object()
    cross = "blocks.0.attn2.processor"
    own = "blocks.0.attn1.processor"
    left = torch.randn((1, 2, 8), generator=rand)
    right = torch.randn((1, 2, 8), generator=rand)
    unet = SimpleNamespace(attn_processors={cross: original, own: self_processor})

    def set_processors(processors):
        attention.set_processor(processors[cross])
        unet.attn_processors = dict(processors)
        processors.clear()  # Diffusers consumes the supplied dictionary.

    unet.set_attn_processor = set_processors
    class Tokenizer:
        model_max_length = 77

        def __call__(self, prompt, **kwargs):
            return SimpleNamespace(input_ids=list(prompt))

    pipe = SimpleNamespace(
        unet=unet, _execution_device="cpu", tokenizer=Tokenizer(), tokenizer_2=Tokenizer(),
        encode_prompt=lambda prompt, **_: ({"red coat": left, "blue coat": right}[prompt], None, None, None),
    )
    spec = RegionalPrompts(
        kind="regional_prompts", method="sdxl-attention-v1",
        regions=(
            PromptRegion(prompt="red coat", left=0, top=0, right=75, bottom=75),
            PromptRegion(prompt="blue coat", left=25, top=0, right=100, bottom=75),
        ),
    )
    source = torch.randn((2, 16, 8), generator=rand)
    tokens = torch.randn((2, 2, 8), generator=rand)
    with torch.no_grad():
        baseline = attention(source, encoder_hidden_states=tokens)
        arm_a = attention(source[1:], encoder_hidden_states=torch.cat((tokens[1:], left), dim=1))
        arm_b = attention(source[1:], encoder_hidden_states=torch.cat((tokens[1:], right), dim=1))
        with pytest.raises(Cancelled), prepared(pipe, spec, None):
            actual = attention(source, encoder_hidden_states=tokens)
            assert unet.attn_processors[own] is self_processor
            torch.testing.assert_close(actual[0], baseline[0])
            torch.testing.assert_close(actual[1, 0], arm_a[0, 0])
            torch.testing.assert_close(actual[1, 1], (arm_a[0, 1] + arm_b[0, 1]) / 2)
            torch.testing.assert_close(actual[1, 3], arm_b[0, 3])
            torch.testing.assert_close(actual[1, 12:], baseline[1, 12:])
            raise Cancelled
        torch.testing.assert_close(attention(source, encoder_hidden_states=tokens), baseline)
    assert attention.processor is original
    assert unet.attn_processors == {cross: original, own: self_processor}


def _save_regional(store):
    spec = RegionalPrompts(
        kind="regional_prompts", method="sdxl-attention-v1",
        regions=(PromptRegion(prompt="red coat", left=0, top=0, right=50, bottom=100),),
    )
    request = GenerateRequest(
        prompt="city square", width=1024, height=1024, steps=28, guidance_scale=5,
        spatial_conditioning=spec,
    )
    with closing(Image.new("RGB", ROOT_SIZE, "red")) as pixels:
        return store.save_image(pixels, request, 1, 1, MODEL_ID, quantization=QUANTIZATION, batch=1)


def test_bundle_carries_spatial_intent_and_refuses_downgraded_manifest(store, monkeypatch, tmp_path):
    made = _save_regional(store)
    bundle = Path(bundles.export_bundle(keep_gallery=True)["path"])
    changed = tmp_path / "downgraded.zip"
    with ZipFile(bundle) as source, ZipFile(changed, "w") as destination:
        manifest = json.loads(source.read("manifest.json"))
        assert manifest["version"] == 4
        manifest["version"] = 1
        for name in source.namelist():
            destination.writestr(name, json.dumps(manifest) if name == "manifest.json" else source.read(name))
    target = use_destination(store, monkeypatch, tmp_path)
    with pytest.raises(BundleError, match="requires bundle version 4"):
        bundles.import_bundle(changed)
    bundles.import_bundle(bundle)
    (row,), _, _ = store.list_images()
    assert row.spatial_conditioning == made.spatial_conditioning
    assert png_metadata(target.outputs_dir / row.filename)["spatial_conditioning"] == (
        made.spatial_conditioning.model_dump(mode="json")
    )


@pytest.mark.parametrize("database", [False, True])
def test_reindex_recovers_regional_recipe_from_both_ingress_paths(store, monkeypatch, tmp_path, database):
    original = store.settings
    made = _save_regional(store)
    target = use_destination(store, monkeypatch, tmp_path)
    monkeypatch.setattr(reindex, "settings", target)
    reindex.reindex(original.outputs_dir, database=original.db_path if database else None)
    (row,), _, _ = store.list_images()
    assert row.spatial_conditioning == made.spatial_conditioning
