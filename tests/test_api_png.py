"""Load settings from PNG over HTTP: a PNG's text chunks, as the page reads
them, answered as the panel's settings after `applied()` - this app's record
in full, an A1111 'parameters' block's prompt, negative and size - and the
refusals in words."""
from __future__ import annotations

import json

from PIL import Image

from backend import api_png, png_settings
from backend.config import settings
from backend.model_defaults import defaults_for
from backend.schemas import GenerateRequest

A1111 = (
    "masterpiece, a lighthouse in fog,\n"
    "(storm clouds:1.2)\n"
    "Negative prompt: lowres, blurry\n"
    "bad hands\n"
    'Steps: 28, Sampler: DPM++ 2M, Schedule type: Karras, CFG scale: 6.5, Seed: 4242, '
    'Size: 832x1216, Model hash: 0f1e2d, Model: ponyDiffusionV6XL, VAE: sdxl_vae.safetensors, '
    'Lora hashes: "detail: 1a2b, style: 3c4d", Version: f2.0.1'
)


def plant(store, prompt="a heron on a post at dawn", *, seed=11, width=256, height=256, **fields):
    model_id = fields.pop("model_id", settings.model_id)
    return store.save_image(
        Image.new("RGB", (width, height), (seed % 255, 40, 90)),
        GenerateRequest(prompt=prompt, width=width, height=height, **fields),
        seed=seed, duration_ms=1, model_id=model_id,
    )


def chunks_of(store, record) -> dict[str, str]:
    """The text chunks as the page sends them: every keyword, every text."""
    with Image.open(store.settings.outputs_dir / record.filename) as image:
        return {key: value for key, value in image.info.items() if isinstance(value, str)}


def ask(client, chunks, name="cat.png", status=200):
    response = client.post("/api/png/settings", json={"name": name, "chunks": chunks})
    assert response.status_code == status, response.text
    return response.json()


def test_a_gallery_png_gives_back_its_whole_record(client, fake, store):
    record = plant(
        store, negative_prompt="blurry", steps=9, guidance_scale=2.5, mu=1.4, seed=77,
        width=512, height=768, lora="ink.safetensors", lora_scale=0.6,
    )
    answer = ask(client, chunks_of(store, record))
    assert (answer["source"], answer["name"], answer["model_id"]) == ("local-diffusion", "cat.png", settings.model_id)
    assert (answer["size"], answer["sampling"]) == (True, True)
    ran = answer["settings"]
    assert (ran["prompt"], ran["negative_prompt"], ran["seed"]) == ("a heron on a post at dawn", "blurry", 77)
    assert (ran["steps"], ran["guidance_scale"], ran["mu"]) == (9, 2.5, 1.4)
    assert (ran["width"], ran["height"]) == (512, 768)
    assert (ran["lora"], ran["lora_scale"]) == ("ink.safetensors", 0.6)
    assert answer["negative_prompt"] == "blurry"
    assert answer["notes"] == []
    # Read, never stored: the gallery holds the one planted image.
    assert client.get("/api/gallery?all_models=true").json()["total"] == 1


def test_an_explicit_null_mu_stays_a_render_with_no_shift(client, fake, store):
    record = plant(store, mu=None)
    assert json.loads(chunks_of(store, record)["local-diffusion"])["mu"] is None
    assert ask(client, chunks_of(store, record))["settings"]["mu"] is None


def test_the_legacy_key_is_read_too(client, fake, store):
    record = plant(store, seed=5)
    chunks = chunks_of(store, record)
    chunks["krea"] = chunks.pop("local-diffusion")
    assert ask(client, chunks)["settings"]["seed"] == 5


def test_a_reference_carries_only_where_this_gallery_holds_the_same_picture(client, fake, store):
    source = plant(store, "the source", seed=1)
    other = plant(store, "another picture", seed=2)
    record = plant(store, reference=[source.id], seed=3)
    chunks = chunks_of(store, record)
    answer = ask(client, chunks)
    assert answer["settings"]["reference"] == [source.id]
    assert answer["notes"] == []
    # The same file read where #N is another picture - WSL's gallery, a backup.
    meta = json.loads(chunks["local-diffusion"])
    meta["reference"] = [other.id]
    answer = ask(client, {**chunks, "local-diffusion": json.dumps(meta)})
    assert answer["settings"]["reference"] == []
    assert answer["notes"] == [
        "One of its reference images is left out: this gallery holds no copy of it "
        "under the id it had where the image was made."
    ]
    # A record with no provenance for an id is not taken on trust either.
    meta["reference"], meta["reference_sources"] = [source.id], None
    assert ask(client, {**chunks, "local-diffusion": json.dumps(meta)})["settings"]["reference"] == []


def test_a_reference_on_a_checkpoint_that_takes_none_is_said(client, fake, store):
    source = plant(store, "the source", seed=1)
    record = plant(store, reference=[source.id], seed=3)
    fake.reference_supported = False
    answer = ask(client, chunks_of(store, record))
    assert answer["settings"]["reference"] == []
    assert answer["notes"] == [
        "Its reference image is not used: this checkpoint takes none, so this renders from the prompt alone."
    ]


def test_a_refine_pass_fills_the_panel_as_a_direct_render_at_its_own_size(client, fake, store):
    source = plant(store, "the source", seed=1)
    record = plant(store, refine=source.id, strength=0.4, width=512, height=512)
    answer = ask(client, chunks_of(store, record))
    assert answer["settings"]["refine"] is None
    assert answer["size"] is False
    assert answer["notes"][0].startswith("It is a refine pass, which is made again from its source")


def test_a_pngs_gan_is_left_out_with_its_gallery_refine_source(client, store):
    from backend import refine

    source = plant(store)
    record = plant(
        store, refine=source.id, upscaler=refine.ANIME6B,
        model_id="LyliaEngine/waiIllustriousSDXL_v170",
    )
    answer = ask(client, chunks_of(store, record))
    assert answer["settings"]["refine"] is None and answer["settings"]["upscaler"] is None
    assert any("upscaler is left out" in note for note in answer["notes"])


def test_an_adapter_this_installation_cannot_load_is_said(client, fake, store):
    record = plant(store, lora="ink.safetensors", lora_scale=0.8)
    fake.lora_reason = "peft is not installed"
    answer = ask(client, chunks_of(store, record))
    assert answer["settings"]["lora"] is None
    assert "Its adapter ink.safetensors @ 0.8 is not applied here" in answer["notes"][0]


def test_a_negative_applied_leaves_out_is_kept_for_the_box(client, fake, store):
    record = plant(store, negative_prompt="blurry", guidance_scale=4)
    meta = json.loads(chunks_of(store, record)["local-diffusion"])
    fake.cfg_threshold = 5.0
    answer = ask(client, {"local-diffusion": json.dumps(meta)})
    assert answer["settings"]["negative_prompt"] is None
    assert answer["negative_prompt"] == "blurry"
    assert answer["notes"] == [
        "Its negative prompt does not run here at these settings; the box keeps it, greyed, as it keeps a typed one."
    ]


def test_an_a1111_block_carries_the_prompt_negative_and_size_and_says_what_does_not(client, fake, store):
    answer = ask(client, {"parameters": A1111, "Software": "Forge"}, name="forge.png")
    profile = defaults_for(settings.model_id)
    assert (answer["source"], answer["model_id"], answer["size"], answer["sampling"]) == (
        "parameters", None, True, False,
    )
    ran = answer["settings"]
    assert ran["prompt"] == "masterpiece, a lighthouse in fog,\n(storm clouds:1.2)"
    assert ran["negative_prompt"] == answer["negative_prompt"] == "lowres, blurry\nbad hands"
    assert (ran["width"], ran["height"]) == (832, 1216)
    # Never a mapping: steps and CFG are the profile's (the page keeps its
    # own, `sampling` False), the seed none.
    assert (ran["steps"], ran["guidance_scale"], ran["seed"]) == (profile["steps"], profile["guidance_scale"], None)
    assert answer["notes"] == [
        "Read from its A1111-style 'parameters' text: the prompt, negative prompt and size.",
        "Its steps 28, sampler DPM++ 2M, schedule type Karras, CFG scale 6.5, seed 4242 and "
        "checkpoint ponyDiffusionV6XL do not carry over: they ran with another tool's sampler and "
        "checkpoint, and here the same numbers would not make the same picture. The panel keeps its "
        "own steps, CFG and seed.",
        "Not read: VAE, Lora hashes, Version.",
    ]
    assert client.get("/api/gallery?all_models=true").json()["total"] == 0


def test_an_a1111_negative_is_judged_at_the_panels_cfg_not_the_profiles(client, fake, store):
    # The page keeps its own steps and CFG for another tool's PNG, so the
    # note on the negative is about the CFG Generate will send: at the
    # panel's 6 over a threshold of 5 it runs, at its 4 it does not.
    fake.cfg_threshold = 5.0
    block = "a castle\nNegative prompt: lowres\nSteps: 30, CFG scale: 7, Seed: 1, Size: 512x512"

    def asked(panel):
        response = client.post(
            "/api/png/settings", json={"name": "c.png", "chunks": {"parameters": block}, "panel": panel},
        )
        assert response.status_code == 200, response.text
        return response.json()

    said = "Its negative prompt does not run here at these settings"
    answer = asked({"steps": 12, "guidance_scale": 6, "true_cfg": None, "mu": None})
    ran = answer["settings"]
    assert (ran["steps"], ran["guidance_scale"], ran["negative_prompt"]) == (12, 6, "lowres")
    assert not any(said in note for note in answer["notes"])
    answer = asked({"guidance_scale": 4})
    assert answer["settings"]["negative_prompt"] is None and answer["negative_prompt"] == "lowres"
    assert any(said in note for note in answer["notes"])
    # A panel value Generate would refuse is the profile's, not a refusal.
    answer = asked({"steps": 500, "guidance_scale": 6})
    assert answer["settings"]["steps"] == defaults_for(settings.model_id)["steps"]
    # This app's own record runs as recorded, whatever the panel holds.
    record = plant(store, negative_prompt="blurry", guidance_scale=4)
    response = client.post("/api/png/settings", json={
        "chunks": chunks_of(store, record), "panel": {"guidance_scale": 9},
    })
    assert response.json()["settings"]["guidance_scale"] == 4


def test_this_apps_record_wins_over_its_own_parameters_line(client, fake, store):
    record = plant(store, seed=9)
    chunks = chunks_of(store, record)
    assert "parameters" in chunks
    assert ask(client, chunks)["source"] == "local-diffusion"


def test_a_size_a_direct_render_does_not_take_leaves_the_panels(client, fake, store):
    answer = ask(client, {"parameters": "a cat\nSteps: 20, Seed: 1, Size: 600x800"})
    assert answer["size"] is False
    assert (answer["settings"]["width"], answer["settings"]["height"]) == (1024, 1024)
    assert "Its size 600x800 is not carried (width: 600 is not a multiple of 16: 592 or 608); " \
        "the panel keeps its own." in answer["notes"]
    side = settings.max_side + 512
    answer = ask(client, {"parameters": f"a cat\nSteps: 20, Seed: 1, Size: {side}x512"})
    assert answer["size"] is False
    assert f"a {side}-pixel side is past the {settings.max_side} a direct render takes" in answer["notes"][1]
    assert "Value error" not in answer["notes"][1] and "request:" not in answer["notes"][1]
    answer = ask(client, {"parameters": "a cat\nnegative-free, no settings line"})
    assert answer["notes"][:2] == [
        "Read from its A1111-style 'parameters' text: the prompt.",
        "It names no size, so the panel keeps its own.",
    ]
    assert answer["settings"]["prompt"] == "a cat\nnegative-free, no settings line"


def test_refusals_are_in_words(client, fake, store):
    def refused(chunks, status=422, name="cat.png"):
        return ask(client, chunks, name=name, status=status)["detail"]

    assert refused({"Software": "GIMP", "Title": "x"}) == (
        "cat.png carries no settings this app reads (its text: Software, Title): neither Local "
        "Diffusion's own record nor an A1111-style 'parameters' text."
    )
    assert refused({}).startswith("cat.png carries no settings this app reads, nor any text at all")
    assert refused({"workflow": json.dumps({"nodes": []}), "prompt": json.dumps({"3": {}})}).startswith(
        "cat.png carries a ComfyUI workflow, not settings this app reads"
    )
    assert refused({"local-diffusion": "{not json"}) == "The Local Diffusion metadata in cat.png is not JSON."
    assert refused({"local-diffusion": "[1, 2]"}) == (
        "The Local Diffusion metadata in cat.png is not a settings record."
    )
    assert refused({"local-diffusion": json.dumps({"prompt": ""})}) == (
        "The Local Diffusion metadata in cat.png carries no prompt."
    )
    assert refused({"parameters": "Negative prompt: only this\nSteps: 1, Seed: 2, Size: 512x512"}) == (
        "The 'parameters' text in cat.png carries no prompt."
    )
    assert refused({"local-diffusion": json.dumps({"prompt": "x", "steps": 500})}) == (
        "The settings in cat.png do not fit this app: steps: Input should be less than or equal to 100."
    )


def test_the_text_payload_is_capped(client, fake, store):
    big = {"name": "big.png", "chunks": {"Comment": "x" * (api_png.MAX_BODY + 1)}}
    response = client.post("/api/png/settings", json=big)
    assert response.status_code == 413
    assert response.json()["detail"].startswith("The PNG's text is over 1 MiB")
    many = {f"k{i}": "v" for i in range(65)}
    response = client.post("/api/png/settings", json={"name": "x.png", "chunks": many})
    assert response.status_code == 422
    response = client.post("/api/png/settings", json={"name": "x.png", "chunks": {"k" * 80: "v"}})
    assert response.status_code == 422 and "is not a PNG text keyword" in response.json()["detail"]


def test_a_stale_page_is_refused_before_anything_is_read(client, fake, store):
    response = client.post(
        "/api/png/settings", json={"chunks": {"parameters": "a cat"}},
        headers={"X-Local-Diffusion-Model-Revision": "999"},
    )
    assert response.status_code == 409


def test_parse_parameters_reads_as_a1111_does():
    prompt, negative, params = png_settings.parse_parameters(A1111)
    assert prompt == "masterpiece, a lighthouse in fog,\n(storm clouds:1.2)"
    assert negative == "lowres, blurry\nbad hands"
    assert params["Lora hashes"] == "detail: 1a2b, style: 3c4d"
    assert list(params)[:3] == ["Steps", "Sampler", "Schedule type"]
    # A last line with fewer than three settings is prompt text.
    assert png_settings.parse_parameters("a cat\nStyle: ink, Mood: calm") == (
        "a cat\nStyle: ink, Mood: calm", "", {},
    )
    assert png_settings.parse_parameters("  a cat  \n") == ("a cat", "", {})
