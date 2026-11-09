"""FLUX 3 [Klein]'s prompt upsampler as Expand: pinned, quantized, offloaded
first, decoded as the card says, and never part of the image call.

No weights and no Hub: the model, its processor and the downloads are
stand-ins. What runs on the card (fit, time, the caption itself) is not
measured here (docs/verified.md).
"""
from __future__ import annotations

import dataclasses
import json

import pytest
import torch

from backend.backends import Cancelled, flux3_loading, flux3_upsampler, loading
from backend.backends import diffusers_backend as module
from backend.backends.flux3_pipeline import Flux3KleinBasePipeline, Flux3KleinPipeline
from tests.backend_helpers import loaded_backend
from tests.test_flux3 import _pipe, _render

PROMPT_IDS = [11, 12, 13]


class _Inputs(dict):
    def to(self, device):
        self["device"] = device
        return self


class _UpsamplerProcessor:
    def __init__(self):
        self.templates: list[tuple[list, dict]] = []
        self.decoded: list[list[int]] = []

    def apply_chat_template(self, messages, **kwargs):
        self.templates.append((messages, kwargs))
        return _Inputs(input_ids=torch.tensor([PROMPT_IDS]), attention_mask=torch.ones(1, len(PROMPT_IDS)))

    def decode(self, ids, skip_special_tokens):
        assert skip_special_tokens
        self.decoded.append(ids.tolist())
        return " A full scene caption. "


class _Upsampler(torch.nn.Module):
    """`generate` as transformers runs it: each step's scores through the
    logits processors, the stopping criteria after each token."""

    def __init__(self, scores=None, steps=3):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.calls: list[dict] = []
        self.scores = scores
        self.steps = steps

    @property
    def device(self):
        return self.weight.device

    @property
    def dtype(self):
        return self.weight.dtype

    def generate(self, input_ids, attention_mask, device, **kwargs):
        self.calls.append(kwargs)
        ids = input_ids
        for _ in range(self.steps):
            scores = self.scores.clone() if self.scores is not None else torch.zeros(1, 32)
            for processor in kwargs["logits_processor"]:
                scores = processor(ids, scores)
            ids = torch.cat([ids, scores.argmax(-1, keepdim=True)], dim=1)
            if any(bool(criterion(ids, scores).all()) for criterion in kwargs["stopping_criteria"]):
                break
        return ids


# --- the call ------------------------------------------------------------------


def test_the_chat_is_diffusers_t2i_with_its_shipped_system_prompt():
    from diffusers.modular_pipelines.flux3.prompt_upsampler import SYSTEM_PROMPTS

    processor, model = _UpsamplerProcessor(), _Upsampler()
    caption = flux3_upsampler.upsample(model, processor, "a man walking his dog", torch.device("cpu"))
    assert caption == "A full scene caption."
    ((messages, kwargs),) = processor.templates
    assert messages == [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPTS["t2i"]}]},
        {"role": "user", "content": [{"type": "text", "text": "a man walking his dog"}]},
    ]
    assert kwargs == {
        "add_generation_prompt": True, "tokenize": True, "return_dict": True,
        "return_tensors": "pt", "enable_thinking": False,
    }
    # Only the answer is decoded, never the prompt's tokens.
    assert len(processor.decoded[0]) == model.steps


def test_decoding_is_greedy_with_the_cards_presence_penalty():
    from diffusers.modular_pipelines.flux3.prompt_upsampler import MAX_NEW_TOKENS

    # Token 5 leads by 1.0 every step: greedy picks it first, and the card's
    # 1.5 presence penalty then hands the next steps to the runner-up, once.
    scores = torch.zeros(1, 32)
    scores[0, 5] = 3.0
    scores[0, 6] = 2.0
    scores[0, 11] = 10.0  # a prompt token: never penalised, so it wins first
    model = _Upsampler(scores=scores, steps=4)
    processor = _UpsamplerProcessor()
    flux3_upsampler.upsample(model, processor, "a fox", torch.device("cpu"))
    (call,) = model.calls
    assert call["do_sample"] is False and call["max_new_tokens"] == MAX_NEW_TOKENS == 3584
    assert "top_p" not in call and "top_k" not in call and "temperature" not in call
    # 11 (prompt id, unpenalised), then 11 is penalised to 8.5, still first;
    # the penalty is once per token however often: 11 again; never 5 over it.
    assert processor.decoded[0] == [11, 11, 11, 11]
    assert flux3_upsampler.PRESENCE_PENALTY == 1.5
    scores[0, 11] = 0.0
    model, processor = _Upsampler(scores=scores, steps=3), _UpsamplerProcessor()
    flux3_upsampler.upsample(model, processor, "a fox", torch.device("cpu"))
    # 5 (3.0); then 5 at 1.5 loses to 6 (2.0); then 6 at 0.5 and 5 at 1.5: 5.
    assert processor.decoded[0] == [5, 6, 5]


def test_stop_ends_the_generation_at_the_next_token_and_raises():
    stopped = {"value": False}
    model = _Upsampler(steps=50)
    original = model.generate

    def generate(*args, **kwargs):
        stopped["value"] = True
        return original(*args, **kwargs)

    model.generate = generate  # type: ignore[method-assign]
    with pytest.raises(Cancelled):
        flux3_upsampler.upsample(model, _UpsamplerProcessor(), "a fox", torch.device("cpu"), lambda: stopped["value"])


def test_an_empty_caption_is_an_error():
    processor = _UpsamplerProcessor()
    processor.decode = lambda ids, skip_special_tokens: "  "  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="empty caption"):
        flux3_upsampler.upsample(_Upsampler(), processor, "a fox", torch.device("cpu"))


# --- the pipeline -----------------------------------------------------------------


@pytest.mark.parametrize("cls", [Flux3KleinPipeline, Flux3KleinBasePipeline])
def test_the_upsampler_is_first_in_the_offload_chain_and_never_in_the_image_call(cls):
    assert cls.model_cpu_offload_seq.split("->") == ["prompt_upsampler", "text_encoder", "transformer", "vae"]
    assert set(cls._optional_components) == {"prompt_upsampler", "prompt_upsampler_processor"}
    pipe = _pipe(cls)
    assert pipe.prompt_upsampler is None
    with pytest.raises(RuntimeError, match="not loaded"):
        pipe.upsample_prompt("a fox")

    class Untouchable(_Upsampler):
        def generate(self, *args, **kwargs):
            pytest.fail("the image call must never run the upsampler")

    pipe.register_modules(prompt_upsampler=Untouchable(), prompt_upsampler_processor=_UpsamplerProcessor())
    images = _render(pipe)
    assert len(images) == 1
    # The encoder saw the prompt as typed.
    assert pipe.tokenizer.prompts[0] == "sunny garden"


def test_the_pipeline_runs_its_registered_upsampler_on_its_execution_device():
    pipe = _pipe(Flux3KleinPipeline)
    model, processor = _Upsampler(), _UpsamplerProcessor()
    pipe.register_modules(prompt_upsampler=model, prompt_upsampler_processor=processor)
    assert pipe.upsample_prompt("a fox") == "A full scene caption."
    assert processor.templates[0][0][1]["content"][0]["text"] == "a fox"


# --- loading ---------------------------------------------------------------------


def _pup_tree(tmp_path, architectures=("Qwen3_5ForConditionalGeneration",)):
    tree = tmp_path / "pup"
    tree.mkdir()
    (tree / "config.json").write_text(json.dumps({"architectures": list(architectures)}), encoding="utf-8")
    return str(tree)


@pytest.mark.parametrize(("quant", "scheme"), [("fp8", "Float8WeightOnlyConfig"), ("int8", "Int8WeightOnlyConfig"),
                                               ("none", "Int8WeightOnlyConfig")])
def test_the_load_is_pinned_configs_first_and_quantized_through_the_mapping(tmp_path, monkeypatch, quant, scheme):
    import huggingface_hub
    import transformers

    tree = _pup_tree(tmp_path)
    snapshots: list[tuple[str, str, list[str]]] = []

    def snapshot(repo_id, revision, allow_patterns):
        snapshots.append((repo_id, revision, list(allow_patterns)))
        return tree

    loads: list[tuple[str, dict]] = []

    class Model(torch.nn.Module):
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            loads.append((path, kwargs))
            return cls()

    class Processor:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            assert kwargs == {"local_files_only": True}
            return cls()

    owned = []
    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot)
    monkeypatch.setattr(transformers, "Qwen3_5ForConditionalGeneration", Model)
    monkeypatch.setattr(transformers, "AutoProcessor", Processor)
    monkeypatch.setattr(loading, "own_tensors", lambda module, names=None: owned.append(module) or 0)
    settings = dataclasses.replace(module.settings, quantization=quant, offload="model")
    components = flux3_upsampler.load(settings)
    assert set(components) == {"prompt_upsampler", "prompt_upsampler_processor"}
    revision = "a57899549f8f6f64c6c4d88db30827d496ff76d0"
    assert snapshots == [
        (flux3_upsampler.MODEL_ID, revision, ["*.json", "chat_template.jinja"]),
        (flux3_upsampler.MODEL_ID, revision, ["*.json", "chat_template.jinja", "model.safetensors"]),
    ]
    ((path, kwargs),) = loads
    assert path == tree and kwargs["dtype"] is torch.bfloat16 and kwargs["local_files_only"] is True
    config = kwargs["quantization_config"]
    # transformers' own TorchAoConfig, from an AOBaseConfig instance.
    assert isinstance(config, transformers.TorchAoConfig)
    assert type(config.quant_type).__name__ == scheme
    assert owned == [components["prompt_upsampler"]]


def test_a_config_naming_another_model_is_refused_before_weights(tmp_path, monkeypatch):
    import huggingface_hub

    tree = _pup_tree(tmp_path, architectures=("Qwen3VLForConditionalGeneration",))
    asked = []
    monkeypatch.setattr(huggingface_hub, "snapshot_download",
                        lambda repo_id, revision, allow_patterns: asked.append(allow_patterns) or tree)
    with pytest.raises(RuntimeError, match="does not name Qwen3_5ForConditionalGeneration; no weights"):
        flux3_upsampler.load(dataclasses.replace(module.settings, offload="model"))
    assert asked == [flux3_upsampler.CONFIG_PATTERNS]


def test_the_cache_check_reads_files_only_at_the_pin(monkeypatch, tmp_path):
    import huggingface_hub

    weights = tmp_path / "model.safetensors"
    asked = []

    def cache(repo_id, filename, revision=None, **kwargs):
        asked.append((repo_id, filename, revision))
        return str(weights)

    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", cache)
    assert not flux3_upsampler.cached()
    weights.write_bytes(b"w")
    assert flux3_upsampler.cached()
    assert asked[0] == (flux3_upsampler.MODEL_ID, "model.safetensors", flux3_upsampler.SOURCE["revision"])


def test_without_offload_it_is_refused_in_words():
    held = flux3_upsampler.refusal(dataclasses.replace(module.settings, offload="none"))
    assert held is not None and "offload none" in held and "Model offload" in held
    for offload in ("model", "sequential"):
        assert flux3_upsampler.refusal(dataclasses.replace(module.settings, offload=offload)) is None


# --- the backend -----------------------------------------------------------------


@pytest.fixture
def flux3_settings(monkeypatch):
    def configure(**changes):
        values = {"model_id": flux3_loading.MODEL_ID, "offload": "model", "device": "cpu", **changes}
        configured = dataclasses.replace(module.settings, **values)
        monkeypatch.setattr(module, "settings", configured)
        return configured
    configure()
    return configure


def test_cold_support_follows_the_flux3_ids(flux3_settings):
    backend = module.DiffusersBackend()
    assert backend.prompt_expansion_supported()
    flux3_settings(model_id=flux3_loading.BASE_MODEL_ID)
    assert backend.prompt_expansion_supported()
    flux3_settings(model_id="black-forest-labs/FLUX.2-klein-4B")
    assert not backend.prompt_expansion_supported()
    assert backend.prompt_expander_status() is None


def test_the_status_names_the_upsampler_download_until_it_is_registered(flux3_settings, monkeypatch):
    pipe = _pipe(Flux3KleinPipeline)
    backend = loaded_backend(pipe)
    cached = {"value": False}
    monkeypatch.setattr(flux3_upsampler, "cached", lambda: cached["value"])
    named = {"name": "FLUX 3's prompt upsampler", "translate": False}
    assert backend.prompt_expander_status() == {
        "loaded": False, "fetch": [("FLUX 3's prompt upsampler", 18_839_738_489)], **named,
    }
    cached["value"] = True
    assert backend.prompt_expander_status() == {"loaded": False, "fetch": [], **named}
    pipe.register_modules(prompt_upsampler=_Upsampler(), prompt_upsampler_processor=_UpsamplerProcessor())
    cached["value"] = False
    assert backend.prompt_expander_status() == {"loaded": True, "fetch": [], **named}
    assert backend.prompt_expansion_refusal() is None
    flux3_settings(offload="none")
    assert "offload none" in backend.prompt_expansion_refusal()


def test_expand_loads_once_joins_the_offload_chain_and_frees_its_hooks(flux3_settings, monkeypatch):
    pipe = _pipe(Flux3KleinPipeline)
    backend = loaded_backend(pipe)
    loads, placed, freed = [], [], []

    def load(settings, should_cancel=None):
        loads.append(settings.quantization)
        return {"prompt_upsampler": _Upsampler(), "prompt_upsampler_processor": _UpsamplerProcessor()}

    monkeypatch.setattr(flux3_upsampler, "load", load)
    monkeypatch.setattr(loading, "place_pipeline", lambda current, settings: placed.append(settings.offload))
    monkeypatch.setattr(pipe, "maybe_free_model_hooks", lambda: freed.append(True))
    assert backend.expand_prompt("a fox", 1264, 848) == "A full scene caption."
    assert backend.expand_prompt("a hare", 1024, 1024) == "A full scene caption."
    assert len(loads) == 1 and placed == ["model"] and len(freed) == 2
    assert pipe.prompt_upsampler is not None
    # No size reaches BFL's upsampler: its chat is the request alone.
    texts = [m[1]["content"][0]["text"] for m, _ in pipe.prompt_upsampler_processor.templates]
    assert texts == ["a fox", "a hare"]


def test_a_stop_during_the_download_never_starts_the_gpu_work(flux3_settings, monkeypatch):
    pipe = _pipe(Flux3KleinPipeline)
    backend = loaded_backend(pipe)
    stopped = {"value": False}

    def load(settings, should_cancel=None):
        stopped["value"] = True
        return {"prompt_upsampler": _Upsampler(), "prompt_upsampler_processor": _UpsamplerProcessor()}

    monkeypatch.setattr(flux3_upsampler, "load", load)
    monkeypatch.setattr(loading, "place_pipeline", lambda *args: pytest.fail("nothing is placed after a stop"))
    with pytest.raises(Cancelled):
        backend.expand_prompt("a fox", 1024, 1024, lambda: stopped["value"])
    assert pipe.prompt_upsampler is None


def test_without_offload_expand_refuses_before_any_load(flux3_settings, monkeypatch):
    flux3_settings(offload="none")
    backend = loaded_backend(_pipe(Flux3KleinPipeline))
    monkeypatch.setattr(flux3_upsampler, "load", lambda *args, **kwargs: pytest.fail("refused before loading"))
    with pytest.raises(RuntimeError, match="offload none"):
        backend.expand_prompt("a fox", 1024, 1024)
