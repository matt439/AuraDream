"""End-to-end through the HTTP layer, with a backend that draws nothing:
the profile defaults a bare request takes, what /api/status says, and the
guards. The stream, cancellation, the grid and the conditioning inputs
have their own test_api_* files on the same fixtures (tests/conftest.py,
tests/fakes.py, tests/api_helpers.py)."""
from __future__ import annotations

import dataclasses
import time

import pytest

from backend import main
from backend.config import settings
from backend.model_defaults import defaults_for, hints_for
from backend.tags import TagLookupUnavailable
from tests.api_helpers import ORIGIN, run_one, stream_job


# ------------------------------------------------------------------ defaults


def test_a_bare_request_takes_the_checkpoints_profile(client, fake):
    # The schema used to carry Turbo's 8 / 0.0 / 1.15 whatever the checkpoint,
    # so `curl -d '{"prompt": "..."}'` ran FLUX at 8 steps.
    run_one(client)
    profile = defaults_for(settings.model_id)
    request = fake.requests[-1]
    assert (request.steps, request.guidance_scale, request.mu) == (
        profile["steps"],
        profile["guidance_scale"],
        profile["mu"],
    )


@pytest.mark.parametrize(
    "model_id",
    ["krea/Krea-2-Turbo", "krea/Krea-2-Raw", "black-forest-labs/FLUX.1-Krea-dev", "nobody/whatever"],
)
def test_every_profile_explains_its_controls(model_id):
    # The hints are format strings over the profile's own numbers, so they can
    # never quote a value the controls are not seeded with - and a mu of None or
    # a renamed key would blow up here rather than on the first hover.
    hints = hints_for(model_id)
    # The Refine dialog's words past the size a pass was measured at - the run
    # it quotes, `refine_tight`, and FLUX.1's `refine_large` - are only where
    # a pass was measured; FLUX.1's name for its CFG box, `cfg_label`, only
    # where the box is a distilled guidance value (test_profiles).
    assert set(hints) - {"refine_large", "refine_tight", "cfg_label"} == {
        "steps", "guidance_scale", "mu", "true_cfg", "reference", "refine", "prompt", "size"
    }
    assert all(text and "{" not in text for text in hints.values())
    # The LoRA box is the one control no profile can speak for: whether an
    # adapter can be applied is the installation's answer, so the hint appears
    # only once something has given one.
    assert set(hints_for(model_id, lora_supported=True)) == set(hints) | {"lora"}


def test_status_hints_quote_the_loaded_pipelines_threshold(client, fake, monkeypatch):
    # Where guidance switches on is the pipeline's answer, not the id's guess.
    # The tooltip and the greyed-out negative prompt field read the same number
    # - on the box that drives the negative prompt, which on FLUX.1 is True CFG.
    described = dict(fake.describe(), cfg_threshold=7.0)
    monkeypatch.setattr(fake, "describe", lambda: described)
    body = client.get("/api/status").json()
    assert body["defaults"]["cfg_threshold"] == 7.0
    knob = "true_cfg" if body["defaults"]["true_cfg_supported"] else "guidance_scale"
    assert "7" in body["hints"][knob]


def test_status_says_whether_real_cfg_is_a_separate_knob(client, fake, monkeypatch):
    # The profile's guess until a pipeline is loaded, then the parameter list -
    # and the hint for the True CFG box follows the answer, so a pipeline the
    # id table did not expect cannot show a live box under a "not a knob" tip.
    guess = defaults_for(settings.model_id)["true_cfg_supported"]
    body = client.get("/api/status").json()
    assert body["defaults"]["true_cfg_supported"] is guess
    described = dict(fake.describe(), true_cfg_supported=not guess)
    monkeypatch.setattr(fake, "describe", lambda: described)
    body = client.get("/api/status").json()
    assert body["defaults"]["true_cfg_supported"] is not guess
    assert ("Not a separate knob" in body["hints"]["true_cfg"]) is guess


def test_status_says_whether_mu_applies_to_the_scheduler(client, fake, monkeypatch):
    guess = defaults_for(settings.model_id)["mu_supported"]
    assert client.get("/api/status").json()["defaults"]["mu_supported"] is guess
    described = dict(fake.describe(), mu_supported=not guess)
    monkeypatch.setattr(fake, "describe", lambda: described)
    assert client.get("/api/status").json()["defaults"]["mu_supported"] is not guess


def test_status_carries_when_the_server_started(client):
    # Set once, at import: a page that sees it change knows the server it was
    # following restarted, and every job with it.
    first = client.get("/api/status").json()["started_at"]
    assert first == main.STARTED_AT
    assert client.get("/api/status").json()["started_at"] == first


def test_status_names_the_checkpoint_by_its_catalogue_label(client, fake, monkeypatch):
    # The start-up menu's name, which carries the pinned version the
    # repository's own name hides; off the catalogue, none, and the page
    # shows the id. The id itself is unchanged either way.
    body = client.get("/api/status").json()
    assert body["model_id"] == settings.model_id
    assert body["model_label"] == next(c.name for c in main.CHECKPOINTS if c.repo == settings.model_id)
    for repo, label in (
        ("duongve/NetaYume-Lumina-Image-2.0", "NetaYume Lumina v4"),
        ("SeeSee21/Z-Anime", "Z-Anime Base"),
        ("circlestone-labs/Anima", "Anima Turbo v1.1"),
        ("someone/a-checkpoint-of-their-own", None),
    ):
        described = dict(fake.describe(), model_id=repo)
        monkeypatch.setattr(fake, "describe", lambda described=described: described)
        body = client.get("/api/status").json()
        assert (body["model_id"], body["model_label"]) == (repo, label)


def test_status_lists_the_live_jobs(client, fake):
    # A page watching no job learns from the poll that another tab's holds
    # the card; one watching its own, of jobs queued behind it from
    # elsewhere and of queued ones cancelled there. Oldest first.
    assert client.get("/api/status").json()["active_jobs"] == []
    fake.step_seconds = 0.02
    ahead = client.post(
        "/api/generate", json={"prompt": "x", "width": 256, "height": 256, "steps": 30}
    ).json()["job_id"]
    behind = client.post(
        "/api/generate", json={"prompt": "y", "width": 256, "height": 256, "steps": 2}
    ).json()["job_id"]
    assert client.get("/api/status").json()["active_jobs"] == [ahead, behind]
    client.post(f"/api/jobs/{behind}/cancel")
    assert client.get("/api/status").json()["active_jobs"] == [ahead]
    stream_job(client, ahead)
    assert client.get("/api/status").json()["active_jobs"] == []


def _gated():
    from huggingface_hub.errors import GatedRepoError

    # As huggingface_hub raises it, without the HTTP response its init wants.
    error = GatedRepoError.__new__(GatedRepoError)
    error.args = ("Cannot access gated repo for url https://huggingface.co/x/resolve/main/model_index.json.",)
    return error


def _wait_for_state(client, state):
    for _ in range(200):
        body = client.get("/api/status").json()
        if body["state"] == state:
            return body
        time.sleep(0.01)
    raise AssertionError(f"status never reached {state}")


@pytest.mark.parametrize("wrapped", [False, True])
def test_a_gated_load_says_what_to_do(client, fake, monkeypatch, wrapped):
    # The default checkpoints are gated, so this is the likeliest first
    # failure: the library's words named a JSON file's URL and no steps. Said
    # with the steps and the gate's kind from the catalogue, the model page
    # beside it - and found through a loader's OSError raised from it too.
    def load():
        if not wrapped:
            raise _gated()
        try:
            raise _gated()
        except Exception as exc:
            raise OSError("black-forest-labs is not a valid model identifier") from exc

    fake.loaded = False
    monkeypatch.setattr(fake, "load", load)
    # The catalogue's entry for the id the status names - a click-through gate.
    monkeypatch.setattr(main, "settings", dataclasses.replace(main.settings, model_id="black-forest-labs/FLUX.1-Krea-dev"))
    client.post("/api/model/load")
    body = _wait_for_state(client, "error")
    assert body["detail"] == (
        "FLUX.1 Krea [dev] is gated on Hugging Face: accept its terms on the model page "
        "(a click-through), run `uv run hf auth login` once with a read token, then press Retry load."
    )
    assert body["detail_kind"] == "gated"
    assert body["detail_url"] == "https://huggingface.co/black-forest-labs/FLUX.1-Krea-dev"
    # A job that ends at the load says the same sentence, not the library's.
    job_id = client.post("/api/generate", json={"prompt": "x", "width": 256, "height": 256}).json()["job_id"]
    done = stream_job(client, job_id)[-1]
    assert (done["status"], done["error"], done["started_at"]) == ("error", body["detail"], None)
    # And a load that then succeeds clears the kind and the page with the detail.
    monkeypatch.setattr(fake, "load", lambda: setattr(fake, "loaded", True))
    client.post("/api/model/load")
    ready = _wait_for_state(client, "ready")
    assert (ready["detail"], ready["detail_kind"], ready["detail_url"]) == (None, None, None)


def test_any_other_load_failure_keeps_its_own_words(client, fake, monkeypatch):
    def load():
        raise RuntimeError("CUDA out of memory")

    fake.loaded = False
    monkeypatch.setattr(fake, "load", load)
    client.post("/api/model/load")
    body = _wait_for_state(client, "error")
    assert (body["detail"], body["detail_kind"], body["detail_url"]) == (
        "RuntimeError: CUDA out of memory", None, None,
    )


def test_status_carries_the_prompting_guide(client):
    # The prose half rides with the tooltips; the data half - the documented
    # pixel range, the word for it, the examples - is its own object, so the
    # size line and the Examples menu read numbers and a list, not a sentence.
    body = client.get("/api/status").json()
    assert body["hints"]["prompt"] and body["hints"]["size"]
    guide = body["guide"]
    assert guide["examples"] and all(isinstance(text, str) and text for text in guide["examples"])
    assert len(guide["pixels"]) == 2 and guide["pixels"][0] <= guide["pixels"][1]
    assert guide["pixels_word"]
    assert "aspects" in guide
    assert "resolutions" in guide
    assert "tag_helper" in guide


def test_tag_vocabulary_endpoints_are_checkpoint_scoped_and_read_only(client, monkeypatch):
    helper = {
        "vocabulary": "danbooru",
        "vocabulary_cutoff": "2025-01-07",
        "groups": [{"tags": ["great score"]}],
        "recommended_suffix": ["absurdres"],
    }

    class Vocabulary:
        def suggest(self, query, cutoff):
            assert (query, cutoff) == ("green h", "2025-01-07")
            return [{
                "text": "green hair", "canonical": "green_hair",
                "category": "general", "post_count": 591_000,
            }]

        def validate(self, tags, special, cutoff, allow_year_tag):
            assert special == {"great score", "absurdres"} and cutoff == "2025-01-07"
            assert allow_year_tag is False
            return [{
                "tag": tag, "valid": True, "canonical": tag,
                "source": "model", "category": None, "post_count": None,
            } for tag in tags]

    monkeypatch.setattr(main, "_tag_helper_config", lambda: helper)
    monkeypatch.setattr(main, "tag_vocabulary", Vocabulary())
    suggestions = client.get("/api/tags/suggest?q=green%20h").json()
    assert suggestions == {
        "available": True,
        "suggestions": [{
            "text": "green hair", "canonical": "green_hair",
            "category": "general", "post_count": 591_000,
        }],
        "detail": None,
    }
    validation = client.post("/api/tags/validate", json={"tags": ["great score"]}).json()
    assert validation["available"] is True
    assert validation["results"][0]["source"] == "model"


def test_an_offline_tag_index_leaves_unknown_tags_unchecked(client, monkeypatch):
    helper = {
        "vocabulary": "danbooru",
        "vocabulary_cutoff": "2025-01-07",
        "groups": [{"tags": ["great score"]}],
        "recommended_suffix": [],
    }

    class OfflineVocabulary:
        def suggest(self, query, cutoff):
            raise TagLookupUnavailable("offline")

        def validate(self, tags, special, cutoff, allow_year_tag):
            assert allow_year_tag is False
            raise TagLookupUnavailable("offline")

    monkeypatch.setattr(main, "_tag_helper_config", lambda: helper)
    monkeypatch.setattr(main, "tag_vocabulary", OfflineVocabulary())
    suggestions = client.get("/api/tags/suggest?q=great").json()
    assert suggestions["available"] is False
    assert suggestions["suggestions"][0]["text"] == "great score"
    body = client.post(
        "/api/tags/validate", json={"tags": ["great score", "green hair"]}
    ).json()
    assert body["available"] is False
    assert [item["valid"] for item in body["results"]] == [True, None]


def test_suggestions_never_lead_with_the_recommended_negative(client, monkeypatch):
    # "lo" then Tab took the first item, and the guide's own tags came first
    # by construction: `low quality` went into the positive prompt.
    helper = {
        "vocabulary": "danbooru",
        "vocabulary_cutoff": "2025-01-07",
        "groups": [{"tags": ["low quality", "great score"]}],
        "recommended_suffix": [],
        "negative_prompt": "lowres, low quality",
    }

    class Vocabulary:
        def suggest(self, query, cutoff):
            rows = {
                "lo": [("lowres", 900_000), ("long_hair", 4_000_000)],
                "gre": [("green_eyes", 900_000), ("green_hair", 590_000)],
            }[query]
            return [
                {"text": name.replace("_", " "), "canonical": name, "category": "general",
                 "post_count": count}
                for name, count in rows
            ]

    monkeypatch.setattr(main, "_tag_helper_config", lambda: helper)
    monkeypatch.setattr(main, "tag_vocabulary", Vocabulary())
    lo = [item["text"] for item in client.get("/api/tags/suggest?q=lo").json()["suggestions"]]
    assert lo[0] == "long hair" and set(lo) == {"long hair", "lowres", "low quality"}
    gre = [item["text"] for item in client.get("/api/tags/suggest?q=gre").json()["suggestions"]]
    assert gre == ["green eyes", "green hair", "great score"]


def test_tag_specials_allow_a_profile_without_quick_actions():
    helper = {"groups": [{"tags": ["masterpiece"]}]}
    assert main._special_tags(helper) == {"masterpiece"}


def test_prompt_expansion_loads_the_model_and_returns_reviewable_text(client, fake):
    fake.prompt_expander_supported = True
    fake.loaded = False
    response = client.post(
        "/api/prompt/expand",
        json={"prompt": "a lighthouse", "width": 1264, "height": 848},
    )
    assert response.status_code == 200
    assert response.json() == {
        "original_prompt": "a lighthouse",
        "expanded_prompt": "Expanded: a lighthouse (1264x848)",
        "translated_prompt": "English: Expanded: a lighthouse (1264x848)",
        "translation_detail": None,
    }
    assert fake.loads == 1 and fake.expansions == [("a lighthouse", 1264, 848)]
    assert fake.translator.translations == ["Expanded: a lighthouse (1264x848)"]
    assert client.get("/api/status").json()["prompt_expansion_supported"] is True


def test_translation_failure_keeps_the_expanded_prompt(client, fake):
    fake.prompt_expander_supported = True
    fake.translator.fail = True
    response = client.post("/api/prompt/expand", json={"prompt": "a lighthouse"})
    assert response.status_code == 200
    assert response.json() == {
        "original_prompt": "a lighthouse",
        "expanded_prompt": "Expanded: a lighthouse (1024x1024)",
        "translated_prompt": None,
        "translation_detail": "RuntimeError: translation broke",
    }


def test_prompt_expansion_refuses_an_unsupported_checkpoint(client):
    response = client.post("/api/prompt/expand", json={"prompt": "a lighthouse"})
    assert response.status_code == 422 and "ERNIE-Image" in response.json()["detail"]


def test_prompt_expansion_does_not_cut_ahead_of_the_queue(client, fake, monkeypatch):
    fake.prompt_expander_supported = True
    monkeypatch.setattr(main.jobs, "busy", lambda: True)
    response = client.post("/api/prompt/expand", json={"prompt": "a lighthouse"})
    assert response.status_code == 409 and "queued or running" in response.json()["detail"]


def test_status_says_whether_the_pipeline_takes_a_negative_prompt(client, fake, monkeypatch):
    # The profile's guess until a pipeline is loaded, then the parameter list.
    # Derived, not a literal: it is the default checkpoint's profile that answers,
    # and klein's says False.
    guess = defaults_for(settings.model_id)["negative_prompt_supported"]
    assert client.get("/api/status").json()["defaults"]["negative_prompt_supported"] is guess
    described = dict(fake.describe(), negative_prompt_supported=not guess)
    monkeypatch.setattr(fake, "describe", lambda: described)
    assert client.get("/api/status").json()["defaults"]["negative_prompt_supported"] is not guess


def test_the_first_generate_loads_the_weights_exactly_once(client, fake):
    fake.loaded = False
    first = client.post("/api/generate", json={"prompt": "x", "width": 256, "height": 256}).json()
    second = client.post("/api/generate", json={"prompt": "y", "width": 256, "height": 256}).json()
    assert stream_job(client, first["job_id"])[-1]["status"] == "done"
    assert stream_job(client, second["job_id"])[-1]["status"] == "done"
    # Two back-to-back requests share one load, and the status agrees.
    assert fake.loads == 1
    assert client.get("/api/status").json()["state"] == "ready"


def test_a_batch_that_multiplies_past_the_budget_is_refused(client):
    response = client.post(
        "/api/generate", json={"prompt": "x", "width": 2048, "height": 2048, "num_images": 8}
    )
    assert response.status_code == 422
    assert "Runs" in response.json()["detail"]
    # 3:4 is the largest nominal-1-MP preset. The old two-at-4-MP default
    # rejected eight of these by 75,776 pixels even though Batch permits eight.
    assert client.post(
        "/api/generate", json={"prompt": "x", "width": 896, "height": 1200, "num_images": 8}
    ).status_code == 200


def test_the_origin_check_understands_a_default_port(client, monkeypatch):
    import dataclasses

    monkeypatch.setattr(main, "settings", dataclasses.replace(settings, port=80))
    # A browser serialises an origin on port 80 without the port.
    assert client.post("/api/model/unload", headers={"Origin": "http://127.0.0.1"}).status_code != 403
    assert client.post("/api/model/unload", headers={"Origin": "http://127.0.0.1:7860"}).status_code == 403


def test_an_explicit_null_mu_is_not_filled_in(client, fake):
    # What the UI sends to mean "no shift of ours"; filling it from the profile
    # would put Turbo's 1.15 back on and leave the record unable to reproduce.
    run_one(client, mu=None)
    assert fake.requests[-1].mu is None


def test_explicit_values_win(client, fake):
    run_one(client, steps=3, guidance_scale=7.5, mu=0.5)
    request = fake.requests[-1]
    assert (request.steps, request.guidance_scale, request.mu) == (3, 7.5, 0.5)


# -------------------------------------------------------------------- guards


def test_unload_is_refused_while_a_job_is_live(client, fake):
    fake.step_seconds = 0.02
    job_id = client.post(
        "/api/generate", json={"prompt": "x", "width": 256, "height": 256, "steps": 30}
    ).json()["job_id"]
    response = client.post("/api/model/unload")
    assert response.status_code == 409
    assert fake.loaded  # 200 here used to free nothing and reload mid-sweep

    stream_job(client, job_id)
    assert client.post("/api/model/unload").status_code == 200
    assert not fake.loaded


def test_a_cross_origin_write_is_refused(client):
    refused = client.post("/api/model/unload", headers={"Origin": "http://evil.example"})
    assert refused.status_code == 403
    # The page's own origin is exactly what a same-origin fetch announces.
    assert client.post("/api/model/unload", headers={"Origin": ORIGIN}).status_code == 200
    # Reads are not writes: this middleware is not a CORS policy.
    assert client.get("/api/status", headers={"Origin": "http://evil.example"}).status_code == 200


def test_a_rebound_host_header_is_refused(client):
    assert client.get("/api/status", headers={"Host": "evil.example"}).status_code == 400
    allowed = {"Host": f"localhost:{settings.port}"}
    assert client.get("/api/status", headers=allowed).status_code == 200
