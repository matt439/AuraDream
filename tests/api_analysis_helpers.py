"""Shared fixtures and request helpers for the api analysis tests."""
from __future__ import annotations

from urllib.error import URLError
import asyncio
import dataclasses

from backend import main, tags
from backend.config import settings
from backend.model_defaults import defaults_for
from backend.schemas import PromptAnalysis
from backend.tags import TagLookupUnavailable


FLUX = "black-forest-labs/FLUX.1-dev"


ANIMAGINE = "cagliostrolab/animagine-xl-4.0"


ANIMA = "circlestone-labs/Anima"


FIREFLY = "firefly \\(honkai: star rail\\)"


# Out of Animagine's documented order only by what Danbooru knows: a
# character after a general tag. Its own groups (1girl, the quality suffix)
# are already in place.
TAG_PROMPT = f"1girl, green hair, {FIREFLY}, masterpiece, high score, great score, absurdres"


IN_ORDER = f"1girl, {FIREFLY}, green hair, masterpiece, high score, great score, absurdres"


def claim(monkeypatch, fake, model_id):
    """Serve `model_id`, with the fake applying requests as its profile says
    - on FLUX.1 a negative prompt runs only under True CFG."""
    monkeypatch.setattr(main, "settings", dataclasses.replace(settings, model_id=model_id))
    profile = defaults_for(model_id)
    fake.cfg_threshold = profile["cfg_threshold"]
    fake.true_cfg_supported = profile["true_cfg_supported"]


class Vocabulary:
    """The tag index, stubbed: known categories, and every call recorded."""

    def __init__(self, categories=None, offline=False):
        self.categories = categories or {}
        self.offline = offline
        self.calls: list[tuple] = []

    def validate(self, tags, special, cutoff, allow_year_tag=True):
        self.calls.append((list(tags), set(special), cutoff, allow_year_tag))
        if self.offline:
            # As tags.py raises it: the connection error is the cause.
            raise TagLookupUnavailable("<urlopen error [Errno 11001] getaddrinfo failed>") from (
                URLError(OSError(11001, "getaddrinfo failed"))
            )
        return [
            {"tag": tag, "valid": tag in self.categories, "category": self.categories.get(tag)}
            for tag in tags
        ]


def rules(findings, rule):
    return [finding for finding in findings if finding["rule"] == rule]


def analyse(client, **body):
    response = client.post("/api/prompt/analyze", json={"prompt": "a lighthouse", **body})
    assert response.status_code == 200, response.text
    # The response model is the contract the page reads.
    return PromptAnalysis.model_validate(response.json()).model_dump()


def on_the_loop() -> bool:
    """Whether the calling thread is running an event loop: the app's, for
    blocking work a route called inline instead of handing it off."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True
