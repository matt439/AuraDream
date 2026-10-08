"""The ? text, the Tags menu and Check say one tag order.

Three places describe where a tag goes on a tag checkpoint: the Prompt ? badge
(the guide's `prompt` prose), the Tags menu (`tag_helper`, served by
`guide_for`) and Check's Order section (`PromptAnalysis.order` /
`author_orders`). They drifted - an empty "Recommended order:" on NoobAI and
RouWei beside a ? text quoting the authors' orders and a Check saying the
guide states none, Animagine's steps named nowhere in its ? text, Illustrious
advising "put the quality tags early" against its own order, NoobAI's two
"Year" headings beside a ? text calling them date tags. These hold them to
one source.
"""
from __future__ import annotations

from backend import launch, main
from backend.model_defaults import _profile_for, guide_for
from backend.prompt_order import documented_order
from backend.prompt_sources import author_orders
from backend.schemas import PromptGuide
from tests.api_analysis_helpers import Vocabulary, analyse, claim


def _tag_repos() -> list[str]:
    """One catalogue repo per guide that has a Tags menu."""
    seen: set[int] = set()
    repos = []
    for checkpoint in launch.CHECKPOINTS:
        guide = _profile_for(checkpoint.repo)["guide"]
        if guide.get("tag_helper") and id(guide) not in seen:
            seen.add(id(guide))
            repos.append(checkpoint.repo)
    return repos


def test_the_catalogue_has_tag_menus_with_and_without_an_order():
    ordered = [repo for repo in _tag_repos() if documented_order(repo)]
    unordered = [repo for repo in _tag_repos() if not documented_order(repo)]
    assert len(ordered) >= 4 and len(unordered) >= 2


def test_the_tags_menu_serves_the_order_check_shows():
    for repo in _tag_repos():
        helper = PromptGuide(**guide_for(repo)).tag_helper
        order = documented_order(repo)
        assert helper.order == (order.steps if order else []), repo
        assert helper.order_cite == (order.cite if order else None), repo
        assert [item.model_dump() for item in helper.author_orders] == author_orders(repo), repo
        # A menu with no order to show still says what the authors give: an
        # empty "Recommended order:" was the menu's line on NoobAI and RouWei.
        assert order or helper.author_orders, repo


def test_the_guides_own_menu_order_is_the_analysis_order():
    # guides.py keeps the steps once, in `tag_helper.order`, which
    # `prompt_order` reads; an `analysis.order.steps` beside it may not differ.
    for repo in _tag_repos():
        guide = _profile_for(repo)["guide"]
        steps = (guide.get("analysis") or {}).get("order", {}).get("steps")
        if steps is not None:
            assert steps == guide["tag_helper"]["order"], repo


def test_every_menu_group_is_placed_in_a_step_and_says_which():
    for repo in _tag_repos():
        if not documented_order(repo):
            continue
        guide = _profile_for(repo)["guide"]
        placed = guide["analysis"]["order"]["groups"]
        helper = PromptGuide(**guide_for(repo)).tag_helper
        for group in helper.groups:
            # Every group the menu offers goes somewhere in the order Check
            # holds it to, and the menu is told where.
            assert group.label in placed, (repo, group.label)
            assert helper.group_steps[group.label] == placed[group.label]
            assert placed[group.label] in helper.order
        if helper.year_tag:
            assert helper.year_step in helper.order, repo


def test_the_menus_headings_are_distinct():
    for repo in _tag_repos():
        helper = PromptGuide(**guide_for(repo)).tag_helper
        headings = [group.label for group in helper.groups] + (["Year"] if helper.year_tag else [])
        assert len(headings) == len(set(headings)), (repo, headings)


def test_the_prompt_help_names_every_step_of_the_order():
    # The ? text is where the order is explained; a step the menu and Check
    # name that it never mentions ("Details") is a word the user cannot
    # place. Kohaku's "General tags / prose" is its "general tags or prose".
    for repo in _tag_repos():
        order = documented_order(repo)
        if not order:
            continue
        prose = _profile_for(repo)["guide"]["prompt"].lower()
        for step in order.steps:
            assert step.lower().replace(" / ", " or ") in prose, (repo, step)


def test_the_prompt_help_never_moves_a_step_against_the_order():
    # Illustrious's ? text said "put the quality tags early" beside an order
    # with quality sixth of seven, which Check then flagged.
    for repo in _tag_repos():
        if documented_order(repo):
            prose = _profile_for(repo)["guide"]["prompt"].lower()
            assert "quality tags early" not in prose, repo


def test_check_and_the_menu_answer_the_same_order(client, fake, monkeypatch):
    monkeypatch.setattr(main, "tag_vocabulary", Vocabulary())
    for repo in _tag_repos():
        claim(monkeypatch, fake, repo)
        body = analyse(client, prompt="1girl, solo, masterpiece")
        helper = PromptGuide(**guide_for(repo)).tag_helper
        assert (body["order"] or {}).get("steps", []) == helper.order, repo
        assert body["author_orders"] == [item.model_dump() for item in helper.author_orders], repo
