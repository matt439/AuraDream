"""Layer 1 of the prompt check: blocking, CPU only, no HTTP.

The encoder reports and the rule findings over one request, why a placed
order was only partly checked, the chips' order slots, and the lint of the
critic's rewrite. The routes (`api_prompt`) run these off the loop.
"""
from __future__ import annotations

import logging

from . import prompt_encoding, prompt_rules, tag_lookup
from .jobs import short_error
from .prompt_sources import author_orders
from .schemas import (
    AuthorOrder,
    EncoderReport,
    GenerateRequest,
    PromptAnalysis,
    PromptAnalysisRequest,
    PromptFinding,
    TagSlot,
)

log = logging.getLogger("local_diffusion")


def order_slots(model_id: str, prompt: str | None, reachable: bool) -> dict:
    """The chips' order slots: each comma item of `prompt` in the step of the
    documented order Check's order rule places it in, with the categories a
    Check of this prompt places with (`tag_lookup.tag_categories`) - Check's reading,
    not one of the page's. Empty where no prompt was sent or the rules place
    nothing in it (`places_items`, asked first, so such a prompt costs the
    tag index nothing). A lone tag is placed too, though no order rule reads
    one: the Tags helper's caret line says from its step where the next tag
    goes, and that is the moment the helper is most used.

    The categories come from the cache the chips' lookup has just filled,
    except a name past the chips' 64 that Check's own 64 reach (guide tags
    are not asked about): that one is fetched here, once, as a Check would.
    `reachable` False: the chips' own lookup could not reach the index, so a
    Check's would not either, and its categories are None
    (`tag_lookup.tag_categories`' failure path) - asked again here, it would wait out
    the same failure a second time.

    Advisory, like the lookup: a failure here costs the slots, never the
    chips' verdicts it is answered beside.
    """
    if prompt is None:
        return {}
    try:
        order = prompt_rules.documented_order(model_id)
        if order is None or not prompt_rules.places_items(model_id, prompt):
            return {}
        categories = tag_lookup.tag_categories(model_id, prompt)[0] if reachable else None
        slots = prompt_rules.order_slots(model_id, prompt, categories)
        if slots is None:
            return {}
        return {
            "order_steps": order.steps,
            "slots": [TagSlot(tag=tag, slot=slot) for tag, slot in slots],
        }
    except Exception as exc:  # noqa: BLE001 - advisory: the verdicts still go out
        log.warning("chip order slots failed: %s", short_error(exc))
        log.debug("chip order slots traceback", exc_info=True)
        return {}


def order_detail(
    model_id: str,
    prompt: str,
    categories: dict[str, str] | None,
    categories_detail: str | None,
    asked: list[str],
) -> str | None:
    """Why a placed order was only partly checked, or None where the rules
    decided it whatever category each item has (`unplaced_items` empty, with
    the categories they placed with: `places_every_item`, the predicate the
    critic reads for a reorder of its own) - a lone item, or one every slot
    of which sits in order, or out of it, beside the rest, included.

    Each unplaced item under the reason it has no category: the lookup's
    own (`categories_detail`: unreachable, no usable reply, past the cap), a
    tag the index answered for with none it could use - not found, and
    deprecated, unused or past the checkpoint's cutoff alike, the chips'
    reasons, so the words claim none of them - a prose prompt's clause -
    prose is never looked up, and only a whole sentence is placed by its
    shape - or an item never asked about at all (lettering, an @ name).
    Every such item, not only those the lookup lost: said of those alone, an
    unknown tag or a prose clause left the order drawn as checked while the
    server treated it as partly placed.
    """
    unplaced = list(dict.fromkeys(prompt_rules.unplaced_items(model_id, prompt, categories)))
    if not unplaced:
        return None
    wanted = set(prompt_rules.category_items(model_id, prompt))
    asked_set = set(asked)
    prose = prompt_rules.detect_style(prompt) not in ("tags", "hybrid")
    lost: list[str] = []
    unknown: list[str] = []
    clauses: list[str] = []
    others: list[str] = []
    for text in unplaced:
        if asked and text in wanted:
            # Asked and given no usable category - or never answered: the
            # lookup failed, or stopped at the cap before it.
            (unknown if categories is not None and text in asked_set else lost).append(text)
        elif text in wanted and prose:
            clauses.append(text)
        else:
            others.append(text)

    def say(texts: list[str], one: str, many: str) -> str:
        # The count, the words for one or many, and the first three items.
        shown = ", ".join(f'"{text}"' for text in texts[:3])
        more = f" and {len(texts) - 3} more" if len(texts) > 3 else ""
        return (one if len(texts) == 1 else many).replace("{n}", str(len(texts))) + (
            f": {shown}{more}"
        )

    reasons = [
        say(texts, one, many)
        for texts, one, many in (
            (lost, f"{categories_detail}, and {{n}} tag is not placed",
             f"{categories_detail}, and {{n}} tags are not placed"),
            (unknown,
             "the tag index has no usable Danbooru category for {n} tag, so it is not placed",
             "the tag index has no usable Danbooru category for {n} tags, so they are not "
             "placed"),
            (clauses,
             "the tag index is not asked about a prose prompt, so {n} clause that is not a "
             "whole sentence is not placed",
             "the tag index is not asked about a prose prompt, so {n} clauses that are not "
             "whole sentences are not placed"),
            (others, "the tag index is not asked about {n} item, so it is not placed",
             "the tag index is not asked about {n} items, so they are not placed"),
        )
        if texts
    ]
    return "; ".join(reasons)


def finding_order(finding: PromptFinding) -> tuple:
    # Warnings first, the prompt's before the negative prompt's, then in
    # reading order - the rules' own sort, kept across the two layers.
    return (
        finding.severity != "warn",
        finding.field != "prompt",
        finding.start is None,
        finding.start or 0,
    )


def checked(
    model_id: str,
    request: GenerateRequest,
    applied: GenerateRequest,
    categories: dict[str, str] | None,
) -> tuple[list[EncoderReport], list[PromptFinding], str | None]:
    """Layer 1 over one request: the encoder reports, every finding, and why
    any encoder went unchecked.

    The negative prompt is measured only as `applied` runs it: one the
    pipeline drops never reaches an encoder, and a budget reported for it
    would be a number about nothing.
    """
    reports, encoder_findings, detail = prompt_encoding.measure(
        model_id, request.prompt, applied.negative_prompt
    )
    rule_findings = prompt_rules.check(model_id, request, applied, categories)
    # One span, one finding: a bracketed slot standing as a whole comma item
    # ("[character], soft rim lighting") is the author's template slot to the
    # rules and A1111's de-emphasis to the encoder layer, whose own message
    # already allows it may be a slot. The author's reading says what to do.
    slots = {
        (finding.field, finding.start, finding.end)
        for finding in rule_findings if finding.rule == "author.placeholder"
    }
    encoder_findings = [
        finding for finding in encoder_findings
        if not (
            finding.rule == "encoder.weighting"
            and (finding.field, finding.start, finding.end) in slots
        )
    ]
    findings = rule_findings + encoder_findings
    return reports, sorted(findings, key=finding_order), detail


def analyse(
    model_id: str, request: PromptAnalysisRequest, applied: GenerateRequest
) -> PromptAnalysis:
    """The prompt against the running checkpoint's guide and tokenizers.

    Blocking - a first use can fetch a tokenizer, and a tag checkpoint asks
    the tag index - so the routes run it off the loop. Never the image
    model: `applied()` answers from the profile before a load, and the
    tokenizers are read on the CPU.
    """
    prompt = request.prompt
    # The shape alone decides whether to ask the tag index; what is reported
    # is the style the rules applied, which the answer's categories can move
    # ("the legend of zelda" is a series, not a phrase).
    categories, categories_detail, asked = tag_lookup.tag_categories(model_id, prompt)
    reports, findings, detail = checked(model_id, request, applied, categories)
    order = prompt_rules.documented_order(model_id)
    order_checked = order is not None and prompt_rules.order_checked(model_id, prompt)
    return PromptAnalysis(
        model_id=model_id,
        style=prompt_rules.detect_style(prompt, categories),
        words=prompt_rules.word_count(prompt),
        characters=prompt_rules.cjk_count(prompt),
        encoders=reports,
        encoders_detail=detail,
        findings=findings,
        order=order,
        order_checked=order_checked,
        categories_detail=categories_detail,
        order_detail=order_detail(model_id, prompt, categories, categories_detail, asked)
        if order_checked else None,
        categories=categories,
        categories_asked=asked,
        # The Tags menu shows the same list (`guide_for`'s tag_helper).
        author_orders=[AuthorOrder(**item) for item in author_orders(model_id)],
        reordered=prompt_rules.reorder_tags(model_id, prompt, categories),
        reordered_detail=prompt_rules.reordered_detail(model_id, prompt, categories),
    )


def lint(
    model_id: str,
    request: GenerateRequest,
    applied: GenerateRequest,
    text: str,
    analysis: PromptAnalysis,
) -> tuple[list[PromptFinding], bool]:
    """Layer 1's findings for `text` in place of the prompt - the critic's
    rewrite, held to the same rules as what the user typed - and whether its
    artist check read every artist it may name (`prompt_rules.artists_read`,
    `PromptCritique.rewrite_artists_checked`).

    Blocking, and like `analyse` it can fetch a tokenizer, so the route runs
    it off the loop and after the job gate. No tag lookup: the analysis asked
    the tag index once for this critique. Its artists are handed on where the
    guide's rules read them (Anima's @), so a rewrite that adds the @ is seen
    to, and one that drops it is flagged; the rest of its categories are not,
    and the order check places only what the guide's groups know, since the
    rewrite's new tags have no category - the page keeps its "may still
    apply" there.
    """
    artists = None
    if prompt_rules.wants_artist_categories(model_id) and analysis.categories is not None:
        artists = {
            name: category for name, category in analysis.categories.items()
            if category == "artist"
        }
    update = {"prompt": text}
    findings = checked(
        model_id, request.model_copy(update=update), applied.model_copy(update=update), artists
    )[1]
    covered = artists is not None and prompt_rules.artists_read(
        model_id, text, analysis.categories_asked, artists
    )
    return findings, covered
