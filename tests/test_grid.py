"""Axis parsing and the cartesian expansion behind a grid search."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from backend import grid
from backend.schemas import GenerateRequest, PromptVariant

BASE = dict(prompt="a lighthouse in fog", width=256, height=256, steps=8, guidance_scale=0.0, mu=1.15)


def request(**overrides) -> GenerateRequest:
    return GenerateRequest(**{**BASE, **overrides})


# ------------------------------------------------------------------- parsing


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("8", [8]),
        ("6,8,10", [6, 8, 10]),
        ("6 8 10", [6, 8, 10]),  # spaces are separators too, for pasted lists
        ("6-12x2", [6, 8, 10, 12]),  # both ends included
        ("6-12", [6, 7, 8, 9, 10, 11, 12]),  # a bare range steps by one
        ("6-7,20", [6, 7, 20]),  # ranges and values mix
        ("8,8,6", [8, 6]),  # a repeat is a night spent rendering it twice
        ("", []),  # empty is "leave the axis alone", not "no values"
    ],
)
def test_integer_axes(spec, expected):
    assert grid.parse_axis(spec, int, label="steps") == expected


def test_float_ranges_do_not_drift():
    # 0.1 steps accumulated naively give 0.30000000000000004, which then shows
    # up in the preview, in the record and in the filename.
    assert grid.parse_axis("0-1x0.1") == [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def test_mu_takes_none_and_nothing_else_does():
    assert grid.parse_axis("none,1.15", allow_none=True, label="Mu") == [None, 1.15]
    with pytest.raises(ValueError, match="only meaningful for Mu"):
        grid.parse_axis("none", label="CFG")


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("6 - 12", [6, 7, 8, 9, 10, 11, 12]),
        ("6-12 x2", [6, 8, 10, 12]),
        ("6 - 12 x 2", [6, 8, 10, 12]),
        ("6-12 * 2", [6, 8, 10, 12]),
        ("2 -5", [2, 3, 4, 5]),  # was 2 and -5, which the schema then refused on the CFG box
        ("6 - 8, 20", [6, 7, 8, 20]),
        ("6 8 10", [6, 8, 10]),  # spaces between plain numbers still separate
    ],
)
def test_a_spaced_range_is_one_range(spec, expected):
    # The range regex always allowed the spaces; the split on whitespace before
    # it never let them through, and `-` alone was refused as a none token.
    assert grid.parse_axis(spec, int, label="Steps") == expected


def test_a_spaced_range_on_mu_is_not_a_none_cell():
    # `0.8 - 1.2` on Mu was three cells - 0.8, a `-` that is Mu's none, 1.2 -
    # with nothing said. A `-` with no number on both sides is still none.
    assert grid.parse_axis("1 - 1.2 x 0.1", allow_none=True, label="Mu") == [1.0, 1.1, 1.2]
    assert grid.parse_axis("- 1.15", allow_none=True, label="Mu") == [None, 1.15]
    assert grid.parse_axis("1.15, -", allow_none=True, label="Mu") == [1.15, None]


@pytest.mark.parametrize("spec", ["0.8 - 1.2", "0.8-1.2", "1.15 - 1.3", "3.5-4"])
def test_a_short_range_without_a_step_is_asked_for_one(spec):
    # Closed up, `0.8 - 1.2` is a range - counted in ones, the one value 0.8,
    # which the preview's totals drop as a factor of one. Asked for its step
    # rather than run as a sweep of one.
    with pytest.raises(ValueError, match=r"^Mu: '[0-9.-]+' needs a step, e\.g\. [0-9.]+-[0-9.]+x0\.1$"):
        grid.parse_axis(spec, allow_none=True, label="Mu")


@pytest.mark.parametrize(
    ("spec", "expected"),
    [("0.8-1.2x0.2", [0.8, 1.0, 1.2]), ("1-1", [1]), ("1.5-1.5", [1.5]), ("3.5-5.5", [3.5, 4.5, 5.5])],
)
def test_a_range_without_a_step_counts_in_ones(spec, expected):
    assert grid.parse_axis(spec, label="CFG") == expected


@pytest.mark.parametrize("spec", ["6..12", "6 .. 12", "0.5...2", "6 to 12", "6TO12"])
def test_another_range_syntax_is_answered_with_ours(spec):
    with pytest.raises(ValueError, match=r"^Steps: write a range as 6-12 \(or 6-12x2\)$"):
        grid.parse_axis(spec, float, label="Steps")


def test_each_axis_is_named_after_its_box():
    # The preview prints the message under the whole fold, so it leads with the
    # box to fix - the Refine dialog's Strength among them - as the page says it.
    adapted = request(lora="style", refine=3, strength=0.4)
    boxes = {
        "Steps": dict(steps="abc"),
        "CFG": dict(cfg="abc"),
        "Mu": dict(mu="abc"),
        "LoRA weight": dict(lora_scale="abc"),
        "Strength": dict(strength="abc"),
    }
    for box, axes in boxes.items():
        spec = {"steps": "", "cfg": "", "mu": "", **axes}
        with pytest.raises(ValueError, match=f"^{box}: 'abc' is not a number$"):
            grid.axis_values(adapted, **spec)
    with pytest.raises(ValueError, match="^Seed list: 'abc' is not a number$"):
        grid.parse_seeds("abc")


@pytest.mark.parametrize(
    "spec", ["12-6", "6-12x0", "6-12x-2", "eight", "6..12", "6-12x0.5", "6.5-12", "8.5"]
)
def test_a_bad_axis_says_so(spec):
    # Named after the field, whichever branch it came through: a fractional
    # step on an integer axis used to escape as int()'s own message.
    with pytest.raises(ValueError, match="steps:"):
        grid.parse_axis(spec, int, label="steps")


def test_an_axis_cannot_be_a_thousand_values():
    with pytest.raises(ValueError, match="more than"):
        grid.parse_axis("1-1000", int, label="steps")


# ----------------------------------------------------------------- expansion


def test_an_empty_grid_is_the_request_it_sits_on():
    cells = grid.expand(request(seed=42))
    assert len(cells) == 1
    assert (cells[0].request.steps, cells[0].request.guidance_scale, cells[0].request.mu) == (
        8,
        0.0,
        1.15,
    )
    assert cells[0].seed == 42


def test_seeds_are_the_outer_loop():
    # A grid that runs out of night has then finished every setting for the
    # seeds it reached, rather than a slice of each.
    cells = grid.expand(request(seed=42), steps="6,8", seeds=2)
    assert [(c.seed, c.request.steps) for c in cells] == [(42, 6), (42, 8), (43, 6), (43, 8)]


def test_the_product_covers_every_combination():
    cells = grid.expand(request(seed=1), steps="6,8", cfg="0,1", mu="none,1.15", seeds=3)
    assert len(cells) == 2 * 2 * 2 * 3
    combinations = {(c.request.steps, c.request.guidance_scale, c.request.mu) for c in cells}
    assert combinations == {
        (steps, cfg, mu) for steps in (6, 8) for cfg in (0.0, 1.0) for mu in (None, 1.15)
    }


def test_a_batch_steps_the_seed_past_itself():
    # The same rule a sweep follows: image i of a batch is seed+i, so the next
    # cell has to start past the whole of the previous one or two images in the
    # grid share a seed and the gallery cannot reproduce either.
    cells = grid.expand(request(seed=100, num_images=4), seeds=3)
    assert [c.seed for c in cells] == [100, 104, 108]


def test_a_budget_completes_the_seeds_whose_last_cell_starts_in_time():
    # Measured on the card (docs/verified.md, Kohaku): Steps 12,28,40 x 6
    # seeds at 301 ms a step, a 2 min budget. A seed's share of the estimate
    # (24.1 s) into 120 s said 4 of 6; the job, which starts no cell after the
    # deadline and lets the one in flight finish, made 5 - the fifth seed's
    # last cell began at 108 s.
    cells = grid.expand(request(seed=1), steps="12,28,40", seeds=6)
    assert grid.seeds_within_budget(cells, 6, 301.0, 120) == 5
    assert grid.estimate_seconds(cells, 301.0) == pytest.approx(144.5)
    # The deadline itself still starts a cell: the job stops only once it
    # has passed. A seed's first cell starting at the deadline is not enough.
    flat = grid.expand(request(seed=1, steps=10), seeds=3)
    assert grid.seeds_within_budget(flat, 3, 1000.0, 10) == 2
    assert grid.seeds_within_budget(flat, 3, 1000.0, 9.99) == 1
    assert grid.seeds_within_budget(flat, 3, 1000.0, 0) == 1
    assert grid.seeds_within_budget(flat, 3, 1000.0, 1000) == 3
    # No price, no count.
    assert grid.seeds_within_budget(cells, 6, None, 120) is None


def test_every_cell_is_a_single_run():
    # `runs` would multiply the night behind the count the preview promised.
    cells = grid.expand(request(seed=1, runs=5), steps="6,8")
    assert {c.request.runs for c in cells} == {1}


def test_a_cell_outside_the_schema_is_refused_up_front():
    # Not at 3am, sixty images into the sweep.
    with pytest.raises(ValueError):
        grid.expand(request(seed=1), steps="90-120x10")


def test_a_value_past_the_schema_is_refused_under_its_box():
    # The request's own bounds, read off the schema, said as the axis parser
    # says the rest: the box's name and the value, never pydantic's field name.
    for kwargs, message in (
        ({"steps": "20-40x10, 150"}, "Steps: 150 is above 100, the most it takes"),
        ({"cfg": "25"}, "CFG: 25 is above 20, the most it takes"),
        ({"mu": "none, 12.5"}, "Mu: 12.5 is above 10, the most it takes"),
    ):
        with pytest.raises(ValueError) as exc:
            grid.expand(request(seed=1), **kwargs)
        assert str(exc.value) == message
    with pytest.raises(ValueError, match="^LoRA weight: 3 is above 2, the most it takes$"):
        grid.expand(request(seed=1, lora="style"), lora_scale="1, 3")
    with pytest.raises(ValueError, match="^Seed list: 9223372036854775808 is above 9223372036854775807"):
        grid.parse_seeds("4471, 9223372036854775808")
    # In bounds, nothing changes.
    assert grid.parse_seeds("0, 9223372036854775807") == [0, 9223372036854775807]


def test_the_whole_grid_has_a_ceiling():
    with pytest.raises(ValueError, match="past the"):
        grid.expand(request(seed=1), steps="1-64", cfg="0-16", seeds=8)


def test_labels_stay_ascii():
    # They go into an overnight log, and a redirected stdout on Windows encodes
    # with the locale codec.
    for cell in grid.expand(request(seed=1), steps="6,8", mu="none,1.15"):
        assert cell.label is not None
        cell.label.encode("ascii")


def test_a_seed_list_replaces_the_count_and_the_base():
    # Night two: the two seeds that gave the composition, crossed with a finer
    # sweep. Taken as given - not stepped by the batch, not wrapped.
    cells = grid.expand(request(seed=42, num_images=2), steps="6,8", seed_list=[4471, 8823])
    assert [(c.seed, c.request.steps) for c in cells] == [(4471, 6), (4471, 8), (8823, 6), (8823, 8)]
    assert grid.parse_seeds("4471, 8823") == [4471, 8823]
    assert grid.parse_seeds("100-103") == [100, 101, 102, 103]
    assert grid.parse_seeds("") == []


def test_a_seed_list_counts_towards_the_ceiling():
    # Swapping only the loop would have let a 64-value list queue 64x past it.
    with pytest.raises(ValueError, match="past the"):
        grid.expand(request(seed=1), steps="1-64", cfg="0-16", seed_list=list(range(64)))


def test_a_listed_seed_outside_the_schema_fails_with_the_schemas_message():
    with pytest.raises(ValueError, match="seed"):
        grid.expand(request(seed=1), seed_list=[-5])


# ------------------------------------------------------------- prompt axis

DOG = dict(BASE, prompt="{size} dog sitting on {ground}")
VARS = "size = big | small\nground = grass | dirt"


def dog(**overrides) -> GenerateRequest:
    return GenerateRequest(**{**DOG, **overrides})


def test_variables_split_on_the_pipe_not_the_comma():
    # `a big, fluffy dog` is one prompt fragment. Splitting it the way the
    # numeric axes split would turn a value into two cells of nonsense.
    assert grid.parse_variables("mood = a big, fluffy dog | a wet dog") == {
        "mood": ["a big, fluffy dog", "a wet dog"]
    }


def test_a_variable_can_be_nothing_at_all():
    # Saying nothing about the size is a comparison in its own right, and the
    # prompt has to close over the gap the placeholder left.
    values = grid.parse_variables("size = big | none")
    assert values == {"size": ["big", ""]}
    prompts = [c.request.prompt for c in grid.expand(dog(seed=1), prompt="size = big | none\nground = grass")]
    assert prompts == ["big dog sitting on grass", "dog sitting on grass"]


INDENTED = (
    "<character_1>\n"
    "    <n>miku</n>\n"
    "    <appearance>chibi, red_eyes, {hair} long_hair</appearance>\n"
    "</character_1>\n"
    "<general_tags>\n"
    "    <style>anime  style</style>\n"
    "</general_tags>"
)


def test_a_none_value_takes_its_space_and_nothing_else():
    # An indented prompt - XML as a model card writes it - lost every line's
    # indentation and every doubled space in its none cell, some twenty Gemma
    # tokens the other cells kept: the comparison was no longer the word alone.
    with_value, without = (
        c.request.prompt
        for c in grid.expand(request(prompt=INDENTED, seed=1), prompt="hair = blue_hair | none")
    )
    assert with_value == INDENTED.replace("{hair}", "blue_hair")
    assert without == INDENTED.replace("{hair} ", "")


@pytest.mark.parametrize(
    ("prompt", "filled"),
    [
        ("a {x} dog", "a dog"),  # the space after the word
        ("{x} dog", "dog"),  # at the start, the space after it
        ("a dog {x}", "a dog"),  # at a line's end, the space before it
        ("a dog {x}\non a mat", "a dog\non a mat"),
        ("a {x}\n  on a mat", "a\n  on a mat"),  # the next line's indent stays
        ("  {x} dog\n  on a mat", "  dog\n  on a mat"),  # so does its own
        ("a  {x} dog", "a  dog"),  # a doubled space the user wrote stays
        ("dog, {x}, cat", "dog, , cat"),  # punctuation belongs inside the value
        ("red_eyes, {x}long_hair", "red_eyes, long_hair"),  # no space to take
        ("a {x} {y} dog", "a dog"),  # two holes side by side
        ("<a>{x} fur</a>", "<a>fur</a>"),  # opening an element
        ("({x} big) dog", "(big) dog"),  # or a bracket
        ("<a>\n    {x}\n</a>", "<a>\n\n</a>"),  # alone on its line: the line empties
        ("a {x}\r\nb", "a\r\nb"),
    ],
)
def test_a_hole_closes_where_it_is(prompt, filled):
    assert grid.substitute(prompt, {"x": "", "y": ""}) == filled


def test_a_value_is_filled_in_as_written():
    # Only an empty value closes anything: a filled one leaves the prompt's
    # whitespace exactly as it was, and a name the grid does not set stays.
    assert grid.substitute("  a {x}  {z} dog", {"x": "big"}) == "  a big  {z} dog"


def test_the_prompt_crosses_with_the_sampling_axes():
    cells = grid.expand(dog(seed=1), prompt=VARS, steps="6,8", seeds=2)
    assert len(cells) == 2 * 2 * 2 * 2
    assert {(c.request.prompt, c.request.steps) for c in cells} == {
        (f"{size} dog sitting on {ground}", steps)
        for size in ("big", "small")
        for ground in ("grass", "dirt")
        for steps in (6, 8)
    }


def test_the_prompt_is_the_outer_loop_under_the_seed():
    # Same reason the seed is outermost: a night that stops early has finished
    # every setting for the prompts it reached, rather than a slice of each.
    cells = grid.expand(dog(seed=42), prompt="size = big | small\nground = grass", steps="6,8")
    assert [(c.request.prompt, c.request.steps) for c in cells] == [
        ("big dog sitting on grass", 6),
        ("big dog sitting on grass", 8),
        ("small dog sitting on grass", 6),
        ("small dog sitting on grass", 8),
    ]


def test_every_cell_of_a_prompt_grid_shares_the_seed():
    # The whole point of the comparison: the same noise, a different word.
    cells = grid.expand(dog(seed=42), prompt=VARS)
    assert {c.seed for c in cells} == {42}


def test_the_negative_prompt_takes_the_same_variables():
    cells = grid.expand(
        dog(seed=1, negative_prompt="no {ground}", prompt="a dog on {ground}"),
        prompt="ground = grass | dirt",
    )
    assert [(c.request.prompt, c.request.negative_prompt) for c in cells] == [
        ("a dog on grass", "no grass"),
        ("a dog on dirt", "no dirt"),
    ]


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        ("size big | small", "is not"),  # no `=`
        ("size =", "no values"),  # nothing after it
        ("size = big\nsize = small", "twice"),  # two lines, one name
        ("colour = red | blue", "nothing in the prompt"),  # unused variable
        ("size = big | small", "has no values"),  # {ground} left unfilled
        ("", "has no values"),  # placeholders written, the box left empty
    ],
)
def test_a_prompt_that_does_not_line_up_is_refused(spec, message):
    # Both halves matter: a variable with no placeholder multiplies the night
    # by a value that changes nothing, and a placeholder with no variable
    # renders the literal `{ground}` into every image of the run.
    with pytest.raises(ValueError, match=message):
        grid.expand(dog(seed=1), prompt=spec)


def test_a_prompt_grid_counts_towards_the_ceiling():
    values = " | ".join(str(n) for n in range(40))
    with pytest.raises(ValueError, match="past the"):
        grid.expand(dog(seed=1), prompt=f"size = {values}\nground = {values}")


def test_a_prompt_product_is_refused_before_it_is_built():
    # Twenty binary variables are a million assignments; the refusal has to
    # come from the count, not from having built them.
    names = [f"v{n}" for n in range(20)]
    prompt = " ".join(f"{{{name}}}" for name in names)
    block = "\n".join(f"{name} = a | b" for name in names)
    with pytest.raises(ValueError, match="past the"):
        grid.prompt_assignments(request(prompt=prompt), grid.parse_variables(block))


def test_a_prompt_without_placeholders_needs_no_variables():
    # The other half of the rule: a grid with nothing set is the image Generate
    # would have made, braces or not being the whole question.
    assert grid.prompt_assignments(request(), {}) == [{}]


def test_prompt_labels_survive_a_locale_encoder():
    for cell in grid.expand(dog(seed=1), prompt="size = café | none\nground = grass"):
        assert cell.label is not None
        cell.label.encode("ascii")
    assert "size=none" in grid.expand(
        dog(seed=1), prompt="size = none\nground = grass"
    )[0].label


def test_prompts_of_lists_each_one_once():
    cells = grid.expand(dog(seed=1), prompt=VARS, steps="6,8", seeds=2)
    assert grid.prompts_of(cells) == [
        "big dog sitting on grass",
        "big dog sitting on dirt",
        "small dog sitting on grass",
        "small dog sitting on dirt",
    ]


# ---------------------------------------------------------- prompt variants


def variants(*pairs: tuple[str, str]) -> list[PromptVariant]:
    return [PromptVariant(label=label, prompt=text) for label, text in pairs]


# The analyser's comparison: the prompt as typed, the same words in another
# order, and a rewrite - written in that order, which is not alphabetical.
COMPARE = variants(
    ("original", "a red umbrella held by a woman in a yellow raincoat"),
    ("reordered", "a woman in a yellow raincoat holding a red umbrella"),
    ("rewrite", "A woman in a yellow raincoat holds a red umbrella on a rainy street."),
)


def test_variants_are_the_prompt_axis_in_the_order_written():
    cells = grid.expand(
        request(seed=42, negative_prompt="blurry"), variants=COMPARE, steps="6,8", seeds=2
    )
    # Seeds outermost, the variants just inside them, the sampling axes inside
    # those - the order every other prompt grid runs in.
    assert [(c.seed, c.variables, c.request.steps) for c in cells] == [
        (seed, {"variant": label}, steps)
        for seed in (42, 43)
        for label in ("original", "reordered", "rewrite")
        for steps in (6, 8)
    ]
    # Each cell renders its variant's text beside the request's negative prompt;
    # the request's own prompt is the base of nothing and is not rendered.
    assert {(c.variables["variant"], c.request.prompt) for c in cells} == {
        (v.label, v.prompt) for v in COMPARE
    }
    assert {c.request.negative_prompt for c in cells} == {"blurry"}
    assert grid.prompts_of(cells) == [v.prompt for v in COMPARE]
    # The label leads the log line, as a variable does.
    assert cells[0].label.startswith("variant=original, 6 steps")


def test_a_variant_is_text_not_a_template():
    # NewBie's XML is all newlines and may carry a `|`, a pasted JSON prompt is
    # all braces, and neither is a placeholder or a second cell. The request's
    # prompt here has unfilled placeholders of its own, which is not refused:
    # it is not what runs.
    xml = "<character_1>\n<n>{name}</n>\n<appearance>red hair | green eyes</appearance>\n</character_1>"
    as_json = '{"subject": "a fox", "style": "ink wash"}'
    cells = grid.expand(dog(seed=1), variants=variants(("xml", xml), ("json", as_json)))
    assert [c.request.prompt for c in cells] == [xml, as_json]
    for cell in cells:
        cell.label.encode("ascii")


def test_a_placeholder_in_the_negative_prompt_still_needs_values():
    # Nothing fills it beside variants, so it would reach the encoder as the
    # literal `{avoid}` in every image of the run.
    with pytest.raises(ValueError, match=r"\{avoid\} has no values"):
        grid.expand(request(seed=1, negative_prompt="{avoid}"), variants=COMPARE)


@pytest.mark.parametrize(
    ("spec", "given", "message"),
    [
        # The schema refuses both of these first; the CLI does not go through it.
        ("size = big | small", COMPARE, "leave the variables empty"),
        ("", variants(("a", "a fox"), (" a ", "a dog")), "its own label"),
        # `min_length` passes a label of spaces, which would be a nameless row.
        ("", variants(("a", "a fox"), ("   ", "a dog")), "needs a label"),
    ],
)
def test_variants_that_do_not_line_up_are_refused(spec, given, message):
    with pytest.raises(ValueError, match=message):
        grid.expand(request(seed=1), prompt=spec, variants=given)


def test_labels_are_recorded_trimmed():
    cells = grid.expand(request(seed=1), variants=variants((" original ", "a fox"), ("rewrite", "A fox.")))
    assert [c.variables for c in cells] == [{"variant": "original"}, {"variant": "rewrite"}]


def test_the_axis_values_carry_the_text_and_the_labels():
    values = grid.axis_values(request(), "6,8", "", "", variants=COMPARE)
    assert values["prompt"] == [v.prompt for v in COMPARE]
    assert values["variant"] == ["original", "reordered", "rewrite"]
    assert grid.describe(values, 4) == "3 variants x 2 steps x 1 CFG x 1 mu x 4 seeds"
    # A variable grid reads exactly as it did.
    plain = grid.axis_values(dog(), "", "", "", VARS)
    assert "variant" not in plain
    assert grid.describe(plain, 1).startswith("4 prompts x")


def test_a_variant_grid_counts_towards_the_ceiling():
    eight = variants(*[(f"v{n}", f"a fox, take {n}") for n in range(8)])
    with pytest.raises(ValueError, match="past the"):
        grid.expand(request(seed=1), variants=eight, steps="1-64", seeds=4)


def test_variants_are_keyed_on_their_text_not_their_label():
    # The label is a caption; the cell is what runs. Renaming the variants is
    # the same run and the same resume, and editing one's text is another run.
    cells = grid.expand(request(seed=7), variants=COMPARE, seeds=2)
    renamed = grid.expand(
        request(seed=7), variants=variants(*[(f"v{n}", v.prompt) for n, v in enumerate(COMPARE)]), seeds=2
    )
    edited = grid.expand(
        request(seed=7), variants=[*COMPARE[:2], PromptVariant(label="rewrite", prompt="A fox.")], seeds=2
    )
    assert grid.run_id(cells, "m", "fp8") == grid.run_id(renamed, "m", "fp8")
    assert grid.run_id(cells, "m", "fp8") != grid.run_id(edited, "m", "fp8")
    assert grid.cell_key(cells[0])[0] == COMPARE[0].prompt
    # Resume subtracts the variant the gallery already holds and nothing else.
    assert grid.prune_done(cells, {grid.cell_key(cells[0])}) == cells[1:]


def test_two_variants_with_one_text_are_counted_not_dropped():
    # A reorder that came back as the prompt it started from is one picture
    # twice. Like any other duplicate the preview says so and the run renders
    # what was asked - dropping it would leave the sheet a row short.
    same = grid.expand(
        request(seed=1),
        variants=variants(("original", "a fox"), ("reordered", "a fox"), ("rewrite", "A fox.")),
    )
    assert len(same) == 3
    assert grid.duplicate_count(same, lambda req: req) == 1


def test_the_sheet_knows_the_variable_the_labels_ride_under():
    # `chooseAxes` gives the variant axis the rows by its name alone - widest
    # first put 3 variants x 3 seeds in columns - and web/ has no test suite,
    # so the one thing that can drift unseen is the spelling on each side.
    sheet = (Path(__file__).resolve().parents[1] / "web" / "sheet.js").read_text(encoding="utf-8")
    assert f"const VARIANT_AXIS = 'var:{grid.VARIANT}';" in sheet


# ------------------------------------------------------------------- run id


def test_a_run_is_named_by_its_definition_not_its_job():
    # The same grid spelled two ways is one run; a resume that recomputes the id
    # from the full grid lands on the id the first job used.
    full = grid.expand(request(seed=42), steps="6,8,10", seeds=2)
    spelled = grid.expand(request(seed=42), steps="10,6-8x2", seeds=2)
    assert grid.run_id(full, "m", "fp8") == grid.run_id(spelled, "m", "fp8")
    # Anything that changes what the cells become is a different run.
    assert grid.run_id(full, "m", "fp8") != grid.run_id(full, "other", "fp8")
    assert grid.run_id(full, "m", "fp8") != grid.run_id(full, "m", "int8")
    assert grid.run_id(full, "m", "fp8") != grid.run_id(
        grid.expand(request(seed=43), steps="6,8,10", seeds=2), "m", "fp8"
    )
    assert grid.run_id(full, "m", "fp8") != grid.run_id(
        grid.expand(request(seed=42, width=512), steps="6,8,10", seeds=2), "m", "fp8"
    )


def test_a_grid_without_true_cfg_keeps_the_id_its_images_carry():
    # Pinned before the field existed: the id every image of every night so
    # far was written under. A cell that asks for no true CFG has to keep it,
    # or the `run:` filter splits a resumed night in two at the upgrade.
    cells = grid.expand(
        GenerateRequest(
            prompt="a {size} fox", negative_prompt="blurry", width=256, height=256, steps=8,
            guidance_scale=2.0, mu=1.15, seed=42,
        ),
        prompt="size = big | small", steps="6,8", seeds=2,
    )
    assert grid.run_id(cells, "krea/Krea-2-Turbo", "fp8") == "0ebe5101c492"
    assert grid.cell_key(cells[0]) == (
        "a big fox", "blurry", 6, 2.0, 1.15, None, None, None, None, None, 42
    )
    # Setting it is a different run - and the element is part of the key.
    with_it = grid.expand(
        GenerateRequest(
            prompt="a {size} fox", negative_prompt="blurry", width=256, height=256, steps=8,
            guidance_scale=2.0, true_cfg=1.5, mu=1.15, seed=42,
        ),
        prompt="size = big | small", steps="6,8", seeds=2,
    )
    assert grid.run_id(with_it, "krea/Krea-2-Turbo", "fp8") != "0ebe5101c492"
    assert grid.cell_key(with_it[0])[5] == 1.5


def test_duplicates_are_counted_as_the_pipeline_would_run_them():
    # A variable that lives in the negative prompt alone is a legal axis; on a
    # checkpoint that takes no negative prompt those cells are one picture.
    cells = grid.expand(
        GenerateRequest(
            prompt="a fox", negative_prompt="{avoid}", width=256, height=256, steps=8,
            guidance_scale=2.0, seed=1,
        ),
        prompt="avoid = blurry | dark | noisy", steps="6,8",
    )
    assert len(cells) == 6
    as_spelled = lambda req: req  # noqa: E731
    without_negative = lambda req: req.applied(negative_prompt_supported=False, cfg_threshold=1.0)  # noqa: E731
    assert grid.duplicate_count(cells, as_spelled) == 0
    assert grid.duplicate_count(cells, without_negative) == 4  # two step counts survive
    # Counted per image of a batch, like resume.
    batched = grid.expand(
        GenerateRequest(
            prompt="a fox", negative_prompt="{avoid}", width=256, height=256, steps=8,
            guidance_scale=2.0, seed=1, num_images=2,
        ),
        prompt="avoid = blurry | dark",
    )
    assert grid.duplicate_count(batched, without_negative) == 2


def test_a_batch_cell_is_not_done_until_every_image_is():
    # Rows are per image; the base seed alone used to mark a four-image cell
    # rendered the moment its first image committed.
    cells = grid.expand(request(seed=1000, num_images=4), steps="8,10", seeds=2)
    assert [c.seed for c in cells] == [1000, 1000, 1004, 1004]
    first_only = {grid.cell_key(cells[0])}
    assert grid.prune_done(cells, first_only) == cells
    whole = set(grid.cell_keys(cells[0]))
    assert [key[-1] for key in grid.cell_keys(cells[0])] == [1000, 1001, 1002, 1003]
    assert grid.prune_done(cells, whole) == cells[1:]


def test_resume_tells_two_prompts_apart():
    # The key carries the prompt, so finishing one prompt's cells does not
    # cancel the identical settings under another.
    cells = grid.expand(dog(seed=1), prompt=VARS)
    done = {grid.cell_key(cells[0])}
    kept = grid.prune_done(cells, done)
    assert len(kept) == len(cells) - 1
    assert cells[0] not in kept


def test_the_page_reads_a_placeholder_and_a_none_as_the_grid_does():
    # Generate's and the comparison's guards and the tag tray's grid-variable
    # chip read a `{name}` by the page's copy of the pattern, and the chip's
    # tip leaves the deliberately empty values unasked by its copy of these.
    web = Path(__file__).resolve().parents[1] / "web"
    generation = (web / "generation.js").read_text(encoding="utf-8")
    match = re.search(r"^const GRID_PLACEHOLDER = /(.+)/;$", generation, re.M)
    unnamed = grid.PLACEHOLDER.pattern.replace("(", "").replace(")", "")
    assert match and match.group(1).replace(r"\w", "[A-Za-z0-9_]") == unnamed
    chips = (web / "tag-chips.js").read_text(encoding="utf-8")
    match = re.search(r"^const NONE_VALUES = new Set\(\[(.+)\]\);$", chips, re.M)
    assert match and set(re.findall(r"'([^']*)'", match.group(1))) == grid.NONE_TOKENS
