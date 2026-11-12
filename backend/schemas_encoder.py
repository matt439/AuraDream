"""What a text encoder reads of a prompt (`prompt_encoding.measure`'s
reports), and the grid preview's budgets of each resolved prompt (PA#15).

A module of their own because both Check (`schemas_prompt`, which re-exports
`EncoderReport`) and the grid preview (`schemas_generate.GridPreview`) answer
with them, and `schemas_prompt` imports `schemas_generate`.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class EncoderReport(BaseModel):
    """How much of one field one text encoder reads.

    `tokens` counts the text's own tokens, the pipeline's template excluded;
    `budget` is how many of them survive the pipeline's truncation. `cut` is
    the character offset where reading stops - None when everything fits -
    because every pipeline here cuts from the right in silence.
    """

    name: str
    field: Literal["prompt", "negative_prompt"] = "prompt"
    tokens: int
    budget: int
    cut: int | None = None
    # What the encoder contributes, where that changes what the count means -
    # FLUX.1's CLIP-L feeds one pooled vector, not the token sequence.
    role: str | None = None
    # A cut here is advice, not lost text: this encoder feeds one pooled vector
    # while another reads the words on (FLUX.1's CLIP-L). Its `encoder.cut` is
    # info rather than a warning; every other role's cut is a real loss.
    pooled_only: bool = False
    # The tokenizer the count came from, repo and subfolder, for the tooltip.
    source: str | None = None


class GridPromptBudget(BaseModel):
    """One distinct resolved prompt of a grid (with its negative as
    `applied()` runs it), measured as Check measures the prompt box."""

    # The values that made it - `size=big, ground=dirt` - or a variant's label.
    label: str | None = None
    variables: dict[str, str] | None = None
    encoders: list[EncoderReport] = Field(default_factory=list)
    # Some encoder that reads the words on cuts it (a pooled-only cut is not).
    cut: bool = False


class GridPromptBudgets(BaseModel):
    """The grid's resolved prompts' budgets: the first `MAX_MEASURED`
    distinct ones in render order (`grid_budgets.measure_cells`)."""

    prompts: list[GridPromptBudget] = Field(default_factory=list)
    # Distinct resolved (prompt, negative) pairs in the whole grid; more than
    # `len(prompts)` where the preview stopped measuring.
    distinct: int = 0
    # What went unmeasured - a tokenizer still loading, a gated repo - so the
    # page says the counts are unknown, never that nothing is cut.
    detail: str | None = None
    # Each variable's values whose every measured prompt loses text while
    # another prompt does not: the values that tip a prompt over.
    cutting_values: dict[str, list[str]] = Field(default_factory=dict)
