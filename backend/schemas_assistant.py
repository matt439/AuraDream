"""Bounded, client-owned conversations with the local prompt assistant."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


INPUT_MAX_CHARS = 12000
OUTPUT_MAX_CHARS = 16000


class AssistantMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=6000)


class PromptAssistantRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ticket: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    assistant_model: Literal["default", "qwen3-8b", "qwen3.5-4b"] = "default"
    mode: Literal["create", "improve", "tags"]
    instruction: str = Field(min_length=1, max_length=2000)
    prompt: str = Field(default="", max_length=4000)
    negative_prompt: str | None = Field(default=None, max_length=4000)
    history: list[AssistantMessage] = Field(default_factory=list, max_length=8)
    requirements: list[str] = Field(default_factory=list, max_length=8)
    brief: str = Field(default="", max_length=2000)
    focus_prompt: str | None = Field(default=None, min_length=1, max_length=4000)

    @field_validator("instruction")
    @classmethod
    def _instruction_has_words(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("describe what you want the assistant to do")
        return value.strip()

    @field_validator("requirements")
    @classmethod
    def _bounded_requirements(cls, values: list[str]) -> list[str]:
        if any(not value.strip() or len(value) > 200 for value in values):
            raise ValueError("each pinned requirement must contain 1 to 200 characters")
        return values

    @field_validator("focus_prompt")
    @classmethod
    def _focus_has_words(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("choose a nonempty prompt option to refine")
        return value

    @model_validator(mode="after")
    def _bounded_conversation(self) -> PromptAssistantRequest:
        if self.mode == "improve" and not (self.focus_prompt or self.prompt).strip():
            raise ValueError("enter a prompt before asking to improve it")
        size = len(self.instruction) + len(self.prompt) + len(self.negative_prompt or "")
        size += sum(len(message.content) for message in self.history)
        size += len(self.brief) + len(self.focus_prompt or "") + sum(map(len, self.requirements))
        if size > INPUT_MAX_CHARS:
            raise ValueError(
                f"the conversation and current text exceed {INPUT_MAX_CHARS} characters; "
                "start a new conversation or shorten the text"
            )
        return self


class AssistantDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1, max_length=100)
    prompt: str = Field(min_length=1, max_length=4000)
    notes: list[str] = Field(default_factory=list, max_length=8)

    @field_validator("notes")
    @classmethod
    def _bounded_notes(cls, values: list[str]) -> list[str]:
        if any(len(value) > 500 for value in values):
            raise ValueError("a draft note is too long")
        return values


TagStatus = Literal[
    "valid", "alias", "not_found", "deprecated", "empty", "no_date", "after_cutoff",
    "grid_variable", "unchecked",
]


class AssistantTag(BaseModel):
    text: str = Field(min_length=1, max_length=200)
    # A model explanation, never passed off as a dictionary definition.
    reason: str = Field(default="", max_length=500)
    status: TagStatus = "unchecked"
    canonical: str | None = None
    detail: str | None = None


class AssistantDraftChanges(BaseModel):
    """Server-computed exact tag-name changes, not model claims or verdicts."""

    model_config = ConfigDict(extra="forbid")

    added: list[str] = Field(default_factory=list)
    removed: list[str] = Field(default_factory=list)


class AssistantTextSegment(BaseModel):
    """Literal word-diff text; clients render this as text, never HTML."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["equal", "added", "removed"]
    text: str = Field(min_length=1, max_length=4000)


class AssistantProseChanges(BaseModel):
    """Lossless comparison computed by the server, outside the model schema."""

    model_config = ConfigDict(extra="forbid")

    segments: list[AssistantTextSegment] = Field(default_factory=list)
    coarse: bool = False


class AssistantRequirementCheck(BaseModel):
    """Exact phrase presence, never a semantic or generated-image guarantee."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=200)
    matched: bool


class PromptAssistantResponse(BaseModel):
    model_id: str
    model: str
    seconds: float = Field(ge=0)
    message: str = Field(max_length=2000)
    drafts: list[AssistantDraft] = Field(default_factory=list, max_length=3)
    # Aligned with drafts; None means the existing prompt reader cannot safely
    # compare the text as tags. Empty keeps older/fake responses compatible.
    draft_changes: list[AssistantDraftChanges | None] = Field(default_factory=list, max_length=3)
    prose_changes: list[AssistantProseChanges | None] = Field(default_factory=list, max_length=3)
    requirement_checks: list[list[AssistantRequirementCheck]] = Field(default_factory=list, max_length=3)
    tags: list[AssistantTag] = Field(default_factory=list, max_length=20)
    warnings: list[str] = Field(default_factory=list)
