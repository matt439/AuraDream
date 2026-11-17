"""Pinned optional writing models; inspection never downloads or loads weights.

The shared critic keeps its original default. These profiles describe the
assistant's opt-in alternatives, not measured latency, memory or writing quality.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from . import critic_runtime


@dataclass(frozen=True)
class AssistantModel:
    key: str
    label: str
    repo_id: str
    filename: str
    revision: str
    size: int
    sha256: str
    enable_thinking: bool
    reasoning_budget: int
    temperature: float
    top_p: float
    # This is the app's bounded window, not the checkpoint's maximum context.
    context: int = 8192
    top_k: int = 20
    min_p: float = 0.0
    presence_penalty: float = 1.5

    @property
    def source(self) -> critic_runtime.ModelSource:
        return {"repo_id": self.repo_id, "filename": self.filename, "revision": self.revision}

    @property
    def sampling(self) -> dict[str, float | int]:
        # llama.cpp's key is repeat_penalty; the Qwen3.5 card calls it
        # repetition_penalty. 1.0 leaves repetition reweighting off.
        return {
            "temperature": self.temperature, "top_p": self.top_p, "top_k": self.top_k,
            "min_p": self.min_p, "presence_penalty": self.presence_penalty, "repeat_penalty": 1.0,
        }

    def cached(self) -> Path | None:
        """The exact revision's existing file, without hashing or networking."""
        return critic_runtime._model_cached(self.source)

    def download(self) -> Path:
        """Fetch only this GGUF and verify its pinned size and SHA256."""
        return critic_runtime.download_model(self.source, size=self.size, sha256=self.sha256)


# HF model API metadata (blobs=true) checked 2026-11-17, without downloading
# model bytes. Qwen publishes the first; llama.cpp's maintainers publish the
# second. Qwen3.5 needs only its main text GGUF here, not mmproj or MTP files.
# https://huggingface.co/api/models/Qwen/Qwen3-8B-GGUF?blobs=true
# https://huggingface.co/api/models/ggml-org/Qwen3.5-4B-GGUF?blobs=true
# Source-level architecture support exists in our unchanged llama.cpp b11206:
# https://github.com/ggml-org/llama.cpp/blob/b11206/src/models/qwen35.cpp
# This establishes source support, not a real-process or GPU qualification.
MODELS = (
    AssistantModel(
        key="qwen3-8b", label="Qwen3-8B Q4_K_M",
        repo_id="Qwen/Qwen3-8B-GGUF", filename="Qwen3-8B-Q4_K_M.gguf",
        revision="7c41481f57cb95916b40956ab2f0b139b296d974", size=5027783488,
        sha256="d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785",
        # Author's thinking-mode recipe; the bounded reasoning budget is the
        # app's, not a claim that the author recommends short reasoning.
        # https://huggingface.co/Qwen/Qwen3-8B-GGUF#best-practices
        enable_thinking=True, reasoning_budget=2048, temperature=0.6, top_p=0.95,
    ),
    AssistantModel(
        key="qwen3.5-4b", label="Qwen3.5-4B Q8_0",
        repo_id="ggml-org/Qwen3.5-4B-GGUF", filename="Qwen3.5-4B-Q8_0.gguf",
        revision="8d43d974efc48399cb58957ad49c8b26bf5a109c", size=4482403264,
        sha256="1b19cef95f45262988c3a82319cc0018022ef4a930ee11182425023587cd2ca5",
        # Author's general-task instruct recipe and explicit thinking switch.
        # https://huggingface.co/Qwen/Qwen3.5-4B#instruct-or-non-thinking-mode
        enable_thinking=False, reasoning_budget=0, temperature=0.7, top_p=0.8,
    ),
)


def get_model(key: str) -> AssistantModel:
    """Resolve a catalogue id; never interpret client text as a model path."""
    for model in MODELS:
        if model.key == key:
            return model
    raise ValueError(f"unknown assistant model: {key}")
