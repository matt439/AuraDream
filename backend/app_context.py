"""The service context the route modules read the app's state through.

main.py owns the lifecycle state - the backend, the job worker, the critic,
the translator, the load lock and the merged defaults - and the tests and
the fake server rebind those names on main. Every field here is a
zero-argument getter (or, for `merged_defaults`, a function) that main
builds over its own globals and that is read when a request arrives, so a
rebinding is seen. A route module holds this object and never imports main.
"""
from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .model_selection import current_model_id, current_settings

if TYPE_CHECKING:
    from .backends import GenerationBackend
    from .config import Settings
    from .jobs import JobManager


@dataclass(frozen=True)
class AppServices:
    settings: Callable[[], Settings]
    backend: Callable[[], GenerationBackend]
    jobs: Callable[[], JobManager]
    critic: Callable[[], Any]
    translator: Callable[[], Any]
    ensure_loaded: Callable[[], Awaitable[None]]
    # Whether a load holds the load lock (`_load_lock.locked()`).
    loading: Callable[[], bool]
    merged_defaults: Callable[[dict], dict]

    def model_id(self) -> str:
        """The selected checkpoint: `current_model_id(settings)`."""
        return current_model_id(self.settings())

    def quantization(self) -> str:
        """The selected quantization: `current_settings(settings).quantization`."""
        return current_settings(self.settings()).quantization

    def model_scope(self, all_models: bool) -> str | None:
        """The gallery's checkpoint scope: the configured id, or None for every one.

        `current_model_id(settings)` rather than the loaded pipeline's, so the scope holds
        before the first load and the page's first paint is already the right
        checkpoint's. Exact, where the search box's `model:` is a substring -
        `model:flux` is both FLUX.1 checkpoints.
        """
        return None if all_models else self.model_id()
