"""Local prompt critic: prepare its pinned runtime, ask once and release the child process.

Runtime installation, process ownership, question construction and answer validation
live in the critic_runtime, critic_server, critic_context and critic_validation modules.
The coordinator keeps one server per critique and releases it before the GPU gate opens."""

from __future__ import annotations

import http.client
import logging
import os
import secrets
import subprocess
import tempfile
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from . import critic_context, critic_runtime, critic_server, critic_validation
from .config import settings
from .schemas import CriticStatus, GenerateRequest, PromptAnalysis, PromptCritique

# Public entry points stay available from this module.
from .critic_context import (
    ANSWER_OVERHEAD as ANSWER_OVERHEAD,
    Citation as Citation,
    CriticContext as CriticContext,
    ISSUE_KINDS as ISSUE_KINDS,
    MAX_ISSUES as MAX_ISSUES,
    MAX_TEXT as MAX_TEXT,
    MIN_TOKENS as MIN_TOKENS,
    PARTS as PARTS,
    SAMPLING as SAMPLING,
    SYSTEM_BUDGET_CHARS as SYSTEM_BUDGET_CHARS,
    build_context as build_context,
    check_fits as check_fits,
    estimate_tokens as estimate_tokens,
    max_tokens as max_tokens,
    request_body as request_body,
    response_schema as response_schema,
)
from .critic_reorder import (
    is_reorder as is_reorder,
)
from .critic_runtime import (
    Archive as Archive,
    BUILDS as BUILDS,
    Build as Build,
    CONTEXT as CONTEXT,
    CriticError as CriticError,
    CriticUnavailable as CriticUnavailable,
    MODEL as MODEL,
    MODEL_LABEL as MODEL_LABEL,
    MODEL_SHA256 as MODEL_SHA256,
    MODEL_SIZE as MODEL_SIZE,
    REASONING_BUDGET as REASONING_BUDGET,
    RELEASE as RELEASE,
    download_model as download_model,
    glibc_version as glibc_version,
    install_root as install_root,
    install_server as install_server,
    installed_server as installed_server,
    missing_libraries as missing_libraries,
    pinned_build_refusal as pinned_build_refusal,
    platform_key as platform_key,
    preflight as preflight,
    verify_model as verify_model,
)
from .critic_server import (
    HEALTH_TIMEOUT as HEALTH_TIMEOUT,
    REQUEST_TIMEOUT as REQUEST_TIMEOUT,
    STOP_TIMEOUT as STOP_TIMEOUT,
    server_command as server_command,
)
from .critic_validation import (
    locate as locate,
    validate as validate,
)

log = logging.getLogger(__name__)


def _content(response: Any) -> tuple[str, str | None]:
    try:
        choice = response["choices"][0]
        content = choice["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise critic_runtime.CriticError("llama-server returned no answer") from exc
    if not isinstance(content, str):
        raise critic_runtime.CriticError("llama-server returned no answer text")
    return content, choice.get("finish_reason")


# --- The critic --------------------------------------------------------------


class PromptCritic:
    """Prepare once, critique on demand, never leave a server running."""

    def __init__(self) -> None:
        self._prepare_lock = threading.Lock()
        self._run_lock = threading.Lock()
        self._process_lock = threading.Lock()
        self._process: subprocess.Popen | None = None
        self._prepared: tuple[Path, Path] | None = None

    def supported(self) -> bool:
        """Switched on, and a server to run: one of the user's, or a pinned
        build this system's glibc can run. A missing library is not checked
        here - one `apt install` away, it is `prepare()`'s to name."""
        if not settings.critic:
            return False
        if settings.critic_server is not None:
            return True
        key = critic_runtime.platform_key()
        return key is not None and critic_runtime._old_glibc(key) is None

    def ready(self) -> bool:
        """The server and the model are both on disk - no network, no hashing."""
        return self.supported() and self._server_ready() and self._model_ready()

    def _server_ready(self) -> bool:
        if settings.critic_server is not None:
            return settings.critic_server.is_file()
        key = critic_runtime.platform_key()
        return key is not None and critic_runtime.installed_server(key) is not None

    def _model_ready(self) -> bool:
        if settings.critic_model is not None:
            return settings.critic_model.is_file()
        return critic_runtime._model_cached() is not None

    def describe(self) -> CriticStatus:
        """The model's name and what a first use would download, for the
        line under the critic's button. File checks only - no network, no
        hashing - as `ready()` is, so `/api/status` can ask it on every poll.
        A file an override names is never a download, whether or not it is
        there: `prepare()` says so if it is not."""
        fetch: list[tuple[str, int]] = []
        key = critic_runtime.platform_key()
        if settings.critic_server is None and key is not None and critic_runtime.installed_server(key) is None:
            size = sum(archive.size for archive in critic_runtime.BUILDS[key].archives)
            fetch.append((f"llama.cpp {critic_runtime.RELEASE}", size))
        model = settings.critic_model
        if model is None and critic_runtime._model_cached() is None:
            fetch.append((critic_runtime.MODEL_LABEL, critic_runtime.MODEL_SIZE))
        return CriticStatus(
            # The footer's name for the same file (`_label`).
            label=model.stem if model is not None else critic_runtime.MODEL_LABEL,
            pinned=settings.critic_server is None and model is None,
            fetch=[name for name, _ in fetch],
            fetch_bytes=sum(size for _, size in fetch),
        )

    def prepare(self) -> None:
        """Download, verify and install what is missing. Idempotent, and no GPU:
        the route runs it before taking the generation gate."""
        if not settings.critic:
            raise critic_runtime.CriticUnavailable("the prompt critic is switched off (AURADREAM_CRITIC)")
        with self._prepare_lock:
            self._prepared = (self._server_path(), self._model_path())

    def _server_path(self) -> Path:
        if settings.critic_server is not None:
            if not settings.critic_server.is_file():
                raise critic_runtime.CriticUnavailable(
                    f"AURADREAM_CRITIC_SERVER={settings.critic_server} is not a file"
                )
            return settings.critic_server
        key = critic_runtime.platform_key()
        if key is None:
            raise critic_runtime.CriticUnavailable(
                "llama.cpp publishes no pinned build for this platform here; "
                "set AURADREAM_CRITIC_SERVER to a llama-server of your own"
            )
        # Ahead of both downloads: `prepare()` fetches the model after this.
        critic_runtime.preflight(key)
        return critic_runtime.install_server(key)

    def _model_path(self) -> Path:
        if settings.critic_model is not None:
            if not settings.critic_model.is_file():
                raise critic_runtime.CriticUnavailable(
                    f"AURADREAM_CRITIC_MODEL={settings.critic_model} is not a file"
                )
            return settings.critic_model
        return critic_runtime.download_model()

    def _label(self, model: Path) -> str:
        return model.stem if settings.critic_model is not None else critic_runtime.MODEL_LABEL

    def critique(
        self,
        model_id: str,
        request: GenerateRequest,
        applied: GenerateRequest,
        analysis: PromptAnalysis,
    ) -> PromptCritique:
        """Ask the critic about `request.prompt` on `model_id`. Blocking, and on
        the GPU: run it through `jobs.run_when_idle`, after `prepare()`.

        The rewrite's layer-1 lint is the route's, once the gate is open,
        since layer 1 can fetch a tokenizer; `rewrite_findings` here holds
        only `validate`'s own checks on it."""
        prepared = self._prepared
        if prepared is None or not all(path.is_file() for path in prepared):
            raise critic_runtime.CriticUnavailable("the prompt critic is not prepared; prepare() fetches it first")
        server, model = prepared
        context = critic_context.build_context(model_id, request, applied, analysis)
        critic_context.check_fits(context, estimated=True)
        with self._run_lock:
            started = time.perf_counter()
            response = self._run(server, model, context, request.prompt)
            seconds = time.perf_counter() - started
        timings = response.get("timings") if isinstance(response, Mapping) else None
        if isinstance(timings, Mapping):
            log.info(
                "critique in %.1f s: %s prompt tokens, %s generated at %.0f tok/s",
                seconds, timings.get("prompt_n"), timings.get("predicted_n"),
                timings.get("predicted_per_second") or 0.0,
            )
        content, finish = _content(response)
        try:
            return critic_validation.validate(
                content, context, request.prompt, model=self._label(model), seconds=seconds
            )
        except critic_runtime.CriticError as exc:
            if finish == "length":
                raise critic_runtime.CriticError(
                    "the critic ran out of tokens before it finished its answer; its answer "
                    f"repeats the prompt {context.copies} times, so a shorter prompt leaves "
                    "it more room"
                ) from exc
            raise

    def _run(self, server: Path, model: Path, context: critic_context.CriticContext, prompt: str) -> Any:
        api_key = secrets.token_urlsafe(32)
        port = critic_server._free_port()
        base = f"http://127.0.0.1:{port}"
        handle, name = tempfile.mkstemp(prefix="llama-server-", suffix=".log")
        log_path = Path(name)
        try:
            with os.fdopen(handle, "wb") as log_file:
                try:
                    process = critic_server._spawn(critic_server.server_command(server, model, port), log_file, api_key)
                except OSError as exc:
                    where = (
                        f"AURADREAM_CRITIC_SERVER={server}"
                        if settings.critic_server is not None else str(server)
                    )
                    raise critic_runtime.CriticError(
                        f"llama-server could not be started from {where}: {exc}"
                    ) from exc
                with self._process_lock:
                    self._process = process
                try:
                    critic_server._wait_until_healthy(process, base, log_path)
                    try:
                        status, _ = critic_server._request("GET", base + "/v1/models", None, None, 5.0)
                        if status != 401:
                            raise critic_runtime.CriticError(
                                f"llama-server answered {status} without its API key; a "
                                "server any web page could reach is not one to send prompts to"
                            )
                        # Last before the key goes out: the gap after
                        # `_free_port` is on the command line for any local
                        # user to race, and a stranger can answer 401 too.
                        owned = critic_server._listener_is_ours(port, process.pids())
                        if owned is False:
                            raise critic_runtime.CriticError(
                                f"what listens on port {port} is not the llama-server this app "
                                "started; neither the prompt nor its key was sent"
                            )
                        if owned is None:
                            log.debug("port %d's listener unchecked here; its key guards it", port)
                        body = critic_context.request_body(context)
                        counted = critic_context._counted(base, api_key, body, context, prompt)
                        if counted is None:
                            log.warning(
                                "llama-server did not count the question's tokens; "
                                "sized by the estimate"
                            )
                        else:
                            critic_context.check_fits(counted)
                            body["max_tokens"] = critic_context.max_tokens(counted)
                        status, payload = critic_server._request(
                            "POST", base + "/v1/chat/completions", api_key, body, critic_server.REQUEST_TIMEOUT
                        )
                    except (OSError, http.client.HTTPException) as exc:
                        raise critic_runtime.CriticError(
                            f"llama-server did not answer: {exc}; {critic_server._tail(log_path)}"
                        ) from exc
                    if status != 200:
                        raise critic_runtime.CriticError(
                            f"llama-server answered {status}: {critic_server._error_message(payload)}"
                        )
                    return payload
                finally:
                    # Always, and before the gate opens: an image job admitted
                    # next must not find 5.5 GiB of the card still taken.
                    try:
                        critic_server._stop(process)
                    finally:
                        with self._process_lock:
                            if self._process is process:
                                self._process = None
        finally:
            critic_server._remove_log(log_path)

    def unload(self) -> None:
        """Stop a server still running: on Unload, or with the app shutting
        down under it. A kill not waited out raises `CriticError` - the card
        may still be taken - which `main._stop_critic` logs."""
        with self._process_lock:
            process = self._process
        if process is not None:
            critic_server._stop(process)


prompt_critic = PromptCritic()
