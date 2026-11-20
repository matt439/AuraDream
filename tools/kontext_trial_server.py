"""Build and serve the private INP#13 candidate on an owned loopback port.

The parent controller plants the scratch gallery and admits real work via the
guard mailbox. Nothing in this launcher or its patch is imported by AuraDream.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = "black-forest-labs/FLUX.1-Kontext-dev"
_TEXT_SUFFIXES = {".css", ".html", ".js", ".json", ".md", ".py", ".svg", ".webmanifest"}


def build_candidate(target: Path) -> str:
    """Apply the trial diff to a new copy. A changed context must refuse startup."""
    target = target.resolve()
    target.mkdir(parents=True, exist_ok=True)
    if any(target.iterdir()):
        raise ValueError("The candidate build directory must be empty.")
    tracked = subprocess.check_output(
        ["git", "ls-files", "-z", "--", "backend", "web", "docs/usage.md", "tests/fakes.py"],
        cwd=ROOT,
    ).decode("utf-8").split("\0")
    names = sorted({name for name in tracked if name} | {"tools/kontext_trial_runtime.py"})
    for name in names:
        destination = target / name
        destination.parent.mkdir(exist_ok=True, parents=True)
        if destination.suffix in _TEXT_SUFFIXES:
            # Git autocrlf changes checkout bytes. Bind the build to LF text
            # while retaining every other byte, including a source's BOM.
            destination.write_bytes((ROOT / name).read_bytes().replace(b"\r\n", b"\n"))
        else:
            shutil.copyfile(ROOT / name, destination)

    # Even a build inside the checkout's ignored work/ must apply relative
    # to this new copy. Do not discover the parent checkout's index.
    patch_env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    patch_env["GIT_CEILING_DIRECTORIES"] = str(target.parent)
    subprocess.run(
        ["git", "-c", "core.autocrlf=false", "apply", "--whitespace=nowarn", "-"],
        input=(ROOT / "tools/kontext_trial.patch").read_text(encoding="utf-8").encode("utf-8"),
        cwd=target, check=True, env=patch_env,
    )
    if "from tools.kontext_trial_runtime import trial_backend" not in (
        target / "backend/backends/__init__.py"
    ).read_text(encoding="utf-8"):
        raise ValueError("The private build did not acquire its guarded backend.")
    digest = hashlib.sha256()
    for name in names:
        digest.update(name.encode("utf-8") + b"\0" + hashlib.sha256((target / name).read_bytes()).digest())
    return digest.hexdigest()


def configure(mode: str, data: Path, port: int) -> dict[int, dict]:
    if not 1 <= port <= 65535:
        raise ValueError("Choose a specific TCP port from 1 through 65535.")
    marker = data / "kontext-fixtures.json"
    manifest = json.loads(marker.read_bytes())
    if manifest.get("protocol") != "inp13-kontext-fixtures-v1":
        raise ValueError("The trial requires its prepared fixture gallery.")
    if not (data / "auradream.db").is_file() or not (data / "outputs").is_dir():
        raise ValueError("The fixture gallery database or outputs directory is missing.")
    if any(path.resolve().parent != data for path in (marker, data / "outputs", data / "auradream.db")):
        raise ValueError("The fixture manifest, outputs and database must belong to this scratch gallery.")
    for name in list(os.environ):
        if name.startswith("AURADREAM_"):
            del os.environ[name]
    os.environ.update({
        "AURADREAM_MODEL_ID": MODEL, "AURADREAM_QUANT": "fp8", "AURADREAM_OFFLOAD": "model",
        "AURADREAM_DEVICE": "cuda:0", "AURADREAM_HOST": "127.0.0.1", "AURADREAM_PORT": str(port),
        "AURADREAM_PRELOAD": "false", "AURADREAM_CRITIC": "false", "AURADREAM_KEEP_AWAKE": "false",
        "AURADREAM_DB_PATH": str(data / "auradream.db"), "AURADREAM_OUTPUTS_DIR": str(data / "outputs"),
        "AURADREAM_LORAS_DIR": str(data / "loras"), "AURADREAM_BUNDLES_DIR": str(data / "bundles"),
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
    })
    if mode == "fake":
        os.environ.update(HF_HOME=str(data / "fake-hub"), HF_HUB_CACHE=str(data / "fake-hub" / "hub"),
                          CUDA_VISIBLE_DEVICES="")
    return {row["id"]: {key: row[key] for key in ("filename", "width", "height", "hash")}
            for row in manifest["fixtures"].values()}


# Only writes used by the qualification or the page's cold prompt controls.
# All model switches, cache operations and independent GPU tools stay closed.
_POSTS = frozenset({
    "/api/model/load", "/api/model/unload", "/api/instruction-edit", "/api/instruction-edit/preview",
    "/api/more-seeds", "/api/more-seeds/preview", "/api/generate", "/api/rerender", "/api/rerender/preview",
    "/api/prompt/measure", "/api/prompt/analyze", "/api/tags/validate",
})


def permitted(method: str, path: str) -> bool:
    if method in {"GET", "HEAD"}:
        return True
    return method == "POST" and (path in _POSTS or bool(re.fullmatch(
        r"/api/(?:jobs/[^/]+/cancel|runs/[^/]+/finish(?:/preview)?)", path,
    )))


def serve(args: argparse.Namespace) -> None:
    if args.mode == "real" and (args.guard_dir is None or sys.platform != "win32"):
        raise ValueError("The real trial requires Windows and its live guard directory.")
    data = args.data.resolve()
    build = args.build.resolve()
    fixture_proofs = configure(args.mode, data, args.port)
    guard = args.guard_dir.resolve() if args.guard_dir else None
    if guard is not None:
        guard.mkdir(parents=True, exist_ok=True)
        if any((guard / name).exists() for name in ("request.json", "permit.json", "request.tmp")):
            raise ValueError("The trial guard mailbox must be fresh.")
    digest = build_candidate(build)
    sys.path.insert(0, str(build))
    from tools.kontext_trial_runtime import Trial
    from backend import config
    if config.ROOT != build:
        raise RuntimeError("The running backend is outside the private candidate build.")
    trial = Trial(args.mode, guard)
    trial.install(fixture_proofs)
    from backend import main
    from fastapi import Request
    from fastapi.responses import JSONResponse

    @main.app.get("/api/kontext-trial", include_in_schema=False)
    async def observed():
        return trial.status(digest)

    @main.app.middleware("http")
    async def trial_requests(request: Request, call_next):
        if not permitted(request.method, request.url.path):
            return JSONResponse({"detail": "This action is outside the one-load Kontext trial."}, status_code=403)
        return await call_next(request)

    main.run()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("fake", "real"), required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--guard-dir", type=Path)
    serve(parser.parse_args())


if __name__ == "__main__":
    main()
