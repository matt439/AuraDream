"""Prepare DEV#3 synthetic trial data through the actual fake-server APIs.

    uv run --no-sync python tools/trial_fixture.py --data <NEW scratch path> --port 8312 --variant A

Never deletes or resets a directory. The owned fake server stops after setup;
the printed command reopens the fixture for a trial. No model is downloaded.

Fixture version 2 (DEV#21) adds a saved setup, a saved gallery search, a
personal snippet and a Then refine job (a base and its pass), each through
its real route, and records the More seeds refusal for another checkpoint.
"""
from __future__ import annotations

import argparse
import json
import os
import runpy
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = "black-forest-labs/FLUX.1-Krea-dev"
FIXTURE_VERSION = 2


def prepare(data: Path, port: int, variant: str) -> dict:
    data = data.resolve()
    if data.exists():
        raise ValueError(f"Refusing existing path: {data}. Choose a NEW scratch path; nothing was changed.")
    if not data.parent.is_dir():
        raise ValueError("The scratch parent must already exist.")
    free = shutil.disk_usage(data.parent).free
    if free < 1024**3:
        raise ValueError(f"Only {free:,} bytes free; fixture needs 1 GiB free headroom.")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", port))
    data.mkdir()
    base = f"http://127.0.0.1:{port}"

    def api(path: str, body: dict | None = None):
        request = urllib.request.Request(
            base + path,
            data=None if body is None else json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=20) as response:
            return json.load(response)

    def wait_job(job_id: str, predicate=None, timeout: float = 90):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            state = api("/api/jobs/" + job_id)
            if predicate is not None and predicate(state):
                return state
            if state["status"] not in ("queued", "running"):
                return state
            time.sleep(0.025)
        raise TimeoutError(f"Job {job_id} did not reach the requested state")

    token = uuid.uuid4().hex
    (data / "fixture-owner.json").write_text(json.dumps({"token": token}), encoding="utf-8")
    command = [sys.executable, str(Path(__file__).resolve()), "--worker-token", token,
               "--port", str(port), "--data", str(data)]
    runtime_pid = None

    def identify_runtime() -> int | None:
        identity_path = data / "fixture-runtime.json"
        if not identity_path.is_file():
            return None
        identity = json.loads(identity_path.read_text(encoding="utf-8"))
        if (identity["token"] != token or identity["pid"] == os.getpid()
                or not (identity["pid"] == process.pid or identity["parent_pid"] == process.pid)
                or not Path(identity["executable"]).samefile(sys.executable)):
            raise RuntimeError("Fake runtime identity did not match the owned launcher")
        return identity["pid"]
    manifest = {"fixture_version": FIXTURE_VERSION, "variant": variant, "data": str(data), "port": port,
                "model": MODEL, "prepared_utc": datetime.now(timezone.utc).isoformat(),
                "free_bytes_before": free, "roles": {}, "jobs": {}}
    with (data / "fixture-server.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        try:
            deadline = time.monotonic() + 60
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"Owned fake server exited ({process.returncode}); see fixture-server.log")
                try:
                    listing = api("/api/gallery/bundles")
                    break
                except (OSError, urllib.error.URLError):
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Owned fake server was not ready in 60 seconds") from None
                    time.sleep(0.1)
            # Verify ownership before the first mutation, even after the port probe.
            if Path(listing["directory"]).resolve() != data / "bundles":
                raise RuntimeError("The responding server does not own this scratch directory; no requests submitted")
            runtime_pid = identify_runtime()
            if runtime_pid is None:
                raise RuntimeError("Owned runtime did not acknowledge its identity; no requests submitted")
            manifest["owned_process"] = {"launcher_pid": process.pid, "runtime_pid": runtime_pid}
            seed = 4100 if variant == "A" else 6100
            subject = "an amber paper lighthouse" if variant == "A" else "a cobalt ceramic observatory"
            places = "quay | ridge" if variant == "A" else "shore | terrace"
            request = {"prompt": f"{subject} on a {{place}}", "prompt_axis": f"place = {places}",
                       "seed": seed, "seeds": 5, "steps_axis": "1,2", "cfg_axis": "2,3,4",
                       "width": 256, "height": 256}
            complete = wait_job(api("/api/grid", request)["job_id"])
            if complete["status"] != "done" or len(complete["images"]) != 60:
                raise RuntimeError(f"Completed fixture grid was unexpected: {complete['status']}")
            manifest["jobs"]["complete"] = {"job_id": complete["job_id"], "run_id": complete["images"][0]["run_id"],
                                               "planned": 60, "made": 60, "request": request}
            api("/api/fake/step", {"seconds": 0.05})
            interrupted_request = {"prompt": f"{subject} in {{weather}}", "prompt_axis": "weather = mist | rain",
                                   "seed": seed + 100, "seeds": 6, "steps": 8,
                                   "guidance_scale": 3, "width": 256, "height": 256}
            interrupted_id = api("/api/grid", interrupted_request)["job_id"]
            progress = wait_job(interrupted_id, lambda state: len(state["images"]) >= 2)
            if progress["status"] not in ("queued", "running"):
                raise RuntimeError("Interrupted fixture finished before cancellation; choose a fresh path and repeat")
            api(f"/api/jobs/{interrupted_id}/cancel", {})
            interrupted = wait_job(interrupted_id)
            if interrupted["status"] != "cancelled" or not 0 < len(interrupted["images"]) < 12:
                raise RuntimeError("The interrupted fixture is not a partial cancelled run")
            manifest["jobs"]["interrupted"] = {"job_id": interrupted_id,
                "run_id": interrupted["images"][0]["run_id"], "planned": 12,
                "made": len(interrupted["images"]), "request": interrupted_request}
            api("/api/fake/step", {"seconds": 0})

            def generate(body: dict) -> list[dict]:
                result = wait_job(api("/api/generate", {"width": 256, "height": 256, "steps": 8,
                                                       "guidance_scale": 3, **body})["job_id"])
                if result["status"] != "done":
                    raise RuntimeError(f"Fixture generation failed: {result}")
                return result["images"]

            source = generate({"prompt": f"{subject}, source study", "seed": seed + 200})[0]
            child = generate({"prompt": f"{subject}, refined study", "seed": seed + 201,
                              "refine": source["id"], "strength": 0.6})[0]
            rejects = generate({"prompt": f"{subject}, reject study", "seed": seed + 300, "num_images": 2})
            # Then refine: one base and its pass in one job, the pass in a
            # run of its own (backend/then_refine.py). Through /api/generate.
            then_request = {"prompt": f"{subject}, then refine study", "seed": seed + 400,
                            "then_refine": {"scale": 1.5, "strength": 0.5}}
            then_made = generate(then_request)
            then_job = then_made[0]["job_id"]
            then_rows = api(f"/api/gallery?limit=16&search=job:{then_job}")["items"]
            then_bases = [row for row in then_rows if row["refine"] is None]
            then_passes = [row for row in then_rows if row["refine"] is not None]
            if (len(then_bases) != 1 or len(then_passes) != 1
                    or then_passes[0]["refine"] != then_bases[0]["id"]
                    or then_passes[0]["run_id"] == then_bases[0]["run_id"]):
                raise RuntimeError("Then refine fixture did not make one base and its pass in their own runs")
            then_base, then_pass = then_bases[0], then_passes[0]
            manifest["jobs"]["then_refine"] = {"job_id": then_job, "base_run_id": then_base["run_id"],
                "pass_run_id": then_pass["run_id"], "planned": 2, "made": 2,
                "request": {"width": 256, "height": 256, "steps": 8, "guidance_scale": 3, **then_request}}
            # Sidecars outside the gallery: their own routes, never the files.
            status = api("/api/status")
            setup_name = "Lighthouse setup A" if variant == "A" else "Observatory setup B"
            setup_payload = {"mode": "generate", "prompt": f"{subject}, saved setup study",
                             "negative_prompt": "", "width": 512, "height": 768, "steps": 8,
                             "seed": seed + 500, "num_images": 1, "runs": 1}
            setups = api("/api/presets", {"name": setup_name, "model_id": MODEL, "payload": setup_payload,
                                          "model_revision": status["model_revision"], "step_cache": None})
            setup = next(item for item in setups["items"] if item["name"] == setup_name)
            search_name = "Lighthouse grid A" if variant == "A" else "Observatory grid B"
            searches = api("/api/gallery/saved-searches", {"name": search_name,
                "query": "run:" + complete["images"][0]["run_id"], "favorites": False, "all_models": False})
            saved_search = next(item for item in searches["items"] if item["name"] == search_name)
            snippet_name = "Harbor light A" if variant == "A" else "Terrace light B"
            snippet_text = "soft harbor light, long shadows" if variant == "A" else "cool terrace light, thin haze"
            snippets = api("/api/snippets", {"name": snippet_name, "text": snippet_text, "target": "prompt"})
            snippet = next(item for item in snippets["items"] if item["name"] == snippet_name)
            if len(setups["items"]) != 1 or len(searches["items"]) != 1 or len(snippets["items"]) != 1:
                raise RuntimeError("A fresh fixture should hold exactly one setup, saved search and snippet")
            keeper = complete["images"][0]
            for record in (keeper, child):
                api(f"/api/images/{record['id']}/favorite?value=true", {})
            api("/api/images/bin", {"ids": [rejects[0]["id"]]})
            gallery = api("/api/gallery?limit=1024&all_models=true")
            binned = api("/api/gallery/bin?limit=1024")
            foreign = [row for row in gallery["items"] if row["model_id"] != MODEL]
            if len(foreign) != 2 or len(binned["items"]) != 1:
                raise RuntimeError("Expected two planted foreign records and one binned reject")
            manifest["roles"] = {"keeper": keeper["id"], "complete_last": complete["images"][-1]["id"],
                "source": source["id"], "child": child["id"], "binned_reject": rejects[0]["id"],
                "live_reject": rejects[1]["id"], "foreign": [row["id"] for row in foreign],
                "pick_in_order": [keeper["id"], child["id"]],
                "then_refine_base": then_base["id"], "then_refine_pass": then_pass["id"],
                "setup": {"id": setup["id"], "name": setup["name"], "prompt": setup_payload["prompt"],
                          "width": setup_payload["width"], "height": setup_payload["height"],
                          "seed": setup_payload["seed"]},
                "saved_search": {"id": saved_search["id"], "name": saved_search["name"],
                                 "query": saved_search["query"]},
                "snippet": {"id": snippet["id"], "name": snippet["name"], "text": snippet["text"]}}
            manifest["counts"] = {"live": gallery["total"], "binned": binned["total"],
                                   "all_rows": gallery["total"] + binned["total"],
                                   "complete_run": 60, "interrupted_made": len(interrupted["images"]),
                                   "then_refine_made": 2, "setups": 1, "saved_searches": 1, "snippets": 1}
            refusals = {}
            for name, path, body in (
                ("foreign_rerender", "/api/rerender", {"ids": [foreign[0]["id"]], "width": 512, "height": 512}),
                ("binned_refine_source", "/api/generate", {"prompt": "refused fixture request", "steps": 8,
                 "width": 256, "height": 256, "refine": rejects[0]["id"], "strength": 0.6}),
                ("foreign_more_seeds", "/api/more-seeds", {"ids": [foreign[0]["id"]], "count": 2}),
            ):
                try:
                    api(path, body)
                except urllib.error.HTTPError as error:
                    if error.code not in (400, 404, 409, 422):
                        raise
                    refusals[name] = {"status": error.code, "response": json.load(error)}
                else:
                    raise RuntimeError(f"Expected refusal was accepted: {name}")
            manifest["refusals"] = refusals
            manifest["limitations"] = ["Synthetic pixels and processing time; no model/GPU evidence.",
                "Picks are browser-only; observer prepares pick_in_order with real input after launch.",
                "Setup, saved search and snippet are sidecars beside the database, outside bundles.",
                "Process exits after setup. This is fixture preparation, not a restart-recovery test."]
            (data / "fixture.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            return manifest
        finally:
            runtime_pid = runtime_pid or identify_runtime()
            # Windows venv python.exe can be only a redirector. Stop the
            # positively acknowledged runtime, not just the Popen launcher.
            if runtime_pid is not None:
                try:
                    os.kill(runtime_pid, signal.SIGTERM)
                except OSError:
                    # Already gone (Windows answers a vanished pid with
                    # WinError 87): the real failure is the one to report.
                    pass
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
            stopped_by = time.monotonic() + 5
            while True:
                with socket.socket() as probe:
                    probe.settimeout(0.25)
                    listening = probe.connect_ex(("127.0.0.1", port)) == 0
                if not listening:
                    break
                if time.monotonic() >= stopped_by:
                    raise RuntimeError("Setup ended but the port is still listening; inspect the owned process log")
                time.sleep(0.05)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="NEW scratch directory (parent must exist)")
    parser.add_argument("--port", type=int, default=8312)
    parser.add_argument("--variant", choices=("A", "B"), default="A")
    parser.add_argument("--worker-token", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_token:
        data = args.data.resolve()
        owner = json.loads((data / "fixture-owner.json").read_text(encoding="utf-8"))
        if owner["token"] != args.worker_token:
            parser.error("owned worker token did not match")
        (data / "fixture-runtime.json").write_text(json.dumps({"token": args.worker_token,
            "pid": os.getpid(), "parent_pid": os.getppid(), "executable": sys.executable}), encoding="utf-8")
        sys.argv = [str(ROOT / "tools/fake_server.py"), "--port", str(args.port), "--data", str(data),
                    "--model", MODEL]
        runpy.run_path(sys.argv[0], run_name="__main__")
        return 0
    if not 1024 <= args.port <= 65535:
        parser.error("use an unoccupied loopback port from 1024 through 65535")
    try:
        manifest = prepare(args.data, args.port, args.variant)
    except (OSError, ValueError, RuntimeError, TimeoutError) as error:
        print(str(error), file=sys.stderr)
        return 1
    print(json.dumps(manifest, indent=2))
    print("Reopen with:")
    print(subprocess.list2cmdline(["uv", "run", "--no-sync", "python", "tools/fake_server.py",
          "--port", str(args.port), "--model", MODEL, "--data", str(args.data.resolve())]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
