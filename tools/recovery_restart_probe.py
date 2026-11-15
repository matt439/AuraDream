"""CPU-only owned fake process-death/resume probe. Requires a NEW scratch path."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parents[1]


def run(data: Path, port: int, *, browser_finish: bool = False) -> dict:
    data = data.resolve()
    if data.exists() or not data.parent.is_dir() or not 1024 <= port <= 65535:
        raise ValueError("Use a new scratch path with existing parent and a loopback port 1024..65535")

    def listening():
        with socket.socket() as sock:
            sock.settimeout(.2)
            return sock.connect_ex(("127.0.0.1", port)) == 0

    if listening():
        raise ValueError("Port already occupied; refusing")
    data.mkdir()
    process = None
    runtime = None
    token = None
    log = (data / "server.log").open("ab")

    def api(path, body=None):
        request = urllib.request.Request(f"http://127.0.0.1:{port}" + path,
            data=None if body is None else json.dumps(body).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=10) as response:
            return json.load(response)

    def wait(fn, timeout=45):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            value = fn()
            if value:
                return value
            time.sleep(.05)
        raise TimeoutError("Owned probe condition timed out")

    def start():
        nonlocal process, runtime, token
        if listening():
            raise RuntimeError("Owned restart port is not closed")
        token = uuid.uuid4().hex
        (data / "fixture-owner.json").write_text(json.dumps({"token": token}), encoding="utf-8")
        process = subprocess.Popen([sys.executable, str(ROOT / "tools/trial_fixture.py"),
            "--data", str(data), "--port", str(port), "--worker-token", token],
            cwd=ROOT, stdout=log, stderr=log, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        def ready():
            nonlocal runtime
            try:
                receipt = json.loads((data / "fixture-runtime.json").read_text(encoding="utf-8"))
                if receipt["token"] != token:
                    return False
                runtime = receipt["pid"]
                return api("/api/gallery/bundles")["directory"].replace("\\", "/").lower() == str(data / "bundles").replace("\\", "/").lower()
            except (OSError, ValueError, urllib.error.URLError):
                return False
        wait(ready)

    def stop():
        nonlocal process, runtime
        if runtime is not None:
            receipt = json.loads((data / "fixture-runtime.json").read_text(encoding="utf-8"))
            if receipt["token"] != token or receipt["pid"] != runtime:
                raise RuntimeError("Runtime ownership changed; refusing to stop")
            try:
                os.kill(runtime, signal.SIGTERM)
            except ProcessLookupError:
                pass
            runtime = None
        if process is not None:
            if process.poll() is None:
                process.terminate()
            process.wait(timeout=15)
            process = None
        wait(lambda: not listening(), 10)

    try:
        start()
        api("/api/fake/step", {"seconds": .08})
        request = {"prompt": "restart probe {weather}", "prompt_axis": "weather = mist | rain",
            "seed": 834600, "seeds": 6, "steps": 8, "guidance_scale": 3,
            "width": 256, "height": 256}
        accepted = api("/api/grid", request)
        job = accepted["job_id"]
        def partial():
            state = api("/api/jobs/" + job)
            return state if len(state["images"]) >= 2 else None
        before = wait(partial)
        assert before["status"] == "running" and 0 < len(before["images"]) < 12
        run_id = before["images"][0]["run_id"]
        first_pid = runtime
        stop()  # Actual owned process termination, not a fake route or cancellation.
        start()
        second_pid = runtime
        assert first_pid != second_pid
        rows = api("/api/gallery?all_models=true")["items"]
        retained = [row for row in rows if row["run_id"] == run_id]
        assert 0 < len(retained) < 12
        assert {r["id"] for r in before["images"]} <= {r["id"] for r in retained}
        retained_by_id = {r["id"]: r for r in retained}
        assert api("/api/jobs") == []
        resume = dict(request, resume=True)
        preview = api("/api/grid/preview", resume)
        assert preview["skipped"] == len(retained) and preview["queued"] == 0
        assert preview["cells"] == preview["images"] == 12 - len(retained)
        api("/api/fake/step", {"seconds": 0})
        if browser_finish:
            # Reuse the existing browser contract, never its process lifecycle.
            if str(ROOT) not in sys.path:
                sys.path.insert(0, str(ROOT))
            from tools.check_run_continue import Check
            browser = Check(port, data)
            browser.browser("restarted-grid-finish", [(run_id, 12 - len(retained), False)])
            jobs = api("/api/jobs?include_finished=true")
            assert len(jobs) == 1, "Browser Finish must create exactly one job"
            resumed = jobs[0]["job_id"]
        else:
            resumed = api("/api/grid", resume)["job_id"]
        def finished():
            state = api("/api/jobs/" + resumed)
            return state if state["status"] in ("done", "error", "cancelled") else None
        result = wait(finished)
        assert result["status"] == "done" and len(result["images"]) == 12 - len(retained)
        final = [r for r in api("/api/gallery?all_models=true")["items"] if r["run_id"] == run_id]
        expected = {(f"restart probe {weather}", seed)
            for weather in ("mist", "rain") for seed in range(834600, 834606)}
        assert len(final) == 12 and {(r["prompt"], r["seed"]) for r in final} == expected
        for row in final:
            assert row["variables"] == {"weather": row["prompt"].split()[-1]}
            assert row["width"] == row["height"] == 256 and row["steps"] == 8
        missing = expected - {(r["prompt"], r["seed"]) for r in retained}
        assert {(r["prompt"], r["seed"]) for r in result["images"]} == missing
        assert all(r["run_id"] == run_id for r in result["images"])
        assert not ({r["id"] for r in result["images"]} & retained_by_id.keys())
        assert all(next(r for r in final if r["id"] == key) == value for key, value in retained_by_id.items())
        complete = api("/api/grid/preview", resume)
        assert complete["cells"] == 0 and complete["skipped"] == 12
        try:
            api("/api/grid", resume)
        except urllib.error.HTTPError as error:
            assert error.code == 409
        else:
            raise AssertionError("Completed resume queued duplicate work")
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        evidence = {"source": revision, "finish_path": "browser" if browser_finish else "api", "run_id": run_id, "first_pid": first_pid,
            "second_pid": second_pid, "original_job": job, "resumed_job": resumed,
            "persisted_after_restart": len(retained), "generated_missing": len(result["images"]),
            "final_unique_cells": len(final), "completed_resume_status": 409,
            "retained_rows_unchanged": True, "exact_missing_cells_verified": True, "preview": preview}
        (data / "restart-evidence.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
        return evidence
    finally:
        stop()
        log.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--browser-finish", action="store_true", help="exercise existing browser Finish and draft-preservation contract after restart")
    args = parser.parse_args()
    print(json.dumps(run(args.data, args.port, browser_finish=args.browser_finish), indent=2))
