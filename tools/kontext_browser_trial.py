"""Run the isolated one-load Kontext browser qualification or its fresh replay.

    python -m tools.kontext_browser_trial --mode fake --output work/kontext-fake-1
    python -m tools.kontext_browser_trial --mode fake --output work/kontext-fake-2 --replay-of work/kontext-fake-1

Real mode has the same browser, guard and evidence requirements as the rehearsal.
Each invocation creates its own output directory and never resumes an earlier run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from typing import Any
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

import psutil

from tools import kontext_trial_steps as steps
from tools.anima_refine_card import atomic_json
from tools.card_guard import KillOnCloseJob
from tools.kontext_trial_environment import capture_environment
from tools.kontext_trial_fixtures import CASES, INSTRUCTION, plant_fixtures, verify_result
from tools.kontext_trial_guard import GuardHeld, TrialGuard

ROOT = Path(__file__).resolve().parents[1]
SMALL = {"control-512": "1:1", **{aspect.replace(":", "x"): aspect for aspect in CASES if aspect != "1:1"}}
FULL = ["load", *SMALL, "projection", "first-1024", "more-seeds", "cancel", "repeat-1024",
        "return-512", "refusals", "unload"]
REPLAY = ["load", "projection", "fresh-1024", "unload"]
GUARD_FILES = ("guard.jsonl", "summary.json", "projection.json")
# The November 18 run recovered to this CUDA allocator residue after accepted unload.
RECOVERED_BYTES = 32 * 1024**2


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> None:
        return None


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def evidence_hashes(folder: Path) -> dict[str, str]:
    hashes = {}
    for name in GUARD_FILES:
        if (folder / name).is_file():
            with (folder / name).open("rb") as stream:
                hashes[name] = hashlib.file_digest(stream, "sha256").hexdigest()
    return hashes


def free_port(requested: int) -> int:
    with socket.socket() as test:
        test.bind(("127.0.0.1", requested))
        return test.getsockname()[1]


def listening(port: int) -> bool:
    with socket.socket() as test:
        test.settimeout(1)
        return test.connect_ex(("127.0.0.1", port)) == 0


class OwnedProcess:
    """Own the Chromium subprocesses as well as the server's workers on every exit."""

    def __init__(self, argv: list[str], log: Path, env: dict[str, str], *, module: bool = False) -> None:
        self.stop_lock = threading.Lock()
        self.stopped = False
        self.job = KillOnCloseJob.open()
        if sys.platform == "win32" and self.job is None:
            raise RuntimeError("Could not create a kill-on-close Job Object")
        # A launcher whose parent dies before handoff sees EOF and exits. No
        # child can be created outside the Windows Job Object before assign().
        bootstrap = (
            "import sys,runpy,subprocess\n"
            "if sys.stdin.buffer.readline() != b'go\\n': sys.exit(2)\n"
            + ("sys.argv.pop(0); runpy.run_module(sys.argv[0],run_name='__main__',alter_sys=True)"
               if module else "sys.exit(subprocess.call(sys.argv[1:], stdin=subprocess.DEVNULL))")
        )
        child_env = env.copy()
        program = python_launcher(child_env)
        try:
            with log.open("xb") as stream:
                self.process = subprocess.Popen([program, "-c", bootstrap, *argv], stdin=subprocess.PIPE,
                                                stdout=stream, stderr=subprocess.STDOUT, cwd=ROOT, env=child_env,
                                                start_new_session=os.name != "nt",
                                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            if self.job is not None and not self.job.assign(self.process.pid):
                self.process.kill()
                self.process.wait(timeout=10)
                raise RuntimeError("Could not bind the launched process to the owned Job Object")
            assert self.process.stdin is not None
            self.process.stdin.write(b"go\n")
            self.process.stdin.close()
        except BaseException:
            if self.job is not None:
                self.job.close()
            if hasattr(self, "process") and self.process.poll() is None:
                self.process.kill()
                self.process.wait(timeout=10)
            raise

    def stop(self) -> None:
        with self.stop_lock:
            if self.stopped:
                return
            if self.job is not None:
                self.job.close()
            else:
                try:
                    os.killpg(self.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            self.process.wait(timeout=15)
            self.stopped = True


def python_launcher(env: dict[str, str]) -> str:
    if sys.platform == "win32":
        # The venv redirector launches a child with another PID. Start the same
        # interpreter directly while preserving its venv site-packages.
        env["__PYVENV_LAUNCHER__"] = sys.executable
        return vars(sys)["_base_executable"]
    return sys.executable


class Trial:
    def __init__(self, out: Path, mode: str, port: int, prior: Path | None):
        self.out = out
        self.mode = mode
        self.port = port
        self.url = f"http://127.0.0.1:{port}"
        self.prior = prior
        self.guard = TrialGuard(out / "guard", fake=mode == "fake")
        self.opener = build_opener(ProxyHandler({}), NoRedirect())
        self.server: OwnedProcess | None = None
        self.pid = 0
        self.fixtures: dict[str, dict] = {}
        self.env = {k: v for k, v in os.environ.items() if not k.upper().startswith("AURADREAM_")}
        if os.environ.get("AURADREAM_BROWSER"):
            self.env["AURADREAM_BROWSER"] = os.environ["AURADREAM_BROWSER"]
        self.env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1",
                        HF_HUB_DISABLE_TELEMETRY="1", PYTHONUNBUFFERED="1")
        self.cases: dict[str, dict] = {name: {"status": "unrun"} for name in (REPLAY if prior else FULL)}
        self.receipt: dict[str, Any] = {
            "protocol": "inp13-kontext-browser-v1", "mode": mode, "kind": "fresh-replay" if prior else "full",
            "status": "running", "cases": self.cases, "port": port, "started": time.time(),
            "candidate": None, "model_loads": 0, "cleanup": {"server_started": False},
        }
        self.previous: dict = {}
        self.active = ""

    def save(self) -> None:
        atomic_json(self.out / "trial.json", self.receipt)

    def get(self, path: str) -> Any:
        with self.opener.open(Request(self.url + path), timeout=5) as response:
            return json.load(response)

    def identify(self) -> dict:
        status = self.get("/api/kontext-trial")
        if status["pid"] != self.pid or status["mode"] != self.mode or not status["build_digest"]:
            raise RuntimeError("Listening endpoint does not match the launched candidate")
        return status

    def launch(self) -> None:
        self.fixtures = plant_fixtures(self.out / "gallery")
        self.receipt["fixtures"] = self.fixtures
        browser = self.out / "browser"
        browser.mkdir()
        shutil.copyfile(ROOT / "tools/kontext_trial_assets/kontext_trial_page.js", browser / "kontext_trial_page.js")
        argv = ["tools.kontext_trial_server", "--mode", self.mode, "--port", str(self.port),
                "--data", str(self.out / "gallery"), "--build", str(self.out / "build"),
                "--guard-dir", str(self.out / "guard")]
        self.server = OwnedProcess(argv, self.out / "server.log", self.env, module=True)
        self.pid = self.server.process.pid
        self.receipt["cleanup"] = {"server_started": True, "pid": self.pid}
        self.receipt["model_loads"] = None
        self.guard.attach(self.pid, self.server.stop)
        deadline = time.monotonic() + 120
        while not listening(self.port):
            self.guard.check()
            if self.server.process.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError("Candidate did not start, see server.log")
            time.sleep(.1)
        state = self.identify()
        if state["ready"] or state["loads"] or state["accepted_unloads"]:
            raise RuntimeError("Expected freshly started unloaded editor")
        if self.prior and self.previous["candidate"]["build_digest"] != state["build_digest"]:
            raise GuardHeld("Fresh replay candidate build differs from first session")
        self.receipt["candidate"] = state
        self.receipt["model_loads"] = state["loads"]
        self.save()

    def phase(self, name: str, sequence: list[steps.Step], actions: list[dict] | None = None) -> dict:
        self.active = name
        case = self.cases[name]
        case.update(status="running", started=time.time())
        self.save()
        self.guard.start_phase(name, actions)
        script = self.out / "browser" / (name + ".json")
        log = script.with_suffix(".log")
        atomic_json(script, steps.begin(self.url, self.pid) + sequence)
        node = shutil.which("node")
        if node is None:
            raise RuntimeError("Node is required by tools/browser.mjs")
        driver = OwnedProcess([node, str(ROOT / "tools/browser.mjs"), str(script)], log, self.env)
        try:
            deadline = time.monotonic() + 750
            while driver.process.poll() is None:
                self.guard.check()
                if time.monotonic() > deadline:
                    raise RuntimeError(f"Browser timed out during {name}")
                time.sleep(.12)
            self.guard.check()
            if driver.process.returncode:
                raise RuntimeError(f"{name}: browser exited {driver.process.returncode}, see {log}")
        finally:
            driver.stop()
        case.update(guard=self.guard.end_phase(name), finished=time.time())
        return case

    def call(self, name: str, aspect: str, recipe: str = "kontext-v1", *,
             seeds: tuple[int, ...] = (12345,), more_of: int | None = None, cancel: bool = False) -> list[dict]:
        fixture = self.fixtures[aspect]
        width, height = (1024, 1024) if recipe == "kontext-1024-v1" else CASES[aspect][1]
        actions = [{"action": "edit", "seed": seed, "recipe": recipe, "source": fixture["id"],
                    "width": width, "height": height, "prompt": INSTRUCTION, "steps": 28, "guidance_scale": 2.5}
                   for seed in seeds]
        sequence = steps.more_seeds(more_of) if more_of else steps.edit(
            fixture["id"], recipe, width, height, seeds[0], cancel=cancel)
        existing = {job["job_id"] for job in self.get("/api/jobs?include_finished=true")}
        case = self.phase(name, sequence, actions)
        added = [job for job in self.get("/api/jobs?include_finished=true") if job["job_id"] not in existing]
        if len(added) != 1:
            raise RuntimeError(f"{name}: expected exactly one new job")
        job = self.get("/api/jobs/" + added[0]["job_id"])
        case["job"] = job
        if cancel:
            if job["status"] != "cancelled" or job["images"] or job["image_count"]:
                raise RuntimeError("Cancel did not reach terminal state with zero saved images")
        elif job["status"] != "done" or [r["seed"] for r in job["images"]] != list(seeds):
            raise RuntimeError(f"{name}: terminal job children differ from expected seeds")
        outputs = [verify_result(self.out / "gallery", fixture, recipe, int(record["seed"]), job,
                                 self.get("/api/images/" + str(record["id"])), real=self.mode == "real")
                   for record in job["images"]]
        case.update(status="completed", outputs=outputs)
        self.save()
        return outputs

    def unload(self) -> None:
        case = self.phase("unload", steps.power(False))
        state = self.identify()
        self.receipt["candidate_final"] = state
        if state["loads"] != 1 or state["accepted_unloads"] != 1 or state["ready"] is not False:
            raise RuntimeError("Did not observe one accepted try_unload of the loaded backend")
        if self.mode == "real":
            # Model offload can leave CUDA at zero immediately after Load.
            # The measured edit peaks are the relevant evidence of recovery.
            outputs = [r for entry in self.cases.values() for r in entry.get("outputs", [])]
            allocated, reserved = (state["allocator"][key] for key in ("allocated_bytes", "reserved_bytes"))
            if not (0 <= allocated <= reserved <= RECOVERED_BYTES and
                    all(allocated < r["vram_peak_allocated"] and reserved < r["vram_peak_reserved"] for r in outputs)):
                raise RuntimeError("CUDA allocator did not fall back to its measured recovery ceiling")
        case["recovery"] = self.guard.observe_unload(True, state["ready"])
        self.receipt["unload"] = state
        case["status"] = "completed"
        self.save()

    def body(self) -> None:
        self.receipt["environment"] = capture_environment(self.mode)
        self.save()
        if self.prior:
            self.previous = read_json(self.prior / "trial.json")
            prior_guard = evidence_hashes(self.prior / "guard")
            if (self.previous["status"] != "completed" or self.previous["kind"] != "full"
                    or self.previous["mode"] != self.mode
                    or self.previous["environment"] != self.receipt["environment"]
                    or set(prior_guard) != set(GUARD_FILES) or self.previous.get("guard_sha256") != prior_guard):
                raise GuardHeld("Fresh replay needs a completed full session with the same mode and environment")
            self.receipt["replay_of"] = str(self.prior)
        with self.guard:
            if self.receipt["environment"]["holds"]:
                raise GuardHeld(", ".join(self.receipt["environment"]["holds"]))
            self.launch()
            self.receipt["model_loads"] = None  # A guard stop can cut off the candidate's actual load reply.
            case = self.phase("load", steps.power(True))
            state = self.identify()
            self.receipt["model_loads"] = state["loads"]
            if state["loads"] != 1 or state["ready"] is not True:
                raise RuntimeError("Browser Load did not load the editor exactly once")
            case.update(status="completed", candidate=state)
            self.save()
            if not self.prior:
                for name, aspect in SMALL.items():
                    self.call(name, aspect)
            self.active = "projection"
            projection = (self.guard.replay_projection(self.prior / "guard") if self.prior else self.guard.project())
            self.cases["projection"] = {"status": "completed" if projection["verdict"] == "ok" else "held",
                                        "decision": projection}
            self.receipt["projection"] = projection
            self.save()
            if projection["verdict"] != "ok":
                self.receipt["status"] = "held"
                self.receipt["error"] = "1024-square call held by projection"
            elif self.prior:
                fresh, = self.call("fresh-1024", "1:1", "kontext-1024-v1")
                first = self.previous["cases"]["first-1024"]["outputs"][0]
                if fresh["rgb_sha256"] != first["rgb_sha256"]:
                    raise RuntimeError("Fresh replay full RGB digest differs")
            else:
                first, = self.call("first-1024", "1:1", "kontext-1024-v1")
                self.call("more-seeds", "1:1", "kontext-1024-v1", seeds=(12346, 12347), more_of=first["image_id"])
                self.call("cancel", "1:1", "kontext-1024-v1", seeds=(777,), cancel=True)
                repeat, = self.call("repeat-1024", "1:1", "kontext-1024-v1")
                if repeat["rgb_sha256"] != first["rgb_sha256"]:
                    raise RuntimeError("Same-load 1024 replay full RGB digest differs")
                control, = self.call("return-512", "1:1")
                if control["rgb_sha256"] != self.cases["control-512"]["outputs"][0]["rgb_sha256"]:
                    raise RuntimeError("Return to 512 control differs")
                case = self.phase("refusals", steps.refusals(self.fixtures["1:1"]["id"], self.fixtures["16:9"]["id"],
                                                             first["image_id"], first["run_id"]))
                case["status"] = "completed"
            self.unload()
            if self.receipt["status"] == "running":
                self.receipt["status"] = "completed"

    def run(self) -> int:
        try:
            self.body()
        except (KeyboardInterrupt, Exception) as exc:
            status = "held" if isinstance(exc, GuardHeld) else "failed"
            self.receipt.update(status=status, error=f"{type(exc).__name__}: {exc}")
            if self.active and self.active in self.cases:
                self.cases[self.active]["status"] = status
        finally:
            try:
                if self.server is not None:
                    children = []
                    try:
                        children = psutil.Process(self.pid).children(recursive=True)
                    except psutil.NoSuchProcess:
                        pass
                    self.server.stop()
                    _, alive = psutil.wait_procs(children, timeout=10)
                    self.receipt["cleanup"].update(exit_code=self.server.process.returncode,
                                                   listener_closed=not listening(self.port),
                                                   surviving_workers=[p.pid for p in alive])
                    if alive or listening(self.port):
                        raise RuntimeError("Owned listener or workers survived shutdown")
            except Exception as exc:
                self.receipt.update(status="failed", cleanup_error=str(exc))
            self.receipt["finished"] = time.time()
            self.receipt["guard_sha256"] = evidence_hashes(self.out / "guard")
            self.receipt["load_permitted"] = self.guard.loaded
            self.save()
        print(f"{self.receipt['status']}: {self.out / 'trial.json'}", flush=True)
        return {"completed": 0, "held": 2}.get(self.receipt["status"], 1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["fake", "real"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replay-of", type=Path)
    parser.add_argument("--port", type=int, default=0, help="loopback port (default: available ephemeral port)")
    args = parser.parse_args()
    output = args.output.resolve()
    port = free_port(args.port)
    output.mkdir(parents=True, exist_ok=False)
    return Trial(output, args.mode, port, args.replay_of.resolve() if args.replay_of else None).run()


if __name__ == "__main__":
    raise SystemExit(main())
