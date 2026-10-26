"""Run browser smokes, each on the fake server its row of the port table names.

    uv run python tools/run_smoke.py smoke-flux                # one smoke
    uv run python tools/run_smoke.py smoke-refine smoke-grid   # several
    uv run python tools/run_smoke.py --all --jobs 4            # every row, four at a time
    uv run python tools/run_smoke.py --list                    # the table as this reads it

The port table in `AGENTS.md` is the one source: its row gives the port, the
`--model` (a short name resolved through the line under the table), any
`--step`, and which smokes share one fake, in the order the row lists them.
Every fake started here gets a gallery of its own in a new temporary
directory, so a row's "fresh scratch `--data`" and `--fresh` hold always and
no fake sees another's planted images. A smoke named alone runs alone on its
row's fake; `--all` runs every row with its smokes in order.

`--offset N` moves every port by N (the fake's and, through
`LOCAL_DIFFUSION_PORT_OFFSET`, the smoke's), so a run can sit beside fakes
already up; `--jobs` runs rows in parallel, each worker 100 ports further
on. The fake's log is kept beside its gallery and named on a failure.
Screenshots are retained in a fresh artifact directory for every smoke,
reported as it starts. `--artifacts PATH` puts this run below a chosen root.
Exit 0 when every smoke passed, 1 when one failed, 2 when the command or a
smoke file was refused (as `browser.mjs` refuses one).
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "krea/Krea-2-Turbo"
WORKER_SPAN = 100
READY_SECONDS = 90.0
_print_lock = threading.Lock()


@dataclass
class Row:
    """One row of the port table: one fake, the smokes run on it in order."""

    port: int
    model: str
    step: float | None
    smokes: list[str] = field(default_factory=list)
    separate: bool = False

    def fake_args(self, port: int, data: Path) -> list[str]:
        args = ["--port", str(port), "--model", self.model, "--data", str(data)]
        if self.step is not None:
            args += ["--step", str(self.step)]
        return args


def _aliases(text: str) -> dict[str, str]:
    """The table's short names to full ids, from the line under it
    ("FLUX.1-Krea-dev is `black-forest-labs/FLUX.1-Krea-dev`; ...")."""
    ids = re.findall(r"`([\w.-]+/[\w.@-]+)`", text.split("| Port |", 1)[1].split("Gotchas:", 1)[0])
    return {full.split("/", 1)[1]: full for full in ids}


def parse_table(text: str) -> list[Row]:
    """Every row of `AGENTS.md`'s port table, in its order."""
    aliases = _aliases(text)
    rows = []
    for port, setup, smokes in re.findall(r"^\| (\d{4}) \|([^|\n]*)\|([^|\n]*)\|$", text, re.M):
        model, step = DEFAULT_MODEL, None
        for part in (p.strip() for p in setup.split(",")):
            if found := re.fullmatch(r"`--step ([\d.]+)`", part):
                step = float(found.group(1))
            elif part == "default" or "fresh" in part or not part:
                continue
            else:
                model = part if "/" in part else aliases.get(part.split("@", 1)[0], part)
        names = []
        for name in re.findall(r"`([^`]+)`", smokes):
            stem = name.removesuffix(".json")
            names.append("smoke" + stem if stem.startswith("-") else stem)
        rows.append(Row(int(port), model, step, names, "run separately" in setup))
    return rows


def _port_free(port: int) -> bool:
    with socket.socket() as probe:
        return probe.connect_ex(("127.0.0.1", port)) != 0


def _say(*lines: str) -> None:
    with _print_lock:
        for line in lines:
            print(line, flush=True)


def _wait_ready(fake: subprocess.Popen, port: int) -> bool:
    deadline = time.monotonic() + READY_SECONDS
    while time.monotonic() < deadline:
        if fake.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/status", timeout=2) as reply:
                if reply.status == 200:
                    return True
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(0.3)
    return False


def _tail(path: Path, lines: int = 30) -> str:
    try:
        return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])
    except OSError:
        return ""


def run_row(row: Row, smokes: list[str], offset: int, stream: bool, artifacts: Path) -> dict[str, int]:
    """Start the row's fake on a new gallery, run `smokes` on it in order,
    stop it; each smoke's exit code (a fake that never came up is 1)."""
    port = row.port + offset
    if not _port_free(port):
        _say(f"port {port} is in use: stop that server or pass --offset")
        return {smoke: 2 for smoke in smokes}
    data = Path(tempfile.mkdtemp(prefix=f"local-diffusion-smoke-{row.port}-"))
    log_path = data / "fake-server.log"
    with open(log_path, "w", encoding="utf-8") as log:
        fake = subprocess.Popen(
            [sys.executable, str(ROOT / "tools" / "fake_server.py"), *row.fake_args(port, data)],
            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
        )
    results: dict[str, int] = {}
    try:
        if not _wait_ready(fake, port):
            _say(f"FAIL  fake on {port} ({row.model}) never answered; its log, {log_path}:", _tail(log_path))
            return {smoke: 1 for smoke in smokes}
        env = {**os.environ, "LOCAL_DIFFUSION_PORT_OFFSET": str(offset)}
        for smoke in smokes:
            output_dir = Path(tempfile.mkdtemp(prefix=f"{smoke}-", dir=artifacts))
            env["LOCAL_DIFFUSION_SCREENSHOT_DIR"] = str(output_dir)
            _say(f"artifacts  {smoke}: {output_dir}")
            started = time.monotonic()
            command = ["node", str(ROOT / "tools" / "browser.mjs"), str(ROOT / "tools" / f"{smoke}.json")]
            if stream:
                code = subprocess.run(command, cwd=ROOT, env=env).returncode
                output = ""
            else:
                done = subprocess.run(
                    command, cwd=ROOT, env=env, capture_output=True, text=True,
                    encoding="utf-8", errors="replace",
                )
                code, output = done.returncode, (done.stdout + done.stderr).rstrip()
            results[smoke] = code
            took = f"{time.monotonic() - started:.0f} s"
            if code == 0:
                _say(f"pass  {smoke} ({took})")
            else:
                _say(*([output] if output else []), f"FAIL  {smoke} (exit {code}, {took}); fake log {log_path}")
    finally:
        fake.terminate()
        try:
            fake.wait(timeout=10)
        except subprocess.TimeoutExpired:
            fake.kill()
            fake.wait()
    if all(code == 0 for code in results.values()):
        # Windows can hold the log a moment after the fake has exited.
        for _ in range(20):
            shutil.rmtree(data, ignore_errors=True)
            if not data.exists():
                break
            time.sleep(0.25)
    return results


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("smokes", nargs="*", help="smoke names, with or without .json or tools/")
    parser.add_argument("--all", action="store_true", help="every row of the port table")
    parser.add_argument("--list", action="store_true", help="print the table as read, and stop")
    parser.add_argument("--offset", type=int, default=0, help="add N to every port (default 0)")
    parser.add_argument("--jobs", type=int, default=1, help="rows run in parallel (default 1)")
    parser.add_argument("--artifacts", type=Path, help="retain each run's screenshot directories below PATH")
    args = parser.parse_args()

    rows = parse_table((ROOT / "AGENTS.md").read_text(encoding="utf-8"))
    if args.list:
        for row in rows:
            step = f" --step {row.step}" if row.step is not None else ""
            print(f"{row.port}  {row.model}{step}  {' '.join(row.smokes)}")
        return 0
    if shutil.which("node") is None:
        print("node is not on PATH")
        return 2

    known = {smoke: row for row in rows for smoke in row.smokes}
    wanted = [Path(name).name.removesuffix(".json") for name in args.smokes]
    unknown = [name for name in wanted if name not in known]
    if unknown or not (wanted or args.all):
        print(f"not in the port table: {', '.join(unknown)}" if unknown else "name a smoke, or pass --all")
        return 2
    if args.all:
        plan = [(row, row.smokes) for row in rows]
    else:
        plan = []
        for row in rows:
            mine = [smoke for smoke in row.smokes if smoke in wanted]
            if mine:
                plan.append((row, mine))

    jobs = max(1, min(args.jobs, len(plan)))
    if args.artifacts is not None:
        args.artifacts.mkdir(parents=True, exist_ok=True)
    artifacts = Path(tempfile.mkdtemp(prefix="local-diffusion-smoke-artifacts-", dir=args.artifacts)).resolve()
    print(f"artifacts: {artifacts}", flush=True)
    started = time.monotonic()
    results: dict[str, int] = {}
    if jobs == 1:
        for row, smokes in plan:
            results.update(run_row(row, smokes, args.offset, stream=True, artifacts=artifacts))
    else:
        # A worker per offset band, so two rows on one port never collide.
        bands = [args.offset + WORKER_SPAN * n for n in range(jobs)]
        free = list(bands)
        band_lock = threading.Lock()

        def work(item: tuple[Row, list[str]]) -> dict[str, int]:
            with band_lock:
                band = free.pop()
            try:
                return run_row(item[0], item[1], band, stream=False, artifacts=artifacts)
            finally:
                with band_lock:
                    free.append(band)

        with ThreadPoolExecutor(jobs) as pool:
            for done in pool.map(work, plan):
                results.update(done)

    failed = [smoke for smoke, code in results.items() if code != 0]
    print(f"\n{len(results) - len(failed)} of {len(results)} passed in {time.monotonic() - started:.0f} s")
    if failed:
        print("failed: " + " ".join(failed))
    if any(code == 2 for code in results.values()):
        return 2
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
