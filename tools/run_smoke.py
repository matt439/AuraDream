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
`AURADREAM_PORT_OFFSET`, the smoke's), so a run can sit beside fakes
already up; `--jobs` runs rows in parallel, each worker 100 ports further
on. The fake's log is kept beside its gallery and named on a failure.
Every smoke retains screenshots, browser.log and result.json in its reported
artifact directory; failures retain fake-server.log too. The run's summary.json
lists all results. `--artifacts PATH` puts this run below a chosen root.
Exit 0 when every smoke passed, 1 when one failed, 2 when the command or a
smoke file was refused (as `browser.mjs` refuses one).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
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
    for match in re.finditer(r"^\| (\d{4}) \|([^|\n]*)\|([^|\n]*)\|$", text, re.M):
        port, setup, smokes = match.group(1), match.group(2), match.group(3)
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


def _run_browser(command: list[str], env: dict[str, str], log_path: Path) -> int:
    """Capture output in order; let the harness clean its browser on interruption."""
    with open(log_path, "w", encoding="utf-8") as log:
        browser = subprocess.Popen(
            command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
        )
        try:
            return browser.wait()
        except BaseException:
            if browser.poll() is None:
                try:
                    browser.send_signal(signal.CTRL_BREAK_EVENT if sys.platform == "win32" else signal.SIGTERM)
                except (OSError, ValueError) as exc:
                    _say(f"browser cleanup signal failed: {exc}")
                try:
                    browser.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    if sys.platform == "win32":
                        subprocess.run(
                            ["taskkill", "/pid", str(browser.pid), "/T", "/F"],
                            stdout=log, stderr=subprocess.STDOUT, check=False,
                        )
                    if browser.poll() is None:
                        browser.kill()
                    browser.wait()
            raise


def run_row(row: Row, smokes: list[str], offset: int, stream: bool, artifacts: Path) -> dict[str, int]:
    """Start the row's fake on a new gallery, run `smokes` on it in order,
    stop it, retaining results even when startup fails."""
    port = row.port + offset
    row_started = time.monotonic()
    outputs = {smoke: Path(tempfile.mkdtemp(prefix=f"{smoke}-", dir=artifacts)) for smoke in smokes}
    for smoke, output_dir in outputs.items():
        _say(f"artifacts  {smoke}: {output_dir}")
    results: dict[str, int] = {}
    reports: dict[str, dict] = {}
    data: Path | None = None
    log_path: Path | None = None
    fake: subprocess.Popen | None = None

    def record(smoke: str, code: int, phase: str, started: float, output: str) -> None:
        results[smoke] = code
        (outputs[smoke] / "browser.log").write_text(output, encoding="utf-8")
        reports[smoke] = {
            "smoke": smoke, "model": row.model, "port": port,
            "phase": phase, "result": "passed" if code == 0 else "failed",
            "exit_code": code, "duration_seconds": round(time.monotonic() - started, 6),
            "browser_log": "browser.log", "fake_log": None,
            "detail": output.strip() if phase != "browser" else None,
        }

    try:
        if not _port_free(port):
            message = f"port {port} is in use: stop that server or pass --offset"
            _say(message)
            for smoke in smokes:
                record(smoke, 2, "port", row_started, message + "\n")
            return results
        data = Path(tempfile.mkdtemp(prefix=f"auradream-smoke-{row.port}-"))
        log_path = data / "fake-server.log"
        try:
            with open(log_path, "w", encoding="utf-8") as log:
                fake = subprocess.Popen(
                    [sys.executable, str(ROOT / "tools" / "fake_server.py"), *row.fake_args(port, data)],
                    cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                )
        except OSError as exc:
            message = f"fake on {port} could not start: {exc}"
            _say(message)
            for smoke in smokes:
                record(smoke, 2, "fake_start", row_started, message + "\n")
            return results
        if not _wait_ready(fake, port):
            message = f"fake on {port} ({row.model}) never answered"
            _say(f"FAIL  {message}; its log, {log_path}:", _tail(log_path))
            code = fake.poll() or 1
            for smoke in smokes:
                record(smoke, code, "fake_start", row_started, message + "\n")
            return results
        env = {**os.environ, "AURADREAM_PORT_OFFSET": str(offset)}
        for smoke in smokes:
            env["AURADREAM_SCREENSHOT_DIR"] = str(outputs[smoke])
            started = time.monotonic()
            command = ["node", str(ROOT / "tools" / "browser.mjs"), str(ROOT / "tools" / f"{smoke}.json")]
            phase = "browser"
            try:
                code = _run_browser(command, env, outputs[smoke] / "browser.log")
                output = (outputs[smoke] / "browser.log").read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                code, output, phase = 2, f"browser could not start: {exc}\n", "browser_start"
            except KeyboardInterrupt:
                output = (outputs[smoke] / "browser.log").read_text(encoding="utf-8", errors="replace")
                output += "\nbrowser interrupted\n"
                record(smoke, 130, "interrupted", started, output)
                raise
            record(smoke, code, phase, started, output)
            took = f"{time.monotonic() - started:.0f} s"
            if output and (stream or code != 0):
                _say(output.rstrip())
            if code == 0:
                _say(f"pass  {smoke} ({took})")
            else:
                _say(f"FAIL  {smoke} (exit {code}, {took}); fake log {log_path}")
    finally:
        if fake is not None:
            if fake.poll() is None:
                fake.terminate()
            try:
                fake.wait(timeout=10)
            except subprocess.TimeoutExpired:
                fake.kill()
                fake.wait()
        for smoke, report in reports.items():
            if results[smoke] != 0 and log_path is not None and log_path.exists():
                shutil.copyfile(log_path, outputs[smoke] / "fake-server.log")
                report["fake_log"] = "fake-server.log"
            (outputs[smoke] / "result.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if results and all(code == 0 for code in results.values()) and data is not None:
        # Windows can hold the log a moment after the fake has exited.
        for _ in range(20):
            shutil.rmtree(data, ignore_errors=True)
            if not data.exists():
                break
            time.sleep(0.25)
    return results


def main() -> int:
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfigure):
        reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
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
    artifacts = Path(tempfile.mkdtemp(prefix="auradream-smoke-artifacts-", dir=args.artifacts)).resolve()
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
    receipts = [
        {**json.loads(path.read_text(encoding="utf-8")), "artifact_dir": path.parent.name}
        for path in sorted(artifacts.glob("*/result.json"))
    ]
    (artifacts / "summary.json").write_text(json.dumps({
        "duration_seconds": round(time.monotonic() - started, 6), "results": receipts,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"\n{len(results) - len(failed)} of {len(results)} passed in {time.monotonic() - started:.0f} s")
    if failed:
        print("failed: " + " ".join(failed))
    if any(code == 2 for code in results.values()):
        return 2
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
