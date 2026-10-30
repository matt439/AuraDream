"""Run browser smokes, each on the fake server its row of the port table names.

    uv run python tools/run_smoke.py smoke-flux                # one smoke
    uv run python tools/run_smoke.py smoke-refine smoke-grid   # several
    uv run python tools/run_smoke.py --all --jobs 4            # every row, four at a time
    uv run python tools/run_smoke.py --list                    # the table as this reads it

The port table in `AGENTS.md` is the one source: its row gives the port, the
`--model` (a short name resolved through the line under the table), any
`--step`, and which smokes share one fake, in the order the row lists them.
Every fake started here gets a gallery of its own in a new temporary
directory passed through `--data`, so each row starts with fresh scratch data
without `--fresh` and no fake sees another's planted images. A smoke named
alone runs alone on its row's fake; `--all` runs every row with its smokes in order.

`--offset N` moves every port by N (the fake's and, through
`AURADREAM_PORT_OFFSET`, the smoke's), so a run can sit beside fakes
already up; `--jobs` runs rows in parallel, each worker 100 ports further
on. The fake's log is kept beside its gallery and named on a failure.
Every smoke retains screenshots, browser.log and result.json in its reported
artifact directory; failures retain fake-server.log too. The run's summary.json
lists all results. `--artifacts PATH` puts this run below a chosen root.
Exit 0 when every smoke passed, 1 when one failed, 2 when the command or a
smoke file was refused (as `browser.mjs` refuses one). An invalid effective
port or overlapping worker bands refuse before artifacts or processes start.

A failure seen under `--jobs` is rerun alone before it is believed:
`--rerun-failed-alone` runs each smoke whose exit was 1 again after the
planned run, serially at the run's `--offset`, each on a fresh fake of its
own, below `rerun/` in the same run root; `--from-summary PATH` does the
same for the exit-1 receipts of an earlier run's summary.json, in a new run
root. summary.json records `jobs` and `offset`, and after a rerun `rerun`
(receipts shaped like `results`) and `verdicts` (`passed-alone` or
`failed-alone` per first-pass failure); `results` and `duration_seconds`
keep the first pass's, written before any rerun starts. The exit
stays the first pass's: a smoke that passes alone is reported, not believed.
Exit 2 is never rerun.
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


def _port_plan_error(plan: list[tuple[Row, list[str]]], bands: list[int]) -> str | None:
    """Every selected row can take any free worker band; validate them cold."""
    used: dict[int, int] = {}
    for row, _smokes in plan:
        for band in bands:
            port = row.port + band
            if not 1 <= port <= 65535:
                return (
                    f"port {port} is outside 1..65535 "
                    f"(row {row.port}, offset {band}); choose another --offset or --jobs"
                )
            if port in used and used[port] != band:
                return (
                    f"worker bands overlap on port {port}; "
                    "use --jobs 1 or select fewer rows"
                )
            # Rows sharing a base port are safe in one band: only one row
            # owns that band at a time. Different bands must never overlap.
            used[port] = band
    return None


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


RERUN_DIR = "rerun"


def _receipts(artifacts: Path, pattern: str) -> list[dict]:
    """Each smoke's result.json under `artifacts`, with its directory relative to it."""
    return [
        {**json.loads(path.read_text(encoding="utf-8")),
         "artifact_dir": path.parent.relative_to(artifacts).as_posix()}
        for path in sorted(artifacts.glob(pattern))
    ]


def _summary_failures(path: Path, known: dict[str, Row]) -> tuple[list[str], str | None]:
    """The smokes an earlier summary.json records as exit 1, in its order,
    or why that file cannot be read as one."""
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return [], f"--from-summary: {path} does not exist"
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        return [], f"--from-summary: {path} is not a readable summary.json ({exc})"
    receipts = summary.get("results") if isinstance(summary, dict) else None
    if not isinstance(receipts, list):
        return [], f"--from-summary: {path} has no results list"
    failed: list[str] = []
    for receipt in receipts:
        smoke = receipt.get("smoke") if isinstance(receipt, dict) else None
        code = receipt.get("exit_code") if isinstance(receipt, dict) else None
        if not isinstance(smoke, str) or type(code) is not int:
            return [], f"--from-summary: {path} has a receipt without a smoke name and exit_code"
        if code == 1 and smoke not in failed:
            failed.append(smoke)
    unknown = [smoke for smoke in failed if smoke not in known]
    if unknown:
        return [], f"--from-summary: no longer in the port table: {', '.join(unknown)}"
    return failed, None


def _rerun_alone(names: list[str], known: dict[str, Row], offset: int, artifacts: Path) -> dict[str, str]:
    """Run each named smoke again, one at a time, on a fresh fake of its own;
    answer each one's verdict."""
    rerun_root = artifacts / RERUN_DIR
    rerun_root.mkdir(exist_ok=True)
    _say("", f"rerunning alone: {' '.join(names)}")
    verdicts: dict[str, str] = {}
    for smoke in names:
        code = run_row(known[smoke], [smoke], offset, stream=True, artifacts=rerun_root).get(smoke, 1)
        verdicts[smoke] = "passed-alone" if code == 0 else "failed-alone"
    return verdicts


def _print_verdicts(verdicts: dict[str, str]) -> None:
    for verdict in ("passed-alone", "failed-alone"):
        names = [smoke for smoke, said in verdicts.items() if said == verdict]
        if names:
            print(f"{verdict}: " + " ".join(names))


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
    parser.add_argument("--rerun-failed-alone", action="store_true",
                        help="after the run, rerun each exit-1 smoke alone, serially")
    parser.add_argument("--from-summary", type=Path, metavar="PATH",
                        help="rerun alone the exit-1 smokes of an earlier summary.json")
    args = parser.parse_args()

    rows = parse_table((ROOT / "AGENTS.md").read_text(encoding="utf-8"))
    if args.list:
        for row in rows:
            step = f" --step {row.step}" if row.step is not None else ""
            print(f"{row.port}  {row.model}{step}  {' '.join(row.smokes)}")
        return 0
    known = {smoke: row for row in rows for smoke in row.smokes}
    if args.from_summary is not None:
        return _main_from_summary(args, known)
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
    bands = [args.offset + WORKER_SPAN * n for n in range(jobs)]
    if error := _port_plan_error(plan, bands):
        print(error)
        return 2
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
        # One owner per band; preflight also excludes cross-band port overlap.
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
    took = time.monotonic() - started
    # The first pass's summary is written before any rerun, so an interrupted
    # rerun still leaves it; `duration_seconds` stays the first pass's.
    summary: dict = {
        "duration_seconds": round(took, 6),
        "results": _receipts(artifacts, "*/result.json"),
        "jobs": jobs, "offset": args.offset,
    }
    (artifacts / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    # Only exit 1 reruns: exit 2 is a refusal no rerun could explain.
    verdicts = None
    if args.rerun_failed_alone and (again := [smoke for smoke, code in results.items() if code == 1]):
        verdicts = _rerun_alone(again, known, args.offset, artifacts)
        summary["rerun"] = _receipts(artifacts, f"{RERUN_DIR}/*/result.json")
        summary["verdicts"] = verdicts
        (artifacts / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\n{len(results) - len(failed)} of {len(results)} passed in {took:.0f} s")
    if failed:
        print("failed: " + " ".join(failed))
    if verdicts is not None:
        _print_verdicts(verdicts)
    if any(code == 2 for code in results.values()):
        return 2
    return 1 if failed else 0


def _main_from_summary(args: argparse.Namespace, known: dict[str, Row]) -> int:
    """Rerun alone, in a new run root, the exit-1 smokes an earlier run recorded.
    Those failures stand: the exit is 1 whenever anything was rerun."""
    if args.smokes or args.all:
        print("--from-summary takes no smoke names and no --all")
        return 2
    again, error = _summary_failures(args.from_summary, known)
    if error is None:
        error = _port_plan_error([(known[smoke], [smoke]) for smoke in again], [args.offset])
    if error:
        print(error)
        return 2
    if not again:
        print(f"nothing to rerun: {args.from_summary} records no exit-1 smoke")
        return 0
    if args.artifacts is not None:
        args.artifacts.mkdir(parents=True, exist_ok=True)
    artifacts = Path(tempfile.mkdtemp(prefix="auradream-smoke-artifacts-", dir=args.artifacts)).resolve()
    print(f"artifacts: {artifacts}", flush=True)
    started = time.monotonic()
    verdicts = _rerun_alone(again, known, args.offset, artifacts)
    (artifacts / "summary.json").write_text(json.dumps({
        "duration_seconds": round(time.monotonic() - started, 6), "results": [],
        "jobs": 1, "offset": args.offset, "from_summary": str(args.from_summary),
        "rerun": _receipts(artifacts, f"{RERUN_DIR}/*/result.json"), "verdicts": verdicts,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"\nfailed in {args.from_summary}: " + " ".join(again))
    _print_verdicts(verdicts)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
