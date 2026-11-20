"""Measure the interactive gallery workflow on a synthetic 5,000-image source.

Pass a new output directory. An optional existing fixture from gallery_workflow_fixture
is copied into scratch before the fake ever opens it. Run from the repo's Python.
"""
from __future__ import annotations

import argparse
import http.client
import json
import math
import os
import shutil
import socket
import sqlite3
import subprocess
import sys
import time
from contextlib import closing
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from tools.gallery_workflow_fixture import MODEL, ROOT, copy_database, plant


def request(url: str, body: dict | None = None) -> Any:
    target = urlsplit(url)
    if target.scheme != "http" or target.hostname != "127.0.0.1" or target.port is None:
        raise ValueError(f"Expected a loopback scratch URL: {url}")
    # HTTPConnection contacts this endpoint directly; a redirect is an error.
    with closing(http.client.HTTPConnection(target.hostname, target.port, timeout=10)) as conn:
        conn.request("POST" if body is not None else "GET", target.path,
                     json.dumps(body) if body is not None else None, {"Content-Type": "application/json"})
        with conn.getresponse() as response:
            if response.status != 200:
                raise RuntimeError(f"{target.path} returned HTTP {response.status}")
            return json.load(response)


def browser_steps(url: str, repeats: int) -> list[dict]:
    steps: list[dict] = [
        {"navigate": url}, {"inject": "gallery_workflow_browser.js"},
        {"expect": "flow.ready()", "fatal": True},
        {"eval": "(async () => {window.bg = await (await fetch('/api/generate', {method: 'POST', "
         "headers:{'Content-Type':'application/json'},body: JSON.stringify({prompt:'background night train', "
         "steps:3, runs:64, width:256, height:256, seed:10001})})).json()})()"},
    ]
    steps.append({"expect": "flow.waitFor(() => total >= 5000 && bg.job_id, 'background accepted')", "fatal": True})

    def bring(selector: str) -> dict:
        return {"eval": f"document.querySelector('{selector}').scrollIntoView({{block:'center',behavior:'instant'}})"}

    def action(name: str, selector: str, event: str = "click", typed: str = "") -> None:
        steps.append({"eval": f"flow.arm('{name}','{selector}','{event}')"})
        steps.append({"type": typed} if typed else {"click": selector})
        steps.append({"expect": "flow.done()", "fatal": True})

    image = '#gallery .card[data-id] > img'
    for i in range(repeats):
        if i:
            for selector in ("#btn-select-clear", "#btn-select", "#fav-only"):
                steps.extend([bring(selector), {"click": selector}])
            steps.append({"expect": "flow.waitFor(() => total === 1000 && !selectMode, 'reset')", "fatal": True})
        steps.extend([bring("#search"), {"eval": "els.search.focus(); els.search.select()"}])
        # Both snapshots sit outside the timed interaction.
        steps.append({"eval": "window.bgCounts ??= []; void 0"})
        count = "(await (await fetch('/api/jobs/' + bg.job_id)).json()).images.length"
        steps.append({"eval": f"(async () => bgCounts.push({count}))()"})
        action("search", "#search", "input", "lighthouse")
        action("starred", "#fav-only")
        steps.extend([bring(image), {"rightclick": image}, {"eval": "flow.menuItem()"}])
        action("sheet", "[data-benchmark-run]")
        action("back", "#btn-sheet")
        steps.extend([bring("#btn-select"), {"click": "#btn-select"}])
        action("pick", image)
        steps.extend([bring("#search"), {"eval": "els.search.focus(); els.search.select()"}])
        action("across", "#search", "input", "orchard")
        action("add", image)
        action("preview", "#btn-more-seeds")
        steps.append({"eval": f"(async () => bgCounts.push({count}))()"})
        action("submit", "#more-seeds-run")
    steps.append({"log": "({samples:flow.samples,counts:bgCounts,background:bg.job_id,browser:navigator.userAgent})"})
    return steps


def validate_source(directory: Path) -> None:
    with closing(sqlite3.connect((directory / "auradream.db").as_uri() + "?mode=ro", uri=True)) as conn:
        rows = conn.execute("SELECT COUNT(*), SUM(favorite), MIN(seed), MAX(seed), COUNT(DISTINCT run_id)"
                            " FROM images WHERE model_id = ?", (MODEL,)).fetchone()
    pngs = {p.stem for p in (directory / "outputs").glob("*.png")}
    thumbs = {p.stem for p in (directory / "outputs/thumbnails").glob("*.webp")}
    if len(pngs) != 5000 or thumbs != pngs:
        raise ValueError("Expected one PNG and one matching WebP per planted image")
    if rows != (5000, 1250, 1, 5000, 250):
        raise ValueError(f"Unexpected seed gallery: {rows}")


def cohort(base: Path, out: Path, repeats: int) -> dict:
    # Separate fake, database, thumbnails, Edge profile and output directory per cohort.
    # The planted PNGs are immutable; normal job appends write new files here.
    data = out / "data"
    data.mkdir()
    copy_database(base / "auradream.db", data / "auradream.db")
    for relative, pattern in (("outputs", "*.png"), ("outputs/thumbnails", "*.webp")):
        (data / relative).mkdir()
        for source in (base / relative).glob(pattern):
            target = data / relative / source.name
            try:
                os.link(source, target)
            except OSError:
                shutil.copyfile(source, target)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    steps = out / "steps.json"
    steps.write_text(json.dumps(browser_steps(url, repeats)), encoding="utf-8")
    shutil.copyfile(ROOT / "tools/gallery_workflow_browser.js", out / "gallery_workflow_browser.js")
    env = {k: v for k, v in os.environ.items() if not k.startswith("AURADREAM_") or k == "AURADREAM_BROWSER"}
    env.update(PYTHONUTF8="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    with (out / "server.log").open("wb") as log:
        server = subprocess.Popen([sys.executable, "tools/fake_server.py", "--model", MODEL, "--step", "0.13",
                                   "--data", str(data), "--port", str(port)], cwd=ROOT, env=env, stdout=log,
                                  stderr=subprocess.STDOUT)
        try:
            start = time.monotonic()
            while True:
                try:
                    request(url + "/api/status")
                    break
                except (ConnectionError, TimeoutError):
                    if time.monotonic() - start > 90 or server.poll() is not None:
                        raise RuntimeError(f"Fake failed; see {log.name}") from None
                    time.sleep(0.1)
            identity = request(url + "/api/fake/activity", {"reset": False})
            if not isinstance(identity.get("counted"), list):
                raise RuntimeError("Scratch fake did not identify its activity monitor")
            directory = request(url + "/api/gallery/bundles")["directory"]
            if Path(directory).resolve() != data / "bundles" or server.poll() is not None:
                raise RuntimeError("Endpoint does not belong to this scratch gallery")
            with (out / "browser.log").open("w", encoding="utf-8") as browser_log:
                subprocess.run(["node", "tools/browser.mjs", str(steps)], cwd=ROOT, env=env, stdout=browser_log,
                               stderr=subprocess.STDOUT, timeout=240, check=True)
            lines = (out / "browser.log").read_text(encoding="utf-8").splitlines()
            result = json.loads(next(line.split(" => ", 1)[1] for line in lines if "samples:flow.samples" in line))
            counts = result["counts"]
            if not all(0 <= a < b < 64 for a, b in zip(counts[::2], counts[1::2], strict=True)):
                raise ValueError(f"Background writer did not overlap every workflow: {counts}")
            # The confirmation must produce exactly the next four seeds on each original run.
            # All native timings are now retained. Finish queued writes promptly.
            request(url + "/api/fake/step", {"seconds": 0})
            deadline = time.monotonic() + 75
            while True:
                with closing(sqlite3.connect(data / "auradream.db")) as conn:
                    outputs = conn.execute("SELECT run_id, seed, filename, job_id FROM images WHERE id > 5002 AND"
                                           " prompt <> 'background night train' ORDER BY run_id, seed").fetchall()
                    rows = [(run, seed) for run, seed, _, _ in outputs]
                    background = conn.execute("SELECT count(*) FROM images WHERE prompt = 'background night train'").fetchone()[0]
                expected = [(f"{g:012x}", seed)
                            for g in (249, 250) for seed in range(20*g+1, 20*g+4*repeats+1)]
                if rows == expected and background == 64:
                    jobs = {record[3] for record in outputs}
                    if len(jobs) != 2 * repeats:
                        raise ValueError(f"Expected two continuation jobs per confirmation, got {len(jobs)}")
                    if all(request(url + "/api/jobs/" + job)["status"] == "done" for job in jobs):
                        for _, _, filename, _ in outputs:
                            if not (data / "outputs" / filename).is_file():
                                raise ValueError(f"Missing continuation PNG: {filename}")
                        break
                if time.monotonic() > deadline:
                    raise ValueError(f"Unexpected final rows: {rows}; writer: {background}")
                time.sleep(0.3)
            result["confirmed_outputs"] = rows
            return result
        finally:
            if server.poll() is None:
                if os.name == "nt":
                    # A Windows venv entry point can delegate to a child interpreter.
                    subprocess.run(["taskkill", "/pid", str(server.pid), "/T", "/F"],
                                   capture_output=True, timeout=15)
                else:
                    server.terminate()
            server.wait(timeout=15)
            stopped_by = time.monotonic() + 10
            while True:
                with socket.socket() as probe:
                    probe.settimeout(1)
                    if probe.connect_ex(("127.0.0.1", port)) != 0:
                        break
                if time.monotonic() >= stopped_by:
                    raise RuntimeError(f"Scratch server still listening on {port}")
                time.sleep(0.1)


def quantiles(values: list[float]) -> dict:
    ordered = sorted(values)
    return {k: round(ordered[math.ceil(len(values)*q)-1], 1) for k, q in (("p50", .5), ("p95", .95))}


def summarize(results: list[dict]) -> dict:
    groups: dict[str, dict[str, list[dict]]] = {"cold": {}, "warm": {}}
    for result in results:
        seen = set()
        for sample in result["samples"]:
            name = sample["name"]
            group = "warm" if name in seen else "cold"
            groups[group].setdefault(name, []).append(sample)
            seen.add(name)
    report: dict = {}
    for group, actions in groups.items():
        report[group] = {}
        for name, samples in actions.items():
            api: dict[str, list[dict]] = {}
            for sample in samples:
                for r in sample["api"]:
                    if r["path"] in {"/api/gallery", "/api/more-seeds/preview", "/api/more-seeds"} or r["path"].startswith("/api/runs/"):
                        api.setdefault(r["path"] + r["query"], []).append(r)
            report[group][name] = {"n": len(samples), "paint_ms": quantiles([s["ms"] for s in samples]),
                                   "dom_ms": quantiles([s["domMs"] for s in samples]),
                                   "api": {key: {"n": len(v), "ms": quantiles([r["duration"] for r in v]),
                                                "start_ms": quantiles([r["start"] for r in v])} for key, v in api.items()}}
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="new scratch results directory")
    parser.add_argument("--fixture", type=Path, help="reuse a planted gallery as a read-only source")
    parser.add_argument("--cohorts", type=int, default=6)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.cohorts < 1 or args.repeats < 1:
        parser.error("cohorts and repeats must be positive")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    base = args.fixture.resolve() if args.fixture else out / "source"
    if not args.fixture:
        plant(base)
    validate_source(base)
    results = []
    for index in range(args.cohorts):
        sub = out / f"session-{index+1}"
        sub.mkdir()
        print(f"Timing cohort {index+1}/{args.cohorts}: {sub}", flush=True)
        result = cohort(base, sub, args.repeats)
        (sub / "raw.json").write_text(json.dumps(result), encoding="utf-8")
        results.append(result)
    (out / "summary.json").write_text(json.dumps(summarize(results), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
