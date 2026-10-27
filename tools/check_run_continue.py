"""Check Finish run in fresh browser pages and after a real fake-server restart.

Uses only the existing Python/Node/Edge installation, small temporary fake
images and a new scratch gallery. Leaves its compact steps/logs for review.
No weights, GPU, production paths, live services or instruction-file edits.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
MODEL = "black-forest-labs/FLUX.1-Krea-dev"


class Check:
    def __init__(self, port: int, directory: Path):
        self.port = port
        self.url = f"http://127.0.0.1:{port}"
        self.directory = directory
        self.process: subprocess.Popen | None = None
        self.log = None
        self.starts = 0

    def api(self, path: str, body=None):
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(
            self.url + path, data=data,
            headers={"Origin": self.url, "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=15) as response:
            return json.load(response)

    def until(self, predicate, *, timeout=30):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            value = predicate()
            if value:
                return value
            time.sleep(0.04)
        raise AssertionError("timed out waiting for the isolated fake server")

    def listening(self):
        with socket.socket() as probe:
            probe.settimeout(0.2)
            return probe.connect_ex(("127.0.0.1", self.port)) == 0

    def start(self, *, step: float):
        # Refuse an occupied port before any API call can reach a server
        # this checker did not start. Connecting checks listeners without
        # mistaking Windows TIME_WAIT after our own restart for a server.
        if self.listening():
            raise RuntimeError(f"port {self.port} already has a listener")
        self.starts += 1
        log = (self.directory / f"server-{self.starts}.log").open("w", encoding="utf-8")
        self.log = log
        process = subprocess.Popen(
            [sys.executable, str(ROOT / "tools/fake_server.py"), "--port", str(self.port),
             "--model", MODEL, "--step", str(step), "--data", str(self.directory / "gallery")],
            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        self.process = process

        def ready():
            if process.poll() is not None:
                raise RuntimeError(f"owned fake server exited: {log.name}")
            try:
                return self.api("/api/status")
            except (OSError, urllib.error.URLError):
                return None

        return self.until(ready)

    def stop(self):
        if self.process is not None:
            if self.process.poll() is None:
                # Kill only this tool's direct fake-server child. This also
                # establishes recovery from a process that wrote no outcome.
                self.process.kill()
            self.process.wait(timeout=15)
            self.process = None
            # Windows can release the process handle just before its
            # listener stops answering. Do not mistake that tail for a
            # different server, or read it as the restarted instance.
            self.until(lambda: not self.listening(), timeout=5)
        if self.log is not None:
            self.log.close()
            self.log = None

    def job(self, job_id):
        return self.api(f"/api/jobs/{job_id}")

    def done(self, accepted):
        return self.until(lambda: (job if (job := self.job(accepted["job_id"]))["status"]
                                  not in ("queued", "running") else None))

    def browser(self, name: str, runs: list[tuple[str, int, bool]], *, budget=None):
        steps = [
            {"navigate": self.url + "/"}, {"wait": 1000},
            {"eval": "window.__until = async (fn, ms=20000) => {const end=Date.now()+ms;while(Date.now()<end){if(fn())return true;await new Promise(r=>setTimeout(r,40));}return false;};"},
            {"expect": "__until(() => document.getElementById('status-dot').className.includes('ready') && modelRevision !== null)"},
            {"expect": "submissions.size === 0"},
            {"eval": "resetPanel();writeBox(els.prompt,'UNRELATED composer stays here');writeBox(els.negative,'unrelated negative');els.seed.value='987654';els.width.value='512';els.height.value='256';els.steps.value='18';els.cfg.value='2.8';own('steps');own('cfg');syncSize();els.gridSteps.value='3,7';els.gridSeeds.value='2';els.gridResume.checked=true;els.gridSection.open=false;persistPanel();window.__snapshot=()=>JSON.stringify([els.prompt.value,els.negative.value,els.seed.value,els.width.value,els.height.value,els.steps.value,els.cfg.value,els.gridSteps.value,els.gridSeeds.value,els.gridResume.checked,els.gridSection.open]);window.__composer=__snapshot();window.__posts=[];window.__fetch=window.fetch;window.fetch=(u,o)=>{if(o?.method==='POST'&&/^\\/api\\/runs\\/[^/]+\\/finish$/.test(u))__posts.push({path:u,body:JSON.parse(o.body)});return __fetch(u,o);};"},
        ]
        for run, missing, empty in runs:
            steps.extend([
                {"log": json.dumps(f"Finish {run}: {missing} missing cells; empty before finish={empty}")},
                {"eval": f"(async()=>{{await loadRuns();window.__run={json.dumps(run)};els.runList.value=__run;els.runList.onchange();}})()"},
                {"expect": "__until(() => sheetMode && !galleryPending && sheetShownRun() === __run && !!els.sheetControls.querySelector('.finish-run-tool'))"},
                {"expect": f"__until(() => els.sheetControls.querySelector('.finish-run-tool').getAttribute('aria-disabled') === 'false' && finishRuns.get(__run)?.preview?.cells === {missing})"},
                {"expect": "__snapshot() === __composer"},
            ])
            if empty:
                steps.extend([
                    {"expect": "sheetRecords.length === 0 && [...els.runList.options].some(o=>o.value===__run)"},
                    {"expect": "els.sheetControls.querySelector('.more-seeds-tool').getAttribute('aria-disabled') === 'true'"},
                ])
            if budget is not None:
                steps.append({"expect": f"finishRuns.get(__run).preview.budget_minutes === {budget} && els.sheetControls.querySelector('.finish-run-tool').title.includes('original {budget}-minute budget')"})
                # A gallery/status refresh can replace this advisory entry
                # while a press is checking. It must recheck once and still
                # submit only the current pin, without a second user press.
                steps.append({"eval": "window.__finishApi=api;window.__backgroundRefreshes=0;api=async(path,options)=>{const answer=await __finishApi(path,options);if(path==='/api/runs/'+__run&&!__backgroundRefreshes){__backgroundRefreshes++;invalidateFinishRuns();}return answer;};"})
            steps.extend([
                {"eval": "window.__beforePosts=__posts.length;window.__button=els.sheetControls.querySelector('.finish-run-tool');window.__sentFocused=false;new MutationObserver(()=>{if(__button.textContent==='Finish sent'&&document.activeElement===__button)__sentFocused=true;}).observe(__button,{childList:true,characterData:true,attributes:true,subtree:true});__button.focus({preventScroll:true});__button.click();__button.click();"},
                {"expect": "__until(() => !busy && !submitting && __posts.length === __beforePosts+1 && __sentFocused, 30000)"},
                {"log": "JSON.stringify({composer:JSON.parse(__composer),after:JSON.parse(__snapshot()),posted:__posts.at(-1)})"},
                {"expect": "__posts.at(-1).path === '/api/runs/'+__run+'/finish' && Object.keys(__posts.at(-1).body).join() === 'expected' && /^[0-9a-f]{16}$/.test(__posts.at(-1).body.expected) && __snapshot() === __composer"},
                {"eval": "(async()=>{await showRun(__run);await loadRuns();})()"},
                {"expect": "__until(() => !galleryPending && finishRuns.get(__run)?.preview?.reason === 'nothing in this run is missing')"},
                {"expect": "__snapshot() === __composer && els.sheetControls.querySelector('.finish-run-tool').getAttribute('aria-disabled') === 'true'"},
            ])
            if budget is not None:
                steps.extend([
                    {"expect": "__backgroundRefreshes === 1"},
                    {"eval": "api=__finishApi;"},
                ])
        path = self.directory / f"{name}.json"
        path.write_text(json.dumps(steps, ensure_ascii=False, indent=2), encoding="utf-8")
        env = dict(os.environ)
        env.pop("LOCAL_DIFFUSION_PORT_OFFSET", None)
        log = self.directory / f"{name}.log"
        with log.open("w", encoding="utf-8") as output:
            result = subprocess.run(
                ["node", str(ROOT / "tools/browser.mjs"), str(path)], cwd=ROOT,
                env=env, stdout=output, stderr=subprocess.STDOUT, timeout=180,
            )
        if result.returncode:
            print(log.read_text(encoding="utf-8"))
            raise AssertionError(f"browser phase failed: {name}, exit {result.returncode}")
        print(f"{name}: {sum('expect' in step for step in steps)} browser expectations passed")


def main():
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfigure):
        reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=7909)
    args = parser.parse_args()
    if shutil.disk_usage(ROOT.anchor).free < 64 * 1024 * 1024:
        raise SystemExit("not enough free disk for compact fake-server evidence")
    directory = Path(tempfile.mkdtemp(prefix="local-diffusion-finish-check-"))
    print(f"Evidence: {directory}", flush=True)
    check = Check(args.port, directory)
    try:
        check.start(step=0.04)
        budget = check.api("/api/grid", {
            "prompt": "retained budget grid", "width": 256, "height": 256,
            "steps": 4, "seed": 1000, "seeds": 8, "budget_minutes": 0.01,
        })
        ended = check.done(budget)
        assert ended["stopped_by"] == "budget" and 0 < ended["image_count"] < 8, ended
        check.api("/api/fake/step", {"seconds": 0})
        missing = check.api(f"/api/runs/{budget['run_id']}/finish/preview", {})["cells"]
        check.browser("fresh-page-budget", [(budget["run_id"], missing, False)], budget=0.01)
        assert check.api(f"/api/gallery?search=run:{budget['run_id']}")["total"] == 8

        check.api("/api/fake/step", {"seconds": 0.04})
        repeat = check.api("/api/generate", {
            "prompt": "repeat interrupted by actual process loss", "width": 256, "height": 256,
            "steps": 8, "seed": 2000, "runs": 4,
        })
        check.until(lambda: check.job(repeat["job_id"])["image_count"] >= 1)
        comparison = check.api("/api/grid", {
            "prompt": "never-started comparison", "width": 256, "height": 256,
            "steps": 8, "seed": 3000, "seeds": 2,
            "prompt_variants": [{"label": "original", "prompt": "a red kite"},
                                {"label": "yours", "prompt": "a silver kite"}],
        })
        assert check.job(comparison["job_id"])["status"] == "queued"
        before = check.api("/api/status")["started_at"]
        check.stop()
        restarted = check.start(step=0)
        assert restarted["started_at"] != before
        for accepted in (repeat, comparison):
            try:
                check.job(accepted["job_id"])
            except urllib.error.HTTPError as error:
                assert error.code == 404
            else:
                raise AssertionError("restarted server retained an in-memory job")
        repeat_preview = check.api(f"/api/runs/{repeat['run_id']}/finish/preview", {})
        assert repeat_preview["available"] and 0 < repeat_preview["cells"] < 4
        zero = check.api(f"/api/runs/{comparison['run_id']}")
        assert zero["images"] == 0 and zero["comparison"] and zero["actions"]["finish"]["available"]
        check.browser("restarted-repeat-and-empty-comparison", [
            (repeat["run_id"], repeat_preview["cells"], False), (comparison["run_id"], 4, True),
        ])
        assert check.api(f"/api/gallery?search=run:{repeat['run_id']}")["total"] == 4
        assert check.api(f"/api/gallery?search=run:{comparison['run_id']}")["total"] == 4
        print("Fresh-page, retained-budget, real restart, Repeat and zero-image comparison recovery passed.")
    finally:
        check.stop()
        size = sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())
        print(f"Owned fake stopped; evidence {size / 1024 / 1024:.2f} MiB in {directory}")


if __name__ == "__main__":
    main()
