"""Write DEV#3 real-input browser steps for a prepared disposable fixture.

    uv run --no-sync python tools/trial_walkthrough.py --manifest <scratch>/fixture.json --output <NEW steps.json>
    node tools/browser.mjs <steps.json>

Run once per fresh fixture. Task mutations use real browser input; eval only
observes, scrolls targets, installs event counters and controls fake pacing.
The first assertion verifies the responding server's scratch ownership.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def steps_for(manifest: dict) -> list[dict]:
    steps: list[dict] = []
    roles = manifest["roles"]
    variant = manifest["variant"]
    port = manifest["port"]
    base = 4101 if variant == "A" else 6101

    def add(kind: str, value):
        steps.append({kind: value})

    def click(selector: str):
        add("eval", f"document.querySelector({json.dumps(selector)}).scrollIntoView({{block:'center'}})")
        add("click", selector)

    def fill(selector: str, value):
        click(selector)
        add("keycombo", ["Control", "a"])
        add("type", str(value))

    def expect(expression: str):
        add("expect", expression)

    def until(expression: str):
        expect(f"__until(() => {expression})")

    def task(name: str):
        add("log", json.dumps(name))
        add("eval", f"window.__task = {json.dumps(name)}; window.__presses = {{click:0,key:0}};")

    def result(extra: str = ""):
        add("log", "JSON.stringify({task:__task, trustedInputEvents:__presses, focus:document.activeElement?.id || document.activeElement?.tagName, draft:els.prompt.value, picks:selectedIds(), " + extra + "})")

    def search(value: str):
        fill("#search", value)
        add("key", "Enter")
        # Allow the input debounce to start its read before observing pending.
        add("wait", 500)
        until("!galleryPending && els.search.value === " + json.dumps(value))

    add("navigate", f"http://127.0.0.1:{port}/")
    add("eval", "window.__until = async(fn, ms=30000) => {const end=Date.now()+ms; while(Date.now()<end){try{if(await fn())return true}catch{} await new Promise(r=>setTimeout(r,50))}return false}; window.__post = (path,body)=>fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}).then(r=>r.json());")
    expected = str(Path(manifest["data"]) / "bundles").replace("\\", "/").lower()
    expect("(async()=>{const listing=await api('/api/gallery/bundles'); return listing.directory.replaceAll('\\\\','/').toLowerCase() === " + json.dumps(expected) + ";})()")
    until("modelState === 'ready' && !galleryPending")
    add("eval", "window.__presses={click:0,key:0}; document.addEventListener('click',e=>{if(e.isTrusted)__presses.click++},true); document.addEventListener('keydown',e=>{if(e.isTrusted)__presses.key++},true); window.__trialRequests=[]; window.__trialFetch=window.fetch; window.fetch=async(path,options)=>{const response=await __trialFetch(path,options); if(options?.method==='POST' && /^\\/api\\/(generate|rerender)$/.test(String(path)) && response.ok){__trialRequests.push({path,body:JSON.parse(options.body),answer:await response.clone().json()})}return response};")
    add("log", "JSON.stringify({userAgent:navigator.userAgent,viewport:[innerWidth,innerHeight],dpr:devicePixelRatio,syntheticFixture:true})")

    task("1 generation / Reset / Undo")
    click("#btn-reset")
    fill("#prompt", "An observatory above clouds" if variant == "A" else "A greenhouse beside a lake")
    # FLUX needs its separate True CFG above one before the negative is usable.
    # Record this real control detour rather than writing a disabled field.
    fill("#true-cfg", 1.5)
    expect("!els.negative.disabled")
    fill("#negative", "blur" if variant == "A" else "low contrast")
    add("eval", "[...els.presets.querySelectorAll('button')].find(b=>b.textContent==='7:9').id='trial-portrait'; [...els.tiers.querySelectorAll('button')].find(b=>b.textContent==='1 MP').id='trial-tier';")
    click("#trial-portrait")
    click("#trial-tier")
    fill("#seed", base)
    fill("#batch", 2)
    fill("#runs", 2)
    fill("#steps", 1)
    add("eval", "window.__generationDraft=JSON.stringify(panelState());")
    click("#btn-generate")
    until("!busy && !submitting && viewerRecords.length===4")
    expect("__trialRequests.at(-1).body.num_images===2 && __trialRequests.at(-1).body.runs===2 && viewerRecords.every(r=>r.width<r.height)")
    # The tall sticky panel remains at its lower controls after Generate.
    # scrollIntoView alone does not reliably raise it: use real upward wheel
    # navigation before pressing Reset again.
    add("wheel", [100, 300, -2000])
    add("wait", 450)
    click("#btn-reset")
    add("log", "JSON.stringify({task:'1 Reset',seed:els.seed.value,prompt:els.prompt.value,focus:document.activeElement.id})")
    expect("els.seed.value==='' && els.prompt.value==='' && document.activeElement===els.prompt")
    add("eval", "window.__resetCleared=els.seed.value==='' && els.prompt.value==='';")
    click("#prompt-note-undo")
    expect("JSON.stringify(panelState())===__generationDraft")
    click("#btn-reset")
    fill("#seed", base + 1000)
    expect(f"els.seed.value==='{base + 1000}'")
    result("made:4, resetClearedSeed:__resetCleared, undoRestoredDraft:true, upwardWheelDetour:true")

    task("2 queue / inspection / cancellation / live-server reconnect")
    add("eval", "__post('/api/fake/step',{seconds:0.4})")
    fill("#width", 256)
    fill("#height", 256)
    fill("#steps", 100)
    fill("#prompt", "Trial running " + variant + "1")
    click("#btn-generate")
    until("busy && currentJob && !submitting")
    add("eval", "window.__running=currentJob;")
    fill("#prompt", "Trial waiting " + variant + "2")
    click("#btn-generate")
    until("queuedIds.length===1 && !submitting")
    add("eval", "window.__waiting=queuedIds[0];")
    search("id:" + str(roles["child"]))
    until(f"galleryRecords.length===1 && galleryRecords[0].id==={roles['child']}")
    click(f"#gallery .card[data-id='{roles['child']}'] > img")
    until("els.lightbox.open")
    add("key", "Escape")
    fill("#prompt", "Trial survivor " + variant + "3")
    fill("#steps", 20)
    click("#btn-generate")
    until("queuedIds.length===2 && !submitting")
    add("eval", "window.__survivor=queuedIds[1];")
    # Tag the observed row so the next real click targets its generated ID.
    add("eval", "els.queueList.querySelector(`li[data-id='${__waiting}'] button`).id='trial-cancel-waiting'; document.getElementById('trial-cancel-waiting').scrollIntoView({block:'center'})")
    add("click", "#trial-cancel-waiting")
    until("queuedIds.length===1 && !queuedIds.includes(__waiting)")
    click("#btn-cancel")
    until("currentJob===__survivor && busy")
    expect("(async()=>{const states=await Promise.all([__running,__waiting].map(id=>api('/api/jobs/'+id))); return states.every(s=>s.status==='cancelled' && s.images.length===0);})()")
    add("eval", "try {sessionStorage.setItem('trial-survivor', __survivor)} catch {}")
    result("runningCancelled:true, queuedCancelled:true, survivorRunning:true, watchedJob:currentJob")
    add("navigate", f"http://127.0.0.1:{port}/")
    add("eval", "window.__until = async(fn,ms=30000)=>{const end=Date.now()+ms;while(Date.now()<end){try{if(await fn())return true}catch{}await new Promise(r=>setTimeout(r,50))}return false}; window.__task='after reload'; window.__presses={click:0,key:0}; document.addEventListener('click',e=>{if(e.isTrusted)__presses.click++},true); document.addEventListener('keydown',e=>{if(e.isTrusted)__presses.key++},true); window.__post=(path,body)=>fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}).then(r=>r.json()); try{window.__survivor=sessionStorage.getItem('trial-survivor')}catch{}")
    until("modelState==='ready' && !galleryPending")
    until("currentJob===__survivor && busy")
    expect("els.prompt.value===" + json.dumps("Trial survivor " + variant + "3"))
    add("log", "JSON.stringify({task:'2 reload',draft:els.prompt.value,watchedJob:currentJob,focus:document.activeElement?.id,liveServerReopenOnly:true})")
    add("eval", "__post('/api/fake/step',{seconds:0})")
    until("!busy && !submitting")
    expect("api('/api/jobs/'+__survivor).then(s=>s.status==='done' && s.images.length===1)")

    task("3 inspect / source compare / sheet")
    search("id:" + str(roles["child"]))
    until("galleryRecords.length===1")
    click(f"#gallery .card[data-id='{roles['child']}'] > img")
    until("els.lightbox.open && !els.lbCompareOn.hidden")
    click("#lb-compare-on")
    until(f"lbCompare?.source?.id==={roles['source']} && !els.lbSrcImg.hidden")
    expect("els.lbPrompt.textContent.includes('refined study')")
    click("#lb-fav")
    until("els.lbFav.getAttribute('aria-pressed')==='false'")
    click("#lb-fav")
    until("els.lbFav.getAttribute('aria-pressed')==='true'")
    add("key", "Escape")
    search("run:" + manifest["jobs"]["complete"]["run_id"])
    click("#btn-sheet")
    until("sheetMode && !galleryPending && document.querySelectorAll('#sheet-grid .sheet-cell:not(.empty)').length>0")
    click(".sheet-fill")
    expect("els.sheet.classList.contains('fill-width') && els.prompt.value===" + json.dumps("Trial survivor " + variant + "3"))
    click("#btn-sheet")
    search("id:" + str(roles["child"]))
    click(f"#gallery .card[data-id='{roles['child']}'] > img")
    until(f"els.lightbox.open && lbList[lbIndex].id==={roles['child']}")
    add("key", "Escape")
    result("sourceCompared:true, sheetFilled:true, returnedChild:true")

    task("4 reuse / Undo / retained picked rerun")
    draft = "Unfinished harbor draft" if variant == "A" else "Unfinished mountain draft"
    fill("#prompt", draft)
    search("id:" + str(roles["keeper"]))
    click(f"#gallery .card[data-id='{roles['keeper']}'] > img")
    click("#lb-reuse")
    until("!els.lightbox.open && reuseNote?.button?.isConnected")
    expect("els.prompt.value.includes(" + json.dumps("lighthouse" if variant == "A" else "observatory") + ")")
    click("#error button.undo")
    expect("els.prompt.value===" + json.dumps(draft))
    fill("#width", 512)
    fill("#height", 512)
    click("#btn-select")
    for record_id in roles["pick_in_order"]:
        search("id:" + str(record_id))
        click(f"#gallery .card[data-id='{record_id}'] > img")
    expect("selectedIds().join()===" + json.dumps(",".join(map(str, roles["pick_in_order"]))))
    click("#fav-only")
    until("!galleryPending")
    expect("els.rerender.textContent.includes('2')")
    click("#btn-rerender")
    until("!busy && !submitting && viewerRecords.length===2")
    expect("viewerRecords.length===2 && selectedIds().length===2 && els.prompt.value===" + json.dumps(draft))
    result("rerendered:2, preservedHiddenPicks:true, preservedDraft:true")

    task("5 whole-gallery password backup / clear / download / import")
    click("#btn-bundles")
    until("els.bundlesDialog.open && bundlePreview && !bundleReading")
    add("eval", "window.__archiveCount=bundlePreview.records; window.__archiveBin=bundlePreview.binned; window.__archiveLive=__archiveCount-__archiveBin;")
    expect("bundleScope()==='all' && bundlePreview.binned===1 && bundlePreview.records>60 && selectedIds().length===2 && els.favOnly.checked")
    fill("#bundle-password", "study-only-" + variant)
    fill("#bundle-password-again", "study-only-" + variant)
    click("#bundle-backup")
    until("bundleStatus.state==='done' && !bundleSettling && !bundleReading")
    expect("bundleStatus.result.records===__archiveCount && bundlePreview.records===__archiveCount")
    fill("#bundle-password", "study-only-" + variant)
    fill("#bundle-password-again", "study-only-" + variant)
    # Four initial images, one surviving job, two size-changed rerenders. The
    # cancelled jobs above are checked to have made zero. Refuse a count drift.
    fill("#bundle-count", manifest["counts"]["all_rows"] + 7)
    expect("+els.bundleCount.value===__archiveCount")
    click("#bundle-export")
    until("bundleStatus.state==='done' && bundlePreview.records===0 && !bundleSettling && !bundleReading")
    click("#bundle-download")
    fill("#bundle-unlock", "study-only-" + variant)
    click("#bundle-list [data-bundle-encrypted]")
    until("bundleStatus.operation==='import' && bundleStatus.state==='done' && !bundleSettling && !bundleReading")
    expect("bundleStatus.result.imported===__archiveCount && bundleStatus.result.binned===__archiveBin && bundleStatus.result.first_id>0")
    expect("(async()=>{const [live,bin]=await Promise.all([api('/api/gallery?all_models=true&limit=200'),api('/api/gallery/bin?limit=200')]); return live.total===__archiveLive && bin.total===__archiveBin && live.items.some(r=>r.model_id!==loadedModel);})()")
    result("records:__archiveCount, binned:__archiveBin, imported:bundleStatus.result.imported, downloadPressed:true")
    add("key", "Escape")
    return steps


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("fixture_version") != 1:
        parser.error("unsupported fixture version")
    if args.output.exists():
        parser.error("output already exists; choose a new evidence filename")
    args.output.write_text(json.dumps(steps_for(manifest), indent=2) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
