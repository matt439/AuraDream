"""Write DEV#3/DEV#21 real-input browser steps for a prepared disposable fixture.

    uv run --no-sync python tools/trial_walkthrough.py --manifest <scratch>/fixture.json --output <NEW steps.json>
    node tools/browser.mjs <steps.json>

Run once per fresh fixture. Task mutations use real browser input; eval only
observes, scrolls targets, tags a runtime-numbered target with a trial-* id,
installs event counters and controls fake pacing. The first assertion
verifies the responding server's scratch ownership.

Fixture version 2 only (DEV#21): a version 1 manifest has no setup, saved
search, snippet or Then refine roles and is refused. Tasks 1-5 are the first
set; 6-8 are the optional second set the observer may offer.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

FIXTURE_VERSION = 2

# Counters, recorders and the Generate/Re-render request log, installed on
# load and again after the reload (a navigation drops page globals). A
# trusted press on a held (aria-disabled) control is recorded with its stated
# reason and never counted as the task's success.
_INSTRUMENT = (
    "window.__presses={click:0,key:0}; window.__held=[];"
    " document.addEventListener('click',e=>{if(!e.isTrusted)return; __presses.click++;"
    " const held=e.target.closest?.('[aria-disabled=\"true\"]');"
    " if(held)__held.push({control:held.id||held.textContent.trim().slice(0,40),reason:held.title})},true);"
    " document.addEventListener('keydown',e=>{if(e.isTrusted)__presses.key++},true);"
    " window.__trialRequests=[]; window.__trialFetch=window.fetch; window.fetch=async(path,options)=>{"
    "const response=await __trialFetch(path,options); if(options?.method==='POST'"
    " && /^\\/api\\/(generate|rerender)$/.test(String(path)) && response.ok){"
    "__trialRequests.push({path,body:JSON.parse(options.body),answer:await response.clone().json()})}"
    "return response};"
)


def steps_for(manifest: dict) -> list[dict]:
    """The browser steps for a version 2 manifest; raises ValueError on any other."""
    if manifest.get("fixture_version") != FIXTURE_VERSION:
        raise ValueError(f"unsupported fixture version {manifest.get('fixture_version')!r}; "
                         f"prepare a fresh version {FIXTURE_VERSION} fixture")
    steps: list[dict] = []
    roles = manifest["roles"]
    variant = manifest["variant"]
    port = manifest["port"]
    base = 4101 if variant == "A" else 6101

    def add(kind: str, value):
        steps.append({kind: value})

    def click(selector: str):
        target = json.dumps(selector)
        add("eval", f"document.querySelector({target}).scrollIntoView({{block:'center'}})")
        # The press must reach its target, not whatever covers its centre:
        # a missed press is a failed step, never a silent no-op.
        until(f"(()=>{{const t=document.querySelector({target}), r=t.getBoundingClientRect(),"
              " at=document.elementFromPoint(r.x+r.width/2,r.y+r.height/2);"
              " return r.width>0 && !!at && (t.contains(at)||at.contains(t));})()")
        add("click", selector)

    def expect(expression: str):
        add("expect", expression)

    def fill(selector: str, value):
        click(selector)
        # A click that misses (a field inside a closed disclosure has no box)
        # would type into whatever kept focus: refuse that rather than go on.
        expect(f"document.activeElement===document.querySelector({json.dumps(selector)})")
        add("keycombo", ["Control", "a"])
        add("type", str(value))

    def until(expression: str):
        expect(f"__until(() => {expression})")

    def task(name: str):
        add("log", json.dumps(name))
        add("eval", f"window.__task = {json.dumps(name)}; window.__presses = {{click:0,key:0}}; window.__held = [];")

    def result(extra: str = ""):
        add("log", "JSON.stringify({task:__task, trustedInputEvents:__presses, heldPresses:__held, focus:document.activeElement?.id || document.activeElement?.tagName, draft:els.prompt.value, picks:selectedIds(), note:els.error.textContent, " + extra + "})")

    def open_custom_size():
        # Compact setup (UI#19) keeps W/H in a closed disclosure. A participant
        # needs the same extra press, so the driver presses its summary.
        click("#custom-size-summary")
        expect("els.customSize.open")

    def search(value: str, route: str = "pointer"):
        if route == "slash":
            # `/` focuses search from the workspace (UI#17), query unchanged.
            add("key", "/")
            expect("document.activeElement===els.search")
            add("keycombo", ["Control", "a"])
            add("type", value)
        else:
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
    add("eval", _INSTRUMENT)
    add("log", "JSON.stringify({userAgent:navigator.userAgent,viewport:[innerWidth,innerHeight],dpr:devicePixelRatio,syntheticFixture:true,fixtureVersion:" + str(FIXTURE_VERSION) + "})")

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
    open_custom_size()
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
    # Tag the observed row's named cancel button (rows also hold move-to-front)
    # so the next real click targets its generated ID.
    add("eval", "els.queueList.querySelector(`li[data-id='${__waiting}'] [data-queue-action='cancel']`).id='trial-cancel-waiting'; document.getElementById('trial-cancel-waiting').scrollIntoView({block:'center'})")
    add("click", "#trial-cancel-waiting")
    until("queuedIds.length===1 && !queuedIds.includes(__waiting)")
    click("#btn-cancel")
    until("currentJob===__survivor && busy")
    expect("(async()=>{const states=await Promise.all([__running,__waiting].map(id=>api('/api/jobs/'+id))); return states.every(s=>s.status==='cancelled' && s.images.length===0);})()")
    add("eval", "try {sessionStorage.setItem('trial-survivor', __survivor)} catch {}")
    result("runningCancelled:true, queuedCancelled:true, survivorRunning:true, watchedJob:currentJob, customSizeDetour:true")
    add("navigate", f"http://127.0.0.1:{port}/")
    add("eval", "window.__until = async(fn,ms=30000)=>{const end=Date.now()+ms;while(Date.now()<end){try{if(await fn())return true}catch{}await new Promise(r=>setTimeout(r,50))}return false}; window.__task='after reload'; " + _INSTRUMENT + " window.__post=(path,body)=>fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}).then(r=>r.json()); try{window.__survivor=sessionStorage.getItem('trial-survivor')}catch{}")
    until("modelState==='ready' && !galleryPending")
    until("currentJob===__survivor && busy")
    expect("els.prompt.value===" + json.dumps("Trial survivor " + variant + "3"))
    add("log", "JSON.stringify({task:'2 reload',draft:els.prompt.value,watchedJob:currentJob,focus:document.activeElement?.id,customSizeOpen:els.customSize.open,width:els.width.value,liveServerReopenOnly:true})")
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
    until("!els.lightbox.open")
    # Contextual help by its key (UI#17), outside text fields.
    add("key", "?")
    until("els.shortcutsDialog.open")
    add("key", "Escape")
    until("!els.shortcutsDialog.open")
    search("run:" + manifest["jobs"]["complete"]["run_id"], route="slash")
    click("#btn-sheet")
    until("sheetMode && !galleryPending && document.querySelectorAll('#sheet-grid .sheet-cell:not(.empty)').length>0")
    click(".sheet-fill")
    expect("els.sheet.classList.contains('fill-width') && els.prompt.value===" + json.dumps("Trial survivor " + variant + "3"))
    click("#btn-sheet")
    search("id:" + str(roles["child"]))
    click(f"#gallery .card[data-id='{roles['child']}'] > img")
    until(f"els.lightbox.open && lbList[lbIndex].id==={roles['child']}")
    add("key", "Escape")
    result("sourceCompared:true, sheetFilled:true, returnedChild:true, helpKey:true, slashSearch:true")

    task("4 reuse / Undo / retained picked rerun")
    draft = "Unfinished harbor draft" if variant == "A" else "Unfinished mountain draft"
    fill("#prompt", draft)
    search("id:" + str(roles["keeper"]))
    click(f"#gallery .card[data-id='{roles['keeper']}'] > img")
    until("els.lightbox.open")
    # More seeds by its key (UI#17): opening previews and queues nothing.
    add("key", "m")
    until("document.getElementById('more-seeds-dialog').open")
    expect("!busy && queuedIds.length===0")
    add("key", "Escape")
    until("!document.getElementById('more-seeds-dialog').open && els.lightbox.open")
    click("#lb-reuse")
    until("!els.lightbox.open && reuseNote?.button?.isConnected")
    expect("els.prompt.value.includes(" + json.dumps("lighthouse" if variant == "A" else "observatory") + ")")
    click("#error button.undo")
    expect("els.prompt.value===" + json.dumps(draft))
    add("log", "JSON.stringify({task:'4 size disclosure',customSizeOpen:els.customSize.open})")
    expect("els.customSize.open")
    fill("#width", 512)
    fill("#height", 512)
    click("#btn-select")
    for record_id in roles["pick_in_order"]:
        search("id:" + str(record_id), route="slash")
        click(f"#gallery .card[data-id='{record_id}'] > img")
    expect("selectedIds().join()===" + json.dumps(",".join(map(str, roles["pick_in_order"]))))
    click("#fav-only")
    until("!galleryPending")
    expect("els.rerender.textContent.includes('2')")
    click("#btn-rerender")
    until("!busy && !submitting && viewerRecords.length===2")
    expect("viewerRecords.length===2 && selectedIds().length===2 && viewerRecords.every(r=>r.width===512 && r.height===512) && els.prompt.value===" + json.dumps(draft))
    result("rerendered:2, preservedHiddenPicks:true, preservedDraft:true, moreSeedsKeyOpenedOnly:true, slashSearch:true")

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
    # Setups, saved searches and snippets are sidecars outside bundles and
    # gallery maintenance: the clear and the import leave them as they were.
    sidecars = [("/api/presets", roles["setup"]["id"]),
                ("/api/gallery/saved-searches", roles["saved_search"]["id"]),
                ("/api/snippets", roles["snippet"]["id"])]
    expect("(async()=>{const lists=await Promise.all(" + json.dumps([path for path, _ in sidecars]) + ".map(p=>api(p))); const ids=" + json.dumps([item for _, item in sidecars]) + "; return lists.every((list,i)=>list.items.length===1 && list.items[0].id===ids[i]);})()")
    result("records:__archiveCount, binned:__archiveBin, imported:bundleStatus.result.imported, downloadPressed:true, sidecarsKept:true")
    add("key", "Escape")
    until("!els.bundlesDialog.open")

    # The optional second set. The import renumbered gallery ids, so these
    # tasks read sidecar ids from the manifest and job ids at run time only.
    task("6 restore a saved setup / Undo")
    setup = roles["setup"]
    setup_draft = "Unfinished setup draft " + variant
    fill("#prompt", setup_draft)
    add("eval", "window.__setupDraft=JSON.stringify(panelState());")
    click("#btn-examples")
    click("#setups-menu-item")
    until("els.setupsDialog.open && setupsReady && !setupsPending")
    click(f"#setups-list option[value='{setup['id']}']")
    until(f"setupsSelected?.id==={json.dumps(setup['id'])} && !setupsPending && els.setupsRestore.getAttribute('aria-disabled')!=='true'")
    click("#setups-restore")
    until(f"!els.setupsDialog.open && els.prompt.value==={json.dumps(setup['prompt'])} && !!setupsUndo")
    expect(f"els.width.value==='{setup['width']}' && els.height.value==='{setup['height']}' && els.seed.value==='{setup['seed']}' && !busy && !submitting")
    click("#prompt-note-undo")
    until("!setupsUndo && els.prompt.value===" + json.dumps(setup_draft))
    expect("JSON.stringify(panelState())===__setupDraft")
    result("restored:true, undoRestoredDraft:true, queued:queuedIds.length")

    task("7 reopen a run through a saved search")
    saved = roles["saved_search"]
    click("#btn-saved-searches")
    until("els.savedSearchDialog.open && savedSearchReady && !savedSearchPending")
    click(f"#saved-searches-list option[value='{saved['id']}']")
    until(f"savedSearchSelected?.id==={json.dumps(saved['id'])} && !savedSearchPending")
    click("#saved-searches-apply")
    until("!els.savedSearchDialog.open && !galleryPending && els.search.value===" + json.dumps(saved["query"]))
    expect("!els.favOnly.checked && galleryRecords.length>0 && galleryRecords.every(r=>'run:'+r.run_id===" + json.dumps(saved["query"]) + ")")
    result("applied:els.search.value, shown:galleryRecords.length, starredOff:!els.favOnly.checked")

    task("8 Generate with Then refine / find the pass beside its base")
    then_prompt = "Trial then refine " + variant
    fill("#prompt", then_prompt)
    fill("#steps", 8)
    add("eval", "els.thenRefineOpen.scrollIntoView({block:'center'})")
    add("log", "(()=>{const r=els.thenRefineOpen.getBoundingClientRect(), at=document.elementFromPoint(r.x+r.width/2,r.y+r.height/2); return JSON.stringify({task:'8 opener before wheel',top:Math.round(r.top),scrollY:Math.round(scrollY),at:at?.id||at?.className||at?.tagName||null})})()")
    # The tall sticky panel moves only with the user's own scrolling
    # (panel.js followScroll), never a programmatic scroll, so its lower rows
    # can sit below the window: real downward wheel navigation (a detour).
    add("wheel", [100, 300, 2000])
    add("wait", 450)
    click("#then-refine-open")
    until("els.thenRefineDialog.open")
    click("#then-refine")
    until("els.thenRefine.checked")
    # The strength the dialog starts at here (0.1) of a few steps leaves no
    # step, and the server refuses it in its own words: name one that runs.
    fill("#then-refine-strength", 0.5)
    until("els.thenRefineOpen.textContent.includes('0.5')")
    click("#then-refine-close")
    until("!els.thenRefineDialog.open && els.generate.textContent==='Generate + refine'")
    click("#btn-generate")
    until("!busy && !submitting && !!__trialRequests.at(-1)?.body.then_refine")
    add("eval", "window.__thenJob=__trialRequests.at(-1).answer.job_id;")
    until("(async()=>{const rows=(await api('/api/gallery?limit=8&search=job:'+__thenJob)).items; window.__thenRows=rows; return rows.length===2;})()")
    expect("(()=>{const base=__thenRows.find(r=>r.refine===null), pass=__thenRows.find(r=>r.refine!==null); window.__thenBase=base?.id; window.__thenPass=pass?.id; return !!base && !!pass && pass.refine===base.id && pass.run_id!==base.run_id && pass.width>base.width;})()")
    # Select mode from task 4 outlives the clear and import (its picks do
    # not): a click would pick the pass, not open it. Leave it first.
    expect("document.getElementById('btn-select').getAttribute('aria-pressed')==='true' && selectedIds().length===0")
    click("#btn-select")
    until("document.getElementById('btn-select').getAttribute('aria-pressed')==='false'")
    # The pass keeps its base's prompt: its words find the two together.
    search(then_prompt)
    until("!galleryPending && galleryRecords.length===2 && galleryRecords.some(r=>r.id===__thenPass) && galleryRecords.some(r=>r.id===__thenBase)")
    add("eval", "document.querySelector(`#gallery .card[data-id='${__thenPass}'] > img`).id='trial-pass-card';")
    click("#trial-pass-card")
    until("els.lightbox.open && lbList[lbIndex].id===__thenPass && !els.lbCompareOn.hidden")
    click("#lb-compare-on")
    until("lbCompare?.source?.id===__thenBase && !els.lbSrcImg.hidden")
    add("key", "Escape")
    result("base:__thenBase, pass:__thenPass, passBesideBase:true, downwardWheelDetour:true, leftSelectMode:true")
    return steps


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if args.output.exists():
        parser.error("output already exists; choose a new evidence filename")
    try:
        steps = steps_for(manifest)
    except ValueError as error:
        parser.error(str(error))
    args.output.write_text(json.dumps(steps, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
