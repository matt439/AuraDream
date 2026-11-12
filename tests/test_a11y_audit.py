"""`tools/a11y_audit.js`, the accessibility audit the dialog smokes inject.

The file checks never start a browser: every smoke that loads the audit
watches after each navigate and asserts before the next and at its end, and
no app script loads it. The rest run the real audit in a real browser over
markup written here, one defect per check, and skip only where Edge or
Chrome is missing (Node is required, as for every browser-script test).
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from tests.test_browser_profile import _browser, _env

ROOT = Path(__file__).resolve().parent.parent
AUDIT = ROOT / "tools" / "a11y_audit.js"
INJECT = {"inject": "a11y_audit.js"}
WATCH = {"eval": "__a11yWatch()"}
CHECK = {"expect": "__a11yCheck()"}
# The dialog-heavy smokes the audit rides. A smoke dropped from here on
# purpose is a decision; one that lost its steps in an edit is caught.
WATCHED = {
    "smoke-bin", "smoke-bundles", "smoke-button-states", "smoke-context-menu", "smoke-delete",
    "smoke-generation-presets", "smoke-inpaint", "smoke-instruction-edit", "smoke-lightbox",
    "smoke-model-storage", "smoke-model-switch", "smoke-more-seeds", "smoke-outpaint", "smoke-phone-dialogs",
    "smoke-phone-gallery", "smoke-refine", "smoke-refine-batch", "smoke-saved-searches",
    "smoke-setup-portability", "smoke-shortcuts", "smoke-snippet-portability", "smoke-snippets",
    "smoke-swap", "smoke-then-refine", "smoke-download-progress", "smoke-idle-activity",
    "smoke-gallery-maintenance",
}


def _smokes() -> dict[str, list[dict]]:
    return {path.stem: json.loads(path.read_text(encoding="utf-8"))
            for path in sorted((ROOT / "tools").glob("smoke*.json"))}


def test_the_dialog_smokes_carry_the_audit():
    using = {name for name, steps in _smokes().items() if INJECT in steps}
    assert using == WATCHED


@pytest.mark.parametrize("name", sorted(WATCHED))
def test_a_watched_smoke_watches_every_page_and_checks_each(name):
    steps = _smokes()[name]
    navigates = [i for i, step in enumerate(steps) if "navigate" in step]
    assert navigates, name
    for n, i in enumerate(navigates):
        # A navigate drops what was injected: the watch starts again after
        # each, and what the page before it gathered is asserted first.
        assert steps[i + 1 : i + 3] == [INJECT, WATCH], f"{name}: step {i + 1} is not followed by the watch"
        if n:
            assert steps[i - 1] == CHECK, f"{name}: no check before the navigate at step {i + 1}"
    assert steps[-1] == CHECK, f"{name}: does not end with the check"


def test_no_app_script_loads_the_audit():
    for path in (ROOT / "web").rglob("*"):
        if path.suffix in {".js", ".html"}:
            assert "a11y_audit" not in path.read_text(encoding="utf-8"), path.name


# Each case: markup for the body, an expression over the report `r`, true when the audit is right.
CASES = [
    ("a clean page",
     "<button>Go</button><label>Name <input></label><button aria-disabled='true' title='No picks'>Bin</button>",
     "r.ok && !r.unnamed.length && !r.unreasoned.length && !r.duplicates.length && !r.dialogs.length"),
    ("a button with nothing inside", "<button id='b'></button>",
     "!r.ok && r.unnamed.length === 1 && r.unnamed[0] === 'button#b'"),
    ("a glyph hidden from the tree names nothing", "<button class='x y z'><span aria-hidden='true'>x</span></button>",
     "r.unnamed.length === 1 && r.unnamed[0] === 'button.x.y'"),
    ("undrawn text names nothing", "<button id='b'><span style='display:none'>Close</span></button>",
     "r.unnamed.length === 1"),
    ("a field with no label", "<input id='i'><select id='s'><option>a</option></select><textarea id='t'></textarea>",
     "r.unnamed.join() === 'input#i,select#s,textarea#t'"),
    ("the ways a control is named",
     "<button>Text</button><button aria-label='Close'></button><button title='Close'></button>"
     "<button><img alt='Close' src='data:,'></button><button><svg aria-label='Close'></svg></button>"
     "<span id='n'>Named</span><button aria-labelledby='n'></button>"
     "<label>Wrapped <input></label><label for='f'>For</label><input id='f'>"
     "<input placeholder='Search'><input title='Titled'><input type='submit'><input type='button' value='Go'>"
     "<a href='#x'>Link</a><div role='button' aria-label='Div'></div><summary>More</summary>",
     "r.ok && !r.unnamed.length"),
    ("a label names its field by its own words, not the field's",
     "<label><select id='s'><option>only the option</option></select></label>",
     "r.unnamed.join() === 'select#s'"),
    ("an aria-labelledby that names nothing", "<span id='n'></span><button id='b' aria-labelledby='n gone'></button>",
     "r.unnamed.join() === 'button#b'"),
    ("a display: contents label still names its radio",
     "<div style='display:grid'><label style='display:contents'><input type='radio'><span>Krea</span></label></div>",
     "r.ok"),
    ("what is not exposed is not audited",
     "<button hidden></button><div style='display:none'><button></button></div>"
     "<div style='visibility:hidden'><button></button></div><div aria-hidden='true'><button></button></div>"
     "<div inert><button aria-disabled='true'>Held</button><input></div>",
     "r.ok"),
    ("a visually hidden field is still read", "<input id='i' style='opacity:0;width:1px;height:1px'>",
     "r.unnamed.join() === 'input#i'"),
    ("a duplicate id, drawn or not", "<i id='twice'></i><div hidden><i id='twice'></i><i id='twice'></i></div><i id='once'></i>",
     "!r.ok && r.duplicates.join() === '#twice x3'"),
    ("a hold with no reason", "<button id='h' aria-disabled='true'>Queue</button>",
     "!r.ok && r.unreasoned.length === 1 && r.unreasoned[0] === 'button#h \"Queue\"'"),
    ("a hold's reason is its title or what describes it",
     "<button aria-disabled='true' title='Nothing picked'>Queue</button><span id='why'>Sent already</span>"
     "<input aria-label='Pad' aria-disabled='true' aria-describedby='why'><button aria-disabled='false'>Free</button>",
     "r.ok"),
    ("a label's title is not its checkbox's reason",
     "<label title='No refine pass here' aria-disabled='true'><input id='c' type='checkbox' aria-disabled='true'> Refine</label>",
     "r.unreasoned.join() === 'input#c \"Refine\"'"),
    ("an empty description is no reason", "<span id='why'> </span><button id='h' aria-disabled='true' aria-describedby='why' title=' '>Q</button>",
     "r.unreasoned.length === 1"),
    ("a held option of a drawn select", "<select aria-label='Model'><option id='o' aria-disabled='true'>Kontext</option></select>",
     "r.unreasoned.join() === 'option#o \"Kontext\"'"),
    ("the standing exception: a phone list's line of no value",
     "<button class='phone-list-row' role='option' data-value='' aria-disabled='true'>No saved setups in this view</button>"
     "<button class='phone-list-row' role='option' data-value='7' aria-disabled='true'>Held setup</button>",
     "r.unreasoned.join() === 'button.phone-list-row \"Held setup\"' && r.excepted.length === 1"),
    ("an open dialog with no name",
     "<dialog open id='d'><button>Close</button></dialog><dialog id='shut'></dialog>"
     "<dialog open aria-label='Named'></dialog><h2 id='t'>Title</h2><dialog open aria-labelledby='t'></dialog>"
     "<div role='dialog' id='r' title='A title is not a name'></div><h2 id='e'></h2><dialog open id='l' aria-labelledby='e'></dialog>",
     "!r.ok && r.dialogs.join() === 'dialog#d,div#r,dialog#l'"),
    ("natively disabled controls are listed and never fail", "<button disabled>Next</button><input aria-label='Mu' disabled>",
     "r.ok && r.disabled.join() === 'button \"Next\",input \"Mu\"'"),
]


def _run(steps: list[dict]) -> subprocess.CompletedProcess:
    # Short, under the system's temp, as test_browser_profile's own (a Linux
    # socket path's 108 bytes). The audit is copied beside the steps: an
    # inject is read from the steps file's folder.
    temp = Path(tempfile.mkdtemp(prefix="ad-"))
    try:
        shutil.copy(AUDIT, temp / "a11y_audit.js")
        (temp / "steps.json").write_text(json.dumps(steps), encoding="utf-8")
        return subprocess.run(
            ["node", str(ROOT / "tools" / "browser.mjs"), str(temp / "steps.json")],
            cwd=ROOT, env=_env(temp), capture_output=True, text=True, encoding="utf-8", timeout=180,
        )
    finally:
        shutil.rmtree(temp, ignore_errors=True)


def _body(markup: str) -> dict:
    return {"eval": f"document.body.innerHTML = {json.dumps(markup)}; true"}


browser = pytest.mark.skipif(_browser() is None, reason="needs Edge or Chrome")


@browser
def test_the_audit_reports_each_defect_and_nothing_else(node):
    steps: list[dict] = [{"navigate": "data:text/html,<title>a11y</title>"}, INJECT]
    for _, markup, verdict in CASES:
        steps += [_body(markup), {"expect": f"(() => {{ const r = __a11yAudit(); return {verdict}; }})()"}]
    steps += [
        _body("<button id='out'></button><section id='in'><button aria-disabled='true'>Held</button></section>"
              "<i id='d'></i><i id='d'></i><dialog open id='x'></dialog>"),
        # A root narrows the controls and holds; duplicates and dialogs stay the page's.
        {"expect": "(() => { const r = __a11yAudit({root: '#in'}); return !r.unnamed.length && r.unreasoned.length === 1"
                   " && r.duplicates.length === 1 && r.dialogs.length === 1; })()"},
        {"expect": "(() => { const r = __a11yAudit({root: document.getElementById('out')}); return r.unnamed.join() === 'button#out'; })()"},
        {"expect": "(() => { const r = __a11yAudit({except: ['#out', '#in button']});"
                   " return !r.unnamed.length && !r.unreasoned.length && r.excepted.length === 2; })()"},
        {"expect": "(() => { try { __a11yAudit({root: '#gone'}); return false; }"
                   " catch (e) { return e.message === '__a11yAudit: no element for root #gone'; } })()"},
    ]
    done = _run(steps)
    failed = [line for line in done.stdout.splitlines() if line.lstrip().startswith(("FAIL", "!"))]
    assert done.returncode == 0 and not failed, "\n".join(failed) + "\n" + done.stdout[-2000:] + done.stderr[-2000:]
    assert done.stdout.count("  ok   ") == len(CASES) + 5, done.stdout


@browser
def test_the_watch_audits_a_dialog_as_it_opens_and_shuts_and_the_check_says_what_failed(node):
    done = _run([
        {"navigate": "data:text/html,<title>a11y</title>"}, INJECT, WATCH,
        _body("<button>Open</button>"
              "<dialog id='clean' aria-label='Clean'><button>Close</button></dialog>"
              "<dialog id='filled' aria-label='Filled'><button id='f'></button></dialog>"
              "<dialog id='esc' aria-label='Esc'><button id='e'>Esc</button></dialog>"
              "<dialog id='late' aria-label='Late'><button id='run'>Run</button></dialog>"
              "<dialog id='bare'><button id='x'></button></dialog>"),
        {"expect": "__a11yCheck() === true && __a11yWatch() === true"},
        # What the opener writes after showModal() is in the audit of the opening...
        {"eval": "(async () => { const d = document.getElementById('clean'); d.showModal();"
                 " await new Promise((r) => setTimeout(r, 50)); d.close(); })()"},
        {"expect": "__a11yCheck() && __a11yWatched().seen.join() === 'clean'"},
        # A name the opener gives after showing is no finding (the button is gone
        # again before the dialog shuts, so only the opening's audit read it):
        # the opening is audited once the opener's script has run on, never
        # inside showModal().
        {"eval": "(async () => { const d = document.getElementById('filled'); d.showModal();"
                 " d.innerHTML = '<button aria-label=\"Filled\"></button>';"
                 " await new Promise((r) => setTimeout(r, 50)); d.close(); })()"},
        {"expect": "__a11yCheck() && __a11yWatched().seen.join() === 'clean,filled'"},
        # Escape shuts a dialog without close(): its cancel is heard, and what the
        # dialog had become by then is read.
        {"eval": "(async () => { const d = document.getElementById('esc'); d.showModal();"
                 " await new Promise((r) => setTimeout(r, 50));"
                 " document.getElementById('e').setAttribute('aria-disabled', 'true');"
                 " d.dispatchEvent(new Event('cancel', {cancelable: true}));"
                 " document.getElementById('e').remove(); d.close(); })()"},
        {"expect": "__a11yWatched().unreasoned.join() === 'button#e \"Esc\" in #esc'"},
        # ...and a hold that lost its reason while the dialog was open is caught as it shuts.
        {"eval": "(async () => { const d = document.getElementById('late'); d.showModal();"
                 " await new Promise((r) => setTimeout(r, 50));"
                 " document.getElementById('run').setAttribute('aria-disabled', 'true'); d.close(); })()"},
        {"expect": "__a11yWatched().unreasoned.at(-1) === 'button#run \"Run\" in #late'"},
        {"eval": "(async () => { const d = document.getElementById('bare'); d.showModal();"
                 " await new Promise((r) => setTimeout(r, 50)); d.close(); d.showModal(); d.close(); })()"},
        # Kept once each, after the dialogs have shut and the page itself is clean again.
        {"expect": "(() => { const w = __a11yWatched(); return w.unnamed.join() === 'button#x in #bare'"
                   " && w.dialogs.join() === 'dialog#bare' && w.seen.join() === 'clean,filled,esc,late,bare' && __a11yAudit().ok; })()"},
        {"expect": "__a11yCheck()"},
    ])
    assert done.returncode == 1, done.stdout + done.stderr
    lines = done.stdout.splitlines()
    assert [line for line in lines if line.lstrip().startswith("FAIL")] == ["  FAIL __a11yCheck()"], done.stdout
    said = next(line for line in lines if "__a11yCheck: " in line)
    assert ("unnamed: button#x in #bare | unreasoned: button#e \"Esc\" in #esc; button#run \"Run\" in #late"
            " | dialogs: dialog#bare") in said, said
