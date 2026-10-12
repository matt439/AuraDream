"""What the Inpaint dialog reads from the server and the page it is part of.

The dialog's refusal says "not enabled yet" for a family diffusers has an
inpaint class for but this app has deferred, and "no inpaint pipeline" for
the rest: `inpaint_deferred` is held here to the same list the backend
refuses by. The page's own behaviour is the smokes' (tools/smoke-inpaint*.json).
"""
from __future__ import annotations

import re
from pathlib import Path

import diffusers
import pytest

from backend import refine
from backend.model_defaults import INPAINT_DEFERRED, defaults_for

# `tests/` has no __init__.py; pytest puts this directory on sys.path.
from test_profiles import CASES

WEB = Path(__file__).resolve().parents[1] / "web"


@pytest.mark.parametrize("model_id, class_name", CASES)
def test_inpaint_deferred_is_exactly_the_deferred_families(model_id, class_name):
    defaults = defaults_for(model_id)
    deferred = class_name in INPAINT_DEFERRED
    assert defaults["inpaint_deferred"] is deferred, model_id
    if deferred:
        # A deferral is of a class that exists: never said of one that does not.
        assert refine.sibling(getattr(diffusers, class_name), "Inpaint") is not None
        assert defaults["inpaint_supported"] is False


def test_an_unknown_checkpoint_is_not_called_deferred():
    assert defaults_for("nobody/whatever")["inpaint_deferred"] is False


def test_the_dialog_scripts_and_styles_load_in_the_page_order():
    html = (WEB / "index.html").read_text(encoding="utf-8")
    scripts = re.findall(r'<script src="/static/([\w-]+\.js)"></script>', html)
    # After the Refine dialog's, whose prompt-check lines it reuses, and
    # before app.js, the one that runs anything.
    assert scripts.index("refine-prompt.js") < scripts.index("inpaint-mask.js") < scripts.index("inpaint.js")
    assert scripts[-1] == "app.js"
    styles = re.findall(r'href="/static/(styles/[\w-]+\.css|style\.css)"', html)
    # After sheet.css, whose card badge it restyles for an inpaint.
    assert styles.index("styles/sheet.css") < styles.index("styles/refine.css") < styles.index("styles/inpaint.css")
    for name in ("inpaint.js", "inpaint-mask.js", "styles/inpaint.css"):
        assert (WEB / name).is_file(), name
