"""Recovery-list ownership and bounded reads; rendered controls use the browser probe."""
import subprocess

import pytest

from tests.test_runs_scope_recovery import HARNESS, ROOT

pytestmark = pytest.mark.usefixtures("node")
RECOVERY_HARNESS = HARNESS.replace(
    "function $() { return null; }",
    """const recoveryNodes = {
      'review-runs': {open: true}, 'review-runs-scope': {textContent: ''},
      'review-runs-error': {textContent: ''},
      'review-runs-list': {children: [], replaceChildren() {this.children = [];}}
    };
    function $(id) {return recoveryNodes[id] ?? null;}
    function showError(text, note, {slot}) {slot.textContent = text;}
    """,
)
CASES = {
    "bounded_empty": """
      const task=loadReviewRuns(); assert.equal(requests.length,1);
      assert.equal(requests[0].path,'/api/runs?limit=20');
      requests[0].resolve([]); await task;
      assert.ok($('review-runs-scope').textContent.includes('No recent runs'));
    """,
    "all_models_explicit": """
      els.allModels.checked=true; const task=loadReviewRuns();
      assert.equal(requests[0].path,'/api/runs?limit=20&all_models=true');
      requests[0].resolve([]); await task;
    """,
    "closed_reply_ignored": """
      const task=loadReviewRuns(); $('review-runs').open=false;
      requests[0].resolve([row('late')]); await task;
      assert.equal($('review-runs-list').children.length,0);
    """,
    "scope_round_trip_stays_stale": """
      const task=loadReviewRuns(); els.allModels.checked=true; syncReviewRuns();
      els.allModels.checked=false; syncReviewRuns();
      requests[0].resolve([row('late')]); await task;
      assert.equal(reviewRunsCurrent(),false);
      assert.equal($('review-runs-list').children.length,0);
    """,
    "superseded_reply_ignored": """
      const old=loadReviewRuns(), fresh=loadReviewRuns();
      requests[1].resolve([]); await fresh;
      requests[0].resolve([row('late')]); await old;
      assert.equal($('review-runs-list').children.length,0);
      assert.ok($('review-runs-scope').textContent.includes('No recent runs'));
    """,
    "failure_is_not_empty": """
      const task=loadReviewRuns(); requests[0].reject(Error('controlled offline')); await task;
      assert.ok($('review-runs-scope').textContent.includes('could not be loaded'));
      assert.equal($('review-runs-error').textContent,'controlled offline');
    """,
    "bin_refuses_read": """
      binMode=true; await loadReviewRuns(); assert.equal(requests.length,0);
      assert.ok($('review-runs-scope').textContent.includes('Leave the recycle bin'));
    """,
}


@pytest.mark.parametrize("case", CASES)
def test_actual_recovery_read_ownership(case):
    done = subprocess.run(["node", "-e", RECOVERY_HARNESS, str(ROOT / "web/runs.js"), "(async()=>{" + CASES[case] + "})()"],
                          capture_output=True, text=True, timeout=30)
    assert done.returncode == 0, done.stdout + done.stderr
