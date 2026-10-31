"""Check dismissal retains computation while retiring reveal/focus ownership.

Actual declaration-only analysis/critic handlers on inert DOM/API ports. No
browser, tokenizer, model, critic service, gallery or real request is executed.
"""
from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCENARIOS = [
    "main-close-keeps-hidden-and-focus", "again-close-keeps-hidden",
    "again-escape-editor-keeps-caret", "again-escape-inside-returns-check",
    "later-explicit-reopens", "same-prompt-cache-retained",
    "stale-background-does-not-enable-write", "newer-check-wins",
    "old-finally-does-not-reset-newer", "clear-retires-pending",
    "ordinary-failure-cleanup", "critic-precheck-close-continues-success",
    "critic-precheck-close-error-routes-outside", "critic-precheck-clear-stays-cancelled",
    "wide-check-focus", "narrow-check-keeps-focus", "visible-again-focus",
]

SCRIPT = r"""
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const root = process.argv[1], scenario = process.argv[2];
function deferred() {
  let resolve,reject;
  const promise = new Promise((yes,no) => { resolve=yes; reject=no; });
  return { promise,resolve,reject };
}
function fixture() {
  const calls = { requests:[],focus:[],errors:[],poisoned:[],criticPaints:0,comparePaints:[] };
  const document = { activeElement:null };
  function element(name) {
    const attrs = new Map(), classes = new Set();
    return { name,value:'',textContent:'',hidden:false,disabled:false,title:'',dataset:{},children:[],parentElement:null,
      scrollTop:0,selectionStart:0,selectionEnd:0,
      setAttribute:(key,value)=>attrs.set(key,String(value)), getAttribute:key=>attrs.get(key)??null,
      removeAttribute:key=>attrs.delete(key),
      classList:{ add:name=>classes.add(name),remove:name=>classes.delete(name),contains:name=>classes.has(name),
        toggle(name,on) { if (on) classes.add(name); else classes.delete(name); } },
      replaceChildren(...children) { this.children=children; }, appendChild(child) { this.children.push(child); return child; },
      querySelector(selector) { return selector === '.analysis-head h2' ? heading : null; },
      querySelectorAll(selector) { return this === els.promptAnalysis && selector === '[data-writes]' ? [write] : []; },
      contains(other) { return other === this || this.children.includes(other); },
      focus() { document.activeElement=this; calls.focus.push(name); },
      setSelectionRange(start,end) { this.selectionStart=start; this.selectionEnd=end; },
    };
  }
  const els = new Proxy({}, { get(target,key) { if (!(key in target)) target[key]=element(key); return target[key]; } });
  const heading = element('heading'), write = element('write'); write.dataset.title='Apply fixture';
  els.promptAnalysis.children=[els.analysisAgain,els.analysisClose,els.criticAsk,heading,write];
  els.promptAnalysis.hidden=true; els.promptAnalysis.parentElement=els.tagValidationWide;
  els.analysisAgain.hidden=true; els.analyze.textContent='Check'; els.analysisCritic.hidden=false;
  els.prompt.value='initial checked text'; els.negative.value=''; els.seed.value='42';
  document.body=element('body'); document.activeElement=els.analyze;
  document.createElement=tag=>element(tag); document.createDocumentFragment=()=>element('fragment');
  const poison = name => function () { calls.poisoned.push(name); throw Error('forbidden port: '+name); };
  const sandbox = vm.createContext({ els,document,referenceIds:[],calls,modelState:'ready',busy:false,elsewhereBusy:false,
    expandingPrompt:false,serverOffline:false,critiquing:false,critiqueOffline:false,promptCritiqueReady:false,
    promptCritiqueSupported:true,trueCfgSupported:false,negativeSupported:true,cfgThreshold:1,lastDefaults:{guidance_scale:2},
    collect:()=>({ prompt:els.prompt.value.trim(),negative_prompt:els.negative.value,reference:[],guidance_scale:2,seed:42 }),
    modelChangeHold:()=>false,sayModelChangeHold:poison('model hold'),checkpointName:id=>id,
    api:(url,options)=>{ const request={ url,options,...deferred() }; calls.requests.push(request); return request.promise; },
    showError:(text,note=false,{slot=els.error}={})=>{ slot.hidden=!text; slot.textContent=text||''; calls.errors.push({text,note}); },
    syncCompareRun:()=>{},syncPromptExpansion:()=>{},refreshStatus:poison('status request'),showJobNote:poison('job note'),
    replaceBoxText:poison('editor write'),persistPanel:poison('persist'),focusReference:poison('reference focus'),
    setTimeout:()=>1,clearTimeout:()=>{},setInterval:()=>1,clearInterval:()=>{},performance:{now:()=>0},
    OFFLINE_MESSAGE:'offline',KEEPS_PROMPT:'',INLINE_CITE:100,
  });
  for (const name of ['page-helpers.js','analysis.js','analysis-critic.js']) {
    new vm.Script(fs.readFileSync(path.join(root,'web',name),'utf8'),{filename:name}).runInContext(sandbox);
  }
  // Preserve real request/cache/stale/close/escape/focus/critic-chain handlers.
  // Presentation builders and clock/model dependencies are inert fixture ports.
  vm.runInContext(`renderEncoders=()=>{}; renderOrder=()=>{}; renderFindings=()=>{};
    renderCritic=()=>{calls.criticPaints++;}; renderCompare=()=>{calls.comparePaints.push(!els.promptAnalysis.hidden);};
    startCriticClock=()=>{}; stopCriticClock=()=>{};`,sandbox);
  const evaluate=text=>vm.runInContext(text,sandbox);
  const body=label=>({ model_id:'cold/check',words:3,characters:0,style:'prose',findings:[],encoders:[],marker:label });
  const reply=(index,label='reply')=>calls.requests[index].resolve(body(label));
  const edit=text=>{ els.prompt.value=text; sandbox.syncAnalysisStale(); };
  const start=(main=false)=>main?sandbox.checkFromButton():sandbox.checkPrompt();
  async function initial() { const work=start(); reply(0,'initial'); await work; assert.equal(els.promptAnalysis.hidden,false); }
  const hidden=()=>assert.equal(els.promptAnalysis.hidden,true,'dismissed Check was revealed by its old reply');
  const clean=()=>{ assert.equal(els.analyze.textContent,'Check'); assert.equal(els.analyze.getAttribute('aria-busy'),null); };
  async function until(predicate) { for (let i=0;i<20;i++) { if (predicate()) return; await new Promise(setImmediate); } assert.ok(predicate(),'inert async chain did not reach its expected boundary'); }
  return { els,heading,write,document,calls,sandbox,evaluate,reply,edit,start,initial,hidden,clean,until };
}
async function main() {
  const f=fixture(), {els,document,calls}=f;
  if (scenario==='wide-check-focus' || scenario==='narrow-check-keeps-focus') {
    if (scenario.startsWith('narrow')) els.promptAnalysis.parentElement={};
    const work=f.start(true); f.reply(0); await work;
    assert.equal(els.promptAnalysis.hidden,false);
    assert.equal(document.activeElement,scenario.startsWith('wide')?f.heading:els.analyze); f.clean();
  } else {
    await f.initial();
    if (['main-close-keeps-hidden-and-focus','again-close-keeps-hidden','again-escape-editor-keeps-caret','again-escape-inside-returns-check'].includes(scenario)) {
      f.edit('new accepted text');
      document.activeElement=scenario.startsWith('main')?els.analyze:els.analysisAgain;
      const work=f.start(scenario.startsWith('main'));
      const epoch=f.evaluate('analysisEpoch'), cache=f.evaluate('analysisResult');
      if (scenario.includes('escape-editor')) {
        els.prompt.focus(); els.prompt.setSelectionRange(2,7); f.sandbox.escapeAnalysis();
      } else if (scenario.includes('escape-inside')) f.sandbox.escapeAnalysis();
      else f.sandbox.closeAnalysis();
      f.hidden(); const focused=document.activeElement;
      assert.equal(focused,scenario.includes('escape-editor')?els.prompt:els.analyze);
      assert.equal(f.evaluate('analysisEpoch'),epoch); assert.equal(f.evaluate('analysisResult'),cache);
      f.reply(1,'background'); await work; f.hidden(); f.clean();
      assert.equal(document.activeElement,focused); assert.equal(f.evaluate('analysisResult.source'),'new accepted text');
      assert.equal(f.evaluate('analysisResult.body.marker'),'background');
      if (scenario.includes('escape-editor')) assert.deepEqual([els.prompt.selectionStart,els.prompt.selectionEnd],[2,7]);
    } else if (scenario==='later-explicit-reopens' || scenario==='same-prompt-cache-retained') {
      const cache=f.evaluate('analysisResult');
      if (scenario.startsWith('same')) f.evaluate("critiqueResult={body:{marker:'retained critique'}}");
      f.sandbox.closeAnalysis(); assert.equal(f.evaluate('analysisResult'),cache);
      const work=f.start(true); f.reply(1,'deliberate'); await work;
      assert.equal(els.promptAnalysis.hidden,false); assert.equal(document.activeElement,f.heading);
      assert.equal(f.evaluate('analysisResult.body.marker'),'deliberate');
      if (scenario.startsWith('same')) assert.equal(f.evaluate('critiqueResult.body.marker'),'retained critique');
      f.clean();
    } else if (scenario==='stale-background-does-not-enable-write') {
      f.edit('request text'); const work=f.start(); f.sandbox.closeAnalysis(); f.edit('later editor text');
      f.reply(1); await work; f.hidden();
      assert.equal(f.evaluate('analysisResult.source'),'request text'); assert.equal(els.prompt.value,'later editor text');
      assert.equal(f.evaluate('analysisStale'),true); assert.equal(f.write.disabled,true); f.clean();
    } else if (scenario==='newer-check-wins' || scenario==='old-finally-does-not-reset-newer') {
      f.edit('older request'); const old=f.start(); f.edit('newer request'); const current=f.start();
      if (scenario.startsWith('old-finally')) {
        f.reply(1,'old'); assert.equal(await old,null);
        assert.equal(els.analyze.getAttribute('aria-busy'),'true'); assert.notEqual(els.analyze.textContent,'Check');
        f.sandbox.closeAnalysis(); f.reply(2,'new'); await current; f.hidden();
      } else {
        f.reply(2,'new'); await current; const cache=f.evaluate('analysisResult');
        f.reply(1,'old'); assert.equal(await old,null); assert.equal(f.evaluate('analysisResult'),cache);
      }
      assert.equal(f.evaluate('analysisResult.source'),'newer request'); assert.equal(f.evaluate('analysisResult.body.marker'),'new'); f.clean();
    } else if (scenario==='clear-retires-pending') {
      f.edit('pending before Reset/model clear'); const work=f.start(); f.sandbox.clearAnalysis();
      f.reply(1); assert.equal(await work,null); f.hidden(); f.clean();
      assert.equal(f.evaluate('analysisResult'),null); assert.equal(f.evaluate('critiqueResult'),null);
      assert.equal(els.compareOwn.value,'');
    } else if (scenario==='ordinary-failure-cleanup') {
      f.edit('failed request'); const cache=f.evaluate('analysisResult'), work=f.start(); f.sandbox.closeAnalysis();
      calls.requests[1].reject(Error('synthetic analyze refusal')); assert.equal(await work,null);
      f.hidden(); f.clean(); assert.equal(f.evaluate('analysisResult'),cache);
      assert.equal(els.error.textContent,'synthetic analyze refusal'); assert.equal(document.activeElement,els.analyze);
    } else if (scenario.startsWith('critic-precheck-')) {
      f.edit('critic checked text'); document.activeElement=els.criticAsk;
      const work=f.sandbox.askCritic();
      assert.equal(els.analyze.disabled,true); assert.equal(els.analysisAgain.disabled,true);
      assert.equal(calls.requests[1].url,'/api/prompt/analyze');
      if (scenario.includes('clear')) f.sandbox.clearAnalysis(); else f.sandbox.closeAnalysis();
      f.reply(1,'critic precheck');
      if (scenario.includes('clear')) {
        await work; assert.equal(calls.requests.length,2); assert.equal(f.evaluate('analysisResult'),null);
      } else {
        await f.until(()=>calls.requests.length===3);
        assert.equal(calls.requests[2].url,'/api/prompt/critique');
        assert.equal(JSON.parse(calls.requests[2].options.body).prompt,'critic checked text');
        if (scenario.includes('error')) calls.requests[2].reject(Error('synthetic critic refusal'));
        else calls.requests[2].resolve({marker:'retained background critique'});
        await work;
        assert.equal(f.evaluate('analysisResult.source'),'critic checked text');
        if (scenario.includes('error')) {
          assert.equal(f.evaluate('critiqueFailure'),'synthetic critic refusal');
          assert.equal(els.error.textContent,'The critic did not answer: synthetic critic refusal.');
        } else {
          assert.equal(f.evaluate('critiqueResult.body.marker'),'retained background critique');
          assert.equal(f.evaluate('promptCritiqueReady'),true);
        }
        assert.equal(document.activeElement,els.prompt,'Close during disabled Check must keep the prompt focused');
      }
      f.hidden(); f.clean(); assert.equal(els.analyze.disabled,false); assert.equal(els.analysisAgain.disabled,false);
      assert.equal(f.evaluate('critiquing'),false);
    } else if (scenario==='visible-again-focus') {
      f.edit('visible Check again'); document.activeElement=els.analysisAgain;
      const work=f.start(); f.reply(1); await work;
      assert.equal(els.promptAnalysis.hidden,false); assert.equal(els.analysisAgain.hidden,true);
      assert.equal(document.activeElement,f.heading); f.clean();
    } else throw Error('unknown scenario '+scenario);
  }
  assert.deepEqual(calls.poisoned,[]); assert.equal(els.seed.value,'42');
  console.log('passed: '+scenario);
}
main().catch(error=>{console.error(error);process.exitCode=1;});
"""


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_check_dismissal(scenario: str, tmp_path: Path):
    node = shutil.which("node")
    assert node is not None, "Node is required for Check-dismissal ownership checks"
    done = subprocess.run(
        [node, "-e", SCRIPT, str(ROOT), scenario], cwd=tmp_path,
        capture_output=True, text=True, encoding="utf-8", timeout=20,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    assert f"passed: {scenario}" in done.stdout
