// Local Diffusion's page script: prompt token counts and reading limits.
// Declarations only; state.js has the shared-global load-order contract.
'use strict';

// --------------------------------------------------------------- the cut

// Where the encoders stop reading the prompt box, in its own units, with the
// encoders that stop there: from the tray's measure of this text, else a
// Check of it, else the last measure while the text up to its cut is
// unchanged - a tag typed at the end of a long list is past the cut before
// the next count lands. Null where nothing counted the text, or nothing that
// feeds the image's words cuts it (FLUX.1's CLIP-L feeds one pooled vector).
function tagCut() {
  const raw = els.prompt.value;
  const text = raw.trim();
  const measured = tagMeasured?.model === loadedModel ? tagMeasured : null;
  let counted = null;
  if (measured?.text === text) counted = measured;
  else if (analysisResult?.source === text) counted = { text, encoders: analysisResult.body.encoders };
  else if (measured) counted = measured;
  const cuts = (counted?.encoders ?? [])
    .filter((report) => report.field === 'prompt' && report.cut != null && !report.pooled_only);
  if (!cuts.length) return null;
  const at = Math.min(...cuts.map((report) => report.cut));
  const prefix = counted.text.slice(0, units(counted.text, at));
  if (!text.startsWith(prefix)) return null;
  const stop = cuts.filter((report) => report.cut === at);
  return {
    offset: raw.length - raw.trimStart().length + prefix.length,
    names: stop.map((report) => report.name).join(' and '),
    budget: stop[0].budget,
  };
}

// The count the label shows: the tightest encoder of the last measure, or
// "cut unknown" where an encoder went uncounted and none is known to cut -
// a tokenizer still loading is never "no cut". `tight` is that encoder's
// report, for the line under the prompt to name.
function tagTokenLine() {
  if (!tagMeasured || tagMeasured.model !== loadedModel) return null;
  const reports = tagMeasured.encoders.filter((report) => report.field === 'prompt' && !report.pooled_only);
  const cut = reports.some((report) => report.cut != null);
  // Past the length every request takes (`PROMPT_MAX_CHARS`), what was
  // counted is its first part (scheduleTagMeasure), and the length is the
  // first thing to say: Generate and Check refuse the prompt whole.
  const long = tagMeasured.truncated ? `over ${PROMPT_MAX_CHARS} characters` : '';
  if (!reports.length || (tagMeasured.detail && !cut)) {
    return long ? { text: long, over: true, tight: null } : { text: 'cut unknown', over: false, tight: null };
  }
  const tight = reports.reduce((a, b) => (b.tokens / b.budget > a.tokens / a.budget ? b : a));
  const count = `${tight.tokens}/${tight.budget} tokens`;
  return { text: long ? `${long} · ${count} in the first ${PROMPT_MAX_CHARS}` : count, over: cut || !!long, tight };
}

// The same count as a line under the prompt box (#prompt-tokens) where no
// chips show it - a checkpoint without them, whose prompt is prose, XML or
// JSON, and a tag checkpoint's sentence - which had no count before Check:
// NewBie's own two-character XML example is 488 of Gemma's 511 tokens, and a
// third character's lines cut <general_tags> off unread. Shown once the
// prompt has used three quarters of the tightest encoder's budget, past it,
// or where an encoder went uncounted ("cut unknown", never "no cut") - at any
// length, there being no count to hold to the three quarters: a tokenizer the
// server cannot fetch or is still fetching, the server not answering (asked
// again when it answers, status.js). Not where the encoder table has no
// entry for the checkpoint at all (`encoders_known` false): nothing will ever
// count its prompt, "cut unknown" stood under every prompt, two words
// included, and Check's Encoders section says why. Hidden is not "no cut".
// Under three quarters it would say only that there is room, and at 1400x900
// its line put Generate under the window's edge. The encoders named, and
// where reading stops when it does (`tagCut`, while the text up to the cut is
// the counted one); the title has every encoder's count. The line is no live
// region - it is rewritten at every pause in typing - and a change to past
// the limit is said, once, in the chips' summary line, which no chips are
// using.
const TOKENS_NEAR = 0.75;
let promptTokensOver = false;
// What the summary line last said about it, so a new reason past the limit -
// the length after the token cut - is said too.
let promptTokensSaid = '';
// Every request's prompt length (GenerateRequest, PromptMeasureRequest:
// `max_length`, held equal by tests/test_api_analysis.py). The measure is
// asked of the first this many code points, as Python counts them: past it
// the whole was refused, and the line said "cut unknown", in its quiet
// colour, taking back the warning the prompt had already earned.
const PROMPT_MAX_CHARS = 4000;
const LENGTH_SAID = `The prompt is past ${PROMPT_MAX_CHARS} characters, which Generate and Check refuse.`;

function promptTokenLine() {
  const tokens = tagTokenLine();
  if (!tokens) return null;
  const long = !!tagMeasured.truncated;
  if (!tokens.tight && tagMeasured.known === false && !tagMeasured.failed && !long) {
    return { text: '', over: false, shown: false, said: '', title: '' };
  }
  if (!tokens.tight && long) {
    const text = `Over ${PROMPT_MAX_CHARS} characters: Generate and Check refuse it`;
    return { text, over: true, shown: true, said: LENGTH_SAID, title: `${text}.` };
  }
  if (!tokens.tight) {
    return {
      text: 'Tokens: cut unknown', over: false, shown: true, said: '',
      title: tagMeasured.detail || 'No tokenizer is known for this checkpoint, so nothing was counted.',
    };
  }
  const { tight } = tokens;
  // The line is cut to one; its title is the whole of it, then every encoder.
  const counts = tagMeasured.encoders.filter((report) => report.field === 'prompt')
    .map((report) => `${report.name} ${report.tokens}/${report.budget}${report.pooled_only ? ' (one pooled vector)' : ''}`);
  const titled = (text) => `${text}\n${counts.join(' · ')}: counted by this checkpoint's own tokenizers, as Check counts`;
  const cut = tokens.over ? tagCut() : null;
  // Over the length: said first, the count of what was counted after it.
  if (long) {
    const where = cut ? ` · ${cut.names} ${cut.names.includes(' and ') ? 'stop' : 'stops'} reading at `
      + `"${excerpt(els.prompt.value.slice(cut.offset))}"` : '';
    const text = `Over ${PROMPT_MAX_CHARS} characters: Generate and Check refuse it · `
      + `${tight.tokens}/${tight.budget} tokens in the first ${PROMPT_MAX_CHARS}${where}`;
    return { text, over: true, shown: true, title: titled(text), said: LENGTH_SAID };
  }
  if (!cut) {
    // Every encoder as tight as the one counted: SDXL's two CLIP towers
    // read the same 75 tokens, and naming CLIP-L alone said only one did.
    const names = tagMeasured.encoders
      .filter((r) => r.field === 'prompt' && !r.pooled_only && r.tokens * tight.budget === tight.tokens * r.budget)
      .map((r) => r.name).join(' and ');
    const text = `${tokens.text} · ${names}`;
    return {
      text, over: tokens.over, shown: tokens.over || tight.tokens >= TOKENS_NEAR * tight.budget,
      title: titled(text), said: '',
    };
  }
  const stops = cut.names.includes(' and ') ? 'stop' : 'stops';
  const text = `${tokens.text} · ${cut.names} ${stops} reading at "${excerpt(els.prompt.value.slice(cut.offset))}"`;
  return {
    text,
    over: true,
    shown: true,
    title: titled(text),
    said: `The prompt is past the token limit: ${cut.names} ${stops} reading at ${cut.budget} tokens.`,
  };
}

// Whether the line shows moves everything under it in the panel by a line -
// Generate, Pause and Cancel, Run grid, the negative box - and that change
// lands a pause and a round trip after the typing stopped, as the pointer may
// be arriving on Generate. So while the mouse is on a control the line would
// move (`promptTokensAimed`, app.js: one whose top is at or below the prompt
// box's bottom) the change waits (`promptTokensHeld`) for the mouse to leave
// it or for a key, which no press of the mouse is; the text and the colour
// change in place, the line being one line. Only on such a control: held for
// the whole lower panel, a prompt past the limit went unsaid until the mouse
// left it, and the mouse goes to Generate.
let promptTokensAimed = false;
let promptTokensHeld = false;

// The line's one writer: hidden while it has nothing to say, and while the
// chips show - their label has the count, and under 901 px they come
// straight after it in the panel. `now` gives a held change (a key).
function syncPromptTokens(now = false) {
  const line = els.promptTokens;
  const chips = !els.tagValidation.hidden;
  const count = !chips && els.prompt.value.trim() ? promptTokenLine() : null;
  const shown = !!count?.shown;
  promptTokensHeld = !now && shown === line.hidden && promptTokensAimed;
  if (promptTokensHeld) return;
  const text = shown ? count.text : '';
  if (line.textContent !== text) line.textContent = text;
  line.title = shown ? count.title : '';
  line.classList.toggle('warn', shown && count.over);
  line.hidden = !shown;
  // Said on the way past the limit, and taken back on the way under it - not
  // read out, an emptied live region says nothing - so the next crossing is
  // said again.
  const over = shown && count.over;
  if (over && count.said && count.said !== promptTokensSaid) els.tagSummary.textContent = count.said;
  else if (!over && promptTokensOver && els.tagSummary.textContent.startsWith('The prompt is past')) {
    els.tagSummary.textContent = '';
  }
  promptTokensOver = over;
  promptTokensSaid = over ? count.said || promptTokensSaid : '';
}

// A measure that got no answer (`failed`: the server not answering, a refusal)
// is asked again for the same text - status.js asks once the server answers
// again - where an answer, "cut unknown" included, stands until the text or
// the checkpoint changes.
function scheduleTagMeasure(delay) {
  clearTimeout(tagMeasureTimer);
  const epoch = ++tagMeasureEpoch;
  const text = els.prompt.value.trim();
  if (!text || (tagMeasured?.text === text && tagMeasured.model === loadedModel && !tagMeasured.failed)) return;
  // The first PROMPT_MAX_CHARS code points: a cut there still lands, where
  // the whole prompt was refused, and the line says the length is over.
  const points = [...text];
  const truncated = points.length > PROMPT_MAX_CHARS;
  const sent = truncated ? points.slice(0, PROMPT_MAX_CHARS).join('') : text;
  tagMeasureTimer = setTimeout(async () => {
    if (modelChangeHold()) return;
    const model = loadedModel;
    try {
      const body = await api('/api/prompt/measure', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: sent }),
      });
      if (epoch !== tagMeasureEpoch) return;
      tagMeasured = {
        model, text, encoders: body.encoders || [], detail: body.encoders_detail || null,
        known: body.encoders_known !== false, truncated,
      };
    } catch (error) {
      if (epoch !== tagMeasureEpoch) return;
      tagMeasured = { model, text, encoders: [], detail: error.message, failed: true, truncated };
    }
    // An open editor is not rebuilt under the user's typing: the label
    // changes now, the chips with the editor's close. The line under the
    // prompt has the count either way.
    if (tagInsertAt !== null || tagEditAt !== null) syncTagTray(lastTagValidationResults, lastTagValidationState);
    else if (!els.tagValidation.hidden) {
      renderTagValidation(lastTagValidationResults, lastTagValidationState, lastTagValidationDetail);
    }
    syncPromptTokens();
  }, delay === 0 ? 0 : 600);
}
