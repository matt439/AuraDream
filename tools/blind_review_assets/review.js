(() => {
  'use strict';
  const el = id => document.getElementById(id);
  function notice(message) {
    el('notice').textContent = message;
    el('notice').hidden = !message;
  }
  const review = window.REVIEW;
  if (!review?.id || !Array.isArray(review.pairs) || !review.pairs.length) {
    notice('Review data could not be opened. Keep index.html and data.js together in the original pack folder.');
    return;
  }
  const pairs = review.pairs;
  const ids = new Set(pairs.map(pair => pair.id));
  const storageKey = 'auradream.blind-review.v1.' + review.id;
  const votes = ['a', 'tie', 'b'];
  let state = {version: 1, pack: review.id, current: pairs[0].id, revealed: false, answers: {}};
  let lastSaved = null, memoryOnly = false, stale = false, ready = false, key = null;
  let resultText = '', imageError = '';
  const index = () => pairs.findIndex(pair => pair.id === state.current);
  const answer = id => state.answers[id] || {vote: null, note: ''};
  const judged = () => pairs.filter(pair => answer(pair.id).vote).length;
  function validate(value) {
    if (value?.version !== 1 || value.pack !== review.id) throw new Error('This backup belongs to another pack or version.');
    if (!ids.has(value.current) || typeof value.revealed !== 'boolean' ||
        !value.answers || typeof value.answers !== 'object' || Array.isArray(value.answers)) {
      throw new Error('The saved review is incomplete.');
    }
    for (const [id, row] of Object.entries(value.answers)) {
      if (!ids.has(id) || !row || !(row.vote === null || votes.includes(row.vote)) || typeof row.note !== 'string') {
        throw new Error('The saved review contains an invalid answer.');
      }
    }
    if (value.revealed && !pairs.every(({id}) => value.answers[id]?.vote)) throw new Error('A revealed review needs every vote.');
    return value;
  }
  try {
    lastSaved = localStorage.getItem(storageKey);
  } catch {
    memoryOnly = true;
  }
  if (lastSaved !== null) {
    try { state = validate(JSON.parse(lastSaved)); }
    catch (error) {
      notice(error.message + ' Existing browser data was preserved. Open a copy of the pack at a new location and restore your saved backup.');
      return;
    }
  }
  function saveStatus() {
    el('save-status').textContent = stale ? 'Another tab changed this review. Reload to use its saved version.' :
      memoryOnly ? 'Browser storage is unavailable. Save a backup before closing this page.' :
      'This browser keeps your place. Save a backup before moving the pack or clearing browser data.';
  }
  function freeze() {
    stale = true;
    saveStatus();
    refresh();
    notice('Another tab changed the saved review. Reload before editing. Save backup exports this tab’s last accepted state.');
  }
  window.addEventListener('storage', event => {
    if (!memoryOnly && (event.key === null || event.key === storageKey)) freeze();
  });
  function commit(next) {
    if (stale) return false;
    if (!memoryOnly) {
      try {
        if (localStorage.getItem(storageKey) !== lastSaved) { freeze(); return false; }
        const serialized = JSON.stringify(next);
        localStorage.setItem(storageKey, serialized);
        lastSaved = serialized;
      } catch {
        memoryOnly = true;
        saveStatus();
        notice('Browser storage is unavailable. Use Save backup to keep this review.');
      }
    }
    state = next;
    return true;
  }
  function hold(button, reason) {
    button.setAttribute('aria-disabled', String(Boolean(reason)));
    button.title = reason;
  }
  function ifHeld(button) {
    if (button.getAttribute('aria-disabled') !== 'true') return false;
    notice(button.title);
    return true;
  }
  pairs.forEach((pair, i) => el('pair').add(new Option(String(i + 1), pair.id)));
  function refresh() {
    const n = judged();
    const editHold = stale ? 'Reload to get the latest saved review.' : state.revealed ? 'Judgments were locked at reveal.' : '';
    el('count').textContent = n + ' of ' + pairs.length + ' judged';
    el('progress').max = pairs.length;
    el('progress').value = n;
    el('pair').value = state.current;
    pairs.forEach((pair, i) => {
      el('pair').options[i].textContent = (i + 1) + (answer(pair.id).vote ? ' ✓' : '');
    });
    for (const vote of votes) {
      const button = el('vote-' + vote);
      button.setAttribute('aria-pressed', String(answer(state.current).vote === vote));
      hold(button, editHold || (ready ? '' : imageError || 'Wait for both images to open.'));
    }
    el('note').readOnly = Boolean(editHold);
    el('note').value = answer(state.current).note;
    hold(el('previous'), index() === 0 ? 'This is the first pair.' : '');
    hold(el('next'), index() === pairs.length - 1 ? 'This is the last pair.' : '');
    hold(el('import'), editHold);
    hold(el('reveal'), stale ? editHold : state.revealed && key ? 'Results are shown below.' :
      n < pairs.length ? 'Judge all ' + pairs.length + ' pairs before revealing the methods.' : '');
    el('reveal').textContent = state.revealed ? 'Show results' : 'Reveal methods & times…';
  }

  const inspect = createInspection((available, error) => {
    ready = available;
    imageError = error;
    if (error) notice(error);
    refresh();
  });
  function display() {
    const pair = pairs[index()];
    el('prompt').textContent = pair.prompt;
    el('dimensions').textContent = pair.width + ' × ' + pair.height + ' px';
    inspect(pair);
  }
  function go(id) {
    notice('');
    if (!ids.has(id)) return;
    // After a reveal, navigation is still available for inspecting the pairs.
    if (commit({...state, current: id})) display();
    else refresh();
  }
  el('pair').onchange = () => go(el('pair').value);
  for (const [button, step] of [['previous', -1], ['next', 1]]) {
    el(button).onclick = () => { if (!ifHeld(el(button))) go(pairs[index() + step].id); };
  }
  function edit(partial) {
    if (state.revealed || stale) return;
    const row = {...answer(state.current), ...partial};
    if (commit({...state, answers: {...state.answers, [state.current]: row}})) refresh();
  }
  votes.forEach(vote => {
    const button = el('vote-' + vote);
    button.onclick = () => {
      if (!ifHeld(button)) { notice(''); edit({vote}); }
    };
  });
  el('note').oninput = () => edit({note: el('note').value});

  function download(name, content, type) {
    const link = document.createElement('a');
    const url = URL.createObjectURL(new Blob([content], {type}));
    link.href = url;
    link.download = name;
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 30000);
  }
  el('export').onclick = () => download('blind-review-' + review.id + '.json', JSON.stringify(state, null, 2), 'application/json');
  el('import').onclick = () => { if (!ifHeld(el('import'))) el('import-file').click(); };

  // Restore only compatible answers. A stale backup can fill missing fields
  // but cannot erase or replace a decision or a note written since export.
  function merge(incoming) {
    const answers = {...state.answers};
    for (const [id, row] of Object.entries(incoming.answers)) {
      const existing = answer(id);
      for (const field of ['vote', 'note']) {
        if (existing[field] && (row[field] || incoming.revealed) && existing[field] !== row[field]) {
          throw new Error('Backup conflicts with pair ' + (pairs.findIndex(pair => pair.id === id) + 1) + '. Current review kept.');
        }
      }
      answers[id] = {vote: existing.vote || row.vote, note: existing.note || row.note};
    }
    return {...state, answers, current: incoming.current, revealed: incoming.revealed};
  }
  el('import-file').onchange = async () => {
    const file = el('import-file').files[0];
    if (!file) return;
    el('import-file').value = '';
    try {
      const incoming = validate(JSON.parse(await file.text()));
      if (state.revealed || stale) throw new Error('The current review is locked.');
      const merged = merge(incoming);
      if (incoming.revealed && !window.confirm('This backup was already revealed. Restore its locked judgments and display the methods?')) return;
      if (!commit(merged)) return;
      display();
      notice('Backup restored.');
      if (state.revealed) await showResults();
    } catch (error) { notice(error.message); }
  };
  let loadingKey = false;
  async function showResults() {
    if (stale || loadingKey || judged() !== pairs.length) return;
    // Persist the lock before requesting any identifying information, even
    // if the key then fails to load or the browser closes during this step.
    if (!state.revealed && !commit({...state, revealed: true})) return;
    refresh();
    loadingKey = true;
    try {
      key = await readRevealKey(review);
      resultText = renderResults(review, key, state);
      el('summary').hidden = false;
      refresh();
      notice('');
    } catch (error) { notice(error.message); }
    finally { loadingKey = false; }
  }
  el('reveal').onclick = () => {
    if (ifHeld(el('reveal'))) return;
    if (state.revealed) void showResults();
    else el('reveal-dialog').showModal();
  };
  el('reveal-dialog').onclose = () => {
    if (el('reveal-dialog').returnValue === 'reveal') void showResults();
    el('reveal-dialog').returnValue = '';
  };
  el('summary-text').onclick = () => download('blind-review-' + review.id + '-results.txt', resultText, 'text/plain;charset=utf-8');
  el('workspace').hidden = false;
  display();
  saveStatus();
  if (state.revealed) void showResults();
})();
