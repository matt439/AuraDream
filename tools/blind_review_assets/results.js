function readRevealKey(review) {
  return new Promise((resolve, reject) => {
    const script = document.createElement('script');
    script.src = 'key.js';
    script.onerror = () => reject(new Error('Could not read key.js beside this review.'));
    script.onload = () => {
      const key = window.REVIEW_KEY;
      const method = side => side && typeof side.method === 'string' && side.method.length &&
        Number.isFinite(side.seconds) && side.seconds >= 0;
      if (key?.id !== review.id || !review.pairs.every(({id}) => {
        const entry = key.pairs?.[id];
        return entry && typeof entry.family === 'string' && typeof entry.model === 'string' &&
          entry.seed != null && method(entry.a) && method(entry.b);
      })) {
        reject(new Error('The reveal key does not match this pack. Restore its original key.js.'));
      } else {
        resolve(key.pairs);
      }
    };
    document.head.append(script);
  });
}

function renderResults(review, key, state) {
  const results = document.getElementById('results');
  results.replaceChildren();
  const lines = ['AuraDream blind image review', 'Pack ' + review.id];
  const groups = new Map();
  review.pairs.forEach((pair, index) => {
    const {family} = key[pair.id];
    if (!groups.has(family)) groups.set(family, []);
    groups.get(family).push({pair, number: index + 1});
  });
  const sideLabel = side => side.method + ' (' + side.seconds.toFixed(3) + ' s)';
  function add(tag, parent, text) {
    const el = document.createElement(tag);
    el.textContent = text;
    parent.append(el);
    return el;
  }
  for (const [family, members] of groups) {
    const wins = new Map();
    let ties = 0;
    const rows = [];
    for (const {pair, number} of members) {
      const entry = key[pair.id], answer = state.answers[pair.id];
      for (const side of [entry.a, entry.b]) {
        if (!wins.has(side.method)) wins.set(side.method, 0);
      }
      let preferred = 'Tie';
      if (answer.vote === 'tie') {
        ties++;
      } else {
        const method = entry[answer.vote].method;
        wins.set(method, wins.get(method) + 1);
        preferred = answer.vote.toUpperCase() + ' · ' + method;
      }
      rows.push({number, pair, entry, preferred, note: answer.note});
    }
    add('h3', results, family);
    const counts = [...wins].map(([name, n]) => name + ' preferred ' + n).concat('Ties ' + ties).join(' · ');
    add('p', results, counts);
    lines.push('', family, counts);
    const wrapper = add('div', results, '');
    wrapper.className = 'table-wrap';
    const table = add('table', wrapper, '');
    table.setAttribute('aria-label', family + ' pair results');
    const head = table.createTHead().insertRow();
    for (const name of ['Pair', 'Checkpoint / seed', 'Image A', 'Image B', 'Preference', 'Notes']) {
      add('th', head, name).scope = 'col';
    }
    const body = table.createTBody();
    for (const {number, pair, entry, preferred, note} of rows) {
      const row = body.insertRow();
      const shape = pair.width + ' × ' + pair.height;
      const values = [number, entry.model + ' / ' + entry.seed, sideLabel(entry.a), sideLabel(entry.b), preferred, note || '—'];
      const scene = add('details', row.insertCell(), '');
      add('summary', scene, 'Pair ' + number);
      add('p', scene, shape + ' px');
      add('p', scene, pair.prompt);
      values.slice(1).forEach(value => { row.insertCell().textContent = value; });
      lines.push('', 'Pair ' + number + ' · ' + pair.id + ' · ' + shape + ' px', values[1], pair.prompt,
        'A · ' + values[2], 'B · ' + values[3], 'Preference · ' + preferred, 'Notes · ' + (note || '—'));
    }
  }
  return lines.join('\n') + '\n';
}
