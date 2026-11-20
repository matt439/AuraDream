/* Injected by the gallery workflow benchmark into the repository Edge harness. */
'use strict';
window.flow = (() => {
  const samples = [];
  const cards = '#gallery .card[data-id]';
  const cells = '#sheet-grid .sheet-cell[data-id]';
  const wanted = (subject, starred) => {
    const residue = subject === 'lighthouse' ? 4 : 3;
    return Array.from({length: 5000}, (_, i) => 5000 - i).filter(id =>
      Math.floor((id - 1) / 20) % 5 === residue && (!starred || (id - 1) % 4 === 0)).slice(0, 48);
  };
  const lists = {search: wanted('lighthouse', false), starred: wanted('lighthouse', true),
    across: wanted('orchard', true), sheet: [4981, 4985, 4989, 4993, 4997]};
  const ids = selector => [...document.querySelectorAll(selector)].map(el => +el.dataset.id);
  const same = (a, b) => a.length === b.length && a.every((v, i) => v === b[i]);
  const inView = el => {
    const r = el.getBoundingClientRect();
    return r.height && r.width && r.bottom > 0 && r.top < innerHeight && r.right > 0 && r.left < innerWidth;
  };
  const paintedImages = selector => {
    const images = [...document.querySelectorAll(selector + ' img')].filter(inView);
    return images.length > 0 && images.every(img => img.complete && img.naturalWidth > 0 &&
      (selector !== cards || new URL(img.currentSrc).pathname.startsWith('/outputs/thumbnails/')));
  };
  const page = (name, count) => !els.gallery.hidden && els.count.textContent === '(' + count + ')' &&
    same(ids(cards), lists[name]) && galleryOthers === ({search: 2, starred: 1, across: 0})[name];
  const correct = {
    search: () => page('search', 1000),
    starred: () => page('starred', 250),
    sheet: () => !els.sheet.hidden && els.count.textContent === '(5)' &&
      same(ids(cells).sort((a, b) => a - b), lists.sheet),
    back: () => page('starred', 250),
    pick: () => els.selectCount.textContent === '1 selected' && document.querySelector(cards + '.selected')?.dataset.id === '4997',
    across: () => page('across', 250) && els.selectCount.textContent.includes('1 not shown'),
    add: () => els.selectCount.textContent.includes('2 selected') && els.selectCount.textContent.includes('1 not shown') &&
      document.querySelector(cards + '.selected')?.dataset.id === '4977',
    preview: () => els.moreSeedsDialog.open && els.moreSeedsScope.textContent === "From the 2 selected images (1 not in the gallery's view)." &&
      els.moreSeedsRun.textContent === 'Queue 8 images (2 jobs)' && els.moreSeedsRun.getAttribute('aria-disabled') === 'false' &&
      same(ids('#more-seeds-results .ms-result').sort((a, b) => a - b), [4977, 4997]),
    submit: () => !els.moreSeedsDialog.open && document.activeElement === els.moreSeeds,
  };
  const pictures = {search: cards, starred: cards, sheet: cells, back: cards, across: cards};
  let pending;
  performance.setResourceTimingBufferSize(16000);
  const frame = () => new Promise(resolve => requestAnimationFrame(resolve));
  async function waitFor(check, label) {
    const start = performance.now();
    while (!check()) {
      if (performance.now() - start > 15000) throw Error('Timed out: ' + label + ', count ' + els.count.textContent +
        ', picks ' + els.selectCount.textContent + ', search ' + els.search.value + ', sheet ' + ids(cells) +
        ', scope ' + els.moreSeedsScope.textContent + ', results ' + ids('#more-seeds-results .ms-result') +
        ', button ' + els.moreSeedsRun.textContent + ', preview ' + els.moreSeedsLine.textContent + ', gallery error ' + els.galleryError.textContent);
      await frame();
    }
    return true;
  }
  function arm(name, selector, eventName) {
    const el = document.querySelector(selector);
    if (!el) throw Error('Missing ' + selector);
    el.scrollIntoView({block: 'center', behavior: 'instant'});
    pending = new Promise((resolve, reject) => {
      const timer = setTimeout(() => reject(Error('Native ' + eventName + ' missed ' + selector)), 16000);
      el.addEventListener(eventName, async event => {
        clearTimeout(timer);
        try {
          if (!event.isTrusted) throw Error('Untrusted ' + eventName);
          const start = performance.now();
          await waitFor(correct[name], name);
          const dom = performance.now();
          const selector = pictures[name];
          if (selector) await waitFor(() => correct[name]() && paintedImages(selector), name + ' thumbnails');
          // Always cross a paint boundary, even when a click updates synchronously before the first rAF.
          await frame();
          await frame();
          // The compositor had an opportunity to paint the verified state.
          const end = performance.now();
          if (!correct[name]() || (selector && !paintedImages(selector))) throw Error(name + ' changed before paint');
          const api = performance.getEntriesByType('resource').filter(r => r.startTime >= start &&
            r.responseEnd <= end && new URL(r.name).pathname.startsWith('/api/') && r.initiatorType === 'fetch')
            .map(r => ({path: new URL(r.name).pathname, query: new URL(r.name).search,
              start: r.startTime - start, duration: r.duration}));
          samples.push({name, domMs: dom - start, ms: end - start, api, cards: ids(cards).length, cells: ids(cells).length});
          resolve(true);
        } catch (err) { reject(err); }
      }, {capture: true, once: true});
    });
  }
  return {arm, done: () => pending, samples, waitFor,
    ready: () => waitFor(() => document.getElementById('status-dot').className.includes('ready') &&
      !els.runList.disabled && ids(cards).length === 48, 'boot'),
    menuItem: () => {
      const item = [...document.querySelectorAll('#context-menu [role=menuitem]')].find(
        el => el.textContent.includes('Show run as sheet'));
      if (!item) throw Error('Missing Show run as sheet');
      item.dataset.benchmarkRun = 'true';
    }};
})();
