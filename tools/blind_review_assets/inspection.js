// The two stages share a source-pixel center. Clamp it against the larger
// viewport so that rounding a two-column grid never desynchronizes the pair.
function createInspection(onAvailability) {
  const stages = [...document.querySelectorAll('.stage')];
  const images = stages.map(stage => stage.querySelector('img'));
  const modes = ['fit', 'native', 'double'];
  let pair, mode = 'fit', cx, cy, scale, drag, generation = 0;

  function constrain(value, extent, view) {
    const half = Math.min(extent, view / scale) / 2;
    return Math.max(half, Math.min(extent - half, value));
  }

  function paint() {
    if (!pair || !stages[0].clientWidth) return;
    const widths = stages.map(stage => stage.clientWidth);
    const heights = stages.map(stage => stage.clientHeight);
    const fit = Math.min(Math.min(...widths) / pair.width, Math.min(...heights) / pair.height);
    scale = mode === 'fit' ? fit : (mode === 'double' ? 2 : 1) / window.devicePixelRatio;
    cx = constrain(cx, pair.width, Math.max(...widths));
    cy = constrain(cy, pair.height, Math.max(...heights));
    stages.forEach((stage, index) => {
      const image = images[index];
      image.style.width = pair.width * scale + 'px';
      image.style.height = pair.height * scale + 'px';
      image.style.left = Math.round((widths[index] / 2 - cx * scale) * devicePixelRatio) / devicePixelRatio + 'px';
      image.style.top = Math.round((heights[index] / 2 - cy * scale) * devicePixelRatio) / devicePixelRatio + 'px';
      stage.dataset.pan = String(scale > fit);
    });
    modes.forEach(name => document.getElementById(name).setAttribute('aria-pressed', String(name === mode)));
  }

  function chooseMode(name) {
    mode = name;
    paint();
  }
  modes.forEach(name => document.getElementById(name).onclick = () => chooseMode(name));
  for (const stage of stages) {
    stage.addEventListener('pointerdown', event => {
      if (stage.dataset.pan !== 'true' || event.button !== 0 || drag) return;
      drag = {id: event.pointerId, x: event.clientX, y: event.clientY, cx, cy};
      stage.setPointerCapture(event.pointerId);
    });
    stage.addEventListener('pointermove', event => {
      if (!drag || event.pointerId !== drag.id) return;
      cx = drag.cx - (event.clientX - drag.x) / scale;
      cy = drag.cy - (event.clientY - drag.y) / scale;
      paint();
    });
    stage.addEventListener('lostpointercapture', () => { drag = null; });
    stage.addEventListener('keydown', event => {
      const moves = {ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1]};
      if (stage.dataset.pan !== 'true' || !(event.key in moves)) return;
      event.preventDefault();
      const [dx, dy] = moves[event.key];
      cx += dx * 64 / scale;
      cy += dy * 64 / scale;
      paint();
    });
  }
  new ResizeObserver(paint).observe(stages[0]);
  window.addEventListener('resize', paint);
  return function inspect(next) {
    const thisLoad = ++generation;
    pair = next;
    drag = null;
    cx = pair.width / 2;
    cy = pair.height / 2;
    chooseMode('fit');
    onAvailability(false, '');
    const ready = new Set();
    for (const [index, side] of ['a', 'b'].entries()) {
      const image = images[index];
      image.hidden = true;
      // decode() rejects if a rapid pair change supersedes this load.
      image.src = next[side];
      image.decode().then(() => {
        if (thisLoad !== generation) return;
        if (image.naturalWidth !== pair.width || image.naturalHeight !== pair.height) {
          throw new Error('Image ' + side.toUpperCase() + ' has the wrong dimensions.');
        }
        ready.add(side);
        if (ready.size === 2) {
          images.forEach(member => member.hidden = false);
          onAvailability(true, '');
        }
      }).catch(error => {
        if (thisLoad === generation) {
          onAvailability(false, 'This pair could not be displayed. Check the pack files. ' + error.message);
        }
      });
    }
  };
}
