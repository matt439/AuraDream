// AuraDream's page script: job cancellation and the progress event stream.
// Declarations only; state.js has the shared-global load-order contract.
'use strict';

// The server keeps whatever finished before the cancel landed, so this reports
// a count rather than treating the run as lost.
async function cancelJob() {
  if (!currentJob || els.cancel.getAttribute('aria-disabled') === 'true') return;
  els.cancel.setAttribute('aria-disabled', 'true');
  els.cancel.textContent = 'Stopping…';
  // This page's own press: its end is said as a plain cancel (finishJob).
  const jobId = currentJob;
  cancelledHere.add(jobId);
  try {
    await api(`/api/jobs/${jobId}/cancel`, { method: 'POST' });
    setProgressText('finishing this step…');
    mirrorJob();
  } catch (err) {
    cancelledHere.delete(jobId);
    showError(err.message);
    els.cancel.setAttribute('aria-disabled', 'false');
    els.cancel.textContent = 'Cancel';
  }
}

// What a queued job waits behind (`Job.waiting_on`), in the counter's words
// and, where it is not a job ahead running, the cell line's: a paused job
// keeps the gate, so a job queued from another tab read "waiting for the job
// ahead…" for as long as the pause lasted, and one submitted during another
// tab's critique named a job that did not exist. This page's own Expand and
// critic hold Generate, so either is another tab's.
const WAITING = {
  queue: { text: 'waiting for the job ahead…' },
  paused: {
    text: 'waiting: job ahead paused',
    // Asked at each redraw: how long the job ahead has held the gate
    // (`ahead_paused_at`), once a minute has gone.
    note: (job) => {
      const held = pausedFor(job?.ahead_paused_at);
      return `The job ahead ${held ? `has been paused for ${held}` : 'is paused'}: this one runs once it is resumed or cancelled, in the tab or window that started it.`;
    },
  },
  critic: { text: 'waiting for the critic…', note: "Another tab's prompt critic has the card; this job starts once it answers." },
  expand: { text: 'waiting for Expand…', note: "Another tab's prompt expansion has the card; this job starts once it is done." },
};

// `reattached` counts the times a stream that gave up found its job still
// running and opened again (gaveUp), so a stream that keeps failing on a live
// job stops trying and says so.
function streamJob(jobId, { reattached = 0 } = {}) {
  // The finish line on show is a job's before this one from here on, until
  // this one's own end replaces it - or for good, if its stream gives up.
  // Another tab's job never replaces it (finishJob), and gives back what it
  // was at its end.
  const staleBefore = finishStale;
  finishStale = true;
  syncFinishLead();
  // Watched: a queued submission's answer that lands after this does not
  // queue it, or take the page with it, again (startJob), and waits on this
  // stream instead (`jobStreams`, the first stream opened for the job, which
  // a reattached one's end resolves).
  watchedJobs.add(jobId);
  // Another tab's job, picked up over this page's own results: the progress
  // row only (isForeign).
  const foreign = isForeign(jobId);
  const stream = new Promise((resolve) => {
    const source = new EventSource(`/api/jobs/${jobId}/stream`);
    // Results replace the viewer only once the first image or frame of this
    // job lands, so the previous one stays up through the queue wait and the
    // model load - and the strip is then laid out for the whole job, the
    // frame in the box its image will take (viewer.js, viewerTile), so the
    // gallery under it does not move when the image replaces it.
    let cleared = false;
    let failures = 0;
    // Every snapshot carries the whole accumulated list, and a stream that
    // reattached missed the individual events, so images arrive by both routes.
    const shown = new Set();
    // The live preview: one tile at the end of the strip, its picture's src
    // replaced as frames arrive, gone the moment the real image lands. Never
    // a record. Or, for a pass, its stand-in (startJob's `placeholder`): the
    // source, dimmed, at the result's shape, captioned with the pass and its
    // step, from the job's first step until a frame or the result replaces
    // it - SDXL sends no frames, and the viewer showed an unrelated earlier
    // image for the whole of a pass.
    let previewEl = null;
    let caption = null;
    // Read again at each step until it is there: a listing can make a
    // submission the watched job before its POST's answer has registered the
    // stand-in, which then comes up at the first step after that answer.
    let placeholder = jobPlaceholders.get(jobId) ?? null;
    // The last status frame: what the job is (a grid's cells or a request's
    // repeats, its size, its budget) and what it waits on.
    let job = null;
    let executionDiagnostics = null;
    // The load's wait on the page's own clock, from the submission: the
    // server's `elapsed` starts only once the job runs, and minutes of a 0%
    // bar under "loading weights…" looked the same as a hang.
    let ticker = 0;
    const waitedFrom = submissions.get(jobId)?.at ?? Date.now();
    // Focus on the preview's tile (goToNewImage) goes with it to what takes
    // its place - the image, or the preview that follows a pass's stand-in -
    // and to Generate where nothing does, never to <body>.
    let previewFocused = false;
    const dropPreview = () => {
      if (previewEl) {
        previewFocused ||= previewEl.contains(document.activeElement);
        previewEl.remove();
        previewEl = null;
        settleStrip(false);
      }
      caption = null;
    };
    const focusReplacement = (el) => {
      if (!previewFocused) return;
      previewFocused = false;
      focusTile(el ?? els.generate);
    };
    const landed = () => {
      if (cleared || foreign) return;
      // What New image pointed at goes with the strip: offered again below
      // for what takes its place, where that is out of sight too.
      withdrawNewImage();
      clearResults(job);
      cleared = true;
      offerNewImage();
    };
    const show = (rec) => {
      markGoneSource(rec);
      if (shown.has(rec.id)) return;
      shown.add(rec.id);
      // This page takes it in at the job's end (finishJob): the status's
      // newest id moving to it is no other tab's (followGalleryHead).
      ownImageIds.add(rec.id);
      // A new CFG value for a ladder the lightbox has asked about.
      noteLadderImages([rec]);
      if (foreign) return;
      landed();
      dropPreview();
      appendResult(rec);
      focusReplacement(els.viewer.querySelector(`img[data-id="${rec.id}"]`));
    };
    // The clock in a part a screen reader is not told about: #progress-text
    // is a status, and a clock rewritten in it every second was read out
    // every second for the minutes a load takes. "loading weights" is said
    // once; the topbar's mirror and the tab's title take the whole text.
    const tick = () => {
      els.progressText.title = '';
      els.progressText.removeAttribute('aria-description');
      const s = Math.max(0, Math.floor((Date.now() - waitedFrom) / 1000));
      let clock = els.progressText.querySelector('.clock');
      if (!clock) {
        els.progressText.textContent = 'loading weights';
        clock = document.createElement('span');
        clock.className = 'clock';
        clock.setAttribute('aria-hidden', 'true');
        els.progressText.append(clock);
      }
      clock.textContent = ` · ${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
      // And in the viewer's line while the strip is empty (viewer.js).
      viewerWaiting(clock.textContent);
      mirrorJob();
    };
    const stopTicker = () => { clearInterval(ticker); ticker = 0; viewerWaiting(null); };
    const note = (data, extra = '') => showCell(data, {
      job,
      budget: job?.deadline && job.started_at ? job.deadline - job.started_at : null,
      note: extra,
      jobId,
    });
    const standIn = () => {
      landed();
      const { figure, tile } = viewerTile('pending', placeholder.width, placeholder.height);
      const img = document.createElement('img');
      img.src = placeholder.src;
      img.alt = '';
      // Where the result's tools will be.
      caption = document.createElement('p');
      caption.className = 'foot';
      tile.append(img, caption);
      previewEl = figure;
      stripAppend(figure);
    };

    source.onmessage = (event) => {
      // Back after a reconnect: the strip's note is its own again.
      if (streamLost) { streamLost = false; noteViewer(); }
      failures = 0;
      const data = JSON.parse(event.data);
      if (['status', 'progress', 'done'].includes(data.type)) {
        executionDiagnostics = mergeExecutionDiagnostics(executionDiagnostics, data.execution_diagnostics);
        data.execution_diagnostics = executionDiagnostics;
      }
      const runs = data.total_runs || 1;
      if (data.type === 'progress') {
        stopTicker();
        setIndeterminate(false);
        const done = (data.run || 1) - 1;
        const within = data.step / data.total_steps;
        setProgress((done + within) / runs);
        // A single image's step count is shown, not said: "1/12" to
        // "12/12" was read out a step at a time. The run counter is said
        // as it changes, and the bar's aria-valuenow has the fraction.
        if (runs > 1) setExecutionProgress(jobId, executionDiagnostics, `${unitWord(job)} ${data.run}/${runs}`);
        else setExecutionProgress(jobId, executionDiagnostics, 'denoising…', `${data.step}/${data.total_steps}`);
        setStep(runs > 1, within, `${data.step}/${data.total_steps}`);
        note(data);
        placeholder ??= jobPlaceholders.get(jobId) ?? null;
        if (placeholder && !cleared && !previewEl) standIn();
        if (caption) {
          caption.textContent = `${placeholder.label} · step ${data.step}/${data.total_steps}`
            + (jobEta ? ` · ${jobEta.replace(/^~/, '')}` : '');
        }
        mirrorJob();
      } else if (data.type === 'image') {
        show(data.image);
      } else if (data.type === 'preview') {
        // A rough picture of the cell so far, from the latents: blurry, but
        // the composition is there from a few steps in, which is soon enough
        // to cancel a bad prompt rather than wait the render out. Not over
        // this page's own results for another tab's job.
        if (foreign) return;
        if (caption) dropPreview();
        landed();
        if (!previewEl) {
          previewEl = previewTile(job?.width, job?.height);
          // At the strip's end, which a long run's scrolling strip follows
          // from the preview on (viewer.js).
          stripAppend(previewEl);
          focusReplacement(previewEl);
        }
        const frame = previewEl.querySelector('img');
        frame.src = data.image;
        frame.title = `live preview · step ${data.step}/${data.total_steps}`;
        previewEl.querySelector('.foot').textContent = `Live preview · step ${data.step}/${data.total_steps}`;
      } else if (data.type === 'status') {
        // Either the snapshot every stream opens with or the start of a run;
        // both carry whatever landed while nobody was listening.
        job = data;
        if (data.images) data.images.forEach(show);
        // A queued job says what it is behind: the weights on their way, or
        // another job. The first read "waiting for the job ahead…" for the
        // whole of a 32 GB download, under a top bar saying the opposite.
        const loading = data.status === 'queued' && data.waiting_on === 'load';
        const unit = unitWord(data);
        setIndeterminate(loading);
        // Paused, the counter stays the counter - "paused — GPU handed back"
        // in its place squeezed the bar to 5 px; the cell line says it.
        // Loading, the ticker below writes it, and leaves "loading weights"
        // as it is when it is there already.
        // Behind the gate, what holds it: a job ahead, one ahead that is
        // paused - which keeps the gate until it is resumed or cancelled,
        // where it was started - or another tab's critic or Expand, with no
        // job ahead at all (`waiting_on`).
        const waiting = data.status !== 'queued' ? null
          : data.waiting_on == null ? { text: 'queued…' } : WAITING[data.waiting_on] ?? WAITING.queue;
        if (!loading) {
          if (data.paused) setExecutionProgress(jobId, null, runs > 1 && data.run ? `${unit} ${data.run}/${runs}` : 'paused');
          else if (waiting) setExecutionProgress(jobId, null, waiting.text);
          else if (runs > 1 && data.run) setExecutionProgress(jobId, executionDiagnostics, `${unit} ${data.run}/${runs}`);
          // A single image's counter is the step count from here on, shown
          // beside these words rather than said (the progress branch).
          else setExecutionProgress(jobId, executionDiagnostics, 'denoising…', 'denoising…');
        }
        syncPause(data);
        syncNow(data);
        // A run boundary, so the steps bar starts over rather than sitting full
        // until the first callback of the next run lands.
        setStep(runs > 1 && data.status !== 'queued' && !!data.run);
        // What it waits on, in words, carried with the cell's own so a
        // redraw says it again (showCell).
        note(data, loading ? 'The first load of a session takes a few minutes.' : waiting?.note ?? '');
        if (loading) {
          if (!ticker) ticker = setInterval(tick, 1000);
          tick();
        } else {
          stopTicker();
        }
        mirrorJob();
      } else if (data.type === 'done') {
        source.close();
        stopTicker();
        const submitted = submissions.get(jobId);
        const sheet = jobsToShow.has(jobId);
        forgetJob(jobId);
        setBusy(false);
        dropPreview();
        if (data.images) data.images.forEach(show);
        focusReplacement(null);
        // Cancelled or failed after its first frame, before any image: the
        // frame that offered New image has gone, and the button offered an
        // image there was none of. A job that never reached the strip left
        // it as it was, and what New image points at with it.
        if (cleared && !shown.size) withdrawNewImage();
        // Watched to its end, its finish line has said it: a reload's banner
        // would only say it again.
        remember(ANNOUNCED_KEY, jobId);
        finishJob(jobId, data, { submitted, sheet, foreign, staleBefore });
        resolve();
        watchNext();
      }
    };

    // The stream has given up - five reconnects, or an answer that was no
    // stream - and whether the job went with it is the server's to say: "the
    // run carries on - reload to pick it up" was false when the server had
    // died, and after a restart a reload found nothing at all.
    const gaveUp = async () => {
      dropPreview();
      focusReplacement(null);
      let alive = null;
      let failure = null;
      try { alive = await api(`/api/jobs/${jobId}`); } catch (err) { failure = err; }
      // Still there: watch it again - events() opens with `done` for one
      // that ended meanwhile.
      if (alive && reattached < 3) return streamJob(jobId, { reattached: reattached + 1 });
      const grid = job?.grid;
      const submitted = submissions.get(jobId);
      const runId = job?.run_id ?? submitted?.run_id;
      forgetJob(jobId);
      setBusy(false);
      // The strip's note was the reconnect's line; it is the strip's again.
      noteViewer();
      const made = shown.size;
      if (cleared && !made) withdrawNewImage();
      const saved = made
        ? `The ${plural(made, 'image')} finished so far ${made === 1 ? 'is' : 'are'} saved.`
        : 'Nothing of it had finished.';
      // What the viewer says where nothing of the job reached it (lostRun).
      let lost = null;
      if (alive) {
        showError('This job is still running, but its progress keeps getting lost: '
          + `reload the page to follow it. ${saved}`);
      } else if (failure.offline) {
        // The status poll's offline state takes it from here, and its first
        // answer takes this back.
        const text = `Can't reach the server — it may have stopped. ${saved} Start it again; the page will notice.`;
        showError(text, false, { offline: true });
        lost = { heading: 'Lost the server', text };
        void refreshStatus();
      } else if (failure.status === 404) {
        const planned = job?.planned_images;
        const text = made
          ? `The server restarted, and this job ended with it after ${made}${planned ? ` of ${planned}` : ''} `
            + `image${(planned ?? made) === 1 ? '' : 's'}. They are in the gallery; `
            + (runId ? 'Open this run from Runs and use Finish run for the rest.'
              : `Use ${grid ? "the panel's Resume tick" : 'Generate'} for this older job.`)
          : 'The server restarted, and this job ended with it before any of it had finished.';
        showError(text);
        lost = { heading: 'The server restarted', text };
        void refreshStatus();
      } else {
        const text = `Lost this job's progress: ${failure.message}`;
        showError(text);
        lost = { heading: 'Lost this job', text };
      }
      // In the topbar's words where the server is what went.
      holdTitle(failure?.offline ? 'Server not answering' : 'Lost the job');
      // Its first frame had cleared the strip and nothing of it landed: the
      // viewer was an empty box, the only words red in the panel.
      if (lost && cleared && !made) showLostRun({ ...lost, run: runId }, submitted);
      else if (lost && runId) setFinish([
        finishRunButton(runId, { className: 'ghost tiny', slot: els.error }),
      ]);
      // With another tab's images owed since this job took over (finishJob);
      // offline, the poll's first answer takes everything in (refreshStatus).
      if (!failure?.offline && (made || elsewhereCatchUp)) {
        if (elsewhereCatchUp) noteUnseenImages();
        elsewhereCatchUp = false;
        void prependFinished(made);
        loadRuns();
      }
    };

    source.onerror = () => {
      // Whatever the stream does next, it is not the load ticking on: the
      // ticker wrote "loading weights" over "reconnecting…" every second
      // while the topbar said the server was not answering. The snapshot a
      // reconnect opens with starts it again if the load goes on.
      stopTicker();
      // EventSource reconnects on its own, and the server replays a snapshot to
      // whoever attaches; close() cancels that retry for good, which is what
      // used to strand a running sweep behind an idle-looking page. Ride out a
      // few failures before asking the server what became of the job.
      if (source.readyState === EventSource.CONNECTING && ++failures <= 5) {
        // In the topbar's words, and nothing beside them that is no longer
        // true: the progress row said "reconnecting… (2)" under the topbar's
        // "server not answering", the step count and "~19s left" frozen
        // beside it, and the tab a third thing. The next frame that lands
        // brings the counts back (`showCell`, `setStep`, `noteViewer`).
        setProgressText(`${LOST_PHRASE} (${failures})`);
        setStep(false);
        jobEta = '';
        els.cellNote.textContent = '';
        if (failures === 1) {
          els.viewerNoteText.replaceChildren(`${LOST_PHRASE[0].toUpperCase()}${LOST_PHRASE.slice(1)}…`);
          void refreshStatus();
        }
        streamLost = true;
        mirrorJob();
        return;
      }
      source.close();
      void gaveUp().then(resolve);
    };
  });
  if (!jobStreams.has(jobId)) {
    jobStreams.set(jobId, stream);
    void stream.finally(() => { if (jobStreams.get(jobId) === stream) jobStreams.delete(jobId); });
  }
  return stream;
}
