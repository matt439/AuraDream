// Loaded only by the external trial browser. The app handles every user action.
window.trialUntil = async (predicate, timeout) => {
  const deadline = performance.now() + timeout;
  do {
    if (await predicate()) return true;
    await new Promise(resolve => setTimeout(resolve, 75));
  } while (performance.now() < deadline);
  return false;
};
window.trialHit = selector => {
  const element = document.querySelector(selector);
  const r = element.getBoundingClientRect();
  return element.contains(document.elementFromPoint(r.x + r.width / 2, r.y + r.height / 2));
};
window.trialSet = (selector, value, event = 'input') => {
  const element = document.querySelector(selector);
  element.value = value;
  element.dispatchEvent(new Event(event, {bubbles: true}));
};
window.trialMarkEditItem = () => {
  const item = [...els.contextMenu.querySelectorAll('[role=menuitem]')]
    .find(el => el.textContent.includes('Edit by instruction'));
  if (!item || els.contextMenu.hidden) return false;
  item.dataset.trialEditItem = '';
  return true;
};
window.trialGet = async path => {
  const response = await fetch(path);
  if (!response.ok) throw Error(path + ' returned ' + response.status);
  return response.json();
};
window.trialPost = async (path, body) => {
  const response = await fetch(path, {method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(body)});
  return {status: response.status, data: await response.json()};
};
window.trialRefusals = {};
window.trialRefused = async (path, body, words) => {
  const answer = await trialPost(path, body);
  trialRefusals[path + ' ' + JSON.stringify(body)] = answer;
  return answer.status === 422 && words.test(JSON.stringify(answer.data.detail));
};
window.trialStatus = async () => {
  const status = await trialGet('/api/status');
  if (status.state === 'error') throw Error(JSON.stringify(status));
  return status.state;
};
window.trialJobs = () => trialGet('/api/jobs?include_finished=true');
window.trialRememberJobs = async () => {
  window.trialBefore = new Set((await trialJobs()).map(job => job.job_id));
  window.trialJob = null;
};
window.trialNewJob = async () => {
  const added = (await trialJobs()).filter(job => !trialBefore.has(job.job_id));
  if (added.length > 1) throw Error('One browser action created multiple jobs');
  window.trialJob = added[0]?.job_id;
  return trialJob;
};
window.trialNoNewJobs = async () => (await trialJobs()).every(job => trialBefore.has(job.job_id));
window.trialCancelOnThree = async () => {
  const job = await trialGet('/api/jobs/' + trialJob);
  if (job.step > 3 || ['error', 'done', 'cancelled'].includes(job.status))
    throw Error('Missed step 3: ' + JSON.stringify(job));
  if (job.status !== 'running' || job.step !== 3) return false;
  if (currentJob !== trialJob || !trialHit('#btn-cancel') || els.cancel.disabled)
    throw Error('Cancel button unavailable at step 3');
  window.trialCancelAt = job.step;
  els.cancel.click();
  return true;
};
window.trialFinished = async wanted => {
  const job = await trialGet('/api/jobs/' + trialJob);
  if (['error', 'cancelled', 'done'].includes(job.status) && job.status !== wanted)
    throw Error(JSON.stringify(job));
  return job.status === wanted && !busy;
};
