// AuraDream's service worker: the page's own answer when the server does not
// answer. Served at /sw.js (main.py), not /static/, so its scope is the whole
// site; registered by app.js where the page is a secure context (127.0.0.1,
// localhost or HTTPS - never a phone on http://<LAN address>, where Safari
// refuses service workers).
//
// It handles one request: the navigation to "/". The network goes first,
// always - a cached page could outlive a `git pull` and talk to a newer
// server in an older page's words - and only when no answer comes (a
// network error, or nothing within NAVIGATION_TIMEOUT_MS, which is a
// sleeping PC on Wi-Fi) does it answer with offline.html, which says so and
// opens AuraDream again once the server answers. Every other request,
// /api/, /static/, /outputs/ and the job streams, passes it untouched.
// Not a classic page script: index.html never loads it.
'use strict';

// offline.html is cached at install only: a change to it needs a new name
// here, or an installed worker keeps the old copy (the hash beside it in
// tests/test_service_worker.py holds the two together).
const CACHE = 'auradream-offline-v2';
const OFFLINE_URL = '/static/offline.html';
// A LAN server sends "/" in milliseconds, busy or not (a load runs off the
// event loop); a PC asleep or gone from the network leaves a connection
// hanging far longer than this before the browser gives up.
const NAVIGATION_TIMEOUT_MS = 6000;

self.addEventListener('install', (event) => {
  event.waitUntil((async () => {
    const cache = await caches.open(CACHE);
    await cache.add(new Request(OFFLINE_URL, { cache: 'reload' }));
    await self.skipWaiting();
  })());
});

self.addEventListener('activate', (event) => {
  event.waitUntil((async () => {
    const names = await caches.keys();
    await Promise.all(names
      .filter((name) => name.startsWith('auradream-') && name !== CACHE)
      .map((name) => caches.delete(name)));
    await self.clients.claim();
  })());
});

self.addEventListener('fetch', (event) => {
  const { request } = event;
  if (request.mode !== 'navigate' || request.method !== 'GET') return;
  const url = new URL(request.url);
  if (url.origin !== self.location.origin || url.pathname !== '/') return;
  event.respondWith(pageOrOffline(url.href));
});

async function pageOrOffline(href) {
  let timer;
  const timeout = new Promise((resolve) => {
    timer = setTimeout(() => resolve(null), NAVIGATION_TIMEOUT_MS);
  });
  // Any answer is the server's to give, a refusal (a host it does not
  // allow, a 500) included: only silence is "not reachable". `no-cache`
  // asks the server every time (a 304 when nothing changed): "/" carries no
  // Cache-Control, and the HTTP cache's heuristic freshness otherwise hands
  // back a page the server never saw asked for. No navigation preload, which
  // would take the navigation's own cache mode.
  const answer = fetch(href, { cache: 'no-cache', credentials: 'same-origin' });
  try {
    const response = await Promise.race([answer, timeout]);
    if (response) return response;
  } catch { /* no answer: the network refused */ } finally {
    clearTimeout(timer);
  }
  answer.catch(() => {});  // a late failure has nobody left to tell
  return offlinePage();
}

async function offlinePage() {
  const cached = await caches.match(OFFLINE_URL);
  if (cached) return cached;
  // The cache was cleared under the worker: the same words, plainly.
  return new Response(
    '<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
    + '<title>AuraDream is not answering</title>'
    + '<p>AuraDream is not answering. Check that the PC is on and AuraDream is running, then reload.</p>',
    { status: 503, headers: { 'Content-Type': 'text/html; charset=utf-8' } },
  );
}
