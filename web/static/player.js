/* player.js — one <audio> for the whole session, the mini dock, and the
   full-screen now-playing view.

   Spec §12: the queue IS the current filtered library list in its displayed
   order plus a cursor (the selected id). Shuffle applies to forward moves
   only. iOS: never recreate the element and never call play() outside a
   gesture chain — the first tap unlocks it and auto-advance rides on that. */

import { $, setIcon, fmtTime, toast, pushLayer, popLayer, dragToDismiss, paintIcons, el, icon } from './ui.js';
import * as store from './store.js';
import * as api from './api.js';

const audio = $('#audio');

let queue = [];          // entry objects, in displayed order
let cursor = -1;         // index into queue
let entry = null;        // the loaded entry
let opts = { shuffle: false, repeat: false, volume: 1 };
let markedPlayed = false;
let seekingNow = false;
let deadRun = 0;         // consecutive files that would not play
const listeners = { change: [], played: [] };

export function on(name, fn) { (listeners[name] ||= []).push(fn); }
const emit = (name, ...a) => (listeners[name] || []).forEach((f) => f(...a));

export const current = () => entry;
export const currentId = () => (entry ? entry.id : null);
export const isPlaying = () => !audio.paused && !audio.ended;

/* ── boot ──────────────────────────────────────────────────────────────── */
export function init(hooks = {}) {
  opts = store.load('player', { shuffle: false, repeat: false, volume: 1 });
  audio.volume = Number(opts.volume) || 1;

  // Any other media element that ever appears must yield: one song at a time.
  document.addEventListener('play', (e) => {
    if (e.target !== audio && e.target instanceof HTMLMediaElement) e.target.pause();
  }, true);

  $('#dock-play').addEventListener('click', toggle);
  $('#dock-next').addEventListener('click', () => next(true));
  $('#dock-prev').addEventListener('click', prev);
  $('#dock-shuffle').addEventListener('click', () => { opts.shuffle = !opts.shuffle; persist(); paintTransport(); });
  $('#dock-repeat').addEventListener('click', () => { opts.repeat = !opts.repeat; persist(); paintTransport(); });
  $('#dock-art').addEventListener('click', expand);
  $('#dock-meta').addEventListener('click', expand);
  const vol = $('#dock-volume');
  vol.value = String(audio.volume);
  vol.addEventListener('input', () => { audio.volume = Number(vol.value); persist(); });

  $('#now-close').addEventListener('click', collapse);
  $('#now-play').addEventListener('click', toggle);
  $('#now-next').addEventListener('click', () => next(true));
  $('#now-prev').addEventListener('click', prev);
  $('#now-shuffle').addEventListener('click', () => { opts.shuffle = !opts.shuffle; persist(); paintTransport(); });
  $('#now-repeat').addEventListener('click', () => { opts.repeat = !opts.repeat; persist(); paintTransport(); });
  $('#now-more').addEventListener('click', () => entry && hooks.onMore?.(entry));

  audio.addEventListener('timeupdate', onTime);
  audio.addEventListener('durationchange', onTime);
  audio.addEventListener('ended', onEnded);
  audio.addEventListener('play', paintPlay);
  audio.addEventListener('pause', paintPlay);
  // A file moved or trashed from another device must not end the session --
  // skip past it and keep the queue running. currentSrc is empty when stop()
  // clears src, which also fires error, and that is what this check covers.
  audio.addEventListener('error', () => {
    if (!audio.currentSrc) return;
    toast('That file would not play — it may sit outside the served folders.', 'err');
    deadRun++;
    if (deadRun < 3) next(false);
    else toast('Three files in a row would not play — the library may be out of date with the disk.', 'err');
  });
  // Reset on evidence of sound, NOT in load(): load() runs on the way into the
  // next file, so resetting there would zero the counter between every failure
  // and the breaker above could never trip -- an infinite skip loop over a
  // wiped folder with repeat on.
  audio.addEventListener('playing', () => { deadRun = 0; });

  // Registered once, not per load, so the lock screen's buttons can never
  // close over a stale queue. An action an engine does not implement throws.
  if ('mediaSession' in navigator) {
    for (const [name, fn] of [
      ['play', () => { if (audio.paused) audio.play().catch(() => {}); }],
      ['pause', () => audio.pause()],
      ['nexttrack', () => next(true)],
      ['previoustrack', prev],
      ['seekto', (d) => { if (d.seekTime != null) audio.currentTime = d.seekTime; }],
    ]) {
      try { navigator.mediaSession.setActionHandler(name, fn); } catch { /* not supported here */ }
    }
  }

  bindDockSeek();
  bindNowSeek();
  dragToDismiss($('#now-head'), $('#now'), collapse);
  dragToDismiss($('#now-art'), $('#now'), collapse);
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && document.body.classList.contains('now-open')) collapse();
  });

  buildNowActions(hooks);
  paintTransport();
  paintPlay();
}

function persist() { store.save('player', { shuffle: opts.shuffle, repeat: opts.repeat, volume: audio.volume }); }

/* ── queue ─────────────────────────────────────────────────────────────── */
/** The visible list became this. Keep the cursor pinned to the same id. */
export function setQueue(entries) {
  queue = entries || [];
  cursor = entry ? queue.findIndex((e) => e.id === entry.id) : -1;
}

export function play(target, list) {
  if (list) setQueue(list);
  const e = typeof target === 'string' ? queue.find((x) => x.id === target) : target;
  if (!e) return;
  const i = queue.findIndex((x) => x.id === e.id);
  cursor = i;
  // A deliberate pick clears the breaker. It is armed by the AUTO-ADVANCE
  // chain (error -> next(false) -> load), which never comes through here, so
  // this cannot reintroduce the oscillation that keeps the reset out of
  // load(). Without it the counter latches at 3 and the next dead file you
  // tap reports "three in a row" on its FIRST failure.
  deadRun = 0;
  load(e, true);
}

/** What the lock screen and the AirPods show. Screen off, phone in a pocket is
    the realistic session here, and without this iOS shows generic controls with
    no title and a next button that does nothing.

    Every call site sits AFTER audio.play(): metadata set before playback has
    started is discarded on iOS, because until the element is producing sound
    there is no session to attach it to. */
/** The artist name credited on the lock screen, from /api/profile.

    Cached rather than fetched per track so setMediaSession() can stay
    synchronous -- it runs immediately after audio.play(), and an await there
    would let the metadata land after iOS has already drawn the controls.

    Empty is a real value and stays empty. This line used to be a hard-coded
    name, which credited every track to a label nobody had typed; unset now
    means the field is omitted and iOS shows the title alone. */
let artistName = '';

export function setArtist(name) {
  artistName = String(name || '');
  setMediaSession();          // the current track keeps the old credit otherwise
}

export const artist = () => artistName;

function setMediaSession() {
  if (!('mediaSession' in navigator) || !entry) return;
  navigator.mediaSession.metadata = new MediaMetadata({
    title: entry.title || 'Untitled',
    artist: artistName,
    album: entry.workspace || 'AudioDev Studio',
    // Artwork must be a real fetchable URL -- a data: URI is ignored. cover_url
    // is already a /api/media path, so it goes across as-is.
    artwork: entry.cover_url ? [{ src: entry.cover_url, sizes: '512x512', type: 'image/png' }] : [],
  });
}

/** Load an ad-hoc result (a generation, an upscale) that is not in the list.

    An id makes it a library member anyway: Create passes the new track's
    library_id, and without it like, dislike, cover, playlist and more are
    silent no-ops on the one track you most want to rate. Restore's callers pass
    no id and stay ad-hoc, which is correct — a degraded copy is not a track.

    The upgrade path is the point: app.js schedules
    library.refresh({thenAgainIn: 2000}) after a generation, refresh() calls
    refreshEntry() for currentId(), and refreshEntry matches on id — so within
    about two seconds this stand-in is replaced by the real library entry and
    its cover art, rating state and meta appear with no user action. */
export function loadAdHoc({ url, title, sub, cover, id }) {
  entry = { id: id || null, adhoc: true, title, audio_url: url, cover_url: cover || null, _sub: sub || '' };
  cursor = -1;
  deadRun = 0;
  markedPlayed = !id;                        // no id, no library member to mark
  paintDock(); paintNow();
  audio.src = url;
  audio.play().catch(() => {});
  setMediaSession();
  document.body.classList.add('has-track');
  $('#dock').hidden = false;
  emit('change');
}

function load(e, autoplay) {
  entry = e;
  markedPlayed = false;
  document.body.classList.add('has-track');
  $('#dock').hidden = false;
  paintDock(); paintNow();
  if (audio.src !== e.audio_url) audio.src = e.audio_url;
  if (autoplay) audio.play().catch(() => { /* blocked outside a gesture */ });
  setMediaSession();
  emit('change');
}

export function toggle() {
  if (!entry) return;
  if (audio.paused) audio.play().catch(() => {}); else audio.pause();
}

export function next(userGesture) {
  if (!queue.length) return;
  // A deliberate skip clears the breaker; the auto-advance chain passes
  // userGesture false and must not, or the counter could never reach three.
  // This lives here rather than at the call sites because the dock and the
  // now-playing view both call next(true) directly.
  if (userGesture) deadRun = 0;
  let i;
  if (opts.shuffle && queue.length > 1) {
    do { i = Math.floor(Math.random() * queue.length); } while (i === cursor);
  } else {
    i = cursor + 1;
    if (i >= queue.length) {
      if (!opts.repeat) { toast('End of playlist.'); audio.pause(); return; }
      i = 0;
    }
  }
  cursor = i;
  load(queue[i], true);
}

export function prev() {
  if (!queue.length) return;
  deadRun = 0;                       // nothing auto-calls prev: always deliberate
  if (audio.currentTime > 3) { audio.currentTime = 0; return; }
  let i = cursor - 1;
  if (i < 0) i = opts.repeat ? queue.length - 1 : 0;
  cursor = i;
  load(queue[i], true);
}

function onEnded() {
  markPlayed(true);
  next(false);
}

/* ── played / unplayed (REQUIREMENTS §2) ───────────────────────────────── */
function markPlayed(force) {
  if (markedPlayed || !entry || !entry.id) return;
  const d = audio.duration;
  if (!force && !(isFinite(d) && d > 0 && audio.currentTime / d >= 0.5)) return;
  markedPlayed = true;
  entry.played = true;
  entry.plays = (entry.plays || 0) + 1;
  entry.last_played = new Date().toISOString();
  emit('played', entry);
  // Server-side so the phone and the desktop agree. PATCH ignores these keys --
  // /played is the endpoint that writes them to the sidecar. Still a soft
  // failure: if the write loses, the dot has already cleared locally.
  api.markPlayed(entry.id, entry.plays)
    .then((r) => { if (r && typeof r.plays === 'number') entry.plays = r.plays; })
    .catch(() => {});
}

/* ── painting ──────────────────────────────────────────────────────────── */
function subLine(e) {
  if (!e) return '';
  if (e.adhoc) return e._sub || '';
  const bits = [];
  if (e.takes > 1) bits.push(`take ${e.take}/${e.takes}`);
  if (e.model) bits.push(e.model);
  if (cursor >= 0 && queue.length) bits.push(`${cursor + 1}/${queue.length}`);
  return bits.join(' · ');
}

function paintDock() {
  const img = $('#dock-img');
  if (entry && entry.cover_url) { img.src = entry.cover_url; img.hidden = false; }
  else { img.removeAttribute('src'); img.hidden = true; }
  $('#dock-title').textContent = entry ? entry.title : 'Nothing loaded';
  $('#dock-sub').textContent = subLine(entry);
}

function paintNow() {
  const img = $('#now-img');
  if (entry && entry.cover_url) { img.src = entry.cover_url; img.hidden = false; }
  else { img.removeAttribute('src'); img.hidden = true; }
  $('#now-title').textContent = entry ? entry.title : '';
  $('#now-style').textContent = !entry ? '' : (entry.adhoc ? (entry._sub || '') : (entry.__style || ''));
  const meta = [];
  if (entry && !entry.adhoc) {
    if (entry.model) meta.push(entry.model);
    if (entry.takes > 1) meta.push(`take ${entry.take}/${entry.takes}`);
    if (entry.seconds) meta.push(`${Math.round(entry.seconds)}s`);
    if (entry.created) meta.push(entry.created.slice(5, 16).replace('T', ' '));
  }
  $('#now-meta').textContent = meta.join(' · ');
  paintActions();
}

function paintPlay() {
  // Cross-task contract with app.css: it keys the now-playing disc's rotation
  // on body.is-playing .now-art via animation-play-state, so a paused disc
  // stops where it was. Inert if that CSS is absent.
  document.body.classList.toggle('is-playing', !audio.paused);
  if ('mediaSession' in navigator) {
    try { navigator.mediaSession.playbackState = audio.paused ? 'paused' : 'playing'; }
    catch { /* not supported here */ }
  }
  const name = audio.paused ? 'play' : 'pause';
  setIcon($('#dock-play'), name);
  setIcon($('#now-play'), name);
  $('#dock-play').setAttribute('aria-label', audio.paused ? 'Play' : 'Pause');
  $('#now-play').setAttribute('aria-label', audio.paused ? 'Play' : 'Pause');
  emit('change');
}

function paintTransport() {
  for (const id of ['#now-shuffle', '#dock-shuffle']) {
    $(id).classList.toggle('iconbtn--on', !!opts.shuffle);
    $(id).classList.toggle('dock-btn--on', !!opts.shuffle);
    $(id).setAttribute('aria-pressed', String(!!opts.shuffle));
  }
  for (const id of ['#now-repeat', '#dock-repeat']) {
    $(id).classList.toggle('iconbtn--on', !!opts.repeat);
    $(id).classList.toggle('dock-btn--on', !!opts.repeat);
    setIcon($(id), opts.repeat ? 'repeat1' : 'repeat');
    $(id).setAttribute('aria-pressed', String(!!opts.repeat));
  }
}

function onTime() {
  const d = audio.duration, t = audio.currentTime;
  const frac = (isFinite(d) && d > 0) ? t / d : 0;
  $('#dock-fill').style.width = (frac * 100).toFixed(2) + '%';
  if (!seekingNow) $('#now-range').value = String(Math.round(frac * 1000));
  $('#now-elapsed').textContent = fmtTime(t);
  $('#now-total').textContent = isFinite(d) ? fmtTime(d) : '0:00';
  const dt = $('#dock-time'); if (dt) dt.textContent = `${fmtTime(t)} / ${isFinite(d) ? fmtTime(d) : '0:00'}`;
  // The lock-screen scrubber. Throws on a NaN duration -- every track has one
  // until metadata lands -- and on a position past the end, which currentTime
  // reaches by a frame at the very end.
  if ('mediaSession' in navigator && isFinite(d) && d > 0) {
    navigator.mediaSession.setPositionState({ duration: d, position: Math.min(t, d), playbackRate: audio.playbackRate });
  }
  markPlayed(false);
}

/* ── seeking ───────────────────────────────────────────────────────────── */
function bindDockSeek() {
  // Drag-only: the band is 12px, and a stray tap must never jump the track.
  const band = $('#dock-seek');
  let dragging = false;
  const apply = (clientX) => {
    const r = band.getBoundingClientRect();
    const f = Math.min(1, Math.max(0, (clientX - r.left) / r.width));
    if (isFinite(audio.duration)) audio.currentTime = f * audio.duration;
    $('#dock-fill').style.width = (f * 100).toFixed(2) + '%';
  };
  band.addEventListener('pointerdown', (e) => {
    if (!entry) return;
    dragging = false;
    band.setPointerCapture?.(e.pointerId);
    band._x0 = e.clientX;
  });
  band.addEventListener('pointermove', (e) => {
    if (band._x0 === undefined) return;
    if (!dragging && Math.abs(e.clientX - band._x0) < 6) return;
    dragging = true;
    apply(e.clientX);
  });
  const end = () => { band._x0 = undefined; dragging = false; };
  band.addEventListener('pointerup', end);
  band.addEventListener('pointercancel', end);
}

function bindNowSeek() {
  const r = $('#now-range');
  const start = () => { seekingNow = true; };
  const finish = () => {
    seekingNow = false;
    if (isFinite(audio.duration)) audio.currentTime = (Number(r.value) / 1000) * audio.duration;
  };
  r.addEventListener('pointerdown', start);
  r.addEventListener('input', () => {
    seekingNow = true;
    if (isFinite(audio.duration)) $('#now-elapsed').textContent = fmtTime((Number(r.value) / 1000) * audio.duration);
  });
  r.addEventListener('change', finish);
  r.addEventListener('pointerup', finish);
}

/* ── expand / collapse ─────────────────────────────────────────────────── */
const nowLayer = { close: () => teardownNow() };
let nowOpen = false;

export function expand() {
  if (!entry || nowOpen) return;
  nowOpen = true;
  const now = $('#now');
  now.classList.add('is-open');
  now.setAttribute('aria-hidden', 'false');
  document.body.classList.add('now-open');
  $('#app').setAttribute('aria-hidden', 'true');
  $('#app').setAttribute('inert', '');
  pushLayer(nowLayer);
  paintNow();
}

export function collapse() {
  if (!nowOpen) return;
  popLayer(nowLayer);
  teardownNow();
}

function teardownNow() {
  if (!nowOpen) return;
  nowOpen = false;
  const now = $('#now');
  now.classList.remove('is-open');
  now.setAttribute('aria-hidden', 'true');
  now.style.transform = '';
  document.body.classList.remove('now-open');
  $('#app').removeAttribute('aria-hidden');
  $('#app').removeAttribute('inert');
}

/* ── the five actions under the transport ──────────────────────────────── */
let actionHooks = {};
function buildNowActions(hooks) {
  actionHooks = hooks;
  const host = $('#now-actions');
  host.replaceChildren(
    btn('like', 'heart', 'Like'),
    btn('dislike', 'down', 'Dislike'),
    btn('cover', 'cover', 'Cover this'),
    btn('playlist', 'plus', 'Add to playlist'),
    btn('more', 'more', 'More'),
  );
  function btn(action, ic, label) {
    return el('button', {
      type: 'button', class: 'iconbtn', 'aria-label': label, dataset: { act: action },
      html: icon(ic),
      // An id, not the absence of `adhoc`, is what makes an action possible:
      // a just-generated track carries its library_id and is rateable.
      onclick: () => entry?.id && hooks.onAction?.(entry, action),
    });
  }
  paintIcons(host);
}

function paintActions() {
  const host = $('#now-actions');
  const like = host.querySelector('[data-act=like]');
  const dis = host.querySelector('[data-act=dislike]');
  if (!like) return;
  const r = entry?.id ? (entry.rating || 0) : 0;
  like.classList.toggle('iconbtn--on', r === 1);
  setIcon(like, r === 1 ? 'heartOn' : 'heart');
  dis.classList.add('iconbtn--down');
  dis.classList.toggle('iconbtn--on', r === -1);
}

/** The library refreshed; re-point at the same id so ratings stay in sync.
    This is also where an ad-hoc generation becomes its real entry, so the lock
    screen swaps the filename for the title and picks up the cover art. */
export function refreshEntry(e) {
  if (entry && e && entry.id === e.id) { entry = e; paintDock(); paintNow(); setMediaSession(); }
}

/** Stop and clear (the loaded track was trashed). */
export function stop() {
  audio.pause();
  audio.removeAttribute('src');
  audio.load();
  entry = null; cursor = -1;
  if ('mediaSession' in navigator) {
    try { navigator.mediaSession.playbackState = 'none'; } catch { /* ignore */ }
  }
  document.body.classList.remove('has-track');
  $('#dock').hidden = true;
  collapse();
  emit('change');
}
