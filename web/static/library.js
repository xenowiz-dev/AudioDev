/* library.js — the row list, its filters, and everything behind the ⋯.

   Rows are rendered client-side and windowed (60, then 40 per sentinel hit):
   400 tracks of server-rendered HTML is a megabyte down a phone radio.

   REQUIREMENTS §1: every row action stopPropagation()s. The old gradio app
   fell through to the row's play handler, so `edit` and `more` just played
   the song. Nothing here may reintroduce that. */

import { $, $$, el, icon, setIcon, toast, openSheet, closeSheet, menu, paintIcons,
         fmtDur, stripMd, setHint, debounce } from './ui.js';
import * as api from './api.js';
import * as store from './store.js';
import * as player from './player.js';

const PAGE_1 = 60, PAGE_N = 40;

const state = {
  q: '', fav: false, unplayed: false, playlist: null, view: 'list',
  entries: [], shown: 0, playlists: [], status: '', selected: null,
  // A workspace is the project you are working IN; selecting one both filters
  // the library and becomes the context new songs are tagged with. '' is
  // Everything (no filter, nothing tagged), '(none)' is the untagged ones.
  workspace: '', workspaces: { active: null, names: [] },
  // Trash is not a filter over library entries -- trashed items are not
  // library members, have no id, and cannot be queued -- so it is a MODE.
  trash: false, trashItems: null,
};

let hooks = {};        // { onReuse, onCover, onRestore, goTab }
let listEl, sentinelEl, statusEl, io = null;

/* Everything that decides WHICH tracks are on screen. `shown` may be carried
   across a refresh only while this string is unchanged: holding a high count
   through a FILTER change would render hundreds of rows for a result that
   narrowed to twelve. Data changing under the same filter is the opposite
   case — a render finishing, a track binned — and there the count must hold. */
const filterKey = () => [state.q, state.fav, state.playlist, state.workspace,
                         state.unplayed, state.trash].join(' ');
let lastKey = null;

/* ── the style line — ported from rowview._style_line, semantics unchanged ─ */
const RE_TECHNICAL = /\b(bpm|khz|hz|kbps|lufs|db|key|scale|tempo|time signature|meter|duration)\b/i;
const RE_NUMERIC = /^[\W\d]*$/;
const RE_LABEL = /^[A-Za-z][A-Za-z &/'-]{2,40}:\s*/;
const STYLE_SEGMENTS = 3, STYLE_MAX = 96;

function* promptSegments(prompt) {
  const lines = String(prompt || '').split('\n').map((s) => s.trim()).filter(Boolean);
  const labelled = [];
  for (const ln of lines) { const m = RE_LABEL.exec(ln); if (m) labelled.push(ln.slice(m[0].length)); }
  for (const chunk of (labelled.length ? labelled : lines)) {
    for (const sentence of chunk.split(/(?<=[.!?])\s+/)) {
      for (let piece of sentence.split(/[,;]/)) {
        piece = piece.split(/\s+/).join(' ').replace(/^[\s.·-]+|[\s.·-]+$/g, '');
        if (!piece || RE_NUMERIC.test(piece) || RE_TECHNICAL.test(piece)) continue;
        yield piece;
      }
    }
  }
}

function clip(s, n) {
  s = String(s || '').split(/\s+/).filter(Boolean).join(' ');
  if (s.length <= n) return s;
  let cut = s.slice(0, n).trimEnd();
  const sp = cut.lastIndexOf(' ');
  if (sp >= n / 2) cut = cut.slice(0, sp);
  return cut.replace(/[\s,;:·-]+$/, '') + '…';
}

export function styleLine(e) {
  const style = String(e.style || '').split(/\s+/).filter(Boolean).join(' ');
  if (style) return [clip(style, STYLE_MAX), false];
  const segs = [];
  for (const s of promptSegments(e.prompt)) { segs.push(s); if (segs.length >= STYLE_SEGMENTS) break; }
  if (segs.length) return [clip(segs.join(' · '), STYLE_MAX), false];
  if (!(e.recorded || e.shared_rec)) return ['no record — found on disk', true];
  return ['no prompt recorded', true];
}

/* ── boot ──────────────────────────────────────────────────────────────── */
export function init(h) {
  hooks = h || {};
  listEl = $('#rowlist'); sentinelEl = $('#rowlist-sentinel'); statusEl = $('#liststatus');

  const saved = store.load('library', { q: '', fav: false, playlist: null, view: 'list', workspace: '' });
  Object.assign(state, { q: saved.q || '', fav: !!saved.fav, playlist: saved.playlist || null,
                         view: saved.view === 'grid' ? 'grid' : 'list',
                         workspace: saved.workspace || '' });

  // search: hidden behind an icon on mobile, permanent chrome on desktop
  const q = $('#q');
  q.value = state.q;
  $('#search-open').addEventListener('click', () => openSearch(true));
  $('#search-close').addEventListener('click', () => openSearch(false));
  $('#searchfield').addEventListener('submit', (e) => { e.preventDefault(); q.blur(); });
  q.addEventListener('input', debounce(() => { state.q = q.value; persist(); refresh(); }, 300));
  if (state.q) openSearch(true, false);

  $('#ws-select')?.addEventListener('change', (e) => chooseWorkspace(e.target.value));

  io = new IntersectionObserver((es) => {
    if (es.some((x) => x.isIntersecting) && state.shown < state.entries.length) {
      state.shown = Math.min(state.entries.length, state.shown + PAGE_N);
      paintRows();
    }
  }, { rootMargin: '400px' });
  io.observe(sentinelEl);

  player.on('change', () => {
    markPlaying();
    // The desktop detail pane follows the player, so skipping to the next
    // track updates it too — not only a row tap.
    const id = player.currentId();
    if (!id) return;
    const e = state.entries.find((x) => x.id === id);
    if (e && state.selected !== e.id) select(e);
  });
  player.on('played', (e) => { const r = rowFor(e.id); if (r) r.querySelector('.row-new')?.remove(); });
  clearDetail();
}

function persist() {
  store.save('library', { q: state.q, fav: state.fav, playlist: state.playlist,
                          view: state.view, workspace: state.workspace });
}

function openSearch(on, focus = true) {
  // At >=900px the field is permanent chrome: "close" clears the query but
  // must not remove the input, or the header loses it until the next reflow.
  if (!on && matchMedia('(min-width: 900px)').matches) {
    if (state.q) { state.q = ''; $('#q').value = ''; persist(); refresh(); }
    $('#q').blur();
    return;
  }
  $('#searchfield').hidden = !on;
  $('#appbar-title').hidden = on && !matchMedia('(min-width: 900px)').matches;
  $('#search-open').hidden = on;
  // Restoring a saved query must NOT focus: on a phone that pops the keyboard
  // over the list on every cold start.
  if (on && focus) $('#q').focus();
  else if (!on && state.q) { state.q = ''; $('#q').value = ''; persist(); refresh(); }
}

/* ── data ──────────────────────────────────────────────────────────────── */
let inflight = null;

export async function refresh(opts = {}) {
  inflight?.abort();
  const ac = new AbortController();
  inflight = ac;
  try {
    if (state.trash) { await refreshTrash(); return; }
    const params = { limit: 500 };
    if (state.q) params.q = state.q;
    if (state.fav) params.fav = 1;
    if (state.playlist) params.playlist = state.playlist;
    if (state.workspace) params.workspace = state.workspace;
    const data = await api.getLibrary(params, { signal: ac.signal });
    let entries = data.entries || [];
    // "Unplayed only" is a client-side filter: the list endpoint has no such
    // parameter, and REQUIREMENTS §2 wants the toggle regardless.
    if (state.unplayed) entries = entries.filter((e) => !e.played);
    // The player renders the same style line as the row, and cannot import
    // this module (it would be a cycle), so cache it on the entry.
    for (const e of entries) e.__style = styleLine(e)[0];
    state.entries = entries;
    state.playlists = data.playlists || [];
    state.workspaces = data.workspaces || { active: null, names: [] };
    // A remembered workspace can name one that has since been deleted; the
    // same stale-id rule the LoRA dropdown follows.
    if (state.workspace && state.workspace !== '(none)'
        && !state.workspaces.names.some((n) => n === state.workspace)) {
      state.workspace = '';
    }
    state.status = data.status || `${entries.length} tracks`;
    // Keep your place. Rebuilding the window on every refresh is what threw
    // you back to the top after adding track #90 to a playlist or finishing a
    // render — replaceChildren() is atomic, so once the row count survives,
    // so does the scroll position. Reset only when the FILTER moved.
    const filterMoved = filterKey() !== lastKey;
    state.shown = filterMoved
      ? Math.min(entries.length, PAGE_1)
      : Math.min(entries.length, Math.max(state.shown, PAGE_1));
    lastKey = filterKey();
    paintChips();
    paintRows();
    // A new filter is a new list, so it starts at the top. Without this the
    // old scrollTop is merely CLAMPED to the shorter list -- type a query while
    // parked at the bottom and you land on match 65 of 73. Worse, the sentinel
    // is then still inside its own rootMargin, and IntersectionObserver only
    // fires on a TRANSITION: no boundary is crossed, so the window sticks at
    // 60 and no amount of scrolling grows it. #main is the scroller, not
    // #rowlist and not document.scrollingElement -- both read 0 forever.
    if (filterMoved) {
      $('#main')?.scrollTo({ top: 0, behavior: 'instant' });
      // Belt and braces for a list shorter than the viewport, where scrolling
      // to 0 crosses no boundary either: re-observing delivers a fresh initial
      // callback with the current intersection state.
      if (io && sentinelEl) { io.unobserve(sentinelEl); io.observe(sentinelEl); }
    }
    player.setQueue(state.entries);
    const cur = player.currentId();
    if (cur) { const e = state.entries.find((x) => x.id === cur); if (e) player.refreshEntry(e); }
    if (state.selected) {
      const still = state.entries.find((e) => e.id === state.selected);
      if (still) renderDetail(still);
      else { state.selected = null; clearDetail('Track left the current view.'); }
    }
  } catch (e) {
    if (e.name === 'AbortError') return;
    toast(e.message || 'Could not load the library.', 'err');
  } finally { if (inflight === ac) inflight = null; }
  if (opts.thenAgainIn) setTimeout(() => refresh(), opts.thenAgainIn);
}

/* ── trash (§16) ───────────────────────────────────────────────────────────
   A MODE, not a filter. Trashed items are not library entries: no id, no
   rating, nothing to queue, and 19 of them have no sidecar at all because
   they predate per-take records. Feeding them through paintRows/setQueue
   would break the player, so they get their own render with one action.

   Restore only. Emptying the trash permanently is deliberately not offered
   here -- the ask was to SEE what was binned, and restore is the reversible
   half of the move that put it there. */
async function refreshTrash() {
  try {
    const d = await api.getTrash();
    state.trashItems = d.items || [];
    state.status = `${d.count ?? state.trashItems.length} in the trash`;
    paintChips();
    paintTrashRows(d.note);
  } catch (e) {
    toast(e.message || 'Could not read the trash.', 'err');
  }
}

function paintTrashRows(note) {
  const items = state.trashItems || [];
  listEl.className = 'rowlist';
  if (!items.length) {
    listEl.replaceChildren(el('p', { class: 'empty', text: 'The trash is empty.' }));
    statusEl.textContent = '';
    return;
  }
  const frag = document.createDocumentFragment();
  for (const it of items) {
    const when = it.mtime ? new Date(it.mtime * 1000).toLocaleString() : '';
    const btn = el('button', {
      type: 'button', class: 'btn btn--ghost', text: 'Restore',
      onclick: async (ev) => {
        ev.stopPropagation();
        ev.currentTarget.disabled = true;
        try {
          await api.restoreTrashed(it.name);
          toast(`Restored ${it.title || it.name}.`);
          await refreshTrash();
        } catch (err) {
          ev.currentTarget.disabled = false;
          toast(err.message || 'Could not restore that.', 'err');
        }
      },
    });
    frag.append(el('div', { class: 'row row--trash' },
      el('div', { class: 'meta' },
        el('div', { class: 'meta-l1' },
          el('span', { class: 'title', text: it.title || it.name })),
        el('span', {
          class: 'style' + (it.has_record ? '' : ' style--dim'),
          text: it.has_record
            ? [it.model, it.prompt].filter(Boolean).join(' · ').slice(0, 90) || when
            : `no record — ${when}`,
        })),
      btn));
  }
  listEl.replaceChildren(frag);
  statusEl.textContent = note || '';
}

/* ── chips ─────────────────────────────────────────────────────────────── */
function paintChips() {
  paintWorkspaceBar();
  const host = $('#chipslot') || $('#chiprow');
  const chips = [];

  if (state.trash) {
    chips.push(chip('← Back to library', false, null, () => {
      state.trash = false; state.trashItems = null; refresh();
    }));
    chips.push(chip(`Trash (${(state.trashItems || []).length})`, true, 'trash', null));
    host.replaceChildren(...chips);
    return;
  }

  if (state.q) chips.push(chip(`✕ “${clip(state.q, 18)}”`, true, null, () => { state.q = ''; $('#q').value = ''; openSearch(false); persist(); refresh(); }));
  chips.push(chip('All', !state.playlist && !state.fav && !state.unplayed, null,
    () => { state.playlist = null; state.fav = false; state.unplayed = false; persist(); refresh(); }));
  chips.push(chip('Favourites', state.fav, 'star', () => { state.fav = !state.fav; persist(); refresh(); }));
  chips.push(chip('Unplayed', state.unplayed, null, () => { state.unplayed = !state.unplayed; persist(); refresh(); }));
  for (const name of state.playlists) {
    chips.push(chip(name, state.playlist === name, null,
      () => { state.playlist = state.playlist === name ? null : name; persist(); refresh(); }));
  }
  // Playlists are curation with a goal, so creating one belongs beside them.
  chips.push(chip('+ Playlist', false, null, newPlaylist));
  chips.push(chip('Trash', false, 'trash', () => {
    state.trash = true; state.selected = null; clearDetail(); refresh();
  }));
  chips.push(chip(state.view === 'grid' ? 'Grid' : 'List', false, null,
    () => { state.view = state.view === 'grid' ? 'list' : 'grid'; persist(); paintChips(); paintRows(); }));
  host.replaceChildren(...chips);
}

/* The workspace picker sets CONTEXT, not a filter: choosing one both narrows
   the library and makes it where new songs land. Two different jobs from the
   chips beside it, hence a different control. */
function paintWorkspaceBar() {
  const bar = $('#wsbar'), sel = $('#ws-select');
  if (!bar || !sel) return;
  bar.hidden = state.trash;
  if (state.trash) return;
  const names = state.workspaces?.names || [];
  sel.replaceChildren(
    el('option', { value: '', text: 'Everything' }),
    el('option', { value: '(none)', text: 'Untagged' }),
    ...names.map((n) => el('option', { value: n, text: n })),
    el('option', { value: '__new', text: '+ New workspace…' }),
    ...(state.workspace && state.workspace !== '(none)'
        ? [el('option', { value: '__del', text: `✕ Delete “${state.workspace}”` })] : []));
  sel.value = state.workspace || '';
  sel.classList.toggle('wsbar--on', !!state.workspace && state.workspace !== '(none)');
}

async function chooseWorkspace(v) {
  if (v === '__new') {
    const name = await promptSheet('New workspace',
      'A project to keep generations together — “Cinematic”, “Album 2”. '
      + 'New songs are tagged with whichever workspace is selected.');
    if (!name) { paintWorkspaceBar(); return; }
    try {
      await api.createWorkspace(name);
      await api.setActiveWorkspace(name);
      state.workspace = name;
    } catch (e) { toast(e.message, 'err'); }
  } else if (v === '__del') {
    const name = state.workspace;
    const ok = await confirmSheet(`Delete workspace “${name}”?`,
      'The songs keep their tag and their audio — only the name goes. '
      + 'Recreating it brings them back into view.');
    if (!ok) { paintWorkspaceBar(); return; }
    try { await api.deleteWorkspace(name); state.workspace = ''; }
    catch (e) { toast(e.message, 'err'); }
  } else {
    state.workspace = v;
    // Selecting a workspace is also "I am working here now", so new songs
    // land in it. Server-side, because the phone and the desktop must agree.
    try { await api.setActiveWorkspace(v && v !== '(none)' ? v : null); }
    catch { /* filtering still works even if the context write fails */ }
  }
  persist();
  refresh();
}

/* Two tiny sheets. `prompt()` and `confirm()` are browser modals — they block
   the page and, per this project's standing rule, must never be used: a modal
   that hangs the tab is a worse failure than the thing it was guarding. */
function promptSheet(title, hint, value = '') {
  return new Promise((resolve) => {
    let done = false;
    const input = el('input', { type: 'text', value, spellcheck: 'false', maxlength: '48' });
    const finish = (v) => { if (done) return; done = true; closeSheet(); resolve(v); };
    input.addEventListener('keydown', (e) => { if (e.key === 'Enter') finish(input.value.trim() || null); });
    openSheet(title, el('div', { class: 'formcol', style: 'padding:0;gap:16px' },
      el('div', { class: 'field' }, input, el('p', { class: 'fhint', text: hint })),
      el('div', { class: 'btnrow' },
        el('button', { type: 'button', class: 'btn', text: 'Cancel', onclick: () => finish(null) }),
        el('button', { type: 'button', class: 'btn btn--primary', text: 'Create',
                       onclick: () => finish(input.value.trim() || null) }))),
      { onClose: () => finish(null) });
    setTimeout(() => input.focus(), 60);
  });
}

function confirmSheet(title, hint) {
  return new Promise((resolve) => {
    let done = false;
    const finish = (v) => { if (done) return; done = true; closeSheet(); resolve(v); };
    openSheet(title, el('div', {},
      el('p', { class: 'note', text: hint }),
      el('div', { class: 'btnrow', style: 'margin-top:16px' },
        el('button', { type: 'button', class: 'btn', text: 'Cancel', onclick: () => finish(false) }),
        el('button', { type: 'button', class: 'btn btn--danger', text: 'Delete',
                       onclick: () => finish(true) }))),
      { onClose: () => finish(false) });
  });
}

async function newPlaylist() {
  const name = await promptSheet('New playlist',
    'A set with a goal — an album, a mixtape. Add tracks from a row’s ⋮ menu.');
  if (!name) return;
  try {
    await api.createPlaylist(name);
    toast(`Created “${name}”. Add tracks from any row’s ⋮ menu.`);
    refresh();
  } catch (e) { toast(e.message, 'err'); }
}

function chip(label, on, ic, onClick) {
  return el('button', { type: 'button', class: 'chip' + (on ? ' chip--on' : ''), onclick: onClick,
                        'aria-pressed': String(!!on) },
    el('span', {}, ic ? el('span', { html: icon(ic), class: 'chip-i' }) : null, label));
}

/* ── rows ──────────────────────────────────────────────────────────────── */
function paintRows() {
  listEl.className = 'rowlist' + (state.view === 'grid' ? ' rowlist--grid' : '');
  if (!state.entries.length) {
    listEl.replaceChildren(el('p', { class: 'empty', text: state.q || state.fav || state.playlist || state.unplayed
      ? 'Nothing matches that filter.' : 'No tracks yet — make one on the Create tab.' }));
    statusEl.textContent = '';
    return;
  }
  const frag = document.createDocumentFragment();
  for (const e of state.entries.slice(0, state.shown)) frag.append(row(e));
  listEl.replaceChildren(frag);
  paintStatus();
  markPlaying();
}

function paintStatus() {
  statusEl.textContent = state.shown < state.entries.length
    ? `${state.shown} of ${state.entries.length} shown — scroll for more`
    : stripMd(state.status);
}

function row(e) {
  const [style, dim] = styleLine(e);
  const dur = fmtDur(e.seconds);
  const r = el('div', {
    class: 'row', dataset: { id: e.id }, role: 'button', tabindex: '0',
    onclick: () => player.play(e, state.entries),
    onkeydown: (ev) => { if (ev.key === 'Enter' || ev.key === ' ') { ev.preventDefault(); player.play(e, state.entries); } },
  });

  if (!e.played) r.append(el('span', { class: 'row-new', title: 'Not played yet' }));

  const thumb = el('button', {
    type: 'button', class: 'thumb', 'aria-label': `Play ${e.title}`,
    onclick: (ev) => { ev.stopPropagation(); player.play(e, state.entries); },
  });
  if (e.cover_url) thumb.append(el('img', { src: e.cover_url, alt: '', loading: 'lazy', decoding: 'async' }));
  else thumb.append(el('span', { html: icon('note') }));
  if (dur) thumb.append(el('span', { class: 'dur', text: dur }));
  thumb.append(el('span', { class: 'thumb-state', html: icon('pause') }));
  r.append(thumb);

  const l1 = el('div', { class: 'meta-l1' }, el('span', { class: 'title', text: e.title, title: e.title }));
  if (e.takes > 1) l1.append(el('span', { class: 'take', text: `${e.take}/${e.takes}`, title: `take ${e.take} of ${e.takes}` }));
  r.append(el('div', { class: 'meta' }, l1,
    el('span', { class: 'style' + (dim ? ' style--dim' : ''), text: style, title: style })));

  const like = el('button', {
    type: 'button', class: 'iconbtn' + (e.rating === 1 ? ' iconbtn--on' : ''),
    'aria-label': e.rating === 1 ? 'Remove like' : 'Like', 'aria-pressed': String(e.rating === 1),
    html: icon(e.rating === 1 ? 'heartOn' : 'heart'),
    onclick: (ev) => { ev.stopPropagation(); rate(e, e.rating === 1 ? 0 : 1); },
  });
  const more = el('button', {
    type: 'button', class: 'iconbtn', 'aria-label': `Actions for ${e.title}`, html: icon('more'),
    onclick: (ev) => { ev.stopPropagation(); openMore(e); },
  });
  r.append(el('div', { class: 'acts' }, like, more));
  return r;
}

const rowFor = (id) => listEl.querySelector(`.row[data-id="${CSS.escape(id)}"]`);

function markPlaying() {
  const id = player.currentId();
  for (const r of $$('.row', listEl)) r.classList.toggle('row--playing', r.dataset.id === id);
  const st = listEl.querySelector('.row--playing .thumb-state');
  if (st) setIcon(st, player.isPlaying() ? 'pause' : 'play');
}

/** Patch one row in place after a mutation — never re-index a stale list. */
function replaceEntry(updated) {
  const i = state.entries.findIndex((e) => e.id === updated.id);
  if (i < 0) return;
  updated.__style = styleLine(updated)[0];
  state.entries[i] = updated;
  const old = rowFor(updated.id);
  if (old) old.replaceWith(row(updated));
  markPlaying();
  player.refreshEntry(updated);
  if (state.selected === updated.id) renderDetail(updated);
}

/** Re-read one track and patch its row. /api/library/<id> answers with the
    entry wrapped alongside detail_md and shared_note, and replaceEntry()
    matches on `.id` — handing it the wrapper finds nothing and silently
    does nothing, which looks exactly like a working call. */
async function patchEntry(id) {
  const got = await api.getTrack(id).catch(() => null);
  if (got?.entry) replaceEntry(got.entry);
}

/** Take one row out of the view in place. Binning a dud and removing from the
    playlist you are looking at both mean "this is not here any more"; a
    refresh() to say so rebuilt the whole list, and both are actions you take
    deep in a night's output, which is precisely where losing your place hurts.
    Trash-mode items never reach this — they have no id (§16). */
function dropEntry(id) {
  const i = state.entries.findIndex((x) => x.id === id);
  if (i < 0) return;
  const gone = state.entries[i];
  rowFor(id)?.remove();
  state.entries.splice(i, 1);
  // The rendered window is one row shorter, so the count follows it down —
  // otherwise statusEl and the next slice() disagree with the DOM.
  if (i < state.shown) state.shown -= 1;
  state.shown = Math.min(state.shown, state.entries.length);
  player.setQueue(state.entries);   // the queue IS the displayed order
  // The desktop detail pane can be showing the row that just went; refresh()
  // says exactly this when a track falls out of the view, so say it here too
  // rather than leaving a pane describing something that is no longer listed.
  if (state.selected === id) { state.selected = null; clearDetail('Track left the current view.'); }
  // The server's status line counted the track that just left. Only that
  // number went stale, and refetching 500 entries to correct it is the round
  // trip this whole path exists to avoid.
  state.status = String(state.status || '')
    .replace(/^\d+(?= tracks\b)/, String(state.entries.length));
  // The second number counts prompts, not tracks, so it only moves when the
  // track that left HAD one -- api.py:1939 counts `recorded or shared_rec`.
  // Decrementing it unconditionally is how "129 tracks - 93 with a prompt"
  // ends up one ahead of the truth after binning an untagged render.
  if (gone && (gone.recorded || gone.shared_rec)) {
    state.status = state.status.replace(
      /(\d+)(?= with a prompt)/, (m) => String(Math.max(0, Number(m) - 1)));
  }
  if (state.entries.length) paintStatus();
  else paintRows();                 // the last row left: show the empty line
}

/* ── mutations ─────────────────────────────────────────────────────────── */
async function rate(e, rating) {
  try {
    const r = await api.setRating(e.id, rating);
    replaceEntry(r.entry || { ...e, rating: r.rating, favorite: r.favorite });
    if (r.message) toast(r.message);
  } catch (err) { toast(err.message, 'err'); }
}

async function favourite(e, on) {
  try {
    const r = await api.setFavorite(e.id, on);
    replaceEntry(r.entry || { ...e, favorite: r.favorite });
    toast(on ? '★ Favourited' : '☆ Favourite removed');
  } catch (err) { toast(err.message, 'err'); }
}

async function trash(e) {
  try {
    const r = await api.trashTrack(e.id);
    toast(r.message || 'Moved to trash.');
    if (player.currentId() === e.id) player.stop();
    if (state.selected === e.id) { state.selected = null; clearDetail(); }
    dropEntry(e.id);
  } catch (err) { toast(err.message, 'err'); }
}

async function reuse(e) {
  try {
    const r = await api.getReuse(e.id);
    hooks.onReuse?.(r);
    toast(r.message || 'Loaded into Create.');
  } catch (err) { toast(err.message, 'err'); }
}

async function coverThis(e) {
  try {
    const r = await api.getCoverSrc(e.id);
    // /cover-source answers with the PATH; the cover-mode banner needs the
    // track's title and art, and the row we just clicked is the only place
    // that has them.
    hooks.onCover?.({ ...r, title: r.title || e.title, cover_url: r.cover_url || e.cover_url });
    toast(r.message || 'Loaded as a cover source.');
  } catch (err) { toast(err.message, 'err'); }
}

/** Hand the track's file to the Restore tab. No round trip: the entry already
    carries `path`, which is exactly what Restore's source field wants. */
function sendToRestore(e) {
  if (!e.path) { toast('That track has no file path.', 'err'); return; }
  hooks.onRestore?.(e);
  toast(`Sent “${e.title}” to Restore.`);
}

async function regenArt(e) {
  try { await api.kickCovers(true, 1); toast('Re-rendering the cover — it appears on the next refresh.'); }
  catch (err) { toast(err.message, 'err'); }
}

/* ── the ⋯ sheet ───────────────────────────────────────────────────────── */
export function openMore(e) {
  // Read once, here: the item below is LABELLED with this playlist, so it must
  // act on that one even if the filter moved while the sheet was open.
  const from = state.playlist;
  // Sheet-to-sheet transitions swap in place (see ui.openSheet); only the
  // items that finish the interaction call closeSheet().
  openSheet(e.title, menu([
    { icon: 'edit', label: 'Edit title and style', onClick: () => openEdit(e) },
    { icon: 'down', label: e.rating === -1 ? 'Remove dislike' : 'Dislike', onClick: () => { closeSheet(); rate(e, e.rating === -1 ? 0 : -1); } },
    { icon: 'star', label: e.favorite ? '★ Favourited' : '☆ Favourite', onClick: () => { closeSheet(); favourite(e, !e.favorite); } },
    '-',
    { icon: 'reuse', label: 'Reuse prompt in Create', onClick: () => { closeSheet(); reuse(e); } },
    { icon: 'cover', label: 'Cover this track', onClick: () => { closeSheet(); coverThis(e); } },
    { icon: 'restore', label: 'Send to Restore', onClick: () => { closeSheet(); sendToRestore(e); } },
    { icon: 'plus', label: 'Add to playlist…', onClick: () => openPlaylistPicker(e) },
    // Only while viewing one: "remove from playlist" is meaningless without
    // knowing which, and offering it everywhere invites removing from the
    // wrong list.
    from ? { icon: 'x', label: `Remove from “${from}”`,
      onClick: async () => {
        closeSheet();
        try {
          await api.removeFromList(from, e.id);
          toast(`Removed from “${from}”.`);
          // Two different outcomes: while that playlist is the filter the
          // track has genuinely left the view, so the row goes. Otherwise only
          // its membership changed and the row is patched where it stands.
          if (state.playlist === from) dropEntry(e.id);
          else await patchEntry(e.id);
        } catch (err) { toast(err.message, 'err'); }
      } } : null,
    { icon: 'art', label: 'Regenerate cover art', onClick: () => { closeSheet(); regenArt(e); } },
    '-',
    { icon: 'text', label: 'Show prompt and lyrics', onClick: () => openRecord(e) },
    { icon: 'copy', label: 'Copy file path', onClick: () => { closeSheet(); copyPath(e); } },
    { icon: 'download', label: 'Download', onClick: () => { closeSheet(); download(e); } },
    '-',
    { icon: 'trash', label: 'Move to trash', danger: true, onClick: () => confirmTrash(e) },
  ]));
}

function confirmTrash(e) {
  openSheet('Move to trash?', el('div', {},
    el('p', { class: 'note', text: `${e.title} moves to trash\\ — a folder move, never a delete. It can be restored.` }),
    el('div', { class: 'btnrow', style: 'margin-top:16px' },
      el('button', { type: 'button', class: 'btn', text: 'Cancel', onclick: () => closeSheet() }),
      el('button', { type: 'button', class: 'btn btn--danger', text: 'Move to trash', onclick: () => { closeSheet(); trash(e); } }))));
}

function copyPath(e) {
  navigator.clipboard?.writeText(e.path)
    .then(() => toast('Path copied.'))
    .catch(() => toast(e.path));
}

function download(e) {
  const a = el('a', { href: e.audio_url, download: e.name });
  document.body.append(a); a.click(); a.remove();
}

/* ── edit (title + style) — must NOT start playback ────────────────────── */
export function openEdit(e) {
  const title = el('input', { type: 'text', value: e.title_override || '', placeholder: e.title, spellcheck: 'false' });
  const style = el('input', { type: 'text', value: e.style || '', placeholder: 'doom folk, post rock', spellcheck: 'false' });
  const msg = el('p', { class: 'fhint' });

  const save = async () => {
    try {
      const r = await api.patchTrack(e.id, { title: title.value, style: style.value });
      replaceEntry(r.entry || e);
      toast(stripMd(r.message) || 'Saved.');
      closeSheet();
    } catch (err) { setHint(msg, err.message); msg.className = 'fhint fhint--warn'; }
  };

  const suggest = async (useLlm) => {
    msg.textContent = useLlm ? 'Asking the local model…' : 'Deriving…';
    msg.className = 'fhint';
    try {
      const r = await api.retitle(e.id, useLlm);
      title.value = r.title;
      setHint(msg, stripMd(r.message || `Titled ${r.title} (${r.how})`));
      replaceEntry(r.entry || e);
    } catch (err) { setHint(msg, err.message); msg.className = 'fhint fhint--warn'; }
  };

  openSheet('Edit track', el('div', { class: 'formcol', style: 'padding:0;gap:16px' },
    el('div', { class: 'field' }, el('label', { class: 'flabel', text: 'Title' }), title,
      el('p', { class: 'fhint', text: 'Empty means "derived from the lyrics or prompt".' })),
    el('div', { class: 'field' }, el('label', { class: 'flabel', text: 'Style / tags' }), style,
      el('p', { class: 'fhint', text: 'Shown as the row’s second line, in place of the prompt digest.' })),
    el('div', { class: 'btnrow' },
      el('button', { type: 'button', class: 'btn btn--ghost', text: 'Suggest (LLM)', onclick: () => suggest(true) }),
      el('button', { type: 'button', class: 'btn btn--ghost', text: 'Derive', onclick: () => suggest(false) })),
    msg,
    el('div', { class: 'btnrow' },
      el('button', { type: 'button', class: 'btn', text: 'Cancel', onclick: () => closeSheet() }),
      el('button', { type: 'button', class: 'btn btn--primary', text: 'Save', onclick: save })),
  ));
}

function openRecord(e) {
  const body = el('div', { class: 'formcol', style: 'padding:0;gap:16px' });
  if (e.shared_rec) body.append(el('p', { class: 'note note--warn',
    text: 'Prompt inherited from the sibling take — ACE-Step writes two takes and only one carries the record.' }));
  body.append(el('div', {}, el('p', { class: 'card-h', text: 'Prompt' }),
    el('pre', { class: 'record', text: e.prompt || '(none recorded)' })));
  body.append(el('div', {}, el('p', { class: 'card-h', text: 'Lyrics' }),
    el('pre', { class: 'record', text: e.lyrics || (e.instrumental ? '(instrumental)' : '(none recorded)') })));
  const kv = el('dl', { class: 'kv' });
  // §11 makes provenance a requirement, and the sidecar has carried it for a
  // while — but "which of these used the LoRA" and "epoch 5 or epoch 10" were
  // unanswerable from here. Labelled in the user's terms, not the wire's.
  for (const [k, v] of [['model', e.model], ['seed', e.seed], ['steps', e.steps],
    ['checkpoint', e.variant],
    ['adapter', e.lora_name ? `${e.lora_name}${e.lora_scale != null ? ` at ${e.lora_scale}` : ''}` : null],
    ['thinking', e.thinking ? `on, weirdness ${e.lm_temperature ?? 1}` : null],
    ['quality', e.quality?.preset || null],
    ['workspace', e.workspace],
    ['duration', e.seconds ? `${Math.round(e.seconds)}s` : e.duration], ['sample rate', e.sampling_rate],
    ['elapsed', e.elapsed ? `${Math.round(e.elapsed)}s` : null], ['peak VRAM', e.vram_peak ? `${e.vram_peak} GB` : null],
    ['file', e.name]]) {
    // Load-bearing, not tidiness: most of the 215-track library predates every
    // one of the fields above, and a blank "adapter" would read as a positive
    // claim that the track was made without one.
    if (v === null || v === undefined || v === '') continue;
    kv.append(el('dt', { text: k }), el('dd', { text: String(v) }));
  }
  body.append(kv);
  openSheet(e.title, body);
}

async function openPlaylistPicker(e) {
  const body = el('div', { class: 'formcol', style: 'padding:0;gap:12px' });
  const input = el('input', { type: 'text', placeholder: 'New playlist name' });
  const add = async (name) => {
    name = (name || '').trim();
    if (!name) { toast('Type a playlist name, or pick an existing one.', 'err'); return; }
    try {
      const r = await api.addToPlaylist(name, e.id);
      toast(stripMd(r.message) || 'Added.');
      closeSheet();
      // One row, not the whole list: adding track #90 to a playlist used to
      // collapse the view back to 60 rows and lose the place you were at.
      await patchEntry(e.id);
      // The chip row is the part that IS stale when the name was just
      // created, so repaint the chips — never the rows.
      try {
        const p = await api.getPlaylists();
        state.playlists = (p.playlists || []).map((x) => x.name || x);
        paintChips();
      } catch { /* the track is in the playlist either way */ }
    }
    catch (err) { toast(err.message, 'err'); }
  };
  let names = state.playlists;
  try { const r = await api.getPlaylists(); names = (r.playlists || []).map((p) => p.name || p); } catch {}
  body.append(menu(names.map((n) => ({ icon: 'plus', label: n, onClick: () => add(n) }))));
  body.append(el('div', { class: 'field' }, input,
    el('button', { type: 'button', class: 'btn btn--primary btn--wide', text: 'Create and add', onclick: () => add(input.value) })));
  openSheet('Add to playlist', body);
}

/* ── the desktop detail pane ───────────────────────────────────────────── */
export function select(e) {
  state.selected = e.id;
  renderDetail(e);
}

function detailHost() { return $('#side-library'); }

function clearDetail(msg) {
  const host = detailHost();
  if (host) host.replaceChildren(el('p', { class: 'empty',
    text: msg || 'Pick a track to see its cover, prompt and lyrics.' }));
}

function renderDetail(e) {
  const host = detailHost();
  if (!host) return;
  const [style] = styleLine(e);
  const art = el('div', { class: 'detail-art' });
  if (e.cover_url) art.append(el('img', { src: e.cover_url, alt: '' }));
  const acts = el('div', { class: 'btnrow' },
    el('button', { type: 'button', class: 'btn btn--primary', text: 'Play', onclick: () => player.play(e, state.entries) }),
    el('button', { type: 'button', class: 'btn btn--ghost', text: 'Edit', onclick: () => openEdit(e) }),
    el('button', { type: 'button', class: 'btn btn--ghost', text: 'Reuse', onclick: () => reuse(e) }),
    el('button', { type: 'button', class: 'btn btn--ghost', text: 'Cover', onclick: () => coverThis(e) }),
    el('button', { type: 'button', class: 'btn btn--ghost', text: 'Playlist', onclick: () => openPlaylistPicker(e) }),
    el('button', { type: 'button', class: 'btn btn--ghost btn--danger', text: 'Trash', onclick: () => confirmTrash(e) }));
  host.replaceChildren(el('div', { class: 'side' },
    art,
    el('h2', { class: 'now-title', text: e.title }),
    el('p', { class: 'note', text: style }),
    acts,
    el('div', { class: 'card' }, el('p', { class: 'card-h', text: 'Prompt' }),
      el('pre', { class: 'record', text: e.prompt || '(none recorded)' })),
    el('div', { class: 'card' }, el('p', { class: 'card-h', text: 'Lyrics' }),
      el('pre', { class: 'record', text: e.lyrics || (e.instrumental ? '(instrumental)' : '(none recorded)') }))));
  paintIcons(host);
}

/* Row taps also select, so the desktop pane follows the list. */
export function bindDesktopSelection() {
  listEl.addEventListener('click', (ev) => {
    const r = ev.target.closest('.row');
    if (r) { const e = state.entries.find((x) => x.id === r.dataset.id); if (e) select(e); }
  });
}

/** The now-playing view's five action buttons route through here. */
export function trackAction(e, action) {
  const live = state.entries.find((x) => x.id === e.id) || e;
  switch (action) {
    case 'like':     return rate(live, live.rating === 1 ? 0 : 1);
    case 'dislike':  return rate(live, live.rating === -1 ? 0 : -1);
    case 'cover':    return coverThis(live);
    case 'playlist': return openPlaylistPicker(live);
    default:         return openMore(live);
  }
}

export const entries = () => state.entries;
export const getState = () => state;
