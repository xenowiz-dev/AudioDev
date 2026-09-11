/* ui.js — DOM helpers, inline SVG icons, toasts, bottom sheets.
   No dependencies. Every icon is inline: the page must work with no network. */

export const $  = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

export function el(tag, attrs = {}, ...kids) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === null || v === undefined || v === false) continue;
    if (k === 'class') n.className = v;
    else if (k === 'text') n.textContent = v;
    else if (k === 'html') n.innerHTML = v;
    else if (k === 'dataset') Object.assign(n.dataset, v);
    else if (k.startsWith('on')) n.addEventListener(k.slice(2), v);
    else if (v === true) n.setAttribute(k, '');
    else n.setAttribute(k, v);
  }
  for (const kid of kids.flat()) {
    if (kid === null || kid === undefined || kid === false) continue;
    n.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
  }
  return n;
}

/* ── icons ─────────────────────────────────────────────────────────────── */
const P = (d, extra = '') =>
  `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"
        stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${d}${extra}</svg>`;

export const ICONS = {
  library:  P('<path d="M4 6h16M4 12h16M4 18h10"/>'),
  create:   P('<circle cx="12" cy="12" r="9"/><path d="M12 8v8M8 12h8"/>'),
  restore:  P('<path d="M2 12h3l2-6 3 13 3-16 3 12 2-3h4"/>'),
  search:   P('<circle cx="11" cy="11" r="6.5"/><path d="M20 20l-4.2-4.2"/>'),
  x:        P('<path d="M6 6l12 12M18 6L6 18"/>'),
  more:     P('<circle cx="12" cy="5" r="1.6" fill="currentColor" stroke="none"/><circle cx="12" cy="12" r="1.6" fill="currentColor" stroke="none"/><circle cx="12" cy="19" r="1.6" fill="currentColor" stroke="none"/>'),
  heart:    P('<path d="M12 20s-7-4.5-7-9.3A3.9 3.9 0 0 1 12 8a3.9 3.9 0 0 1 7 2.7C19 15.5 12 20 12 20z"/>'),
  heartOn:  P('<path d="M12 20s-7-4.5-7-9.3A3.9 3.9 0 0 1 12 8a3.9 3.9 0 0 1 7 2.7C19 15.5 12 20 12 20z" fill="currentColor"/>'),
  down:     P('<path d="M7 4v9M17 4h2a1 1 0 0 1 1 1v7a1 1 0 0 1-1 1h-2M17 4H9.5L7 13l1.2 5a2 2 0 0 0 2.6 1.3L11 19l1.6-6H17z"/>'),
  star:     P('<path d="M12 4l2.3 4.9 5.2.7-3.8 3.7 1 5.3L12 16l-4.7 2.6 1-5.3L4.5 9.6l5.2-.7z"/>'),
  play:     P('<path d="M8 5.5v13l11-6.5z" fill="currentColor"/>'),
  pause:    P('<path d="M9 5v14M15 5v14" stroke-width="2.6"/>'),
  next:     P('<path d="M6 5.5v13l9-6.5z" fill="currentColor"/><path d="M18 5v14" stroke-width="2.2"/>'),
  prev:     P('<path d="M18 5.5v13L9 12z" fill="currentColor"/><path d="M6 5v14" stroke-width="2.2"/>'),
  shuffle:  P('<path d="M17 4l3 3-3 3M17 14l3 3-3 3M4 7h3.5l9 10H20M4 17h3.5l2.6-2.9M20 7h-3.4l-2.6 2.9"/>'),
  repeat:   P('<path d="M4 11V9a3 3 0 0 1 3-3h11l-3-3M20 13v2a3 3 0 0 1-3 3H6l3 3"/>'),
  repeat1:  P('<path d="M4 11V9a3 3 0 0 1 3-3h11l-3-3M20 13v2a3 3 0 0 1-3 3H6l3 3"/><path d="M12 9.5v5M12 9.5l-1.2 1" stroke-width="1.6"/>'),
  'chevron-down': P('<path d="M6 9.5l6 6 6-6"/>'),
  chevron:  P('<path d="M6 9.5l6 6 6-6"/>'),
  note:     P('<path d="M9 18V6l10-2v12"/><circle cx="7" cy="18" r="2.4"/><circle cx="17" cy="16" r="2.4"/>'),
  edit:     P('<path d="M4 20h4L20 8l-4-4L4 16z"/>'),
  copy:     P('<rect x="9" y="9" width="11" height="11" rx="2"/><path d="M15 5H6a2 2 0 0 0-2 2v9"/>'),
  trash:    P('<path d="M4 7h16M9 7V4h6v3M6 7l1 13h10l1-13"/>'),
  plus:     P('<path d="M12 5v14M5 12h14"/>'),
  reuse:    P('<path d="M20 12a8 8 0 1 1-2.6-5.9M20 4v5h-5"/>'),
  cover:    P('<circle cx="12" cy="12" r="8"/><circle cx="12" cy="12" r="2.2"/>'),
  art:      P('<rect x="4" y="4" width="16" height="16" rx="2.5"/><path d="M4 15l4.5-4 4 3.5L16 11l4 4"/><circle cx="9" cy="9" r="1.3"/>'),
  download: P('<path d="M12 4v11M7.5 11L12 15.5 16.5 11M5 20h14"/>'),
  text:     P('<path d="M5 6h14M5 11h14M5 16h9"/>'),
  gpu:      P('<rect x="3" y="6" width="18" height="12" rx="2"/><path d="M8 10h8v4H8z"/>'),
  check:    P('<path d="M5 12.5l4.5 4.5L19 7"/>'),
  settings: P('<circle cx="12" cy="12" r="3.2"/><path d="M12 3.2v2.1M12 18.7v2.1M20.8 12h-2.1M5.3 12H3.2M18.2 5.8l-1.5 1.5M7.3 16.7l-1.5 1.5M18.2 18.2l-1.5-1.5M7.3 7.3L5.8 5.8"/>'),
};

export function icon(name) {
  return ICONS[name] || ICONS.note;
}

/** Fill every [data-icon] under root with its inline SVG (idempotent). */
export function paintIcons(root = document) {
  for (const n of $$('[data-icon]', root)) {
    if (n.dataset.painted === n.dataset.icon) continue;
    n.innerHTML = icon(n.dataset.icon);
    n.dataset.painted = n.dataset.icon;
  }
}

export function setIcon(node, name) {
  if (!node) return;
  node.dataset.icon = name;
  node.innerHTML = icon(name);
  node.dataset.painted = name;
}

/* ── toasts ────────────────────────────────────────────────────────────── */
let toastHost = null;
export function toast(msg, kind = '') {
  if (!msg) return;
  toastHost = toastHost || $('#toasts');
  const t = el('div', { class: 'toast' + (kind === 'err' ? ' toast--err' : ''), text: stripMd(msg) });
  toastHost.append(t);
  setTimeout(() => t.remove(), kind === 'err' ? 6000 : 3400);
}

/** The API returns markdown-ish strings (**bold**, *italic*). Flatten them. */
export function stripMd(s) {
  return String(s == null ? '' : s).replace(/\*\*(.+?)\*\*/g, '$1').replace(/\*(.+?)\*/g, '$1');
}

/** Put a server message in an inline hint line without letting it eat the form.
 *
 *  The helpers under B:\AudioDev shell out to other venvs, and when one of them
 *  dies its stderr comes back as the error message -- lyrics_probe.py raises
 *  UnicodeEncodeError on a cp1252 console for any track carrying emoji or CJK,
 *  and the whole traceback was landing in a 12px `.fhint` on a 390px screen.
 *  Keep the first meaningful line, cap it, and park the full text in the title
 *  so nothing is actually lost.
 */
export function setHint(node, s, max = 160) {
  const full = String(s == null ? '' : s).trim();
  const first = full.split('\n').map((x) => x.trim()).filter(Boolean)[0] || '';
  node.textContent = first.length > max ? first.slice(0, max - 1).trimEnd() + '…' : first;
  if (full && full !== node.textContent) node.title = full;
  else node.removeAttribute('title');
}

/* ── modal stack: one history entry per open layer ─────────────────────── */
const stack = [];
let popBound = false, suppress = 0;

function bindPop() {
  if (popBound) return;
  popBound = true;
  window.addEventListener('popstate', () => {
    // A programmatic close already removed its own entry and called back();
    // swallow exactly that many popstates so an older layer is not also closed.
    if (suppress > 0) { suppress--; return; }
    const top = stack.pop();
    if (top) top.close(true);
  });
}

/** Push a closable layer. `close(fromPop)` must tear the layer down. */
export function pushLayer(layer) {
  bindPop();
  stack.push(layer);
  history.pushState({ layer: stack.length }, '');
}

/** Drop a layer's history entry. The caller still tears its own layer down. */
export function popLayer(layer) {
  const i = stack.lastIndexOf(layer);
  if (i < 0) return false;
  stack.splice(i, 1);
  suppress++;
  history.back();
  return true;
}

/* ── bottom sheet ──────────────────────────────────────────────────────── */
let sheetEl, scrimEl, sheetBody, sheetTitle, sheetOpen = false, onSheetClose = null;

function initSheet() {
  if (sheetEl) return;
  sheetEl = $('#sheet'); scrimEl = $('#scrim');
  sheetBody = $('#sheet-body'); sheetTitle = $('#sheet-title');
  $('#sheet-close').addEventListener('click', () => closeSheet());
  scrimEl.addEventListener('click', () => closeSheet());
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape' && sheetOpen) closeSheet(); });
  dragToDismiss($('#sheet-grab'), sheetEl, () => closeSheet());
}

const sheetLayer = { close: () => teardownSheet() };

export function openSheet(title, content, opts = {}) {
  initSheet();
  // A sheet replacing a sheet SWAPS IN PLACE and keeps the same history entry.
  // Closing then reopening would fire history.back() and history.pushState()
  // in one tick, and the browser is free to resolve that traversal against
  // the wrong entry — which walks the page off the app entirely.
  const swap = sheetOpen;
  sheetTitle.textContent = title || '';
  sheetBody.replaceChildren();
  if (typeof content === 'string') sheetBody.innerHTML = content;
  else if (content) sheetBody.append(content);
  paintIcons(sheetBody);
  onSheetClose = opts.onClose || null;
  sheetEl.hidden = false; scrimEl.hidden = false;
  requestAnimationFrame(() => { sheetEl.classList.add('is-open'); scrimEl.classList.add('is-open'); });
  sheetBody.scrollTop = 0;
  sheetOpen = true;
  if (!swap) pushLayer(sheetLayer);
  return sheetBody;
}

export function closeSheet() {
  if (!sheetOpen) return;
  popLayer(sheetLayer);
  teardownSheet();
}

function teardownSheet(immediate = false) {
  if (!sheetOpen) return;
  sheetOpen = false;
  sheetEl.classList.remove('is-open'); scrimEl.classList.remove('is-open');
  sheetEl.style.transform = '';
  const done = () => { sheetEl.hidden = true; scrimEl.hidden = true; };
  if (immediate) done(); else setTimeout(done, 260);
  const cb = onSheetClose; onSheetClose = null;
  if (cb) cb();
}

/** A vertical menu of {icon,label,onClick,danger} for the ⋯ sheet. */
export function menu(items) {
  const wrap = el('div', { class: 'menu' });
  for (const it of items) {
    if (it === '-') { wrap.append(el('div', { class: 'menu-sep' })); continue; }
    if (!it) continue;
    const b = el('button', {
      type: 'button',
      class: 'menu-item' + (it.danger ? ' menu-item--danger' : ''),
      onclick: (e) => { e.stopPropagation(); it.onClick && it.onClick(e); },
    }, el('span', { class: 'menu-i', html: icon(it.icon || 'note') }), el('span', { text: it.label }));
    wrap.append(b);
  }
  return wrap;
}

/* ── drag-to-dismiss (sheets and the now-playing view) ─────────────────── */
export function dragToDismiss(handle, panel, onDismiss, axis = 'y') {
  if (!handle || !panel) return;
  let start = null, last = null, moved = 0;
  handle.addEventListener('pointerdown', (e) => {
    if (e.button) return;
    start = { x: e.clientX, y: e.clientY, t: performance.now() };
    last = start; moved = 0;
    handle.setPointerCapture?.(e.pointerId);
    panel.style.transition = 'none';
  });
  handle.addEventListener('pointermove', (e) => {
    if (!start) return;
    const d = axis === 'y' ? e.clientY - start.y : e.clientX - start.x;
    moved = Math.max(moved, Math.abs(d));
    if (d > 0) panel.style.transform = axis === 'y' ? `translateY(${d}px)` : `translateX(${d}px)`;
    last = { x: e.clientX, y: e.clientY, t: performance.now() };
  });
  const end = (e) => {
    if (!start) return;
    const d = axis === 'y' ? (last.y - start.y) : (last.x - start.x);
    const dt = Math.max(1, last.t - start.t);
    panel.style.transition = '';
    panel.style.transform = '';
    const fling = d / dt > 0.5;
    start = null;
    if (d > 96 || (fling && d > 24)) onDismiss();
  };
  handle.addEventListener('pointerup', end);
  handle.addEventListener('pointercancel', end);
}

/* ── misc formatting ───────────────────────────────────────────────────── */
export function fmtTime(sec) {
  if (!isFinite(sec) || sec < 0) return '0:00';
  const s = Math.floor(sec % 60), m = Math.floor(sec / 60);
  return `${m}:${String(s).padStart(2, '0')}`;
}
export function fmtDur(sec) {
  if (!isFinite(sec) || sec <= 0) return '';
  return sec < 60 ? `${Math.round(sec)}s` : fmtTime(sec);
}
export function pad(t) { return `[${String(Math.round(t)).padStart(5, ' ')}s] `; }
export function debounce(fn, ms) {
  let h; return (...a) => { clearTimeout(h); h = setTimeout(() => fn(...a), ms); };
}
