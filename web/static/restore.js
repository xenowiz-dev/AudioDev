/* restore.js — inspect a file's bandwidth, pick a region on the spectrogram,
   upscale, splice.

   The click-to-region maths is client-side arithmetic on the geometry the
   server publishes (spec §6.3). Outside the plot rectangle the click is
   IGNORED, never clamped: a clamped margin click silently produces a
   degenerate region. Every re-render carries a marker drawn where the client
   believes the click landed — that crosshair is the only thing that makes a
   coordinate-space bug visible, so it is not decoration. */

import { $, el, toast, stripMd, setHint, debounce, icon } from './ui.js';
import * as api from './api.js';
import * as store from './store.js';
import * as player from './player.js';
import { JobView } from './jobs.js';

let cfg = null, view = null, hooks = {};
let geom = null, anchor = null, busy = false;
let source = { path: null, uploaded: false, playable: false };
let region = { t0: 0, t1: 0, flo: 16000, fhi: 0 };
const NUM = {};

export function init(config, h = {}) {
  cfg = config; hooks = h;
  region = { ...(cfg.defaults?.region || { t0: 0, t1: 0, flo: 16000, fhi: 0 }) };

  for (const [k, label] of Object.entries(cfg.upscalers)) {
    if (k === 'none') continue;                      // upscale rejects "none"
    $('#r-kind').append(el('option', { value: k, text: label }));
  }
  // The two new selects are built from the server's catalogue, so adding a
  // mode is a backend-only change. A server that predates them sends nothing
  // and the select stays empty-but-harmless, hence the fallbacks.
  for (const [id, cat] of [['#r-degrade', cfg.degrades], ['#r-natural', cfg.naturalizers]]) {
    for (const [k, label] of Object.entries(cat || { off: 'Off' })) {
      $(id).append(el('option', { value: k, text: label }));
    }
  }

  const saved = store.load('restore', {
    kind: cfg.defaults?.upscaler || 'apollo',
    auto_lowpass: cfg.defaults?.auto_lowpass ?? true,
    fill: cfg.defaults?.fill ?? true,
    degrade: cfg.defaults?.degrade || 'off',
    naturalize: cfg.defaults?.naturalize || 'off',
  });
  if ([...$('#r-kind').options].some((o) => o.value === saved.kind)) $('#r-kind').value = saved.kind;
  $('#r-lowpass').checked = !!saved.auto_lowpass;
  $('#r-fill').checked = !!saved.fill;
  setSelect('#r-degrade', saved.degrade, 'off');
  setSelect('#r-natural', saved.naturalize, 'off');
  $('#r-kind-note').textContent = cfg.fingerprint_note || '';

  buildRegionInputs();

  view = new JobView($('#side-restore'), {
    onResult: (r) => onResult(r),
    onEnd: () => { setBusy(false); hooks.onJobEnd?.(); },
    onVram: (v) => hooks.onVram?.(v),
  });
  renderSpec(null);

  $('#r-path').addEventListener('input', debounce(resolve, 400));
  $('#r-upload').addEventListener('change', onUpload);
  $('#r-inspect').addEventListener('click', () => inspect());
  $('#r-cliff').addEventListener('click', cliff);
  $('#r-reset').addEventListener('click', resetRegion);
  $('#r-go').addEventListener('click', upscale);
  for (const id of ['r-kind', 'r-lowpass', 'r-fill', 'r-degrade', 'r-natural']) {
    $('#' + id).addEventListener('change', persist);
  }
  // "Auto" means "the damage this upscaler was trained on", so the hint has to
  // follow the upscaler, not just the degrade select.
  for (const id of ['r-kind', 'r-degrade']) $('#' + id).addEventListener('change', paintDegradeHint);
  paintDegradeHint();
}

/** Select a value if the option exists, else fall back. */
function setSelect(sel, value, fallback) {
  const n = $(sel);
  n.value = [...n.options].some((o) => o.value === value) ? value : fallback;
}

const AUTO_PICK = { apollo: 'an MP3 round-trip', audiosr: 'a cut above 11 kHz', flashsr: 'a cut above 11 kHz' };

function paintDegradeHint() {
  const note = $('#r-degrade-note');
  if (!note) return;
  const mode = $('#r-degrade').value;
  note.textContent = mode === 'auto'
    ? `Auto picks ${AUTO_PICK[$('#r-kind').value] || 'a band cut'} — the damage `
      + `${$('#r-kind').selectedOptions[0]?.text.split('—')[0].trim() || 'this upscaler'} was trained on.`
    : '';
}

function persist() {
  store.save('restore', {
    kind: $('#r-kind').value,
    auto_lowpass: $('#r-lowpass').checked,
    fill: $('#r-fill').checked,
    degrade: $('#r-degrade').value,
    naturalize: $('#r-natural').value,
  });
}

/* ── the four region numbers, each with 44px nudges ────────────────────── */
function buildRegionInputs() {
  const host = $('#region-grid');
  const defs = [
    ['t0', 'Start (s)', 1, 3], ['t1', 'End (s) — 0 = end of file', 1, 3],
    ['flo', 'Low (Hz)', 500, 0], ['fhi', 'High (Hz) — 0 = Nyquist', 500, 0],
  ];
  host.replaceChildren();
  for (const [key, label, step, dp] of defs) {
    const input = el('input', { type: 'number', value: fmt(region[key], dp), inputmode: 'decimal',
                                'aria-label': label });
    NUM[key] = { input, dp };
    input.addEventListener('input', debounce(() => {
      region[key] = Number(input.value) || 0;
      if (source.path) inspect();
    }, 400));
    const bump = (d) => {
      region[key] = Math.max(0, (Number(input.value) || 0) + d * step);
      input.value = fmt(region[key], dp);
      if (source.path) inspectSoon();
    };
    host.append(el('div', { class: 'field' },
      el('label', { class: 'flabel', text: label }),
      el('div', { class: 'nudge' },
        el('button', { type: 'button', class: 'iconbtn', 'aria-label': `${label} down`, text: '−', onclick: () => bump(-1) }),
        input,
        el('button', { type: 'button', class: 'iconbtn', 'aria-label': `${label} up`, text: '+', onclick: () => bump(1) }))));
  }
}

const fmt = (v, dp) => (dp ? Number(v || 0).toFixed(dp).replace(/\.?0+$/, '') || '0' : String(Math.round(v || 0)));

function paintRegionInputs() {
  for (const [k, { input, dp }] of Object.entries(NUM)) input.value = fmt(region[k], dp);
}

/* ── source ────────────────────────────────────────────────────────────── */
async function resolve() {
  const typed = $('#r-path').value.trim().replace(/^"|"$/g, '');
  const msg = $('#r-msg');
  if (!typed) { source = { path: null, uploaded: false, playable: false }; msg.textContent = ''; return; }
  try {
    const st = await api.fsStat(typed);
    if (!st.is_file) { msg.textContent = 'No file at that path.'; source.path = null; return; }
    source = { path: typed, uploaded: false, playable: !!st.playable };
    geom = null; anchor = null;
    msg.textContent = st.playable
      ? `${st.name} — ${(st.size / 1048576).toFixed(1)} MB.`
      : `${st.name} — outside the served folders: processable here, but the browser cannot stream it. Upload it to play it.`;
  } catch (e) { setHint(msg, e.message || 'Could not check that path.'); }
}

async function onUpload(ev) {
  const file = ev.target.files?.[0];
  if (!file) return;
  const msg = $('#r-msg');
  msg.textContent = 'Uploading…';
  try {
    const r = await api.upload(file, (f) => { msg.textContent = `Uploading… ${Math.round(f * 100)}%`; });
    source = { path: r.path, uploaded: true, playable: true, url: r.url };
    $('#r-path').value = r.path;
    geom = null; anchor = null;
    msg.textContent = `Uploaded ${r.name} — ${(r.size / 1048576).toFixed(1)} MB.`;
    inspect();
  } catch (e) { setHint(msg, e.message || 'Upload failed.'); }
  ev.target.value = '';
}

/* ── inspect ───────────────────────────────────────────────────────────── */
let inspectAc = null;
const inspectSoon = debounce(() => inspect(), 400);

async function inspect(marker) {
  if (!source.path) { toast('Pick a file first.', 'err'); return; }
  inspectAc?.abort();
  const ac = new AbortController(); inspectAc = ac;
  setStatus('Rendering the spectrogram…');
  try {
    const r = await api.inspect({ path: source.path, region, marker: marker || null }, { signal: ac.signal });
    geom = r.geometry || null;
    renderSpec(r);
    setStatus(r.truncated_note || r.note || '');
  } catch (e) {
    if (e.name === 'AbortError') return;
    setStatus(e.message || 'Inspect failed.');
  }
}

async function cliff() {
  if (!source.path) { toast('Pick a file first.', 'err'); return; }
  setStatus('Measuring the cliff…');
  try {
    const r = await api.detectCliff({ path: source.path, render: true, region });
    region.flo = r.cliff_hz;
    paintRegionInputs();
    if (r.inspect) { geom = r.inspect.geometry || null; renderSpec(r.inspect); }
    setStatus(r.weak
      ? stripMd(r.message)
      : `Cliff ${Math.round(r.cliff_hz)} Hz, ${r.drop_db?.toFixed?.(1) ?? r.drop_db} dB drop — applied as the low edge.`);
  } catch (e) {
    setStatus(e.code === 'unmeasurable' ? 'Could not measure a cliff for that file.' : (e.message || 'Failed.'));
  }
}

function resetRegion() {
  anchor = null;
  region = { t0: 0, t1: 0, flo: (geom && geom.cliff) || 16000, fhi: 0 };
  paintRegionInputs();
  setStatus(`Reset to whole file above ${(region.flo / 1000).toFixed(2)} kHz.`);
  if (source.path) inspect();
}

function setStatus(t) { const n = $('#r-status'); if (n) n.textContent = stripMd(t || ''); }

/* ── the spectrogram, and the two-click selection ──────────────────────── */
function renderSpec(r) {
  const host = $('#side-restore');
  let card = $('#r-spec-card');
  if (!card) {
    card = el('div', { class: 'side', id: 'r-spec-card' });
    host.prepend(card);
  }
  card.replaceChildren();
  card.append(el('p', { class: 'note', id: 'r-status' }));
  if (r && r.image_url) {
    const img = el('img', { src: r.image_url, alt: 'spectrogram', id: 'r-spec' });
    bindPick(img);
    card.append(el('div', { class: 'card speccard' }, el('div', { class: 'specwrap' }, img)));
  } else {
    card.append(el('div', { class: 'card' },
      el('p', { class: 'note', text: r ? 'The spectrogram could not be rendered — the bandwidth verdict below still stands.' : 'Pick a file and press Inspect to see its spectrogram.' })));
  }
  if (r && r.bandwidth) {
    card.append(el('div', { class: 'card' }, el('p', { class: 'card-h', text: 'Bandwidth' }),
      el('pre', { class: 'record', text: r.bandwidth })));
  }
  if (source.playable && source.url) {
    card.append(el('button', { type: 'button', class: 'btn btn--ghost btn--wide', text: 'Play the source',
      onclick: () => player.loadAdHoc({ url: source.url, title: source.path.split('\\').pop(), sub: 'restore source' }) }));
  }
}

function bindPick(img) {
  let down = null;
  img.style.touchAction = 'manipulation';
  img.addEventListener('pointerdown', (e) => { down = { x: e.clientX, y: e.clientY }; });
  img.addEventListener('pointerup', (e) => {
    if (!down) return;
    const moved = Math.hypot(e.clientX - down.x, e.clientY - down.y);
    const a = moved > 10 ? map(down.x, down.y, img) : null;   // drag = both corners
    const b = map(e.clientX, e.clientY, img);
    down = null;
    if (!b) { setStatus('Click inside the spectrogram (not the margins).'); return; }
    if (a) { commit(a, b); return; }
    if (!anchor) {
      anchor = b;
      setStatus(`Anchor at ${b.t.toFixed(2)}s / ${(b.f / 1000).toFixed(2)} kHz — click the opposite corner.`);
      inspect({ t: b.t, f: b.f });
    } else {
      commit(anchor, b);
    }
  });
}

function map(clientX, clientY, img) {
  if (!geom) return null;
  const rect = img.getBoundingClientRect();
  const sx = geom.img_w / rect.width, sy = geom.img_h / rect.height;
  const x = (clientX - rect.left) * sx, y = (clientY - rect.top) * sy;
  if (!(geom.x0 <= x && x <= geom.x1 && geom.y0 <= y && y <= geom.y1)) return null;
  const t = geom.t0 + (x - geom.x0) / (geom.x1 - geom.x0) * (geom.t1 - geom.t0);
  const f = geom.f0 + (geom.y1 - y) / (geom.y1 - geom.y0) * (geom.f1 - geom.f0);   // image y grows down
  return { t, f };
}

function commit(a, b) {
  region.t0 = +Math.min(a.t, b.t).toFixed(3);
  region.t1 = +Math.max(a.t, b.t).toFixed(3);
  region.flo = Math.round(Math.min(a.f, b.f));
  region.fhi = Math.round(Math.max(a.f, b.f));
  anchor = null;
  paintRegionInputs();
  setStatus(`Region set: ${region.t0}–${region.t1}s, ${(region.flo / 1000).toFixed(2)}–${(region.fhi / 1000).toFixed(2)} kHz. Click again to start a new one.`);
  inspect({ t: b.t, f: b.f });
}

/* ── upscale ───────────────────────────────────────────────────────────── */
async function upscale() {
  if (busy) { toast('A GPU job is already running.', 'err'); return; }
  if (!source.path) { toast('Pick a file first.', 'err'); return; }
  try {
    setBusy(true);
    const r = await api.postUpscale({
      path: source.path, kind: $('#r-kind').value,
      auto_lowpass: $('#r-lowpass').checked, fill: $('#r-fill').checked,
      region, compare: $('#r-compare').checked,
      degrade: $('#r-degrade').value, naturalize: $('#r-natural').value,
    });
    hooks.onJob?.(r.job_id);
    view.attach(r.job_id, { kind: 'upscale' });
  } catch (e) { setBusy(false); toast(e.message, 'err'); }
}

function setBusy(on) {
  busy = on;
  $('#r-go').disabled = on;
  $('#r-go').textContent = on ? 'Upscaling…' : 'Upscale';
}

function onResult(r) {
  const chain = ['input'];
  if (r.degraded_url) chain.push(`degraded (${r.degrade})`);
  chain.push(r.kind);
  if (r.spliced) chain.push('spliced');
  if (r.natural_url) chain.push(`analogue (${r.naturalize})`);

  view.results.replaceChildren(el('div', { class: 'card' },
    el('p', { class: 'card-h', text: r.natural_url ? 'Finished (analogue pass)' : (r.spliced ? 'Hybrid (spliced)' : 'Upscaled') }),
    el('p', { class: 'note', text: chain.join(' → ') }),
    el('p', { class: 'note', text: `region ${r.region}` }),
    r.bandwidth_degraded ? el('pre', { class: 'record', text: r.bandwidth_degraded }) : null,
    r.degraded_url ? el('div', { class: 'btnrow' },
      el('button', { type: 'button', class: 'btn btn--ghost', text: 'Play the degraded input',
        onclick: () => player.loadAdHoc({ url: r.degraded_url, title: (r.degraded_path || '').split('\\').pop(), sub: `degraded · ${r.degrade}` }) })) : null,
    el('p', { class: 'note', text: 'Upscaler output lives in upscaled\\ and is not a library member — playable, but it will not appear in the list.' }),
    el('div', { class: 'btnrow' },
      el('button', { type: 'button', class: 'btn btn--primary', text: 'Play the result',
        onclick: () => player.loadAdHoc({ url: r.url, title: (r.path || '').split('\\').pop(), sub: `${r.kind}${r.spliced ? ' · spliced' : ''}` }) }),
      // The A/B: "more natural" is a claim only your ears can settle.
      r.pre_natural_url ? el('button', { type: 'button', class: 'btn btn--ghost', text: 'A/B: without the analogue pass',
        onclick: () => player.loadAdHoc({ url: r.pre_natural_url, title: (r.pre_natural_path || '').split('\\').pop(), sub: 'before the analogue pass' }) }) : null,
      r.upscaled_url && r.spliced ? el('button', { type: 'button', class: 'btn btn--ghost', text: 'Play whole-file version',
        onclick: () => player.loadAdHoc({ url: r.upscaled_url, title: (r.upscaled_path || '').split('\\').pop(), sub: r.kind }) }) : null),
    r.bandwidth_after ? el('pre', { class: 'record', text: r.bandwidth_after }) : null));
  if (r.compare_url) {
    view.results.append(el('div', { class: 'card speccard' },
      el('div', { class: 'specwrap' }, el('img', { src: r.compare_url, alt: 'before and after' }))));
  }
  player.loadAdHoc({ url: r.url, title: (r.path || '').split('\\').pop(), sub: `${r.kind}${r.spliced ? ' · spliced' : ''}` });
}

export const jobView = () => view;

/** The Restore tab can be handed a file from elsewhere (the library's
    "Send to Restore"). Inspect immediately: arriving to an empty pane and
    having to press Inspect is the clunky version. */
export async function setSource(path, title) {
  $('#r-path').value = path || '';
  region = { ...(cfg?.defaults?.region || { t0: 0, t1: 0, flo: 16000, fhi: 0 }) };
  geom = null; anchor = null;
  paintRegionInputs();
  await resolve();
  if (title) setStatus(`${title} — inspecting…`);
  if (source.path) await inspect();
}
