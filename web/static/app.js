/* app.js — boot, navigation, the mobile/desktop reflow, VRAM polling.

   Panes are three <section>s that stay in the DOM with `hidden` toggled, never
   re-rendered: leaving gradio was largely about form state surviving a tab
   switch, and a re-render would throw that away again. */

import { $, $$, el, icon, paintIcons, toast, setIcon, openSheet, closeSheet, stripMd } from './ui.js';
import * as api from './api.js';
import * as player from './player.js';
import * as library from './library.js';
import * as create from './create.js';
import * as restore from './restore.js';

const TABS = [
  { id: 'library', label: 'Library', icon: 'library' },
  { id: 'create',  label: 'Create',  icon: 'create' },
  { id: 'restore', label: 'Restore', icon: 'restore' },
];

const WIDE = matchMedia('(min-width: 900px)');
let tab = 'library';
let cfg = null;
const scrollPos = {};
let vramTimer = null, streaming = 0;

/* ── navigation ────────────────────────────────────────────────────────── */
function buildNav() {
  for (const host of [$('#tabbar'), $('#rail')]) {
    host.replaceChildren(...TABS.map((t) => el('button', {
      type: 'button', class: 'tab', dataset: { tab: t.id },
      'aria-label': t.label, onclick: () => go(t.id),
    }, el('span', { html: icon(t.icon) }), el('span', { class: 'tab-l', text: t.label }))));
  }
}

function go(id) {
  if (id === tab) return;
  scrollPos[tab] = $('#main').scrollTop;
  tab = id;
  for (const t of TABS) $(`#pane-${t.id}`).hidden = t.id !== id;
  for (const b of $$('.tab')) {
    const on = b.dataset.tab === id;
    b.classList.toggle('tab--on', on);
    if (on) b.setAttribute('aria-current', 'page'); else b.removeAttribute('aria-current');
  }
  // The accent follows the pane: app.css re-points --accent off this
  // attribute, so one rule set carries library/create/restore.
  document.body.dataset.pane = id;
  $('#appbar-title').textContent = TABS.find((t) => t.id === id).label;
  $('#chiprow').hidden = id !== 'library';
  if (id !== 'library') { $('#searchfield').hidden = true; $('#search-open').hidden = true; $('#appbar-title').hidden = false; }
  else $('#search-open').hidden = !$('#searchfield').hidden;
  for (const s of ['library', 'create', 'restore']) $(`#detailslot-${s}`).hidden = s !== id;
  $('#main').scrollTop = scrollPos[id] || 0;
  reflow();
}

/* ── mobile/desktop reflow: the same nodes, a different container ──────── */
function reflow() {
  const wide = WIDE.matches;
  $('#detail').hidden = !wide;
  for (const name of ['library', 'create', 'restore']) {
    const side = $(`#side-${name}`);
    const home = wide ? $(`#detailslot-${name}`) : $(`#homeslot-${name}`);
    if (side && side.parentElement !== home) home.append(side);
  }
  // The library's detail is a sheet on a phone and a pane on a desktop, so its
  // mobile home is a hidden holding pen rather than a visible slot.
  $('#homeslot-library').hidden = true;

  // The scrubber is a 12px band over the mini bar on a phone and a real inline
  // slider with elapsed/total on a desktop. Same node, two homes.
  const seek = $('#dock-seek'), extra = $('#dock-extra'), dock = $('#dock');
  if (wide && seek.parentElement !== extra) extra.prepend(seek);
  else if (!wide && seek.parentElement !== dock) dock.prepend(seek);

  if (wide) { $('#searchfield').hidden = false; $('#search-open').hidden = true; $('#appbar-title').hidden = false; }
  else if (tab === 'library' && $('#searchfield').hidden) $('#search-open').hidden = false;
}

/* ── VRAM pill ─────────────────────────────────────────────────────────── */
function paintVram(v) {
  const chip = $('#gpu-chip'), text = $('#gpu-text');
  if (!v || v.available === false) {
    text.textContent = 'GPU: unavailable';
    chip.className = 'gpu-chip'; chip.title = 'nvidia-smi returned nothing.';
    return;
  }
  text.textContent = `${v.used_gb?.toFixed?.(1) ?? v.used_gb} / ${v.total_gb} GB`;
  chip.className = 'gpu-chip' + (v.loaded ? ' gpu-chip--loaded' : (v.warn_low_free ? ' gpu-chip--warn' : ''));
  chip.title = (v.loaded
    ? `${v.loaded_label || v.loaded} is resident`
    : (v.warn_low_free
        ? '⚠ under 8 GB free — MiniMax will spill to system memory and run several times slower'
        : `${v.free_gb} GB free`)) + ' — tap for jobs and recovery';
}

async function pollVram() {
  if (document.visibilityState !== 'visible' || streaming > 0) return;
  try { paintVram(await api.getVram()); } catch { /* transient */ }
}

function startVram() {
  clearInterval(vramTimer);
  vramTimer = setInterval(pollVram, 8000);
  pollVram();
}

/* ── the on-screen keyboard (§5) ───────────────────────────────────────── */
function bindKeyboard() {
  const vv = window.visualViewport;
  if (!vv) return;
  const onResize = () => {
    const kb = Math.max(0, window.innerHeight - vv.height - vv.offsetTop);
    document.documentElement.style.setProperty('--kb', kb + 'px');
    document.body.classList.toggle('is-typing', kb > 120);
  };
  vv.addEventListener('resize', onResize);
  vv.addEventListener('scroll', onResize);
}

/* ── the GPU sheet: the recovery path ──────────────────────────────────────
   There is exactly one GPU and one lane, so when something wedges there is
   nothing else to try — you need to see what is holding it and be able to take
   it away. Tapping the chip opens this rather than silently unloading:
     * every live job, with its own Cancel (queued jobs too)
     * Unload model, when the lane is clear
     * Force reset, which cancels the queue AND kills whatever is running
   Force is the only action here that works when a job is stuck, which is the
   whole point of it. */
let gpuSheetTimer = null;
// The 3 s repaint rebuilds every button, which would wipe an armed "tap again"
// or a "Cancelling…" mid-request. Any in-flight action holds this.
let gpuSheetLock = false;
// The wrap of the sheet that is currently on screen. A force reset can still
// be polling when the user swipes the sheet away and opens it again: without
// this, the old instance's final repaint would clear the NEW sheet's interval
// and leave it frozen on "Reading GPU state…".
let gpuSheetWrap = null;

/* ── settings ───────────────────────────────────────────────────────────
   Studio-wide values, as opposed to the per-render knobs on Create. Today
   that is the artist name credited on the lock screen.

   The field starts EMPTY and its placeholder is generic. It replaced a
   hard-coded name, so pre-filling a suggestion here would re-invent exactly
   the label this exists to remove -- the credit is the user's to type. */
async function openSettingsSheet() {
  const input = el('input', {
    type: 'text', id: 'set-artist', placeholder: 'Artist name',
    autocomplete: 'off', spellcheck: 'false', maxlength: '64',
  });
  const msg = el('p', { class: 'fhint', id: 'set-msg' });
  const save = el('button', { type: 'button', class: 'btn btn--primary', text: 'Save' });

  const body = el('div', { class: 'formcol', style: 'padding:0;gap:16px' },
    el('div', { class: 'field' },
      el('label', { for: 'set-artist', text: 'Artist name' }),
      input,
      el('p', { class: 'fhint', text: 'Shown on the lock screen and with AirPods. '
                                    + 'Leave it empty and no credit is sent.' })),
    msg,
    el('div', { class: 'btnrow' },
      el('button', { type: 'button', class: 'btn', text: 'Close', onclick: () => closeSheet() }),
      save));

  openSheet('Settings', body);

  // Read the stored value rather than trusting the cached one: another device
  // may have changed it since this page loaded.
  try {
    const d = await api.getProfile();
    input.value = d.artist || '';
    player.setArtist(d.artist || '');
  } catch (e) {
    msg.textContent = `Could not read the current setting: ${e.message}`;
  }

  const commit = async () => {
    save.disabled = true;
    try {
      const d = await api.setProfile(input.value);
      input.value = d.artist || '';
      player.setArtist(d.artist || '');      // repaints the CURRENT track's credit
      msg.textContent = d.note || '';
      toast(d.artist ? `Credited to ${d.artist}.` : 'Artist name cleared.');
      if (!d.note) closeSheet();
    } catch (e) {
      msg.textContent = e.message;
    } finally {
      save.disabled = false;
    }
  };
  save.addEventListener('click', commit);
  input.addEventListener('keydown', (e) => { if (e.key === 'Enter') commit(); });
  setTimeout(() => input.focus(), 60);
}

function openGpuSheet() {
  // Everything is rendered into `wrap`, not into the sheet body, so the poll
  // can tell whether it is still on screen: opening another sheet REPLACES the
  // body's children without firing onClose, which would otherwise leave this
  // timer redrawing GPU rows over someone else's sheet.
  const wrap = el('div', { class: 'side' },
    el('p', { class: 'note', text: 'Reading GPU state…' }));
  openSheet('GPU', wrap, {
    onClose: () => { clearInterval(gpuSheetTimer); gpuSheetTimer = null; },
  });
  gpuSheetWrap = wrap;
  gpuSheetLock = false;
  paintGpuSheet(wrap);
  clearInterval(gpuSheetTimer);
  gpuSheetTimer = setInterval(() => paintGpuSheet(wrap), 3000);
}

const JOB_LABEL = { generate: 'Generating', upscale: 'Upscaling', transcribe: 'Transcribing', write: 'Writing', test: 'Test job' };

async function paintGpuSheet(body, force = false) {
  // Never touch gpuSheetTimer here — openGpuSheet and its onClose own it. A
  // stale instance must go quiet, not tear down the live one.
  if (body !== gpuSheetWrap || !body.isConnected) return;
  if (gpuSheetLock && !force) return;
  const [v, jl] = await Promise.all([
    api.getVram().catch(() => null),
    api.listJobs(true).catch(() => ({ jobs: [] })),
  ]);
  if (v) paintVram(v);
  const jobs = jl?.jobs || [];
  const idle = jobs.length === 0;

  const rows = jobs.map((j) => {
    const running = j.state === 'running' || j.state === 'starting';
    const detail = running
      ? (j.live?.msg || 'running…')
      : `queued — position ${j.queue_position ?? 0}`;
    const btn = el('button', {
      type: 'button', class: 'btn btn--ghost', text: 'Cancel',
      onclick: async () => {
        gpuSheetLock = true;
        btn.disabled = true; btn.textContent = 'Cancelling…';
        try { await api.cancelJob(j.id); toast('Cancelling…'); }
        catch (e) { btn.disabled = false; btn.textContent = 'Cancel'; toast(e.message, 'err'); }
        setTimeout(() => { gpuSheetLock = false; paintGpuSheet(body, true); }, 1500);
      },
    });
    return el('div', { class: 'card' },
      el('div', { class: 'gpu-job' },
        el('div', { class: 'meta' },
          el('strong', { text: `${JOB_LABEL[j.kind] || j.kind}${running ? '' : ' (queued)'}` }),
          el('span', { class: 'note', text: stripMd(detail) })),
        btn));
  });

  const forceBtn = el('button', {
    type: 'button', class: 'btn btn--wide btn--danger', text: 'Force reset GPU',
  });
  forceBtn.addEventListener('click', () => forceReset(forceBtn, body));

  // replaceChildren() stringifies a null into the literal text "null", so the
  // conditional button is filtered out rather than passed through.
  body.replaceChildren(...[
    el('p', {
      class: 'note',
      text: v
        ? `${v.used_gb} / ${v.total_gb} GB used · ${v.free_gb} GB free` +
          (v.loaded ? ` · ${v.loaded_label || v.loaded} resident` : ' · nothing resident')
        : 'GPU state unavailable.',
    }),
    ...rows,
    idle ? el('button', {
      type: 'button', class: 'btn btn--wide', text: 'Unload model',
      disabled: !v?.loaded,
      onclick: async (e) => {
        gpuSheetLock = true;
        e.currentTarget.disabled = true;
        try { const r = await api.unloadGpu(); paintVram(r.vram); toast('Unloaded — the GPU is free.'); }
        catch (err) { toast(err.message, 'err'); }
        gpuSheetLock = false;
        paintGpuSheet(body, true);
      },
    }) : null,
    forceBtn,
    el('p', {
      class: 'note',
      text: 'Force reset cancels everything on the queue, kills the running job, '
          + 'and frees the memory. Use it when a job is stuck or the GPU stays full.',
    }),
  ].filter(Boolean));
  paintIcons(body);
}

/** Two taps, because it throws away work — but no browser confirm() dialog,
    which would block the page on a phone. */
async function forceReset(btn, body) {
  if (btn.dataset.armed !== '1') {
    btn.dataset.armed = '1';
    btn.textContent = 'Tap again to force reset';
    gpuSheetLock = true;
    setTimeout(() => {
      if (btn.dataset.armed !== '1') return;      // already fired
      btn.dataset.armed = ''; btn.textContent = 'Force reset GPU';
      gpuSheetLock = false;
    }, 5000);
    return;
  }
  btn.dataset.armed = ''; btn.disabled = true; btn.textContent = 'Resetting…';
  gpuSheetLock = true;
  const done = (msg, kind) => {
    if (msg) toast(msg, kind);
    gpuSheetLock = false;
    paintGpuSheet(body, true);
  };

  let r = null;
  try { r = await api.unloadGpu(true); }
  catch (e) { done(e.message || 'Force reset failed.', 'err'); return; }

  if (!r.killing) {
    paintVram(r.vram);
    done(r.cancelled?.length
      ? `Cleared ${r.cancelled.length} queued job${r.cancelled.length > 1 ? 's' : ''} — the GPU is free.`
      : 'The GPU is free.');
    return;
  }
  // The kill is asynchronous: the process has to die and the model has to be
  // released before the memory actually comes back. Poll rather than lie.
  toast('Killing the job — freeing memory…');
  for (let i = 0; i < 20; i++) {
    await new Promise((res) => setTimeout(res, 1500));
    const v = await api.getVram().catch(() => null);
    if (v) paintVram(v);
    const jl = await api.listJobs(true).catch(() => null);
    if (jl && jl.jobs.length === 0) {
      done(v?.loaded ? 'Job stopped — the model is still resident.'
                     : 'Job stopped — the GPU is free.');
      return;
    }
  }
  done('Still shutting down. Give it a moment, then check again.', 'err');
}

/* ── re-attach to whatever is running (§2.9) ───────────────────────────── */
async function reattach() {
  try {
    const { jobs = [] } = await api.listJobs(true);
    for (const j of jobs) {
      // A write owns its own view inside the writer sheet; attaching it to the
      // Create pane would tear down whatever render is streaming there.
      if (j.kind === 'write') continue;
      if (j.kind === 'upscale') { restore.jobView?.()?.attach?.(j.id, { kind: j.kind }); go('restore'); }
      else { create.jobView()?.attach(j.id, { kind: j.kind }); go('create'); }
      toast(`Re-attached to a running ${j.kind} job.`);
      break;
    }
  } catch { /* the endpoint may not be up yet */ }
}

/* ── boot ──────────────────────────────────────────────────────────────── */
async function boot() {
  await api.init();
  paintIcons();
  buildNav();
  bindKeyboard();

  try {
    cfg = await api.getConfig();
  } catch (e) {
    document.body.append(el('div', { class: 'toast toast--err', style: 'position:fixed;left:12px;right:12px;bottom:12px;z-index:99',
      text: `Could not reach the server: ${e.message}. Append ?mock=1 to browse the UI with canned data.` }));
    return;
  }

  player.init({
    onMore: (e) => library.openMore(e),
    onAction: (e, action) => library.trackAction(e, action),
  });

  // Before any track can start, so the first lock screen is already right.
  // A failure here is not fatal: no credit is the correct unset state anyway.
  api.getProfile().then((d) => player.setArtist(d.artist || '')).catch(() => {});

  library.init({
    onReuse: (r) => { create.applyReuse(r); go('create'); },
    onCover: (r) => { create.applyCover(r); go('create'); },
    onRestore: (e) => { restore.setSource(e.path, e.title); go('restore'); },
  });
  library.bindDesktopSelection();

  // While a job stream is attached its `vram` events supersede polling (§4.3).
  const onJob = () => { streaming++; };
  const onJobEnd = () => { streaming = Math.max(0, streaming - 1); pollVram(); };

  create.init(cfg, {
    onVram: paintVram, onJob, onJobEnd,
    onGenerated: () => { library.refresh({ thenAgainIn: 2000 }); },
    scrollToSide: () => { if (!WIDE.matches) $('#side-create').scrollIntoView({ behavior: 'smooth', block: 'start' }); },
  });
  restore.init(cfg, { onVram: paintVram, onJob, onJobEnd });

  go('create'); go('library');            // paint both, land on Library
  reflow();
  WIDE.addEventListener('change', reflow);

  await library.refresh();
  startVram();
  document.addEventListener('visibilitychange', () => { if (document.visibilityState === 'visible') pollVram(); });

  $('#gpu-chip').addEventListener('click', openGpuSheet);
  $('#settings-open').addEventListener('click', openSettingsSheet);
  // JobView raises this from its "Free the GPU" button after a failure.
  document.addEventListener('gpu-recover', openGpuSheet);

  reattach();
}

boot();
