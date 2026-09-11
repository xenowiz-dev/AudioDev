/* create.js — the generate form.

   Two ways the form gets values, and they must stay distinct (spec §11):
     * a HUMAN model change runs the cascade (stock prompt, stock steps, panel
       visibility, the info copy);
     * a PROGRAMMATIC set (Reuse, Cover, localStorage restore) applies panel
       visibility ONLY — running the cascade would overwrite the very prompt
       being restored. */

import { $, $$, el, toast, stripMd, setHint, debounce, openSheet, closeSheet } from './ui.js';
import * as api from './api.js';
import * as store from './store.js';
import * as player from './player.js';
import { JobView } from './jobs.js';

let cfg = null, view = null, hooks = {};
let src = { path: null, uploaded: false, url: null };
let foundLyrics = null;
let busy = false;

/* Cover mode is a MODE, not a panel (§7): it pins the model, renames the
   primary button and puts a banner at the top of the form. `coverInfo` is
   presentation only — the title and art of the track being covered. */
let coverOn = false;
let coverInfo = { title: null, art: null };
let modelBeforeCover = null;

/* Quality (§6). `qual` is the server's block from /api/config; `serverQuality`
   records whether the server actually sent one, because that decides who does
   the budget arithmetic. */
let qual = null, serverQuality = false, gpuState = null;
const q = { preset: null, llm: '4bit', rvq: 'bf16', kv: 'quantized', reserve: '1GB' };

const F = {};
const saveSettings = store.saver('generate', 400);

/* ── the quality model, mirrored ───────────────────────────────────────────
   These numbers belong to the server (`ar_cache.budget`), and when
   /api/config carries a `quality` block we use the server's arithmetic and
   never touch this table. It exists for one case: a server that predates the
   quality contract answers /api/budget with the HARDCODED defaults and
   ignores llm/rvq/kv, so the fit line would warn about settings the user did
   not choose. REQUIREMENTS §6: "a wrong warning is worse than none."

   Measured component footprints (GB), and the linear fit that reproduces every
   published 4-minute total to the cent:
     need = LLM + RVQ + seconds * 0.008 * (kv_bits/16) + 0.77 workspace
   4bit/4bit/kv4 7.87 · 4bit/bf16/kv4 8.74 · 4bit/bf16/kv16 10.18 ·
   8bit/bf16/kv16 12.37 · bf16/bf16/kv16 19.88 */
const LLM_GB = { '4bit': 6.29, '8bit': 8.47, bf16: 15.99 };
const RVQ_GB = { '4bit': 0.33, '8bit': 0.64, bf16: 1.20 };
const KV_GB_PER_S = 0.008;      // bf16 key/value cache, per second of song
const WORKSPACE_GB = 0.77;      // activations + CUDA context

const kvBits = (kv) => (kv === 'quantized' ? 4 : 16);

function localNeed(seconds, llm, rvq, kv) {
  return (LLM_GB[llm] ?? LLM_GB['4bit']) + (RVQ_GB[rvq] ?? RVQ_GB.bf16)
       + Number(seconds || 0) * KV_GB_PER_S * (kvBits(kv) / 16) + WORKSPACE_GB;
}

const FALLBACK_PRESETS = [
  { id: 'smallest',  label: 'Smallest',            llm: '4bit', rvq: '4bit', kv: 'quantized', needs_4min_gb: 7.87,  note: 'Everything squeezed. Fits any card, and you can hear the squeeze.' },
  { id: 'balanced',  label: 'Balanced',            llm: '4bit', rvq: '8bit', kv: 'quantized', needs_4min_gb: 8.17,  note: 'Some acoustic detail back for a third of a gigabyte.' },
  { id: 'default10', label: 'Default (10 GB)',     llm: '4bit', rvq: 'bf16', kv: 'quantized', needs_4min_gb: 8.74,  note: 'Full-detail decoder, song memory stored coarser. What this box has been shipping.' },
  { id: 'fullkv',    label: 'Full cache (12 GB+)', llm: '4bit', rvq: 'bf16', kv: 'static',    needs_4min_gb: 10.18, note: 'Keeps the song memory at full precision.' },
  { id: 'high',      label: 'High (16 GB+)',       llm: '8bit', rvq: 'bf16', kv: 'static',    needs_4min_gb: 12.37, note: 'A better-behaved music model — the first real quality step up.' },
  { id: 'maximum',   label: 'Maximum (24 GB+)',    llm: 'bf16', rvq: 'bf16', kv: 'dynamic',   needs_4min_gb: 19.88, note: 'Nothing traded away. Needs a 24 GB card.' },
];

const FALLBACK_QUALITY = {
  detected_total_gb: null, detected_free_gb: null,
  recommended: 'default10', applies_to: ['minimax'],
  presets: FALLBACK_PRESETS,
  fields: { llm: ['4bit', '8bit', 'bf16'], rvq: ['4bit', '8bit', 'bf16'],
            kv: ['auto', 'quantized', 'static', 'dynamic'], reserve: ['0.5GB', '1GB', '2GB'] },
  costs: { llm: 'musicality and prompt adherence', rvq: 'acoustic detail',
           kv: 'storage only - full history, stored coarser' },
};

/* Same rule the server uses, for the fallback path only. */
const recommendFor = (totalGb) =>
  !totalGb ? 'default10' : totalGb < 11 ? 'default10' : totalGb < 14 ? 'fullkv' : totalGb < 20 ? 'high' : 'maximum';

/* Plain language, never the wire value: "4-bit" not "4bit", and a word for
   what the setting is rather than what the flag is called. */
const PREC_LABEL = { '4bit': '4-bit (smallest)', '8bit': '8-bit', bf16: 'bf16 (full)' };
const KV_LABEL = {
  auto: 'Auto — let the server decide',
  quantized: 'Compressed (4-bit) — smallest',
  static: 'Full precision — biggest',
  dynamic: 'Grows as the song does',
};
const SHORT_WORDS = { '4bit': '4-bit', '8bit': '8-bit', bf16: 'full', auto: 'auto', quantized: 'compressed', static: 'full', dynamic: 'growing' };
/* A value this build has never heard of prints as itself, not "undefined" —
   the server is free to grow a mode before the UI knows the word for it. */
const short = (v) => SHORT_WORDS[v] || String(v ?? '');
const COST_LABEL = { llm: 'Music model', rvq: 'Audio decoder', kv: 'Song memory' };

export function init(config, h = {}) {
  cfg = config; hooks = h;

  Object.assign(F, {
    model: $('#c-model'), prompt: $('#c-prompt'), lyrics: $('#c-lyrics'),
    duration: $('#c-duration'), steps: $('#c-steps'), seed: $('#c-seed'),
    instrumental: $('#c-instrumental'), post: $('#c-post'),
    srcPath: $('#c-src-path'), cstr: $('#c-cstr'), nstr: $('#c-nstr'),
    coverOn: $('#c-cover-on'),
    qpreset: $('#c-qpreset'), qllm: $('#c-qllm'), qrvq: $('#c-qrvq'),
    qkv: $('#c-qkv'), qreserve: $('#c-qreserve'),
    lora: $('#c-lora'), loraScale: $('#c-lora-scale'),
    variant: $('#c-variant'),
    takes: $('#c-takes'),
    thinking: $('#c-thinking'), lmTemp: $('#c-lmtemp'),
    autoDur: $('#c-autodur'),
    cot: $('#c-cot'), abc: $('#c-abc'),
  });

  // The ceiling is the SERVER's, read from the card. Hardcoding 2 here would
  // mean a bigger GPU silently inherits a 10 GB card's limit.
  const tmax = Math.max(1, cfg.takes?.max ?? 2);
  F.takes.max = String(tmax);

  fillVariantSelect(cfg.variants);
  fillLoraSelect(cfg.loras?.items || []);
  fillCotSelect(cfg.score);

  // A backend this box cannot run stays IN the list, disabled, with the
  // reason on the option: hiding it would hide the reason to buy the bigger
  // card. `available` is absent on an older server, which means usable.
  for (const [name, b] of Object.entries(cfg.backends)) {
    const off = b.available === false;
    const o = el('option', {
      value: name,
      text: b.label + (off ? ` — ${b.unavailable_reason || 'not available here'}` : ''),
    });
    if (off) o.disabled = true;
    F.model.append(o);
  }
  for (const [k, label] of Object.entries(cfg.upscalers)) {
    F.post.append(el('option', { value: k, text: label }));
  }

  const lim = cfg.limits || {};
  applyLimits(F.duration, lim.duration, [10, 300, 5]);
  applyLimits(F.steps, lim.steps, [4, 60, 1]);
  applyLimits(F.cstr, lim.cover_strength, [0, 1, 0.05]);
  applyLimits(F.nstr, lim.noise_strength, [0, 1, 0.05]);

  view = new JobView($('#side-create'), {
    onResult: (r) => onResult(r),
    // A finished job is exactly when what is resident on the GPU changes, so
    // the "loaded now" line is re-read then rather than polled.
    onEnd: () => { setBusy(false); hooks.onJobEnd?.(); refreshGpuState(); },
    onVram: (v) => hooks.onVram?.(v),
  });

  // ── restore, per field, then apply visibility WITHOUT the cascade ──
  const d = cfg.defaults || {};
  const saved = store.load('generate', {
    model: cfg.default_model, prompt: null, lyrics: cfg.default_lyrics ?? '',
    duration: d.duration ?? 30, steps: d.steps ?? 30, seed: d.seed ?? 7,
    instrumental: !!d.instrumental, post_kind: d.post_kind ?? 'none',
    quality: null, lora: '', lora_scale: 1,
    // store.load() projects onto THIS key set (store.js:27-32), so a field
    // missing here reads back undefined however complete the saved blob is.
    // These five were read below and never listed, so they never came back.
    takes: 1, thinking: false, auto_duration: false, lm_temperature: 1,
    variant: '', cot: '', abc: '',
  });
  const model = usable(saved.model) ? saved.model
    : usable(cfg.default_model) ? cfg.default_model
      : (Object.keys(cfg.backends).find(usable) || cfg.default_model);
  F.model.value = model;
  if (saved.cot && [...F.cot.options].some((o) => o.value === saved.cot)) F.cot.value = saved.cot;
  F.abc.value = saved.abc || '';
  F.prompt.value = saved.prompt ?? cfg.backends[model].default_prompt;
  F.lyrics.value = saved.lyrics ?? '';
  // The range has min=10, so a blob written by a build that stored the -1 wire
  // signal as form state would clamp silently to a 10-second song.
  F.duration.value = Number(saved.duration) > 0 ? saved.duration : (d.duration ?? 30);
  F.steps.value = saved.steps; F.seed.value = saved.seed;
  // A remembered LoRA that has since been deleted must not silently select
  // something else, so only restore an id that is still on disk.
  if (saved.lora && [...F.lora.options].some((o) => o.value === saved.lora)) F.lora.value = saved.lora;
  F.loraScale.value = saved.lora_scale ?? 1;
  if (saved.variant && variantItems.some((x) => x.name === saved.variant)) {
    F.variant.value = saved.variant;
  }
  F.takes.value = Math.min(saved.takes ?? 1, tmax);
  F.thinking.checked = !!saved.thinking;
  F.autoDur.checked = !!saved.auto_duration;
  F.lmTemp.value = saved.lm_temperature ?? 1;
  F.instrumental.checked = !!saved.instrumental;
  F.post.value = cfg.upscalers[saved.post_kind] ? saved.post_kind : 'none';
  F.cstr.value = d.cover_strength ?? 1; F.nstr.value = d.noise_strength ?? 0.75;
  initQuality(saved.quality);
  applyModelVisibility(model);

  // ── wiring ──
  F.model.addEventListener('change', () => cascade(F.model.value));
  for (const n of ['prompt', 'lyrics', 'seed']) F[n].addEventListener('input', persist);
  // §12 made ACE-Step byte-for-byte deterministic and the seed persists, so
  // Generate twice unchanged renders the identical song. That is the point, so
  // the reroll is an action rather than a random default — and the value that
  // ran stays in the box afterwards, which is what makes it repeatable.
  $('#c-seed-roll').addEventListener('click', () => {
    F.seed.value = String(Math.floor(Math.random() * 2147483647) + 1);
    persist();
  });
  F.instrumental.addEventListener('change', () => { persist(); syncLyricsEnabled(); });
  F.post.addEventListener('change', persist);
  F.duration.addEventListener('input', () => { paintOutputs(); persist(); budgetSoon(); });
  F.steps.addEventListener('input', () => { paintOutputs(); persist(); });
  F.cstr.addEventListener('input', paintOutputs);
  F.nstr.addEventListener('input', paintOutputs);
  F.srcPath.addEventListener('input', debounce(resolveSource, 400));
  $('#c-src-clear').addEventListener('click', () => {
    src = { path: null, uploaded: false, url: null };
    F.srcPath.value = ''; $('#c-src-msg').textContent = '';
    coverInfo = { title: null, art: null };
    paintCover();
  });
  $('#c-upload').addEventListener('change', onUpload);

  F.coverOn.addEventListener('change', () => setCoverMode(F.coverOn.checked));
  $('#c-cover-off').addEventListener('click', () => setCoverMode(false));

  F.qpreset.addEventListener('change', () => { applyPreset(F.qpreset.value); persist(); });
  for (const n of ['qllm', 'qrvq', 'qkv', 'qreserve']) {
    F[n].addEventListener('change', () => {
      q.llm = F.qllm.value; q.rvq = F.qrvq.value; q.kv = F.qkv.value; q.reserve = F.qreserve.value;
      paintQuality(); persist(); budgetSoon();
    });
  }
  // Nothing polls the GPU for this panel — it is only interesting while the
  // panel is open, so refresh it on open.
  $('#c-adv').addEventListener('toggle', () => { if ($('#c-adv').open) refreshGpuState(); });

  F.variant.addEventListener('change', () => { paintVariant(); persist(); });

  F.cot.addEventListener('change', () => { paintScore(); persist(); });
  F.abc.addEventListener('input', () => { paintScore(); persist(); });
  $('#c-plan').addEventListener('click', () => go(false, { planOnly: true }));
  $('#c-abc-clear').addEventListener('click', () => { F.abc.value = ''; paintScore(); persist(); });

  $('#c-lora').addEventListener('change', () => {
    // DEFENSIVE ONLY since the field became capability-gated: the select is
    // hidden on a model with no adapters, so a user cannot reach this. A
    // restored form blob or a Reuse can still set a LoRA while MiniMax is
    // active, and switching beats letting the server 400 it four minutes in.
    const applies = supports('loras');
    if (F.lora.value && !applies) {
      const target = (cfg.loras?.applies_to || ['acestep'])[0];
      if (cfg.backends?.[target]) {
        F.model.value = target;
        // The PROGRAMMATIC path, not cascade(): cascade replaces the prompt
        // with the backend's stock one, which would throw away whatever the
        // user had just written. Header rule at the top of this file.
        applyModelVisibility(target);
        paintInfo(target);
        paintOutputs();
        syncLyricsEnabled();
        budgetSoon();
        toast(`Switched to ${cfg.backends[target].label} — LoRAs only apply there.`);
      }
    }
    paintLora(); persist();
  });
  F.loraScale.addEventListener('input', () => { paintOutputs(); persist(); });
  F.takes.addEventListener('input', () => { paintOutputs(); paintTakes(); persist(); });
  F.thinking.addEventListener('change', () => { paintThinking(); paintAutoDur(); persist(); });
  F.autoDur.addEventListener('change', () => {
    if (F.autoDur.checked && !F.thinking.checked) {
      // Auto duration IS the LM's decision; without it the "-1" is a constant.
      F.thinking.checked = true;
      paintThinking();
      toast('Thinking mode on — the model needs it to choose a length.');
    }
    paintAutoDur(); persist();
  });
  F.lmTemp.addEventListener('input', () => { paintOutputs(); persist(); });

  $('#c-lyr-find').addEventListener('click', () => probe(true));
  $('#c-lyr-apply').addEventListener('click', applyFound);
  $('#c-lyr-tx').addEventListener('click', transcribe);

  // The writer is hidden rather than disabled when the weights are absent:
  // a button that can only ever fail is worse than no button.
  if (cfg.writer?.ready) {
    for (const [id, target] of [['#c-write-style', 'style'], ['#c-write-lyrics', 'lyrics']]) {
      $(id).hidden = false;
      $(id).addEventListener('click', () => openWriter(target));
    }
  }

  $('#c-go').addEventListener('click', () => go(false));
  $('#c-preview').addEventListener('click', () => go(true));

  paintOutputs();
  paintInfo(model);
  refreshWorkspaceNote();
  syncLyricsEnabled();
  paintCover();
  budgetSoon();
  refreshGpuState();
}

/* Can the CURRENT model do this?

   The server composes `capabilities` from the same feature blocks the client
   reads, so this is a lookup rather than a rule kept in two places. The
   fallback keeps an older server (or a mock that predates the map) working:
   read the feature's own `applies_to`, which is what every call site used to
   do inline.

   Feature names are the config's own block names -- `loras`, not `lora` --
   so there is no translation layer to get wrong. */
function supports(feature, model) {
  const name = model || F.model.value;
  const caps = cfg.capabilities?.[name];
  if (caps && feature in caps) return !!caps[feature];
  // Steps and duration were universal before YuE2 (which has neither); a
  // server that predates the blocks must not hide them from everyone.
  if (!cfg[feature] && UNIVERSAL.has(feature)) return true;
  return (cfg[feature]?.applies_to || ['acestep']).includes(name);
}
const UNIVERSAL = new Set(['steps', 'duration']);

/* Can this backend be started on THIS box. Absent on an older server = yes. */
const usable = (name) => !!cfg.backends[name] && cfg.backends[name].available !== false;

/* What a feature needs switched on before it means anything. */
function requiredFor(feature) {
  return cfg.requires?.[feature] || null;
}

/* What the ACTIVE CHECKPOINT adds on top of the backend. base can continue a
   track, pull stems and honour guidance; turbo cannot. Straight off
   acestep/constants.py via /api/config, not a rule restated here. */
function variantSupports(task) {
  const v = variantItems.find((x) => x.name === F.variant.value);
  return !!v?.supports?.[task];
}

function applyLimits(input, spec, fallback) {
  const [lo, hi, step] = Array.isArray(spec) && spec.length === 3 ? spec : fallback;
  input.min = lo; input.max = hi; input.step = step;
}

function persist() {
  saveSettings({
    v: 2, model: F.model.value, prompt: F.prompt.value, lyrics: F.lyrics.value,
    // Form state, always the slider's number. -1 is the POST-time encoding for
    // "you decide" and go() derives it there; storing it here would come back
    // as a saved length and clamp to min=10, i.e. a 10-second song.
    // The trap this replaces: persist() runs on every keystroke, so everything
    // it touches has to be reachable from module scope. A bare `model` was not
    // (the three `const model`s are block-scoped), and ES modules are strict —
    // so ticking "Let the model choose" threw a ReferenceError here and from
    // then on nothing saved and budgetSoon() never ran. `auto_duration` below
    // is the field that actually remembers the tick.
    duration: Number(F.duration.value),
    steps: Number(F.steps.value), seed: Number(F.seed.value),
    instrumental: F.instrumental.checked, post_kind: F.post.value,
    variant: F.variant.value,
    lora: F.lora.value, lora_scale: Number(F.loraScale.value),
    takes: Number(F.takes.value),
    thinking: F.thinking.checked, lm_temperature: Number(F.lmTemp.value),
    auto_duration: F.autoDur.checked,
    cot: F.cot.value, abc: F.abc.value,
    // Remembered so a 24 GB card is not handed a 10 GB compromise on every
    // visit, and so Custom comes back exactly as it was left (§6).
    quality: { ...q },
  });
}

function paintOutputs() {
  $('#c-lora-scale-out').textContent = Number(F.loraScale.value).toFixed(2);
  $('#c-lmtemp-out').textContent = Number(F.lmTemp.value).toFixed(2);
  $('#c-takes-out').textContent = F.takes.value;
  $('#c-duration-out').textContent = F.autoDur?.checked ? 'auto' : `${F.duration.value}s`;
  $('#c-steps-out').textContent = F.steps.value;
  $('#c-cstr-out').textContent = Number(F.cstr.value).toFixed(2);
  $('#c-nstr-out').textContent = Number(F.nstr.value).toFixed(2);
}

/* ── the model cascade (HUMAN changes only) ────────────────────────────── */
function cascade(name) {
  const b = cfg.backends[name];
  if (!b) return;
  F.prompt.value = b.default_prompt || '';
  F.steps.value = b.default_steps ?? F.steps.value;
  if (!b.supports_instrumental) F.instrumental.checked = false;
  applyModelVisibility(name);
  paintInfo(name);
  paintOutputs();
  syncLyricsEnabled();
  persist();
  budgetSoon();
}

/* One pass over the groups: a section whose every control is hidden hides
   itself, or its label floats above nothing.

   Runs after the per-feature paints, never instead of them -- two functions
   deciding visibility would fight, and the losing one would be invisible in
   the source. */
function paintGroups() {
  $('#cg-cover').hidden = !coverOn;
  for (const g of $$('.cgroup')) {
    if (g.id === 'cg-cover') continue;          // its own mode decides it
    const kids = Array.from(g.children)
      .filter((n) => !n.classList.contains('eyebrow'));
    g.hidden = kids.length > 0 && kids.every((n) => n.hidden);
  }
}

/* Name what the chosen model adds and what it gives up, because hiding a
   control removes the evidence it ever existed. Without this, switching to
   MiniMax silently deletes checkpoints, adapters, thinking and takes from the
   form and nothing on screen accounts for them. */
function paintCaps() {
  const el0 = $('#c-caps');
  if (!el0) return;
  const LABELS = {
    score: 'score planning', variants: 'checkpoints', loras: 'LoRA adapters',
    thinking: 'thinking mode', takes: 'multiple takes', quality: 'VRAM trade-offs',
    cover: 'covers', instrumental: 'instrumental',
  };
  const has = [], hasnt = [];
  for (const [k, label] of Object.entries(LABELS)) {
    (supports(k) ? has : hasnt).push(label);
  }
  const bits = [];
  if (has.length) bits.push(`Supports ${has.join(', ')}.`);
  if (hasnt.length) bits.push(`No ${hasnt.join(', ')}.`);
  el0.textContent = bits.join(' ');
}

function applyModelVisibility(name) {
  const b = cfg.backends[name] || {};
  $('#c-instr-wrap').hidden = !b.supports_instrumental;
  // The switch shows whenever SOME backend can cover — turning it on is what
  // pins the model, so hiding it under MiniMax would hide the only way in.
  // The cover CONTROLS follow the mode, not the model.
  $('#c-cover-toggle').hidden = !Object.values(cfg.backends).some((x) => x.supports_cover);
  if (!b.supports_cover && coverOn) setCoverMode(false);   // defensive only
  // §6: the quality trades are MiniMax's. ACE-Step never sees this panel.
  const adv = $('#c-adv');
  adv.hidden = !qualityApplies(name);
  if (adv.hidden) adv.open = false;
  paintCover();
  // Every per-feature paint has to have run before the groups are measured,
  // so this is last.
  paintVariant();
  paintLora();
  paintThinking();
  paintTakes();
  paintAutoDur();
  paintScore();
  paintSteps();
  paintCaps();
  paintGroups();
}

/* ── score (YuE2) ──────────────────────────────────────────────────────────
   YuE2 writes an ABC score before it sings, and that score is text: it can be
   read, edited and handed back, or pasted from elsewhere to cover it. Steps
   and duration are the other side of the same fact -- length follows the
   lyrics and the plan, so those two controls leave with the engines that
   have them. */
function fillCotSelect(s) {
  const modes = s?.modes || [];
  F.cot.replaceChildren(...modes.map((m) => el('option', { value: m.id, text: m.label })));
  if (s?.default && modes.some((m) => m.id === s.default)) F.cot.value = s.default;
}

function paintScore() {
  const applies = supports('score');
  $('#c-cot-field').hidden = !applies;
  $('#c-abc-field').hidden = !applies;
  if (!applies) return;
  const s = cfg.score || {};
  const mode = (s.modes || []).find((m) => m.id === F.cot.value);
  $('#c-cot-note').textContent = mode?.note || '';
  const abc = F.abc.value.trim();
  const off = F.cot.value === 'off';
  const note = $('#c-abc-note');
  if (abc && off) {
    note.textContent = 'A pasted score is ignored with No plan — pick Full plan or Melody only to sing to it.';
    note.classList.add('fhint--warn');
  } else {
    note.textContent = s.note || '';
    note.classList.remove('fhint--warn');
  }
  const max = s.max_abc_chars || 65536;
  const msg = $('#c-abc-msg');
  msg.textContent = abc
    ? `${abc.length.toLocaleString()} characters${abc.length > max ? ` — over the ${max.toLocaleString()} limit` : ''}`
    : '';
  msg.classList.toggle('fhint--warn', abc.length > max);
  // Planning needs a plan mode; the button says so rather than silently
  // failing on the server.
  $('#c-plan').disabled = off;
  $('#c-plan').title = off ? 'Pick Full plan or Melody only first' : '';
}

function paintSteps() {
  $('#c-steps-field').hidden = !supports('steps');
  $('#c-duration-field').hidden = !supports('duration');
}

const qualityApplies = (name) => !!qual && (qual.applies_to || ['minimax']).includes(name);

function paintInfo(name) {
  const b = cfg.backends[name] || {};
  $('#c-model-note').textContent = b.note || '';
  $('#c-prompt-info').textContent = b.prompt_info || '';
  $('#c-lyrics-info').textContent = b.lyrics_info || '';
  $('#c-post-note').textContent = cfg.notes?.post_kind || '';
  $('#c-cover-note').textContent = cfg.notes?.cover || '';
  F.prompt.rows = Math.min(12, Math.max(4, b.prompt_lines || 6));
  paintVariant();
  paintLora();
  paintTakes();
  paintThinking();
  paintAutoDur();
  paintScore();
  paintSteps();
}

/* ── LoRA (§11) ────────────────────────────────────────────────────────────
   A LoRA is an adapter on ACE-Step's DiT, so the whole control hides on any
   other model rather than sitting there greyed out — same reasoning as the
   Advanced quality panel, which is MiniMax-only. The list comes from the
   server so dropping a folder in `B:\AudioDev\loras` is all it takes. */
let loraItems = [];

function fillLoraSelect(items) {
  loraItems = items || [];
  const sel = $('#c-lora');
  const keep = sel.value;
  sel.replaceChildren(el('option', { value: '', text: 'None — the base model' }));
  for (const e of loraItems) {
    sel.append(el('option', {
      value: e.id,
      text: e.label + (e.compatible === false ? '  ⚠ different base model' : ''),
    }));
  }
  if ([...sel.options].some((o) => o.value === keep)) sel.value = keep;
}

/* Automatic duration (§14). ACE-Step's docstring says duration < 0 lets the
   model choose, and it does — measured 13 s / 75 s / 192 s for a sketch, a
   song and an epic. But the length comes from the LM's CoT metadata, so with
   thinking OFF it is not a choice at all: it is a flat 120 s every time.
   Rather than ship a control that silently means two different things, ticking
   this turns thinking on. */
function paintAutoDur() {
  // Auto-duration needs the LM that decides the length, so it is gated on
  // BOTH its own capability and the one it requires.
  const need = requiredFor('auto_duration');
  const applies = supports('auto_duration') && (!need || supports(need));
  $('#c-autodur-wrap').hidden = !applies;
  const on = applies && F.autoDur.checked;
  F.duration.disabled = on;
  F.duration.style.opacity = on ? '.4' : '';
  $('#c-duration-out').textContent = on ? 'auto' : `${F.duration.value}s`;
  return on;
}

/* Which workspace new songs land in. Auto-tagging that you cannot see is how
   a month of work ends up in the wrong bucket, so it is stated next to the
   button that does it. */
async function refreshWorkspaceNote() {
  const n = $('#c-ws-note');
  if (!n) return;
  try {
    const d = await api.getWorkspaces();
    n.hidden = false;
    n.textContent = d.active
      ? `New songs are tagged “${d.active}” — change it in the Library.`
      : 'New songs are untagged. Pick a workspace in the Library to group them.';
  } catch { n.hidden = true; }
}

function paintThinking() {
  const th = cfg.thinking || {};
  const applies = supports('thinking');
  $('#c-think-field').hidden = !applies;
  if (!applies) return;
  const on = F.thinking.checked;
  $('#c-lmtemp-wrap').hidden = !on;
  $('#c-think-note').textContent = on
    ? `${th.model || 'the 5Hz LM'} plans the song first. First use loads it — about 20 s.`
    : (th.note || '');
}

function paintTakes() {
  const applies = supports('takes');
  const tmax = Math.max(1, cfg.takes?.max ?? 2);
  $('#c-takes-field').hidden = !applies || tmax < 2;
  if (!applies) return;
  const n = Number(F.takes.value);
  $('#c-takes-note').textContent = n > 1
    ? `${n} different songs from one prompt — each take after the first gets `
      + `its own seed. Roughly ${n}x the time and VRAM.`
    : `One song. Up to ${tmax} on this card — each extra take is a different `
      + `song from the same prompt, not a copy.`;
}

/* Which DiT checkpoint runs. Discovered from disk by the server, so dropping
   one into acestep\checkpoints is all it takes to offer it. Switching is a
   LOAD-time change -- the worker restarts, ~40 s -- which is why this sits
   next to the model rather than among the per-request knobs. */
let variantItems = [];

function fillVariantSelect(v) {
  variantItems = (v?.items || []).filter((x) => x.fits !== false);
  const sel = F.variant;
  sel.replaceChildren(...variantItems.map((x) => el('option', {
    value: x.name, text: x.label || x.name,
  })));
  const want = (v?.active) || '';
  if (variantItems.some((x) => x.name === want)) sel.value = want;
}

function paintVariant() {
  const applies = supports('variants');
  // Hidden when there is nothing to choose between: one checkpoint is not a
  // decision, it is a fact.
  $('#c-variant-field').hidden = !applies || variantItems.length < 2;
  if (!applies) return;
  const cur = variantItems.find((x) => x.name === F.variant.value);
  $('#c-variant-note').textContent = cur
    ? `${cur.note || ''} ~${cur.needs_gb} GB, ${cur.steps} steps.`
      + (cur.active ? '' : ' Switching reloads the model (~40 s).')
    : '';
}

function paintLora() {
  const applies = supports('loras');
  const note = $('#c-lora-note');
  // Hidden where it does not apply. This was deliberately always-visible
  // once, because hiding it made the feature invisible to anyone sitting on
  // the default model — but that was before adapters lived in a "Model &
  // adapter" group directly under the model select, and before the capability
  // line below it named what each model adds. Discoverability now comes from
  // the grouping rather than from leaving a dead control on screen.
  $('#c-lora-field').hidden = !applies;
  // NOT disabled when it does not apply. A disabled <select> cannot be opened
  // at all on iOS Safari, so "LoRAs exist but you are on MiniMax" and "there
  // are no LoRAs" look identical from a phone: you tap it and nothing happens.
  // Picking one switches the model instead — the same move cover mode makes.
  F.lora.disabled = false;
  const chosen = loraItems.find((e) => e.id === $('#c-lora').value);
  $('#c-lora-scale-wrap').hidden = !chosen;
  if (!applies) return;   // the whole field is hidden; nothing to say
  if (!loraItems.length) {
    note.textContent = `No adapters found. Drop one into ${cfg.loras?.drop_path || 'the loras folder'} — `
      + 'a folder with adapter_config.json and adapter_model.safetensors.';
  } else if (chosen) {
    note.textContent = `${chosen.peft_type || 'LORA'} on ${(chosen.target_modules || []).join(', ')}`
      + (chosen.base_model_name ? ` · trained against ${chosen.base_model_name}` : '')
      + (chosen.compatible === false
        ? ' — this was trained against a different base model and will most likely produce noise.'
        : '');
  } else {
    note.textContent = cfg.loras?.note || '';
  }
}

function syncLyricsEnabled() {
  const off = F.instrumental.checked;
  F.lyrics.disabled = off;
  F.lyrics.style.opacity = off ? '.5' : '';
}

/* ── the VRAM fit note ─────────────────────────────────────────────────────
   One request, two lines: the one under Duration (which has always been
   there) and the one inside Advanced. Both must reflect the CHOSEN quality,
   not the hardcoded defaults — §6 is explicit that a warning computed from
   settings the user did not pick is worse than no warning at all. */
let budgetAc = null;
const budgetSoon = debounce(async () => {
  const out = $('#c-fit'), qout = $('#c-qfit');
  const model = F.model.value, seconds = Number(F.duration.value);
  budgetAc?.abort();
  const ac = new AbortController(); budgetAc = ac;
  try {
    const applies = qualityApplies(model);
    let r = await api.getBudget(model, seconds, applies ? q : null, { signal: ac.signal });
    // An older server ignores llm/rvq/kv and answers for its own hardcoded
    // defaults. Keep its free_gb (only it can measure that) and redo the need.
    if (applies && !serverQuality && r.applies) {
      const need = localNeed(seconds, q.llm, q.rvq, q.kv);
      r = { ...r, need_gb: +need.toFixed(2), headroom_gb: +(r.free_gb - need).toFixed(2), fits: need <= r.free_gb };
    }
    if (!r.applies) {
      out.textContent = ''; out.className = 'fhint';
      qout.textContent = ''; qout.className = 'fhint';
      return;
    }
    if (r.fits) {
      out.textContent = `needs ~${r.need_gb} GB, ${r.free_gb} GB free — fits with ${Math.abs(r.headroom_gb)} GB spare`;
      out.className = 'fhint fhint--ok';
      qout.textContent = `${songWords(seconds)} needs ~${r.need_gb} GB, you have ${r.free_gb} GB free — it fits, with ${Math.abs(r.headroom_gb)} GB spare.`;
      qout.className = 'fhint fhint--ok';
    } else {
      out.textContent = `⚠ needs ~${r.need_gb} GB but only ${r.free_gb} GB is free (${Math.abs(r.headroom_gb)} GB short) — this will spill to system memory and run several times slower. Close browsers/ComfyUI, or shorten it.`;
      out.className = 'fhint fhint--warn';
      qout.textContent = `⚠ Does not fit: ${songWords(seconds)} needs ~${r.need_gb} GB, you have ${r.free_gb} GB free — ${Math.abs(r.headroom_gb)} GB short. It will spill into system memory and crawl. Pick a smaller preset, or a shorter song.`;
      qout.className = 'fhint fhint--warn';
    }
  } catch (e) {
    if (e.name !== 'AbortError') { out.textContent = ''; qout.textContent = ''; }
  }
}, 250);

/* "a 4-minute song" reads like a person wrote it; "240s" does not. */
function songWords(seconds) {
  const s = Number(seconds) || 0;
  if (s < 60) return `a ${Math.round(s)}-second song`;
  const m = Math.round((s / 60) * 10) / 10;
  return `a ${m}-minute song`;
}

/* ── quality (§6) ──────────────────────────────────────────────────────── */
function initQuality(savedQ) {
  serverQuality = !!(cfg.quality && Array.isArray(cfg.quality.presets) && cfg.quality.presets.length);
  qual = serverQuality ? cfg.quality : { ...FALLBACK_QUALITY };

  fillSelect(F.qpreset, qual.presets.map((p) => [p.id, p.label]));
  F.qpreset.append(el('option', { value: 'custom', text: 'Custom…' }));
  const f = qual.fields || FALLBACK_QUALITY.fields;
  fillSelect(F.qllm, f.llm.map((v) => [v, PREC_LABEL[v] || v]));
  fillSelect(F.qrvq, f.rvq.map((v) => [v, PREC_LABEL[v] || v]));
  fillSelect(F.qkv, f.kv.map((v) => [v, KV_LABEL[v] || v]));
  fillSelect(F.qreserve, (f.reserve || ['1GB']).map((v) => [v, v.replace('GB', ' GB')]));

  $('#c-adv-intro').textContent =
    'MiniMax is squeezed to fit a small card. These are the trades that squeeze it — every one costs something, so the panel says what.';

  q.reserve = qual.defaults?.reserve || q.reserve;
  if (savedQ && savedQ.preset) {                       // remembered choice wins
    Object.assign(q, savedQ);
    applyPreset(q.preset);          // 'custom' keeps the restored llm/rvq/kv
  } else {
    applyPreset(qual.recommended || recommendFor(qual.detected_total_gb));
    // The fallback table has no detected sizes; /api/vram does, and a 24 GB
    // card must not silently inherit the 10 GB compromise (§6).
    if (!serverQuality) refineFromVram();
  }
}

function fillSelect(sel, pairs) {
  sel.replaceChildren(...pairs.map(([value, text]) => el('option', { value, text })));
}

async function refineFromVram() {
  try {
    const v = await api.getVram();
    if (!v || v.available === false) return;
    qual.detected_total_gb = v.total_gb;
    qual.detected_free_gb = v.free_gb;
    qual.recommended = recommendFor(v.total_gb);
    if (!store.load('generate', { quality: null }).quality?.preset) applyPreset(qual.recommended);
    else paintQuality();
  } catch { /* the pill will report the same failure */ }
}

function applyPreset(id) {
  const p = qual.presets.find((x) => x.id === id);
  q.preset = p ? p.id : 'custom';
  // Custom starts from wherever you already were — the three controls it
  // reveals are pre-filled with the preset you just left, not with defaults.
  if (p) { q.llm = p.llm; q.rvq = p.rvq; q.kv = p.kv; }
  paintQuality();
  budgetSoon();
}

function paintQuality() {
  F.qpreset.value = q.preset;
  F.qllm.value = q.llm; F.qrvq.value = q.rvq; F.qkv.value = q.kv; F.qreserve.value = q.reserve;
  $('#c-qcustom').hidden = q.preset !== 'custom';

  const p = qual.presets.find((x) => x.id === q.preset);
  $('#c-qpreset-note').textContent = p
    ? `${p.note || ''} About ${p.needs_4min_gb} GB for a 4-minute song.`.trim()
    : `Your own mix: ${short(q.llm)} music model, ${short(q.rvq)} decoder, ${short(q.kv)} song memory.`;

  const costs = qual.costs || FALLBACK_QUALITY.costs;
  $('#c-qcosts').replaceChildren(...['llm', 'rvq', 'kv'].map((k) => el('li', {},
    el('b', { text: COST_LABEL[k] }), ` — ${plainCost(costs[k])}`)));

  // §6: on a 10 GB card the honest ceiling is Default, and saying so beats
  // letting someone discover it by watching a generation crawl.
  const total = qual.detected_total_gb;
  $('#c-qceiling').textContent = (total && total < 12)
    ? `This card has ${total} GB, so Default is the honest ceiling — the big lever, music-model precision, needs a 24 GB card.`
    : '';

  paintLoaded();
}

/* The server writes these as fragments ("storage only - full history, stored
   coarser"); the panel reads them as "<label> — <fragment>.", so the hyphen
   inside becomes a colon rather than a second dash in the same sentence. */
function plainCost(s) {
  const t = String(s || '').replace(/\s+-\s+/, ': ').trim();
  return t ? `${t}.` : '';
}

async function refreshGpuState() {
  try { gpuState = await api.getGpuState(); }
  catch { gpuState = null; }          // a server without the endpoint: say less
  paintLoaded();
}

function paintLoaded() {
  const out = $('#c-qloaded');
  if (!out) return;
  // The server owns this sentence when it sends one — it knows what its own
  // worker does on a precision change.
  const restart = gpuState?.restart_note || qual?.restart_note
    || 'Changing the music model or decoder precision reloads MiniMax (~20 s) before the song starts. Song memory and the reserve apply to the next song with no reload.';
  const name = gpuState?.loaded_label || gpuState?.loaded;
  const lq = gpuState?.quality;
  if (name && lq && lq.llm && lq.rvq) {
    const same = lq.llm === q.llm && lq.rvq === q.rvq;
    out.textContent = same
      ? `Loaded now: ${name} at ${short(lq.llm)} music model, ${short(lq.rvq)} decoder — your choice matches, so nothing reloads.`
      : `Loaded now: ${name} at ${short(lq.llm)} music model, ${short(lq.rvq)} decoder. Your choice (${short(q.llm)} / ${short(q.rvq)}) differs. ${restart}`;
    out.className = same ? 'fhint' : 'fhint fhint--warn';
  } else if (name) {
    out.textContent = `${name} is loaded. ${restart}`;
    out.className = 'fhint';
  } else {
    out.textContent = `Nothing is loaded — MiniMax loads when you generate. ${restart}`;
    out.className = 'fhint';
  }
}

/* ── cover mode (§7) ───────────────────────────────────────────────────── */
function setCoverMode(on, opts = {}) {
  on = !!on;
  if (on === coverOn) { paintCover(); return; }
  coverOn = on;
  if (on) {
    // Pinned, not silently moved: the select is locked and the reason is
    // printed under it and in the banner.
    modelBeforeCover = F.model.value;
    if (cfg.backends.acestep && F.model.value !== 'acestep') {
      F.model.value = 'acestep';
      paintInfo('acestep');
    }
    F.model.disabled = true;
  } else {
    F.model.disabled = false;
    // Leaving clears the source and puts the form back the way it was.
    if (!opts.keepSource) {
      src = { path: null, uploaded: false, url: null };
      F.srcPath.value = '';
      $('#c-src-msg').textContent = '';
      foundLyrics = null;
    }
    coverInfo = { title: null, art: null };
    if (modelBeforeCover && cfg.backends[modelBeforeCover] && F.model.value !== modelBeforeCover) {
      F.model.value = modelBeforeCover;
      paintInfo(modelBeforeCover);        // the prompt stays: no cascade (§11)
    }
    modelBeforeCover = null;
  }
  applyModelVisibility(F.model.value);
  syncLyricsEnabled();
  persist();
  budgetSoon();
}

function paintCover() {
  F.coverOn.checked = coverOn;
  $('#c-cover-banner').hidden = !coverOn;
  $('#c-cover-fields').hidden = !coverOn;

  const pin = $('#c-model-pin');
  pin.hidden = !coverOn;
  pin.textContent = coverOn
    ? 'Pinned to ACE-Step: it is the only model here that can cover an existing track.'
    : '';

  const name = coverInfo.title || (src.path ? src.path.split(/[\\/]/).pop() : null);
  $('#c-cover-title').textContent = name || 'no source picked yet';
  $('#c-cover-why').textContent = name
    ? 'ACE-Step re-sings this track with the style and lyrics below.'
    : 'Point at a file below, and it becomes the track being covered.';

  const img = $('#c-cover-img');
  if (coverInfo.art) { img.src = coverInfo.art; img.hidden = false; }
  else { img.removeAttribute('src'); img.hidden = true; }
  $('#c-cover-ph').hidden = !!coverInfo.art;

  $('#cg-cover').hidden = !coverOn;
  paintPrimary();
}

function paintPrimary() {
  $('#c-go').textContent = busy
    ? (coverOn ? 'Covering…' : 'Generating…')
    : (coverOn ? 'Create cover' : 'Generate');
}

/* ── cover source (§7.4) ───────────────────────────────────────────────── */
async function resolveSource() {
  const typed = F.srcPath.value.trim().replace(/^"|"$/g, '');
  const msg = $('#c-src-msg');
  if (typed) {
    try {
      const st = await api.fsStat(typed);
      if (st.is_file) {
        src = { path: typed, uploaded: false, url: st.playable ? api.mediaUrl(typed) : null };
        msg.textContent = st.playable ? `Using ${st.name}.` : `Using ${st.name} — outside the served folders, so it is processable but not playable here.`;
        // A hand-typed path has no library entry, so the banner falls back to
        // the file name — but it must still name the track being covered.
        coverInfo = { title: null, art: null };
        paintCover();
        probe(false);
        return;
      }
      msg.textContent = 'No file at that path.';
    } catch { msg.textContent = 'Could not check that path.'; }
  }
  if (src.uploaded && src.path) msg.textContent = `Using the uploaded copy.`;
}

async function onUpload(ev) {
  const file = ev.target.files?.[0];
  if (!file) return;
  const msg = $('#c-src-msg');
  msg.textContent = 'Uploading…';
  try {
    const r = await api.upload(file, (f) => { msg.textContent = `Uploading… ${Math.round(f * 100)}%`; });
    // A real on-disk path beats the browser's copy — an upload lands in a
    // folder with no sidecar neighbours.
    src = { path: r.path, uploaded: true, url: r.url };
    msg.textContent = `Uploaded ${r.name}.`;
    coverInfo = { title: r.name || null, art: null };
    paintCover();
    probe(false);
  } catch (e) { setHint(msg, e.message || 'Upload failed.'); }
  ev.target.value = '';
}

/* ── lyrics ────────────────────────────────────────────────────────────── */
async function probe(explicit) {
  const out = $('#c-lyr-msg');
  if (!src.path) {
    out.textContent = 'Pick a source track to search for its lyrics.';
    foundLyrics = null;
    return;
  }
  out.textContent = 'Searching…';
  try {
    const r = await api.probeLyrics({ path: src.path, uploaded: src.uploaded });
    foundLyrics = r.found ? r.text : null;
    setHint(out, stripMd(r.message || ''));
    if (explicit && !r.found) toast('Nothing found — try Transcribe.');
  } catch (e) { setHint(out, e.message); }
}

function applyFound() {
  if (!foundLyrics) { toast('Nothing found to apply — try Transcribe.'); return; }
  const cur = F.lyrics.value.trim();
  if (cur && cur !== foundLyrics.trim()) toast('Replaced the lyrics box with the lyrics that were found.');
  F.lyrics.value = foundLyrics;
  persist();
}

async function transcribe() {
  if (!src.path) { toast('Pick a source track first.', 'err'); return; }
  if (busy) { toast('The GPU is busy.', 'err'); return; }
  try {
    setBusy(true);
    const r = await api.postTranscribe({
      path: src.path, separate: cfg.defaults?.separate ?? true,
      model: cfg.defaults?.whisper_model || (cfg.whisper_models || ['large-v3'])[0],
      seconds: cfg.defaults?.seconds ?? 0,
    });
    hooks.onJob?.(r.job_id);
    view.attach(r.job_id, {
      kind: 'transcribe',
      onResult: (res) => {
        F.lyrics.value = res.text || '';
        persist();
        setHint($('#c-lyr-msg'), `${res.summary || ''} ${res.note || ''}`.trim());
      },
    });
  } catch (e) { setBusy(false); toast(e.message, 'err'); }
}

/* ── the writer (§10) ──────────────────────────────────────────────────────
   One sheet per field, three modes. It runs on the CPU lane, so it neither
   waits for the GPU nor blocks it — which is why it does NOT touch `busy` and
   uses its own JobView instead of the shared one: a write must be able to
   happen while a song is rendering.

   The view is deliberately NOT detached when the sheet closes. The handlers
   keep firing into a detached node, `onResult` still writes into the textarea,
   and a minute-long job survives an impatient swipe. */
let writeView = null, writeBusy = false;

const PLACEHOLDER = {
  'lyrics:generate': 'a defiant synth-pop song about quitting a job you hated',
  'lyrics:extend':   'optional — anything the new sections should cover',
  'lyrics:edit':     'make the chorus angrier and cut the second verse',
  'style:generate':  'warm acoustic ballad, female vocal, brushed drums',
  'style:extend':    'optional — what to add more detail about',
  'style:edit':      'make it 140 bpm and swap the piano for a rhodes',
};

function openWriter(target) {
  const field = target === 'lyrics' ? F.lyrics : F.prompt;
  const modes = cfg.writer?.modes || ['generate', 'extend', 'edit'];
  const info = cfg.writer?.mode_info || {};

  const mode = el('select', {},
    ...modes.map((m) => el('option', { value: m, text: m[0].toUpperCase() + m.slice(1) })));
  const brief = el('textarea', { rows: '3', spellcheck: 'false' });
  // Sampling is random by default — a lyricist wants a reroll, not the same
  // answer twice. The seed comes back in the result so a good one can be
  // typed in here to get it again, which is the only reason this box exists.
  const seed = el('input', { type: 'number', inputmode: 'numeric',
                             placeholder: 'random', 'aria-label': 'Seed' });
  const hint = el('p', { class: 'fhint' });
  const msg = el('p', { class: 'fhint' });
  const progress = el('div', {});
  const goBtn = el('button', { type: 'button', class: 'btn btn--primary', text: 'Write' });

  const sync = () => {
    const m = mode.value;
    hint.textContent = info[m] || '';
    brief.placeholder = PLACEHOLDER[`${target}:${m}`] || '';
    // extend and edit need something to work from; say so before the 400 does.
    const empty = !field.value.trim();
    goBtn.disabled = writeBusy || (empty && m !== 'generate');
    msg.textContent = (empty && m !== 'generate')
      ? `Nothing in the ${target === 'lyrics' ? 'lyrics' : 'style'} box yet — ${m} works from what is already there.`
      : '';
  };
  mode.addEventListener('change', sync);

  goBtn.addEventListener('click', async () => {
    const m = mode.value;
    if (m !== 'generate' && !field.value.trim()) return;
    if (m === 'generate' && !brief.value.trim()) { toast('Say what the song is about.', 'err'); brief.focus(); return; }
    if (m === 'edit' && !brief.value.trim()) { toast('Say what to change.', 'err'); brief.focus(); return; }
    writeBusy = true; goBtn.disabled = true; goBtn.textContent = 'Writing…';
    try {
      const r = await api.postWrite({
        target, mode: m, model: F.model.value,
        brief: brief.value.trim(), existing: field.value,
        seed: seed.value.trim() === '' ? null : Number(seed.value),
      });
      const done = () => {
        writeBusy = false; goBtn.disabled = false; goBtn.textContent = 'Write';
      };
      // Handlers set BEFORE attach: a job that fails instantly would otherwise
      // run the previous sheet's onEnd, which closes over a dead button.
      writeView = writeView || new JobView(progress, {});
      writeView.host = progress;
      writeView.h = { onEnd: done };
      writeView.render();
      writeView.attach(r.job_id, {
        kind: 'write',
        onResult: (res) => {
          done();
          if (res.text) { field.value = res.text; persist(); }
          // Put the seed back in the box so the roll that just happened is
          // repeatable without the user having to copy it out of a toast.
          if (res.seed != null) seed.value = res.seed;
          toast(res.warning
            ? 'Written — but check it: ' + stripMd(res.warning)
            : res.scoped_to
              ? `Rewrote [${res.scoped_to}] — everything else is untouched.`
              : `Written on the ${res.device === 'cuda' ? 'GPU' : 'CPU'}. `
                + `Seed ${res.seed} is in the box — clear it to reroll.`,
            res.warning ? 'err' : '');
          closeSheet();
        },
      });
    } catch (e) {
      writeBusy = false; goBtn.disabled = false; goBtn.textContent = 'Write';
      toast(e.message, 'err');
    }
  });

  openSheet(target === 'lyrics' ? 'Write lyrics' : 'Write a style prompt',
    el('div', { class: 'formcol', style: 'padding:0;gap:16px' },
      el('div', { class: 'field' },
        el('label', { class: 'flabel', text: 'Mode' }), mode, hint),
      el('div', { class: 'field' },
        el('label', { class: 'flabel', text: 'Brief' }), brief,
        el('p', { class: 'fhint',
          text: target === 'style' && F.model.value === 'minimax'
            ? 'MiniMax was trained on sectioned captions, so the writer fills in the '
              + 'Global Metadata / Vocal Details / Arrangement skeleton and checks every '
              + 'heading and field survived before handing it back.'
            : 'Name a section and only that section is rewritten — “make the chorus '
              + 'angrier” replaces the chorus and splices the rest back untouched.' })),
      el('div', { class: 'field' },
        el('label', { class: 'flabel', text: 'Seed' }), seed,
        el('p', { class: 'fhint',
          text: `Blank rolls a new one. ${cfg.writer?.note || ''}` })),
      msg, progress,
      el('div', { class: 'btnrow' },
        el('button', { type: 'button', class: 'btn', text: 'Close', onclick: () => closeSheet() }),
        goBtn)));
  sync();
  brief.focus();
}

/* ── generate ──────────────────────────────────────────────────────────── */

/* The request that actually ran. The seed is not in the result payload and the
   box can be edited during a four-minute render, so the card cannot read it off
   the form. Null when the page reloaded mid-render and reattached to a running
   job (§8), so every read of it is guarded rather than assumed. */
let lastReq = null;

async function go(preview, opts = {}) {
  if (busy) { toast('A GPU job is already running.', 'err'); return; }
  const model = F.model.value;
  const b = cfg.backends[model] || {};
  if (b.available === false) {
    toast(`${b.label} is not available here — ${b.unavailable_reason || 'pick another engine'}.`, 'err');
    F.model.focus(); return;
  }
  const prompt = F.prompt.value.trim();
  const lyrics = F.lyrics.value.trim();
  if (!prompt) { toast('A style description is required.', 'err'); F.prompt.focus(); return; }
  if (b.requires_lyrics && !lyrics && !F.instrumental.checked) {
    toast(`${b.label || 'This engine'} requires lyrics — it has no instrumental mode. Use ACE-Step for instrumentals.`, 'err');
    F.lyrics.focus(); return;
  }
  // Score fields ride only with an engine that plans one; the server 400s
  // them elsewhere, which is right, and the form should never get there.
  const score = {};
  if (supports('score', model)) {
    score.cot = F.cot.value || 'full';
    const abc = F.abc.value.trim();
    if (abc && score.cot === 'off') {
      toast('A pasted score needs Full plan or Melody only — with No plan it would be ignored.', 'err');
      F.cot.focus(); return;
    }
    if (abc) score.abc = abc;
    if (opts.planOnly) {
      if (score.cot === 'off') { toast('Plan score only needs a plan mode.', 'err'); F.cot.focus(); return; }
      score.plan_only = true;
    }
  } else if (opts.planOnly) {
    return;                                  // the button is hidden here anyway
  }
  // Cover mode with no source would quietly become an ordinary generation —
  // the button says "Create cover", so it must either cover or refuse.
  if (coverOn && !src.path) {
    toast('Pick a source track to cover, or leave cover mode.', 'err');
    F.srcPath.focus(); return;
  }
  const body = {
    model, prompt, lyrics,
    // -1 is the wire signal for "you decide" (api.py treats it as auto, not
    // as a length). Only sent when the model supports it.
    duration: (F.autoDur.checked && supports('auto_duration', model)
               && supports(requiredFor('auto_duration') || 'thinking', model))
      ? -1 : Number(F.duration.value),
    steps: Number(F.steps.value), seed: Number(F.seed.value),
    instrumental: F.instrumental.checked && !!b.supports_instrumental,
    post_kind: F.post.value, preview: !!preview,
    src_path: (coverOn && b.supports_cover && src.path) ? src.path : null,
    cover_strength: Number(F.cstr.value), noise_strength: Number(F.nstr.value),
  };
  // Resolved values, not just the preset id: the server should never have to
  // guess what "default10" meant in the build that sent it.
  if (qualityApplies(model)) {
    body.quality = { preset: q.preset, llm: q.llm, rvq: q.rvq, kv: q.kv, reserve: q.reserve };
  }
  // Only when the control is actually applicable and set — the server 400s a
  // LoRA sent alongside MiniMax, which is right, and the UI should never put
  // it in that position.
  if (supports('takes', model)
      && Number(F.takes.value) > 1) {
    body.takes = Number(F.takes.value);
  }
  if (supports('thinking', model) && F.thinking.checked) {
    body.thinking = true;
    body.lm_temperature = Number(F.lmTemp.value);
  }
  if (supports('variants', model) && F.variant.value) {
    body.variant = F.variant.value;
  }
  const loraApplies = supports('loras', model);
  if (loraApplies && F.lora.value) {
    body.lora = F.lora.value;
    body.lora_scale = Number(F.loraScale.value);
  }
  Object.assign(body, score);
  try {
    setBusy(true);
    lastReq = body;
    const r = await api.postGenerate(body);
    hooks.onJob?.(r.job_id);
    view.attach(r.job_id, { kind: r.kind || 'generate' });
    hooks.scrollToSide?.();
  } catch (e) {
    setBusy(false);
    toast(e.message, 'err');
  }
}

function setBusy(on) {
  busy = on;
  $('#c-go').disabled = on;
  $('#c-preview').disabled = on;
  paintPrimary();          // never hardcode "Generate": cover mode renames it
}

/* The text up to the first comma or newline, clipped — a style prompt's opening
   phrase reads as a name, the whole prompt does not. */
function firstSeg(s) {
  const t = String(s ?? '').split(/[,\n]/)[0].replace(/\s+/g, ' ').trim();
  if (!t) return '';
  return t.length > 48 ? `${t.slice(0, 47).trimEnd()}…` : t;
}

/* The score as a result: shown, and one tap puts it in the editor so the next
   Generate sings to it. Editing happens in the textarea, not here. */
function scoreCard(r, heading) {
  const abc = r.score || '';
  const lines = abc.split('\n').length;
  const trunc = r.truncated && Object.values(r.truncated).some(Boolean);
  return el('div', { class: 'card' },
    el('p', { class: 'card-h', text: heading }),
    el('p', { class: 'note', text: `${lines} lines · ${r.cot || 'full'} plan${r.elapsed ? ` · ${Math.round(r.elapsed)}s` : ''}` }),
    trunc ? el('p', { class: 'note fhint--warn', text: 'Truncated: the plan hit the model’s length limit. Shorter lyrics give it room.' }) : null,
    el('pre', { class: 'score-pre', text: abc.length > 4000 ? abc.slice(0, 4000) + '\n…' : abc }),
    el('div', { class: 'btnrow' },
      el('button', { type: 'button', class: 'btn btn--primary', text: 'Edit this score',
        onclick: () => {
          F.abc.value = abc;
          if (F.cot.value === 'off') F.cot.value = 'full';
          paintScore(); persist();
          F.abc.focus();
          toast('Score loaded into the editor — change it, then Generate.');
        } })));
}

async function onResult(r) {
  if (r.plan_only) {
    view.results.replaceChildren(scoreCard(r, 'Score planned'));
    toast('Score ready — edit it, then Generate.');
    return;
  }
  // /api/library/{id} answers {entry, detail_md, shared_note}, so the title is
  // one level down — reading it off the top level silently gives undefined.
  // Worth the round trip: r.library_id is os.path.basename(path), and 124 of
  // the 127 tracks on this box are UUID-named, so the dock read
  // "2c68b822-8f85-….flac". JobView does not await this.
  const info = (await api.getTrack(r.library_id).catch(() => null))?.entry || null;
  const title = info?.title || firstSeg(lastReq?.prompt) || 'new generation';
  const cover = info?.cover_url || null;

  const rt = r.realtime_ratio ? `${r.realtime_ratio.toFixed(1)}x realtime` : '';
  const bits = [r.seconds ? `${Math.round(r.seconds)}s` : null, r.elapsed ? `${Math.round(r.elapsed)}s elapsed` : null, rt]
    .filter(Boolean).join(' · ');

  // Read back from the RESPONSE — what was in the forward pass — not from the
  // form, which can have moved on during the render. Model and seed are the
  // two the payload does not carry, so they come off the request that ran; on
  // a page that reattached to a job there is no request, and those pieces are
  // dropped rather than printed as "undefined".
  const prov = [
    lastReq?.model ? (cfg.backends?.[lastReq.model]?.label || lastReq.model) : null,
    r.variant ? `checkpoint ${r.variant}` : null,
    r.lora_name ? `${r.lora_name}${r.lora_scale != null ? ` at ${r.lora_scale}` : ''}` : null,
    r.thinking ? `thinking, weirdness ${r.lm_temperature ?? 1}` : null,
    r.quality?.preset || null,
    r.score ? `${r.cot || 'full'} plan${lastReq?.abc ? ' from a supplied score' : ''}` : null,
    lastReq ? `seed ${lastReq.seed}` : null,
  ].filter(Boolean).join(' · ');

  // §13: r.takes holds only the EXTRA takes — the primary is r.url — and each
  // one is a different song from the same prompt, not a copy. With one Play
  // button the siblings were reachable only by hunting the library.
  const takes = Array.isArray(r.takes) ? r.takes : [];
  // The library numbers takes by mtime (library.py:group_takes), not by the
  // order they come back in, so labelling the primary "take 1" was wrong
  // whenever it happened to be written second: "Play take 1" played what the
  // library calls take 2. Ask each sibling for its own number instead of
  // inventing one -- a handful of small GETs after a multi-minute render.
  const sibIds = takes.map((t) => (t.path || '').split(/[\\/]/).pop() || null);
  const sibs = await Promise.all(sibIds.map(
    (id) => (id ? api.getTrack(id).catch(() => null) : Promise.resolve(null))));
  // Fall back to position when a sidecar carries no take number -- an entry
  // the library has not caught up with yet.
  const takeNo = (e, i) => e?.entry?.take ?? i;
  const primaryNo = info?.take ?? 1;
  view.results.replaceChildren(el('div', { class: 'card' },
    el('p', { class: 'card-h', text: r.preview ? 'Preview ready' : 'Generated' }),
    el('p', { class: 'note', text: bits }),
    prov ? el('p', { class: 'note', text: prov }) : null,
    r.post ? el('p', { class: 'note', text: `post: ${r.post.kind} — ${r.post.ok ? 'ok' : 'failed, the generation is still yours'}` }) : null,
    el('div', { class: 'btnrow' },
      el('button', { type: 'button', class: 'btn btn--primary',
        text: takes.length ? `Play take ${primaryNo}` : 'Play',
        onclick: () => loadIt(r, title, cover) }),
      ...takes.map((t, i) => el('button', {
        type: 'button', class: 'btn btn--ghost', text: `take ${takeNo(sibs[i], i + 2)}`,
        // A sibling is its own library entry with its own sidecar, so it gets
        // its own id and not the primary's, or the now-playing actions would
        // rate and retitle the wrong song. Its cover is its own too, and we
        // have not fetched it — null is the honest answer.
        onclick: () => player.loadAdHoc({
          url: t.url,
          title: `${sibs[i]?.entry?.title || title} · take ${takeNo(sibs[i], i + 2)}`,
          sub: 'just generated',
          cover: sibs[i]?.entry?.cover_url || null, id: sibIds[i],
        }),
      }))),
    r.score ? scoreCard(r, 'Score it sang to') : null));
  loadIt(r, title, cover);
  hooks.onGenerated?.(r);
  // jobs.js toasts on cancel and on error but never on success, so a render
  // that landed while you were on another pane was completely silent — and on
  // iOS the un-gestured play() in loadIt is blocked, so the audio does not
  // start either. This is the only thing that says it finished.
  toast(`Rendered “${title}” in ${Math.round(r.elapsed || 0)}s`);
}

function loadIt(r, title, cover) {
  // `id` is the contract with the player: the now-playing actions need to know
  // which library entry this is, and a fresh render is not in the list yet.
  player.loadAdHoc({ url: r.final_url || r.url, title,
                     sub: r.post ? `upscaled — ${r.post.kind}` : 'just generated',
                     cover: cover || null, id: r.library_id });
}

/* ── programmatic loads (Reuse / Cover) — no cascade ───────────────────── */
export function applyReuse(r) {
  // Reuse is an ordinary generation: if cover mode was on it must end, or the
  // pinned model would fight the model this record asks for.
  if (coverOn) setCoverMode(false);
  if (r.model && cfg.backends[r.model]) {
    if (usable(r.model)) F.model.value = r.model;
    else toast(`${cfg.backends[r.model].label} is not available here — ${cfg.backends[r.model].unavailable_reason || ''}. Prompt loaded on ${cfg.backends[F.model.value]?.label}.`, 'err');
  }
  if (r.prompt != null) F.prompt.value = r.prompt;
  F.lyrics.value = r.lyrics || '';
  // The score travels with a YuE2 record; any other record clears it.
  if (r.cot && [...F.cot.options].some((o) => o.value === r.cot)) F.cot.value = r.cot;
  F.abc.value = r.score || '';
  if (r.duration != null) F.duration.value = r.duration;
  if (r.steps != null) F.steps.value = r.steps;
  if (r.seed != null) F.seed.value = r.seed;
  F.instrumental.checked = !!r.instrumental;
  applyModelVisibility(F.model.value);
  paintInfo(F.model.value);
  paintOutputs(); syncLyricsEnabled(); persist(); budgetSoon();
}

/** Arriving from the library's ↻ Cover: turn cover mode ON automatically. */
export function applyCover(r) {
  applyReuse(r);                       // clears any previous cover mode first
  setCoverMode(true);
  F.model.value = 'acestep';
  applyModelVisibility('acestep');
  paintInfo('acestep');
  src = { path: r.src_path, uploaded: false, url: r.src_url || null };
  F.srcPath.value = r.src_path || '';
  if (r.cover_strength != null) F.cstr.value = r.cover_strength;
  if (r.noise_strength != null) F.nstr.value = r.noise_strength;
  // The banner needs a name and a face; library.js sends both alongside the
  // endpoint's payload, because /cover-source knows the path, not the art.
  coverInfo = {
    title: r.title || (r.src_path || '').split(/[\\/]/).pop() || null,
    art: r.cover_url || null,
  };
  $('#c-src-msg').textContent = `Source: ${(r.src_path || '').split(/[\\/]/).pop()}`;
  if (r.lyrics_probe) {
    foundLyrics = r.lyrics_probe.found ? r.lyrics_probe.text : null;
    setHint($('#c-lyr-msg'), stripMd(r.lyrics_probe.message || ''));
  }
  paintCover();
  paintOutputs(); persist(); budgetSoon();
}

export const jobView = () => view;
