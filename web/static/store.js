/* store.js — localStorage, per-field. Spec §12.1.
   Four keys, plain JSON, no obfuscation. A blob written by an older build can
   be missing keys, so every read is `saved[k] ?? fallback[k]` — never
   all-or-nothing, which would throw away the fields it does have. */

const KEYS = {
  generate: 'audiodev.studio.generate.v2',
  library:  'audiodev.studio.library.v1',
  restore:  'audiodev.studio.restore.v1',
  player:   'audiodev.studio.player.v1',
};

function readRaw(key) {
  try {
    const s = localStorage.getItem(key);
    if (!s) return {};
    const o = JSON.parse(s);
    return (o && typeof o === 'object') ? o : {};
  } catch { return {}; }
}

function writeRaw(key, obj) {
  try { localStorage.setItem(key, JSON.stringify(obj)); } catch { /* private mode */ }
}

/** Merge saved over defaults, one field at a time. */
export function load(name, defaults = {}) {
  const saved = readRaw(KEYS[name]);
  const out = {};
  for (const k of Object.keys(defaults)) out[k] = saved[k] ?? defaults[k];
  return out;
}

export function save(name, obj) {
  writeRaw(KEYS[name], obj);
}

/** Debounced writer — the generate form saves on every keystroke. */
export function saver(name, ms = 400) {
  let h = null, pending = null;
  return (obj) => {
    pending = obj;
    clearTimeout(h);
    h = setTimeout(() => { writeRaw(KEYS[name], pending); }, ms);
  };
}

export const GENERATE_FIELDS = ['model', 'prompt', 'lyrics', 'duration', 'steps', 'seed', 'instrumental', 'post_kind'];
