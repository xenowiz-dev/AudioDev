/* mock.js — canned JSON so the UI is developable and screenshottable with no
   server. Enabled by `?mock=1`. It implements exactly the transport interface
   api.js expects (json / upload / stream), so nothing else in the app knows.

   The dataset is deliberately awkward: take-pairs with a shared record, a row
   with no record at all, a missing cover, all three rating states, played and
   unplayed. Those are the rows that break layouts. */

const OUT = 'B:\\AudioDev\\Music\\studio';

/* ── deterministic noise ───────────────────────────────────────────────── */
let _s = 1234567;
const rnd = () => (_s = (_s * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff;
const pick = (a) => a[Math.floor(rnd() * a.length)];

/* ── a real (tiny) wav so the scrubber, timeupdate and ended all work ──── */
function wavUrl(seconds, freq) {
  const sr = 8000, n = Math.floor(sr * seconds);
  const buf = new ArrayBuffer(44 + n * 2);
  const v = new DataView(buf);
  const str = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
  str(0, 'RIFF'); v.setUint32(4, 36 + n * 2, true); str(8, 'WAVEfmt ');
  v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
  v.setUint32(24, sr, true); v.setUint32(28, sr * 2, true);
  v.setUint16(32, 2, true); v.setUint16(34, 16, true);
  str(36, 'data'); v.setUint32(40, n * 2, true);
  for (let i = 0; i < n; i++) {
    const env = Math.min(1, i / (sr * 0.05), (n - i) / (sr * 0.4));
    v.setInt16(44 + i * 2, Math.sin(i / sr * freq * 6.283) * 7000 * env, true);
  }
  return URL.createObjectURL(new Blob([buf], { type: 'audio/wav' }));
}
const AUDIO = [wavUrl(23, 196), wavUrl(31, 262), wavUrl(17, 147)];

/* ── covers, drawn once each ───────────────────────────────────────────── */
function coverUrl(i) {
  const c = document.createElement('canvas');
  c.width = c.height = 96;
  const g = c.getContext('2d');
  const h = (i * 47) % 360;
  const grad = g.createLinearGradient(0, 0, 96, 96);
  grad.addColorStop(0, `hsl(${h} 45% 26%)`);
  grad.addColorStop(1, `hsl(${(h + 60) % 360} 40% 12%)`);
  g.fillStyle = grad; g.fillRect(0, 0, 96, 96);
  g.strokeStyle = `hsl(${h} 70% 62% / .8)`; g.lineWidth = 2;
  g.beginPath();
  for (let x = 0; x <= 96; x += 3) {
    const y = 48 + Math.sin((x + i * 9) / 7) * (10 + (i % 5) * 4) * Math.sin(x / 30);
    x ? g.lineTo(x, y) : g.moveTo(x, y);
  }
  g.stroke();
  return c.toDataURL('image/png');
}

/* ── prompts, in both dialects ─────────────────────────────────────────── */
const MM = (bpm, key, genre, mood) =>
  `Global Metadata\nBasic Attributes: bpm is ${bpm}. key is ${key}, and scale is major. ${genre}.\n` +
  `Vocal Details: ${mood} female lead, close-mic, light doubling on the chorus.\n` +
  `Arrangement: fingerpicked acoustic guitar, brushed kit, upright bass, string pad from the second chorus.`;
const AS = (...tags) => tags.join(', ');

const STYLES = [
  ['minimax', MM(96, 'C', 'Acoustic Pop', 'Warm')],
  ['minimax', MM(128, 'A', 'Synthwave', 'Breathy')],
  ['minimax', MM(72, 'D', 'Doom Folk', 'Low')],
  ['acestep', AS('upbeat indie pop', 'jangly electric guitar', 'live drums', 'warm analog production', '110 bpm')],
  ['acestep', AS('post-rock', 'reverb-drenched guitars', 'slow build', 'tape saturation', '74 bpm')],
  ['acestep', AS('lo-fi hip hop', 'dusty rhodes', 'vinyl crackle', 'boom bap drums', '86 bpm')],
  ['acestep', AS('shoegaze', 'wall of fuzz', 'buried vocals', 'octave pedal', '132 bpm')],
];
const LYRICS = `[verse]\nMorning light filtering through the pine\nEvery quiet street is yours and mine\n[chorus]\nSoftly the world begins to breathe\nAnd everything I held starts to leave`;

const TITLES = ['Softly The World Begins', 'Filtering Through The Pine', 'Every Quiet Street',
  'Held Starts To Leave', 'Brushed Kit At Dawn', 'Tape Saturation', 'Wall Of Fuzz',
  'Dusty Rhodes', 'Slow Build', 'Octave Pedal', 'Upright Bass Nocturne', 'Vinyl Crackle',
  'Close Mic Confession', 'String Pad Second Chorus', 'Low Female Lead'];

/* ── the library ───────────────────────────────────────────────────────── */
function buildEntries() {
  const out = [];
  const now = Date.now() / 1000;
  let i = 0;
  while (out.length < 82) {
    const [model, prompt] = STYLES[i % STYLES.length];
    const pair = model === 'acestep' && i % 3 === 0;   // ACE-Step writes two takes
    const takes = pair ? 2 : 1;
    for (let t = 1; t <= takes; t++) {
      const n = out.length;
      const stamp = new Date((now - n * 5400) * 1000);
      const id = `${stamp.toISOString().slice(0, 10).replace(/-/g, '')}-${String(100000 + n).slice(1)}_${model}.wav`;
      const shared = takes > 1 && t === 2;
      const norec = n === 6 || n === 19;              // found on disk, no sidecar
      const noprompt = n === 11;                      // sidecar, but no prompt
      const rating = norec ? 0 : pick([0, 0, 0, 1, -1, 0]);
      const played = n > 3 && rnd() > 0.45;
      out.push({
        id, name: id, path: `${OUT}\\${id}`,
        title: norec ? id : TITLES[n % TITLES.length],
        audio_url: AUDIO[n % AUDIO.length],
        cover_url: (n === 2 || n === 9 || n === 23) ? null : coverUrl(n),
        mtime: now - n * 5400,
        created: stamp.toISOString().slice(0, 19),
        recorded: !norec && !shared,
        shared_rec: shared,
        take: t, takes,
        favorite: !shared && rating === 1,
        rating: shared ? 0 : rating,
        model: norec ? null : model,
        prompt: (norec || noprompt) ? null : prompt,
        lyrics: (norec || model === 'acestep') ? null : LYRICS,
        instrumental: model === 'acestep' && n % 7 === 0,
        duration: 30, steps: model === 'minimax' ? 30 : 8, seed: 7 + n,
        seconds: [23.4, 31.2, 17.8][n % 3],
        sampling_rate: model === 'minimax' ? 44100 : 48000,
        elapsed: 140 + n, vram_peak: 8.71, preview: false, post_kind: 'none',
        cover_src: null, cover_strength: null, noise_strength: null,
        // Provenance (REQUIREMENTS 11, 13), shaped exactly as api_entry sends
        // it: absent is null, never '' and never 'none' — the record sheet
        // omits null, so a printed 'none' would claim something false about
        // how a track was made. The loop below fills two ACE-Step rows in.
        workspace: null,
        lora: null, lora_name: null, lora_scale: null, variant: null,
        thinking: null, lm_temperature: null,
        quality: (norec || model !== 'minimax') ? null
          : { preset: 'default10', llm: '4bit', rvq: 'bf16', kv: 'quantized', reserve: '1GB' },
        title_override: '', style: n % 9 === 0 ? 'doom folk, post rock' : '',
        played, plays: played ? 1 + Math.floor(rnd() * 4) : 0,
        last_played: played ? new Date((now - n * 900) * 1000).toISOString() : null,
        caption: '',
      });
    }
    i++;
  }
  // Two rows carry the LoRA/thinking provenance so `?mock=1&q=epoch10` finds
  // the A/B pair REQUIREMENTS 11 exists to enable, and q=soundtrack finds a
  // workspace rather than only prompts that happen to say the word. Every
  // other row keeps all seven fields null, which is what 91 real tracks look
  // like — the record sheet must render both without inventing a value.
  for (const e of out.filter((x) => x.model === 'acestep' && x.recorded).slice(0, 2)) {
    e.workspace = 'Soundtrack';
    e.lora = 'trained:epoch10';
    e.lora_name = 'epoch10';
    e.lora_scale = 0.8;
    e.variant = 'acestep-v15-turbo';       // the name the worker records
    e.thinking = true;
    e.lm_temperature = 1.1;
    e.style = 'doom folk, carnival';
  }
  for (const e of out) {
    const bits = [];
    if (e.takes > 1) bits.push(`take ${e.take}/${e.takes}`);
    if (e.seconds) bits.push(`${Math.round(e.seconds)}s`);
    if (e.model) bits.push(e.model);
    e.caption = `${e.favorite ? '★ ' : ''}${e.title}\n${bits.join(' · ')}`;
  }
  return out;
}

const ENTRIES = buildEntries();
const PLAYLISTS = { 'late night': ENTRIES.slice(2, 9).map((e) => e.id), demos: ENTRIES.slice(0, 4).map((e) => e.id) };
// The workspace registry the server keeps in Music/studio/workspaces.json.
// 'Soundtrack' is the one two seeded rows are tagged with, so the bar has a
// name that actually selects something.
// Generic on purpose. These were the user's own folder names, which is
// exactly the kind of label the app should not be inventing for itself.
const WORKSPACES = ['Soundtrack', 'Singles', 'Sketches'];
let ARTIST = '';
let ACTIVE_WS = null;

const CONFIG = {
  version: 1,
  outdir: OUT,
  default_model: 'minimax',
  backends: {
    minimax: {
      label: 'MiniMax Music 3',
      note: '44.1 kHz stereo, sung lyrics. 4-bit LLM. ~7x realtime.',
      vram_gb: 8.0, requires_lyrics: true, supports_instrumental: false, supports_cover: false,
      default_steps: 30, default_prompt: MM(96, 'C', 'Acoustic Pop', 'Warm'), prompt_lines: 10,
      prompt_info: 'MiniMax was trained on sectioned captions — keep the Global Metadata / Vocal Details / Arrangement headings; short prompts lose arrangement control.',
      lyrics_info: 'Required — MiniMax has no instrumental mode.',
    },
    acestep: {
      label: 'ACE-Step 1.5',
      note: '48 kHz. Fast, and the only one here that can cover a track.',
      vram_gb: 6.0, requires_lyrics: false, supports_instrumental: true, supports_cover: true,
      default_steps: 8, default_prompt: STYLES[3][1], prompt_lines: 4,
      prompt_info: 'Plain keyword-style prompt.',
      lyrics_info: 'Leave blank, or tick Instrumental above.',
    },
  },
  default_lyrics: LYRICS,
  upscalers: {
    none: 'None',
    apollo: 'Apollo — fast, trained on codec artifacts (7.00 dB)',
    audiosr: 'AudioSR — most accurate, ~10x slower (4.37 dB)',
    flashsr: 'FlashSR — fastest, least accurate (11.34 dB)',
  },
  degrades: {
    off: 'Off — upscale the file exactly as it is',
    auto: 'Auto — the damage this upscaler was trained on',
    cut14: 'Cut above 14 kHz — gentle, leaves most of the air',
    cut11: 'Cut above 11 kHz — a big band to rebuild',
    mp3: 'MP3 128k round-trip — real codec artifacts',
    both: 'MP3 128k + cut above 11 kHz — the most to rebuild',
  },
  writer: {
    ready: true,
    model_id: 'Qwen/Qwen2.5-1.5B-Instruct',
    targets: ['lyrics', 'style'],
    modes: ['generate', 'extend', 'edit'],
    mode_info: {
      generate: 'Writes the whole thing from your brief. Anything already in the box is ignored.',
      extend: 'Continues from what is already there, and does not repeat it. Good for adding a bridge and a final chorus.',
      edit: 'Changes only what you ask for and returns the rest word for word. The brief is the instruction: “make the chorus angrier”.',
    },
    note: 'Takes the GPU when it is idle (~30 s for a full lyric) and falls back to the CPU when it is not.',
  },
  naturalizers: {
    off: 'Off',
    subtle: 'Subtle — you will only hear it in an A/B',
    medium: 'Medium — the default analogue pass',
    strong: 'Strong — obvious colour',
  },
  fingerprint_note: 'AudioSR and FlashSR stamp a 100 Hz neural-vocoder comb on their output (hop 480 @ 48 kHz, confirmed in their source) — upscaling adds a detectable AI fingerprint that was not there before.',
  whisper_models: ['large-v3', 'medium', 'small'],
  defaults: {
    duration: 30, steps: 30, seed: 7, instrumental: false,
    post_kind: 'none', cover_strength: 1.0, noise_strength: 0.75,
    upscaler: 'apollo', auto_lowpass: true, fill: true,
    degrade: 'off', naturalize: 'off', lufs: -14.0,
    region: { t0: 0, t1: 0, flo: 16000, fhi: 0 },
    whisper_model: 'large-v3', separate: true, seconds: 0,
  },
  limits: {
    duration: [10, 300, 5], steps: [4, 60, 1],
    cover_strength: [0.0, 1.0, 0.05], noise_strength: [0.0, 1.0, 0.05],
    preview_seconds: 15, upload_mb: 200,
  },
  /* The quality contract (§6). Mirrors what a 10 GB card reports, so the
     Advanced panel is developable and screenshottable with no GPU. */
  quality: {
    detected_total_gb: 10.0,
    detected_free_gb: 9.0,
    recommended: 'default10',
    applies_to: ['minimax'],
    presets: [
      { id: 'smallest',  label: 'Smallest',            llm: '4bit', rvq: '4bit', kv: 'quantized', needs_4min_gb: 7.87,  note: 'Everything squeezed. Fits any card, and you can hear the squeeze.' },
      { id: 'balanced',  label: 'Balanced',            llm: '4bit', rvq: '8bit', kv: 'quantized', needs_4min_gb: 8.17,  note: 'Some acoustic detail back for a third of a gigabyte.' },
      { id: 'default10', label: 'Default (10 GB)',     llm: '4bit', rvq: 'bf16', kv: 'quantized', needs_4min_gb: 8.74,  note: 'Full-detail decoder, song memory stored coarser. What this box has been shipping.' },
      { id: 'fullkv',    label: 'Full cache (12 GB+)', llm: '4bit', rvq: 'bf16', kv: 'static',    needs_4min_gb: 10.18, note: 'Keeps the song memory at full precision.' },
      { id: 'high',      label: 'High (16 GB+)',       llm: '8bit', rvq: 'bf16', kv: 'static',    needs_4min_gb: 12.37, note: 'A better-behaved music model — the first real quality step up.' },
      { id: 'maximum',   label: 'Maximum (24 GB+)',    llm: 'bf16', rvq: 'bf16', kv: 'dynamic',   needs_4min_gb: 19.88, note: 'Nothing traded away. Needs a 24 GB card.' },
    ],
    fields: { llm: ['4bit', '8bit', 'bf16'], rvq: ['4bit', '8bit', 'bf16'],
              kv: ['auto', 'quantized', 'static', 'dynamic'], reserve: ['0.5GB', '1GB', '2GB'] },
    costs: { llm: 'musicality and prompt adherence', rvq: 'acoustic detail',
             kv: 'storage only - full history, stored coarser' },
    defaults: { llm: '4bit', rvq: 'bf16', kv: 'auto', reserve: '1GB' },
    restart_note: 'LLM and RVQ precision are applied while the weights load, so changing either restarts MiniMax — about 20 s before the next song starts. KV cache mode and the offload reserve are per request and cost nothing.',
  },
  // Parity with /api/config. The server COMPOSES `capabilities` from the
  // feature blocks below it; the mock states them, because there is no server
  // here to compose anything -- but the shape and the values must match, or
  // the offline shell shows a form the real one never would.
  capabilities: {
    minimax: {
      thinking: false, auto_duration: false, takes: false, variants: false,
      loras: false, quality: true,
      cover: false, instrumental: false, lyrics_required: true,
    },
    acestep: {
      thinking: true, auto_duration: true, takes: true, variants: true,
      loras: true, quality: false,
      cover: true, instrumental: true, lyrics_required: false,
    },
  },
  requires: { auto_duration: 'thinking' },

  thinking: {
    default: false, applies_to: ['acestep'], model: 'acestep-5Hz-lm-1.7B',
    temperature: [0.0, 2.0, 0.05],
    note: "ACE-Step's 5Hz language model reasons over your prompt before any audio is made.",
  },
  auto_duration: {
    applies_to: ['acestep'], requires: 'thinking',
    note: 'The model sizes the song to the lyrics and caption.',
  },
  takes: {
    default: 1, max: 2, applies_to: ['acestep'],
    note: 'Each take is a different song from the same prompt, not a copy.',
  },
  variants: {
    active: 'acestep-v15-turbo', applies_to: ['acestep'],
    note: 'The DiT checkpoint. Switching restarts the worker (~40 s).',
    items: [
      { name: 'acestep-v15-base', label: 'Base (2B) - continue, extract, guidance',
        steps: 32, needs_gb: 7.3, installed: true, fits: true, active: false,
        note: 'CFG and 32-100 steps, and the only 2B variant that can continue a track.',
        tasks: ['text2music', 'repaint', 'cover', 'cover-nofsq', 'extract', 'lego', 'complete'],
        supports: { cover: true, repaint: true, complete: true, extract: true, lego: true, guidance: true } },
      { name: 'acestep-v15-turbo', label: 'Turbo (2B) - fast',
        steps: 8, needs_gb: 7.0, installed: true, fits: true, active: true,
        note: '8 steps, no CFG.',
        tasks: ['text2music', 'repaint', 'cover', 'cover-nofsq'],
        supports: { cover: true, repaint: true, complete: false, extract: false, lego: false, guidance: false } },
    ],
  },
  loras: {
    applies_to: ['acestep'], drop_path: 'B:\\AudioDev\\loras',
    note: 'A LoRA nudges ACE-Step toward the style it was trained on.',
    items: [
      { id: 'trained:epoch10', name: 'epoch10', label: 'epoch10 - rank 16, alpha 32, 42 MB',
        source: 'trained', rank: 16, alpha: 32, peft_type: 'LORA',
        target_modules: ['k_proj', 'o_proj', 'q_proj', 'v_proj'],
        base_model_name: 'acestep-v15-turbo', size_bytes: 44089608 },
    ],
  },
  llm_title: { ready: true, model_id: 'Qwen/Qwen2.5-1.5B-Instruct', hint: 'Qwen/Qwen2.5-1.5B-Instruct on CPU, ~9 s. Off uses the instant heuristic.' },
  phone: { url: 'https://box.tailnet.ts.net/', qr_url: null },
  notes: {
    post_kind: 'Runs after generation. Measured: neither generator leaves a codec cliff (MiniMax 0.6 dB, ACE-Step 3.9 dB), and upresing audio with no cliff makes it worse — so this is normally for lossy source material, on the Restore tab. The bandwidth verdict is logged either way.',
    cover: 'Upload a track to regenerate it in the style above. Noise strength is the parameter that makes it a cover — 0.0 means pure noise and produces an unrelated song. 0.4–0.8.',
    auto_lowpass: 'AudioSR was trained on lowpass-filtered audio only; fed a raw lossy file it hallucinates from codec artifacts. Leave on for anything compressed.',
    region_fill: 'An upscaler rewrites the whole file (~13% relative error below the crossover); splicing adds only the masked correction, so everything outside the box stays bit-identical.',
    sidecar: 'Each track keeps a JSON sidecar holding its prompt, lyrics and settings, so the record travels with the file. Not written into the audio\u2019s tags. Cover art is rendered from each track\u2019s own spectrogram. Trash is a folder move, never a delete.',
  },
};

/* a placeholder spectrogram, drawn so Restore has something to click */
function specPng() {
  const c = document.createElement('canvas');
  c.width = 1300; c.height = 560;
  const g = c.getContext('2d');
  g.fillStyle = '#101014'; g.fillRect(0, 0, 1300, 560);
  for (let x = 78; x < 1248; x++) {
    for (let y = 28; y < 504; y += 4) {
      const f = 1 - (y - 28) / 476;
      const e = Math.max(0, (f < 0.72 ? 1 : 0.06) * (0.35 + 0.65 * Math.abs(Math.sin(x / 40 + y / 90))));
      g.fillStyle = `hsl(${260 - e * 220} 80% ${8 + e * 46}%)`;
      g.fillRect(x, y, 1, 4);
    }
  }
  g.strokeStyle = '#7A5CFF'; g.lineWidth = 1;   // --violet, per the design system
  g.strokeRect(78, 28, 1170, 476);
  g.fillStyle = '#8B85B8'; g.font = '13px monospace';   // --muted
  g.fillText('22.05 kHz', 6, 34); g.fillText('16 kHz', 22, 160); g.fillText('0', 62, 504);
  return c.toDataURL('image/png');
}
let SPEC = null;

const GEOM = {
  png: `${OUT}\\upscaled\\_spec_9f21c4e0.png`,
  img_w: 1300, img_h: 560, x0: 78, x1: 1248, y0: 28, y1: 504,
  t0: 0.0, t1: 182.4, f0: 0.0, f1: 22050.0,
  sr: 44100, full_duration: 182.4, truncated: false, cliff: 16000.0, drop: 12.4,
};
const BANDWIDTH = `check_bandwidth  song.mp3
  sample rate      44100 Hz  (Nyquist 22050 Hz)
  energy cliff     16000 Hz
  drop across it   12.4 dB
  verdict          lossy source — a real codec cliff, worth restoring`;

/* ── fake jobs ─────────────────────────────────────────────────────────── */
const JOBS = new Map();
let jobN = 0;

function newJob(kind, request) {
  const id = `j_mock${++jobN}`;
  const job = {
    id, kind, state: 'queued', created: Date.now() / 1000, started: null, ended: null,
    queue_position: 0, seq: 0, log: [], live: null, artifacts: [],
    result: null, error: null,
    vram: { used_gb: 0.2, total_gb: 10.0, free_gb: 9.8, loaded: null },
    request, subs: new Set(), cancelled: false,
  };
  JOBS.set(id, job);
  return job;
}

const SCRIPTS = {
  generate: [
    [0, 'progress', { msg: 'starting MiniMax Music 3 (first load takes ~20 s)' }],
    [1, 'vram', { used_gb: 8.4, total_gb: 10, free_gb: 1.6, loaded: 'minimax' }],
    [2, 'progress', { msg: 'encoding the prompt' }],
    ...Array.from({ length: 10 }, (_, k) => [3 + k, 'progress',
      { msg: `autoregressive ${(k + 1) * 10}%`, stage: 'ar', frac: (k + 1) / 10 }]),
    [13, 'progress', { msg: 'diffusion decode' }],
    ...Array.from({ length: 8 }, (_, k) => [14 + k, 'progress',
      { msg: `decoding ${(k + 1) * 12}%`, stage: 'decode', frac: (k + 1) / 8 }]),
    [23, 'artifact', { kind: 'audio', label: 'generated', path: `${OUT}\\mock_generated.wav`, url: AUDIO[1] }],
    [24, 'progress', { msg: 'wrote the sidecar' }],
    [25, 'result', {
      path: `${OUT}\\mock_generated.wav`, url: AUDIO[1],
      library_id: 'mock_generated.wav',
      final_path: `${OUT}\\mock_generated.wav`, final_url: AUDIO[1],
      seconds: 31.2, elapsed: 214.0, sampling_rate: 44100, realtime_ratio: 6.86,
      vram_peak: 8.71, vram_idle: 0.15, preview: false, post: null,
    }],
  ],
  upscale: [
    [0, 'progress', { msg: '--- Apollo — fast, trained on codec artifacts ---' }],
    [1, 'progress', { msg: BANDWIDTH.split('\n')[4].trim() }],
    [2, 'phase', { phase: 'upscale', label: 'Apollo' }],
    ...Array.from({ length: 6 }, (_, k) => [3 + k, 'progress',
      { msg: `chunk ${k + 1}/6`, stage: 'apollo', frac: (k + 1) / 6 }]),
    [10, 'phase', { phase: 'splice', label: 'region fill' }],
    [11, 'progress', { msg: 'splicing 16000 Hz upward' }],
    [12, 'result', {
      path: `${OUT}\\upscaled\\source_apollo_hybrid.wav`, url: AUDIO[0],
      kind: 'apollo', spliced: true,
      upscaled_path: `${OUT}\\upscaled\\source_apollo.wav`, upscaled_url: AUDIO[0],
      compare_url: null,
      bandwidth_before: BANDWIDTH, bandwidth_after: BANDWIDTH.replace('16000 Hz', '21500 Hz'),
      region: '0.000:*:16000:*',
    }],
  ],
  transcribe: [
    [0, 'progress', { msg: 'transcribing song.mp3' }],
    [1, 'progress', { msg: 'separating vocals (demucs)' }],
    ...Array.from({ length: 5 }, (_, k) => [2 + k, 'progress',
      { msg: `whisper ${(k + 1) * 20}%`, stage: 'whisper', frac: (k + 1) / 5 }]),
    [8, 'result', {
      text: LYRICS, segments: 42, language: 'en', language_probability: 0.99,
      elapsed: 88.0, separated: true,
      summary: '42 segments, language en (0.99), 88s · vocals isolated',
      note: 'Structure tags are NOT invented — add [verse] / [chorus] yourself if you want ACE-Step to follow them.',
    }],
  ],
  write: [
    [0, 'progress', { msg: 'generate lyrics for minimax — on the GPU (idle), about 30 s' }],
    [1, 'phase', { phase: 'write', label: 'Generate lyrics' }],
    ...Array.from({ length: 5 }, (_, k) => [2 + k, 'progress',
      { msg: `writing ${(k + 1) * 70} tokens` }]),
    [8, 'result', {
      text: LYRICS, target: 'lyrics', mode: 'generate', model: 'minimax',
      seed: 428913, tokens: 351, retried: false, device: 'cuda', warning: null,
    }],
  ],
};

const TERMINAL = new Set(['result', 'error', 'cancelled']);

function runJob(job) {
  const script = SCRIPTS[job.kind] || SCRIPTS.generate;
  job.state = 'running'; job.started = Date.now() / 1000;
  const step = 140;   // ms per scripted tick — fast enough for a test run
  for (const [tick, name, data] of script) {
    setTimeout(() => {
      if (job.cancelled) return;
      job.seq += 1;
      const payload = { seq: job.seq, t: (Date.now() / 1000) - job.started, ...data };
      if (name === 'result') { job.state = 'done'; job.result = payload; }
      if (name === 'error') { job.state = 'error'; job.error = payload; }
      if (name === 'artifact') job.artifacts.push(payload);
      if (name === 'vram') job.vram = { ...job.vram, ...data };
      for (const s of job.subs) s(name, payload);
      if (TERMINAL.has(name)) job.subs.clear();
    }, 200 + tick * step);
  }
}

/* ── the transport ─────────────────────────────────────────────────────── */
const delay = (ms) => new Promise((r) => setTimeout(r, ms));

function libraryResponse(params) {
  let list = ENTRIES.slice();
  const pl = params.get('playlist');
  if (pl) {
    const want = PLAYLISTS[pl] || [];
    list = want.map((id) => list.find((e) => e.id === id)).filter(Boolean);
  }
  if (params.get('fav') === '1') list = list.filter((e) => e.favorite);
  if (params.get('unplayed') === '1') list = list.filter((e) => !e.played);
  const q = (params.get('q') || '').toLowerCase();
  if (q) {
    // Same haystack and same rule as get_library: style (what the Edit sheet
    // writes), workspace, lora_name and variant are searchable, and EVERY
    // whitespace-separated term must appear rather than the whole query as
    // one literal substring — so "doom folk" matches a style of "doom, folk".
    // Nothing an older query matched is lost: a contiguous substring already
    // contains all of its terms.
    const terms = q.split(/\s+/).filter(Boolean);
    list = list.filter((e) => {
      const hay = [e.title, e.name, e.prompt, e.lyrics, e.model, e.style,
        e.workspace, e.lora_name, e.variant].filter(Boolean).join(' ').toLowerCase();
      return terms.every((t) => hay.includes(t));
    });
  }
  const total = list.length;
  const offset = Number(params.get('offset') || 0);
  const limit = Number(params.get('limit') || 500);
  const page = list.slice(offset, offset + limit);
  const withPrompt = list.filter((e) => e.recorded || e.shared_rec).length;
  const awaiting = list.filter((e) => !e.cover_url).length;
  return {
    entries: page, total, returned: page.length, offset,
    with_prompt: withPrompt, awaiting_cover: awaiting,
    status: `${total} tracks · ${withPrompt} with a prompt  ·  ${awaiting} awaiting cover art`,
    playlists: Object.keys(PLAYLISTS),
    // The server sends this and the workspace bar reads it; without it the
    // offline shell cannot exercise §16 at all.
    workspaces: { active: ACTIVE_WS, names: [...WORKSPACES] },
  };
}

function byId(id) { return ENTRIES.find((e) => e.id === id); }

export function transport() {
  return {
    async json(method, url, body, opts = {}) {
      await delay(40);
      // Honour AbortController like the real transport, so debounce-then-abort
      // code paths behave the same in mock mode.
      if (opts.signal?.aborted) { const e = new Error('aborted'); e.name = 'AbortError'; throw e; }
      const u = new URL(url, location.origin);
      const p = u.pathname, P = u.searchParams;
      const seg = p.split('/').filter(Boolean);      // ["api","library","x.wav",...]

      if (p === '/api/profile') {
        // Studio-wide settings. Starts UNSET here too: the mock exists to
        // show what a fresh box looks like, and a seeded name would hide
        // the empty state the sheet has to handle.
        if (method === 'PUT') {
          const raw = String(body?.artist ?? '');
          ARTIST = raw.replace(/[\u0000-\u001f\u007f]+/g, ' ')
            .replace(/\s{2,}/g, ' ').trim().slice(0, 64);
          return { artist: ARTIST,
                   note: raw.trim().length > 64 ? 'Shortened to 64 characters.' : null };
        }
        return { artist: ARTIST };
      }
      if (p === '/api/config') return CONFIG;
      if (p === '/api/health') return { ok: true, uptime_s: 42.0, jobs_active: 0 };
      if (p === '/api/vram') {
        const loaded = [...JOBS.values()].some((x) => x.state === 'running') ? 'minimax' : null;
        return loaded
          ? { available: true, used_gb: 8.4, total_gb: 10.0, free_gb: 1.6, loaded, loaded_label: 'MiniMax Music 3', warn_low_free: false }
          : { available: true, used_gb: 2.1, total_gb: 10.0, free_gb: 7.9, loaded: null, loaded_label: null, warn_low_free: true };
      }
      if (p === '/api/budget') {
        if (P.get('model') !== 'minimax') return { applies: false, reason: 'model' };
        const d = Number(P.get('duration') || 30);
        // llm/rvq/kv are OPTIONAL: with none of them, answer for the shipped
        // defaults exactly as the old endpoint did.
        const LLM = { '4bit': 6.29, '8bit': 8.47, bf16: 15.99 };
        const RVQ = { '4bit': 0.33, '8bit': 0.64, bf16: 1.20 };
        const llm = P.get('llm') || '4bit', rvq = P.get('rvq') || 'bf16', kv = P.get('kv') || 'static';
        const need = (LLM[llm] ?? LLM['4bit']) + (RVQ[rvq] ?? RVQ.bf16)
                   + d * 0.008 * (kv === 'quantized' ? 0.25 : 1) + 0.77;
        const free = 9.0;
        return { applies: true, need_gb: +need.toFixed(2), free_gb: free, headroom_gb: +(free - need).toFixed(2), fits: need <= free };
      }
      if (p === '/api/gpu/state') {
        const loaded = [...JOBS.values()].some((x) => x.state === 'running') ? 'minimax' : null;
        return {
          loaded, loaded_label: loaded ? 'MiniMax Music 3' : null,
          quality: loaded ? { llm: '4bit', rvq: '4bit', reserve: '1GB', base_gb: 6.63 } : null,
          applies_to: ['minimax'],
          restart_note: CONFIG.quality.restart_note,
        };
      }
      if (p === '/api/gpu/unload') return { unloaded: true, vram: { available: true, used_gb: 0.2, total_gb: 10, free_gb: 9.8, loaded: null, loaded_label: null, warn_low_free: false } };

      if (p === '/api/library') return libraryResponse(P);
      if (p === '/api/library/covers') return { queued: 3, running: true };
      if (seg[0] === 'api' && seg[1] === 'library' && seg[2]) {
        const e = byId(decodeURIComponent(seg[2]));
        if (!e) throw mkErr(404, 'not_found', 'No such track.');
        const tail = seg[3];
        if (!tail && method === 'GET') return { entry: e, detail_md: `### ${e.name}\n\nmodel **${e.model}** · seed **${e.seed}**`, shared_note: e.shared_rec ? '*Prompt inherited from take 2 of the same run — ACE-Step writes two takes and only one carries the record.*' : null };
        if (!tail && method === 'PATCH') {
          // Mirrors the real endpoint: title/style only. played/plays/last_played
          // are deliberately ignored here so mock mode cannot pass a test the
          // live server would fail -- POST .../played is the one that writes them.
          if (body.title !== undefined) { e.title_override = body.title; e.title = body.title || TITLES[0]; }
          if (body.style !== undefined) e.style = body.style;
          return { entry: e, message: `Saved details for ${e.name}` };
        }
        if (tail === 'played' && method === 'POST') {
          e.played = true;
          e.plays = Number(body.plays) || (e.plays || 0) + 1;
          e.last_played = new Date().toISOString();
          return { played: true, plays: e.plays, entry: e };
        }
        if (tail === 'retitle') { e.title = 'Softly The World Begins'; e.title_override = e.title; return { title: e.title, how: 'LLM', entry: e, message: `Titled **${e.title}** (LLM)` }; }
        if (tail === 'favorite') { e.favorite = !!body.on; return { favorite: e.favorite, entry: e }; }
        if (tail === 'rating') {
          e.rating = body.rating;
          if (body.rating === 1) e.favorite = true;
          if (body.rating !== 1) e.favorite = false;
          return { rating: e.rating, favorite: e.favorite, entry: e, message: body.rating === 1 ? `Liked — ${e.name}` : `Rating cleared — ${e.name}` };
        }
        if (tail === 'trash') {
          const i = ENTRIES.indexOf(e); if (i >= 0) ENTRIES.splice(i, 1);
          return { trashed: true, dest: `${OUT}\\trash\\${e.name}`, message: `Moved **${e.name}** to trash\\ (recoverable).` };
        }
        if (tail === 'reuse') {
          if (!e.recorded) throw mkErr(409, 'no_record', 'That track has no saved prompt — it predates the library, or is an alternate take.');
          return { model: e.model, prompt: e.prompt, lyrics: e.lyrics || '', duration: e.duration, steps: e.steps, seed: e.seed, instrumental: e.instrumental, message: `Loaded **${e.name}** into Generate ✓` };
        }
        if (tail === 'cover-source') {
          return {
            model: 'acestep', prompt: e.prompt || STYLES[3][1], lyrics: e.lyrics || '', instrumental: false,
            src_path: e.path, src_url: e.audio_url, cover_strength: 1.0, noise_strength: 0.75,
            lyrics_probe: { ok: true, found: !!e.lyrics, text: e.lyrics, message: e.lyrics ? '**Found lyrics** in the track\u2019s own sidecar (5 lines). Apply to use.' : 'No lyrics found beside that file.' },
            message: `Covering **${e.title}** — source loaded in Generate. Edit the style, then Generate.`,
          };
        }
      }

      if (p === '/api/playlists' && method === 'GET') return { playlists: Object.entries(PLAYLISTS).map(([name, t]) => ({ name, count: t.length })) };
      if (p === '/api/playlists' && method === 'POST') { PLAYLISTS[body.name] = PLAYLISTS[body.name] || []; return { name: body.name, created: true }; }
      if (seg[1] === 'playlists' && seg[2]) {
        const name = decodeURIComponent(seg[2]);
        if (!seg[3] && method === 'GET') return { name, tracks: PLAYLISTS[name] || [] };
        if (!seg[3] && method === 'DELETE') { delete PLAYLISTS[name]; return { deleted: true }; }
        if (seg[3] === 'tracks' && method === 'POST') {
          PLAYLISTS[name] = PLAYLISTS[name] || [];
          const had = PLAYLISTS[name].includes(body.id);
          if (!had) PLAYLISTS[name].push(body.id);
          return { added: !had, count: PLAYLISTS[name].length, message: `Added to **${name}** (${PLAYLISTS[name].length} tracks)` };
        }
        if (seg[3] === 'tracks' && method === 'DELETE') {
          const id = decodeURIComponent(seg[4] || '');
          PLAYLISTS[name] = (PLAYLISTS[name] || []).filter((x) => x !== id);
          return { removed: true };
        }
        if (seg[3] === 'reorder') return { ok: true, tracks: PLAYLISTS[name] || [] };
      }
      if (p === '/api/trash') return { items: [] };

      if (p === '/api/lyrics/probe') {
        if (!body.path) return { ok: false, found: false, text: null, message: 'Pick a source track to search for its lyrics.' };
        return { ok: true, found: true, text: LYRICS, message: '**Found lyrics** in file .lrc — matched by stem (5 lines). Apply to use.', best: { source: 'sidecar', field: '.lrc', text: LYRICS, why: 'matched by stem' }, embedded: [], sidecars: [`${OUT}\\song.lrc`] };
      }
      if (p === '/api/restore/inspect') {
        SPEC = SPEC || specPng();
        return { bandwidth: BANDWIDTH, image_url: SPEC, geometry: GEOM, note: 'Click a corner of the region you want to fill.', truncated_note: null };
      }
      if (p === '/api/restore/cliff') {
        SPEC = SPEC || specPng();
        return { cliff_hz: 16000.0, drop_db: 12.4, weak: false, message: null, inspect: { bandwidth: BANDWIDTH, image_url: SPEC, geometry: GEOM, note: 'Click a corner of the region you want to fill.', truncated_note: null } };
      }
      if (p === '/api/fs/stat') return { exists: true, is_file: true, size: 8412345, playable: false, name: (P.get('p') || '').split('\\').pop() };

      if (p === '/api/generate' || p === '/api/upscale'
          || p === '/api/lyrics/transcribe' || p === '/api/write') {
        const kind = p === '/api/generate' ? 'generate'
          : p === '/api/upscale' ? 'upscale'
            : p === '/api/write' ? 'write' : 'transcribe';
        const job = newJob(kind, body);
        runJob(job);
        return { job_id: job.id, kind, state: 'queued', stream_url: `/api/jobs/${job.id}/events`, out_name: 'mock_generated.wav' };
      }
      if (p === '/api/jobs') return { jobs: [...JOBS.values()].filter((x) => !['done', 'error', 'cancelled'].includes(x.state)).map(snapshot) };
      if (seg[1] === 'jobs' && seg[2]) {
        const job = JOBS.get(seg[2]);
        if (!job) throw mkErr(404, 'not_found', 'No such job.');
        if (method === 'DELETE') {
          if (job.kind !== 'generate') throw mkErr(409, 'not_cancellable', 'That job cannot be cancelled.');
          job.cancelled = true;
          setTimeout(() => {
            job.state = 'cancelled'; job.seq += 1;
            for (const s of job.subs) s('cancelled', { seq: job.seq });
            job.subs.clear();
          }, 300);
          return { cancelling: true };
        }
        return snapshot(job);
      }
      throw mkErr(404, 'not_found', `mock: no route for ${method} ${p}`);
    },

    async upload(file, onProgress) {
      for (let i = 1; i <= 5; i++) { await delay(60); onProgress?.(i / 5); }
      return { path: `${OUT}\\uploads\\9f21c4e0_${file.name}`, url: URL.createObjectURL(file), name: file.name, size: file.size, uploaded_only: true };
    },

    stream(jobId, handlers) {
      const job = JOBS.get(jobId);
      let closed = false;
      const close = () => { closed = true; if (job) job.subs.delete(sub); };
      const sub = (name, data) => {
        if (closed) return;
        if (TERMINAL.has(name)) close();
        ({ result: handlers.result, error: handlers.failed, cancelled: handlers.cancelled,
           progress: handlers.progress, vram: handlers.vram, artifact: handlers.artifact,
           phase: handlers.phase, queued: handlers.queued }[name])?.(data);
      };
      if (!job) { setTimeout(() => handlers.failed?.({ code: 'not_found', message: 'No such job.' }), 10); return { close }; }
      job.subs.add(sub);
      setTimeout(() => !closed && handlers.hello?.(snapshot(job)), 10);
      return { close };
    },
  };
}

function snapshot(job) {
  const { subs, cancelled, ...rest } = job;
  return rest;
}

function mkErr(status, code, message) {
  const e = new Error(message);
  e.status = status; e.code = code; e.detail = null;
  e.__api = true;
  return e;
}
