/* api.js — the only place that talks HTTP. Spec §0 (error envelope), §2 (SSE).

   Same-origin, no CORS, no absolute URLs with a host: this is reached over a
   `tailscale serve` proxy and must not assume its own origin.

   `?mock=1` swaps the transport for canned JSON from mock.js so the whole UI
   is developable and screenshottable with no server. */

export class ApiError extends Error {
  constructor(status, code, message, detail) {
    super(message || code || `HTTP ${status}`);
    this.status = status; this.code = code || 'internal'; this.detail = detail ?? null;
  }
}

export const MOCK = new URLSearchParams(location.search).get('mock') === '1';

let T = null;   // transport

export async function init() {
  if (T) return T;
  T = MOCK ? (await import('./mock.js')).transport() : realTransport();
  return T;
}

/* ── real transport ────────────────────────────────────────────────────── */
function realTransport() {
  return {
    async json(method, url, body, opts = {}) {
      const init = { method, headers: {}, signal: opts.signal };
      if (body !== undefined && body !== null) {
        init.headers['Content-Type'] = 'application/json; charset=utf-8';
        init.body = JSON.stringify(body);
      }
      const r = await fetch(url, init);
      let data = null;
      try { data = await r.json(); } catch { /* empty body */ }
      if (!r.ok) {
        const e = (data && data.error) || {};
        throw new ApiError(r.status, e.code, e.message || r.statusText, e.detail);
      }
      return data;
    },
    async upload(file, onProgress) {
      const fd = new FormData();
      fd.append('file', file, file.name);
      // XHR, not fetch: fetch has no upload progress and a 200 MB file over a
      // phone radio needs one.
      return new Promise((res, rej) => {
        const x = new XMLHttpRequest();
        x.open('POST', '/api/upload');
        x.upload.addEventListener('progress', (e) => {
          if (e.lengthComputable && onProgress) onProgress(e.loaded / e.total);
        });
        x.addEventListener('load', () => {
          let d = null; try { d = JSON.parse(x.responseText); } catch {}
          if (x.status >= 200 && x.status < 300) res(d);
          else { const e = (d && d.error) || {}; rej(new ApiError(x.status, e.code, e.message || x.statusText, e.detail)); }
        });
        x.addEventListener('error', () => rej(new ApiError(0, 'internal', 'Upload failed.')));
        x.send(fd);
      });
    },
    stream(jobId, handlers) {
      const es = new EventSource(`/api/jobs/${encodeURIComponent(jobId)}/events`);
      let closed = false;
      const close = () => { if (!closed) { closed = true; es.close(); } };
      const on = (name, fn) => es.addEventListener(name, (ev) => {
        let d = {}; try { d = JSON.parse(ev.data); } catch {}
        fn(d);
      });
      on('hello',     (d) => handlers.hello?.(d));
      on('queued',    (d) => handlers.queued?.(d));
      on('progress',  (d) => handlers.progress?.(d));
      on('vram',      (d) => handlers.vram?.(d));
      on('artifact',  (d) => handlers.artifact?.(d));
      on('phase',     (d) => handlers.phase?.(d));
      on('ping',      () => {});
      // Terminal events MUST close the stream — EventSource otherwise
      // reconnects forever to a finished job.
      on('result',    (d) => { close(); handlers.result?.(d); });
      on('cancelled', (d) => { close(); handlers.cancelled?.(d); });
      // The server also sends `failed`; accept both spellings so renaming the
      // server event does not need a client deploy in lockstep.
      on('failed',    (d) => { close(); handlers.failed?.(d); });

      // ONE listener for the name 'error', because two things arrive under it:
      //
      //   * the server's `event: error`  -> the JOB failed
      //   * EventSource's own DOM error  -> the CONNECTION dropped
      //
      // They are indistinguishable by name, and treating the second as the
      // first is what made a phone locking its screen look like a failed
      // generation: the handler reported failure AND closed the stream, so it
      // never reconnected to a job that was still happily running.
      //
      // Only a real server event carries data. A transport error has none.
      es.addEventListener('error', (ev) => {
        if (typeof ev.data === 'string') {
          let d = {}; try { d = JSON.parse(ev.data); } catch {}
          close();
          handlers.failed?.(d);
          return;
        }
        if (closed) return;             // already finished; nothing to recover
        // Do NOT close: EventSource retries on its own, and because the server
        // stamps every event with `id:`, the browser sends Last-Event-ID and
        // the stream resumes rather than restarting.
        handlers.dropped?.(es.readyState);
      });
      es.addEventListener('open', () => handlers.reattached?.());
      return { close };
    },
  };
}

/* ── request helpers ───────────────────────────────────────────────────── */
const j = (m, u, b, o) => T.json(m, u, b, o);
const eid = encodeURIComponent;
const qs = (o) => {
  const p = new URLSearchParams();
  for (const [k, v] of Object.entries(o)) if (v !== undefined && v !== null && v !== '') p.set(k, v);
  const s = p.toString();
  return s ? '?' + s : '';
};

/** Build a media URL for a path the server did not hand us one for. */
export const mediaUrl = (path) => '/api/media?p=' + encodeURIComponent(path);

/* config + status */
export const getConfig = () => j('GET', '/api/config');
export const getHealth = () => j('GET', '/api/health');
export const getVram   = () => j('GET', '/api/vram');
/* The quality params are OPTIONAL on purpose: a server that predates the
   quality contract must still answer, and `qs()` drops empty values. */
export const getBudget = (model, duration, q, opts) =>
  j('GET', `/api/budget${qs({ model, duration, llm: q?.llm, rvq: q?.rvq, kv: q?.kv })}`, null, opts);
export const getGpuState = (opts) => j('GET', '/api/gpu/state', null, opts);
/* `force` clears the queue and kills whatever is running. It is the recovery
   path for a wedged job — without it a stuck worker holds the GPU forever. */
export const unloadGpu = (force = false) => j('POST', '/api/gpu/unload', { force: !!force });

/* jobs */
export const listJobs  = (active = true) => j('GET', `/api/jobs${active ? '?active=1' : ''}`);
export const getJob    = (id) => j('GET', `/api/jobs/${eid(id)}`);
export const cancelJob = (id) => j('DELETE', `/api/jobs/${eid(id)}`);
export const openJobStream = (id, handlers) => T.stream(id, handlers);

/* long-running creates */
export const postGenerate   = (body) => j('POST', '/api/generate', body);
export const postUpscale    = (body) => j('POST', '/api/upscale', body);
export const postTranscribe = (body) => j('POST', '/api/lyrics/transcribe', body);
/* The writer is a job like the rest, but it runs on the CPU lane — it does not
   queue behind a render, and a render does not queue behind it. */
export const postWrite     = (body) => j('POST', '/api/write', body);

/* slow-sync, no GPU */
export const probeLyrics = (body, opts) => j('POST', '/api/lyrics/probe', body, opts);
export const inspect     = (body, opts) => j('POST', '/api/restore/inspect', body, opts);
export const detectCliff = (body, opts) => j('POST', '/api/restore/cliff', body, opts);

/* library */
export const getLibrary   = (params = {}, opts) => j('GET', `/api/library${qs(params)}`, null, opts);
export const getTrack     = (id) => j('GET', `/api/library/${eid(id)}`);
export const patchTrack   = (id, body) => j('PATCH', `/api/library/${eid(id)}`, body);
export const retitle      = (id, use_llm) => j('POST', `/api/library/${eid(id)}/retitle`, { use_llm });
export const setFavorite  = (id, on) => j('PUT', `/api/library/${eid(id)}/favorite`, { on });
export const setRating    = (id, rating) => j('PUT', `/api/library/${eid(id)}/rating`, { rating });
/* PATCH takes only title/style (spec 8.4). Played state has its own endpoint,
   which is what actually writes played/plays/last_played into the sidecar. */
export const markPlayed   = (id, plays) => j('POST', `/api/library/${eid(id)}/played`, { plays });
export const trashTrack   = (id) => j('POST', `/api/library/${eid(id)}/trash`, {});
export const getReuse     = (id) => j('GET', `/api/library/${eid(id)}/reuse`);
export const getCoverSrc  = (id) => j('GET', `/api/library/${eid(id)}/cover-source`);
export const kickCovers   = (force = false, max = 60) => j('POST', '/api/library/covers', { force, max });

/* workspaces — the project a generation is made FOR (see REQUIREMENTS §16) */
export const getProfile         = () => j('GET', '/api/profile');
export const setProfile         = (artist) => j('PUT', '/api/profile', { artist });
export const getWorkspaces      = () => j('GET', '/api/workspaces');
export const createWorkspace    = (name) => j('POST', '/api/workspaces', { name });
export const setActiveWorkspace = (name) => j('PUT', '/api/workspaces/active', { name });
export const deleteWorkspace    = (name) => j('DELETE', `/api/workspaces/${eid(name)}`);

/* playlists + trash */
export const getPlaylists    = () => j('GET', '/api/playlists');
export const createPlaylist  = (name) => j('POST', '/api/playlists', { name });
export const deletePlaylist  = (name) => j('DELETE', `/api/playlists/${eid(name)}`);
export const getPlaylist     = (name) => j('GET', `/api/playlists/${eid(name)}`);
export const addToPlaylist   = (name, id) => j('POST', `/api/playlists/${eid(name)}/tracks`, { id });
export const removeFromList  = (name, id) => j('DELETE', `/api/playlists/${eid(name)}/tracks/${eid(id)}`);
export const reorderPlaylist = (name, id, delta) => j('POST', `/api/playlists/${eid(name)}/reorder`, { id, delta });
export const getTrash        = () => j('GET', '/api/trash');
export const restoreTrashed  = (name) => j('POST', `/api/trash/${eid(name)}/restore`, {});

/* files */
export const upload = (file, onProgress) => T.upload(file, onProgress);
export const fsStat = (path) => j('GET', `/api/fs/stat${qs({ p: path })}`);
