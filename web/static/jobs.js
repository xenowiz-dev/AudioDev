/* jobs.js — the shared long-running-job view: live progress bar, ETA, the log,
   and Cancel. Used by Create (generate, transcribe) and Restore (upscale).

   The log rendering rule is spec §2.6, verbatim: a `frac` line REPLACES the
   previous one instead of adding a hundred near-identical rows, but each
   *stage* keeps its last line. `live` renders as the bar, never as a log row. */

import { el, pad, toast, stripMd, paintIcons } from './ui.js';
import * as api from './api.js';

export class JobView {
  /** @param {HTMLElement} host  @param {object} h  {onResult, onArtifact, title} */
  constructor(host, h = {}) {
    this.host = host;
    this.h = h;
    this.reset();
    this.render();
  }

  reset() {
    this.jobId = null;
    this.stream = null;
    this.log = [];
    this.live = null;          // {text, stage, frac}
    this.phase = '';
    this.t0 = Date.now() / 1000;
    this.state = 'idle';
    this.cancelling = false;
  }

  render() {
    this.barFill = el('div', { class: 'prog-fill' });
    this.lineEl = el('p', { class: 'prog-line', text: 'Idle.' });
    this.etaEl = el('p', { class: 'prog-eta' });
    this.logEl = el('pre', { class: 'log' });
    this.cancelBtn = el('button', {
      type: 'button', class: 'btn btn--ghost btn--wide', text: 'Cancel', hidden: true,
      onclick: () => this.cancel(),
    });
    // Shown only after a failure. A job that died mid-generation can leave the
    // model resident and the lane occupied, and until this existed there was
    // no way to get the GPU back without restarting the server.
    this.recoverBtn = el('button', {
      type: 'button', class: 'btn btn--ghost btn--wide', text: 'Free the GPU', hidden: true,
      onclick: () => document.dispatchEvent(new CustomEvent('gpu-recover')),
    });
    this.logBox = el('details', { class: 'disclose' },
      el('summary', {}, 'Log', el('span', { class: 'chevron', dataset: { icon: 'chevron' } })),
      el('div', { class: 'disclose-body' }, this.logEl));
    this.card = el('div', { class: 'card' },
      el('div', { class: 'prog' },
        this.lineEl,
        el('div', { class: 'prog-bar' }, this.barFill),
        this.etaEl),
      this.cancelBtn, this.recoverBtn);
    this.results = el('div', { class: 'side', style: 'padding:0' });
    this.host.replaceChildren(el('div', { class: 'side' }, this.card, this.logBox, this.results));
    paintIcons(this.host);
  }

  /** Attach to a job id: POST already returned, this is the stream.
   *  `onResult` overrides the constructor's handler for this job only — a
   *  transcribe writes into the lyrics box, a generate loads the player. */
  attach(jobId, { kind, onResult } = {}) {
    this.detach();
    this.reset();
    this.jobId = jobId;
    this.kind = kind;
    this.onResultOnce = onResult || null;
    this.state = 'queued';
    // Every kind is cancellable now: queued jobs come off the lane, running
    // ones have their process killed. Hiding this was the reason a stuck
    // upscale had no way out.
    this.cancelBtn.hidden = false;
    this.cancelBtn.disabled = false;
    this.cancelBtn.textContent = 'Cancel';
    this.recoverBtn.hidden = true;
    this.setLine('queued…', null);
    this.results.replaceChildren();

    this.stream = api.openJobStream(jobId, {
      hello: (j) => {
        // A fresh attach (or a replay gap) replaces the whole job state.
        this.log = (j.log || []).slice();
        this.live = j.live ? { text: pad(0) + j.live.msg, stage: j.live.stage, frac: j.live.frac } : null;
        this.t0 = (j.started || j.created || Date.now() / 1000);
        this.state = j.state;
        this.paintLog();
        if (j.state === 'done' && j.result) this.finish('result', j.result);
        else if (j.state === 'error' && j.error) this.finish('error', j.error);
        else if (j.state === 'cancelled') this.finish('cancelled', {});
      },
      queued: (d) => this.setLine(`queued — position ${d.position ?? 0}`, null),
      progress: (d) => this.onProgress(d),
      phase: (d) => { this.phase = d.label || d.phase || ''; this.pushLog(pad(this.t(d)) + `--- ${this.phase} ---`); },
      artifact: (d) => { this.h.onArtifact?.(d); this.addArtifact(d); },
      vram: (d) => this.h.onVram?.(d),
      result: (d) => this.finish('result', d),
      failed: (d) => this.finish('error', d),
      cancelled: (d) => this.finish('cancelled', d),
      dropped: () => this.onDropped(),
      reattached: () => this.onReattached(),
    });
  }

  /* ── surviving a phone that went to sleep ──────────────────────────────
     A dropped connection is not a dead job. The stream reconnects on its own,
     but a backgrounded tab can stay disconnected for minutes, so poll the job
     snapshot as well: it is the only way to notice that the job actually
     finished (or failed) while we were not listening. */
  onDropped() {
    this.setLine('connection lost — reattaching…', null);
    if (this.resyncTimer) return;
    this.resyncTimer = setInterval(() => this.resync(), 5000);
    this.resync();
  }

  onReattached() {
    this.stopResync();
    // Do not paint "reconnected" — `hello` replays the true state immediately
    // after and would overwrite it anyway.
  }

  stopResync() {
    if (this.resyncTimer) { clearInterval(this.resyncTimer); this.resyncTimer = null; }
  }

  async resync() {
    if (!this.jobId) return;
    try {
      const j = await api.getJob(this.jobId);
      if (!j || j.id !== this.jobId) return;
      if (j.state === 'done' && j.result) { this.stopResync(); this.finish('result', j.result); }
      else if (j.state === 'error') { this.stopResync(); this.finish('error', j.error || {}); }
      else if (j.state === 'cancelled') { this.stopResync(); this.finish('cancelled', {}); }
      else if (j.live) {
        // still alive: show real progress rather than a frozen "reattaching…"
        this.setLine(j.live.msg || 'running…', j.live.frac ?? null);
      }
    } catch { /* server unreachable; keep trying */ }
  }

  detach() {
    this.stopResync();
    this.stream?.close(); this.stream = null; this.onResultOnce = null;
  }

  t(d) { return typeof d.t === 'number' ? d.t : (Date.now() / 1000 - this.t0); }

  /* spec §2.6 — do not "improve" this */
  onProgress(d) {
    const t = this.t(d);
    const frac = (d.frac === undefined) ? null : d.frac;
    const stage = d.stage ?? null;
    if (frac === null || frac === undefined) {
      if (this.live) { this.log.push(this.live.text); this.live = null; }
      this.log.push(pad(t) + (d.msg || ''));
    } else {
      if (this.live && this.live.stage !== stage) this.log.push(this.live.text);
      this.live = { text: pad(t) + (d.msg || ''), stage, frac };
    }
    this.state = 'running';
    this.paintLog();
    this.setLine(d.msg || '', frac, t);
  }

  pushLog(line) { this.log.push(line); this.paintLog(); }

  paintLog() {
    if (this.log.length > 1000) this.log = this.log.slice(-1000);
    this.logEl.textContent = this.log.join('\n');
    this.logEl.scrollTop = this.logEl.scrollHeight;
  }

  setLine(msg, frac, t) {
    this.lineEl.textContent = (this.phase ? this.phase + ' — ' : '') + stripMd(msg);
    if (frac === null || frac === undefined) {
      this.barFill.style.width = this.state === 'running' ? '100%' : '0%';
      this.barFill.style.opacity = this.state === 'running' ? '.25' : '1';
      this.etaEl.textContent = '';
    } else {
      this.barFill.style.opacity = '1';
      this.barFill.style.width = (Math.max(0, Math.min(1, frac)) * 100).toFixed(1) + '%';
      const secs = t ?? (Date.now() / 1000 - this.t0);
      const eta = frac > 0.02 ? (secs / frac) * (1 - frac) : null;
      this.etaEl.textContent = `${Math.round(frac * 100)}%` +
        (eta ? ` · about ${eta < 90 ? Math.round(eta) + 's' : Math.round(eta / 60) + ' min'} left` : '');
    }
  }

  addArtifact(d) {
    if (d.kind === 'image') {
      this.results.append(el('div', { class: 'card speccard' },
        el('div', { class: 'specwrap' }, el('img', { src: d.url, alt: d.label || 'figure' }))));
    } else if (d.kind === 'audio' && this.h.onAudio) {
      this.h.onAudio(d);
    }
  }

  finish(kind, d) {
    const once = this.onResultOnce;      // detach() clears it
    this.detach();
    if (this.live) { this.log.push(this.live.text); this.live = null; }
    this.cancelBtn.hidden = true;
    this.recoverBtn.hidden = kind !== 'error';
    if (kind === 'result') {
      this.state = 'done';
      this.pushLog(pad(this.t(d)) + 'done');
      this.setLine('done', 1);
      (once || this.h.onResult)?.(d);
    } else if (kind === 'cancelled') {
      this.state = 'cancelled';
      this.setLine('cancelled', null);
      this.pushLog('cancelled');
      toast('Cancelled.');
      this.h.onEnd?.('cancelled');
    } else {
      this.state = 'error';
      this.setLine(d.message || 'failed', null);
      this.pushLog(`ERROR ${d.code || ''}: ${d.message || ''}`);
      toast(d.message || 'The job failed.', 'err');
      this.h.onEnd?.('error');
    }
    if (kind === 'result') this.h.onEnd?.('done');
  }

  async cancel() {
    if (!this.jobId || this.cancelling) return;
    this.cancelling = true;
    this.cancelBtn.disabled = true;
    this.cancelBtn.textContent = 'Cancelling…';
    try { await api.cancelJob(this.jobId); }
    catch (e) {
      this.cancelling = false;
      this.cancelBtn.disabled = false;
      this.cancelBtn.textContent = 'Cancel';
      toast(e.message || 'That job cannot be cancelled.', 'err');
    }
  }

  get busy() { return ['queued', 'starting', 'running'].includes(this.state); }
}
