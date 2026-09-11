"""Jobs, the single GPU lane, and the SSE fan-out.

Why this is its own module: everything else in the API is a thin call into a
portable module, but a job has to outlive the HTTP request that created it.
The phone locks its screen mid-generation, the socket dies, and the client
re-attaches later expecting to catch up -- so the job owns a replayable event
buffer and a set of subscribers, and the HTTP stream is a pure consumer of it.

Three invariants this file exists to hold:

  * ONE GPU job at a time. A single daemon thread drains a FIFO; `generate`,
    `upscale` and `transcribe` all go through it, so two jobs can never share
    the 10 GB card. Depth is capped so a phone tapping Generate five times gets
    a 409 rather than an hour of queued work.
  * Pushing an event NEVER blocks. `on_progress` runs on the supervisor's
    worker-reading thread; if a slow client could stall it, a slow phone would
    stall the GPU. Subscriber queues are bounded and an overflowing subscriber
    is dropped (it reconnects and gets a `hello` snapshot, losing nothing).
  * The log/live split (spec 2.6) is maintained HERE as well as on the client,
    because the `hello` snapshot has to reproduce the pane a client that missed
    the first half of a job would otherwise never see.
"""

import asyncio
import json
import queue
import threading
import time
import uuid
from collections import deque

TERMINAL_EVENTS = ("result", "error", "cancelled")
TERMINAL_STATES = ("done", "error", "cancelled")
GPU_KINDS = ("generate", "upscale", "transcribe")

EVENT_BUFFER = 2000        # replayable events per job (spec 2.7)
LOG_LINES = 1000           # flushed log lines in the snapshot (spec 2.2)
SUB_QUEUE_MAX = 256        # per-subscriber backlog before we drop it (spec 2.5)
QUEUE_DEPTH = 4            # GPU queue depth; a 5th create is 409 gpu_busy
RETAIN_SECONDS = 30 * 60   # terminal jobs stay attachable this long (spec 2.7)
PING_SECONDS = 15.0
VRAM_TICK_SECONDS = 5.0

# Set by the API layer to a zero-argument callable returning the /api/vram
# shape. Kept as a hook so this module needs no import of the supervisor.
VRAM_PROBE = None


class GpuBusy(RuntimeError):
    """The GPU queue is full."""


def _pad(t):
    return "[%5.0fs] " % (t or 0.0)


def sse(event, data, seq=None):
    """One SSE frame. `id:` is omitted for pings so it cannot move Last-Event-ID."""
    head = f"id: {seq}\n" if seq is not None else ""
    body = json.dumps(data, ensure_ascii=False, default=str)
    return f"{head}event: {event}\ndata: {body}\n\n"


class Subscriber:
    __slots__ = ("loop", "q", "dropped")

    def __init__(self, loop):
        self.loop = loop
        self.q = asyncio.Queue()
        self.dropped = False


class Job:
    def __init__(self, kind, request=None):
        self.id = "j_" + uuid.uuid4().hex[:6]
        self.kind = kind
        self.state = "queued"
        self.created = time.time()
        self.started = None
        self.ended = None
        self.queue_position = None
        self.seq = 0
        self.log = deque(maxlen=LOG_LINES)
        self.live = None
        self.artifacts = []
        self.result = None
        self.error = None
        self.vram = None
        self.request = request or {}
        self.cancel_requested = False
        self.events = deque(maxlen=EVENT_BUFFER)
        self.subs = set()
        self.lock = threading.RLock()
        self.extra = {}          # scratch space for the executor

    # -- events ------------------------------------------------------------
    def push(self, event, data=None):
        """Append an event, update derived state, fan out. Never blocks."""
        data = dict(data or {})
        with self.lock:
            self.seq += 1
            data["seq"] = self.seq
            if event == "progress":
                self._apply_progress(data)
            elif event == "vram":
                self.vram = {k: v for k, v in data.items() if k != "seq"}
            elif event == "artifact":
                self.artifacts.append({k: v for k, v in data.items()
                                       if k != "seq"})
            elif event in TERMINAL_EVENTS:
                self._flush_live()
            self.events.append((self.seq, event, data))
            subs = list(self.subs)
        item = (self.seq, event, data)
        for sub in subs:
            self._deliver(sub, item)
        return self.seq

    def _deliver(self, sub, item):
        if sub.dropped:
            return
        try:
            if sub.q.qsize() >= SUB_QUEUE_MAX:
                # A client too slow to keep up is forced to reconnect rather
                # than allowed to hold up the worker thread that produced this.
                sub.dropped = True
                item = None
            sub.loop.call_soon_threadsafe(sub.q.put_nowait, item)
        except RuntimeError:
            pass        # loop already closed (shutdown)

    def _apply_progress(self, data):
        text = _pad(data.get("t")) + (data.get("msg") or "")
        frac, stage = data.get("frac"), data.get("stage")
        if frac is None:
            self._flush_live()
            self.log.append(text)
        else:
            if self.live and self.live.get("stage") != stage:
                self.log.append(self.live["text"])
            self.live = {"text": text, "stage": stage, "frac": frac,
                         "msg": data.get("msg") or ""}

    def _flush_live(self):
        if self.live:
            self.log.append(self.live["text"])
            self.live = None

    # -- progress relay (spec 2.5) ----------------------------------------
    def emit(self, m):
        """on_progress bridge. `m` is a dict (supervisor) OR a str (CLI tools)."""
        if isinstance(m, str):
            m = {"msg": m}
        if self.state == "starting":
            self.set_state("running")
        base = self.started or self.created
        self.push("progress", {"msg": m.get("msg", ""),
                               "stage": m.get("stage"),
                               "frac": m.get("frac"),
                               "t": round(time.time() - base, 2)})

    def line(self, msg):
        """A plain, non-fractional log line."""
        self.emit({"msg": msg})

    def phase(self, phase, label):
        self.push("phase", {"phase": phase, "label": label})
        self.emit_vram()

    def artifact(self, kind, label, path, url):
        self.push("artifact", {"kind": kind, "label": label,
                               "path": path, "url": url})

    def emit_vram(self):
        if VRAM_PROBE is None:
            return
        try:
            v = VRAM_PROBE()
        except Exception:
            return
        self.push("vram", {"used_gb": v.get("used_gb"),
                           "total_gb": v.get("total_gb"),
                           "free_gb": v.get("free_gb"),
                           "loaded": v.get("loaded")})

    # -- state -------------------------------------------------------------
    def set_state(self, state):
        with self.lock:
            self.state = state
            if state in TERMINAL_STATES and self.ended is None:
                self.ended = time.time()

    @property
    def terminal(self):
        return self.state in TERMINAL_STATES

    def succeed(self, payload):
        with self.lock:
            if self.terminal:
                return
            self.result = dict(payload or {})
            self.set_state("done")
        self.push("result", payload or {})

    def fail(self, code, message):
        with self.lock:
            if self.terminal:
                return
            self.error = {"code": code, "message": str(message)[:2000]}
            self.set_state("error")
        self.push("error", {"code": code, "message": str(message)[:2000]})

    def cancel(self):
        with self.lock:
            if self.terminal:
                return
            self.set_state("cancelled")
        self.push("cancelled", {})

    # -- snapshot ----------------------------------------------------------
    def snapshot(self):
        with self.lock:
            return {
                "id": self.id,
                "kind": self.kind,
                "state": self.state,
                "created": self.created,
                "started": self.started,
                "ended": self.ended,
                "queue_position": self.queue_position,
                "seq": self.seq,
                "log": list(self.log),
                "live": dict(self.live) if self.live else None,
                "artifacts": [dict(a) for a in self.artifacts],
                "result": dict(self.result) if self.result else None,
                "error": dict(self.error) if self.error else None,
                "vram": dict(self.vram) if self.vram else None,
                "request": self.request,
            }

    # -- subscribers -------------------------------------------------------
    def attach(self, loop, last_event_id):
        """(subscriber, backlog, send_hello) -- atomic against push()."""
        sub = Subscriber(loop)
        with self.lock:
            self.subs.add(sub)
            buffered = list(self.events)
            floor = buffered[0][0] if buffered else self.seq + 1
            send_hello = True
            backlog = []
            last = None
            if last_event_id is not None:
                try:
                    last = int(str(last_event_id).strip())
                except (TypeError, ValueError):
                    last = None
            if last is not None:
                if last >= floor - 1:
                    # Everything the client missed is still buffered: replay it
                    # and skip the hello, exactly as spec 2.7 requires.
                    send_hello = False
                    backlog = [e for e in buffered if e[0] > last]
                else:
                    backlog = buffered
            hello_seq = self.seq
        return sub, backlog, send_hello, hello_seq

    def detach(self, sub):
        with self.lock:
            self.subs.discard(sub)


class Registry:
    """Every job, plus the one GPU lane they queue on."""

    def __init__(self):
        self.lock = threading.RLock()
        self.jobs = {}
        self.order = []
        self.pending = []
        self.running = None
        self.q = queue.Queue()
        self._threads = []
        self._stop = threading.Event()

    # -- lifecycle ---------------------------------------------------------
    def start(self):
        if self._threads:
            return
        for target in (self._drain, self._housekeeping):
            t = threading.Thread(target=target, daemon=True)
            t.start()
            self._threads.append(t)

    def stop(self):
        self._stop.set()
        self.q.put(None)

    # -- registry ----------------------------------------------------------
    def get(self, job_id):
        with self.lock:
            return self.jobs.get(job_id)

    def list(self, active_only=False):
        with self.lock:
            js = [self.jobs[i] for i in self.order if i in self.jobs]
        js.sort(key=lambda j: -j.created)
        if active_only:
            js = [j for j in js if not j.terminal]
        return js

    def register(self, job):
        with self.lock:
            self.jobs[job.id] = job
            self.order.append(job.id)

    def active_gpu(self):
        with self.lock:
            return self.running if (self.running and not self.running.terminal) \
                else None

    def gpu_idle(self):
        with self.lock:
            return self.running is None and not self.pending

    # -- the one GPU lane --------------------------------------------------
    def submit_gpu(self, job, fn):
        """Queue a GPU job. Raises GpuBusy when the lane is already 4 deep."""
        with self.lock:
            depth = len(self.pending) + (1 if self.running else 0)
            if depth >= QUEUE_DEPTH:
                raise GpuBusy(f"{depth} jobs already queued on the GPU.")
            self.register(job)
            self.pending.append(job)
            job.queue_position = len(self.pending) - 1 + (1 if self.running else 0)
        job.push("queued", {"position": job.queue_position})
        self.q.put((job, fn))
        return job

    def _renumber(self):
        with self.lock:
            pend = list(self.pending)
            base = 1 if self.running else 0
        for i, j in enumerate(pend):
            if j.queue_position != i + base:
                j.queue_position = i + base
                j.push("queued", {"position": j.queue_position})

    def _drain(self):
        while not self._stop.is_set():
            item = self.q.get()
            if item is None:
                return
            job, fn = item
            with self.lock:
                if job.terminal or job.state != "queued":
                    # Cancelled while queued: it already emitted `cancelled`.
                    if job in self.pending:
                        self.pending.remove(job)
                    continue
                self.pending.remove(job)
                self.running = job
                job.queue_position = None
                job.started = time.time()
                job.state = "starting"
            self._renumber()
            try:
                job.emit_vram()
                fn(job)
                if not job.terminal:
                    job.succeed(job.result or {})
            except BaseException as exc:                  # noqa: BLE001
                if job.cancel_requested and not job.terminal:
                    job.cancel()
                elif not job.terminal:
                    job.fail(_code_for(exc), f"{type(exc).__name__}: {exc}")
            finally:
                with self.lock:
                    self.running = None
                self._renumber()

    # -- the CPU side ------------------------------------------------------
    def submit_cpu(self, job, fn):
        """Run `fn` off the GPU lane, one thread per job.

        The lyric writer runs on CPU for a minute at a time. Putting that on
        the single GPU lane would mean writing a chorus blocks a render, for
        no reason at all -- nothing here touches the card. There is no queue
        because there is no scarce resource to serialise: a second writer is
        just a second thread.

        Everything else is deliberately identical to `_drain`: the same
        succeed / fail / cancel-conversion, so the SSE contract and the
        cancellation semantics of §8 do not fork.
        """
        self.register(job)

        def body():
            with self.lock:
                if job.terminal:
                    return
                job.started = time.time()
                job.queue_position = None
                job.state = "starting"
            try:
                fn(job)
                if not job.terminal:
                    job.succeed(job.result or {})
            except BaseException as exc:                   # noqa: BLE001
                if job.cancel_requested and not job.terminal:
                    job.cancel()
                elif not job.terminal:
                    job.fail(_code_for(exc), f"{type(exc).__name__}: {exc}")

        threading.Thread(target=body, daemon=True).start()
        return job

    def cancel(self, job):
        """Cancel a queued job in place. Returns True when it was queued."""
        with self.lock:
            if job.state == "queued" and not job.terminal:
                if job in self.pending:
                    self.pending.remove(job)
                job.cancel_requested = True
                job.queue_position = None
                job.cancel()
                return True
        return False

    # -- background --------------------------------------------------------
    def _housekeeping(self):
        last_vram = 0.0
        while not self._stop.wait(1.0):
            now = time.time()
            job = self.active_gpu()
            if job is not None and now - last_vram >= VRAM_TICK_SECONDS:
                last_vram = now
                job.emit_vram()
            if int(now) % 60 == 0:
                self.evict(now)

    def evict(self, now=None):
        now = now or time.time()
        with self.lock:
            dead = [i for i, j in self.jobs.items()
                    if j.terminal and j.ended and now - j.ended > RETAIN_SECONDS]
            for i in dead:
                self.jobs.pop(i, None)
                if i in self.order:
                    self.order.remove(i)
        return len(dead)


def _code_for(exc):
    name = type(exc).__name__
    if name == "WorkerDied":
        return "worker_died"
    if name in ("FileNotFoundError", "OSError"):
        return "io_error"
    return "internal"


async def event_stream(job, request, last_event_id):
    """The SSE body: hello / replay / live, with a 15 s keepalive."""
    loop = asyncio.get_running_loop()
    sub, backlog, send_hello, hello_seq = job.attach(loop, last_event_id)
    try:
        if send_hello:
            yield sse("hello", job.snapshot(), hello_seq)
        done = False
        for seq, event, data in backlog:
            yield sse(event, data, seq)
            if event in TERMINAL_EVENTS:
                done = True
        if done:
            return
        if job.terminal:
            # A phone that slept through a three-minute generation learns it
            # succeeded here: the hello above carried result/error, and this is
            # the terminal event that tells the client to stop listening.
            with job.lock:
                term = next((e for e in reversed(job.events)
                             if e[1] in TERMINAL_EVENTS), None)
            if term:
                yield sse(term[1], term[2], term[0])
            return
        while True:
            try:
                item = await asyncio.wait_for(sub.q.get(), timeout=PING_SECONDS)
            except asyncio.TimeoutError:
                if await request.is_disconnected():
                    return
                yield sse("ping", {"t": time.time()})
                continue
            if item is None:                 # overflowed: force a reconnect
                return
            seq, event, data = item
            yield sse(event, data, seq)
            if event in TERMINAL_EVENTS:
                return
    finally:
        job.detach(sub)
