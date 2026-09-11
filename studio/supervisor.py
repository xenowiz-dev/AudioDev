"""Run one music model at a time, each in its own venv, as a warm subprocess.

Why subprocesses at all: ACE-Step and MiniMax Music 3 cannot share a Python
environment (ACE-Step's diffusers fights the pinned dev commit MiniMax needs),
so no single process can import both. Workers also give the property that
matters most on a 10 GB card -- killing a process returns *all* of its VRAM,
with no reliance on a library releasing it politely.

Invariant: at most one worker is alive. Switching models kills the previous one.

Protocol is JSON, one object per line, over the worker's stdin/stdout:

    -> {"type": "generate", ...params}      supervisor to worker
    -> {"type": "shutdown"}
    <- {"type": "ready",    "model": ...}   worker to supervisor, after loading
    <- {"type": "progress", "msg": ...}
    <- {"type": "done",     "path": ..., "elapsed": ..., ...}
    <- {"type": "error",    "msg": ...}

Two things this file is careful about, both of which silently break otherwise:

  * stderr is drained by a background thread. tqdm and transformers write a lot
    there, and on Windows a full pipe buffer blocks the *worker*, which looks
    exactly like a hung model.
  * pipes are UTF-8. Windows defaults these to cp1252 and real lyrics are full
    of typographic quotes and em dashes.
"""

import json
import os
import queue
import subprocess
import sys
import threading
from collections import deque

# Root discovery: AUDIODEV_ROOT if the launcher set it, else relative to
# this file. Every path below derives from it, so the tree moves as one.
_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("AUDIODEV_ROOT") or os.path.normpath(os.path.join(_HERE, ".."))

STUDIO = os.path.dirname(os.path.abspath(__file__))

BACKENDS = {
    "minimax": dict(
        label="MiniMax Music 3",
        python=os.path.join(ROOT, "minimax", ".venv", "Scripts", "python.exe"),
        script=os.path.join(STUDIO, "workers", "minimax_worker.py"),
        vram_gb=8.0,
        note="44.1 kHz stereo, sung lyrics. 4-bit LLM. ~7x realtime.",
    ),
    "acestep": dict(
        label="ACE-Step 1.5",
        python=os.path.join(ROOT, "acestep", ".venv", "Scripts", "python.exe"),
        script=os.path.join(STUDIO, "workers", "acestep_worker.py"),
        vram_gb=6.0,
        note="48 kHz. Fast, and the only one here that can cover a track.",
    ),
    "yue2": dict(
        label="YuE2 (3B)",
        python=os.path.join(ROOT, "yue2", ".venv", "Scripts", "python.exe"),
        script=os.path.join(STUDIO, "workers", "yue2_worker.py"),
        # The documented floor: bf16, the 3B model plus the VAE plus the plan
        # and semantic caches. YuE2 does carry an fp8 mode (Ada/Blackwell
        # cards) and an AR-offload switch -- see yue2_worker.py -- so a 16 GB
        # card MAY work with YUE2_QUANT=fp8; lower the floor with
        # YUE2_VRAM_GB to try it. Nothing below 24 has been measured.
        vram_gb=float(os.environ.get("YUE2_VRAM_GB", "24") or 24),
        note="48 kHz stereo. Plans an editable score (melody, chords, "
             "structure, tempo) before singing. Needs a 24 GB card; "
             "weights are non-commercial (CC BY-NC 4.0).",
    ),
}


class WorkerDied(RuntimeError):
    pass


class Worker:
    def __init__(self, name, env=None):
        self.name = name
        self.cfg = BACKENDS[name]
        self.proc = None
        self.lora = None          # adapter this process has loaded, if any
        # The DiT this process loaded. Set from the env it was started with,
        # because the checkpoint is chosen at load and cannot be swapped after.
        self.variant = (env or {}).get("ACESTEP_VARIANT") or None
        self.err = deque(maxlen=500)
        self._err_thread = None
        # Load-time settings for THIS process. Weights are quantized while the
        # model loads, so these can only be changed by starting a new worker --
        # which is exactly why they live on the worker object and not on a
        # request. Empty for every caller that does not ask for anything.
        self.env_extra = {str(k): str(v) for k, v in (env or {}).items()}
        self.ready = {}

    @property
    def quality(self):
        """What the worker itself says it loaded, or None if it never said.

        Ground truth, not what we asked for: an older worker binary, or a
        backend with no quality settings at all (ACE-Step), reports nothing and
        this stays None.
        """
        q = self.ready.get("quality")
        return dict(q) if isinstance(q, dict) else None

    # -- lifecycle ---------------------------------------------------------
    def start(self, on_progress=None):
        env = dict(os.environ)
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUNBUFFERED"] = "1"
        env.update(self.env_extra)
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self.proc = subprocess.Popen(
            [self.cfg["python"], self.cfg["script"]],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, encoding="utf-8",
            errors="replace", bufsize=1, env=env, creationflags=flags)

        self._err_thread = threading.Thread(
            target=self._drain_err, daemon=True)
        self._err_thread.start()

        for msg in self._read_until(("ready",), on_progress):
            if msg.get("type") == "ready":
                self.ready = msg
        return self

    def _drain_err(self):
        for line in self.proc.stderr:
            self.err.append(line.rstrip())

    def stop(self):
        if self.proc is None:
            return
        try:
            self._send({"type": "shutdown"})
            self.proc.wait(timeout=8)
        except Exception:
            pass
        for finish in (self.proc.terminate, self.proc.kill):
            if self.proc.poll() is None:
                try:
                    finish()
                    self.proc.wait(timeout=5)
                except Exception:
                    pass
        self.proc = None

    @property
    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    # -- protocol ----------------------------------------------------------
    def _send(self, obj):
        self.proc.stdin.write(json.dumps(obj) + "\n")
        self.proc.stdin.flush()

    def _read_until(self, terminal, on_progress):
        """Yield protocol messages until one of `terminal` types arrives."""
        while True:
            line = self.proc.stdout.readline()
            if line == "":
                tail = "\n".join(list(self.err)[-25:])
                raise WorkerDied(
                    f"{self.name} worker exited unexpectedly.\n{tail}")
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                # A stray print that escaped the worker's stdout redirect.
                self.err.append("[stdout] " + line)
                continue
            if msg.get("type") == "progress" and on_progress:
                # The WHOLE message, not just the text: workers attach `stage`
                # and `frac` for the progress bar. Callers that only want text
                # read msg["msg"].
                on_progress(msg)
            if msg.get("type") == "error":
                raise WorkerDied(msg.get("msg", "unknown worker error"))
            yield msg
            if msg.get("type") in terminal:
                return

    def generate(self, params, on_progress=None):
        # Remember which adapter this process now has. ACE-Step can LOAD a
        # LoRA reliably but cannot remove one -- PEFT's delete_adapter raises
        # a bare KeyError for the adapter it was just asked about (measured) --
        # so the supervisor treats "a different LoRA" as a load-time change and
        # restarts, exactly as it does for quantization.
        if "lora_path" in params:
            self.lora = params.get("lora_path") or None
        self._send({"type": "generate", **params})
        result = None
        for msg in self._read_until(("done",), on_progress):
            if msg.get("type") == "done":
                result = msg
        return result


class Supervisor:
    """Keeps at most one worker alive; serialises every request."""

    def __init__(self):
        self.worker = None
        self.lock = threading.RLock()

    def current(self):
        return self.worker.name if self.worker and self.worker.alive else None

    def loaded_quality(self):
        """The load-time settings of the live worker, as it reported them."""
        return self.worker.quality if self.worker and self.worker.alive else None

    @staticmethod
    def _quality_matches(worker, want):
        """Does the live worker already carry the requested load-time settings?

        Compared on the requested keys ONLY, so a caller asking for
        {"llm", "rvq"} does not force a reload because some other field the
        worker reports (reserve, kv cache) differs -- those are per-request.
        """
        have = worker.quality
        if have is None:
            return False        # it cannot tell us; assume it is not what we want
        return all(have.get(k) == v for k, v in want.items())

    @staticmethod
    def _lora_matches(worker, lora):
        a = getattr(worker, "lora", None) or None
        b = lora or None
        if a == b:
            return True
        # A worker with nothing loaded can load anything; only an already
        # adapted process has to be replaced.
        return a is None

    def ensure(self, name, on_progress=None, env=None, quality=None,
               lora=None):
        """Return a live worker for `name`, starting or restarting as needed.

        `quality` is the LOAD-TIME settings the caller needs (e.g.
        {"llm": "4bit", "rvq": "bf16"}); a live worker that reports anything
        else is stopped and started again with `env`, because quantization
        happens while the weights load and cannot be changed afterwards.
        `quality=None` means "whatever is loaded is fine" -- that is the old
        behaviour, and the path every existing caller takes.
        """
        with self.lock:
            if self.worker and self.worker.alive and self.worker.name == name:
                want_variant = (env or {}).get("ACESTEP_VARIANT")
                if want_variant and (self.worker.variant or "") != want_variant:
                    if on_progress:
                        on_progress({"msg": f"loading {want_variant} "
                                            f"(was {self.worker.variant or 'default'})"
                                            f" - the checkpoint is chosen at "
                                            f"load, so this costs ~40 s"})
                    self.worker.stop()
                    self.worker = None
                elif not self._lora_matches(self.worker, lora):
                    if on_progress:
                        have = getattr(self.worker, "lora", None)
                        on_progress({"msg": "reloading "
                                     f"{BACKENDS[name]['label']} to change the "
                                     f"LoRA ({os.path.basename(have or '')} -> "
                                     f"{os.path.basename(lora or 'none')}) — "
                                     f"the adapter cannot be swapped in place"})
                    self.worker.stop()
                    self.worker = None
                elif not quality or self._quality_matches(self.worker, quality):
                    return self.worker
                else:
                    # `else`, not a fall-through: the LoRA branch above has
                    # already cleared self.worker, and reading .quality off it
                    # here is an AttributeError -- which is exactly what the
                    # switching test hit on its fourth render.
                    if on_progress:
                        have = self.worker.quality or {}
                        diff = ", ".join(
                            f"{k} {have.get(k, '?')} -> {v}"
                            for k, v in quality.items() if have.get(k) != v)
                        on_progress({"msg": f"reloading {BACKENDS[name]['label']} "
                                            f"({diff}) — quantization happens at "
                                            f"load, so this costs ~20 s"})
                    self.worker.stop()
                    self.worker = None
            if self.worker:
                if on_progress:
                    on_progress({"msg": f"unloading {self.worker.name} "
                                        f"to free VRAM"})
                self.worker.stop()
                self.worker = None
            if on_progress:
                on_progress({"msg": f"starting {BACKENDS[name]['label']} "
                                    f"(first load takes ~20 s)"})
            self.worker = Worker(name, env=env).start(on_progress)
            return self.worker

    def generate(self, name, params, on_progress=None, env=None, quality=None):
        with self.lock:
            w = self.ensure(name, on_progress, env=env, quality=quality,
                            lora=params.get("lora_path"))
            return w.generate(params, on_progress)

    def unload(self):
        with self.lock:
            if self.worker:
                self.worker.stop()
                self.worker = None


def gpu_memory():
    """(used_gb, total_gb) from nvidia-smi -- no torch needed in the UI venv."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=6,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        used, total = out.stdout.strip().splitlines()[0].split(",")
        return int(used) / 1024, int(total) / 1024
    except Exception:
        return None, None
