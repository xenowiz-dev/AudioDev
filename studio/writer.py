"""Studio-side wrapper for songwriter.py -- the LLM that writes lyrics/styles.

The studio venv has no torch, so the model runs as a subprocess under the
MiniMax venv's python. Same split as llm_title.py, and for the same reason,
with two differences that matter here:

  * this streams. A lyric is 300-600 tokens on CPU, which is a minute, not the
    3 s a title takes -- so stderr is drained line by line into `on_progress`
    and the caller can show a live bar instead of a spinner.
  * `on_proc` hands the live Popen back, so a write is cancellable like every
    other job (REQUIREMENTS §8).

Long text goes through temp FILES, never argv: lyrics carry newlines and
typographic quotes, and argv on Windows mangles both.
"""

import json
import os
import subprocess
import sys
import tempfile
import threading

# Root discovery: AUDIODEV_ROOT if the launcher set it, else relative to
# this file. Every path below derives from it, so the tree moves as one.
_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("AUDIODEV_ROOT") or os.path.normpath(os.path.join(_HERE, ".."))

MINIMAX_PY = os.path.join(ROOT, "minimax", ".venv", "Scripts", "python.exe")
SONGWRITER = os.path.join(ROOT, "songwriter.py")
HF_HOME = os.path.join(ROOT, "minimax", "hf")
NOWIN = getattr(subprocess, "CREATE_NO_WINDOW", 0)

TARGETS = ("lyrics", "style")
MODES = ("generate", "extend", "edit")

MODE_INFO = {
    "generate": "Writes the whole thing from your brief. Anything already in "
                "the box is ignored.",
    "extend":   "Continues from what is already there, and does not repeat it. "
                "Good for adding a bridge and a final chorus.",
    "edit":     "Changes only what you ask for and returns the rest word for "
                "word. The brief is the instruction: “make the chorus angrier”.",
}


def is_ready():
    """True when the weights are on disk. Imports nothing heavy."""
    sys.path.insert(0, os.path.dirname(SONGWRITER))
    try:
        import songwriter
        return songwriter.is_ready()
    except Exception:
        return False


def model_id():
    sys.path.insert(0, os.path.dirname(SONGWRITER))
    try:
        import songwriter
        return songwriter.MODEL_ID
    except Exception:
        return "Qwen/Qwen2.5-1.5B-Instruct"


VRAM_NEEDED_GB = 4.0    # 3.1 GB of fp16 weights, plus room for the KV cache


def write(target, mode, brief="", existing="", model="minimax", seed=None,
          temperature=0.9, timeout=600, device="cpu",
          on_progress=None, on_proc=None):
    """Run one write. Returns songwriter's result dict.

    `device` is chosen by the caller, which is the only place that knows
    whether the GPU lane is free. Measured on this box, 1.5B fp32/8 threads:
    CPU 2-4 tok/s, CUDA fp16 20 tok/s -- a full lyric is ~3 minutes against
    ~25 seconds. Worth taking the card when it is genuinely idle, never worth
    waiting for it.

    Raises RuntimeError with the child's stderr tail when it fails, so the
    job's error message says something useful rather than "exit 1".
    """
    if target not in TARGETS:
        raise ValueError(f"unknown target {target!r}")
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}")
    if not os.path.exists(MINIMAX_PY):
        raise RuntimeError(f"{MINIMAX_PY} is missing — the writer needs the "
                           f"MiniMax venv's torch.")

    cmd = [MINIMAX_PY, SONGWRITER, "--json", "--target", target,
           "--mode", mode, "--model", model, "--device", device,
           "--temperature", str(temperature),
           # The child stops a little early so it still gets to print JSON;
           # the parent's timeout is the hard wall behind that.
           "--timeout", str(max(10.0, timeout - 10.0))]
    if seed is not None:
        cmd += ["--seed", str(int(seed))]

    tmp, p = [], None
    try:
        for flag, text in (("--brief-file", brief),
                           ("--existing-file", existing)):
            if (text or "").strip():
                fd, path = tempfile.mkstemp(suffix=".txt", text=True)
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    fh.write(text)
                tmp.append(path)
                cmd += [flag, path]

        env = dict(os.environ)
        env["PYTHONIOENCODING"] = "utf-8"
        env["HF_HOME"] = HF_HOME
        if device == "cpu":
            env["CUDA_VISIBLE_DEVICES"] = "-1"
        else:
            env.pop("CUDA_VISIBLE_DEVICES", None)

        p = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, text=True,
                             encoding="utf-8", errors="replace", bufsize=1,
                             env=env, creationflags=NOWIN)
        if on_proc:
            on_proc(p)

        # stderr carries progress, stdout the single JSON line, so they are
        # read separately rather than merged.
        err = []

        def drain():
            for line in p.stderr:
                line = line.rstrip()
                if line:
                    err.append(line)
                    if on_progress:
                        on_progress(line[:160])

        t = threading.Thread(target=drain, daemon=True)
        t.start()
        # This blocks until the child closes stdout. There is no timeout on it
        # on purpose: songwriter.py enforces its own deadline between tokens,
        # and if the child wedges before that, cancelling the job kills the
        # process (§8) which closes the pipe and unblocks this read. A timeout
        # here would only add a second way to leave an orphan running.
        out = p.stdout.read()
        p.wait(timeout=timeout)
        t.join(timeout=5)
    finally:
        for path in tmp:
            try:
                os.unlink(path)
            except OSError:
                pass

    if p is None or p.returncode != 0:
        raise RuntimeError("\n".join(err[-8:]) or f"writer exited {p.returncode}")

    # Last JSON-looking line, not all of stdout: a library that prints before
    # we can redirect it should not break parsing.
    for line in reversed((out or "").splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    raise RuntimeError("the writer produced no result:\n"
                       + "\n".join(err[-8:]))
