"""Optional LLM song titles, for the tracks the heuristic cannot name.

`titles.derive()` stays the default: it reads a name out of the record, is
instant, needs no model, and is right whenever the lyric holds a usable line.
It has one blind spot -- an instrumental described only by a style prompt,
where the honest heuristic answer is to read the genre back ("Indie Folk
Guitar"). This module asks a small local instruct model for an image instead
("Jangly Jamboree"), and returns [] the moment anything looks off, so the
caller simply keeps its existing fallback.

The model is measured, not assumed (see the benchmark): Qwen2.5-1.5B-Instruct
is the smallest candidate that stops parroting the genre. Every sub-1B model
tried hands back the words it was given, which is what titles.py already does
for free and in 0 ms. The prompt below -- two worked examples carried as real
conversation turns -- is load-bearing, not decoration: zero-shot, this same
model answers '"Paycheck to Paycheck"', quotes included.

Two environments, one file:

  * The studio venv has no torch. It may `import llm_title` for `MODEL_ID` and
    `is_ready()` (both pure stdlib -- nothing heavy is imported at module
    level) and then call `suggest()`, which notices torch is missing and
    re-enters this same file as a subprocess under the MiniMax venv's python.
  * The MiniMax venv has torch and runs the model in-process, cached in module
    state, so the ~6 s load is paid once per process rather than per title.

CPU ONLY. The 3080 is shared with music generation and a titler must never be
the reason a render OOMs. `--device cuda` exists but is never the default, and
the CPU path checks where the weights actually landed rather than trusting the
environment variable that was supposed to hide the card.

Determinism is the module's half of titles.py's contract: the seed is a CRC of
the input text, so the same lyric always yields the same title, while different
tracks still get different ones. A constant seed would name every indie-pop
track "Jangle Jamboree".

Titling never downloads. `is_ready()` is the gate and an uncached model is an
immediate no, because the alternative measured badly both ways: 3 GB cannot
arrive inside a caller's timeout, and a dead endpoint took 82 s to give up --
which is a UI hang, not a fallback. A fresh box runs `--fetch` once.

Cost here, fp32 on 8 threads: ~9 s from cold process to first title, ~3 s per
title after, ~6.5 GB RSS steady state. Budget for the subprocess path
accordingly -- it is a cold process every time.
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import unicodedata
import zlib

# Root discovery: AUDIODEV_ROOT if the launcher set it, else relative to
# this file. Every path below derives from it, so the tree moves as one.
_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("AUDIODEV_ROOT") or os.path.normpath(os.path.join(_HERE, ".."))

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

HERE = os.path.dirname(os.path.abspath(__file__))
MINIMAX_PY = os.path.join(ROOT, "minimax", ".venv", "Scripts", "python.exe")
NOWIN = getattr(subprocess, "CREATE_NO_WINDOW", 0)

# HF_HOME is not set machine-wide on this box, so a process that forgets it
# downloads a second 3 GB copy into C:\Users\Kevin\.cache. setdefault, not [],
# so a caller that has already pointed at another cache keeps it.
os.environ.setdefault("HF_HOME", os.path.join(ROOT, "minimax", "hf"))
# huggingface_hub probes for symlink support, caches "yes", then os.symlink
# fails with WinError 1314 -- which is not a PermissionError, so its own
# fallback-to-copy path never fires and the download dies half way. Only
# `--fetch` downloads anything, but that is the run it would ruin.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Same-directory import, the way workers/minimax_worker.py imports minimax's
# gen.py: the casing, the 56-char cap and the genre list are titles.py's rules,
# and a private copy here would drift out of sync with the module whose output
# this one is meant to be interchangeable with.
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import titles  # noqa: E402


# --- the prompt (frozen; copied verbatim from the benchmark harness) --------

SYSTEM = (
    "You are a music editor who names songs for a record sleeve. "
    "Reply with the title and nothing else: 2 to 6 words, Title Case, "
    "no quotation marks, no punctuation at the end, no explanation, "
    "and never the word Title."
)

USER_LYRICS = (
    "Name this song. Take a vivid, concrete phrase from the lyric -- do not "
    "summarise it and do not name the genre.\n\nLyrics:\n{text}"
)

USER_STYLE = (
    "Name a song that sounds like this. Invent an evocative, concrete image "
    "-- do not repeat the genre, key, or bpm words back to me.\n\n"
    "Style:\n{text}"
)

# One example per mode, on material unrelated to anything the studio renders.
# They demonstrate the three things the instruction alone did not buy at this
# size: no quotes, no artist credit, and an image instead of the genre read
# back. Removing them measurably breaks the output format.
SHOTS = [
    (USER_LYRICS.format(
        text="The kitchen radio hums all night / I count the tiles until "
             "it's light / Nobody calls, nobody writes"),
     "Counting Tiles Until Light"),
    (USER_STYLE.format(
        text="downtempo trip hop, dusty vinyl crackle, muted trumpet, "
             "78 bpm"),
     "Dust on the Needle"),
]

# Sampled rather than greedy on purpose: greedy is dtype-sensitive (the same
# blues prompt gave "Blues in the Jungle" at fp32 and "Soulful Echoes" at
# bf16), while these sampled titles were identical across both. 16 tokens is
# ample -- a title lands in 5-8 and stops on EOS.
GEN = dict(max_new_tokens=16, do_sample=True,
           temperature=0.7, top_p=0.8, top_k=20)

# Model state, kept warm for the life of the process. Loading costs ~6 s.
_TOK = None
_MODEL = None
_LOADED_DEVICE = None


def _warn(msg):
    """Progress and diagnostics. Never stdout -- the CLI owns that for JSON."""
    print(f"[llm_title] {msg}", file=sys.stderr, flush=True)


# --- cache probing (pure stdlib: callable from the studio venv) -------------

def _repo_dir():
    root = os.environ.get("HF_HUB_CACHE") or os.path.join(
        os.environ.get("HF_HOME", ""), "hub")
    return os.path.join(root, "models--" + MODEL_ID.replace("/", "--"))


def is_ready():
    """True if the weights are already on disk, so a call will not hit the net.

    Deliberately a filesystem walk and not `huggingface_hub`: the studio venv
    has neither hub nor torch, and this is what the UI checks before offering
    the feature at all. os.path.exists follows symlinks, so a snapshot of
    dangling links (the WinError 1314 failure mode) correctly reads as absent.
    """
    for snap in glob.glob(os.path.join(_repo_dir(), "snapshots", "*")):
        have = set(os.listdir(snap)) if os.path.isdir(snap) else set()
        if not any(f.endswith(".safetensors") for f in have):
            continue
        if "config.json" in have and ("tokenizer.json" in have
                                      or "tokenizer_config.json" in have):
            return True
    return False


# --- loading ---------------------------------------------------------------

def _load(device="cpu", fetch=False):
    """Import torch, load the model once, keep it in module state.

    `fetch` is the one-time setup path (`--fetch`) and the only thing in this
    file allowed to touch the network. Titling never downloads: 3 GB cannot
    arrive inside a caller's timeout, and a dead endpoint took 82 s to admit
    defeat here -- measured, with HF_HUB_ETAG_TIMEOUT already lowered.
    """
    global _TOK, _MODEL, _LOADED_DEVICE
    if _MODEL is not None and _LOADED_DEVICE == device:
        return _TOK, _MODEL

    if device == "cpu" and "torch" not in sys.modules:
        # "-1", not "": an empty CUDA_VISIBLE_DEVICES is treated as unset on
        # Windows and torch still enumerates the 3080. Only before torch is
        # first imported, and only when we are the CPU path -- clobbering this
        # inside a process that already holds the card would be worse than the
        # problem it solves.
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    t0 = time.perf_counter()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # No hub revision check at all on the titling path: it is faster, it is
    # the only way an offline box starts, and it means a flaky network cannot
    # turn naming a track into a stall.
    kw = dict(local_files_only=not fetch)

    tok = AutoTokenizer.from_pretrained(MODEL_ID, **kw)
    # `dtype`, not `torch_dtype`: renamed in transformers 5.x, and the old name
    # warns on every load. fp32 because bf16 is *slower* on this CPU (no
    # AVX512-BF16); bf16 is only worth it if 6.5 GB of RSS is the problem.
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float32, device_map=device, **kw)
    model.eval()

    if device == "cpu":
        # Checked, not assumed: CUDA_VISIBLE_DEVICES has already been observed
        # to be ignored on this platform, and a titler quietly taking 3 GB of
        # VRAM would surface as a failed music render. `if`, not `assert` --
        # assert vanishes under -O, and this guard has to outlive that.
        stray = next((n for n, p in model.named_parameters()
                      if p.device.type != "cpu"), None)
        if stray is not None:
            raise RuntimeError(f"{stray} landed on GPU despite device_map=cpu")

    _TOK, _MODEL, _LOADED_DEVICE = tok, model, device
    _warn(f"loaded {MODEL_ID} ({device}) in {time.perf_counter() - t0:.1f}s")
    return tok, model


# --- generation ------------------------------------------------------------

def _messages(kind, text):
    tmpl = USER_LYRICS if kind == "lyrics" else USER_STYLE
    msgs = [{"role": "system", "content": SYSTEM}]
    for shot_user, shot_title in SHOTS:
        msgs.append({"role": "user", "content": shot_user})
        msgs.append({"role": "assistant", "content": shot_title})
    msgs.append({"role": "user", "content": tmpl.format(text=text)})
    return msgs


def _generate(tok, model, kind, text, seed, deadline):
    """One sampled title. Returns (raw_text, stopped_on_eos)."""
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList

    class Deadline(StoppingCriteria):
        """The only place generation can be interrupted is between tokens."""

        def __call__(self, input_ids, scores, **kw):
            done = time.monotonic() >= deadline
            return torch.full((input_ids.shape[0],), done,
                              dtype=torch.bool, device=input_ids.device)

    prompt = tok.apply_chat_template(_messages(kind, text), tokenize=False,
                                     add_generation_prompt=True)
    ids = tok(prompt, return_tensors="pt").to(model.device)
    torch.manual_seed(seed)
    with torch.inference_mode():
        out = model.generate(
            **ids, **GEN,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
            stopping_criteria=StoppingCriteriaList([Deadline()]))
    new = out[0][ids["input_ids"].shape[1]:]
    eos = new.numel() and int(new[-1]) in (
        tok.eos_token_id, tok.pad_token_id)
    return tok.decode(new, skip_special_tokens=True), bool(eos)


# --- post-processing -------------------------------------------------------

QUOTES = "\"'`\u2018\u2019\u201c\u201d\u00ab\u00bb\u201e\u201f\u2039\u203a"
EDGE = " \t" + QUOTES + ".,;:!-\u2013\u2014/|&*_#~"

# "Title:", "Song name:", "Here's one:" -- a meta preamble the model bolted on.
# The word list is the guard: it keeps a real title that happens to contain a
# colon ("Requiem: Part Two") from being cut in half.
META = re.compile(r"\b(title|titles|name|names|song|answer|suggestion|option|"
                  r"here|sure|okay)\b", re.I)

# What a small model says instead of a title when it decides not to play.
REFUSAL = re.compile(
    r"\b(sorry|i cannot|i can'?t|cannot|unable|as an ai|i'?m an ai|"
    r"please provide|i do not have|there (is|are) no)\b", re.I)


def _clean(raw, finished=True):
    """One model completion -> a usable title, or None if it is junk.

    Everything here has been seen from a model of this size: quotes around the
    answer, a "Title:" preamble, markdown bullets, an emoji, a trailing full
    stop, and several lines where one was asked for.
    """
    for line in str(raw or "").splitlines():
        s = line.strip()
        if not s:
            continue

        # Markdown scaffolding: "**", "### ", "- ", "1. ".
        s = re.sub(r"^[\s>#*_`~-]+", "", s)
        s = re.sub(r"^\d+[.)]\s+", "", s)
        s = s.replace("**", "").replace("`", "")

        # A preamble ending in a colon, but only if it reads like one.
        head, sep, tail = s.partition(":")
        if (sep and tail.strip() and len(head.split()) <= 6
                and META.search(head)):
            s = tail

        # Emoji (So/Sk) and zero-width formatting characters (C*) survive
        # str.strip and would end up in a filename via titles.slugify.
        s = "".join(c for c in s
                    if not unicodedata.category(c).startswith("C")
                    and unicodedata.category(c) not in ("So", "Sk"))

        # Repeated because the model nests them: '"**Foo**."'
        for _ in range(3):
            s = " ".join(s.split()).strip(EDGE)

        if not s or not any(c.isalpha() for c in s):
            continue
        # The instruction says never the word Title, so anything still holding
        # it is the model talking about the task ("Untitled Track" included).
        if "title" in s.casefold() or REFUSAL.search(s):
            continue
        words = re.findall(r"[A-Za-z][A-Za-z']*", s.lower())
        # Genre echo: the exact failure this module exists to avoid. Whole
        # title, not just one word -- "Blues Rock" is as useless as "Blues".
        if words and all(w in titles.GENRES for w in words):
            continue
        # A completion that never reached EOS was cut off mid-phrase by the
        # token cap; short ones are still a name, long ones are a fragment.
        if not finished and len(words) > 6:
            continue

        out = titles._titlecase(titles._fit(s, titles.MAX_LEN)).strip(EDGE)
        if out and any(c.isalpha() for c in out):
            return out
    return None


# --- public API ------------------------------------------------------------

# Structure tags and non-lexical vocals carry no subject matter. A lyric of
# "[verse]\n[chorus]" is *syntactically* present but says nothing, and the model
# will still answer confidently -- it invented "Fire in the Hole" from exactly
# that. Better to decline and let titles.derive() fall through to the prompt.
_HOLLOW = re.compile(r"[\[\(<][^\]\)>]*[\]\)>]|\b(?:la|na|ooh?|ahh?|mmm?|hmm|"
                     r"yeah|oh|uh|doo|dum)\b|[^\w\s]", re.I)
_MIN_CONTENT = 12


def _substance(text):
    """What is left once tags and filler syllables are removed."""
    return " ".join(_HOLLOW.sub(" ", text or "").split())


def _pick(lyrics, prompt):
    """Lyrics win when both are present -- the same order derive() falls in."""
    lyr = (lyrics or "").strip()
    if lyr and len(_substance(lyr)) >= _MIN_CONTENT:
        return "lyrics", lyr
    if (prompt or "").strip():
        return "style", prompt.strip()
    # Lyrics existed but were hollow, and there is no prompt to fall back on.
    return None, None


def suggest(lyrics=None, prompt=None, n=1, timeout=120, device="cpu"):
    """Up to `n` candidate titles, best first. [] on any failure at all.

    Callable from either venv: without torch it re-enters this file as a
    subprocess under the MiniMax python. Failure is always [] and never an
    exception -- the caller's fallback is titles.derive(), which cannot fail,
    so there is nothing here worth propagating.
    """
    return _suggest(lyrics, prompt, n, timeout, device)[0]


def _suggest(lyrics=None, prompt=None, n=1, timeout=120, device="cpu"):
    """suggest() plus the reason it came back empty, for the CLI to report."""
    kind, text = _pick(lyrics, prompt)
    if not text:
        return [], "no lyrics or prompt given"
    n = max(1, int(n or 1))
    timeout = max(5.0, float(timeout or 120))

    # One gate for both paths, before torch is even looked for: there is no
    # download on this path, so an uncached model is a fast, final no.
    if not is_ready():
        return [], (f"{MODEL_ID} is not in the local cache "
                    f"({_repo_dir()}) -- run this file with --fetch once")

    # find_spec rather than `import torch`: importing it here would pin the
    # CUDA state before _load gets to hide the GPU.
    try:
        import importlib.util
        have_torch = importlib.util.find_spec("torch") is not None
    except (ImportError, ValueError):
        have_torch = False
    if not have_torch:
        return _via_subprocess(lyrics, prompt, n, timeout)

    deadline = time.monotonic() + timeout
    try:
        tok, model = _load(device)
    except Exception as e:                      # corrupt cache, OOM, bad venv
        _warn(f"load failed: {type(e).__name__}: {e}")
        return [], f"load failed: {type(e).__name__}: {e}"

    # Seeded off the text, so a re-title of the same track is the same title.
    base = zlib.crc32(text.encode("utf-8"))
    out, seen, err = [], set(), None
    for i in range(n + 2):                      # a couple of spare rolls
        if len(out) >= n or time.monotonic() >= deadline:
            break
        try:
            raw, finished = _generate(tok, model, kind, text, base + i,
                                      deadline)
        except Exception as e:
            _warn(f"generate failed: {type(e).__name__}: {e}")
            err = f"generate failed: {type(e).__name__}: {e}"
            break
        got = _clean(raw, finished)
        if got and got.casefold() not in seen:
            seen.add(got.casefold())
            out.append(got)
        elif not got:
            _warn(f"rejected {raw!r}")
    if not out and err is None:
        err = "no usable title survived post-processing"
    return out, err


def _via_subprocess(lyrics, prompt, n, timeout):
    """Run this same file under the MiniMax venv. Returns (titles, error)."""
    if not os.path.exists(MINIMAX_PY):
        return [], f"no torch here and {MINIMAX_PY} is missing"
    # If we ARE that python and torch still would not import, spawning a second
    # copy of ourselves changes nothing and recurses until the box gives up.
    if os.path.normcase(sys.executable) == os.path.normcase(MINIMAX_PY):
        return [], "torch does not import inside the MiniMax venv"

    tmp = []
    cmd = [MINIMAX_PY, os.path.abspath(__file__), "--json", "--n", str(n),
           # The child stops itself a little early so it can print a JSON
           # object; the parent's timeout is the hard wall behind that.
           "--timeout", str(max(5.0, timeout - 5.0))]
    try:
        for flag, text in (("--lyrics-file", lyrics),
                           ("--prompt-file", prompt)):
            if (text or "").strip():
                fd, path = tempfile.mkstemp(suffix=".txt", text=True)
                # Lyrics carry newlines and typographic quotes; a temp file
                # sidesteps both argv length limits and cp1252 mangling.
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(text)
                tmp.append(path)
                cmd += [flag, path]

        env = dict(os.environ)
        env["PYTHONIOENCODING"] = "utf-8"
        env["CUDA_VISIBLE_DEVICES"] = "-1"      # the child is always CPU
        r = subprocess.run(cmd, capture_output=True, text=True,
                           encoding="utf-8", errors="replace",
                           timeout=timeout, env=env, creationflags=NOWIN)
    except subprocess.TimeoutExpired:
        return [], f"subprocess exceeded {timeout:.0f}s"
    except OSError as e:
        return [], f"subprocess failed: {e}"
    finally:
        for path in tmp:
            try:
                os.unlink(path)
            except OSError:
                pass

    # Last JSON-looking line, not the whole of stdout: belt and braces against
    # a library that prints before we can redirect it.
    for line in reversed((r.stdout or "").splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                got = json.loads(line)
            except ValueError:
                continue
            if got.get("error"):
                _warn(str(got["error"]))
            return ([str(t) for t in (got.get("titles") or [])][:n],
                    got.get("error"))
    return [], (f"no JSON from subprocess (rc={r.returncode}): "
                f"{(r.stderr or '')[-400:]}")


# --- CLI -------------------------------------------------------------------

def _read(path):
    if not path:
        return None
    # utf-8-sig: Notepad and PowerShell's Out-File both leave a BOM, which
    # otherwise becomes the first character of the lyric.
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        return f.read()


def main(argv=None):
    # The JSON object is the protocol and owns the real stdout; transformers,
    # tqdm and any warning must land on stderr or they corrupt it. Same move as
    # workers/minimax_worker.py.
    proto = sys.stdout
    sys.stdout = sys.stderr
    for stream in (proto, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")   # cp1252 cannot hold a title
        except (AttributeError, ValueError):
            pass

    ap = argparse.ArgumentParser(description=f"Song titles from {MODEL_ID}.")
    ap.add_argument("--lyrics-file")
    ap.add_argument("--prompt-file")
    ap.add_argument("--lyrics", help="inline alternative to --lyrics-file")
    ap.add_argument("--prompt", help="inline alternative to --prompt-file")
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--timeout", type=float, default=120)
    # Never a default. The 3080 belongs to music generation.
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    ap.add_argument("--json", action="store_true",
                    help="accepted for clarity; stdout is always one JSON "
                         "object either way")
    ap.add_argument("--fetch", action="store_true",
                    help="one-time setup: download the model. Titling itself "
                         "never touches the network")
    a = ap.parse_args(argv)

    t0 = time.perf_counter()
    err = None
    got = []
    try:
        if a.fetch:
            _load(a.device, fetch=True)
            # A successful fetch has no titles to report, so `ok` tracks the
            # cache instead -- the same question the caller asked.
            ok = is_ready()
            err = None if ok else "download finished, cache still incomplete"
        else:
            lyrics = a.lyrics if a.lyrics is not None else _read(a.lyrics_file)
            prompt = a.prompt if a.prompt is not None else _read(a.prompt_file)
            got, err = _suggest(lyrics=lyrics, prompt=prompt, n=a.n,
                                timeout=a.timeout, device=a.device)
            ok = bool(got)
    except Exception as e:                      # a CLI never traces to stdout
        ok, err = False, f"{type(e).__name__}: {e}"

    proto.write(json.dumps(
        {"ok": ok, "titles": got, "error": err, "model": MODEL_ID,
         "elapsed": round(time.perf_counter() - t0, 2)},
        ensure_ascii=False) + "\n")
    proto.flush()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
