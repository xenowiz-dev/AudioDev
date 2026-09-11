"""songwriter.py -- a local LLM that writes lyrics and style prompts.

Six jobs from one model: {lyrics, style} x {generate, extend, edit}.

  generate  a brief in, a finished piece out
  extend    continue what is already written, without repeating it
  edit      change one named thing and return the whole piece otherwise intact

The style target is not free text. MiniMax was trained on **sectioned
captions** -- Global Metadata / Vocal Details / Arrangement, each with named
fields -- and a short keyword prompt measurably loses arrangement control. So
the style prompt for MiniMax is generated against that exact skeleton and then
CHECKED against it: if the model drops a heading it is told so and asked again,
once. ACE-Step wants the opposite (a plain keyword line), so it gets a
different template entirely. `--model` picks.

CPU ONLY by default, for the same reason llm_title.py is: the 3080 is shared
with music generation, and writing a chorus must never be why a render OOMs.
The CUDA_VISIBLE_DEVICES dance and the stray-parameter guard are lifted from
that module, where they were needed because the environment variable was
observed being ignored on this platform.

Progress goes to stderr, the single JSON result to stdout -- the same split
transcribe.py uses, so the studio can stream one and parse the other.

    python songwriter.py --target lyrics --mode generate \
        --brief "a defiant synth-pop song about quitting a job" --json
"""

import argparse
import glob
import json
import os
import re
import sys
import time

# Root discovery: AUDIODEV_ROOT if the launcher set it, else relative to
# this file. Every path below derives from it, so the tree moves as one.
_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("AUDIODEV_ROOT") or _HERE

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
HF_HOME = os.path.join(ROOT, "minimax", "hf")

TARGETS = ("lyrics", "style")
MODES = ("generate", "extend", "edit")

_TOK = _MODEL = None


def _warn(msg):
    print(msg, file=sys.stderr, flush=True)


# --------------------------------------------------------------- readiness

def _repo_dir():
    return os.path.join(HF_HOME, "hub",
                        "models--" + MODEL_ID.replace("/", "--"))


def is_ready():
    """True when the weights are already on disk. Never downloads in-band.

    Same gate as llm_title.is_ready, and for the same measured reason: 3 GB
    cannot arrive inside a request, and a dead endpoint took 82 s to say so.
    A filesystem walk, so the studio venv (no torch, no hub) can call it.
    """
    for snap in glob.glob(os.path.join(_repo_dir(), "snapshots", "*")):
        have = set(os.listdir(snap)) if os.path.isdir(snap) else set()
        if not any(f.endswith(".safetensors") for f in have):
            continue
        if "config.json" in have and ("tokenizer.json" in have
                                      or "tokenizer_config.json" in have):
            return True
    return False


# ----------------------------------------------------------------- prompts

LYRIC_RULES = """You are a professional lyricist. You output song lyrics and \
nothing else -- no title, no commentary, no explanation, no markdown.

Structure the song with section tags alone on their own line: {tags}

Rules:
- 4 to 6 lines per section.
- Concrete images and specific detail. Name things. No "feel the fire",
  no "reach for the sky", no rhyming dictionary filler.
- Sing-able lines: keep them short enough to breathe.
- Do not restate the brief back as lyrics."""

TAGS_MINIMAX = ("[Intro], [Verse 1], [Pre-Chorus], [Chorus], [Verse 2], "
                "[Bridge], [Final Chorus], [Outro]. Descriptive tags are fine "
                "and MiniMax uses them, e.g. [Half-Time Bridge].")
TAGS_ACESTEP = "[verse], [chorus], [bridge] -- lowercase, plain."

# The MiniMax caption skeleton, verbatim from what the model was trained on.
STYLE_MINIMAX = """You write production briefs for the MiniMax Music model.

Reply with EXACTLY this structure -- the three headings on their own lines, \
each field on one line, nothing before or after:

Global Metadata
Basic Attributes: bpm is <number>. key is <letter>, and scale is <major|minor>. \
<Genre>.
Sonics & Production Profile: <4-8 production descriptors>
Vocal Details
Vocal Gender & Timbre: Singer A (<Female|Male>). <timbre in a short phrase>
Vocal Style: <how it is sung, and how that changes between sections>
Arrangement
Primary: <the lead instrument and when it plays>
Secondary: <supporting instruments and when they enter>

The headings are load-bearing: this model was trained on sectioned captions \
and loses arrangement control without them. Do not add headings, do not use \
markdown, do not explain."""

STYLE_ACESTEP = """You write style prompts for the ACE-Step music model.

Reply with ONE line: comma-separated keywords covering genre, instrumentation, \
production character and tempo. No headings, no sentences, no explanation.

Example: upbeat indie pop, jangly electric guitar, live drums, warm analog \
production, 110 bpm"""

# What makes a MiniMax caption a caption. The headings alone are not enough:
# measured, this model will happily answer with the right three headings in
# markdown over bullet lists ("**Global Metadata**" then "- **BPM:** 120"),
# which passes a heading-only check and is not the format the music model was
# trained on. The field prefixes are the real contract.
REQUIRED_HEADINGS = ("Global Metadata", "Vocal Details", "Arrangement")
REQUIRED_FIELDS = ("Basic Attributes:", "Sonics & Production Profile:",
                   "Vocal Gender & Timbre:", "Vocal Style:",
                   "Primary:", "Secondary:")


def system_for(target, model):
    if target == "style":
        return STYLE_MINIMAX if model == "minimax" else STYLE_ACESTEP
    return LYRIC_RULES.format(
        tags=TAGS_MINIMAX if model == "minimax" else TAGS_ACESTEP)


def user_for(target, mode, brief, existing):
    what = "lyrics" if target == "lyrics" else "style prompt"
    brief = (brief or "").strip()
    existing = (existing or "").strip()

    if mode == "generate":
        return f"Write {what} for this brief:\n\n{brief}"

    if mode == "extend":
        if not existing:
            raise ValueError("extend needs the existing text")
        head = (f"Continue the song below. Write the NEXT sections ONLY -- do "
                f"not repeat or rewrite anything already there, and do not "
                f"add commentary."
                if target == "lyrics" else
                "Enrich the style prompt below with more specific "
                "instrumentation and production detail. Return the COMPLETE "
                "prompt in the required structure.")
        tail = f"\n\nKeep to this brief: {brief}" if brief else ""
        return f"{head}{tail}\n\n---\n{existing}\n---"

    # edit
    if not existing:
        raise ValueError("edit needs the existing text")
    if not brief:
        raise ValueError("edit needs an instruction saying what to change")
    # "Do not add sections" is spelled out because, measured, this model reads
    # "return the complete lyrics" as an invitation to finish the song: asked
    # to rewrite the chorus of a two-section draft it returned seven sections,
    # four of them invented. The section count is checked afterwards too.
    extra = ("\n\nDo NOT add any new sections. The result must contain exactly "
             "the same section tags, in the same order, as the text below -- "
             "no more, no fewer." if target == "lyrics" else "")
    return (f"Apply this change to the {what} below: {brief}\n\n"
            f"Change ONLY what that asks for. Leave every other section "
            f"exactly as it is, word for word. Return the COMPLETE "
            f"{what}.{extra}\n\n---\n{existing}\n---")


# ---------------------------------------------------------------- the model

def _load(device="cpu", fetch=False):
    global _TOK, _MODEL
    if _MODEL is not None:
        return _TOK, _MODEL

    os.environ.setdefault("HF_HOME", HF_HOME)
    if device == "cpu" and "torch" not in sys.modules:
        # "-1", not "": an empty CUDA_VISIBLE_DEVICES reads as unset on
        # Windows and torch still enumerates the 3080. Only valid before torch
        # is first imported.
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    t0 = time.perf_counter()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    kw = dict(local_files_only=not fetch)
    tok = AutoTokenizer.from_pretrained(MODEL_ID, **kw)
    # fp32 on CPU because bf16 is slower there (no AVX512-BF16); fp16 on the
    # card because 3.1 GB fits beside almost anything and 6.2 GB does not.
    dtype = torch.float32 if device == "cpu" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=dtype, device_map=device, **kw)
    model.eval()

    if device == "cpu":
        # Checked, not assumed -- CUDA_VISIBLE_DEVICES has been seen ignored
        # here, and a writer silently taking VRAM surfaces as a failed render.
        stray = next((n for n, p in model.named_parameters()
                      if p.device.type != "cpu"), None)
        if stray is not None:
            raise RuntimeError(f"{stray} landed on GPU despite device_map=cpu")

    _TOK, _MODEL = tok, model
    _warn(f"loaded {MODEL_ID} ({device}) in {time.perf_counter() - t0:.1f}s")
    return tok, model


def _generate(tok, model, msgs, max_new, seed, temperature, deadline):
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList

    class Tick(StoppingCriteria):
        """Progress and the deadline. Between tokens is the only place a
        generate() loop can be interrupted at all."""

        def __init__(self, n_in):
            self.n_in, self.last = n_in, time.monotonic()

        def __call__(self, input_ids, scores, **kw):
            now = time.monotonic()
            if now - self.last >= 2.0:
                self.last = now
                _warn(f"writing {input_ids.shape[1] - self.n_in} tokens")
            return now >= deadline

    enc = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True)
    # apply_chat_template always builds on CPU; generate() warns and slows down
    # if the ids and the weights disagree about where they live.
    enc = {k: v.to(model.device) for k, v in enc.items()}
    n_in = enc["input_ids"].shape[1]
    torch.manual_seed(seed)
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(
            **enc, max_new_tokens=max_new, do_sample=True,
            temperature=temperature, top_p=0.95, repetition_penalty=1.05,
            pad_token_id=tok.eos_token_id,
            stopping_criteria=StoppingCriteriaList([Tick(n_in)]))
    n = out.shape[1] - n_in
    dt = time.perf_counter() - t0
    _warn(f"{n} tokens in {dt:.0f}s ({n / max(dt, .01):.1f} tok/s)")
    return tok.decode(out[0][n_in:], skip_special_tokens=True).strip(), n


# ---------------------------------------------------------------- cleanup

FENCE = re.compile(r"^\s*```[a-zA-Z]*\s*|\s*```\s*$")
LEAD_IN = re.compile(
    r"^\s*(sure|certainly|here('s| is| are)|of course|below is)\b[^\n]*\n+",
    re.I)


def tidy(text, target):
    """Strip the wrappers small instruct models add no matter what you ask."""
    text = FENCE.sub("", text or "").strip()
    text = LEAD_IN.sub("", text).strip()
    text = re.sub(r"^---\s*$", "", text, flags=re.M).strip()
    # A title line the prompt asked it not to write.
    text = re.sub(r'^(Title|Song Title):.*\n+', "", text, flags=re.I).strip()
    if target == "style":
        # Markdown the prompt asked it not to use: bold around headings and
        # field names, and bullet markers. Stripping these turns a near-miss
        # into a pass often enough to be worth doing before the retry.
        text = re.sub(r"\*\*", "", text)
        text = re.sub(r"^\s*[-*]\s+", "", text, flags=re.M)
        text = re.sub(r"\n{3,}", "\n", text)
    return text.strip()


def check_style(text, model):
    """[] when the caption is shaped right, else what is missing.

    Only MiniMax has a structure to violate; ACE-Step's answer is one line.
    """
    if model != "minimax":
        return [] if text.strip() else ["any text at all"]
    if not text.strip():
        return ["any text at all"]
    missing = [h for h in REQUIRED_HEADINGS
               if not re.search(rf"^\s*{re.escape(h)}\s*$", text, re.M)]
    missing += [f for f in REQUIRED_FIELDS if f not in text]
    return missing


TAG_LINE = re.compile(r"^[ \t]*\[([^\]\n]{1,40})\][ \t]*$", re.M)


def section_tags(text):
    return TAG_LINE.findall(text or "")


def check_lyrics(text):
    return len(section_tags(text))


def split_sections(text):
    """[(tag, block)] where block includes the tag line. Preamble tag is None."""
    text = text or ""
    marks = list(TAG_LINE.finditer(text))
    if not marks:
        return [(None, text)]
    out = []
    if marks[0].start() > 0 and text[:marks[0].start()].strip():
        out.append((None, text[:marks[0].start()]))
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        out.append((m.group(1), text[m.start():end]))
    return out


def target_section(brief, sections):
    """The one section the brief names, or None.

    "rewrite the chorus" is answerable without a model, and answering it
    structurally is the difference between an edit that works and one that
    doesn't: asked to return a whole song with one section changed, a 1.5B
    model reliably rewrites the wrong things and invents new sections
    (measured twice: 2 sections in, 7 out, and the instruction itself lost).
    Cutting the problem down to "rewrite these four lines" removes the failure
    rather than asking the model not to have it.
    """
    b = (brief or "").lower()
    hits = [i for i, (tag, _) in enumerate(sections)
            if tag and tag.lower() in b]
    if len(hits) == 1:
        return hits[0]
    # "the chorus" should still find [Big Poppy Chorus]; "verse 2" should not
    # match [Verse 1]. Only accept a keyword when it identifies ONE section.
    for word in ("pre-chorus", "chorus", "verse 1", "verse 2", "verse 3",
                 "bridge", "intro", "outro", "hook", "refrain"):
        if word not in b:
            continue
        hits = [i for i, (tag, _) in enumerate(sections)
                if tag and word in tag.lower()]
        if len(hits) == 1:
            return hits[0]
    return None


# ------------------------------------------------------------------ public

def write(target, mode, brief="", existing="", model="minimax", seed=None,
          max_new=None, temperature=0.9, timeout=600, device="cpu"):
    if target not in TARGETS:
        raise ValueError(f"unknown target {target!r}")
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}")
    if seed is None:
        seed = int.from_bytes(os.urandom(4), "big")
    if max_new is None:
        max_new = 260 if target == "style" else 640

    tok, mdl = _load(device)
    deadline = time.monotonic() + timeout
    warning, retried, scoped = None, False, None

    # -- scoped edit: rewrite one section, splice it back ------------------
    if target == "lyrics" and mode == "edit":
        sections = split_sections(existing)
        idx = target_section(brief, sections)
        if idx is not None:
            scoped = sections[idx][0]
            _warn(f"editing only [{scoped}] - the rest is spliced back "
                  f"untouched")
            sys_msg = ("You rewrite one section of a song. Output ONLY the "
                       "replacement for that section, starting with its "
                       f"[{scoped}] tag line. Same number of lines or close to "
                       "it. No other sections, no commentary.")
            usr = (f"Rewrite this section so that: {brief}\n\n"
                   f"---\n{sections[idx][1].strip()}\n---")
            piece, n = _generate(
                tok, mdl,
                [{"role": "system", "content": sys_msg},
                 {"role": "user", "content": usr}],
                240, seed, temperature, deadline)
            piece = tidy(piece, "lyrics").strip()
            # Keep only the first section the model returned, and force the
            # original tag back on: a replacement that renames the section
            # would silently restructure the song.
            parts = split_sections(piece)
            body = next((b for t, b in parts if t), piece)
            body = TAG_LINE.sub(f"[{scoped}]", body, count=1)
            if not TAG_LINE.match(body):
                body = f"[{scoped}]\n{body}"
            sections[idx] = (scoped, body.rstrip() + "\n\n")
            text = "".join(b for _, b in sections).strip()
            return {"text": text, "target": target, "mode": mode,
                    "model": model, "seed": seed, "tokens": n, "retried": False,
                    "scoped_to": scoped, "warning": None}
        warning = None      # fall through to the whole-text edit

    msgs = [{"role": "system", "content": system_for(target, model)},
            {"role": "user", "content": user_for(target, mode, brief, existing)}]
    text, n = _generate(tok, mdl, msgs, max_new, seed, temperature, deadline)
    text = tidy(text, target)

    if target == "style":
        missing = check_style(text, model)
        if missing and time.monotonic() < deadline - 30:
            # One corrective round trip. Small models drop a heading roughly
            # as often as they keep one, and being shown the omission fixes it
            # far more reliably than raising the temperature or re-rolling.
            _warn(f"missing {', '.join(missing)} - asking again")
            retried = True
            msgs += [{"role": "assistant", "content": text},
                     {"role": "user", "content":
                      "That is missing the required heading(s): "
                      + ", ".join(missing)
                      + ". Reply again with the COMPLETE structure, every "
                        "heading present, and nothing else."}]
            text2, n2 = _generate(tok, mdl, msgs, max_new, seed + 1,
                                  temperature, deadline)
            text2 = tidy(text2, target)
            if not check_style(text2, model):
                text, n = text2, n + n2
            else:
                n += n2
        missing = check_style(text, model)
        if missing:
            warning = ("Missing " + ", ".join(missing)
                       + " — MiniMax loses arrangement control without the "
                         "headings. Edit it, or roll again.")
    else:
        tags = check_lyrics(text)
        if tags < 2:
            warning = (f"Only {tags} section tag(s). MiniMax and ACE-Step both "
                       f"follow [Verse] / [Chorus] markers — add them, or roll "
                       f"again.")
        elif mode == "edit":
            # A small model treats "return the complete lyrics" as permission
            # to finish the song. Say so rather than let it pass as an edit.
            was = len(section_tags(existing))
            if was and tags > was:
                warning = (f"The model added {tags - was} section(s) you did "
                           f"not ask for ({was} in, {tags} out). Keep what you "
                           f"want and undo the rest.")

    return {"text": text, "target": target, "mode": mode, "model": model,
            "seed": seed, "tokens": n, "retried": retried,
            "scoped_to": scoped, "warning": warning}


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", choices=TARGETS, default="lyrics")
    ap.add_argument("--mode", choices=MODES, default="generate")
    ap.add_argument("--model", choices=("minimax", "acestep"),
                    default="minimax")
    ap.add_argument("--brief", default="")
    ap.add_argument("--brief-file")
    ap.add_argument("--existing", default="")
    ap.add_argument("--existing-file")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--max-new", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--fetch", action="store_true",
                    help="one-time download of the weights")
    ap.add_argument("--ready", action="store_true",
                    help="exit 0 if the weights are on disk, else 1")
    a = ap.parse_args(argv)

    if a.ready:
        ok = is_ready()
        print("ready" if ok else "missing")
        return 0 if ok else 1
    if a.fetch:
        _load(a.device, fetch=True)
        print("fetched")
        return 0

    def read(inline, path):
        if path:
            with open(path, encoding="utf-8") as fh:
                return fh.read()
        return inline

    res = write(a.target, a.mode,
                brief=read(a.brief, a.brief_file),
                existing=read(a.existing, a.existing_file),
                model=a.model, seed=a.seed, max_new=a.max_new,
                temperature=a.temperature, timeout=a.timeout, device=a.device)
    if a.json:
        print(json.dumps(res))
    else:
        print(res["text"])
        if res["warning"]:
            _warn("WARNING: " + res["warning"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
