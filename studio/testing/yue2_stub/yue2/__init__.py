"""A stand-in for the `yue2` package, for exercising the studio's YuE2 path on
a box that cannot run the model.

Carries the SAME public signatures as yue2 0.1.6 (pipeline.py, read on
2026-09-11): from_pretrained(...) -> pipeline usable as a context manager;
plan(...) -> object with .abc/.truncated/.save(dir); __call__(...) -> object
with .audio/.sample_rate/.abc/.truncated/.save_artifacts(dir). It writes the
same file names. It does NOT model quality, VRAM or timing -- it proves the
plumbing, not the music.

Use: PYTHONPATH=<this dir's parent> so `import yue2` finds this before the
real package in the venv.
"""
import json
import os
import time

import numpy as np

STUB = True
__version__ = "0.0-stub"


def _abc_for(style, lyrics, cot, seed):
    lines = [l for l in (lyrics or "").splitlines() if l.strip() and not l.strip().startswith("[")]
    bars = max(4, min(64, len(lines) * 2))
    body = " | ".join("C2 E2 G2 c2" if i % 2 == 0 else "A2 c2 e2 a2" for i in range(bars))
    chords = "" if cot != "full" else '\n%%chords "C" "Am" "F" "G"'
    return (f"X:1\nT:stub plan (seed {seed})\nM:4/4\nL:1/8\nQ:1/4=96\nK:C\n"
            f"%% style: {style[:60]}{chords}\n{body} |]\n")


class _Plan:
    def __init__(self, request, abc, truncated=False):
        self.request = request
        self.abc = abc
        self.truncated = truncated
        self.timing = {"seconds": 0.1}

    def save(self, directory):
        os.makedirs(directory, exist_ok=True)
        if self.abc is not None:
            with open(os.path.join(directory, "score.abc"), "w", encoding="utf-8") as fh:
                fh.write(self.abc)
        with open(os.path.join(directory, "plan.json"), "w", encoding="utf-8") as fh:
            json.dump({"request": self.request, "timing": self.timing,
                       "truncated": self.truncated, "abc": self.abc}, fh)


class _Song:
    def __init__(self, plan):
        self.semantic_plan = plan
        self.sample_rate = 48000
        n = int(3.0 * self.sample_rate)
        t = np.arange(n) / self.sample_rate
        tone = 0.2 * np.sin(2 * np.pi * 220.0 * t)
        self.audio = np.stack([tone, tone], axis=1).astype(np.float32)
        self.timing = {"e2e_seconds": 0.3}

    @property
    def abc(self):
        return self.semantic_plan.abc

    @property
    def truncated(self):
        return {"abc": self.semantic_plan.truncated, "semantic": False}

    def save_artifacts(self, directory):
        import soundfile as sf
        os.makedirs(directory, exist_ok=True)
        self.semantic_plan.save(directory)
        sf.write(os.path.join(directory, "audio.flac"), self.audio, self.sample_rate,
                 subtype="PCM_24")
        with open(os.path.join(directory, "result.json"), "w", encoding="utf-8") as fh:
            json.dump({"status": "complete", "stub": True,
                       "truncated": self.truncated}, fh)
        return {"status": "complete"}


class YuE2Pipeline:
    def __init__(self, **kw):
        self.kw = kw
        self.closed = False

    @classmethod
    def from_pretrained(cls, model="m-a-p/YuE2-3B", *, vae="m-a-p/YuE2-Vae",
                        revision=None, vae_revision=None, local_files_only=False,
                        token=None, cache_dir=None, progress=True, **kwargs):
        time.sleep(0.2)
        return cls(model=model, vae=vae, **kwargs)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        self.closed = True

    def _request(self, style=None, lyrics=None, *, tags=None, **kwargs):
        if style is None or lyrics is None:
            raise ValueError("Provide style and lyrics")
        cot = kwargs.get("cot", "full")
        if cot not in ("off", "melody", "full"):
            raise ValueError("cot must be off, melody, or full")
        abc = kwargs.get("abc")
        if abc is not None and (cot == "off" or not str(abc).strip()):
            raise ValueError("External ABC requires cot != off and nonempty text")
        cfg = kwargs.get("cfg_scale")
        if cfg is not None and not (0 <= float(cfg) <= 20):
            raise ValueError("cfg_scale out of range")
        return dict(style=style, lyrics=lyrics, cot=cot, seed=int(kwargs.get("seed", 831001)),
                    abc=abc, cfg_scale=cfg, id=kwargs.get("id", "song"))

    def plan(self, style=None, lyrics=None, *, tags=None, request=None, abc_sampling=None,
             cancelled=None, on_token=None, **kwargs):
        request = request or self._request(style, lyrics, tags=tags, **kwargs)
        if request["cot"] == "off":
            return _Plan(request, None)
        if request["abc"] is not None:
            return _Plan(request, request["abc"])
        for i in range(30):
            if on_token:
                on_token("abc", i)
        return _Plan(request, _abc_for(request["style"], request["lyrics"],
                                       request["cot"], request["seed"]))

    def __call__(self, style=None, lyrics=None, *, tags=None, abc_sampling=None,
                 semantic_sampling=None, cancelled=None, on_token=None, **kwargs):
        request = self._request(style, lyrics, tags=tags, **kwargs)
        plan = self.plan(request=request, on_token=on_token)
        for i in range(60):
            if on_token:
                on_token("semantic", i)
        time.sleep(0.3)
        return _Song(plan)
