"""Measure bpm, keyscale and timesignature for the training set.

WHY THIS EXISTS
build_dataset.py wrote the literal string "N/A" into all three fields for every
sample. They are not inert: preprocess_prompt.py:63-73 formats them into the
`# Metas` block of SFT_GEN_PROMPT, which feeds the text encoder in training AND
at inference (core/generation/handler/conditioning_text.py:116,168). So the
adapter learned "the trigger word co-occurs with unspecified tempo, key and
meter" -- and the 5Hz planner supplies real values at generation time, which is
off-distribution.

WHAT IT USES
  beats/downbeats  beat_this (ISMIR 2024). Chosen over librosa's tracker
                   because it emits DOWNBEATS, which is the only reliable way
                   to get a time signature -- inter-beat spacing alone cannot
                   distinguish 3/4 from 6/8 from 4/4.
  key              Krumhansl-Schmuckler correlation over chroma. The model's
                   vocabulary is only major/minor (constants.py KEYSCALE_MODES),
                   so a 24-profile correlation is exactly the right shape.
  cross-check      librosa.beat.beat_track, purely to disagree with beat_this.

CPU ONLY, by design: the GPU is shared with ComfyUI and this is a batch job
over 184 files. Nothing here touches CUDA.

HONESTY RULES BAKED IN
  * Every track carries a confidence and the reason for it.
  * Octave ambiguity (half/double time) is the known failure mode of every beat
    tracker on a catalogue that spans doom folk and trap EDM. It is FLAGGED,
    never silently resolved.
  * A track we are not confident about keeps "N/A" rather than getting a guess.
    A wrong tempo binds the trigger token to a false fact; "N/A" merely says
    nothing. Wrong is worse than silent.
"""
import argparse
import json
import io
import os
import re
import sys
import time
import warnings

# Root discovery: AUDIODEV_ROOT if the launcher set it, else relative to
# this file. Every path below derives from it, so the tree moves as one.
_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("AUDIODEV_ROOT") or os.path.normpath(os.path.join(_HERE, ".."))

warnings.filterwarnings("ignore")

DATASET = os.path.join(ROOT, "train", "dataset.json")
OUT = os.path.join(ROOT, "train", "labels.json")
SIDECAR_DIR = r"C:\Users\Kevin\Downloads\Suno Playlist 8-11-2026"

VALID_TIME_SIGNATURES = [2, 3, 4, 6]
BPM_MIN, BPM_MAX = 30, 300
NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Krumhansl-Kessler key profiles.
KS_MAJOR = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
KS_MINOR = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]


def corr(a, b):
    import numpy as np
    a = np.asarray(a, float) - np.mean(a)
    b = np.asarray(b, float) - np.mean(b)
    d = (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / d) if d else 0.0


def estimate_key(path, sr=22050):
    """Best major/minor key, plus how the decision was reached.

    The margin over the runner-up is the honest confidence measure, with one
    structural exception worth handling rather than discarding: the runner-up
    is nearly always the RELATIVE major/minor, which shares all seven notes.
    Chroma correlation cannot separate those by construction -- C major and A
    minor have identical pitch-class content -- so a small margin there means
    "the note set is certain, the tonic is not", which is a different problem
    from "no key found".

    Relative pairs are broken on BASS energy instead: the root of the key is
    the pitch class the low end sits on. Everything else with a small margin is
    genuinely undecided and gets rejected.

    Returns (name, corr, margin, method).
    """
    import librosa
    import numpy as np
    y, _sr = librosa.load(path, sr=sr, mono=True)
    if y.size == 0:
        return None, 0.0, 0.0, "empty file"
    yh = librosa.effects.harmonic(y, margin=3)         # de-emphasise drums
    prof = np.mean(librosa.feature.chroma_cqt(y=yh, sr=_sr), axis=1)

    scores = []
    for i in range(12):
        scores.append((corr(np.roll(prof, -i), KS_MAJOR), i, "major"))
        scores.append((corr(np.roll(prof, -i), KS_MINOR), i, "minor"))
    scores.sort(reverse=True)
    (b_sc, b_i, b_mode), (r_sc, r_i, r_mode) = scores[0], scores[1]
    margin = b_sc - r_sc
    name = f"{NOTES[b_i]} {b_mode}"

    if margin >= 0.05:
        return name, round(b_sc, 3), round(margin, 3), "clear margin"

    # Is the runner-up the relative key? (major tonic + 9 semitones = relative
    # minor; equivalently minor tonic + 3 = relative major.)
    rel = (b_mode == "major" and r_mode == "minor" and (b_i + 9) % 12 == r_i) or \
          (b_mode == "minor" and r_mode == "major" and (b_i + 3) % 12 == r_i)
    if not rel:
        return None, round(b_sc, 3), round(margin, 3), "undecided: runner-up unrelated"

    # Bass chroma: which of the two candidate tonics does the low end sit on?
    # Reuses the harmonic signal already computed -- HPSS is the expensive step
    # in this function and running it twice on identical input bought nothing.
    bass = librosa.feature.chroma_cqt(y=yh, sr=_sr, fmin=librosa.note_to_hz("C1"),
                                      n_octaves=3)
    bprof = np.mean(bass, axis=1)
    b_energy, r_energy = float(bprof[b_i]), float(bprof[r_i])
    if abs(b_energy - r_energy) < 1e-6:
        return None, round(b_sc, 3), round(margin, 3), "relative pair, bass undecided"
    if b_energy >= r_energy:
        return name, round(b_sc, 3), round(margin, 3), "relative pair, bass picked tonic"
    return (f"{NOTES[r_i]} {r_mode}", round(r_sc, 3), round(margin, 3),
            "relative pair, bass picked the runner-up")


def beats_to_meter(beats, downbeats):
    """beats per bar, from the ratio of bar length to beat length."""
    import numpy as np
    if len(beats) < 8 or len(downbeats) < 3:
        return None, None, "too few beats"
    ib = float(np.median(np.diff(beats)))
    bar = float(np.median(np.diff(downbeats)))
    if ib <= 0:
        return None, None, "degenerate beat spacing"
    raw = bar / ib
    near = min(VALID_TIME_SIGNATURES, key=lambda v: abs(v - raw))
    # 0.18 beats of slack: enough for tracker jitter, tight enough that 3.5
    # (which is neither 3 nor 4) is rejected rather than rounded.
    if abs(near - raw) > 0.18:
        return None, round(raw, 2), f"beats/bar {raw:.2f} matches no valid meter"
    return near, round(raw, 2), None


def user_bpm(filename):
    """A BPM the owner typed into the Suno prompt, if any. Ground truth for
    INTENT -- not proof of what Suno rendered, but the only external check we
    have on whether the tracker is octave-erring systematically."""
    p = os.path.join(SIDECAR_DIR, filename + ".txt")
    if not os.path.isfile(p):
        return None
    try:
        txt = io.open(p, encoding="utf-8", errors="replace").read()
    except OSError:
        return None
    m = re.search(r"(\d{2,3})\s*bpm", txt, re.I)
    return int(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    import numpy as np
    from beat_this.inference import File2Beats

    d = json.load(io.open(DATASET, encoding="utf-8-sig"))
    samples = d["samples"]
    if args.limit:
        samples = samples[:args.limit]

    print(f"labelling {len(samples)} tracks on CPU", flush=True)
    f2b = File2Beats(checkpoint_path="final0", dbn=False, device="cpu")

    out, t0 = [], time.time()
    for i, s in enumerate(samples, 1):
        path = s["audio_path"]
        rec = {"filename": s["filename"], "title": s.get("_title"),
               "duration": s.get("duration")}
        try:
            # beat_this prints a progress bar per file; keep the log readable.
            _stdout = sys.stdout
            sys.stdout = io.StringIO()
            try:
                beats, downbeats = f2b(path)
            finally:
                sys.stdout = _stdout

            if len(beats) < 8:
                rec.update(bpm=None, confidence="none", reason="fewer than 8 beats found")
            else:
                ib = float(np.median(np.diff(beats)))
                bpm = 60.0 / ib if ib > 0 else 0
                rec["bpm_raw"] = round(bpm, 1)

                # independent second opinion
                import librosa
                y, sr = librosa.load(path, sr=22050, mono=True)
                # Current librosa returns tempo as an ndarray, not a scalar.
                lb = float(np.atleast_1d(librosa.beat.beat_track(y=y, sr=sr)[0])[0])
                rec["bpm_librosa"] = round(lb, 1)

                agree = abs(bpm - lb) <= 3.0
                ratio = (max(bpm, lb) / min(bpm, lb)) if min(bpm, lb) > 0 else 0
                octave = abs(ratio - 2.0) < 0.12

                # A tracker reporting 70 on a trap track and 70 on a doom folk
                # track is right once. Flag the range where doubling is
                # plausible instead of pretending to know which.
                halftime_suspect = bpm < 90
                if not (BPM_MIN <= bpm <= BPM_MAX):
                    rec.update(bpm=None, confidence="none",
                               reason=f"{bpm:.0f} outside the model's {BPM_MIN}-{BPM_MAX} range")
                elif agree and not halftime_suspect:
                    rec.update(bpm=int(round(bpm)), confidence="high",
                               reason="two trackers agree")
                elif agree and halftime_suspect:
                    rec.update(bpm=int(round(bpm)), confidence="medium",
                               reason=f"trackers agree, but {bpm:.0f} may be half of {bpm*2:.0f}")
                elif octave:
                    rec.update(bpm=None, confidence="none",
                               reason=f"octave ambiguity: {bpm:.0f} vs {lb:.0f}")
                else:
                    rec.update(bpm=None, confidence="none",
                               reason=f"trackers disagree: {bpm:.0f} vs {lb:.0f}")

            ts, raw, why = beats_to_meter(beats, downbeats)
            rec["timesignature"] = ts
            rec["beats_per_bar_raw"] = raw
            if why:
                rec["ts_reason"] = why

            name, score, margin, method = estimate_key(path)
            rec["key_corr"] = score
            rec["key_margin"] = margin
            rec["key_method"] = method
            rec["keyscale"] = name
            rec["key_confidence"] = ("high" if method == "clear margin"
                                     else "medium" if name else "none")

            ub = user_bpm(s["filename"])
            if ub:
                rec["bpm_stated_by_owner"] = ub

        except Exception as e:                       # one bad file must not end the run
            rec.update(bpm=None, keyscale=None, timesignature=None,
                       confidence="none", reason=f"{type(e).__name__}: {e}")

        out.append(rec)
        if i % 10 == 0 or i == len(samples):
            el = time.time() - t0
            print(f"  {i}/{len(samples)}  {el/i:.1f}s/track  eta {(len(samples)-i)*el/i/60:.0f}m",
                  flush=True)

    json.dump({"generated": time.strftime("%Y-%m-%d %H:%M:%S"),
               "tool": "beat_this final0 + librosa K-S",
               "labels": out}, io.open(args.out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
