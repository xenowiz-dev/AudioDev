"""Resolve half/double-time ambiguity in labels.json.

The two trackers found tempi in a 2:1 relationship: they agree on the beat
GRID and disagree only on which level counts as the beat. That is not a
measurement failure, it is a genuine musical ambiguity -- 70 BPM with two
subdivisions and 140 BPM are the same pulse described two ways.

RULE: prefer the candidate nearest 120 BPM, the perceptual centre of the
tempo octave. This is the standard MIR convention and it is what the one piece
of ground truth we have supports: "Factory Soul" states 135 BPM in its Suno
prompt, the candidates were 68.2 and 136.0, and nearest-120 picks 136.0.

Confidence is graded by how decisive the choice is, because the rule is a
convention rather than a measurement. A pair like 81 / 161.5 sits almost
equidistant from 120 and is a coin flip; it stays MEDIUM so it can be excluded
later without re-running the 40-minute labelling pass.

Non-octave disagreements are left alone. When two trackers report unrelated
tempi neither is corroborated, and a guess would bind the trigger token to a
tempo the audio does not have.
"""
import io
import json
import shutil
import os

# Root discovery: AUDIODEV_ROOT if the launcher set it, else relative to
# this file. Every path below derives from it, so the tree moves as one.
_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("AUDIODEV_ROOT") or os.path.normpath(os.path.join(_HERE, ".."))

PATH = os.path.join(ROOT, "train", "labels.json")
CENTRE = 120.0
DECISIVE = 20.0      # BPM difference in distance-to-centre to call it clear


def main():
    d = json.load(io.open(PATH, encoding="utf-8"))
    shutil.copyfile(PATH, PATH + ".pre-octave")

    # ---- rule 2: jitter, not disagreement -----------------------------
    # Two trackers within 5% of each other have AGREED; the gap is estimation
    # noise, not a dispute about the pulse. The 3 BPM absolute threshold used
    # during labelling was too tight at speed -- it rejected 130.4 vs 136.0 on
    # a track whose Suno prompt states 134, which is what both were circling.
    # The mean is the better estimate than either.
    jitter = 0
    for r in d["labels"]:
        if "trackers disagree" not in (r.get("reason") or ""):
            continue
        a_, b_ = r.get("bpm_raw"), r.get("bpm_librosa")
        if not a_ or not b_:
            continue
        if abs(a_ - b_) / max(a_, b_) <= 0.05:
            mean = (a_ + b_) / 2
            r["bpm"] = int(round(mean))
            r["confidence"] = "high"
            r["reason"] = (f"trackers within 5% ({a_:.0f}/{b_:.0f}); "
                           f"mean {mean:.0f} taken as the estimate")
            jitter += 1

    resolved = marginal = left = 0
    for r in d["labels"]:
        if "octave ambiguity" not in (r.get("reason") or ""):
            continue
        a, b = r.get("bpm_raw"), r.get("bpm_librosa")
        if not a or not b:
            left += 1
            continue
        da, db = abs(a - CENTRE), abs(b - CENTRE)
        pick, other = (a, b) if da < db else (b, a)
        decisive = abs(da - db) >= DECISIVE
        r["bpm"] = int(round(pick))
        r["bpm_octave_rejected"] = round(other, 1)
        r["confidence"] = "high" if decisive else "medium"
        r["reason"] = (f"octave pair {min(a,b):.0f}/{max(a,b):.0f}; "
                       f"nearest-{CENTRE:.0f} rule picked {pick:.0f}"
                       + ("" if decisive else " (marginal — near-equidistant)"))
        if decisive:
            resolved += 1
        else:
            marginal += 1

    json.dump(d, io.open(PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    total = sum(1 for r in d["labels"] if r.get("bpm"))
    print(f"jitter -> mean      : {jitter}")
    print(f"resolved decisively : {resolved}")
    print(f"resolved marginally : {marginal}  (confidence=medium)")
    print(f"left unresolved     : {left}")
    print(f"BPM coverage now    : {total}/{len(d['labels'])}"
          f"  ({100*total//len(d['labels'])}%)")
    print(f"backup              : {PATH}.pre-octave")


if __name__ == "__main__":
    main()
