"""Check dataset.json against ACE-Step's OWN loaders. No GPU, no training.

Everything here is the code the trainer will actually run, imported directly:
if `discover_audio_files` cannot see a file, preprocessing will not see it
either. That is the point -- validating against my own idea of the schema
would only prove I am self-consistent.

    python validate_dataset.py
"""
import json
import os
import sys

# Root discovery: AUDIODEV_ROOT if the launcher set it, else relative to
# this file. Every path below derives from it, so the tree moves as one.
_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("AUDIODEV_ROOT") or os.path.normpath(os.path.join(_HERE, ".."))

ACE = os.path.join(ROOT, "acestep", "ACE-Step-1.5")
sys.path.insert(0, ACE)

DATASET = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "dataset.json")

from acestep.training_v2.preprocess_discovery import (   # noqa: E402
    discover_audio_files, load_sample_metadata, load_dataset_metadata)
from acestep.training_v2.preprocess_prompt import build_simple_prompt  # noqa: E402

passed = failed = 0


def check(label, ok, detail=""):
    global passed, failed
    if ok:
        passed += 1
        print(f"  PASS  {label}")
    else:
        failed += 1
        print(f"  FAIL  {label}  {detail}")


doc = json.load(open(DATASET, encoding="utf-8"))
samples = doc["samples"]
print(f"dataset: {len(samples)} samples\n")

print("=" * 62)
print("ACE-STEP'S OWN DISCOVERY")
print("=" * 62)
files = discover_audio_files(None, DATASET)
check("every sample is discoverable", len(files) == len(samples),
      f"discovered {len(files)} of {len(samples)}")

meta_map = load_sample_metadata(DATASET, files)
check("every discovered file has metadata", len(meta_map) >= len(files),
      f"{len(meta_map)} entries for {len(files)} files")

ds_meta = load_dataset_metadata(DATASET)
check("dataset-level metadata parses",
      ds_meta.get("tag_position") == "prepend",
      str(ds_meta))
check("trigger word survives the loader",
      bool(ds_meta.get("custom_tag")), f"custom_tag={ds_meta.get('custom_tag')!r}")

print()
print("=" * 62)
print("PROMPT CONSTRUCTION (what the model is actually conditioned on)")
print("=" * 62)
missing_tag = empty = 0
for s in samples:
    # Mirror preprocess.py lines 111-116: the dataset-level tag is a fallback
    # for samples without their own. Testing build_simple_prompt without this
    # tests a code path the trainer never takes.
    sm = dict(s)
    if not sm.get("custom_tag"):
        sm["custom_tag"] = ds_meta.get("custom_tag", "")
    p = build_simple_prompt(sm, tag_position=ds_meta["tag_position"])
    if ds_meta["custom_tag"] and ds_meta["custom_tag"] not in p:
        missing_tag += 1
    if not s.get("caption", "").strip():
        empty += 1
check("every prompt carries the trigger word", missing_tag == 0,
      f"{missing_tag} without it")
check("no empty captions", empty == 0, f"{empty} empty")

_s0 = dict(samples[0])
_s0.setdefault("custom_tag", ds_meta.get("custom_tag", ""))
sample_prompt = build_simple_prompt(_s0, tag_position=ds_meta["tag_position"])
print("\n  --- prompt for the top track, verbatim ---")
for line in sample_prompt.splitlines():
    print("  | " + line[:100])

print()
print("=" * 62)
print("FILES ON DISK")
print("=" * 62)
gone = [s for s in samples if not os.path.exists(s["audio_path"])]
check("every audio_path exists", not gone,
      f"{len(gone)} missing, e.g. {gone[0]['audio_path'] if gone else ''}")
tiny = [s for s in samples if os.path.exists(s["audio_path"])
        and os.path.getsize(s["audio_path"]) < 100_000]
check("no suspiciously small files", not tiny, f"{len(tiny)} under 100 KB")
dupe_paths = len(samples) - len({s["audio_path"] for s in samples})
check("no repeated audio_path", dupe_paths == 0, f"{dupe_paths} repeats")
dupe_ids = [s["_id"] for s in samples if s.get("_id")]
check("no repeated clip id", len(dupe_ids) == len(set(dupe_ids)),
      f"{len(dupe_ids) - len(set(dupe_ids))} repeats")

print()
print("=" * 62)
print("TRAINING SHAPE")
print("=" * 62)
durs = sorted(s["duration"] for s in samples)
hours = sum(durs) / 3600
check("at least 30 tracks (a LoRA needs material)", len(samples) >= 30,
      f"{len(samples)}")
check("at least 1 hour of audio", hours >= 1.0, f"{hours:.2f} h")
print(f"  duration: min {durs[0]:.0f}s  median {durs[len(durs) // 2]:.0f}s  "
      f"max {durs[-1]:.0f}s  total {hours:.2f} h")
rates = {s.get("_sr") for s in samples}
check("one sample rate across the set", len(rates) == 1, f"rates={rates}")

print()
print(f"{passed} passed, {failed} failed")
raise SystemExit(1 if failed else 0)
