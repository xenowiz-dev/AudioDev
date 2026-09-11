"""Show exactly which tracks the sibling-take rule merges, and why.

Collapsing two different songs into one is a silent dataset bug: you lose a
track and never find out. This prints every group the rule formed so the
merges can be eyeballed before anything trains on them.
"""
import sys
from collections import defaultdict
import os

# Root discovery: AUDIODEV_ROOT if the launcher set it, else relative to
# this file. Every path below derives from it, so the tree moves as one.
_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("AUDIODEV_ROOT") or os.path.normpath(os.path.join(_HERE, ".."))

sys.path.insert(0, os.path.join(ROOT, "train"))
from build_dataset import (dedup_key, norm_title, parse_sidecar, rank, scan,  # noqa: E402
                           probe)
from collections import Counter                                    # noqa: E402

SRC = [os.environ.get("SUNO_DIR", r"C:\Users\Kevin\Downloads\Suno Playlist 8-11-2026")]

drops = Counter()
found = scan(SRC, drops)
for s in found:
    s["real_duration"] = s["duration"]      # metadata is enough for this audit

by_id = {}
for s in found:
    by_id.setdefault(s["id"] or s["filename"], s)

groups = defaultdict(list)
for s in by_id.values():
    groups[dedup_key(s)].append(s)

merged = {k: v for k, v in groups.items() if len(v) > 1}
print(f"{len(by_id)} unique clips -> {len(groups)} songs "
      f"({len(merged)} groups merged more than one take)\n")

suspicious = 0
for k, v in sorted(merged.items(), key=lambda kv: -len(kv[1])):
    v.sort(key=rank, reverse=True)
    titles = {s["title"] for s in v}
    caps = {s["caption"][:60] for s in v}
    # Same normalized key but genuinely different titles or wildly different
    # style prompts is the signature of an over-merge.
    flag = "  <-- CHECK" if len(titles) > 1 and len(caps) > 1 else ""
    if flag:
        suspicious += 1
    print(f"[{k}]  {len(v)} takes{flag}")
    for i, s in enumerate(v):
        mark = "KEEP" if i == 0 else " drop"
        print(f"   {mark}  plays={s['play_count']:4d} up={s['upvote_count']:3d} "
              f"{s['duration']:6.1f}s  {s['title'][:44]:44s} | {s['filename'][:38]}")
    print()

print(f"groups worth a human look: {suspicious}")
