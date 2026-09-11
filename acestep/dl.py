import os
from huggingface_hub import snapshot_download
import os

# Root discovery: AUDIODEV_ROOT if the launcher set it, else relative to
# this file. Every path below derives from it, so the tree moves as one.
_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("AUDIODEV_ROOT") or os.path.normpath(os.path.join(_HERE, ".."))
d = snapshot_download("ACE-Step/Ace-Step1.5",
      local_dir=os.path.join(ROOT, "acestep", "checkpoints"),
      allow_patterns=["config.json","vae/*","acestep-v15-turbo/*","Qwen3-Embedding-0.6B/*"])
print("downloaded to", d)
for root,_,fs in os.walk(d):
    for f in fs:
        p=os.path.join(root,f)
        if os.path.getsize(p) > 50_000_000:
            print("  %8.1f MB  %s" % (os.path.getsize(p)/1e6, os.path.relpath(p,d)))
