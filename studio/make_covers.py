"""Render cover art for every track that lacks one.

Run in the media venv. Safe to re-run: existing covers are skipped unless
--force, so this doubles as the backfill for tracks that predate the library.

    B:\\AudioDev\\watermark\\.venv\\Scripts\\python.exe make_covers.py [--force]
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import art


def main():
    force = "--force" in sys.argv
    files = []
    for ext in (".wav", ".flac"):
        import glob
        files += glob.glob(os.path.join(art.OUTDIR, f"*{ext}"))
    files.sort()

    made = skipped = failed = 0
    t0 = time.time()
    for i, f in enumerate(files, 1):
        name = os.path.basename(f)
        if not force and os.path.exists(art.cover_path(f)):
            skipped += 1
            continue
        try:
            out = art.ensure_cover(f, force=force)
            made += bool(out)
            failed += not out
            print(f"[{i}/{len(files)}] {'ok ' if out else 'FAIL'} {name}",
                  flush=True)
        except Exception as e:
            failed += 1
            print(f"[{i}/{len(files)}] FAIL {name}: "
                  f"{type(e).__name__}: {e}", flush=True)

    print(f"\n{made} rendered, {skipped} already had one, {failed} failed "
          f"in {time.time() - t0:.0f}s")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
