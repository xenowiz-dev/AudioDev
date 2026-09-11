"""
Dump a RIFF/WAV file's chunk structure.

The classifier only ever sees decoded PCM, so nothing in the container can
influence it -- but the container is still worth reading. Unexpected chunks are
where metadata, editor breadcrumbs, and provenance manifests actually live, and
a size mismatch between the declared and actual data is worth knowing about.

Usage:
    python wav_chunks.py file.wav
"""

import os
import struct
import sys

KNOWN = {
    b"fmt ": "format",
    b"data": "PCM samples",
    b"LIST": "metadata list (INFO tags etc)",
    b"id3 ": "ID3 tag block",
    b"ID3 ": "ID3 tag block",
    b"bext": "Broadcast Wave extension (originator, date, coding history)",
    b"iXML": "iXML metadata (production/recorder info)",
    b"_PMX": "XMP metadata",
    b"cue ": "cue points",
    b"fact": "fact chunk",
    b"smpl": "sampler chunk",
    b"acid": "ACID loop metadata",
    b"JUNK": "padding",
    b"PAD ": "padding",
    b"c2pa": "C2PA provenance manifest",
    b"jumb": "JUMBF box (may carry C2PA)",
}


def main(path):
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        riff = f.read(4)
        declared = struct.unpack("<I", f.read(4))[0]
        wave = f.read(4)
        print(f"{os.path.basename(path)}")
        print(f"  {riff.decode(errors='replace')}/{wave.decode(errors='replace')}"
              f"  declared {declared + 8} bytes, actual {size} bytes"
              f"{'  <-- MISMATCH' if declared + 8 != size else ''}")
        print()
        print(f"  {'chunk':<8} {'size':>12}  {'offset':>12}  meaning")
        total = 12
        while True:
            hdr = f.read(8)
            if len(hdr) < 8:
                break
            cid, csz = hdr[:4], struct.unpack("<I", hdr[4:])[0]
            off = f.tell() - 8
            meaning = KNOWN.get(cid, "** UNKNOWN **")
            print(f"  {cid.decode(errors='replace'):<8} {csz:>12}  {off:>12}  {meaning}")
            if cid == b"fmt ":
                raw = f.read(csz)
                if csz >= 16:
                    (afmt, ch, sr, br, ba, bits) = struct.unpack("<HHIIHH", raw[:16])
                    names = {1: "PCM", 3: "IEEE float", 0xFFFE: "extensible"}
                    print(f"           format={names.get(afmt, afmt)} ch={ch} "
                          f"sr={sr} bits={bits} byterate={br}")
            elif cid in (b"LIST", b"id3 ", b"ID3 ", b"bext", b"iXML", b"_PMX"):
                raw = f.read(csz)
                txt = raw.decode("latin-1", errors="replace")
                printable = "".join(c if 32 <= ord(c) < 127 else "." for c in txt)
                print(f"           {printable[:300]}")
            else:
                f.seek(csz, 1)
            if csz % 2:
                f.seek(1, 1)
            total += 8 + csz + (csz % 2)
        trailing = size - total
        if trailing > 0:
            print(f"\n  {trailing} trailing bytes after the last chunk "
                  f"<-- data hidden past the chunk list")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        main(p)
        print()
