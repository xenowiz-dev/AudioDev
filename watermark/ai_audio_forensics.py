"""
AI-audio forensics: check a file across all three provenance layers.

  1. PASSIVE  - vocoder "fakeprint" classifier (lofcz/ai-music-detector).
                Needs no cooperation from the generator: it looks for the
                periodic spectral peaks that transposed-convolution upsampling
                layers leave behind. Trained for Suno <=5 / Udio <=1.5.
  2. ACTIVE   - AudioSeal watermark detector (Meta, MIT). Only fires if
                someone deliberately embedded an AudioSeal watermark.
  3. METADATA - container tags (ID3/Vorbis/etc) and a byte-scan for a C2PA
                manifest. Trivially stripped, but free to check.

Usage:
    python ai_audio_forensics.py track.wav [more.wav ...]
    python ai_audio_forensics.py --json track.wav
"""

import argparse
import json
import os
import sys

# AudioSeal wraps its SEANet blocks in torch.compile, which needs Triton and so
# fails on Windows. Must be set before audioseal is imported.
os.environ.setdefault("NO_TORCH_COMPILE", "1")

import numpy as np


# ---------------------------------------------------------------- layer 1
class FakeprintDetector:
    """Re-implementation of the ISMIR-2025 'Fourier explanation of AI-music
    artifacts' feature, matching lofcz/ai-music-detector's extractor exactly.

    Neural vocoders upsample with transposed convolutions, which stamp evenly
    spaced peaks into the spectrum. A minimum filter estimates the local noise
    floor; subtracting it isolates those peaks. Real recordings have no such
    comb structure.
    """

    SR = 16000
    N_FFT = 8192
    HOP = 4096          # torchaudio Spectrogram default: n_fft // 2
    F_MIN, F_MAX = 1000, 8000
    HULL = 10
    MIN_DB, MAX_DB = -45.0, 5.0
    MAX_SECONDS = 300

    def __init__(self, onnx_path):
        import onnxruntime as ort
        import torchaudio
        self.sess = ort.InferenceSession(onnx_path,
                                         providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name
        self.n_features = self.sess.get_inputs()[0].shape[1]
        # Must mirror the reference extractor's ops, not merely its maths: the
        # feature band's top edge (8000 Hz) sits exactly at Nyquist for a
        # 16 kHz signal, so a different resampler's transition band lands
        # inside the features and shifts every score.
        self.stft = torchaudio.transforms.Spectrogram(n_fft=self.N_FFT,
                                                      power=2,
                                                      normalized=False)
        self._resamplers = {}

    def _load(self, path):
        import torch
        import torchaudio
        try:
            audio, sr = torchaudio.load(path)
        except Exception:
            import soundfile as sf
            x, sr = sf.read(path, always_2d=True, dtype="float32")
            audio = torch.from_numpy(x.T)
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        if sr != self.SR:
            if sr not in self._resamplers:
                self._resamplers[sr] = torchaudio.transforms.Resample(sr, self.SR)
            audio = self._resamplers[sr](audio)
        max_samples = self.MAX_SECONDS * self.SR
        return audio[:, :max_samples]

    def fakeprint(self, path):
        import torch
        from scipy.ndimage import minimum_filter1d

        audio = self._load(path)
        if audio.shape[1] < self.N_FFT:
            raise ValueError("clip shorter than one FFT window")

        with torch.no_grad():
            S = self.stft(audio)
        S_db = 10 * torch.log10(torch.clamp(S, min=1e-10, max=1e6))
        mean_spec = S_db.mean(dim=(0, 2)).numpy()

        freqs = np.linspace(0, self.SR / 2, self.N_FFT // 2 + 1)
        mask = (freqs >= self.F_MIN) & (freqs <= self.F_MAX)
        spec = mean_spec[mask]

        hull = minimum_filter1d(spec, size=self.HULL, mode="nearest")
        hull = np.clip(hull, self.MIN_DB, None)
        residue = np.clip(spec - hull, 0, None)
        residue = np.clip(residue, 0, self.MAX_DB)
        fp = (residue / (residue.max() + 1e-6)).astype(np.float32)
        return fp, freqs[mask]

    def predict(self, path):
        fp, freqs = self.fakeprint(path)
        if len(fp) != self.n_features:
            raise ValueError(f"got {len(fp)} features, model wants "
                             f"{self.n_features}")
        p = float(self.sess.run(None, {self.input_name: fp[None, :]})[0][0, 0])

        # Comb strength: how periodic the residual peaks are. Reported as
        # supporting evidence, not as the decision.
        spec = fp - fp.mean()
        ac = np.correlate(spec, spec, mode="full")[len(spec) - 1:]
        ac = ac / (ac[0] + 1e-12)
        comb = float(ac[3:400].max()) if len(ac) > 400 else float("nan")
        return p, comb


# ---------------------------------------------------------------- layer 2
def audioseal_detect(path):
    """Meta AudioSeal. Returns (probability, decoded_16bit_message or None)."""
    import torch
    import soundfile as sf
    from audioseal import AudioSeal

    import librosa

    x, sr = sf.read(path, always_2d=True, dtype="float32")
    mono = x.mean(axis=1)
    # AudioSeal 0.2+ does NOT resample internally -- passing sample_rate= is a
    # no-op and the model silently sees the wrong rate. The 16-bit models are
    # trained at 16 kHz, so resample here.
    if sr != 16000:
        mono = librosa.resample(mono, orig_sr=sr, target_sr=16000)
    wav = torch.from_numpy(mono)[None, None, :]

    detector = AudioSeal.load_detector("audioseal_detector_16bits")
    with torch.no_grad():
        result, message = detector.detect_watermark(wav, sample_rate=16000)

    prob = float(result)
    # The decoder always emits 16 bits; they are noise unless a watermark is
    # actually present, so don't report a message that means nothing.
    msg = None
    if prob > 0.5 and message is not None:
        bits = message.squeeze().tolist()
        if isinstance(bits, list):
            msg = "".join(str(int(round(b))) for b in bits)
    return prob, msg


# ---------------------------------------------------------------- layer 3
AI_TAG_HINTS = ("suno", "udio", "stable audio", "musicgen", "lyria",
                "elevenlabs", "generated by ai", "ai-generated", "ai generated",
                "synthid", "c2pa", "riffusion", "mubert", "boomy", "soundraw")


def read_riff_info(path):
    """Parse RIFF LIST/INFO, bext and iXML chunks out of a WAV.

    mutagen does NOT surface these for a plain PCM WAV -- it handles WAVE-with-
    ID3 only -- so relying on it alone silently misses the single most common
    place a generator stamps its name. This exact gap hid a literal
    'made with suno' comment during one investigation while elaborate signal
    analysis was run instead. Always read the container directly.
    """
    import struct
    tags = {}
    try:
        with open(path, "rb") as f:
            if f.read(4) != b"RIFF":
                return tags
            f.read(4)
            if f.read(4) != b"WAVE":
                return tags
            while True:
                hdr = f.read(8)
                if len(hdr) < 8:
                    break
                cid, csz = hdr[:4], struct.unpack("<I", hdr[4:])[0]
                if cid in (b"LIST", b"bext", b"iXML", b"_PMX", b"id3 ", b"ID3 "):
                    raw = f.read(csz)
                    if cid == b"LIST" and raw[:4] == b"INFO":
                        pos = 4
                        while pos + 8 <= len(raw):
                            sid = raw[pos:pos + 4]
                            ssz = struct.unpack("<I", raw[pos + 4:pos + 8])[0]
                            val = raw[pos + 8:pos + 8 + ssz]
                            tags[sid.decode("latin-1").strip()] = \
                                val.decode("latin-1", errors="replace").strip("\x00").strip()
                            pos += 8 + ssz + (ssz % 2)
                    else:
                        txt = raw.decode("latin-1", errors="replace")
                        txt = "".join(c if 32 <= ord(c) < 127 else " " for c in txt)
                        if txt.strip():
                            tags[cid.decode("latin-1").strip()] = txt.strip()[:300]
                else:
                    f.seek(csz, 1)
                if csz % 2:
                    f.seek(1, 1)
    except Exception:
        pass
    return tags


def read_metadata(path):
    """Container tags plus a byte-scan for a C2PA manifest."""
    found_tags, hints = {}, []
    try:
        import mutagen
        f = mutagen.File(path)
        if f is not None and f.tags:
            for k, v in dict(f.tags).items():
                found_tags[str(k)] = str(v)[:300]
    except Exception as e:
        found_tags["<mutagen error>"] = f"{type(e).__name__}: {e}"

    # RIFF chunks mutagen won't show us.
    found_tags.update(read_riff_info(path))

    for k, s in found_tags.items():
        low = (str(k) + " " + str(s)).lower()
        for h in AI_TAG_HINTS:
            if h in low:
                hints.append(f"{k}={s}")
                break

    # C2PA manifests live in a JUMBF box; the ASCII markers survive in the
    # raw bytes regardless of container. Presence != valid signature --
    # use c2patool to actually verify the claim chain.
    c2pa = False
    try:
        with open(path, "rb") as fh:
            head = fh.read(4 * 1024 * 1024)
            fh.seek(max(0, os.path.getsize(path) - 4 * 1024 * 1024))
            tail = fh.read(4 * 1024 * 1024)
        blob = head + tail
        c2pa = (b"jumb" in blob and b"c2pa" in blob) or b"c2pa.claim" in blob
    except Exception:
        pass
    return found_tags, sorted(set(hints)), c2pa


# ---------------------------------------------------------------- report
def analyse(path, detector):
    import soundfile as sf
    info = sf.info(path)
    row = {"file": os.path.basename(path), "sample_rate": info.samplerate,
           "channels": info.channels, "duration_s": round(info.duration, 2)}

    try:
        p, comb = detector.predict(path)
        row["ai_probability"] = round(p, 4)
        row["comb_periodicity"] = round(comb, 3)
        row["passive_verdict"] = "AI-GENERATED" if p >= 0.5 else "real"
    except Exception as e:
        row["passive_verdict"] = f"error: {type(e).__name__}: {e}"

    try:
        wp, msg = audioseal_detect(path)
        row["audioseal_prob"] = round(wp, 4)
        row["audioseal_verdict"] = "WATERMARK PRESENT" if wp > 0.5 else "none"
        if msg:
            row["audioseal_message"] = msg
    except Exception as e:
        row["audioseal_verdict"] = f"error: {type(e).__name__}: {e}"

    tags, hints, c2pa = read_metadata(path)
    row["n_tags"] = len(tags)
    row["metadata_ai_hints"] = hints or None
    row["c2pa_manifest"] = c2pa
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--model", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "models", "ai_music_detector.onnx"))
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    det = FakeprintDetector(args.model)
    rows = [analyse(f, det) for f in args.files]

    if args.json:
        print(json.dumps(rows, indent=2))
        return 0

    for r in rows:
        print(f"\n{r['file']}")
        print(f"  {r['sample_rate']} Hz, {r['channels']} ch, "
              f"{r['duration_s']}s")
        pv = r.get("passive_verdict", "?")
        if "ai_probability" in r:
            print(f"  [1] vocoder fakeprint : {pv}  "
                  f"(p={r['ai_probability']:.4f}, "
                  f"comb={r['comb_periodicity']:.3f})")
        else:
            print(f"  [1] vocoder fakeprint : {pv}")
        av = r.get("audioseal_verdict", "?")
        extra = f", msg={r['audioseal_message']}" if "audioseal_message" in r else ""
        if "audioseal_prob" in r:
            print(f"  [2] AudioSeal        : {av}  "
                  f"(p={r['audioseal_prob']:.4f}{extra})")
        else:
            print(f"  [2] AudioSeal        : {av}")
        print(f"  [3] metadata         : {r['n_tags']} tags, "
              f"C2PA manifest: {r['c2pa_manifest']}, "
              f"AI hints: {r['metadata_ai_hints'] or 'none'}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
