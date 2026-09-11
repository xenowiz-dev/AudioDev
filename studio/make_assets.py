"""Generate the studio's static assets: app icon and phone-access QR.

Run from the watermark venv (has Pillow via matplotlib; qrcode installed
alongside). Re-run if the tailnet URL ever changes.

    B:\\AudioDev\\watermark\\.venv\\Scripts\\python.exe make_assets.py
"""

import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "assets")

INK_BG = (20, 20, 22, 255)        # #141416, the tool-wide panel colour
AMBER = (245, 197, 66, 255)       # cliff-line accent
TEAL = (79, 209, 197, 255)        # selection accent


def icon(path, size=512):
    """Rounded dark tile with a waveform in the house accent colours.

    512px because `pwa=True` builds the manifest from the favicon and a
    home-screen icon wants at least that; browsers downscale for the tab.
    """
    from PIL import Image, ImageDraw

    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([0, 0, size - 1, size - 1], radius=size // 5,
                        fill=INK_BG)

    heights = [0.30, 0.52, 0.40, 0.78, 0.58, 0.95, 0.66, 0.84, 0.46,
               0.60, 0.34]
    n = len(heights)
    pad = size * 0.16
    span = size - 2 * pad
    bar_w = span / (n * 1.7)
    gap = (span - n * bar_w) / (n - 1)
    mid = size / 2
    for i, h in enumerate(heights):
        x0 = pad + i * (bar_w + gap)
        half = h * (size * 0.315)
        color = TEAL if i == 5 else AMBER
        d.rounded_rectangle([x0, mid - half, x0 + bar_w, mid + half],
                            radius=bar_w / 2, fill=color)
    img.save(path)
    print("wrote", path)


def qr(path, url):
    """Classic dark-on-light QR -- the reliable polarity for phone cameras."""
    import qrcode
    from PIL import Image, ImageDraw

    q = qrcode.QRCode(border=2, box_size=10,
                      error_correction=qrcode.constants.ERROR_CORRECT_M)
    q.add_data(url)
    q.make(fit=True)
    code = q.make_image(fill_color="#141416", back_color="#ffffff").convert(
        "RGBA")

    # A soft frame so it sits nicely on the dark UI.
    padded = Image.new("RGBA", (code.width + 48, code.height + 48),
                       (255, 255, 255, 255))
    padded.paste(code, (24, 24))
    d = ImageDraw.Draw(padded)
    d.rounded_rectangle([0, 0, padded.width - 1, padded.height - 1],
                        radius=24, outline="#d9d9de", width=2)
    padded.save(path)
    print("wrote", path, "->", url)


def tailnet_url():
    try:
        import json
        out = subprocess.run(["tailscale", "status", "--json"],
                             capture_output=True, text=True, timeout=15)
        dns = (json.loads(out.stdout).get("Self", {}).get("DNSName")
               or "").rstrip(".")
        return f"https://{dns}" if dns else None
    except Exception:
        return None


if __name__ == "__main__":
    os.makedirs(ASSETS, exist_ok=True)
    icon(os.path.join(ASSETS, "icon.png"))
    url = tailnet_url()
    if url:
        qr(os.path.join(ASSETS, "phone_qr.png"), url)
        with open(os.path.join(ASSETS, "phone_url.txt"), "w") as fh:
            fh.write(url)
    else:
        print("tailscale not detected -- skipped the QR")
