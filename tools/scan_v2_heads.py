from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from analyze_atlas_rects import run_components


ROOT = Path(r"D:\skin")
V2 = ROOT / "nailong_v2"
ATLASES = ["Knight.png", "Sprint.png", "Cloak.png", "Shriek.png", "Wings.png", "Webbed.png", "DreamArrival.png", "Dreamnail.png", "Birthplace.png"]


def head_boxes(image: Image.Image):
    arr = np.asarray(image.convert("RGBA"))
    r, g, b, a = [arr[:, :, i].astype(np.int16) for i in range(4)]
    yellow = (a > 20) & (r > 238) & (g > 188) & (b < 85) & (r > g * 1.08)
    green = (a > 20) & (g > 65) & (g > r * 1.2) & (g > b * 1.25)
    out = []
    for x0, y0, x1, y1, count in run_components(yellow):
        w, h = x1 - x0, y1 - y0
        if count < 18 or w < 4 or h < 4 or w > 180 or h > 180:
            continue
        if max(w / h, h / w) > 3.0 or count / (w * h) < 0.10:
            continue
        pad = max(4, round(max(w, h) * 0.32))
        gx0, gy0 = max(0, x0-pad), max(0, y0-pad)
        gx1, gy1 = min(image.width, x1+pad), min(image.height, y1+pad)
        if int(green[gy0:gy1, gx0:gx1].sum()) < 1:
            continue
        out.append((x0, y0, x1, y1, count))
    return out


def main() -> None:
    total = 0
    for name in ATLASES:
        boxes = head_boxes(Image.open(V2 / name))
        total += len(boxes)
        print(f"{name}: {len(boxes)}")
    print(f"total: {total}")


if __name__ == "__main__":
    main()
