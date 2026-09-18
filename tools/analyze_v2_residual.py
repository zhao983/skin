from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(r"D:\skin")
RECTS = ROOT / "_v2_diagnostics" / "knight_candidate_rects.csv"


def main() -> None:
    default = Image.open(ROOT / "Default" / "Knight.png").convert("RGBA")
    v2 = Image.open(ROOT / "nailong_v2" / "Knight.png").convert("RGBA")
    skadi = Image.open(ROOT / "斯卡蒂" / "Knight.png").convert("RGBA")
    rows = list(csv.DictReader(RECTS.open(encoding="utf-8")))
    residual = []
    for row in rows:
        box = tuple(int(row[k]) for k in ("x0", "y0", "x1", "y1"))
        a = np.asarray(default.crop(box)).astype(np.int16)
        b = np.asarray(v2.crop(box)).astype(np.int16)
        visible = np.maximum(a[:, :, 3], b[:, :, 3]) > 8
        changed = np.any(a != b, axis=2) & visible
        n = int(changed.sum())
        if n < 20:
            row["v2_changed_pixels"] = n
            residual.append(row)
    out = ROOT / "_v2_diagnostics"
    if residual:
        with (out / "v2_residual_candidates.csv").open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(residual[0].keys()))
            w.writeheader(); w.writerows(residual)
    print(f"candidate_rects={len(rows)} residual_unchanged={len(residual)}")
    print("ids", ",".join(row["id"] for row in residual))

    cell_w, cell_h, cols, rows_per_page = 420, 190, 3, 5
    per_page = cols * rows_per_page
    font = ImageFont.load_default()
    for page_no in range((len(residual) + per_page - 1) // per_page):
        sheet = Image.new("RGB", (cell_w * cols, cell_h * rows_per_page), (28, 31, 35))
        draw = ImageDraw.Draw(sheet)
        for n, row in enumerate(residual[page_no * per_page:(page_no + 1) * per_page]):
            col, rr = n % cols, n // cols
            ox, oy = col * cell_w, rr * cell_h
            box = tuple(int(row[k]) for k in ("x0", "y0", "x1", "y1"))
            for ci, src in enumerate((default, v2, skadi)):
                crop = src.crop(box)
                scale = min(1.0, 125 / max(crop.size))
                crop = crop.resize((max(1, round(crop.width * scale)), max(1, round(crop.height * scale))), Image.Resampling.LANCZOS)
                panel = Image.new("RGB", (132, 145), (47, 51, 57))
                panel.paste(crop, ((132-crop.width)//2, (145-crop.height)//2), crop)
                sheet.paste(panel, (ox + 4 + ci*138, oy + 34))
            draw.text((ox+5, oy+6), f'ID {row["id"]} ({row["x0"]},{row["y0"]}) {row["w"]}x{row["h"]}', fill=(245,245,245), font=font)
            draw.text((ox+5, oy+20), "Default | v2 | Skadi", fill=(180,190,205), font=font)
        sheet.save(out / f"v2_residual_{page_no:02d}.png", optimize=True)


if __name__ == "__main__":
    main()
