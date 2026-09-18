from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(r"D:\skin")
CSV = ROOT / "_v3_diagnostics" / "v3_transformed_components.csv"
OUT = ROOT / "_v3_diagnostics" / "semantic_qa"


def render_panel(image: Image.Image, box: tuple[int, int, int, int], size=(130, 120)) -> Image.Image:
    crop = image.crop(box)
    scale = min(1.0, (size[0] - 8) / crop.width, (size[1] - 8) / crop.height)
    crop = crop.resize((max(1, round(crop.width * scale)), max(1, round(crop.height * scale))), Image.Resampling.LANCZOS)
    panel = Image.new("RGB", size, (48, 52, 58))
    panel.paste(crop, ((size[0] - crop.width) // 2, (size[1] - crop.height) // 2), crop)
    return panel


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(CSV.open(encoding="utf-8-sig")))
    atlases = {}
    font = ImageFont.load_default()
    for semantic in sorted({r["semantic"] for r in rows}):
        group = [r for r in rows if r["semantic"] == semantic]
        per_page, cols, cell_w, cell_h = 12, 3, 410, 160
        for page in range((len(group) + per_page - 1) // per_page):
            batch = group[page * per_page:(page + 1) * per_page]
            sheet = Image.new("RGB", (cols * cell_w, 4 * cell_h), (28, 31, 35))
            draw = ImageDraw.Draw(sheet)
            for i, row in enumerate(batch):
                atlas_name = row["atlas"]
                if atlas_name not in atlases:
                    atlases[atlas_name] = (
                        Image.open(ROOT / "Default" / atlas_name).convert("RGBA"),
                        Image.open(ROOT / "nailong_v2" / atlas_name).convert("RGBA"),
                        Image.open(ROOT / "nailong_v3" / atlas_name).convert("RGBA"),
                    )
                x, y, w, h = [int(row[k]) for k in ("x", "y", "w", "h")]
                pad = 12
                base = atlases[atlas_name][0]
                box = (max(0, x-pad), max(0, y-pad), min(base.width, x+w+pad), min(base.height, y+h+pad))
                ox, oy = (i % cols) * cell_w, (i // cols) * cell_h
                for j, atlas in enumerate(atlases[atlas_name]):
                    sheet.paste(render_panel(atlas, box), (ox + j * 134, oy + 28))
                draw.text((ox + 4, oy + 4), f"{atlas_name} ({x},{y}) {w}x{h} donor={row['donor']}", font=font, fill=(245,245,245))
                draw.text((ox + 4, oy + 16), "Default             v2                  v3", font=font, fill=(175,188,206))
            sheet.save(OUT / f"{semantic}_{page:02d}.jpg", quality=92)
    print(f"wrote semantic QA pages to {OUT}")


if __name__ == "__main__":
    main()
