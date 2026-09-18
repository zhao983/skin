from __future__ import annotations

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import csv
import hashlib
import numpy as np


ROOT = Path(r"D:\skin")
CSV_PATH = ROOT / "_v2_diagnostics" / "knight_candidate_rects.csv"


def normalized_visible(im: Image.Image) -> Image.Image:
    rgba = np.array(im.convert("RGBA"), copy=True)
    rgba[rgba[:, :, 3] == 0, :3] = 0
    alpha = rgba[:, :, 3]
    ys, xs = np.nonzero(alpha)
    if len(xs) == 0:
        return Image.new("RGBA", (1, 1))
    return Image.fromarray(rgba[ys.min():ys.max()+1, xs.min():xs.max()+1], "RGBA")


def signatures(im: Image.Image):
    out = []
    current = normalized_visible(im)
    for angle in (0, 90, 180, 270):
        rot = current if angle == 0 else current.rotate(angle, expand=True)
        raw = np.asarray(rot.convert("RGBA")).copy()
        raw[raw[:, :, 3] == 0, :3] = 0
        payload = raw.tobytes()
        out.append((hashlib.sha256(payload).hexdigest(), angle, rot.size))
    return out


def main() -> None:
    default = Image.open(ROOT / "Default" / "Knight.png").convert("RGBA")
    failed = Image.open(ROOT / "nailong" / "Knight.png").convert("RGBA")
    skadi = Image.open(ROOT / "斯卡蒂" / "Knight.png").convert("RGBA")
    rows = list(csv.DictReader(CSV_PATH.open(encoding="utf-8")))
    donor_by_sig: dict[str, list[tuple[int, int, tuple[int, int]]]] = {}
    changed_rows = []
    untouched_rows = []
    for row in rows:
        box = tuple(int(row[k]) for k in ("x0", "y0", "x1", "y1"))
        d = np.asarray(default.crop(box)).astype(np.int16)
        f = np.asarray(failed.crop(box)).astype(np.int16)
        visible = np.maximum(d[:, :, 3], f[:, :, 3]) > 8
        changed = np.any(d != f, axis=2) & visible
        changed_n = int(changed.sum())
        total_n = int(visible.sum()) or 1
        row["visible_changed"] = changed_n
        row["visible_changed_ratio"] = round(changed_n / total_n, 5)
        if changed_n >= 20:
            changed_rows.append(row)
            for sig, angle, size in signatures(default.crop(box)):
                donor_by_sig.setdefault(sig, []).append((int(row["id"]), angle, size))
        else:
            untouched_rows.append(row)

    mapped = []
    unresolved = []
    for row in untouched_rows:
        box = tuple(int(row[k]) for k in ("x0", "y0", "x1", "y1"))
        matches = []
        for sig, angle, size in signatures(default.crop(box)):
            for donor_id, donor_angle, donor_size in donor_by_sig.get(sig, []):
                matches.append((donor_id, angle, donor_angle, size, donor_size))
        if matches:
            row["matches"] = ";".join(f"{a}:{b}:{c}" for a,b,c,_,_ in matches[:12])
            mapped.append(row)
        else:
            row["matches"] = ""
            unresolved.append(row)

    out = ROOT / "_v2_diagnostics"
    for name, data in (("changed_candidate_rects.csv", changed_rows), ("untouched_exact_mapped.csv", mapped), ("untouched_unresolved.csv", unresolved)):
        if not data:
            continue
        keys = list(data[0].keys())
        with (out / name).open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader(); w.writerows(data)
    print(f"candidates={len(rows)} changed={len(changed_rows)} untouched={len(untouched_rows)} exact_mapped={len(mapped)} unresolved={len(unresolved)}")
    print("unresolved ids:", ",".join(r["id"] for r in unresolved))

    cell_w, cell_h, cols, rows_per_page = 420, 190, 3, 5
    per_page = cols * rows_per_page
    font = ImageFont.load_default()
    for page_no in range((len(unresolved) + per_page - 1) // per_page):
        sheet = Image.new("RGB", (cell_w * cols, cell_h * rows_per_page), (28, 31, 35))
        draw = ImageDraw.Draw(sheet)
        batch = unresolved[page_no * per_page:(page_no + 1) * per_page]
        for n, row in enumerate(batch):
            col, rr = n % cols, n // cols
            ox, oy = col * cell_w, rr * cell_h
            box = tuple(int(row[k]) for k in ("x0", "y0", "x1", "y1"))
            for ci, src in enumerate((default, failed, skadi)):
                crop = src.crop(box)
                scale = min(1.0, 125 / max(crop.size))
                crop = crop.resize((max(1, round(crop.width * scale)), max(1, round(crop.height * scale))), Image.Resampling.LANCZOS)
                panel = Image.new("RGB", (132, 145), (47, 51, 57))
                panel.paste(crop, ((132-crop.width)//2, (145-crop.height)//2), crop)
                sheet.paste(panel, (ox + 4 + ci*138, oy + 34))
            draw.text((ox+5, oy+6), f'ID {row["id"]} ({row["x0"]},{row["y0"]}) {row["w"]}x{row["h"]}', fill=(245,245,245), font=font)
            draw.text((ox+5, oy+20), 'Default | failed | Skadi', fill=(180,190,205), font=font)
        sheet.save(out / f"unresolved_player_{page_no:02d}.png", optimize=True)

    key_ids = {241, 244, 245, 246, 250, 251, 252, 257, 258, 260, 274, 275, 276, 278, 279, 284, 287, 291, 292, 296, 297, 299, 300, 302, 306, 310, 311, 313, 317}
    key_rows = [r for r in rows if int(r["id"]) in key_ids]
    for page_no in range((len(key_rows) + per_page - 1) // per_page):
        sheet = Image.new("RGB", (cell_w * cols, cell_h * rows_per_page), (28, 31, 35))
        draw = ImageDraw.Draw(sheet)
        batch = key_rows[page_no * per_page:(page_no + 1) * per_page]
        for n, row in enumerate(batch):
            col, rr = n % cols, n // cols
            ox, oy = col * cell_w, rr * cell_h
            box = tuple(int(row[k]) for k in ("x0", "y0", "x1", "y1"))
            for ci, src in enumerate((default, failed, skadi)):
                crop = src.crop(box)
                scale = min(1.0, 125 / max(crop.size))
                crop = crop.resize((max(1, round(crop.width * scale)), max(1, round(crop.height * scale))), Image.Resampling.LANCZOS)
                panel = Image.new("RGB", (132, 145), (47, 51, 57))
                panel.paste(crop, ((132-crop.width)//2, (145-crop.height)//2), crop)
                sheet.paste(panel, (ox + 4 + ci*138, oy + 34))
            draw.text((ox+5, oy+6), f'ID {row["id"]} ({row["x0"]},{row["y0"]}) {row["w"]}x{row["h"]}', fill=(245,245,245), font=font)
            draw.text((ox+5, oy+20), 'Default | failed | Skadi', fill=(180,190,205), font=font)
        sheet.save(out / f"key_frames_{page_no:02d}.png", optimize=True)


if __name__ == "__main__":
    main()
