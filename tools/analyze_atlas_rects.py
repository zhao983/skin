from __future__ import annotations

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import csv


ROOT = Path(r"D:\skin")


def run_components(mask: np.ndarray):
    """4-connected components over horizontal runs; returns bboxes and pixel counts."""
    parent: list[int] = []
    rank: list[int] = []
    runs_by_row: list[list[tuple[int, int, int]]] = []

    def make() -> int:
        i = len(parent)
        parent.append(i)
        rank.append(0)
        return i

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        a, b = find(a), find(b)
        if a == b:
            return
        if rank[a] < rank[b]:
            a, b = b, a
        parent[b] = a
        if rank[a] == rank[b]:
            rank[a] += 1

    previous: list[tuple[int, int, int]] = []
    for y, row in enumerate(mask):
        padded = np.pad(row.astype(np.int8), (1, 1))
        edges = np.diff(padded)
        starts = np.flatnonzero(edges == 1)
        ends = np.flatnonzero(edges == -1) - 1
        current = [(int(s), int(e), make()) for s, e in zip(starts, ends)]
        i = j = 0
        while i < len(current) and j < len(previous):
            cs, ce, cid = current[i]
            ps, pe, pid = previous[j]
            if ce < ps:
                i += 1
            elif pe < cs:
                j += 1
            else:
                union(cid, pid)
                if ce <= pe:
                    i += 1
                else:
                    j += 1
        runs_by_row.append(current)
        previous = current

    stats: dict[int, list[int]] = {}
    for y, runs in enumerate(runs_by_row):
        for s, e, cid in runs:
            root = find(cid)
            if root not in stats:
                stats[root] = [s, y, e + 1, y + 1, e - s + 1]
            else:
                st = stats[root]
                st[0] = min(st[0], s)
                st[1] = min(st[1], y)
                st[2] = max(st[2], e + 1)
                st[3] = max(st[3], y + 1)
                st[4] += e - s + 1
    return stats.values()


def main() -> None:
    default = np.asarray(Image.open(ROOT / "Default" / "Knight.png").convert("RGBA"))
    skadi = np.asarray(Image.open(ROOT / "斯卡蒂" / "Knight.png").convert("RGBA"))
    diff = np.any(default != skadi, axis=2)
    comps = list(run_components(diff))
    rows = []
    for x0, y0, x1, y1, count in comps:
        w, h = x1 - x0, y1 - y0
        area = w * h
        rows.append((area, count / area, x0, y0, x1, y1, w, h, count))
    rows.sort(reverse=True)
    print(f"components={len(rows)} changed_pixels={int(diff.sum())}")
    print("area fill x0 y0 x1 y1 w h pixels")
    for row in rows[:300]:
        print(f"{row[0]:7d} {row[1]:.3f} {row[2]:4d} {row[3]:4d} {row[4]:4d} {row[5]:4d} {row[6]:3d} {row[7]:3d} {row[8]:7d}")

    out = ROOT / "_v2_diagnostics"
    out.mkdir(exist_ok=True)
    skadi_im = Image.fromarray(skadi, "RGBA")
    default_im = Image.fromarray(default, "RGBA")
    candidates = []
    for _, fill, x0, y0, x1, y1, w, h, count in rows:
        crop = default[y0:y1, x0:x1]
        a = crop[:, :, 3] > 16
        rgb = crop[:, :, :3]
        bright = rgb.mean(axis=2)
        spread = rgb.max(axis=2) - rgb.min(axis=2)
        white = a & (bright > 170) & (spread < 65)
        dark = a & (bright < 95)
        white_n = int(white.sum())
        dark_n = int(dark.sum())
        alpha_n = int(a.sum())
        if 24 <= w <= 260 and 24 <= h <= 260 and alpha_n >= 250 and white_n >= 80 and dark_n >= 40:
            candidates.append({
                "x0": x0, "y0": y0, "x1": x1, "y1": y1, "w": w, "h": h,
                "diff_fill": round(fill, 4), "changed": count,
                "alpha": alpha_n, "white": white_n, "dark": dark_n,
            })
    candidates.sort(key=lambda r: (r["y0"], r["x0"]))
    for idx, row in enumerate(candidates):
        row["id"] = idx
    with (out / "knight_candidate_rects.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(candidates[0].keys()))
        writer.writeheader()
        writer.writerows(candidates)

    cell_w, cell_h = 300, 180
    cols, rows_per_page = 4, 5
    per_page = cols * rows_per_page
    font = ImageFont.load_default()
    for page_no in range((len(candidates) + per_page - 1) // per_page):
        sheet = Image.new("RGB", (cell_w * cols, cell_h * rows_per_page), (28, 31, 35))
        draw = ImageDraw.Draw(sheet)
        batch = candidates[page_no * per_page:(page_no + 1) * per_page]
        for n, row in enumerate(batch):
            col, rr = n % cols, n // cols
            ox, oy = col * cell_w, rr * cell_h
            box = (row["x0"], row["y0"], row["x1"], row["y1"])
            d = default_im.crop(box)
            s = skadi_im.crop(box)
            scale = min(1.0, 130 / max(d.width, s.width), 135 / max(d.height, s.height))
            size = (max(1, round(d.width * scale)), max(1, round(d.height * scale)))
            d = d.resize(size, Image.Resampling.LANCZOS)
            s = s.resize(size, Image.Resampling.LANCZOS)
            left = Image.new("RGB", (140, 140), (47, 51, 57))
            right = Image.new("RGB", (140, 140), (47, 51, 57))
            left.paste(d, ((140 - d.width)//2, (140 - d.height)//2), d)
            right.paste(s, ((140 - s.width)//2, (140 - s.height)//2), s)
            sheet.paste(left, (ox + 6, oy + 30))
            sheet.paste(right, (ox + 153, oy + 30))
            draw.text((ox + 6, oy + 6), f'ID {row["id"]:03d}  ({row["x0"]},{row["y0"]}) {row["w"]}x{row["h"]}', font=font, fill=(240, 240, 240))
        sheet.save(out / f"knight_candidates_{page_no:02d}.png", optimize=True)
    print(f"candidate_player_rects={len(candidates)} pages={(len(candidates)+per_page-1)//per_page}")


if __name__ == "__main__":
    main()
