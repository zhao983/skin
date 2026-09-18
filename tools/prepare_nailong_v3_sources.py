from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(r"D:\skin")
SOURCE = ROOT / "_v3_sources"
DONORS = SOURCE / "donors"


def components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[np.ndarray] = []
    for sy, sx in zip(*np.nonzero(mask)):
        if seen[sy, sx]:
            continue
        q = deque([(int(sy), int(sx))])
        seen[sy, sx] = True
        pts: list[tuple[int, int]] = []
        while q:
            y, x = q.popleft()
            pts.append((y, x))
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    q.append((ny, nx))
        cm = np.zeros_like(mask, dtype=bool)
        yy, xx = zip(*pts)
        cm[np.asarray(yy), np.asarray(xx)] = True
        out.append(cm)
    return out


def fill_holes(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    outside = np.zeros_like(mask, dtype=bool)
    inv = ~mask
    q: deque[tuple[int, int]] = deque()
    for x in range(w):
        if inv[0, x]:
            q.append((0, x)); outside[0, x] = True
        if inv[h - 1, x] and not outside[h - 1, x]:
            q.append((h - 1, x)); outside[h - 1, x] = True
    for y in range(h):
        if inv[y, 0] and not outside[y, 0]:
            q.append((y, 0)); outside[y, 0] = True
        if inv[y, w - 1] and not outside[y, w - 1]:
            q.append((y, w - 1)); outside[y, w - 1] = True
    while q:
        y, x = q.popleft()
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= ny < h and 0 <= nx < w and inv[ny, nx] and not outside[ny, nx]:
                outside[ny, nx] = True
                q.append((ny, nx))
    return ~outside


def remove_checkerboard(cell: Image.Image) -> Image.Image:
    """Recover a clean alpha silhouette from ImageGen's baked preview grid."""
    rgb = np.asarray(cell.convert("RGB"))
    spread = rgb.max(axis=2).astype(np.int16) - rgb.min(axis=2).astype(np.int16)
    mean = rgb.mean(axis=2)
    # The checker is neutral gray at about 204/255. Character outlines, yellow,
    # green and brown are chromatic or darker. White nail interiors are filled
    # after selecting their connected dark outline.
    seed = (spread > 12) | (mean < 188)
    parts = components(seed)
    if not parts:
        raise RuntimeError("no foreground component found")
    yellow = (rgb[:, :, 0] > 170) & (rgb[:, :, 1] > 90) & (rgb[:, :, 2] < 120)
    green = (rgb[:, :, 1] > rgb[:, :, 0] * 0.85) & (rgb[:, :, 1] > rgb[:, :, 2] * 1.25) & (rgb[:, :, 1] > 70)
    ranked = []
    for cm in parts:
        n = int(cm.sum())
        anchor = int((cm & (yellow | green)).sum())
        if n >= 40 and anchor >= 8:
            ranked.append((anchor, n, cm))
    if not ranked:
        raise RuntimeError("no Nailoong-anchored foreground component found")
    # The body is always the largest yellow/green-anchored component. Filling
    # holes recovers cream belly and pale blade interiors without the grid.
    body = max(ranked, key=lambda item: (item[0], item[1]))[2]
    mask = fill_holes(body)
    ys, xs = np.nonzero(mask)
    pad = 4
    x0, x1 = max(0, int(xs.min()) - pad), min(cell.width, int(xs.max()) + pad + 1)
    y0, y1 = max(0, int(ys.min()) - pad), min(cell.height, int(ys.max()) + pad + 1)
    rgba = np.zeros((cell.height, cell.width, 4), dtype=np.uint8)
    rgba[:, :, :3] = rgb
    rgba[:, :, 3] = mask.astype(np.uint8) * 255
    return Image.fromarray(rgba, "RGBA").crop((x0, y0, x1, y1))


def split_grid(path: Path, cols: int, rows: int, prefix: str) -> list[Path]:
    source = Image.open(path).convert("RGB")
    paths: list[Path] = []
    for row in range(rows):
        for col in range(cols):
            x0 = round(col * source.width / cols)
            x1 = round((col + 1) * source.width / cols)
            y0 = round(row * source.height / rows)
            y1 = round((row + 1) * source.height / rows)
            donor = remove_checkerboard(source.crop((x0, y0, x1, y1)))
            out = DONORS / f"{prefix}_{row * cols + col:02d}.png"
            donor.save(out, optimize=True)
            paths.append(out)
    return paths


def checker(size: tuple[int, int], step: int = 16) -> Image.Image:
    out = Image.new("RGB", size, (230, 230, 230))
    draw = ImageDraw.Draw(out)
    for y in range(0, size[1], step):
        for x in range(0, size[0], step):
            if (x // step + y // step) % 2:
                draw.rectangle((x, y, x + step - 1, y + step - 1), fill=(190, 194, 199))
    return out


def make_contact(paths: list[Path]) -> None:
    cols, cw, ch = 4, 280, 260
    rows = (len(paths) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cw, rows * ch), (35, 38, 43))
    font = ImageFont.load_default()
    draw = ImageDraw.Draw(sheet)
    for i, path in enumerate(paths):
        im = Image.open(path).convert("RGBA")
        scale = min((cw - 20) / im.width, (ch - 35) / im.height, 1.0)
        im = im.resize((max(1, round(im.width * scale)), max(1, round(im.height * scale))), Image.Resampling.LANCZOS)
        panel = checker((cw - 12, ch - 28))
        panel.paste(im, ((panel.width - im.width) // 2, (panel.height - im.height) // 2), im)
        x, y = (i % cols) * cw, (i // cols) * ch
        sheet.paste(panel, (x + 6, y + 22))
        draw.text((x + 7, y + 5), path.stem, font=font, fill=(245, 245, 245))
    sheet.save(SOURCE / "donor_contact.png", optimize=True)


def main() -> None:
    DONORS.mkdir(parents=True, exist_ok=True)
    movement = split_grid(SOURCE / "nailong_movement_master.png", 4, 3, "move")
    attack = split_grid(SOURCE / "nailong_attack_master.png", 3, 2, "attack")
    make_contact(movement + attack)
    print(f"prepared {len(movement)} movement + {len(attack)} attack donors")
    print(SOURCE / "donor_contact.png")


if __name__ == "__main__":
    main()
