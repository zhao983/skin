from __future__ import annotations

from collections import deque
import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


ROOT = Path(r"D:\skin")
ATLAS = ROOT / "Default" / "Knight.png"
RECTS = ROOT / "_v2_diagnostics" / "knight_candidate_rects.csv"
OUT = ROOT / "_v2_diagnostics" / "recolor_prototype.png"


def components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    result: list[np.ndarray] = []
    for y, x in zip(*np.nonzero(mask & ~seen)):
        if seen[y, x]:
            continue
        q = deque([(int(y), int(x))])
        seen[y, x] = True
        pts: list[tuple[int, int]] = []
        while q:
            cy, cx = q.popleft()
            pts.append((cy, cx))
            for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    q.append((ny, nx))
        cm = np.zeros_like(mask, dtype=bool)
        yy, xx = zip(*pts)
        cm[np.asarray(yy), np.asarray(xx)] = True
        result.append(cm)
    return result


def fill_holes(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    inv = ~mask
    outside = np.zeros_like(mask, dtype=bool)
    q: deque[tuple[int, int]] = deque()
    for x in range(w):
        if inv[0, x]: q.append((0, x)); outside[0, x] = True
        if inv[h - 1, x] and not outside[h - 1, x]: q.append((h - 1, x)); outside[h - 1, x] = True
    for y in range(h):
        if inv[y, 0] and not outside[y, 0]: q.append((y, 0)); outside[y, 0] = True
        if inv[y, w - 1] and not outside[y, w - 1]: q.append((y, w - 1)); outside[y, w - 1] = True
    while q:
        y, x = q.popleft()
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= ny < h and 0 <= nx < w and inv[ny, nx] and not outside[ny, nx]:
                outside[ny, nx] = True
                q.append((ny, nx))
    return ~outside


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.copy()
    im = Image.fromarray((mask * 255).astype(np.uint8), "L")
    return np.asarray(im.filter(ImageFilter.MaxFilter(radius * 2 + 1))) > 0


def erode(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.copy()
    im = Image.fromarray((mask * 255).astype(np.uint8), "L")
    return np.asarray(im.filter(ImageFilter.MinFilter(radius * 2 + 1))) > 0


def paste_rotated_ellipse(
    base: Image.Image,
    center: tuple[float, float],
    size: tuple[float, float],
    angle: float,
    fill: tuple[int, int, int, int],
    outline: tuple[int, int, int, int] = (24, 20, 12, 255),
    width: int = 2,
) -> None:
    w, h = max(3, round(size[0])), max(3, round(size[1]))
    pad = width + 3
    stamp = Image.new("RGBA", (w + pad * 2, h + pad * 2), (0, 0, 0, 0))
    d = ImageDraw.Draw(stamp)
    d.ellipse((pad, pad, pad + w - 1, pad + h - 1), fill=fill, outline=outline, width=width)
    stamp = stamp.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)
    x = round(center[0] - stamp.width / 2)
    y = round(center[1] - stamp.height / 2)
    base.alpha_composite(stamp, (x, y))


def recolor_sprite(crop: Image.Image) -> tuple[Image.Image, dict[str, int]]:
    arr = np.array(crop.convert("RGBA"), copy=True)
    rgb = arr[:, :, :3]
    alpha = arr[:, :, 3]
    spread = rgb.max(axis=2).astype(np.int16) - rgb.min(axis=2).astype(np.int16)
    mean = rgb.mean(axis=2)
    white = (alpha > 25) & (mean > 158) & (spread < 70)
    all_white_components = components(white)
    white_components = []
    for cm in all_white_components:
        ys, xs = np.nonzero(cm)
        if len(xs) < 80:
            continue
        bw, bh = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
        fill = len(xs) / (bw * bh)
        if bw <= 8 or bh <= 8 or max(bw / bh, bh / bw) > 3.3 or fill < 0.18:
            continue
        white_components.append((len(xs), cm))
    if not white_components:
        return crop.copy(), {"heads": 0, "head_pixels": 0, "body_pixels": 0, "eyes": 0}

    # The Knight's mask is consistently the largest compact off-white component
    # inside a packed player sprite. Slashes/nails are thinner and fail the test.
    _, primary_head = max(white_components, key=lambda item: item[0])
    py, px = np.nonzero(primary_head)
    pbox = (px.min(), py.min(), px.max(), py.max())
    pad_x = max(5, round((pbox[2] - pbox[0] + 1) * 0.28))
    pad_y = max(5, round((pbox[3] - pbox[1] + 1) * 0.28))
    head = primary_head.copy()
    for candidate in all_white_components:
        area = int(candidate.sum())
        if candidate is primary_head or area > int(primary_head.sum() * 0.65) or area < 4:
            continue
        cy0, cx0 = np.nonzero(candidate)
        if not len(cx0):
            continue
        ccx, ccy = float(cx0.mean()), float(cy0.mean())
        if (pbox[0] - pad_x <= ccx <= pbox[2] + pad_x and
                pbox[1] - pad_y <= ccy <= pbox[3] + pad_y):
            head |= candidate
    # Close narrow antialias gaps before filling. Several real Knight masks have
    # an eye touching the outer contour by one or two pixels; requiring a
    # literally closed hole would miss those frames and cause white flashes.
    close_radius = max(1, min(3, round(np.sqrt(float(head.sum())) / 20)))
    sealed = erode(dilate(head, close_radius), close_radius)
    filled = fill_holes(sealed | head)
    holes = filled & ~head
    hole_components = [c for c in components(holes) if 3 <= int(c.sum()) <= int(head.sum() * 0.35)]
    eye_mask = np.zeros_like(head)
    for c in hole_components:
        eye_mask |= c

    # Find the opaque connected sprite component containing the mask. It includes
    # the cloak/body and sometimes the nail; distance from the mask keeps the nail
    # and external effects out of the body recolour.
    opaque = alpha > 20
    player_component = np.zeros_like(head)
    for cm in components(opaque):
        if np.any(cm & head):
            player_component = cm
            break
    ys, xs = np.nonzero(head)
    radius = max(8, round(min(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1) * 0.75))
    near_head = dilate(filled, radius)
    body = player_component & near_head & ~dilate(filled, 2)

    # Warm yellow motion accents while keeping darkest outline pixels dark and leaving the
    # original nail/effects untouched outside the local body neighbourhood.
    body_lum = mean
    paintable = body & (body_lum > 20)
    br = np.clip(116 + body_lum * 0.72, 0, 244)
    bg = np.clip(55 + body_lum * 0.62, 0, 205)
    bb = np.clip(5 + body_lum * 0.14, 0, 48)
    for ch, val in enumerate((br, bg, bb)):
        arr[:, :, ch][paintable] = val[paintable].astype(np.uint8)

    # Derive a stable head centre after filling the Knight eye holes and eroding
    # away the thin horns. Then cover the old mask completely with Nailoong's
    # round head and compact body. This changes the silhouette, not just colour.
    core = erode(filled, max(2, round(np.sqrt(head.sum()) / 16)))
    core_parts = components(core)
    core = max(core_parts, key=lambda c: int(c.sum())) if core_parts else filled
    cy, cx = np.nonzero(core)
    head_cx, head_cy = float(cx.mean()), float(cy.mean())
    head_radius = max(9.0, np.sqrt(float(filled.sum()) / np.pi) * 0.93)

    body_candidates = player_component & near_head & ~dilate(filled, 4) & (mean < 160)
    by, bx = np.nonzero(body_candidates)
    if len(bx):
        vx, vy = float(bx.mean() - head_cx), float(by.mean() - head_cy)
    else:
        vx, vy = 0.0, 1.0
    mag = max(1.0, float(np.hypot(vx, vy)))
    ux, uy = vx / mag, vy / mag
    angle = float(np.degrees(np.arctan2(uy, ux)) - 90.0)
    body_center = (head_cx + ux * head_radius * 0.88, head_cy + uy * head_radius * 0.88)

    # Remove the white mask, eyes, horns, and their dark border. No white Knight
    # pixels remain underneath the replacement head.
    old_head = player_component & dilate(filled, 7)
    arr[old_head] = (0, 0, 0, 0)
    result = Image.fromarray(arr, "RGBA")

    paste_rotated_ellipse(
        result, body_center, (head_radius * 1.34, head_radius * 1.62), angle,
        (243, 177, 17, 255), width=max(1, round(head_radius / 13)),
    )
    belly_center = (body_center[0] + ux * head_radius * 0.08, body_center[1] + uy * head_radius * 0.08)
    paste_rotated_ellipse(
        result, belly_center, (head_radius * 0.72, head_radius * 0.86), angle,
        (250, 229, 176, 255), outline=(228, 187, 94, 255), width=max(1, round(head_radius / 18)),
    )
    # Two tiny limbs read clearly at game scale without changing attack direction.
    px, py = -uy, ux
    for side in (-1.0, 1.0):
        limb_center = (
            body_center[0] + px * side * head_radius * 0.63 + ux * head_radius * 0.12,
            body_center[1] + py * side * head_radius * 0.63 + uy * head_radius * 0.12,
        )
        paste_rotated_ellipse(
            result, limb_center, (head_radius * 0.38, head_radius * 0.58), angle,
            (239, 165, 13, 255), width=max(1, round(head_radius / 16)),
        )

    # Head is deliberately wider than the old central mask core, masking the horn
    # bases and producing the large-headed Nailoong proportion.
    paste_rotated_ellipse(
        result, (head_cx, head_cy), (head_radius * 1.85, head_radius * 1.80), 0,
        (255, 207, 25, 255), width=max(1, round(head_radius / 12)),
    )

    # Reuse eye positions only as semantic anchors. Render large green eyes with
    # dark pupils and highlights so side/front orientations remain readable.
    eye_count = 0
    rd = ImageDraw.Draw(result)
    for eye in hole_components[:2]:
        ey, ex = np.nonzero(eye)
        if not len(ex):
            continue
        eye_count += 1
        ecx, ecy = float(ex.mean()), float(ey.mean())
        er = max(2.2, np.sqrt(float(eye.sum()) / np.pi) * 0.78)
        rd.ellipse((ecx-er, ecy-er*1.15, ecx+er, ecy+er*1.15), fill=(21, 157, 67, 255), outline=(10, 73, 34, 255), width=max(1, round(er/3)))
        rd.ellipse((ecx-er*0.30, ecy-er*0.62, ecx+er*0.34, ecy+er*0.42), fill=(2, 30, 13, 255))
        hr = max(1.0, er * 0.18)
        rd.ellipse((ecx-er*0.30-hr, ecy-er*0.58-hr, ecx-er*0.30+hr, ecy-er*0.58+hr), fill=(244, 255, 238, 255))

    return result, {
        "heads": 1,
        "head_pixels": int(head.sum()),
        "body_pixels": int(paintable.sum()),
        "eyes": eye_count,
    }


def panel(sprite: Image.Image, size: tuple[int, int] = (150, 150)) -> Image.Image:
    scale = min(1.0, 135 / max(sprite.size))
    sprite = sprite.resize((max(1, round(sprite.width * scale)), max(1, round(sprite.height * scale))), Image.Resampling.LANCZOS)
    p = Image.new("RGB", size, (47, 51, 57))
    p.paste(sprite, ((size[0] - sprite.width) // 2, (size[1] - sprite.height) // 2), sprite)
    return p


def main() -> None:
    atlas = Image.open(ATLAS).convert("RGBA")
    rows = list(csv.DictReader(RECTS.open(encoding="utf-8")))
    ids = [241, 244, 245, 246, 250, 251, 252, 257, 258, 260, 274, 275, 276, 278, 279, 284, 287, 291, 292, 296, 297, 299]
    picked = [next(r for r in rows if int(r["id"]) == sid) for sid in ids]
    font = ImageFont.load_default()
    sheet = Image.new("RGB", (630, 190 * ((len(picked) + 1) // 2)), (26, 29, 33))
    draw = ImageDraw.Draw(sheet)
    for i, row in enumerate(picked):
        col, rr = i % 2, i // 2
        ox, oy = col * 315, rr * 190
        box = tuple(int(row[k]) for k in ("x0", "y0", "x1", "y1"))
        src = atlas.crop(box)
        dst, stats = recolor_sprite(src)
        sheet.paste(panel(src), (ox, oy + 28))
        sheet.paste(panel(dst), (ox + 155, oy + 28))
        draw.text((ox + 4, oy + 5), f'ID {row["id"]}  Default | v2 prototype  {stats}', fill=(235, 235, 235), font=font)
    sheet.save(OUT, optimize=True)
    print(OUT)


if __name__ == "__main__":
    main()
