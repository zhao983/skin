from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from analyze_atlas_rects import run_components
from prototype_nailong_recolor import components, dilate, erode, fill_holes
from scan_v2_heads import head_boxes as scan_v2_head_boxes


ROOT = Path(r"D:\skin")
DEFAULT = ROOT / "Default"
V2 = ROOT / "nailong_v2"
OUTPUT = ROOT / "nailong_v3"
QA = ROOT / "_v3_diagnostics"
DONORS_DIR = ROOT / "_v3_sources" / "donors"
V2_COVERAGE_CSV = ROOT / "_v2_diagnostics" / "v2_transformed_components.csv"

PLAYER_ATLASES = [
    "Knight.png",
    "Sprint.png",
    "Cloak.png",
    "Shriek.png",
    "Wings.png",
    "Webbed.png",
    "DreamArrival.png",
    "Dreamnail.png",
    "Birthplace.png",
]

REVIEWED_KNIGHT_RECTS = [
    (2977, 3116, 3072, 3225),
    (2401, 3900, 2530, 3989),
    (267, 3678, 387, 3810),
]
REVIEWED_WEBBED_RECT = (316, 771, 421, 891)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def masks_from_runs(mask: np.ndarray) -> list[np.ndarray]:
    """Materialize component masks from the run-based scanner.

    Packed sprite crops almost never contain nested disconnected opaque islands,
    so clipping the source mask to each component box is equivalent here and is
    much faster than a Python pixel flood-fill over hundreds of frames.
    """
    out: list[np.ndarray] = []
    for x0, y0, x1, y1, _ in run_components(mask):
        cm = np.zeros_like(mask, dtype=bool)
        cm[y0:y1, x0:x1] = mask[y0:y1, x0:x1]
        out.append(cm)
    return out


def connected_from_seed(mask: np.ndarray, seed_mask: np.ndarray) -> np.ndarray:
    """Return only the opaque component that actually contains the head."""
    overlap = np.argwhere(mask & seed_mask)
    if len(overlap) == 0:
        return np.zeros_like(mask, dtype=bool)
    sy, sx = map(int, overlap[0])
    out = np.zeros_like(mask, dtype=bool)
    stack = [(sy, sx)]
    out[sy, sx] = True
    h, w = mask.shape
    while stack:
        y, x = stack.pop()
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not out[ny, nx]:
                out[ny, nx] = True
                stack.append((ny, nx))
    return out


def discover_player_components(source: Image.Image) -> list[tuple[int, int, int, int, int]]:
    arr = np.asarray(source.convert("RGBA"))
    alpha = arr[:, :, 3]
    found = []
    for x0, y0, x1, y1, count in run_components(alpha > 20):
        w, h = x1 - x0, y1 - y0
        if count < 120 or w < 12 or h < 12 or w > 520 or h > 520:
            continue
        crop = arr[y0:y1, x0:x1]
        a = crop[:, :, 3] > 20
        rgb = crop[:, :, :3]
        mean = rgb.mean(axis=2)
        spread = rgb.max(axis=2).astype(np.int16) - rgb.min(axis=2).astype(np.int16)
        white = a & (mean > 158) & (spread < 70)
        dark = a & (mean < 100)
        if int(white.sum()) >= 80 and int(dark.sum()) >= 18:
            found.append((x0, y0, x1, y1, count))
    return found


def locate_knight(crop: Image.Image) -> dict[str, object] | None:
    arr = np.asarray(crop.convert("RGBA"))
    rgb = arr[:, :, :3]
    alpha = arr[:, :, 3]
    spread = rgb.max(axis=2).astype(np.int16) - rgb.min(axis=2).astype(np.int16)
    mean = rgb.mean(axis=2)
    white = (alpha > 25) & (mean > 158) & (spread < 70)

    all_white = masks_from_runs(white)
    compact: list[tuple[int, np.ndarray]] = []
    for cm in all_white:
        ys, xs = np.nonzero(cm)
        if len(xs) < 80:
            continue
        w, h = int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)
        fill = len(xs) / (w * h)
        if w <= 8 or h <= 8 or max(w / h, h / w) > 3.3 or fill < 0.18:
            continue
        compact.append((len(xs), cm))
    if not compact:
        return None

    _, primary = max(compact, key=lambda item: item[0])
    py, px = np.nonzero(primary)
    pbox = (int(px.min()), int(py.min()), int(px.max()), int(py.max()))
    pad_x = max(5, round((pbox[2] - pbox[0] + 1) * 0.28))
    pad_y = max(5, round((pbox[3] - pbox[1] + 1) * 0.28))
    head = primary.copy()
    for candidate in all_white:
        area = int(candidate.sum())
        if candidate is primary or area > int(primary.sum() * 0.65) or area < 4:
            continue
        cy, cx = np.nonzero(candidate)
        if not len(cx):
            continue
        ccx, ccy = float(cx.mean()), float(cy.mean())
        if pbox[0] - pad_x <= ccx <= pbox[2] + pad_x and pbox[1] - pad_y <= ccy <= pbox[3] + pad_y:
            head |= candidate

    close_radius = max(1, min(3, round(np.sqrt(float(head.sum())) / 20)))
    sealed = erode(dilate(head, close_radius), close_radius)
    combined = sealed | head
    fx0, fy0, fx1, fy1 = bbox(combined)
    fx0, fy0 = max(0, fx0 - 2), max(0, fy0 - 2)
    fx1, fy1 = min(combined.shape[1], fx1 + 2), min(combined.shape[0], fy1 + 2)
    filled = np.zeros_like(combined, dtype=bool)
    filled[fy0:fy1, fx0:fx1] = fill_holes(combined[fy0:fy1, fx0:fx1])
    holes = filled & ~head
    hole_parts = [c for c in masks_from_runs(holes) if 3 <= int(c.sum()) <= int(head.sum() * 0.35)]
    if len(hole_parts) < 1:
        return None

    player = connected_from_seed(alpha > 20, head)
    if not np.any(player):
        return None

    core = erode(filled, max(2, round(np.sqrt(float(head.sum())) / 16)))
    core_parts = masks_from_runs(core)
    core = max(core_parts, key=lambda c: int(c.sum())) if core_parts else filled
    cy, cx = np.nonzero(core)
    head_cx, head_cy = float(cx.mean()), float(cy.mean())
    head_radius = max(8.0, np.sqrt(float(filled.sum()) / np.pi) * 0.93)
    hx0, hy0, hx1, hy1 = bbox(filled)
    head_extent = max(12.0, (hx1 - hx0 + hy1 - hy0) / 2.0)

    near = dilate(filled, max(8, round(head_extent * 1.05)))
    body = player & near & ~dilate(filled, 3) & (mean < 150)
    by, bx = np.nonzero(body)
    if len(bx):
        vx, vy = float(bx.mean() - head_cx), float(by.mean() - head_cy)
    else:
        vx, vy = 0.0, max(1.0, head_radius)

    yy, xx = np.indices(player.shape)
    dist = np.hypot(xx - head_cx, yy - head_cy)
    pale = (mean > 82) & (spread < 105)
    chromatic_fx = (spread > 95) & (mean > 65)
    far_seed = player & (pale | chromatic_fx) & (dist > head_extent * 1.48)
    fy, fx = np.nonzero(far_seed)
    attack = False
    weapon_vx = weapon_vy = 0.0
    max_dist = float(dist[far_seed].max()) if len(fx) else 0.0
    if len(fx) >= 7 and max_dist > head_extent * 1.82:
        dvals = dist[fy, fx]
        cutoff = float(np.percentile(dvals, 70))
        keep = dvals >= cutoff
        weapon_vx = float(np.mean(fx[keep] - head_cx))
        weapon_vy = float(np.mean(fy[keep] - head_cy))
        attack = math.hypot(weapon_vx, weapon_vy) > head_extent * 0.72

    return {
        "arr": arr,
        "mean": mean,
        "spread": spread,
        "head": head,
        "filled": filled,
        "player": player,
        "head_cx": head_cx,
        "head_cy": head_cy,
        "head_radius": head_radius,
        "head_extent": head_extent,
        "body_vx": vx,
        "body_vy": vy,
        "eyes": min(2, len(hole_parts)),
        "raw_eye_holes": len(hole_parts),
        "attack": attack,
        "weapon_vx": weapon_vx,
        "weapon_vy": weapon_vy,
        "far_seed": far_seed,
        "dist": dist,
    }


def locate_from_v2_anchor(
    crop: Image.Image,
    v2_crop: Image.Image,
    target_box: tuple[int, int, int, int] | None = None,
) -> dict[str, object] | None:
    """Reuse only v2's verified head anchor, never its pose or body mapping."""
    arr = np.asarray(crop.convert("RGBA"))
    old = np.asarray(v2_crop.convert("RGBA"))
    rgb, alpha = arr[:, :, :3], arr[:, :, 3]
    vrgb, va = old[:, :, :3], old[:, :, 3]
    vr, vg, vb = [vrgb[:, :, i].astype(np.int16) for i in range(3)]
    # The v2 replacement head used a unique lemon-yellow fill. The orange body,
    # cream belly, green eyes and retained effects all fail this compact mask.
    yellow_head = (va > 20) & (vr > 238) & (vg > 188) & (vb < 85) & (vr > vg * 1.08)
    if target_box is not None:
        tx0, ty0, tx1, ty1 = target_box
        restricted = np.zeros_like(yellow_head)
        restricted[ty0:ty1, tx0:tx1] = True
        yellow_head &= restricted
    head_parts = [cm for cm in masks_from_runs(yellow_head) if int(cm.sum()) >= 70]
    if not head_parts:
        return None
    head = max(head_parts, key=lambda cm: int(cm.sum()))
    x0, y0, x1, y1 = bbox(head)
    local = head[y0:y1, x0:x1]
    filled = np.zeros_like(head, dtype=bool)
    filled[y0:y1, x0:x1] = fill_holes(local)

    green = (va > 20) & (vg > 65) & (vg > vr * 1.2) & (vg > vb * 1.25)
    eye_parts = [cm for cm in masks_from_runs(green & dilate(filled, 4)) if int(cm.sum()) >= 3]
    eyes = min(2, max(1, len(eye_parts)))
    player = connected_from_seed(alpha > 20, dilate(filled, 5))
    if not np.any(player):
        return None

    cy, cx = np.nonzero(filled)
    head_cx, head_cy = float(cx.mean()), float(cy.mean())
    head_radius = max(8.0, np.sqrt(float(filled.sum()) / np.pi) * 0.94)
    hx0, hy0, hx1, hy1 = bbox(filled)
    head_extent = max(12.0, (hx1 - hx0 + hy1 - hy0) / 2.0)

    mean = rgb.mean(axis=2)
    spread = rgb.max(axis=2).astype(np.int16) - rgb.min(axis=2).astype(np.int16)
    near = dilate(filled, max(8, round(head_extent * 1.05)))
    body = player & near & ~dilate(filled, 3) & (mean < 150)
    by, bx = np.nonzero(body)
    if len(bx):
        body_vx, body_vy = float(bx.mean() - head_cx), float(by.mean() - head_cy)
    else:
        body_vx, body_vy = 0.0, max(1.0, head_radius)

    yy, xx = np.indices(player.shape)
    dist = np.hypot(xx - head_cx, yy - head_cy)
    pale = (mean > 82) & (spread < 105)
    chromatic_fx = (spread > 95) & (mean > 65)
    far_seed = player & (pale | chromatic_fx) & (dist > head_extent * 1.48)
    fy, fx = np.nonzero(far_seed)
    attack = False
    weapon_vx = weapon_vy = 0.0
    max_dist = float(dist[far_seed].max()) if len(fx) else 0.0
    if len(fx) >= 7 and max_dist > head_extent * 1.82:
        dvals = dist[fy, fx]
        keep = dvals >= float(np.percentile(dvals, 70))
        weapon_vx = float(np.mean(fx[keep] - head_cx))
        weapon_vy = float(np.mean(fy[keep] - head_cy))
        attack = math.hypot(weapon_vx, weapon_vy) > head_extent * 0.72
        # A horizontal motion smear/cloak trail points in the same direction as
        # the old body mass. A real forward nail points to the other side.
        if attack and abs(weapon_vx) > abs(weapon_vy) * 1.12 and abs(body_vx) > head_extent * 0.08:
            if weapon_vx * body_vx > 0:
                attack = False

    return {
        "arr": arr, "mean": mean, "spread": spread,
        "head": head, "filled": filled, "player": player,
        "head_cx": head_cx, "head_cy": head_cy,
        "head_radius": head_radius, "head_extent": head_extent,
        "body_vx": body_vx, "body_vy": body_vy,
        "eyes": eyes, "raw_eye_holes": eyes,
        "attack": attack, "weapon_vx": weapon_vx, "weapon_vy": weapon_vy,
        "far_seed": far_seed, "dist": dist,
    }


class DonorLibrary:
    def __init__(self) -> None:
        self.images: dict[str, Image.Image] = {}
        self.anchors: dict[str, tuple[float, float, float]] = {}
        for path in sorted(DONORS_DIR.glob("*.png")):
            image = Image.open(path).convert("RGBA")
            self.images[path.stem] = image
            self.anchors[path.stem] = self._measure(image)

    @staticmethod
    def _measure(image: Image.Image) -> tuple[float, float, float]:
        arr = np.asarray(image)
        rgb, alpha = arr[:, :, :3], arr[:, :, 3] > 20
        r, g, b = [rgb[:, :, i].astype(np.int16) for i in range(3)]
        green = alpha & (g > 70) & (g > r * 1.25) & (g > b * 1.25)
        eye_parts = [cm for cm in components(green) if int(cm.sum()) >= 15]
        if not eye_parts:
            raise RuntimeError("cannot find green eye anchor")
        centers = []
        for cm in eye_parts:
            ys, xs = np.nonzero(cm)
            centers.append((float(xs.mean()), float(ys.mean()), int(cm.sum())))
        total = sum(p[2] for p in centers)
        eye_x = sum(p[0] * p[2] for p in centers) / total
        eye_y = sum(p[1] * p[2] for p in centers) / total
        front = len(centers) >= 2 and max(p[0] for p in centers) - min(p[0] for p in centers) > image.width * 0.08
        anchor_x = eye_x if front else eye_x - image.height * 0.12
        anchor_y = eye_y + image.height * 0.075

        yellow = alpha & (r > 145) & (g > 75) & (r > g * 1.08) & (b < 160)
        spans = []
        y0 = max(0, round(anchor_y - image.height * 0.08))
        y1 = min(image.height, round(anchor_y + image.height * 0.09) + 1)
        for y in range(y0, y1):
            xs = np.flatnonzero(yellow[y])
            if len(xs):
                spans.append(int(xs.max() - xs.min() + 1))
        head_width = float(np.percentile(spans, 70)) if spans else image.width * 0.52
        return anchor_x, anchor_y, max(10.0, head_width)

    def get(self, name: str) -> tuple[Image.Image, tuple[float, float, float]]:
        return self.images[name], self.anchors[name]


def transform_donor(
    image: Image.Image,
    anchor: tuple[float, float],
    scale: float,
    mirror: bool,
    angle: float,
) -> tuple[Image.Image, tuple[float, float]]:
    nw = max(1, round(image.width * scale))
    nh = max(1, round(image.height * scale))
    stamp = image.resize((nw, nh), Image.Resampling.LANCZOS)
    ax, ay = anchor[0] * scale, anchor[1] * scale
    if mirror:
        stamp = stamp.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        ax = stamp.width - 1 - ax
    if abs(angle) < 0.5:
        return stamp, (ax, ay)

    side = max(stamp.width, stamp.height) * 2 + 16
    canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    center = side / 2.0
    canvas.alpha_composite(stamp, (round(center - ax), round(center - ay)))
    canvas = canvas.rotate(angle, resample=Image.Resampling.BICUBIC, expand=False)
    alpha = np.asarray(canvas)[:, :, 3] > 0
    if not np.any(alpha):
        return stamp, (ax, ay)
    x0, y0, x1, y1 = bbox(alpha)
    return canvas.crop((x0, y0, x1, y1)), (center - x0, center - y0)


def choose_donor(info: dict[str, object], variant_seed: int) -> tuple[str, bool, float, str]:
    attack = bool(info["attack"])
    vx, vy = float(info["body_vx"]), float(info["body_vy"])
    if attack:
        wx, wy = float(info["weapon_vx"]), float(info["weapon_vy"])
        if abs(wx) > abs(wy) * 1.12:
            return "attack_01", wx < 0, 0.0, "horizontal_attack"
        if wy < 0:
            return "attack_03", False, 0.0, "up_attack"
        return "attack_04", False, 0.0, "down_attack"

    if abs(vx) > max(4.0, abs(vy) * 0.72):
        # Rotate a clean running pose instead of baking motion streaks into the
        # body layer; the original atlas still supplies its own dash effects.
        return "move_04", vx > 0, (90.0 if vx > 0 else -90.0), "dash_or_horizontal_air"

    desired = math.degrees(math.atan2(vy, vx))
    rotation = max(-180.0, min(180.0, 90.0 - desired))
    if abs(rotation) > 24:
        return "move_05", vx > 0, rotation, "airborne_tilt"
    if int(info["eyes"]) == 2:
        return "move_00", False, rotation, "front_idle_or_focus"
    lean = abs(vx) / max(1.0, abs(vy))
    if lean > 0.30:
        return "move_04", vx > 0, rotation, "run"
    if lean > 0.13:
        return ("move_02" if variant_seed % 2 == 0 else "move_03"), vx > 0, rotation, "walk"
    return "move_01", vx > 0, rotation, "side_idle"


def replace_sprite(
    crop: Image.Image,
    donors: DonorLibrary,
    variant_seed: int,
    v2_crop: Image.Image | None = None,
    target_box: tuple[int, int, int, int] | None = None,
) -> tuple[Image.Image, dict[str, object]]:
    info = locate_from_v2_anchor(crop, v2_crop, target_box) if v2_crop is not None else None
    if info is None:
        info = locate_knight(crop)
    if info is None:
        return crop.copy(), {"matched": False}

    arr = np.array(crop.convert("RGBA"), copy=True)
    original = arr.copy()
    player = np.asarray(info["player"], dtype=bool)
    dist = np.asarray(info["dist"])
    far_seed = np.asarray(info["far_seed"], dtype=bool)
    head_extent = float(info["head_extent"])

    # Remove the complete connected Knight body first. Restore only distant,
    # luminous weapon/slash pixels; dark cloak/tendrils are never restored.
    arr[player] = (0, 0, 0, 0)
    preserve_seed = far_seed & (dist > head_extent * 1.82)
    preserve = dilate(preserve_seed, 2) & player
    arr[preserve] = original[preserve]
    result = Image.fromarray(arr, "RGBA")

    donor_name, mirror, angle, semantic = choose_donor(info, variant_seed)
    donor, (anchor_x, anchor_y, donor_head_width) = donors.get(donor_name)
    target_head_width = float(info["head_radius"]) * 2.02
    scale = max(0.045, min(1.5, target_head_width / donor_head_width))
    stamp, stamp_anchor = transform_donor(donor, (anchor_x, anchor_y), scale, mirror, angle)
    # Keep the replacement within the original connected sprite's alpha box.
    # This is the atlas-frame safety boundary: a more rounded silhouette is
    # allowed, but it must not leak into a neighbouring packed frame.
    px0, py0, px1, py1 = bbox(player)
    px0, py0 = max(0, px0 - 2), max(0, py0 - 2)
    px1, py1 = min(crop.width, px1 + 2), min(crop.height, py1 + 2)
    hx, hy = float(info["head_cx"]), float(info["head_cy"])
    left, right = stamp_anchor[0], stamp.width - stamp_anchor[0]
    top, bottom = stamp_anchor[1], stamp.height - stamp_anchor[1]
    fit = min(
        (hx - px0) / max(1.0, left),
        (px1 - hx) / max(1.0, right),
        (hy - py0) / max(1.0, top),
        (py1 - hy) / max(1.0, bottom),
        1.0,
    )
    if fit < 0.995:
        scale *= max(0.25, fit * 0.97)
        stamp, stamp_anchor = transform_donor(donor, (anchor_x, anchor_y), scale, mirror, angle)
    x = round(float(info["head_cx"]) - stamp_anchor[0])
    y = round(float(info["head_cy"]) - stamp_anchor[1])
    result.alpha_composite(stamp, (x, y))

    return result, {
        "matched": True,
        "eyes": int(info["eyes"]),
        "raw_eye_holes": int(info["raw_eye_holes"]),
        "attack": bool(info["attack"]),
        "semantic": semantic,
        "donor": donor_name,
        "mirror": bool(mirror),
        "angle": round(float(angle), 2),
        "head_extent": round(head_extent, 2),
        "scale": round(scale, 4),
        "preserved_effect_pixels": int(preserve.sum()),
        "player_bbox": bbox(player),
    }


def apply_atlas(name: str, donors: DonorLibrary) -> list[dict[str, object]]:
    source = Image.open(DEFAULT / name).convert("RGBA")
    v2_source = Image.open(V2 / name).convert("RGBA")
    result = source.copy()
    records: list[dict[str, object]] = []
    coverage = scan_v2_head_boxes(v2_source)
    for index, (head_x0, head_y0, head_x1, head_y1, head_pixels) in enumerate(coverage):
        head_cx = (head_x0 + head_x1) / 2
        head_cy = (head_y0 + head_y1) / 2
        if name == "Webbed.png" and (
            REVIEWED_WEBBED_RECT[0] <= head_cx <= REVIEWED_WEBBED_RECT[2]
            and REVIEWED_WEBBED_RECT[1] <= head_cy <= REVIEWED_WEBBED_RECT[3]
        ):
            continue
        pad = 96
        box = (
            max(0, head_x0 - pad), max(0, head_y0 - pad),
            min(source.width, head_x1 + pad), min(source.height, head_y1 + pad),
        )
        before = source.crop(box)
        v2_before = v2_source.crop(box)
        local_target = (
            head_x0 - box[0], head_y0 - box[1],
            head_x1 - box[0], head_y1 - box[1],
        )
        after, stats = replace_sprite(before, donors, head_x0 * 31 + head_y0 * 17 + index, v2_before, local_target)
        if not stats.get("matched"):
            raise RuntimeError(f"v2 visible Nailoong head no longer matches: {name} {head_x0},{head_y0},{head_x1},{head_y1}")
        a0, a1 = np.asarray(before), np.asarray(after)
        changed = np.any(a0 != a1, axis=2)
        if not np.any(changed):
            continue
        result.paste(after, (box[0], box[1]), Image.fromarray(changed.astype(np.uint8) * 255, "L"))
        local_player = tuple(int(v) for v in stats.pop("player_bbox"))
        x0, y0 = box[0] + local_player[0], box[1] + local_player[1]
        x1, y1 = box[0] + local_player[2], box[1] + local_player[3]
        records.append({
            "atlas": name, "source": "v2_visible_yellow_head_green_eye_scan",
            "x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0,
            "head_x": head_x0, "head_y": head_y0,
            "head_w": head_x1 - head_x0, "head_h": head_y1 - head_y0,
            "alpha_pixels": int((a0[:, :, 3] > 20).sum()),
            "head_pixels": head_pixels, "changed_pixels": int(changed.sum()), **stats,
        })

    if name == "Webbed.png":
        # This player is fused into a large beam component. Replace the local
        # Knight body without touching the beam outside the reviewed rectangle.
        rect = REVIEWED_WEBBED_RECT
        before = source.crop(rect)
        arr = np.array(before, copy=True)
        yy, xx = np.indices(arr.shape[:2])
        clear = ((xx - 53) / 45) ** 2 + ((yy - 58) / 50) ** 2 <= 1
        arr[clear] = (0, 0, 0, 0)
        after = Image.fromarray(arr, "RGBA")
        donor, (ax, ay, hw) = donors.get("move_00")
        scale = 57.0 / hw
        stamp, anchor = transform_donor(donor, (ax, ay), scale, False, 0.0)
        after.alpha_composite(stamp, (round(53 - anchor[0]), round(50 - anchor[1])))
        a0, a1 = np.asarray(before), np.asarray(after)
        changed = np.any(a0 != a1, axis=2)
        result.paste(after, (rect[0], rect[1]), Image.fromarray(changed.astype(np.uint8) * 255, "L"))
        records.append({
            "atlas": name, "source": "manual_embedded_in_web_beam_review",
            "x": rect[0], "y": rect[1], "w": rect[2] - rect[0], "h": rect[3] - rect[1],
            "head_x": rect[0] + 24, "head_y": rect[1] + 4, "head_w": 58, "head_h": 78,
            "alpha_pixels": int((a0[:, :, 3] > 20).sum()), "head_pixels": 0,
            "changed_pixels": int(changed.sum()),
            "matched": True, "eyes": 2, "attack": False, "semantic": "webbed",
            "raw_eye_holes": 2,
            "donor": "move_00", "mirror": False, "angle": 0.0,
            "head_extent": 57.0, "scale": round(scale, 4), "preserved_effect_pixels": 0,
        })

    result.save(OUTPUT / name, optimize=True)
    return records


def panel(sprite: Image.Image, size: tuple[int, int] = (180, 165)) -> Image.Image:
    scale = min(1.0, 148 / max(sprite.size))
    sprite = sprite.resize((max(1, round(sprite.width * scale)), max(1, round(sprite.height * scale))), Image.Resampling.LANCZOS)
    out = Image.new("RGB", size, (47, 51, 57))
    out.paste(sprite, ((size[0] - sprite.width) // 2, (size[1] - sprite.height) // 2), sprite)
    return out


def make_keyframes(records: list[dict[str, object]]) -> None:
    knight = [r for r in records if r["atlas"] == "Knight.png" and str(r["source"]).startswith("v2_visible")]
    if not knight:
        return
    wanted = 18
    positions = np.linspace(0, len(knight) - 1, wanted).round().astype(int)
    picked = [knight[int(i)] for i in positions]
    default = Image.open(DEFAULT / "Knight.png").convert("RGBA")
    v2 = Image.open(V2 / "Knight.png").convert("RGBA") if (V2 / "Knight.png").exists() else default
    v3 = Image.open(OUTPUT / "Knight.png").convert("RGBA")
    pw, ph = 180, 195
    sheet = Image.new("RGB", (pw * 3, ph * len(picked)), (29, 32, 37))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for row, rec in enumerate(picked):
        pad = 18
        box = (
            max(0, int(rec["x"]) - pad), max(0, int(rec["y"]) - pad),
            min(default.width, int(rec["x"]) + int(rec["w"]) + pad),
            min(default.height, int(rec["y"]) + int(rec["h"]) + pad),
        )
        for col, atlas in enumerate((default, v2, v3)):
            sheet.paste(panel(atlas.crop(box), (pw, ph - 28)), (col * pw, row * ph + 28))
        draw.text((5, row * ph + 5), f"{rec['semantic']}  donor={rec['donor']}  ({rec['x']},{rec['y']})", font=font, fill=(245, 245, 245))
        draw.text((5, row * ph + 17), "Default                       v2                            v3", font=font, fill=(174, 188, 207))
    sheet.save(QA / "v3_keyframes_Default_v2_v3.png", optimize=True)


def make_overview(name: str, count: int) -> None:
    a = Image.open(DEFAULT / name).convert("RGBA")
    b = Image.open(OUTPUT / name).convert("RGBA")
    scale = min(1.0, 1000 / max(a.size))
    size = (max(1, round(a.width * scale)), max(1, round(a.height * scale)))
    panel_image = Image.new("RGB", (size[0] * 2, size[1] + 34), (31, 34, 39))
    for i, source in enumerate((a, b)):
        source = source.resize(size, Image.Resampling.LANCZOS)
        bg = Image.new("RGB", size, (54, 58, 65))
        bg.paste(source, (0, 0), source)
        panel_image.paste(bg, (i * size[0], 34))
    ImageDraw.Draw(panel_image).text((6, 8), f"{name}: Default | nailong_v3    transformed={count}", fill=(245, 245, 245), font=ImageFont.load_default())
    panel_image.save(QA / f"v3_overview_{Path(name).stem}.jpg", quality=92)


def write_reports(records: list[dict[str, object]]) -> None:
    with (QA / "v3_transformed_components.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader(); writer.writerows(records)

    default_files = {p.relative_to(DEFAULT).as_posix(): p for p in DEFAULT.rglob("*") if p.is_file()}
    output_files = {p.relative_to(OUTPUT).as_posix(): p for p in OUTPUT.rglob("*") if p.is_file()}
    modified, unchanged, mismatches = [], [], []
    for rel, src in default_files.items():
        dst = output_files.get(rel)
        if dst is None:
            mismatches.append({"file": rel, "issue": "missing"})
            continue
        if src.suffix.lower() == ".png":
            with Image.open(src) as a, Image.open(dst) as b:
                if a.size != b.size or a.mode != b.mode:
                    mismatches.append({"file": rel, "issue": f"metadata {a.mode} {a.size} -> {b.mode} {b.size}"})
        (unchanged if sha256(src) == sha256(dst) else modified).append(rel)

    per_atlas = Counter(str(r["atlas"]) for r in records)
    per_semantic = Counter(str(r["semantic"]) for r in records)
    per_donor = Counter(str(r["donor"]) for r in records)
    report = {
        "base": str(DEFAULT), "output": str(OUTPUT),
        "default_file_count": len(default_files), "output_file_count": len(output_files),
        "modified_files": sorted(modified), "unchanged_file_count": len(unchanged),
        "extra_files": sorted(set(output_files) - set(default_files)),
        "metadata_mismatches": mismatches,
        "transformed_component_count": len(records),
        "transformed_components_by_atlas": dict(sorted(per_atlas.items())),
        "semantics": dict(sorted(per_semantic.items())),
        "donors": dict(sorted(per_donor.items())),
        "attack_components": sum(bool(r["attack"]) for r in records),
        "method": "Scan every unique visible v2 lemon-yellow head with a nearby green eye (the gameplay-verified no-flash coverage); use only that head coordinate as an anchor; erase the connected Default Knight body; restore only distant luminous nail/slash pixels; place a full-body Nailoong donor with direction-aware action selection",
        "runtime_validation": "Pending external Hollow Knight / CustomKnight test; not installed in this workspace",
    }
    (QA / "v3_build_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main() -> None:
    if not OUTPUT.exists():
        raise SystemExit(f"Copy Default to {OUTPUT} before building")
    QA.mkdir(parents=True, exist_ok=True)
    donors = DonorLibrary()
    all_records: list[dict[str, object]] = []
    for name in PLAYER_ATLASES:
        if not (DEFAULT / name).exists():
            continue
        print(f"processing {name}", flush=True)
        records = apply_atlas(name, donors)
        all_records.extend(records)
        make_overview(name, len(records))
        print(f"  transformed={len(records)}", flush=True)
    make_keyframes(all_records)
    write_reports(all_records)


if __name__ == "__main__":
    main()
