from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

import build_nailong_v3 as build
from scan_v2_heads import ATLASES, head_boxes


ROOT = Path(r"D:\skin")
QA = ROOT / "_v3_diagnostics"
RECORDS = QA / "v3_transformed_components.csv"


def normalized_head_key(atlas: str, x: int, y: int, w: int, h: int):
    cx, cy = x + w / 2, y + h / 2
    rect = build.REVIEWED_WEBBED_RECT
    if atlas == "Webbed.png" and rect[0] <= cx <= rect[2] and rect[1] <= cy <= rect[3]:
        return (atlas, "manual_webbed")
    return (atlas, x, y, w, h)


def main() -> None:
    actual = list(csv.DictReader(RECORDS.open(encoding="utf-8-sig")))
    expected_by_atlas = {}
    expected_keys = Counter()
    actual_keys = Counter()
    for atlas in ATLASES:
        v2_image = Image.open(build.V2 / atlas).convert("RGBA")
        boxes = head_boxes(v2_image)
        expected_by_atlas[atlas] = boxes
        for x0, y0, x1, y1, _ in boxes:
            expected_keys[normalized_head_key(atlas, x0, y0, x1-x0, y1-y0)] += 1
    for row in actual:
        actual_keys[normalized_head_key(
            row["atlas"], int(row["head_x"]), int(row["head_y"]),
            int(row["head_w"]), int(row["head_h"]),
        )] += 1

    residual = []
    unchanged_heads = []
    outside_changes = {}
    for atlas in ATLASES:
        default_image = Image.open(build.DEFAULT / atlas).convert("RGBA")
        v2_image = Image.open(build.V2 / atlas).convert("RGBA")
        v3_image = Image.open(build.OUTPUT / atlas).convert("RGBA")
        default = np.asarray(default_image)
        final = np.asarray(v3_image)
        changed = np.any(default != final, axis=2)
        allowed = np.zeros(changed.shape, dtype=bool)
        for row in [r for r in actual if r["atlas"] == atlas]:
            x, y, w, h = [int(row[k]) for k in ("x", "y", "w", "h")]
            allowed[max(0,y-4):min(changed.shape[0],y+h+4), max(0,x-4):min(changed.shape[1],x+w+4)] = True
        outside_changes[atlas] = int((changed & ~allowed).sum())

        for x0, y0, x1, y1, _ in expected_by_atlas[atlas]:
            pad = 96
            crop_box = (
                max(0, x0-pad), max(0, y0-pad),
                min(default_image.width, x1+pad), min(default_image.height, y1+pad),
            )
            local_target = (
                x0-crop_box[0], y0-crop_box[1],
                x1-crop_box[0], y1-crop_box[1],
            )
            info = build.locate_from_v2_anchor(
                default_image.crop(crop_box), v2_image.crop(crop_box), local_target,
            )
            final_crop = np.asarray(v3_image.crop(crop_box))
            default_crop = np.asarray(default_image.crop(crop_box))
            if info is None:
                # The fused Webbed beam is deliberately handled by the manual
                # local replacement; audit its reviewed rectangle directly.
                rect = build.REVIEWED_WEBBED_RECT
                cx, cy = (x0+x1)/2, (y0+y1)/2
                if atlas == "Webbed.png" and rect[0] <= cx <= rect[2] and rect[1] <= cy <= rect[3]:
                    if not np.any(changed[rect[1]:rect[3], rect[0]:rect[2]]):
                        unchanged_heads.append({"atlas": atlas, "head": [x0,y0,x1,y1]})
                    continue
                residual.append({"atlas": atlas, "head": [x0,y0,x1,y1], "issue": "anchor_not_found"})
                continue
            player = np.asarray(info["player"], dtype=bool)
            head_zone = build.dilate(
                np.asarray(info["filled"], dtype=bool),
                max(3, round(float(info["head_extent"]) * 0.12)),
            )
            mean = default_crop[:, :, :3].mean(axis=2)
            spread = default_crop[:, :, :3].max(axis=2).astype(np.int16) - default_crop[:, :, :3].min(axis=2).astype(np.int16)
            knight_white = (
                (default_crop[:, :, 3] > 20)
                & (mean > 158) & (spread < 70)
                & player & head_zone
            )
            identical = np.all(default_crop == final_crop, axis=2)
            remain = knight_white & identical
            # A one- or two-pixel exact RGB coincidence can occur where the
            # new cream/yellow antialias lands on the old pale mask. It is not
            # a surviving Knight shape; only a coherent residual is actionable.
            if int(remain.sum()) > 2:
                residual.append({
                    "atlas": atlas, "head": [x0,y0,x1,y1],
                    "unchanged_default_head_pixels": int(remain.sum()),
                })
            if not np.any(np.any(default_crop != final_crop, axis=2) & player):
                unchanged_heads.append({"atlas": atlas, "head": [x0,y0,x1,y1]})

    report = {
        "unique_visible_v2_head_count": sum(len(v) for v in expected_by_atlas.values()),
        "v3_transformed_record_count": len(actual),
        "unique_head_key_match": expected_keys == actual_keys,
        "missing_heads": [list(v) for v in (expected_keys - actual_keys).elements()],
        "extra_heads": [list(v) for v in (actual_keys - expected_keys).elements()],
        "unchanged_visible_heads": unchanged_heads,
        "unchanged_default_knight_head_residuals": residual,
        "changed_pixels_outside_player_boxes_by_atlas": outside_changes,
        "pass": (
            expected_keys == actual_keys
            and not unchanged_heads
            and not residual
            and all(v == 0 for v in outside_changes.values())
        ),
        "runtime_validation": "Not run here: Hollow Knight / CustomKnight is not installed in this workspace.",
    }
    (QA / "v3_residual_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
