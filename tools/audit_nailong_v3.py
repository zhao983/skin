from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from analyze_atlas_rects import run_components
from prototype_nailong_recolor import dilate, fill_holes


ROOT = Path(r"D:\skin")
DEFAULT = ROOT / "Default"
V2 = ROOT / "nailong_v2"
V3 = ROOT / "nailong_v3"
QA = ROOT / "_v3_diagnostics"
COVERAGE = QA.parent / "_v2_diagnostics" / "v2_transformed_components.csv"
RECORDS = QA / "v3_transformed_components.csv"


def component_masks(mask: np.ndarray):
    for x0, y0, x1, y1, count in run_components(mask):
        cm = np.zeros_like(mask, dtype=bool)
        cm[y0:y1, x0:x1] = mask[y0:y1, x0:x1]
        yield count, cm


def main() -> None:
    expected = list(csv.DictReader(COVERAGE.open(encoding="utf-8-sig")))
    actual = list(csv.DictReader(RECORDS.open(encoding="utf-8-sig")))
    expected_keys = Counter((r["atlas"], r["source"], int(r["x"]), int(r["y"]), int(r["w"]), int(r["h"])) for r in expected)
    actual_keys = Counter((r["atlas"], r["source"], int(r["x"]), int(r["y"]), int(r["w"]), int(r["h"])) for r in actual)

    residual = []
    unchanged_records = []
    outside_changes = {}
    per_atlas = Counter(r["atlas"] for r in expected)
    for atlas_name in sorted(per_atlas):
        default = np.asarray(Image.open(DEFAULT / atlas_name).convert("RGBA"))
        v2 = np.asarray(Image.open(V2 / atlas_name).convert("RGBA"))
        v3 = np.asarray(Image.open(V3 / atlas_name).convert("RGBA"))
        changed = np.any(default != v3, axis=2)
        allowed = np.zeros(changed.shape, dtype=bool)
        for row in [r for r in expected if r["atlas"] == atlas_name]:
            x, y, w, h = [int(row[k]) for k in ("x", "y", "w", "h")]
            x0, y0 = max(0, x - 3), max(0, y - 3)
            x1, y1 = min(changed.shape[1], x + w + 3), min(changed.shape[0], y + h + 3)
            allowed[y0:y1, x0:x1] = True
            if not np.any(changed[y:y+h, x:x+w]):
                unchanged_records.append({"atlas": atlas_name, "x": x, "y": y, "w": w, "h": h})
            if row["source"] == "manual_embedded_in_web_beam_review":
                continue

            old_crop = v2[y:y+h, x:x+w]
            default_crop = default[y:y+h, x:x+w]
            final_crop = v3[y:y+h, x:x+w]
            r, g, b = [old_crop[:, :, i].astype(np.int16) for i in range(3)]
            yellow = (old_crop[:, :, 3] > 20) & (r > 238) & (g > 188) & (b < 85) & (r > g * 1.08)
            parts = [(count, cm) for count, cm in component_masks(yellow) if count >= 40]
            if not parts:
                continue
            head = max(parts, key=lambda item: item[0])[1]
            hx0, hy0, hx1, hy1, _ = next(iter(run_components(head)))
            filled = np.zeros_like(head)
            filled[hy0:hy1, hx0:hx1] = fill_holes(head[hy0:hy1, hx0:hx1])
            radius = max(5, round(max(hx1 - hx0, hy1 - hy0) * 0.32))
            head_zone = dilate(filled, radius)
            drgb = default_crop[:, :, :3]
            mean = drgb.mean(axis=2)
            spread = drgb.max(axis=2).astype(np.int16) - drgb.min(axis=2).astype(np.int16)
            knight_white = (default_crop[:, :, 3] > 20) & (mean > 158) & (spread < 70) & head_zone
            identical = np.all(default_crop == final_crop, axis=2)
            remain = knight_white & identical
            if np.any(remain):
                residual.append({
                    "atlas": atlas_name, "x": x, "y": y, "w": w, "h": h,
                    "unchanged_knight_head_pixels": int(remain.sum()),
                })
        outside_changes[atlas_name] = int((changed & ~allowed).sum())

    report = {
        "expected_v2_verified_coverage_count": len(expected),
        "v3_record_count": len(actual),
        "coverage_key_match": expected_keys == actual_keys,
        "coverage_missing": list((expected_keys - actual_keys).elements()),
        "coverage_extra": list((actual_keys - expected_keys).elements()),
        "records_with_no_changed_pixels": unchanged_records,
        "unchanged_default_knight_head_residual_count": len(residual),
        "unchanged_default_knight_head_residuals": residual,
        "changed_pixels_outside_verified_boxes_by_atlas": outside_changes,
        "pass": (
            expected_keys == actual_keys
            and not unchanged_records
            and not residual
            and all(value == 0 for value in outside_changes.values())
        ),
        "note": "Runtime gameplay validation still requires Hollow Knight + CustomKnight; unavailable in this workspace.",
    }
    (QA / "v3_residual_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
