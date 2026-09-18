from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from analyze_atlas_rects import run_components
from prototype_nailong_recolor import paste_rotated_ellipse, recolor_sprite


ROOT = Path(r"D:\skin")
DEFAULT = ROOT / "Default"
OUTPUT = ROOT / "nailong_v2"
QA = ROOT / "_v2_diagnostics"

# These are the player-bearing materials actually loaded separately by
# CustomKnight. Other files are copied byte-for-byte from Default.
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

# These two normal-form Knight frames are connected to unusually large effects,
# so their alpha-component boxes exceed the automatic safety cap. They were
# reviewed side-by-side as Default | v2 | Skadi and are explicit player frames.
REVIEWED_KNIGHT_RECTS = [
    (2977, 3116, 3072, 3225),  # candidate ID 250
    (2401, 3900, 2530, 3989),  # candidate ID 354
    (267, 3678, 387, 3810),     # Knight embedded in the oversized black burst
]

REVIEWED_WEBBED_RECT = (316, 771, 421, 891)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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
        # This is a semantic prefilter for the Knight's pale mask and black
        # facial/body details. Final acceptance additionally requires enclosed
        # eye holes, so white slashes and nail blades are not treated as bodies.
        if int(white.sum()) >= 80 and int(dark.sum()) >= 18:
            found.append((x0, y0, x1, y1, count))
    return found


def apply_atlas(name: str) -> list[dict[str, int | str]]:
    src_path = DEFAULT / name
    out_path = OUTPUT / name
    source = Image.open(src_path).convert("RGBA")
    result = source.copy()
    records: list[dict[str, int | str]] = []
    for x0, y0, x1, y1, alpha_pixels in discover_player_components(source):
        pad = 20
        box = (
            max(0, x0 - pad),
            max(0, y0 - pad),
            min(source.width, x1 + pad),
            min(source.height, y1 + pad),
        )
        before = source.crop(box)
        after, stats = recolor_sprite(before)
        # Enclosed eye holes are the decisive player-mask signal. This is why
        # the build does not depend on dimensions or visual similarity alone.
        if stats["heads"] != 1 or stats["eyes"] not in (1, 2):
            continue
        a0 = np.asarray(before)
        a1 = np.asarray(after)
        changed = np.any(a0 != a1, axis=2)
        if not np.any(changed):
            continue
        diff_mask = Image.fromarray((changed * 255).astype(np.uint8), "L")
        result.paste(after, (box[0], box[1]), diff_mask)
        records.append({
            "atlas": name,
            "source": "semantic_scan",
            "x": x0,
            "y": y0,
            "w": x1 - x0,
            "h": y1 - y0,
            "alpha_pixels": alpha_pixels,
            "head_pixels": stats["head_pixels"],
            "body_pixels": stats["body_pixels"],
            "eyes": stats["eyes"],
            "changed_pixels": int(changed.sum()),
        })

    if name == "Knight.png":
        for box in REVIEWED_KNIGHT_RECTS:
            before = source.crop(box)
            after, stats = recolor_sprite(before)
            if stats["heads"] != 1 or stats["eyes"] not in (1, 2):
                raise RuntimeError(f"reviewed Knight rectangle no longer matches player semantics: {box} {stats}")
            a0 = np.asarray(before)
            a1 = np.asarray(after)
            changed = np.any(a0 != a1, axis=2)
            diff_mask = Image.fromarray((changed * 255).astype(np.uint8), "L")
            result.paste(after, (box[0], box[1]), diff_mask)
            records.append({
                "atlas": name,
                "source": "manual_default_skadi_review",
                "x": box[0],
                "y": box[1],
                "w": box[2] - box[0],
                "h": box[3] - box[1],
                "alpha_pixels": int((a0[:, :, 3] > 20).sum()),
                "head_pixels": stats["head_pixels"],
                "body_pixels": stats["body_pixels"],
                "eyes": stats["eyes"],
                "changed_pixels": int(changed.sum()),
            })
    if name == "Webbed.png":
        # The player mask is fused into one 597x325 white beam component, so the
        # generic pale-mask component is the effect itself. This local overlay
        # was reviewed on the Default frame and deliberately leaves the beam's
        # direction and surrounding pixels unchanged.
        box = REVIEWED_WEBBED_RECT
        before = source.crop(box)
        after = before.copy()
        clear = Image.new("L", before.size, 0)
        ImageDraw.Draw(clear).ellipse((24, 4, 82, 82), fill=255)
        transparent = Image.new("RGBA", before.size, (0, 0, 0, 0))
        after.paste(transparent, (0, 0), clear)
        paste_rotated_ellipse(after, (53, 74), (34, 40), 0, (243, 177, 17, 255), width=2)
        paste_rotated_ellipse(after, (53, 77), (19, 24), 0, (250, 229, 176, 255), outline=(228, 187, 94, 255), width=1)
        paste_rotated_ellipse(after, (53, 50), (58, 56), 0, (255, 207, 25, 255), width=2)
        draw = ImageDraw.Draw(after)
        for ecx in (43, 63):
            draw.ellipse((ecx-6, 42, ecx+6, 58), fill=(21,157,67,255), outline=(10,73,34,255), width=2)
            draw.ellipse((ecx-2, 45, ecx+3, 55), fill=(2,30,13,255))
            draw.ellipse((ecx-3, 44, ecx-1, 46), fill=(244,255,238,255))
        a0 = np.asarray(before)
        a1 = np.asarray(after)
        changed = np.any(a0 != a1, axis=2)
        diff_mask = Image.fromarray((changed * 255).astype(np.uint8), "L")
        result.paste(after, (box[0], box[1]), diff_mask)
        records.append({
            "atlas": name,
            "source": "manual_embedded_in_web_beam_review",
            "x": box[0], "y": box[1], "w": box[2]-box[0], "h": box[3]-box[1],
            "alpha_pixels": int((a0[:, :, 3] > 20).sum()),
            "head_pixels": 0, "body_pixels": 0, "eyes": 2,
            "changed_pixels": int(changed.sum()),
        })
    result.save(out_path, optimize=True)
    return records


def make_overview(name: str, records: list[dict[str, int | str]]) -> None:
    default = Image.open(DEFAULT / name).convert("RGBA")
    final = Image.open(OUTPUT / name).convert("RGBA")
    max_side = 1000
    scale = min(1.0, max_side / max(default.size))
    size = (max(1, round(default.width * scale)), max(1, round(default.height * scale)))
    d = default.resize(size, Image.Resampling.LANCZOS)
    f = final.resize(size, Image.Resampling.LANCZOS)
    panel = Image.new("RGB", (size[0] * 2, size[1] + 34), (31, 34, 39))
    for i, im in enumerate((d, f)):
        bg = Image.new("RGB", size, (54, 58, 65))
        bg.paste(im, (0, 0), im)
        panel.paste(bg, (i * size[0], 34))
    draw = ImageDraw.Draw(panel)
    font = ImageFont.load_default()
    draw.text((6, 8), f"{name}: Default | nailong_v2    transformed player components: {len(records)}", font=font, fill=(245, 245, 245))
    panel.save(QA / f"v2_overview_{Path(name).stem}.jpg", quality=92)


def write_reports(all_records: list[dict[str, int | str]]) -> None:
    csv_path = QA / "v2_transformed_components.csv"
    if all_records:
        with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_records[0].keys()))
            writer.writeheader()
            writer.writerows(all_records)

    default_files = {p.relative_to(DEFAULT).as_posix(): p for p in DEFAULT.rglob("*") if p.is_file()}
    output_files = {p.relative_to(OUTPUT).as_posix(): p for p in OUTPUT.rglob("*") if p.is_file()}
    modified = []
    unchanged = []
    mismatches = []
    for rel, src in default_files.items():
        dst = output_files.get(rel)
        if dst is None:
            mismatches.append({"file": rel, "issue": "missing from v2"})
            continue
        if src.suffix.lower() == ".png":
            with Image.open(src) as a, Image.open(dst) as b:
                if a.size != b.size or a.mode != b.mode:
                    mismatches.append({"file": rel, "issue": f"image metadata {a.mode} {a.size} -> {b.mode} {b.size}"})
        if sha256(src) == sha256(dst):
            unchanged.append(rel)
        else:
            modified.append(rel)
    extras = sorted(set(output_files) - set(default_files))
    per_atlas = {}
    for rec in all_records:
        per_atlas.setdefault(str(rec["atlas"]), 0)
        per_atlas[str(rec["atlas"])] += 1
    report = {
        "base": str(DEFAULT),
        "output": str(OUTPUT),
        "default_file_count": len(default_files),
        "output_file_count": len(output_files),
        "modified_files": sorted(modified),
        "unchanged_file_count": len(unchanged),
        "extra_files": extras,
        "metadata_mismatches": mismatches,
        "transformed_components_by_atlas": per_atlas,
        "transformed_component_count": len(all_records),
        "method": "Default alpha component + compact pale mask + enclosed black eye holes; pose, pivot, nail and effects retained from Default",
        "runtime_validation": "NOT RUN: Hollow Knight / CustomKnight installation not present on this machine",
    }
    (QA / "v2_build_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main() -> None:
    if not OUTPUT.exists():
        raise SystemExit(f"Create {OUTPUT} by copying Default before running this script")
    all_records: list[dict[str, int | str]] = []
    for name in PLAYER_ATLASES:
        if not (DEFAULT / name).exists():
            print(f"skip missing: {name}")
            continue
        print(f"processing {name}", flush=True)
        records = apply_atlas(name)
        all_records.extend(records)
        make_overview(name, records)
        print(f"  transformed={len(records)}", flush=True)
    write_reports(all_records)


if __name__ == "__main__":
    main()
