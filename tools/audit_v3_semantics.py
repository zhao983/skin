from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

from PIL import Image

import build_nailong_v3 as build


def main() -> None:
    rows = list(csv.DictReader(build.V2_COVERAGE_CSV.open(encoding="utf-8-sig")))
    counts: Counter[str] = Counter()
    per_atlas: Counter[str] = Counter()
    cached = {}
    for index, row in enumerate(rows):
        if row["source"] == "manual_embedded_in_web_beam_review":
            counts["webbed"] += 1
            per_atlas[row["atlas"]] += 1
            continue
        name = row["atlas"]
        if name not in cached:
            cached[name] = (
                Image.open(build.DEFAULT / name).convert("RGBA"),
                Image.open(build.V2 / name).convert("RGBA"),
            )
        x0, y0, w, h = [int(row[k]) for k in ("x", "y", "w", "h")]
        x1, y1 = x0 + w, y0 + h
        source, v2 = cached[name]
        pad = 64
        box = (max(0, x0-pad), max(0, y0-pad), min(source.width, x1+pad), min(source.height, y1+pad))
        local_target = (x0 - box[0], y0 - box[1], x1 - box[0], y1 - box[1])
        info = build.locate_from_v2_anchor(source.crop(box), v2.crop(box), local_target)
        if info is None:
            raise RuntimeError(f"anchor failed: {name} {x0},{y0}")
        _, _, _, semantic = build.choose_donor(info, x0 * 31 + y0 * 17 + index)
        counts[semantic] += 1
        per_atlas[name] += 1
    print("total", sum(counts.values()))
    print("semantics")
    for key, value in sorted(counts.items()):
        print(f"  {key}: {value}")
    print("atlases")
    for key, value in sorted(per_atlas.items()):
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
