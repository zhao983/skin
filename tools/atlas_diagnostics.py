from __future__ import annotations

from pathlib import Path
from PIL import Image, ImageChops, ImageDraw


ROOT = Path(r"D:\skin")
OUT = ROOT / "_v2_diagnostics"
OUT.mkdir(exist_ok=True)


def checker(size: tuple[int, int], cell: int = 24) -> Image.Image:
    w, h = size
    bg = Image.new("RGB", size, (38, 42, 48))
    draw = ImageDraw.Draw(bg)
    alt = (58, 63, 70)
    for y in range(0, h, cell):
        for x in range(0, w, cell):
            if (x // cell + y // cell) % 2:
                draw.rectangle((x, y, min(x + cell - 1, w - 1), min(y + cell - 1, h - 1)), fill=alt)
    return bg


def composite_preview(src: Path, out_name: str, max_side: int = 1600) -> None:
    image = Image.open(src).convert("RGBA")
    scale = min(1.0, max_side / max(image.size))
    if scale < 1:
        image = image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS)
    bg = checker(image.size)
    bg.paste(image, mask=image.getchannel("A"))
    bg.save(OUT / out_name, optimize=True)


def quadrant_previews(src: Path, prefix: str) -> None:
    image = Image.open(src).convert("RGBA")
    tile = 2048
    for row in range(2):
        for col in range(2):
            crop = image.crop((col * tile, row * tile, (col + 1) * tile, (row + 1) * tile))
            bg = checker(crop.size, 32)
            bg.paste(crop, mask=crop.getchannel("A"))
            bg.resize((1024, 1024), Image.Resampling.LANCZOS).save(
                OUT / f"{prefix}_q{row}{col}.png", optimize=True
            )


def diff_preview(a_path: Path, b_path: Path, out_name: str) -> None:
    a = Image.open(a_path).convert("RGBA")
    b = Image.open(b_path).convert("RGBA")
    diff = ImageChops.difference(a, b)
    alpha = diff.convert("RGB").convert("L").point(lambda x: 255 if x else 0)
    shown = Image.new("RGBA", a.size, (255, 55, 50, 0))
    shown.putalpha(alpha)
    bg = checker(a.size, 48)
    bg.paste(shown, mask=shown.getchannel("A"))
    bg.resize((1600, 1600), Image.Resampling.NEAREST).save(OUT / out_name, optimize=True)


if __name__ == "__main__":
    targets = [
        (ROOT / "Default" / "Knight.png", "default_knight"),
        (ROOT / "nailong" / "Knight.png", "failed_knight"),
        (ROOT / "斯卡蒂" / "Knight.png", "skadi_knight"),
    ]
    for path, prefix in targets:
        composite_preview(path, f"{prefix}_overview.png")
        quadrant_previews(path, prefix)
    diff_preview(targets[0][0], targets[1][0], "default_vs_failed_diff.png")
    diff_preview(targets[0][0], targets[2][0], "default_vs_skadi_diff.png")
