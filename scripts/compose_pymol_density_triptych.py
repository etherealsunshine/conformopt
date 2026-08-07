"""Crop PyMOL framebuffer renders and compose a deck-ready triptych."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFont


PANELS = (
    ("8Q6Q_B_ASP81_experimental_3d.png", "Experimental density", "omit mFo–DFc"),
    ("8Q6Q_B_ASP81_denoised_3d.png", "Denoised density", "frozen 3D U-Net"),
    ("8Q6Q_B_ASP81_ground_truth_3d.png", "Ground-truth density", "deposited A/B synthetic target"),
)


def content_crop(image: Image.Image, padding: int = 35) -> Image.Image:
    rgb = image.convert("RGB")
    difference = ImageChops.difference(rgb, Image.new("RGB", rgb.size, "white"))
    difference = difference.point(lambda value: 255 if value > 8 else 0)
    box = difference.getbbox()
    if box is None:
        return rgb
    left, top, right, bottom = box
    return rgb.crop((
        max(0, left - padding), max(0, top - padding),
        min(rgb.width, right + padding), min(rgb.height, bottom + padding),
    ))


def fit(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    scale = min(size[0] / image.width, size[1] / image.height)
    return image.resize(
        (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
        Image.Resampling.LANCZOS,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    bold = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 58)
    regular = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 34)
    small = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 29)
    main_title = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 66)

    width, height = 4500, 1520
    panel_width = width // 3
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    heading = "8Q6Q  B:ASP81  ·  matched 3D density meshes"
    heading_box = draw.textbbox((0, 0), heading, font=main_title)
    draw.text(((width - (heading_box[2] - heading_box[0])) / 2, 28), heading, fill="#111827", font=main_title)

    for index, (filename, title, subtitle) in enumerate(PANELS):
        source = Image.open(args.input / filename)
        cropped = content_crop(source)
        cropped.save(args.output / filename.replace(".png", "_deck.png"), dpi=(300, 300))
        rendered = fit(cropped, (panel_width - 90, 1040))
        x = index * panel_width + (panel_width - rendered.width) // 2
        y = 275 + (1000 - rendered.height) // 2
        canvas.paste(rendered, (x, y))
        title_box = draw.textbbox((0, 0), title, font=bold)
        title_x = index * panel_width + (panel_width - (title_box[2] - title_box[0])) / 2
        draw.text((title_x, 135), title, fill="#111827", font=bold)
        subtitle_box = draw.textbbox((0, 0), subtitle, font=regular)
        subtitle_x = index * panel_width + (panel_width - (subtitle_box[2] - subtitle_box[0])) / 2
        draw.text((subtitle_x, 205), subtitle, fill="#4b5563", font=regular)

    legend_y = 1432
    items = (
        ("#00d8dc", "positive density, 1.5σ"),
        ("#ff9900", "deposited conformer A, 57%"),
        ("#e500e5", "deposited conformer B, 43%"),
    )
    widths = [95 + draw.textlength(label, font=small) for _, label in items]
    start = (width - sum(widths) - 130 * (len(items) - 1)) / 2
    for (color, label), item_width in zip(items, widths):
        draw.line((start, legend_y + 17, start + 68, legend_y + 17), fill=color, width=12)
        draw.text((start + 88, legend_y), label, fill="#1f2937", font=small)
        start += item_width + 130

    canvas.save(args.output / "8Q6Q_B_ASP81_density_comparison_3d_deck.png", dpi=(300, 300))


if __name__ == "__main__":
    main()
