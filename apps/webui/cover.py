from __future__ import annotations

import math
import shutil
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


DEFAULT_SAMPLE_RATIOS = (0.06, 0.12, 0.2, 0.32, 0.44, 0.56, 0.68, 0.8, 0.9)
COVER_SIZE = (1920, 1080)
CONTACT_CELL = (480, 270)
COVER_TEMPLATES = ("cinema", "news", "clean")


def run_command(args: list[str]) -> str:
    completed = subprocess.run(
        args,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return completed.stdout.strip()


def require_tool(name: str) -> None:
    if not shutil.which(name):
        raise RuntimeError(f"{name} was not found on PATH.")


def probe_duration(video_path: Path) -> float:
    require_tool("ffprobe")
    output = run_command(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ]
    )
    duration = float(output)
    if not math.isfinite(duration) or duration <= 0:
        raise RuntimeError(f"Could not read duration for {video_path}.")
    return duration


def parse_times(value: str | None, duration: float) -> list[float]:
    if value:
        times = [float(item.strip()) for item in value.split(",") if item.strip()]
    else:
        times = [duration * ratio for ratio in DEFAULT_SAMPLE_RATIOS]
    return [min(max(time_value, 0.5), max(duration - 0.5, 0.5)) for time_value in times]


def extract_frames(video_path: Path, frame_dir: Path, times: list[float]) -> list[Path]:
    require_tool("ffmpeg")
    frame_dir.mkdir(parents=True, exist_ok=True)
    frames: list[Path] = []
    for index, time_value in enumerate(times, start=1):
        frame_path = frame_dir / f"frame_{index:02d}_{time_value:.1f}s.jpg"
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-ss",
                f"{time_value:.3f}",
                "-i",
                str(video_path),
                "-frames:v",
                "1",
                "-q:v",
                "2",
                str(frame_path),
            ],
            check=True,
        )
        frames.append(frame_path)
    return frames


def resampling_filter() -> int:
    try:
        return Image.Resampling.LANCZOS
    except AttributeError:
        return Image.LANCZOS


def cover_crop(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    target_w, target_h = size
    source_w, source_h = image.size
    target_ratio = target_w / target_h
    source_ratio = source_w / source_h
    if source_ratio > target_ratio:
        crop_w = int(source_h * target_ratio)
        left = (source_w - crop_w) // 2
        box = (left, 0, left + crop_w, source_h)
    else:
        crop_h = int(source_w / target_ratio)
        top = (source_h - crop_h) // 2
        box = (0, top, source_w, top + crop_h)
    return image.crop(box).resize(size, resampling_filter())


def find_font(bold: bool = False) -> str | None:
    windows = Path("C:/Windows/Fonts")
    names = ("msyhbd.ttc", "simhei.ttf", "arialbd.ttf") if bold else ("msyh.ttc", "simhei.ttf", "arial.ttf")
    for name in names:
        font_path = windows / name
        if font_path.exists():
            return str(font_path)
    return None


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    font_path = find_font(bold=bold)
    if font_path:
        return ImageFont.truetype(font_path, size=size)
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def text_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> int:
    if not text:
        return 0
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0]


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    if not text:
        return []
    has_spaces = " " in text
    units = text.split(" ") if has_spaces else list(text)
    lines: list[str] = []
    current = ""
    joiner = " " if has_spaces else ""
    for unit in units:
        candidate = unit if not current else f"{current}{joiner}{unit}"
        if text_width(draw, candidate, font) <= max_width:
            current = candidate
            continue
        if current:
            lines.append(current)
        current = unit
    if current:
        lines.append(current)
    return lines


def fit_wrapped_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    max_width: int,
    max_lines: int,
    start_size: int,
    min_size: int,
    bold: bool,
) -> tuple[ImageFont.ImageFont, list[str]]:
    for size in range(start_size, min_size - 1, -2):
        font = load_font(size=size, bold=bold)
        lines = wrap_text(draw, text, font, max_width)
        if len(lines) <= max_lines:
            return font, lines
    font = load_font(size=min_size, bold=bold)
    lines = wrap_text(draw, text, font, max_width)[:max_lines]
    if lines:
        lines[-1] = lines[-1].rstrip(". ") + "..."
    return font, lines


def add_bottom_gradient(image: Image.Image) -> Image.Image:
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    width, height = image.size
    start_y = int(height * 0.42)
    for y in range(start_y, height):
        alpha = int(205 * ((y - start_y) / (height - start_y)) ** 1.35)
        draw.line((0, y, width, y), fill=(0, 0, 0, alpha))
    return Image.alpha_composite(image.convert("RGBA"), overlay)


def add_top_gradient(image: Image.Image) -> Image.Image:
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    width, height = image.size
    end_y = int(height * 0.48)
    for y in range(0, end_y):
        alpha = int(150 * (1 - y / end_y) ** 1.4)
        draw.line((0, y, width, y), fill=(0, 0, 0, alpha))
    return Image.alpha_composite(image.convert("RGBA"), overlay)


def draw_cover_text(image: Image.Image, title: str, subtitle: str, accent: str, template: str = "cinema") -> Image.Image:
    template = template if template in COVER_TEMPLATES else "cinema"
    image = add_top_gradient(image) if template == "clean" else add_bottom_gradient(image)
    draw = ImageDraw.Draw(image)
    max_width = image.width - 220
    x = 110
    y = 700
    title_fill = (255, 255, 255, 255)
    subtitle_fill = (239, 244, 250, 255)
    accent_fill = (255, 218, 92, 255)

    if template == "news":
        band_top = int(image.height * 0.66)
        draw.rectangle((0, band_top, image.width, image.height), fill=(12, 24, 31, 218))
        draw.rectangle((0, band_top, 18, image.height), fill=(22, 134, 111, 255))
        y = band_top + 70
        max_width = image.width - 260
    elif template == "clean":
        x = 92
        y = 92
        max_width = int(image.width * 0.72)
        accent_fill = (112, 214, 181, 255)

    title_font, title_lines = fit_wrapped_text(
        draw,
        title,
        max_width=max_width,
        max_lines=2,
        start_size=118 if template != "news" else 104,
        min_size=58,
        bold=True,
    )
    subtitle_font, subtitle_lines = fit_wrapped_text(
        draw,
        subtitle,
        max_width=max_width,
        max_lines=1,
        start_size=48,
        min_size=34,
        bold=False,
    )

    accent_font = load_font(size=34, bold=True)
    if accent:
        draw.text((x, y - 54), accent.upper(), font=accent_font, fill=accent_fill)

    shadow = (0, 0, 0, 190)
    line_gap = 10
    for line in title_lines:
        draw.text((x + 3, y + 3), line, font=title_font, fill=shadow)
        draw.text((x, y), line, font=title_font, fill=title_fill)
        bbox = draw.textbbox((x, y), line, font=title_font)
        y = bbox[3] + line_gap

    y += 14
    for line in subtitle_lines:
        draw.text((x + 2, y + 2), line, font=subtitle_font, fill=shadow)
        draw.text((x, y), line, font=subtitle_font, fill=subtitle_fill)
        bbox = draw.textbbox((x, y), line, font=subtitle_font)
        y = bbox[3] + 8

    return image.convert("RGB")


def build_contact_sheet(frames: list[Path], output_path: Path) -> None:
    cols = 3
    rows = math.ceil(len(frames) / cols)
    label_h = 38
    sheet = Image.new("RGB", (CONTACT_CELL[0] * cols, (CONTACT_CELL[1] + label_h) * rows), "white")
    draw = ImageDraw.Draw(sheet)
    label_font = load_font(size=20, bold=False)

    for index, frame_path in enumerate(frames):
        image = Image.open(frame_path).convert("RGB")
        thumb = cover_crop(image, CONTACT_CELL)
        x = (index % cols) * CONTACT_CELL[0]
        y = (index // cols) * (CONTACT_CELL[1] + label_h)
        sheet.paste(thumb, (x, y))
        draw.text((x + 12, y + CONTACT_CELL[1] + 8), frame_path.stem, font=label_font, fill=(30, 30, 30))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)


def build_cover_candidates(frames: list[Path], output_dir: Path, title: str, subtitle: str, accent: str, template: str = "cinema") -> list[Path]:
    if not frames:
        raise ValueError("No frames were extracted.")
    choices = [
        (0, "cover_01_opening.jpg"),
        (min(2, len(frames) - 1), "cover_02_early.jpg"),
        (min(4, len(frames) - 1), "cover_03_middle.jpg"),
        (min(6, len(frames) - 1), "cover_04_late.jpg"),
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for frame_index, filename in choices:
        base = Image.open(frames[frame_index]).convert("RGB")
        cover = cover_crop(base, COVER_SIZE)
        cover = draw_cover_text(cover, title=title, subtitle=subtitle, accent=accent, template=template)
        output_path = output_dir / filename
        cover.save(output_path, quality=94, subsampling=0)
        outputs.append(output_path)
    return outputs


def suggest_title_from_video(video_path: Path) -> str:
    title = video_path.stem
    for separator in (" ｜ ", " | ", " - ", "_"):
        title = title.replace(separator, " ")
    title = " ".join(title.split())
    return title[:80] if title else "VIDEO COVER"
