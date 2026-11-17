from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

DEFAULT_MIN_TEXT_SIDE = 2000
DEFAULT_MAX_SCALE = 2.0
DEFAULT_MAX_SKEW_DEGREES = 5.0
SKEW_STEP_DEGREES = 0.25
SKEW_SEARCH_SIDE = 800


@dataclass(frozen=True)
class PreparedPage:
    """A page prepared for OCR while keeping the source page untouched."""

    image: Image.Image
    source_size: tuple[int, int]
    scale: float
    skew_degrees: float
    contrast_adjusted: bool

    @property
    def size(self) -> tuple[int, int]:
        return self.image.size

    def metadata(self) -> dict[str, object]:
        return {
            "source_size": list(self.source_size),
            "prepared_size": list(self.size),
            "scale": round(self.scale, 4),
            "skew_degrees": round(self.skew_degrees, 3),
            "contrast_adjusted": self.contrast_adjusted,
        }


def _ink_projection_score(image: Image.Image, angle: float) -> float:
    rotated = image.rotate(
        angle,
        resample=Image.BILINEAR,
        expand=False,
        fillcolor=0,
    )
    rows = np.asarray(rotated, dtype=np.float32).sum(axis=1)
    return float(np.var(rows))


def _as_grayscale(image: Image.Image) -> Image.Image:
    if image.mode == "L":
        return image
    return image.convert("L")


def estimate_skew_degrees(
    image: Image.Image,
    max_degrees: float = DEFAULT_MAX_SKEW_DEGREES,
    step: float = SKEW_STEP_DEGREES,
) -> float:
    """Estimate the rotation that makes text rows horizontal.

    The returned value is the angle to pass to ``Image.rotate`` so that a
    skewed page is straightened. The estimate uses a projection-profile
    search on a downscaled binarized copy, which is deterministic and needs
    no OpenCV.
    """
    image = _as_grayscale(image)
    width, height = image.size
    if width < 2 or height < 2 or max_degrees <= 0:
        return 0.0
    factor = min(1.0, SKEW_SEARCH_SIDE / max(width, height))
    small = image.resize(
        (max(2, int(round(width * factor))), max(2, int(round(height * factor)))),
        Image.BILINEAR,
    )
    ink = (np.asarray(small) < 128).astype(np.uint8) * 255
    binary = Image.fromarray(ink, mode="L")

    straight_score = _ink_projection_score(binary, 0.0)
    best_angle = 0.0
    best_score = straight_score
    angle = -max_degrees
    while angle <= max_degrees + 1e-9:
        score = _ink_projection_score(binary, angle)
        if score > best_score:
            best_score = score
            best_angle = angle
        angle += step

    if abs(best_angle) < step or straight_score >= best_score * 1.02:
        return 0.0
    return best_angle


def prepare_page(
    image_path: Path,
    *,
    min_text_side: int = DEFAULT_MIN_TEXT_SIDE,
    max_scale: float = DEFAULT_MAX_SCALE,
    deskew: bool = True,
    max_skew_degrees: float = DEFAULT_MAX_SKEW_DEGREES,
    grayscale: bool = True,
) -> PreparedPage:
    """Load the first frame, normalize scale/contrast, and straighten it.

    No file is written: the prepared image stays in memory. Callers keep
    normalizing bounding boxes against ``prepared.size``; because scaling and
    deskew preserve the page frame, those normalized boxes still map onto the
    original TIFF that Step 5 crops from.
    """
    with Image.open(image_path) as source:
        try:
            source.seek(0)
        except EOFError:
            pass
        page = source.convert("L") if grayscale else source.convert("RGB")
    source_size = page.size

    scale = 1.0
    if min_text_side > 0 and min(page.size) < min_text_side:
        scale = min(min_text_side / min(page.size), max_scale)
        if scale > 1.001:
            page = page.resize(
                (
                    max(1, int(round(page.width * scale))),
                    max(1, int(round(page.height * scale))),
                ),
                Image.LANCZOS,
            )
        else:
            scale = 1.0

    contrast_adjusted = False
    pixels = np.asarray(page)
    low, high = np.percentile(pixels, (1, 99))
    if 0 < float(high) - float(low) < 90:
        page = ImageOps.autocontrast(page, cutoff=1)
        contrast_adjusted = True

    skew_degrees = 0.0
    if deskew:
        skew_degrees = estimate_skew_degrees(page, max_skew_degrees)
        if skew_degrees:
            page = page.rotate(
                skew_degrees,
                resample=Image.BICUBIC,
                expand=False,
                fillcolor=255,
            )

    return PreparedPage(
        image=page,
        source_size=source_size,
        scale=scale,
        skew_degrees=skew_degrees,
        contrast_adjusted=contrast_adjusted,
    )
