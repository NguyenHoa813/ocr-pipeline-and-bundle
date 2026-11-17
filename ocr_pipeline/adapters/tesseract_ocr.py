from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

from ocr_pipeline.steps.step4_ocr_pages import (
    LayoutRegion,
    PageOCRContent,
    Step4Error,
)


_LANGUAGES = {
    "vi": "vie",
    "zh-Hant": "chi_tra",
    "fr": "fra",
    "en": "eng",
}


def _integer(value: object, field: str, index: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise Step4Error(
            f"Tesseract returned an invalid {field} for word {index}."
        )
    try:
        return int(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise Step4Error(
            f"Tesseract returned an invalid {field} for word {index}."
        ) from exc


def _word_bbox(
    left: object,
    top: object,
    width: object,
    height: object,
    image_width: int,
    image_height: int,
    index: int,
) -> list[float]:
    x0 = _integer(left, "left coordinate", index)
    y0 = _integer(top, "top coordinate", index)
    word_width = _integer(width, "width", index)
    word_height = _integer(height, "height", index)
    x1 = x0 + word_width
    y1 = y0 + word_height
    if (
        x0 < 0
        or y0 < 0
        or x1 > image_width
        or y1 > image_height
        or x0 >= x1
        or y0 >= y1
    ):
        raise Step4Error(
            f"Tesseract returned out-of-page coordinates for word {index}."
        )
    return [
        x0 / image_width,
        y0 / image_height,
        x1 / image_width,
        y1 / image_height,
    ]


def _normalize_tesseract_result(
    data: dict[str, Any],
    image_width: int,
    image_height: int,
) -> PageOCRContent:
    required_fields = (
        "text",
        "left",
        "top",
        "width",
        "height",
        "block_num",
        "par_num",
        "line_num",
    )
    if any(not isinstance(data.get(field), (list, tuple)) for field in required_fields):
        raise Step4Error("Tesseract returned an invalid TSV result.")
    lengths = {len(data[field]) for field in required_fields}
    if len(lengths) != 1:
        raise Step4Error("Tesseract returned mismatched TSV columns.")

    regions: list[LayoutRegion] = []
    lines: list[str] = []
    previous_line: tuple[object, object, object] | None = None
    previous_paragraph: tuple[object, object] | None = None
    for index, value in enumerate(data["text"]):
        text = value.strip() if isinstance(value, str) else ""
        if not text:
            continue
        line = (
            data["block_num"][index],
            data["par_num"][index],
            data["line_num"][index],
        )
        paragraph = line[:2]
        if previous_line is None:
            lines.append(text)
        elif line == previous_line:
            lines[-1] += f" {text}"
        elif paragraph == previous_paragraph:
            lines.append(text)
        else:
            lines.extend(("", text))
        previous_line = line
        previous_paragraph = paragraph
        regions.append(
            LayoutRegion(
                type="text",
                text=text,
                bbox=_word_bbox(
                    data["left"][index],
                    data["top"][index],
                    data["width"][index],
                    data["height"][index],
                    image_width,
                    image_height,
                    index,
                ),
            )
        )

    return PageOCRContent(markdown="\n".join(lines), regions=regions)


class TesseractOCRAdapter:
    """Tesseract 5 OCR adapter; emits word-level text regions."""

    model_name = "Tesseract OCR"
    model_version = "Tesseract 5"

    def __init__(
        self,
        language: str = "vi",
        tesseract_cmd: str | None = None,
    ) -> None:
        if language not in _LANGUAGES:
            raise Step4Error(
                "Tesseract language must be one of: "
                + ", ".join(_LANGUAGES)
                + "."
            )
        try:
            import pytesseract
        except ImportError as exc:
            raise Step4Error(
                "Tesseract adapter requires the optional dependency. "
                "Install it with `python -m pip install -e \".[tesseract]\"`."
            ) from exc

        executable = tesseract_cmd or os.environ.get("TESSERACT_CMD")
        if executable:
            pytesseract.pytesseract.tesseract_cmd = executable
        self._pytesseract = pytesseract
        self.language = language
        self.tesseract_language = _LANGUAGES[language]
        try:
            version = pytesseract.get_tesseract_version()
        except Exception as exc:
            raise Step4Error(
                "Cannot run the Tesseract executable. Install Tesseract 5 "
                "and its language data, or set TESSERACT_CMD. "
                f"Details: {exc}"
            ) from exc
        self.model_version = f"Tesseract {version}"

    def recognize(self, image_path: Path) -> tuple[PageOCRContent, object]:
        try:
            with Image.open(image_path) as image:
                image_width, image_height = image.size
                data = self._pytesseract.image_to_data(
                    image,
                    lang=self.tesseract_language,
                    output_type=self._pytesseract.Output.DICT,
                )
            raw_response: object = {
                "engine": self.model_name,
                "version": self.model_version,
                "language": self.tesseract_language,
                "data": data,
            }
            if not isinstance(data, dict):
                raise Step4Error("Tesseract returned a non-object TSV result.")
            content = _normalize_tesseract_result(
                data,
                image_width,
                image_height,
            )
            return content, raw_response
        except Step4Error as exc:
            if getattr(exc, "raw_response", None) is not None:
                raise
            raise Step4Error(str(exc), locals().get("raw_response")) from exc
        except (OSError, UnidentifiedImageError) as exc:
            raise Step4Error(
                f"Cannot read OCR page image {image_path}: {exc}"
            ) from exc
        except Exception as exc:
            raise Step4Error(
                f"Tesseract inference failed for {image_path}: {exc}"
            ) from exc
