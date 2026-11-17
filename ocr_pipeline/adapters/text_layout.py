from __future__ import annotations

import re
import statistics
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass

_CJK_LANGUAGES = {"zh-Hant"}
_CJK_CHARACTERS = (
    "\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff"
    "。，、；：！？「」『』（）【】《》〈〉…—"
)
_CJK_SPACE_PATTERN = re.compile(
    rf"(?<=[{_CJK_CHARACTERS}])\s+(?=[{_CJK_CHARACTERS}])"
)
_BOX_COORDS = tuple[float, float, float, float]


@dataclass(frozen=True)
class TextBox:
    text: str
    bbox: _BOX_COORDS


@dataclass(frozen=True)
class TextLine:
    text: str
    bbox: _BOX_COORDS

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]


@dataclass
class _LineAccumulator:
    boxes: list[TextBox]

    def add(self, box: TextBox) -> None:
        self.boxes.append(box)

    @property
    def bbox(self) -> _BOX_COORDS:
        return union_bbox(box.bbox for box in self.boxes)


def normalize_text(text: str) -> str:
    """Normalize Unicode and OCR spacing without changing wording."""
    normalized = unicodedata.normalize("NFC", text)
    normalized = (
        normalized.replace("\u00a0", " ")
        .replace("\u200b", "")
        .replace("\ufeff", "")
    )
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r"\s+([,.;:!?])", r"\1", normalized)
    normalized = re.sub(r"(?<=[^\s:]):{2,}", ":", normalized)
    normalized = _CJK_SPACE_PATTERN.sub("", normalized)
    return normalized.strip()


def is_noise_text(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    return not any(character.isalnum() for character in stripped)


def union_bbox(
    boxes: Iterable[tuple[float, float, float, float]],
) -> _BOX_COORDS:
    coordinates = list(boxes)
    return (
        min(box[0] for box in coordinates),
        min(box[1] for box in coordinates),
        max(box[2] for box in coordinates),
        max(box[3] for box in coordinates),
    )


def join_text_boxes(boxes: list[TextBox], language: str = "vi") -> str:
    parts = [normalize_text(box.text) for box in boxes if box.text.strip()]
    parts = [part for part in parts if part]
    if not parts:
        return ""
    separator = "" if language in _CJK_LANGUAGES else " "
    return normalize_text(separator.join(parts))


def lines_from_boxes(
    boxes: list[TextBox],
    language: str = "vi",
) -> list[TextLine]:
    """Cluster free-form OCR boxes into visual lines in reading order."""
    usable = [
        box
        for box in boxes
        if box.text.strip() and not is_noise_text(box.text)
    ]
    ordered = sorted(
        usable,
        key=lambda box: ((box.bbox[1] + box.bbox[3]) / 2, box.bbox[0]),
    )
    accumulators: list[_LineAccumulator] = []
    for box in ordered:
        target: _LineAccumulator | None = None
        for accumulator in accumulators:
            line_bbox = accumulator.bbox
            overlap = min(line_bbox[3], box.bbox[3]) - max(
                line_bbox[1], box.bbox[1]
            )
            smallest_height = min(
                line_bbox[3] - line_bbox[1],
                box.bbox[3] - box.bbox[1],
            )
            if smallest_height > 0 and overlap / smallest_height >= 0.5:
                target = accumulator
                break
        if target is None:
            accumulators.append(_LineAccumulator([box]))
        else:
            target.add(box)

    lines: list[TextLine] = []
    for accumulator in accumulators:
        line_boxes = sorted(
            accumulator.boxes,
            key=lambda box: (box.bbox[0], box.bbox[1]),
        )
        lines.append(
            TextLine(
                text=join_text_boxes(line_boxes, language),
                bbox=accumulator.bbox,
            )
        )
    lines.sort(key=lambda line: ((line.bbox[1] + line.bbox[3]) / 2, line.bbox[0]))
    return [line for line in lines if line.text]


def lines_to_paragraphs(
    lines: list[TextLine],
) -> list[list[TextLine]]:
    """Split visual lines into paragraphs using gaps and indentation."""
    if not lines:
        return []
    heights = [line.height for line in lines if line.height > 0]
    median_height = statistics.median(heights) if heights else 1.0
    left_edge = min(line.bbox[0] for line in lines)
    paragraphs: list[list[TextLine]] = [[lines[0]]]
    for previous, line in zip(lines, lines[1:]):
        gap = line.bbox[1] - previous.bbox[3]
        indent = line.bbox[0] - left_edge
        previous_indent = previous.bbox[0] - left_edge
        starts_paragraph = (
            gap > max(2.0, 0.55 * median_height)
            or (
                indent > 0.9 * median_height
                and previous_indent < 0.4 * median_height
            )
        )
        if starts_paragraph:
            paragraphs.append([line])
        else:
            paragraphs[-1].append(line)
    return paragraphs


def _join_paragraph_text(lines: list[TextLine], language: str) -> str:
    result = ""
    for line in lines:
        text = normalize_text(line.text)
        if not text:
            continue
        if not result:
            result = text
            continue
        starts_lower = text[0].islower()
        if result.endswith("-") and starts_lower:
            if language in _CJK_LANGUAGES or language == "vi":
                result = result[:-1].rstrip() + " " + text
            else:
                result = result[:-1] + text
        else:
            result += "\n" + text
    return result


def markdown_from_paragraphs(
    paragraphs: list[list[TextLine]],
    language: str = "vi",
) -> str:
    blocks = [_join_paragraph_text(lines, language) for lines in paragraphs]
    return "\n\n".join(block for block in blocks if block)


def normalize_box(
    bbox: _BOX_COORDS,
    width: int,
    height: int,
) -> list[float]:
    if width <= 0 or height <= 0:
        raise ValueError("Page dimensions must be positive.")
    x0, y0, x1, y1 = bbox
    return [
        max(0.0, min(1.0, x0 / width)),
        max(0.0, min(1.0, y0 / height)),
        max(0.0, min(1.0, x1 / width)),
        max(0.0, min(1.0, y1 / height)),
    ]
