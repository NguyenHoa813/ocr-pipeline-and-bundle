from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Protocol

from ocr_pipeline.adapters.paddle_ocr import PaddleOCRAdapter
from ocr_pipeline.adapters.vietocr_ocr import VietOCRAdapter
from ocr_pipeline.steps.step4_ocr_pages import (
    LayoutRegion,
    PageOCRContent,
    Step4Error,
)


class OCRAdapter(Protocol):
    model_name: str
    model_version: str

    def recognize(self, image_path: Path) -> tuple[PageOCRContent, object]: ...


_TEXT_LAYOUT_TYPES = {
    "heading",
    "page_header",
    "page_footer",
    "page_number",
}
_GROUPED_LAYOUT_TYPES = {"formula", "table"}


def _bbox_coverage(
    text_bbox: list[float] | None,
    layout_bbox: list[float] | None,
) -> float:
    if text_bbox is None or layout_bbox is None:
        return 0.0
    tx0, ty0, tx1, ty1 = text_bbox
    lx0, ly0, lx1, ly1 = layout_bbox
    text_area = (tx1 - tx0) * (ty1 - ty0)
    if text_area <= 0:
        return 0.0
    intersection_width = max(0.0, min(tx1, lx1) - max(tx0, lx0))
    intersection_height = max(0.0, min(ty1, ly1) - max(ty0, ly0))
    return intersection_width * intersection_height / text_area


def merge_text_and_layout(
    text_content: PageOCRContent,
    layout_content: PageOCRContent,
) -> PageOCRContent:
    """Keep recognized text from VietOCR and use Paddle only for layout."""
    layout_matches: dict[int, list[int]] = {}
    line_layout_matches: dict[int, int] = {}
    for text_index, text_region in enumerate(text_content.regions):
        candidates: list[tuple[float, float, int]] = []
        for layout_index, layout_region in enumerate(layout_content.regions):
            if layout_region.type not in (
                _TEXT_LAYOUT_TYPES | _GROUPED_LAYOUT_TYPES
            ):
                continue
            coverage = _bbox_coverage(text_region.bbox, layout_region.bbox)
            if coverage < 0.5:
                continue
            layout_area = (
                (layout_region.bbox[2] - layout_region.bbox[0])
                * (layout_region.bbox[3] - layout_region.bbox[1])
                if layout_region.bbox is not None
                else 1.0
            )
            candidates.append((coverage, -layout_area, layout_index))
        if not candidates:
            continue
        layout_index = max(candidates)[2]
        layout_region = layout_content.regions[layout_index]
        if layout_region.type in _GROUPED_LAYOUT_TYPES:
            layout_matches.setdefault(layout_index, []).append(text_index)
        else:
            line_layout_matches[text_index] = layout_index

    text_regions: list[LayoutRegion] = []
    grouped_text_indices = {
        text_index
        for indices in layout_matches.values()
        for text_index in indices
    }
    for index, region in enumerate(text_content.regions):
        if index in grouped_text_indices:
            continue
        layout_index = line_layout_matches.get(index)
        text_regions.append(
            replace(region, type=layout_content.regions[layout_index].type)
            if layout_index is not None
            else region
        )

    grouped_regions: list[LayoutRegion] = []
    figure_regions: list[LayoutRegion] = []
    other_regions: list[LayoutRegion] = []
    for layout_index, region in enumerate(layout_content.regions):
        if region.type in _GROUPED_LAYOUT_TYPES:
            text = "\n".join(
                text_content.regions[index].text
                for index in layout_matches.get(layout_index, [])
                if text_content.regions[index].text
            )
            grouped_regions.append(replace(region, text=text))
        elif region.type in _TEXT_LAYOUT_TYPES and not any(
            matched_layout_index == layout_index
            for matched_layout_index in line_layout_matches.values()
        ):
            other_regions.append(replace(region, text=""))
        elif region.type in {"figure", "stamp"}:
            structural_region = replace(region, text="")
            if region.type == "figure":
                figure_regions.append(structural_region)
            else:
                other_regions.append(structural_region)

    figure_regions.sort(
        key=lambda region: (
            region.bbox[1] if region.bbox is not None else 1.0,
            region.bbox[0] if region.bbox is not None else 1.0,
        )
    )
    return PageOCRContent(
        markdown=text_content.markdown,
        regions=[*text_regions, *grouped_regions, *other_regions, *figure_regions],
    )


class VietOCRLayoutAdapter:
    """VietOCR/Tesseract transcription with PaddleOCR layout regions."""

    model_name = "VietOCR + Tesseract (PaddleOCR layout)"

    def __init__(
        self,
        language: str = "vi",
        preprocess: bool = True,
        min_text_side: int = 2400,
        variant: str = "vgg_transformer",
        weights_path: Path | None = None,
        text_adapter: OCRAdapter | None = None,
        layout_adapter: OCRAdapter | None = None,
    ) -> None:
        self.language = language
        self._text_adapter = text_adapter or VietOCRAdapter(
            language=language,
            preprocess=preprocess,
            min_text_side=min_text_side,
            variant=variant,
            weights_path=weights_path,
        )
        self._layout_adapter = layout_adapter or PaddleOCRAdapter(
            language="vi",
            preprocess=True,
            min_text_side=0,
        )
        self.model_version = (
            f"{self._text_adapter.model_version}; "
            f"{self._layout_adapter.model_version} (layout)"
        )
        self.configuration = {
            "text_model": self._text_adapter.model_name,
            "text_configuration": getattr(
                self._text_adapter,
                "configuration",
                {},
            ),
            "layout_model": self._layout_adapter.model_name,
            "layout_configuration": getattr(
                self._layout_adapter,
                "configuration",
                {},
            ),
        }

    def recognize(self, image_path: Path) -> tuple[PageOCRContent, object]:
        text_content, text_raw = self._text_adapter.recognize(image_path)
        try:
            layout_content, layout_raw = self._layout_adapter.recognize(image_path)
        except Step4Error as exc:
            raise Step4Error(
                f"PaddleOCR layout extraction failed: {exc}",
                {
                    "text_recognition": text_raw,
                    "layout_error": str(exc),
                },
            ) from exc
        return (
            merge_text_and_layout(text_content, layout_content),
            {
                "engine": self.model_name,
                "version": self.model_version,
                "language": self.language,
                "text_recognition": text_raw,
                "layout": layout_raw,
            },
        )
