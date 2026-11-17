import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from PIL import Image

from ocr_pipeline.adapters.tesseract_ocr import (
    TesseractOCRAdapter,
    _normalize_tesseract_result,
)
from ocr_pipeline.adapters.vietocr_ocr import (
    _line_boxes,
    _merge_symbol_tokens,
)
from ocr_pipeline.steps.step4_ocr_pages import (
    Step4Error,
    load_ocr_adapter,
)


class AlternativeOCRAdapterTests(unittest.TestCase):
    def test_tesseract_normalizes_words_and_groups_markdown_lines(self) -> None:
        content = _normalize_tesseract_result(
            {
                "text": ["", "Xin", "chào", "Việt"],
                "left": [0, 10, 40, 10],
                "top": [0, 10, 10, 30],
                "width": [0, 25, 30, 20],
                "height": [0, 10, 10, 10],
                "block_num": [0, 1, 1, 1],
                "par_num": [0, 1, 1, 1],
                "line_num": [0, 1, 1, 2],
            },
            image_width=100,
            image_height=100,
        )

        self.assertEqual(content.markdown, "Xin chào\nViệt")
        self.assertEqual(
            [region.text for region in content.regions],
            ["Xin chào", "Việt"],
        )
        self.assertEqual(content.regions[0].bbox, [0.1, 0.1, 0.7, 0.2])

    def test_vietocr_groups_tesseract_words_into_lines(self) -> None:
        lines = _line_boxes(
            {
                "text": ["Xin", "chào", "Việt"],
                "left": [10, 40, 10],
                "top": [10, 10, 30],
                "width": [25, 30, 20],
                "height": [10, 10, 10],
                "block_num": [1, 1, 1],
                "par_num": [1, 1, 1],
                "line_num": [1, 1, 2],
            }
        )

        self.assertEqual(
            [line.tesseract_text for line in lines],
            ["Xin chào", "Việt"],
        )
        self.assertEqual(lines[0].bbox, (10.0, 10.0, 70.0, 20.0))

    def test_vietocr_restores_matching_tesseract_punctuation(self) -> None:
        self.assertEqual(
            _merge_symbol_tokens("Xin chao", "XIN chao."),
            "XIN chao.",
        )

    def test_tesseract_adapter_runs_through_the_model_loader(self) -> None:
        tesseract_module = ModuleType("pytesseract")
        setattr(tesseract_module, "pytesseract", SimpleNamespace(tesseract_cmd=None))
        setattr(tesseract_module, "Output", SimpleNamespace(DICT="dict"))
        setattr(tesseract_module, "get_tesseract_version", lambda: "5.5.0")
        def fake_image_to_data(image: Image.Image, **kwargs: object) -> dict[str, object]:
            if image.size != (200, 200):
                raise AssertionError("Unexpected test image size.")
            if kwargs["lang"] != "vie":
                raise AssertionError("Vietnamese must use the Tesseract vie model.")
            return {
                "text": ["Xin", "chào"],
                "left": [10, 40],
                "top": [10, 10],
                "width": [25, 30],
                "height": [10, 10],
                "block_num": [1, 1],
                "par_num": [1, 1],
                "line_num": [1, 1],
            }

        setattr(tesseract_module, "image_to_data", fake_image_to_data)
        with patch.dict(sys.modules, {"pytesseract": tesseract_module}):
            adapter = load_ocr_adapter("tesseract", language="vi")
            with TemporaryDirectory() as temp_dir:
                image_path = Path(temp_dir) / "page.png"
                Image.new("RGB", (100, 100), "white").save(image_path)
                content, raw = adapter.recognize(image_path)

        self.assertIsInstance(adapter, TesseractOCRAdapter)
        self.assertEqual(content.markdown, "Xin chào")
        assert isinstance(raw, dict)
        self.assertEqual(raw["language"], "vie")
        json.dumps(raw)

if __name__ == "__main__":
    unittest.main()
