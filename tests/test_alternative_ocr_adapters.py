import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from PIL import Image

from ocr_pipeline.adapters.easyocr import (
    EASYOCR_CACHE_DIR,
    EasyOCRAdapter,
    _normalize_easyocr_result,
)
from ocr_pipeline.adapters.tesseract_ocr import (
    TesseractOCRAdapter,
    _normalize_tesseract_result,
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
            ["Xin", "chào", "Việt"],
        )
        self.assertEqual(content.regions[0].bbox, [0.1, 0.1, 0.35, 0.2])

    def test_easyocr_normalizes_polygons_and_preserves_reading_order(self) -> None:
        content = _normalize_easyocr_result(
            [
                ([[10, 10], [50, 10], [50, 30], [10, 30]], "Xin chào", 0.9),
                ([[5, 40], [45, 40], [45, 60], [5, 60]], "Việt Nam", 0.8),
            ],
            image_width=100,
            image_height=100,
        )

        self.assertEqual(content.markdown, "Xin chào\nViệt Nam")
        self.assertEqual(content.regions[0].type, "text")
        self.assertEqual(content.regions[0].bbox, [0.1, 0.1, 0.5, 0.3])

    def test_easyocr_rejects_out_of_page_boxes(self) -> None:
        with self.assertRaisesRegex(Step4Error, "does not intersect the page"):
            _normalize_easyocr_result(
                [([[-20, 0], [-5, 0], [-5, 10], [-20, 10]], "text", 0.9)],
                image_width=100,
                image_height=100,
            )

    def test_easyocr_clips_partially_out_of_page_boxes(self) -> None:
        content = _normalize_easyocr_result(
            [([[-5, 10], [20, 10], [20, 30], [-5, 30]], "text", 0.9)],
            image_width=100,
            image_height=100,
        )

        self.assertEqual(content.regions[0].bbox, [0.0, 0.1, 0.2, 0.3])

    def test_tesseract_adapter_runs_through_the_model_loader(self) -> None:
        tesseract_module = ModuleType("pytesseract")
        setattr(tesseract_module, "pytesseract", SimpleNamespace(tesseract_cmd=None))
        setattr(tesseract_module, "Output", SimpleNamespace(DICT="dict"))
        setattr(tesseract_module, "get_tesseract_version", lambda: "5.5.0")
        def fake_image_to_data(image: Image.Image, **kwargs: object) -> dict[str, object]:
            if image.size != (100, 100):
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

    def test_easyocr_adapter_uses_local_model_cache_and_language(self) -> None:
        initialized: dict[str, object] = {}

        class FakePolygon:
            def tolist(self) -> list[list[int]]:
                return [[10, 10], [50, 10], [50, 30], [10, 30]]

        class FakeReader:
            def __init__(self, languages: list[str], **kwargs: object) -> None:
                initialized["languages"] = languages
                initialized.update(kwargs)

            def readtext(
                self,
                image_array: object,
                **kwargs: object,
            ) -> list[object]:
                if getattr(image_array, "shape", None) != (100, 100, 3):
                    raise AssertionError("EasyOCR must receive a decoded RGB image.")
                if kwargs != {"detail": 1, "paragraph": False}:
                    raise AssertionError("Unexpected EasyOCR output options.")
                return [
                    (FakePolygon(), "Xin chào", 0.9)
                ]

        easyocr_module = ModuleType("easyocr")
        setattr(easyocr_module, "Reader", FakeReader)
        setattr(easyocr_module, "__version__", "1.7.2")
        with patch.dict(sys.modules, {"easyocr": easyocr_module}):
            adapter = load_ocr_adapter("easyocr", language="zh-Hant")
            with TemporaryDirectory() as temp_dir:
                image_path = Path(temp_dir) / "page.png"
                Image.new("RGB", (100, 100), "white").save(image_path)
                content, raw = adapter.recognize(image_path)

        self.assertIsInstance(adapter, EasyOCRAdapter)
        self.assertEqual(initialized["languages"], ["ch_tra"])
        self.assertFalse(initialized["gpu"])
        self.assertEqual(
            initialized["model_storage_directory"],
            str(EASYOCR_CACHE_DIR),
        )
        self.assertEqual(content.markdown, "Xin chào")
        assert isinstance(raw, dict)
        json.dumps(raw)


if __name__ == "__main__":
    unittest.main()
