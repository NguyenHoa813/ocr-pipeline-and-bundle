from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image

from ocr_pipeline.adapters.paddle_ocr import (
    PADDLEX_CACHE_DIR,
    PaddleOCRAdapter,
    TEXT_DET_LIMIT_SIDE_LEN,
    _parse_structure_result,
)
from ocr_pipeline.steps.step4_ocr_pages import Step4Error


class PaddleOCRAdapterTests(unittest.TestCase):
    def test_language_selects_a_compatible_recognition_model(self) -> None:
        cases = {
            "vi": "latin_PP-OCRv5_mobile_rec",
            "zh-Hant": "PP-OCRv6_medium_rec",
            "fr": "latin_PP-OCRv5_mobile_rec",
            "en": "PP-OCRv6_medium_rec",
        }
        for language, recognition_model in cases.items():
            with self.subTest(language=language):
                options: dict[str, object] = {}

                class FakePPStructure:
                    def __init__(self, **kwargs: object) -> None:
                        options.update(kwargs)

                paddleocr = ModuleType("paddleocr")
                setattr(paddleocr, "PPStructureV3", FakePPStructure)
                with patch.dict(sys.modules, {"paddleocr": paddleocr}):
                    adapter = PaddleOCRAdapter(language=language)

                self.assertEqual(
                    options["text_detection_model_name"],
                    "PP-OCRv6_medium_det",
                )
                self.assertEqual(
                    options["text_recognition_model_name"],
                    recognition_model,
                )
                self.assertIn(recognition_model, adapter.model_version)

    def test_paddlex_cache_is_project_local_and_created_before_runtime_import(self) -> None:
        import os

        self.assertEqual(
            os.environ["PADDLE_PDX_CACHE_HOME"],
            str(PADDLEX_CACHE_DIR),
        )
        self.assertEqual(
            PADDLEX_CACHE_DIR,
            Path(__file__).resolve().parents[1] / "cache" / "paddlex",
        )
        self.assertTrue(PADDLEX_CACHE_DIR.is_dir())
        self.assertEqual(os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"], "False")

    def test_layout_blocks_are_normalized_into_pipeline_regions(self) -> None:
        content = _parse_structure_result(
            {
                "parsing_res_list": [
                    {
                        "block_label": "paragraph_title",
                        "block_content": "Document title",
                        "block_bbox": [20, 10, 180, 30],
                    },
                    {
                        "block_label": "figure",
                        "block_content": "",
                        "block_bbox": [50, 25, 150, 75],
                    },
                    {
                        "block_label": "table",
                        "block_content": "<table><tr><td>data</td></tr></table>",
                        "block_bbox": [10, 40, 90, 90],
                    },
                    {
                        "block_label": "formula",
                        "block_content": "x+y=2",
                        "block_bbox": [100, 40, 190, 90],
                    },
                    {
                        "block_label": "seal",
                        "block_content": "approved",
                        "block_bbox": [10, 10, 40, 30],
                    },
                    {
                        "block_label": "header",
                        "block_content": "running head",
                        "block_bbox": [0, 0, 200, 8],
                    },
                    {
                        "block_label": "footer",
                        "block_content": "footer text",
                        "block_bbox": [0, 92, 200, 100],
                    },
                    {
                        "block_label": "number",
                        "block_content": "7",
                        "block_bbox": [180, 92, 198, 100],
                    },
                ]
            },
            width=200,
            height=100,
        )

        self.assertEqual(
            [region.type for region in content.regions],
            [
                "heading",
                "figure",
                "table",
                "formula",
                "stamp",
                "page_header",
                "page_footer",
                "page_number",
            ],
        )
        self.assertEqual(content.regions[1].bbox, [0.25, 0.25, 0.75, 0.75])
        self.assertIn("## Document title", content.markdown)
        self.assertIn("<table>", content.markdown)
        self.assertEqual(content.regions[1].image_kind, None)

    def test_out_of_page_layout_bbox_is_rejected(self) -> None:
        with self.assertRaisesRegex(Step4Error, "out-of-page"):
            _parse_structure_result(
                {
                    "parsing_res_list": [
                        {
                            "block_label": "image",
                            "block_content": "",
                            "block_bbox": [-1, 0, 20, 20],
                        }
                    ]
                },
                width=100,
                height=100,
            )

    def test_diagram_regions_are_marked_as_drawings(self) -> None:
        content = _parse_structure_result(
            {
                "parsing_res_list": [
                    {
                        "block_label": "chart",
                        "block_content": "",
                        "block_bbox": [1, 1, 99, 99],
                    }
                ]
            },
            width=100,
            height=100,
        )

        self.assertEqual(content.regions[0].type, "figure")
        self.assertEqual(content.regions[0].image_kind, "drawing")

    def test_paddle_image_regions_are_marked_as_photos(self) -> None:
        content = _parse_structure_result(
            {
                "parsing_res_list": [
                    {
                        "block_label": "image",
                        "block_content": "",
                        "block_bbox": [1, 1, 99, 99],
                    }
                ]
            },
            width=100,
            height=100,
        )

        self.assertEqual(content.regions[0].image_kind, "photo")

    def test_parser_error_keeps_the_unmodified_paddle_result(self) -> None:
        raw_result = {
            "res": {
                "parsing_res_list": [
                    {
                        "block_label": "image",
                        "block_content": "",
                        "block_bbox": [-1, 0, 20, 20],
                    }
                ]
            }
        }

        class FakeEngine:
            options: dict[str, object] = {}

            def predict(self, **_kwargs: object) -> list[object]:
                if "input" not in _kwargs:
                    raise AssertionError("PaddleOCR predict() needs the input path.")
                self.options = _kwargs
                return [SimpleNamespace(json=raw_result)]

        with TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "page.tif"
            Image.new("RGB", (100, 100), "white").save(image_path)
            adapter = object.__new__(PaddleOCRAdapter)
            adapter._engine = FakeEngine()

            with self.assertRaisesRegex(Step4Error, "out-of-page") as context:
                adapter.recognize(image_path)

        raw_response = context.exception.raw_response
        assert isinstance(raw_response, dict)
        self.assertEqual(raw_response["result"], raw_result)
        self.assertEqual(
            adapter._engine.options["text_det_limit_side_len"],
            TEXT_DET_LIMIT_SIDE_LEN,
        )
        self.assertEqual(adapter._engine.options["text_det_limit_type"], "max")


if __name__ == "__main__":
    unittest.main()
