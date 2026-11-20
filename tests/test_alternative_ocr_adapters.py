import json
import io
import base64
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
from ocr_pipeline.adapters.gemini_ocr import GeminiOCRAdapter
from ocr_pipeline.adapters.openai_ocr import OpenAIOCRAdapter
from ocr_pipeline.adapters.claude_ocr import ClaudeOCRAdapter
from ocr_pipeline.adapters.vietocr_ocr import (
    VietOCRAdapter,
    _line_boxes,
    _merge_symbol_tokens,
)
from ocr_pipeline.steps.step4_ocr_pages import (
    Step4Error,
    load_ocr_adapter,
)


class AlternativeOCRAdapterTests(unittest.TestCase):
    def test_claude_transcribes_page_and_reports_token_usage(self) -> None:
        raw = {
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": "乾。元亨利貞。"}],
            "usage": {
                "input_tokens": 400, "cache_creation_input_tokens": 30,
                "cache_read_input_tokens": 10, "output_tokens": 25,
                "output_tokens_details": {"thinking_tokens": 5},
            },
        }
        calls = []

        def create(**kwargs: object) -> SimpleNamespace:
            calls.append(kwargs)
            return SimpleNamespace(model_dump=lambda **unused: raw)

        adapter = ClaudeOCRAdapter.__new__(ClaudeOCRAdapter)
        adapter.language = "vi+zh-Hant"
        adapter.model_version = "test-model"
        adapter._api_error = RuntimeError
        adapter._client = SimpleNamespace(messages=SimpleNamespace(create=create))
        with TemporaryDirectory() as temp_dir:
            page = Path(temp_dir) / "page.tif"
            Image.new("RGB", (10, 10), "white").save(page)
            original = page.read_bytes()
            content, response = adapter.recognize(page)
            self.assertEqual(page.read_bytes(), original)
        self.assertEqual(content.markdown, "乾。元亨利貞。")
        self.assertEqual(content.regions, [])
        self.assertEqual(calls[0]["max_tokens"], 8192)
        image = calls[0]["messages"][0]["content"][0]
        self.assertEqual(image["type"], "image")
        self.assertEqual(image["source"]["media_type"], "image/png")
        self.assertEqual(
            Image.open(io.BytesIO(base64.b64decode(image["source"]["data"]))).size,
            (10, 10),
        )
        self.assertEqual(adapter.token_usage(response), {
            "prompt_tokens": 440, "output_tokens": 25, "thoughts_tokens": 5,
            "cached_tokens": 10, "total_tokens": 465,
        })

    def test_claude_rejects_incomplete_or_refused_output_with_usage(self) -> None:
        adapter = ClaudeOCRAdapter.__new__(ClaudeOCRAdapter)
        adapter.language = "vi"
        adapter.model_version = "test-model"
        adapter._api_error = RuntimeError
        with TemporaryDirectory() as temp_dir:
            page = Path(temp_dir) / "page.tif"
            Image.new("RGB", (10, 10)).save(page)
            for reason in ("max_tokens", "refusal"):
                raw = {
                    "stop_reason": reason, "content": [{"type": "text", "text": "partial"}],
                    "usage": {"input_tokens": 100, "output_tokens": 20},
                }
                adapter._client = SimpleNamespace(messages=SimpleNamespace(
                    create=lambda **kwargs: SimpleNamespace(model_dump=lambda **kw: raw)
                ))
                with self.assertRaises(Step4Error) as caught:
                    adapter.recognize(page)
                usage = adapter.token_usage(caught.exception.raw_response)
                assert usage is not None
                self.assertEqual(usage["output_tokens"], 20)
                self.assertIsNone(usage["total_tokens"])
                self.assertIsNone(usage["cached_tokens"])

    def test_claude_needs_environment_key(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(Step4Error, "ANTHROPIC_API_KEY"):
                ClaudeOCRAdapter()

    def test_openai_transcribes_page_and_reports_token_usage(self) -> None:
        raw = {
            "status": "completed",
            "output": [{
                "type": "message",
                "content": [{"type": "output_text", "text": "乾。元亨利貞。"}],
            }],
            "usage": {
                "input_tokens": 410, "output_tokens": 25, "total_tokens": 435,
                "input_tokens_details": {"cached_tokens": 10},
                "output_tokens_details": {"reasoning_tokens": 5},
            },
        }
        calls = []

        def create(**kwargs: object) -> SimpleNamespace:
            calls.append(kwargs)
            return SimpleNamespace(model_dump=lambda **unused: raw)

        adapter = OpenAIOCRAdapter.__new__(OpenAIOCRAdapter)
        adapter.language = "vi+zh-Hant"
        adapter.model_version = "test-model"
        adapter._api_error = RuntimeError
        adapter._client = SimpleNamespace(responses=SimpleNamespace(create=create))
        with TemporaryDirectory() as temp_dir:
            page = Path(temp_dir) / "page.tif"
            Image.new("RGB", (10, 10), "white").save(page)
            original = page.read_bytes()
            content, response = adapter.recognize(page)
            self.assertEqual(page.read_bytes(), original)
        self.assertEqual(content.markdown, "乾。元亨利貞。")
        self.assertEqual(content.regions, [])
        self.assertFalse(calls[0]["store"])
        image = calls[0]["input"][0]["content"][1]
        self.assertEqual(image["type"], "input_image")
        self.assertEqual(image["detail"], "high")
        prefix = "data:image/png;base64,"
        self.assertTrue(image["image_url"].startswith(prefix))
        decoded = base64.b64decode(image["image_url"][len(prefix):])
        self.assertEqual(Image.open(io.BytesIO(decoded)).size, (10, 10))
        self.assertEqual(adapter.token_usage(response), {
            "prompt_tokens": 410, "output_tokens": 25, "thoughts_tokens": 5,
            "cached_tokens": 10, "total_tokens": 435,
        })

    def test_openai_rejects_incomplete_or_refused_output_with_usage(self) -> None:
        adapter = OpenAIOCRAdapter.__new__(OpenAIOCRAdapter)
        adapter.language = "vi"
        adapter.model_version = "test-model"
        adapter._api_error = RuntimeError
        responses = [
            {"status": "incomplete", "usage": {"total_tokens": 99}},
            {
                "status": "completed",
                "output": [{"type": "message", "content": [
                    {"type": "refusal", "refusal": "Cannot comply"}
                ]}],
                "usage": {"total_tokens": 42},
            },
        ]
        with TemporaryDirectory() as temp_dir:
            page = Path(temp_dir) / "page.tif"
            Image.new("RGB", (10, 10)).save(page)
            for raw in responses:
                adapter._client = SimpleNamespace(responses=SimpleNamespace(
                    create=lambda **kwargs: SimpleNamespace(model_dump=lambda **kw: raw)
                ))
                with self.assertRaises(Step4Error) as caught:
                    adapter.recognize(page)
                usage = adapter.token_usage(caught.exception.raw_response)
                assert usage is not None
                self.assertEqual(usage["total_tokens"], raw["usage"]["total_tokens"])
                self.assertIsNone(usage["prompt_tokens"])

    def test_openai_needs_environment_key(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(Step4Error, "OPENAI_API_KEY"):
                OpenAIOCRAdapter()

    def test_gemini_transcribes_png_and_reports_provider_usage(self) -> None:
        raw = {
            "candidates": [{
                "finish_reason": "STOP",
                "content": {"parts": [{"text": "乾。元亨利貞。"}]},
            }],
            "usage_metadata": {
                "prompt_token_count": 410,
                "candidates_token_count": 20,
                "thoughts_token_count": 5,
                "total_token_count": 435,
            },
        }
        calls = []

        def generate_content(**kwargs: object) -> SimpleNamespace:
            calls.append(kwargs)
            return SimpleNamespace(model_dump=lambda **unused: raw)

        adapter = GeminiOCRAdapter.__new__(GeminiOCRAdapter)
        adapter.language = "vi+zh-Hant"
        adapter.model_version = "gemini-test"
        adapter.thinking_level = "low"
        adapter._request_errors = (RuntimeError,)
        adapter._client = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
        types_module = ModuleType("fake_genai_types")
        setattr(types_module, "Part", SimpleNamespace(from_bytes=lambda **kwargs: kwargs))
        setattr(types_module, "GenerateContentConfig", lambda **kwargs: kwargs)
        setattr(types_module, "ThinkingConfig", lambda **kwargs: kwargs)
        adapter._types = types_module
        with TemporaryDirectory() as temp_dir:
            page = Path(temp_dir) / "page.tif"
            Image.new("RGB", (10, 10), "white").save(page)
            original = page.read_bytes()
            content, response = adapter.recognize(page)
            self.assertEqual(page.read_bytes(), original)
        self.assertEqual(content.markdown, "乾。元亨利貞。")
        self.assertEqual(content.regions, [])
        image_part = calls[0]["contents"][1]
        self.assertEqual(image_part["mime_type"], "image/png")
        self.assertEqual(Image.open(io.BytesIO(image_part["data"])).size, (10, 10))
        self.assertEqual(calls[0]["config"]["thinking_config"], {"thinking_level": "low"})
        self.assertEqual(adapter.token_usage(response), {
            "prompt_tokens": 410, "output_tokens": 20, "thoughts_tokens": 5,
            "cached_tokens": None, "total_tokens": 435,
        })

    def test_gemini_rejects_truncated_response_and_retains_usage(self) -> None:
        raw = {
            "candidates": [{"finish_reason": "MAX_TOKENS"}],
            "usage_metadata": {"total_token_count": 123},
        }
        adapter = GeminiOCRAdapter.__new__(GeminiOCRAdapter)
        adapter.language = "vi"
        adapter.model_version = "gemini-test"
        adapter.thinking_level = "low"
        adapter._request_errors = (RuntimeError,)
        adapter._client = SimpleNamespace(models=SimpleNamespace(
            generate_content=lambda **kwargs: SimpleNamespace(model_dump=lambda **kw: raw)
        ))
        types_module = ModuleType("fake_genai_types")
        setattr(types_module, "Part", SimpleNamespace(from_bytes=lambda **kwargs: kwargs))
        setattr(types_module, "GenerateContentConfig", lambda **kwargs: kwargs)
        setattr(types_module, "ThinkingConfig", lambda **kwargs: kwargs)
        adapter._types = types_module
        with TemporaryDirectory() as temp_dir:
            page = Path(temp_dir) / "page.tif"
            Image.new("RGB", (10, 10)).save(page)
            with self.assertRaises(Step4Error) as caught:
                adapter.recognize(page)
        usage = adapter.token_usage(caught.exception.raw_response)
        assert usage is not None
        self.assertEqual(usage["total_tokens"], 123)

    def test_gemini_needs_environment_key(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(Step4Error, "GEMINI_API_KEY"):
                GeminiOCRAdapter()

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

    def test_vietocr_confidence_matches_the_text_source(self) -> None:
        adapter = VietOCRAdapter.__new__(VietOCRAdapter)
        adapter.language = "vi"
        adapter.variant = "vgg_transformer"
        adapter.detection_language = "vie"
        adapter._tesseract_version = "test"
        data = {
            "text": ["Xin"], "left": [10], "top": [10], "width": [20],
            "height": [10], "block_num": [1], "par_num": [1], "line_num": [1],
        }
        setattr(adapter, "_pytesseract", SimpleNamespace(
            image_to_data=lambda *args, **kwargs: data,
            Output=SimpleNamespace(DICT="dict"),
        ))
        with patch.object(
            VietOCRAdapter, "_load_page", return_value=(Image.new("L", (100, 100)), {})
        ):
            for prediction, probability, expected_source in (
                ("Xin", None, "vietocr"),
                ("!!!", 0.99, "tesseract-fallback"),
            ):
                with self.subTest(source=expected_source):
                    setattr(adapter, "_predictor", SimpleNamespace(
                        predict=lambda *args, **kwargs: (prediction, probability),
                    ))
                    content, raw = adapter.recognize(Path("unused.tif"))
                    self.assertEqual(content.markdown, "Xin")
                    assert isinstance(raw, dict)
                    self.assertEqual(raw["lines"][0]["source"], expected_source)
                    self.assertIsNone(raw["lines"][0]["confidence"])

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
