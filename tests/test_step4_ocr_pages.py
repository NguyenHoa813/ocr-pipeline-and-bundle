import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from ocr_pipeline.cli import main
from ocr_pipeline.steps.run_tracking import _checksum
from ocr_pipeline.steps.step1_check_batch import check_batch, write_reports
from ocr_pipeline.steps.step2_split_documents import (
    DocumentRecord,
    SplitReport,
    UnitSplitReport,
    write_split_report,
)
from ocr_pipeline.steps.step3_flag_blank_pages import (
    flag_document_blank_pages,
    write_blank_page_report,
)
from ocr_pipeline.steps.step4_ocr_pages import (
    LayoutRegion,
    PageOCRContent,
    ReportedOCRCost,
    Step4Error,
    run_document_ocr,
    write_document_ocr,
)


class FakeOCRAdapter:
    model_name = "test-ocr"
    model_version = "test-1"

    def __init__(self) -> None:
        self.calls: list[Path] = []

    def recognize(self, image_path: Path) -> tuple[PageOCRContent, object]:
        self.calls.append(image_path)
        if image_path.name.endswith("00003.tif"):
            raw_response = {"choices": [{"message": {"content": "not-json"}}]}
            raise Step4Error("Invalid OCR output", raw_response)
        return (
            PageOCRContent(
                markdown="Recognized text",
                regions=[
                    LayoutRegion(
                        type="text",
                        text="Recognized text",
                        bbox=[0.1, 0.2, 0.9, 0.8],
                    )
                ],
            ),
            {"choices": [{"message": {"content": "raw response"}}]},
        )


class Step4Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.batch = Path(self.temporary_directory.name) / "batch01"
        self.raw_unit = self.batch / "raw" / "B0012" / "01"
        self.raw_unit.mkdir(parents=True)
        self.box_list = self.batch / "box_list.csv"
        self.box_list.write_text("box_code\nB0012\n", encoding="utf-8")
        self.step1_folder = self.batch / "work" / "step-1-checks"
        self.step2_folder = self.batch / "work" / "step-2-documents"
        self.step3_folder = self.batch / "work" / "step-3-blank-pages"
        self.step1_folder.mkdir(parents=True)
        self.step2_folder.mkdir(parents=True)
        self.step3_folder.mkdir(parents=True)
        for page, color in [(1, "white"), (2, "black"), (3, "black")]:
            Image.new("RGB", (20, 20), color).save(
                self.raw_unit / f"B0012-01-{page:05d}.tif"
            )

        check_report = check_batch(
            self.batch / "raw",
            self.box_list,
            "book",
        )
        write_reports(check_report, self.step1_folder)
        split_report = SplitReport(
            schema_version=1,
            batch_folder=str(self.batch / "raw"),
            source_check_report=str(self.step1_folder / "batch-report.json"),
            status="pass",
            units=[
                UnitSplitReport(
                    unit_id="B0012-01",
                    status="pass",
                    documents=[
                        DocumentRecord(
                            id="B0012-01-01",
                            first_page=1,
                            last_page=3,
                            pages=[1, 2, 3],
                        )
                    ],
                    separator_pages=[],
                )
            ],
        )
        write_split_report(split_report, self.step2_folder)
        blank_report = flag_document_blank_pages(
            self.batch,
            "B0012-01-01",
            0.01,
        )
        write_blank_page_report(
            blank_report,
            self.step3_folder / "B0012-01-01.json",
        )
        self.output_folder = self.batch / "work" / "step-4-ocr" / "B0012-01-01"

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_only_nonblank_pages_are_sent_to_model(self) -> None:
        adapter = FakeOCRAdapter()

        report, normalized, raw = run_document_ocr(
            self.batch,
            "B0012-01-01",
            adapter,
        )

        self.assertEqual(
            [path.name for path in adapter.calls],
            ["B0012-01-00002.tif", "B0012-01-00003.tif"],
        )
        self.assertEqual(
            [page.status for page in report.pages],
            ["blank", "success", "failed"],
        )
        self.assertEqual(normalized[1].markdown, "")
        self.assertIn(2, raw)
        self.assertIn(3, raw)

    def test_invalid_model_output_fails_page_and_keeps_raw_response(self) -> None:
        adapter = FakeOCRAdapter()
        report, normalized, raw = run_document_ocr(
            self.batch,
            "B0012-01-01",
            adapter,
        )
        write_document_ocr(report, normalized, raw, self.output_folder)

        written_report = json.loads(
            (self.output_folder / "ocr-report.json").read_text(encoding="utf-8")
        )
        raw_response = json.loads(
            (self.output_folder / "raw" / "page-00003.json").read_text(
                encoding="utf-8"
            )
        )
        page_json = json.loads(
            (self.output_folder / "pages" / "page-00003.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(report.status, "fail")
        self.assertEqual(written_report["pages"][2]["status"], "failed")
        self.assertEqual(raw_response["choices"][0]["message"]["content"], "not-json")
        self.assertEqual(page_json["error"], "Invalid OCR output")

    def test_raw_and_normalized_outputs_are_written_separately(self) -> None:
        adapter = FakeOCRAdapter()
        report, normalized, raw = run_document_ocr(
            self.batch,
            "B0012-01-01",
            adapter,
        )
        write_document_ocr(report, normalized, raw, self.output_folder)

        self.assertEqual(
            (self.output_folder / "pages" / "page-00002.md").read_text(
                encoding="utf-8"
            ),
            "Recognized text",
        )
        self.assertTrue(
            (self.output_folder / "raw" / "page-00002.json").is_file()
        )
        self.assertEqual(
            (self.output_folder / "pages" / "page-00001.md").read_text(
                encoding="utf-8"
            ),
            "",
        )
        self.assertFalse(
            (self.output_folder / "raw" / "page-00001.json").exists()
        )

    def test_token_usage_includes_failed_responses_and_unknown_pages(self) -> None:
        class MeteredAdapter(FakeOCRAdapter):
            def token_usage(self, raw: object) -> dict[str, int | None] | None:
                if not isinstance(raw, dict) or "usage" not in raw:
                    return None
                return {
                    "prompt_tokens": 10, "output_tokens": 3,
                    "thoughts_tokens": None, "cached_tokens": None,
                    "total_tokens": 13,
                }

            def recognize(self, image_path: Path) -> tuple[PageOCRContent, object]:
                if image_path.name.endswith("00003.tif"):
                    raise Step4Error("Incomplete", {"usage": True})
                return PageOCRContent("Text", []), {"usage": True}

        report, _, _ = run_document_ocr(self.batch, "B0012-01-01", MeteredAdapter())
        assert report.token_usage is not None
        assert report.pages[2].token_usage is not None
        self.assertEqual(report.token_usage["total_tokens"], 26)
        self.assertEqual(report.token_usage["reported_pages"], 2)
        self.assertIsNone(report.token_usage["thoughts_tokens"])
        self.assertEqual(report.pages[2].token_usage["total_tokens"], 13)

        class MissingUsageAdapter(MeteredAdapter):
            def token_usage(self, raw: object) -> dict[str, int | None] | None:
                return None

        report, _, _ = run_document_ocr(self.batch, "B0012-01-01", MissingUsageAdapter())
        assert report.token_usage is not None
        self.assertEqual(report.token_usage["unreported_pages"], 2)
        self.assertIsNone(report.token_usage["total_tokens"])

    def test_gemini_cli_keeps_local_ocr_output_separate(self) -> None:
        self.output_folder.mkdir(parents=True)
        local_file = self.output_folder / "ocr-report.json"
        local_file.write_text("local result", encoding="utf-8")
        with patch("ocr_pipeline.cli.load_ocr_adapter", return_value=FakeOCRAdapter()) as load:
            result = main([
                "ocr-document", "--batch", str(self.batch),
                "--document-id", "B0012-01-01", "--model", "gemini",
                "--gemini-model", "test-model", "--run-id", "test-gemini",
            ])
        self.assertEqual(result, 1)
        load.assert_called_once_with(
            "gemini", language="vi", model="test-model", thinking_level="low"
        )
        self.assertEqual(local_file.read_text(encoding="utf-8"), "local result")
        self.assertTrue(
            (self.batch / "work" / "step-4-ocr-gemini" / "B0012-01-01"
             / "ocr-report.json").is_file()
        )
        log_path = (
            self.batch / "work" / "step-4-ocr-gemini" / "B0012-01-01"
            / "ocr-test-gemini.log"
        )
        log = log_path.read_text(encoding="utf-8")
        self.assertIn("OCR page 00002: success", log)
        self.assertIn("OCR page 00003: failed", log)
        self.assertIn("OCR fail for B0012-01-01", log)
        self.assertIn("Output:", log)

    def test_gemini_cli_logs_token_usage_and_preserves_run_history(self) -> None:
        class MeteredAdapter(FakeOCRAdapter):
            def recognize(self, image_path: Path) -> tuple[PageOCRContent, object]:
                return PageOCRContent("Text", []), {"usage": True}

            def token_usage(self, raw: object) -> dict[str, int | None]:
                return {
                    "prompt_tokens": 10, "output_tokens": 3,
                    "thoughts_tokens": None, "cached_tokens": None,
                    "total_tokens": 13,
                }

        args = [
            "ocr-document", "--batch", str(self.batch),
            "--document-id", "B0012-01-01", "--model", "gemini",
            "--run-id", "metered-test",
        ]
        with patch("ocr_pipeline.cli.load_ocr_adapter", return_value=MeteredAdapter()):
            self.assertEqual(main(args), 0)
            self.assertEqual(main(args), 0)
        output = self.batch / "work" / "step-4-ocr-gemini" / "B0012-01-01"
        log = (output / "ocr-metered-test.log").read_text(encoding="utf-8")
        self.assertEqual(log.count("Starting Gemini OCR run"), 2)
        self.assertEqual(log.count("OCR pass for B0012-01-01"), 2)
        self.assertIn("'prompt_tokens': 20", log)
        self.assertIn("'total_tokens': 26", log)
        self.assertIn("'thoughts_tokens': None", log)
        run = json.loads(
            (self.batch / "work" / "runs" / "metered-test.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            run["steps"]["step-4"]["output_sha256"][str(output.resolve())],
            _checksum(output),
        )

    def test_gemini_cli_logs_initialization_errors_and_releases_handler(self) -> None:
        args = [
            "ocr-document", "--batch", str(self.batch),
            "--document-id", "B0012-01-01", "--model", "gemini",
            "--run-id", "init-error",
        ]
        with patch("ocr_pipeline.cli.load_ocr_adapter", side_effect=Step4Error("Missing key")):
            self.assertEqual(main(args), 2)
        output = self.batch / "work" / "step-4-ocr-gemini" / "B0012-01-01"
        log = output / "ocr-init-error.log"
        before = log.read_text(encoding="utf-8")
        self.assertIn("ERROR: Missing key", before)
        with patch("ocr_pipeline.cli.load_ocr_adapter", return_value=FakeOCRAdapter()):
            self.assertEqual(main(args[:-1] + ["next-run"]), 1)
        self.assertEqual(log.read_text(encoding="utf-8"), before)

    def test_openai_cli_separates_outputs_and_logs_usage(self) -> None:
        class MeteredAdapter(FakeOCRAdapter):
            def recognize(self, image_path: Path) -> tuple[PageOCRContent, object]:
                return PageOCRContent("Text", []), {"usage": True}

            def token_usage(self, raw: object) -> dict[str, int | None]:
                return {
                    "prompt_tokens": 10, "output_tokens": 3,
                    "thoughts_tokens": 0, "cached_tokens": 0, "total_tokens": 13,
                }

        gemini = self.batch / "work" / "step-4-ocr-gemini" / "B0012-01-01"
        gemini.mkdir(parents=True)
        original = gemini / "ocr-report.json"
        original.write_text("Gemini result", encoding="utf-8")
        with patch("ocr_pipeline.cli.load_ocr_adapter", return_value=MeteredAdapter()) as load:
            result = main([
                "ocr-document", "--batch", str(self.batch),
                "--document-id", "B0012-01-01", "--model", "openai",
                "--openai-model", "test-model", "--run-id", "openai-test",
            ])
        self.assertEqual(result, 0)
        load.assert_called_once_with("openai", language="vi", model="test-model")
        self.assertEqual(original.read_text(encoding="utf-8"), "Gemini result")
        output = self.batch / "work" / "step-4-ocr-openai" / "B0012-01-01"
        report = json.loads((output / "ocr-report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["token_usage"]["total_tokens"], 26)
        log = (output / "ocr-openai-test.log").read_text(encoding="utf-8")
        self.assertIn("Starting OpenAI OCR run", log)
        self.assertIn("OCR page 00002: success", log)
        self.assertIn("'total_tokens': 26", log)

    def test_claude_cli_separates_outputs_and_logs_usage(self) -> None:
        class MeteredAdapter(FakeOCRAdapter):
            def recognize(self, image_path: Path) -> tuple[PageOCRContent, object]:
                return PageOCRContent("Text", []), {"usage": True}

            def token_usage(self, raw: object) -> dict[str, int | None]:
                return {
                    "prompt_tokens": 10, "output_tokens": 3,
                    "thoughts_tokens": None, "cached_tokens": 0, "total_tokens": 13,
                }

        gemini = self.batch / "work" / "step-4-ocr-gemini" / "B0012-01-01"
        gemini.mkdir(parents=True)
        original = gemini / "ocr-report.json"
        original.write_text("Gemini result", encoding="utf-8")
        with patch("ocr_pipeline.cli.load_ocr_adapter", return_value=MeteredAdapter()) as load:
            result = main([
                "ocr-document", "--batch", str(self.batch),
                "--document-id", "B0012-01-01", "--model", "claude",
                "--claude-model", "test-model", "--run-id", "claude-test",
            ])
        self.assertEqual(result, 0)
        load.assert_called_once_with("claude", language="vi", model="test-model")
        self.assertEqual(original.read_text(encoding="utf-8"), "Gemini result")
        output = self.batch / "work" / "step-4-ocr-claude" / "B0012-01-01"
        report = json.loads((output / "ocr-report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["token_usage"]["total_tokens"], 26)
        log = (output / "ocr-claude-test.log").read_text(encoding="utf-8")
        self.assertIn("Starting Claude OCR run", log)
        self.assertIn("OCR page 00002: success", log)
        self.assertIn("'total_tokens': 26", log)

    def test_local_and_plugin_runs_write_logs_with_unknown_cost(self) -> None:
        for model in ("tesseract", "third-party-plugin"):
            with self.subTest(model=model):
                args = [
                    "ocr-document", "--batch", str(self.batch),
                    "--document-id", "B0012-01-01", "--model", model,
                    "--run-id", model,
                ]
                with patch("ocr_pipeline.cli.load_ocr_adapter", return_value=FakeOCRAdapter()):
                    self.assertEqual(main(args), 1)
                log = (self.output_folder / f"ocr-{model}.log").read_text(encoding="utf-8")
                self.assertIn("OCR page 00002: success", log)
                self.assertIn("Elapsed time:", log)
                self.assertIn("Token usage: not reported", log)
                self.assertIn("'status': 'not_reported'", log)
                self.assertIn("'amount': None", log)

    def test_provider_costs_are_summed_only_when_complete_and_same_currency(self) -> None:
        class CostedAdapter(FakeOCRAdapter):
            def __init__(self, second_cost: ReportedOCRCost | None) -> None:
                super().__init__()
                self.second_cost = second_cost

            def recognize(self, image_path: Path) -> tuple[PageOCRContent, object]:
                if image_path.name.endswith("00003.tif"):
                    raise Step4Error("Incomplete", {"cost": self.second_cost})
                return PageOCRContent("Text", []), {"cost": ReportedOCRCost(0.1, "USD")}

            def cost_usage(self, raw: object) -> ReportedOCRCost | None:
                assert isinstance(raw, dict)
                return raw["cost"]

        report, _, _ = run_document_ocr(
            self.batch, "B0012-01-01", CostedAdapter(ReportedOCRCost(0.2, "USD"))
        )
        assert report.cost is not None
        self.assertEqual(report.cost["status"], "reported")
        self.assertEqual(report.cost["amount"], 0.3)
        self.assertEqual(report.cost["currency"], "USD")

        for second_cost, expected in (
            (None, "partial"),
            (ReportedOCRCost(0.2, "EUR"), "mixed_currencies"),
        ):
            with self.subTest(status=expected):
                report, _, _ = run_document_ocr(
                    self.batch, "B0012-01-01", CostedAdapter(second_cost)
                )
                assert report.cost is not None
                self.assertEqual(report.cost["status"], expected)
                self.assertIsNone(report.cost["amount"])

    def test_invalid_provider_cost_fails_without_fabricating_a_total(self) -> None:
        class InvalidCostAdapter(FakeOCRAdapter):
            def cost_usage(self, raw: object) -> ReportedOCRCost:
                return ReportedOCRCost(float("nan"), "USD")

        report, _, _ = run_document_ocr(self.batch, "B0012-01-01", InvalidCostAdapter())
        self.assertEqual(report.status, "fail")
        self.assertIn("invalid reported cost", report.pages[1].error)
        assert report.cost is not None
        self.assertIsNone(report.cost["amount"])


if __name__ == "__main__":
    unittest.main()
