import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

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


if __name__ == "__main__":
    unittest.main()
