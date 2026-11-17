import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from ocr_pipeline.cli import main
from ocr_pipeline.steps.step1_check_batch import check_batch, write_reports
from ocr_pipeline.steps.step2_split_documents import (
    DocumentRecord,
    SplitReport,
    UnitSplitReport,
    write_split_report,
)
from ocr_pipeline.steps.step3_flag_blank_pages import (
    Step3Error,
    flag_document_blank_pages,
    write_blank_page_report,
)


class Step3Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.batch = Path(self.temporary_directory.name) / "batch01"
        self.raw_unit = self.batch / "raw" / "B0012" / "01"
        self.raw_unit.mkdir(parents=True)
        self.box_list = self.batch / "box_list.csv"
        self.box_list.write_text("box_code\nB0012\n", encoding="utf-8")
        self.step1_folder = self.batch / "work" / "step-1-checks"
        self.step2_folder = self.batch / "work" / "step-2-documents"
        self.step1_folder.mkdir(parents=True)
        self.step2_folder.mkdir(parents=True)
        self.output_folder = self.batch / "work" / "step-3-blank-pages"
        self._create_page(1, "white")
        self._create_page(2, "black")

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
                            last_page=2,
                            pages=[1, 2],
                        )
                    ],
                    separator_pages=[],
                )
            ],
        )
        write_split_report(split_report, self.step2_folder)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _create_page(self, page: int, color: str) -> Path:
        path = self.raw_unit / f"B0012-01-{page:05d}.tif"
        Image.new("RGB", (20, 20), color).save(path)
        return path

    def test_white_page_flagged_blank_and_black_page_not_blank(self) -> None:
        report = flag_document_blank_pages(
            self.batch,
            "B0012-01-01",
            0.01,
        )

        self.assertEqual(report.status, "pass")
        self.assertEqual(
            [(page.page_number, page.is_blank, page.dark_pixel_share) for page in report.pages],
            [(1, True, 0.0), (2, False, 1.0)],
        )

    def test_threshold_comparison_is_strictly_less_than(self) -> None:
        report = flag_document_blank_pages(
            self.batch,
            "B0012-01-01",
            1.0,
        )

        self.assertFalse(report.pages[1].is_blank)

    def test_custom_split_and_check_report_paths_are_supported(self) -> None:
        split_report = self.batch / "custom-split.json"
        check_report = self.batch / "custom-check.json"
        split_report.write_bytes(
            (self.step2_folder / "split-report.json").read_bytes()
        )
        check_report.write_bytes(
            (self.step1_folder / "batch-report.json").read_bytes()
        )

        report = flag_document_blank_pages(
            self.batch,
            "B0012-01-01",
            0.01,
            split_report,
            check_report,
        )

        self.assertEqual(len(report.pages), 2)

    def test_missing_document_is_rejected(self) -> None:
        with self.assertRaisesRegex(Step3Error, "was not found"):
            flag_document_blank_pages(
                self.batch,
                "B0012-01-99",
                0.01,
            )

    def test_invalid_threshold_is_rejected(self) -> None:
        with self.assertRaisesRegex(Step3Error, "between 0 and 1"):
            flag_document_blank_pages(
                self.batch,
                "B0012-01-01",
                1.1,
            )

    def test_report_contains_one_flag_per_document_page(self) -> None:
        report = flag_document_blank_pages(
            self.batch,
            "B0012-01-01",
            0.01,
        )
        output_path = self.output_folder / "B0012-01-01.json"

        write_blank_page_report(report, output_path)

        saved = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertEqual(
            [page["page_number"] for page in saved["pages"]],
            [1, 2],
        )
        self.assertEqual(saved["dark_pixel_intensity_cutoff"], 128)

    def test_cli_writes_default_output(self) -> None:
        exit_code = main(
            [
                "flag-blank-pages",
                "--batch",
                str(self.batch),
                "--document-id",
                "B0012-01-01",
                "--dark-pixel-share-threshold",
                "0.01",
            ]
        )

        self.assertEqual(exit_code, 0)
        self.assertTrue(
            (self.output_folder / "B0012-01-01.json").is_file()
        )


if __name__ == "__main__":
    unittest.main()
