import json
import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from ocr_pipeline.cli import main
from ocr_pipeline.steps.step1_check_batch import check_batch, write_reports


class CheckBatchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.batch = self.root / "batch"
        self.batch.mkdir()
        self.box_list = self.root / "boxes.csv"
        self.box_list.write_text("box_code\nB0012\n", encoding="utf-8")

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def add_pages(self, unit: str, pages: list[int]) -> None:
        for page in pages:
            (self.batch / f"B0012-{unit}-{page:05d}.tif").touch()

    def test_valid_duplex_folder_passes(self) -> None:
        self.add_pages("01", [1, 2, 3, 4])

        report = check_batch(self.batch, self.box_list, "folder")

        self.assertEqual(report.status, "pass")
        self.assertEqual(report.units[0].document_unit_id, "B0012-01")
        self.assertEqual(report.units[0].page_numbers, [1, 2, 3, 4])

    def test_batch_root_layout_checks_raw_and_ignores_root_metadata(self) -> None:
        raw_unit = self.batch / "raw" / "B0012" / "01"
        raw_unit.mkdir(parents=True)
        for page in [1, 2]:
            (raw_unit / f"B0012-01-{page:05d}.tif").touch()
        (self.batch / "box_list.csv").write_text(
            "box_code\nB0012\n",
            encoding="utf-8",
        )
        (self.batch / "config.json").write_text("{}", encoding="utf-8")

        report = check_batch(self.batch / "raw", self.batch / "box_list.csv", "folder")

        self.assertEqual(report.status, "pass")
        self.assertEqual(report.units[0].files, [
            "B0012/01/B0012-01-00001.tif",
            "B0012/01/B0012-01-00002.tif",
        ])

    def test_scan_path_must_match_box_and_unit_filename(self) -> None:
        wrong_unit = self.batch / "raw" / "B0012" / "02"
        wrong_unit.mkdir(parents=True)
        (wrong_unit / "B0012-01-00001.tif").touch()
        (wrong_unit / "B0012-01-00002.tif").touch()

        report = check_batch(self.batch / "raw", self.box_list, "folder")

        self.assertIn(
            "invalid_scan_path",
            {issue.code for issue in report.units[0].issues},
        )

    def test_gap_malformed_odd_count_and_unknown_box_are_reported(self) -> None:
        (self.batch / "B9999-02-00001.tif").touch()
        (self.batch / "B9999-02-00003.tif").touch()
        (self.batch / "B9999-02-00003-copy.tif").touch()

        report = check_batch(self.batch, self.box_list, "folder")

        unit = report.units[0]
        issue_codes = {issue.code for issue in unit.issues}
        self.assertEqual(report.status, "fail")
        self.assertIn("invalid_filename", issue_codes)
        self.assertIn("page_gap", issue_codes)
        self.assertIn("unknown_box_code", issue_codes)
        self.assertIn("odd_page_count", issue_codes)

    def test_duplicate_page_is_detected_across_subfolders(self) -> None:
        (self.batch / "scan-a").mkdir()
        (self.batch / "scan-b").mkdir()
        (self.batch / "scan-a" / "B0012-01-00001.tif").touch()
        (self.batch / "scan-b" / "B0012-01-00001.tif").touch()

        report = check_batch(self.batch, self.box_list, "book")

        self.assertIn(
            "duplicate_page",
            {issue.code for issue in report.units[0].issues},
        )

    def test_book_does_not_require_an_even_page_count(self) -> None:
        self.add_pages("03", [1, 2, 3])

        report = check_batch(self.batch, self.box_list, "book")

        self.assertEqual(report.status, "pass")

    def test_manifest_supports_mixed_folder_and_book_units(self) -> None:
        raw = self.batch / "raw"
        folder = raw / "B0012" / "01"
        book = raw / "B0012" / "02"
        folder.mkdir(parents=True)
        book.mkdir(parents=True)
        for page in [1, 2]:
            (folder / f"B0012-01-{page:05d}.tif").touch()
        for page in [1, 2, 3]:
            (book / f"B0012-02-{page:05d}.tif").touch()
        manifest = self.batch / "batch_manifest.yaml"
        manifest.write_text(
            """batch_id: test-batch
operator:
  name: Test Operator
scan:
  date: 2026-10-01
  scanner_model: test-scanner
  resolution_dpi: 300
  color_mode: grayscale_8bit
  format: TIFF
  duplex: true
items:
  - box_code: B0012
    item_number: "01"
    item_type: folder
    total_pages: 2
    damaged_pages: []
  - box_code: B0012
    item_number: "02"
    item_type: book
    total_pages: 3
    printed_pages: 2
    damaged_pages: [3]
""",
            encoding="utf-8",
        )

        report = check_batch(raw, self.box_list, None, manifest)

        self.assertEqual(report.status, "pass")
        self.assertEqual(report.kind, "mixed")
        self.assertEqual(
            {unit.document_unit_id: unit.kind for unit in report.units},
            {"B0012-01": "folder", "B0012-02": "book"},
        )

    def test_reports_are_written_as_batch_and_per_unit_json(self) -> None:
        self.add_pages("01", [1, 2])
        report = check_batch(self.batch, self.box_list, "folder")
        output = self.root / "reports"

        write_reports(report, output)

        batch_json = json.loads(
            (output / "batch-report.json").read_text(encoding="utf-8")
        )
        unit_json = json.loads(
            (output / "B0012-01.json").read_text(encoding="utf-8")
        )
        self.assertEqual(batch_json["status"], "pass")
        self.assertEqual(unit_json["document_unit_id"], "B0012-01")

    def test_empty_batch_fails(self) -> None:
        report = check_batch(self.batch, self.box_list, "folder")

        self.assertEqual(report.status, "fail")
        self.assertIn(
            "no_tiff_files",
            {issue.code for issue in report.issues},
        )

    def test_zero_page_number_is_reported(self) -> None:
        self.add_pages("01", [0, 1])

        report = check_batch(self.batch, self.box_list, "book")

        self.assertIn(
            "invalid_page_number",
            {issue.code for issue in report.units[0].issues},
        )

    def test_non_tif_scan_filename_is_rejected(self) -> None:
        (self.batch / "B0012-01-00001.jpg").touch()

        report = check_batch(self.batch, self.box_list, "book")

        self.assertIn(
            "invalid_filename",
            {issue.code for issue in report.units[0].issues},
        )

    def test_empty_batch_cli_exits_with_failure(self) -> None:
        output = self.root / "reports"
        stdout = io.StringIO()
        stderr = io.StringIO()

        with redirect_stdout(stdout), redirect_stderr(stderr):
            exit_code = main(
                [
                    "check-batch",
                    "--batch",
                    str(self.batch),
                    "--box-list",
                    str(self.box_list),
                    "--kind",
                    "folder",
                    "--output",
                    str(output),
                ]
            )

        self.assertEqual(exit_code, 1)
        self.assertTrue((output / "batch-report.json").exists())


if __name__ == "__main__":
    unittest.main()
