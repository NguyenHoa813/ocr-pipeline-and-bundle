import json
import tempfile
import unittest
from pathlib import Path

import zxingcpp
from PIL import Image

from ocr_pipeline.steps.step1_check_batch import UnitKind, check_batch, write_reports
from ocr_pipeline.steps.step2_split_documents import (
    ZXingBarcodeDetector,
    split_checked_batch,
    write_split_report,
)


class FilenameBarcodeDetector:
    def __init__(self, separator_pages: set[int]) -> None:
        self.separator_pages = separator_pages

    def contains_barcode(self, image_path: Path) -> bool:
        page_number = int(image_path.stem.rsplit("-", 1)[1])
        return page_number in self.separator_pages


class CheckBatchBarcodeDetector:
    def contains_barcode(self, image_path: Path) -> bool:
        raise AssertionError(
            f"Book page was unexpectedly sent to barcode detector: {image_path}"
        )


class Step2Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.batch = self.root / "batch"
        self.batch.mkdir()
        self.box_list = self.root / "boxes.csv"
        self.box_list.write_text("box_code\nB0012\n", encoding="utf-8")
        self.step1_output = self.root / "step-1"
        self.step1_output.mkdir()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def add_pages(self, unit: str, pages: list[int]) -> None:
        for page in pages:
            (self.batch / f"B0012-{unit}-{page:05d}.tif").touch()

    def create_check_report(self, kind: UnitKind) -> Path:
        report = check_batch(self.batch, self.box_list, kind)
        write_reports(report, self.step1_output)
        return self.step1_output / "batch-report.json"

    def test_folder_splits_on_separator_and_excludes_separator_pages(self) -> None:
        self.add_pages("01", list(range(1, 11)))
        check_report_path = self.create_check_report("folder")

        report = split_checked_batch(
            self.batch,
            check_report_path,
            FilenameBarcodeDetector({3, 8}),
        )

        self.assertEqual(report.status, "pass")
        unit = report.units[0]
        self.assertEqual(unit.separator_pages, [3, 4, 7, 8])
        self.assertEqual(
            [(doc.id, doc.first_page, doc.last_page, doc.pages) for doc in unit.documents],
            [
                ("B0012-01-01", 1, 2, [1, 2]),
                ("B0012-01-02", 5, 6, [5, 6]),
                ("B0012-01-03", 9, 10, [9, 10]),
            ],
        )

    def test_leading_and_trailing_separators_are_reported_as_anomalies(self) -> None:
        self.add_pages("01", list(range(1, 9)))
        check_report_path = self.create_check_report("folder")

        report = split_checked_batch(
            self.batch,
            check_report_path,
            FilenameBarcodeDetector({1, 4, 8}),
        )

        self.assertEqual(report.units[0].status, "fail")
        self.assertIn("start", report.units[0].error or "")

    def test_book_is_one_document_numbered_01(self) -> None:
        self.add_pages("07", [1, 2, 3])
        check_report_path = self.create_check_report("book")

        report = split_checked_batch(
            self.batch,
            check_report_path,
            CheckBatchBarcodeDetector(),
        )

        self.assertEqual(report.status, "pass")
        self.assertEqual(
            report.units[0].documents[0].id,
            "B0012-07-01",
        )
        self.assertEqual(report.units[0].documents[0].pages, [1, 2, 3])

    def test_step1_failed_unit_is_skipped(self) -> None:
        (self.batch / "B9999-01-00001.tif").touch()
        check_report_path = self.create_check_report("book")

        report = split_checked_batch(
            self.batch,
            check_report_path,
            CheckBatchBarcodeDetector(),
        )

        self.assertEqual(report.units[0].status, "skipped")
        self.assertEqual(report.units[0].documents, [])

    def test_scan_files_sorted_by_path_are_matched_to_page_numbers(self) -> None:
        (self.batch / "z-scan").mkdir()
        (self.batch / "a-scan").mkdir()
        (self.batch / "z-scan" / "B0012-01-00001.tif").touch()
        (self.batch / "a-scan" / "B0012-01-00002.tif").touch()
        check_report_path = self.create_check_report("book")

        report = split_checked_batch(
            self.batch,
            check_report_path,
            CheckBatchBarcodeDetector(),
        )

        self.assertEqual(report.units[0].documents[0].pages, [1, 2])

    def test_batch_root_layout_resolves_scans_under_raw(self) -> None:
        raw_unit = self.batch / "raw" / "B0012" / "01"
        raw_unit.mkdir(parents=True)
        for page in [1, 2, 3, 4, 5, 6]:
            (raw_unit / f"B0012-01-{page:05d}.tif").touch()
        check_report_path = self.step1_output / "batch-report.json"
        report = check_batch(
            self.batch / "raw",
            self.box_list,
            "folder",
        )
        write_reports(report, self.step1_output)

        split_report = split_checked_batch(
            self.batch,
            check_report_path,
            FilenameBarcodeDetector({3}),
        )

        self.assertEqual(split_report.status, "pass")
        self.assertEqual(split_report.units[0].separator_pages, [3, 4])
        self.assertEqual(
            [document.pages for document in split_report.units[0].documents],
            [[1, 2], [5, 6]],
        )

    def test_missing_scan_makes_the_unit_fail(self) -> None:
        self.add_pages("01", [1, 2])
        check_report_path = self.create_check_report("book")
        (self.batch / "B0012-01-00001.tif").unlink()

        report = split_checked_batch(
            self.batch,
            check_report_path,
            CheckBatchBarcodeDetector(),
        )

        self.assertEqual(report.units[0].status, "fail")
        self.assertIn("is missing", report.units[0].error or "")

    def test_split_reports_write_batch_and_per_unit_files(self) -> None:
        self.add_pages("01", list(range(1, 5)))
        check_report_path = self.create_check_report("book")
        report = split_checked_batch(
            self.batch,
            check_report_path,
            CheckBatchBarcodeDetector(),
        )
        output = self.root / "step-2"

        write_split_report(report, output)

        batch_json = json.loads(
            (output / "split-report.json").read_text(encoding="utf-8")
        )
        unit_json = json.loads(
            (output / "B0012-01-documents.json").read_text(encoding="utf-8")
        )
        self.assertEqual(batch_json["status"], "pass")
        self.assertEqual(unit_json["documents"][0]["id"], "B0012-01-01")

    def test_zxing_detects_barcode_in_tiff(self) -> None:
        barcode = zxingcpp.create_barcode(
            "separator",
            zxingcpp.BarcodeFormat.QRCode,
        )
        bitmap = zxingcpp.write_barcode_to_image(barcode, scale=8)
        scan_path = self.root / "separator.tif"
        Image.fromarray(bitmap).save(scan_path)

        self.assertTrue(ZXingBarcodeDetector().contains_barcode(scan_path))
