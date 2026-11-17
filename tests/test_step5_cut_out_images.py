import json
import hashlib
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from ocr_pipeline.steps.step5_cut_out_images import (
    Step5Error,
    cut_out_document_images,
    write_cut_out_report,
)


class Step5Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.batch = Path(self.temporary_directory.name) / "batch01"
        self.raw_unit = self.batch / "raw" / "B0012" / "01"
        self.raw_unit.mkdir(parents=True)
        self.ocr_folder = self.batch / "work" / "step-4-ocr" / "B0012-01-01"
        (self.ocr_folder / "pages").mkdir(parents=True)
        self.output_folder = self.batch / "work" / "step-5-images" / "B0012-01-01"
        self._make_master_page(7)
        self._write_page_ocr(
            7,
            [
                {
                    "type": "figure",
                    "text": "diagram",
                    "bbox": [0.25, 0.25, 0.75, 0.75],
                },
                {
                    "type": "figure",
                    "text": "photo",
                    "bbox": [0.0, 0.0, 0.5, 0.5],
                    "image_kind": "photo",
                },
                {
                    "type": "text",
                    "text": "not a figure",
                    "bbox": [0, 0, 1, 1],
                },
            ],
        )
        self._write_ocr_report(
            [
                {
                    "page_number": 7,
                    "file": "B0012/01/B0012-01-00007.tif",
                    "status": "success",
                    "markdown_file": "pages/page-00007.md",
                    "raw_output_file": "raw/page-00007.json",
                }
            ]
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _make_master_page(self, page: int) -> None:
        image = Image.new("RGB", (200, 100), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((50, 25, 149, 74), fill="red")
        image.save(self.raw_unit / f"B0012-01-{page:05d}.tif")

    def _write_page_ocr(
        self,
        page: int,
        regions: list[dict[str, object]],
        status: str = "success",
    ) -> None:
        page_data = {
            "page_number": page,
            "status": status,
            "regions": regions,
        }
        (self.ocr_folder / "pages" / f"page-{page:05d}.json").write_text(
            json.dumps(page_data),
            encoding="utf-8",
        )

    def _write_ocr_report(self, pages: list[dict[str, object]]) -> None:
        report = {
            "schema_version": 1,
            "document_id": "B0012-01-01",
            "model": "test-ocr",
            "model_version": "test-version",
            "status": "pass",
            "pages": pages,
        }
        (self.ocr_folder / "ocr-report.json").write_text(
            json.dumps(report),
            encoding="utf-8",
        )

    def test_crops_full_resolution_uses_format_metadata_and_discards_bbox(self) -> None:
        source_path = self.raw_unit / "B0012-01-00007.tif"
        original_digest = hashlib.sha256(source_path.read_bytes()).digest()
        report = cut_out_document_images(
            self.batch,
            "B0012-01-01",
        )

        self.assertEqual(report.status, "pass")
        self.assertEqual(
            [image.file for image in report.images],
            ["images/p007-1.png", "images/p007-2.jpg"],
        )
        self.assertEqual(
            [image.image_format for image in report.images],
            ["PNG", "JPEG"],
        )
        png_path = self.output_folder / "images" / "p007-1.png"
        jpeg_path = self.output_folder / "images" / "p007-2.jpg"
        with Image.open(png_path) as image:
            self.assertEqual(image.size, (100, 50))
            self.assertEqual(image.format, "PNG")
        with Image.open(jpeg_path) as image:
            self.assertEqual(image.size, (100, 50))
            self.assertEqual(image.format, "JPEG")

        write_cut_out_report(report, self.output_folder)
        saved_report = json.loads(
            (self.output_folder / "cut-out-report.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertNotIn("bbox", saved_report["images"][0])
        self.assertEqual(
            hashlib.sha256(source_path.read_bytes()).digest(),
            original_digest,
        )

    def test_missing_bbox_fails_instead_of_guessing_crop(self) -> None:
        self._write_page_ocr(
            7,
            [{"type": "figure", "text": "figure without coordinates"}],
        )

        report = cut_out_document_images(self.batch, "B0012-01-01")

        self.assertEqual(report.status, "fail")
        self.assertIn("needs a normalized bbox", report.errors[0])

    def test_failed_ocr_report_is_rejected(self) -> None:
        report_path = self.ocr_folder / "ocr-report.json"
        ocr_report = json.loads(report_path.read_text(encoding="utf-8"))
        ocr_report["status"] = "fail"
        report_path.write_text(json.dumps(ocr_report), encoding="utf-8")

        with self.assertRaisesRegex(Step5Error, "status"):
            cut_out_document_images(self.batch, "B0012-01-01")

    def test_failed_ocr_page_is_skipped_without_failing_other_crops(self) -> None:
        self._make_master_page(8)
        self._write_page_ocr(8, [], status="failed")
        self._write_ocr_report(
            [
                {
                    "page_number": 7,
                    "file": "B0012/01/B0012-01-00007.tif",
                    "status": "success",
                    "markdown_file": "pages/page-00007.md",
                    "raw_output_file": "raw/page-00007.json",
                },
                {
                    "page_number": 8,
                    "file": "B0012/01/B0012-01-00008.tif",
                    "status": "failed",
                    "markdown_file": "pages/page-00008.md",
                    "raw_output_file": "raw/page-00008.json",
                },
            ]
        )
        report_path = self.ocr_folder / "ocr-report.json"
        ocr_report = json.loads(report_path.read_text(encoding="utf-8"))
        ocr_report["status"] = "fail"
        report_path.write_text(json.dumps(ocr_report), encoding="utf-8")

        report = cut_out_document_images(self.batch, "B0012-01-01")

        self.assertEqual(report.status, "pass")
        self.assertEqual([image.page_number for image in report.images], [7, 7])

    def test_blank_pages_have_no_figures(self) -> None:
        self._write_page_ocr(7, [], status="blank")

        report = cut_out_document_images(self.batch, "B0012-01-01")

        self.assertEqual(report.status, "pass")
        self.assertEqual(report.images, [])

    def test_rerun_removes_stale_pipeline_crops_after_success(self) -> None:
        images_folder = self.output_folder / "images"
        images_folder.mkdir(parents=True)
        stale_crop = images_folder / "p007-9.png"
        stale_crop.write_bytes(b"stale")

        report = cut_out_document_images(self.batch, "B0012-01-01")

        self.assertEqual(report.status, "pass")
        self.assertFalse(stale_crop.exists())

    def test_invalid_bbox_order_is_rejected(self) -> None:
        self._write_page_ocr(
            7,
            [
                {
                    "type": "figure",
                    "text": "invalid",
                    "bbox": [0.8, 0.2, 0.1, 0.9],
                }
            ],
        )

        report = cut_out_document_images(self.batch, "B0012-01-01")

        self.assertEqual(report.status, "fail")
        self.assertIn("non-positive bbox", report.errors[0])

    def test_bbox_outside_master_page_is_rejected(self) -> None:
        self._write_page_ocr(
            7,
            [
                {
                    "type": "figure",
                    "text": "out of bounds",
                    "bbox": [0.1, 0.1, 1.01, 0.9],
                }
            ],
        )

        report = cut_out_document_images(self.batch, "B0012-01-01")

        self.assertEqual(report.status, "fail")
        self.assertIn("normalized bbox", report.errors[0])


if __name__ == "__main__":
    unittest.main()
