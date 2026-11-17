import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from ocr_pipeline.steps.step6_assemble_document import (
    Step6Error,
    assemble_document,
)


class Step6Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.batch = Path(self.temporary_directory.name) / "batch01"
        self.raw_unit = self.batch / "raw" / "B0012" / "01"
        self.raw_unit.mkdir(parents=True)
        self.work = self.batch / "work"
        self.ocr_folder = self.work / "step-4-ocr" / "B0012-01-01"
        self.pages_folder = self.ocr_folder / "pages"
        self.pages_folder.mkdir(parents=True)
        self.image_folder = self.work / "step-5-images" / "B0012-01-01"
        (self.image_folder / "images").mkdir(parents=True)
        for page in (1, 2):
            Image.new("RGB", (10, 10), "white").save(
                self.raw_unit / f"B0012-01-{page:05d}.tif"
            )
        self._write_json(
            self.work / "step-1-checks" / "batch-report.json",
            {
                "schema_version": 1,
                "units": [
                    {
                        "document_unit_id": "B0012-01",
                        "status": "pass",
                        "files": [
                            "B0012/01/B0012-01-00001.tif",
                            "B0012/01/B0012-01-00002.tif",
                        ],
                    }
                ],
            },
        )
        self._write_json(
            self.work / "step-2-documents" / "split-report.json",
            {
                "schema_version": 1,
                "units": [
                    {
                        "unit_id": "B0012-01",
                        "status": "pass",
                        "documents": [
                            {
                                "id": "B0012-01-01",
                                "first_page": 1,
                                "last_page": 2,
                                "pages": [1, 2],
                            }
                        ],
                    }
                ],
            },
        )
        self._write_page(1, "Nội dung trang đầu.", "success")
        self._write_page(2, "", "blank")
        self._write_json(
            self.ocr_folder / "ocr-report.json",
            {
                "schema_version": 1,
                "document_id": "B0012-01-01",
                "model": "test-ocr",
                "model_version": "test-version",
                "status": "pass",
                "pages": [
                    {
                        "page_number": 1,
                        "file": "B0012/01/B0012-01-00001.tif",
                        "status": "success",
                        "markdown_file": "pages/page-00001.md",
                    },
                    {
                        "page_number": 2,
                        "file": "B0012/01/B0012-01-00002.tif",
                        "status": "blank",
                        "markdown_file": "pages/page-00002.md",
                    },
                ],
            },
        )
        Image.new("RGB", (4, 4), "red").save(
            self.image_folder / "images" / "p001-1.png"
        )
        self._write_json(
            self.image_folder / "cut-out-report.json",
            {
                "schema_version": 1,
                "document_id": "B0012-01-01",
                "status": "pass",
                "images": [
                    {
                        "page_number": 1,
                        "order_on_page": 1,
                        "file": "images/p001-1.png",
                    }
                ],
            },
        )
        self._write_json(
            self.work
            / "step-7-image-descriptions"
            / "B0012-01-01.json",
            {
                "schema_version": 1,
                "document_id": "B0012-01-01",
                "model": "test-vision",
                "status": "pass",
                "images": [
                    {
                        "file": "images/p001-1.png",
                        "status": "success",
                        "alt_text": "Sơ đồ thiết bị",
                        "mermaid": None,
                    }
                ],
            },
        )
        self.output = self.work / "step-6-assemble" / "B0012-01-01"

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    @staticmethod
    def _write_json(path: Path, value: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    def _write_page(self, page: int, markdown: str, status: str) -> None:
        (self.pages_folder / f"page-{page:05d}.md").write_text(
            markdown,
            encoding="utf-8",
        )
        self._write_json(
            self.pages_folder / f"page-{page:05d}.json",
            {
                "page_number": page,
                "status": status,
                "regions": (
                    [{"type": "figure", "text": "Sơ đồ thiết bị"}]
                    if page == 1
                    else []
                ),
            },
        )

    def test_assembles_markers_frontmatter_and_portable_images(self) -> None:
        result = assemble_document(self.batch, "B0012-01-01")

        self.assertEqual(result.pages, [1, 2])
        self.assertEqual(result.images, ["images/p001-1.png"])
        markdown = (self.output / "document.md").read_text(encoding="utf-8")
        self.assertTrue(markdown.startswith("---\nid: \"B0012-01-01\""))
        self.assertIn('title: ""', markdown)
        self.assertIn('type: "unknown"', markdown)
        self.assertIn("pages: [1, 2]", markdown)
        self.assertIn(
            'source: ["B0012-01-00001.tif", "B0012-01-00002.tif"]',
            markdown,
        )
        self.assertIn("<!-- page 1 -->", markdown)
        self.assertIn("<!-- page 2 -->", markdown)
        self.assertIn("![Sơ đồ thiết bị](images/p001-1.png)", markdown)
        self.assertTrue(markdown.endswith("<!-- page 2 -->\n"))
        self.assertTrue((self.output / "images" / "p001-1.png").is_file())

    def test_assembly_rejects_failed_ocr(self) -> None:
        report_path = self.ocr_folder / "ocr-report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["status"] = "fail"
        report_path.write_text(json.dumps(report), encoding="utf-8")

        with self.assertRaisesRegex(Step6Error, "status"):
            assemble_document(self.batch, "B0012-01-01")

    def test_failed_ocr_page_keeps_marker_and_allows_assembly(self) -> None:
        self._write_page(2, "", "failed")
        report_path = self.ocr_folder / "ocr-report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["status"] = "fail"
        report["pages"][1]["status"] = "failed"
        report["pages"][1]["error"] = "request timed out"
        report_path.write_text(json.dumps(report), encoding="utf-8")

        assemble_document(self.batch, "B0012-01-01")

        markdown = (self.output / "document.md").read_text(encoding="utf-8")
        self.assertIn("<!-- page 2 -->", markdown)

    def test_joins_plain_paragraph_continuations_before_next_page_marker(self) -> None:
        self._write_page(1, "Nội dung tiếp tục ở", "success")
        self._write_page(2, "trang kế tiếp.\n\nĐiều 2. Nội dung mới.", "success")

        assemble_document(self.batch, "B0012-01-01")

        markdown = (self.output / "document.md").read_text(encoding="utf-8")
        self.assertIn("Nội dung tiếp tục ở trang kế tiếp.", markdown)
        self.assertLess(
            markdown.index("trang kế tiếp."),
            markdown.index("<!-- page 2 -->"),
        )
        self.assertIn(
            "<!-- page 2 -->\n\nĐiều 2. Nội dung mới.",
            markdown,
        )

    def test_title_printed_page_and_repeated_header_are_assembled(self) -> None:
        self._write_page(1, "# Contract title\n\nCompany header\n\nunfinished body", "success")
        self._write_page(2, "Company header\n\ncontinues here.", "success")
        first_regions = json.loads(
            (self.pages_folder / "page-00001.json").read_text(encoding="utf-8")
        )
        first_regions["regions"].append(
            {"type": "page_header", "text": "Company header"}
        )
        (self.pages_folder / "page-00001.json").write_text(
            json.dumps(first_regions),
            encoding="utf-8",
        )
        second_regions = json.loads(
            (self.pages_folder / "page-00002.json").read_text(encoding="utf-8")
        )
        second_regions["regions"].extend(
            [
                {"type": "page_header", "text": "Company header"},
                {"type": "page_number", "text": "2"},
            ]
        )
        (self.pages_folder / "page-00002.json").write_text(
            json.dumps(second_regions),
            encoding="utf-8",
        )

        assemble_document(self.batch, "B0012-01-01")

        markdown = (self.output / "document.md").read_text(encoding="utf-8")
        self.assertIn('title: "Contract title"', markdown)
        self.assertNotIn("Company header", markdown)
        self.assertIn("unfinished body continues here.", markdown)
        self.assertIn("<!-- page 2, printed 2 -->", markdown)

    def test_assembly_rejects_image_paths_outside_step5_folder(self) -> None:
        report_path = self.image_folder / "cut-out-report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["images"][0]["file"] = "../outside.png"
        report_path.write_text(json.dumps(report), encoding="utf-8")

        with self.assertRaisesRegex(Step6Error, "inside its images folder"):
            assemble_document(self.batch, "B0012-01-01")

    def test_assembly_requires_descriptions_for_extracted_figures(self) -> None:
        description_report = (
            self.work
            / "step-7-image-descriptions"
            / "B0012-01-01.json"
        )
        description_report.unlink()

        with self.assertRaisesRegex(Step6Error, "required"):
            assemble_document(self.batch, "B0012-01-01")

    def test_assembly_can_skip_step7_and_use_ocr_figure_label(self) -> None:
        description_report = (
            self.work
            / "step-7-image-descriptions"
            / "B0012-01-01.json"
        )
        description_report.unlink()

        result = assemble_document(
            self.batch,
            "B0012-01-01",
            allow_missing_descriptions=True,
        )

        markdown = (result.output_folder / "document.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("![Sơ đồ thiết bị](images/p001-1.png)", markdown)


if __name__ == "__main__":
    unittest.main()
