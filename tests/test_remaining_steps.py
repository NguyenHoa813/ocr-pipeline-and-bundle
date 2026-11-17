import csv
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path

from PIL import Image

from ocr_pipeline.cli import main
from ocr_pipeline.steps.step7_describe_images import (
    Step7Error,
    describe_document_images,
    write_image_description_report,
)
from ocr_pipeline.steps.step8_quality_check import check_document_quality
from ocr_pipeline.steps.step9_local_storage import (
    Step9Error,
    publish_document_bundle,
)


class FakeImageAdapter:
    model_name = "fake-vision"

    def __init__(self) -> None:
        self.inputs: list[Path] = []

    def describe(self, image_path: Path) -> tuple[str, str | None]:
        self.inputs.append(image_path)
        if image_path.parent.name != "images":
            raise AssertionError("Only image crops may reach the adapter.")
        return "A sample diagram.", "flowchart LR\nA --> B"


class RemainingStepTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.document_id = "B0012-01-01"

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_vertex_upload_requires_explicit_consent_flag(self) -> None:
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            exit_code = main(
                [
                    "describe-images",
                    "--batch",
                    str(self.root / "batch"),
                    "--document-id",
                    self.document_id,
                    "--project",
                    "test-project",
                    "--location",
                    "us-central1",
                ]
            )

        self.assertEqual(exit_code, 2)
        self.assertIn("disabled", stderr.getvalue())

    @staticmethod
    def _write_json(path: Path, value: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    def test_step7_uses_crop_report_and_writes_mermaid(self) -> None:
        image_folder = self.root / "step-5"
        image_path = image_folder / "images" / "p001-1.png"
        image_path.parent.mkdir(parents=True)
        Image.new("RGB", (20, 20), "white").save(image_path)
        cut_report = image_folder / "cut-out-report.json"
        self._write_json(
            cut_report,
            {
                "schema_version": 1,
                "document_id": self.document_id,
                "status": "pass",
                "images": [
                    {"file": "images/p001-1.png", "page_number": 1}
                ],
            },
        )
        adapter = FakeImageAdapter()

        report = describe_document_images(
            self.document_id,
            image_folder,
            cut_report,
            adapter,
        )
        output = self.root / "description.json"
        write_image_description_report(report, output)

        self.assertEqual(report.status, "pass")
        self.assertEqual([path for path in adapter.inputs], [image_path.resolve()])
        saved = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(saved["images"][0]["alt_text"], "A sample diagram.")
        self.assertTrue(saved["images"][0]["mermaid"].startswith("flowchart"))

    def test_step7_rejects_non_crop_paths_before_adapter_call(self) -> None:
        image_folder = self.root / "step-5"
        cut_report = image_folder / "cut-out-report.json"
        self._write_json(
            cut_report,
            {
                "schema_version": 1,
                "document_id": self.document_id,
                "status": "pass",
                "images": [
                    {"file": "../raw/source.tif", "page_number": 1}
                ],
            },
        )
        adapter = FakeImageAdapter()

        with self.assertRaisesRegex(Step7Error, "crop path"):
            describe_document_images(
                self.document_id,
                image_folder,
                cut_report,
                adapter,
            )
        self.assertEqual(adapter.inputs, [])

    def _make_valid_bundle_and_ocr_report(self) -> tuple[Path, Path]:
        bundle = self.root / self.document_id
        bundle.mkdir()
        (bundle / "images").mkdir()
        Image.new("RGB", (20, 20), "white").save(bundle / "images" / "p001-1.png")
        markdown = (
            "---\n"
            f'id: "{self.document_id}"\n'
            'title: "Sample"\n'
            'type: "unknown"\n'
            'language: "vi"\n'
            "pages: [1, 4]\n"
            'source: ["B0012-01-00001.tif", "B0012-01-00004.tif"]\n'
            'ocr: "test-model 1"\n'
            "---\n\n"
            "<!-- page 1 -->\n\n"
            "![A sample diagram](images/p001-1.png)\n\n"
            "<!-- page 2 -->\n\n"
            "<!-- page 3 -->\n\n"
            "<!-- page 4 -->\n"
        )
        (bundle / "document.md").write_text(markdown, encoding="utf-8")
        ocr_folder = self.root / "ocr"
        pages_folder = ocr_folder / "pages"
        pages_folder.mkdir(parents=True)
        repeated_text = "This generated text is deliberately repeated across OCR pages."
        (pages_folder / "page-00001.md").write_text(repeated_text, encoding="utf-8")
        (pages_folder / "page-00002.md").write_text("", encoding="utf-8")
        (pages_folder / "page-00003.md").write_text(repeated_text, encoding="utf-8")
        ocr_report = ocr_folder / "ocr-report.json"
        self._write_json(
            ocr_report,
            {
                "schema_version": 1,
                "document_id": self.document_id,
                "model": "test-model",
                "model_version": "1",
                "status": "fail",
                "pages": [
                    {"page_number": 1, "status": "success", "markdown_file": "pages/page-00001.md"},
                    {"page_number": 2, "status": "success", "markdown_file": "pages/page-00002.md"},
                    {"page_number": 3, "status": "success", "markdown_file": "pages/page-00003.md"},
                    {"page_number": 4, "status": "failed", "error": "request timed out"},
                ],
            },
        )
        return bundle, ocr_report

    def test_step8_separates_valid_bundle_from_page_quality_flags(self) -> None:
        bundle, ocr_report = self._make_valid_bundle_and_ocr_report()
        vocabulary = self.root / "vocabulary.txt"
        vocabulary.write_text("this\ngenerated\ntext\n", encoding="utf-8")
        flag_log = self.root / "flag_log.csv"

        report = check_document_quality(
            bundle,
            self.document_id,
            ocr_report,
            vocabulary,
            0.9,
            run_id="run-test",
            flag_log_path=flag_log,
        )

        self.assertEqual(report.bundle_status, "pass")
        flag_types = {flag.flag_type for flag in report.page_flags}
        self.assertTrue({"EMPTY_OCR", "REPEATED_OCR", "LOW_VOCABULARY", "TIMEOUT"} <= flag_types)
        with flag_log.open(encoding="utf-8", newline="") as file:
            rows = list(csv.DictReader(file))
        self.assertEqual({row["run_id"] for row in rows}, {"run-test"})

    def test_step8_rejects_missing_image_file(self) -> None:
        bundle, ocr_report = self._make_valid_bundle_and_ocr_report()
        (bundle / "images" / "p001-1.png").unlink()

        report = check_document_quality(
            bundle,
            self.document_id,
            ocr_report,
            run_id="run-test",
        )

        self.assertEqual(report.bundle_status, "fail")
        self.assertTrue(any("missing" in error for error in report.errors))

    def test_step9_publishes_only_valid_bundle_without_overwrite(self) -> None:
        bundle, ocr_report = self._make_valid_bundle_and_ocr_report()
        report = check_document_quality(
            bundle,
            self.document_id,
            ocr_report,
            run_id="run-publish",
        )
        report_path = self.root / "quality.json"
        self._write_json(report_path, {
            "schema_version": 1,
            "document_id": report.document_id,
            "run_id": report.run_id,
            "bundle_status": report.bundle_status,
            "errors": report.errors,
            "page_flags": [],
        })

        result = publish_document_bundle(
            bundle,
            self.document_id,
            report_path,
            self.root / "storage",
            "run-publish",
        )

        self.assertEqual(result.status, "published")
        self.assertTrue((result.destination / "document.md").is_file())
        self.assertTrue((result.destination / "images" / "p001-1.png").is_file())
        self.assertEqual(
            {path.name for path in result.destination.iterdir()},
            {"document.md", "images"},
        )
        with self.assertRaisesRegex(Step9Error, "refusing to overwrite"):
            publish_document_bundle(
                bundle,
                self.document_id,
                report_path,
                self.root / "storage",
                "run-publish",
            )

    def test_step9_quarantines_structurally_failed_bundle(self) -> None:
        bundle, _ = self._make_valid_bundle_and_ocr_report()
        report_path = self.root / "quality-fail.json"
        self._write_json(
            report_path,
            {
                "schema_version": 1,
                "document_id": self.document_id,
                "run_id": "run-reject",
                "bundle_status": "fail",
                "errors": ["broken bundle"],
                "page_flags": [],
            },
        )

        result = publish_document_bundle(
            bundle,
            self.document_id,
            report_path,
            self.root / "storage",
            "run-reject",
        )

        self.assertEqual(result.status, "rejected")
        self.assertEqual(result.destination.parent.name, "rejected")
        self.assertFalse((self.root / "storage" / "output" / "valid").exists())


if __name__ == "__main__":
    unittest.main()
