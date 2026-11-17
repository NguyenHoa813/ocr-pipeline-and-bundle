import json
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml
from PIL import Image, ImageDraw

from ocr_pipeline.steps.step1_check_batch import check_batch, write_reports
from ocr_pipeline.steps.step2_split_documents import (
    ZXingBarcodeDetector,
    split_checked_batch,
    write_split_report,
)
from ocr_pipeline.steps.step3_flag_blank_pages import (
    flag_document_blank_pages,
    write_blank_page_report,
)
from ocr_pipeline.steps.step4_ocr_pages import (
    LayoutRegion,
    PageOCRContent,
    run_document_ocr,
    write_document_ocr,
)
from ocr_pipeline.steps.step5_cut_out_images import (
    cut_out_document_images,
    write_cut_out_report,
)
from ocr_pipeline.steps.step6_assemble_document import assemble_document
from ocr_pipeline.steps.step7_describe_images import (
    ImageDescriptionAdapter,
    describe_document_images,
    write_image_description_report,
)
from ocr_pipeline.steps.step8_quality_check import (
    check_document_quality,
    write_quality_report,
)
from ocr_pipeline.steps.step9_local_storage import publish_document_bundle
from ocr_pipeline.steps.run_tracking import record_step_run


class SelfTestError(Exception):
    """Raised when the generated end-to-end example does not pass."""


class _FakeOCRAdapter:
    model_name = "self-test-ocr"
    model_version = "1"

    def recognize(self, image_path: Path) -> tuple[PageOCRContent, object]:
        page_number = int(image_path.stem.rsplit("-", 1)[1])
        content = {
            1: PageOCRContent(
                markdown="# Sample document\n\nA generated first page.",
                regions=[
                    LayoutRegion(type="heading", text="Sample document"),
                    LayoutRegion(
                        type="figure",
                        text="A colored sample diagram",
                        bbox=[0.2, 0.2, 0.8, 0.8],
                        image_kind="drawing",
                    ),
                ],
            ),
            2: PageOCRContent(
                markdown="The second page contains generated test text.",
                regions=[LayoutRegion(type="text", text="Generated test text.")],
            ),
            4: PageOCRContent(
                markdown="The fourth page verifies the final scan marker.",
                regions=[LayoutRegion(type="text", text="Final marker.")],
            ),
        }[page_number]
        return content, {"self_test_page": page_number}


class _FakeImageDescriptionAdapter(ImageDescriptionAdapter):
    model_name = "self-test-image-description"

    def describe(self, image_path: Path) -> tuple[str, str | None]:
        if image_path.parent.name != "images":
            raise SelfTestError("Self-test attempted to describe a non-cropped image.")
        return (
            "Generated colored diagram used to test image descriptions.",
            "flowchart LR\nA[Input] --> B[Output]",
        )


@dataclass(frozen=True)
class SelfTestResult:
    status: str
    batch_root: Path
    document_id: str
    run_id: str
    published_bundle: Path
    checks: list[str]


def run_self_test(output_root: Path, run_id: str) -> SelfTestResult:
    """Generate TIFF fixtures and run all nine steps without network access."""
    output_root = Path(output_root).resolve()
    if output_root.exists():
        raise SelfTestError(f"Self-test output already exists: {output_root}")
    output_root.mkdir(parents=True)
    batch_root = output_root / "batch"
    scan_folder = batch_root / "raw" / "B0012" / "01"
    scan_folder.mkdir(parents=True)
    document_id = "B0012-01-01"
    for page_number in range(1, 5):
        image = Image.new("RGB", (240, 320), "white")
        if page_number != 3:
            draw = ImageDraw.Draw(image)
            draw.text((20, 20), f"Generated test page {page_number}", fill="black")
            draw.rectangle((50, 80, 190, 250), outline="navy", width=4)
            if page_number == 1:
                draw.ellipse((80, 120, 160, 200), fill="skyblue", outline="black")
        image.save(scan_folder / f"B0012-01-{page_number:05d}.tif")

    batch_root.mkdir(parents=True, exist_ok=True)
    (batch_root / "box_list.csv").write_text(
        "box_code\nB0012\n",
        encoding="utf-8",
    )
    (batch_root / "batch_manifest.yaml").write_text(
        yaml.safe_dump(
            {
                "batch_id": "generated-self-test",
                "operator": {"name": "Self test"},
                "scan": {
                    "date": "2026-01-01",
                    "scanner_model": "synthetic",
                    "resolution_dpi": 300,
                    "color_mode": "grayscale_8bit",
                    "format": "TIFF",
                    "duplex": False,
                },
                "items": [
                    {
                        "box_code": "B0012",
                        "item_number": "01",
                        "item_type": "book",
                        "total_pages": 4,
                        "printed_pages": 3,
                        "damaged_pages": [],
                    }
                ],
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    report_folder = batch_root / "work" / "step-1-checks"
    check_report = check_batch(
        batch_root / "raw",
        batch_root / "box_list.csv",
        None,
        batch_root / "batch_manifest.yaml",
    )
    write_reports(check_report, report_folder)
    if check_report.status != "pass":
        raise SelfTestError("Generated batch failed Step 1.")
    record_step_run(
        batch_root,
        run_id,
        "step-1",
        "pass",
        [scan_folder, batch_root / "box_list.csv", batch_root / "batch_manifest.yaml"],
        [report_folder],
    )

    split_folder = batch_root / "work" / "step-2-documents"
    split_report = split_checked_batch(
        batch_root,
        report_folder / "batch-report.json",
        ZXingBarcodeDetector(),
    )
    write_split_report(split_report, split_folder)
    if split_report.status != "pass":
        raise SelfTestError("Generated batch failed Step 2.")
    record_step_run(
        batch_root,
        run_id,
        "step-2",
        "pass",
        [scan_folder, report_folder / "batch-report.json"],
        [split_folder],
    )

    blank_report = flag_document_blank_pages(
        batch_root,
        document_id,
        0.01,
    )
    blank_path = batch_root / "work" / "step-3-blank-pages" / f"{document_id}.json"
    write_blank_page_report(blank_report, blank_path)
    if not any(page.page_number == 3 and page.is_blank for page in blank_report.pages):
        raise SelfTestError("Generated white page was not identified as blank.")
    record_step_run(
        batch_root,
        run_id,
        "step-3",
        "pass",
        [split_folder / "split-report.json", report_folder / "batch-report.json", scan_folder],
        [blank_path],
        {"dark_pixel_share_threshold": 0.01},
    )

    ocr_report, normalized, raw = run_document_ocr(
        batch_root,
        document_id,
        _FakeOCRAdapter(),
    )
    ocr_folder = batch_root / "work" / "step-4-ocr" / document_id
    write_document_ocr(ocr_report, normalized, raw, ocr_folder)
    if ocr_report.status != "pass":
        raise SelfTestError("Generated document failed Step 4.")
    record_step_run(
        batch_root,
        run_id,
        "step-4",
        "pass",
        [scan_folder, split_folder / "split-report.json", blank_path],
        [ocr_folder],
        {"model": "self-test-ocr", "model_version": "1"},
    )

    image_folder = batch_root / "work" / "step-5-images" / document_id
    crop_report = cut_out_document_images(batch_root, document_id)
    write_cut_out_report(crop_report, image_folder)
    if crop_report.status != "pass" or len(crop_report.images) != 1:
        raise SelfTestError("Generated figure failed Step 5.")
    record_step_run(
        batch_root,
        run_id,
        "step-5",
        "pass",
        [scan_folder, ocr_folder],
        [image_folder],
    )

    description_path = (
        batch_root / "work" / "step-7-image-descriptions" / f"{document_id}.json"
    )
    description_report = describe_document_images(
        document_id,
        image_folder,
        image_folder / "cut-out-report.json",
        _FakeImageDescriptionAdapter(),
    )
    write_image_description_report(description_report, description_path)
    if description_report.status != "pass":
        raise SelfTestError("Generated crop failed Step 7.")
    record_step_run(
        batch_root,
        run_id,
        "step-7",
        "pass",
        [image_folder],
        [description_path],
        {"model": "self-test-image-description", "external_crop_upload_consent": False},
    )

    bundle = assemble_document(
        batch_root,
        document_id,
        description_report_path=description_path,
    )
    record_step_run(
        batch_root,
        run_id,
        "step-6",
        "pass",
        [split_folder, report_folder, ocr_folder, image_folder, description_path],
        [bundle.output_folder],
    )
    markdown = (bundle.output_folder / "document.md").read_text(encoding="utf-8")
    if "```mermaid" not in markdown or "printed" in markdown:
        raise SelfTestError("Generated Markdown did not assemble Step 7 outputs.")

    quality_folder = batch_root / "work" / "step-8-quality"
    flag_log_path = batch_root / "work" / "logs" / "flag_log.csv"
    quality_report = check_document_quality(
        bundle.output_folder,
        document_id,
        ocr_folder / "ocr-report.json",
        run_id=run_id,
        flag_log_path=flag_log_path,
    )
    quality_path = quality_folder / f"{document_id}.json"
    write_quality_report(quality_report, quality_path)
    if quality_report.bundle_status != "pass" or quality_report.page_flags:
        raise SelfTestError(
            "Generated bundle failed Step 8: "
            + "; ".join(quality_report.errors)
        )
    record_step_run(
        batch_root,
        run_id,
        "step-8",
        "pass",
        [bundle.output_folder, ocr_folder / "ocr-report.json"],
        [quality_path, flag_log_path],
    )

    publish = publish_document_bundle(
        bundle.output_folder,
        document_id,
        quality_path,
        output_root / "storage",
        run_id,
    )
    if publish.status != "published":
        raise SelfTestError("Generated valid bundle was not published.")
    record_step_run(
        batch_root,
        run_id,
        "step-9",
        "pass",
        [bundle.output_folder, quality_path],
        [publish.destination],
    )

    checks = [
        "manifest-and-input-validation",
        "document-splitting",
        "blank-page-preservation",
        "mock-local-ocr",
        "lossless-figure-crop",
        "mock-image-description",
        "markdown-assembly",
        "bundle-and-page-quality",
        "atomic-local-publication",
    ]
    result = SelfTestResult(
        status="pass",
        batch_root=batch_root,
        document_id=document_id,
        run_id=run_id,
        published_bundle=publish.destination,
        checks=checks,
    )
    (output_root / "self-test-report.json").write_text(
        json.dumps(asdict(result), ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return result
