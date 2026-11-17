import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from PIL import Image, UnidentifiedImageError

from ocr_pipeline.steps.step1_check_batch import resolve_scan_folder


class Step5Error(Exception):
    """Raised when figure regions cannot be cropped from master scans."""


ImageFormat = Literal["PNG", "JPEG"]

PAGE_FILENAME_PATTERN = re.compile(r"-(?P<page>\d{5})\.tif$", re.IGNORECASE)
DOCUMENT_ID_PATTERN = re.compile(r"[A-Za-z0-9]+-\d+-\d{2}")
CROP_FILENAME_PATTERN = re.compile(r"p\d{3}-\d+\.(?:png|jpg)$", re.IGNORECASE)


@dataclass(frozen=True)
class CutOutImage:
    page_number: int
    order_on_page: int
    file: str
    source_file: str
    image_kind: Literal["drawing", "photo"] | None
    image_format: ImageFormat


@dataclass
class CutOutReport:
    schema_version: int
    document_id: str
    status: Literal["pass", "fail"]
    format_policy: str
    images: list[CutOutImage]
    errors: list[str]


def _read_json(path: Path, label: str) -> dict[str, object]:
    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        raise Step5Error(f"Cannot read {label} {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise Step5Error(f"{label} must contain a JSON object.")
    return data


def _validated_page_number(page: dict[str, object]) -> int:
    page_number = page.get("page_number")
    if type(page_number) is not int or page_number <= 0:
        raise Step5Error("Step 4 OCR report contains an invalid page number.")
    return page_number


def _load_ocr_pages(
    ocr_report_path: Path,
    document_id: str,
) -> list[dict[str, object]]:
    report = _read_json(ocr_report_path, "Step 4 OCR report")
    if report.get("schema_version") != 1 or report.get("document_id") != document_id:
        raise Step5Error(
            "Step 4 report is invalid or belongs to a different document."
        )
    pages = report.get("pages")
    if not isinstance(pages, list):
        raise Step5Error("Step 4 OCR report has no pages list.")
    report_status = report.get("status")
    if report_status not in {"pass", "fail"}:
        raise Step5Error("Step 4 OCR report has an invalid overall status.")
    result: list[dict[str, object]] = []
    seen_pages: set[int] = set()
    for page in pages:
        if not isinstance(page, dict):
            raise Step5Error("Step 4 OCR report contains an invalid page.")
        page_number = _validated_page_number(page)
        status = page.get("status")
        if page_number in seen_pages:
            raise Step5Error(f"Duplicate page {page_number:05d} in Step 4 report.")
        if status not in {"success", "blank", "failed"}:
            raise Step5Error(
                f"Page {page_number:05d} has an invalid OCR status."
            )
        seen_pages.add(page_number)
        result.append(page)
    has_failed_pages = any(page.get("status") == "failed" for page in result)
    if (report_status == "fail") != has_failed_pages:
        raise Step5Error(
            "Step 4 report status does not match its per-page OCR statuses."
        )
    return sorted(result, key=_validated_page_number)


def _load_figure_regions(
    page_regions_path: Path,
    page_number: int,
) -> list[dict[str, object]]:
    page_data = _read_json(page_regions_path, "Step 4 page regions")
    if type(page_data.get("page_number")) is not int or (
        page_data["page_number"] != page_number
    ):
        raise Step5Error(
            f"Page regions file {page_regions_path} has the wrong page number."
        )
    if page_data.get("status") != "success":
        if page_data.get("status") == "blank":
            return []
        raise Step5Error(
            f"Page {page_number:05d} has no successful OCR regions."
        )
    regions = page_data.get("regions")
    if not isinstance(regions, list):
        raise Step5Error(
            f"Page {page_number:05d} OCR output has no regions list."
        )
    figures: list[dict[str, object]] = []
    for index, region in enumerate(regions):
        if not isinstance(region, dict):
            raise Step5Error(
                f"Page {page_number:05d} region {index} is invalid."
            )
        if region.get("type") == "figure":
            figures.append(region)
    return figures


def _crop_bounds(
    bbox: object,
    image_width: int,
    image_height: int,
    page_number: int,
    region_index: int,
) -> tuple[int, int, int, int]:
    if (
        not isinstance(bbox, list)
        or len(bbox) != 4
        or not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            and 0 <= value <= 1
            for value in bbox
        )
    ):
        raise Step5Error(
            f"Page {page_number:05d} figure region {region_index} needs a "
            "normalized bbox with four values in 0..1."
        )
    x0, y0, x1, y1 = bbox
    if x0 >= x1 or y0 >= y1:
        raise Step5Error(
            f"Page {page_number:05d} figure region {region_index} has "
            "non-positive bbox dimensions."
        )
    left = max(0, min(image_width, math.floor(x0 * image_width)))
    top = max(0, min(image_height, math.floor(y0 * image_height)))
    right = max(0, min(image_width, math.ceil(x1 * image_width)))
    bottom = max(0, min(image_height, math.ceil(y1 * image_height)))
    if left >= right or top >= bottom:
        raise Step5Error(
            f"Page {page_number:05d} figure region {region_index} produces "
            "an empty crop."
        )
    return left, top, right, bottom


def _save_crop(
    master_path: Path,
    bbox: object,
    output_path: Path,
    page_number: int,
    region_index: int,
    image_kind: object,
    jpeg_quality: int,
) -> ImageFormat:
    if image_kind not in (None, "drawing", "photo"):
        raise Step5Error(
            f"Page {page_number:05d} figure region {region_index} has an "
            "unsupported image_kind."
        )
    image_format: ImageFormat = "JPEG" if image_kind == "photo" else "PNG"
    try:
        with Image.open(master_path) as master:
            master.seek(0)
            crop = master.crop(
                _crop_bounds(
                    bbox,
                    master.width,
                    master.height,
                    page_number,
                    region_index,
                )
            )
            if image_format == "JPEG":
                crop.convert("RGB").save(
                    output_path,
                    format=image_format,
                    quality=jpeg_quality,
                    subsampling=0,
                )
            else:
                crop.save(output_path, format=image_format)
    except (OSError, UnidentifiedImageError) as exc:
        raise Step5Error(f"Cannot crop master TIFF {master_path}: {exc}") from exc
    return image_format


def cut_out_document_images(
    batch_root: Path,
    document_id: str,
    ocr_output_folder: Path | None = None,
    output_folder: Path | None = None,
    jpeg_quality: int = 95,
) -> CutOutReport:
    """Cut each figure region from its original TIFF page."""
    if not DOCUMENT_ID_PATTERN.fullmatch(document_id):
        raise Step5Error(f"Invalid document ID: {document_id}")
    if not 1 <= jpeg_quality <= 100:
        raise Step5Error("JPEG quality must be between 1 and 100.")

    batch_root = Path(batch_root).resolve()
    scan_folder = resolve_scan_folder(batch_root)
    if not scan_folder.is_dir():
        raise Step5Error(f"Scan folder does not exist: {scan_folder}")
    ocr_output_folder = Path(
        ocr_output_folder
        or batch_root / "work" / "step-4-ocr" / document_id
    ).resolve()
    output_folder = Path(
        output_folder
        or batch_root / "work" / "step-5-images" / document_id
    ).resolve()
    if output_folder.is_relative_to(scan_folder):
        raise Step5Error("Image output must be outside raw scans.")
    ocr_report_path = ocr_output_folder / "ocr-report.json"
    pages_folder = ocr_output_folder / "pages"
    pages = _load_ocr_pages(ocr_report_path, document_id)

    images_folder = output_folder / "images"
    images_folder.mkdir(parents=True, exist_ok=True)
    cut_images: list[CutOutImage] = []
    errors: list[str] = []

    for page in pages:
        page_number = _validated_page_number(page)
        relative_file = page.get("file")
        if not isinstance(relative_file, str):
            errors.append(f"Page {page_number:05d} has no source TIFF path.")
            continue
        page_match = PAGE_FILENAME_PATTERN.search(relative_file)
        if page_match is None or int(page_match.group("page")) != page_number:
            errors.append(
                f"Page {page_number:05d} has an invalid source filename."
            )
            continue
        master_path = (scan_folder / relative_file).resolve()
        if not master_path.is_relative_to(scan_folder):
            errors.append(
                f"Page {page_number:05d} source path escapes raw scans."
            )
            continue
        if not master_path.is_file():
            errors.append(f"Page {page_number:05d} source TIFF is missing.")
            continue
        if page.get("status") in {"blank", "failed"}:
            continue

        page_regions_path = pages_folder / f"page-{page_number:05d}.json"
        try:
            figure_regions = _load_figure_regions(
                page_regions_path,
                page_number,
            )
            for order, region in enumerate(figure_regions, start=1):
                image_kind = region.get("image_kind")
                if image_kind not in {None, "drawing", "photo"}:
                    raise Step5Error(
                        f"Page {page_number:05d} figure region {order} has "
                        "an unsupported image_kind."
                    )
                normalized_image_kind: Literal["drawing", "photo"] | None
                if image_kind == "drawing":
                    normalized_image_kind = "drawing"
                elif image_kind == "photo":
                    normalized_image_kind = "photo"
                else:
                    normalized_image_kind = None
                extension = (
                    ".jpg" if normalized_image_kind == "photo" else ".png"
                )
                output_name = f"p{page_number:03d}-{order}{extension}"
                output_path = images_folder / output_name
                image_format = _save_crop(
                    master_path,
                    region.get("bbox"),
                    output_path,
                    page_number,
                    order,
                    normalized_image_kind,
                    jpeg_quality,
                )
                cut_images.append(
                    CutOutImage(
                        page_number=page_number,
                        order_on_page=order,
                        file=f"images/{output_name}",
                        source_file=relative_file,
                        image_kind=normalized_image_kind,
                        image_format=image_format,
                    )
                )
        except Step5Error as exc:
            errors.append(str(exc))

    report = CutOutReport(
        schema_version=1,
        document_id=document_id,
        status="pass" if not errors else "fail",
        format_policy=(
            "Use Step 4 image_kind when available; default to lossless PNG "
            "when the type is absent or uncertain."
        ),
        images=cut_images,
        errors=errors,
    )
    if report.status == "pass" and images_folder.is_dir():
        expected_files = {Path(image.file).name for image in cut_images}
        for existing_file in images_folder.iterdir():
            if (
                existing_file.is_file()
                and CROP_FILENAME_PATTERN.fullmatch(existing_file.name)
                and existing_file.name not in expected_files
            ):
                existing_file.unlink()
    return report


def write_cut_out_report(report: CutOutReport, output_folder: Path) -> None:
    """Write crop metadata without retaining any region coordinates."""
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    (output_folder / "cut-out-report.json").write_text(
        json.dumps(asdict(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
