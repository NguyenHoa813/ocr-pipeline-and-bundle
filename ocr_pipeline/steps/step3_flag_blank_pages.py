import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from PIL import Image, UnidentifiedImageError

from ocr_pipeline.steps.step1_check_batch import resolve_scan_folder


class Step3Error(Exception):
    """Raised when blank-page detection inputs cannot be processed."""


@dataclass(frozen=True)
class PageBlankFlag:
    page_number: int
    file: str
    is_blank: bool
    dark_pixel_share: float
    dark_pixel_share_threshold: float


@dataclass
class BlankPageReport:
    schema_version: int
    document_id: str
    status: Literal["pass", "fail"]
    dark_pixel_intensity_cutoff: int
    dark_pixel_share_threshold: float
    pages: list[PageBlankFlag]


PAGE_FILENAME_PATTERN = re.compile(r"-(?P<page>\d{5})\.tif$", re.IGNORECASE)
DARK_PIXEL_INTENSITY_CUTOFF = 128


def _read_json(path: Path, label: str) -> dict[str, object]:
    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        raise Step3Error(f"Cannot read {label} {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise Step3Error(f"{label} must contain a JSON object.")
    return data


def _index_unit_scans(check_report: dict[str, object]) -> dict[str, dict[int, str]]:
    source_units = check_report.get("units")
    if not isinstance(source_units, list):
        raise Step3Error("Step 1 check report has no units list.")

    scans_by_unit: dict[str, dict[int, str]] = {}
    for source_unit in source_units:
        if not isinstance(source_unit, dict):
            raise Step3Error("Step 1 check report contains an invalid unit entry.")
        unit_id = source_unit.get("document_unit_id")
        files = source_unit.get("files")
        if not isinstance(unit_id, str) or not isinstance(files, list):
            raise Step3Error("Step 1 check report contains invalid unit data.")
        page_files: dict[int, str] = {}
        for relative_file in files:
            if not isinstance(relative_file, str):
                raise Step3Error(f"Invalid scan path for unit {unit_id}.")
            page_match = PAGE_FILENAME_PATTERN.search(relative_file)
            if page_match is None:
                raise Step3Error(
                    f"Invalid scan filename in Step 1 report: {relative_file}"
                )
            page_number = int(page_match.group("page"))
            if page_number in page_files:
                raise Step3Error(
                    f"Duplicate page {page_number:05d} in Step 1 report for "
                    f"{unit_id}."
                )
            page_files[page_number] = relative_file
        scans_by_unit[unit_id] = page_files
    return scans_by_unit


def _document_pages(
    split_report: dict[str, object],
    document_id: str,
) -> tuple[str, list[int]]:
    units = split_report.get("units")
    if not isinstance(units, list):
        raise Step3Error("Step 2 split report has no units list.")

    for unit in units:
        if not isinstance(unit, dict):
            raise Step3Error("Step 2 split report contains an invalid unit entry.")
        unit_id = unit.get("unit_id")
        status = unit.get("status")
        if not isinstance(unit_id, str):
            raise Step3Error("Step 2 split report contains an invalid unit ID.")
        if status != "pass":
            continue
        documents = unit.get("documents")
        if not isinstance(documents, list):
            raise Step3Error(f"Step 2 report has no document list for {unit_id}.")
        for document in documents:
            if not isinstance(document, dict):
                raise Step3Error(f"Invalid document entry for {unit_id}.")
            if document.get("id") != document_id:
                continue
            pages = document.get("pages")
            if (
                not isinstance(pages, list)
                or not pages
                or not all(type(page) is int and page > 0 for page in pages)
                or pages != sorted(set(pages))
            ):
                raise Step3Error(
                    f"Invalid or empty page list for document {document_id}."
                )
            return unit_id, pages
    raise Step3Error(
        f"Document {document_id} was not found in a passing Step 2 report."
    )


def _measure_dark_pixel_share(image_path: Path) -> float:
    try:
        with Image.open(image_path) as image:
            image.seek(0)
            grayscale = image.convert("L")
            histogram = grayscale.histogram()
            pixel_count = grayscale.width * grayscale.height
    except (OSError, UnidentifiedImageError) as exc:
        raise Step3Error(f"Cannot read scan image {image_path}: {exc}") from exc

    if pixel_count == 0:
        raise Step3Error(f"Scan image has no pixels: {image_path}")
    dark_pixel_count = sum(histogram[:DARK_PIXEL_INTENSITY_CUTOFF])
    return dark_pixel_count / pixel_count


def flag_document_blank_pages(
    batch_root: Path,
    document_id: str,
    dark_pixel_share_threshold: float,
    split_report_path: Path | None = None,
    check_report_path: Path | None = None,
) -> BlankPageReport:
    """Measure each page belonging to one document from Step 1 and Step 2 outputs."""
    if not 0 <= dark_pixel_share_threshold <= 1:
        raise Step3Error("Dark-pixel-share threshold must be between 0 and 1.")

    batch_root = Path(batch_root).resolve()
    scan_folder = resolve_scan_folder(batch_root)
    split_report_path = Path(
        split_report_path
        or batch_root / "work" / "step-2-documents" / "split-report.json"
    ).resolve()
    check_report_path = Path(
        check_report_path
        or batch_root / "work" / "step-1-checks" / "batch-report.json"
    ).resolve()
    if not scan_folder.is_dir():
        raise Step3Error(f"Scan folder does not exist: {scan_folder}")

    split_report = _read_json(split_report_path, "Step 2 split report")
    check_report = _read_json(check_report_path, "Step 1 check report")
    unit_id, document_pages = _document_pages(split_report, document_id)
    scans_by_unit = _index_unit_scans(check_report)
    unit_scans = scans_by_unit.get(unit_id)
    if unit_scans is None:
        raise Step3Error(
            f"Unit {unit_id} was not found in the Step 1 check report."
        )

    flags: list[PageBlankFlag] = []
    for page_number in document_pages:
        relative_file = unit_scans.get(page_number)
        if relative_file is None:
            raise Step3Error(
                f"Page {page_number:05d} of {document_id} is missing from "
                "the Step 1 check report."
            )
        image_path = (scan_folder / relative_file).resolve()
        if not image_path.is_relative_to(scan_folder):
            raise Step3Error(
                f"Scan path in Step 1 report escapes raw folder: {relative_file}"
            )
        if not image_path.is_file():
            raise Step3Error(f"Scan file is missing: {image_path}")
        dark_pixel_share = _measure_dark_pixel_share(image_path)
        flags.append(
            PageBlankFlag(
                page_number=page_number,
                file=relative_file,
                is_blank=dark_pixel_share < dark_pixel_share_threshold,
                dark_pixel_share=dark_pixel_share,
                dark_pixel_share_threshold=dark_pixel_share_threshold,
            )
        )

    return BlankPageReport(
        schema_version=1,
        document_id=document_id,
        status="pass",
        dark_pixel_intensity_cutoff=DARK_PIXEL_INTENSITY_CUTOFF,
        dark_pixel_share_threshold=dark_pixel_share_threshold,
        pages=flags,
    )


def write_blank_page_report(report: BlankPageReport, output_path: Path) -> None:
    """Write blank flags as a standalone JSON step output."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(asdict(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
