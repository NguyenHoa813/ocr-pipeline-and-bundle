import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Protocol

from PIL import Image, UnidentifiedImageError

from ocr_pipeline.steps.step1_check_batch import resolve_scan_folder


class Step2Error(Exception):
    """Raised when Step 2 inputs cannot be processed."""


class BarcodeDetector(Protocol):
    def contains_barcode(self, image_path: Path) -> bool:
        """Return whether an image contains at least one decodable barcode."""
        ...


class ZXingBarcodeDetector:
    """Detect any supported barcode format with ZXing-C++."""

    def contains_barcode(self, image_path: Path) -> bool:
        try:
            import zxingcpp
        except ImportError as exc:
            raise Step2Error(
                "Barcode detection requires zxing-cpp; install project "
                "dependencies with 'python -m pip install -e .'."
            ) from exc

        try:
            with Image.open(image_path) as image:
                image.seek(0)
                rgb_image = image.convert("RGB")
                return bool(zxingcpp.read_barcodes(rgb_image))
        except (OSError, UnidentifiedImageError) as exc:
            raise Step2Error(f"Cannot read scan image {image_path}: {exc}") from exc


@dataclass(frozen=True)
class DocumentRecord:
    id: str
    first_page: int
    last_page: int
    pages: list[int]


@dataclass
class UnitSplitReport:
    unit_id: str
    status: Literal["pass", "fail", "skipped"]
    documents: list[DocumentRecord]
    separator_pages: list[int]
    error: str | None = None


@dataclass
class SplitReport:
    schema_version: int
    batch_folder: str
    source_check_report: str
    status: Literal["pass", "fail"]
    units: list[UnitSplitReport]


PAGE_FILENAME_PATTERN = re.compile(r"-(?P<page>\d{5})\.tif$", re.IGNORECASE)


def _read_check_report(path: Path) -> dict[str, object]:
    try:
        with path.open("r", encoding="utf-8") as file:
            report = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        raise Step2Error(f"Cannot read Step 1 check report {path}: {exc}") from exc
    if not isinstance(report, dict) or report.get("schema_version") != 1:
        raise Step2Error("Unsupported or invalid Step 1 check report.")
    if report.get("kind") not in ("folder", "book", "mixed"):
        raise Step2Error("Step 1 check report has an invalid unit kind.")
    if not isinstance(report.get("units"), list):
        raise Step2Error("Step 1 check report has no units list.")
    return report


def _split_pages(
    unit_id: str,
    pages: list[int],
    separator_pages: set[int],
) -> list[DocumentRecord]:
    documents: list[DocumentRecord] = []
    current_pages: list[int] = []

    def add_document() -> None:
        if not current_pages:
            return
        document_number = len(documents) + 1
        documents.append(
            DocumentRecord(
                id=f"{unit_id}-{document_number:02d}",
                first_page=current_pages[0],
                last_page=current_pages[-1],
                pages=current_pages.copy(),
            )
        )
        current_pages.clear()

    for page in pages:
        if page in separator_pages:
            add_document()
        else:
            current_pages.append(page)
    add_document()
    return documents


def split_checked_batch(
    batch_folder: Path,
    check_report_path: Path,
    detector: BarcodeDetector,
) -> SplitReport:
    """Split passing units from a Step 1 report into document page ranges."""
    batch_folder = resolve_scan_folder(batch_folder)
    check_report_path = Path(check_report_path).resolve()
    if not batch_folder.is_dir():
        raise Step2Error(f"Batch folder does not exist: {batch_folder}")

    check_report = _read_check_report(check_report_path)
    source_units = check_report["units"]
    assert isinstance(source_units, list)
    results: list[UnitSplitReport] = []

    for source_unit in source_units:
        if not isinstance(source_unit, dict):
            raise Step2Error("Step 1 check report contains an invalid unit entry.")
        unit_id = source_unit.get("document_unit_id")
        status = source_unit.get("status")
        kind = source_unit.get("kind")
        files = source_unit.get("files")
        page_numbers = source_unit.get("page_numbers")
        if (
            not isinstance(unit_id, str)
            or status not in ("pass", "fail")
            or kind not in ("folder", "book")
            or not isinstance(files, list)
            or not isinstance(page_numbers, list)
            or not all(isinstance(page, int) for page in page_numbers)
            or not all(isinstance(name, str) for name in files)
        ):
            raise Step2Error("Step 1 check report contains invalid unit data.")

        if status != "pass":
            results.append(
                UnitSplitReport(
                    unit_id=unit_id,
                    status="skipped",
                    documents=[],
                    separator_pages=[],
                    error="Skipped because Step 1 did not pass.",
                )
            )
            continue

        try:
            scans: list[tuple[int, Path]] = []
            for relative_file in files:
                page_match = PAGE_FILENAME_PATTERN.search(relative_file)
                if page_match is None:
                    raise Step2Error(
                        f"Invalid scan filename in Step 1 report: {relative_file}"
                    )
                image_path = (batch_folder / relative_file).resolve()
                if not image_path.is_relative_to(batch_folder):
                    raise Step2Error(
                        f"Scan path in check report escapes batch folder: "
                        f"{relative_file}"
                    )
                if not image_path.is_file():
                    raise Step2Error(
                        f"Scan file from check report is missing: {image_path}"
                    )
                scans.append((int(page_match.group("page")), image_path))

            scans.sort(key=lambda scan: scan[0])
            scan_pages = [page for page, _ in scans]
            if scan_pages != sorted(page_numbers):
                raise Step2Error(
                    f"Step 1 report file/page mismatch for {unit_id}."
                )

            separators: set[int] = set()
            if kind == "folder":
                detected_pages: set[int] = set()
                for page_number, image_path in scans:
                    if detector.contains_barcode(image_path):
                        detected_pages.add(page_number)
                available_pages = set(scan_pages)
                separator_sheets: set[int] = set()
                for page_number in detected_pages:
                    first_side = page_number if page_number % 2 == 1 else page_number - 1
                    separator_sheets.add(first_side)
                    for separator_side in (first_side, first_side + 1):
                        if separator_side in available_pages:
                            separators.add(separator_side)
                if 1 in separator_sheets:
                    raise Step2Error(
                        f"Separator at the start of {unit_id}; verify scanning."
                    )
                if (
                    separator_sheets
                    and max(separator_sheets) + 1 == max(scan_pages)
                ):
                    raise Step2Error(
                        f"Separator at the end of {unit_id}; verify scanning."
                    )
                ordered_sheets = sorted(separator_sheets)
                if any(
                    current == previous + 2
                    for previous, current in zip(ordered_sheets, ordered_sheets[1:])
                ):
                    raise Step2Error(
                        f"Consecutive separator sheets in {unit_id}; verify scanning."
                    )
            else:
                separators = set()

            documents = _split_pages(
                unit_id,
                page_numbers,
                separators,
            )
            if not documents:
                raise Step2Error(f"No document pages found for {unit_id}.")
            results.append(
                UnitSplitReport(
                    unit_id=unit_id,
                    status="pass",
                    documents=documents,
                    separator_pages=sorted(separators),
                )
            )
        except Step2Error as exc:
            results.append(
                UnitSplitReport(
                    unit_id=unit_id,
                    status="fail",
                    documents=[],
                    separator_pages=[],
                    error=str(exc),
                )
            )

    return SplitReport(
        schema_version=1,
        batch_folder=str(batch_folder),
        source_check_report=str(check_report_path),
        status=(
            "pass"
            if results and all(result.status in ("pass", "skipped") for result in results)
            else "fail"
        ),
        units=results,
    )


def write_split_report(report: SplitReport, output_folder: Path) -> None:
    """Write a batch summary and a per-unit document list."""
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    (output_folder / "split-report.json").write_text(
        json.dumps(asdict(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    for unit in report.units:
        safe_unit_id = unit.unit_id.replace("/", "_").replace("\\", "_")
        (output_folder / f"{safe_unit_id}-documents.json").write_text(
            json.dumps(asdict(unit), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
