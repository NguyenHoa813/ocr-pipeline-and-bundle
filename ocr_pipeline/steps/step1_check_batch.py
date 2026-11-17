import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from ocr_pipeline.steps.batch_manifest import validate_batch_manifest

UnitKind = Literal["folder", "book"]

FILENAME_PATTERN = re.compile(
    r"(?P<box>[A-Za-z0-9]+)-(?P<unit>\d+)-(?P<page>\d{5})\.tif",
    re.IGNORECASE,
)
UNIT_PREFIX_PATTERN = re.compile(
    r"(?P<box>[A-Za-z0-9]+)-(?P<unit>\d+)-",
    re.IGNORECASE,
)


class BatchCheckError(Exception):
    """Raised when the batch or box-list input cannot be checked."""


@dataclass(frozen=True)
class CheckIssue:
    code: str
    message: str
    file: str | None = None
    page_number: int | None = None


@dataclass
class UnitReport:
    document_unit_id: str
    box_code: str
    unit_number: str
    kind: UnitKind
    status: Literal["pass", "fail"]
    page_count: int
    page_numbers: list[int]
    missing_pages: list[int]
    duplicate_pages: list[int]
    files: list[str]
    issues: list[CheckIssue]


@dataclass
class BatchReport:
    schema_version: int
    batch_folder: str
    kind: Literal["folder", "book", "mixed"]
    status: Literal["pass", "fail"]
    units: list[UnitReport]
    issues: list[CheckIssue]

    @property
    def passed_count(self) -> int:
        return sum(unit.status == "pass" for unit in self.units)

    @property
    def failed_count(self) -> int:
        return sum(unit.status == "fail" for unit in self.units)


def _read_box_codes(box_list_path: Path) -> set[str]:
    try:
        with box_list_path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            if reader.fieldnames is None or "box_code" not in reader.fieldnames:
                raise BatchCheckError(
                    "Box-list CSV must contain a 'box_code' column."
                )
            return {
                (row.get("box_code") or "").strip().upper()
                for row in reader
                if (row.get("box_code") or "").strip()
            }
    except UnicodeDecodeError as exc:
        raise BatchCheckError("Box-list CSV must be UTF-8 encoded.") from exc


def _scan_files(batch_folder: Path, excluded_file: Path) -> list[Path]:
    if not batch_folder.exists() or not batch_folder.is_dir():
        raise BatchCheckError(f"Batch folder does not exist: {batch_folder}")
    return sorted(
        (
            path
            for path in batch_folder.rglob("*")
            if path.is_file() and path.resolve() != excluded_file
        ),
        key=lambda path: path.as_posix().casefold(),
    )


def resolve_scan_folder(batch_root: Path) -> Path:
    """Resolve a batch root to its raw scan directory when it has one."""
    batch_root = Path(batch_root).resolve()
    raw_folder = batch_root / "raw"
    if raw_folder.is_dir():
        return raw_folder
    return batch_root


def check_batch(
    batch_folder: Path,
    box_list_path: Path,
    kind: UnitKind | None,
    manifest_path: Path | None = None,
) -> BatchReport:
    """Check all TIFF scans and return one report for each inferred unit."""
    batch_folder = Path(batch_folder).resolve()
    box_list_path = Path(box_list_path).resolve()
    if kind not in ("folder", "book", None):
        raise BatchCheckError("kind must be either 'folder' or 'book'.")
    if kind is None and manifest_path is None:
        raise BatchCheckError("Provide --kind or a batch manifest.")

    known_boxes = _read_box_codes(box_list_path)
    manifest_items = {}
    if manifest_path is not None:
        _, manifest_items = validate_batch_manifest(manifest_path)
    files = _scan_files(batch_folder, box_list_path)
    grouped_files: dict[tuple[str, str], list[tuple[Path, int | None]]] = {}
    unit_issues: dict[tuple[str, str], list[CheckIssue]] = {}
    batch_issues: list[CheckIssue] = []

    for path in files:
        relative_path = path.relative_to(batch_folder).as_posix()
        match = FILENAME_PATTERN.fullmatch(path.name)
        if match is None:
            prefix_match = UNIT_PREFIX_PATTERN.match(path.name)
            issue = CheckIssue(
                code="invalid_filename",
                message=(
                    "Scan filename must match "
                    "'<box-code>-<unit-number>-<5-digit-page>.tif'."
                ),
                file=relative_path,
            )
            if prefix_match is None:
                batch_issues.append(issue)
            else:
                key = (
                    prefix_match.group("box").upper(),
                    prefix_match.group("unit"),
                )
                unit_issues.setdefault(key, []).append(issue)
                grouped_files.setdefault(key, []).append((path, None))
            continue

        box_code = match.group("box").upper()
        unit_number = match.group("unit")
        page_number = int(match.group("page"))
        key = (box_code, unit_number)
        grouped_files.setdefault(key, []).append((path, page_number))
        relative_parts = path.relative_to(batch_folder).parts
        if batch_folder.name.casefold() == "raw":
            expected_parts = (box_code, unit_number, path.name)
            actual_parts = (
                relative_parts[0].upper() if len(relative_parts) > 0 else "",
                relative_parts[1] if len(relative_parts) > 1 else "",
                relative_parts[2] if len(relative_parts) > 2 else "",
            )
            if len(relative_parts) != 3 or (
                actual_parts[0] != expected_parts[0]
                or actual_parts[1] != expected_parts[1]
                or actual_parts[2] != expected_parts[2]
            ):
                unit_issues.setdefault(key, []).append(
                    CheckIssue(
                        code="invalid_scan_path",
                        message=(
                            "Scan must be stored as "
                            "'raw/<box-code>/<unit-number>/<filename>'."
                        ),
                        file=relative_path,
                    )
                )

    if not files:
        batch_issues.append(
            CheckIssue(
                code="no_tiff_files",
                message="No scan files were found in the batch folder.",
            )
        )

    units: list[UnitReport] = []
    for (box_code, unit_number), unit_files in sorted(grouped_files.items()):
        issues = list(unit_issues.get((box_code, unit_number), []))
        page_numbers = sorted(
            page for _, page in unit_files if page is not None
        )
        page_counts: dict[int, int] = {}
        for page_number in page_numbers:
            page_counts[page_number] = page_counts.get(page_number, 0) + 1
        duplicate_pages = sorted(
            page for page, count in page_counts.items() if count > 1
        )
        for page_number in duplicate_pages:
            issues.append(
                CheckIssue(
                    code="duplicate_page",
                    message=f"Page number {page_number:05d} occurs more than once.",
                    page_number=page_number,
                )
            )

        for path, page_number in unit_files:
            if page_number == 0:
                issues.append(
                    CheckIssue(
                        code="invalid_page_number",
                        message="Page numbering starts at 00001; 00000 is invalid.",
                        file=path.relative_to(batch_folder).as_posix(),
                        page_number=page_number,
                    )
                )

        missing_pages: list[int] = []
        if page_numbers:
            expected_pages = set(range(1, max(page_numbers) + 1))
            missing_pages = sorted(expected_pages - set(page_numbers))
            for page_number in missing_pages:
                issues.append(
                    CheckIssue(
                        code="page_gap",
                        message=f"Page number {page_number:05d} is missing.",
                        page_number=page_number,
                    )
                )

        if box_code not in known_boxes:
            issues.append(
                CheckIssue(
                    code="unknown_box_code",
                    message=f"Box code {box_code} is not present in the box list.",
                )
            )
        manifest_item = manifest_items.get((box_code, str(int(unit_number))))
        unit_kind = manifest_item.item_type if manifest_item else kind
        if unit_kind is None:
            issues.append(
                CheckIssue(
                    code="manifest_item_missing",
                    message=f"Unit {box_code}-{unit_number} is not defined in the manifest.",
                )
            )
            unit_kind = "folder"
        if unit_kind == "folder" and len(unit_files) % 2 != 0:
            issues.append(
                CheckIssue(
                    code="odd_page_count",
                    message=(
                        f"Folder has {len(unit_files)} TIFF file(s); "
                        "duplex folder scans must have an even page count."
                    ),
                )
            )

        units.append(
            UnitReport(
                document_unit_id=f"{box_code}-{unit_number}",
                box_code=box_code,
                unit_number=unit_number,
                kind=unit_kind,
                status="fail" if issues else "pass",
                page_count=len(unit_files),
                page_numbers=page_numbers,
                missing_pages=missing_pages,
                duplicate_pages=duplicate_pages,
                files=sorted(
                    path.relative_to(batch_folder).as_posix()
                    for path, _ in unit_files
                ),
                issues=issues,
            )
        )

    if batch_issues:
        for unit in units:
            unit.issues.extend(batch_issues)
            unit.status = "fail"

    if not units:
        batch_issues.append(
            CheckIssue(
                code="no_valid_units",
                message="No unit could be identified from the TIFF filenames.",
            )
        )

    if manifest_path is not None:
        page_counts_by_unit = {
            (unit.box_code, str(int(unit.unit_number))): unit.page_count
            for unit in units
        }
        manifest_report, _ = validate_batch_manifest(
            manifest_path,
            expected_kind=kind,
            unit_page_counts=page_counts_by_unit,
        )
        manifest_issues = [
            CheckIssue(code="manifest_mismatch", message=message)
            for message in manifest_report.issues
        ]
        if manifest_issues:
            batch_issues.extend(manifest_issues)
            for unit in units:
                unit.issues.extend(manifest_issues)
                unit.status = "fail"
    unit_kinds = {unit.kind for unit in units}
    report_kind: Literal["folder", "book", "mixed"] = "mixed"
    if len(unit_kinds) == 1:
        report_kind = units[0].kind
    return BatchReport(
        schema_version=1,
        batch_folder=str(batch_folder),
        kind=report_kind,
        status=(
            "pass"
            if units and all(unit.status == "pass" for unit in units)
            else "fail"
        ),
        units=units,
        issues=batch_issues,
    )


def write_reports(report: BatchReport, output_folder: Path) -> None:
    """Write a batch summary and a separate JSON report for each unit."""
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    batch_data = asdict(report)
    (output_folder / "batch-report.json").write_text(
        json.dumps(batch_data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    for unit in report.units:
        unit_path = output_folder / f"{unit.document_unit_id}.json"
        unit_path.write_text(
            json.dumps(asdict(unit), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
