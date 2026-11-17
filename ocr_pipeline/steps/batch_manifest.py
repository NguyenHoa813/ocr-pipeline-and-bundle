import json
import re
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Literal

import yaml


class ManifestError(Exception):
    """Raised when the batch manifest is malformed or inconsistent."""


@dataclass(frozen=True)
class ManifestItem:
    box_code: str
    item_number: str
    item_type: Literal["folder", "book"]
    total_pages: int
    damaged_pages: list[int]
    printed_pages: int | None


@dataclass
class BatchManifestReport:
    schema_version: int
    batch_id: str
    status: Literal["pass", "fail"]
    issues: list[str]
    damaged_pages: list[dict[str, object]]


def validate_batch_manifest(
    manifest_path: Path,
    expected_kind: str | None = None,
    unit_page_counts: dict[tuple[str, str], int] | None = None,
) -> tuple[BatchManifestReport, dict[tuple[str, str], ManifestItem]]:
    try:
        with Path(manifest_path).open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file)
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ManifestError(f"Cannot read batch manifest {manifest_path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ManifestError("Batch manifest must contain a YAML mapping.")
    batch_id = data.get("batch_id")
    operator = data.get("operator")
    scan = data.get("scan")
    raw_items = data.get("items")
    if not isinstance(batch_id, str) or not batch_id.strip():
        raise ManifestError("Manifest batch_id must be a non-empty string.")
    if not isinstance(operator, dict) or not isinstance(operator.get("name"), str) or not operator["name"].strip():
        raise ManifestError("Manifest operator.name must be a non-empty string.")
    if not isinstance(scan, dict):
        raise ManifestError("Manifest scan must be a mapping.")
    scan_date = scan.get("date")
    if isinstance(scan_date, date):
        scan_date_valid = True
    elif isinstance(scan_date, str):
        try:
            date.fromisoformat(scan_date)
            scan_date_valid = True
        except ValueError:
            scan_date_valid = False
    else:
        scan_date_valid = False
    if not scan_date_valid:
        raise ManifestError("Manifest scan.date must be an ISO date.")
    for key in ("scanner_model", "color_mode", "format"):
        if not isinstance(scan.get(key), str) or not scan[key].strip():
            raise ManifestError(f"Manifest scan.{key} must be a non-empty string.")
    if scan["format"].casefold() not in {"tif", "tiff"}:
        raise ManifestError("Manifest scan.format must be TIFF.")
    if type(scan.get("resolution_dpi")) is not int or scan["resolution_dpi"] <= 0:
        raise ManifestError("Manifest scan.resolution_dpi must be a positive integer.")
    if type(scan.get("duplex")) is not bool:
        raise ManifestError("Manifest scan.duplex must be a boolean.")
    if not isinstance(raw_items, list) or not raw_items:
        raise ManifestError("Manifest items must be a non-empty list.")

    items: dict[tuple[str, str], ManifestItem] = {}
    issues: list[str] = []
    damaged_pages: list[dict[str, object]] = []
    for index, raw_item in enumerate(raw_items):
        if not isinstance(raw_item, dict):
            raise ManifestError(f"Manifest item {index} must be a mapping.")
        box_code = raw_item.get("box_code")
        item_number = raw_item.get("item_number")
        item_type = raw_item.get("item_type")
        total_pages = raw_item.get("total_pages")
        damaged = raw_item.get("damaged_pages", [])
        printed = raw_item.get("printed_pages")
        if (
            not isinstance(box_code, str)
            or not re.fullmatch(r"[A-Za-z0-9]+", box_code)
            or not isinstance(item_number, str)
            or not re.fullmatch(r"\d+", item_number)
        ):
            raise ManifestError(f"Manifest item {index} has invalid box_code/item_number.")
        if item_type not in {"folder", "book"}:
            raise ManifestError(f"Manifest item {index} item_type must be folder or book.")
        if type(total_pages) is not int or total_pages <= 0:
            raise ManifestError(f"Manifest item {index} total_pages must be positive.")
        if not isinstance(damaged, list) or not all(
            type(page) is int and 1 <= page <= total_pages for page in damaged
        ) or len(damaged) != len(set(damaged)):
            raise ManifestError(f"Manifest item {index} damaged_pages are invalid.")
        if printed is not None and (type(printed) is not int or printed < 0):
            raise ManifestError(f"Manifest item {index} printed_pages must be non-negative.")
        key = (box_code.upper(), str(int(item_number)))
        if key in items:
            raise ManifestError(f"Duplicate manifest item: {box_code}-{item_number}.")
        normalized = ManifestItem(
            box_code=box_code.upper(),
            item_number=item_number,
            item_type=item_type,
            total_pages=total_pages,
            damaged_pages=sorted(damaged),
            printed_pages=printed,
        )
        items[key] = normalized
        if expected_kind is not None and item_type != expected_kind:
            issues.append(
                f"{box_code}-{item_number}: manifest type {item_type} does not match {expected_kind} check."
            )
        if unit_page_counts is not None:
            actual_count = unit_page_counts.get(key)
            if actual_count is None:
                issues.append(f"{box_code}-{item_number}: no matching scan unit was found.")
            elif actual_count != total_pages:
                issues.append(
                    f"{box_code}-{item_number}: manifest declares {total_pages} pages, found {actual_count}."
                )
        damaged_pages.extend(
            {
                "box_code": box_code.upper(),
                "item_number": str(int(item_number)),
                "page_number": page_number,
            }
            for page_number in normalized.damaged_pages
        )
    if unit_page_counts is not None:
        for box_code, item_number in sorted(set(unit_page_counts) - set(items)):
            issues.append(f"{box_code}-{item_number}: scan unit is missing from manifest.")
    report = BatchManifestReport(
        schema_version=1,
        batch_id=batch_id.strip(),
        status="pass" if not issues else "fail",
        issues=issues,
        damaged_pages=damaged_pages,
    )
    return report, items


def write_manifest_report(report: BatchManifestReport, output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(asdict(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
