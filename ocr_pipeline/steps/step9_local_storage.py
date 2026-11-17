import json
import os
import re
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal


class Step9Error(Exception):
    """Raised when a bundle cannot be safely published locally."""


SAFE_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


@dataclass(frozen=True)
class PublishResult:
    document_id: str
    status: Literal["published", "rejected"]
    destination: Path
    errors: list[str]


def publish_document_bundle(
    bundle_folder: Path,
    document_id: str,
    quality_report_path: Path,
    storage_root: Path,
    run_id: str,
) -> PublishResult:
    """Atomically publish valid bundles and quarantine structurally invalid ones."""
    for label, component in (("document ID", document_id), ("run ID", run_id)):
        if not SAFE_COMPONENT.fullmatch(component):
            raise Step9Error(f"Invalid {label}: {component}")
    bundle_folder = Path(bundle_folder).resolve()
    storage_root = Path(storage_root).resolve()
    if not bundle_folder.is_dir() or bundle_folder.name != document_id:
        raise Step9Error("Bundle folder must exist and match the document ID.")
    markdown_path = bundle_folder / "document.md"
    images_path = bundle_folder / "images"
    if not markdown_path.is_file() or not images_path.is_dir():
        raise Step9Error("Bundle must contain document.md and images/.")
    try:
        with Path(quality_report_path).open("r", encoding="utf-8") as file:
            quality_report = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        raise Step9Error(f"Cannot read quality report: {exc}") from exc
    if (
        not isinstance(quality_report, dict)
        or quality_report.get("schema_version") != 1
        or quality_report.get("document_id") != document_id
        or quality_report.get("run_id") != run_id
    ):
        raise Step9Error("Quality report is invalid or does not match this run.")
    errors = quality_report.get("errors")
    bundle_status = quality_report.get("bundle_status")
    if not isinstance(errors, list) or not all(isinstance(item, str) for item in errors):
        raise Step9Error("Quality report has invalid errors.")
    if bundle_status not in {"pass", "fail"}:
        raise Step9Error("Quality report has invalid bundle status.")
    category = "valid" if bundle_status == "pass" else "rejected"
    destination_root = storage_root / "output" / category
    destination_root.mkdir(parents=True, exist_ok=True)
    destination = destination_root / document_id
    if category == "rejected":
        destination = destination_root / f"{document_id}-{run_id}"
    if destination.exists():
        raise Step9Error(f"Destination already exists; refusing to overwrite: {destination}")
    staging_root = storage_root / "output" / ".staging"
    staging_root.mkdir(parents=True, exist_ok=True)
    staging_path = staging_root / f"{document_id}-{uuid.uuid4().hex}"
    try:
        staging_path.mkdir()
        shutil.copy2(markdown_path, staging_path / "document.md")
        shutil.copytree(images_path, staging_path / "images")
        os.replace(staging_path, destination)
    except OSError as exc:
        if staging_path.exists():
            shutil.rmtree(staging_path)
        raise Step9Error(f"Could not atomically publish bundle: {exc}") from exc

    logs_folder = storage_root / "logs"
    logs_folder.mkdir(parents=True, exist_ok=True)
    publish_record = {
        "schema_version": 1,
        "document_id": document_id,
        "run_id": run_id,
        "status": "published" if category == "valid" else "rejected",
        "destination": str(destination),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "page_flags": quality_report.get("page_flags", []),
        "errors": errors,
    }
    record_path = logs_folder / f"publish-{document_id}-{run_id}.json"
    record_path.write_text(
        json.dumps(publish_record, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return PublishResult(
        document_id=document_id,
        status="published" if category == "valid" else "rejected",
        destination=destination,
        errors=errors,
    )
