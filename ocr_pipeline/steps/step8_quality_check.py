import csv
import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import yaml
from PIL import Image, UnidentifiedImageError


class Step8Error(Exception):
    """Raised when quality-check inputs cannot be processed."""


REQUIRED_METADATA = {"id", "title", "type", "language", "pages", "source", "ocr"}
PAGE_MARKER_PATTERN = re.compile(r"<!--\s*page\s+(\d+)(?:\s*,\s*printed\s+[^>]+)?\s*-->")
IMAGE_MARKDOWN_PATTERN = re.compile(r"!\[((?:\\.|[^\]])*)\]\(([^)]+)\)")
TOKEN_PATTERN = re.compile(r"[^\W_]+", re.UNICODE)


@dataclass(frozen=True)
class QualityFlag:
    document_id: str
    page_number: int | None
    flag_type: str
    model: str
    model_version: str
    run_id: str
    details: str


@dataclass
class QualityReport:
    schema_version: int
    document_id: str
    run_id: str
    bundle_status: Literal["pass", "fail"]
    checks: list[str]
    errors: list[str]
    page_flags: list[QualityFlag]


def _read_utf8(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise Step8Error(f"Cannot read UTF-8 file {path}: {exc}") from exc


def _read_json(path: Path, label: str) -> dict[str, object]:
    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        raise Step8Error(f"Cannot read {label} {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise Step8Error(f"{label} must contain a JSON object.")
    return data


def _frontmatter(markdown: str) -> tuple[dict[str, object], str]:
    lines = markdown.splitlines()
    if not lines or lines[0].strip() != "---":
        raise Step8Error("document.md must start with YAML frontmatter.")
    try:
        closing = lines.index("---", 1)
    except ValueError as exc:
        raise Step8Error("YAML frontmatter has no closing delimiter.") from exc
    yaml_text = "\n".join(lines[1:closing])
    try:
        metadata = yaml.safe_load(yaml_text)
    except yaml.YAMLError as exc:
        raise Step8Error(f"Invalid YAML frontmatter: {exc}") from exc
    if not isinstance(metadata, dict):
        raise Step8Error("YAML frontmatter must be a mapping.")
    return metadata, "\n".join(lines[closing + 1 :])


def _check_bundle(bundle_folder: Path, document_id: str) -> tuple[list[str], list[str], dict[str, object]]:
    checks: list[str] = []
    errors: list[str] = []
    markdown_path = bundle_folder / "document.md"
    if not markdown_path.is_file():
        return checks, ["document.md is missing."], {}
    try:
        markdown = _read_utf8(markdown_path)
    except Step8Error as exc:
        return checks, [str(exc)], {}
    checks.append("utf8")
    try:
        metadata, body = _frontmatter(markdown)
    except Step8Error as exc:
        return checks, [str(exc)], {}
    checks.append("yaml_frontmatter")
    missing = REQUIRED_METADATA - set(metadata)
    if missing:
        errors.append(f"Missing metadata: {', '.join(sorted(missing))}.")
    if metadata.get("id") != document_id:
        errors.append("Frontmatter id must match the bundle folder name.")
    if not isinstance(metadata.get("title"), str):
        errors.append("Metadata title must be a string.")
    if not isinstance(metadata.get("type"), str) or not metadata.get("type"):
        errors.append("Metadata type must be a non-empty string.")
    if not isinstance(metadata.get("language"), str) or not metadata.get("language"):
        errors.append("Metadata language must be a non-empty string.")
    if not isinstance(metadata.get("ocr"), str) or not metadata.get("ocr"):
        errors.append("Metadata ocr must be a non-empty string.")
    pages = metadata.get("pages")
    valid_page_range = False
    expected_pages: list[int] = []
    if isinstance(pages, list) and len(pages) == 2:
        first_page, last_page = pages
        if (
            type(first_page) is int
            and type(last_page) is int
            and first_page > 0
            and last_page >= first_page
        ):
            expected_pages = list(range(first_page, last_page + 1))
            valid_page_range = True
    if not valid_page_range:
        errors.append("Metadata pages must contain positive [first_page, last_page].")
    sources = metadata.get("source")
    source_list: list[str] = []
    if isinstance(sources, list):
        source_list = [source for source in sources if isinstance(source, str)]
    expected_source_count = 1 if len(expected_pages) == 1 else 2
    if (
        not isinstance(sources, list)
        or len(sources) != expected_source_count
        or len(source_list) != expected_source_count
        or not all(source for source in source_list)
    ):
        errors.append("Metadata source must list the first and last source TIFF.")
    elif expected_pages:
        source_pages = (
            [expected_pages[0]]
            if len(expected_pages) == 1
            else [expected_pages[0], expected_pages[-1]]
        )
        for source, source_page in zip(source_list, source_pages):
            source_match = re.search(r"-(\d{5})\.tif$", source, re.IGNORECASE)
            if source_match is None or int(source_match.group(1)) != source_page:
                errors.append(
                    f"Metadata source does not match scan page {source_page:05d}."
                )
    if expected_pages:
        marker_pages = [int(value) for value in PAGE_MARKER_PATTERN.findall(body)]
        if marker_pages != expected_pages:
            errors.append(
                "Page markers must exactly match every scan page in the frontmatter range."
            )
    checks.append("required_metadata_and_page_markers")
    if unicodedata.normalize("NFC", markdown) != markdown:
        errors.append("document.md must be normalized to Unicode NFC.")
    checks.append("unicode_nfc")
    seen_images: set[str] = set()
    for escaped_alt, image_reference in IMAGE_MARKDOWN_PATTERN.findall(body):
        alt_text = re.sub(r"\\(.)", r"\1", escaped_alt)
        if not alt_text.strip():
            errors.append(f"Image link has empty alt text: {image_reference}")
            continue
        if (
            image_reference.startswith(("/", "\\"))
            or re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", image_reference)
            or "?" in image_reference
            or "#" in image_reference
        ):
            errors.append(f"Image link must be a local relative path: {image_reference}")
            continue
        relative = Path(image_reference)
        if ".." in relative.parts:
            errors.append(f"Image link escapes its bundle: {image_reference}")
            continue
        image_path = (bundle_folder / relative).resolve()
        if not image_path.is_relative_to(bundle_folder.resolve()):
            errors.append(f"Image link escapes its bundle: {image_reference}")
            continue
        if not image_path.is_file():
            errors.append(f"Linked image is missing: {image_reference}")
            continue
        try:
            with Image.open(image_path) as image:
                image.verify()
        except (OSError, UnidentifiedImageError) as exc:
            errors.append(f"Linked image is unreadable ({image_reference}): {exc}")
            continue
        if unicodedata.normalize("NFC", alt_text) != alt_text:
            errors.append(f"Image alt text is not Unicode NFC: {image_reference}")
        seen_images.add(image_reference)
    if re.search(r"!\[[^\]]*\]\(", body) and not seen_images and not errors:
        errors.append("Markdown contains invalid or unreadable image links.")
    checks.append("image_links_and_files")
    return checks, errors, metadata


def _page_flags(
    document_id: str,
    ocr_report_path: Path | None,
    vocabulary_path: Path | None,
    vocabulary_threshold: float | None,
    run_id: str,
) -> tuple[str, str, list[QualityFlag]]:
    if vocabulary_threshold is not None and not 0 <= vocabulary_threshold <= 1:
        raise Step8Error("Vocabulary threshold must be between 0 and 1.")
    if vocabulary_threshold is not None and vocabulary_path is None:
        raise Step8Error("A vocabulary file is required with a vocabulary threshold.")
    if vocabulary_path is not None:
        vocabulary_words = {
            token.casefold()
            for token in TOKEN_PATTERN.findall(_read_utf8(vocabulary_path))
        }
        if not vocabulary_words:
            raise Step8Error("Vocabulary file contains no usable words.")
    else:
        vocabulary_words = set()
    if ocr_report_path is None or not ocr_report_path.is_file():
        return "", "", [
            QualityFlag(
                document_id,
                None,
                "OCR_REPORT_MISSING",
                "",
                "",
                run_id,
                "Step 4 OCR report is missing.",
            )
        ]
    report = _read_json(ocr_report_path, "Step 4 OCR report")
    if report.get("document_id") != document_id:
        raise Step8Error("Step 4 report belongs to another document.")
    model = report.get("model")
    version = report.get("model_version")
    pages = report.get("pages")
    if not isinstance(model, str) or not isinstance(version, str) or not isinstance(pages, list):
        raise Step8Error("Step 4 OCR report has invalid model metadata or pages.")
    extracted: list[tuple[int, str, str, str | None]] = []
    flags: list[QualityFlag] = []
    for page in pages:
        if not isinstance(page, dict) or type(page.get("page_number")) is not int:
            raise Step8Error("Step 4 OCR report contains an invalid page.")
        page_number = page["page_number"]
        status = page.get("status")
        if status == "failed":
            error = page.get("error")
            error_text = error if isinstance(error, str) else ""
            flag_type = "TIMEOUT" if "timeout" in error_text.casefold() or "timed out" in error_text.casefold() else "OCR_FAILURE"
            flags.append(
                QualityFlag(document_id, page_number, flag_type, model, version, run_id, error_text)
            )
            continue
        if status == "blank":
            continue
        if status != "success":
            flags.append(
                QualityFlag(document_id, page_number, "OCR_FAILURE", model, version, run_id, f"Unexpected OCR status: {status}")
            )
            continue
        markdown_file = page.get("markdown_file")
        if not isinstance(markdown_file, str) or Path(markdown_file).is_absolute() or ".." in Path(markdown_file).parts:
            raise Step8Error(f"Invalid OCR Markdown path for page {page_number}.")
        page_path = Path(ocr_report_path).parent / markdown_file
        text = _read_utf8(page_path).strip()
        if not text:
            flags.append(
                QualityFlag(document_id, page_number, "EMPTY_OCR", model, version, run_id, "Non-blank page has empty OCR text.")
            )
        extracted.append((page_number, text, model, version))
        if vocabulary_threshold is not None and text:
            tokens = [token.casefold() for token in TOKEN_PATTERN.findall(text)]
            ratio = sum(token in vocabulary_words for token in tokens) / len(tokens) if tokens else 0.0
            if tokens and ratio < vocabulary_threshold:
                flags.append(
                    QualityFlag(document_id, page_number, "LOW_VOCABULARY", model, version, run_id, f"Vocabulary ratio {ratio:.4f} is below {vocabulary_threshold:.4f}.")
                )
    normalized_text: dict[str, list[int]] = {}
    for page_number, text, _, _ in extracted:
        normalized = " ".join(text.casefold().split())
        if len(normalized) >= 30:
            normalized_text.setdefault(normalized, []).append(page_number)
    for page_numbers in normalized_text.values():
        if len(page_numbers) > 1:
            for page_number in page_numbers:
                flags.append(
                    QualityFlag(document_id, page_number, "REPEATED_OCR", model, version, run_id, f"Same OCR text occurs on pages {', '.join(map(str, page_numbers))}.")
                )
    return model, version, flags


def _append_flag_log(flags: list[QualityFlag], output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not output_path.exists() or output_path.stat().st_size == 0
    with output_path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "document_id",
                "page_number",
                "flag_type",
                "model",
                "model_version",
                "run_id",
                "details",
            ],
        )
        if needs_header:
            writer.writeheader()
        for flag in flags:
            writer.writerow(
                {
                    **asdict(flag),
                    "page_number": "" if flag.page_number is None else flag.page_number,
                }
            )


def check_document_quality(
    bundle_folder: Path,
    document_id: str,
    ocr_report_path: Path | None = None,
    vocabulary_path: Path | None = None,
    vocabulary_threshold: float | None = None,
    run_id: str = "manual",
    flag_log_path: Path | None = None,
) -> QualityReport:
    bundle_folder = Path(bundle_folder).resolve()
    if bundle_folder.name != document_id:
        raise Step8Error("Bundle folder name must match document ID.")
    checks, errors, _ = _check_bundle(bundle_folder, document_id)
    if ocr_report_path is not None and not Path(ocr_report_path).is_absolute():
        ocr_report_path = bundle_folder / ocr_report_path
    if ocr_report_path is None:
        candidate = bundle_folder.parent.parent / "step-4-ocr" / document_id / "ocr-report.json"
        ocr_report_path = candidate if candidate.is_file() else None
    model, version, page_flags = _page_flags(
        document_id,
        ocr_report_path,
        vocabulary_path,
        vocabulary_threshold,
        run_id,
    )
    if errors:
        page_flags.append(
            QualityFlag(
                document_id,
                None,
                "BUNDLE_INVALID",
                model,
                version,
                run_id,
                "; ".join(errors),
            )
        )
    if flag_log_path is not None:
        _append_flag_log(page_flags, flag_log_path)
    return QualityReport(
        schema_version=1,
        document_id=document_id,
        run_id=run_id,
        bundle_status="pass" if not errors else "fail",
        checks=checks,
        errors=errors,
        page_flags=page_flags,
    )


def write_quality_report(report: QualityReport, output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(asdict(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
