import hashlib
import json
import re
import shutil
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from ocr_pipeline.steps.step1_check_batch import resolve_scan_folder


class Step6Error(Exception):
    """Raised when a document cannot be assembled from prior step outputs."""


DOCUMENT_ID_PATTERN = re.compile(r"[A-Za-z0-9]+-\d+-\d{2}")
PAGE_FILENAME_PATTERN = re.compile(r"-(?P<page>\d{5})\.tif$", re.IGNORECASE)
IMAGE_FILENAME_PATTERN = re.compile(
    r"^images/p(?P<page>\d{3,})-(?P<order>\d+)\.(?:png|jpg)$",
    re.IGNORECASE,
)
PRINTED_PAGE_PATTERN = re.compile(
    r"^\s*(?:page\s*)?(\d+[A-Za-z]?)\s*$",
    re.IGNORECASE,
)
MARKDOWN_BLOCK_PREFIX = re.compile(
    r"^(?:#{1,6}\s|>|[-*+]\s|\d+[.)]\s|```|~~~|<|!?\[|\|)"
)


@dataclass(frozen=True)
class AssemblyResult:
    document_id: str
    output_folder: Path
    pages: list[int]
    images: list[str]


def _read_json(path: Path, label: str) -> dict[str, object]:
    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        raise Step6Error(f"Cannot read {label} {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise Step6Error(f"{label} must contain a JSON object.")
    return data


def _contained_file(root: Path, relative: object, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise Step6Error(f"{label} must be a non-empty relative path.")
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise Step6Error(f"{label} escapes its output folder: {relative}")
    if not path.is_file():
        raise Step6Error(f"{label} does not exist: {path}")
    return path


def _read_utf8(path: Path, label: str) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise Step6Error(f"Cannot read {label} {path}: {exc}") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_spellcheck_output(
    spellcheck_folder: Path,
    ocr_report_path: Path,
    document_id: str,
    ocr_pages: dict[int, dict[str, object]],
) -> None:
    report = _read_json(
        spellcheck_folder / "spellcheck-report.json",
        "Step 4.1 spellcheck report",
    )
    if (
        report.get("schema_version") != 1
        or report.get("document_id") != document_id
        or report.get("status") != "pass"
    ):
        raise Step6Error(
            "Step 4.1 report is invalid, failed, or belongs to another document."
        )
    if report.get("source_ocr_report_sha256") != _sha256(ocr_report_path):
        raise Step6Error("Step 4.1 output is stale for the current Step 4 report.")
    log_path = _contained_file(
        spellcheck_folder,
        report.get("log_file"),
        "Step 4.1 correction log",
    )
    if report.get("correction_log_sha256") != _sha256(log_path):
        raise Step6Error("Step 4.1 correction log checksum is invalid.")
    source_pages = _read_json(
        spellcheck_folder / "source-pages.json",
        "Step 4.1 source-page checksums",
    )
    output_pages = report.get("output_page_sha256")
    if not isinstance(output_pages, dict):
        raise Step6Error("Step 4.1 output-page checksums are invalid.")
    page_paths: set[str] = set()
    for page in ocr_pages.values():
        relative = page.get("markdown_file")
        if not isinstance(relative, str):
            raise Step6Error("Step 4 report has invalid Markdown paths.")
        page_paths.add(relative)
    if set(source_pages) != page_paths:
        raise Step6Error("Step 4.1 source pages do not match Step 4.")
    if set(output_pages) != page_paths:
        raise Step6Error("Step 4.1 output pages do not match Step 4.")
    for page in ocr_pages.values():
        relative = page.get("markdown_file")
        if not isinstance(relative, str):
            raise Step6Error("Step 4 report has an invalid Markdown path.")
        source_hash = source_pages.get(relative)
        if not isinstance(source_hash, str):
            raise Step6Error("Step 4.1 source-page checksums are invalid.")
        source_markdown = _contained_file(
            ocr_report_path.parent,
            relative,
            "Step 4 OCR Markdown",
        )
        if source_hash != _sha256(source_markdown):
            raise Step6Error("Step 4.1 output is stale for the current OCR text.")
        corrected_markdown = _contained_file(
            spellcheck_folder,
            relative,
            "Step 4.1 corrected Markdown",
        )
        output_hash = output_pages.get(relative)
        if not isinstance(output_hash, str) or output_hash != _sha256(
            corrected_markdown
        ):
            raise Step6Error("Step 4.1 corrected Markdown checksum is invalid.")


def _is_plain_paragraph(block: str) -> bool:
    stripped = block.strip()
    if not stripped:
        return False
    return not any(
        MARKDOWN_BLOCK_PREFIX.match(line.lstrip())
        for line in stripped.splitlines()
    )


def _merge_page_continuations(
    pages: list[int],
    page_text: dict[int, str],
) -> dict[int, str]:
    """Conservatively join plain paragraphs split at adjacent scan pages."""
    merged = dict(page_text)
    tail_owners = {page: page for page in pages}
    sentence_endings = ".!?;:…。！？"
    for previous_page, page_number in zip(pages, pages[1:]):
        if page_number != previous_page + 1:
            continue
        owner = tail_owners[previous_page]
        previous_blocks = re.split(r"\n{2,}", merged[owner].strip())
        current_blocks = re.split(r"\n{2,}", merged[page_number].strip())
        if not previous_blocks[-1] or not current_blocks[0]:
            continue
        previous_tail = previous_blocks[-1].strip()
        current_head = current_blocks[0].strip()
        if (
            not _is_plain_paragraph(previous_tail)
            or not _is_plain_paragraph(current_head)
            or previous_tail[-1] in sentence_endings
            or not current_head[0].islower()
        ):
            continue
        joined_text = " ".join(
            part.strip()
            for part in (previous_tail, current_head)
            if part.strip()
        )
        previous_blocks[-1] = joined_text
        merged[owner] = "\n\n".join(previous_blocks)
        remaining_blocks = current_blocks[1:]
        merged[page_number] = "\n\n".join(remaining_blocks)
        tail_owners[page_number] = owner if not remaining_blocks else page_number
    return merged


def _load_document_pages(
    split_report_path: Path,
    document_id: str,
) -> tuple[str, list[int]]:
    report = _read_json(split_report_path, "Step 2 split report")
    if report.get("schema_version") != 1:
        raise Step6Error("Unsupported or invalid Step 2 split report.")
    units = report.get("units")
    if not isinstance(units, list):
        raise Step6Error("Step 2 split report has no units list.")
    for unit in units:
        if not isinstance(unit, dict) or unit.get("status") != "pass":
            continue
        unit_id = unit.get("unit_id")
        documents = unit.get("documents")
        if not isinstance(unit_id, str) or not isinstance(documents, list):
            raise Step6Error("Step 2 split report contains invalid unit data.")
        for document in documents:
            if not isinstance(document, dict) or document.get("id") != document_id:
                continue
            pages = document.get("pages")
            if (
                not isinstance(pages, list)
                or not pages
                or not all(type(page) is int and page > 0 for page in pages)
                or pages != sorted(set(pages))
            ):
                raise Step6Error(
                    f"Invalid or empty page list for document {document_id}."
                )
            return unit_id, pages
    raise Step6Error(
        f"Document {document_id} was not found in a passing Step 2 report."
    )


def _load_source_files(check_report_path: Path, unit_id: str) -> dict[int, str]:
    report = _read_json(check_report_path, "Step 1 check report")
    if report.get("schema_version") != 1:
        raise Step6Error("Unsupported or invalid Step 1 check report.")
    units = report.get("units")
    if not isinstance(units, list):
        raise Step6Error("Step 1 check report has no units list.")
    for unit in units:
        if not isinstance(unit, dict) or unit.get("document_unit_id") != unit_id:
            continue
        if unit.get("status") != "pass":
            raise Step6Error(f"Unit {unit_id} did not pass Step 1.")
        files = unit.get("files")
        if not isinstance(files, list):
            raise Step6Error(f"Step 1 report has no scan files for {unit_id}.")
        result: dict[int, str] = {}
        for relative_file in files:
            if not isinstance(relative_file, str):
                raise Step6Error(f"Invalid scan path for unit {unit_id}.")
            match = PAGE_FILENAME_PATTERN.search(relative_file)
            if match is None:
                raise Step6Error(
                    f"Invalid scan filename in Step 1 report: {relative_file}"
                )
            page_number = int(match.group("page"))
            if page_number in result:
                raise Step6Error(
                    f"Duplicate page {page_number:05d} in Step 1 report."
                )
            result[page_number] = relative_file
        return result
    raise Step6Error(f"Unit {unit_id} was not found in the Step 1 report.")


def _yaml_string(value: str) -> str:
    return json.dumps(unicodedata.normalize("NFC", value), ensure_ascii=False)


def _load_ocr_pages(
    ocr_report_path: Path,
    document_id: str,
    pages: list[int],
) -> tuple[str, str, dict[int, dict[str, object]]]:
    report = _read_json(ocr_report_path, "Step 4 OCR report")
    if (
        report.get("schema_version") != 1
        or report.get("document_id") != document_id
    ):
        raise Step6Error("Step 4 report is invalid or belongs to another document.")
    report_status = report.get("status")
    if report_status not in {"pass", "fail"}:
        raise Step6Error("Step 4 report has an invalid overall status.")
    model = report.get("model")
    model_version = report.get("model_version")
    page_data = report.get("pages")
    if (
        not isinstance(model, str)
        or not model.strip()
        or not isinstance(model_version, str)
        or not model_version.strip()
        or not isinstance(page_data, list)
    ):
        raise Step6Error("Step 4 report has invalid model metadata or pages.")
    result: dict[int, dict[str, object]] = {}
    for page in page_data:
        if not isinstance(page, dict):
            raise Step6Error("Step 4 report contains an invalid page.")
        page_number = page.get("page_number")
        if type(page_number) is not int or page_number <= 0:
            raise Step6Error("Step 4 report contains an invalid page number.")
        if page_number in result:
            raise Step6Error(f"Duplicate page {page_number:05d} in Step 4 report.")
        if page.get("status") not in {"success", "blank", "failed"}:
            raise Step6Error(
                f"Page {page_number:05d} has no successful OCR output."
            )
        result[page_number] = page
    has_failed_pages = any(page.get("status") == "failed" for page in result.values())
    if (report_status == "fail") != has_failed_pages:
        raise Step6Error(
            "Step 4 report status does not match its per-page OCR statuses."
        )
    if set(result) != set(pages):
        raise Step6Error("Step 4 OCR pages do not match the Step 2 document pages.")
    return model, model_version, result


def _load_image_records(
    cut_out_report_path: Path,
    document_id: str,
    step5_folder: Path,
    ocr_output_folder: Path,
    pages: list[int],
) -> list[tuple[int, str, Path, str]]:
    report = _read_json(cut_out_report_path, "Step 5 cut-out report")
    if (
        report.get("schema_version") != 1
        or report.get("document_id") != document_id
    ):
        raise Step6Error("Step 5 report is invalid or belongs to another document.")
    if report.get("status") != "pass":
        raise Step6Error("Step 5 report failed; rerun image extraction.")
    images = report.get("images")
    if not isinstance(images, list):
        raise Step6Error("Step 5 report has no images list.")
    results: list[tuple[int, str, Path, str]] = []
    seen_files: set[str] = set()
    for image in images:
        if not isinstance(image, dict):
            raise Step6Error("Step 5 report contains an invalid image record.")
        page_number = image.get("page_number")
        if type(page_number) is not int or page_number not in pages:
            raise Step6Error("Step 5 report contains an invalid image page.")
        relative_file = image.get("file")
        if (
            not isinstance(relative_file, str)
            or Path(relative_file).is_absolute()
            or ".." in Path(relative_file).parts
            or Path(relative_file).parts[:1] != ("images",)
        ):
            raise Step6Error("Step 5 image path must be inside its images folder.")
        image_match = IMAGE_FILENAME_PATTERN.fullmatch(relative_file)
        order = image.get("order_on_page")
        if (
            image_match is None
            or int(image_match.group("page")) != page_number
            or type(order) is not int
            or order <= 0
            or int(image_match.group("order")) != order
        ):
            raise Step6Error("Step 5 image record has an invalid page or order.")
        if relative_file in seen_files:
            raise Step6Error(f"Duplicate image path in Step 5 report: {relative_file}")
        seen_files.add(relative_file)
        source = _contained_file(step5_folder, relative_file, "Step 5 image")
        alt_text = ""
        page_json = (
            ocr_output_folder / "pages" / f"page-{page_number:05d}.json"
        )
        page_regions = _read_json(page_json, "Step 4 page regions")
        regions = page_regions.get("regions")
        if isinstance(regions, list) and type(order) is int and order > 0:
            figures = [
                region
                for region in regions
                if isinstance(region, dict) and region.get("type") == "figure"
            ]
            if order <= len(figures) and isinstance(figures[order - 1].get("text"), str):
                alt_text = figures[order - 1]["text"].strip()
        if not alt_text:
            alt_text = f"Figure from page {page_number}"
        results.append((page_number, relative_file, source, alt_text))
    return results


def _load_descriptions(
    report_path: Path | None,
    document_id: str,
) -> dict[str, tuple[str, str | None]]:
    if report_path is None or not report_path.is_file():
        return {}
    report = _read_json(report_path, "Step 7 image-description report")
    if (
        report.get("schema_version") != 1
        or report.get("document_id") != document_id
        or report.get("status") != "pass"
    ):
        raise Step6Error(
            "Step 7 report is invalid, failed, or belongs to another document."
        )
    images = report.get("images")
    if not isinstance(images, list):
        raise Step6Error("Step 7 report has no images list.")
    descriptions: dict[str, tuple[str, str | None]] = {}
    for image in images:
        if (
            not isinstance(image, dict)
            or image.get("status") != "success"
            or not isinstance(image.get("file"), str)
            or not isinstance(image.get("alt_text"), str)
            or not image["alt_text"].strip()
        ):
            raise Step6Error("Step 7 report contains an invalid image description.")
        mermaid = image.get("mermaid")
        if mermaid is not None and not isinstance(mermaid, str):
            raise Step6Error("Step 7 Mermaid output must be a string or null.")
        descriptions[image["file"]] = (
            image["alt_text"].strip(),
            mermaid.strip() if mermaid else None,
        )
    return descriptions


def _load_page_regions(
    ocr_output_folder: Path,
    pages: list[int],
) -> dict[int, list[dict[str, object]]]:
    result: dict[int, list[dict[str, object]]] = {}
    for page_number in pages:
        page_data = _read_json(
            ocr_output_folder / "pages" / f"page-{page_number:05d}.json",
            f"Page {page_number:05d} OCR regions",
        )
        if (
            page_data.get("page_number") != page_number
            or page_data.get("status") not in {"success", "blank", "failed"}
            or not isinstance(page_data.get("regions"), list)
        ):
            raise Step6Error(f"Page {page_number:05d} has invalid OCR regions.")
        regions_data = page_data.get("regions")
        if not isinstance(regions_data, list):
            raise Step6Error(f"Page {page_number:05d} has invalid OCR regions.")
        regions: list[dict[str, object]] = []
        for region in regions_data:
            if not isinstance(region, dict):
                raise Step6Error(f"Page {page_number:05d} has an invalid OCR region.")
            regions.append(region)
        result[page_number] = regions
    return result


def _strip_repeated_margins(
    page_text: dict[int, str],
    page_regions: dict[int, list[dict[str, object]]],
) -> dict[int, str]:
    occurrences: dict[str, set[int]] = {}
    for page_number, regions in page_regions.items():
        for region in regions:
            if region.get("type") not in {"page_header", "page_footer"}:
                continue
            text = region.get("text")
            if isinstance(text, str) and text.strip():
                normalized = unicodedata.normalize("NFC", text.strip())
                occurrences.setdefault(normalized, set()).add(page_number)
    repeated = {text for text, found_pages in occurrences.items() if len(found_pages) > 1}
    result: dict[int, str] = {}
    for page_number, markdown in page_text.items():
        lines = markdown.splitlines()
        result[page_number] = "\n".join(
            line
            for line in lines
            if unicodedata.normalize("NFC", line.strip()) not in repeated
        ).strip()
    return result


def _printed_page_text(regions: list[dict[str, object]]) -> str | None:
    for region in regions:
        if region.get("type") != "page_number":
            continue
        text = region.get("text")
        if not isinstance(text, str):
            continue
        match = PRINTED_PAGE_PATTERN.fullmatch(text)
        if match:
            return match.group(1)
    return None


def assemble_document(
    batch_root: Path,
    document_id: str,
    split_report_path: Path | None = None,
    check_report_path: Path | None = None,
    ocr_output_folder: Path | None = None,
    image_output_folder: Path | None = None,
    output_folder: Path | None = None,
    description_report_path: Path | None = None,
    allow_missing_descriptions: bool = False,
    spellcheck_output_folder: Path | None = None,
) -> AssemblyResult:
    """Create a portable Markdown-and-images bundle from pipeline outputs."""
    if not DOCUMENT_ID_PATTERN.fullmatch(document_id):
        raise Step6Error(f"Invalid document ID: {document_id}")
    batch_root = Path(batch_root).resolve()
    scan_folder = resolve_scan_folder(batch_root)
    if not scan_folder.is_dir():
        raise Step6Error(f"Scan folder does not exist: {scan_folder}")

    work_folder = batch_root / "work"
    split_report_path = Path(
        split_report_path
        or work_folder / "step-2-documents" / "split-report.json"
    ).resolve()
    check_report_path = Path(
        check_report_path
        or work_folder / "step-1-checks" / "batch-report.json"
    ).resolve()
    ocr_output_folder = Path(
        ocr_output_folder or work_folder / "step-4-ocr" / document_id
    ).resolve()
    if spellcheck_output_folder is None:
        default_spellcheck_folder = (
            work_folder / "step-4.1-spellcheck" / document_id
        ).resolve()
        spellcheck_output_folder = (
            default_spellcheck_folder
            if default_spellcheck_folder.exists()
            else None
        )
    else:
        spellcheck_output_folder = Path(spellcheck_output_folder).resolve()
    image_output_folder = Path(
        image_output_folder or work_folder / "step-5-images" / document_id
    ).resolve()
    output_folder = Path(
        output_folder or work_folder / "step-6-assemble" / document_id
    ).resolve()
    description_report_path = Path(
        description_report_path
        or work_folder
        / "step-7-image-descriptions"
        / f"{document_id}.json"
    ).resolve()
    if output_folder.is_relative_to(scan_folder):
        raise Step6Error("Assembled bundle output must be outside raw scans.")

    unit_id, pages = _load_document_pages(split_report_path, document_id)
    scan_paths = _load_source_files(check_report_path, unit_id)
    model, model_version, ocr_pages = _load_ocr_pages(
        ocr_output_folder / "ocr-report.json",
        document_id,
        pages,
    )
    if spellcheck_output_folder is not None:
        _validate_spellcheck_output(
            spellcheck_output_folder,
            ocr_output_folder / "ocr-report.json",
            document_id,
            ocr_pages,
        )
    image_records = _load_image_records(
        image_output_folder / "cut-out-report.json",
        document_id,
        image_output_folder,
        ocr_output_folder,
        pages,
    )
    has_description_report = description_report_path.is_file()
    if (
        image_records
        and not has_description_report
        and not allow_missing_descriptions
    ):
        raise Step6Error(
            "Step 7 image descriptions are required for every extracted figure."
        )
    descriptions = _load_descriptions(description_report_path, document_id)
    page_regions = _load_page_regions(ocr_output_folder, pages)
    if has_description_report and set(descriptions) != {
        image[1] for image in image_records
    }:
        raise Step6Error("Step 7 image list does not match Step 5 image crops.")

    page_text: dict[int, str] = {}
    for page_number in pages:
        relative_scan = scan_paths.get(page_number)
        if relative_scan is None:
            raise Step6Error(
                f"Page {page_number:05d} is missing from the Step 1 report."
            )
        scan_path = (scan_folder / relative_scan).resolve()
        if not scan_path.is_relative_to(scan_folder) or not scan_path.is_file():
            raise Step6Error(f"Source scan is missing or invalid: {relative_scan}")
        page = ocr_pages[page_number]
        markdown_path = _contained_file(
            spellcheck_output_folder or ocr_output_folder,
            page.get("markdown_file"),
            (
                f"Page {page_number:05d} Step 4.1 corrected Markdown"
                if spellcheck_output_folder is not None
                else f"Page {page_number:05d} OCR Markdown"
            ),
        )
        page_text[page_number] = unicodedata.normalize(
            "NFC",
            _read_utf8(markdown_path, f"Page {page_number:05d} OCR Markdown").strip(),
        )
    page_text = _strip_repeated_margins(page_text, page_regions)
    page_text = _merge_page_continuations(pages, page_text)
    document_title = next(
        (
            unicodedata.normalize("NFC", heading_match.group(1).strip())
            for page_number in pages
            for heading_match in [
                re.match(r"^#\s+(.+)$", page_text[page_number], re.MULTILINE)
            ]
            if heading_match is not None
        ),
        "",
    )

    source_pages = [pages[0]] if len(pages) == 1 else [pages[0], pages[-1]]
    frontmatter = [
        "---",
        f"id: {_yaml_string(document_id)}",
        f"title: {_yaml_string(document_title)}",
        'type: "unknown"',
        'language: "unknown"',
        f"pages: [{pages[0]}, {pages[-1]}]",
        "source: ["
        + ", ".join(
            _yaml_string(Path(scan_paths[page]).name)
            for page in source_pages
        )
        + "]",
        f"ocr: {_yaml_string(f'{model} {model_version}')}",
        "---",
        "",
    ]
    images_by_page: dict[int, list[tuple[str, str, str | None]]] = {}
    for page_number, relative_file, _, alt_text in image_records:
        described_alt, mermaid = descriptions.get(
            relative_file,
            (alt_text, None),
        )
        images_by_page.setdefault(page_number, []).append(
            (relative_file, described_alt, mermaid)
        )

    body = list(frontmatter)
    for page_number in pages:
        printed_page = _printed_page_text(page_regions[page_number])
        marker = (
            f"<!-- page {page_number}, printed {printed_page} -->"
            if printed_page is not None
            else f"<!-- page {page_number} -->"
        )
        body.append(marker)
        body.append("")
        if page_text[page_number]:
            body.append(page_text[page_number])
            body.append("")
        for relative_file, alt_text, mermaid in images_by_page.get(page_number, []):
            alt_text = unicodedata.normalize("NFC", alt_text)
            alt_text = alt_text.replace("\r", " ").replace("\n", " ")
            escaped_alt = (
                alt_text.replace("\\", "\\\\")
                .replace("[", "\\[")
                .replace("]", "\\]")
            )
            body.append(f"![{escaped_alt}]({relative_file})")
            body.append("")
            if mermaid:
                body.extend(("```mermaid", mermaid, "```", ""))

    output_images = output_folder / "images"
    output_images.mkdir(parents=True, exist_ok=True)
    copied_images: list[str] = []
    for _, relative_file, source, _ in image_records:
        destination = (output_folder / relative_file).resolve()
        if not destination.is_relative_to(output_folder):
            raise Step6Error(f"Image path escapes bundle output: {relative_file}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied_images.append(relative_file)

    output_folder.mkdir(parents=True, exist_ok=True)
    (output_folder / "document.md").write_text(
        "\n".join(body).rstrip() + "\n",
        encoding="utf-8",
    )
    return AssemblyResult(
        document_id=document_id,
        output_folder=output_folder,
        pages=pages,
        images=copied_images,
    )
