import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any

from ocr_pipeline.steps.spell_correction import (
    ACTION_AUTO_CORRECT,
    SpellDecision,
    correct_text,
    decision_dict,
    traditional_script_guard_available,
)
from ocr_pipeline.steps.spell_detection import (
    SpellDetectionError,
    SpellDetector,
    detect_language,
    load_dictionary,
    normalize_language,
)
from ocr_pipeline.steps.step1_check_batch import resolve_scan_folder


DOCUMENT_ID_PATTERN = re.compile(r"[A-Za-z0-9]+-\d+-\d{2}")


class Step41Error(Exception):
    """Raised when Step 4.1 inputs or outputs are invalid."""


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Step41Error(f"Cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise Step41Error(f"{label} must contain a JSON object.")
    return value


def _contained_file(root: Path, relative: object, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise Step41Error(f"{label} must be a non-empty relative path.")
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise Step41Error(f"{label} escapes its input folder: {relative}")
    if not path.is_file():
        raise Step41Error(f"{label} does not exist: {path}")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_report(
    report: dict[str, Any],
    document_id: str,
) -> list[dict[str, Any]]:
    model = report.get("model")
    model_version = report.get("model_version")
    if (
        report.get("schema_version") != 1
        or report.get("document_id") != document_id
        or report.get("status") not in {"pass", "fail"}
        or not isinstance(model, str)
        or not model.strip()
        or not isinstance(model_version, str)
        or not model_version.strip()
    ):
        raise Step41Error(
            "Step 4 report is invalid or belongs to another document."
        )
    pages = report.get("pages")
    if not isinstance(pages, list) or not pages:
        raise Step41Error("Step 4 report has no pages.")
    seen: set[int] = set()
    seen_markdown: set[str] = set()
    for page in pages:
        if not isinstance(page, dict):
            raise Step41Error("Step 4 report contains an invalid page.")
        page_number = page.get("page_number")
        status = page.get("status")
        if (
            type(page_number) is not int
            or page_number <= 0
            or page_number in seen
            or status not in {"success", "blank", "failed"}
        ):
            raise Step41Error("Step 4 report contains invalid page metadata.")
        seen.add(page_number)
        markdown_file = page.get("markdown_file")
        if not isinstance(markdown_file, str):
            raise Step41Error(
                f"Step 4 page {page_number:05d} has no Markdown path."
            )
        if markdown_file in seen_markdown:
            raise Step41Error(
                f"Duplicate Markdown path in Step 4 report: {markdown_file}"
            )
        seen_markdown.add(markdown_file)
    if (report.get("status") == "fail") != any(
        page["status"] == "failed" for page in pages
    ):
        raise Step41Error(
            "Step 4 report status does not match its per-page OCR statuses."
        )
    return pages


def _page_decision_record(
    page_number: int,
    decision: SpellDecision,
) -> dict[str, object]:
    return {"page_number": page_number, **decision_dict(decision)}


def spellcheck_document(
    batch_root: Path,
    document_id: str,
    ocr_output_folder: Path | None = None,
    output_folder: Path | None = None,
    language: str | None = None,
    dictionary_file: Path | None = None,
    confidence_threshold: float = 0.95,
) -> dict[str, object]:
    """Create a corrected text copy and an auditable decision log."""
    if not DOCUMENT_ID_PATTERN.fullmatch(document_id):
        raise Step41Error(f"Invalid document ID: {document_id}")
    if not 0.0 <= confidence_threshold <= 1.0:
        raise Step41Error("Confidence threshold must be between 0 and 1.")

    batch_root = Path(batch_root).resolve()
    scan_folder = resolve_scan_folder(batch_root)
    work_folder = batch_root / "work"
    source_folder = Path(
        ocr_output_folder or work_folder / "step-4-ocr" / document_id
    ).resolve()
    target_folder = Path(
        output_folder or work_folder / "step-4.1-spellcheck" / document_id
    ).resolve()
    if not source_folder.is_dir():
        raise Step41Error(f"Step 4 output folder does not exist: {source_folder}")
    if target_folder == source_folder or target_folder.is_relative_to(source_folder):
        raise Step41Error("Step 4.1 output must be separate from Step 4 output.")
    if target_folder.is_relative_to(scan_folder):
        raise Step41Error("Step 4.1 output must be outside raw scans.")

    report_path = source_folder / "ocr-report.json"
    ocr_report = _read_json(report_path, "Step 4 OCR report")
    pages = _validate_report(ocr_report, document_id)
    language_hint = normalize_language(language) or normalize_language(
        ocr_report.get("language")
    )
    if language is not None and language_hint is None:
        raise Step41Error(f"Unsupported language override: {language!r}")

    try:
        dictionary_words, domain_terms, context_phrases = load_dictionary(
            dictionary_file
        )
    except SpellDetectionError as exc:
        raise Step41Error(str(exc)) from exc
    detector = SpellDetector(dictionary_words, domain_terms, context_phrases)
    corrected_pages: list[tuple[str, str, bytes]] = []
    log_records: list[dict[str, object]] = []
    page_records: list[dict[str, object]] = []
    source_page_hashes: dict[str, str] = {}
    automatic_corrections = 0
    kept_decisions = 0
    warnings: list[str] = []
    languages_used: set[str] = set()

    for page in pages:
        page_number = page["page_number"]
        markdown_path = _contained_file(
            source_folder,
            page["markdown_file"],
            f"Page {page_number:05d} OCR Markdown",
        )
        page_json_path = _contained_file(
            source_folder,
            Path(page["markdown_file"]).with_suffix(".json").as_posix(),
            f"Page {page_number:05d} OCR layout",
        )
        source_text = markdown_path.read_text(encoding="utf-8")
        source_page_hashes[page["markdown_file"]] = _sha256(markdown_path)
        page_language = detect_language(source_text, language_hint)
        if page_language.language == "vi+zh-Hant":
            languages_used.update({"vi", "zh-Hant"})
        elif page_language.language is not None:
            languages_used.add(page_language.language)
        if page["status"] == "success":
            try:
                corrected_text, decisions = correct_text(
                    source_text,
                    page_language.language,
                    detector,
                    confidence_threshold,
                )
            except SpellDetectionError as exc:
                raise Step41Error(str(exc)) from exc
        else:
            corrected_text = source_text
            decisions = []
        corrected_pages.append(
            (page["markdown_file"], corrected_text, page_json_path.read_bytes())
        )
        automatic_corrections += sum(
            decision.action == ACTION_AUTO_CORRECT for decision in decisions
        )
        kept_decisions += sum(
            decision.action != ACTION_AUTO_CORRECT for decision in decisions
        )
        log_records.extend(
            _page_decision_record(page_number, decision) for decision in decisions
        )
        page_records.append(
            {
                "page_number": page_number,
                "status": page["status"],
                "language": page_language.language,
                "language_confidence": page_language.confidence,
                "language_reason": page_language.reason,
                "corrections": sum(
                    decision.action == ACTION_AUTO_CORRECT
                    for decision in decisions
                ),
                "kept": sum(
                    decision.action != ACTION_AUTO_CORRECT
                    for decision in decisions
                ),
            }
        )
        if page_language.language is None:
            warnings.append(
                f"Page {page_number:05d}: language could not be determined; "
                "text was kept unchanged."
            )

    warnings.extend(detector.warnings)
    if "zh-Hant" in languages_used and not traditional_script_guard_available():
        warnings.append(
            "zh-Hant: automatic corrections require the OpenCC script-safety "
            "guard; without it candidates are kept unchanged."
        )
    for selected_language in sorted(languages_used & {"vi", "zh-Hant"}):
        if not dictionary_words.get(selected_language) and not domain_terms.get(
            selected_language
        ):
            warnings.append(
                f"{selected_language}: no local vocabulary supplied; "
                "language-specific candidates cannot be generated."
            )
    target_folder.mkdir(parents=True, exist_ok=True)
    for relative_markdown, corrected_text, page_json in corrected_pages:
        markdown_target = (target_folder / relative_markdown).resolve()
        json_target = markdown_target.with_suffix(".json")
        if not markdown_target.is_relative_to(target_folder):
            raise Step41Error(
                f"Page output escapes Step 4.1 folder: {relative_markdown}"
            )
        markdown_target.parent.mkdir(parents=True, exist_ok=True)
        markdown_target.write_text(corrected_text, encoding="utf-8")
        json_target.write_bytes(page_json)

    log_path = target_folder / "correction-log.jsonl"
    log_path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            for record in log_records
        ),
        encoding="utf-8",
    )
    output_page_hashes: dict[str, str] = {}
    for relative_markdown, _, _ in corrected_pages:
        markdown_target = (target_folder / relative_markdown).resolve()
        output_page_hashes[relative_markdown] = _sha256(markdown_target)
    result: dict[str, object] = {
        "schema_version": 1,
        "document_id": document_id,
        "status": "pass",
        "source_ocr_report_sha256": _sha256(report_path),
        "language_override": language,
        "ocr_language": ocr_report.get("language"),
        "confidence_threshold": confidence_threshold,
        "confidence_kind": "heuristic-not-probability",
        "decisions": {
            "AUTO_CORRECT": automatic_corrections,
            "KEEP_ORIGINAL": kept_decisions,
        },
        "pages": page_records,
        "output_page_sha256": output_page_hashes,
        "correction_log_sha256": _sha256(log_path),
        "warnings": sorted(set(warnings)),
        "evidence_availability": {
            "semantic_context_model": False,
            "context_phrases": any(context_phrases.values()),
            "ocr_confidence": False,
            "visual_evidence": False,
        },
        "log_file": "correction-log.jsonl",
    }
    (target_folder / "spellcheck-report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (target_folder / "source-pages.json").write_text(
        json.dumps(source_page_hashes, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if dictionary_file is not None:
        dictionary_target = target_folder / "dictionary-used.json"
        if Path(dictionary_file).resolve() != dictionary_target.resolve():
            shutil.copy2(dictionary_file, dictionary_target)
    return result
