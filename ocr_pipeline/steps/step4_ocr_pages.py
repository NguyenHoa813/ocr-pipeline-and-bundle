import json
import logging
import math
import re
import time
from dataclasses import asdict, dataclass
from decimal import Decimal
from importlib.metadata import entry_points
from pathlib import Path
from typing import Callable, Literal, Protocol, runtime_checkable

from ocr_pipeline.steps.step1_check_batch import resolve_scan_folder


OCR_ADAPTER_ENTRY_POINT_GROUP = "ocr_pipeline.ocr_adapters"
_LOGGER = logging.getLogger(__name__)


class Step4Error(Exception):
    """Raised when OCR inputs or model outputs cannot be processed."""

    def __init__(self, message: str, raw_response: object | None = None) -> None:
        super().__init__(message)
        self.raw_response = raw_response


RegionType = Literal[
    "text",
    "table",
    "formula",
    "figure",
    "stamp",
    "page_header",
    "page_footer",
    "page_number",
    "heading",
    "image",
    "signature",
    "handwriting",
]
ImageKind = Literal["drawing", "photo"]

PAGE_FILENAME_PATTERN = re.compile(r"-(?P<page>\d{5})\.tif$", re.IGNORECASE)


@dataclass(frozen=True)
class LayoutRegion:
    type: RegionType
    text: str
    bbox: list[float] | None = None
    image_kind: ImageKind | None = None


@dataclass(frozen=True)
class PageOCRContent:
    markdown: str
    regions: list[LayoutRegion]


class OCRModelAdapter(Protocol):
    model_name: str
    model_version: str

    def recognize(self, image_path: Path) -> tuple[PageOCRContent, object]:
        """Return normalized OCR content and the unmodified model response."""
        ...


@runtime_checkable
class TokenUsageAdapter(Protocol):
    def token_usage(self, raw: object) -> dict[str, int | None] | None:
        ...


@dataclass(frozen=True)
class ReportedOCRCost:
    amount: float
    currency: str


@runtime_checkable
class CostUsageAdapter(Protocol):
    def cost_usage(self, raw: object) -> ReportedOCRCost | None:
        """Return provider-reported cost, or None when unavailable."""
        ...


class _InvalidCostError(Step4Error):
    pass


def _read_reported_cost(
    reader: Callable[[object], ReportedOCRCost | None] | None,
    raw: object,
) -> ReportedOCRCost | None:
    if reader is None:
        return None
    cost = reader(raw)
    if cost is None:
        return None
    if (
        not isinstance(cost, ReportedOCRCost)
        or type(cost.amount) not in (int, float)
        or not math.isfinite(cost.amount)
        or cost.amount < 0
        or not isinstance(cost.currency, str)
        or re.fullmatch(r"[A-Z]{3}", cost.currency) is None
    ):
        raise _InvalidCostError("OCR adapter returned invalid reported cost metadata.", raw)
    return cost


def load_ocr_adapter(model_name: str, **options: object) -> OCRModelAdapter:
    """Load a built-in or installed third-party OCR adapter plugin."""
    builtin_adapters: dict[str, tuple[str, str]] = {
        "tesseract": (
            "ocr_pipeline.adapters.tesseract_ocr",
            "TesseractOCRAdapter",
        ),
        "vietocr": (
            "ocr_pipeline.adapters.vietocr_layout",
            "VietOCRLayoutAdapter",
        ),
        "gemini": (
            "ocr_pipeline.adapters.gemini_ocr",
            "GeminiOCRAdapter",
        ),
        "openai": (
            "ocr_pipeline.adapters.openai_ocr",
            "OpenAIOCRAdapter",
        ),
        "claude": (
            "ocr_pipeline.adapters.claude_ocr",
            "ClaudeOCRAdapter",
        ),
    }
    if model_name in builtin_adapters:
        module_name, class_name = builtin_adapters[model_name]
        try:
            adapter_type = getattr(
                __import__(module_name, fromlist=[class_name]),
                class_name,
            )
        except Exception as exc:
            raise Step4Error(
                f"Cannot load built-in OCR adapter {model_name!r}: {exc}"
            ) from exc
    else:
        matches = [
            plugin
            for plugin in entry_points(group=OCR_ADAPTER_ENTRY_POINT_GROUP)
            if plugin.name == model_name
        ]
        if len(matches) != 1:
            available = sorted(
                set(builtin_adapters)
                | {
                    plugin.name
                    for plugin in entry_points(
                        group=OCR_ADAPTER_ENTRY_POINT_GROUP
                    )
                }
            )
            if len(matches) > 1:
                raise Step4Error(
                    f"Multiple OCR plugins are registered as {model_name!r}."
                )
            raise Step4Error(
                f"Unknown OCR model plugin {model_name!r}. "
                f"Available plugins: {', '.join(available)}."
            )
        try:
            adapter_type = matches[0].load()
        except Exception as exc:
            raise Step4Error(
                f"Cannot load OCR model plugin {model_name!r}: {exc}"
            ) from exc

    try:
        adapter = adapter_type(**options)
    except Step4Error:
        raise
    except Exception as exc:
        raise Step4Error(
            f"Cannot initialize OCR model plugin {model_name!r}: {exc}"
        ) from exc
    if not isinstance(adapter.model_name, str) or not isinstance(
        adapter.model_version, str
    ):
        raise Step4Error(
            f"OCR model plugin {model_name!r} has invalid model metadata."
        )
    if not callable(adapter.recognize):
        raise Step4Error(
            f"OCR model plugin {model_name!r} has no recognize() method."
        )
    return adapter


@dataclass(frozen=True)
class PageOCRResult:
    page_number: int
    file: str
    status: Literal["success", "blank", "failed"]
    markdown_file: str
    raw_output_file: str | None
    error: str | None = None
    duration_seconds: float | None = None
    token_usage: dict[str, int | None] | None = None
    reported_cost: ReportedOCRCost | None = None


@dataclass
class DocumentOCRReport:
    schema_version: int
    document_id: str
    model: str
    model_version: str
    status: Literal["pass", "fail"]
    pages: list[PageOCRResult]
    language: str | None = None
    token_usage: dict[str, object] | None = None
    cost: dict[str, object] | None = None


def _load_document_pages(
    split_report_path: Path,
    document_id: str,
) -> tuple[str, list[int]]:
    try:
        with split_report_path.open("r", encoding="utf-8") as file:
            report = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        raise Step4Error(
            f"Cannot read Step 2 split report {split_report_path}: {exc}"
        ) from exc
    if not isinstance(report, dict) or report.get("schema_version") != 1:
        raise Step4Error("Unsupported or invalid Step 2 split report.")
    units = report.get("units")
    if not isinstance(units, list):
        raise Step4Error("Step 2 split report has no units list.")
    for unit in units:
        if not isinstance(unit, dict) or unit.get("status") != "pass":
            continue
        unit_id = unit.get("unit_id")
        documents = unit.get("documents")
        if not isinstance(unit_id, str) or not isinstance(documents, list):
            raise Step4Error("Step 2 split report contains invalid unit data.")
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
                raise Step4Error(
                    f"Invalid or empty page list for document {document_id}."
                )
            return unit_id, pages
    raise Step4Error(
        f"Document {document_id} was not found in a passing Step 2 report."
    )


def _load_scan_paths(
    check_report_path: Path,
    unit_id: str,
) -> dict[int, str]:
    try:
        with check_report_path.open("r", encoding="utf-8") as file:
            report = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        raise Step4Error(
            f"Cannot read Step 1 check report {check_report_path}: {exc}"
        ) from exc
    if not isinstance(report, dict) or report.get("schema_version") != 1:
        raise Step4Error("Unsupported or invalid Step 1 check report.")
    units = report.get("units")
    if not isinstance(units, list):
        raise Step4Error("Step 1 check report has no units list.")
    for unit in units:
        if not isinstance(unit, dict) or unit.get("document_unit_id") != unit_id:
            continue
        if unit.get("status") != "pass":
            raise Step4Error(f"Unit {unit_id} did not pass Step 1.")
        files = unit.get("files")
        if not isinstance(files, list):
            raise Step4Error(f"Step 1 report has no scan files for {unit_id}.")
        scan_paths: dict[int, str] = {}
        for relative_file in files:
            if not isinstance(relative_file, str):
                raise Step4Error(f"Invalid scan path for unit {unit_id}.")
            page_match = PAGE_FILENAME_PATTERN.search(relative_file)
            if page_match is None:
                raise Step4Error(
                    f"Invalid scan filename in Step 1 report: {relative_file}"
                )
            page_number = int(page_match.group("page"))
            if page_number in scan_paths:
                raise Step4Error(
                    f"Duplicate page {page_number:05d} in Step 1 report."
                )
            scan_paths[page_number] = relative_file
        return scan_paths
    raise Step4Error(f"Unit {unit_id} was not found in the Step 1 report.")


def _load_blank_flags(
    blank_report_path: Path,
    document_id: str,
    pages: list[int],
) -> set[int]:
    try:
        with blank_report_path.open("r", encoding="utf-8") as file:
            report = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        raise Step4Error(
            f"Cannot read Step 3 blank-page report {blank_report_path}: {exc}"
        ) from exc
    if (
        not isinstance(report, dict)
        or report.get("schema_version") != 1
        or report.get("status") != "pass"
        or report.get("document_id") != document_id
    ):
        raise Step4Error(
            "Step 3 report is invalid, failed, or belongs to another document."
        )
    page_flags = report.get("pages")
    if not isinstance(page_flags, list):
        raise Step4Error("Step 3 report has no page flags.")
    flags_by_page: dict[int, bool] = {}
    for flag in page_flags:
        if not isinstance(flag, dict):
            raise Step4Error("Step 3 report contains an invalid page flag.")
        page_number = flag.get("page_number")
        is_blank = flag.get("is_blank")
        if type(page_number) is not int or type(is_blank) is not bool:
            raise Step4Error("Step 3 report contains invalid page flag values.")
        if page_number in flags_by_page:
            raise Step4Error(f"Duplicate blank flag for page {page_number:05d}.")
        flags_by_page[page_number] = is_blank
    if set(flags_by_page) != set(pages):
        raise Step4Error(
            f"Step 3 page flags do not match document pages for {document_id}."
        )
    return {page for page, is_blank in flags_by_page.items() if is_blank}


def run_document_ocr(
    batch_root: Path,
    document_id: str,
    adapter: OCRModelAdapter,
    split_report_path: Path | None = None,
    check_report_path: Path | None = None,
    blank_report_path: Path | None = None,
) -> tuple[DocumentOCRReport, dict[int, PageOCRContent], dict[int, object]]:
    """OCR each non-blank page and retain each raw model response."""
    if not re.fullmatch(r"[A-Za-z0-9]+-\d+-\d{2}", document_id):
        raise Step4Error(f"Invalid document ID: {document_id}")
    batch_root = Path(batch_root).resolve()
    scan_folder = resolve_scan_folder(batch_root)
    if not scan_folder.is_dir():
        raise Step4Error(f"Scan folder does not exist: {scan_folder}")
    work_folder = batch_root / "work"
    split_report_path = Path(
        split_report_path or work_folder / "step-2-documents" / "split-report.json"
    ).resolve()
    check_report_path = Path(
        check_report_path or work_folder / "step-1-checks" / "batch-report.json"
    ).resolve()
    blank_report_path = Path(
        blank_report_path
        or work_folder
        / "step-3-blank-pages"
        / f"{document_id}.json"
    ).resolve()

    unit_id, pages = _load_document_pages(split_report_path, document_id)
    scan_paths = _load_scan_paths(check_report_path, unit_id)
    blank_pages = _load_blank_flags(blank_report_path, document_id, pages)
    page_results: list[PageOCRResult] = []
    normalized_outputs: dict[int, PageOCRContent] = {}
    raw_outputs: dict[int, object] = {}
    usage_reader: Callable[[object], dict[str, int | None] | None] | None = (
        adapter.token_usage if isinstance(adapter, TokenUsageAdapter) else None
    )
    cost_reader: Callable[[object], ReportedOCRCost | None] | None = (
        adapter.cost_usage if isinstance(adapter, CostUsageAdapter) else None
    )
    for page_number in pages:
        page_started = time.perf_counter()
        relative_file = scan_paths.get(page_number)
        if relative_file is None:
            raise Step4Error(
                f"Page {page_number:05d} of {document_id} is missing from "
                "the Step 1 report."
            )
        image_path = (scan_folder / relative_file).resolve()
        if not image_path.is_relative_to(scan_folder):
            raise Step4Error(
                f"Scan path in Step 1 report escapes raw folder: {relative_file}"
            )
        if not image_path.is_file():
            raise Step4Error(f"Scan file is missing: {image_path}")
        markdown_relative = f"pages/page-{page_number:05d}.md"
        if page_number in blank_pages:
            duration_seconds = round(time.perf_counter() - page_started, 3)
            normalized_outputs[page_number] = PageOCRContent(
                markdown="",
                regions=[],
            )
            page_results.append(
                PageOCRResult(
                    page_number=page_number,
                    file=relative_file,
                    status="blank",
                    markdown_file=markdown_relative,
                    raw_output_file=None,
                    duration_seconds=duration_seconds,
                )
            )
            _LOGGER.info(
                "OCR page %05d: blank (%.3f seconds)",
                page_number,
                duration_seconds,
            )
            continue
        try:
            content, raw_output = adapter.recognize(image_path)
            reported_cost = _read_reported_cost(cost_reader, raw_output)
            duration_seconds = round(time.perf_counter() - page_started, 3)
            normalized_outputs[page_number] = content
            raw_outputs[page_number] = raw_output
            page_results.append(
                PageOCRResult(
                    page_number=page_number,
                    file=relative_file,
                    status="success",
                    markdown_file=markdown_relative,
                    raw_output_file=f"raw/page-{page_number:05d}.json",
                    duration_seconds=duration_seconds,
                    token_usage=(
                        usage_reader(raw_output)
                        if usage_reader is not None
                        else None
                    ),
                    reported_cost=reported_cost,
                )
            )
            _LOGGER.info(
                "OCR page %05d: success (%.3f seconds)",
                page_number,
                duration_seconds,
            )
            _LOGGER.info(
                "OCR page %05d: token usage=%s; reported cost=%s",
                page_number, page_results[-1].token_usage, reported_cost,
            )
        except Step4Error as exc:
            duration_seconds = round(time.perf_counter() - page_started, 3)
            raw_response = getattr(exc, "raw_response", None)
            error_message = str(exc)
            failed_cost = None
            if raw_response is not None:
                raw_outputs[page_number] = raw_response
                if not isinstance(exc, _InvalidCostError):
                    try:
                        failed_cost = _read_reported_cost(cost_reader, raw_response)
                    except _InvalidCostError as cost_exc:
                        error_message = f"{error_message}; {cost_exc}"
            page_results.append(
                PageOCRResult(
                    page_number=page_number,
                    file=relative_file,
                    status="failed",
                    markdown_file=markdown_relative,
                    raw_output_file=(
                        f"raw/page-{page_number:05d}.json"
                        if raw_response is not None
                        else None
                    ),
                    error=error_message,
                    duration_seconds=duration_seconds,
                    token_usage=(
                        usage_reader(raw_response)
                        if usage_reader is not None and raw_response is not None
                        else None
                    ),
                    reported_cost=failed_cost,
                )
            )
            _LOGGER.info(
                "OCR page %05d: failed (%.3f seconds): %s",
                page_number,
                duration_seconds,
                error_message,
            )
            _LOGGER.info(
                "OCR page %05d: token usage=%s; reported cost=%s",
                page_number, page_results[-1].token_usage, page_results[-1].reported_cost,
            )

    overall_status: Literal["pass", "fail"] = (
        "pass"
        if all(page.status in ("success", "blank") for page in page_results)
        else "fail"
    )
    usage_summary: dict[str, object] | None = None
    if usage_reader is not None:
        attempted = [page for page in page_results if page.status != "blank"]
        reported = [page.token_usage for page in attempted if page.token_usage is not None]
        usage_summary = {
            "reported_pages": len(reported),
            "unreported_pages": len(attempted) - len(reported),
        }
        for field in ("prompt_tokens", "output_tokens", "thoughts_tokens",
                      "cached_tokens", "total_tokens"):
            values = [usage[field] for usage in reported]
            usage_summary[field] = (
                sum(value for value in values if value is not None)
                if attempted and len(reported) == len(attempted)
                and all(value is not None for value in values)
                else None
            )
    attempted = [page for page in page_results if page.status != "blank"]
    reported_costs = [page.reported_cost for page in attempted if page.reported_cost is not None]
    currencies = {cost.currency for cost in reported_costs}
    complete_cost = bool(attempted) and len(reported_costs) == len(attempted) and len(currencies) == 1
    # Never turn missing provider billing data or mixed currencies into a total.
    cost_summary: dict[str, object] = {
        "status": (
            "reported" if complete_cost
            else "mixed_currencies" if len(currencies) > 1
            else "partial" if reported_costs
            else "not_reported"
        ),
        "amount": (
            float(sum((Decimal(str(cost.amount)) for cost in reported_costs), Decimal(0)))
            if complete_cost else None
        ),
        "currency": next(iter(currencies)) if complete_cost else None,
        "reported_pages": len(reported_costs),
        "unreported_pages": len(attempted) - len(reported_costs),
    }
    return (
        DocumentOCRReport(
            schema_version=1,
            document_id=document_id,
            model=adapter.model_name,
            model_version=adapter.model_version,
            status=overall_status,
            pages=page_results,
            language=getattr(adapter, "language", None),
            token_usage=usage_summary,
            cost=cost_summary,
        ),
        normalized_outputs,
        raw_outputs,
    )


def write_document_ocr(
    report: DocumentOCRReport,
    normalized_outputs: dict[int, PageOCRContent],
    raw_outputs: dict[int, object],
    output_folder: Path,
) -> None:
    """Write normalized page outputs and raw responses locally."""
    output_folder = Path(output_folder)
    pages_folder = output_folder / "pages"
    raw_folder = output_folder / "raw"
    pages_folder.mkdir(parents=True, exist_ok=True)
    for page in report.pages:
        content = normalized_outputs.get(page.page_number)
        if content is None:
            markdown = ""
            regions: list[LayoutRegion] = []
        else:
            markdown = content.markdown
            regions = content.regions
        page_stem = f"page-{page.page_number:05d}"
        (pages_folder / f"{page_stem}.md").write_text(
            markdown,
            encoding="utf-8",
        )
        (pages_folder / f"{page_stem}.json").write_text(
            json.dumps(
                {
                    "page_number": page.page_number,
                    "status": page.status,
                    "markdown_file": page.markdown_file,
                    "regions": [asdict(region) for region in regions],
                    "error": page.error,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    if raw_outputs:
        raw_folder.mkdir(parents=True, exist_ok=True)
        for page_number, raw_output in raw_outputs.items():
            (raw_folder / f"page-{page_number:05d}.json").write_text(
                json.dumps(raw_output, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
    (output_folder / "ocr-report.json").write_text(
        json.dumps(asdict(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
