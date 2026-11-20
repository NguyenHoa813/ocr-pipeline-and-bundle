import json
import importlib
import mimetypes
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol


class Step7Error(Exception):
    """Raised when cropped figures cannot be described."""


IMAGE_PATH_PATTERN = re.compile(
    r"^images/p\d{3,}-\d+\.(?:png|jpg)$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ImageDescription:
    file: str
    status: str
    alt_text: str
    mermaid: str | None
    error: str | None = None


@dataclass
class ImageDescriptionReport:
    schema_version: int
    document_id: str
    model: str
    status: str
    images: list[ImageDescription]


class ImageDescriptionAdapter(Protocol):
    model_name: str

    def describe(self, image_path: Path) -> tuple[str, str | None]:
        """Return concise alternative text and optional Mermaid source."""
        ...


class GeminiVertexAIAdapter:
    """Gemini image descriptions through Google Vertex AI using ADC."""

    def __init__(
        self,
        project: str,
        location: str,
        model: str = "gemini-4-hydrogen",
    ) -> None:
        if not project.strip() or not location.strip() or not model.strip():
            raise Step7Error("Vertex project, location, and model are required.")
        try:
            genai = importlib.import_module("google.genai")
            genai_errors = importlib.import_module("google.genai.errors")
            types = importlib.import_module("google.genai.types")
        except ModuleNotFoundError as exc:
            raise Step7Error(
                "Vertex image descriptions require the optional dependency. "
                "Install with 'python -m pip install -e .[vertex]'."
            ) from exc
        self.model_name = model
        self._api_error: type[Exception] = getattr(genai_errors, "APIError")
        self._types = types
        client_class = getattr(genai, "Client")
        self._client = client_class(
            vertexai=True,
            project=project,
            location=location,
        )

    def describe(self, image_path: Path) -> tuple[str, str | None]:
        mime_type, _ = mimetypes.guess_type(image_path.name)
        if mime_type not in {"image/png", "image/jpeg"}:
            raise Step7Error(f"Unsupported cropped image type: {image_path}")
        prompt = (
            "Describe only the visible cropped image. Do not infer facts not "
            "shown. Return JSON with exactly alt_text (non-empty concise "
            "description in the image's language) and mermaid (string or null). "
            "Use Mermaid only when a clear flow or relationship diagram is "
            "visible; otherwise return null. Do not include Markdown fences."
        )
        try:
            response = self._client.models.generate_content(
                model=self.model_name,
                contents=[
                    prompt,
                    self._types.Part.from_bytes(
                        data=image_path.read_bytes(),
                        mime_type=mime_type,
                    ),
                ],
                config=self._types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0,
                ),
            )
        except (self._api_error, OSError) as exc:
            raise Step7Error(f"Vertex image description failed: {exc}") from exc
        if not isinstance(response.text, str):
            raise Step7Error("Vertex returned no text for the image description.")
        try:
            result = json.loads(response.text)
        except json.JSONDecodeError as exc:
            raise Step7Error("Vertex returned invalid description JSON.") from exc
        if not isinstance(result, dict):
            raise Step7Error("Vertex description must be a JSON object.")
        alt_text = result.get("alt_text")
        mermaid = result.get("mermaid")
        if (
            not isinstance(alt_text, str)
            or not alt_text.strip()
            or (mermaid is not None and not isinstance(mermaid, str))
        ):
            raise Step7Error("Vertex JSON must include non-empty alt_text and optional mermaid.")
        return alt_text.strip(), mermaid.strip() if mermaid else None


def describe_document_images(
    document_id: str,
    image_folder: Path,
    cut_out_report_path: Path,
    adapter: ImageDescriptionAdapter,
) -> ImageDescriptionReport:
    """Describe only Step 5 image files; TIFF originals are never inputs."""
    try:
        with Path(cut_out_report_path).open("r", encoding="utf-8") as file:
            report = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        raise Step7Error(f"Cannot read Step 5 report {cut_out_report_path}: {exc}") from exc
    if (
        not isinstance(report, dict)
        or report.get("schema_version") != 1
        or report.get("document_id") != document_id
        or report.get("status") != "pass"
        or not isinstance(report.get("images"), list)
    ):
        raise Step7Error("Step 5 report is invalid, failed, or for another document.")
    image_folder = Path(image_folder).resolve()
    descriptions: list[ImageDescription] = []
    seen: set[str] = set()
    for record in report["images"]:
        if not isinstance(record, dict):
            raise Step7Error("Step 5 report contains an invalid image record.")
        relative_file = record.get("file")
        if (
            not isinstance(relative_file, str)
            or not IMAGE_PATH_PATTERN.fullmatch(relative_file)
            or ".." in Path(relative_file).parts
            or relative_file in seen
        ):
            raise Step7Error("Step 5 report contains an invalid or duplicate crop path.")
        seen.add(relative_file)
        image_path = (image_folder / relative_file).resolve()
        if not image_path.is_relative_to(image_folder) or not image_path.is_file():
            raise Step7Error(f"Cropped image is missing or escapes output: {relative_file}")
        try:
            alt_text, mermaid = adapter.describe(image_path)
            if not alt_text.strip():
                raise Step7Error("Image adapter returned empty alt text.")
            if mermaid and not re.search(r"(?m)^\s*(?:flowchart|graph|sequenceDiagram|classDiagram|stateDiagram)", mermaid):
                raise Step7Error(f"Image adapter returned invalid Mermaid for {relative_file}.")
            descriptions.append(
                ImageDescription(
                    file=relative_file,
                    status="success",
                    alt_text=alt_text.strip(),
                    mermaid=mermaid.strip() if mermaid else None,
                )
            )
        except Step7Error as exc:
            descriptions.append(
                ImageDescription(
                    file=relative_file,
                    status="failed",
                    alt_text="",
                    mermaid=None,
                    error=str(exc),
                )
            )
    return ImageDescriptionReport(
        schema_version=1,
        document_id=document_id,
        model=adapter.model_name,
        status="pass" if all(item.status == "success" for item in descriptions) else "fail",
        images=descriptions,
    )


def write_image_description_report(
    report: ImageDescriptionReport,
    output_path: Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(asdict(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
