import importlib
import os
from pathlib import Path

from ocr_pipeline.adapters.external_ocr import (
    SUPPORTED_LANGUAGES,
    page_png,
    token_count,
    transcription_prompt,
)
from ocr_pipeline.steps.step4_ocr_pages import PageOCRContent, Step4Error


class GeminiOCRAdapter:
    """Transcribe a page with the Gemini API; layout coordinates are not inferred."""

    model_name = "Gemini API OCR"

    def __init__(
        self,
        language: str = "vi",
        model: str = "gemini-4-hydrogen",
        thinking_level: str = "low",
    ) -> None:
        if language not in SUPPORTED_LANGUAGES:
            raise Step4Error(f"Unsupported Gemini OCR language: {language}")
        if not model.strip():
            raise Step4Error("A Gemini model name is required.")
        if thinking_level not in {"low", "medium", "high", "xhigh"}:
            raise Step4Error(f"Unsupported Gemini thinking level: {thinking_level}")
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key or not api_key.strip():
            raise Step4Error("Set GEMINI_API_KEY or GOOGLE_API_KEY in the environment.")
        try:
            genai = importlib.import_module("google.genai")
            errors = importlib.import_module("google.genai.errors")
            self._types = importlib.import_module("google.genai.types")
        except ModuleNotFoundError as exc:
            raise Step4Error(
                "Gemini OCR requires the optional dependency: "
                "python -m pip install -e '.[gemini]'"
            ) from exc
        request_errors: list[type[Exception]] = [errors.APIError]
        for module_name in ("httpx", "httpx2"):
            try:
                transport = importlib.import_module(module_name)
            except ModuleNotFoundError:
                continue
            request_errors.append(transport.RequestError)
        self._request_errors = tuple(request_errors)
        self._client = genai.Client(api_key=api_key)
        self.model_version = model
        self.language = language
        self.thinking_level = thinking_level
        self.configuration = {
            "gemini_model": model,
            "gemini_thinking_level": thinking_level,
        }

    def recognize(self, image_path: Path) -> tuple[PageOCRContent, object]:
        image_bytes = page_png(image_path, "Gemini")
        prompt = transcription_prompt(self.language)
        try:
            response = self._client.models.generate_content(
                model=self.model_version,
                contents=[
                    prompt,
                    self._types.Part.from_bytes(
                        data=image_bytes, mime_type="image/png"
                    ),
                ],
                config=self._types.GenerateContentConfig(
                    temperature=0,
                    thinking_config=self._types.ThinkingConfig(
                        thinking_level=self.thinking_level
                    ),
                ),
            )
        except self._request_errors as exc:
            code = getattr(exc, "code", None)
            suffix = f" (status {code})" if type(code) is int else ""
            raise Step4Error(f"Gemini OCR request failed{suffix}: {type(exc).__name__}") from exc
        raw = response.model_dump(mode="json", exclude_none=True)
        candidates = raw.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise Step4Error("Gemini returned no OCR candidate.", raw)
        candidate = candidates[0]
        if not isinstance(candidate, dict) or candidate.get("finish_reason") != "STOP":
            raise Step4Error("Gemini OCR response was blocked or incomplete.", raw)
        content = candidate.get("content")
        parts = content.get("parts") if isinstance(content, dict) else None
        if not isinstance(parts, list):
            raise Step4Error("Gemini OCR response contains no text.", raw)
        text = "".join(
            part["text"] for part in parts
            if isinstance(part, dict) and isinstance(part.get("text"), str)
            and not part.get("thought", False)
        ).strip()
        if not text:
            raise Step4Error("Gemini OCR response contains no text.", raw)
        return PageOCRContent(markdown=text, regions=[]), raw

    @staticmethod
    def token_usage(raw: object) -> dict[str, int | None] | None:
        if not isinstance(raw, dict) or not isinstance(raw.get("usage_metadata"), dict):
            return None
        usage = raw["usage_metadata"]
        fields = {
            "prompt_tokens": "prompt_token_count",
            "output_tokens": "candidates_token_count",
            "thoughts_tokens": "thoughts_token_count",
            "cached_tokens": "cached_content_token_count",
            "total_tokens": "total_token_count",
        }
        return {name: token_count(usage, key) for name, key in fields.items()}
