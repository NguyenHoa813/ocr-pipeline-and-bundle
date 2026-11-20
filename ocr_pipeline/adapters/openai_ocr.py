import base64
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


class OpenAIOCRAdapter:
    """Transcribe a page using the OpenAI Responses API."""

    model_name = "OpenAI API OCR"

    def __init__(self, language: str = "vi", model: str = "gpt-4.1-mini") -> None:
        if language not in SUPPORTED_LANGUAGES:
            raise Step4Error(f"Unsupported OpenAI OCR language: {language}")
        if not model.strip():
            raise Step4Error("An OpenAI model name is required.")
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key or not api_key.strip():
            raise Step4Error("Set OPENAI_API_KEY in the environment.")
        try:
            openai = importlib.import_module("openai")
        except ModuleNotFoundError as exc:
            raise Step4Error(
                "OpenAI OCR requires the optional dependency: "
                "python -m pip install -e '.[openai]'"
            ) from exc
        self._api_error = openai.APIError
        self._client = openai.OpenAI(api_key=api_key)
        self.model_version = model
        self.language = language
        self.configuration = {"openai_model": model}

    def recognize(self, image_path: Path) -> tuple[PageOCRContent, object]:
        image_data = base64.b64encode(page_png(image_path, "OpenAI")).decode("ascii")
        try:
            response = self._client.responses.create(
                model=self.model_version,
                input=[{
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": transcription_prompt(self.language)},
                        {
                            "type": "input_image",
                            "image_url": f"data:image/png;base64,{image_data}",
                            "detail": "high",
                        },
                    ],
                }],
                store=False,
            )
        except self._api_error as exc:
            code = getattr(exc, "status_code", None)
            suffix = f" (status {code})" if type(code) is int else ""
            raise Step4Error(f"OpenAI OCR request failed{suffix}: {type(exc).__name__}") from exc
        raw = response.model_dump(mode="json", exclude_none=True)
        if raw.get("status") != "completed":
            raise Step4Error("OpenAI OCR response was incomplete or failed.", raw)
        output = raw.get("output")
        if not isinstance(output, list):
            raise Step4Error("OpenAI OCR response contains no text.", raw)
        texts: list[str] = []
        for item in output:
            if not isinstance(item, dict) or item.get("type") != "message":
                continue
            parts = item.get("content")
            if not isinstance(parts, list):
                raise Step4Error("OpenAI OCR response has invalid message content.", raw)
            for part in parts:
                if not isinstance(part, dict):
                    raise Step4Error("OpenAI OCR response has invalid message content.", raw)
                if part.get("type") == "refusal":
                    raise Step4Error("OpenAI refused the OCR request.", raw)
                if part.get("type") == "output_text" and isinstance(part.get("text"), str):
                    texts.append(part["text"])
        text = "".join(texts).strip()
        if not text:
            raise Step4Error("OpenAI OCR response contains no text.", raw)
        return PageOCRContent(markdown=text, regions=[]), raw

    @staticmethod
    def token_usage(raw: object) -> dict[str, int | None] | None:
        if not isinstance(raw, dict) or not isinstance(raw.get("usage"), dict):
            return None
        usage = raw["usage"]
        input_details = usage.get("input_tokens_details")
        output_details = usage.get("output_tokens_details")

        return {
            "prompt_tokens": token_count(usage, "input_tokens"),
            "output_tokens": token_count(usage, "output_tokens"),
            "thoughts_tokens": token_count(output_details, "reasoning_tokens"),
            "cached_tokens": token_count(input_details, "cached_tokens"),
            "total_tokens": token_count(usage, "total_tokens"),
        }
