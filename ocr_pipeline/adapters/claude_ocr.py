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


class ClaudeOCRAdapter:
    """Transcribe a page using Anthropic's Messages API."""

    model_name = "Claude API OCR"

    def __init__(
        self, language: str = "vi", model: str = "claude-sonnet-4-5"
    ) -> None:
        if language not in SUPPORTED_LANGUAGES:
            raise Step4Error(f"Unsupported Claude OCR language: {language}")
        if not model.strip():
            raise Step4Error("A Claude model name is required.")
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key or not api_key.strip():
            raise Step4Error("Set ANTHROPIC_API_KEY in the environment.")
        try:
            anthropic = importlib.import_module("anthropic")
        except ModuleNotFoundError as exc:
            raise Step4Error(
                "Claude OCR requires the optional dependency: "
                "python -m pip install -e '.[claude]'"
            ) from exc
        self._api_error = anthropic.APIError
        self._client = anthropic.Anthropic(api_key=api_key)
        self.model_version = model
        self.language = language
        self.configuration = {"claude_model": model}

    def recognize(self, image_path: Path) -> tuple[PageOCRContent, object]:
        image_data = base64.b64encode(page_png(image_path, "Claude")).decode("ascii")
        try:
            response = self._client.messages.create(
                model=self.model_version,
                max_tokens=8192,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": image_data,
                            },
                        },
                        {"type": "text", "text": transcription_prompt(self.language)},
                    ],
                }],
            )
        except self._api_error as exc:
            code = getattr(exc, "status_code", None)
            suffix = f" (status {code})" if type(code) is int else ""
            raise Step4Error(f"Claude OCR request failed{suffix}: {type(exc).__name__}") from exc
        raw = response.model_dump(mode="json", exclude_none=True)
        if raw.get("stop_reason") != "end_turn":
            raise Step4Error("Claude OCR response was incomplete or refused.", raw)
        blocks = raw.get("content")
        if not isinstance(blocks, list):
            raise Step4Error("Claude OCR response contains no text.", raw)
        text = "".join(
            block["text"] for block in blocks
            if isinstance(block, dict) and block.get("type") == "text"
            and isinstance(block.get("text"), str)
        ).strip()
        if not text:
            raise Step4Error("Claude OCR response contains no text.", raw)
        return PageOCRContent(markdown=text, regions=[]), raw

    @staticmethod
    def token_usage(raw: object) -> dict[str, int | None] | None:
        if not isinstance(raw, dict) or not isinstance(raw.get("usage"), dict):
            return None
        usage = raw["usage"]

        input_tokens = token_count(usage, "input_tokens")
        cache_creation = token_count(usage, "cache_creation_input_tokens")
        cache_read = token_count(usage, "cache_read_input_tokens")
        output_tokens = token_count(usage, "output_tokens")
        input_parts = (input_tokens, cache_creation, cache_read)
        prompt_tokens = (
            sum(part for part in input_parts if part is not None)
            if all(part is not None for part in input_parts)
            else None
        )
        return {
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "thoughts_tokens": token_count(
                usage.get("output_tokens_details"), "thinking_tokens"
            ),
            "cached_tokens": cache_read,
            "total_tokens": (
                prompt_tokens + output_tokens
                if prompt_tokens is not None and output_tokens is not None
                else None
            ),
        }
