import io
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from ocr_pipeline.steps.step4_ocr_pages import Step4Error


SUPPORTED_LANGUAGES = frozenset({"vi", "en", "fr", "zh-Hant", "vi+zh-Hant"})


def token_count(data: object, key: str) -> int | None:
    if not isinstance(data, dict):
        return None
    value = data.get(key)
    return value if type(value) is int and value >= 0 else None


def page_png(image_path: Path, provider: str) -> bytes:
    try:
        with Image.open(image_path) as image:
            if getattr(image, "n_frames", 1) != 1:
                raise Step4Error(f"{provider} OCR requires a single-frame page image.")
            buffer = io.BytesIO()
            image.convert("RGB").save(buffer, format="PNG")
            return buffer.getvalue()
    except (OSError, UnidentifiedImageError) as exc:
        raise Step4Error(f"Cannot read OCR page image: {image_path}") from exc


def transcription_prompt(language: str) -> str:
    return (
        "Transcribe all visible text on this document page in reading order. "
        f"The expected language is {language}. Preserve the original script, "
        "Traditional Chinese characters, spelling, punctuation and line breaks. "
        "Do not translate, correct, guess missing text, add commentary or "
        "describe images. Return only the transcription."
    )
